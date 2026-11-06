"""Build paired source packets; final explanation quality is evaluated separately.

The reference supplies questions and ranges, never the expected answer. Both
conditions receive identical speech and the same picture budget. Full packets
must stay in the private experiment home; public output contains only metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from vidlens.learning import LearningReader
from vidlens.service import VideoService
from vidlens.visual_relevance import QueryVisualSelector


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "heldout", "all"), default="heldout")
    parser.add_argument("--case")
    parser.add_argument("--images", type=int, choices=range(1, 13), default=4)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.packets.resolve().is_relative_to(args.home.resolve()):
        parser.error("Source packets must remain inside the private home directory")
    document = json.loads(args.questions.read_text())
    service = VideoService(args.home)
    results = []
    for case in document["cases"]:
        if args.case and case["id"] != args.case:
            continue
        if args.split != "all" and case["split"].replace("_", "") != args.split:
            continue
        video = service.store.get(case["video_id"])
        start, end = case["range"]
        _, _, scan = service.learning._scan_data(video.id)
        if scan is None or scan.get("status") != "complete":
            asset = service._asset(video, start, end)
            scan = service.visual.scan(asset, start=start, end=end, limit=1000)
            raw_candidates = list(scan["candidates"])
            cursor = scan.get("next_cursor")
            while cursor is not None and scan.get("status") == "complete":
                page = service.visual.scan(asset, start=start, end=end, cursor=cursor, limit=1000)
                if page.get("scan_id") != scan.get("scan_id") or page.get("status") != "complete":
                    parser.error("The benchmark interval scan changed or failed")
                raw_candidates.extend(page["candidates"])
                cursor = page.get("next_cursor")
            if scan.get("status") != "complete":
                parser.error("The benchmark requires a complete scan of its own interval")
            scan = {**scan, "candidates": raw_candidates}
        raw = [row for row in scan["candidates"] if start <= row["timestamp"] < end]
        baseline = LearningReader._overview_candidates({"start": start, "end": end}, raw, args.images)
        before = service.read_frame_times(video.id, [row["timestamp"] for row in baseline], record_delivery=False)
        speech = [asdict(s) for s in service.store.segments(video.id)
                  if s.kind != "ocr" and s.start < end and s.end > start]
        started = time.perf_counter()
        probes = []
        select = QueryVisualSelector.select

        def traced_select(question, frames, *arguments, _probes=probes, _select=select, **options):
            _probes[:] = frames
            return _select(question, frames, *arguments, **options)

        QueryVisualSelector.select = staticmethod(traced_select)
        try:
            focused = service.read_frames(video.id, start, end, question=case["question"],
                                          visual_queries=case.get("visual_queries"),
                                          max_frames=args.images, preview=args.preview, record_delivery=False)
        finally:
            QueryVisualSelector.select = staticmethod(select)
        packet = {"case": case, "url": video.url, "speech": speech,
                  "baseline_frames": before, "question_frames": focused["frames"],
                  "selection": focused["visual"], "probe_frames": probes,
                  "previews": focused.get("previews", []),
                  "preview_candidates": focused.get("candidates", [])}
        write(args.packets / f"{case['id']}.json", packet)
        results.append({
            "id": case["id"], "split": case["split"], "url": video.url,
            "range": [start, end], "image_budget": args.images,
            "speech_sha256": hashlib.sha256(json.dumps(speech, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "baseline_times": [row["timestamp"] for row in before],
            "question_times": [row["timestamp"] for row in focused["frames"]],
            "selection_seconds": round(time.perf_counter() - started, 3),
            "selection": focused["visual"]["selection"],
            "probe_count": focused["visual"]["probe_count"],
            "refinement_count": focused["visual"]["refinement_count"],
            "screen_text_frames": focused["visual"]["screen_text_frames"],
            "preview_image_count": len(focused.get("previews", [])),
            "preview_candidate_count": len(focused.get("candidates", [])),
        })
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    if not results:
        parser.error("No cases matched")
    write(args.output, {"selector_version": QueryVisualSelector.VERSION,
                        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                        "interpretation": "Image selection packets; not a final-answer quality score.",
                        "cases": results})


if __name__ == "__main__":
    main()

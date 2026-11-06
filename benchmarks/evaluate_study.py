"""Measure complete chapter traversal. Visual mode explicitly enables media acquisition.

This measures evidence delivery and pagination, not the quality of AI-written answers.
Source text and machine-specific paths are stored only in --artifacts, inside --home.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from vidlens import __version__
from vidlens.service import VideoService


def evaluate(service, video_id, *, visual=False, artifacts=None):
    started = time.perf_counter()
    plan = service.learning.study.plan(video_id)
    video = service.store.get(video_id)
    speech = {s.id: s for s in service.store.segments(video_id) if s.kind != "ocr"}
    chunks = {identity: {} for identity in speech}
    pages, frames, issues, section_results = 0, 0, [], []
    for index, section in enumerate(plan["sections"]):
        cursor = None
        section_pages, section_frames = 0, 0
        while True:
            page = service.read_section(video_id, section["id"], max_tokens=8000,
                                        cursor=cursor, include_frames=visual and cursor is None)
            if artifacts:
                (artifacts / f"section-{index + 1:02d}-page-{section_pages + 1:02d}.json").write_text(
                    json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8",
                )
            for evidence in page["evidence"]:
                text = evidence["text"]
                if evidence.get("excerpt_truncated"):
                    text = text[:-1]
                chunks[evidence["id"]][evidence.get("text_offset", 0)] = text
            section_frames += len(page["frames"])
            if page["visual"].get("status") == "unavailable":
                issues.append({"section": index + 1, "error": page["visual"].get("error")})
            notices = sorted({notice for frame in page["frames"]
                              for notice in frame.get("warnings", [])})
            if notices:
                issues.append({"section": index + 1, "warnings": notices})
            section_pages += 1
            next_cursor = page["next_cursor"]
            if next_cursor is None:
                break
            if next_cursor == cursor:
                raise RuntimeError("Chapter cursor did not advance")
            cursor = next_cursor
        section_results.append({"index": index + 1, "start": section["start"],
                                "end": section["end"], "pages": section_pages,
                                "images_returned": section_frames})
        pages += section_pages
        frames += section_frames
    exact = sum("".join(part for _, part in sorted(chunks[identity].items())) == segment.text
                for identity, segment in speech.items())
    return {"video_id": video_id, "url": video.url, "title": video.title, "duration": video.duration,
            "speech_scope": service.reader._speech_scope(video, list(speech.values())),
            "source_sha256": hashlib.sha256("\n".join(s.text for s in speech.values()).encode()).hexdigest(),
            "indexed_speech_segments": len(speech), "exactly_reconstructed_segments": exact,
            "lossless_text_read": exact == len(speech) and bool(speech),
            "chapter_count": len(plan["sections"]), "text_pages": pages,
            "images_returned": frames, "visual_enabled": visual, "visual_issues": issues,
            "elapsed_seconds": round(time.perf_counter() - started, 3), "chapters": section_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(".vidlens"))
    parser.add_argument("--video-id", action="append", required=True)
    parser.add_argument("--visual", action="store_true", help="Allow first-time media downloads and OCR")
    parser.add_argument("--artifacts", type=Path, help="Save private reading payloads inside --home")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    home = args.home.resolve()
    if args.artifacts and not args.artifacts.resolve().is_relative_to(home):
        parser.error("--artifacts must be inside --home")
    service = VideoService(home)
    results = []
    for video_id in args.video_id:
        folder = None
        if args.artifacts:
            folder = args.artifacts / video_id.replace(":", "-")
            folder.mkdir(parents=True, exist_ok=True)
        result = evaluate(service, video_id, visual=args.visual, artifacts=folder)
        results.append(result)
        print(json.dumps({"video_id": video_id, "chapters": result["chapter_count"],
                          "lossless_text_read": result["lossless_text_read"],
                          "images_returned": result["images_returned"]}), flush=True)
    report = {"schema_version": 1, "version": __version__, "measured_at_unix": int(time.time()),
              "scope": "Cached transcript traversal, exact cue reconstruction, and optional visual delivery.",
              "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Compare visual navigation and an experimental four-image ranking on cached videos.

Candidate counts are not semantic quality scores. With --artifacts, save both
image sets inside --home for independent pixel inspection at the same budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vidlens.learning import LearningReader
from vidlens.reading import validate_interval
from vidlens.selection import VisualSelector
from vidlens.service import VideoService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(".vidlens"))
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--images", type=int, default=4, choices=range(1, 13))
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.artifacts and not args.artifacts.resolve().is_relative_to(args.home.resolve()):
        parser.error("Private artifacts must stay inside --home")
    service = VideoService(args.home)
    video = service.store.get(args.video_id)
    start, end = validate_interval(video, args.start, video.duration if args.end is None else args.end)
    service.scan_video(video.id, view="full")
    _, _, scan = service.learning._scan_data(video.id)
    _, metadata, index = service.learning._candidate_view(video.id, scan, "navigation")
    if index is None:
        parser.error(f"Visual navigation unavailable: {metadata.get('selection_error')}")
    raw = [row for row in scan["candidates"] if start <= row["timestamp"] < end]
    if not raw:
        parser.error("No raw candidates in the requested interval")
    before = LearningReader._overview_candidates({"start": start, "end": end}, raw, args.images)
    after = [VisualSelector.public(row)
             for row in VisualSelector.representatives(index, start, end, args.images)]
    report = {
        "video_id": video.id, "url": video.url, "range": [start, end],
        "image_budget": args.images, "selector_version": VisualSelector.VERSION,
        "metrics": index["metrics"], "selector_cached": index["cached"],
        "before_timestamps": [row["timestamp"] for row in before],
        "experimental_ranked_timestamps": [row["timestamp"] for row in after],
        "production_default": "stable timeline overview; compact navigation for expansion",
        "interpretation": "Ranking is an experiment, not a semantic improvement claim.",
    }
    if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=True)
        packet = {**report}
        for label, rows in [("before", before), ("experimental_ranked", after)]:
            packet[f"{label}_frames"] = service.read_frame_times(
                video.id, [row["timestamp"] for row in rows], record_delivery=False
            ) if rows else []
        (args.artifacts / "comparison.json").write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"video_id": video.id, "metrics": index["metrics"]}))


if __name__ == "__main__":
    main()

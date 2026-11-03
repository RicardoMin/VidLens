"""Reproducible cached-source evidence retrieval measurements; no model calls.

Example:
  .venv/bin/python benchmarks/evaluate_reading.py --home .vidlens \
      --output benchmarks/results/reading.json

The source SQLite database is backed up into a temporary isolated cache. Both
"fresh process" and "warm process" runs use already acquired source material;
neither measures a cold URL fetch. Output token counts include the entire JSON
tool payload, while the explicit baseline is the original transcript text alone.
Exports replace evidence text with hashes and character counts by default. Debug
text export requires --include-evidence and an --output path inside --home.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class OfflineOnlyProvider:
    def acquire(self, *args, **kwargs):
        raise RuntimeError(
            "Benchmark cache miss: acquisition is disabled; seed the source cache first."
        )


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def service_for(home: Path):
    from vidlens.service import VideoService

    service = VideoService(home)
    if not hasattr(service, "read_video"):
        raise RuntimeError("VideoService.read_video is not implemented in this checkout yet.")
    service._provider = OfflineOnlyProvider()
    return service


def worker(args) -> None:
    # Timed separately from the parent-observed subprocess wall clock.
    started = time.perf_counter()
    service = service_for(Path(args.home))
    initialized = time.perf_counter()
    payload = service.read_video(args.url, question=args.question, max_tokens=args.max_tokens)
    finished = time.perf_counter()
    print(
        json_text(
            {
                "initialization_seconds": initialized - started,
                "call_seconds": finished - initialized,
                "initialization_and_call_seconds": finished - started,
                "payload": payload,
            }
        )
    )


def copy_cached_source(source: Path, destination: Path) -> None:
    database = source / "evidence.sqlite3"
    if not database.is_file():
        raise RuntimeError(f"No cached evidence database at {database}")
    destination.mkdir(parents=True, exist_ok=True)
    # SQLite's backup API includes committed WAL contents without mutating input.
    with (
        closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as original,
        closing(sqlite3.connect(destination / "evidence.sqlite3")) as duplicate,
    ):
        original.backup(duplicate)
    tokenizer = source / "tokenizer"
    if tokenizer.is_dir():
        shutil.copytree(tokenizer, destination / "tokenizer", dirs_exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(destination / "tokenizer")


def evidence_spans(payload: dict) -> list[dict]:
    """Read actual evidence text, excluding navigation and enclosing query bounds."""
    result: list[dict] = []
    for key in ("evidence", "samples"):
        for item in payload.get(key, []):
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            start, end = item.get("start"), item.get("end")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
                result.append(item)
    return result


def export_payload(payload: dict, include_evidence: bool = False) -> dict:
    """Redact source wording only after scoring and counting the complete payload."""
    if include_evidence:
        return payload
    exported = dict(payload)
    for key in ("evidence", "samples"):
        if key not in payload:
            continue
        exported[key] = []
        for item in payload[key]:
            metadata = {
                name: item[name]
                for name in ("start", "end", "source", "excerpt_truncated")
                if name in item
            }
            text = item.get("text", "")
            metadata.update(
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                text_chars=len(text),
            )
            exported[key].append(metadata)
    return exported


def covered_seconds(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    clipped = sorted((max(a, start), min(b, end)) for a, b in intervals if a < end and b > start)
    total = 0.0
    last_end = start
    for left, right in clipped:
        left = max(left, last_end)
        if right > left:
            total += right - left
            last_end = right
    return total


def score_windows(payload: dict, question: dict, minimum_coverage: float) -> dict:
    spans = evidence_spans(payload)
    intervals = [(item["start"], item["end"]) for item in spans]
    scored = []
    for target in question["required_windows"]:
        fraction = covered_seconds(intervals, target["start"], target["end"]) / (
            target["end"] - target["start"]
        )
        overlapping = [
            item
            for item in spans
            if item["start"] < target["end"] and item["end"] > target["start"]
        ]
        text = "\n".join(item["text"] for item in overlapping).casefold()
        anchors = target.get("anchors_any", [])
        scored.append(
            {
                "label": target["label"],
                "start": target["start"],
                "end": target["end"],
                "temporal_coverage_fraction": round(fraction, 6),
                "window_hit": fraction >= minimum_coverage,
                "any_anchor_found": any(anchor.casefold() in text for anchor in anchors)
                if anchors
                else None,
                "overlapping_text_was_truncated": any(
                    item.get("excerpt_truncated", False) for item in overlapping
                ),
            }
        )
    hits = sum(item["window_hit"] for item in scored)
    return {
        "windows": scored,
        "window_hits": hits,
        "required_window_count": len(scored),
        "target_window_recall": hits / len(scored) if scored else None,
        "all_target_windows_hit": hits == len(scored),
        "evidence_passage_count": len(spans),
    }


def source_hashes() -> dict[str, str]:
    from vidlens import service

    directory = Path(service.__file__).parent
    return {
        filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for filename in ("service.py", "reading.py", "retrieval.py", "budget.py", "store.py")
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run(args) -> dict:
    fixture_path = Path(args.questions).resolve()
    fixture = json.loads(fixture_path.read_text())
    output = Path(args.output).resolve()
    source_home = Path(args.home).expanduser().resolve()
    if args.include_evidence and not output.is_relative_to(source_home):
        raise RuntimeError(
            "--include-evidence requires --output inside the private --home cache, "
            "for example --output .vidlens/benchmarks/reading-with-evidence.json."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    initial_hashes = source_hashes()
    measurements = []

    with tempfile.TemporaryDirectory(prefix="vidlens-reading-eval-") as work_directory:
        isolated_home = Path(work_directory) / "cache"
        copy_cached_source(source_home, isolated_home)
        service = service_for(isolated_home)
        video = service.store.get(fixture["video_id"])
        if video.url != fixture["source_url"]:
            raise RuntimeError("Fixture URL does not match the cached source video.")
        # Personal follow-ups may have indexed OCR/ASR after the baseline run.
        # Restrict only the disposable backup to the annotated source track.
        if track := fixture.get("caption_track"):
            with service.store.connect() as db:
                db.execute(
                    "DELETE FROM segments WHERE video_id=? AND NOT (kind='subtitle' AND source=? AND language=?)",
                    (video.id, track["source"], track["language"]),
                )
        segments = [item for item in service.store.segments(video.id) if item.kind != "ocr"]
        if not segments:
            raise RuntimeError(
                "No captions matching the annotated track. Open the video with the fixture language first."
            )
        transcript = "\n".join(item.text for item in segments)
        baseline = {
            "definition": "Original cached transcript text joined with newlines; no JSON metadata, prompts, or model output.",
            "text_tokens_estimate": service.budget.count(transcript),
            "token_count_method": service.budget.method,
            "segment_count": len(segments),
            "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
            "caption_track": fixture.get("caption_track"),
            "subtitle_sources": sorted({item.source for item in segments}),
            "subtitle_languages": sorted({item.language for item in segments}),
        }
        for budget in args.budgets:
            for question in fixture["questions"]:
                for repetition in range(args.cold_repeats):
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--home",
                        str(isolated_home),
                        "--url",
                        video.url,
                        "--question",
                        question["question"],
                        "--max-tokens",
                        str(budget),
                    ]
                    started = time.perf_counter()
                    result = subprocess.run(
                        command,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout,
                    )
                    wall = time.perf_counter() - started
                    observed = json.loads(result.stdout)
                    measurements.append(
                        measure(
                            service,
                            question,
                            observed["payload"],
                            baseline,
                            budget,
                            "fresh_process_cached_source",
                            repetition,
                            wall,
                            args.minimum_window_coverage,
                            {key: value for key, value in observed.items() if key != "payload"},
                        )
                    )
                # An unmeasured prime ensures the warm measurement has used this
                # question/entry point in this same Python service instance.
                service.read_video(video.url, question=question["question"], max_tokens=budget)
                for repetition in range(args.warm_repeats):
                    started = time.perf_counter()
                    payload = service.read_video(
                        video.url, question=question["question"], max_tokens=budget
                    )
                    wall = time.perf_counter() - started
                    measurements.append(
                        measure(
                            service,
                            question,
                            payload,
                            baseline,
                            budget,
                            "warm_process_cached_source",
                            repetition,
                            wall,
                            args.minimum_window_coverage,
                            {},
                        )
                    )

    final_hashes = source_hashes()
    for measurement in measurements:
        measurement["payload"] = export_payload(measurement["payload"], args.include_evidence)
    report = {
        "schema_version": 2,
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "video_id": fixture["video_id"],
        "source_url": fixture["source_url"],
        "title": fixture["title"],
        "questions_file": fixture_path.name,
        "questions_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "source_hashes": initial_hashes,
        "source_changed_during_run": initial_hashes != final_hashes,
        "baseline": baseline,
        "settings": {
            "budgets": args.budgets,
            "fresh_process_repeats": args.cold_repeats,
            "warm_process_repeats": args.warm_repeats,
            "minimum_target_window_coverage": args.minimum_window_coverage,
            "evidence_text_included": args.include_evidence,
        },
        "methodology": [
            "No paid model or model API was called; source acquisition is disabled.",
            "Measurements use an isolated SQLite backup and a copied local tokenizer cache.",
            "Fresh process latency includes launching Python but source subtitles and tokenizer files are already cached.",
            "Warm process latency excludes initialization and follows one unmeasured prime for each question and budget.",
            "Both latency modes measure evidence retrieval only, not URL download, ASR, video decoding, or a model's answer.",
            "Payload tokens are counted independently over the complete compact JSON response; they are not Astra billing tokens.",
            "Evidence text is replaced by its SHA-256 and character count only after full-payload token counting and window scoring, unless private-cache debug export is explicitly enabled.",
            "The transcript baseline contains text only and therefore deliberately exposes metadata overhead.",
            "Target time-window recall is a retrieval proxy. It does not establish that a model understood or correctly answered a question.",
            "Window overlap may overstate coverage when a returned passage's text is truncated; per-window flags retain this limitation.",
            "Manually authored questions from one development video, including paraphrases of one topic, are not representative or held out.",
            "Percentiles over these small samples are descriptive and must not be presented as production latency guarantees.",
        ],
        "summary": summarize(measurements),
        "measurements": measurements,
        "manual_review": {
            "model_answer_accuracy_measured": False,
            "status": "Reference answers describe the development annotations. This run measures retrieval.",
            "reference_answers": [
                {
                    key: item[key]
                    for key in ("id", "question", "reference_answer", "manual_review_notes")
                }
                for item in fixture["questions"]
            ],
        },
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if args.include_evidence:
        output.chmod(0o600)
    return report


def measure(
    service, question, payload, baseline, budget, mode, repetition, wall, threshold, timings
):
    encoded = json_text(payload)
    count = service.budget.count(encoded)
    source_tokens = baseline["text_tokens_estimate"]
    return {
        "question_id": question["id"],
        "question": question["question"],
        "budget": budget,
        "mode": mode,
        "repetition": repetition,
        "wall_seconds": round(wall, 6),
        "worker_timings": timings,
        "payload_text_tokens_estimate": count,
        "payload_utf8_bytes": len(encoded.encode()),
        "reported_text_tokens_estimate": payload.get("accounting", {}).get("text_tokens_estimate"),
        "within_requested_budget": count <= budget,
        "payload_to_full_transcript_token_ratio": count / source_tokens if source_tokens else None,
        "saved_vs_full_transcript_text_fraction": 1 - count / source_tokens
        if source_tokens
        else None,
        **score_windows(payload, question, threshold),
        "payload": payload,
    }


def summarize(measurements: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for measurement in measurements:
        groups[(measurement["budget"], measurement["mode"])].append(measurement)
    summaries = []
    for (budget, mode), group in sorted(groups.items()):
        latencies = [item["wall_seconds"] for item in group]
        hits = sum(item["window_hits"] for item in group)
        targets = sum(item["required_window_count"] for item in group)
        summaries.append(
            {
                "budget": budget,
                "mode": mode,
                "samples": len(group),
                "question_count": len({item["question_id"] for item in group}),
                "latency_p50_seconds": statistics.median(latencies),
                "latency_p95_seconds": percentile(latencies, 0.95),
                "mean_payload_text_tokens_estimate": statistics.mean(
                    item["payload_text_tokens_estimate"] for item in group
                ),
                "all_responses_within_budget": all(
                    item["within_requested_budget"] for item in group
                ),
                "micro_target_window_recall": hits / targets if targets else None,
                "all_windows_hit_fraction": statistics.mean(
                    item["all_target_windows_hit"] for item in group
                ),
                "mean_payload_to_full_transcript_token_ratio": statistics.mean(
                    item["payload_to_full_transcript_token_ratio"] for item in group
                ),
            }
        )
    return summaries


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--home", default=str(directory.parent / ".vidlens"))
    parser.add_argument("--questions", default=str(directory / "reading_questions.json"))
    parser.add_argument("--output", default=str(directory / "results" / "reading.json"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[2500, 4000])
    parser.add_argument("--cold-repeats", type=int, default=1)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--minimum-window-coverage", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--include-evidence",
        action="store_true",
        help="Debug only: retain evidence text; --output must be inside the private --home cache.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--url", help=argparse.SUPPRESS)
    parser.add_argument("--question", help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.cold_repeats < 1 or args.warm_repeats < 1:
        parser.error("Both repeat counts must be positive.")
    if not 0 < args.minimum_window_coverage <= 1:
        parser.error("Window coverage threshold must be in (0, 1].")
    report = run(args)
    print(
        json_text(
            {
                "output": str(Path(args.output).resolve()),
                "baseline": report["baseline"],
                "summary": report["summary"],
                "source_changed_during_run": report["source_changed_during_run"],
            }
        )
    )


if __name__ == "__main__":
    main()

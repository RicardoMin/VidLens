"""Measure time-local term retention against independently verified video captions.

This is an anchor check, not WER/CER or a measure of overall explanation quality.
The output contains only scores, timestamps, terms and provenance hashes; source
transcript text stays in the private input files. References must be frozen before
the compared transcripts are inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path


def contains_term(text: str, term: str) -> bool:
    """Allow case/spacing/punctuation variants, but never SYN inside Sync."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[，。！？、,.!?;；:：()（）\[\]【】'\"]", " ", text)
    term = unicodedata.normalize("NFKC", term).casefold()
    letters = [letter for letter in term if not letter.isspace()]
    if not letters:
        return False
    pattern = r"\s*".join(re.escape(letter) for letter in letters)
    if re.fullmatch(r"[a-z0-9]", letters[0]):
        pattern = r"(?<![a-z0-9])" + pattern
    if re.fullmatch(r"[a-z0-9]", letters[-1]):
        pattern += r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def validate_interval(interval, label):
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(isinstance(value, bool) or not isinstance(value, (float, int))
               or not math.isfinite(value) for value in interval)
        or not 0 <= interval[0] < interval[1]
    ):
        raise ValueError(f"{label} must contain finite source times with 0 <= start < end")


def evaluate(case, transcript):
    if not case.get("term_score_eligible"):
        raise ValueError("This reference has no independently verified term anchors")
    validate_interval(case["asr_clip"], "reference clip")
    validate_interval(transcript["range"], "transcript range")
    if transcript["range"] != case["asr_clip"]:
        raise ValueError("Compare the same source interval for every condition")
    segments = transcript["segments"]
    for segment in segments:
        validate_interval([segment["start"], segment["end"]], "transcript segment")
        if not isinstance(segment["text"], str):
            raise TypeError("Transcript segments must contain text")
        if segment["start"] < case["asr_clip"][0] or segment["end"] > case["asr_clip"][1]:
            raise ValueError("Segment is outside the source clip; check relative versus source time")
    results = []
    for anchor in case["asr_anchors"]:
        validate_interval(anchor["window"], "anchor window")
        left, right = anchor["window"]
        observed = " ".join(segment["text"] for segment in segments
                            if segment["start"] < right and segment["end"] > left)
        retained = any(contains_term(observed, word) for word in anchor["alternatives"])
        results.append({"id": anchor["id"], "term": anchor["term"],
                        "window": anchor["window"], "retained": retained})
    if not results:
        raise ValueError("The chosen case has no anchors")
    model = transcript["model"]
    if Path(model).is_absolute():
        model = Path(model).name
    return {
        "schema_version": 1,
        "matcher": "nfkc_casefold_spacing_ascii_token_boundaries_v1",
        "metric": "time-local burnt-in-caption anchor retention; not WER/CER",
        "case": case["id"],
        "source_url": case["url"],
        "range": transcript["range"],
        "backend": transcript["backend"],
        "model": model,
        "timestamp_precision": transcript.get("timestamp_precision", "model_segment"),
        "vad": transcript.get("vad", "backend_default"),
        "window_rule": "Segments overlapping an anchor window are searched in full; coarse audio windows include extra context.",
        "vocabulary": transcript.get("vocabulary", []),
        "elapsed_seconds": transcript.get("elapsed_seconds"),
        "matched": sum(row["retained"] for row in results),
        "total": len(results),
        "anchors": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--case", help="Required when the reference contains multiple eligible cases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    cases = [case for case in reference["cases"] if case.get("term_score_eligible")
             and (args.case is None or case["id"] == args.case)]
    if len(cases) != 1:
        parser.error("Choose exactly one eligible reference case with --case")
    try:
        report = evaluate(cases[0], transcript)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    report["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    report["transcript_sha256"] = hashlib.sha256(args.transcript.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case", "model", "matched", "total")}))


if __name__ == "__main__":
    main()

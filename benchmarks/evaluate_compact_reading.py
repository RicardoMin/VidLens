"""Compare lossless evidence protocols using an existing cache, opened read-only.

Example (from the repository root):
    .venv/bin/python benchmarks/evaluate_compact_reading.py \
        --home .vidlens --video-id bilibili:BV1KT2yBJESn:p1 \
        --output benchmarks/results/compact-reading.json

No media, transcription, network acquisition or model inference is run. Results
contain hashes/counts rather than transcript text or local cache paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from vidlens.budget import TokenBudget, compact_json
from vidlens.evidence_format import expand_evidence
from vidlens.models import Segment, Video
from vidlens.reading import EvidenceReader, validate_budget


class CachedSnapshot:
    """The reader's snapshot contract without a writable Store or delivery methods."""

    def __init__(self, video, segments, digest):
        self.video, self.segments, self.version = video, segments, (digest,)

    def snapshot(self, video_id, version):
        if video_id != self.video.id:
            raise ValueError("Video does not belong to this snapshot")
        if version == self.version:
            return None
        return self.version, self.video, self.segments, []


def evaluate(home: Path, video_id: str, max_tokens: int = 6000):
    validate_budget(max_tokens)
    database = (home / "evidence.sqlite3").resolve()
    if not database.is_file():
        raise ValueError("No existing evidence cache; read the video with VidLens first")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        row = db.execute("SELECT data FROM videos WHERE id=?", (video_id,)).fetchone()
        if row is None:
            raise ValueError("Requested video is not in the existing cache")
        video = Video(**json.loads(row["data"]))
        rows = db.execute("SELECT id,start,end,text,kind,source,language FROM segments "
                          "WHERE video_id=? AND kind!='ocr' ORDER BY start,end,id", (video_id,)).fetchall()
    segments = [Segment(**dict(row)) for row in rows]
    if not segments or video.duration <= 0:
        raise ValueError("The video needs indexed speech and a known duration")
    # Match the reader's whole-interval overlap, including no invented suffix
    # beyond a platform duration. The digest binds every source character.
    segments = [segment for segment in segments if segment.end > 0 and segment.start < video.duration]
    source_digest = hashlib.sha256(compact_json([
        [s.id, s.start, s.end, s.text, s.kind, s.source, s.language] for s in segments
    ]).encode()).hexdigest()
    budget = TokenBudget(home)
    reader = EvidenceReader(CachedSnapshot(video, segments, source_digest), budget)
    result = {
        "benchmark": "cached_lossless_evidence_protocol_v1",
        "video_id": video_id, "duration_seconds": video.duration,
        "source_sha256": source_digest, "speech_segments": len(segments),
        "source_characters": sum(len(s.text) for s in segments),
        "budget_per_page": max_tokens, "token_method": budget.method,
        "raw_text_tokens_estimate": budget.count("\n".join(s.text for s in segments)),
        "scope": "Existing cached speech; read-only snapshot; no pictures or acquisition.",
        "model_usage": "not_measured",
        "excludes": ["images", "model reasoning", "tool schemas", "conversation history",
                     "transport envelopes", "download and ASR time"],
        "results": {},
    }
    expected = {segment.id: segment.text for segment in segments}
    for representation in ("verbose", "compact"):
        cursor, pages, tokens, groups, received, cursors = None, 0, 0, 0, {}, set()
        while True:
            page = reader.read_interval(video_id, 0, video.duration, max_tokens=max_tokens,
                                        cursor=cursor, record_delivery=False, speech_only=True,
                                        representation=representation)
            count = budget.count(compact_json(page))
            if count > max_tokens:
                raise AssertionError("Serialized page exceeds its advertised budget")
            pages += 1
            tokens += count
            groups += len(page["evidence"])
            for item in expand_evidence(page):
                previous = received.get(item["id"], "")
                if item.get("text_offset", 0) != len(previous):
                    raise AssertionError("Evidence character ranges overlap or skip source text")
                text = item["text"][:-1] if item.get("excerpt_truncated") else item["text"]
                received[item["id"]] = previous + text
            cursor = page["next_cursor"]
            if cursor is None:
                break
            if cursor in cursors:
                raise AssertionError("Reader repeated a continuation cursor")
            cursors.add(cursor)
        if received != expected:
            raise AssertionError("Protocol did not reconstruct every source ID and character")
        result["results"][representation] = {
            "pages": pages, "serialized_text_tokens_estimate": tokens,
            "paragraphs_or_cues": groups, "source_characters_reconstructed": sum(map(len, received.values())),
            "exact_source_reconstruction": True,
        }
    verbose = result["results"]["verbose"]["serialized_text_tokens_estimate"]
    compact = result["results"]["compact"]["serialized_text_tokens_estimate"]
    result["estimated_text_reduction_fraction"] = 1 - compact / verbose
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.home, args.video_id, args.max_tokens)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()

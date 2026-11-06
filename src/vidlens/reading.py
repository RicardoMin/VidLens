"""Select and paginate local evidence independently of source and media acquisition."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import re
import time
from dataclasses import dataclass
from functools import cached_property

from .budget import TokenBudget, compact_json
from .models import Segment, Video, VidLensError
from .retrieval import PassageIndex, navigation, source_link
from .store import Store, union_intervals


def validate_budget(max_tokens: int):
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 768 <= max_tokens <= 32000
    ):
        raise VidLensError("invalid_budget", "max_tokens must be an integer from 768 to 32000.")


def validate_interval(video: Video, start: float, end: float, max_duration: float = 7200):
    if not all(
        isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
        for v in (start, end)
    ):
        raise VidLensError("invalid_interval", "Start and end must be finite seconds.")
    if start < 0 or end <= start or end - start > max_duration:
        raise VidLensError(
            "invalid_interval",
            f"Use 0 <= start < end and an interval <= {max_duration:g} seconds.",
        )
    if video.duration > 0 and start >= video.duration:
        raise VidLensError("invalid_interval", "Start is past the end of the video.")
    return float(start), min(float(end), video.duration) if video.duration > 0 else float(end)


@dataclass
class ReadingSnapshot:
    version: tuple
    video: Video
    segments: list[Segment]
    warnings: list[str]
    budget: TokenBudget

    @cached_property
    def index(self):
        return PassageIndex(self.segments)

    @cached_property
    def transcript_tokens(self):
        return self.budget.count("\n".join(s.text for s in self.segments if s.kind != "ocr"))


class EvidenceReader:
    def __init__(self, store: Store, budget: TokenBudget):
        self.store = store
        self.budget = budget
        self._current: ReadingSnapshot | None = None

    def _snapshot(self, video_id):
        cached = self._current
        data = self.store.snapshot(video_id, cached.version if cached else None)
        if data is None:
            return cached
        snapshot = ReadingSnapshot(*data, self.budget)
        self._current = snapshot
        return snapshot

    def overview(self, video: Video) -> dict:
        snapshot = self._snapshot(video.id)
        video, segments = snapshot.video, snapshot.segments
        speech = [s for s in segments if s.kind != "ocr"]
        chapters = navigation(video, segments)
        warnings = self._warnings(snapshot.warnings, segments)
        if not speech:
            warnings = warnings + [
                "No transcript available. Visual reading or optional local transcription is required."
            ]
        return {
            "video_id": video.id,
            "title": video.title[:500],
            "url": video.url,
            "duration": video.duration,
            "language": video.language,
            "speech_scope": self._speech_scope(video, segments),
            "source_notes": self._source_notes(video),
            "transcript": {
                "segments": len(speech),
                "sources": sorted({s.source for s in speech}),
                "full_text_tokens_estimate": snapshot.transcript_tokens,
            },
            "outline_method": "sampled extractive navigation; not a complete summary",
            "outline": chapters,
            "warnings": warnings,
            "next_actions": [
                "Search using the question and the video's spoken terminology.",
                "For a full overview, inspect every topic; this outline contains samples.",
                "Inspect frames for slides, code, formulas, experimental conditions and visual-only details.",
            ],
        }

    @staticmethod
    def _warnings(warnings, segments):
        warnings = list(warnings)
        if any(s.kind == "transcript" for s in segments):
            warnings = [w for w in warnings if "metadata only" not in w.lower()]
        return warnings

    @staticmethod
    def _speech_scope(video, segments):
        if any(s.kind == "subtitle" for s in segments):
            return "available_platform_or_imported_captions"
        if any(s.kind == "transcript" for s in segments):
            intervals = union_intervals(
                [tuple(i) for i in video.metadata.get("transcribed_intervals", [])]
            )
            whole = bool(intervals and intervals[0][0] <= 0 and intervals[0][1] >= video.duration)
            return {
                "kind": "local_asr",
                "processed_intervals": intervals[:20],
                "whole_video_processed": whole,
            }
        return "no_transcript"

    @staticmethod
    def _source_notes(video):
        """Expose concise author errata already present in metadata, separately from speech."""
        notes = []
        for line in video.description.splitlines():
            if re.match(
                r"\s*(?:(?:corrections?|errata|erratum)\b|更正|勘误|勘誤|修正|訂正|订正)",
                line,
                re.IGNORECASE,
            ):
                notes.append(line.strip()[:400])
        return notes[:4]

    def read(self, video: Video, question: str, max_tokens: int, hit: bool, started: float):
        if not isinstance(question, str) or not question.strip():
            raise VidLensError("invalid_query", "Provide a question, or use the study plan for a whole-video reading.")
        snapshot = self._snapshot(video.id)
        video, segments = snapshot.video, snapshot.segments
        candidates = snapshot.index.rank(question, 8)
        original_tokens = snapshot.transcript_tokens
        # Source URL and provenance occur once; citations append t=<start> to this URL.
        evidence = [
            {
                "start": round(p.start, 3),
                "end": round(p.end, 3),
                "text": p.text,
                "source": f"{p.kind}/{p.source}",
            }
            for p in candidates
        ]
        result = {
            "video_id": video.id,
            "title": video.title[:300],
            "source_url": video.url,
            "timestamp_parameter": "t",
            "duration": video.duration,
            "mode": "question_evidence",
            "evidence": evidence,
            "warnings": self._warnings(snapshot.warnings, segments),
            "speech_scope": self._speech_scope(video, segments),
            "source_notes": self._source_notes(video),
            "cache_hit": hit,
            "available_transcript_tokens_estimate": original_tokens,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "transcript_complete": False,
        }
        if not evidence:
            result["warnings"].append(
                "No matching indexed text. Inspect chapters/frames or use local ASR; this is not proof of absence."
            )
            result["local_asr_installed"] = importlib.util.find_spec("faster_whisper") is not None
        packed = self.budget.finish(result, max_tokens, "evidence", len(candidates))
        self.store.delivered(video.id, packed["evidence"], "excerpt")
        return packed

    def search(self, video_id, query, max_tokens=3000, limit=6):
        validate_budget(max_tokens)
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise VidLensError("invalid_query", "Use a nonempty query up to 2000 characters.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise VidLensError("invalid_limit", "limit must be from 1 to 20.")
        snapshot = self._snapshot(video_id)
        video = snapshot.video
        evidence = [p.to_dict(video) for p in snapshot.index.rank(query, limit)]
        result = {
            "video_id": video_id,
            "query": query,
            "retrieval": "local BM25 + technical term aliases + temporal passages",
            "evidence": evidence,
            "warnings": [],
        }
        if not evidence:
            result["warnings"] = [
                "No matching indexed evidence. This does not prove the video lacks the answer."
            ]
            result["next_actions"] = [
                "Use the video's language or exact code identifiers; inspect the outline or unseen frames."
            ]
        else:
            result["next_actions"] = [
                "Read adjacent intervals for conditions and counterexamples; OCR is not visually verified."
            ]
        self.budget.finish(result, max_tokens, "evidence", len(evidence))
        self.store.delivered(video_id, result["evidence"], "excerpt")
        return result

    def read_interval(self, video_id, start, end, max_tokens=4000, cursor=None, *,
                      speech_only=False, record_delivery=True):
        validate_budget(max_tokens)
        snapshot = self._snapshot(video_id)
        video = snapshot.video
        start, end = validate_interval(video, start, end,
                                      max_duration=float("inf") if speech_only else 7200)
        segments = [
            s for s in snapshot.segments
            if s.end > start and s.start < end and (not speech_only or s.kind != "ocr")
        ]
        revision = hashlib.sha256(
            compact_json([
                video_id, start, end, video.metadata.get("revision", ""),
                [(s.id, s.start, s.end, s.source, s.language) for s in segments],
            ]).encode()
        ).hexdigest()[:12]
        index, offset = 0, 0
        if cursor not in (None, 0, "0"):
            try:
                previous_revision, index_text, offset_text = str(cursor).split(":")
                index, offset = int(index_text), int(offset_text)
                if (
                    previous_revision != revision
                    or index < 0
                    or offset < 0
                    or index >= len(segments)
                ):
                    raise ValueError
                if offset >= len(segments[index].text):
                    raise ValueError
            except (ValueError, TypeError) as exc:
                raise VidLensError(
                    "invalid_cursor",
                    "This cursor does not match the interval or its current evidence.",
                    "Restart the interval without a cursor after changing the transcript or OCR index.",
                ) from exc
        items = [
            {
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "text": s.text[offset if n == index else 0 :],
                "text_offset": offset if n == index else 0,
                "kind": s.kind,
                "source": s.source,
                "url": source_link(video, s.start),
            }
            for n, s in enumerate(segments)
            if n >= index
        ]
        result = {
            "video_id": video_id,
            "start": start,
            "end": end,
            "cursor": cursor,
            "next_cursor": None,
            "evidence": items,
            "warnings": [],
        }
        if not items:
            result["warnings"] = [
                "No indexed text in this interval; the visual/audio content may still contain information."
            ]
        # Reserve the pagination field before applying the budget.
        result["next_cursor"] = f"{revision}:{len(segments)}:99999999" if items else None
        # Leave room for the final cursor/accounting without clipping an already paginated result.
        self.budget.finish(result, max_tokens - 64, "evidence", len(segments))
        consumed = len(result["evidence"])
        if consumed and result["evidence"][-1].get("excerpt_truncated"):
            last = result["evidence"][-1]
            used_chars = len(last["text"]) - 1  # TokenBudget appends a single ellipsis.
            if used_chars <= 0:
                raise VidLensError("budget_too_small", "Increase max_tokens to read this evidence.")
            result["next_cursor"] = (
                f"{revision}:{index + consumed - 1}:{last['text_offset'] + used_chars}"
            )
        else:
            result["next_cursor"] = (
                f"{revision}:{index + consumed}:0"
                if consumed and index + consumed < len(segments)
                else None
            )
        result["accounting"]["budget"] = max_tokens
        if items and not consumed:
            raise VidLensError("budget_too_small", "Increase max_tokens to read this evidence.")
        self.budget.recount(result)
        if record_delivery:
            self.store.delivered_text(video_id, result["evidence"])
        return result

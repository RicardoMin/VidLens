"""Bounded screen-text discovery for a question, without an extra model call."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from .models import VidLensError
from .retrieval import source_link


def validate_visual_query(question, visual_queries):
    if not isinstance(question, str) or len(question) > 4000:
        raise VidLensError("invalid_query", "question must be text of at most 4000 characters.")
    if visual_queries is None:
        visual_queries = []
    if (
        not isinstance(visual_queries, (list, tuple))
        or len(visual_queries) > 6
        or any(not isinstance(q, str) or not q.strip() or len(q) > 300 for q in visual_queries)
    ):
        raise VidLensError("invalid_visual_queries", "Use up to six short screen-evidence questions.")
    if visual_queries and not question.strip():
        raise VidLensError("invalid_query", "Provide the user's question with visual_queries.")
    return question.strip(), [q.strip() for q in visual_queries]


class QuestionFrames:
    """Probe original changes, refine near relevant states, deliver only selected images."""

    PROBE_LIMIT = 48
    REFINEMENT_LIMIT = 48

    def __init__(self, service):
        self.service = service

    def read(self, video, asset, start, end, question, visual_queries, limit, *, preview=False):
        from .learning import LearningReader
        from .visual_relevance import QueryVisualSelector

        service = self.service
        # Reuse a complete full scan when present, but a local question should
        # not require downloading and decoding an entire multi-hour lecture.
        _, _, cached_scan = service.learning._scan_data(video.id)
        scan = cached_scan if cached_scan and cached_scan.get("status") == "complete" else None
        cursor_scope = "video" if scan is not None else "interval"
        if scan is None:
            scan = service.visual.scan(asset, start=start, end=end, limit=1000)
            candidates = list(scan["candidates"])
            cursor = scan.get("next_cursor")
            while cursor is not None and scan.get("status") == "complete":
                page = service.visual.scan(asset, start=start, end=end, cursor=cursor, limit=1000)
                if page.get("scan_id") != scan.get("scan_id"):
                    raise VidLensError("scan_changed", "The screen index changed; retry the interval.")
                candidates.extend(page["candidates"])
                if page.get("status") != "complete":
                    scan = {**scan, "status": page.get("status"), "error": page.get("error")}
                    break
                cursor = page.get("next_cursor")
            scan = {**scan, "candidates": candidates}
        candidates = [
            {**row, "source_candidate_cursor": index}
            for index, row in enumerate(scan["candidates"])
            if start <= row["timestamp"] < end
        ]
        segments = [asdict(s) for s in service.store.segments(video.id)
                    if s.kind != "ocr" and s.start < end and s.end > start]
        probes = QueryVisualSelector.probe_candidates(
            question, candidates, segments, start=start, end=end,
            visual_queries=visual_queries, limit=self.PROBE_LIMIT,
        )
        if not probes:
            # A stable screen can begin before this interval, with no detected
            # change inside it. Retrieve its actual displayed state anyway.
            probes = [{"timestamp": start + (i + 0.5) * (end - start) / min(limit, 4),
                       "probe_reason": "stable_interval"} for i in range(min(limit, 4))]
        notices, rows, decoded = [], [], []

        def decode(planned):
            for offset in range(0, len(planned), 12):
                batch = planned[offset:offset + 12]
                frames = service.visual.read_at(asset, [row["timestamp"] for row in batch], ocr=True)
                notices.extend(getattr(service.visual, "last_warnings", []))
                decoded.extend(frames)
                for frame in frames:
                    requested = (frame.requested_timestamp if frame.requested_timestamp is not None
                                 else frame.timestamp)
                    origin = min(batch, key=lambda row: abs(row["timestamp"] - requested))
                    row = {**origin, **frame.to_dict(),
                           "url": source_link(video, frame.timestamp),
                           "evidence_id": f"frame:{frame.timestamp}"}
                    row["image_digest"] = hashlib.sha256(Path(frame.path).read_bytes()).hexdigest()
                    rows.append(row)

        decode(probes)
        preliminary = QueryVisualSelector.select(
            question, rows, segments, start=start, end=end,
            visual_queries=visual_queries, limit=max(limit, 8),
        )
        refinement = QueryVisualSelector.refinement_candidates(
            preliminary["selected"], candidates, start=start, end=end,
            limit=self.REFINEMENT_LIMIT,
            seen=[row.get("requested_timestamp", row["timestamp"]) for row in rows],
        )
        decode(refinement)
        ranked = QueryVisualSelector.select(
            question, rows, segments, start=start, end=end,
            visual_queries=visual_queries, limit=12 if preview else limit,
        )
        selected = ranked["selected"]
        if preview:
            # A visual review needs alternatives to the lexical ranking. Keep
            # twelve time-spanning states as well as twelve query-ranked ones,
            # allowing the AI to find relationships that OCR cannot express.
            selected = list(selected)
            available = sorted(rows, key=lambda row: row["timestamp"])
            for index in range(12):
                target = start + (index + 0.5) * (end - start) / 12
                if available:
                    choice = min(available, key=lambda row: abs(row["timestamp"] - target))
                    if all(abs(row["timestamp"] - choice["timestamp"]) > 1e-6 for row in selected):
                        selected.append(choice)
            selected.sort(key=lambda row: row["timestamp"])
        mode = "question_screen_text"
        if not preview and not any(row.get("ocr", "").strip() for row in rows):
            # No reliable screen-text signal: retain an honest timeline fallback
            # rather than presenting an ASR-only ranking as screen understanding.
            mode = "timeline_fallback"
            overview = LearningReader._overview_candidates({"start": start, "end": end}, candidates, limit)
            if not overview:
                overview = [{"timestamp": start + (i + 0.5) * (end - start) / limit}
                            for i in range(limit)]
            previous = len(rows)
            decode(overview)
            selected = rows[previous:]
        result = []
        for row in selected:
            clean = {k: v for k, v in row.items()
                     if k not in {"image_digest", "signature", "transcript", "speech", "text"}}
            clean["ocr"] = service.budget.truncate(clean.get("ocr", ""), 400)
            for field in ("matched_terms", "selection_reasons"):
                if isinstance(clean.get(field), list):
                    clean[field] = [service.budget.truncate(str(value), 40)
                                    for value in clean[field][:8]]
            # A duplicate in the probe pool may not be in the delivered set.
            if clean.get("duplicate_of") not in {item["timestamp"] for item in result}:
                clean["duplicate_of"] = None
            result.append(clean)
        return result, {
            "selection": mode,
            "question": question,
            "visual_queries": visual_queries,
            "probe_count": len(probes),
            "refinement_count": len(refinement),
            "raw_candidate_count": len(candidates),
            "scan_status": scan.get("status"),
            "scan_error": scan.get("error"),
            "scan_id": scan.get("scan_id"),
            "candidate_cursor_scope": cursor_scope,
            "scan_range": [0, video.duration] if cursor_scope == "video" else [start, end],
            "query_terms": ranked.get("query_terms", [])[:48],
            "unmatched_query_terms": ranked.get("unmatched_query_terms", [])[:24],
            "visual_query_matches": ranked.get("visual_query_matches", []),
            "unmatched_visual_queries": ranked.get("unmatched_visual_queries", []),
            "relevance_status": ranked.get("status"),
            "metrics": ranked.get("metrics", {}),
            "screen_text_frames": sum(bool(row.get("ocr", "").strip()) for row in rows),
            "screen_text_status": ("available" if any(row.get("ocr", "").strip() for row in rows)
                                   else "unavailable"),
            "selection_scope": "lexical_screen_and_speech_relevance; inspect images for reasoning steps",
            "warnings": list(dict.fromkeys(notices)),
        }, decoded

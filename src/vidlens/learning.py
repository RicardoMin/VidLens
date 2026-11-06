"""Whole-video reading, visual discovery and reusable, source-linked learning notes."""

from __future__ import annotations

import json
import math
import re
import time

from .budget import compact_json
from .models import VidLensError
from .reading import validate_budget
from .retrieval import query_tokens, source_link, tokenize


def reading_intent(question, intent):
    if intent not in {"auto", "summary", "question"}:
        raise VidLensError("invalid_intent", "intent must be auto, summary, or question.")
    if intent != "auto":
        if intent == "question" and not question.strip():
            raise VidLensError("invalid_query", "Question reading needs a question.")
        return intent
    if not question.strip():
        return "summary"
    # A tutorial may discuss model.summary() or an `overview` variable. Route
    # the requested task, rather than those identifiers, to whole-video reading.
    prose = re.sub(r"`[^`]*`|\b[\w.]+\s*\([^()\n]*\)", " ", question).strip()
    summary = re.search(
        r"(?:总结|概括|梳理)(?:一下|一遍|下)?(?:这|整|本|该|视频|课程|[。！？!?]|$)|"
        r"整[个段部]?(?:视频|课程)|学懂|讲了什么|主要(?:内容|观点)|"
        r"(?:讲解|解释|带我理解)(?:一下)?(?:这[个段部]?|本|该)(?:视频|课程)|"
        r"\b(?:summari[sz]e|key (?:points|takeaways))\b|"
        r"\b(?:summary|overview)\s+(?:of|for|about)\b|"
        r"\b(?:give|provide|write|make|create)\b.{0,50}\b(?:summary|overview)\b|"
        r"\bwalk me through (?:this|the|entire|whole) (?:video|lecture|course)\b|"
        r"^(?:(?:a|brief|quick|short|detailed|full|complete)\s+)*(?:summary|overview)[.!?]?$",
        prose,
        re.IGNORECASE,
    )
    return "summary" if summary else "question"


class LearningReader:
    def __init__(self, service):
        from .references import ReferenceStore
        from .study import StudyEngine

        self.service = service
        self.store, self.budget = service.store, service.budget
        self.study = StudyEngine(self.store)
        self.references = ReferenceStore(self.store, self.study)
        with self.store.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS visual_scans (
                video_id TEXT PRIMARY KEY REFERENCES videos(id),
                revision TEXT NOT NULL, data TEXT NOT NULL
            )""")

    def status(self, video_id, max_tokens=4000, cursor=0):
        validate_budget(max_tokens)
        plan = self.study.plan(video_id)
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or cursor < 0
            or cursor > len(plan["sections"])
        ):
            raise VidLensError("invalid_cursor", "Use the study plan's next_cursor.")
        sections = []
        for section in plan["sections"][cursor:]:
            item = {key: section[key] for key in ("id", "start", "end", "title")}
            item["progress"] = {
                key: section["progress"][key]
                for key in ("text_status", "visual_status", "note_status")
            }
            sections.append(item)
        payload = {
            "video_id": video_id,
            "revision": plan["revision"],
            "mode": "study_plan",
            "progress": {
                key: value
                for key, value in plan["progress"].items()
                if key not in {"meaning", "full_video_review_complete"}
            },
            "sections": sections,
            "section_count": len(plan["sections"]),
            "next_cursor": len(plan["sections"]),
            "workflow": ["scan_video", "read_section", "save_note"],
            "speech_scope": self.service.reader._speech_scope(
                self.store.get(video_id), self.store.segments(video_id)
            ),
            "duration_known": not plan.get("gaps", {}).get("unknown_duration", False),
        }
        if payload["speech_scope"] != "available_platform_or_imported_captions":
            payload["transcription"] = self.service.transcription_options()
        self.budget.finish(payload, max_tokens - 32, "sections", len(sections))
        consumed = len(payload["sections"])
        if sections and not consumed:
            raise VidLensError("budget_too_small", "Increase max_tokens to fit a chapter entry.")
        payload["next_cursor"] = (
            cursor + consumed if cursor + consumed < len(plan["sections"]) else None
        )
        payload["accounting"]["budget"] = max_tokens
        return self.budget.recount(payload)

    def summary(self, video, question, max_tokens, cache_hit, started):
        if max_tokens < 1200:
            raise VidLensError("budget_too_small", "Whole-video planning needs max_tokens >= 1200.")
        metadata = {
            "intent": "summary",
            "source_url": video.url,
            "title": self.budget.truncate(video.title, 100),
            "learning_goal": self.budget.truncate(question, 100),
            "source_notes": self.service.reader._source_notes(video),
            "resource_links": list(
                dict.fromkeys(re.findall(r"https?://[^\s<>]+", video.description))
            )[:6],
            "transcript_complete": False,
            "cache_hit": cache_hit,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        reserve = self.budget.count(compact_json(metadata)) + 64
        if max_tokens - reserve < 768:
            raise VidLensError(
                "budget_too_small",
                "Increase max_tokens to include the source's notes and chapter plan.",
            )
        payload = self.status(video.id, max_tokens=max_tokens - reserve)
        payload.update(metadata)
        payload["accounting"]["budget"] = max_tokens
        return self.budget.recount(payload)

    def _scan_data(self, video_id):
        from .storyboard import StoryboardScanner

        video = self.store.get(video_id)
        revision = f"scan-v{StoryboardScanner.VERSION}:" + str(video.metadata.get("revision", ""))
        local = video.metadata.get("local_media_path")
        if local:
            from pathlib import Path

            try:
                stat = Path(local).stat()
            except OSError as exc:
                raise VidLensError(
                    "media_missing",
                    "The imported local media is unavailable.",
                    "Restore the media file to inspect images; imported speech remains readable.",
                ) from exc
            revision += f":{Path(local).resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        with self.store.connect() as db:
            row = db.execute(
                "SELECT data FROM visual_scans WHERE video_id=? AND revision=?",
                (video_id, revision),
            ).fetchone()
        return video, revision, json.loads(row["data"]) if row else None

    @staticmethod
    def _validate_view(view):
        if view not in {"navigation", "full"}:
            raise VidLensError("invalid_visual_view", "Use navigation or full for visual candidates.")

    def _candidate_view(self, video_id, scan, view):
        """Keep raw detection intact; derive an independently versioned reading index."""
        self._validate_view(view)
        raw = scan["candidates"]
        metadata = {"view": "full", "raw_candidate_count": scan.get("candidate_count", len(raw))}
        if view == "full" or not raw:
            return raw, metadata, None
        from .selection import VisualSelector

        if scan.get("status") != "complete" or not scan.get("scan_id"):
            metadata["selection_status"] = "raw_scan_incomplete"
            return raw, metadata, None
        try:
            with self.service._file_lock("content", video_id), self.service._visual_lock:
                video, revision, current = self._scan_data(video_id)
                if current is None or current.get("scan_id") != scan["scan_id"]:
                    raise VidLensError("stale_revision", "The visual scan changed; reload its map.")
                asset = self.service.media.get(
                    video, 0, video.duration, False, lambda: self.service.provider, max_height=360
                )
                selector = VisualSelector(self.service.home / "visual", self.service.visual._ffmpeg)
                index = selector.build(asset, scan)
                if self._scan_data(video_id)[1] != revision:
                    raise VidLensError("stale_revision", "The video changed during visual selection.")
            metadata.update(
                selection_status=index.get("status"),
                selection_id=index.get("selection_id"),
                selection_cached=index.get("cached", False),
                selection_method=index.get("method"),
                selection_metrics=index.get("metrics", {}),
            )
            if index.get("status") == "complete" and index.get("navigation_candidates"):
                metadata["view"] = "navigation"
                return [VisualSelector.public(row) for row in index["navigation_candidates"]], metadata, index
            metadata["selection_error"] = index.get("error")
        except VidLensError as exc:
            if exc.code == "stale_revision":
                raise
            metadata.update(selection_status="unavailable", selection_error=exc.to_dict())
        except (OSError, ValueError) as exc:
            metadata.update(selection_status="unavailable", selection_error={
                "code": "selection_io_failed", "message": str(exc),
            })
        return raw, metadata, None

    def scan(self, video_id, cursor=0, limit=100, refresh=False, view="navigation"):
        self._validate_view(view)
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise VidLensError("invalid_cursor", "cursor must be a nonnegative integer.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise VidLensError("invalid_limit", "limit must be from 1 to 1000.")
        with self.service._file_lock("content", video_id), self.service._visual_lock:
            video, revision, data = self._scan_data(video_id)
            # A partial scan is a stable pagination snapshot. Only a fresh scan
            # request (cursor zero) or explicit refresh should repeat its decode.
            cache_hit = (
                data is not None
                and not refresh
                and (data.get("status") == "complete" or cursor > 0)
            )
            if not cache_hit:
                if video.duration <= 0:
                    raise VidLensError(
                        "unknown_duration",
                        "A known video duration is needed for whole-video scanning.",
                    )
                try:
                    asset = self.service.media.get(
                        video,
                        0,
                        video.duration,
                        False,
                        lambda: self.service.provider,
                        max_height=360,
                    )
                    data = self.service.visual.scan(asset, limit=1000)
                    candidates = list(data["candidates"])
                    next_cursor = data.get("next_cursor")
                    while next_cursor is not None and data.get("status") == "complete":
                        page = self.service.visual.scan(asset, cursor=next_cursor, limit=1000)
                        if (
                            page.get("scan_id") != data.get("scan_id")
                            or page.get("status") != "complete"
                        ):
                            raise VidLensError(
                                "scan_changed",
                                "The visual scan changed while collecting its pages; retry.",
                            )
                        candidates.extend(page["candidates"])
                        next_cursor = page.get("next_cursor")
                    data["candidates"] = candidates
                    data["retained_candidate_count"] = len(candidates)
                    data["truncated"] = next_cursor is not None
                    if data["truncated"]:
                        data["warnings"] = [
                            *data.get("warnings", []),
                            (
                                "This incomplete scan retains only its first candidate page. "
                                "Continue the retained snapshot with next_cursor; start again at "
                                "cursor 0 or refresh to retry the scan."
                            ),
                        ]
                except VidLensError as exc:
                    data = {
                        "status": "unavailable",
                        "error": exc.to_dict(),
                        "candidates": [],
                        "candidate_count": 0,
                    }
                data.pop("manifest_path", None)
                with self.store.connect() as db:
                    db.execute(
                        "INSERT OR REPLACE INTO visual_scans VALUES (?,?,?)",
                        (video_id, revision, compact_json(data)),
                    )
        candidates, selection, _ = self._candidate_view(video_id, data, view)
        if cursor > len(candidates):
            raise VidLensError("invalid_cursor", "cursor is past the last visual candidate.")
        return {
            **data,
            "video_id": video_id,
            "cached": cache_hit,
            **selection,
            "candidate_count": (len(candidates) if selection["view"] == "navigation"
                                else data.get("candidate_count", len(candidates))),
            "retained_raw_candidate_count": data.get("retained_candidate_count", len(data["candidates"])),
            "retained_candidate_count": len(candidates),
            "cursor": cursor,
            "candidates": candidates[cursor : cursor + limit],
            "next_cursor": cursor + limit if cursor + limit < len(candidates) else None,
        }

    def _frames(self, video_id, section, frame_cursor, frame_limit, frame_view="navigation"):
        _, _, scan = self._scan_data(video_id)
        if scan is None:
            self.scan(video_id, view="full")
            _, _, scan = self._scan_data(video_id)
        available, selection, _ = self._candidate_view(video_id, scan, frame_view)
        candidates = [
            candidate
            for candidate in available
            if section["start"] <= candidate["timestamp"] < section["end"]
        ]
        if not candidates:
            width = (section["end"] - section["start"]) / frame_limit
            candidates = [
                {"timestamp": section["start"] + (i + 0.5) * width} for i in range(frame_limit)
            ]
        if frame_cursor is None:
            # Independent fixed-budget review did not establish a semantic gain
            # from pixel-density ranking. Keep the stable timeline overview;
            # compact navigation is available when the reader expands images.
            overview = [row for row in scan["candidates"]
                        if section["start"] <= row["timestamp"] < section["end"]]
            selected = self._overview_candidates(section, overview or candidates, frame_limit)
        else:
            if frame_cursor > len(candidates):
                raise VidLensError(
                    "invalid_cursor", "frame_cursor is past this chapter's candidates."
                )
            selected = candidates[frame_cursor : frame_cursor + frame_limit]
        frames = (
            self.service.read_frame_times(
                video_id,
                [item["timestamp"] for item in selected],
                record_delivery=False,
            )
            if selected
            else []
        )
        for frame in frames:
            frame["evidence_id"] = f"frame:{frame['timestamp']}"
            frame["ocr"] = self.budget.truncate(frame.get("ocr", ""), 200)
        next_cursor = frame_cursor + len(selected) if frame_cursor is not None else 0
        visual = {
            **selection,
            "raw_candidate_count": sum(
                section["start"] <= row["timestamp"] < section["end"] for row in scan["candidates"]
            ),
            "candidate_count": len(candidates),
            "selection": "timeline_overview" if frame_cursor is None else "sequence",
            "next_frame_cursor": next_cursor if next_cursor < len(candidates) else None,
            "scan_status": scan.get("status"),
            "scan_error": scan.get("error"),
        }
        # A displayed frame can begin before the requested time (VFR/low FPS).
        # Keep the request-to-candidate mapping through MCP image encoding too.
        matched = set()
        for frame in frames:
            requested = frame.get("requested_timestamp")
            stamp = frame["timestamp"] if requested is None else requested
            tolerance = 1 / 30 + 1e-6 if requested is None else 1e-6
            available = [index for index in range(len(selected)) if index not in matched]
            if not available:
                break
            closest = min(available, key=lambda index: abs(selected[index]["timestamp"] - stamp))
            if abs(selected[closest]["timestamp"] - stamp) <= tolerance:
                matched.add(closest)
                if frame_cursor is not None:
                    frame["candidate_cursor"] = frame_cursor + closest
                if "source_candidate_cursor" in selected[closest]:
                    frame["source_candidate_cursor"] = selected[closest]["source_candidate_cursor"]
        missing = [index for index in range(len(selected)) if index not in matched]
        if missing:
            retry = frame_cursor + missing[0] if frame_cursor is not None else 0
            visual.update(
                status="partial",
                retry_frame_cursor=retry,
                missing_candidate_times=[selected[index]["timestamp"] for index in missing],
                next_frame_cursor=retry,
            )
        return frames, visual

    @staticmethod
    def _overview_candidates(section, candidates, limit):
        if len(candidates) <= limit:
            # Small sets may contain a one-frame flash; preserve every exact
            # candidate instead of advancing past a brief but important state.
            return candidates
        width = (section["end"] - section["start"]) / limit
        holds = [
            (
                (candidates[i + 1]["timestamp"] if i + 1 < len(candidates) else section["end"])
                - candidate["timestamp"]
            )
            for i, candidate in enumerate(candidates)
        ]
        selected = []
        for i in range(limit):
            left, right = section["start"] + i * width, section["start"] + (i + 1) * width
            midpoint = (left + right) / 2
            choices = [
                (candidate, hold)
                for candidate, hold in zip(candidates, holds)
                if left <= candidate["timestamp"] < right
            ]
            if not choices:
                selected.append({"timestamp": midpoint})
                continue
            candidate, hold = max(
                choices,
                key=lambda item: (
                    item[1] >= 0.65,
                    min(item[1], 3.0),
                    -abs(item[0]["timestamp"] - midpoint),
                ),
            )
            # A longer interval until the next detected change is a useful
            # stability heuristic, not a semantic guarantee. Move just inside
            # that hold while retaining the unchanged full candidate sequence.
            # ``end`` is the last decoded PTS, not the start of a hold extending
            # to the platform's rounded duration. Advancing it can leave the
            # actual video stream, even though it remains inside the chapter.
            timestamp = candidate["timestamp"] if "end" in candidate.get("reasons", ()) else min(
                right - min(0.00001, width / 1000),
                candidate["timestamp"] + min(0.35, max(0, hold) / 3),
            )
            selected.append({**candidate, "timestamp": timestamp})
        return selected

    def section(
        self,
        video_id,
        section_id,
        max_tokens=6000,
        cursor=None,
        include_frames=True,
        frame_cursor=None,
        frame_limit=4,
        frame_view="navigation",
        *,
        question="",
        visual_queries=None,
        record_delivery=True,
    ):
        from .visual_query import validate_visual_query

        question, visual_queries = validate_visual_query(question, visual_queries)
        if question and frame_cursor is not None:
            raise VidLensError("invalid_cursor", "Question selection has no frame cursor; use navigation separately.")
        validate_budget(max_tokens)
        self._validate_view(frame_view)
        if frame_cursor is not None and (
            isinstance(frame_cursor, bool) or not isinstance(frame_cursor, int) or frame_cursor < 0
        ):
            raise VidLensError("invalid_cursor", "frame_cursor must be a nonnegative integer.")
        if (
            isinstance(frame_limit, bool)
            or not isinstance(frame_limit, int)
            or not 1 <= frame_limit <= 12
        ):
            raise VidLensError("invalid_frame_count", "frame_limit must be from 1 to 12.")
        item = self.study.get_section(video_id, section_id)
        section = item["section"]
        payload = {
            "video_id": video_id,
            "revision": item["revision"],
            "section": {
                key: section[key] for key in ("id", "start", "end", "title", "boundary_reason")
            },
            "frames": [],
            "visual": {"status": "not_requested"},
            "progress": self._compact_progress(item["progress"]),
        }
        if include_frames:
            try:
                if question:
                    focused = self.service.read_frames(
                        video_id, section["start"], section["end"], max_frames=frame_limit,
                        question=question, visual_queries=visual_queries, record_delivery=False,
                    )
                    payload["frames"], payload["visual"] = focused["frames"], focused["visual"]
                    # Search diagnostics remain in read_frames; leave chapter
                    # budget for its original speech and the selected evidence.
                    payload["visual"] = {k: v for k, v in payload["visual"].items()
                                         if k not in {"query_terms", "unmatched_query_terms", "metrics", "question", "visual_queries"}}
                    for frame in payload["frames"]:
                        frame["ocr"] = self.budget.truncate(frame.get("ocr", ""), 200)
                else:
                    payload["frames"], payload["visual"] = self._frames(
                        video_id, section, frame_cursor, frame_limit, frame_view
                    )
            except VidLensError as exc:
                payload["visual"] = {"status": "unavailable", "error": exc.to_dict()}
        overhead = self.budget.count(compact_json(payload)) + 100
        if max_tokens - overhead < 768:
            raise VidLensError(
                "budget_too_small", "Increase max_tokens or reduce frame_limit for this chapter."
            )
        page = self.service.reader.read_interval(
            video_id,
            section["start"],
            section["end"],
            max_tokens=max_tokens - overhead,
            cursor=cursor,
            speech_only=True,
            record_delivery=False,
        )
        payload.update(
            evidence=page["evidence"],
            next_cursor=page["next_cursor"],
            text_page_complete=page["next_cursor"] is None,
            accounting=page["accounting"],
            truncated=page["truncated"],
        )
        payload["accounting"]["budget"] = max_tokens
        self.budget.recount(payload)
        return self.record_delivery(payload) if record_delivery else payload

    def record_delivery(self, payload, frames=None):
        """Validate the final payload before atomically fencing it against source refresh."""
        video_id, section_id = payload["video_id"], payload["section"]["id"]
        with self.service._file_lock("content", video_id):
            current = self.study.get_section(video_id, section_id)
            if current["revision"] != payload["revision"]:
                raise VidLensError(
                    "stale_revision", "The chapter changed during reading; reload its plan."
                )
            result = {**payload, "accounting": dict(payload["accounting"])}
            if frames is not None:
                failed = [frame for frame in result["frames"] if not frame.get("image_index")]
                if failed and result["visual"].get("selection") in {
                    "question_screen_text", "timeline_fallback"
                }:
                    result["visual"] = {**result["visual"], "status": "partial",
                                        "retry": "repeat_question_selection",
                                        "missing_candidate_times": [f["timestamp"] for f in failed]}
                elif failed:
                    visual = dict(result["visual"])
                    retry_points = [frame.get("candidate_cursor", 0) for frame in failed]
                    if visual.get("retry_frame_cursor") is not None:
                        retry_points.append(visual["retry_frame_cursor"])
                    retry = min(retry_points)
                    missing_times = [
                        frame.get("requested_timestamp")
                        if frame.get("requested_timestamp") is not None else frame["timestamp"]
                        for frame in failed
                    ]
                    visual.update(
                        status="partial", retry_frame_cursor=retry, next_frame_cursor=retry,
                        missing_candidate_times=sorted(set(
                            visual.get("missing_candidate_times", []) + missing_times
                        )),
                    )
                    result["visual"] = visual
            progress = self._compact_progress(current["progress"])
            # Progress can grow after delivery (e.g. 99 -> 100 cues). Check a
            # conservative upper bound before writes, including MCP image indexes.
            maximum = {
                **result,
                "accounting": dict(result["accounting"]),
                "progress": {
                    **progress,
                    "text_status": "not_delivered",
                    "visual_status": "not_delivered",
                    "text_segments_delivered": progress["text_segments"],
                },
            }
            self.budget.recount(maximum)
            self.store.delivered_text(video_id, result["evidence"])
            self.store.delivered(video_id, result["frames"] if frames is None else frames, "frame")
            result["progress"] = self._compact_progress(
                self.study.get_section(video_id, section_id)["progress"]
            )
            return self.budget.recount(result)

    @staticmethod
    def _compact_progress(progress):
        return {
            key: progress[key]
            for key in (
                "text_status",
                "text_segments",
                "text_segments_delivered",
                "visual_status",
                "note_status",
            )
        }

    def search(self, video_id, query, max_tokens=3000, limit=6):
        validate_budget(max_tokens)
        result = self.references.search(video_id, query, limit)
        # A result is a query-specific excerpt, not an entire saved chapter. A
        # large first note must not force the budget packer to discard every hit.
        terms = set(query_tokens(query))
        result["matches"] = [self._knowledge_excerpt(match, terms) for match in result["matches"]]
        total = len(result["matches"])
        packed = self.budget.finish(dict(result), max_tokens, "matches", total)
        if packed["matches"] or not total or result["matches"][0].get("match_scope") != "relation_path":
            return packed
        # The generic packer can shorten a text excerpt, but a graph result has
        # nested edges. Reduce descriptions while preserving endpoints/citations.
        first = result["matches"][0]
        for allowance in (80, 40, 20, 8, 0):
            compact = {
                **first,
                "title": self.budget.truncate(first["title"], 32),
                "relations": [
                    {**edge, "relation": self.budget.truncate(edge["relation"], allowance)}
                    for edge in first["relations"]
                ],
                "excerpt_truncated": True,
            }
            # Citations are already attached to each edge; avoid repeating them.
            compact.pop("evidence_ids", None)
            packed = self.budget.finish(
                {**result, "matches": [compact], "truncated": True}, max_tokens, "matches", total
            )
            if packed["matches"]:
                return packed
        raise VidLensError(
            "budget_too_small", "Increase max_tokens to retain the relationship endpoints and citations."
        )

    def _knowledge_excerpt(self, match, terms):
        if match["kind"] != "section_note":
            return {
                key: self.budget.truncate(value, 600)
                if key in {"excerpt", "explanation"}
                else value
                for key, value in match.items()
            }
        if match.get("match_reason") == "relation_path":
            indices = match["matched_relation_indices"]
            relations = [match["relations"][index] for index in indices]
            # Preserve each edge of a retrieved reasoning chain. Selecting only
            # the best lexical sentence can otherwise hide its bridging step.
            allowance = max(40, 600 // max(1, len(relations)))
            excerpt = [
                {**relation, "relation": self.budget.truncate(relation["relation"], allowance)}
                for relation in relations
            ]
            return {
                **{key: match[key] for key in
                   ("kind", "id", "section_id", "start", "end", "score") if key in match},
                "title": self.budget.truncate(match["title"], 120),
                "match_scope": "relation_path",
                "matched_relation_indices": indices,
                "relations": excerpt,
                "evidence_ids": list(dict.fromkeys(
                    identity for relation in relations for identity in relation["evidence_ids"]
                )),
                "excerpt_truncated": any(a != b for a, b in zip(relations, excerpt)),
            }
        choices = []
        for index, claim in enumerate(match.get("claims", [])):
            choices.append(("claim", index, claim["text"], claim["evidence_ids"]))
        for index, relation in enumerate(match.get("relations", [])):
            body = f"{relation['source']} — {relation['relation']} → {relation['target']}"
            choices.append(("relation", index, body, relation["evidence_ids"]))
        summary = ("summary", None, match.get("summary", ""), [])
        choices.append(summary)

        def score(choice):
            overlap = len(terms & set(tokenize(choice[2])))
            return overlap, int(choice[0] != "summary") if overlap else int(choice[0] == "summary")

        kind, index, body, evidence_ids = max(choices, key=score)
        compact = {
            key: match[key]
            for key in ("kind", "id", "section_id", "start", "end", "score")
            if key in match
        }
        text = self.budget.truncate(body, 600)
        compact.update(
            title=self.budget.truncate(match["title"], 120),
            text=text,
            match_scope=kind,
            evidence_ids=list(evidence_ids),
            excerpt_truncated=text != body,
        )
        if index is not None:
            compact[f"{kind}_index"] = index
        # section_id lets callers reopen the original chapter via read_section;
        # evidence_ids point to the original speech/frame records, not the note.
        return compact

    def export(self, video_id):
        from pathlib import Path

        # Read every attribution carrier from the same snapshot. Service locks
        # do not fence another process writing through Store directly.
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            video, segments, _, source_revision, revision, sections = self.study._state(db, video_id)
            notes = self.study._notes(db, video_id, source_revision)
            references = self.references._list(db, video_id, source_revision)
            frames = {
                row["timestamp"]: json.loads(row["data"])
                for row in db.execute("SELECT timestamp,data FROM frames WHERE video_id=?", (video_id,))
            }
        lookup = {cue.id: cue.start for cue in segments if cue.kind != "ocr"}
        lines = [f"# {video.title}", "", f"[视频来源]({video.url})", ""]
        shown_images, image_count = set(), 0

        def evidence_links(identities):
            links = []
            for identity in identities:
                is_frame = identity.startswith("frame:")
                stamp = float(identity[6:]) if is_frame else lookup.get(identity)
                if stamp is not None and math.isfinite(stamp):
                    label = "画面" if is_frame else "原文"
                    links.append(f"[{label} {stamp:.1f}s]({source_link(video, stamp)})")
            return " ".join(dict.fromkeys(links))

        for section in sections:
            note = notes.get(section["id"])
            title = note["title"] if note else section["title"]
            lines += [f"## {title}", "",
                      f"[跳转到视频]({source_link(video, section['start'])})", ""]
            if note:
                lines += [note["summary"], ""]
                for claim in note.get("claims", []):
                    lines.append(f"- {claim['text']} {evidence_links(claim['evidence_ids'])}")
                lines.append("")
                for relation in note.get("relations", []):
                    lines.append(
                        f"- {relation['source']} → {relation['target']}：{relation['relation']} "
                        f"{evidence_links(relation['evidence_ids'])}"
                    )
                cited = dict.fromkeys(
                    identity
                    for row in [*note.get("claims", []), *note.get("relations", [])]
                    for identity in row["evidence_ids"] if identity.startswith("frame:")
                )
                for identity in cited:
                    stamp = float(identity[6:])
                    frame = frames.get(stamp)
                    if frame is None and frames:
                        nearest = min(frames, key=lambda value: (abs(value - stamp), value))
                        if abs(nearest - stamp) <= 0.01:
                            frame = frames[nearest]
                    if frame is None:
                        continue
                    path = Path(frame["path"])
                    if not path.is_file() or path in shown_images:
                        continue
                    shown_images.add(path)
                    image_count += 1
                    lines += ["", f"![视频画面 {frame['timestamp']:.1f}s](<{path}>)",
                              "", evidence_links([identity]), ""]
            else:
                lines += ["此章节的学习笔记待整理。", ""]
            for reference in references:
                if reference["section_id"] == section["id"]:
                    lines += [reference["explanation"], "",
                              f"[{reference['title']}]({reference['url']})", ""]
        # Local media can change outside SQLite and the service's file lock.
        # Recheck the complete study identity, including transcript and file stat.
        current = self.study.plan(video_id)
        if current["revision"] != revision or current["source_revision"] != source_revision:
            raise VidLensError("stale_revision", "The study changed during export; retry.")
        return {"video_id": video_id, "revision": revision,
                "image_count": image_count, "markdown": "\n".join(lines)}

"""Durable chapter candidates and model-written notes tied to original evidence.

This module plans reading work. It does not claim to summarize or understand a
video: notes are supplied by the caller, and delivery is tracked independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from .models import Segment, Video, VidLensError
from .retrieval import source_link, tokenize
from .store import Store, frame_interval, union_intervals

_ALGORITHM = "study-v4"
_SENTENCE_END = re.compile(r"[.!?。！？][\"'”’）)\]]*\s*$")
# Navigation language is a candidate signal, not a semantic chapter label.
# Unlike punctuation, these cues survive many continuous Chinese ASR tracks.
_TOPIC_TRANSITION = re.compile(
    r"^\s*(?:(?:那么|好了?|所以|现在)[,，]?\s*)?"
    r"(?:(?:在)?(?:接下来|下面)(?:的|我们|来看|来|将|要|就|开始|分析|讨论|介绍|学习)|"
    r"(?:让我们|我们)?(?:回到|进入)(?:第|[0-9])|"
    r"(?:在|进入|来到)?(?:第[一二三四五六七八九十百0-9]+章|"
    r"[0-9]+(?:\.[0-9]+)+\s*(?:小节|节))|"
    r"(?:next|now|turning to|moving on to)\b.*\b(?:discuss|consider|look at|section|chapter|topic)\b)",
    re.IGNORECASE,
)
_ROLES = {"definition", "mechanism", "derivation", "code", "example", "experiment", "correction"}
_EPSILON = 1e-6


def _digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:24]


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _overlaps(start, end, section) -> bool:
    return start < section["end"] and end > section["start"]


def _covered(intervals, start, end):
    """Check delivery without rounding away a short undelivered interval."""
    cursor = start
    for left, right in sorted(intervals):
        if right < cursor:
            continue
        if left > cursor + _EPSILON:
            return False
        cursor = max(cursor, right)
        if cursor >= end - _EPSILON:
            return True
    return False


def _gaps(intervals, start, end):
    cursor, result = start, []
    for left, right in union_intervals(intervals):
        left, right = max(start, left), min(end, right)
        if right <= start or left >= end:
            continue
        if left > cursor:
            result.append([cursor, left])
        cursor = max(cursor, right)
    if cursor < end:
        result.append([cursor, end])
    return result


class StudyEngine:
    """One source revision can have several layouts; notes belong to exact scopes.

    All public operations use one SQLite transaction. OCR and frame additions do
    not change the source revision. Caption changes, source refreshes, duration
    changes and platform chapter changes do. A rename preserves a note's scope;
    regrouping creates new section IDs and cannot inherit unrelated notes.
    """

    def __init__(self, store: Store):
        self.store = store
        with store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS study_plans (
                    video_id TEXT PRIMARY KEY REFERENCES videos(id),
                    source_revision TEXT NOT NULL, revision TEXT NOT NULL,
                    sections TEXT NOT NULL, updated REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS study_notes (
                    video_id TEXT NOT NULL REFERENCES videos(id),
                    source_revision TEXT NOT NULL, section_id TEXT NOT NULL,
                    revision TEXT NOT NULL, note TEXT NOT NULL, updated REAL NOT NULL,
                    PRIMARY KEY(video_id, source_revision, section_id)
                );
                CREATE TABLE IF NOT EXISTS study_media_sources (
                    video_id TEXT PRIMARY KEY REFERENCES videos(id), fingerprint TEXT NOT NULL
                );
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {r["name"] for r in db.execute("PRAGMA table_info(study_plans)")}
            if "algorithm" not in columns:
                db.execute("ALTER TABLE study_plans ADD COLUMN algorithm TEXT NOT NULL DEFAULT ''")
            if "layout_policy" not in columns:
                db.execute(
                    "ALTER TABLE study_plans ADD COLUMN layout_policy TEXT NOT NULL DEFAULT 'automatic'"
                )

    def _state(self, db, video_id):
        row = db.execute("SELECT data FROM videos WHERE id=?", (video_id,)).fetchone()
        if row is None:
            raise VidLensError("video_not_found", f"Unknown video: {video_id}")
        video = Video(**json.loads(row["data"]))
        row = db.execute("SELECT * FROM study_plans WHERE video_id=?", (video_id,)).fetchone()
        local_media = self._local_media_revision(video)
        media_fingerprint = _digest(local_media)
        prior_media = db.execute(
            "SELECT fingerprint FROM study_media_sources WHERE video_id=?", (video_id,)
        ).fetchone()
        if (prior_media and prior_media["fingerprint"] != media_fingerprint) or (
            prior_media is None and row is not None and local_media is not None
        ):
            # Old captures cannot be associated with a replacement local file.
            # In pre-fingerprint caches their provenance is unknown, so discard
            # only derived visual evidence and retain unchanged spoken text.
            db.execute(
                "DELETE FROM deliveries WHERE video_id=? AND (kind='frame' OR evidence_id IN "
                "(SELECT id FROM segments WHERE video_id=? AND kind='ocr'))",
                (video_id, video_id),
            )
            db.execute("DELETE FROM frames WHERE video_id=?", (video_id,))
            db.execute(
                "DELETE FROM text_deliveries WHERE video_id=? AND evidence_id IN "
                "(SELECT id FROM segments WHERE video_id=? AND kind='ocr')", (video_id, video_id),
            )
            db.execute("DELETE FROM segments WHERE video_id=? AND kind='ocr'", (video_id,))
        if prior_media is None or prior_media["fingerprint"] != media_fingerprint:
            db.execute(
                "INSERT OR REPLACE INTO study_media_sources VALUES (?,?)",
                (video_id, media_fingerprint),
            )
        rows = db.execute(
            "SELECT * FROM segments WHERE video_id=? ORDER BY start,kind,id", (video_id,)
        ).fetchall()
        segments = self.store._decode_segments(rows)
        speech = [s for s in segments if s.kind != "ocr"]
        duration = max(
            video.duration if _finite(video.duration) else 0,
            max((s.end for s in speech), default=0),
        )
        source_data = [
            video.id,
            video.metadata.get("revision", ""),
            local_media,
            duration,
            video.chapters,
            [(s.id, s.start, s.end, s.text, s.kind, s.source, s.language) for s in speech],
        ]
        source_revision = _digest(source_data)
        # Keep a v2 source token when its raw evidence is unchanged. Layout
        # upgrades should not invalidate notes on unchanged chapter scopes.
        if row and row["source_revision"] == _digest(["study-v2", *source_data]):
            source_revision = row["source_revision"]
        source_changed = row is None or row["source_revision"] != source_revision
        layout_upgrade = row is not None and row["algorithm"] != _ALGORITHM
        if not source_changed and layout_upgrade:
            previous = json.loads(row["sections"])
            has_notes = db.execute(
                "SELECT 1 FROM study_notes WHERE video_id=? AND source_revision=? LIMIT 1",
                (video_id, source_revision),
            ).fetchone()
            if has_notes or any(s.get("title_source") == "ai" for s in previous):
                # A better automatic splitter must not detach an authored note
                # from its scope. Keep the exact IDs and revision; an explicit
                # revise_sections call can adopt different boundaries later.
                db.execute(
                    "UPDATE study_plans SET algorithm=?,layout_policy='preserved_on_upgrade' "
                    "WHERE video_id=?",
                    (_ALGORITHM, video_id),
                )
                return video, segments, duration, source_revision, row["revision"], previous
        if source_changed or layout_upgrade:
            sections = self._generate(video, speech, duration, source_revision)
            revision = self._write_plan(db, video_id, source_revision, sections)
        else:
            sections, revision = json.loads(row["sections"]), row["revision"]
        return video, segments, duration, source_revision, revision, sections

    @staticmethod
    def _local_media_revision(video):
        local = video.metadata.get("local_media_path")
        if not local:
            return None
        path = Path(local)
        try:
            stat = path.stat()
            return [str(path), stat.st_size, stat.st_mtime_ns]
        except OSError:
            # Losing a media file invalidates its visual provenance, but should
            # not prevent reading already imported transcript text.
            return [str(path), "missing"]

    @staticmethod
    def _write_plan(db, video_id, source_revision, sections, layout_policy="automatic"):
        revision = _digest([source_revision, sections])
        db.execute(
            "INSERT INTO study_plans(video_id,source_revision,revision,sections,updated,algorithm,layout_policy) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(video_id) DO UPDATE SET "
            "source_revision=excluded.source_revision,revision=excluded.revision,"
            "sections=excluded.sections,updated=excluded.updated,algorithm=excluded.algorithm,"
            "layout_policy=excluded.layout_policy",
            (
                video_id,
                source_revision,
                revision,
                json.dumps(sections, ensure_ascii=False),
                time.time(),
                _ALGORITHM,
                layout_policy,
            ),
        )
        return revision

    @staticmethod
    def _section(source_revision, start, end, title, title_source, reason, platform_title=""):
        return {
            "id": "section:" + _digest([source_revision, start, end])[:16],
            "start": start,
            "end": end,
            "title": title,
            "title_source": title_source,
            "platform_title": platform_title,
            "boundary_reason": reason,
        }

    @staticmethod
    def _candidate_boundaries(speech: list[Segment], duration):
        """Prefer sentence ends; rolling captions may overlap a time boundary.

        Reading always returns complete intersecting cues. A capacity boundary
        consequently shares a little context between adjacent chapters instead
        of dropping or slicing any original caption text.
        """
        tracks = defaultdict(list)
        for s in speech:
            tracks[s.kind, s.source, s.language].append(s)
        primary = max(tracks.values(), key=lambda cues: sum(len(s.text) for s in cues), default=[])
        starts, maximum_ends = [], []
        for s in speech:
            starts.append(s.start)
            maximum_ends.append(max(s.end, maximum_ends[-1] if maximum_ends else 0))

        boundaries = {0.0: {"reason": "start", "pause": 0, "shift": 0}}
        for i, cue in enumerate(primary[:-1]):
            following = primary[i + 1]
            pause = following.start - cue.end
            continuation = pause < 2 and not _SENTENCE_END.search(cue.text)
            point = (
                following.start
                if continuation
                else cue.end
                if pause < 2
                else (cue.end + following.start) / 2
            )
            if not 0 < point < duration:
                continue
            before = set(tokenize(" ".join(s.text for s in primary[max(0, i - 3) : i + 1])))
            after = set(tokenize(" ".join(s.text for s in primary[i + 1 : i + 5])))
            shift = 1 - len(before & after) / max(1, len(before | after))
            # Repeated "next we calculate ..." instructions inside the same
            # derivation are not chapters. Require some changed context too.
            transition = bool(_TOPIC_TRANSITION.search(following.text)) and shift >= 0.35
            # Short, unrelated phrases naturally have little token overlap.
            # Require context on both sides and a stronger change when there
            # is no sentence end, while keeping every original cue intact.
            topic_change = min(len(before), len(after)) >= 5 and shift >= (
                0.9 if continuation else 0.8
            )
            reason = (
                "pause" if pause >= 4
                else "topic_transition" if transition
                else "topic_change" if topic_change
                else "capacity_continuation" if continuation
                else "sentence"
            )
            boundaries[point] = {
                "reason": reason,
                "pause": max(0, pause),
                "shift": shift,
                "transition": transition,
            }
        for point in range(300, math.ceil(duration), 300):
            index = bisect_right(starts, point + 5) - 1
            if index < 0 or maximum_ends[index] <= point - 5:
                boundaries[float(point)] = {
                    "reason": "untranscribed_interval",
                    "pause": 0,
                    "shift": 0,
                }
        boundaries[float(duration)] = {"reason": "end", "pause": 0, "shift": 0}

        return boundaries

    def _generate(self, video, speech, duration, source_revision):
        if duration <= 0:
            return []
        boundaries = self._candidate_boundaries(speech, duration)
        points = sorted(boundaries)
        preferred = [p for p in points if boundaries[p]["reason"] != "capacity_continuation"]
        anchors = {0.0: {"title": "", "reason": "start"}}
        for chapter in video.chapters:
            if not isinstance(chapter, dict):
                continue
            raw = chapter.get("start_time", chapter.get("start", 0))
            if not _finite(raw) or not 0 <= raw < duration:
                continue
            point = float(raw)
            title = str(chapter.get("title", "")).strip()
            old_title = anchors.get(point, {}).get("title", "")
            anchors[point] = {
                "title": " / ".join(t for t in (old_title, title) if t),
                "reason": "platform",
            }
        anchors[float(duration)] = {"title": "", "reason": "end"}
        chapter_points = sorted(anchors)
        speech_starts = [s.start for s in speech]
        prefix = [0]
        for cue in speech:
            prefix.append(prefix[-1] + len(cue.text))
        by_end = sorted(speech, key=lambda s: s.end)
        speech_ends, end_prefix = [s.end for s in by_end], [0]
        for cue in by_end:
            end_prefix.append(end_prefix[-1] + len(cue.text))

        def chars(start, end):
            return (
                prefix[bisect_left(speech_starts, end)]
                - end_prefix[bisect_right(speech_ends, start)]
            )

        result = []
        for begin, end in pairwise(chapter_points):
            start, reason, part = begin, anchors[begin]["reason"], 1
            platform_title = anchors[begin]["title"]
            candidates = [p for p in points if begin < p < end]
            for point in candidates + [end]:
                count, elapsed = chars(start, point), point - start
                remaining = chars(point, end)
                info = boundaries.get(point, {})
                capacity = count >= 6000 or elapsed >= 420
                if capacity and info.get("reason") == "capacity_continuation":
                    index = bisect_right(preferred, point)
                    following = preferred[index] if index < len(preferred) else end
                    if (
                        following <= end
                        and following - point <= 45
                        and chars(start, following) <= 9000
                    ):
                        capacity = False
                natural = (
                    elapsed >= 90
                    and count >= (400 if info.get("transition") else 1200)
                    and remaining >= 500
                    and info.get("reason") in {"pause", "topic_transition", "topic_change"}
                )
                if point != end and not (capacity or natural):
                    continue
                title = platform_title or f"Candidate section {len(result) + 1}"
                if platform_title and (part > 1 or point != end):
                    title += f" · {part}"
                section = self._section(
                    source_revision,
                    start,
                    point,
                    title,
                    "platform" if platform_title else "candidate",
                    reason,
                    platform_title,
                )
                section["text_characters"] = count
                section["exceeds_preferred_capacity"] = count > 9000
                result.append(section)
                start, part = point, part + 1
                reason = (
                    info.get("reason", "capacity")
                    if natural or info.get("reason") == "capacity_continuation"
                    else "capacity"
                )
        return result

    @staticmethod
    def _notes(db, video_id, source_revision):
        rows = db.execute(
            "SELECT * FROM study_notes WHERE video_id=? AND source_revision=?",
            (video_id, source_revision),
        ).fetchall()
        return {
            r["section_id"]: {
                **json.loads(r["note"]),
                "saved_revision": r["revision"],
                "updated": r["updated"],
            }
            for r in rows
        }

    @staticmethod
    def _deliveries(db, video_id):
        return [
            dict(r)
            for r in db.execute(
                "SELECT kind,start,end,evidence_id FROM deliveries WHERE video_id=?", (video_id,)
            )
        ]

    @staticmethod
    def _progress(section, segments, deliveries, note):
        speech = [s for s in segments if s.kind != "ocr" and _overlaps(s.start, s.end, section)]
        matching = [d for d in deliveries if _overlaps(d["start"], d["end"], section)]
        delivered = defaultdict(list)
        for d in matching:
            if d["kind"] == "text" and d["evidence_id"]:
                delivered[d["evidence_id"]].append((d["start"], d["end"]))
        read_ids = {s.id for s in speech if _covered(delivered[s.id], s.start, s.end)}
        frames = sorted(
            {
                d["start"]
                for d in deliveries
                if d["kind"] == "frame" and (
                    section["start"] <= d["start"] < section["end"]
                    or _overlaps(d["start"], d["end"], section)
                )
            }
        )
        evidence_linked = bool(note and (note.get("claims") or note.get("relations")))
        text_complete = bool(speech) and len(read_ids) == len(speech)
        return {
            "text_status": "no_transcript"
            if not speech
            else "delivered"
            if text_complete
            else "pending",
            "text_segments": len(speech),
            "text_segments_delivered": len(read_ids),
            "pending_evidence_ids": [s.id for s in speech if s.id not in read_ids][:12],
            "pending_evidence_ids_truncated": len(speech) - len(read_ids) > 12,
            "text_delivered_intervals": union_intervals(
                [
                    (max(section["start"], d["start"]), min(section["end"], d["end"]))
                    for d in matching
                    if d["kind"] == "text" and d["evidence_id"] in read_ids
                ]
            ),
            "visual_status": "sampled" if frames else "not_delivered",
            "frame_timestamps_delivered": frames,
            "note_status": "evidence_linked" if evidence_linked else "draft" if note else "missing",
            "text_reviewed": text_complete and evidence_linked,
        }

    def _describe_plan(self, db, state):
        video, segments, duration, source_revision, revision, sections = state
        notes = self._notes(db, video.id, source_revision)
        deliveries = self._deliveries(db, video.id)
        rendered = []
        for section in sections:
            note = notes.get(section["id"])
            item = dict(section)
            item["candidate_title"] = section["title"]
            if note:
                item.update(title=note["title"], title_source="ai_note")
            item["progress"] = self._progress(section, segments, deliveries, note)
            item["url"] = source_link(video, section["start"])
            rendered.append(item)
        speech = [s for s in segments if s.kind != "ocr"]
        layout = db.execute(
            "SELECT layout_policy FROM study_plans WHERE video_id=?", (video.id,)
        ).fetchone()
        return {
            "video_id": video.id,
            "revision": revision,
            "source_revision": source_revision,
            "algorithm": _ALGORITHM,
            "layout_policy": layout["layout_policy"],
            "duration": duration,
            "method": "candidate chapters from platform markers, sentence boundaries, pauses and lexical changes; not a summary",
            "sections": rendered,
            "gaps": {
                "transcript_uncovered_intervals": _gaps(
                    [(s.start, s.end) for s in speech], 0, duration
                ),
                "unknown_duration": duration <= 0,
            },
            "progress": {
                "sections": len(sections),
                "text_delivered_sections": sum(
                    s["progress"]["text_status"] == "delivered" for s in rendered
                ),
                "notes_saved": sum(s["id"] in notes for s in sections),
                "text_reviewed_sections": sum(s["progress"]["text_reviewed"] for s in rendered),
                "visual_sampled_sections": sum(
                    s["progress"]["visual_status"] == "sampled" for s in rendered
                ),
            },
            "note_schema": {
                "title": "string",
                "summary": "string",
                "concepts": ["string"],
                "claims": [
                    {"text": "string", "evidence_ids": ["original cue ID or frame:<timestamp>"]}
                ],
                "relations": [
                    {
                        "source": "concept",
                        "target": "concept",
                        "relation": "string",
                        "evidence_ids": ["evidence ID"],
                    }
                ],
                "roles": sorted(_ROLES),
            },
        }

    def plan(self, video_id: str) -> dict:
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return self._describe_plan(db, self._state(db, video_id))

    def notes(self, video_id: str) -> dict:
        """Read all currently valid chapter notes in one snapshot, without evidence bodies."""
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _, _, _, source_revision, revision, sections = self._state(db, video_id)
            notes = self._notes(db, video_id, source_revision)
            return {
                "video_id": video_id,
                "revision": revision,
                "source_revision": source_revision,
                "notes": [
                    {"section_id": s["id"], "start": s["start"], "end": s["end"], **notes[s["id"]]}
                    for s in sections
                    if s["id"] in notes
                ],
            }

    @staticmethod
    def _find_section(sections, section_id):
        section = next((s for s in sections if s["id"] == section_id), None)
        if section is None:
            raise VidLensError(
                "section_not_found",
                "Unknown or obsolete section ID.",
                "Get the current study plan.",
            )
        return section

    @staticmethod
    def _section_frames(db, video_id, section):
        frames = []
        for row in db.execute(
            "SELECT timestamp,data FROM frames WHERE video_id=? AND timestamp<? ORDER BY timestamp",
            (video_id, section["end"]),
        ):
            frame = json.loads(row["data"])
            start, end = frame_interval(frame)
            if start >= section["start"] or _overlaps(start, end, section):
                frames.append(frame | {"id": f"frame:{row['timestamp']}"})
        return frames

    def get_section(self, video_id: str, section_id: str) -> dict:
        """Return stored evidence; callers must record delivery after applying budgets."""
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = self._state(db, video_id)
            video, segments, _, source_revision, revision, sections = state
            section = self._find_section(sections, section_id)
            note = self._notes(db, video_id, source_revision).get(section_id)
            frames = self._section_frames(db, video_id, section)
            display = dict(section)
            display["candidate_title"] = section["title"]
            if note:
                display.update(title=note["title"], title_source="ai_note")
            return {
                "video_id": video_id,
                "revision": revision,
                "source_revision": source_revision,
                "section": display,
                "evidence": [
                    {
                        "id": s.id,
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "kind": s.kind,
                        "source": s.source,
                        "url": source_link(video, s.start),
                    }
                    for s in segments
                    if _overlaps(s.start, s.end, section)
                ],
                "frames": frames,
                "note": note,
                "progress": self._progress(section, segments, self._deliveries(db, video_id), note),
            }

    @staticmethod
    def _assert_revision(expected, actual):
        if not isinstance(expected, str) or expected != actual:
            raise VidLensError(
                "stale_study_revision",
                "The study source or chapter layout has changed.",
                "Get the current study plan and re-read changed sections.",
            )

    def _validate_note(self, db, video_id, section, segments, note):
        if not isinstance(note, dict):
            raise VidLensError("invalid_note", "A note must be an object.")

        def string(value, name):
            if not isinstance(value, str) or not value.strip():
                raise VidLensError("invalid_note", f"{name} must be a nonempty string.")
            return value.strip()

        def array(value, name):
            if not isinstance(value, list):
                raise VidLensError("invalid_note", f"{name} must be a list.")
            return value

        allowed = {s.id for s in segments if s.kind != "ocr" and _overlaps(s.start, s.end, section)}
        frame_times = [frame["timestamp"] for frame in self._section_frames(db, video_id, section)]

        def references(value):
            refs = array(value, "evidence_ids")
            if not refs:
                raise VidLensError(
                    "invalid_note", "Every claim and relation needs original evidence IDs."
                )
            output = []
            for ref in refs:
                if not isinstance(ref, str):
                    raise VidLensError(
                        "invalid_evidence_reference", "Evidence IDs must be strings."
                    )
                valid = ref in allowed
                if ref.startswith("frame:"):
                    try:
                        stamp = float(ref[6:])
                        valid = math.isfinite(stamp) and any(
                            abs(stamp - t) < _EPSILON for t in frame_times
                        )
                    except ValueError:
                        valid = False
                if not valid:
                    raise VidLensError(
                        "invalid_evidence_reference",
                        f"Unknown or out-of-section evidence: {ref}",
                        "Use this section's original subtitle/ASR IDs or stored frame:<timestamp> IDs.",
                    )
                if ref not in output:
                    output.append(ref)
            return output

        cleaned = {
            "title": string(note.get("title"), "title"),
            "summary": string(note.get("summary"), "summary"),
            "concepts": [string(v, "concept") for v in array(note.get("concepts", []), "concepts")],
            "claims": [],
            "relations": [],
            "roles": [string(v, "role") for v in array(note.get("roles", []), "roles")],
        }
        # Roles are extensible: this is a general video reader, not a technical-only ontology.
        for key, fields in (("claims", ("text",)), ("relations", ("source", "target", "relation"))):
            for item in array(note.get(key, []), key):
                if not isinstance(item, dict):
                    raise VidLensError("invalid_note", f"Each {key} entry must be an object.")
                entry = {field: string(item.get(field), field) for field in fields}
                entry["evidence_ids"] = references(item.get("evidence_ids"))
                if "roles" in item:
                    entry["roles"] = [string(v, "role") for v in array(item["roles"], "roles")]
                cleaned[key].append(entry)
        return cleaned

    def save_note(self, video_id: str, section_id: str, revision: str, note: dict) -> dict:
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = self._state(db, video_id)
            _, segments, _, source_revision, current, sections = state
            self._assert_revision(revision, current)
            section = self._find_section(sections, section_id)
            cleaned = self._validate_note(db, video_id, section, segments, note)
            db.execute(
                "INSERT INTO study_notes VALUES (?,?,?,?,?,?) ON CONFLICT(video_id,source_revision,section_id) "
                "DO UPDATE SET revision=excluded.revision,note=excluded.note,updated=excluded.updated",
                (
                    video_id,
                    source_revision,
                    section_id,
                    current,
                    json.dumps(cleaned, ensure_ascii=False),
                    time.time(),
                ),
            )
            return {
                "video_id": video_id,
                "section_id": section_id,
                "revision": current,
                "note": cleaned,
                "progress": self._progress(
                    section, segments, self._deliveries(db, video_id), cleaned
                ),
            }

    def revise_sections(self, video_id: str, revision: str, sections: list[dict]) -> dict:
        """Replace the layout with a continuous, cue-preserving timeline partition."""
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = self._state(db, video_id)
            video, segments, duration, source_revision, current, previous = state
            self._assert_revision(revision, current)
            if not isinstance(sections, list) or not sections:
                raise VidLensError(
                    "invalid_sections", "Provide a nonempty list covering the video timeline."
                )
            speech = [s for s in segments if s.kind != "ocr"]
            cleaned, cursor = [], 0.0
            for item in sections:
                if not isinstance(item, dict):
                    raise VidLensError("invalid_sections", "Each section must be an object.")
                start, end, title = item.get("start"), item.get("end"), item.get("title")
                if (
                    not all(_finite(v) for v in (start, end))
                    or abs(start - cursor) > _EPSILON
                    or end <= start
                    or end > duration + _EPSILON
                ):
                    raise VidLensError(
                        "invalid_sections",
                        "Sections must be ordered, contiguous and cover 0 through duration.",
                    )
                if not isinstance(title, str) or not title.strip():
                    raise VidLensError("invalid_sections", "Each section needs a nonempty title.")
                section = self._section(
                    source_revision, cursor, float(end), title.strip(), "ai", "ai_revision"
                )
                old = next((s for s in previous if s["id"] == section["id"]), None)
                if old:
                    section["platform_title"] = old["platform_title"]
                section["text_characters"] = sum(
                    len(s.text) for s in speech if _overlaps(s.start, s.end, section)
                )
                section["exceeds_preferred_capacity"] = section["text_characters"] > 9000
                cleaned.append(section)
                cursor = float(end)
            if abs(cursor - duration) > _EPSILON:
                raise VidLensError(
                    "invalid_sections", "Sections must cover the entire video duration."
                )
            new_revision = self._write_plan(
                db, video_id, source_revision, cleaned, layout_policy="authored"
            )
            # An explicit rename should also rename a retained note. Otherwise
            # its title would hide the user's requested chapter title in plans.
            for section in cleaned:
                row = db.execute(
                    "SELECT note FROM study_notes WHERE video_id=? AND source_revision=? AND section_id=?",
                    (video_id, source_revision, section["id"]),
                ).fetchone()
                if row:
                    note = json.loads(row["note"])
                    if note["title"] != section["title"]:
                        note["title"] = section["title"]
                        db.execute(
                            "UPDATE study_notes SET note=?,revision=?,updated=? "
                            "WHERE video_id=? AND source_revision=? AND section_id=?",
                            (
                                json.dumps(note, ensure_ascii=False),
                                new_revision,
                                time.time(),
                                video_id,
                                source_revision,
                                section["id"],
                            ),
                        )
            return self._describe_plan(
                db, (video, segments, duration, source_revision, new_revision, cleaned)
            )

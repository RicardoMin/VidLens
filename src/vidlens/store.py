"""Transactional local evidence store, independent of any model conversation."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from .budget import compact_json
from .models import Frame, Segment, Video, VidLensError


def segment_id(video_id: str, segment: Segment) -> str:
    value = f"{video_id}|{segment.kind}|{segment.start:.3f}|{segment.end:.3f}|{segment.text}"
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _segment_key(segment) -> tuple:
    return (float(segment.start), float(segment.end), segment.text, segment.kind,
            segment.source, segment.language)


def frame_interval(frame: dict) -> tuple[float, float]:
    """Use a decoder-confirmed display interval, or a legacy point capture."""
    start = frame["timestamp"]
    end = frame.get("visible_until")
    if not isinstance(end, (int, float)) or isinstance(end, bool) or not math.isfinite(end) or end <= start:
        end = start
    return start, end


def frame_source_revision(video: Video) -> str:
    """Shared identity for decoded pixels, independent of speech/learning changes."""
    identity = [video.id, video.metadata.get("revision", ""), video.duration]
    if local := video.metadata.get("local_media_path"):
        path = Path(local)
        stat = path.stat()
        identity.extend([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(compact_json(identity).encode()).hexdigest()[:24]


def _retained_frame_quality(data: dict) -> tuple[int, int, bool] | None:
    """Inspect actual full-frame pixels, including a display image's larger source."""
    if data.get("crop_box"):
        return None
    path = Path(data["path"])
    def size(image):
        width, height = image.size
        return (height, width) if image.getexif().get(274) in {5, 6, 7, 8} else (width, height)

    try:
        with Image.open(path) as image:
            display_size = size(image)
        try:
            info = json.loads(path.with_suffix(".json").read_text())
        except (OSError, ValueError):
            info = {}
        value = info.get("source") if isinstance(info, dict) else None
        source = Path(value) if isinstance(value, str) and Path(value).is_file() else path
        with Image.open(source) as image:
            width, height = size(image)
            lossless = image.format == "PNG"
        if not math.isclose(width / height, display_size[0] / display_size[1], rel_tol=.01):
            return None
        return width, height, lossless
    except (OSError, ValueError, KeyError):
        return None


def _keep_retained_frame(previous: dict, incoming: dict) -> bool:
    old, new = _retained_frame_quality(previous), _retained_frame_quality(incoming)
    if old is None:
        return False
    if new is None:
        return True
    if not math.isclose(old[0] / old[1], new[0] / new[1], rel_tol=.01):
        return False
    # A larger area alone does not establish more detail in both dimensions.
    return old[0] >= new[0] and old[1] >= new[1] and (
        old[:2] != new[:2] or (old[2] and not new[2])
    )


def _matching_frame_geometry(previous: dict, incoming: dict) -> bool:
    values = [previous.get("width"), previous.get("height"),
              incoming.get("width"), incoming.get("height")]
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in values):
        return False
    return math.isclose(values[0] / values[1], values[2] / values[3], rel_tol=.01)


class Store:
    def __init__(self, home: Path):
        self.home = home.resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / "evidence.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY, url TEXT NOT NULL, data TEXT NOT NULL,
                    warnings TEXT NOT NULL, updated REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS videos_url ON videos(url);
                CREATE TABLE IF NOT EXISTS segments (
                    id TEXT PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(id),
                    start REAL NOT NULL, end REAL NOT NULL, text TEXT NOT NULL,
                    kind TEXT NOT NULL, source TEXT NOT NULL, language TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS segments_time ON segments(video_id, start, end);
                CREATE TABLE IF NOT EXISTS frames (
                    video_id TEXT NOT NULL REFERENCES videos(id), timestamp REAL NOT NULL,
                    data TEXT NOT NULL, PRIMARY KEY(video_id, timestamp)
                );
                CREATE TABLE IF NOT EXISTS evidence_versions (
                    video_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS url_aliases (
                    digest TEXT PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(id)
                );
                CREATE TABLE IF NOT EXISTS media_assets (
                    id TEXT NOT NULL, revision TEXT NOT NULL, profile TEXT NOT NULL,
                    start REAL NOT NULL, end REAL NOT NULL, path TEXT PRIMARY KEY
                );
                CREATE INDEX IF NOT EXISTS assets_time ON media_assets(id,revision,profile,start,end);
                CREATE TABLE IF NOT EXISTS deliveries (
                    video_id TEXT NOT NULL REFERENCES videos(id), kind TEXT NOT NULL,
                    start REAL NOT NULL, end REAL NOT NULL, evidence_id TEXT NOT NULL,
                    created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS deliveries_evidence ON deliveries(video_id,kind,start,end,evidence_id);
                CREATE TABLE IF NOT EXISTS text_deliveries (
                    video_id TEXT NOT NULL REFERENCES videos(id), evidence_id TEXT NOT NULL,
                    start_char INTEGER NOT NULL, end_char INTEGER NOT NULL,
                    PRIMARY KEY(video_id,evidence_id,start_char,end_char)
                );
            """)

            # Invalidate reader snapshots even when another CLI or older MCP process writes.
            for operation, row in (("INSERT", "NEW"), ("UPDATE", "NEW"), ("DELETE", "OLD")):
                db.execute(f"""
                    CREATE TRIGGER IF NOT EXISTS segments_epoch_{operation} AFTER {operation} ON segments
                    BEGIN
                        INSERT INTO evidence_versions VALUES ({row}.video_id, 1)
                        ON CONFLICT(video_id) DO UPDATE SET epoch=epoch+1;
                    END
                """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def save(
        self,
        video: Video,
        segments: list[Segment],
        warnings: list[str],
        *,
        replace=True,
        invalidate_derived=False,
        aliases=(),
        replace_asr_range: tuple[float, float, str | None] | None = None,
    ):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior_ids = {}
            if replace and replace_asr_range is None and not invalidate_derived:
                prior_ids = {
                    _segment_key(s): s.id for s in self._decode_segments(db.execute(
                        "SELECT * FROM segments WHERE video_id=?", (video.id,),
                    ))
                }
            db.execute(
                "INSERT INTO videos VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "url=excluded.url, data=excluded.data, warnings=excluded.warnings, updated=excluded.updated",
                (
                    video.id,
                    video.url,
                    json.dumps(video.to_dict(), ensure_ascii=False),
                    json.dumps(warnings, ensure_ascii=False),
                    time.time(),
                ),
            )
            for alias in aliases:
                db.execute(
                    "INSERT OR REPLACE INTO url_aliases VALUES (?,?)",
                    (hashlib.sha256(alias.encode()).hexdigest(), video.id),
                )
            if invalidate_derived:
                db.execute("DELETE FROM media_assets WHERE id=?", (video.id,))
                db.execute("DELETE FROM segments WHERE video_id=?", (video.id,))
                db.execute("DELETE FROM frames WHERE video_id=?", (video.id,))
                db.execute("DELETE FROM deliveries WHERE video_id=?", (video.id,))
                db.execute("DELETE FROM text_deliveries WHERE video_id=?", (video.id,))
            elif replace and replace_asr_range is None:
                db.execute("DELETE FROM segments WHERE video_id=? AND kind != 'ocr'", (video.id,))
            if replace_asr_range is not None:
                start, end, language = replace_asr_range
                predicate = (
                    "video_id=? AND kind='transcript' "
                    "AND (source LIKE 'whisper:%' OR source LIKE 'mlx-whisper:%') "
                    "AND language=? AND start>=? AND end<=?"
                )
                params = (video.id, language or "", start, end)
                # Delete attribution before deleting its source, in this same
                # transaction. A later cue with the same text ID must not inherit
                # pages read from an earlier recognition run.
                for table in ("deliveries", "text_deliveries"):
                    db.execute(
                        f"DELETE FROM {table} WHERE video_id=? AND evidence_id IN "
                        f"(SELECT id FROM segments WHERE {predicate})",
                        (video.id, *params),
                    )
                # Older complete-read responses did not carry raw IDs. Their
                # aggregate attribution cannot survive replacement of any cue.
                db.execute(
                    "DELETE FROM deliveries WHERE video_id=? AND kind IN ('text','excerpt') "
                    "AND NOT EXISTS (SELECT 1 FROM segments WHERE id=evidence_id) "
                    f"AND EXISTS (SELECT 1 FROM segments s WHERE {predicate} "
                    "AND s.start < deliveries.end AND s.end > deliveries.start)",
                    (video.id, *params),
                )
                db.execute(
                    f"DELETE FROM segments WHERE {predicate}", params,
                )
            self._add_segments(db, video.id, segments, prior_ids)

    def _add_segments(self, db, video_id: str, segments: list[Segment], prior_ids=None):
        prior_ids = prior_ids or {}
        reserved = set(prior_ids.values())
        for s in segments:
            if s.end < s.start or s.start < 0 or not s.text.strip():
                continue
            key = _segment_key(s)
            existing = db.execute(
                "SELECT id FROM segments WHERE video_id=? AND start=? AND end=? "
                "AND text=? AND kind=? AND source=? AND language=?", (video_id, *key),
            ).fetchone()
            if existing:
                s.id = existing["id"]
            elif key in prior_ids:
                s.id = prior_ids[key]
            else:
                s.id = segment_id(video_id, s)
                if s.id in reserved or db.execute("SELECT 1 FROM segments WHERE id=?", (s.id,)).fetchone():
                    # Legacy IDs omitted track provenance and rounded time to
                    # milliseconds. Preserve them unless two distinct cues
                    # collide; qualify only the additional cue's identity.
                    value = json.dumps([video_id, *key], ensure_ascii=False, separators=(",", ":"))
                    s.id = hashlib.sha256(value.encode()).hexdigest()[:16]
            db.execute(
                "INSERT OR REPLACE INTO segments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (s.id, video_id, s.start, s.end, s.text, s.kind, s.source, s.language),
            )

    def add_segments(self, video_id: str, segments: list[Segment]):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._add_segments(db, video_id, segments)

    def get(self, video_id: str) -> Video:
        with self.connect() as db:
            row = db.execute("SELECT data FROM videos WHERE id=?", (video_id,)).fetchone()
        if not row:
            raise VidLensError(
                "video_not_found", f"Unknown video: {video_id}", "Open or import the video first."
            )
        return Video(**json.loads(row["data"]))

    def remember_alias(self, alias, video_id):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO url_aliases VALUES (?,?)",
                (hashlib.sha256(alias.encode()).hexdigest(), video_id),
            )

    def by_url(self, url: str) -> Video | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT data FROM videos WHERE url=? ORDER BY updated DESC LIMIT 1", (url,)
            ).fetchone()
            if not row:
                row = db.execute(
                    "SELECT data FROM videos JOIN url_aliases ON video_id=id WHERE digest=?",
                    (hashlib.sha256(url.encode()).hexdigest(),),
                ).fetchone()
        return Video(**json.loads(row["data"])) if row else None

    def warnings(self, video_id: str) -> list[str]:
        with self.connect() as db:
            row = db.execute("SELECT warnings FROM videos WHERE id=?", (video_id,)).fetchone()
        return json.loads(row["warnings"]) if row else []

    def segments(
        self, video_id: str, start: float | None = None, end: float | None = None
    ) -> list[Segment]:
        sql, args = "SELECT * FROM segments WHERE video_id=?", [video_id]
        if start is not None:
            sql += " AND end>?"
            args.append(start)
        if end is not None:
            sql += " AND start<?"
            args.append(end)
        with self.connect() as db:
            rows = db.execute(sql + " ORDER BY start, kind, id", args).fetchall()
        return self._decode_segments(rows)

    @staticmethod
    def _decode_segments(rows):
        return [
            Segment(
                id=r["id"],
                start=r["start"],
                end=r["end"],
                text=r["text"],
                kind=r["kind"],
                source=r["source"],
                language=r["language"],
            )
            for r in rows
        ]

    def snapshot(self, video_id, previous=None):
        """Read metadata, warnings and evidence from one SQLite snapshot on a miss."""
        with self.connect() as db:
            db.execute("BEGIN")
            row = db.execute(
                """
                SELECT v.data,v.warnings,v.updated,COALESCE(e.epoch,0) AS epoch
                FROM videos v LEFT JOIN evidence_versions e ON e.video_id=v.id
                WHERE v.id=?
            """,
                (video_id,),
            ).fetchone()
            if row is None:
                raise VidLensError(
                    "video_not_found", f"Unknown video: {video_id}", "Open or import it first."
                )
            version = (video_id, row["updated"], row["epoch"])
            if version == previous:
                return None
            segments = db.execute(
                "SELECT * FROM segments WHERE video_id=? ORDER BY start,kind,id", (video_id,)
            ).fetchall()
            return (
                version,
                Video(**json.loads(row["data"])),
                self._decode_segments(segments),
                json.loads(row["warnings"]),
            )

    def save_frames(self, video_id: str, frames: list[Frame]):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM videos WHERE id=?", (video_id,)).fetchone()
            if row is None:
                raise VidLensError("video_not_found", f"Unknown video: {video_id}")
            source_identity = frame_source_revision(Video(**json.loads(row["data"])))
            for frame in frames:
                data = frame.to_dict(include_layout=True)
                previous = db.execute(
                    "SELECT data FROM frames WHERE video_id=? AND timestamp=?",
                    (video_id, frame.timestamp),
                ).fetchone()
                old = json.loads(previous["data"]) if previous else None
                if old and old.get("_source_identity") == source_identity:
                    if _keep_retained_frame(old, data):
                        incoming = data
                        data = dict(old)
                        for key in ("selection", "requested_timestamp"):
                            data[key] = incoming.get(key)
                        if frame_interval(incoming)[1] > frame.timestamp:
                            data["visible_until"] = incoming["visible_until"]
                        if not data.get("ocr") and incoming.get("ocr"):
                            data["ocr"] = incoming["ocr"]
                            data["ocr_regions"] = incoming.get("ocr_regions", [])
                    elif (not data.get("ocr") and old.get("ocr")
                          and _matching_frame_geometry(old, data)):
                        data["ocr"] = old["ocr"]
                        data["ocr_regions"] = old.get("ocr_regions", [])
                    if frame_interval(data)[1] == frame.timestamp:
                        known_end = frame_interval(old)[1]
                        if known_end > frame.timestamp:
                            data["visible_until"] = known_end
                data["_source_identity"] = source_identity
                db.execute(
                    "INSERT OR REPLACE INTO frames VALUES (?, ?, ?)",
                    (video_id, frame.timestamp, json.dumps(data, ensure_ascii=False)),
                )
                db.execute(
                    "DELETE FROM segments WHERE video_id=? AND kind='ocr' AND start=?",
                    (video_id, frame.timestamp),
                )
                if data["ocr"].strip():
                    self._add_segments(
                        db,
                        video_id,
                        [
                            Segment(
                                frame.timestamp,
                                frame.timestamp + 0.001,
                                data["ocr"],
                                kind="ocr",
                                source="local_ocr",
                                language="mixed",
                            )
                        ],
                    )

    def frame_at(self, video_id: str, timestamp: float) -> Frame | None:
        with self.connect() as db:
            row = db.execute(
                """
                SELECT data FROM frames WHERE video_id=? AND timestamp BETWEEN ? AND ?
                ORDER BY ABS(timestamp-?), timestamp LIMIT 1
            """,
                (video_id, timestamp - 0.01, timestamp + 0.01, timestamp),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row["data"])
        data.pop("_source_identity", None)
        data["path"] = Path(data["path"])
        return Frame(**data) if data["path"].is_file() else None

    def delivered(self, video_id: str, items: list[dict], kind: str = "text"):
        with self.connect() as db:
            for item in items:
                start = item.get("start", item.get("timestamp", 0))
                end = item.get("end", item.get("timestamp", 0))
                if kind == "frame" and "timestamp" in item:
                    start, end = frame_interval(item)
                    stored = db.execute(
                        "SELECT data FROM frames WHERE video_id=? AND timestamp=?",
                        (video_id, item["timestamp"]),
                    ).fetchone()
                    if stored:
                        start, end = frame_interval(json.loads(stored["data"]))
                db.execute(
                    """INSERT INTO deliveries SELECT ?1, ?2, ?3, ?4, ?5, ?6
                       WHERE NOT EXISTS (SELECT 1 FROM deliveries WHERE video_id=?1 AND kind=?2
                       AND start=?3 AND end=?4 AND evidence_id=?5)""",
                    (
                        video_id,
                        kind,
                        start,
                        end,
                        item.get("id", ""),
                        time.time(),
                    ),
                )

    def deliveries(self, video_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT DISTINCT kind,start,end,evidence_id FROM deliveries WHERE video_id=?",
                (video_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delivered_text(self, video_id: str, items: list[dict]):
        """Accumulate character ranges so a paginated cue can eventually count as read."""
        with self.connect() as db:
            for item in items:
                row = db.execute(
                    "SELECT * FROM segments WHERE video_id=? AND id=?",
                    (video_id, item["id"]),
                ).fetchone()
                if row is None:
                    continue
                start = item.get("text_offset", 0)
                text = item["text"]
                if item.get("excerpt_truncated"):
                    text = text[:-1]
                end = start + len(text)
                if not text or row["text"][start:end] != text:
                    continue
                db.execute(
                    "INSERT OR IGNORE INTO text_deliveries VALUES (?,?,?,?)",
                    (video_id, row["id"], start, end),
                )
                intervals = db.execute(
                    "SELECT start_char,end_char FROM text_deliveries "
                    "WHERE video_id=? AND evidence_id=? ORDER BY start_char",
                    (video_id, row["id"]),
                ).fetchall()
                covered = 0
                for left, right in intervals:
                    if left > covered:
                        break
                    covered = max(covered, right)
                if covered >= len(row["text"]):
                    db.execute(
                        "INSERT INTO deliveries SELECT ?, 'text', ?, ?, ?, ? "
                        "WHERE NOT EXISTS (SELECT 1 FROM deliveries "
                        "WHERE video_id=? AND kind='text' AND evidence_id=?)",
                        (
                            video_id,
                            row["start"],
                            row["end"],
                            row["id"],
                            time.time(),
                            video_id,
                            row["id"],
                        ),
                    )

    def list(self, limit: int) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT data,updated FROM videos ORDER BY updated DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "id": (v := json.loads(r["data"]))["id"],
                "title": v["title"],
                "url": v["url"],
                "duration": v["duration"],
                "updated": r["updated"],
            }
            for r in rows
        ]


def union_intervals(intervals: list[tuple[float, float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [[round(a, 3), round(b, 3)] for a, b in merged]

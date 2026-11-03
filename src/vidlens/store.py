"""Transactional local evidence store, independent of any model conversation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .models import Frame, Segment, Video, VidLensError


def segment_id(video_id: str, segment: Segment) -> str:
    value = f"{video_id}|{segment.kind}|{segment.start:.3f}|{segment.end:.3f}|{segment.text}"
    return hashlib.sha256(value.encode()).hexdigest()[:16]


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
    ):
        with self.connect() as db:
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
            elif replace:
                db.execute("DELETE FROM segments WHERE video_id=? AND kind != 'ocr'", (video.id,))
            self._add_segments(db, video.id, segments)

    def _add_segments(self, db, video_id: str, segments: list[Segment]):
        for s in segments:
            if s.end < s.start or s.start < 0 or not s.text.strip():
                continue
            s.id = segment_id(video_id, s)
            db.execute(
                "INSERT OR REPLACE INTO segments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (s.id, video_id, s.start, s.end, s.text, s.kind, s.source, s.language),
            )

    def add_segments(self, video_id: str, segments: list[Segment]):
        with self.connect() as db:
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
            for frame in frames:
                db.execute(
                    "INSERT OR REPLACE INTO frames VALUES (?, ?, ?)",
                    (video_id, frame.timestamp, json.dumps(frame.to_dict(), ensure_ascii=False)),
                )
                if frame.ocr.strip():
                    db.execute(
                        "DELETE FROM segments WHERE video_id=? AND kind='ocr' AND start=?",
                        (video_id, frame.timestamp),
                    )
                    self._add_segments(
                        db,
                        video_id,
                        [
                            Segment(
                                frame.timestamp,
                                frame.timestamp + 0.001,
                                frame.ocr,
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
        data["path"] = Path(data["path"])
        return Frame(**data) if data["path"].is_file() else None

    def delivered(self, video_id: str, items: list[dict], kind: str = "text"):
        with self.connect() as db:
            for item in items:
                db.execute(
                    """INSERT INTO deliveries SELECT ?1, ?2, ?3, ?4, ?5, ?6
                       WHERE NOT EXISTS (SELECT 1 FROM deliveries WHERE video_id=?1 AND kind=?2
                       AND start=?3 AND end=?4 AND evidence_id=?5)""",
                    (
                        video_id,
                        kind,
                        item.get("start", item.get("timestamp", 0)),
                        item.get("end", item.get("timestamp", 0)),
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

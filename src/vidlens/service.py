"""Astra-facing orchestration: progressive reading with durable original evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import threading
import time
import uuid
from pathlib import Path

from filelock import FileLock

from . import __version__
from .budget import TokenBudget
from .media_cache import MediaCache
from .models import MediaAsset, Segment, Video, VidLensError
from .reading import EvidenceReader, validate_budget, validate_interval
from .retrieval import source_link
from .store import Store, union_intervals


class VideoService:
    def __init__(self, home: Path | str | None = None):
        self.home = (
            Path(home or os.environ.get("VIDLENS_HOME", Path.home() / ".cache" / "vidlens"))
            .expanduser()
            .resolve()
        )
        self.store = Store(self.home)
        self.budget = TokenBudget(self.home)
        self.media = MediaCache(self.store)
        self.reader = EvidenceReader(self.store, self.budget)
        self._visual_lock = threading.RLock()
        self._locks = self.home / "locks"
        self._locks.mkdir(exist_ok=True)
        self._provider = None
        self._visual = None

    @property
    def provider(self):
        if self._provider is None:
            from .sources import SourceProvider

            cookie = os.environ.get("VIDLENS_COOKIE_FILE")
            self._provider = SourceProvider(
                self.home / "sources", cookie_file=Path(cookie) if cookie else None
            )
        return self._provider

    @property
    def visual(self):
        if self._visual is None:
            from .visual import VisualReader

            self._visual = VisualReader(self.home / "visual")
        return self._visual

    def open_video(self, url, languages=("zh", "en"), refresh=False, max_tokens=2500):
        validate_budget(max_tokens)
        started = time.perf_counter()
        video, hit = self._ensure_video(url, languages, refresh)
        result = self.reader.overview(video)
        result["cache_hit"] = hit
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return self.budget.finish(result, max_tokens, "outline", len(result["outline"]))

    def _ensure_video(self, url, languages, refresh):
        from .sources import normalize_url

        if (
            not languages
            or isinstance(languages, str)
            or any(not isinstance(x, str) or not x.strip() for x in languages)
        ):
            raise VidLensError("invalid_languages", "Provide one or more language codes.")
        if not isinstance(url, str):
            raise VidLensError("invalid_url", "Supply a YouTube or Bilibili video URL.")
        raw = url.strip()
        preferences = list(languages)

        def lookup(key):
            cached = self.store.by_url(key)
            if cached and cached.metadata.get("language_preferences", preferences) == preferences:
                return cached
            return None

        if not refresh and (cached := lookup(raw)):
            return cached, True
        normalized = normalize_url(raw)
        with self._file_lock("open", normalized):
            if not refresh and (cached := lookup(normalized)):
                self.store.remember_alias(raw, cached.id)
                return cached, True
            acquired = self.provider.acquire(normalized, languages=tuple(preferences))
            self._validate_video(acquired.video, acquired.segments)
            acquired.video.metadata.update(
                language_preferences=preferences, revision=uuid.uuid4().hex
            )
            with self._file_lock("content", acquired.video.id):
                self.store.save(
                    acquired.video,
                    acquired.segments,
                    acquired.warnings,
                    invalidate_derived=True,
                    aliases=(raw, normalized),
                )
            return acquired.video, False

    def _file_lock(self, scope, value):
        key = hashlib.sha256(value.encode()).hexdigest()[:24]
        return FileLock(str(self._locks / f"{scope}-{key}.lock"), timeout=1500)

    def read_video(self, url, question="", max_tokens=4000, languages=("zh", "en"), refresh=False):
        """One-call evidence entry point: no intermediate outline/search payload duplication."""
        validate_budget(max_tokens)
        if not isinstance(question, str) or len(question) > 2000:
            raise VidLensError("invalid_query", "question must be text up to 2000 characters.")
        started = time.perf_counter()
        video, hit = self._ensure_video(url, languages, refresh)
        return self.reader.read(video, question, max_tokens, hit, started)

    def overview(self, video_id, max_tokens=2500):
        validate_budget(max_tokens)
        result = self.reader.overview(self.store.get(video_id))
        return self.budget.finish(result, max_tokens, "outline", len(result["outline"]))

    def search(self, video_id, query, max_tokens=3000, limit=6):
        return self.reader.search(video_id, query, max_tokens, limit)

    def read_interval(self, video_id, start, end, max_tokens=4000, cursor=None):
        return self.reader.read_interval(video_id, start, end, max_tokens, cursor)

    def _asset(self, video: Video, start: float, end: float, audio_only=False) -> MediaAsset:
        return self.media.get(video, start, end, audio_only, lambda: self.provider)

    def read_frames(self, video_id, start, end, max_frames=4, strategy="changes", ocr=True):
        if (
            isinstance(max_frames, bool)
            or not isinstance(max_frames, int)
            or not 1 <= max_frames <= 12
        ):
            raise VidLensError("invalid_frame_count", "max_frames must be from 1 to 12.")
        if strategy not in ("changes", "uniform"):
            raise VidLensError("invalid_strategy", "strategy must be changes or uniform.")
        started = time.perf_counter()
        with self._file_lock("content", video_id), self._visual_lock:
            video = self.store.get(video_id)
            start, end = validate_interval(video, start, end, max_duration=600)
            asset = self._asset(video, start, end)
            frames = self.visual.read(
                asset, start, end, max_frames=max_frames, strategy=strategy, ocr=ocr
            )
            self.store.save_frames(video_id, frames)
            items = [f.to_dict() | {"url": source_link(video, f.timestamp)} for f in frames]
            for item in items:
                item["ocr"] = self.budget.truncate(item.get("ocr", ""), 1000)
            self.store.delivered(video_id, items, "frame")
            warnings = list(getattr(self.visual, "last_warnings", []))
        return {
            "video_id": video_id,
            "start": start,
            "end": end,
            "frames": items,
            "warnings": warnings,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "coverage_note": "These are sampled instants, not continuous visual coverage. OCR can be wrong.",
            "accounting": {
                "image_count": len(frames),
                "unique_image_count": sum(f.duplicate_of is None for f in frames),
                "image_tokens": "model-dependent; not included in text budgets",
            },
        }

    def crop_frame(self, video_id, timestamp, box):
        self.store.get(video_id)
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise VidLensError(
                "invalid_timestamp", "timestamp must be finite seconds from a returned frame."
            )
        with self._file_lock("content", video_id), self._visual_lock:
            frame = self.store.frame_at(video_id, timestamp)
            if frame is None:
                raise VidLensError(
                    "frame_not_found", "Read frames first, then use a returned timestamp."
                )
            cropped = self.visual.crop(frame, tuple(box))
            return {
                "video_id": video_id,
                "frame": cropped.to_dict(),
                "crop": list(box),
                "warnings": list(getattr(self.visual, "last_warnings", [])),
            }

    def coverage(self, video_id):
        video = self.store.get(video_id)
        segments = self.store.segments(video_id)
        speech = [s for s in segments if s.kind != "ocr"]
        delivered = self.store.deliveries(video_id)
        text_intervals = union_intervals(
            [(i["start"], i["end"]) for i in delivered if i["kind"] == "text"]
        )
        excerpts = union_intervals(
            [(i["start"], i["end"]) for i in delivered if i["kind"] == "excerpt"]
        )
        frames = sorted({i["start"] for i in delivered if i["kind"] == "frame"})
        indexed = union_intervals([(s.start, s.end) for s in speech])
        return {
            "video_id": video_id,
            "duration": video.duration,
            "indexed_transcript_seconds": round(sum(b - a for a, b in indexed), 3),
            "transcript_sources": sorted({s.source for s in speech}),
            "text_delivered_intervals": text_intervals,
            "excerpt_delivered_intervals": excerpts,
            "frame_timestamps_delivered": frames,
            "ocr_frames_indexed": sum(s.kind == "ocr" for s in segments),
            "meaning": "Cumulative delivery to tool callers, not proof Astra read/understood it; frames cover instants only.",
            "warnings": self.store.warnings(video_id),
        }

    def transcribe(self, video_id, start=0, end=None, language=None, model="base"):
        with self._file_lock("content", video_id):
            video = self.store.get(video_id)
            start, end = validate_interval(video, start, video.duration if end is None else end)
            history = video.metadata.setdefault("transcriptions", [])
            same_track = sorted(
                (r["start"], r["end"])
                for r in history
                if r["language"] == language and r["model"] == model
            )
            # Cues already present on this track must not appear twice when follow-up
            # questions request overlapping audio windows.
            missing = []
            cursor = start
            for a, b in same_track:
                if b <= cursor or a >= end:
                    continue
                if a > cursor:
                    missing.append((cursor, a))
                cursor = max(cursor, b)
            if cursor < end:
                missing.append((cursor, end))
            if not missing:
                available = sum(
                    s.kind == "transcript" and s.source == f"whisper:{model}"
                    for s in self.store.segments(video_id, start, end)
                )
                return {
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                    "cache_hit": True,
                    "segments_added": 0,
                    "segments_available": available,
                    "source": "local_asr",
                }
            added = 0
            for a, b in missing:
                asset = self.media.audio_interval(self._asset(video, a, b, audio_only=True), a, b)
                segments = self.provider.transcribe(asset, language=language, model=model)
                self._validate_video(video, segments)
                history.append(
                    {
                        "start": a,
                        "end": b,
                        "language": language,
                        "model": model,
                        "segments": len(segments),
                    }
                )
                video.metadata["transcribed_intervals"] = union_intervals(
                    [tuple(i) for i in video.metadata.get("transcribed_intervals", [])] + [(a, b)]
                )
                self.store.save(video, segments, self.store.warnings(video_id), replace=False)
                added += len(segments)
        return {
            "video_id": video_id,
            "start": start,
            "end": end,
            "segments_added": added,
            "cache_hit": False,
            "source": "local_asr",
            "next_action": "Search or read the interval; ASR may misrecognize technical terms.",
        }

    @staticmethod
    def _validate_video(video: Video, segments: list[Segment]):
        if not video.id or not math.isfinite(video.duration) or video.duration < 0:
            raise VidLensError("invalid_video", "Invalid video metadata.")
        for s in segments:
            if (
                not math.isfinite(s.start)
                or not math.isfinite(s.end)
                or s.start < 0
                or s.end < s.start
            ):
                raise VidLensError(
                    "invalid_transcript",
                    "Subtitle timestamps must be finite, nonnegative and ordered.",
                )

    def import_transcript(self, path, title="", video_url="", media_path=None):
        from .sources import parse_subtitles

        source = Path(path).expanduser().resolve()
        if not source.is_file() or source.stat().st_size > 20_000_000:
            raise VidLensError(
                "invalid_import", "Provide an existing subtitle file no larger than 20 MB."
            )
        content = source.read_text(encoding="utf-8-sig")
        fmt = source.suffix.lower().lstrip(".")
        segments = parse_subtitles(content, fmt, source="imported_subtitles")
        if not segments:
            raise VidLensError("empty_transcript", "No timed text was found in this file.")
        media = Path(media_path).expanduser().resolve() if media_path else None
        if media and not media.is_file():
            raise VidLensError("media_missing", "The supplied local video does not exist.")
        video_id = (
            "local-"
            + hashlib.sha256((content + video_url + str(media or "")).encode()).hexdigest()[:16]
        )
        duration = max(s.end for s in segments)
        if media:
            import re
            import subprocess

            from .visual import ffmpeg_binary

            probe = subprocess.run(
                [ffmpeg_binary(), "-hide_banner", "-i", str(media)],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
            if match:
                duration = max(
                    duration, float(match[1]) * 3600 + float(match[2]) * 60 + float(match[3])
                )
            else:
                raise VidLensError(
                    "media_probe_failed", "Could not determine the supplied video's duration."
                )
        video = Video(
            video_id,
            video_url,
            "local",
            title or source.stem,
            duration,
            metadata={"local_media_path": str(media)} if media else {},
        )
        self._validate_video(video, segments)
        with self._file_lock("content", video_id):
            self.store.save(video, segments, [], invalidate_derived=True)
        return self.overview(video_id)

    def doctor(self):
        capabilities = self.visual.capabilities()
        return {
            "version": __version__,
            "home": str(self.home),
            "python": os.sys.version.split()[0],
            "capabilities": capabilities,
            "optional_asr_installed": importlib.util.find_spec("faster_whisper") is not None,
            "cookie_file_configured": bool(os.environ.get("VIDLENS_COOKIE_FILE")),
            "token_count_method": self.budget.method,
            "llm_api_required": False,
            "notes": [
                "Astra interprets returned evidence. No external LLM API is invoked.",
                "Platform access, subtitles, authentication and region restrictions can affect individual URLs.",
            ],
        }

    def list_videos(self, limit=20):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise VidLensError("invalid_limit", "limit must be from 1 to 100.")
        return {"videos": self.store.list(limit)}

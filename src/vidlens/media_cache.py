"""Reusable media intervals. Paths and offsets always refer to a single source revision."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from filelock import FileLock

from .budget import compact_json
from .models import MediaAsset, VidLensError


class MediaCache:
    def __init__(self, store):
        self.store = store
        self.root = store.home / "media"

    def get(self, video, start, end, audio_only, provider):
        local = video.metadata.get("local_media_path")
        if local:
            path = Path(local)
            if not path.is_file():
                raise VidLensError("media_missing", "The imported local video no longer exists.")
            return MediaAsset(path, 0, video.duration, False)

        revision = video.metadata.get("revision", "")
        profile = "audio" if audio_only else os.environ.get("VIDLENS_MAX_VIDEO_HEIGHT", "1080")
        group = (video.id, revision, profile)
        self.root.mkdir(parents=True, exist_ok=True)
        lock_key = hashlib.sha256(compact_json(group).encode()).hexdigest()[:24]
        with FileLock(str(self.root / f"{lock_key}.lock"), timeout=330):
            with self.store.connect() as db:
                rows = db.execute(
                    """
                    SELECT path,start,end FROM media_assets
                    WHERE id=? AND revision=? AND profile=? AND start<=? AND end>=?
                    ORDER BY end-start
                """,
                    (*group, start, end),
                )
                for row in rows:
                    if Path(row["path"]).is_file():
                        return MediaAsset(
                            Path(row["path"]), row["start"], row["end"] - row["start"], audio_only
                        )

            # Default profile keeps compatibility with pre-release interval manifests.
            identity = f"{video.id}|{revision}|{start:.3f}|{end:.3f}|{audio_only}"
            if profile not in ("audio", "1080"):
                identity += f"|{profile}"
            directory = self.root / hashlib.sha256(identity.encode()).hexdigest()[:24]
            manifest = directory / "asset.json"
            asset = self._existing(manifest, start, end, audio_only)
            if asset is None:
                directory.mkdir(exist_ok=True)
                # A failed yt-dlp run can leave a playable-looking partial file.
                # Only publish the result of a successfully completed attempt.
                with tempfile.TemporaryDirectory(dir=directory) as work:
                    target = Path(work) / ("audio.m4a" if audio_only else "clip.mp4")
                    asset = provider().fetch_media(
                        video, target, start=start, end=end, audio_only=audio_only
                    )
                    saved = directory / asset.path.name
                    asset.path.replace(saved)
                    asset.path = saved
                info = {
                    "path": str(asset.path),
                    "offset": asset.offset,
                    "duration": asset.duration,
                    "audio_only": asset.audio_only,
                }
                temporary = manifest.with_suffix(".tmp")
                temporary.write_text(compact_json(info))
                temporary.replace(manifest)
            with self.store.connect() as db:
                db.execute(
                    "INSERT OR REPLACE INTO media_assets VALUES (?,?,?,?,?,?)",
                    (*group, asset.offset, asset.offset + asset.duration, str(asset.path)),
                )
            return asset

    @staticmethod
    def _existing(manifest, start, end, audio_only):
        try:
            data = json.loads(manifest.read_text())
            asset = MediaAsset(
                Path(data["path"]),
                float(data["offset"]),
                float(data["duration"]),
                data["audio_only"],
            )
            if (
                asset.path.is_file()
                and asset.audio_only == audio_only
                and asset.offset <= start
                and asset.offset + asset.duration >= end - 1e-7
            ):
                return asset
        except (ValueError, TypeError, KeyError, OSError):
            return None
        return None

    def audio_interval(self, asset, start, end):
        if abs(asset.offset - start) < 1e-7 and abs(asset.duration - (end - start)) < 1e-7:
            return asset
        self.root.mkdir(parents=True, exist_ok=True)
        from .visual import ffmpeg_binary

        stat = asset.path.stat()
        identity = (str(asset.path), stat.st_mtime_ns, stat.st_size, asset.offset, start, end)
        target = self.root / (
            "asr-" + hashlib.sha256(compact_json(identity).encode()).hexdigest()[:24] + ".wav"
        )
        with FileLock(str(target) + ".lock", timeout=185):
            if not target.is_file():
                partial = target.with_suffix(".part.wav")
                try:
                    subprocess.run(
                        [
                            ffmpeg_binary(),
                            "-nostdin",
                            "-y",
                            "-ss",
                            str(start - asset.offset),
                            "-i",
                            str(asset.path),
                            "-t",
                            str(end - start),
                            "-vn",
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            str(partial),
                        ],
                        check=True,
                        capture_output=True,
                        timeout=180,
                    )
                    partial.replace(target)
                except (subprocess.SubprocessError, OSError) as exc:
                    raise VidLensError(
                        "audio_extract_failed", "Could not extract the requested audio interval."
                    ) from exc
                finally:
                    partial.unlink(missing_ok=True)
        return MediaAsset(target, start, end - start, True)

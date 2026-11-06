"""Bounded, on-demand video frames and local OCR on the original timeline."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from pathlib import Path

from filelock import FileLock, Timeout
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps, ImageStat

from .models import Frame, MediaAsset, VidLensError


class VisualReader:
    OCR_VERSION = 3
    MAX_FRAMES = 16
    MAX_CANDIDATES = 48
    MAX_READ_SECONDS = 90
    ORIGINAL_SIZE = (3840, 2160)
    DISPLAY_SIZE = (1920, 1080)

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_warnings: list[str] = []
        self._ocr_status = "not_tested"

    @staticmethod
    def _ffmpeg() -> str:
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError, OSError) as exc:
            raise VidLensError(
                "ffmpeg_unavailable",
                "Video frame extraction needs FFmpeg.",
                "Install imageio-ffmpeg or make ffmpeg available on PATH.",
            ) from exc

    def capabilities(self) -> dict:
        try:
            ffmpeg = self._ffmpeg()
        except VidLensError:
            ffmpeg = None
        swift = shutil.which("swiftc") if platform.system() == "Darwin" else None
        tesseract = shutil.which("tesseract")
        return {
            "ffmpeg": ffmpeg,
            "ocr_backend": "macos_vision" if swift else "tesseract" if tesseract else None,
            "ocr_backend_detected": bool(swift or tesseract),
            "ocr_runtime_status": self._ocr_status,
            "max_frames": self.MAX_FRAMES,
            "max_candidates": self.MAX_CANDIDATES,
            "max_image_size": list(self.DISPLAY_SIZE),
            "max_source_size": list(self.ORIGINAL_SIZE),
        }

    def _warn(self, message: str) -> None:
        if message not in self.last_warnings:
            self.last_warnings.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    @staticmethod
    def _sidecar(path: Path) -> Path:
        return path.with_suffix(".json")

    @staticmethod
    def _finite(value: float) -> bool:
        return (
            isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
        )

    def scan(
        self,
        asset: MediaAsset,
        *,
        start: float | None = None,
        end: float | None = None,
        cursor: int = 0,
        limit: int = 200,
    ) -> dict:
        """Index an entire local timeline; paginate metadata, then read selected frames.

        Operational failures return explicit failed/partial status and never publish
        a completed cache. Invalid arguments retain the usual VidLensError contract.
        """
        from .storyboard import StoryboardScanner

        self.last_warnings = []
        result = StoryboardScanner(self.cache_dir, self._ffmpeg).scan(
            asset, start=start, end=end, cursor=cursor, limit=limit
        )
        self.last_warnings = list(result["warnings"])
        return result

    def read_at(
        self, asset: MediaAsset, timestamps: list[float], ocr: bool = True, *, native: bool = False,
    ) -> list[Frame]:
        """Read the frame displayed at each requested source time, including VFR holds.

        timestamp is the actual decoded PTS. requested_timestamp and visible_until
        preserve the request mapping and the decoded frame's display interval.
        """
        self.last_warnings = []
        if not isinstance(native, bool):
            raise VidLensError("invalid_native_mode", "native must be a boolean.")
        if not isinstance(timestamps, (list, tuple)) or not 1 <= len(timestamps) <= self.MAX_FRAMES:
            raise VidLensError("invalid_frame_count", f"Use 1..{self.MAX_FRAMES} frame times.")
        if not all(self._finite(value) for value in timestamps):
            raise VidLensError("invalid_timestamp", "Frame times must be finite seconds.")
        if (
            not self._finite(asset.offset)
            or not self._finite(asset.duration)
            or asset.offset < 0
            or asset.duration < 0
        ):
            raise VidLensError("invalid_asset", "Media offset and duration must be finite.")
        if any(
            stamp < asset.offset or (asset.duration and stamp >= asset.offset + asset.duration)
            for stamp in timestamps
        ):
            raise VidLensError(
                "interval_outside_asset", "A frame time is outside this media asset."
            )
        if asset.audio_only:
            raise VidLensError("no_video_stream", "This media asset contains audio only.")
        asset.path = Path(asset.path)
        if not asset.path.is_file():
            raise VidLensError("media_missing", "The local media asset does not exist.")
        directory = self._asset_directory(asset)
        frames, digests = [], {}
        try:
            with FileLock(directory / "frames.lock", timeout=self.MAX_READ_SECONDS):
                executable = self._ffmpeg()
                deadline = time.monotonic() + self.MAX_READ_SECONDS
                for timestamp in sorted(set(timestamps)):
                    if time.monotonic() >= deadline:
                        raise VidLensError(
                            "frame_timeout",
                            "Not all requested candidate frames fit the extraction time limit.",
                        )
                    actual, visible_until = self._display_interval(
                        executable, asset, timestamp, directory, deadline
                    )
                    target = max(asset.offset, actual - 0.00001)
                    frame = self._extract(
                        executable, asset, target, directory, deadline,
                        absolute_pts=True, native=native,
                    )
                    if abs(frame.timestamp - actual) > 0.000002:
                        raise VidLensError(
                            "frame_timeline_mismatch",
                            "The extracted image did not match its decoded timeline.",
                        )
                    frame.selection = "displayed_at_requested_time"
                    frame.requested_timestamp = timestamp
                    frame.visible_until = visible_until
                    source = Path(json.loads(self._sidecar(frame.path).read_text())["source"])
                    digest = hashlib.sha256(source.read_bytes()).digest()
                    frame.duplicate_of = digests.get(digest)
                    digests.setdefault(digest, frame.timestamp)
                    frames.append(frame)
                if ocr:
                    unique = {
                        frame.timestamp: frame for frame in frames if frame.duplicate_of is None
                    }
                    self._ocr(list(unique.values()))
                    for frame in frames:
                        if frame.duplicate_of in unique:
                            frame.ocr = unique[frame.duplicate_of].ocr
                            frame.ocr_regions = unique[frame.duplicate_of].ocr_regions
                return frames
        except Timeout as exc:
            raise VidLensError(
                "frame_cache_busy", "Another reader is working on this media; retry shortly."
            ) from exc

    def _display_interval(self, executable, asset, timestamp, directory, deadline):
        """Decode a bounded keyframe neighborhood; persist adjacent-PTS intervals.

        Accurate input seeking drops the frame already on screen at the seek
        point. noaccurate_seek retains that preceding keyframe neighborhood, and
        copyts/AVTB lets us choose the actual presentation interval ourselves.
        """
        index = directory / "display-intervals-v1.json"
        try:
            intervals = json.loads(index.read_text())
            if not isinstance(intervals, list) or not all(
                isinstance(row, list)
                and len(row) == 2
                and all(self._finite(value) for value in row)
                and row[0] < row[1]
                for row in intervals
            ):
                intervals = []
        except (OSError, ValueError, TypeError):
            intervals = []

        def lookup(rows):
            for left, right in reversed(rows):
                if left <= timestamp + 0.0000005 and timestamp < right - 0.0000005:
                    return left, right
            return None

        if hit := lookup(intervals):
            return hit
        local = max(0, timestamp - asset.offset - 0.00001)
        command = [
            executable,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "info",
            "-copyts",
            "-start_at_zero",
            "-ss",
            f"{local:.9f}",
            "-noaccurate_seek",
            "-i",
            str(asset.path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-vf",
            (
                "settb=AVTB,showinfo=checksum=0,"
                f"select='gt(t,{timestamp - asset.offset + 0.0000005:.9f})'"
            ),
            "-frames:v",
            "1",
            "-threads",
            "1",
            "-fps_mode",
            "passthrough",
            "-f",
            "null",
            "-",
        ]
        try:
            with tempfile.TemporaryFile() as output:
                result = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=output,
                    check=False,
                    timeout=max(0.1, min(12, deadline - time.monotonic())),
                )
                if output.tell() > 4 * 1024 * 1024:
                    raise VidLensError(
                        "frame_timeline_limit", "The frame timeline metadata limit was reached."
                    )
                output.seek(0)
                details = output.read().decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            raise VidLensError(
                "frame_timeout", "FFmpeg timed out while resolving the displayed frame."
            ) from exc
        except OSError as exc:
            raise VidLensError(
                "ffmpeg_failed", "FFmpeg could not resolve the frame timeline."
            ) from exc
        points = []
        for line in details.splitlines():
            match = re.search(r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:", line)
            if match:
                stamp = round(asset.offset + int(match[1]) / 1_000_000, 6)
                duration = re.search(r"\bduration:\s*(\d+)", line)
                points.append((stamp, int(duration[1]) / 1_000_000 if duration else 0))
        if result.returncode or not points:
            raise VidLensError(
                "frame_decode_failed", "FFmpeg did not return a valid presentation timeline."
            )
        points = sorted(dict(points).items())
        container = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", details)
        container_end = (
            (asset.offset + int(container[1]) * 3600 + int(container[2]) * 60 + float(container[3]))
            if container
            else None
        )
        discovered = []
        for i, (stamp, duration) in enumerate(points):
            # The final decoded point may be a lookahead frame. Its nominal
            # duration does not establish the next VFR PTS, so cache it only when
            # no future point was found and decoding really reached EOF.
            if i + 1 == len(points) and stamp > timestamp + 0.0000005:
                continue
            until = (
                points[i + 1][0]
                if i + 1 < len(points)
                else (round(stamp + duration, 6) if duration else container_end)
            )
            if until is not None and until > stamp:
                discovered.append([stamp, until])
        merged = {left: right for left, right in intervals}
        merged.update(discovered)
        intervals = sorted([left, right] for left, right in merged.items())
        self._write_json(index, intervals)
        if hit := lookup(intervals):
            return hit
        raise VidLensError(
            "frame_not_visible", "No decoded frame is displayed at the requested time."
        )

    def _asset_directory(self, asset: MediaAsset) -> Path:
        stat = asset.path.stat()
        identity = f"v2:{asset.path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{asset.offset}"
        directory = self.cache_dir / hashlib.sha256(identity.encode()).hexdigest()[:24]
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def read(
        self,
        asset: MediaAsset,
        start: float,
        end: float,
        max_frames: int = 6,
        strategy: str = "changes",
        ocr: bool = True,
    ) -> list[Frame]:
        """Sample [start, end); returned timestamps refer to actual decoded frames.

        The change strategy samples within timeline buckets and keeps one frame per
        bucket, so static sections remain represented. Extracted frames are cached;
        repeated visual states are annotated instead of erasing coverage.
        """
        self.last_warnings = []
        if not self._finite(start) or not self._finite(end) or start < 0 or end <= start:
            raise VidLensError("invalid_interval", "Use finite times with 0 <= start < end.")
        if (
            isinstance(max_frames, bool)
            or not isinstance(max_frames, int)
            or not (1 <= max_frames <= self.MAX_FRAMES)
        ):
            raise VidLensError("invalid_frame_count", f"max_frames must be 1..{self.MAX_FRAMES}.")
        if strategy not in {"changes", "uniform"}:
            raise VidLensError("invalid_strategy", "strategy must be 'changes' or 'uniform'.")
        if not self._finite(asset.offset) or not self._finite(asset.duration) or asset.duration < 0:
            raise VidLensError("invalid_asset", "Media offset and duration must be finite.")
        if asset.audio_only:
            raise VidLensError("no_video_stream", "This media asset contains audio only.")
        asset.path = Path(asset.path)
        if not asset.path.is_file():
            raise VidLensError("media_missing", "The local media asset does not exist.")
        if start < asset.offset - 0.001:
            raise VidLensError(
                "interval_outside_asset", "The interval begins before this media asset."
            )
        if asset.duration and end > asset.offset + asset.duration + 0.001:
            raise VidLensError(
                "interval_outside_asset", "The interval ends after this media asset."
            )

        directory = self._asset_directory(asset)
        request = f"v3:{float(start).hex()}:{float(end).hex()}:{max_frames}:{strategy}"
        manifest = directory / f"selection-{hashlib.sha256(request.encode()).hexdigest()[:24]}.json"
        try:
            with FileLock(directory / "frames.lock", timeout=self.MAX_READ_SECONDS):
                frames = self._cached_selection(directory, manifest)
                if frames is None:
                    frames = self._sample(asset, start, end, max_frames, strategy, directory)
                    if not self.last_warnings:
                        rows = [[f.path.name, f.selection, f.duplicate_of] for f in frames]
                        self._write_json(manifest, rows)
                if ocr:
                    unique = {f.timestamp: f for f in frames if f.duplicate_of is None}
                    self._ocr(list(unique.values()))
                    for frame in frames:
                        if frame.duplicate_of in unique:
                            frame.ocr = unique[frame.duplicate_of].ocr
                            frame.ocr_regions = unique[frame.duplicate_of].ocr_regions
                return frames
        except Timeout as exc:
            raise VidLensError(
                "frame_cache_busy", "Another reader is working on this media; retry shortly."
            ) from exc

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _cached_frame(self, path: Path) -> Frame | None:
        try:
            info = json.loads(self._sidecar(path).read_text())
            source = Path(info["source"])
            if not path.is_file() or not source.is_file():
                return None
            if "bytes" in info and path.stat().st_size != info["bytes"]:
                return None
            if "source_bytes" in info and source.stat().st_size != info["source_bytes"]:
                return None
            return Frame(
                info["timestamp"], path, info["width"], info["height"], info.get("ocr", ""),
                ocr_regions=self._validated_regions(info.get("ocr_regions")) or [],
                crop_box=info.get("crop"),
            )
        except (ValueError, TypeError, KeyError, OSError):
            return None

    def _cached_selection(self, directory: Path, manifest: Path) -> list[Frame] | None:
        try:
            rows = json.loads(manifest.read_text())
            if not isinstance(rows, list) or not 0 < len(rows) <= self.MAX_FRAMES:
                return None
            frames = []
            for name, selection, duplicate in rows:
                if Path(name).name != name:
                    return None
                frame = self._cached_frame(directory / name)
                if frame is None:
                    return None
                frame.selection, frame.duplicate_of = selection, duplicate
                frames.append(frame)
            return frames
        except (ValueError, TypeError, KeyError, OSError):
            return None

    def _preview(
        self,
        executable: str,
        asset: MediaAsset,
        start: float,
        end: float,
        targets: list[float],
        deadline: float,
    ) -> dict[float, tuple[float, Image.Image]] | None:
        # Select the first decoded frame crossing each target. FFmpeg's prev_t
        # refers to the preceding INPUT frame, including on variable-rate media.
        terms = []
        for target in targets:
            at = f"{target - start:.9f}"
            terms.append(f"gte(t+1e-8,{at})*(isnan(prev_t)+lt(prev_t+1e-8,{at}))")
        graph = (
            f"select='{'+'.join(terms)}',scale=96:54:flags=area,format=rgb24,showinfo=checksum=0"
        )
        command = [
            executable,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "info",
            "-ss",
            f"{start - asset.offset:.9f}",
            "-t",
            f"{end - start:.9f}",
            "-i",
            str(asset.path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-vf",
            graph,
            "-frames:v",
            str(len(targets)),
            "-fps_mode",
            "passthrough",
            "-c:v",
            "rawvideo",
            "-threads",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=max(0.1, min(15, deadline - time.monotonic())),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        stamps = re.findall(rb"\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:([-+\d.eE]+)", result.stderr)
        stride = 96 * 54 * 3
        if result.returncode or not stamps or len(result.stdout) != len(stamps) * stride:
            return None
        previews = {}
        index = 0
        for i, stamp in enumerate(stamps):
            actual = start + float(stamp)
            image = Image.frombytes("RGB", (96, 54), result.stdout[i * stride : (i + 1) * stride])
            while index < len(targets) and targets[index] <= actual + 1e-5:
                previews[targets[index]] = (actual, image)
                index += 1
        return previews

    def _sample(
        self,
        asset: MediaAsset,
        start: float,
        end: float,
        count: int,
        strategy: str,
        directory: Path,
    ) -> list[Frame]:
        executable = self._ffmpeg()
        deadline = time.monotonic() + self.MAX_READ_SECONDS
        width = (end - start) / count
        buckets = []
        for i in range(count):
            # The first bucket has no earlier visual state to compare against.
            fractions = (1 / 6, 1 / 2, 5 / 6) if strategy == "changes" and i else (0.5,)
            buckets.append([start + width * (i + fraction) for fraction in fractions])
        targets = [target for bucket in buckets for target in bucket]
        previews = None
        if strategy == "changes" and count > 1 and end - start <= min(30, len(targets) * 2.5):
            previews = self._preview(executable, asset, start, end, targets, deadline)
        selected, signatures, digests = [], [], {}

        def extract(target: float) -> Frame | None:
            if time.monotonic() >= deadline:
                self._warn("Frame extraction reached its time budget; coverage is partial.")
                return None
            try:
                frame = self._extract(executable, asset, target, directory, deadline)
            except VidLensError:
                return None
            return frame if start - 0.001 <= frame.timestamp < end else None

        for index, bucket in enumerate(buckets):
            left, right = start + index * width, start + (index + 1) * width
            candidates = []
            for target in bucket:
                if previews is not None:
                    if target in previews and left - 0.000001 <= previews[target][0] < right:
                        candidates.append((target, previews[target][1]))
                elif (frame := extract(target)) and left - 0.000001 <= frame.timestamp < right:
                    signature = (
                        self._signature(frame.path) if strategy == "changes" and count > 1 else None
                    )
                    candidates.append((target, signature))
            if not candidates:
                # A sparse stream may have only an early frame in the bucket.
                # Seek its left edge before declaring that interval uncovered.
                fallback = extract(left)
                if fallback and left - 0.000001 <= fallback.timestamp < right:
                    signature = (
                        self._signature(fallback.path)
                        if strategy == "changes" and count > 1
                        else None
                    )
                    candidates.append((left, signature))
            if not candidates:
                self._warn(
                    "At least one timeline bucket has no decoded frame; coverage is partial."
                )
                continue
            if signatures:
                target, signature = max(
                    candidates, key=lambda p: min(self._distance(p[1], old) for old in signatures)
                )
            else:
                target, signature = candidates[0]
            frame = extract(target)
            if frame is None:
                self._warn(
                    "A selected frame could not be decoded at full resolution; coverage is partial."
                )
                continue
            if any(abs(frame.timestamp - old.timestamp) < 0.0001 for old in selected):
                continue
            frame.selection = "changes+coverage" if strategy == "changes" else "uniform"
            # Hash the stored source JPEG. Downscaling a 4K code slide to 1080p
            # can hide a changed digit; a thumbnail cannot prove exact equality.
            source = Path(json.loads(self._sidecar(frame.path).read_text())["source"])
            digest = hashlib.sha256(source.read_bytes()).digest()
            frame.duplicate_of = digests.get(digest)
            digests.setdefault(digest, frame.timestamp)
            selected.append(frame)
            if signature is not None:
                signatures.append(signature)
        if not selected:
            raise VidLensError(
                "frames_unavailable",
                "No video frame could be decoded inside the requested interval.",
                "Try a wider interval, and check that the media file includes a video stream.",
            )
        return sorted(selected, key=lambda f: f.timestamp)

    def _extract(
        self,
        executable: str,
        asset: MediaAsset,
        target: float,
        directory: Path,
        deadline: float,
        *,
        absolute_pts: bool = False,
        native: bool = False,
    ) -> Frame:
        local = max(0.0, target - asset.offset)
        prefix = "frame-at" if absolute_pts else "frame"
        if native:
            prefix = f"native-{prefix}"
        extension = "png" if native else "jpg"
        path = directory / f"{prefix}-{local:.6f}.{extension}"
        metadata = self._sidecar(path)
        cached = self._cached_frame(path)
        if cached is not None:
            return cached
        original = directory / f"{prefix}-{local:.6f}.source.{extension}"
        # With input seeking, showinfo PTS is relative to the seek target. Adding
        # that residual retains real frame time instead of labelling the requested time.
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostdin",
            "-y",
            *(["-copyts", "-start_at_zero"] if absolute_pts else []),
            "-ss",
            f"{local:.9f}",
            "-i",
            str(asset.path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-an",
            "-sn",
            "-vf",
            ("settb=AVTB," if absolute_pts else "")
            + "showinfo,scale=w='min(3840,iw)':h='min(2160,ih)':force_original_aspect_ratio=decrease",
            *([] if native else ["-q:v", "2"]),
            "-threads",
            "1",
            "-update",
            "1",
            str(original),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(0.1, min(12, deadline - time.monotonic())),
            )
        except subprocess.TimeoutExpired as exc:
            raise VidLensError(
                "frame_timeout", "FFmpeg exceeded the frame extraction time limit."
            ) from exc
        except OSError as exc:
            raise VidLensError("ffmpeg_failed", "FFmpeg could not be started.") from exc
        matches = re.findall(r"\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:([-+\d.eE]+)", result.stderr)
        sizes = re.findall(r"\bn:\s*\d+[^\n]*?\bs:(\d+)x(\d+)", result.stderr)
        if result.returncode or not original.is_file() or not matches:
            raise VidLensError(
                "frame_decode_failed", "FFmpeg did not return a timestamped video frame."
            )
        if absolute_pts:
            points = re.findall(r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:", result.stderr)
            if not points:
                raise VidLensError("frame_decode_failed", "FFmpeg did not return a source PTS.")
            timestamp = round(asset.offset + int(points[0]) / 1_000_000, 6)
        else:
            timestamp = round(asset.offset + local + float(matches[0]), 6)
        try:
            with Image.open(original) as image:
                width, height = image.size
                if not native and (width > self.DISPLAY_SIZE[0] or height > self.DISPLAY_SIZE[1]):
                    display = ImageOps.exif_transpose(image).convert("RGB")
                    display.thumbnail(self.DISPLAY_SIZE, Image.Resampling.LANCZOS)
                    display.save(path, quality=92)
                    width, height = display.size
                else:
                    original.replace(path)
                    original = path
        except (OSError, ValueError) as exc:
            raise VidLensError("invalid_frame", "The extracted frame could not be read.") from exc
        self._write_json(
            metadata,
            {
                "timestamp": timestamp,
                "width": width,
                "height": height,
                "source": str(original.resolve()),
                "bytes": path.stat().st_size,
                "source_bytes": original.stat().st_size,
                "decoded_source_size": list(map(int, sizes[0])) if sizes else None,
                "ocr": "",
                "ocr_complete": False,
            },
        )
        return Frame(timestamp, path, width, height)

    @staticmethod
    def _signature(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB").resize((96, 54), Image.Resampling.BILINEAR)

    @staticmethod
    def _distance(first: Image.Image, second: Image.Image) -> float:
        return sum(ImageStat.Stat(ImageChops.difference(first, second)).mean) / (3 * 255)

    @classmethod
    def _source_path(cls, frame: Frame, info=None) -> Path:
        source = Path(frame.path)
        if info is None:
            try:
                info = json.loads(cls._sidecar(source).read_text())
            except (OSError, ValueError):
                info = {}
        value = info.get("source") if isinstance(info, dict) else None
        if isinstance(value, str) and (candidate := Path(value)).is_file():
            return candidate
        return source

    def source_info(self, frame: Frame) -> dict:
        """Inspect retained source size without creating crops or running OCR."""
        source = self._source_path(frame)
        try:
            with Image.open(source) as image:
                size = list(ImageOps.exif_transpose(image).size)
                encoding = (image.format or source.suffix.lstrip(".")).lower()
        except (OSError, ValueError) as exc:
            raise VidLensError("invalid_frame", "The retained source image is unreadable.") from exc
        try:
            metadata = json.loads(self._sidecar(frame.path).read_text())
            decoded = metadata.get("decoded_source_size") if isinstance(metadata, dict) else None
        except (OSError, ValueError):
            decoded = None
        if (not isinstance(decoded, list) or len(decoded) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                           for v in decoded)):
            decoded = None
        scale = [size[0] / decoded[0], size[1] / decoded[1]] if decoded else None
        return {"path": source, "size": size, "encoding": encoding,
                "decoded_source_size": decoded, "source_scale": scale,
                "source_limited": any(v < 1 for v in scale) if scale else None}

    def crop(
        self, frame: Frame, box: tuple[float, float, float, float], *, ocr: bool = True,
    ) -> Frame:
        """Crop normalized x/y/width/height from the cached higher-resolution source."""
        self.last_warnings = []
        if len(box) != 4 or not all(self._finite(value) for value in box):
            raise VidLensError(
                "invalid_crop", "Crop must contain finite normalized x, y, width, height."
            )
        x, y, width, height = box
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise VidLensError("invalid_crop", "Crop must lie inside the normalized [0, 1] image.")
        source = self._source_path(frame)
        try:
            stat = source.stat()
        except OSError as exc:
            raise VidLensError(
                "crop_failed", "The source image is unavailable; read the frames again."
            ) from exc

        key = f"crop-v3:{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{tuple(map(float, box))}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        directory = self.cache_dir / "crops"
        directory.mkdir(exist_ok=True)
        destination = directory / f"{digest}.png"
        try:
            with FileLock(directory / f"{digest}.lock", timeout=self.MAX_READ_SECONDS):
                result = self._cached_frame(destination)
                if result is None:
                    result = self._crop_image(source, destination, frame.timestamp, box)
                # The same pixels can occur at several source times. Reusing a
                # crop must not reuse the first caller's citation timestamp.
                result.timestamp = frame.timestamp
                result.selection = "crop"
                if ocr:
                    self._ocr([result])
                return result
        except Timeout as exc:
            raise VidLensError(
                "frame_cache_busy", "Another reader is working on this crop; retry shortly."
            ) from exc

    def detail(self, frame: Frame, boxes=None, *, max_tiles: int = 12, ocr: bool = True) -> dict:
        """Cover requested source regions with overlapping, unscaled PNG tiles.

        The surrounding source frame remains the context image. No sharpening,
        interpolation or invented detail enters these tiles. Large requests fail
        before cropping rather than silently omitting the right/bottom edge.
        """
        self.last_warnings = []
        if isinstance(max_tiles, bool) or not isinstance(max_tiles, int) or not 1 <= max_tiles <= 12:
            raise VidLensError("invalid_tile_count", "max_tiles must be from 1 to 12.")
        boxes = [[0, 0, 1, 1]] if boxes is None else boxes
        if not isinstance(boxes, (list, tuple)) or not 1 <= len(boxes) <= 6:
            raise VidLensError("invalid_detail_regions", "Provide 1..6 normalized detail regions.")
        for box in boxes:
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or not all(self._finite(value) for value in box)):
                raise VidLensError("invalid_crop", "Each region must be finite x/y/width/height.")
            x, y, width, height = box
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
                raise VidLensError("invalid_crop", "Detail regions must lie inside the source image.")
        info = self.source_info(frame)
        source_width, source_height = info["size"]

        def starts(left, right, size):
            if right - left <= size:
                return [left]
            values = list(range(left, right - size + 1, size - 64))
            if values[-1] + size < right:
                values.append(right - size)
            return values

        regions, specifications = [], []
        for region_index, (x, y, width, height) in enumerate(boxes, 1):
            left, top = int(x * source_width), int(y * source_height)
            right = min(source_width, math.ceil((x + width) * source_width))
            bottom = min(source_height, math.ceil((y + height) * source_height))
            regions.append([left / source_width, top / source_height,
                            (right - left) / source_width, (bottom - top) / source_height])
            for yy in starts(top, bottom, self.DISPLAY_SIZE[1]):
                for xx in starts(left, right, self.DISPLAY_SIZE[0]):
                    specifications.append((region_index, [
                        # Nudge inward by far less than one pixel so converting
                        # normalized bounds back via floor/ceil cannot create a
                        # 1921-pixel tile and silently trigger a resize.
                        (xx + 1e-7) / source_width, (yy + 1e-7) / source_height,
                        (min(right - xx, self.DISPLAY_SIZE[0]) - 2e-7) / source_width,
                        (min(bottom - yy, self.DISPLAY_SIZE[1]) - 2e-7) / source_height,
                    ]))
        if len(specifications) > max_tiles:
            raise VidLensError(
                "detail_tile_limit", f"These regions need {len(specifications)} native tiles.",
                "Narrow the regions or increase max_tiles, up to 12.",
            )
        tiles = [self.crop(frame, tuple(box), ocr=False) for _, box in specifications]
        if ocr:
            self._ocr(tiles)
        return {
            "source_size": [source_width, source_height],
            "source_encoding": info["encoding"],
            "frame_size": [frame.width, frame.height],
            "regions": regions,
            "tiles": tiles,
            "tile_regions": [index for index, _ in specifications],
            "decoded_source_size": info["decoded_source_size"],
            "source_scale": info["source_scale"],
            "source_limited": info["source_limited"],
            "pixel_policy": "cached_source_pixels_no_upscale",
            "minimum_overlap_pixels": 64,
        }

    def _crop_image(
        self,
        source: Path,
        destination: Path,
        timestamp: float,
        box: tuple[float, float, float, float],
    ) -> Frame:
        x, y, width, height = box
        original = destination.with_suffix(".source.png")
        try:
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                left, top = int(x * image.width), int(y * image.height)
                right = min(image.width, math.ceil((x + width) * image.width))
                bottom = min(image.height, math.ceil((y + height) * image.height))
                actual_box = [left / image.width, top / image.height,
                              (right - left) / image.width, (bottom - top) / image.height]
                image = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
                if image.width > self.DISPLAY_SIZE[0] or image.height > self.DISPLAY_SIZE[1]:
                    image.save(original)
                    image.thumbnail(self.DISPLAY_SIZE, Image.Resampling.LANCZOS)
                else:
                    original = destination
                image.save(destination)
                result = Frame(timestamp, destination, *image.size,
                               selection="crop", crop_box=actual_box)
        except (OSError, ValueError) as exc:
            raise VidLensError("crop_failed", "The frame could not be cropped.") from exc
        self._write_json(
            self._sidecar(destination),
            {
                "timestamp": result.timestamp,
                "width": result.width,
                "height": result.height,
                "source": str(original.resolve()),
                "bytes": destination.stat().st_size,
                "source_bytes": original.stat().st_size,
                "crop": actual_box,
                "ocr": "",
                "ocr_complete": False,
            },
        )
        return result

    @staticmethod
    def _timestamp(seconds: float) -> str:
        milliseconds = round(seconds * 1000)
        whole, remainder = divmod(milliseconds, 1000)
        minutes, seconds = divmod(whole, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{remainder:03}"

    def contact_sheet(self, frames: list[Frame], destination: Path) -> Path:
        if not frames or len(frames) > self.MAX_FRAMES:
            raise VidLensError(
                "invalid_frame_count", f"Use 1..{self.MAX_FRAMES} frames in a contact sheet."
            )
        columns = min(3, len(frames))
        tile_width, tile_height, label_height = 600, 338, 34
        rows = math.ceil(len(frames) / columns)
        sheet = Image.new(
            "RGB", (columns * tile_width, rows * (tile_height + label_height)), "#161a23"
        )
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=20)
        for index, frame in enumerate(frames):
            x, y = (index % columns) * tile_width, (index // columns) * (tile_height + label_height)
            with Image.open(frame.path) as image:
                thumbnail = ImageOps.contain(image.convert("RGB"), (tile_width, tile_height))
                sheet.paste(thumbnail, (x + (tile_width - thumbnail.width) // 2, y))
            label = self._timestamp(frame.timestamp)
            if frame.duplicate_of is not None:
                label += "  repeated visual"
            draw.text((x + 10, y + tile_height + 5), label, fill="white", font=font)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(destination)
        return destination

    def _vision_binary(self) -> Path:
        source = Path(__file__).with_name("assets") / "ocr.swift"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        directory = self.cache_dir / "ocr-runtime"
        directory.mkdir(exist_ok=True)
        binary = directory / f"vision-{digest}"
        if binary.is_file():
            return binary
        compiler = shutil.which("swiftc")
        if not compiler:
            raise RuntimeError("Swift compiler unavailable")
        temporary = directory / f"vision-{digest}-{os.getpid()}.tmp"
        command = [
            compiler,
            "-O",
            "-module-cache-path",
            str(directory / "modules"),
            str(source),
            "-o",
            str(temporary),
        ]
        # Use the Command Line Tools' selected SDK symlink. xcrun can otherwise
        # choose a newer side-by-side SDK that its Swift compiler cannot import.
        sdk = os.environ.get("SDKROOT") or "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
        if Path(sdk).is_dir():
            command.extend(["-sdk", sdk])
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode:
            errors = [line for line in result.stderr.splitlines() if "error:" in line]
            detail = errors[0] if errors else result.stderr[-500:].strip()
            raise RuntimeError(f"Vision helper compilation failed: {detail}")
        temporary.replace(binary)
        return binary

    @classmethod
    def _validated_regions(cls, regions) -> list[dict] | None:
        """Validate recognition geometry; these are OCR candidates, not verified facts."""
        if not isinstance(regions, list) or len(regions) > 8192:
            return None
        normalized = []
        for region in regions:
            if not isinstance(region, dict) or not isinstance(region.get("text"), str):
                return None
            box, confidence = region.get("box"), region.get("confidence")
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or not all(cls._finite(value) for value in box)
                    or not cls._finite(confidence) or not 0 <= confidence <= 1):
                return None
            x, y, width, height = box
            if (x < -1e-6 or y < -1e-6 or width <= 0 or height <= 0
                    or x + width > 1 + 1e-6 or y + height > 1 + 1e-6):
                return None
            # Vision can produce tiny floating-point excursions at an image edge.
            x, y = max(0.0, x), max(0.0, y)
            normalized.append({"text": region["text"], "confidence": float(confidence),
                               "box": [x, y, min(width, 1 - x), min(height, 1 - y)]})
        return normalized

    def _ocr(self, frames: list[Frame]) -> None:
        pending: list[Frame] = []
        metadata: dict[Path, dict] = {}
        inputs: dict[Path, Frame] = {}
        identities: dict[Path, str] = {}
        for frame in frames:
            sidecar = self._sidecar(frame.path)
            try:
                info = json.loads(sidecar.read_text())
                if not isinstance(info, dict):
                    info = {}
            except (OSError, ValueError):
                info = {}
            metadata[frame.path] = info
            # Recognition should use retained source pixels, not the smaller
            # image prepared for ordinary model delivery. Boxes are normalized,
            # so their coordinates remain valid on either representation.
            source = self._source_path(frame, info)
            try:
                stat = source.stat()
            except OSError:
                source = frame.path
                stat = source.stat()
            identities[frame.path] = hashlib.sha256(
                f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            ).hexdigest()
            # Keep a legacy transcript if a geometry upgrade cannot run offline.
            frame.ocr = info.get("ocr", frame.ocr)
            regions = self._validated_regions(info.get("ocr_regions"))
            if (info.get("ocr_complete") and info.get("ocr_version") == self.OCR_VERSION
                    and info.get("ocr_input_identity") == identities[frame.path]
                    and regions is not None):
                frame.ocr_regions = regions
            else:
                try:
                    with Image.open(source) as image:
                        width, height = ImageOps.exif_transpose(image).size
                except (OSError, ValueError):
                    source, width, height = frame.path, frame.width, frame.height
                    stat = source.stat()
                    identities[frame.path] = hashlib.sha256(
                        f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
                    ).hexdigest()
                inputs[frame.path] = Frame(frame.timestamp, source, width, height)
                frame.ocr_regions = []
                pending.append(frame)
        if not pending:
            return
        self._ocr_status = "failed"
        recognized: dict[str, dict] = {}
        pending_inputs = list({inputs[frame.path].path: inputs[frame.path]
                               for frame in pending}.values())
        try:
            if platform.system() == "Darwin" and shutil.which("swiftc"):
                binary = self._vision_binary()
                result = subprocess.run(
                    [str(binary), *(str(frame.path) for frame in pending_inputs)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=min(90, 10 + 8 * len(pending)),
                )
                if result.returncode:
                    raise RuntimeError(f"Vision OCR failed: {result.stderr[-300:].strip()}")
                for item in json.loads(result.stdout):
                    if item.get("error"):
                        self._warn(
                            f"OCR unavailable for one frame: {item['error']}. "
                            "If running in a restricted sandbox, macOS Vision may require host access; "
                            "the image is still available."
                        )
                    else:
                        regions = self._validated_regions(item.get("regions"))
                        if regions is None or not isinstance(item.get("text"), str):
                            raise ValueError("OCR returned invalid text geometry")
                        recognized[item["path"]] = {"text": item["text"], "regions": regions}
            elif not shutil.which("tesseract"):
                self._warn(
                    "Local OCR is unavailable; images are still returned. Install Swift on macOS or Tesseract."
                )
        except (OSError, RuntimeError, ValueError, TypeError, KeyError,
                subprocess.TimeoutExpired) as exc:
            self._warn(f"Local OCR failed; images are still returned: {exc}")
        remaining = [frame for frame in pending_inputs if str(frame.path) not in recognized]
        if remaining and (executable := shutil.which("tesseract")):
            recognized.update(self._tesseract(executable, remaining))
        for frame in pending:
            input_path = str(inputs[frame.path].path)
            if input_path in recognized:
                item = recognized[input_path]
                frame.ocr, frame.ocr_regions = item["text"], item["regions"]
                info = metadata[frame.path]
                info.update(ocr=frame.ocr, ocr_regions=frame.ocr_regions,
                            ocr_version=self.OCR_VERSION, ocr_complete=True,
                            ocr_input_identity=identities[frame.path],
                            ocr_input_size=[inputs[frame.path].width, inputs[frame.path].height])
                self._write_json(self._sidecar(frame.path), info)
        if recognized:
            self._ocr_status = "working" if len(recognized) == len(pending_inputs) else "partial"

    @classmethod
    def _parse_tesseract(cls, tsv: str, width: int, height: int) -> dict:
        """Keep word boxes separate so columns are not silently treated as table rows."""
        reader = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
        required = {"level", "page_num", "block_num", "par_num", "line_num", "left", "top",
                    "width", "height", "conf", "text"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Tesseract did not return TSV geometry")
        regions, lines = [], {}
        for row in reader:
            text = row.get("text", "")
            if row.get("level") != "5" or not text or not text.strip():
                continue
            confidence = float(row["conf"]) / 100
            left, top, w, h = [float(row[key]) for key in ("left", "top", "width", "height")]
            regions.append({"text": text.strip(), "confidence": confidence,
                            "box": [left / width, top / height, w / width, h / height]})
            key = tuple(row[name] for name in ("page_num", "block_num", "par_num", "line_num"))
            lines.setdefault(key, []).append(text.strip())
        regions = cls._validated_regions(regions)
        if regions is None:
            raise ValueError("Tesseract returned invalid text geometry")
        return {"text": "\n".join(" ".join(words) for words in lines.values()), "regions": regions}

    def _tesseract(self, executable: str, frames: list[Frame]) -> dict[str, dict]:
        recognized: dict[str, dict] = {}
        deadline = time.monotonic() + 60
        try:
            result = subprocess.run(
                [executable, "--list-langs"],
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
            )
            available = set(result.stdout.split())
            languages = [
                language for language in ("eng", "chi_sim", "chi_tra") if language in available
            ]
            if not any(language.startswith("chi_") for language in languages):
                self._warn("Tesseract Chinese language data is missing; OCR may omit Chinese text.")
            language_args = ["-l", "+".join(languages)] if languages else []
            for frame in frames:
                if time.monotonic() >= deadline:
                    self._warn("Tesseract reached its time budget; OCR is partial.")
                    break
                result = subprocess.run(
                    [executable, str(frame.path), "stdout", "--psm", "11", *language_args, "tsv"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=max(0.1, min(12, deadline - time.monotonic())),
                )
                if result.returncode:
                    self._warn("Tesseract could not read one frame; the image is still available.")
                else:
                    try:
                        recognized[str(frame.path)] = self._parse_tesseract(
                            result.stdout, frame.width, frame.height
                        )
                    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
                        self._warn(f"Tesseract returned unusable geometry for one frame: {exc}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._warn(f"Tesseract failed; images are still returned: {exc}")
        return recognized


def ffmpeg_binary() -> str:
    """Resolve the shared system or packaged FFmpeg executable."""
    return VisualReader._ffmpeg()

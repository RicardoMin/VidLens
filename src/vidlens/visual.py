"""Bounded, on-demand video frames and local OCR on the original timeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
import warnings
from pathlib import Path

from filelock import FileLock, Timeout
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps, ImageStat

from .models import Frame, MediaAsset, VidLensError


class VisualReader:
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

        stat = asset.path.stat()
        identity = f"v2:{asset.path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{asset.offset}"
        directory = self.cache_dir / hashlib.sha256(identity.encode()).hexdigest()[:24]
        directory.mkdir(parents=True, exist_ok=True)
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
                info["timestamp"], path, info["width"], info["height"], info.get("ocr", "")
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
    ) -> Frame:
        local = max(0.0, target - asset.offset)
        path = directory / f"frame-{local:.6f}.jpg"
        metadata = self._sidecar(path)
        cached = self._cached_frame(path)
        if cached is not None:
            return cached
        original = directory / f"frame-{local:.6f}.source.jpg"
        # With input seeking, showinfo PTS is relative to the seek target. Adding
        # that residual retains real frame time instead of labelling the requested time.
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostdin",
            "-y",
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
            "showinfo,scale=w='min(3840,iw)':h='min(2160,ih)':force_original_aspect_ratio=decrease",
            "-q:v",
            "2",
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
        if result.returncode or not original.is_file() or not matches:
            raise VidLensError(
                "frame_decode_failed", "FFmpeg did not return a timestamped video frame."
            )
        timestamp = round(asset.offset + local + float(matches[0]), 6)
        try:
            with Image.open(original) as image:
                width, height = image.size
                if width > self.DISPLAY_SIZE[0] or height > self.DISPLAY_SIZE[1]:
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

    def crop(self, frame: Frame, box: tuple[float, float, float, float]) -> Frame:
        """Crop normalized x/y/width/height from the cached higher-resolution source."""
        self.last_warnings = []
        if len(box) != 4 or not all(self._finite(value) for value in box):
            raise VidLensError(
                "invalid_crop", "Crop must contain finite normalized x, y, width, height."
            )
        x, y, width, height = box
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise VidLensError("invalid_crop", "Crop must lie inside the normalized [0, 1] image.")
        source = Path(frame.path)
        sidecar = self._sidecar(source)
        if sidecar.is_file():
            info = json.loads(sidecar.read_text())
            original = Path(info.get("source", str(source)))
            if original.is_file():
                source = original
        try:
            stat = source.stat()
        except OSError as exc:
            raise VidLensError(
                "crop_failed", "The source image is unavailable; read the frames again."
            ) from exc
        key = f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{tuple(map(float, box))}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        directory = self.cache_dir / "crops"
        directory.mkdir(exist_ok=True)
        destination = directory / f"{digest}.jpg"
        try:
            with FileLock(directory / f"{digest}.lock", timeout=self.MAX_READ_SECONDS):
                result = self._cached_frame(destination)
                if result is None:
                    result = self._crop_image(source, destination, frame.timestamp, box)
                result.selection = "crop"
                self._ocr([result])
                return result
        except Timeout as exc:
            raise VidLensError(
                "frame_cache_busy", "Another reader is working on this crop; retry shortly."
            ) from exc

    def _crop_image(
        self,
        source: Path,
        destination: Path,
        timestamp: float,
        box: tuple[float, float, float, float],
    ) -> Frame:
        x, y, width, height = box
        original = destination.with_suffix(".source.jpg")
        try:
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                left, top = int(x * image.width), int(y * image.height)
                right = math.ceil((x + width) * image.width)
                bottom = math.ceil((y + height) * image.height)
                image = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
                if image.width > self.DISPLAY_SIZE[0] or image.height > self.DISPLAY_SIZE[1]:
                    image.save(original, quality=95)
                    image.thumbnail(self.DISPLAY_SIZE, Image.Resampling.LANCZOS)
                else:
                    original = destination
                image.save(destination, quality=95)
                result = Frame(timestamp, destination, *image.size, selection="crop")
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
                "crop": list(box),
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

    def _ocr(self, frames: list[Frame]) -> None:
        pending: list[Frame] = []
        metadata: dict[Path, dict] = {}
        for frame in frames:
            sidecar = self._sidecar(frame.path)
            info = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            metadata[frame.path] = info
            if info.get("ocr_complete"):
                frame.ocr = info.get("ocr", "")
            else:
                pending.append(frame)
        if not pending:
            return
        self._ocr_status = "failed"
        recognized: dict[str, str] = {}
        try:
            if platform.system() == "Darwin" and shutil.which("swiftc"):
                binary = self._vision_binary()
                result = subprocess.run(
                    [str(binary), *(str(frame.path) for frame in pending)],
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
                        recognized[item["path"]] = item["text"]
            elif not shutil.which("tesseract"):
                self._warn(
                    "Local OCR is unavailable; images are still returned. Install Swift on macOS or Tesseract."
                )
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            self._warn(f"Local OCR failed; images are still returned: {exc}")
        remaining = [frame for frame in pending if str(frame.path) not in recognized]
        if remaining and (executable := shutil.which("tesseract")):
            recognized.update(self._tesseract(executable, remaining))
        for frame in pending:
            if str(frame.path) in recognized:
                frame.ocr = recognized[str(frame.path)]
                info = metadata[frame.path]
                info.update(ocr=frame.ocr, ocr_complete=True)
                self._write_json(self._sidecar(frame.path), info)
        if recognized:
            self._ocr_status = "working" if len(recognized) == len(pending) else "partial"

    def _tesseract(self, executable: str, frames: list[Frame]) -> dict[str, str]:
        recognized: dict[str, str] = {}
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
                    [executable, str(frame.path), "stdout", "--psm", "11", *language_args],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=max(0.1, min(12, deadline - time.monotonic())),
                )
                if result.returncode:
                    self._warn("Tesseract could not read one frame; the image is still available.")
                else:
                    recognized[str(frame.path)] = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._warn(f"Tesseract failed; images are still returned: {exc}")
        return recognized


def ffmpeg_binary() -> str:
    """Resolve the shared system or packaged FFmpeg executable."""
    return VisualReader._ffmpeg()

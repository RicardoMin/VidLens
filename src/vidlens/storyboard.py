"""Whole-timeline scene indexing without materializing full-resolution images."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from filelock import FileLock, Timeout

from .models import MediaAsset, VidLensError


class StoryboardScanner:
    """Decode every frame at low resolution and retain time-ordered candidates.

    Pagination bounds the response, never the timeline being scanned. A candidate
    is a pointer for subsequent high-resolution inspection, not a semantic claim.
    """

    VERSION = 2
    WIDTH = 640
    HEIGHT = 360
    ANCHOR_SECONDS = 10.0
    CHANGE_THRESHOLD = 0.000001
    SCENE_THRESHOLD = 0.08
    CHANGE_WINDOW = 0.35
    MAX_SCAN_SECONDS = 900
    # Three compact channel-difference streams, including their PTS headers.
    # 128 MiB accommodates a 116-minute 60 fps lecture without unbounded output.
    MAX_OUTPUT_BYTES = 128 * 1024 * 1024
    MAX_ERROR_BYTES = 1024 * 1024
    MAX_PAGE_SIZE = 1000
    _STAMP = re.compile(rb"^frame:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:")

    def __init__(self, cache_dir: Path, executable):
        self.cache_dir = Path(cache_dir) / "storyboards"
        self.executable = executable

    @staticmethod
    def _finite(value):
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )

    def scan(self, asset: MediaAsset, *, start=None, end=None, cursor=0, limit=200) -> dict:
        if not all(self._finite(value) for value in (asset.offset, asset.duration)):
            raise VidLensError("invalid_asset", "Media offset and duration must be finite.")
        if asset.offset < 0 or asset.duration < 0:
            raise VidLensError("invalid_asset", "Media offset and duration cannot be negative.")
        start = asset.offset if start is None else start
        end = asset.offset + asset.duration if end is None else end
        if not all(self._finite(value) for value in (start, end)) or not 0 <= start < end:
            raise VidLensError("invalid_interval", "Use finite times with 0 <= start < end.")
        if start < asset.offset - 1e-6 or (
            asset.duration and end > asset.offset + asset.duration + 1e-6
        ):
            raise VidLensError("interval_outside_asset", "The scan interval is outside this asset.")
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or cursor < 0
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= self.MAX_PAGE_SIZE
        ):
            raise VidLensError(
                "invalid_scan_page",
                f"Use cursor >= 0 and limit between 1 and {self.MAX_PAGE_SIZE}.",
            )
        if asset.audio_only:
            raise VidLensError("no_video_stream", "This media asset contains audio only.")
        path = Path(asset.path)
        if not path.is_file():
            raise VidLensError("media_missing", "The local media asset does not exist.")
        stat = path.stat()
        identity = (
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            asset.offset,
            asset.duration,
        )
        rules = {
            "method": "all_decoded_frames_ffmpeg_signalstats",
            "scan_width": self.WIDTH,
            "scan_height": self.HEIGHT,
            "anchor_seconds": self.ANCHOR_SECONDS,
            "small_change_threshold": self.CHANGE_THRESHOLD,
            "small_change_metric": "max_Y_U_V_mean_absolute_difference_divided_by_255",
            "scene_change_threshold": self.SCENE_THRESHOLD,
            "change_merge_window_seconds": self.CHANGE_WINDOW,
            "merge_rule": "first and last nearby small change; preserve scene cuts and anchors",
            "end_rule": "actual last decoded frame",
            "limitations": "Low-resolution visual change candidates require image inspection; "
            "they do not establish semantic coverage or detect every small text change.",
        }
        key = json.dumps([self.VERSION, identity, float(start), float(end), rules], sort_keys=True)
        scan_id = hashlib.sha256(key.encode()).hexdigest()[:32]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.cache_dir / f"{scan_id}.json"
        result = {
            "status": "failed",
            "scan_id": scan_id,
            "range": {"start": start, "end": end},
            "rules": rules,
            "coverage": {
                "complete": False,
                "decoded_frames": 0,
                "first_timestamp": None,
                "last_timestamp": None,
            },
            "candidates": [],
            "warnings": [],
        }
        cached = False
        try:
            with FileLock(str(manifest) + ".lock", timeout=self.MAX_SCAN_SECONDS + 5):
                previous = self._load(manifest, scan_id)
                if previous is not None:
                    result, cached = previous, True
                else:
                    try:
                        result = self._decode(asset, start, end, result)
                    except VidLensError as exc:
                        result["error"] = exc.to_dict()
                        result["warnings"].append(exc.message)
                    latest = path.stat()
                    if (latest.st_size, latest.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                        result["status"] = "partial"
                        result["coverage"]["complete"] = False
                        result["warnings"].append("The source file changed during the scan; retry.")
                    if result["status"] == "complete":
                        self._save(manifest, result)
        except Timeout:
            result["error"] = {
                "code": "scan_cache_busy",
                "message": "Another scanner is working on this media; retry shortly.",
            }
        except OSError as exc:
            result["status"] = "partial" if result["candidates"] else "failed"
            result["coverage"]["complete"] = False
            result["error"] = {"code": "scan_io_failed", "message": str(exc)}
        candidates = result["candidates"]
        total = len(candidates)
        return {
            **result,
            "cached": cached,
            "manifest_path": str(manifest) if result["status"] == "complete" else None,
            "candidate_count": total,
            "cursor": cursor,
            "candidates": candidates[cursor : cursor + limit],
            "next_cursor": cursor + limit if cursor + limit < total else None,
        }

    @staticmethod
    def _load(path, scan_id):
        try:
            data = json.loads(path.read_text())
            if (
                data["scan_id"] == scan_id
                and data["status"] == "complete"
                and data["coverage"]["complete"] is True
                and isinstance(data["warnings"], list)
                and isinstance(data["candidates"], list)
                and data["candidates"]
                and all(
                    isinstance(row, dict)
                    and StoryboardScanner._finite(row.get("timestamp"))
                    and StoryboardScanner._finite(row.get("score"))
                    and isinstance(row.get("reasons"), list)
                    for row in data["candidates"]
                )
            ):
                return data
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    @staticmethod
    def _save(path, data):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(data, ensure_ascii=False))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _decode(self, asset, start, end, result):
        # AVTB supplies microsecond PTS; pts_time's display precision is
        # insufficient on long recordings. copyts/start_at_zero preserves source
        # times without adding the rounding error of an input seek's time base.
        # scdet rounds raw difference to three decimals and measures only luma.
        # signalstats retains sub-pixel-average precision and chroma changes.
        # Print only the three differences; full histograms cost ~6x more bytes.
        graph = (
            f"settb=AVTB,scale=w='min({self.WIDTH},iw)':h='min({self.HEIGHT},ih)':"
            "force_original_aspect_ratio=decrease:force_divisible_by=2:flags=area,"
            "format=yuv420p,signalstats,"
            + ",".join(
                f"metadata=mode=print:key=lavfi.signalstats.{channel}DIF:file='pipe\\:1':direct=1"
                for channel in "YUV"
            )
        )
        command = [
            self.executable(),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-copyts",
            "-start_at_zero",
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
            "-fps_mode",
            "passthrough",
            "-threads",
            "1",
            "-f",
            "null",
            "-",
        ]
        candidates = result["candidates"]
        coverage = result["coverage"]
        next_anchor, pending_stamp, pending_difference = start, None, 0.0
        last_score, previous_stamp, last_gap = 0.0, None, 0.0
        cluster_start, cluster_first = None, None

        def offer(timestamp, score, reason):
            nonlocal cluster_start, cluster_first
            if candidates and abs(candidates[-1]["timestamp"] - timestamp) < 1e-6:
                if reason not in candidates[-1]["reasons"]:
                    candidates[-1]["reasons"].append(reason)
                candidates[-1]["score"] = max(candidates[-1]["score"], score)
                return
            if (
                reason == "small_change"
                and candidates
                and candidates[-1]["reasons"] == ["small_change"]
                and cluster_start is not None
                and timestamp - cluster_start < self.CHANGE_WINDOW
            ):
                if len(candidates) - 1 == cluster_first:
                    candidates.append({"timestamp": timestamp, "score": score, "reasons": [reason]})
                else:
                    candidates[-1].update(timestamp=timestamp, score=score, reasons=[reason])
                return
            candidates.append({"timestamp": timestamp, "score": score, "reasons": [reason]})
            cluster_start = timestamp if reason == "small_change" else None
            cluster_first = len(candidates) - 1 if reason == "small_change" else None

        def consume(line):
            nonlocal pending_stamp, pending_difference, next_anchor
            nonlocal last_score, previous_stamp, last_gap
            if match := self._STAMP.match(line):
                stamp = round(asset.offset + int(match[1]) / 1_000_000, 6)
                if stamp != pending_stamp:
                    pending_stamp, pending_difference = stamp, 0.0
            elif line.startswith((b"lavfi.signalstats.YDIF=", b"lavfi.signalstats.UDIF=")):
                pending_difference = max(pending_difference, float(line.split(b"=", 1)[1]) / 255)
            elif line.startswith(b"lavfi.signalstats.VDIF=") and pending_stamp is not None:
                score = max(pending_difference, float(line.split(b"=", 1)[1]) / 255)
                scene_score = min(score, abs(score - last_score))
                timestamp, pending_stamp = pending_stamp, None
                if not math.isfinite(score) or not start - 1e-6 <= timestamp < end:
                    return
                if previous_stamp is not None:
                    if timestamp < previous_stamp:
                        raise VidLensError("scan_timestamp_order", "Decoded frame times decreased.")
                    last_gap = timestamp - previous_stamp
                else:
                    coverage["first_timestamp"] = timestamp
                    offer(timestamp, score, "start")
                previous_stamp, last_score = timestamp, score
                coverage["decoded_frames"] += 1
                coverage["last_timestamp"] = timestamp
                if timestamp + 1e-6 >= next_anchor:
                    offer(timestamp, score, "periodic_anchor")
                    next_anchor = (
                        start
                        + (math.floor((timestamp - start) / self.ANCHOR_SECONDS) + 1)
                        * self.ANCHOR_SECONDS
                    )
                if score >= self.CHANGE_THRESHOLD:
                    reason = (
                        "scene_change" if scene_score >= self.SCENE_THRESHOLD else "small_change"
                    )
                    offer(timestamp, score, reason)

        error = None
        started, byte_count, buffer = time.monotonic(), 0, b""
        with tempfile.TemporaryFile() as errors:
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
            except OSError as exc:
                raise VidLensError("scan_start_failed", "FFmpeg could not start the scan.") from exc
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while True:
                        if time.monotonic() - started > self.MAX_SCAN_SECONDS:
                            raise VidLensError("scan_timeout", "The whole-timeline scan timed out.")
                        if os.fstat(errors.fileno()).st_size > self.MAX_ERROR_BYTES:
                            raise VidLensError(
                                "scan_output_limit", "FFmpeg emitted too many errors."
                            )
                        if not selector.select(timeout=0.1):
                            continue
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            break
                        byte_count += len(chunk)
                        if byte_count > self.MAX_OUTPUT_BYTES:
                            raise VidLensError(
                                "scan_output_limit", "The scan metadata limit was reached."
                            )
                        buffer += chunk
                        lines = buffer.split(b"\n")
                        buffer = lines.pop()
                        if len(buffer) > 4096:
                            raise VidLensError(
                                "scan_output_invalid", "Invalid frame metadata from FFmpeg."
                            )
                        for line in lines:
                            consume(line)
                    if buffer:
                        consume(buffer)
                process.wait(timeout=max(0.1, self.MAX_SCAN_SECONDS - (time.monotonic() - started)))
            except (VidLensError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
                error = (
                    exc
                    if isinstance(exc, VidLensError)
                    else VidLensError(
                        "scan_decode_failed", "Could not finish decoding the scan interval."
                    )
                )
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                process.stdout.close()
            errors.seek(0)
            detail = errors.read(2000).decode("utf-8", errors="replace").strip()
        if previous_stamp is not None:
            offer(previous_stamp, last_score, "end")
        if error is None and (process.returncode or detail):
            error = VidLensError(
                "scan_decode_failed", "FFmpeg reported errors during the scan.", detail
            )
        if error is None and previous_stamp is None:
            error = VidLensError("scan_no_frames", "No video frames were decoded for the scan.")
        if error is None and end - previous_stamp > max(1.0, last_gap * 2 + 0.001):
            error = VidLensError(
                "scan_incomplete_range", "The video stream ended before the requested scan range."
            )
        if error:
            result["status"] = "partial" if coverage["decoded_frames"] else "failed"
            result["error"] = error.to_dict()
            result["warnings"].append(error.message)
        else:
            result["status"] = "complete"
            coverage["complete"] = True
        result["scan_stats"] = {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "metadata_bytes": byte_count,
        }
        return result

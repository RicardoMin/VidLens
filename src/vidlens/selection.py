"""A compact visual navigation layer above the lossless candidate timeline.

Pixel heuristics prioritize legible, different states. They never establish what
a diagram means, and the scanner's full candidate list remains available.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import statistics
import subprocess
import tempfile
import time
from itertools import pairwise
from pathlib import Path

from filelock import FileLock, Timeout
from PIL import Image, ImageFilter, ImageStat

from .models import VidLensError
from .storyboard import StoryboardScanner


class VisualSelector:
    VERSION = 5
    PROBE_SECONDS = 3.0
    MAX_PROBES = 1200
    WIDTH, HEIGHT = 160, 90
    MAX_SECONDS = 450

    def __init__(self, cache_dir: Path, executable):
        self.cache_dir = Path(cache_dir) / "visual-selection"
        self.executable = executable

    def build(self, asset, scan):
        """Return private pixel features plus a time-ordered navigation index.

        Strip ``signature`` before transmitting rows to an AI client. It is a
        compact local RGB descriptor used by ``representatives`` only.
        """
        try:
            return self._build(asset, scan)
        except Timeout as exc:
            raise VidLensError("selection_cache_busy", "Another reader is preparing visual navigation.") from exc
        except OSError as exc:
            raise VidLensError("selection_io_failed", "Visual navigation could not access its media or cache.") from exc

    def _build(self, asset, scan):
        if scan.get("status") != "complete":
            raise VidLensError("selection_incomplete_scan", "Finish the full scan before prioritizing it.")
        stat = Path(asset.path).stat()
        identity = [self.VERSION, scan["scan_id"], str(Path(asset.path).resolve()),
                    stat.st_size, stat.st_mtime_ns, asset.offset, asset.duration]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:32]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        with FileLock(str(path) + ".lock", timeout=self.MAX_SECONDS + 5):
            try:
                cached = json.loads(path.read_text())
                rows = cached["navigation_candidates"]
                if (cached["selection_id"] == key and cached["status"] == "complete"
                    and isinstance(rows, list) and rows
                    and all(isinstance(row, dict)
                            and isinstance(row["source_candidate_cursor"], int)
                            and 0 <= row["source_candidate_cursor"] < len(scan["candidates"])
                            and all(isinstance(row[field], (int, float))
                                    and not isinstance(row[field], bool)
                                    and math.isfinite(row[field]) for field in (
                                        "timestamp", "source_timestamp", "detail", "contrast",
                                        "score", "hold_seconds"))
                            and isinstance(row["blank"], bool) and isinstance(row["protected"], bool)
                            and len(base64.b64decode(row["signature"], validate=True)) == 64 * 36 * 3
                            for row in rows)):
                    return {**cached, "cached": True}
            except (OSError, ValueError, KeyError, TypeError):
                pass
            started = time.monotonic()
            probes = self.probes(scan)
            if not probes:
                raise VidLensError("selection_no_candidates", "No visual candidates to prioritize.")
            rows = self._decode(asset, probes)
            navigation = self._navigation(rows)
            latest = Path(asset.path).stat()
            if (latest.st_size, latest.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise VidLensError("stale_revision", "The media changed during visual selection.")
            result = {
                "status": "complete", "selection_id": key, "version": self.VERSION,
                "method": "stable_states_pixel_detail_and_local_diversity",
                "source_scan_id": scan["scan_id"], "range": scan["range"],
                "navigation_candidates": navigation,
                "metrics": {"raw_candidate_count": len(scan["candidates"]),
                            "probe_count": len(probes), "decoded_probe_count": len(rows),
                            "navigation_count": len(navigation),
                            "blank_probes": sum(row["blank"] for row in rows),
                            "elapsed_seconds": round(time.monotonic() - started, 3)},
                "limitations": "Visual priorities are pixel heuristics, not semantic coverage. "
                "Use the full candidate view for brief details, code changes and close inspection.",
                "cached": False,
            }
            StoryboardScanner._save(path, result)
            return result

    @classmethod
    def probes(cls, scan):
        candidates = scan["candidates"]
        start, end = scan["range"]["start"], scan["range"]["end"]
        if not candidates:
            return []
        # Keep work finite for long courses while retaining temporal coverage.
        width = max(cls.PROBE_SECONDS, (end - start) / (cls.MAX_PROBES * 0.75))
        gaps = [b["timestamp"] - a["timestamp"] for a, b in pairwise(candidates)
                if b["timestamp"] - a["timestamp"] > 0.00001]
        brief_limit = max(0.12, min(gaps, default=1 / 30) * 1.5)
        buckets, protected = {}, {}
        for index, candidate in enumerate(candidates):
            stamp = candidate["timestamp"]
            previous = candidates[index - 1]["timestamp"] if index else start
            following = candidates[index + 1]["timestamp"] if index + 1 < len(candidates) else end
            hold = max(0, following - stamp)
            isolated_brief = index > 0 and stamp - previous >= 0.65 and hold <= brief_limit
            neighbors = [other["score"] for j, other in enumerate(
                candidates[max(0, index - 5):index + 6], start=max(0, index - 5)
            ) if j != index]
            background = statistics.median(neighbors) if neighbors else 0
            relative_change = candidate["score"] / max(background, 0.000001)
            next_score = candidates[index + 1]["score"] if index + 1 < len(candidates) else 0
            return_ratio = next_score / max(candidate["score"], 0.000001)
            brief_outlier = (hold <= brief_limit and relative_change >= 4
                             and 0.5 <= return_ratio <= 2)
            special = isolated_brief or brief_outlier or "scene_change" in candidate["reasons"]
            row = {**candidate, "source_candidate_cursor": index, "source_timestamp": stamp,
                   "hold_seconds": round(hold, 6), "protected": special}
            row["relative_change"] = round(relative_change, 4)
            # Exact short events must not advance into the following state.
            row["timestamp"] = round(stamp + (
                0 if special or index + 1 == len(candidates) else min(0.35, hold / 3)
            ), 6)
            if len(candidates) <= 12:
                row["timestamp"] = stamp
                row["protected"] = True
                protected[index] = row
            elif special:
                protected[index] = row
            bucket = int((stamp - start) / width)
            midpoint = start + (bucket + 0.5) * width
            rank = (hold >= 0.6, min(hold, 3), -abs(stamp - midpoint))
            if bucket not in buckets or rank > buckets[bucket][0]:
                buckets[bucket] = (rank, row)
        chosen = {row["source_candidate_cursor"]: row for _, row in buckets.values()}
        chosen.update(protected)
        if len(chosen) > cls.MAX_PROBES:
            # This only bounds the thumbnail analysis, never the raw scan.
            # Report incomplete prioritization instead of silently dropping events.
            raise VidLensError("selection_probe_limit", "Too many distinct short visual events; "
                               "use the full candidate view for this video.")
        return sorted(chosen.values(), key=lambda row: row["timestamp"])

    def _decode(self, asset, probes):
        # Each threshold emits its first decoded frame. close timestamps mapping
        # to one displayed frame share the same pixels, while source cursors stay.
        targets = [max(0, row["timestamp"] - asset.offset) for row in probes]
        terms = [
            f"gte(t,{stamp:.6f})*(isnan(prev_selected_t)+lt(prev_selected_t,{stamp:.6f}))"
            for stamp in targets
        ]

        def balanced_sum(items):
            if len(items) == 1:
                return items[0]
            middle = len(items) // 2
            return f"({balanced_sum(items[:middle])}+{balanced_sum(items[middle:])})"

        # FFmpeg's expression parser has a finite recursion depth. A flat sum of
        # hundreds of terms becomes a deep left-associated tree.
        expression = balanced_sum(terms)
        graph = (f"settb=AVTB,select='{expression}',scale={self.WIDTH}:{self.HEIGHT}:flags=area,"
                 "format=rgb24,showinfo=checksum=0")
        command = [self.executable(), "-hide_banner", "-nostdin", "-loglevel", "info",
                   "-copyts", "-start_at_zero", "-i", str(asset.path), "-map", "0:v:0",
                   "-an", "-sn", "-vf", graph, "-frames:v", str(len(probes)),
                   "-fps_mode", "passthrough", "-threads", "1", "-f", "rawvideo", "pipe:1"]
        frame_bytes = self.WIDTH * self.HEIGHT * 3
        with tempfile.TemporaryFile() as pixels, tempfile.TemporaryFile() as errors:
            try:
                process = subprocess.run(command, stdout=pixels, stderr=errors,
                                         check=False, timeout=self.MAX_SECONDS)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise VidLensError("selection_decode_failed", "Could not analyze visual candidates.") from exc
            if pixels.tell() > frame_bytes * len(probes) or errors.tell() > 8 * 1024 * 1024:
                raise VidLensError("selection_output_limit", "Visual selection exceeded its output bound.")
            errors.seek(0)
            details = errors.read().decode("utf-8", errors="replace")
            stamps = [asset.offset + int(stamp) / 1_000_000 for stamp in re.findall(
                r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:", details
            )]
            if process.returncode or not stamps or pixels.tell() != frame_bytes * len(stamps):
                raise VidLensError("selection_decode_failed", "Visual candidate pixels and times did not align.")
            pixels.seek(0)
            pictures = [(stamp, self._features(Image.frombytes(
                "RGB", (self.WIDTH, self.HEIGHT), pixels.read(frame_bytes)))) for stamp in stamps]
        # A target after the final decoded frame can still lie in its display
        # hold. Use the nearest preceding decoded probe only inside its own hold.
        rows, picture_index = [], 0
        for probe in probes:
            while picture_index + 1 < len(pictures) and pictures[picture_index][0] + 1e-6 < probe["timestamp"]:
                picture_index += 1
            stamp, features = pictures[picture_index]
            if stamp + 1e-6 < probe["timestamp"]:
                continue
            rows.append({**probe, **features, "timestamp": round(stamp, 6)})
        if not rows:
            raise VidLensError("selection_no_frames", "No candidate thumbnails could be analyzed.")
        return rows

    @staticmethod
    def _features(image):
        gray = image.convert("L")
        inner = gray.crop((2, 2, gray.width - 2, gray.height - 2))
        contrast = ImageStat.Stat(inner).stddev[0] / 128
        edge = inner.filter(ImageFilter.FIND_EDGES)
        edge = edge.crop((1, 1, edge.width - 1, edge.height - 1))
        detail = ImageStat.Stat(edge).mean[0] / 32
        return {"signature": base64.b64encode(image.resize((64, 36)).tobytes()).decode(),
                "detail": round(min(1, detail), 5), "contrast": round(min(1, contrast), 5),
                "blank": contrast < 0.025 and detail < 0.015}

    @staticmethod
    def distance(first, second):
        left, right = base64.b64decode(first["signature"]), base64.b64decode(second["signature"])
        differences = [abs(a - b) for a, b in zip(left, right)]
        mean = sum(differences) / (len(left) * 255)
        # A changed formula can occupy a small part of an otherwise blank slide.
        # Global mean alone would collapse different equations into one state.
        local = max(
            sum(differences[(y * 64 + x) * 3 + channel]
                for y in range(top, top + 6) for x in range(column, column + 8)
                for channel in range(3)) / (6 * 8 * 3 * 255)
            for top in range(0, 36, 6) for column in range(0, 64, 8)
        )
        return max(mean, local * 0.75)

    @classmethod
    def _quality(cls, row):
        return (0.5 * row["detail"] + 0.15 * row["contrast"]
                + 0.25 * min(row["hold_seconds"], 3) / 3
                + 0.1 * min(row["score"] * 20, 1))

    @classmethod
    def _navigation(cls, rows):
        selected = []
        for row in rows:
            if row["blank"] and not row["protected"]:
                continue
            # Keep evidence of isolated short states even if they resemble the
            # surrounding diagram; a single changed symbol may be the lesson.
            if selected and not row["protected"] and not selected[-1]["protected"]:
                previous = selected[-1]
                if row["timestamp"] - previous["timestamp"] <= 30 and cls.distance(row, previous) < 0.012:
                    if cls._quality(row) > cls._quality(previous) + 0.015:
                        selected[-1] = row
                    continue
            selected.append(row)
        return selected or rows[:1]

    @classmethod
    def representatives(cls, index, start, end, limit):
        candidates = [row for row in index["navigation_candidates"] if start <= row["timestamp"] < end]
        if len(candidates) <= limit:
            return candidates
        width = (end - start) / limit
        selected = []
        for slot in range(limit):
            left, right = start + slot * width, start + (slot + 1) * width
            choices = [row for row in candidates if left <= row["timestamp"] < right]
            if not choices:
                continue
            midpoint = (left + right) / 2

            def score(row, midpoint=midpoint):
                novelty = min((cls.distance(row, previous) for previous in selected), default=0.08)
                center = 1 - abs(row["timestamp"] - midpoint) / width
                return (not row["blank"], cls._quality(row) + min(novelty, 0.15) * 2 + center * 0.1)

            selected.append(max(choices, key=score))
        # A retained state can span a time bucket with no new navigation event.
        # Fill the remaining slots with distinct states instead of returning a
        # misleadingly small image budget.
        while len(selected) < min(limit, len(candidates)):
            choices = [row for row in candidates if row not in selected]
            selected.append(max(choices, key=lambda row: (
                not row["blank"], cls._quality(row) + 2 * min(
                    (cls.distance(row, other) for other in selected), default=0.08
                ),
            )))
        return sorted(selected, key=lambda row: row["timestamp"])

    @staticmethod
    def public(row):
        return {key: value for key, value in row.items() if key != "signature"}

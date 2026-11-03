"""Measure local visual sampling with fresh frame caches and immediate cache hits."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from vidlens.models import MediaAsset
from vidlens.visual import VisualReader


def fixture(directory):
    path = directory / "slides.mp4"
    for index in range(16):
        image = Image.new("RGB", (1280, 720), "#151b24")
        draw = ImageDraw.Draw(image)
        draw.text((62, 80), "Technical lecture", font_size=48)
        draw.text((62, 165), f"batch_size = {64 if index < 8 else 128}", font_size=32)
        if index == 10:
            draw.rectangle((300, 180, 420, 215), outline="orange", width=4)
        image.save(directory / f"in-{index:02d}.png")
    subprocess.run(
        [
            VisualReader._ffmpeg(),
            "-v",
            "error",
            "-framerate",
            "2",
            "-i",
            str(directory / "in-%02d.png"),
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-y",
            str(path),
        ],
        check=True,
        timeout=60,
    )
    return path


def measure(reader_type, path, offset, duration, count, trials):
    times = []
    for _ in range(trials):
        with tempfile.TemporaryDirectory(prefix="vidlens-visual-cache-") as temp:
            cache = Path(temp)
            reader = reader_type(cache)
            runs = []
            for _ in range(2):
                with patch("subprocess.run", wraps=subprocess.run) as runner:
                    started = time.perf_counter()
                    frames = reader.read(
                        MediaAsset(path, offset, duration),
                        offset,
                        offset + duration,
                        max_frames=count,
                        strategy="changes",
                        ocr=False,
                    )
                    elapsed = time.perf_counter() - started
                    runs.append(
                        {"seconds": round(elapsed, 5), "ffmpeg_processes": runner.call_count}
                    )
            files = [p for p in cache.rglob("*") if p.is_file()]
            times.append(
                {
                    "cold": runs[0],
                    "warm": runs[1],
                    "cached_bytes": sum(p.stat().st_size for p in files),
                    "cached_jpegs": sum(p.suffix == ".jpg" for p in files),
                    "frames": [
                        {"timestamp": f.timestamp, "duplicate_of": f.duplicate_of} for f in frames
                    ],
                }
            )
    return {
        "media_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "offset": offset,
        "duration": duration,
        "returned_frame_limit": count,
        "trials": times,
        "median_cold_seconds": statistics.median(t["cold"]["seconds"] for t in times),
        "median_warm_seconds": statistics.median(t["warm"]["seconds"] for t in times),
    }


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--media", type=Path, help="Defaults to generated technical slides.")
    arguments.add_argument("--offset", type=float, default=0)
    arguments.add_argument("--duration", type=float, default=8)
    arguments.add_argument("--max-frames", type=int, default=4)
    arguments.add_argument("--trials", type=int, default=3)
    arguments.add_argument(
        "--reader-file", type=Path, help="Optional saved vidlens.visual source for comparison."
    )
    args = arguments.parse_args()
    reader_type = VisualReader
    if args.reader_file:
        spec = importlib.util.spec_from_file_location("vidlens.benchmark_reader", args.reader_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reader_type = module.VisualReader
    with tempfile.TemporaryDirectory(prefix="vidlens-visual-fixture-") as temp:
        media = args.media or fixture(Path(temp))
        print(
            json.dumps(
                measure(
                    reader_type, media, args.offset, args.duration, args.max_frames, args.trials
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

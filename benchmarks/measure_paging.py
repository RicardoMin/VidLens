"""Time the first bounded page of a 600-cue synthetic transcript in an isolated database."""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from vidlens.models import Segment, Video
from vidlens.service import VideoService


def measure(home):
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(home, "tokenizer").resolve()
        if cache.is_dir():
            (Path(directory) / "tokenizer").symlink_to(cache, target_is_directory=True)
        service = VideoService(directory)
        cues = [
            Segment(
                4 * i,
                4 * i + 4,
                f"第{i}步：计算 attention 后检查当前权重。Check the previous input, then record the result.",
            )
            for i in range(600)
        ]
        service.store.save(Video("test", "", "local", "Lesson", 2400), cues, [])
        runs = []
        for _ in range(3):
            started = time.perf_counter()
            page = service.read_interval("test", 0, 2400, 1600)
            runs.append(time.perf_counter() - started)
        return {
            "seconds": runs,
            "median": statistics.median(runs),
            "items": len(page["evidence"]),
            "tokens": page["accounting"]["text_tokens_estimate"],
            "method": service.budget.method,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(".vidlens"))
    print(json.dumps(measure(parser.parse_args().home)))

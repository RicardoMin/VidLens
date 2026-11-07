"""The protocol benchmark reads an immutable source and publishes no transcript text."""

import json
import runpy
from pathlib import Path

import pytest

from vidlens.models import Segment, Video
from vidlens.store import Store

EVALUATE = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks" /
                            "evaluate_compact_reading.py"))["evaluate"]


def test_protocol_benchmark_is_lossless_read_only_and_redacts_local_material(tmp_path):
    store = Store(tmp_path / "private-library")
    video = Video("fixture", "https://example.org/video", "test", "Local fixture", 100)
    store.save(video, [Segment(index, index + 1, f"DO_NOT_PUBLISH_TRANSCRIPT_{index}",
                               kind="transcript", source="local") for index in range(100)], [])
    before = store.path.read_bytes()
    result = EVALUATE(store.home, video.id, max_tokens=1800)
    assert store.path.read_bytes() == before
    assert store.deliveries(video.id) == []
    assert result["speech_segments"] == 100
    assert all(mode["exact_source_reconstruction"] for mode in result["results"].values())
    assert all(mode["source_characters_reconstructed"] == result["source_characters"]
               for mode in result["results"].values())
    assert result["results"]["compact"]["pages"] < result["results"]["verbose"]["pages"]
    output = json.dumps(result)
    assert "DO_NOT_PUBLISH_TRANSCRIPT" not in output
    assert str(tmp_path) not in output
    assert result["model_usage"] == "not_measured"


def test_protocol_benchmark_does_not_create_missing_source_cache(tmp_path):
    missing = tmp_path / "not-acquired"
    with pytest.raises(ValueError, match="existing evidence cache"):
        EVALUATE(missing, "unknown")
    assert not missing.exists()

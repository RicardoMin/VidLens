"""Protect the measured SYN result from substring and timing mistakes."""

import runpy
from pathlib import Path

import pytest

benchmark = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/evaluate_quality.py"))


@pytest.mark.parametrize("text,expected", [("SYN", True), ("S Y N", True),
                                          ("Sync", False), ("Synchronization", False)])
def test_syn_anchor_does_not_count_a_different_english_word(text, expected):
    assert benchmark["contains_term"](text, "SYN") is expected


def test_coarse_window_metric_exposes_its_time_precision_and_rejects_relative_times():
    case = {"id": "tcp", "url": "https://example.org/lesson", "term_score_eligible": True,
            "asr_clip": [210, 240], "asr_anchors": [
                {"id": "syn", "term": "SYN", "alternatives": ["SYN"], "window": [213, 221]},
            ]}
    transcript = {"range": [210, 240], "model": "model", "backend": "example",
                  "timestamp_precision": "audio_window", "segments": [
                      {"start": 210, "end": 240, "text": "SYN"},
                  ]}
    result = benchmark["evaluate"](case, transcript)
    assert result["matched"] == 1 and result["timestamp_precision"] == "audio_window"
    transcript["segments"][0].update(start=0, end=30)
    with pytest.raises(ValueError, match="relative versus source time"):
        benchmark["evaluate"](case, transcript)

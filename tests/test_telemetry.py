import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from vidlens import telemetry
from vidlens.models import Segment, Video, VidLensError
from vidlens.service import VideoService


class Reader:
    def __init__(self, home):
        self.home, self.run_id = home, "read-run"

    @telemetry.measured
    def read(self, value):
        if isinstance(value, Exception):
            raise value
        return value


def test_observation_failures_preserve_return_values_and_original_exception(tmp_path, monkeypatch):
    reader = Reader(tmp_path)

    def failed_log(*args):
        raise RuntimeError("Metrics unavailable.")

    monkeypatch.setattr(telemetry, "_record", failed_log)
    value = {"video_id": "lesson", "frames": [None], "accounting": object()}
    assert reader.read(value) is value
    error = VidLensError("source_failed", "Real reading failure.")
    with pytest.raises(VidLensError) as raised:
        reader.read(error)
    assert raised.value is error
    items = ["a", "non-dict", "result"]
    assert reader.read(items) is items


def test_concurrent_metrics_preserve_records_and_do_not_store_evidence(tmp_path):
    reader = Reader(tmp_path)
    payload = {"video_id": "lesson", "accounting": {"text_tokens_estimate": 123,
               "method": "o200k_base_estimate"}, "evidence": [{"text": "PRIVATE RAW TEXT"}],
               "frames": [{"width": 852, "height": 480, "path": "/private/local/file.png"}]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(reader.read, [payload] * 12))
    data = telemetry.operation_metrics(tmp_path, run_id="read-run", limit=100)
    assert data["by_operation"]["read"]["calls"] == 12
    raw = (tmp_path / "metrics/operations.jsonl").read_text()
    assert "PRIVATE RAW TEXT" not in raw and "/private/local/file.png" not in raw
    assert "billing" in data["token_scope"]
    assert "must not be summed" in data["scope"]
    assert all(event["estimate_method"] == "o200k_base_estimate" for event in data["operations"])


def test_partial_or_malformed_metrics_do_not_break_reporting(tmp_path):
    reader = Reader(tmp_path)
    reader.read({"video_id": "lesson"})
    path = tmp_path / "metrics/operations.jsonl"
    with path.open("a") as output:
        for item in [None, [], {"operation": "read", "status": []},
                     {"operation": "read", "status": "ok", "elapsed_seconds": float("nan")}]:
            output.write(json.dumps(item) + "\n")
        output.write('{"operation": "partial')
    data = telemetry.operation_metrics(tmp_path)
    assert data["by_operation"]["read"]["calls"] == 1


def prepared_service(tmp_path):
    service = VideoService(tmp_path / "library")
    video = Video("lesson", "https://www.bilibili.com/video/BV1KT2yBJESn/?p=1",
                  "bilibili", "Lesson", 900)
    service.store.save(video, [], [])
    return service, video


def test_prepare_returns_post_asr_plan_and_avoids_retranscribing_cached_speech(tmp_path, monkeypatch):
    service, video = prepared_service(tmp_path)
    before = service.study_status(video.id)["revision"]
    calls = []

    def transcribe(video_id, **kwargs):
        calls.append(kwargs)
        current = service.store.get(video_id)
        current.metadata["transcribed_intervals"] = [[0, 900]]
        service.store.save(current, [Segment(0, 200, "A complete explanation.",
                                            kind="transcript", source="mlx-whisper:large-v3-turbo")], [])
        return {"video_id": video_id}

    monkeypatch.setattr(service, "transcription_options", lambda: {
        "suggested": {"backend": "mlx-whisper", "model": "large-v3-turbo"}})
    monkeypatch.setattr(service, "transcribe", transcribe)
    result = service.read_video(video.url, intent="summary", prepare=True)
    assert result["revision"] != before
    assert result["revision"] == service.study_status(video.id)["revision"]
    assert result["speech_scope"]["whole_video_processed"]
    service.read_video(video.url, intent="summary", prepare=True)
    assert len(calls) == 1


def test_invalid_prepare_fails_before_source_acquisition(tmp_path, monkeypatch):
    service, video = prepared_service(tmp_path)

    def acquire(*args):
        raise AssertionError("Invalid request must not acquire media or metadata.")

    monkeypatch.setattr(service, "_ensure_video", acquire)
    with pytest.raises(VidLensError) as error:
        service.read_video(video.url, intent="summary", prepare="yes")
    assert error.value.code == "invalid_prepare"


def test_prepare_completes_existing_partial_asr(tmp_path, monkeypatch):
    service, video = prepared_service(tmp_path)
    video.metadata["transcribed_intervals"] = [[60, 120]]
    service.store.save(video, [Segment(60, 120, "One earlier answer.", kind="transcript")], [])
    calls = []

    def transcribe(video_id, **kwargs):
        calls.append(kwargs)
        current = service.store.get(video_id)
        current.metadata["transcribed_intervals"] = [[0, 900]]
        service.store.save(current, [Segment(0, 900, "Full speech.", kind="transcript")], [])
        return {"video_id": video_id}

    monkeypatch.setattr(service, "transcription_options", lambda: {
        "suggested": {"backend": "mlx-whisper", "model": "large-v3-turbo"}})
    monkeypatch.setattr(service, "transcribe", transcribe)
    result = service.read_video(video.url, intent="summary", prepare=True)
    assert len(calls) == 1
    assert result["speech_scope"]["whole_video_processed"]

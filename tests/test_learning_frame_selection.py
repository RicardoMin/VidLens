import subprocess

from vidlens.models import MediaAsset
from vidlens.service import VideoService
from vidlens.visual import VisualReader
from vidlens.visual_relevance import QueryVisualSelector


def test_representative_frames_cover_time_buckets_and_advance_into_stable_holds(
    tmp_path, monkeypatch
):
    service = VideoService(tmp_path)
    candidates = [
        {"timestamp": i / 10, "score": 0.2, "reasons": ["small_change"]} for i in range(90)
    ]
    candidates += [
        {"timestamp": t, "score": 0.1, "reasons": ["small_change"]}
        for t in (9, 10, 10.1, 12, 17, 20, 20.1, 23, 28, 30, 30.1, 32, 37, 39.9)
    ]
    scan = {"status": "complete", "candidates": candidates}
    learning = service.learning
    monkeypatch.setattr(learning, "_scan_data", lambda _: (None, None, scan))
    requested = []

    def read_times(_video_id, timestamps, **_kwargs):
        requested.extend(timestamps)
        return [{"timestamp": t} for t in timestamps]

    monkeypatch.setattr(service, "read_frame_times", read_times)
    section = {"start": 0, "end": 40}
    frames, metadata = learning._frames("video", section, None, 4)
    assert len(frames) == 4
    assert all(i * 10 <= timestamp < (i + 1) * 10 for i, timestamp in enumerate(requested))
    assert 12 < requested[1] < 20 and 23 < requested[2] < 28 and 32 < requested[3] < 37
    assert metadata["next_frame_cursor"] == 0
    assert candidates[0]["timestamp"] == 0  # The public sequence remains unchanged.
    requested.clear()
    learning._frames("video", section, 0, 4)
    assert requested == [c["timestamp"] for c in candidates[:4]]


def test_small_candidate_set_keeps_a_single_frame_flash_exactly(tmp_path):
    service = VideoService(tmp_path)
    candidates = [{"timestamp": t, "reasons": ["small_change"]} for t in (0, 1, 1.033333, 2)]
    selected = service.learning._overview_candidates({"start": 0, "end": 3}, candidates, 4)
    assert [row["timestamp"] for row in selected] == [0, 1, 1.033333, 2]


def test_rounded_platform_duration_never_moves_the_end_marker_past_video_eof(tmp_path):
    media = tmp_path / "rounded-duration.mp4"
    subprocess.run([
        VisualReader._ffmpeg(), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "testsrc2=size=96x64:rate=10:duration=3.3", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-threads", "1", str(media),
    ], capture_output=True, check=True, timeout=20)
    # Public metadata can round 3.3 seconds up to 4, as observed on Bilibili.
    asset = MediaAsset(media, duration=4)
    reader = VisualReader(tmp_path / "visual")
    scan = reader.scan(asset, limit=1000)
    assert scan["status"] == "complete"
    assert scan["candidates"][-1]["timestamp"] == 3.2
    assert "end" in scan["candidates"][-1]["reasons"]
    overview = VideoService(tmp_path / "service").learning._overview_candidates(
        {"start": 0, "end": 4}, scan["candidates"], 4)
    probes = QueryVisualSelector.probe_candidates(
        "最终结果", scan["candidates"], start=0, end=4, limit=48)
    assert overview[-1]["timestamp"] == 3.2
    assert probes[-1]["timestamp"] == 3.2
    # Exercise the actual frame resolver: before the fix, the end candidate
    # became 3.466667 / 3.4 seconds and failed with frame_not_visible.
    frames = reader.read_at(asset, [row["timestamp"] for row in overview], ocr=False)
    assert len(frames) == 4 and frames[-1].timestamp == 3.2

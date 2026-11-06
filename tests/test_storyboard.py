import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image, ImageDraw

from vidlens.models import MediaAsset, VidLensError
from vidlens.storyboard import StoryboardScanner
from vidlens.visual import VisualReader


def test_scan_retains_a_single_tiny_code_change_after_1080p_downscale(tmp_path):
    # One changed 14px digit occupies only a few pixels after a 3x downscale.
    # scdet's 0.001% printed-MAFD precision rounded this event to zero.
    for number in range(4):
        image = Image.new("RGB", (1920, 1080), "white")
        ImageDraw.Draw(image).text(
            (25, 25), "x = 9" if number == 2 else "x = 8", fill="black", font_size=14
        )
        image.save(tmp_path / f"tiny-{number}.png")
    path = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            VisualReader._ffmpeg(),
            "-v",
            "error",
            "-framerate",
            "10",
            "-i",
            str(tmp_path / "tiny-%d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    result = VisualReader(tmp_path / "cache").scan(MediaAsset(path, duration=0.4))
    assert result["status"] == "complete", result
    events = {row["timestamp"]: row for row in result["candidates"]}
    assert "small_change" in events[0.2]["reasons"]
    assert events[0.2]["score"] >= StoryboardScanner.CHANGE_THRESHOLD
    assert result["coverage"]["decoded_frames"] == 4


def test_scan_detects_color_only_change_with_identical_luma(tmp_path):
    # Lossless YUV source isolates a syntax/highlight color change: Y is constant.
    width, height = 160, 90
    raw = b"".join(
        bytes([100]) * (width * height)
        + bytes([210 if i == 2 else 90]) * (width * height // 4)
        + bytes([80]) * (width * height // 4)
        for i in range(4)
    )
    path = tmp_path / "chroma.mkv"
    subprocess.run(
        [
            VisualReader._ffmpeg(),
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{width}x{height}",
            "-r",
            "10",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            str(path),
        ],
        input=raw,
        check=True,
        capture_output=True,
        timeout=20,
    )
    result = VisualReader(tmp_path / "cache").scan(MediaAsset(path, duration=0.4))
    assert result["status"] == "complete", result
    events = {row["timestamp"]: row for row in result["candidates"]}
    assert events[0.2]["score"] == pytest.approx(120 / 255, rel=1e-5)
    assert "scene_change" in events[0.2]["reasons"]
    assert result["coverage"]["decoded_frames"] == 4


@pytest.fixture
def lecture(tmp_path):
    """24 seconds at 2 fps, a single changed digit, and an event on the last frame."""
    for number in range(48):
        image = Image.new("RGB", (640, 360), "black")
        text = "token_limit = 129" if number == 27 else "token_limit = 128"
        ImageDraw.Draw(image).text((25, 25), text, fill="white", font_size=14)
        if number == 47:
            ImageDraw.Draw(image).rectangle((80, 100, 560, 260), fill="blue")
        image.save(tmp_path / f"slide-{number:02}.png")
    path = tmp_path / "lecture.mp4"
    subprocess.run(
        [
            VisualReader._ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "2",
            "-i",
            str(tmp_path / "slide-%02d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    return MediaAsset(path, duration=24)


def test_full_scan_detects_small_digit_change_static_anchors_and_final_frame(tmp_path, lecture):
    reader = VisualReader(tmp_path / "cache")
    result = reader.scan(lecture)
    assert result["status"] == "complete", result
    assert result["coverage"] == {
        "complete": True,
        "decoded_frames": 48,
        "first_timestamp": 0,
        "last_timestamp": 23.5,
    }
    candidates = {row["timestamp"]: row for row in result["candidates"]}
    assert {0, 10, 13.5, 14, 20, 23.5} <= candidates.keys()
    assert "small_change" in candidates[13.5]["reasons"]
    assert "periodic_anchor" in candidates[20]["reasons"]
    assert "end" in candidates[23.5]["reasons"]
    assert result["candidate_count"] < 12  # Static encoded frames do not explode into events.
    assert list(candidates) == sorted(candidates)
    assert not list(reader.cache_dir.rglob("*.jpg"))
    assert not list(reader.cache_dir.rglob("*.png"))
    # The final event is a real source frame; high-resolution retrieval stays separate.
    frames = reader.read(lecture, 23.4, 24, max_frames=1, ocr=False)
    assert frames[0].timestamp == pytest.approx(23.5, abs=0.001)
    with Image.open(frames[0].path) as image:
        assert image.getpixel((320, 180))[2] > 230


def test_scan_pagination_reuses_full_manifest_without_decoding(tmp_path, lecture, monkeypatch):
    reader = VisualReader(tmp_path / "cache")
    first = reader.scan(lecture, limit=2)
    assert first["candidate_count"] > 2
    assert first["next_cursor"] == 2
    assert first["cached"] is False

    def unexpected(*args, **kwargs):
        pytest.fail("A warm scan must not launch FFmpeg")

    monkeypatch.setattr(subprocess, "Popen", unexpected)
    rows = first["candidates"][:]
    cursor = first["next_cursor"]
    while cursor is not None:
        page = reader.scan(lecture, cursor=cursor, limit=2)
        assert page["cached"] is True and page["status"] == "complete"
        assert page["scan_id"] == first["scan_id"]
        rows.extend(page["candidates"])
        cursor = page["next_cursor"]
    assert len(rows) == first["candidate_count"]
    assert rows[-1]["timestamp"] == 23.5
    assert len({row["timestamp"] for row in rows}) == len(rows)


def test_scan_timestamps_are_decoded_original_times_after_seek(tmp_path, lecture):
    lecture.offset = 100
    result = VisualReader(tmp_path / "cache").scan(lecture, start=100.1, end=104.1)
    assert result["status"] == "complete", result
    assert result["coverage"]["first_timestamp"] == pytest.approx(100.5, abs=0.000001)
    assert result["coverage"]["last_timestamp"] == pytest.approx(104, abs=0.000001)
    assert all(100.1 <= row["timestamp"] < 104.1 for row in result["candidates"])
    assert result["coverage"]["decoded_frames"] == 8


def test_source_revision_invalidates_scan_cache(tmp_path, lecture):
    reader = VisualReader(tmp_path / "cache")
    first = reader.scan(lecture)
    stat = lecture.path.stat()
    os.utime(lecture.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    second = reader.scan(lecture)
    assert second["cached"] is False
    assert second["scan_id"] != first["scan_id"]
    assert second["candidates"] == first["candidates"]


def test_output_limit_is_explicit_and_partial_scan_is_never_cached(tmp_path, lecture, monkeypatch):
    reader = VisualReader(tmp_path / "cache")
    monkeypatch.setattr(StoryboardScanner, "MAX_OUTPUT_BYTES", 1)
    failed = reader.scan(lecture)
    assert failed["status"] in {"failed", "partial"}
    assert failed["error"]["code"] == "scan_output_limit"
    assert failed["coverage"]["complete"] is False
    assert failed["manifest_path"] is None
    assert not list(reader.cache_dir.rglob("*.json"))
    monkeypatch.setattr(StoryboardScanner, "MAX_OUTPUT_BYTES", 64 * 1024 * 1024)
    retried = reader.scan(lecture)
    assert retried["status"] == "complete" and retried["cached"] is False


def test_truncated_timeline_cannot_claim_complete_scan(tmp_path, lecture):
    lecture.duration = 60
    result = VisualReader(tmp_path / "cache").scan(lecture)
    assert result["status"] == "partial"
    assert result["error"]["code"] == "scan_incomplete_range"
    assert result["coverage"]["complete"] is False
    assert result["coverage"]["last_timestamp"] == 23.5
    assert result["manifest_path"] is None


def test_scan_lock_publishes_one_atomic_manifest(tmp_path, lecture, monkeypatch):
    reader = VisualReader(tmp_path / "cache")
    real_popen, calls = subprocess.Popen, []

    def tracked(*args, **kwargs):
        calls.append(args[0])
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", tracked)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reader.scan(lecture), range(2)))
    assert len(calls) == 1
    assert sorted(row["cached"] for row in results) == [False, True]
    manifests = list(reader.cache_dir.rglob("*.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["coverage"]["complete"] is True
    assert not list(reader.cache_dir.rglob("*.tmp"))


def test_brief_scene_cut_survives_nearby_return_to_old_scene(tmp_path):
    path = tmp_path / "brief.mp4"
    subprocess.run(
        [
            VisualReader._ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=320x180:r=10:d=2",
            "-vf",
            "drawbox=color=white:t=fill:enable='between(t,1,1.05)'",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    result = VisualReader(tmp_path / "cache").scan(MediaAsset(path, duration=2))
    assert result["status"] == "complete", result
    rows = {row["timestamp"]: row for row in result["candidates"]}
    assert "scene_change" in rows[1.0]["reasons"]
    assert 1.1 in rows  # The 100 ms scene and the following return are both represented.
    assert rows[1.9]["reasons"] == ["end"]


def test_long_scan_has_no_global_read_frame_budget(tmp_path):
    path = tmp_path / "continuous.mp4"
    subprocess.run(
        [
            VisualReader._ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=1:d=90",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    result = VisualReader(tmp_path / "cache").scan(MediaAsset(path, duration=90), limit=1000)
    assert result["status"] == "complete", result
    assert result["coverage"]["decoded_frames"] == 90
    assert result["candidate_count"] > VisualReader.MAX_CANDIDATES
    assert result["candidates"][-1]["timestamp"] == 89
    assert result["next_cursor"] is None


@pytest.mark.parametrize(
    "options", [{"limit": 0}, {"limit": 1001}, {"cursor": -1}, {"cursor": True}]
)
def test_scan_rejects_invalid_pagination(tmp_path, lecture, options):
    with pytest.raises(VidLensError, match="cursor"):
        VisualReader(tmp_path / "cache").scan(lecture, **options)

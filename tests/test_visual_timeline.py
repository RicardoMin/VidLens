"""Real decoded PTS, presentation holds, and request/cache mapping."""

import subprocess

import pytest
from PIL import Image

from vidlens.models import MediaAsset
from vidlens.visual import VisualReader


def clip(tmp_path, *, variable=False, single=False):
    colors = ["red", "green", "blue", "yellow"][: 1 if single else 4]
    for i, color in enumerate(colors):
        Image.new("RGB", (160, 90), color).save(tmp_path / f"{i}.png")
    target = tmp_path / "clip.mp4"
    subprocess.run(
        [
            VisualReader._ffmpeg(),
            "-v",
            "error",
            "-framerate",
            "2",
            "-i",
            str(tmp_path / "%d.png"),
            *(
                ["-vf", "setpts='if(lt(N,2),N,if(eq(N,2),4,6))'", "-fps_mode", "vfr"]
                if variable
                else []
            ),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            str(target),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    return MediaAsset(target, duration=0.5 if single else 3.5 if variable else 2)


def test_low_frame_rate_returns_displayed_frame_and_final_hold(tmp_path):
    asset = clip(tmp_path)
    reader = VisualReader(tmp_path / "cache")
    frames = reader.read_at(asset, [0.1, 0.4, 0.5, 1.75], ocr=False)
    assert [frame.timestamp for frame in frames] == [0, 0, 0.5, 1.5]
    assert [frame.requested_timestamp for frame in frames] == [0.1, 0.4, 0.5, 1.75]
    assert [frame.visible_until for frame in frames] == [0.5, 0.5, 1, 2]
    assert frames[0].path == frames[1].path
    assert frames[1].duplicate_of == 0
    with Image.open(frames[-1].path) as image:
        r, g, b = image.getpixel((80, 45))
        assert r > 230 and g > 230 and b < 20


def test_vfr_hold_uses_adjacent_pts_including_cross_section_requests(tmp_path):
    asset = clip(tmp_path, variable=True)
    asset.offset = 300.125
    reader = VisualReader(tmp_path / "cache")
    frames = reader.read_at(asset, [301.5, 302.125, 303.6], ocr=False)
    assert [frame.timestamp for frame in frames] == [300.625, 302.125, 303.125]
    assert [frame.visible_until for frame in frames] == [302.125, 303.125, 303.625]
    assert frames[0].timestamp < 301 < frames[0].requested_timestamp
    with Image.open(frames[0].path) as image:
        r, g, b = image.getpixel((80, 45))
        assert g > 100 and r < 20 and b < 20


def test_single_frame_short_video_and_warm_targets_share_pixels_not_requests(tmp_path, monkeypatch):
    asset = clip(tmp_path, single=True)
    reader = VisualReader(tmp_path / "cache")
    first = reader.read_at(asset, [0.125], ocr=False)[0]

    def unexpected(*args, **kwargs):
        pytest.fail("Decoded presentation interval and pixels should be reused")

    monkeypatch.setattr(subprocess, "run", unexpected)
    second = reader.read_at(asset, [0.499], ocr=False)[0]
    assert first.timestamp == second.timestamp == 0
    assert first.visible_until == second.visible_until == 0.5
    assert first.path == second.path
    assert first.requested_timestamp == 0.125
    assert second.requested_timestamp == 0.499

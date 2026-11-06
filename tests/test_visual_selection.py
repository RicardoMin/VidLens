"""Pixel-informed navigation must remain an additive view of raw visual evidence."""

import copy
import json
import subprocess

import pytest
from PIL import Image, ImageDraw

from vidlens.models import MediaAsset, VidLensError
from vidlens.selection import VisualSelector
from vidlens.visual import VisualReader


@pytest.fixture
def slides(tmp_path):
    for number in range(80):
        image = Image.new("RGB", (640, 360), "white")
        if number >= 4:
            draw = ImageDraw.Draw(image)
            text = "x = 999" if number == 35 else "x = 128" if number < 48 else "y = 2 * x"
            draw.text((80, 100), text, fill="black", font_size=32)
            # Pointer noise must not create a separate lesson state each time.
            draw.rectangle((600 + number % 2 * 4, 335, 603 + number % 2 * 4, 339), fill="red")
        image.save(tmp_path / f"slide-{number:02}.png")
    media = tmp_path / "lecture.mp4"
    subprocess.run([VisualReader._ffmpeg(), "-v", "error", "-framerate", "2", "-i",
                    str(tmp_path / "slide-%02d.png"), "-c:v", "libx264", "-crf", "10",
                    "-pix_fmt", "yuv420p", "-threads", "1", str(media)],
                   check=True, capture_output=True, timeout=20)
    return MediaAsset(media, duration=40)


def test_navigation_reduces_pointer_repetition_without_mutating_raw_scan(tmp_path, slides):
    reader = VisualReader(tmp_path / "cache")
    scan = reader.scan(slides, limit=1000)
    before = copy.deepcopy(scan)
    selector = VisualSelector(reader.cache_dir, reader._ffmpeg)
    index = selector.build(slides, scan)
    assert scan == before
    assert index["metrics"]["navigation_count"] < len(scan["candidates"]) / 2
    assert index["metrics"]["blank_probes"] > 0
    for row in index["navigation_candidates"]:
        original = scan["candidates"][row["source_candidate_cursor"]]
        assert row["source_timestamp"] == original["timestamp"]
        assert "signature" not in selector.public(row)
    chosen = selector.representatives(index, 2, 40, 4)
    assert 2 <= len(chosen) <= 4 and not any(row["blank"] for row in chosen)
    assert any(row["timestamp"] >= 24 for row in chosen)
    assert selector.build(slides, scan)["cached"] is True


def test_very_short_code_change_stays_reachable_in_the_full_view(tmp_path, slides):
    reader = VisualReader(tmp_path / "cache")
    scan = reader.scan(slides, limit=1000)
    assert any(row["timestamp"] == 17.5 for row in scan["candidates"])
    selector = VisualSelector(reader.cache_dir, reader._ffmpeg)
    index = selector.build(slides, scan)
    assert any(row["source_timestamp"] == 17.5 and row["protected"]
               for row in index["navigation_candidates"])
    assert any(row["timestamp"] == 17.5 for row in scan["candidates"])
    frame = reader.read_at(slides, [17.5], ocr=False)[0]
    assert frame.timestamp == 17.5


def test_isolated_one_frame_change_is_protected_at_low_frame_rates(tmp_path):
    # The source interval is 0.5s at 2fps, not the fixed duration of a 30fps flash.
    candidates = [{"timestamp": stamp, "score": 0.00001, "reasons": ["small_change"]}
                  for stamp in [0, 5, 10, 13.5, 14, 20, 25, 30, 35, 40, 45, 50, 55]]
    scan = {"candidates": candidates, "range": {"start": 0, "end": 60}}
    probes = VisualSelector.probes(scan)
    flash = next(row for row in probes if row["source_timestamp"] == 13.5)
    assert flash["timestamp"] == 13.5 and flash["protected"]


def test_thumbnail_failure_never_publishes_a_completed_index(tmp_path, slides, monkeypatch):
    reader = VisualReader(tmp_path / "cache")
    scan = reader.scan(slides, limit=1000)
    selector = VisualSelector(reader.cache_dir, reader._ffmpeg)
    monkeypatch.setattr(selector, "_decode", lambda *args: (_ for _ in ()).throw(
        VidLensError("selection_decode_failed", "fixture failure")))
    with pytest.raises(VidLensError, match="fixture failure"):
        selector.build(slides, scan)
    assert not list(selector.cache_dir.glob("*.json"))


def test_replaced_media_cannot_reuse_an_old_visual_navigation_index(tmp_path, slides, monkeypatch):
    reader = VisualReader(tmp_path / "cache")
    scan = reader.scan(slides, limit=1000)
    selector = VisualSelector(reader.cache_dir, reader._ffmpeg)
    index = selector.build(slides, scan)
    slides.path.write_bytes(b"a replaced media file")
    monkeypatch.setattr(selector, "_decode", lambda *args: (_ for _ in ()).throw(
        VidLensError("selection_decode_failed", "must decode the new media")))
    with pytest.raises(VidLensError, match="must decode the new media"):
        selector.build(slides, scan)
    manifests = [json.loads(path.read_text()) for path in selector.cache_dir.glob("*.json")]
    assert [m["selection_id"] for m in manifests] == [index["selection_id"]]


def test_invalid_cached_signature_is_rebuilt_before_visual_comparison(tmp_path, slides):
    reader = VisualReader(tmp_path / "cache")
    scan = reader.scan(slides, limit=1000)
    selector = VisualSelector(reader.cache_dir, reader._ffmpeg)
    index = selector.build(slides, scan)
    path = selector.cache_dir / f"{index['selection_id']}.json"
    index["navigation_candidates"][0]["signature"] = "AA=="
    path.write_text(json.dumps(index))
    recovered = selector.build(slides, scan)
    assert recovered["cached"] is False
    assert recovered["navigation_candidates"][0]["signature"] != "AA=="
    assert selector.representatives(recovered, 0, 40, 4)

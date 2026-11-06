"""Small figures retain native pixels, full regional coverage and OCR source identity."""

import json
import os
import subprocess

import pytest
from PIL import Image, ImageDraw

from vidlens.models import Frame, MediaAsset, VidLensError
from vidlens.visual import VisualReader


def source_frame(tmp_path, size=(3840, 2160)):
    image = Image.new("RGB", size, "#223344")
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 13):
        draw.line((x, 0, x, size[1] - 1), fill=(x % 255, 100, 200), width=1)
    source = tmp_path / "native.png"
    image.save(source)
    display = image.copy()
    display.thumbnail((1920, 1080))
    path = tmp_path / "display.jpg"
    display.save(path)
    info = {"source": str(source), "timestamp": 3.5, "width": display.width,
            "height": display.height, "bytes": path.stat().st_size,
            "source_bytes": source.stat().st_size, "ocr": "", "ocr_complete": False}
    path.with_suffix(".json").write_text(json.dumps(info))
    return Frame(3.5, path, *display.size), source, image


def test_detail_tiles_cover_region_at_original_pixel_density(tmp_path):
    frame, _, source = source_frame(tmp_path, (4091, 2160))
    reader = VisualReader(tmp_path / "cache")
    result = reader.detail(frame, [[.3, .3, .6, .6]], ocr=False)
    assert result["source_size"] == [4091, 2160]
    assert result["source_encoding"] == "png"
    assert len(result["tiles"]) == 4
    covered = Image.new("1", source.size)
    coverage = ImageDraw.Draw(covered)
    for tile in result["tiles"]:
        x, y, w, h = tile.crop_box
        left, top = round(x * source.width), round(y * source.height)
        right, bottom = round((x + w) * source.width), round((y + h) * source.height)
        with Image.open(tile.path) as actual:
            assert actual.size == (right - left, bottom - top)
            assert actual.tobytes() == source.crop((left, top, right, bottom)).tobytes()
            assert actual.width <= 1920 and actual.height <= 1080
        coverage.rectangle((left, top, right - 1, bottom - 1), fill=1)
        assert tile.timestamp == frame.timestamp
    assert covered.crop((1227, 648, 3682, 1944)).getextrema() == (1, 1)


def test_tile_budget_rejects_incomplete_coverage_before_creating_crops(tmp_path):
    frame, _, _ = source_frame(tmp_path)
    reader = VisualReader(tmp_path / "cache")
    with pytest.raises(VidLensError) as exc:
        reader.detail(frame, max_tiles=1, ocr=False)
    assert exc.value.code == "detail_tile_limit"
    assert not (reader.cache_dir / "crops").exists()


def test_normalized_tile_rounding_never_triggers_one_pixel_downsampling(tmp_path):
    frame, _, source = source_frame(tmp_path, (1983, 720))
    result = VisualReader(tmp_path / "cache").detail(frame, ocr=False)
    assert len(result["tiles"]) == 2
    for tile, expected_left in zip(result["tiles"], [0, 63]):
        assert tile.width == 1920 and tile.height == 720
        with Image.open(tile.path) as actual:
            assert actual.tobytes() == source.crop((expected_left, 0, expected_left + 1920, 720)).tobytes()


def test_ocr_uses_retained_source_and_invalidates_same_size_changed_source(tmp_path, monkeypatch):
    from vidlens import visual

    frame, source, _ = source_frame(tmp_path)
    reader = VisualReader(tmp_path / "cache")
    monkeypatch.setattr(visual.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(visual.shutil, "which", lambda name: "/swift" if name == "swiftc" else None)
    monkeypatch.setattr(reader, "_vision_binary", lambda: tmp_path / "helper")
    calls = []
    regions = [{"text": "1.25 ms", "confidence": 1.0, "box": [.1, .1, .2, .03]}]

    def recognize(command, **kwargs):
        calls.append(command)
        assert command[1:] == [str(source)]
        return subprocess.CompletedProcess(command, 0, json.dumps([
            {"path": str(source), "text": "1.25 ms", "regions": regions}
        ]), "")

    monkeypatch.setattr(visual.subprocess, "run", recognize)
    reader._ocr([frame])
    saved = json.loads(frame.path.with_suffix(".json").read_text())
    assert saved["ocr_input_size"] == [3840, 2160]
    assert frame.ocr_regions == regions
    reader._ocr([frame])
    assert len(calls) == 1
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    reader._ocr([frame])
    assert len(calls) == 2
    refreshed = json.loads(frame.path.with_suffix(".json").read_text())
    assert refreshed["ocr_input_identity"] != saved["ocr_input_identity"]


def test_native_decode_is_lossless_and_does_not_reuse_display_jpeg(tmp_path):
    _frame, source_path, source = source_frame(tmp_path, (2560, 1440))
    movie = tmp_path / "native.mkv"
    subprocess.run([
        VisualReader._ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-framerate", "1", "-i", str(source_path), "-t", "2",
        "-c:v", "ffv1", "-pix_fmt", "bgr0", "-threads", "1", str(movie),
    ], check=True, capture_output=True, timeout=20)
    reader = VisualReader(tmp_path / "cache")
    asset = MediaAsset(movie, duration=2)
    ordinary = reader.read_at(asset, [0], ocr=False)[0]
    native = reader.read_at(asset, [0], ocr=False, native=True)[0]
    assert (ordinary.width, ordinary.height) == (1920, 1080)
    assert (native.width, native.height) == source.size
    assert ordinary.path != native.path and native.path.suffix == ".png"
    assert reader.source_info(native)["decoded_source_size"] == [2560, 1440]
    assert reader.source_info(native)["source_limited"] is False
    with Image.open(native.path) as actual:
        assert actual.convert("RGB").tobytes() == source.tobytes()
    cached = reader.read_at(asset, [0], ocr=False, native=True)[0]
    assert cached.path == native.path and cached.timestamp == native.timestamp


@pytest.mark.parametrize("boxes", [[], [[0, 0, 1.1, 1]], [[True, 0, .5, .5]], [[0, 0, 1]]])
def test_detail_rejects_invalid_regions(tmp_path, boxes):
    frame, _, _ = source_frame(tmp_path, (128, 64))
    with pytest.raises(VidLensError):
        VisualReader(tmp_path / "cache").detail(frame, boxes, ocr=False)


def test_native_limit_discloses_original_size_and_applied_scale(tmp_path):
    movie = tmp_path / "wide.mkv"
    subprocess.run([
        VisualReader._ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
        "-i", "color=c=black:s=4096x128:d=1:r=1", "-c:v", "ffv1", "-threads", "1", str(movie),
    ], check=True, capture_output=True, timeout=20)
    reader = VisualReader(tmp_path / "cache")
    native = reader.read_at(MediaAsset(movie, duration=1), [0], ocr=False, native=True)[0]
    info = reader.source_info(native)
    assert info["decoded_source_size"] == [4096, 128]
    assert info["size"] == [3840, 120]
    assert info["source_scale"] == [0.9375, 0.9375]
    assert info["source_limited"] is True
    details = reader.detail(native, [[0, 0, .2, 1]], ocr=False)
    assert details["pixel_policy"] == "cached_source_pixels_no_upscale"
    assert details["source_limited"] is True


def test_native_flag_is_not_implicitly_enabled_by_non_boolean(tmp_path):
    with pytest.raises(VidLensError) as exc:
        VisualReader(tmp_path / "cache").read_at(MediaAsset(tmp_path / "missing"), [0], native=1)
    assert exc.value.code == "invalid_native_mode"

"""Recognition layout survives backend parsing, cache upgrades and lossless crops."""

import json
import subprocess

import pytest
from PIL import Image

from vidlens.models import Frame
from vidlens.visual import VisualReader


def tsv(*rows):
    return (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        + "\n".join("\t".join(map(str, row)) for row in rows)
    )


def test_tesseract_keeps_metric_value_and_unit_positions_separate():
    result = VisualReader._parse_tesseract(
        tsv(
            (4, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, ""),
            (5, 1, 1, 1, 1, 1, 10, 20, 90, 10, 97, "Duration"),
            (5, 1, 1, 1, 1, 2, 200, 20, 40, 10, 84, "761.86"),
            (5, 1, 1, 1, 1, 3, 250, 20, 20, 10, 70, "µs"),
            (5, 1, 1, 1, 2, 1, 10, 45, 90, 10, 90, "Throughput"),
            (5, 1, 1, 1, 2, 2, 200, 45, 40, 10, 99, "96.11%"),
        ), 400, 100,
    )
    assert result["text"] == "Duration 761.86 µs\nThroughput 96.11%"
    assert [region["text"] for region in result["regions"]] == [
        "Duration", "761.86", "µs", "Throughput", "96.11%"
    ]
    assert result["regions"][1]["box"] == [0.5, 0.2, 0.1, 0.1]
    assert result["regions"][4]["box"][1] == 0.45
    assert result["regions"][1]["confidence"] == 0.84


@pytest.mark.parametrize("output", [
    "Duration 761.86 us",
    tsv((5, 1, 1, 1, 1, 1, 390, 20, 90, 10, 97, "761.86")),
    tsv((5, 1, 1, 1, 1, 1, 10, 20, 90, 10, "nan", "761.86")),
])
def test_tesseract_rejects_missing_or_impossible_geometry(output):
    with pytest.raises(ValueError):
        VisualReader._parse_tesseract(output, 400, 100)


def image_frame(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (400, 100), "white").save(path)
    info = {"source": str(path), "timestamp": 3.5, "width": 400, "height": 100,
            "bytes": path.stat().st_size, "source_bytes": path.stat().st_size,
            "ocr": "legacy OCR", "ocr_complete": True}
    path.with_suffix(".json").write_text(json.dumps(info))
    return Frame(3.5, path, 400, 100), info


def test_legacy_ocr_and_damaged_layout_are_upgraded_once(tmp_path, monkeypatch):
    from vidlens import visual

    frame, _ = image_frame(tmp_path)
    reader = VisualReader(tmp_path / "cache")
    calls = []
    regions = [{"text": "Duration", "confidence": 0.97, "box": [0.05, 0.2, 0.3, 0.1]}]
    monkeypatch.setattr(visual.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(visual.shutil, "which", lambda name: "/swift" if name == "swiftc" else None)
    monkeypatch.setattr(reader, "_vision_binary", lambda: tmp_path / "helper")

    def recognize(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps([
            {"path": str(frame.path), "text": "Duration", "regions": regions}
        ]), "")

    monkeypatch.setattr(visual.subprocess, "run", recognize)
    reader._ocr([frame])
    assert frame.ocr == "Duration" and frame.ocr_regions == regions
    assert len(calls) == 1
    saved = json.loads(frame.path.with_suffix(".json").read_text())
    assert saved["ocr_version"] == reader.OCR_VERSION
    reloaded = reader._cached_frame(frame.path)
    reader._ocr([reloaded])
    assert len(calls) == 1 and reloaded.ocr_regions == regions
    assert reloaded.to_dict(include_layout=True)["ocr_regions"] == regions
    assert "ocr_regions" not in reloaded.to_dict()
    saved["ocr_regions"][0]["box"] = [0, 0, 100, 10]
    frame.path.with_suffix(".json").write_text(json.dumps(saved))
    reader._ocr([reloaded])
    assert len(calls) == 2 and reloaded.ocr_regions == regions


def test_failed_layout_upgrade_keeps_old_text_without_certifying_layout(tmp_path, monkeypatch):
    from vidlens import visual

    frame, _ = image_frame(tmp_path)
    reader = VisualReader(tmp_path / "cache")
    monkeypatch.setattr(visual.platform, "system", lambda: "Linux")
    monkeypatch.setattr(visual.shutil, "which", lambda name: None)
    with pytest.warns(RuntimeWarning, match="OCR is unavailable"):
        reader._ocr([frame])
    assert frame.ocr == "legacy OCR" and frame.ocr_regions == []
    info = json.loads(frame.path.with_suffix(".json").read_text())
    assert "ocr_version" not in info


def test_crop_is_lossless_and_maps_actual_pixel_bounds(tmp_path, monkeypatch):
    path = tmp_path / "source.png"
    image = Image.new("RGB", (101, 51))
    image.putdata([(x % 256, (x * 13) % 256, (x * 53) % 256) for x in range(101 * 51)])
    image.save(path)
    reader = VisualReader(tmp_path / "cache")
    monkeypatch.setattr(reader, "_ocr", lambda frames: None)
    result = reader.crop(Frame(1.25, path, 101, 51), (0.1, 0.2, 0.3, 0.4))
    assert result.path.suffix == ".png"
    assert result.crop_box == pytest.approx([10 / 101, 10 / 51, 31 / 101, 21 / 51])
    with Image.open(result.path) as actual:
        assert actual.tobytes() == image.crop((10, 10, 41, 31)).tobytes()
    cached = reader.crop(Frame(1.25, path, 101, 51), (0.1, 0.2, 0.3, 0.4))
    assert cached.crop_box == result.crop_box and cached.path == result.path

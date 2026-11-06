"""Independent regression checks for detail-source geometry and cache reuse."""

import json

from PIL import Image
from test_fact_check_interfaces import seeded

from vidlens.models import Frame, MediaAsset


def test_source_upgrade_preserves_geometry_and_both_pixel_axes(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    cached = service.store.frame_at(video.id, 2)
    source = tmp_path / "wide-source.png"
    Image.new("RGB", (2400, 1200), "blue").save(source)
    cached.path.with_suffix(".json").write_text(json.dumps({"source": str(source)}))
    candidate = tmp_path / "square-source.png"
    Image.new("RGB", (2160, 2160), "red").save(candidate)
    monkeypatch.setattr(service.media, "get", lambda *a, **k: MediaAsset(candidate, 0, 10))
    monkeypatch.setattr(
        service.visual, "read_at", lambda *a, **k: [Frame(2, candidate, 2160, 2160)]
    )

    result = service.read_detail(video.id, 2, [[0, 0, .5, .5]], upgrade_source=True)

    assert result["detail"]["source_size"] == [2400, 1200]
    with Image.open(result["frames"][1]["path"]) as tile:
        assert tile.size == (1200, 600)
        assert tile.getpixel((0, 0)) == (0, 0, 255)


def test_ordinary_reread_does_not_forget_an_upgraded_detail_source(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    ordinary = service.store.frame_at(video.id, 2)
    source = tmp_path / "native-source.png"
    Image.new("RGB", (2560, 1280), "blue").save(source)
    native = Frame(2, source, 2560, 1280)
    monkeypatch.setattr(service.media, "get", lambda *a, **k: MediaAsset(source, 0, 10))
    monkeypatch.setattr(
        service.visual, "read_at",
        lambda *a, **k: [native if k.get("native") else ordinary],
    )
    upgraded = service.read_detail(video.id, 2, [[0, 0, .5, .5]], upgrade_source=True)
    assert upgraded["detail"]["source_size"] == [2560, 1280]

    service.read_frames(video.id, 1, 3, timestamps=[2], ocr=False)
    repeated = service.read_detail(video.id, 2, [[0, 0, .5, .5]])

    assert repeated["detail"]["source_size"] == [2560, 1280]
    with Image.open(repeated["frames"][1]["path"]) as tile:
        assert tile.size == (1280, 640)
        assert tile.getpixel((0, 0)) == (0, 0, 255)


def test_display_sidecar_retains_larger_source_ocr_and_confirmed_interval(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    old = service.store.frame_at(video.id, 2)
    source = tmp_path / "retained.bin"
    Image.new("RGB", (2400, 1200), "blue").save(source, format="JPEG")
    old.path.with_suffix(".json").write_text(json.dumps({"source": str(source)}))
    old.ocr = "Bandwidth 12 ms"
    old.ocr_regions = [{"text": "12 ms", "confidence": 1.0, "box": [.1, .1, .2, .1]}]
    old.visible_until = 2.5
    service.store.save_frames(video.id, [old])
    path = tmp_path / "fresh.bin"
    Image.new("RGB", (1200, 600), "red").save(path, format="PNG")
    fresh = Frame(2, path, 1200, 600, requested_timestamp=2.2, duplicate_of=1)
    service.store.save_frames(video.id, [fresh])
    stored = service.store.frame_at(video.id, 2)
    assert stored.path == old.path
    assert stored.ocr == old.ocr and stored.ocr_regions == old.ocr_regions
    assert stored.visible_until == 2.5 and stored.requested_timestamp == 2.2
    assert stored.duplicate_of is None
    assert service.store.segments(video.id)[0].text == old.ocr


def test_equal_dimensions_prefer_png_encoding_without_filename_heuristics(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    paths = [tmp_path / name for name in ["first.opaque", "second.opaque"]]
    for path, encoding in zip(paths, ["PNG", "JPEG"]):
        Image.new("RGB", (400, 200), "blue").save(path, format=encoding)
        service.store.save_frames(video.id, [Frame(2, path, 400, 200)])
    assert service.store.frame_at(video.id, 2).path == paths[0]


def test_source_revision_change_drops_old_pixels_ocr_and_interval(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    path = tmp_path / "large.png"
    Image.new("RGB", (800, 400), "blue").save(path)
    service.store.save_frames(video.id, [Frame(2, path, 800, 400, "old text", visible_until=3)])
    video.metadata["revision"] = "v2"
    service.store.save(video, [], [])
    new = tmp_path / "new.png"
    Image.new("RGB", (100, 50), "red").save(new)
    service.store.save_frames(video.id, [Frame(2, new, 100, 50)])
    stored = service.store.frame_at(video.id, 2)
    assert stored.path == new and not stored.ocr and stored.visible_until is None
    assert service.store.segments(video.id) == []


def test_missing_or_unattributed_old_source_does_not_win_quality_comparison(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    large = tmp_path / "large.png"
    Image.new("RGB", (800, 400), "blue").save(large)
    small = tmp_path / "small.png"
    Image.new("RGB", (100, 50), "red").save(small)
    service.store.save_frames(video.id, [Frame(2, large, 800, 400)])
    with service.store.connect() as db:
        data = json.loads(db.execute("SELECT data FROM frames WHERE video_id=?", (video.id,)).fetchone()[0])
        data.pop("_source_identity")
        db.execute("UPDATE frames SET data=? WHERE video_id=?", (json.dumps(data), video.id))
    service.store.save_frames(video.id, [Frame(2, small, 100, 50)])
    assert service.store.frame_at(video.id, 2).path == small
    service.store.save_frames(video.id, [Frame(2, large, 800, 400)])
    large.unlink()
    service.store.save_frames(video.id, [Frame(2, small, 100, 50)])
    assert service.store.frame_at(video.id, 2).path == small


def test_ocr_geometry_is_not_inherited_across_changed_aspect_ratio(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    old = service.store.frame_at(video.id, 2)
    old.ocr = "wide figure"
    old.ocr_regions = [{"text": old.ocr, "confidence": 1.0, "box": [.1, .1, .2, .1]}]
    service.store.save_frames(video.id, [old])
    path = tmp_path / "portrait.png"
    Image.new("RGB", (100, 200), "red").save(path)
    service.store.save_frames(video.id, [Frame(2, path, 100, 200)])
    stored = service.store.frame_at(video.id, 2)
    assert stored.path == path and stored.ocr == "" and stored.ocr_regions == []

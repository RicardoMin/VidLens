import subprocess
from itertools import pairwise
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from vidlens.models import Frame, MediaAsset, VidLensError
from vidlens.visual import VisualReader


@pytest.fixture
def clip(tmp_path):
    """Six seconds, 2 fps; a brief blue slide at 3.5s amid red slides."""
    for number in range(12):
        image = Image.new("RGB", (640, 360), "blue" if number == 7 else "red")
        ImageDraw.Draw(image).text((40, 40), "SLIDE", fill="white", font_size=28)
        image.save(tmp_path / f"input-{number:02}.png")
    path = tmp_path / "fixture.mp4"
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
            str(tmp_path / "input-%02d.png"),
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
    return path


def test_actual_frame_timestamp_with_asset_offset_and_seek_rounding(tmp_path, clip):
    reader = VisualReader(tmp_path / "cache")
    frames = reader.read(
        MediaAsset(clip, offset=100, duration=6),
        100.1,
        103.1,
        max_frames=3,
        strategy="uniform",
        ocr=False,
    )
    # Requested midpoints are .6, 1.6, 2.6; actual decoded frames are 1, 2, 3.
    assert [frame.timestamp for frame in frames] == pytest.approx([101, 102, 103], abs=0.002)
    assert all(frame.path.is_file() for frame in frames)
    assert all(100.1 <= frame.timestamp < 103.1 for frame in frames)


def test_changes_find_brief_slide_and_preserve_bucket_coverage(tmp_path, clip):
    reader = VisualReader(tmp_path / "cache")
    frames = reader.read(MediaAsset(clip, duration=6), 0, 6, max_frames=2, ocr=False)
    assert len(frames) == 2
    assert 0 <= frames[0].timestamp < 3
    assert 3 <= frames[1].timestamp < 6
    with Image.open(frames[1].path) as image:
        red, _, blue = image.getpixel((320, 180))
    assert blue > 220 and red < 20
    assert frames[1].timestamp == pytest.approx(3.5, abs=0.002)


def test_uniform_repeated_visuals_keep_time_coverage(tmp_path, clip):
    reader = VisualReader(tmp_path / "cache")
    frames = reader.read(
        MediaAsset(clip, duration=6), 0, 6, max_frames=2, strategy="uniform", ocr=False
    )
    assert len(frames) == 2
    assert frames[1].duplicate_of == frames[0].timestamp
    assert frames[0].timestamp != frames[1].timestamp


def test_small_code_change_is_not_marked_as_duplicate(tmp_path):
    for number in range(4):
        image = Image.new("RGB", (1280, 720), "black")
        text = "token_limit = 128" if number < 2 else "token_limit = 129"
        ImageDraw.Draw(image).text((50, 50), text, fill="white", font_size=20)
        image.save(tmp_path / f"code-{number}.png")
    path = tmp_path / "code-change.mp4"
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
            str(tmp_path / "code-%d.png"),
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
    frames = VisualReader(tmp_path / "cache").read(
        MediaAsset(path, duration=2),
        0,
        2,
        max_frames=2,
        strategy="uniform",
        ocr=False,
    )
    assert len(frames) == 2
    assert frames[1].duplicate_of is None


def test_cached_frames_do_not_decode_again(tmp_path, clip, monkeypatch):
    reader = VisualReader(tmp_path / "cache")
    first = reader.read(
        MediaAsset(clip, duration=6), 0, 6, max_frames=2, strategy="uniform", ocr=False
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Cached frame unexpectedly invokes a subprocess")

    monkeypatch.setattr(subprocess, "run", unexpected)
    second = reader.read(
        MediaAsset(clip, duration=6), 0, 6, max_frames=2, strategy="uniform", ocr=False
    )
    assert [(frame.timestamp, frame.path) for frame in first] == [
        (frame.timestamp, frame.path) for frame in second
    ]


def test_crop_uses_high_resolution_source_and_preserves_timestamp(tmp_path, monkeypatch):
    import json

    reader = VisualReader(tmp_path / "cache")
    original = tmp_path / "source.jpg"
    source = Image.new("RGB", (3840, 2160), "white")
    ImageDraw.Draw(source).rectangle((1920, 0, 3839, 2159), fill="blue")
    source.save(original)
    display = tmp_path / "display.jpg"
    source.resize((1920, 1080)).save(display)
    display.with_suffix(".json").write_text(json.dumps({"source": str(original)}))
    monkeypatch.setattr(reader, "_ocr", lambda frames: None)
    result = reader.crop(Frame(123.5, display, 1920, 1080), (0.5, 0.25, 0.5, 0.5))
    assert (result.width, result.height) == (1920, 1080)
    assert result.timestamp == 123.5
    with Image.open(result.path) as image:
        assert image.getpixel((100, 100))[2] > 240


def test_contact_sheet_has_all_tiles_and_timestamp_labels(tmp_path):
    reader = VisualReader(tmp_path / "cache")
    frames = []
    for index in range(4):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (640, 360), "red").save(path)
        frames.append(Frame(3600 + index + 0.125, path, 640, 360))
    destination = reader.contact_sheet(frames, tmp_path / "sheet.jpg")
    with Image.open(destination) as image:
        assert image.size == (1800, 744)
        # The label band contains drawn text, rather than just the background.
        assert len(image.crop((0, 338, 600, 372)).getcolors(maxcolors=100000)) > 20
    assert reader._timestamp(3601.125) == "01:00:01.125"


def test_ocr_failure_does_not_drop_image(tmp_path, clip, monkeypatch):
    from vidlens import visual

    reader = VisualReader(tmp_path / "cache")
    monkeypatch.setattr(visual.platform, "system", lambda: "Darwin")
    real_which = visual.shutil.which
    monkeypatch.setattr(
        visual.shutil, "which", lambda name: "/bin/false" if name == "swiftc" else real_which(name)
    )
    monkeypatch.setattr(
        reader, "_vision_binary", lambda: (_ for _ in ()).throw(RuntimeError("test failure"))
    )
    with pytest.warns(RuntimeWarning, match="Local OCR failed"):
        frames = reader.read(MediaAsset(clip, duration=6), 0, 2, max_frames=1)
    assert len(frames) == 1 and frames[0].path.is_file()
    assert frames[0].ocr == ""
    assert "test failure" in reader.last_warnings[0]


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 1), (2, 1), (float("nan"), 1)])
def test_reject_invalid_interval(tmp_path, start, end):
    reader = VisualReader(tmp_path / "cache")
    with pytest.raises(VidLensError, match="finite times"):
        reader.read(MediaAsset(Path("missing.mp4")), start, end)


def test_reject_interval_before_asset_and_invalid_crop(tmp_path, clip):
    reader = VisualReader(tmp_path / "cache")
    with pytest.raises(VidLensError, match="before"):
        reader.read(MediaAsset(clip, offset=10, duration=6), 9, 11)
    with pytest.raises(VidLensError, match="inside"):
        reader.crop(Frame(0, Path("unused.jpg"), 1, 1), (0.9, 0, 0.2, 1))


def test_dense_changes_only_persist_selected_images_and_warm_read_skips_pixels(
    tmp_path, clip, monkeypatch
):
    reader = VisualReader(tmp_path / "cache")
    asset = MediaAsset(clip, duration=6)
    first = reader.read(asset, 0, 6, max_frames=2, ocr=False)
    assert len(list(reader.cache_dir.rglob("*.jpg"))) == len(first)

    def unexpected(*args, **kwargs):
        pytest.fail("Cached selection unexpectedly reads pixels or decodes video")

    monkeypatch.setattr(Image, "open", unexpected)
    monkeypatch.setattr(subprocess, "run", unexpected)
    warm = reader.read(asset, 0.0, 6.0, max_frames=2, ocr=False)
    assert [f.to_dict() for f in warm] == [f.to_dict() for f in first]


@pytest.mark.parametrize("damage", ["missing", "truncated"])
def test_damaged_cached_image_is_rebuilt_and_duplicates_share_ocr(
    tmp_path, clip, monkeypatch, damage
):
    reader = VisualReader(tmp_path / "cache")
    asset = MediaAsset(clip, duration=6)
    first = reader.read(asset, 0, 6, max_frames=2, strategy="uniform", ocr=False)
    if damage == "missing":
        first[1].path.unlink()
    else:
        first[1].path.write_bytes(b"truncated JPEG")
    seen = []

    def recognize(frames):
        seen.extend(f.timestamp for f in frames)
        for frame in frames:
            frame.ocr = "SLIDE"
            frame.ocr_regions = [{"text": "SLIDE", "confidence": 0.9,
                                  "box": [0.1, 0.1, 0.4, 0.2]}]

    monkeypatch.setattr(reader, "_ocr", recognize)
    second = reader.read(asset, 0, 6, max_frames=2, strategy="uniform")
    assert seen == [first[0].timestamp]
    assert second[1].path.is_file()
    assert second[1].duplicate_of == first[0].timestamp
    assert all(frame.ocr == "SLIDE" for frame in second)
    assert second[0].ocr_regions and second[1].ocr_regions == second[0].ocr_regions


def test_crop_reuses_saved_image_and_successful_ocr(tmp_path, clip, monkeypatch):
    import hashlib
    import json

    reader = VisualReader(tmp_path / "cache")
    frame = reader.read(MediaAsset(clip, duration=6), 0, 6, max_frames=1, ocr=False)[0]
    box = (0, 0, 0.5, 0.5)
    with monkeypatch.context() as patch:
        patch.setattr(reader, "_ocr", lambda frames: None)
        crop = reader.crop(frame, box)
    sidecar = reader._sidecar(crop.path)
    metadata = json.loads(sidecar.read_text())
    stat = crop.path.stat()
    identity = hashlib.sha256(
        f"{crop.path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()
    metadata.update(ocr="SLIDE", ocr_complete=True,
                    ocr_version=reader.OCR_VERSION, ocr_regions=[],
                    ocr_input_identity=identity, ocr_input_size=[crop.width, crop.height])
    sidecar.write_text(json.dumps(metadata))

    def unexpected(*args, **kwargs):
        pytest.fail("Repeated crop unexpectedly encodes the image or runs OCR")

    monkeypatch.setattr(Image, "open", unexpected)
    monkeypatch.setattr(subprocess, "run", unexpected)
    repeated = reader.crop(frame, box)
    assert repeated.path == crop.path and repeated.ocr == "SLIDE"
    assert repeated.timestamp == crop.timestamp and repeated.selection == "crop"


def test_sparse_bucket_fallback_recovers_last_available_frame(tmp_path, clip):
    reader = VisualReader(tmp_path / "cache")
    # The midpoint is after the last presentation timestamp of this 2 fps video.
    frames = reader.read(MediaAsset(clip, offset=10, duration=6), 15.2, 16, max_frames=1, ocr=False)
    assert not reader.last_warnings
    assert [f.timestamp for f in frames] == pytest.approx([15.5], abs=0.001)


def test_variable_rate_preview_returns_colors_at_their_source_timestamps(tmp_path):
    import re

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    durations = [0.12, 0.60, 1.20, 0.08, 1.20]
    sequence = []
    for i, (color, duration) in enumerate(zip(colors, durations, strict=True)):
        Image.new("RGB", (320, 180), color).save(tmp_path / f"color-{i}.png")
        sequence.extend([f"file 'color-{i}.png'", f"duration {duration}"])
    sequence.append("file 'color-4.png'")
    concat = tmp_path / "list.txt"
    concat.write_text("\n".join(sequence))
    path = tmp_path / "variable.mkv"
    executable = VisualReader._ffmpeg()
    subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )
    decoded = subprocess.run(
        [
            executable,
            "-hide_banner",
            "-i",
            str(path),
            "-vf",
            "showinfo=checksum=0",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    stamps = [float(t) for t in re.findall(r"\bpts_time:([-+\d.eE]+)", decoded.stderr)]
    assert len(stamps) == len(colors) + 1
    assert len({round(b - a, 2) for a, b in pairwise(stamps)}) > 2
    asset = MediaAsset(path, offset=123.0, duration=stamps[-1] + 0.04)
    frames = VisualReader(tmp_path / "cache").read(
        asset, 123.02, 123.0 + stamps[-1] + 0.04, max_frames=2, ocr=False
    )
    assert len(frames) == 2
    for frame in frames:
        index = min(
            range(len(stamps)), key=lambda i: abs(frame.timestamp - asset.offset - stamps[i])
        )
        assert frame.timestamp == pytest.approx(asset.offset + stamps[index], abs=0.002)
        with Image.open(frame.path) as image:
            pixel = image.convert("RGB").getpixel((160, 90))
        assert max(abs(a - b) for a, b in zip(pixel, colors[min(index, len(colors) - 1)])) < 8

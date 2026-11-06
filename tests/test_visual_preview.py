"""Preview pages preserve source images and provide unambiguous time/tile maps."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from vidlens.models import VidLensError
from vidlens.visual_preview import build_previews


def make_frames(tmp_path, count):
    frames = []
    for index in range(count):
        path = tmp_path / f"frame-{index}.png"
        image = Image.new("RGB", (854, 480), (20 + index * 13, 80, 120))
        ImageDraw.Draw(image).text((50, 150), f"STATE {index + 1}   x = {index} + 1",
                                  fill="white", font_size=36)
        image.save(path)
        frames.append({"path": str(path), "timestamp": 120 + index * 7.125,
                       "requested_timestamp": 120.01 + index * 7.125,
                       "source_candidate_cursor": 50 + index})
    return frames


def test_pages_keep_readable_images_exact_source_maps_and_numbered_labels(tmp_path):
    frames = make_frames(tmp_path, 7)
    originals = [Path(row["path"]).read_bytes() for row in frames]
    service = SimpleNamespace(home=tmp_path / "library")
    result = build_previews(service, list(reversed(frames)))
    assert result["candidate_count"] == 7
    assert [len(page["candidates"]) for page in result["previews"]] == [6, 1]
    assert [row["candidate_index"] for row in result["candidates"]] == list(range(1, 8))
    assert [row["tile_index"] for row in result["candidates"]] == [1, 2, 3, 4, 5, 6, 1]
    for index, row in enumerate(result["candidates"]):
        assert row["timestamp"] == frames[index]["timestamp"]
        assert row["requested_timestamp"] == frames[index]["requested_timestamp"]
        assert row["read_timestamp"] == frames[index]["requested_timestamp"]
        assert row["source_candidate_cursor"] == frames[index]["source_candidate_cursor"]
    for page in result["previews"]:
        with Image.open(page["path"]) as sheet:
            assert max(sheet.size) <= 1920
            assert sheet.size == (page["width"], page["height"])
            # Text belongs to a separate footer, not the source content region.
            for row in page["candidates"]:
                index = row["tile_index"] - 1
                x, y = 12 + index % 2 * 866, 12 + index // 2 * 540
                label = sheet.crop((x, y + 480, x + 854, y + 528))
                assert label.getchannel("R").getextrema()[1] > 240
    assert [Path(row["path"]).read_bytes() for row in frames] == originals
    assert not result["truncated"]


def test_adjacent_duplicates_share_tile_but_returning_state_stays_in_sequence(tmp_path):
    frames = make_frames(tmp_path, 2)
    repeated = [frames[0], {**frames[0], "timestamp": 121},
                frames[1], {**frames[0], "timestamp": 145}]
    result = build_previews(SimpleNamespace(home=tmp_path), repeated)
    assert [row["timestamp"] for row in result["candidates"]] == [120, 127.125, 145]
    assert [row["timestamp"] for row in result["candidates"][0]["occurrences"]] == [120, 121]
    assert result["candidates"][0]["path"] == result["candidates"][-1]["path"]


def test_cached_pages_follow_image_bytes_and_timestamps_and_repair_broken_file(tmp_path):
    frames = make_frames(tmp_path, 1)
    service = SimpleNamespace(home=tmp_path / "library")
    first = Path(build_previews(service, frames)["previews"][0]["path"])
    timestamp = first.stat().st_mtime_ns
    assert Path(build_previews(service, frames)["previews"][0]["path"]) == first
    assert first.stat().st_mtime_ns == timestamp
    first.write_bytes(b"interrupted cache")
    assert Path(build_previews(service, frames)["previews"][0]["path"]) == first
    with Image.open(first) as image:
        image.load()
    later = build_previews(service, [{**frames[0], "timestamp": 121}])["previews"][0]["path"]
    assert str(first) != later
    Image.new("RGB", (854, 480), "red").save(frames[0]["path"])
    replaced = build_previews(service, frames)["previews"][0]["path"]
    assert str(first) != replaced
    assert not list(first.parent.glob(".*.jpg"))


def test_overflow_keeps_timeline_coverage_and_reports_omission(tmp_path):
    frames = make_frames(tmp_path, 10)
    times = [0, 0.1, 0.2, 0.3, 100, 200, 300, 400, 500, 600]
    for row, timestamp in zip(frames, times):
        row["timestamp"] = timestamp
    result = build_previews(SimpleNamespace(home=tmp_path), frames, max_candidates=6)
    selected = [row["timestamp"] for row in result["candidates"]]
    assert selected == [0, 100, 200, 400, 500, 600]
    assert result["truncated"] and result["available_candidate_count"] == 10


def test_invalid_source_or_timestamp_fails_without_publishing_partial_page(tmp_path):
    service = SimpleNamespace(home=tmp_path / "library")
    frames = make_frames(tmp_path, 1)
    with pytest.raises(VidLensError, match="finite"):
        build_previews(service, [{**frames[0], "timestamp": float("nan")}])
    with pytest.raises(VidLensError, match="could not be read"):
        build_previews(service, [*frames, {"timestamp": 999, "path": str(tmp_path / "missing")}])
    assert not list(service.home.rglob("preview-*.jpg"))
    assert build_previews(service, [])["previews"] == []

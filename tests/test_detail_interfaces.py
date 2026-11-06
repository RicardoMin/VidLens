"""Small-text evidence keeps its pixel, timeline and delivery provenance."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image
from test_fact_check_interfaces import seeded

from vidlens import mcp_server
from vidlens.cli import build_parser, dispatch
from vidlens.mcp_server import create_server
from vidlens.models import Frame, MediaAsset, VidLensError


@pytest.mark.asyncio
async def test_detail_transmits_context_and_distinct_source_pixel_regions(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    async with create_connected_server_and_client_session(create_server(service=service)) as client:
        response = await client.call_tool("read_detail", {
            "video_id": video.id, "timestamp": 2, "boxes": [[0, 0, 1, .5], [0, .5, 1, .5]],
        })
        payload = json.loads(response.content[0].text)
        assert not response.isError
        assert [frame["image_index"] for frame in payload["frames"]] == [1, 2, 3]
        assert [frame["role"] for frame in payload["frames"]] == ["context", "detail", "detail"]
        assert all(block.mimeType == "image/png" for block in response.content[2:])
        assert payload["detail"]["source_size"] == [200, 100]
        assert len(service.store.deliveries(video.id)) == 1


def test_detail_reports_transmitted_context_size_separately_from_native_pixels(tmp_path):
    source = tmp_path / "source.png"
    tile = tmp_path / "tile.png"
    Image.new("RGB", (3840, 2160), "white").save(source)
    Image.new("RGB", (1200, 600), "blue").save(tile)
    response = mcp_server.result_content({"frames": [
        Frame(2, source, 3840, 2160).to_dict() | {"role": "context"},
        Frame(2, tile, 1200, 600, crop_box=[0, 0, .3125, .277778]).to_dict()
        | {"role": "detail"},
    ]}, images=True)
    context, detail = json.loads(response.content[0].text)["frames"]
    assert context["width"] == 3840 and context["transmitted_size"] == [1920, 1080]
    assert context["transmitted_mime_type"] == "image/jpeg"
    assert detail["transmitted_size"] == [1200, 600]
    assert detail["transmitted_mime_type"] == "image/png"


def test_higher_requested_profile_cannot_replace_larger_cached_source(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    frame = service.store.frame_at(video.id, 2)
    source = tmp_path / "source.png"
    Image.new("RGB", (2400, 1200), "blue").save(source)
    frame.path.with_suffix(".json").write_text(json.dumps({"source": str(source)}))
    smaller = tmp_path / "smaller.png"
    Image.new("RGB", (800, 400), "red").save(smaller)
    calls = []

    def media(*args, **kwargs):
        calls.append(kwargs)
        return MediaAsset(smaller, 0, 10)

    monkeypatch.setattr(service.media, "get", media)
    monkeypatch.setattr(service.visual, "read_at", lambda *a, **k: [Frame(2, smaller, 800, 400)])
    result = service.read_detail(video.id, 2, [[0, 0, .5, .5]], upgrade_source=True)
    assert result["acquisition"]["status"] == "kept_larger_cached_source"
    assert result["detail"]["source_size"] == [2400, 1200]
    assert calls[0]["max_height"] == 2160
    with Image.open(result["frames"][1]["path"]) as tile:
        assert tile.size == (1200, 600) and tile.getpixel((0, 0)) == (0, 0, 255)


def test_failed_quality_upgrade_returns_cached_pixels_and_explicit_reason(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)

    def unavailable(*a, **k):
        raise VidLensError("source_unavailable", "The source is unavailable.")

    monkeypatch.setattr(service.media, "get", unavailable)
    result = service.read_detail(video.id, 2, upgrade_source=True)
    assert result["frames"] and result["acquisition"]["status"] == "cached_source_fallback"
    assert result["acquisition"]["error"]["code"] == "source_unavailable"


def test_successful_decode_recovers_from_corrupt_retained_source(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    frame = service.store.frame_at(video.id, 2)
    damaged = tmp_path / "damaged.png"
    damaged.write_bytes(b"incomplete image")
    frame.path.with_suffix(".json").write_text(json.dumps({"source": str(damaged)}))
    recovered = tmp_path / "recovered.png"
    Image.new("RGB", (800, 400), "blue").save(recovered)
    monkeypatch.setattr(service.media, "get", lambda *a, **k: MediaAsset(recovered, 0, 10))
    monkeypatch.setattr(service.visual, "read_at", lambda *a, **k: [Frame(2, recovered, 800, 400)])
    result = service.read_detail(video.id, 2, upgrade_source=True)
    assert result["acquisition"]["status"] == "native_decode_recovered"
    assert result["detail"]["source_size"] == [800, 400]


@pytest.mark.asyncio
async def test_missing_context_image_does_not_mark_crop_as_whole_frame(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    path = service.store.frame_at(video.id, 2).path
    original = mcp_server._image_content

    def encode(source, **kwargs):
        if source == str(path):
            raise OSError("context encoding failed")
        return original(source, **kwargs)

    monkeypatch.setattr(mcp_server, "_image_content", encode)
    async with create_connected_server_and_client_session(create_server(service=service)) as client:
        response = await client.call_tool("read_detail", {
            "video_id": video.id, "timestamp": 2, "boxes": [[0, 0, 1, .5]],
        })
        payload = json.loads(response.content[0].text)
        assert not response.isError
        assert payload["frames"][0]["image_index"] is None
        assert payload["frames"][1]["image_index"] == 1
        assert service.store.deliveries(video.id) == []


def test_detail_rejects_stale_source_and_invalid_arguments_before_acquisition(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    result = service.read_detail(video.id, 2, record_delivery=False)
    video.metadata["revision"] = "v2"
    service.store.save(video, [], [])
    with pytest.raises(VidLensError) as caught:
        service.record_detail_delivery(result)
    assert caught.value.code == "stale_revision"
    monkeypatch.setattr(service.media, "get", lambda *a, **k: pytest.fail("invalid acquisition"))
    for kwargs in ({"boxes": [[0, 0, 2, 1]]}, {"upgrade_source": "yes"}, {"max_tiles": True}):
        with pytest.raises(VidLensError):
            service.read_detail(video.id, 2, **kwargs)


def test_detail_cli_preserves_source_upgrade_and_multiple_boxes():
    from types import SimpleNamespace

    calls = []
    service = SimpleNamespace(read_detail=lambda *a, **k: calls.append((a, k)))
    args = build_parser().parse_args([
        "detail", "video", "1:02", "--box", "0", "0", "1", ".5",
        "--box", "0", ".5", "1", ".5", "--upgrade-source", "--max-tiles", "4",
    ])
    dispatch(args, service)
    assert calls == [(("video", 62, [[0, 0, 1, .5], [0, .5, 1, .5]]),
                      {"upgrade_source": True, "max_tiles": 4})]

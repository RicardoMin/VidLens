"""Figure checks must deliver distinct crops without certifying their meaning."""

import base64
import io
import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image

from vidlens.cli import build_parser, dispatch
from vidlens.mcp_server import create_server, result_content
from vidlens.models import Frame, Video, VidLensError
from vidlens.service import VideoService


def seeded(tmp_path, monkeypatch):
    service = VideoService(tmp_path / "library")
    path = tmp_path / "figure.png"
    Image.new("RGB", (200, 100), "white").save(path)
    video = Video("figure", "https://example.com/figure", "local", "Figure", 10,
                  metadata={"revision": "v1"})
    service.store.save(video, [], [])
    service.store.save_frames(video.id, [Frame(2, path, 200, 100)])
    monkeypatch.setattr(service.visual, "_ocr", lambda frames: None)
    return service, video


def claim(identity="bandwidth", **updates):
    return {"id": identity, "timestamp": 2, "label": "Bandwidth", "value": "93",
            "unit": "%", "scope": "one kernel", "meaning": "throughput",
            **updates}


@pytest.mark.asyncio
async def test_different_regions_of_same_frame_are_delivered_as_distinct_lossless_images(
    tmp_path, monkeypatch,
):
    service, video = seeded(tmp_path, monkeypatch)
    claims = [claim(box=[0, 0, 1, .5]), claim("lower", box=[0, .5, 1, .5])]
    async with create_connected_server_and_client_session(create_server(service=service)) as client:
        response = await client.call_tool("check_facts", {"video_id": video.id, "claims": claims})
        payload = json.loads(response.content[0].text)
        assert not response.isError
        assert [frame["image_index"] for frame in payload["frames"]] == [1, 2]
        assert [row["frame_index"] for row in payload["checks"]] == [1, 2]
        assert payload["status"] == "requires_visual_and_semantic_review"
        images = response.content[1:]
        assert len(images) == 2
        for content in images:
            assert content.mimeType == "image/png"
            with Image.open(io.BytesIO(base64.b64decode(content.data))) as image:
                assert image.format == "PNG"
        assert service.store.deliveries(video.id) == []


def test_shared_crop_is_reused_and_missing_ocr_does_not_become_verified(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    result = service.check_facts(video.id, [claim(), claim("other", value="91")])
    assert len(result["frames"]) == 1
    assert result["frames"][0]["claim_ids"] == ["bandwidth", "other"]
    assert all(row["status"] == "unreadable" for row in result["checks"])
    assert all(row["requires_visual_review"] and row["requires_semantic_review"] for row in result["checks"])
    assert len(result["frames"][0]["sha256"]) == 64
    with pytest.raises(VidLensError, match="Read the cited original"):
        service.check_facts(video.id, [claim(timestamp=5)])


def test_source_change_during_or_after_checking_is_rejected(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    result = service.check_facts(video.id, [claim()])
    video.metadata["revision"] = "v2"
    service.store.save(video, [], [])
    with pytest.raises(VidLensError) as caught:
        service.record_fact_check_delivery(result)
    assert caught.value.code == "stale_revision"
    original = service.visual.crop

    def replace_video(*args):
        cropped = original(*args)
        video.metadata["revision"] = "v3"
        service.store.save(video, [], [])
        return cropped

    monkeypatch.setattr(service.visual, "crop", replace_video)
    with pytest.raises(VidLensError) as caught:
        service.check_facts(video.id, [claim()])
    assert caught.value.code == "stale_revision"


def test_cli_rejects_oversize_claims_and_forwards_valid_array(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    path = tmp_path / "claims.json"
    path.write_text(json.dumps([claim()]))
    args = build_parser().parse_args(["check-facts", video.id, "--file", str(path)])
    assert dispatch(args, service)["checks"]
    path.write_text(" " * (64 * 1024 + 1))
    with pytest.raises(VidLensError) as caught:
        dispatch(args, service)
    assert caught.value.code == "input_too_large"


def test_missing_crop_image_is_not_given_an_image_index(tmp_path):
    result = result_content({"frames": [{"timestamp": 2, "path": str(tmp_path / "absent.png"),
                                        "crop_box": [0, 0, 1, .5]}]}, images=True)
    payload = json.loads(result.content[0].text)
    assert result.isError and payload["frames"][0]["image_index"] is None


def test_reused_crop_pixels_keep_each_cited_timestamp(tmp_path, monkeypatch):
    service, video = seeded(tmp_path, monkeypatch)
    first = service.store.frame_at(video.id, 2)
    service.store.save_frames(video.id, [Frame(6, first.path, first.width, first.height)])
    result = service.check_facts(video.id, [claim(), claim("later", timestamp=6)])
    assert [frame["timestamp"] for frame in result["frames"]] == [2, 6]
    assert result["frames"][0]["path"] == result["frames"][1]["path"]

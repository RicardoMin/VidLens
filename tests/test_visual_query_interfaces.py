"""Question probes must preserve evidence, timestamps and delivery boundaries."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from test_learning_visual import flash as flash_fixture
from test_learning_visual import seed_service

from vidlens.cli import build_parser, dispatch
from vidlens.mcp_server import create_server
from vidlens.models import Segment, VidLensError

flash = flash_fixture


@pytest.mark.asyncio
async def test_query_probes_return_only_final_images_and_ocr_failure_falls_back(tmp_path, flash, monkeypatch):
    service, video, _ = seed_service(tmp_path, flash, monkeypatch)
    result = service.read_frames(video.id, 0, 2, question="equation", max_frames=2, record_delivery=False)
    assert result["visual"]["selection"] == "timeline_fallback"
    assert result["visual"]["candidate_cursor_scope"] == "interval"
    assert result["visual"]["screen_text_status"] == "unavailable"
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM deliveries WHERE kind='frame'").fetchone()[0] == 0
    async with create_connected_server_and_client_session(create_server(service=service)) as client:
        response = await client.call_tool("read_frames", {
            "video_id": video.id, "start": 0, "end": 2, "question": "equation", "max_frames": 2,
        })
        payload = json.loads(response.content[0].text)
        assert not response.isError
        images = [item for item in response.content if item.type == "image"]
        assert len(images) <= 2
        assert all(1 <= frame["image_index"] <= len(images) for frame in payload["frames"])
        with service.store.connect() as db:
            count = db.execute("SELECT COUNT(*) FROM deliveries WHERE kind='frame'").fetchone()[0]
        assert count <= 2


@pytest.mark.asyncio
async def test_preview_then_exact_frames_preserves_image_mapping_without_false_delivery(tmp_path, flash, monkeypatch):
    service, video, _ = seed_service(tmp_path, flash, monkeypatch)
    async with create_connected_server_and_client_session(create_server(service=service)) as client:
        response = await client.call_tool("read_frames", {
            "video_id": video.id, "start": 0, "end": 2, "question": "equation", "preview": True,
        })
        payload = json.loads(response.content[0].text)
        assert not response.isError and payload["frames"] == []
        images = [item for item in response.content if item.type == "image"]
        assert images and payload["previews"][0]["image_index"] == 1
        assert payload["visual"]["selection"] == "question_preview"
        with service.store.connect() as db:
            assert db.execute("SELECT COUNT(*) FROM deliveries WHERE kind='frame'").fetchone()[0] == 0
        times = [row["timestamp"] for row in payload["candidates"][:2]]
        response = await client.call_tool("read_frames", {
            "video_id": video.id, "start": 0, "end": 2, "timestamps": times,
        })
        exact = json.loads(response.content[0].text)
        assert not response.isError
        assert [frame["requested_timestamp"] for frame in exact["frames"]] == sorted(times)
        assert all(frame["image_index"] for frame in exact["frames"])
        assert "previews" not in exact


def test_stale_source_does_not_store_probe_ocr(tmp_path, flash, monkeypatch):
    service, video, _ = seed_service(tmp_path, flash, monkeypatch)
    original = service.visual.read_at
    reads = 0

    def changed(*args, **kwargs):
        nonlocal reads
        frames = original(*args, **kwargs)
        reads += 1
        return frames

    monkeypatch.setattr(service.visual, "read_at", changed)
    monkeypatch.setattr(service, "_frame_revision", lambda _: str(bool(reads)))
    with pytest.raises(VidLensError) as caught:
        service.read_frames(video.id, 0, 2, question="equation")
    assert caught.value.code == "stale_revision"
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 0


def test_speech_change_between_selection_and_delivery_is_rejected(tmp_path, flash, monkeypatch):
    service, video, _ = seed_service(tmp_path, flash, monkeypatch)
    result = service.read_frames(video.id, 0, 2, question="equation", record_delivery=False)
    service.store.save(video, [Segment(0, 2, "corrected equation", source="manual")], [])
    with pytest.raises(VidLensError) as caught:
        service.record_frame_delivery(result)
    assert caught.value.code == "stale_revision"


def test_question_navigation_cursor_and_disabled_ocr_are_explicit_errors(tmp_path, flash, monkeypatch):
    service, video, section = seed_service(tmp_path, flash, monkeypatch)
    with pytest.raises(VidLensError, match="no frame cursor"):
        service.read_section(video.id, section["id"], question="equation", frame_cursor=0)
    with pytest.raises(VidLensError, match="enable OCR"):
        service.read_frames(video.id, 0, 2, question="equation", ocr=False)
    with pytest.raises(VidLensError, match="six short"):
        service.read_frames(video.id, 0, 2, question="equation", visual_queries=["x"] * 7)


def test_cli_forwards_question_and_distinct_screen_goals():
    from types import SimpleNamespace

    calls = []
    service = SimpleNamespace(read_frames=lambda *args, **kwargs: calls.append(kwargs))
    args = build_parser().parse_args([
        "frames", "video", "10", "20", "--question", "如何更新？",
        "--visual-query", "更新前", "--visual-query", "更新后",
    ])
    dispatch(args, service)
    assert calls[0]["question"] == "如何更新？"
    assert calls[0]["visual_queries"] == ["更新前", "更新后"]


def test_later_scan_failure_is_not_reported_as_complete(tmp_path, flash, monkeypatch):
    service, video, _ = seed_service(tmp_path, flash, monkeypatch)
    monkeypatch.setattr(service.visual, "scan", lambda *args, cursor=0, **kwargs: {
        "scan_id": "scan", "status": "complete" if cursor == 0 else "partial",
        "candidates": [{"timestamp": cursor / 2}], "next_cursor": 1 if cursor == 0 else None,
        "error": None if cursor == 0 else {"code": "decode_failed"},
    })
    result = service.read_frames(video.id, 0, 2, question="equation")
    assert result["visual"]["scan_status"] == "partial"
    assert result["visual"]["scan_error"]["code"] == "decode_failed"

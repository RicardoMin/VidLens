"""Regression cases found while reviewing preview and exact-frame integration."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image

from vidlens.mcp_server import create_server
from vidlens.models import Frame, MediaAsset, Video
from vidlens.service import VideoService


def held_slide(tmp_path, monkeypatch):
    service = VideoService(tmp_path / "library")
    path = tmp_path / "held-formula.png"
    Image.new("RGB", (854, 480), "#204868").save(path)
    video = Video("held", "https://example.com/lecture", "local", "Held formula", 200,
                  metadata={"revision": "v1"})
    service.store.save(video, [], [])
    asset = MediaAsset(path, 0, 200)
    monkeypatch.setattr(service, "_asset", lambda *args: asset)
    # The entire interval 100..150 still displays the state beginning at zero.
    monkeypatch.setattr(service.learning, "_scan_data", lambda _: (video, "scan-v1", {
        "status": "complete", "scan_id": "held-scan", "candidates": [{"timestamp": 0.0}],
    }))
    calls = []

    def read_at(_asset, times, ocr=True):
        calls.extend(times)
        return [Frame(0.0, path, 854, 480, ocr="m = max(x)", requested_timestamp=t,
                      visible_until=200.0) for t in times]

    monkeypatch.setattr(service.visual, "read_at", read_at)
    return service, video, calls


@pytest.mark.asyncio
async def test_preview_inside_unchanged_screen_still_returns_images_without_delivery(tmp_path, monkeypatch):
    service, video, calls = held_slide(tmp_path, monkeypatch)
    async with create_connected_server_and_client_session(create_server(service=service)) as client:
        response = await client.call_tool("read_frames", {
            "video_id": video.id, "start": 100, "end": 150,
            "question": "这个 max 公式如何更新？", "preview": True,
        })
        payload = json.loads(response.content[0].text)
        assert not response.isError
        assert calls, "A held screen must remain visible even without new change candidates"
        assert payload["previews"] and payload["candidates"]
        assert all(100 <= t < 150 for t in calls)
        assert all(row["timestamp"] == 0 for row in payload["candidates"])
        assert all(100 <= row["requested_timestamp"] < 150 for row in payload["candidates"])
        assert any(item.type == "image" for item in response.content)
        with service.store.connect() as db:
            assert db.execute("SELECT COUNT(*) FROM deliveries WHERE kind='frame'").fetchone()[0] == 0
        # A held frame's source PTS is before the interval. The read timestamp
        # must still round-trip through the exact retrieval API in this scope.
        times = [row["read_timestamp"] for row in payload["candidates"]]
        exact = await client.call_tool("read_frames", {
            "video_id": video.id, "start": 100, "end": 150, "timestamps": times,
        })
        exact_payload = json.loads(exact.content[0].text)
        assert not exact.isError
        assert all(row["timestamp"] == 0 for row in exact_payload["frames"])
        assert [row["requested_timestamp"] for row in exact_payload["frames"]] == times
        assert all(row["image_index"] is not None for row in exact_payload["frames"])


def test_preview_accounting_includes_contact_sheet_images(tmp_path, monkeypatch):
    service, video, _ = held_slide(tmp_path, monkeypatch)
    # Include a candidate inside the interval to isolate accounting from the
    # separate no-change-candidates regression above.
    monkeypatch.setattr(service.learning, "_scan_data", lambda _: (video, "scan-v1", {
        "status": "complete", "scan_id": "held-scan", "candidates": [{"timestamp": 100.0}],
    }))
    result = service.read_frames(video.id, 100, 150, question="max", preview=True)
    assert result["frames"] == [] and result["previews"]
    assert result["accounting"]["image_count"] == len(result["previews"])

"""Cross-layer contracts for selectable recognition and visual navigation."""

import json
from types import SimpleNamespace

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from test_asr_replacement import setup_service
from test_learning_visual import flash as flash_fixture
from test_learning_visual import seed_service

from vidlens.asr import asr_source
from vidlens.cli import build_parser, dispatch
from vidlens.mcp_server import create_server
from vidlens.models import Segment, VidLensError
from vidlens.selection import VisualSelector

flash = flash_fixture


def test_backend_switch_replaces_asr_and_keeps_cache_identity_separate(tmp_path):
    service, video, _ = setup_service(tmp_path)
    calls = []

    def recognize(asset, language, model, backend="faster-whisper", **kwargs):
        calls.append((backend, model))
        return [Segment(asset.offset, asset.offset + asset.duration, "TCP and SYN",
                        kind="transcript", source=asr_source(backend, model), language=language)]

    service._provider = SimpleNamespace(transcribe=recognize)
    cpu = service.transcribe(video.id, 0, 20, "zh", model="large-v3-turbo")
    accelerated = service.transcribe(video.id, 0, 20, "zh", backend="mlx-whisper")
    assert cpu["signature"] != accelerated["signature"]
    assert calls == [("faster-whisper", "large-v3-turbo"), ("mlx-whisper", "large-v3-turbo")]
    assert [s.source for s in service.store.segments(video.id)] == ["mlx-whisper:large-v3-turbo"]
    assert service.transcribe(video.id, 0, 20, "zh", backend="mlx-whisper")["cache_hit"]
    assert not service.transcribe(video.id, 0, 20, "zh", model="large-v3-turbo")["cache_hit"]
    assert [s.source for s in service.store.segments(video.id)] == ["whisper:large-v3-turbo"]


def test_failed_backend_upgrade_keeps_original_cues_and_history(tmp_path):
    service, video, _ = setup_service(tmp_path)
    service.transcribe(video.id, 0, 20, "zh")
    cues, metadata = service.store.segments(video.id), service.store.get(video.id).metadata

    def fail(*args, **kwargs):
        raise VidLensError("ASR_DEPENDENCY", "The selected recognizer is not installed.")

    service._provider = SimpleNamespace(transcribe=fail)
    with pytest.raises(VidLensError, match="not installed"):
        service.transcribe(video.id, 0, 20, "zh", backend="mlx-whisper")
    assert service.store.segments(video.id) == cues
    assert service.store.get(video.id).metadata == metadata


def test_cli_backend_defaults_and_visual_views_are_forwarded():
    calls = []
    service = SimpleNamespace(
        transcribe=lambda *args, **kwargs: calls.append(kwargs),
        scan_video=lambda *args, **kwargs: calls.append(kwargs),
        read_section=lambda *args, **kwargs: calls.append(kwargs),
    )
    parser = build_parser()
    dispatch(parser.parse_args(["transcribe", "id", "--backend", "mlx-whisper"]), service)
    dispatch(parser.parse_args(["scan", "id", "--view", "full"]), service)
    dispatch(parser.parse_args(["section", "id", "section", "--frame-view", "full"]), service)
    assert calls[0]["backend"] == "mlx-whisper" and calls[0]["model"] == "large-v3-turbo"
    assert calls[1]["view"] == calls[2]["frame_view"] == "full"


@pytest.mark.asyncio
async def test_navigation_keeps_private_pixels_local_and_full_candidates_reachable(tmp_path, flash, monkeypatch):
    service, video, section = seed_service(tmp_path, flash, monkeypatch)
    full = service.scan_video(video.id, view="full")
    raw = list(full["candidates"])

    def compact(_self, _asset, scan):
        rows = [{**row, "signature": "private descriptor", "source_candidate_cursor": i}
                for i, row in enumerate(scan["candidates"]) if i in (0, len(raw) - 1)]
        return {"status": "complete", "selection_id": "fixture", "metrics": {},
                "navigation_candidates": rows}

    monkeypatch.setattr(VisualSelector, "build", compact)
    async with create_connected_server_and_client_session(create_server(service=service)) as client:
        response = await client.call_tool("scan_video", {"video_id": video.id, "limit": 1})
        page = json.loads(response.content[0].text)
        assert page["view"] == "navigation" and page["candidate_count"] == 2
        assert page["raw_candidate_count"] == len(raw)
        assert "signature" not in json.dumps(page)
        response = await client.call_tool("scan_video", {
            "video_id": video.id, "view": page["view"], "cursor": page["next_cursor"],
        })
        last = json.loads(response.content[0].text)
        assert last["next_cursor"] is None
        assert last["candidates"][0]["source_candidate_cursor"] == len(raw) - 1
        response = await client.call_tool("read_section", {
            "video_id": video.id, "section_id": section["id"], "frame_view": "navigation",
            "frame_cursor": 0, "max_tokens": 8000,
        })
        chapter = json.loads(response.content[0].text)
        assert not response.isError and chapter["frames"]
        assert chapter["visual"]["view"] == "navigation"
    assert service.scan_video(video.id, view="full")["candidates"] == raw


def test_static_chapter_without_navigation_event_still_returns_its_screen(tmp_path, flash, monkeypatch):
    service, video, _ = seed_service(tmp_path, flash, monkeypatch)
    service.scan_video(video.id, view="full")
    index = {"navigation_candidates": [{"timestamp": 0}]}
    monkeypatch.setattr(service.learning, "_candidate_view", lambda *args: (
        index["navigation_candidates"], {"view": "navigation"}, index
    ))
    frames, metadata = service.learning._frames(video.id, {"start": 1.5, "end": 1.9}, None, 2)
    assert len(frames) == 2
    assert metadata["view"] == "navigation"
    assert all(1.5 <= frame["requested_timestamp"] < 1.9 for frame in frames)


def test_selection_failure_returns_an_explicit_full_view_cursor(tmp_path, flash, monkeypatch):
    service, video, _ = seed_service(tmp_path, flash, monkeypatch)

    def unavailable(*args):
        raise VidLensError("selection_decode", "Selection decoder unavailable.")

    monkeypatch.setattr(VisualSelector, "build", unavailable)
    first = service.scan_video(video.id, view="navigation", limit=1)
    assert first["view"] == "full" and first["selection_status"] == "unavailable"
    second = service.scan_video(video.id, view=first["view"], cursor=first["next_cursor"], limit=1)
    assert second["view"] == "full" and second["candidates"] != first["candidates"]


def test_initial_overview_keeps_stable_timeline_while_navigation_stays_compact(tmp_path, flash, monkeypatch):
    service, video, section = seed_service(tmp_path, flash, monkeypatch)
    scan = service.scan_video(video.id, view="full")
    raw = scan["candidates"]
    requested = []

    def read(_id, times, **kwargs):
        requested.extend(times)
        return [{"timestamp": stamp, "requested_timestamp": stamp} for stamp in times]

    monkeypatch.setattr(service, "read_frame_times", read)
    monkeypatch.setattr(service.learning, "_candidate_view", lambda *args: (
        raw[:1], {"view": "navigation"}, {"navigation_candidates": raw[:1]}
    ))
    service.learning._frames(video.id, section, None, 4)
    expected = service.learning._overview_candidates(section, raw, 4)
    assert requested == [row["timestamp"] for row in expected]
    requested.clear()
    _, visual = service.learning._frames(video.id, section, 0, 4)
    assert requested == [raw[0]["timestamp"]]
    assert visual["candidate_count"] == 1 and visual["raw_candidate_count"] == len(raw)

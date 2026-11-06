"""Offline integration of scan candidates, source pixels, speech and delivery state."""

import base64
import io
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image
from test_storyboard import lecture as synthetic_lecture

from vidlens import mcp_server
from vidlens.mcp_server import create_server
from vidlens.media_cache import MediaCache
from vidlens.models import MediaAsset, Segment, Video, VidLensError
from vidlens.service import VideoService
from vidlens.visual import VisualReader

lecture = synthetic_lecture


def test_optional_whole_video_failure_backs_off_across_clients_and_source_revisions(tmp_path):
    service = VideoService(tmp_path / "library")
    video = Video("remote", "", "youtube", "Lecture", 1200, metadata={"revision": "a"})
    attempts = []

    class Provider:
        def fetch_media(self, video, target, *, start, end, audio_only):
            attempts.append((start, end))
            if start == 0 and end == video.duration:
                raise VidLensError("source_failed", "Whole download failed.")
            target.write_bytes(b"interval")
            return MediaAsset(target, start, end - start)

    provider = Provider()
    assert service.media.optional_full_video(video, lambda: provider) is None
    other = MediaCache(service.store)
    assert other.optional_full_video(video, lambda: provider) is None
    assert other.get(video, 9, 11, False, lambda: provider).duration == 2
    assert attempts == [(0, 1200), (9, 11)]
    # A new source may support whole acquisition; old failures cannot suppress it.
    video.metadata["revision"] = "b"
    assert other.optional_full_video(video, lambda: provider) is None
    assert attempts[-1] == (0, 1200)


def seed_service(tmp_path, asset, monkeypatch, *, remote=False):
    service = VideoService(tmp_path / "library")
    metadata = {"revision": "source-v1"}
    if not remote:
        metadata["local_media_path"] = str(asset.path)
    video = Video(
        "visual-lesson",
        "https://www.youtube.com/watch?v=aircAruvnKk",
        "youtube",
        "A visual explanation",
        asset.duration,
        metadata=metadata,
    )
    service.store.save(
        video,
        [
            Segment(0.125, 0.875, "The equation is shown briefly.", source="manual"),
            Segment(
                1.125, min(1.875, asset.duration), "The caveat is stated aloud.", source="manual"
            ),
        ],
        [],
    )
    # The visual pipeline is real. OCR is separate and must not invoke host tools/models.
    monkeypatch.setattr(service.visual, "_ocr", lambda frames: None)
    section = service.study_status(video.id)["sections"][0]
    return service, video, section


@pytest.fixture
def flash(tmp_path):
    path = tmp_path / "one-frame-equation.mp4"
    subprocess.run(
        [
            VisualReader._ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=320x180:r=30:d=2",
            "-vf",
            "drawbox=color=white:t=fill:enable='eq(n,32)'",
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
    return MediaAsset(path, duration=2)


@pytest.mark.asyncio
async def test_mcp_section_returns_real_flash_pixels_and_original_speech_together(
    tmp_path, flash, monkeypatch
):
    service, video, section = seed_service(tmp_path, flash, monkeypatch)
    server = create_server(service=service)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool(
            "read_section",
            {
                "video_id": video.id,
                "section_id": section["id"],
                "max_tokens": 6000,
            },
        )
        assert not result.isError, result
        assert result.structuredContent is None
        assert result.content[0].type == "text"
        assert all(item.type == "image" for item in result.content[1:])
        payload = json.loads(result.content[0].text)
        assert [item["text"] for item in payload["evidence"]] == [
            "The equation is shown briefly.",
            "The caveat is stated aloud.",
        ]
        assert all(item["kind"] != "ocr" for item in payload["evidence"])
        frame = min(payload["frames"], key=lambda item: abs(item["timestamp"] - 32 / 30))
        assert frame["timestamp"] == pytest.approx(32 / 30, abs=0.000001)
        assert frame["evidence_id"] == f"frame:{frame['timestamp']}"
        returned = result.content[frame["image_index"]]
        with Image.open(io.BytesIO(base64.b64decode(returned.data))) as image:
            assert image.getpixel((160, 90))[0] > 245
        assert payload["visual"]["selection"] == "timeline_overview"
        assert payload["progress"]["text_status"] == "delivered"
        assert payload["progress"]["visual_status"] == "sampled"
        saved = await client.call_tool(
            "save_note",
            {
                "video_id": video.id,
                "section_id": section["id"],
                "revision": payload["revision"],
                "note": {
                    "title": "Brief visual",
                    "summary": "A white frame appears briefly.",
                    "concepts": ["visual transition"],
                    "roles": ["example"],
                    "relations": [],
                    "claims": [
                        {
                            "text": "The brief frame is white.",
                            "evidence_ids": [frame["evidence_id"]],
                        }
                    ],
                },
            },
        )
        assert not saved.isError, saved
        stored = service.learning.study.get_section(video.id, section["id"])
        assert stored["note"]["claims"][0]["evidence_ids"] == [frame["evidence_id"]]


def test_read_at_preserves_fractional_candidate_and_warm_read_reuses_pixels(
    tmp_path, flash, monkeypatch
):
    reader = VisualReader(tmp_path / "visual")
    scan = reader.scan(flash)
    target = next(row["timestamp"] for row in scan["candidates"] if 1.06 < row["timestamp"] < 1.07)
    frames = reader.read_at(flash, [target], ocr=False)
    assert frames[0].timestamp == pytest.approx(target, abs=0.000001)
    with Image.open(frames[0].path) as image:
        assert image.getpixel((160, 90))[0] > 245

    def unexpected(*args, **kwargs):
        pytest.fail("A repeated candidate must not invoke FFmpeg")

    monkeypatch.setattr(subprocess, "run", unexpected)
    assert reader.read_at(flash, [target], ocr=False)[0].to_dict() == frames[0].to_dict()


@pytest.mark.parametrize("options", [{"max_tokens": 768}, {"cursor": "invalid"}])
def test_failed_section_page_never_advances_text_or_visual_delivery(
    tmp_path, lecture, monkeypatch, options
):
    service, video, section = seed_service(tmp_path, lecture, monkeypatch)
    with pytest.raises(VidLensError):
        service.read_section(video.id, section["id"], **options)
    assert service.store.deliveries(video.id) == []
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM text_deliveries").fetchone()[0] == 0
    progress = service.learning.study.get_section(video.id, section["id"])["progress"]
    assert progress["text_segments_delivered"] == 0
    assert progress["visual_status"] == "not_delivered"
    successful = service.read_section(video.id, section["id"])
    assert successful["progress"]["text_status"] == "delivered"
    assert successful["progress"]["visual_status"] == "sampled"


def test_source_revision_change_before_delivery_leaves_no_progress(tmp_path, lecture, monkeypatch):
    service, video, section = seed_service(tmp_path, lecture, monkeypatch)
    original = service.reader.read_interval

    def change_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        video.metadata["revision"] = "source-v2"
        service.store.save(video, [Segment(0, 2, "Replacement speech.", source="manual")], [])
        return result

    monkeypatch.setattr(service.reader, "read_interval", change_after_read)
    with pytest.raises(VidLensError) as error:
        service.read_section(video.id, section["id"])
    assert error.value.code in {"stale_revision", "section_not_found"}
    assert service.store.deliveries(video.id) == []
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM text_deliveries").fetchone()[0] == 0


def test_scan_database_pagination_is_reused_and_source_change_invalidates_it(
    tmp_path, lecture, monkeypatch
):
    service, video, _ = seed_service(tmp_path, lecture, monkeypatch)
    cold = service.scan_video(video.id, limit=2)
    warm = service.scan_video(video.id, cursor=2, limit=2)
    assert cold["cached"] is False
    assert warm["cached"] is True
    assert warm["cursor"] == 2
    assert warm["scan_id"] == cold["scan_id"]
    stat = lecture.path.stat()
    os.utime(lecture.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    changed = service.scan_video(video.id, limit=2)
    assert changed["cached"] is False
    assert changed["scan_id"] != cold["scan_id"]


def test_scans_from_two_services_share_one_completed_scan(tmp_path, lecture, monkeypatch):
    service, video, _ = seed_service(tmp_path, lecture, monkeypatch)
    other = VideoService(service.home)
    _ = other.learning
    original, calls = VisualReader.scan, []

    def counted(self, *args, **kwargs):
        calls.append(args[0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VisualReader, "scan", counted)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda reader: reader.scan_video(video.id), [service, other]))
    assert len(calls) == 1
    assert all(result["status"] == "complete" for result in results)
    assert results[0]["candidates"] == results[1]["candidates"]
    assert sorted(result["cached"] for result in results) == [False, True]


def test_failed_scan_uses_chapter_wide_fallback_and_retries(tmp_path, lecture, monkeypatch):
    service, video, section = seed_service(tmp_path, lecture, monkeypatch)
    original = service.visual.scan
    failure = {
        "status": "failed",
        "candidates": [],
        "next_cursor": None,
        "candidate_count": 0,
        "coverage": {"complete": False},
        "warnings": ["Synthetic scan timeout."],
        "error": {"code": "scan_timeout", "message": "Synthetic scan timeout."},
    }
    monkeypatch.setattr(service.visual, "scan", lambda *args, **kwargs: failure.copy())
    result = service.read_section(video.id, section["id"], frame_limit=4)
    assert len(result["frames"]) == 4
    assert (
        max(frame["timestamp"] for frame in result["frames"])
        - min(frame["timestamp"] for frame in result["frames"])
        > 12
    )
    assert result["visual"]["scan_status"] == "failed"
    assert "scan_timeout" in json.dumps(result["visual"])
    monkeypatch.setattr(service.visual, "scan", original)
    retried = service.scan_video(video.id)
    assert retried["status"] == "complete"
    assert retried["cached"] is False


def test_high_resolution_candidates_reuse_one_media_asset(tmp_path, lecture, monkeypatch):
    service, video, section = seed_service(tmp_path, lecture, monkeypatch, remote=True)
    downloads = []

    class LocalProvider:
        def fetch_media(self, video, target, *, start, end, audio_only, max_height=None):
            downloads.append((start, end, audio_only, max_height))
            shutil.copyfile(lecture.path, target)
            return MediaAsset(target, 0, lecture.duration)

    service._provider = LocalProvider()
    first = service.read_section(video.id, section["id"])
    assert len(first["frames"]) == 4
    assert [item[-1] for item in downloads] == [360, None]
    again = service.read_section(video.id, section["id"], frame_cursor=0)
    assert again["visual"]["selection"] == "sequence"
    assert len(downloads) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["read_section", "read_frames"])
async def test_failed_mcp_image_encoding_never_marks_images_delivered(
    tmp_path, flash, monkeypatch, method
):
    service, video, section = seed_service(tmp_path, flash, monkeypatch)

    def unavailable(path):
        raise OSError("Synthetic image encoding failure")

    monkeypatch.setattr(mcp_server, "_image_content", unavailable)
    arguments = {"video_id": video.id}
    if method == "read_section":
        arguments["section_id"] = section["id"]
    else:
        arguments.update(start=0, end=2, max_frames=2, ocr=False)
    async with create_connected_server_and_client_session(create_server(service=service)) as client:
        result = await client.call_tool(method, arguments)
    assert result.isError
    assert all(item.type == "text" for item in result.content)
    payload = json.loads(result.content[0].text)
    assert all(frame["image_index"] is None for frame in payload["frames"])
    assert not [row for row in service.store.deliveries(video.id) if row["kind"] == "frame"]
    if method == "read_section":
        assert payload["progress"]["visual_status"] == "not_delivered"
        assert payload["progress"]["text_status"] == "delivered"  # Speech is still returned.


@pytest.mark.asyncio
async def test_partially_encoded_mcp_images_record_only_the_transmitted_groups(
    tmp_path, flash, monkeypatch
):
    service, video, section = seed_service(tmp_path, flash, monkeypatch)
    original = mcp_server._image_content

    def reject_white_frame(path):
        with Image.open(path) as image:
            if image.getpixel((160, 90))[0] > 245:
                raise OSError("Synthetic failure of the brief white frame")
        return original(path)

    monkeypatch.setattr(mcp_server, "_image_content", reject_white_frame)
    async with create_connected_server_and_client_session(create_server(service=service)) as client:
        result = await client.call_tool(
            "read_section",
            {
                "video_id": video.id,
                "section_id": section["id"],
            },
        )
    assert not result.isError
    payload = json.loads(result.content[0].text)
    assert any(frame["image_index"] is None for frame in payload["frames"])
    transmitted = {frame["timestamp"] for frame in payload["frames"] if frame["image_index"]}
    delivered = {
        row["start"] for row in service.store.deliveries(video.id) if row["kind"] == "frame"
    }
    assert delivered == transmitted
    assert all(abs(stamp - 32 / 30) > 0.00002 for stamp in delivered)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["read_section", "read_frames"])
async def test_source_refresh_during_mcp_encoding_rejects_stale_delivery(
    tmp_path, flash, monkeypatch, method
):
    service, video, section = seed_service(tmp_path, flash, monkeypatch)
    original, refreshed = mcp_server._image_content, False

    def refresh_then_encode(path):
        nonlocal refreshed
        if not refreshed:
            video.metadata["revision"] = "source-v2"
            with service._file_lock("content", video.id):
                service.store.save(video, [Segment(0, 2, "New source speech.")], [])
            refreshed = True
        return original(path)

    monkeypatch.setattr(mcp_server, "_image_content", refresh_then_encode)
    arguments = {"video_id": video.id}
    if method == "read_section":
        arguments["section_id"] = section["id"]
    else:
        arguments.update(start=0, end=2, max_frames=2, ocr=False)
    async with create_connected_server_and_client_session(create_server(service=service)) as client:
        result = await client.call_tool(method, arguments)
    assert result.isError
    error = json.loads(result.content[0].text)["error"]
    assert error["code"] in {"stale_revision", "section_not_found"}
    assert service.store.deliveries(video.id) == []


@pytest.mark.asyncio
async def test_final_mcp_metadata_budget_failure_records_no_delivery(tmp_path, flash, monkeypatch):
    service, video, section = seed_service(tmp_path, flash, monkeypatch)
    original = service.budget.recount

    def reject_encoded_metadata(payload):
        if "transmitted_image_count" in payload.get("accounting", {}):
            raise VidLensError("budget_too_small", "Synthetic final metadata overflow")
        return original(payload)

    monkeypatch.setattr(service.budget, "recount", reject_encoded_metadata)
    async with create_connected_server_and_client_session(create_server(service=service)) as client:
        result = await client.call_tool(
            "read_section",
            {
                "video_id": video.id,
                "section_id": section["id"],
            },
        )
    assert result.isError
    assert json.loads(result.content[0].text)["error"]["code"] == "budget_too_small"
    assert service.store.deliveries(video.id) == []
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM text_deliveries").fetchone()[0] == 0


def test_delivery_commit_excludes_a_second_service_source_refresh(tmp_path, lecture, monkeypatch):
    service, video, section = seed_service(tmp_path, lecture, monkeypatch)
    other = VideoService(service.home)
    enter, refreshed = Event(), Event()
    original = service.store.delivered_text

    def refresh():
        assert enter.wait(timeout=5)
        with other._file_lock("content", video.id):
            video.metadata["revision"] = "source-v2"
            other.store.save(
                video, [Segment(0, 2, "New source speech.")], [], invalidate_derived=True
            )
        refreshed.set()

    def delivery_while_refresh_waits(*args, **kwargs):
        enter.set()
        # The second service must wait for the revision-check/delivery critical section.
        assert not refreshed.wait(timeout=0.1)
        return original(*args, **kwargs)

    monkeypatch.setattr(service.store, "delivered_text", delivery_while_refresh_waits)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(refresh)
        try:
            result = service.read_section(video.id, section["id"])
            assert result["progress"]["text_status"] == "delivered"
        finally:
            enter.set()
            future.result(timeout=5)
    assert refreshed.is_set()
    assert other.store.deliveries(video.id) == []  # Refresh cleared old-source progress.

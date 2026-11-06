"""Behavioral regressions found in the November 6 cross-layer review."""

import json
import os

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image
from test_learning import library
from test_learning_visual import flash as flash_fixture
from test_learning_visual import seed_service

from vidlens import mcp_server
from vidlens.models import Frame, Segment, Video, VidLensError
from vidlens.service import VideoService
from vidlens.storyboard import StoryboardScanner

flash = flash_fixture


def test_chapter_preserves_ocr_warning_without_losing_images(tmp_path, flash, monkeypatch):
    service, video, section = seed_service(tmp_path, flash, monkeypatch)

    def unavailable_ocr(frames):
        service.visual.last_warnings.append("OCR unavailable; original images remain readable.")

    monkeypatch.setattr(service.visual, "_ocr", unavailable_ocr)
    page = service.read_section(video.id, section["id"], max_tokens=8000)
    assert page["frames"] and page["evidence"]
    assert all(frame["warnings"] and os.path.isfile(frame["path"]) for frame in page["frames"])
    assert service.study_status(video.id)["progress"]["visual_sampled_sections"] > 0


@pytest.mark.parametrize("question", [
    "Explain model.summary()", "Why is overview a list?", "解释 `summary` 函数",
])
def test_technical_identifiers_do_not_turn_questions_into_whole_video_tasks(tmp_path, question):
    service, video = library(tmp_path, [Segment(0, 1200, "model.summary() prints the overview list.")])
    result = service.read_video(video.url, question)
    assert result["mode"] == "question_evidence"
    assert any("summary()" in row["text"] for row in result["evidence"])


@pytest.mark.parametrize("question", [
    "用一张图讲解这个视频", "Give me an overview of this lecture", "brief summary",
])
def test_whole_video_explanations_return_a_complete_chapter_plan(tmp_path, question):
    service, video = library(tmp_path)
    result = service.read_video(video.url, question)
    assert result["mode"] == "study_plan"
    assert result["section_count"] > 0


@pytest.mark.asyncio
async def test_encoded_image_failure_keeps_visual_cursor_at_the_missing_candidate(
    tmp_path, flash, monkeypatch
):
    service, video, section = seed_service(tmp_path, flash, monkeypatch)
    scan = service.scan_video(video.id)
    target_cursor = next(i for i, c in enumerate(scan["candidates"])
                         if abs(c["timestamp"] - 32 / 30) < 0.00001)
    original = mcp_server._image_content

    def reject_flash(path):
        with Image.open(path) as picture:
            if picture.getpixel((160, 90))[0] > 245:
                raise OSError("Image cannot be transmitted on this attempt")
        return original(path)

    monkeypatch.setattr(mcp_server, "_image_content", reject_flash)
    async with create_connected_server_and_client_session(
        mcp_server.create_server(service=service)
    ) as client:
        arguments = {"video_id": video.id, "section_id": section["id"],
                     "frame_cursor": 0, "frame_limit": 4}
        result = await client.call_tool("read_section", arguments)
        payload = json.loads(result.content[0].text)
        assert payload["visual"]["status"] == "partial"
        assert payload["visual"]["next_frame_cursor"] == target_cursor
        assert payload["evidence"]  # Text remains usable when a picture fails.
        monkeypatch.setattr(mcp_server, "_image_content", original)
        arguments["frame_cursor"] = payload["visual"]["next_frame_cursor"]
        retried = await client.call_tool("read_section", arguments)
        page = json.loads(retried.content[0].text)
        white = next(f for f in page["frames"] if abs(f["timestamp"] - 32 / 30) < 0.00001)
        assert white["image_index"] is not None
        assert page["visual"]["next_frame_cursor"] != target_cursor


def test_crop_does_not_reuse_pixels_after_local_media_replacement(tmp_path, flash, monkeypatch):
    service, video, _ = seed_service(tmp_path, flash, monkeypatch)
    old = service.read_frame_times(video.id, [1.1])[0]
    stat = flash.path.stat()
    os.utime(flash.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    with pytest.raises(VidLensError) as error:
        service.crop_frame(video.id, old["timestamp"], [0, 0, 0.5, 0.5])
    assert error.value.code == "frame_not_found"


def test_export_keeps_cited_pictures_and_relations_without_rereading_every_chapter(
    tmp_path, monkeypatch
):
    service, video = library(tmp_path, [Segment(0, 1200, "Cached values are reused.")])
    plan = service.study_status(video.id)
    section_id = plan["sections"][0]["id"]
    cue = service.store.segments(video.id)[0]
    picture = tmp_path / "source picture.jpg"
    Image.new("RGB", (64, 32), "blue").save(picture)
    service.store.save_frames(video.id, [Frame(20, picture, 64, 32)])
    service.save_note(video.id, section_id, plan["revision"], {
        "title": "Reuse a previous value", "summary": "The next query reads the existing cache.",
        "claims": [{"text": "The screen shows cached values.", "evidence_ids": ["frame:20"]}],
        "relations": [{"source": "cache", "target": "recomputation",
                       "relation": "removes repeated projection",
                       "evidence_ids": [cue.id, "frame:20"]}],
    })
    def unexpected(*args, **kwargs):
        pytest.fail("Export should use one note snapshot, not reread each chapter's evidence.")
    monkeypatch.setattr(service.learning.study, "get_section", unexpected)
    result = service.export_study(video.id)
    assert "## Reuse a previous value" in result["markdown"]
    assert "removes repeated projection" in result["markdown"]
    assert f"(<{picture}>)" in result["markdown"]
    assert "画面 20.0s" in result["markdown"]
    assert result["image_count"] == 1
    picture.unlink()
    without_file = service.export_study(video.id)
    assert without_file["image_count"] == 0
    assert "画面 20.0s" in without_file["markdown"]  # The source-time link still works.


def test_saved_relationship_bridge_survives_compact_search_and_question_context(tmp_path):
    service, video = library(tmp_path, [
        Segment(0, 390, "Quantization changes storage."),
        Segment(400, 790, "Memory footprint affects batch size."),
        Segment(800, 1190, "Batch size affects throughput."),
    ])
    plan = service.study_status(video.id)
    plan = service.revise_sections(video.id, plan["revision"], [
        {"start": i * 400, "end": (i + 1) * 400, "title": f"Stage {i}"}
        for i in range(3)
    ])
    edges = [("quantization", "memory footprint"),
             ("memory footprint", "batch size"), ("batch size", "throughput")]
    cues = service.store.segments(video.id)
    for section, cue, (source, target) in zip(plan["sections"], cues, edges):
        service.save_note(video.id, section["id"], plan["revision"], {
            "title": section["title"], "summary": cue.text,
            "relations": [{"source": source, "target": target, "relation": "affects",
                           "evidence_ids": [cue.id]}],
        })
    question = "How does quantization affect throughput?"
    for result in [service.search_knowledge(video.id, question, max_tokens=4000),
                   service.read_video(video.url, question, max_tokens=4000)]:
        hits = result.get("matches", result.get("knowledge", []))
        relations = [r for hit in hits for r in hit.get("relations", [])]
        assert {(r["source"], r["target"]) for r in relations} == set(edges)
        assert all(r["evidence_ids"] for r in relations)


@pytest.mark.parametrize("budget", [768, 1200])
def test_small_knowledge_budget_retains_path_endpoints_and_citations(tmp_path, budget):
    service, video = library(tmp_path, [Segment(0, 60, "Quantization affects memory and throughput.")])
    plan = service.study_status(video.id)
    cue = service.store.segments(video.id)[0]
    edges = [("quantization", "memory"), ("memory", "throughput")]
    service.save_note(video.id, plan["sections"][0]["id"], plan["revision"], {
        "title": "Effects of quantization", "summary": cue.text,
        "relations": [{"source": source, "target": target, "relation": "changes " * 56,
                       "evidence_ids": [cue.id]} for source, target in edges],
    })
    result = service.search_knowledge(video.id, "quantization throughput", max_tokens=budget)
    assert result["matches"]
    relations = result["matches"][0]["relations"]
    assert [(edge["source"], edge["target"]) for edge in relations] == edges
    assert all(edge["evidence_ids"] == [cue.id] for edge in relations)
    assert result["accounting"]["text_tokens_estimate"] <= budget


def test_local_media_replaced_during_crop_rejects_the_old_image(tmp_path, flash, monkeypatch):
    service, video, _ = seed_service(tmp_path, flash, monkeypatch)
    frame = service.read_frame_times(video.id, [0.5])[0]
    crop = service.visual.crop

    def change_source(*args):
        cropped = crop(*args)
        flash.path.write_bytes(b"a replacement video arrived during cropping")
        return cropped

    monkeypatch.setattr(service.visual, "crop", change_source)
    with pytest.raises(VidLensError, match="changed during cropping"):
        service.crop_frame(video.id, frame["timestamp"], [0, 0, 0.5, 0.5])


def test_local_file_changed_during_decoding_cannot_publish_mixed_source_frames(tmp_path):
    service = VideoService(tmp_path / "library")
    media = tmp_path / "video.mp4"
    media.write_bytes(b"instrumented source")
    picture = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 32), "red").save(picture)
    video = Video("local", "", "local", "Local lecture", 10,
                  metadata={"local_media_path": str(media)})
    service.store.save(video, [Segment(0, 10, "An original explanation.")], [])
    class ChangingVisual:
        def read(self, *args, **kwargs):
            media.write_bytes(b"a replacement source with different size")
            return [Frame(1, picture, 32, 32)]
    service._visual = ChangingVisual()
    with pytest.raises(VidLensError, match="changed during decoding"):
        service.read_frames(video.id, 0, 5)
    assert service.store.frame_at(video.id, 1) is None
    assert service.store.deliveries(video.id) == []


def test_algorithm_upgrade_invalidates_the_service_scan_cache(tmp_path, flash, monkeypatch):
    service, video, _ = seed_service(tmp_path, flash, monkeypatch)
    original = service.scan_video(video.id)
    assert service.scan_video(video.id)["cached"] is True
    monkeypatch.setattr(StoryboardScanner, "VERSION", StoryboardScanner.VERSION + 1)
    upgraded = service.scan_video(video.id)
    assert upgraded["cached"] is False
    assert upgraded["scan_id"] != original["scan_id"]

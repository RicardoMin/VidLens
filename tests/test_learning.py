"""Regression checks for whole-video reading and reusable grounded knowledge."""

from itertools import pairwise

import pytest

from vidlens.budget import compact_json
from vidlens.models import Segment, Video, VidLensError
from vidlens.service import VideoService


def library(tmp_path, segments=None):
    service = VideoService(tmp_path)
    video = Video("lecture", "https://www.youtube.com/watch?v=aircAruvnKk", "youtube", "Course", 1200)
    if segments is None:
        segments = [Segment(i * 20, i * 20 + 18, f"Topic {i}. " + "Attention stores keys and values. " * 8)
                    for i in range(60)]
    service.store.save(video, segments, [])
    return service, video


@pytest.mark.parametrize("question", ["总结这个视频", "帮我梳理课程内容", "Summarize this lecture", ""])
def test_summary_intent_reads_chapters_instead_of_query_matching(tmp_path, question):
    service, video = library(tmp_path)
    result = service.read_video(video.url, question, max_tokens=1800)
    assert result["mode"] == "study_plan"
    sections = list(result["sections"])
    while result["next_cursor"] is not None:
        result = service.study_status(video.id, cursor=result["next_cursor"], max_tokens=1800)
        assert service.budget.count(compact_json(result)) <= 1800
        sections.extend(result["sections"])
    assert sections[0]["start"] == 0 and sections[-1]["end"] == video.duration
    assert all(a["end"] == b["start"] for a, b in pairwise(sections))
    for section in sections:
        cursor = None
        while True:
            page = service.read_section(video.id, section["id"], max_tokens=2000,
                                        cursor=cursor, include_frames=False)
            assert service.budget.count(compact_json(page)) <= 2000
            cursor = page["next_cursor"]
            if cursor is None:
                break
    restarted = VideoService(tmp_path)
    assert restarted.study_status(video.id)["progress"]["text_delivered_sections"] == len(sections)


def test_character_pages_remain_valid_when_ocr_arrives_and_accumulate_delivery(tmp_path):
    text = "详细推导与符号解释。" * 1600
    service, video = library(tmp_path, [Segment(0, 1200, text)])
    section = service.study_status(video.id)["sections"][0]
    pieces, cursor = [], None
    for iteration in range(150):
        page = service.read_section(video.id, section["id"], max_tokens=1800,
                                    cursor=cursor, include_frames=False)
        for evidence in page["evidence"]:
            pieces.append(evidence["text"][:-1] if evidence.get("excerpt_truncated") else evidence["text"])
        if iteration == 0:
            assert page["progress"]["text_status"] == "pending"
            service.store.add_segments(video.id, [Segment(1, 1.001, "screen", kind="ocr")])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    else:
        pytest.fail("Text pagination did not terminate")
    assert "".join(pieces) == text
    assert page["progress"]["text_status"] == "delivered"


def test_notes_sources_search_export_and_layout_rename_preserve_provenance(tmp_path):
    service, video = library(tmp_path, [Segment(0, 1200, "Attention uses cached keys and values.")])
    plan = service.study_status(video.id)
    section = plan["sections"][0]
    page = service.read_section(video.id, section["id"], include_frames=False)
    note = {"title": "KV cache", "summary": "Reuse cached keys and values during decoding.",
            "concepts": ["KV cache", "attention"], "roles": ["mechanism"],
            "claims": [{"text": "Keys and values are cached.", "evidence_ids": [page["evidence"][0]["id"]]}]}
    service.save_note(video.id, section["id"], plan["revision"], note)
    reference = {"url": "https://example.org/paper", "title": "Attention paper",
                 "excerpt": "An external source excerpt supplied by the client.",
                 "explanation": "Explains attention and KV cache.", "concepts": ["KV cache"]}
    added = service.add_reference(video.id, section["id"], plan["revision"], reference)
    renamed = service.revise_sections(video.id, plan["revision"],
                                      [{"start": 0, "end": 1200, "title": "Caching during decoding"}])
    matches = service.search_knowledge(video.id, "KV cache")["matches"]
    assert {match["kind"] for match in matches} == {"section_note", "external_reference"}
    assert next(match for match in matches if match["kind"] == "external_reference")["id"] == added["id"]
    assert renamed["sections"][0]["title"] == "Caching during decoding"
    answer = service.read_video(video.url, "What is KV cache?", max_tokens=4000)
    assert answer["knowledge"]
    exported = service.export_study(video.id)["markdown"]
    assert "https://example.org/paper" in exported and "&t=0" in exported
    with pytest.raises(VidLensError, match="source|layout|changed"):
        service.save_note(video.id, section["id"], plan["revision"], note)
    service.store.save(video, [Segment(0, 1200, "Replacement lecture.")], [], invalidate_derived=True)
    assert not service.search_knowledge(video.id, "KV cache")["matches"]


def test_explicit_intent_and_external_url_validation(tmp_path):
    service, video = library(tmp_path, [Segment(0, 1200, "The summary function prints a value.")])
    assert service.read_video(video.url, "summary function", intent="question")["mode"] == "question_evidence"
    with pytest.raises(VidLensError):
        service.read_video(video.url, intent="question")
    plan = service.study_status(video.id)
    reference = {"url": "javascript:alert(1)", "title": "Invalid", "excerpt": "Text", "explanation": "Context"}
    with pytest.raises(VidLensError, match="HTTP"):
        service.add_reference(video.id, plan["sections"][0]["id"], plan["revision"], reference)

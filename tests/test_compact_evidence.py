"""Compact transport must preserve every source character, citation and read state."""

import json
from collections import defaultdict

import pytest

from vidlens.budget import compact_json
from vidlens.evidence_format import expand_evidence
from vidlens.models import Segment, Video, VidLensError
from vidlens.service import VideoService


def library(tmp_path, cues, duration=180):
    service = VideoService(tmp_path)
    video = Video("lesson", "https://www.bilibili.com/video/BV1KT2yBJESn/?p=2",
                  "bilibili", "Technical lesson", duration)
    service.store.save(video, cues, [])
    return service, video, service.store.segments(video.id)


def test_compact_pages_reconstruct_mixed_tracks_and_exact_characters(tmp_path):
    cues = [Segment(i, i + 1, f"第{i}步：x >= 3；单位μs\n原样保留。 ",
                    kind="transcript", source="asr" if i < 55 else "corrected",
                    language="zh" if i < 55 else "mixed") for i in range(100)]
    service, video, stored = library(tmp_path, cues)
    expected = {cue.id: cue for cue in stored}
    received, cursor, pages = defaultdict(str), None, 0
    while True:
        page = service.reader.read_interval(video.id, 0, 180, max_tokens=1600, cursor=cursor,
                                            representation="compact")
        assert service.budget.count(compact_json(page)) <= 1600
        assert page["source_url"].endswith("?p=2")
        for paragraph in page["evidence"]:
            track = page["tracks"][paragraph["track"]]
            for evidence_id, _, _ in paragraph["refs"]:
                original = expected[evidence_id]
                assert (track["kind"], track["source"], track["language"]) == (
                    original.kind, original.source, original.language,
                )
                assert paragraph["start"] <= original.start < original.end <= paragraph["end"]
        for item in expand_evidence(json.loads(compact_json(page))):
            assert item["text_offset"] == len(received[item["id"]])
            received[item["id"]] += item["text"]
        pages += 1
        cursor = page["next_cursor"]
        if cursor is None:
            break
        assert pages < 100
    assert received == {key: cue.text for key, cue in expected.items()}
    assert {item["evidence_id"] for item in service.store.deliveries(video.id)} == set(expected)


def test_long_cue_switches_representation_and_records_only_transmitted_characters(tmp_path):
    text = "宽高误差：sqrt(0.04) - sqrt(0.09)，不要丢掉小数。\n" * 100
    service, video, stored = library(tmp_path, [Segment(0, 120, text)], duration=120)
    section = service.study_status(video.id)["sections"][0]
    cursor, received = None, ""
    for index in range(100):
        representation = "compact" if index % 2 == 0 else "verbose"
        page = service.learning.section(
            video.id, section["id"], max_tokens=1800, cursor=cursor, include_frames=False,
            representation=representation, record_delivery=False,
        )
        assert service.budget.count(compact_json(page)) <= 1800
        if index == 0:
            assert not service.store.deliveries(video.id)
        # MCP serializes before registering delivery. No hidden Python objects
        # may be needed to expand compact paragraphs after this round trip.
        page = service.learning.record_delivery(json.loads(compact_json(page)), frames=[])
        for item in expand_evidence(page):
            received += item["text"][:-1] if item.get("excerpt_truncated") else item["text"]
        cursor = page["next_cursor"]
        if cursor is None:
            break
        assert page["progress"]["text_status"] == "pending"
    else:
        pytest.fail("Compact character pagination did not finish")
    assert received == text
    assert page["progress"]["text_status"] == "delivered"
    assert service.store.deliveries(video.id)[0]["evidence_id"] == stored[0].id


def test_same_budget_delivers_more_source_without_losing_note_citations(tmp_path):
    cues = [Segment(i * 2, i * 2 + 2, f"第{i}步观察网格中心和目标边界", kind="transcript",
                    source="mlx-whisper:large-v3-turbo", language="zh") for i in range(80)]
    service, video, _ = library(tmp_path, cues)
    verbose = service.reader.read_interval(video.id, 0, 180, max_tokens=3000)
    compact = service.reader.read_interval(video.id, 0, 180, max_tokens=3000,
                                          representation="compact")
    assert len(expand_evidence(compact)) >= len(verbose["evidence"]) * 2
    plan = service.study_status(video.id)
    evidence_id = compact["evidence"][0]["refs"][0][0]
    saved = service.save_note(video.id, plan["sections"][0]["id"], plan["revision"], {
        "title": "网格", "summary": "观察目标中心与边界。",
        "claims": [{"text": "讲解观察网格中心。", "evidence_ids": [evidence_id]}],
    })
    assert saved["note"]["claims"][0]["evidence_ids"] == [evidence_id]


def test_invalid_compact_reference_cannot_record_unseen_text(tmp_path):
    service, video, _ = library(tmp_path, [Segment(0, 20, "原始语音内容")])
    section = service.study_status(video.id)["sections"][0]
    page = service.learning.section(video.id, section["id"], include_frames=False,
                                    representation="compact", record_delivery=False)
    page["evidence"][0]["refs"][0][2] += 1
    with pytest.raises(VidLensError, match="exceeds"):
        service.learning.record_delivery(page, frames=[])
    assert not service.store.deliveries(video.id)
    with pytest.raises(VidLensError, match="compact or verbose"):
        service.reader.read_interval(video.id, 0, 20, representation="lossy")


def test_compact_scan_keeps_navigation_but_diagnostics_are_opt_in(tmp_path):
    service, video, _ = library(tmp_path, [Segment(0, 20, "Lecture")])
    learning = service.learning
    _, revision, _ = learning._scan_data(video.id)
    data = {
        "status": "complete", "scan_id": "scan", "rules": {"threshold": .001},
        "scan_stats": {"metadata_bytes": 10000}, "candidate_count": 2,
        "candidates": [
            {"timestamp": 4, "score": .4, "relative_change": 12,
             "source_candidate_cursor": 3, "reasons": ["scene_change"]},
            {"timestamp": 12, "score": .01, "reasons": ["end"]},
        ],
    }
    with service.store.connect() as db:
        db.execute("INSERT INTO visual_scans VALUES (?,?,?)", (video.id, revision, compact_json(data)))
    first = learning.scan(video.id, view="full", limit=1)
    assert first["candidates"] == [{"timestamp": 4, "source_candidate_cursor": 3,
                                     "reasons": ["scene_change"]}]
    assert "rules" not in first and "scan_stats" not in first
    last = learning.scan(video.id, cursor=first["next_cursor"], view=first["view"], limit=1)
    assert last["candidates"][0]["timestamp"] == 12 and last["next_cursor"] is None
    diagnostic = learning.scan(video.id, view="full", diagnostics=True)
    assert diagnostic["candidates"] == data["candidates"]
    assert diagnostic["rules"] == data["rules"]

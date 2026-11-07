"""Regression checks for incomplete scans and bounded saved-knowledge retrieval."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from vidlens import visual
from vidlens.budget import compact_json
from vidlens.evidence_format import expand_evidence
from vidlens.mcp_server import create_server
from vidlens.models import Frame, MediaAsset, Segment, Video, VidLensError
from vidlens.service import VideoService
from vidlens.storyboard import StoryboardScanner
from vidlens.visual import VisualReader


def seed(tmp_path, *, duration=120, media=None):
    service = VideoService(tmp_path / "library")
    metadata = {"revision": "v1"}
    if media is not None:
        metadata["local_media_path"] = str(media)
    video = Video(
        "lesson",
        "https://www.youtube.com/watch?v=aircAruvnKk",
        "youtube",
        "Attention lesson",
        duration,
        metadata=metadata,
    )
    cue = Segment(0, min(duration, 120), "Original attention explanation.", source="manual")
    service.store.save(video, [cue], [])
    plan = service.learning.study.plan(video.id)
    return service, video, cue, plan


@pytest.mark.parametrize("status", ["complete", "partial"])
def test_candidate_pagination_does_not_repeat_an_incomplete_decode(tmp_path, monkeypatch, status):
    path = tmp_path / "media.mp4"
    path.write_bytes(b"metadata-only fixture; decoding is instrumented")
    service, video, _, _ = seed(tmp_path, duration=1000, media=path)
    decodes = []

    def decode(self, asset, start, end, result):
        decodes.append((start, end))
        result["status"] = status
        result["coverage"].update(
            complete=status == "complete",
            decoded_frames=50000,
            first_timestamp=0,
            last_timestamp=750,
        )
        result["candidates"] = [
            {"timestamp": index / 2, "score": 0.1, "reasons": ["scene_change"]}
            for index in range(1501)
        ]
        return result

    monkeypatch.setattr(StoryboardScanner, "_decode", decode)
    page = service.scan_video(video.id, limit=100)
    assert len(decodes) == 1
    assert page["status"] == status
    assert page["candidate_count"] == 1501
    expected = 1501 if status == "complete" else 1000
    assert page["retained_candidate_count"] == expected
    assert page["truncated"] is (status == "partial")
    collected = list(page["candidates"])
    while page["next_cursor"] is not None:
        page = service.scan_video(video.id, cursor=page["next_cursor"], limit=100)
        assert page["cached"] is True
        collected.extend(page["candidates"])
    assert len(collected) == expected
    assert len(decodes) == 1
    if status == "partial":
        assert "incomplete scan" in " ".join(page["warnings"])
        service.scan_video(video.id, cursor=0)
        assert len(decodes) == 2  # A new first-page request can retry the failed scan.


@pytest.mark.asyncio
async def test_missing_local_media_keeps_imported_speech_readable_in_mcp(tmp_path):
    service, video, cue, plan = seed(tmp_path, media=tmp_path / "removed.mp4")
    server = create_server(service=service)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool(
            "read_section",
            {
                "video_id": video.id,
                "section_id": plan["sections"][0]["id"],
            },
        )
    assert not result.isError, result
    payload = json.loads(result.content[0].text)
    assert payload["representation"] == "compact"
    assert expand_evidence(payload)[0]["id"] == cue.id
    assert expand_evidence(payload)[0]["text"] == cue.text
    assert payload["frames"] == []
    assert payload["visual"]["status"] == "unavailable"
    assert payload["visual"]["error"]["code"] == "media_missing"
    assert payload["progress"]["text_status"] == "delivered"
    assert payload["progress"]["visual_status"] == "not_delivered"


@pytest.mark.parametrize("budget", [768, 3000])
def test_large_note_returns_best_claim_with_original_identity_and_evidence(tmp_path, budget):
    service, video, cue, plan = seed(tmp_path)
    section_id = plan["sections"][0]["id"]
    claims = [
        {"text": "Background mechanism and example. " * 100, "evidence_ids": [cue.id]}
        for _ in range(7)
    ]
    claims.append(
        {"text": "Temperature changes attention concentration. " * 100, "evidence_ids": [cue.id]}
    )
    service.save_note(
        video.id,
        section_id,
        plan["revision"],
        {
            "title": "Attention",
            "summary": "A long chapter's teaching notes.",
            "concepts": ["attention", "temperature"],
            "claims": claims,
            "relations": [],
            "roles": ["mechanism"],
        },
    )
    original_note = service.learning.study.get_section(video.id, section_id)["note"]
    result = service.search_knowledge(video.id, "temperature concentration", max_tokens=budget)
    assert service.budget.count(compact_json(result)) <= budget
    assert len(result["matches"]) == 1
    hit = result["matches"][0]
    assert hit["section_id"] == section_id
    assert hit["match_scope"] == "claim" and hit["claim_index"] == 7
    assert hit["text"].startswith("Temperature changes attention concentration.")
    assert hit["evidence_ids"] == [cue.id]
    assert hit["excerpt_truncated"] is True
    assert "claims" not in hit  # The stored full note is unchanged and separately reusable.
    saved = service.learning.study.get_section(video.id, section_id)["note"]
    assert saved == original_note
    original = service.read_section(video.id, hit["section_id"], include_frames=False)
    assert set(hit["evidence_ids"]) <= {item["id"] for item in original["evidence"]}


def test_query_specific_relation_and_external_reference_keep_provenance(tmp_path):
    service, video, cue, plan = seed(tmp_path)
    section_id = plan["sections"][0]["id"]
    service.save_note(
        video.id,
        section_id,
        plan["revision"],
        {
            "title": "A mechanism",
            "summary": "An overview of attention.",
            "concepts": ["cache"],
            "claims": [{"text": "An attention definition.", "evidence_ids": [cue.id]}],
            "relations": [
                {
                    "source": "cache",
                    "target": "latency",
                    "relation": "reduces warm latency",
                    "evidence_ids": [cue.id],
                }
            ],
            "roles": ["mechanism"],
        },
    )
    hit = service.search_knowledge(video.id, "cache warm latency")["matches"][0]
    assert hit["match_scope"] == "relation" and hit["relation_index"] == 0
    assert hit["evidence_ids"] == [cue.id]
    assert "reduces warm latency" in hit["text"]
    added = service.add_reference(
        video.id,
        section_id,
        plan["revision"],
        {
            "title": "Official source",
            "url": "https://example.org/paper",
            "excerpt": "A verification experiment.",
            "explanation": "Explains the experiment.",
            "concepts": ["verification"],
        },
    )
    external = service.search_knowledge(video.id, "verification")["matches"][0]
    assert external["id"] == added["id"]
    assert external["url"] == added["url"]
    assert external["provenance"] == "external_source_read_by_client"


@pytest.mark.parametrize(
    "actual,missing,retry",
    [
        ([0.02], [1, 2, 3], 1),
        ([0.02, 3.02], [1, 2], 1),
        ([3.02], [0, 1, 2], 0),
        ([], [0, 1, 2, 3], 0),
    ],
)
def test_partial_frame_result_cannot_skip_unreturned_candidates(
    tmp_path, monkeypatch, actual, missing, retry
):
    service, video, _, _ = seed(tmp_path, duration=10)
    data = {"status": "complete", "candidates": [{"timestamp": time} for time in range(5)]}
    _, revision, _ = service.learning._scan_data(video.id)
    with service.store.connect() as db:
        db.execute("INSERT INTO visual_scans VALUES (?,?,?)", (video.id, revision, compact_json(data)))
    monkeypatch.setattr(
        service,
        "read_frame_times",
        lambda *args, **kwargs: [{"timestamp": stamp, "ocr": ""} for stamp in actual],
    )
    frames, details = service.learning._frames(video.id, {"start": 0, "end": 10}, 0, 4)
    assert len(frames) == len(actual)
    assert details["status"] == "partial"
    assert details["missing_candidate_times"] == missing
    assert details["next_frame_cursor"] == details["retry_frame_cursor"] == retry


def test_read_at_time_limit_raises_instead_of_returning_a_silent_prefix(tmp_path, monkeypatch):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"instrumented decoder")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"instrumented pixels")
    frame.with_suffix(".json").write_text(json.dumps({"source": str(frame)}))
    reader = VisualReader(tmp_path / "visual")
    clock = [0]
    monkeypatch.setattr(visual.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(reader, "_ffmpeg", lambda: "unused")
    monkeypatch.setattr(reader, "_display_interval", lambda *args: (0, 0.5))

    def extract(*args, **kwargs):
        clock[0] = reader.MAX_READ_SECONDS + 1
        return Frame(0, frame, 10, 10)

    monkeypatch.setattr(reader, "_extract", extract)
    with pytest.raises(VidLensError) as error:
        reader.read_at(MediaAsset(media, duration=2), [0, 1], ocr=False)
    assert error.value.code == "frame_timeout"

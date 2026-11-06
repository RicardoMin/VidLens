import json
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import pytest

from vidlens.models import Frame, Segment, Video, VidLensError
from vidlens.store import Store
from vidlens.study import StudyEngine, _digest


def seed(tmp_path, *, duration=120, segments=None, chapters=None, name="lesson"):
    store = Store(tmp_path / "library")
    video = Video(
        name,
        f"https://example.org/{name}",
        "test",
        "Lesson",
        duration,
        chapters=chapters or [],
        metadata={"revision": "original"},
    )
    cues = (
        segments
        if segments is not None
        else [
            Segment(0, 10, "The model uses a cache.", source="manual"),
            Segment(60, 70, "The cache does not help cold requests.", source="manual"),
        ]
    )
    store.save(video, cues, [])
    return store, StudyEngine(store), video, cues


def note(cues, **changes):
    return {
        "title": "Cache and its limits",
        "summary": "Caching reuses work, subject to a warm state.",
        "concepts": ["cache", "cold start"],
        "claims": [{"text": "The cache is conditional.", "evidence_ids": [cues[0].id]}],
        "relations": [],
        "roles": ["mechanism", "correction"],
        **changes,
    }


def check_partition(plan, cues):
    sections = plan["sections"]
    assert sections[0]["start"] == 0
    assert sections[-1]["end"] == plan["duration"]
    assert all(a["end"] == b["start"] for a, b in pairwise(sections))
    assert all(s["start"] < s["end"] for s in sections)
    assert all(any(cue.start < s["end"] and cue.end > s["start"] for s in sections) for cue in cues)


def test_long_platform_chapters_keep_titles_and_return_whole_original_cues(tmp_path):
    cues = [Segment(i * 10, i * 10 + 9, "A mathematical explanation. " * 18) for i in range(120)]
    store, study, _, cues = seed(
        tmp_path,
        duration=1200,
        segments=cues,
        chapters=[
            {"start_time": 0, "title": "Foundations"},
            {"start_time": 305, "title": "Experiments"},
        ],
    )
    plan = study.plan("lesson")
    check_partition(plan, cues)
    assert len(plan["sections"]) > 5
    assert {s["platform_title"] for s in plan["sections"]} == {"Foundations", "Experiments"}
    assert all(s["text_characters"] <= 9000 for s in plan["sections"])
    assert 305 in {s["start"] for s in plan["sections"]}
    originals = {s.id: s.text for s in cues}
    read = [
        e
        for section in plan["sections"]
        for e in study.get_section("lesson", section["id"])["evidence"]
    ]
    assert {e["id"] for e in read} == set(originals)
    assert all(e["text"] == originals[e["id"]] for e in read)
    assert "not a summary" in plan["method"]
    assert StudyEngine(store).plan("lesson")["revision"] == plan["revision"]


def test_single_long_cue_is_kept_whole_even_when_it_exceeds_capacity(tmp_path):
    cues = [
        Segment(0, 60, "The premise " * 900),
        Segment(60, 120, "requires an important qualification."),
        Segment(120, 180, "Now the next topic begins."),
    ]
    _, study, _, _ = seed(
        tmp_path,
        duration=180,
        segments=cues,
        chapters=[{"start_time": 60, "title": "Qualification"}],
    )
    plan = study.plan("lesson")
    check_partition(plan, cues)
    assert any(s["exceeds_preferred_capacity"] for s in plan["sections"])


def test_unpunctuated_auto_captions_get_bounded_continuation_chapters(tmp_path):
    cues = [
        Segment(i * 5, i * 5 + 4.9, "continuing an explanation without punctuation " * 3)
        for i in range(1400)
    ]
    _, study, _, _ = seed(tmp_path, duration=7000, segments=cues)
    plan = study.plan("lesson")
    check_partition(plan, cues)
    assert len(plan["sections"]) > 20
    assert all(s["text_characters"] <= 9000 for s in plan["sections"])
    assert any(s["boundary_reason"] == "capacity_continuation" for s in plan["sections"])


def test_capacity_boundary_prefers_nearby_sentence_end(tmp_path):
    cues = [Segment(i * 10, i * 10 + 9, "explanation without punctuation " * 20) for i in range(16)]
    cues[11].text += "."
    _, study, _, _ = seed(tmp_path, duration=160, segments=cues)
    assert study.plan("lesson")["sections"][0]["end"] == 119


def test_topic_changes_can_split_before_capacity_limit(tmp_path):
    cues = [
        Segment(
            i * 10,
            i * 10 + 9,
            (
                "Cache memory latency decoder keys values. "
                if i < 12
                else "Biology genes cells plants photosynthesis. "
            )
            * 5,
        )
        for i in range(24)
    ]
    _, study, _, _ = seed(tmp_path, duration=240, segments=cues)
    plan = study.plan("lesson")
    assert plan["sections"][0]["end"] == 119
    assert plan["sections"][1]["boundary_reason"] == "topic_change"


def test_platform_marker_in_tiny_caption_gap_keeps_original_position(tmp_path):
    cues = [Segment(0, 59.8, "This premise"), Segment(60.2, 120, "needs a qualification.")]
    _, study, _, _ = seed(
        tmp_path, segments=cues, chapters=[{"start_time": 60, "title": "Qualifier"}]
    )
    assert 60 in {s["start"] for s in study.plan("lesson")["sections"]}


def test_rolling_overlapping_captions_keep_official_boundaries_and_bounded_chapters(tmp_path):
    cues = [
        Segment(i * 2, i * 2 + 3, "rolling captions overlap without punctuation " * 2)
        for i in range(1000)
    ]
    _, study, _, _ = seed(
        tmp_path,
        duration=2001,
        segments=cues,
        chapters=[{"start_time": 0, "title": "First"}, {"start_time": 1001.5, "title": "Second"}],
    )
    plan = study.plan("lesson")
    check_partition(plan, cues)
    assert 1001.5 in {s["start"] for s in plan["sections"]}
    assert max(s["end"] - s["start"] for s in plan["sections"]) <= 465
    originals = {s.id: s.text for s in cues}
    evidence = [
        e for s in plan["sections"] for e in study.get_section("lesson", s["id"])["evidence"]
    ]
    assert {e["id"] for e in evidence} == set(originals)
    assert all(e["text"] == originals[e["id"]] for e in evidence)
    assert len(evidence) > len(originals)  # Complete overlapping cues provide adjacent context.


def test_layout_algorithm_upgrade_preserves_matching_v2_source_notes(tmp_path):
    store, study, video, cues = seed(tmp_path)
    plan = study.plan("lesson")
    speech = store.segments("lesson")
    historical_source = _digest(
        [
            "study-v2",
            video.id,
            "original",
            None,
            120,
            [],
            [(s.id, s.start, s.end, s.text, s.kind, s.source, s.language) for s in speech],
        ]
    )
    with store.connect() as db:
        row = db.execute("SELECT sections FROM study_plans WHERE video_id='lesson'").fetchone()
        sections = json.loads(row["sections"])
        for section in sections:
            section["id"] = (
                "section:" + _digest([historical_source, section["start"], section["end"]])[:16]
            )
        db.execute(
            "UPDATE study_plans SET source_revision=?,sections=?,algorithm='study-v2' WHERE video_id='lesson'",
            (historical_source, json.dumps(sections)),
        )
        db.execute(
            "INSERT INTO study_notes VALUES (?,?,?,?,?,?)",
            (
                video.id,
                historical_source,
                sections[0]["id"],
                plan["revision"],
                json.dumps(note(cues)),
                1,
            ),
        )
    upgraded = study.plan("lesson")
    assert upgraded["algorithm"] == "study-v3"
    assert upgraded["source_revision"] == historical_source
    assert upgraded["progress"]["notes_saved"] == 1
    assert study.notes("lesson")["notes"][0]["summary"] == note(cues)["summary"]


def test_ai_can_group_unpunctuated_captions_after_reading_without_cutting_cues(tmp_path):
    cues = [
        Segment(0, 60, "First idea without punctuation"),
        Segment(60, 120, "A separate idea without punctuation"),
    ]
    _, study, _, _ = seed(tmp_path, segments=cues)
    plan = study.plan("lesson")
    revised = study.revise_sections(
        "lesson",
        plan["revision"],
        [
            {"start": 0, "end": 60, "title": "First idea"},
            {"start": 60, "end": 120, "title": "Separate idea"},
        ],
    )
    check_partition(revised, cues)
    assert len(revised["sections"]) == 2


def test_no_transcript_still_plans_entire_visual_timeline(tmp_path):
    _, study, _, _ = seed(tmp_path, duration=1800, segments=[])
    plan = study.plan("lesson")
    check_partition(plan, [])
    assert len(plan["sections"]) >= 3
    assert plan["gaps"]["transcript_uncovered_intervals"] == [[0, 1800]]
    assert all(s["progress"]["text_status"] == "no_transcript" for s in plan["sections"])


def test_notes_persist_and_ocr_additions_do_not_change_speech_revision(tmp_path):
    store, study, _, cues = seed(tmp_path)
    plan = study.plan("lesson")
    section_id = plan["sections"][0]["id"]
    study.save_note("lesson", section_id, plan["revision"], note(cues))
    store.save_frames("lesson", [Frame(5, tmp_path / "frame.png", 100, 100, "OCR text")])
    other = StudyEngine(Store(store.home))
    current = other.plan("lesson")
    assert current["revision"] == plan["revision"]
    assert current["sections"][0]["id"] == section_id
    assert current["sections"][0]["title"] == "Cache and its limits"
    assert current["sections"][0]["candidate_title"] == plan["sections"][0]["title"]
    assert other.get_section("lesson", section_id)["note"]["summary"] == note(cues)["summary"]
    assert current["progress"]["text_reviewed_sections"] == 0


def test_replaced_or_missing_local_media_invalidates_notes_without_blocking_text(tmp_path):
    store, study, video, cues = seed(tmp_path)
    media = tmp_path / "local.mp4"
    media.write_bytes(b"original local video")
    video.metadata["local_media_path"] = str(media)
    store.save(video, cues, [])
    original = study.plan("lesson")
    study.save_note("lesson", original["sections"][0]["id"], original["revision"], note(cues))
    media.write_bytes(b"replacement local video with different size")
    replaced = study.plan("lesson")
    assert replaced["source_revision"] != original["source_revision"]
    assert replaced["progress"]["notes_saved"] == 0
    media.unlink()
    missing = study.plan("lesson")
    assert missing["source_revision"] != replaced["source_revision"]
    assert study.get_section("lesson", missing["sections"][0]["id"])["evidence"]
    assert study.plan("lesson")["revision"] == missing["revision"]


def test_local_replacement_removes_stale_frames_and_preserves_spoken_delivery(tmp_path):
    store, study, video, cues = seed(tmp_path)
    media = tmp_path / "local.mp4"
    media.write_bytes(b"first media")
    video.metadata["local_media_path"] = str(media)
    store.save(video, cues, [])
    study.plan("lesson")
    store.save_frames("lesson", [Frame(5, tmp_path / "frame.png", 100, 100, "Old screen")])
    store.delivered("lesson", [{"timestamp": 5}], "frame")
    store.delivered("lesson", [{"id": s.id, "start": s.start, "end": s.end} for s in cues])
    assert study.plan("lesson")["sections"][0]["progress"]["visual_status"] == "sampled"
    media.write_bytes(b"replacement media contents")
    updated = study.plan("lesson")
    section_id = updated["sections"][0]["id"]
    result = study.get_section("lesson", section_id)
    assert result["frames"] == []
    assert result["progress"]["visual_status"] == "not_delivered"
    assert result["progress"]["text_status"] == "delivered"
    assert all(s.kind != "ocr" for s in store.segments("lesson"))
    with pytest.raises(VidLensError) as error:
        study.save_note(
            "lesson",
            section_id,
            updated["revision"],
            note(
                cues,
                claims=[
                    {"text": "Claim from an obsolete screen", "evidence_ids": ["frame:5"]},
                ],
            ),
        )
    assert error.value.code == "invalid_evidence_reference"


@pytest.mark.parametrize("change", ["source", "transcript"])
def test_source_refresh_and_corrected_transcript_reject_stale_notes(tmp_path, change):
    store, study, video, cues = seed(tmp_path)
    plan = study.plan("lesson")
    section_id = plan["sections"][0]["id"]
    study.save_note("lesson", section_id, plan["revision"], note(cues))
    if change == "source":
        video.metadata["revision"] = "edited-source"
    else:
        cues[0].text = "The cache is disabled."
    store.save(video, cues, [])
    with pytest.raises(VidLensError, match="changed") as error:
        study.save_note("lesson", section_id, plan["revision"], note(cues))
    assert error.value.code == "stale_study_revision"
    refreshed = study.plan("lesson")
    assert refreshed["revision"] != plan["revision"]
    assert refreshed["progress"]["notes_saved"] == 0


def test_delivery_and_notes_are_independent_and_caption_pauses_are_not_unread(tmp_path):
    store, study, _, cues = seed(tmp_path)
    plan = study.plan("lesson")
    section_id = plan["sections"][0]["id"]
    store.delivered(
        "lesson", [{"id": s.id, "start": s.start, "end": s.end} for s in cues], "excerpt"
    )
    study.save_note("lesson", section_id, plan["revision"], note(cues))
    assert study.plan("lesson")["sections"][0]["progress"]["text_segments_delivered"] == 0
    store.delivered("lesson", [{"id": s.id, "start": s.start, "end": s.end} for s in cues])
    current = study.plan("lesson")
    assert current["sections"][0]["progress"]["text_status"] == "delivered"
    assert current["progress"]["text_reviewed_sections"] == 1
    assert current["gaps"]["transcript_uncovered_intervals"]
    study.save_note("lesson", section_id, plan["revision"], note(cues, claims=[]))
    assert study.plan("lesson")["sections"][0]["progress"]["note_status"] == "draft"
    assert study.plan("lesson")["progress"]["text_reviewed_sections"] == 0


def test_obsolete_or_partial_delivery_does_not_mark_current_text_read(tmp_path):
    store, study, _, cues = seed(tmp_path)
    store.delivered("lesson", [{"id": "outdated", "start": 0, "end": 120}])
    store.delivered("lesson", [{"start": 0, "end": 120}])
    store.delivered("lesson", [{"id": cues[0].id, "start": 0, "end": 5}])
    progress = study.plan("lesson")["sections"][0]["progress"]
    assert progress["text_segments_delivered"] == 0
    store.delivered("lesson", [{"id": cues[0].id, "start": 5, "end": 10}])
    assert study.plan("lesson")["sections"][0]["progress"]["text_segments_delivered"] == 1


def test_delivery_completeness_preserves_submillisecond_timestamps(tmp_path):
    store, study, _, cues = seed(
        tmp_path, segments=[Segment(0.1234567, 10.1234567, "A precise cue.")]
    )
    cue = cues[0]
    store.delivered("lesson", [{"id": cue.id, "start": cue.start, "end": 5}])
    store.delivered("lesson", [{"id": cue.id, "start": 5.01, "end": cue.end}])
    assert study.plan("lesson")["sections"][0]["progress"]["text_segments_delivered"] == 0
    store.delivered("lesson", [{"id": cue.id, "start": cue.start, "end": cue.end}])
    assert study.plan("lesson")["sections"][0]["progress"]["text_segments_delivered"] == 1


@pytest.mark.parametrize(
    "reference", ["invented", "frame:7", "frame:nan", "foreign", "outside", "ocr"]
)
def test_references_must_exist_in_this_video_and_section(tmp_path, reference):
    store, study, _, cues = seed(
        tmp_path,
        chapters=[
            {"start_time": 0, "title": "Beginning"},
            {"start_time": 40, "title": "End"},
        ],
    )
    foreign = Segment(0, 10, "A statement from a different video.")
    store.save(Video("other", "", "test", "Other", 10), [foreign], [])
    store.save_frames("lesson", [Frame(5, tmp_path / "frame.png", 100, 100, "OCR assertion")])
    ocr = next(s for s in store.segments("lesson") if s.kind == "ocr")
    ref = {"foreign": foreign.id, "outside": cues[-1].id, "ocr": ocr.id}.get(reference, reference)
    plan = study.plan("lesson")
    with pytest.raises(VidLensError) as error:
        study.save_note(
            "lesson",
            plan["sections"][0]["id"],
            plan["revision"],
            note(
                cues,
                claims=[
                    {"text": "Unsupported claim", "evidence_ids": [ref]},
                ],
            ),
        )
    assert error.value.code == "invalid_evidence_reference"
    assert study.plan("lesson")["progress"]["notes_saved"] == 0


def test_relations_keep_multiple_original_sources_and_frame_precision(tmp_path):
    store, study, _, cues = seed(tmp_path)
    stamp = 5.123456789
    store.save_frames("lesson", [Frame(stamp, tmp_path / "frame.png", 100, 100)])
    plan = study.plan("lesson")
    section_id = plan["sections"][0]["id"]
    relation = {
        "source": "cache",
        "target": "cold start",
        "relation": "does not accelerate",
        "evidence_ids": [cues[0].id, cues[1].id, f"frame:{stamp}"],
        "roles": ["mechanism", "correction"],
    }
    saved = study.save_note(
        "lesson", section_id, plan["revision"], note(cues, relations=[relation])
    )
    assert saved["note"]["relations"] == [relation]
    assert study.get_section("lesson", section_id)["frames"][0]["id"] == f"frame:{stamp}"


def test_regrouping_does_not_attach_old_notes_and_retries_are_idempotent(tmp_path):
    store, study, _, cues = seed(
        tmp_path,
        chapters=[
            {"start_time": 0, "title": "Beginning"},
            {"start_time": 40, "title": "End"},
        ],
    )
    plan = study.plan("lesson")
    first = plan["sections"][0]
    study.save_note("lesson", first["id"], plan["revision"], note(cues))
    rename = [{"start": s["start"], "end": s["end"], "title": "Renamed"} for s in plan["sections"]]
    renamed = study.revise_sections("lesson", plan["revision"], rename)
    assert renamed["sections"][0]["id"] == first["id"]
    assert renamed["sections"][0]["title"] == "Renamed"
    assert study.notes("lesson")["notes"][0]["title"] == "Renamed"
    assert renamed["progress"]["notes_saved"] == 1
    assert (
        study.revise_sections("lesson", renamed["revision"], rename)["revision"]
        == renamed["revision"]
    )
    combined = study.revise_sections(
        "lesson", renamed["revision"], [{"start": 0, "end": 120, "title": "Combined"}]
    )
    assert combined["progress"]["notes_saved"] == 0
    assert study.notes("lesson")["notes"] == []
    assert study.get_section("lesson", combined["sections"][0]["id"])["note"] is None
    with pytest.raises(VidLensError) as error:
        study.revise_sections("lesson", plan["revision"], rename)
    assert error.value.code == "stale_study_revision"
    assert StudyEngine(store).plan("lesson")["revision"] == combined["revision"]


@pytest.mark.parametrize(
    "sections",
    [
        [{"start": 1, "end": 120, "title": "Missing start"}],
        [{"start": 0, "end": 30, "title": "A"}, {"start": 40, "end": 120, "title": "B"}],
        [{"start": 0, "end": 40, "title": "A"}, {"start": 30, "end": 120, "title": "B"}],
        [{"start": 0, "end": 120, "title": ""}],
        [{"start": 0, "end": float("nan"), "title": "A"}],
    ],
)
def test_invalid_revisions_leave_plan_unchanged(tmp_path, sections):
    _, study, _, _ = seed(tmp_path)
    plan = study.plan("lesson")
    with pytest.raises(VidLensError) as error:
        study.revise_sections("lesson", plan["revision"], sections)
    assert error.value.code == "invalid_sections"
    assert study.plan("lesson")["revision"] == plan["revision"]


def test_two_instances_save_independent_chapter_notes_without_lost_updates(tmp_path):
    store, study, _, cues = seed(
        tmp_path,
        chapters=[
            {"start_time": 0, "title": "Beginning"},
            {"start_time": 40, "title": "End"},
        ],
    )
    plan = study.plan("lesson")

    def save(index):
        other = StudyEngine(Store(store.home))
        return other.save_note(
            "lesson", plan["sections"][index]["id"], plan["revision"], note([cues[index]])
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, range(2)))
    assert len(results) == 2
    assert study.plan("lesson")["progress"]["notes_saved"] == 2
    snapshot = study.notes("lesson")
    assert len(snapshot["notes"]) == 2
    assert {item["section_id"] for item in snapshot["notes"]} == {s["id"] for s in plan["sections"]}
    assert all("evidence" not in item for item in snapshot["notes"])

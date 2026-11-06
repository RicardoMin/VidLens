"""Regressions for source replacement, durable reading and connected knowledge."""

import json

import pytest

from vidlens.budget import TokenBudget
from vidlens.models import Frame, Segment, Video, VidLensError
from vidlens.reading import EvidenceReader
from vidlens.references import ReferenceStore
from vidlens.store import Store
from vidlens.study import StudyEngine


def library(tmp_path, cues, *, duration=30, chapters=None):
    store = Store(tmp_path / "library")
    video = Video("lesson", "https://example.org/lesson", "test", "Lesson", duration,
                  chapters=chapters or [], metadata={"revision": "original"})
    store.save(video, cues, [])
    return store, video, StudyEngine(store)


def test_asr_replacement_forgets_old_full_and_partial_pages_but_keeps_other_evidence(tmp_path):
    old = [
        Segment(10, 14, "old text", "transcript", "whisper:base", "en"),
        Segment(15, 19, "partially read text", "transcript", "whisper:base", "en"),
        Segment(5, 12, "crossing cue", "transcript", "whisper:base", "en"),
        Segment(10, 19, "native caption", "subtitle", "manual", "en"),
        Segment(11, 18, "other language", "transcript", "whisper:base", "zh"),
        Segment(21, 25, "outside range", "transcript", "whisper:base", "en"),
    ]
    store, video, study = library(tmp_path, old)
    for cue in old:
        store.delivered_text(video.id, [{"id": cue.id, "text": cue.text if cue != old[1] else "partially"}])
    # Legacy aggregate responses had no raw evidence IDs.
    store.delivered(video.id, [{"start": 10, "end": 19}])
    store.save(video, [Segment(10, 14, "old text", "transcript", "whisper:small", "en")], [],
               replace=False, replace_asr_range=(10, 20, "en"))
    assert {d["evidence_id"] for d in store.deliveries(video.id)} == {s.id for s in old[2:]}
    with store.connect() as db:
        partial_ids = {r[0] for r in db.execute("SELECT evidence_id FROM text_deliveries")}
    assert partial_ids == {s.id for s in old[2:]}
    progress = study.plan(video.id)["sections"][0]["progress"]
    assert old[0].id in progress["pending_evidence_ids"]
    assert progress["text_segments_delivered"] == 4


def test_asr_storage_failure_restores_delivery_attribution(tmp_path, monkeypatch):
    old = Segment(0, 10, "Keep this evidence", "transcript", "whisper:base", "en")
    store, video, _ = library(tmp_path, [old])
    store.delivered_text(video.id, [{"id": old.id, "text": old.text}])
    before = store.deliveries(video.id)

    def fail(*_args):
        raise RuntimeError("write interrupted")

    monkeypatch.setattr(store, "_add_segments", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        store.save(video, [], [], replace=False, replace_asr_range=(0, 10, "en"))
    assert store.deliveries(video.id) == before
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM text_deliveries").fetchone()[0] == 1
    assert store.segments(video.id)[0].text == old.text


def test_identical_words_on_other_language_track_are_not_overwritten_or_duplicated(tmp_path):
    english = Segment(0, 10, "CUDA", "transcript", "whisper:base", "en")
    store, video, study = library(tmp_path, [english])
    store.delivered_text(video.id, [{"id": english.id, "text": english.text}])
    chinese = Segment(0, 10, "CUDA", "transcript", "whisper:small", "zh")
    store.save(video, [chinese], [], replace=False, replace_asr_range=(0, 10, "zh"))
    assert chinese.id != english.id
    assert {s.language for s in store.segments(video.id)} == {"en", "zh"}
    before = study.plan(video.id)["source_revision"]
    store.add_segments(video.id, [Segment(0, 10, "CUDA", "transcript", "whisper:small", "zh")])
    assert len(store.segments(video.id)) == 2
    assert study.plan(video.id)["source_revision"] == before
    # Replacing a complete track set in a different order retains its stable IDs.
    store.save(video, [chinese, english], [])
    assert study.plan(video.id)["source_revision"] == before
    assert store.deliveries(video.id)[0]["evidence_id"] == english.id
    # A refresh of only Chinese cannot delete the already-read English track.
    store.save(video, [Segment(0, 10, "CUDA", "transcript", "whisper:small", "zh")], [],
               replace=False, replace_asr_range=(0, 10, "zh"))
    assert len(store.segments(video.id)) == 2
    assert store.deliveries(video.id)[0]["evidence_id"] == english.id


@pytest.mark.parametrize("change", ["model", "source_revision"])
def test_interval_cursor_is_bound_to_source_provenance_even_when_text_ids_match(tmp_path, change):
    cue = Segment(0, 10, "A long explanation. " * 1000, "transcript", "whisper:base", "en")
    store, video, _ = library(tmp_path, [cue])
    reader = EvidenceReader(store, TokenBudget(tmp_path / "tokens"))
    first = reader.read_interval(video.id, 0, 10, 900, speech_only=True)
    assert first["next_cursor"]
    if change == "model":
        replacement = Segment(0, 10, cue.text, "transcript", "whisper:small", "en")
        store.save(video, [replacement], [], replace=False, replace_asr_range=(0, 10, "en"))
        assert replacement.id == cue.id  # The old cursor checked IDs alone.
    else:
        video.metadata["revision"] = "new edit"
        store.save(video, [cue], [], invalidate_derived=True)
    with pytest.raises(VidLensError) as error:
        reader.read_interval(video.id, 0, 10, 900, first["next_cursor"], speech_only=True)
    assert error.value.code == "invalid_cursor"


def test_related_chapter_search_keeps_the_causal_bridge_and_exact_evidence(tmp_path):
    cues = [Segment(t, t + 9, "Original explanation.") for t in (0, 10, 20)]
    store, video, study = library(tmp_path, cues, chapters=[
        {"start_time": t, "title": f"Part {i}"} for i, t in enumerate((0, 10, 20))
    ])
    references = ReferenceStore(store, study)
    plan = study.plan(video.id)
    nodes = ["quantization", "memory footprint", "batch size", "throughput"]
    for i, section in enumerate(plan["sections"]):
        study.save_note(video.id, section["id"], plan["revision"], {
            "title": f"Part {i}", "summary": "A mechanism.",
            "relations": [{"source": nodes[i], "target": nodes[i + 1],
                           "relation": "influences", "evidence_ids": [cues[i].id]}],
        })
    matches = references.search(video.id, "quantization throughput")["matches"]
    assert {m["section_id"] for m in matches} == {s["id"] for s in plan["sections"]}
    for match in matches:
        assert match["matched_relation_indices"] == [0]
        assert match["match_reason"] == "relation_path"
        assert match["relations"][0]["evidence_ids"] == [cues[int(match["title"][-1])].id]
    # Schema names and raw identifiers are provenance, not semantic content.
    assert references.search(video.id, "target")["matches"] == []
    assert references.search(video.id, cues[0].id)["matches"] == []
    store.save(video, [Segment(0, 30, "A replacement lecture.")], [], invalidate_derived=True)
    assert references.search(video.id, "quantization throughput")["matches"] == []


def test_reference_validation_and_write_use_the_same_current_source(tmp_path, monkeypatch):
    from vidlens import references as module

    store, video, study = library(tmp_path, [Segment(0, 30, "Original speech.")])
    references = ReferenceStore(store, study)
    plan = study.plan(video.id)
    parse = module.urlsplit

    def source_changes_during_validation(url):
        video.metadata["revision"] = "replacement"
        store.save(video, [Segment(0, 30, "Revised speech.")], [], invalidate_derived=True)
        return parse(url)

    monkeypatch.setattr(module, "urlsplit", source_changes_during_validation)
    with pytest.raises(VidLensError) as error:
        references.add(video.id, plan["sections"][0]["id"], plan["revision"], {
            "url": "https://example.org/paper", "title": "Paper", "excerpt": "Source text.",
            "explanation": "Explains the lesson.",
        })
    assert error.value.code == "stale_revision"
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM study_references").fetchone()[0] == 0


def test_frame_display_interval_survives_later_capture_and_supports_both_chapters(tmp_path):
    store, video, study = library(tmp_path, [], duration=3, chapters=[
        {"start_time": 0, "title": "First"}, {"start_time": 1, "title": "Second"},
        {"start_time": 2, "title": "Third"},
    ])
    plan = study.plan(video.id)
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"stored image")
    frame = Frame(0.5, path, 100, 100, requested_timestamp=1.1, visible_until=2.0)
    store.save_frames(video.id, [frame])
    store.delivered(video.id, [frame.to_dict()], "frame")
    # A later ordinary capture has no decoded hold interval. Preserve known PTS facts.
    store.save_frames(video.id, [Frame(0.5, path, 100, 100)])
    for section in plan["sections"][:2]:
        state = study.get_section(video.id, section["id"])
        assert state["frames"][0]["timestamp"] == 0.5
        assert state["frames"][0]["visible_until"] == 2.0
        assert state["progress"]["frame_timestamps_delivered"] == [0.5]
        study.save_note(video.id, section["id"], plan["revision"], {
            "title": section["title"], "summary": "The held frame is visible.",
            "claims": [{"text": "The diagram persists.", "evidence_ids": ["frame:0.5"]}],
        })
    third = plan["sections"][2]
    assert not study.get_section(video.id, third["id"])["frames"]
    with pytest.raises(VidLensError, match="out-of-section"):
        study.save_note(video.id, third["id"], plan["revision"], {
            "title": "Third", "summary": "Different frame.",
            "claims": [{"text": "Unsupported.", "evidence_ids": ["frame:0.5"]}],
        })
    with store.connect() as db:
        assert json.loads(db.execute("SELECT data FROM frames").fetchone()[0])["visible_until"] == 2.0

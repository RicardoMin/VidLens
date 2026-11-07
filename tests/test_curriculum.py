from concurrent.futures import ThreadPoolExecutor

import pytest

from vidlens.curriculum import Curriculum
from vidlens.models import Frame, Segment, Video, VidLensError
from vidlens.store import Store
from vidlens.study import StudyEngine


def seed(tmp_path):
    store = Store(tmp_path / "library")
    video = Video("lesson", "https://example.org/lesson", "test", "Lesson", 120)
    cues = [Segment(0, 20, "The grid assigns each object to one cell."),
            Segment(60, 90, "Two box predictors share one class distribution.")]
    store.save(video, cues, [])
    study = StudyEngine(store)
    curriculum = Curriculum(store, study)
    revision = study.plan(video.id)["revision"]
    return store, study, curriculum, video, cues, revision


def unit(cues, key="grid", **changes):
    return {"id": key, "title": "Grid assignment", "roles": ["mechanism"],
            "concepts": ["grid", "object center"],
            "explanation": "The cell containing the object center is responsible for it.",
            "evidence_ids": [cue.id for cue in cues], "requires": [], **changes}


def plan(*keys):
    return {"title": "Detection explained", "goal": "Decode a predicted box and its score.",
            "sections": [{"id": key, "title": key, "objective": f"Understand {key}",
                          "unit_ids": [key]} for key in keys]}


def draft(*keys):
    return {"markdown": "\n".join(f"<!-- section:{key} -->\n## {key}\n"
                                  f"A worked explanation of {key}." for key in keys),
            "coverage": [{"unit_id": key, "section_id": key,
                          "excerpt": f"A worked explanation of {key}."} for key in keys]}


def test_cross_chapter_units_survive_layout_changes_and_retrieve_dependencies(tmp_path):
    _, study, curriculum, _, cues, revision = seed(tmp_path)
    curriculum.save("lesson", revision, [unit(cues), unit(
        cues[1:], "score", title="Confidence scoring", roles=["derivation"],
        concepts=["confidence"], explanation="Multiply conditional class by confidence.",
        requires=["grid"])])
    regrouped = study.revise_sections("lesson", revision, [
        {"start": 0, "end": 50, "title": "Beginning"},
        {"start": 50, "end": 120, "title": "End"}])
    result = curriculum.query("lesson", "confidence", roles=["derivation"])
    assert result["revision"] == regrouped["revision"]
    assert result["matched_unit_ids"] == ["score"]
    assert [item["id"] for item in result["units"]] == ["grid", "score"]
    assert result["units"][0]["evidence_ranges"] == [[0, 20], [60, 90]]
    assert result["dependencies_complete"]
    with pytest.raises(VidLensError, match="changed"):
        curriculum.save("lesson", revision, [unit(cues)])


def test_dependency_updates_are_atomic_and_cycles_rejected(tmp_path):
    _, _, curriculum, _, cues, revision = seed(tmp_path)
    curriculum.save("lesson", revision, [unit(cues)])
    with pytest.raises(VidLensError, match="Unknown required"):
        curriculum.save("lesson", revision, [unit(cues, explanation="Changed", requires=["missing"])])
    with pytest.raises(VidLensError, match="Cyclic"):
        curriculum.save("lesson", revision, [unit(cues, requires=["score"]),
                                             unit(cues, "score", requires=["grid"])])
    result = curriculum.query("lesson")
    assert result["unit_count"] == 1
    assert result["units"][0]["explanation"].startswith("The cell")


def test_current_video_evidence_and_source_revision_are_enforced(tmp_path):
    store, study, curriculum, video, cues, revision = seed(tmp_path)
    foreign = Video("other", "https://example.org/other", "test", "Other", 120)
    foreign_cue = Segment(0, 20, "Another video.")
    store.save(foreign, [foreign_cue], [])
    with pytest.raises(VidLensError) as error:
        curriculum.save("lesson", revision, [unit([foreign_cue])])
    assert error.value.code == "invalid_evidence_reference"
    curriculum.save("lesson", revision, [unit(cues)])
    store.save(video, [Segment(0, 100, "Replacement recognition.")], [])
    current = study.plan("lesson")["revision"]
    assert current != revision
    assert curriculum.query("lesson")["units"] == []
    with pytest.raises(VidLensError):
        curriculum.save("lesson", current, [unit(cues)])
    with pytest.raises(VidLensError):
        curriculum.save("lesson", None, [unit(cues)])


def test_frame_references_require_current_media_identity(tmp_path):
    store, _, curriculum, video, cues, revision = seed(tmp_path)
    store.save_frames("lesson", [Frame(10, tmp_path / "frame.png", 10, 10)])
    result = curriculum.save("lesson", revision, [unit(cues, evidence_ids=["frame:10"])])
    assert result["unit_count"] == 1
    with store.connect() as db:
        db.execute("DELETE FROM frames WHERE video_id='lesson'")
    with pytest.raises(VidLensError):
        curriculum.query("lesson")
    store.save_frames("lesson", [Frame(10, tmp_path / "frame.png", 10, 10)])
    video.metadata["revision"] = "new-source"
    store.save(video, cues, [])
    revision = curriculum.study.plan("lesson")["revision"]
    with pytest.raises(VidLensError):
        curriculum.save("lesson", revision, [unit(cues, evidence_ids=["frame:10"])])


def test_query_budget_never_returns_dependency_fragments(tmp_path):
    _, _, curriculum, _, cues, revision = seed(tmp_path)
    curriculum.save("lesson", revision, [unit(cues, explanation="Long prerequisite. " * 300),
        unit(cues, "score", title="Score", concepts=["confidence"], requires=["grid"],
             explanation="Conditional probability times confidence.")])
    result = curriculum.query("lesson", "confidence", limit=1, max_tokens=768)
    assert result["truncated"]
    assert result["units"] == []
    assert result["matched_unit_ids"] == []
    assert result["accounting"]["text_tokens_estimate"] <= 768
    assert curriculum.query("lesson", roles=["experiment"])["units"] == []


def test_concurrent_unit_saves_merge_without_lost_updates(tmp_path):
    _, _, curriculum, _, cues, revision = seed(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda key: curriculum.save("lesson", revision, [unit(cues, key)]),
                      ["one", "two", "three", "four"]))
    assert curriculum.query("lesson")["unit_count"] == 4


def test_plan_becomes_stale_when_knowledge_changes_but_not_on_identical_save(tmp_path):
    _, _, curriculum, _, cues, revision = seed(tmp_path)
    curriculum.save("lesson", revision, [unit(cues)])
    curriculum.plan("lesson", revision, plan("grid"))
    curriculum.save("lesson", revision, [unit(cues)])
    assert not curriculum.plan("lesson", revision)["stale"]
    curriculum.save("lesson", revision, [unit(cues, explanation="A refined account.")])
    assert curriculum.plan("lesson", revision)["stale"]
    audit = curriculum.audit("lesson", revision, draft("grid"))
    assert not audit["ready_for_review"]
    assert "stale_article_plan" in audit["issues"]


def test_article_audit_requires_core_units_and_actual_section_local_excerpts(tmp_path):
    _, _, curriculum, _, cues, revision = seed(tmp_path)
    empty = curriculum.audit("lesson", revision, {"markdown": "", "coverage": []})
    assert not empty["ready_for_review"]
    assert "no_core_knowledge_units" in empty["issues"]
    curriculum.save("lesson", revision, [unit(cues), unit(cues, "score", requires=["grid"])])
    curriculum.plan("lesson", revision, plan("grid", "score"))
    valid = draft("grid", "score")
    assert curriculum.audit("lesson", revision, valid)["ready_for_review"]
    # A claim cannot be attributed to a section merely because its text appears elsewhere.
    valid["coverage"][0]["excerpt"] = "A worked explanation of score."
    audit = curriculum.audit("lesson", revision, valid)
    assert not audit["ready_for_review"]
    assert audit["uncovered_core"] == ["grid"]
    assert audit["uncovered_dependencies"] == ["grid"]
    assert audit["semantic_validation"] == "not_performed"


def test_omitting_core_does_not_pass_and_duplicate_anchors_do_not_hide_missing_text(tmp_path):
    _, _, curriculum, _, cues, revision = seed(tmp_path)
    curriculum.save("lesson", revision, [unit(cues)])
    curriculum.plan("lesson", revision, plan("grid"))
    audit = curriculum.audit("lesson", revision, {
        "markdown": "A short introduction.", "omissions": [{"unit_id": "grid", "reason": "Brief"}]})
    assert not audit["ready_for_review"]
    assert audit["unaccounted_units"] == []
    repeated = draft("grid")
    repeated["markdown"] += "\n<!-- section:grid -->\n" + repeated["markdown"]
    audit = curriculum.audit("lesson", revision, repeated)
    assert not audit["ready_for_review"]
    assert "duplicate_section_anchor:grid" in audit["issues"]


def test_plan_validates_visual_evidence_and_unit_references(tmp_path):
    _, _, curriculum, _, cues, revision = seed(tmp_path)
    curriculum.save("lesson", revision, [unit(cues)])
    document = plan("grid")
    document["sections"][0]["visuals"] = [{"purpose": "show assignment", "evidence_ids": ["missing"]}]
    with pytest.raises(VidLensError):
        curriculum.plan("lesson", revision, document)
    assert curriculum.plan("lesson", revision)["plan"] is None
    with pytest.raises(VidLensError):
        curriculum.plan("lesson", revision, plan("missing"))


def test_fenced_anchors_and_hidden_comments_cannot_supply_coverage(tmp_path):
    _, _, curriculum, _, cues, revision = seed(tmp_path)
    curriculum.save("lesson", revision, [unit(cues)])
    curriculum.plan("lesson", revision, plan("grid"))
    document = draft("grid")
    document["markdown"] = "```markdown\n" + document["markdown"] + "\n```"
    assert not curriculum.audit("lesson", revision, document)["ready_for_review"]
    document["markdown"] = ("<!-- section:grid -->\n## Grid\n"
                            "<!-- A worked explanation of grid. -->")
    assert not curriculum.audit("lesson", revision, document)["ready_for_review"]


def test_missing_transitive_supporting_dependencies_are_reported(tmp_path):
    _, _, curriculum, _, cues, revision = seed(tmp_path)
    curriculum.save("lesson", revision, [
        unit(cues, importance="supporting"),
        unit(cues, "score", importance="supporting", requires=["grid"]),
        unit(cues, "suppression", requires=["score"]),
    ])
    curriculum.plan("lesson", revision, plan("suppression"))
    audit = curriculum.audit("lesson", revision, draft("suppression"))
    assert audit["uncovered_core"] == []
    assert audit["uncovered_dependencies"] == ["grid", "score"]
    assert not audit["ready_for_review"]


def test_missing_local_video_does_not_block_speech_knowledge(tmp_path):
    store, study, curriculum, video, cues, _ = seed(tmp_path)
    video.metadata["local_media_path"] = str(tmp_path / "missing.mp4")
    store.save(video, cues, [])
    revision = study.plan("lesson")["revision"]
    curriculum.save("lesson", revision, [unit(cues)])
    assert curriculum.query("lesson")["matched_unit_ids"] == ["grid"]


def test_planned_visual_and_exercise_need_actual_content_in_their_section(tmp_path):
    _, _, curriculum, _, cues, revision = seed(tmp_path)
    curriculum.save("lesson", revision, [unit(cues), unit(cues, "score")])
    document = plan("grid", "score")
    document["sections"][0]["visuals"] = [{"purpose": "Locate the center",
                                          "evidence_ids": [cues[0].id]}]
    document["sections"][0]["exercises"] = [{"question": "Which cell is responsible?",
                                            "unit_ids": ["grid"]}]
    curriculum.plan("lesson", revision, document)
    article = draft("grid", "score")
    result = curriculum.audit("lesson", revision, article)
    assert not result["ready_for_review"]
    assert result["uncovered_visuals"] == [{"section_id": "grid", "visual_index": 0}]
    assert result["uncovered_exercises"] == [{"section_id": "grid", "exercise_index": 0}]
    article["assets"] = [{"name": "grid.svg", "kind": "diagram", "svg": "<svg/>"}]
    article["visual_coverage"] = [{"section_id": "grid", "visual_index": 0,
                                   "asset_name": "grid.svg"}]
    article["exercise_coverage"] = [{"section_id": "grid", "exercise_index": 0,
                                     "excerpt": "Which cell is responsible?"}]
    # Content elsewhere in the article cannot fulfill this section's planned tasks.
    article["markdown"] += "\n![Grid](asset:grid.svg)\nWhich cell is responsible?\n"
    assert not curriculum.audit("lesson", revision, article)["ready_for_review"]
    article["markdown"] = article["markdown"].replace(
        "<!-- section:score -->", "![Grid][grid]\nWhich cell is responsible?\n\n<!-- section:score -->")
    # The reference definition may be at the end of the document, outside the grid section.
    article["markdown"] += "\n[grid]: asset:grid.svg\n"
    result = curriculum.audit("lesson", revision, article)
    assert result["ready_for_review"]
    assert result["semantic_validation"] == "not_performed"


def test_visual_markup_inside_a_code_sample_does_not_fulfill_plan(tmp_path):
    _, _, curriculum, _, cues, revision = seed(tmp_path)
    curriculum.save("lesson", revision, [unit(cues)])
    document = plan("grid")
    document["sections"][0]["visuals"] = [{"purpose": "Locate the center",
                                          "evidence_ids": [cues[0].id]}]
    curriculum.plan("lesson", revision, document)
    article = draft("grid")
    article["assets"] = [{"name": "grid.svg", "kind": "diagram", "svg": "<svg/>"}]
    article["visual_coverage"] = [{"section_id": "grid", "visual_index": 0,
                                   "asset_name": "grid.svg"}]
    article["markdown"] += "\n```markdown\n![Grid](asset:grid.svg)\n```\n"
    assert not curriculum.audit("lesson", revision, article)["ready_for_review"]

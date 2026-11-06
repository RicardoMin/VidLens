"""The image budget should cover question-specific states, not repeated titles."""

import copy

from vidlens.models import Segment
from vidlens.visual_relevance import QueryVisualSelector


def test_held_frame_uses_the_requested_speech_context_and_keeps_the_relevant_occurrence():
    frames = [{"timestamp": 0, "requested_timestamp": t, "ocr": "equation"} for t in (10, 100)]
    result = QueryVisualSelector.select(
        "normalization reset", frames,
        [Segment(99, 102, "normalization reset", source="manual")], start=0, end=120,
    )
    assert len(result["selected"]) == 1
    assert result["selected"][0]["timestamp"] == 0
    assert result["selected"][0]["requested_timestamp"] == 100
    assert "normalization" in result["selected"][0]["speech_matched_terms"]


def test_four_images_preserve_a_crowded_split_sequence_and_its_result():
    # Four uniform timeline bins miss the intermediate states at 12 and 13.
    # The third screen is only a brief intermediate step, not a denser slide.
    frames = [
        {"timestamp": stamp, "ocr": text, "path": f"/{stamp}.jpg"}
        for stamp, text in [
            (0, "B+树 数据结构"),
            (10, "B+树 叶子 10 20 插入 25"),
            (11, "B+树 叶子 10 20 25 30 满"),
            (12, "B+树 分裂 10 20 | 25 30"),
            (13, "B+树 提升 25 父节点"),
            (30, "B+树 新根 25 子节点 10 20 25 30"),
            (40, "B+树 谢谢观看"),
        ]
    ]
    original = copy.deepcopy(frames)
    result = QueryVisualSelector.select(
        "插入25如何导致B+树分裂和新根变化？", frames,
        visual_queries=["插入 25 叶子", "分裂 10 20 | 25 30", "提升 25 父节点", "新根 25"],
        limit=4,
    )
    assert [row["timestamp"] for row in result["selected"]] == [10, 12, 13, 30]
    assert frames == original
    assert result["unmatched_visual_queries"] == []
    assert all(row["path"] for row in result["selected"])


def test_repeated_heading_does_not_outrank_rare_screen_evidence():
    frames = [{"timestamp": index, "ocr": "Raft 日志复制 一致性算法课程"} for index in range(12)]
    frames[3]["ocr"] += " 冲突 prevLogTerm=3 leader term=4"
    frames[8]["ocr"] += " 截断 index=6 重新发送"
    result = QueryVisualSelector.select(
        "Raft日志复制在冲突后截断哪里？", frames,
        visual_queries=["冲突 prevLogTerm", "截断 index=6"], limit=2,
    )
    assert [row["timestamp"] for row in result["selected"]] == [3, 8]


def test_single_number_change_is_not_discarded_as_near_duplicate():
    common = "online softmax m l O exp normalization accumulator " * 8
    frames = [
        {"timestamp": 0, "ocr": common + "m = 2 l = 1.5", "image_digest": "before"},
        {"timestamp": 1, "ocr": common + "m = 4 l = 1.5", "image_digest": "middle"},
        {"timestamp": 2, "ocr": common + "m = 4 l = 2.7", "image_digest": "after"},
    ]
    result = QueryVisualSelector.select(
        "m从2到4后l如何从1.5变成2.7？", frames, limit=3,
    )
    assert [row["timestamp"] for row in result["selected"]] == [0, 1, 2]
    assert {"1.5", "2.7"} <= set().union(*(row["matched_terms"] for row in result["selected"]))


def test_speech_matches_never_claim_screen_matches_when_ocr_failed():
    frames = [{"timestamp": index * 10, "ocr": ""} for index in range(9)]
    speech = [Segment(10, 30, "这一段进行分裂，然后提升父节点")]
    result = QueryVisualSelector.select("分裂并提升父节点", frames, speech,
                                        visual_queries=["分裂", "提升父节点"], limit=4)
    assert result["status"] == "ocr_unavailable"
    assert result["unmatched_visual_queries"] == ["分裂", "提升父节点"]
    assert all(not row["matched_terms"] for row in result["selected"])
    assert any(row["speech_matched_terms"] for row in result["selected"])
    assert all(not row["matched_terms"] for row in result["visual_query_matches"])


def test_actual_duplicate_images_and_pts_are_deduplicated_but_ocr_is_not():
    frames = [
        {"timestamp": 1, "ocr": "节点 25", "image_digest": "a"},
        {"timestamp": 2, "ocr": "节点 25", "image_digest": "a"},
        {"timestamp": 3, "ocr": "节点 25", "image_digest": "b"},
        {"timestamp": 4, "ocr": "节点 25"},
        {"timestamp": 4, "ocr": "节点 25"},
    ]
    result = QueryVisualSelector.select("节点25的箭头变化", frames, limit=8)
    assert [row["timestamp"] for row in result["selected"]] == [1, 3, 4]


def test_return_to_a_previous_screen_remains_a_distinct_process_step():
    frames = [
        {"timestamp": 1, "ocr": "x = 10", "image_digest": "before"},
        {"timestamp": 2, "ocr": "x = 10", "image_digest": "before"},
        {"timestamp": 10, "ocr": "x = 11", "image_digest": "changed"},
        {"timestamp": 20, "ocr": "x = 10", "image_digest": "before"},
    ]
    result = QueryVisualSelector.select("x先改变再恢复到10，怎样发生的？", frames, limit=3)
    assert [row["timestamp"] for row in result["selected"]] == [1, 10, 20]
    assert result["selected"][0]["occurrence_timestamps"] == [1, 2]


def test_static_screen_occurrence_near_question_speech_retains_its_timestamp():
    frames = [{"timestamp": stamp, "ocr": "日志结构", "image_digest": "same"} for stamp in (0, 20)]
    result = QueryVisualSelector.select(
        "日志的冲突怎样解决", frames, [Segment(19, 21, "日志发生冲突")], limit=1)
    assert result["selected"][0]["timestamp"] == 20
    assert result["selected"][0]["occurrence_timestamps"] == [0, 20]


def test_vfr_frame_displayed_at_interval_start_keeps_its_actual_pts():
    frames = [
        {"timestamp": 4, "requested_timestamp": 5, "ocr": "x = 1"},
        {"timestamp": 6, "requested_timestamp": 7, "ocr": "x = 2"},
    ]
    result = QueryVisualSelector.select("x如何变化", frames, start=5, end=7, limit=4)
    assert len(result["selected"]) == 1
    assert result["selected"][0]["timestamp"] == 4
    assert result["selected"][0]["requested_timestamp"] == 5


def test_no_matches_use_timeline_coverage_and_report_the_missing_terms():
    frames = [{"timestamp": index * 10, "ocr": "课程目录"} for index in range(12)]
    result = QueryVisualSelector.select("局部最大值 rescale", frames, limit=4)
    assert result["status"] == "screen_no_query_match"
    assert result["unmatched_visual_queries"] == ["局部最大值 rescale"]
    assert "rescale" in result["unmatched_query_terms"]
    assert [row["timestamp"] for row in result["selected"]] == [10, 40, 70, 100]


def test_raw_probes_retain_brief_event_outside_navigation_and_speech_neighbors():
    # OCR has not run yet. The intermediate state is one frame at 31.0 seconds,
    # after a stable interval. It must survive before lexical ranking is possible.
    stamps = sorted({float(i) for i in range(100)} | {31.04})
    raw = [{"timestamp": stamp, "score": 0.01} for stamp in stamps]
    original = copy.deepcopy(raw)
    result = QueryVisualSelector.probe_candidates(
        "节点分裂", raw, [Segment(61, 64, "节点分裂以后更新父节点")],
        start=0, end=100, limit=24,
    )
    assert raw == original
    assert len(result) <= 24
    assert any(row["source_timestamp"] == 31.0 for row in result)
    assert any(60 <= row["timestamp"] <= 65 and row["probe_reason"] == "speech_neighborhood"
               for row in result)
    assert all(raw[row["source_candidate_cursor"]]["timestamp"] == row["source_timestamp"]
               for row in result)
    assert all(0 <= row["timestamp"] < 100 for row in result)


def test_refinement_returns_unseen_states_on_both_sides_of_the_anchor():
    raw = [{"timestamp": stamp, "protected": True} for stamp in (1, 4, 8, 9.5, 10, 10.5, 12, 16, 19)]
    result = QueryVisualSelector.refinement_candidates(
        [{"timestamp": 10}], raw, start=0, end=20, seen=[9.5, {"timestamp": 10.5}], limit=6,
    )
    assert all(row["timestamp"] not in (9.5, 10, 10.5) for row in result)
    assert any(row["timestamp"] < 10 for row in result)
    assert any(row["timestamp"] > 10 for row in result)
    assert all(row["probe_reason"] == "adjacent_screen_state" for row in result)

"""Prevent omitted claims and failed reproduction from disappearing into a score."""

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

benchmark = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/evaluate_fact_review.py"))


@pytest.fixture
def review_data(tmp_path):
    answer = "Active = 7.92 warp。L1/TEX = 93%。"
    writer = tmp_path / "answer.md"
    reader = tmp_path / "reader.json"
    writer.write_text(answer)
    reader.write_text('{"result": 0.2}')

    def artifact(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    reference = {"cases": {"case": {"required_claims": [{"id": "F1"}]}}}
    questions = {"case": [{"id": "R1"}]}
    inventory = benchmark["numeric_inventory"](answer)
    audit = []
    for row in inventory:
        if row["text"] == "1":
            audit.append(row | {"classification": "non_claim", "reason": "L1 field identifier"})
        else:
            audit.append(row | {"classification": "claim",
                                "claim_ids": ["F1" if row["text"] == "7.92" else "E1"]})
    review = {
        "case": "case", "label": "A", "reference_sha256": "reference-digest",
        "questions_sha256": "questions-digest", "writer": artifact(writer),
        "reader": artifact(reader), "audit_scope": "entire_answer", "numeric_audit": audit,
        "required_facts": [{"id": "F1", "status": "correct", "dimension": "binding",
                            "quote": "Active = 7.92 warp", "reason": "Source row matches",
                            "critical": False}],
        "extra_claims": [{"id": "E1", "status": "incorrect", "dimension": "binding",
                          "quote": "L1/TEX = 93%", "reason": "Source says 96.11%",
                          "critical": True}],
        "reader_tasks": [{"id": "R1", "status": "passed", "reason": "Reproduced"}],
    }
    return reference, questions, review, artifact


def evaluate(tmp_path, data, **kwargs):
    reference, questions, review, _ = data
    return benchmark["aggregate"](reference, questions, review, base=tmp_path,
                                  reference_hash="reference-digest",
                                  questions_hash="questions-digest", **kwargs)


def test_full_required_coverage_does_not_hide_extra_metric_error(tmp_path, review_data):
    result = evaluate(tmp_path, review_data)
    assert result["required_facts"]["correct"] == 1
    assert result["extra_claims"]["incorrect"] == 1
    assert result["critical_errors"] == 1
    assert result["claims"][1]["quote"] == "L1/TEX = 93%"
    assert result["reader_task_counts"] == {"passed": 1}


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["numeric_audit"].pop(), "every answer numeric token"),
    (lambda r: r["required_facts"].clear(), "Every required fact"),
    (lambda r: r["required_facts"].append(copy.deepcopy(r["required_facts"][0])), "Duplicate"),
    (lambda r: r["extra_claims"].clear(), "unknown or omitted claim"),
    (lambda r: r["reader_tasks"].clear(), "Every reader question"),
    (lambda r: r.update(audit_scope="required_facts_only"), "entire answer"),
    (lambda r: r["extra_claims"][0].update(quote="a corrected 96.11%"), "absent"),
])
def test_incomplete_or_changed_audit_is_rejected(tmp_path, review_data, mutation, match):
    mutation(review_data[2])
    with pytest.raises(ValueError, match=match):
        evaluate(tmp_path, review_data)


def test_writer_changes_invalidate_frozen_judgment(tmp_path, review_data):
    (tmp_path / "answer.md").write_text("A corrected answer")
    with pytest.raises(ValueError, match="writer hash mismatch"):
        evaluate(tmp_path, review_data)


def test_repeated_number_cannot_borrow_a_different_sentence_review(tmp_path, review_data):
    writer = tmp_path / "answer.md"
    writer.write_text(writer.read_text() + "DRAM = 93%。")
    review_data[2]["writer"] = review_data[3](writer)
    token = benchmark["numeric_inventory"](writer.read_text())[-1]
    review_data[2]["numeric_audit"].append(token | {"classification": "claim", "claim_ids": ["E1"]})
    with pytest.raises(ValueError, match="location is outside"):
        evaluate(tmp_path, review_data)


def test_reader_cannot_pass_missing_or_failed_execution(tmp_path, review_data):
    with pytest.raises(ValueError, match="requires an execution log"):
        evaluate(tmp_path, review_data, require_execution=["R1"])
    log = tmp_path / "execution.json"
    log.write_text(json.dumps({"case": "case", "task_id": "R1",
                              "reader_sha256": review_data[2]["reader"]["sha256"],
                              "status": "failed",
                              "checks": [{"actual": 0.4, "expected": 0.2, "passed": False}]}))
    review_data[2]["reader_tasks"][0]["execution"] = review_data[3](log)
    with pytest.raises(ValueError, match="without successful execution"):
        evaluate(tmp_path, review_data, require_execution=["R1"])
    review_data[2]["reader_tasks"][0]["status"] = "failed"
    result = evaluate(tmp_path, review_data, require_execution=["R1"])
    assert result["reader_tasks"][0]["execution"]["status"] == "failed"
    assert result["reader_task_counts"] == {"failed": 1}


def test_numeric_inventory_keeps_visible_labels_and_code_not_url_digits():
    text = "[Figure 2](https://example.com/p/93?t=480) `16 * 34_000_000 * 2` = 1.088e9"
    values = [row["text"] for row in benchmark["numeric_inventory"](text)]
    assert values == ["2", "16", "34_000_000", "2", "1.088e9"]


@pytest.mark.parametrize("url", ["https://example.com/docs", "<https://example.com/docs>"])
def test_url_exclusion_does_not_swallow_chinese_sentence_numbers(url):
    values = [row["text"] for row in benchmark["numeric_inventory"](
        f"参见 {url}。实际值为93%。"
    )]
    assert values == ["93%"]


def test_superscript_and_leading_decimal_remain_in_numeric_audit():
    values = [row["text"] for row in benchmark["numeric_inventory"]("10³ m² = .5e6 mm²")]
    assert values == ["10", "³", "²", ".5e6", "²"]


def test_unknown_execution_requirement_cannot_silently_skip_check(tmp_path, review_data):
    with pytest.raises(ValueError, match="Unknown required execution"):
        evaluate(tmp_path, review_data, require_execution=["wrong-task-id"])


@pytest.mark.parametrize("field", ["case", "task_id", "reader_sha256"])
def test_execution_cannot_be_borrowed_from_another_submission(tmp_path, review_data, field):
    payload = {"case": "case", "task_id": "R1",
               "reader_sha256": review_data[2]["reader"]["sha256"],
               "status": "passed", "checks": [{"passed": True, "actual": 0.2}]}
    payload[field] = "someone-else"
    log = tmp_path / "execution.json"
    log.write_text(json.dumps(payload))
    review_data[2]["reader_tasks"][0]["execution"] = review_data[3](log)
    with pytest.raises(ValueError, match="belongs to another"):
        evaluate(tmp_path, review_data, require_execution=["R1"])


def test_success_label_cannot_mask_a_failed_execution_check(tmp_path, review_data):
    log = tmp_path / "execution.json"
    log.write_text(json.dumps({"case": "case", "task_id": "R1",
                              "reader_sha256": review_data[2]["reader"]["sha256"],
                              "status": "passed", "checks": [{"passed": False}]}))
    review_data[2]["reader_tasks"][0]["execution"] = review_data[3](log)
    with pytest.raises(ValueError, match="despite a failed check"):
        evaluate(tmp_path, review_data, require_execution=["R1"])


def test_easier_execution_inputs_cannot_replace_frozen_variants(tmp_path, review_data):
    review_data[0]["reader_expected"] = {"R1": {"hidden_cases": [{"input": [0, 1]}]}}
    log = tmp_path / "execution.json"
    payload = {"case": "case", "task_id": "R1",
               "reader_sha256": review_data[2]["reader"]["sha256"],
               "status": "passed", "checks": [{"input": [1, 1], "passed": True}]}
    log.write_text(json.dumps(payload))
    review_data[2]["reader_tasks"][0]["execution"] = review_data[3](log)
    with pytest.raises(ValueError, match="missing a frozen input"):
        evaluate(tmp_path, review_data, require_execution=["R1"])
    payload["checks"].append({"input": [0, 1], "passed": True})
    log.write_text(json.dumps(payload))
    review_data[2]["reader_tasks"][0]["execution"] = review_data[3](log)
    result = evaluate(tmp_path, review_data, require_execution=["R1"])
    assert result["reader_tasks"][0]["execution"]["frozen_inputs_checked"] == 1

"""Aggregate a frozen fact/reader review without hiding errors in a total score.

This checks provenance and review coverage, not semantic truth. A judge must inspect
the source images, label every required fact, audit additional claims, and map every
numeric token in the author's answer. Reader execution logs are hashed, not executed
by this script. Use --inventory ANSWER.md to prepare the numeric audit skeleton.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

FACT_STATUSES = {"correct", "incorrect", "omitted", "unverifiable"}
TASK_STATUSES = {"passed", "partial", "failed"}
NUMBER = re.compile(
    r"[+-]?(?:\d[\d_,]*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?|[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+"
)
LINK_TARGET = re.compile(
    r"!?\[[^\]\n]*\]\((?:[^()\n]|\([^()\n]*\))*\)"
    r"|<https?://[^<>\s]+>|https?://[^\s<>\[\]{}()，。；！？、（）]+"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_inventory(answer: str) -> list[dict]:
    """Return exact offsets, retaining code numbers and identifier digits.

    Markdown labels remain visible prose. Link destinations and bare URLs are
    excluded because their timestamp/path digits are not asserted quantities.
    A judge can mark list numbering, identifiers and citations as non_claim.
    """
    excluded = []
    for match in LINK_TARGET.finditer(answer):
        token = match.group()
        offset = token.find("](")
        excluded.append((match.start() + offset + 2 if offset >= 0 else match.start(),
                         match.end()))
    return [{"start": match.start(), "end": match.end(), "text": match.group()}
            for match in NUMBER.finditer(answer)
            if not any(left <= match.start() < right for left, right in excluded)]


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _indexed(rows, label):
    if not isinstance(rows, list):
        raise TypeError(f"{label} must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"{label} rows must be objects")
        key = _text(row.get("id"), f"{label} id")
        if key in result:
            raise ValueError(f"Duplicate {label} id: {key}")
        result[key] = row
    return result


def _artifact(record, label, base):
    if not isinstance(record, dict):
        raise TypeError(f"{label} must include path and sha256")
    path = Path(_text(record.get("path"), f"{label} path"))
    if not path.is_absolute():
        path = base / path
    actual = sha256(path)
    if record.get("sha256") != actual:
        raise ValueError(f"{label} hash mismatch")
    return path, actual


def _quote_covers(answer, quote, start, end):
    position = answer.find(quote)
    while position >= 0:
        if position <= start and end <= position + len(quote):
            return True
        position = answer.find(quote, position + 1)
    return False


def aggregate(reference, questions, review, *, base: Path, reference_hash: str,
              questions_hash: str, require_execution=()) -> dict:
    """Validate one blinded author/reader pair and return publishable counts."""
    if review.get("reference_sha256") != reference_hash:
        raise ValueError("Reference hash mismatch")
    if review.get("questions_sha256") != questions_hash:
        raise ValueError("Reader questions hash mismatch")
    case = _text(review.get("case"), "case")
    if case not in reference["cases"] or case not in questions:
        raise ValueError("Unknown case")
    label = _text(review.get("label"), "blinded label")
    writer, writer_hash = _artifact(review.get("writer"), "writer", base)
    _, reader_hash = _artifact(review.get("reader"), "reader", base)
    answer = writer.read_text(encoding="utf-8")
    required = _indexed(reference["cases"][case]["required_claims"], "reference fact")
    judgments = _indexed(review.get("required_facts"), "required fact")
    if required.keys() != judgments.keys():
        raise ValueError("Every required fact must be reviewed exactly once")
    extras = _indexed(review.get("extra_claims"), "extra claim")
    if required.keys() & extras.keys():
        raise ValueError("Extra claim IDs overlap required facts")

    rows = []
    for kind, source in (("required", judgments), ("extra", extras)):
        for key, row in source.items():
            status = row.get("status")
            if status not in FACT_STATUSES or (kind == "extra" and status == "omitted"):
                raise ValueError(f"Invalid fact status for {key}")
            reason = _text(row.get("reason"), f"{key} reason")
            dimension = _text(row.get("dimension"), f"{key} dimension")
            critical = row.get("critical")
            if not isinstance(critical, bool):
                raise TypeError(f"{key} critical must be boolean")
            if critical and status == "correct":
                raise ValueError(f"Correct claim {key} cannot be a critical error")
            quote = row.get("quote", "")
            if status != "omitted":
                _text(quote, f"{key} exact answer quote")
                if quote not in answer:
                    raise ValueError(f"{key} quote is absent from the frozen answer")
            elif quote:
                raise ValueError(f"Omitted claim {key} cannot have an answer quote")
            rows.append({"id": key, "kind": kind, "dimension": dimension,
                         "status": status, "critical": critical, "reason": reason,
                         "quote": quote})

    if review.get("audit_scope") != "entire_answer":
        raise ValueError("Review must audit the entire answer, including extra claims")
    inventory = numeric_inventory(answer)
    audit = review.get("numeric_audit")
    if not isinstance(audit, list) or len(audit) != len(inventory):
        raise ValueError("Numeric audit must cover every answer numeric token")
    all_claims = judgments | extras
    audited_numbers = []
    for expected, item in zip(inventory, audit):
        if not isinstance(item, dict) or any(item.get(k) != v for k, v in expected.items()):
            raise ValueError("Numeric audit offsets/text differ from the frozen answer")
        classification = item.get("classification")
        ids = item.get("claim_ids", [])
        if classification == "claim":
            if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
                raise ValueError("Each numeric claim must link to reviewed claim IDs")
            for claim_id in ids:
                if claim_id not in all_claims or all_claims[claim_id]["status"] == "omitted":
                    raise ValueError("Numeric audit links to an unknown or omitted claim")
                if not _quote_covers(answer, all_claims[claim_id].get("quote", ""),
                                     expected["start"], expected["end"]):
                    raise ValueError("Numeric token location is outside its linked claim quote")
        elif classification == "non_claim":
            if ids:
                raise ValueError("Non-claim numeric tokens cannot link to factual claims")
            _text(item.get("reason"), "non-claim numeric audit reason")
        else:
            raise ValueError("Numeric audit classification must be claim or non_claim")
        audited_numbers.append(expected | {"classification": classification,
                                          "claim_ids": ids,
                                          "reason": item.get("reason", "")})

    expected_tasks = _indexed(questions[case], "reader question")
    tasks = _indexed(review.get("reader_tasks"), "reader task")
    if expected_tasks.keys() != tasks.keys():
        raise ValueError("Every reader question must be judged exactly once")
    known_task_ids = {row["id"] for items in questions.values() for row in items}
    if set(require_execution) - known_task_ids:
        raise ValueError("Unknown required execution task")
    required_executions = set(require_execution) & tasks.keys()
    reader_results = []
    for key, row in tasks.items():
        status = row.get("status")
        if status not in TASK_STATUSES:
            raise ValueError(f"Invalid reader status for {key}")
        result = {"id": key, "status": status,
                  "reason": _text(row.get("reason"), f"{key} reader reason")}
        execution = row.get("execution")
        if key in required_executions and execution is None:
            raise ValueError(f"Reader task {key} requires an execution log")
        if execution is not None:
            log, digest = _artifact(execution, f"{key} execution", base)
            logged = json.loads(log.read_text(encoding="utf-8"))
            if (logged.get("case") != case or logged.get("task_id") != key
                    or logged.get("reader_sha256") != reader_hash):
                raise ValueError(f"Reader execution {key} belongs to another case, task or reader")
            execution_status = logged.get("status")
            if execution_status not in {"passed", "failed", "not_run"}:
                raise ValueError(f"Invalid execution status for {key}")
            if status == "passed" and execution_status != "passed":
                raise ValueError(f"Reader task {key} passed without successful execution")
            if not isinstance(logged.get("checks"), list) or not logged["checks"]:
                raise ValueError(f"Reader execution {key} must contain observed checks")
            if any(not isinstance(check, dict) or not isinstance(check.get("passed"), bool)
                   for check in logged["checks"]):
                raise ValueError(f"Reader execution {key} needs explicit observed check outcomes")
            if execution_status == "passed" and not all(c["passed"] for c in logged["checks"]):
                raise ValueError(f"Reader execution {key} passed despite a failed check")
            frozen_inputs = [item["input"] for item in
                             reference.get("reader_expected", {}).get(key, {}).get("hidden_cases", [])]
            if any(not any(check.get("input") == value for check in logged["checks"])
                   for value in frozen_inputs):
                raise ValueError(f"Reader execution {key} is missing a frozen input")
            result["execution"] = {"sha256": digest, "status": execution_status,
                                   "checks": logged["checks"],
                                   "frozen_inputs_checked": len(frozen_inputs)}
        reader_results.append(result)

    dimensions = {}
    for row in rows:
        counts = dimensions.setdefault(row["dimension"], Counter())
        counts[row["status"]] += 1
    required_counts = Counter(row["status"] for row in rows if row["kind"] == "required")
    extra_counts = Counter(row["status"] for row in rows if row["kind"] == "extra")
    return {
        "schema_version": 1, "case": case, "label": label,
        "metric": "per-claim factual audit and independent reader tasks; no total score",
        "provenance": {"reference_sha256": reference_hash,
                       "reader_questions_sha256": questions_hash,
                       "writer_sha256": writer_hash, "reader_sha256": reader_hash},
        "required_facts": {status: required_counts[status] for status in sorted(FACT_STATUSES)},
        "extra_claims": {status: extra_counts[status] for status in sorted(FACT_STATUSES)},
        "critical_errors": sum(row["critical"] for row in rows),
        "dimensions": {key: dict(value) for key, value in dimensions.items()},
        "claims": rows, "numeric_audit": audited_numbers,
        "reader_task_counts": dict(Counter(row["status"] for row in reader_results)),
        "reader_tasks": reader_results,
        "execution_required": sorted(required_executions),
        "scope": "Manual semantic judgments; script verifies coverage and hashes, not truth. "
                 "Reader code execution is limited reproduction, not a real GPU experiment.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-execution", action="append", default=[])
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(numeric_inventory(args.inventory.read_text(encoding="utf-8")),
                         ensure_ascii=False, indent=2))
        return
    if not all((args.reference, args.questions, args.review, args.output)):
        parser.error("Provide --reference, --questions, --review and --output")
    try:
        report = aggregate(
            json.loads(args.reference.read_text(encoding="utf-8")),
            json.loads(args.questions.read_text(encoding="utf-8")),
            json.loads(args.review.read_text(encoding="utf-8")),
            base=args.review.resolve().parent,
            reference_hash=sha256(args.reference), questions_hash=sha256(args.questions),
            require_execution=args.require_execution,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    report["provenance"]["review_sha256"] = sha256(args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("case", "label", "required_facts", "critical_errors", "reader_task_counts")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

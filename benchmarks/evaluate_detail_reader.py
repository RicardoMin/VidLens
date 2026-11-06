"""Execute frozen SQL, packet-bin and classification-metric reader tasks.
Review submitted Python before running. This is not a security sandbox.
"""

import argparse
import contextlib
import copy
import datetime
import hashlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path

E = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_text())


def jsonable(v):
    try:
        return json.loads(json.dumps(v, ensure_ascii=False))
    except (TypeError, ValueError):
        return repr(v)


def eq(a, b):
    if isinstance(b, bool):
        return isinstance(a, bool) and a == b
    if isinstance(b, int):
        return isinstance(a, (int, float)) and not isinstance(a, bool) and a == b
    if isinstance(b, float):
        return (
            isinstance(a, (int, float))
            and not isinstance(a, bool)
            and math.isfinite(a)
            and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
        )
    if isinstance(b, dict):
        return (
            isinstance(a, dict) and a.keys() == b.keys() and all(eq(a[k], v) for k, v in b.items())
        )
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return a == b


def samples(ex):
    if ex["case"] == "zh-flower-confusion-smalltext":
        return [
            ("source_video" if i == 0 else f"hidden_{i - 1}", v)
            for i, v in enumerate(ex["hidden_cases"])
        ]
    return [("public_sample", ex["public_case"])] + [
        (f"hidden_{i}", v) for i, v in enumerate(ex["hidden_cases"])
    ]


def validated_worker_result(result, ex):
    """Reject incomplete/invalid worker output before assigning a success status."""
    expected = samples(ex)
    if not isinstance(result, dict):
        raise TypeError("Worker result must be an object")
    checks = result.get("checks")
    if not isinstance(checks, list) or not checks or len(checks) != len(expected):
        raise ValueError("Worker must return every frozen check exactly once")
    for check, (name, sample) in zip(checks, expected):
        if (
            not isinstance(check, dict)
            or check.get("name") != name
            or check.get("input") != sample["input"]
            or check.get("expected") != sample["expected"]
            or not isinstance(check.get("passed"), bool)
            or "observed" not in check
        ):
            raise ValueError("Worker check does not match its frozen input and result schema")
    # Do not let an unexpected worker field override case, hashes or status in
    # the outer log. The parent, not the submission process, owns those fields.
    return {"checks": checks, "captured_output": result.get("captured_output", "")}


def worker(code, ex):
    logs = io.StringIO()
    checks = []
    try:
        with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
            spec = importlib.util.spec_from_file_location("reader_submission", code)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    except (Exception, SystemExit) as error:  # noqa: BLE001 - preserve failed submissions
        return {
            "checks": [
                {
                    "name": n,
                    "input": v["input"],
                    "expected": v["expected"],
                    "observed": {"import_error": repr(error)},
                    "passed": False,
                }
                for n, v in samples(ex)
            ],
            "captured_output": logs.getvalue(),
        }
    for name, sample in samples(ex):
        inp = copy.deepcopy(sample["input"])
        expected = copy.deepcopy(sample["expected"])
        check = {"name": name, "input": inp, "expected": expected}
        details = []
        try:
            with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
                if ex["case"] != "zh-flower-confusion-smalltext":
                    got = getattr(module, ex["interface"])(copy.deepcopy(inp))
                    check.update(observed=jsonable(got), passed=eq(got, expected))
                else:
                    if "source" in inp:
                        C = module.video_case()
                        k = inp["k"]
                        target_metrics = expected["metrics"]
                        details.append(
                            {
                                "name": "video_case",
                                "observed": jsonable(C),
                                "expected": expected["C"],
                                "passed": eq(C, expected["C"]),
                            }
                        )
                    else:
                        C = copy.deepcopy(inp["C"])
                        k = inp["k"]
                        target_metrics = expected
                    try:
                        got = module.metrics(copy.deepcopy(C), k)
                        for key, value in target_metrics.items():
                            observed = got.get(key) if isinstance(got, dict) else None
                            details.append(
                                {
                                    "name": f"metrics.{key}",
                                    "observed": jsonable(observed),
                                    "expected": value,
                                    "passed": eq(observed, value),
                                }
                            )
                    except (Exception, SystemExit) as err:  # noqa: BLE001 - preserve failed submissions
                        details.append(
                            {
                                "name": "metrics",
                                "observed": repr(err),
                                "expected": target_metrics,
                                "passed": False,
                            }
                        )
                    try:
                        cells = module.annotation_cells(copy.deepcopy(C))
                        cells = [tuple(row) for row in cells]
                        target = [
                            (j, i, str(value))
                            for i, row in enumerate(C)
                            for j, value in enumerate(row)
                        ]
                        details.append(
                            {
                                "name": "annotation_cells",
                                "observed": jsonable(cells),
                                "expected": jsonable(target),
                                "passed": cells == target,
                            }
                        )
                    except (Exception, SystemExit) as err:  # noqa: BLE001 - preserve failed submissions
                        details.append(
                            {
                                "name": "annotation_cells",
                                "observed": repr(err),
                                "expected": "row order with x=j,y=i,text=str(C[i][j])",
                                "passed": False,
                            }
                        )
                    check.update(observed=details, passed=all(x["passed"] for x in details))
        except (Exception, SystemExit) as error:  # noqa: BLE001 - preserve failed submissions
            check.update(observed={"error": repr(error)}, passed=False)
        checks.append(check)
    return {"checks": checks, "captured_output": logs.getvalue()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--reader", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--timeout", type=float, default=20)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--reference", type=Path, default=E / "detail-cases.json")
    p.add_argument("--questions", type=Path, default=E / "detail-questions.json")
    a = p.parse_args()
    ref = load(a.reference)
    matches = [(k, v) for k, v in ref["reader_expected"].items() if v["case"] == a.case]
    if not math.isfinite(a.timeout) or a.timeout <= 0:
        p.error("timeout must be finite and positive")
    if len(matches) != 1:
        p.error("Unknown/ambiguous case")
    task, ex = matches[0]
    if a.worker:
        print(json.dumps(worker(a.code.resolve(), ex), ensure_ascii=False))
        return
    if not a.reader or not a.output:
        p.error("--reader and --output required")
    reader = a.reader.resolve()
    code = a.code.resolve()
    before = {
        "reader_sha256": sha(reader),
        "code_sha256": sha(code),
        "reference_sha256": sha(a.reference),
        "questions_sha256": sha(a.questions),
    }
    start = time.monotonic()
    began = datetime.datetime.now(datetime.UTC).isoformat()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--case",
                a.case,
                "--code",
                str(code),
                "--reference",
                str(a.reference.resolve()),
                "--questions",
                str(a.questions.resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=a.timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"worker exited {proc.returncode}: {proc.stderr[-2000:]}")
        result = validated_worker_result(json.loads(proc.stdout), ex)
        status = "passed" if all(c["passed"] for c in result["checks"]) else "failed"
    except (Exception, SystemExit) as error:  # noqa: BLE001 - preserve failed submissions
        result = {
            "checks": [
                {
                    "name": n,
                    "input": v["input"],
                    "expected": v["expected"],
                    "observed": {"execution_error": repr(error)},
                    "passed": False,
                }
                for n, v in samples(ex)
            ]
        }
        status = "not_run" if isinstance(error, subprocess.TimeoutExpired) else "failed"
    if sha(reader) != before["reader_sha256"] or sha(code) != before["code_sha256"]:
        raise RuntimeError("Submission changed during execution")
    if (sha(a.reference) != before["reference_sha256"]
            or sha(a.questions) != before["questions_sha256"]):
        raise RuntimeError("Frozen evaluation inputs changed during execution")
    log = {
        "schema_version": 1,
        "case": a.case,
        "task_id": task,
        **before,
        "reader_path": str(reader),
        "code_path": str(code),
        "started_at_utc": began,
        "elapsed_seconds": time.monotonic() - start,
        "status": status,
        **result,
        "scope": "Real execution of frozen synthetic/data-reconstruction tasks; not an original database/pcap/training rerun.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "path": str(a.output.resolve()),
                "sha256": sha(a.output),
                "case": a.case,
                "task_id": task,
                "status": status,
                "checks": len(result["checks"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

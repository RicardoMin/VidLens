"""Source reconstruction and unseen boundaries must survive executable evaluation."""

import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "benchmarks/evaluate_detail_reader.py"
REFERENCE = json.loads((ROOT / "benchmarks/detail-cases.json").read_text())


def execute_submission(tmp_path, case, source):
    code = tmp_path / "reader.py"
    reader = tmp_path / "reader.md"
    output = tmp_path / "execution.json"
    code.write_text(source)
    reader.write_text("Regression fixture: intentionally incorrect reader submission.")
    subprocess.run(
        [sys.executable, str(RUNNER), "--case", case, "--code", str(code),
         "--reader", str(reader), "--output", str(output)],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return json.loads(output.read_text())


def test_wrong_source_matrix_fails_even_when_all_reported_metrics_and_variants_match(tmp_path):
    wrong = copy.deepcopy(REFERENCE["reader_expected"]["2"]["source_case"]["expected"]["C"])
    # Swap two off-diagonal values away from class 0: n, trace, daisy P/R and
    # every requested aggregate stay correct, while the video reconstruction is wrong.
    wrong[1][2], wrong[2][1] = wrong[2][1], wrong[1][2]
    source = f"def video_case():\n    return {wrong!r}\n" + '''
def metrics(C, k):
    total = sum(sum(row) for row in C)
    diagonal = sum(C[i][i] for i in range(len(C)))
    accuracy = diagonal / total if total else 0
    support = sum(C[k])
    predicted = sum(row[k] for row in C)
    return dict(n=total, correct=diagonal, accuracy=accuracy,
                micro_precision=accuracy, micro_recall=accuracy, micro_f1=accuracy,
                class_precision=C[k][k] / predicted if predicted else 0,
                class_recall=C[k][k] / support if support else 0)

def annotation_cells(C):
    return [(column, row, str(value))
            for row, values in enumerate(C) for column, value in enumerate(values)]
'''
    log = execute_submission(tmp_path, "zh-flower-confusion-smalltext", source)
    assert log["status"] == "failed"
    source_check, *variants = log["checks"]
    assert source_check["name"] == "source_video" and not source_check["passed"]
    assert source_check["observed"][0]["name"] == "video_case"
    assert source_check["observed"][0]["passed"] is False
    assert all(check["passed"] for check in source_check["observed"][1:])
    assert len(variants) == 3 and all(check["passed"] for check in variants)


def test_public_packet_sample_does_not_hide_decimal_boundary_failure(tmp_path):
    source = '''
def summarize_io(records):
    if not records:
        return []
    bins = {}
    for packet in records:
        index = int(packet['t'] / 0.05)  # Deliberate decimal-boundary bug.
        bins.setdefault(index, []).append(packet)
    result = []
    for index in range(max(bins) + 1):
        packets = bins.get(index, [])
        rtt = [p['ack_rtt'] for p in packets if 'ack_rtt' in p]
        bif = [p['bytes_in_flight'] for p in packets if 'bytes_in_flight' in p]
        sent = sum(p['frame_len'] for p in packets if p['src'] == '10.10.12.1')
        result.append(dict(bin=index, rtt_seconds=max(rtt) if rtt else None,
                           bif_bytes=max(bif) if bif else None, sender_bytes=sent,
                           sender_bits_per_second=sent * 8 / 0.05))
    return result
'''
    log = execute_submission(tmp_path, "wireshark_io_graph_units", source)
    assert log["status"] == "failed"
    assert log["checks"][0]["name"] == "public_sample" and log["checks"][0]["passed"]
    boundary = next(c for c in log["checks"] if c["name"] == "hidden_1")
    assert boundary["passed"] is False
    assert boundary["input"][0]["t"] == 0.15
    assert boundary["observed"][2]["sender_bytes"] == 100
    assert boundary["expected"][2]["sender_bytes"] == 0
    assert len(log["checks"]) == 4


def test_empty_worker_output_is_a_failed_log_with_all_frozen_inputs(tmp_path, monkeypatch):
    benchmark = runpy.run_path(str(RUNNER))
    code, reader, output = (tmp_path / name for name in ["reader.py", "reader.md", "log.json"])
    code.write_text("# No valid execution result was returned.\n")
    reader.write_text("Transport failure fixture.")
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--case", "mysql_explain_rows_filtered",
                                    "--code", str(code), "--reader", str(reader),
                                    "--output", str(output)])
    monkeypatch.setattr(
        benchmark["subprocess"], "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, '{"checks": []}', ""),
    )
    benchmark["main"]()
    log = json.loads(output.read_text())
    expected = REFERENCE["reader_expected"]["m_reproduce_queries"]
    assert log["status"] == "failed"
    assert len(log["checks"]) == 1 + len(expected["hidden_cases"])
    assert all(check["passed"] is False for check in log["checks"])
    assert [c["input"] for c in log["checks"][1:]] == [c["input"] for c in expected["hidden_cases"]]
    assert "every frozen check" in log["checks"][0]["observed"]["execution_error"]

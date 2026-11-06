"""A matching number somewhere in a chart must not certify a proposed metric."""

import copy
from decimal import Decimal

import pytest

from vidlens.fact_check import audit_claim, parse_value, validate_claims
from vidlens.models import VidLensError


def claim(**changes):
    return {"id": "cache", "timestamp": 476.3, "label": "L1/TEX Cache Throughput",
            "value": "96.11", "unit": "%", "scope": "this kernel invocation",
            "meaning": "percentage of peak throughput", **changes}


def region(text, x=0.1, y=0.2, width=0.7, height=0.025):
    return {"text": text, "confidence": 0.98, "box": [x, y, width, height]}


def audit(item, regions):
    return audit_claim(item, regions, image_width=852, image_height=480)


def test_vision_line_binds_metric_value_unit_and_never_certifies_meaning():
    result = audit(claim(), [region("L1/TEX Cache Throughput   96.11 %")])
    assert result["status"] == "exact_literal"
    assert result["requires_visual_review"] and result["requires_semantic_review"]
    assert result["semantic_status"] == "not_checked"
    assert result["matched_rows"][0]["regions"][0]["box"] == [0.1, 0.2, 0.7, 0.025]
    # A different interpretation does not acquire a semantic pass from matching digits.
    wrong = audit(claim(meaning="percentage of requests that hit L1"),
                  [region("L1/TEX Cache Throughput   96.11 %")])
    assert wrong["semantic_status"] == "not_checked"


def test_wrong_value_cannot_match_a_neighboring_metric():
    regions = [region("L1/TEX Cache Throughput 96.11 %"),
               region("L1/TEX Hit Rate 93.00 %", y=0.24)]
    result = audit(claim(value="93"), regions)
    assert result["status"] == "mismatch"
    assert result["observed_values"][0]["value"] == "96.11"
    assert len(result["matched_rows"]) == 1


def test_superscripts_do_not_silently_change_magnitude_or_unit():
    with pytest.raises(ValueError):
        parse_value("10³")
    assert audit(claim(label="Count", value="103", unit=""),
                 [region("Count 10³")])["status"] != "exact_literal"
    assert audit(claim(label="Area", value="12", unit="m2"),
                 [region("Area 12 m²")])["status"] == "mismatch"
    assert parse_value("１２．５") == Decimal("12.5")


def test_word_boxes_and_units_before_values_preserve_row_binding():
    regions = [region("L1/TEX", width=0.1), region("Cache", x=0.22, width=0.07),
               region("Throughput", x=0.31, width=0.14),
               region("%", x=0.6, width=0.03), region("96.11", x=0.72, width=0.1),
               region("L1/TEX", y=0.24, width=0.1),
               region("Hit", x=0.22, y=0.24, width=0.05),
               region("Rate", x=0.29, y=0.24, width=0.07),
               region("93.00", x=0.72, y=0.24, width=0.1)]
    result = audit(claim(), regions)
    assert result["status"] == "exact_literal"
    assert result["observed_values"][0]["unit"] == "%"
    assert len(result["matched_rows"][0]["regions"]) == 5


def test_duplicate_labels_and_multiple_columns_require_disambiguation():
    result = audit(claim(), [region("L1/TEX Cache Throughput 96.11 %"),
                            region("L1/TEX Cache Throughput 90.2 %", y=0.4)])
    assert result["status"] == "ambiguous"
    assert result["matched_row_count"] == 2
    result = audit(claim(), [region("L1/TEX Cache Throughput 96.11 % 90.2 %")])
    assert result["status"] == "ambiguous"
    assert "multiple_numbers_on_label_row" in result["issues"]


def test_unit_scale_case_and_percentage_are_not_implicitly_converted():
    for expected, visible in [("MB/s", "GB/s"), ("MB/s", "Mb/s"), ("", "%")]:
        result = audit(claim(label="Bandwidth", value="12", unit=expected),
                       [region(f"Bandwidth 12 {visible}")])
        assert result["status"] == "mismatch"
        assert "unit_mismatch" in result["issues"]
    result = audit(claim(value="0.9611", unit=""),
                   [region("L1/TEX Cache Throughput 96.11 %")])
    assert result["status"] == "mismatch"


def test_grouped_thousands_scientific_notation_and_unicode_minus():
    assert parse_value("1,234.50") == Decimal("1234.5")
    assert parse_value("−1.25e−3") == Decimal("-0.00125")
    result = audit(claim(label="Duration", value="7.6186e2", unit="us"),
                   [region("Duration 761.86 us")])
    assert result["status"] == "exact_literal"
    result = audit(claim(label="Count", value="1234.5", unit=""),
                   [region("Count 1,234.50")])
    assert result["status"] == "exact_literal"


def test_missing_layout_header_only_units_and_invalid_regions_do_not_pass():
    assert audit(claim(), [])["status"] == "unreadable"
    regions = [region("Unit: %", y=0.1), region("L1/TEX Cache Throughput 96.11")]
    result = audit(claim(), regions)
    assert result["status"] == "unreadable"
    assert "unit_not_visible_on_label_row" in result["issues"]
    result = audit(claim(), [{"text": "L1/TEX Cache Throughput 96.11 %", "box": [0, 0, 2, 1]}])
    assert result["status"] == "unreadable"
    assert "malformed_ocr_regions_ignored" in result["issues"]


def test_no_substring_labels_or_cross_row_chaining():
    assert audit(claim(label="Hit Rate"), [region("OtherHit Rate 96.11 %")])["status"] == "unreadable"
    regions = [region("L1/TEX Cache Throughput", height=0.03),
               region("unrelated", x=0.81, y=0.212, width=0.12, height=0.03),
               region("96.11 %", x=0.81, y=0.23, width=0.12, height=0.03)]
    assert audit(claim(), regions)["status"] != "exact_literal"


@pytest.mark.parametrize("change", [
    {"value": "nan"}, {"value": "1,23"}, {"value": "93%"}, {"value": "1/4"},
    {"value": 96.11}, {"timestamp": True}, {"timestamp": float("inf")},
    {"timestamp": 10 ** 1000},
    {"timestamp": -1}, {"box": [0.9, 0.1, 0.2, 0.3]}, {"box": [0, 0, True, 1]},
    {"scope": ""}, {"unit": None}, {"meaning": " "}, {"extra": "unsupported"},
])
def test_claim_boundary_rejects_malformed_claims(change):
    with pytest.raises(VidLensError, match=".") as caught:
        validate_claims([claim(**change)])
    assert caught.value.code == "invalid_fact_claims"


def test_claim_validation_preserves_input_and_rejects_duplicate_ids():
    source = [claim(box=[0.1, 0.1, 0.8, 0.8])]
    original = copy.deepcopy(source)
    cleaned = validate_claims(source)
    cleaned[0]["box"][0] = 0
    assert source == original
    with pytest.raises(VidLensError):
        validate_claims([claim(), claim()])
    with pytest.raises(VidLensError):
        validate_claims([])

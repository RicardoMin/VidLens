"""Conservative, spatial OCR checks for quantitative claims.

This module checks a proposed transcription against a screen row. It cannot establish
that OCR is correct, that a metric means what the author says, or that a selected row
belongs to the right experiment. Even an exact literal match needs visual and semantic
review by the host AI.
"""

from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation

from .models import VidLensError

_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_VALUE = re.compile(rf"{_NUMBER}\Z")
_SCREEN_NUMBER = re.compile(rf"(?<![\w.,]){_NUMBER}(?![\w.,])")
_CLAIM_FIELDS = {"id", "timestamp", "label", "value", "unit", "scope", "meaning", "box"}
_SEPARATORS = " \t\r\n:：=|….,;；[](){}【】（）"


def _text(value: str) -> str:
    # Compatibility normalization would turn 10³ into 103 and m² into m2.
    # Fold only full-width ASCII, whose numerical meaning is unchanged.
    value = unicodedata.normalize("NFC", value)
    value = "".join(chr(ord(char) - 0xFEE0) if "\uff01" <= char <= "\uff5e" else char
                    for char in value)
    return value.replace("−", "-").strip()


def parse_value(value: str) -> Decimal:
    """Parse a scalar without silently converting units, percentages or fractions."""
    if not isinstance(value, str) or len(value) > 100:
        raise ValueError("Use a finite decimal string of at most 100 characters.")
    normalized = _text(value)
    if not _VALUE.fullmatch(normalized):
        raise ValueError("Use a finite decimal, scientific notation or grouped thousands.")
    try:
        result = Decimal(normalized.replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("The decimal is outside the supported representation.") from exc
    if not result.is_finite():
        raise ValueError("The decimal must be finite.")
    return result


def _finite(value) -> bool:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _box(value) -> bool:
    return (
        isinstance(value, (list, tuple)) and len(value) == 4
        and all(_finite(item) for item in value)
        and 0 <= value[0] < 1 and 0 <= value[1] < 1
        and 0 < value[2] <= 1 and 0 < value[3] <= 1
        and value[0] + value[2] <= 1.000001
        and value[1] + value[3] <= 1.000001
    )


def validate_claims(claims: list[dict]) -> list[dict]:
    """Return independent, normalized inputs or one stable boundary error."""
    def invalid(message):
        raise VidLensError("invalid_fact_claims", message)

    if not isinstance(claims, list) or not 1 <= len(claims) <= 20:
        invalid("Provide 1..20 quantitative claims.")
    result, identifiers = [], set()
    limits = {"id": 80, "label": 200, "value": 100, "unit": 80, "scope": 500, "meaning": 600}
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) - _CLAIM_FIELDS:
            invalid("Each claim must be an object containing only documented claim fields.")
        cleaned = {}
        for key, limit in limits.items():
            value = claim.get(key)
            if (not isinstance(value, str) or len(value) > limit
                    or (key != "unit" and not value.strip())):
                invalid(f"Claim {key} must be a {'possibly empty ' if key == 'unit' else ''}"
                        f"string of at most {limit} characters.")
            cleaned[key] = value.strip()
        if cleaned["id"] in identifiers:
            invalid("Claim IDs must be unique.")
        identifiers.add(cleaned["id"])
        try:
            parse_value(cleaned["value"])
        except ValueError as exc:
            invalid(str(exc))
        timestamp = claim.get("timestamp")
        if not _finite(timestamp) or timestamp < 0:
            invalid("Each claim needs a finite, non-negative source timestamp.")
        cleaned["timestamp"] = timestamp
        if "box" in claim:
            if not _box(claim["box"]):
                invalid("box must be normalized x/y/width/height inside the frame.")
            cleaned["box"] = list(claim["box"])
        result.append(cleaned)
    return result


def _union(regions):
    left = min(region["box"][0] for region in regions)
    top = min(region["box"][1] for region in regions)
    right = max(region["box"][0] + region["box"][2] for region in regions)
    bottom = max(region["box"][1] + region["box"][3] for region in regions)
    return [left, top, right - left, bottom - top]


def _rows(regions, image_height):
    """Group same-baseline words, never grow a row through adjacent-row chaining."""
    rows = []
    for region in sorted(regions, key=lambda item: item["box"][1] + item["box"][3] / 2):
        _, y, _, height = region["box"]
        center = y + height / 2
        if rows:
            anchor = rows[-1][0]["box"]
            anchor_center = anchor[1] + anchor[3] / 2
            overlap = min(y + height, anchor[1] + anchor[3]) - max(y, anchor[1])
            # Pixel tolerance is only for OCR box rounding; vertical overlap remains required.
            if (abs(center - anchor_center) <= max(1.5 / image_height,
                                                  0.4 * min(height, anchor[3]))
                    and overlap >= 0.5 * min(height, anchor[3])):
                rows[-1].append(region)
                continue
        rows.append([region])
    return [sorted(row, key=lambda item: item["box"][0]) for row in rows]


def _label_pattern(label):
    chars = [char for char in _text(label) if not char.isspace()]
    return re.compile(r"\s*".join(re.escape(char) for char in chars), re.IGNORECASE)


def _label_matches(pattern, text):
    for match in pattern.finditer(text):
        # A label may span OCR words, but must not be a substring of another identifier.
        if (match.start() and text[match.start() - 1].isalnum()
                and text[match.start()].isalnum()):
            continue
        if (match.end() < len(text) and text[match.end()].isalnum()
                and text[match.end() - 1].isalnum()):
            continue
        yield match


def _unit(text):
    return re.sub(r"\s+", "", _text(text).strip(_SEPARATORS))


def _observations(suffix):
    numbers = list(_SCREEN_NUMBER.finditer(suffix))
    result = []
    for match in numbers:
        try:
            parsed = parse_value(match.group())
        except ValueError:
            continue
        unit, unit_text = None, None
        if len(numbers) == 1:
            before, after = suffix[:match.start()], suffix[match.end():]
            left, right = _unit(before), _unit(after)
            if not left or not right or left == right:
                unit, unit_text = left or right, before.strip() if left else after.strip()
        result.append({"text": match.group(), "value": str(parsed),
                       "unit": unit, "unit_text": unit_text})
    return result


def audit_claim(claim: dict, regions: list[dict], *, image_width: int, image_height: int) -> dict:
    """Compare label/value/unit on one OCR row; leave interpretation explicitly unchecked.

    Regions use top-left normalized coordinates relative to the image being inspected.
    The caller must select a crop before calling this function when ``claim.box`` is set.
    A repeated label or multiple values needs a narrower crop or host visual judgment.
    """
    claim = validate_claims([claim])[0]
    if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0
           for size in (image_width, image_height)):
        raise VidLensError("invalid_fact_image", "Image dimensions must be positive integers.")
    result = {
        "id": claim["id"], "stated": claim, "status": "unreadable",
        "literal_checks": {"label": "not_found", "value": "not_readable", "unit": "not_readable"},
        "observed_values": [], "matched_rows": [], "issues": [],
        "requires_visual_review": True, "requires_semantic_review": True,
        "semantic_status": "not_checked",
    }
    valid = []
    if not isinstance(regions, list) or len(regions) > 8192:
        result["issues"].append("ocr_layout_unavailable")
        return result
    for index, region in enumerate(regions):
        if (not isinstance(region, dict) or not isinstance(region.get("text"), str)
                or not region["text"].strip() or len(region["text"]) > 4096
                or not _box(region.get("box"))):
            if "malformed_ocr_regions_ignored" not in result["issues"]:
                result["issues"].append("malformed_ocr_regions_ignored")
            continue
        item = {"index": index, "text": region["text"], "box": list(region["box"])}
        confidence = region.get("confidence")
        if _finite(confidence) and 0 <= confidence <= 1:
            item["confidence"] = confidence
        valid.append(item)
    if not valid:
        result["issues"].append("ocr_layout_unavailable")
        return result
    if sum(len(region["text"]) for region in valid) > 100_000:
        result["issues"].append("ocr_layout_too_large_use_crop")
        return result

    pattern = _label_pattern(claim["label"])
    matches = []
    for row in _rows(valid, image_height):
        text = " ".join(_text(region["text"]) for region in row)
        for match in _label_matches(pattern, text):
            if len(row) > 64 or len(text) > 4096:
                result["issues"].append("matched_row_too_large_use_crop")
                return result
            observations = _observations(text[match.end():])
            matches.append({"text": text, "box": _union(row), "regions": row,
                            "observed_values": observations})
    result["matched_row_count"] = len(matches)
    # Bound duplicate-label diagnostics, retaining the true ambiguity count.
    result["matched_rows"] = matches[:6]
    result["observed_values"] = [item for row in matches[:6] for item in row["observed_values"]]
    if not matches:
        result["issues"].append("label_not_found")
        return result
    result["literal_checks"]["label"] = "matched"
    if len(matches) != 1:
        result["status"] = "ambiguous"
        result["literal_checks"].update(label="ambiguous", value="ambiguous", unit="ambiguous")
        result["issues"].append("multiple_label_matches")
        return result
    values = matches[0]["observed_values"]
    if not values:
        result["issues"].append("value_not_readable_on_label_row")
        return result
    if len(values) != 1:
        result["status"] = "ambiguous"
        result["literal_checks"].update(value="ambiguous", unit="ambiguous")
        result["issues"].append("multiple_numbers_on_label_row")
        return result

    observed = values[0]
    value_matches = parse_value(observed["value"]) == parse_value(claim["value"])
    result["literal_checks"]["value"] = "match" if value_matches else "mismatch"
    if not value_matches:
        result["issues"].append("value_mismatch")
    expected_unit = _unit(claim["unit"])
    if observed["unit"] is None:
        result["literal_checks"]["unit"] = "ambiguous"
        result["issues"].append("unit_binding_ambiguous")
    elif not observed["unit"] and expected_unit:
        result["issues"].append("unit_not_visible_on_label_row")
    elif observed["unit"] != expected_unit:
        result["literal_checks"]["unit"] = "mismatch"
        result["issues"].append("unit_mismatch")
    else:
        result["literal_checks"]["unit"] = "match"
    checks = result["literal_checks"]
    if "mismatch" in (checks["value"], checks["unit"]):
        result["status"] = "mismatch"
    elif checks["value"] == checks["unit"] == "match":
        result["status"] = "exact_literal"
    elif checks["unit"] == "ambiguous":
        result["status"] = "ambiguous"
    return result

"""Parse platform captions without losing their original video timestamps."""

from __future__ import annotations

import html
import json
import math
import re

from .models import Segment, VidLensError

_TIMING = re.compile(
    r"(?P<start>(?:\d+:)?\d{2}:\d{2}[.,]\d+)\s*-->\s*"
    r"(?P<end>(?:\d+:)?\d{2}:\d{2}[.,]\d+)"
)
_TAG = re.compile(
    r"</?(?:b|i|u|c|v|lang|ruby|rt|font)(?:\.[^>\s]+|\s[^>]*)?\s*>"
    r"|<(?:\d{2}:)?\d{2}:\d{2}\.\d+>",
    re.IGNORECASE,
)


def _clean(text: str, markup: bool = False) -> str:
    cleaned = _TAG.sub("", str(text)) if markup else str(text)
    return " ".join(html.unescape(cleaned).split()).strip()


def _seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    return sum(float(part) * 60**i for i, part in enumerate(reversed(parts)))


def _segment(
    start: object, end: object, text: object, source: str, language: str, markup: bool = False
) -> Segment | None:
    try:
        begin, finish = float(start), float(end)
    except (TypeError, ValueError):
        return None
    content = _clean(str(text or ""), markup=markup)
    if not content or not all(map(math.isfinite, (begin, finish))):
        return None
    if begin < 0 or finish <= begin:
        return None
    return Segment(begin, finish, content, source=source, language=language)


def _overlap_prefix(previous: str, current: str) -> int:
    """Find a substantial repeated suffix, preserving genuine repeated words."""
    if current.startswith(previous):
        boundary = len(previous)
        if boundary == len(current) or current[boundary].isspace():
            return boundary
        if re.search(r"[\u3400-\u9fff]", previous):
            return boundary
    for size in range(min(len(previous), len(current)), 3, -1):
        prefix = current[:size]
        if not previous.endswith(prefix):
            continue
        if re.search(r"[\u3400-\u9fff]", prefix):
            return size
        if len(prefix.split()) >= 2 and (size == len(current) or current[size].isspace()):
            return size
    return 0


def deduplicate_rolling(segments: list[Segment]) -> list[Segment]:
    """Deduplicate overlapping rolling windows, never distant repeated speech.

    New text retains the source cue's timestamp: a cue-level caption does not
    justify fabricating word-level timing for its newly appended words.
    """
    result: list[Segment] = []
    previous_raw: Segment | None = None
    for item in sorted(segments, key=lambda s: (s.start, s.end)):
        text = item.text
        if previous_raw and item.start < previous_raw.end - 0.001:
            if text == previous_raw.text:
                if result and result[-1].text == text:
                    result[-1].end = max(result[-1].end, item.end)
                previous_raw = item
                continue
            repeated = _overlap_prefix(previous_raw.text, text)
            if repeated:
                text = text[repeated:].strip()
        elif previous_raw and abs(item.start - previous_raw.end) <= 0.08:
            # Adjacent progressive windows are recognizable only when they grow.
            if text != previous_raw.text and text.startswith(previous_raw.text):
                repeated = _overlap_prefix(previous_raw.text, text)
                text = text[repeated:].strip() if repeated else text
        previous_raw = item
        if text:
            result.append(
                Segment(item.start, item.end, text, item.kind, item.source, item.language)
            )
    return result


def parse_subtitles(
    content: str, format: str, source: str = "manual", language: str = ""
) -> list[Segment]:
    """Read WebVTT, SRT, YouTube json3, Bilibili JSON, or timed segment JSON."""
    fmt = format.lower().lstrip(".")
    segments: list[Segment] = []
    try:
        if fmt in {"vtt", "webvtt", "srt"}:
            lines = content.lstrip("\ufeff").replace("\r\n", "\n").splitlines()
            index = 0
            while index < len(lines):
                timing = _TIMING.search(lines[index])
                index += 1
                if not timing:
                    continue
                body: list[str] = []
                while index < len(lines) and lines[index].strip():
                    if _TIMING.search(lines[index]):
                        break
                    body.append(lines[index])
                    index += 1
                item = _segment(
                    _seconds(timing["start"]),
                    _seconds(timing["end"]),
                    " ".join(body),
                    source,
                    language,
                    markup=True,
                )
                if item:
                    segments.append(item)
        elif fmt in {"json", "json3", "bilibili"}:
            payload = json.loads(content)
            if isinstance(payload, dict) and "events" in payload:
                for event in payload.get("events", []):
                    if not isinstance(event, dict) or not event.get("segs"):
                        continue
                    start = float(event.get("tStartMs", 0)) / 1000
                    end = start + float(event.get("dDurationMs", 0)) / 1000
                    text = "".join(str(part.get("utf8", "")) for part in event["segs"])
                    item = _segment(start, end, text, source, language)
                    if item:
                        segments.append(item)
            else:
                rows = payload
                if isinstance(payload, dict):
                    rows = payload.get("body", payload.get("segments", []))
                if not isinstance(rows, list):
                    raise ValueError("Expected a list of timed caption segments")
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    start = row.get("from", row.get("start", 0))
                    end = row.get("to", row.get("end"))
                    if end is None:
                        end = float(start) + float(row.get("duration", 0))
                    item = _segment(
                        start, end, row.get("content", row.get("text", "")), source, language
                    )
                    if item:
                        segments.append(item)
        else:
            raise VidLensError(
                "SUBTITLE_FORMAT",
                f"Unsupported subtitle format: {fmt}",
                "Use VTT, SRT, json3, or Bilibili JSON.",
            )
    except (ValueError, TypeError, KeyError) as exc:
        raise VidLensError(
            "SUBTITLE_PARSE",
            "The subtitle file is not valid timed caption data.",
            "Export UTF-8 VTT/SRT or JSON with start/end/text fields.",
        ) from exc
    segments.sort(key=lambda s: (s.start, s.end))
    if fmt != "srt":
        segments = deduplicate_rolling(segments)
    return segments

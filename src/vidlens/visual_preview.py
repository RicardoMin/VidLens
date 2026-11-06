"""Readable, numbered screen previews for the host AI's visual selection.

The preview is navigation material: it does not mark original frames delivered
or decide which diagram answers a question. Source images are never edited.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .models import VidLensError

_VERSION = 2
_PAGE_SIZE = 6
_TILE_WIDTH, _TILE_HEIGHT = 854, 480
_LABEL_HEIGHT, _GUTTER = 48, 12


def _valid_time(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _number(value):
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _representatives(rows, limit):
    """Keep temporal coverage without another question relevance ranking."""
    if len(rows) <= limit:
        return rows
    if limit == 1:
        center = (rows[0]["timestamp"] + rows[-1]["timestamp"]) / 2
        return [min(rows, key=lambda row: abs(row["timestamp"] - center))]
    # A narrow burst of many candidates should not consume the preview budget.
    start, end = rows[0]["timestamp"], rows[-1]["timestamp"]
    selected = {0, len(rows) - 1}
    for index in range(1, limit - 1):
        target = start + (end - start) * index / (limit - 1)
        choice = min((i for i in range(len(rows)) if i not in selected),
                     key=lambda i: (abs(rows[i]["timestamp"] - target), i))
        selected.add(choice)
    return [rows[i] for i in sorted(selected)]


def _cached(path, size):
    try:
        with Image.open(path) as image:
            if image.size != size or image.format != "JPEG":
                return False
            image.load()
        return True
    except (OSError, ValueError):
        return False


def _write_sheet(path, rows, size):
    sheet = Image.new("RGB", size, "#10151d")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=24)
    for index, row in enumerate(rows):
        left = _GUTTER + (index % 2) * (_TILE_WIDTH + _GUTTER)
        top = _GUTTER + (index // 2) * (_TILE_HEIGHT + _LABEL_HEIGHT + _GUTTER)
        with Image.open(io.BytesIO(row["_bytes"])) as source:
            image = ImageOps.contain(source.convert("RGB"), (_TILE_WIDTH, _TILE_HEIGHT),
                                     Image.Resampling.LANCZOS)
            sheet.paste(image, (left + (_TILE_WIDTH - image.width) // 2,
                               top + (_TILE_HEIGHT - image.height) // 2))
        label = f"#{row['candidate_index']:02d}  t={_number(row['timestamp'])}s"
        if abs(row["read_timestamp"] - row["timestamp"]) > 0.000001:
            label += f"  read={_number(row['read_timestamp'])}s"
        draw.text((left + 10, top + _TILE_HEIGHT + 10), label, fill="white", font=font)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".jpg", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            sheet.save(output, format="JPEG", quality=94, subsampling=0)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build_previews(service, frames, *, max_candidates=24):
    """Paginate query-selected frames into contact sheets and an exact tile map.

    Input rows need ``path`` and source ``timestamp``. At most 24 candidates are
    retained. Adjacent exact duplicates share one tile, with all occurrence
    times retained; a return to the same image after another state stays visible.
    ``candidate_index``, ``preview_index`` and ``tile_index`` are all one-based.
    Pass ``read_timestamp`` to exact-frame retrieval: a held frame's source PTS
    can precede the requested interval even while that frame is still visible.
    """
    if (isinstance(max_candidates, bool) or not isinstance(max_candidates, int)
            or not 1 <= max_candidates <= 24):
        raise VidLensError("invalid_frame_count", "Preview max_candidates must be from 1 to 24.")
    if not isinstance(frames, (list, tuple)):
        raise VidLensError("invalid_frames", "Preview frames must be a list of frame records.")
    for row in frames:
        if not isinstance(row, dict) or not _valid_time(row.get("timestamp")):
            raise VidLensError("invalid_timestamp", "Preview timestamps must be finite source seconds.")
        requested = row.get("requested_timestamp")
        if requested is not None and not _valid_time(requested):
            raise VidLensError("invalid_timestamp", "Requested preview times must be finite source seconds.")
    rows = []
    for frame in sorted(frames, key=lambda row: row["timestamp"]):
        try:
            path = Path(frame["path"]).resolve()
            body = path.read_bytes()
            with Image.open(io.BytesIO(body)) as image:
                image.verify()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise VidLensError("invalid_frame", "A preview source image could not be read.") from exc
        digest = hashlib.sha256(body).hexdigest()
        occurrence = {"timestamp": frame["timestamp"],
                      "requested_timestamp": frame.get("requested_timestamp")}
        if rows and rows[-1]["_digest"] == digest:
            rows[-1]["occurrences"].append(occurrence)
            continue
        row = {key: frame[key] for key in (
            "timestamp", "requested_timestamp", "visible_until", "source_candidate_cursor",
            "source_timestamp", "evidence_id", "url",
        ) if key in frame}
        rows.append({**row, "path": str(path), "occurrences": [occurrence],
                     "read_timestamp": (frame["timestamp"] if frame.get("requested_timestamp") is None
                                        else frame["requested_timestamp"]),
                     "_bytes": body, "_digest": digest})
    available_count = len(rows)
    rows = _representatives(rows, max_candidates)
    if not rows:
        return {"previews": [], "candidates": [], "candidate_count": 0,
                "available_candidate_count": 0, "truncated": False}
    directory = Path(service.home) / "visual" / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    previews, candidates = [], []
    for index, row in enumerate(rows):
        row.update(candidate_index=index + 1, preview_index=index // _PAGE_SIZE + 1,
                   tile_index=index % _PAGE_SIZE + 1)
    for offset in range(0, len(rows), _PAGE_SIZE):
        page = rows[offset:offset + _PAGE_SIZE]
        columns, lines = min(2, len(page)), math.ceil(len(page) / 2)
        size = (_GUTTER + columns * (_TILE_WIDTH + _GUTTER),
                _GUTTER + lines * (_TILE_HEIGHT + _LABEL_HEIGHT + _GUTTER))
        identity = [_VERSION, size, [(row["_digest"], row["timestamp"],
                                      row.get("requested_timestamp"), row["candidate_index"])
                                     for row in page]]
        key = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        destination = directory / f"preview-{key}.jpg"
        if not _cached(destination, size):
            _write_sheet(destination, page, size)
        entries = [{key: value for key, value in row.items() if not key.startswith("_")}
                   for row in page]
        candidates.extend(entries)
        previews.append({"path": str(destination.resolve()), "width": size[0], "height": size[1],
                         "preview_index": page[0]["preview_index"], "candidates": entries})
    return {"previews": previews, "candidates": candidates, "candidate_count": len(candidates),
            "available_candidate_count": available_count, "truncated": len(rows) < available_count}

"""Lossless, source-linked paragraph transport for short subtitle/ASR cues.

Paragraphs group nearby cues for reading; they do not assert semantic boundaries.
Every reference retains the original evidence ID and character range, so clients
can cite existing evidence and delivery can be verified against the source text.
"""

from __future__ import annotations

from .models import VidLensError


def validate_representation(representation):
    if not isinstance(representation, str) or representation not in {"verbose", "compact"}:
        raise VidLensError("invalid_representation", "Use compact or verbose evidence.")


def compact_paragraphs(items, *, max_seconds=45, max_characters=600):
    """Return shared tracks and paragraphs without rewriting any source characters."""
    paragraphs, tracks, track_ids = [], [], {}
    for item in items:
        identity = (item["kind"], item["source"], item.get("language", ""))
        if identity not in track_ids:
            track_ids[identity] = len(tracks)
            tracks.append(dict(zip(("kind", "source", "language"), identity)))
        track = track_ids[identity]
        previous = paragraphs[-1] if paragraphs else None
        merge = (
            previous is not None
            and previous["track"] == track
            and 0 <= item["start"] - previous["end"] <= 2
            and item["end"] - previous["start"] <= max_seconds
            and len(previous["text"]) + 1 + len(item["text"]) <= max_characters
        )
        reference = [item["id"], item.get("text_offset", 0), len(item["text"])]
        if merge:
            previous["text"] += "\n" + item["text"]
            previous["end"] = item["end"]
            previous["refs"].append(reference)
        else:
            paragraphs.append({
                "start": item["start"], "end": item["end"], "track": track,
                "text": item["text"], "refs": [reference],
            })
    return tracks, paragraphs


def expand_evidence(payload):
    """Recover exact delivered character ranges, including after JSON transport."""
    if payload.get("representation") != "compact":
        return payload["evidence"]
    result = []
    for paragraph in payload["evidence"]:
        text, refs = paragraph.get("text"), paragraph.get("refs")
        if not isinstance(text, str) or not isinstance(refs, list) or not refs:
            raise VidLensError("invalid_compact_evidence", "A paragraph needs text and references.")
        position = 0
        for index, ref in enumerate(refs):
            if (
                not isinstance(ref, list) or len(ref) != 3 or not isinstance(ref[0], str)
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in ref[1:])
            ):
                raise VidLensError("invalid_compact_evidence", "Invalid evidence character range.")
            evidence_id, offset, length = ref
            if index:
                if text[position:position + 1] != "\n":
                    raise VidLensError("invalid_compact_evidence", "Paragraph separators changed.")
                position += 1
            if position + length > len(text):
                raise VidLensError("invalid_compact_evidence", "Paragraph reference exceeds its text.")
            result.append({
                "id": evidence_id, "text_offset": offset,
                "text": text[position:position + length],
            })
            position += length
        if position != len(text):
            raise VidLensError("invalid_compact_evidence", "Paragraph has unreferenced text.")
    return result

"""Budget the serialized evidence actually sent to the caller."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

from .models import VidLensError


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


# The cache key and content checksum published with tiktoken's o200k_base registry.
_CACHE_FILE = "fb374d419588a4632f3f557e76b4b70aebbca790"
_CHECKSUM = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
_ENCODING_LOCK = Lock()


@lru_cache(maxsize=4)
def _verified_encoder(path: Path, mtime_ns: int):
    """Load only a verified local vocabulary. mtime invalidates a replaced cache."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != _CHECKSUM:
        return None
    import tiktoken

    with _ENCODING_LOCK:
        previous = os.environ.get("TIKTOKEN_CACHE_DIR")
        os.environ["TIKTOKEN_CACHE_DIR"] = str(path.parent)
        try:
            return tiktoken.get_encoding("o200k_base")
        finally:
            if previous is None:
                del os.environ["TIKTOKEN_CACHE_DIR"]
            else:
                os.environ["TIKTOKEN_CACHE_DIR"] = previous


class TokenBudget:
    def __init__(self, cache_dir: Path):
        path = cache_dir / "tokenizer" / _CACHE_FILE
        try:
            self.encoding = _verified_encoder(path, path.stat().st_mtime_ns)
        except (OSError, ImportError, ValueError):
            self.encoding = None
        self.method = "o200k_base_estimate" if self.encoding else "utf8_bytes_upper_bound"

    def count(self, text: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(text, disallowed_special=()))
        return len(text.encode("utf-8"))

    def truncate(self, text: str, limit: int) -> str:
        if self.count(text) <= limit:
            return text
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.count(text[:mid] + "…") <= limit:
                low = mid
            else:
                high = mid - 1
        return text[:low] + "…" if low else ""

    def recount(self, payload: dict) -> dict:
        accounting = payload["accounting"]
        for _ in range(5):
            size = self.count(compact_json(payload))
            if accounting["text_tokens_estimate"] == size:
                break
            accounting["text_tokens_estimate"] = size
        if self.count(compact_json(payload)) > accounting["budget"]:
            raise VidLensError(
                "budget_too_small", "Increase max_tokens to fit the result metadata."
            )
        return payload

    def finish(
        self, payload: dict, budget: int, items_key: str, total_available: int | None = None
    ) -> dict:
        """Find a fitting prefix or timeline sample in logarithmically many size checks."""
        payload["accounting"] = {
            "text_tokens_estimate": budget,
            "budget": budget,
            "method": self.method,
            "excludes": "images, model reasoning, transport envelope",
        }
        payload.setdefault("truncated", False)
        if total_available is not None:
            payload["available_items"] = total_available

        def fits() -> bool:
            return self.count(compact_json(payload)) <= budget

        if fits():
            return self.recount(payload)

        if len(payload.get("query", "")) > 80:
            payload["query"] = self.truncate(payload["query"][:80], 60)
            payload["query_truncated"] = True
        if payload.get("title"):
            payload["title"] = self.truncate(payload["title"], max(50, budget // 8))
        payload.pop("next_actions", None)
        if items_key != "outline" and len(payload.get("outline", [])) > 3:
            outline = payload["outline"]
            payload["outline"] = [outline[0], outline[len(outline) // 2], outline[-1]]
            payload["outline_sampled"] = True
        if fits():
            return self.recount(payload)

        original = payload[items_key]
        spread = items_key == "outline" or payload.get("mode") in (
            "complete_transcript",
            "sampled_transcript",
            "partial_transcript",
            "visual_evidence",
        )
        payload["truncated"] = True
        if payload.get("mode") == "complete_transcript":
            payload["mode"] = "sampled_transcript"

        # Start with small candidates. Long transcripts usually fit only a small prefix.
        def select(n):
            if spread and n > 1:
                return [original[round(i * (len(original) - 1) / (n - 1))] for i in range(n)]
            return original[:n]

        low, high = 0, min(1, len(original))
        while high:
            payload[items_key] = select(high)
            if not fits():
                break
            low = high
            if low == len(original):
                return self.recount(payload)
            high = min(len(original), high * 2)
        while low + 1 < high:
            mid = (low + high) // 2
            payload[items_key] = select(mid)
            if fits():
                low = mid
            else:
                high = mid
        payload[items_key] = select(low)
        if not low and original and isinstance(original[0].get("text"), str):
            item = dict(original[0], excerpt_truncated=True)
            payload[items_key] = [item]
            text = original[0]["text"]
            left, right = 0, len(text)
            while left < right:
                mid = (left + right + 1) // 2
                item["text"] = text[:mid] + "…"
                if fits():
                    left = mid
                else:
                    right = mid - 1
            if left:
                item["text"] = text[:left] + "…"
            else:
                payload[items_key] = []
        return self.recount(payload)

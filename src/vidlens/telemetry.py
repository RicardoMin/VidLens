"""Local operation timings and transport estimates; never model billing."""

from __future__ import annotations

import functools
import json
import math
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime

from filelock import FileLock


def _details(result):
    """Observe a return value without imposing a schema on the operation itself."""
    if not isinstance(result, dict):
        return {}
    event = {"video_id": result.get("video_id"), "cache_hit": result.get("cache_hit")}
    accounting = result.get("accounting")
    if isinstance(accounting, dict):
        event["text_tokens_estimate"] = accounting.get("text_tokens_estimate")
        event["estimate_method"] = accounting.get("method")
    frames = result.get("frames")
    if isinstance(frames, list):
        event["returned_frames"] = len(frames)
        event["frame_sizes"] = [[f.get("width"), f.get("height")]
                                for f in frames if isinstance(f, dict)]
    return event


def _record(home, event):
    root = home / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root / "operations.lock"), timeout=.25)
    with (lock.acquire(poll_interval=.005),
          (root / "operations.jsonl").open("a", encoding="utf-8") as out):
        out.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def measured(function):
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        started = time.perf_counter()
        event = {"id": uuid.uuid4().hex, "operation": function.__name__,
                 "run_id": self.run_id, "started_at": datetime.now(UTC).isoformat()}
        try:
            result = function(self, *args, **kwargs)
            event["status"] = "ok"
            try:
                event.update(_details(result))
            except Exception:  # noqa: BLE001, S110 -- best-effort observer cannot change the result
                pass
            return result
        except Exception as exc:
            event.update(status="error", error_code=getattr(exc, "code", type(exc).__name__))
            raise
        finally:
            event["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            # Observability must not turn a successful read into a failed operation.
            try:
                _record(self.home, event)
            except Exception:  # noqa: BLE001, S110 -- logging must never replace the original error
                pass
    return wrapped


def operation_metrics(home, run_id=None, limit=100):
    from .models import VidLensError

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
        raise VidLensError("invalid_limit", "Use 1..10000 operations.")
    events = []
    path = home / "metrics" / "operations.jsonl"
    if path.exists():
        # Read linewise; a concurrently incomplete final append is skipped.
        from collections import deque
        tail = deque(maxlen=limit)
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if (not isinstance(event, dict) or not isinstance(event.get("operation"), str)
                        or not isinstance(event.get("status"), str)
                        or event.get("status") not in {"ok", "error"}
                        or isinstance(event.get("elapsed_seconds"), bool)
                        or not isinstance(event.get("elapsed_seconds"), (int, float))
                        or not math.isfinite(event["elapsed_seconds"])
                        or event["elapsed_seconds"] < 0):
                    continue
                if run_id is None or event.get("run_id") == run_id:
                    tail.append(event)
        events = list(tail)
    totals = defaultdict(lambda: {"calls": 0, "errors": 0, "elapsed_seconds": 0.0})
    for event in events:
        item = totals[event["operation"]]
        item["calls"] += 1
        item["errors"] += event["status"] != "ok"
        item["elapsed_seconds"] = round(item["elapsed_seconds"] + event["elapsed_seconds"], 6)
    return {"run_id": run_id, "operations": events, "by_operation": dict(totals),
            "scope": "Local operation wall times; nested/parallel durations must not be summed as task latency.",
            "token_scope": "Serialized text estimates only; excludes model reasoning, image tokens, history and billing."}

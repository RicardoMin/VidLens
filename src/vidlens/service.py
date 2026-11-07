"""AI-facing orchestration for video acquisition, reading and persistent learning."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import threading
import time
import uuid
from pathlib import Path

from filelock import FileLock

from . import __version__
from .budget import TokenBudget, compact_json
from .media_cache import MediaCache
from .models import MediaAsset, Segment, Video, VidLensError
from .reading import EvidenceReader, validate_budget, validate_interval
from .retrieval import source_link
from .store import Store, frame_source_revision, union_intervals
from .telemetry import measured


class VideoService:
    def __init__(self, home: Path | str | None = None):
        self.home = (
            Path(home or os.environ.get("VIDLENS_HOME", Path.home() / ".cache" / "vidlens"))
            .expanduser()
            .resolve()
        )
        self.store = Store(self.home)
        self.run_id = os.environ.get("VIDLENS_RUN_ID") or uuid.uuid4().hex
        self.budget = TokenBudget(self.home)
        self.media = MediaCache(self.store)
        self.reader = EvidenceReader(self.store, self.budget)
        self._visual_lock = threading.RLock()
        self._locks = self.home / "locks"
        self._locks.mkdir(exist_ok=True)
        self._provider = None
        self._visual = None
        self._learning = None
        self._curriculum = None

    @property
    def curriculum(self):
        if self._curriculum is None:
            from .curriculum import Curriculum

            self._curriculum = Curriculum(self.store, self.learning.study)
        return self._curriculum

    @property
    def learning(self):
        if self._learning is None:
            from .learning import LearningReader

            self._learning = LearningReader(self)
        return self._learning

    @property
    def provider(self):
        if self._provider is None:
            from .sources import SourceProvider

            cookie = os.environ.get("VIDLENS_COOKIE_FILE")
            self._provider = SourceProvider(
                self.home / "sources", cookie_file=Path(cookie) if cookie else None
            )
        return self._provider

    @property
    def visual(self):
        if self._visual is None:
            from .visual import VisualReader

            self._visual = VisualReader(self.home / "visual")
        return self._visual

    def open_video(self, url, languages=("zh", "en"), refresh=False, max_tokens=2500):
        validate_budget(max_tokens)
        started = time.perf_counter()
        video, hit = self._ensure_video(url, languages, refresh)
        result = self.reader.overview(video)
        result["cache_hit"] = hit
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return self.budget.finish(result, max_tokens, "outline", len(result["outline"]))

    def _ensure_video(self, url, languages, refresh):
        from .sources import normalize_url

        if (
            not languages
            or isinstance(languages, str)
            or any(not isinstance(x, str) or not x.strip() for x in languages)
        ):
            raise VidLensError("invalid_languages", "Provide one or more language codes.")
        if not isinstance(url, str):
            raise VidLensError("invalid_url", "Supply a YouTube or Bilibili video URL.")
        raw = url.strip()
        preferences = list(languages)

        def lookup(key):
            cached = self.store.by_url(key)
            if cached and cached.metadata.get("language_preferences", preferences) == preferences:
                return cached
            return None

        if not refresh and (cached := lookup(raw)):
            return cached, True
        normalized = normalize_url(raw)
        with self._file_lock("open", normalized):
            if not refresh and (cached := lookup(normalized)):
                self.store.remember_alias(raw, cached.id)
                return cached, True
            acquired = self.provider.acquire(normalized, languages=tuple(preferences))
            self._validate_video(acquired.video, acquired.segments)
            acquired.video.metadata.update(
                language_preferences=preferences, revision=uuid.uuid4().hex
            )
            with self._file_lock("content", acquired.video.id):
                self.store.save(
                    acquired.video,
                    acquired.segments,
                    acquired.warnings,
                    invalidate_derived=True,
                    aliases=(raw, normalized),
                )
            return acquired.video, False

    def _file_lock(self, scope, value):
        key = hashlib.sha256(value.encode()).hexdigest()[:24]
        return FileLock(str(self._locks / f"{scope}-{key}.lock"), timeout=1500)

    @measured
    def read_video(
        self,
        url,
        question="",
        max_tokens=4000,
        languages=("zh", "en"),
        refresh=False,
        intent="auto",
        prepare=False,
    ):
        """One-call evidence entry point: no intermediate outline/search payload duplication."""
        validate_budget(max_tokens)
        if not isinstance(prepare, bool):
            raise VidLensError("invalid_prepare", "prepare must be a boolean.")
        if not isinstance(question, str) or len(question) > 2000:
            raise VidLensError("invalid_query", "question must be text up to 2000 characters.")
        from .learning import reading_intent

        selected_intent = reading_intent(question, intent)
        started = time.perf_counter()
        video, hit = self._ensure_video(url, languages, refresh)
        if prepare and selected_intent == "summary":
            scope = self.reader._speech_scope(video, self.store.segments(video.id))
            if scope == "no_transcript" or (isinstance(scope, dict)
                                            and not scope["whole_video_processed"]):
                suggested = self.transcription_options().get("suggested")
                if suggested:
                    self.transcribe(video.id, language=languages[0], **suggested)
                    video = self.store.get(video.id)
        if selected_intent == "summary":
            return self.learning.summary(video, question, max_tokens, hit, started)
        if max_tokens >= 1800:
            units = self.query_knowledge(video.id, question, limit=3,
                                         max_tokens=min(2500, max_tokens // 2))
            if units["units"]:
                reserved = self.budget.count(compact_json({"knowledge_units": units})) + 32
                if reserved < max_tokens - 768:
                    result = self.reader.read(video, question, max_tokens - reserved, hit, started)
                    result["knowledge_units"] = units
                    result["accounting"]["budget"] = max_tokens
                    return self.budget.recount(result)
        knowledge = self.learning.search(video.id, question, max_tokens=max_tokens, limit=3)
        if knowledge["matches"]:
            matches = knowledge["matches"]
            for match in matches:
                for field in ("summary", "excerpt", "explanation", "text"):
                    if field in match:
                        match[field] = self.budget.truncate(match[field], 180)
                if match.get("match_scope") != "relation_path":
                    match.pop("relations", None)
            reserved = self.budget.count(compact_json({"knowledge": matches})) + 32
            if reserved < max_tokens - 768:
                result = self.reader.read(video, question, max_tokens - reserved, hit, started)
                result["knowledge"] = matches
                result["accounting"]["budget"] = max_tokens
                return self.budget.recount(result)
        return self.reader.read(video, question, max_tokens, hit, started)

    def study_status(self, video_id, max_tokens=4000, cursor=0):
        return self.learning.status(video_id, max_tokens=max_tokens, cursor=cursor)

    @measured
    def read_section(
        self,
        video_id,
        section_id,
        max_tokens=6000,
        cursor=None,
        include_frames=True,
        frame_cursor=None,
        frame_limit=4,
        frame_view="navigation",
        *,
        question="",
        visual_queries=None,
        record_delivery=True,
        representation="verbose",
    ):
        return self.learning.section(
            video_id,
            section_id,
            max_tokens=max_tokens,
            cursor=cursor,
            include_frames=include_frames,
            frame_cursor=frame_cursor,
            frame_limit=frame_limit,
            frame_view=frame_view,
            question=question,
            visual_queries=visual_queries,
            record_delivery=record_delivery,
            representation=representation,
        )

    def record_section_delivery(self, payload, frames=None):
        return self.learning.record_delivery(payload, frames=frames)

    @measured
    def scan_video(self, video_id, cursor=0, limit=100, refresh=False, view="navigation",
                   diagnostics=True):
        return self.learning.scan(video_id, cursor=cursor, limit=limit, refresh=refresh,
                                  view=view, diagnostics=diagnostics)

    @measured
    def save_knowledge(self, video_id, revision, units, replace=False):
        with self._file_lock("content", video_id):
            return self.curriculum.save(video_id, revision, units, replace=replace)

    def query_knowledge(self, video_id, query="", roles=None, limit=12, max_tokens=6000):
        return self.curriculum.query(video_id, query, roles, limit, max_tokens)

    def article_plan(self, video_id, revision, plan=None):
        with self._file_lock("content", video_id):
            return self.curriculum.plan(video_id, revision, plan)

    def audit_article(self, video_id, revision, draft):
        return self.curriculum.audit(video_id, revision, draft)

    @measured
    def export_article(self, video_id, revision, draft, output):
        from .article import export_article

        return export_article(self, video_id, revision, draft, output)

    def operation_metrics(self, run_id=None, limit=100):
        from .telemetry import operation_metrics

        return operation_metrics(self.home, run_id, limit)

    def save_note(self, video_id, section_id, revision, note):
        with self._file_lock("content", video_id):
            return self.learning.study.save_note(video_id, section_id, revision, note)

    def revise_sections(self, video_id, revision, sections):
        with self._file_lock("content", video_id):
            return self.learning.study.revise_sections(video_id, revision, sections)

    def add_reference(self, video_id, section_id, revision, reference):
        with self._file_lock("content", video_id):
            return self.learning.references.add(video_id, section_id, revision, reference)

    def search_knowledge(self, video_id, query, max_tokens=3000, limit=6):
        return self.learning.search(video_id, query, max_tokens=max_tokens, limit=limit)

    def export_study(self, video_id):
        with self._file_lock("content", video_id):
            return self.learning.export(video_id)

    def overview(self, video_id, max_tokens=2500):
        validate_budget(max_tokens)
        result = self.reader.overview(self.store.get(video_id))
        return self.budget.finish(result, max_tokens, "outline", len(result["outline"]))

    def search(self, video_id, query, max_tokens=3000, limit=6):
        return self.reader.search(video_id, query, max_tokens, limit)

    @measured
    def read_interval(self, video_id, start, end, max_tokens=4000, cursor=None,
                      representation="verbose"):
        return self.reader.read_interval(video_id, start, end, max_tokens, cursor,
                                         representation=representation)

    def _asset(self, video: Video, start: float, end: float, audio_only=False) -> MediaAsset:
        return self.media.get(video, start, end, audio_only, lambda: self.provider)

    @measured
    def read_frames(
        self,
        video_id,
        start,
        end,
        max_frames=4,
        strategy="changes",
        ocr=True,
        *,
        question="",
        visual_queries=None,
        preview=False,
        timestamps=None,
        record_delivery=True,
    ):
        from .visual_query import QuestionFrames, validate_visual_query

        question, visual_queries = validate_visual_query(question, visual_queries)
        if not isinstance(preview, bool) or (preview and (not question or timestamps is not None)):
            raise VidLensError("invalid_preview", "Preview requires a question and no exact timestamps.")
        if timestamps is not None and (
            not isinstance(timestamps, list) or not 1 <= len(timestamps) <= 12
            or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
                   for t in timestamps)
        ):
            raise VidLensError("invalid_timestamps", "Provide 1..12 finite frame timestamps.")
        if timestamps is not None and (question or visual_queries):
            raise VidLensError("invalid_timestamps", "Use exact timestamps after question selection, separately.")
        if question and not ocr:
            raise VidLensError("ocr_required", "Question-guided selection needs screen text; enable OCR.")
        if (
            isinstance(max_frames, bool)
            or not isinstance(max_frames, int)
            or not 1 <= max_frames <= 12
        ):
            raise VidLensError("invalid_frame_count", "max_frames must be from 1 to 12.")
        if strategy not in ("changes", "uniform"):
            raise VidLensError("invalid_strategy", "strategy must be changes or uniform.")
        started = time.perf_counter()
        with self._file_lock("content", video_id), self._visual_lock:
            video = self.store.get(video_id)
            if video.metadata.get("local_media_path"):
                self.learning.study.plan(video_id)
            source_revision = self._frame_revision(video)
            speech_revision = self._speech_revision(video_id) if question else None
            start, end = validate_interval(video, start, end, max_duration=600)
            if timestamps is not None and any(t < start or t >= end for t in timestamps):
                raise VidLensError("invalid_timestamps", "Frame timestamps must be inside the requested interval.")
            asset = self._asset(video, start, end)
            selection = None
            if question:
                items, selection, decoded = QuestionFrames(self).read(
                    video, asset, start, end, question, visual_queries, max_frames, preview=preview
                )
                frames = None
            elif timestamps is not None:
                frames = self.visual.read_at(asset, timestamps, ocr=ocr)
            else:
                frames = self.visual.read(
                    asset, start, end, max_frames=max_frames, strategy=strategy, ocr=ocr
                )
            if source_revision != self._frame_revision(video):
                raise VidLensError("stale_revision", "The local video changed during decoding.")
            if question:
                if speech_revision != self._speech_revision(video_id):
                    raise VidLensError("stale_revision", "The speech changed during screen selection; retry.")
                self.store.save_frames(video_id, decoded)
            if frames is not None:
                self.store.save_frames(video_id, frames)
                items = [
                    f.to_dict()
                    | {"url": source_link(video, f.timestamp), "evidence_id": f"frame:{f.timestamp}"}
                    for f in frames
                ]
            preview_payload = {}
            if preview:
                from .visual_preview import build_previews

                preview_payload = build_previews(self, items)
                items = []
                selection["selection"] = "question_preview"
            for item in items:
                item["ocr"] = self.budget.truncate(item.get("ocr", ""), 1000)
            if record_delivery:
                self.store.delivered(video_id, items, "frame")
            warnings = (selection["warnings"] if selection is not None
                        else list(getattr(self.visual, "last_warnings", [])))
        return {
            "video_id": video_id,
            "source_revision": source_revision,
            **({"speech_revision": speech_revision} if question else {}),
            "start": start,
            "end": end,
            "frames": items,
            "warnings": warnings,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "visual_scope": "sampled_frames",
            **preview_payload,
            **({"visual": selection} if selection is not None else {}),
            "accounting": {
                "image_count": len(items) + len(preview_payload.get("previews", [])),
                "unique_image_count": (sum(f.get("duplicate_of") is None for f in items)
                                       + len(preview_payload.get("previews", []))),
                **({"preview_image_count": len(preview_payload["previews"]),
                    "candidate_thumbnail_count": len(preview_payload["candidates"]),
                    "full_frame_image_count": 0} if preview else {}),
                "image_tokens": "model-dependent; not included in text budgets",
            },
        }

    @staticmethod
    def _frame_revision(video):
        return frame_source_revision(video)

    def _speech_revision(self, video_id):
        speech = [(s.id, s.start, s.end, s.text, s.source, s.language)
                  for s in self.store.segments(video_id) if s.kind != "ocr"]
        return hashlib.sha256(compact_json(speech).encode()).hexdigest()[:24]

    def record_frame_delivery(self, payload, frames=None):
        """Commit only images successfully encoded for the client, against the source read."""
        video_id = payload["video_id"]
        with self._file_lock("content", video_id):
            if payload.get("source_revision") != self._frame_revision(self.store.get(video_id)):
                raise VidLensError(
                    "stale_revision", "The video changed while preparing its images."
                )
            if (payload.get("speech_revision") is not None
                    and payload["speech_revision"] != self._speech_revision(video_id)):
                raise VidLensError("stale_revision", "The speech changed while preparing the images.")
            self.store.delivered(video_id, payload["frames"] if frames is None else frames, "frame")
        return payload

    def read_frame_times(self, video_id, timestamps, *, record_delivery=True):
        """Retrieve exact scan candidates, sharing media downloads within nearby groups."""
        if (
            not isinstance(timestamps, list)
            or not 1 <= len(timestamps) <= 12
            or any(
                isinstance(stamp, bool)
                or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp)
                or stamp < 0
                for stamp in timestamps
            )
        ):
            raise VidLensError("invalid_timestamps", "Provide 1..12 finite nonnegative timestamps.")
        with self._file_lock("content", video_id), self._visual_lock:
            video = self.store.get(video_id)
            if video.metadata.get("local_media_path"):
                self.learning.study.plan(video_id)
            source_revision = self._frame_revision(video)
            if video.duration <= 0 or max(timestamps) >= video.duration:
                raise VidLensError("invalid_timestamps", "Timestamps must fall inside the video.")
            groups = []
            for stamp in sorted(set(timestamps)):
                if not groups or stamp - groups[-1][0] > 240:
                    groups.append([])
                groups[-1].append(stamp)
            frames, frame_warnings = [], []
            full_asset = None
            if video.duration <= 1800:
                full_asset = self.media.optional_full_video(video, lambda: self.provider)
            for group in groups:
                asset = full_asset or self._asset(
                    video, max(0, group[0] - 1), min(video.duration, group[-1] + 1)
                )
                selected = self.visual.read_at(asset, group, ocr=True)
                notices = list(getattr(self.visual, "last_warnings", []))
                frames.extend(selected)
                frame_warnings.extend([notices] * len(selected))
            if source_revision != self._frame_revision(video):
                raise VidLensError("stale_revision", "The local video changed during decoding.")
            self.store.save_frames(video_id, frames)
            items = [
                frame.to_dict()
                | {
                    "url": source_link(video, frame.timestamp),
                    "evidence_id": f"frame:{frame.timestamp}",
                }
                | ({"warnings": notices} if notices else {})
                for frame, notices in zip(frames, frame_warnings)
            ]
            if record_delivery:
                self.store.delivered(video_id, items, "frame")
        return items

    def crop_frame(self, video_id, timestamp, box):
        self.store.get(video_id)
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise VidLensError(
                "invalid_timestamp", "timestamp must be finite seconds from a returned frame."
            )
        with self._file_lock("content", video_id), self._visual_lock:
            # A local media file can change without a remote refresh. Synchronize
            # its source identity before retrieving an older cached screenshot.
            self.learning.study.plan(video_id)
            video = self.store.get(video_id)
            source_revision = self._frame_revision(video)
            frame = self.store.frame_at(video_id, timestamp)
            if frame is None:
                raise VidLensError(
                    "frame_not_found", "Read frames first, then use a returned timestamp."
                )
            cropped = self.visual.crop(frame, tuple(box))
            if source_revision != self._frame_revision(self.store.get(video_id)):
                raise VidLensError("stale_revision", "The local video changed during cropping.")
            return {
                "video_id": video_id,
                "frame": cropped.to_dict(include_layout=True)
                | {
                    "evidence_id": f"frame:{frame.timestamp}",
                    "url": source_link(self.store.get(video_id), frame.timestamp),
                },
                "crop": list(box),
                "warnings": list(getattr(self.visual, "last_warnings", [])),
            }

    @measured
    def read_detail(self, video_id, timestamp, boxes=None, *, upgrade_source=False, max_tiles=12,
                    record_delivery=True):
        """Read source-pixel regions with their surrounding figure and provenance."""
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0):
            raise VidLensError("invalid_timestamp", "Use the actual timestamp of a cached frame.")
        if not isinstance(upgrade_source, bool):
            raise VidLensError("invalid_detail", "upgrade_source must be boolean.")
        if (isinstance(max_tiles, bool) or not isinstance(max_tiles, int)
                or not 1 <= max_tiles <= 12):
            raise VidLensError("invalid_tile_count", "max_tiles must be from 1 to 12.")
        if boxes is not None and (
            not isinstance(boxes, list) or not 1 <= len(boxes) <= 6
            or any(not isinstance(box, (list, tuple)) or len(box) != 4
                   or any(isinstance(v, bool) or not isinstance(v, (int, float))
                          or not math.isfinite(v) for v in box)
                   or box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0
                   or box[0] + box[2] > 1 or box[1] + box[3] > 1 for box in boxes)
        ):
            raise VidLensError("invalid_crop", "Provide 1..6 normalized x/y/width/height boxes.")
        with self._file_lock("content", video_id), self._visual_lock:
            self.learning.study.plan(video_id)
            video = self.store.get(video_id)
            revision = self._frame_revision(video)
            frame = self.store.frame_at(video_id, timestamp)
            if frame is None:
                raise VidLensError("frame_not_found", "Read the original frame before its details.")
            prior_size = [frame.width, frame.height]
            acquisition = {"requested": upgrade_source, "status": "cached_source"}
            notices = []
            if upgrade_source:
                read_time = frame.requested_timestamp
                if read_time is None or not frame.timestamp <= read_time < video.duration:
                    read_time = frame.timestamp
                left, right = max(0, read_time - 1), min(video.duration, read_time + 1)
                try:
                    asset = self.media.get(video, left, right, False, lambda: self.provider,
                                           max_height=2160)
                    decoded = self.visual.read_at(asset, [read_time], ocr=False, native=True)[0]
                    # Normalized regions only transfer between compatible frame geometry.
                    # Permit the pixel rounding of ordinary resolution variants, but not
                    # a different aspect ratio or a loss of detail along either axis.
                    old_source_readable = True
                    try:
                        old_source = self.visual.source_info(frame)["size"]
                    except VidLensError:
                        old_source = [frame.width, frame.height]
                        old_source_readable = False
                    compatible = abs(decoded.width * old_source[1]
                                     - decoded.height * old_source[0]) <= 2 * max(
                                         *old_source, decoded.width, decoded.height)
                    if compatible and (not old_source_readable or (
                        decoded.width >= old_source[0] and decoded.height >= old_source[1]
                    )):
                        frame = decoded
                        acquisition.update(status=("native_decode" if old_source_readable
                                                   else "native_decode_recovered"),
                                           requested_max_height=2160)
                    else:
                        acquisition.update(status=("kept_larger_cached_source" if compatible
                                                   else "kept_cached_source_geometry"),
                                           requested_max_height=2160,
                                           available_size=[decoded.width, decoded.height])
                except VidLensError as exc:
                    acquisition.update(status="cached_source_fallback", error=exc.to_dict())
                    notices.append("Higher-resolution source was unavailable; using cached pixels.")
            details = self.visual.detail(frame, boxes, max_tiles=max_tiles)
            if revision != self._frame_revision(self.store.get(video_id)):
                raise VidLensError("stale_revision", "The video changed while reading details.")
            self.store.save_frames(video_id, [frame])
            source = {"evidence_id": f"frame:{frame.timestamp}",
                      "url": source_link(video, frame.timestamp)}
            pictures = [frame.to_dict() | source | {"role": "context"}]
            for index, (tile, region_index) in enumerate(zip(
                details["tiles"], details["tile_regions"]
            ), start=1):
                pictures.append(tile.to_dict(include_layout=True) | source | {
                    "role": "detail", "tile_index": index, "region_index": region_index,
                    "sha256": hashlib.sha256(tile.path.read_bytes()).hexdigest(),
                })
            result = {
                "video_id": video_id, "source_revision": revision,
                "requested_timestamp": timestamp, "timestamp": frame.timestamp,
                "frames": pictures,
                "detail": {k: v for k, v in details.items() if k != "tiles"},
                "acquisition": acquisition | {"previous_frame_size": prior_size},
                "warnings": notices + list(getattr(self.visual, "last_warnings", [])),
                "reading_hint": ("Use context for headers, axes, run identity and units; "
                                 "use detail tiles for glyphs. OCR remains candidate text. "
                                 "If a digit is unresolved, seek another source frame or "
                                 "keep only supported precision."),
            }
            if record_delivery:
                self.store.delivered(video_id, pictures[:1], "frame")
            return result

    def record_detail_delivery(self, payload, frames=None):
        with self._file_lock("content", payload["video_id"]):
            self.learning.study.plan(payload["video_id"])
            video = self.store.get(payload["video_id"])
            if payload["source_revision"] != self._frame_revision(video):
                raise VidLensError("stale_revision", "The video changed while encoding detail images.")
            visible = payload["frames"] if frames is None else frames
            self.store.delivered(video.id, [f for f in visible if f.get("role") == "context"], "frame")
        return payload

    @measured
    def check_facts(self, video_id, claims):
        """Reopen cited figures and check literal field bindings before explanation.

        This is evidence preparation, not a semantic verifier. The host must inspect
        the returned crops and resolve metric definitions and calculations itself.
        """
        from .fact_check import audit_claim, validate_claims

        claims = validate_claims(claims)
        with self._file_lock("content", video_id), self._visual_lock:
            self.learning.study.plan(video_id)
            video = self.store.get(video_id)
            revision = self._frame_revision(video)
            pictures, checks, warnings, reused = [], [], [], {}
            for claim in claims:
                frame = self.store.frame_at(video_id, claim["timestamp"])
                if frame is None:
                    raise VidLensError(
                        "frame_not_found", "Read the cited original frames before checking facts."
                    )
                box = tuple(claim.get("box") or (0, 0, 1, 1))
                key = (frame.timestamp, box)
                if key not in reused:
                    cropped = self.visual.crop(frame, box)
                    reused[key] = (cropped, len(pictures) + 1)
                    # Matched OCR rows live in checks; do not repeat the entire
                    # figure's text/layout for each quantitative assertion.
                    picture = {k: v for k, v in cropped.to_dict().items() if k != "ocr"}
                    pictures.append(picture | {
                        "evidence_id": f"frame:{frame.timestamp}",
                        "url": source_link(video, frame.timestamp),
                        "sha256": hashlib.sha256(cropped.path.read_bytes()).hexdigest(),
                        "claim_ids": [],
                    })
                    warnings.extend(getattr(self.visual, "last_warnings", []))
                cropped, frame_index = reused[key]
                pictures[frame_index - 1]["claim_ids"].append(claim["id"])
                checks.append(audit_claim(
                    claim, cropped.ocr_regions,
                    image_width=cropped.width, image_height=cropped.height,
                ) | {"frame_index": frame_index})
            if revision != self._frame_revision(self.store.get(video_id)):
                raise VidLensError("stale_revision", "The video changed while checking figures.")
            return {
                "video_id": video_id,
                "source_revision": revision,
                "status": "requires_visual_and_semantic_review",
                "checks": checks,
                "frames": pictures,
                "warnings": list(dict.fromkeys(warnings)),
                "review": [
                    "Inspect each crop: label, value, unit, axis/header and run must agree.",
                    ("Check scope and meaning against the video or primary documentation; "
                     "literal matches do not verify either."),
                    ("Recompute derived values with explicit units and distinguish them "
                     "from on-screen measurements."),
                    ("Check every material quantitative claim in the final draft, including "
                     "captions and added examples; resolve or omit unsupported precision."),
                ],
            }

    def record_fact_check_delivery(self, payload, frames=None):
        # Crops do not count as delivery of a complete source frame. Recheck the
        # revision after image encoding, just as the normal frame path does.
        with self._file_lock("content", payload["video_id"]):
            self.learning.study.plan(payload["video_id"])
            video = self.store.get(payload["video_id"])
            if payload["source_revision"] != self._frame_revision(video):
                raise VidLensError("stale_revision", "The video changed while encoding fact crops.")
        return payload

    def coverage(self, video_id):
        video = self.store.get(video_id)
        if video.metadata.get("local_media_path"):
            self.learning.study.plan(video_id)
        segments = self.store.segments(video_id)
        speech = [s for s in segments if s.kind != "ocr"]
        delivered = self.store.deliveries(video_id)
        text_intervals = union_intervals(
            [(i["start"], i["end"]) for i in delivered if i["kind"] == "text"]
        )
        excerpts = union_intervals(
            [(i["start"], i["end"]) for i in delivered if i["kind"] == "excerpt"]
        )
        frames = sorted({i["start"] for i in delivered if i["kind"] == "frame"})
        indexed = union_intervals([(s.start, s.end) for s in speech])
        return {
            "video_id": video_id,
            "duration": video.duration,
            "indexed_transcript_seconds": round(sum(b - a for a, b in indexed), 3),
            "transcript_sources": sorted({s.source for s in speech}),
            "text_delivered_intervals": text_intervals,
            "excerpt_delivered_intervals": excerpts,
            "frame_timestamps_delivered": frames,
            "ocr_frames_indexed": sum(s.kind == "ocr" for s in segments),
            "scope": "evidence_delivered_to_clients",
            "warnings": self.store.warnings(video_id),
        }

    @measured
    def transcribe(
        self,
        video_id,
        start=0,
        end=None,
        language=None,
        model=None,
        vocabulary=None,
        refresh=False,
        backend="faster-whisper",
    ):
        from .asr import asr_source, resolve_asr
        from .sources import normalize_vocabulary

        backend, model = resolve_asr(backend, model)
        source = asr_source(backend, model)
        vocabulary = normalize_vocabulary(vocabulary)
        if not isinstance(refresh, bool):
            raise VidLensError("invalid_refresh", "refresh must be a boolean.")
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise VidLensError(
                "ASR_LANGUAGE", "language must be a language code or null for detection."
            )
        language = language.strip() if language else None
        from .budget import compact_json

        signature = hashlib.sha256(
            compact_json([backend, model, language, vocabulary]).encode()
        ).hexdigest()[:24]
        with self._file_lock("content", video_id):
            video = self.store.get(video_id)
            start, end = validate_interval(video, start, video.duration if end is None else end)
            history = video.metadata.get("transcriptions", [])
            same_track = sorted(
                (r["start"], r["end"])
                for r in history
                if r.get("language") == language
                and r.get("model") == model
                and r.get("backend", "faster-whisper") == backend
                and r.get("vocabulary", []) == vocabulary
            )
            # Cues already present on this track must not appear twice when follow-up
            # questions request overlapping audio windows.
            missing = []
            cursor = start
            for a, b in same_track:
                if b <= cursor or a >= end:
                    continue
                if a > cursor:
                    missing.append((cursor, a))
                cursor = max(cursor, b)
            if cursor < end:
                missing.append((cursor, end))
            if refresh:
                missing = [(start, end)]
            if not missing:
                available = sum(
                    s.kind == "transcript" and s.source == source
                    for s in self.store.segments(video_id, start, end)
                )
                return {
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                    "cache_hit": True,
                    "segments_added": 0,
                    "segments_available": available,
                    "source": "local_asr",
                    "signature": signature,
                    "backend": backend,
                    "model": model,
                }
            added = 0
            for a, b in missing:
                asset = self.media.audio_interval(self._asset(video, a, b, audio_only=True), a, b)
                options = {"vocabulary": vocabulary} if vocabulary else {}
                if backend != "faster-whisper":
                    options["backend"] = backend
                segments = self.provider.transcribe(
                    asset, language=language, model=model, **options
                )
                self._validate_video(video, segments)
                # Remove stale configuration coverage only after recognition
                # succeeds. Keep the portions outside this replacement window.
                revised = []
                for record in history:
                    if (
                        record.get("language") != language
                        or record["end"] <= a
                        or record["start"] >= b
                    ):
                        revised.append(record)
                        continue
                    for left, right in (
                        (record["start"], min(record["end"], a)),
                        (max(record["start"], b), record["end"]),
                    ):
                        if left < right:
                            piece = {**record, "start": left, "end": right}
                            piece.pop("segments", None)
                            revised.append(piece)
                history = revised
                history.append(
                    {
                        "start": a,
                        "end": b,
                        "language": language,
                        "model": model,
                        "backend": backend,
                        "segments": len(segments),
                        "vocabulary": vocabulary,
                        "signature": signature,
                    }
                )
                video.metadata["transcriptions"] = history
                video.metadata["transcribed_intervals"] = union_intervals(
                    [(r["start"], r["end"]) for r in history]
                )
                self.store.save(
                    video,
                    segments,
                    self.store.warnings(video_id),
                    replace=False,
                    replace_asr_range=(a, b, language),
                )
                added += len(segments)
        return {
            "video_id": video_id,
            "start": start,
            "end": end,
            "segments_added": added,
            "cache_hit": False,
            "source": "local_asr",
            "signature": signature,
            "backend": backend,
            "model": model,
            "next_action": "Read the refreshed interval and match technical symbols with the screen.",
        }

    @staticmethod
    def _validate_video(video: Video, segments: list[Segment]):
        if not video.id or not math.isfinite(video.duration) or video.duration < 0:
            raise VidLensError("invalid_video", "Invalid video metadata.")
        for s in segments:
            if (
                not math.isfinite(s.start)
                or not math.isfinite(s.end)
                or s.start < 0
                or s.end < s.start
            ):
                raise VidLensError(
                    "invalid_transcript",
                    "Subtitle timestamps must be finite, nonnegative and ordered.",
                )

    def import_transcript(self, path, title="", video_url="", media_path=None):
        from .sources import parse_subtitles

        source = Path(path).expanduser().resolve()
        if not source.is_file() or source.stat().st_size > 20_000_000:
            raise VidLensError(
                "invalid_import", "Provide an existing subtitle file no larger than 20 MB."
            )
        content = source.read_text(encoding="utf-8-sig")
        fmt = source.suffix.lower().lstrip(".")
        segments = parse_subtitles(content, fmt, source="imported_subtitles")
        if not segments:
            raise VidLensError("empty_transcript", "No timed text was found in this file.")
        media = Path(media_path).expanduser().resolve() if media_path else None
        if media and not media.is_file():
            raise VidLensError("media_missing", "The supplied local video does not exist.")
        video_id = (
            "local-"
            + hashlib.sha256((content + video_url + str(media or "")).encode()).hexdigest()[:16]
        )
        duration = max(s.end for s in segments)
        if media:
            import re
            import subprocess

            from .visual import ffmpeg_binary

            probe = subprocess.run(
                [ffmpeg_binary(), "-hide_banner", "-i", str(media)],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
            if match:
                duration = max(
                    duration, float(match[1]) * 3600 + float(match[2]) * 60 + float(match[3])
                )
            else:
                raise VidLensError(
                    "media_probe_failed", "Could not determine the supplied video's duration."
                )
        video = Video(
            video_id,
            video_url,
            "local",
            title or source.stem,
            duration,
            metadata={"local_media_path": str(media)} if media else {},
        )
        self._validate_video(video, segments)
        with self._file_lock("content", video_id):
            self.store.save(video, segments, [], invalidate_derived=True)
        return self.overview(video_id)

    @staticmethod
    def transcription_options():
        import platform

        available = []
        if importlib.util.find_spec("faster_whisper") is not None:
            available.append("faster-whisper")
        if (platform.system() == "Darwin" and platform.machine() == "arm64"
                and importlib.util.find_spec("mlx_whisper") is not None
                and "faster-whisper" in available):
            available.append("mlx-whisper")
        preferred = "mlx-whisper" if "mlx-whisper" in available else "faster-whisper"
        return {
            "available_backends": available,
            "suggested": {"backend": preferred,
                          "model": "large-v3-turbo" if preferred == "mlx-whisper" else "base"}
            if available else None,
        }

    def doctor(self):
        capabilities = self.visual.capabilities()
        return {
            "version": __version__,
            "home": str(self.home),
            "python": os.sys.version.split()[0],
            "capabilities": capabilities,
            "optional_asr_installed": importlib.util.find_spec("faster_whisper") is not None,
            "transcription": self.transcription_options(),
            "cookie_file_configured": bool(os.environ.get("VIDLENS_COOKIE_FILE")),
            "token_count_method": self.budget.method,
            "llm_api_required": False,
            "notes": [
                "The connected AI client interprets the evidence and writes learning notes.",
                "Platform access, subtitles, authentication and region restrictions can affect individual URLs.",
            ],
        }

    def list_videos(self, limit=20):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise VidLensError("invalid_limit", "limit must be from 1 to 100.")
        return {"videos": self.store.list(limit)}

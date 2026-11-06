"""Question-guided screen sampling and lexical evidence selection.

This module does no IO. Speech locates intervals; only OCR supplies screen term
matches. A lexical match is a navigation hint, not a judgement about a diagram.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter

from .retrieval import query_tokens, tokenize


def _value(row, field, default=None):
    return row.get(field, default) if isinstance(row, dict) else getattr(row, field, default)


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _terms(text):
    # Preserve code identifiers, numeric literals and isolated mathematical
    # variables through the same bilingual tokenizer used by text retrieval.
    return set(query_tokens(text or ""))


def _queries(question, visual_queries):
    targets = list(dict.fromkeys(q.strip() for q in (visual_queries or ())
                                if isinstance(q, str) and q.strip()))
    return targets or [question.strip()]


def _speech(segments, stamp, radius=5.0):
    return " ".join(_value(s, "text", "") for s in segments
                    if _value(s, "kind", "subtitle") != "ocr"
                    and _value(s, "start", 0) <= stamp + radius
                    and _value(s, "end", 0) >= stamp - radius)


def _position(row):
    requested = row.get("requested_timestamp")
    return requested if _finite(requested) else row["timestamp"]


def _rows(candidates, start, end):
    rows = []
    for index, candidate in enumerate(candidates):
        stamp = candidate.get("timestamp")
        if not _finite(stamp) or not start <= stamp < end:
            continue
        row = dict(candidate)
        row.setdefault("source_candidate_cursor", index)
        row.setdefault("source_timestamp", stamp)
        rows.append(row)
    rows.sort(key=lambda row: row["timestamp"])
    unique = {}
    for index, row in enumerate(rows):
        following = rows[index + 1]["timestamp"] if index + 1 < len(rows) else end
        hold = max(0, following - row["timestamp"])
        row.setdefault("hold_seconds", hold)
        unique.setdefault(round(row["timestamp"], 6), row)
    return list(unique.values())


def _probe(row, reason):
    result = dict(row)
    # Do not move an isolated brief event into its successor. Long holds are
    # sampled just after the change, while the raw scan remains untouched.
    if not row.get("protected") and "end" not in row.get("reasons", ()):
        result["timestamp"] = round(row["timestamp"] + min(0.2, row["hold_seconds"] / 3), 6)
    result["probe_reason"] = reason
    return result


class QueryVisualSelector:
    VERSION = 1

    @staticmethod
    def probe_candidates(question, candidates, segments=(), *, start, end,
                         visual_queries=(), limit=48):
        """Mix timeline coverage with raw changes around speech evidence.

        ``candidates`` must be the full scan, not only compact navigation. This
        keeps a short intermediate state eligible even if pixel dedup omitted it.
        Returned rows retain their full-scan cursor and original timestamp.
        """
        if limit <= 0 or end <= start:
            return []
        rows = _rows(candidates, start, end)
        if len(rows) <= limit:
            return [_probe(row, "all_available_states") for row in rows]
        chosen = {}

        def add(index, reason):
            if len(chosen) < limit:
                chosen.setdefault(index, _probe(rows[index], reason))

        stamps = [row["timestamp"] for row in rows]
        # Keep coverage even if speech recognition missed every query term.
        buckets = max(1, limit // 3)
        for bucket in range(buckets):
            left = start + (end - start) * bucket / buckets
            right = start + (end - start) * (bucket + 1) / buckets
            choices = range(bisect.bisect_left(stamps, left), bisect.bisect_left(stamps, right))
            if choices:
                index = max(choices, key=lambda i: (
                    min(rows[i]["hold_seconds"], 2), -abs(stamps[i] - (left + right) / 2)))
                add(index, "timeline_coverage")

        # Protect raw one-frame/brief states before ranking speech neighborhoods.
        brief = [i for i in range(1, len(rows))
                 if rows[i].get("protected") or (
                     stamps[i] - stamps[i - 1] >= 0.65 and rows[i]["hold_seconds"] <= 0.15)]
        for index in brief[:max(1, limit // 8)]:
            add(index, "brief_transition")

        targets = _queries(question, visual_queries)
        terms = _terms(question + " " + " ".join(targets))
        speech = [s for s in segments if _value(s, "kind", "subtitle") != "ocr"
                  and _value(s, "end", 0) > start and _value(s, "start", 0) < end]
        documents = [_terms(_value(s, "text", "")) for s in speech]
        frequencies = Counter(term for doc in documents for term in doc)
        weights = {term: math.log(1 + (len(documents) + 1) / (frequencies[term] + 1))
                   for term in terms}
        ranked = sorted(range(len(speech)), key=lambda i: (
            sum(weights[t] for t in terms & documents[i]), -_value(speech[i], "start", 0)), reverse=True)
        anchors = []
        for index in ranked:
            if not terms & documents[index]:
                continue
            segment = speech[index]
            left, right = max(start, _value(segment, "start", 0)), min(end, _value(segment, "end", end))
            center = (left + right) / 2
            if any(abs(center - anchor) < 4 for anchor in anchors):
                continue
            anchors.append(center)
            # Include the preceding state and the state after the utterance;
            # technical narration often leads or lags the actual screen change.
            for stamp in (center, left - 0.5, right + 0.5):
                pos = min(len(rows) - 1, max(0, bisect.bisect_right(stamps, stamp) - 1))
                add(pos, "speech_neighborhood")
            if len(chosen) >= limit:
                break

        # Fill remaining slots at the largest uncovered time gaps, never by raw
        # change density (which otherwise spends the budget on pointer movement).
        while len(chosen) < min(limit, len(rows)):
            index = max((i for i in range(len(rows)) if i not in chosen), key=lambda i: (
                min(abs(stamps[i] - stamps[j]) for j in chosen) if chosen else end - start,
                min(rows[i]["hold_seconds"], 2)))
            add(index, "remaining_time_coverage")
        return sorted(chosen.values(), key=lambda row: row["timestamp"])

    @staticmethod
    def refinement_candidates(selected, candidates, *, start, end, limit=48, seen=()):
        """Probe before and after promising screens using the full raw timeline."""
        if limit <= 0 or end <= start:
            return []
        rows = _rows(candidates, start, end)
        if not rows:
            return []
        seen_values = [_value(row, "timestamp") if not _finite(row) else row for row in seen]
        seen_times = {round(value, 6) for value in seen_values if _finite(value)}
        chosen = {}
        anchors = [row["timestamp"] for row in selected if _finite(row.get("timestamp"))]
        # Round-robin distances ensure one strong scene cannot monopolize all
        # refinements. Near changes come first; farther states supply context.
        for offset in (-0.5, 0.5, -2.0, 2.0, -6.0, 6.0):
            for anchor in anchors:
                target = anchor + offset
                if not start <= target < end:
                    continue
                candidates_here = [i for i, row in enumerate(rows)
                                   if (row["timestamp"] < anchor if offset < 0 else row["timestamp"] > anchor)
                                   and abs(row["timestamp"] - target) <= max(0.75, abs(offset) / 2)]
                if not candidates_here:
                    continue
                index = min(candidates_here, key=lambda i: (
                    abs(rows[i]["timestamp"] - target), -min(rows[i]["hold_seconds"], 2)))
                row = _probe(rows[index], "adjacent_screen_state")
                if round(row["timestamp"], 6) not in seen_times:
                    chosen.setdefault(index, row)
                if len(chosen) >= limit:
                    return sorted(chosen.values(), key=lambda row: row["timestamp"])
        return sorted(chosen.values(), key=lambda row: row["timestamp"])

    @staticmethod
    def select(question, frames, segments=(), *, visual_queries=(), limit=4, start=None, end=None):
        """Select a chronological evidence sequence, reporting lexical gaps.

        ``visual_queries`` can name separate screen evidence goals. Shared title
        words are downweighted across frames. Different numeric/symbol states
        are retained: text similarity alone never removes an image.
        """
        targets = _queries(question, visual_queries)
        target_terms = [_terms(target) for target in targets]
        terms = _terms(question) | set().union(*target_terms)
        rows = []
        for frame in sorted(frames, key=lambda row: row.get("timestamp", math.inf)):
            stamp = frame.get("timestamp")
            requested = frame.get("requested_timestamp")
            # A low-frame-rate image can start before the requested interval and
            # still be displayed at the requested time. Keep its actual PTS for
            # citation, but use the request time for interval membership.
            position = requested if _finite(requested) else stamp
            if (not _finite(stamp) or (start is not None and position < start)
                    or (end is not None and position >= end)):
                continue
            # Equal OCR is not equal pixels. Collapse only consecutive identical
            # images; returning to an earlier state is meaningful process evidence.
            digest = frame.get("image_digest") or frame.get("sha256")
            if rows and (round(stamp, 6) == round(rows[-1]["timestamp"], 6) or (
                digest and digest == (rows[-1].get("image_digest") or rows[-1].get("sha256"))
            )):
                previous = rows[-1]
                occurrences = previous.get("occurrence_timestamps", [previous["timestamp"]]) + [stamp]
                previous_speech = previous.get("transcript", "") or _speech(segments, _position(previous))
                current_speech = frame.get("transcript", "") or _speech(segments, position)
                if len(terms & _terms(current_speech)) > len(terms & _terms(previous_speech)):
                    rows[-1] = dict(frame)
                rows[-1]["occurrence_timestamps"] = occurrences
                continue
            rows.append(dict(frame))
        if not rows or limit <= 0:
            return {"selected": [], "query_terms": sorted(terms), "unmatched_query_terms": sorted(terms),
                    "visual_query_matches": [], "unmatched_visual_queries": targets,
                    "metrics": {"candidate_count": len(rows), "ocr_frames": 0}, "status": "no_candidates"}

        screen = [set(tokenize(row.get("ocr", ""))) for row in rows]
        speech = [_terms(row.get("transcript", "") or _speech(segments, _position(row))) for row in rows]
        frequencies = Counter(term for doc in screen for term in doc)
        total = len(rows)
        # The quadratic factor suppresses persistent headers/watermarks much
        # more than ordinary IDF. A common word remains a weak locating hint.
        weights = {term: max(0.025, math.log(1 + total / (frequencies[term] + 1))
                            * (1 - frequencies[term] / total) ** 2)
                   for term in terms}
        for term in weights:
            if len(term) == 1 and term.isalpha() and term.isascii():
                weights[term] *= 0.65
        salient = [{term for term in doc if frequencies[term] / total < 0.8} for doc in screen]
        hits = [doc & terms for doc in screen]
        speech_hits = [doc & terms for doc in speech]
        score = [sum(weights[t] for t in hit) + 0.18 * sum(weights[t] for t in spoken)
                 for hit, spoken in zip(hits, speech_hits)]
        peak = max(score, default=0)
        facet_scores = []
        for hit in hits:
            facet_scores.append([
                sum(weights[t] for t in hit & target) / max(0.001, sum(weights[t] for t in target))
                for target in target_terms])

        def similarity(first, second):
            left, right = salient[first], salient[second]
            if not left and not right:
                return 1.0 if screen[first] and screen[first] == screen[second] else 0.0
            return len(left & right) / max(1, len(left | right))

        chosen, covered, facet_coverage = [], set(), [0.0] * len(targets)
        first_time, last_time = rows[0]["timestamp"], rows[-1]["timestamp"]
        span = max(1, last_time - first_time)
        ocr_count = sum(bool(doc) for doc in screen)
        while len(chosen) < min(limit, total):
            def gain(index, facet_coverage=facet_coverage):
                new_terms = sum(weights[t] for t in hits[index] - covered)
                new_facets = sum(max(0, min(0.7, value) - min(0.7, previous))
                                 for value, previous in zip(facet_scores[index], facet_coverage))
                relevance = score[index] / max(peak, 0.001)
                diversity = (1 - max(similarity(index, other) for other in chosen)) if chosen else 1
                # Time diversity breaks ties between equivalent lexical states;
                # it does not overpower a rare screen match or a missing facet.
                temporal = min((abs(rows[index]["timestamp"] - rows[other]["timestamp"]) / span
                                for other in chosen), default=1)
                continuity = 0.0
                if chosen and relevance > 0:
                    distance = min(abs(rows[index]["timestamp"] - rows[other]["timestamp"])
                                   for other in chosen)
                    continuity = diversity * relevance * max(0, 1 - distance / max(12, span / 3))
                # No OCR/query signal: return a time overview rather than an
                # arbitrary first-page sample and report the actual OCR state.
                if peak <= 0:
                    target = first_time + span * (len(chosen) + 0.5) / min(limit, total)
                    return (-abs(rows[index]["timestamp"] - target), 0, -index)
                value = (2 * new_terms + 2 * new_facets + relevance * (0.45 + 0.7 * diversity)
                         + 0.45 * continuity + 0.15 * temporal)
                return (value, len(salient[index]), -index)

            index = max((i for i in range(total) if i not in chosen), key=gain)
            chosen.append(index)
            covered.update(hits[index])
            facet_coverage = [max(previous, value)
                              for previous, value in zip(facet_coverage, facet_scores[index])]

        selected = []
        for index in sorted(chosen, key=lambda i: rows[i]["timestamp"]):
            reasons = []
            if hits[index]:
                reasons.append("screen_query_terms")
            if speech_hits[index]:
                reasons.append("nearby_speech_terms")
            if len(chosen) > 1 and any(salient[index] != salient[other] for other in chosen):
                reasons.append("different_screen_state")
            if not reasons:
                reasons.append("timeline_context")
            selected.append({**rows[index], "selection_reasons": reasons,
                             "matched_terms": sorted(hits[index]),
                             "speech_matched_terms": sorted(speech_hits[index]),
                             "relevance_score": round(score[index], 5)})
        matches, unmatched_targets = [], []
        for target, target_set in zip(targets, target_terms):
            matched = sorted(target_set & covered)
            row = {"query": target, "matched_terms": matched,
                   "timestamps": [rows[i]["timestamp"] for i in sorted(chosen)
                                  if hits[i] & target_set]}
            matches.append(row)
            if not matched:
                unmatched_targets.append(target)
        return {"selected": selected, "query_terms": sorted(terms),
                "unmatched_query_terms": sorted(terms - covered),
                "visual_query_matches": matches, "unmatched_visual_queries": unmatched_targets,
                "metrics": {"candidate_count": total, "ocr_frames": ocr_count,
                            "selected_count": len(selected), "query_screen_terms": len(covered)},
                "status": ("ocr_unavailable" if not ocr_count else
                           "screen_no_query_match" if not any(hits) else "selected"),
                "matching": "lexical_screen_terms_not_semantic_verification"}

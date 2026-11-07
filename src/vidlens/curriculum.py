"""Source-linked knowledge units and article coverage, independent of time chapters.

The client authors explanations and learning goals. This module validates their
references and structure; it does not certify their meaning or teaching quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time

from .budget import TokenBudget, compact_json
from .models import VidLensError
from .reading import validate_budget
from .retrieval import query_tokens, tokenize
from .store import frame_source_revision

_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}\Z")
_ANCHOR = re.compile(r"(?m)^[ \t]*<!--\s*section:([A-Za-z0-9][A-Za-z0-9_.:-]{0,79})\s*-->[ \t]*$")


def _anchors(markdown):
    """An example containing an anchor inside a code fence is not an article section."""
    fenced = []
    fence, start, offset = None, None, 0
    for line in markdown.splitlines(keepends=True):
        token = re.match(r"^[ ]{0,3}(`{3,}|~{3,})", line)
        if token:
            value = token.group(1)
            if fence is None:
                fence, start = value, offset
            elif value[0] == fence[0] and len(value) >= len(fence):
                fenced.append((start, offset + len(line)))
                fence, start = None, None
        offset += len(line)
    if fence:
        fenced.append((start, len(markdown)))
    return [match for match in _ANCHOR.finditer(markdown)
            if not any(left <= match.start() < right for left, right in fenced)]


def _error(message, code="invalid_curriculum"):
    raise VidLensError(code, message)


def _text(value, name, limit=12000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        _error(f"{name} must contain 1..{limit} characters.")
    return value.strip()


def _key(value, name="id"):
    value = _text(value, name, 80)
    if not _KEY.fullmatch(value):
        _error(f"{name} must be an ASCII identifier using letters, digits, . _ : or -.")
    return value


def _list(value, name, limit=200):
    if not isinstance(value, list) or len(value) > limit:
        _error(f"{name} must be a list with at most {limit} entries.")
    return value


def _strings(value, name, limit=100, width=200):
    return list(dict.fromkeys(_text(v, name, width) for v in _list(value, name, limit)))


def _hash(value):
    return hashlib.sha256(compact_json(value).encode()).hexdigest()


class Curriculum:
    def __init__(self, store, study):
        self.store, self.study = store, study
        self.budget = TokenBudget(store.home)
        with store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_units (
                    video_id TEXT NOT NULL REFERENCES videos(id),
                    source_revision TEXT NOT NULL, unit_id TEXT NOT NULL,
                    data TEXT NOT NULL, updated REAL NOT NULL,
                    PRIMARY KEY(video_id, source_revision, unit_id)
                );
                CREATE TABLE IF NOT EXISTS article_plans (
                    video_id TEXT NOT NULL REFERENCES videos(id),
                    source_revision TEXT NOT NULL, knowledge_revision TEXT NOT NULL,
                    data TEXT NOT NULL, updated REAL NOT NULL,
                    PRIMARY KEY(video_id, source_revision)
                );
            """)

    def _snapshot(self, db, video_id, revision=None):
        state = self.study._state(db, video_id)
        video, segments, _, source_revision, current, _ = state
        if revision is not None:
            self.study._assert_revision(revision, current)
        evidence = {s.id: (s.start, s.end) for s in segments if s.kind != "ocr"}
        try:
            identity = frame_source_revision(video)
        except OSError:
            identity = None  # Retain speech access when a local media file is unavailable.
        for row in db.execute("SELECT timestamp,data FROM frames WHERE video_id=?", (video_id,)):
            data = json.loads(row["data"])
            if identity is not None and data.get("_source_identity", identity) == identity:
                stamp = float(row["timestamp"])
                evidence[f"frame:{stamp}"] = (stamp, stamp)
        return video, source_revision, current, evidence

    @staticmethod
    def _refs(value, evidence):
        refs = _strings(value, "evidence_ids", limit=200)
        if not refs:
            _error("Each knowledge unit, relation and visual needs original evidence IDs.")
        result = []
        for ref in refs:
            if ref.startswith("frame:"):
                try:
                    stamp = float(ref[6:])
                    if not math.isfinite(stamp):
                        raise ValueError
                    ref = f"frame:{stamp}"
                except ValueError:
                    _error("Invalid frame reference.", "invalid_evidence_reference")
            if ref not in evidence:
                _error(f"Unknown or stale evidence for this video: {ref}",
                       "invalid_evidence_reference")
            if ref not in result:
                result.append(ref)
        return result

    def _unit(self, unit, evidence):
        if not isinstance(unit, dict):
            _error("Each unit must be an object.")
        result = {"id": _key(unit.get("id")), "title": _text(unit.get("title"), "title", 300),
                  "explanation": _text(unit.get("explanation"), "explanation"),
                  "roles": _strings(unit.get("roles", []), "roles", 12),
                  "concepts": _strings(unit.get("concepts", []), "concepts", 40),
                  "evidence_ids": self._refs(unit.get("evidence_ids"), evidence),
                  "requires": [_key(v, "requires") for v in
                               _list(unit.get("requires", []), "requires")],
                  "importance": unit.get("importance", "core"), "relations": []}
        if (not result["roles"] or not isinstance(result["importance"], str)
                or result["importance"] not in {"core", "supporting"}):
            _error("Supply at least one role and importance='core' or 'supporting'.")
        result["requires"] = list(dict.fromkeys(result["requires"]))
        for relation in _list(unit.get("relations", []), "relations", 40):
            if not isinstance(relation, dict):
                _error("Each relation must be an object.")
            result["relations"].append({
                **{key: _text(relation.get(key), key, 1000)
                   for key in ("source", "target", "relation")},
                "evidence_ids": self._refs(relation.get("evidence_ids"), evidence),
            })
        if "questions" in unit:
            result["questions"] = _strings(unit["questions"], "questions", 20, 1000)
        if "example" in unit:
            result["example"] = _text(unit["example"], "example")
        refs = [*result["evidence_ids"], *(ref for relation in result["relations"]
                                        for ref in relation["evidence_ids"])]
        result["evidence_ranges"] = [list(evidence[ref]) for ref in dict.fromkeys(refs)]
        return result

    @staticmethod
    def _load(db, video_id, source_revision):
        return {row["unit_id"]: json.loads(row["data"]) for row in db.execute(
            "SELECT unit_id,data FROM knowledge_units WHERE video_id=? AND source_revision=? "
            "ORDER BY unit_id", (video_id, source_revision))}

    @staticmethod
    def _dependencies(units):
        visiting, visited = set(), set()

        def visit(key):
            if key not in units:
                _error(f"Unknown required unit: {key}")
            if key in visiting:
                _error(f"Cyclic knowledge dependency at: {key}")
            if key in visited:
                return
            visiting.add(key)
            for dependency in units[key]["requires"]:
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in units:
            visit(key)

    @staticmethod
    def _revision(units):
        return _hash([units[key] for key in sorted(units)])[:24]

    def save(self, video_id, revision, units, *, replace=False):
        """Atomically upsert units, or explicitly replace the current source's inventory."""
        _text(revision, "revision", 100)
        units = _list(units, "units", 200)
        if not units or not isinstance(replace, bool):
            _error("Provide at least one unit and a boolean replace option.")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _, source, current, evidence = self._snapshot(db, video_id, revision)
            existing = {} if replace else self._load(db, video_id, source)
            cleaned = [self._unit(unit, evidence) for unit in units]
            if len({unit["id"] for unit in cleaned}) != len(cleaned):
                _error("Unit IDs must be unique within a save request.")
            existing.update((unit["id"], unit) for unit in cleaned)
            if len(existing) > 200:
                _error("A video may contain at most 200 knowledge units.")
            self._dependencies(existing)
            if replace:
                db.execute("DELETE FROM knowledge_units WHERE video_id=? AND source_revision=?",
                           (video_id, source))
            for unit in cleaned:
                db.execute("INSERT INTO knowledge_units VALUES (?,?,?,?,?) "
                           "ON CONFLICT(video_id,source_revision,unit_id) DO UPDATE SET "
                           "data=excluded.data,updated=excluded.updated",
                           (video_id, source, unit["id"], compact_json(unit), time.time()))
            return {"video_id": video_id, "revision": current, "source_revision": source,
                    "knowledge_revision": self._revision(existing),
                    "saved_unit_ids": [unit["id"] for unit in cleaned],
                    "unit_count": len(existing), "validation": "references_and_structure_only"}

    def query(self, video_id, query="", roles=None, limit=12, max_tokens=6000):
        validate_budget(max_tokens)
        if not isinstance(query, str) or len(query) > 2000:
            _error("query must be text up to 2000 characters.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            _error("limit must be an integer from 1 to 200.")
        roles = _strings([] if roles is None else roles, "roles", 12)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            video, source, current, evidence = self._snapshot(db, video_id)
            units = self._load(db, video_id, source)
            for unit in units.values():
                self._unit(unit, evidence)
        terms = set(query_tokens(query))
        ranked = []
        for key, unit in units.items():
            if roles and not set(roles).intersection(unit["roles"]):
                continue
            text = " ".join([unit["title"], unit["explanation"], *unit["concepts"],
                             unit.get("example", ""), *unit.get("questions", []),
                             *(r[f] for r in unit["relations"]
                               for f in ("source", "target", "relation"))])
            overlap = terms.intersection(tokenize(text))
            if terms and not overlap:
                continue
            ranked.append((len(overlap) / max(1, len(terms)), key))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        result = {"video_id": video_id, "revision": current, "source_revision": source,
                  "knowledge_revision": self._revision(units), "source_url": video.url,
                  "matched_unit_ids": [], "units": [], "unit_count": len(units),
                  "available_matches": len(ranked), "truncated": False,
                  "dependencies_complete": True,
                  "accounting": {"text_tokens_estimate": 0, "budget": max_tokens,
                                 "method": self.budget.method,
                                 "excludes": "images, model reasoning, transport envelope"}}
        selected = set()

        def closure(key, output):
            if key not in output:
                for dependency in units[key]["requires"]:
                    closure(dependency, output)
                output.append(key)

        for _, key in ranked[:limit]:
            needed = []
            closure(key, needed)
            added = [unit_id for unit_id in needed if unit_id not in selected]
            count = len(result["units"])
            result["units"].extend(units[unit_id] for unit_id in added)
            result["matched_unit_ids"].append(key)
            if self.budget.count(compact_json(result)) > max_tokens - 16:
                del result["units"][count:]
                result["matched_unit_ids"].pop()
                result["truncated"] = True
                continue
            selected.update(added)
        result["truncated"] |= len(result["matched_unit_ids"]) < len(ranked)
        return self.budget.recount(result)

    @staticmethod
    def _unit_ids(values, units, name="unit_ids", allow_empty=False):
        values = list(dict.fromkeys(_key(v, name) for v in _list(values, name)))
        if not values and not allow_empty:
            _error(f"{name} must not be empty.")
        if any(value not in units for value in values):
            _error(f"{name} contains an unknown current knowledge unit.")
        return values

    def _omissions(self, value, units):
        output = []
        for item in _list(value, "omissions"):
            if not isinstance(item, dict):
                _error("Each omission needs a unit_id and reason.")
            unit_id = self._unit_ids([item.get("unit_id")], units)[0]
            if any(old["unit_id"] == unit_id for old in output):
                _error("Duplicate omission.")
            output.append({"unit_id": unit_id, "reason": _text(item.get("reason"), "reason")})
        return output

    def _clean_plan(self, plan, units, evidence):
        if not isinstance(plan, dict):
            _error("plan must be an object.")
        output = {"title": _text(plan.get("title"), "title", 300),
                  "goal": _text(plan.get("goal"), "goal", 2000), "sections": [],
                  "omissions": self._omissions(plan.get("omissions", []), units)}
        for section in _list(plan.get("sections"), "sections", 100):
            if not isinstance(section, dict):
                _error("Each article section must be an object.")
            item = {"id": _key(section.get("id")),
                    "title": _text(section.get("title"), "title", 300),
                    "objective": _text(section.get("objective"), "objective", 2000),
                    "unit_ids": self._unit_ids(section.get("unit_ids"), units),
                    "visuals": [], "exercises": []}
            for visual in _list(section.get("visuals", []), "visuals", 20):
                if not isinstance(visual, dict):
                    _error("Each visual needs purpose and evidence_ids.")
                item["visuals"].append({"purpose": _text(visual.get("purpose"), "purpose", 2000),
                                        "evidence_ids": self._refs(visual.get("evidence_ids"),
                                                                   evidence)})
            for exercise in _list(section.get("exercises", []), "exercises", 20):
                if not isinstance(exercise, dict):
                    _error("Each exercise needs question and unit_ids.")
                item["exercises"].append({"question": _text(exercise.get("question"),
                                                           "question", 2000),
                                          "unit_ids": self._unit_ids(exercise.get("unit_ids"),
                                                                     units)})
            output["sections"].append(item)
        ids = [section["id"] for section in output["sections"]]
        if not ids or len(set(ids)) != len(ids):
            _error("Provide nonempty article sections with unique IDs.")
        assigned = {key for section in output["sections"] for key in section["unit_ids"]}
        if assigned.intersection(item["unit_id"] for item in output["omissions"]):
            _error("A unit cannot be both planned and omitted.")
        return output

    def plan(self, video_id, revision, plan=None):
        _text(revision, "revision", 100)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _, source, current, evidence = self._snapshot(db, video_id, revision)
            units = self._load(db, video_id, source)
            knowledge_revision = self._revision(units)
            if plan is not None:
                clean = self._clean_plan(plan, units, evidence)
                db.execute("INSERT INTO article_plans VALUES (?,?,?,?,?) "
                           "ON CONFLICT(video_id,source_revision) DO UPDATE SET "
                           "knowledge_revision=excluded.knowledge_revision,"
                           "data=excluded.data,updated=excluded.updated",
                           (video_id, source, knowledge_revision, compact_json(clean), time.time()))
            row = db.execute("SELECT * FROM article_plans WHERE video_id=? AND source_revision=?",
                             (video_id, source)).fetchone()
            return {"video_id": video_id, "revision": current, "source_revision": source,
                    "knowledge_revision": knowledge_revision,
                    "plan": json.loads(row["data"]) if row else None,
                    "stale": bool(row and row["knowledge_revision"] != knowledge_revision)}

    @staticmethod
    def _tasks(draft, sections, spans, markdown):
        """Check that planned visual/exercise tasks have actual section-local content."""
        from markdown_it import MarkdownIt

        renderer, environment = MarkdownIt("commonmark"), {}
        renderer.parse(markdown, environment)

        def image_sources(tokens):
            for token in tokens:
                if token.type == "image":
                    yield token.attrGet("src")
                if token.children:
                    yield from image_sources(token.children)

        image_map = {key: set(image_sources(renderer.parse(text, dict(environment))))
                     for key, text in spans.items()}
        assets = _list(draft.get("assets", []), "assets", 40)
        if any(not isinstance(asset, dict) for asset in assets):
            _error("Each article asset must be an object.")
        asset_names = {asset.get("name") for asset in assets if isinstance(asset.get("name"), str)}
        issues, result = [], {}
        for kind, field in (("visual", "visuals"), ("exercise", "exercises")):
            expected = {(key, index) for key, section in sections.items()
                        for index in range(len(section[field]))}
            covered, seen, mappings = set(), set(), []
            for item in _list(draft.get(f"{kind}_coverage", []), f"{kind}_coverage", 400):
                if not isinstance(item, dict):
                    _error(f"Each {kind} coverage entry must be an object.")
                section = _key(item.get("section_id"), "section_id")
                index = item.get(f"{kind}_index")
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    _error(f"{kind}_index must be a nonnegative integer (zero-based).")
                key = (section, index)
                if key in seen:
                    issues.append(f"duplicate_{kind}_mapping:{section}:{index}")
                seen.add(key)
                if key not in expected:
                    issues.append(f"unplanned_{kind}_mapping:{section}:{index}")
                    continue
                mapping = {"section_id": section, f"{kind}_index": index}
                if kind == "visual":
                    name = _text(item.get("asset_name"), "asset_name", 91)
                    if (name not in asset_names or f"asset:{name}" not in image_map.get(section, set())):
                        issues.append(f"missing_section_visual:{section}:{index}")
                        continue
                    mapping["asset_name"] = name
                else:
                    excerpt = _text(item.get("excerpt"), "excerpt")
                    if section not in spans or excerpt not in spans[section]:
                        issues.append(f"missing_section_exercise:{section}:{index}")
                        continue
                covered.add(key)
                mappings.append(mapping)
            result[f"{kind}_coverage"] = mappings
            result[f"uncovered_{field}"] = [{"section_id": key, f"{kind}_index": index}
                                            for key, index in sorted(expected - covered)]
            if expected - covered:
                issues.append(f"uncovered_{field}")
        return issues, result

    def audit(self, video_id, revision, draft):
        """Check explicit article mapping, not truth, prose quality or reader comprehension."""
        _text(revision, "revision", 100)
        if not isinstance(draft, dict) or not isinstance(draft.get("markdown"), str):
            _error("draft must contain a markdown string.")
        markdown = draft["markdown"]
        if len(markdown) > 500000:
            _error("Draft exceeds 500000 characters.")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _, source, current, evidence = self._snapshot(db, video_id, revision)
            units = self._load(db, video_id, source)
            for unit in units.values():
                self._unit(unit, evidence)
            knowledge_revision = self._revision(units)
            row = db.execute("SELECT * FROM article_plans WHERE video_id=? AND source_revision=?",
                             (video_id, source)).fetchone()
        issues = []
        plan = json.loads(row["data"]) if row else {"sections": []}
        if not row:
            issues.append("missing_article_plan")
        elif row["knowledge_revision"] != knowledge_revision:
            issues.append("stale_article_plan")
        sections = {section["id"]: section for section in plan["sections"]}
        core = {key for key, unit in units.items() if unit["importance"] == "core"}
        if not core:
            issues.append("no_core_knowledge_units")
        if not markdown.strip():
            issues.append("empty_markdown")
        matches = _anchors(markdown)
        spans = {}
        for index, match in enumerate(matches):
            key = match.group(1)
            if key in spans:
                issues.append(f"duplicate_section_anchor:{key}")
            if key not in sections:
                issues.append(f"unknown_section_anchor:{key}")
            text = markdown[match.end():matches[index + 1].start()
                            if index + 1 < len(matches) else len(markdown)]
            spans[key] = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        coverage, covered = [], set()
        for item in _list(draft.get("coverage", []), "coverage", 400):
            if not isinstance(item, dict):
                _error("Each coverage entry needs unit_id, section_id and excerpt.")
            unit_id = self._unit_ids([item.get("unit_id")], units)[0]
            section_id = _key(item.get("section_id"), "section_id")
            excerpt = _text(item.get("excerpt"), "excerpt")
            if section_id not in sections or unit_id not in sections[section_id]["unit_ids"]:
                issues.append(f"unplanned_mapping:{unit_id}:{section_id}")
            elif section_id not in spans or excerpt not in spans[section_id]:
                issues.append(f"missing_section_excerpt:{unit_id}:{section_id}")
            else:
                covered.add(unit_id)
                coverage.append({"unit_id": unit_id, "section_id": section_id})
        omissions = self._omissions(draft.get("omissions", []), units)
        omitted = {item["unit_id"] for item in omissions}
        if covered & omitted:
            issues.append("covered_units_also_omitted")
        required, pending = set(), list(covered)
        while pending:
            for dependency in units[pending.pop()]["requires"]:
                if dependency not in required:
                    required.add(dependency)
                    pending.append(dependency)
        missing_dependencies = sorted(required - covered)
        uncovered_core = sorted(core - covered)
        if uncovered_core:
            issues.append("uncovered_core_units")
        if missing_dependencies:
            issues.append("uncovered_dependencies")
        task_issues, task_coverage = self._tasks(draft, sections, spans, markdown)
        issues.extend(task_issues)
        return {"video_id": video_id, "revision": current, "knowledge_revision": knowledge_revision,
                "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                "ready_for_review": not issues, "issues": list(dict.fromkeys(issues)),
                "coverage": coverage, "core_unit_count": len(core),
                "covered_core_count": len(core & covered), "uncovered_core": uncovered_core,
                "uncovered_dependencies": missing_dependencies, "omissions": omissions,
                "unaccounted_units": sorted(set(units) - covered - omitted),
                **task_coverage,
                "semantic_validation": "not_performed",
                "review_scope": "Exact excerpts and evidence references are checked; facts, "
                                "visual explanations and learning outcomes require reader review."}

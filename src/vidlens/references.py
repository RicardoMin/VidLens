"""Persist client-reviewed external sources alongside the video sections they explain."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict, deque
from urllib.parse import urlsplit, urlunsplit

from .budget import compact_json
from .models import VidLensError
from .retrieval import normalized_text, query_tokens, tokenize


class ReferenceStore:
    def __init__(self, store, study):
        self.store, self.study = store, study
        with store.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS study_references (
                id TEXT PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(id),
                section_id TEXT NOT NULL, revision TEXT NOT NULL, data TEXT NOT NULL,
                created REAL NOT NULL
            )""")

    def add(self, video_id, section_id, revision, reference):
        if not isinstance(reference, dict):
            raise VidLensError("invalid_reference", "Provide a reference object.")
        fields = {"url": 4000, "title": 500, "excerpt": 12000, "explanation": 4000}
        data = {}
        for key, limit in fields.items():
            value = reference.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise VidLensError("invalid_reference", f"{key} must contain 1..{limit} characters.")
            data[key] = value.strip()
        try:
            parsed = urlsplit(data["url"])
            if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError
            _port = parsed.port  # Validate malformed or out-of-range ports.
        except ValueError as exc:
            raise VidLensError("invalid_reference", "Use a public HTTP(S) source URL without credentials.") from exc
        data["url"] = urlunsplit(parsed)
        concepts = reference.get("concepts", [])
        if not isinstance(concepts, list) or len(concepts) > 30 or any(
            not isinstance(value, str) or not value.strip() or len(value) > 200 for value in concepts
        ):
            raise VidLensError("invalid_reference", "concepts must be a list of up to 30 short names.")
        data["concepts"] = list(dict.fromkeys(value.strip() for value in concepts))
        data["provenance"] = "external_source_read_by_client"
        data["retrieved_at"] = int(time.time())
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _, _, _, source_revision, current_revision, sections = self.study._state(db, video_id)
            if revision != current_revision:
                raise VidLensError("stale_revision", "Read the current study plan before adding a source.")
            self.study._find_section(sections, section_id)
            identity = [video_id, section_id, source_revision, data["url"], data["excerpt"]]
            identity = "ref:" + hashlib.sha256(compact_json(identity).encode()).hexdigest()[:20]
            db.execute("INSERT OR REPLACE INTO study_references VALUES (?,?,?,?,?,?)",
                       (identity, video_id, section_id, source_revision, compact_json(data), time.time()))
        return {"id": identity, "section_id": section_id, **data}

    def list(self, video_id, revision):
        with self.store.connect() as db:
            return self._list(db, video_id, revision)

    @staticmethod
    def _list(db, video_id, revision):
        rows = db.execute(
            "SELECT id,section_id,data FROM study_references WHERE video_id=? AND revision=? "
            "ORDER BY created,id", (video_id, revision),
        ).fetchall()
        return [{"id": row["id"], "section_id": row["section_id"], **json.loads(row["data"])} for row in rows]

    @staticmethod
    def _document_text(document):
        # Index authored content, never dict keys, evidence IDs or timestamps.
        parts = [document.get(key, "") for key in ("title", "summary", "excerpt", "explanation")]
        parts.extend(document.get("concepts", []))
        parts.extend(claim["text"] for claim in document.get("claims", []))
        parts.extend(
            relation[key]
            for relation in document.get("relations", [])
            for key in ("source", "target", "relation")
        )
        return " ".join(parts)

    @staticmethod
    def _related_relations(documents, terms):
        """Find short connections using only stored, evidence-backed relations.

        Traversal is undirected for discovery; returned relations retain their
        authored direction and wording, including corrections/negative claims.
        Entity labels are normalized, not merged by broad synonym heuristics.
        """
        adjacency = defaultdict(list)
        labels = {}
        for doc_index, document in enumerate(documents):
            for relation_index, relation in enumerate(document.get("relations", [])):
                source, target = (
                    " ".join(normalized_text(relation[key]).split()) for key in ("source", "target")
                )
                labels[source], labels[target] = relation["source"], relation["target"]
                edge = (doc_index, relation_index)
                adjacency[source].append((target, edge))
                adjacency[target].append((source, edge))
        seeds = [node for node, label in labels.items() if (words := set(tokenize(label))) and words <= terms]
        # Bound graph work even for broad questions or very large note libraries.
        seeds = seeds[:12]
        selected = defaultdict(set)
        if len(seeds) < 2:
            return selected
        seed_set = set(seeds)
        for origin in seeds:
            queue = deque([(origin, (), frozenset({origin}))])
            best_depth = {origin: 0}
            expanded, examined = 0, 0
            while queue and examined < 4096:
                node, path, visited = queue.popleft()
                expanded += 1
                if path and node in seed_set:
                    if len(path) > 1:
                        for doc_index, relation_index in path:
                            selected[doc_index].add(relation_index)
                    continue
                if len(path) >= 3:
                    continue
                for neighbor, edge in adjacency[node]:
                    examined += 1
                    if examined >= 4096 or expanded + len(queue) >= 512:
                        break
                    if neighbor in visited or best_depth.get(neighbor, 4) < len(path) + 1:
                        continue
                    best_depth[neighbor] = len(path) + 1
                    queue.append((neighbor, (*path, edge), visited | {neighbor}))
        return selected

    def search(self, video_id, query, limit=6):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise VidLensError("invalid_query", "Provide a question of 1..2000 characters.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise VidLensError("invalid_limit", "limit must be from 1 to 20.")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _, _, _, source_revision, revision, sections = self.study._state(db, video_id)
            notes = self.study._notes(db, video_id, source_revision)
            documents = [
                {"kind": "section_note", "section_id": s["id"], "start": s["start"],
                 "end": s["end"], **notes[s["id"]]}
                for s in sections if s["id"] in notes
            ]
            section_ids = {s["id"] for s in sections}
            documents.extend({"kind": "external_reference", **ref}
                             for ref in self._list(db, video_id, source_revision)
                             if ref["section_id"] in section_ids)
        terms = set(query_tokens(query))
        related = self._related_relations(documents, terms)
        scored = []
        for index, document in enumerate(documents):
            words = set(tokenize(self._document_text(document)))
            overlap = terms & words
            if overlap or index in related:
                score = len(overlap) / max(1, len(terms))
                if index in related:
                    document = dict(document, matched_relation_indices=sorted(related[index]),
                                    match_reason="relation_path")
                    score += 1
                scored.append((score, document))
        scored.sort(key=lambda item: -item[0])
        return {"video_id": video_id, "revision": revision,
                "matches": [dict(document, score=round(score, 3)) for score, document in scored[:limit]]}

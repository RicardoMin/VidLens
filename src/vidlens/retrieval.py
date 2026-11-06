"""Local bilingual lexical retrieval, temporal context, and extractive navigation."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import Segment, Video


@lru_cache(maxsize=1)
def _chinese_converter():
    from opencc import OpenCC

    return OpenCC("t2s")


@lru_cache(maxsize=8192)
def normalized_text(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold().replace("−", "-")
    return folded if folded.isascii() else _chinese_converter().convert(folded)


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "to",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "at",
    "from",
    "with",
    "that",
    "this",
    "it",
    "as",
    "by",
    "how",
    "what",
    "why",
    "does",
    "do",
    "can",
    "could",
    "would",
    "should",
    "video",
    "explain",
    "tell",
    "me",
    "about",
    "please",
    "author",
    "says",
    "said",
    "using",
    "use",
    "used",
    "example",
    "examples",
    "actually",
    "just",
    "also",
    "so",
    "you",
    "we",
    "i",
    "they",
    "he",
    "she",
    "have",
    "has",
    "had",
    "than",
    "then",
    "more",
    "very",
    "which",
    "when",
    "where",
    "into",
    "your",
    "our",
    "its",
    "these",
    "those",
}
ZH_STOP = {
    "这个",
    "那个",
    "这里",
    "什么",
    "为什么",
    "怎么",
    "如何",
    "我们",
    "你们",
    "它们",
    "一个",
    "就是",
    "然后",
    "可以",
    "视频",
    "作者",
    "讲解",
    "介绍",
    "请问",
    "告诉",
    "一下",
    "哪些",
    "进行",
    "以及",
    "其中",
    "因为",
    "所以",
}
ALIASES = (
    ("kv cache", "kv缓存", "键值缓存"),
    ("batch size", "批大小", "批量大小"),
    ("latency", "延迟", "时延"),
    ("throughput", "吞吐", "吞吐量"),
    ("memory", "内存", "显存"),
    ("baseline", "基线", "对比基准"),
    ("attention", "注意力"),
    ("gradient", "梯度"),
    ("learning rate", "学习率"),
    ("training", "训练"),
    ("inference", "推理"),
    ("token", "词元"),
    ("limitation", "limitations", "限制", "局限"),
    ("assumption", "前提", "假设"),
    ("complexity", "复杂度"),
    ("benchmark", "基准测试"),
)

_ZH_STOPS = tuple(sorted(ZH_STOP, key=lambda item: (-len(item), item)))
_TERMS = re.compile(
    r"[\u3400-\u9fff]+|--[a-z][a-z0-9_-]*|(?<![a-z0-9_])[+-](?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?|"
    r"[a-z0-9_]+(?:[./#-][a-z0-9_]+)*(?:\+\+|#)?|[α-ω]+|!=|==|<=|>=|≤|≥|≠|<|>"
)
_OPERATORS = {"≤": "<=", "≥": ">=", "≠": "!="}


def tokenize(text: str) -> list[str]:
    """Keep identifiers and CJK unigrams/bigrams instead of stripping non-ASCII text."""
    text = normalized_text(text)
    result = []
    for part in _TERMS.findall(text):
        if "\u3400" <= part[0] <= "\u9fff":
            # Remove question boilerplate before indexing adjacent characters.
            for stop in _ZH_STOPS:
                part = part.replace(stop, " ")
            for word in part.split():
                if len(word) == 1:
                    result.append(word)
                else:
                    result.extend(word[i : i + 2] for i in range(len(word) - 1))
                    if len(word) <= 6:
                        result.append(word)
        elif part not in STOPWORDS:
            result.append(_OPERATORS.get(part, part))
            if (
                re.search(r"[._/]", part)
                and not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?", part)
            ):
                # Preserve the qualified identifier for exact matches, while
                # allowing questions about `summary` to find model.summary().
                # Numeric literals such as 0.001 remain indivisible.
                result.extend(p for p in re.split(r"[._/]", part)
                              if p and p != part and p not in STOPWORDS)
    return result


def query_tokens(query: str) -> list[str]:
    normalized = normalized_text(query)
    expanded = query
    for group in ALIASES:
        if any(term in normalized for term in group):
            expanded += " " + " ".join(group)
    return list(dict.fromkeys(tokenize(expanded)))


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def source_link(video: Video, start: float) -> str:
    if not video.url.startswith(("https://", "http://")):
        return ""
    parts = urlsplit(video.url)
    params = [(k, v) for k, v in parse_qsl(parts.query) if k not in ("t", "start")]
    params.append(("t", str(max(0, int(start)))))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), ""))


@dataclass
class Passage:
    start: float
    end: float
    text: str
    kind: str
    source: str
    evidence_ids: list[str] = field(default_factory=list)
    score: float = 0

    def to_dict(self, video: Video) -> dict:
        key = hashlib.sha256("|".join(self.evidence_ids).encode()).hexdigest()[:10]
        result = {
            "id": f"{self.kind}:{key}",
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "time": f"{timestamp(self.start)}–{timestamp(self.end)}",
            "text": self.text,
            "kind": self.kind,
            "source": self.source,
            "url": source_link(video, self.start),
        }
        if self.score:
            result["relevance"] = round(self.score, 3)
        return result


def _groups(segments: list[Segment], max_seconds: float, max_chars: int) -> Iterator[list[Segment]]:
    current: list[Segment] = []
    characters = 0
    for segment in sorted(segments, key=lambda s: (s.start, s.kind)):
        # OCR describes one screen instant and is independent of surrounding speech.
        if current and (
            segment.kind != current[0].kind
            or segment.start - current[-1].end > 15
            or segment.end - current[0].start > max_seconds
            or characters + len(segment.text) > max_chars
        ):
            yield current
            current = []
            characters = 0
        if segment.kind == "ocr":
            yield [segment]
        else:
            current.append(segment)
            characters += len(segment.text)
    if current:
        yield current


def _passage(cues: list[Segment], score: float = 0) -> Passage:
    return Passage(
        cues[0].start,
        max(s.end for s in cues),
        " ".join(s.text.strip() for s in cues),
        cues[0].kind,
        ",".join(dict.fromkeys(s.source for s in cues)),
        [s.id for s in cues],
        score,
    )


def passages(
    segments: list[Segment], max_seconds: float = 55, max_chars: int = 1200
) -> list[Passage]:
    return [_passage(group) for group in _groups(segments, max_seconds, max_chars)]


@dataclass
class _Document:
    cues: list[Segment]
    cue_terms: list[frozenset[str]]
    phrase: str


def _indexed_groups(segments: list[Segment]) -> Iterator[list[Segment]]:
    # Adding a screen capture or a second caption track must not repartition
    # every later passage of an existing speech track.
    tracks: dict[tuple[str, str, str], list[Segment]] = defaultdict(list)
    for segment in segments:
        tracks[segment.kind, segment.source, segment.language].append(segment)
    for track in tracks.values():
        yield from _groups(track, 55, 1200)


_CONTINUATION = re.compile(
    r"^(?:\([^)]{1,40}\)\s*)?(?:但|不过|除非|否则|前提是|也就是|意思是|更准确|换句话说|"
    r"没错|是的|对[,， ]|however\b|unless\b|otherwise\b|but\b|that means\b|it means\b|"
    r"in other words\b|more precisely\b|yes\b|no\b)"
)


class PassageIndex:
    """BM25 index for one version of a video. Treat its source segments as immutable."""

    def __init__(self, segments: list[Segment]):
        self._docs: list[_Document] = []
        self._postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        frequencies = []
        document_counts: Counter[str] = Counter()
        lengths = []
        for cues in _indexed_groups(segments):
            tokens = [tokenize(cue.text) for cue in cues]
            words = [word for cue_words in tokens for word in cue_words]
            counts = Counter(words)
            frequencies.append(counts)
            lengths.append(len(words))
            document_counts.update(counts.keys())
            self._docs.append(_Document(cues, [frozenset(row) for row in tokens], " ".join(words)))

        count = len(self._docs)
        average_length = sum(lengths) / max(1, count) or 1
        self._idf = {
            term: math.log1p((count - df + 0.5) / (df + 0.5))
            for term, df in document_counts.items()
        }
        for index, counts in enumerate(frequencies):
            length_term = 1.2 * (0.25 + 0.75 * lengths[index] / average_length)
            for term, frequency in counts.items():
                weight = self._idf[term] * frequency * 2.2 / (frequency + length_term)
                self._postings[term].append((index, weight))

    def _bounds(self, doc: _Document, terms: set[str]) -> tuple[int, int]:
        cues = doc.cues
        if len(cues) < 6 or cues[-1].end - cues[0].start <= 35:
            return 0, len(cues)

        # Find a consecutive run carrying most of this passage's query signal.
        # Keep original neighboring cues so a matched term doesn't lose its answer.
        weights = [math.fsum(self._idf[t] for t in words & terms) for words in doc.cue_terms]
        target = sum(weights) * 0.70
        if not target:
            return 0, len(cues)
        best = (0, len(cues) - 1)
        best_duration = cues[-1].end - cues[0].start
        left, mass = 0, 0.0
        for right, weight in enumerate(weights):
            mass += weight
            while left <= right and mass - weights[left] >= target:
                mass -= weights[left]
                left += 1
            duration = cues[right].end - cues[left].start
            if mass >= target and duration < best_duration:
                best, best_duration = (left, right), duration

        left, right = best
        left = max(0, left - 1)
        right = min(len(cues) - 1, right + 2)
        # An isolated cue is often only a question. Allow time for the explanation.
        while cues[right].end - cues[left].start < 24 and (left or right + 1 < len(cues)):
            if right + 1 < len(cues):
                right += 1
            if left and cues[right].end - cues[left].start < 24:
                left -= 1
        if right + 1 < len(cues):
            following = cues[right + 1]
            asks_question = (
                normalized_text(cues[right].text).rstrip().endswith(("?", "吗", "呢", "吧"))
            )
            if (
                following.end - cues[right].end <= 15
                and len(following.text) <= 360
                and (asks_question or _CONTINUATION.match(normalized_text(following.text).lstrip()))
            ):
                right += 1
        return left, right + 1

    def rank(self, query: str, limit: int = 8) -> list[Passage]:
        terms = query_tokens(query)
        if not terms or limit <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            for index, weight in self._postings.get(term, ()):
                scores[index] += weight
        phrase = " ".join(tokenize(query))
        if phrase:
            for index in scores:
                if phrase in self._docs[index].phrase:
                    scores[index] *= 1.25
        ranked = sorted(scores, key=lambda i: (-scores[i], self._docs[i].cues[0].start))[:limit]
        query_terms = set(terms)
        bounds = {i: self._bounds(self._docs[i], query_terms) for i in ranked}
        result = []
        for i in ranked:
            original = self._docs[i].cues
            left, end = bounds[i]
            cues = original[left:end]
            # A question, negation, or setup often ends at an arbitrary grouping
            # boundary. Include its spoken continuation without repeating cues
            # that another selected excerpt will already return.
            if end == len(original) and cues[-1].kind != "ocr" and i + 1 < len(self._docs):
                following = self._docs[i + 1].cues
                available = bounds[i + 1][0] if i + 1 in bounds else len(following)
                boundary = cues[-1]
                extra_chars = 0
                for cue in following[: min(3, available)]:
                    if (
                        cue.kind != boundary.kind
                        or cue.source != boundary.source
                        or cue.language != boundary.language
                        or cue.start - cues[-1].end > 2
                        or cue.end - boundary.end > 15
                        or extra_chars + len(cue.text) > 360
                    ):
                        break
                    cues.append(cue)
                    extra_chars += len(cue.text)
            result.append(_passage(cues, scores[i]))
        return result


def rank(segments: list[Segment], query: str, limit: int = 8) -> list[Passage]:
    return PassageIndex(segments).rank(query, limit)


def navigation(video: Video, segments: list[Segment], max_items: int = 12) -> list[dict]:
    speech = [s for s in segments if s.kind != "ocr"]
    if not speech and not video.chapters:
        return []
    duration = max(video.duration, max((s.end for s in speech), default=0))
    if video.chapters:
        sections = [
            {
                "title": str(c.get("title", "")),
                "start": float(c.get("start_time", c.get("start", 0))),
                "end": float(c.get("end_time", c.get("end", duration))),
            }
            for c in video.chapters
        ]
    else:
        step = max(60, duration / max_items)
        sections = [
            {"title": "", "start": i * step, "end": min(duration, (i + 1) * step)}
            for i in range(max(1, math.ceil(duration / step)))
        ]
    # Sample chapters evenly if there are too many; do not silently retain only the beginning.
    if len(sections) > max_items:
        indices = [round(i * (len(sections) - 1) / (max_items - 1)) for i in range(max_items)]
        sections = [sections[i] for i in indices]
    result = []
    for section in sections:
        matches = [s for s in speech if s.start < section["end"] and s.end > section["start"]]
        # Extractive navigation is explicitly labeled, never disguised as a model summary.
        sample = " ".join(s.text for s in matches[:3])
        keywords = [
            t for t, _ in Counter(tokenize(" ".join(s.text for s in matches))).most_common(5)
        ]
        result.append(
            {
                "start": round(section["start"], 3),
                "end": round(section["end"], 3),
                "time": timestamp(section["start"]),
                "title": section["title"],
                "terms": keywords,
                "text": sample[:240],
                "url": source_link(video, section["start"]),
            }
        )
    return result

from vidlens.models import Segment, Video
from vidlens.retrieval import PassageIndex, passages, rank, source_link, tokenize


def test_chinese_identifiers_are_preserved():
    tokens = tokenize("KV缓存与函数 torch.nn.functional scaled_dot_product_attention C++")
    assert "缓存" in tokens
    assert "torch.nn.functional" in tokens
    assert "scaled_dot_product_attention" in tokens
    assert "c++" in tokens


def test_never_merge_ocr_with_spoken_evidence():
    items = passages(
        [
            Segment(0, 10, "spoken", kind="subtitle", id="a"),
            Segment(5, 5.001, "visual", kind="ocr", id="b"),
            Segment(10, 20, "more spoken", kind="subtitle", id="c"),
        ]
    )
    assert [p.kind for p in items] == ["subtitle", "ocr", "subtitle"]


def test_bilibili_links_keep_part_and_replace_time():
    video = Video(
        "test", "https://www.bilibili.com/video/BV1xx411c7mD/?p=3&t=99", "bilibili", "", 100
    )
    assert source_link(video, 25) == "https://www.bilibili.com/video/BV1xx411c7mD/?p=3&t=25"


def test_query_keeps_language_operators_and_signed_values_distinct():
    assert tokenize("β ≠ −1.5e-3") == ["β", "!=", "-1.5e-3"]
    inputs = [
        Segment(0, 4, "The C++ backend tests n != -1.", id="cpp"),
        Segment(100, 104, "The C# backend tests n == 1.", id="csharp"),
        Segment(200, 204, "The C backend tests n ≤ 2.", id="c"),
    ]
    index = PassageIndex(inputs)
    for query, expected in (("C++", "cpp"), ("C#", "csharp"), ("!= -1", "cpp"), ("<=", "c")):
        assert [row.evidence_ids for row in index.rank(query)] == [[expected]]


def test_focused_excerpt_keeps_answer_across_chunk_boundary():
    segments = [
        Segment(5 * i, 5 * (i + 1), "Preparing the next demonstration.", id=str(i))
        for i in range(24)
    ]
    segments[10].text = "Can we disable the timeout?"
    segments[11].text = "Only when the server advertises keepalive."
    segments[12].text = "Otherwise this could hang forever."
    [answer] = rank(segments, "disable the timeout")
    assert answer.start > 0
    assert "Only when" in answer.text and "Otherwise" in answer.text
    kept = [segments[int(cue_id)] for cue_id in answer.evidence_ids]
    assert answer.text == " ".join(cue.text for cue in kept)
    assert (answer.start, answer.end) == (kept[0].start, kept[-1].end)


def test_a_question_at_an_excerpt_boundary_keeps_its_answer():
    segments = [
        Segment(4 * i, 4 * i + 4, "Now continue the demonstration.", id=str(i)) for i in range(14)
    ]
    segments[5].text = "The policy is called backoff."
    segments[8].text = "Does it retry immediately?"
    segments[9].text = "Only after three consecutive failures."
    [answer] = rank(segments, "backoff")
    assert "Does it retry immediately? Only after three consecutive failures." in answer.text


def test_excerpts_do_not_duplicate_cues_or_cross_a_speech_gap():
    spoken = [
        Segment(5 * i, 5 * (i + 1), "A buffer example for the reader.", id=str(i))
        for i in range(24)
    ]
    spoken.append(Segment(300, 305, "A separate demonstration after the break.", id="distant"))
    items = rank(spoken, "buffer")
    ids = [cue_id for item in items for cue_id in item.evidence_ids]
    assert len(ids) == len(set(ids))
    assert "distant" not in ids


def test_new_index_observes_corrected_subtitles_and_return_values_are_independent():
    original = [Segment(15, 20, "The cache is enabled by default.", id="cue")]
    index = PassageIndex(original)
    first = index.rank("cache")[0]
    first.text = "changed by caller"
    first.evidence_ids.clear()
    assert index.rank("cache")[0].evidence_ids == ["cue"]
    assert "enabled" in index.rank("cache")[0].text
    corrected = [Segment(15, 20, "The cache is not enabled by default.", id="cue")]
    assert "not enabled" in PassageIndex(corrected).rank("cache")[0].text


def test_bilingual_queries_preserve_negative_qualification_and_ocr_provenance():
    data = [
        Segment(0, 6, "Latency falls with a warm cache.", source="manual", id="early"),
        Segment(
            90, 96, "However, latency does not fall on a cold start.", source="manual", id="later"
        ),
        Segment(100, 100.001, "時延 48 ms", kind="ocr", source="vision", id="screen"),
    ]
    result = PassageIndex(data).rank("时延")
    assert {cue for item in result for cue in item.evidence_ids} == {"early", "later", "screen"}
    assert any("does not" in item.text for item in result)
    assert any(item.kind == "ocr" and item.source == "vision" for item in result)


def test_adding_ocr_and_another_caption_track_does_not_move_speech_excerpts():
    core = [
        Segment(
            i * 5,
            i * 5 + 5,
            "Continuing the discussion.",
            source="manual",
            language="en",
            id=str(i),
        )
        for i in range(30)
    ]
    core[20].text = "This strategy is called backoff."
    core[21].text = "It means delaying the next retry."
    extras = [
        Segment(36, 36.001, "slide 2", kind="ocr", source="vision", id="screen"),
        Segment(75, 82, "另一种讲法。", source="manual", language="zh", id="translation"),
    ]
    [before] = rank(core, "backoff")
    [after] = rank(core + extras, "backoff")
    assert before.evidence_ids == after.evidence_ids
    assert before.text == after.text
    assert "It means delaying" in after.text

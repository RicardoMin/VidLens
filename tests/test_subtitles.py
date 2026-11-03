import json

import pytest

from vidlens.models import VidLensError
from vidlens.subtitles import parse_subtitles


def test_vtt_rolling_captions_keep_original_times_and_new_text():
    content = """WEBVTT

00:00:10.000 --> 00:00:13.000 align:start position:0%
<c>We cache the keys</c>

00:00:12.000 --> 00:00:15.000
We cache the keys and values

00:00:14.000 --> 00:00:17.000
and values on the GPU
"""
    result = parse_subtitles(content, "vtt", "automatic", "en")
    assert [(s.start, s.end, s.text) for s in result] == [
        (10, 13, "We cache the keys"),
        (12, 15, "and values"),
        (14, 17, "on the GPU"),
    ]
    assert all(s.language == "en" and s.source == "automatic" for s in result)


def test_repeated_speech_is_not_removed_across_time():
    content = """WEBVTT

00:00:01.000 --> 00:00:02.000
Yes, yes.

00:00:02.000 --> 00:00:03.000
Yes, yes.

00:00:15.000 --> 00:00:16.000
Yes, yes.
"""
    assert len(parse_subtitles(content, "vtt")) == 3


def test_duplicate_overlapping_window_is_only_emitted_once():
    content = json.dumps(
        [
            {"start": 3, "end": 6, "text": "缓存命中"},
            {"start": 4, "end": 7, "text": "缓存命中"},
            {"start": 6, "end": 9, "text": "缓存命中减少访存"},
        ]
    )
    result = parse_subtitles(content, "json")
    assert [(s.start, s.end, s.text) for s in result] == [(3, 7, "缓存命中"), (6, 9, "减少访存")]


def test_srt_accepts_comma_milliseconds_entities_and_multiline():
    result = parse_subtitles("1\n00:01:02,125 --> 00:01:04,250\n<b>KV &amp; Q</b>\n第二行\n", "srt")
    assert [(s.start, s.end, s.text) for s in result] == [(62.125, 64.25, "KV & Q 第二行")]


def test_json3_ignores_window_metadata_and_empty_events():
    payload = {
        "events": [
            {"tStartMs": 0, "wWinId": 1},
            {
                "tStartMs": 1500,
                "dDurationMs": 2300,
                "segs": [{"utf8": "Hello "}, {"utf8": "世界", "tOffsetMs": 700}],
            },
            {"tStartMs": 4000, "dDurationMs": 300, "segs": [{"utf8": "\n"}]},
        ]
    }
    result = parse_subtitles(json.dumps(payload), "json3", "automatic", "zh")
    assert [(s.start, s.end, s.text) for s in result] == [(1.5, 3.8, "Hello 世界")]


def test_bilibili_json_preserves_fractional_seconds():
    result = parse_subtitles(
        json.dumps({"body": [{"from": 92.04, "to": 95.92, "content": "分页注意力"}]}), "bilibili"
    )
    assert (result[0].start, result[0].end, result[0].text) == (92.04, 95.92, "分页注意力")


def test_programming_angle_brackets_survive_caption_cleanup():
    code = "std::vector<int> values; if (x < 3 && y > 2) f<T>();"
    vtt = f"WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<c.code>{code}</c>\n"
    assert parse_subtitles(vtt, "vtt")[0].text == code
    payload = json.dumps([{"start": 1, "end": 3, "text": "template<i> plus <b>"}])
    assert parse_subtitles(payload, "json")[0].text == "template<i> plus <b>"


def test_invalid_intervals_are_not_turned_into_evidence():
    rows = [
        {"start": -1, "end": 2, "text": "negative"},
        {"start": 2, "end": 1, "text": "backwards"},
        {"start": 0, "end": float("inf"), "text": "infinite"},
        {"start": 1, "duration": 2, "text": "valid"},
    ]
    result = parse_subtitles(json.dumps(rows), "json")
    assert len(result) == 1 and result[0].text == "valid"


@pytest.mark.parametrize(
    "content,fmt,code", [("not json", "json", "SUBTITLE_PARSE"), ("", "ttml", "SUBTITLE_FORMAT")]
)
def test_parser_errors_are_actionable(content, fmt, code):
    with pytest.raises(VidLensError) as error:
        parse_subtitles(content, fmt)
    assert error.value.code == code
    assert error.value.hint

import json
from pathlib import Path

import pytest

from vidlens.budget import compact_json
from vidlens.models import AcquiredVideo, Frame, Segment, Video, VidLensError
from vidlens.service import VideoService


@pytest.fixture
def service(tmp_path):
    return VideoService(tmp_path / "library")


def seed(service, count=80):
    video = Video(
        "lesson", "https://www.youtube.com/watch?v=aircAruvnKk", "youtube", "技术课程", count * 10
    )
    items = [
        Segment(
            i * 10,
            i * 10 + 9,
            f"第{i}段介绍计算过程。The intermediate result is stored in a tensor.",
            source="manual",
            language="zh",
        )
        for i in range(count)
    ]
    items[12].text = "KV cache 保存已经计算的 key 和 value，解码时不需要重复计算历史token。"
    items[13].text = "这个优化只适用于自回归解码。内存开销随着序列长度增长。"
    items[30].text = "The benchmark uses an A100 GPU and batch size 8. Baseline is eager execution."
    items[65].text = "限制：吞吐量提升不代表单请求延迟降低。The latency may increase."
    service.store.save(video, items, [])
    return video, items


def test_open_reuses_durable_cache_across_instances(service):
    video, segments = seed(service)

    class Provider:
        def acquire(self, *args, **kwargs):
            raise AssertionError("Network called despite cached source")

    other = VideoService(service.home)
    other._provider = Provider()
    result = other.open_video(video.url)
    assert result["cache_hit"]
    assert result["transcript"]["segments"] == len(segments)


def test_refresh_replaces_speech_atomically_invalidating_visual_evidence(service):
    video, _ = seed(service)
    service.store.add_segments(video.id, [Segment(5, 5.001, "Screen-only condition", kind="ocr")])

    class Provider:
        def acquire(self, *args, **kwargs):
            return AcquiredVideo(
                video, [Segment(0, 10, "New corrected transcript", source="manual")]
            )

    service._provider = Provider()
    service.open_video(video.url, refresh=True)
    assert len(service.store.segments(video.id)) == 1


@pytest.mark.parametrize(
    "query,needle",
    [
        ("KV缓存为什么能加速解码", "KV cache"),
        ("benchmark GPU batch size", "A100"),
        ("吞吐量和延迟的限制", "单请求"),
        ("基准测试 batch size", "A100"),
    ],
)
def test_bilingual_technical_retrieval(service, query, needle):
    seed(service)
    result = service.search("lesson", query, max_tokens=3000, limit=3)
    assert any(needle in item["text"] for item in result["evidence"])
    assert all("t=" in item["url"] for item in result["evidence"])


def test_no_match_does_not_claim_absence(service):
    seed(service)
    result = service.search("lesson", "unmentioned_potato_identifier")
    assert result["evidence"] == []
    assert "does not prove" in result["warnings"][0]


def test_actual_json_budget_and_pagination_no_skipped_segments(service):
    _, segments = seed(service)
    cursor = None
    seen = []
    while True:
        result = service.read_interval("lesson", 0, 800, max_tokens=1200, cursor=cursor)
        assert service.budget.count(compact_json(result)) <= 1200
        seen.extend(item["id"] for item in result["evidence"])
        if result["next_cursor"] is None:
            break
        assert result["next_cursor"] != cursor
        cursor = result["next_cursor"]
    assert seen == [s.id for s in segments]


def test_overview_keeps_ending_and_labels_extractive_samples(service):
    seed(service)
    result = service.overview("lesson", max_tokens=2200)
    assert "not a complete summary" in result["outline_method"]
    assert result["outline"][0]["start"] == 0
    assert result["outline"][-1]["end"] == 800
    assert service.budget.count(compact_json(result)) <= 2200


def test_coverage_distinguishes_indexed_delivered_and_sampled(service, tmp_path):
    seed(service)
    before = service.coverage("lesson")
    assert before["indexed_transcript_seconds"] > 0
    assert before["text_delivered_intervals"] == []
    service.search("lesson", "KV cache")
    assert service.coverage("lesson")["excerpt_delivered_intervals"]
    assert service.coverage("lesson")["text_delivered_intervals"] == []
    service.read_interval("lesson", 120, 140)
    after = service.coverage("lesson")
    assert after["text_delivered_intervals"] == [[120.0, 129.0], [130.0, 139.0]]
    assert after["frame_timestamps_delivered"] == []


def test_ocr_search_is_point_evidence_not_speech(service):
    seed(service)
    service.store.add_segments(
        "lesson",
        [Segment(222, 222.001, "Hidden chart: speedup=2.7x", kind="ocr", source="local_ocr")],
    )
    result = service.search("lesson", "speedup")
    assert result["evidence"][0]["kind"] == "ocr"
    assert result["evidence"][0]["start"] == 222


def test_import_timed_json_and_source_media(service, tmp_path):
    path = tmp_path / "lesson.json"
    path.write_text(json.dumps({"segments": [{"start": 0, "end": 10, "text": "批大小为8。"}]}))
    result = service.import_transcript(path, title="真实标题")
    assert result["title"] == "真实标题"
    assert service.search(result["video_id"], "批大小")["evidence"]


@pytest.mark.parametrize(
    "start,end", [(float("nan"), 10), (0, float("inf")), (-1, 5), (5, 5), (800, 900)]
)
def test_invalid_temporal_requests_are_rejected(service, start, end):
    seed(service)
    with pytest.raises(VidLensError):
        service.read_interval("lesson", start, end)


def test_store_failure_does_not_destroy_cached_transcript(service):
    video, segments = seed(service)
    invalid = Video(
        video.id,
        video.url,
        video.platform,
        video.title,
        800,
        metadata={"bad": Path("unserializable")},
    )
    with pytest.raises(TypeError):
        service.store.save(invalid, [], [])
    assert len(service.store.segments(video.id)) == len(segments)


def test_frames_cache_and_ocr_retrieval(service, tmp_path):
    video, _ = seed(service)
    media = tmp_path / "media.mp4"
    media.write_bytes(b"fixture")
    video.metadata["local_media_path"] = str(media)
    service.store.save(video, [], [], replace=False)
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"fixture")

    class Visual:
        def __init__(self):
            self.last_warnings = []

        def read(self, *args, **kwargs):
            return [Frame(42, frame_path, 1280, 720, "CUDA_VISIBLE_DEVICES=0")]

    service._visual = Visual()
    result = service.read_frames("lesson", 40, 50)
    assert result["frames"][0]["timestamp"] == 42
    assert service.search("lesson", "CUDA_VISIBLE_DEVICES")["evidence"][0]["kind"] == "ocr"
    assert service.coverage("lesson")["frame_timestamps_delivered"] == [42]


def test_long_caption_paginates_every_character_without_silent_loss(service):
    text = "很长的字幕中的每一个条件都应该保留。" * 300
    service.store.save(Video("long", "", "local", "Long cue", 10), [Segment(0, 10, text)], [])
    cursor, recovered = None, ""
    while True:
        page = service.read_interval("long", 0, 10, max_tokens=900, cursor=cursor)
        for item in page["evidence"]:
            recovered += item["text"][:-1] if item.get("excerpt_truncated") else item["text"]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert recovered == text


def test_cursor_rejects_changed_evidence(service):
    seed(service)
    page = service.read_interval("lesson", 0, 800, max_tokens=1200)
    service.store.add_segments("lesson", [Segment(15, 16, "new OCR", kind="ocr")])
    with pytest.raises(VidLensError, match="cursor"):
        service.read_interval("lesson", 0, 800, max_tokens=1200, cursor=page["next_cursor"])


def test_simplified_query_matches_traditional_captions(service):
    service.store.save(
        Video("traditional", "", "local", "中文", 10),
        [Segment(0, 10, "這個神經網路的輸入包含784個神經元。")],
        [],
    )
    result = service.search("traditional", "神经网络 输入 神经元")
    assert result["evidence"]
    assert "神經" in result["evidence"][0]["text"]


def test_one_call_reader_is_bounded_and_preserves_original_evidence(service):
    video, _ = seed(service)
    result = service.read_video(video.url, question="KV cache", max_tokens=1200)
    assert result["mode"] == "question_evidence"
    assert result["transcript_complete"] is False
    assert result["source_url"] == video.url
    assert any("KV cache" in item["text"] for item in result["evidence"])
    assert service.budget.count(compact_json(result)) <= 1200


def test_whole_video_reader_does_not_call_sample_complete(service):
    video, _ = seed(service)
    result = service.read_video(video.url, max_tokens=1800)
    assert result["mode"] == "sampled_transcript"
    assert not result["transcript_complete"]
    assert result["evidence"][0]["start"] == 0
    assert result["evidence"][-1]["end"] == 799


def test_small_transcript_can_be_read_in_full_once(service):
    video = Video("short", "https://www.youtube.com/watch?v=aircAruvnKk", "youtube", "Short", 20)
    service.store.save(
        video, [Segment(0, 10, "A small lesson."), Segment(10, 20, "An important caveat.")], []
    )
    result = service.read_video(video.url, max_tokens=2000)
    assert result["transcript_complete"]
    assert result["mode"] == "complete_transcript"
    assert "An important caveat" in result["evidence"][0]["text"]


def test_open_language_preference_change_refreshes_evidence(service):
    video, _ = seed(service)
    video.metadata["language_preferences"] = ["zh", "en"]
    service.store.save(video, [], [], replace=False)
    calls = []

    class Provider:
        def acquire(self, url, languages):
            calls.append(languages)
            return AcquiredVideo(video, [Segment(0, 5, "English caption.", language="en")])

    service._provider = Provider()
    result = service.open_video(video.url, languages=("en",))
    assert calls == [("en",)]
    assert not result["cache_hit"]


def test_author_correction_is_exposed_separately_from_speech(service):
    video = Video(
        "errata",
        "https://www.youtube.com/watch?v=aircAruvnKk",
        "youtube",
        "Lesson",
        20,
        description="Some description.\nCorrection 14:45: index should be k.\nMore links.",
    )
    service.store.save(video, [Segment(0, 20, "Original lecture")], [])
    result = service.read_video(video.url, question="index")
    assert result["source_notes"] == ["Correction 14:45: index should be k."]
    assert not result["evidence"]


def test_partial_asr_is_never_labeled_complete_video(service):
    video = Video(
        "partial",
        "https://www.youtube.com/watch?v=aircAruvnKk",
        "youtube",
        "Long",
        600,
        metadata={"transcribed_intervals": [[60, 68]]},
    )
    service.store.save(
        video,
        [Segment(60, 68, "Locally recognized speech", kind="transcript", source="whisper:base")],
        ["No captions. This is metadata only; use transcription."],
    )
    result = service.read_video(video.url)
    assert not result["transcript_complete"]
    assert result["mode"] == "partial_transcript"
    assert result["speech_scope"]["processed_intervals"] == [[60, 68]]
    assert not any("metadata only" in warning for warning in result["warnings"])


def test_transcription_reuses_cached_recognition_and_tracks_processed_interval(service, tmp_path):
    from vidlens.models import MediaAsset

    video = Video("asr", "", "local", "Long", 600)
    service.store.save(video, [], [])
    calls = []

    class Provider:
        def transcribe(self, asset, language, model):
            calls.append(asset)
            return [Segment(60, 68, "Recognized speech", kind="transcript", source="whisper:base")]

    service._provider = Provider()
    service._asset = lambda *a, **k: MediaAsset(tmp_path / "clip.wav", 60, 8, True)
    assert not service.transcribe("asr", start=60, end=68, language="zh")["cache_hit"]
    assert service.transcribe("asr", start=60, end=68, language="zh")["cache_hit"]
    assert len(calls) == 1
    assert service.store.get("asr").metadata["transcribed_intervals"] == [[60.0, 68.0]]

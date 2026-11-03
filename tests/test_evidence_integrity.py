"""End-to-end invariants at the evidence, storage and media boundaries."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from vidlens.models import AcquiredVideo, Frame, MediaAsset, Segment, Video
from vidlens.retrieval import rank
from vidlens.service import VideoService


def test_retrieval_keeps_later_negation_of_same_claim():
    segments = [
        Segment(0, 4, "The method is safe for production.", id="early-claim"),
        Segment(90, 94, "The method is not safe for production.", id="later-correction"),
    ]
    evidence = rank(segments, "method safe production", limit=6)
    assert len(evidence) == 2, "Near-duplicate removal must not erase a contradictory later claim."


def test_no_captions_still_exposes_platform_chapters(tmp_path):
    service = VideoService(tmp_path)
    video = Video(
        "lecture",
        "",
        "local",
        "Lecture",
        120,
        chapters=[
            {"title": "Setup", "start": 0, "end": 60},
            {"title": "Results", "start": 60, "end": 120},
        ],
    )
    service.store.save(video, [], [])
    overview = service.overview(video.id)
    assert [item["title"] for item in overview["outline"]] == ["Setup", "Results"]


def test_refresh_does_not_keep_obsolete_visual_evidence(tmp_path):
    service = VideoService(tmp_path)
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    video = Video("youtube-jNQXAC9IVRw", url, "youtube", "Old edit", 100)
    service.store.save(video, [Segment(0, 5, "Old introduction")], [])
    frame = tmp_path / "old-frame.jpg"
    frame.write_bytes(b"cached image")
    service.store.save_frames(video.id, [Frame(90, frame, 400, 200, ocr="OBSOLETE FIGURE")])
    service.store.delivered(video.id, [{"start": 90, "end": 99, "id": "old"}], "text")

    class ChangedSource:
        def acquire(self, url, languages):
            return AcquiredVideo(
                Video(video.id, url, "youtube", "Edited video", 10),
                [Segment(0, 5, "New introduction")],
            )

    service._provider = ChangedSource()
    service.open_video(url, refresh=True)
    assert service.search(video.id, "OBSOLETE")["evidence"] == []
    assert service.coverage(video.id)["text_delivered_intervals"] == []


def test_parallel_media_requests_publish_one_reusable_clip(tmp_path):
    services = [VideoService(tmp_path), VideoService(tmp_path)]
    video = Video("lecture", "", "local", "Lecture", 60)
    calls = []
    entered = threading.Barrier(2)

    class Provider:
        def fetch_media(self, video, destination, start, end, audio_only):
            calls.append((start, end))
            destination.write_bytes(b"video")
            return MediaAsset(destination, start, end - start, audio_only)

    def get_clip(service):
        service._provider = Provider()
        entered.wait(timeout=5)
        return service._asset(video, 2, 8)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assets = list(pool.map(get_clip, services))
    nested = services[0]._asset(video, 4, 6)
    assert nested == assets[0] == assets[1]
    assert calls == [(2, 8)]
    # A revision change cannot serve a previous edit's frames.
    video.metadata["revision"] = "edited"
    assert services[0]._asset(video, 4, 6).path != nested.path
    assert calls[-1] == (4, 6)


def test_concurrent_crop_preserves_frame_read_warning(tmp_path):
    service = VideoService(tmp_path)
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fixture")
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"fixture")
    frame = Frame(2, frame_path, 400, 200)
    service.store.save(
        Video("lecture", "", "local", "Lecture", 60, metadata={"local_media_path": str(media)}),
        [],
        [],
    )
    service.store.save_frames("lecture", [frame])
    reading, resume, cropped = threading.Event(), threading.Event(), threading.Event()

    class Visual:
        def __init__(self):
            self.last_warnings = []

        def read(self, *args, **kwargs):
            self.last_warnings = ["OCR failed; inspect images directly."]
            reading.set()
            assert resume.wait(timeout=5)
            return [frame]

        def crop(self, *args, **kwargs):
            self.last_warnings = []
            cropped.set()
            return frame

    service._visual = Visual()
    with ThreadPoolExecutor(max_workers=2) as pool:
        frame_result = pool.submit(service.read_frames, "lecture", 0, 5)
        assert reading.wait(timeout=5)
        crop_result = pool.submit(service.crop_frame, "lecture", 2, (0, 0, 1, 1))
        cropped.wait(timeout=0.2)
        resume.set()
        returned = frame_result.result(timeout=5)
        crop_result.result(timeout=5)
    assert "OCR failed; inspect images directly." in returned["warnings"]


def test_long_query_echo_does_not_displace_all_found_evidence(tmp_path):
    service = VideoService(tmp_path)
    service.budget.encoding = None
    service.budget.method = "utf8_bytes_upper_bound"
    service.store.save(
        Video("lecture", "", "local", "Lecture", 60),
        [Segment(0, 4, "The benchmark uses batch size 8.")],
        [],
    )
    query = "batch size " + "Please inspect every relevant condition in this experiment. " * 12
    result = service.search("lecture", query, max_tokens=768)
    assert result["evidence"], "Trim the echoed query before deleting the only relevant evidence."


def test_reader_notices_other_process_writes_and_preserves_middle_coverage(tmp_path):
    first = VideoService(tmp_path)
    second = VideoService(tmp_path)
    video = Video("lesson", "https://youtu.be/aircAruvnKk", "youtube", "Lecture", 2400)
    segments = [
        Segment(i * 60, i * 60 + 4, f"section {i} " + "demonstration " * 12) for i in range(40)
    ]
    first.store.save(video, segments, [])
    before = first.read_video(video.url, max_tokens=1800)
    starts = [e["start"] for e in before["evidence"]]
    assert before["mode"] == "sampled_transcript"
    assert any(800 < t < 1600 for t in starts)
    assert not first.search(video.id, "matrix_kernel")["evidence"]
    second.store.add_segments(video.id, [Segment(900, 900.001, "matrix_kernel", kind="ocr")])
    assert first.search(video.id, "matrix_kernel")["evidence"][0]["kind"] == "ocr"


def test_ocr_only_result_never_claims_a_complete_transcript(tmp_path):
    service = VideoService(tmp_path)
    video = Video("screens", "https://youtu.be/aircAruvnKk", "youtube", "Slides", 60)
    service.store.save(video, [Segment(22, 22.001, "Figure 1", kind="ocr")], [])
    read = service.read_video(video.url)
    assert read["mode"] == "visual_evidence"
    assert not read["transcript_complete"]


def test_reopening_cached_short_url_works_without_resolving_redirect(tmp_path, monkeypatch):
    from vidlens import sources

    service = VideoService(tmp_path)
    short_url = "https://b23.tv/abc"
    full_url = "https://www.bilibili.com/video/BV1SmrSBfEg6/?p=2"

    class Provider:
        def acquire(self, url, languages):
            return AcquiredVideo(
                Video("p2", full_url, "bilibili", "Part two", 30), [Segment(0, 4, "test")]
            )

    service._provider = Provider()
    monkeypatch.setattr(sources, "normalize_url", lambda _: full_url)
    service.open_video(short_url)

    def offline(_):
        raise AssertionError("Warm short link attempted network access")

    monkeypatch.setattr(sources, "normalize_url", offline)
    assert VideoService(tmp_path).open_video(short_url)["cache_hit"]


def test_overlapping_asr_reads_only_the_missing_audio_and_retries_a_failed_gap(tmp_path):
    service = VideoService(tmp_path)
    service.store.save(Video("talk", "", "local", "Talk", 100), [], [])
    calls = []

    def clip(_video, start, end, audio_only):
        calls.append((start, end))
        return MediaAsset(tmp_path / "audio", start, end - start, audio_only)

    class Provider:
        fail = True

        def transcribe(self, asset, language, model):
            if asset.offset == 30 and self.fail:
                self.fail = False
                raise OSError("unavailable")
            return [
                Segment(
                    asset.offset,
                    asset.offset + asset.duration,
                    str(asset.offset),
                    kind="transcript",
                    source="whisper:base",
                )
            ]

    service._asset = clip
    service._provider = Provider()
    service.transcribe("talk", 20, 30, "en")
    import pytest

    with pytest.raises(OSError):
        service.transcribe("talk", 10, 40, "en")
    service.transcribe("talk", 10, 40, "en")
    assert service.transcribe("talk", 15, 32, "en")["cache_hit"]
    assert calls == [(20, 30), (10, 20), (30, 40), (30, 40)]
    assert service.store.get("talk").metadata["transcribed_intervals"] == [[10, 40]]

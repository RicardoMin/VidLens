"""Interval identities and publication metadata must agree with actual cached bytes."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

from vidlens.media_cache import MediaCache
from vidlens.models import MediaAsset, Video, VidLensError
from vidlens.service import VideoService


def setup(tmp_path):
    service = VideoService(tmp_path / "library")
    video = Video("remote", "", "youtube", "Lecture", 1200, metadata={"revision": "a"})
    attempts = []

    class Provider:
        def fetch_media(self, video, target, *, start, end, audio_only):
            attempts.append((start, end))
            target.write_text(f"source pixels {start}:{end}")
            return MediaAsset(target, start, end - start, audio_only)

    return service, video, attempts, Provider


def test_precise_intervals_never_overwrite_another_cached_clip(tmp_path):
    service, video, attempts, provider = setup(tmp_path)
    first = service.media.get(video, 1.0001, 2.0001, False, provider)
    second = service.media.get(video, 1.0002, 2.0002, False, provider)
    again = service.media.get(video, 1.0001, 2.0001, False, provider)
    assert first.path != second.path
    assert first.path.read_text() == "source pixels 1.0001:2.0001"
    assert second.path.read_text() == "source pixels 1.0002:2.0002"
    assert again == first
    assert attempts == [(1.0001, 2.0001), (1.0002, 2.0002)]


def legacy_asset(service, video, *, declared_start=1.0001, declared_end=2.0001):
    identity = f"{video.id}|a|1.000|2.000|False"
    directory = service.media.root / hashlib.sha256(identity.encode()).hexdigest()[:24]
    directory.mkdir(parents=True)
    path = directory / "clip.mp4"
    path.write_text("legacy source pixels")
    (directory / "asset.json").write_text(
        json.dumps(
            {
                "path": str(path),
                "offset": declared_start,
                "duration": declared_end - declared_start,
                "audio_only": False,
            }
        )
    )
    return path


def test_legacy_manifest_is_reused_without_rewriting_when_it_covers_request(tmp_path):
    service, video, attempts, provider = setup(tmp_path)
    path = legacy_asset(service, video)
    result = service.media.get(video, 1.0001, 2.0001, False, provider)
    assert result.path == path and result.offset == 1.0001
    assert attempts == []
    # A neighboring precise request must not overwrite the legacy file either.
    other = service.media.get(video, 1.0002, 2.0002, False, provider)
    assert other.path != path
    assert path.read_text() == "legacy source pixels"


def test_legacy_database_row_cannot_mislabel_an_overwritten_clip(tmp_path):
    service, video, attempts, provider = setup(tmp_path)
    path = legacy_asset(service, video, declared_start=1.0002, declared_end=2.0002)
    with service.store.connect() as db:
        db.execute(
            "INSERT INTO media_assets VALUES (?,?,?,?,?,?)",
            (
                video.id,
                "a",
                "1080",
                1.0001,
                2.0001,
                str(path),
            ),
        )
    result = service.media.get(video, 1.0001, 2.0001, False, provider)
    assert result.path != path
    assert result.offset == 1.0001
    assert result.path.read_text() == "source pixels 1.0001:2.0001"
    assert attempts == [(1.0001, 2.0001)]


def test_modified_cached_bytes_trigger_reacquisition(tmp_path):
    service, video, attempts, provider = setup(tmp_path)
    result = service.media.get(video, 0, 10, False, provider)
    result.path.write_bytes(b"truncated")
    result = service.media.get(video, 0, 10, False, provider)
    assert result.path.read_text() == "source pixels 0:10"
    assert attempts == [(0, 10), (0, 10)]


def test_concurrent_failed_full_prefetch_is_attempted_once(tmp_path):
    service, video, attempts, _ = setup(tmp_path)

    class Provider:
        def fetch_media(self, *args, **kwargs):
            attempts.append(1)
            time.sleep(0.05)
            raise VidLensError("unavailable", "This source cannot be prefetched")

    caches = [MediaCache(service.store) for _ in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda cache: cache.optional_full_video(video, Provider), caches))
    assert results == [None, None, None]
    assert attempts == [1]

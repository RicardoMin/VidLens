import json
import sys
import wave
from pathlib import Path

import httpx
import pytest

from vidlens.models import MediaAsset, Video, VidLensError
from vidlens.sources import SourceProvider, _decode_asr_audio, normalize_url, video_identity

YT = "https://www.youtube.com/watch?v=aircAruvnKk"
BILI = "https://www.bilibili.com/video/BV1xx411c7mD/?p=2"


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/aircAruvnKk?t=12",
        "https://m.youtube.com/watch?v=aircAruvnKk&list=xyz",
        "https://www.youtube.com/shorts/aircAruvnKk",
    ],
)
def test_youtube_url_variants_share_identity(url):
    assert normalize_url(url) == YT
    assert video_identity(url) == "youtube:aircAruvnKk"


def test_bilibili_parts_have_distinct_stable_identities():
    assert normalize_url("https://bilibili.com/video/BV1xx411c7mD?p=02&share_source=x") == BILI
    assert video_identity(BILI).endswith(":p2")
    assert video_identity(BILI) != video_identity(BILI.replace("p=2", "p=1"))


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/video",
        "https://youtube.com.evil.test/watch?v=aircAruvnKk",
        "https://user:pass@youtube.com/watch?v=aircAruvnKk",
        "https://127.0.0.1/watch?v=aircAruvnKk",
        "https://www.youtube.com/playlist?list=abc",
        "https://www.bilibili.com/video/BV1xx411c7mD?p=0",
    ],
)
def test_rejects_untrusted_or_non_video_urls(url):
    with pytest.raises(VidLensError):
        normalize_url(url)


def test_short_link_redirect_cannot_escape_platforms(monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
    )
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    with pytest.raises(VidLensError) as error:
        normalize_url("https://b23.tv/abcdef")
    assert error.value.code == "UNSUPPORTED_URL"


def test_short_link_resolves_and_preserves_part(monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(302, headers={"location": BILI}))
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    assert normalize_url("https://b23.tv/abcdef") == BILI


def test_caption_host_redirect_is_validated(tmp_path, monkeypatch):
    provider = SourceProvider(tmp_path)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            302, headers={"location": "https://private.example/captions"}
        )
    )
    monkeypatch.setattr(provider, "_client", lambda: httpx.Client(transport=transport))
    with pytest.raises(VidLensError) as error:
        provider._get("https://www.youtube.com/api/timedtext")
    assert error.value.code == "SOURCE_HOST"


def test_native_captions_prefer_requested_language_without_asr(tmp_path, monkeypatch):
    provider = SourceProvider(tmp_path)
    info = {
        "title": "Lecture",
        "duration": 60,
        "subtitles": {
            "en": [{"ext": "srt", "data": "1\n00:00:01,000 --> 00:00:03,000\nEnglish\n"}]
        },
        "automatic_captions": {
            "zh-Hans": [{"ext": "json3", "url": "https://www.youtube.com/caption"}]
        },
    }
    monkeypatch.setattr(provider, "_yt_info", lambda _: info)
    monkeypatch.setattr(
        provider,
        "_get",
        lambda _: json.dumps(
            {"events": [{"tStartMs": 2000, "dDurationMs": 3000, "segs": [{"utf8": "缓存键值"}]}]}
        ),
    )
    monkeypatch.setattr(
        provider, "_youtube_transcript", lambda *_: pytest.fail("native captions already available")
    )
    acquired = provider.acquire(YT)
    assert acquired.video.title == "Lecture"
    assert acquired.segments[0].text == "缓存键值"
    assert acquired.segments[0].source == "automatic"
    assert acquired.segments[0].language == "zh-Hans"


def test_missing_captions_is_reported_as_metadata_only(tmp_path, monkeypatch):
    provider = SourceProvider(tmp_path)
    monkeypatch.setattr(
        provider, "_yt_info", lambda _: {"title": "Silent demonstration", "duration": 10}
    )

    def fail(*_):
        raise VidLensError("SOURCE_FAILED", "No captions")

    monkeypatch.setattr(provider, "_youtube_transcript", fail)
    acquired = provider.acquire(YT)
    assert acquired.segments == []
    assert any("metadata only" in warning for warning in acquired.warnings)


def test_bilibili_native_caption_uses_requested_cid(tmp_path, monkeypatch):
    provider = SourceProvider(tmp_path)
    calls = []

    def api(url, params=None):
        calls.append((url, params))
        if "web-interface/view" in url:
            return {
                "code": 0,
                "data": {
                    "bvid": "BV1xx411c7mD",
                    "aid": 123,
                    "title": "课程",
                    "pages": [
                        {"page": 1, "cid": 111, "duration": 10},
                        {"page": 2, "cid": 222, "duration": 20, "part": "缓存"},
                    ],
                },
            }
        assert params["cid"] == 222
        return {
            "code": 0,
            "data": {
                "subtitle": {
                    "subtitles": [
                        {"lan": "zh-CN", "subtitle_url": "https://i0.hdslb.com/subtitle.json"}
                    ]
                }
            },
        }

    monkeypatch.setattr(provider, "_json", api)
    monkeypatch.setattr(
        provider,
        "_get",
        lambda _: json.dumps({"body": [{"from": 1, "to": 3, "content": "第二集内容"}]}),
    )
    acquired = provider.acquire(BILI)
    assert acquired.video.id == "bilibili:BV1xx411c7mD:p2"
    assert acquired.video.duration == 20
    assert acquired.video.metadata["cid"] == 222
    assert acquired.segments[0].text == "第二集内容"


def test_bilibili_fallback_rejects_wrong_part(tmp_path, monkeypatch):
    provider = SourceProvider(tmp_path)

    def fail(*_):
        raise VidLensError("SOURCE_ACCESS", "API unavailable")

    monkeypatch.setattr(provider, "_json", fail)
    monkeypatch.setattr(
        provider, "_yt_info", lambda _: {"id": "BV1xx411c7mD_p1", "title": "wrong part"}
    )
    with pytest.raises(VidLensError) as error:
        provider.acquire(BILI)
    assert error.value.code == "PART_UNVERIFIED"


def test_bilibili_inline_ytdlp_captions_are_supported(tmp_path):
    provider = SourceProvider(tmp_path)
    segments = provider._captions_from_info(
        {
            "subtitles": {
                "zh-CN": [{"ext": "srt", "data": "1\n00:00:01,000 --> 00:00:03,000\n字幕\n"}]
            }
        },
        ("zh", "en"),
    )
    assert segments[0].text == "字幕"


def test_download_interval_keeps_original_timeline(tmp_path, monkeypatch):
    monkeypatch.delenv("VIDLENS_MAX_VIDEO_HEIGHT", raising=False)
    provider = SourceProvider(tmp_path / "cache")
    commands = []

    def run(command, timeout):
        commands.append(command)
        template = command[command.index("-o") + 1]
        media = Path(template.replace("%(ext)s", "webm"))
        media.write_bytes(b"test fixture")
        return str(media)

    monkeypatch.setattr(provider, "_run", run)
    monkeypatch.setattr(provider, "_ffmpeg", lambda: "/fake/ffmpeg")
    video = Video("youtube:aircAruvnKk", YT, "youtube", "Lecture", 100)
    asset = provider.fetch_media(video, tmp_path / "media" / "clip.mp4", 12.5, 18.5)
    assert asset.offset == 12.5 and asset.duration == 6
    assert asset.path.suffix == ".webm"
    assert "*12.5-18.5" in commands[0]
    assert "--force-keyframes-at-cuts" in commands[0]
    assert "--cookies-from-browser" not in commands[0]
    assert (
        commands[0][commands[0].index("-f") + 1]
        == "bestvideo[height<=1080]/best[height<=1080]/best"
    )


def test_transcription_offsets_are_added_once(tmp_path, monkeypatch):
    media = tmp_path / "part.webm"
    media.write_bytes(b"fixture")
    provider = SourceProvider(tmp_path)
    monkeypatch.setattr(
        provider, "_run", lambda *_args, **_kwargs: '[{"start":1.25,"end":3.5,"text":"字幕"}]'
    )
    result = provider.transcribe(MediaAsset(media, offset=120, duration=10), language="zh")
    assert (result[0].start, result[0].end) == (121.25, 123.5)
    assert result[0].source == "whisper:base"


def test_asr_cannot_extend_evidence_past_known_media_interval(tmp_path, monkeypatch):
    media = tmp_path / "clip.m4a"
    media.write_bytes(b"fixture")
    provider = SourceProvider(tmp_path)
    monkeypatch.setattr(
        provider,
        "_run",
        lambda *_args, **_kwargs: (
            '[{"start":4.88,"end":8.04,"text":"末尾"},{"start":8.04,"end":8.2,"text":"padding"}]'
        ),
    )
    segments = provider.transcribe(MediaAsset(media, offset=60, duration=8, audio_only=True))
    assert len(segments) == 1
    assert (segments[0].start, segments[0].end) == (64.88, 68)


def test_transcription_downloads_models_inside_configured_cache(tmp_path, monkeypatch):
    media = tmp_path / "clip.m4a"
    media.write_bytes(b"fixture")
    provider = SourceProvider(tmp_path / "source-cache")
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return "[]"

    monkeypatch.setattr(provider, "_run", run)
    provider.transcribe(MediaAsset(media, audio_only=True))
    assert Path(commands[0][-1]) == tmp_path / "source-cache" / "models" / "faster-whisper"


def test_no_audio_error_explains_how_to_get_transcribable_media(tmp_path):
    provider = SourceProvider(tmp_path)
    with pytest.raises(VidLensError) as error:
        provider._run([sys.executable, "-c", "raise SystemExit('VIDLENS_NO_AUDIO')"])
    assert error.value.code == "ASR_NO_AUDIO"
    assert "audio_only=True" in error.value.hint


@pytest.mark.parametrize(
    "diagnostic,code",
    [
        ("ModuleNotFoundError: No module named faster_whisper", "ASR_UNAVAILABLE"),
        ("huggingface_hub snapshot_download ProxyError: 503", "ASR_MODEL_DOWNLOAD"),
        ("RuntimeError: audio decode failed", "ASR_FAILED"),
    ],
)
def test_asr_errors_have_model_specific_remediation(tmp_path, monkeypatch, diagnostic, code):
    class FailedProcess:
        returncode = 1

        def communicate(self, **_kwargs):
            return "", diagnostic

    monkeypatch.setattr(
        "vidlens.sources.subprocess.Popen", lambda *_args, **_kwargs: FailedProcess()
    )
    provider = SourceProvider(tmp_path)
    with pytest.raises(VidLensError) as error:
        provider._run([sys.executable, "-m", "vidlens.sources", "whisper"])
    assert error.value.code == code
    assert "yt-dlp" not in error.value.hint


def test_process_timeout_becomes_structured_error(tmp_path):
    provider = SourceProvider(tmp_path)
    with pytest.raises(VidLensError) as error:
        provider._run([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.05)
    assert error.value.code == "SOURCE_TIMEOUT"


def test_asr_decoder_outputs_normalized_16khz_mono_without_pyav_decode_api(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "audio.wav"
    samples = np.full((4000, 2), 16384, dtype=np.int16)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples.tobytes())
    decoded = _decode_asr_audio(path)
    assert decoded.shape == (4000,)
    assert decoded.dtype == np.float32
    assert np.allclose(decoded, 0.5)


def test_system_proxy_is_forwarded_to_ffmpeg_process_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.setattr("vidlens.sources.getproxies", lambda: {"http": "http://proxy.example:3128"})
    provider = SourceProvider(tmp_path)
    result = provider._run([sys.executable, "-c", "import os; print(os.environ['http_proxy'])"])
    assert result.strip() == "http://proxy.example:3128"


def test_process_errors_do_not_expose_signed_urls(tmp_path):
    provider = SourceProvider(tmp_path)
    with pytest.raises(VidLensError) as error:
        provider._run(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('403 https://media.test/?signature=SECRET'); sys.exit(1)",
            ]
        )
    assert "SECRET" not in str(error.value.to_dict())
    assert "cookie" in error.value.hint

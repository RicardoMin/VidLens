import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from vidlens.cli import build_parser, dispatch
from vidlens.mcp_server import create_server
from vidlens.models import MediaAsset, Segment, Video, VidLensError
from vidlens.service import VideoService
from vidlens.sources import SourceProvider, normalize_vocabulary


def setup_service(tmp_path):
    service = VideoService(tmp_path / "library")
    video = Video("asr", "", "local", "Speech", 100)
    service.store.save(video, [], [])
    service._asset = lambda _video, start, end, audio_only: MediaAsset(
        tmp_path / "audio.wav", start, end - start, audio_only
    )
    calls = []

    class Provider:
        fail = False

        def transcribe(self, asset, language, model, vocabulary=None):
            if self.fail:
                raise VidLensError("ASR_FAILED", "Recognition failed.")
            calls.append((asset.offset, asset.duration, language, model, vocabulary or []))
            text = f"{model} recognizes {', '.join(vocabulary or ['generic vocabulary'])}"
            return [
                Segment(
                    asset.offset,
                    asset.offset + asset.duration,
                    text,
                    kind="transcript",
                    source=f"whisper:{model}",
                    language=language or "",
                )
            ]

    service._provider = Provider()
    return service, video, calls


def test_new_model_replaces_active_asr_and_old_history_only_inside_successful_range(tmp_path):
    service, video, calls = setup_service(tmp_path)
    service.transcribe(video.id, 0, 20, "zh", "base")
    service.transcribe(video.id, 20, 40, "zh", "base")
    service.transcribe(video.id, 40, 60, "zh", "base")
    service.transcribe(video.id, 20, 40, "zh", "small")
    assert [(s.start, s.end, s.source) for s in service.store.segments(video.id)] == [
        (0, 20, "whisper:base"),
        (20, 40, "whisper:small"),
        (40, 60, "whisper:base"),
    ]
    assert service.transcribe(video.id, 20, 40, "zh", "small")["cache_hit"]
    assert not service.transcribe(video.id, 20, 40, "zh", "base")["cache_hit"]
    assert len(calls) == 5


def test_vocabulary_changes_cache_signature_and_refresh_recomputes_same_configuration(tmp_path):
    service, video, calls = setup_service(tmp_path)
    generic = service.transcribe(video.id, 10, 20, "zh")
    guided = service.transcribe(video.id, 10, 20, "zh", vocabulary=[" CUDA ", "KV cache", "CUDA"])
    assert generic["signature"] != guided["signature"]
    assert len(service.store.segments(video.id)) == 1
    assert "CUDA" in service.store.segments(video.id)[0].text
    assert service.transcribe(video.id, 10, 20, "zh", vocabulary=["CUDA", "KV cache"])["cache_hit"]
    assert not service.transcribe(
        video.id, 10, 20, "zh", vocabulary=["CUDA", "KV cache"], refresh=True
    )["cache_hit"]
    assert len(calls) == 3
    assert calls[-1][-1] == ["CUDA", "KV cache"]


def test_failed_replacement_preserves_old_transcript_and_cache_history(tmp_path):
    service, video, _ = setup_service(tmp_path)
    service.transcribe(video.id, 10, 20, "zh")
    old_segments = service.store.segments(video.id)
    old_metadata = service.store.get(video.id).metadata
    service._provider.fail = True
    with pytest.raises(VidLensError, match="Recognition failed"):
        service.transcribe(video.id, 10, 20, "zh", model="small", vocabulary=["FP8"], refresh=True)
    assert service.store.segments(video.id) == old_segments
    assert service.store.get(video.id).metadata == old_metadata


def test_replacement_keeps_crossing_cues_native_captions_ocr_and_other_languages(tmp_path):
    service, video, _ = setup_service(tmp_path)
    old = [
        Segment(5, 15, "crossing left", kind="transcript", source="whisper:base", language="zh"),
        Segment(11, 19, "obsolete inside", kind="transcript", source="whisper:base", language="zh"),
        Segment(18, 25, "crossing right", kind="transcript", source="whisper:base", language="zh"),
        Segment(11, 19, "native caption", kind="subtitle", source="manual", language="zh"),
        Segment(12, 12.001, "screen text", kind="ocr", source="local_ocr"),
        Segment(11, 19, "English track", kind="transcript", source="whisper:base", language="en"),
    ]
    video.metadata["transcriptions"] = [{"start": 0, "end": 30, "language": "zh", "model": "base"}]
    service.store.save(video, old, [])
    service.transcribe(video.id, 10, 20, "zh", model="small")
    current = service.store.segments(video.id)
    texts = {s.text for s in current}
    assert "obsolete inside" not in texts
    assert {s.text for s in old if s.text != "obsolete inside"} <= texts
    assert next(s for s in current if s.text == "crossing left").end == 15
    history = service.store.get(video.id).metadata["transcriptions"]
    assert sorted((r["start"], r["end"], r["model"]) for r in history) == [
        (0, 10, "base"),
        (10, 20, "small"),
        (20, 30, "base"),
    ]


def test_storage_failure_rolls_back_asr_deletion_and_new_history(tmp_path, monkeypatch):
    service, video, _ = setup_service(tmp_path)
    service.transcribe(video.id, 10, 20, "zh")
    old_segments = service.store.segments(video.id)
    old_metadata = service.store.get(video.id).metadata

    def fail(*_args):
        raise RuntimeError("Write failed after replacement deletion")

    monkeypatch.setattr(service.store, "_add_segments", fail)
    with pytest.raises(RuntimeError, match="Write failed"):
        service.transcribe(video.id, 10, 20, "zh", model="small")
    assert service.store.segments(video.id) == old_segments
    assert service.store.get(video.id).metadata == old_metadata


@pytest.mark.parametrize(
    "vocabulary",
    ["CUDA", [""], ["x" * 81], ["x"] * 65, ["x" * 77 + f"{i:03}" for i in range(30)], [42]],
)
def test_invalid_vocabulary_is_rejected_before_any_audio_work(vocabulary):
    with pytest.raises(VidLensError) as error:
        normalize_vocabulary(vocabulary)
    assert error.value.code == "ASR_VOCABULARY"


def test_vocabulary_is_forwarded_as_whisper_initial_prompt(tmp_path, monkeypatch, capsys):
    from vidlens import asr_models, sources

    captured = {}

    class Model:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            return [SimpleNamespace(start=0, end=1, text="CUDA and KV cache")], None

    media = SimpleNamespace(streams=SimpleNamespace(audio=[object()]))
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _: nullcontext(media)))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    monkeypatch.setattr(asr_models, "cached_official_model", lambda *_: tmp_path)
    monkeypatch.setattr(sources, "_decode_asr_audio", lambda _: [0])
    monkeypatch.setattr(
        sys,
        "argv",
        ["worker", "whisper", "audio", "base", "zh", str(tmp_path), '["CUDA","KV cache"]'],
    )
    monkeypatch.setenv("HF_HOME", "before")
    monkeypatch.setenv("HF_XET_CACHE", "before")
    sources._worker()
    assert captured["initial_prompt"] == "CUDA, KV cache"
    assert captured["language"] == "zh" and captured["vad_filter"] is True
    assert json.loads(capsys.readouterr().out)[0]["text"] == "CUDA and KV cache"


def test_source_subprocess_receives_bounded_vocabulary_json(tmp_path, monkeypatch):
    provider = SourceProvider(tmp_path / "sources")
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixture")
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return "[]"

    monkeypatch.setattr(provider, "_run", run)
    provider.transcribe(MediaAsset(audio), vocabulary=["KV cache", "FP8"])
    assert json.loads(commands[0][-1]) == ["KV cache", "FP8"]


def test_cli_forwards_vocabulary_and_refresh():
    captured = {}
    service = SimpleNamespace(transcribe=lambda _id, **kwargs: captured.update(kwargs) or {})
    args = build_parser().parse_args(
        ["transcribe", "asr", "--vocabulary", "FP8", "KV cache", "--refresh"]
    )
    dispatch(args, service)
    assert captured["vocabulary"] == ["FP8", "KV cache"]
    assert captured["refresh"] is True and captured["model"] == "base"


@pytest.mark.asyncio
async def test_mcp_forwards_vocabulary_and_refresh():
    captured = {}
    service = SimpleNamespace(transcribe=lambda _id, **kwargs: captured.update(kwargs) or {})
    server = create_server(service=service)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool(
            "transcribe_video",
            {
                "video_id": "asr",
                "vocabulary": ["FP8"],
                "refresh": True,
            },
        )
    assert not result.isError
    assert captured["vocabulary"] == ["FP8"] and captured["refresh"] is True

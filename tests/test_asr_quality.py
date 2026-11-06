"""Backend provenance, optional dependencies and verified MLX model lifecycle."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vidlens import asr_mlx
from vidlens.asr import resolve_asr
from vidlens.models import MediaAsset, VidLensError
from vidlens.sources import SourceProvider


def test_backend_default_and_explicit_model_cannot_silently_select_other_weights(tmp_path):
    assert resolve_asr() == ("faster-whisper", "base")
    assert resolve_asr("mlx-whisper") == ("mlx-whisper", "large-v3-turbo")
    assert resolve_asr("mlx-whisper", "mlx-community/whisper-large-v3-turbo") == (
        "mlx-whisper", "large-v3-turbo"
    )
    with pytest.raises(VidLensError, match="MLX supports"):
        resolve_asr("mlx-whisper", "base")
    with pytest.raises(VidLensError):
        resolve_asr("cloud")
    assert resolve_asr("mlx-whisper", str(tmp_path)) == ("mlx-whisper", str(tmp_path.resolve()))


def test_mlx_provider_preserves_timing_and_backend_source(tmp_path, monkeypatch):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio fixture")
    provider = SourceProvider(tmp_path / "source")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return json.dumps([{"start": 1, "end": 5, "text": "Q读取K和V"},
                           {"start": 11, "end": 12, "text": "outside"}])

    monkeypatch.setattr(provider, "_run", run)
    segments = provider.transcribe(MediaAsset(audio, 473, 4, True), language="zh",
                                   backend="mlx-whisper", vocabulary=["Q", "K", "V"])
    assert len(segments) == 1
    assert (segments[0].start, segments[0].end, segments[0].text) == (474, 477, "Q读取K和V")
    assert segments[0].source == "mlx-whisper:large-v3-turbo"
    assert segments[0].kind == "transcript"
    command = commands[0]
    assert command[2] == "vidlens.asr_mlx"
    assert Path(command[-2]) == tmp_path / "source/models/mlx-whisper"
    assert json.loads(command[-1]) == ["Q", "K", "V"]


def test_verified_cache_rejects_same_size_corruption_and_honors_offline(tmp_path, monkeypatch):
    directory = tmp_path / "verified/large-v3-turbo"
    directory.mkdir(parents=True)
    manifest = {"repository": asr_mlx.MODEL_REPOSITORY, "commit": "a" * 40, "files": {}}
    for name in asr_mlx.MODEL_FILES:
        content = b"original-model-content"
        (directory / name).write_bytes(content)
        manifest["files"][name] = {"bytes": len(content), "kind": "sha256",
                                   "digest": hashlib.sha256(content).hexdigest()}
    (directory / "verified.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert asr_mlx.prepare_model("large-v3-turbo", tmp_path) == directory
    (directory / "weights.safetensors").write_bytes(b"corrupt!-model-content")
    assert asr_mlx.cached_model(tmp_path) is None
    with pytest.raises(VidLensError) as error:
        asr_mlx.prepare_model("large-v3-turbo", tmp_path)
    assert error.value.code == "ASR_MODEL_DOWNLOAD"


def test_failed_model_download_never_publishes_a_verified_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def fail(client, repo, revision, filename, directory):
        (directory / filename).write_bytes(b"partial")
        raise ValueError("checksum mismatch")

    monkeypatch.setattr(asr_mlx, "_download_file", fail)
    with pytest.raises(VidLensError):
        asr_mlx.prepare_model("large-v3-turbo", tmp_path)
    assert not (tmp_path / "verified/large-v3-turbo/verified.json").exists()
    assert asr_mlx.cached_model(tmp_path) is None


def test_local_model_requires_usable_file_layout(tmp_path):
    with pytest.raises(VidLensError, match="config.json and weights"):
        asr_mlx.prepare_model(str(tmp_path), tmp_path / "cache")


def test_worker_preserves_raw_text_and_only_passes_observed_vocabulary(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("faster_whisper")

    calls = []

    def transcribe(audio, **options):
        calls.append(options)
        return {"segments": [{"start": 0.2, "end": 2.4, "text": "a possible misrecognition",
                              "tokens": [123]}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps",
                        lambda _: [{"start": 0, "end": 3 * 16000}])
    assert asr_mlx.transcribe_audio(np.zeros(3 * 16000), "local-model", "zh", ["TCP", "SYN"]) == [
        {"start": 0.2, "end": 2.4, "text": "a possible misrecognition"}
    ]
    assert calls[0]["initial_prompt"] == "TCP, SYN"
    assert calls[0]["language"] == "zh"


def test_silence_skips_decoder_and_long_pauses_restore_original_times(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("faster_whisper")

    audio = np.zeros(20 * 16000, dtype=np.float32)
    called = []

    def transcribe(audio, **options):
        called.append(len(audio))
        return {"segments": [{"start": 0, "end": 2, "text": "TCP"},
                              {"start": 2, "end": 4, "text": "ACK"},
                              {"start": 5, "end": 29.98, "text": "padded hallucination"}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    assert asr_mlx.transcribe_audio(audio, "local-model", "zh", []) == []
    assert called == []
    monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps",
                        lambda _: [{"start": 16000, "end": 3 * 16000},
                                   {"start": 15 * 16000, "end": 17 * 16000}])
    segments = asr_mlx.transcribe_audio(audio, "local-model", "zh", [])
    assert [(s["start"], s["end"], s["text"]) for s in segments] == [
        (1, 3, "TCP"), (15, 17, "ACK")
    ]
    assert called == [4 * 16000]


@pytest.mark.parametrize("diagnostic,code,hint", [
    ("ModuleNotFoundError: No module named mlx", "ASR_UNAVAILABLE", "asr-mlx"),
    ("VIDLENS_ASR_MLX_PLATFORM", "ASR_DEVICE", "faster-whisper"),
    ("VIDLENS_ASR_MLX_DEVICE", "ASR_DEVICE", "GPU"),
    ("VIDLENS_ASR_MODEL_DOWNLOAD", "ASR_MODEL_DOWNLOAD", "backend"),
])
def test_mlx_failure_explains_specific_recovery(tmp_path, monkeypatch, diagnostic, code, hint):
    class Failure:
        returncode = 1

        def communicate(self, **kwargs):
            return "", diagnostic

    monkeypatch.setattr("vidlens.sources.subprocess.Popen", lambda *a, **k: Failure())
    with pytest.raises(VidLensError) as error:
        SourceProvider(tmp_path)._run([sys.executable, "-m", "vidlens.asr_mlx"])
    assert error.value.code == code
    assert hint in error.value.hint

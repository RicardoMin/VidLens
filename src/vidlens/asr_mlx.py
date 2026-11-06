"""Optional Apple Silicon transcription worker with a verified local model cache."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
from pathlib import Path

import httpx
from filelock import FileLock, Timeout

from .asr import resolve_asr
from .asr_models import _digest, _download_file
from .models import VidLensError

MODEL_REPOSITORY = "mlx-community/whisper-large-v3-turbo"
MODEL_FILES = ("config.json", "weights.safetensors")


def cached_model(cache):
    directory = Path(cache) / "verified" / "large-v3-turbo"
    try:
        manifest = json.loads((directory / "verified.json").read_text())
        if manifest["repository"] != MODEL_REPOSITORY:
            return None
        if not re.fullmatch(r"[0-9a-f]{40}", manifest["commit"]):
            return None
        if set(manifest["files"]) != set(MODEL_FILES):
            return None
        for name, expected in manifest["files"].items():
            path = directory / name
            if expected["kind"] not in {"sha256", "git-sha1"}:
                return None
            if path.stat().st_size != expected["bytes"]:
                return None
            if _digest(path, expected["kind"]) != expected["digest"]:
                return None
        return directory
    except (OSError, ValueError, KeyError, TypeError):
        return None


def prepare_model(model, cache):
    _, model = resolve_asr("mlx-whisper", model)
    if Path(model).is_dir():
        directory = Path(model)
        if not (directory / "config.json").is_file() or not any(
            (directory / name).is_file() for name in ("weights.safetensors", "weights.npz")
        ):
            raise VidLensError("ASR_MODEL", "The local MLX model needs config.json and weights.")
        return directory
    directory = Path(cache) / "verified" / model
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(directory / "download.lock", timeout=900):
            if cached := cached_model(cache):
                return cached
            if os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}:
                raise VidLensError("ASR_MODEL_DOWNLOAD", "The MLX model is not cached for offline use.")
            manifest = {"repository": MODEL_REPOSITORY, "commit": "", "files": {}}
            with httpx.Client(
                timeout=httpx.Timeout(40, connect=12), follow_redirects=True, max_redirects=8
            ) as client:
                for filename in MODEL_FILES:
                    commit, metadata = _download_file(
                        client, MODEL_REPOSITORY, manifest["commit"] or "main", filename, directory
                    )
                    manifest["commit"] = commit
                    manifest["files"][filename] = metadata
            pending = directory / "verified.json.part"
            pending.write_text(json.dumps(manifest, indent=2))
            pending.replace(directory / "verified.json")
            return directory
    except (httpx.HTTPError, OSError, ValueError, Timeout) as exc:
        raise VidLensError("ASR_MODEL_DOWNLOAD", "Could not download and verify the MLX model.") from exc


def transcribe_audio(audio, model, language, vocabulary):
    import numpy as np
    from faster_whisper.vad import SpeechTimestampsMap, get_speech_timestamps

    # MLX Whisper's decoder can invent subtitles on pure silence. Use the same
    # conservative Silero defaults as the CPU backend, then restore source time.
    chunks = get_speech_timestamps(audio)
    if not chunks:
        return []
    speech = np.concatenate([audio[chunk["start"]:chunk["end"]] for chunk in chunks])
    duration = len(speech) / 16000
    import mlx_whisper

    result = mlx_whisper.transcribe(
        speech, path_or_hf_repo=str(model), language=language or None, verbose=None,
        initial_prompt=", ".join(vocabulary) if vocabulary else None,
    )
    timeline = SpeechTimestampsMap(chunks, 16000, time_precision=6)
    segments = []
    for segment in result["segments"]:
        if any(isinstance(segment[key], bool) or not isinstance(segment[key], (int, float))
               or not math.isfinite(segment[key]) for key in ("start", "end")):
            raise VidLensError("ASR_FAILED", "MLX returned an invalid source timestamp.")
        start, end = max(0, segment["start"]), min(duration, segment["end"])
        if start >= end:
            continue
        segments.append({"start": timeline.get_original_time(start),
                         "end": timeline.get_original_time(end, is_end=True),
                         "text": segment["text"]})
    return segments


def main():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("VIDLENS_ASR_MLX_PLATFORM: MLX requires Apple Silicon macOS")
    audio_path, model, language, cache = sys.argv[1:5]
    os.environ["HF_HOME"] = cache
    os.environ["HF_XET_CACHE"] = str(Path(cache) / "xet")
    import mlx.core as mx

    if not mx.metal.is_available():
        raise SystemExit("VIDLENS_ASR_MLX_DEVICE: Apple Metal GPU is unavailable")
    from .sources import _decode_asr_audio, normalize_vocabulary

    vocabulary = normalize_vocabulary(json.loads(sys.argv[5]) if len(sys.argv) > 5 else None)
    try:
        directory = prepare_model(model, cache)
    except VidLensError as exc:
        raise SystemExit(f"VIDLENS_{exc.code}: {exc.message}") from exc
    segments = transcribe_audio(_decode_asr_audio(Path(audio_path)), directory, language, vocabulary)
    print(json.dumps(segments, ensure_ascii=False))


if __name__ == "__main__":
    main()

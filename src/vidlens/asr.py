"""Lightweight ASR selection; importing this module never loads a model."""

import re
from pathlib import Path

from .models import VidLensError

BACKENDS = ("faster-whisper", "mlx-whisper")


def resolve_asr(backend="faster-whisper", model=None):
    if backend not in BACKENDS:
        raise VidLensError("ASR_BACKEND", "Use faster-whisper or mlx-whisper.")
    if model is None:
        model = "large-v3-turbo" if backend == "mlx-whisper" else "base"
    if not isinstance(model, str) or not model.strip():
        raise VidLensError("ASR_MODEL", "model must be a nonempty model name or local path.")
    model = model.strip()
    if backend == "mlx-whisper" and model == "mlx-community/whisper-large-v3-turbo":
        model = "large-v3-turbo"
    if Path(model).is_dir():
        return backend, str(Path(model).resolve())
    if backend == "mlx-whisper" and model != "large-v3-turbo":
        raise VidLensError(
            "ASR_MODEL", "MLX supports large-v3-turbo or an existing local MLX Whisper directory."
        )
    if not re.fullmatch(r"[A-Za-z0-9_./-]{1,160}", model):
        raise VidLensError("ASR_MODEL", "Invalid local Whisper model name or path.")
    return backend, model


def asr_source(backend, model):
    return f"{'mlx-whisper' if backend == 'mlx-whisper' else 'whisper'}:{model}"

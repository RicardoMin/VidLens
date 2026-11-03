"""Verified downloads from faster-whisper's official Systran repositories.

This is a fallback when the public Hugging Face metadata API is unavailable
but the same repository's public resolve endpoints work. No mirror is used.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import httpx
from filelock import FileLock, Timeout

from .models import VidLensError

OFFICIAL_MODELS = frozenset(
    {
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
    }
)
_REQUIRED = ("config.json", "model.bin", "tokenizer.json")
_VOCABULARIES = ("vocabulary.json", "vocabulary.txt")
_OPTIONAL = (*_VOCABULARIES, "preprocessor_config.json")


def _directory(name: str, cache: Path) -> Path:
    if name not in OFFICIAL_MODELS:
        raise VidLensError(
            "ASR_MODEL_DOWNLOAD",
            "Direct download is limited to known official Whisper models.",
            "Use a supported model name or an existing local model directory.",
        )
    return Path(cache) / "verified" / name


def _digest(path: Path, kind: str) -> str:
    hasher = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
    if kind == "git-sha1":
        hasher.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def cached_official_model(name: str, cache: Path) -> Path | None:
    if name not in OFFICIAL_MODELS:
        return None
    directory = _directory(name, cache)
    try:
        manifest = json.loads((directory / "verified.json").read_text())
        if manifest.get("repository") != f"Systran/faster-whisper-{name}":
            return None
        if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("commit", "")):
            return None
        files = manifest["files"]
        if not set(_REQUIRED).issubset(files) or not set(files).issubset((*_REQUIRED, *_OPTIONAL)):
            return None
        if not any(name in files for name in _VOCABULARIES):
            return None
        for filename, expected in files.items():
            path = directory / filename
            if (
                expected["kind"] not in {"sha256", "git-sha1"}
                or path.stat().st_size != expected["bytes"]
            ):
                return None
            if _digest(path, expected["kind"]) != expected["digest"]:
                return None
        return directory
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _download_file(
    client: httpx.Client, repository: str, revision: str, filename: str, directory: Path
) -> tuple[str, dict] | None:
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{filename}?download=true"
    temporary = directory / (filename + ".part")
    started = time.monotonic()
    try:
        with client.stream("GET", url) as response:
            if response.status_code == 404 and filename in _OPTIONAL:
                return None
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("The official server did not return the complete file")
            headers = [r.headers for r in [*response.history, response]]
            commit = next((h["x-repo-commit"] for h in headers if h.get("x-repo-commit")), "")
            if not re.fullmatch(r"[0-9a-f]{40}", commit) or (
                revision != "main" and commit != revision
            ):
                raise ValueError("Missing or inconsistent official repository revision")
            expected = next((h["x-linked-etag"] for h in headers if h.get("x-linked-etag")), "")
            if not expected:
                expected = next((h["etag"] for h in headers if h.get("etag")), "")
            expected = expected.strip('"')
            if re.fullmatch(r"[0-9a-f]{64}", expected):
                kind = "sha256"
            elif re.fullmatch(r"[0-9a-f]{40}", expected):
                kind = "git-sha1"
            else:
                raise ValueError("Missing official Git/LFS content checksum")
            size = 0
            with temporary.open("wb") as target:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    size += len(chunk)
                    if size > 4 * 1024**3 or time.monotonic() - started > 900:
                        raise ValueError("The model download exceeded its size or time budget")
                    target.write(chunk)
            length = response.headers.get("content-length")
            if length and size != int(length):
                raise ValueError("Incomplete model file")
            if _digest(temporary, kind) != expected:
                raise ValueError("Official model checksum mismatch")
            temporary.replace(directory / filename)
            return commit, {"kind": kind, "digest": expected, "bytes": size}
    finally:
        temporary.unlink(missing_ok=True)


def download_official_model(name: str, cache: Path) -> Path:
    directory = _directory(name, cache)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(directory / "download.lock", timeout=900):
            return _download_official_model_locked(name, cache)
    except Timeout as exc:
        raise VidLensError(
            "ASR_MODEL_DOWNLOAD",
            "Another process is still downloading this Whisper model.",
            "Wait for that download to finish, then retry.",
        ) from exc


def _download_official_model_locked(name: str, cache: Path) -> Path:
    directory = _directory(name, cache)
    cached = cached_official_model(name, cache)
    if cached:
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    repository = f"Systran/faster-whisper-{name}"
    manifest: dict = {"repository": repository, "commit": "", "files": {}}
    try:
        with httpx.Client(
            timeout=httpx.Timeout(40, connect=12), follow_redirects=True, max_redirects=8
        ) as client:
            for filename in (*_REQUIRED, *_OPTIONAL):
                result = _download_file(
                    client, repository, manifest["commit"] or "main", filename, directory
                )
                if result:
                    commit, metadata = result
                    manifest["commit"] = commit
                    manifest["files"][filename] = metadata
        if not any(name in manifest["files"] for name in _VOCABULARIES):
            raise ValueError("The official repository did not provide a vocabulary file")
        temporary = directory / "verified.json.part"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(directory / "verified.json")
        return directory
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise VidLensError(
            "ASR_MODEL_DOWNLOAD",
            "Could not download and verify the official Whisper model.",
            "Retry model-source connectivity or supply an existing local CTranslate2 model directory.",
        ) from exc

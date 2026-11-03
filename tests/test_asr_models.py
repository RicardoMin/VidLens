import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from vidlens.asr_models import cached_official_model, download_official_model
from vidlens.models import VidLensError


def _official_transport(requests, corrupt=False):
    commit = "a" * 40
    files = {
        "config.json": b'{"num_mel_bins":80}',
        "model.bin": b"official model bytes",
        "tokenizer.json": b'{"version":"1.0"}',
        "vocabulary.json": b'["hello"]',
    }

    def handle(request):
        requests.append(request)
        filename = request.url.path.rsplit("/", 1)[-1]
        if filename in {"preprocessor_config.json", "vocabulary.txt"}:
            return httpx.Response(404)
        content = files[filename]
        if filename == "model.bin":
            expected = hashlib.sha256(content).hexdigest()
        else:
            expected = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if corrupt and filename == "model.bin":
            content = b"corrupted weight bytes"
        return httpx.Response(
            200,
            content=content,
            headers={"x-repo-commit": commit, "x-linked-etag": f'"{expected}"'},
        )

    return httpx.MockTransport(handle)


def test_official_download_pins_revision_and_checks_cache_bytes(tmp_path, monkeypatch):
    requests = []
    original = httpx.Client
    transport = _official_transport(requests)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    directory = download_official_model("base", tmp_path)
    assert cached_official_model("base", tmp_path) == directory
    assert all(request.url.host == "huggingface.co" for request in requests)
    assert "/resolve/main/config.json" in requests[0].url.path
    assert all(f"/resolve/{'a' * 40}/" in request.url.path for request in requests[1:])
    before = len(requests)
    assert download_official_model("base", tmp_path) == directory
    assert len(requests) == before
    (directory / "model.bin").write_bytes(b"tampered")
    assert cached_official_model("base", tmp_path) is None


def test_checksum_mismatch_never_marks_model_ready(tmp_path, monkeypatch):
    original = httpx.Client
    transport = _official_transport([], corrupt=True)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    with pytest.raises(VidLensError) as error:
        download_official_model("base", tmp_path)
    assert error.value.code == "ASR_MODEL_DOWNLOAD"
    assert cached_official_model("base", tmp_path) is None
    assert not (tmp_path / "verified" / "base" / "model.bin").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_concurrent_model_downloads_share_one_verified_result(tmp_path, monkeypatch):
    import vidlens.asr_models as models

    requests = []
    original_client = httpx.Client
    transport = _official_transport(requests)
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=transport, **kwargs)
    )
    original_download = models._download_file

    def slow_first_file(*args, **kwargs):
        if args[3] == "config.json":
            time.sleep(0.08)
        return original_download(*args, **kwargs)

    monkeypatch.setattr(models, "_download_file", slow_first_file)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download_official_model, "base", tmp_path) for _ in range(2)]
        directories = [future.result() for future in futures]
    assert directories[0] == directories[1]
    assert len(requests) == 6  # Three required files, two vocabulary candidates, one optional file.
    assert cached_official_model("base", tmp_path) == directories[0]


@pytest.mark.parametrize(
    "model", ["../../outside", "https://example.com/model", "unknown/weights", "turbo"]
)
def test_resolve_fallback_rejects_non_allowlisted_model_sources(tmp_path, model):
    with pytest.raises(VidLensError):
        download_official_model(model, tmp_path)
    assert not list(tmp_path.iterdir())

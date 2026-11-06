from __future__ import annotations

import base64
import io
import json
import runpy
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image

from vidlens.cli import build_parser, dispatch, seconds
from vidlens.mcp_server import create_server, result_content
from vidlens.models import VidLensError


class EvidenceService:
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.read_requests = []

    def read_video(self, url, question, max_tokens, languages, refresh, intent="auto"):
        if url == "missing":
            raise VidLensError("not_found", "Video not cached.", "Provide a valid video URL.")
        self.read_requests.append((url, question, max_tokens, languages, refresh))
        return {"video_id": "lecture", "evidence": [{
            "start": 5, "end": 9, "text": "实验设置为 batch size 8。", "source": "manual",
        }]}

    def overview(self, video_id, max_tokens):
        return {"video_id": video_id, "text": "原始字幕，而不是再次生成的摘要。"}

    def read_frames(self, video_id, start, end, **kwargs):
        return {"video_id": video_id, "frames": [{
            "timestamp": start + 2.5, "path": str(self.image_path), "ocr": "batch_size = 8",
        }]}

    def search(self, video_id, query, **kwargs):
        raise VidLensError("not_found", "Video not cached.", "Call open_video first.")

    def read_interval(self, video_id, start, end, max_tokens, cursor=None):
        return {"interval": [start, end], "cursor": cursor, "budget": max_tokens}


@pytest.mark.asyncio
async def test_mcp_protocol_returns_images_without_duplicate_structured_text(tmp_path, monkeypatch):
    monkeypatch.delenv("VIDLENS_ADVANCED_TOOLS", raising=False)
    frame = tmp_path / "frame.png"
    Image.new("RGB", (400, 200), "white").save(frame)
    server = create_server(service=EvidenceService(frame))
    async with create_connected_server_and_client_session(server) as client:
        tools = await client.list_tools()
        assert {"read_video", "read_interval", "read_frames", "crop_frame",
                "transcribe_video", "video_coverage", "study_status", "scan_video",
                "read_section", "save_note", "revise_sections", "add_reference",
                "search_knowledge", "check_facts", "read_detail"} == {t.name for t in tools.tools}
        result = await client.call_tool("read_frames", {
            "video_id": "lecture", "start": 120.0, "end": 130.0,
        })
        assert not result.isError
        assert result.structuredContent is None
        assert [item.type for item in result.content] == ["text", "image"]
        metadata = json.loads(result.content[0].text)
        assert metadata["frames"][0]["timestamp"] == 122.5
        with Image.open(io.BytesIO(base64.b64decode(result.content[1].data))) as returned:
            assert returned.size == (400, 200)
        error = await client.call_tool("read_video", {"url": "missing", "question": "cache"})
        assert error.isError
        assert json.loads(error.content[0].text)["error"]["code"] == "not_found"


def test_frame_mapping_survives_missing_first_files_and_exact_duplicates(tmp_path):
    available = tmp_path / "available.png"
    Image.new("RGB", (10, 10), "white").save(available)
    missing = str(tmp_path / "missing.png")
    source = {"frames": [
        {"timestamp": 0.0, "path": missing},
        {"timestamp": 1.0, "path": missing},
        {"timestamp": 2.0, "duplicate_of": 1.0, "path": str(available)},
        {"timestamp": 3.0, "duplicate_of": 1.0, "path": str(available)},
        {"timestamp": 4.0, "path": str(available)},
    ]}
    result = result_content(source, images=True)
    assert not result.isError
    assert [item.type for item in result.content] == ["text", "image", "image"]
    metadata = json.loads(result.content[0].text)
    assert [frame["image_index"] for frame in metadata["frames"]] == [None, 1, 1, 1, 2]
    assert "image_error" in metadata["frames"][0]
    assert metadata["accounting"]["transmitted_image_count"] == 2
    assert "image_index" not in source["frames"][1]


@pytest.mark.asyncio
@pytest.mark.parametrize("advanced", [False, True])
async def test_advanced_mcp_tools_are_opt_in(tmp_path, monkeypatch, advanced):
    if advanced:
        monkeypatch.setenv("VIDLENS_ADVANCED_TOOLS", "1")
    else:
        monkeypatch.delenv("VIDLENS_ADVANCED_TOOLS", raising=False)
    server = create_server(service=EvidenceService(tmp_path / "unused.png"))
    async with create_connected_server_and_client_session(server) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools.tools}
        extra = {"open_video", "video_overview", "find_evidence", "list_videos", "doctor"}
        assert len(names) == (20 if advanced else 15)
        assert extra <= names if advanced else not names.intersection(extra)
        result = await client.call_tool("video_overview", {"video_id": "lecture"})
        if advanced:
            assert not result.isError
            assert json.loads(result.content[0].text)["video_id"] == "lecture"
        else:
            assert result.isError


@pytest.mark.asyncio
async def test_primary_mcp_entry_returns_source_evidence_in_one_call(tmp_path):
    service = EvidenceService(tmp_path / "unused.png")
    server = create_server(service=service)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("read_video", {
            "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "question": "实验的 batch size 是多少？",
            "max_tokens": 2000,
        })
        assert not result.isError
        assert result.structuredContent is None
        assert json.loads(result.content[0].text)["evidence"][0]["text"] == "实验设置为 batch size 8。"
        assert service.read_requests == [(
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "实验的 batch size 是多少？", 2000, ("zh", "en"), False,
        )]


def test_primary_cli_entry_preserves_question_language_and_budget(tmp_path):
    service = EvidenceService(tmp_path / "unused.png")
    args = build_parser().parse_args([
        "inspect", "https://www.bilibili.com/video/BV1xx411c7mD/", "--question", "实验条件",
        "--languages", "en,zh", "--max-tokens", "2500", "--refresh",
    ])
    result = dispatch(args, service)
    assert "batch size 8" in result["evidence"][0]["text"]
    assert service.read_requests == [(
        "https://www.bilibili.com/video/BV1xx411c7mD/", "实验条件", 2500, ("en", "zh"), True,
    )]


def test_cli_preserves_time_interval_pagination_and_home(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "--home", str(tmp_path), "read", "lecture", "01:20", "02:30", "--cursor", "revision:7:0",
    ])
    assert args.home == tmp_path
    assert dispatch(args, EvidenceService(tmp_path / "frame.png")) == {
        "interval": [80.0, 150.0], "cursor": "revision:7:0", "budget": 4000,
    }
    args = parser.parse_args(["read", "lecture", "80", "150", "--home", str(tmp_path)])
    assert args.home == tmp_path


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "1:60", "1:2:3:4"])
def test_cli_rejects_invalid_time(value):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        seconds(value)


def test_installation_produces_portable_launch_config_without_global_writes(tmp_path):
    import tomllib

    project = tmp_path / "Video project"
    binary = project / ".venv" / "bin" / "python"
    binary.parent.mkdir(parents=True)
    binary.write_text("runtime")
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "install-astra.py"))
    output = project / "integration"
    result = namespace["prepare"](project, output, project / "cache")
    config = tomllib.loads((output / "codex.toml").read_text())
    server = config["mcp_servers"]["vidlens"]
    assert server["command"] == str(binary)
    assert server["args"] == ["-m", "vidlens", "mcp"]
    assert server["env"]["VIDLENS_HOME"] == str(project / "cache")
    assert result["global_settings_changed"] is False


@pytest.mark.asyncio
async def test_stdio_server_reads_persistent_evidence_from_another_working_directory(tmp_path):
    from vidlens.service import VideoService

    subtitle = tmp_path / "lecture.srt"
    subtitle.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nKV cache stores keys and values.\n\n"
        "2\n00:00:05,000 --> 00:00:09,000\n实验设置为 batch size 8。\n",
        encoding="utf-8",
    )
    home = tmp_path / "cache"
    imported = VideoService(home=home).import_transcript(subtitle)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "vidlens", "mcp", "--home", str(home)],
        cwd=str(tmp_path),
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write, read_timeout_seconds=timedelta(seconds=15)) as client,
    ):
        await client.initialize()
        result = await client.call_tool("read_interval", {
            "video_id": imported["video_id"], "start": 0.0, "end": 9.0,
        })
        assert not result.isError
        assert result.structuredContent is None
        assert "batch size 8" in result.content[0].text
        assert "KV cache" in result.content[0].text

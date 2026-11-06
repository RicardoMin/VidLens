#!/usr/bin/env python3
"""Set up the local runtime and optional project-scoped AI integration."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    asr = parser.add_mutually_exclusive_group()
    asr.add_argument(
        "--captions-only", action="store_true", help="Skip the optional local ASR runtime"
    )
    asr.add_argument(
        "--apple-asr", action="store_true", help="Add Apple Silicon MLX transcription with speech detection"
    )
    parser.add_argument(
        "--activate", action="store_true", help="Install this project's skill and Codex MCP settings"
    )
    args = parser.parse_args()
    if args.apple_asr and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        parser.error("--apple-asr requires Apple Silicon macOS")
    project = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if not uv:
        print(
            "Install uv from https://docs.astral.sh/uv/getting-started/installation/ then rerun.",
            file=sys.stderr,
        )
        return 1
    env = os.environ.copy()
    # The generated MCP configuration launches this specific environment.
    env["UV_PROJECT_ENVIRONMENT"] = str(project / ".venv")
    env["UV_CACHE_DIR"] = str(project / ".vidlens" / "uv-cache")
    env["TIKTOKEN_CACHE_DIR"] = str(project / ".vidlens" / "tokenizer")
    command = [uv, "sync", "--frozen", "--python", "3.12"]
    if not args.captions_only:
        command += ["--extra", "asr-mlx" if args.apple_asr else "asr"]
    subprocess.run(command, cwd=project, env=env, check=True)
    python = project / ".venv" / "bin" / "python"
    try:
        subprocess.run(
            [str(python), "-c", "import tiktoken; tiktoken.get_encoding('o200k_base')"],
            cwd=project,
            env=env,
            check=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        print(
            "Tokenizer download unavailable; VidLens will use its conservative byte-count budget.",
            file=sys.stderr,
        )
    install = [str(python), str(project / "scripts" / "install-ai.py")]
    if args.activate:
        install += ["--project-skill", "--project-mcp"]
    subprocess.run(install, cwd=project, env=env, check=True)
    subprocess.run(
        [str(python), "-m", "vidlens", "--home", str(project / ".vidlens"), "doctor", "--pretty"],
        cwd=project,
        env=env,
        check=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"VidLens setup stopped because {Path(exc.cmd[0]).name} exited with {exc.returncode}.", file=sys.stderr)
        raise SystemExit(exc.returncode) from None

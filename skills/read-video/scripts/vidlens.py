#!/usr/bin/env python3
"""Locate this skill's VidLens installation and hand off to its isolated runtime."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    skill = Path(__file__).resolve().parents[1]
    runtime = skill / "runtime.json"
    settings = json.loads(runtime.read_text()) if runtime.is_file() else {}
    configured = os.environ.get("VIDLENS_PROJECT") or settings.get("project_root")
    candidates = [Path(configured)] if configured else list(skill.parents)
    for project in candidates:
        if (project / "src" / "vidlens" / "__init__.py").is_file():
            break
    else:
        print(json.dumps({"error": {
            "code": "runtime_not_found",
            "message": "Set VIDLENS_PROJECT to the VidLens project directory.",
        }}))
        raise SystemExit(1)
    python = project / ".venv" / "bin" / "python"
    if not python.is_file():
        print(json.dumps({"error": {
            "code": "runtime_missing", "message": f"Run uv sync --python 3.12 in {project}.",
        }}))
        raise SystemExit(1)
    env = os.environ.copy()
    env.setdefault("VIDLENS_HOME", str(settings.get("home") or project / ".vidlens"))
    os.execve(str(python), [str(python), "-m", "vidlens", *sys.argv[1:]], env)


if __name__ == "__main__":
    main()

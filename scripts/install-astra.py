#!/usr/bin/env python3
"""Compatibility entry point for install-ai.py; accepts the same arguments."""

from pathlib import Path
from runpy import run_path

_installer = run_path(str(Path(__file__).with_name("install-ai.py")))
prepare = _installer["prepare"]
main = _installer["main"]


if __name__ == "__main__":
    raise SystemExit(main())

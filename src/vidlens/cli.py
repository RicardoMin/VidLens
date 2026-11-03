"""Small JSON command line interface shared with the Astra skill."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .models import VidLensError


def seconds(value: str) -> float:
    """Accept seconds, MM:SS, or HH:MM:SS without silently accepting NaN."""
    try:
        pieces = value.split(":")
        if len(pieces) > 3:
            raise ValueError
        numbers = [float(piece) for piece in pieces]
        if any(not math.isfinite(number) or number < 0 for number in numbers):
            raise ValueError
        if len(numbers) > 1 and any(number >= 60 for number in numbers[1:]):
            raise ValueError
        return sum(number * 60 ** index for index, number in enumerate(reversed(numbers)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use nonnegative seconds, MM:SS, or HH:MM:SS.") from exc


def _common(parser: argparse.ArgumentParser, *, subcommand: bool = False) -> None:
    default = argparse.SUPPRESS if subcommand else None
    parser.add_argument("--home", type=Path, default=default, help="Cache root (or VIDLENS_HOME)")
    parser.add_argument(
        "--pretty", action="store_true", default=argparse.SUPPRESS if subcommand else False,
        help="Indent JSON output (default: compact)",
    )


def _budget(parser: argparse.ArgumentParser, default: int) -> None:
    parser.add_argument("--max-tokens", type=int, default=default, help="Text-result token budget")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidlens", description="Give Astra bounded, timestamped video evidence."
    )
    parser.add_argument("--version", action="version", version=f"vidlens {__version__}")
    _common(parser)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text)
        _common(sub, subcommand=True)
        return sub

    sub = command("inspect", "Read a video URL once and return original evidence for your question")
    sub.add_argument("url")
    sub.add_argument("--question", default="", help="Question or learning goal; omit for whole-video reading")
    sub.add_argument("--languages", default="zh,en", help="Preferred caption languages, comma-separated")
    sub.add_argument("--refresh", action="store_true")
    _budget(sub, 4000)

    sub = command("open", "Get only metadata and a navigation map (inspect is the usual entry point)")
    sub.add_argument("url")
    sub.add_argument("--languages", default="zh,en", help="Preferred caption languages, comma-separated")
    sub.add_argument("--refresh", action="store_true")
    _budget(sub, 2500)

    sub = command("overview", "Get a compact whole-video navigation map")
    sub.add_argument("video_id")
    _budget(sub, 2500)

    sub = command("search", "Find timestamped transcript and OCR evidence")
    sub.add_argument("video_id")
    sub.add_argument("query")
    sub.add_argument("--limit", type=int, default=6)
    _budget(sub, 3000)

    sub = command("read", "Expand original transcript evidence in a time interval")
    sub.add_argument("video_id")
    sub.add_argument("start", type=seconds)
    sub.add_argument("end", type=seconds)
    sub.add_argument("--cursor", default=None, help="Pass next_cursor unchanged to continue reading")
    _budget(sub, 4000)

    sub = command("frames", "Read selected frames and local OCR from a time interval")
    sub.add_argument("video_id")
    sub.add_argument("start", type=seconds)
    sub.add_argument("end", type=seconds)
    sub.add_argument("--max-frames", type=int, default=4)
    sub.add_argument("--strategy", choices=("changes", "uniform"), default="changes")
    sub.add_argument("--no-ocr", action="store_true")

    sub = command("crop", "Enlarge a cached frame region (normalized x y width height)")
    sub.add_argument("video_id")
    sub.add_argument("timestamp", type=seconds)
    sub.add_argument("box", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))

    sub = command("coverage", "See which evidence has actually been returned")
    sub.add_argument("video_id")

    sub = command("transcribe", "Run optional local speech recognition; downloads model on first use")
    sub.add_argument("video_id")
    sub.add_argument("--start", type=seconds, default=0.0)
    sub.add_argument("--end", type=seconds)
    sub.add_argument("--language")
    sub.add_argument("--model", default="base")

    sub = command("import", "Import a local subtitle file, optionally paired with a video")
    sub.add_argument("path", type=Path)
    sub.add_argument("--title", default="")
    sub.add_argument("--url", default="", help="Original video URL for citations")
    sub.add_argument("--media", type=Path, help="Local video file for visual inspection")

    command("doctor", "Check the local runtime and optional video/OCR/ASR capabilities")
    sub = command("list", "List locally cached videos")
    sub.add_argument("--limit", type=int, default=20)
    command("mcp", "Run the Astra MCP server over stdio")
    return parser


def dispatch(args: argparse.Namespace, service: Any) -> dict[str, Any]:
    command = args.command
    if command in ("inspect", "open"):
        languages = tuple(item.strip() for item in args.languages.split(",") if item.strip())
        if not languages:
            raise VidLensError("invalid_languages", "At least one caption language is required.")
        if command == "inspect":
            return service.read_video(
                args.url, question=args.question, languages=languages,
                refresh=args.refresh, max_tokens=args.max_tokens,
            )
        return service.open_video(
            args.url, languages=languages, refresh=args.refresh, max_tokens=args.max_tokens
        )
    if command == "overview":
        return service.overview(args.video_id, max_tokens=args.max_tokens)
    if command == "search":
        return service.search(args.video_id, args.query, max_tokens=args.max_tokens, limit=args.limit)
    if command == "read":
        return service.read_interval(args.video_id, args.start, args.end,
                                     max_tokens=args.max_tokens, cursor=args.cursor)
    if command == "frames":
        return service.read_frames(
            args.video_id, args.start, args.end, max_frames=args.max_frames,
            strategy=args.strategy, ocr=not args.no_ocr,
        )
    if command == "crop":
        return service.crop_frame(args.video_id, args.timestamp, tuple(args.box))
    if command == "coverage":
        return service.coverage(args.video_id)
    if command == "transcribe":
        return service.transcribe(
            args.video_id, start=args.start, end=args.end, language=args.language, model=args.model
        )
    if command == "import":
        return service.import_transcript(
            args.path, title=args.title, video_url=args.url, media_path=args.media
        )
    if command == "doctor":
        return service.doctor()
    if command == "list":
        return service.list_videos(limit=args.limit)
    raise VidLensError("invalid_command", f"Unknown command: {command}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "mcp":
            from .mcp_server import create_server

            create_server(home=args.home).run(transport="stdio")
            return 0
        from .service import VideoService

        result = dispatch(args, VideoService(home=args.home))
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None,
                         separators=None if args.pretty else (",", ":"), default=str))
        return 0
    except VidLensError as exc:
        print(json.dumps({"error": exc.to_dict()}, ensure_ascii=False, separators=(",", ":")))
        return 1
    except KeyboardInterrupt:
        print('{"error":{"code":"cancelled","message":"Operation cancelled."}}')
        return 130
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": {"code": "local_error", "message": str(exc)}}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())

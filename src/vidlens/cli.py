"""JSON command line tools for AI clients and local video study."""

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
        return sum(number * 60**index for index, number in enumerate(reversed(numbers)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use nonnegative seconds, MM:SS, or HH:MM:SS.") from exc


def _common(parser: argparse.ArgumentParser, *, subcommand: bool = False) -> None:
    default = argparse.SUPPRESS if subcommand else None
    parser.add_argument("--home", type=Path, default=default, help="Cache root (or VIDLENS_HOME)")
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=argparse.SUPPRESS if subcommand else False,
        help="Indent JSON output (default: compact)",
    )


def _budget(parser: argparse.ArgumentParser, default: int) -> None:
    parser.add_argument("--max-tokens", type=int, default=default, help="Text-result token budget")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidlens", description="Read, study and explain videos with source-linked evidence."
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
    sub.add_argument(
        "--question", default="", help="Question or learning goal; omit for whole-video reading"
    )
    sub.add_argument("--intent", choices=("auto", "summary", "question"), default="auto")
    sub.add_argument(
        "--languages", default="zh,en", help="Preferred caption languages, comma-separated"
    )
    sub.add_argument("--refresh", action="store_true")
    sub.add_argument("--prepare", action="store_true", help="For summaries, transcribe missing speech before returning the current plan")
    _budget(sub, 4000)

    sub = command("study", "Read the chapter plan and progress")
    sub.add_argument("video_id")
    sub.add_argument("--cursor", type=int, default=0)
    _budget(sub, 4000)

    sub = command("section", "Read a chapter's speech and screen images")
    sub.add_argument("video_id")
    sub.add_argument("section_id")
    sub.add_argument("--cursor")
    sub.add_argument("--no-frames", action="store_true")
    sub.add_argument("--frame-cursor", type=int)
    sub.add_argument("--frame-limit", type=int, default=4)
    sub.add_argument("--frame-view", choices=["navigation", "full"], default="navigation")
    sub.add_argument("--question", default="")
    sub.add_argument("--visual-query", action="append", dest="visual_queries")
    sub.add_argument("--representation", choices=["compact", "verbose"], default="compact")
    _budget(sub, 6000)

    sub = command("scan", "Index visual changes across the whole video")
    sub.add_argument("video_id")
    sub.add_argument("--cursor", type=int, default=0)
    sub.add_argument("--limit", type=int, default=100)
    sub.add_argument("--refresh", action="store_true")
    sub.add_argument("--view", choices=["navigation", "full"], default="navigation")
    sub.add_argument("--diagnostics", action="store_true")

    sub = command("units", "Query reusable knowledge units and their prerequisites")
    sub.add_argument("video_id")
    sub.add_argument("--query", default="")
    sub.add_argument("--role", action="append", dest="roles")
    sub.add_argument("--limit", type=int, default=12)
    _budget(sub, 6000)

    for name, help_text in (("save-units", "Save source-linked knowledge units"),
                            ("article-plan", "Save or inspect a teaching plan"),
                            ("audit-article", "Check article coverage against the teaching plan"),
                            ("article", "Export a portable Markdown lesson and HTML preview")):
        sub = command(name, help_text)
        sub.add_argument("video_id")
        sub.add_argument("--revision", required=True)
        sub.add_argument("--file", type=Path, required=name != "article-plan")
        if name == "save-units":
            sub.add_argument("--replace", action="store_true")
        if name == "article":
            sub.add_argument("--output", type=Path, required=True)

    sub = command("metrics", "Inspect local operation times and text estimates, not model billing")
    sub.add_argument("--run-id")
    sub.add_argument("--limit", type=int, default=100)

    for name, help_text in (
        ("note", "Save a source-linked chapter note"),
        ("reference", "Attach an external source read by the client"),
    ):
        sub = command(name, help_text)
        sub.add_argument("video_id")
        sub.add_argument("section_id")
        sub.add_argument("--revision", required=True)
        sub.add_argument("--file", type=Path, required=True, help="UTF-8 JSON object")

    sub = command("chapters", "Refine the chapter plan using an ordered JSON section list")
    sub.add_argument("video_id")
    sub.add_argument("--revision", required=True)
    sub.add_argument("--file", type=Path, required=True)

    sub = command("knowledge", "Search saved notes, concepts and external references")
    sub.add_argument("video_id")
    sub.add_argument("query")
    sub.add_argument("--limit", type=int, default=6)
    _budget(sub, 3000)

    sub = command("export", "Export the current study notes to Markdown")
    sub.add_argument("video_id")
    sub.add_argument("--output", type=Path)

    sub = command(
        "open", "Get only metadata and a navigation map (inspect is the usual entry point)"
    )
    sub.add_argument("url")
    sub.add_argument(
        "--languages", default="zh,en", help="Preferred caption languages, comma-separated"
    )
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
    sub.add_argument(
        "--cursor", default=None, help="Pass next_cursor unchanged to continue reading"
    )
    _budget(sub, 4000)
    sub.add_argument("--representation", choices=["compact", "verbose"], default="compact")

    sub = command("frames", "Read selected frames and local OCR from a time interval")
    sub.add_argument("video_id")
    sub.add_argument("start", type=seconds)
    sub.add_argument("end", type=seconds)
    sub.add_argument("--max-frames", type=int, default=4)
    sub.add_argument("--strategy", choices=("changes", "uniform"), default="changes")
    sub.add_argument("--no-ocr", action="store_true")
    sub.add_argument("--question", default="")
    sub.add_argument("--visual-query", action="append", dest="visual_queries")
    sub.add_argument("--preview", action="store_true")
    sub.add_argument("--timestamps", type=seconds, nargs="+")

    sub = command("crop", "Enlarge a cached frame region (normalized x y width height)")
    sub.add_argument("video_id")
    sub.add_argument("timestamp", type=seconds)
    sub.add_argument("box", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))

    sub = command("check-facts", "Inspect numeric field bindings and return source crops")
    sub.add_argument("video_id")
    sub.add_argument("--file", type=Path, required=True, help="JSON array of figure claims")

    sub = command("detail", "Inspect small text in native source-pixel regions")
    sub.add_argument("video_id")
    sub.add_argument("timestamp", type=seconds)
    sub.add_argument("--box", type=float, nargs=4, action="append", dest="boxes",
                     metavar=("X", "Y", "WIDTH", "HEIGHT"))
    sub.add_argument("--upgrade-source", action="store_true")
    sub.add_argument("--max-tiles", type=int, default=12)

    sub = command("coverage", "See which evidence has actually been returned")
    sub.add_argument("video_id")

    sub = command(
        "transcribe", "Run optional local speech recognition; downloads model on first use"
    )
    sub.add_argument("video_id")
    sub.add_argument("--start", type=seconds, default=0.0)
    sub.add_argument("--end", type=seconds)
    sub.add_argument("--language")
    sub.add_argument("--backend", choices=["faster-whisper", "mlx-whisper"], default="faster-whisper")
    sub.add_argument("--model", help="Defaults to base (CPU) or large-v3-turbo (MLX)")
    sub.add_argument(
        "--vocabulary", nargs="+", help="Technical words or quoted phrases to guide local ASR"
    )
    sub.add_argument(
        "--refresh",
        action="store_true",
        help="Redo this audio interval even if the configuration is cached",
    )

    sub = command("import", "Import a local subtitle file, optionally paired with a video")
    sub.add_argument("path", type=Path)
    sub.add_argument("--title", default="")
    sub.add_argument("--url", default="", help="Original video URL for citations")
    sub.add_argument("--media", type=Path, help="Local video file for visual inspection")

    command("doctor", "Check the local runtime and optional video/OCR/ASR capabilities")
    sub = command("list", "List locally cached videos")
    sub.add_argument("--limit", type=int, default=20)
    command("mcp", "Run the MCP server over stdio")
    return parser


def dispatch(args: argparse.Namespace, service: Any) -> dict[str, Any]:
    command = args.command
    if command in ("inspect", "open"):
        languages = tuple(item.strip() for item in args.languages.split(",") if item.strip())
        if not languages:
            raise VidLensError("invalid_languages", "At least one caption language is required.")
        if command == "inspect":
            return service.read_video(
                args.url,
                question=args.question,
                languages=languages,
                refresh=args.refresh,
                max_tokens=args.max_tokens,
                intent=args.intent,
                **({"prepare": True} if args.prepare else {}),
            )
        return service.open_video(
            args.url, languages=languages, refresh=args.refresh, max_tokens=args.max_tokens
        )
    if command == "study":
        return service.study_status(args.video_id, max_tokens=args.max_tokens, cursor=args.cursor)
    if command == "section":
        return service.read_section(
            args.video_id,
            args.section_id,
            max_tokens=args.max_tokens,
            cursor=args.cursor,
            include_frames=not args.no_frames,
            frame_cursor=args.frame_cursor,
            frame_limit=args.frame_limit,
            frame_view=args.frame_view,
            question=args.question,
            visual_queries=args.visual_queries,
            representation=args.representation,
        )
    if command == "scan":
        return service.scan_video(
            args.video_id, cursor=args.cursor, limit=args.limit, refresh=args.refresh, view=args.view,
            diagnostics=args.diagnostics,
        )
    if command == "metrics":
        return service.operation_metrics(args.run_id, args.limit)
    if command == "units":
        return service.query_knowledge(args.video_id, args.query, args.roles, args.limit, args.max_tokens)
    if command in {"save-units", "article-plan", "audit-article", "article"}:
        document = None
        if args.file:
            if args.file.stat().st_size > 8 * 1024 * 1024:
                raise VidLensError("input_too_large", "Article JSON must be at most 8 MiB.")
            document = json.loads(args.file.read_text(encoding="utf-8"))
        if command == "save-units":
            return service.save_knowledge(args.video_id, args.revision, document, args.replace)
        if command == "article-plan":
            return service.article_plan(args.video_id, args.revision, document)
        if command == "audit-article":
            return service.audit_article(args.video_id, args.revision, document)
        return service.export_article(args.video_id, args.revision, document, args.output)
    if command in ("note", "reference", "chapters"):
        if args.file.stat().st_size > 256 * 1024:
            raise VidLensError("input_too_large", "Study JSON must be at most 256 KiB.")
        document = json.loads(args.file.read_text(encoding="utf-8"))
        if command == "chapters":
            return service.revise_sections(args.video_id, args.revision, document)
        operation = service.save_note if command == "note" else service.add_reference
        return operation(args.video_id, args.section_id, args.revision, document)
    if command == "knowledge":
        return service.search_knowledge(
            args.video_id, args.query, max_tokens=args.max_tokens, limit=args.limit
        )
    if command == "export":
        result = service.export_study(args.video_id)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(result.pop("markdown"))
            result["output"] = str(args.output.resolve())
        return result
    if command == "overview":
        return service.overview(args.video_id, max_tokens=args.max_tokens)
    if command == "search":
        return service.search(
            args.video_id, args.query, max_tokens=args.max_tokens, limit=args.limit
        )
    if command == "read":
        return service.read_interval(
            args.video_id, args.start, args.end, max_tokens=args.max_tokens, cursor=args.cursor,
            representation=args.representation,
        )
    if command == "frames":
        return service.read_frames(
            args.video_id,
            args.start,
            args.end,
            max_frames=args.max_frames,
            strategy=args.strategy,
            ocr=not args.no_ocr,
            question=args.question,
            visual_queries=args.visual_queries,
            preview=args.preview,
            timestamps=args.timestamps,
        )
    if command == "crop":
        return service.crop_frame(args.video_id, args.timestamp, tuple(args.box))
    if command == "check-facts":
        if args.file.stat().st_size > 64 * 1024:
            raise VidLensError("input_too_large", "Fact JSON must be at most 64 KiB.")
        return service.check_facts(args.video_id, json.loads(args.file.read_text(encoding="utf-8")))
    if command == "detail":
        return service.read_detail(args.video_id, args.timestamp, args.boxes,
                                   upgrade_source=args.upgrade_source, max_tiles=args.max_tiles)
    if command == "coverage":
        return service.coverage(args.video_id)
    if command == "transcribe":
        from .asr import resolve_asr

        backend, model = resolve_asr(args.backend, args.model)
        return service.transcribe(
            args.video_id,
            start=args.start,
            end=args.end,
            language=args.language,
            model=model,
            backend=backend,
            vocabulary=args.vocabulary,
            refresh=args.refresh,
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
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2 if args.pretty else None,
                separators=None if args.pretty else (",", ":"),
                default=str,
            )
        )
        return 0
    except VidLensError as exc:
        print(json.dumps({"error": exc.to_dict()}, ensure_ascii=False, separators=(",", ":")))
        return 1
    except KeyboardInterrupt:
        print('{"error":{"code":"cancelled","message":"Operation cancelled."}}')
        return 130
    except (OSError, ValueError) as exc:
        print(
            json.dumps({"error": {"code": "local_error", "message": str(exc)}}, ensure_ascii=False)
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

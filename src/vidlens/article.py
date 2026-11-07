"""Export an AI-authored, source-linked lesson as portable Markdown and assets."""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageOps

from .curriculum import _anchors
from .models import VidLensError
from .store import frame_source_revision


def passive_svg(text):
    """Keep diagrams portable: no scripts, remote assets or embedded HTML."""
    if not isinstance(text, str) or len(text.encode()) > 2_000_000:
        raise VidLensError("invalid_asset", "SVG must be text up to 2 MB.")
    if "<!" in text or "<?" in text:
        raise VidLensError("invalid_asset", "SVG declarations and entities are unsupported.")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise VidLensError("invalid_asset", "Invalid SVG XML.") from exc
    allowed = {"svg", "g", "rect", "circle", "ellipse", "line", "polyline", "polygon",
               "path", "text", "tspan", "defs", "marker", "title", "desc", "clipPath"}
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise VidLensError("invalid_asset", "A diagram must have an SVG root.")
    for element in root.iter():
        if element.tag.startswith("{") and not element.tag.startswith("{http://www.w3.org/2000/svg}"):
            raise VidLensError("invalid_asset", "Use the SVG namespace only.")
        if element.tag.rsplit("}", 1)[-1] not in allowed:
            raise VidLensError("invalid_asset", "Use passive SVG shapes and text only.")
        for key, value in element.attrib.items():
            attr = key.rsplit("}", 1)[-1].lower()
            if (attr.startswith("on") or attr in {"href", "src", "style", "base"}
                    or "\\" in value):
                raise VidLensError("invalid_asset", "SVG scripts, links and CSS are unsupported.")
            if "url(" in value.lower() and not re.fullmatch(r"url\(#[\w-]+\)", value):
                raise VidLensError("invalid_asset", "SVG resources must be local fragment IDs.")
    return text.encode("utf-8")


def _source_image(service, video_id, asset):
    """Resolve pixels from this video's frame record, never from an arbitrary cache path."""
    ref = asset.get("evidence_id")
    try:
        if not isinstance(ref, str) or not ref.startswith("frame:"):
            raise ValueError
        timestamp = float(ref[6:])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError
    except ValueError as exc:
        raise VidLensError("invalid_asset", "A source image needs evidence_id='frame:TIMESTAMP'.") from exc
    video = service.store.get(video_id)
    frame = service.store.frame_at(video_id, timestamp)
    if frame is None or abs(frame.timestamp - timestamp) > 1e-6:
        raise VidLensError("invalid_asset", "Read this video's source frame before exporting it.")
    with service.store.connect() as db:
        row = db.execute("SELECT data FROM frames WHERE video_id=? AND timestamp=?",
                         (video_id, frame.timestamp)).fetchone()
    identity = frame_source_revision(video)
    if not row or json.loads(row["data"]).get("_source_identity", identity) != identity:
        raise VidLensError("invalid_asset", "The source frame belongs to an old media revision.")
    visual_root = (service.home / "visual").resolve()
    original = service.visual._source_path(frame).resolve()
    if not original.is_relative_to(visual_root) or not original.is_file():
        raise VidLensError("invalid_asset", "The registered frame must reside in the visual cache.")
    if original.stat().st_size > 40_000_000:
        raise VidLensError("invalid_asset", "Source image exceeds export limits.")
    source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    if "box" in asset and (not isinstance(asset["box"], (list, tuple)) or len(asset["box"]) != 4):
        raise VidLensError("invalid_asset", "box must contain normalized x, y, width and height.")
    selected = service.visual.crop(frame, asset["box"], ocr=False) if "box" in asset else frame
    selected_path = selected.path.resolve() if "box" in asset else original
    if not selected_path.is_relative_to(visual_root):
        raise VidLensError("invalid_asset", "The source image escapes the visual cache.")
    if "path" in asset and (not isinstance(asset["path"], str)
                            or Path(asset["path"]).expanduser().resolve() != selected_path):
        raise VidLensError("invalid_asset", "The supplied path does not match the cited frame and crop.")
    try:
        if selected_path.stat().st_size > 12_000_000:
            raise ValueError
        with Image.open(selected_path) as picture:
            if picture.format not in {"PNG", "JPEG", "WEBP"} or picture.width * picture.height > 16_000_000:
                raise ValueError
            pixels = ImageOps.exif_transpose(picture).convert("RGB")
            # A fresh image drops EXIF/text fields without altering the decoded source pixels.
            clean = Image.frombytes("RGB", pixels.size, pixels.tobytes())
            stream = io.BytesIO()
            clean.save(stream, format="PNG")
    except (OSError, ValueError) as exc:
        raise VidLensError("invalid_asset", "Source image is unreadable or exceeds export limits.") from exc
    return stream.getvalue(), {"evidence_id": f"frame:{frame.timestamp}",
                              "timestamp": frame.timestamp, "crop_box": selected.crop_box,
                              "source_sha256": source_hash, "pixel_size": list(clean.size)}


def _destinations(tokens):
    for token in tokens:
        if (token.type in {"html_block", "html_inline"}
                and not re.fullmatch(r"<!--\s*section:[A-Za-z0-9_.:-]+\s*-->", token.content.strip())):
            raise VidLensError("nonportable_article", "Use Markdown and registered assets, not raw HTML.")
        if token.type == "image":
            yield "image", token.attrGet("src")
        elif token.type == "link_open":
            yield "link", token.attrGet("href")
        if token.children:
            yield from _destinations(token.children)


def _section_anchors(markdown):
    """Create only the validated section anchors; arbitrary source HTML stays forbidden."""
    published, preview, offset = [], [], 0
    for match in _anchors(markdown):
        anchor = f'\n<a id="{html.escape(match.group(1), quote=True)}"></a>\n'
        published.extend((markdown[offset:match.end()], anchor))
        preview.extend((markdown[offset:match.start()], anchor))
        offset = match.end()
    published.append(markdown[offset:])
    preview.append(markdown[offset:])
    return "".join(published), "".join(preview)


def _publish(stage, output):
    """Claim a new directory exclusively; a raced empty directory is never replaced."""
    created_files, created_dirs = [], []
    try:
        output.mkdir()
        created_dirs.append(output)
        (output / "assets").mkdir()
        created_dirs.append(output / "assets")
        paths = (sorted((stage / "assets").iterdir())
                 + sorted(path for path in stage.iterdir()
                          if path.is_file() and path.name != "manifest.json")
                 + [stage / "manifest.json"])
        for source in paths:
            target = output / source.relative_to(stage)
            os.link(source, target)  # Exclusive creation; manifest is the final completion marker.
            created_files.append(target)
    except BaseException:
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        for path in reversed(created_dirs):
            try:
                path.rmdir()
            except OSError:
                pass
        raise


def export_article(service, video_id, revision, draft, output):
    """Coverage is checked mechanically; the author still reviews teaching and facts."""
    if not isinstance(draft, dict):
        raise VidLensError("invalid_article", "Article must be an object.")
    audit = service.audit_article(video_id, revision, draft)
    if not audit.get("ready_for_review"):
        raise VidLensError("article_incomplete", json.dumps(audit, ensure_ascii=False))
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise VidLensError("output_exists", "Choose a new output directory; existing work is preserved.")
    output = output.resolve()
    plan_record = service.curriculum.plan(video_id, revision)
    if plan_record["stale"] or plan_record["knowledge_revision"] != audit["knowledge_revision"]:
        raise VidLensError("stale_article", "Knowledge changed during article export.")
    assets = draft.get("assets", [])
    if not isinstance(assets, list) or len(assets) > 40:
        raise VidLensError("invalid_asset", "Use at most 40 article images.")
    markdown = draft["markdown"]
    for private in (str(service.home), str(Path.home()), "file://"):
        if private in markdown:
            raise VidLensError("nonportable_article", "Remove local machine paths from the article.")
    from markdown_it import MarkdownIt

    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    destinations = list(_destinations(MarkdownIt("commonmark", {"html": True}).parse(markdown)))
    image_sources = {value for kind, value in destinations if kind == "image"}
    link_destinations = {value for kind, value in destinations if kind == "link"}
    attachments = draft.get("attachments", [])
    if not isinstance(attachments, list) or len(attachments) > 10:
        raise VidLensError("invalid_attachment", "Use at most 10 text attachments.")
    attachment_names, attachment_files = set(), []
    reserved = {"readme.md", "preview.html", "manifest.json", "assets"}
    for attachment in attachments:
        if not isinstance(attachment, dict):
            raise VidLensError("invalid_attachment", "Each attachment needs name and text.")
        name, text = attachment.get("name"), attachment.get("text")
        if (not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", name)
                or Path(name).suffix.lower() not in {".py", ".json", ".txt", ".csv"}
                or name.lower() in reserved | {n.lower() for n in attachment_names}):
            raise VidLensError("invalid_attachment", "Use unique .py/.json/.txt/.csv filenames without paths.")
        if (not isinstance(text, str) or len(text.encode("utf-8")) > 100_000
                or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text)):
            raise VidLensError("invalid_attachment", "Each attachment must be ordinary UTF-8 text up to 100 KB.")
        if any(private in text for private in (str(service.home), str(Path.home()), "file://")):
            raise VidLensError("nonportable_article", "Remove local machine paths from attachments.")
        if not {name, f"./{name}"}.intersection(link_destinations):
            raise VidLensError("unused_attachment", f"Link {name} from the article.")
        attachment_names.add(name)
        attachment_files.append((name, text.encode("utf-8")))
    names, compiled = set(), []
    for asset in assets:
        if not isinstance(asset, dict):
            raise VidLensError("invalid_asset", "An asset must be an object.")
        name, kind = asset.get("name", ""), asset.get("kind")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", name):
            raise VidLensError("invalid_asset", "Use a plain asset filename.")
        if (name.lower() in {n.lower() for n in names}
                or not isinstance(kind, str) or kind not in {"source", "diagram"}):
            raise VidLensError("invalid_asset", "Asset names must be unique; kind is source or diagram.")
        names.add(name)
        if f"asset:{name}" not in image_sources:
            raise VidLensError("unused_asset", f"The article does not reference {name}.")
        provenance = {}
        if kind == "diagram":
            if not name.endswith(".svg"):
                raise VidLensError("invalid_asset", "Diagrams use .svg filenames.")
            data = passive_svg(asset.get("svg"))
            if str(service.home).encode() in data or str(Path.home()).encode() in data:
                raise VidLensError("nonportable_article", "Remove local machine paths from the diagram.")
            if "evidence_ids" in asset:
                with service.store.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    _, _, _, evidence = service.curriculum._snapshot(db, video_id, revision)
                    provenance["evidence_ids"] = service.curriculum._refs(asset["evidence_ids"], evidence)
        else:
            if not name.endswith(".png"):
                raise VidLensError("invalid_asset", "Source images export as lossless .png files.")
            data, provenance = _source_image(service, video_id, asset)
        compiled.append((name, data, kind, provenance))
    for name in sorted(names, key=len, reverse=True):
        markdown = re.sub(r"asset:" + re.escape(name) + r"(?![A-Za-z0-9_.-])",
                          f"assets/{name}", markdown)
    if "asset:" in markdown:
        raise VidLensError("missing_asset", "An article image has no matching asset.")
    # Every embedded image travels with the Markdown, without machine-specific paths.
    anchor_ids = {match.group(1) for match in _anchors(markdown)}
    for kind, destination in _destinations(renderer.parse(markdown)):
        if kind == "image" and destination not in {f"assets/{n}" for n in names}:
            raise VidLensError("nonportable_image", "Use asset:NAME for every embedded image.")
        if kind == "link":
            parsed = urlsplit(unquote(destination))
            if destination.startswith("#") and unquote(destination[1:]) not in anchor_ids:
                raise VidLensError("missing_anchor", "A contents link has no matching article section.")
            if (parsed.scheme not in {"http", "https", "mailto"}
                    and not destination.startswith("#")
                    and destination not in {"README.md", *(f"assets/{name}" for name in names),
                                            *attachment_names,
                                            *(f"./{name}" for name in attachment_names)}):
                raise VidLensError("nonportable_article", "Use public or bundle-relative links.")
    markdown, preview_markdown = _section_anchors(markdown)
    # Input HTML was rejected above. Only these generated, validated <a id> tags are active.
    preview_body = MarkdownIt("commonmark", {"html": True}).enable("table").render(preview_markdown)
    title = plan_record["plan"]["title"]
    preview = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{margin:0;background:#f6f7f9;color:#202a35;font:17px/1.9 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:880px;margin:32px auto;padding:44px 52px;background:white;border-radius:14px}}
h1,h2,h3{{line-height:1.35;color:#172c42}}h2{{margin-top:2.5em;border-bottom:1px solid #dce4ec;padding-bottom:.4em}}
a{{color:#1466a7}}img{{display:block;max-width:100%;height:auto;margin:22px auto}}table{{border-collapse:collapse;width:100%;font-size:.94em}}
th,td{{border:1px solid #dce4ec;padding:9px 13px;text-align:left}}th{{background:#eef3f7}}blockquote{{margin-left:0;padding:3px 20px;border-left:4px solid #83b4d8;background:#f5f9fc}}
pre{{overflow-x:auto;padding:18px;background:#eef3f7;border-radius:8px}}code{{font-size:.88em}}@media(max-width:640px){{main{{margin:0;padding:22px 18px;border-radius:0}}body{{font-size:16px}}table{{display:block;overflow-x:auto}}}}
</style><main>{preview_body}</main></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".vidlens-article-", dir=output.parent))
    try:
        (stage / "assets").mkdir()
        inventory = []
        for name, data, kind, provenance in compiled:
            (stage / "assets" / name).write_bytes(data)
            inventory.append({"path": f"assets/{name}", "kind": kind,
                              "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                              **provenance})
        (stage / "README.md").write_text(markdown, encoding="utf-8")
        (stage / "preview.html").write_text(preview, encoding="utf-8")
        attachment_inventory = []
        for name, data in attachment_files:
            (stage / name).write_bytes(data)
            attachment_inventory.append({"path": name, "bytes": len(data),
                                         "sha256": hashlib.sha256(data).hexdigest()})
        manifest = {"video_id": video_id, "revision": revision,
                    "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                    "preview_sha256": hashlib.sha256(preview.encode()).hexdigest(),
                    "assets": inventory, "attachments": attachment_inventory, "coverage_audit": audit,
                    "review_scope": "Structural coverage only; author must inspect facts, figures and rendered output."}
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
        # Revalidate the knowledge snapshot immediately before publishing the local bundle.
        final_audit = service.audit_article(video_id, revision, draft)
        final_plan = service.curriculum.plan(video_id, revision)
        if (not final_audit.get("ready_for_review")
                or final_audit["knowledge_revision"] != audit["knowledge_revision"]
                or final_plan["plan"] != plan_record["plan"]):
            raise VidLensError("stale_article", "Knowledge changed during article export.")
        for asset, (_, data, kind, _) in zip(assets, compiled):
            if kind == "source" and _source_image(service, video_id, asset)[0] != data:
                raise VidLensError("stale_article", "Source pixels changed during article export.")
        try:
            _publish(stage, output)
        except FileExistsError as exc:
            raise VidLensError("output_exists", "The output directory was created concurrently.") from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"video_id": video_id, "output": str(output), "markdown": str(output / "README.md"),
            "preview": str(output / "preview.html"), "manifest": str(output / "manifest.json"),
            "assets": len(compiled), "attachments": len(attachment_files), "audit": audit}

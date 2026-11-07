import hashlib
import json

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from vidlens import article
from vidlens.models import Frame, Segment, Video, VidLensError
from vidlens.service import VideoService


def seed(tmp_path):
    service = VideoService(tmp_path / "private-library")
    video = Video("lesson", "https://www.bilibili.com/video/BV1KT2yBJESn/?p=1",
                  "bilibili", "Detection", 120)
    cues = [Segment(0, 30, "The center of the object determines its grid cell.")]
    service.store.save(video, cues, [])
    path = service.home / "visual" / "source.png"
    path.parent.mkdir()
    metadata = PngInfo()
    metadata.add_text("local_source", str(path))
    Image.new("RGB", (100, 50), (12, 34, 56)).save(path, pnginfo=metadata)
    service.store.save_frames(video.id, [Frame(10, path, 100, 50)])
    revision = service.study_status(video.id)["revision"]
    unit = {"id": "grid", "title": "Grid", "roles": ["mechanism"],
            "explanation": "The center determines the responsible cell.",
            "evidence_ids": [cues[0].id, "frame:10.0"]}
    service.save_knowledge(video.id, revision, [unit])
    plan = {"title": "How detection works", "goal": "Decode a box.", "sections": [
        {"id": "grid", "title": "Grid", "objective": "Find the responsible cell.",
         "unit_ids": ["grid"], "visuals": [{"purpose": "Locate the center",
                                             "evidence_ids": ["frame:10.0"]}]}]}
    service.article_plan(video.id, revision, plan)
    draft = {
        "markdown": "# Detection\n\n<!-- section:grid -->\n## Grid\n\n"
                    "The center determines the responsible cell.\n\n"
                    "![Source][original]\n\n[original]: asset:source.png\n\n"
                    "![Diagram](asset:grid.svg)\n\n"
                    "[Video](https://www.bilibili.com/video/BV1KT2yBJESn/?p=1&t=10)",
        "coverage": [{"unit_id": "grid", "section_id": "grid",
                      "excerpt": "The center determines the responsible cell."}],
        "visual_coverage": [{"section_id": "grid", "visual_index": 0,
                             "asset_name": "source.png"}],
        "assets": [{"name": "source.png", "kind": "source", "evidence_id": "frame:10",
                    "box": [.1, .1, .8, .8]},
                   {"name": "grid.svg", "kind": "diagram", "evidence_ids": [cues[0].id],
                    "svg": '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">'
                           '<rect width="200" height="100" fill="white"/>'
                           '<circle cx="60" cy="40" r="5" fill="blue"/></svg>'}],
    }
    return service, video, revision, draft, path, unit, plan


def test_export_is_portable_pixel_bound_and_manifest_verifiable(tmp_path):
    service, video, revision, draft, _, _, _ = seed(tmp_path)
    output = tmp_path / "lesson"
    result = service.export_article(video.id, revision, draft, output)
    manifest = json.loads((output / "manifest.json").read_text())
    markdown = (output / "README.md").read_text()
    preview = (output / "preview.html").read_text()
    assert "assets/source.png" in markdown and 'src="assets/source.png"' in preview
    assert manifest["markdown_sha256"] == hashlib.sha256(markdown.encode()).hexdigest()
    assert manifest["preview_sha256"] == hashlib.sha256(preview.encode()).hexdigest()
    assert result["audit"]["ready_for_review"]
    for asset in manifest["assets"]:
        data = (output / asset["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == asset["sha256"]
    with Image.open(output / "assets/source.png") as image:
        assert image.size == (80, 40)
        assert image.getpixel((5, 5)) == (12, 34, 56)
        assert "local_source" not in image.info
    for path in output.rglob("*"):
        if path.is_file():
            assert str(service.home).encode() not in path.read_bytes()


def test_source_asset_cannot_substitute_another_cached_image(tmp_path):
    service, video, revision, draft, path, _, _ = seed(tmp_path)
    fake = path.with_name("other-video.png")
    Image.new("RGB", (80, 40), "red").save(fake)
    draft["assets"][0]["path"] = str(fake)
    with pytest.raises(VidLensError, match="does not match"):
        service.export_article(video.id, revision, draft, tmp_path / "out")
    del draft["assets"][0]["path"]
    draft["assets"][0]["evidence_id"] = "frame:119"
    with pytest.raises(VidLensError, match="Read this video's"):
        service.export_article(video.id, revision, draft, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("svg", [
    '<svg onload="alert(1)"/>',
    '<svg><script>alert(1)</script></svg>',
    '<svg><foreignObject><iframe/></foreignObject></svg>',
    '<svg><rect fill="url(https://example.org/a)"/></svg>',
    '<svg xml:base="https://example.org"><rect fill="url(#x)"/></svg>',
    '<?xml-stylesheet href="https://example.org/a.css"?><svg/>',
    '<svg><rect fill="u\\72l(https://example.org/a)"/></svg>',
    '<svg xmlns="https://example.org/foreign"/>',
])
def test_diagrams_reject_active_or_remote_content(svg):
    with pytest.raises(VidLensError):
        article.passive_svg(svg)


@pytest.mark.parametrize("extra", [
    "![Outside][photo]\n\n[photo]: https://example.org/tracking.png",
    '<img src="https://example.org/tracking.png">',
    "[Private](../../private/secrets.txt)",
])
def test_all_markdown_asset_forms_are_checked(tmp_path, extra):
    service, video, revision, draft, _, _, _ = seed(tmp_path)
    draft["markdown"] += "\n\n" + extra
    with pytest.raises(VidLensError) as error:
        service.export_article(video.id, revision, draft, tmp_path / "out")
    assert error.value.code in {"nonportable_image", "nonportable_article"}


def test_export_never_overwrites_existing_or_raced_output(tmp_path, monkeypatch):
    service, video, revision, draft, _, _, _ = seed(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "user.md").write_text("Keep this work.")
    with pytest.raises(VidLensError) as error:
        service.export_article(video.id, revision, draft, output)
    assert error.value.code == "output_exists"
    assert (output / "user.md").read_text() == "Keep this work."
    original = article._publish

    def raced(stage, target):
        target.mkdir()
        (target / "created-during-export.txt").write_text("Keep this too.")
        original(stage, target)

    monkeypatch.setattr(article, "_publish", raced)
    output = tmp_path / "raced"
    with pytest.raises(VidLensError) as error:
        service.export_article(video.id, revision, draft, output)
    assert error.value.code == "output_exists"
    assert (output / "created-during-export.txt").read_text() == "Keep this too."
    assert not list(tmp_path.glob(".vidlens-article-*"))


def test_knowledge_change_blocks_export_even_when_replanned_draft_is_valid(tmp_path, monkeypatch):
    service, video, revision, draft, _, unit, plan = seed(tmp_path)
    original = service.audit_article
    calls = 0

    def changed(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            service.save_knowledge(video.id, revision, [{**unit, "explanation": "New semantics."}])
            service.article_plan(video.id, revision, plan)
        return original(*args)

    monkeypatch.setattr(service, "audit_article", changed)
    with pytest.raises(VidLensError) as error:
        service.export_article(video.id, revision, draft, tmp_path / "out")
    assert error.value.code == "stale_article"
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".vidlens-article-*"))


def test_changed_source_pixels_block_export(tmp_path, monkeypatch):
    service, video, revision, draft, path, _, _ = seed(tmp_path)
    original = article._source_image
    calls = 0

    def changed(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            Image.new("RGB", (100, 50), "red").save(path)
        return original(*args)

    monkeypatch.setattr(article, "_source_image", changed)
    with pytest.raises(VidLensError) as error:
        service.export_article(video.id, revision, draft, tmp_path / "out")
    assert error.value.code == "stale_article"
    assert not (tmp_path / "out").exists()


def test_uncovered_core_and_private_paths_are_not_published(tmp_path):
    service, video, revision, draft, path, _, _ = seed(tmp_path)
    draft["coverage"] = []
    with pytest.raises(VidLensError) as error:
        service.export_article(video.id, revision, draft, tmp_path / "out")
    assert error.value.code == "article_incomplete"
    draft["coverage"] = [{"unit_id": "grid", "section_id": "grid",
                          "excerpt": "The center determines the responsible cell."}]
    draft["markdown"] += f"\nOriginal file: {path}\n"
    with pytest.raises(VidLensError) as error:
        service.export_article(video.id, revision, draft, tmp_path / "out")
    assert error.value.code == "nonportable_article"


def test_text_attachments_are_linked_hashed_portable_and_never_executed(tmp_path):
    service, video, revision, draft, _, _, _ = seed(tmp_path)
    script = 'raise RuntimeError("This file must not execute during export")\n'
    draft["attachments"] = [{"name": "calculations.py", "text": script},
                            {"name": "inputs.json", "text": '{"boxes": 2}\n'}]
    draft["markdown"] += "\n\n[Calculations](calculations.py) · [Inputs](./inputs.json)\n"
    output = tmp_path / "out"
    result = service.export_article(video.id, revision, draft, output)
    assert result["attachments"] == 2
    assert (output / "calculations.py").read_text() == script
    manifest = json.loads((output / "manifest.json").read_text())
    for item in manifest["attachments"]:
        data = (output / item["path"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(data).hexdigest()
        assert item["bytes"] == len(data)
    assert 'href="calculations.py"' in (output / "preview.html").read_text()


@pytest.mark.parametrize("attachment", [
    {"name": "../calculations.py", "text": "print(1)"},
    {"name": "manifest.json", "text": "{}"},
    {"name": "script.html", "text": "<script>run()</script>"},
    {"name": "too-large.txt", "text": "字" * 40000},
    {"name": "binary.txt", "text": "abc\x00def"},
])
def test_invalid_attachments_never_escape_or_replace_bundle_files(tmp_path, attachment):
    service, video, revision, draft, _, _, _ = seed(tmp_path)
    draft["attachments"] = [attachment]
    with pytest.raises(VidLensError) as error:
        service.export_article(video.id, revision, draft, tmp_path / "out")
    assert error.value.code == "invalid_attachment"
    assert not (tmp_path / "out").exists()


def test_attachment_requires_real_link_and_case_insensitive_unique_name(tmp_path):
    service, video, revision, draft, _, _, _ = seed(tmp_path)
    draft["attachments"] = [{"name": "calc.py", "text": "print(1)\n"}]
    draft["markdown"] += "\n`[Code](calc.py)`\n"
    with pytest.raises(VidLensError) as error:
        service.export_article(video.id, revision, draft, tmp_path / "out")
    assert error.value.code == "unused_attachment"
    draft["markdown"] += "\n[Code](calc.py) [Again](CALC.py)\n"
    draft["attachments"].append({"name": "CALC.py", "text": "print(2)\n"})
    with pytest.raises(VidLensError) as error:
        service.export_article(video.id, revision, draft, tmp_path / "out")
    assert error.value.code == "invalid_attachment"


def test_contents_links_have_real_markdown_and_html_section_targets(tmp_path):
    service, video, revision, draft, _, _, _ = seed(tmp_path)
    draft["markdown"] = "[Go to grid](#grid)\n\n" + draft["markdown"]
    output = tmp_path / "out"
    service.export_article(video.id, revision, draft, output)
    markdown = (output / "README.md").read_text()
    preview = (output / "preview.html").read_text()
    assert '<!-- section:grid -->\n<a id="grid"></a>' in markdown
    assert '<a href="#grid">Go to grid</a>' in preview
    assert '<a id="grid"></a>' in preview
    assert '&lt;a id="grid"' not in preview
    assert preview.index('<a id="grid"></a>') < preview.index("<h2>Grid</h2>")
    draft["markdown"] += "\n[Missing](#not-a-section)\n"
    with pytest.raises(VidLensError) as error:
        service.export_article(video.id, revision, draft, tmp_path / "bad")
    assert error.value.code == "missing_anchor"

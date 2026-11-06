#!/usr/bin/env python3
"""Render an offline HTML lesson; embedding local source images requires Pillow."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
import re
import sys
import warnings
from pathlib import Path
from urllib.parse import urlsplit

MAX_IMAGE_INPUT_BYTES = 20_000_000
MAX_IMAGE_PIXELS = 24_000_000
MAX_IMAGE_EDGE = 1920
MAX_IMAGE_BYTES = 1_200_000
MAX_TOTAL_IMAGE_BYTES = 8_000_000
MAX_IMAGES = 24
MAX_HTML_BYTES = 16_000_000


def _image_asset(path: Path) -> tuple[dict, int]:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:
        raise ValueError("Source images require Pillow; use the VidLens Python environment") from exc
    if not path.is_file():
        raise ValueError(f"Image must be a regular local file: {path}")
    if path.stat().st_size > MAX_IMAGE_INPUT_BYTES:
        raise ValueError("Image input exceeds 20 MB; use a smaller frame or crop")
    with path.open("rb") as source:
        raw = source.read(MAX_IMAGE_INPUT_BYTES + 1)
    if len(raw) > MAX_IMAGE_INPUT_BYTES:
        raise ValueError("Image input exceeds 20 MB; use a smaller frame or crop")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as original:
                if original.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Only JPEG, PNG and WebP source images are supported")
                if getattr(original, "n_frames", 1) != 1:
                    raise ValueError("Use one still frame, not an animated image")
                if original.width * original.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 24 megapixel decoding limit")
                original.load()
                oriented = ImageOps.exif_transpose(original).convert("RGBA")
                oriented.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
                # A fresh RGB canvas drops metadata and flattens transparency onto white.
                pixels = Image.new("RGB", oriented.size, "white")
                pixels.paste(oriented, mask=oriented.getchannel("A"))
                for quality in (88, 80, 72):
                    buffer = io.BytesIO()
                    pixels.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=0)
                    encoded = buffer.getvalue()
                    if len(encoded) <= MAX_IMAGE_BYTES:
                        break
                else:
                    raise ValueError("Compressed image exceeds 1.2 MB; crop or resize the frame")
    except (UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning, OSError) as exc:
        raise ValueError(f"Cannot decode a safe still image: {path.name}") from exc
    return {
        "data_uri": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
        "width": pixels.width,
        "height": pixels.height,
    }, len(encoded)


def _source_image(value: object, field: str, source_url: str) -> tuple[dict, str]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    path = _text(value.get("path"), f"{field}.path", required=True)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", path) or path.startswith("//"):
        raise ValueError(f"{field}.path must be a local file path, not a URL or data URI")
    timestamp = _number(value.get("timestamp"), f"{field}.timestamp")
    if timestamp < 0:
        raise ValueError(f"{field}.timestamp must be nonnegative")
    url = _source(value.get("source_url", source_url), f"{field}.source_url")
    if not url:
        raise ValueError(f"{field} requires a source_url (or the step's source_url)")
    seconds = int(timestamp)
    clock = f"{seconds // 60:02d}:{seconds % 60:02d}"
    if seconds >= 3600:
        clock = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    return {
        "alt": _text(value.get("alt"), f"{field}.alt", required=True),
        "caption": _text(value.get("caption"), f"{field}.caption", required=True),
        "timestamp": timestamp,
        "time_label": clock,
        "source_url": url,
    }, path


def _text(value: object, field: str, *, required: bool = False) -> str:
    if not isinstance(value, str) or len(value) > 20000 or (required and not value.strip()):
        raise ValueError(
            f"{field} must be {'a nonempty' if required else 'a'} string (max 20000 chars)"
        )
    return value


def _source(value: object, field: str) -> str:
    value = _text(value, field)
    if not value:
        return value
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or any(ord(char) < 33 for char in value)
    ):
        raise ValueError(f"{field} must be an http(s) URL without credentials or whitespace")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _simulation(value: object, field: str) -> dict:
    if not isinstance(value, dict) or value.get("type") != "softmax":
        raise ValueError(f"{field} must be an object with type 'softmax'")
    labels, logits = value.get("labels"), value.get("logits")
    if (
        not isinstance(labels, list)
        or not isinstance(logits, list)
        or not 2 <= len(labels) <= 12
        or len(labels) != len(logits)
    ):
        raise ValueError(f"{field}.labels and logits must be equal-length arrays with 2..12 items")
    names = [
        _text(label, f"{field}.labels[{index}]", required=True)
        for index, label in enumerate(labels)
    ]
    if any(len(label) > 160 for label in names):
        raise ValueError(f"{field}.labels must have at most 160 characters each")
    scores = [_number(score, f"{field}.logits[{index}]") for index, score in enumerate(logits)]
    temperature = _number(value.get("temperature", 1), f"{field}.temperature")
    if not 0.1 <= temperature <= 3:
        raise ValueError(f"{field}.temperature must be between 0.1 and 3")
    return {"type": "softmax", "labels": names, "logits": scores, "temperature": temperature}


def validate_lesson(value: object, *, base_dir: Path | None = None) -> dict:
    if not isinstance(value, dict):
        raise TypeError("The lesson must be a JSON object")
    language = value.get("language", "zh-CN")
    if language not in {"zh-CN", "en"}:
        raise ValueError("language must be zh-CN or en")
    steps = value.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 200:
        raise ValueError("steps must contain between 1 and 200 steps")
    result = {
        "title": _text(value.get("title"), "title", required=True),
        "intro": _text(value.get("intro", ""), "intro"),
        "language": language,
        "steps": [],
        "images": {},
    }
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    paths: dict[Path, str] = {}
    image_bytes = 0
    for index, step in enumerate(steps):
        field = f"steps[{index}]"
        if not isinstance(step, dict):
            raise TypeError(f"{field} must be an object")
        normalized = {
            "title": _text(step.get("title"), f"{field}.title", required=True),
            "caption": _text(step.get("caption", ""), f"{field}.caption"),
            "explanation": _text(step.get("explanation"), f"{field}.explanation", required=True),
            "diagram_label": _text(step.get("diagram_label", ""), f"{field}.diagram_label"),
            "source_url": _source(step.get("source_url", ""), f"{field}.source_url"),
            "source_label": _text(step.get("source_label", ""), f"{field}.source_label"),
            "state": [],
        }
        state = step.get("state", [])
        if not isinstance(state, list) or len(state) > 24:
            raise ValueError(f"{field}.state must contain at most 24 items")
        for item in state:
            if not isinstance(item, dict) or not isinstance(item.get("active", False), bool):
                raise TypeError(f"{field}.state items must be objects with a boolean active field")
            normalized["state"].append(
                {
                    "label": _text(item.get("label"), f"{field}.state.label", required=True),
                    "value": _text(item.get("value", ""), f"{field}.state.value"),
                    "active": item.get("active", False),
                }
            )
        if "simulation" in step:
            normalized["simulation"] = _simulation(step["simulation"], f"{field}.simulation")
        if "image" in step:
            picture, path_text = _source_image(
                step["image"], f"{field}.image", normalized["source_url"]
            )
            path = Path(path_text).expanduser()
            path = (path if path.is_absolute() else base_dir / path).resolve()
            if path not in paths:
                asset, size = _image_asset(path)
                asset_id = hashlib.sha256(asset["data_uri"].encode("ascii")).hexdigest()
                if asset_id not in result["images"]:
                    image_bytes += size
                    if len(result["images"]) >= MAX_IMAGES or image_bytes > MAX_TOTAL_IMAGE_BYTES:
                        raise ValueError("Use at most 24 unique images and 8 MB of encoded JPEG data")
                    result["images"][asset_id] = asset
                paths[path] = asset_id
            normalized["image"] = {**picture, "asset_id": paths[path]}
        result["steps"].append(normalized)
    return result


SCRIPT = r"""
'use strict';
const lesson = JSON.parse(document.getElementById('lesson-data').textContent);
const byId = id => document.getElementById(id);
const english = lesson.language === 'en';
const labels = english
  ? {previous:'Previous', next:'Next', step:'Step', of:'of', choose:'Choose a step',
     source:'Source', active:'Active', help:'Use ← / → to move; Home / End to jump.', diagram:'State',
     experiment:'Adjust temperature and inspect the probabilities', temperature:'Temperature T',
     simulationHelp:'Focus the slider and use arrow keys to adjust it. Lower temperature concentrates unequal scores; higher temperature flattens them. Equal scores stay equal.',
     total:'Probability sum', rounding:'Values are rounded for display.', score:'logit', frame:'Video frame'}
  : {previous:'上一步', next:'下一步', step:'步骤', of:'/', choose:'选择步骤',
     source:'查看来源', active:'当前变化', help:'使用 ← / → 切换；Home / End 跳到首尾。', diagram:'状态示意',
     experiment:'调节温度，观察概率怎样变化', temperature:'温度 T',
     simulationHelp:'聚焦滑块后可用方向键调节。有分数差异时，低温使概率更集中，高温使概率更平缓；分数相同时始终均匀。',
     total:'概率之和', rounding:'显示数值经过四舍五入。', score:'分数', frame:'视频画面'};
let current = 0;
const temperatures = lesson.steps.map(step => step.simulation ? step.simulation.temperature : null);
byId('heading').textContent = lesson.title;
byId('intro').textContent = lesson.intro;
byId('previous').textContent = labels.previous;
byId('next').textContent = labels.next;
byId('choose-label').textContent = labels.choose;
byId('help').textContent = labels.help;
byId('simulation-title').textContent = labels.experiment;
byId('temperature-label').textContent = labels.temperature;
byId('simulation-help').textContent = labels.simulationHelp;
lesson.steps.forEach((step, index) => {
  const option = document.createElement('option');
  option.value = String(index);
  option.textContent = `${index + 1}. ${step.title}`;
  byId('choose').append(option);
});
function softmax(logits, temperature) {
  const maximum = Math.max(...logits);
  const weights = logits.map(value => Math.exp((value - maximum) / temperature));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map(value => value / total);
}
function renderProbabilities() {
  const simulation = lesson.steps[current].simulation;
  if (!simulation) return;
  const supplied = Number(byId('temperature').value);
  const temperature = Number.isFinite(supplied)
    ? Math.max(0.1, Math.min(3, supplied)) : temperatures[current];
  temperatures[current] = temperature;
  byId('temperature').value = String(temperature);
  byId('temperature-value').textContent = temperature.toFixed(2);
  byId('temperature').setAttribute('aria-valuetext', `${labels.temperature}: ${temperature.toFixed(2)}`);
  const probabilities = softmax(simulation.logits, temperature);
  byId('probabilities').replaceChildren();
  probabilities.forEach((probability, index) => {
    const row = document.createElement('li');
    row.className = 'probability-row';
    const heading = document.createElement('div');
    heading.className = 'probability-heading';
    const name = document.createElement('strong');
    name.textContent = simulation.labels[index];
    const score = document.createElement('span');
    score.textContent = `${labels.score} = ${simulation.logits[index]}`;
    const value = document.createElement('output');
    value.textContent = `${probability.toFixed(6)} (${(probability * 100).toFixed(2)}%)`;
    heading.append(name, score, value);
    const track = document.createElement('div');
    track.className = 'probability-track';
    track.setAttribute('aria-hidden', 'true');
    const bar = document.createElement('div');
    bar.className = 'probability-bar';
    bar.style.width = `${probability * 100}%`;
    track.append(bar);
    row.append(heading, track);
    byId('probabilities').append(row);
  });
  const sum = probabilities.reduce((total, value) => total + value, 0);
  byId('probability-total').textContent = `${labels.total} = ${sum.toFixed(6)} · ${labels.rounding}`;
}
function render(index) {
  current = Math.max(0, Math.min(lesson.steps.length - 1, index));
  const step = lesson.steps[current];
  byId('counter').textContent = `${labels.step} ${current + 1} ${labels.of} ${lesson.steps.length}`;
  byId('step-title').textContent = step.title;
  byId('caption').textContent = step.caption;
  byId('caption').hidden = !step.caption;
  byId('explanation').textContent = step.explanation;
  byId('source-figure').hidden = !step.image;
  byId('source-figure').replaceChildren();
  if (step.image) {
    const picture = step.image;
    const asset = lesson.images[picture.asset_id];
    const img = document.createElement('img');
    img.src = asset.data_uri;
    img.alt = picture.alt;
    img.width = asset.width;
    img.height = asset.height;
    const caption = document.createElement('figcaption');
    const description = document.createElement('p');
    description.textContent = picture.caption;
    const link = document.createElement('a');
    link.href = picture.source_url;
    link.textContent = `${labels.frame} · ${picture.time_label}`;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    caption.append(description, link);
    byId('source-figure').append(img, caption);
  }
  byId('diagram-label').textContent = step.diagram_label || labels.diagram;
  byId('diagram').hidden = !step.state.length;
  byId('state').replaceChildren();
  step.state.forEach(item => {
    const card = document.createElement('li');
    card.className = item.active ? 'state-item active' : 'state-item';
    const label = document.createElement('strong');
    label.textContent = item.label;
    const value = document.createElement('span');
    value.textContent = item.value;
    card.append(label, value);
    if (item.active) {
      const badge = document.createElement('small');
      badge.textContent = labels.active;
      card.append(badge);
    }
    byId('state').append(card);
  });
  byId('source').hidden = !step.source_url;
  byId('source').href = step.source_url || '#';
  byId('source').textContent = step.source_label || labels.source;
  byId('choose').value = String(current);
  byId('previous').disabled = current === 0;
  byId('next').disabled = current === lesson.steps.length - 1;
  byId('progress').value = current + 1;
  byId('progress').max = lesson.steps.length;
  byId('progress').setAttribute('aria-label', byId('counter').textContent);
  byId('simulation').hidden = !step.simulation;
  byId('probabilities').replaceChildren();
  if (step.simulation) {
    byId('temperature').value = String(temperatures[current]);
    renderProbabilities();
  }
}
byId('previous').addEventListener('click', () => render(current - 1));
byId('next').addEventListener('click', () => render(current + 1));
byId('choose').addEventListener('change', event => render(Number(event.target.value)));
byId('temperature').addEventListener('input', renderProbabilities);
document.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey
      || ['SELECT', 'INPUT', 'TEXTAREA'].includes(event.target.tagName)
      || event.target.isContentEditable) return;
  const target = {ArrowLeft:current - 1, ArrowRight:current + 1, Home:0, End:lesson.steps.length - 1};
  if (Object.hasOwn(target, event.key)) { event.preventDefault(); render(target[event.key]); }
});
render(0);
""".strip()


TEMPLATE = """<!doctype html>
<html lang="__LANGUAGE__"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; script-src 'sha256-__SCRIPT_HASH__'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>__TITLE__</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,-apple-system,sans-serif;line-height:1.65;color:#172d35;background:#f4f7f8}
*{box-sizing:border-box}body{margin:0;padding:32px 20px}main{max-width:920px;margin:auto}
h1{font-size:clamp(1.7rem,5vw,2.5rem);line-height:1.25;margin:0 0 16px}h2{line-height:1.35;margin:8px 0 22px}
p{white-space:pre-wrap;overflow-wrap:anywhere}#intro,#help{color:#52656c}#intro{max-width:75ch}
.reader{padding:clamp(20px,4vw,36px);background:#fff;border:1px solid #dce5e7;border-radius:16px;margin:24px 0}
#counter{font-size:.9rem;color:#52656c}#caption{border-left:3px solid #58a5a6;padding:10px 18px;background:#f1f7f7}
#explanation{font-size:1.06rem}#diagram{margin:28px 0}#diagram-label{font-weight:600}
#source-figure{margin:24px 0}#source-figure img{display:block;width:100%;height:auto;border-radius:8px;background:#fff}
#source-figure figcaption{font-size:.92rem;margin-top:10px}#source-figure figcaption p{margin:0 0 4px}
#state{display:flex;flex-wrap:wrap;gap:12px;padding:0;list-style:none;align-items:stretch}
.state-item{flex:1 1 170px;max-width:100%;padding:16px;border:1px solid #cad9de;border-radius:10px;background:#f7fafb;overflow-wrap:anywhere}
.state-item strong,.state-item span{display:block;white-space:pre-wrap}.state-item span{font-family:ui-monospace,monospace;margin-top:8px}
.state-item.active{background:#e6f5f2;border:2px solid #137367;padding:15px}.state-item small{display:block;color:#11675d;margin-top:12px}
a{color:#126e7b}nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button,select{font:inherit;padding:9px 14px;border:1px solid #a8bdc4;border-radius:8px;background:#fff;color:inherit;min-height:44px}
button{cursor:pointer}button:disabled{opacity:.45;cursor:default}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #2189a0;outline-offset:3px}
select{max-width:100%;flex:1;min-width:140px}progress{display:block;width:100%;height:7px;accent-color:#147a70;margin-bottom:22px}
#source{display:inline-block;margin-top:8px}#help{font-size:.85rem;margin-bottom:0}[hidden]{display:none!important}
#simulation{margin:28px 0;padding:20px;border:1px solid #b9d3d1;border-radius:12px;background:#f3faf8}
#simulation h3{margin:0 0 12px}#simulation-help,#probability-total{font-size:.9rem}#softmax-formula{font-family:ui-monospace,monospace;overflow-wrap:anywhere}
.temperature-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;font-weight:600}
#temperature{display:block;width:100%;min-height:44px;accent-color:#147a70;margin:4px 0 12px}#temperature:focus-visible{outline:3px solid #2189a0;outline-offset:3px}
#probabilities{list-style:none;padding:0;margin:18px 0}.probability-row{margin:16px 0}.probability-heading{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:baseline}
.probability-heading strong{flex:1 1 100px;overflow-wrap:anywhere}.probability-heading span,.probability-heading output{font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.probability-track{height:16px;margin-top:7px;border-radius:5px;background:#d9e8e5;overflow:hidden}.probability-bar{height:100%;background:#13796c;border-radius:5px}
@media(prefers-color-scheme:dark){:root{background:#14252d;color:#e5eff0}.reader,button,select{background:#1d333d;border-color:#405963}#intro,#help,#counter{color:#b4c7ce}#caption,.state-item{background:#243f47;border-color:#58757b}.state-item.active{background:#224e49;border-color:#70c8b2}.state-item small,a{color:#95dbcb}#simulation{background:#213f3b;border-color:#527c73}.probability-track{background:#42615b}.probability-bar{background:#79d8bd}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style></head><body><main>
<h1 id="heading">__TITLE__</h1><p id="intro"></p>
<section class="reader" aria-labelledby="step-title">
<progress id="progress" value="1" max="1"></progress>
<div id="counter" aria-live="polite" aria-atomic="true"></div>
<h2 id="step-title"></h2><p id="caption" hidden></p><p id="explanation"></p>
<figure id="source-figure" hidden></figure>
<figure id="diagram" aria-labelledby="diagram-label" hidden><figcaption id="diagram-label"></figcaption><ol id="state"></ol></figure>
<section id="simulation" aria-labelledby="simulation-title" hidden>
<h3 id="simulation-title"></h3><p id="simulation-help"></p>
<p id="softmax-formula">pᵢ = exp((zᵢ − max(z)) / T) / Σⱼ exp((zⱼ − max(z)) / T)</p>
<div class="temperature-heading"><label for="temperature" id="temperature-label"></label><output id="temperature-value" for="temperature" aria-live="polite"></output></div>
<input id="temperature" type="range" min="0.1" max="3" step="0.01" value="1" aria-describedby="simulation-help">
<ol id="probabilities"></ol><p id="probability-total"></p>
</section>
<a id="source" target="_blank" rel="noopener noreferrer" hidden></a>
</section>
<nav aria-label="Lesson navigation"><button id="previous" type="button"></button>
<label for="choose" id="choose-label"></label><select id="choose"></select><button id="next" type="button"></button></nav>
<p id="help"></p><noscript>This lesson needs JavaScript to switch steps. It runs entirely offline.</noscript>
</main><script id="lesson-data" type="application/json">__LESSON_DATA__</script>
<script>__SCRIPT__</script></body></html>
"""


def render_lesson(value: object, *, base_dir: Path | None = None) -> str:
    lesson = validate_lesson(value, base_dir=base_dir)
    # Prevent raw-text script termination and HTML parsing of all external strings.
    payload = json.dumps(lesson, ensure_ascii=False).replace("&", "\\u0026")
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    digest = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
    substitutions = {
        "SCRIPT_HASH": digest,
        "SCRIPT": SCRIPT,
        "LANGUAGE": lesson["language"],
        "TITLE": html.escape(lesson["title"]),
        "LESSON_DATA": payload,
    }
    # A single pass keeps input strings containing template markers unchanged.
    rendered = re.sub(
        r"__(SCRIPT_HASH|SCRIPT|LANGUAGE|TITLE|LESSON_DATA)__",
        lambda match: substitutions[match[1]],
        TEMPLATE,
    )
    if len(rendered.encode("utf-8")) > MAX_HTML_BYTES:
        raise ValueError("Lesson HTML exceeds 16 MB; use fewer images or shorter steps")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Lesson JSON (see references/interactive.md)")
    parser.add_argument("--output", type=Path, required=True, help="Self-contained HTML output")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    try:
        if args.input.stat().st_size > 2_000_000:
            raise ValueError("Lesson JSON must be smaller than 2 MB")
        lesson = json.loads(args.input.read_text(encoding="utf-8"))
        rendered = render_lesson(lesson, base_dir=args.input.resolve().parent)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w" if args.force else "x", encoding="utf-8") as output:
            output.write(rendered)
        print(
            json.dumps(
                {"html": str(args.output.resolve()), "steps": len(lesson["steps"])},
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import base64
import hashlib
import io
import json
import re
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LESSON = runpy.run_path(str(ROOT / "skills/read-video/scripts/build_lesson.py"))


def example():
    return json.loads((ROOT / "skills/read-video/assets/lesson-example.json").read_text())


def embedded_data(rendered):
    payload = re.search(
        r'<script id="lesson-data" type="application/json">(.*?)</script>', rendered, re.DOTALL
    )[1]
    return json.loads(payload)


def test_stepper_preserves_example_states_and_is_deterministic():
    data = example()
    rendered = LESSON["render_lesson"](data)
    assert rendered == LESSON["render_lesson"](data)
    actual = embedded_data(rendered)
    assert actual["title"] == data["title"]
    expected = [[{"active": False, **item} for item in step["state"]] for step in data["steps"]]
    assert [step["state"] for step in actual["steps"]] == expected
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", rendered, re.DOTALL)
    assert len(scripts) == 2
    script_hash = base64.b64encode(hashlib.sha256(scripts[1].encode()).digest()).decode()
    assert f"script-src 'sha256-{script_hash}'" in rendered


def test_external_text_cannot_terminate_script_or_act_as_template():
    hostile = '</script><script>alert("bad")</script><img src=x onerror=alert(1)>&__LESSON_DATA__'
    data = {
        "title": hostile,
        "steps": [
            {
                "title": "__TITLE__",
                "explanation": hostile,
                "caption": "text\u2028next",
                "state": [{"label": hostile, "value": hostile}],
            }
        ],
    }
    rendered = LESSON["render_lesson"](data)
    assert rendered.count("<script") == 2
    assert "<img" not in rendered
    assert embedded_data(rendered)["steps"][0]["explanation"] == hostile
    assert embedded_data(rendered)["title"] == hostile
    assert embedded_data(rendered)["steps"][0]["title"] == "__TITLE__"


def test_softmax_labels_remain_data_and_keep_the_script_hash_boundary():
    hostile = '</script><script>alert("bad")</script><img src=x onerror=alert(1)>'
    data = example()
    data["steps"][1]["simulation"]["labels"][0] = hostile
    rendered = LESSON["render_lesson"](data)
    assert rendered.count("<script") == 2
    assert "<img" not in rendered
    assert embedded_data(rendered)["steps"][1]["simulation"]["labels"][0] == hostile
    assert "innerHTML" not in LESSON["SCRIPT"]
    assert 'id="temperature" type="range" min="0.1" max="3"' in rendered
    assert 'for="temperature" id="temperature-label"' in rendered
    assert "prefers-reduced-motion:reduce" in rendered


@pytest.mark.parametrize(
    "simulation",
    [
        None,
        [],
        {"type": "execute"},
        {"type": "softmax", "labels": ["A"], "logits": [1]},
        {"type": "softmax", "labels": ["A", "B"], "logits": [1]},
        {"type": "softmax", "labels": ["A"] * 13, "logits": [1] * 13},
        {"type": "softmax", "labels": "AB", "logits": [1, 2]},
        {"type": "softmax", "labels": ["A", "B"], "logits": "12"},
        {"type": "softmax", "labels": ["A", ""], "logits": [1, 2]},
        {"type": "softmax", "labels": ["A", 2], "logits": [1, 2]},
        {"type": "softmax", "labels": ["A", "B" * 161], "logits": [1, 2]},
    ],
)
def test_softmax_rejects_invalid_type_labels_and_dimensions(simulation):
    data = example()
    data["steps"][1]["simulation"] = simulation
    with pytest.raises((TypeError, ValueError)):
        LESSON["render_lesson"](data)


@pytest.mark.parametrize(
    "number", [True, "2", None, float("nan"), float("inf"), -float("inf"), 10**400]
)
def test_softmax_rejects_nonfinite_or_nonnumeric_logits(number):
    data = example()
    data["steps"][1]["simulation"]["logits"] = [0, number]
    with pytest.raises((TypeError, ValueError), match="finite number"):
        LESSON["render_lesson"](data)


@pytest.mark.parametrize(
    "temperature", [True, "1", None, float("nan"), float("inf"), -1, 0, 0.099, 3.01]
)
def test_softmax_rejects_unsafe_temperature(temperature):
    data = example()
    data["steps"][1]["simulation"]["temperature"] = temperature
    with pytest.raises((TypeError, ValueError), match="temperature"):
        LESSON["render_lesson"](data)


def test_softmax_temperature_default_and_full_finite_input_range():
    data = example()
    data["steps"][1]["simulation"].pop("temperature")
    data["steps"][1]["simulation"]["logits"] = [1e308, -1e308]
    normalized = embedded_data(LESSON["render_lesson"](data))
    assert normalized["steps"][1]["simulation"]["temperature"] == 1
    assert normalized["steps"][1]["simulation"]["logits"] == [1e308, -1e308]


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/html,hi",
        "file:///tmp/a",
        "https://user:secret@example.com/",
        "https://example.com/\nattack",
    ],
)
def test_stepper_rejects_non_source_links(url):
    data = example()
    data["steps"][0]["source_url"] = url
    with pytest.raises(ValueError, match="http"):
        LESSON["render_lesson"](data)


def test_stepper_accepts_timestamped_source_links():
    data = example()
    data["steps"][0]["source_url"] = "https://www.bilibili.com/video/BV123?p=2&t=12"
    rendered = LESSON["render_lesson"](data)
    assert embedded_data(rendered)["steps"][0]["source_url"] == data["steps"][0]["source_url"]


def test_cli_does_not_replace_existing_artifact(tmp_path, monkeypatch, capsys):
    source = tmp_path / "lesson.json"
    output = tmp_path / "lesson.html"
    source.write_text(json.dumps(example()))
    monkeypatch.setattr("sys.argv", ["build_lesson.py", str(source), "--output", str(output)])
    assert LESSON["main"]() == 0
    assert json.loads(capsys.readouterr().out)["steps"] == 3
    output.write_text("user edits")
    assert LESSON["main"]() == 1
    assert output.read_text() == "user edits"


def run_javascript(data, assertions):
    # Execute the fixed script against a tiny DOM model; this does not test browser layout.
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional for the JS event/computation tests")
    payload = json.dumps(LESSON["validate_lesson"](data), ensure_ascii=True)
    harness = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tagName = 'DIV') {
    this.tagName = tagName; this.children = []; this.events = {}; this.style = {};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(name, handler) { this.events[name] = handler; }
}
const elements = new Map();
const document = {
  events: {},
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  },
  createElement(tag) { return new Element(tag.toUpperCase()); },
  addEventListener(name, handler) { this.events[name] = handler; }
};
document.getElementById('lesson-data').textContent = PAYLOAD;
""".replace("PAYLOAD", json.dumps(payload))
    result = subprocess.run(
        [node, "-"],
        input=harness + LESSON["SCRIPT"] + assertions,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_navigation_and_keyboard_events_update_the_lesson_state():
    assertions = r"""
assert.equal(byId('counter').textContent, '步骤 1 / 3');
assert.equal(byId('previous').disabled, true);
assert.equal(byId('state').children[0].children[1].textContent, '[2, 1]');
byId('next').events.click();
assert.equal(byId('counter').textContent, '步骤 2 / 3');
assert.equal(byId('state').children[2].children[1].textContent, '[0, -1]');
let prevented = 0;
const press = (key, tagName = 'BODY') => document.events.keydown({key, target:{tagName},
  preventDefault() { prevented++; }});
press('End');
assert.equal(byId('counter').textContent, '步骤 3 / 3');
assert.equal(byId('next').disabled, true);
press('Home', 'SELECT');
assert.equal(byId('counter').textContent, '步骤 3 / 3');
press('Home');
assert.equal(byId('counter').textContent, '步骤 1 / 3');
assert.equal(prevented, 2);
byId('choose').events.change({target:{value:'2'}});
assert.equal(byId('counter').textContent, '步骤 3 / 3');
byId('previous').events.click();
assert.equal(byId('counter').textContent, '步骤 2 / 3');
"""
    run_javascript(example(), assertions)


def test_slider_input_recomputes_actual_softmax_bars_and_visible_values():
    run_javascript(
        example(),
        r"""
assert.equal(byId('simulation').hidden, true);
byId('next').events.click();
assert.equal(byId('simulation').hidden, false);
assert.equal(byId('temperature-value').textContent, '1.00');
const firstRow = () => byId('probabilities').children[0];
const firstProbability = () => parseFloat(firstRow().children[1].children[0].style.width) / 100;
assert.equal(firstRow().children[0].children[0].textContent, '词 A');
assert.equal(firstRow().children[0].children[1].textContent, '分数 = 2');
assert.equal(firstRow().children[0].children[2].textContent, '0.731059 (73.11%)');
assert.ok(Math.abs(firstProbability() - 0.7310585786300049) < 1e-14);
byId('temperature').value = '0.5';
byId('temperature').events.input({target:byId('temperature')});
assert.ok(Math.abs(firstProbability() - 0.8807970779778823) < 1e-14);
assert.equal(firstRow().children[0].children[2].textContent, '0.880797 (88.08%)');
assert.equal(byId('temperature')['aria-valuetext'], '温度 T: 0.50');
assert.ok(byId('probability-total').textContent.startsWith('概率之和 = 1.000000'));
byId('temperature').value = '3';
byId('temperature').events.input({target:byId('temperature')});
assert.ok(Math.abs(firstProbability() - 0.5825702064623147) < 1e-14);
// Native range keyboard handling must not bubble into lesson navigation.
document.events.keydown({key:'ArrowRight', target:{tagName:'INPUT'},
  preventDefault() { assert.fail('Lesson navigation intercepted the temperature slider'); }});
assert.equal(byId('counter').textContent, '步骤 2 / 3');
""",
    )


def test_each_step_restores_its_temperature_and_hides_absent_simulations():
    run_javascript(
        example(),
        r"""
const setTemperature = value => {
  byId('temperature').value = String(value);
  byId('temperature').events.input({target:byId('temperature')});
};
byId('next').events.click();
setTemperature(0.25);
byId('next').events.click();
assert.equal(byId('temperature').value, '0.5');
setTemperature(3);
byId('previous').events.click();
assert.equal(byId('temperature').value, '0.25');
byId('previous').events.click();
assert.equal(byId('simulation').hidden, true);
assert.equal(byId('probabilities').children.length, 0);
byId('next').events.click();
assert.equal(byId('temperature').value, '0.25');
byId('next').events.click();
assert.equal(byId('temperature').value, '3');
setTemperature(100);
assert.equal(byId('temperature').value, '3');
setTemperature(0);
assert.equal(byId('temperature').value, '0.1');
setTemperature('not a number');
assert.equal(byId('temperature').value, '0.1');
""",
    )


def test_softmax_computation_is_stable_for_extreme_logits_and_sums_to_one():
    run_javascript(
        example(),
        r"""
for (const temperature of [0.1, 0.5, 1, 3]) {
  for (const logits of [[10000,9999], [-10000,-10001], [1e308,-1e308,0],
                        Array(12).fill(1e308), [-1e308,-1e308], [1e-300,-1e-300]]) {
    const probabilities = softmax(logits, temperature);
    assert.ok(probabilities.every(value => Number.isFinite(value) && value >= 0 && value <= 1));
    assert.ok(Math.abs(probabilities.reduce((a,b) => a+b, 0) - 1) < 1e-14);
  }
}
assert.deepEqual(softmax([1e308,-1e308,0], 0.1), [1,0,0]);
assert.deepEqual(softmax([1e308,1e308], 0.1), [0.5,0.5]);
assert.ok(Math.abs(softmax([10000,9999], 1)[0] - 0.7310585786300049) < 1e-14);
""",
    )


def image_lesson(path):
    data = example()
    data["steps"][0]["image"] = {
        "path": str(path),
        "alt": "Q, K and V for the three input tokens",
        "caption": "Each row belongs to one token; the labels distinguish Q, K and V.",
        "timestamp": 135.5,
        "source_url": "https://www.bilibili.com/video/BV123?p=2&t=135",
    }
    return data


def test_local_frame_is_oriented_resized_reencoded_and_metadata_free(tmp_path):
    path = tmp_path / "frame.png"
    original = Image.new("RGB", (2400, 1200), "red")
    exif = Image.Exif()
    exif[274] = 6  # Rotate to portrait before resizing.
    exif[315] = "private author metadata"
    original.save(path, exif=exif)
    rendered = LESSON["render_lesson"](image_lesson(path))
    data = embedded_data(rendered)
    asset = data["images"][data["steps"][0]["image"]["asset_id"]]
    assert str(path) not in rendered
    assert "private author metadata" not in rendered
    assert asset["data_uri"].startswith("data:image/jpeg;base64,")
    binary = base64.b64decode(asset["data_uri"].split(",", 1)[1])
    with Image.open(io.BytesIO(binary)) as image:
        assert image.format == "JPEG"
        assert image.size == (960, 1920)
        assert not image.getexif()
        assert image.getpixel((0, 0))[0] > 240
    assert (asset["width"], asset["height"]) == (960, 1920)
    assert "img-src data:" in rendered
    assert not re.search(r'<(?:script|img|link)[^>]+(?:src|href)=["\']https?://', rendered)


def test_transparency_is_flattened_and_repeated_image_data_is_shared(tmp_path):
    path = tmp_path / "frame.webp"
    Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(path, lossless=True)
    data = image_lesson(path)
    data["steps"][1]["image"] = {**data["steps"][0]["image"], "caption": "Second observation"}
    copy = tmp_path / "same-pixels.webp"
    copy.write_bytes(path.read_bytes())
    data["steps"][2]["image"] = {**data["steps"][0]["image"], "path": str(copy)}
    normalized = embedded_data(LESSON["render_lesson"](data))
    assert len(normalized["images"]) == 1
    first, second = [step["image"] for step in normalized["steps"][:2]]
    assert first["asset_id"] == second["asset_id"]
    assert normalized["steps"][2]["image"]["asset_id"] == first["asset_id"]
    assert first["caption"] != second["caption"]
    encoded = normalized["images"][first["asset_id"]]["data_uri"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)


def test_image_can_inherit_step_source_and_keeps_fractional_timestamp(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (10, 10)).save(path)
    data = image_lesson(path)
    step = data["steps"][0]
    step["source_url"] = step["image"].pop("source_url")
    step["image"]["timestamp"] = 3601.75
    picture = LESSON["validate_lesson"](data)["steps"][0]["image"]
    assert picture["source_url"] == step["source_url"]
    assert picture["timestamp"] == 3601.75
    assert picture["time_label"] == "01:00:01"


@pytest.mark.parametrize("path", ["https://example.com/frame.jpg", "file:///tmp/a.png",
                                  "data:image/svg+xml,<svg/>", "//example.com/a.png"])
def test_images_reject_urls_without_fetching(path):
    with pytest.raises(ValueError, match="local file path"):
        LESSON["render_lesson"](image_lesson(path))


def test_images_require_safe_attribution_and_accessible_descriptions(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (10, 10)).save(path)
    for field, value in [("alt", ""), ("caption", ""), ("timestamp", -1),
                         ("timestamp", True), ("timestamp", float("inf")),
                         ("source_url", "javascript:alert(1)")]:
        data = image_lesson(path)
        data["steps"][0]["image"][field] = value
        with pytest.raises((TypeError, ValueError)):
            LESSON["render_lesson"](data)


def test_images_reject_nonimages_animation_and_decode_bombs(tmp_path, monkeypatch):
    path = tmp_path / "pretend.png"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')
    with pytest.raises(ValueError, match="decode"):
        LESSON["render_lesson"](image_lesson(path))
    Image.new("RGB", (4, 4)).save(path, format="GIF")
    with pytest.raises(ValueError, match="JPEG, PNG and WebP"):
        LESSON["render_lesson"](image_lesson(path))
    Image.new("RGB", (4, 4), "red").save(
        path, format="PNG", save_all=True, append_images=[Image.new("RGB", (4, 4), "blue")]
    )
    with pytest.raises(ValueError, match="still frame"):
        LESSON["render_lesson"](image_lesson(path))
    Image.new("RGB", (4, 4)).save(path, format="PNG")
    monkeypatch.setitem(LESSON["_image_asset"].__globals__, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="megapixel"):
        LESSON["render_lesson"](image_lesson(path))


def test_image_and_document_budgets_fail_before_output(tmp_path, monkeypatch):
    path = tmp_path / "frame.png"
    Image.new("RGB", (10, 10)).save(path)
    values = LESSON["_image_asset"].__globals__
    for name, limit, error in [("MAX_IMAGE_INPUT_BYTES", 10, "input exceeds"),
                               ("MAX_IMAGE_BYTES", 10, "Compressed image"),
                               ("MAX_TOTAL_IMAGE_BYTES", 10, "24 unique images"),
                               ("MAX_IMAGES", 0, "24 unique images"),
                               ("MAX_HTML_BYTES", 10, "HTML exceeds")]:
        with monkeypatch.context() as change:
            change.setitem(values, name, limit)
            with pytest.raises(ValueError, match=error):
                LESSON["render_lesson"](image_lesson(path))


def test_cli_resolves_frames_relative_to_json_not_working_directory(tmp_path, monkeypatch, capsys):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    Image.new("RGB", (10, 10)).save(source_dir / "frame.png")
    source = source_dir / "lesson.json"
    source.write_text(json.dumps(image_lesson("frame.png")))
    output = tmp_path / "lesson.html"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["build_lesson.py", str(source), "--output", str(output)])
    assert LESSON["main"]() == 0
    assert json.loads(capsys.readouterr().out)["html"] == str(output)
    assert len(embedded_data(output.read_text())["images"]) == 1


def test_image_navigation_keeps_caption_alt_and_timestamp_with_the_frame(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (10, 10)).save(path)
    data = image_lesson(path)
    hostile = '</script><img src=x onerror=alert(1)>'
    data["steps"][0]["image"]["caption"] = hostile
    rendered = LESSON["render_lesson"](data)
    assert rendered.count("<script") == 2
    assert "<img src=x" not in rendered
    run_javascript(data, r"""
assert.equal(byId('source-figure').hidden, false);
const frame = () => byId('source-figure').children[0];
const caption = () => byId('source-figure').children[1];
assert.ok(frame().src.startsWith('data:image/jpeg;base64,'));
assert.equal(frame().alt, 'Q, K and V for the three input tokens');
assert.equal(frame().width, 10);
assert.equal(caption().children[0].textContent, '</script><img src=x onerror=alert(1)>');
assert.equal(caption().children[1].textContent, '视频画面 · 02:15');
assert.equal(caption().children[1].href, 'https://www.bilibili.com/video/BV123?p=2&t=135');
byId('next').events.click();
assert.equal(byId('source-figure').hidden, true);
assert.equal(byId('source-figure').children.length, 0);
assert.equal(byId('simulation').hidden, false);
byId('previous').events.click();
assert.equal(byId('source-figure').hidden, false);
assert.equal(caption().children[1].textContent, '视频画面 · 02:15');
""")

# Deliver an offline interactive explanation

Use this when stepping through a process helps the user understand changing state: a token entering a decoder, a tensor changing shape, a request moving through queues, or an experimental setup changing one variable. A short factual answer does not need an artifact.

The helper accepts JSON and creates one HTML file. It works offline, has keyboard navigation, respects reduced-motion preferences, and loads no CDN resources. Text fields are displayed as text; local source images are decoded and re-encoded before embedding. It never runs code from a transcript, a note or the JSON. Its fixed softmax simulation also accepts numerical data and computes an interactive probability distribution locally.

## Prepare the lesson

After reading the source, choose a concrete input and make each step explain one change. Preserve the role of each component across steps. Include a caption only when it adds source evidence or a helpful input; do not invent quotations. Add a timestamped source URL for each step based on the video, or clearly label the example as a teaching construction.

Required top-level fields: `title`, `steps` (1–200). Optional: `intro`, `language` (`zh-CN` or `en`). Each step has:

| Field | Content |
|---|---|
| `title` | Short step title; required |
| `explanation` | What changed and why; required |
| `caption` | Source excerpt or labeled teaching input; optional |
| `diagram_label` | Description of the displayed state; optional |
| `state` | Up to 24 cards with `label`, optional `value`, and boolean `active` |
| `source_url` | HTTP(S) URL of the actual source, preferably timestamped |
| `source_label` | Link text such as `视频 03:12` |
| `simulation` | Optional fixed softmax simulation, described below |
| `image` | Optional inspected local source frame with caption, timestamp and source URL |

Cards show state, not automatically inferred relationships. Describe a causal or data-flow link in the explanation when it matters. Static card values are supplied by the author; the softmax chart is calculated from its current slider value. Label any static comparison values with their reference temperature so they cannot be mistaken for the live result.

See [the interactive attention-weight example](../assets/lesson-example.json). Its logits are a labeled teaching example, not source evidence. Replace them with sourced values only when the video actually provides those values. Do not impose softmax on unrelated videos.

## Put a real video frame beside the explanation

Choose a few frames that carry the explanation. For each, point out a visible detail and explain its role in the process; avoid turning the lesson into a screenshot gallery. A step can combine a source image with a short state comparison or the softmax simulation.

Add this object to the appropriate step, using a path returned by VidLens that you have actually opened:

```json
"image": {
  "path": "frames/current-step.jpg",
  "alt": "The K matrix has three cached rows and one new row for 月",
  "caption": "The separate bottom row belongs to the new token 月; the earlier three rows are reused.",
  "timestamp": 520,
  "source_url": "https://www.bilibili.com/video/BV1SmrSBfEg6/?p=1&t=520"
}
```

`path`, `alt`, `caption` and a nonnegative numeric `timestamp` are required. Use the frame's returned timestamp, including fractions when present. `source_url` may be omitted if the step already has one; an HTTP(S) source is required either way. The figure displays its own timestamped source link, which can differ from a step's supplementary reference. Relative paths resolve against the lesson JSON directory, not the shell's working directory; absolute local paths also work.

The image loader needs Pillow, which is included in VidLens: run `.venv/bin/python <skill-directory>/scripts/build_lesson.py ...` from the checkout. Text-only lessons use the standard library. The loader accepts still JPEG, PNG and WebP, rejects URLs/data URIs, animation and undecodable inputs, and never downloads an image. It limits each source to 20 MB and 24 megapixels, applies EXIF orientation, removes metadata and flattens transparency onto white. Images fit within 1920×1920 and are re-encoded as JPEG, up to 1.2 MB each. Crop tiny formulas before embedding when necessary; inspect the resulting image for legibility.

Identical processed images share one embedded asset even when used in several steps. A lesson permits up to 24 unique images and 8 MB of JPEG data; the final HTML is capped at 16 MB. If a limit is exceeded, choose fewer or smaller frames. The output contains the image bytes and source attribution, with no dependency on the input files or their local paths. Source pages open only when the reader clicks a link.

## Add an adjustable softmax chart

Attach this object to a step:

```json
"simulation": {
  "type": "softmax",
  "labels": ["词 A", "词 B", "词 C"],
  "logits": [2, 1, 0],
  "temperature": 1
}
```

The two arrays must have the same length, between 2 and 12. Labels must be nonempty strings of at most 160 characters. Logits must be finite numbers; booleans, numeric strings, NaN and infinity are rejected. Initial temperature defaults to 1 and must be a finite number between 0.1 and 3. No user-supplied formula or executable code is accepted.

The page computes `exp((logit - max(logits)) / T)` and divides each weight by their sum. Subtracting the maximum prevents positive exponential overflow. The chart displays each label, original logit, probability and percentage, plus the probability sum. Displayed values are rounded; the calculation uses the unrounded probabilities. Changing the slider recomputes the chart immediately. Lower temperature concentrates unequal scores; higher temperature makes the distribution flatter. Equal scores stay equal at every temperature.

The native range slider supports keyboard arrows, Home and End while focused. Those keys do not navigate lesson steps while an input has focus. Each step retains its own adjusted temperature when leaving and returning; a step without a simulation hides the chart. The simulation requires no network, model, library or browser storage.

For an attention lesson, explain what the supplied scores represent. If they already include attention scaling, say so. Treat this extra temperature control as a teaching comparison unless the source itself uses that parameter. Attention weights are not the final value-weighted output, and this chart does not sample a token.

## Generate and check

From any working directory:

```sh
python3 <skill-directory>/scripts/build_lesson.py lesson.json --output lesson.html
```

The script refuses to overwrite an existing artifact unless `--force` is supplied. The script prints the absolute output path. When permitted, open that file in a browser, inspect the first and last steps, and test the next/previous buttons, slider and keyboard. Check image legibility, captions, wrapping at a narrow width, source links and the correctness of values. If a browser surface refuses local-file access, do not bypass that refusal; use the offline Node DOM tests for computation and event behavior, extract embedded JPEGs for image inspection, and report that browser layout was not verified. Deliver the HTML as a local file link. Keep the JSON when the user may want to edit it.

For a quick local demonstration:

```sh
python3 <skill-directory>/scripts/build_lesson.py <skill-directory>/assets/lesson-example.json --output /tmp/vidlens-lesson.html
```

The result is self-contained and can be shared as a file. External source links open only when the reader chooses them. A browser is still required to execute the fixed navigation script.

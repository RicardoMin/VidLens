# VidLens reading and notes

MCP uses ordinary stdio transport. The host must support MCP image content to inspect returned frames. VidLens retrieves evidence and stores notes; it does not call a separate hosted language model.

For small text, use `read_detail(video_id, timestamp, boxes=[[x,y,w,h], ...])` on a cached frame. Inspect its context image and source-pixel PNG tiles together, keeping headers and units within a region. `upgrade_source=True` explicitly requests a higher available stream and native PNG decode; compare actual `detail.source_size`, `decoded_source_size` and `source_scale`, not the requested quality. At most six regions and twelve tiles are returned; a smaller budget rejects incomplete coverage. A crop does not count as another whole frame read. CLI: `detail VIDEO_ID TIMESTAMP --box X Y W H --upgrade-source`.

For numerical charts and reports, use `check_facts(video_id, claims)` after reading original frames. Each claim supplies `id`, `timestamp`, `label`, `value` (numeric string), `unit`, `scope`, `meaning`, and an optional normalized `box`. The tool returns row-bound OCR comparisons and lossless source crops. It never approves metric meaning or calculations. Follow [the figure fact-check workflow](fact-checking.md), including an audit of every material quantitative assertion in the final prose, tables and captions. CLI: `check-facts VIDEO_ID --file claims.json` (JSON array).

For a local question, first locate the relevant speech, then use `read_frames(video_id, start, end, question=..., visual_queries=[...])`. The optional queries describe up to six distinct screen-evidence goals from the question. Local screen-text matching and a second pass around related changes select the final images; internal OCR probes do not count as images delivered to the AI. `visual.selection` reports a timeline fallback if OCR is unavailable. Check initial state, intermediate changes and result in the actual images; retrieve a narrower interval for any missing step. `read_section` accepts the same question options without `frame_cursor`. Intervals are limited to ten minutes.

When layout, arrows or handwriting carry the answer, add `preview=True` to see up to 24 labeled candidates in six-image sheets. Read the actual previews, choose their `read_timestamp` values, then call `read_frames(..., timestamps=[...])` without `question` to retrieve original pictures. Keep source `timestamp` for citation; a held low-frame-rate image can start before the requested interval. The preview pass costs additional image context; its thumbnails are not recorded as full-frame delivery. CLI equivalents are `frames ... --question '...' --preview` and `frames ... --timestamps 218 228 238 250`.

```sh
.venv/bin/python -m vidlens --home .vidlens frames VIDEO_ID 180 270 \
  --question '插入新值后，叶节点和父节点分别怎样变化？' \
  --visual-query '插入前的叶节点' --visual-query '分裂后向父节点提升的值' \
  --max-frames 4
```

## Whole-video reading

1. `read_video(url, intent="summary")` returns `mode="study_plan"`, `revision` and `sections`. It has not supplied the chapter's source text. Continue a truncated chapter map with `study_status(video_id, cursor=next_cursor)` before treating it as the full plan.
2. Read each chapter using `read_section(video_id, section_id, cursor=None, max_tokens=6000, include_frames=True)`. Continue with the returned `next_cursor` until it is absent; use `include_frames=False` for text-only continuation. The default images span the chapter. To inspect the candidate sequence in more detail, set `frame_cursor=0` and follow `visual.next_frame_cursor`; this is independent of the text cursor. `frame_limit` selects up to 12 images per request; the default is 4. Read the returned images, not only their OCR.
3. `scan_video(video_id, cursor=0, limit=100, view="navigation")` returns a compact map based on stable states, local image differences and visible detail. `view="full"` retains every raw change candidate. A navigation row's `source_candidate_cursor` points into the full map. Continue using the returned `view` and `next_cursor`; a failed selector returns the full view explicitly. Chapter image pagination uses `frame_view` and returns `visual.view`. Expand the full sequence or take a specific interval when an essential step is missing from the initial pictures.
4. Save each chapter's explanation with its current revision and evidence IDs. Notes are useful for synthesis and later questions, but never replace unread source material.

### Note payload

`save_note(video_id, section_id, revision, note)` accepts:

```json
{
  "title": "What this chapter establishes",
  "summary": "A concise explanation after reading the source.",
  "concepts": ["the actual concept name"],
  "roles": ["mechanism", "example"],
  "claims": [{"text": "A factual claim", "evidence_ids": ["COPY_RETURNED_ID"]}],
  "relations": [{
    "source": "component A",
    "target": "component B",
    "relation": "the specific causal or data-flow relationship",
    "evidence_ids": ["COPY_RETURNED_ID"]
  }]
}
```

Replace the illustrative content and IDs with actual evidence; omit unneeded optional fields. Roles describe the chapter's function: `definition`, `mechanism`, `derivation`, `code`, `example`, `experiment` or `correction`. Copy IDs and revision exactly from the tools. If the service reports stale evidence, refresh the chapter and review the affected claims before saving again.

### External references

When a prerequisite or extension would help, search and read primary documentation with the host's web tool. Then call `add_reference(video_id, section_id, revision, reference)` with `url`, `title`, a brief `excerpt`, `explanation` of its connection, and optionally `concepts` (a string list). Keep excerpts short and preserve attribution. A reference is background from that source, not a statement by the video's speaker.

`search_knowledge(video_id, query)` retrieves saved chapter notes and external references. Return to source evidence when a follow-up depends on a detail the notes do not establish.

## CLI equivalents

From the installed skill, replace `python -m vidlens` below with `python3 <skill-directory>/scripts/vidlens.py`. In the checkout, use `.venv/bin/python -m vidlens`.

```sh
python -m vidlens inspect 'VIDEO_URL' --intent summary
python -m vidlens inspect 'VIDEO_URL' --intent question --question 'a concrete question'
python -m vidlens study VIDEO_ID
python -m vidlens section VIDEO_ID SECTION_ID --max-tokens 6000
python -m vidlens section VIDEO_ID SECTION_ID --cursor 'RETURNED_CURSOR'
python -m vidlens section VIDEO_ID SECTION_ID --frame-cursor 0
python -m vidlens scan VIDEO_ID
python -m vidlens scan VIDEO_ID --view full
python -m vidlens section VIDEO_ID SECTION_ID --frame-view full --frame-cursor 0
python -m vidlens note VIDEO_ID SECTION_ID --revision REVISION --file note.json
python -m vidlens reference VIDEO_ID SECTION_ID --revision REVISION --file reference.json
python -m vidlens knowledge VIDEO_ID 'concept or question'
python -m vidlens export VIDEO_ID --output lesson.md
```

CLI results contain local image paths. Open those files with an image viewer before verifying a diagram, code or value. Use `--help` on a command for its paging and frame options. The exported notes are an intermediate artifact; edit their organization for the user's intended explanation.

## Retry a technical speech passage

The chapter plan exposes installed backends under `transcription`. Start with its suggested configuration without a vocabulary prompt. On Apple Silicon with the `asr-mlx` extra:

```sh
python -m vidlens transcribe VIDEO_ID --start 60 --end 120 --language zh \
  --backend mlx-whisper
```

This selects `large-v3-turbo`, detects speech before decoding, and restores segment times to the original audio. `--backend faster-whisper` uses the CPU path and defaults to `base`. Backend and model are explicit in the returned provenance. The first model download is separate from cached inference.

If ASR still misreads technical names, inspect nearby screen images and try a few observed terms:

```sh
python -m vidlens transcribe VIDEO_ID --start 60 --end 120 --language zh \
  --backend mlx-whisper --vocabulary 'KV cache' Query Key Value --refresh
```

MCP uses the same fields: `transcribe_video(video_id, start=60, end=120, language="zh", backend="mlx-whisper", vocabulary=["KV cache", "Query", "Key", "Value"], refresh=True)`. Omit `model` to use that backend's default. Compare the new evidence: hints are not an automatic improvement.

The vocabulary is a hint to the recognizer. Check the result against images for symbols and dimensions. Backend, model and vocabulary are part of the cache identity; a successful retry replaces contained ASR cues in the same language and interval, preserves native captions and out-of-range evidence, and updates the study revision. A failed retry leaves previous evidence available. Read the refreshed section and update its note after a successful replacement.

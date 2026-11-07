# From reading to a complete technical lesson

Use this workflow for a learning summary, detailed explanation or Markdown article. The host AI authors the knowledge, prose and diagrams; VidLens supplies original evidence, persistent structure and portable export. Ordinary focused answers do not need an article plan.

## Organize what was learned

After reading all available speech and relevant screens, use `save_knowledge(video_id, revision, units)`. A unit may cite evidence from several chapters. Preserve stable IDs when editing a unit. The default upserts; `replace=True` intentionally replaces the inventory.

```json
{
  "id": "coordinate-decoding",
  "title": "Decode a grid-relative coordinate",
  "roles": ["mechanism", "example"],
  "concepts": ["coordinate", "normalization"],
  "explanation": "Write the actual mechanism and its conditions after reading.",
  "evidence_ids": ["COPY_ORIGINAL_ID_FROM_REFS"],
  "requires": ["grid-assignment"],
  "importance": "core",
  "questions": ["Can the reader recover the pixel position for a different input?"],
  "relations": []
}
```

Roles can include definition, mechanism, derivation, example, code, experiment and limitation. Use multiple roles where useful. `requires` describes learning prerequisites between existing unit IDs; `relations` records specific source-supported connections using `{source,target,relation,evidence_ids}`. References verify provenance, not the truth of AI wording. Put illustrative values in `example` with a clear teaching-example label, without implying the speaker supplied them.

Use `query_knowledge(video_id, query, roles=[...])` to retrieve concepts or a role and its prerequisite closure. With an empty query it browses units. A truncated result may omit an entire dependency group: narrow the query or increase the budget. Whole-video writing still needs a coverage inventory; search hits alone cannot establish it.

## Plan for the reader

`article_plan(video_id, revision, plan)` saves:

```json
{
  "title": "A concrete teaching title",
  "goal": "What the reader should be able to explain or calculate",
  "sections": [{
    "id": "coordinates",
    "title": "From a predicted value to a position",
    "objective": "Recover a position with the correct reference frame",
    "unit_ids": ["grid-assignment", "coordinate-decoding"],
    "visuals": [{"purpose": "Show the grid origin and offset together", "evidence_ids": ["COPY_ID"]}],
    "exercises": [{"question": "Change one coordinate and calculate the result", "unit_ids": ["coordinate-decoding"]}]
  }],
  "omissions": []
}
```

Choose the order that explains the causal chain. A useful pattern is a whole input-to-output pass, then the mechanism's details, a worked calculation, experimental evidence and limitations. It is not a mandatory heading template. Merge repeated exposition while retaining prerequisites, intermediate states and reasons for the design.

For a long article, give a short orientation and navigable sections. Follow one concrete example far enough to connect the abstractions. Explain a formula's inputs and outputs before compacting them into symbols. Use an adjacent diagram with a specific reading instruction. Include a small change-of-input question and its answer when it tests the mechanism. Do not pad length with repeated conclusions or exhaustive model-family history.

Original screenshots preserve evidence. Redrawn diagrams explain relations at readable size: keep objects and colors consistent across steps, label what changes, and mark invented numbers as teaching examples. Use passive SVG or code-generated diagrams for exact geometry and labels. Verify the diagram against source evidence and any calculation before export. Interactivity is useful when a parameter change teaches something; use the existing HTML lesson helper when appropriate.

External extensions should answer a question raised by the video. Follow the user's source constraint: if they request only VidLens/video material, do not browse new facts or present background knowledge as video evidence. Otherwise consult primary sources, save `add_reference`, and attribute the extension near its explanation.

## Audit the draft and export

Start each planned section with a standalone `<!-- section:SECTION_ID -->` line. For each core unit, supply `{unit_id,section_id,excerpt}` in `coverage`; the excerpt must occur in that section's actual prose. Map every planned visual and exercise to content in the same section. Both task indexes start at **0** and refer to that section's `visuals` or `exercises` list in the saved plan.

The draft has this structure; the prose and SVG below illustrate the schema, not a completed lesson:

```json
{
  "markdown": "[Coordinates](#coordinates)\n\n<!-- section:coordinates -->\n## Coordinates\n\nThe grid containing the center owns the object. Decode an offset from that grid's origin.\n\n![Grid origin and offset](asset:coordinates.svg)\n\nChange the horizontal offset and recompute the pixel position.\n\n[Teaching calculation](calculations.py)",
  "coverage": [
    {"unit_id": "grid-assignment", "section_id": "coordinates", "excerpt": "The grid containing the center owns the object."},
    {"unit_id": "coordinate-decoding", "section_id": "coordinates", "excerpt": "Decode an offset from that grid's origin."}
  ],
  "visual_coverage": [
    {"section_id": "coordinates", "visual_index": 0, "asset_name": "coordinates.svg"}
  ],
  "exercise_coverage": [
    {"section_id": "coordinates", "exercise_index": 0, "excerpt": "Change the horizontal offset and recompute the pixel position."}
  ],
  "omissions": [],
  "assets": [
    {"name": "coordinates.svg", "kind": "diagram", "svg": "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"320\" height=\"100\"><text x=\"20\" y=\"50\">Grid origin + offset</text></svg>"}
  ],
  "attachments": [
    {"name": "calculations.py", "text": "# Teaching example: a 100-pixel cell with an offset of 0.25.\nprint(100 * 0.25)\n"}
  ]
}
```

The visual's `asset_name` must exist in `assets` and appear as a real Markdown image in the claimed section; image syntax inside a code sample does not count. Reference-style images are supported, including definitions at the end of the document. The exercise excerpt must occur in its own section. Plans without visual or exercise tasks need no corresponding mappings.

`audit_article` reports missing core units, prerequisites, visual/exercise tasks, stale plans and invalid mappings. Its output includes `uncovered_core`, `uncovered_dependencies`, `uncovered_visuals` and `uncovered_exercises`. An explicitly omitted core unit still blocks readiness. Edit knowledge first and refresh its plan when the scope changes, rather than relabeling a missing mechanism just to pass.

Use `[Coordinates](#coordinates)` for a contents link to a section ID. During export the validated comment marker produces a real `<a id="coordinates"></a>` target in the Markdown and HTML preview; the Markdown also retains its comment marker. Author only the comment markers in the draft. The exporter inserts the anchors, rejects unmatched fragment links and continues to reject arbitrary input HTML.

Structural coverage does not check meaning. Before export, reread the draft against its evidence and try the worked examples. Check numeric source values separately from arithmetic. A second reader should solve a changed-input problem using only the article, then compare with a reference prepared independently; record failures without silently changing the reference. The user remains the judge of clarity and usefulness.

CLI workflow from the project environment:

```sh
python -m vidlens save-units VIDEO_ID --revision REVISION --file units.json
python -m vidlens article-plan VIDEO_ID --revision REVISION --file plan.json
python -m vidlens audit-article VIDEO_ID --revision REVISION --file draft.json
python -m vidlens article VIDEO_ID --revision REVISION --file draft.json --output lesson-directory
```

Use `![descriptive alt](asset:NAME.svg)` in the Markdown. Each asset is either:

- `{name:"diagram.svg",kind:"diagram",svg,evidence_ids:[...]}`: passive SVG text created by the author; `evidence_ids` is optional. Include source-supported or explicitly illustrative content. Scripts, remote resources and embedded HTML are rejected.
- `{name:"source.png",kind:"source",evidence_id:"frame:TIME",box:[x,y,width,height]}`: a cached frame from this video; `box` is optional and uses normalized coordinates. Read and inspect it before writing about it. The exporter resolves the original frame itself and writes a PNG without private image metadata; do not substitute an unrelated cache path.

Optional `attachments` carry reproducible teaching calculations or small data files: each entry is `{name:"calculations.py",text:"..."}`. Use a plain filename with `.py`, `.json`, `.txt` or `.csv`, and link it from the prose as `[Teaching calculation](calculations.py)` or `./calculations.py`. Up to ten attachments are accepted, each at most 100,000 UTF-8 bytes. Duplicate names, reserved output names and directory paths are rejected. The exporter copies text and **never executes it**; run and inspect the relevant calculations separately before calling them verified. Attachments should contain portable examples rather than machine-specific paths.

The exporter creates `README.md`, relative `assets/`, a local `preview.html`, optional attachments in the bundle root, and `manifest.json` with file hashes and structural-audit results. Existing output directories are preserved. Open the preview and check figures, contents links, text wrapping and narrow screens. The preview is a reading aid; standard Markdown math may need a math-aware reader. Keep rendering limitations next to the artifact if any remain.

Set `VIDLENS_RUN_ID` for a reproducible reading run and use `vidlens metrics --run-id NAME`. Timings are local operations and can overlap or nest; token counts are text estimates, not model bills. Distinguish a cached rewrite from first acquisition/ASR.

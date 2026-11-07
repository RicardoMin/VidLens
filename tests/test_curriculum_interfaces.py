"""Exercise reading, reusable knowledge and article review through the real MCP API."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from vidlens.evidence_format import expand_evidence
from vidlens.mcp_server import create_server
from vidlens.models import Segment, Video
from vidlens.service import VideoService


@pytest.mark.asyncio
async def test_compact_mcp_evidence_can_build_persistent_knowledge_and_audited_article(tmp_path):
    service = VideoService(tmp_path / "library")
    video = Video("lesson", "https://www.bilibili.com/video/BV1KT2yBJESn/?p=2",
                  "bilibili", "Detection lesson", 120)
    cues = [Segment(0, 20, "The object center assigns a responsible grid cell."),
            Segment(20, 40, "Multiply conditional class probability by box confidence.")]
    service.store.save(video, cues, [])
    server = create_server(service=service)
    async with create_connected_server_and_client_session(server) as client:
        async def call(name, **args):
            result = await client.call_tool(name, args)
            assert not result.isError, result
            assert result.structuredContent is None
            return json.loads(result.content[0].text)

        status = await call("read_video", url=video.url, intent="summary")
        section, revision = status["sections"][0], status["revision"]
        page = await call("read_section", video_id=video.id, section_id=section["id"],
                          include_frames=False)
        assert page["representation"] == "compact"
        assert [item["text"] for item in expand_evidence(page)] == [cue.text for cue in cues]
        assert page["progress"]["text_status"] == "delivered"
        refs = [item["id"] for item in expand_evidence(page)]
        note = await call("save_note", video_id=video.id, section_id=section["id"],
                          revision=revision, note={
                              "title": "Grid and scoring", "summary": "Assign a grid, then score a box.",
                              "claims": [{"text": cues[0].text, "evidence_ids": [refs[0]]}],
                          })
        assert note["note"]["claims"][0]["evidence_ids"] == [refs[0]]
        units = [
            {"id": "grid", "title": "Grid assignment", "explanation": cues[0].text,
             "roles": ["mechanism"], "concepts": ["grid"], "evidence_ids": [refs[0]],
             "requires": []},
            {"id": "score", "title": "Confidence scoring", "explanation": cues[1].text,
             "roles": ["derivation"], "concepts": ["confidence"], "evidence_ids": [refs[1]],
             "requires": ["grid"]},
        ]
        saved = await call("save_knowledge", video_id=video.id, revision=revision, units=units)
        assert saved["saved_unit_ids"] == ["grid", "score"]
        queried = await call("query_knowledge", video_id=video.id, query="confidence",
                             roles=["derivation"])
        assert queried["matched_unit_ids"] == ["score"]
        assert [unit["id"] for unit in queried["units"]] == ["grid", "score"]
        assert [unit["evidence_ids"] for unit in queried["units"]] == [[refs[0]], [refs[1]]]
        assert queried["source_url"] == video.url
        plan = {"title": "Decode a detection", "goal": "Explain assignment and compute a score.",
                "sections": [{"id": key, "title": key, "objective": unit["explanation"],
                              "unit_ids": [key]} for key, unit in zip(("grid", "score"), units)]}
        await call("article_plan", video_id=video.id, revision=revision, plan=plan)
        draft = {"markdown": "\n".join(
            f"<!-- section:{key} -->\n## {key}\n{unit['explanation']}"
            for key, unit in zip(("grid", "score"), units)),
            "coverage": [{"unit_id": key, "section_id": key, "excerpt": unit["explanation"]}
                         for key, unit in zip(("grid", "score"), units)]}
        audit = await call("audit_article", video_id=video.id, revision=revision, draft=draft)
        assert audit["ready_for_review"] and audit["covered_core_count"] == 2
        assert audit["semantic_validation"] == "not_performed"
        # A real missing prerequisite in the delivered prose must fail through MCP,
        # even when the submitted coverage manifest still claims both units.
        draft["markdown"] = draft["markdown"].replace(cues[0].text, "An introduction only.")
        failed = await call("audit_article", video_id=video.id, revision=revision, draft=draft)
        assert not failed["ready_for_review"]
        assert failed["uncovered_core"] == ["grid"]
        assert failed["uncovered_dependencies"] == ["grid"]

    restarted = create_server(service=VideoService(tmp_path / "library"))
    async with create_connected_server_and_client_session(restarted) as client:
        result = await client.call_tool("query_knowledge", {"video_id": video.id})
        assert not result.isError
        persisted = json.loads(result.content[0].text)
        assert {ref for unit in persisted["units"] for ref in unit["evidence_ids"]} == set(refs)

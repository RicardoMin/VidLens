import importlib.util
import tomllib
from pathlib import Path

import pytest


def installer():
    path = Path(__file__).resolve().parents[1] / "scripts/install-astra.py"
    spec = importlib.util.spec_from_file_location("vidlens_installer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_project_mcp_install_preserves_other_settings_and_is_idempotent(tmp_path):
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").touch()
    (tmp_path / ".codex").mkdir()
    config = tmp_path / ".codex/config.toml"
    config.write_text('model = "keep-existing"\n[mcp_servers.other]\ncommand = "existing-tool"\n')
    for _ in range(2):
        installer().prepare(tmp_path, tmp_path / "integration", tmp_path / "cache", project_mcp=True)
    observed = tomllib.loads(config.read_text())
    assert observed["model"] == "keep-existing"
    assert observed["mcp_servers"]["other"]["command"] == "existing-tool"
    assert observed["mcp_servers"]["vidlens"]["command"] == str(tmp_path / ".venv/bin/python")


def test_conflicting_mcp_configuration_is_not_overwritten(tmp_path):
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").touch()
    (tmp_path / ".codex").mkdir()
    config = tmp_path / ".codex/config.toml"
    original = '[mcp_servers.vidlens]\ncommand = "user-customized-command"\n'
    config.write_text(original)
    with pytest.raises(ValueError, match="differ"):
        installer().prepare(tmp_path, tmp_path / "integration", tmp_path / "cache", project_mcp=True)
    assert config.read_text() == original


def test_reactivation_handles_unicode_path_and_preserves_advanced_tool_preference(tmp_path):
    project = tmp_path / "视频🔎"
    (project / ".venv/bin").mkdir(parents=True)
    (project / ".venv/bin/python").touch()
    prepare = installer().prepare
    options = (project, project / "integration", project / "cache")
    result = prepare(*options, project_mcp=True)
    config = Path(result["project_mcp"])
    customized = config.read_text() + 'VIDLENS_ADVANCED_TOOLS = "1"\n'
    config.write_text(customized)
    result = prepare(*options, project_mcp=True)
    settings = tomllib.loads(config.read_text())["mcp_servers"]["vidlens"]
    assert settings["command"] == str(project / ".venv/bin/python")
    assert config.read_text() == customized
    assert "Reconnect" in result["next_step"]


def test_conflicting_skill_does_not_partially_activate_project_mcp(tmp_path):
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").touch()
    for relative, text in (("skills/read-video", "upstream"), (".agents/skills/read-video", "custom")):
        skill = tmp_path / relative / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(text)
    with pytest.raises(ValueError, match="Existing skill differs"):
        installer().prepare(tmp_path, tmp_path / "integration", tmp_path / "cache",
                            project_skill=True, project_mcp=True)
    assert not (tmp_path / ".codex/config.toml").exists()
    assert (tmp_path / ".agents/skills/read-video/SKILL.md").read_text() == "custom"

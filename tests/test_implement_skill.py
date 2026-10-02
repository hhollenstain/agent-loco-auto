"""Tests for the bundled implement skill."""

from pathlib import Path

import yaml

from agent_loco.runtime.project import write_default_project_files
from agent_loco.runtime.skills import (
    bundled_skills_root,
    compose_system_prompt,
    list_skills,
)


def test_init_enables_implement_skill(tmp_path: Path) -> None:
    """write_default_project_files enables implement skill by default."""
    write_default_project_files(tmp_path)
    config_path = tmp_path / ".loco" / "config.yaml"
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    enabled = config.get("skills", {}).get("enabled", [])
    assert "implement" in enabled
    assert "research" in enabled
    assert "debug" in enabled
    assert "explore" in enabled
    assert "review" in enabled
    assert "security" in enabled
    assert "tdd" in enabled
    assert "ui" in enabled


def test_implement_skill_is_bundled_and_injected(tmp_path: Path) -> None:
    """implement skill is bundled and appears in system prompt when enabled."""
    skill_path = bundled_skills_root() / "implement" / "SKILL.md"
    assert skill_path.is_file()
    skills = {item.name: item for item in list_skills(tmp_path)}
    assert "implement" in skills
    assert skills["implement"].origin == "bundled"
    assert "Implementation Process" in skills["implement"].body
    assert "web_search" in skills["implement"].body
    assert "fetch_url" in skills["implement"].body

    write_default_project_files(tmp_path)

    # Enable implement skill
    from agent_loco.runtime.skills import save_enabled_skills

    save_enabled_skills(tmp_path, ["implement"])

    # Check it appears in system prompt
    prompt = compose_system_prompt(tmp_path)
    assert "### implement" in prompt
    # Check a unique phrase from SKILL.md appears
    assert "Implementation Process" in prompt


def test_init_does_not_overwrite_existing_enabled_list(tmp_path: Path) -> None:
    """Init does not overwrite config if it already exists with enabled list."""
    # Pre-create config with only tdd enabled
    loco_dir = tmp_path / ".loco"
    loco_dir.mkdir()
    config_path = loco_dir / "config.yaml"
    config_path.write_text(
        "name: test\ntest_command: pytest\nskills:\n  enabled:\n    - tdd\n",
        encoding="utf-8",
    )

    # Run init again
    write_default_project_files(tmp_path)

    # Config should remain unchanged
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    enabled = config.get("skills", {}).get("enabled", [])
    assert enabled == ["tdd"]
    assert "implement" not in enabled

from __future__ import annotations

from pathlib import Path

from agent_loco.progress import bind_progress, current_events, reset_progress
from agent_loco.runtime.project import infer_lint_command, load_project, write_default_project_files
from agent_loco.sandbox import Workspace
from agent_loco.tools import build_tools
from agent_loco.tools.lint import run_project_lint


def test_infer_ruff_for_src_and_tests(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    assert infer_lint_command(tmp_path) == "ruff format src tests && ruff check src tests"


def test_infer_ruff_dot_without_src_layout(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert infer_lint_command(tmp_path) == "ruff format . && ruff check ."


def test_empty_lint_command_disables_inference(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    loco = tmp_path / ".loco"
    loco.mkdir()
    (loco / "config.yaml").write_text("name: demo\nlint_command: ''\n", encoding="utf-8")
    assert load_project(tmp_path).lint_command is None


def test_init_writes_inferred_lint_command(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    write_default_project_files(tmp_path)
    config = (tmp_path / ".loco" / "config.yaml").read_text(encoding="utf-8")
    assert "lint_command: ruff format src tests && ruff check src tests" in config


def test_run_lint_applies_ruff_format(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    messy = src / "app.py"
    messy.write_text("x=1+2\n", encoding="utf-8")
    result = run_project_lint(Workspace(tmp_path), infer_lint_command(tmp_path), 30)
    assert result.ok
    assert messy.read_text(encoding="utf-8") == "x = 1 + 2\n"


def test_run_lint_is_a_wired_tool(tmp_path: Path) -> None:
    tools = build_tools(
        Workspace(tmp_path),
        test_command=None,
        lint_command="ruff check .",
        command_timeout_seconds=10,
        git_author_name=None,
        git_author_email=None,
    )
    assert "run_lint" in {tool.name for tool in tools}


def test_run_project_lint_records_failure(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    command = "python3 -c \"raise SystemExit('E501 line too long')\""
    token = bind_progress()
    try:
        result = run_project_lint(workspace, command, 10, phase="after")
        events = current_events()
    finally:
        reset_progress(token)
    assert result.ok is False
    assert "E501" in result.output
    assert events[0]["kind"] == "lint"
    assert events[0]["ok"] is False
    assert events[0]["phase"] == "after"

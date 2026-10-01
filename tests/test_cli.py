from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from agent_loco import __version__
from agent_loco.cli import app

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_TEST_ENV_VARS = {"LOCO_MODEL_NAME": "test-model", "LOCO_MODEL_BASE_URL": "http://test:9000"}


@pytest.fixture
def monkeypatch_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _TEST_ENV_VARS.items():
        monkeypatch.setenv(k, v)


def _help_text(result) -> str:
    """Flag names stay intact even when rich wraps help to an 80-column CI terminal."""
    return _ANSI_RE.sub("", result.stdout).replace("\n", "")


def test_help_text_rejoins_wrapped_flags() -> None:
    class _Result:
        stdout = "\x1b[1m--max-\nconcurrent\x1b[0m  --web-\nui  --model"

    assert "--max-concurrent" in _help_text(_Result())
    assert "--web-ui" in _help_text(_Result())
    assert "--model" in _help_text(_Result())


def test_version(monkeypatch_env_vars: None) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in _help_text(result)


def test_run_help_includes_web_ui(monkeypatch_env_vars: None) -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--web-ui" in _help_text(result)


def test_ui_help(monkeypatch_env_vars: None) -> None:
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    text = _help_text(result)
    assert "--max-concurrent" in text
    assert "--port" in text
    assert "--model" in text
    assert "-m" in text
    assert "--base-url" in text


def test_clone_help_and_local_repo(tmp_path, monkeypatch, monkeypatch_env_vars: None) -> None:
    from tests.support import init_git_repo

    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("demo\n", encoding="utf-8")
    init_git_repo(source)
    parent = tmp_path / "parent"
    parent.mkdir()
    monkeypatch.chdir(parent)
    result = runner.invoke(app, ["clone", str(source), "checkout"])
    assert result.exit_code == 0, result.stdout
    assert "cloned" in result.stdout
    dest = parent / "checkout"
    assert (dest / "README.md").read_text(encoding="utf-8") == "demo\n"
    assert (dest / ".loco" / "config.yaml").exists()
    missing = runner.invoke(app, ["clone", str(source), "checkout"])
    assert missing.exit_code != 0


def test_run_help_includes_model_flag(monkeypatch_env_vars: None) -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    text = _help_text(result)
    assert "--model" in text
    assert "-m" in text
    assert "--base-url" in text

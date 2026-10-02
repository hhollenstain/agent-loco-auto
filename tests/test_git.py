from __future__ import annotations

import subprocess
from pathlib import Path

from tests.support import init_git_repo

from agent_loco.sandbox import Workspace
from agent_loco.tools.git import (
    CO_AUTHORED_BY,
    agent_commit,
    commit_changes,
    commits_ahead_of_base,
    create_pull_request,
    current_branch,
    current_sha,
    ensure_pr_branch,
    existing_pull_request,
    extract_pr_url,
    has_changes,
    is_runtime_artifact,
    is_tracked,
    parse_github_remote,
    pull_request_state,
    push_changes,
    resume_workspace,
    run_git,
    update_pull_request,
    upstream_state,
    with_loco_coauthor,
)


def test_extract_pr_url_from_gh_output() -> None:
    assert extract_pr_url(None) is None
    assert extract_pr_url("no link here") is None
    assert (
        extract_pr_url("Creating pull request\nhttps://github.com/acme/repo/pull/12\n")
        == "https://github.com/acme/repo/pull/12"
    )
    assert extract_pr_url("https://example.test/pull/1") == "https://example.test/pull/1"
    assert (
        extract_pr_url("opened https://gitlab.com/acme/app/-/merge_requests/7.")
        == "https://gitlab.com/acme/app/-/merge_requests/7"
    )


def test_parse_github_remote_from_common_urls() -> None:
    assert parse_github_remote("git@github.com:acme/repo.git") == ("acme", "repo")
    assert parse_github_remote("https://github.com/acme/repo.git") == ("acme", "repo")
    assert parse_github_remote("https://github.com/acme/repo") == ("acme", "repo")
    assert parse_github_remote("ssh://git@github.com/acme/repo.git") == ("acme", "repo")
    assert parse_github_remote("https://gitlab.com/acme/repo.git") is None


def test_commit_changes_never_adds_loco_files(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".loco/\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    run_git(workspace, ["checkout", "-b", "loco/feature"])
    (tmp_path / "app.py").write_text("print('changed')\n", encoding="utf-8")
    shot = tmp_path / ".loco" / "ui-screenshots" / "ui-review_goal.png"
    shot.parent.mkdir(parents=True)
    shot.write_bytes(b"png")
    config = tmp_path / ".loco" / "config.yaml"
    config.write_text("name: demo\n", encoding="utf-8")
    assert not is_tracked(workspace, shot)
    result = commit_changes(workspace, "update app", extra_paths=[shot, config])
    assert result.ok
    assert not is_tracked(workspace, shot)
    assert not is_tracked(workspace, config)
    listed = run_git(workspace, ["ls-files", "--", ".loco"])
    assert listed.stdout.strip() == ""


def test_with_loco_coauthor_adds_github_trailer() -> None:
    assert with_loco_coauthor("") == ""
    assert with_loco_coauthor("  fix the thing  ") == f"fix the thing\n\n{CO_AUTHORED_BY}"
    already = f"fix the thing\n\n{CO_AUTHORED_BY}"
    assert with_loco_coauthor(already) == already


def test_commits_ahead_of_base_counts_feature_commits(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("one\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    assert commits_ahead_of_base(workspace) == 0
    run_git(workspace, ["checkout", "-b", "loco/feature"])
    (tmp_path / "app.py").write_text("two\n", encoding="utf-8")
    committed = commit_changes(workspace, "two")
    assert committed.ok
    assert commits_ahead_of_base(workspace) == 1


def test_existing_pull_request_reads_gh_json(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "readme.txt").write_text("hello\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    real_run = subprocess.run

    def fake_run(args, **kwargs):
        if args and args[0] == "gh" and "view" in args:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout='{"url":"https://github.com/acme/repo/pull/4"}\n',
                stderr="",
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr("agent_loco.tools.git.subprocess.run", fake_run)
    assert existing_pull_request(workspace) == "https://github.com/acme/repo/pull/4"


def test_pull_request_state_reads_gh_json(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        assert args[:4] == ["gh", "pr", "view", "https://github.com/acme/repo/pull/4"]
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"state":"MERGED"}\n',
            stderr="",
        )

    monkeypatch.setattr("agent_loco.tools.git.subprocess.run", fake_run)
    assert pull_request_state("https://github.com/acme/repo/pull/4") == "merged"


def test_update_pull_request_edits_the_open_pr(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "readme.txt").write_text("hello\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    seen: dict[str, list[str]] = {}

    def fake_run(args, **kwargs):
        seen["args"] = list(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="https://github.com/acme/repo/pull/4\n",
            stderr="",
        )

    monkeypatch.setattr("agent_loco.tools.git.subprocess.run", fake_run)
    result = update_pull_request(workspace, "Keep the branch", "Added more context")
    assert result.ok
    assert seen["args"][:3] == ["gh", "pr", "edit"]
    assert "Keep the branch" in seen["args"]
    assert any("Added more context" in part for part in seen["args"])


def test_existing_pull_request_missing_gh_is_none(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "readme.txt").write_text("hello\n", encoding="utf-8")
    init_git_repo(tmp_path)

    def fake_run(args, **kwargs):
        if args and args[0] == "gh":
            raise FileNotFoundError("gh")
        raise AssertionError(args)

    monkeypatch.setattr("agent_loco.tools.git.subprocess.run", fake_run)
    assert existing_pull_request(Workspace(tmp_path)) is None


def test_create_pull_request_treats_already_exists_as_success(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "readme.txt").write_text("hello\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    run_git(workspace, ["checkout", "-b", "loco/feature"])
    real_run = subprocess.run

    def fake_run(args, **kwargs):
        if args and args[0] == "gh":
            return subprocess.CompletedProcess(
                args,
                1,
                stdout="",
                stderr=(
                    'a pull request for branch "loco/feature" already exists:\n'
                    "https://github.com/acme/repo/pull/9\n"
                ),
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr("agent_loco.tools.git.subprocess.run", fake_run)
    result = create_pull_request(workspace, "Add feature", "details", base="main")
    assert result.ok
    assert extract_pr_url(result.output) == "https://github.com/acme/repo/pull/9"


def test_create_pull_request_does_not_pass_unknown_coauthor_flag(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "readme.txt").write_text("hello\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    run_git(workspace, ["checkout", "-b", "loco/feature"])
    captured: dict[str, list[str]] = {}
    real_run = subprocess.run

    def fake_run(args, **kwargs):
        if args and args[0] == "gh":
            captured["args"] = list(args)
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="https://github.com/acme/repo/pull/1\n",
                stderr="",
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr("agent_loco.tools.git.subprocess.run", fake_run)
    result = create_pull_request(workspace, "Add feature", "details", base="main")
    assert result.ok
    args = captured["args"]
    assert args[:3] == ["gh", "pr", "create"]
    assert "--add-co-author" not in args
    body = args[args.index("--body") + 1]
    assert CO_AUTHORED_BY in body
    assert "--base" in args
    assert args[args.index("--base") + 1] == "main"


def test_run_log_paths_are_runtime_artifacts() -> None:
    assert is_runtime_artifact(".loco/runs/cycle.json")
    assert is_runtime_artifact(".loco/runs")
    assert is_runtime_artifact(".loco/servers.json")
    assert is_runtime_artifact(".loco/workspaces.json")
    assert is_runtime_artifact(".loco/config.yaml")
    assert is_runtime_artifact(".loco/ui-screenshots/ui-review.png")
    assert is_runtime_artifact(".loco/.gitignore")
    assert is_runtime_artifact("history.json")
    assert not is_runtime_artifact("src/agent_loco/cli.py")
    assert not is_runtime_artifact("loco/runs/cycle.json")


def test_tool_caches_are_runtime_artifacts() -> None:
    assert is_runtime_artifact("__pycache__/app.cpython-312.pyc")
    assert is_runtime_artifact("src/pkg/__pycache__/mod.cpython-312.pyc")
    assert is_runtime_artifact(".pytest_cache/v/cache/nodeids")
    assert is_runtime_artifact(".ruff_cache/CACHEDIR.TAG")
    assert is_runtime_artifact("build/mod.pyc")
    assert not is_runtime_artifact("src/pkg/mod.py")
    assert not is_runtime_artifact("docs/pycache_notes.md")


def test_bytecode_left_by_the_test_command_is_not_project_work(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "app.cpython-312.pyc").write_bytes(b"\x00")
    assert not has_changes(workspace)
    (tmp_path / "app.py").write_text("print('changed')\n", encoding="utf-8")
    assert has_changes(workspace)
    result = commit_changes(workspace, "change app")
    assert result.ok
    tracked = run_git(workspace, ["ls-files"]).stdout.split()
    assert "app.py" in tracked
    assert not any(path.startswith("__pycache__/") for path in tracked)


def test_commit_happy_path(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("hello\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    (tmp_path / "readme.txt").write_text("hello world\n", encoding="utf-8")
    assert has_changes(workspace)
    result = commit_changes(workspace, "update readme")
    assert result.ok
    assert "committed" in result.output
    assert not has_changes(workspace)
    log = run_git(workspace, ["log", "-1", "--format=%B"]).stdout
    assert "update readme" in log
    assert CO_AUTHORED_BY in log


def test_commit_rejects_env_file(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    result = commit_changes(workspace, "add env")
    assert not result.ok
    assert "secrets" in result.output


def test_commit_skips_run_logs(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    runs = tmp_path / ".loco" / "runs"
    runs.mkdir(parents=True)
    (runs / "cycle.json").write_text('{"status": "success"}\n', encoding="utf-8")
    (tmp_path / "app.py").write_text("print('changed')\n", encoding="utf-8")
    assert has_changes(workspace)
    result = commit_changes(workspace, "update app")
    assert result.ok
    tracked = run_git(workspace, ["ls-files", ".loco/runs"])
    assert tracked.stdout.strip() == ""
    assert (runs / "cycle.json").exists()


def test_commit_skips_history_json(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    (tmp_path / "history.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('changed')\n", encoding="utf-8")
    assert has_changes(workspace)
    result = commit_changes(workspace, "update app")
    assert result.ok
    tracked = run_git(workspace, ["ls-files", "history.json"])
    assert tracked.stdout.strip() == ""
    assert (tmp_path / "history.json").exists()


def test_run_logs_alone_are_not_changes(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    runs = tmp_path / ".loco" / "runs"
    runs.mkdir(parents=True)
    (runs / "cycle.json").write_text("{}\n", encoding="utf-8")
    assert not has_changes(workspace)
    result = commit_changes(workspace, "should not commit logs")
    assert not result.ok


def test_agent_commit_refuses_protected_branch(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    (tmp_path / "app.py").write_text("print('changed')\n", encoding="utf-8")
    result = agent_commit(workspace, "update app")
    assert result.ok is False
    assert "refusing to commit" in result.output
    assert has_changes(workspace)
    assert current_branch(workspace) in {"main", "master"}


def test_agent_commit_allows_feature_branch(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    run_git(workspace, ["checkout", "-b", "loco/feature"])
    (tmp_path / "app.py").write_text("print('changed')\n", encoding="utf-8")
    result = agent_commit(workspace, "update app")
    assert result.ok
    assert "committed" in result.output
    assert not has_changes(workspace)


def test_cycle_commit_helper_still_works_on_protected_branch(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    (tmp_path / "app.py").write_text("print('changed')\n", encoding="utf-8")
    result = commit_changes(workspace, "update app")
    assert result.ok
    assert current_branch(workspace) in {"main", "master"}


def test_push_refuses_protected_branches(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    for name in ("main", "master"):
        result = push_changes(workspace, "origin", name)
        assert result.ok is False
        assert "refusing" in result.output


def test_ensure_pr_branch_moves_commits_off_main(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("one\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    protected = current_branch(workspace)
    sha_before = current_sha(workspace)
    (tmp_path / "app.py").write_text("two\n", encoding="utf-8")
    committed = commit_changes(workspace, "change app")
    assert committed.ok
    after = current_sha(workspace)
    result = ensure_pr_branch(workspace, "Improve the adder", sha_before)
    assert result.ok
    assert result.output.startswith("loco/")
    assert current_branch(workspace) == result.output
    assert current_sha(workspace) == after
    assert run_git(workspace, ["rev-parse", protected or "HEAD"]).stdout.strip() == sha_before


def test_upstream_state_without_remote_is_not_pushed(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    state = upstream_state(Workspace(tmp_path))
    assert state.pushed is False
    assert state.tracking is None
    assert "not pushed" in state.detail


def test_upstream_state_matches_tracking_ref(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    branch = current_branch(workspace)
    sha = current_sha(workspace)
    run_git(workspace, ["update-ref", f"refs/remotes/origin/{branch}", sha or ""])
    run_git(workspace, ["branch", f"--set-upstream-to=origin/{branch}"])
    state = upstream_state(workspace)
    assert state.pushed is True
    assert state.ahead == 0
    assert "pushed" in state.detail


def test_upstream_state_detects_unpushed_local_commits(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    branch = current_branch(workspace)
    sha = current_sha(workspace)
    run_git(workspace, ["update-ref", f"refs/remotes/origin/{branch}", sha or ""])
    run_git(workspace, ["branch", f"--set-upstream-to=origin/{branch}"])
    (tmp_path / "app.py").write_text("print('next')\n", encoding="utf-8")
    commit_changes(workspace, "local only")
    state = upstream_state(workspace)
    assert state.pushed is False
    assert state.ahead == 1
    assert "not pushed" in state.detail


def test_resume_workspace_checks_out_feature_branch(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    start = current_branch(workspace)
    run_git(workspace, ["checkout", "-b", "loco/feature"])
    (tmp_path / "app.py").write_text("print('next')\n", encoding="utf-8")
    run_git(workspace, ["add", "app.py"])
    run_git(workspace, ["commit", "-m", "wip"])
    run_git(workspace, ["checkout", start or "master"])
    assert current_branch(workspace) != "loco/feature"
    result = resume_workspace(workspace, branch="loco/feature")
    assert result.ok
    assert current_branch(workspace) == "loco/feature"

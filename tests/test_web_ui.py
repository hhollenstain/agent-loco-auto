from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_loco.config import Settings
from agent_loco.llm.client import model_ids_from_payload, normalize_model_base_url
from agent_loco.runtime.improve import CycleResult
from agent_loco.runtime.servers import remember_server
from agent_loco.runtime.tasks import Task, TaskManager
from agent_loco.web_ui import create_app


def _ok_result(goal: str | None = "do the thing") -> CycleResult:
    return CycleResult(
        status="success",
        goal=goal,
        summary="done",
        tests_passed=True,
        committed=False,
        published=False,
        commit_sha=None,
        reason="completed",
    )


def test_submit_rejects_missing_workspace(settings: Settings, tmp_path: Path) -> None:
    manager = TaskManager(settings, runner=lambda task: _ok_result())
    try:
        try:
            manager.submit(tmp_path / "missing", "goal")
        except ValueError as exc:
            assert "not a directory" in str(exc)
        else:
            raise AssertionError("expected ValueError")
    finally:
        manager.shutdown(wait=False)


def test_queue_runs_one_at_a_time(settings: Settings, tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    running_seen = []

    def runner(task: Task) -> CycleResult:
        running_seen.append(task.id)
        started.set()
        assert release.wait(timeout=2)
        return _ok_result(task.goal)

    manager = TaskManager(settings, max_concurrent=1, runner=runner)
    try:
        first = manager.submit(tmp_path, "one")
        second = manager.submit(tmp_path, "two")
        assert started.wait(timeout=2)
        assert manager.get(first.id).status == "running"
        assert manager.get(second.id).status == "queued"
        release.set()
        manager.shutdown(wait=True, cancel_futures=False)
        assert manager.get(first.id).status == "success"
        assert manager.get(second.id).status == "success"
        assert running_seen == [first.id, second.id]
    finally:
        release.set()
        manager.shutdown(wait=False)


def test_live_task_logs_follow_the_running_task(settings: Settings, tmp_path: Path) -> None:
    from agent_loco.web_ui import _live_task_logs

    gate = threading.Event()

    def runner(task: Task) -> CycleResult:
        task.logs.append("model hello from the server")
        gate.wait(timeout=2)
        return _ok_result(task.goal)

    manager = TaskManager(settings, runner=runner)
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        queued = client.post(
            "/api/tasks",
            json={"workspace": str(tmp_path), "goal": "watch the console"},
        )
        assert queued.status_code == 201
        deadline = time.time() + 2
        lines: list[str] = []
        while time.time() < deadline:
            _task_id, lines = _live_task_logs(manager)
            if any("hello from the server" in line for line in lines):
                break
            time.sleep(0.05)
        assert any("hello from the server" in line for line in lines)
        assert any(getattr(route, "path", "") == "/api/events/stream" for route in app.routes)
    finally:
        gate.set()
        manager.shutdown(wait=False)


def test_serve_returns_after_keyboard_interrupt(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_loco.web_ui import serve

    captured: dict[str, int | None] = {}

    class FakeServer:
        def __init__(self, config: object) -> None:
            captured["timeout"] = getattr(config, "timeout_graceful_shutdown", None)

        def run(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr("agent_loco.web_ui.uvicorn.Server", FakeServer)
    serve(tmp_path, settings, host="127.0.0.1", port=8765)
    assert captured["timeout"] == 1


def test_console_event_payload_splits_agent_and_model() -> None:
    from agent_loco.web_ui import _console_event_payload

    payload = _console_event_payload(
        {
            "kind": "llm",
            "purpose": "agent",
            "ok": True,
            "at": "2026-09-29T01:00:00Z",
            "messages": [
                {"role": "system", "content": "You are loco."},
                {"role": "user", "content": "fix the live console colors"},
                {"role": "assistant", "content": "I will look at index.html"},
            ],
            "response": "Colored agent and model lines in the live console.",
        }
    )
    assert payload["agent"] == "fix the live console colors"
    assert payload["model"] == "Colored agent and model lines in the live console."
    assert payload["kind"] == "llm"


def test_web_ui_queues_and_lists_tasks(settings: Settings, tmp_path: Path) -> None:
    gate = threading.Event()

    def runner(task: Task) -> CycleResult:
        gate.wait(timeout=2)
        return _ok_result(task.goal)

    manager = TaskManager(settings, runner=runner)
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        home = client.get("/")
        assert home.status_code == 200
        assert home.content.strip().startswith(b"<!DOCTYPE html>")
        assert b"</html>" in home.content
        assert b"loco" in home.content
        assert b'id="app-logo"' in home.content
        assert b"<title>LoCO</title>" in home.content
        assert b"<title>" in home.content
        assert b"<header>" not in home.content
        assert b'id="toggle-sidebar"' in home.content
        assert b'id="toasts"' in home.content
        assert b'id="notify-toggle"' in home.content
        assert b'id="notify-history"' in home.content
        assert b"notice-success" in home.content
        assert b"notice-error" in home.content
        assert b"notice-warning" in home.content
        assert b"notice-skipped" in home.content
        assert b"function notify(" in home.content
        assert b"class=\"pr-link\"" in home.content
        assert b"function prLinkHtml(" in home.content
        assert b"function rerunFailedTask(" in home.content
        assert b"data-rerun-id" in home.content
        assert b'id="rerun-goal-picker"' in home.content
        assert b'id="editor-scheme"' in home.content
        assert b'<option value="cursor">Cursor</option>' in home.content
        assert b"file-open-link" in home.content
        assert b"function editorFileUrl(" in home.content
        assert b'id="open-live-console"' in home.content
        assert b'id="live-console"' in home.content
        assert b'id="close-live-console"' in home.content
        assert b"/api/events/stream" in home.content
        assert b"function appendConsoleEntry(" in home.content
        assert b"log-chat log-agent" in home.content
        assert b"log-chat log-model" in home.content
        assert b"log-speaker" in home.content
        assert b"data.agent" in home.content
        assert b"data.model" in home.content
        assert b"data-history-update" in home.content
        assert b"card-actions" in home.content
        assert b"pr-state" in home.content
        assert b"/api/pulls/status" in home.content
        assert b"/rerun" in home.content
        assert b"GITHUB_MARK" in home.content
        assert b'event.kind === "pr"' in home.content
        assert b'event.kind === "ui"' in home.content
        assert b"function renderFileChanges(" in home.content
        assert b"function fileReviewDiff(" in home.content
        assert b'data-task-pane="changes"' in home.content
        assert b'data-task-pane="screenshots"' in home.content
        assert b'<link rel="stylesheet" href="/static/themes.css">' in home.content
        assert b'<link rel="stylesheet" href="/static/layout.css">' in home.content
        assert b"<style>" in home.content
        assert home.content.index(b"<style>") < home.content.index(b"* { box-sizing")
        assert b'data-main-pane="current"' in home.content
        assert b'data-main-pane="history"' in home.content
        assert b'id="current-run"' in home.content
        assert b'id="running-chicken"' in home.content
        assert b"function runningChickenHtml(" in home.content
        assert b"function syncRunningChicken(" in home.content
        assert b'id="task-tokens"' in home.content
        assert b"data-task-tokens" in home.content
        assert b"function syncTaskTokens(" in home.content
        assert b"function tokenUsageFromItem(" in home.content
        assert b"function parseTokenValue(" in home.content
        assert b"context window" in home.content
        assert b'id="past-runs"' in home.content
        assert b'id="history-detail"' in home.content
        assert b'id="history-picker"' in home.content
        assert b'id="open-history-picker"' in home.content
        assert b"function setHistoryPickerOpen(" in home.content
        assert b"function ensureLatestHistorySelection(" in home.content
        assert home.content.index(b'id="history-detail"') < home.content.index(b'id="history-list"')
        assert home.content.index(b'id="pagination"') < home.content.index(b'id="history-list"')
        assert b".task-pane {\n      display: none;" not in home.content
        assert b'id="task-pagination"' in home.content
        assert b"taskPageSize" in home.content
        assert b"function taskPaneFromTab(" in home.content
        assert b"/api/ui-screenshot" in home.content
        assert b'id="file-changes"' in home.content
        assert b"function focusTimelineStage(" in home.content
        assert b'data-stage="' in home.content
        assert b"pinTimelineStage" in home.content
        assert b"function renderProgressStageBar(" in home.content
        assert b"return progressHtml + renderTaskPanes" in home.content
        assert b"${taskProgress}" not in home.content
        assert b"Before tests" in home.content
        assert b"label || name" not in home.content
        assert b"\n  10|" not in home.content

        created = client.post(
            "/api/tasks",
            json={"workspace": str(tmp_path), "goal": "Improve the UI", "auto_commit": False},
        )
        assert created.status_code == 201
        task_id = created.json()["id"]

        listed = client.get("/api/tasks")
        assert listed.status_code == 200
        body = listed.json()
        assert len(body) == 1
        assert body[0]["id"] == task_id
        assert body[0]["goal"] == "Improve the UI"
        assert body[0]["model"] == settings.model_name
        assert body[0]["base_url"] == settings.model_base_url
        assert body[0]["status"] in {"queued", "running"}
        assert "tokens_used" in body[0]
        assert "context_window" in body[0]
        if body[0]["status"] == "running":
            assert body[0]["started_at"]
            assert body[0]["finished_at"] is None
        live = manager.get(task_id)
        assert live is not None
        live.events.append(
            {
                "kind": "llm",
                "ok": True,
                "prompt_tokens": 1200,
                "completion_tokens": 80,
                "total_tokens": 1280,
            }
        )
        live.context_window = 32768
        usage = client.get(f"/api/tasks/{task_id}").json()
        assert usage["tokens_used"] == 1200
        assert usage["tokens_total"] == 1280
        assert usage["context_window"] == 32768

        missing = client.get("/api/tasks/nope")
        assert missing.status_code == 404
        assert missing.json()["error"] == "task not found"

        bad = client.post("/api/tasks", json={"workspace": str(tmp_path / "nope")})
        assert bad.status_code == 400
        assert "error" in bad.json()
    finally:
        gate.set()
        manager.shutdown(wait=False)


def test_web_ui_paginates_tasks_and_serves_favicon(
    settings: Settings, tmp_path: Path
) -> None:
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        for index in range(3):
            created = client.post(
                "/api/tasks",
                json={
                    "workspace": str(tmp_path),
                    "goal": f"Task {index}",
                    "auto_commit": False,
                },
            )
            assert created.status_code == 201
        listed = client.get("/api/tasks")
        assert listed.status_code == 200
        assert isinstance(listed.json(), list)
        assert len(listed.json()) == 3
        paged = client.get("/api/tasks", params={"page": 1, "page_size": 2})
        assert paged.status_code == 200
        body = paged.json()
        assert body["total"] == 3
        assert body["total_pages"] == 2
        assert body["page"] == 1
        assert len(body["items"]) == 2
        page_two = client.get("/api/tasks", params={"page": 2, "page_size": 2}).json()
        assert page_two["page"] == 2
        assert len(page_two["items"]) == 1
        icon = client.get("/favicon.ico")
        assert icon.status_code == 200
        assert b"<svg" in icon.content
        assert b"LoCO" in icon.content
        assert b"#c4e38a" in icon.content
        assert b"#11140f" in icon.content
        themes = client.get("/static/themes.css")
        assert themes.status_code == 200
        assert b"--bg:" in themes.content
        assert b"--accent:" in themes.content
        layout = client.get("/static/layout.css")
        assert layout.status_code == 200
        assert b"box-sizing: border-box" in layout.content
    finally:
        manager.shutdown(wait=False)


def test_finished_task_records_start_and_finish_times(
    settings: Settings, tmp_path: Path
) -> None:
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        created = client.post(
            "/api/tasks",
            json={"workspace": str(tmp_path), "goal": "Time this", "auto_commit": False},
        )
        assert created.status_code == 201
        task_id = created.json()["id"]
        body = None
        for _ in range(50):
            listed = client.get("/api/tasks").json()
            match = next((item for item in listed if item["id"] == task_id), None)
            if (
                match
                and match["status"] not in {"queued", "running"}
                and match["finished_at"]
            ):
                body = match
                break
            time.sleep(0.05)
        assert body is not None
        assert body["started_at"]
        assert body["finished_at"]
        assert body["finished_at"] >= body["started_at"]
        assert body["pr_url"] is None
    finally:
        manager.shutdown(wait=False)


def test_finished_task_includes_pr_url(settings: Settings, tmp_path: Path) -> None:
    pr = "https://github.com/acme/repo/pull/12"

    def runner(task: Task) -> CycleResult:
        return CycleResult(
            status="success",
            goal=task.goal,
            summary="opened a pull request",
            tests_passed=True,
            committed=True,
            published=True,
            commit_sha="abc123",
            reason="completed",
            pr_url=pr,
            events=[{"kind": "pr", "url": pr, "message": pr}],
        )

    manager = TaskManager(settings, runner=runner)
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        created = client.post(
            "/api/tasks",
            json={"workspace": str(tmp_path), "goal": "Ship it", "auto_commit": False},
        )
        assert created.status_code == 201
        task_id = created.json()["id"]
        body = None
        for _ in range(50):
            listed = client.get("/api/tasks").json()
            match = next((item for item in listed if item["id"] == task_id), None)
            if match and match["status"] not in {"queued", "running"}:
                body = match
                break
            time.sleep(0.05)
        assert body is not None
        assert body["published"] is True
        assert body["pr_url"] == pr
    finally:
        manager.shutdown(wait=False)


def test_submit_uses_requested_model(settings: Settings, tmp_path: Path) -> None:
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    try:
        task = manager.submit(
            tmp_path,
            "goal",
            model_name="other-coder",
            model_base_url="10.0.0.8:8000",
        )
        assert task.model_name == "other-coder"
        assert task.model_base_url == "http://10.0.0.8:8000/v1"
        fallback = manager.submit(tmp_path, "goal")
        assert fallback.model_name == settings.model_name
        assert fallback.model_base_url == settings.model_base_url
    finally:
        manager.shutdown(wait=False)


def test_web_ui_lists_models_and_queues_with_selection(
    settings: Settings, tmp_path: Path
) -> None:
    gate = threading.Event()

    def runner(task: Task) -> CycleResult:
        gate.wait(timeout=2)
        return _ok_result(task.goal)

    manager = TaskManager(
        settings,
        runner=runner,
        models_fn=lambda: ["alpha-coder", "beta-coder"],
    )
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        home = client.get("/")
        assert home.status_code == 200
        assert b'id="model"' in home.content
        assert b'id="base-url"' in home.content
        assert b'id="server-history"' in home.content
        assert b"Load models" in home.content

        models = client.get("/api/models")
        assert models.status_code == 200
        body = models.json()
        assert body["default"] == settings.model_name
        assert body["models"][0] == settings.model_name
        assert "alpha-coder" in body["models"]
        assert "beta-coder" in body["models"]

        created = client.post(
            "/api/tasks",
            json={
                "workspace": str(tmp_path),
                "goal": "Use the other model",
                "model": "beta-coder",
                "auto_commit": False,
            },
        )
        assert created.status_code == 201
        assert created.json()["model"] == "beta-coder"
        assert created.json()["base_url"] == settings.model_base_url
    finally:
        gate.set()
        manager.shutdown(wait=False)


def test_web_ui_lists_models_from_requested_host(
    settings: Settings, tmp_path: Path
) -> None:
    gate = threading.Event()

    def runner(task: Task) -> CycleResult:
        gate.wait(timeout=2)
        return _ok_result(task.goal)

    def models_fn(url: str, key: str) -> list[str]:
        if "10.0.0.8" in url:
            return ["remote-coder"]
        return ["alpha-coder", "beta-coder"]

    manager = TaskManager(settings, runner=runner, models_fn=models_fn)
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        remote = client.get("/api/models", params={"base_url": "10.0.0.8:8000"})
        assert remote.status_code == 200
        body = remote.json()
        assert body["base_url"] == "http://10.0.0.8:8000/v1"
        assert body["models"] == ["remote-coder"]
        assert settings.model_name not in body["models"]

        posted = client.post(
            "/api/models",
            json={"base_url": "10.0.0.8:8000", "model": "remote-coder"},
        )
        assert posted.status_code == 200
        assert posted.json()["models"] == ["remote-coder"]
        assert posted.json()["base_url"] == "http://10.0.0.8:8000/v1"
        assert posted.json()["servers"][0]["url"] == "http://10.0.0.8:8000/v1"
        assert posted.json()["last_model"] == "remote-coder"

        listed = client.get("/api/servers")
        assert listed.status_code == 200
        assert listed.json()["servers"][0]["url"] == "http://10.0.0.8:8000/v1"
        assert listed.json()["last_base_url"] == "http://10.0.0.8:8000/v1"
        assert listed.json()["last_model"] == "remote-coder"
        saved = json.loads((tmp_path / ".loco" / "servers.json").read_text(encoding="utf-8"))
        assert saved["servers"][0]["url"] == "http://10.0.0.8:8000/v1"
        assert saved["last_model"] == "remote-coder"

        home = client.get("/")
        assert b"http://10.0.0.8:8000/v1" in home.content
        assert b"remote-coder" in home.content

        created = client.post(
            "/api/tasks",
            json={
                "workspace": str(tmp_path),
                "goal": "Use the remote host",
                "model": "remote-coder",
                "base_url": "10.0.0.8:8000",
                "auto_commit": False,
            },
        )
        assert created.status_code == 201
        payload = created.json()
        assert payload["model"] == "remote-coder"
        assert payload["base_url"] == "http://10.0.0.8:8000/v1"
    finally:
        gate.set()
        manager.shutdown(wait=False)


def test_list_models_skips_local_default_for_other_host(settings: Settings) -> None:
    manager = TaskManager(
        settings,
        runner=lambda task: _ok_result(task.goal),
        models_fn=lambda url, key: ["remote-a"],
    )
    try:
        remote = manager.list_models(base_url="http://10.0.0.5:11434")
        assert remote == ["remote-a"]
        local = manager.list_models()
        assert local[0] == settings.model_name
        assert "remote-a" in local
    finally:
        manager.shutdown(wait=False)


def test_history_endpoint_reads_run_logs(settings: Settings, tmp_path: Path) -> None:
    runs = tmp_path / ".loco" / "runs"
    runs.mkdir(parents=True)
    (runs / "20260919T180000Z.json").write_text(
        '{"status": "success", "goal": "Past goal", "summary": "done"}\n',
        encoding="utf-8",
    )
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        history = client.get("/api/history")
        assert history.status_code == 200
        body = history.json()
        # Paginated response format
        assert len(body["items"]) == 1
        assert body["items"][0]["id"] == "20260919T180000Z"
        assert body["items"][0]["created_at"] == "2026-09-19T18:00:00Z"
        assert body["items"][0]["goal"] == "Past goal"
        assert body["page"] == 1
        assert body["page_size"] == 10
        assert body["total"] == 1
        assert body["total_pages"] == 1
    finally:
        manager.shutdown(wait=False)


def test_history_search_matches_pr_url(settings: Settings, tmp_path: Path) -> None:
    (tmp_path / "history.json").write_text(
        json.dumps(
            [
                {
                    "status": "success",
                    "goal": "Ship it",
                    "published": True,
                    "pr_url": "https://github.com/acme/repo/pull/12",
                    "created_at": "2026-09-20T12:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        found = client.get("/api/history", params={"search": "pull/12"})
        assert found.status_code == 200
        items = found.json()["items"]
        assert len(items) == 1
        assert items[0]["pr_url"] == "https://github.com/acme/repo/pull/12"
        missed = client.get("/api/history", params={"search": "no-such-pr"})
        assert missed.json()["items"] == []
    finally:
        manager.shutdown(wait=False)


def test_task_and_history_polls_omit_llm_transcripts(
    settings: Settings, tmp_path: Path
) -> None:
    blob = ("x" * 800) + "SECRET_TRANSCRIPT"
    (tmp_path / "history.json").write_text(
        json.dumps(
            [
                {
                    "status": "success",
                    "goal": "Past run",
                    "events": [
                        {
                            "kind": "llm",
                            "ok": True,
                            "messages": [{"role": "system", "content": blob}],
                            "response": blob,
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    gate = threading.Event()

    def runner(task: Task) -> CycleResult:
        task.events.append(
            {
                "kind": "llm",
                "ok": True,
                "messages": [{"role": "user", "content": blob}],
                "response": blob,
                "prompt_tokens": 9,
            }
        )
        gate.wait(timeout=2)
        return _ok_result(task.goal)

    manager = TaskManager(settings, runner=runner)
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        home = client.get("/")
        assert b"refreshQueued" in home.content
        assert b"async function refreshNow(" in home.content
        created = client.post(
            "/api/tasks",
            json={"workspace": str(tmp_path), "goal": "stay responsive"},
        )
        assert created.status_code == 201
        task_id = created.json()["id"]
        body = None
        for _ in range(50):
            listed = client.get("/api/tasks").json()
            match = next((item for item in listed if item["id"] == task_id), None)
            if match and match.get("events"):
                body = match
                break
            time.sleep(0.05)
        assert body is not None
        dumped = json.dumps(body)
        assert "SECRET_TRANSCRIPT" not in dumped
        assert "messages" not in body["events"][0]
        history = client.get("/api/history").json()
        assert "SECRET_TRANSCRIPT" not in json.dumps(history)
        assert "messages" not in history["items"][0]["events"][0]
    finally:
        gate.set()
        manager.shutdown(wait=False)


def test_history_load_reuses_unchanged_files(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_loco.web_ui import UiState

    (tmp_path / "history.json").write_text(
        json.dumps([{"status": "success", "goal": "cached"}]),
        encoding="utf-8",
    )
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    try:
        ui = UiState(
            manager,
            default_workspace=tmp_path,
            default_goal=None,
            default_create_pr=False,
        )
        first = ui.load_history(tmp_path)
        assert first[0]["goal"] == "cached"

        def boom(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("history should be cached")

        monkeypatch.setattr("agent_loco.web_ui.json.loads", boom)
        monkeypatch.setattr("agent_loco.web_ui.json.load", boom)
        second = ui.load_history(tmp_path)
        assert second is first
    finally:
        manager.shutdown(wait=False)


def test_remember_server_dedupes_and_keeps_newest_first(tmp_path: Path) -> None:
    first = remember_server(tmp_path, "10.0.0.8:8000", default="http://127.0.0.1:11434/v1")
    assert first[0]["url"] == "http://10.0.0.8:8000/v1"
    assert "http://127.0.0.1:11434/v1" in [s["url"] for s in first]
    again = remember_server(tmp_path, "http://10.0.0.8:8000/v1")
    assert again[0]["url"] == "http://10.0.0.8:8000/v1"
    assert again.count({"url": "http://10.0.0.8:8000/v1", "alias": None}) == 1
    newer = remember_server(tmp_path, "http://10.0.0.9:11434/v1")
    assert newer[0]["url"] == "http://10.0.0.9:11434/v1"
    assert newer[1]["url"] == "http://10.0.0.8:8000/v1"
    gitignore = (tmp_path / ".loco" / ".gitignore").read_text(encoding="utf-8")
    assert "*" in gitignore
    assert ".loco/" in (tmp_path / ".gitignore").read_text(encoding="utf-8")
    remember_server(tmp_path, "http://10.0.0.8:8000/v1", model="remote-coder")
    remember_server(tmp_path, "http://10.0.0.8:8000/v1")
    saved = json.loads((tmp_path / ".loco" / "servers.json").read_text(encoding="utf-8"))
    assert saved["last_base_url"] == "http://10.0.0.8:8000/v1"
    assert saved["last_model"] == "remote-coder"


def test_model_ids_from_openai_payload() -> None:
    names = model_ids_from_payload(
        {"data": [{"id": "qwen2.5-coder:14b"}, {"id": "llama3.1"}, {"id": "llama3.1"}]}
    )
    assert names == ["qwen2.5-coder:14b", "llama3.1"]


def test_normalize_model_base_url() -> None:
    assert (
        normalize_model_base_url("http://127.0.0.1:11434/v1")
        == "http://127.0.0.1:11434/v1"
    )
    assert (
        normalize_model_base_url("http://10.0.0.5:11434") == "http://10.0.0.5:11434/v1"
    )
    assert normalize_model_base_url("10.0.0.5:8000") == "http://10.0.0.5:8000/v1"
    assert (
        normalize_model_base_url("https://llm.example.com/v1/")
        == "https://llm.example.com/v1"
    )
    try:
        normalize_model_base_url("  ")
    except ValueError as exc:
        assert "required" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_web_ui_workspace_picker_browse_select_create(
    settings: Settings, tmp_path: Path
) -> None:
    other = tmp_path / "other-app"
    other.mkdir()
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        home = client.get("/")
        assert home.status_code == 200
        assert b'id="open-workspace-picker"' in home.content
        assert b'id="workspace-picker"' in home.content
        assert b'id="workspace-tabs"' in home.content
        assert b'id="add-workspace-tab"' in home.content
        assert b'id="archived-workspaces"' in home.content
        assert b"workspace-tab-archive" in home.content
        assert b'id="guidelines"' in home.content
        assert b'id="open-settings"' in home.content
        assert b'id="workspace-location"' in home.content
        assert b'id="workspace-summary"' in home.content
        assert b'class="icon-github"' in home.content
        assert b'class="icon-folder"' in home.content
        assert b'id="github-issues"' in home.content
        assert b'id="github-goals-section"' in home.content
        assert b'id="github-owner"' not in home.content
        assert b'id="github-repo"' not in home.content
        assert b'id="load-goals"' not in home.content
        assert b'id="settings-kind-label"' in home.content
        assert b'id="save-guidelines"' in home.content
        assert b'id="workspace-skills"' in home.content
        assert home.content.count(b'id="workspace-skills"') == 1
        assert b'id="open-skills"' in home.content
        assert b'id="skills-panel"' in home.content
        assert b'id="skills-search"' in home.content
        assert b'id="open-guidelines"' in home.content
        assert b'id="guidelines-panel"' in home.content
        assert b'id="skills-popup"' not in home.content
        assert b'id="guidelines-popup"' not in home.content
        assert b'id="clone-skill-source"' in home.content
        assert b'id="skill-info-pop"' in home.content
        assert b"skill-markdown" in home.content
        assert b'id="workspace" name="workspace" type="hidden"' in home.content
        assert b"Clone a repository" in home.content

        listed = client.get("/api/workspaces")
        assert listed.status_code == 200
        body = listed.json()
        assert body["current"] == str(tmp_path.resolve())
        assert body["home"]
        assert any(item["path"] == str(tmp_path.resolve()) for item in body["workspaces"])
        assert "is_git" in body["workspaces"][0]
        assert "git_remote" in body["workspaces"][0]

        browse = client.get("/api/workspaces/browse", params={"path": str(tmp_path)})
        assert browse.status_code == 200
        listing = browse.json()
        assert listing["path"] == str(tmp_path.resolve())
        names = [item["name"] for item in listing["entries"]]
        assert "other-app" in names

        selected = client.post("/api/workspaces/select", json={"path": str(other)})
        assert selected.status_code == 200
        assert selected.json()["current"] == str(other.resolve())
        saved = json.loads((tmp_path / ".loco" / "workspaces.json").read_text(encoding="utf-8"))
        assert saved["last"] == str(other.resolve())

        created = client.post(
            "/api/workspaces/create",
            json={"path": str(tmp_path / "fresh-app")},
        )
        assert created.status_code == 200
        fresh = tmp_path / "fresh-app"
        assert created.json()["current"] == str(fresh.resolve())
        assert (fresh / ".loco" / "config.yaml").exists()
        assert (fresh / ".git").exists()

        missing = client.post("/api/workspaces/select", json={"path": str(tmp_path / "nope")})
        assert missing.status_code == 400
        empty_clone = client.post("/api/workspaces/clone", json={"url": "  "})
        assert empty_clone.status_code == 400
    finally:
        manager.shutdown(wait=False)


def test_web_ui_workspace_guidelines_and_tabs(
    settings: Settings, tmp_path: Path
) -> None:
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        home = client.get("/")
        assert home.status_code == 200
        listed = client.get("/api/workspaces")
        body = listed.json()
        assert body["current"] == str(tmp_path.resolve())
        assert "You are loco" in body["default_guidelines"]
        assert body["custom_guidelines"] is False
        saved = client.put(
            "/api/workspaces/guidelines",
            json={
                "path": str(tmp_path),
                "guidelines": "Prefer pytest. Never skip tests.",
            },
        )
        assert saved.status_code == 200
        assert saved.json()["custom_guidelines"] is True
        assert "Prefer pytest" in saved.json()["guidelines"]
        created = client.post(
            "/api/workspaces/create",
            json={
                "path": str(tmp_path / "fresh-rules"),
                "guidelines": "Use Go modules.",
            },
        )
        assert created.status_code == 200
        fresh = tmp_path / "fresh-rules"
        payload = created.json()
        assert payload["current"] == str(fresh.resolve())
        assert payload["custom_guidelines"] is True
        assert "Use Go modules" in payload["guidelines"]
        paths = [item["path"] for item in payload["workspaces"]]
        assert paths[0] == str(tmp_path.resolve())
        assert str(fresh.resolve()) in paths
        switched = client.post(
            "/api/workspaces/select", json={"path": str(tmp_path)}
        )
        assert switched.status_code == 200
        assert switched.json()["current"] == str(tmp_path.resolve())
        assert [item["path"] for item in switched.json()["workspaces"]][:2] == [
            str(tmp_path.resolve()),
            str(fresh.resolve()),
        ]
        assert switched.json()["custom_guidelines"] is True
        reset = client.put(
            "/api/workspaces/guidelines",
            json={"path": str(tmp_path), "guidelines": ""},
        )
        assert reset.status_code == 200
        assert reset.json()["custom_guidelines"] is False
    finally:
        manager.shutdown(wait=False)


def test_web_ui_restores_last_workspace(settings: Settings, tmp_path: Path) -> None:
    last = tmp_path / "picked"
    last.mkdir()
    loco = tmp_path / ".loco"
    loco.mkdir()
    (loco / "workspaces.json").write_text(
        json.dumps(
            {
                "last": str(last.resolve()),
                "workspaces": [{"path": str(last.resolve())}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        listed = client.get("/api/workspaces")
        assert listed.status_code == 200
        assert listed.json()["current"] == str(last.resolve())
        home = client.get("/")
        assert str(last.resolve()).encode() in home.content
    finally:
        manager.shutdown(wait=False)


def test_web_ui_defaults_create_pr_on_for_a_git_workspace(
    settings: Settings, tmp_path: Path
) -> None:
    (tmp_path / ".git").mkdir()
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        meta = client.get("/api/meta")
        assert meta.status_code == 200
        assert meta.json()["default_create_pr"] is True
        home = client.get("/")
        assert b'id="create-pr"' in home.content
        assert b'id="create-pr" type="checkbox" checked' in home.content
    finally:
        manager.shutdown(wait=False)


def test_web_ui_archives_and_forgets_workspaces(
    settings: Settings, tmp_path: Path
) -> None:
    extra = tmp_path / "extra-app"
    extra.mkdir()
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        home = client.get("/")
        assert home.status_code == 200
        selected = client.post("/api/workspaces/select", json={"path": str(extra)})
        assert selected.status_code == 200
        archived = client.post("/api/workspaces/archive", json={"path": str(extra)})
        assert archived.status_code == 200
        body = archived.json()
        assert body["current"] == str(tmp_path.resolve())
        assert [item["path"] for item in body["workspaces"]] == [str(tmp_path.resolve())]
        assert [item["path"] for item in body["archived"]] == [str(extra.resolve())]
        refused = client.post("/api/workspaces/archive", json={"path": str(tmp_path)})
        assert refused.status_code == 400
        restored = client.post("/api/workspaces/restore", json={"path": str(extra)})
        assert restored.status_code == 200
        assert restored.json()["current"] == str(extra.resolve())
        assert restored.json()["archived"] == []
        client.post("/api/workspaces/archive", json={"path": str(extra)})
        forgotten = client.post("/api/workspaces/forget", json={"path": str(extra)})
        assert forgotten.status_code == 200
        assert forgotten.json()["archived"] == []
        assert extra.exists()
        assert [item["path"] for item in forgotten.json()["workspaces"]] == [
            str(tmp_path.resolve())
        ]
    finally:
        manager.shutdown(wait=False)


def test_web_ui_clones_repository_into_local_workspace(
    settings: Settings, tmp_path: Path
) -> None:
    from tests.support import init_git_repo

    source = tmp_path / "upstream"
    source.mkdir()
    (source / "readme.txt").write_text("hello\n", encoding="utf-8")
    init_git_repo(source)
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        cloned = client.post(
            "/api/workspaces/clone",
            json={
                "url": str(source),
                "parent": str(tmp_path / "projects"),
                "name": "checkout",
            },
        )
        assert cloned.status_code == 200
        dest = tmp_path / "projects" / "checkout"
        assert cloned.json()["current"] == str(dest.resolve())
        assert (dest / "readme.txt").read_text(encoding="utf-8") == "hello\n"
        assert (dest / ".loco" / "config.yaml").exists()
        again = client.post(
            "/api/workspaces/clone",
            json={"url": str(source), "parent": str(tmp_path / "projects"), "name": "checkout"},
        )
        assert again.status_code == 400
    finally:
        manager.shutdown(wait=False)


def test_web_ui_lists_enables_and_clones_workspace_skills(
    settings: Settings, tmp_path: Path
) -> None:
    from tests.support import init_git_repo

    remote = tmp_path / "skill-origin"
    skill = remote / "skills" / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: code-review\ndescription: Review diffs.\n---\n\nRead the diff.\n",
        encoding="utf-8",
    )
    init_git_repo(remote)
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        listed = client.get("/api/workspaces/skills", params={"path": str(tmp_path)})
        assert listed.status_code == 200
        names = [item["name"] for item in listed.json()["skills"]]
        assert "tdd" in names
        assert listed.json()["enabled"] == []
        tdd = next(item for item in listed.json()["skills"] if item["name"] == "tdd")
        assert tdd["summary"] == "Test-driven development."
        assert "body" not in tdd
        detail = client.get(
            "/api/workspaces/skills/detail",
            params={"path": str(tmp_path), "name": "tdd"},
        )
        assert detail.status_code == 200
        payload = detail.json()
        assert "red" in payload["body"].lower()
        assert "<h1>" in payload["html"]
        assert "<h2>" in payload["html"]
        assert "<li>" in payload["html"]
        enabled = client.put(
            "/api/workspaces/skills",
            json={"path": str(tmp_path), "enabled": ["tdd"]},
        )
        assert enabled.status_code == 200
        assert "tdd" in enabled.json()["enabled"]
        cloned = client.post(
            "/api/workspaces/skills/sources",
            json={"path": str(tmp_path), "url": str(remote)},
        )
        assert cloned.status_code == 200
        cloned_names = [item["name"] for item in cloned.json()["skills"]]
        assert "code-review" in cloned_names
        assert cloned.json()["sources"]
        pulled = client.post(
            "/api/workspaces/skills/sources/sync",
            json={
                "path": str(tmp_path),
                "slug": cloned.json()["added"]["slug"],
            },
        )
        assert pulled.status_code == 200
    finally:
        manager.shutdown(wait=False)


def test_web_ui_progress_includes_file_history_and_timestamps(
    settings: Settings, tmp_path: Path
) -> None:
    from agent_loco.progress import record_event

    def runner(task: Task) -> CycleResult:
        record_event(
            kind="file",
            path="hello.py",
            action="created",
            diff="--- /dev/null\n+++ b/hello.py\n+print('hi')\n",
        )
        record_event(kind="llm", purpose="agent", ok=True, elapsed_ms=12)
        record_event(
            kind="test",
            command="python3 check.py",
            ok=False,
            phase="after",
            elapsed_ms=90,
            output="AssertionError: expected 5",
        )
        return _ok_result(task.goal)

    manager = TaskManager(settings, runner=runner)
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        home = client.get("/")
        assert home.status_code == 200
        assert b".timeline" in home.content
        assert b'id="progress-timeline"' in home.content
        assert b"captureProgressScroll" in home.content
        assert b"restoreProgressScroll" in home.content
        assert b"data-history-key" in home.content
        created = client.post(
            "/api/tasks",
            json={"workspace": str(tmp_path), "goal": "Show diffs", "auto_commit": False},
        )
        assert created.status_code == 201
        task_id = created.json()["id"]
        body = None
        for _ in range(50):
            match = next(
                (item for item in client.get("/api/tasks").json() if item["id"] == task_id),
                None,
            )
            if match and match["status"] not in {"queued", "running"}:
                body = match
                break
            time.sleep(0.05)
        assert body is not None
        assert body["logs"]
        assert body["logs"][0][:4].isdigit()
        assert body["logs"][0].split(" ", 1)[0].endswith("Z")
        kinds = [event["kind"] for event in body["events"]]
        assert "file" in kinds
        assert "llm" in kinds
        assert "test" in kinds
        file_event = next(event for event in body["events"] if event["kind"] == "file")
        assert file_event["path"] == "hello.py"
        assert "print('hi')" in file_event["diff"]
        test_event = next(event for event in body["events"] if event["kind"] == "test")
        assert test_event["ok"] is False
        assert "AssertionError" in test_event["output"]
        assert b"Tests failed" in home.content
        assert b'event.kind === "test"' in home.content
        assert b"groupTimelineEvents" in home.content
        assert b"Thinking" in home.content
        assert b"timeline-item think" in home.content
        assert b"function renderFileChanges(" in home.content
        assert b'data-task-pane="changes"' in home.content
        assert b'data-task-pane="screenshots"' in home.content
        assert b"No file changes in this task." in home.content
    finally:
        manager.shutdown(wait=False)


def test_ui_screenshot_api_serves_pngs(settings: Settings, tmp_path: Path) -> None:
    loco = tmp_path / ".loco"
    shots = loco / "ui-screenshots"
    shots.mkdir(parents=True)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
    (loco / "ui-review.png").write_bytes(png)
    (shots / "ui-review_goal.png").write_bytes(png)
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        legacy = client.get("/api/ui-screenshot", params={"name": "ui-review.png"})
        assert legacy.status_code == 200
        assert legacy.content[:8] == b"\x89PNG\r\n\x1a\n"
        unique = client.get("/api/ui-screenshot", params={"name": "ui-review_goal.png"})
        assert unique.status_code == 200
        missing = client.get("/api/ui-screenshot", params={"name": "missing.png"})
        assert missing.status_code == 404
        traversal = client.get("/api/ui-screenshot", params={"name": "../config.yaml"})
        assert traversal.status_code == 404
    finally:
        manager.shutdown(wait=False)


def _github_workspace(root: Path) -> None:
    from tests.support import init_git_repo

    from agent_loco.sandbox import Workspace
    from agent_loco.tools.git import run_git

    (root / "readme.txt").write_text("hello\n", encoding="utf-8")
    init_git_repo(root)
    run_git(Workspace(root), ["remote", "add", "origin", "git@github.com:acme/demo.git"])


def test_web_ui_hides_github_issues_for_local_folders(
    settings: Settings, tmp_path: Path
) -> None:
    manager = TaskManager(settings, runner=lambda task: _ok_result())
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        home = client.get("/")
        assert home.status_code == 200
        assert b'id="github-goals-section" hidden' in home.content
        missing = client.get("/api/goals")
        assert missing.status_code == 400
        assert missing.json()["error"] == "workspace is not a GitHub repository"
        assert missing.json()["issues"] == []
    finally:
        manager.shutdown(wait=False)


def test_web_ui_loads_github_issues_from_workspace_remote(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _github_workspace(tmp_path)

    def fake_load(owner: str, repo: str, state: str = "open", per_page: int = 100) -> dict:
        assert owner == "acme"
        assert repo == "demo"
        assert state == "open"
        return {
            "issues": [
                {
                    "id": 1,
                    "number": 12,
                    "title": "Add a favicon",
                    "body": "Put favicon.ico in static files.",
                    "url": "https://github.com/acme/demo/issues/12",
                    "state": "open",
                    "label": "#12 Add a favicon",
                    "goal": (
                        "#12 Add a favicon\n"
                        "https://github.com/acme/demo/issues/12\n\n"
                        "Implement this GitHub issue. Do the work it describes; "
                        "do not only summarize it.\n\n"
                        "Put favicon.ico in static files.\n"
                    ),
                }
            ],
            "owner": owner,
            "repo": repo,
            "state": state,
        }

    monkeypatch.setattr(
        "agent_loco.runtime.importer.load_goals_from_issues", fake_load
    )
    manager = TaskManager(settings, runner=lambda task: _ok_result())
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        home = client.get("/")
        assert home.status_code == 200
        assert b'id="github-goals-section"' in home.content
        assert b'id="github-goals-section" hidden' not in home.content
        assert b'id="github-owner"' not in home.content
        loaded = client.get("/api/goals")
        assert loaded.status_code == 200
        body = loaded.json()
        assert body["owner"] == "acme"
        assert body["repo"] == "demo"
        assert body["issues"][0]["label"] == "#12 Add a favicon"
        assert body["issues"][0]["goal"].startswith("#12 Add a favicon")
        assert "Put favicon.ico in static files." in body["issues"][0]["goal"]
        assert b"issue.label" in home.content
        assert b"fillGoalFromIssue" in home.content
        filtered = client.get(
            "/api/goals",
            params={"workspace": str(tmp_path), "state": "open"},
        )
        assert filtered.status_code == 200
        assert filtered.json()["issues"][0]["number"] == 12
    finally:
        manager.shutdown(wait=False)


def _failed_result(goal: str | None = "do the thing", branch: str | None = None) -> CycleResult:
    return CycleResult(
        status="failed",
        goal=goal,
        summary="not done",
        tests_passed=False,
        committed=False,
        published=False,
        commit_sha=None,
        reason="goal not met",
        branch=branch,
    )


def test_web_ui_reruns_failed_task(settings: Settings, tmp_path: Path) -> None:
    statuses = iter(["failed", "success"])

    def runner(task: Task) -> CycleResult:
        status = next(statuses)
        if status == "failed":
            return _failed_result(task.goal, branch="loco/feature")
        return _ok_result(task.goal)

    manager = TaskManager(settings, runner=runner)
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        created = client.post(
            "/api/tasks",
            json={"workspace": str(tmp_path), "goal": "Allow rerun", "auto_commit": False},
        )
        assert created.status_code == 201
        task_id = created.json()["id"]
        body = None
        for _ in range(50):
            listed = client.get("/api/tasks").json()
            match = next((item for item in listed if item["id"] == task_id), None)
            if match and match["status"] not in {"queued", "running"}:
                body = match
                break
            time.sleep(0.05)
        assert body is not None
        assert body["status"] == "failed"
        rerun = client.post(f"/api/tasks/{task_id}/rerun", json={})
        assert rerun.status_code == 201
        assert rerun.json()["id"] != task_id
        assert rerun.json()["goal"] == "Allow rerun"
        success = client.post(f"/api/tasks/{rerun.json()['id']}/rerun", json={})
        assert success.status_code == 400
    finally:
        manager.shutdown(wait=False)


def test_web_ui_updates_successful_task_on_its_branch(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent_loco.web_ui.pull_request_state", lambda _url: "open")
    seen: list[Task] = []

    def runner(task: Task) -> CycleResult:
        seen.append(task)
        return CycleResult(
            status="success",
            goal=task.goal,
            summary="done",
            tests_passed=True,
            committed=True,
            published=True,
            commit_sha="abc123",
            reason="completed",
            pr_url="https://example.test/pull/9",
            branch="loco/feature",
        )

    manager = TaskManager(settings, runner=runner)
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        created = client.post(
            "/api/tasks",
            json={"workspace": str(tmp_path), "goal": "Ship the panel", "auto_commit": False},
        )
        assert created.status_code == 201
        task_id = created.json()["id"]
        body = None
        for _ in range(50):
            listed = client.get("/api/tasks").json()
            match = next((item for item in listed if item["id"] == task_id), None)
            if match and match["status"] not in {"queued", "running"}:
                body = match
                break
            time.sleep(0.05)
        assert body is not None
        assert body["status"] == "success"
        assert body["branch"] == "loco/feature"
        updated = client.post(
            f"/api/tasks/{task_id}/rerun",
            json={"goal": "Ship the panel\n\nAlso leave room for the PR link"},
        )
        assert updated.status_code == 201
        payload = updated.json()
        assert payload["id"] != task_id
        assert "PR link" in payload["goal"]
        assert payload["create_pr"] is True
        follow = None
        for _ in range(50):
            if len(seen) >= 2:
                follow = seen[1]
                break
            time.sleep(0.05)
        assert follow is not None
        assert follow.resume_branch == "loco/feature"
        assert follow.resume_sha == "abc123"
    finally:
        manager.shutdown(wait=False)


def test_web_ui_refuses_update_when_pull_request_is_not_open(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agent_loco.web_ui.pull_request_state", lambda _url: "merged")

    def runner(task: Task) -> CycleResult:
        return CycleResult(
            status="success",
            goal=task.goal,
            summary="done",
            tests_passed=True,
            committed=True,
            published=True,
            commit_sha="abc123",
            reason="completed",
            pr_url="https://example.test/pull/10",
            branch="loco/feature",
        )

    manager = TaskManager(settings, runner=runner)
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        created = client.post(
            "/api/tasks",
            json={"workspace": str(tmp_path), "goal": "Ship the panel", "auto_commit": False},
        )
        task_id = created.json()["id"]
        body = None
        for _ in range(50):
            listed = client.get("/api/tasks").json()
            match = next((item for item in listed if item["id"] == task_id), None)
            if match and match["status"] not in {"queued", "running"}:
                body = match
                break
            time.sleep(0.05)
        assert body is not None
        refused = client.post(
            f"/api/tasks/{task_id}/rerun",
            json={"goal": "Ship the panel\n\nMore context"},
        )
        assert refused.status_code == 409
        queued = client.post(
            "/api/tasks",
            json={
                "workspace": str(tmp_path),
                "goal": "More context",
                "pr_url": "https://example.test/pull/10",
                "resume_branch": "loco/feature",
            },
        )
        assert queued.status_code == 409
    finally:
        manager.shutdown(wait=False)


def test_web_ui_reports_pull_request_states(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_state(url: str) -> str:
        if url.endswith("/9"):
            return "open"
        if url.endswith("/10"):
            return "merged"
        return ""

    monkeypatch.setattr("agent_loco.web_ui.pull_request_state", fake_state)
    manager = TaskManager(settings, runner=lambda task: _ok_result(task.goal))
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        listed = client.post(
            "/api/pulls/status",
            json={
                "urls": [
                    "https://example.test/pull/9",
                    "https://example.test/pull/10",
                    "not a pull request",
                ]
            },
        )
        assert listed.status_code == 200
        states = listed.json()["states"]
        assert states["https://example.test/pull/9"] == "open"
        assert states["https://example.test/pull/10"] == "merged"
        assert "not a pull request" not in states
    finally:
        manager.shutdown(wait=False)


def test_submit_accepts_resume_branch(settings: Settings, tmp_path: Path) -> None:
    seen: list[str | None] = []

    def runner(task: Task) -> CycleResult:
        seen.append(task.resume_branch)
        return _ok_result(task.goal)

    manager = TaskManager(settings, runner=runner)
    app = create_app(manager, default_workspace=tmp_path)
    client = TestClient(app)
    try:
        created = client.post(
            "/api/tasks",
            json={
                "workspace": str(tmp_path),
                "goal": "Continue the work",
                "resume_branch": "loco/feature",
                "auto_commit": False,
            },
        )
        assert created.status_code == 201
        for _ in range(50):
            listed = client.get("/api/tasks").json()
            if listed and listed[0]["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        assert seen == ["loco/feature"]
    finally:
        manager.shutdown(wait=False)


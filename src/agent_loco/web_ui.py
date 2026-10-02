from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from agent_loco.config import Settings
from agent_loco.llm.client import normalize_model_base_url
from agent_loco.progress import attach_token_usage, clip_text, llm_turn_snippets, public_run_item
from agent_loco.runtime.importer import load_goals_from_workspace
from agent_loco.runtime.project import (
    default_guidelines,
    guidelines_are_custom,
    save_guidelines,
)
from agent_loco.runtime.servers import (
    list_known_servers,
    load_selection,
    remember_server,
    update_server_alias,
)
from agent_loco.runtime.skills import (
    add_skill_source,
    get_skill,
    save_enabled_skills,
    skills_catalog,
    sync_skill_source,
)
from agent_loco.runtime.tasks import Task, TaskManager
from agent_loco.runtime.uireview import resolve_ui_screenshot
from agent_loco.runtime.workspaces import (
    archive_workspace,
    browse_directory,
    clone_workspace,
    create_workspace,
    forget_workspace,
    last_workspace,
    load_archived_workspaces,
    load_workspaces,
    remember_workspace,
)
from agent_loco.tools.git import extract_pr_url, pull_request_state

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
FAVICON_SVG = (STATIC_DIR / "favicon.svg").read_text(encoding="utf-8")


_PR_STATE_TTL = 45.0
_PR_STATE_CACHE: dict[str, tuple[float, str]] = {}
_PR_STATE_LOCK = threading.Lock()


def lookup_pull_state(url: str) -> str:
    """Cached pull-request state. Empty when the URL is missing or gh cannot tell."""
    clean = extract_pr_url(url or "") or ""
    if not clean:
        return ""
    now = time.monotonic()
    with _PR_STATE_LOCK:
        cached = _PR_STATE_CACHE.get(clean)
        if cached is not None and now - cached[0] < _PR_STATE_TTL:
            return cached[1]
    state = pull_request_state(clean)
    with _PR_STATE_LOCK:
        _PR_STATE_CACHE[clean] = (time.monotonic(), state)
    return state


class TaskCreate(BaseModel):
    workspace: str | None = None
    goal: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    auto_commit: bool | None = None
    create_pr: bool | None = None
    resume_branch: str | None = None
    resume_sha: str | None = None
    pr_url: str | None = None


class RerunPayload(BaseModel):
    sha: str | None = None
    goal: str | None = None


class PullStatusQuery(BaseModel):
    urls: list[str] = []


class ModelsQuery(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


class HistoryQuery(BaseModel):
    page: int = 1
    page_size: int = 10
    search: str | None = None


class ServerAliasUpdate(BaseModel):
    url: str
    alias: str | None = None


class UrlCreate(BaseModel):
    url: str
    alias: str | None = None


class WorkspaceSelect(BaseModel):
    path: str
    guidelines: str | None = None


class WorkspaceGuidelines(BaseModel):
    path: str | None = None
    guidelines: str | None = None


class WorkspaceSkillsUpdate(BaseModel):
    path: str | None = None
    enabled: list[str] | None = None


class WorkspaceSkillSource(BaseModel):
    path: str | None = None
    url: str
    ref: str | None = None


class WorkspaceSkillSync(BaseModel):
    path: str | None = None
    slug: str
    ref: str | None = None


class WorkspaceClone(BaseModel):
    url: str
    parent: str | None = None
    name: str | None = None
    guidelines: str | None = None


class UiState:
    def __init__(
        self,
        manager: TaskManager,
        *,
        default_workspace: Path,
        default_goal: str | None,
        default_create_pr: bool | None,
    ) -> None:
        self.manager = manager
        self.store_root = str(default_workspace.expanduser().resolve())
        self.default_workspace = last_workspace(
            Path(self.store_root), default=self.store_root
        )
        self.default_goal = default_goal or ""
        self.default_auto_commit = manager.settings.auto_commit
        self._history_lock = threading.Lock()
        self._history_cache: dict[str, tuple[object, list[dict[str, Any]]]] = {}
        if default_create_pr is not None:
            self.default_create_pr = bool(default_create_pr)
        else:
            workspace = Path(self.default_workspace)
            self.default_create_pr = bool(
                manager.settings.create_pr or (workspace / ".git").exists()
            )

    def known_servers(self) -> list[dict[str, str]]:
        return list_known_servers(
            Path(self.default_workspace),
            default=self.manager.settings.model_base_url,
        )

    def selection(self) -> dict[str, str | list[dict[str, str]] | None]:
        return load_selection(
            Path(self.default_workspace),
            default_url=self.manager.settings.model_base_url,
            default_model=self.manager.settings.model_name,
        )

    def remember_current_workspace(self) -> list[dict[str, str | bool]]:
        return remember_workspace(Path(self.store_root), self.default_workspace)

    def set_workspace(self, path: str) -> list[dict[str, str | bool]]:
        remembered = remember_workspace(Path(self.store_root), path)
        self.default_workspace = last_workspace(
            Path(self.store_root), default=path
        )
        return remembered

    def _refresh_current_workspace(self) -> None:
        self.default_workspace = last_workspace(
            Path(self.store_root), default=self.store_root
        )

    def archive_workspace_path(self, path: str) -> list[dict[str, str | bool]]:
        remembered = archive_workspace(Path(self.store_root), path)
        self._refresh_current_workspace()
        return remembered

    def forget_workspace_path(self, path: str) -> list[dict[str, str | bool]]:
        remembered = forget_workspace(Path(self.store_root), path)
        self._refresh_current_workspace()
        return remembered

    def remember_server(self, url: str, model: str | None = None) -> list[dict[str, str]]:
        return remember_server(
            Path(self.default_workspace),
            url,
            model=model,
            default=self.manager.settings.model_base_url,
        )

    def template_vars(self) -> dict[str, Any]:
        selected = self.selection()
        workspaces = self.remember_current_workspace()
        current = next(
            (item for item in workspaces if item["path"] == self.default_workspace),
            None,
        )
        return {
            "default_workspace": self.default_workspace,
            "known_workspaces": workspaces,
            "workspace_is_git": bool((current or {}).get("is_git")),
            "workspace_git_remote": (current or {}).get("git_remote") or "",
            "guidelines": (current or {}).get("guidelines") or default_guidelines(),
            "custom_guidelines": bool((current or {}).get("custom_guidelines")),
            "default_guidelines": default_guidelines(),
            "default_goal": self.default_goal,
            "default_auto_commit": self.default_auto_commit,
            "default_create_pr": self.default_create_pr,
            "default_model": selected["last_model"] or self.manager.settings.model_name,
            "default_base_url": selected["last_base_url"]
            or self.manager.settings.model_base_url,
            "known_servers": selected["servers"],
            "max_concurrent": self.manager.max_concurrent,
        }

    def meta(self) -> dict[str, Any]:
        selected = self.selection()
        return {
            "default_workspace": self.default_workspace,
            "known_workspaces": load_workspaces(
                Path(self.store_root), default=self.default_workspace
            ),
            "default_goal": self.default_goal,
            "default_auto_commit": self.default_auto_commit,
            "default_create_pr": self.default_create_pr,
            "default_model": selected["last_model"] or self.manager.settings.model_name,
            "default_base_url": selected["last_base_url"]
            or self.manager.settings.model_base_url,
            "known_servers": selected["servers"],
            "max_concurrent": self.manager.max_concurrent,
            **self.manager.counts(),
        }

    def load_history(self, workspace_root: Path) -> list[dict[str, Any]]:
        """Load cycle logs from `.loco/runs/` and history.json, newest first."""
        root = Path(workspace_root).expanduser().resolve()
        key = str(root)
        fingerprint = _history_fingerprint(root)
        with self._history_lock:
            cached = self._history_cache.get(key)
            if cached is not None and cached[0] == fingerprint:
                return cached[1]
            items = [public_run_item(item) for item in self._read_history(root)]
            self._history_cache[key] = (fingerprint, items)
            return items

    def _read_history(self, workspace_root: Path) -> list[dict[str, Any]]:
        """Read run files and history.json without caching or slimming."""
        runs_dir = Path(workspace_root) / ".loco" / "runs"
        results: list[dict[str, Any]] = []

        # Load from .loco/runs/ directory (current runs)
        if runs_dir.exists():
            for path in sorted(runs_dir.glob("*.json"), reverse=True):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(data, dict):
                    continue
                data.setdefault("id", path.stem)
                if not data.get("created_at"):
                    parsed = _created_at_from_run_id(path.stem)
                    if parsed:
                        data["created_at"] = parsed
                results.append(data)

        # Load from history.json (persisted history)
        history_file = Path(workspace_root) / "history.json"
        try:
            if history_file.exists():
                with open(history_file, encoding="utf-8") as f:
                    history = json.load(f)
                if isinstance(history, list):
                    # Add all historical items (the file has them newest first)
                    results.extend(history)
        except (OSError, json.JSONDecodeError):
            # If we can't read the history file, continue with current runs
            pass

        return results

    def load_goals_from_github(
        self,
        workspace: str | Path | None = None,
        state: str = "open",
    ) -> dict[str, Any]:
        """Load issues as selectable goals from the workspace's GitHub remote."""
        root = Path(workspace or self.default_workspace)
        return load_goals_from_workspace(root, state=state)

    def paginate_history(
        self,
        workspace_root: Path,
        *,
        page: int = 1,
        page_size: int = 10,
        search: str | None = None,
    ) -> dict[str, Any]:
        """Load and paginate history with optional search filter."""
        all_items = self.load_history(workspace_root)

        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            all_items = [
                item for item in all_items
                if any(
                    str(value).lower().find(search_lower) >= 0
                    for value in [
                        item.get("goal", ""),
                        item.get("summary", ""),
                        item.get("reason", ""),
                        item.get("status", ""),
                        item.get("id", ""),
                        item.get("pr_url", ""),
                        item.get("created_at", ""),
                        " ".join(
                            str(
                                event.get("path")
                                or event.get("url")
                                or event.get("message")
                                or event.get("output")
                                or event.get("command")
                                or ""
                            )
                            for event in item.get("events") or []
                            if isinstance(event, dict)
                        ),
                    ]
                )
            ]

        total = len(all_items)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        items = [attach_token_usage(item) for item in all_items[start_idx:end_idx]]

        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        }


_CONSOLE_SNIPPET = 400


def _history_fingerprint(
    root: Path,
) -> tuple[tuple[tuple[str, int, int], ...], tuple[int, int]]:
    """Cheap mtime/size signature so history polls can reuse a parsed cache."""
    runs_dir = root / ".loco" / "runs"
    entries: list[tuple[str, int, int]] = []
    if runs_dir.is_dir():
        for path in sorted(runs_dir.glob("*.json")):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((path.name, stat.st_mtime_ns, stat.st_size))
    history_file = root / "history.json"
    hist = (0, 0)
    try:
        if history_file.exists():
            stat = history_file.stat()
            hist = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        pass
    return (tuple(entries), hist)


def _live_task(manager: TaskManager) -> Task | None:
    """Newest running or queued task, else the newest task."""
    tasks = manager.list()
    running = [task for task in tasks if task.status in {"queued", "running"}]
    return running[0] if running else (tasks[0] if tasks else None)


def _console_snippet(text: object, limit: int = _CONSOLE_SNIPPET) -> str:
    return clip_text(text, limit)


def _agent_model_snippets(event: dict[str, Any]) -> tuple[str, str]:
    """Last non-assistant prompt and the model reply, clipped for the live console."""
    agent = str(event.get("agent") or "")
    model = str(event.get("model") or "")
    if agent or model:
        return agent, model or _console_snippet(event.get("response"))
    return llm_turn_snippets(event.get("messages"), event.get("response"))


def _console_event_payload(event: dict[str, Any]) -> dict[str, Any]:
    agent, model = _agent_model_snippets(event)
    return {
        "kind": event.get("kind"),
        "at": event.get("at"),
        "purpose": event.get("purpose"),
        "elapsed_ms": event.get("elapsed_ms"),
        "ok": event.get("ok"),
        "path": event.get("path"),
        "action": event.get("action"),
        "net_diff": event.get("net_diff") or "",
        "diff": event.get("diff") or "",
        "command": event.get("command"),
        "phase": event.get("phase"),
        "message": event.get("message"),
        "url": event.get("url"),
        "agent": agent,
        "model": model,
    }


def _live_task_logs(manager: TaskManager) -> tuple[str, list[str]]:
    """Logs for the newest running task, or the newest task if none is running.

    Returns both the traditional log lines AND the detailed event-based logs
    for the in-depth back-and-forth between LLM and agent.
    """
    from agent_loco.logging import format_elapsed

    task = _live_task(manager)
    if task is None:
        return "", []
    
    # Build enriched log output: interleaved timestamps from original logs
    # plus detailed event logs showing LLM reasoning, file changes, tests, and PRs
    all_lines = []
    seen_logs = set()
    
    # Process events in order to create rich console output
    for event in task.events:
        kind = event.get("kind", "")
        at = event.get("at", "")
        purpose = event.get("purpose", "")
        elapsed = event.get("elapsed_ms")
        ok = event.get("ok")
        path = event.get("path", "")
        action = event.get("action", "")
        diff = event.get("net_diff", "") or event.get("diff", "")
        command = event.get("command", "")
        test_ok = event.get("ok", ok)
        phase = event.get("phase", "")
        message = event.get("message", "")
        url = event.get("url", "")
        
        parts = [f"[{at}]"]
        
        if kind == "llm":
            status = "✓" if ok else "✗"
            messages = event.get("messages")
            response = event.get("response")
            if elapsed:
                time_str = format_elapsed(elapsed/1000)
            else:
                time_str = ""
            
            # Show detailed LLM conversation when messages/response are available
            if messages and response:
                # LLM interaction details
                parts.append(f" {status} LLM ({purpose})")
                count = len(messages) if isinstance(messages, list) else "msg"
                parts.append(f"  ↓ {count} messages")
                if isinstance(messages, list) and len(messages) > 0:
                    for i, msg in enumerate(messages[:3]):
                        if isinstance(msg, dict):
                            role = str(msg.get("role", "unknown"))[:3]
                        else:
                            role = str(msg)[:20]
                        parts.append(f"  → [{i+1}] {role}")
                    if len(messages) > 3:
                        all_lines.append(" ".join(parts))
                        parts = [f"[{at}]"]
                        parts.append(f"  · {len(messages)-3} more messages")
                        all_lines.append(" ".join(parts))
                        parts = [f"[{at}] {status} LLM ({purpose})"]
                parts.append(f"  ↑ Response: {response[:100]}{'…' if len(response) > 100 else ''}")
                all_lines.append(" ".join(parts))
                parts = [f"[{at}]"]
            else:
                # Sparse legacy format - no detailed conversation data
                if time_str:
                    parts.append(f" {status} LLM ({purpose}) in {time_str}")
                else:
                    parts.append(f" {status} LLM ({purpose})")
                all_lines.append(" ".join(parts))
        
        elif kind == "file":
            if diff:
                # Show a trimmed repr of the diff
                preview = diff.replace("\n", " ").strip()[:80]
                if len(diff) > 80:
                    preview += "…"
                parts.append(f" {action} {path} [{preview}]")
            else:
                parts.append(f" {action} {path}")
            all_lines.append(" ".join(parts))
        
        elif kind == "test":
            status = "✓" if test_ok else "✗"
            if command:
                cmd_display = command[:50] + ("…" if len(command) > 50 else "")
                parts.append(f" {status} test ({cmd_display})")
            elif phase:
                parts.append(f" {status} {phase} tests")
            else:
                parts.append(f" {status} test run")
            all_lines.append(" ".join(parts))
        
        elif kind == "step":
            if message:
                parts.append(f" → {message}")
            else:
                parts.append(f" step at {at}")
            all_lines.append(" ".join(parts))
        
        elif kind == "pr":
            status = "✓" if ok else "✗"
            if url:
                parts.append(f" {status} PR: {url}")
            elif message:
                parts.append(f" {status} PR: {message}")
            else:
                parts.append(f" {status} PR created")
            all_lines.append(" ".join(parts))
        
        elif message:
            parts.append(f" {message}")
            all_lines.append(" ".join(parts))
    
    # Also include original logs for background details (deduplicated)
    for log_line in task.logs:
        stripped = log_line.strip()
        if stripped and log_line not in seen_logs:
            all_lines.append(log_line)
            seen_logs.add(log_line)
    
    return task.id, all_lines


def create_app(
    manager: TaskManager,
    *,
    default_workspace: Path,
    default_goal: str | None = None,
    default_create_pr: bool | None = None,
) -> FastAPI:
    app = FastAPI(title="loco", docs_url=None, redoc_url=None)
    app.state.ui = UiState(
        manager,
        default_workspace=default_workspace,
        default_goal=default_goal,
        default_create_pr=default_create_pr,
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        ui: UiState = request.app.state.ui
        return templates.TemplateResponse(request, "index.html", ui.template_vars())

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(FAVICON_SVG, media_type="image/svg+xml")

    @app.get("/api/meta")
    def meta(request: Request) -> dict[str, Any]:
        ui: UiState = request.app.state.ui
        return ui.meta()

    @app.get("/api/events/stream")
    async def events_stream(request: Request) -> StreamingResponse:
        """Stream the running task's agent and LLM log as server-sent events."""
        ui: UiState = request.app.state.ui
        manager = ui.manager

        async def event_stream():
            seen_id = ""
            seen_events = 0
            seen_logs = 0
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    task = _live_task(manager)
                    task_id = task.id if task else ""
                    if task_id != seen_id:
                        seen_id = task_id
                        seen_events = 0
                        seen_logs = 0
                    if task is not None:
                        events = list(task.events)
                        for event in events[seen_events:]:
                            yield f"data: {json.dumps(_console_event_payload(event))}\n\n"
                        seen_events = len(events)
                        logs = list(task.logs)
                        for line in logs[seen_logs:]:
                            yield f"data: {json.dumps({'line': line})}\n\n"
                        seen_logs = len(logs)
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(0.4)
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    def _models_payload(
        ui: UiState,
        base_url: str | None,
        api_key: str | None,
        *,
        remember: bool = False,
        model: str | None = None,
    ) -> Any:
        try:
            resolved = normalize_model_base_url(
                base_url or ui.manager.settings.model_base_url
            )
            names = ui.manager.list_models(base_url=resolved, api_key=api_key)
        except ValueError as exc:
            return JSONResponse({"error": str(exc), "models": []}, status_code=400)
        if remember:
            servers = ui.remember_server(resolved, model=model)
        else:
            servers = ui.known_servers()
        selected = ui.selection()
        return {
            "default": ui.manager.settings.model_name,
            "last_model": selected["last_model"],
            "base_url": resolved,
            "models": names,
            "servers": servers,
        }

    @app.get("/api/models", response_model=None)
    def models(
        request: Request,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> Any:
        return _models_payload(request.app.state.ui, base_url, api_key)

    @app.post("/api/models", response_model=None)
    def models_post(
        request: Request,
        payload: ModelsQuery | None = None,
    ) -> Any:
        body = payload or ModelsQuery()
        return _models_payload(
            request.app.state.ui,
            body.base_url,
            body.api_key,
            remember=True,
            model=body.model,
        )

    @app.get("/api/servers")
    def servers(request: Request) -> dict[str, Any]:
        ui: UiState = request.app.state.ui
        selected = ui.selection()
        return {
            "default": ui.manager.settings.model_base_url,
            "servers": selected["servers"],
            "last_base_url": selected["last_base_url"],
            "last_model": selected["last_model"],
        }

    @app.get("/api/goals", response_model=None)
    def goals_from_issues(
        request: Request,
        workspace: str | None = None,
        state: str = "open",
    ) -> Any:
        """Get selectable goals from the current workspace's GitHub issues."""
        ui: UiState = request.app.state.ui
        result = ui.load_goals_from_github(workspace, state=state)
        if result.get("error"):
            status = (
                400
                if result["error"] == "workspace is not a GitHub repository"
                else 502
            )
            return JSONResponse(result, status_code=status)
        return result


    @app.get("/api/tasks", response_model=None)
    def list_tasks(
        request: Request,
        page: int | None = None,
        page_size: int | None = None,
    ) -> Any:
        ui: UiState = request.app.state.ui
        tasks = ui.manager.list()
        if page is None and page_size is None:
            return [task.to_dict() for task in tasks]
        size = max(1, min(page_size or 10, 100))
        total = len(tasks)
        total_pages = max(1, math.ceil(total / size) if size else 1)
        current = max(1, min(page or 1, total_pages))
        start = (current - 1) * size
        return {
            "items": [task.to_dict() for task in tasks[start : start + size]],
            "page": current,
            "page_size": size,
            "total": total,
            "total_pages": total_pages,
        }

    @app.get("/api/tasks/{task_id}", response_model=None)
    def get_task(task_id: str, request: Request) -> Any:
        ui: UiState = request.app.state.ui
        task = ui.manager.get(task_id)
        if task is None:
            return JSONResponse({"error": "task not found"}, status_code=404)
        return task.to_dict()

    @app.post("/api/tasks/{task_id}/rerun")
    def rerun_task(task_id: str, request: Request, payload: RerunPayload | None = None) -> Any:
        ui: UiState = request.app.state.ui
        run_from_sha = payload.sha if payload else None
        new_goal = payload.goal if payload else None
        old_task = ui.manager.get(task_id)
        if old_task is None:
            return JSONResponse({"error": "task not found"}, status_code=404)
        if old_task.status not in ("failed", "error", "success"):
            return JSONResponse({"error": "task not rerunnable"}, status_code=400)
        if old_task.status == "success" and lookup_pull_state(old_task.pr_url or "") != "open":
            return JSONResponse({"error": "pull request is not open"}, status_code=409)
        goal_to_use = new_goal or old_task.goal
        new_task = ui.manager.rerun(
            task_id,
            from_sha=run_from_sha,
            goal=goal_to_use,
        )
        if new_task is None:
            return JSONResponse({"error": "task not found or not rerunnable"}, status_code=404)
        return JSONResponse(new_task.to_dict(), status_code=201)

    @app.post("/api/tasks")
    def create_task(
        request: Request,
        payload: TaskCreate | None = None,
    ) -> JSONResponse:
        ui: UiState = request.app.state.ui
        body = payload or TaskCreate()
        if body.pr_url and lookup_pull_state(body.pr_url) != "open":
            return JSONResponse({"error": "pull request is not open"}, status_code=409)
        workspace_raw = body.workspace or ui.default_workspace
        try:
            task = ui.manager.submit(
                Path(workspace_raw),
                body.goal,
                auto_commit=body.auto_commit,
                create_pr=body.create_pr,
                model_name=body.model,
                model_base_url=body.base_url,
                model_api_key=body.api_key,
                resume_branch=body.resume_branch,
                resume_sha=body.resume_sha,
            )
            ui.remember_server(task.model_base_url, model=task.model_name)
            ui.set_workspace(task.workspace)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(task.to_dict(), status_code=201)

    @app.post("/api/pulls/status")
    def pull_status(payload: PullStatusQuery | None = None) -> dict[str, Any]:
        body = payload or PullStatusQuery()
        states: dict[str, str] = {}
        for raw in body.urls[:20]:
            clean = extract_pr_url(raw or "") or ""
            if not clean or clean in states:
                continue
            states[clean] = lookup_pull_state(clean)
        return {"states": states}

    @app.get("/api/history")
    def get_history(
        request: Request,
        page: int = 1,
        page_size: int = 10,
        search: str | None = None,
    ) -> dict[str, Any]:
        ui: UiState = request.app.state.ui
        return ui.paginate_history(
            Path(ui.default_workspace),
            page=page,
            page_size=page_size,
            search=search,
        )

    @app.get("/api/ui-screenshot", response_model=None)
    def ui_screenshot(
        request: Request,
        name: str,
        workspace: str | None = None,
    ) -> Any:
        ui: UiState = request.app.state.ui
        root = Path(workspace or ui.default_workspace).expanduser()
        path = resolve_ui_screenshot(root, name)
        if path is None:
            return JSONResponse({"error": "screenshot not found"}, status_code=404)
        return FileResponse(path, media_type="image/png")

    @app.post("/api/servers/alias")
    def update_alias(
        request: Request,
        payload: ServerAliasUpdate | None = None,
    ) -> dict[str, Any]:
        """Update or add an alias for a server URL."""
        ui: UiState = request.app.state.ui
        body = payload or ServerAliasUpdate(url="", alias=None)
        if not body.url or not body.url.strip():
            return JSONResponse({"error": "url is required"}, status_code=400)
        servers = update_server_alias(
            Path(ui.default_workspace),
            body.url,
            body.alias,
        )
        return {
            "servers": servers,
            "last_base_url": ui.selection()["last_base_url"],
            "last_model": ui.selection()["last_model"],
        }

    @app.post("/api/servers")
    def create_server(
        request: Request,
        payload: UrlCreate | None = None,
    ) -> dict[str, Any]:
        """Create a new server with optional alias."""
        from agent_loco.runtime.servers import create_or_update_server

        ui: UiState = request.app.state.ui
        body = payload or UrlCreate(url="", alias=None)
        if not body.url or not body.url.strip():
            return JSONResponse({"error": "url is required"}, status_code=400)

        servers = create_or_update_server(
            Path(ui.default_workspace),
            body.url,
            body.alias or None,
        )
        return {
            "servers": servers,
            "last_base_url": ui.selection()["last_base_url"],
            "last_model": ui.selection()["last_model"],
        }

    def _workspace_payload(ui: UiState) -> dict[str, Any]:
        workspaces = load_workspaces(
            Path(ui.store_root), default=ui.default_workspace
        )
        current = next(
            (item for item in workspaces if item["path"] == ui.default_workspace),
            None,
        )
        return {
            "current": ui.default_workspace,
            "home": str(Path.home()),
            "default_guidelines": default_guidelines(),
            "guidelines": (current or {}).get("guidelines") or default_guidelines(),
            "custom_guidelines": bool((current or {}).get("custom_guidelines")),
            "workspaces": workspaces,
            "archived": load_archived_workspaces(Path(ui.store_root)),
        }

    def _seed_guidelines(path: str, text: str | None) -> None:
        if text is None:
            return
        root = Path(path)
        if guidelines_are_custom(root):
            return
        save_guidelines(root, text)

    @app.get("/api/workspaces")
    def list_workspaces(request: Request) -> dict[str, Any]:
        return _workspace_payload(request.app.state.ui)

    @app.get("/api/workspaces/browse")
    def browse_workspaces(path: str | None = None) -> Any:
        try:
            return browse_directory(path)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/workspaces/select")
    def select_workspace(request: Request, payload: WorkspaceSelect) -> Any:
        ui: UiState = request.app.state.ui
        try:
            _seed_guidelines(payload.path, payload.guidelines)
            ui.set_workspace(payload.path)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return _workspace_payload(ui)

    @app.post("/api/workspaces/create")
    def make_workspace(request: Request, payload: WorkspaceSelect) -> Any:
        ui: UiState = request.app.state.ui
        try:
            created = create_workspace(payload.path)
            _seed_guidelines(str(created["path"]), payload.guidelines)
            ui.set_workspace(str(created["path"]))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return {**_workspace_payload(ui), "created": created}

    @app.post("/api/workspaces/clone")
    def clone_repo(request: Request, payload: WorkspaceClone) -> Any:
        ui: UiState = request.app.state.ui
        parent = payload.parent or str(Path.home() / "projects")
        try:
            cloned = clone_workspace(payload.url, parent, name=payload.name)
            _seed_guidelines(str(cloned["path"]), payload.guidelines)
            ui.set_workspace(str(cloned["path"]))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return {**_workspace_payload(ui), "created": cloned}

    @app.put("/api/workspaces/guidelines")
    def update_workspace_guidelines(
        request: Request, payload: WorkspaceGuidelines | None = None
    ) -> Any:
        ui: UiState = request.app.state.ui
        body = payload or WorkspaceGuidelines()
        path = body.path or ui.default_workspace
        try:
            save_guidelines(Path(path), body.guidelines)
            ui.set_workspace(path)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return _workspace_payload(ui)

    def _workspace_root(ui: UiState, path: str | None) -> Path:
        return Path(path or ui.default_workspace).expanduser().resolve()

    @app.get("/api/workspaces/skills")
    def list_workspace_skills(request: Request, path: str | None = None) -> Any:
        ui: UiState = request.app.state.ui
        try:
            root = _workspace_root(ui, path)
            if not root.is_dir():
                raise ValueError(f"workspace is not a directory: {root}")
            return skills_catalog(root)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/workspaces/skills/detail")
    def workspace_skill_detail(
        request: Request, name: str, path: str | None = None
    ) -> Any:
        ui: UiState = request.app.state.ui
        try:
            root = _workspace_root(ui, path)
            if not root.is_dir():
                raise ValueError(f"workspace is not a directory: {root}")
            skill = get_skill(root, name)
            if skill is None:
                return JSONResponse({"error": f"unknown skill: {name}"}, status_code=404)
            return skill.public_dict(include_body=True)
        except ValueError as extra:
            return JSONResponse({"error": str(extra)}, status_code=400)

    @app.put("/api/workspaces/skills")
    def update_workspace_skills(
        request: Request, payload: WorkspaceSkillsUpdate | None = None
    ) -> Any:
        ui: UiState = request.app.state.ui
        body = payload or WorkspaceSkillsUpdate()
        try:
            root = _workspace_root(ui, body.path)
            if not root.is_dir():
                raise ValueError(f"workspace is not a directory: {root}")
            save_enabled_skills(root, body.enabled or [])
            return skills_catalog(root)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/workspaces/skills/sources")
    def add_workspace_skill_source(
        request: Request, payload: WorkspaceSkillSource
    ) -> Any:
        ui: UiState = request.app.state.ui
        try:
            root = _workspace_root(ui, payload.path)
            if not root.is_dir():
                raise ValueError(f"workspace is not a directory: {root}")
            source = add_skill_source(root, payload.url, ref=payload.ref)
            return {**skills_catalog(root), "added": source.public_dict()}
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/workspaces/skills/sources/sync")
    def sync_workspace_skill_source(
        request: Request, payload: WorkspaceSkillSync
    ) -> Any:
        ui: UiState = request.app.state.ui
        try:
            root = _workspace_root(ui, payload.path)
            if not root.is_dir():
                raise ValueError(f"workspace is not a directory: {root}")
            source = sync_skill_source(root, payload.slug, ref=payload.ref)
            return {**skills_catalog(root), "synced": source.public_dict()}
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/workspaces/archive")
    def archive_workspace_tab(request: Request, payload: WorkspaceSelect) -> Any:
        ui: UiState = request.app.state.ui
        try:
            ui.archive_workspace_path(payload.path)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return _workspace_payload(ui)

    @app.post("/api/workspaces/restore")
    def restore_workspace_tab(request: Request, payload: WorkspaceSelect) -> Any:
        ui: UiState = request.app.state.ui
        try:
            ui.set_workspace(payload.path)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return _workspace_payload(ui)

    @app.post("/api/workspaces/forget")
    def forget_workspace_tab(request: Request, payload: WorkspaceSelect) -> Any:
        ui: UiState = request.app.state.ui
        try:
            ui.forget_workspace_path(payload.path)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return _workspace_payload(ui)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def _created_at_from_run_id(stem: str) -> str | None:
    try:
        parsed = datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def serve(
    workspace: Path,
    settings: Settings,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    max_concurrent: int = 1,
    default_goal: str | None = None,
    default_create_pr: bool | None = None,
) -> None:
    manager = TaskManager(settings, max_concurrent=max_concurrent)
    app = create_app(
        manager,
        default_workspace=workspace,
        default_goal=default_goal,
        default_create_pr=default_create_pr,
    )
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_graceful_shutdown=1,
        access_log=False,
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        manager.shutdown(wait=False)

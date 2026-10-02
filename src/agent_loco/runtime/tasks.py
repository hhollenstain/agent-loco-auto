from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from agent_loco.config import Settings
from agent_loco.llm.client import (
    LLMClient,
    OpenAICompatClient,
    list_remote_models,
    lookup_context_window,
    normalize_model_base_url,
)
from agent_loco.logging import UtcFormatter, utcnow_iso
from agent_loco.progress import (
    bind_progress,
    public_event,
    record_event,
    reset_progress,
    token_usage_from_events,
)
from agent_loco.runtime.improve import CycleResult, run_cycle

log = logging.getLogger("loco")

Runner = Callable[["Task"], CycleResult]


def _utcnow() -> str:
    return utcnow_iso()


def _default_llm(settings: Settings) -> LLMClient:
    return OpenAICompatClient(
        model=settings.model_name,
        base_url=settings.model_base_url,
        api_key=settings.model_api_key,
    )


@dataclass
class Task:
    id: str
    workspace: str
    goal: str | None
    auto_commit: bool
    create_pr: bool | None
    model_name: str
    model_base_url: str
    model_api_key: str
    status: str = "queued"
    created_at: str = field(default_factory=_utcnow)
    started_at: str | None = None
    finished_at: str | None = None
    logs: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    summary: str | None = None
    reason: str | None = None
    tests_passed: bool | None = None
    committed: bool = False
    published: bool = False
    commit_sha: str | None = None
    pr_url: str | None = None
    error: str | None = None
    sha_before: str | None = None
    branch: str | None = None
    resume_branch: str | None = None
    resume_sha: str | None = None
    context_window: int | None = None

    def to_dict(self, *, include_logs: bool = True) -> dict:
        payload = {
            "id": self.id,
            "workspace": self.workspace,
            "goal": self.goal,
            "model": self.model_name,
            "base_url": self.model_base_url,
            "auto_commit": self.auto_commit,
            "create_pr": self.create_pr,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "summary": self.summary,
            "reason": self.reason,
            "tests_passed": self.tests_passed,
            "committed": self.committed,
            "published": self.published,
            "commit_sha": self.commit_sha,
            "pr_url": self.pr_url,
            "error": self.error,
            "sha_before": self.sha_before,
            "branch": self.branch,
            "events": [public_event(event) for event in self.events],
            **token_usage_from_events(self.events),
            "context_window": self.context_window,
        }
        if include_logs:
            payload["logs"] = list(self.logs)
        return payload


class _TaskLogHandler(logging.Handler):
    """Capture `loco` log lines from the worker thread that owns a task."""

    def __init__(self, task: Task, thread_id: int) -> None:
        super().__init__(level=logging.INFO)
        self._task = task
        self._thread_id = thread_id
        self.setFormatter(UtcFormatter("%(asctime)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if threading.get_ident() != self._thread_id:
            return
        try:
            self._task.logs.append(self.format(record))
        except Exception:
            self.handleError(record)


class TaskManager:
    """Queue coding-agent cycles and run a bounded number of them at once."""

    def __init__(
        self,
        settings: Settings,
        *,
        max_concurrent: int = 1,
        runner: Runner | None = None,
        llm_factory: Callable[[Settings], LLMClient] | None = None,
        models_fn: Callable[..., list[str]] | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.settings = settings
        self.max_concurrent = max_concurrent
        self._runner = runner
        self._llm_factory = llm_factory or _default_llm
        self._models_fn = models_fn
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="loco-task",
        )

    def submit(
        self,
        workspace: Path,
        goal: str | None,
        *,
        auto_commit: bool | None = None,
        create_pr: bool | None = None,
        model_name: str | None = None,
        model_base_url: str | None = None,
        model_api_key: str | None = None,
        resume_branch: str | None = None,
        resume_sha: str | None = None,
    ) -> Task:
        workspace = workspace.expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace}")
        selected_model = (model_name or "").strip() or self.settings.model_name
        selected_url = normalize_model_base_url(
            (model_base_url or "").strip() or self.settings.model_base_url
        )
        selected_key = (
            model_api_key if model_api_key is not None else self.settings.model_api_key
        )
        task = Task(
            id=uuid4().hex,
            workspace=str(workspace),
            goal=(goal.strip() if goal and goal.strip() else None),
            auto_commit=self.settings.auto_commit if auto_commit is None else auto_commit,
            create_pr=create_pr,
            model_name=selected_model,
            model_base_url=selected_url,
            model_api_key=selected_key,
            resume_branch=(resume_branch or "").strip() or None,
            resume_sha=(resume_sha or "").strip() or None,
            branch=(resume_branch or "").strip() or None,
        )
        with self._lock:
            self._tasks[task.id] = task
        self._executor.submit(self._run, task)
        return task

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> list[Task]:
        with self._lock:
            tasks = list(self._tasks.values())
        tasks.sort(key=lambda item: item.created_at, reverse=True)
        return tasks

    def list_for_workspace(self, workspace: str) -> list[Task]:
        """Return tasks for a specific workspace, sorted newest first."""
        workspace = workspace.expanduser().resolve() if workspace else ""
        with self._lock:
            tasks = [
                task
                for task in self._tasks.values()
                if Path(task.workspace).resolve() == Path(workspace).resolve()
            ]
        tasks.sort(key=lambda item: item.created_at, reverse=True)
        return tasks

    def has_running_task(self, workspace: str, model_name: str, model_base_url: str) -> bool:
        """Check if there's already a running task for the same LLM server on this workspace."""
        workspace_path = workspace.expanduser().resolve() if workspace else ""
        with self._lock:
            for task in self._tasks.values():
                if Path(task.workspace).resolve() != Path(workspace_path).resolve():
                    continue
                if task.status != "running":
                    continue
                if task.model_name == model_name and task.model_base_url == model_base_url:
                    return True
        return False

    def counts(self) -> dict[str, int]:
        with self._lock:
            statuses = [task.status for task in self._tasks.values()]
        return {
            "queued": statuses.count("queued"),
            "running": statuses.count("running"),
            "total": len(statuses),
        }

    def rerun(
        self,
        task_id: str,
        from_sha: str | None = None,
        goal: str | None = None,
    ) -> Task | None:
        """Queue a new cycle on the task's branch, with an optional revised goal."""
        with self._lock:
            old_task = self._tasks.get(task_id)
        if not old_task:
            return None
        if old_task.status not in ("failed", "error", "success"):
            return None
        resume_branch = old_task.branch
        resume_sha = (from_sha or old_task.commit_sha or old_task.sha_before or "").strip() or None
        revised = (goal or "").strip()
        create_pr = True if old_task.pr_url else old_task.create_pr
        return self.submit(
            Path(old_task.workspace),
            revised or old_task.goal,
            auto_commit=old_task.auto_commit,
            create_pr=create_pr,
            model_name=old_task.model_name,
            model_base_url=old_task.model_base_url,
            model_api_key=old_task.model_api_key,
            resume_branch=resume_branch,
            resume_sha=resume_sha,
        )

    def list_models(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        url = normalize_model_base_url(base_url or self.settings.model_base_url)
        key = self.settings.model_api_key if not api_key else api_key
        if self._models_fn is not None:
            try:
                names = list(self._models_fn(url, key))
            except TypeError:
                names = list(self._models_fn())
        else:
            names = list_remote_models(url, key)
        default_url = normalize_model_base_url(self.settings.model_base_url)
        default = self.settings.model_name
        if url == default_url and default and default not in names:
            names.insert(0, default)
        return names

    def shutdown(self, wait: bool = False, *, cancel_futures: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _run(self, task: Task) -> None:
        task.status = "running"
        task.started_at = _utcnow()
        if self._runner is None and task.context_window is None:
            task.context_window = lookup_context_window(
                task.model_base_url,
                task.model_api_key,
                task.model_name,
            )
        handler = _TaskLogHandler(task, threading.get_ident())
        loco_log = logging.getLogger("loco")
        loco_log.setLevel(logging.INFO)
        loco_log.addHandler(handler)
        exchange_logs = [logging.getLogger(name) for name in ("httpx", "openai")]
        for exchange_log in exchange_logs:
            exchange_log.addHandler(handler)
        progress = bind_progress(task.events)
        log.info("task %s model=%s url=%s", task.id, task.model_name, task.model_base_url)
        try:
            result = self._execute(task)
            task.summary = result.summary
            task.reason = result.reason
            task.tests_passed = result.tests_passed
            task.committed = result.committed
            task.published = result.published
            task.commit_sha = result.commit_sha
            task.pr_url = result.pr_url
            if getattr(result, "branch", None):
                task.branch = result.branch
            self._refresh_git_state(task)
            task.finished_at = _utcnow()
            task.status = result.status
        except Exception as exc:
            log.exception("task %s crashed", task.id)
            task.error = str(exc)
            task.reason = f"task crashed: {exc}"
            task.finished_at = _utcnow()
            task.status = "error"
        finally:
            reset_progress(progress)
            loco_log.removeHandler(handler)
            for exchange_log in exchange_logs:
                exchange_log.removeHandler(handler)

    def _execute(self, task: Task) -> CycleResult:
        self._resume_workspace(task)
        if self._runner is not None:
            return self._runner(task)
        settings = self.settings.model_copy(
            update={
                "auto_commit": task.auto_commit,
                "model_name": task.model_name,
                "model_base_url": task.model_base_url,
                "model_api_key": task.model_api_key,
            }
        )
        llm = self._llm_factory(settings)
        return run_cycle(
            Path(task.workspace),
            settings,
            llm,
            task.goal,
            cli_create_pr=task.create_pr,
        )

    def _resume_workspace(self, task: Task) -> None:
        from agent_loco.sandbox import SandboxError, Workspace
        from agent_loco.tools.git import current_branch, current_sha, resume_workspace

        try:
            workspace = Workspace(Path(task.workspace))
        except (OSError, SandboxError):
            return
        if not task.sha_before:
            task.sha_before = current_sha(workspace)
        if not task.branch:
            task.branch = current_branch(workspace)
        branch = task.resume_branch
        sha = task.resume_sha
        if not branch and not sha:
            return
        result = resume_workspace(workspace, branch=branch, sha=sha)
        if result.ok:
            task.branch = result.output or branch or task.branch
            log.info("resumed task on %s", task.branch)
            record_event(
                kind="step",
                message=f"Resuming on branch {task.branch}",
            )
        else:
            log.warning("could not resume %s: %s", branch or sha, result.output)
            record_event(
                kind="step",
                message=f"Could not checkout {branch or sha}; continuing on the current tree",
            )

    def _refresh_git_state(self, task: Task) -> None:
        from agent_loco.sandbox import SandboxError, Workspace
        from agent_loco.tools.git import current_branch, current_sha

        try:
            workspace = Workspace(Path(task.workspace))
        except (OSError, SandboxError):
            return
        task.branch = current_branch(workspace) or task.branch
        if not task.commit_sha:
            task.commit_sha = current_sha(workspace)

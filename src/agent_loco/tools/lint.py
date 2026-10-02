from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from agent_loco.progress import record_lint_run
from agent_loco.sandbox import Workspace
from agent_loco.tools.base import ToolResult, ToolSpec, object_schema
from agent_loco.tools.files import SKIP_DIR_NAMES
from agent_loco.tools.git import is_runtime_artifact
from agent_loco.tools.shell import run_command


@dataclass
class _CachedRun:
    fingerprint: str
    result: ToolResult


_CACHE: dict[str, _CachedRun] = {}


def lint_tools(
    workspace: Workspace,
    lint_command: str | None,
    timeout_seconds: int,
) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="run_lint",
            description=(
                "Run the project's configured linter (for example ruff format "
                "then ruff check). Call this after edits and before considering "
                "a pull request done. If the workspace tree has not changed "
                "since the last run, returns that result without running the "
                "linter again."
            ),
            parameters=object_schema({}),
            handler=lambda: run_project_lint(
                workspace, lint_command, timeout_seconds, phase="agent"
            ),
        )
    ]


def run_project_lint(
    workspace: Workspace,
    lint_command: str | None,
    timeout_seconds: int,
    *,
    phase: str = "lint",
) -> ToolResult:
    if not lint_command:
        result = ToolResult(False, "no lint command configured for this project")
        record_lint_run(command="", ok=False, output=result.output, phase=phase)
        return result
    fingerprint = _tree_fingerprint(workspace, lint_command)
    cached = _CACHE.get(str(workspace.root))
    if cached is not None and cached.fingerprint == fingerprint:
        record_lint_run(
            command=lint_command,
            ok=cached.result.ok,
            output=cached.result.output,
            phase=phase,
            elapsed_ms=0,
            reused=True,
        )
        return cached.result
    started = time.perf_counter()
    result = run_command(workspace, lint_command, timeout_seconds)
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    record_lint_run(
        command=lint_command,
        ok=result.ok,
        output=result.output,
        phase=phase,
        elapsed_ms=elapsed_ms,
    )
    _CACHE[str(workspace.root)] = _CachedRun(fingerprint=fingerprint, result=result)
    return result


def _tree_fingerprint(workspace: Workspace, lint_command: str) -> str:
    digest = hashlib.sha256()
    digest.update(lint_command.encode("utf-8"))
    root = workspace.root
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        rel = path.relative_to(root).as_posix()
        if is_runtime_artifact(rel):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}\n".encode())
    return digest.hexdigest()

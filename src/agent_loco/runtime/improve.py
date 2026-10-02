from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agent_loco.agent.loop import CodingAgent
from agent_loco.config import Settings
from agent_loco.llm.client import LLMClient
from agent_loco.progress import bind_progress, current_events, record_event, reset_progress
from agent_loco.runtime.importer import goal_headline
from agent_loco.runtime.project import (
    ProjectConfig,
    collect_context,
    ensure_run_gitignore,
    load_goals,
    load_project,
    mark_goal_done,
)
from agent_loco.runtime.review import (
    GoalReview,
    half_baked_diff_markers,
    incomplete_agent_run,
    invalid_doc_commands,
    is_open_pr_goal,
    review_goal,
    review_reason,
    unused_new_symbols,
    unwired_ui_markers,
)
from agent_loco.runtime.skills import compose_system_prompt
from agent_loco.runtime.uireview import (
    UiEvidence,
    collect_ui_evidence,
    format_ui_evidence,
    resolve_ui_screenshot,
    ui_review_needed,
    unverified_interactive_ui,
)
from agent_loco.runtime.workdiff import collect_current_evidence, collect_work_diff
from agent_loco.sandbox import SandboxError, Workspace
from agent_loco.tools import build_tools
from agent_loco.tools.git import (
    CO_AUTHORED_BY,
    commit_changes,
    commits_ahead_of_base,
    create_pull_request,
    current_branch,
    current_sha,
    default_base_branch,
    ensure_pr_branch,
    existing_pull_request,
    extract_pr_url,
    github_owner_repo,
    has_changes,
    is_protected_branch,
    is_tracked,
    push_changes,
    update_pull_request,
    upstream_state,
)
from agent_loco.tools.lint import run_project_lint
from agent_loco.tools.tests import run_project_tests

log = logging.getLogger("loco")


def log_progress(message: str) -> None:
    record_event(kind="step", message=message)
    log.info("%s", message)


def compact_handoff(
    *,
    goal: str,
    summary: str,
    diff: str,
    last_tool_error: str | None,
    stopped_reason: str,
) -> str:
    """Build a compact handoff brief for the agent after hitting max_iterations.

    Includes goal, changed file paths (not full hunks), last tool error, and
    a prompt to finish remaining work. Cap at 4000 characters.
    """
    # Extract file paths from diff
    paths: list[str] = []
    if diff:
        for line in diff.splitlines():
            # Match diff headers like "diff --git a/... b/file" or "+++ b/file"
            if line.startswith("diff --git "):
                parts = line.split(" b/")
                if len(parts) >= 2:
                    paths.append(parts[1].strip())
            elif line.startswith("+++ b/"):
                path = line.split(" b/")[1].strip()
                if path and path != "/dev/null":
                    paths.append(path)
    # Limit paths to 20
    paths = paths[:20]
    # Build handoff message
    lines = [
        "COMPACT HANDOFF",
        f"Goal: {goal.strip()}",
        f"Previous run summary: {summary.strip() if summary else 'None'}",
        f"Stopped: {stopped_reason}",
    ]
    if paths:
        lines.append(f"Changed files ({len(paths)}):")
        for p in paths:
            lines.append(f"  - {p}")
    if last_tool_error:
        lines.append(f"Last tool error: {last_tool_error[:200]}")
    lines.append(
        "\nFinish the remaining work. Do not re-read the whole repo.\n"
        "Only edit files that implement the goal."
    )
    result = "\n".join(lines)
    if len(result) > 3997:
        result = result[:3997]
    return result


def _compute_goal_key(goal: str) -> str:
    """Compute a 16-char hex prefix of sha256 of stripped goal."""
    stripped = goal.strip()
    digest = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
    return digest[:16]


def _write_last_failure(root: Path, goal: str, result: CycleResult) -> None:
    """Write last-failure.json after a failed cycle.

    Does not fail the cycle if JSON write fails; logs and continues.
    """
    LAST_FAILURE_FILE = ".loco/last-failure.json"
    try:
        failure_path = root / LAST_FAILURE_FILE
        # Ensure .loco directory exists before writing the file
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        summary = (result.summary or "")[:500] if result.summary else ""
        data = {
            "goal": goal,
            "goal_key": _compute_goal_key(goal),
            "status": result.status,
            "reason": result.reason or "unknown",
            "summary": summary,
            "finished_at": result.created_at,
        }
        failure_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log_progress("Remembered last failure for the next run.")
    except (OSError, TypeError) as e:
        log.warning(f"Failed to write last-failure.json: {e}")
        # Do not fail the cycle; just log and continue


def _delete_last_failure(root: Path) -> None:
    """Delete last-failure.json if it exists.

    Does not fail if deletion fails; logs and continues.
    """
    LAST_FAILURE_FILE = ".loco/last-failure.json"
    try:
        failure_path = root / LAST_FAILURE_FILE
        if failure_path.exists():
            failure_path.unlink()
            log.debug("Deleted last-failure.json after successful cycle.")
    except OSError as e:
        log.warning(f"Failed to delete last-failure.json: {e}")


@dataclass
class CycleResult:
    status: str
    goal: str | None
    summary: str | None
    tests_passed: bool | None
    committed: bool
    published: bool
    commit_sha: str | None
    reason: str | None
    pr_url: str | None = None
    branch: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    events: list[dict] = field(default_factory=list)


@dataclass
class _EmptyDiffOutcome:
    result: CycleResult | None
    summary: str
    tests_passed: bool | None


def resolve_create_pr(
    settings: Settings,
    project: ProjectConfig,
    cli_create_pr: bool | None = None,
) -> bool:
    """`--no-create-pr` always wins. Otherwise CLI, settings, or a git repo can enable it."""
    if cli_create_pr is False:
        return False
    if cli_create_pr is True:
        return True
    return bool(settings.create_pr or project.publish_enabled)


def run_cycle(
    workspace_path: Path,
    settings: Settings,
    llm: LLMClient,
    goal: str | None = None,
    *,
    mark_checkbox: bool = True,
    cli_create_pr: bool | None = None,
) -> CycleResult:
    progress = bind_progress()
    try:
        return _run_cycle(
            workspace_path,
            settings,
            llm,
            goal,
            mark_checkbox=mark_checkbox,
            cli_create_pr=cli_create_pr,
        )
    finally:
        reset_progress(progress)


def _run_cycle(
    workspace_path: Path,
    settings: Settings,
    llm: LLMClient,
    goal: str | None = None,
    *,
    mark_checkbox: bool = True,
    cli_create_pr: bool | None = None,
) -> CycleResult:
    log_progress("Initializing workspace...")
    workspace = Workspace(workspace_path)
    ensure_run_gitignore(workspace.root)
    project = load_project(workspace.root)
    allow_create_pr = resolve_create_pr(settings, project, cli_create_pr)
    tests_before = None
    if goal is None:
        log_progress("Running before tests...")
        tests_before = _maybe_test(workspace, project, settings, phase="before")
    selected_goal = goal or _choose_goal(project, tests_before)
    if not selected_goal:
        log_progress("No pending goals and tests are green.")
        result = CycleResult(
            status="skipped",
            goal=None,
            summary=None,
            tests_passed=None if tests_before is None else tests_before.ok,
            committed=False,
            published=False,
            commit_sha=None,
            reason="no open goals and tests are green",
        )
        _write_run_log(workspace.root, result)
        _append_to_history(workspace.root, result)
        return result

    tools = build_tools(
        workspace,
        test_command=project.test_command,
        lint_command=project.lint_command,
        command_timeout_seconds=settings.command_timeout_seconds,
        git_author_name=settings.git_author_name,
        git_author_email=settings.git_author_email,
        allow_publish=False,
        goal=selected_goal,
    )

    log_progress(f"Selected goal: {goal_headline(selected_goal)}")
    log_progress("Fetching current SHA...")
    sha_before = current_sha(workspace)
    log_progress("Initializing coding agent...")
    agent = CodingAgent(
        llm,
        tools,
        max_iterations=settings.max_iterations,
        system_prompt=compose_system_prompt(workspace.root),
    )
    log_progress("Running coding agent...")
    agent_result = agent.run(
        selected_goal,
        collect_context(workspace.root, project, allow_publish=allow_create_pr),
    )

    # Check for compact handoff if max_iterations was hit after changes
    sha = current_sha(workspace)
    has_work = has_changes(workspace) or bool(sha_before and sha and sha != sha_before)
    if agent_result.stopped_reason == "max_iterations" and has_work:
        log_progress("Iteration limit reached; continuing once from a compact handoff.")
        work_diff = collect_work_diff(workspace, sha_before, goal=selected_goal)
        handoff_prompt = compact_handoff(
            goal=selected_goal,
            summary=agent_result.summary,
            diff=work_diff,
            last_tool_error=agent_result.last_error,
            stopped_reason=agent_result.stopped_reason,
        )
        agent_result_2 = agent.run(
            handoff_prompt,
            collect_context(workspace.root, project, allow_publish=allow_create_pr),
            require_change=True,
        )
        agent_result = agent_result_2

    log_progress("Running after tests...")
    tests_after = _maybe_test(workspace, project, settings, phase="after")
    tests_after = _repair_failing_tests(
        workspace,
        project,
        settings,
        agent,
        tests_after,
        allow_create_pr=allow_create_pr,
    )

    tests_passed = None if tests_after is None else tests_after.ok
    if settings.require_tests and tests_after is not None and not tests_after.ok:
        log_progress("Tests failed after repairs.")
        result = CycleResult(
            status="failed",
            goal=selected_goal,
            summary=agent_result.summary,
            tests_passed=False,
            committed=False,
            published=False,
            commit_sha=None,
            reason="tests failed; commit skipped",
        )
        _write_run_log(workspace.root, result)
        _append_to_history(workspace.root, result)
        _write_last_failure(workspace.root, selected_goal, result)
        return result

    log_progress("Checking for changes...")
    sha = current_sha(workspace)
    has_work = has_changes(workspace) or bool(sha_before and sha and sha != sha_before)
    if not has_work:
        empty = _handle_empty_diff(
            workspace,
            project,
            settings,
            llm,
            agent,
            selected_goal,
            agent_result.summary,
            tests_passed,
            sha,
            sha_before=sha_before,
            mark_checkbox=mark_checkbox,
            allow_create_pr=allow_create_pr,
        )
        if empty.result is not None:
            return empty.result
        agent_result.summary = empty.summary
        tests_passed = empty.tests_passed
        sha = current_sha(workspace)

    review = _ensure_goal_met(
        workspace,
        settings,
        project,
        llm,
        agent,
        selected_goal,
        agent_result.summary,
        sha_before,
        tests_passed,
        allow_create_pr,
        stopped_reason=agent_result.stopped_reason,
    )
    tests_passed = review["tests_passed"]
    agent_result_summary = review["summary"]
    if not review["met"]:
        log_progress(f"Goal not met: {review['reason']}")
        stopped_reason = agent_result.stopped_reason
        if stopped_reason == "max_iterations":
            reason_info = "iteration limit"
        else:
            reason_info = review.get("reason", "goal not met")
        reason = (
            "tests failed; commit skipped"
            if review.get("tests_failed")
            else f"goal not met; PR skipped: {reason_info}"
        )
        result = CycleResult(
            status="failed",
            goal=selected_goal,
            summary=agent_result_summary,
            tests_passed=tests_passed,
            committed=False,
            published=False,
            commit_sha=current_sha(workspace),
            reason=reason,
        )
        _write_run_log(workspace.root, result)
        _append_to_history(workspace.root, result)
        _write_last_failure(workspace.root, selected_goal, result)
        return result
    agent_result.summary = agent_result_summary

    log_progress("Running linter before publish...")
    lint_after = _ensure_lint(
        workspace,
        project,
        settings,
        agent,
        allow_create_pr=allow_create_pr,
    )
    if lint_after is not None and not lint_after.ok:
        log_progress("Lint failed after repairs.")
        result = CycleResult(
            status="failed",
            goal=selected_goal,
            summary=agent_result.summary,
            tests_passed=tests_passed,
            committed=False,
            published=False,
            commit_sha=current_sha(workspace),
            reason="lint failed; commit skipped",
        )
        _write_run_log(workspace.root, result)
        _append_to_history(workspace.root, result)
        _write_last_failure(workspace.root, selected_goal, result)
        return result

    log_progress("Goal confirmed; checking for publishable changes...")
    committed = False
    published = False
    pr_url = None
    sha = current_sha(workspace)
    if sha_before and sha and sha != sha_before:
        committed = True

    screenshot_files = _pr_screenshot_files(workspace) if allow_create_pr else []
    will_commit = settings.auto_commit and has_changes(workspace)
    if allow_create_pr and (committed or will_commit):
        log_progress("Moving work onto a pull-request branch...")
        branched = ensure_pr_branch(workspace, selected_goal, sha_before)
        if not branched.ok:
            log_progress(f"Could not create PR branch: {branched.output}")
            result = CycleResult(
                status="failed",
                goal=selected_goal,
                summary=agent_result.summary,
                tests_passed=tests_passed,
                committed=False,
                published=False,
                commit_sha=sha,
                reason=f"could not leave main: {branched.output}",
            )
            _write_run_log(workspace.root, result)
            _append_to_history(workspace.root, result)
            return result
        log_progress(f"On branch {branched.output}")

    if will_commit:
        log_progress("Committing changes...")
        message = _commit_message(selected_goal, agent_result.summary)
        commit = commit_changes(
            workspace,
            message,
            env=_git_env(settings),
        )
        if not commit.ok:
            log_progress("Commit failed.")
            result = CycleResult(
                status="failed",
                goal=selected_goal,
                summary=agent_result.summary,
                tests_passed=tests_passed,
                committed=False,
                published=False,
                commit_sha=None,
                reason=f"commit failed: {commit.output}",
            )
            _write_run_log(workspace.root, result)
            _append_to_history(workspace.root, result)
            return result
        committed = True
        sha = current_sha(workspace)
        log_progress(f"Committed SHA: {sha}")

    if allow_create_pr:
        published, pr_url, failed = _publish_pull_request(
            workspace,
            project,
            selected_goal,
            agent_result.summary,
            tests_passed=tests_passed,
            sha=sha,
            committed=committed,
            screenshot_files=screenshot_files,
        )
        if failed is not None:
            return failed

    if committed and mark_checkbox:
        log_progress("Marking goal as done...")
        mark_goal_done(workspace.root, project.goals_file, selected_goal)

    log_progress("Generating cycle result...")
    result = CycleResult(
        status="success",
        goal=selected_goal,
        summary=agent_result.summary,
        tests_passed=tests_passed,
        committed=committed,
        published=published,
        commit_sha=sha,
        reason=agent_result.stopped_reason,
        pr_url=pr_url,
    )
    _write_run_log(workspace.root, result)
    _append_to_history(workspace.root, result)
    _delete_last_failure(workspace.root)
    return result


def _publish_pull_request(
    workspace: Workspace,
    project: ProjectConfig,
    goal: str,
    summary: str,
    *,
    tests_passed: bool | None,
    sha: str | None,
    committed: bool,
    screenshot_files: list[Path],
) -> tuple[bool, str | None, CycleResult | None]:
    """Push the current feature branch and open a PR if this branch does not already have one."""
    branch = current_branch(workspace)
    if is_protected_branch(branch):
        if committed:
            log_progress(f"Refusing to publish {branch} directly.")
            result = CycleResult(
                status="failed",
                goal=goal,
                summary=summary,
                tests_passed=tests_passed,
                committed=committed,
                published=False,
                commit_sha=sha,
                reason=f"refusing to push commits on {branch}",
            )
            _write_run_log(workspace.root, result)
            _append_to_history(workspace.root, result)
            return False, None, result
        return False, None, None
    existing = existing_pull_request(workspace)
    base = default_base_branch(workspace, project.publish_branch)
    if not committed and commits_ahead_of_base(workspace, base) <= 0:
        if existing:
            log_progress(f"Branch already has pull request {existing}")
            record_event(kind="pr", url=existing, message="existing pull request")
            return True, existing, None
        return False, None, None
    log_progress(f"Pushing branch {branch}...")
    pushed = push_changes(workspace, project.publish_remote, branch)
    if not pushed.ok:
        log_progress(f"Push failed: {pushed.output}")
        result = CycleResult(
            status="failed",
            goal=goal,
            summary=summary,
            tests_passed=tests_passed,
            committed=committed,
            published=False,
            commit_sha=sha,
            reason=f"push failed: {pushed.output}",
        )
        _write_run_log(workspace.root, result)
        _append_to_history(workspace.root, result)
        return False, None, result
    if existing:
        log_progress(f"Updating pull request {existing}")
        updated = update_pull_request(
            workspace,
            _commit_message(goal, summary),
            _pr_body(
                goal,
                summary,
                tests_passed=tests_passed,
                commit_sha=sha,
                branch=branch or "",
                screenshots=[path.name for path in screenshot_files if is_tracked(workspace, path)],
                image_base=_pr_image_base(
                    workspace,
                    sha,
                    project.publish_remote,
                ),
            ),
        )
        if updated.ok:
            log_progress(f"Updated pull request {existing}")
        else:
            log_progress(
                f"Pushed the branch; pull request text was not refreshed: {updated.output}"
            )
        record_event(kind="pr", url=existing, message=updated.output or "existing pull request")
        return True, existing, None
    log_progress("Opening pull request...")
    pr = create_pull_request(
        workspace,
        _commit_message(goal, summary),
        _pr_body(
            goal,
            summary,
            tests_passed=tests_passed,
            commit_sha=sha,
            branch=branch or "",
            screenshots=[path.name for path in screenshot_files if is_tracked(workspace, path)],
            image_base=_pr_image_base(
                workspace,
                sha,
                project.publish_remote,
            ),
        ),
        base=base,
    )
    pr_url = extract_pr_url(pr.output)
    if not pr.ok:
        log_progress(f"PR creation failed: {pr.output}")
        result = CycleResult(
            status="failed",
            goal=goal,
            summary=summary,
            tests_passed=tests_passed,
            committed=committed,
            published=False,
            commit_sha=sha,
            reason=f"PR failed: {pr.output}",
        )
        _write_run_log(workspace.root, result)
        _append_to_history(workspace.root, result)
        return False, None, result
    log_progress(pr.output)
    if pr_url:
        record_event(kind="pr", url=pr_url, message=pr.output)
    return True, pr_url, None


def _handle_empty_diff(
    workspace: Workspace,
    project: ProjectConfig,
    settings: Settings,
    llm: LLMClient,
    agent: CodingAgent,
    goal: str,
    summary: str,
    tests_passed: bool | None,
    sha: str | None,
    *,
    sha_before: str | None,
    mark_checkbox: bool,
    allow_create_pr: bool,
) -> _EmptyDiffOutcome:
    log_progress("No project files changed; checking whether the goal is already met...")
    upstream = upstream_state(workspace, project.publish_remote)
    log_progress(f"Upstream: {upstream.detail}")
    evidence = collect_current_evidence(workspace, goal)
    verdict = _review_goal(
        workspace,
        project,
        llm,
        goal,
        evidence,
        summary,
        tests_passed,
        existing=True,
        upstream=upstream.detail,
    )
    log_progress(f"Goal review: met={verdict.met} ({review_reason(verdict)})")
    if verdict.met or is_open_pr_goal(goal):
        published = False
        pr_url = None
        if allow_create_pr:
            published, pr_url, failed = _publish_pull_request(
                workspace,
                project,
                goal,
                summary,
                tests_passed=tests_passed,
                sha=sha,
                committed=False,
                screenshot_files=[],
            )
            if failed is not None:
                return _EmptyDiffOutcome(result=failed, summary=summary, tests_passed=tests_passed)
        if verdict.met or published:
            log_progress(f"No work needed: {review_reason(verdict)}")
            if mark_checkbox:
                log_progress("Marking goal as done...")
                mark_goal_done(workspace.root, project.goals_file, goal)
            if published and pr_url:
                reason = f"opened pull request without new files: {pr_url}"
                cycle_summary = (
                    f"Opened pull request {pr_url} from the current branch "
                    "without new file changes."
                )
            else:
                reason = f"no changes needed: {review_reason(verdict)} {upstream.detail}"
                cycle_summary = _no_work_summary(summary, verdict, upstream.detail)
            result = CycleResult(
                status="success",
                goal=goal,
                summary=cycle_summary,
                tests_passed=tests_passed,
                committed=False,
                published=published,
                commit_sha=sha,
                reason=reason,
                pr_url=pr_url,
            )
            _write_run_log(workspace.root, result)
            _append_to_history(workspace.root, result)
            return _EmptyDiffOutcome(result=result, summary=summary, tests_passed=tests_passed)

    context = collect_context(workspace.root, project, allow_publish=allow_create_pr)
    for attempt in range(project.max_repair_attempts):
        log_progress(
            "Agent finished without edits; sending it back to implement "
            f"({attempt + 1}/{project.max_repair_attempts})"
        )
        follow = agent.run(
            _empty_diff_retry_prompt(goal, review_reason(verdict)),
            context,
            require_change=True,
        )
        if follow.summary:
            summary = follow.summary
        tests_after = _maybe_test(workspace, project, settings, phase="retry")
        tests_after = _repair_failing_tests(
            workspace,
            project,
            settings,
            agent,
            tests_after,
            allow_create_pr=allow_create_pr,
        )
        tests_passed = None if tests_after is None else tests_after.ok
        if settings.require_tests and tests_after is not None and not tests_after.ok:
            result = CycleResult(
                status="failed",
                goal=goal,
                summary=summary,
                tests_passed=False,
                committed=False,
                published=False,
                commit_sha=current_sha(workspace),
                reason="tests failed after empty-diff retry; commit skipped",
            )
            _write_run_log(workspace.root, result)
            _append_to_history(workspace.root, result)
            return _EmptyDiffOutcome(result=result, summary=summary, tests_passed=False)
        sha_now = current_sha(workspace)
        if has_changes(workspace) or bool(sha_before and sha_now and sha_now != sha_before):
            log_progress("Agent produced file changes on retry.")
            return _EmptyDiffOutcome(result=None, summary=summary, tests_passed=tests_passed)

    log_progress("Goal is not already met; no files were changed.")
    result = CycleResult(
        status="skipped",
        goal=goal,
        summary=_no_work_summary(summary, verdict, upstream.detail),
        tests_passed=tests_passed,
        committed=False,
        published=False,
        commit_sha=sha,
        reason=(
            "no files changed and the goal is not already met: "
            f"{review_reason(verdict)} {upstream.detail}"
        ),
    )
    _write_run_log(workspace.root, result)
    _append_to_history(workspace.root, result)
    return _EmptyDiffOutcome(result=result, summary=summary, tests_passed=tests_passed)


def _empty_diff_retry_prompt(goal: str, reason: str) -> str:
    return (
        "You inspected the repo but did not change any files that implement the goal. "
        "The goal is not already done. Do not summarize. Do not only read more files "
        "or run tests. Do not write a placeholder, status note, or unrelated "
        "verification test. Do not ship a stub, mock, or unused form field. "
        "Call str_replace on the existing files (or write_file "
        "for a new/small file) and implement this exact goal. Do not rewrite a "
        "large file with write_file. If existing tests assert old markup, APIs, "
        "or layout that this goal replaces, update those tests in the same change.\n\n"
        f"Goal:\n{goal.strip()}\n\n"
        f"Why it is not done:\n{reason.strip()}"
    )


def _no_work_summary(agent_summary: str, verdict: GoalReview, upstream_detail: str) -> str:
    if verdict.met:
        lines = [
            f"No work to do: {verdict.reason}",
            upstream_detail,
        ]
    else:
        lines = [
            "No files changed this cycle, and the current tree does not already "
            f"meet the goal: {review_reason(verdict)}",
            upstream_detail,
        ]
    notes = (agent_summary or "").strip()
    skip_notes = {
        "stopped after reaching the iteration limit.",
        "agent finished without a summary.",
    }
    if notes and notes.lower() not in skip_notes:
        lines.extend(["", "Agent notes:", notes])
    return "\n".join(lines)


def _review_goal(
    workspace: Workspace,
    project: ProjectConfig,
    llm: LLMClient,
    goal: str,
    diff: str,
    summary: str,
    tests_passed: bool | None,
    *,
    existing: bool = False,
    upstream: str | None = None,
    stopped_reason: str | None = None,
) -> GoalReview:
    ui_text = ""
    ui_evidence = None
    if ui_review_needed(goal, diff, workspace.root):
        log_progress("Reviewing rendered UI...")
        try:
            ui_evidence = collect_ui_evidence(workspace, project, goal)
        except Exception as exc:  # noqa: BLE001 - UI capture must not abort the cycle
            ui_evidence = UiEvidence(ok=False, notes=f"could not render UI: {exc}")
        if ui_evidence is not None:
            ui_text = format_ui_evidence(ui_evidence)
            log_progress(ui_evidence.summary)
    verdict = review_goal(
        llm,
        goal,
        diff=diff,
        summary=summary,
        tests_passed=tests_passed,
        existing=existing,
        upstream=upstream,
        ui_evidence=ui_text or None,
    )
    blocking = []
    smashed = []
    if ui_evidence is not None:
        blocking = ui_evidence.blocking_errors
        smashed = ui_evidence.smashed
    ui_errors = tuple(blocking[:12])
    if verdict.met and blocking:
        first = blocking[0]
        js = bool(ui_evidence and (ui_evidence.page_errors or ui_evidence.console_errors))
        reason = (
            f"Rendered UI has JavaScript errors: {first}"
            if js
            else f"Rendered UI failed to load a required resource: {first}"
        )
        overridden = GoalReview(False, reason, parsed=True, ui_errors=ui_errors)
        record_event(kind="review", attempt=1, met=False, parsed=True, reason=overridden.reason)
        return overridden
    if verdict.met and smashed:
        overridden = GoalReview(
            False,
            f"Rendered UI controls are unusable: {smashed[0]}",
            parsed=True,
            ui_errors=ui_errors,
        )
        record_event(kind="review", attempt=1, met=False, parsed=True, reason=overridden.reason)
        return overridden
    unverified = unverified_interactive_ui(goal, ui_evidence) if ui_evidence else None
    if verdict.met and unverified:
        overridden = GoalReview(False, unverified, parsed=True, ui_errors=ui_errors)
        record_event(kind="review", attempt=1, met=False, parsed=True, reason=overridden.reason)
        return overridden
    markers = half_baked_diff_markers(diff)
    if verdict.met and markers and not existing:
        overridden = GoalReview(
            False,
            f"diff still has unfinished work: {markers[0]}",
            parsed=True,
            ui_errors=ui_errors,
        )
        record_event(kind="review", attempt=1, met=False, parsed=True, reason=overridden.reason)
        return overridden
    unwired = unwired_ui_markers(diff)
    if verdict.met and unwired and not existing:
        overridden = GoalReview(
            False,
            f"diff adds a control or API that is not wired: {unwired[0]}",
            parsed=True,
            ui_errors=ui_errors,
        )
        record_event(kind="review", attempt=1, met=False, parsed=True, reason=overridden.reason)
        return overridden
    invented = invalid_doc_commands(diff)
    if verdict.met and invented and not existing:
        overridden = GoalReview(
            False,
            f"diff documents a command or mount that cannot work: {invented[0]}",
            parsed=True,
            ui_errors=ui_errors,
        )
        record_event(kind="review", attempt=1, met=False, parsed=True, reason=overridden.reason)
        return overridden
    unused = unused_new_symbols(diff)
    if verdict.met and unused and not existing:
        overridden = GoalReview(
            False,
            f"diff adds a helper that nothing calls: {unused[0]}",
            parsed=True,
            ui_errors=ui_errors,
        )
        record_event(kind="review", attempt=1, met=False, parsed=True, reason=overridden.reason)
        return overridden
    incomplete = incomplete_agent_run(summary, stopped_reason)
    ui_verified = bool(
        ui_evidence is not None
        and ui_evidence.ok
        and not blocking
        and not smashed
        and not ui_evidence.dead_controls
        and (ui_evidence.snapshot or ui_evidence.screenshot)
    )
    if verdict.met and incomplete and not existing and not ui_verified:
        overridden = GoalReview(False, incomplete, parsed=True, ui_errors=ui_errors)
        record_event(kind="review", attempt=1, met=False, parsed=True, reason=overridden.reason)
        return overridden
    if ui_errors and verdict.ui_errors != ui_errors:
        return GoalReview(
            met=verdict.met,
            reason=verdict.reason,
            raw=verdict.raw,
            parsed=verdict.parsed,
            ui_errors=ui_errors,
        )
    return verdict


def _ensure_goal_met(
    workspace: Workspace,
    settings: Settings,
    project: ProjectConfig,
    llm: LLMClient,
    agent: CodingAgent,
    goal: str,
    summary: str,
    sha_before: str | None,
    tests_passed: bool | None,
    allow_create_pr: bool,
    *,
    stopped_reason: str | None = None,
) -> dict[str, object]:
    context = collect_context(workspace.root, project, allow_publish=allow_create_pr)
    work_diff = collect_work_diff(workspace, sha_before, goal=goal)
    verdict = _review_goal(
        workspace,
        project,
        llm,
        goal,
        work_diff,
        summary,
        tests_passed,
        stopped_reason=stopped_reason,
    )
    log_progress(f"Goal review: met={verdict.met} ({review_reason(verdict)})")
    attempts = 0
    stalls = 0
    extra_granted = False
    last_diff = work_diff
    limit = project.max_repair_attempts
    while not verdict.met and attempts < limit:
        attempts += 1
        log_progress(f"Goal retry {attempts}/{limit}: {review_reason(verdict)}")
        work_diff = collect_work_diff(workspace, sha_before, goal=goal)
        follow = agent.run(
            _goal_retry_prompt(
                goal,
                review_reason(verdict),
                work_diff,
                ui_errors=verdict.ui_errors,
                stalled=stalls > 0,
            ),
            context,
            require_change=True,
        )
        if follow.summary:
            summary = follow.summary
        tests_after = _maybe_test(workspace, project, settings, phase="retry")
        tests_after = _repair_failing_tests(
            workspace,
            project,
            settings,
            agent,
            tests_after,
            allow_create_pr=allow_create_pr,
        )
        tests_passed = None if tests_after is None else tests_after.ok
        if settings.require_tests and tests_after is not None and not tests_after.ok:
            return {
                "met": False,
                "reason": "tests failed after goal retry",
                "summary": summary,
                "tests_passed": False,
                "tests_failed": True,
            }
        work_diff = collect_work_diff(workspace, sha_before, goal=goal)
        if work_diff == last_diff:
            stalls += 1
            if not extra_granted and attempts >= project.max_repair_attempts:
                extra_granted = True
                limit = attempts + GOAL_RETRY_STALL_EXTRA
                log_progress("Retry made no file changes; continuing instead of giving up.")
        else:
            stalls = 0
        last_diff = work_diff
        verdict = _review_goal(
            workspace,
            project,
            llm,
            goal,
            work_diff,
            summary,
            tests_passed,
            stopped_reason=follow.stopped_reason,
        )
        log_progress(f"Goal review: met={verdict.met} ({review_reason(verdict)})")
    return {
        "met": verdict.met,
        "reason": review_reason(verdict),
        "summary": summary,
        "tests_passed": tests_passed,
        "tests_failed": False,
    }


GOAL_RETRY_STALL_EXTRA = 2


def _goal_retry_prompt(
    goal: str,
    reason: str,
    diff: str,
    *,
    ui_errors: list[str] | tuple[str, ...] | None = None,
    stalled: bool = False,
) -> str:
    stall = ""
    if stalled:
        stall = (
            "You inspected the repo on the last retry but did not change files. "
            "This is not done. Call str_replace or write_file now.\n\n"
        )
    broken = ""
    errors = [str(item).strip() for item in (ui_errors or []) if str(item).strip()]
    if errors:
        listed = "\n".join(f"- {item}" for item in errors[:12])
        broken = (
            "\n\nRendered UI still has broken resources. Fix them in this retry. "
            "A 404 means the HTML/CSS/JS href does not match a file the web "
            "server actually serves. Do not only create the files; mount or route "
            "them (FastAPI StaticFiles, express.static, or the project's existing "
            "static path) or correct the href. Then call review_ui and confirm "
            "those URLs load.\n"
            f"Broken resources:\n{listed}"
        )
    return (
        f"{stall}"
        "The stated goal is not done. The current diff does not fulfill it. "
        "Do not summarize. Do not switch to a different task. "
        "Do not write placeholder, status, or unrelated verification files. "
        "Do not leave a stub, mock, unused required field, or a follow-up for "
        "a later run. If the workspace already has the data (git remote, config), "
        "use it instead of asking a human to re-type it. "
        "If the diff is unrelated, replace or remove it and implement this exact goal. "
        "Use str_replace for surgical edits; do not rewrite large files. "
        "If existing tests assert old markup, APIs, or layout that this goal "
        "replaces, update those tests. "
        "If this is a UI change, call review_ui after editing, click new tabs, "
        "and fix render errors, 404s, or dead controls. Do not stop while the "
        "page fails to load CSS or JS you added. "
        "Follow enabled workspace skills. Do not add a helper that nothing calls. "
        "If you edited docs or compose files, the commands must actually exist "
        "(loco clone, docker compose, git clone — never docker clone). Run tests. "
        "Do not bind-mount this app's .loco over the mounted workspace.\n\n"
        f"Goal:\n{goal.strip()}\n\n"
        f"Why it is not done:\n{reason.strip()}"
        f"{broken}\n\n"
        f"Current diff:\n{diff}"
    )


def _test_repair_prompt(output: str) -> str:
    return (
        "The test suite failed after the last changes. Fix the failures. "
        "If a test asserts old markup, API shape, or behavior that this goal "
        "intentionally changed, update that test to match the new implementation. "
        "Do not revert the goal. Do not write placeholder or unrelated "
        "verification tests.\n\n"
        f"{(output or '').strip()}"
    )


def _repair_failing_tests(
    workspace: Workspace,
    project: ProjectConfig,
    settings: Settings,
    agent: CodingAgent,
    tests_after,
    *,
    allow_create_pr: bool,
):
    if tests_after is None or tests_after.ok:
        return tests_after
    context = collect_context(workspace.root, project, allow_publish=allow_create_pr)
    for attempt in range(project.max_repair_attempts):
        log_progress(f"Repair attempt {attempt + 1}/{project.max_repair_attempts}")
        agent.run(
            _test_repair_prompt(tests_after.output),
            context,
            require_change=True,
        )
        tests_after = _maybe_test(workspace, project, settings, phase="repair")
        if tests_after is None or tests_after.ok:
            break
    return tests_after


def _choose_goal(project: ProjectConfig, tests_before) -> str | None:
    if tests_before is not None and not tests_before.ok:
        return "Make the project's test suite pass.\n\n" + tests_before.output
    goals = load_goals(project.root, project.goals_file)
    return goals[0] if goals else None


def _maybe_test(
    workspace: Workspace,
    project: ProjectConfig,
    settings: Settings,
    *,
    phase: str = "tests",
):
    if not project.test_command:
        return None
    return run_project_tests(
        workspace,
        project.test_command,
        settings.command_timeout_seconds,
        phase=phase,
    )


def _lint_repair_prompt(output: str) -> str:
    return (
        "The linter failed after the last changes. Fix every reported error. "
        "Ruff F401 means remove or use the unused import. E501 means wrap the "
        "line to 100 characters or less. I001 means sort imports the way ruff "
        "does. If ruff reports files that would be reformatted, run the "
        "formatter so the files match `ruff format`. Do not disable the "
        "linter. Do not skip a finding. After edits, the configured lint "
        "command must pass.\n\n"
        f"{(output or '').strip()}"
    )


def _maybe_lint(
    workspace: Workspace,
    project: ProjectConfig,
    settings: Settings,
    *,
    phase: str = "lint",
):
    if not project.lint_command:
        return None
    return run_project_lint(
        workspace,
        project.lint_command,
        settings.command_timeout_seconds,
        phase=phase,
    )


def _ensure_lint(
    workspace: Workspace,
    project: ProjectConfig,
    settings: Settings,
    agent: CodingAgent,
    *,
    allow_create_pr: bool,
):
    lint_after = _maybe_lint(workspace, project, settings, phase="after")
    if lint_after is None or lint_after.ok:
        return lint_after
    context = collect_context(workspace.root, project, allow_publish=allow_create_pr)
    for attempt in range(project.max_repair_attempts):
        log_progress(f"Lint repair {attempt + 1}/{project.max_repair_attempts}")
        agent.run(
            _lint_repair_prompt(lint_after.output),
            context,
            require_change=True,
        )
        lint_after = _maybe_lint(workspace, project, settings, phase="repair")
        if lint_after is None or lint_after.ok:
            break
    return lint_after


def _commit_message(goal: str, summary: str) -> str:
    first_goal_line = goal_headline(goal)[:72]
    first_summary = summary.strip().splitlines()[0][:72] if summary.strip() else first_goal_line
    if first_goal_line.lower().startswith("make the project's test suite pass"):
        return first_summary
    return first_goal_line


_AGENT_REVIEW_UI_SLUG = "ui-review_review-ui_"


def _pr_screenshot_rel(name: str) -> str:
    filename = Path(name).name
    if filename == "ui-review.png":
        return f".loco/{filename}"
    return f".loco/ui-screenshots/{filename}"


def _pr_screenshot_names() -> list[str]:
    """Keep the last successful capture of the change, not every review_ui dump."""
    change_shots: list[str] = []
    fallback: list[str] = []
    for event in current_events():
        if event.get("kind") != "ui":
            continue
        if event.get("ok") is False:
            continue
        name = str(event.get("screenshot_filename") or "").strip()
        if not name:
            name = Path(str(event.get("screenshot") or "")).name
        name = Path(name).name
        if not name:
            continue
        if _AGENT_REVIEW_UI_SLUG in name.lower():
            fallback.append(name)
            continue
        change_shots.append(name)
    chosen = change_shots or fallback
    return chosen[-1:] if chosen else []


def _pr_screenshot_files(workspace: Workspace) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for name in _pr_screenshot_names():
        path = resolve_ui_screenshot(workspace.root, name)
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        files.append(path)
    return files


def _pr_image_base(
    workspace: Workspace,
    commit_sha: str | None,
    remote: str = "origin",
) -> str | None:
    if not commit_sha:
        return None
    parsed = github_owner_repo(workspace, remote)
    if not parsed:
        return None
    owner, repo = parsed
    return f"https://github.com/{owner}/{repo}/raw/{commit_sha}"


def _pr_body(
    goal: str,
    summary: str,
    *,
    tests_passed: bool | None,
    commit_sha: str | None,
    branch: str,
    screenshots: list[str] | None = None,
    image_base: str | None = None,
) -> str:
    detail = (summary or "").strip() or goal.strip()
    if tests_passed is True:
        tests_line = "- [x] Project tests passed locally"
    elif tests_passed is False:
        tests_line = "- [ ] Project tests failed locally — do not merge until green"
    else:
        tests_line = "- [ ] No project test command; verify manually"

    # Extract changes and reason from summary
    changes_section = _extract_changes_from_summary(summary)
    testing_steps = _extract_testing_steps_from_summary(summary)

    reason = _extract_reason_from_summary(summary)
    if not reason:
        reason = f"To address the stated goal: {goal.strip()}"
    lines = [
        "## What Changed",
        "",
        changes_section if changes_section else (detail or "No changes described."),
        "",
        "## Why This Update",
        "",
        reason,
        "",
        "## Goal",
        "",
        goal.strip(),
        "",
        "## Test plan",
        "",
        tests_line,
        "- [x] Goal review confirmed the requested outcome",
        "- [ ] Review this feature branch; do not merge unreviewed commits to main",
    ]

    # Add testing steps if available
    if testing_steps:
        lines.extend(["", "## Steps to Test"])
        lines.extend([""])
        for i, step in enumerate(testing_steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    shots = [Path(name).name for name in (screenshots or []) if str(name).strip()]
    hosted = (image_base or "").rstrip("/")
    if shots and hosted:
        lines.extend(["", "## Screenshots", ""])
        for name in shots:
            lines.append(f"![{name}]({hosted}/{_pr_screenshot_rel(name)})")
    if commit_sha:
        lines.extend(["", f"Commit: `{commit_sha}`"])
    if branch:
        lines.append(f"Branch: `{branch}`")
    lines.extend(["", CO_AUTHORED_BY])
    return "\n".join(lines)


def _extract_changes_from_summary(summary: str | None) -> str:
    """Extract what changed from the summary, focusing on file changes and modifications."""
    if not summary:
        return "No changes described."

    summary_lines = [line.strip() for line in summary.split("\n") if line.strip()]

    # Look for sections in the summary like "Files changed:", "Changes:", etc.
    change_sections = []
    current_section = []

    for line in summary_lines:
        lower = line.lower()
        if any(
            trigger in lower
            for trigger in (
                "files changed",
                "changes:",
                "this commit",
                "modified ",
                "added ",
                "removed ",
            )
        ):
            if current_section:
                change_sections.append("\n".join(current_section))
            current_section = [line]
        elif current_section:
            current_section.append(line)

    if current_section:
        change_sections.append("\n".join(current_section))

    # If we found explicit change sections, use them
    if change_sections:
        return change_sections[0] if len(change_sections) == 1 else "\n\n".join(change_sections)

    # Otherwise, use the first non-goal line from the summary
    for index, line in enumerate(summary_lines):
        lowered = line.lower()
        if lowered.startswith("goal:") or lowered.startswith("why:"):
            continue
        rest = summary_lines[index + 1 : index + 50]
        if rest:
            return line + "\n" + "\n".join(rest)
        return line
    return ""


def _extract_reason_from_summary(summary: str | None) -> str:
    """Extract why the update was made from the summary."""
    if not summary:
        return ""

    summary_lower = summary.lower()

    reason_words = ("why", "rationale", "motivation", "justification")
    if any(word in summary_lower for word in reason_words):
        lines = [line.strip() for line in summary.split("\n") if line.strip()]
        for index, line in enumerate(lines):
            if any(trigger in line.lower() for trigger in (*reason_words, "because")):
                return " ".join(lines[index : min(index + 4, len(lines))])

    # Look for "to address" or "to implement" patterns
    to_patterns = ["to address", "to implement", "to fix", "to add", "to update", "to resolve"]
    for pattern in to_patterns:
        if pattern in summary_lower:
            idx = summary_lower.index(pattern)
            # Get the rest of the line after the pattern
            after = summary[idx:].split("\n")[0]
            return after.strip()

    return ""


def _extract_testing_steps_from_summary(summary: str | None) -> list[str]:
    """Extract testing steps from the summary."""
    if not summary:
        return []

    lines = [line.strip() for line in summary.split("\n") if line.strip()]
    testing_steps = []

    # Look for testing-related sections
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(
            trigger in lower
            for trigger in (
                "test",
                "verify",
                "check",
                "validate",
                "steps",
                "how to test",
                "testing",
            )
        ):
            # Collect testing-related content
            for j in range(i, min(i + 6, len(lines))):
                test_line = lines[j]
                headers = ("testing notes:", "testing:", "tests:", "verify:", "check:")
                if test_line.lower().strip() in headers:
                    # Skip headers, start from the next line
                    continue
                numbered = test_line[:2] in {"1.", "2.", "3.", "4.", "5."}
                if numbered or test_line.startswith("-"):
                    if numbered:
                        test_line = test_line.split(".", 1)[-1].strip()
                    else:
                        test_line = test_line.lstrip("-").strip()
                    if test_line:
                        testing_steps.append(test_line)
                elif test_line.startswith("##") and "testing" not in lower:
                    break
                elif test_line:
                    testing_steps.append(test_line)

    return testing_steps


def _git_env(settings: Settings) -> dict[str, str]:
    env: dict[str, str] = {}
    if settings.git_author_name:
        env["GIT_AUTHOR_NAME"] = settings.git_author_name
        env["GIT_COMMITTER_NAME"] = settings.git_author_name
    if settings.git_author_email:
        env["GIT_AUTHOR_EMAIL"] = settings.git_author_email
        env["GIT_COMMITTER_EMAIL"] = settings.git_author_email
    return env


def _stamp_git(root: Path, result: CycleResult) -> None:
    try:
        workspace = Workspace(root)
    except (OSError, SandboxError):
        return
    if not result.branch:
        result.branch = current_branch(workspace)
    if not result.commit_sha:
        result.commit_sha = current_sha(workspace)


def _attach_events(result: CycleResult) -> None:
    if not result.events:
        result.events = current_events()


def _write_run_log(root: Path, result: CycleResult) -> None:
    _stamp_git(root, result)
    _attach_events(result)
    ensure_run_gitignore(root)
    runs = root / ".loco" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = runs / f"{stamp}.json"
    payload = asdict(result)
    payload.setdefault("id", stamp)
    payload.setdefault("created_at", result.created_at)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_to_history(root: Path, result: CycleResult) -> None:
    """Append cycle result to history.json for persistent storage across restarts."""
    history_file = root / "history.json"
    # Read existing history
    try:
        if history_file.exists():
            with open(history_file, encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
    except (OSError, json.JSONDecodeError):
        # If we can't read the file for any reason, start fresh
        history = []

    # Add the new result to the beginning of the list
    # (most recent at the beginning) and keep only the last 100 entries
    _stamp_git(root, result)
    _attach_events(result)
    history.insert(0, asdict(result))

    # Limit history size to avoid file becoming too large
    if len(history) > 100:
        history = history[:100]

    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except OSError:
        # If we can't write the file, silently ignore (not critical for functionality)
        pass

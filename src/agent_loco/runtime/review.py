from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agent_loco.llm.client import LLMClient
from agent_loco.progress import record_event, timed_complete

REVIEW_RAW_LIMIT = 8_000

REVIEW_SYSTEM = """You are a strict reviewer for an unattended coding agent.

Decide whether the current code change fulfills the stated goal. Passing tests
is not enough. A related, partial, planned, or hidden change is not enough.

Rules:
- If the goal says to remove something, it must actually be gone from the diff.
- If the goal asks for a user-visible control, it must be visible and wired, not
  display:none or otherwise non-functional.
- When a Rendered UI section is present, JavaScript exceptions (ReferenceError,
  uncaught errors) mean the goal is unmet. Missing JS or CSS the page requested
  is unmet. Favicon 404s and missing historical screenshots are not unmet.
  Controls with 0x0 or tiny sizes are unmet. A template diff is not enough if the
  rendered page is blank, crushed, or throws.
- Dead controls (a tab or button that does not reveal its panel when clicked) are
  unmet. Markup for a Screenshots tab is not enough. A Current/Past runs tab that
  leaves Progress, Changes, or Screenshots at 0x0 or display:none is unmet.
  A screenshot of the default main screen is not evidence that a new sidebar
  panel, modal, or tab works; the Rendered UI section must show that control
  was clicked and its panel is visible.
- If capture could not click (static dump-dom) and the goal adds tabs, screenshots,
  or other interactive controls, the goal is unmet.
- A summary that claims the work is done does not count unless the diff shows it.
- If the goal is a GitHub issue, the body and comments are the spec. Matching
  only the title is unmet. The requested behavior must be in the diff.
- A new button, tab, or form control with no JavaScript handler in the same
  change is unmet. An API route with no client that calls it is unmet unless
  the goal is backend-only.
- Mocks, stubs, NotImplementedError, generated sample data, or comments like
  "in a real implementation" mean unmet unless the goal is explicitly to add a stub.
- A UI that asks the user to re-type owner/repo (or similar) when the workspace
  git remote already identifies the repository is unmet. Infer it and load the data.
- Changing flags, guards, or docs so a later step *could* do the goal is unmet.
  The requested behavior must actually happen (create the PR, load the issues,
  serve the CSS).
- Documented commands must exist. `docker clone` is unmet. A `loco` subcommand
  that is not in the CLI is unmet. Docs that init a path before it exists, then
  clone into it, are unmet.
- Bind-mounting the agent app's `.loco` over `/workspaces/.loco` is unmet. The
  mounted project volume must keep its own `.loco` (config and run history).
- An unused helper is unmet. A new function, class, or method that nothing in
  the change calls (no route, CLI, UI, or test) is not the requested behavior.
- An agent that stopped at the iteration limit has not finished, unless a
  Rendered UI section is present and the requested controls are visible
  without JS errors, missing CSS/JS, dead buttons, or overlapping controls.
  In that case judge the rendered page, not the stop reason.
- Opening a PR is done by the cycle after review, not by editing publish guards
  or shell tools. If the goal is to open a PR from the current branch and that
  branch already contains the work, set met=true.
- Use the changed-file list. A lockfile may be summarized as `package: old -> new`
  instead of a hash dump; that still counts as updating the package.
- Do not infer that a dependency was not updated just because hashes were omitted.
- Set met=true only when a careful reviewer would accept the work as complete.

Reply with ONLY a JSON object:
{"met": true or false, "reason": "one sentence"}
"""

EXISTING_REVIEW_SYSTEM = """You are a strict reviewer for an unattended coding agent.

The agent made no file changes this cycle. Decide whether the CURRENT workspace
already fulfills the stated goal.

Rules:
- Passing tests is not enough by itself.
- Use the current evidence: dependency versions, manifests, and files.
- When a Rendered UI section is present, JavaScript exceptions mean the goal is
  unmet. Missing JS or CSS the page requested is unmet. Favicon 404s and missing
  historical screenshots are not unmet. Controls with 0x0 or tiny sizes are unmet.
  A missing diff does not save a broken page.
- Dead controls and static dump-dom captures of interactive UI (tabs, screenshots,
  buttons) mean the goal is unmet.
- A missing diff does not mean the goal is unmet if the tree already has the
  requested result (for example a lockfile already on the requested version).
- Scaffolding, mocks, unused required fields, or "this enables a later step" mean
  the goal is not already met.
- If the goal is to open a pull request from the current feature branch and
  that branch already has the work, the goal is met. The cycle opens the PR.
- Set met=true only if a careful reviewer would accept the current tree as complete.
- Set met=false if the goal still requires work.

Reply with ONLY a JSON object:
{"met": true or false, "reason": "one sentence"}
"""

REVIEW_JSON_NUDGE = (
    "Your previous reply was not valid. Reply with ONLY this JSON object and "
    'no other text:\n{"met": true or false, "reason": "one sentence"}'
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_HALF_BAKED_ADDED = re.compile(
    r"(?i)("
    r"\bNotImplementedError\b|"
    r"in a real implementation|"
    r"for demonstration|"
    r"generate_mock_|"
    r"mock[_ ]issues?|"
    r"fake data|"
    r"coming soon|"
    r"half[- ]baked|"
    r"pass\s*#\s*(?:stub|todo|later|not implemented)"
    r")"
)


def half_baked_diff_markers(diff: str | None) -> list[str]:
    """Added lines that show a stub, mock, or placeholder instead of the real work."""
    hits: list[str] = []
    untracked_body = False
    for line in (diff or "").splitlines():
        if line.startswith("+++ "):
            untracked_body = True
            continue
        if line.startswith("diff --git ") or line.startswith("@@") or line.startswith("--- "):
            untracked_body = False
            continue
        added = False
        text = line
        if line.startswith("+") and not line.startswith("+++"):
            added = True
            text = line[1:]
        elif untracked_body and not line.startswith("-"):
            added = True
        if added and _HALF_BAKED_ADDED.search(text):
            hits.append(text.strip()[:160])
    return hits


_BUTTON_CLASS_RE = re.compile(
    r"<button\b[^>]*\bclass\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_APP_ROUTE_RE = re.compile(
    r"""@app\.(?:post|get|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]""",
    re.IGNORECASE,
)
_HANDLER_HINTS = (
    "addEventListener",
    "onclick",
    "fetch(",
    "closest(",
    "querySelector",
    "getElementById",
)
_GENERIC_BUTTON_CLASSES = frozenset(
    {
        "task",
        "secondary",
        "primary",
        "selected",
        "sidebar-toggle",
        "workspace-tab-select",
        "workspace-tab-add",
        "workspace-tab-archive",
        "progress-stage",
    }
)
_GENERIC_ROUTE_SEGMENTS = frozenset(
    {"", "api", "tasks", "history", "goals", "models", "workspaces", "servers"}
)


def unwired_ui_markers(diff: str | None) -> list[str]:
    """Buttons or API routes added in the diff without a matching client call."""
    html_added: list[str] = []
    js_added: list[str] = []
    py_added: list[str] = []
    current = ""
    untracked_body = False
    for line in (diff or "").splitlines():
        if line.startswith("+++ "):
            current = line[4:].strip()
            if current.startswith("b/"):
                current = current[2:]
            untracked_body = True
            continue
        if line.startswith("diff --git ") or line.startswith("@@") or line.startswith("--- "):
            untracked_body = False
            continue
        added = False
        text = line
        if line.startswith("+") and not line.startswith("+++"):
            added = True
            text = line[1:]
        elif untracked_body and not line.startswith("-"):
            added = True
        if not added:
            continue
        lowered = current.lower()
        if lowered.endswith((".js", ".mjs", ".ts", ".tsx", ".jsx")):
            js_added.append(text)
        elif lowered.endswith((".html", ".htm")) or "templates/" in lowered:
            html_added.append(text)
            if any(hint in text for hint in _HANDLER_HINTS):
                js_added.append(text)
        if lowered.endswith(".py"):
            py_added.append(text)

    script = "\n".join(js_added)
    hits: list[str] = []
    for line in html_added:
        for classes in _BUTTON_CLASS_RE.findall(line):
            for cls in classes.split():
                name = cls.strip()
                if not name or name in _GENERIC_BUTTON_CLASSES:
                    continue
                if name in script:
                    continue
                hits.append(f"button.{name} has no click handler")
    for line in py_added:
        match = _APP_ROUTE_RE.search(line)
        if not match:
            continue
        path = match.group(1)
        segments = [
            part.strip("{}")
            for part in path.split("/")
            if part and not part.startswith("{") and part not in _GENERIC_ROUTE_SEGMENTS
        ]
        if not segments:
            continue
        token = segments[-1]
        client_text = "\n".join(html_added + js_added)
        if path in client_text or token in script or f"/{token}" in script:
            continue
        hits.append(f"route {path} is not called from the UI")
    return hits


_DOCKER_SUBCOMMANDS = frozenset(
    {
        "attach",
        "build",
        "builder",
        "buildx",
        "commit",
        "compose",
        "config",
        "container",
        "context",
        "cp",
        "create",
        "desktop",
        "diff",
        "events",
        "exec",
        "export",
        "extension",
        "history",
        "image",
        "images",
        "import",
        "info",
        "inspect",
        "kill",
        "load",
        "login",
        "logout",
        "logs",
        "manifest",
        "network",
        "node",
        "pause",
        "plugin",
        "port",
        "ps",
        "pull",
        "push",
        "rename",
        "restart",
        "rm",
        "rmi",
        "run",
        "save",
        "scan",
        "scout",
        "search",
        "secret",
        "service",
        "stack",
        "start",
        "stats",
        "stop",
        "swarm",
        "system",
        "tag",
        "top",
        "trust",
        "unpause",
        "update",
        "version",
        "volume",
        "wait",
    }
)
_LOCO_SUBCOMMANDS = frozenset({"clone", "doctor", "init", "run", "ui", "watch"})
_DOC_SUFFIXES = (".md", ".rst", ".adoc", ".txt")
_OPS_NAMES = (
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
)
_LINE_DOCKER_RE = re.compile(
    r"^\s*(?:\$\s*)?docker(?:-compose)?\s+([a-z][\w-]*)",
    re.IGNORECASE,
)
_LINE_LOCO_RE = re.compile(
    r"^\s*(?:\$\s*)?(?:uv\s+run\s+)?loco\s+([a-z][\w-]*)",
    re.IGNORECASE,
)
_DOCKER_CLONE_RE = re.compile(r"\bdocker(?:-compose)?\s+clone\b", re.IGNORECASE)
_LOCO_OVERLAY_RE = re.compile(r"(?:\./)?\.loco\s*:\s*/workspaces/\.loco")


def _added_diff_lines(diff: str | None) -> list[tuple[str, str]]:
    """Return (path, added_line) pairs from a unified diff."""
    current = ""
    untracked_body = False
    rows: list[tuple[str, str]] = []
    for line in (diff or "").splitlines():
        if line.startswith("+++ "):
            current = line[4:].strip()
            if current.startswith("b/"):
                current = current[2:]
            untracked_body = True
            continue
        if line.startswith("diff --git ") or line.startswith("@@") or line.startswith("--- "):
            untracked_body = False
            continue
        added = False
        text = line
        if line.startswith("+") and not line.startswith("+++"):
            added = True
            text = line[1:]
        elif untracked_body and not line.startswith("-"):
            added = True
        if added:
            rows.append((current, text))
    return rows


def _is_doc_path(path: str) -> bool:
    lowered = (path or "").replace("\\", "/").lower()
    name = lowered.rsplit("/", 1)[-1]
    return lowered.endswith(_DOC_SUFFIXES) or name.startswith("readme")


def _is_compose_path(path: str) -> bool:
    lowered = (path or "").replace("\\", "/").lower()
    name = lowered.rsplit("/", 1)[-1]
    return name in _OPS_NAMES or "docker-compose" in name


def invalid_doc_commands(diff: str | None) -> list[str]:
    """Documented CLI that is not a real docker/loco command, or a .loco overlay mount."""
    hits: list[str] = []
    for path, text in _added_diff_lines(diff):
        if _is_compose_path(path) and _LOCO_OVERLAY_RE.search(text):
            hits.append("compose bind-mounts host .loco over /workspaces/.loco")
        if not _is_doc_path(path) and not _is_compose_path(path):
            continue
        if _DOCKER_CLONE_RE.search(text):
            hits.append("docs invent `docker clone`")
        match = _LINE_DOCKER_RE.match(text)
        if match:
            command = match.group(1).lower()
            if command not in _DOCKER_SUBCOMMANDS:
                hits.append(f"docs invent `docker {command}`")
        match = _LINE_LOCO_RE.match(text)
        if match:
            command = match.group(1).lower()
            if command not in _LOCO_SUBCOMMANDS:
                hits.append(f"docs invent `loco {command}`")
    return list(dict.fromkeys(hits))


_PY_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PY_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:(]")
_JS_FN_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_JS_CONST_FN_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?:async\s+)?(?:function\b|\()"
)
_ROUTE_DECORATOR_RE = re.compile(
    r"^\s*@(?:app|router|api_router|blueprint)\.",
    re.IGNORECASE,
)
_SKIP_SYMBOL_NAMES = frozenset({"main", "setup", "teardown"})


def _is_test_path(path: str) -> bool:
    raw = (path or "").replace("\\", "/").lower()
    name = raw.rsplit("/", 1)[-1]
    return (
        "/tests/" in f"/{raw}"
        or "/test/" in f"/{raw}"
        or name.startswith("test_")
        or name.endswith(("_test.py", "_test.ts", "_test.js", "_spec.py", "_spec.ts"))
        or ".spec." in name
        or ".test." in name
    )


def _def_name(text: str) -> str | None:
    for regex in (_PY_DEF_RE, _PY_CLASS_RE, _JS_FN_RE, _JS_CONST_FN_RE):
        match = regex.match(text or "")
        if match:
            return match.group(1)
    return None


def _removed_def_names(diff: str | None) -> set[str]:
    names: set[str] = set()
    for line in (diff or "").splitlines():
        if line.startswith("-") and not line.startswith("---"):
            name = _def_name(line[1:])
            if name:
                names.add(name)
    return names


def _symbol_used(
    name: str,
    added: list[tuple[str, str]],
    origin_path: str,
    origin_line: str,
) -> bool:
    call = re.compile(rf"\b{re.escape(name)}\s*\(")
    attr = re.compile(rf"\.{re.escape(name)}\b")
    word = re.compile(rf"\b{re.escape(name)}\b")
    for path, text in added:
        if path == origin_path and text == origin_line:
            continue
        if _def_name(text) == name:
            continue
        if call.search(text) or attr.search(text):
            return True
        if path != origin_path and word.search(text):
            return True
    return False


def unused_new_symbols(diff: str | None) -> list[str]:
    """New public functions/classes in the diff that nothing else calls."""
    added = _added_diff_lines(diff)
    removed_names = _removed_def_names(diff)
    last_line: dict[str, str] = {}
    hits: list[str] = []
    for path, text in added:
        prev = last_line.get(path, "")
        last_line[path] = text
        if _is_test_path(path):
            continue
        name = _def_name(text)
        if not name:
            continue
        if name.startswith("_") or name.startswith("test") or name in _SKIP_SYMBOL_NAMES:
            continue
        if name in removed_names:
            continue
        if _ROUTE_DECORATOR_RE.match(prev):
            continue
        if _symbol_used(name, added, path, text):
            continue
        hits.append(f"{name} is added but nothing calls it")
    return list(dict.fromkeys(hits))


def incomplete_agent_run(summary: str | None, stopped_reason: str | None = None) -> str | None:
    """True when the coding agent ran out of turns instead of finishing."""
    if (stopped_reason or "").strip() == "max_iterations":
        return "agent stopped after reaching the iteration limit"
    text = (summary or "").strip().lower()
    if "stopped after reaching the iteration limit" in text:
        return "agent stopped after reaching the iteration limit"
    return None


@dataclass(frozen=True)
class GoalReview:
    met: bool
    reason: str
    raw: str = ""
    parsed: bool = True
    ui_errors: tuple[str, ...] = ()


def is_test_suite_goal(goal: str) -> bool:
    first = goal.strip().splitlines()[0].lower() if goal.strip() else ""
    return first.startswith("make the project's test suite pass")


_OPEN_PR_GOAL_RE = re.compile(
    r"\b(create|open|make|publish)\b.{0,48}\b(pr|pull request)\b",
    re.IGNORECASE,
)


def is_open_pr_goal(goal: str) -> bool:
    """True when the user asked to open a PR from the current branch, not to change code."""
    return bool(_OPEN_PR_GOAL_RE.search(goal or ""))


def parse_review(text: str | None) -> GoalReview:
    raw = (text or "").strip()
    if not raw:
        return GoalReview(False, "reviewer returned no verdict", "", parsed=False)
    for blob in _candidate_blobs(raw):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "met" not in data:
            continue
        met = _as_bool(data.get("met"))
        if met is None:
            continue
        reason = str(data.get("reason") or data.get("why") or "").strip()
        if not reason:
            reason = "goal met" if met else "goal not met"
        return GoalReview(met, reason, raw, parsed=True)
    return GoalReview(
        False,
        'reviewer did not return a {"met": ...} verdict',
        raw,
        parsed=False,
    )


def review_goal(
    llm: LLMClient,
    goal: str,
    *,
    diff: str,
    summary: str,
    tests_passed: bool | None,
    existing: bool = False,
    upstream: str | None = None,
    ui_evidence: str | None = None,
) -> GoalReview:
    if is_test_suite_goal(goal) and tests_passed is True:
        verdict = GoalReview(True, "project tests passed after the change")
        _record_review(verdict, attempt=1)
        return verdict
    evidence_label = "Current workspace:" if existing else "Diff:"
    user_parts = [
        "Goal:",
        goal.strip(),
        "",
        "Agent summary:",
        (summary or "").strip() or "(none)",
        "",
        f"Tests passed: {tests_passed}",
    ]
    if existing:
        user_parts.extend(
            [
                "",
                "This cycle produced no file changes.",
            ]
        )
        if upstream:
            user_parts.extend(["", "Upstream:", upstream])
    user_parts.extend(["", evidence_label, diff.strip() or "(no diff)"])
    if ui_evidence and ui_evidence.strip():
        user_parts.extend(["", ui_evidence.strip()])
    user = "\n".join(user_parts)
    messages: list[dict] = [
        {
            "role": "system",
            "content": EXISTING_REVIEW_SYSTEM if existing else REVIEW_SYSTEM,
        },
        {"role": "user", "content": user},
    ]
    turn = timed_complete(llm, messages, [], purpose="review")
    verdict = parse_review(turn.text)
    _record_review(verdict, attempt=1)
    if verdict.parsed:
        return verdict
    messages.append({"role": "assistant", "content": turn.text or ""})
    messages.append({"role": "user", "content": REVIEW_JSON_NUDGE})
    turn = timed_complete(llm, messages, [], purpose="review")
    verdict = parse_review(turn.text)
    _record_review(verdict, attempt=2)
    return verdict


def review_reason(verdict: GoalReview) -> str:
    """Human-readable review outcome, including a raw snippet when JSON parse failed."""
    if verdict.parsed or not verdict.raw:
        return verdict.reason
    snippet = " ".join(verdict.raw.split())
    if len(snippet) > 240:
        snippet = snippet[:240] + "..."
    return f"{verdict.reason} ({snippet})"


def _record_review(verdict: GoalReview, *, attempt: int) -> None:
    record_event(
        kind="review",
        attempt=attempt,
        met=verdict.met,
        parsed=verdict.parsed,
        reason=verdict.reason,
        raw=_clip_raw(verdict.raw),
    )


def _clip_raw(text: str, limit: int = REVIEW_RAW_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... truncated"


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "met", "done"}:
            return True
        if lowered in {"false", "no", "unmet", "not met"}:
            return False
    return None


def _candidate_blobs(text: str) -> list[str]:
    blobs = [match.group(1).strip() for match in _FENCE.finditer(text)]
    blobs.extend(_embedded_json_objects(text))
    stripped = text.strip()
    if stripped.startswith("{"):
        blobs.append(stripped)
    return blobs


def _embedded_json_objects(text: str) -> list[str]:
    blobs: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        end = _match_json_object(text, index)
        if end is None:
            index += 1
            continue
        blobs.append(text[index:end])
        index = end
    return blobs


def _match_json_object(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from agent_loco.agent.prompts import SYSTEM_PROMPT, adapt_system_prompt, user_prompt
from agent_loco.llm.client import AssistantTurn, LLMClient, ToolCall
from agent_loco.llm.toolparse import parse_tool_calls
from agent_loco.progress import record_event, timed_complete
from agent_loco.tools import ToolSpec, execute_tool

log = logging.getLogger("loco")

MUTATING_TOOLS = {"write_file", "str_replace"}
VERIFY_TOOLS = {"review_ui", "run_tests"}
UI_SUFFIXES = {".html", ".htm", ".css", ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte"}
OPS_NAMES = {
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
}
OPS_SUFFIXES = {".md", ".rst", ".adoc"}
MAX_PLAN_NUDGES = 3
MAX_REQUIRE_CHANGE_NUDGES = 8
MAX_UNFINISHED_NUDGES = 4
MAX_VALIDATE_NUDGES = 4
MAX_INSPECT_ROUNDS = 3
CONTINUE_NUDGE = (
    "You have not changed any files that implement the goal. Inspection is over. "
    "Call str_replace on the existing files (or write_file for a new/small file) "
    "and implement this exact goal — not a placeholder, status note, "
    "verification test, or a different task you noticed in the repo. "
    "Do not rewrite a large file with write_file."
)
UNFINISHED_NUDGE = (
    "That reply is not a finish. You still have work left on the stated goal. "
    "Finish the real behavior now: no stubs, mocks, TODOs, unused fields, or "
    "follow-up for a later run. Call a tool and apply the next edit. "
    "Do not narrate the change; str_replace it."
)
MUST_EDIT_NUDGE = (
    "The goal is still unmet. Do not stop. Call str_replace or write_file "
    "and fix the failure. If CSS or JS returned 404, the file is missing or "
    "the web server is not serving that path — add the static mount/route "
    "or correct the href, then call review_ui until those URLs load."
)
INSPECT_NUDGE = (
    "You have been inspecting the repo without changing files. "
    "Stop reading. Call str_replace and implement this exact goal. "
    "Use write_file only for a new or small file. "
    "Do not write placeholder, status, or verification files."
)
VALIDATE_UI_NUDGE = (
    "You changed UI files but have not called review_ui since the last edit. "
    "Call review_ui, click the new control, and fix errors, 404s, or dead buttons. "
    "A control that is visible but does not fetch or change the page is not done."
)
VALIDATE_TESTS_NUDGE = (
    "You changed code but have not called run_tests since the last edit. "
    "Run the project tests and fix failures before summarizing."
)
VALIDATE_OPS_NUDGE = (
    "You changed Docker, compose, or docs but have not called run_tests since "
    "the last edit. Run tests. Commands in README must exist: use loco clone or "
    "git clone, never docker clone. Do not bind-mount this app's .loco over "
    "/workspaces/.loco. init requires a directory that already exists."
)
RECOVERY_TWICE_NUDGE = (
    "Tool {name} failed twice: {message}. Do not repeat the same arguments. "
    "If this is str_replace, call read_file on {path}, copy a unique old_string "
    "from the file body (not the N| prefix), then str_replace once."
)
RECOVERY_THREE_NUDGE = (
    "Tool {name} failed three times in a row. "
    "Search the workspace or list the directory to understand the current state "
    "before trying again."
)
_UNFINISHED_RE = re.compile(
    r"(?is)("
    r"\blet me\b|"
    r"\bi(?:'m about to|'ll| will| am going to)\b|"
    r"\bi need to (?:fix|update|write|add|change|edit|run|implement|adjust)\b|"
    r"\bhalf[- ]baked\b|"
    r"\bfor now\b|"
    r"\bin a real implementation\b|"
    r"\bleave (?:it|this|the rest) (?:for|to) (?:a )?later\b|"
    r"\bthis enables\b|"
    r"\bgenerated? sample\b|"
    r"\bmock(?:ed)? data\b|"
    r"\bnot fully wired\b|"
    r"\bstarting point\b|"
    r"\bhandler will\b|"
    r"\biteration limit\b"
    r")"
)


def looks_unfinished(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if value.endswith(":") or value.endswith("..."):
        return True
    return bool(_UNFINISHED_RE.search(value))


def _is_ui_path(path: str) -> bool:
    raw = (path or "").replace("\\", "/").lower()
    suffix = Path(raw).suffix
    return suffix in UI_SUFFIXES or "/templates/" in raw or raw.endswith(".html")


def _is_ops_path(path: str) -> bool:
    raw = (path or "").replace("\\", "/").lower()
    name = Path(raw).name
    suffix = Path(raw).suffix
    return (
        name in OPS_NAMES
        or name.startswith("dockerfile")
        or "docker-compose" in name
        or suffix in OPS_SUFFIXES
        or name.startswith("readme")
    )


def _tool_path(arguments: dict) -> str:
    value = arguments.get("path") if isinstance(arguments, dict) else None
    return str(value or "")


@dataclass
class FailureTracker:
    """Track consecutive failures for recovery logic."""

    consecutive_failures: int = 0
    last_failed: str | None = None
    failed_by_tool_path: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    last_path: str | None = None

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.last_failed = None
        self.failed_by_tool_path.clear()
        self.last_path = None

    def record_failure(self, tool_name: str, path: str, message: str) -> None:
        """Record a tool failure, updating counters and tracking."""
        self.consecutive_failures += 1
        self.last_failed = message
        self.last_path = path if path else None
        key = (tool_name, path)
        self.failed_by_tool_path[key] += 1

    def get_same_tool_failures(self, tool_name: str, path: str) -> int:
        """Get failure count for the same tool on the same path."""
        key = (tool_name, path)
        return self.failed_by_tool_path.get(key, 0)

    def has_recovery_triggered(self) -> bool:
        """Check if recovery should trigger (3 consecutive failures but not auto-read)."""
        return self.consecutive_failures >= 3


@dataclass
class AgentResult:
    summary: str
    iterations: int
    tool_calls: int
    stopped_reason: str
    last_error: str | None = None


class CodingAgent:
    def __init__(
        self,
        llm: LLMClient,
        tools: list[ToolSpec],
        *,
        max_iterations: int,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        prompt = (system_prompt or "").strip()
        model = getattr(llm, "model", "") or ""
        self.system_prompt = adapt_system_prompt(prompt or SYSTEM_PROMPT, model)

    def run(
        self,
        goal: str,
        context: str = "",
        *,
        require_change: bool = False,
        require_tests: bool = False,
    ) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt(goal, context)},
        ]
        schemas = [tool.openai_schema() for tool in self.tools]
        known_names = {tool.name for tool in self.tools}
        tool_calls = 0
        mutated = False
        mutated_ui = False
        mutated_ops = False
        verified_ui = False
        verified_tests = False
        plan_nudges = 0
        unfinished_nudges = 0
        validate_nudges = 0
        inspect_rounds = 0

        last_text = ""
        last_error: str | None = None
        failure_tracker = FailureTracker()

        for iteration in range(1, self.max_iterations + 1):
            turn = timed_complete(self.llm, messages, schemas, purpose="agent")
            last_text = turn.text or last_text
            snippet = " ".join((turn.text or "").split())
            if snippet:
                log.info("model %s", snippet[:400])
            calls = turn.tool_calls or parse_tool_calls(turn.text, known_names)
            if calls:
                native = bool(turn.tool_calls)
                if native:
                    messages.append(_assistant_tool_message(turn))
                else:
                    messages.append({"role": "assistant", "content": turn.text or ""})
                wrote = False
                inspected = False
                for call in calls:
                    tool_calls += 1
                    result = execute_tool(self.tools, call.name, call.arguments)
                    log.info("tool %s ok=%s", call.name, result.ok)
                    log.debug("%s args=%s", call.name, call.arguments)
                    if not result.ok:
                        last_error = (result.output or "")[:200]
                        # Track failure with path
                        path = _tool_path(call.arguments)
                        failure_tracker.record_failure(call.name, path, result.output or "")

                        # Check for recovery trigger after 3 consecutive failures
                        if failure_tracker.has_recovery_triggered():
                            # Avoid auto-reading on a failed auto-read
                            if call.name != "read_file" or "recovery" not in (result.output or ""):
                                path_to_read = failure_tracker.last_path
                                if path_to_read:
                                    log.info("tool %s ok=… (recovery)", "read_file")
                                    read_result = execute_tool(
                                        self.tools, "read_file", {"path": path_to_read}
                                    )
                                    tool_calls += 1  # Count the recovery read_file call
                                    log.info("tool read_file ok=%s (recovery)", read_result.ok)
                                    messages.append(
                                        _tool_result_message(
                                            ToolCall(
                                                id="auto-read",
                                                name="read_file",
                                                arguments={"path": path_to_read},
                                            ),
                                            read_result.output,
                                            native=True,
                                        )
                                    )
                                    failure_tracker.reset()
                                else:
                                    # No path known, nudge agent to search/list
                                    nudge_msg = RECOVERY_THREE_NUDGE.format(
                                        name=call.name,
                                    )
                                    log.info("nudging agent to search or list after hammering")
                                    messages.append(
                                        {"role": "user", "content": _nudge(nudge_msg, goal)}
                                    )
                                    failure_tracker.reset()
                                continue
                        # Check for 2 failures of same tool on same path
                        same_path_failures = failure_tracker.get_same_tool_failures(call.name, path)
                        if same_path_failures >= 2:
                            nudge_msg = RECOVERY_TWICE_NUDGE.format(
                                name=call.name,
                                message=(result.output or "")[:200],
                                path=path or "the file",
                            )
                            log.info("injecting recovery nudge: tool %s failed twice", call.name)
                            messages.append({"role": "user", "content": _nudge(nudge_msg, goal)})
                    else:
                        # Success resets failure counter
                        failure_tracker.reset()

                        if call.name in MUTATING_TOOLS:
                            mutated = True
                            wrote = True
                            verified_tests = False
                            tool_path = _tool_path(call.arguments)
                            if _is_ui_path(tool_path):
                                mutated_ui = True
                                verified_ui = False
                            if _is_ops_path(tool_path):
                                mutated_ops = True
                        elif call.name == "review_ui":
                            verified_ui = bool(result.ok)
                        elif call.name == "run_tests":
                            verified_tests = bool(result.ok)
                        elif call.name not in VERIFY_TOOLS:
                            inspected = True
                    messages.append(_tool_result_message(call, result.output, native=native))
                if wrote:
                    inspect_rounds = 0
                elif inspected and not mutated:
                    inspect_rounds += 1
                    if inspect_rounds >= MAX_INSPECT_ROUNDS:
                        inspect_rounds = 0
                        log.info("nudging agent to stop inspecting and write files")
                        messages.append({"role": "user", "content": _nudge(INSPECT_NUDGE, goal)})
                continue

            if not mutated:
                nudge_limit = MAX_REQUIRE_CHANGE_NUDGES if require_change else MAX_PLAN_NUDGES
                if plan_nudges < nudge_limit:
                    plan_nudges += 1
                    log.info("nudging agent to keep working after a plan-only turn")
                    messages.append({"role": "assistant", "content": turn.text or ""})
                    nudge = MUST_EDIT_NUDGE if require_change else CONTINUE_NUDGE
                    messages.append({"role": "user", "content": _nudge(nudge, goal)})
                    continue

            if looks_unfinished(turn.text) and unfinished_nudges < MAX_UNFINISHED_NUDGES:
                unfinished_nudges += 1
                log.info("nudging agent after an unfinished reply")
                record_event(kind="step", message="Agent tried to stop mid-work; continuing.")
                messages.append({"role": "assistant", "content": turn.text or ""})
                messages.append({"role": "user", "content": _nudge(UNFINISHED_NUDGE, goal)})
                continue

            if mutated_ui and not verified_ui and validate_nudges < MAX_VALIDATE_NUDGES:
                validate_nudges += 1
                log.info("nudging agent to validate UI changes")
                record_event(kind="step", message="Agent skipped UI validation; continuing.")
                messages.append({"role": "assistant", "content": turn.text or ""})
                messages.append({"role": "user", "content": _nudge(VALIDATE_UI_NUDGE, goal)})
                continue

            needs_tests = require_tests or mutated_ops
            if (
                needs_tests
                and mutated
                and not verified_tests
                and "run_tests" in known_names
                and validate_nudges < MAX_VALIDATE_NUDGES
            ):
                validate_nudges += 1
                log.info("nudging agent to run tests after edits")
                record_event(kind="step", message="Agent skipped tests; continuing.")
                messages.append({"role": "assistant", "content": turn.text or ""})
                nudge = VALIDATE_OPS_NUDGE if mutated_ops else VALIDATE_TESTS_NUDGE
                messages.append({"role": "user", "content": _nudge(nudge, goal)})
                continue

            summary = (turn.text or "").strip() or "Agent finished without a summary."
            messages.append({"role": "assistant", "content": summary})
            return AgentResult(
                summary=summary,
                iterations=iteration,
                tool_calls=tool_calls,
                stopped_reason="completed",
                last_error=last_error,
            )

        if mutated_ui and not verified_ui and "review_ui" in known_names:
            result = execute_tool(self.tools, "review_ui", {})
            tool_calls += 1
            log.info("tool review_ui ok=%s (end-of-run)", result.ok)
            if result.ok:
                record_event(kind="step", message="Verified rendered UI at the end of the run.")
                summary = (last_text or "").strip() or "Verified rendered UI after edits."
                return AgentResult(
                    summary=summary,
                    iterations=self.max_iterations,
                    tool_calls=tool_calls,
                    stopped_reason="completed",
                )
            record_event(
                kind="step",
                message="Rendered UI still failing at iteration limit.",
            )

        return AgentResult(
            summary="Stopped after reaching the iteration limit.",
            iterations=self.max_iterations,
            tool_calls=tool_calls,
            stopped_reason="max_iterations",
            last_error=last_error,
        )


def _nudge(template: str, goal: str) -> str:
    return f"{template}\n\nGoal:\n{goal.strip()}\n\nOnly edit files that implement that goal."


def _assistant_tool_message(turn: AssistantTurn) -> dict:
    payload: dict = {"role": "assistant", "content": turn.text or None, "tool_calls": []}
    for call in turn.tool_calls:
        payload["tool_calls"].append(_openai_tool_call(call))
    return payload


def _tool_result_message(call: ToolCall, output: str, *, native: bool) -> dict:
    clipped = _clip(output)
    if native:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": clipped,
        }
    return {
        "role": "user",
        "content": f"Tool {call.name} returned:\n{clipped}",
    }


def _openai_tool_call(call: ToolCall) -> dict:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments),
        },
    }


def _clip(text: str, limit: int = 16_000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... truncated"

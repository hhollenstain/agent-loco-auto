from __future__ import annotations

import difflib
import logging
import time
from contextvars import ContextVar, Token
from typing import Any

from agent_loco.llm.client import AssistantTurn, LLMClient
from agent_loco.logging import format_elapsed, utcnow_iso

log = logging.getLogger("loco")

MAX_DIFF_CHARS = 12_000
MAX_TEST_OUTPUT_CHARS = 16_000
LLM_SNIPPET = 400
ProgressToken = Token[list[dict[str, Any]] | None]

_events: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "loco_progress_events", default=None
)
_originals: ContextVar[dict[str, str] | None] = ContextVar(
    "loco_file_originals", default=None
)


def current_events() -> list[dict[str, Any]]:
    return list(_events.get() or [])


def bind_progress(events: list[dict[str, Any]] | None = None) -> ProgressToken | None:
    """Attach an event list to this task/cycle. Reuses a list already bound."""
    if _events.get() is not None and events is None:
        return None
    _originals.set({})
    return _events.set([] if events is None else events)


def reset_progress(token: ProgressToken | None) -> None:
    if token is not None:
        _events.reset(token)
        _originals.set(None)


def record_event(kind: str, **fields: Any) -> dict[str, Any]:
    event = {"kind": kind, "at": utcnow_iso(), **fields}
    bucket = _events.get()
    if bucket is not None:
        bucket.append(event)
    return event


def unified_file_diff(path: str, before: str, after: str, *, created: bool) -> str:
    from_file = "/dev/null" if created else f"a/{path}"
    to_file = f"b/{path}"
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=from_file,
            tofile=to_file,
        )
    )
    if not diff:
        return f"--- {from_file}\n+++ {to_file}\n"
    if len(diff) > MAX_DIFF_CHARS:
        return diff[:MAX_DIFF_CHARS] + "\n... truncated"
    return diff


def record_file_change(
    path: str,
    *,
    before: str | None,
    after: str,
    created: bool,
) -> dict[str, Any]:
    action = "created" if created else "updated"
    if before is None:
        diff = f"(could not read previous contents of {path})"
        net_diff = diff
        net_action = action
    else:
        diff = unified_file_diff(path, before, after, created=created)
        originals = _originals.get()
        if originals is None:
            originals = {}
            _originals.set(originals)
        if path not in originals:
            originals[path] = "" if created else before
        original = originals[path]
        net_created = original == ""
        net_diff = unified_file_diff(path, original, after, created=net_created)
        net_action = "created" if net_created else "updated"
    event = record_event(
        kind="file",
        path=path,
        action=action,
        diff=diff,
        net_diff=net_diff,
        net_action=net_action,
    )
    log.info("file %s %s", action, path)
    return event


def clip_output(text: str, limit: int = MAX_TEST_OUTPUT_CHARS) -> str:
    value = text or ""
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"... truncated {omitted} chars\n" + value[-limit:]


def clip_text(text: object, limit: int = LLM_SNIPPET) -> str:
    """Collapse whitespace and keep a short prefix for UI and logs."""
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def llm_turn_snippets(messages: object, response: object) -> tuple[str, str]:
    """Last non-assistant prompt and the model reply, clipped for the UI."""
    agent = ""
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                continue
            agent = clip_text(message.get("content"))
            if agent:
                break
    return agent, clip_text(response)


def public_event(event: dict[str, Any] | object) -> dict[str, Any] | object:
    """Copy a progress event without the live LLM transcript."""
    if not isinstance(event, dict):
        return event
    payload = {key: value for key, value in event.items() if key != "messages"}
    if event.get("kind") == "llm":
        if not payload.get("agent") and not payload.get("model"):
            agent, model = llm_turn_snippets(event.get("messages"), event.get("response"))
            payload["agent"] = agent
            payload["model"] = model
        response = payload.get("response")
        if isinstance(response, str) and len(response) > LLM_SNIPPET:
            payload["response"] = clip_text(response)
    return payload


def public_run_item(item: dict[str, Any] | object) -> dict[str, Any] | object:
    """Copy a run or history payload without LLM transcripts in events."""
    if not isinstance(item, dict):
        return item
    events = item.get("events")
    if not isinstance(events, list):
        return item
    slim = dict(item)
    slim["events"] = [public_event(event) for event in events]
    return slim


def record_test_run(
    *,
    command: str,
    ok: bool,
    output: str,
    phase: str = "tests",
    elapsed_ms: int | None = None,
    reused: bool = False,
) -> dict[str, Any]:
    event = record_event(
        kind="test",
        command=command,
        ok=ok,
        phase=phase,
        elapsed_ms=elapsed_ms,
        reused=reused,
        output=clip_output(output),
    )
    if reused:
        log.info("tests %s ok=%s reused", phase, ok)
    else:
        log.info("tests %s ok=%s", phase, ok)
    return event


def record_lint_run(
    *,
    command: str,
    ok: bool,
    output: str,
    phase: str = "lint",
    elapsed_ms: int | None = None,
    reused: bool = False,
) -> dict[str, Any]:
    event = record_event(
        kind="lint",
        command=command,
        ok=ok,
        phase=phase,
        elapsed_ms=elapsed_ms,
        reused=reused,
        output=clip_output(output),
    )
    if reused:
        log.info("lint %s ok=%s reused", phase, ok)
    else:
        log.info("lint %s ok=%s", phase, ok)
    return event


def timed_complete(
    llm: LLMClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    purpose: str,
) -> AssistantTurn:
    """Run an LLM completion and record how long the server took to answer."""
    started = time.perf_counter()
    try:
        turn = llm.complete(messages, tools)
    except Exception:
        elapsed = time.perf_counter() - started
        record_event(
            kind="llm",
            purpose=purpose,
            ok=False,
            elapsed_ms=int(elapsed * 1000),
        )
        log.info("llm %s failed after %s", purpose, format_elapsed(elapsed))
        raise
    elapsed = time.perf_counter() - started
    elapsed_ms = int(round(elapsed * 1000))
    agent, model = llm_turn_snippets(messages, turn.text or "")
    record_event(
        kind="llm",
        purpose=purpose,
        ok=True,
        elapsed_ms=elapsed_ms,
        agent=agent,
        model=model,
        prompt_tokens=turn.prompt_tokens,
        completion_tokens=turn.completion_tokens,
        total_tokens=turn.total_tokens,
    )
    log.info("llm %s response in %s", purpose, format_elapsed(elapsed))
    return turn


def token_usage_from_events(events: list[dict[str, Any]] | None) -> dict[str, int | None]:
    """Current context fill and billed total from recorded LLM turns."""
    used: int | None = None
    billed = 0
    saw_billed = False
    for event in events or []:
        if event.get("kind") != "llm" or event.get("ok") is False:
            continue
        prompt = _token_int(event.get("prompt_tokens"))
        completion = _token_int(event.get("completion_tokens"))
        total = _token_int(event.get("total_tokens"))
        if total is None and (prompt is not None or completion is not None):
            total = (prompt or 0) + (completion or 0)
        if total is not None:
            billed += total
            saw_billed = True
        if prompt is not None:
            used = prompt
        elif total is not None:
            used = total
    return {
        "tokens_used": used,
        "tokens_total": billed if saw_billed else None,
    }


def attach_token_usage(item: dict[str, Any]) -> dict[str, Any]:
    """Fill tokens_used / tokens_total on a task or history payload."""
    usage = token_usage_from_events(item.get("events") if isinstance(item, dict) else None)
    if item.get("tokens_used") is None:
        item["tokens_used"] = usage["tokens_used"]
    if item.get("tokens_total") is None:
        item["tokens_total"] = usage["tokens_total"]
    item.setdefault("context_window", None)
    return item


def _token_int(value: object) -> int | None:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None

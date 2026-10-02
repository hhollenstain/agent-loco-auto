from __future__ import annotations

from pathlib import Path

from agent_loco.config import Settings
from agent_loco.llm.client import (
    completion_usage,
    context_window_from_models_payload,
    context_window_from_ollama_show,
)
from agent_loco.progress import attach_token_usage, token_usage_from_events
from agent_loco.runtime.improve import CycleResult
from agent_loco.runtime.tasks import TaskManager


class _Usage:
    def __init__(self) -> None:
        self.prompt_tokens = 2048
        self.completion_tokens = 64
        self.total_tokens = 2112


class _Response:
    usage = _Usage()


def test_completion_usage_reads_openai_object() -> None:
    prompt, completion, total = completion_usage(_Response())
    assert (prompt, completion, total) == (2048, 64, 2112)
    assert completion_usage({"usage": {"prompt_tokens": 10, "total_tokens": 12}}) == (
        10,
        None,
        12,
    )
    assert completion_usage({}) == (None, None, None)


def test_context_window_from_ollama_show() -> None:
    assert (
        context_window_from_ollama_show(
            {"model_info": {"qwen3.context_length": 32768, "general.architecture": "qwen3"}}
        )
        == 32768
    )
    assert context_window_from_ollama_show({"parameters": "num_ctx 8192\ntemperature 0.7"}) == 8192
    assert context_window_from_ollama_show({}) is None


def test_context_window_from_models_payload() -> None:
    payload = {
        "data": [
            {"id": "other", "max_model_len": 4096},
            {"id": "Qwen3.5:27b", "context_length": 262144},
        ]
    }
    assert context_window_from_models_payload(payload, "Qwen3.5:27b") == 262144
    assert context_window_from_models_payload({"data": []}, "missing") is None


def test_token_usage_from_events_uses_latest_prompt() -> None:
    events = [
        {"kind": "llm", "ok": True, "prompt_tokens": 800, "total_tokens": 900},
        {"kind": "file", "path": "app.py"},
        {"kind": "llm", "ok": True, "prompt_tokens": 1400, "completion_tokens": 50},
        {"kind": "llm", "ok": False, "prompt_tokens": 9999},
    ]
    usage = token_usage_from_events(events)
    assert usage["tokens_used"] == 1400
    assert usage["tokens_total"] == 900 + 1450


def test_task_dict_and_history_include_token_fields(
    settings: Settings,
    tmp_path: Path,
) -> None:
    manager = TaskManager(
        settings,
        runner=lambda task: CycleResult(
            status="success",
            goal=task.goal,
            summary="done",
            tests_passed=True,
            committed=False,
            published=False,
            commit_sha=None,
            reason="ok",
        ),
    )
    try:
        task = manager.submit(tmp_path, "count tokens")
        task.events.append(
            {
                "kind": "llm",
                "ok": True,
                "prompt_tokens": 1200,
                "completion_tokens": 80,
                "total_tokens": 1280,
            }
        )
        task.context_window = 32768
        payload = task.to_dict()
    finally:
        manager.shutdown(wait=False)
    assert payload["tokens_used"] == 1200
    assert payload["tokens_total"] == 1280
    assert payload["context_window"] == 32768
    history = attach_token_usage(
        {
            "events": [
                {"kind": "llm", "ok": True, "prompt_tokens": 512, "total_tokens": 600},
            ]
        }
    )
    assert history["tokens_used"] == 512
    assert history["tokens_total"] == 600
    assert history["context_window"] is None

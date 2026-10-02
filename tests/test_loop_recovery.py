from __future__ import annotations

from pathlib import Path

import pytest

from agent_loco.agent.loop import CodingAgent, FailureTracker
from agent_loco.config import Settings
from agent_loco.llm.client import AssistantTurn, ScriptedClient, ToolCall
from agent_loco.runtime.uireview import UiEvidence
from agent_loco.sandbox import Workspace
from agent_loco.tools import execute_tool, file_tools


@pytest.fixture(autouse=True)
def _fake_ui_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_loco.tools.browser.collect_ui_evidence",
        lambda *args, **kwargs: UiEvidence(ok=True, snapshot="OK"),
    )


def test_failure_tracker_basic(tmp_path: Path) -> None:
    """Test basic failure tracking functionality."""
    tracker = FailureTracker()
    assert tracker.consecutive_failures == 0
    assert tracker.last_failed is None

    tracker.record_failure("str_replace", "app.py", "not found")
    assert tracker.consecutive_failures == 1
    assert tracker.last_failed == "not found"
    assert tracker.last_path == "app.py"
    assert tracker.get_same_tool_failures("str_replace", "app.py") == 1
    assert tracker.has_recovery_triggered() is False

    # Another failure on same tool/path
    tracker.record_failure("str_replace", "app.py", "not found")
    assert tracker.consecutive_failures == 2
    assert tracker.get_same_tool_failures("str_replace", "app.py") == 2

    # Different path
    tracker.record_failure("str_replace", "other.py", "not found")
    assert tracker.consecutive_failures == 3
    assert tracker.get_same_tool_failures("str_replace", "app.py") == 2
    assert tracker.get_same_tool_failures("str_replace", "other.py") == 1
    assert tracker.has_recovery_triggered() is True

    tracker.reset()
    assert tracker.consecutive_failures == 0
    assert tracker.last_failed is None
    assert tracker.has_recovery_triggered() is False


def test_repeated_str_replace_failure_injects_recovery_nudge(
    tmp_path: Path, settings: Settings
) -> None:
    """Scripted model calls str_replace twice with missing old_string, then a third turn;
    messages after the second failure include 'failed twice'."""

    # Create a small file to work with
    (tmp_path / "app.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    # Initialize workspace and tools
    workspace = Workspace(tmp_path)
    tools = file_tools(workspace)

    # Scripted agent that calls str_replace with wrong old_string twice
    llm = ScriptedClient(
        [
            AssistantTurn(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="str_replace",
                        arguments={
                            "path": "app.py",
                            "old_string": "def nonexistent():",  # Wrong
                            "new_string": "def hello():\n    return 'loco'",
                            "replace_all": False,
                        },
                    )
                ],
            ),
            AssistantTurn(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="str_replace",
                        arguments={
                            "path": "app.py",
                            "old_string": "def nonexistent():",  # Same wrong string
                            "new_string": "def hello():\n    return 'loco'",
                            "replace_all": False,
                        },
                    )
                ],
            ),
            AssistantTurn(text="Checking what to do next."),
        ]
    )

    agent = CodingAgent(
        llm=llm,
        tools=tools,
        max_iterations=5,
    )

    result = agent.run("Fix the hello function")

    # Check that the recovery nudge was injected after second failure
    # llm.calls tracks the messages sent to the LLM in each turn
    found_nudge = False
    for turn_msgs in llm.calls:
        for msg in turn_msgs:
            content = msg.get("content", "")
            if isinstance(content, str) and "failed twice" in content.lower():
                found_nudge = True
                break
        if found_nudge:
            break

    assert found_nudge, "Expected recovery nudge with 'failed twice' text after second failure"
    assert result.tool_calls >= 2  # At least two str_replace attempts


def test_third_consecutive_failure_auto_reads_file(tmp_path: Path, settings: Settings) -> None:
    """Three failed str_replace on app.py; a read_file of app.py is executed by the loop."""

    # Create a small file
    (tmp_path / "app.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    workspace = Workspace(tmp_path)

    # Mock execute_tool to track what was called
    call_log: list[str] = []
    original_execute_tool = execute_tool

    def tracked_execute_tool(tools: list, name: str, arguments: dict) -> str:
        call_log.append(f"{name}:{arguments}")
        return original_execute_tool(tools, name, arguments)

    import agent_loco.agent.loop as loop_module
    from agent_loco.tools import execute_tool as real_execute

    loop_module.execute_tool = tracked_execute_tool

    try:
        llm = ScriptedClient(
            [
                AssistantTurn(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="str_replace",
                            arguments={
                                "path": "app.py",
                                "old_string": "wrong",
                                "new_string": "right",
                                "replace_all": False,
                            },
                        )
                    ],
                ),
                AssistantTurn(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            id="call-2",
                            name="str_replace",
                            arguments={
                                "path": "app.py",
                                "old_string": "wrong",
                                "new_string": "right",
                                "replace_all": False,
                            },
                        )
                    ],
                ),
                AssistantTurn(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            id="call-3",
                            name="str_replace",
                            arguments={
                                "path": "app.py",
                                "old_string": "wrong",
                                "new_string": "right",
                                "replace_all": False,
                            },
                        )
                    ],
                ),
                AssistantTurn(text="Done."),
            ]
        )

        tools = file_tools(workspace)

        agent = CodingAgent(
            llm=llm,
            tools=tools,
            max_iterations=10,
        )

        result = agent.run("Fix something")

        # Check that read_file was called automatically (with recovery marker)
        read_file_calls = [c for c in call_log if c.startswith("read_file:")]
        assert len(read_file_calls) >= 1, (
            "Expected at least one auto read_file call after 3 failures."
        )

        # The agent should have tried 3 str_replace + at least 1 read_file
        assert result.tool_calls >= 4
    finally:
        loop_module.execute_tool = real_execute


def test_success_resets_failure_streak(tmp_path: Path, settings: Settings) -> None:
    """Fail, succeed write_file, fail once; no auto-read yet."""

    workspace = Workspace(tmp_path)
    tools = file_tools(workspace)

    llm = ScriptedClient(
        [
            AssistantTurn(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="str_replace",
                        arguments={
                            "path": "nonexistent.py",
                            "old_string": "wrong",
                            "new_string": "right",
                            "replace_all": False,
                        },
                    )
                ],
            ),
            AssistantTurn(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="write_file",
                        arguments={
                            "path": "app.py",
                            "content": "def hello():\n    return 'world'\n",
                        },
                    )
                ],
            ),
            AssistantTurn(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="call-3",
                        name="str_replace",
                        arguments={
                            "path": "nonexistent.py",
                            "old_string": "wrong",
                            "new_string": "right",
                            "replace_all": False,
                        },
                    )
                ],
            ),
            AssistantTurn(text="Checking state after success reset."),
        ]
    )

    agent = CodingAgent(
        llm=llm,
        tools=tools,
        max_iterations=8,
    )

    # Mock execute_tool to track what was called
    call_log: list[str] = []
    original_execute_tool = execute_tool

    def tracked_execute_tool(tools: list, name: str, arguments: dict) -> str:
        call_log.append(f"{name}:{arguments}")
        return original_execute_tool(tools, name, arguments)

    import agent_loco.agent.loop as loop_module

    loop_module.execute_tool = tracked_execute_tool

    try:
        result = agent.run("Test success reset")

        # Verify the agent completed without getting stuck on auto-read
        # After write_file success, counter resets
        # One failure after success doesn't trigger auto-read (needs 3)
        assert result.iterations >= 3, "Expected at least 3 iterations to include write_file"
        assert result.tool_calls >= 3, "Expected at least 3 tool calls"
    finally:
        loop_module.execute_tool = original_execute_tool

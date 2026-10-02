from __future__ import annotations

from agent_loco.agent.loop import AgentResult
from agent_loco.runtime.improve import compact_handoff


class TestCompactHandoff:
    def test_compact_handoff_lists_changed_paths_not_full_diff(self):
        diff = """diff --git a/file1.py b/file1.py
index 123..456 100644
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-foo
+bar

diff --git a/file2.py b/file2.py
index 789..012 100644
--- a/file2.py
+++ b/file2.py
@@ -10,5 +10,5 @@
  baz"""
        result = compact_handoff(
            goal="test goal",
            summary="done",
            diff=diff,
            last_tool_error=None,
            stopped_reason="max_iterations",
        )
        assert "COMPACT HANDOFF" in result
        assert "test goal" in result
        assert "file1.py" in result
        assert "file2.py" in result
        assert "Stop after 40 iterations" in result or "Finish the remaining work" in result
        assert len(result) <= 4000

    def test_compact_handoff_limits_changed_paths_to_20(self):
        lines = "\n".join(f"diff --git a/file{i}.py b/file{i}.py" for i in range(30))
        result = compact_handoff(
            goal="test",
            summary="done",
            diff=lines,
            last_tool_error=None,
            stopped_reason="max_iterations",
        )
        # Count file paths (lines starting with "  - file")
        path_count = sum(1 for line in result.splitlines() if line.strip().startswith("- file"))
        assert path_count <= 20

    def test_compact_handoff_includes_last_tool_error(self):
        result = compact_handoff(
            goal="test",
            summary="done",
            diff="diff --git a/x.py b/x.py",
            last_tool_error="file not found: /tmp/nonexistent",
            stopped_reason="max_iterations",
        )
        assert "Last tool error" in result
        assert "file not found" in result

    def test_compact_handoff_capped_at_4000(self):
        long_file = "x.py" * 1000
        result = compact_handoff(
            goal="test",
            summary="done",
            diff=f"diff --git a/{long_file} b/{long_file}",
            last_tool_error=None,
            stopped_reason="max_iterations",
        )
        assert len(result) <= 3997


class TestAgentResultLastError:
    def test_agent_result_includes_last_error_on_max_iterations(self):
        result = AgentResult(
            summary="done",
            iterations=40,
            tool_calls=5,
            stopped_reason="max_iterations",
            last_error="file not found",
        )
        assert result.last_error == "file not found"

    def test_agent_result_defaults_last_error_to_none(self):
        result = AgentResult(
            summary="done",
            iterations=1,
            tool_calls=0,
            stopped_reason="completed",
        )
        assert result.last_error is None

    def test_agent_result_completed_with_last_error(self):
        result = AgentResult(
            summary="done",
            iterations=1,
            tool_calls=1,
            stopped_reason="completed",
            last_error="file not found",
        )
        assert result.last_error == "file not found"

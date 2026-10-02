from __future__ import annotations

from pathlib import Path

from agent_loco.agent.prompts import SYSTEM_PROMPT
from agent_loco.llm.client import AssistantTurn, ScriptedClient
from agent_loco.progress import bind_progress, record_event, reset_progress
from agent_loco.runtime.improve import _pr_body, _pr_screenshot_names, _review_goal
from agent_loco.runtime.project import load_project
from agent_loco.runtime.review import REVIEW_SYSTEM, review_goal
from agent_loco.runtime.uireview import (
    UiEvidence,
    _from_eval,
    _playwright_probe,
    classify_network_failure,
    format_ui_evidence,
    is_generic_resource_console,
    resolve_ui_screenshot,
    ui_review_needed,
    unverified_interactive_ui,
)
from agent_loco.sandbox import Workspace
from agent_loco.tools import build_tools, execute_tool


def test_ui_review_needed_for_template_diff_and_ui_goals(tmp_path: Path) -> None:
    (tmp_path / "src" / "agent_loco" / "templates").mkdir(parents=True)
    (tmp_path / "src" / "agent_loco" / "templates" / "index.html").write_text(
        "<html></html>\n", encoding="utf-8"
    )
    assert ui_review_needed(
        "Add a progress bar",
        "diff --git a/src/agent_loco/templates/index.html",
    )
    assert ui_review_needed("Add a progress bar", "", tmp_path)
    assert not ui_review_needed("update discord.py", "Pipfile.lock hashes changed")


def test_format_ui_evidence_lists_errors_and_tiny_controls() -> None:
    text = format_ui_evidence(
        UiEvidence(
            ok=False,
            url="http://127.0.0.1:9/",
            title="loco",
            page_errors=["label is not defined"],
            smashed=["Init (0x0)"],
            dead_controls=['[data-task-pane="screenshots"] did not show the screenshots pane'],
            snapshot="button: Queue task (120x32)",
            interactive=False,
        )
    )
    assert "label is not defined" in text
    assert "Init (0x0)" in text
    assert "Queue task" in text
    assert "did not show the screenshots pane" in text
    assert "static dump-dom" in text


def test_from_eval_marks_js_errors_and_zero_size_controls() -> None:
    evidence = _from_eval(
        "http://127.0.0.1:9/",
        {
            "title": "loco",
            "text": "Queue task",
            "errors": ["label is not defined"],
            "tabs": [
                {
                    "name": "Screenshots",
                    "pane": "screenshots",
                    "selected": False,
                    "panelHidden": True,
                }
            ],
            "elements": [
                {"tag": "button", "name": "Init", "w": 0, "h": 0},
                {"tag": "button", "name": "Queue task", "w": 120, "h": 32},
            ],
        },
        page_errors=[],
        console_errors=[],
        screenshot=None,
    )
    assert evidence.ok is False
    assert "label is not defined" in evidence.page_errors
    assert "Init (0x0)" in evidence.smashed
    assert "Queue task (120x32)" in evidence.snapshot
    assert "tab: Screenshots" in evidence.snapshot
    hidden = _from_eval(
        "http://127.0.0.1:9/",
        {
            "title": "loco",
            "text": "Queue task",
            "errors": [],
            "elements": [
                {"tag": "button", "name": "Init", "w": 0, "h": 0, "hidden": True},
                {"tag": "button", "name": "Queue task", "w": 120, "h": 32},
            ],
        },
        page_errors=[],
        console_errors=[],
        screenshot=None,
    )
    assert hidden.ok is True
    assert hidden.smashed == []
    crushed_tab = _from_eval(
        "http://127.0.0.1:9/",
        {
            "title": "loco",
            "text": "Past runs",
            "errors": [],
            "tabs": [
                {
                    "name": "Past runs",
                    "pane": "history",
                    "selected": True,
                    "panelHidden": False,
                    "panelHeight": 0,
                }
            ],
            "elements": [{"tag": "button", "name": "Past runs", "w": 90, "h": 32}],
        },
        page_errors=[],
        console_errors=[],
        screenshot=None,
    )
    assert crushed_tab.ok is False
    assert crushed_tab.dead_controls
    assert "Past runs" in crushed_tab.dead_controls[0]


def test_from_eval_marks_overlapping_rerun_and_line_stats() -> None:
    overlap = _from_eval(
        "http://127.0.0.1:9/",
        {
            "title": "loco",
            "text": "Past runs",
            "errors": [],
            "elements": [
                {
                    "tag": "button",
                    "name": "↻ Rerun",
                    "cls": "rerun-btn",
                    "w": 72,
                    "h": 24,
                    "x": 16,
                    "y": 400,
                },
                {
                    "tag": "span",
                    "name": "+12 −3",
                    "cls": "history-line-stats",
                    "w": 48,
                    "h": 16,
                    "x": 20,
                    "y": 404,
                },
            ],
        },
        page_errors=[],
        console_errors=[],
        screenshot=None,
    )
    assert overlap.ok is False
    assert any("overlaps" in item for item in overlap.smashed)

    separated = _from_eval(
        "http://127.0.0.1:9/",
        {
            "title": "loco",
            "text": "Past runs",
            "errors": [],
            "elements": [
                {
                    "tag": "button",
                    "name": "↻ Rerun",
                    "cls": "rerun-btn",
                    "w": 72,
                    "h": 24,
                    "x": 900,
                    "y": 400,
                },
                {
                    "tag": "span",
                    "name": "+12 −3",
                    "cls": "history-line-stats",
                    "w": 48,
                    "h": 16,
                    "x": 16,
                    "y": 400,
                },
            ],
        },
        page_errors=[],
        console_errors=[],
        screenshot=None,
    )
    assert separated.ok is True
    assert separated.smashed == []


def test_classify_network_failure_ignores_favicon_and_screenshots() -> None:
    assert classify_network_failure("http://127.0.0.1:9/favicon.ico", 404) == "ignore"
    assert classify_network_failure("http://127.0.0.1:9/apple-touch-icon.png", 404) == "ignore"
    assert (
        classify_network_failure("http://127.0.0.1:9/api/ui-screenshot?name=ui-review.png", 404)
        == "note"
    )
    assert classify_network_failure("http://127.0.0.1:9/static/app.js", 404) == "block"
    assert is_generic_resource_console(
        "Failed to load resource: the server responded with a status of 404 (Not Found)"
    )


def test_format_ui_evidence_keeps_optional_404s_out_of_js_errors() -> None:
    text = format_ui_evidence(
        UiEvidence(
            ok=True,
            url="http://127.0.0.1:9/",
            console_errors=[
                "Failed to load resource: the server responded with a status of 404 (Not Found)"
            ],
            network_notes=["404 http://127.0.0.1:9/api/ui-screenshot?name=ui-review.png"],
            snapshot="button: Queue task (120x32)",
        )
    )
    assert "Optional resources missing" in text
    assert "JavaScript errors:" not in text
    assert "ui-screenshot" in text


def test_review_goal_includes_rendered_ui_section() -> None:
    llm = ScriptedClient(
        [AssistantTurn(text='{"met": false, "reason": "progress stages have no labels"}')]
    )
    token = bind_progress()
    try:
        verdict = review_goal(
            llm,
            "Add a task stage progress bar",
            diff="--- a/index.html\n+++ b/index.html\n",
            summary="updated css",
            tests_passed=True,
            ui_evidence="Rendered UI:\nJavaScript errors:\n- label is not defined",
        )
    finally:
        reset_progress(token)
    assert verdict.met is False
    user = llm.calls[0][-1]["content"]
    assert "Rendered UI:" in user
    assert "label is not defined" in user


def test_review_goal_overrides_met_when_rendered_ui_throws(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_loco.runtime.improve.collect_ui_evidence",
        lambda *args, **kwargs: UiEvidence(
            ok=False,
            page_errors=["label is not defined"],
            smashed=["Init (0x0)"],
        ),
    )
    (tmp_path / ".loco").mkdir()
    (tmp_path / ".loco" / "config.yaml").write_text("name: fixture\n", encoding="utf-8")
    llm = ScriptedClient([AssistantTurn(text='{"met": true, "reason": "css layout looks fine"}')])
    token = bind_progress()
    try:
        verdict = _review_goal(
            Workspace(tmp_path),
            load_project(tmp_path),
            llm,
            "Add a progress bar",
            "diff --git a/src/agent_loco/templates/index.html",
            "updated css",
            True,
        )
    finally:
        reset_progress(token)
    assert verdict.met is False
    assert "label is not defined" in verdict.reason


def test_review_goal_does_not_override_met_for_favicon_or_screenshot_404s(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "agent_loco.runtime.improve.collect_ui_evidence",
        lambda *args, **kwargs: UiEvidence(
            ok=True,
            console_errors=[
                "Failed to load resource: the server responded with a status of 404 (Not Found)"
            ],
            network_notes=["404 http://127.0.0.1:9/favicon.ico"],
        ),
    )
    (tmp_path / ".loco").mkdir()
    (tmp_path / ".loco" / "config.yaml").write_text("name: fixture\n", encoding="utf-8")
    llm = ScriptedClient(
        [AssistantTurn(text='{"met": true, "reason": "pagination is at the top"}')]
    )
    token = bind_progress()
    try:
        verdict = _review_goal(
            Workspace(tmp_path),
            load_project(tmp_path),
            llm,
            "Add task pagination at the top",
            "diff --git a/src/agent_loco/templates/index.html",
            "added top pagination",
            True,
        )
    finally:
        reset_progress(token)
    assert verdict.met is True


def test_review_goal_overrides_met_when_required_script_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "agent_loco.runtime.improve.collect_ui_evidence",
        lambda *args, **kwargs: UiEvidence(
            ok=False,
            network_failures=["404 http://127.0.0.1:9/static/app.js"],
        ),
    )
    (tmp_path / ".loco").mkdir()
    (tmp_path / ".loco" / "config.yaml").write_text("name: fixture\n", encoding="utf-8")
    llm = ScriptedClient(
        [AssistantTurn(text='{"met": true, "reason": "template includes the script tag"}')]
    )
    token = bind_progress()
    try:
        verdict = _review_goal(
            Workspace(tmp_path),
            load_project(tmp_path),
            llm,
            "Add a progress bar",
            "diff --git a/src/agent_loco/templates/index.html",
            "added script",
            True,
        )
    finally:
        reset_progress(token)
    assert verdict.met is False
    assert "app.js" in verdict.reason


def test_review_goal_overrides_met_when_screenshot_tab_is_dead(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_loco.runtime.improve.collect_ui_evidence",
        lambda *args, **kwargs: UiEvidence(
            ok=False,
            dead_controls=['[data-task-pane="screenshots"] did not show the screenshots pane'],
        ),
    )
    (tmp_path / ".loco").mkdir()
    (tmp_path / ".loco" / "config.yaml").write_text("name: fixture\n", encoding="utf-8")
    llm = ScriptedClient(
        [AssistantTurn(text='{"met": true, "reason": "screenshots tab is in the template"}')]
    )
    token = bind_progress()
    try:
        verdict = _review_goal(
            Workspace(tmp_path),
            load_project(tmp_path),
            llm,
            "Show UI screenshots in another tab",
            "diff --git a/src/agent_loco/templates/index.html",
            "added screenshots tab markup",
            True,
        )
    finally:
        reset_progress(token)
    assert verdict.met is False
    assert "did not show the screenshots pane" in verdict.reason


def test_review_goal_overrides_met_when_dump_dom_cannot_click_tabs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "agent_loco.runtime.improve.collect_ui_evidence",
        lambda *args, **kwargs: UiEvidence(
            ok=True,
            interactive=False,
            notes="Captured with Chrome --dump-dom",
        ),
    )
    (tmp_path / ".loco").mkdir()
    (tmp_path / ".loco" / "config.yaml").write_text("name: fixture\n", encoding="utf-8")
    llm = ScriptedClient(
        [AssistantTurn(text='{"met": true, "reason": "the screenshots tab exists in html"}')]
    )
    token = bind_progress()
    try:
        verdict = _review_goal(
            Workspace(tmp_path),
            load_project(tmp_path),
            llm,
            "Add a Screenshots tab next to Changes",
            "diff --git a/src/agent_loco/templates/index.html",
            "added tab markup",
            True,
        )
    finally:
        reset_progress(token)
    assert verdict.met is False
    assert "dump-dom" in verdict.reason


def test_unverified_interactive_ui_requires_clicking_visible_screenshots_tab() -> None:
    reason = unverified_interactive_ui(
        "Show screenshots in a tab",
        UiEvidence(
            ok=True,
            snapshot="tab: Screenshots selected=False panelHidden=True",
            clicked=["#history-list button.task"],
        ),
    )
    assert reason is not None
    assert "Screenshots tab" in reason


def test_review_ui_tool_reports_capture(tmp_path: Path, monkeypatch) -> None:
    workspace = Workspace(tmp_path)
    (tmp_path / ".loco").mkdir()
    (tmp_path / ".loco" / "config.yaml").write_text("name: fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        "agent_loco.tools.browser.collect_ui_evidence",
        lambda *args, **kwargs: UiEvidence(
            ok=False,
            url="http://127.0.0.1:9/",
            page_errors=["label is not defined"],
            snapshot="progress-stage: Init (0x0)",
        ),
    )
    tools = build_tools(
        workspace,
        test_command=None,
        command_timeout_seconds=5,
        git_author_name=None,
        git_author_email=None,
    )
    result = execute_tool(tools, "review_ui", {})
    assert not result.ok
    assert "label is not defined" in result.output
    assert "Init (0x0)" in result.output


def test_agent_prompt_requires_review_ui_after_ui_edits() -> None:
    assert "review_ui" in SYSTEM_PROMPT
    assert "zero-size" in SYSTEM_PROMPT
    assert "Click new tabs" in SYSTEM_PROMPT
    assert "404" in SYSTEM_PROMPT
    assert "Finish the stated goal" in SYSTEM_PROMPT
    assert "later run can build on" not in SYSTEM_PROMPT
    assert "Markup for a Screenshots tab is not enough" in REVIEW_SYSTEM
    assert "0x0 or display:none" in REVIEW_SYSTEM
    assert "in a real implementation" in REVIEW_SYSTEM
    assert "owner/repo" in REVIEW_SYSTEM


def test_resolve_ui_screenshot_serves_unique_and_legacy_pngs(tmp_path: Path) -> None:
    loco = tmp_path / ".loco"
    shots = loco / "ui-screenshots"
    shots.mkdir(parents=True)
    unique = shots / "ui-review_goal_1.png"
    legacy = loco / "ui-review.png"
    unique.write_bytes(b"unique")
    legacy.write_bytes(b"legacy")
    assert resolve_ui_screenshot(tmp_path, "ui-review_goal_1.png") == unique.resolve()
    assert resolve_ui_screenshot(tmp_path, "ui-review.png") == legacy.resolve()
    assert resolve_ui_screenshot(tmp_path, "../config.yaml") is None
    assert resolve_ui_screenshot(tmp_path, "missing.png") is None


def test_playwright_probe_marks_dead_and_working_tabs() -> None:
    class Locator:
        def __init__(self, count: int) -> None:
            self._count = count
            self.first = self

        def count(self) -> int:
            return self._count

        def click(self, timeout: int = 0) -> None:
            return None

    class Page:
        def __init__(self, *, hidden: bool, selected: str) -> None:
            self.hidden = hidden
            self.selected = selected

        def locator(self, _selector: str) -> Locator:
            return Locator(1)

        def get_by_role(self, *_args, **_kwargs) -> Locator:
            return Locator(0)

        def wait_for_timeout(self, _ms: int) -> None:
            return None

        def evaluate(self, _script: str, _payload: object = None) -> dict[str, object]:
            return {
                "selected": self.selected,
                "hidden": self.hidden,
                "display": "none" if self.hidden else "block",
                "visibility": "hidden" if self.hidden else "visible",
                "height": 0 if self.hidden else 120,
                "textLen": 0 if self.hidden else 40,
            }

    clicked, dead = _playwright_probe(
        Page(hidden=True, selected="false"),
        ['[data-task-pane="screenshots"]'],
        500,
    )
    assert clicked == ['[data-task-pane="screenshots"]']
    assert dead and "screenshots" in dead[0]

    clicked, dead = _playwright_probe(
        Page(hidden=False, selected="true"),
        ['[data-task-pane="screenshots"]'],
        500,
    )
    assert clicked
    assert dead == []

    clicked, dead = _playwright_probe(
        Page(hidden=False, selected="true"),
        ['[data-main-pane="history"]'],
        500,
    )
    assert clicked == ['[data-main-pane="history"]']
    assert dead == []


def test_playwright_probe_fails_when_pane_is_css_hidden() -> None:
    class Locator:
        def __init__(self) -> None:
            self.first = self

        def count(self) -> int:
            return 1

        def click(self, timeout: int = 0) -> None:
            return None

    class Page:
        def locator(self, _selector: str) -> Locator:
            return Locator()

        def get_by_role(self, *_args, **_kwargs) -> Locator:
            return Locator()

        def wait_for_timeout(self, _ms: int) -> None:
            return None

        def evaluate(self, _script: str, _payload: object = None) -> dict[str, object]:
            return {
                "selected": "true",
                "hidden": False,
                "display": "none",
                "visibility": "visible",
                "height": 0,
                "textLen": 0,
            }

    clicked, dead = _playwright_probe(
        Page(),
        ['[data-main-pane="history"]'],
        500,
    )
    assert clicked == ['[data-main-pane="history"]']
    assert dead and "history" in dead[0]
    assert "display=none" in dead[0]


def test_loco_review_clicks_include_main_and_task_tabs(tmp_path: Path) -> None:
    from agent_loco.runtime.uireview import _review_clicks

    (tmp_path / "src" / "agent_loco" / "web_ui.py").parent.mkdir(parents=True)
    (tmp_path / "src" / "agent_loco" / "web_ui.py").write_text("# loco\n", encoding="utf-8")
    clicks = _review_clicks(tmp_path, '[data-task-pane="progress"]')
    assert '[data-main-pane="history"]' in clicks
    assert '[data-main-pane="current"]' in clicks
    assert '[data-task-pane="screenshots"]' in clicks
    assert '[data-task-pane="progress"]' in clicks
    assert clicks.index('[data-main-pane="history"]') < clicks.index("#open-history-picker")
    assert clicks.index("#open-history-picker") < clicks.index("#history-list button.task")
    assert clicks.index("#history-list button.task") < clicks.index('[data-task-pane="changes"]')
    assert clicks.index('[data-task-pane="screenshots"]') < clicks.index(
        '[data-main-pane="current"]'
    )


def test_loco_review_clicks_open_goal_panels_last(tmp_path: Path) -> None:
    from agent_loco.runtime.uireview import _review_clicks

    (tmp_path / "src" / "agent_loco" / "web_ui.py").parent.mkdir(parents=True)
    (tmp_path / "src" / "agent_loco" / "web_ui.py").write_text("# loco\n", encoding="utf-8")
    clicks = _review_clicks(
        tmp_path,
        None,
        "For overall user experience move the skills out of the settings cog",
    )
    assert clicks[-1] == "#open-skills"
    assert "#open-settings" not in clicks
    assert '[data-main-pane="current"]' in clicks
    assert clicks.index('[data-main-pane="current"]') < clicks.index("#open-skills")

    rule_clicks = _review_clicks(tmp_path, None, "Move rules into the left pane")
    assert rule_clicks[-1] == "#open-guidelines"


def test_unverified_interactive_ui_requires_skills_click() -> None:
    evidence = UiEvidence(
        ok=True,
        interactive=True,
        clicked=['[data-main-pane="current"]', "#task-list button.task"],
        snapshot="overlay: skills-panel hidden=True 0x0",
    )
    reason = unverified_interactive_ui(
        "Move the skills out of the settings cog into the left pane",
        evidence,
    )
    assert reason and "Skills" in reason


def test_unverified_interactive_ui_ignores_panels_the_page_does_not_have() -> None:
    skills = unverified_interactive_ui(
        "Enable workspace skills from the repo",
        UiEvidence(ok=True, interactive=True, snapshot="div: settings (200x40)"),
    )
    rules = unverified_interactive_ui(
        "Move guidelines into a settings menu",
        UiEvidence(ok=True, interactive=True, snapshot="div: settings (200x40)"),
    )
    assert skills is None
    assert rules is None


def test_unverified_interactive_ui_requires_rules_click_when_panel_is_present() -> None:
    reason = unverified_interactive_ui(
        "Move guidelines into a settings menu",
        UiEvidence(
            ok=True,
            interactive=True,
            snapshot="overlay: guidelines-panel hidden=True 0x0",
            clicked=['[data-main-pane="current"]'],
        ),
    )
    assert reason and "Rules" in reason


def test_playwright_probe_marks_dead_aria_controls_panel() -> None:
    class Locator:
        def __init__(self) -> None:
            self.first = self

        def count(self) -> int:
            return 1

        def click(self, timeout: int = 0) -> None:
            return None

    class Page:
        def locator(self, _selector: str) -> Locator:
            return Locator()

        def get_by_role(self, *_args, **_kwargs) -> Locator:
            return Locator()

        def wait_for_timeout(self, _ms: int) -> None:
            return None

        def evaluate(self, script: str, payload: object = None) -> object:
            if "aria-controls" in script and "getElementById" not in script:
                return "skills-panel"
            return {
                "missing": False,
                "expanded": "true",
                "hidden": True,
                "display": "none",
                "visibility": "hidden",
                "height": 0,
                "textLen": 0,
            }

    clicked, dead = _playwright_probe(Page(), ["#open-skills"], 500)
    assert clicked == ["#open-skills"]
    assert dead and "skills-panel" in dead[0]


def test_review_ui_tool_passes_cycle_goal(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_collect(*args, **kwargs):
        captured["goal"] = kwargs.get("goal") or (args[2] if len(args) > 2 else None)
        return UiEvidence(ok=True, snapshot="overlay: skills-panel hidden=False 320x640")

    monkeypatch.setattr("agent_loco.tools.browser.collect_ui_evidence", fake_collect)
    workspace = Workspace(tmp_path)
    (tmp_path / ".loco").mkdir()
    (tmp_path / ".loco" / "config.yaml").write_text("name: fixture\n", encoding="utf-8")
    tools = build_tools(
        workspace,
        test_command=None,
        command_timeout_seconds=5,
        git_author_name=None,
        git_author_email=None,
        goal="Move the skills into the sidebar",
    )
    result = execute_tool(tools, "review_ui", {})
    assert result.ok
    assert captured["goal"] == "Move the skills into the sidebar"


def test_unverified_interactive_ui_requires_main_tab_clicks() -> None:
    evidence = UiEvidence(
        ok=True,
        interactive=True,
        clicked=['[data-task-pane="changes"]'],
        snapshot="tab: Past runs selected=false panelHidden=True",
    )
    reason = unverified_interactive_ui(
        "Move current tasks and past runs onto their own tabs",
        evidence,
    )
    assert reason and "Current/Past runs" in reason


def test_pr_screenshot_names_keeps_final_successful_change() -> None:
    token = bind_progress()
    try:
        record_event(
            kind="ui",
            ok=True,
            screenshot_filename="ui-review_review-ui_20260920_1.png",
        )
        record_event(
            kind="ui",
            ok=False,
            screenshot_filename="ui-review_goal_fail.png",
        )
        record_event(
            kind="ui",
            ok=True,
            screenshot_filename="ui-review_goal_a.png",
        )
        record_event(
            kind="ui",
            ok=True,
            screenshot_filename="ui-review_goal_b.png",
        )
        record_event(
            kind="ui",
            ok=True,
            screenshot_filename="ui-review_review-ui_20260920_2.png",
        )
        assert _pr_screenshot_names() == ["ui-review_goal_b.png"]
    finally:
        reset_progress(token)


def test_pr_screenshot_names_falls_back_to_last_successful_review() -> None:
    token = bind_progress()
    try:
        record_event(
            kind="ui",
            ok=True,
            screenshot_filename="ui-review_review-ui_one.png",
        )
        record_event(
            kind="ui",
            ok=False,
            screenshot_filename="ui-review_review-ui_bad.png",
        )
        record_event(
            kind="ui",
            ok=True,
            screenshot_filename="ui-review_review-ui_two.png",
        )
        assert _pr_screenshot_names() == ["ui-review_review-ui_two.png"]
    finally:
        reset_progress(token)


def test_pr_body_embeds_hosted_screenshot_urls() -> None:
    body = _pr_body(
        "Show screenshots",
        "added tab",
        tests_passed=True,
        commit_sha="abc",
        branch="loco/x",
        screenshots=["ui-review_goal.png", "ui-review.png"],
        image_base="https://github.com/acme/repo/raw/abc",
    )
    assert (
        "![ui-review_goal.png](https://github.com/acme/repo/raw/abc/"
        ".loco/ui-screenshots/ui-review_goal.png)"
    ) in body
    assert "![ui-review.png](https://github.com/acme/repo/raw/abc/.loco/ui-review.png)" in body
    assert "](.loco/ui-screenshots/" not in body


def test_pr_body_omits_unhosted_screenshots() -> None:
    body = _pr_body(
        "Show screenshots",
        "added tab",
        tests_passed=True,
        commit_sha="abc",
        branch="loco/x",
        screenshots=["ui-review_goal.png"],
    )
    assert "## Screenshots" not in body
    assert ".loco/ui-screenshots" not in body


def test_load_project_reads_preview_command(tmp_path: Path) -> None:
    loco = tmp_path / ".loco"
    loco.mkdir()
    (loco / "config.yaml").write_text(
        "preview_command: python3 -m http.server {port}\n",
        encoding="utf-8",
    )
    project = load_project(tmp_path)
    assert project.preview_command == "python3 -m http.server {port}"


def test_loco_template_paths_use_the_running_app(tmp_path: Path, monkeypatch) -> None:
    from agent_loco.runtime.project import load_project
    from agent_loco.runtime.uireview import _is_loco_template_path, start_preview

    (tmp_path / "src" / "agent_loco" / "templates").mkdir(parents=True)
    (tmp_path / "src" / "agent_loco" / "web_ui.py").write_text("# loco\n", encoding="utf-8")
    template = tmp_path / "src" / "agent_loco" / "templates" / "index.html"
    template.write_text("<html>{{ ws.name }}</html>\n", encoding="utf-8")
    workspace = Workspace(tmp_path)
    assert _is_loco_template_path(workspace, "src/agent_loco/templates/index.html")
    calls: list[str] = []

    def fake_loco(ws, port):
        calls.append(f"loco:{port}")

        class Preview:
            url = f"http://127.0.0.1:{port}/"

            def close(self) -> None:
                return None

        return Preview()

    monkeypatch.setattr("agent_loco.runtime.uireview._start_loco_preview", fake_loco)
    preview = start_preview(
        workspace,
        load_project(tmp_path),
        path="src/agent_loco/templates/index.html",
        port=9,
    )
    assert calls == ["loco:9"]
    assert preview is not None
    assert preview.url == "http://127.0.0.1:9/"


def test_review_goal_keeps_met_when_iteration_limit_but_ui_verified(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "agent_loco.runtime.improve.collect_ui_evidence",
        lambda *args, **kwargs: UiEvidence(
            ok=True,
            snapshot="button: ↻ Rerun (72x24 @ 900,400)\nspan: +12 −3 (48x16 @ 16,400)",
            screenshot="ok.png",
            clicked=['[data-main-pane="history"]'],
        ),
    )
    (tmp_path / ".loco").mkdir()
    (tmp_path / ".loco" / "config.yaml").write_text("name: fixture\n", encoding="utf-8")
    llm = ScriptedClient(
        [AssistantTurn(text='{"met": true, "reason": "rerun no longer overlaps +/- stats"}')]
    )
    token = bind_progress()
    try:
        verdict = _review_goal(
            Workspace(tmp_path),
            load_project(tmp_path),
            llm,
            "Fix the overlapping of the past runs rerun button",
            "diff --git a/src/agent_loco/templates/index.html",
            "Stopped after reaching the iteration limit.",
            True,
            stopped_reason="max_iterations",
        )
    finally:
        reset_progress(token)
    assert verdict.met is True

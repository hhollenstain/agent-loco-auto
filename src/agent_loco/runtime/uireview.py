from __future__ import annotations

import html
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from agent_loco.progress import record_event
from agent_loco.runtime.project import ProjectConfig
from agent_loco.sandbox import Workspace

log = logging.getLogger("loco")

UI_FILE_RE = re.compile(
    r"(?:^|[/\s])(?:[\w./-]+\.(?:html?|css|js|jsx|tsx|vue|svelte)|templates/)",
    re.IGNORECASE,
)
UI_GOAL_RE = re.compile(
    r"\b(ui|ux|css|html|layout|sidebar|progress bar|template|frontend|"
    r"web page|web-ui|web ui|button|panel|dark mode|responsive|"
    r"screenshot|screen shot|tab)\b",
    re.IGNORECASE,
)
INTERACTIVE_UI_RE = re.compile(
    r"\b(tab|tabs|screenshot|screen shot|panel|modal|button|"
    r"skill|skills|rules?|guidelines?)\b",
    re.IGNORECASE,
)
MAX_SNAPSHOT_CHARS = 6_000
IGNORABLE_RESOURCE_RE = re.compile(
    r"(?:^|/)(?:favicon\.ico|apple-touch-icon[^/]*|site\.webmanifest|manifest\.json)$",
    re.IGNORECASE,
)
SCREENSHOT_RESOURCE_RE = re.compile(r"/api/ui-screenshot(?:/|$)", re.IGNORECASE)
BLOCKING_RESOURCE_RE = re.compile(r"\.(?:js|mjs|cjs|css)$", re.IGNORECASE)
GENERIC_RESOURCE_CONSOLE_RE = re.compile(
    r"failed to load resource|net::err_|status of 404|file not found",
    re.IGNORECASE,
)
PAGE_EVAL = """() => {
  const nodes = [...document.querySelectorAll(
    'button, a, [role="tab"], [role="button"], h1, h2, h3, label, input,'
    + ' select, textarea, .progress-stage, .task, .task-card,'
    + ' .rerun-btn, .history-line-stats, .pr-link, .sidebar-overlay, .sidebar-tools button'
  )];
  const isHidden = (el) => {
    if (!el) return true;
    if (el.closest("[hidden]")) return true;
    const style = window.getComputedStyle(el);
    return style.display === "none" || style.visibility === "hidden";
  };
    return {
    title: document.title || "",
    url: location.href,
    text: (document.body && document.body.innerText || "").slice(0, 8000),
    errors: window.__locoErrors || [],
    tabs: [...document.querySelectorAll("[data-main-pane].main-tab, [data-task-pane]")]
      .filter((el) => !el.closest("[hidden]"))
      .slice(0, 16)
      .map((el) => {
      const main = el.getAttribute("data-main-pane") || "";
      const pane = el.getAttribute("data-task-pane") || "";
      const panel = main
        ? document.querySelector('.main-pane[data-main-pane="' + main + '"]')
        : (pane ? document.querySelector('[data-pane="' + pane + '"]') : null);
      const style = panel ? window.getComputedStyle(panel) : null;
      const box = panel ? panel.getBoundingClientRect() : null;
      const hidden = !panel || isHidden(panel) || (
        style && (style.display === "none" || style.visibility === "hidden")
      );
      return {
        name: (el.innerText || "").trim().slice(0, 80),
        pane: main || pane,
        selected: el.getAttribute("aria-selected") === "true",
        panelHidden: hidden,
        panelHeight: box ? Math.round(box.height) : 0,
        panelText: panel ? (panel.innerText || "").trim().slice(0, 80) : "",
      };
    }),
    overlays: [...document.querySelectorAll(".sidebar-overlay, .settings-panel")]
      .slice(0, 8)
      .map((el) => {
      const style = window.getComputedStyle(el);
      const box = el.getBoundingClientRect();
      const hidden = isHidden(el) || style.display === "none" || style.visibility === "hidden";
      return {
        id: el.id || "",
        hidden,
        w: Math.round(box.width),
        h: Math.round(box.height),
        text: (el.innerText || "").trim().slice(0, 80),
      };
    }),
    elements: nodes.slice(0, 80).map((el) => {
      const box = el.getBoundingClientRect();
      const name = (
        el.getAttribute("aria-label")
        || el.innerText
        || el.getAttribute("title")
        || el.id
        || ""
      ).trim().slice(0, 80);
      return {
        tag: el.tagName.toLowerCase(),
        name,
        cls: (typeof el.className === "string" ? el.className : "").slice(0, 80),
        hidden: isHidden(el),
        w: Math.round(box.width),
        h: Math.round(box.height),
        x: Math.round(box.x),
        y: Math.round(box.y),
      };
    }),
  };
}"""
INIT_SCRIPT = """
window.__locoErrors = [];
window.addEventListener("error", (event) => {
  window.__locoErrors.push(String(event.message || event.error || "error"));
});
window.addEventListener("unhandledrejection", (event) => {
  window.__locoErrors.push(String(event.reason || "unhandledrejection"));
});
"""


@dataclass
class UiEvidence:
    ok: bool
    url: str = ""
    title: str = ""
    page_errors: list[str] = field(default_factory=list)
    console_errors: list[str] = field(default_factory=list)
    network_failures: list[str] = field(default_factory=list)
    network_notes: list[str] = field(default_factory=list)
    smashed: list[str] = field(default_factory=list)
    dead_controls: list[str] = field(default_factory=list)
    clicked: list[str] = field(default_factory=list)
    snapshot: str = ""
    screenshot: str | None = None
    screenshot_filename: str | None = None
    notes: str = ""
    interactive: bool = True

    @property
    def blocking_errors(self) -> list[str]:
        items = []
        for item in [*self.page_errors, *self.console_errors, *self.network_failures]:
            text = str(item).strip()
            if not text or is_generic_resource_console(text):
                continue
            items.append(text)
        return items

    @property
    def summary(self) -> str:
        if not self.ok and self.notes:
            return f"UI review skipped: {self.notes}"
        errors = self.blocking_errors
        bits = ["Rendered UI"]
        if self.title:
            bits.append(self.title)
        if errors:
            bits.append(f"{len(errors)} JS error(s)")
        if self.smashed:
            bits.append(f"{len(self.smashed)} unreadable control(s)")
        if self.dead_controls:
            bits.append(f"{len(self.dead_controls)} dead control(s)")
        if not self.interactive:
            bits.append("static capture, clicks unverified")
        if not errors and not self.smashed and not self.dead_controls and self.interactive:
            bits.append("no console errors")
        return " · ".join(bits)


def ui_review_needed(goal: str, diff: str = "", workspace: Path | None = None) -> bool:
    goal_is_ui = bool(UI_GOAL_RE.search(goal or ""))
    diff_is_ui = bool(UI_FILE_RE.search(diff or ""))
    if diff_is_ui:
        return True
    if goal_is_ui and workspace and _has_web_preview(workspace):
        return True
    return False


def format_ui_evidence(evidence: UiEvidence | None) -> str:
    if evidence is None:
        return ""
    lines = ["Rendered UI:"]
    if evidence.url:
        lines.append(f"URL: {evidence.url}")
    if evidence.title:
        lines.append(f"Title: {evidence.title}")
    if evidence.notes:
        lines.append(evidence.notes)
    js_errors = [
        item
        for item in [*evidence.page_errors, *evidence.console_errors]
        if item and not is_generic_resource_console(item)
    ]
    if js_errors:
        lines.append("JavaScript errors:")
        lines.extend(f"- {item}" for item in js_errors[:12])
    if evidence.network_failures:
        lines.append("Missing page resources:")
        lines.extend(f"- {item}" for item in evidence.network_failures[:12])
    if evidence.network_notes:
        lines.append("Optional resources missing (not a review failure):")
        lines.extend(f"- {item}" for item in evidence.network_notes[:12])
    if evidence.smashed:
        lines.append("Controls with no usable size (hidden, crushed, or off-screen):")
        lines.extend(f"- {item}" for item in evidence.smashed[:12])
    if evidence.dead_controls:
        lines.append("Controls that did not reveal their panel when clicked:")
        lines.extend(f"- {item}" for item in evidence.dead_controls[:12])
    if not evidence.interactive:
        lines.append(
            "Capture could not click controls (static dump-dom). "
            "New tabs, panels, and buttons are unverified."
        )
    if evidence.clicked:
        lines.append("Clicked: " + ", ".join(evidence.clicked[:8]))
    if evidence.snapshot:
        lines.append("Visible page:")
        lines.append(evidence.snapshot.strip())
    if evidence.screenshot:
        lines.append(f"Screenshot: {evidence.screenshot}")
    if evidence.screenshot_filename:
        lines.append(f"Screenshot (for PRs): ui-screenshots/{evidence.screenshot_filename}")
    return "\n".join(lines)


def resolve_ui_screenshot(workspace: Path, name: str) -> Path | None:
    filename = Path(str(name or "").strip()).name
    if not filename or filename in {".", ".."}:
        return None
    if "/" in filename or "\\" in filename:
        return None
    if not filename.lower().endswith(".png"):
        return None
    root = Path(workspace).expanduser().resolve()
    loco = (root / ".loco").resolve()
    for candidate in (loco / "ui-screenshots" / filename, loco / filename):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if loco not in resolved.parents and resolved.parent != loco:
            continue
        if resolved.is_file():
            return resolved
    return None


def collect_ui_evidence(
    workspace: Workspace,
    project: ProjectConfig,
    goal: str,
    *,
    url: str | None = None,
    path: str | None = None,
    click: str | None = None,
    wait_ms: int = 4000,
) -> UiEvidence:
    screenshot_dir = workspace.root / ".loco" / "ui-screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    goal_slug = re.sub(r"\W+", "-", goal.strip().lower())[:40] or "untitled"
    screenshot_name = f"ui-review_{goal_slug}_{timestamp}_{uuid.uuid4().hex[:6]}.png"
    screenshot = screenshot_dir / screenshot_name
    preview: _Preview | None = None
    target = (url or "").strip()
    try:
        if not target:
            preview = start_preview(workspace, project, path=path)
            if preview is None:
                return UiEvidence(
                    ok=False,
                    notes=(
                        "No preview server for this workspace. "
                        "Set preview_command in .loco/config.yaml."
                    ),
                )
            target = preview.url
        page: UiEvidence = capture_page(
            target,
            clicks=_review_clicks(workspace.root, click, goal),
            screenshot=screenshot,
            wait_ms=wait_ms,
        )
        if screenshot.exists():
            page.screenshot = str(screenshot)
            page.screenshot_filename = screenshot.name
        elif page.screenshot:
            page.screenshot_filename = page.screenshot_filename or Path(page.screenshot).name
        record_event(
            kind="ui",
            ok=page.ok,
            url=page.url or target,
            message=page.summary,
            snapshot=page.snapshot,
            screenshot=page.screenshot,
            screenshot_filename=page.screenshot_filename,
            errors=page.blocking_errors,
        )
        return page
    except Exception as exc:  # noqa: BLE001 - capture failures become evidence
        return UiEvidence(ok=False, notes=f"could not render UI: {exc}")
    finally:
        if preview is not None:
            preview.close()


def start_preview(
    workspace: Workspace,
    project: ProjectConfig,
    *,
    path: str | None = None,
    port: int | None = None,
) -> _Preview | None:
    host_port = port or _free_port()
    command = getattr(project, "preview_command", None)
    if command:
        return _start_command_preview(workspace, str(command), host_port)
    # Never serve agent-loco Jinja templates as static HTML. That captures
    # `{{ ws.name }}` and 404s /static CSS instead of the running app.
    if _is_loco_project(workspace.root) and (not path or _is_loco_template_path(workspace, path)):
        return _start_loco_preview(workspace, host_port)
    if path:
        html_path = workspace.resolve(path)
        if html_path.is_file():
            return _start_static_preview(html_path.parent, host_port, html_path.name)
    html = _first_html_file(workspace.root)
    if html is not None:
        return _start_static_preview(html.parent, host_port, html.name)
    return None


def capture_page(
    url: str,
    *,
    click: str | None = None,
    clicks: list[str] | None = None,
    screenshot: Path | None = None,
    wait_ms: int = 4000,
) -> UiEvidence:
    selectors = [item for item in [click, *(clicks or [])] if item]
    try:
        return _playwright_capture(url, clicks=selectors, screenshot=screenshot, wait_ms=wait_ms)
    except Exception as exc:
        log.info("playwright UI capture unavailable: %s", exc)
    try:
        return _chrome_capture(url, screenshot=screenshot, wait_ms=wait_ms)
    except Exception as exc:
        return UiEvidence(ok=False, url=url, notes=f"could not render UI: {exc}")


def _playwright_capture(
    url: str,
    *,
    clicks: list[str],
    screenshot: Path | None,
    wait_ms: int,
) -> UiEvidence:
    from playwright.sync_api import sync_playwright

    console: list[str] = []
    page_errors: list[str] = []
    network_failures: list[str] = []
    network_notes: list[str] = []

    def on_console(msg: Any) -> None:
        if getattr(msg, "type", None) != "error":
            return
        text = str(getattr(msg, "text", "") or "")
        if is_generic_resource_console(text):
            return
        console.append(text)

    def on_response(response: Any) -> None:
        try:
            status = int(response.status)
        except (TypeError, ValueError):
            return
        if status < 400:
            return
        record_network_failure(
            str(getattr(response, "url", "") or ""),
            status,
            network_failures,
            network_notes,
        )

    with sync_playwright() as playwright:
        browser = _launch_playwright(playwright)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.add_init_script(INIT_SCRIPT)
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on("console", on_console)
        page.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded", timeout=max(wait_ms, 8000))
        page.wait_for_timeout(min(wait_ms, 2500))
        clicked, dead = _playwright_probe(page, clicks, wait_ms)
        raw = page.evaluate(PAGE_EVAL)
        shot = None
        if screenshot is not None:
            page.screenshot(path=str(screenshot), full_page=False)
            shot = str(screenshot)
        browser.close()
    return _from_eval(
        url,
        raw,
        page_errors=page_errors,
        console_errors=console,
        network_failures=network_failures,
        network_notes=network_notes,
        screenshot=shot,
        clicked=clicked,
        dead_controls=dead,
        interactive=True,
    )


def _launch_playwright(playwright: Any) -> Any:
    errors: list[str] = []
    for kwargs in ({"channel": "chrome"}, {"channel": "chromium"}, {}):
        try:
            return playwright.chromium.launch(headless=True, **kwargs)
        except Exception as exc:  # noqa: BLE001 - try the next browser backend
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors) or "playwright could not launch chromium")


PANEL_STATE_EVAL = """({ attr, name }) => {
  const tab = document.querySelector('[' + attr + '="' + name + '"]');
  const panel = attr === "data-main-pane"
    ? document.querySelector('.main-pane[data-main-pane="' + name + '"]')
    : (document.querySelector('.main-pane:not([hidden]) [data-pane="' + name + '"]')
       || document.querySelector('[data-pane="' + name + '"]'));
  if (!panel) {
    return {
      selected: tab ? tab.getAttribute("aria-selected") : null,
      hidden: true,
      display: "none",
      height: 0,
      textLen: 0,
    };
  }
  const style = window.getComputedStyle(panel);
  const box = panel.getBoundingClientRect();
  const ancestorHidden = Boolean(panel.closest("[hidden]"));
  return {
    selected: tab ? tab.getAttribute("aria-selected") : null,
    hidden: Boolean(panel.hidden) || ancestorHidden,
    display: style.display,
    visibility: style.visibility,
    height: box.height,
    textLen: (panel.innerText || "").trim().length,
  };
}"""


ARIA_CONTROLS_EVAL = """(sel) => {
  const el = document.querySelector(sel);
  if (!el) return "";
  return el.getAttribute("aria-controls") || "";
}"""

ARIA_PANEL_EVAL = """({ id }) => {
  const panel = document.getElementById(id);
  const trigger = document.querySelector('[aria-controls="' + id + '"]');
  if (!panel) {
    return {
      missing: true,
      expanded: trigger ? trigger.getAttribute("aria-expanded") : null,
      hidden: true,
      display: "none",
      height: 0,
      textLen: 0,
    };
  }
  const style = window.getComputedStyle(panel);
  const box = panel.getBoundingClientRect();
  const ancestorHidden = Boolean(panel.closest("[hidden]"));
  return {
    missing: false,
    expanded: trigger ? trigger.getAttribute("aria-expanded") : null,
    hidden: Boolean(panel.hidden) || ancestorHidden,
    display: style.display,
    visibility: style.visibility,
    height: box.height,
    textLen: (panel.innerText || "").trim().length,
  };
}"""


def _playwright_click(page: Any, click: str, wait_ms: int) -> None:
    timeout = min(max(wait_ms, 500), 8000)
    locators = [page.locator(click), page.get_by_role("button", name=click)]
    for locator in locators:
        try:
            if locator.count() > 0:
                locator.first.click(timeout=timeout)
                return
        except Exception:  # noqa: BLE001 - try the next locator
            continue


def _playwright_has(page: Any, click: str) -> bool:
    for probe in (
        lambda: page.locator(click).count() > 0,
        lambda: page.get_by_role("button", name=click).count() > 0,
    ):
        try:
            if probe():
                return True
        except Exception:  # noqa: BLE001 - try the next locator
            continue
    return False


def _tab_target(selector: str) -> tuple[str, str] | None:
    for attr in ("data-main-pane", "data-task-pane"):
        match = re.search(rf'{re.escape(attr)}=["\']([^"\']+)', selector)
        if match:
            return attr, match.group(1)
    return None


def _pane_from_selector(selector: str) -> str | None:
    target = _tab_target(selector)
    return target[1] if target else None


def _aria_controls_id(page: Any, selector: str) -> str | None:
    try:
        value = page.evaluate(ARIA_CONTROLS_EVAL, selector)
    except Exception:  # noqa: BLE001 - missing locator is not a probe failure
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _playwright_probe(page: Any, clicks: list[str], wait_ms: int) -> tuple[list[str], list[str]]:
    clicked: list[str] = []
    dead: list[str] = []
    for selector in clicks:
        if not _playwright_has(page, selector):
            continue
        _playwright_click(page, selector, wait_ms)
        page.wait_for_timeout(400)
        clicked.append(selector)
        target = _tab_target(selector)
        if target:
            attr, pane = target
            state = page.evaluate(PANEL_STATE_EVAL, {"attr": attr, "name": pane})
            if isinstance(state, dict):
                visible = (
                    state.get("selected") == "true"
                    and state.get("hidden") is not True
                    and str(state.get("display") or "") not in {"none"}
                    and str(state.get("visibility") or "visible") != "hidden"
                    and float(state.get("height") or 0) >= 8
                    and int(state.get("textLen") or 0) > 0
                )
                if not visible:
                    dead.append(
                        f"{selector} did not show the {pane} pane "
                        f"(aria-selected={state.get('selected')}, hidden={state.get('hidden')}, "
                        f"display={state.get('display')}, height={state.get('height')}, "
                        f"textLen={state.get('textLen')})"
                    )
        panel_id = _aria_controls_id(page, selector)
        if not panel_id:
            continue
        state = page.evaluate(ARIA_PANEL_EVAL, {"id": panel_id})
        if not isinstance(state, dict):
            continue
        expanded = str(state.get("expanded") or "")
        visible = (
            expanded != "false"
            and state.get("missing") is not True
            and state.get("hidden") is not True
            and str(state.get("display") or "") not in {"none"}
            and str(state.get("visibility") or "visible") != "hidden"
            and float(state.get("height") or 0) >= 8
            and int(state.get("textLen") or 0) > 0
        )
        if not visible:
            dead.append(
                f"{selector} did not show #{panel_id} "
                f"(aria-expanded={state.get('expanded')}, hidden={state.get('hidden')}, "
                f"display={state.get('display')}, height={state.get('height')}, "
                f"textLen={state.get('textLen')})"
            )
    return clicked, dead


def _chrome_capture(url: str, *, screenshot: Path | None, wait_ms: int) -> UiEvidence:
    chrome = _chrome_executable()
    if not chrome:
        raise RuntimeError("Chrome/Chromium not found; pip install playwright")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--disable-dev-shm-usage",
        "--window-size=1280,800",
        f"--virtual-time-budget={max(wait_ms, 1000)}",
        "--dump-dom",
        url,
    ]
    if screenshot is not None:
        cmd.insert(-1, f"--screenshot={screenshot}")
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(wait_ms / 1000 + 8, 12),
    )
    if result.returncode not in {0, None}:
        raise RuntimeError(result.stderr.strip() or f"chrome exited {result.returncode}")
    text = _visible_text(result.stdout)
    errors = _chrome_logged_errors(result.stderr)
    smashed: list[str] = []
    if not text.strip():
        smashed.append("page body was empty after render")
    return UiEvidence(
        ok=not errors,
        url=url,
        page_errors=errors,
        snapshot=text[:MAX_SNAPSHOT_CHARS],
        screenshot=str(screenshot) if screenshot and screenshot.exists() else None,
        notes="Captured with Chrome --dump-dom (console coverage is limited).",
        smashed=smashed,
        interactive=False,
    )


def classify_network_failure(url: str, status: int | str | None = None) -> str:
    """Classify a failed request as ignore, note, or block.

    Favicon and touch-icon 404s are noise. Historical screenshot files are
    optional. Missing JS/CSS the page requested is a real render failure.
    """
    path = urlparse(str(url or "")).path
    if IGNORABLE_RESOURCE_RE.search(path):
        return "ignore"
    if SCREENSHOT_RESOURCE_RE.search(path):
        return "note"
    if BLOCKING_RESOURCE_RE.search(path):
        return "block"
    return "note"


def is_generic_resource_console(message: str) -> bool:
    return bool(GENERIC_RESOURCE_CONSOLE_RE.search(message or ""))


def record_network_failure(
    url: str,
    status: int | str | None,
    failures: list[str],
    notes: list[str],
) -> None:
    kind = classify_network_failure(url, status)
    if kind == "ignore":
        return
    code = str(status).strip() if status is not None else "failed"
    line = f"{code} {url}".strip()
    if not line or line in failures or line in notes:
        return
    if kind == "block":
        failures.append(line)
    else:
        notes.append(line)


def _from_eval(
    url: str,
    raw: Any,
    *,
    page_errors: list[str],
    console_errors: list[str],
    screenshot: str | None,
    clicked: list[str] | None = None,
    dead_controls: list[str] | None = None,
    interactive: bool = True,
    network_failures: list[str] | None = None,
    network_notes: list[str] | None = None,
) -> UiEvidence:
    data = raw if isinstance(raw, dict) else {}
    elements = data.get("elements") if isinstance(data.get("elements"), list) else []
    tabs = data.get("tabs") if isinstance(data.get("tabs"), list) else []
    smashed = []
    for item in elements:
        if not isinstance(item, dict):
            continue
        if item.get("hidden"):
            continue
        name = str(item.get("name") or item.get("tag") or "control").strip()
        width = int(item.get("w") or 0)
        height = int(item.get("h") or 0)
        if name and width < 8 and height < 8:
            smashed.append(f"{name} ({width}x{height})")
    overlays = data.get("overlays") if isinstance(data.get("overlays"), list) else []
    smashed.extend(_overlapping_controls(elements))
    injected = data.get("errors") if isinstance(data.get("errors"), list) else []
    errors = [str(item) for item in [*page_errors, *injected] if str(item).strip()]
    text = str(data.get("text") or "")
    dead = list(dead_controls or [])
    for item in tabs:
        if not isinstance(item, dict) or not item.get("selected"):
            continue
        hidden = bool(item.get("panelHidden"))
        height = item.get("panelHeight")
        short = height is not None and int(height or 0) < 8
        if hidden or short:
            name = str(item.get("name") or item.get("pane") or "tab").strip()
            dead.append(
                f"selected tab {name} did not render its pane "
                f"(panelHidden={item.get('panelHidden')} height={height})"
            )
    for item in overlays:
        if not isinstance(item, dict) or item.get("hidden"):
            continue
        name = str(item.get("id") or "overlay").strip() or "overlay"
        try:
            width = int(item.get("w") or 0)
            height = int(item.get("h") or 0)
        except (TypeError, ValueError):
            width, height = 0, 0
        if width < 8 or height < 8:
            dead.append(f"overlay {name} is open but unusable ({width}x{height})")
    snapshot = _format_snapshot(text, elements, tabs, overlays)
    failures = [item for item in (network_failures or []) if item]
    notes = [item for item in (network_notes or []) if item]
    evidence = UiEvidence(
        url=str(data.get("url") or url),
        title=str(data.get("title") or ""),
        page_errors=errors,
        console_errors=[item for item in console_errors if item],
        network_failures=failures,
        network_notes=notes,
        smashed=smashed,
        dead_controls=dead,
        clicked=list(clicked or []),
        snapshot=snapshot,
        screenshot=screenshot,
        interactive=interactive,
        ok=True,
    )
    evidence.ok = not evidence.blocking_errors and not smashed and not dead
    return evidence


def _format_snapshot(
    text: str,
    elements: list[Any],
    tabs: list[Any] | None = None,
    overlays: list[Any] | None = None,
) -> str:
    lines: list[str] = []
    for item in tabs or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("pane") or "tab").strip()
        if not name:
            continue
        lines.append(
            f"tab: {name} selected={item.get('selected')} "
            f"panelHidden={item.get('panelHidden')} "
            f"panelHeight={item.get('panelHeight')}"
        )
    for item in overlays or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("id") or "overlay").strip() or "overlay"
        lines.append(f"overlay: {name} hidden={item.get('hidden')} {item.get('w')}x{item.get('h')}")
    for item in elements:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        tag = item.get("tag") or "node"
        loc = ""
        if item.get("x") is not None and item.get("y") is not None:
            loc = f" @ {item.get('x')},{item.get('y')}"
        lines.append(f"{tag}: {name} ({item.get('w')}x{item.get('h')}{loc})")
        if len(lines) >= 40:
            break
    body = text.strip()
    if body:
        lines.extend(["", body[:4000]])
    snapshot = "\n".join(lines)
    if len(snapshot) > MAX_SNAPSHOT_CHARS:
        return snapshot[:MAX_SNAPSHOT_CHARS] + "\n... truncated"
    return snapshot


def unverified_interactive_ui(goal: str, evidence: UiEvidence) -> str | None:
    if evidence.dead_controls:
        return f"Rendered UI control did not work: {evidence.dead_controls[0]}"
    if not evidence.interactive and INTERACTIVE_UI_RE.search(goal or ""):
        return (
            "Rendered UI could not click tabs or buttons (static dump-dom capture); "
            "interactive controls are unverified"
        )
    snapshot = evidence.snapshot or ""
    clicked = " ".join(evidence.clicked).lower()
    if re.search(r"screenshot|screen shot", goal or "", re.I):
        if re.search(r"tab:.*screenshots", snapshot, re.I) and "screenshots" not in clicked:
            return "Rendered UI never exercised the Screenshots tab; the control was not clicked"
    if re.search(
        r"past run|current run|history tab|workspace views|own tabs?",
        goal or "",
        re.I,
    ):
        if "data-main-pane" not in clicked and "past runs" not in clicked:
            return (
                "Rendered UI never clicked the Current/Past runs tabs; "
                "those controls were not verified"
            )
    if re.search(r"\bskills?\b", goal or "", re.I) and re.search(
        r"skills-panel|open-skills", snapshot, re.I
    ):
        if "open-skills" not in clicked and "skills-panel" not in clicked:
            return "Rendered UI never opened Skills; the control was not clicked"
    if re.search(r"\bguidelines?\b|\brules?\b", goal or "", re.I) and re.search(
        r"guidelines-panel|open-guidelines", snapshot, re.I
    ):
        if "open-guidelines" not in clicked and "guidelines-panel" not in clicked:
            return "Rendered UI never opened Rules; the control was not clicked"
    if re.search(r"overlap|overlapping", goal or "", re.I):
        if evidence.smashed:
            return f"Rendered UI controls overlap or are unusable: {evidence.smashed[0]}"
        snapshot = (evidence.snapshot or "").lower()
        if "rerun" in (goal or "").lower() and "rerun" not in snapshot:
            return "Rendered UI never showed the rerun control after capture"
    return None


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._skip = tag in {"script", "style", "noscript"}

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def _visible_text(markup: str) -> str:
    parser = _TextParser()
    parser.feed(markup or "")
    return html.unescape("\n".join(parser.parts))


def _chrome_logged_errors(stderr: str) -> list[str]:
    errors: list[str] = []
    for line in (stderr or "").splitlines():
        if re.search(r"Uncaught|TypeError|ReferenceError|SyntaxError", line):
            errors.append(line.strip())
    return errors[:12]


def _chrome_executable() -> str | None:
    named = os.environ.get("CHROME_PATH")
    candidates = [
        named,
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("microsoft-edge"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


def _has_web_preview(root: Path) -> bool:
    if _is_loco_project(root):
        return True
    config = root / ".loco" / "config.yaml"
    if config.exists() and "preview_command" in config.read_text(encoding="utf-8"):
        return True
    return _first_html_file(root) is not None


def _is_loco_project(root: Path) -> bool:
    return (root / "src" / "agent_loco" / "web_ui.py").is_file() or (
        root / "src" / "agent_loco" / "templates" / "index.html"
    ).is_file()


def _is_loco_template_path(workspace: Workspace, path: str | None) -> bool:
    """True when path is this app's Jinja template, not a standalone HTML file."""
    if not path or not _is_loco_project(workspace.root):
        return False
    try:
        resolved = workspace.resolve(path)
        rel = resolved.resolve().relative_to(workspace.root.resolve()).as_posix()
    except (OSError, ValueError, Exception):
        rel = str(path or "").replace("\\", "/")
    lowered = rel.lower()
    return "/templates/" in f"/{lowered}" and lowered.endswith((".html", ".htm"))


def _control_class(item: dict) -> str:
    return str(item.get("cls") or "").lower()


def _control_rect(item: dict) -> tuple[int, int, int, int] | None:
    try:
        width = int(item.get("w") or 0)
        height = int(item.get("h") or 0)
        x = int(item.get("x") or 0)
        y = int(item.get("y") or 0)
    except (TypeError, ValueError):
        return None
    if width < 4 or height < 4:
        return None
    return (x, y, x + width, y + height)


def _rects_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    *,
    slack: int = 1,
) -> bool:
    return (
        left[0] < right[2] - slack
        and right[0] < left[2] - slack
        and left[1] < right[3] - slack
        and right[1] < left[3] - slack
    )


def _overlapping_controls(elements: list[Any]) -> list[str]:
    """Rerun / PR / +/- stats that share screen space are a layout failure."""
    interesting: list[tuple[str, str, tuple[int, int, int, int]]] = []
    for item in elements:
        if not isinstance(item, dict) or item.get("hidden"):
            continue
        cls = _control_class(item)
        name = str(item.get("name") or item.get("tag") or "control").strip()
        kind = None
        if "rerun-btn" in cls:
            kind = "rerun"
        elif "history-line-stats" in cls:
            kind = "stats"
        elif "pr-link" in cls:
            kind = "pr"
        if not kind:
            continue
        rect = _control_rect(item)
        if rect is None:
            continue
        interesting.append((kind, name or kind, rect))
    hits: list[str] = []
    for index, (left_kind, left_name, left_rect) in enumerate(interesting):
        for right_kind, right_name, right_rect in interesting[index + 1 :]:
            if left_kind == right_kind:
                continue
            if _rects_overlap(left_rect, right_rect):
                hits.append(f"{left_kind} ({left_name}) overlaps {right_kind} ({right_name})")
    return hits


def _first_html_file(root: Path) -> Path | None:
    for candidate in (
        root / "index.html",
        root / "src" / "agent_loco" / "templates" / "index.html",
        *sorted(root.glob("*.html")),
    ):
        if candidate.is_file():
            return candidate
    return None


def _default_clicks(root: Path) -> list[str]:
    if _is_loco_project(root):
        return [
            '[data-main-pane="history"]',
            "#open-history-picker",
            "#history-list button.task",
            '[data-task-pane="changes"]',
            '[data-task-pane="screenshots"]',
            '[data-task-pane="progress"]',
            '[data-main-pane="current"]',
            "#task-list button.task",
        ]
    return []


def _goal_clicks(root: Path, goal: str) -> list[str]:
    if not goal or not _is_loco_project(root):
        return []
    clicks: list[str] = []
    moved_out_of_settings = bool(re.search(r"out of (the )?settings", goal, re.I))
    if re.search(r"\bguidelines?\b|\brules?\b", goal, re.I):
        clicks.append("#open-guidelines")
    if re.search(r"\bskills?\b", goal, re.I):
        clicks.append("#open-skills")
    if re.search(r"\bsettings?\b", goal, re.I) and not moved_out_of_settings:
        clicks.append("#open-settings")
    return clicks


def _review_clicks(root: Path, click: str | None, goal: str = "") -> list[str]:
    clicks: list[str] = []
    for item in [*_default_clicks(root), click or "", *_goal_clicks(root, goal)]:
        text = str(item or "").strip()
        if text and text not in clicks:
            clicks.append(text)
    return clicks


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class _Preview:
    url: str
    close: Callable[[], None]


def _wait_for_port(port: int, timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"preview server did not start on port {port}")


def _start_command_preview(workspace: Workspace, command: str, port: int) -> _Preview:
    rendered = command.replace("{port}", str(port))
    proc = subprocess.Popen(  # noqa: S603 - project-configured preview
        ["/bin/bash", "-lc", rendered],
        cwd=workspace.root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_port(port)
    return _Preview(
        url=f"http://127.0.0.1:{port}/",
        close=lambda: proc.terminate(),
    )


def _start_static_preview(directory: Path, port: int, filename: str) -> _Preview:
    proc = subprocess.Popen(
        ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=directory,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_port(port)
    return _Preview(
        url=urljoin(f"http://127.0.0.1:{port}/", filename),
        close=lambda: proc.terminate(),
    )


def _start_loco_preview(workspace: Workspace, port: int) -> _Preview:
    import uvicorn

    from agent_loco.config import Settings
    from agent_loco.runtime.improve import CycleResult
    from agent_loco.runtime.tasks import TaskManager
    from agent_loco.web_ui import create_app

    def runner(task: Any) -> CycleResult:
        return CycleResult(
            status="skipped",
            goal=task.goal,
            summary="ui review preview",
            tests_passed=True,
            committed=False,
            published=False,
            commit_sha=None,
            reason="preview",
        )

    manager = TaskManager(Settings(), runner=runner, max_concurrent=1)
    app = create_app(manager, default_workspace=workspace.root)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_port(port)
    return _Preview(
        url=f"http://127.0.0.1:{port}/",
        close=lambda: setattr(server, "should_exit", True),
    )

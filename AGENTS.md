# AGENTS.md

Practices for changing this repository. The cycle checklist for working on loco
itself is also in `skills/agent-loco/SKILL.md`.

## Layout

- Package code lives in `src/agent_loco/`.
- Tests live in `tests/test_<area>.py`. Put a test next to the behavior it
  covers. Do not append a new feature's tests onto an unrelated file.
- Web UI is `src/agent_loco/templates/index.html`, `static/`, and `web_ui.py`.
- The cycle is `runtime/improve.py`, `runtime/review.py`, and `agent/loop.py`.
- Bundled skills ship in `src/agent_loco/skills/<name>/SKILL.md`.
- Repo skills live at `skills/<name>/SKILL.md`.
- Prove a new behavior at a public seam: the `loco` CLI, an `/api/...` route,
  or the rendered page. Do not add a helper, route, or control that nothing calls.

## Python

- Python 3.12+. Start modules that use annotations with
  `from __future__ import annotations`.
- Line length is 100. Before finishing, run `uv run ruff format src tests`
  then `uv run ruff check src tests`. Ruff selects `E`, `F`, `I`, `UP`, and `B`.
- Use `pathlib.Path` for filesystem paths.
- Name caught exceptions `exc`. Tool and HTTP handlers keep failures in-band:
  tools return `ToolResult(False, message)`, and HTTP errors are
  `{"error": "..."}` with a 4xx status. Do not let a tool exception escape the
  tool runner.
- Prefer a public function over importing another module's `_private` helper.
- Keep imports at the top of the file, in the order ruff applies. A function
  may import a heavy or circular dependency locally when the top-level import
  would cycle.

## Web UI

- JavaScript in `index.html` uses double quotes and `function` declarations for
  named behavior. Iterate with `for...of`.
- Sidebar panels (`#skills-panel`, `#guidelines-panel`) are overlays on the
  sidebar. The skills list fills the overlay and scrolls. Close the skills
  panel from its Close button or a click outside the panel. Clicks inside the
  panel, including search and clone, leave it open.
- The main task pane stays visible beside an open sidebar overlay.
- UI layout is the bundled `ui` skill. New workspaces enable it with `tdd`.
  After a UI edit, exercise the control in a browser: open it, use it, and
  close it. A template diff is not done. `review_ui` must show the control
  without overlap, unreadable controls, dead buttons, or 404s.
## CSS

- Extract inline styles from templates into `static/layout.css`.
- Do not add inline `style=` attributes to HTML elements in templates.
- Define CSS classes in `layout.css` and apply them to HTML elements.
- After CSS changes, run `review_ui` to verify the page renders correctly.

## Skills and the cycle

- Agent process belongs in a skill, not in `SYSTEM_PROMPT` or the agent loop.
- If you add a bundled skill, keep `package-data` including `skills/*/SKILL.md`.
- A cycle is finished only when the stated goal is wired through. A later cycle
  is not a substitute for missing data or UI.
- After behavior changes, run `uv run pytest -q`. After Python edits, format
  with ruff, then lint.

## Git

- Do not commit, stage, or push `.loco/` or `history.json`. Config, cloned
  skills, run logs, and screenshots stay local.
- Do not force-push, skip hooks, or rewrite published history.
- Do not bind-mount this app's `.loco` over `/workspaces/.loco`.
- `loco init` needs a directory that already exists. Do not document
  `docker clone`.
- Do not ship stubs, TODOs, unused form fields, or "this enables X later".

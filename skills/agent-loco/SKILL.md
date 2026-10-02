---
name: agent-loco
description: >-
  Conventions for changing this agent-loco repository. Use when editing the
  cycle loop, web UI, skills, Docker, git publish path, or tests in this repo.
---

# agent-loco

This workspace is loco itself. Finish the stated goal in this run. A later
cycle is not a substitute for missing wiring, data, or UI. Repo-wide practices
are in `AGENTS.md` at the git root.

## Layout

- Package: `src/agent_loco/`
- Tests: `tests/` — `uv run pytest -q`
- Web UI: `templates/index.html`, `static/`, `web_ui.py`
- Cycle: `runtime/improve.py`, `runtime/review.py`, `agent/loop.py`
- Bundled skills: `src/agent_loco/skills/<name>/SKILL.md` (shipped with the package)
- Repo skills: `skills/<name>/SKILL.md` at the git root (this file)

Prove new behavior at a public seam: CLI (`loco`), HTTP (`/api/...`), or the
rendered page. Do not add a helper, route, or control that nothing calls.

## Process

- Process belongs in a skill, not hardcoded into `SYSTEM_PROMPT` or the agent loop.
- After Python edits, rewrite files with `uv run ruff format src tests`, then
  verify with `uv run ruff check src tests`. Call `run_lint` after that. CI
  runs `ruff format --check`; unformatted Python is a failed cycle.
- After behavior changes, call `run_tests`.
- After UI edits, follow the bundled `ui` skill: call `review_ui`, click the
  new control, and fix overlap, unreadable controls, JS errors, 404s, dead
  buttons, and zero-size panels. A template diff is not done.
- If you add a bundled skill, keep `package-data` `skills/*/SKILL.md`.

## Do not

- Commit, stage, or push `.loco/` or `history.json`. Config, cloned skills,
  run logs, and screenshots stay local.
- Force-push, skip hooks, or rewrite published history.
- Bind-mount this app's `.loco` over `/workspaces/.loco`.
- Document `docker clone`. `loco init` needs a directory that already exists.
- Attach every UI screenshot to a PR.
- Ship stubs, TODOs, unused form fields, or "this enables X later".

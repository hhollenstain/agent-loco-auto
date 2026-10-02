---
name: implement
description: >-
  The process for implementing a task: restate the goal, name files to change,
  use str_replace on existing files, and call run_tests and review_ui as needed.
---

# Implementation Process

This skill guides the agent on how to **start** a task cleanly and execute changes reliably.

## Restate the goal

Before any tool call, restate the goal in one sentence. This ensures alignment and prevents drift. The restate must appear in the text body (not just internal reasoning).

## Name files to change

Identify 1–5 files that will be edited. Prefer `str_replace` on existing files rather than creating new ones. List these files explicitly before any edit.

## Look up unknowns

When the API, architecture, or a best practice is not already in this repo,
call `web_search` and then `fetch_url` on an official docs page before
inventing one. Do not guess library APIs or current recommendations.

Follow the other enabled process skills: **explore** before adding structure,
**debug** when the goal is a failure, **security** on input or secrets,
and **review** before you stop.

## Edit files

- Use `str_replace` for precision. Provide the exact `old_string` from the file.
- Do not repeat the same `str_replace` if it fails; adjust the arguments instead.
- Do not add scaffolding, mocks, TODOs, or unused form fields that don't serve this goal.

## Verify

- After functional edits: call `run_tests` once. If the tree has not changed since the last run, loco reuses that result.
- After tests, call `run_lint` if the workspace has a lint command. That
  command formats with ruff, then checks. Fix F401, E501, import order, and
  any files ruff would reformat before stopping.
- After HTML/CSS/JS/template edits: call `review_ui`, click the new control, fix overlap/404/dead buttons.
- If a tool fails, change arguments; do not repeat the same failed command.

## Stop when done

- If the tree already satisfies the goal, say so and stop. Do not continue editing.
- Do not leave the skill disabled or skip verification steps.
- Do not add process to `SYSTEM_PROMPT`. This skill contains the process.

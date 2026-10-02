---
name: review
description: >-
  Self-review the diff against the stated goal before stopping. Use on
  every change. Catch stubs, unused helpers, and unmet spec in this run.
---

# Review

The cycle has a separate reviewer that will reject an unfinished diff.
Review your own change before you summarize.

## Spec

Walk the goal (issue body and comments count) and the files you changed:

- Every requested behavior is in the tree, not in a comment or plan.
- Nothing the goal said to remove is still present.
- A new control, route, or CLI flag is wired to something that calls it.
- You did not ship a stub, mock, TODO, unused form field, or
  "this enables X later".

## Standards

- The change matches this repo's existing style and test command.
- No unused helper: a new function, class, or route that nothing calls.
- No invented API that you did not see in the repo or in fetched docs.
- After behavior changes: `run_tests` has run since the last edit.
- After UI changes: `review_ui` has been called and the new control works.
- After lint is configured: `run_lint` is clean, including ruff format.

## If anything fails

Fix it in this run. Do not stop to "leave it for review". A later cycle
is not a substitute for a missing wire, test, or file.

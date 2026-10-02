---
name: tdd
description: >-
  Test-driven development. Use when building features or fixing bugs
  test-first, when the user mentions red-green-refactor, or when a change
  must prove behavior at a public interface before it counts as done.
---

# Test-Driven Development

TDD is the red → green loop. Tests verify behavior through public interfaces,
not implementation details. A good test reads like a specification and survives
refactors because it does not care about internal structure.

## Seams

A seam is the public boundary you test at: HTTP API, CLI, or rendered UI.
Do not unit-test an unused helper. Do not add a function, class, or route
that nothing calls.

## Rules of the loop

- **Red before green.** Write one failing test at the public seam first.
  Call `run_tests` and confirm it fails on the current tree.
- **Then implement.** Only enough production code to make that test pass.
  Wire the API, CLI, or UI the test actually invokes.
- **One slice at a time.** One seam, one test, one minimal implementation.
- Do not write all tests first, then all implementation.
- After the implementation, call `run_tests` once. If the tree has not changed
  since the last run, loco reuses that result instead of running the suite again.
  Do not keep calling `run_tests` on an unchanged tree.
- After tests, call `run_lint` when the project has a lint command. That
  formats with ruff, then checks. Fix every finding, including files that
  would be reformatted. The cycle will not commit or open a PR while lint fails.

## Anti-patterns

- **Unused helper:** a new function that nothing in the change calls.
- **Implementation-coupled:** mocking internals or testing private methods.
- **Tautological:** asserting a value the same way the code computes it.
- **Already green:** a test that passed before the behavior existed is not proof.

## In this workspace

- After the failing test, implement, then call `run_tests` once more.
  Loco reuses a previous result when the tree has not changed.
- Then call `run_lint` if a lint command is configured. Unused imports,
  long lines, and unformatted files must be fixed before a PR.
- If the change is UI, `review_ui` is the rendered-page seam: click the new
  control and fix errors, 404s, or dead buttons.
- If existing tests assert old markup or APIs this goal replaces, update those
  tests so they describe the new public behavior.

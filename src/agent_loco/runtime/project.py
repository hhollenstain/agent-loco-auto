from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from agent_loco.agent.prompts import SYSTEM_PROMPT
from agent_loco.runtime.skills import enabled_skill_names
from agent_loco.tools.files import SKIP_DIR_NAMES

GUIDELINES_FILE = "guidelines.md"
DEFAULT_MAX_REPAIR_ATTEMPTS = 4


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    root: Path
    test_command: str | None
    setup_command: str | None
    publish_enabled: bool
    publish_remote: str
    publish_branch: str | None
    create_pr: bool
    goals_file: str
    max_repair_attempts: int
    preview_command: str | None = None
    lint_command: str | None = None
    is_git: bool = False


def load_project(root: Path) -> ProjectConfig:
    root = root.resolve()
    raw: dict = {}
    config_path = root / ".loco" / "config.yaml"
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"invalid project config: {config_path}")
        raw = loaded

    publish = raw.get("publish") or {}
    if not isinstance(publish, dict):
        publish = {}

    configured = raw.get("test_command")
    test_command = str(configured) if configured else infer_test_command(root)
    configured_lint = raw.get("lint_command")
    if configured_lint is None:
        lint_command = infer_lint_command(root)
    elif configured_lint == "" or configured_lint is False:
        lint_command = None
    else:
        lint_command = str(configured_lint)
    is_git = _is_git_repository(root)
    publish_enabled = bool(publish["enabled"]) if "enabled" in publish else is_git
    create_pr = bool(publish["create_pr"]) if "create_pr" in publish else is_git
    return ProjectConfig(
        name=str(raw.get("name") or root.name),
        root=root,
        test_command=test_command,
        setup_command=raw.get("setup_command"),
        publish_enabled=publish_enabled,
        publish_remote=str(publish.get("remote") or "origin"),
        publish_branch=publish.get("branch"),
        create_pr=create_pr,
        goals_file=str(raw.get("goals_file") or "goals.md"),
        max_repair_attempts=_max_repair_attempts(raw.get("max_repair_attempts")),
        preview_command=str(raw["preview_command"]) if raw.get("preview_command") else None,
        lint_command=lint_command,
        is_git=is_git,
    )


def _max_repair_attempts(value: object) -> int:
    if value is None or value == "":
        return DEFAULT_MAX_REPAIR_ATTEMPTS
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return DEFAULT_MAX_REPAIR_ATTEMPTS


def infer_test_command(root: Path) -> str | None:
    if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists():
        return "pytest -q"
    package_json = root / "package.json"
    if package_json.exists() and '"test"' in package_json.read_text(encoding="utf-8"):
        return "npm test"
    makefile = root / "Makefile"
    if makefile.exists() and _makefile_has_target(makefile, "test"):
        return "make test"
    if (root / "cargo.toml").exists() or (root / "Cargo.toml").exists():
        return "cargo test"
    if (root / "go.mod").exists():
        return "go test ./..."
    return None


def infer_lint_command(root: Path) -> str | None:
    """Infer a linter that matches CI for this workspace."""
    has_ruff = (root / "ruff.toml").exists() or (root / "pyproject.toml").exists()
    if has_ruff:
        src = root / "src"
        tests_dir = root / "tests"
        if src.is_dir() and tests_dir.is_dir():
            return "ruff check src tests"
        return "ruff check ."
    package_json = root / "package.json"
    if package_json.exists() and '"lint"' in package_json.read_text(encoding="utf-8"):
        return "npm run lint"
    return None


def load_goals(root: Path, goals_file: str) -> list[str]:
    path = root / ".loco" / goals_file
    if not path.exists():
        path = root / goals_file
    if not path.exists():
        return []

    goals: list[str] = []
    current: list[str] | None = None

    def flush() -> None:
        nonlocal current
        if current:
            text = "\n".join(current).strip()
            if text:
                goals.append(text)
        current = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- [x]"):
            flush()
            continue
        if stripped.startswith("- [ ]"):
            flush()
            goal = stripped[5:].strip()
            current = [goal] if goal else []
            continue
        if current is None:
            continue
        if line.startswith(" ") or line.startswith("\t"):
            current.append(line.strip())
        elif not stripped:
            current.append("")
        else:
            flush()
    flush()
    return goals


def mark_goal_done(root: Path, goals_file: str, goal: str) -> bool:
    first = goal.strip().splitlines()[0].strip() if goal.strip() else ""
    for candidate in (root / ".loco" / goals_file, root / goals_file):
        if not candidate.exists():
            continue
        lines = candidate.read_text(encoding="utf-8").splitlines()
        changed = False
        updated: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped == f"- [ ] {goal}" or (first and stripped == f"- [ ] {first}"):
                updated.append(line.replace("- [ ]", "- [x]", 1))
                changed = True
            else:
                updated.append(line)
        if changed:
            candidate.write_text("\n".join(updated) + "\n", encoding="utf-8")
            return True
    return False


def write_default_project_files(root: Path) -> list[Path]:
    loco = root / ".loco"
    loco.mkdir(exist_ok=True)
    created: list[Path] = []

    config_path = loco / "config.yaml"
    if not config_path.exists():
        is_git = _is_git_repository(root)
        test_command = infer_test_command(root) or "pytest -q"
        lint_command = infer_lint_command(root)
        publish_on = "true" if is_git else "false"
        lint_line = f"lint_command: {lint_command}\n" if lint_command else ""
        config_path.write_text(
            (
                f"name: {root.name}\n"
                f"test_command: {test_command}\n"
                f"{lint_line}"
                f"max_repair_attempts: {DEFAULT_MAX_REPAIR_ATTEMPTS}\n"
                "publish:\n"
                f"  enabled: {publish_on}\n"
                "  remote: origin\n"
                f"  create_pr: {publish_on}\n"
                "goals_file: goals.md\n"
                "skills:\n"
                "  enabled:\n"
                "    - debug\n"
                "    - explore\n"
                "    - implement\n"
                "    - research\n"
                "    - review\n"
                "    - security\n"
                "    - tdd\n"
                "    - ui\n"
            ),
            encoding="utf-8",
        )
        created.append(config_path)

    goals_path = loco / "goals.md"
    if not goals_path.exists():
        goals_path.write_text(
            "# Goals\n\n- [ ] Add a regression test for the next change\n",
            encoding="utf-8",
        )
        created.append(goals_path)
    guidelines = guidelines_path(root)
    if not guidelines.exists():
        guidelines.write_text(_default_guidelines_text(), encoding="utf-8")
        created.append(guidelines)
    gitignore = ensure_run_gitignore(root)
    if gitignore is not None:
        created.append(gitignore)
    return created


def ensure_run_gitignore(root: Path) -> Path | None:
    """Keep the whole `.loco/` directory out of git. Returns a path only when created."""
    created = _ensure_root_loco_ignore(root)
    loco = root / ".loco"
    loco.mkdir(parents=True, exist_ok=True)
    path = loco / ".gitignore"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        lines = {line.strip() for line in existing.splitlines()}
        if "*" not in lines:
            suffix = "" if existing.endswith("\n") or not existing else "\n"
            path.write_text(f"{existing}{suffix}*\n", encoding="utf-8")
        return created
    path.write_text("*\n", encoding="utf-8")
    return created or path


def _ensure_root_loco_ignore(root: Path) -> Path | None:
    path = Path(root) / ".gitignore"
    marker = ".loco/"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if _ignores_loco_dir(existing):
            return None
        suffix = "" if existing.endswith("\n") or not existing else "\n"
        path.write_text(f"{existing}{suffix}{marker}\n", encoding="utf-8")
        return None
    path.write_text(f"{marker}\n", encoding="utf-8")
    return path


def _ignores_loco_dir(text: str) -> bool:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in {".loco", ".loco/", "/.loco", "/.loco/", ".loco/**"}:
            return True
    return False


def guidelines_path(root: Path) -> Path:
    return Path(root) / ".loco" / GUIDELINES_FILE


def _default_guidelines_text() -> str:
    text = SYSTEM_PROMPT.strip()
    return f"{text}\n"


def default_guidelines() -> str:
    return _default_guidelines_text()


def load_guidelines(root: Path) -> str:
    """Return workspace rules, or the built-in default when none are set."""
    path = guidelines_path(root)
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return text
    except OSError:
        pass
    return _default_guidelines_text()


def guidelines_are_custom(root: Path) -> bool:
    path = guidelines_path(root)
    try:
        if not path.exists():
            return False
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return bool(text) and text != SYSTEM_PROMPT.strip()


def save_guidelines(root: Path, text: str | None) -> str:
    """Persist custom rules, or restore the default when the text is empty."""
    path = guidelines_path(root)
    content = (text or "").strip()
    if not content or content == SYSTEM_PROMPT.strip():
        if path.exists():
            path.unlink()
        return _default_guidelines_text()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{content}\n", encoding="utf-8")
    return path.read_text(encoding="utf-8")


def collect_context(
    root: Path,
    project: ProjectConfig,
    *,
    allow_publish: bool | None = None,
    brief_context: bool = False,
) -> str:
    """Collect workspace context or task brief.

    Args:
        root: Workspace root path
        project: ProjectConfig
        allow_publish: Override publish_enabled flag
        brief_context: If True, use build_task_brief; otherwise use legacy collect_context
    """
    if brief_context:
        from agent_loco.runtime.brief import build_task_brief

        return build_task_brief(root, project.goals_file, project)
    publish_on = project.publish_enabled if allow_publish is None else allow_publish
    parts = [
        f"Project: {project.name}",
        f"Root: {root}",
        f"Test command: {project.test_command or '(none)'}",
        f"Lint command: {project.lint_command or '(none)'}",
        f"Create PR: {'on' if publish_on else 'off'} via {project.publish_remote}"
        + (" (git repo)" if project.is_git else ""),
        " (never pushes to main)",
    ]
    goals = load_goals(root, project.goals_file)
    if goals:
        parts.append("Open goals:")
        parts.extend(f"- {goal}" for goal in goals[:8])
    enabled = enabled_skill_names(root)
    if enabled:
        parts.append("Enabled skills: " + ", ".join(enabled))
    tree = render_tree(root)
    if tree:
        parts.append("Workspace files:")
        parts.append(tree)
    return "\n".join(parts)


def render_tree(root: Path, *, max_entries: int = 80, max_depth: int = 3) -> str:
    lines: list[str] = []

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if len(lines) >= max_entries or depth > max_depth:
            return
        children = sorted(
            (child for child in directory.iterdir() if child.name not in SKIP_DIR_NAMES),
            key=lambda path: (not path.is_dir(), path.name.lower()),
        )
        for child in children:
            if len(lines) >= max_entries:
                lines.append(f"{prefix}...")
                return
            lines.append(f"{prefix}{child.name}{'/' if child.is_dir() else ''}")
            if child.is_dir():
                walk(child, prefix + "  ", depth + 1)

    walk(root, "", 1)
    return "\n".join(lines)


def _is_git_repository(root: Path) -> bool:
    return (root / ".git").exists()


def _makefile_has_target(path: Path, target: str) -> bool:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{target}:") or line.startswith(f"{target} :"):
            return True
    return False

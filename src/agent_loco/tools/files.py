from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agent_loco.progress import record_file_change
from agent_loco.sandbox import Workspace
from agent_loco.tools.base import ToolResult, ToolSpec, object_schema

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".loco",
}

MAX_READ_CHARS = 200_000
MAX_WRITE_BYTES = 1_000_000
MAX_LIST_ENTRIES = 200
MAX_SEARCH_HITS = 50
MAX_HITS_PER_FILE = 5
_NUMBERED_READ_RE = re.compile(r"(?m)^[ \t]*\d+\|")


def file_tools(workspace: Workspace) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="list_dir",
            description="List files and directories under a workspace-relative path.",
            parameters=object_schema(
                {
                    "path": {
                        "type": "string",
                        "description": "Directory to list. Use '.' for the workspace root.",
                    }
                },
                ["path"],
            ),
            handler=lambda path: _list_dir(workspace, path),
        ),
        ToolSpec(
            name="read_file",
            description="Read a text file. Optionally slice by 1-based start_line and end_line. "
            "The `N|` prefix is a line number; do not copy it into str_replace.",
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "File path inside the workspace."},
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-based first line to include.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-based last line to include.",
                    },
                },
                ["path"],
            ),
            handler=lambda path, start_line=None, end_line=None: _read_file(
                workspace, path, start_line, end_line
            ),
        ),
        ToolSpec(
            name="write_file",
            description=(
                "Create or overwrite a text file. Prefer str_replace for edits to "
                "existing files; write_file must include the full file contents and "
                "is a poor fit for large files. Creates parent directories."
            ),
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "File path inside the workspace."},
                    "content": {"type": "string", "description": "Full file contents to write."},
                },
                ["path", "content"],
            ),
            handler=lambda path, content: _write_file(workspace, path, content),
        ),
        ToolSpec(
            name="str_replace",
            description=(
                "Replace exact text in an existing file. Prefer this over write_file "
                "for edits. old_string must match exactly once unless replace_all is true. "
                "Copy the file text without the numbered `N|` prefix from read_file."
            ),
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "File path inside the workspace."},
                    "old_string": {
                        "type": "string",
                        "description": (
                            "Exact text to find. Copy from the file, "
                            "not from numbered read_file output."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "Replace every match. Default false requires a unique match."
                        ),
                    },
                },
                ["path", "old_string", "new_string"],
            ),
            handler=lambda path, old_string, new_string, replace_all=False: _str_replace(
                workspace, path, old_string, new_string, replace_all
            ),
        ),
        ToolSpec(
            name="search_text",
            description=(
                "Search workspace files for a regex or literal string. "
                "Skips VCS and dependency dirs."
            ),
            parameters=object_schema(
                {
                    "query": {"type": "string", "description": "Text or regex to search for."},
                    "glob": {
                        "type": "string",
                        "description": "Optional glob such as '*.py' or 'src/**/*.ts'.",
                    },
                },
                ["query"],
            ),
            handler=lambda query, glob=None: _search_text(workspace, query, glob),
        ),
    ]


def _list_dir(workspace: Workspace, path: str) -> ToolResult:
    directory = workspace.resolve(path)
    if not directory.exists():
        return ToolResult(False, f"not found: {path}")
    if not directory.is_dir():
        return ToolResult(False, f"not a directory: {path}")

    entries: list[str] = []
    for child in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name in SKIP_DIR_NAMES:
            continue
        suffix = "/" if child.is_dir() else ""
        entries.append(f"{child.name}{suffix}")
        if len(entries) >= MAX_LIST_ENTRIES:
            entries.append(f"... truncated after {MAX_LIST_ENTRIES} entries")
            break
    return ToolResult(True, "\n".join(entries) if entries else "(empty)")


def _read_file(
    workspace: Workspace,
    path: str,
    start_line: int | None,
    end_line: int | None,
) -> ToolResult:
    file_path = workspace.resolve(path)
    if not file_path.exists():
        return ToolResult(False, f"not found: {path}")
    if not file_path.is_file():
        return ToolResult(False, f"not a file: {path}")
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult(False, f"not a utf-8 text file: {path}")

    lines = text.splitlines()
    start = 1 if start_line is None else max(1, start_line)
    end = len(lines) if end_line is None else min(len(lines), end_line)
    if start > end:
        return ToolResult(False, "start_line must be <= end_line")
    sliced = lines[start - 1 : end]
    numbered = [f"{idx + start:>4}|{line}" for idx, line in enumerate(sliced)]
    header = f"{path}  lines {start}-{end} of {len(lines)}"
    body = header + ("\n" + "\n".join(numbered) if numbered else "\n(empty file)")
    if len(body) > MAX_READ_CHARS:
        body = body[:MAX_READ_CHARS] + "\n... truncated"
    return ToolResult(True, body)


def _write_file(workspace: Workspace, path: str, content: str) -> ToolResult:
    file_path = workspace.resolve(path)
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        return ToolResult(False, f"refusing to write more than {MAX_WRITE_BYTES} bytes")
    rel = workspace.relative(file_path)
    created = not file_path.exists()
    before: str | None = ""
    if not created:
        try:
            before = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            before = None
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return ToolResult(False, f"write failed: {exc}")
    record_file_change(rel, before=before, after=content, created=created)
    return ToolResult(True, f"wrote {rel} ({len(encoded)} bytes)")


def _str_replace(
    workspace: Workspace,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: Any = False,
) -> ToolResult:
    if not old_string:
        return ToolResult(False, "old_string is required")
    if old_string == new_string:
        return ToolResult(False, "old_string and new_string are identical")
    file_path = workspace.resolve(path)
    if not file_path.exists():
        return ToolResult(False, f"not found: {path}")
    if not file_path.is_file():
        return ToolResult(False, f"not a file: {path}")
    try:
        before = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult(False, f"not a utf-8 text file: {path}")
    except OSError as exc:
        return ToolResult(False, f"read failed: {exc}")
    needle = old_string
    stripped_numbers = False
    matches = before.count(needle)
    if matches == 0:
        stripped = _without_read_prefixes(needle)
        if stripped != needle:
            needle = stripped
            stripped_numbers = True
            matches = before.count(needle)
    if matches == 0:
        hint = _nearby_hint(before, old_string)
        return ToolResult(False, f"old_string not found in {path}.{hint}")
    replace_every = _as_bool(replace_all)
    if matches > 1 and not replace_every:
        return ToolResult(
            False,
            f"old_string matched {matches} times in {path}; "
            "provide more context or set replace_all=true",
        )
    if replace_every:
        after = before.replace(needle, new_string)
    else:
        after = before.replace(needle, new_string, 1)
    encoded = after.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        return ToolResult(False, f"refusing to write more than {MAX_WRITE_BYTES} bytes")
    try:
        file_path.write_text(after, encoding="utf-8")
    except OSError as exc:
        return ToolResult(False, f"write failed: {exc}")
    rel = workspace.relative(file_path)
    record_file_change(rel, before=before, after=after, created=False)
    replaced = matches if replace_every else 1
    note = " (ignored numbered read_file prefixes)" if stripped_numbers else ""
    return ToolResult(
        True,
        f"updated {rel} ({replaced} replacement{'s' if replaced != 1 else ''}){note}",
    )


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _search_text(workspace: Workspace, query: str, glob: str | None) -> ToolResult:
    if not query:
        return ToolResult(False, "query is required")

    rg = shutil.which("rg")
    if rg:
        return _search_with_rg(workspace, rg, query, glob)
    return _search_walk(workspace, query, glob)


def _search_with_rg(workspace: Workspace, rg: str, query: str, glob: str | None) -> ToolResult:
    cmd = [rg, "--line-number", "--no-heading", "--color", "never", "--max-count", "5", query]
    if glob:
        cmd.extend(["--glob", glob])
    for name in SKIP_DIR_NAMES:
        cmd.extend(["--glob", f"!{name}"])
    result = subprocess.run(
        cmd,
        cwd=workspace.root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode not in {0, 1}:
        return ToolResult(False, result.stderr.strip() or "search failed")
    hits = result.stdout.splitlines()[:MAX_SEARCH_HITS]
    return ToolResult(True, "\n".join(hits) if hits else "no matches")


def _search_walk(workspace: Workspace, query: str, glob: str | None) -> ToolResult:
    hits: list[str] = []
    for path in workspace.root.rglob(glob or "*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        try:
            file_hits = 0
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if query in line:
                    rel = workspace.relative(path)
                    hits.append(f"{rel}:{lineno}:{line.strip()}")
                    if len(hits) >= MAX_SEARCH_HITS:
                        return ToolResult(True, "\n".join(hits))
                    file_hits += 1
                    if file_hits >= MAX_HITS_PER_FILE:
                        break
        except (OSError, UnicodeDecodeError):
            continue
    return ToolResult(True, "\n".join(hits) if hits else "no matches")


def _without_read_prefixes(text: str) -> str:
    return _NUMBERED_READ_RE.sub("", text)


def _nearby_hint(before: str, needle: str) -> str:
    token = " ".join(needle.split())
    token = _without_read_prefixes(token)[:48]
    lines = before.splitlines()
    if token:
        hits = [
            f"{index:>4}|{line}" for index, line in enumerate(lines, start=1) if token[:24] in line
        ]
        if hits:
            return " Nearby lines:\n" + "\n".join(hits[:8])
    preview = "\n".join(f"{index:>4}|{line}" for index, line in enumerate(lines[:12], start=1))
    return f" File starts with:\n{preview}" if preview else ""


def is_probably_secret_path(path: Path) -> bool:
    name = path.name.lower()
    if name in {".env", "credentials.json", "secrets.yaml", "secrets.yml", "id_rsa", "id_ed25519"}:
        return True
    if name.startswith(".env."):
        return True
    return path.suffix.lower() in {".pem", ".p12", ".key"}

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

SKILLS_DIRNAME = "skills"
SOURCES_DIRNAME = "sources"
LOCAL_DIRNAME = "local"
SKILL_FILENAMES = ("SKILL.md", "skill.md")
SKIP_WALK = {".git", "node_modules", "__pycache__", ".venv"}
MAX_SKILL_CHARS = 8_000
SUMMARY_LIMIT = 72
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n(.*))?\Z", re.DOTALL)
_MARKDOWN = None


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    origin: str
    path: str
    source_url: str = ""
    enabled: bool = False

    def public_dict(self, *, include_body: bool = False) -> dict[str, str | bool]:
        payload = asdict(self)
        payload["summary"] = skill_summary(self.description or self.body)
        if include_body:
            payload["html"] = render_skill_html(self.body)
        else:
            payload.pop("body", None)
        return payload


@dataclass(frozen=True)
class SkillSource:
    url: str
    slug: str
    ref: str = ""
    path: str = ""

    def public_dict(self) -> dict[str, str]:
        return asdict(self)


def bundled_skills_root() -> Path:
    return Path(__file__).resolve().parent.parent / "skills"


def skills_root(workspace: Path) -> Path:
    return Path(workspace).expanduser().resolve() / ".loco" / SKILLS_DIRNAME


def sources_root(workspace: Path) -> Path:
    return skills_root(workspace) / SOURCES_DIRNAME


def local_skills_root(workspace: Path) -> Path:
    return skills_root(workspace) / LOCAL_DIRNAME


def source_slug(url: str) -> str:
    from agent_loco.runtime.workspaces import repo_name_from_url

    return repo_name_from_url(url) or "skills"


def skill_summary(text: str, *, limit: int = SUMMARY_LIMIT) -> str:
    """First sentence of a skill description, clipped for list rows."""
    compact = " ".join((text or "").split())
    if not compact:
        return "No description."
    for sep in (". ", "? ", "! "):
        index = compact.find(sep)
        if index != -1:
            compact = compact[: index + 1]
            break
    if len(compact) <= limit:
        return compact
    clipped = compact[: limit - 1].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{clipped or compact[: limit - 1]}…"


def get_skill(root: Path, name: str) -> Skill | None:
    wanted = _clean_name(name)
    if not wanted:
        return None
    return next((skill for skill in list_skills(root) if skill.name == wanted), None)


def render_skill_html(text: str) -> str:
    """Turn a SKILL.md body into HTML for the settings popover."""
    body = (text or "").strip()
    if not body:
        return ""
    return _markdown_renderer().render(body).strip()


def _markdown_renderer():
    from markdown_it import MarkdownIt

    global _MARKDOWN
    if _MARKDOWN is None:
        renderer = MarkdownIt("commonmark", {"html": False})
        renderer.enable(["table", "strikethrough"])
        _MARKDOWN = renderer
    return _MARKDOWN


def parse_skill_markdown(text: str, *, fallback_name: str) -> tuple[str, str, str]:
    raw = (text or "").replace("\r\n", "\n")
    match = _FRONTMATTER_RE.match(raw.strip())
    meta: dict = {}
    body = raw.strip()
    if match:
        loaded = yaml.safe_load(match.group(1) or "") or {}
        if isinstance(loaded, dict):
            meta = loaded
        body = (match.group(2) or "").strip()
    name = _clean_name(str(meta.get("name") or fallback_name))
    description = str(meta.get("description") or "").strip()
    return name, description, body


def load_skill_settings(root: Path) -> dict:
    raw = _load_raw_config(root)
    block = raw.get("skills")
    if not isinstance(block, dict):
        block = {}
    enabled = _string_list(block.get("enabled"))
    sources: list[SkillSource] = []
    for item in block.get("sources") or []:
        parsed = _parse_source(item)
        if parsed is not None:
            sources.append(parsed)
    return {"enabled": enabled, "sources": sources}


def enabled_skill_names(root: Path) -> list[str]:
    return list(load_skill_settings(root)["enabled"])


def list_skill_sources(root: Path) -> list[SkillSource]:
    found: list[SkillSource] = []
    for source in load_skill_settings(root)["sources"]:
        dest = sources_root(root) / source.slug
        if dest.is_dir():
            source = replace(source, path=str(dest))
        found.append(source)
    return found


def list_skills(root: Path) -> list[Skill]:
    enabled = set(enabled_skill_names(root))
    by_name: dict[str, Skill] = {}
    for skill in _discover(bundled_skills_root(), origin="bundled"):
        by_name[skill.name] = skill
    for source in list_skill_sources(root):
        dest = Path(source.path) if source.path else sources_root(root) / source.slug
        if not dest.is_dir():
            continue
        for skill in _discover(dest, origin="source", source_url=source.url):
            by_name[skill.name] = skill
    local = local_skills_root(root)
    if local.is_dir():
        for skill in _discover(local, origin="local", skip_paths={}):
            by_name[skill.name] = skill
    repo_skills = _discover_repo_skills(root)
    for skill in repo_skills:
        by_name[skill.name] = skill
    return [
        replace(skill, enabled=skill.name in enabled)
        for skill in sorted(by_name.values(), key=lambda item: item.name)
    ]


def save_enabled_skills(root: Path, names: list[str] | None) -> list[Skill]:
    known = {skill.name for skill in list_skills(root)}
    enabled = [name for name in _string_list(names) if name in known]
    _update_skills_config(root, enabled=enabled)
    return list_skills(root)


def add_skill_source(
    root: Path,
    url: str,
    *,
    ref: str | None = None,
) -> SkillSource:
    repo = (url or "").strip()
    if not repo:
        raise ValueError("repository URL is required")
    source = SkillSource(url=repo, slug=source_slug(repo), ref=(ref or "").strip())
    dest = _clone_or_pull(root, source)
    sources = [item for item in list_skill_sources(root) if item.slug != source.slug]
    sources.append(replace(source, path=str(dest)))
    _update_skills_config(root, sources=sources)
    return replace(source, path=str(dest))


def sync_skill_source(root: Path, slug: str, *, ref: str | None = None) -> SkillSource:
    wanted = (slug or "").strip()
    if not wanted:
        raise ValueError("skill source is required")
    match = next((item for item in list_skill_sources(root) if item.slug == wanted), None)
    if match is None:
        raise ValueError(f"unknown skill source: {wanted}")
    source = replace(match, ref=(ref or match.ref or "").strip())
    dest = _clone_or_pull(root, source)
    sources = [
        replace(source, path=str(dest)) if item.slug == source.slug else item
        for item in list_skill_sources(root)
    ]
    _update_skills_config(root, sources=sources)
    return replace(source, path=str(dest))


def format_enabled_skills(root: Path) -> str:
    enabled = [skill for skill in list_skills(root) if skill.enabled]
    if not enabled:
        return ""
    parts = [
        "## Enabled skills",
        "",
        "Follow these skills for this workspace. They are the process for this run.",
    ]
    for skill in enabled:
        parts.extend(["", f"### {skill.name}"])
        if skill.description:
            parts.append(skill.description)
        body = skill.body.strip()
        if len(body) > MAX_SKILL_CHARS:
            body = body[:MAX_SKILL_CHARS].rstrip() + "\n... truncated"
        if body:
            parts.extend(["", body])
    return "\n".join(parts).strip()


def compose_system_prompt(root: Path) -> str:
    from agent_loco.runtime.project import load_guidelines

    base = load_guidelines(root).rstrip()
    skills = format_enabled_skills(root)
    if not skills:
        return f"{base}\n"
    return f"{base}\n\n{skills}\n"


def skills_catalog(root: Path) -> dict[str, object]:
    return {
        "skills": [skill.public_dict() for skill in list_skills(root)],
        "sources": [source.public_dict() for source in list_skill_sources(root)],
        "enabled": enabled_skill_names(root),
    }


def _discover(
    root: Path,
    *,
    origin: str,
    source_url: str = "",
    skip_paths: set[str] | None = None,
) -> list[Skill]:
    if not root.is_dir():
        return []
    found: list[Skill] = []
    seen: set[Path] = set()
    skip = SKIP_WALK | (skip_paths or set())
    for filename in SKILL_FILENAMES:
        for path in root.rglob(filename):
            resolved = path.resolve()
            if resolved in seen:
                continue
            if any(part in skip for part in path.parts):
                continue
            seen.add(resolved)
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            name, description, body = parse_skill_markdown(text, fallback_name=path.parent.name)
            if not name:
                continue
            found.append(
                Skill(
                    name=name,
                    description=description,
                    body=body,
                    origin=origin,
                    path=str(path),
                    source_url=source_url,
                )
            )
    return found


def _discover_repo_skills(root: Path) -> list[Skill]:
    """Discover skills that exist in the git repository root (outside .loco)."""
    workspace = Path(root).expanduser().resolve()
    repo_root = _find_git_repo_root(workspace)
    if repo_root is None or not repo_root.is_dir():
        return []
    bundled = bundled_skills_root().resolve()
    found: list[Skill] = []
    for skill in _discover(repo_root, origin="repo", skip_paths={".loco"}):
        path = Path(skill.path).resolve()
        if path == bundled or bundled in path.parents:
            continue
        found.append(skill)
    return found


def _find_git_repo_root(start: Path) -> Path | None:
    """Find the root of the git repository containing start."""
    current = start.resolve()
    while current != current.parent:
        if (current / ".git").is_dir():
            return current
        current = current.parent
    return None


def _clean_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", (value or "").strip().lower()).strip("-")
    slug = slug[:64]
    if _NAME_RE.match(slug):
        return slug
    return ""


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = _clean_name(str(item))
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def _parse_source(item: object) -> SkillSource | None:
    if isinstance(item, str):
        url = item.strip()
        if not url:
            return None
        return SkillSource(url=url, slug=source_slug(url))
    if not isinstance(item, dict):
        return None
    url = str(item.get("url") or "").strip()
    if not url:
        return None
    ref = str(item.get("ref") or item.get("branch") or "").strip()
    slug = _clean_name(str(item.get("slug") or "")) or source_slug(url)
    path = str(item.get("path") or "").strip()
    return SkillSource(url=url, slug=slug, ref=ref, path=path)


def _load_raw_config(root: Path) -> dict:
    path = Path(root).expanduser().resolve() / ".loco" / "config.yaml"
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_raw_config(root: Path, raw: dict) -> None:
    path = Path(root).expanduser().resolve() / ".loco" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    dumped = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    path.write_text(dumped, encoding="utf-8")


def _update_skills_config(
    root: Path,
    *,
    enabled: list[str] | None = None,
    sources: list[SkillSource] | None = None,
) -> None:
    raw = _load_raw_config(root)
    block = raw.get("skills")
    if not isinstance(block, dict):
        block = {}
    if enabled is None:
        enabled = _string_list(block.get("enabled"))
    if sources is None:
        sources = [
            item
            for item in (_parse_source(raw_item) for raw_item in (block.get("sources") or []))
            if item is not None
        ]
    raw["skills"] = {
        "enabled": enabled,
        "sources": [
            {key: value for key, value in asdict(source).items() if key != "path" and value}
            for source in sources
        ],
    }
    if not raw["skills"]["sources"]:
        raw["skills"].pop("sources", None)
    _write_raw_config(root, raw)


def _clone_or_pull(root: Path, source: SkillSource) -> Path:
    dest = sources_root(root) / source.slug
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if not (dest / ".git").exists():
            raise ValueError(f"already exists: {dest}")
        _run_git(["git", "pull", "--ff-only"], cwd=dest)
        return dest
    command = ["git", "clone", "--depth", "1"]
    if source.ref:
        command.extend(["--branch", source.ref])
    command.extend([source.url, str(dest)])
    _run_git(command, cwd=dest.parent)
    return dest


def _run_git(command: list[str], *, cwd: Path) -> None:
    from agent_loco.runtime.workspaces import CLONE_TIMEOUT_SECONDS

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("git timed out") from exc
    except OSError as exc:
        raise ValueError(f"git failed: {exc}") from exc
    if result.returncode != 0:
        raise ValueError((result.stderr or result.stdout or "git failed").strip())

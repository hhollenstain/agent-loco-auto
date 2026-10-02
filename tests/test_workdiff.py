from __future__ import annotations

from pathlib import Path

from tests.support import init_git_repo

from agent_loco.runtime.workdiff import (
    collect_current_evidence,
    collect_work_diff,
    extract_version_changes,
    format_work_diff,
    is_lockfile,
    summarize_lockfile_diff,
    summarize_lockfile_pair,
)
from agent_loco.sandbox import Workspace
from agent_loco.tools.git import current_sha


def test_is_lockfile_detects_common_names() -> None:
    assert is_lockfile("Pipfile.lock")
    assert is_lockfile("poetry.lock")
    assert is_lockfile("package-lock.json")
    assert is_lockfile("pnpm-lock.yaml")
    assert is_lockfile("Cargo.lock")
    assert not is_lockfile("setup.py")
    assert not is_lockfile("src/app.py")


def test_extract_version_changes_from_pipfile_lock_hunk() -> None:
    chunk = """diff --git a/Pipfile.lock b/Pipfile.lock
--- a/Pipfile.lock
+++ b/Pipfile.lock
@@ -1,12 +1,12 @@
         "aiohttp": {
             "hashes": [
-                "sha256:aaaaaaaa",
+                "sha256:bbbbbbbb",
             ],
             "version": "==3.8.1"
         },
         "discord.py": {
             "hashes": [
-                "sha256:oldhasholdhash",
+                "sha256:newhashnewhash",
             ],
-            "version": "==2.3.2"
+            "version": "==2.7.1"
         },
"""
    rows = extract_version_changes(chunk, goal="update discord.py")
    assert rows == ["- discord.py: 2.3.2 -> 2.7.1"]


def test_lockfile_summary_prioritizes_goal_package() -> None:
    lines = ["diff --git a/Pipfile.lock b/Pipfile.lock"]
    for name in ("alpha", "beta", "zzz-target"):
        lines.append(f'         "{name}": {{')
        lines.append('-            "version": "==1.0.0"')
        lines.append('+            "version": "==2.0.0"')
        lines.append("         },")
    rows = extract_version_changes("\n".join(lines), goal="bump zzz-target")
    assert rows[0] == "- zzz-target: 1.0.0 -> 2.0.0"
    assert "- alpha: 1.0.0 -> 2.0.0" in rows


def test_format_work_diff_keeps_source_and_lockfile_versions() -> None:
    hashes = ",\n".join(f'                "sha256:{index:064x}"' for index in range(80))
    unified = f"""diff --git a/Pipfile.lock b/Pipfile.lock
--- a/Pipfile.lock
+++ b/Pipfile.lock
@@ -1,8 +1,8 @@
         "discord.py": {{
             "hashes": [
{hashes}
             ],
-            "version": "==2.3.2"
+            "version": "==2.7.1"
         }},
diff --git a/setup.py b/setup.py
--- a/setup.py
+++ b/setup.py
@@ -1,3 +1,3 @@
 INSTALL = [
-    'discord.py==2.3.2',
+    'discord.py==2.7.1',
 ]
"""
    text = format_work_diff(
        status="## master",
        name_status="M\tPipfile.lock\nM\tsetup.py",
        unified=unified,
        goal="This repo is using a really outdated version of discord.py",
        limit=16_000,
    )
    assert "Changed files:" in text
    assert "setup.py" in text
    assert "discord.py==2.7.1" in text
    assert "discord.py: 2.3.2 -> 2.7.1" in text
    assert text.count("sha256") < 3
    assert "2.7.1" in text


def test_hash_only_lockfile_diff_does_not_claim_version_update() -> None:
    chunk = """diff --git a/Pipfile.lock b/Pipfile.lock
         "aiohttp": {
             "hashes": [
-                "sha256:old",
+                "sha256:new",
             ],
             "version": "==3.8.1"
         },
"""
    summary = summarize_lockfile_diff("Pipfile.lock", chunk)
    assert "no package version changes" in summary
    assert "->" not in summary


def test_lockfile_pair_finds_version_when_hunk_omits_package_name() -> None:
    old = _pipfile_lock("2.3.2", hashes=80)
    new = _pipfile_lock("2.7.1", hashes=80)
    summary = summarize_lockfile_pair("Pipfile.lock", old, new, goal="update discord.py")
    assert "discord.py: 2.3.2 -> 2.7.1" in summary
    assert "sha256" not in summary


def test_collect_work_diff_survives_huge_lockfile(tmp_path: Path) -> None:
    (tmp_path / "setup.py").write_text("INSTALL = ['discord.py==2.3.2']\n", encoding="utf-8")
    (tmp_path / "Pipfile.lock").write_text(_pipfile_lock("2.3.2", hashes=120), encoding="utf-8")
    init_git_repo(tmp_path)
    workspace = Workspace(tmp_path)
    sha = current_sha(workspace)
    (tmp_path / "setup.py").write_text("INSTALL = ['discord.py==2.7.1']\n", encoding="utf-8")
    (tmp_path / "Pipfile.lock").write_text(_pipfile_lock("2.7.1", hashes=120), encoding="utf-8")
    text = collect_work_diff(
        workspace,
        sha,
        goal="outdated version of discord.py library it should be updated",
    )
    assert "setup.py" in text
    assert "discord.py==2.7.1" in text
    assert "discord.py: 2.3.2 -> 2.7.1" in text
    assert len(text) <= 16_000 + 80
    assert text.count("sha256") < 3


def _pipfile_lock(discord_version: str, *, hashes: int) -> str:
    blob = ",\n".join(f'                "sha256:{index:064x}"' for index in range(hashes))
    return (
        "{\n"
        '    "default": {\n'
        '        "aiohttp": {\n'
        f'            "hashes": [\n{blob}\n            ],\n'
        '            "version": "==3.8.1"\n'
        "        },\n"
        '        "discord.py": {\n'
        f'            "hashes": [\n{blob}\n            ],\n'
        f'            "version": "=={discord_version}"\n'
        "        }\n"
        "    }\n"
        "}\n"
    )


def test_collect_current_evidence_includes_goal_package_version(tmp_path: Path) -> None:
    (tmp_path / "setup.py").write_text("INSTALL = ['discord.py==2.7.1']\n", encoding="utf-8")
    (tmp_path / "Pipfile.lock").write_text(_pipfile_lock("2.7.1", hashes=4), encoding="utf-8")
    init_git_repo(tmp_path)
    text = collect_current_evidence(
        Workspace(tmp_path),
        "update the outdated discord.py library",
    )
    assert "setup.py" in text
    assert "discord.py: 2.7.1" in text
    assert "Current HEAD:" in text

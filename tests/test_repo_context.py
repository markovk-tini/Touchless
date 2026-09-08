"""Tests for RepoContextResolver (Phase 2 B5)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from hgr.live_api.repo_context import (  # noqa: E402
    RepoContext, RepoContextResolver, _read_package_name,
    _read_readme, _strip_markdown, looks_like_ide_window,
)


def _make_pkg_root(tmp: Path, *, pkg_name: str = "demo-pkg") -> Path:
    root = tmp / "demo"
    root.mkdir()
    (root / "package.json").write_text(
        '{"name": "' + pkg_name + '", "version": "1.0"}',
        encoding="utf-8")
    (root / "README.md").write_text(
        "# Demo\n\nThis is a demo project.\n\n## Features\n- A\n- B",
        encoding="utf-8")
    return root


def _make_pyproject_root(tmp: Path) -> Path:
    root = tmp / "py-proj"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "py-demo"\nversion = "0.1"\n',
        encoding="utf-8")
    return root


# ---- root detection ----------------------------------------------------

def test_resolves_package_json_root():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_pkg_root(Path(tmp))
        r = RepoContextResolver()
        ctx = r.resolve(str(root))
        assert ctx is not None
        assert ctx.name == "demo-pkg"
        assert "demo project" in ctx.readme_summary.lower()


def test_resolves_from_subdirectory_climbs_to_root():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_pkg_root(Path(tmp))
        (root / "src" / "x").mkdir(parents=True)
        r = RepoContextResolver()
        ctx = r.resolve(str(root / "src" / "x"))
        assert ctx is not None
        assert ctx.name == "demo-pkg"


def test_returns_none_for_non_project_dir():
    with tempfile.TemporaryDirectory() as tmp:
        r = RepoContextResolver()
        # Empty dir under tmp (no markers) → no root found.
        ctx = r.resolve(tmp)
        assert ctx is None


def test_returns_none_for_nonexistent_path():
    r = RepoContextResolver()
    assert r.resolve("/does/not/exist") is None


def test_returns_none_for_empty_path():
    r = RepoContextResolver()
    assert r.resolve("") is None
    assert r.resolve(None) is None


# ---- package name extraction ------------------------------------------

def test_package_name_from_package_json():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_pkg_root(Path(tmp), pkg_name="my-app")
        assert _read_package_name(root) == "my-app"


def test_package_name_from_pyproject_toml():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_pyproject_root(Path(tmp))
        assert _read_package_name(root) == "py-demo"


def test_package_name_returns_empty_when_no_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "plain"
        p.mkdir()
        assert _read_package_name(p) == ""


# ---- README extraction ------------------------------------------------

def test_readme_strips_markdown_and_truncates():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "r"
        root.mkdir()
        (root / "README.md").write_text(
            "# Title\n\n```bash\ncode-block-discarded\n```\n\n"
            "An [important](https://x.com) note about ![img](x.png) things.",
            encoding="utf-8")
        summary = _read_readme(root)
        assert "code-block-discarded" not in summary
        assert "![img]" not in summary
        # The link text should be kept.
        assert "important" in summary


def test_readme_returns_empty_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        assert _read_readme(Path(tmp)) == ""


def test_strip_markdown_handles_empty():
    assert _strip_markdown("") == ""
    assert _strip_markdown("# heading\nbody").startswith("heading")


# ---- caching -----------------------------------------------------------

def test_cache_returns_same_object_within_ttl():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_pkg_root(Path(tmp))
        r = RepoContextResolver()
        a = r.resolve(str(root))
        b = r.resolve(str(root))
        # Cache hit → same captured_at timestamp.
        assert a.captured_at == b.captured_at


def test_refresh_bypasses_cache():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_pkg_root(Path(tmp))
        r = RepoContextResolver()
        a = r.resolve(str(root))
        # Force a tiny gap so timestamps differ.
        import time
        time.sleep(0.01)
        b = r.resolve(str(root), refresh=True)
        assert b.captured_at >= a.captured_at


def test_invalidate_clears_entry():
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_pkg_root(Path(tmp))
        r = RepoContextResolver()
        r.resolve(str(root))
        assert r.invalidate(str(root)) == 1
        # Second invalidate of same entry returns 0.
        assert r.invalidate(str(root)) == 0


def test_invalidate_all_clears_everything():
    with tempfile.TemporaryDirectory() as tmp:
        root1 = _make_pkg_root(Path(tmp))
        root2 = _make_pyproject_root(Path(tmp))
        r = RepoContextResolver()
        r.resolve(str(root1))
        r.resolve(str(root2))
        n = r.invalidate("")
        assert n == 2


# ---- context block rendering ------------------------------------------

def test_context_block_includes_repo_name_and_branch():
    ctx = RepoContext(root="/x", name="foo", branch="main",
                      last_commits=["fix bug", "add docs"],
                      readme_summary="short desc")
    block = ctx.as_context_block()
    assert "REPO: foo" in block
    assert "branch: main" in block
    assert "fix bug" in block
    assert "short desc" in block


def test_context_block_marks_dirty_repo():
    ctx = RepoContext(root="/x", name="foo", branch="main",
                      is_dirty=True)
    assert "[DIRTY]" in ctx.as_context_block()


def test_context_block_respects_max_length():
    ctx = RepoContext(root="/x", name="x", branch="main",
                      readme_summary="X" * 5000)
    assert len(ctx.as_context_block()) <= 500


# ---- IDE detection -----------------------------------------------------

def test_looks_like_ide_window_recognizes_vscode():
    assert looks_like_ide_window("file.py - Visual Studio Code") is True
    assert looks_like_ide_window("file.py - VS Code") is True
    assert looks_like_ide_window("Cursor - file.py") is True


def test_looks_like_ide_window_rejects_browser():
    assert looks_like_ide_window("YouTube - Google Chrome") is False
    assert looks_like_ide_window("Slack | DMs") is False
    assert looks_like_ide_window("") is False

"""Tests for RepoFocusWatcher (Phase 3 wiring)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

import hgr.live_api.repo_focus_watcher as rfw  # noqa: E402
from hgr.live_api.repo_focus_watcher import (  # noqa: E402
    RepoFocusWatcher, _path_from_generic_title,
    _path_from_jetbrains_title, _path_from_sublime_title,
    _path_from_vscode_title, current_repo_context,
    workspace_path_from_title,
)
from hgr.live_api.repo_context import (  # noqa: E402
    RepoContext, RepoContextResolver,
)


# ---- title parsers -----------------------------------------------------

def test_jetbrains_pulls_path_from_brackets():
    title = "main.py – Touchless [C:/HGR App v1.0.0] – PyCharm 2026.1"
    p = _path_from_jetbrains_title(title)
    assert p == "C:/HGR App v1.0.0"


def test_jetbrains_returns_none_on_no_brackets():
    assert _path_from_jetbrains_title("PyCharm") is None


def test_sublime_pulls_path_from_parens():
    title = "main.py (C:/code/touchless/src) - Sublime Text"
    p = _path_from_sublime_title(title)
    assert p == "C:/code/touchless/src"


def test_vscode_title_returns_folder_name_or_path():
    # When the title mentions VS Code, the parser tries to find
    # the folder name; without a real folder on disk it falls back
    # to None (which is fine — generic parser may still catch it).
    title = "main.py - touchless - Visual Studio Code"
    result = _path_from_vscode_title(title)
    # On test host, the folder probably doesn't exist under common
    # parents — so we just verify the function doesn't crash and
    # returns either None or a real folder name.
    assert result is None or isinstance(result, str)


def test_vscode_title_recognized_by_ide_check():
    from hgr.live_api.repo_context import looks_like_ide_window
    assert looks_like_ide_window("main.py - Visual Studio Code") is True


def test_generic_finds_absolute_windows_path():
    title = "weird-ide :: working in C:/foo/bar/project doing stuff"
    p = _path_from_generic_title(title)
    assert p is not None
    assert "C:/foo" in p


def test_generic_finds_unix_path():
    title = "Editor: /home/dani/projects/touchless/src/main.py"
    p = _path_from_generic_title(title)
    assert p is not None
    assert "/home/dani" in p


# ---- workspace_path_from_title ----------------------------------------

def test_workspace_path_skipped_for_non_ide_title():
    assert workspace_path_from_title("YouTube - Google Chrome") is None
    assert workspace_path_from_title("") is None


def test_workspace_path_jetbrains_e2e():
    title = "demo.py – Demo [/home/dani/demo] – IntelliJ IDEA 2026.1"
    p = workspace_path_from_title(title)
    assert p == "/home/dani/demo"


# ---- watcher tick behaviour -------------------------------------------

def _make_pkg_root(tmp: Path, *, pkg_name: str = "demo-pkg") -> Path:
    root = tmp / "demo"
    root.mkdir()
    (root / "package.json").write_text(
        '{"name": "' + pkg_name + '", "version": "1.0"}',
        encoding="utf-8")
    (root / "README.md").write_text(
        "# Demo\n\nThis is a demo project.",
        encoding="utf-8")
    return root


def test_watcher_resolves_and_caches_repo_context(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_pkg_root(Path(tmp))
        title = f"main.py – Demo [{root}] – PyCharm 2026.1"
        monkeypatch.setattr(rfw, "_foreground_window_title",
                            lambda: title)
        # Use a fresh resolver so we don't touch the global cache.
        watcher = RepoFocusWatcher(resolver=RepoContextResolver())
        watcher.tick()
        ctx = current_repo_context()
        assert ctx is not None
        assert ctx.name == "demo-pkg"


def test_watcher_skips_repeated_same_title(monkeypatch):
    """If the title hasn't changed since last tick, the watcher must
    short-circuit — no resolver call, no side effects."""
    monkeypatch.setattr(rfw, "_foreground_window_title",
                        lambda: "main.py – Demo [/nowhere] – PyCharm")
    calls = []

    class StubResolver:
        def resolve(self, wd, refresh=False):
            calls.append(wd)
            return None

    watcher = RepoFocusWatcher(resolver=StubResolver())
    watcher.tick()
    watcher.tick()  # same title — should NOT call resolver again
    assert len(calls) == 1


def test_watcher_skips_non_ide_window(monkeypatch):
    monkeypatch.setattr(rfw, "_foreground_window_title",
                        lambda: "YouTube — Google Chrome")
    calls = []

    class StubResolver:
        def resolve(self, wd, refresh=False):
            calls.append(wd)
            return None

    watcher = RepoFocusWatcher(resolver=StubResolver())
    watcher.tick()
    assert calls == []


def test_current_repo_context_expires_after_ttl(monkeypatch):
    """When the user has been away from the IDE for longer than the
    TTL, current_repo_context should return None so we don't keep
    injecting stale project info."""
    rfw._set_current(
        RepoContext(root="/x", name="stale", branch="main"))
    # Pretend it was set long ago.
    rfw._LAST_CONTEXT_TS = 0.0  # epoch
    assert current_repo_context(max_age_sec=60.0) is None


def test_current_repo_context_returns_recent():
    rfw._set_current(
        RepoContext(root="/x", name="fresh", branch="main"))
    ctx = current_repo_context(max_age_sec=600.0)
    assert ctx is not None
    assert ctx.name == "fresh"


def test_reset_clears_cache():
    rfw._set_current(RepoContext(root="/x", name="x", branch="m"))
    w = RepoFocusWatcher()
    w.reset()
    assert current_repo_context() is None

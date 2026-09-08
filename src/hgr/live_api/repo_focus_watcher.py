"""IDE focus watcher — auto-loads repo context when the user switches
focus to a known IDE window with a real project open.

Phase-3 wiring. Glues the Phase-2 `RepoContextResolver` to a small
Sentinel watcher that polls the foreground window every few seconds.
When the title looks like an IDE (`looks_like_ide_window`) AND the
workspace path can be inferred from the title, the watcher resolves
the repo + caches the context block. Iris's planner can then read
the most-recent repo context via `current_repo_context()` and
inject it into prompts ("when I'm in this project, prefer X" etc.).

Two pieces:
  - `RepoFocusWatcher` — Sentinel-tickable; reads foreground window,
    extracts workspace path from title, calls
    `RepoContextResolver.resolve()`, stores the result.
  - `current_repo_context()` — module-level read for callers
    (planner, manager, etc.) that just want to know "what's the
    user looking at right now?"

Path-extraction heuristics per IDE:
  - **VS Code / Cursor / Codium**: title is "<file> — <folder>" or
    "<file> — <folder> [<workspace>]". Take the trailing folder name
    and try common parents (~/code, ~/projects, ~/Documents).
  - **JetBrains (IntelliJ, PyCharm, WebStorm, ...)**: title is
    "<project> [<workspace_path>] - <ide> <version>" or similar.
    Pull the explicit path from inside `[...]`.
  - **Sublime / Notepad++**: title is "<file> (<full_path>) — Sublime".
    Pull the path from `(...)`.

For unknown IDEs, fall back to scanning the title for any absolute
path. Resolver itself does the project-root climbing; we just need
to point it at something inside the project.

Design note: this is a BEST-EFFORT enhancer. When the watcher can't
parse a title, it just keeps the prior context. No crash, no false
positives.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, List, Optional

from .repo_context import (RepoContext, RepoContextResolver,
                            global_resolver, looks_like_ide_window)


DEFAULT_INTERVAL_SEC = 8.0
# How long the last-known context stays "current" after the user
# switches away from the IDE — after this, current_repo_context()
# returns None so callers don't keep injecting stale project info.
CURRENCY_TTL_SEC = 600.0


def _foreground_window_title() -> str:
    """Return the title of the currently-focused window. Empty string
    when unavailable (non-Windows, focus-less state, permission)."""
    if os.name != "nt":
        return ""
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value or ""
    except Exception:
        return ""


# Per-IDE title parsers. Order matters — more specific first.
# Each returns a candidate workspace path or None.
def _path_from_jetbrains_title(title: str) -> Optional[str]:
    # JetBrains: "<file> – <project> [<path>] – PyCharm 2026.1"
    m = re.search(r"\[([A-Za-z]:[\\/][^\]]+|/[^\]]+)\]", title)
    if m:
        return m.group(1).strip()
    return None


def _path_from_sublime_title(title: str) -> Optional[str]:
    # Sublime: "<file> (<full_path>) - Sublime Text"
    m = re.search(r"\(([A-Za-z]:[\\/][^)]+|/[^)]+)\)", title)
    if m:
        return m.group(1).strip()
    return None


def _path_from_vscode_title(title: str) -> Optional[str]:
    # VS Code / Cursor: "<file> - <folder> - Visual Studio Code"
    # The folder is just a name; we have to search common parents to
    # find a real path. Returns the folder name; resolver will check
    # ~/code, ~/projects, ~/Documents, etc.
    if "visual studio code" not in title.lower() \
            and "cursor" not in title.lower() \
            and "codium" not in title.lower():
        return None
    # Strip the trailing " - <ide>" tail.
    parts = re.split(r"\s+[-–—]\s+", title)
    if len(parts) < 2:
        return None
    # The folder is the second-to-last segment.
    folder = parts[-2].strip() if len(parts) >= 2 else ""
    folder = folder.strip("[]")
    if not folder or "/" in folder or "\\" in folder:
        # Already a path-like — use as-is.
        return folder or None
    # Try common parents.
    for parent_env in ("USERPROFILE", "HOME"):
        home = os.environ.get(parent_env)
        if not home:
            continue
        for sub in ("code", "projects", "Projects", "src", "dev",
                    "Documents", "Documents/GitHub",
                    "Documents/Projects"):
            cand = Path(home) / sub / folder
            if cand.exists() and cand.is_dir():
                return str(cand)
    return None


def _path_from_generic_title(title: str) -> Optional[str]:
    # Last-resort scan for any absolute path embedded in the title.
    m = re.search(
        r"([A-Za-z]:[\\/][\w .\-\\/]+(?:[\\/][\w .\-]+)?"
        r"|/(?:home|Users|opt|var)/[\w .\-/]+)",
        title)
    if m:
        return m.group(1).strip()
    return None


_PARSERS = (
    _path_from_jetbrains_title,
    _path_from_sublime_title,
    _path_from_vscode_title,
    _path_from_generic_title,
)


def workspace_path_from_title(title: str) -> Optional[str]:
    """Best-effort: pull a workspace directory path out of a window
    title. Returns None when no parser matches."""
    if not title or not looks_like_ide_window(title):
        return None
    for parser in _PARSERS:
        try:
            candidate = parser(title)
        except Exception:
            continue
        if candidate:
            return candidate
    return None


# Module-level cache of the most-recent context, so callers don't
# have to re-resolve on every prompt build.
_LAST_CONTEXT_LOCK = threading.RLock()
_LAST_CONTEXT: Optional[RepoContext] = None
_LAST_CONTEXT_TS: float = 0.0


def current_repo_context(*, max_age_sec: float = CURRENCY_TTL_SEC
                         ) -> Optional[RepoContext]:
    """Caller-side read. Returns the most-recently-resolved repo
    context, or None when nothing was resolved recently. The
    `max_age_sec` window means switching away from the IDE for a
    long while drops the context (so Iris doesn't keep injecting
    stale project info into prompts hours later)."""
    import time
    with _LAST_CONTEXT_LOCK:
        if _LAST_CONTEXT is None:
            return None
        if time.time() - _LAST_CONTEXT_TS > max_age_sec:
            return None
        return _LAST_CONTEXT


def _set_current(ctx: Optional[RepoContext]) -> None:
    import time
    global _LAST_CONTEXT, _LAST_CONTEXT_TS
    with _LAST_CONTEXT_LOCK:
        _LAST_CONTEXT = ctx
        _LAST_CONTEXT_TS = time.time()


class RepoFocusWatcher:
    """Sentinel-tickable watcher: polls the foreground window, extracts
    workspace path if it's an IDE, resolves the repo, stores the
    result for callers to read."""

    def __init__(self, *, resolver: Optional[RepoContextResolver]
                 = None) -> None:
        self._resolver = resolver or global_resolver()
        self._last_title: str = ""

    def tick(self) -> None:
        title = _foreground_window_title()
        if not title or title == self._last_title:
            # Optimization: same window as last tick → no work.
            return
        self._last_title = title
        if not looks_like_ide_window(title):
            return
        workspace = workspace_path_from_title(title)
        if not workspace:
            return
        try:
            ctx = self._resolver.resolve(workspace)
        except Exception:
            return
        if ctx is not None:
            _set_current(ctx)

    def reset(self) -> None:
        self._last_title = ""
        _set_current(None)


def register_with_sentinel(*, sentinel: Optional[Any] = None,
                           interval_sec: float = DEFAULT_INTERVAL_SEC
                           ) -> RepoFocusWatcher:
    watcher = RepoFocusWatcher()
    from .sentinel import global_sentinel
    s = sentinel or global_sentinel()
    s.register("repo_focus", watcher.tick,
               interval_sec=interval_sec, max_run_ms=400)
    return watcher

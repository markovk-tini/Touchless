"""Repo context auto-restore.

Phase-2. When the user switches focus to an IDE (VS Code, Cursor,
Codium, Sublime, RStudio, Visual Studio, JetBrains) with a folder
open, Iris should automatically pull in the project's context
(name, current branch, last 3 commit messages, top-level README
summary) and make it available as a memory-context block for the
planner/synthesizer.

Why this matters: without it, every Iris question while coding
becomes "what project am I in?" — the user has to repeat
context that's plainly visible on their screen. With it, "open
the docs" / "what did I just commit" / "summarize the README"
become first-class.

Auto-restore is OPT-OUT (env: TOUCHLESS_REPO_AUTO_RESTORE=0). The
implementation does only what's safe with NO network calls:
  * Detect IDE windows via psutil + window title heuristic.
  * Resolve the working directory from the window title or open
    file path (best-effort — IDE-specific).
  * Read README.md (first 800 chars), package metadata, branch
    via git, last 3 commit subjects via git.
  * Cache the result for `REPO_CACHE_TTL_SEC` so frequent focus
    changes don't trigger re-scans.

The detection layer is deliberately external — this module
exposes `RepoContextResolver` with a `resolve(working_dir)` method.
A separate UI component watches the focused window and calls
resolve() with the discovered path.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_CACHE_TTL_SEC = 300  # 5 min — branch + log refresh on focus.
README_MAX_CHARS = 800
MAX_COMMITS = 3


_IDE_HINTS = (
    "visual studio code", "vs code", "cursor", "codium", "sublime",
    "rstudio", "visual studio", "jetbrains", "intellij", "pycharm",
    "webstorm", "rider", "clion", "goland", "android studio",
)


@dataclass
class RepoContext:
    """Snapshot of a repo at a moment in time."""
    root: str
    name: str
    branch: str = ""
    last_commits: List[str] = field(default_factory=list)
    readme_summary: str = ""
    package_name: str = ""    # from package.json / pyproject.toml
    is_dirty: bool = False
    captured_at: float = field(default_factory=time.time)

    def as_context_block(self) -> str:
        """Render as a memory-context block ready for the planner
        prompt. ≤500 chars."""
        parts: List[str] = [f"REPO: {self.name} ({self.root})"]
        if self.branch:
            parts.append(f"  branch: {self.branch}"
                         + ("  [DIRTY]" if self.is_dirty else ""))
        if self.last_commits:
            parts.append("  recent:")
            parts.extend(f"    - {c}" for c in self.last_commits)
        if self.readme_summary:
            parts.append("  readme: "
                         + self.readme_summary[:300].replace("\n", " "))
        block = "\n".join(parts)
        return block[:500]


class RepoContextResolver:
    """Reads project metadata from a working directory. Caches
    results to avoid re-scanning on every focus change."""

    def __init__(self, *, cache_ttl_sec: int = REPO_CACHE_TTL_SEC,
                 git_cmd: str = "git") -> None:
        self._ttl = cache_ttl_sec
        self._git = git_cmd
        self._lock = threading.RLock()
        self._cache: Dict[str, RepoContext] = {}

    def resolve(self, working_dir: str,
                refresh: bool = False) -> Optional[RepoContext]:
        """Resolve repo context for `working_dir`. Returns None if
        the path doesn't exist or isn't actually a project."""
        wd = (working_dir or "").strip()
        if not wd:
            return None
        try:
            p = Path(wd).resolve()
        except Exception:
            return None
        if not p.is_dir():
            return None
        # Climb upward to find a project root (git or package marker).
        root = self._find_root(p)
        if root is None:
            return None
        key = str(root)
        with self._lock:
            cached = self._cache.get(key)
            if (cached and not refresh
                    and (time.time() - cached.captured_at) < self._ttl):
                return cached
        ctx = self._build_context(root)
        with self._lock:
            self._cache[key] = ctx
        return ctx

    def invalidate(self, working_dir: str = "") -> int:
        """Drop a single entry (or all if "") from the cache."""
        with self._lock:
            if not working_dir:
                n = len(self._cache)
                self._cache.clear()
                return n
            try:
                p = str(Path(working_dir).resolve())
            except Exception:
                p = working_dir
            return 1 if self._cache.pop(p, None) is not None else 0

    # ---- helpers ------------------------------------------------------

    @staticmethod
    def _find_root(start: Path) -> Optional[Path]:
        """Walk up to find the first parent containing .git/, or
        package.json / pyproject.toml / setup.py / Cargo.toml /
        go.mod. Returns None when nothing is found in the chain."""
        markers = (".git", "package.json", "pyproject.toml",
                   "setup.py", "Cargo.toml", "go.mod",
                   "Gemfile", "build.gradle", "pom.xml")
        cur: Optional[Path] = start
        for _ in range(12):  # at most 12 levels up
            if cur is None:
                return None
            if any((cur / m).exists() for m in markers):
                return cur
            if cur.parent == cur:
                return None
            cur = cur.parent
        return None

    def _build_context(self, root: Path) -> RepoContext:
        name = root.name
        readme = _read_readme(root)
        package_name = _read_package_name(root)
        branch, commits, dirty = "", [], False
        if (root / ".git").exists():
            branch = self._git_branch(root)
            commits = self._git_recent_commits(root)
            dirty = self._git_dirty(root)
        return RepoContext(
            root=str(root),
            name=package_name or name,
            branch=branch,
            last_commits=commits,
            readme_summary=readme,
            package_name=package_name,
            is_dirty=dirty,
        )

    def _git_branch(self, root: Path) -> str:
        try:
            out = subprocess.run(
                [self._git, "-C", str(root), "branch",
                 "--show-current"],
                capture_output=True, text=True, timeout=5,
                **_hidden_subproc_kwargs(),
            )
            return (out.stdout or "").strip()
        except Exception:
            return ""

    def _git_recent_commits(self, root: Path) -> List[str]:
        try:
            out = subprocess.run(
                [self._git, "-C", str(root), "log",
                 f"-n{MAX_COMMITS}", "--pretty=%s"],
                capture_output=True, text=True, timeout=5,
                **_hidden_subproc_kwargs(),
            )
            return [ln.strip() for ln in (out.stdout or "").splitlines()
                    if ln.strip()][:MAX_COMMITS]
        except Exception:
            return []

    def _git_dirty(self, root: Path) -> bool:
        try:
            out = subprocess.run(
                [self._git, "-C", str(root), "status",
                 "--porcelain"],
                capture_output=True, text=True, timeout=5,
                **_hidden_subproc_kwargs(),
            )
            return bool((out.stdout or "").strip())
        except Exception:
            return False


# ---- module helpers ----------------------------------------------------

def _read_readme(root: Path) -> str:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        p = root / name
        if p.exists() and p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            text = _strip_markdown(text)
            return text[:README_MAX_CHARS].strip()
    return ""


def _strip_markdown(text: str) -> str:
    # Remove fenced code blocks (rarely good context summary).
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Strip leading hashes from headings.
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    # Strip link syntax but keep text.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Strip image syntax outright.
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # Collapse multiple newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_package_name(root: Path) -> str:
    """Best-effort: read package name from package.json / pyproject /
    setup.py / Cargo.toml. Empty string when nothing found."""
    # package.json
    pj = root / "package.json"
    if pj.exists():
        try:
            import json
            data = json.loads(pj.read_text(encoding="utf-8", errors="ignore"))
            n = str(data.get("name") or "").strip()
            if n:
                return n
        except Exception:
            pass
    # pyproject.toml
    pp = root / "pyproject.toml"
    if pp.exists():
        try:
            txt = pp.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r'^\s*name\s*=\s*"([^"]+)"',
                          txt, re.MULTILINE)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
    # Cargo.toml
    cargo = root / "Cargo.toml"
    if cargo.exists():
        try:
            txt = cargo.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r'^\s*name\s*=\s*"([^"]+)"',
                          txt, re.MULTILINE)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
    return ""


def _hidden_subproc_kwargs() -> Dict[str, Any]:
    """Hide the console window on Windows when shelling out to git."""
    if os.name != "nt":
        return {}
    try:
        import subprocess as _sp
        startupinfo = _sp.STARTUPINFO()
        startupinfo.dwFlags |= _sp.STARTF_USESHOWWINDOW
        return {"startupinfo": startupinfo,
                "creationflags": _sp.CREATE_NO_WINDOW}
    except Exception:
        return {}


def looks_like_ide_window(window_title: str) -> bool:
    """True when the window title smells like an IDE we know about.
    Used by the focus watcher to decide whether to call resolve()."""
    t = (window_title or "").lower()
    return any(h in t for h in _IDE_HINTS)


# ---- module singleton --------------------------------------------------

_resolver: Optional[RepoContextResolver] = None
_lock = threading.Lock()


def global_resolver() -> RepoContextResolver:
    global _resolver
    if _resolver is None:
        with _lock:
            if _resolver is None:
                _resolver = RepoContextResolver()
    return _resolver


def _reset_for_tests() -> None:
    global _resolver
    _resolver = None

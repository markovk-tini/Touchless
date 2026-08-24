"""Project profile substrate — Iris knows what you're working on.

Phase-6 cognition. Today the planner has full access to memory
(`MemoryManager`), the entity graph (`entity_graph`), and screen
awareness (`screen_awareness`). What it DOESN'T have is a
project-level mental model: "what repo / project am I in right
now, what files have I been touching, what was the LAST thing I
did here, what's my open branch about?".

This substrate keeps a small per-project profile keyed by ABSOLUTE
project root path. For each profile we track:

  * Friendly name (defaults to the repo dir name, can be overridden
    by the user: 'this is my "Touchless" project').
  * Recent files touched (capped, sliding window).
  * Recent activity log — sparse, structured (e.g. "ran tests",
    "committed", "opened Iris UI").
  * Currently observed branch (git head) if available.
  * One-line "what is this?" summary the user can dictate.

When the planner is invoked, it consults `current_profile()` to
locate which project the user is in (active window's CWD, or the
folder of the file in the title bar). The profile is then woven
into the recall context like any other modality.

Persistence is SQLite-backed at:
    %LOCALAPPDATA%/Touchless/private/project_profiles.db

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_DB_DIR_ENV = "TOUCHLESS_PROJECT_PROFILE_DIR"
_MAX_RECENT_FILES = 24
_MAX_RECENT_ACTIVITY = 32


def _default_db_path() -> Path:
    override = os.environ.get(_DEFAULT_DB_DIR_ENV)
    if override:
        d = Path(override)
    else:
        local = os.environ.get(
            "LOCALAPPDATA",
            str(Path.home() / "AppData" / "Local"))
        d = Path(local) / "Touchless" / "private"
    d.mkdir(parents=True, exist_ok=True)
    return d / "project_profiles.db"


@dataclass
class ProjectProfile:
    root: str                                        # absolute path
    name: str = ""
    summary: str = ""
    branch: str = ""
    recent_files: List[str] = field(default_factory=list)
    recent_activity: List[Dict[str, Any]] = field(
        default_factory=list)
    last_seen_at: float = 0.0
    created_at: float = field(default_factory=time.time)

    def short_label(self) -> str:
        return self.name or Path(self.root).name or self.root

    def to_row(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "name": self.name,
            "summary": self.summary,
            "branch": self.branch,
            "recent_files": json.dumps(self.recent_files),
            "recent_activity": json.dumps(self.recent_activity),
            "last_seen_at": self.last_seen_at,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ProjectProfile":
        return cls(
            root=row["root"],
            name=row["name"] or "",
            summary=row["summary"] or "",
            branch=row["branch"] or "",
            recent_files=json.loads(row["recent_files"] or "[]"),
            recent_activity=json.loads(
                row["recent_activity"] or "[]"),
            last_seen_at=row["last_seen_at"] or 0.0,
            created_at=row["created_at"] or time.time(),
        )


class ProjectProfileStore:
    """SQLite-backed store. Single table + RLock for thread safety
    (per dictation architecture memory)."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS profiles (
        root TEXT PRIMARY KEY,
        name TEXT,
        summary TEXT,
        branch TEXT,
        recent_files TEXT,
        recent_activity TEXT,
        last_seen_at REAL,
        created_at REAL
    );
    CREATE INDEX IF NOT EXISTS idx_profiles_last_seen
        ON profiles(last_seen_at DESC);
    """

    def __init__(self, *, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path) if db_path else _default_db_path()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path),
            isolation_level=None,
            check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(self._SCHEMA)

    def upsert(self, profile: ProjectProfile) -> None:
        with self._lock:
            row = profile.to_row()
            self._conn.execute(
                """INSERT INTO profiles
                (root, name, summary, branch, recent_files,
                 recent_activity, last_seen_at, created_at)
                VALUES
                (:root, :name, :summary, :branch, :recent_files,
                 :recent_activity, :last_seen_at, :created_at)
                ON CONFLICT(root) DO UPDATE SET
                    name=excluded.name,
                    summary=excluded.summary,
                    branch=excluded.branch,
                    recent_files=excluded.recent_files,
                    recent_activity=excluded.recent_activity,
                    last_seen_at=excluded.last_seen_at""",
                row,
            )

    def get(self, root: str) -> Optional[ProjectProfile]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM profiles WHERE root = ?",
                (str(root),))
            r = cur.fetchone()
        return ProjectProfile.from_row(r) if r else None

    def recent(self, limit: int = 10) -> List[ProjectProfile]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM profiles "
                "ORDER BY last_seen_at DESC LIMIT ?", (int(limit),))
            return [ProjectProfile.from_row(r) for r in cur.fetchall()]

    def all(self) -> List[ProjectProfile]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM profiles ORDER BY last_seen_at DESC")
            return [ProjectProfile.from_row(r) for r in cur.fetchall()]

    def delete(self, root: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM profiles WHERE root = ?", (str(root),))
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ---- helpers ----------------------------------------------------------

def _is_project_root(p: Path) -> bool:
    """A directory is a 'project root' when it contains common
    project markers."""
    markers = (".git", "pyproject.toml", "package.json", "Cargo.toml",
               "go.mod", "pom.xml", "build.gradle", "Makefile",
               "CMakeLists.txt", ".hg", ".svn")
    try:
        for m in markers:
            if (p / m).exists():
                return True
    except Exception:
        pass
    return False


def find_project_root(path: Path | str) -> Optional[Path]:
    """Walk up from `path` until we hit a project marker. Returns
    None if we hit the filesystem root without finding one."""
    p = Path(path).resolve()
    if p.is_file():
        p = p.parent
    visited = 0
    while p and p != p.parent and visited < 50:
        if _is_project_root(p):
            return p
        p = p.parent
        visited += 1
    return None


def _git_branch(root: Path) -> str:
    """Read `.git/HEAD` without running git so it's fast and stays
    sandboxed."""
    head = root / ".git" / "HEAD"
    try:
        if not head.exists():
            return ""
        content = head.read_text(encoding="utf-8", errors="ignore"
                                ).strip()
        if content.startswith("ref:"):
            # 'ref: refs/heads/feature/foo' → 'feature/foo'
            parts = content.split("refs/heads/", 1)
            if len(parts) == 2:
                return parts[1].strip()
        # Detached HEAD → short sha
        return content[:8] if content else ""
    except Exception:
        return ""


# ---- singleton + facade ----------------------------------------------

_store_lock = threading.Lock()
_store: Optional[ProjectProfileStore] = None


def global_store() -> ProjectProfileStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = ProjectProfileStore()
        return _store


def reset_global_store() -> None:
    global _store
    with _store_lock:
        if _store is not None:
            try:
                _store.close()
            except Exception:
                pass
        _store = None


def touch(root: Path | str,
          *,
          file_hint: Optional[str] = None,
          activity: Optional[Dict[str, Any]] = None,
          name: Optional[str] = None,
          summary: Optional[str] = None) -> ProjectProfile:
    """Mark a project as currently active. Updates recency, optional
    file touch, optional activity event."""
    root = str(Path(root).resolve())
    store = global_store()
    existing = store.get(root)
    if existing is None:
        existing = ProjectProfile(root=root,
                                  name=Path(root).name)
    if name is not None:
        existing.name = name
    if summary is not None:
        existing.summary = summary
    branch = _git_branch(Path(root))
    if branch:
        existing.branch = branch
    if file_hint:
        norm = str(file_hint)
        if norm in existing.recent_files:
            existing.recent_files.remove(norm)
        existing.recent_files.insert(0, norm)
        existing.recent_files = (
            existing.recent_files[:_MAX_RECENT_FILES])
    if activity:
        entry = dict(activity)
        entry.setdefault("ts", time.time())
        existing.recent_activity.insert(0, entry)
        existing.recent_activity = (
            existing.recent_activity[:_MAX_RECENT_ACTIVITY])
    existing.last_seen_at = time.time()
    store.upsert(existing)
    return existing


def current_profile(active_path: Optional[Path | str] = None
                    ) -> Optional[ProjectProfile]:
    """Find the project profile for the path the user is currently
    in. If `active_path` is None, falls back to the most recent
    profile."""
    store = global_store()
    if active_path is not None:
        root = find_project_root(active_path)
        if root is not None:
            return store.get(str(root))
    recent = store.recent(limit=1)
    return recent[0] if recent else None


def render_for_planner(profile: ProjectProfile) -> str:
    """Render a profile as a short context block the planner can
    consume as the 'project' modality."""
    if profile is None:
        return ""
    parts: List[str] = []
    label = profile.short_label()
    if profile.summary:
        parts.append(f"In {label}: {profile.summary}.")
    else:
        parts.append(f"In {label} ({profile.root}).")
    if profile.branch:
        parts.append(f"Branch: {profile.branch}.")
    if profile.recent_files:
        # Show 5 most recent.
        files = profile.recent_files[:5]
        parts.append("Recent files: " + ", ".join(
            Path(f).name for f in files) + ".")
    if profile.recent_activity:
        # Show the most recent activity item only — older events
        # rot fast in voice context.
        last = profile.recent_activity[0]
        kind = last.get("kind", "activity")
        detail = last.get("detail", "")
        if detail:
            parts.append(f"Last {kind}: {detail}.")
        else:
            parts.append(f"Last {kind}.")
    return " ".join(parts)

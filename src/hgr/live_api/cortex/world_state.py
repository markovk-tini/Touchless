"""WorldState — persistent world model behind the Cortex visualization.

The Cortex is Iris's world model rendered as a network. This module
owns the persistent slice: projects Iris has touched, files Iris has
opened, apps Iris has launched, and the cross-links between them.
Lives at ``%LOCALAPPDATA%\\Touchless\\cortex_world.json`` on Windows
(falls back to ``~/.touchless/cortex_world.json`` elsewhere). Override
via the ``TOUCHLESS_CORTEX_WORLD`` env var.

Design:

  - **Append-on-touch.** Every Iris interaction emits a touch through
    ``cortex_emit.touch(...)``. The bus calls into this module. Each
    touch bumps a count + last_touched_at, auto-detects the project
    root if a file path was passed, and triggers a debounced save.
  - **Best-effort, never blocks.** All public methods catch and log
    exceptions. The cortex viz is decorative; a corrupt JSON or a
    permission error must NEVER break Iris's main loop.
  - **Thread-safe writes.** A single ``RLock`` guards all mutations.
    The save thread runs out of band — a QTimer in the GUI thread
    isn't appropriate because touches arrive on the WS reader thread.
    A ``threading.Timer``-based debounce is used instead.
  - **Atomic disk writes.** Write to ``cortex_world.json.tmp`` then
    ``os.replace()`` so a crash mid-write can't corrupt the file.

Schema (see docs/IRIS_CORTEX_EXPANSION.md for the full spec):

    {
      "version": 1,
      "projects": { "<project_id>": {label, root_path, color, ...} },
      "files":    { "<abs_path>":   {touches, last_touched_at, project_id} },
      "apps":     { "<app_name>":   {launches, last_at} },
      "cross_links": [ {from, to, kind, weight} ]
    }

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from math import log1p
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Files / folders that mark a directory as a "project root."
# Order is significant: earlier markers win when multiple are present
# (more specific markers preferred over README).
_PROJECT_MARKERS = (
    "CLAUDE.md",
    "OPEN_ISSUES.md",
    ".git",
    "pyproject.toml",
    "package.json",
    "README.md",
)

# Save debounce — coalesce rapid touches into one disk write.
_SAVE_DEBOUNCE_S = 5.0

# Stable schema version. Bump only when we change the shape of the
# JSON (and add a migration path).
_SCHEMA_VERSION = 1

# Default color used for new auto-detected projects.
_DEFAULT_COLOR = "teal"


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[cortex-world {time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_world_path() -> Path:
    """Return the default location for ``cortex_world.json``.

    Matches the ``default_memory_path()`` pattern in memory.manager so
    the cortex world sits next to memory.db. Override via
    ``TOUCHLESS_CORTEX_WORLD``.
    """
    override = os.environ.get("TOUCHLESS_CORTEX_WORLD")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Touchless" / "cortex_world.json"
    return Path.home() / ".touchless" / "cortex_world.json"


def detect_project_root(file_path: str | os.PathLike[str]) -> Optional[Path]:
    """Walk up from ``file_path`` looking for a project-root marker.

    Returns the first directory containing any of ``_PROJECT_MARKERS``,
    or None if we hit the filesystem root without finding one. Used to
    auto-attach a touched file to a (possibly new) project.

    The walk also stops at the user's home directory to avoid
    promoting "C:/Users/<name>" into a project just because there's a
    README somewhere upstream.
    """
    try:
        path = Path(file_path).resolve()
    except Exception:
        return None
    home = Path.home().resolve()
    # If the path is a file, start from its parent; if a directory,
    # check the directory itself first.
    cur = path.parent if path.is_file() else path
    while True:
        try:
            for marker in _PROJECT_MARKERS:
                if (cur / marker).exists():
                    return cur
        except Exception:
            return None
        # Stop conditions: at filesystem root, or at home dir (don't
        # treat ~/ as a project root).
        try:
            parent = cur.parent
        except Exception:
            return None
        if parent == cur:
            return None
        if cur == home:
            return None
        cur = parent


def _walk_up_to_project_root(
    file_path: str | os.PathLike[str],
    max_depth: int = 4,
) -> Optional[Path]:
    """Auto-promotion fallback: walk up from ``file_path`` looking for a
    project-root marker, with blocklist + depth guards.

    Distinct from ``detect_project_root`` in that it:
      * caps the walk at ``max_depth`` levels (default 4) to avoid
        runaway upward scans on slow / network filesystems;
      * reuses the system-folder blocklist from ``project_autodetect``
        (appdata, windows, program files, system32, $recycle.bin,
        node_modules, venv, build, dist, etc.) so we never promote a
        Windows system directory or a build artifact root into a
        project just because someone dropped a README there;
      * stops at the first matching marker — earlier markers in
        ``_PROJECT_MARKERS`` win (CLAUDE.md before README.md), same as
        ``detect_project_root``.

    Returns the first directory containing any of ``_PROJECT_MARKERS``,
    or None if max_depth is exhausted, the home dir is reached, the
    filesystem root is reached, or a blocked folder is hit.
    """
    # Lazy import to avoid a circular dependency at module load time —
    # project_autodetect doesn't import world_state today, but keeping
    # the import local is cheap insurance.
    try:
        from hgr.live_api.project_autodetect import _is_blocked_name
    except Exception:
        # Fallback inline blocklist so promotion still works if the
        # import path changes (e.g. relocated module). Mirrors the
        # contents of project_autodetect._NAME_BLOCKLIST + dot-prefix
        # rule.
        def _is_blocked_name(name: str) -> bool:  # type: ignore[no-redef]
            if not name:
                return True
            if name.startswith("."):
                return True
            return name.lower() in {
                "appdata", "windows", "program files",
                "program files (x86)", "programdata", "system32",
                "$recycle.bin", "system volume information", "recovery",
                "node_modules", "__pycache__", ".pytest_cache",
                "venv", ".venv", "env",
                "build", "dist", "out", "target",
            }

    try:
        path = Path(file_path).resolve()
    except Exception:
        return None
    try:
        home = Path.home().resolve()
    except Exception:
        home = None  # type: ignore[assignment]
    # If the path is a file, start from its parent; if a directory,
    # check the directory itself first.
    try:
        cur = path.parent if path.is_file() else path
    except Exception:
        cur = path.parent
    depth = 0
    while depth < max_depth:
        # Blocklist check: never promote system / build dirs.
        try:
            if _is_blocked_name(cur.name):
                return None
        except Exception:
            return None
        # Stop at home dir BEFORE checking markers so a README in
        # ~/ doesn't promote the home folder into a project.
        if home is not None and cur == home:
            return None
        # Marker check: first match wins, earlier markers preferred.
        try:
            for marker in _PROJECT_MARKERS:
                if (cur / marker).exists():
                    return cur
        except Exception:
            return None
        # Step up. Stop at filesystem root.
        try:
            parent = cur.parent
        except Exception:
            return None
        if parent == cur:
            return None
        cur = parent
        depth += 1
    return None


def derive_project_id(root_path: Path) -> str:
    """Stable, slug-ish id for a project given its root path."""
    name = root_path.name or "project"
    # Lowercase, replace whitespace + dots + non-alphanum with '-'.
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"


def derive_project_label(root_path: Path) -> str:
    """Human-readable label from the folder name."""
    name = root_path.name or "Project"
    # Replace separators with spaces; preserve casing roughly.
    return re.sub(r"[-_]+", " ", name).strip() or "Project"


class WorldState:
    """The persistent backing store for the Cortex world model.

    Instantiate once; call ``touch_*`` methods from anywhere. Saves
    automatically (debounced 5s) when the data has actually changed.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or default_world_path()
        self._lock = threading.RLock()
        self._dirty = False
        self._save_timer: Optional[threading.Timer] = None
        self._closed = False
        self._data: Dict[str, Any] = self._initial_data()
        # ---- observer registry (OPEN_ISSUES #10) -------------------------
        # Listeners get notified on add events so live consumers (the
        # cortex bridge, the iris simulator) can push updates to JS
        # without polling. Each observer is a callable(event_type, payload)
        # where event_type is one of {"project_added", "file_added"} and
        # payload is a small dict describing the change. Observers fire
        # OUTSIDE the lock to avoid deadlocks if a callback re-enters the
        # world. All errors swallowed — touches must never break Iris.
        self._observers: List[Any] = []
        # Set by _add_project_unlocked when a NEW project is created,
        # consumed and cleared by the wrapping public method (add_project /
        # touch_file) right after lock release. Keeps the "fire outside
        # the lock" invariant without changing _add_project_unlocked's
        # signature (still callable from anywhere under the lock).
        self._pending_project_event: Optional[Dict[str, Any]] = None
        self._load()

    # ---- public touch methods ---------------------------------------

    def touch_project(self, project_id: str) -> None:
        """Bump touch_count + last_touched on an existing project.

        No-op if the project_id isn't known."""
        with self._lock:
            proj = self._data["projects"].get(project_id)
            if proj is None:
                return
            proj["touch_count"] = int(proj.get("touch_count", 0)) + 1
            proj["last_touched_at"] = _now_iso()
            self._mark_dirty()

    def touch_file(self, file_path: str | os.PathLike[str]) -> Optional[str]:
        """Record a file touch. Auto-detects the project root and
        adds the project if it's new.

        Returns the project_id the file was attached to (or None if
        no project root could be detected). All errors are swallowed
        and logged — touches must never break Iris.
        """
        try:
            abs_path = str(Path(file_path).resolve())
        except Exception as exc:
            _log(f"touch_file: bad path {file_path!r}: {exc}")
            return None
        pending_project_evt: Optional[Dict[str, Any]] = None
        is_new_file = False
        with self._lock:
            project_id: Optional[str] = None
            root = detect_project_root(abs_path)
            # Fallback: if the unrestricted walk didn't find a project,
            # try the bounded + blocklisted auto-promotion walk. This
            # catches folders that have a marker file but are under a
            # parent we haven't seen before (e.g. a fresh clone outside
            # the user's usual project dirs). Wrapped in try/except so
            # any failure cleanly degrades to the existing "unattached
            # file" behavior — we must never break a touch.
            if root is None:
                try:
                    promoted = _walk_up_to_project_root(abs_path, max_depth=4)
                    if promoted is not None:
                        root = promoted
                except Exception as exc:
                    _log(f"touch_file: auto-promote failed for {abs_path!r}: {exc}")
                    root = None
            if root is not None:
                project_id = derive_project_id(root)
                # Auto-add the project if we've never seen this root.
                # Idempotent — _add_project_unlocked no-ops if the id
                # already exists, so the same path touched twice won't
                # produce duplicates or duplicate observer events.
                if project_id not in self._data["projects"]:
                    self._add_project_unlocked(
                        project_id=project_id,
                        label=derive_project_label(root),
                        root_path=str(root),
                        color=_DEFAULT_COLOR,
                    )
                # Bump it inline (saves one round-trip vs calling
                # touch_project after).
                proj = self._data["projects"][project_id]
                proj["touch_count"] = int(proj.get("touch_count", 0)) + 1
                proj["last_touched_at"] = _now_iso()

            # File log entry.
            files = self._data["files"]
            entry = files.get(abs_path)
            if entry is None:
                files[abs_path] = {
                    "touches": 1,
                    "first_touched_at": _now_iso(),
                    "last_touched_at": _now_iso(),
                    "project_id": project_id,
                }
                is_new_file = True
            else:
                entry["touches"] = int(entry.get("touches", 0)) + 1
                entry["last_touched_at"] = _now_iso()
                if project_id and not entry.get("project_id"):
                    entry["project_id"] = project_id

            self._mark_dirty()
            # Drain any pending project-add event from this call.
            if self._pending_project_event is not None:
                pending_project_evt = self._pending_project_event
                self._pending_project_event = None
        # ---- observer notifications (outside the lock) -------------------
        if pending_project_evt is not None:
            self._fire_observers("project_added", pending_project_evt)
        if is_new_file:
            self._fire_observers("file_added", {
                "path": abs_path,
                "label": Path(abs_path).name,
                "project_id": project_id,
            })
        return project_id

    def touch_app(self, app_name: str) -> None:
        """Record an app launch."""
        if not app_name:
            return
        name = str(app_name).strip()
        if not name:
            return
        with self._lock:
            apps = self._data["apps"]
            entry = apps.get(name)
            if entry is None:
                apps[name] = {
                    "launches": 1,
                    "first_at": _now_iso(),
                    "last_at": _now_iso(),
                }
            else:
                entry["launches"] = int(entry.get("launches", 0)) + 1
                entry["last_at"] = _now_iso()
            self._mark_dirty()

    # ---- public reads -----------------------------------------------

    def list_projects(self) -> List[Dict[str, Any]]:
        """Return a list of projects with their (computed) weight."""
        with self._lock:
            out: List[Dict[str, Any]] = []
            now = time.time()
            for pid, proj in self._data["projects"].items():
                weight = self._compute_project_weight_unlocked(pid, now)
                out.append({
                    "id": pid,
                    "label": proj.get("label", pid),
                    "color": proj.get("color", _DEFAULT_COLOR),
                    "root_path": proj.get("root_path"),
                    "touch_count": int(proj.get("touch_count", 0)),
                    "last_touched_at": proj.get("last_touched_at"),
                    "weight": weight,
                })
            # Stable order: highest weight first.
            out.sort(key=lambda p: p["weight"], reverse=True)
            return out

    def recent_files(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most-recently-touched files (newest first)."""
        with self._lock:
            items = list(self._data["files"].items())

        def _key(item: Tuple[str, Dict[str, Any]]) -> str:
            return item[1].get("last_touched_at", "") or ""

        items.sort(key=_key, reverse=True)
        out = []
        for abs_path, entry in items[: max(0, int(limit))]:
            out.append({
                "path": abs_path,
                "label": Path(abs_path).name,
                "touches": int(entry.get("touches", 0)),
                "last_touched_at": entry.get("last_touched_at"),
                "project_id": entry.get("project_id"),
            })
        return out

    def files_for_project(self, project_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Most-recently-touched files attached to a single project,
        newest first. Returns up to ``limit`` entries.

        Used by the cortex window to spawn preview-child leaves
        around each project sphere."""
        with self._lock:
            items = [
                (path, entry)
                for path, entry in self._data["files"].items()
                if entry.get("project_id") == project_id
            ]

        def _key(item: Tuple[str, Dict[str, Any]]) -> str:
            return item[1].get("last_touched_at", "") or ""

        items.sort(key=_key, reverse=True)
        out = []
        for abs_path, entry in items[: max(0, int(limit))]:
            out.append({
                "path": abs_path,
                "label": Path(abs_path).name,
                "touches": int(entry.get("touches", 0)),
                "last_touched_at": entry.get("last_touched_at"),
            })
        return out

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            proj = self._data["projects"].get(project_id)
            if proj is None:
                return None
            return dict(proj)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "projects": len(self._data["projects"]),
                "files": len(self._data["files"]),
                "apps": len(self._data["apps"]),
                "cross_links": len(self._data.get("cross_links", [])),
                "path": str(self._path),
            }

    # ---- explicit add (used by tests, future user UI) ---------------

    def add_project(
        self,
        project_id: str,
        label: str,
        root_path: Optional[str] = None,
        color: str = _DEFAULT_COLOR,
    ) -> None:
        pending_evt: Optional[Dict[str, Any]] = None
        with self._lock:
            self._add_project_unlocked(project_id, label, root_path, color)
            self._mark_dirty()
            # Drain pending event under the lock so we don't race with
            # another add_project from another thread.
            if self._pending_project_event is not None:
                pending_evt = self._pending_project_event
                self._pending_project_event = None
        if pending_evt is not None:
            self._fire_observers("project_added", pending_evt)

    def remove_project(self, project_id: str) -> bool:
        """Delete ``project_id`` from the world state.

        Also unlinks any files whose ``project_id`` pointed at it
        (they keep their entries but with ``project_id=None`` so the
        underlying touch history isn't lost). Fires a ``project_removed``
        observer event AFTER the lock is released so cortex bridges
        can refresh the visualization.

        Returns ``True`` if a project was actually removed,
        ``False`` if the id was unknown (idempotent no-op).
        """
        if not project_id:
            return False
        pid = str(project_id).strip()
        if not pid:
            return False
        removed_payload: Optional[Dict[str, Any]] = None
        with self._lock:
            proj = self._data["projects"].pop(pid, None)
            if proj is None:
                return False
            # Unlink files attached to the removed project so they don't
            # render under a missing parent. We keep the file entries
            # (touch history is valuable) but clear the project_id.
            files = self._data.get("files") or {}
            for fdata in files.values():
                if isinstance(fdata, dict) and fdata.get("project_id") == pid:
                    fdata["project_id"] = None
            removed_payload = {
                "project_id": pid,
                "label": proj.get("label", pid),
                "root_path": proj.get("root_path"),
                "color": proj.get("color"),
            }
            self._mark_dirty()
        # Fire outside the lock so callbacks can re-enter the world
        # without deadlocking (same pattern as add_project).
        if removed_payload is not None:
            self._fire_observers("project_removed", removed_payload)
        return True

    def _add_project_unlocked(
        self,
        project_id: str,
        label: str,
        root_path: Optional[str],
        color: str,
    ) -> None:
        if project_id in self._data["projects"]:
            return
        now = _now_iso()
        self._data["projects"][project_id] = {
            "label": label,
            "color": color,
            "root_path": root_path,
            "created_at": now,
            "last_touched_at": now,
            "touch_count": 0,
            "categories": [],
        }
        _log(f"auto-added project {project_id!r} (root={root_path})")
        # Stash so the outer caller can fire observers AFTER releasing
        # the lock — observers must never run under self._lock or a
        # callback that re-enters touch_* will deadlock.
        self._pending_project_event = {
            "project_id": project_id,
            "label": label,
            "root_path": root_path,
            "color": color,
        }

    # ---- observer registry (OPEN_ISSUES #10) ----------------------------

    def add_observer(self, callback) -> None:
        """Register a callable(event_type, payload) for add events.

        Fired AFTER the lock is released so callbacks can safely
        re-enter the world. All exceptions in callbacks are caught and
        logged — observer failures must never break a touch."""
        with self._lock:
            if callback not in self._observers:
                self._observers.append(callback)

    def remove_observer(self, callback) -> None:
        with self._lock:
            try:
                self._observers.remove(callback)
            except ValueError:
                pass

    def _fire_observers(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Notify all observers. MUST be called outside the lock."""
        # Snapshot under the lock so a concurrent add/remove doesn't
        # mutate the list while we iterate.
        with self._lock:
            observers = list(self._observers)
        for cb in observers:
            try:
                cb(event_type, payload)
            except Exception as exc:
                _log(f"observer {cb!r} failed for {event_type}: {exc}")
        # Cortex bridge direct emit — wired so non-simulator code paths
        # (tool_executor, weather, planner, …) cause the iris_simulator
        # constellation to refresh without each subsystem needing to know
        # about the bridge. No-op when the bridge isn't registered
        # (get_active_bridge() returns None). Decorative — touches must
        # NEVER break Iris.
        try:
            from .bridge import get_active_bridge
            bridge = get_active_bridge()
            if bridge is not None:
                if event_type == "project_added":
                    try:
                        bridge.emit_project_added(payload)
                    except Exception as exc:
                        _log(f"bridge.emit_project_added failed: {exc}")
                elif event_type == "file_added":
                    try:
                        bridge.emit_leaf_added(payload)
                    except Exception as exc:
                        _log(f"bridge.emit_leaf_added failed: {exc}")
                elif event_type == "project_removed":
                    # Best-effort: the bridge may not have a discrete
                    # signal for removals yet — getattr-probe so we
                    # work against older bridges and don't crash if
                    # the helper is absent.
                    emitter = getattr(bridge, "emit_project_removed", None)
                    if callable(emitter):
                        try:
                            emitter(payload)
                        except Exception as exc:
                            _log(f"bridge.emit_project_removed failed: {exc}")
        except Exception as exc:
            _log(f"bridge emit dispatch failed for {event_type}: {exc}")

    # ---- weight computation -----------------------------------------

    def _compute_project_weight_unlocked(self, pid: str, now_epoch: float) -> float:
        """Map (touch_count, file_count, recency) → 0.15..1.0 weight.

        Held under the lock by callers; do not call directly.
        """
        proj = self._data["projects"].get(pid)
        if proj is None:
            return 0.15
        touch_count = int(proj.get("touch_count", 0))
        # File count is the number of files attached to this project.
        # O(N) over the files table; fine for the small N we expect
        # (<= few thousand files).
        file_count = sum(
            1 for entry in self._data["files"].values()
            if entry.get("project_id") == pid
        )
        # Recency bonus: 0 at >30 days, ramps to 0.25 at very recent.
        recency = 0.0
        last_iso = proj.get("last_touched_at")
        if last_iso:
            try:
                last = datetime.strptime(last_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now_epoch - last.timestamp()) / 86400.0)
                recency = max(0.0, 0.25 * (1.0 - min(1.0, age_days / 30.0)))
            except Exception:
                recency = 0.0
        raw = (log1p(touch_count) / 5.0) + (log1p(file_count) / 8.0) + recency
        # Clamp into the visible range — never zero, never >1.
        return max(0.15, min(1.0, raw))

    # ---- persistence ------------------------------------------------

    def save_now(self) -> None:
        """Synchronous, atomic write. Use sparingly — prefer the
        debounced path via ``_mark_dirty``."""
        with self._lock:
            if not self._dirty:
                return
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(self._path.suffix + ".tmp")
                payload = json.dumps(self._data, indent=2, ensure_ascii=False)
                tmp.write_text(payload, encoding="utf-8")
                os.replace(tmp, self._path)
                self._dirty = False
            except Exception as exc:
                _log(f"save_now failed: {exc}")

    def shutdown(self) -> None:
        """Cancel any pending timer + flush to disk."""
        with self._lock:
            self._closed = True
            if self._save_timer is not None:
                try:
                    self._save_timer.cancel()
                except Exception:
                    pass
                self._save_timer = None
        # Final synchronous flush outside the lock to avoid blocking
        # other callers (save_now reacquires the lock).
        self.save_now()

    # ---- internals --------------------------------------------------

    @staticmethod
    def _initial_data() -> Dict[str, Any]:
        return {
            "version": _SCHEMA_VERSION,
            "projects": {},
            "files": {},
            "apps": {},
            "cross_links": [],
        }

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                _log(f"world JSON not a dict ({type(parsed).__name__}); ignoring")
                return
            # Shape check + fill missing sections.
            for key, default in self._initial_data().items():
                parsed.setdefault(key, default)
            self._data = parsed
            _log(f"loaded {len(self._data['projects'])} projects, "
                 f"{len(self._data['files'])} files from {self._path}")
        except Exception as exc:
            _log(f"load failed ({exc}); starting fresh")
            self._data = self._initial_data()

    def _mark_dirty(self) -> None:
        """Mark state dirty and schedule a debounced disk write.

        Caller must hold ``self._lock``."""
        self._dirty = True
        if self._closed:
            return
        if self._save_timer is not None:
            try:
                self._save_timer.cancel()
            except Exception:
                pass
        self._save_timer = threading.Timer(_SAVE_DEBOUNCE_S, self._timer_fire)
        self._save_timer.daemon = True
        self._save_timer.start()

    def _timer_fire(self) -> None:
        try:
            self.save_now()
        except Exception as exc:
            _log(f"timer_fire save failed: {exc}")


# ---- module-level singleton accessor ------------------------------------

_singleton_lock = threading.Lock()
_singleton: Optional[WorldState] = None


def get_world() -> WorldState:
    """Return the process-wide WorldState, instantiating on first use.

    Lazy so tests / standalone tools can preempt it by setting the
    ``TOUCHLESS_CORTEX_WORLD`` env var before the first call."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = WorldState()
        return _singleton


def reset_world(new_world: Optional[WorldState] = None) -> None:
    """Replace the singleton (used by tests). Shuts down the prior
    instance to flush pending writes."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            try:
                _singleton.shutdown()
            except Exception:
                pass
        _singleton = new_world

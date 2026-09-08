"""File-system watcher daemon for the RulesEngine.

Phase-3 wiring. The Phase-3-built `file_watcher_rules.RulesEngine`
matches `FsEvent` objects against user-saved rules and dispatches
to a callable. This module produces the `FsEvent` stream.

Two backends:

  * **polling** (default, always available) — walks each unique
    rule-root every N seconds (default 2 s), keeps a snapshot of
    `{path: (mtime, size)}` per root, emits CREATED / MODIFIED /
    DELETED events on diff.

  * **watchdog** (optional, when `watchdog` is installed) — real-
    time `Observer`-backed events. Not the default because watchdog
    isn't in `requirements.txt` yet (per CLAUDE.md rule #6, adding
    it requires PyInstaller hiddenimports too).

The poller is intentionally cheap: scans only the directories that
ANY enabled rule's path-glob references, not the whole disk. Caps
events per tick so a 50,000-file dump doesn't pin the Sentinel.

Wiring:
  - `register_with_sentinel(engine, dispatcher)` builds a watcher
    instance, registers its tick as a Sentinel watcher, and binds
    the dispatcher (typically `tool_registry.call`) to the engine.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .file_watcher_rules import FsEvent, FsEventKind, Rule


# Default poll interval. 2 sec is the rate-limit floor most rules
# care about anyway (the engine itself rate-limits per-rule).
DEFAULT_INTERVAL_SEC = 2.0

# Cap how many file-system events we emit per tick. A user dumping
# 50,000 files into a watched folder shouldn't pin the Sentinel —
# we'll just process the first N per tick and pick up the rest on
# subsequent ticks.
MAX_EVENTS_PER_TICK = 200

# Hard cap on number of files we'll keep snapshots for per root.
# Protects memory; a root with >50k files is the wrong place to
# put a watch-rule.
MAX_FILES_PER_ROOT = 50_000


Dispatcher = Callable[[str, Dict[str, Any]], Any]


def _extract_roots_from_rules(rules: Iterable[Rule]) -> List[Path]:
    """Pull the (deduped, expanded) set of root paths from the rules'
    path_globs. A glob like `~/Downloads/**/*.pdf` → `~/Downloads`.
    `*.png` (no slash) → user home as fallback root.

    Conservative: when in doubt, returns the user's home so the
    poller scans SOMETHING rather than nothing. Caller can tighten
    with require_path_contains."""
    roots: List[Path] = []
    seen: set = set()
    home = Path.home()
    for rule in rules:
        g = (rule.path_glob or "").strip()
        if not g:
            continue
        # Expand ~ first.
        if g.startswith("~"):
            try:
                g = str(Path(g).expanduser())
            except Exception:
                continue
        # Normalize separators.
        g = g.replace("\\", "/")
        # Take the prefix up to the first glob meta-character.
        meta_pos = _first_meta_pos(g)
        if meta_pos < 0:
            # No glob chars: the whole string is a directory or file.
            candidate = Path(g)
        else:
            prefix = g[:meta_pos]
            if "/" in prefix:
                # prefix is "~/Downloads/" — strip the trailing /
                # before treating it as the root. os.path.dirname on
                # a trailing-slash string returns the directory itself.
                candidate = Path(prefix.rstrip("/"))
            else:
                # prefix is "" or a bare filename — fall back to home.
                candidate = home
        if not candidate.parts:
            candidate = home
        try:
            root = candidate.resolve()
        except Exception:
            continue
        if not root.exists() or not root.is_dir():
            continue
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


_GLOB_META_RE = re.compile(r"[*?\[]")


def _first_meta_pos(s: str) -> int:
    m = _GLOB_META_RE.search(s)
    return m.start() if m else -1


def _snapshot_root(root: Path) -> Dict[str, Tuple[float, int]]:
    """Walk `root` and return `{path: (mtime, size)}` for every regular
    file, capped at MAX_FILES_PER_ROOT to bound memory."""
    out: Dict[str, Tuple[float, int]] = {}
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                try:
                    st = os.stat(full)
                    out[full] = (st.st_mtime, st.st_size)
                except OSError:
                    continue
                if len(out) >= MAX_FILES_PER_ROOT:
                    return out
    except (OSError, PermissionError):
        return out
    return out


def _diff_snapshots(prev: Dict[str, Tuple[float, int]],
                    curr: Dict[str, Tuple[float, int]]
                    ) -> List[FsEvent]:
    """Produce FsEvent objects describing the difference between two
    snapshots of the same root."""
    events: List[FsEvent] = []
    now = time.time()
    prev_keys = set(prev)
    curr_keys = set(curr)
    # CREATED.
    for path in curr_keys - prev_keys:
        mtime, size = curr[path]
        events.append(FsEvent(kind=FsEventKind.CREATED,
                              path=path, size_bytes=size,
                              ts=mtime or now))
    # DELETED.
    for path in prev_keys - curr_keys:
        events.append(FsEvent(kind=FsEventKind.DELETED,
                              path=path, size_bytes=0, ts=now))
    # MODIFIED (mtime or size changed).
    for path in prev_keys & curr_keys:
        if prev[path] != curr[path]:
            mtime, size = curr[path]
            events.append(FsEvent(kind=FsEventKind.MODIFIED,
                                  path=path, size_bytes=size,
                                  ts=mtime or now))
    return events


class FileSystemWatcher:
    """Polling watcher. Subscribes the dispatcher to the RulesEngine
    output. Stateful — keeps per-root snapshots between ticks."""

    def __init__(self, *,
                 engine: Any,
                 dispatcher: Dispatcher,
                 max_events_per_tick: int = MAX_EVENTS_PER_TICK
                 ) -> None:
        self._engine = engine
        self._dispatcher = dispatcher
        self._max = int(max_events_per_tick)
        # Per-root snapshot. Lazy-built on first tick so the very
        # first scan establishes a baseline WITHOUT firing events
        # for everything already in the folder.
        self._snapshots: Dict[str, Dict[str, Tuple[float, int]]] = {}
        self._first_tick = True

    def tick(self) -> None:
        """One pass: re-fetch rules, walk each root, diff against the
        previous snapshot, dispatch matched events. Cheap.
        Idempotent — safe to call from the Sentinel."""
        try:
            rules = self._engine.rules(include_disabled=False)
        except Exception:
            return
        if not rules:
            return
        roots = _extract_roots_from_rules(rules)
        if not roots:
            return
        events: List[FsEvent] = []
        for root in roots:
            curr = _snapshot_root(root)
            prev = self._snapshots.get(str(root))
            if prev is None:
                # First time we see this root — just baseline; never
                # emit synthetic CREATED events for every file that
                # was already there before the user added the rule.
                self._snapshots[str(root)] = curr
                continue
            diff = _diff_snapshots(prev, curr)
            self._snapshots[str(root)] = curr
            events.extend(diff)
            if len(events) >= self._max:
                break
        if self._first_tick:
            # Even after the loop, never act on first-tick events —
            # all of those came from "the world existed before we
            # started watching".
            self._first_tick = False
            return
        # Cap before dispatch.
        for event in events[:self._max]:
            try:
                self._engine.dispatch(event, self._dispatcher)
            except Exception:
                continue

    def reset(self) -> None:
        """Drop snapshots so the next tick re-baselines. Useful when
        rules change and the user removes a watched root."""
        self._snapshots.clear()
        self._first_tick = True


def register_with_sentinel(*,
                           engine: Any,
                           dispatcher: Dispatcher,
                           sentinel: Optional[Any] = None,
                           interval_sec: float = DEFAULT_INTERVAL_SEC,
                           ) -> FileSystemWatcher:
    """Build the watcher, register the tick with the Sentinel.
    Returns the watcher instance so callers can reset() it after
    rule changes.

    Phase-4: when `watchdog` is installed, ALSO start a real-time
    observer on each rule root so events fire instantly instead of
    waiting up to `interval_sec` for the next poll. The polling
    tick is kept as a safety net (watchdog can miss events on
    certain network filesystems / virtualized FS / Windows VSS)."""
    watcher = FileSystemWatcher(engine=engine, dispatcher=dispatcher)
    from .sentinel import global_sentinel
    s = sentinel or global_sentinel()
    s.register("file_watcher", watcher.tick,
               interval_sec=interval_sec, max_run_ms=1500)
    # Best-effort real-time layer. Doesn't replace polling — runs
    # alongside it. Failure to start (watchdog not installed,
    # PermissionError on the root) just leaves us with the polling
    # path; user experience is unchanged.
    try:
        _attach_watchdog_realtime(watcher)
    except Exception:
        pass
    return watcher


def _attach_watchdog_realtime(watcher: "FileSystemWatcher") -> None:
    """Start a watchdog `Observer` on each rule root so the engine
    sees events with sub-second latency. No-ops when watchdog isn't
    available."""
    try:
        from watchdog.observers import Observer  # type: ignore
        from watchdog.events import (  # type: ignore
            FileSystemEventHandler)
    except Exception:
        return  # watchdog not installed — polling stays the only path

    class _Handler(FileSystemEventHandler):  # type: ignore[misc]
        def __init__(self, w):
            super().__init__()
            self._w = w

        def _fire(self, kind, path):
            if not path:
                return
            try:
                size = os.path.getsize(path) if os.path.exists(path) else 0
            except OSError:
                size = 0
            evt = FsEvent(kind=kind, path=path, size_bytes=size)
            try:
                self._w._engine.dispatch(evt, self._w._dispatcher)
            except Exception:
                return

        def on_created(self, event):
            if getattr(event, "is_directory", False):
                return
            self._fire(FsEventKind.CREATED, event.src_path)

        def on_modified(self, event):
            if getattr(event, "is_directory", False):
                return
            self._fire(FsEventKind.MODIFIED, event.src_path)

        def on_deleted(self, event):
            if getattr(event, "is_directory", False):
                return
            self._fire(FsEventKind.DELETED, event.src_path)

        def on_moved(self, event):
            if getattr(event, "is_directory", False):
                return
            self._fire(FsEventKind.MOVED,
                       getattr(event, "dest_path", "") or event.src_path)

    try:
        rules = watcher._engine.rules(include_disabled=False)
    except Exception:
        return
    roots = _extract_roots_from_rules(rules)
    if not roots:
        return
    handler = _Handler(watcher)
    observer = Observer()
    for r in roots:
        try:
            observer.schedule(handler, str(r), recursive=True)
        except Exception:
            continue
    try:
        observer.daemon = True
        observer.start()
    except Exception:
        return
    # Stash on the watcher so future rule changes can stop+restart.
    watcher._watchdog_observer = observer

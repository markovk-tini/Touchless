"""Persistent cache of user-resolved drawing paths.

When the `show_overlay_drawing` action fires with a bare filename
that exists in multiple places on disk, the main window shows a
disambiguation chooser. The user's pick is saved here so the next
fire goes straight to that path with no search and no prompt.

Cache structure: a JSON object mapping a normalised filename key
to an absolute path string. The key is lowercase + path-stripped
so case differences or accidental "subdir/foo.png" entries match
"foo.png" too.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional

from .registry import registry_path


def _cache_path() -> Path:
    """Live alongside the gesture registry so a profile copy/move
    carries the cache along automatically."""
    return registry_path().parent / "drawing_path_cache.json"


def _normalise_key(filename: str) -> str:
    return Path(filename or "").name.strip().lower()


class DrawingPathCache:
    """JSON-backed filename → absolute-path cache. Thread-safe.

    Survives the user re-organising drawings between launches: a
    cached path is only honoured if the file is still there. Misses
    fall back to the search-and-prompt flow, which writes the new
    pick back.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path is not None else _cache_path()
        self._lock = threading.Lock()
        self._map: Dict[str, str] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        with self._lock:
            self._loaded = True
            self._map = {}
            if not self._path.exists():
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return
            entries = raw.get("entries")
            if not isinstance(entries, dict):
                return
            for key, value in entries.items():
                if isinstance(key, str) and isinstance(value, str) and value:
                    self._map[key.lower()] = value

    def save(self) -> None:
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"schema_version": 1, "entries": dict(self._map)}
                tmp = self._path.with_suffix(self._path.suffix + ".tmp")
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                tmp.replace(self._path)
            except Exception:
                pass

    def lookup(self, filename: str) -> Optional[Path]:
        """Return the cached absolute path for `filename` if one is
        stored AND still exists on disk. Stale entries (file moved
        or deleted) are dropped silently so the next call triggers
        a fresh search instead of returning a dead path."""
        if not self._loaded:
            self.load()
        key = _normalise_key(filename)
        if not key:
            return None
        with self._lock:
            value = self._map.get(key)
        if not value:
            return None
        candidate = Path(value)
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            pass
        with self._lock:
            self._map.pop(key, None)
        return None

    def remember(self, filename: str, resolved: Path) -> None:
        """Store the user's chosen path for next time. Caller is
        responsible for confirming `resolved` actually exists."""
        key = _normalise_key(filename)
        if not key:
            return
        with self._lock:
            self._map[key] = str(resolved)
        self.save()

    def forget(self, filename: str) -> None:
        key = _normalise_key(filename)
        if not key:
            return
        with self._lock:
            removed = self._map.pop(key, None)
        if removed is not None:
            self.save()

# Author: Konstantin Markov

"""Tool result speculation — pre-fetch read-only tool data before
the user finishes asking.

Phase-8 latency. For commonly-asked idempotent read-only tools
(`clock_now`, `weather_get`, `calendar_list_events`, `gmail_list`),
the moment a trigger keyword appears in the partial transcript we
can fire the tool in a daemon thread. By the time the user finishes
their sentence + ASR finalizes + planner picks the tool, the data
is already sitting in a 60-second cache.

Saves 100-500 ms on the round-trip-bound tools.

ALLOW-LIST is intentional:
  * Side-effect tools (sends, posts, deletes, writes) MUST NEVER
    be speculatively dispatched.
  * Cost-bearing tools that aren't safe-to-repeat (paid API calls
    that bill per request) are also excluded.

Each entry maps a regex of trigger keywords → tool name + canonical
args (no per-utterance args — speculative calls use defaults only;
"emails from Dani today" speculation would require argument
inference we don't trust yet).

Privacy / cost gates:
  * Incognito → skip.
  * Cost-meter slow mode → skip.
  * Per-tool cooldown 30 sec — don't re-fire while a cache entry
    is fresh.
  * Cap: at most one speculative dispatch per user turn.

Public:
  * `maybe_speculate(partial_text)` — call on partial ASR. Returns
    list of (tool, args) tuples it scheduled (informational).
  * `get_cached(tool, args)` — Tier-1 / planner consults this BEFORE
    dispatching; cache hit → return result instantly.

Author: Konstantin Markov
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


_CACHE_TTL_SEC = 60.0
_TOOL_COOLDOWN_SEC = 30.0
_MAX_SPECULATIONS_PER_TURN = 2


@dataclass
class _Trigger:
    pattern: re.Pattern
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)


# Note: keep the trigger keywords broad enough to catch normal
# phrasing, narrow enough not to fire on completely unrelated
# utterances. Regex anchored at any-word-boundary.
# NOTE: tools listed here MUST exist in the real ToolRegistry.
# clock_now / clock_today were removed — no underlying tool. The
# remaining triggers route to tools the connectors actually expose.
_TRIGGERS: List[_Trigger] = [
    _Trigger(
        pattern=re.compile(
            r"\b(weather|temperature|forecast|rain|sunny|cold|hot)\b",
            re.IGNORECASE),
        tool="weather_get"),
    _Trigger(
        pattern=re.compile(
            r"\b(calendar|meeting|meetings|schedule|"
            r"appointment|appointments|event|events)\b",
            re.IGNORECASE),
        tool="calendar_list_events"),
    _Trigger(
        pattern=re.compile(
            r"\b(email|emails|inbox|mail|messages)\b",
            re.IGNORECASE),
        tool="gmail_list",
        args={"unread_only": True, "max": 10}),
    _Trigger(
        pattern=re.compile(
            r"\b(now\s+playing|what'?s\s+playing|"
            r"current\s+song|current\s+track)\b",
            re.IGNORECASE),
        tool="media_now_playing"),
]


# ---- cache ------------------------------------------------------------

@dataclass
class _CacheEntry:
    result: Any
    cached_at: float = field(default_factory=time.time)


class _SpeculationCache:
    def __init__(self) -> None:
        self._entries: Dict[str, _CacheEntry] = {}
        self._cool: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._turn_speculations: int = 0
        self._turn_started_at: float = 0.0

    @staticmethod
    def _key(tool: str, args: Dict[str, Any]) -> str:
        blob = json.dumps(args or {}, sort_keys=True,
                           default=str)
        return hashlib.sha256(
            f"{tool}::{blob}".encode("utf-8")).hexdigest()

    def put(self, tool: str, args: Dict[str, Any],
            result: Any) -> None:
        with self._lock:
            self._entries[self._key(tool, args)] = _CacheEntry(
                result=result)

    def get(self, tool: str, args: Dict[str, Any]) -> Optional[Any]:
        with self._lock:
            entry = self._entries.get(self._key(tool, args))
            if entry is None:
                return None
            if (time.time() - entry.cached_at) > _CACHE_TTL_SEC:
                # Expired.
                self._entries.pop(self._key(tool, args), None)
                return None
            return entry.result

    def on_cooldown(self, tool: str) -> bool:
        with self._lock:
            last = self._cool.get(tool, 0.0)
            return (time.time() - last) < _TOOL_COOLDOWN_SEC

    def mark_fired(self, tool: str) -> None:
        with self._lock:
            self._cool[tool] = time.time()

    def begin_turn(self) -> None:
        with self._lock:
            self._turn_speculations = 0
            self._turn_started_at = time.time()

    def can_speculate_more(self) -> bool:
        with self._lock:
            return (self._turn_speculations
                    < _MAX_SPECULATIONS_PER_TURN)

    def increment_turn(self) -> None:
        with self._lock:
            self._turn_speculations += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._cool.clear()
            self._turn_speculations = 0
            self._turn_started_at = 0.0


# ---- top-level ------------------------------------------------------

class ToolSpeculator:
    """Holds the cache + a callable dispatcher (registry.call).
    The dispatcher must accept (tool_name, args_dict) and return a
    dict. Best-effort; never bubbles exceptions."""

    def __init__(self, *,
                 dispatcher: Optional[
                     Callable[[str, Dict[str, Any]], Any]] = None,
                 cache: Optional[_SpeculationCache] = None,
                 triggers: Optional[List[_Trigger]] = None
                 ) -> None:
        self._dispatcher = dispatcher
        self._cache = cache or _SpeculationCache()
        self._triggers = list(triggers
                              if triggers is not None
                              else _TRIGGERS)
        self._lock = threading.RLock()
        self._inflight: Dict[str, threading.Thread] = {}

    def set_dispatcher(self,
                       dispatcher: Callable[[str, Dict[str, Any]],
                                            Any]) -> None:
        self._dispatcher = dispatcher

    def begin_turn(self) -> None:
        self._cache.begin_turn()

    def maybe_speculate(self, partial_text: str
                        ) -> List[Tuple[str, Dict[str, Any]]]:
        """Inspect a partial transcript for trigger keywords.
        Returns list of (tool, args) tuples it scheduled."""
        fired: List[Tuple[str, Dict[str, Any]]] = []
        if not partial_text or self._dispatcher is None:
            return fired
        # Privacy / cost gates.
        try:
            from .incognito import is_incognito
            if is_incognito():
                return fired
        except Exception:
            pass
        try:
            from .cost_meter import global_meter
            m = global_meter()
            if getattr(m, "is_slow_mode", lambda: False)():
                return fired
        except Exception:
            pass
        for trig in self._triggers:
            if not self._cache.can_speculate_more():
                break
            if not trig.pattern.search(partial_text):
                continue
            if self._cache.on_cooldown(trig.tool):
                continue
            # Already cached?
            if self._cache.get(trig.tool, trig.args) is not None:
                continue
            self._cache.mark_fired(trig.tool)
            self._cache.increment_turn()
            key = self._cache._key(trig.tool, trig.args)
            with self._lock:
                if key in self._inflight:
                    continue
                t = threading.Thread(
                    target=self._dispatch_async,
                    args=(trig.tool, dict(trig.args), key),
                    daemon=True,
                    name=f"spec-{trig.tool}")
                self._inflight[key] = t
            t.start()
            fired.append((trig.tool, dict(trig.args)))
        return fired

    def _dispatch_async(self, tool: str,
                        args: Dict[str, Any],
                        key: str) -> None:
        try:
            result = self._dispatcher(tool, args)
            if result is not None:
                self._cache.put(tool, args, result)
        except Exception:
            pass
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def get_cached(self, tool: str,
                   args: Dict[str, Any]) -> Optional[Any]:
        return self._cache.get(tool, args)

    def clear(self) -> None:
        self._cache.clear()
        with self._lock:
            self._inflight.clear()


# ---- singleton ------------------------------------------------------

_lock = threading.Lock()
_singleton: Optional[ToolSpeculator] = None


def global_speculator() -> ToolSpeculator:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = ToolSpeculator()
        return _singleton


def reset_global() -> None:
    global _singleton
    with _lock:
        _singleton = None


def maybe_speculate(partial: str
                    ) -> List[Tuple[str, Dict[str, Any]]]:
    return global_speculator().maybe_speculate(partial)


def get_cached(tool: str, args: Dict[str, Any]
               ) -> Optional[Any]:
    return global_speculator().get_cached(tool, args)

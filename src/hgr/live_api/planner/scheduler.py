"""Rate-aware scheduler — sliding-window memory of recent calls and
rate-limit hits per "lane", so the orchestrator can avoid hammering a
lane that just got 429'd.

Two lanes today:
  - "realtime"   : the gpt-realtime websocket session
  - "cheap-llm"  : Chat Completions used by the planner / synthesizer

Process-wide singleton so the realtime session and the planner share the
same picture. Thread-safe; cheap (a small append-and-filter list).

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
import time
from typing import Callable, List, Tuple


class RateScheduler:
    """Tracks (timestamp, lane, kind) events where kind ∈ {'call','429'}.
    All `window` arguments are in seconds."""

    # Default windows (seconds). Realtime 429s linger longer because the
    # API's TPM bucket refills over ~minutes, not seconds.
    _REALTIME_429_WINDOW = 120.0
    _CHEAP_429_WINDOW = 30.0
    _GC_HORIZON = 300.0

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._events: List[Tuple[float, str, str]] = []

    # ---- ingest ------------------------------------------------------------
    def record_call(self, lane: str) -> None:
        self._push(lane, "call")

    def record_rate_limit(self, lane: str) -> None:
        self._push(lane, "429")

    # ---- queries -----------------------------------------------------------
    def is_throttled(self, lane: str, window: float = 60.0) -> bool:
        now = self._clock()
        with self._lock:
            return any(
                t >= now - window and l == lane and k == "429"
                for t, l, k in self._events
            )

    def prefer_cheap_planner(self) -> bool:
        """Realtime is rate-limited and cheap-LLM is healthy → planner
        should run Phase 2 even when the explicit opt-in flag is off."""
        return (
            self.is_throttled("realtime", self._REALTIME_429_WINDOW)
            and not self.is_throttled("cheap-llm", self._CHEAP_429_WINDOW)
        )

    def allow_cheap_synthesis(self) -> bool:
        """Skip the synthesizer call when cheap-LLM is throttled — the
        caller falls back to the deterministic _format_plan_message."""
        return not self.is_throttled("cheap-llm", self._CHEAP_429_WINDOW)

    # ---- internal ----------------------------------------------------------
    def _push(self, lane: str, kind: str) -> None:
        now = self._clock()
        with self._lock:
            self._events.append((now, lane, kind))
            # Keep only the last few minutes; the list stays tiny.
            cutoff = now - self._GC_HORIZON
            if self._events and self._events[0][0] < cutoff:
                self._events = [e for e in self._events if e[0] >= cutoff]


# ---- process-wide singleton -----------------------------------------------
_INSTANCE = RateScheduler()


def scheduler() -> RateScheduler:
    return _INSTANCE

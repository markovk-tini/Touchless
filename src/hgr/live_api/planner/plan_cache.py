"""Tiny in-memory plan cache. Skips the cheap-LLM planning call when the
same goal text recurs inside a short TTL — e.g. "set up my morning
briefing" said twice. Process-local, no disk persistence; the planner
is cheap enough that a multi-process cache isn't worth the complexity.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
import threading
import time
from typing import Callable, Dict, Optional, Tuple

from .plan import Plan


class PlanCache:
    """Normalized goal text → (timestamp, Plan). LRU on entry count, TTL
    per entry. Tiny: holds the last ~32 plans for ~10 minutes."""

    _WS_RE = re.compile(r"\s+")

    def __init__(self, ttl: float = 600.0, max_entries: int = 32,
                 clock: Callable[[], float] = time.time) -> None:
        self._ttl = ttl
        self._max = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._d: Dict[str, Tuple[float, Plan]] = {}

    @classmethod
    def _norm(cls, text: str) -> str:
        return cls._WS_RE.sub(" ", (text or "").strip().lower())

    def get(self, text: str) -> Optional[Plan]:
        key = self._norm(text)
        with self._lock:
            entry = self._d.get(key)
            if entry is None:
                return None
            ts, plan = entry
            if self._clock() - ts > self._ttl:
                self._d.pop(key, None)
                return None
            return plan

    def put(self, text: str, plan: Optional[Plan]) -> None:
        if plan is None or not plan.steps:
            return
        key = self._norm(text)
        with self._lock:
            # Evict the oldest if we've outgrown the cap (oldest by insertion
            # timestamp, not last-access — this is a near-LRU; good enough at
            # 32 entries).
            if len(self._d) >= self._max and key not in self._d:
                oldest = min(self._d.items(), key=lambda kv: kv[1][0])[0]
                self._d.pop(oldest, None)
            self._d[key] = (self._clock(), plan)

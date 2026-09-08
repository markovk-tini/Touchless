"""Per-session conversation buffer for short-term context.

Phase-3. The Tier-2 LLM planner gets a fresh prompt every call —
no history of what the user said three turns ago, what Iris
replied. That's fine when each request is self-contained ("set
volume to 30") but bites hard for follow-ups ("and on the other
monitor", "the second one", "do it again with Alice instead of
Dani").

This module owns a bounded rolling buffer of recent user/assistant
turns + a small render helper that produces a compact "RECENT
CONVERSATION" block for the planner prompt.

Bounded to ~12 turns / ~4000 chars so the prompt doesn't bloat;
the context_compressor module handles longer-term compaction
separately for memory-backed episodic recall.

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional


DEFAULT_MAX_TURNS = 12
DEFAULT_MAX_CHARS = 4000
# Turns older than this are dropped even if we're under the count
# limit — yesterday's conversation isn't relevant.
DEFAULT_TTL_SEC = 60 * 60.0  # 1 hour


@dataclass
class TurnEntry:
    role: str       # "user" | "assistant"
    text: str
    ts: float = field(default_factory=time.time)


class SessionBuffer:
    """Rolling buffer of recent conversation turns. Honors incognito
    (writes are no-ops when private mode is on)."""

    def __init__(self, *,
                 max_turns: int = DEFAULT_MAX_TURNS,
                 max_chars: int = DEFAULT_MAX_CHARS,
                 ttl_sec: float = DEFAULT_TTL_SEC) -> None:
        self._max_turns = int(max_turns)
        self._max_chars = int(max_chars)
        self._ttl = float(ttl_sec)
        self._lock = threading.RLock()
        self._turns: Deque[TurnEntry] = deque(maxlen=max_turns)

    def add_user(self, text: str) -> None:
        self._add("user", text)

    def add_assistant(self, text: str) -> None:
        self._add("assistant", text)

    def _add(self, role: str, text: str) -> None:
        try:
            from .incognito import is_incognito
            if is_incognito():
                return
        except Exception:
            pass
        clean = (text or "").strip()
        if not clean:
            return
        with self._lock:
            self._turns.append(TurnEntry(role=role, text=clean))
            self._evict_stale_locked()

    def _evict_stale_locked(self) -> None:
        cutoff = time.time() - self._ttl
        while self._turns and self._turns[0].ts < cutoff:
            self._turns.popleft()

    def recent(self, *, max_turns: Optional[int] = None
               ) -> List[TurnEntry]:
        with self._lock:
            self._evict_stale_locked()
            items = list(self._turns)
        if max_turns is not None:
            items = items[-int(max_turns):]
        return items

    def render(self, *, max_turns: Optional[int] = None) -> str:
        """Compact rendering for the planner prompt. Empty string
        when nothing in buffer (so the orchestrator's recall block
        can skip the heading)."""
        items = self.recent(max_turns=max_turns)
        if not items:
            return ""
        # Format: "user: ...\nassistant: ...\nuser: ..."
        # Cap each turn at 240 chars so a single long monologue
        # doesn't dominate the context budget.
        lines: List[str] = []
        for t in items:
            label = "user" if t.role == "user" else "iris"
            text = t.text if len(t.text) <= 240 else t.text[:237] + "…"
            lines.append(f"{label}: {text}")
        block = "\n".join(lines)
        # Final char cap — drop oldest lines until we fit.
        while len(block) > self._max_chars and len(lines) > 1:
            lines.pop(0)
            block = "\n".join(lines)
        return block

    def reset(self) -> None:
        with self._lock:
            self._turns.clear()


# ---- module singleton --------------------------------------------------

_buffer: Optional[SessionBuffer] = None
_lock = threading.Lock()


def global_session_buffer() -> SessionBuffer:
    global _buffer
    if _buffer is None:
        with _lock:
            if _buffer is None:
                _buffer = SessionBuffer()
    return _buffer


def _reset_for_tests() -> None:
    global _buffer
    with _lock:
        _buffer = None

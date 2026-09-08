"""Dictation ↔ Iris bridge.

Phase-2. Today the dictation system (whisper-stream → text-input
controller → paste into focused window) and the Iris assistant are
parallel universes. The user can dictate into Slack, then switch
context to ask Iris something, then go back to dictating. There's
no awareness between them. Iris can't read what was just dictated
to help reword it; dictation can't ask Iris a clarifying question
mid-flow.

This module establishes the bridge — a small, dependency-free
event bus that:

  * **Dictation publishes**: `text_inserted` events whenever the
    text-input controller pastes / types into a target window.
    Carries (text, target_window_title, ts).
  * **Iris publishes**: `iris_command` events when the assistant
    issues a command that should also be auditable in the
    dictation flow (e.g., Iris drafted a paragraph, dictation
    needs to know not to also paste it).
  * **Both subscribe**: each can listen to the other's events to
    avoid stepping on each other and to enrich context.

The intent isn't to fuse the two systems into one stream. It's to
make them stop being blind to each other so the user gets a
coherent experience.

The bus is process-local (single-process Touchless install). Phase
3 will likely add a multi-process variant for the Sentinel daemon.

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class BridgeEventKind(str, Enum):
    DICTATION_TEXT_INSERTED = "dictation_text_inserted"
    DICTATION_STARTED = "dictation_started"
    DICTATION_STOPPED = "dictation_stopped"
    IRIS_COMMAND_ISSUED = "iris_command_issued"
    IRIS_TEXT_DRAFTED = "iris_text_drafted"
    # Reserved for future use (Phase 3): focus changes, app switches.
    FOCUS_CHANGED = "focus_changed"


@dataclass
class BridgeEvent:
    kind: BridgeEventKind
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    source: str = ""              # "dictation" | "iris" | "system"


class DictationIrisBridge:
    """Tiny pub/sub event bus shared between dictation and Iris.

    Subscribers are simple callables; the bus catches exceptions so
    one bad subscriber can't break the rest."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: List[Callable[[BridgeEvent], None]] = []
        self._recent: List[BridgeEvent] = []
        self._max_recent = 64
        # Last seen dictation insertion — Iris reads this to know what
        # was just typed, so "summarize what I just dictated" works
        # without manual copy/paste.
        self._last_dictation_text: str = ""
        self._last_dictation_window: str = ""
        self._last_dictation_ts: float = 0.0

    # ---- pub/sub ------------------------------------------------------

    def publish(self, evt: BridgeEvent) -> None:
        with self._lock:
            self._recent.append(evt)
            if len(self._recent) > self._max_recent:
                self._recent.pop(0)
            if evt.kind == BridgeEventKind.DICTATION_TEXT_INSERTED:
                self._last_dictation_text = str(
                    evt.payload.get("text") or "")
                self._last_dictation_window = str(
                    evt.payload.get("window_title") or "")
                self._last_dictation_ts = evt.ts
            subs = list(self._subs)
        for s in subs:
            try:
                s(evt)
            except Exception as exc:
                import sys
                print(f"[bridge] subscriber raised "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)

    def subscribe(self, cb: Callable[[BridgeEvent], None]
                  ) -> Callable[[], None]:
        with self._lock:
            self._subs.append(cb)

        def _unsub():
            with self._lock:
                try:
                    self._subs.remove(cb)
                except ValueError:
                    pass
        return _unsub

    # ---- convenience accessors used by Iris ---------------------------

    def last_dictation_text(self, *, max_age_sec: float = 60.0
                            ) -> Optional[str]:
        """The most recently inserted text from dictation, if any
        and if recent enough that it's still relevant. Returns None
        when no dictation has happened recently."""
        with self._lock:
            if not self._last_dictation_text:
                return None
            if time.time() - self._last_dictation_ts > max_age_sec:
                return None
            return self._last_dictation_text

    def last_dictation_window(self) -> str:
        with self._lock:
            return self._last_dictation_window

    def recent_events(self, *, kind: Optional[BridgeEventKind] = None,
                      limit: int = 16) -> List[BridgeEvent]:
        with self._lock:
            evts = list(self._recent)
        if kind is not None:
            evts = [e for e in evts if e.kind == kind]
        return evts[-limit:]

    # ---- convenience publishers ---------------------------------------

    def publish_dictation_text(self, text: str, window_title: str
                               = "") -> None:
        self.publish(BridgeEvent(
            kind=BridgeEventKind.DICTATION_TEXT_INSERTED,
            payload={"text": text, "window_title": window_title},
            source="dictation",
        ))

    def publish_iris_command(self, command: str, tool: str = "",
                             result_summary: str = "") -> None:
        self.publish(BridgeEvent(
            kind=BridgeEventKind.IRIS_COMMAND_ISSUED,
            payload={"command": command, "tool": tool,
                     "result_summary": result_summary},
            source="iris",
        ))

    def reset(self) -> None:
        with self._lock:
            self._recent.clear()
            self._last_dictation_text = ""
            self._last_dictation_window = ""
            self._last_dictation_ts = 0.0


# ---- module singleton --------------------------------------------------

_bridge: Optional[DictationIrisBridge] = None
_lock = threading.Lock()


def global_bridge() -> DictationIrisBridge:
    global _bridge
    if _bridge is None:
        with _lock:
            if _bridge is None:
                _bridge = DictationIrisBridge()
    return _bridge


def _reset_for_tests() -> None:
    global _bridge
    _bridge = None

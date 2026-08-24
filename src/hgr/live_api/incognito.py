"""Incognito / private-session gate.

The single source of truth for "is Iris currently in private mode?".
Phase-1 trust substrate. When ON:

  * No new ToolInvocation rows are written to the audit log.
  * Memory store writes (facts, episodic turns) are dropped.
  * Telemetry / cortex events that contain user content are suppressed.
  * Self-learner skips its observation cycle for the session.
  * Activity pill displays a clear purple "INCOGNITO" indicator.

Read paths are NOT gated — Iris can still answer questions, use tools,
recall prior facts. The promise is "nothing this session enters
persistent state", NOT "Iris loses memory of last week".

Implementation: a module-level boolean guarded by a lock. Cheaper than
ContextVar (no per-thread state needed — the whole process toggles
together), and trivial to read from any thread.

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
from typing import Callable, List


_state_lock = threading.RLock()
_incognito: bool = False
_listeners: List[Callable[[bool], None]] = []


def is_incognito() -> bool:
    """Cheap read — called on EVERY tool invocation. Reading a bool
    under a lock is sub-microsecond; lock is a defensive choice
    against torn writes on 32-bit ARM if we ever cross-compile."""
    with _state_lock:
        return _incognito


def set_incognito(on: bool) -> bool:
    """Toggle private mode. Notifies listeners (activity pill, status
    overlay) on change. Returns the new state."""
    global _incognito
    with _state_lock:
        changed = bool(on) != _incognito
        _incognito = bool(on)
        listeners = list(_listeners) if changed else []
        new_state = _incognito
    for fn in listeners:
        try:
            fn(new_state)
        except Exception:
            # A listener bug must never block the toggle itself —
            # losing private-mode would be a privacy bug.
            import sys
            print(f"[incognito] listener raised: {fn}",
                  file=sys.stderr, flush=True)
    return new_state


def toggle_incognito() -> bool:
    """Flip state. Returns the new state. Bound to the global hotkey
    (Ctrl+Shift+I) elsewhere."""
    return set_incognito(not is_incognito())


def subscribe(fn: Callable[[bool], None]) -> Callable[[], None]:
    """Register a listener for state changes. Returns an unsubscribe
    callable. Listener gets the new boolean state."""
    with _state_lock:
        _listeners.append(fn)
    def _unsub() -> None:
        with _state_lock:
            try:
                _listeners.remove(fn)
            except ValueError:
                pass
    return _unsub


def _reset_for_tests() -> None:
    """Test helper. Resets the module to initial state. Do NOT call
    from production code."""
    global _incognito
    with _state_lock:
        _incognito = False
        _listeners.clear()

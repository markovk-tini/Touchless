"""Unified ToolInvocation contract.

THE single observation point for every tool call Iris makes, regardless
of execution path. Both the planner Executor (deterministic Tier-2
multi-step plans) and the realtime tool_executor (realtime model
calling tools mid-conversation) emit the same `ToolInvocation` shape
to the same bus.

Why a single contract: every Phase-1 trust feature subscribes here.
Activity pill, audit log, universal undo speed-bump, confirmation
gate, incognito guard, cost ledger — each is a downstream subscriber
that reads/writes a known shape. If the two execution paths kept
emitting different shapes (StepResult vs raw tool result dict), every
substrate feature would need two implementations and would silently
fail when only one path was instrumented.

This module is intentionally tiny + dependency-free so it can be
imported anywhere (including the realtime worker thread). The bus is
process-local; multi-process IPC (Sentinel daemon ↔ chat UI) is a
Phase-3 concern that wraps this with a serializer.

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .tool_metadata import Destructiveness, get_metadata


class InvocationSource(str, Enum):
    """Which execution path produced this invocation."""
    PLANNER = "planner"       # Tier-2 LLM plan, executed by planner.Executor
    REALTIME = "realtime"     # realtime model called this tool directly
    ROUTER = "router"         # Layer-0 deterministic voice/command router
    CLASSIFIER = "classifier" # Tier-1 deterministic regex match
    SKILL = "skill"           # Saved-skill replay (Tier-0.5)
    PSEUDO = "pseudo"         # Iris pseudo-tool (memory ops, etc.)
    BACKGROUND = "background" # Future Sentinel daemon / scheduled task


@dataclass
class ToolInvocation:
    """One observable tool call. Every Iris tool dispatch produces one
    of these. The fields are chosen so that every downstream feature
    (audit, undo, pill, gate, incognito) reads the same shape — adding
    a new subscriber should NEVER require changing this dataclass."""

    # ---- identity ----
    invocation_id: str           # unique per call (uuid4 hex)
    turn_id: Optional[str]       # groups invocations from the same user turn
    tool: str
    args: Dict[str, Any]
    source: InvocationSource

    # ---- result ----
    # status: "ok" | "error" | "cancelled" | "needs_confirmation"
    status: str = "ok"
    output: Any = None
    error: Optional[str] = None

    # ---- timing (epoch seconds) ----
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0

    # ---- safety metadata (snapshotted at invocation time) ----
    destructiveness: Destructiveness = Destructiveness.WRITE
    reversible: bool = False
    was_confirmed: bool = False
    was_undone: bool = False

    # ---- free-form context (subscriber-specific; don't store secrets here) ----
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        if not self.ended_at:
            return 0
        return int((self.ended_at - self.started_at) * 1000)

    @classmethod
    def starting(cls, *, tool: str, args: Dict[str, Any],
                 source: InvocationSource,
                 turn_id: Optional[str] = None,
                 was_confirmed: bool = False) -> "ToolInvocation":
        """Build an in-progress ToolInvocation. Metadata pulled from the
        DEFAULT_METADATA registry; caller fills in result fields before
        publishing."""
        meta = get_metadata(tool)
        return cls(
            invocation_id=uuid.uuid4().hex,
            turn_id=turn_id,
            tool=tool,
            args=dict(args or {}),
            source=source,
            destructiveness=meta.destructiveness,
            reversible=meta.reversible,
            was_confirmed=was_confirmed,
        )

    def complete(self, *, status: str, output: Any = None,
                 error: Optional[str] = None) -> "ToolInvocation":
        """Fill in the result fields + stamp ended_at. Returns self for
        chaining."""
        self.status = status
        self.output = output
        self.error = error
        self.ended_at = time.time()
        return self


# ---- bus ------------------------------------------------------------------


SubscriberFn = Callable[[ToolInvocation], None]


class InvocationBus:
    """Thread-safe pub/sub. Subscribers are called synchronously on the
    publishing thread — they MUST be fast (or do their own work-offload
    to a queue). The bus catches subscriber exceptions individually so
    one bad subscriber can't break the others or the execution path."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: List[SubscriberFn] = []

    def subscribe(self, fn: SubscriberFn) -> Callable[[], None]:
        """Register a subscriber. Returns an unsubscribe callable."""
        with self._lock:
            self._subscribers.append(fn)
        def _unsub() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(fn)
                except ValueError:
                    pass
        return _unsub

    def publish(self, inv: ToolInvocation) -> None:
        """Notify all subscribers. Subscriber exceptions are swallowed
        and printed to stderr — never propagated to the publisher.
        Snapshotting the list under lock so subscribe() during publish
        doesn't deadlock or skip.

        Honors INCOGNITO mode: when the user toggled private mode,
        invocations are tagged via inv.extra and NOT delivered to
        persistent subscribers (audit log, memory). In-memory observers
        that ONLY display state (activity pill) opt-in by reading
        inv.extra['incognito'] and rendering differently.

        The audit log + memory subscribers check is_incognito() in
        their own on_invocation handler — we tag rather than hard-skip
        so future ephemeral subscribers can still see what happened
        during the private session for UI purposes."""
        # Tag the invocation so subscribers can decide per-subscriber.
        try:
            from .incognito import is_incognito
            if is_incognito():
                inv.extra["incognito"] = True
        except Exception:
            pass
        with self._lock:
            subs = list(self._subscribers)
        for fn in subs:
            try:
                fn(inv)
            except Exception as exc:  # pragma: no cover - defensive
                import sys
                print(f"[invocation_bus] subscriber raised: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)

    def clear(self) -> None:
        """Drop all subscribers. For tests."""
        with self._lock:
            self._subscribers.clear()

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


# ---- module-level singleton ------------------------------------------------


_global_bus: Optional[InvocationBus] = None
_global_bus_lock = threading.Lock()


def global_bus() -> InvocationBus:
    """Lazy-initialized process-local singleton. All execution paths
    publish to this; all substrate features subscribe to this."""
    global _global_bus
    if _global_bus is None:
        with _global_bus_lock:
            if _global_bus is None:
                _global_bus = InvocationBus()
    return _global_bus


def publish(inv: ToolInvocation) -> None:
    """Convenience wrapper. Equivalent to `global_bus().publish(inv)`."""
    global_bus().publish(inv)

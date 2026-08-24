"""Proactive nudges — Iris's ambient "I noticed X" surface.

Phase-3 polish. Most user-facing Iris interactions are reactive
(user says something, Iris responds). The substrate that landed in
Phase-3 makes a small set of PROACTIVE nudges feasible without
adding new infrastructure:

  * **Long-idle break**: user hasn't touched the machine for ≥45 min
    during work hours → a gentle "want a break?" surface. Reads
    `USER_IDLE_SEC` from InterruptionGate. Fires once per idle
    session (resets when input resumes).

  * **Cost cap approaching**: today's spend crossed 80% of the cap →
    a one-time "you're at 80% of today's $5 budget" surface. Lets
    the user decide to bump the cap, kill a feature, or finish.
    Fires once per day per threshold (80, 95).

  * **Cumulative failures**: same connector errored ≥5 times across
    distinct turns → "this connector seems unhealthy; want me to
    pause it for this session?" Subscribes to the reliability
    ledger's per-tool counters.

Architecture mirrors the briefing watcher: a Sentinel-tickable
class checks state every ~30s, emits a `Nudge` through a callback
the UI / voice layer renders. The callback is supplied by the
caller so this module stays UI-free.

All nudges respect:
  * Incognito (no nudges in private mode)
  * InterruptionGate LOW severity (deferred during call / screen-share)
  * Per-kind cooldowns (no spam)

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set


class NudgeKind(str, Enum):
    LONG_IDLE = "long_idle"
    NEAR_COST_CAP = "near_cost_cap"
    AT_COST_CAP = "at_cost_cap"
    UNHEALTHY_TOOL = "unhealthy_tool"


@dataclass
class Nudge:
    kind: NudgeKind
    body: str
    severity: str = "low"   # 'low' | 'normal' | 'high'
    suggested_action: str = ""
    ts: float = field(default_factory=time.time)


NudgeHandler = Callable[[Nudge], None]


# Defaults (env-tunable).
LONG_IDLE_THRESHOLD_SEC = 45 * 60.0
NEAR_CAP_PCT = 80
AT_CAP_PCT = 95
UNHEALTHY_ERROR_THRESHOLD = 5
UNHEALTHY_WINDOW_SEC = 30 * 60.0
DEFAULT_TICK_INTERVAL_SEC = 30.0
# How long a nudge of a given kind stays "muted" after firing so we
# don't spam the user.
PER_KIND_COOLDOWN_SEC = {
    NudgeKind.LONG_IDLE:        60 * 60.0,   # once per hour
    NudgeKind.NEAR_COST_CAP:   12 * 3600.0,  # once per ~half-day
    NudgeKind.AT_COST_CAP:     24 * 3600.0,  # once per day
    NudgeKind.UNHEALTHY_TOOL:  30 * 60.0,    # once per 30 min per tool
}


class ProactiveNudges:
    """Sentinel-tickable: every ~30s, check the state of the world
    and fire any nudges whose conditions hold."""

    def __init__(self, *, handler: NudgeHandler,
                 cost_meter: Optional[Any] = None,
                 ledger: Optional[Any] = None,
                 gate: Optional[Any] = None) -> None:
        self._handler = handler
        self._cost_meter = cost_meter
        self._ledger = ledger
        self._gate = gate
        self._lock = threading.RLock()
        # Last-fire timestamp keyed by (kind, sub_key). sub_key is
        # the tool name for UNHEALTHY_TOOL, "" for the others.
        self._last_fire: Dict[tuple, float] = {}
        # Tracks which idle-session we've already nudged for so a
        # user idle for 2 hours only gets ONE break-suggest.
        self._idle_session_handled = False

    def tick(self) -> None:
        # Incognito: silently skip.
        try:
            from .incognito import is_incognito
            if is_incognito():
                return
        except Exception:
            pass
        # Be polite — only push nudges when the InterruptionGate
        # says it's OK to interrupt at LOW severity.
        try:
            from .interruption_gate import (global_gate,
                                             InterruptSeverity)
            gate = self._gate or global_gate()
            decision = gate.can_interrupt(InterruptSeverity.LOW)
            if not decision.allow:
                return
        except Exception:
            pass

        now = time.time()
        self._check_long_idle(now)
        self._check_cost_cap(now)
        self._check_unhealthy_tools(now)

    # ---- per-check ----------------------------------------------------

    def _check_long_idle(self, now: float) -> None:
        idle = self._read_user_idle_sec()
        if idle is None:
            return
        if idle < LONG_IDLE_THRESHOLD_SEC:
            # User is active — reset the session flag so the next
            # long-idle period can fire again.
            with self._lock:
                self._idle_session_handled = False
            return
        # Skip during night hours (1am-7am) — nudging someone awake
        # at 3am because they walked away is rude.
        h = datetime.now().hour
        if 1 <= h < 7:
            return
        with self._lock:
            if self._idle_session_handled:
                return
            if not self._cooldown_ok(NudgeKind.LONG_IDLE, "", now):
                return
            self._idle_session_handled = True
        minutes = int(idle // 60)
        self._emit(Nudge(
            kind=NudgeKind.LONG_IDLE,
            body=(f"You've been idle for {minutes} minutes — want me "
                  "to summarize anything when you're back?"),
            severity="low",
            suggested_action="ask_for_summary_on_return",
        ))

    def _check_cost_cap(self, now: float) -> None:
        meter = self._cost_meter
        if meter is None:
            try:
                from .cost_meter import global_meter
                meter = global_meter()
            except Exception:
                return
        try:
            from .cost_surfaces import cost_badge_state
            state = cost_badge_state(meter)
        except Exception:
            return
        if state.cap_usd <= 0:
            return  # user disabled the cap
        pct = state.percent_of_cap
        # AT cap takes precedence over NEAR cap.
        if pct >= AT_CAP_PCT:
            if self._cooldown_ok(NudgeKind.AT_COST_CAP, "", now):
                self._emit(Nudge(
                    kind=NudgeKind.AT_COST_CAP,
                    body=(f"Heads up — you're at ${state.spent_usd:.2f} "
                          f"of today's ${state.cap_usd:.2f} budget "
                          f"({pct:.0f}%). Paid model calls will block "
                          "soon."),
                    severity="high",
                    suggested_action="bump_or_pause_cap",
                ))
                self._mark_fired(NudgeKind.AT_COST_CAP, "", now)
            return
        if pct >= NEAR_CAP_PCT:
            if self._cooldown_ok(NudgeKind.NEAR_COST_CAP, "", now):
                self._emit(Nudge(
                    kind=NudgeKind.NEAR_COST_CAP,
                    body=(f"You're at {pct:.0f}% of today's "
                          f"${state.cap_usd:.2f} budget. Want to keep "
                          "going on local model for the rest of the "
                          "day?"),
                    severity="normal",
                    suggested_action="switch_to_local",
                ))
                self._mark_fired(NudgeKind.NEAR_COST_CAP, "", now)

    def _check_unhealthy_tools(self, now: float) -> None:
        ledger = self._ledger
        if ledger is None:
            try:
                from .reliability_ledger import global_ledger
                ledger = global_ledger()
            except Exception:
                return
        try:
            healths = ledger.all_tool_health() or []
        except Exception:
            return
        for h in healths:
            tool = h.get("tool") or ""
            if not tool:
                continue
            try:
                if not ledger.is_currently_flaky(
                        tool, threshold=0.5):
                    continue
            except Exception:
                continue
            try:
                recent = ledger.recent_errors(
                    tool, window_sec=UNHEALTHY_WINDOW_SEC, limit=50)
            except Exception:
                recent = []
            if len(recent) < UNHEALTHY_ERROR_THRESHOLD:
                continue
            with self._lock:
                if not self._cooldown_ok(
                        NudgeKind.UNHEALTHY_TOOL, tool, now):
                    continue
            # Most common error class for context.
            try:
                counts = ledger.error_class_counts(
                    tool, window_sec=UNHEALTHY_WINDOW_SEC)
            except Exception:
                counts = {}
            top_err = (max(counts.items(), key=lambda kv: kv[1])[0]
                       if counts else "other")
            self._emit(Nudge(
                kind=NudgeKind.UNHEALTHY_TOOL,
                body=(f"`{tool}` has failed {len(recent)} times in the "
                      f"last 30 min (mostly: {top_err}). Want me to "
                      "skip it for this session?"),
                severity="normal",
                suggested_action=f"pause_tool:{tool}",
            ))
            self._mark_fired(NudgeKind.UNHEALTHY_TOOL, tool, now)

    # ---- helpers ------------------------------------------------------

    def _read_user_idle_sec(self) -> Optional[float]:
        try:
            from .interruption_gate import global_gate, SignalKind
            gate = self._gate or global_gate()
            val = gate.get_signal(SignalKind.USER_IDLE_SEC)
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    def _emit(self, nudge: Nudge) -> None:
        try:
            self._handler(nudge)
        except Exception:
            pass

    def _cooldown_ok(self, kind: NudgeKind, sub_key: str,
                     now: float) -> bool:
        cooldown = PER_KIND_COOLDOWN_SEC.get(kind, 600.0)
        with self._lock:
            last = self._last_fire.get((kind, sub_key), 0.0)
            return (now - last) >= cooldown

    def _mark_fired(self, kind: NudgeKind, sub_key: str,
                    now: float) -> None:
        with self._lock:
            self._last_fire[(kind, sub_key)] = now

    def reset(self) -> None:
        with self._lock:
            self._last_fire.clear()
            self._idle_session_handled = False


def register_with_sentinel(*, handler: NudgeHandler,
                           sentinel: Optional[Any] = None,
                           interval_sec: float
                           = DEFAULT_TICK_INTERVAL_SEC
                           ) -> ProactiveNudges:
    """Build the nudge engine, register its tick with the Sentinel."""
    nudges = ProactiveNudges(handler=handler)
    from .sentinel import global_sentinel
    s = sentinel or global_sentinel()
    s.register("proactive_nudges", nudges.tick,
               interval_sec=interval_sec, max_run_ms=500)
    return nudges

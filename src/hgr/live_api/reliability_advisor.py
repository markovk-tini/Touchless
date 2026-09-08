"""Autonomous reliability advisor — weekly health digest.

Phase-5. The `proactive_nudges.UNHEALTHY_TOOL` signal already fires
when a specific tool has 5+ errors in 30 min. That catches acute
spikes. This module catches CHRONIC issues: a tool that's been at
40% error rate all week, or a connector with a clear alternate the
user could switch to.

How it works:
  * Sentinel-tickable advisor runs every ~6 hours (configurable).
  * Pulls per-tool health from `reliability_ledger`.
  * For each chronically-unhealthy tool, looks up a known alternate
    (gmail_send → ms_mail_send, faster_whisper → whisper_cpp, etc.).
  * Emits a "ReliabilityAdvice" the manager surfaces as a chat nudge.

Honors incognito + InterruptionGate (LOW severity). Hard 7-day
cooldown per-tool so a single chronic issue doesn't surface
every 6 hours.

Author: Konstantin Markov
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# Default thresholds (overrideable per-instance).
DEFAULT_CHRONIC_ERROR_RATE = 0.30   # 30% errors over the window
DEFAULT_CHRONIC_WINDOW_SEC = 7 * 24 * 3600.0   # 1 week
DEFAULT_MIN_SAMPLES = 20            # need at least 20 calls
DEFAULT_TICK_INTERVAL_SEC = 6 * 3600.0          # 6 hours
PER_TOOL_COOLDOWN_SEC = 7 * 24 * 3600.0         # 1 week


# Known alternates per tool — when the chronic-unhealthy tool has
# a same-domain alternate, suggest switching. Keys are the failing
# tool; values are (alternate_tool, friendly_name).
_KNOWN_ALTERNATES: Dict[str, tuple] = {
    "gmail_send": ("ms_mail_send", "Outlook"),
    "gmail_list": ("ms_mail_list", "Outlook inbox"),
    "ms_mail_send": ("gmail_send", "Gmail"),
    "ms_mail_list": ("gmail_list", "Gmail inbox"),
    "outlook_compose": ("gmail_send", "Gmail"),
    "outlook_send": ("gmail_send", "Gmail"),
    "drive_upload": ("onedrive_upload", "OneDrive"),
    "onedrive_upload": ("drive_upload", "Google Drive"),
    "gdocs_create": ("word_create", "Word"),
    "sheets_create": ("excel_create", "Excel"),
    "slides_create": ("powerpoint_create", "PowerPoint"),
    # STT backends (when one is flaky, suggest the other).
    "faster_whisper": ("whisper_cpp", "whisper.cpp"),
    "whisper_cpp": ("faster_whisper", "faster-whisper"),
}


@dataclass
class ReliabilityAdvice:
    """One actionable health insight."""
    tool: str
    error_rate: float
    samples: int
    headline: str
    suggested_alternate: str = ""
    alternate_friendly_name: str = ""
    suggested_action: str = ""
    ts: float = field(default_factory=time.time)


AdviceHandler = Callable[[ReliabilityAdvice], None]


class ReliabilityAdvisor:
    """Periodic tick-based advisor. Stateful only for per-tool
    cooldown bookkeeping."""

    def __init__(self, *,
                 handler: AdviceHandler,
                 ledger: Optional[Any] = None,
                 gate: Optional[Any] = None,
                 chronic_error_rate: float = DEFAULT_CHRONIC_ERROR_RATE,
                 chronic_window_sec: float = DEFAULT_CHRONIC_WINDOW_SEC,
                 min_samples: int = DEFAULT_MIN_SAMPLES) -> None:
        self._handler = handler
        self._ledger = ledger
        self._gate = gate
        self._chronic_rate = float(chronic_error_rate)
        self._window = float(chronic_window_sec)
        self._min_samples = int(min_samples)
        self._last_fired: Dict[str, float] = {}

    def tick(self) -> None:
        # Incognito: silent.
        try:
            from .incognito import is_incognito
            if is_incognito():
                return
        except Exception:
            pass
        # Gate: only fire when we can interrupt at LOW severity.
        try:
            from .interruption_gate import (global_gate,
                                             InterruptSeverity)
            gate = self._gate or global_gate()
            decision = gate.can_interrupt(InterruptSeverity.LOW)
            if not decision.allow:
                return
        except Exception:
            pass
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
        now = time.time()
        for h in healths:
            try:
                self._maybe_advise(ledger, h, now)
            except Exception:
                continue

    def _maybe_advise(self, ledger: Any, health_row: Dict[str, Any],
                      now: float) -> None:
        tool = str(health_row.get("tool") or "")
        if not tool:
            return
        samples = int(health_row.get("samples") or 0)
        if samples < self._min_samples:
            return
        # Decay-weighted error rate over the window.
        try:
            rate = float(ledger.error_rate_recent(
                tool, window_sec=self._window))
        except Exception:
            return
        if rate < self._chronic_rate:
            return
        # Cooldown.
        last = self._last_fired.get(tool, 0.0)
        if (now - last) < PER_TOOL_COOLDOWN_SEC:
            return
        # Build advice.
        alt = _KNOWN_ALTERNATES.get(tool)
        if alt is not None:
            alternate_tool, alternate_name = alt
            headline = (
                f"`{tool}` has been failing {rate * 100:.0f}% of the "
                f"time this week ({samples} samples). Want me to "
                f"prefer {alternate_name} for the rest of the week?")
            action = f"prefer_alternate:{alternate_tool}"
        else:
            alternate_tool = ""
            alternate_name = ""
            headline = (
                f"`{tool}` has been failing {rate * 100:.0f}% of the "
                f"time this week ({samples} samples). Worth a look?")
            action = f"investigate_tool:{tool}"
        advice = ReliabilityAdvice(
            tool=tool, error_rate=rate, samples=samples,
            headline=headline,
            suggested_alternate=alternate_tool,
            alternate_friendly_name=alternate_name,
            suggested_action=action,
        )
        try:
            self._handler(advice)
        except Exception:
            pass
        self._last_fired[tool] = now

    def reset(self) -> None:
        self._last_fired.clear()


def register_with_sentinel(*, handler: AdviceHandler,
                           sentinel: Optional[Any] = None,
                           interval_sec: float = DEFAULT_TICK_INTERVAL_SEC,
                           ) -> ReliabilityAdvisor:
    """Build the advisor, register its tick with the Sentinel."""
    advisor = ReliabilityAdvisor(handler=handler)
    from .sentinel import global_sentinel
    s = sentinel or global_sentinel()
    s.register("reliability_advisor", advisor.tick,
               interval_sec=interval_sec, max_run_ms=500)
    return advisor

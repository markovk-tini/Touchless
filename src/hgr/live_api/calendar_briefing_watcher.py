"""Sentinel-registered watcher that surfaces upcoming-meeting briefings.

Phase-3 wiring. Glues the three Phase-3 modules that needed each other:

  - `calendar_connector.calendar_list_events` (read upcoming events)
  - `calendar_briefing.BriefingScheduler` + `build_briefing` (build the
    user-facing prose for an event)
  - `interruption_gate` (decide whether NOW is OK to interrupt)

Runs every ~60 seconds via the Sentinel. Inexpensive per tick:
single calendar list + cooldown dict lookup. The actual notifier
(voice / earcon / chat-panel chip) is supplied by the caller as a
callback so the watcher doesn't reach into UI code from the
substrate layer.

Author: Konstantin Markov
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

_log = logging.getLogger(__name__)

from .calendar_briefing import (BriefingScheduler, CalendarEvent,
                                 build_briefing)
# InterruptSeverity is consumed inside `tick()` via a lazy import to
# keep import-time wiring cheap and avoid circular deps.


# How often the watcher polls the calendar. 60 sec is the sweet
# spot — events that start in <5 min get one re-check before they
# fire, and we don't burn the Google Calendar quota.
DEFAULT_INTERVAL_SEC = 60.0
DEFAULT_LEAD_MINUTES = 7
# Events closer than this get bumped to HIGH severity so they pass
# the "user on a call" defer (gate passes HIGH as a muted earcon).
_IMMINENT_THRESHOLD_SEC = 120.0


BriefingNotifier = Callable[[Any, Any], None]
"""Callback signature: notifier(briefing, interrupt_decision).
The notifier decides voice vs earcon vs chat-chip based on
`interrupt_decision.suggested_channel`."""


def _parse_event_ts(raw: str) -> Optional[float]:
    """Parse Google Calendar's RFC3339 'start' string to epoch seconds.
    Handles both full-datetime ('2026-06-04T14:30:00-07:00') and
    all-day ('2026-06-04') formats. Returns None on bad input."""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        if "T" in s:
            # Python's fromisoformat accepts RFC3339 with offset since
            # 3.11; for older Pythons strip a trailing 'Z'.
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s).timestamp()
        # All-day event — start of day in local tz.
        d = datetime.fromisoformat(s)
        return d.replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def _normalize_events(raw_events: List[dict]) -> List[CalendarEvent]:
    out: List[CalendarEvent] = []
    for e in raw_events or []:
        if not isinstance(e, dict):
            continue
        start_ts = _parse_event_ts(str(e.get("start") or ""))
        if start_ts is None:
            continue
        out.append(CalendarEvent(
            id=str(e.get("id") or e.get("summary") or "evt"),
            title=str(e.get("summary") or "(untitled)"),
            start_ts=start_ts,
            end_ts=_parse_event_ts(str(e.get("end") or "")) or 0.0,
            location=str(e.get("location") or ""),
            description=str(e.get("description") or ""),
            attendees=list(e.get("attendees") or []),
            organizer=str(e.get("organizer") or ""),
            is_remote=bool(e.get("hangoutLink") or e.get("meeting_url")),
            meeting_url=str(e.get("hangoutLink")
                            or e.get("meeting_url") or ""),
        ))
    return out


class CalendarBriefingWatcher:
    """Sentinel-tickable watcher that surfaces meeting briefings via a
    user-supplied notifier callback. Stateful only for per-event
    cooldowns — everything else is computed per-tick."""

    def __init__(self, *,
                 registry: Any,
                 notifier: BriefingNotifier,
                 lead_minutes: int = DEFAULT_LEAD_MINUTES,
                 calendar_tool: str = "calendar_list_events",
                 max_events_per_tick: int = 5) -> None:
        self._registry = registry
        self._notifier = notifier
        self._lead = int(lead_minutes)
        self._tool = calendar_tool
        self._max = int(max_events_per_tick)
        self._scheduler = BriefingScheduler()

    def tick(self) -> None:
        """One pass: fetch upcoming events, build briefings for any
        that are eligible + permitted by the gate, notify."""
        if self._registry is None:
            return
        # 1. Pull upcoming events from the calendar connector.
        try:
            result = self._registry.call(
                self._tool, {"max": max(10, self._max * 2)})
        except Exception:
            return
        if not isinstance(result, dict) or result.get("status") != "ok":
            return
        events = _normalize_events(result.get("events") or [])
        if not events:
            return

        # 2. Filter to events starting within the lead window.
        upcoming = self._scheduler.upcoming_within(
            events, lead_minutes=self._lead)
        if not upcoming:
            return

        # 3. For each eligible event, check the gate + build + notify.
        from .interruption_gate import (global_gate,
                                         InterruptSeverity as _Sev)
        gate = global_gate()
        now_ts = time.time()
        for event in upcoming[:self._max]:
            if not self._scheduler.is_eligible(event.id):
                continue
            # Briefings are NORMAL severity by default; imminent events
            # (<=2 min away) bump to HIGH so the gate's call-in-progress
            # defer doesn't swallow the meeting-starts-NOW nudge.
            seconds_until = event.start_ts - now_ts
            sev = (_Sev.HIGH if seconds_until <= _IMMINENT_THRESHOLD_SEC
                   else _Sev.NORMAL)
            decision = gate.can_interrupt(sev)
            if not decision.allow:
                try:
                    _log.debug(
                        "briefing suppressed event=%s sev=%s reason=%s "
                        "delay_until=%s seconds_until=%.0f",
                        event.id, sev.value, decision.reason,
                        decision.delay_until, seconds_until,
                    )
                except Exception:
                    pass
                continue
            try:
                briefing = build_briefing(event)
            except Exception:
                continue
            try:
                self._notifier(briefing, decision)
                self._scheduler.mark_briefed(event.id)
            except Exception:
                # Notifier failure: don't crash the tick. Don't
                # mark briefed so the next tick retries.
                continue


def register_with_sentinel(*, registry: Any,
                           notifier: BriefingNotifier,
                           sentinel: Optional[Any] = None,
                           interval_sec: float = DEFAULT_INTERVAL_SEC,
                           lead_minutes: int = DEFAULT_LEAD_MINUTES
                           ) -> CalendarBriefingWatcher:
    """Wire the watcher into the global Sentinel. Returns the
    watcher instance so the caller can keep a reference (e.g., to
    re-register with a different notifier later)."""
    watcher = CalendarBriefingWatcher(
        registry=registry, notifier=notifier,
        lead_minutes=lead_minutes,
    )
    from .sentinel import global_sentinel
    s = sentinel or global_sentinel()
    s.register("calendar_briefing", watcher.tick,
               interval_sec=interval_sec, max_run_ms=2000)
    return watcher

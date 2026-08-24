"""Anticipatory action engine — Iris predicts what's next.

Phase-6 cognition. Proactive nudges (Phase 3) fire on REACTIVE
signals: long-idle, cost-cap, unhealthy-tool. The anticipation
engine fires on PREDICTIVE signals: cross-referencing calendar +
recent active windows + recurring patterns + memory + standing
orders to surface "you probably want X next."

Heuristic catalogue (initial set):

  * **Meeting imminent** — calendar event in <15 min; the
    briefing watcher already covers the 5-min announce; this
    catches the longer window with prep suggestions.

  * **Recurring morning routine** — same (app, time-bucket)
    opened ≥3 times in the past week → suggest auto-opening.

  * **Stale standing order** — an order whose trigger fired N
    hours ago but the user never acted on the result.

  * **Inbox triage opportunity** — N+ unread emails accrued
    since the last `gmail_list` call AND the user hasn't been
    in their inbox app recently.

  * **Document needs follow-up** — a doc the user authored was
    last edited >24h ago but is referenced in a recent
    conversation turn.

The engine doesn't TAKE actions; it produces `Anticipation`
records the manager surfaces through the same pending-notes path
as proactive nudges. The user accepts/dismisses; nothing fires
automatically.

Per-pattern cooldowns so we don't nag. Honors incognito +
InterruptionGate.

Author: Konstantin Markov
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AnticipationKind(str, Enum):
    MEETING_IMMINENT = "meeting_imminent"
    RECURRING_ROUTINE = "recurring_routine"
    STALE_STANDING_ORDER = "stale_standing_order"
    INBOX_BACKLOG = "inbox_backlog"
    DOC_FOLLOWUP = "doc_followup"


@dataclass
class Anticipation:
    """One predicted action / opportunity surfaced to the user."""
    kind: AnticipationKind
    headline: str
    suggested_action: str = ""
    severity: str = "low"               # 'low' | 'normal' | 'high'
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


AnticipationHandler = Callable[[Anticipation], None]


# Per-kind cooldowns (seconds). Anticipation should feel sharp,
# not nagging — once-per-window per pattern.
_PER_KIND_COOLDOWN_SEC: Dict[AnticipationKind, float] = {
    AnticipationKind.MEETING_IMMINENT:     20 * 60.0,
    AnticipationKind.RECURRING_ROUTINE:    24 * 3600.0,
    AnticipationKind.STALE_STANDING_ORDER: 12 * 3600.0,
    AnticipationKind.INBOX_BACKLOG:        4 * 3600.0,
    AnticipationKind.DOC_FOLLOWUP:         8 * 3600.0,
}


# How long before a meeting should "imminent" fire. Larger than
# the BriefingScheduler window (5 min) so we get a prep nudge
# before the formal briefing.
MEETING_IMMINENT_LEAD_MIN = 15
# Cap on how many imminent meetings to surface per tick — we
# don't want to dump the whole day's calendar at once.
MAX_IMMINENT_PER_TICK = 1


class AnticipationEngine:
    """Sentinel-tickable predictor. Stateful only for per-pattern
    cooldown bookkeeping + a small day-bucket activity history."""

    def __init__(self, *,
                 handler: AnticipationHandler,
                 registry: Optional[Any] = None,
                 gate: Optional[Any] = None) -> None:
        self._handler = handler
        self._registry = registry
        self._gate = gate
        # Cooldown bookkeeping keyed by (kind, sub_key).
        self._last_fire: Dict[tuple, float] = {}
        # Bucketed activity history: (window_title, hour_bucket) ->
        # set of date strings on which we observed it. Used by the
        # routine-detection pass.
        self._activity_log: Dict[tuple, set] = {}

    def set_registry(self, registry: Any) -> None:
        self._registry = registry

    # ---- the tick -----------------------------------------------------

    def tick(self) -> None:
        try:
            from .incognito import is_incognito
            if is_incognito():
                return
        except Exception:
            pass
        # Respect the gate — anticipation is LOW severity by default.
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
        self._record_current_activity(now)
        self._check_meeting_imminent(now)
        self._check_recurring_routine(now)
        self._check_stale_standing_orders(now)

    # ---- per-check ----------------------------------------------------

    def _check_meeting_imminent(self, now: float) -> None:
        if self._registry is None:
            return
        try:
            result = self._registry.call(
                "calendar_list_events", {"max": 5})
        except Exception:
            return
        if not isinstance(result, dict) or result.get("status") != "ok":
            return
        events = result.get("events") or []
        # Find the soonest event in the imminent window.
        soon = []
        for e in events:
            if not isinstance(e, dict):
                continue
            start_ts = _parse_event_ts(str(e.get("start") or ""))
            if start_ts is None:
                continue
            mins_away = (start_ts - now) / 60.0
            # 5 min ≤ mins_away ≤ 15 min → "imminent" window.
            # < 5 min is the briefing watcher's job.
            if 5.0 < mins_away <= MEETING_IMMINENT_LEAD_MIN:
                soon.append((mins_away, e))
        soon.sort(key=lambda kv: kv[0])
        for mins_away, event in soon[:MAX_IMMINENT_PER_TICK]:
            event_id = str(event.get("id") or "")
            sub_key = f"meeting:{event_id}"
            if not self._cooldown_ok(
                    AnticipationKind.MEETING_IMMINENT,
                    sub_key, now):
                continue
            title = str(event.get("summary") or "your next meeting")
            attendees = event.get("attendees") or []
            with_str = ""
            if isinstance(attendees, list) and attendees:
                names = [_friendly_name(a) for a in attendees[:3]]
                names = [n for n in names if n]
                if names:
                    with_str = f" with {', '.join(names)}"
            self._emit(Anticipation(
                kind=AnticipationKind.MEETING_IMMINENT,
                headline=(f"Your {title} starts in "
                          f"{int(round(mins_away))} min{with_str}. "
                          "Want me to mute Slack and pull up the doc?"),
                suggested_action="prep_for_meeting",
                severity="normal",
                payload={"event_id": event_id, "title": title,
                         "mins_away": mins_away},
            ))
            self._mark_fired(AnticipationKind.MEETING_IMMINENT,
                             sub_key, now)

    def _check_recurring_routine(self, now: float) -> None:
        """Detect (window, hour-bucket) patterns the user has done
        on ≥3 distinct days in the past week."""
        focused = _current_window_title()
        if not focused:
            return
        bucket = _hour_bucket(now)
        key = (_normalize_window(focused), bucket)
        seen_days = self._activity_log.get(key, set())
        # 3+ DISTINCT days in the past 7 → recurring routine.
        if len(seen_days) < 3:
            return
        today = datetime.fromtimestamp(now).date().isoformat()
        # Only fire when today is one of the matching days AND
        # we haven't already fired this routine today.
        if today not in seen_days:
            return
        sub_key = f"routine:{key[0]}:{bucket}"
        if not self._cooldown_ok(
                AnticipationKind.RECURRING_ROUTINE, sub_key, now):
            return
        self._emit(Anticipation(
            kind=AnticipationKind.RECURRING_ROUTINE,
            headline=(f"You usually open {focused[:60]} around "
                      f"this time. Save it as a morning routine?"),
            suggested_action=f"propose_routine:{focused[:80]}",
            severity="low",
            payload={"window": focused, "hour_bucket": bucket,
                     "days_seen": len(seen_days)},
        ))
        self._mark_fired(
            AnticipationKind.RECURRING_ROUTINE, sub_key, now)

    def _check_stale_standing_orders(self, now: float) -> None:
        """An order that fired ≥12h ago but is still ACTIVE +
        recurring (not terminal). Surface so the user can decide
        whether to cancel or refresh it."""
        try:
            from .standing_orders import global_store, OrderState
        except Exception:
            return
        try:
            active = global_store().all_active()
        except Exception:
            return
        for order in active:
            if order.state != OrderState.ACTIVE.value:
                continue
            if order.fire_count == 0:
                continue
            age = now - order.last_fired_at
            if age < 12 * 3600.0:
                continue
            sub_key = f"order:{order.id}"
            if not self._cooldown_ok(
                    AnticipationKind.STALE_STANDING_ORDER,
                    sub_key, now):
                continue
            hours_ago = int(age // 3600)
            self._emit(Anticipation(
                kind=AnticipationKind.STALE_STANDING_ORDER,
                headline=(
                    f"📌 Heads up: \"{order.short_label()}\" fired "
                    f"{hours_ago}h ago and you didn't act on it. "
                    "Still want it watching?"),
                suggested_action=f"refresh_or_cancel:{order.id}",
                severity="low",
                payload={"order_id": order.id,
                         "hours_since_fire": hours_ago},
            ))
            self._mark_fired(
                AnticipationKind.STALE_STANDING_ORDER, sub_key, now)

    # ---- helpers ------------------------------------------------------

    def _record_current_activity(self, now: float) -> None:
        """Daily 1-row append to the activity log so the routine
        detector has data to draw on."""
        focused = _current_window_title()
        if not focused:
            return
        bucket = _hour_bucket(now)
        key = (_normalize_window(focused), bucket)
        today = datetime.fromtimestamp(now).date().isoformat()
        seen = self._activity_log.setdefault(key, set())
        seen.add(today)
        # Prune anything older than 7 days.
        cutoff = (datetime.fromtimestamp(now)
                  .date() - _timedelta(days=7)).isoformat()
        # set objects don't support ordering; just rebuild.
        keep = {d for d in seen if d > cutoff}
        self._activity_log[key] = keep

    def _emit(self, anticipation: Anticipation) -> None:
        try:
            self._handler(anticipation)
        except Exception:
            pass

    def _cooldown_ok(self, kind: AnticipationKind,
                     sub_key: str, now: float) -> bool:
        cooldown = _PER_KIND_COOLDOWN_SEC.get(kind, 3600.0)
        last = self._last_fire.get((kind, sub_key), 0.0)
        return (now - last) >= cooldown

    def _mark_fired(self, kind: AnticipationKind,
                    sub_key: str, now: float) -> None:
        self._last_fire[(kind, sub_key)] = now

    def reset(self) -> None:
        self._last_fire.clear()
        self._activity_log.clear()


# ---- module helpers ---------------------------------------------------

def _parse_event_ts(raw: str) -> Optional[float]:
    """Parse RFC3339-ish calendar 'start' string to epoch seconds."""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        from datetime import datetime, timezone
        if "T" in s:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s).timestamp()
        return datetime.fromisoformat(s).replace(
            tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def _friendly_name(attendee: Any) -> str:
    """Best-effort 'Dani' from various calendar attendee shapes."""
    if isinstance(attendee, str):
        s = attendee.strip()
        if "<" in s:
            return s.split("<", 1)[0].strip().strip('"')
        if "@" in s:
            return s.split("@", 1)[0].title()
        return s
    if isinstance(attendee, dict):
        for k in ("displayName", "name", "email"):
            v = attendee.get(k)
            if v:
                return str(v).split("@", 1)[0].title()
    return ""


def _current_window_title() -> str:
    """Forward to the existing focus_window helper. Empty string
    on non-Windows / probe failure."""
    try:
        from .repo_focus_watcher import _foreground_window_title
        return _foreground_window_title()
    except Exception:
        return ""


def _normalize_window(title: str) -> str:
    """Strip noisy suffixes from a window title so two slightly-
    different titles (different document names in same IDE) hash
    to the same routine bucket."""
    if not title:
        return ""
    parts = title.split(" - ")
    if len(parts) >= 2:
        # Last part is usually the app name ('Visual Studio Code').
        return parts[-1].strip()
    return title.strip()


def _hour_bucket(now: float) -> int:
    """Round to a 1-hour bucket so 'I open VS Code around 9am' is a
    pattern even when 'around' is 8:50 vs 9:15."""
    return datetime.fromtimestamp(now).hour


def _timedelta(*, days: int):
    from datetime import timedelta
    return timedelta(days=days)


def register_with_sentinel(*, handler: AnticipationHandler,
                           sentinel: Optional[Any] = None,
                           interval_sec: float = 120.0,
                           ):
    """Build the engine, register tick with the Sentinel."""
    engine = AnticipationEngine(handler=handler)
    from .sentinel import global_sentinel
    s = sentinel or global_sentinel()
    s.register("anticipation", engine.tick,
               interval_sec=interval_sec, max_run_ms=1000)
    return engine

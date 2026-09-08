"""SuggestionEngine — ranked next-action recommendations for Iris.

Composes signals from several existing subsystems into a deduped,
score-ranked list of "what the user might want to do next" suggestions:

  1. **Time-of-day tool patterns** (``temporal_patterns.find_time_of_day_patterns``)
     — "you usually open Notion Monday 9am"
  2. **Repeated query habits** (``patterns.find_repeated_patterns``)
     — "you've asked variations of this 5 times this week"
  3. **Tool sequences** (``patterns.find_tool_sequences``)
     — "open_app usually follows gmail_list"
  4. **Recently-touched projects** (``world_state.list_projects``)
     — "Touchless website was touched 2h ago — keep going?"
  5. **Contacts with typical_hours matching now** (``contacts``)
     — "Vesko gets pinged around 3pm — follow up?"
  6. **Recent in-progress episodes** (``MemoryStore.list_episodic``)
     — "you mentioned 'finalize the README' but never closed it"

Scoring formula (each component 0-1):

    score = temporal_match * 0.4 + recency * 0.3 + frequency * 0.3

Suggestions below ``_SCORE_FLOOR`` (0.4) are dropped. Results are
deduped by ``(kind, action_hint)`` keeping the highest score, then
sorted desc by score, then capped at ``max_suggestions``.

Anti-noise rules:

  * Anything touched in the last 60 minutes is suppressed
    (recent-action dedupe via ``tool_call_log``).
  * Each suggestion carries a human-readable ``reason`` string so the
    Iris UI can explain *why* it surfaced.

This is a pure module: no LLM call, no network. All inputs are
read-only against existing on-disk stores; failures in any single
signal source degrade silently (we just contribute fewer suggestions).

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Kinds the UI can render with distinct icons / styling.
KIND_OPEN_APP = "open_app"
KIND_CONTINUE_TASK = "continue_task"
KIND_FOLLOW_UP_PERSON = "follow_up_person"
KIND_PROJECT_FOCUS = "project_focus"
KIND_IDEA = "idea"
KIND_REMINDER = "reminder"

_ALL_KINDS = frozenset({
    KIND_OPEN_APP, KIND_CONTINUE_TASK, KIND_FOLLOW_UP_PERSON,
    KIND_PROJECT_FOCUS, KIND_IDEA, KIND_REMINDER,
})

# Score floor — suggestions below this are dropped from the output.
_SCORE_FLOOR = 0.4

# Anti-noise: anything touched / fired within this window is suppressed
# from "open_app" / "continue_task" suggestions. Avoids the obvious
# "you just did that" annoyance.
_RECENT_ACTION_WINDOW_S = 60 * 60  # 1 hour

# Episodic / temporal lookups are bounded so a 10k-row DB doesn't burn
# the suggest_now latency budget.
_EPISODE_SCAN_LIMIT = 500
_PATTERN_DAYS = 7

# Reminder lookahead — patterns whose next_predicted_iso falls inside
# this window become "reminder" suggestions.
_REMINDER_LOOKAHEAD_S = 2 * 3600  # 2 hours

# Tool ids → app names mapping for the most common "open" tools.
# Used to convert a tool_id from a time-of-day pattern into a sensible
# "open <app>" suggestion. Unknown tool ids fall back to "continue_task"
# with the raw tool id as the action hint.
_TOOL_TO_APP_LABEL: Dict[str, str] = {
    "open_app": "your usual app",
    "spotify_play": "Spotify",
    "ms_calendar_list": "your calendar",
    "ms_calendar_create": "your calendar",
    "gmail_list": "Gmail",
    "gmail_send": "Gmail",
    "notion_search": "Notion",
    "notion_create": "Notion",
}

# Day-of-week labels used in reasons (Monday-first to match the rest
# of the codebase). Index = datetime.weekday().
_DOW_LABELS = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _hour_label(hour: int) -> str:
    """Human-friendly hour label ('9am' / '3pm' / 'midnight')."""
    try:
        h = int(hour) % 24
    except (TypeError, ValueError):
        return "around now"
    if h == 0:
        return "midnight"
    if h == 12:
        return "noon"
    if h < 12:
        return f"{h}am"
    return f"{h - 12}pm"


def _safe_ts(iso_or_ts: Any) -> float:
    """Convert either an epoch float or an ISO 8601 string to epoch."""
    if iso_or_ts is None:
        return 0.0
    if isinstance(iso_or_ts, (int, float)):
        try:
            return float(iso_or_ts)
        except (TypeError, ValueError):
            return 0.0
    if isinstance(iso_or_ts, str):
        s = iso_or_ts.strip()
        if not s:
            return 0.0
        # Try ISO 8601 — both the "Z" UTC form (world_state) and the
        # local-time form (temporal_patterns).
        try:
            # Trim trailing 'Z' which fromisoformat refuses on older
            # Pythons (<3.11). Treat as naive — we only use the diff,
            # not the absolute value, so DST is fine.
            return datetime.fromisoformat(s.rstrip("Zz")).timestamp()
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _recency_score(ts: float, now: float,
                   half_life_days: float = 3.0) -> float:
    """Fresher events → higher score. Linear ramp from 1.0 at ts==now
    down to 0.0 at ``2 * half_life_days`` days ago. Anything older
    contributes nothing."""
    if not ts:
        return 0.0
    age_days = max(0.0, (now - ts) / 86400.0)
    full_window = 2.0 * max(half_life_days, 0.1)
    if age_days >= full_window:
        return 0.0
    return max(0.0, 1.0 - (age_days / full_window))


def _frequency_score(count: int, saturates_at: int = 10) -> float:
    """Diminishing returns: 1 fire = ~0.1, 10 = 1.0, 50 = 1.0 (capped)."""
    if not count or count <= 0:
        return 0.0
    return min(1.0, count / float(max(saturates_at, 1)))


def _temporal_match(pattern_hour: Optional[int],
                    pattern_dow_label: Optional[str],
                    current_hour: int,
                    current_dow_label: str) -> float:
    """How well does ``now`` match the pattern's typical time-of-day?

    Returns 1.0 for an exact match (same hour ±0), tapers to 0.0 at
    ±6 hours. Day-of-week boosts the match by 0.2 when the labels agree.
    """
    if pattern_hour is None:
        return 0.3  # baseline — no time signal at all
    try:
        ph = int(pattern_hour) % 24
    except (TypeError, ValueError):
        return 0.3
    delta = min(abs(ph - current_hour), 24 - abs(ph - current_hour))
    hour_score = max(0.0, 1.0 - (delta / 6.0))
    if (pattern_dow_label
            and pattern_dow_label not in ("any", "")
            and pattern_dow_label.lower() == current_dow_label.lower()):
        hour_score = min(1.0, hour_score + 0.2)
    return hour_score


def _normalize_action_hint(hint: str) -> str:
    """Lowercase + strip whitespace for stable dedupe key."""
    return (hint or "").strip().lower()


def _path_for_world_state() -> Optional[Any]:
    """Best-effort: instantiate the WorldState singleton. Returns None
    on any failure — caller treats it as 'no project signals'."""
    try:
        from hgr.live_api.cortex.world_state import get_world
        return get_world()
    except Exception:
        return None


class SuggestionEngine:
    """Rule-based next-action recommender.

    Parameters:
        memory_manager: A ``MemoryManager`` instance (gives access to
            ``_store`` for episodic / facts and ``_embedder`` for the
            repeated-pattern clustering).
        tool_call_log_path: Optional ``Path`` to ``tool_call_log.db``
            (used for time-of-day patterns + recent-action dedup).
            Defaults to ``cortex.tool_call_log.default_log_path()``.
        contacts: Optional iterable of contact dicts (from
            ``memory.contacts.gather_contacts`` + ``enrich_*``). When
            omitted, person follow-ups are silently skipped.
        world_state: Optional ``WorldState`` instance. Defaults to the
            module singleton. Pass None to skip project-momentum.
        logger: Optional logger object with ``.event(name, **kwargs)``
            and ``.exception(name, exc)`` methods. Silent when absent.
    """

    def __init__(self,
                 memory_manager: Any,
                 tool_call_log_path: Optional[Path] = None,
                 contacts: Optional[Iterable[Dict[str, Any]]] = None,
                 world_state: Any = None,
                 logger: Any = None) -> None:
        self._memory = memory_manager
        if tool_call_log_path is None:
            try:
                from hgr.live_api.cortex.tool_call_log import default_log_path
                tool_call_log_path = default_log_path()
            except Exception:
                tool_call_log_path = None
        self._tool_log_path = (
            Path(tool_call_log_path) if tool_call_log_path else None
        )
        self._contacts = list(contacts) if contacts else []
        if world_state is None:
            world_state = _path_for_world_state()
        self._world = world_state
        self._logger = logger

    # ---- public API -----------------------------------------------------

    def suggest_now(self,
                    context: Optional[Dict[str, Any]] = None,
                    max_suggestions: int = 8) -> List[Dict[str, Any]]:
        """Return a ranked list of next-action suggestions.

        Args:
            context: optional dict; recognized keys: ``now`` (epoch
                float), ``hour`` (0-23), ``weekday`` (0=Mon..6=Sun),
                ``weekday_label`` (e.g. "Monday").
            max_suggestions: hard cap on returned suggestions.

        Returns:
            List of suggestion dicts, each:
              {
                "id": str,
                "kind": str (one of KIND_*),
                "label": str (UI title),
                "action_hint": str (stable dedupe key),
                "score": float (0-1),
                "reason": str (human-readable explanation),
                "related_contacts": [{name, kind}],
                "related_projects": [{project_id, label, weight}],
                "predicted_at_iso": str | None,
                "ts_suggested": float,
                "confidence": float (0-1, internal — same as score),
              }
        """
        ctx = self._build_context(context)
        now = ctx["now"]

        # Pre-compute the recent-action set once so every signal source
        # can dedupe cheaply against "you just did that."
        recent_actions = self._recent_action_set(now)

        all_suggestions: List[Dict[str, Any]] = []

        # 1. Time-of-day patterns from tool_call_log
        try:
            all_suggestions.extend(self._from_time_patterns(ctx, recent_actions))
        except Exception as exc:
            self._log_exc("suggestion_time_patterns_failed", exc)

        # 2. Repeated query habits (cluster episodes by embedding)
        try:
            all_suggestions.extend(self._from_repeated_patterns(ctx))
        except Exception as exc:
            self._log_exc("suggestion_repeated_patterns_failed", exc)

        # 3. Tool sequences (toolA → toolB)
        try:
            all_suggestions.extend(self._from_tool_sequences(ctx, recent_actions))
        except Exception as exc:
            self._log_exc("suggestion_tool_sequences_failed", exc)

        # 4. Project momentum
        try:
            all_suggestions.extend(self._from_projects(ctx))
        except Exception as exc:
            self._log_exc("suggestion_projects_failed", exc)

        # 5. Contacts with typical_hours matching now
        try:
            all_suggestions.extend(self._from_contacts(ctx))
        except Exception as exc:
            self._log_exc("suggestion_contacts_failed", exc)

        # 6. Recent in-progress episodes (continue_task)
        try:
            all_suggestions.extend(self._from_recent_episodes(ctx))
        except Exception as exc:
            self._log_exc("suggestion_recent_episodes_failed", exc)

        # Score-gate, dedupe, cap.
        return self._finalize(all_suggestions, max_suggestions)

    # ---- context --------------------------------------------------------

    def _build_context(self,
                       override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Materialize the runtime context (now, weekday, hour) from a
        partial override + the current wall clock."""
        override = override or {}
        try:
            now = float(override.get("now") or time.time())
        except (TypeError, ValueError):
            now = time.time()
        try:
            dt = datetime.fromtimestamp(now)
        except (OSError, OverflowError, ValueError):
            dt = datetime.now()
        hour = override.get("hour")
        try:
            hour = int(hour) if hour is not None else dt.hour
        except (TypeError, ValueError):
            hour = dt.hour
        weekday = override.get("weekday")
        try:
            weekday = int(weekday) if weekday is not None else dt.weekday()
        except (TypeError, ValueError):
            weekday = dt.weekday()
        weekday_label = override.get("weekday_label")
        if not weekday_label:
            try:
                weekday_label = _DOW_LABELS[weekday]
            except (IndexError, TypeError):
                weekday_label = "Monday"
        return {
            "now": now,
            "hour": hour,
            "weekday": weekday,
            "weekday_label": str(weekday_label),
        }

    # ---- signal sources -------------------------------------------------

    def _from_time_patterns(self, ctx: Dict[str, Any],
                            recent_actions: set) -> List[Dict[str, Any]]:
        """Convert time-of-day tool patterns into suggestions."""
        if not self._tool_log_path or not self._tool_log_path.exists():
            return []
        try:
            from hgr.live_api.memory.temporal_patterns import (
                find_time_of_day_patterns,
            )
        except Exception:
            return []
        try:
            patterns = find_time_of_day_patterns(
                self._tool_log_path,
                now_ts=ctx["now"],
            )
        except Exception:
            return []
        if not patterns:
            return []
        out: List[Dict[str, Any]] = []
        now = ctx["now"]
        for p in patterns:
            tool_id = str(p.get("tool_id") or "").strip()
            if not tool_id:
                continue
            action_hint = _normalize_action_hint(tool_id)
            # Skip if the user already fired this tool recently — don't
            # nag with "open Notion" when they just opened it.
            if action_hint in recent_actions:
                continue
            pattern_hour = p.get("hour")
            pattern_dow = p.get("day_of_week")
            occurrences = int(p.get("occurrences") or 0)
            last_seen_ts = _safe_ts(p.get("last_seen_iso"))
            next_iso = p.get("next_predicted_iso") or None
            next_ts = _safe_ts(next_iso)

            temporal = _temporal_match(
                pattern_hour, pattern_dow,
                ctx["hour"], ctx["weekday_label"],
            )
            recency = _recency_score(last_seen_ts, now, half_life_days=3.0)
            frequency = _frequency_score(occurrences, saturates_at=10)
            score = (temporal * 0.4) + (recency * 0.3) + (frequency * 0.3)

            # Promote to "reminder" if the next predicted fire is within
            # the lookahead window (and clearly in the future).
            kind = KIND_OPEN_APP if tool_id in _TOOL_TO_APP_LABEL else KIND_CONTINUE_TASK
            if next_ts and 0 < (next_ts - now) <= _REMINDER_LOOKAHEAD_S:
                kind = KIND_REMINDER

            app_label = _TOOL_TO_APP_LABEL.get(tool_id, tool_id.replace("_", " "))
            if kind == KIND_REMINDER:
                label = f"Reminder: {app_label} (in {self._minutes_until(next_ts, now)} min)"
            else:
                label = f"Open {app_label}"

            reason_bits = []
            dow_label_clean = (pattern_dow or "").strip()
            if dow_label_clean and dow_label_clean != "any":
                reason_bits.append(f"You usually do this {dow_label_clean} {_hour_label(pattern_hour or ctx['hour'])}")
            elif pattern_hour is not None:
                reason_bits.append(f"You usually do this around {_hour_label(pattern_hour)}")
            if occurrences:
                reason_bits.append(f"{occurrences}× in the last 3 weeks")
            reason = " — ".join(reason_bits) or "Frequent time-of-day pattern"

            out.append({
                "id": _new_id(),
                "kind": kind,
                "label": label,
                "action_hint": action_hint,
                "score": round(score, 4),
                "reason": reason,
                "related_contacts": [],
                "related_projects": [],
                "predicted_at_iso": next_iso,
                "ts_suggested": now,
                "confidence": round(score, 4),
            })
        return out

    def _from_repeated_patterns(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Cluster recent episodic queries — surface repeat habits as
        ``idea`` suggestions."""
        store = getattr(self._memory, "_store", None)
        embedder = getattr(self._memory, "_embedder", None)
        if store is None or embedder is None:
            return []
        try:
            from hgr.live_api.memory.patterns import find_repeated_patterns
            result = find_repeated_patterns(store, embedder, days=_PATTERN_DAYS)
        except Exception:
            return []
        if not result or result.get("cold_start"):
            return []
        out: List[Dict[str, Any]] = []
        now = ctx["now"]
        for p in (result.get("patterns") or []):
            label_text = (p.get("label") or "").strip()
            if not label_text:
                continue
            count = int(p.get("count") or 0)
            last_seen = float(p.get("last_seen_at") or 0.0)
            action_hint = _normalize_action_hint(f"query::{label_text}")
            recency = _recency_score(last_seen, now, half_life_days=5.0)
            frequency = _frequency_score(count, saturates_at=5)
            # Habit patterns don't carry an inherent time-of-day, so we
            # give them a moderate temporal baseline (0.5) — they're
            # always-applicable ideas.
            temporal = 0.5
            score = (temporal * 0.4) + (recency * 0.3) + (frequency * 0.3)
            out.append({
                "id": _new_id(),
                "kind": KIND_IDEA,
                "label": f"Revisit: {label_text}",
                "action_hint": action_hint,
                "score": round(score, 4),
                "reason": f"You've asked variations of this {count}× this week",
                "related_contacts": [],
                "related_projects": [],
                "predicted_at_iso": None,
                "ts_suggested": now,
                "confidence": round(score, 4),
            })
        return out

    def _from_tool_sequences(self, ctx: Dict[str, Any],
                             recent_actions: set) -> List[Dict[str, Any]]:
        """Convert (toolA → toolB) sequences into continue_task suggestions
        for the *next* tool when toolA was used recently."""
        store = getattr(self._memory, "_store", None)
        if store is None:
            return []
        try:
            from hgr.live_api.memory.patterns import find_tool_sequences
            result = find_tool_sequences(store, days=_PATTERN_DAYS)
        except Exception:
            return []
        if not result:
            return []
        out: List[Dict[str, Any]] = []
        now = ctx["now"]
        for seq in (result.get("sequences") or []):
            from_tool = str(seq.get("from_tool") or "").strip()
            to_tool = str(seq.get("to_tool") or "").strip()
            if not (from_tool and to_tool):
                continue
            # Only suggest the NEXT tool when the FROM tool was used in
            # the recent window — that's when chaining is most useful.
            if _normalize_action_hint(from_tool) not in recent_actions:
                continue
            # Don't re-suggest a tool the user already ran recently.
            if _normalize_action_hint(to_tool) in recent_actions:
                continue
            count = int(seq.get("count") or 0)
            last_seen = float(seq.get("last_seen_at") or 0.0)
            recency = _recency_score(last_seen, now, half_life_days=4.0)
            frequency = _frequency_score(count, saturates_at=5)
            # Tool-sequence has high temporal match when the predecessor
            # just fired — represents an "in flow" moment.
            temporal = 0.85
            score = (temporal * 0.4) + (recency * 0.3) + (frequency * 0.3)
            label = f"Try {to_tool.replace('_', ' ')} next"
            reason = (
                f"After {from_tool.replace('_', ' ')} you usually do "
                f"{to_tool.replace('_', ' ')} ({count}× this week)"
            )
            out.append({
                "id": _new_id(),
                "kind": KIND_CONTINUE_TASK,
                "label": label,
                "action_hint": _normalize_action_hint(f"seq::{to_tool}"),
                "score": round(score, 4),
                "reason": reason,
                "related_contacts": [],
                "related_projects": [],
                "predicted_at_iso": None,
                "ts_suggested": now,
                "confidence": round(score, 4),
            })
        return out

    def _from_projects(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Surface recently-touched projects as ``project_focus``."""
        if self._world is None:
            return []
        try:
            projects = self._world.list_projects()
        except Exception:
            return []
        if not projects:
            return []
        now = ctx["now"]
        out: List[Dict[str, Any]] = []
        # list_projects() is already sorted by weight desc; take the top
        # handful before we score them, but cap so a sprawling cortex
        # world doesn't drown the rest of the signals.
        for proj in projects[:6]:
            pid = str(proj.get("id") or "")
            label = str(proj.get("label") or pid or "project")
            if not pid:
                continue
            weight = float(proj.get("weight") or 0.0)
            last_iso = proj.get("last_touched_at")
            last_ts = _safe_ts(last_iso)
            touch_count = int(proj.get("touch_count") or 0)

            # Filter for "active" — last touch within ~1 week, not zero.
            if last_ts and (now - last_ts) > (7 * 86400):
                continue

            recency = _recency_score(last_ts, now, half_life_days=2.0)
            frequency = _frequency_score(touch_count, saturates_at=20)
            # Project momentum has no inherent time-of-day; use weight
            # as a proxy for "is this project meaningful right now."
            temporal = max(0.3, min(1.0, weight))
            score = (temporal * 0.4) + (recency * 0.3) + (frequency * 0.3)

            reason_bits = [f"Last touched {self._humanize_age(now - last_ts)}"]
            if touch_count:
                reason_bits.append(f"{touch_count} interactions")
            reason = " — ".join(reason_bits)
            out.append({
                "id": _new_id(),
                "kind": KIND_PROJECT_FOCUS,
                "label": f"Focus on {label}",
                "action_hint": _normalize_action_hint(f"project::{pid}"),
                "score": round(score, 4),
                "reason": reason,
                "related_contacts": [],
                "related_projects": [
                    {"project_id": pid, "label": label, "weight": weight},
                ],
                "predicted_at_iso": None,
                "ts_suggested": now,
                "confidence": round(score, 4),
            })
        return out

    def _from_contacts(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Follow-up suggestions for contacts whose ``typical_hours`` or
        ``typical_days`` match the current time."""
        if not self._contacts:
            return []
        now = ctx["now"]
        current_hour = int(ctx["hour"])
        current_dow_label = ctx["weekday_label"]
        # The contacts enricher labels weekdays as the short 3-letter
        # form ("Mon", "Tue", ...); match against the prefix.
        current_dow_short = current_dow_label[:3]

        out: List[Dict[str, Any]] = []
        for c in self._contacts:
            try:
                name = str(c.get("name") or "").strip()
                if not name:
                    continue
                # Skip transactional senders — never suggest "follow up
                # with Marketing Newsletter".
                if c.get("kind") == "transactional":
                    continue
                typical_hours = c.get("typical_hours") or []
                typical_days = c.get("typical_days") or []
                mention_count = int(c.get("mention_count") or 0)
                last_seen_iso = c.get("last_seen") or ""
                last_ts = _safe_ts(last_seen_iso)
                # Need at least SOME contact signal — drop pure name-only
                # rows that came from gmail headers with zero context.
                if mention_count < 1 and not typical_hours:
                    continue

                # Hour match: pick the closest typical hour, or 0.3 if
                # we have no hour signal.
                if typical_hours:
                    hour_deltas = [
                        min(abs(h - current_hour),
                            24 - abs(h - current_hour))
                        for h in typical_hours
                    ]
                    hour_match = max(0.0, 1.0 - (min(hour_deltas) / 6.0))
                else:
                    hour_match = 0.3
                # Day match: bonus 0.2 if today is one of their typical
                # days; baseline otherwise.
                day_match = 0.2 if current_dow_short in typical_days else 0.0
                temporal = min(1.0, hour_match + day_match)
                recency = _recency_score(last_ts, now, half_life_days=4.0)
                frequency = _frequency_score(mention_count, saturates_at=10)
                score = (temporal * 0.4) + (recency * 0.3) + (frequency * 0.3)

                reason_bits = []
                if mention_count >= 2:
                    reason_bits.append(
                        f"{name} mentioned {mention_count}× this week"
                    )
                if typical_hours:
                    reason_bits.append(
                        f"usually around {_hour_label(typical_hours[0])}"
                    )
                if current_dow_short in typical_days:
                    reason_bits.append(f"often on {current_dow_label}")
                reason = " — ".join(reason_bits) or f"Recent contact with {name}"

                related_projects = []
                for pid in (c.get("related_projects") or [])[:2]:
                    related_projects.append({
                        "project_id": str(pid),
                        "label": str(pid),
                        "weight": 0.0,
                    })

                out.append({
                    "id": _new_id(),
                    "kind": KIND_FOLLOW_UP_PERSON,
                    "label": f"Follow up with {name}",
                    "action_hint": _normalize_action_hint(f"person::{name}"),
                    "score": round(score, 4),
                    "reason": reason,
                    "related_contacts": [
                        {"name": name, "kind": "person"},
                    ],
                    "related_projects": related_projects,
                    "predicted_at_iso": None,
                    "ts_suggested": now,
                    "confidence": round(score, 4),
                })
            except Exception:
                # One bad contact shouldn't poison the rest.
                continue
        return out

    def _from_recent_episodes(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Surface unfinished-looking recent episodes as ``continue_task``.

        Heuristic: an episode whose outcome is empty / explicitly an
        error / mentions 'TODO|later|tomorrow' in the user_text is
        probably something the user wanted to continue."""
        store = getattr(self._memory, "_store", None)
        if store is None:
            return []
        try:
            episodes = store.list_episodic(limit=_EPISODE_SCAN_LIMIT)
        except Exception:
            return []
        if not episodes:
            return []
        now = ctx["now"]
        out: List[Dict[str, Any]] = []
        unfinished_re = re.compile(
            r"\b(todo|later|tomorrow|finish|wrap up|remind me|don'?t forget)\b",
            re.IGNORECASE,
        )
        seen_texts: set = set()
        for ep in episodes[:50]:  # only consider the freshest 50
            text = (getattr(ep, "user_text", "") or "").strip()
            outcome = (getattr(ep, "outcome", "") or "").strip().lower()
            ts = float(getattr(ep, "ts", 0.0) or 0.0)
            if not text or not ts:
                continue
            # Skip very recent episodes — those are the ones the user
            # JUST handled, and we don't want to nag.
            if (now - ts) < _RECENT_ACTION_WINDOW_S:
                continue
            looks_unfinished = bool(unfinished_re.search(text))
            errored_out = ("error" in outcome
                           or "failed" in outcome
                           or outcome == "")
            if not (looks_unfinished or errored_out):
                continue
            short = text[:60].rstrip()
            short_key = short.lower()
            if short_key in seen_texts:
                continue
            seen_texts.add(short_key)
            recency = _recency_score(ts, now, half_life_days=2.0)
            # Frequency on a per-episode basis is always 1 — flatten to
            # a baseline so the score doesn't bottom out.
            frequency = 0.3
            temporal = 0.5
            score = (temporal * 0.4) + (recency * 0.3) + (frequency * 0.3)
            reason = (
                f"You started this {self._humanize_age(now - ts)} but "
                "didn't seem to finish"
            )
            out.append({
                "id": _new_id(),
                "kind": KIND_CONTINUE_TASK,
                "label": f"Continue: {short}",
                "action_hint": _normalize_action_hint(f"episode::{short_key}"),
                "score": round(score, 4),
                "reason": reason,
                "related_contacts": [],
                "related_projects": [],
                "predicted_at_iso": None,
                "ts_suggested": now,
                "confidence": round(score, 4),
            })
        return out

    # ---- dedupe + ranking ----------------------------------------------

    def _finalize(self, suggestions: List[Dict[str, Any]],
                  max_suggestions: int) -> List[Dict[str, Any]]:
        """Score-gate, dedupe by (kind, action_hint), sort, cap."""
        floor = _SCORE_FLOOR
        filtered = [s for s in suggestions if float(s.get("score", 0)) >= floor]
        # Dedupe: keep highest-scoring per (kind, action_hint).
        best_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for s in filtered:
            key = (str(s.get("kind", "")),
                   str(s.get("action_hint", "")))
            cur = best_by_key.get(key)
            if cur is None or float(s.get("score", 0)) > float(cur.get("score", 0)):
                best_by_key[key] = s
        deduped = list(best_by_key.values())
        deduped.sort(key=lambda s: float(s.get("score", 0)), reverse=True)
        cap = max(0, int(max_suggestions))
        return deduped[:cap] if cap else deduped

    # ---- recent-action set ---------------------------------------------

    def _recent_action_set(self, now: float) -> set:
        """Return the set of tool_ids fired within the last hour.

        Drawn from ``tool_call_log.db``; the tool_call_log only carries
        per-session earliest_ts via ``load_sessions`` (not per-row ts),
        so we open the DB directly to get true row-level timestamps.
        Empty set on any failure."""
        recent: set = set()
        if not self._tool_log_path or not self._tool_log_path.exists():
            return recent
        cutoff = now - _RECENT_ACTION_WINDOW_S
        try:
            import sqlite3
            conn = sqlite3.connect(str(self._tool_log_path), timeout=2.0)
            try:
                cursor = conn.execute(
                    "SELECT DISTINCT tool_id FROM tool_call_log WHERE ts >= ?",
                    (cutoff,),
                )
                for row in cursor.fetchall():
                    tid = row[0]
                    if tid:
                        recent.add(_normalize_action_hint(str(tid)))
            finally:
                conn.close()
        except Exception:
            return recent
        return recent

    # ---- small helpers --------------------------------------------------

    @staticmethod
    def _minutes_until(ts: float, now: float) -> int:
        if not ts or ts <= now:
            return 0
        return max(1, int((ts - now) / 60.0))

    @staticmethod
    def _humanize_age(age_s: float) -> str:
        """Render an age delta as 'just now' / '12 min ago' / '3h ago' /
        '2 days ago'. Negative or zero ages render as 'just now'."""
        try:
            age_s = float(age_s)
        except (TypeError, ValueError):
            return "recently"
        if age_s <= 60:
            return "just now"
        if age_s < 3600:
            return f"{int(age_s / 60)} min ago"
        if age_s < 86400:
            return f"{int(age_s / 3600)}h ago"
        return f"{int(age_s / 86400)} days ago"

    def _log_exc(self, name: str, exc: BaseException) -> None:
        if self._logger is None:
            return
        try:
            self._logger.exception(name, exc)
        except Exception:
            pass


__all__ = [
    "SuggestionEngine",
    "KIND_OPEN_APP",
    "KIND_CONTINUE_TASK",
    "KIND_FOLLOW_UP_PERSON",
    "KIND_PROJECT_FOCUS",
    "KIND_IDEA",
    "KIND_REMINDER",
]

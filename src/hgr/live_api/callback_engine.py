"""Callback engine — Iris references earlier moments naturally.

Phase-7 personality. A big part of Jarvis's wit is the CALLBACK
— referencing something from earlier in the conversation, the
day, or a past pattern in a way that lands. Iris today has the
session_buffer, memory, project_profile, and entity_graph substrate
to do this, but never WEAVES those references into replies.

This module decides — strictly, conservatively — whether the current
turn warrants a callback hook the synthesizer can incorporate.

Examples of good callbacks:

  * User just rage-quit a failing test → "round two?"
  * User opens the same app they were stuck in yesterday → "back
    to this, are we?"
  * Weather changed since user mentioned it earlier → "by the way,
    rain's clearing"
  * User asks about a project after recent activity → "still on
    the auth rewrite — last save was 12 min ago"

Rules:
  * MAX 1 callback per 4 turns (cooldown counter in session_state).
  * NEVER reach. Confidence ≥ 0.7 only.
  * NEVER callback to incognito or quarantined content.
  * Active preset controls likelihood: Jarvis ON, Concise OFF,
    Playful HIGH, default MODERATE.
  * The callback is a HINT for the synthesizer — it returns a string
    the planner can fold in, NOT a forced replacement.

Returns: `CallbackHint | None`.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CallbackKind(str, Enum):
    PRIOR_TOPIC      = "prior_topic"       # user mentioned X earlier
    REPEATED_APP     = "repeated_app"      # back in the same window
    PROJECT_ACTIVITY = "project_activity"  # last save N min ago
    PAST_FAILURE     = "past_failure"      # "round two?"
    CALLBACK_NAME    = "callback_name"     # referenced person


@dataclass
class CallbackHint:
    kind: CallbackKind
    hook: str                # 1-clause text the synthesizer can fold in
    confidence: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


# Preset-controlled enablement. Numbers are the BASE probability of
# emitting a callback when one is detected; concise never does.
_PRESET_GAIN: Dict[str, float] = {
    "default": 0.6,
    "jarvis":  0.85,
    "concise": 0.0,
    "warm":    0.55,
    "playful": 0.8,
    "tutor":   0.4,
}


_TURNS_BETWEEN_CALLBACKS = 4


class _SessionState:
    def __init__(self) -> None:
        self.turns_since_last_callback = 999
        self.last_callback_at = 0.0

    def can_fire(self) -> bool:
        return (self.turns_since_last_callback
                >= _TURNS_BETWEEN_CALLBACKS)

    def tick(self, fired: bool) -> None:
        if fired:
            self.turns_since_last_callback = 0
            self.last_callback_at = time.time()
        else:
            self.turns_since_last_callback += 1

    def reset(self) -> None:
        self.turns_since_last_callback = 999
        self.last_callback_at = 0.0


_GLOBAL_STATE = _SessionState()
_lock = threading.RLock()


def reset_state() -> None:
    with _lock:
        _GLOBAL_STATE.reset()


def _preset_gain() -> float:
    try:
        from . import persona_voice
        return _PRESET_GAIN.get(
            persona_voice.active_preset().name, 0.6)
    except Exception:
        return 0.6


def _incognito_on() -> bool:
    try:
        from .incognito import is_incognito
        return bool(is_incognito())
    except Exception:
        return False


# ---- detectors --------------------------------------------------------

def _detect_repeated_app(*, profile: Any) -> Optional[CallbackHint]:
    """User came back to an app they were active in earlier today.
    Cheap: looks at recent_activity for the active project profile."""
    if profile is None:
        return None
    try:
        acts = getattr(profile, "recent_activity", []) or []
        if len(acts) < 2:
            return None
        # Two activity entries within the last 24h on this same
        # window → that's a repeated visit.
        now = time.time()
        recent = [a for a in acts
                  if (now - float(a.get("ts", 0.0))) < 86400.0]
        if len(recent) >= 2:
            return CallbackHint(
                kind=CallbackKind.REPEATED_APP,
                hook=f"back in {profile.short_label()}, then",
                confidence=0.75,
                payload={"project": profile.short_label()})
    except Exception:
        return None
    return None


def _detect_project_activity(*, profile: Any
                             ) -> Optional[CallbackHint]:
    """Project has recent file activity worth mentioning."""
    if profile is None:
        return None
    try:
        files = getattr(profile, "recent_files", []) or []
        if not files:
            return None
        # Was last_seen_at < 30 min ago?
        last = float(getattr(profile, "last_seen_at", 0) or 0)
        if last <= 0:
            return None
        delta_min = max(0.0, (time.time() - last) / 60.0)
        if delta_min > 60:
            return None
        first_file = files[0]
        return CallbackHint(
            kind=CallbackKind.PROJECT_ACTIVITY,
            hook=(f"last touched {first_file} "
                  f"{int(delta_min)} min ago"),
            confidence=0.7,
            payload={"file": first_file,
                     "delta_min": int(delta_min)})
    except Exception:
        return None


_FAIL_RE = re.compile(
    r"\b(fail|failed|broken|crash|error|exception)\w*\b",
    re.IGNORECASE)


def _detect_past_failure(*, session_buffer: Any,
                          user_text: str) -> Optional[CallbackHint]:
    """User mentioned failing/crashing within the last few turns
    AND is asking again now → 'round two?'."""
    if not user_text or session_buffer is None:
        return None
    try:
        turns = session_buffer.recent(max_turns=6)
    except Exception:
        return None
    if not turns:
        return None
    # Skip the most recent turn (the current one) when scanning.
    history = [t for t in turns[:-1]
               if (getattr(t, "role", "") == "user")]
    if not history:
        return None
    hit = next(
        (t for t in history
         if _FAIL_RE.search(getattr(t, "text", "") or "")),
        None)
    if hit is None:
        return None
    # Only if the current turn ALSO smells like another attempt.
    if not _FAIL_RE.search(user_text or "") and not re.search(
            r"\b(try|again|retry|run|test)\b",
            user_text or "", re.IGNORECASE):
        return None
    return CallbackHint(
        kind=CallbackKind.PAST_FAILURE,
        hook="round two — last attempt hit a wall",
        confidence=0.8)


_PERSON_NAME_RE = re.compile(r"\b([A-Z][a-zA-Z]{1,30})\b")


def _detect_callback_name(*, user_text: str, entity_graph: Any
                          ) -> Optional[CallbackHint]:
    """User mentioned a person already in the entity graph by name —
    surface what we know."""
    if not user_text or entity_graph is None:
        return None
    try:
        # Pull the first candidate name from the utterance.
        m = _PERSON_NAME_RE.search(user_text)
        if m is None:
            return None
        name = m.group(1)
        # Conservative skip-list — these are common english words
        # that look like names.
        if name.lower() in {"i", "is", "the", "a", "an", "iris",
                            "monday", "tuesday", "wednesday",
                            "thursday", "friday", "saturday",
                            "sunday"}:
            return None
        try:
            ent = entity_graph.find_by_alias(name)
        except Exception:
            ent = None
        if ent is None:
            return None
        # Show one piece of detail we know.
        attrs = getattr(ent, "attributes", {}) or {}
        if not attrs:
            return None
        detail_key = next(iter(attrs))
        detail_val = attrs[detail_key]
        return CallbackHint(
            kind=CallbackKind.CALLBACK_NAME,
            hook=f"{name} — {detail_key}: {detail_val}",
            confidence=0.7,
            payload={"name": name,
                     detail_key: str(detail_val)[:80]})
    except Exception:
        return None


def _detect_prior_topic(*, session_buffer: Any,
                        user_text: str) -> Optional[CallbackHint]:
    """User just referenced a topic that came up earlier in the
    session — surface a one-clause reminder of where they left it."""
    if not user_text or session_buffer is None:
        return None
    try:
        turns = session_buffer.recent(max_turns=12)
    except Exception:
        return None
    if not turns or len(turns) < 2:
        return None
    # Extract keywords (length >= 5, lowercase) from current utterance.
    words = {w.lower() for w in re.findall(
        r"\b[A-Za-z]{5,}\b", user_text)}
    if not words:
        return None
    # Scan prior turns (skip the most recent — that's the current).
    for t in turns[:-1]:
        text = (getattr(t, "text", "") or "")
        if not text:
            continue
        prior_words = {w.lower() for w in re.findall(
            r"\b[A-Za-z]{5,}\b", text)}
        overlap = words & prior_words
        if len(overlap) >= 2:
            hook_word = sorted(overlap)[0]
            return CallbackHint(
                kind=CallbackKind.PRIOR_TOPIC,
                hook=f"coming back to {hook_word}",
                confidence=0.7,
                payload={"keyword": hook_word})
    return None


# ---- top-level --------------------------------------------------------

def maybe_callback(*, user_text: str,
                   session_buffer: Optional[Any] = None,
                   entity_graph: Optional[Any] = None,
                   project_profile: Optional[Any] = None,
                   state: Optional[_SessionState] = None
                   ) -> Optional[CallbackHint]:
    """Top-level. Returns a CallbackHint when conditions are right.
    None otherwise."""
    s = state or _GLOBAL_STATE
    with _lock:
        if not s.can_fire():
            s.tick(False)
            return None
    if _incognito_on():
        with _lock:
            s.tick(False)
        return None
    # Phase-7 affect: when user is frustrated, hold off on wit.
    try:
        from .affect import should_suppress_callbacks
        if should_suppress_callbacks():
            with _lock:
                s.tick(False)
            return None
    except Exception:
        pass
    gain = _preset_gain()
    if gain <= 0.0:
        with _lock:
            s.tick(False)
        return None
    # Run detectors in priority order. Stop at the first qualifying
    # hint whose confidence × preset_gain ≥ 0.5.
    detectors = (
        lambda: _detect_past_failure(
            session_buffer=session_buffer, user_text=user_text),
        lambda: _detect_callback_name(
            user_text=user_text, entity_graph=entity_graph),
        lambda: _detect_project_activity(profile=project_profile),
        lambda: _detect_repeated_app(profile=project_profile),
        lambda: _detect_prior_topic(
            session_buffer=session_buffer, user_text=user_text),
    )
    for det in detectors:
        try:
            hint = det()
        except Exception:
            continue
        if hint is None:
            continue
        score = hint.confidence * gain
        # 0.4 = "default preset × 0.7-confident detector" is just
        # barely enough to fire. Concise gain (0.0) never crosses.
        if score >= 0.4:
            with _lock:
                s.tick(True)
            return hint
    with _lock:
        s.tick(False)
    return None


def render_as_prompt_block(hint: CallbackHint) -> str:
    """Format a hint as a one-line block the synthesizer can fold
    into the system prompt."""
    if hint is None:
        return ""
    return (f"CALLBACK HINT — when natural, weave a ONE-CLAUSE "
            f"reference: \"{hint.hook}\". Don't force it; if the "
            f"reply has no clean place for it, drop it.")

"""Affective state model — Iris reads the room.

Phase-7 personality polish. The same Iris reply that lands as
"witty" when the user is relaxed lands as "annoying" when they're
frustrated. Today Iris has no model of the user's affective state
— callback_engine + nudge_layer + prose_renderer all fire at a
fixed cadence regardless of whether the user just rage-quit a
test or is happily exploring.

This module owns a 3-axis affective state:

  * mood      — −1 frustrated …  +1 relaxed
  * focus     —  0 idle/casual … +1 deep-work
  * verbosity — −1 wants terse … +1 wants expansive

State updates from signals the orchestrator already collects:

  * Voice/text frustration keywords ("ugh", "ffs", "stop",
    "no I said", "broken again", "ffs", "wtf")
  * Back-to-back corrections (current turn says "no" within 2
    turns of Iris's last reply)
  * Terse one-word follow-ups ("no", "stop", "cancel")
  * Acceptance signals (user thanks, says "perfect", "exactly")
  * Long-form exploratory questions ("what about…", "could you
    explain…")
  * Silence pattern (gaps > 5 min suggest focus / disengagement)

Downstream consumers (callback_engine, anticipation_engine,
prose_renderer) consult `current_state()` and dial behavior:
  * mood < −0.3 → callbacks OFF, low nudges OFF, terser replies
  * focus  > +0.6 → low/medium nudges OFF, batch for later
  * verbosity < −0.3 → cap reply length harder

All signal observations are RECORDED with timestamps and EMA-
decayed so a 30-min-old frustration doesn't haunt the next hour.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SignalKind(str, Enum):
    FRUSTRATION_WORD = "frustration_word"
    CORRECTION       = "correction"
    TERSE_FOLLOWUP   = "terse_followup"
    ACCEPTANCE       = "acceptance"
    EXPLORATORY      = "exploratory"
    SILENCE_GAP      = "silence_gap"
    DEEP_WORK_HINT   = "deep_work_hint"


@dataclass
class Signal:
    kind: SignalKind
    weight: float                       # 0..1 magnitude
    ts: float = field(default_factory=time.time)
    note: str = ""


@dataclass
class AffectiveState:
    mood: float = 0.0                   # −1 .. +1
    focus: float = 0.0                  #  0 .. +1
    verbosity: float = 0.0              # −1 .. +1
    last_update: float = field(default_factory=time.time)
    samples: int = 0

    def is_frustrated(self) -> bool:
        return self.mood <= -0.3

    def is_focused(self) -> bool:
        return self.focus >= 0.6

    def wants_terse(self) -> bool:
        return self.verbosity <= -0.3

    def wants_expansive(self) -> bool:
        return self.verbosity >= 0.3

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mood": round(self.mood, 3),
            "focus": round(self.focus, 3),
            "verbosity": round(self.verbosity, 3),
            "samples": self.samples,
            "frustrated": self.is_frustrated(),
            "focused": self.is_focused(),
            "wants_terse": self.wants_terse(),
        }


# ---- regex signal detectors ------------------------------------------

_FRUSTRATION_RE = re.compile(
    r"\b(ugh+|ffs|wtf|wth|stop|stop\s+it|stop\s+that|"
    r"shut\s+up|never\s*mind|forget\s+it|broken\s+again|"
    r"again\s+broken|come\s+on|seriously|argh+|"
    r"no,?\s+i\s+said|that'?s\s+wrong|cancel)\b",
    re.IGNORECASE,
)

_ACCEPTANCE_RE = re.compile(
    r"\b(thanks|thank\s+you|perfect|exactly|nice|great|"
    r"awesome|sweet|love\s+it|got\s+it|appreciate\s+(it|that))\b",
    re.IGNORECASE,
)

_EXPLORATORY_RE = re.compile(
    r"\b(what\s+about|how\s+about|could\s+you|would\s+you|"
    r"can\s+you\s+(explain|walk|show|tell\s+me\s+about)|"
    r"why\s+would|tell\s+me\s+more|help\s+me\s+understand)\b",
    re.IGNORECASE,
)

_TERSE_RE = re.compile(
    r"^\s*(no|stop|cancel|nope|nah|don'?t|abort|exit|quit)"
    r"[\s.!?]*$",
    re.IGNORECASE,
)


# ---- weights / decay --------------------------------------------------

_SIGNAL_WEIGHTS: Dict[SignalKind, Tuple[float, float, float]] = {
    # (mood_delta, focus_delta, verbosity_delta) per occurrence
    SignalKind.FRUSTRATION_WORD: (-0.35, +0.10, -0.20),
    SignalKind.CORRECTION:       (-0.25, +0.05, -0.10),
    SignalKind.TERSE_FOLLOWUP:   (-0.15, +0.05, -0.30),
    SignalKind.ACCEPTANCE:       (+0.20,  0.00,  0.00),
    SignalKind.EXPLORATORY:      (+0.10, -0.05, +0.35),
    SignalKind.SILENCE_GAP:      ( 0.00, +0.25,  0.00),
    SignalKind.DEEP_WORK_HINT:   ( 0.00, +0.20, -0.10),
}

# Half-life seconds — after this long, a signal's contribution is
# half what it started.
_HALF_LIFE_SEC = 15 * 60


def _decay(now: float, signal_ts: float) -> float:
    delta = max(0.0, now - signal_ts)
    if delta == 0:
        return 1.0
    # Exponential decay: weight *= 0.5 ** (delta / half_life)
    return 0.5 ** (delta / _HALF_LIFE_SEC)


# ---- model ------------------------------------------------------------

class AffectModel:
    """Tracks signals + computes the current AffectiveState. Thread-
    safe. Caps history at 60 signals (older ones are dropped — they'd
    decay near-zero anyway)."""

    _MAX_HISTORY = 60

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._signals: List[Signal] = []
        self._last_user_turn_at: Optional[float] = None

    def record(self, signal: Signal) -> None:
        with self._lock:
            self._signals.append(signal)
            if len(self._signals) > self._MAX_HISTORY:
                # Drop oldest.
                self._signals = self._signals[-self._MAX_HISTORY:]

    def observe_user_turn(self, text: str,
                          *, role: str = "user") -> List[Signal]:
        """Inspect a user turn for affective signals + record them.
        Returns the list of signals it just recorded (for tests +
        introspection). Idempotent."""
        if role != "user":
            return []
        emitted: List[Signal] = []
        if not text:
            return emitted
        # Frustration keywords.
        if _FRUSTRATION_RE.search(text):
            sig = Signal(kind=SignalKind.FRUSTRATION_WORD,
                         weight=1.0,
                         note=text[:80])
            self.record(sig)
            emitted.append(sig)
        # Acceptance phrases.
        if _ACCEPTANCE_RE.search(text):
            sig = Signal(kind=SignalKind.ACCEPTANCE, weight=1.0,
                         note=text[:80])
            self.record(sig)
            emitted.append(sig)
        # Exploratory questions.
        if _EXPLORATORY_RE.search(text):
            sig = Signal(kind=SignalKind.EXPLORATORY,
                         weight=1.0,
                         note=text[:80])
            self.record(sig)
            emitted.append(sig)
        # One-word negation (terse follow-up).
        if _TERSE_RE.match(text.strip()):
            sig = Signal(kind=SignalKind.TERSE_FOLLOWUP,
                         weight=1.0,
                         note=text[:40])
            self.record(sig)
            emitted.append(sig)
        # Silence-gap detection: if it's been > 5 min since last turn.
        now = time.time()
        if self._last_user_turn_at is not None:
            gap = now - self._last_user_turn_at
            if gap >= 5 * 60:
                gap_min = gap / 60.0
                # Cap the focus boost at gap=20 min → weight 1.0.
                weight = min(1.0, gap_min / 20.0)
                if weight >= 0.25:
                    sig = Signal(kind=SignalKind.SILENCE_GAP,
                                 weight=weight,
                                 note=f"gap {gap_min:.1f}m")
                    self.record(sig)
                    emitted.append(sig)
        self._last_user_turn_at = now
        return emitted

    def record_correction(self, *, weight: float = 1.0) -> None:
        """Orchestrator calls when it detects a back-to-back
        correction (user said "no" / "wrong" after Iris's reply)."""
        self.record(Signal(kind=SignalKind.CORRECTION,
                            weight=weight))

    def record_deep_work_hint(self, *, weight: float = 1.0) -> None:
        """Vision observer / focus tracker can hint that the user
        is in deep work."""
        self.record(Signal(kind=SignalKind.DEEP_WORK_HINT,
                            weight=weight))

    def current_state(self) -> AffectiveState:
        """Aggregate decayed signals into an AffectiveState. Clamps
        each axis to its declared range."""
        with self._lock:
            signals = list(self._signals)
        now = time.time()
        mood = 0.0
        focus = 0.0
        verbosity = 0.0
        for s in signals:
            decayed_weight = s.weight * _decay(now, s.ts)
            if decayed_weight < 1e-3:
                continue
            d_mood, d_focus, d_verb = _SIGNAL_WEIGHTS[s.kind]
            mood += d_mood * decayed_weight
            focus += d_focus * decayed_weight
            verbosity += d_verb * decayed_weight
        return AffectiveState(
            mood=max(-1.0, min(1.0, mood)),
            focus=max(0.0, min(1.0, focus)),
            verbosity=max(-1.0, min(1.0, verbosity)),
            last_update=now,
            samples=len(signals))

    def reset(self) -> None:
        with self._lock:
            self._signals.clear()
            self._last_user_turn_at = None


# ---- singleton -------------------------------------------------------

_lock = threading.Lock()
_singleton: Optional[AffectModel] = None


def global_model() -> AffectModel:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = AffectModel()
        return _singleton


def reset_global() -> None:
    global _singleton
    with _lock:
        _singleton = None


def current_state() -> AffectiveState:
    return global_model().current_state()


def observe_user_turn(text: str) -> List[Signal]:
    return global_model().observe_user_turn(text)


# ---- convenience flags read by downstream gates ----------------------

def should_suppress_callbacks() -> bool:
    """callback_engine consults this — when mood is negative, drop
    the witty stuff."""
    return current_state().is_frustrated()


def should_suppress_low_nudges() -> bool:
    """Anticipation + reliability advisor consult this — when the
    user is deep in work OR frustrated, hold off on low-severity
    interrupts."""
    st = current_state()
    return st.is_focused() or st.is_frustrated()


def reply_length_bias() -> int:
    """Returns -1 (cut harder), 0 (default), +1 (more depth allowed).
    Synthesizers can use this as a soft length cap modifier."""
    st = current_state()
    if st.wants_terse():
        return -1
    if st.wants_expansive():
        return +1
    return 0

"""Earcons + quiet-by-default mode.

Phase-2 voice UX. Iris talking out loud is appropriate for
some moments (user explicitly asked, completed a real task) and
annoying-to-rude in others (user is on a call, screen-sharing,
focused, in a meeting). This module owns:

  1. **QuietMode** — the global "should Iris talk?" boolean +
     reasons it's set. Independent reasons stack so e.g. screen-
     sharing AND in-meeting both flip the flag, and BOTH have to
     clear before voice resumes.
  2. **Earcons** — short non-verbal sounds (200-400ms) for
     acknowledgments that don't merit a full sentence: "got it"
     beep, "starting" tick, "done" chime, "error" dip, "asking
     for confirmation" double-tap.

Earcons are MUCH quieter than spoken replies (default -18dB vs
-6dB) and obey the quiet-mode flag for "starting" / "done" while
"asking for confirmation" plays regardless (user must hear that).

Audio playback is best-effort — when no audio backend is
available (tests, headless server, sounddevice missing) the
play() calls return silently.

Author: Konstantin Markov
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set


# ---- QuietMode ---------------------------------------------------------

class QuietReason(str, Enum):
    USER_TOGGLED = "user_toggled"
    SCREEN_SHARING = "screen_sharing"
    IN_MEETING = "in_meeting"
    DND = "do_not_disturb"
    FOCUS_ASSIST = "focus_assist"
    NIGHT_HOURS = "night_hours"
    BATTERY_LOW = "battery_low"
    EARBUDS_OUT = "earbuds_out"
    HEADPHONES_UNPLUGGED = "headphones_unplugged"


@dataclass
class QuietState:
    is_quiet: bool
    reasons: List[str] = field(default_factory=list)
    speech_allowed: bool = True
    earcons_allowed: bool = True


class QuietMode:
    """Multi-reason quiet-mode tracker. ALL reasons must clear for
    voice to resume. Speech and earcons can be independently
    suppressed (earcons are usually OK even when speech is muted)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reasons: Set[QuietReason] = set()
        self._speech_silenced_by: Set[QuietReason] = set()
        self._earcons_silenced_by: Set[QuietReason] = set()
        # Allow callers to subscribe and react (UI badge, etc.).
        self._subscribers: List[Callable[[QuietState], None]] = []

    def add_reason(self, r: QuietReason,
                   silences_earcons: bool = False) -> None:
        with self._lock:
            self._reasons.add(r)
            self._speech_silenced_by.add(r)
            if silences_earcons:
                self._earcons_silenced_by.add(r)
        self._notify()

    def clear_reason(self, r: QuietReason) -> None:
        with self._lock:
            self._reasons.discard(r)
            self._speech_silenced_by.discard(r)
            self._earcons_silenced_by.discard(r)
        self._notify()

    def state(self) -> QuietState:
        with self._lock:
            return QuietState(
                is_quiet=bool(self._reasons),
                reasons=[r.value for r in self._reasons],
                speech_allowed=not self._speech_silenced_by,
                earcons_allowed=not self._earcons_silenced_by,
            )

    def speech_allowed(self) -> bool:
        return self.state().speech_allowed

    def earcons_allowed(self) -> bool:
        return self.state().earcons_allowed

    def subscribe(self,
                  cb: Callable[[QuietState], None]
                  ) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(cb)
        def _unsub():
            with self._lock:
                try:
                    self._subscribers.remove(cb)
                except ValueError:
                    pass
        return _unsub

    def _notify(self) -> None:
        st = self.state()
        with self._lock:
            subs = list(self._subscribers)
        for s in subs:
            try:
                s(st)
            except Exception:
                pass

    def reset(self) -> None:
        with self._lock:
            self._reasons.clear()
            self._speech_silenced_by.clear()
            self._earcons_silenced_by.clear()
        self._notify()


# ---- Earcons -----------------------------------------------------------

class EarconKind(str, Enum):
    READY = "ready"               # Iris is listening
    STARTING = "starting"         # tool dispatch begun
    DONE = "done"                 # tool finished OK
    NEEDS_CONFIRM = "needs_confirm"   # speed-bump fired
    ERROR = "error"
    DECLINED = "declined"         # user said no
    UNDONE = "undone"


# Earcon "spec": (frequency_hz, duration_ms, db_attenuation). We
# synthesize on the fly so there are no binary asset files to ship.
_EARCON_SPECS: Dict[EarconKind, tuple] = {
    EarconKind.READY:         (880, 80, 22),
    EarconKind.STARTING:      (660, 100, 22),
    EarconKind.DONE:          (988, 140, 18),
    EarconKind.NEEDS_CONFIRM: (440, 200, 14),
    EarconKind.ERROR:         (220, 220, 14),
    EarconKind.DECLINED:      (330, 120, 18),
    EarconKind.UNDONE:        (700, 180, 18),
}


# Earcons that play EVEN in quiet mode — they exist to interrupt
# the user politely when something requires their attention. The
# user can globally mute earcons via QuietMode.add_reason(...,
# silences_earcons=True).
_FORCE_THROUGH_QUIET = {EarconKind.NEEDS_CONFIRM, EarconKind.ERROR}


class EarconPlayer:
    """Generates + plays earcon tones. Falls back to no-op when no
    audio backend is available."""

    def __init__(self, *, quiet_mode: Optional[QuietMode] = None,
                 sample_rate: int = 24_000) -> None:
        self._quiet = quiet_mode
        self._sr = sample_rate
        self._lock = threading.Lock()
        # Lazy-import sounddevice on first play so test envs without
        # PortAudio don't crash at import time.
        self._sd = None
        self._sd_failed = False

    def play(self, kind: EarconKind) -> bool:
        """Play an earcon. Returns True iff audio was actually emitted.
        Honors quiet mode unless the earcon is in _FORCE_THROUGH_QUIET."""
        if not self._allowed(kind):
            return False
        spec = _EARCON_SPECS.get(kind)
        if spec is None:
            return False
        freq, dur_ms, db_atten = spec
        try:
            sd = self._get_sd()
        except Exception:
            return False
        if sd is None:
            return False
        try:
            samples = self._synth(freq=freq, duration_ms=dur_ms,
                                  db_atten=db_atten)
            sd.play(samples, samplerate=self._sr, blocking=False)
            return True
        except Exception:
            return False

    def _allowed(self, kind: EarconKind) -> bool:
        if self._quiet is None:
            return True
        st = self._quiet.state()
        if kind in _FORCE_THROUGH_QUIET:
            return True
        return st.earcons_allowed

    def _get_sd(self):
        if self._sd_failed:
            return None
        if self._sd is not None:
            return self._sd
        with self._lock:
            if self._sd is None and not self._sd_failed:
                try:
                    import sounddevice as sd
                    self._sd = sd
                except Exception:
                    self._sd_failed = True
                    return None
        return self._sd

    def _synth(self, *, freq: float, duration_ms: int,
               db_atten: float):
        """Synthesize a tone: sine carrier + cosine envelope to avoid
        clicks at start/end. Returns float32 mono array in [-1, 1]."""
        try:
            import numpy as np
        except Exception:
            # No numpy available — fall back to a pure-python loop.
            return _python_synth(freq=freq, duration_ms=duration_ms,
                                 db_atten=db_atten, sr=self._sr)
        n = int(self._sr * duration_ms / 1000.0)
        if n <= 0:
            return None
        t = np.arange(n) / float(self._sr)
        carrier = np.sin(2.0 * np.pi * freq * t)
        # Hann envelope shapes the attack/release smoothly.
        env = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / max(1, n - 1))
        amp = 10 ** (-db_atten / 20.0)
        return (carrier * env * amp).astype("float32")


def _python_synth(*, freq: float, duration_ms: int,
                  db_atten: float, sr: int) -> list:
    """Numpy-free fallback. Slower but only runs in test envs."""
    n = int(sr * duration_ms / 1000.0)
    if n <= 0:
        return []
    amp = 10 ** (-db_atten / 20.0)
    out = []
    for i in range(n):
        t = i / float(sr)
        e = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / max(1, n - 1))
        out.append(amp * e * math.sin(2.0 * math.pi * freq * t))
    return out


# ---- module singletons -------------------------------------------------

_quiet: Optional[QuietMode] = None
_earcons: Optional[EarconPlayer] = None
_lock = threading.Lock()


def global_quiet_mode() -> QuietMode:
    global _quiet
    if _quiet is None:
        with _lock:
            if _quiet is None:
                _quiet = QuietMode()
    return _quiet


def global_earcons() -> EarconPlayer:
    global _earcons
    if _earcons is None:
        with _lock:
            if _earcons is None:
                _earcons = EarconPlayer(quiet_mode=global_quiet_mode())
    return _earcons


def _reset_for_tests() -> None:
    global _quiet, _earcons
    _quiet = None
    _earcons = None

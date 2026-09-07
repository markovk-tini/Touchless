"""Runtime matcher for pose-sequence custom gestures.

A pose sequence is an ordered list of held static poses (e.g. finger
counts 3 → 2 → 1). Timing is part of the gesture:

- Each pose must match continuously for at least `dwell_ms`.
- Holding the same pose continuously longer than `max_hold_ms` fails
  (resets) the sequence.
- After a counted pose is released, the next pose must appear within
  `max_gap_ms` or the sequence fails.
- Intermediate steps advance on release after dwell; the final step
  fires as soon as dwell is met.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .action import fire_once
from .classifier import GestureClassifier
from .recorder import normalize_landmarks
from .registry import (
    CustomGesture,
    GestureRegistry,
    registry_path,
)


_DEFAULT_MATCH_THRESHOLD = 0.70
_HAND_LOST_GRACE_S = 1.0
# Learned recordings can store a 150 ms gap / ~1 s max-hold. GPU
# tracking flicker and a natural 3→2 finger fold both exceed that, so
# live matching floors these without rewriting the saved JSON.
_LIVE_MIN_GAP_S = 0.85
_LIVE_MIN_MAX_HOLD_S = 1.80
# Recording a 3→2→1 often stores ~300 ms dwell because the user held
# each pose slowly. Live counting is faster; cap so a quick 3-2-1
# still advances. Tests that pass live_timing_floor=False are unchanged.
_LIVE_MAX_DWELL_S = 0.16
_HINT_LABELS = frozenset({
    "one", "two", "three", "four", "fist", "ok", "peace", "mute",
})


@dataclass
class _SequenceState:
    step_index: int = 0
    dwell_started_at: Optional[float] = None
    last_match_at: Optional[float] = None
    # True once dwell was met while still holding — waiting for release
    # to advance (non-final steps only).
    dwell_met: bool = False
    # Builtin labels locked to completed/current steps (hint assist).
    locked_hints: List[str] = field(default_factory=list)


@dataclass
class _HintMatch:
    score: float = 0.99


def _reset_state(state: _SequenceState) -> None:
    state.step_index = 0
    state.dwell_started_at = None
    state.last_match_at = None
    state.dwell_met = False
    state.locked_hints = []


class PoseSequenceRuntime:
    """Live matcher for kind='pose_sequence' gestures."""

    def __init__(
        self,
        *,
        match_threshold: float = _DEFAULT_MATCH_THRESHOLD,
        live_timing_floor: bool = True,
    ) -> None:
        self._match_threshold = float(match_threshold)
        self._live_timing_floor = bool(live_timing_floor)
        self._registry: Optional[GestureRegistry] = None
        self._entries: List[Tuple[CustomGesture, List[GestureClassifier]]] = []
        self._states: dict = {}  # name -> _SequenceState
        self._registry_path = registry_path()
        self._registry_mtime: float = 0.0
        self._last_mtime_check_at: float = 0.0
        self._last_fire_name: Optional[str] = None
        self._last_fire_at: float = 0.0
        self._absent_since: Optional[float] = None
        self._debug_enabled = os.environ.get("HGR_CUSTOM_GESTURES_DEBUG", "1") != "0"
        self._last_debug_log_at: float = 0.0

    def has_sequences(self) -> bool:
        return bool(self._entries)

    def is_in_progress(self) -> bool:
        for state in self._states.values():
            if int(state.step_index) > 0 or state.dwell_started_at is not None:
                return True
        return False

    def current_banner(self) -> Optional[tuple]:
        """(label, handedness) while a sequence is mid-count, so the
        overlay shows `countdown 2/3` instead of builtin `three`."""
        for gesture, step_clfs in self._entries:
            state = self._states.get(gesture.name)
            if state is None:
                continue
            if int(state.step_index) <= 0 and state.dwell_started_at is None:
                continue
            n = max(1, len(step_clfs))
            step = min(n, int(state.step_index) + 1)
            # Handedness None so GPU's Left/Right flicker still shows
            # the sequence name on the visible hand.
            return (f"{gesture.name} {step}/{n}", None)
        return None

    def reload(self) -> None:
        try:
            self._registry = GestureRegistry()
            self._registry.load()
        except Exception:
            self._registry = None
            self._entries = []
            self._states = {}
            self._absent_since = None
            return
        entries: List[Tuple[CustomGesture, List[GestureClassifier]]] = []
        for g in self._registry.list():
            if str(getattr(g, "kind", "") or "") != "pose_sequence":
                continue
            steps = list(g.pose_sequence_steps or [])
            if len(steps) < 2:
                continue
            step_clfs: List[GestureClassifier] = []
            ok = True
            for i, step in enumerate(steps):
                if not step.samples:
                    ok = False
                    break
                # Pseudo static gesture so GestureClassifier can score
                # this step's samples in isolation.
                pseudo = CustomGesture(
                    name=f"{g.name}::__step_{i}",
                    samples=list(step.samples),
                    action=g.action,
                    created_at=g.created_at,
                    kind="static",
                    handedness=g.handedness,
                )
                clf = GestureClassifier(
                    gestures=[pseudo],
                    threshold=self._match_threshold,
                    confidence_margin=0.0,
                )
                clf.reload()
                step_clfs.append(clf)
            if ok and step_clfs:
                entries.append((g, step_clfs))
        self._entries = entries
        self._states = {g.name: _SequenceState() for g, _ in entries}
        self._absent_since = None
        try:
            self._registry_mtime = float(self._registry_path.stat().st_mtime)
        except Exception:
            self._registry_mtime = 0.0

    def maybe_reload_if_changed(self, now: Optional[float] = None) -> None:
        now = float(now if now is not None else time.monotonic())
        if now - self._last_mtime_check_at < 3.0:
            return
        self._last_mtime_check_at = now
        try:
            mtime = float(self._registry_path.stat().st_mtime)
        except Exception:
            return
        if mtime != self._registry_mtime:
            self.reload()

    def hand_lost(self, now: Optional[float] = None) -> None:
        """Reset only after a sustained absence.

        A one-frame GPU tracking drop must not wipe a mid-count 3→2→1
        sequence. Gap timing in process_landmarks still applies once
        landmarks return; this path is the hard reset.
        """
        if not self._entries:
            return
        now = float(now if now is not None else time.monotonic())
        if not self.is_in_progress():
            self._absent_since = None
            return
        if self._absent_since is None:
            self._absent_since = now
            return
        if (now - float(self._absent_since)) >= _HAND_LOST_GRACE_S:
            for state in self._states.values():
                _reset_state(state)
            self._absent_since = None

    def process_landmarks(
        self,
        landmarks: np.ndarray,
        *,
        handedness: str = "",
        timestamp: Optional[float] = None,
        dispatch: bool = True,
        strict_hand: bool = True,
        hint_label: str = "",
    ) -> Optional[str]:
        """Advance sequence state from a (21, 3) landmark frame.

        Returns the fired gesture name, or None.
        """
        if not self._entries:
            return None
        now = float(timestamp if timestamp is not None else time.monotonic())
        if self._absent_since is not None:
            pause = max(0.0, now - float(self._absent_since))
            self._absent_since = None
            if pause > 0.0:
                for state in self._states.values():
                    if state.dwell_started_at is not None:
                        state.dwell_started_at += pause
                    if state.last_match_at is not None:
                        state.last_match_at += pause
        try:
            features = normalize_landmarks(landmarks)
        except Exception:
            return None
        hand = str(handedness or "").strip()
        fired: Optional[str] = None
        for gesture, step_clfs in self._entries:
            wanted = gesture.handedness
            if (
                strict_hand
                and wanted in ("Left", "Right")
                and hand in ("Left", "Right")
                and wanted != hand
            ):
                continue
            state = self._states.setdefault(gesture.name, _SequenceState())
            dwell_s = max(0.08, float(gesture.pose_sequence_dwell_ms) / 1000.0)
            max_hold_s = max(
                dwell_s,
                float(gesture.pose_sequence_max_hold_ms) / 1000.0,
            )
            gap_s = max(0.1, float(gesture.pose_sequence_max_gap_ms) / 1000.0)
            if self._live_timing_floor:
                dwell_s = min(dwell_s, _LIVE_MAX_DWELL_S)
                max_hold_s = max(max_hold_s, _LIVE_MIN_MAX_HOLD_S)
                gap_s = max(gap_s, _LIVE_MIN_GAP_S)

            # Same-frame retry after a release-advance so a tight 3→2
            # fold can count as pose 2 immediately instead of burning
            # a gap-window frame.
            for _attempt in range(2):
                step_i = int(state.step_index)
                if step_i < 0 or step_i >= len(step_clfs):
                    _reset_state(state)
                    step_i = 0
                clf = step_clfs[step_i]
                is_last = step_i >= len(step_clfs) - 1
                next_clf = None if is_last else step_clfs[step_i + 1]
                # No sticky hysteresis: 3 vs 2 are similar, and sticky
                # would keep step 1 locked after the fold.
                match = clf.classify_raw(features)
                if match is not None and next_clf is not None:
                    try:
                        nxt = float(next_clf.raw_score(features))
                        if nxt >= float(clf.threshold) and nxt > float(match.score):
                            match = None
                    except Exception:
                        pass
                if match is None:
                    match = self._hint_match(state, step_i, hint_label)
                if self._debug_enabled and (now - self._last_debug_log_at) >= 0.5:
                    try:
                        _, score = clf.best_score_for(landmarks)
                    except Exception:
                        score = 0.0
                    self._maybe_debug(
                        now,
                        gesture.name,
                        f"step={step_i + 1}/{len(step_clfs)}",
                        "match" if match is not None else "miss",
                        f"score={score:.2f}",
                        f"dwell={'Y' if state.dwell_met else 'n'}",
                    )

                if match is not None:
                    self._lock_hint(state, step_i, hint_label)
                    state.last_match_at = now
                    if state.dwell_started_at is None:
                        state.dwell_started_at = now
                        state.dwell_met = False
                    held = now - float(state.dwell_started_at)
                    if held > max_hold_s:
                        _reset_state(state)
                        break
                    if held >= dwell_s:
                        if is_last:
                            _reset_state(state)
                            if dispatch:
                                try:
                                    fire_once(gesture.name, gesture.action)
                                except Exception:
                                    pass
                            self._last_fire_name = gesture.name
                            self._last_fire_at = now
                            fired = gesture.name
                        else:
                            state.dwell_met = True
                    break
                if state.dwell_met and not is_last:
                    state.step_index = step_i + 1
                    state.dwell_started_at = None
                    state.dwell_met = False
                    state.last_match_at = now
                    continue
                if state.step_index > 0 or state.dwell_started_at is not None:
                    last = state.last_match_at
                    if last is None or (now - float(last)) > gap_s:
                        _reset_state(state)
                    else:
                        state.dwell_started_at = None
                        state.dwell_met = False
                break
        return fired

    def _lock_hint(self, state: _SequenceState, step_i: int, hint_label: str) -> None:
        hint = str(hint_label or "").strip().lower()
        if hint not in _HINT_LABELS:
            return
        locks = state.locked_hints
        if step_i == len(locks) and hint not in locks:
            locks.append(hint)

    def _hint_match(
        self, state: _SequenceState, step_i: int, hint_label: str
    ) -> Optional[_HintMatch]:
        """When KNN misses, still count a held builtin pose (three/two/one).

        The live chip already labels those poses. A recorded 3→2→1 often
        fails KNN because the samples are Left/Right or GPU-drifted, while
        the builtin recognizer is sure. Require a *new* builtin label per
        step so three-three-three cannot walk the whole sequence.
        """
        hint = str(hint_label or "").strip().lower()
        if hint not in _HINT_LABELS:
            return None
        locks = state.locked_hints
        if 0 <= step_i < len(locks):
            return _HintMatch() if hint == locks[step_i] else None
        if step_i != len(locks):
            return None
        if hint in locks:
            return None
        locks.append(hint)
        return _HintMatch()

    def _maybe_debug(self, now: float, *parts: object) -> None:
        if not self._debug_enabled:
            return
        if now - self._last_debug_log_at < 0.5:
            return
        self._last_debug_log_at = now
        print("[pose-sequence]", *parts)

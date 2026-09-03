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

import time
from dataclasses import dataclass
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


_DEFAULT_MATCH_THRESHOLD = 0.78


@dataclass
class _SequenceState:
    step_index: int = 0
    dwell_started_at: Optional[float] = None
    last_match_at: Optional[float] = None
    # True once dwell was met while still holding — waiting for release
    # to advance (non-final steps only).
    dwell_met: bool = False


def _reset_state(state: _SequenceState) -> None:
    state.step_index = 0
    state.dwell_started_at = None
    state.last_match_at = None
    state.dwell_met = False


class PoseSequenceRuntime:
    """Live matcher for kind='pose_sequence' gestures."""

    def __init__(self, *, match_threshold: float = _DEFAULT_MATCH_THRESHOLD) -> None:
        self._match_threshold = float(match_threshold)
        self._registry: Optional[GestureRegistry] = None
        self._entries: List[Tuple[CustomGesture, List[GestureClassifier]]] = []
        self._states: dict = {}  # name -> _SequenceState
        self._registry_path = registry_path()
        self._registry_mtime: float = 0.0
        self._last_mtime_check_at: float = 0.0
        self._last_fire_name: Optional[str] = None
        self._last_fire_at: float = 0.0

    def has_sequences(self) -> bool:
        return bool(self._entries)

    def reload(self) -> None:
        try:
            self._registry = GestureRegistry()
            self._registry.load()
        except Exception:
            self._registry = None
            self._entries = []
            self._states = {}
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

    def hand_lost(self) -> None:
        for state in self._states.values():
            _reset_state(state)

    def process_landmarks(
        self,
        landmarks: np.ndarray,
        *,
        handedness: str = "",
        timestamp: Optional[float] = None,
        dispatch: bool = True,
    ) -> Optional[str]:
        """Advance sequence state from a (21, 3) landmark frame.

        Returns the fired gesture name, or None.
        """
        if not self._entries:
            return None
        now = float(timestamp if timestamp is not None else time.monotonic())
        try:
            features = normalize_landmarks(landmarks)
        except Exception:
            return None
        hand = str(handedness or "").strip()
        fired: Optional[str] = None
        for gesture, step_clfs in self._entries:
            wanted = gesture.handedness
            if wanted in ("Left", "Right") and hand in ("Left", "Right"):
                if wanted != hand:
                    continue
            state = self._states.setdefault(gesture.name, _SequenceState())
            step_i = int(state.step_index)
            if step_i < 0 or step_i >= len(step_clfs):
                _reset_state(state)
                step_i = 0
            clf = step_clfs[step_i]
            match = clf.classify_raw(features, sticky_name=clf._gestures[0].name)
            dwell_s = max(0.08, float(gesture.pose_sequence_dwell_ms) / 1000.0)
            max_hold_s = max(
                dwell_s,
                float(gesture.pose_sequence_max_hold_ms) / 1000.0,
            )
            gap_s = max(0.1, float(gesture.pose_sequence_max_gap_ms) / 1000.0)
            is_last = step_i >= len(step_clfs) - 1

            if match is not None:
                state.last_match_at = now
                if state.dwell_started_at is None:
                    state.dwell_started_at = now
                    state.dwell_met = False
                held = now - float(state.dwell_started_at)
                if held > max_hold_s:
                    # Held this pose too long — fail the sequence.
                    _reset_state(state)
                    continue
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
            else:
                # No match on current step.
                if state.dwell_met and not is_last:
                    # Released after a valid hold — advance to next pose.
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
                        # Still within gap; clear dwell so they must
                        # re-hold the current step continuously.
                        state.dwell_started_at = None
                        state.dwell_met = False
        return fired

"""Captures N takes of a dynamic gesture.

The recorder is a pure state machine — it accepts per-frame
landmark inputs from the engine's hot path and groups them into
individual takes based on the user's recording-mode preference and
start/stop signals.

Three duration modes are supported (matching the UI radio buttons
the wizard exposes):

  * FIXED_SHORT (1.5 seconds): start collecting on `begin_take()`,
    auto-stop after 1.5 s of real wall-clock time.
  * FIXED_LONG (3 seconds): same, but 3 s.
  * UNTIL_STOPPED: collect until the caller fires `end_take()`. UI
    binds this to "press Start, then press Stop / Space again".

After 10 takes (or the configured target), the recorder is "full"
and the caller can run `build_artifacts()` to project takes through
the key-point selector + assemble runtime templates.

This module deliberately knows nothing about Qt, OpenCV, threading,
or file paths. The host wizard owns those concerns; the recorder is
a testable callable that consumes frames and emits state changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from .dynamic_classifier import (
    DynamicGestureTemplate,
    build_template_from_takes,
)
from .dynamic_recording import (
    DynamicGestureTake,
    NUM_LANDMARKS,
    RESAMPLED_FRAME_COUNT,
    normalize_frame,
)
from .key_point_selector import KeyPointSelection, select_key_points


# Number of takes the wizard collects by default. The user's spec is
# explicit: "do the recording 10 times (label each one 1/10...)".
DEFAULT_TARGET_TAKES = 10


class DurationMode(str, Enum):
    """Recording duration semantics. Backing values match the UI
    radio-button data slugs so the wizard can pass strings straight
    through without a translation table."""

    FIXED_SHORT = "fixed_short"   # 1.5 s auto-stop
    FIXED_LONG = "fixed_long"     # 3.0 s auto-stop
    UNTIL_STOPPED = "until_stopped"  # caller drives end_take()

    @property
    def auto_stop_seconds(self) -> Optional[float]:
        """Wall-clock duration after which `feed_frame` will close
        the active take automatically. `None` for until-stopped mode."""
        return {
            DurationMode.FIXED_SHORT: 1.5,
            DurationMode.FIXED_LONG: 3.0,
            DurationMode.UNTIL_STOPPED: None,
        }[self]


class RecorderState(str, Enum):
    """Top-level state machine. Transitions:
        IDLE        → ACTIVE        on begin_take()
        ACTIVE      → IDLE          on end_take() or auto-stop
        IDLE        → COMPLETE      automatically when take count
                                    reaches the target
        COMPLETE    is terminal (caller resets to start over)
    """

    IDLE = "idle"           # between takes, waiting for begin_take
    ACTIVE = "active"       # collecting frames for the current take
    COMPLETE = "complete"   # target take count reached


@dataclass
class _ActiveTakeBuffer:
    """In-progress accumulator before we freeze into a DynamicGestureTake."""

    started_at: float
    landmarks: List[np.ndarray] = field(default_factory=list)
    # Per-frame absolute wrist position divided by palm scale.
    # NOT wrist-subtracted — carries the whole-hand translation signal
    # that the wrist-relative `landmarks` discards. The classifier uses
    # this as its second DTW channel.
    wrist_palm_scaled: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    handedness: Optional[str] = None


class DynamicGestureRecorder:
    """Multi-take recorder for dynamic custom gestures.

    Usage from the UI:

        recorder = DynamicGestureRecorder(DurationMode.FIXED_SHORT)
        recorder.begin_take()
        # ... pump frames from the engine via feed_frame() ...
        # for FIXED modes, feed_frame auto-stops once duration elapses.
        # for UNTIL_STOPPED, call end_take() manually.
        # Repeat 10 times.
        artifacts = recorder.build_artifacts()
        # artifacts.takes -> List[DynamicGestureTake]
        # artifacts.key_points -> KeyPointSelection
        # artifacts.template -> DynamicGestureTemplate (ready for the
        #                       classifier / registry).

    Thread safety: not safe. The wizard pumps frames from the main
    Qt thread; recorder methods should only be called from that
    thread. (The engine's worker thread emits landmarks via a Qt
    signal, so they arrive on the GUI thread for the wizard to feed
    in.)
    """

    def __init__(
        self,
        duration_mode: DurationMode = DurationMode.FIXED_SHORT,
        *,
        target_takes: int = DEFAULT_TARGET_TAKES,
        on_state_changed: Optional[Callable[["RecorderState"], None]] = None,
        on_take_completed: Optional[Callable[[int, DynamicGestureTake], None]] = None,
    ) -> None:
        self._duration_mode = DurationMode(duration_mode)
        self._target_takes = max(1, int(target_takes))
        self._on_state_changed = on_state_changed
        self._on_take_completed = on_take_completed

        self._takes: List[DynamicGestureTake] = []
        self._state: RecorderState = RecorderState.IDLE
        self._active: Optional[_ActiveTakeBuffer] = None

    # ---- public state ----

    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def duration_mode(self) -> DurationMode:
        return self._duration_mode

    @property
    def target_takes(self) -> int:
        return self._target_takes

    @property
    def completed_takes(self) -> int:
        return len(self._takes)

    @property
    def takes(self) -> List[DynamicGestureTake]:
        """Read-only view of the takes captured so far. Index 0 is
        take 1/10, index 1 is take 2/10, etc."""
        return list(self._takes)

    @property
    def active_seconds(self) -> float:
        """Seconds elapsed in the current take, or 0 if not active.
        UI binds this to a "00:0.7 / 1.5" progress label."""
        if self._active is None:
            return 0.0
        return max(0.0, time.monotonic() - self._active.started_at)

    # ---- recording control ----

    def begin_take(self) -> bool:
        """Start a new take. No-op if already active or complete.
        Returns True if a take was started, False otherwise.

        `started_at` is provisionally set from `time.monotonic()`,
        but the first `feed_frame` call REPLACES it with that frame's
        timestamp. That way the auto-stop comparison `ts - started_at`
        stays consistent whether the caller is the live engine (real
        monotonic timestamps) or a test (synthetic timestamps).
        """
        if self._state != RecorderState.IDLE:
            return False
        self._active = _ActiveTakeBuffer(started_at=time.monotonic())
        self._transition(RecorderState.ACTIVE)
        return True

    def end_take(self) -> Optional[DynamicGestureTake]:
        """Close the active take. For FIXED durations the caller may
        still invoke this to truncate early (e.g. user pressed the
        Stop button). Returns the frozen take, or None if there was
        no active recording."""
        if self._state != RecorderState.ACTIVE or self._active is None:
            return None
        take = self._freeze_active_take()
        self._active = None
        self._takes.append(take)
        if self._on_take_completed is not None:
            try:
                self._on_take_completed(len(self._takes), take)
            except Exception:
                pass
        if len(self._takes) >= self._target_takes:
            self._transition(RecorderState.COMPLETE)
        else:
            self._transition(RecorderState.IDLE)
        return take

    def discard_active(self) -> None:
        """Throw away the in-progress take without saving. Useful for
        "I sneezed during this one, retake" UX."""
        if self._state != RecorderState.ACTIVE:
            return
        self._active = None
        self._transition(RecorderState.IDLE)

    def discard_take(self, index: int) -> bool:
        """Drop a completed take by index (0-based). UI exposes this
        on the take review screen so the user can throw away a bad
        recording without scrapping the whole gesture. Returns True
        if a take was removed."""
        if not (0 <= index < len(self._takes)):
            return False
        del self._takes[index]
        # Re-evaluate state — going back below target unlocks more
        # recording. Don't auto-transition from COMPLETE if we're
        # still at or above target (e.g. user overshot).
        if self._state == RecorderState.COMPLETE and len(self._takes) < self._target_takes:
            self._transition(RecorderState.IDLE)
        return True

    def reset(self) -> None:
        """Wipe everything and return to the initial IDLE state."""
        self._takes = []
        self._active = None
        self._transition(RecorderState.IDLE)

    # ---- frame intake ----

    def feed_frame(
        self,
        landmarks: np.ndarray,
        palm_scale: float,
        *,
        timestamp: Optional[float] = None,
        handedness: Optional[str] = None,
    ) -> bool:
        """Push one (21, 3) frame from the engine into the active take.

        `palm_scale` is the live engine's `hand_reading.palm.scale`
        — used to scale-normalize the landmarks so motion is
        invariant to camera distance. `timestamp` is monotonic
        seconds; defaults to `time.monotonic()` if omitted (which is
        right for the live path but lets tests inject deterministic
        timestamps).

        Returns True if the frame was accepted, False otherwise (no
        active take, bad shape, etc.). Auto-stops the take if the
        duration-mode timer has elapsed.
        """
        if self._state != RecorderState.ACTIVE or self._active is None:
            return False
        if landmarks.shape != (NUM_LANDMARKS, 3):
            return False
        ts = float(timestamp) if timestamp is not None else time.monotonic()
        # First frame in this take pins down the time origin so the
        # auto-stop comparison below works against either monotonic
        # wall-clock (live engine) or deterministic test timestamps.
        if not self._active.landmarks:
            self._active.started_at = ts
        raw = landmarks.astype(np.float32)
        scale = max(float(palm_scale), 1e-6)
        # Capture absolute wrist BEFORE normalize_frame subtracts it.
        wrist_ps = (raw[0] / scale).astype(np.float32)
        normalized = normalize_frame(raw, scale)
        self._active.landmarks.append(normalized.astype(np.float32))
        self._active.wrist_palm_scaled.append(wrist_ps)
        self._active.timestamps.append(ts)
        if handedness and self._active.handedness is None:
            # First handedness reading wins so a momentary classifier
            # flip mid-take doesn't overwrite the established hand.
            self._active.handedness = str(handedness)

        # Auto-stop if duration mode has a wall-clock cap.
        cap = self._duration_mode.auto_stop_seconds
        if cap is not None:
            elapsed = ts - self._active.started_at
            if elapsed >= cap:
                self.end_take()
        return True

    # ---- artifact construction ----

    def build_artifacts(
        self,
        *,
        gesture_name: str = "dynamic_gesture",
    ) -> "DynamicArtifacts":
        """Run key-point selection on the captured takes and build a
        runtime template ready for the classifier / registry.

        Caller is responsible for having reached the target take
        count, though we permit shorter sets so a half-finished
        gesture can still be inspected in the wizard's preview pane.
        """
        if not self._takes:
            raise ValueError("no takes to build artifacts from")
        selection = select_key_points(self._takes)
        template = build_template_from_takes(
            gesture_name,
            self._takes,
            selection.indices,
        )
        return DynamicArtifacts(
            takes=list(self._takes),
            key_points=selection,
            template=template,
        )

    # ---- internal ----

    def _freeze_active_take(self) -> DynamicGestureTake:
        assert self._active is not None
        if not self._active.landmarks:
            # Capture started but no frames arrived (engine paused,
            # hand off-screen the entire time). Synthesize a single
            # zero-frame take so the count math holds — the key-point
            # selector will down-rank it via low motion anyway.
            zeros = np.zeros((1, NUM_LANDMARKS, 3), dtype=np.float32)
            return DynamicGestureTake(
                timestamps=np.array([self._active.started_at], dtype=np.float64),
                landmarks=zeros,
                handedness=self._active.handedness,
                raw_duration_seconds=0.0,
                wrist_palm_scaled=np.zeros((1, 3), dtype=np.float32),
            )
        landmarks = np.stack(self._active.landmarks, axis=0)
        wrist = np.stack(self._active.wrist_palm_scaled, axis=0) if self._active.wrist_palm_scaled else None
        timestamps = np.asarray(self._active.timestamps, dtype=np.float64)
        duration = float(timestamps[-1] - timestamps[0]) if timestamps.size > 1 else 0.0
        return DynamicGestureTake(
            timestamps=timestamps,
            landmarks=landmarks,
            handedness=self._active.handedness,
            raw_duration_seconds=duration,
            wrist_palm_scaled=wrist,
        )

    def _transition(self, new_state: RecorderState) -> None:
        if new_state == self._state:
            return
        self._state = new_state
        if self._on_state_changed is not None:
            try:
                self._on_state_changed(new_state)
            except Exception:
                pass


@dataclass(frozen=True)
class DynamicArtifacts:
    """Bundle returned from `build_artifacts()` — everything the
    registry / UI needs to persist a new dynamic gesture."""

    takes: List[DynamicGestureTake]
    key_points: KeyPointSelection
    template: DynamicGestureTemplate

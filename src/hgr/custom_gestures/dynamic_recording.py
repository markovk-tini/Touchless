"""Data structures for a single "take" of a custom dynamic gesture.

A `DynamicGestureTake` is one full recording of the user performing
the gesture once — the raw timestamped landmark stream captured
between the user's start and stop signals.

We store ALL 21 landmarks per frame even though most gestures only
care about a subset (just the index tip for a swipe, all 5 tips for
a fist squeeze, etc.). The `key_point_selector` module looks across
all takes at consolidation time and decides which landmarks are
actually informative, then drops the rest. Keeping the full
recording in memory during the session means the user can re-pick
key points later without re-recording, and the picked "example clip"
that lives on disk after save still has the full skeleton for replay.

Frames are normalized at sample time:
  * Translation-normalized: wrist (landmark 0) is at the origin.
  * Scale-normalized: divided by `palm_scale` so the gesture is
    invariant to how close the hand is to the camera.

Time alignment (resampling all takes to the same frame count) happens
in `key_point_selector` since that's where it's actually needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# Resampled length used by the key-point selector + matcher. All takes
# are interpolated to this length before any comparison. 32 frames
# covers gestures from ~1 to ~3 seconds cleanly at our typical 15-30
# fps capture rates, and is small enough for DTW to be fast.
RESAMPLED_FRAME_COUNT = 32

# Number of landmarks per frame. MediaPipe convention.
NUM_LANDMARKS = 21


@dataclass
class DynamicGestureTake:
    """One recording of the user performing the gesture.

    Attributes:
      timestamps:  shape (T,)   monotonic seconds since take start
      landmarks:   shape (T, 21, 3)  wrist-relative + palm-scaled coords
      handedness:  "Left" / "Right" / None (None = either-hand take)
      raw_duration_seconds:  end - start time, useful for sanity
      wrist_palm_scaled: shape (T, 3) — the ABSOLUTE wrist position per
        frame divided by palm scale. Crucially, this is NOT wrist-
        subtracted — it carries the whole-hand translation signal that
        `landmarks` (wrist-relative) discards. The classifier uses this
        as a second DTW channel so swipes match by the wrist's path and
        stationary-wrist gestures (fist squeeze) match by fingers alone,
        without a binary wrist-travel gate. Defaults to a zero (T, 3)
        for back-compat with takes constructed without wrist data — the
        classifier treats zero motion as "stationary template" and
        weights the wrist channel to ~0.
    """

    timestamps: np.ndarray
    landmarks: np.ndarray
    handedness: Optional[str] = None
    raw_duration_seconds: float = 0.0
    wrist_palm_scaled: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.landmarks.ndim != 3 or self.landmarks.shape[1:] != (NUM_LANDMARKS, 3):
            raise ValueError(
                f"landmarks must have shape (T, {NUM_LANDMARKS}, 3); got "
                f"{self.landmarks.shape}"
            )
        if self.timestamps.shape[0] != self.landmarks.shape[0]:
            raise ValueError(
                "timestamps and landmarks must have the same first dim"
            )
        t = self.landmarks.shape[0]
        if self.wrist_palm_scaled is None:
            self.wrist_palm_scaled = np.zeros((t, 3), dtype=np.float32)
        else:
            self.wrist_palm_scaled = np.asarray(self.wrist_palm_scaled, dtype=np.float32)
            if self.wrist_palm_scaled.shape != (t, 3):
                raise ValueError(
                    f"wrist_palm_scaled must have shape ({t}, 3); got "
                    f"{self.wrist_palm_scaled.shape}"
                )

    @property
    def num_frames(self) -> int:
        return int(self.landmarks.shape[0])

    def resampled(self, target_length: int = RESAMPLED_FRAME_COUNT) -> np.ndarray:
        """Return a (target_length, 21, 3) array interpolated to a fixed length.

        Linear interpolation along the time axis. Used by the key-point
        selector so every take is the same shape regardless of how fast
        the user performed the gesture.
        """
        return _resample_landmarks(self.landmarks, target_length)

    def resampled_wrist(self, target_length: int = RESAMPLED_FRAME_COUNT) -> np.ndarray:
        """Return the absolute (palm-scaled) wrist trajectory resampled
        to `target_length` frames, shape (target_length, 3). Used by the
        template builder to populate the wrist DTW channel.
        """
        return _resample_wrist(self.wrist_palm_scaled, target_length)


def palm_scale_from_landmarks(landmarks: np.ndarray) -> float:
    """Compute the engine-compatible palm scale from a (21, 3) array.

    This MUST match `hgr.gesture.analysis.hand_shape.analyze_hand_shape`'s
    `palm_scale` formula exactly, otherwise templates recorded outside
    the engine (e.g. from the custom-gesture recorder dialog) will be
    in a different normalized space than the live frames the runtime
    classifier sees, and DTW will never match.

    Formula: average of (index_mcp ↔ pinky_mcp) and (wrist ↔ middle_mcp)
    distances — the same two distances `analyze_hand_shape` uses.
    """
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]
    palm_width = float(np.linalg.norm(index_mcp - pinky_mcp))
    palm_height = float(np.linalg.norm(middle_mcp - wrist))
    palm_width = max(palm_width, 1e-6)
    palm_height = max(palm_height, 1e-6)
    return max((palm_width + palm_height) * 0.5, 1e-6)


def normalize_frame(landmarks: np.ndarray, palm_scale: float) -> np.ndarray:
    """Translate to wrist origin + scale by palm size.

    `landmarks` shape (21, 3). `palm_scale` is the live engine's
    `hand_reading.palm.scale` value — callers OUTSIDE the engine
    should use `palm_scale_from_landmarks(landmarks)` to compute a
    matching scalar.
    """
    wrist = landmarks[0]
    out = (landmarks - wrist[np.newaxis, :]) / max(float(palm_scale), 1e-6)
    return out


def _resample_wrist(wrist: np.ndarray, target_length: int) -> np.ndarray:
    """Linear-interpolate a (T, 3) wrist trajectory to (target_length, 3).
    Mirror of `_resample_landmarks` for the 2-axis wrist channel."""
    if wrist.ndim != 2 or wrist.shape[1] != 3:
        raise ValueError(f"expected (T, 3); got shape {wrist.shape}")
    src_count = wrist.shape[0]
    if src_count == target_length:
        return wrist.astype(np.float32, copy=True)
    if src_count < 2:
        return np.repeat(
            wrist.astype(np.float32, copy=True)[:1] if src_count else np.zeros((1, 3), dtype=np.float32),
            target_length,
            axis=0,
        )
    src_indices = np.linspace(0.0, 1.0, src_count, dtype=np.float64)
    dst_indices = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    out = np.empty((target_length, 3), dtype=np.float32)
    for ch in range(3):
        out[:, ch] = np.interp(dst_indices, src_indices, wrist[:, ch].astype(np.float32))
    return out


def _resample_landmarks(landmarks: np.ndarray, target_length: int) -> np.ndarray:
    """Linear-interpolate a (T, 21, 3) array to (target_length, 21, 3)."""
    if landmarks.ndim != 3:
        raise ValueError(f"expected (T, 21, 3); got shape {landmarks.shape}")
    src_count = landmarks.shape[0]
    if src_count == target_length:
        return landmarks.astype(np.float32, copy=True)
    if src_count < 2:
        # Single-frame take: just repeat. The key-point selector
        # will flag this as "no motion" via its motion scores.
        return np.repeat(
            landmarks.astype(np.float32, copy=True)[:1],
            target_length,
            axis=0,
        )
    src_indices = np.linspace(0.0, 1.0, src_count, dtype=np.float64)
    dst_indices = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    out = np.empty((target_length, *landmarks.shape[1:]), dtype=np.float32)
    flat_src = landmarks.reshape(src_count, -1).astype(np.float32)
    flat_dst = np.empty((target_length, flat_src.shape[1]), dtype=np.float32)
    for ch in range(flat_src.shape[1]):
        flat_dst[:, ch] = np.interp(dst_indices, src_indices, flat_src[:, ch])
    out[:] = flat_dst.reshape(target_length, *landmarks.shape[1:])
    return out

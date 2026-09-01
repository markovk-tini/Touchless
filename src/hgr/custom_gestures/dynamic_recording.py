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

    def resampled_timestamps(self, target_length: int = RESAMPLED_FRAME_COUNT) -> np.ndarray:
        """Return the timestamps linearly resampled to target_length.
        v1.1.8.2 SPRING architecture: features consume dt-normalized
        velocities that only make sense with time information carried
        alongside the resampled trajectories."""
        src = np.asarray(self.timestamps, dtype=np.float64)
        if src.size == target_length:
            return src.astype(np.float64, copy=True)
        if src.size < 2:
            base = float(src[0]) if src.size else 0.0
            return np.linspace(base, base + 1.0, target_length, dtype=np.float64)
        src_indices = np.linspace(0.0, 1.0, src.size, dtype=np.float64)
        dst_indices = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
        return np.interp(dst_indices, src_indices, src).astype(np.float64)

    def trim_to_motion(self) -> "DynamicGestureTake":
        """v1.1.8.2 step 3: trim leading + trailing frames whose per-
        frame max-landmark step is below half the take's own mid-frame
        step. Removes click-timing slop (~200-500 ms of static hand at
        the beginning and end of every recorder session) so the 3 takes
        of "wave up" all align on their own motion peaks before we
        resample to 32 frames.

        Returns a NEW take; does not mutate self. Falls back to self
        (unchanged) when the take is too short or the trim would remove
        more than 60% of the frames (defensive — better a slightly
        loose template than an over-trimmed one).
        """
        if self.landmarks.shape[0] < 4:
            return self
        try:
            steps = _per_frame_max_landmark_step(self.landmarks)
        except Exception:
            return self
        if steps.size == 0:
            return self
        # Reference step: mean of the middle 50% of frames (robust to
        # outliers at either end).
        mid_lo = steps.size // 4
        mid_hi = steps.size - mid_lo
        if mid_hi - mid_lo < 2:
            mid_lo, mid_hi = 0, steps.size
        mid_slice = steps[mid_lo:mid_hi]
        # Robust reference: use the 60th percentile of the mid-take
        # slice rather than the mean so a slow deliberate motion still
        # gets a reasonable threshold. Guarded so a zero-motion take
        # doesn't produce a nonsensical trim.
        ref = float(np.percentile(mid_slice, 60))
        if ref <= 1e-6:
            return self
        threshold = 0.5 * ref
        # Leading trim: first frame where step >= threshold.
        first = 0
        for i in range(steps.size):
            if steps[i] >= threshold:
                first = max(0, i - 1)  # keep 1 frame of lead-in
                break
        # Trailing trim: last frame where step >= threshold.
        last = steps.size - 1
        for j in range(steps.size - 1, -1, -1):
            if steps[j] >= threshold:
                last = min(steps.size - 1, j + 1)  # keep 1 frame of tail
                break
        if last <= first:
            return self
        trimmed_count = (last - first + 1)
        # Defensive: refuse to keep less than 40% of the original take.
        if trimmed_count < max(4, int(0.4 * steps.size)):
            return self
        sl = slice(first, last + 1)
        return DynamicGestureTake(
            timestamps=self.timestamps[sl].copy(),
            landmarks=self.landmarks[sl].copy(),
            handedness=self.handedness,
            raw_duration_seconds=float(
                self.timestamps[last] - self.timestamps[first]
            ),
            wrist_palm_scaled=(
                None if self.wrist_palm_scaled is None
                else self.wrist_palm_scaled[sl].copy()
            ),
        )


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


# ─────────────────────────────────────────────────────────────────────────
# v1.1.8.2 SPRING-architecture feature builder (steps 1 + 2).
#
# The classifier switched from batch DTW at segment close to streaming
# SPRING that scores every frame. SPRING needs a feature vector that
# (a) has sharp local minima at motion peaks (not a smooth ramp),
# (b) is direction-signed so up ≠ down, and
# (c) fits in the palm's own coordinate frame so the same "wave up"
#     gesture looks identical whether the user's hand is centered,
#     off-axis, or slightly tilted.
#
# palm_frame_basis  → (3,3) rotation per frame: right / up / forward.
# build_dynamic_features → per-frame feature vector combining
#   key-point positions (unchanged) + velocities projected into the
#   palm frame + direction-signed wrist motion scalars.


def palm_frame_basis(landmarks: np.ndarray) -> np.ndarray:
    """Return a (T, 3, 3) rotation matrix, one per frame. Rows are
    (right, up, forward) unit vectors in the world frame. Multiplying
    a world-frame vector by this matrix yields the vector in the palm
    frame. Falls back to identity when the palm is nearly edge-on
    (|cross| < 1e-4) so a degenerate side-on view degrades gracefully.

    forward = norm(L9 - L0)   (wrist → middle_mcp)
    right   = norm(L5 - L17)  (index_mcp → pinky_mcp)  — approx orthogonal
    up      = norm(forward × right)  — recomputed to enforce orthogonality
    right   = norm(up × forward)     — final basis is orthonormal

    Works on both (T, 21, 3) and (21, 3) input; returns (T, 3, 3) or
    (3, 3) accordingly.
    """
    arr = np.asarray(landmarks, dtype=np.float32)
    single_frame = arr.ndim == 2
    if single_frame:
        arr = arr[np.newaxis, ...]
    T = arr.shape[0]
    wrist = arr[:, 0, :]
    mid_mcp = arr[:, 9, :]
    index_mcp = arr[:, 5, :]
    pinky_mcp = arr[:, 17, :]
    forward = mid_mcp - wrist
    approx_right = index_mcp - pinky_mcp
    # Normalize forward
    fnorm = np.linalg.norm(forward, axis=1, keepdims=True)
    fnorm = np.maximum(fnorm, 1e-6)
    forward = forward / fnorm
    # up = forward × approx_right, normalized
    up = np.cross(forward, approx_right)
    unorm = np.linalg.norm(up, axis=1, keepdims=True)
    degenerate = (unorm < 1e-4).squeeze(-1)
    unorm = np.maximum(unorm, 1e-6)
    up = up / unorm
    # right = up × forward — guaranteed orthonormal now
    right = np.cross(up, forward)
    rnorm = np.linalg.norm(right, axis=1, keepdims=True)
    rnorm = np.maximum(rnorm, 1e-6)
    right = right / rnorm
    R = np.stack([right, up, forward], axis=1).astype(np.float32)  # (T, 3, 3)
    if np.any(degenerate):
        R[degenerate] = np.eye(3, dtype=np.float32)
    if single_frame:
        return R[0]
    return R


def _per_frame_max_landmark_step(landmarks: np.ndarray) -> np.ndarray:
    """Return a (T,) array where element i is the max L2 landmark step
    between frame i-1 and frame i. Element 0 is 0.0 by convention.
    Wrist-relative space so translation is invariant."""
    if landmarks.shape[0] < 2:
        return np.zeros(landmarks.shape[0], dtype=np.float32)
    diff = np.diff(landmarks.astype(np.float32), axis=0)  # (T-1, 21, 3)
    step_norms = np.linalg.norm(diff, axis=-1).max(axis=-1)  # (T-1,)
    out = np.zeros(landmarks.shape[0], dtype=np.float32)
    out[1:] = step_norms
    return out


# Feature layout produced by build_dynamic_features(indices), for a
# frame count T:
#   [0 : len(indices) * 3]                 wrist-relative keypoint xyz
#   [K:K+3]  K = above end                  palm-frame wrist velocity (r, u, f)
#   [K+3:K+18]                              palm-frame fingertip velocities
#                                           (5 tips × (r, u, f))
# Total F = 3*len(indices) + 3 + 15 = 3*(len(indices) + 6)
_FINGERTIP_INDICES = (4, 8, 12, 16, 20)


def dynamic_feature_dim(num_key_indices: int) -> int:
    """Deterministic feature width. Keeps schema consistent between
    template build time and live feature construction."""
    return 3 * (num_key_indices + 6)


def build_dynamic_features(
    landmarks: np.ndarray,
    wrist_palm_scaled: np.ndarray,
    timestamps: np.ndarray,
    key_indices: List[int],
) -> np.ndarray:
    """Build a (T, F) feature matrix for SPRING matching.

    `landmarks` is (T, 21, 3) wrist-relative palm-scaled.
    `wrist_palm_scaled` is (T, 3) absolute wrist in palm units.
    `timestamps` is (T,) monotonic seconds.
    `key_indices` picks which of the 21 landmarks contribute their xyz.

    The wrist velocity + fingertip velocities are computed in world
    frame and then rotated into the palm frame at the same timestep so
    "up" always means "toward the fingertips" regardless of hand tilt.
    """
    lm = np.asarray(landmarks, dtype=np.float32)
    ws = np.asarray(wrist_palm_scaled, dtype=np.float32)
    ts = np.asarray(timestamps, dtype=np.float64)
    T = lm.shape[0]
    kp = list(int(i) for i in key_indices)
    F = dynamic_feature_dim(len(kp))
    out = np.zeros((T, F), dtype=np.float32)
    if T == 0:
        return out
    # (a) wrist-relative keypoint positions (already normalized).
    out[:, : 3 * len(kp)] = lm[:, kp, :].reshape(T, -1)
    if T < 2:
        return out
    R = palm_frame_basis(lm)  # (T, 3, 3)
    # dt per frame; first frame gets a synthetic dt so velocity=0.
    dt = np.empty(T, dtype=np.float32)
    dt[0] = 1.0
    dt[1:] = np.clip(ts[1:] - ts[:-1], 1e-3, 0.5).astype(np.float32)
    # (b) wrist velocity in world → palm frame.
    wv = np.zeros_like(ws)
    wv[1:] = (ws[1:] - ws[:-1]) / dt[1:, np.newaxis]
    # Rotate wrist velocity into palm frame at frame t.
    wv_palm = np.einsum("tij,tj->ti", R, wv)  # (T, 3)
    K = 3 * len(kp)
    out[:, K : K + 3] = wv_palm
    # (c) fingertip velocities in world → palm frame.
    tip_positions = lm[:, list(_FINGERTIP_INDICES), :]  # (T, 5, 3)
    tv = np.zeros_like(tip_positions)
    tv[1:] = (tip_positions[1:] - tip_positions[:-1]) / dt[1:, np.newaxis, np.newaxis]
    # Rotate each tip's velocity by R at frame t.
    tv_palm = np.einsum("tij,tkj->tki", R, tv)  # (T, 5, 3)
    out[:, K + 3 : K + 3 + 15] = tv_palm.reshape(T, -1)
    return out.astype(np.float32)


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

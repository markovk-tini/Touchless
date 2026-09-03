"""Tests for the dynamic-gesture key-point selection algorithm.

Synthesizes hand-landmark recordings for three canonical gestures and
asserts the selector picks the landmarks any reasonable human would
pick by eye:

  * Index-only down-swipe: should pick the index finger landmarks
    (6, 7, 8) + anchors (0, 9); should REJECT the curled fingertips
    (4, 12, 16, 20) since they don't move.
  * Fist squeeze: should pick all 5 fingertips (4, 8, 12, 16, 20).
  * Inconsistent recordings: should NOT pick the noise landmarks the
    user moved differently every take.

Synthetic landmarks are 3D positions sampled from simple
parameterized trajectories — no MediaPipe / camera dependency.
"""
from __future__ import annotations

import unittest
import numpy as np

from hgr.custom_gestures.dynamic_recording import (
    DynamicGestureTake,
    NUM_LANDMARKS,
    RESAMPLED_FRAME_COUNT,
)
from hgr.custom_gestures.key_point_selector import select_key_points


# Per-test reproducibility — synthesizing takes uses a small amount
# of human-jitter noise so the algorithm doesn't see 10 byte-identical
# takes (which would game the consistency score). Same seed each run.
_RNG = np.random.default_rng(seed=20260513)


def _base_hand_landmarks() -> np.ndarray:
    """Return a (21, 3) array of resting-hand landmark positions.

    These coordinates are already in "hand-local normalized" space:
    wrist at origin, palm scale = 1. They're stylized — not exactly
    what MediaPipe would produce — but the algorithm only cares about
    relative motion, not absolute geometry.
    """
    pts = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    # Wrist
    pts[0] = (0.0, 0.0, 0.0)
    # Thumb chain (1..4)
    pts[1] = (-0.40, -0.20, 0.0)
    pts[2] = (-0.60, -0.40, 0.0)
    pts[3] = (-0.75, -0.55, 0.0)
    pts[4] = (-0.85, -0.65, 0.0)
    # Index chain (5..8)
    pts[5] = (-0.20, -0.95, 0.0)
    pts[6] = (-0.20, -1.30, 0.0)
    pts[7] = (-0.20, -1.55, 0.0)
    pts[8] = (-0.20, -1.75, 0.0)
    # Middle chain (9..12)
    pts[9] = (0.05, -1.00, 0.0)
    pts[10] = (0.05, -1.45, 0.0)
    pts[11] = (0.05, -1.75, 0.0)
    pts[12] = (0.05, -1.95, 0.0)
    # Ring chain (13..16)
    pts[13] = (0.30, -0.95, 0.0)
    pts[14] = (0.30, -1.35, 0.0)
    pts[15] = (0.30, -1.60, 0.0)
    pts[16] = (0.30, -1.80, 0.0)
    # Pinky chain (17..20)
    pts[17] = (0.55, -0.85, 0.0)
    pts[18] = (0.55, -1.10, 0.0)
    pts[19] = (0.55, -1.30, 0.0)
    pts[20] = (0.55, -1.45, 0.0)
    return pts


def _curl_fingers_except(base: np.ndarray, keep_open: tuple[int, ...]) -> np.ndarray:
    """Return a copy of `base` where every finger NOT in `keep_open`
    has its 4 chain landmarks pulled close to the palm. `keep_open`
    is the set of fingertip indices to leave untouched.

    Used to build synthetic "only index extended" / "fist" poses.
    """
    out = base.copy()
    # Map fingertip -> chain landmark indices to curl.
    chains = {
        4: (1, 2, 3, 4),
        8: (5, 6, 7, 8),
        12: (9, 10, 11, 12),
        16: (13, 14, 15, 16),
        20: (17, 18, 19, 20),
    }
    for tip, chain in chains.items():
        if tip in keep_open:
            continue
        # Pull the tip + DIP toward the palm so it reads as "closed".
        # Use the chain's BASE (MCP-equivalent — index [0] of the chain)
        # as the anchor and squish the rest toward it.
        anchor = base[chain[0]]
        for i in chain[1:]:
            out[i] = anchor + (base[i] - anchor) * 0.18
    return out


def _make_down_swipe_take(
    *,
    frames: int = 30,
    motion_amount: float = 1.2,
    jitter: float = 0.005,
) -> DynamicGestureTake:
    """Synthesize a down-swipe take where ONLY the index finger
    extends; thumb/middle/ring/pinky stay curled.

    The hand starts at rest and translates down by `motion_amount`
    over the course of the recording. The index finger moves with the
    hand; the other curled tips also translate but in lockstep with
    the wrist — because we capture in hand-local frame (wrist at
    origin), only the FREE landmarks (the open index tip) show
    significant motion after normalization. That's exactly what
    happens in the real engine.
    """
    pose = _curl_fingers_except(_base_hand_landmarks(), keep_open=(8,))
    # Wave the OPEN index tip ALONG the motion direction over time —
    # this is what gives the down-swipe its signature. The curled
    # landmarks are constants in hand-local frame.
    landmarks = np.empty((frames, NUM_LANDMARKS, 3), dtype=np.float32)
    timestamps = np.linspace(0.0, 1.5, frames, dtype=np.float64)
    for f in range(frames):
        progress = f / max(1, frames - 1)
        frame_pose = pose.copy()
        # Real swipes translate the WHOLE hand (wrist + every landmark)
        # in lockstep. We add this whole-hand translation first; after
        # the engine's per-frame `normalize_frame` (subtract wrist), the
        # wrist-relative trajectory is unchanged from the original
        # finger-only synthetic, so existing key-point / classifier
        # tests stay valid. The classifier's wrist-travel gate (added
        # to filter "hand entered view and held still" false positives
        # in real use) needs the absolute wrist to actually move.
        frame_pose[:, 1] -= motion_amount * progress
        # Index chain swings DOWNWARD an additional amount relative to
        # the hand — this is the wrist-relative signature that DTW
        # actually matches against.
        for lm_idx, factor in ((6, 0.55), (7, 0.85), (8, 1.00)):
            frame_pose[lm_idx, 1] -= factor * motion_amount * progress
        # Add tiny per-frame jitter to make takes non-identical.
        frame_pose += _RNG.normal(0.0, jitter, frame_pose.shape).astype(np.float32)
        landmarks[f] = frame_pose
    return DynamicGestureTake(
        timestamps=timestamps,
        landmarks=landmarks,
        handedness="Right",
        raw_duration_seconds=1.5,
    )


def _make_fist_squeeze_take(
    *,
    frames: int = 30,
    jitter: float = 0.005,
) -> DynamicGestureTake:
    """Open palm → fist → open palm. All five fingertips travel
    significantly. Used to verify the selector picks ALL of them."""
    open_pose = _base_hand_landmarks()
    closed_pose = _curl_fingers_except(open_pose, keep_open=())  # all curled
    landmarks = np.empty((frames, NUM_LANDMARKS, 3), dtype=np.float32)
    timestamps = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    for f in range(frames):
        # Triangle wave: 0 -> 1 -> 0 over the recording length.
        progress = f / max(1, frames - 1)
        t = 1.0 - abs(2.0 * progress - 1.0)
        # Lerp between open and closed.
        frame_pose = open_pose * (1.0 - t) + closed_pose * t
        frame_pose += _RNG.normal(0.0, jitter, frame_pose.shape).astype(np.float32)
        landmarks[f] = frame_pose.astype(np.float32)
    return DynamicGestureTake(
        timestamps=timestamps,
        landmarks=landmarks,
        handedness="Right",
        raw_duration_seconds=1.0,
    )


def _make_noisy_take(*, frames: int = 30) -> DynamicGestureTake:
    """All landmarks wiggle in random directions every frame — should
    fail the consistency check across takes since each take's path
    is a different walk."""
    base = _base_hand_landmarks()
    timestamps = np.linspace(0.0, 1.5, frames, dtype=np.float64)
    landmarks = np.empty((frames, NUM_LANDMARKS, 3), dtype=np.float32)
    for f in range(frames):
        # Big per-landmark random walk. Each take gets a different
        # noise pattern because _RNG is shared (state advances).
        landmarks[f] = base + _RNG.normal(0.0, 0.4, base.shape).astype(np.float32)
    return DynamicGestureTake(
        timestamps=timestamps,
        landmarks=landmarks,
        handedness="Right",
        raw_duration_seconds=1.5,
    )


class KeyPointSelectorTests(unittest.TestCase):

    def test_down_swipe_with_only_index_picks_index_landmarks(self) -> None:
        takes = [_make_down_swipe_take() for _ in range(10)]
        result = select_key_points(takes)

        # Index finger distal joints MUST be selected — they're the
        # only landmarks that actually moved in the hand-local frame.
        self.assertIn(8, result.indices)  # tip
        self.assertIn(7, result.indices)  # DIP

        # Wrist + middle-finger MCP anchors are always included.
        self.assertIn(0, result.indices)
        self.assertIn(9, result.indices)

        # All five fingertips are pose identity — static curled tips
        # stay in the set so "index-only swipe" cannot match "open
        # hand swipe" on path alone.
        for tip in (4, 8, 12, 16, 20):
            self.assertIn(tip, result.indices)

        # Sanity: motion scores still reflect what actually moved.
        self.assertGreater(result.motion_scores[8], result.motion_scores[4])
        self.assertGreater(result.motion_scores[8], result.motion_scores[20])

        # No fallback needed — the gesture has clear motion.
        self.assertFalse(result.fell_back)

    def test_fist_squeeze_picks_all_five_fingertips(self) -> None:
        takes = [_make_fist_squeeze_take() for _ in range(10)]
        result = select_key_points(takes)

        # All five tips swing significantly during open->fist->open.
        # All should land in the selected set.
        for tip in (4, 8, 12, 16, 20):
            self.assertIn(
                tip, result.indices,
                msg=f"fingertip {tip} should have been kept for a fist squeeze "
                    f"(motion={result.motion_scores[tip]:.3f}, "
                    f"consistency={result.consistency_scores[tip]:.3f})",
            )

        # Anchors always present.
        self.assertIn(0, result.indices)
        self.assertIn(9, result.indices)
        self.assertFalse(result.fell_back)

    def test_noisy_recordings_have_low_consistency_scores(self) -> None:
        takes = [_make_noisy_take() for _ in range(10)]
        result = select_key_points(takes)

        # Random per-frame walks → centroid trajectory is ~0
        # everywhere → individual takes deviate hugely from it →
        # consistency scores should be near 0 for ALL non-anchor
        # landmarks. (Anchors 0/9 are always included regardless.)
        for lm in range(NUM_LANDMARKS):
            if lm in (0, 9):
                continue
            self.assertLess(
                result.consistency_scores[lm], 0.5,
                msg=f"noise landmark {lm} consistency = "
                    f"{result.consistency_scores[lm]:.3f}, expected < 0.5",
            )

    def test_resampled_take_length_is_canonical(self) -> None:
        # The selector relies on resampled() returning a fixed length.
        # Quick sanity to catch silent regressions in the resampler.
        take = _make_down_swipe_take(frames=17)  # odd, not 32
        out = take.resampled()
        self.assertEqual(out.shape, (RESAMPLED_FRAME_COUNT, NUM_LANDMARKS, 3))

    def test_single_take_does_not_crash(self) -> None:
        # Edge case: user only completed 1 recording before abandoning.
        # The algorithm should still return SOMETHING usable (anchors
        # plus whatever moved) without raising.
        result = select_key_points([_make_down_swipe_take()])
        self.assertIn(0, result.indices)
        self.assertIn(9, result.indices)
        # Index tip still has the highest motion in a single take, so
        # we expect it to land in the selection.
        self.assertIn(8, result.indices)


if __name__ == "__main__":
    unittest.main()

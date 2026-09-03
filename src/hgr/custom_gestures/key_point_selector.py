"""Decide which hand landmarks actually matter for a dynamic gesture.

Given N takes of the user performing the same gesture, this module
inspects every landmark's trajectory and picks the subset that is:

  (1) MOVING during the gesture (i.e. contributes path information), and
  (2) MOVING THE SAME WAY across all N takes (i.e. it's the gesture
      and not random hand wobble that happens to be high-motion in
      this one recording).

All five fingertips are ALWAYS kept even when they do not move. A
curled pinky in an index-only swipe is not "noise" — it is the pose
that distinguishes that gesture from an open-hand swipe on the same
path. Intermediate joints that fail (1) are still dropped.
A landmark that fails (2) — say the thumb sliding in random
directions every time — is dropped because it's noise that would hurt
the runtime matcher's accuracy.

The output is a small list of landmark indices (typically 3-8 from
the 21 MediaPipe landmarks) that the dynamic-gesture classifier will
use to build its template + match incoming frames at runtime. Anchor
landmarks (wrist, middle-finger MCP) are ALWAYS included so the
matcher has a stable reference frame even when the user's whole hand
translates across the camera.

Algorithm at a glance:
  for each landmark i in 0..20:
    motion_score[i]     = median across takes of the trajectory's
                          frame-to-frame path length (after the wrist
                          + scale normalization already done at
                          capture time).
    consistency_score[i] = inverse of the inter-take dispersion of
                          landmark i's resampled trajectory around
                          the cross-take centroid.
    relevance[i]        = motion_score[i] * consistency_score[i]
  pick anchors + top landmarks where relevance exceeds an adaptive
  threshold.

Tunables live as module-level constants so they're easy to find when
real-world recordings turn up surprises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .dynamic_recording import (
    DynamicGestureTake,
    NUM_LANDMARKS,
    RESAMPLED_FRAME_COUNT,
)


# Anchors always included regardless of motion/consistency. Wrist (0)
# is the hand-local origin after normalization (so it's always at 0,
# but we keep it for clarity in the saved schema). Middle-finger MCP
# (9) is the standard "scale reference" point — including it lets the
# runtime matcher detect whole-hand-tilt motions that no other
# landmark alone would catch.
_ANCHOR_LANDMARKS: Sequence[int] = (0, 9)

# Fingertip landmark indices, in MediaPipe order. The selection step
# fills these in BEFORE intermediate joints (PIP / DIP) so a gesture
# that exercises all 5 fingers doesn't accidentally drop a fingertip
# because PIPs of other fingers ranked higher numerically.
_FINGERTIP_INDICES: Sequence[int] = (4, 8, 12, 16, 20)

# A landmark with normalized path-length below this is "not moving"
# during the gesture. Tuned against synthetic recordings: a closed
# finger from a fist-squeeze sees ~0.03; an active index-tip swipe
# sees ~1.2-2.0. Floor of 0.15 cleanly separates the two.
_MOTION_FLOOR = 0.15

# Consistency = (centroid trajectory's path length) / (per-take median
# path length). For a consistent gesture both numbers track each
# other, so the ratio is near 1.0. For random walks the centroid
# averages toward zero motion while each take is a long noisy path,
# so the ratio plummets. Landmarks below this threshold are dropped.
_CONSISTENCY_FLOOR = 0.55

# When NO landmarks pass both thresholds (rare — e.g. the user
# recorded 10 mostly-static frames), fall back to keeping the top-K
# by raw relevance so the gesture still has SOMETHING to match on.
_FALLBACK_TOP_K = 3

# Identity set (always kept): 2 anchors + 5 fingertips = 7. Cap is
# higher so a one-finger swipe can still add that finger's DIP/PIP
# for path shape without evicting a static (but identifying) tip.
#   * 1-finger swipe: 7 identity + 2-3 active-finger joints.
#   * 5-finger fist/clap/spread: 7 identity, extra joints optional.
_MAX_SELECTED = 12


@dataclass(frozen=True)
class KeyPointSelection:
    """Result of running the algorithm on a stack of takes."""

    indices: List[int]
    motion_scores: np.ndarray         # shape (21,)
    consistency_scores: np.ndarray    # shape (21,)
    relevance: np.ndarray             # shape (21,)
    # Diagnostics so the wizard's "see why these were picked" panel
    # can show the user which landmarks won and why.
    fell_back: bool = False
    centroid_trajectory: np.ndarray = None  # type: ignore[assignment]  # shape (resampled, 21, 3)


def select_key_points(
    takes: Sequence[DynamicGestureTake],
    *,
    resampled_length: int = RESAMPLED_FRAME_COUNT,
    motion_floor: float = _MOTION_FLOOR,
    consistency_floor: float = _CONSISTENCY_FLOOR,
    max_selected: int = _MAX_SELECTED,
    anchor_indices: Sequence[int] = _ANCHOR_LANDMARKS,
    fingertip_indices: Sequence[int] = _FINGERTIP_INDICES,
) -> KeyPointSelection:
    """Return the list of landmark indices that matter for the gesture.

    `takes` should be ≥ 2 recordings of the same gesture. With 10
    takes (the UI default), the consistency score gets enough signal
    to reliably reject noise.

    Selection order (so the cap drops sensibly):
      1. Anchors (always) + all five fingertips (always — pose identity).
      2. Intermediate joints (PIP / DIP / MCP / proximal) that pass.

    Result indices are sorted ascending so the gesture's saved
    trajectory has a stable column order across saves.
    """
    if len(takes) == 0:
        raise ValueError("need at least one take to pick key points")

    # 1. Stack resampled trajectories: shape (N, L, 21, 3).
    # Subtract per-frame wrist so the per-landmark motion scores
    # below reflect WRIST-RELATIVE motion — the same space the
    # classifier matches against. Recorder-produced takes already
    # have wrist at origin (normalize_frame ran per-frame), so this
    # is a no-op for them; raw / synthetic takes that include
    # whole-hand translation get reduced to their relative motion.
    resampled = np.stack(
        [t.resampled(resampled_length) for t in takes],
        axis=0,
    ).astype(np.float64)
    resampled = resampled - resampled[..., 0:1, :]
    n_takes, frame_count, n_landmarks, n_coords = resampled.shape
    if n_landmarks != NUM_LANDMARKS:
        raise ValueError(f"expected {NUM_LANDMARKS} landmarks; got {n_landmarks}")

    # 2. Per-landmark motion: take-wise path length, median across takes.
    motion_per_take = np.zeros((n_takes, n_landmarks), dtype=np.float64)
    for take_idx in range(n_takes):
        diffs = resampled[take_idx, 1:] - resampled[take_idx, :-1]
        step_norms = np.linalg.norm(diffs, axis=-1)  # (L-1, 21)
        motion_per_take[take_idx] = step_norms.sum(axis=0)
    motion_scores = np.median(motion_per_take, axis=0)  # (21,)

    # 3. Cross-take centroid trajectory.
    centroid = resampled.mean(axis=0)  # (L, 21, 3)
    centroid_path = np.linalg.norm(
        centroid[1:] - centroid[:-1], axis=-1
    ).sum(axis=0)  # (21,)

    # 4. Consistency = ratio of centroid path length to per-take
    # path length. Bounded [0, ~1].
    #   * Consistent gesture: every take traces nearly the same path,
    #     so centroid_path tracks motion_scores closely → ratio ≈ 1.
    #   * Random walks: each take has a long noisy path but takes
    #     don't agree on direction, so the centroid averages out
    #     toward zero motion → ratio is small.
    # Add a small epsilon so near-zero motion divides cleanly.
    consistency_scores = np.clip(
        centroid_path / (motion_scores + 1e-4),
        0.0,
        1.0,
    )

    # 5. Combined relevance.
    relevance = motion_scores * consistency_scores

    # 6. Pick.
    anchors_set = set(int(i) for i in anchor_indices)
    fingertip_set = set(int(i) for i in fingertip_indices)
    # Pose identity: static fingertips are HOW the user holds the
    # hand, not matcher-noise. Always keep them.
    selected: list[int] = sorted(anchors_set | fingertip_set)

    def _passes(i: int) -> bool:
        return (
            motion_scores[i] >= motion_floor
            and consistency_scores[i] >= consistency_floor
        )

    # Extra moving joints (beyond identity) by relevance.
    other_candidates = sorted(
        (
            i for i in range(n_landmarks)
            if i not in selected and i not in fingertip_set and _passes(i)
        ),
        key=lambda i: -relevance[i],
    )
    for idx in other_candidates:
        if len(selected) >= max_selected:
            break
        selected.append(idx)

    fell_back = False
    if len(selected) < 2:
        # Degenerate: identity set failed to populate. Fall back to
        # top-K by raw relevance so the gesture still has something
        # to match. UI should surface this as a warning.
        fell_back = True
        ranked = sorted(
            (i for i in range(n_landmarks) if i not in selected),
            key=lambda i: -relevance[i],
        )
        for idx in ranked[:_FALLBACK_TOP_K]:
            if len(selected) >= max_selected:
                break
            selected.append(idx)

    final = sorted(set(selected))
    return KeyPointSelection(
        indices=final,
        motion_scores=motion_scores.astype(np.float32),
        consistency_scores=consistency_scores.astype(np.float32),
        relevance=relevance.astype(np.float32),
        fell_back=fell_back,
        centroid_trajectory=centroid.astype(np.float32),
    )

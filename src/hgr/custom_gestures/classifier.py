from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .recorder import normalize_landmarks
from .registry import CustomGesture, GestureRegistry


# Score curve: tuned for the v5 106-dim feature vector. Higher
# _SCORE_ZERO_DISTANCE keeps typical same-pose distances mapping to
# scores that comfortably clear the default 0.88 threshold. Bumped
# from 7.0 (v4) to 8.5 to absorb the extra distance contributed by
# the new direction + thumb-index-spread + palm-normal regions.
_SCORE_ZERO_DISTANCE = 8.5

_LANDMARK_FEATURE_LEN = 63
_SPACING_FEATURE_LEN = 3
_EXTENSION_FEATURE_LEN = 5
_JOINT_ANGLE_FEATURE_LEN = 10
_CURL_CLASS_FEATURE_LEN = 5
_SPREAD_CLASS_FEATURE_LEN = 1
# v5 additions:
_DIRECTION_FEATURE_LEN = 15
_THUMB_INDEX_SPREAD_FEATURE_LEN = 1
_PALM_NORMAL_FEATURE_LEN = 3

# Spacing + extension features have small magnitudes (~0.1-4 range), so
# they need amplification to compete with the 63-dim landmark portion.
# Thumb-index spread is also a normalized distance and uses the same
# weight.
_DISTANCE_FEATURE_WEIGHT = 4.0

# Joint angles are in radians (0..π), inherently larger magnitude. Kept
# at 1.0 — they still discriminate strongly between distinct poses
# (1-2 rad gaps between extended and curled) but small natural angle
# wiggles (~0.05-0.2 rad per joint) don't punish same-pose recognition.
_JOINT_ANGLE_WEIGHT = 1.0

# Categorical class features are integer ordinals (0..4 curl, 0..3
# spread). Same pose held steady → zero variance (the bucket boundaries
# are far from any natural noise floor for clearly-in-class poses).
# Different poses → 1-4 unit gaps. Weight 2.0 makes between-class
# differences contribute meaningfully without punishing rare
# boundary-jitter cases.
_CLASS_FEATURE_WEIGHT = 2.0

# Per-finger direction unit-vectors (15 dims, 5 fingers × 3). Each
# 3-dim sub-vector has magnitude 1 by construction, so per-finger
# distance maxes at 2.0 between opposite directions. We boost above
# unit weight so direction differences (e.g. thumbs-up vs thumbs-side)
# pull scores apart meaningfully, but keep below the distance weight
# so a small finger curl doesn't get swamped by direction wiggle.
_DIRECTION_FEATURE_WEIGHT = 1.5

# Palm-normal unit vector (3 dims). Same unit-magnitude character as
# the direction features; weighted similarly so palm-toward-camera
# vs palm-away poses are pulled apart but don't dominate the score.
_PALM_NORMAL_FEATURE_WEIGHT = 1.5

# Region offsets used during reload/match.
_DISTANCE_REGION_LEN = _SPACING_FEATURE_LEN + _EXTENSION_FEATURE_LEN
_CLASS_REGION_LEN = _CURL_CLASS_FEATURE_LEN + _SPREAD_CLASS_FEATURE_LEN

# Default minimum gap between best and second-best gesture scores.
_DEFAULT_CONFIDENCE_MARGIN = 0.05

# Hysteresis: once a gesture is actively matching, allow its score to dip
# this far below the entry threshold before letting go. Prevents the
# "flickers on/off while held still" pattern caused by per-frame noise
# crossing the boundary. Only kicks in when classify() is called with a
# `sticky_name` hint (test.py passes the current hold_name).
_DEFAULT_HYSTERESIS = 0.06


def _distance_to_score(distance: float) -> float:
    """Convert Euclidean distance to a 0..1 score (1 = identical)."""
    if distance <= 0.0:
        return 1.0
    return max(0.0, 1.0 - distance / _SCORE_ZERO_DISTANCE)


def _apply_region_weights(arr: np.ndarray) -> None:
    """Scale each feature region in place so the unweighted Euclidean
    distance at match time matches our intended weighting. Accepts
    either a 1-D query vector or a 2-D matrix (rows = samples).

    Region order (cumulative):
      [0:63]   landmarks                              (weight 1.0)
      [63:71]  spacing + extension (distance region)  → DISTANCE
      [71:81]  joint angles                           → JOINT
      [81:87]  curl + spread (class region)           → CLASS
      [87:102] per-finger direction unit vectors      → DIRECTION
      [102:103] thumb-index spread distance           → DISTANCE
      [103:106] palm-normal unit vector               → PALM
    """
    if arr.ndim == 1:
        cols_view = arr
        n_cols = arr.shape[0]
    else:
        cols_view = arr
        n_cols = arr.shape[1]

    def _scale(start: int, end: int, weight: float) -> None:
        if end > n_cols:
            return
        if arr.ndim == 1:
            cols_view[start:end] *= weight
        else:
            cols_view[:, start:end] *= weight

    dist_start = _LANDMARK_FEATURE_LEN
    dist_end = dist_start + _DISTANCE_REGION_LEN
    joint_end = dist_end + _JOINT_ANGLE_FEATURE_LEN
    class_end = joint_end + _CLASS_REGION_LEN
    direction_end = class_end + _DIRECTION_FEATURE_LEN
    tspread_end = direction_end + _THUMB_INDEX_SPREAD_FEATURE_LEN
    palm_end = tspread_end + _PALM_NORMAL_FEATURE_LEN

    _scale(dist_start, dist_end, _DISTANCE_FEATURE_WEIGHT)
    _scale(dist_end, joint_end, _JOINT_ANGLE_WEIGHT)
    _scale(joint_end, class_end, _CLASS_FEATURE_WEIGHT)
    _scale(class_end, direction_end, _DIRECTION_FEATURE_WEIGHT)
    _scale(direction_end, tspread_end, _DISTANCE_FEATURE_WEIGHT)
    _scale(tspread_end, palm_end, _PALM_NORMAL_FEATURE_WEIGHT)


@dataclass(frozen=True)
class MatchResult:
    gesture: CustomGesture
    score: float  # 1.0 = identical, 0.0 = very different (see _distance_to_score)
    distance: float  # raw weighted Euclidean distance
    sample_index: int  # which stored sample was the best match
    runner_up_name: Optional[str] = None  # name of the second-best gesture (if any)
    runner_up_score: float = 0.0


class GestureClassifier:
    """KNN matcher over saved custom gestures. Uses weighted Euclidean
    distance on normalized landmark features + structural features
    (spacing / extension / joint angles), and returns the single best
    match whose score clears the threshold AND beats its runner-up by at
    least `confidence_margin`.

    Why Euclidean and not cosine: cosine ignores absolute per-landmark
    position, so two poses with the same overall direction but different
    finger spread (e.g., "four fingers together" vs "four fingers apart
    pointing up") score ~0.97 either way — the small lateral differences
    get swamped by the dominant up-direction. Euclidean keeps those
    differences visible.

    Why a confidence margin: if the user's hand is between two registered
    poses, both might score above threshold. Without a margin we'd fire
    whichever happened to win that frame — usually the wrong action.

    Threshold default 0.88: identical recordings score ~1.0, the same pose
    held with natural jitter scores 0.92-0.98, similar-but-distinct poses
    (finger spread changes) typically drop to 0.6-0.8. Raise for stricter
    matching, lower for more forgiving.
    """

    def __init__(
        self,
        registry: Optional[GestureRegistry] = None,
        *,
        gestures: Optional[Iterable[CustomGesture]] = None,
        threshold: float = 0.88,
        confidence_margin: float = _DEFAULT_CONFIDENCE_MARGIN,
        hysteresis: float = _DEFAULT_HYSTERESIS,
    ) -> None:
        if registry is None and gestures is None:
            raise ValueError("either registry or gestures must be provided")
        self._registry = registry
        self._explicit_gestures: Optional[List[CustomGesture]] = (
            list(gestures) if gestures is not None else None
        )
        self._threshold = float(threshold)
        self._confidence_margin = max(0.0, float(confidence_margin))
        self._hysteresis = max(0.0, float(hysteresis))
        self._matrix: Optional[np.ndarray] = None
        self._sample_to_gesture_idx: List[int] = []
        self._sample_to_local_idx: List[int] = []
        self._gestures: List[CustomGesture] = []

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def confidence_margin(self) -> float:
        return self._confidence_margin

    def reload(self) -> None:
        """Rebuild the sample matrix from the current source. If `gestures`
        was passed at construction, that list is used; otherwise the
        registry's in-memory state is used. The classifier deliberately
        does NOT force a registry reload from disk — call registry.load()
        first if you want fresh-from-disk state."""
        if self._explicit_gestures is not None:
            gestures = self._explicit_gestures
        elif self._registry is not None:
            gestures = self._registry.list()
        else:
            gestures = []
        self._gestures = list(gestures)
        self._sample_to_gesture_idx = []
        self._sample_to_local_idx = []
        rows: List[List[float]] = []
        for g_idx, g in enumerate(self._gestures):
            for s_idx, sample in enumerate(g.samples):
                rows.append(sample.features)
                self._sample_to_gesture_idx.append(g_idx)
                self._sample_to_local_idx.append(s_idx)
        if not rows:
            self._matrix = None
            return
        matrix = np.asarray(rows, dtype=np.float32)
        # Apply per-region weights in-place so classify() is a plain
        # unweighted Euclidean — simpler and faster than weighting on every
        # call. See _apply_region_weights for the full region layout.
        _apply_region_weights(matrix)
        self._matrix = matrix

    def _match_from_features(
        self,
        features: np.ndarray,
        *,
        sticky_name: Optional[str] = None,
    ) -> Optional[MatchResult]:
        if self._matrix is None:
            self.reload()
        if self._matrix is None or self._matrix.size == 0:
            return None
        # Apply the same per-region weights to the query so matrix rows
        # and query live in the same weighted space.
        q = np.array(features, dtype=np.float32, copy=True)
        _apply_region_weights(q)
        diffs = self._matrix - q
        distances = np.linalg.norm(diffs, axis=1)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        score = _distance_to_score(best_distance)
        best_g_idx = self._sample_to_gesture_idx[best_idx]
        best_g = self._gestures[best_g_idx]

        # Hysteresis: if the best match is the gesture the caller says is
        # currently active, relax the threshold. Stops the "flickers while
        # held still" pattern when the score sits near the boundary.
        effective_threshold = self._threshold
        if sticky_name is not None and best_g.name == sticky_name:
            effective_threshold = max(0.0, self._threshold - self._hysteresis)

        if score < effective_threshold:
            return None

        # Confidence-margin check: best score must beat the best score of
        # any OTHER gesture by at least confidence_margin. Find the best
        # competitor by skipping samples that belong to the winning gesture.
        runner_up_score = 0.0
        runner_up_name: Optional[str] = None
        if len(self._gestures) > 1:
            sample_g_idx = np.asarray(self._sample_to_gesture_idx)
            other_mask = sample_g_idx != best_g_idx
            if np.any(other_mask):
                other_distances = distances[other_mask]
                ru_local = int(np.argmin(other_distances))
                ru_distance = float(other_distances[ru_local])
                runner_up_score = _distance_to_score(ru_distance)
                # Map back to the gesture: take the indices of all "other"
                # samples and find the one ru_local picked.
                other_indices = np.flatnonzero(other_mask)
                ru_g_idx = int(sample_g_idx[other_indices[ru_local]])
                runner_up_name = self._gestures[ru_g_idx].name

        if (
            self._confidence_margin > 0.0
            and runner_up_name is not None
            and (score - runner_up_score) < self._confidence_margin
        ):
            return None

        local_idx = self._sample_to_local_idx[best_idx]
        return MatchResult(
            gesture=best_g,
            score=score,
            distance=best_distance,
            sample_index=local_idx,
            runner_up_name=runner_up_name,
            runner_up_score=runner_up_score,
        )

    def classify(
        self,
        landmarks: np.ndarray,
        *,
        sticky_name: Optional[str] = None,
    ) -> Optional[MatchResult]:
        """Classify a live (21, 3) landmark array. Returns the best match
        above threshold, or None.

        Pass `sticky_name` to engage hysteresis: if the best match would
        be the named gesture, the threshold drops by the configured
        hysteresis amount. Use this to keep an actively-held gesture
        recognized despite per-frame score wiggle near the boundary.
        """
        return self._match_from_features(
            normalize_landmarks(landmarks), sticky_name=sticky_name
        )

    def classify_raw(
        self,
        feature_vector: Sequence[float],
        *,
        sticky_name: Optional[str] = None,
    ) -> Optional[MatchResult]:
        """Alternate entry point when the caller already has a normalized
        feature vector (e.g., reusing an existing pipeline's output)."""
        return self._match_from_features(
            np.asarray(feature_vector, dtype=np.float32),
            sticky_name=sticky_name,
        )

    def best_score_for(self, landmarks: np.ndarray) -> tuple[Optional[str], float]:
        """Diagnostic: return (best_gesture_name, score) regardless of
        threshold. Used by the live runner's debug log so the user can
        see how close they are to firing — distinguishes 'score=0.86,
        just below threshold' from 'score=0.20, totally wrong pose'."""
        if self._matrix is None:
            self.reload()
        if self._matrix is None or self._matrix.size == 0:
            return None, 0.0
        try:
            features = normalize_landmarks(landmarks)
        except Exception:
            return None, 0.0
        q = np.array(features, dtype=np.float32, copy=True)
        _apply_region_weights(q)
        diffs = self._matrix - q
        distances = np.linalg.norm(diffs, axis=1)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        score = _distance_to_score(best_distance)
        best_g_idx = self._sample_to_gesture_idx[best_idx]
        best_name = self._gestures[best_g_idx].name
        return best_name, score

# Author: Konstantin Markov

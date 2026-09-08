"""Analyze a continuous pose-sequence recording.

The user performs the full sequence in one take (e.g. three → two → one).
This module segments the landmark stream into ordered held poses, measures
hold durations and gaps, and derives match timing limits with a small
buffer so live performance can vary slightly without becoming a free-form
preset slider.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .recorder import augment_samples, landmarks_to_sample, normalize_landmarks
from .registry import GestureSample, PoseSequenceStep


# Timing buffers relative to the measured demo.
_HOLD_LO_FACTOR = 0.65   # dwell: slightly under shortest hold
_HOLD_HI_FACTOR = 1.35   # max hold: slightly over longest hold
_GAP_HI_FACTOR = 1.40    # max gap: slightly over longest gap
_MIN_DWELL_MS = 80
_MIN_GAP_MS = 800
_MIN_MAX_HOLD_MS = 1600
_MIN_HOLD_S = 0.10
_CHANGE_SIM = 0.90
_MERGE_SIM = 0.945
_CONFIRM_FRAMES = 3
_MAX_SAMPLES_PER_STEP = 18
_DEFAULT_STEP_NAMES = ("Pose 1", "Pose 2", "Pose 3", "Pose 4", "Pose 5")


@dataclass(frozen=True)
class SequenceFrame:
    """One captured frame from the continuous recording."""
    t: float
    landmarks: np.ndarray  # (21, 3)
    handedness: str = ""


@dataclass(frozen=True)
class SequenceAnalysis:
    steps: List[PoseSequenceStep]
    hold_ms: List[int]
    gap_ms: List[int]
    dwell_ms: int
    max_hold_ms: int
    max_gap_ms: int
    handedness: Optional[str]
    step_names: List[str]


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (float(np.linalg.norm(a)) + 1e-9)
    bn = b / (float(np.linalg.norm(b)) + 1e-9)
    return float(np.dot(an, bn))


def _centroid(feats: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(list(feats), axis=0), axis=0).astype(np.float32)


def _segment_frames(
    frames: Sequence[SequenceFrame],
    features: Sequence[np.ndarray],
    *,
    min_hold_s: float = _MIN_HOLD_S,
    change_sim: float = _CHANGE_SIM,
    confirm: int = _CONFIRM_FRAMES,
) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx_exclusive) stable pose spans."""
    n = len(frames)
    if n < 4:
        return []

    spans: List[Tuple[int, int]] = []
    cur_start = 0
    cur_feats: List[np.ndarray] = [features[0]]
    pending: List[int] = []

    def _close(start: int, end: int) -> None:
        if end - start < 4:
            return
        hold = float(frames[end - 1].t - frames[start].t)
        if hold < min_hold_s:
            return
        spans.append((start, end))

    for i in range(1, n):
        c = _centroid(cur_feats)
        if _cos_sim(c, features[i]) >= change_sim:
            if pending:
                # Brief flicker — absorb into current pose.
                for j in pending:
                    cur_feats.append(features[j])
                pending = []
            cur_feats.append(features[i])
        else:
            pending.append(i)
            if len(pending) >= confirm:
                _close(cur_start, pending[0])
                cur_start = pending[0]
                cur_feats = [features[j] for j in pending]
                pending = []

    if pending:
        _close(cur_start, pending[0])
        _close(pending[0], n)
    else:
        _close(cur_start, n)

    if not spans:
        return []

    # Merge adjacent spans whose centroids are nearly the same pose.
    merged: List[Tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_s, prev_e = merged[-1]
        prev_c = _centroid(features[prev_s:prev_e])
        cur_c = _centroid(features[start:end])
        if _cos_sim(prev_c, cur_c) >= _MERGE_SIM:
            merged[-1] = (prev_s, end)
        else:
            merged.append((start, end))
    return merged


def _subsample_indices(n: int, max_n: int) -> List[int]:
    if n <= max_n:
        return list(range(n))
    if max_n <= 1:
        return [n // 2]
    return [int(round(i * (n - 1) / (max_n - 1))) for i in range(max_n)]


def _majority_hand(frames: Sequence[SequenceFrame]) -> Optional[str]:
    left = sum(1 for f in frames if f.handedness == "Left")
    right = sum(1 for f in frames if f.handedness == "Right")
    total = left + right
    if total <= 0:
        return None
    if left / total >= 0.6:
        return "Left"
    if right / total >= 0.6:
        return "Right"
    return None


def _derive_timing(
    hold_ms: Sequence[int],
    gap_ms: Sequence[int],
) -> Tuple[int, int, int]:
    if not hold_ms:
        raise ValueError("no pose holds to learn timing from")
    shortest = min(hold_ms)
    longest = max(hold_ms)
    dwell = max(_MIN_DWELL_MS, int(round(shortest * _HOLD_LO_FACTOR)))
    max_hold = max(
        dwell + 40,
        _MIN_MAX_HOLD_MS,
        int(round(longest * _HOLD_HI_FACTOR)),
    )
    if gap_ms:
        max_gap = max(_MIN_GAP_MS, int(round(max(gap_ms) * _GAP_HI_FACTOR)))
    else:
        max_gap = max(_MIN_GAP_MS, int(round(longest * 1.5)))
    return dwell, max_hold, max_gap


def analyze_pose_sequence(
    frames: Sequence[SequenceFrame],
    *,
    expected_steps: int = 3,
    step_names: Optional[Sequence[str]] = None,
) -> SequenceAnalysis:
    """Segment a one-take recording into ordered poses + learned timing.

    Raises ValueError when the take cannot be segmented into enough
    distinct held poses.
    """
    if len(frames) < 8:
        raise ValueError("Recording is too short — perform the full sequence, then stop.")

    expected = max(2, min(5, int(expected_steps)))
    features: List[np.ndarray] = []
    for fr in frames:
        features.append(normalize_landmarks(np.asarray(fr.landmarks, dtype=np.float32)))

    spans = _segment_frames(frames, features)
    if len(spans) < expected:
        raise ValueError(
            f"Found {len(spans)} held pose(s), need {expected}. "
            "Hold each pose briefly, then switch — try again in one continuous take."
        )
    if len(spans) > expected:
        # Keep the `expected` longest holds (typical countdown holds are
        # longer than transition noise fragments).
        ranked = sorted(
            spans,
            key=lambda se: float(frames[se[1] - 1].t - frames[se[0]].t),
            reverse=True,
        )[:expected]
        spans = sorted(ranked, key=lambda se: se[0])

    names: List[str]
    if step_names is not None and len(step_names) >= expected:
        names = [str(step_names[i]) for i in range(expected)]
    else:
        names = [
            _DEFAULT_STEP_NAMES[i] if i < len(_DEFAULT_STEP_NAMES) else f"step{i + 1}"
            for i in range(expected)
        ]

    steps: List[PoseSequenceStep] = []
    hold_ms_list: List[int] = []
    used_frames: List[SequenceFrame] = []
    for idx, (start, end) in enumerate(spans):
        seg_frames = list(frames[start:end])
        used_frames.extend(seg_frames)
        hold_ms_list.append(max(1, int(round((seg_frames[-1].t - seg_frames[0].t) * 1000))))
        pick = _subsample_indices(len(seg_frames), _MAX_SAMPLES_PER_STEP)
        originals = [
            landmarks_to_sample(np.asarray(seg_frames[j].landmarks, dtype=np.float32))
            for j in pick
        ]
        augmented = augment_samples(originals)
        steps.append(PoseSequenceStep(name=names[idx], samples=list(augmented)))

    gap_ms_list: List[int] = []
    for i in range(len(spans) - 1):
        _, end_a = spans[i]
        start_b, _ = spans[i + 1]
        gap = float(frames[start_b].t - frames[end_a - 1].t)
        gap_ms_list.append(max(0, int(round(gap * 1000))))

    dwell_ms, max_hold_ms, max_gap_ms = _derive_timing(hold_ms_list, gap_ms_list)
    hand = _majority_hand(used_frames)

    # Guard: adjacent learned poses must differ enough that order matters.
    for i in range(len(spans) - 1):
        a_s, a_e = spans[i]
        b_s, b_e = spans[i + 1]
        if _cos_sim(_centroid(features[a_s:a_e]), _centroid(features[b_s:b_e])) >= _MERGE_SIM:
            raise ValueError(
                "Adjacent poses look too similar — make each step a clearer "
                "distinct hand shape and record again."
            )

    return SequenceAnalysis(
        steps=steps,
        hold_ms=hold_ms_list,
        gap_ms=gap_ms_list,
        dwell_ms=dwell_ms,
        max_hold_ms=max_hold_ms,
        max_gap_ms=max_gap_ms,
        handedness=hand,
        step_names=names,
    )

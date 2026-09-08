"""Human-readable summaries of saved gestures, computed from the
categorical curl/spread features in each stored sample.

These are the same features the classifier uses for matching, so what
you see here is exactly the "shape signature" the system has memorized
for the gesture.

`pose_signature()` returns RANGES (min, max) — a finger that fluctuated
between class 1 and class 2 during recording is reported as both, and
the description shows "slightly/half curled". This means:
  - The user gets an honest report of natural variation.
  - The classifier's KNN nearest-neighbor matching naturally accepts
    queries that fall anywhere within the recorded range.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .action import describe as describe_action
from .registry import (
    CustomGesture,
    _CURL_CLASS_FEATURE_LEN,
    _EXTENSION_FEATURE_LEN,
    _JOINT_ANGLE_FEATURE_LEN,
    _LANDMARK_FEATURE_LEN,
    _SPACING_FEATURE_LEN,
    _SPREAD_CLASS_FEATURE_LEN,
)


_CURL_OFFSET = (
    _LANDMARK_FEATURE_LEN
    + _SPACING_FEATURE_LEN
    + _EXTENSION_FEATURE_LEN
    + _JOINT_ANGLE_FEATURE_LEN
)
_SPREAD_OFFSET = _CURL_OFFSET + _CURL_CLASS_FEATURE_LEN

_FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
_CURL_LABELS = (
    "fully extended (pointing out)",
    "slightly curled",
    "half curled",
    "mostly curled",
    "closed (curled into palm)",
)
_CURL_LABELS_SHORT = (
    "extended",
    "slightly curled",
    "half curled",
    "mostly curled",
    "closed",
)
_SPREAD_LABELS = (
    "tight (fingers touching)",
    "small (fingers close)",
    "medium spread",
    "wide spread",
)
_SPREAD_LABELS_SHORT = ("tight", "small", "medium", "wide")


@dataclass(frozen=True)
class CategoricalRange:
    """The min/max/mode of a categorical feature across samples."""
    min_class: int
    max_class: int
    mode_class: int


def _range(values: List[int]) -> CategoricalRange:
    if not values:
        return CategoricalRange(0, 0, 0)
    return CategoricalRange(
        min_class=min(values),
        max_class=max(values),
        mode_class=Counter(values).most_common(1)[0][0],
    )


def pose_signature(gesture: CustomGesture) -> Dict[str, CategoricalRange]:
    """Range descriptors for each finger's curl class and the overall
    spread class. Computed across every stored sample (originals +
    augmented). For a perfectly stable pose, min == max for every
    feature — the categorical features were designed to snap to a single
    value across natural noise."""
    curls_per_finger: List[List[int]] = [[], [], [], [], []]
    spreads: List[int] = []
    for sample in gesture.samples:
        feats = sample.features
        if len(feats) < _SPREAD_OFFSET + _SPREAD_CLASS_FEATURE_LEN:
            continue
        for i in range(5):
            curls_per_finger[i].append(int(feats[_CURL_OFFSET + i]))
        spreads.append(int(feats[_SPREAD_OFFSET]))
    return {
        "thumb_curl": _range(curls_per_finger[0]),
        "index_curl": _range(curls_per_finger[1]),
        "middle_curl": _range(curls_per_finger[2]),
        "ring_curl": _range(curls_per_finger[3]),
        "pinky_curl": _range(curls_per_finger[4]),
        "spread": _range(spreads),
    }


def live_signature(features: Sequence[float]) -> Dict[str, int]:
    """Single-frame readout: curl class for each finger + the spread class
    extracted from a freshly-computed feature vector."""
    feats = np.asarray(features, dtype=np.float32)
    if feats.shape[0] < _SPREAD_OFFSET + _SPREAD_CLASS_FEATURE_LEN:
        return {}
    return {
        "thumb_curl": int(feats[_CURL_OFFSET + 0]),
        "index_curl": int(feats[_CURL_OFFSET + 1]),
        "middle_curl": int(feats[_CURL_OFFSET + 2]),
        "ring_curl": int(feats[_CURL_OFFSET + 3]),
        "pinky_curl": int(feats[_CURL_OFFSET + 4]),
        "spread": int(feats[_SPREAD_OFFSET]),
    }


def _curl_range_label(rng: CategoricalRange) -> str:
    lo = max(0, min(len(_CURL_LABELS) - 1, rng.min_class))
    hi = max(0, min(len(_CURL_LABELS) - 1, rng.max_class))
    if lo == hi:
        return _CURL_LABELS[lo]
    if hi - lo == 1:
        return f"{_CURL_LABELS_SHORT[lo]}/{_CURL_LABELS_SHORT[hi]}"
    return f"{_CURL_LABELS_SHORT[lo]} to {_CURL_LABELS_SHORT[hi]}"


def _spread_range_label(rng: CategoricalRange) -> str:
    lo = max(0, min(len(_SPREAD_LABELS) - 1, rng.min_class))
    hi = max(0, min(len(_SPREAD_LABELS) - 1, rng.max_class))
    if lo == hi:
        return _SPREAD_LABELS[lo]
    if hi - lo == 1:
        return f"{_SPREAD_LABELS_SHORT[lo]}/{_SPREAD_LABELS_SHORT[hi]}"
    return f"{_SPREAD_LABELS_SHORT[lo]} to {_SPREAD_LABELS_SHORT[hi]}"


def short_curl_label(curl_class: int) -> str:
    """Single-frame curl class -> short label (used by live overlays)."""
    idx = max(0, min(len(_CURL_LABELS_SHORT) - 1, curl_class))
    return _CURL_LABELS_SHORT[idx]


def short_spread_label(spread_class: int) -> str:
    idx = max(0, min(len(_SPREAD_LABELS_SHORT) - 1, spread_class))
    return _SPREAD_LABELS_SHORT[idx]


def format_gesture_summary(gesture: CustomGesture) -> str:
    """Multi-line human-readable summary of a saved gesture, including the
    pose recipe derived from the categorical features. Ranges are shown
    when natural recording variation produced them."""
    kind = str(getattr(gesture, "kind", "static") or "static")
    if kind == "pose_sequence":
        lines: List[str] = []
        title = f"How to do '{gesture.name}'"
        banner = "=" * min(len(title) + 2, 48)
        lines.append(banner)
        lines.append(f" {title}")
        lines.append(banner)
        if gesture.description:
            lines.append(f"Description: {gesture.description}")
        lines.append(f"Action:      {describe_action(gesture.action)}")
        if gesture.handedness in ("Left", "Right"):
            lines.append(f"Hand:        {gesture.handedness} (only fires on this hand)")
        else:
            lines.append("Hand:        either (fires on left or right)")
        lines.append(
            f"Sequence:    {len(gesture.pose_sequence_steps)} poses · "
            f"learned hold {gesture.pose_sequence_dwell_ms}–"
            f"{gesture.pose_sequence_max_hold_ms} ms · "
            f"gap ≤ {gesture.pose_sequence_max_gap_ms} ms"
        )
        lines.append("")
        lines.append("Perform poses in this order (timing from your demo):")
        for i, step in enumerate(gesture.pose_sequence_steps):
            lines.append(
                f"  {i + 1}. {step.name or f'step {i + 1}'} "
                f"({len(step.samples)} samples)"
            )
        return "\n".join(lines)

    sig = pose_signature(gesture)
    lines = []
    title = f"How to do '{gesture.name}'"
    # Cap banner width so a long gesture name can't force the
    # settings card wider than the scroll viewport.
    banner = "=" * min(len(title) + 2, 48)
    lines.append(banner)
    lines.append(f" {title}")
    lines.append(banner)

    if gesture.description:
        lines.append(f"Description: {gesture.description}")
    lines.append(f"Action:      {describe_action(gesture.action)}")
    if gesture.handedness in ("Left", "Right"):
        lines.append(f"Hand:        {gesture.handedness} (only fires on this hand)")
    else:
        lines.append("Hand:        either (fires on left or right)")
    lines.append(f"Samples:     {len(gesture.samples)} stored "
                 f"(includes augmentation variants)")
    lines.append("")
    lines.append("Hand pose:")
    finger_keys = (
        "thumb_curl", "index_curl", "middle_curl", "ring_curl", "pinky_curl",
    )
    for name, key in zip(_FINGER_NAMES, finger_keys):
        lines.append(f"  {name:<7}  {_curl_range_label(sig[key])}")
    lines.append(f"  Spread:  {_spread_range_label(sig['spread'])}")
    return "\n".join(lines)

# Author: Konstantin Markov

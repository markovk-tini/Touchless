from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .registry import GestureSample


_LANDMARK_COUNT = 21
_LANDMARK_DIM = _LANDMARK_COUNT * 3  # 63 — x, y, z per landmark
# 87 was the v4 size: 63 landmark + 3 spacing + 5 extension + 10 joint
# + 5 curl-class + 1 spread-class.
# v5 adds direction + structure features that nail down occlusion-
# robustness and direction-of-pointing for poses like thumbs-up:
#   + 15  per-finger MCP→tip unit vectors  (5 fingers × 3 dims)
#   +  1  thumb-to-index spread distance
#   +  3  palm normal unit vector  (cross of wrist→idx-MCP, wrist→pinky-MCP)
# Total v5: 87 + 19 = 106.
_FEATURE_DIM = 106

# Per-finger wrist-to-tip distance thresholds (normalized landmark units,
# where wrist-to-L9 = 1.0). Calibrated against real MediaPipe outputs.
# Format: (extended_floor, slightly_floor, half_floor, mostly_floor)
_CURL_DISTANCE_THRESHOLDS: Tuple[Tuple[float, float, float, float], ...] = (
    (1.00, 0.90, 0.80, 0.72),  # thumb
    (1.70, 1.40, 1.05, 0.75),  # index
    (1.80, 1.50, 1.10, 0.75),  # middle
    (1.65, 1.35, 1.00, 0.65),  # ring
    (1.40, 1.15, 0.85, 0.60),  # pinky
)

# Per-finger landmark chains (4 landmarks each), used to compute joint
# bend angles. MediaPipe's hand model: 0=wrist, 1-4=thumb, 5-8=index,
# 9-12=middle, 13-16=ring, 17-20=pinky.
_FINGER_CHAINS: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 2, 3, 4),     # thumb
    (5, 6, 7, 8),     # index
    (9, 10, 11, 12),  # middle
    (13, 14, 15, 16), # ring
    (17, 18, 19, 20), # pinky
)


# Default augmentation: small rotations around each axis to give the
# classifier tolerance for natural hand tilt / roll / yaw without requiring
# the user to record the same pose 50 times. Angles are in degrees.
#   - Z roll (in-plane): ±8°, ±16° — most natural wrist motion
#   - X pitch (tilt forward/back): ±6°, ±12°, ±18° — extended for back-tilt
#   - Y yaw (pan left/right):      ±6°, ±12°, ±18° — same
_DEFAULT_AUGMENT_ANGLES_DEG: Dict[str, List[float]] = {
    "x": [6.0, -6.0, 12.0, -12.0, 18.0, -18.0],
    "y": [6.0, -6.0, 12.0, -12.0, 18.0, -18.0],
    "z": [8.0, -8.0, 16.0, -16.0],
}

# Thumb landmarks are MediaPipe indices 1..4 (CMC, MCP, IP, TIP). Natural
# thumb wobble during a held pose moves these by up to ~0.1-0.2 in
# normalized-landmark units. Rotational augmentation rotates the whole
# hand, so thumb-specific variance needs its own augmentation pass.
_THUMB_LANDMARK_INDICES = (1, 2, 3, 4)
_THUMB_JITTER_STD = 0.06  # one-sigma; captures ~0.12 of peak jitter
_THUMB_JITTER_VARIANTS = 3  # per captured sample
_THUMB_JITTER_SEED = 0xC5057  # deterministic so tests are stable

# General per-finger jitter covers natural micro-variation in finger
# positions during pinch / contact poses (e.g., the OK sign where the
# index curls to touch the thumb — exact contact point varies). Applies
# small noise to all finger landmarks (1..20, skipping wrist).
_FINGER_LANDMARK_INDICES = tuple(range(1, 21))
_FINGER_JITTER_STD = 0.03  # smaller than thumb-specific; less aggressive
_FINGER_JITTER_VARIANTS = 2
_FINGER_JITTER_SEED = 0xF1A6E2


def _spacing_features_from_landmarks(lm: np.ndarray) -> np.ndarray:
    """Adjacent fingertip-pair distances on a (21, 3) landmark array.
    Rotation-invariant; directly encodes finger-grouping structure."""
    pairs = [(8, 12), (12, 16), (16, 20)]
    return np.asarray(
        [float(np.linalg.norm(lm[a] - lm[b])) for a, b in pairs],
        dtype=np.float32,
    )


def _extension_features_from_landmarks(lm: np.ndarray) -> np.ndarray:
    """Wrist-to-fingertip distances (5 values: thumb, index, middle, ring,
    pinky). Encodes how extended each finger is — a curled finger gives a
    small value (~1.5 normalized units), an extended finger a large one
    (~3-4). Rotation-invariant like the spacing features.
    """
    tips = [4, 8, 12, 16, 20]
    return np.asarray(
        [float(np.linalg.norm(lm[t] - lm[0])) for t in tips],
        dtype=np.float32,
    )


def _bend_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two 3D segment vectors in radians.
    0 = colinear (extended); π/2 = perpendicular (90° bent); π = folded back.
    """
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return float(np.arccos(cos))


# Thumb curl thresholds on the SUM of inner-joint bend angles
# (bend at L2 + bend at L3, both in radians). The thumb chain is
# L1 (CMC) → L2 (MCP) → L3 (IP) → L4 (TIP). Joint angles live
# entirely on the thumb itself, so they're immune to hand
# rotation / foreshortening — the two-distance MAX rule the
# original code used would push intermediate poses straight to
# closed because `tip_to_index_MCP` collapses fast when the thumb
# heads toward the palm in 2D. Joint angles give a real graduated
# signal across all 5 curl classes.
#
# Why joint angles work for the thumb but NOT for the other four
# fingers: the other fingers curl FORWARD (toward the camera),
# which puts their joints into MediaPipe's noisy z-axis. The
# thumb curls SIDEWAYS across the palm — its joints stay in the
# xy plane, which MediaPipe predicts cleanly.
#
# Buckets are on the SUM of the two inner-joint bends so partial
# bends accumulate naturally — a thumb with each joint at ~0.4 rad
# (slight C-shape) lands at sum ~0.8 → class 1 (slightly curled),
# without one signal having to single-handedly cross a threshold.
_THUMB_JOINT_BEND_SUM_THRESHOLDS: Tuple[float, float, float, float] = (
    0.5,   # < → fully extended (0)
    1.0,   # < → slightly curled (1)
    1.5,   # < → half curled (2)
    2.0,   # < → mostly curled (3)
    # >= → closed (4)
)


def _curl_class_features(
    extension_distances: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:
    """5 ordinal curl-class labels per finger (0=fully extended,
    1=slightly curled, 2=half curled, 3=mostly curled, 4=closed/fist).

    Primary signal: wrist-to-fingertip distance. Distance is more robust
    than joint-angle sums because MediaPipe's z-coordinate is noisy enough
    that bend angles for fingers curling forward come out artificially
    small. Wrist-to-tip distance is a single 2D-dominant measurement that
    doesn't require accurate depth inference at every joint.

    Thumb classification uses a different scheme: the SUM of bend
    angles at the inner thumb joints (L2 and L3) is the only signal.
    The thumb curls SIDEWAYS across the palm rather than forward
    toward the camera, so its joint angles aren't corrupted by
    MediaPipe's z-axis noise (which is what made joint angles
    unreliable for the OTHER fingers). Joint-angle sum gives a real
    graduated 5-class signal instead of the binary 0/closed jump
    the previous distance-MAX rule produced.

    Hard bucketing — values snap to a stable integer that doesn't change
    under small landmark noise.
    """
    out: List[float] = []
    for finger_idx in range(5):
        dist = float(extension_distances[finger_idx])
        thresholds = _CURL_DISTANCE_THRESHOLDS[finger_idx]
        if dist >= thresholds[0]:
            cls = 0.0
        elif dist >= thresholds[1]:
            cls = 1.0
        elif dist >= thresholds[2]:
            cls = 2.0
        elif dist >= thresholds[3]:
            cls = 3.0
        else:
            cls = 4.0

        if finger_idx == 0:
            # Thumb: replace the distance-based class entirely with
            # joint-bend geometry. See _THUMB_JOINT_BEND_SUM_THRESHOLDS
            # for the rationale — short version: joint angles on the
            # thumb are orientation-invariant AND graduated, so they
            # naturally span all 5 curl classes. The two distance
            # signals (wrist→tip, tip→index-MCP) both collapse in 2D
            # under common hand rotations and would either skip past
            # the intermediate classes (binary 0/4 jump) or
            # over-report curl on a geometrically-straight thumb.
            seg_12 = landmarks[2] - landmarks[1]
            seg_23 = landmarks[3] - landmarks[2]
            seg_34 = landmarks[4] - landmarks[3]
            bend_sum = _bend_angle(seg_12, seg_23) + _bend_angle(seg_23, seg_34)
            t = _THUMB_JOINT_BEND_SUM_THRESHOLDS
            if bend_sum < t[0]:
                cls = 0.0
            elif bend_sum < t[1]:
                cls = 1.0
            elif bend_sum < t[2]:
                cls = 2.0
            elif bend_sum < t[3]:
                cls = 3.0
            else:
                cls = 4.0

        out.append(cls)
    return np.asarray(out, dtype=np.float32)


def _spread_class_features(spacing_features: np.ndarray) -> np.ndarray:
    """1 ordinal spread-class label (0=tight..3=wide). Derived from the
    SUM of the 3 adjacent fingertip-pair distances. Calibrated against
    real MediaPipe outputs."""
    total = float(np.sum(spacing_features))
    if total < 0.35:
        return np.asarray([0.0], dtype=np.float32)
    if total < 0.65:
        return np.asarray([1.0], dtype=np.float32)
    if total < 1.05:
        return np.asarray([2.0], dtype=np.float32)
    return np.asarray([3.0], dtype=np.float32)


def _direction_features_from_landmarks(lm: np.ndarray) -> np.ndarray:
    """Per-finger MCP→TIP unit-direction vectors. 5 fingers × 3 dims =
    15 floats.

    For each finger, take the vector from its MCP (proximal knuckle)
    to its TIP and L2-normalize. This explicitly encodes which way
    each finger is POINTING, independent of finger length. Critical
    for poses like thumbs-up (thumb points roughly +Y), thumbs-side
    (thumb points roughly ±X), peace sign (index + middle both point
    +Y, others curled). The raw-landmark feature region already
    contains this info implicitly, but mixed into 63 dims it gets
    drowned by the curl variance — making it explicit pulls direction
    into its own discriminative channel.

    Also robust to occlusion: a unit-direction vector averages over
    two landmarks (MCP + tip), so per-landmark noise from a partially-
    hidden tip doesn't blow up the magnitude — only the direction
    matters here, and the MCP anchors it.
    """
    pairs = ((2, 4), (5, 8), (9, 12), (13, 16), (17, 20))
    out: List[float] = []
    for mcp_idx, tip_idx in pairs:
        v = lm[tip_idx] - lm[mcp_idx]
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            out.extend([0.0, 0.0, 0.0])
        else:
            out.extend((float(v[0] / n), float(v[1] / n), float(v[2] / n)))
    return np.asarray(out, dtype=np.float32)


def _thumb_index_spread_from_landmarks(lm: np.ndarray) -> np.ndarray:
    """Distance L4 (thumb tip) → L8 (index tip), normalized by the
    standard hand scale (already applied to `lm` by the caller). The
    existing spacing region covers adjacent tip pairs (8-12, 12-16,
    16-20) but NOT thumb↔index — so the difference between "C-shape
    hand" and "fully-open hand" was previously only encoded by the
    raw-landmark region. Making this explicit gives the classifier
    a single dimension that goes from ~0.3 (pinched) to ~1.5
    (fully spread), which is hard to confuse with anything else.
    """
    return np.asarray(
        [float(np.linalg.norm(lm[4] - lm[8]))],
        dtype=np.float32,
    )


def _palm_normal_from_landmarks(lm: np.ndarray) -> np.ndarray:
    """3-dim palm normal unit vector. Distinguishes palm-toward-
    camera from palm-toward-monitor poses — the same hand shape
    (e.g. open palm, or thumbs-up) reads very differently to the
    user depending on which way the palm faces, so the classifier
    needs to see it.

    Computed as the cross product of (wrist → index MCP) × (wrist →
    pinky MCP), then L2-normalized. The result points OUT of the
    palm side of the hand — for a right hand held vertically palm-
    toward-camera, this is roughly the +Z axis (toward the user);
    for the same hand back-toward-camera, it flips to -Z.
    """
    v1 = lm[5] - lm[0]
    v2 = lm[17] - lm[0]
    n = np.cross(v1, v2)
    mag = float(np.linalg.norm(n))
    if mag < 1e-6:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray((n / mag).astype(np.float32), dtype=np.float32)


def _joint_angle_features_from_landmarks(lm: np.ndarray) -> np.ndarray:
    """10 joint bend angles, 2 per finger.

    For each finger chain (a, b, c, d), measure the bend at b (between
    segments a→b and b→c) and at c (between b→c and c→d). Extended
    fingers produce angles near 0; curled fingers produce angles near π/2
    or larger. Rotation-invariant — angles between segments don't change
    under rigid rotation of the whole hand.

    Output order: thumb (L2, L3), index (L6, L7), middle (L10, L11),
                  ring (L14, L15), pinky (L18, L19).
    """
    out: List[float] = []
    for a, b, c, d in _FINGER_CHAINS:
        seg_ab = lm[b] - lm[a]
        seg_bc = lm[c] - lm[b]
        seg_cd = lm[d] - lm[c]
        out.append(_bend_angle(seg_ab, seg_bc))
        out.append(_bend_angle(seg_bc, seg_cd))
    return np.asarray(out, dtype=np.float32)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Turn raw MediaPipe hand landmarks into a 66-dim feature vector that's
    invariant to hand position and size.

    Input: (21, 3) array of (x, y, z). MediaPipe emits these in a roughly
    unit-square image frame, but exact scale varies with camera / hand
    distance, so we normalize:

    1. Translate so the wrist (landmark 0) is at the origin.
    2. Scale so the middle-finger MCP (landmark 9) — a stable interior
       joint — sits at unit distance from the wrist.
    3. Compute 3 fingertip-pair distances: |L8-L12|, |L12-L16|, |L16-L20|.
       These are rotation-invariant shape features that discriminate
       finger-grouping patterns which raw-landmark Euclidean misses (e.g.,
       V-split vs fingers-together).

    Rotation is NOT normalized on the landmark portion — the classifier
    keeps orientation as a distinguishing feature, which is usually what
    users want (thumbs-up vs thumbs-sideways).
    """
    if landmarks.shape != (_LANDMARK_COUNT, 3):
        raise ValueError(
            f"expected landmarks of shape ({_LANDMARK_COUNT}, 3), "
            f"got {landmarks.shape}"
        )
    arr = landmarks.astype(np.float32, copy=True)
    wrist = arr[0].copy()
    arr -= wrist

    # Scale normalization: distance wrist -> middle-finger MCP (landmark 9).
    scale_vec = arr[9]
    scale = float(np.linalg.norm(scale_vec))
    if scale < 1e-6:
        scale = 1.0
    arr /= scale

    landmark_feats = arr.reshape(_LANDMARK_DIM)
    spacing_feats = _spacing_features_from_landmarks(arr)
    extension_feats = _extension_features_from_landmarks(arr)
    joint_feats = _joint_angle_features_from_landmarks(arr)
    curl_feats = _curl_class_features(extension_feats, arr)
    spread_feats = _spread_class_features(spacing_feats)
    direction_feats = _direction_features_from_landmarks(arr)
    thumb_index_spread = _thumb_index_spread_from_landmarks(arr)
    palm_normal = _palm_normal_from_landmarks(arr)
    return np.concatenate(
        [landmark_feats, spacing_feats, extension_feats, joint_feats,
         curl_feats, spread_feats,
         direction_feats, thumb_index_spread, palm_normal]
    ).astype(np.float32)


def landmarks_to_sample(landmarks: np.ndarray) -> GestureSample:
    feats = normalize_landmarks(landmarks)
    return GestureSample(features=feats.tolist())


class GestureRecorder:
    """Accumulates normalized samples while the user holds a pose. Callers
    feed raw landmark arrays via capture(); when enough samples are
    collected, finalize() returns the list for persistence.

    Expects the user to hold the pose for ~1 second across ~10+ frames; the
    classifier averages over samples to tolerate small jitter.
    """

    def __init__(self, *, target_samples: int = 15) -> None:
        self._target = max(1, int(target_samples))
        self._samples: List[GestureSample] = []

    @property
    def samples(self) -> List[GestureSample]:
        return list(self._samples)

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def target(self) -> int:
        return self._target

    def reset(self) -> None:
        self._samples.clear()

    def capture(self, landmarks: np.ndarray) -> GestureSample:
        sample = landmarks_to_sample(landmarks)
        self._samples.append(sample)
        return sample

    def is_complete(self) -> bool:
        return len(self._samples) >= self._target

    def finalize(self) -> List[GestureSample]:
        if not self._samples:
            raise ValueError("no samples captured")
        return list(self._samples)


def _rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    raise ValueError(f"unknown axis: {axis!r}")


def _landmarks_from_sample(sample: GestureSample) -> np.ndarray:
    """Unpack the landmark portion of a sample's feature vector into a
    (21, 3) array. Strips the trailing spacing features."""
    feats = np.asarray(sample.features, dtype=np.float32)[:_LANDMARK_DIM]
    return feats.reshape(_LANDMARK_COUNT, 3)


def _sample_from_landmarks(lm: np.ndarray) -> GestureSample:
    """Build a fresh 106-dim sample from a (21, 3) landmark array. All
    derived features (spacing, extension, joint angles, curl classes,
    spread class, per-finger direction, thumb-index spread, palm normal)
    are recomputed so they always reflect the current landmarks."""
    landmark_feats = lm.reshape(_LANDMARK_DIM)
    spacing_feats = _spacing_features_from_landmarks(lm)
    extension_feats = _extension_features_from_landmarks(lm)
    joint_feats = _joint_angle_features_from_landmarks(lm)
    curl_feats = _curl_class_features(extension_feats, lm)
    spread_feats = _spread_class_features(spacing_feats)
    direction_feats = _direction_features_from_landmarks(lm)
    thumb_index_spread = _thumb_index_spread_from_landmarks(lm)
    palm_normal = _palm_normal_from_landmarks(lm)
    feats = np.concatenate(
        [landmark_feats, spacing_feats, extension_feats, joint_feats,
         curl_feats, spread_feats,
         direction_feats, thumb_index_spread, palm_normal]
    ).astype(np.float32)
    return GestureSample(features=feats.tolist())


def augment_sample(
    sample: GestureSample,
    *,
    angles_deg: Dict[str, List[float]] | None = None,
    thumb_jitter_variants: int = _THUMB_JITTER_VARIANTS,
    thumb_jitter_std: float = _THUMB_JITTER_STD,
    finger_jitter_variants: int = _FINGER_JITTER_VARIANTS,
    finger_jitter_std: float = _FINGER_JITTER_STD,
) -> List[GestureSample]:
    """Expand one normalized sample into itself plus small variants:
       - rotational variants (tilt/roll/yaw) for orientation tolerance
       - thumb-jitter variants for thumb wobble tolerance
       - per-finger jitter variants for pinch/contact-pose tolerance

    Rotations rotate the whole hand around the wrist (origin in normalized
    space). Thumb jitter adds Gaussian noise to thumb landmarks only;
    finger jitter adds smaller noise to all finger landmarks (1..20),
    covering natural micro-variation in pinch/contact poses where the
    exact contact point between fingers drifts (e.g., the OK sign).
    Spacing/extension features are recomputed after each transform so
    they stay consistent with the modified landmarks.
    """
    angles = angles_deg if angles_deg is not None else _DEFAULT_AUGMENT_ANGLES_DEG
    base_landmarks = _landmarks_from_sample(sample)

    out: List[GestureSample] = [sample]

    # Rotational variants.
    for axis, angle_list in angles.items():
        for deg in angle_list:
            R = _rotation_matrix(axis, float(np.radians(deg)))
            rotated = base_landmarks @ R.T
            out.append(_sample_from_landmarks(rotated))

    # Thumb-jitter variants (deterministic via seeded RNG).
    if thumb_jitter_variants > 0 and thumb_jitter_std > 0:
        rng = np.random.default_rng(_THUMB_JITTER_SEED)
        for _ in range(thumb_jitter_variants):
            jittered = base_landmarks.copy()
            noise = rng.normal(
                0.0,
                thumb_jitter_std,
                size=(len(_THUMB_LANDMARK_INDICES), 3),
            ).astype(np.float32)
            for i, idx in enumerate(_THUMB_LANDMARK_INDICES):
                jittered[idx] = base_landmarks[idx] + noise[i]
            out.append(_sample_from_landmarks(jittered))

    # Per-finger jitter variants (smaller noise across all finger landmarks).
    if finger_jitter_variants > 0 and finger_jitter_std > 0:
        rng = np.random.default_rng(_FINGER_JITTER_SEED)
        for _ in range(finger_jitter_variants):
            jittered = base_landmarks.copy()
            noise = rng.normal(
                0.0,
                finger_jitter_std,
                size=(len(_FINGER_LANDMARK_INDICES), 3),
            ).astype(np.float32)
            for i, idx in enumerate(_FINGER_LANDMARK_INDICES):
                jittered[idx] = base_landmarks[idx] + noise[i]
            out.append(_sample_from_landmarks(jittered))

    return out


def augment_samples(
    samples: Sequence[GestureSample],
    *,
    angles_deg: Dict[str, List[float]] | None = None,
) -> List[GestureSample]:
    """Apply augment_sample to every sample in the list and concatenate."""
    out: List[GestureSample] = []
    for s in samples:
        out.extend(augment_sample(s, angles_deg=angles_deg))
    return out


def landmarks_from_mediapipe(
    mp_landmarks: Sequence[Sequence[float]],
) -> np.ndarray:
    """Adapter from MediaPipe's landmark list (21 objects with .x/.y/.z or
    tuples) to the (21, 3) numpy array the recorder expects. Handles both
    the `.landmark` list from a HandLandmarkerResult (objects with attrs)
    and plain sequences of (x, y, z) tuples.
    """
    rows: List[List[float]] = []
    for lm in mp_landmarks:
        if hasattr(lm, "x") and hasattr(lm, "y") and hasattr(lm, "z"):
            rows.append([float(lm.x), float(lm.y), float(lm.z)])
        else:
            x, y, z = lm[0], lm[1], lm[2]
            rows.append([float(x), float(y), float(z)])
    if len(rows) != _LANDMARK_COUNT:
        raise ValueError(
            f"expected {_LANDMARK_COUNT} landmarks, got {len(rows)}"
        )
    return np.asarray(rows, dtype=np.float32)

# Author: Konstantin Markov

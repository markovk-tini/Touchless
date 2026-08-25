from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Storage location. Overridable via env var so tests and alternate profiles
# can point elsewhere without editing code.
_ENV_REGISTRY_PATH = "HGR_CUSTOM_GESTURES_PATH"
_DEFAULT_REGISTRY_PATH = Path.home() / ".hgr_app" / "custom_gestures.json"

# Hard cap on the number of user custom gestures. Enforced at the UI
# layer (the create flow and the bundle-import flow both check it) so
# the current release ships with a bounded set; a future feature
# update may raise or remove this. The registry itself does NOT
# enforce it on add() — keeping the limit in the UI lets imports/tests
# stay flexible while the user-facing buttons honour the cap.
MAX_CUSTOM_GESTURES = 5

# Feature vector layout (total 87):
#   [0:63]   — 21 landmarks * 3 coords, wrist-centered, scaled by |L9|
#   [63:66]  — 3 adjacent fingertip-pair distances (grouping signal)
#   [66:71]  — 5 wrist-to-fingertip distances (extension signal)
#   [71:81]  — 10 joint-bend angles (in radians)
#   [81:86]  — 5 per-finger curl-class ordinals (0..4):
#              0=fully extended, 1=slightly, 2=half, 3=mostly, 4=closed.
#              Derived from wrist-to-fingertip distance (more robust than
#              joint angles, which are corrupted by MediaPipe z-noise for
#              fingers curling toward the palm).
#   [86:87]  — 1 spread-class ordinal (0..3): tight/small/medium/wide.
#
# The categorical features SNAP to integers so they don't flicker under
# small landmark noise the way the continuous features do. They give the
# classifier a stable shape signature on top of the precise (but jittery)
# continuous values.
# v5 feature vector (106 dims). Legacy v1-v4 samples auto-upgrade
# on load — the always-present first 63 raw landmarks are enough to
# re-derive every newer feature.
_FEATURE_VECTOR_LEN = 106
_LANDMARK_FEATURE_LEN = 63
_SPACING_FEATURE_LEN = 3
_EXTENSION_FEATURE_LEN = 5
_JOINT_ANGLE_FEATURE_LEN = 10
_CURL_CLASS_FEATURE_LEN = 5
_SPREAD_CLASS_FEATURE_LEN = 1
# v5 additions:
_DIRECTION_FEATURE_LEN = 15        # 5 fingers × 3-dim MCP→tip unit vector
_THUMB_INDEX_SPREAD_FEATURE_LEN = 1  # |L4 - L8| normalized distance
_PALM_NORMAL_FEATURE_LEN = 3       # palm-normal unit vector
# Pre-v5 vector length (used to detect "needs v5 direction features
# appended" during auto-upgrade).
_V4_FEATURE_VECTOR_LEN = 87


def registry_path() -> Path:
    env = os.getenv(_ENV_REGISTRY_PATH, "").strip()
    if env:
        return Path(env)
    return _DEFAULT_REGISTRY_PATH


@dataclass(frozen=True)
class Action:
    """A declarative action to execute when a gesture is matched.

    `kind` picks the executor. `payload` carries executor-specific params.
    Executors live in action.py and are pure dispatchers on `kind`.
    """
    kind: str  # keystroke | hotkey | text | open_url | run_command | noop
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        return cls(
            kind=str(data.get("kind", "noop")),
            payload=dict(data.get("payload") or {}),
        )


@dataclass(frozen=True)
class GestureSample:
    """One captured hand pose as a normalized feature vector."""
    features: List[float]

    def __post_init__(self) -> None:
        if len(self.features) != _FEATURE_VECTOR_LEN:
            raise ValueError(
                f"GestureSample.features must have {_FEATURE_VECTOR_LEN} "
                f"elements, got {len(self.features)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {"features": list(self.features)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GestureSample":
        feats = [float(x) for x in data.get("features", [])]

        def _dist(lm: List[float], a: int, b: int) -> float:
            ax, ay, az = lm[a * 3], lm[a * 3 + 1], lm[a * 3 + 2]
            bx, by, bz = lm[b * 3], lm[b * 3 + 1], lm[b * 3 + 2]
            dx, dy, dz = ax - bx, ay - by, az - bz
            return (dx * dx + dy * dy + dz * dz) ** 0.5

        def _angle(lm: List[float], a: int, b: int, c: int) -> float:
            """Bend angle at landmark b between segments (a→b) and (b→c).
            0 = colinear (extended); π = folded back."""
            ax, ay, az = lm[a * 3], lm[a * 3 + 1], lm[a * 3 + 2]
            bx, by, bz = lm[b * 3], lm[b * 3 + 1], lm[b * 3 + 2]
            cx, cy, cz = lm[c * 3], lm[c * 3 + 1], lm[c * 3 + 2]
            v1x, v1y, v1z = bx - ax, by - ay, bz - az
            v2x, v2y, v2z = cx - bx, cy - by, cz - bz
            n1 = (v1x * v1x + v1y * v1y + v1z * v1z) ** 0.5
            n2 = (v2x * v2x + v2y * v2y + v2z * v2z) ** 0.5
            if n1 < 1e-6 or n2 < 1e-6:
                return 0.0
            cos = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)
            cos = max(-1.0, min(1.0, cos))
            import math
            return float(math.acos(cos))

        def _derive_spacing(lm: List[float]) -> List[float]:
            return [_dist(lm, 8, 12), _dist(lm, 12, 16), _dist(lm, 16, 20)]

        def _derive_extension(lm: List[float]) -> List[float]:
            return [_dist(lm, 0, t) for t in (4, 8, 12, 16, 20)]

        def _derive_joint_angles(lm: List[float]) -> List[float]:
            chains = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                      (13, 14, 15, 16), (17, 18, 19, 20)]
            out: List[float] = []
            for a, b, c, d in chains:
                out.append(_angle(lm, a, b, c))
                out.append(_angle(lm, b, c, d))
            return out

        def _derive_curl_classes(extension: List[float]) -> List[float]:
            """Bucket each finger's wrist-to-tip distance into 5 categories
            (0=extended .. 4=closed). Per-finger thresholds calibrated
            against real MediaPipe outputs."""
            thresholds = (
                (1.00, 0.90, 0.80, 0.72),  # thumb
                (1.70, 1.40, 1.05, 0.75),  # index
                (1.80, 1.50, 1.10, 0.75),  # middle
                (1.65, 1.35, 1.00, 0.65),  # ring
                (1.40, 1.15, 0.85, 0.60),  # pinky
            )
            classes: List[float] = []
            for finger_idx in range(5):
                d = float(extension[finger_idx])
                t = thresholds[finger_idx]
                if d >= t[0]:
                    classes.append(0.0)
                elif d >= t[1]:
                    classes.append(1.0)
                elif d >= t[2]:
                    classes.append(2.0)
                elif d >= t[3]:
                    classes.append(3.0)
                else:
                    classes.append(4.0)
            return classes

        def _derive_direction(lm: List[float]) -> List[float]:
            """Per-finger MCP→tip unit-direction vectors. 5 fingers × 3
            dims = 15 floats. Mirrors the recorder's
            _direction_features_from_landmarks; kept inline so the
            legacy-upgrade path doesn't have to import from recorder
            (recorder imports registry, would create a cycle)."""
            pairs = ((2, 4), (5, 8), (9, 12), (13, 16), (17, 20))
            out: List[float] = []
            for mcp_idx, tip_idx in pairs:
                mx, my, mz = lm[mcp_idx * 3], lm[mcp_idx * 3 + 1], lm[mcp_idx * 3 + 2]
                tx, ty, tz = lm[tip_idx * 3], lm[tip_idx * 3 + 1], lm[tip_idx * 3 + 2]
                dx, dy, dz = tx - mx, ty - my, tz - mz
                n = (dx * dx + dy * dy + dz * dz) ** 0.5
                if n < 1e-6:
                    out.extend([0.0, 0.0, 0.0])
                else:
                    out.extend((dx / n, dy / n, dz / n))
            return out

        def _derive_thumb_index_spread(lm: List[float]) -> List[float]:
            """L4 (thumb tip) → L8 (index tip) distance, already
            normalized because the stored landmark region is
            wrist-centered + scaled to wrist→L9 = 1."""
            return [_dist(lm, 4, 8)]

        def _derive_palm_normal(lm: List[float]) -> List[float]:
            """Palm normal: (L5-L0) × (L17-L0), L2-normalized."""
            v1 = (lm[5 * 3] - lm[0], lm[5 * 3 + 1] - lm[1], lm[5 * 3 + 2] - lm[2])
            v2 = (lm[17 * 3] - lm[0], lm[17 * 3 + 1] - lm[1], lm[17 * 3 + 2] - lm[2])
            cx = v1[1] * v2[2] - v1[2] * v2[1]
            cy = v1[2] * v2[0] - v1[0] * v2[2]
            cz = v1[0] * v2[1] - v1[1] * v2[0]
            mag = (cx * cx + cy * cy + cz * cz) ** 0.5
            if mag < 1e-6:
                return [0.0, 0.0, 0.0]
            return [cx / mag, cy / mag, cz / mag]

        def _append_v5_features(feats_so_far: List[float]) -> List[float]:
            """Top off any pre-v5 feature vector with the direction +
            thumb-index-spread + palm-normal trio. Caller has already
            ensured everything up to v4 (87 dims) is in place."""
            lm = list(feats_so_far[:_LANDMARK_FEATURE_LEN])
            return (list(feats_so_far)
                    + _derive_direction(lm)
                    + _derive_thumb_index_spread(lm)
                    + _derive_palm_normal(lm))

        def _derive_spread_class(spacing: List[float]) -> List[float]:
            """Bucket total fingertip spread into 4 categories (tight..wide).
            Calibrated against real MediaPipe outputs."""
            total = sum(spacing)
            if total < 0.35:
                return [0.0]
            if total < 0.65:
                return [1.0]
            if total < 1.05:
                return [2.0]
            return [3.0]

        if len(feats) == _LANDMARK_FEATURE_LEN:
            # Legacy schema 1: landmarks only (63 floats). Derive everything.
            lm = feats
            spacing = _derive_spacing(lm)
            extension = _derive_extension(lm)
            joints = _derive_joint_angles(lm)
            feats = (list(feats) + spacing + extension + joints
                     + _derive_curl_classes(extension)
                     + _derive_spread_class(spacing))
            feats = _append_v5_features(feats)
        elif len(feats) == _LANDMARK_FEATURE_LEN + _SPACING_FEATURE_LEN:
            # Legacy schema 2: landmarks + spacing (66 floats).
            lm = feats[:_LANDMARK_FEATURE_LEN]
            spacing = list(feats[_LANDMARK_FEATURE_LEN:])
            extension = _derive_extension(lm)
            joints = _derive_joint_angles(lm)
            feats = (list(feats) + extension + joints
                     + _derive_curl_classes(extension)
                     + _derive_spread_class(spacing))
            feats = _append_v5_features(feats)
        elif len(feats) == _LANDMARK_FEATURE_LEN + _SPACING_FEATURE_LEN + _EXTENSION_FEATURE_LEN:
            # Legacy schema 3: landmarks + spacing + extension (71 floats).
            lm = feats[:_LANDMARK_FEATURE_LEN]
            spacing = list(feats[_LANDMARK_FEATURE_LEN:_LANDMARK_FEATURE_LEN + _SPACING_FEATURE_LEN])
            extension = list(feats[_LANDMARK_FEATURE_LEN + _SPACING_FEATURE_LEN
                                   :_LANDMARK_FEATURE_LEN + _SPACING_FEATURE_LEN + _EXTENSION_FEATURE_LEN])
            joints = _derive_joint_angles(lm)
            feats = (list(feats) + joints
                     + _derive_curl_classes(extension)
                     + _derive_spread_class(spacing))
            feats = _append_v5_features(feats)
        elif len(feats) == (_LANDMARK_FEATURE_LEN + _SPACING_FEATURE_LEN
                            + _EXTENSION_FEATURE_LEN + _JOINT_ANGLE_FEATURE_LEN):
            # Legacy schema 4: landmarks + spacing + extension + joints (81 floats).
            spacing = list(feats[_LANDMARK_FEATURE_LEN:_LANDMARK_FEATURE_LEN + _SPACING_FEATURE_LEN])
            extension = list(feats[_LANDMARK_FEATURE_LEN + _SPACING_FEATURE_LEN
                                   :_LANDMARK_FEATURE_LEN + _SPACING_FEATURE_LEN + _EXTENSION_FEATURE_LEN])
            feats = (list(feats)
                     + _derive_curl_classes(extension)
                     + _derive_spread_class(spacing))
            feats = _append_v5_features(feats)
        elif len(feats) == _V4_FEATURE_VECTOR_LEN:
            # Legacy schema 4 finalised (87 floats — landmarks + spacing
            # + extension + joints + curl + spread, no v5 direction
            # features yet). Re-derive v5 from the always-present
            # landmark region.
            feats = _append_v5_features(feats)
        return cls(features=feats)


@dataclass(frozen=True)
class CustomGesture:
    name: str
    samples: List[GestureSample]
    action: Action
    created_at: str  # ISO-8601 UTC timestamp
    description: str = ""
    # "Left" / "Right" / None. None means "either hand" (legacy
    # gestures recorded before handedness tracking, or gestures the
    # user explicitly wants to fire on either hand). The live runner
    # only fires the gesture when the tracked hand matches this value
    # (or when this value is None).
    handedness: Optional[str] = None
    # Filename (relative to <registry_dir>/gesture_thumbnails/) of the
    # picked representative frame. Stored as a PNG cropped to ~2× the
    # hand bbox during recording. Empty string means no thumbnail (legacy
    # gestures or user skipped the picker).
    image_filename: str = ""

    # ---- dynamic-gesture fields ----
    # `kind` discriminates the runtime path:
    #   "static"  -> samples field holds 1-N feature vectors; matched
    #                by the existing cosine-similarity classifier
    #                (custom_gestures/classifier.py).
    #   "dynamic" -> sample_trajectories holds N (=takes) trajectories
    #                of the SELECTED key-point landmarks over time;
    #                matched by DynamicGestureClassifier via DTW.
    # Default is "static" so any gesture deserialized from a v1 file
    # (which didn't have this field) behaves identically to before.
    kind: str = "static"
    # Dynamic-only: which landmark indices the runtime classifier
    # should extract from each incoming frame before DTW. Picked by
    # key_point_selector.select_key_points at gesture-save time.
    key_point_indices: List[int] = field(default_factory=list)
    # Dynamic-only: N x (resampled_length, num_key_points, 3) arrays
    # flattened to nested Python lists for JSON storage. We store
    # ALL takes (not a centroid) so DTW can match against the
    # variant that best resembles the user's current attempt.
    sample_trajectories: List[List[List[List[float]]]] = field(default_factory=list)
    # Dynamic-only: parallel list of absolute (palm-scaled) wrist
    # trajectories — one (resampled_length, 3) per take. Carries the
    # whole-hand translation signal that the wrist-relative
    # `sample_trajectories` discards, so the classifier can match
    # swipes by the wrist's path and stationary-wrist gestures (fist
    # squeeze, finger wiggle) by fingers alone — no binary
    # wrist-travel gate required. Empty for legacy records (loaded
    # back with wrist_motion_strength=0 → finger-only matching).
    wrist_trajectories: List[List[List[float]]] = field(default_factory=list)
    # Dynamic-only: [0, _WRIST_WEIGHT_MAX] weight derived from the
    # takes' wrist path lengths at save time. Drives the classifier's
    # finger-vs-wrist DTW blend. Stored on the record so the runtime
    # doesn't have to recompute it on every reload.
    wrist_motion_strength: float = 0.0
    # Dynamic-only: recording duration policy used when the takes
    # were captured. Useful for the wizard's "edit gesture" flow so
    # the user re-records with the same mode by default. One of:
    # "fixed_short" / "fixed_long" / "until_stopped".
    duration_mode: str = ""
    # v1.1.8.1 dynamic-only. Per-template DTW match threshold derived
    # from intra-take pairwise distance at build time. None → runtime
    # falls back to the classifier's global _DEFAULT_MATCH_THRESHOLD
    # (which is what every legacy schema=1 record uses).
    match_threshold: Optional[float] = None
    # v1.1.8.1 dynamic-only. Schema version for the wrist channel.
    # 1 = absolute palm-scaled position (legacy). 2 = displacement
    # from the take's first frame (new semantics; matches live
    # classifier post-fix). Registry.load() migrates 1 → 2 in-memory
    # and persists on next save so users don't have to re-record.
    wrist_schema: int = 1

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "handedness": self.handedness,
            "image_filename": self.image_filename,
            "action": self.action.to_dict(),
            "samples": [s.to_dict() for s in self.samples],
        }
        # Only emit the dynamic fields when they're actually populated
        # so the JSON for static gestures stays unchanged byte-for-byte
        # (clean diffs + interop with anyone editing the file by hand).
        if self.kind != "static":
            out["kind"] = self.kind
        if self.key_point_indices:
            out["key_point_indices"] = list(self.key_point_indices)
        if self.sample_trajectories:
            out["sample_trajectories"] = self.sample_trajectories
        if self.wrist_trajectories:
            out["wrist_trajectories"] = self.wrist_trajectories
        if self.wrist_motion_strength:
            out["wrist_motion_strength"] = float(self.wrist_motion_strength)
        if self.duration_mode:
            out["duration_mode"] = self.duration_mode
        # v1.1.8.1 — emit only when present so legacy static JSON stays
        # byte-for-byte identical to before.
        if self.match_threshold is not None:
            out["match_threshold"] = float(self.match_threshold)
        if self.wrist_schema and self.wrist_schema != 1:
            out["wrist_schema"] = int(self.wrist_schema)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomGesture":
        raw_hand = data.get("handedness")
        hand = str(raw_hand) if raw_hand in ("Left", "Right") else None
        kind = str(data.get("kind", "static") or "static").lower()
        if kind not in ("static", "dynamic"):
            kind = "static"
        # Static gestures must still load their per-frame feature
        # vectors. Dynamic gestures don't HAVE static samples but the
        # field is required by the dataclass, so default to empty.
        if kind == "dynamic":
            samples = []  # static-pose samples are not used in this kind
        else:
            samples = [GestureSample.from_dict(s) for s in data.get("samples", [])]
        # Defensive coercion on the dynamic fields — a hand-edited
        # JSON might have nonsense in any of them.
        raw_kp = data.get("key_point_indices") or []
        try:
            key_point_indices = [int(i) for i in raw_kp]
        except Exception:
            key_point_indices = []
        raw_traj = data.get("sample_trajectories") or []
        # We accept it as-is and rely on the runtime template builder
        # to validate shape (the registry doesn't own numpy import).
        sample_trajectories = list(raw_traj)
        raw_wrist = data.get("wrist_trajectories") or []
        wrist_trajectories = list(raw_wrist)
        try:
            wrist_motion_strength = float(data.get("wrist_motion_strength", 0.0) or 0.0)
        except (TypeError, ValueError):
            wrist_motion_strength = 0.0
        duration_mode = str(data.get("duration_mode", "") or "")
        # v1.1.8.1 — new dynamic-only fields.
        try:
            raw_mt = data.get("match_threshold")
            match_threshold = float(raw_mt) if raw_mt is not None else None
        except (TypeError, ValueError):
            match_threshold = None
        try:
            wrist_schema = int(data.get("wrist_schema", 1) or 1)
        except (TypeError, ValueError):
            wrist_schema = 1
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            created_at=str(data.get("created_at", "")),
            handedness=hand,
            image_filename=str(data.get("image_filename", "") or ""),
            action=Action.from_dict(data.get("action") or {}),
            samples=samples,
            kind=kind,
            key_point_indices=key_point_indices,
            sample_trajectories=sample_trajectories,
            wrist_trajectories=wrist_trajectories,
            wrist_motion_strength=wrist_motion_strength,
            duration_mode=duration_mode,
            match_threshold=match_threshold,
            wrist_schema=wrist_schema,
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class GestureRegistry:
    """JSON-backed store of user-defined gestures. Thread-safe for load/save
    but callers should coordinate writes to avoid lost updates.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path is not None else registry_path()
        self._lock = threading.Lock()
        self._gestures: Dict[str, CustomGesture] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        with self._lock:
            self._gestures = {}
            self._loaded = True
            if not self._path.exists():
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                # Corrupt file — start fresh rather than crashing the caller.
                # A future version can back up the broken file here.
                return
            migrated_any = False
            for entry in raw.get("gestures", []):
                try:
                    gesture = CustomGesture.from_dict(entry)
                except Exception:
                    continue
                # v1.1.8.1: migrate legacy wrist_schema=1 (absolute
                # palm-scaled position) to schema=2 (displacement from
                # first frame). Fixes the correctness bug where a
                # swipe recorded at one in-frame position couldn't
                # match the same swipe performed elsewhere. Idempotent
                # via the schema flag: schema>=2 skips.
                if (
                    gesture.kind == "dynamic"
                    and gesture.wrist_trajectories
                    and gesture.wrist_schema < 2
                ):
                    try:
                        migrated_wrist: List[List[List[float]]] = []
                        for take in gesture.wrist_trajectories:
                            if not take:
                                migrated_wrist.append(take)
                                continue
                            first = take[0]
                            new_take = []
                            for row in take:
                                new_take.append(
                                    [
                                        float(row[k]) - float(first[k])
                                        for k in range(len(row))
                                    ]
                                )
                            migrated_wrist.append(new_take)
                        # Frozen dataclass — swap in a new instance with
                        # the migrated field. Same object identity
                        # replaced in the dict below.
                        object.__setattr__(gesture, "wrist_trajectories", migrated_wrist)
                        object.__setattr__(gesture, "wrist_schema", 2)
                        migrated_any = True
                    except Exception:
                        # Bad migration → leave the record alone.
                        pass
                self._gestures[gesture.name] = gesture
            if migrated_any:
                # Persist the migration immediately so a later process
                # doesn't re-migrate. Guard against write failures —
                # in-memory migration is enough for THIS process even
                # if the disk write can't happen.
                try:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "schema_version": self._SCHEMA_VERSION,
                        "gestures": [g.to_dict() for g in self._gestures.values()],
                    }
                    tmp = self._path.with_suffix(self._path.suffix + ".tmp")
                    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    tmp.replace(self._path)
                except Exception:
                    pass

    def save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": self._SCHEMA_VERSION,
                "gestures": [g.to_dict() for g in self._gestures.values()],
            }
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._path)

    def add(
        self,
        name: str,
        samples: List[GestureSample],
        action: Action,
        *,
        description: str = "",
        overwrite: bool = False,
        handedness: Optional[str] = None,
        image_filename: str = "",
    ) -> CustomGesture:
        if not self._loaded:
            self.load()
        name = name.strip()
        if not name:
            raise ValueError("gesture name must be non-empty")
        if not samples:
            raise ValueError("gesture must have at least one sample")
        if handedness is not None and handedness not in ("Left", "Right"):
            raise ValueError(
                f"handedness must be 'Left', 'Right', or None — got {handedness!r}"
            )
        with self._lock:
            if name in self._gestures and not overwrite:
                raise ValueError(
                    f"gesture {name!r} already exists (pass overwrite=True to replace)"
                )
            gesture = CustomGesture(
                name=name,
                samples=list(samples),
                action=action,
                created_at=_utc_now_iso(),
                description=description,
                handedness=handedness,
                image_filename=str(image_filename or ""),
            )
            self._gestures[name] = gesture
        return gesture

    def add_dynamic(
        self,
        name: str,
        key_point_indices: List[int],
        sample_trajectories,  # numpy arrays or nested lists
        action: Action,
        *,
        description: str = "",
        overwrite: bool = False,
        handedness: Optional[str] = None,
        image_filename: str = "",
        duration_mode: str = "",
        wrist_trajectories=None,  # parallel iterable of (T, 3) per take
        wrist_motion_strength: float = 0.0,
        match_threshold: Optional[float] = None,
        wrist_schema: int = 2,  # v1.1.8.1 default: displacement semantics
    ) -> CustomGesture:
        """Register a dynamic gesture. `sample_trajectories` is an
        iterable of arrays/lists with shape (resampled_length,
        num_key_points, 3). We coerce each to nested lists for JSON
        serialization so callers can pass numpy arrays directly from
        the recorder.

        `wrist_trajectories` (optional) is a parallel iterable of
        (resampled_length, 3) absolute-wrist arrays — one per take —
        and `wrist_motion_strength` is the [0, _WRIST_WEIGHT_MAX]
        weight derived at template-build time. Pre-existing dynamic
        records persist without these; loading falls back to
        finger-only matching for them."""
        if not self._loaded:
            self.load()
        name = name.strip()
        if not name:
            raise ValueError("gesture name must be non-empty")
        if not key_point_indices:
            raise ValueError("dynamic gesture must have at least one key point")
        if not sample_trajectories:
            raise ValueError("dynamic gesture must have at least one sample trajectory")
        if handedness is not None and handedness not in ("Left", "Right"):
            raise ValueError(
                f"handedness must be 'Left', 'Right', or None — got {handedness!r}"
            )
        # Coerce numpy arrays → nested lists for JSON. tolist() is
        # the only numpy thing we touch here, gracefully falls back
        # for already-list inputs.
        serialized: List[List[List[List[float]]]] = []
        for traj in sample_trajectories:
            if hasattr(traj, "tolist"):
                serialized.append(traj.tolist())
            else:
                serialized.append([[[float(v) for v in coord] for coord in frame] for frame in traj])
        serialized_wrist: List[List[List[float]]] = []
        for w in (wrist_trajectories or []):
            if hasattr(w, "tolist"):
                serialized_wrist.append(w.tolist())
            else:
                serialized_wrist.append([[float(v) for v in coord] for coord in w])
        with self._lock:
            if name in self._gestures and not overwrite:
                raise ValueError(
                    f"gesture {name!r} already exists (pass overwrite=True to replace)"
                )
            gesture = CustomGesture(
                name=name,
                samples=[],
                action=action,
                created_at=_utc_now_iso(),
                description=description,
                handedness=handedness,
                image_filename=str(image_filename or ""),
                kind="dynamic",
                key_point_indices=[int(i) for i in key_point_indices],
                sample_trajectories=serialized,
                wrist_trajectories=serialized_wrist,
                wrist_motion_strength=float(wrist_motion_strength or 0.0),
                duration_mode=str(duration_mode or ""),
                match_threshold=(
                    float(match_threshold) if match_threshold is not None else None
                ),
                wrist_schema=int(wrist_schema),
            )
            self._gestures[name] = gesture
        return gesture

    def thumbnails_dir(self) -> Path:
        """Directory holding per-gesture thumbnail PNGs. Created lazily
        on first read/write."""
        d = self._path.parent / "gesture_thumbnails"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return d

    def thumbnail_path(self, gesture: "CustomGesture") -> Optional[Path]:
        """Resolve a gesture's stored thumbnail to an absolute path. Returns
        None if the gesture has no image_filename set or the file doesn't
        exist on disk."""
        if not gesture.image_filename:
            return None
        candidate = self.thumbnails_dir() / gesture.image_filename
        try:
            return candidate if candidate.exists() else None
        except Exception:
            return None

    def remove(self, name: str) -> bool:
        if not self._loaded:
            self.load()
        with self._lock:
            return self._gestures.pop(name, None) is not None

    def get(self, name: str) -> Optional[CustomGesture]:
        if not self._loaded:
            self.load()
        with self._lock:
            return self._gestures.get(name)

    def list(self) -> List[CustomGesture]:
        if not self._loaded:
            self.load()
        with self._lock:
            return list(self._gestures.values())

# Author: Konstantin Markov

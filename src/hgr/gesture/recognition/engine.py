from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Dict, Iterable

import numpy as np

from ..analysis.fingers import analyze_fingers
from ..analysis.geometry import distance
from ..analysis.hand_shape import analyze_hand_shape
from ..models import GestureCandidate, GestureFrameResult, GesturePrediction, HandReading, TrackedHand
from ...core.classifiers.static_registry import classify_static
from ...core.features.static_features import extract_static_features
from ..tracking.detector import HandDetector
from ..tracking.types import build_bounds
from .dynamic_recognizer import DynamicGestureRecognizer
from .static_recognizer import StaticGestureRecognizer


@dataclass
class _StableLabelState:
    candidate: str = "neutral"
    count: int = 0
    stable: str = "neutral"


def _swap_handedness_label(label: str) -> str:
    if label == "Right":
        return "Left"
    if label == "Left":
        return "Right"
    return label


def _swap_tracked_hand_handedness(hand: TrackedHand | None) -> TrackedHand | None:
    """Return `hand` with its Left/Right label flipped (for left-handed mode).

    Done at the single point where detection enters the engine so every
    downstream `handedness == "Right"` check across the app sees swapped
    roles without per-site edits. Non-Left/Right labels pass through.
    """
    if hand is None:
        return None
    swapped = _swap_handedness_label(hand.handedness)
    if swapped == hand.handedness:
        return hand
    return replace(hand, handedness=swapped)


class GestureRecognitionEngine:
    def __init__(
        self,
        detector: HandDetector | None = None,
        static_recognizer: StaticGestureRecognizer | None = None,
        dynamic_recognizer: DynamicGestureRecognizer | None = None,
        stable_frames_required: int = 3,
        low_fps_mode: bool = False,
    ) -> None:
        self.detector = detector or HandDetector()
        self.static_recognizer = static_recognizer or StaticGestureRecognizer()
        self.dynamic_recognizer = dynamic_recognizer or DynamicGestureRecognizer(low_fps_mode=low_fps_mode)
        self.stable_frames_required = int(stable_frames_required)
        self._stable_state = _StableLabelState()
        self._secondary_stable_state = _StableLabelState()
        self._secondary_dynamic_recognizer = DynamicGestureRecognizer(low_fps_mode=low_fps_mode)
        self._frame_index = 0
        self._last_static_scores: Dict[str, float] = {}
        self._last_dynamic_scores: Dict[str, float] = {}
        # When True, the physical Left/Right hand labels are swapped at
        # detection entry so all gesture/mouse/drawing/wheel roles move
        # to the opposite hand (left-handed mode). Set live by the app
        # from AppConfig.left_handed_mode each tick.
        self.swap_handedness = False

    def close(self) -> None:
        self.detector.close()

    def reset(self) -> None:
        self.detector.reset()
        self.dynamic_recognizer.reset()
        self._secondary_dynamic_recognizer.reset()
        self._stable_state = _StableLabelState()
        self._secondary_stable_state = _StableLabelState()
        self._last_static_scores = {}
        self._last_dynamic_scores = {}

    @property
    def last_static_scores(self) -> Dict[str, float]:
        return dict(self._last_static_scores)

    @property
    def last_dynamic_scores(self) -> Dict[str, float]:
        return dict(self._last_dynamic_scores)

    @property
    def stable_count(self) -> int:
        return int(self._stable_state.count)

    def _neutral_prediction(self) -> GesturePrediction:
        return GesturePrediction(
            raw_label="neutral",
            stable_label="neutral",
            confidence=0.0,
            candidates=tuple(),
            dynamic_label="neutral",
            dynamic_candidates=tuple(),
        )

    def _update_stable_label(
        self,
        raw_label: str,
        confidence: float,
        state: _StableLabelState | None = None,
    ) -> str:
        target = state if state is not None else self._stable_state
        if raw_label == "neutral":
            target.candidate = "neutral"
            target.count = 0
            target.stable = "neutral"
            return "neutral"
        if raw_label == target.candidate:
            target.count += 1
        else:
            target.candidate = raw_label
            target.count = 1
        if target.count >= self.stable_frames_required and confidence >= 0.56:
            target.stable = raw_label
        return target.stable

    def _volume_pose_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        spread = hand_reading.spreads["index_middle"]
        # Direct fingertip-to-fingertip distance check is the
        # decisive signal here. spread.distance is a blend
        # (0.52*tip + 0.24*dip + 0.14*pip + 0.10*skew); PIP joints
        # (landmarks 6, 10) sit anatomically close at the base of
        # the V regardless of how wide the user spreads their
        # fingers, so the blend averages down to ~0.35 even for a
        # genuinely wide V — which sneaks through any sane
        # blended-ratio threshold. Comparing landmark 8 (index
        # tip) and landmark 12 (middle tip) directly, normalized
        # by palm scale, is what the eye actually evaluates.
        #
        # Threshold: tips must be within 0.22 * palm_scale to
        # count as "together". Real volume_pose (fingers
        # touching) measures ~0.05-0.18; even ~1 cm gap pushes
        # past 0.22 and rejects.
        try:
            lm = hand_reading.landmarks
            palm_scale = max(float(hand_reading.palm.scale), 1e-6)
            tip_dx = float(lm[8][0]) - float(lm[12][0])
            tip_dy = float(lm[8][1]) - float(lm[12][1])
            tip_distance_ratio = (tip_dx * tip_dx + tip_dy * tip_dy) ** 0.5 / palm_scale
        except Exception:
            tip_distance_ratio = 1.0
        index_middle_close = (
            spread.state == "together"
            and tip_distance_ratio <= 0.22
        )
        return (
            index_middle_close
            and self._volume_primary_ready(fingers["index"])
            and self._volume_primary_ready(fingers["middle"])
            and fingers["ring"].state in {"mostly_curled", "closed"}
            and fingers["pinky"].state in {"mostly_curled", "closed"}
            and fingers["thumb"].state in {"mostly_curled", "closed", "partially_curled"}
            and fingers["ring"].openness <= 0.70
            and fingers["pinky"].openness <= 0.70
            and fingers["thumb"].openness <= 0.70
        )

    def _volume_primary_ready(self, finger) -> bool:
        return (
            finger.state == "fully_open"
            or (
                finger.state == "partially_curled"
                and finger.openness >= 0.56
                and finger.curl <= 0.52
                and finger.bend_proximal >= 122.0
                and finger.bend_distal >= 138.0
            )
        )

    def _wheel_open_ready(self, finger) -> bool:
        return (
            finger.state == "fully_open"
            or (
                finger.state == "partially_curled"
                and finger.openness >= 0.48
                and finger.curl <= 0.64
                and finger.bend_proximal >= 102.0
            )
        )

    def _mute_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        pinky = fingers["pinky"]
        # Pinky must actually be extended (shaka). Thumbs-up keeps the
        # pinky curled in the fist; older gates treated mostly_curled +
        # residual openness as "open enough" and promoted mute.
        pinky_curl = float(getattr(pinky, "curl", 0.0) or 0.0)
        # Reject closed / mostly_curled pinky (thumbs-up in a fist).
        # Real shaka often reads as partially_curled with high reach
        # rather than fully_open.
        pinky_ready = pinky.state == "fully_open" or (
            pinky.state == "partially_curled"
            and pinky.openness >= 0.56
            and pinky_curl <= 0.48
            and (
                pinky.reach >= 0.22
                or pinky.palm_distance >= 0.90
            )
        )
        thumb_ready = fingers["thumb"].state == "fully_open" or fingers["thumb"].openness >= 0.70
        core_names = ("index", "middle", "ring")
        partial_count = sum(1 for name in core_names if fingers[name].state == "partially_curled")
        strong_fold_core = sum(1 for name in core_names if fingers[name].state in {"mostly_curled", "closed"}) >= 2
        folded_core = all(
            fingers[name].state in {"partially_curled", "mostly_curled", "closed"}
            and fingers[name].openness <= 0.76
            for name in core_names
        )
        return thumb_ready and pinky_ready and folded_core and strong_fold_core and partial_count <= 1

    def _ok_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        palm_scale = max(hand_reading.palm.scale, 1e-6)
        thumb_index_ratio = distance(hand_reading.landmarks[4], hand_reading.landmarks[8]) / palm_scale
        thumb_index_side_ratio = min(
            distance(hand_reading.landmarks[4], hand_reading.landmarks[5]),
            distance(hand_reading.landmarks[4], hand_reading.landmarks[6]),
        ) / palm_scale
        pinky_open_ready = (
            fingers["pinky"].state == "fully_open"
            or (
                fingers["pinky"].openness >= 0.62
                and fingers["pinky"].bend_proximal >= 165.0
                and fingers["pinky"].bend_distal >= 168.0
                and fingers["pinky"].reach >= 0.28
            )
        )
        return (
            thumb_index_ratio <= 0.27
            and thumb_index_side_ratio >= 0.52
            and (
                fingers["index"].state in {"mostly_curled", "closed", "partially_curled"}
                or fingers["index"].openness <= 0.66
            )
            and (fingers["middle"].state == "fully_open" or fingers["middle"].openness >= 0.70)
            and (fingers["ring"].state == "fully_open" or fingers["ring"].openness >= 0.66)
            and pinky_open_ready
            and fingers["middle"].openness >= 0.70
            and fingers["ring"].openness >= 0.66
            and fingers["pinky"].openness >= 0.56
        )

    def _fist_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        non_thumb = [fingers[name] for name in ("index", "middle", "ring", "pinky")]
        closed_count = sum(1 for finger in non_thumb if finger.state == "closed")
        folded_count = sum(1 for finger in non_thumb if finger.state in {"mostly_curled", "closed"})
        base_folded_count = sum(1 for finger in non_thumb if finger.bend_base <= 140.0)
        avg_curl = sum(finger.curl for finger in non_thumb) / 4.0
        avg_reach = sum(finger.reach for finger in non_thumb) / 4.0
        thumb_folded = fingers["thumb"].state in {"mostly_curled", "closed", "partially_curled"}
        return (
            thumb_folded
            and (
                (closed_count >= 3 and folded_count == 4)
                or (
                    closed_count >= 2
                    and folded_count == 4
                    and base_folded_count >= 3
                    and avg_curl >= 0.58
                    and avg_reach <= 0.15
                )
            )
        )

    def _zero_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        palm_scale = max(hand_reading.palm.scale, 1e-6)
        thumb_index_ratio = distance(hand_reading.landmarks[4], hand_reading.landmarks[8]) / palm_scale
        thumb_index_side_ratio = min(
            distance(hand_reading.landmarks[4], hand_reading.landmarks[5]),
            distance(hand_reading.landmarks[4], hand_reading.landmarks[6]),
        ) / palm_scale
        tail_folded = all(fingers[name].state in {"mostly_curled", "closed"} for name in ("middle", "ring", "pinky"))
        classic_zero_ready = (
            fingers["index"].state in {"mostly_curled", "closed"}
            and tail_folded
            and thumb_index_ratio <= 0.30
            and thumb_index_side_ratio >= 0.70
            and fingers["thumb"].palm_distance <= 0.55
        )
        thumb_out_zero_ready = (
            all(fingers[name].state in {"mostly_curled", "closed"} for name in ("index", "middle", "ring", "pinky"))
            and fingers["thumb"].state == "fully_open"
            and fingers["thumb"].openness >= 0.70
            and fingers["thumb"].palm_distance >= 0.72
            and thumb_index_ratio >= 0.90
            and thumb_index_side_ratio >= 0.82
        )
        return classic_zero_ready or thumb_out_zero_ready

    def _four_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        return (
            all(fingers[name].state == "fully_open" for name in ("index", "middle", "ring", "pinky"))
            and fingers["thumb"].state in {"mostly_curled", "closed"}
            and fingers["thumb"].openness <= 0.48
        )

    def _one_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        thumb_folded = (
            fingers["thumb"].state in {"mostly_curled", "closed", "partially_curled"}
            and fingers["thumb"].openness <= 0.70
            and fingers["thumb"].palm_distance <= 0.62
        )
        index_ready = fingers["index"].extended or fingers["index"].openness >= 0.68
        folded_rest = all(
            fingers[name].state in {"partially_curled", "mostly_curled", "closed"}
            and fingers[name].openness <= 0.70
            for name in ("middle", "ring", "pinky")
        )
        return thumb_folded and index_ready and folded_rest

    def _two_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        thumb_folded = (
            fingers["thumb"].state in {"mostly_curled", "closed", "partially_curled"}
            and fingers["thumb"].openness <= 0.62
            and fingers["thumb"].palm_distance <= 0.58
        )
        index_ready = fingers["index"].extended or fingers["index"].openness >= 0.72
        middle_ready = fingers["middle"].extended or fingers["middle"].openness >= 0.74
        folded_rest = all(
            fingers[name].state in {"partially_curled", "mostly_curled", "closed"}
            and fingers[name].openness <= 0.70
            for name in ("ring", "pinky")
        )
        index_middle = hand_reading.spreads["index_middle"]
        separated_pair = (
            index_middle.state == "apart"
            or index_middle.distance >= 0.16
            or index_middle.apart_strength >= 0.18
        )
        return thumb_folded and index_ready and middle_ready and folded_rest and separated_pair

    def _thumb_index_family_active(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        palm_scale = max(hand_reading.palm.scale, 1e-6)
        thumb_index_ratio = distance(hand_reading.landmarks[4], hand_reading.landmarks[8]) / palm_scale
        thumb_ready = (
            fingers["thumb"].state == "fully_open"
            or (fingers["thumb"].state == "partially_curled" and fingers["thumb"].openness >= 0.32)
        )
        index_ready = fingers["index"].state in {"fully_open", "partially_curled"} and fingers["index"].openness >= 0.30
        ring_pinky_folded = all(
            fingers[name].state in {"partially_curled", "mostly_curled", "closed"}
            and fingers[name].openness <= 0.78
            for name in ("ring", "pinky")
        )
        return thumb_ready and index_ready and ring_pinky_folded and thumb_index_ratio <= 0.86

    def _thumb_index_control_pair_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        thumb_ready = (
            fingers["thumb"].state == "fully_open"
            or (fingers["thumb"].state == "partially_curled" and fingers["thumb"].openness >= 0.42)
        )
        index_ready = fingers["index"].state in {"fully_open", "partially_curled"} and fingers["index"].openness >= 0.36
        middle_not_open = not (fingers["middle"].state == "fully_open" or fingers["middle"].openness >= 0.58)
        folded_outer = sum(
            1
            for name in ("middle", "ring", "pinky")
            if fingers[name].state in {"partially_curled", "mostly_curled", "closed"} and fingers[name].openness <= 0.74
        ) >= 2
        return thumb_ready and index_ready and middle_not_open and folded_outer

    def _thumb_index_control_pinch_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        palm_scale = max(hand_reading.palm.scale, 1e-6)
        thumb_index_ratio = distance(hand_reading.landmarks[4], hand_reading.landmarks[8]) / palm_scale
        thumb_ready = (
            fingers["thumb"].state == "fully_open"
            or (fingers["thumb"].state == "partially_curled" and fingers["thumb"].openness >= 0.42)
        )
        index_ready = fingers["index"].state in {"fully_open", "partially_curled"} and fingers["index"].openness >= 0.36
        middle_not_open = not (fingers["middle"].state == "fully_open" or fingers["middle"].openness >= 0.62)
        folded_outer = sum(
            1
            for name in ("middle", "ring", "pinky")
            if fingers[name].state in {"partially_curled", "mostly_curled", "closed"} and fingers[name].openness <= 0.68
        ) >= 2
        return thumb_ready and index_ready and middle_not_open and folded_outer and thumb_index_ratio <= 0.44

    def _wheel_pose_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        thumb_index = hand_reading.spreads["thumb_index"]
        ring_pinky = hand_reading.spreads["ring_pinky"]
        return (
            self._wheel_open_ready(fingers["thumb"])
            and self._wheel_open_ready(fingers["index"])
            and self._wheel_open_ready(fingers["pinky"])
            and fingers["middle"].state in {"mostly_curled", "closed"}
            and fingers["ring"].state in {"mostly_curled", "closed"}
            and (
                thumb_index.state == "apart"
                or thumb_index.distance >= 0.38
                or thumb_index.apart_strength >= 0.32
            )
            and (
                ring_pinky.state == "apart"
                or ring_pinky.distance >= 0.14
                or ring_pinky.apart_strength >= 0.20
            )
        )

    def _chrome_wheel_pose_ready(self, hand_reading: HandReading) -> bool:
        fingers = hand_reading.fingers
        index_middle = hand_reading.spreads["index_middle"]
        ring_pinky = hand_reading.spreads["ring_pinky"]
        thumb_ready = (
            fingers["thumb"].state in {"mostly_curled", "closed", "partially_curled"}
            or (
                fingers["thumb"].state == "fully_open"
                and fingers["thumb"].openness <= 0.58
                and fingers["thumb"].palm_distance <= 0.46
            )
        )
        return (
            self._wheel_open_ready(fingers["index"])
            and self._wheel_open_ready(fingers["pinky"])
            and fingers["middle"].state in {"mostly_curled", "closed"}
            and fingers["ring"].state in {"mostly_curled", "closed"}
            and thumb_ready
            and (
                index_middle.state == "apart"
                or index_middle.distance >= 0.16
                or index_middle.apart_strength >= 0.22
            )
            and (
                ring_pinky.state == "apart"
                or ring_pinky.distance >= 0.12
                or ring_pinky.apart_strength >= 0.18
            )
        )

    def _finger_axis_vertical(self, hand_reading: HandReading, name: str) -> float:
        finger_indices = {
            "thumb": (2, 3),
            "index": (5, 6),
            "middle": (9, 10),
            "ring": (13, 14),
            "pinky": (17, 18),
        }
        base_idx, next_idx = finger_indices[name]
        base = hand_reading.landmarks[base_idx]
        next_joint = hand_reading.landmarks[next_idx]
        vector = next_joint - base
        denom = max(float(np.linalg.norm(vector[:2])), 1e-6)
        return float(base[1] - next_joint[1]) / denom

    def _gesture_not_inverted(self, hand_reading: HandReading, fingers: Iterable[str]) -> bool:
        roll = float(hand_reading.palm.roll_deg)
        if not (23.0 <= roll <= 157.0):
            return False
        finger_names = tuple(fingers)
        if not finger_names:
            return True
        tip_indices = {
            "thumb": 4,
            "index": 8,
            "middle": 12,
            "ring": 16,
            "pinky": 20,
        }
        wrist_y = float(hand_reading.landmarks[0][1])
        tolerance = max(0.015, float(hand_reading.palm.scale) * 0.06)
        heights: list[float] = []
        for name in finger_names:
            tip_y = float(hand_reading.landmarks[tip_indices[name]][1])
            heights.append(wrist_y - tip_y)
        positive = sum(1 for height in heights if height >= tolerance)
        required_positive = max(1, int(len(heights) * 0.55 + 0.499))
        return (sum(heights) / len(heights)) >= tolerance * 0.35 and positive >= required_positive

    def _gesture_fist_sideways(self, hand_reading: HandReading) -> bool:
        roll = float(hand_reading.palm.roll_deg)
        return 23.0 <= roll <= 157.0

    def _analyze_tracked_hand(
        self,
        tracked_hand: TrackedHand,
        frame_bgr: np.ndarray,
        timestamp: float,
        frame_index: int,
        *,
        stable_state: _StableLabelState | None = None,
        dynamic_recognizer: DynamicGestureRecognizer | None = None,
        record_debug: bool = True,
    ) -> GestureFrameResult:
        palm, spreads, shape_confidence = analyze_hand_shape(tracked_hand.landmarks)
        fingers = analyze_fingers(tracked_hand.landmarks, palm.center, palm.scale)
        occlusion_score = sum(1.0 - finger.confidence for finger in fingers.values()) / max(1, len(fingers))
        hand_reading = HandReading(
            handedness=tracked_hand.handedness,
            handedness_confidence=tracked_hand.handedness_confidence,
            bbox=tracked_hand.bbox,
            palm=palm,
            fingers=fingers,
            spreads=spreads,
            landmarks=tracked_hand.landmarks,
            finger_count_extended=sum(1 for finger in fingers.values() if finger.extended),
            occlusion_score=float(occlusion_score),
            shape_confidence=float(shape_confidence),
            debug_values={
                "bbox_area": tracked_hand.bbox.area,
                "pitch_deg": palm.pitch_deg,
                "yaw_deg": palm.yaw_deg,
                "roll_deg": palm.roll_deg,
            },
        )

        legacy_features = extract_static_features(tracked_hand.landmarks)
        legacy_prediction = classify_static(legacy_features)
        static_label, static_confidence, static_candidates, static_scores = self.static_recognizer.predict(hand_reading)
        merged_scores = dict(legacy_prediction.candidate_scores)
        merged_scores["zero"] = max(merged_scores.get("zero", 0.0), static_scores.get("zero", 0.0))
        merged_scores["ok"] = static_scores.get("ok", 0.0)
        merged_scores["finger_together"] = static_scores.get("finger_together", 0.0)
        merged_scores["finger_apart"] = static_scores.get("finger_apart", 0.0)
        merged_scores["volume_pose"] = max(merged_scores.get("volume_pose", 0.0), static_scores.get("volume_pose", 0.0))
        merged_scores["wheel_pose"] = max(merged_scores.get("wheel_pose", 0.0), static_scores.get("wheel_pose", 0.0))
        merged_scores["chrome_wheel_pose"] = max(
            merged_scores.get("chrome_wheel_pose", 0.0),
            static_scores.get("chrome_wheel_pose", 0.0),
        )
        ok_ready = self._ok_ready(hand_reading)
        volume_pose_ready = self._volume_pose_ready(hand_reading)
        mute_ready = self._mute_ready(hand_reading)
        fist_ready = self._fist_ready(hand_reading)
        zero_ready = self._zero_ready(hand_reading)
        four_ready = self._four_ready(hand_reading)
        one_ready = self._one_ready(hand_reading)
        two_ready = self._two_ready(hand_reading)
        control_family_active = self._thumb_index_family_active(hand_reading)
        control_pair_ready = self._thumb_index_control_pair_ready(hand_reading)
        control_pinch_ready = self._thumb_index_control_pinch_ready(hand_reading)
        wheel_pose_ready = self._wheel_pose_ready(hand_reading)
        chrome_wheel_pose_ready = self._chrome_wheel_pose_ready(hand_reading)
        orientation_ready = {
            "one": self._gesture_not_inverted(hand_reading, ("index",)),
            "two": self._gesture_not_inverted(hand_reading, ("index", "middle")),
            "three": self._gesture_not_inverted(hand_reading, ("index", "middle", "ring")),
            "four": self._gesture_not_inverted(hand_reading, ("index", "middle", "ring", "pinky")),
            "mute": self._gesture_not_inverted(hand_reading, ("pinky",)),
            "fist": self._gesture_fist_sideways(hand_reading),
            "zero": self._gesture_not_inverted(hand_reading, ()),
            "ok": self._gesture_not_inverted(hand_reading, ()),
            "volume_pose": self._gesture_not_inverted(hand_reading, ("index", "middle")),
            "wheel_pose": self._gesture_not_inverted(hand_reading, ()),
            "chrome_wheel_pose": self._gesture_not_inverted(hand_reading, ()),
        }
        if volume_pose_ready:
            merged_scores["volume_pose"] = max(merged_scores.get("volume_pose", 0.0), static_scores.get("volume_pose", 0.0), 0.58)
        if mute_ready:
            merged_scores["mute"] = max(merged_scores.get("mute", 0.0), static_scores.get("mute", 0.0), 0.56)
        if wheel_pose_ready:
            merged_scores["wheel_pose"] = max(merged_scores.get("wheel_pose", 0.0), static_scores.get("wheel_pose", 0.0), 0.58)
        if chrome_wheel_pose_ready:
            merged_scores["chrome_wheel_pose"] = max(
                merged_scores.get("chrome_wheel_pose", 0.0),
                static_scores.get("chrome_wheel_pose", 0.0),
                0.58,
            )
        fist_closed_count = sum(
            1 for name in ("index", "middle", "ring", "pinky") if hand_reading.fingers[name].state == "closed"
        )
        fist_folded_count = sum(
            1 for name in ("index", "middle", "ring", "pinky")
            if hand_reading.fingers[name].state in {"mostly_curled", "closed"}
        )
        fist_base_folded_count = sum(
            1 for name in ("index", "middle", "ring", "pinky")
            if hand_reading.fingers[name].bend_base <= 140.0
        )
        fist_thumb_folded = hand_reading.fingers["thumb"].state in {"mostly_curled", "closed", "partially_curled"}
        fist_avg_curl = sum(hand_reading.fingers[name].curl for name in ("index", "middle", "ring", "pinky")) / 4.0
        fist_avg_reach = sum(hand_reading.fingers[name].reach for name in ("index", "middle", "ring", "pinky")) / 4.0
        fist_state_heavy = fist_thumb_folded and (
            (fist_closed_count >= 3 and fist_folded_count == 4)
            or (
                fist_closed_count >= 2
                and fist_folded_count == 4
                and fist_base_folded_count >= 3
                and fist_avg_curl >= 0.58
                and fist_avg_reach <= 0.15
            )
        )
        if not one_ready:
            merged_scores["one"] *= 0.12
        if not two_ready:
            merged_scores["two"] *= 0.12
        if not ok_ready:
            merged_scores["ok"] *= 0.12
        if not volume_pose_ready:
            merged_scores["volume_pose"] *= 0.12
        if not mute_ready:
            merged_scores["mute"] *= 0.18
        if not fist_ready:
            merged_scores["fist"] *= 0.18
        if not zero_ready:
            merged_scores["zero"] *= 0.12
        if not four_ready:
            merged_scores["four"] *= 0.12
        if not wheel_pose_ready:
            merged_scores["wheel_pose"] *= 0.12
        if not chrome_wheel_pose_ready:
            merged_scores["chrome_wheel_pose"] *= 0.12
        for label, allowed in orientation_ready.items():
            if not allowed:
                merged_scores[label] *= 0.05
        actual_label = legacy_prediction.raw_gesture
        actual_confidence = legacy_prediction.confidence
        non_thumb_extended = sum(1 for name in ("index", "middle", "ring", "pinky") if hand_reading.fingers[name].extended)
        volume_primary_count = sum(1 for name in ("index", "middle") if self._volume_primary_ready(hand_reading.fingers[name]))
        if ok_ready and static_scores.get("ok", 0.0) >= 0.46:
            actual_label = "ok"
            actual_confidence = max(static_confidence, static_scores["ok"])
        elif zero_ready and static_scores.get("zero", 0.0) >= 0.42:
            actual_label = "zero"
            actual_confidence = max(static_confidence, static_scores["zero"])
        elif chrome_wheel_pose_ready and merged_scores.get("chrome_wheel_pose", 0.0) >= 0.40:
            actual_label = "chrome_wheel_pose"
            actual_confidence = max(static_confidence, merged_scores.get("chrome_wheel_pose", 0.0), 0.58)
        elif wheel_pose_ready and merged_scores.get("wheel_pose", 0.0) >= 0.38:
            actual_label = "wheel_pose"
            actual_confidence = max(static_confidence, merged_scores.get("wheel_pose", 0.0), 0.58)
        elif mute_ready and merged_scores.get("mute", 0.0) >= 0.46:
            actual_label = "mute"
            actual_confidence = max(static_confidence, merged_scores.get("mute", 0.0), 0.56)
        elif fist_state_heavy:
            actual_label = "fist"
            actual_confidence = max(
                static_confidence,
                actual_confidence,
                merged_scores.get("fist", 0.0),
                0.64,
            )
        elif (
            volume_pose_ready
            and volume_primary_count == 2
        ):
            actual_label = "volume_pose"
            actual_confidence = max(static_confidence, merged_scores.get("volume_pose", 0.0), 0.58)
        elif (
            two_ready
            and static_label == "two"
            and non_thumb_extended == 2
            and static_scores.get("two", 0.0) >= 0.58
        ):
            actual_label = "two"
            actual_confidence = max(static_confidence, legacy_prediction.confidence * 0.96)
        elif legacy_prediction.raw_gesture == "neutral":
            actual_label = static_label
            actual_confidence = static_confidence
        elif static_label != "neutral" and static_confidence >= legacy_prediction.confidence + 0.12:
            actual_label = static_label
            actual_confidence = static_confidence

        if actual_label == "volume_pose" and not volume_pose_ready:
            actual_label = static_label if static_label != "volume_pose" and static_confidence >= 0.56 else "neutral"
            actual_confidence = static_confidence if actual_label != "neutral" else max(static_confidence, actual_confidence) * 0.55
        if actual_label == "mute" and not mute_ready:
            actual_label = static_label if static_label != "mute" and static_confidence >= 0.56 else "neutral"
            actual_confidence = static_confidence if actual_label != "neutral" else max(static_confidence, actual_confidence) * 0.55
        if actual_label == "fist" and not fist_ready:
            actual_label = static_label if static_label != "fist" and static_confidence >= 0.56 else "neutral"
            actual_confidence = static_confidence if actual_label != "neutral" else max(static_confidence, actual_confidence) * 0.55
        if actual_label == "ok" and not ok_ready:
            actual_label = static_label if static_label != "ok" and static_confidence >= 0.56 else "neutral"
            actual_confidence = static_confidence if actual_label != "neutral" else max(static_confidence, actual_confidence) * 0.55
        if actual_label == "zero" and not zero_ready:
            actual_label = static_label if static_label != "zero" and static_confidence >= 0.56 else "neutral"
            actual_confidence = static_confidence if actual_label != "neutral" else max(static_confidence, actual_confidence) * 0.55
        if actual_label == "four" and not four_ready:
            actual_label = static_label if static_label != "four" and static_confidence >= 0.56 else "neutral"
            actual_confidence = static_confidence if actual_label != "neutral" else max(static_confidence, actual_confidence) * 0.55
        if actual_label == "one" and (not one_ready or control_family_active or control_pair_ready or control_pinch_ready):
            actual_label = "neutral"
            actual_confidence *= 0.30
        if actual_label == "two" and (not two_ready or control_family_active or control_pair_ready or control_pinch_ready):
            actual_label = "neutral"
            actual_confidence *= 0.30
        if actual_label == "wheel_pose" and not wheel_pose_ready:
            actual_label = static_label if static_label != "wheel_pose" and static_confidence >= 0.56 else "neutral"
            actual_confidence = static_confidence if actual_label != "neutral" else max(static_confidence, actual_confidence) * 0.55
        if actual_label == "chrome_wheel_pose" and not chrome_wheel_pose_ready:
            actual_label = (
                static_label
                if static_label != "chrome_wheel_pose" and static_confidence >= 0.56
                else "neutral"
            )
            actual_confidence = static_confidence if actual_label != "neutral" else max(static_confidence, actual_confidence) * 0.55
        if actual_label in orientation_ready and not orientation_ready[actual_label]:
            actual_label = "neutral"
            actual_confidence *= 0.30
        if actual_label == "open_hand":
            actual_label = "neutral"
            actual_confidence = max(static_confidence, actual_confidence) * 0.40

        static_candidates = tuple(
            sorted(
                (GestureCandidate(label, score, "static") for label, score in merged_scores.items()),
                key=lambda item: item.score,
                reverse=True,
            )[:6]
        )
        active_dynamic = dynamic_recognizer if dynamic_recognizer is not None else self.dynamic_recognizer
        dynamic_label, dynamic_candidates, dynamic_scores = active_dynamic.update(hand_reading, timestamp)
        dynamic_orientation_ready = self._gesture_not_inverted(hand_reading, ())
        if not dynamic_orientation_ready:
            dynamic_label = "neutral"
            dynamic_scores = {label: 0.0 for label in dynamic_scores}
            dynamic_candidates = tuple(
                GestureCandidate(candidate.label, 0.0, candidate.kind)
                for candidate in dynamic_candidates
            )
        raw_label = dynamic_label if dynamic_label != "neutral" else actual_label
        raw_confidence = max(
            actual_confidence,
            dynamic_candidates[0].score if dynamic_candidates else 0.0,
        )
        stable_label = self._update_stable_label(raw_label, raw_confidence, stable_state)
        if record_debug:
            self._last_static_scores = merged_scores
            self._last_dynamic_scores = dynamic_scores
        prediction = GesturePrediction(
            raw_label=raw_label,
            stable_label=stable_label,
            confidence=float(raw_confidence),
            candidates=static_candidates,
            dynamic_label=dynamic_label,
            dynamic_candidates=dynamic_candidates,
        )
        return GestureFrameResult(
            found=True,
            frame_index=frame_index,
            tracked_hand=tracked_hand,
            hand_reading=hand_reading,
            prediction=prediction,
            # Reference, not copy. Was `.copy()` here, which paired
            # with another `.copy()` in draw_hand_overlay to spend
            # ~6-8 ms per gesturing frame at 720p BGR for two
            # successive copies of the exact same buffer. The
            # downstream consumers either copy it themselves
            # (tutorial / debug windows) or pass it to
            # draw_hand_overlay which now mutates in place, so the
            # caller-side ownership story is unchanged but the per
            # frame cost drops measurably.
            annotated_frame=frame_bgr,
        )

    def neutral_result_for_frame(self, frame_bgr: np.ndarray) -> GestureFrameResult:
        """Synthetic 'no hand' result used by skip-frame inference.

        Returns the same shape `process_frame` returns when the detector
        finds no hand, but without actually invoking the detector — so
        the caller can advertise "this tick saw no hand" while skipping
        the 12-25 ms MediaPipe inference cost. Also resets the engine's
        recognizer state so the next real inference call starts fresh,
        matching the contract `process_frame` enforces on its no-hand
        branch.
        """
        frame_index = self._frame_index
        self._frame_index += 1
        self.reset()
        return GestureFrameResult(
            found=False,
            frame_index=frame_index,
            tracked_hand=None,
            hand_reading=None,
            prediction=self._neutral_prediction(),
            annotated_frame=frame_bgr,
        )

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> GestureFrameResult:
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        detection = self.detector.process(frame_bgr)
        frame_index = self._frame_index
        self._frame_index += 1
        # Single-point handedness swap for left-handed mode: flip the
        # Left/Right labels here, before any analysis, so the rest of
        # the pipeline (and every downstream role check) is consistent.
        primary_hand = detection.tracked_hand
        secondary_hand = detection.secondary_hand
        if self.swap_handedness:
            primary_hand = _swap_tracked_hand_handedness(primary_hand)
            secondary_hand = _swap_tracked_hand_handedness(secondary_hand)
        if primary_hand is None:
            self.reset()
            return GestureFrameResult(
                found=False,
                frame_index=frame_index,
                tracked_hand=None,
                hand_reading=None,
                prediction=self._neutral_prediction(),
                annotated_frame=detection.frame_bgr,
            )
        primary_result = self._analyze_tracked_hand(primary_hand, detection.frame_bgr, timestamp, frame_index)
        if secondary_hand is None:
            self._secondary_stable_state = _StableLabelState()
            self._secondary_dynamic_recognizer.reset()
            return primary_result
        secondary_result = self._analyze_tracked_hand(
            secondary_hand,
            detection.frame_bgr,
            timestamp,
            frame_index,
            stable_state=self._secondary_stable_state,
            dynamic_recognizer=self._secondary_dynamic_recognizer,
            record_debug=False,
        )
        return GestureFrameResult(
            found=primary_result.found,
            frame_index=primary_result.frame_index,
            tracked_hand=primary_result.tracked_hand,
            hand_reading=primary_result.hand_reading,
            prediction=primary_result.prediction,
            annotated_frame=primary_result.annotated_frame,
            secondary_tracked_hand=secondary_hand,
            secondary_hand_reading=secondary_result.hand_reading,
            secondary_prediction=secondary_result.prediction,
        )

    def _build_secondary_reading(self, tracked_hand: TrackedHand) -> HandReading:
        palm, spreads, shape_confidence = analyze_hand_shape(tracked_hand.landmarks)
        fingers = analyze_fingers(tracked_hand.landmarks, palm.center, palm.scale)
        occlusion_score = sum(1.0 - finger.confidence for finger in fingers.values()) / max(1, len(fingers))
        return HandReading(
            handedness=tracked_hand.handedness,
            handedness_confidence=tracked_hand.handedness_confidence,
            bbox=tracked_hand.bbox,
            palm=palm,
            fingers=fingers,
            spreads=spreads,
            landmarks=tracked_hand.landmarks,
            finger_count_extended=sum(1 for finger in fingers.values() if finger.extended),
            occlusion_score=float(occlusion_score),
            shape_confidence=float(shape_confidence),
            debug_values={
                "bbox_area": tracked_hand.bbox.area,
                "pitch_deg": palm.pitch_deg,
                "yaw_deg": palm.yaw_deg,
                "roll_deg": palm.roll_deg,
            },
        )

    def process_landmarks(
        self,
        landmarks: np.ndarray,
        frame_bgr: np.ndarray | None = None,
        handedness: str | None = None,
        *,
        handedness_confidence: float = 1.0,
        timestamp: float | None = None,
    ) -> GestureFrameResult:
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        frame_index = self._frame_index
        self._frame_index += 1
        tracked_hand = TrackedHand(
            landmarks=landmarks.astype(np.float32),
            handedness=handedness or "Unknown",
            handedness_confidence=float(handedness_confidence),
            bbox=build_bounds(landmarks.astype(np.float32)),
        )
        frame = frame_bgr if frame_bgr is not None else np.zeros((1, 1, 3), dtype=np.uint8)
        return self._analyze_tracked_hand(tracked_hand, frame, timestamp, frame_index)

# Author: Konstantin Markov

from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

from ..models import TrackedHand
from .face_filter import FaceExclusionFilter
from .runtime import HandRuntime, load_hand_runtime
from .smoothing import AdaptiveLandmarkSmoother
from .types import build_bounds


@dataclass(frozen=True)
class DetectionResult:
    tracked_hand: TrackedHand | None
    frame_bgr: np.ndarray
    secondary_hand: TrackedHand | None = None


class HandDetector:
    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.72,
        min_tracking_confidence: float = 0.72,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        miss_tolerance_seconds: float = 0.0,
        max_process_width: int = 0,
        smoother: AdaptiveLandmarkSmoother | None = None,
        secondary_smoother: AdaptiveLandmarkSmoother | None = None,
        prefer_gpu: bool = False,
    ) -> None:
        # prefer_gpu plumbs the user's GPU Mode toggle through to
        # the runtime loader; the loader honours it best-effort and
        # falls back to CPU MediaPipe if no GPU path is reachable.
        # Keeping this as a constructor arg (not a global state read)
        # so the engine builder in noop_engine can rebuild the
        # detector cleanly when the user toggles GPU Mode mid-session,
        # same way it already does for low_fps_mode + lite_mode.
        self.runtime: HandRuntime = load_hand_runtime(prefer_gpu=prefer_gpu)
        self.model_complexity = int(model_complexity)
        self.miss_tolerance_seconds = max(0.0, float(miss_tolerance_seconds))
        self.max_process_width = max(0, int(max_process_width))
        self.hands = self.runtime.hands_module.Hands(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.smoother = smoother or AdaptiveLandmarkSmoother()
        self.secondary_smoother = secondary_smoother or AdaptiveLandmarkSmoother()
        # Conservative, fail-safe filter that drops "hands" the model
        # latched onto a face. No-op on any error so detection is never
        # degraded; disable with env TOUCHLESS_FACE_FILTER=0.
        self._face_filter = FaceExclusionFilter()
        self._last_primary_hand: TrackedHand | None = None
        self._last_primary_seen_at = 0.0
        self._last_secondary_hand: TrackedHand | None = None
        self._last_secondary_seen_at = 0.0

    def close(self) -> None:
        self.hands.close()
        self._face_filter.close()

    def reset(self) -> None:
        self.smoother.reset()
        self.secondary_smoother.reset()
        self._last_primary_hand = None
        self._last_primary_seen_at = 0.0
        self._last_secondary_hand = None
        self._last_secondary_seen_at = 0.0

    def _no_detection_result(self, frame: np.ndarray) -> DetectionResult:
        """Result for a frame with no usable hand — honours the existing
        miss-tolerance bridge (return the last hand briefly) or resets.
        Shared by the raw no-hands case and the case where the face
        filter removes every candidate."""
        now = time.monotonic()
        primary_recent = (
            self._last_primary_hand is not None
            and self.miss_tolerance_seconds > 0.0
            and (now - self._last_primary_seen_at) <= self.miss_tolerance_seconds
        )
        secondary_recent = (
            self._last_secondary_hand is not None
            and self.miss_tolerance_seconds > 0.0
            and (now - self._last_secondary_seen_at) <= self.miss_tolerance_seconds
        )
        if primary_recent:
            return DetectionResult(
                tracked_hand=self._last_primary_hand,
                frame_bgr=frame,
                secondary_hand=self._last_secondary_hand if secondary_recent else None,
            )
        self.reset()
        return DetectionResult(tracked_hand=None, frame_bgr=frame, secondary_hand=None)

    def process(self, frame_bgr: np.ndarray) -> DetectionResult:
        frame = frame_bgr.copy()
        processing_frame = frame_bgr
        try:
            height, width = frame_bgr.shape[:2]
        except Exception:
            height, width = 0, 0
        if self.max_process_width > 0 and width > self.max_process_width and height > 0:
            scaled_height = max(1, int(round(height * (self.max_process_width / float(width)))))
            processing_frame = cv2.resize(
                frame_bgr,
                (self.max_process_width, scaled_height),
                interpolation=cv2.INTER_LINEAR,
            )
        rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not getattr(result, "multi_hand_landmarks", None):
            return self._no_detection_result(frame)

        hand_entries: list[tuple[np.ndarray, str, float]] = []
        multi_landmarks = result.multi_hand_landmarks
        multi_handedness = getattr(result, "multi_handedness", None) or []
        for idx, hand_landmarks in enumerate(multi_landmarks):
            raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            label = "Unknown"
            score = 0.0
            if idx < len(multi_handedness):
                try:
                    classification = multi_handedness[idx].classification[0]
                    label = str(getattr(classification, "label", "Unknown") or "Unknown")
                    score = float(getattr(classification, "score", 0.0))
                except Exception:
                    label = "Unknown"
                    score = 0.0
            hand_entries.append((raw, label, score))

        # Drop any "hand" the model latched onto a face. Runs on the same
        # RGB frame the hand model saw, so landmark and face boxes share
        # one normalized coordinate space. Fail-safe: returns the list
        # unchanged on any error. If it removes the only candidate, treat
        # the frame as a no-detection so nothing downstream acts on a face.
        hand_entries = self._face_filter.filter(rgb, hand_entries)
        if not hand_entries:
            return self._no_detection_result(frame)

        if len(hand_entries) >= 2:
            centers = [(float(raw[:, 0].mean()), float(raw[:, 1].mean())) for raw, _, _ in hand_entries]
            keep = [True] * len(hand_entries)
            for i in range(len(hand_entries)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(hand_entries)):
                    if not keep[j]:
                        continue
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    if (dx * dx + dy * dy) ** 0.5 < 0.09:
                        if hand_entries[i][2] >= hand_entries[j][2]:
                            keep[j] = False
                        else:
                            keep[i] = False
                            break
            hand_entries = [hand_entries[i] for i in range(len(hand_entries)) if keep[i]]
            centers = [centers[i] for i in range(len(centers)) if keep[i]]
            if (
                len(hand_entries) == 2
                and self._last_primary_hand is not None
                and self._last_secondary_hand is None
            ):
                scores = [float(entry[2]) for entry in hand_entries]
                weak_idx = 0 if scores[0] < scores[1] else 1
                strong_idx = 1 - weak_idx
                try:
                    last_primary_x = float(self._last_primary_hand.landmarks[:, 0].mean())
                except Exception:
                    last_primary_x = None
                weak_far = last_primary_x is None or abs(centers[weak_idx][0] - last_primary_x) > 0.35
                if scores[weak_idx] < 0.6 and weak_far:
                    hand_entries = [hand_entries[strong_idx]]
                    centers = [centers[strong_idx]]
        else:
            centers = [(float(raw[:, 0].mean()), float(raw[:, 1].mean())) for raw, _, _ in hand_entries]

        if len(hand_entries) >= 1:
            history_entries: list[tuple[str, str, float]] = []
            if self._last_primary_hand is not None:
                try:
                    history_entries.append((
                        "primary",
                        str(self._last_primary_hand.handedness),
                        float(self._last_primary_hand.landmarks[:, 0].mean()),
                    ))
                except Exception:
                    pass
            if self._last_secondary_hand is not None:
                try:
                    history_entries.append((
                        "secondary",
                        str(self._last_secondary_hand.handedness),
                        float(self._last_secondary_hand.landmarks[:, 0].mean()),
                    ))
                except Exception:
                    pass

            centers_x = [cx for cx, _ in centers]
            resolved_labels: list[str | None] = [None] * len(hand_entries)
            used_history: set[str] = set()

            MP_TRUST_THRESHOLD = 0.82
            for i in range(len(hand_entries)):
                mp_label = hand_entries[i][1]
                mp_score = float(hand_entries[i][2])
                if mp_label in ("Left", "Right") and mp_score >= MP_TRUST_THRESHOLD:
                    resolved_labels[i] = mp_label
                    best_slot: str | None = None
                    best_dist = 0.35
                    for slot, _label, hist_x in history_entries:
                        if slot in used_history:
                            continue
                        dist = abs(centers_x[i] - hist_x)
                        if dist <= best_dist:
                            best_dist = dist
                            best_slot = slot
                    if best_slot is not None:
                        used_history.add(best_slot)

            match_order = sorted(
                [i for i in range(len(hand_entries)) if resolved_labels[i] is None],
                key=lambda i: -float(hand_entries[i][2]),
            )
            for i in match_order:
                best_slot = None
                best_label: str | None = None
                best_dist = 0.35
                for slot, label, hist_x in history_entries:
                    if slot in used_history:
                        continue
                    dist = abs(centers_x[i] - hist_x)
                    if dist <= best_dist:
                        best_dist = dist
                        best_slot = slot
                        best_label = label
                if best_slot is not None and best_label is not None:
                    resolved_labels[i] = best_label
                    used_history.add(best_slot)

            for i in range(len(hand_entries)):
                if resolved_labels[i] is not None:
                    continue
                mp_label = hand_entries[i][1]
                resolved_labels[i] = mp_label if mp_label in ("Left", "Right") else ("Right" if centers_x[i] >= 0.5 else "Left")

            if len(hand_entries) >= 2 and resolved_labels[0] == resolved_labels[1]:
                idx_hi = 0 if centers_x[0] >= centers_x[1] else 1
                idx_lo = 1 - idx_hi
                resolved_labels[idx_hi] = "Right"
                resolved_labels[idx_lo] = "Left"

            hand_entries = [
                (hand_entries[i][0], resolved_labels[i] or hand_entries[i][1], max(float(hand_entries[i][2]), 0.5))
                for i in range(len(hand_entries))
            ]

        primary_index = 0
        for idx, (_, label, _) in enumerate(hand_entries):
            if label == "Right":
                primary_index = idx
                break

        primary_raw, primary_label, primary_score = hand_entries[primary_index]
        primary_landmarks = self.smoother.update(primary_raw)
        primary_hand = TrackedHand(
            landmarks=primary_landmarks,
            handedness=primary_label,
            handedness_confidence=primary_score,
            bbox=build_bounds(primary_landmarks),
        )
        self._last_primary_hand = primary_hand
        self._last_primary_seen_at = time.monotonic()

        secondary_hand: TrackedHand | None = None
        secondary_entry = next(((raw, label, score) for idx, (raw, label, score) in enumerate(hand_entries) if idx != primary_index), None)
        if secondary_entry is not None:
            sec_raw, sec_label, sec_score = secondary_entry
            sec_landmarks = self.secondary_smoother.update(sec_raw)
            secondary_hand = TrackedHand(
                landmarks=sec_landmarks,
                handedness=sec_label,
                handedness_confidence=sec_score,
                bbox=build_bounds(sec_landmarks),
            )
            self._last_secondary_hand = secondary_hand
            self._last_secondary_seen_at = time.monotonic()
        else:
            self.secondary_smoother.reset()
            self._last_secondary_hand = None
            self._last_secondary_seen_at = 0.0

        return DetectionResult(tracked_hand=primary_hand, frame_bgr=frame, secondary_hand=secondary_hand)

# Author: Konstantin Markov

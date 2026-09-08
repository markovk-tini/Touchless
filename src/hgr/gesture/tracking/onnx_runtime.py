"""GPU-accelerated hand tracking via ONNX Runtime + DirectML.

This is the real Windows GPU path. It runs MediaPipe's palm
detector and hand-landmark models converted to ONNX (sourced from
OpenCV Zoo, same weights as `mediapipe.solutions.hands` so accuracy
is identical) on `onnxruntime-directml` — DirectX 12 acceleration
that works on any modern GPU (NVIDIA / AMD / Intel) without
vendor-specific runtimes.

Why this module exists:
  - MediaPipe's official Tasks-API GPU delegate raises
    NotImplementedError on Windows (BaseOptions.to_pb2 in the pip
    wheel). So Tasks-API GPU is a non-starter on Windows.
  - onnxruntime-directml ships a DML execution provider that any
    DX12 GPU can use, no special drivers needed beyond
    Windows 10 + recent NVIDIA / AMD / Intel.

Module surface mirrors `mediapipe.solutions.hands` so HandDetector
doesn't notice which path it's running on:

    runtime = build_onnx_directml_runtime()  # returns _HandsModuleShim or None
    hands = runtime.Hands(static_image_mode=False, max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    result = hands.process(rgb_frame)
    # result.multi_hand_landmarks: List[NormalizedLandmarkList]
    # result.multi_handedness: List[ClassificationList]
    hands.close()

Pipeline reference: OpenCV Zoo's mp_palmdet.py and mp_handpose.py
(BSD-licensed). We adapt their preprocess / decode / postprocess
math to onnxruntime instead of cv2.dnn.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------
# Asset resolution — find the .onnx + anchor .npy files whether
# we're running from source (assets/models/ at repo root) or from
# a PyInstaller bundle (sys._MEIPASS / assets / models /).
# ---------------------------------------------------------------

PALM_DETECTOR_FILENAME = "palm_detector.onnx"
HAND_LANDMARK_FILENAME = "hand_landmark.onnx"
PALM_ANCHORS_FILENAME = "palm_anchors.npy"


def _resolve_asset(name: str) -> Path | None:
    candidates: list[Path] = []
    try:
        from ...utils.runtime_paths import app_base_path

        candidates.append(app_base_path() / "assets" / "models" / name)
    except Exception:
        pass
    try:
        repo_root = Path(__file__).resolve().parents[4]
        candidates.append(repo_root / "assets" / "models" / name)
    except Exception:
        pass
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except Exception:
            continue
    return None


# ---------------------------------------------------------------
# Duck-typed result wrappers (same shape solutions.hands returns).
# HandDetector reads .x/.y/.z on landmarks and .label/.score on
# the [0]th classification, so this is the entire interface we
# need to satisfy.
# ---------------------------------------------------------------


class _LandmarkPoint:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _NormalizedLandmarkList:
    __slots__ = ("landmark",)

    def __init__(self, points: list[_LandmarkPoint]) -> None:
        self.landmark = points


class _Classification:
    __slots__ = ("label", "score")

    def __init__(self, label: str, score: float) -> None:
        self.label = label
        self.score = float(score)


class _ClassificationList:
    __slots__ = ("classification",)

    def __init__(self, label: str, score: float) -> None:
        self.classification = [_Classification(label, score)]


class _HandsResult:
    __slots__ = ("multi_hand_landmarks", "multi_handedness")

    def __init__(
        self,
        landmarks: list[_NormalizedLandmarkList],
        handedness: list[_ClassificationList],
    ) -> None:
        self.multi_hand_landmarks = landmarks
        self.multi_handedness = handedness


# ---------------------------------------------------------------
# Palm detector — runs the 192x192 ONNX model and decodes anchors
# into palm bboxes + the 7 keypoints the landmark detector needs
# for its rotation-aligned crop.
# ---------------------------------------------------------------


class _OnnxPalmDetector:
    """Adapted from OpenCV Zoo mp_palmdet.py (BSD-3) — same model,
    same preprocess + decode math, but uses onnxruntime + DML
    instead of cv2.dnn. Returns a list of palm dicts with keys
    `bbox` (xyxy float pixels in the input image), `keypoints`
    (7×2 float pixels), and `score` (sigmoid prob 0-1)."""

    INPUT_SIZE = 192  # the ONNX model's expected square input

    def __init__(self, session, anchors: np.ndarray, *, score_threshold: float, nms_threshold: float, top_k: int) -> None:
        self._session = session
        self._anchors = anchors
        self._score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)
        self._top_k = int(top_k)
        self._input_name = session.get_inputs()[0].name
        # Inference timing samples (seconds). Logged externally; we
        # just collect here so the runner-thread diagnostic can split
        # palm-detect vs hand-landmark cost. Critical for diagnosing
        # silent DML-to-CPU per-op fallback inside an InferenceSession.
        self._last_inference_seconds: float = 0.0

    def detect(
        self,
        rgb_frame: np.ndarray,
        *,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        h, w = rgb_frame.shape[:2]
        blob, pad_bias, ratio = self._preprocess(rgb_frame)
        _t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: blob})
        self._last_inference_seconds = time.perf_counter() - _t0
        # outputs ordering matches the ONNX model: Identity (boxes
        # + keypoint deltas), Identity_1 (scores).
        return self._postprocess(
            outputs,
            np.array([w, h]),
            pad_bias,
            ratio,
            score_threshold=score_threshold,
        )

    def _preprocess(self, rgb_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        # Letterbox — preserve aspect ratio, pad shorter side with black.
        h, w = rgb_frame.shape[:2]
        ratio = min(self.INPUT_SIZE / h, self.INPUT_SIZE / w)
        new_h = int(round(h * ratio))
        new_w = int(round(w * ratio))
        if new_w != w or new_h != h:
            resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = rgb_frame
        pad_h = self.INPUT_SIZE - new_h
        pad_w = self.INPUT_SIZE - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # /255 normalisation; no mean subtraction. NHWC float32 with
        # leading batch dimension. Use a fresh array (the ONNX
        # session may keep a reference).
        blob = padded.astype(np.float32, copy=False) / 255.0
        blob = np.expand_dims(blob, 0)
        # pad bias is in *output-space* pixels of the original
        # image — divide back through the resize ratio.
        pad_bias = np.array([left / ratio, top / ratio], dtype=np.float32)
        return blob, pad_bias, ratio

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        original_wh: np.ndarray,
        pad_bias: np.ndarray,
        ratio: float,
        *,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        threshold = (
            self._score_threshold if score_threshold is None else float(score_threshold)
        )
        scores_raw = outputs[1][0, :, 0]
        deltas = outputs[0][0]
        # Sigmoid the logits.
        # Clip first to avoid overflow on extreme negatives that
        # never decode to a real palm anyway.
        scores_raw = np.clip(scores_raw, -50.0, 50.0)
        scores = 1.0 / (1.0 + np.exp(-scores_raw.astype(np.float64)))

        # bbox + keypoint deltas in *input-space* pixels.
        cxy_delta = deltas[:, :2] / float(self.INPUT_SIZE)
        wh_delta = deltas[:, 2:4] / float(self.INPUT_SIZE)
        kp_delta = deltas[:, 4:].reshape(-1, 7, 2) / float(self.INPUT_SIZE)

        # In input-space [0, 1] coordinates: bbox = cxy + anchor ± wh/2.
        scale = float(np.max(original_wh))  # max(W, H) of original frame
        xy1 = (cxy_delta - wh_delta / 2.0 + self._anchors) * scale
        xy2 = (cxy_delta + wh_delta / 2.0 + self._anchors) * scale
        boxes = np.concatenate([xy1, xy2], axis=1)
        boxes -= np.array([pad_bias[0], pad_bias[1], pad_bias[0], pad_bias[1]], dtype=np.float32)

        # NMS via cv2.dnn.NMSBoxes — boxes wants xywh.
        nms_boxes = np.column_stack([
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        ])
        keep = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(),
            scores.tolist(),
            threshold,
            self._nms_threshold,
            top_k=self._top_k,
        )
        if len(keep) == 0:
            return []
        keep_idx = np.array(keep).flatten()

        results: list[dict[str, Any]] = []
        for idx in keep_idx:
            kp = kp_delta[idx] + self._anchors[idx]
            kp = kp * scale - pad_bias
            results.append({
                "bbox": boxes[idx].astype(np.float32),
                "keypoints": kp.astype(np.float32),
                "score": float(scores[idx]),
            })
        return results


# ---------------------------------------------------------------
# Hand landmark detector — takes a palm dict from the detector,
# crops + rotates the original frame to a 224x224 ROI aligned to
# hand orientation, runs landmark inference, and unrotates the 21
# landmarks back into image coordinates.
# ---------------------------------------------------------------


class _OnnxHandLandmarker:
    """Adapted from OpenCV Zoo mp_handpose.py — same ROI math.
    Returns either None (low confidence) or a dict with keys
    `landmarks` (21×3 image-space pixels), `handedness` ("Left"/
    "Right"), `handedness_score`, and `presence_score`."""

    INPUT_SIZE = 224
    PALM_LANDMARKS_INDEX_OF_PALM_BASE = 0
    PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE = 2
    PALM_BOX_PRE_ENLARGE_FACTOR = 4.0
    PALM_BOX_PRE_SHIFT_VECTOR = (0.0, 0.0)
    PALM_BOX_ENLARGE_FACTOR = 3.0
    PALM_BOX_SHIFT_VECTOR = (0.0, -0.4)

    def __init__(self, session, *, conf_threshold: float) -> None:
        self._session = session
        self._conf_threshold = float(conf_threshold)
        self._input_name = session.get_inputs()[0].name
        # See _OnnxPalmDetector — same timing instrumentation, used
        # to attribute hand-landmark inference cost separately from
        # palm-detect.
        self._last_inference_seconds: float = 0.0

    def detect(self, rgb_frame: np.ndarray, palm: dict[str, Any]) -> dict[str, Any] | None:
        crop, rotated_palm_bbox, angle, rotation_matrix, pad_bias = self._preprocess(
            rgb_frame, palm
        )
        if crop is None:
            return None
        _t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: crop})
        self._last_inference_seconds = time.perf_counter() - _t0
        return self._postprocess(outputs, rotated_palm_bbox, angle, rotation_matrix, pad_bias)

    def _crop_and_pad_from_palm(
        self,
        image: np.ndarray,
        palm_bbox: np.ndarray,
        *,
        for_rotation: bool,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        wh_palm_bbox = palm_bbox[1] - palm_bbox[0]
        if for_rotation:
            shift_vec = np.asarray(self.PALM_BOX_PRE_SHIFT_VECTOR, dtype=np.float32)
            enlarge = self.PALM_BOX_PRE_ENLARGE_FACTOR
        else:
            shift_vec = np.asarray(self.PALM_BOX_SHIFT_VECTOR, dtype=np.float32)
            enlarge = self.PALM_BOX_ENLARGE_FACTOR
        palm_bbox = palm_bbox + shift_vec * wh_palm_bbox
        center = np.sum(palm_bbox, axis=0) / 2.0
        wh_palm_bbox = palm_bbox[1] - palm_bbox[0]
        new_half = wh_palm_bbox * enlarge / 2.0
        palm_bbox = np.array(
            [center - new_half, center + new_half],
            dtype=np.float32,
        )
        palm_bbox = palm_bbox.astype(np.int32)
        palm_bbox[:, 0] = np.clip(palm_bbox[:, 0], 0, image.shape[1])
        palm_bbox[:, 1] = np.clip(palm_bbox[:, 1], 0, image.shape[0])
        crop = image[palm_bbox[0][1]:palm_bbox[1][1], palm_bbox[0][0]:palm_bbox[1][0], :]
        if crop.size == 0:
            return None, palm_bbox, np.zeros(2, dtype=np.int32)
        if for_rotation:
            side_len = int(np.linalg.norm(crop.shape[:2]))
        else:
            side_len = int(max(crop.shape[:2]))
        pad_h = side_len - crop.shape[0]
        pad_w = side_len - crop.shape[1]
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        padded = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        bias = palm_bbox[0] - np.array([left, top], dtype=np.int32)
        return padded, palm_bbox, bias

    def _preprocess(
        self, rgb_frame: np.ndarray, palm: dict[str, Any]
    ) -> tuple[np.ndarray | None, np.ndarray, float, np.ndarray, np.ndarray]:
        pad_bias_total = np.array([0, 0], dtype=np.int32)
        palm_bbox = palm["bbox"].reshape(2, 2).astype(np.float32)
        cropped, palm_bbox, bias = self._crop_and_pad_from_palm(rgb_frame, palm_bbox, for_rotation=True)
        if cropped is None:
            return None, palm_bbox, 0.0, np.zeros((2, 3), dtype=np.float32), pad_bias_total
        pad_bias_total = pad_bias_total + bias
        # Rotation angle from palm landmarks.
        landmarks = palm["keypoints"] - pad_bias_total
        p1 = landmarks[self.PALM_LANDMARKS_INDEX_OF_PALM_BASE]
        p2 = landmarks[self.PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE]
        radians = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        radians = radians - 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = float(np.rad2deg(radians))
        # Rotate cropped image around its palm-bbox center.
        local_palm_bbox = palm_bbox - pad_bias_total
        center = np.sum(local_palm_bbox, axis=0) / 2.0
        rotation_matrix = cv2.getRotationMatrix2D(tuple(center.astype(np.float32)), angle, 1.0)
        rotated = cv2.warpAffine(cropped, rotation_matrix, (cropped.shape[1], cropped.shape[0]))
        # Recompute palm bbox in rotated frame.
        ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
        homog = np.concatenate([landmarks.astype(np.float32), ones], axis=1)
        rotated_palm_landmarks = np.array([
            np.dot(homog, rotation_matrix[0]),
            np.dot(homog, rotation_matrix[1]),
        ])
        rotated_palm_bbox = np.array([
            np.amin(rotated_palm_landmarks, axis=1),
            np.amax(rotated_palm_landmarks, axis=1),
        ], dtype=np.float32)
        # Tight crop + pad to square + resize to 224.
        crop2, rotated_palm_bbox, _ = self._crop_and_pad_from_palm(
            rotated, rotated_palm_bbox, for_rotation=False
        )
        if crop2 is None or crop2.size == 0:
            return None, rotated_palm_bbox, angle, rotation_matrix, pad_bias_total
        blob = cv2.resize(crop2, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_AREA)
        blob = blob.astype(np.float32, copy=False) / 255.0
        blob = np.expand_dims(blob, 0)
        return blob, rotated_palm_bbox, angle, rotation_matrix, pad_bias_total

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        rotated_palm_bbox: np.ndarray,
        angle: float,
        rotation_matrix: np.ndarray,
        pad_bias: np.ndarray,
    ) -> dict[str, Any] | None:
        # Output ordering matches the ONNX model:
        #   Identity   (1, 63) — landmarks (21*xyz)
        #   Identity_1 (1, 1)  — confidence
        #   Identity_2 (1, 1)  — handedness  (0=Left, 1=Right; sigmoid score)
        #   Identity_3 (1, 63) — landmarks_world
        landmarks_raw = outputs[0]
        conf_arr = outputs[1]
        handedness_arr = outputs[2]
        conf = float(conf_arr.flatten()[0])
        if conf < self._conf_threshold:
            return None

        landmarks = landmarks_raw.reshape(-1, 3).astype(np.float32, copy=True)
        wh = rotated_palm_bbox[1] - rotated_palm_bbox[0]
        scale_factor = wh / float(self.INPUT_SIZE)
        s = float(np.max(scale_factor))
        landmarks[:, :2] = (landmarks[:, :2] - self.INPUT_SIZE / 2.0) * s
        landmarks[:, 2] = landmarks[:, 2] * s

        coord_rot = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated = np.dot(landmarks[:, :2], coord_rot[:, :2])
        rotated = np.column_stack([rotated, landmarks[:, 2]])

        rot_part = np.array([
            [rotation_matrix[0][0], rotation_matrix[1][0]],
            [rotation_matrix[0][1], rotation_matrix[1][1]],
        ], dtype=np.float32)
        trans_part = np.array(
            [rotation_matrix[0][2], rotation_matrix[1][2]], dtype=np.float32
        )
        inverted_trans = np.array([
            -np.dot(rot_part[0], trans_part),
            -np.dot(rot_part[1], trans_part),
        ], dtype=np.float32)
        inv_mat = np.column_stack([rot_part, inverted_trans])
        center_h = np.append(np.sum(rotated_palm_bbox, axis=0) / 2.0, 1.0)
        original_center = np.array([
            np.dot(center_h, inv_mat[0]),
            np.dot(center_h, inv_mat[1]),
        ], dtype=np.float32)
        landmarks[:, :2] = rotated[:, :2] + original_center + pad_bias

        handedness_score = float(handedness_arr.flatten()[0])
        # The OpenCV Zoo ONNX export's handedness score is reversed
        # relative to mediapipe.solutions.hands on the same mirrored
        # selfie input: their convention has high score = "Left",
        # low score = "Right". MediaPipe's convention (which the rest
        # of the engine — left-fist voice toggle, four-finger
        # YouTube mode, handedness reconciliation in detector.py) is
        # the opposite. Swapping the labels here keeps the rest of
        # the gesture pipeline unchanged. (User reported every gesture
        # fired on the wrong hand on the first GPU build; flipping
        # this is the single-line fix.)
        if handedness_score >= 0.5:
            label = "Left"
            score = handedness_score
        else:
            label = "Right"
            score = 1.0 - handedness_score
        return {
            "landmarks": landmarks,
            "handedness": label,
            "handedness_score": score,
            "presence_score": conf,
        }


# ---------------------------------------------------------------
# Hands adapter — solutions.hands.Hands shape on top of the two
# ONNX models.
# ---------------------------------------------------------------


class _OnnxHands:
    """`solutions.hands.Hands` drop-in with inter-frame palm-detect
    tracking. On a successful landmark inference we cache the palm
    keypoints + bbox derived from those landmarks; on the next frame
    we feed those directly to the landmark model, skipping palm
    detection entirely. Palm detection only re-runs when tracking
    is lost (low confidence, or hand left frame). This is exactly
    the loop MediaPipe's solutions.hands runs — same model weights,
    same accuracy — and saves ~3 ms / frame whenever a hand is
    actively tracked."""

    # Indices into the 21-landmark output that match
    # `_OnnxHandLandmarker.PALM_LANDMARK_IDS` (the 7 keypoints the
    # palm detector emits and the landmark detector's preprocess
    # step expects). Wrist + 5 finger bases + thumb-CMC, in the
    # order MediaPipe documents.
    _PALM_LANDMARK_INDICES = (0, 5, 9, 13, 17, 1, 2)

    # ---- Re-acquisition (see Stage 3 in _process_locked) ----------
    # Losing a hand we were already tracking and finding a brand-new
    # hand are different problems, but the pre-r55 code solved both
    # with one strict palm-detect gate. A hand whose landmark
    # presence dipped for a single frame (motion blur at the end of
    # a swipe) was dropped outright, and re-entry then required the
    # palm model to score >= min_detection_confidence (0.72) on a
    # pose it scores poorly — a curled hand with one finger up
    # measures ~0.45-0.50. The result was multi-second skeleton
    # dropouts with the hand plainly in frame.
    #
    # So for a short window after losing a track we (a) retry the
    # lost ROI with the cheap landmark pass, which is far better at
    # re-finding a known hand than palm-detect is, and (b) relax the
    # palm-detect floor. Discovering NEW hands still uses the strict
    # threshold, so idle-frame false positives are unchanged.
    #
    # Cost control: a relaxed floor surfaces more palm candidates, and
    # every candidate costs a landmark inference (~2-3 ms) to reject.
    # Letting all of them through would turn each gap frame into a
    # visible hitch, so only the single best sub-threshold candidate is
    # ever tried per frame. Candidates are score-sorted, so that is the
    # one most likely to be the hand we're chasing.
    _REACQUIRE_SCORE_FLOOR = 0.50
    _REACQUIRE_WINDOW_FRAMES = 20   # ~0.4 s at 50 fps
    _STALE_ROI_RETRY_FRAMES = 5     # ~0.1 s of ROI retries
    _MAX_RELAXED_CANDIDATES_PER_FRAME = 1

    def __init__(
        self,
        palm_session,
        landmark_session,
        anchors: np.ndarray,
        *,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        static_image_mode: bool,
        model_complexity: int,  # noqa: ARG002 — single ONNX model bundled
    ) -> None:
        self._palm = _OnnxPalmDetector(
            palm_session,
            anchors,
            score_threshold=float(min_detection_confidence),
            nms_threshold=0.3,
            top_k=max(int(max_num_hands) * 4, 8),
        )
        # 0.60 (was 0.50): the upstream OpenCV-Zoo example uses 0.5,
        # but in practice scores in [0.50, 0.60) at extreme yaw / roll
        # produced visibly wonky landmark scatters — fingers landing
        # in implausible positions that downstream gesture logic
        # would briefly mis-classify (false thumb-open during pen
        # rotation, e.g.). Bumping the floor to 0.60 drops those
        # bad reads entirely; the user sees a brief tracking gap
        # instead of a wrong gesture, which is the safer failure
        # mode. Stage-1 (tracked) reads still face the separate
        # _tracking_threshold gate (= min_tracking_confidence, 0.72
        # by default), so this only changes new-detection behaviour
        # via Stage-2 palm-detect.
        self._landmarker = _OnnxHandLandmarker(
            landmark_session,
            conf_threshold=0.60,
        )
        self._max_num_hands = max(1, int(max_num_hands))
        # min_tracking_confidence gates "is the previous landmark
        # result still trustworthy enough to skip palm detection".
        # Below this threshold we discard the cached track and
        # re-run the palm detector on the next frame. Mirrors
        # MediaPipe's solutions.hands behaviour. Static-image
        # callers (custom gesture recording) get tracking off
        # automatically — every frame is independent.
        self._tracking_threshold = float(min_tracking_confidence)
        self._tracking_disabled = bool(static_image_mode)
        # Never let the re-acquire floor sit ABOVE the configured
        # detection threshold — a caller that already runs loose
        # (low-FPS mode uses 0.34) must not get stricter mid-gap.
        self._palm_strict_threshold = float(min_detection_confidence)
        self._reacquire_score_threshold = min(
            float(min_detection_confidence), self._REACQUIRE_SCORE_FLOOR
        )
        self._reacquire_frames_left = 0
        self._stale_roi_frames_left = 0
        self._lock = threading.Lock()
        # Cached tracks: each entry is a dict shaped exactly like
        # what `_OnnxHandLandmarker.detect` accepts for `palm`,
        # plus the 'handedness' label so we keep stable per-hand
        # labels across frames without re-running palm-detect.
        self._tracked_palms: list[dict[str, Any]] = []
        # When at least one hand is tracked, only re-fire the palm
        # detector every Nth frame to look for additional hands.
        # Without this gate, max_num_hands=2 + a single hand visible
        # would keep palm-detect firing on every frame (since
        # survivors < max_num_hands every frame), wasting the entire
        # tracking speedup. ~1 s between scans for new hands feels
        # responsive; users introducing a second hand notice at
        # most a 30-frame delay before it gets tracked. Doesn't
        # affect single-hand max_num_hands=1 callers — they're
        # fully tracked the moment palm-detect succeeds once.
        self._scan_for_extra_hands_every = 30
        self._frames_since_last_scan = 0
        # Per-stage timing aggregation — sampled every process()
        # call, logged every ~2 seconds. Lets us tell DML inference
        # (palm: ~2 ms, landmark: ~3 ms expected) from silent CPU
        # fallback (palm: ~5 ms, landmark: ~12 ms+). Without this
        # split the only knob the upstream timing log shows is total
        # engine wall time, which mixes inference, signal-queue
        # wait, and per-op CPU/GPU placement.
        self._palm_samples_seconds: list[float] = []
        self._landmark_samples_seconds: list[float] = []
        self._stage_log_last_at: float = 0.0
        # Aggregate Stage-1 (tracking) outcomes across all process()
        # calls inside the 2-second log window so we can see exactly
        # why tracking is or isn't holding from frame to frame.
        self._stage1_attempts_total: int = 0
        self._stage1_lost_crop_total: int = 0
        self._stage1_lost_conf_total: int = 0
        self._stage1_ok_total: int = 0
        # Per-frame snapshots of Stage-1 outcomes — set inside
        # _process_locked, drained by _process_locked_with_stats.
        self._last_stage1_attempts: int = 0
        self._last_stage1_lost_crop: int = 0
        self._last_stage1_lost_conf: int = 0
        self._last_stage1_ok: int = 0

    def process(self, rgb_frame: np.ndarray) -> _HandsResult:
        # Serialise concurrent processes per session — the two
        # ONNX sessions are not necessarily thread-safe across all
        # backends and gestures don't gain anything from parallel
        # frame inference.
        with self._lock:
            result, stage1_stats = self._process_locked_with_stats(rgb_frame)
        # Sample after the lock so the log lines never block the
        # gesture loop.
        self._stage1_attempts_total += stage1_stats[0]
        self._stage1_lost_crop_total += stage1_stats[1]
        self._stage1_lost_conf_total += stage1_stats[2]
        self._stage1_ok_total += stage1_stats[3]
        self._record_stage_samples()
        return result

    def _process_locked_with_stats(
        self, rgb_frame: np.ndarray
    ) -> tuple[_HandsResult, tuple[int, int, int, int]]:
        # Wrapper that returns Stage-1 outcome counters alongside
        # the result so they can be logged from process() without
        # mutating self under the lock.
        result = self._process_locked(rgb_frame)
        stats = (
            self._last_stage1_attempts,
            self._last_stage1_lost_crop,
            self._last_stage1_lost_conf,
            self._last_stage1_ok,
        )
        return result, stats

    def _record_stage_samples(self) -> None:
        # Drain-and-zero the per-detector last-inference timer so each
        # value is sampled exactly once. Prior version re-sampled the
        # last positive value on every process() call, inflating the
        # palm-detect sample count by ~25x and making it look like
        # palm-detect was firing every frame when it was actually
        # firing only every 30 frames as designed.
        palm_t = float(getattr(self._palm, "_last_inference_seconds", 0.0))
        lm_t = float(getattr(self._landmarker, "_last_inference_seconds", 0.0))
        if palm_t > 0.0:
            self._palm_samples_seconds.append(palm_t)
            self._palm._last_inference_seconds = 0.0  # type: ignore[attr-defined]
        if lm_t > 0.0:
            self._landmark_samples_seconds.append(lm_t)
            self._landmarker._last_inference_seconds = 0.0  # type: ignore[attr-defined]
        now = time.monotonic()
        if self._stage_log_last_at == 0.0:
            self._stage_log_last_at = now
            return
        if now - self._stage_log_last_at < 2.0:
            return
        if len(self._palm_samples_seconds) + len(self._landmark_samples_seconds) < 4:
            return
        self._stage_log_last_at = now
        palm_samples = self._palm_samples_seconds
        lm_samples = self._landmark_samples_seconds
        self._palm_samples_seconds = []
        self._landmark_samples_seconds = []
        try:
            # Average alone hides spike contention from co-resident GPU
            # workloads (whisper-stream.exe with -ngl, llama-server.exe
            # serving Qwen grammar correction). p95 + max expose those
            # spikes so a steady avg=3ms with max=25ms is visibly the
            # signature of GPU contention rather than a smooth track.
            def _stats_ms(samples: list[float]) -> tuple[float, float, float, float]:
                if not samples:
                    return 0.0, 0.0, 0.0, 0.0
                ms = sorted(s * 1000.0 for s in samples)
                avg = sum(ms) / len(ms)
                p50 = ms[len(ms) // 2]
                p95_idx = max(0, min(len(ms) - 1, int(round(0.95 * (len(ms) - 1)))))
                p95 = ms[p95_idx]
                mx = ms[-1]
                return avg, p50, p95, mx

            palm_avg, palm_p50, palm_p95, palm_max = _stats_ms(palm_samples)
            lm_avg, lm_p50, lm_p95, lm_max = _stats_ms(lm_samples)
            stage1_a = self._stage1_attempts_total
            stage1_lc = self._stage1_lost_crop_total
            stage1_lp = self._stage1_lost_conf_total
            stage1_ok = self._stage1_ok_total
            self._stage1_attempts_total = 0
            self._stage1_lost_crop_total = 0
            self._stage1_lost_conf_total = 0
            self._stage1_ok_total = 0
            gate_scan = getattr(self, "_gate_fire_scan_due", 0)
            gate_nos = getattr(self, "_gate_fire_no_survivors", 0)
            gate_dis = getattr(self, "_gate_fire_disabled", 0)
            tp0 = getattr(self, "_gate_fire_tp0", 0)
            tp1 = getattr(self, "_gate_fire_tp1", 0)
            tpN = getattr(self, "_gate_fire_tpN", 0)
            reacq = getattr(self, "_gate_fire_reacquire", 0)
            self._gate_fire_reacquire = 0
            self._gate_fire_scan_due = 0
            self._gate_fire_no_survivors = 0
            self._gate_fire_disabled = 0
            self._gate_fire_tp0 = 0
            self._gate_fire_tp1 = 0
            self._gate_fire_tpN = 0
            sys.stderr.write(
                f"[onnx_runtime] palm avg={palm_avg:.1f} p50={palm_p50:.1f} "
                f"p95={palm_p95:.1f} max={palm_max:.1f}ms ({len(palm_samples)}) "
                f"landmark avg={lm_avg:.1f} p50={lm_p50:.1f} "
                f"p95={lm_p95:.1f} max={lm_max:.1f}ms ({len(lm_samples)}) "
                f"track1[attempts={stage1_a} ok={stage1_ok} "
                f"lost_crop_or_None={stage1_lc} lost_conf={stage1_lp}] "
                f"gate[scan={gate_scan} no_surv={gate_nos} dis={gate_dis}] "
                f"tp_at_fire[0={tp0} 1={tp1} N={tpN}] "
                f"reacq={reacq}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass

    def _process_locked(self, rgb_frame: np.ndarray) -> _HandsResult:
        h, w = rgb_frame.shape[:2]
        if h <= 0 or w <= 0:
            self._tracked_palms = []
            return _HandsResult([], [])

        # ----- Stage 1: try to track from cached landmarks ----------
        # When we have one or more tracked palms from the previous
        # frame, run the landmark detector on each cached ROI. If
        # the landmark confidence stays above the tracking threshold,
        # that hand is still tracked — derive a fresh ROI from the
        # new landmarks and stash it for next frame. If confidence
        # drops, that track is lost; we re-run the palm detector at
        # the bottom of this method to find any hands again. Net
        # effect: palm detection only fires on hand-appearance frames
        # (~3 ms saved on every steady-state tracking frame).
        landmark_results: list[_NormalizedLandmarkList] = []
        handedness_results: list[_ClassificationList] = []
        survivors: list[dict[str, Any]] = []
        # Stage-1 outcome counters — periodic stderr log so we can
        # see whether tracking is succeeding (palm-detect should
        # then fire only every 30 frames) or failing (which forces
        # palm-detect to re-fire every frame, blowing the GPU
        # speedup). Buckets: lost_crop = preprocess clipped to
        # zero area; lost_conf = inference returned None or
        # presence below threshold; ok = track survived.
        stage1_attempts = 0
        stage1_lost_crop = 0
        stage1_lost_conf = 0
        stage1_ok = 0
        # ROIs whose landmark pass failed this frame. Stage 3 may keep
        # them for a few more frames so a momentary dip doesn't force a
        # full palm-detect re-acquisition.
        lost_rois: list[dict[str, Any]] = []
        if not self._tracking_disabled and self._tracked_palms:
            for prev in self._tracked_palms:
                stage1_attempts += 1
                lm = self._landmarker.detect(rgb_frame, prev)
                if lm is None:
                    # Could be crop=None OR conf below threshold;
                    # the landmarker collapses both into None so we
                    # can't tell apart here without instrumenting it
                    # directly. Treat as crop-loss bucket — we'll
                    # widen the diagnostic if needed.
                    stage1_lost_crop += 1
                    lost_rois.append(prev)
                    continue
                if lm["presence_score"] < self._tracking_threshold:
                    stage1_lost_conf += 1
                    lost_rois.append(prev)
                    continue
                stage1_ok += 1
                next_palm = self._derive_palm_from_landmarks(lm["landmarks"])
                # Carry the previously-resolved handedness on the
                # track. Re-running the landmark model produces a
                # fresh handedness score every frame, but that score
                # is noisy at small ROI shifts. Sticking with the
                # initial label keeps left-fist / four-finger
                # toggles stable across the duration of a hand
                # being in view, just like MediaPipe's behaviour.
                next_palm["handedness"] = lm["handedness"]
                next_palm["handedness_score"] = lm["handedness_score"]
                survivors.append(next_palm)
                self._append_landmark_result(
                    lm, w, h, landmark_results, handedness_results
                )

        # ----- Stage 2: palm-detect to discover new / replacement hands -
        # Decide whether to actually run palm-detect this frame.
        # Three reasons we'd run it:
        #   (a) tracking disabled (static image mode)
        #   (b) ZERO survivors — no hand currently tracked, palm-
        #       detect must run to find any hand at all
        #   (c) at least one survivor but fewer than max_num_hands
        #       AND we haven't scanned in N frames — periodic look
        #       for additional hands when one is already tracked
        # Without case (c)'s frame gate, max_num_hands=2 + single
        # hand visible would re-fire palm-detect every frame and
        # waste the whole tracking speedup; the user would never
        # see the engine= cost drop below ~13 ms even when the
        # hand was steady-tracked.
        cond_disabled = bool(self._tracking_disabled)
        cond_no_survivors = not survivors
        cond_scan = (
            len(survivors) < self._max_num_hands
            and self._frames_since_last_scan >= self._scan_for_extra_hands_every
        )
        run_palm_detect = cond_disabled or cond_no_survivors or cond_scan
        # Aggregate gate-fire reasons across the 2-second log
        # window. Diagnoses the "palm-detect fires every frame
        # despite tracking succeeding" case by attributing each
        # fire to its triggering condition (disabled/no_survivors/
        # scan_due). One of these MUST be true on every fire — the
        # one with count = N (frame count) is the leak.
        if run_palm_detect:
            if cond_scan:
                self._gate_fire_scan_due = getattr(self, "_gate_fire_scan_due", 0) + 1
            if cond_no_survivors:
                self._gate_fire_no_survivors = getattr(self, "_gate_fire_no_survivors", 0) + 1
            if cond_disabled:
                self._gate_fire_disabled = getattr(self, "_gate_fire_disabled", 0) + 1
            # Also track tracked_palms entry count per fire so we
            # can see whether tracked_palms is empty when gate fires
            # (which would explain no_survivors=True every frame).
            self._gate_fire_tp0 = getattr(self, "_gate_fire_tp0", 0) + (1 if not self._tracked_palms else 0)
            self._gate_fire_tp1 = getattr(self, "_gate_fire_tp1", 0) + (1 if len(self._tracked_palms) == 1 else 0)
            self._gate_fire_tpN = getattr(self, "_gate_fire_tpN", 0) + (1 if len(self._tracked_palms) > 1 else 0)
        # Relax the palm-detect floor while we're chasing a hand we
        # just lost (see _REACQUIRE_SCORE_FLOOR). `lost_rois` covers
        # the frame of the loss itself; the counters cover the frames
        # after it.
        reacquiring = bool(
            lost_rois
            or self._reacquire_frames_left > 0
            or self._stale_roi_frames_left > 0
        )
        if run_palm_detect:
            self._frames_since_last_scan = 0
            palms = self._palm.detect(
                rgb_frame,
                score_threshold=(
                    self._reacquire_score_threshold if reacquiring else None
                ),
            )
            if reacquiring:
                self._gate_fire_reacquire = getattr(self, "_gate_fire_reacquire", 0) + 1
            palms = sorted(palms, key=lambda p: -p["score"])
            relaxed_tried = 0
            for palm in palms:
                if len(survivors) >= self._max_num_hands:
                    break
                if float(palm["score"]) < self._palm_strict_threshold:
                    # Sorted by score, so everything from here down is
                    # sub-threshold too — spend one landmark inference
                    # on the best of them and stop.
                    if relaxed_tried >= self._MAX_RELAXED_CANDIDATES_PER_FRAME:
                        break
                    relaxed_tried += 1
                # Skip palm candidates that overlap an existing
                # tracked hand — those would just produce a duplicate
                # landmark inference for the same hand.
                if self._overlaps_any(palm, survivors):
                    continue
                lm = self._landmarker.detect(rgb_frame, palm)
                if lm is None:
                    continue
                bootstrap = self._derive_palm_from_landmarks(lm["landmarks"])
                bootstrap["handedness"] = lm["handedness"]
                bootstrap["handedness_score"] = lm["handedness_score"]
                survivors.append(bootstrap)
                self._append_landmark_result(
                    lm, w, h, landmark_results, handedness_results
                )
        else:
            self._frames_since_last_scan += 1

        # ----- Stage 3: re-acquisition bookkeeping --------------------
        # Nothing survived and something was lost this frame: hold the
        # lost ROI so next frame's Stage 1 gets another (cheap) shot at
        # it, and keep the relaxed palm floor armed for a short window.
        # Any survivor clears the whole thing — we're tracking again.
        if self._tracking_disabled:
            self._tracked_palms = []
            self._reacquire_frames_left = 0
            self._stale_roi_frames_left = 0
        elif survivors:
            self._tracked_palms = survivors
            self._reacquire_frames_left = 0
            self._stale_roi_frames_left = 0
        elif lost_rois:
            if self._stale_roi_frames_left <= 0:
                self._stale_roi_frames_left = self._STALE_ROI_RETRY_FRAMES
                self._reacquire_frames_left = self._REACQUIRE_WINDOW_FRAMES
            else:
                self._stale_roi_frames_left -= 1
            self._tracked_palms = (
                lost_rois if self._stale_roi_frames_left > 0 else []
            )
        else:
            self._tracked_palms = []
            self._stale_roi_frames_left = 0
            if self._reacquire_frames_left > 0:
                self._reacquire_frames_left -= 1
        # Snapshot Stage-1 counters for this frame so process() can
        # aggregate them across the 2-second log window.
        self._last_stage1_attempts = stage1_attempts
        self._last_stage1_lost_crop = stage1_lost_crop
        self._last_stage1_lost_conf = stage1_lost_conf
        self._last_stage1_ok = stage1_ok
        return _HandsResult(landmark_results, handedness_results)

    def _append_landmark_result(
        self,
        lm: dict[str, Any],
        w: int,
        h: int,
        landmark_results: list[_NormalizedLandmarkList],
        handedness_results: list[_ClassificationList],
    ) -> None:
        # Normalise landmarks into [0, 1] image coords — what
        # solutions.hands returns and what HandDetector downstream
        # expects. z stays roughly normalized (divide by width)
        # since solutions.hands' z is also unitless-ish — depth
        # relative to wrist, not metric.
        lm_norm: list[_LandmarkPoint] = []
        for x_px, y_px, z_rel in lm["landmarks"]:
            lm_norm.append(_LandmarkPoint(
                x=float(x_px) / float(w),
                y=float(y_px) / float(h),
                z=float(z_rel) / float(w),
            ))
        landmark_results.append(_NormalizedLandmarkList(lm_norm))
        handedness_results.append(
            _ClassificationList(label=lm["handedness"], score=lm["handedness_score"])
        )

    def _derive_palm_from_landmarks(self, landmarks: np.ndarray) -> dict[str, Any]:
        """Build the dict the landmark detector's preprocess step
        accepts (`bbox` + `keypoints`) directly from a previous
        frame's landmarks. Skips the palm detector entirely on the
        next call — the speedup that makes this whole class worth
        having.

        Critical sizing detail: the landmark preprocess applies a
        4x pre-enlarge for rotation alignment and then a 3x enlarge
        on the rotated bbox (PALM_BOX_PRE_ENLARGE_FACTOR /
        PALM_BOX_ENLARGE_FACTOR). The palm detector's output bbox is
        sized for those enlarges. If we feed in a whole-hand bbox
        (from min/max of all 21 landmarks — finger TIPS to wrist)
        the post-enlarge crop falls well outside the frame, the
        landmark model sees mostly black padding, presence_score
        collapses below the tracking threshold, and tracking is
        dropped every single frame — defeating the entire tracking
        speedup. The fix: bound only the palm-region landmarks
        (wrist + finger BASE knuckles + thumb base), which matches
        the palm-only bbox shape the palm detector emits.

        keypoints = the 7 palm landmarks the rotation step uses
        (indices 0, 5, 9, 13, 17, 1, 2)."""
        xy = landmarks[:, :2].astype(np.float32, copy=False)
        palm_xy = xy[list(self._PALM_LANDMARK_INDICES)]
        x_min = float(np.min(palm_xy[:, 0]))
        y_min = float(np.min(palm_xy[:, 1]))
        x_max = float(np.max(palm_xy[:, 0]))
        y_max = float(np.max(palm_xy[:, 1]))
        # Pad bbox by 10 % of its size to absorb fast inter-frame
        # motion. Below this threshold the next-frame landmark
        # detector's enlarge factor (3x) easily covers any hand
        # motion that doesn't exceed ~30 px between frames.
        pad_x = (x_max - x_min) * 0.10
        pad_y = (y_max - y_min) * 0.10
        bbox = np.array(
            [x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y],
            dtype=np.float32,
        )
        keypoints = palm_xy.astype(np.float32, copy=True)
        return {"bbox": bbox, "keypoints": keypoints, "score": 0.99}

    @staticmethod
    def _overlaps_any(palm: dict[str, Any], existing: list[dict[str, Any]]) -> bool:
        """Cheap centre-distance overlap test. Two palm bboxes
        whose centres are within half the smaller bbox's diagonal
        are considered the same hand. Avoids calling the landmark
        detector twice on the same hand when palm-detect surfaces
        a bbox that already has a track."""
        if not existing:
            return False
        pb = palm["bbox"]
        pcx = (pb[0] + pb[2]) / 2.0
        pcy = (pb[1] + pb[3]) / 2.0
        pw = max(1.0, pb[2] - pb[0])
        ph = max(1.0, pb[3] - pb[1])
        for tracked in existing:
            tb = tracked["bbox"]
            tcx = (tb[0] + tb[2]) / 2.0
            tcy = (tb[1] + tb[3]) / 2.0
            tw = max(1.0, tb[2] - tb[0])
            th = max(1.0, tb[3] - tb[1])
            min_size = min(pw, ph, tw, th)
            dx = pcx - tcx
            dy = pcy - tcy
            if (dx * dx + dy * dy) ** 0.5 < min_size * 0.5:
                return True
        return False

    def close(self) -> None:
        # onnxruntime InferenceSession doesn't have an explicit
        # close — Python GC + the session's internal CUDA / DML
        # context destructors handle cleanup. We null our refs so
        # the user's handle becomes obviously unusable.
        self._palm = None  # type: ignore[assignment]
        self._tracked_palms = []
        self._landmarker = None  # type: ignore[assignment]


# ---------------------------------------------------------------
# Module-shaped shim that exposes a `Hands` attribute the rest of
# the engine reads — same dance tasks_runtime.py uses.
# ---------------------------------------------------------------


class _HandsModuleShim:
    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    )

    def __init__(self, palm_session, landmark_session, anchors: np.ndarray) -> None:
        self._palm_session = palm_session
        self._landmark_session = landmark_session
        self._anchors = anchors

    def Hands(
        self,
        *,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> _OnnxHands:  # noqa: N802 — match solutions.hands API
        return _OnnxHands(
            self._palm_session,
            self._landmark_session,
            self._anchors,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
        )


def build_onnx_directml_runtime() -> _HandsModuleShim | None:
    """Return a hands_module shim wrapping ONNX Runtime + DirectML
    sessions for the palm detector + hand landmark models. Returns
    None if onnxruntime-directml isn't installed, the DML provider
    isn't reachable, or any required asset is missing — the caller
    falls back to CPU MediaPipe transparently."""
    palm_path = _resolve_asset(PALM_DETECTOR_FILENAME)
    lm_path = _resolve_asset(HAND_LANDMARK_FILENAME)
    anchors_path = _resolve_asset(PALM_ANCHORS_FILENAME)
    missing = [
        n for n, p in (
            (PALM_DETECTOR_FILENAME, palm_path),
            (HAND_LANDMARK_FILENAME, lm_path),
            (PALM_ANCHORS_FILENAME, anchors_path),
        ) if p is None
    ]
    if missing:
        try:
            sys.stderr.write(
                f"[onnx_runtime] missing asset(s): {missing}. GPU path "
                "unavailable; falling back to CPU MediaPipe.\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        return None

    try:
        import onnxruntime as ort
    except Exception as exc:
        try:
            sys.stderr.write(
                f"[onnx_runtime] onnxruntime not importable: "
                f"{type(exc).__name__}: {exc!s}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        return None

    providers_available = list(ort.get_available_providers())
    if "DmlExecutionProvider" not in providers_available:
        try:
            sys.stderr.write(
                "[onnx_runtime] DmlExecutionProvider not present. "
                f"Available: {providers_available}. Falling back to CPU MediaPipe.\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        return None

    # Prefer DML; fall back to CPU within the *same session* if
    # DML can't load this particular model. (CPU within ONNX is
    # still the GPU-port path; if onnxruntime can't even DML it,
    # the wider runtime falls through to MediaPipe-CPU.)
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    try:
        palm_session = ort.InferenceSession(str(palm_path), providers=providers)
        lm_session = ort.InferenceSession(str(lm_path), providers=providers)
    except Exception as exc:
        try:
            sys.stderr.write(
                f"[onnx_runtime] InferenceSession construction failed: "
                f"{type(exc).__name__}: {exc!s}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        return None

    try:
        anchors = np.load(str(anchors_path)).astype(np.float32, copy=False)
    except Exception as exc:
        try:
            sys.stderr.write(
                f"[onnx_runtime] anchor table load failed: "
                f"{type(exc).__name__}: {exc!s}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        return None

    if anchors.ndim != 2 or anchors.shape[1] != 2:
        try:
            sys.stderr.write(
                f"[onnx_runtime] anchor table shape unexpected: {anchors.shape}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        return None

    return _HandsModuleShim(palm_session, lm_session, anchors)

# Author: Konstantin Markov

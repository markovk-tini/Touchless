from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

_HAS_DEPS = (
    importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("numpy") is not None
)

if _HAS_DEPS:
    import numpy as np

    from hgr.gesture.tracking.onnx_runtime import _OnnxHands


def _landmarks() -> "np.ndarray":
    """21 plausible landmark pixel coords (wrist at the bottom)."""
    pts = []
    for i in range(21):
        pts.append([300.0 + (i % 5) * 6.0, 260.0 - (i // 5) * 9.0, 0.0])
    return np.array(pts, dtype=np.float32)


class _FakeSession:
    def get_inputs(self):
        return [SimpleNamespace(name="input")]


class _FakeLandmarker:
    def __init__(self) -> None:
        self.presence = 0.95
        self.rois: list = []

    def detect(self, _rgb, palm):
        self.rois.append(palm)
        if self.presence is None:
            return None
        return {
            "landmarks": _landmarks(),
            "handedness": "Right",
            "handedness_score": 0.9,
            "presence_score": float(self.presence),
        }


class _FakePalmDetector:
    def __init__(self) -> None:
        self.thresholds: list = []
        self.result: list = []

    def detect(self, _rgb, *, score_threshold=None):
        self.thresholds.append(score_threshold)
        return [dict(p) for p in self.result]


def _palm_candidate() -> dict:
    return {
        "bbox": np.array([280.0, 200.0, 340.0, 260.0], dtype=np.float32),
        "keypoints": np.zeros((7, 2), dtype=np.float32),
        "score": 0.81,
    }


@unittest.skipUnless(_HAS_DEPS, "numpy/cv2 unavailable in this environment")
class OnnxHandReacquireTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hands = _OnnxHands(
            _FakeSession(),
            _FakeSession(),
            np.zeros((1, 2), dtype=np.float32),
            max_num_hands=1,
            min_detection_confidence=0.72,
            min_tracking_confidence=0.72,
            static_image_mode=False,
            model_complexity=1,
        )
        self.palm = _FakePalmDetector()
        self.landmarker = _FakeLandmarker()
        self.hands._palm = self.palm
        self.hands._landmarker = self.landmarker
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def _step(self):
        return self.hands._process_locked(self.frame)

    def _acquire(self) -> None:
        self.palm.result = [_palm_candidate()]
        self._step()
        self.palm.result = []
        self.assertTrue(self.hands._tracked_palms, "setup: hand should be tracked")

    def test_new_hand_search_uses_strict_threshold(self) -> None:
        self._step()

        self.assertEqual(self.palm.thresholds, [None])

    def test_lost_track_relaxes_palm_threshold(self) -> None:
        self._acquire()
        self.landmarker.presence = 0.50  # below the 0.72 tracking gate

        self._step()

        self.assertEqual(self.palm.thresholds[-1], self.hands._reacquire_score_threshold)
        self.assertLess(self.hands._reacquire_score_threshold, 0.72)

    def test_reacquire_floor_never_exceeds_detection_threshold(self) -> None:
        loose = _OnnxHands(
            _FakeSession(),
            _FakeSession(),
            np.zeros((1, 2), dtype=np.float32),
            max_num_hands=1,
            min_detection_confidence=0.34,  # low-FPS mode's setting
            min_tracking_confidence=0.22,
            static_image_mode=False,
            model_complexity=1,
        )

        self.assertLessEqual(loose._reacquire_score_threshold, 0.34)

    def test_lost_roi_is_retried_and_recovers_without_palm_detect(self) -> None:
        self._acquire()
        tracked_roi = self.hands._tracked_palms[0]

        self.landmarker.presence = 0.50
        self._step()
        self.assertEqual(self.hands._tracked_palms, [tracked_roi])

        palm_calls_before = len(self.palm.thresholds)
        self.landmarker.presence = 0.95
        result = self._step()

        self.assertEqual(self.landmarker.rois[-1], tracked_roi)
        self.assertEqual(len(self.palm.thresholds), palm_calls_before)
        self.assertEqual(len(result.multi_hand_landmarks), 1)
        self.assertEqual(self.hands._reacquire_frames_left, 0)
        self.assertEqual(self.hands._stale_roi_frames_left, 0)

    def test_stale_roi_is_dropped_after_retry_budget(self) -> None:
        self._acquire()
        self.landmarker.presence = None  # landmark pass fails outright

        for _ in range(self.hands._STALE_ROI_RETRY_FRAMES + 1):
            self._step()

        self.assertEqual(self.hands._tracked_palms, [])

    def test_reacquire_window_expires_when_hand_stays_gone(self) -> None:
        self._acquire()
        self.landmarker.presence = None

        for _ in range(
            self.hands._STALE_ROI_RETRY_FRAMES
            + self.hands._REACQUIRE_WINDOW_FRAMES
            + 4
        ):
            self._step()

        self.assertEqual(self.hands._reacquire_frames_left, 0)
        self.assertIsNone(self.palm.thresholds[-1], "should be strict again once idle")

    def test_relaxed_search_costs_at_most_one_extra_landmark_pass(self) -> None:
        # A relaxed floor surfaces junk candidates (face, background).
        # Rejecting each one costs a landmark inference, so the loop
        # must stop after the best sub-threshold candidate.
        self._acquire()
        weak = []
        for score in (0.68, 0.64, 0.61, 0.58, 0.55, 0.52):
            cand = _palm_candidate()
            cand["score"] = score
            weak.append(cand)
        self.palm.result = weak
        self.landmarker.presence = None  # nothing will pass
        landmark_calls_before = len(self.landmarker.rois)

        self._step()

        stage1_retry = 1  # the stale ROI from the lost track
        self.assertEqual(
            len(self.landmarker.rois) - landmark_calls_before,
            stage1_retry + self.hands._MAX_RELAXED_CANDIDATES_PER_FRAME,
        )

    def test_strong_candidates_are_not_capped(self) -> None:
        strong = []
        for score in (0.91, 0.85):
            cand = _palm_candidate()
            cand["score"] = score
            strong.append(cand)
        self.palm.result = strong
        self.landmarker.presence = None
        self.hands._max_num_hands = 2

        self._step()

        self.assertEqual(len(self.landmarker.rois), 2)

    def test_steady_tracking_still_skips_palm_detect(self) -> None:
        self._acquire()
        palm_calls_before = len(self.palm.thresholds)

        for _ in range(20):
            self._step()

        self.assertEqual(len(self.palm.thresholds), palm_calls_before)


if __name__ == "__main__":
    unittest.main()

# Author: Konstantin Markov

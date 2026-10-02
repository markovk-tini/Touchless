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

    def test_reacquire_threshold_matches_strict_after_neutralize(self) -> None:
        # v1.1.9.2 tracking-loss fix (see onnx_runtime.py neutralize
        # comment): _REACQUIRE_SCORE_FLOOR was raised to 1.0 so the
        # __init__ min() clamp keeps _reacquire_score_threshold equal
        # to the strict palm-detect floor. This test used to assert
        # the reacquire path relaxed the threshold; it now documents
        # that palm-detect stays strict on a loss (v1.1.7 behavior).
        self._acquire()
        self.landmarker.presence = 0.50  # below the 0.72 tracking gate

        self._step()

        self.assertEqual(
            self.hands._reacquire_score_threshold,
            self.hands._palm_strict_threshold,
        )
        self.assertEqual(self.hands._reacquire_score_threshold, 0.72)

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

    def test_lost_roi_survives_a_short_presence_dip_then_drops(self) -> None:
        """A brief landmark-presence dip must NOT drop the tracked ROI.

        This case used to assert the opposite -- immediate drop -- back
        when `_STALE_ROI_RETRY_FRAMES` was 0. `02e343a` restored it to 2
        on a user report: dropping the hand the moment motion blur dipped
        landmark confidence meant palm-detect had to re-find the hand
        centered, so the user "had to swipe almost all the way across the
        camera view" to get both swipe endpoints detected. Two retries on
        the same ROI bridge a ~30-60 ms dip for the cost of a cheap
        landmark pass, with no palm-detect scan.

        Both halves are pinned: the ROI survives while the budget lasts,
        and it is still dropped once the budget runs out, so restoring
        the old immediate-drop OR making the bridge unbounded fails here.
        """
        self._acquire()
        self.landmarker.presence = 0.50

        self._step()
        self.assertTrue(self.hands._tracked_palms,
                        "a single-frame presence dip dropped the ROI")
        self.assertEqual(self.hands._stale_roi_frames_left,
                         self.hands._STALE_ROI_RETRY_FRAMES)

        self._step()
        self.assertTrue(self.hands._tracked_palms,
                        "the ROI was dropped before the retry budget ran out")
        self.assertEqual(self.hands._stale_roi_frames_left,
                         self.hands._STALE_ROI_RETRY_FRAMES - 1)

        # Budget exhausted on the next pass: hand goes, and re-acquiring
        # needs a full palm-detect scan again.
        self._step()
        self.assertEqual(self.hands._tracked_palms, [])
        self.assertEqual(self.hands._stale_roi_frames_left, 0)
        self.assertEqual(self.hands._reacquire_frames_left, 0)

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

    def test_relaxed_search_spends_no_landmark_after_neutralize(self) -> None:
        # v1.1.9.2 tracking-loss fix: _MAX_RELAXED_CANDIDATES_PER_FRAME
        # was set to 0 so sub-strict palm candidates never trigger a
        # landmark inference. Combined with _STALE_ROI_RETRY_FRAMES=0
        # (stale ROI dropped on the loss frame), Stage-1 has no retry
        # to burn either. Net expected landmark inferences on a
        # loss frame with only weak candidates: ZERO.
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

        # Stage-1 still burns one landmark inference on the tracked
        # palm from the previous frame (which returns None here — that's
        # the loss). With _MAX_RELAXED_CANDIDATES_PER_FRAME=0 Stage-2
        # spends ZERO landmark inferences on the sub-strict candidates
        # (breaks on the first < strict-threshold entry). Total = 1.
        self.assertEqual(
            len(self.landmarker.rois) - landmark_calls_before,
            1,
            "neutralized reacquire must not spend landmark inferences on "
            "sub-strict palm candidates (only Stage-1 retry counts)",
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

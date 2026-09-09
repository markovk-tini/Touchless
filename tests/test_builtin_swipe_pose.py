"""Builtin swipe_left/right require an open hand.

Index-only ('one') horizontal motion must not publish swipe_right —
that pose belongs to custom '1 swipe right' and to drawing undo.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from hgr.gesture.recognition.engine import GestureRecognitionEngine

from .helpers import make_pose, translate_landmarks


def _right_swipe_frames(pose_name: str):
    base = make_pose(pose_name)
    return [
        translate_landmarks(base, dx=value)
        for value in (-0.18, -0.11, -0.03, 0.06, 0.14, 0.22)
    ]


class BuiltinSwipePoseTests(unittest.TestCase):
    def _run(self, pose_name: str):
        engine = GestureRecognitionEngine(stable_frames_required=1)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        labels = []
        right_scores = []
        one_pose_labels = []
        for index, landmarks in enumerate(_right_swipe_frames(pose_name)):
            result = engine.process_landmarks(
                landmarks, frame, "Right", timestamp=1.0 + 0.05 * index,
            )
            labels.append(result.prediction.dynamic_label)
            right_scores.append(engine._last_dynamic_scores.get("swipe_right", 0.0))
            one_pose_labels.append(
                engine.dynamic_recognizer.last_one_pose_horizontal_label
            )
        return labels, right_scores, one_pose_labels

    def test_open_hand_swipe_right_still_detects(self) -> None:
        labels, scores, _ = self._run("open_hand")
        self.assertGreater(max(scores), 0.48)
        self.assertIn("swipe_right", labels)

    def test_index_only_swipe_does_not_publish_builtin_swipe_right(self) -> None:
        labels, scores, one_pose = self._run("one")
        self.assertLess(
            max(scores), 0.48,
            msg=f"index-only swipe_right score {max(scores):.3f} must stay below floor",
        )
        self.assertNotIn("swipe_right", labels)
        self.assertNotIn("swipe_left", labels)

    def test_index_only_swipe_still_available_for_drawing(self) -> None:
        _, _, one_pose = self._run("one")
        self.assertIn(
            "swipe_right",
            one_pose,
            msg="drawing undo still needs index-only horizontal swipe",
        )


class DarwinBuiltinSwipePoseTests(unittest.TestCase):
    @patch("hgr.gesture.recognition.dynamic_recognizer.sys.platform", "darwin")
    def test_open_hand_swipe_right_still_detects_on_darwin(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=1)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        scores = []
        labels = []
        for index, landmarks in enumerate(_right_swipe_frames("open_hand")):
            result = engine.process_landmarks(
                landmarks, frame, "Right", timestamp=1.0 + 0.05 * index,
            )
            labels.append(result.prediction.dynamic_label)
            scores.append(engine._last_dynamic_scores.get("swipe_right", 0.0))
        self.assertGreater(max(scores), 0.48)
        self.assertIn("swipe_right", labels)

    @patch("hgr.gesture.recognition.dynamic_recognizer.sys.platform", "darwin")
    def test_short_drift_stays_below_darwin_floor(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=1)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        base = make_pose("open_hand")
        sequence = [
            translate_landmarks(base, dx=value)
            for value in (-0.07, -0.04, -0.01, 0.02, 0.05, 0.09)
        ]
        scores = []
        labels = []
        for index, landmarks in enumerate(sequence):
            result = engine.process_landmarks(
                landmarks, frame, "Right", timestamp=1.0 + 0.05 * index,
            )
            labels.append(result.prediction.dynamic_label)
            scores.append(engine._last_dynamic_scores.get("swipe_right", 0.0))
        self.assertLess(max(scores), 0.54)
        self.assertNotIn("swipe_right", labels)
        self.assertNotIn("swipe_left", labels)


if __name__ == "__main__":
    unittest.main()

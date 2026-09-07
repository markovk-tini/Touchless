from __future__ import annotations

import math
import unittest

import numpy as np

from hgr.core.arbitration.smoother import GestureSmoother
from hgr.core.pipeline.gesture_backend import GestureBackend
from hgr.core.tracking.landmark_smoother import LandmarkSmoother

from .helpers import make_landmarks, make_pose, translate_landmarks


class DummyTracker:
    def close(self) -> None:
        return


def _run_sequence(sequence):
    backend = GestureBackend(
        tracker=DummyTracker(),
        landmark_smoother=LandmarkSmoother(alpha=1.0),
        gesture_smoother=GestureSmoother(required_frames=2),
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    results = []
    for index, landmarks in enumerate(sequence):
        results.append(backend.process_landmarks(landmarks, frame, 'Right', timestamp=1.0 + 0.05 * index))
    return results


class DynamicGestureTest(unittest.TestCase):
    def test_swipe_right_detects(self) -> None:
        base = make_pose('open_hand')
        sequence = [translate_landmarks(base, dx=value) for value in (-0.18, -0.11, -0.03, 0.06, 0.14, 0.22)]
        results = _run_sequence(sequence)
        best = max(result.dynamic_candidate_scores.get('swipe_right', 0.0) for result in results)
        self.assertGreater(best, 0.58)
        self.assertTrue(all(result.raw_gesture != 'swipe_right' for result in results[:4]))

    def test_swipe_right_stays_low_for_short_motion(self) -> None:
        base = make_pose('open_hand')
        sequence = [translate_landmarks(base, dx=value) for value in (-0.07, -0.04, -0.01, 0.02, 0.05, 0.09)]
        results = _run_sequence(sequence)
        best = max(result.dynamic_candidate_scores.get('swipe_right', 0.0) for result in results)
        self.assertLess(best, 0.56)

    def test_swipe_left_detects(self) -> None:
        base = make_pose('open_hand')
        sequence = [translate_landmarks(base, dx=value) for value in (0.18, 0.11, 0.03, -0.06, -0.14, -0.22)]
        results = _run_sequence(sequence)
        best = max(result.dynamic_candidate_scores.get('swipe_left', 0.0) for result in results)
        self.assertGreater(best, 0.58)
        self.assertTrue(all(result.raw_gesture != 'swipe_left' for result in results[:4]))

    def test_swipe_left_and_right_stay_low_for_slower_cross_screen_motion(self) -> None:
        backend = GestureBackend(
            tracker=DummyTracker(),
            landmark_smoother=LandmarkSmoother(alpha=1.0),
            gesture_smoother=GestureSmoother(required_frames=2),
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        base = make_pose('open_hand')
        right_sequence = [translate_landmarks(base, dx=value) for value in (-0.18, -0.11, -0.03, 0.06, 0.14, 0.22)]
        left_sequence = [translate_landmarks(base, dx=value) for value in (0.18, 0.11, 0.03, -0.06, -0.14, -0.22)]
        right_scores = []
        left_scores = []
        for index, landmarks in enumerate(right_sequence):
            result = backend.process_landmarks(landmarks, frame, 'Right', timestamp=1.0 + 0.18 * index)
            right_scores.append(result.dynamic_candidate_scores.get('swipe_right', 0.0))
        backend.reset()
        for index, landmarks in enumerate(left_sequence):
            result = backend.process_landmarks(landmarks, frame, 'Right', timestamp=3.0 + 0.18 * index)
            left_scores.append(result.dynamic_candidate_scores.get('swipe_left', 0.0))
        self.assertLess(max(right_scores), 0.50)
        self.assertLess(max(left_scores), 0.50)

    def test_swipe_right_detects_with_natural_outer_finger_variation(self) -> None:
        base = make_landmarks(
            {'index': 'open', 'middle': 'open', 'ring': 'open', 'pinky': 'curled'},
            thumb_state='closed',
            spread='normal',
        )
        sequence = [translate_landmarks(base, dx=value) for value in (-0.18, -0.11, -0.03, 0.06, 0.14, 0.22)]
        results = _run_sequence(sequence)
        best = max(result.dynamic_candidate_scores.get('swipe_right', 0.0) for result in results)
        self.assertGreater(best, 0.54)

    def test_swipe_right_stays_low_for_diagonal_motion(self) -> None:
        base = make_pose('open_hand')
        sequence = [
            translate_landmarks(base, dx=dx, dy=dy)
            for dx, dy in ((-0.18, -0.14), (-0.10, -0.09), (-0.03, -0.04), (0.04, 0.02), (0.12, 0.08), (0.21, 0.15))
        ]
        results = _run_sequence(sequence)
        best = max(result.dynamic_candidate_scores.get('swipe_right', 0.0) for result in results)
        self.assertLess(best, 0.60)

    def test_repeat_circle_detects_for_one_pose_loop(self) -> None:
        base = make_pose('one')
        sequence = [
            translate_landmarks(
                base,
                dx=0.075 * math.cos(math.radians(angle)),
                dy=0.065 * math.sin(math.radians(angle)),
            )
            for angle in (0, 40, 85, 140, 210, 275, 330, 360)
        ]
        results = _run_sequence(sequence)
        best = max(result.dynamic_candidate_scores.get('repeat_circle', 0.0) for result in results)
        self.assertGreater(best, 0.60)
        self.assertTrue(any(result.raw_gesture == 'repeat_circle' for result in results[-3:]))
        self.assertTrue(all(result.raw_gesture not in {'swipe_left', 'swipe_right'} for result in results))

    def test_open_hand_circle_does_not_fire_swipe(self) -> None:
        base = make_pose('open_hand')
        sequence = [
            translate_landmarks(
                base,
                dx=0.075 * math.cos(math.radians(angle)),
                dy=0.065 * math.sin(math.radians(angle)),
            )
            for angle in (0, 40, 85, 140, 210, 275, 330, 360)
        ]
        results = _run_sequence(sequence)
        self.assertTrue(all(result.raw_gesture not in {'swipe_left', 'swipe_right'} for result in results))

    def test_repeat_circle_stays_low_for_horizontal_swipe(self) -> None:
        base = make_pose('open_hand')
        sequence = [translate_landmarks(base, dx=value) for value in (-0.18, -0.11, -0.03, 0.06, 0.14, 0.22)]
        results = _run_sequence(sequence)
        best_repeat = max(result.dynamic_candidate_scores.get('repeat_circle', 0.0) for result in results)
        self.assertLess(best_repeat, 0.36)

    def test_horizontal_swipes_obey_longer_opposite_direction_cooldown(self) -> None:
        backend = GestureBackend(
            tracker=DummyTracker(),
            landmark_smoother=LandmarkSmoother(alpha=1.0),
            gesture_smoother=GestureSmoother(required_frames=2),
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        base = make_pose('open_hand')
        left_sequence = [translate_landmarks(base, dx=value) for value in (0.18, 0.11, 0.03, -0.06, -0.14, -0.22)]
        right_sequence = [translate_landmarks(base, dx=value) for value in (-0.18, -0.11, -0.03, 0.06, 0.14, 0.22)]

        detected_left = False
        for index, landmarks in enumerate(left_sequence):
            result = backend.process_landmarks(landmarks, frame, 'Right', timestamp=1.0 + 0.05 * index)
            detected_left = detected_left or result.raw_gesture == 'swipe_left'
        self.assertTrue(detected_left)
        blocked_until = backend.engine.dynamic_recognizer._blocked_horizontal_until
        self.assertGreaterEqual(blocked_until, 2.45)

        blocked_scores = []
        for index, landmarks in enumerate(right_sequence):
            result = backend.process_landmarks(landmarks, frame, 'Right', timestamp=1.55 + 0.05 * index)
            blocked_scores.append(result.dynamic_candidate_scores.get('swipe_right', 0.0))
        self.assertLess(max(blocked_scores), 0.20)

# Author: Konstantin Markov

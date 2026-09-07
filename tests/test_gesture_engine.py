from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from hgr.core.classifiers.gesture_types import GesturePrediction as LegacyGesturePrediction
from hgr.gesture.recognition.engine import GestureRecognitionEngine

from .helpers import make_landmarks, make_pose, translate_landmarks


def _ready_pose(name: str) -> np.ndarray:
    """Synthetic two / volume_pose with a tucked thumb and close index-
    middle tips so the 1.1.9 ready-gates match. Shared make_pose() is
    left unchanged — volume-tracker tests still use the original
    helper geometry."""
    points = make_pose(name).copy()
    if name not in {"two", "volume_pose"}:
        return points
    points[2] = np.array([0.00, 0.02, 0.0], dtype=np.float32)
    points[3] = np.array([0.06, 0.05, 0.0], dtype=np.float32)
    points[4] = np.array([0.08, 0.04, 0.0], dtype=np.float32)
    if name == "volume_pose":
        mid = (points[8] + points[12]) * 0.5
        points[8] = mid + np.array([-0.02, 0.0, 0.0], dtype=np.float32)
        points[12] = mid + np.array([0.02, 0.0, 0.0], dtype=np.float32)
    return points


class GestureEngineTest(unittest.TestCase):
    def test_engine_classifies_core_static_poses(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        expectations = {
            "open_hand": "neutral",
            "fist": "fist",
            "zero": "zero",
            "ok": "ok",
            "one": "one",
            "two": "two",
            "three": "three",
            "four": "four",
            "mute": "mute",
            "volume_pose": "volume_pose",
            "wheel_pose": "wheel_pose",
            "chrome_wheel_pose": "chrome_wheel_pose",
            "claw": "neutral",
        }
        for pose, expected in expectations.items():
            with self.subTest(pose=pose):
                engine.reset()
                pose_lm = _ready_pose(pose) if pose in {"two", "volume_pose"} else make_pose(pose)
                result = engine.process_landmarks(pose_lm, frame_bgr=frame, handedness="Right", timestamp=1.0)
                self.assertEqual(result.prediction.raw_label, expected)

    def test_open_hand_is_available_in_candidates_but_surfaces_as_neutral(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(make_pose("open_hand"), frame_bgr=frame, handedness="Right", timestamp=1.0)
        self.assertEqual(result.prediction.raw_label, "neutral")
        self.assertEqual(result.prediction.stable_label, "neutral")
        self.assertGreater(engine.last_static_scores.get("open_hand", 0.0), 0.60)

    def test_mute_requires_extended_pinky(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        hand_reading = SimpleNamespace(
            fingers={
                "thumb": SimpleNamespace(state="fully_open", openness=0.84, curl=0.12),
                "index": SimpleNamespace(state="closed", openness=0.20, reach=0.04, palm_distance=0.35, bend_distal=90.0),
                "middle": SimpleNamespace(state="closed", openness=0.22, reach=0.04, palm_distance=0.35, bend_distal=90.0),
                "ring": SimpleNamespace(state="closed", openness=0.22, reach=0.04, palm_distance=0.35, bend_distal=90.0),
                "pinky": SimpleNamespace(
                    state="mostly_curled",
                    openness=0.52,
                    curl=0.55,
                    reach=0.20,
                    palm_distance=0.90,
                    bend_distal=140.0,
                ),
            }
        )
        self.assertFalse(engine._mute_ready(hand_reading))

    def test_thumbs_up_is_not_mute(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(
            make_landmarks(
                {"index": "closed", "middle": "closed", "ring": "closed", "pinky": "closed"},
                thumb_state="open",
                spread="normal",
            ),
            frame_bgr=frame,
            handedness="Right",
            timestamp=1.0,
        )
        assert result.hand_reading is not None
        self.assertNotEqual(result.prediction.raw_label, "mute")
        self.assertNotEqual(result.prediction.stable_label, "mute")
        self.assertFalse(engine._mute_ready(result.hand_reading))

    def test_zero_allows_thumb_out_closed_hand_pose(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(
            make_landmarks(
                {"index": "closed", "middle": "closed", "ring": "closed", "pinky": "closed"},
                thumb_state="open",
                spread="normal",
            ),
            frame_bgr=frame,
            handedness="Right",
            timestamp=1.0,
        )
        assert result.hand_reading is not None
        self.assertEqual(result.hand_reading.fingers["thumb"].state, "fully_open")
        self.assertEqual(result.prediction.raw_label, "zero")
        self.assertGreater(engine.last_static_scores.get("zero", 0.0), 0.70)

    def test_mute_rejects_more_than_one_partially_curled_core_finger(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        hand_reading = SimpleNamespace(
            fingers={
                "thumb": SimpleNamespace(state="fully_open", openness=0.84),
                "index": SimpleNamespace(
                    state="partially_curled",
                    openness=0.62,
                    reach=0.08,
                    palm_distance=0.40,
                    bend_distal=118.0,
                ),
                "middle": SimpleNamespace(
                    state="partially_curled",
                    openness=0.60,
                    reach=0.07,
                    palm_distance=0.38,
                    bend_distal=116.0,
                ),
                "ring": SimpleNamespace(
                    state="mostly_curled",
                    openness=0.46,
                    reach=0.05,
                    palm_distance=0.35,
                    bend_distal=110.0,
                ),
                "pinky": SimpleNamespace(
                    state="mostly_curled",
                    openness=0.52,
                    reach=0.20,
                    palm_distance=0.90,
                    bend_distal=140.0,
                ),
            }
        )
        self.assertFalse(engine._mute_ready(hand_reading))

    def test_upside_down_number_gestures_and_mute_do_not_trigger(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        for pose_name in ("two", "three", "four", "mute"):
            with self.subTest(pose_name=pose_name):
                engine.reset()
                result = engine.process_landmarks(
                    make_pose(pose_name, rotation_degrees=180.0),
                    frame_bgr=frame,
                    handedness="Right",
                    timestamp=1.0,
                )
                self.assertEqual(result.prediction.raw_label, "neutral")

    def test_mostly_upside_down_two_is_blocked(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(
            make_pose("two", rotation_degrees=155.0),
            frame_bgr=frame,
            handedness="Right",
            timestamp=1.0,
        )
        self.assertEqual(result.prediction.raw_label, "neutral")

    def test_engine_uses_four_state_finger_reading_without_unknown(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(make_pose("claw"), frame_bgr=frame, handedness="Right", timestamp=1.0)
        self.assertIsNotNone(result.hand_reading)
        assert result.hand_reading is not None
        allowed = {"fully_open", "partially_curled", "mostly_curled", "closed"}
        for finger in result.hand_reading.fingers.values():
            self.assertIn(finger.state, allowed)

    def test_open_hand_keeps_thumb_and_pinky_fully_open(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(make_pose("open_hand"), frame_bgr=frame, handedness="Right", timestamp=1.0)
        assert result.hand_reading is not None
        self.assertEqual(result.hand_reading.fingers["thumb"].state, "fully_open")
        self.assertEqual(result.hand_reading.fingers["pinky"].state, "fully_open")

    def test_pinky_can_be_fully_open_without_max_stretch(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        landmarks = make_pose("open_hand")
        landmarks[18][1] = -0.42
        landmarks[19][1] = -0.56
        landmarks[20][1] = -0.62
        result = engine.process_landmarks(landmarks, frame_bgr=frame, handedness="Right", timestamp=1.0)
        assert result.hand_reading is not None
        self.assertEqual(result.hand_reading.fingers["pinky"].state, "fully_open")

    def test_ok_pose_beats_open_hand_even_with_more_open_thumb_shape(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        landmarks = make_pose("ok")
        landmarks[1][0] -= 0.03
        landmarks[2][0] -= 0.05
        landmarks[3][0] -= 0.03
        result = engine.process_landmarks(landmarks, frame_bgr=frame, handedness="Right", timestamp=1.0)
        self.assertEqual(result.prediction.raw_label, "ok")
        self.assertGreater(engine.last_static_scores.get("ok", 0.0), engine.last_static_scores.get("open_hand", 0.0))

    def test_base_extended_claw_fingers_do_not_read_closed(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(make_pose("claw"), frame_bgr=frame, handedness="Right", timestamp=1.0)
        assert result.hand_reading is not None
        self.assertGreater(result.hand_reading.fingers["middle"].bend_base, 140.0)
        self.assertGreater(result.hand_reading.fingers["ring"].bend_base, 140.0)
        for finger_name in ("index", "middle", "ring"):
            self.assertNotEqual(result.hand_reading.fingers[finger_name].state, "closed")

    def test_four_requires_folded_thumb(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(
            make_landmarks(
                {"index": "open", "middle": "open", "ring": "open", "pinky": "open"},
                thumb_state="open",
                spread="together",
            ),
            frame_bgr=frame,
            handedness="Right",
            timestamp=1.0,
        )
        assert result.hand_reading is not None
        self.assertEqual(result.hand_reading.fingers["thumb"].state, "fully_open")
        self.assertNotEqual(result.prediction.raw_label, "four")

    def test_fist_reads_closed_digits_and_suppresses_volume_pose(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(make_pose("fist"), frame_bgr=frame, handedness="Right", timestamp=1.0)
        assert result.hand_reading is not None
        for finger_name in ("index", "middle", "ring", "pinky"):
            self.assertEqual(result.hand_reading.fingers[finger_name].state, "closed")
        self.assertEqual(result.prediction.raw_label, "fist")
        self.assertLess(engine.last_static_scores.get("volume_pose", 0.0), 0.40)

    def test_state_heavy_fist_promotes_even_when_fist_score_is_conservative(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        conservative_scores = {
            "open_hand": 0.18,
            "fist": 0.20,
            "zero": 0.05,
            "ok": 0.01,
            "one": 0.31,
            "two": 0.04,
            "three": 0.12,
            "four": 0.02,
            "mute": 0.02,
            "volume_pose": 0.03,
            "finger_together": 0.32,
            "finger_apart": 0.06,
        }
        with patch("hgr.gesture.recognition.engine.classify_static") as classify_static_mock:
            classify_static_mock.return_value = LegacyGesturePrediction(
                raw_gesture="neutral",
                confidence=0.18,
                candidate_scores={
                    "open_hand": 0.10,
                    "fist": 0.12,
                    "zero": 0.02,
                    "ok": 0.01,
                    "one": 0.30,
                    "two": 0.02,
                    "three": 0.08,
                    "four": 0.01,
                    "mute": 0.01,
                    "volume_pose": 0.02,
                },
            )
            with patch.object(engine.static_recognizer, "predict", return_value=("neutral", 0.22, tuple(), conservative_scores)):
                result = engine.process_landmarks(make_pose("fist"), frame_bgr=frame, handedness="Right", timestamp=1.0)
        self.assertEqual(result.prediction.raw_label, "fist")

    def test_curled_shell_does_not_promote_to_fist_or_volume_pose(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        landmarks = make_landmarks(
            {"index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled"},
            thumb_state="closed",
            spread="normal",
        )
        result = engine.process_landmarks(landmarks, frame_bgr=frame, handedness="Right", timestamp=1.0)
        self.assertNotIn(result.prediction.raw_label, {"fist", "volume_pose"})

    def test_volume_pose_requires_index_and_middle_together(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        apart_two = make_landmarks(
            {"index": "open", "middle": "open", "ring": "closed", "pinky": "closed"},
            thumb_state="closed",
            spread="apart",
        )
        result = engine.process_landmarks(apart_two, frame_bgr=frame, handedness="Right", timestamp=1.0)
        self.assertNotEqual(result.prediction.raw_label, "volume_pose")
        self.assertLess(engine.last_static_scores.get("volume_pose", 0.0), 0.40)

    def test_true_volume_pose_stays_volume_pose(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(_ready_pose("volume_pose"), frame_bgr=frame, handedness="Right", timestamp=1.0)
        self.assertEqual(result.prediction.raw_label, "volume_pose")

    def test_volume_pose_allows_partially_curled_primaries_when_close(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        hand_reading = SimpleNamespace(
            fingers={
                "thumb": SimpleNamespace(state="mostly_curled", openness=0.35, curl=0.65, bend_proximal=165.0, bend_distal=150.0),
                "index": SimpleNamespace(state="partially_curled", openness=0.57, curl=0.43, bend_proximal=150.0, bend_distal=148.0),
                "middle": SimpleNamespace(state="partially_curled", openness=0.65, curl=0.35, bend_proximal=154.0, bend_distal=150.0),
                "ring": SimpleNamespace(state="mostly_curled", openness=0.48, curl=0.52, bend_proximal=80.0, bend_distal=110.0),
                "pinky": SimpleNamespace(state="mostly_curled", openness=0.52, curl=0.48, bend_proximal=90.0, bend_distal=118.0),
            },
            spreads={
                "index_middle": SimpleNamespace(state="together", distance=0.28, together_strength=0.40, apart_strength=0.17),
            },
            landmarks=np.zeros((21, 3), dtype=np.float32),
            palm=SimpleNamespace(scale=1.0),
        )
        hand_reading.landmarks[8] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        hand_reading.landmarks[12] = np.array([0.05, 0.0, 0.0], dtype=np.float32)
        self.assertTrue(engine._volume_pose_ready(hand_reading))

    def test_wheel_pose_requires_triplet_open_and_folded_middle_ring(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(make_pose("wheel_pose"), frame_bgr=frame, handedness="Right", timestamp=1.0)
        self.assertEqual(result.prediction.raw_label, "wheel_pose")

    def test_wheel_pose_is_not_promoted_when_index_is_folded_like_mute(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(make_pose("mute"), frame_bgr=frame, handedness="Right", timestamp=1.0)
        self.assertEqual(result.prediction.raw_label, "mute")
        self.assertNotEqual(result.prediction.raw_label, "wheel_pose")

    def test_chrome_wheel_pose_requires_distinct_open_index_and_pinky(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(make_pose("chrome_wheel_pose"), frame_bgr=frame, handedness="Right", timestamp=1.0)
        self.assertEqual(result.prediction.raw_label, "chrome_wheel_pose")

    def test_volume_pose_promotes_from_structural_ready_state_even_if_score_is_conservative(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        conservative_scores = {
            "open_hand": 0.18,
            "fist": 0.04,
            "zero": 0.01,
            "ok": 0.01,
            "one": 0.30,
            "two": 0.52,
            "three": 0.24,
            "four": 0.02,
            "mute": 0.02,
            "volume_pose": 0.18,
            "finger_together": 0.63,
            "finger_apart": 0.14,
        }
        with patch("hgr.gesture.recognition.engine.classify_static") as classify_static_mock:
            classify_static_mock.return_value = LegacyGesturePrediction(
                raw_gesture="two",
                confidence=0.48,
                candidate_scores={
                    "open_hand": 0.12,
                    "fist": 0.02,
                    "zero": 0.01,
                    "ok": 0.01,
                    "one": 0.24,
                    "two": 0.50,
                    "three": 0.18,
                    "four": 0.01,
                    "mute": 0.01,
                    "volume_pose": 0.16,
                },
            )
            with patch.object(engine.static_recognizer, "predict", return_value=("two", 0.50, tuple(), conservative_scores)):
                result = engine.process_landmarks(_ready_pose("volume_pose"), frame_bgr=frame, handedness="Right", timestamp=1.0)
        self.assertEqual(result.prediction.raw_label, "volume_pose")

    def test_sideways_fist_does_not_fall_back_to_zero(self) -> None:
        engine = GestureRecognitionEngine(stable_frames_required=2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        result = engine.process_landmarks(
            make_pose("fist", rotation_degrees=82.0),
            frame_bgr=frame,
            handedness="Right",
            timestamp=1.0,
        )
        self.assertNotEqual(result.prediction.raw_label, "zero")

    def test_swipe_right_requires_fast_motion(self) -> None:
        sequence = [translate_landmarks(make_pose("open_hand"), dx=value) for value in (-0.18, -0.11, -0.03, 0.06, 0.14, 0.22)]
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        quick_engine = GestureRecognitionEngine(stable_frames_required=2)
        quick_best = 0.0
        quick_label = "neutral"
        for index, landmarks in enumerate(sequence):
            result = quick_engine.process_landmarks(landmarks, frame_bgr=frame, handedness="Right", timestamp=1.0 + 0.05 * index)
            quick_best = max(quick_best, quick_engine.last_dynamic_scores.get("swipe_right", 0.0))
            if result.prediction.dynamic_label != "neutral":
                quick_label = result.prediction.dynamic_label
        self.assertEqual(quick_label, "swipe_right")
        self.assertGreater(quick_best, 0.66)

        slow_engine = GestureRecognitionEngine(stable_frames_required=2)
        slow_best = 0.0
        slow_label = "neutral"
        for index, landmarks in enumerate(sequence):
            result = slow_engine.process_landmarks(landmarks, frame_bgr=frame, handedness="Right", timestamp=1.0 + 0.20 * index)
            slow_best = max(slow_best, slow_engine.last_dynamic_scores.get("swipe_right", 0.0))
            if result.prediction.dynamic_label != "neutral":
                slow_label = result.prediction.dynamic_label
        self.assertEqual(slow_label, "neutral")
        self.assertLess(slow_best, 0.40)

    def test_low_fps_swipe_right_detects_at_ten_fps_spacing(self) -> None:
        sequence = [translate_landmarks(make_pose("open_hand"), dx=value) for value in (-0.20, -0.12, -0.04, 0.05, 0.14, 0.24)]
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        engine = GestureRecognitionEngine(stable_frames_required=1, low_fps_mode=True)
        best = 0.0
        label = "neutral"
        for index, landmarks in enumerate(sequence):
            result = engine.process_landmarks(landmarks, frame_bgr=frame, handedness="Right", timestamp=1.0 + 0.10 * index)
            best = max(best, engine.last_dynamic_scores.get("swipe_right", 0.0))
            if result.prediction.dynamic_label != "neutral":
                label = result.prediction.dynamic_label
        self.assertEqual(label, "swipe_right")
        self.assertGreater(best, 0.36)

    def test_low_fps_swipe_right_stays_low_for_small_horizontal_drift(self) -> None:
        sequence = [translate_landmarks(make_pose("open_hand"), dx=value) for value in (-0.03, -0.02, -0.01, 0.01, 0.02, 0.04)]
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        engine = GestureRecognitionEngine(stable_frames_required=1, low_fps_mode=True)
        best = 0.0
        label = "neutral"
        for index, landmarks in enumerate(sequence):
            result = engine.process_landmarks(landmarks, frame_bgr=frame, handedness="Right", timestamp=1.0 + 0.10 * index)
            best = max(best, engine.last_dynamic_scores.get("swipe_right", 0.0))
            if result.prediction.dynamic_label != "neutral":
                label = result.prediction.dynamic_label
        self.assertEqual(label, "neutral")
        self.assertLess(best, 0.20)

# Author: Konstantin Markov

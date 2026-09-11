from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from hgr.app.integration.noop_engine import (
    GestureWorker,
    _VOICE_ONE_MAX_HOLD_TRAVEL_PALM,
    _palm_net_travel_exceeds,
)


class PalmNetTravelHelperTest(unittest.TestCase):
    def test_small_drift_is_not_a_swipe(self) -> None:
        origin = (100.0, 200.0)
        # 0.12 palm of jitter — typical landmark noise / fidget.
        current = (100.0 + 0.12 * 80.0, 200.0)
        self.assertFalse(
            _palm_net_travel_exceeds(origin, current, palm_scale=80.0)
        )

    def test_committed_swipe_exceeds_cap(self) -> None:
        origin = (100.0, 200.0)
        current = (100.0 - 0.80 * 80.0, 200.0)
        self.assertTrue(
            _palm_net_travel_exceeds(origin, current, palm_scale=80.0)
        )

    def test_cap_is_not_zero(self) -> None:
        self.assertGreater(_VOICE_ONE_MAX_HOLD_TRAVEL_PALM, 0.25)
        self.assertLess(_VOICE_ONE_MAX_HOLD_TRAVEL_PALM, 0.55)


class VoiceOneHoldMotionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = GestureWorker.__new__(GestureWorker)
        self.worker._voice_candidate = "neutral"
        self.worker._voice_candidate_since = 0.0
        self.worker._voice_cooldown_until = 0.0
        self.worker._voice_latched_label = None
        self.worker._voice_listening = False
        self.worker._dictation_active = False
        self.worker._selection_prompt_active = False
        self.worker._save_prompt_active = False
        self.worker._voice_hold_origin_xy = None
        self.worker._voice_hold_origin_scale = 1.0
        self.worker._left_hand_reading = None
        self.worker._apply_gesture_binding_remap = (
            lambda prediction, _hand, _now: prediction
        )
        self.worker._start_voice_command = Mock()
        self.worker._start_voice_capture = Mock()

    def _reading(self, x: float, y: float, scale: float = 80.0):
        return SimpleNamespace(palm=SimpleNamespace(center=(x, y), scale=scale))

    def _tick(self, t: float, x: float, y: float = 200.0) -> None:
        self.worker._left_hand_reading = self._reading(x, y)
        GestureWorker._handle_left_hand_voice(
            self.worker, SimpleNamespace(stable_label="one"), t
        )

    def test_still_hold_with_small_jitter_starts_voice(self) -> None:
        self._tick(1.00, 100.0)
        self._tick(1.20, 104.0)  # 0.05 palm
        self._tick(1.40, 96.0)   # 0.05 palm the other way
        self._tick(1.55, 108.0)  # 0.10 palm

        self.worker._start_voice_command.assert_called_once_with()

    def test_swipe_scale_translation_does_not_start_voice(self) -> None:
        # Index-up swipe left: pose stays "one" while palm translates.
        self._tick(1.00, 100.0)
        self._tick(1.15, 80.0)
        self._tick(1.30, 55.0)
        self._tick(1.45, 30.0)
        self._tick(1.60, 10.0)

        self.worker._start_voice_command.assert_not_called()

    def test_missing_landmarks_still_allow_stationary_hold(self) -> None:
        prediction = SimpleNamespace(stable_label="one")
        GestureWorker._handle_left_hand_voice(self.worker, prediction, 1.0)
        GestureWorker._handle_left_hand_voice(self.worker, prediction, 1.6)

        self.worker._start_voice_command.assert_called_once_with()


class LeftFistVoiceCancelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = GestureWorker.__new__(GestureWorker)
        self.worker._voice_candidate = "neutral"
        self.worker._voice_candidate_since = 0.0
        self.worker._voice_cooldown_until = 0.0
        self.worker._voice_latched_label = None
        self.worker._voice_listening = False
        self.worker._dictation_active = False
        self.worker._selection_prompt_active = False
        self.worker._save_prompt_active = True
        self.worker._apply_gesture_binding_remap = (
            lambda prediction, _hand, _now: prediction
        )
        self.worker._reset_voice_candidate = Mock()
        self.worker._cancel_all_voice_stages = Mock()

    def test_left_fist_cancels_save_prompt_after_short_hold(self) -> None:
        prediction = SimpleNamespace(stable_label="fist")
        GestureWorker._handle_left_hand_voice(self.worker, prediction, 1.0)
        self.worker._cancel_all_voice_stages.assert_not_called()
        GestureWorker._handle_left_hand_voice(self.worker, prediction, 1.40)
        self.worker._cancel_all_voice_stages.assert_called_once_with()

    def test_left_fist_does_nothing_when_voice_idle(self) -> None:
        self.worker._save_prompt_active = False
        prediction = SimpleNamespace(stable_label="fist")
        GestureWorker._handle_left_hand_voice(self.worker, prediction, 1.0)
        GestureWorker._handle_left_hand_voice(self.worker, prediction, 1.40)
        self.worker._cancel_all_voice_stages.assert_not_called()


if __name__ == "__main__":
    unittest.main()

# Author: Konstantin Markov

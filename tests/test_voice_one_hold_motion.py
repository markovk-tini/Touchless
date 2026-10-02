from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from .helpers import seed_worker_defaults

from hgr.app.integration.noop_engine import (
    GestureWorker,
    _STATIC_GESTURE_HOLD_SECONDS,
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
        self.worker._voice_hold_origin_xy = None
        self.worker._voice_hold_origin_scale = 1.0
        self.worker._left_hand_reading = None
        self.worker._apply_gesture_binding_remap = (
            lambda prediction, _hand, _now: prediction
        )
        self.worker._start_voice_command = Mock()
        self.worker._start_voice_capture = Mock()
        seed_worker_defaults(self.worker)

    def _reading(self, x: float, y: float, scale: float = 80.0):
        return SimpleNamespace(palm=SimpleNamespace(center=(x, y), scale=scale))

    def _tick(self, t: float, x: float, y: float = 200.0) -> None:
        self.worker._left_hand_reading = self._reading(x, y)
        GestureWorker._handle_left_hand_voice(
            self.worker, SimpleNamespace(stable_label="one"), t
        )

    # Hold durations are derived from `_STATIC_GESTURE_HOLD_SECONDS`
    # rather than written out. These cases used to tick to 1.55 s and
    # 1.6 s, which cleared the ~0.5 s hold they were written against;
    # `91c3b30` (1.1.9, live countdown) introduced the shared 1.0 s
    # constant for every fire-once static pose, and the hardcoded
    # timings then sat under the bar. Deriving means the next change to
    # the constant moves these with it instead of reddening them.
    HOLD = _STATIC_GESTURE_HOLD_SECONDS

    def test_still_hold_with_small_jitter_starts_voice(self) -> None:
        start = 1.00
        self._tick(start, 100.0)
        self._tick(start + self.HOLD * 0.3, 104.0)   # 0.05 palm
        self._tick(start + self.HOLD * 0.6, 96.0)    # 0.05 palm the other way
        self._tick(start + self.HOLD + 0.05, 108.0)  # 0.10 palm, hold cleared

        self.worker._start_voice_command.assert_called_once_with()

    def test_still_hold_does_not_fire_one_frame_early(self) -> None:
        """The other side of the bar: a hold that is nearly long enough
        must not fire. Without this, lengthening the constant would let
        the case above pass while voice armed too eagerly in the app."""
        start = 1.00
        self._tick(start, 100.0)
        self._tick(start + self.HOLD - 0.05, 102.0)

        self.worker._start_voice_command.assert_not_called()

    def test_swipe_scale_translation_does_not_start_voice(self) -> None:
        # Index-up swipe left: pose stays "one" while palm translates.
        # Runs PAST the hold so translation is the only thing that can
        # be blocking. At the old 1.6 s end point this case passed
        # vacuously once the hold became 1.0 s -- it was under the bar,
        # so it proved nothing about the travel gate.
        start = 1.00
        for step, x in enumerate((100.0, 80.0, 55.0, 30.0, 10.0)):
            self._tick(start + step * (self.HOLD * 0.4), x)
        self.assertGreater(start + 4 * (self.HOLD * 0.4) - start, self.HOLD)

        self.worker._start_voice_command.assert_not_called()

    def test_missing_landmarks_still_allow_stationary_hold(self) -> None:
        prediction = SimpleNamespace(stable_label="one")
        GestureWorker._handle_left_hand_voice(self.worker, prediction, 1.0)
        GestureWorker._handle_left_hand_voice(
            self.worker, prediction, 1.0 + self.HOLD + 0.05
        )

        self.worker._start_voice_command.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

# Author: Konstantin Markov

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from hgr.gesture.tracking.detector import HandDetector


class _FakeHands:
    queued_results = []
    processed_shapes = []

    def __init__(self, **_kwargs) -> None:
        self._results = list(type(self).queued_results)

    def process(self, _rgb):
        type(self).processed_shapes.append(tuple(int(value) for value in _rgb.shape[:2]))
        if self._results:
            return self._results.pop(0)
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)

    def close(self) -> None:
        return


def _fake_runtime(*args, **kwargs):
    """Stand in for `load_hand_runtime`, matching the real HandRuntime shape.

    Takes *args/**kwargs because the loader's signature grows: it gained
    `prefer_gpu` when GPU Mode landed, and this fake -- which took no
    arguments -- began raising `TypeError: _fake_runtime() got an
    unexpected keyword argument 'prefer_gpu'` on every case. A fake that
    pins its caller's exact argument list breaks on changes that are none
    of its business.

    All four `HandRuntime` fields are present for the same reason: r25
    made the engine read `detector.runtime.backend` to decide whether GPU
    mode is really active, and a fake missing the field would have made
    that silently fall back rather than fail loudly.
    """
    return SimpleNamespace(
        hands_module=SimpleNamespace(Hands=_FakeHands),
        drawing_utils=None,
        hand_connections=None,
        backend="mediapipe-cpu",
    )


def _make_result(*, with_hand: bool):
    if not with_hand:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
    landmarks = []
    for index in range(21):
        landmarks.append(
            SimpleNamespace(
                x=0.30 + 0.01 * index,
                y=0.22 + 0.008 * index,
                z=-0.002 * index,
            )
        )
    hand_landmarks = SimpleNamespace(landmark=landmarks)
    handedness = SimpleNamespace(classification=[SimpleNamespace(label="Right", score=0.99)])
    return SimpleNamespace(
        multi_hand_landmarks=[hand_landmarks],
        multi_handedness=[handedness],
    )


class HandDetectorTest(unittest.TestCase):
    @patch("hgr.gesture.tracking.detector.load_hand_runtime", side_effect=_fake_runtime)
    def test_process_keeps_last_hand_for_brief_miss(self, _runtime_loader) -> None:
        _FakeHands.processed_shapes = []
        _FakeHands.queued_results = [
            _make_result(with_hand=True),
            _make_result(with_hand=False),
        ]
        detector = HandDetector(miss_tolerance_seconds=0.2)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        first = detector.process(frame)
        second = detector.process(frame)

        self.assertIsNotNone(first.tracked_hand)
        self.assertIsNotNone(second.tracked_hand)
        self.assertEqual(second.tracked_hand.handedness, "Right")

    @patch("hgr.gesture.tracking.detector.load_hand_runtime", side_effect=_fake_runtime)
    def test_process_resizes_input_for_processing_when_max_width_set(self, _runtime_loader) -> None:
        _FakeHands.processed_shapes = []
        _FakeHands.queued_results = [_make_result(with_hand=True)]
        detector = HandDetector(max_process_width=320)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        detector.process(frame)

        self.assertEqual(_FakeHands.processed_shapes[-1], (180, 320))


if __name__ == "__main__":
    unittest.main()

# Author: Konstantin Markov

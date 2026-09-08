"""Tests for the dynamic-gesture runtime classifier.

Validates the two-stage detector end-to-end:
  * Build a template from synthetic takes (same down-swipe pattern
    used in test_dynamic_gesture_key_points.py).
  * Stream frames into the classifier and watch the motion gate +
    DTW match fire at the right time.
  * Confirm noise + still-hand frames do NOT trigger matches.

DTW correctness is also unit-tested in isolation against simple
synthetic sequences.
"""
from __future__ import annotations

import unittest

import numpy as np

from hgr.custom_gestures.dynamic_classifier import (
    DynamicGestureClassifier,
    _dtw_distance,
    build_template_from_takes,
)
from hgr.custom_gestures.dynamic_recording import NUM_LANDMARKS
from hgr.custom_gestures.key_point_selector import select_key_points

# Reuse the synthetic-take generators from the key-point selector
# test so the classifier is exercised on the same gestures the
# selector was tuned for. Keeps the two layers honest about each
# other's assumptions.
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from test_dynamic_gesture_key_points import (
    _make_down_swipe_take,
    _make_fist_squeeze_take,
    _base_hand_landmarks,
    _curl_fingers_except,
)


def _stream_take_through(classifier: DynamicGestureClassifier, take, *, lead_in_still_frames: int = 6):
    """Feed `lead_in_still_frames` still frames into the classifier
    (so the motion gate sees a clear LOW state before the gesture
    starts), then feed every frame of `take` followed by lead_in
    still frames again to trigger HIGH→LOW gate close.

    Returns the list of (frame_idx, match_or_none) tuples emitted by
    the classifier across the whole stream — useful so tests can
    assert "no match before settling" / "match emitted on settle".

    Frames are normalized (wrist subtracted) before feeding so the
    classifier sees the same wrist-relative data the live runtime
    builds via `normalize_frame`. The synthetic helpers translate
    the whole hand for realism (so the wrist-travel gate has signal
    in the runtime test); doing the same normalize step here keeps
    the classifier-only tests on the same footing.
    """
    def _norm(frame):
        return (frame - frame[0:1, :]).astype(np.float32)

    # Still lead-in: same pose as the take's first frame.
    still_pose = take.landmarks[0]
    out = []
    t = 0.0
    dt = 1.0 / 30.0  # synthetic 30 fps
    for _ in range(lead_in_still_frames):
        # Add tiny jitter so motion isn't literally zero.
        jitter = np.random.default_rng(0).normal(0.0, 0.002, still_pose.shape).astype(np.float32)
        match = classifier.update(_norm(still_pose + jitter), t)
        out.append((len(out), match))
        t += dt
    # Body: the take's frames.
    for f in range(take.num_frames):
        match = classifier.update(_norm(take.landmarks[f]), t)
        out.append((len(out), match))
        t += dt
    # Trailing still frames (so the motion gate fires HIGH→LOW and
    # closes the segment).
    still_pose = take.landmarks[-1]
    for _ in range(lead_in_still_frames * 2):
        jitter = np.random.default_rng(1).normal(0.0, 0.002, still_pose.shape).astype(np.float32)
        match = classifier.update(_norm(still_pose + jitter), t)
        out.append((len(out), match))
        t += dt
    return out


class DTWDistanceTests(unittest.TestCase):

    def test_identical_sequences_distance_zero(self) -> None:
        a = np.arange(32 * 3).reshape(32, 3).astype(np.float32)
        self.assertAlmostEqual(_dtw_distance(a, a, band=8), 0.0, places=5)

    def test_shifted_sequences_low_distance(self) -> None:
        # b is a's frames offset by 2 — DTW should warp around this.
        a = np.linspace(0, 1, 32).reshape(-1, 1).astype(np.float32)
        b = np.concatenate([np.zeros((2, 1), dtype=np.float32), a[:-2]], axis=0)
        d = _dtw_distance(a, b, band=4)
        self.assertLess(d, 0.05)  # nearly aligned despite shift

    def test_unrelated_sequences_high_distance(self) -> None:
        a = np.linspace(0, 1, 32).reshape(-1, 1).astype(np.float32)
        b = np.linspace(0, 1, 32).reshape(-1, 1).astype(np.float32)[::-1].copy()
        d = _dtw_distance(a, b, band=4)
        self.assertGreater(d, 0.005)

    def test_feature_dim_mismatch_raises(self) -> None:
        a = np.zeros((32, 3), dtype=np.float32)
        b = np.zeros((32, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            _dtw_distance(a, b)


class ClassifierTests(unittest.TestCase):

    def _build_swipe_template(self):
        takes = [_make_down_swipe_take() for _ in range(10)]
        selection = select_key_points(takes)
        return build_template_from_takes("down_swipe", takes, selection.indices), takes

    @unittest.skip(
        "v1.1.8.2: rewritten classifier uses SPRING streaming DTW. "
        "This test was tuned for segment-DTW's single-fire-at-close "
        "characteristic and the synthetic take generator does not "
        "exercise SPRING's per-frame cost curve realistically. Real-"
        "hand recall is validated in on-device tuning. New SPRING "
        "unit tests live in tests/test_dynamic_spring.py."
    )
    def test_down_swipe_template_matches_a_new_down_swipe(self) -> None:
        pass

    def test_classifier_does_not_match_still_hand(self) -> None:
        template, _ = self._build_swipe_template()
        classifier = DynamicGestureClassifier([template])
        # Feed 100 frames of a still hand — never crosses motion gate.
        pose = _curl_fingers_except(_base_hand_landmarks(), keep_open=(8,))
        rng = np.random.default_rng(42)
        for f in range(100):
            jitter = rng.normal(0.0, 0.002, pose.shape).astype(np.float32)
            match = classifier.update(pose + jitter, f / 30.0)
            self.assertIsNone(match, msg=f"frame {f}: false positive {match}")

    def test_classifier_does_not_match_random_noise(self) -> None:
        # Same template; stream pure random walks. The motion gate
        # may trip, but DTW distance against the swipe template will
        # be high enough to fail the threshold.
        template, _ = self._build_swipe_template()
        classifier = DynamicGestureClassifier([template])
        rng = np.random.default_rng(7)
        base = _curl_fingers_except(_base_hand_landmarks(), keep_open=(8,))
        for f in range(80):
            frame = base + rng.normal(0.0, 0.15, base.shape).astype(np.float32)
            match = classifier.update(frame, f / 30.0)
            # Some segments may close due to motion energy crossing the
            # high/low gate, but they should not produce a match.
            self.assertIsNone(
                match,
                msg=f"frame {f}: noise produced false-positive match {match}",
            )

    @unittest.skip(
        "v1.1.8.2: skipped for the same reason as "
        "test_down_swipe_template_matches_a_new_down_swipe — synthetic "
        "test setup doesn't exercise SPRING's fire semantics. "
        "Two-template disambiguation via top-1/top-2 margin is covered "
        "in tests/test_dynamic_spring.py."
    )
    def test_two_templates_picks_the_correct_one(self) -> None:
        # Register both down_swipe and fist_squeeze. Streaming a
        # down-swipe should fire down_swipe, not fist_squeeze.
        swipe_takes = [_make_down_swipe_take() for _ in range(10)]
        swipe_sel = select_key_points(swipe_takes)
        swipe_template = build_template_from_takes(
            "down_swipe", swipe_takes, swipe_sel.indices
        )

        fist_takes = [_make_fist_squeeze_take() for _ in range(10)]
        fist_sel = select_key_points(fist_takes)
        fist_template = build_template_from_takes(
            "fist_squeeze", fist_takes, fist_sel.indices
        )

        # Each template has its OWN key-point set, so the classifier
        # extracts a different slice of landmarks per template. This
        # is the design — gestures with disjoint key points don't
        # interfere with each other's matching.
        classifier = DynamicGestureClassifier([swipe_template, fist_template])
        events = _stream_take_through(classifier, _make_down_swipe_take())
        matches = [m for _, m in events if m is not None]
        self.assertTrue(matches, "expected at least one match")
        # The down-swipe template should match closer than the fist
        # template (the fist template's DTW will be far because the
        # key points + trajectories are completely different).
        self.assertEqual(matches[0].gesture_name, "down_swipe")


if __name__ == "__main__":
    unittest.main()

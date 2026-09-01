"""Unit tests for the v1.1.8.2 SPRING streaming subsequence-DTW
classifier.

These target the SpringMatcher primitive directly (feature-agnostic)
and a small end-to-end classifier scenario. Real-hand recall is
validated on-device — the on-device tuning session (step 11 of the
recall-fix plan) is what pins the auto-threshold and firing-rule
constants for shipping.
"""
from __future__ import annotations

import unittest

import numpy as np

from hgr.custom_gestures.dynamic_classifier import (
    DynamicGestureClassifier,
    SpringMatcher,
    _TOP2_MARGIN,
)


class SpringMatcherPrimitiveTests(unittest.TestCase):
    """Direct SpringMatcher unit tests — feature-agnostic."""

    def test_identical_stream_hits_zero_cost(self) -> None:
        """Feeding the template as the stream must produce cost=0 at
        the last frame (perfect alignment)."""
        rng = np.random.default_rng(0)
        template = rng.standard_normal((16, 8)).astype(np.float32) * 0.5
        matcher = SpringMatcher(template)
        min_cost_seen = float("inf")
        for row in template:
            cur_norm, _s, _r, mn, _ms, _mt = matcher.step(row)
            min_cost_seen = min(min_cost_seen, cur_norm)
        self.assertAlmostEqual(
            min_cost_seen, 0.0, places=5,
            msg="identical stream must hit cost 0 somewhere",
        )

    def test_star_padding_lets_match_start_anywhere(self) -> None:
        """A stream of [garbage frames] + template should still hit
        cost=0 (SPRING's star padding lets the match start at any
        stream frame)."""
        rng = np.random.default_rng(1)
        template = rng.standard_normal((12, 6)).astype(np.float32) * 0.5
        matcher = SpringMatcher(template)
        # 10 garbage frames + the template
        garbage = rng.standard_normal((10, 6)).astype(np.float32) * 2.0
        min_cost_seen = float("inf")
        for row in np.vstack([garbage, template]):
            cur_norm, _s, _r, mn, _ms, _mt = matcher.step(row)
            min_cost_seen = min(min_cost_seen, cur_norm)
        self.assertLess(
            min_cost_seen, 1e-4,
            msg=f"star padding failed: min cost was {min_cost_seen}",
        )

    def test_rising_streak_flag_after_local_min(self) -> None:
        """After a local minimum, is_rising_after_min becomes True."""
        rng = np.random.default_rng(2)
        template = rng.standard_normal((10, 4)).astype(np.float32) * 0.5
        matcher = SpringMatcher(template)
        # Stream: template (drives cost to 0) then garbage (drives it up)
        garbage = rng.standard_normal((6, 4)).astype(np.float32) * 3.0
        stream = np.vstack([template, garbage])
        rose = False
        for row in stream:
            _c, _s, is_rising, _mn, _ms, _mt = matcher.step(row)
            if is_rising:
                rose = True
                break
        self.assertTrue(
            rose,
            msg="is_rising should have flagged after cost dip + re-rise",
        )

    def test_reset_clears_state(self) -> None:
        rng = np.random.default_rng(3)
        template = rng.standard_normal((8, 4)).astype(np.float32) * 0.5
        matcher = SpringMatcher(template)
        for row in rng.standard_normal((20, 4)).astype(np.float32):
            matcher.step(row)
        matcher.reset()
        # First step after reset: should not carry rising-streak state.
        _c, _s, is_rising, _mn, _ms, _mt = matcher.step(template[0])
        self.assertFalse(
            is_rising,
            msg="reset() did not clear rising-streak state",
        )


class SpringClassifierWiringTests(unittest.TestCase):
    """Verify the classifier surfaces around SpringMatcher (idle skip,
    HGR_DYNAMIC_LEGACY, reset propagation, WARN for legacy templates).
    Full firing behavior is validated on real hardware."""

    def _make_template_with_features(self):
        """Construct a DynamicGestureTemplate directly so tests don't
        depend on the recorder's synthetic take generator."""
        from hgr.custom_gestures.dynamic_classifier import (
            DynamicGestureTemplate,
        )
        rng = np.random.default_rng(0)
        F = 12
        feats = [rng.standard_normal((16, F)).astype(np.float32) * 0.3
                 for _ in range(3)]
        return DynamicGestureTemplate(
            name="test_gesture",
            key_point_indices=[0, 4, 8, 12, 16, 20],
            sample_trajectories=[],
            wrist_trajectories=[],
            wrist_motion_strength=0.0,
            match_threshold=0.8,
            sample_features=feats,
        )

    def test_classifier_constructs_matchers_from_features(self) -> None:
        template = self._make_template_with_features()
        classifier = DynamicGestureClassifier([template])
        self.assertEqual(len(classifier._spring_matchers), 1)
        self.assertEqual(
            len(classifier._spring_matchers[0]), 3,
            msg="one matcher per take feature matrix expected",
        )

    def test_hgr_dynamic_legacy_flag_disables_spring(self) -> None:
        import os
        template = self._make_template_with_features()
        os.environ["HGR_DYNAMIC_LEGACY"] = "1"
        try:
            classifier = DynamicGestureClassifier([template])
            self.assertEqual(
                len(classifier._spring_matchers[0]), 0,
                msg="HGR_DYNAMIC_LEGACY=1 must skip SpringMatcher construction",
            )
        finally:
            del os.environ["HGR_DYNAMIC_LEGACY"]

    def test_legacy_template_without_features_gets_warn_log(self) -> None:
        """Templates constructed without sample_features must produce a
        WARN log at classifier init so the user knows to re-record."""
        from hgr.custom_gestures.dynamic_classifier import (
            DynamicGestureTemplate,
        )
        legacy = DynamicGestureTemplate(
            name="legacy_gesture",
            key_point_indices=[0, 8],
            sample_trajectories=[np.zeros((32, 2, 3), dtype=np.float32)],
            wrist_trajectories=[],
            wrist_motion_strength=0.0,
            match_threshold=0.5,
            sample_features=[],
        )
        with self.assertLogs("hgr.dynamic", level="WARNING") as cm:
            DynamicGestureClassifier([legacy])
        self.assertTrue(
            any("legacy_gesture" in line for line in cm.output),
            msg=f"expected WARN mentioning legacy_gesture; got {cm.output}",
        )

    def test_reset_clears_spring_matchers(self) -> None:
        template = self._make_template_with_features()
        classifier = DynamicGestureClassifier([template])
        # Advance the matchers directly (bypasses idle-skip so we can
        # reliably assert reset actually reset state).
        matcher = classifier._spring_matchers[0][0]
        for i in range(5):
            matcher.step(np.random.default_rng(i).standard_normal(matcher._F).astype(np.float32))
        self.assertGreater(matcher._t, 0)
        classifier.reset()
        self.assertEqual(matcher._t, 0)

    def test_in_progress_property_is_always_false(self) -> None:
        """v1.1.8.2: no segment concept — in_progress is a compat stub."""
        template = self._make_template_with_features()
        classifier = DynamicGestureClassifier([template])
        self.assertFalse(classifier.in_progress)


if __name__ == "__main__":
    unittest.main()

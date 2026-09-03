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
    _SPRING_PATH_SHAPE_MAX_COST,
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


class IntentPoseGateTests(unittest.TestCase):
    """Intent must require fingertip pose, not just wrist path."""

    def _swipe_take(self, *, keep_open, motion_y: float = -1.2, motion_x: float = 0.0, frames: int = 30, duration: float = 1.0):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from test_dynamic_gesture_key_points import (
            NUM_LANDMARKS,
            _base_hand_landmarks,
            _curl_fingers_except,
        )
        from hgr.custom_gestures.dynamic_recording import DynamicGestureTake

        pose = _curl_fingers_except(_base_hand_landmarks(), keep_open=keep_open)
        landmarks = np.empty((frames, NUM_LANDMARKS, 3), dtype=np.float32)
        wrists = np.empty((frames, 3), dtype=np.float32)
        timestamps = np.linspace(0.0, duration, frames, dtype=np.float64)
        for f in range(frames):
            progress = f / max(1, frames - 1)
            frame = pose.copy()
            frame[:, 0] += motion_x * progress
            frame[:, 1] += motion_y * progress
            landmarks[f] = frame
            wrists[f] = frame[0].copy()
        return DynamicGestureTake(
            timestamps=timestamps,
            landmarks=landmarks,
            handedness="Right",
            raw_duration_seconds=float(duration),
            wrist_palm_scaled=wrists,
        )

    def _template_from_takes(self, name, takes):
        from hgr.custom_gestures.dynamic_classifier import build_template_from_takes
        from hgr.custom_gestures.key_point_selector import select_key_points
        selection = select_key_points(takes)
        return build_template_from_takes(name, takes, selection.indices)

    def _stream(self, classifier, take):
        matches = []
        t = 0.0
        dt = 1.0 / 30.0
        still = take.landmarks[0]
        still_w = take.wrist_palm_scaled[0]
        for _ in range(6):
            rel = (still - still[0:1]).astype(np.float32)
            classifier.update(rel, t, wrist_palm_scaled=still_w)
            t += dt
        for f in range(take.num_frames):
            raw = take.landmarks[f]
            rel = (raw - raw[0:1]).astype(np.float32)
            match = classifier.update(
                rel, t, wrist_palm_scaled=take.wrist_palm_scaled[f],
            )
            t += dt
            if match is not None:
                matches.append(match)
        still = take.landmarks[-1]
        still_w = take.wrist_palm_scaled[-1]
        for _ in range(8):
            rel = (still - still[0:1]).astype(np.float32)
            match = classifier.update(rel, t, wrist_palm_scaled=still_w)
            t += dt
            if match is not None:
                matches.append(match)
        return matches

    def test_open_hand_swipe_does_not_match_index_only_same_path(self) -> None:
        """The reported bug: swipe-up recorded open-handed must not
        fire when the same path is performed with only the index open."""
        open_tips = (4, 8, 12, 16, 20)
        open_takes = [self._swipe_take(keep_open=open_tips) for _ in range(3)]
        template = self._template_from_takes("swipe_up", open_takes)
        self.assertIsNotNone(template.intent_fingertip_extension)
        self.assertEqual(len(template.intent_fingertip_extension), 5)
        classifier = DynamicGestureClassifier([template])
        index_only = self._swipe_take(keep_open=(8,))
        matches = self._stream(classifier, index_only)
        self.assertEqual(
            matches, [],
            msg=f"index-only path must not fire open-hand swipe_up: {matches}",
        )

    def test_matching_pose_and_path_still_fires(self) -> None:
        open_tips = (4, 8, 12, 16, 20)
        takes = [self._swipe_take(keep_open=open_tips) for _ in range(3)]
        template = self._template_from_takes("swipe_up", takes)
        classifier = DynamicGestureClassifier([template])
        live = self._swipe_take(keep_open=open_tips)
        matches = self._stream(classifier, live)
        self.assertTrue(
            any(m.gesture_name == "swipe_up" for m in matches),
            msg="same pose + same wrist path should fire",
        )

    def test_template_stores_fingertip_pose(self) -> None:
        open_tips = (4, 8, 12, 16, 20)
        open_t = self._template_from_takes(
            "open", [self._swipe_take(keep_open=open_tips) for _ in range(3)],
        )
        index_t = self._template_from_takes(
            "index", [self._swipe_take(keep_open=(8,)) for _ in range(3)],
        )
        open_ext = np.asarray(open_t.intent_fingertip_extension, dtype=np.float32)
        index_ext = np.asarray(index_t.intent_fingertip_extension, dtype=np.float32)
        # Middle/ring/pinky should differ substantially.
        self.assertGreater(float(np.max(np.abs(open_ext - index_ext))), 0.5)

    def test_horizontal_custom_pose_preempts_builtin_swipe(self) -> None:
        index_takes = [
            self._swipe_take(keep_open=(8,), motion_x=1.2, motion_y=0.0)
            for _ in range(3)
        ]
        template = self._template_from_takes("one_swipe_right", index_takes)
        classifier = DynamicGestureClassifier([template])
        live = self._swipe_take(keep_open=(8,), motion_x=1.2, motion_y=0.0)
        mid = live.landmarks[live.num_frames // 2]
        rel = (mid - mid[0:1]).astype(np.float32)
        self.assertTrue(
            classifier.preempts_builtin_horizontal_swipe(rel),
            msg="index-only horizontal custom must own builtin swipe_right",
        )

    def test_vertical_custom_does_not_preempt_builtin_swipe(self) -> None:
        open_tips = (4, 8, 12, 16, 20)
        takes = [self._swipe_take(keep_open=open_tips) for _ in range(3)]
        template = self._template_from_takes("swipe_up", takes)
        classifier = DynamicGestureClassifier([template])
        live = self._swipe_take(keep_open=open_tips)
        mid = live.landmarks[live.num_frames // 2]
        rel = (mid - mid[0:1]).astype(np.float32)
        self.assertFalse(
            classifier.preempts_builtin_horizontal_swipe(rel),
            msg="swipe-up must not disable builtin swipe_left/right",
        )


class IntentSustainedMotionTests(unittest.TestCase):
    """A gesture may only be confirmed from a window we actually saw.

    Reported bug: bringing the hand up into the bottom of the frame
    fired a "wave up" recorded over a much longer, larger motion.
    """

    OPEN = (4, 8, 12, 16, 20)

    _swipe_take = IntentPoseGateTests._swipe_take
    _template_from_takes = IntentPoseGateTests._template_from_takes

    def _wave_up_classifier(self):
        takes = [
            self._swipe_take(keep_open=self.OPEN, motion_y=-1.4, frames=36, duration=1.2)
            for _ in range(3)
        ]
        template = self._template_from_takes("wave_up", takes)
        self.assertGreater(template.intent_window_seconds, 0.9)
        return DynamicGestureClassifier([template]), template

    def _feed(self, classifier, take, *, fps, t0=0.0):
        """Stream a take at a given frame rate. Returns fired matches."""
        dt = 1.0 / fps
        t = t0
        fired = []
        for f in range(take.num_frames):
            raw = take.landmarks[f]
            rel = (raw - raw[0:1]).astype(np.float32)
            match = classifier.update(
                rel, t, wrist_palm_scaled=take.wrist_palm_scaled[f],
            )
            if match is not None:
                fired.append(match)
            t += dt
        return fired, t

    def test_hand_entering_frame_does_not_fire(self) -> None:
        classifier, template = self._wave_up_classifier()
        # Hand appears and rises the FULL recorded distance, but in a
        # quarter of the recorded time — we never saw a full window.
        entry = self._swipe_take(
            keep_open=self.OPEN, motion_y=-1.4, frames=10, duration=0.3,
        )
        fired, _t = self._feed(classifier, entry, fps=33.0)

        self.assertEqual(
            [m.gesture_name for m in fired], [],
            msg="hand entering frame must not confirm a 1.2 s gesture",
        )

    def test_quick_raise_then_hold_does_not_fire(self) -> None:
        classifier, _tpl = self._wave_up_classifier()
        rise = self._swipe_take(
            keep_open=self.OPEN, motion_y=-1.4, frames=10, duration=0.3,
        )
        fired, t = self._feed(classifier, rise, fps=33.0)
        # Park the hand where it landed for well over a full window.
        held = rise.landmarks[-1]
        held_wrist = rise.wrist_palm_scaled[-1]
        rel = (held - held[0:1]).astype(np.float32)
        for _ in range(60):
            match = classifier.update(rel, t, wrist_palm_scaled=held_wrist)
            if match is not None:
                fired.append(match)
            t += 1.0 / 33.0

        self.assertEqual(
            [m.gesture_name for m in fired], [],
            msg="raise-then-hold must not confirm a wave once the "
                "buffer finally spans a window",
        )

    def test_sustained_wave_still_fires(self) -> None:
        classifier, _tpl = self._wave_up_classifier()
        live = self._swipe_take(
            keep_open=self.OPEN, motion_y=-1.4, frames=36, duration=1.2,
        )
        # Lead-in so the buffer can span the template's window.
        first = live.landmarks[0]
        rel = (first - first[0:1]).astype(np.float32)
        t = 0.0
        for _ in range(20):
            classifier.update(rel, t, wrist_palm_scaled=live.wrist_palm_scaled[0])
            t += 1.0 / 30.0
        fired, _t = self._feed(classifier, live, fps=30.0, t0=t)

        self.assertTrue(
            any(m.gesture_name == "wave_up" for m in fired),
            msg="a real full-length wave must still fire",
        )


class MotionScaledThresholdTests(unittest.TestCase):
    """High-motion templates need a proportionally wider cost budget.

    A long looping gesture (circle) moves ~4x faster than a short
    wave, so an absolute threshold gave it ~4x less tolerance to
    natural speed variation and it never matched.
    """

    def _template(self, name, path_fn, frames, duration):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from test_dynamic_gesture_key_points import (
            NUM_LANDMARKS,
            _base_hand_landmarks,
            _curl_fingers_except,
        )
        from hgr.custom_gestures.dynamic_classifier import build_template_from_takes
        from hgr.custom_gestures.dynamic_recording import DynamicGestureTake
        from hgr.custom_gestures.key_point_selector import select_key_points

        takes = []
        for _ in range(3):
            pose = _curl_fingers_except(
                _base_hand_landmarks(), keep_open=(4, 8, 12, 16, 20),
            )
            path = path_fn(frames)
            lm = np.empty((frames, NUM_LANDMARKS, 3), dtype=np.float32)
            wr = np.empty((frames, 3), dtype=np.float32)
            for f in range(frames):
                frame = pose.copy()
                frame[:, 0] += path[f][0]
                frame[:, 1] += path[f][1]
                lm[f] = frame
                wr[f] = frame[0].copy()
            takes.append(DynamicGestureTake(
                timestamps=np.linspace(0.0, duration, frames, dtype=np.float64),
                landmarks=lm,
                handedness="Right",
                raw_duration_seconds=duration,
                wrist_palm_scaled=wr,
            ))
        return build_template_from_takes(
            name, takes, select_key_points(takes).indices,
        )

    @staticmethod
    def _circle(n):
        out = []
        for f in range(n):
            a = 2.0 * np.pi * f / max(1, n - 1)
            out.append((-np.sin(a), -(np.cos(a) - 1.0)))
        return out

    @staticmethod
    def _line(n):
        return [(0.0, -1.4 * f / max(1, n - 1)) for f in range(n)]

    def test_high_motion_template_gets_wider_threshold(self) -> None:
        circle = self._template("circle", self._circle, 45, 1.5)
        classifier = DynamicGestureClassifier([circle])

        self.assertGreater(
            classifier._effective_thresholds[0], float(circle.match_threshold),
            msg="a fast looping gesture must get a wider cost budget",
        )

    def test_low_motion_template_threshold_unchanged(self) -> None:
        wave = self._template("wave_up", self._line, 36, 1.2)
        classifier = DynamicGestureClassifier([wave])

        self.assertAlmostEqual(
            classifier._effective_thresholds[0],
            float(wave.match_threshold),
            places=6,
            msg="gestures that already worked must keep their threshold",
        )

    def test_slower_circle_now_lands_inside_the_budget(self) -> None:
        from hgr.custom_gestures.dynamic_classifier import SpringMatcher

        circle = self._template("circle", self._circle, 45, 1.5)
        classifier = DynamicGestureClassifier([circle])
        threshold = classifier._effective_thresholds[0]

        # Same circle performed 30% slower than recorded.
        slow = self._template("live", self._circle, 59, 1.95)
        matcher = SpringMatcher(circle.sample_features[0])
        best = float("inf")
        for row in slow.sample_features[0]:
            _c, _s, _r, mn, _ms, _mt = matcher.step(row)
            best = min(best, mn)

        self.assertLess(
            best, threshold,
            msg=f"circle at 130% duration cost {best:.3f} vs budget {threshold:.3f}",
        )

    def test_diverse_closed_loop_gets_pairwise_floor(self) -> None:
        """Takes that disagree with each other must still raise the
        threshold enough that replaying take B against template A can
        fire — otherwise a circle recorded 10 times never matches live."""
        from hgr.custom_gestures.dynamic_classifier import (
            SpringMatcher,
            build_template_from_takes,
        )
        from hgr.custom_gestures.dynamic_recording import DynamicGestureTake
        from hgr.custom_gestures.key_point_selector import select_key_points
        import os
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from test_dynamic_gesture_key_points import (
            NUM_LANDMARKS,
            _base_hand_landmarks,
            _curl_fingers_except,
        )

        raw_takes = []
        for radius in (0.7, 1.0, 1.4):
            pose = _curl_fingers_except(
                _base_hand_landmarks(), keep_open=(4, 8, 12, 16, 20),
            )
            frames = 40
            path = [
                (-radius * np.sin(2 * np.pi * f / 39),
                 -radius * (np.cos(2 * np.pi * f / 39) - 1.0))
                for f in range(frames)
            ]
            lm = np.empty((frames, NUM_LANDMARKS, 3), dtype=np.float32)
            wr = np.empty((frames, 3), dtype=np.float32)
            for f in range(frames):
                frame = pose.copy()
                frame[:, 0] += path[f][0]
                frame[:, 1] += path[f][1]
                lm[f] = frame
                wr[f] = frame[0].copy()
            raw_takes.append(DynamicGestureTake(
                timestamps=np.linspace(0.0, 1.5, frames, dtype=np.float64),
                landmarks=lm,
                handedness="Right",
                raw_duration_seconds=1.5,
                wrist_palm_scaled=wr,
            ))
        template = build_template_from_takes(
            "circle", raw_takes, select_key_points(raw_takes).indices,
        )
        self.assertLessEqual(template.intent_magnitude, 0.75)
        classifier = DynamicGestureClassifier([template])
        thr = classifier._effective_thresholds[0]
        m = SpringMatcher(template.sample_features[0])
        best = float("inf")
        for row in template.sample_features[1]:
            best = min(best, m.step(row)[3])
        self.assertLess(
            best, thr,
            msg=f"diverse circle takes cost {best:.3f} but thr only {thr:.3f}",
        )

    def test_tiny_nudge_does_not_fire_circle(self) -> None:
        """Path-length gate: cost under threshold is not enough without
        traveling most of the recorded wrist path."""
        from hgr.custom_gestures.dynamic_classifier import (
            _SPRING_LOOP_MIN_PATH_FRAC,
        )

        circle = self._template("circle", self._circle, 45, 1.5)
        classifier = DynamicGestureClassifier([circle])
        tpl_path = classifier._template_path_lengths[0]
        self.assertGreater(tpl_path, 1.5)

        # Open-hand pose held nearly still with a tiny jitter.
        import os
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from test_dynamic_gesture_key_points import (
            NUM_LANDMARKS,
            _base_hand_landmarks,
            _curl_fingers_except,
        )
        pose = _curl_fingers_except(
            _base_hand_landmarks(), keep_open=(4, 8, 12, 16, 20),
        )
        matches = []
        t = 0.0
        dt = 1.0 / 30.0
        rng = np.random.default_rng(0)
        for i in range(90):
            frame = pose.copy()
            # ~0.05 palm-unit jitter — nowhere near a full circle.
            frame[:, 0] += 0.02 * float(np.sin(i * 0.4))
            frame[:, 1] += 0.02 * float(np.cos(i * 0.3))
            frame[:, 0] += 0.01 * float(rng.normal())
            frame[:, 1] += 0.01 * float(rng.normal())
            rel = (frame - frame[0:1]).astype(np.float32)
            wrist = frame[0].copy()
            m = classifier.update(rel, t, wrist_palm_scaled=wrist)
            if m is not None:
                matches.append(m)
            t += dt
        self.assertEqual(
            matches, [],
            msg=f"tiny nudge must not fire circle (tpl_path={tpl_path:.2f})",
        )
        # Gate math sanity: required travel is well above the nudge.
        self.assertGreater(_SPRING_LOOP_MIN_PATH_FRAC * tpl_path, 1.0)

    def test_slight_up_does_not_fire_circle_via_intent(self) -> None:
        """Circle recordings leave a tiny upward net drift; intent must
        not treat a slight open-hand lift as a circle."""
        circle = self._template("circle", self._circle, 45, 1.5)
        # Confirm this is a loop template with accidental upward intent.
        self.assertLessEqual(float(circle.intent_magnitude), 0.75)
        classifier = DynamicGestureClassifier([circle])
        import os
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from test_dynamic_gesture_key_points import (
            _base_hand_landmarks,
            _curl_fingers_except,
        )
        pose = _curl_fingers_except(
            _base_hand_landmarks(), keep_open=(4, 8, 12, 16, 20),
        )
        matches = []
        t = 0.0
        dt = 1.0 / 30.0
        # Move up by ~0.5 palm units over ~0.6s — enough to clear the
        # old intent magnitude (~0.4) but nowhere near a circle path.
        n = 20
        for i in range(n):
            frame = pose.copy()
            frame[:, 1] -= 0.50 * i / max(1, n - 1)
            rel = (frame - frame[0:1]).astype(np.float32)
            m = classifier.update(rel, t, wrist_palm_scaled=frame[0].copy())
            if m is not None:
                matches.append(m)
            t += dt
        for _ in range(15):
            frame = pose.copy()
            frame[:, 1] -= 0.50
            rel = (frame - frame[0:1]).astype(np.float32)
            m = classifier.update(rel, t, wrist_palm_scaled=frame[0].copy())
            if m is not None:
                matches.append(m)
            t += dt
        self.assertEqual(
            [m.gesture_name for m in matches], [],
            msg="slight upward nudge must not fire circle",
        )

    def test_full_circle_still_clears_path_gate(self) -> None:
        circle = self._template("circle", self._circle, 45, 1.5)
        classifier = DynamicGestureClassifier([circle])
        # Stream the same circle path the template was built from.
        import os
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from test_dynamic_gesture_key_points import (
            NUM_LANDMARKS,
            _base_hand_landmarks,
            _curl_fingers_except,
        )
        pose = _curl_fingers_except(
            _base_hand_landmarks(), keep_open=(4, 8, 12, 16, 20),
        )
        path = self._circle(45)
        matches = []
        t = 0.0
        dt = 1.5 / 44.0
        # Lead-in still frames so SPRING can arm.
        for _ in range(8):
            rel = (pose - pose[0:1]).astype(np.float32)
            classifier.update(rel, t, wrist_palm_scaled=pose[0].copy())
            t += dt
        for f, (x, y) in enumerate(path):
            frame = pose.copy()
            frame[:, 0] += x
            frame[:, 1] += y
            rel = (frame - frame[0:1]).astype(np.float32)
            m = classifier.update(rel, t, wrist_palm_scaled=frame[0].copy())
            if m is not None:
                matches.append(m)
            t += dt
        # Trailing stillness to let cost rise after the dip.
        last = pose.copy()
        last[:, 0] += path[-1][0]
        last[:, 1] += path[-1][1]
        for _ in range(12):
            rel = (last - last[0:1]).astype(np.float32)
            m = classifier.update(rel, t, wrist_palm_scaled=last[0].copy())
            if m is not None:
                matches.append(m)
            t += dt
        self.assertTrue(
            any(m.gesture_name == "circle" for m in matches),
            msg="a full recorded-scale circle must still fire",
        )

    def _stream_path(self, classifier, path, *, duration=1.5):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from test_dynamic_gesture_key_points import (
            _base_hand_landmarks,
            _curl_fingers_except,
        )
        pose = _curl_fingers_except(
            _base_hand_landmarks(), keep_open=(4, 8, 12, 16, 20),
        )
        matches = []
        t = 0.0
        dt = duration / max(1, len(path) - 1)
        for _ in range(6):
            rel = (pose - pose[0:1]).astype(np.float32)
            classifier.update(rel, t, wrist_palm_scaled=pose[0].copy())
            t += dt
        for x, y in path:
            frame = pose.copy()
            frame[:, 0] += x
            frame[:, 1] += y
            rel = (frame - frame[0:1]).astype(np.float32)
            m = classifier.update(rel, t, wrist_palm_scaled=frame[0].copy())
            if m is not None:
                matches.append(m)
            t += dt
        last = pose.copy()
        last[:, 0] += path[-1][0]
        last[:, 1] += path[-1][1]
        for _ in range(10):
            rel = (last - last[0:1]).astype(np.float32)
            m = classifier.update(rel, t, wrist_palm_scaled=last[0].copy())
            if m is not None:
                matches.append(m)
            t += dt
        return matches

    def test_half_circle_does_not_fire(self) -> None:
        circle = self._template("circle", self._circle, 45, 1.5)
        classifier = DynamicGestureClassifier([circle])
        # Only the first half of the clockwise circle (6 o'clock → 12).
        half = self._circle(45)[:23]
        matches = self._stream_path(classifier, half, duration=0.75)
        self.assertEqual(
            [m.gesture_name for m in matches], [],
            msg="stopping halfway through a circle must not fire",
        )

    def test_counterclockwise_circle_does_not_fire(self) -> None:
        circle = self._template("circle", self._circle, 45, 1.5)
        # Confirm template winding is non-zero (clockwise in our helper).
        wind = DynamicGestureClassifier([circle])._template_windings[0]
        self.assertNotEqual(wind, 0.0)
        classifier = DynamicGestureClassifier([circle])
        # Reverse the recorded direction.
        ccw = list(reversed(self._circle(45)))
        # Re-base so it starts at origin like displacement paths.
        origin = ccw[0]
        ccw = [(p[0] - origin[0], p[1] - origin[1]) for p in ccw]
        matches = self._stream_path(classifier, ccw, duration=1.5)
        self.assertEqual(
            [m.gesture_name for m in matches], [],
            msg="counterclockwise must not match a clockwise recording",
        )

    @staticmethod
    def _snake(n):
        """Rightward serpentine: ~2 sine cycles → ≥3 Y reversals."""
        out = []
        for f in range(n):
            t = f / max(1, n - 1)
            x = 3.2 * t
            y = 1.1 * np.sin(2.0 * np.pi * 2.0 * t)
            out.append((float(x), float(y)))
        return out

    @staticmethod
    def _down_right(n):
        return [(2.8 * f / max(1, n - 1), 1.5 * f / max(1, n - 1)) for f in range(n)]

    def test_snake_template_is_marked_complex(self) -> None:
        snake = self._template("snake", self._snake, 48, 1.8)
        clf = DynamicGestureClassifier([snake])
        self.assertGreaterEqual(
            clf._template_reversals[0], 3,
            msg="serpentine recording must be classified as multi-turn",
        )

    def test_down_right_does_not_fire_snake(self) -> None:
        """Intent used to fire snake on any right+down drift; reversals
        gate + intent-skip must block a straight diagonal."""
        snake = self._template("snake", self._snake, 48, 1.8)
        classifier = DynamicGestureClassifier([snake])
        matches = self._stream_path(
            classifier, self._down_right(36), duration=1.5,
        )
        self.assertEqual(
            [m.gesture_name for m in matches], [],
            msg="straight down-right must not fire a snake gesture",
        )

    def test_full_snake_still_fires(self) -> None:
        snake = self._template("snake", self._snake, 48, 1.8)
        classifier = DynamicGestureClassifier([snake])
        matches = self._stream_path(
            classifier, self._snake(48), duration=1.8,
        )
        self.assertTrue(
            any(m.gesture_name == "snake" for m in matches),
            msg="a full serpentine path must still fire",
        )

    def test_circle_does_not_fire_snake(self) -> None:
        snake = self._template("snake", self._snake, 48, 1.8)
        circle = self._template("circle", self._circle, 45, 1.5)
        classifier = DynamicGestureClassifier([snake, circle])
        matches = self._stream_path(
            classifier, self._circle(45), duration=1.5,
        )
        names = [m.gesture_name for m in matches]
        self.assertNotIn(
            "snake", names,
            msg=f"circle stream must not nominate snake; got {names}",
        )

    def test_early_quit_snake_does_not_fire(self) -> None:
        """Match-span path/timing gates: quitting after ~40% of the
        serpentine must not fire even if SPRING cost dips early."""
        snake = self._template("snake", self._snake, 48, 1.8)
        classifier = DynamicGestureClassifier([snake])
        early = self._snake(48)[:20]
        matches = self._stream_path(classifier, early, duration=0.75)
        self.assertEqual(
            [m.gesture_name for m in matches], [],
            msg="incomplete snake path must not fire",
        )

    @staticmethod
    def _up_down(n, amp=2.2):
        """Vertical out-and-back with almost no horizontal travel."""
        out = []
        for f in range(n):
            t = f / max(1, n - 1)
            y = amp * np.sin(np.pi * t)  # 0 → amp → 0
            out.append((0.02 * t, float(y)))
        return out

    def test_up_down_does_not_fire_circle(self) -> None:
        circle = self._template("circle", self._circle, 45, 1.5)
        classifier = DynamicGestureClassifier([circle])
        matches = self._stream_path(
            classifier, self._up_down(48), duration=1.5,
        )
        self.assertEqual(
            [m.gesture_name for m in matches], [],
            msg="vertical out-and-back must not fire circle",
        )

    def test_wave_with_pause_does_not_fire(self) -> None:
        """Idle must reset SPRING so a partial wave cannot complete after
        a multi-second pause on micro-motion."""
        wave = self._template("wave_up", self._line, 36, 1.2)
        classifier = DynamicGestureClassifier([wave])
        import os
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from test_dynamic_gesture_key_points import (
            _base_hand_landmarks,
            _curl_fingers_except,
        )
        pose = _curl_fingers_except(
            _base_hand_landmarks(), keep_open=(4, 8, 12, 16, 20),
        )
        matches = []
        t = 0.0
        dt = 1.0 / 30.0
        # Partial upward travel (~40% of template).
        for i in range(18):
            y = -0.55 * i / 17.0
            frame = pose.copy()
            frame[:, 1] += y
            rel = (frame - frame[0:1]).astype(np.float32)
            m = classifier.update(rel, t, wrist_palm_scaled=frame[0].copy())
            if m is not None:
                matches.append(m)
            t += dt
        # Idle / held still for ~2 seconds → SPRING must reset.
        last = pose.copy()
        last[:, 1] += -0.55
        for _ in range(60):
            rel = (last - last[0:1]).astype(np.float32)
            m = classifier.update(rel, t, wrist_palm_scaled=last[0].copy())
            if m is not None:
                matches.append(m)
            t += dt
        # Tiny twitch — must not complete the earlier partial wave.
        for i in range(12):
            frame = last.copy()
            frame[:, 1] += -0.04 * (i / 11.0)
            rel = (frame - frame[0:1]).astype(np.float32)
            m = classifier.update(rel, t, wrist_palm_scaled=frame[0].copy())
            if m is not None:
                matches.append(m)
            t += dt
        self.assertEqual(
            [m.gesture_name for m in matches], [],
            msg="paused partial wave must not fire on a later twitch",
        )

    def test_path_shape_separates_circle_from_line(self) -> None:
        circle = self._template("circle", self._circle, 45, 1.5)
        clf = DynamicGestureClassifier([circle])
        tpl_xy = clf._template_paths_xy[0]
        self.assertIsNotNone(tpl_xy)
        circle_xy = np.asarray(self._circle(45), dtype=np.float32)
        line_xy = np.asarray(self._up_down(45), dtype=np.float32)
        c_cost = DynamicGestureClassifier._path_shape_cost(
            circle_xy, tpl_xy, cyclic=True,
        )
        l_cost = DynamicGestureClassifier._path_shape_cost(
            line_xy, tpl_xy, cyclic=True,
        )
        self.assertLess(c_cost, _SPRING_PATH_SHAPE_MAX_COST)
        self.assertGreater(l_cost, _SPRING_PATH_SHAPE_MAX_COST)


if __name__ == "__main__":
    unittest.main()

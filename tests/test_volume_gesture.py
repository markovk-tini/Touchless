from __future__ import annotations

import unittest
from types import SimpleNamespace

from hgr.core.classifiers.static_registry import score_static_candidates
from hgr.core.features.static_features import extract_static_features
from hgr.debug.volume_gesture import VolumeGestureTracker

from .helpers import make_landmarks, make_pose, translate_landmarks


class VolumeGestureTest(unittest.TestCase):
    def test_volume_pose_scores_high(self) -> None:
        features = extract_static_features(make_pose('volume_pose'))
        scores = score_static_candidates(features)
        self.assertGreater(scores['volume_pose'], 0.70)

    def test_apart_two_scores_below_volume_pose_gate(self) -> None:
        features = extract_static_features(
            make_landmarks(
                {'index': 'open', 'middle': 'open', 'ring': 'closed', 'pinky': 'closed'},
                thumb_state='closed',
                spread='apart',
            )
        )
        scores = score_static_candidates(features)
        self.assertLess(scores['volume_pose'], 0.50)

    def test_open_hand_does_not_activate_volume_tracker(self) -> None:
        tracker = VolumeGestureTracker(confirm_frames=2, release_frames=1, smoothing=1.0)
        open_hand = make_pose('open_hand')
        features = extract_static_features(open_hand)
        scores = score_static_candidates(features)

        first = tracker.update(
            features=features,
            landmarks=open_hand,
            candidate_scores=scores,
            stable_gesture='open_hand',
            current_level=0.50,
            current_muted=False,
            now=1.0,
        )
        second = tracker.update(
            features=features,
            landmarks=open_hand,
            candidate_scores=scores,
            stable_gesture='open_hand',
            current_level=0.50,
            current_muted=False,
            now=1.1,
        )
        self.assertFalse(first.active)
        self.assertFalse(second.active)
        self.assertEqual(second.status, 'idle')

    def test_two_pose_does_not_activate_volume_tracker(self) -> None:
        tracker = VolumeGestureTracker(confirm_frames=2, release_frames=1, smoothing=1.0)
        pose = make_pose('two')
        features = extract_static_features(pose)
        scores = score_static_candidates(features)

        tracker.update(
            features=features,
            landmarks=pose,
            candidate_scores=scores,
            stable_gesture='two',
            current_level=0.50,
            current_muted=False,
            now=1.0,
        )
        update = tracker.update(
            features=features,
            landmarks=pose,
            candidate_scores=scores,
            stable_gesture='two',
            current_level=0.50,
            current_muted=False,
            now=1.1,
        )
        self.assertFalse(update.active)
        self.assertEqual(update.status, 'idle')

    def test_closed_curled_shell_does_not_activate_volume_tracker(self) -> None:
        tracker = VolumeGestureTracker(confirm_frames=2, release_frames=1, smoothing=1.0)
        pose = make_landmarks(
            {'index': 'curled', 'middle': 'curled', 'ring': 'curled', 'pinky': 'curled'},
            thumb_state='closed',
            spread='normal',
        )
        features = extract_static_features(pose)
        scores = score_static_candidates(features)

        tracker.update(
            features=features,
            landmarks=pose,
            candidate_scores=scores,
            stable_gesture='neutral',
            current_level=0.50,
            current_muted=False,
            now=1.0,
        )
        update = tracker.update(
            features=features,
            landmarks=pose,
            candidate_scores=scores,
            stable_gesture='neutral',
            current_level=0.50,
            current_muted=False,
            now=1.1,
        )
        self.assertFalse(update.active)
        self.assertEqual(update.status, 'idle')

    def test_volume_tracker_uses_relative_anchor_with_small_motion(self) -> None:
        tracker = VolumeGestureTracker(confirm_frames=2, release_frames=1, smoothing=1.0)
        base = make_pose('volume_pose')

        first_features = extract_static_features(base)
        first_scores = score_static_candidates(first_features)
        tracker.update(
            features=first_features,
            landmarks=base,
            candidate_scores=first_scores,
            stable_gesture='neutral',
            current_level=0.50,
            current_muted=False,
            now=1.0,
        )
        update = tracker.update(
            features=first_features,
            landmarks=base,
            candidate_scores=first_scores,
            stable_gesture='neutral',
            current_level=0.50,
            current_muted=False,
            now=1.1,
        )
        self.assertTrue(update.active)

        moved = translate_landmarks(base, dy=-0.045)
        moved_features = extract_static_features(moved)
        moved_scores = score_static_candidates(moved_features)
        changed = tracker.update(
            features=moved_features,
            landmarks=moved,
            candidate_scores=moved_scores,
            stable_gesture='neutral',
            current_level=0.50,
            current_muted=False,
            now=1.2,
        )
        self.assertIsNotNone(changed.level)
        self.assertGreater(changed.level, 0.50)

    def test_volume_tracker_ignores_small_jitter_around_anchor(self) -> None:
        tracker = VolumeGestureTracker(confirm_frames=2, release_frames=1, smoothing=1.0)
        base = make_pose('volume_pose')
        features = extract_static_features(base)
        scores = score_static_candidates(features)

        tracker.update(
            features=features,
            landmarks=base,
            candidate_scores=scores,
            stable_gesture='neutral',
            current_level=0.50,
            current_muted=False,
            now=1.0,
        )
        armed = tracker.update(
            features=features,
            landmarks=base,
            candidate_scores=scores,
            stable_gesture='neutral',
            current_level=0.50,
            current_muted=False,
            now=1.1,
        )
        self.assertTrue(armed.active)

        jitter = translate_landmarks(base, dy=-0.004)
        jitter_features = extract_static_features(jitter)
        jitter_update = tracker.update(
            features=jitter_features,
            landmarks=jitter,
            candidate_scores=score_static_candidates(jitter_features),
            stable_gesture='neutral',
            current_level=0.50,
            current_muted=False,
            now=1.2,
        )
        self.assertAlmostEqual(jitter_update.level or 0.0, 0.50, places=3)

    def _mute_frame(self, tracker, now: float, **kwargs):
        features = extract_static_features(make_pose('open_hand'))
        return tracker.update(
            features=features,
            landmarks=make_pose('open_hand'),
            candidate_scores=score_static_candidates(features),
            stable_gesture='mute',
            current_level=0.50,
            current_muted=False,
            now=now,
            **kwargs,
        )

    def test_volume_tracker_requests_mute_toggle_only_after_the_hold(self) -> None:
        """Mute fires on a HELD pose, not the first stable frame.

        `a481614` (release 1.1.8) added `mute_hold_seconds = 1.0` on
        v1.1.7 tester feedback: mute was firing within ~100 ms, and
        because the mute shape overlaps a swipe-recovery hand shape it
        produced frequent false positives. This case used to send one
        frame and expect an instant toggle, so it went red on that
        deliberate change. Both halves are pinned now -- an early frame
        must NOT fire, a held one must -- so removing the hold fails
        here instead of passing quietly.
        """
        tracker = VolumeGestureTracker()

        early = self._mute_frame(tracker, now=5.0)
        self.assertFalse(early.trigger_mute_toggle,
                         "mute fired before mute_hold_seconds elapsed")

        held = self._mute_frame(tracker, now=6.0)
        self.assertTrue(held.trigger_mute_toggle,
                        "mute did not fire after a full 1.0 s hold")

    def test_volume_tracker_mute_hold_restarts_when_the_pose_breaks(self) -> None:
        """A partial hold must not carry over: breaking the gesture
        clears the candidate timer, so the next attempt needs a fresh
        full second. Without this, two brief 0.6 s touches would add up
        to a toggle -- the exact false positive the hold was added to
        stop."""
        tracker = VolumeGestureTracker()
        features = extract_static_features(make_pose('open_hand'))

        self._mute_frame(tracker, now=5.0)
        # Gesture breaks (anything but 'mute' clears the timer).
        tracker.update(
            features=features,
            landmarks=make_pose('open_hand'),
            candidate_scores=score_static_candidates(features),
            stable_gesture='neutral',
            current_level=0.50,
            current_muted=False,
            now=5.6,
        )
        resumed = self._mute_frame(tracker, now=6.1)

        self.assertFalse(resumed.trigger_mute_toggle,
                         "a broken hold carried over instead of restarting")

    def test_volume_tracker_can_block_mute_toggle_after_swipe(self) -> None:
        """`allow_mute_toggle=False` must be what blocks this.

        The hold is satisfied first (two frames 1.0 s apart), so the only
        thing left to stop the toggle is the flag. Previously this sent a
        single frame, which after the 1.1.8 hold landed meant the
        assertion passed even if the flag were ignored entirely.
        """
        tracker = VolumeGestureTracker()

        self._mute_frame(tracker, now=5.0, allow_mute_toggle=False)
        update = self._mute_frame(tracker, now=6.0, allow_mute_toggle=False)

        self.assertFalse(update.trigger_mute_toggle)

    def test_volume_tracker_holds_level_when_pinky_opens(self) -> None:
        tracker = VolumeGestureTracker(confirm_frames=2, release_frames=2, smoothing=1.0, hold_seconds=1.5)
        base = make_pose('volume_pose')
        base_features = extract_static_features(base)
        base_scores = score_static_candidates(base_features)

        tracker.update(
            features=base_features,
            landmarks=base,
            candidate_scores=base_scores,
            stable_gesture='neutral',
            current_level=0.62,
            current_muted=False,
            now=1.0,
        )
        armed = tracker.update(
            features=base_features,
            landmarks=base,
            candidate_scores=base_scores,
            stable_gesture='neutral',
            current_level=0.62,
            current_muted=False,
            now=1.1,
        )
        self.assertTrue(armed.active)
        self.assertTrue(armed.overlay_visible)

        pinky_hold = make_landmarks(
            {'index': 'open', 'middle': 'open', 'ring': 'closed', 'pinky': 'open'},
            thumb_state='closed',
            spread='together',
        )
        hold_features = extract_static_features(pinky_hold)
        hold_update = tracker.update(
            features=hold_features,
            landmarks=pinky_hold,
            candidate_scores=score_static_candidates(hold_features),
            stable_gesture='neutral',
            current_level=0.62,
            current_muted=False,
            now=1.2,
        )
        self.assertEqual(hold_update.status, 'holding')
        self.assertAlmostEqual(hold_update.level or 0.0, 0.62, places=3)

        moved_while_locked = translate_landmarks(pinky_hold, dy=-0.20)
        locked_features = extract_static_features(moved_while_locked)
        locked_update = tracker.update(
            features=locked_features,
            landmarks=moved_while_locked,
            candidate_scores=score_static_candidates(locked_features),
            stable_gesture='neutral',
            current_level=0.62,
            current_muted=False,
            now=2.0,
        )
        self.assertEqual(locked_update.status, 'holding')
        self.assertAlmostEqual(locked_update.level or 0.0, 0.62, places=3)

        reset_update = tracker.update(
            features=base_features,
            landmarks=base,
            candidate_scores=base_scores,
            stable_gesture='neutral',
            current_level=0.62,
            current_muted=False,
            now=2.8,
        )
        self.assertTrue(reset_update.active)
        self.assertAlmostEqual(reset_update.level or 0.0, 0.62, places=3)

        moved_after_hold = translate_landmarks(base, dy=-0.05)
        moved_features = extract_static_features(moved_after_hold)
        moved_update = tracker.update(
            features=moved_features,
            landmarks=moved_after_hold,
            candidate_scores=score_static_candidates(moved_features),
            stable_gesture='neutral',
            current_level=0.62,
            current_muted=False,
            now=2.9,
        )
        self.assertGreater(moved_update.level or 0.0, 0.62)

    def test_pinky_hold_does_not_activate_without_active_volume_pose(self) -> None:
        tracker = VolumeGestureTracker(confirm_frames=2, release_frames=2, smoothing=1.0, hold_seconds=1.5)
        pinky_hold = make_landmarks(
            {'index': 'open', 'middle': 'open', 'ring': 'closed', 'pinky': 'open'},
            thumb_state='closed',
            spread='together',
        )
        features = extract_static_features(pinky_hold)
        scores = score_static_candidates(features)
        first = tracker.update(
            features=features,
            landmarks=pinky_hold,
            candidate_scores=scores,
            stable_gesture='neutral',
            current_level=0.62,
            current_muted=False,
            now=1.0,
        )
        second = tracker.update(
            features=features,
            landmarks=pinky_hold,
            candidate_scores=scores,
            stable_gesture='neutral',
            current_level=0.62,
            current_muted=False,
            now=1.1,
        )
        self.assertFalse(first.active)
        self.assertFalse(second.active)
        self.assertNotEqual(second.status, 'holding')

    def test_volume_tracker_accepts_relaxed_mostly_curled_outer_fingers(self) -> None:
        tracker = VolumeGestureTracker(confirm_frames=2, release_frames=1, smoothing=1.0)
        pose = make_pose('volume_pose')
        features = SimpleNamespace(
            # Taken from the pose instead of hardcoded, because
            # `_is_volume_ready_pose` now measures tip 8 -> tip 12
            # directly and divides by `features.palm_scale`. That ties
            # the stub to the landmarks, which used to be independent:
            # a hand-written 0.10 against this pose's real 0.3303 turns
            # a 0.08 tip gap into a ratio of 0.800 and the entry gate is
            # 0.26, so the pose was rejected on geometry before any of
            # the curl states this case is actually about were consulted.
            # With the pose's own scale the ratio is 0.242 and passes.
            palm_scale=extract_static_features(pose).palm_scale,
            open_scores={
                'thumb': 0.38,
                'index': 0.80,
                'middle': 0.83,
                'ring': 0.44,
                'pinky': 0.40,
            },
            states={
                'thumb': 'closed',
                'index': 'open',
                'middle': 'open',
                'ring': 'closed',
                'pinky': 'closed',
            },
            fine_states={
                'thumb': 'mostly_curled',
                'index': 'fully_open',
                'middle': 'fully_open',
                'ring': 'mostly_curled',
                'pinky': 'mostly_curled',
            },
            finger_count_open=2,
            spread_states={'index_middle': 'together'},
            spread_together_strengths={'index_middle': 0.84},
            spread_apart_strengths={'index_middle': 0.08},
        )
        first = tracker.update(
            features=features,
            landmarks=pose,
            candidate_scores={'volume_pose': 0.0},
            stable_gesture='neutral',
            current_level=0.43,
            current_muted=False,
            now=1.0,
        )
        second = tracker.update(
            features=features,
            landmarks=pose,
            candidate_scores={'volume_pose': 0.0},
            stable_gesture='neutral',
            current_level=0.43,
            current_muted=False,
            now=1.1,
        )
        self.assertFalse(first.active)
        self.assertTrue(second.active)

    def test_volume_tracker_accepts_partially_curled_primary_fingers(self) -> None:
        tracker = VolumeGestureTracker(confirm_frames=2, release_frames=1, smoothing=1.0)
        pose = make_pose('volume_pose')
        features = SimpleNamespace(
            # Same reason as the case above: palm_scale has to come from
            # the pose these landmarks describe, or the tip-distance gate
            # rejects on geometry and this case never reaches the curl
            # states it exists to check.
            palm_scale=extract_static_features(pose).palm_scale,
            open_scores={
                'thumb': 0.34,
                'index': 0.57,
                'middle': 0.65,
                'ring': 0.48,
                'pinky': 0.52,
            },
            states={
                'thumb': 'closed',
                'index': 'open',
                'middle': 'open',
                'ring': 'closed',
                'pinky': 'closed',
            },
            fine_states={
                'thumb': 'mostly_curled',
                'index': 'partially_curled',
                'middle': 'partially_curled',
                'ring': 'mostly_curled',
                'pinky': 'mostly_curled',
            },
            finger_count_open=0,
            spread_states={'index_middle': 'neutral'},
            spread_ratios={'index_middle': 0.28},
            spread_together_strengths={'index_middle': 0.40},
            spread_apart_strengths={'index_middle': 0.17},
        )
        first = tracker.update(
            features=features,
            landmarks=pose,
            candidate_scores={'volume_pose': 0.0},
            stable_gesture='neutral',
            current_level=0.43,
            current_muted=False,
            now=1.0,
        )
        second = tracker.update(
            features=features,
            landmarks=pose,
            candidate_scores={'volume_pose': 0.0},
            stable_gesture='neutral',
            current_level=0.43,
            current_muted=False,
            now=1.1,
        )
        self.assertFalse(first.active)
        self.assertTrue(second.active)

# Author: Konstantin Markov

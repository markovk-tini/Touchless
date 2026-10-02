from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from hgr.debug.mouse_gesture import MouseGestureTracker
from hgr.gesture.recognition.engine import GestureRecognitionEngine

from .helpers import make_landmarks, make_pose


_OPEN_FINGERS = {"index": "open", "middle": "open", "ring": "open", "pinky": "open"}


def pinch_pose(primary: str = "index"):
    """Build the CURRENT click pose: thumb tip touching one fingertip.

    `5db7c5a` ("fix(mouse): pinch detection uses 3D distance -- clicks
    work with fingers straight up") replaced the curl-based click with a
    pinch. `_click_pose_active` now just delegates to `_pinch_active`,
    whose own comment spells out the user-facing gesture: thumb tip +
    index tip held together is left-click, thumb tip + middle tip is
    right-click, with the other three fingers relaxed so a fist cannot
    fire it.

    Six cases in this file still built a HOOKED finger -- the pre-5db7c5a
    pose -- and expected a click. They were asserting the old gesture
    against the new tracker. Measured on these synthetic hands, the
    thumb-to-tip distance ratio is 0.122 for a pinch and 2.6-2.9 for an
    open hand, against a 0.42 threshold, so the poses discriminate
    cleanly and each pinch arms only its own button.
    """
    if primary == "index":
        return make_landmarks(
            _OPEN_FINGERS, thumb_state="open", spread="apart",
            pinch_thumb_index=True,
        )
    tip_index = {"middle": 12}[primary]
    points = make_landmarks(_OPEN_FINGERS, thumb_state="open", spread="apart")
    midpoint = (points[tip_index] + points[4]) * 0.5
    points[4] = midpoint + np.array([-0.03, 0.0, 0.0], dtype=np.float32)
    points[tip_index] = midpoint + np.array([0.03, 0.0, 0.0], dtype=np.float32)
    return points


class MouseGestureTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = GestureRecognitionEngine(stable_frames_required=2)
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self.tracker = MouseGestureTracker(
            toggle_hold_seconds=0.30,
            toggle_cooldown_seconds=0.35,
            drag_hold_seconds=0.24,
            cursor_smoothing=0.35,
            scroll_hold_seconds=0.18,
            scroll_step_distance=0.06,
            scroll_deadzone=0.015,
            pose_grace_seconds=0.12,
            no_hand_grace_seconds=0.10,
        )

    def _update(
        self,
        landmarks,
        now: float,
        *,
        center: tuple[float, float] = (0.50, 0.50),
        handedness: str = "Right",
        cursor_seed: tuple[float, float] | None = None,
        desktop_bounds: tuple[int, int, int, int] | None = None,
        prediction_override=None,
        finger_overrides: dict[str, dict[str, float | str]] | None = None,
    ):
        if desktop_bounds is not None:
            self.tracker.set_desktop_bounds(desktop_bounds)
        result = self.engine.process_landmarks(
            landmarks,
            frame_bgr=self.frame,
            handedness=handedness,
            timestamp=now,
        )
        assert result.hand_reading is not None
        palm = replace(
            result.hand_reading.palm,
            center=np.array([center[0], center[1], 0.0], dtype=np.float32),
        )
        hand_reading = replace(result.hand_reading, palm=palm)
        if finger_overrides:
            fingers = dict(hand_reading.fingers)
            for name, overrides in finger_overrides.items():
                fingers[name] = replace(fingers[name], **overrides)
            hand_reading = replace(hand_reading, fingers=fingers)
        return self.tracker.update(
            hand_reading=hand_reading,
            prediction=prediction_override if prediction_override is not None else result.prediction,
            hand_handedness=handedness,
            cursor_seed=cursor_seed,
            now=now,
        )

    def _toggle_mouse_mode_on(self, start: float = 1.0):
        last_update = None
        for offset in (0.00, 0.12, 0.34):
            last_update = self._update(make_pose("three"), start + offset, handedness="Left")
        assert last_update is not None
        self.assertTrue(last_update.mode_enabled)
        return last_update

    def test_left_hand_three_hold_toggles_mouse_mode_on_and_off(self) -> None:
        enabled = self._toggle_mouse_mode_on()
        self.assertTrue(enabled.consume_other_routes)
        self.assertEqual(enabled.control_text, "mouse mode on")

        self._update(make_pose("open_hand"), 1.55)

        updates = [
            self._update(make_pose("three"), 2.00, handedness="Left"),
            self._update(make_pose("three"), 2.12, handedness="Left"),
            self._update(make_pose("three"), 2.36, handedness="Left"),
        ]
        self.assertFalse(updates[-1].mode_enabled)
        self.assertEqual(updates[-1].control_text, "mouse mode off")

    def test_right_hand_three_does_not_toggle_mouse_mode(self) -> None:
        updates = [
            self._update(make_pose("three"), 1.00, handedness="Right"),
            self._update(make_pose("three"), 1.12, handedness="Right"),
            self._update(make_pose("three"), 1.34, handedness="Right"),
        ]
        self.assertFalse(updates[-1].mode_enabled)
        self.assertFalse(updates[-1].consume_other_routes)

    def test_mouse_ready_pose_emits_smoothed_cursor_positions(self) -> None:
        self._toggle_mouse_mode_on()
        first = self._update(make_pose("open_hand"), 1.60, center=(0.30, 0.32))
        second = self._update(make_pose("open_hand"), 1.72, center=(0.75, 0.68))

        assert first.cursor_position is not None
        assert second.cursor_position is not None
        self.assertEqual(first.status, "ready")
        self.assertGreater(second.cursor_position[0], first.cursor_position[0])
        self.assertGreater(second.cursor_position[1], first.cursor_position[1])
        self.assertLess(second.cursor_position[0], 0.82)

    def test_multi_monitor_layout_produces_compact_camera_control_box(self) -> None:
        update = self._update(
            make_pose("open_hand"),
            1.00,
            desktop_bounds=(0, 0, 5120, 1440),
            cursor_seed=(0.50, 0.50),
        )

        bounds = self.tracker.debug_state.camera_control_bounds
        assert bounds is not None
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        self.assertIsNotNone(update)
        # Mouse-pad-style defaults: box stays a forearm-sized patch
        # of the camera frame even with a 32:9 virtual desktop.
        # Aspect-power compression still slightly tilts width/height
        # but not enough to push the box back to the old "fills most
        # of the frame" geometry.
        self.assertGreater(width, 0.30)
        self.assertLess(width, 0.62)
        self.assertGreater(height, 0.30)
        self.assertLess(height, 0.62)

    def test_small_hand_motion_stays_controlled_near_entry_anchor(self) -> None:
        self._toggle_mouse_mode_on()
        first = self._update(make_pose("open_hand"), 1.60, center=(0.50, 0.50), cursor_seed=(0.50, 0.50))
        second = self._update(make_pose("open_hand"), 1.72, center=(0.58, 0.50), cursor_seed=(0.50, 0.50))

        assert first.cursor_position is not None
        assert second.cursor_position is not None
        delta = second.cursor_position[0] - first.cursor_position[0]
        # The mouse-pad-style control box is intentionally smaller
        # than the previous "fills the frame" geometry, so the same
        # hand-displacement maps to a larger fraction of the screen
        # — that's the explicit design intent of the resize. Upper
        # bound widened from 0.10 to 0.20 to reflect the new ratio
        # while still asserting the smoothing damps the motion.
        self.assertGreater(delta, 0.01)
        self.assertLess(delta, 0.20)

    def test_entry_anchor_mapping_reaches_full_desktop_without_full_frame_sweep(self) -> None:
        self._toggle_mouse_mode_on()
        anchored = self._update(
            make_pose("open_hand"),
            1.60,
            center=(0.63, 0.58),
            cursor_seed=(0.46, 0.54),
            desktop_bounds=(0, 0, 5120, 1440),
        )

        assert anchored.cursor_position is not None
        # Starts FROM the seed, but is not pinned to it: `_update_cursor`
        # seeds `_cursor_position` only when it is None and then applies
        # the same frame's smoothing step toward the palm target. The
        # snap this used to assert was removed on purpose -- its comment
        # records the symptom, "cursor rubber-bands toward the OS
        # cursor's last position the moment a click is detected", caused
        # by warping the smoothed cursor onto the lagged OS cursor after
        # a one-frame pose dropout. What still matters here is that entry
        # begins near the seed rather than somewhere mid-screen, so allow
        # one smoothing step (measured: 0.043 away).
        self.assertAlmostEqual(anchored.cursor_position[0], 0.46, delta=0.06)
        self.assertAlmostEqual(anchored.cursor_position[1], 0.54, delta=0.06)
        bounds = self.tracker.debug_state.camera_control_bounds
        assert bounds is not None
        left_top_center = (bounds[0] + 0.02, bounds[1] + 0.02)
        right_bottom_center = (bounds[2] - 0.02, bounds[3] - 0.02)

        left_top = anchored
        for index in range(6):
            left_top = self._update(
                make_pose("open_hand"),
                1.72 + index * 0.10,
                center=left_top_center,
                cursor_seed=(0.46, 0.54),
                desktop_bounds=(0, 0, 5120, 1440),
            )

        assert left_top.cursor_position is not None
        self.assertLess(left_top.cursor_position[0], 0.06)
        self.assertLess(left_top.cursor_position[1], 0.06)

        right_bottom = left_top
        for index in range(8):
            right_bottom = self._update(
                make_pose("open_hand"),
                2.40 + index * 0.10,
                center=right_bottom_center,
                cursor_seed=(0.46, 0.54),
                desktop_bounds=(0, 0, 5120, 1440),
            )

        assert right_bottom.cursor_position is not None
        self.assertGreater(right_bottom.cursor_position[0], 0.94)
        self.assertGreater(right_bottom.cursor_position[1], 0.94)

    def test_index_pinch_press_then_release_delivers_a_left_click(self) -> None:
        """A tap-length pinch emits press then release -- and must NOT
        also set `left_click`.

        Renamed from `..._index_open_curl_open_triggers_left_click`,
        which asserted `released.left_click` and the text "mouse left
        click". `_update_left_sequence` deliberately stopped emitting
        `left_click`: the controller's own `left_click()` helper fires a
        synthetic down+up, and stacking that on top of the press/release
        pair sent down-up-DOWN-UP per tap, which Windows read as a
        double-click or dropped half of -- the user-reported "doesn't
        actually click even though the app detected clicking". The
        press/release pair is what Windows turns into one click, so
        `left_click` staying off IS the fix, and pinning that is now the
        point of this case.
        """
        self._toggle_mouse_mode_on()
        self._update(make_pose("open_hand"), 1.60)
        self._update(make_pose("open_hand"), 1.72)
        index_pinch = pinch_pose("index")
        pressed = self._update(index_pinch, 1.84)
        self._update(index_pinch, 1.92)
        released = self._update(make_pose("open_hand"), 2.00)

        self.assertTrue(pressed.left_press)
        self.assertTrue(released.left_release)
        self.assertFalse(released.dragging)
        self.assertEqual(released.control_text, "mouse drag release")
        self.assertFalse(pressed.left_click,
                         "left_click must stay off -- see docstring")
        self.assertFalse(released.left_click,
                         "left_click must stay off -- see docstring")

    def test_left_click_allows_softer_open_support_fingers(self) -> None:
        self._toggle_mouse_mode_on()
        soft_support = {
            "middle": {"state": "partially_curled", "openness": 0.68, "curl": 0.32, "bend_proximal": 146.0, "bend_distal": 144.0},
            "ring": {"state": "partially_curled", "openness": 0.64, "curl": 0.36, "bend_proximal": 142.0, "bend_distal": 140.0},
            "pinky": {"state": "partially_curled", "openness": 0.62, "curl": 0.38, "bend_proximal": 140.0, "bend_distal": 138.0},
        }
        self._update(make_pose("open_hand"), 1.60, finger_overrides=soft_support)
        self._update(make_pose("open_hand"), 1.72, finger_overrides=soft_support)
        index_pinch = pinch_pose("index")
        self._update(index_pinch, 1.84, finger_overrides=soft_support)
        self._update(index_pinch, 1.92, finger_overrides=soft_support)
        released = self._update(make_pose("open_hand"), 2.00, finger_overrides=soft_support)

        # Same press/release contract as the case above; what this one
        # adds is that softly-curled support fingers do not block the
        # pinch (`_pinch_others_relaxed` accepts partial curl).
        self.assertTrue(released.left_release)
        self.assertFalse(released.dragging)

    def test_index_click_pose_keeps_mouse_ready_instead_of_waiting_pose(self) -> None:
        self._toggle_mouse_mode_on()
        self._update(make_pose("open_hand"), 1.60, center=(0.52, 0.50))
        self._update(make_pose("open_hand"), 1.72, center=(0.52, 0.50))
        index_pinch = pinch_pose("index")
        clicking = self._update(
            index_pinch,
            1.84,
            center=(0.54, 0.50),
            finger_overrides={
                "index": {"state": "mostly_curled", "openness": 0.14, "curl": 0.94, "bend_proximal": 150.0, "bend_distal": 106.0},
                "middle": {"state": "fully_open", "openness": 0.85, "curl": 0.15},
                "ring": {"state": "fully_open", "openness": 0.85, "curl": 0.15},
                "pinky": {"state": "fully_open", "openness": 0.68, "curl": 0.32},
                "thumb": {"state": "fully_open", "openness": 0.80, "curl": 0.20, "palm_distance": 0.78},
            },
        )

        # The point of this case is that holding the click pose must not
        # drop the cursor into "waiting_pose" (which abandons cursor
        # tracking mid-click). Under the pinch design the clicking state
        # is "drag" and the cursor keeps moving, so assert that rather
        # than the old "mouse ready" text.
        self.assertNotEqual(clicking.status, "waiting_pose")
        self.assertEqual(clicking.status, "drag")
        self.assertEqual(clicking.control_text, "mouse drag start")
        self.assertIsNotNone(clicking.cursor_position)

    def test_index_hold_starts_drag_and_open_releases_it(self) -> None:
        self._toggle_mouse_mode_on()
        self._update(make_pose("open_hand"), 1.60)
        self._update(make_pose("open_hand"), 1.72)
        index_pinch = pinch_pose("index")
        pressed = self._update(index_pinch, 1.84)
        self._update(index_pinch, 1.92)
        dragging = self._update(index_pinch, 2.22)
        released = self._update(make_pose("open_hand"), 2.32)

        # Press lands on the FIRST pinch frame, by design: "users expect
        # tips touch -> button down with no buffering delay". It used to
        # be asserted on a later frame because the old curl pose had to
        # clear `drag_hold_seconds` first.
        self.assertTrue(pressed.left_press)
        self.assertTrue(pressed.dragging)
        self.assertFalse(dragging.left_press,
                         "press must not be re-emitted while the pinch is held")
        self.assertTrue(dragging.dragging)
        self.assertTrue(released.left_release)
        self.assertFalse(released.dragging)

    def test_middle_open_curl_open_triggers_right_click(self) -> None:
        self._toggle_mouse_mode_on()
        self._update(make_pose("open_hand"), 1.60)
        self._update(make_pose("open_hand"), 1.72)
        middle_pinch = pinch_pose("middle")
        clicked = self._update(middle_pinch, 1.84)
        held = self._update(middle_pinch, 1.92)
        self._update(make_pose("open_hand"), 2.04)

        # Right-click fires on the leading edge of the pinch and is
        # one-shot: `_update_right_sequence` says the OS has no
        # meaningful "right-click drag", so it emits once and then holds
        # off until release. The old assertion looked on the RELEASE
        # frame, which is one frame too late under this design.
        self.assertTrue(clicked.right_click)
        self.assertEqual(clicked.control_text, "mouse right click")
        self.assertFalse(held.right_click,
                         "right-click must not repeat while the pinch is held")

    def test_wheel_pose_emits_scroll_steps_after_hold(self) -> None:
        self._toggle_mouse_mode_on()
        wheel_pose = make_pose("wheel_pose")
        self._update(wheel_pose, 1.60, center=(0.50, 0.50))
        self._update(wheel_pose, 1.72, center=(0.50, 0.50))
        active = self._update(wheel_pose, 1.90, center=(0.50, 0.50))
        moved = self._update(wheel_pose, 2.02, center=(0.50, 0.34))

        self.assertTrue(active.scrolling)
        self.assertGreater(moved.scroll_steps, 0)
        self.assertEqual(moved.status, "scroll")

    def test_wheel_pose_can_scroll_from_hand_shape_when_prediction_is_neutral(self) -> None:
        self._toggle_mouse_mode_on()
        wheel_pose = make_pose("wheel_pose")
        neutral_prediction = replace(
            self.engine.process_landmarks(
                wheel_pose,
                frame_bgr=self.frame,
                handedness="Right",
                timestamp=1.50,
            ).prediction,
            raw_label="neutral",
            stable_label="neutral",
            confidence=0.0,
        )
        self._update(wheel_pose, 1.60, center=(0.50, 0.50), prediction_override=neutral_prediction)
        self._update(wheel_pose, 1.72, center=(0.50, 0.50), prediction_override=neutral_prediction)
        active = self._update(wheel_pose, 1.90, center=(0.50, 0.50), prediction_override=neutral_prediction)
        moved = self._update(wheel_pose, 2.02, center=(0.50, 0.34), prediction_override=neutral_prediction)

        self.assertTrue(active.scrolling)
        self.assertGreater(moved.scroll_steps, 0)

    def test_drag_releases_after_no_hand_grace(self) -> None:
        self._toggle_mouse_mode_on()
        self._update(make_pose("open_hand"), 1.60)
        self._update(make_pose("open_hand"), 1.72)
        index_pinch = pinch_pose("index")
        self._update(index_pinch, 1.84)
        self._update(index_pinch, 1.92)
        self._update(index_pinch, 2.22)

        lost = self.tracker.update(hand_reading=None, prediction=None, now=2.36)
        self.assertTrue(lost.left_release)
        self.assertFalse(lost.dragging)

# Author: Konstantin Markov

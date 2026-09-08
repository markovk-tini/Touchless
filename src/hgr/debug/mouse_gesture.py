from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..gesture.analysis.geometry import clamp01


@dataclass(frozen=True)
class MouseGestureUpdate:
    mode_enabled: bool
    consume_other_routes: bool
    cursor_position: tuple[float, float] | None
    control_text: str
    status: str
    dragging: bool
    scrolling: bool
    left_press: bool = False
    left_release: bool = False
    left_click: bool = False
    right_click: bool = False
    scroll_steps: int = 0


@dataclass(frozen=True)
class MouseDebugState:
    mode_enabled: bool
    status: str
    cursor_position: tuple[float, float] | None
    cursor_anchor_position: tuple[float, float] | None
    cursor_reach_bounds: tuple[float, float, float, float] | None
    camera_control_bounds: tuple[float, float, float, float] | None
    camera_anchor_position: tuple[float, float] | None
    dragging: bool
    scrolling: bool


@dataclass
class _FingerSequenceState:
    open_frames: int = 0
    curl_frames: int = 0
    press_started_at: float | None = None


class MouseGestureTracker:
    def __init__(
        self,
        *,
        toggle_hold_seconds: float = 1.0,
        toggle_cooldown_seconds: float = 0.90,
        open_confirm_frames: int = 2,
        curl_confirm_frames: int = 2,
        drag_hold_seconds: float = 0.42,
        cursor_smoothing: float = 0.24,
        cursor_deadzone: float = 0.018,
        horizontal_margin: float = 0.10,
        vertical_margin: float = 0.12,
        cursor_reference_width: float = 0.40,
        cursor_reference_height: float = 0.34,
        control_box_center_x: float = 0.67,
        control_box_center_y: float = 0.55,
        control_box_area: float = 0.18,
        control_box_aspect_power: float = 0.40,
        # Mouse-pad-style box: small forearm-sized patch in the
        # mirrored frame that maps to the full monitor. Min widths
        # and heights here used to be ~0.58/0.42 (i.e. always at
        # least half the frame), which clamped the smaller box back
        # up to the old "fills most of the camera" footprint.
        control_box_min_width: float = 0.30,
        control_box_max_width: float = 0.62,
        control_box_min_height: float = 0.32,
        control_box_max_height: float = 0.62,
        scroll_confirm_frames: int = 2,
        # Tightened scroll feel: shorter hold to enter scroll mode,
        # smaller step distance so each unit of hand movement scrolls
        # more, and a higher per-update step cap so big sweeps land
        # in fewer frames. Previous values (0.28 hold, 0.065 step,
        # cap 5) made scrolling feel sluggish — the user reported
        # "make it scroll more per movement".
        scroll_hold_seconds: float = 0.20,
        scroll_step_distance: float = 0.034,
        scroll_deadzone: float = 0.014,
        scroll_tip_blend: float = 0.42,
        scroll_idle_decay: float = 0.72,
        scroll_reverse_deadband: float = 1.10,
        scroll_max_steps_per_update: int = 12,
        pose_grace_seconds: float = 0.22,
        no_hand_grace_seconds: float = 0.18,
    ) -> None:
        self.toggle_hold_seconds = float(toggle_hold_seconds)
        self.toggle_cooldown_seconds = float(toggle_cooldown_seconds)
        self.open_confirm_frames = int(open_confirm_frames)
        self.curl_confirm_frames = int(curl_confirm_frames)
        self.drag_hold_seconds = float(drag_hold_seconds)
        self.cursor_smoothing = float(cursor_smoothing)
        self.cursor_deadzone = float(cursor_deadzone)
        self.horizontal_margin = float(horizontal_margin)
        self.vertical_margin = float(vertical_margin)
        self.cursor_reference_width = max(0.08, float(cursor_reference_width))
        self.cursor_reference_height = max(0.08, float(cursor_reference_height))
        self.control_box_center_x = float(control_box_center_x)
        self.control_box_center_y = float(control_box_center_y)
        self.control_box_area = float(control_box_area)
        self.control_box_aspect_power = float(control_box_aspect_power)
        self.control_box_min_width = float(control_box_min_width)
        self.control_box_max_width = float(control_box_max_width)
        self.control_box_min_height = float(control_box_min_height)
        self.control_box_max_height = float(control_box_max_height)
        self.scroll_confirm_frames = int(scroll_confirm_frames)
        self.scroll_hold_seconds = float(scroll_hold_seconds)
        self.scroll_step_distance = float(scroll_step_distance)
        self.scroll_deadzone = float(scroll_deadzone)
        self.scroll_tip_blend = float(scroll_tip_blend)
        self.scroll_idle_decay = float(scroll_idle_decay)
        self.scroll_reverse_deadband = float(scroll_reverse_deadband)
        self.scroll_max_steps_per_update = int(scroll_max_steps_per_update)
        self.pose_grace_seconds = float(pose_grace_seconds)
        self.no_hand_grace_seconds = float(no_hand_grace_seconds)
        self._desktop_aspect_ratio = 16.0 / 9.0
        # Always use aspect-compression now. Previously this was True
        # for single-monitor setups so the camera box visually matched
        # the monitor's 16:9 — but that produced a wide, stretched box
        # that didn't match the natural ergonomic shape of a hand-reach
        # area. The compressed formula (aspect ** control_box_aspect_power
        # with a small power) gives a more upright/square box that feels
        # better to control while still leaning slightly wider for
        # wider desktops.
        self._use_raw_aspect = False
        self.reset()

    @property
    def mode_enabled(self) -> bool:
        return self._mode_enabled

    def force_enable_mode(self, now: float) -> None:
        if self._mode_enabled:
            return
        self._mode_enabled = True
        self._toggle_candidate = "neutral"
        self._toggle_candidate_since = 0.0
        self._toggle_latched = True
        self._toggle_cooldown_until = float(now) + self.toggle_cooldown_seconds
        self._clear_interaction_state(reset_cursor=True)
        self._control_text = "mouse mode on"
        self._status = "active"

    @property
    def debug_state(self) -> MouseDebugState:
        camera_bounds = self._camera_bounds()
        reach_bounds = None
        if self._cursor_reference_active and self._cursor_anchor_screen is not None:
            reach_bounds = (0.0, 0.0, 1.0, 1.0)
        return MouseDebugState(
            mode_enabled=bool(self._mode_enabled),
            status=self._status,
            cursor_position=None if self._cursor_position is None else tuple(self._cursor_position),
            cursor_anchor_position=None if self._cursor_anchor_screen is None else tuple(self._cursor_anchor_screen),
            cursor_reach_bounds=reach_bounds,
            camera_control_bounds=camera_bounds,
            camera_anchor_position=None if self._cursor_anchor_hand is None else tuple(self._cursor_anchor_hand),
            dragging=bool(self._dragging),
            scrolling=bool(self._scrolling),
        )

    def set_desktop_bounds(self, bounds: tuple[int, int, int, int] | None) -> None:
        if bounds is None:
            return
        _left, _top, width, height = bounds
        width = max(1.0, float(width))
        height = max(1.0, float(height))
        self._desktop_aspect_ratio = max(0.80, min(4.50, width / height))
        # Always use aspect_power compression — produces an upright,
        # ergonomic hand-reach box for single-monitor and a still-
        # widened but not comically flat box for multi-monitor. The
        # box aspect tracks desktop aspect but more gently than raw.
        self._use_raw_aspect = False

    def reset(self) -> None:
        self._mode_enabled = False
        self._toggle_candidate = "neutral"
        self._toggle_candidate_since = 0.0
        self._toggle_cooldown_until = 0.0
        self._toggle_latched = False
        self._cursor_position: tuple[float, float] | None = None
        self._cursor_anchor_hand: tuple[float, float] | None = None
        self._cursor_anchor_screen: tuple[float, float] | None = None
        self._cursor_reference_active = False
        # Grace window after motion_pose_ready last returned True.
        # MediaPipe's per-frame finger-state classification flickers
        # when the hand is held at certain tilts -- a single misclassified
        # frame would otherwise drop the cursor, then re-anchor on the
        # next True frame, producing the "jitters unless tilted slightly
        # right" symptom. Keeping the cursor live for pose_grace_seconds
        # (0.22 s, ~6-7 frames at 30 fps) absorbs single-frame flickers.
        self._motion_ready_grace_until = 0.0
        self._index_state = _FingerSequenceState()
        self._middle_state = _FingerSequenceState()
        self._dragging = False
        self._scrolling = False
        self._scroll_candidate_frames = 0
        self._scroll_candidate_since = 0.0
        self._scroll_pose_grace_until = 0.0
        self._scroll_last_y: float | None = None
        # Anchor-velocity scroll state: the Y at which the user
        # entered scroll mode acts as a "neutral" point. Distance
        # above the anchor produces an upward scroll RATE; distance
        # below produces a downward rate. Bigger offset = faster.
        # _scroll_anchor_y is the captured pose-start position;
        # _scroll_last_emit_time tracks the last update for dt-based
        # accumulation. _scroll_residual carries fractional steps
        # over between frames.
        self._scroll_anchor_y: float | None = None
        self._scroll_last_emit_time: float | None = None
        self._scroll_residual = 0.0
        self._scroll_velocity_ema = 0.0
        self._scroll_last_direction = 0
        self._last_seen_time = 0.0
        self._control_text = "mouse mode off"
        self._status = "off"

    def update(
        self,
        *,
        hand_reading,
        prediction,
        hand_handedness: str | None = None,
        cursor_seed: tuple[float, float] | None = None,
        now: float,
    ) -> MouseGestureUpdate:
        toggle_active = self._toggle_pose_active(prediction, hand_handedness)
        left_release, toggled = self._update_toggle(toggle_active, now)
        consume = self._mode_enabled or toggle_active or self._toggle_candidate == "left_three"

        if toggled:
            return self._snapshot(
                consume_other_routes=True,
                cursor_position=None,
                left_release=left_release,
            )

        if not self._mode_enabled:
            return self._snapshot(
                consume_other_routes=consume,
                cursor_position=None,
                left_release=left_release,
            )

        if toggle_active:
            self._clear_cursor_reference(reset_position=False)
            self._control_text = "hold left hand three to turn mouse mode off"
            self._status = "toggle"
            return self._snapshot(
                consume_other_routes=True,
                cursor_position=None,
                left_release=left_release,
            )

        if hand_reading is None:
            if self._dragging and (now - self._last_seen_time) >= self.no_hand_grace_seconds:
                self._dragging = False
                left_release = True
            if self._scrolling and (now - self._last_seen_time) >= self.no_hand_grace_seconds:
                self._stop_scrolling()
            self._clear_cursor_reference(reset_position=False)
            self._reset_sequence(self._index_state)
            self._reset_sequence(self._middle_state)
            self._control_text = "mouse waiting for hand"
            self._status = "waiting"
            return self._snapshot(
                consume_other_routes=True,
                cursor_position=None,
                left_release=left_release,
            )

        self._last_seen_time = float(now)
        scroll_steps = 0
        left_press = False
        left_click = False
        right_click = False
        cursor_position: tuple[float, float] | None = None

        scroll_pose_active = self._scroll_pose_active(prediction, hand_reading)
        if self._scrolling:
            if scroll_pose_active:
                self._scroll_pose_grace_until = now + self.pose_grace_seconds
                scroll_steps = self._update_scroll(hand_reading, now)
            elif now >= self._scroll_pose_grace_until:
                self._stop_scrolling()
            if self._scrolling:
                self._clear_cursor_reference(reset_position=False)
                self._control_text = "mouse scroll active"
                self._status = "scroll"
                if scroll_steps > 0:
                    self._control_text = f"mouse scroll up x{scroll_steps}"
                elif scroll_steps < 0:
                    self._control_text = f"mouse scroll down x{abs(scroll_steps)}"
                return self._snapshot(
                    consume_other_routes=True,
                    cursor_position=None,
                    left_release=left_release,
                    scroll_steps=scroll_steps,
                )
        elif not self._dragging and scroll_pose_active:
            if self._scroll_candidate_frames == 0:
                self._scroll_candidate_since = now
            self._scroll_candidate_frames += 1
            if (
                self._scroll_candidate_frames >= self.scroll_confirm_frames
                and (now - self._scroll_candidate_since) >= self.scroll_hold_seconds
            ):
                self._scrolling = True
                # Capture the Y at pose-confirmation as the neutral
                # anchor. From here, hand-above = scroll up, hand-
                # below = scroll down, with rate proportional to
                # distance from the anchor (anchor-velocity model).
                anchor_y = self._scroll_control_y(hand_reading)
                self._scroll_anchor_y = anchor_y
                self._scroll_last_y = anchor_y
                self._scroll_last_emit_time = now
                self._scroll_residual = 0.0
                self._scroll_velocity_ema = 0.0
                self._scroll_last_direction = 0
                self._scroll_pose_grace_until = now + self.pose_grace_seconds
                self._control_text = "mouse scroll active"
                self._status = "scroll"
                self._clear_cursor_reference(reset_position=False)
                return self._snapshot(
                    consume_other_routes=True,
                    cursor_position=None,
                    left_release=left_release,
                )
            self._clear_cursor_reference(reset_position=False)
            self._control_text = "hold wheel pose to scroll"
            self._status = "scroll_hold"
            return self._snapshot(
                consume_other_routes=True,
                cursor_position=None,
                left_release=left_release,
            )
        else:
            self._scroll_candidate_frames = 0
            self._scroll_candidate_since = 0.0

        allow_index_curled = (
            self._dragging
            or self._sequence_active(self._index_state)
            or self._click_pose_active(hand_reading, primary="index")
        )
        allow_middle_curled = (
            self._sequence_active(self._middle_state)
            or self._click_pose_active(hand_reading, primary="middle")
        )
        motion_ready = self._motion_pose_ready(
            hand_reading,
            allow_index_curled=allow_index_curled,
            allow_middle_curled=allow_middle_curled,
        )

        left_press, left_release_event, left_click = self._update_left_sequence(hand_reading, now)
        right_click = self._update_right_sequence(hand_reading, now)
        left_release = left_release or left_release_event

        # Pose grace: MediaPipe's per-frame finger-state classifier
        # flickers when the hand is held at non-ideal tilts (the
        # "smooth only when tilted slightly right" bug). Without
        # grace, a single failed motion_ready frame drops the
        # cursor, then re-anchors on the next True frame, producing
        # the jitter the user reported. Refresh the grace deadline
        # whenever motion_ready is True; treat the cursor as live
        # for pose_grace_seconds afterwards.
        if motion_ready:
            self._motion_ready_grace_until = now + self.pose_grace_seconds
        cursor_alive = motion_ready or (now < self._motion_ready_grace_until)

        if self._dragging or cursor_alive:
            cursor_position = self._update_cursor(
                hand_reading,
                cursor_seed=cursor_seed,
                reanchor=not self._cursor_reference_active,
            )
            self._cursor_reference_active = True
        else:
            self._clear_cursor_reference(reset_position=False)

        if left_press:
            self._control_text = "mouse drag start"
            self._status = "drag"
        elif left_release:
            self._control_text = "mouse drag release"
            self._status = "ready" if self._mode_enabled else "off"
        elif left_click:
            self._control_text = "mouse left click"
            self._status = "ready"
        elif right_click:
            self._control_text = "mouse right click"
            self._status = "ready"
        elif self._dragging:
            self._control_text = "mouse drag active"
            self._status = "drag"
        elif cursor_position is not None:
            self._control_text = "mouse ready"
            self._status = "ready"
        else:
            self._control_text = "show open hand mouse pose"
            self._status = "waiting_pose"

        return self._snapshot(
            consume_other_routes=True,
            cursor_position=cursor_position,
            left_press=left_press,
            left_release=left_release,
            left_click=left_click,
            right_click=right_click,
            scroll_steps=scroll_steps,
        )

    def _snapshot(
        self,
        *,
        consume_other_routes: bool,
        cursor_position: tuple[float, float] | None,
        left_press: bool = False,
        left_release: bool = False,
        left_click: bool = False,
        right_click: bool = False,
        scroll_steps: int = 0,
    ) -> MouseGestureUpdate:
        return MouseGestureUpdate(
            mode_enabled=self._mode_enabled,
            consume_other_routes=bool(consume_other_routes),
            cursor_position=cursor_position,
            control_text=self._control_text,
            status=self._status,
            dragging=self._dragging,
            scrolling=self._scrolling,
            left_press=bool(left_press),
            left_release=bool(left_release),
            left_click=bool(left_click),
            right_click=bool(right_click),
            scroll_steps=int(scroll_steps),
        )

    def _update_toggle(self, toggle_active: bool, now: float) -> tuple[bool, bool]:
        if not toggle_active:
            self._toggle_candidate = "neutral"
            self._toggle_candidate_since = 0.0
            self._toggle_latched = False
            return False, False

        if self._toggle_latched or now < self._toggle_cooldown_until:
            return False, False

        if self._toggle_candidate != "left_three":
            self._toggle_candidate = "left_three"
            self._toggle_candidate_since = now
            if self._mode_enabled:
                self._control_text = "hold left hand three to turn mouse mode off"
            else:
                self._control_text = "hold left hand three to turn mouse mode on"
            self._status = "toggle"
            return False, False

        if (now - self._toggle_candidate_since) < self.toggle_hold_seconds:
            if self._mode_enabled:
                self._control_text = "hold left hand three to turn mouse mode off"
            else:
                self._control_text = "hold left hand three to turn mouse mode on"
            self._status = "toggle"
            return False, False

        self._toggle_candidate = "neutral"
        self._toggle_candidate_since = 0.0
        self._toggle_latched = True
        self._toggle_cooldown_until = now + self.toggle_cooldown_seconds
        self._mode_enabled = not self._mode_enabled
        released_drag = self._clear_interaction_state(reset_cursor=True)
        if self._mode_enabled:
            self._control_text = "mouse mode on"
            self._status = "active"
        else:
            self._control_text = "mouse mode off"
            self._status = "off"
        return released_drag, True

    def _clear_interaction_state(self, *, reset_cursor: bool) -> bool:
        released_drag = self._dragging
        self._dragging = False
        self._stop_scrolling()
        self._reset_sequence(self._index_state)
        self._reset_sequence(self._middle_state)
        self._clear_cursor_reference(reset_position=reset_cursor)
        return released_drag

    def _clear_cursor_reference(self, *, reset_position: bool) -> None:
        self._cursor_anchor_hand = None
        self._cursor_anchor_screen = None
        self._cursor_reference_active = False
        if reset_position:
            self._cursor_position = None

    def _stop_scrolling(self) -> None:
        self._scrolling = False
        self._scroll_candidate_frames = 0
        self._scroll_candidate_since = 0.0
        self._scroll_pose_grace_until = 0.0
        self._scroll_last_y = None
        self._scroll_anchor_y = None
        self._scroll_last_emit_time = None
        self._scroll_residual = 0.0
        self._scroll_velocity_ema = 0.0
        self._scroll_last_direction = 0

    # Anchor-velocity scroll tunables. Offset is "anchor_y -
    # current_y", measured in normalized frame coords (0..1). The
    # camera frame Y axis runs top-to-bottom, so a hand moving UP
    # in the camera view produces a POSITIVE offset → scroll up.
    #
    # Curve choices:
    #   deadzone = 0.025   small jitter near the anchor doesn't scroll
    #   full_offset = 0.18 ~18% of frame from anchor = max rate
    #   rate range 4..40 steps/sec gives a clearly-paced slow start
    #     (just over deadzone) up to a fast page-flip rate at full
    #     offset. Page lines per Windows wheel notch are 3 by default
    #     so 40 steps/sec ≈ 120 lines/sec at the extreme — fast but
    #     not jarring.
    #   curve = 1.3 is slightly super-linear: progress 50% → rate
    #     ~17, progress 100% → rate 40. Feels like "the more I lean,
    #     the faster" without going too aggressive too quickly.
    _SCROLL_ANCHOR_DEADZONE = 0.025
    _SCROLL_ANCHOR_FULL_OFFSET = 0.18
    _SCROLL_RATE_MIN = 4.0
    _SCROLL_RATE_MAX = 40.0
    _SCROLL_RATE_CURVE = 1.3

    def _update_scroll(self, hand_reading, now: float) -> int:
        """Anchor-velocity scroll: the Y captured at scroll-pose
        confirmation acts as a neutral position. Hand above anchor →
        scroll up at a rate proportional to the offset; hand below →
        scroll down. Bigger offset → faster, with a slow-start curve
        so small movements scroll gently and big movements ramp up.

        Replaces the previous delta-based model (which scrolled by
        the per-frame change in hand Y) — that model required the
        user to keep moving their hand to keep scrolling. The new
        rate-based model is what the user asked for: hold the hand
        offset above/below the anchor and the page scrolls
        continuously, faster when held further away.
        """
        current_y = self._scroll_control_y(hand_reading)
        if self._scroll_anchor_y is None or self._scroll_last_emit_time is None:
            # Mode just started this frame and the start path
            # already initialized the anchor for the next call —
            # nothing to do this tick.
            self._scroll_anchor_y = current_y
            self._scroll_last_emit_time = now
            return 0

        # Time delta since the last scroll-rate evaluation. Clamp to
        # avoid pathological catch-up if a frame stalls (e.g. window
        # was hidden), which would otherwise emit a huge step burst.
        dt = max(0.0, float(now) - float(self._scroll_last_emit_time))
        dt = min(dt, 0.10)
        self._scroll_last_emit_time = now

        # Camera Y increases downward, so anchor_y - current_y is
        # POSITIVE when the hand is ABOVE the anchor (= scroll up).
        offset = float(self._scroll_anchor_y) - current_y
        magnitude = abs(offset)

        # Deadzone: small jitter near the anchor doesn't scroll.
        if magnitude <= self._SCROLL_ANCHOR_DEADZONE:
            # Bleed off any leftover fractional residual so the next
            # move-out doesn't fire a stale step from earlier hold.
            if abs(self._scroll_residual) < 0.2:
                self._scroll_residual = 0.0
            else:
                self._scroll_residual *= 0.5
            self._scroll_last_direction = 0
            return 0

        # Map (deadzone..full_offset) → (0..1) progress, then a
        # gentle exponent gives the slow-start ramp the user asked
        # for ("slowly starts scrolling … the more they move … the
        # faster"). Beyond full_offset, progress saturates at 1.
        usable = max(1e-6, self._SCROLL_ANCHOR_FULL_OFFSET - self._SCROLL_ANCHOR_DEADZONE)
        progress = clamp01((magnitude - self._SCROLL_ANCHOR_DEADZONE) / usable)
        ramp = progress ** self._SCROLL_RATE_CURVE
        rate = self._SCROLL_RATE_MIN + (self._SCROLL_RATE_MAX - self._SCROLL_RATE_MIN) * ramp
        signed_rate = rate if offset > 0 else -rate

        # Reset residual on a clean direction reversal so the new
        # direction starts from "no built-up debt" rather than
        # immediately emitting an extra step.
        direction = 1 if signed_rate > 0 else -1
        if self._scroll_last_direction and direction != self._scroll_last_direction:
            self._scroll_residual = 0.0
        self._scroll_last_direction = direction

        self._scroll_residual += signed_rate * dt
        steps = int(math.trunc(self._scroll_residual))
        if steps == 0:
            return 0
        steps = max(-self.scroll_max_steps_per_update, min(self.scroll_max_steps_per_update, steps))
        self._scroll_residual -= steps
        return steps

    def _scroll_control_y(self, hand_reading) -> float:
        palm_y = float(hand_reading.palm.center[1])
        tip_center_y = float((hand_reading.landmarks[8][1] + hand_reading.landmarks[20][1]) * 0.5)
        return (1.0 - self.scroll_tip_blend) * palm_y + self.scroll_tip_blend * tip_center_y

    def _absolute_cursor_target(self, hand_reading) -> tuple[float, float]:
        palm_center = hand_reading.palm.center
        min_x, min_y, max_x, max_y = self._camera_bounds()
        target_x = clamp01((float(palm_center[0]) - min_x) / max(1e-6, max_x - min_x))
        target_y = clamp01((float(palm_center[1]) - min_y) / max(1e-6, max_y - min_y))
        return target_x, target_y

    def _camera_bounds(self) -> tuple[float, float, float, float]:
        available_left = self.horizontal_margin
        available_top = self.vertical_margin
        available_right = 1.0 - self.horizontal_margin
        available_bottom = 1.0 - self.vertical_margin
        available_width = max(0.24, available_right - available_left)
        available_height = max(0.24, available_bottom - available_top)

        # The previous version computed the box's normalized
        # width/height from the desktop aspect directly, but the
        # camera frame itself is 16:9 — so a normalized box aspect
        # of 1.78 ends up displaying as a 1.78 * (16/9) = 3.16
        # visual aspect (way wider than the monitor). The user
        # reported this as "too much mouse control area to the
        # left and right of the monitor". Fix: divide by the
        # camera-frame aspect so visual aspect ≈ monitor aspect.
        # Assumes 16:9 frame (the common webcam case); ultrawide
        # frame cameras would still get a slightly off match,
        # but the result is far closer than treating the frame as
        # square.
        FRAME_ASPECT = 16.0 / 9.0
        if getattr(self, "_use_raw_aspect", True):
            target_visual_aspect = max(0.90, min(2.40, self._desktop_aspect_ratio))
        else:
            target_visual_aspect = max(0.90, min(2.40, self._desktop_aspect_ratio ** self.control_box_aspect_power))
        # Convert visual-aspect to normalized-aspect (divide by frame
        # aspect). For a 16:9 monitor on a 16:9 frame this yields 1.0
        # → square in normalized coords → 16:9 visually, matching the
        # monitor's aspect exactly.
        box_aspect = target_visual_aspect / FRAME_ASPECT
        # Auto-scale the effective area by desktop aspect. Single-
        # monitor 16:9 setups use control_box_area as-is (default 0.12,
        # tuned so the red box matches the displayed green Monitor 1
        # outline). Multi-monitor (wider) desktops scale up so the
        # user gets a proportionally wider control area without needing
        # to manually tune the sensitivity slider when they add a
        # monitor. The clamp prevents super-wide setups from blowing
        # past the camera frame.
        _SINGLE_MONITOR_VISUAL_ASPECT = 16.0 / 9.0
        aspect_scale = max(
            1.0,
            min(2.6, target_visual_aspect / _SINGLE_MONITOR_VISUAL_ASPECT),
        )
        target_area = max(0.06, min(0.50, self.control_box_area * aspect_scale))
        width = math.sqrt(target_area * box_aspect)
        height = math.sqrt(target_area / box_aspect)
        width = min(max(width, min(self.control_box_min_width, available_width)), min(self.control_box_max_width, available_width))
        height = min(max(height, min(self.control_box_min_height, available_height)), min(self.control_box_max_height, available_height))

        center_x = min(max(self.control_box_center_x, available_left + width * 0.5), available_right - width * 0.5)
        center_y = min(max(self.control_box_center_y, available_top + height * 0.5), available_bottom - height * 0.5)
        return (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        )

    def _edge_relative_axis_target(
        self,
        *,
        value: float,
        anchor_value: float,
        anchor_screen: float,
        lower_bound: float,
        upper_bound: float,
    ) -> float:
        bounded_value = min(max(value, lower_bound), upper_bound)
        if bounded_value >= anchor_value:
            available_hand_room = max(upper_bound - anchor_value, 1e-6)
            screen_room = max(1.0 - anchor_screen, 1e-6)
            progress = clamp01((bounded_value - anchor_value) / available_hand_room)
            eased = progress * progress * (3.0 - 2.0 * progress)
            return clamp01(anchor_screen + screen_room * eased)

        available_hand_room = max(anchor_value - lower_bound, 1e-6)
        screen_room = max(anchor_screen, 1e-6)
        progress = clamp01((anchor_value - bounded_value) / available_hand_room)
        eased = progress * progress * (3.0 - 2.0 * progress)
        return clamp01(anchor_screen - screen_room * eased)

    def _update_cursor(
        self,
        hand_reading,
        *,
        cursor_seed: tuple[float, float] | None,
        reanchor: bool,
    ) -> tuple[float, float]:
        target_x, target_y = self._absolute_cursor_target(hand_reading)
        if self._cursor_position is None:
            self._cursor_position = cursor_seed if cursor_seed is not None else (target_x, target_y)
        if reanchor or self._cursor_anchor_hand is None or self._cursor_anchor_screen is None:
            # Refresh the anchor refs (used by the debug overlay
            # only) but DO NOT warp _cursor_position to cursor_seed.
            # The previous version snapped here, which produced the
            # "cursor rubber-bands toward the OS cursor's last
            # position the moment a click is detected" symptom: a
            # single-frame motion-pose dropout between cursor
            # tracking and pinch detection cleared
            # _cursor_reference_active, so the next frame's pinch
            # entered _update_cursor with reanchor=True and warped
            # the smoothed cursor onto the lagged OS cursor — often
            # mid-screen, producing the visible jump-on-click. The
            # absolute palm→screen mapping with smoothing converges
            # naturally without any warp here.
            self._cursor_anchor_hand = (float(hand_reading.palm.center[0]), float(hand_reading.palm.center[1]))
            self._cursor_anchor_screen = (target_x, target_y)

        dx = target_x - self._cursor_position[0]
        dy = target_y - self._cursor_position[1]
        motion = math.hypot(dx, dy)
        if motion <= self.cursor_deadzone:
            return self._cursor_position

        # Velocity-adaptive alpha tuned for cursor precision and
        # smoothness. User reported normal-speed hover/aim still
        # felt slightly jittery — slow band tightened from 0.40 to
        # 0.34 for more damping during fine targeting. Fast band
        # stays at 0.86 so big sweeps still arrive promptly without
        # adding any new perceived lag. Curve points:
        #
        #   motion just above deadzone (~0.012, slow precision):
        #     alpha = 0.34  -> heavier smoothing for hover/aim
        #   motion ~0.05 (deliberate move):
        #     alpha ~ 0.60  -> smooth but responsive
        #   motion >= 0.10 (fast sweep):
        #     alpha = 0.86  -> near-passthrough so big sweeps
        #                      arrive in the same frame batch
        if motion >= 0.10:
            alpha = 0.86
        else:
            t = motion / 0.10  # 0..1 across the slow-to-fast band
            alpha = 0.34 + (0.86 - 0.34) * t

        # Click-latch damping: shrink alpha hard for the FIRST
        # ~200 ms after a pinch starts so the click lands on
        # whatever the user was aiming at when they began the
        # pinch — protects against the natural index-curl-toward-
        # thumb motion of pinching dragging the cursor off-target.
        # After that brief settle, drop back to normal alpha so a
        # click-and-drag doesn't feel sluggish for the whole drag
        # duration. Earlier version applied the damping for the
        # entire pinch hold, which the user reported as "very
        # leggy when clicking" — every drag felt stuck because
        # alpha was 0.35x normal the whole way through.
        #
        # User reported clicks were still drifting off small
        # buttons. The previous values (0.12 s @ 0.18x) didn't
        # damp hard enough or long enough — a slow pinch finishes
        # AFTER the latch window, so the cursor was free to drift
        # again right as the actual button-down fired. Bumped to
        # 0.20 s @ 0.08x: cursor essentially freezes for the full
        # duration of a normal click gesture (typical pinch
        # completes in 80-180 ms).
        #
        # We pick the most recent press timestamp across both
        # finger states (left/right pinch) so right-click pinches
        # get the same stabilization on their first frames.
        latest_press = None
        for state in (self._index_state, self._middle_state):
            if state.press_started_at is not None:
                if latest_press is None or state.press_started_at > latest_press:
                    latest_press = state.press_started_at
        if latest_press is not None:
            press_age = max(0.0, time.monotonic() - latest_press)
            if press_age < 0.20:
                # Click moment — cursor essentially frozen so the
                # click lands exactly where the user was aiming.
                alpha *= 0.08
            # else: pinch is held but past the click-settle window;
            # use the natural alpha so click-and-drag tracks the
            # hand normally.

        self._cursor_position = (
            clamp01(self._cursor_position[0] + alpha * dx),
            clamp01(self._cursor_position[1] + alpha * dy),
        )
        return self._cursor_position

    def _update_left_sequence(self, hand_reading, now: float) -> tuple[bool, bool, bool]:
        """Pinch-driven left button sequence. State machine:
            not-pinching -> pinching: emit left_press, hold drag.
            pinching -> not-pinching: emit left_release.
        The press/release pair turns into a clean MOUSEEVENTF_LEFTDOWN
        + MOUSEEVENTF_LEFTUP at the controller layer, which Windows
        interprets as a single left click for short taps and as a
        click-and-drag for held pinches — both for free, no separate
        click event needed.

        We do NOT also emit left_click on tap-style releases: the
        controller's left_click() helper fires its own synthetic
        down+up, which when stacked on top of the press/release pair
        produces down-up-DOWN-UP per tap. Windows reads that as a
        double-click or, more commonly, drops one half of it
        entirely — that's the "doesn't actually click always even
        though the app detected clicking" symptom the user reported.
        """
        pinching = self._pinch_active(hand_reading, "index")
        was_pressing = self._index_state.press_started_at is not None

        if self._dragging:
            # We're already in held-mouse-button mode. Stay there
            # until the pinch is released.
            if not pinching:
                self._dragging = False
                self._reset_sequence(self._index_state, preserve_open=not pinching)
                return False, True, False
            return False, False, False

        if pinching and not was_pressing:
            # Pinch just started. Emit press + enter drag immediately
            # so the mouse button goes down on the first frame —
            # users expect "tips touch -> button down" with no
            # buffering delay.
            self._index_state.press_started_at = now
            self._dragging = True
            return True, False, False

        if not pinching and was_pressing:
            # Pinch just released. Emit release; the press+release
            # already produces a single Windows click for short
            # pinches, so no extra left_click event needed.
            self._reset_sequence(self._index_state, preserve_open=True)
            return False, True, False

        # No edge — either still pinching but already dragging
        # (handled above), or still not pinching.
        return False, False, False

    def _update_right_sequence(self, hand_reading, now: float) -> bool:
        """Pinch-driven right button sequence. Right-click is one-
        shot rather than held (the OS doesn't have a meaningful
        "right-click drag" mode), so we emit right_click on the
        not-pinching -> pinching edge and then hold off until the
        user releases."""
        pinching = self._pinch_active(hand_reading, "middle")
        was_pressing = self._middle_state.press_started_at is not None

        if pinching and not was_pressing:
            self._middle_state.press_started_at = now
            return True
        if not pinching and was_pressing:
            self._reset_sequence(self._middle_state, preserve_open=True)
        return False

    def _reset_sequence(self, state: _FingerSequenceState, *, preserve_open: bool = False) -> None:
        state.curl_frames = 0
        state.press_started_at = None
        state.open_frames = 1 if preserve_open else 0

    def _sequence_active(self, state: _FingerSequenceState) -> bool:
        return state.press_started_at is not None or state.curl_frames > 0

    def _toggle_pose_active(self, prediction, hand_handedness: str | None) -> bool:
        if prediction is None:
            return False
        if str(hand_handedness or "").lower() != "left":
            return False
        stable_label = str(getattr(prediction, "stable_label", "neutral") or "neutral")
        raw_label = str(getattr(prediction, "raw_label", "neutral") or "neutral")
        confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        return (
            (raw_label == "three" and confidence >= 0.58)
            or (stable_label == "three" and raw_label in {"three", "neutral"} and confidence >= 0.38)
        )

    def _scroll_pose_active(self, prediction, hand_reading) -> bool:
        if prediction is not None:
            stable_label = str(getattr(prediction, "stable_label", "neutral") or "neutral")
            raw_label = str(getattr(prediction, "raw_label", "neutral") or "neutral")
            confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
            if (
                (raw_label == "wheel_pose" and confidence >= 0.56)
                or (stable_label == "wheel_pose" and raw_label in {"wheel_pose", "neutral"} and confidence >= 0.38)
            ):
                return True
        if hand_reading is None:
            return False
        return self._scroll_pose_ready_from_hand(hand_reading) or self._two_finger_scroll_pose_ready_from_hand(hand_reading)

    def _two_finger_scroll_pose_ready_from_hand(self, hand_reading) -> bool:
        """Two-finger scroll pose: index + middle extended and
        TOUCHING (closed peace sign), ring + pinky curled in.

        The previous version required `_primary_open` for both index
        AND middle (strict bend angles + palm distance) AND
        `_scroll_folded` for ring AND pinky (must be partially-curled
        with curl >= 0.42). In practice, holding a closed peace sign
        relaxes the proximal joints just enough that a casual user
        rarely hits all four thresholds simultaneously — the user
        reported the gesture "doesn't work". Here we relax all four
        gates and add a geometric landmark-distance fallback so the
        pose triggers reliably without requiring the user to
        hyper-extend their fingers.
        """
        fingers = hand_reading.fingers
        index_middle = hand_reading.spreads["index_middle"]

        # Index + middle: looser "extended" check. Use the standard
        # motion-ready relaxation, OR accept any non-fully-curled
        # state with reasonable openness. This catches the natural
        # peace-sign-closed pose where the joints are slightly bent.
        def _two_up(finger) -> bool:
            if self._motion_primary_ready(finger):
                return True
            if finger.state in {"fully_open", "mostly_open"}:
                return True
            if finger.state == "partially_curled" and finger.openness >= 0.50 and finger.curl <= 0.55:
                return True
            return False

        # Ring + pinky: looser "not extended" check. Anything that
        # ISN'T clearly open passes. Most users naturally curl these
        # when holding a peace sign without forcing them into a
        # tight fist.
        def _two_down(finger) -> bool:
            if finger.state in {"partially_curled", "mostly_curled", "closed"}:
                return True
            if finger.state == "mostly_open" and finger.curl >= 0.30:
                return True
            return finger.openness <= 0.55

        if not (_two_up(fingers["index"]) and _two_up(fingers["middle"])):
            return False
        if not (_two_down(fingers["ring"]) and _two_down(fingers["pinky"])):
            return False

        # "Together" check: rely on the spread descriptor, but also
        # accept a direct landmark-distance fallback so the pose
        # registers even if the spread classifier hasn't latched
        # onto "together" yet.
        if index_middle.state == "together":
            return True
        if index_middle.distance <= 0.42:
            return True
        if index_middle.together_strength >= 0.15:
            return True
        try:
            lm = hand_reading.landmarks
            if lm is not None and len(lm) > 12:
                ix, iy = float(lm[8][0]), float(lm[8][1])
                mx, my = float(lm[12][0]), float(lm[12][1])
                tip_dist = math.hypot(ix - mx, iy - my)
                hand_size = self._hand_size(lm)
                # Tip-to-tip distance under ~half a hand-size = the
                # tips are visibly touching. Same scale-invariance
                # reasoning as the pinch detection.
                if tip_dist <= 0.55 * hand_size:
                    return True
        except Exception:
            pass
        return False

    def _motion_pose_ready(
        self,
        hand_reading,
        *,
        allow_index_curled: bool,
        allow_middle_curled: bool,
    ) -> bool:
        fingers = hand_reading.fingers
        index_ready = self._motion_primary_ready(fingers["index"]) or (
            allow_index_curled and self._click_curled(fingers["index"])
        )
        middle_ready = self._motion_primary_ready(fingers["middle"]) or (
            allow_middle_curled and self._click_curled(fingers["middle"])
        )
        return (
            self._thumb_open(fingers["thumb"])
            and self._support_open(fingers["ring"])
            and self._support_open(fingers["pinky"])
            and index_ready
            and middle_ready
        )

    def _click_pose_active(self, hand_reading, *, primary: str) -> bool:
        # Was: curl-based check ("the named finger is currently
        # curled, and the rest of the hand is in the click context").
        # Now: pinch-based — _pinch_active already does the
        # "tip-to-thumb close + other 3 fingers relaxed" composite
        # check, which IS the new click pose. The primary's not-fully-
        # extended state is implicit (you can't pinch with a fully-
        # extended finger), so the original click-context gates are
        # subsumed.
        return self._pinch_active(hand_reading, primary)

    def _scroll_pose_ready_from_hand(self, hand_reading) -> bool:
        fingers = hand_reading.fingers
        thumb_index = hand_reading.spreads["thumb_index"]
        ring_pinky = hand_reading.spreads["ring_pinky"]
        return (
            self._thumb_open(fingers["thumb"])
            and self._primary_open(fingers["index"])
            and self._support_open(fingers["pinky"])
            and self._scroll_folded(fingers["middle"])
            and self._scroll_folded(fingers["ring"])
            and (
                thumb_index.state == "apart"
                or thumb_index.distance >= 0.30
                or thumb_index.apart_strength >= 0.18
            )
            and (
                ring_pinky.state == "apart"
                or ring_pinky.distance >= 0.12
                or ring_pinky.apart_strength >= 0.12
            )
        )

    def _left_click_context_ready(self, hand_reading) -> bool:
        fingers = hand_reading.fingers
        return (
            self._thumb_open(fingers["thumb"])
            and self._support_open(fingers["ring"])
            and self._support_open(fingers["pinky"])
            and self._motion_primary_ready(fingers["middle"])
            and (self._motion_primary_ready(fingers["index"]) or self._click_curled(fingers["index"]))
        )

    def _right_click_context_ready(self, hand_reading) -> bool:
        fingers = hand_reading.fingers
        return (
            self._thumb_open(fingers["thumb"])
            and self._support_open(fingers["ring"])
            and self._support_open(fingers["pinky"])
            and self._motion_primary_ready(fingers["index"])
            and (self._motion_primary_ready(fingers["middle"]) or self._click_curled(fingers["middle"]))
        )

    def _thumb_open(self, finger) -> bool:
        return finger.state == "fully_open" or (
            finger.openness >= 0.60
            and finger.curl <= 0.46
            and finger.bend_distal >= 122.0
            and finger.palm_distance >= 0.54
        )

    def _motion_primary_ready(self, finger) -> bool:
        return self._primary_open(finger) or (
            finger.openness >= 0.50
            and finger.curl <= 0.58
            and finger.bend_proximal >= 126.0
            and finger.bend_distal >= 120.0
            and finger.palm_distance >= 0.52
        )

    def _primary_open(self, finger) -> bool:
        return finger.state == "fully_open" or (
            finger.openness >= 0.58
            and finger.curl <= 0.48
            and finger.bend_proximal >= 132.0
            and finger.bend_distal >= 126.0
            and (finger.reach >= 0.02 or finger.palm_distance >= 0.58)
        )

    def _support_open(self, finger) -> bool:
        return finger.state == "fully_open" or (
            finger.openness >= 0.58
            and finger.curl <= 0.46
            and finger.bend_proximal >= 136.0
            and finger.bend_distal >= 134.0
            and finger.palm_distance >= 0.60
        )

    def _thumb_folded(self, finger) -> bool:
        return finger.state in {"mostly_curled", "closed"} or (
            finger.state == "partially_curled" and finger.openness <= 0.62
        ) or (finger.openness <= 0.56 and finger.curl >= 0.42)

    def _scroll_folded(self, finger) -> bool:
        return finger.state in {"mostly_curled", "closed"} or (
            finger.state == "partially_curled"
            and finger.openness <= 0.58
            and finger.curl >= 0.42
            and finger.bend_distal <= 148.0
        )

    def _click_curled(self, finger) -> bool:
        return finger.state in {"mostly_curled", "closed"} or (
            finger.state == "partially_curled"
            and finger.openness <= 0.70
            and finger.curl >= 0.40
            and finger.bend_distal <= 146.0
        )

    # ---- Pinch (thumb-tip ↔ finger-tip) click detection -----------
    # Replaces the original curl-based click logic. User-facing
    # behavior: bring the thumb tip and the index tip close together
    # to hold left-click; thumb tip + middle tip = right-click. The
    # other three fingers must be extended or partial-curl (not a
    # full fist) so we don't fire on closed-hand poses.
    #
    # Distance is normalized by hand size (wrist-to-index-MCP) so the
    # threshold works at any camera distance — the same pinch
    # gesture produces the same ratio whether the hand is 30 cm or
    # 90 cm from the camera. Threshold 0.42 was tuned empirically:
    # actual tip-touch lands at ~0.10-0.18, comfortable air-pinch
    # (1-2 cm gap) at ~0.30-0.38, neutral relaxed hand at ~0.55+.
    _PINCH_THUMB_TIP_LM = 4
    _PINCH_TIP_LMS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    _PINCH_WRIST_LM = 0
    _PINCH_INDEX_MCP_LM = 5
    _PINCH_THRESHOLD = 0.42  # tip-distance / hand-size below this = pinch

    def _hand_size(self, landmarks) -> float:
        """Wrist-to-index-MCP distance, used as the normalization
        reference. This segment of the hand barely changes shape
        across poses, which makes it a more stable scale ref than
        bbox dimensions (which stretch with finger spread)."""
        try:
            wrist = landmarks[self._PINCH_WRIST_LM]
            mcp = landmarks[self._PINCH_INDEX_MCP_LM]
            dx = float(wrist[0]) - float(mcp[0])
            dy = float(wrist[1]) - float(mcp[1])
            return max(0.001, math.hypot(dx, dy))
        except Exception:
            return 0.001

    def _pinch_distance_ratio(self, hand_reading, finger_name: str) -> float:
        """3D distance from thumb tip to the named finger's tip,
        divided by hand size. Returns +inf if landmarks aren't
        available so the caller's threshold check falls through
        to 'not pinching'.

        Includes the Z (depth) component because the user's
        natural pinch with fingers pointing straight up mostly
        moves the thumb along Z (toward / away from the camera)
        with little X/Y motion. A 2D-only distance stayed below
        threshold the whole time -- no transition, no click.
        Adding Z gives a clear transition: not-pinching is
        ~0.6-0.9 of hand-size in 3D, pinching is ~0-0.2.

        MediaPipe Z is normalized to roughly the same scale as
        X/Y per landmark, so we don't need a Z weight."""
        try:
            landmarks = hand_reading.landmarks
            if landmarks is None or len(landmarks) <= self._PINCH_TIP_LMS[finger_name]:
                return float("inf")
            thumb = landmarks[self._PINCH_THUMB_TIP_LM]
            target = landmarks[self._PINCH_TIP_LMS[finger_name]]
            dx = float(thumb[0]) - float(target[0])
            dy = float(thumb[1]) - float(target[1])
            # Defensive: some landmark formats omit Z. Default to
            # 0 so a missing depth reduces gracefully to the old
            # 2D behaviour rather than raising.
            try:
                dz = float(thumb[2]) - float(target[2])
            except (IndexError, TypeError):
                dz = 0.0
            tip_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            return tip_dist / self._hand_size(landmarks)
        except Exception:
            return float("inf")

    def _pinch_others_relaxed(self, hand_reading, primary: str) -> bool:
        """The 3 fingers NOT involved in the current pinch must be
        extended or partial-curl (not closed/fist). Without this
        guard, a closed-hand pose with thumb tucked over index would
        still register as a pinch and fire spurious clicks. Uses
        finger.state from the standard reading rather than landmark
        distance so the reading's own smoothing applies."""
        if primary == "index":
            others = ("middle", "ring", "pinky")
        elif primary == "middle":
            others = ("index", "ring", "pinky")
        else:
            return False
        for name in others:
            finger = hand_reading.fingers.get(name)
            if finger is None:
                return False
            # Reject anything that's clearly a full curl. "fully_open",
            # "mostly_open", and "partially_curled" all pass; only
            # "mostly_curled" and "closed" fail.
            if finger.state in {"mostly_curled", "closed"}:
                return False
        return True

    def _pinch_active(self, hand_reading, primary: str) -> bool:
        """True iff the user is currently holding a {primary}-pinch:
        thumb tip + named-finger tip close, other 3 fingers relaxed.
        Drives both the press/release logic and the "is the cursor
        pose still acceptable while clicking" gates in the main
        update loop."""
        if self._pinch_distance_ratio(hand_reading, primary) > self._PINCH_THRESHOLD:
            return False
        return self._pinch_others_relaxed(hand_reading, primary)

# Author: Konstantin Markov

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from ..analysis.geometry import clamp01
from ..models import GestureCandidate, HandReading


@dataclass(frozen=True)
class MotionSample:
    timestamp: float
    center: np.ndarray
    index_tip: np.ndarray
    scale: float
    pose_gate: float
    one_pose_gate: float


class DynamicGestureRecognizer:
    def __init__(self, *, low_fps_mode: bool = False) -> None:
        self.history: Deque[MotionSample] = deque(maxlen=24)
        self._blocked_horizontal_label: str | None = None
        self._blocked_horizontal_until = 0.0
        self.low_fps_mode = bool(low_fps_mode)
        # Index-only horizontal swipe for drawing undo/clear. Not
        # exposed as dynamic_label — builtin swipe_left/right are
        # open-hand only so they don't collide with custom "1 swipe".
        self.last_one_pose_horizontal_label: str = "neutral"

    def reset(self) -> None:
        self.history.clear()
        self._blocked_horizontal_label = None
        self._blocked_horizontal_until = 0.0
        self.last_one_pose_horizontal_label = "neutral"

    def _effective_low_fps(self) -> bool:
        """True when swipes should use the relaxed (low-fps) gates — either the
        engine forced low_fps_mode, OR (Darwin only) actual capture cadence is
        slow (median inter-sample dt > ~0.072 s, i.e. under ~14 fps).

        The swipe gates use a fixed-SAMPLE window, so at low fps that window
        spans too much wall-clock time and the duration/step gates silently
        zero the score. macOS can still dip into a ~12–14 fps band where
        the engine's own low_fps auto-toggle (fps<12) hasn't engaged yet.
        Deriving this per-frame from the timestamps already on the samples
        fixes that without waiting on the engine. At normal ~30 fps
        (dt ~0.033 s) this returns False, so Windows behavior is unchanged."""
        if self.low_fps_mode:
            return True
        # macOS-only: the dt-based auto-relax trades some precision (slow
        # motion can read as a swipe) for recall at the mac's low fps. Windows
        # keeps its exact tuned behavior (and its tests) — it engages the
        # relaxed gates only via the explicit engine low_fps_mode there.
        # 0.055 s (~18 fps) was firing on the 15–18 fps band after the
        # overlay fix, so casual hand drift scored as swipe_left/right.
        # Only relax when we are actually in the engine's low-fps zone.
        if sys.platform != "darwin":
            return False
        recent = list(self.history)[-8:]
        dts = [b.timestamp - a.timestamp for a, b in zip(recent, recent[1:])
               if b.timestamp > a.timestamp]
        if len(dts) < 2:
            return False
        dts.sort()
        median_dt = dts[len(dts) // 2]
        return median_dt > 0.072

    def _fold_gate(self, finger) -> float:
        if finger.state == "closed":
            return 1.0
        if finger.state == "mostly_curled":
            return max(0.76, finger.curl)
        if finger.state == "partially_curled":
            return clamp01((finger.curl - 0.30) / 0.38) * clamp01((0.74 - finger.openness) / 0.30)
        return 0.0

    def _one_pose_gate(self, hand: HandReading) -> float:
        index = hand.fingers["index"]
        index_gate = max(
            1.0 if index.state == "fully_open" else 0.0,
            0.72 if index.state == "partially_curled" and index.openness >= 0.70 and index.curl <= 0.42 else 0.0,
            clamp01((index.openness - 0.68) / 0.18),
        )
        folded_avg = sum(self._fold_gate(hand.fingers[name]) for name in ("thumb", "middle", "ring", "pinky")) / 4.0
        finger_count_gate = clamp01((2.3 - float(hand.finger_count_extended)) / 1.1)
        return clamp01((0.56 * index_gate + 0.44 * folded_avg) * (0.34 + 0.66 * finger_count_gate))

    def update(self, hand: HandReading, timestamp: float) -> tuple[str, tuple[GestureCandidate, ...], dict[str, float]]:
        # Open-hand swipe: index AND middle must be extended. Averaging
        # let index-only ("one") score ~0.5 and fire builtin swipe_right,
        # colliding with custom "1 swipe right". Match the core
        # classifier: min(index, middle) plus some ring/pinky support.
        index_open = float(hand.fingers["index"].openness)
        middle_open = float(hand.fingers["middle"].openness)
        ring_open = float(hand.fingers["ring"].openness)
        pinky_open = float(hand.fingers["pinky"].openness)
        primary_open_gate = clamp01((min(index_open, middle_open) - 0.52) / 0.20)
        support_open_gate = clamp01((max(ring_open, pinky_open) - 0.34) / 0.28)
        pose_gate = clamp01(primary_open_gate * (0.55 + 0.45 * support_open_gate))
        one_pose_gate = self._one_pose_gate(hand)
        self.history.append(
            MotionSample(
                timestamp=timestamp,
                center=hand.palm.center.copy(),
                index_tip=hand.landmarks[8].copy(),
                scale=max(hand.palm.scale, 1e-6),
                pose_gate=pose_gate,
                one_pose_gate=one_pose_gate,
            )
        )
        effective_low_fps = self._effective_low_fps()
        min_samples = 3 if effective_low_fps else 4
        if len(self.history) < min_samples:
            self.last_one_pose_horizontal_label = "neutral"
            return "neutral", tuple(), {}

        if timestamp >= self._blocked_horizontal_until:
            self._blocked_horizontal_label = None
            self._blocked_horizontal_until = 0.0

        window_size = 6 if effective_low_fps else 9
        window = list(self.history)[-window_size:]
        first = window[0]
        last = window[-1]
        scale = max(last.scale, 1e-6)
        duration = max(last.timestamp - first.timestamp, 1e-6)
        displacement = (last.center - first.center) / scale
        horizontal = float(displacement[0])
        vertical = abs(float(displacement[1]))
        depth = abs(float(displacement[2]))

        path = 0.0
        peak_horizontal_speed = 0.0
        vertical_noise = 0.0
        depth_noise = 0.0
        positive_x_steps = 0
        negative_x_steps = 0
        step_threshold = 0.007 if effective_low_fps else 0.02
        for prev, current in zip(window, window[1:]):
            step = (current.center - prev.center) / max(current.scale, 1e-6)
            step_duration = max(current.timestamp - prev.timestamp, 1e-6)
            path += float(np.linalg.norm(step))
            vertical_noise += abs(float(step[1]))
            depth_noise += abs(float(step[2]))
            peak_horizontal_speed = max(peak_horizontal_speed, abs(float(step[0])) / step_duration)
            if step[0] > step_threshold:
                positive_x_steps += 1
            if step[0] < -step_threshold:
                negative_x_steps += 1

        # Builtin swipe_left/right: open-hand pose only. Index-only
        # horizontal motion is scored separately for drawing undo/clear
        # and is NOT published as dynamic_label.
        open_pose_strength = sum(sample.pose_gate for sample in window) / len(window)
        one_pose_strength = sum(sample.one_pose_gate for sample in window) / len(window)
        straightness = clamp01(abs(horizontal) / max(path, 1e-6))
        # Circles have comparable X and Y travel. Require a clearly
        # horizontal axis (was 1.35) so an arc of a loop cannot pass
        # as swipe_left/right.
        horizontal_axis_gate = clamp01(((abs(horizontal) / max(vertical + 0.62 * depth, 1e-6)) - 1.75) / 0.80)
        if effective_low_fps:
            horizontal_min_duration_gate = clamp01((duration - 0.04) / 0.05)
            horizontal_max_duration_gate = clamp01((1.25 - duration) / 0.50)
            positive_x_gate = clamp01((positive_x_steps - negative_x_steps - 0.4) / 1.2)
            negative_x_gate = clamp01((negative_x_steps - positive_x_steps - 0.4) / 1.2)
            # Darwin: need a bit more palm travel before commit. Windows
            # low-fps keeps the original 0.18 floor.
            commit_floor = 0.28 if sys.platform == "darwin" else 0.18
            horizontal_commit_gate = clamp01((abs(horizontal) - commit_floor) / 0.12)
        else:
            horizontal_min_duration_gate = clamp01((duration - 0.12) / 0.08)
            horizontal_max_duration_gate = clamp01((0.78 - duration) / 0.30)
            positive_x_gate = clamp01((positive_x_steps - negative_x_steps - 1.6) / 1.5)
            negative_x_gate = clamp01((negative_x_steps - positive_x_steps - 1.4) / 1.6)
            horizontal_commit_gate = 1.0
        horizontal_duration_gate = horizontal_min_duration_gate * horizontal_max_duration_gate

        if effective_low_fps:
            right_h_floor = 0.28
            left_h_floor = 0.27
            speed_floor_r = 0.44
            speed_floor_l = 0.40
            path_floor_r = 0.34
            path_floor_l = 0.32
            if sys.platform == "darwin":
                right_h_floor = 0.36
                left_h_floor = 0.34
                speed_floor_r = 0.52
                speed_floor_l = 0.48
                path_floor_r = 0.42
                path_floor_l = 0.40
        else:
            # r52: loosened Normal-mode swipe geometry and speed
            # floors so an ordinary committed swipe registers on
            # the first try. Prior values (0.67/0.61, 1.35/1.22,
            # 0.80/0.76) required nearly a full-palm sweep at
            # 1.2+ palm/s — casual users had to over-swipe. New
            # floors still sit well above Low-FPS's permissive
            # values (0.28/0.27, 0.44/0.40, 0.34/0.32) so idle
            # hand drift still fails the horizontal gates.
            right_h_floor = 0.50
            left_h_floor = 0.46
            speed_floor_r = 0.95
            speed_floor_l = 0.85
            path_floor_r = 0.60
            path_floor_l = 0.58
            # Darwin: a little less twitchy than Windows r52. Still
            # well below the old 0.67/0.61 floors so a committed swipe
            # fires; idle drift should not.
            if sys.platform == "darwin":
                right_h_floor = 0.56
                left_h_floor = 0.52
                speed_floor_r = 1.05
                speed_floor_l = 0.95
                path_floor_r = 0.66
                path_floor_l = 0.64

        def _horizontal_scores(pose_strength: float) -> tuple[float, float]:
            right = clamp01(
                (
                    0.32 * clamp01((horizontal - right_h_floor) / 0.26)
                    + 0.18 * clamp01((path - path_floor_r) / 0.46)
                    + 0.16 * clamp01((peak_horizontal_speed - speed_floor_r) / 0.95)
                    + 0.14 * clamp01((horizontal - 1.45 * vertical - 0.66 * depth - 0.06) / 0.24)
                    + 0.10 * straightness
                    + 0.10 * clamp01((0.16 - vertical_noise) / 0.14)
                )
                * (0.28 + 0.72 * pose_strength)
                * horizontal_duration_gate
                * horizontal_commit_gate
                * positive_x_gate
                * horizontal_axis_gate
                * clamp01((0.26 - depth_noise) / 0.20)
            )
            left = clamp01(
                (
                    0.32 * clamp01(((-horizontal) - left_h_floor) / 0.30)
                    + 0.18 * clamp01((path - path_floor_l) / 0.50)
                    + 0.16 * clamp01((peak_horizontal_speed - speed_floor_l) / 1.00)
                    + 0.14 * clamp01(((-horizontal) - 1.40 * vertical - 0.62 * depth - 0.04) / 0.26)
                    + 0.10 * straightness
                    + 0.10 * clamp01((0.16 - vertical_noise) / 0.16)
                )
                * (0.28 + 0.72 * pose_strength)
                * horizontal_duration_gate
                * horizontal_commit_gate
                * negative_x_gate
                * horizontal_axis_gate
                * clamp01((0.28 - depth_noise) / 0.22)
            )
            return left, right

        left_score, right_score = _horizontal_scores(open_pose_strength)
        drawing_left, drawing_right = _horizontal_scores(one_pose_strength)

        repeat_score = 0.0
        circle_window = list(self.history)[-12:]
        if len(circle_window) >= 6:
            circle_first = circle_window[0]
            circle_last = circle_window[-1]
            circle_duration = max(circle_last.timestamp - circle_first.timestamp, 1e-6)
            circle_scale = max(circle_last.scale, 1e-6)
            tip_path = 0.0
            positive_tip_x = 0
            negative_tip_x = 0
            positive_tip_y = 0
            negative_tip_y = 0
            x_values = [float(sample.index_tip[0]) for sample in circle_window]
            y_values = [float(sample.index_tip[1]) for sample in circle_window]
            for prev, current in zip(circle_window, circle_window[1:]):
                tip_step = (current.index_tip - prev.index_tip) / max(current.scale, 1e-6)
                tip_path += float(np.linalg.norm(tip_step[:2]))
                if tip_step[0] > 0.02:
                    positive_tip_x += 1
                if tip_step[0] < -0.02:
                    negative_tip_x += 1
                if tip_step[1] > 0.02:
                    positive_tip_y += 1
                if tip_step[1] < -0.02:
                    negative_tip_y += 1

            x_span = (max(x_values) - min(x_values)) / circle_scale
            y_span = (max(y_values) - min(y_values)) / circle_scale
            closure = float(np.linalg.norm((circle_last.index_tip - circle_first.index_tip) / circle_scale))
            one_pose_strength = sum(sample.one_pose_gate for sample in circle_window) / len(circle_window)
            aspect = min(x_span, y_span) / max(max(x_span, y_span), 1e-6)
            turn_gate = min(
                clamp01((min(positive_tip_x, negative_tip_x) - 1.0) / 1.2),
                clamp01((min(positive_tip_y, negative_tip_y) - 1.0) / 1.2),
            )
            circle_duration_gate = clamp01((circle_duration - 0.24) / 0.16) * clamp01((1.10 - circle_duration) / 0.36)
            repeat_score = clamp01(
                (
                    0.24 * min(clamp01((x_span - 0.14) / 0.14), clamp01((y_span - 0.14) / 0.14))
                    + 0.22 * clamp01((tip_path - 0.74) / 0.44)
                    + 0.20 * clamp01((0.40 - closure) / 0.22)
                    + 0.18 * clamp01((aspect - 0.36) / 0.26)
                    + 0.16 * turn_gate
                )
                * (0.18 + 0.82 * one_pose_strength)
                * circle_duration_gate
            )

        # A looping path reverses in X and travels in Y. That is a
        # circle / arc, not a committed swipe — even when one short
        # window of the loop looks mostly horizontal.
        plus_x = minus_x = plus_y = minus_y = 0
        loop_window = list(self.history)[-12:]
        for prev, current in zip(loop_window, loop_window[1:]):
            scale = max(current.scale, 1e-6)
            dx = float(current.center[0] - prev.center[0]) / scale
            dy = float(current.center[1] - prev.center[1]) / scale
            tip_dx = float(current.index_tip[0] - prev.index_tip[0]) / scale
            tip_dy = float(current.index_tip[1] - prev.index_tip[1]) / scale
            if dx > 0.02 or tip_dx > 0.02:
                plus_x += 1
            if dx < -0.02 or tip_dx < -0.02:
                minus_x += 1
            if dy > 0.02 or tip_dy > 0.02:
                plus_y += 1
            if dy < -0.02 or tip_dy < -0.02:
                minus_y += 1
        if min(plus_x, minus_x) >= 2 and min(plus_y, minus_y) >= 1:
            left_score = 0.0
            right_score = 0.0
        elif repeat_score >= 0.40 and repeat_score + 0.02 >= max(left_score, right_score):
            left_score = 0.0
            right_score = 0.0

        if self._blocked_horizontal_label == "swipe_left" and timestamp < self._blocked_horizontal_until:
            left_score = 0.0
        if self._blocked_horizontal_label == "swipe_right" and timestamp < self._blocked_horizontal_until:
            right_score = 0.0

        scores = {
            "swipe_left": left_score,
            "swipe_right": right_score,
            "repeat_circle": repeat_score,
        }
        ranked = tuple(
            sorted(
                (GestureCandidate(label, score, "dynamic") for label, score in scores.items()),
                key=lambda item: item.score,
                reverse=True,
            )
        )
        best = ranked[0] if ranked else GestureCandidate("neutral", 0.0, "dynamic")
        # r52: Normal score_floor 0.59 → 0.48. Swipes scoring in the
        # 0.45-0.55 band (moderate committed motion) were silently
        # dropped to "neutral". 0.48 still sits ~14 pts above the
        # neutral noise band.
        score_floor = 0.34 if effective_low_fps else 0.48
        if sys.platform == "darwin":
            score_floor = 0.42 if effective_low_fps else 0.54
        drawing_best_score = max(drawing_left, drawing_right)
        if drawing_best_score >= score_floor:
            self.last_one_pose_horizontal_label = (
                "swipe_right" if drawing_right >= drawing_left else "swipe_left"
            )
        else:
            self.last_one_pose_horizontal_label = "neutral"
        if best.score < score_floor:
            return "neutral", ranked, scores

        if best.label == "swipe_left":
            self._blocked_horizontal_label = "swipe_right"
            self._blocked_horizontal_until = timestamp + 1.2
        elif best.label == "swipe_right":
            self._blocked_horizontal_label = "swipe_left"
            self._blocked_horizontal_until = timestamp + 1.2
        return best.label, ranked, scores

# Author: Konstantin Markov

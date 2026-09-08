from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VolumeGestureUpdate:
    active: bool
    level: float | None
    muted: bool
    message: str
    status: str
    overlay_visible: bool
    trigger_mute_toggle: bool = False


class VolumeGestureTracker:
    def __init__(
        self,
        *,
        confirm_frames: int = 3,
        # Bumped from 2 -> 5: at 30 fps that's ~165ms of consecutive
        # invalid frames before deactivating, vs 65ms previously.
        # Pairs with the active-state hysteresis in
        # _is_volume_ready_pose so a brief landmark jitter in the
        # finger-spread metric doesn't toggle volume mode off.
        release_frames: int = 5,
        hold_seconds: float = 1.5,
        mute_cooldown_seconds: float = 1.0,
        # v1.1.7 tester feedback: mute was firing instantly on the
        # first stable frame (~100 ms), which combined with the mute
        # pose's overlap with a swipe-recovery hand shape produced
        # frequent false positives. Requiring the mute pose to hold
        # for 1.0 s before firing eliminates transient false
        # positives (a hand naturally passing through a shaka-shape
        # can't hold it that long) while remaining fast enough that
        # a deliberate mute still feels responsive.
        mute_hold_seconds: float = 1.0,
        deadzone_fraction: float = 0.20,
        smoothing: float = 0.14,
        # Was 0.40 — bumped to 0.60 so a brief pose-invalid window
        # (~half a second of bad frames) doesn't deactivate volume
        # mode while the user is still trying to hold it. Combined
        # with active-state hysteresis above this gives a much more
        # forgiving stay-active behavior without changing entry.
        pose_grace_seconds: float = 0.60,
        no_hand_grace_seconds: float = 0.20,
    ) -> None:
        self.confirm_frames = int(confirm_frames)
        self.release_frames = int(release_frames)
        self.hold_seconds = float(hold_seconds)
        self.mute_cooldown_seconds = float(mute_cooldown_seconds)
        self.mute_hold_seconds = float(mute_hold_seconds)
        self.deadzone_fraction = float(deadzone_fraction)
        self.smoothing = float(smoothing)
        self.pose_grace_seconds = float(pose_grace_seconds)
        self.no_hand_grace_seconds = float(no_hand_grace_seconds)
        self.reset()

    def rebase(self, level: float | None) -> None:
        if level is None:
            return
        clamped = max(0.0, min(1.0, float(level)))
        self._anchor_level = clamped
        self._level = clamped
        self._rebase_pending = True

    def reset(self, level: float | None = None, muted: bool = False) -> None:
        self._active = False
        self._confirm_frames = 0
        self._release_frames = 0
        self._anchor_y: float | None = None
        self._anchor_level: float | None = level
        self._level = level
        self._muted = bool(muted)
        self._message = 'idle'
        self._status = 'idle'
        self._last_mute_toggle_time = 0.0
        self._mute_gesture_latched = False
        # Timestamp when the current continuous run of stable_gesture
        # == "mute" started, or None if the gesture is not currently
        # held. Fires the actual mute toggle only after this run has
        # persisted for `mute_hold_seconds` (default 1.0 s).
        self._mute_candidate_since: float | None = None
        self._lock_until = 0.0
        self._pinky_hold_latched = False
        self._activation_ready_frames = 0
        self._last_pose_valid_time = 0.0
        self._last_seen_time = 0.0
        self._rebase_pending = False

    def update(
        self,
        *,
        features,
        landmarks,
        candidate_scores,
        stable_gesture: str,
        current_level: float | None,
        current_muted: bool,
        now: float,
        allow_mute_toggle: bool = True,
        palm_roll_deg: float | None = None,
    ) -> VolumeGestureUpdate:
        self._muted = bool(current_muted)
        if current_level is not None:
            self._level = current_level

        trigger_mute = False
        if stable_gesture == 'mute':
            # Start (or continue) the candidate-hold timer. If the
            # gesture was just released and reappears, we restart
            # the timer — a fresh 1.0 s hold is required each time.
            if self._mute_candidate_since is None:
                self._mute_candidate_since = now
            candidate_held_for = now - self._mute_candidate_since
            if (
                allow_mute_toggle
                and not self._mute_gesture_latched
                and candidate_held_for >= self.mute_hold_seconds
                and now - self._last_mute_toggle_time >= self.mute_cooldown_seconds
            ):
                self._last_mute_toggle_time = now
                self._mute_gesture_latched = True
                trigger_mute = True
                self._message = 'mute toggle'
                self._status = 'muted' if not current_muted else 'unmuted'
        else:
            # Gesture broken — clear the latch AND the candidate
            # timer so the next mute pose must hold a full 1.0 s
            # from a fresh start, not from wherever the previous
            # incomplete hold left off.
            self._mute_gesture_latched = False
            self._mute_candidate_since = None

        upright_ok = True
        if palm_roll_deg is not None:
            upright_ok = 23.0 <= float(palm_roll_deg) <= 157.0

        if features is None or landmarks is None or not upright_ok:
            if self._active and now - self._last_seen_time <= self.no_hand_grace_seconds and upright_ok:
                return self._snapshot(trigger_mute_toggle=trigger_mute)
            self._deactivate('idle')
            return self._snapshot(trigger_mute_toggle=trigger_mute)

        self._last_seen_time = now
        pose_score = float((candidate_scores or {}).get('volume_pose', 0.0))
        pose_valid = self._is_volume_ready_pose(features, landmarks, pose_score, stable_gesture, active=self._active)
        if pose_valid:
            self._last_pose_valid_time = now
        control_y = float((landmarks[8][1] + landmarks[12][1]) * 0.5)
        if self._rebase_pending:
            self._rebase_pending = False
            self._anchor_y = control_y
        pinky_hold = self._active and self._is_pinky_hold_pose(features)

        if pinky_hold and not self._pinky_hold_latched:
            self._pinky_hold_latched = True
            self._lock_until = max(self._lock_until, now + self.hold_seconds)
            self._anchor_y = control_y
            self._anchor_level = self._level if self._level is not None else current_level
            self._message = 'hold'
            self._status = 'holding'
        elif not pinky_hold:
            self._pinky_hold_latched = False

        lock_active = self._active and now < self._lock_until
        if self._active and self._lock_until and not lock_active:
            self._lock_until = 0.0
            self._anchor_y = control_y
            self._anchor_level = self._level if self._level is not None else current_level

        pose_within_grace = self._active and not pose_valid and (now - self._last_pose_valid_time) <= self.pose_grace_seconds

        if pose_valid:
            self._confirm_frames += 1
            self._release_frames = 0
            if not self._active:
                self._activation_ready_frames += 1
        elif pose_within_grace or lock_active:
            self._confirm_frames = max(self._confirm_frames, 0)
            self._activation_ready_frames = max(self._activation_ready_frames, 0)
            self._release_frames = 0
        else:
            self._confirm_frames = 0
            self._activation_ready_frames = 0
            if self._active:
                self._release_frames += 1

        if pose_valid and not self._active and self._confirm_frames >= self.confirm_frames and self._activation_ready_frames >= self.confirm_frames:
            self._active = True
            self._anchor_y = control_y
            self._anchor_level = self._level if self._level is not None else 0.5
            self._message = 'ready'
            self._status = 'active'
        elif self._active and lock_active:
            self._message = 'hold'
            self._status = 'holding'
        elif self._active and pose_within_grace:
            self._message = 'ready'
            self._status = 'tracking'
        elif self._active and pose_valid:
            if self._anchor_y is not None and self._anchor_level is not None:
                travel_span = max(0.082, min(0.175, float(features.palm_scale) * 0.60))
                delta = self._anchor_y - control_y
                deadzone = travel_span * self.deadzone_fraction
                if abs(delta) <= deadzone:
                    target_level = self._anchor_level
                else:
                    adjusted_delta = delta - deadzone if delta > 0.0 else delta + deadzone
                    raw_level = self._anchor_level + (adjusted_delta / max(travel_span - deadzone, 1e-6))
                    target_level = max(0.0, min(1.0, raw_level))
                if self._level is not None:
                    target_level = (1.0 - self.smoothing) * float(self._level) + self.smoothing * target_level
                self._level = target_level
                self._message = 'ready'
                self._status = 'changing'
        elif self._active and self._release_frames >= self.release_frames:
            self._deactivate('stopped')

        return self._snapshot(trigger_mute_toggle=trigger_mute)

    def _deactivate(self, status: str) -> None:
        self._active = False
        self._confirm_frames = 0
        self._release_frames = 0
        self._anchor_y = None
        self._anchor_level = self._level
        self._lock_until = 0.0
        self._pinky_hold_latched = False
        self._activation_ready_frames = 0
        self._last_pose_valid_time = 0.0
        self._last_seen_time = 0.0
        self._status = status
        if status != 'idle':
            self._message = status

    def _is_volume_ready_pose(self, features, landmarks, pose_score: float, stable_gesture: str, *, active: bool) -> bool:
        open_scores = features.open_scores
        spread_states = getattr(features, 'spread_states', {})
        # Disallowed stable gestures dropped from the previous
        # allow-list: 'two' was being treated as compatible with
        # volume control, so a peace sign with clearly separated
        # fingers kept the volume tracker active and let the user
        # change volume just by tilting their hand. The user
        # explicitly reported this as a bug. Peace sign now
        # disqualifies regardless of state, just like open_hand /
        # four / finger_apart / mute already did.
        disallowed_stable = {'open_hand', 'four', 'finger_apart', 'mute', 'two'}
        if not active and stable_gesture in disallowed_stable:
            return False
        if features.finger_count_open > 2:
            return False
        # Direct tip-to-tip closeness check, mirroring the engine's
        # _volume_pose_ready gate. The blended spread.distance
        # ratio is too loose at the base of a V (PIP joints sit
        # close anatomically regardless of spread), so we measure
        # landmark 8 -> 12 directly.
        try:
            palm_scale = max(float(features.palm_scale), 1e-6)
            dx = float(landmarks[8][0]) - float(landmarks[12][0])
            dy = float(landmarks[8][1]) - float(landmarks[12][1])
            tip_distance_ratio = (dx * dx + dy * dy) ** 0.5 / palm_scale
        except Exception:
            tip_distance_ratio = 1.0
        # Hysteresis: lenient on entry too (so users can actually
        # turn volume mode on with a normal-feeling pose), even more
        # lenient once already active so jitter doesn't kick it off.
        # Earlier values (entry 0.22, open 0.56, etc.) were too
        # strict — users with naturally slightly-spread peace-sign
        # fingers couldn't activate volume mode at all.
        max_tip_distance = 0.32 if active else 0.26
        # Drop the strict 'together' spread-state check entirely.
        # tip_distance_ratio already captures "fingers close enough";
        # the spread-state classifier was double-gating with no
        # added discriminative power and was the main reason real
        # volume poses got rejected on entry.
        index_middle_close = tip_distance_ratio <= max_tip_distance
        if active:
            min_open = 0.46     # was 0.56 originally, now 0.46
            max_fold = 0.80     # was 0.70 originally, now 0.80
            max_thumb = 0.80    # was 0.70 originally, now 0.80
        else:
            min_open = 0.50     # was 0.56 — slightly more permissive entry
            max_fold = 0.74     # was 0.70 — slightly more permissive entry
            max_thumb = 0.74    # was 0.70 — slightly more permissive entry
        structural_ready = (
            self._is_volume_primary_open(features, 'index')
            and self._is_volume_primary_open(features, 'middle')
            and self._is_folded(features, 'ring')
            and self._is_folded(features, 'pinky')
            and self._is_folded(features, 'thumb', allow_partial=True)
            and open_scores['index'] >= min_open
            and open_scores['middle'] >= min_open
            and open_scores['ring'] <= max_fold
            and open_scores['pinky'] <= max_fold
            and open_scores['thumb'] <= max_thumb
            and index_middle_close
        )
        if not structural_ready:
            return False
        # 'two' removed from the allow-list here too — even with a
        # structurally-valid pose, if the stable label is 'two'
        # we don't activate fresh; the user must produce a real
        # volume_pose label to start.
        return active or pose_score >= 0.08 or stable_gesture in {'volume_pose', 'finger_together', 'neutral'}

    def _is_pinky_hold_pose(self, features) -> bool:
        open_scores = features.open_scores
        spread_states = getattr(features, 'spread_states', {})
        together_strength = float(features.spread_together_strengths.get('index_middle', 0.0))
        return (
            features.finger_count_open == 3
            and self._is_volume_primary_open(features, 'index')
            and self._is_volume_primary_open(features, 'middle')
            and self._is_folded(features, 'ring')
            and self._is_folded(features, 'thumb', allow_partial=True)
            and self._is_fully_open(features, 'pinky')
            and open_scores['index'] >= 0.56
            and open_scores['middle'] >= 0.56
            and open_scores['ring'] <= 0.56
            and open_scores['thumb'] <= 0.58
            and open_scores['pinky'] >= 0.72
            and spread_states.get('index_middle') == 'together'
            and together_strength >= 0.78
        )

    def _fine_state(self, features, finger_name: str) -> str | None:
        fine_states = getattr(features, 'fine_states', None)
        if fine_states is None:
            return None
        return fine_states.get(finger_name)

    def _is_fully_open(self, features, finger_name: str) -> bool:
        fine_state = self._fine_state(features, finger_name)
        if fine_state is not None:
            return fine_state == 'fully_open'
        return features.states[finger_name] == 'open'

    def _is_volume_primary_open(self, features, finger_name: str) -> bool:
        fine_state = self._fine_state(features, finger_name)
        openness = float(features.open_scores.get(finger_name, 0.0))
        if fine_state is not None:
            return fine_state == 'fully_open' or (fine_state == 'partially_curled' and openness >= 0.56)
        return features.states[finger_name] == 'open' and openness >= 0.56

    def _is_folded(self, features, finger_name: str, *, allow_partial: bool = False) -> bool:
        fine_state = self._fine_state(features, finger_name)
        if fine_state is not None:
            allowed = {'mostly_curled', 'closed'}
            if allow_partial:
                allowed.add('partially_curled')
            return fine_state in allowed
        return features.states[finger_name] != 'open'

    def _snapshot(self, *, trigger_mute_toggle: bool) -> VolumeGestureUpdate:
        return VolumeGestureUpdate(
            active=self._active,
            level=self._level,
            muted=self._muted,
            message=self._message,
            status=self._status,
            overlay_visible=self._active,
            trigger_mute_toggle=trigger_mute_toggle,
        )

# Author: Konstantin Markov

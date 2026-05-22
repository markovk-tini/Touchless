from __future__ import annotations

from dataclasses import dataclass

from .youtube_controller import YouTubeController


@dataclass(frozen=True)
class YouTubeGestureSnapshot:
    control_text: str
    info_text: str
    last_action: str
    mode_active: bool
    forced_mode: bool
    consume_other_routes: bool
    action_counter: int = 0


class YouTubeGestureRouter:
    """Routes swipes / fist to YouTube media keys.

    Activation is explicit only: user toggles the mode on via the 'four' static
    gesture (held). There is no auto-activation based on audio/tab state —
    auto-activation used to probe Chrome's audio session via pycaw on the hot
    loop, which triggered a Razer BlackShark + Genshin driver stall (see
    project_audio_ducking memory). Forced mode consumes swipes/fist
    unconditionally and beats both Chrome and Spotify.
    """

    _CONSUMABLE_DYNAMIC = {"swipe_left", "swipe_right"}
    # thumb_up / thumb_down drive like / dislike while forced YouTube
    # mode is active. Plain three is reserved for open_chrome and is
    # NOT consumed here — skip-ad as a gesture is replaced by the
    # auto-skip-ads background timer in the Settings dialog. mute /
    # wheel_pose are intentionally NOT consumed either — they keep
    # their global handlers (system mute, gesture wheel) so volume
    # control still works while watching YouTube.
    _CONSUMABLE_STATIC = {"fist", "thumb_up", "thumb_down"}
    # Only four_together activates YT mode now — plain "four" is
    # reserved for the open_touchless action (handled in the engine,
    # not here). Without this restriction, plain four would race the
    # YT toggle hold-timer against open_touchless's hold-fire.
    _TOGGLE_LABELS = {"four_together"}

    def __init__(
        self,
        *,
        static_hold_seconds: float = 0.5,
        static_cooldown_seconds: float = 1.5,
        dynamic_cooldown_seconds: float = 0.9,
        toggle_hold_seconds: float = 0.7,
        toggle_cooldown_seconds: float = 1.5,
    ) -> None:
        self.static_hold_seconds = float(static_hold_seconds)
        self.static_cooldown_seconds = float(static_cooldown_seconds)
        self.dynamic_cooldown_seconds = float(dynamic_cooldown_seconds)
        self.toggle_hold_seconds = float(toggle_hold_seconds)
        self.toggle_cooldown_seconds = float(toggle_cooldown_seconds)
        self.reset()

    def reset(self) -> None:
        self._forced_mode = False
        self._static_candidate = "neutral"
        self._static_candidate_since = 0.0
        self._static_cooldown_until = 0.0
        self._static_latched_label: str | None = None
        self._dynamic_cooldown_until = 0.0
        self._dynamic_latched_label: str | None = None
        self._toggle_candidate_since: float | None = None
        self._toggle_cooldown_until = 0.0
        self._toggle_latched = False
        self._control_text = "youtube idle"
        self._info_text = "off"
        self._last_action = "-"
        self._mode_active = False
        self._consume_other_routes = False
        self._action_counter = 0

    def _set_action(self, label: str) -> None:
        self._last_action = label
        self._action_counter += 1

    @property
    def forced_mode(self) -> bool:
        return self._forced_mode

    def snapshot(self) -> YouTubeGestureSnapshot:
        return YouTubeGestureSnapshot(
            control_text=self._control_text,
            info_text=self._info_text,
            last_action=self._last_action,
            mode_active=self._mode_active,
            forced_mode=self._forced_mode,
            consume_other_routes=self._consume_other_routes,
            action_counter=self._action_counter,
        )

    def update(
        self,
        *,
        stable_label: str,
        dynamic_label: str,
        controller: YouTubeController,
        now: float,
    ) -> YouTubeGestureSnapshot:
        tab_present = False
        try:
            tab_present = bool(controller.has_youtube_tab())
        except Exception:
            tab_present = False
        if self._forced_mode and not tab_present:
            self._forced_mode = False
            self._control_text = "youtube mode off (tab closed)"
            self._set_action("youtube_mode_off")
        self._update_forced_toggle(stable_label, now, controller)
        self._mode_active = self._forced_mode
        self._info_text = "forced" if self._forced_mode else "off"

        consumes_static = stable_label in self._CONSUMABLE_STATIC
        consumes_dynamic = dynamic_label in self._CONSUMABLE_DYNAMIC
        self._consume_other_routes = self._mode_active and (consumes_static or consumes_dynamic)

        if not self._mode_active:
            if self._dynamic_latched_label is not None and dynamic_label == "neutral":
                self._dynamic_latched_label = None
            if self._static_latched_label is not None and stable_label == "neutral":
                self._static_latched_label = None
            return self.snapshot()

        self._update_dynamic(dynamic_label, stable_label, controller, now)
        self._update_static(stable_label, dynamic_label, controller, now)
        return self.snapshot()

    def _update_forced_toggle(self, stable_label: str, now: float, controller: YouTubeController) -> None:
        if stable_label not in self._TOGGLE_LABELS:
            self._toggle_candidate_since = None
            self._toggle_latched = False
            return
        if self._toggle_latched:
            return
        if self._toggle_candidate_since is None:
            self._toggle_candidate_since = now
            return
        if now < self._toggle_cooldown_until:
            return
        if now - self._toggle_candidate_since < self.toggle_hold_seconds:
            return

        if self._forced_mode:
            self._toggle_latched = True
            self._toggle_cooldown_until = now + self.toggle_cooldown_seconds
            self._forced_mode = False
            self._control_text = "youtube mode off"
            self._set_action("youtube_mode_off")
            return

        has_tab = False
        try:
            has_tab = bool(controller.has_youtube_tab())
        except Exception:
            has_tab = False
        if not has_tab:
            self._control_text = "youtube mode unavailable (no tab)"
            return

        self._toggle_latched = True
        self._toggle_cooldown_until = now + self.toggle_cooldown_seconds
        self._forced_mode = True
        self._control_text = "youtube mode on"
        self._set_action("youtube_mode_on")

    def force_on(self, *, now: float, controller: YouTubeController) -> bool:
        # Tutorial-only entry. Skips the 'four'-hold timer and flips
        # straight into forced mode after the user successfully opens
        # YouTube via the voice-command step. Returns True iff the
        # tab is still present at activation time (no point forcing
        # a mode the engine will immediately disable on the next
        # tick when has_youtube_tab() returns False).
        if self._forced_mode:
            return True
        has_tab = False
        try:
            has_tab = bool(controller.has_youtube_tab())
        except Exception:
            has_tab = False
        if not has_tab:
            return False
        self._forced_mode = True
        self._control_text = "youtube mode on"
        self._info_text = "forced"
        self._set_action("youtube_mode_on")
        # Suppress the next user-side toggle so a residual 'four'
        # pose at activation time doesn't immediately flip it back
        # off via the regular hold timer.
        self._toggle_latched = True
        self._toggle_cooldown_until = now + self.toggle_cooldown_seconds
        self._toggle_candidate_since = None
        return True

    def _update_static(self, stable_label: str, dynamic_label: str, controller: YouTubeController, now: float) -> None:
        actionable = self._CONSUMABLE_STATIC
        if dynamic_label == "repeat_circle" and stable_label == "one":
            self._static_candidate = "neutral"
            self._static_candidate_since = 0.0
            return
        if stable_label == self._static_latched_label:
            if stable_label not in actionable:
                self._static_latched_label = None
            return

        if stable_label not in actionable:
            self._static_candidate = "neutral"
            self._static_candidate_since = 0.0
            if self._static_latched_label is not None:
                self._static_latched_label = None
            return

        if stable_label != self._static_candidate:
            self._static_candidate = stable_label
            self._static_candidate_since = now
            return

        if now < self._static_cooldown_until:
            return
        if now - self._static_candidate_since < self.static_hold_seconds:
            return

        self._static_cooldown_until = now + self.static_cooldown_seconds
        self._static_latched_label = stable_label
        if stable_label == "fist":
            ok = controller.toggle_playback()
            self._control_text = controller.message
            self._set_action("youtube_toggle" if ok else "youtube_toggle_failed")
        elif stable_label == "thumb_up":
            ok = controller.like_video()
            self._control_text = controller.message
            self._set_action("youtube_like" if ok else "youtube_like_failed")
        elif stable_label == "thumb_down":
            ok = controller.dislike_video()
            self._control_text = controller.message
            self._set_action("youtube_dislike" if ok else "youtube_dislike_failed")

    def _update_dynamic(self, dynamic_label: str, stable_label: str, controller: YouTubeController, now: float) -> None:
        actionable = self._CONSUMABLE_DYNAMIC
        if dynamic_label == self._dynamic_latched_label:
            if dynamic_label == "neutral":
                self._dynamic_latched_label = None
            return

        if dynamic_label not in actionable:
            if dynamic_label == "neutral":
                self._dynamic_latched_label = None
            return

        if now < self._dynamic_cooldown_until:
            return

        self._dynamic_cooldown_until = now + self.dynamic_cooldown_seconds
        self._dynamic_latched_label = dynamic_label
        seek_mode = stable_label == "two"
        if dynamic_label == "swipe_left":
            if seek_mode:
                ok = controller.seek_backward()
                self._set_action("youtube_seek_back" if ok else "youtube_seek_back_failed")
            else:
                ok = controller.previous_track()
                self._set_action("youtube_previous" if ok else "youtube_previous_failed")
        else:
            if seek_mode:
                ok = controller.seek_forward()
                self._set_action("youtube_seek_forward" if ok else "youtube_seek_forward_failed")
            else:
                ok = controller.next_track()
                self._set_action("youtube_next" if ok else "youtube_next_failed")
        self._control_text = controller.message

# Author: Konstantin Markov

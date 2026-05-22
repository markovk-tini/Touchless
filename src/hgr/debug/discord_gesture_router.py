"""Routes mode-scoped gestures to the Discord RPC controller.

Mode model
----------
Left-hand `three` held for `toggle_hold_seconds` toggles Discord mode
on/off. While Discord mode is active:

  * Right-hand `three` → toggle self-mute
  * Right-hand `four`  → toggle self-deafen
  * Mouse-control mode is suppressed (engine gates `_handle_mouse_control`
    on `snapshot.suppress_mouse`). The user is presumed to be on a voice
    call and wants free hand movement for mute/deafen without
    accidentally driving the cursor.

What still works while Discord mode is active
---------------------------------------------
Everything that's NOT mouse-control or NOT one of the two consumed
right-hand poses: Spotify gestures (right `two`, `fist`, `ok`, swipes),
volume gestures, left-hand voice activation (`one` + `fist` cancel),
the gesture wheels, etc. The router only consumes its own pose set —
no global side effects beyond the mouse suppression.

Why left hand for the activator (vs right hand like YouTube)
------------------------------------------------------------
YouTube mode toggles on RIGHT-hand `four` because the user is sitting
in front of YouTube one-handed; the toggle pose and the action poses
share the same hand and just take turns. Discord users are typically
on a voice call AND using their PC, so the right hand is doing other
things (Spotify control, volume, etc). Left-hand activator means the
right hand can do its normal job and the user only needs to bring the
left hand up to enter Discord mode. Same logic as how Touchless's
voice command listener uses left-hand `one`.

Co-existence with Spotify
-------------------------
Spotify gestures consume `two`, `fist`, `ok` (right hand). Discord
gestures consume `three`, `four` (right hand) ONLY while Discord
mode is on. Zero overlap — both routers can fire on the same frame
without stepping on each other.
"""

from __future__ import annotations

from dataclasses import dataclass

from .discord_controller import DiscordController


@dataclass(frozen=True)
class DiscordGestureSnapshot:
    control_text: str
    info_text: str
    last_action: str
    mode_active: bool
    consume_other_routes: bool
    suppress_mouse: bool
    action_counter: int = 0


class DiscordGestureRouter:
    """See module docstring for the mode model + gesture vocabulary."""

    # Left-hand poses that fire the mode toggle when held. The
    # synthetic label `three_mrp` is emitted by the engine when the
    # left hand is showing the custom Middle+Ring+Pinky geometry
    # (index folded, thumb folded). The base recogniser doesn't
    # distinguish which 3 fingers are up — it just labels any three-
    # finger pose as plain "three" — so an engine-side geometry check
    # in `_is_left_three_mrp` is what makes "three_mrp" appear here.
    # Using a distinct label keeps the Discord activator fully
    # disjoint from mouse-mode's standard left-`three` toggle.
    _ACTIVATOR_LABELS = {"three_mrp"}
    # Right-hand poses that fire Discord actions while mode is on.
    # `fist` overlaps with Spotify's play/pause toggle — when Discord
    # mode is active, fist consumes the gesture for Discord (via
    # `consume_other_routes` in the snapshot) so Spotify doesn't also
    # receive it. When Discord mode is OFF, Spotify owns fist
    # unchanged. Same pattern YouTube mode uses to hijack fist for
    # YouTube play/pause while forced.
    _ACTIONABLE = {"three", "four", "fist"}

    def __init__(
        self,
        *,
        static_hold_seconds: float = 0.5,
        static_cooldown_seconds: float = 1.5,
        toggle_hold_seconds: float = 0.7,
        toggle_cooldown_seconds: float = 1.5,
    ) -> None:
        self.static_hold_seconds = float(static_hold_seconds)
        self.static_cooldown_seconds = float(static_cooldown_seconds)
        self.toggle_hold_seconds = float(toggle_hold_seconds)
        self.toggle_cooldown_seconds = float(toggle_cooldown_seconds)
        self.reset()

    def reset(self) -> None:
        self._mode_active = False
        self._toggle_candidate_since: float | None = None
        self._toggle_cooldown_until = 0.0
        self._toggle_latched = False
        self._static_candidate = "neutral"
        self._static_candidate_since = 0.0
        self._static_cooldown_until = 0.0
        self._static_latched_label: str | None = None
        self._control_text = "discord idle"
        self._info_text = "off"
        self._last_action = "-"
        self._action_counter = 0
        self._consume_other_routes = False

    def _set_action(self, label: str) -> None:
        self._last_action = label
        self._action_counter += 1

    @property
    def mode_active(self) -> bool:
        return self._mode_active

    def snapshot(self) -> DiscordGestureSnapshot:
        return DiscordGestureSnapshot(
            control_text=self._control_text,
            info_text=self._info_text,
            last_action=self._last_action,
            mode_active=self._mode_active,
            consume_other_routes=self._consume_other_routes,
            suppress_mouse=self._mode_active,
            action_counter=self._action_counter,
        )

    def update(
        self,
        *,
        stable_label: str,
        left_stable_label: str,
        controller: DiscordController,
        now: float,
    ) -> DiscordGestureSnapshot:
        self._update_toggle(left_stable_label, now)
        if self._mode_active:
            self._update_static(stable_label, controller, now)
        else:
            # Drop any stale static-pose latches so re-entering the
            # mode starts from a clean state.
            if self._static_latched_label is not None and stable_label == "neutral":
                self._static_latched_label = None
        # consume_other_routes is True only when an ACTION pose is
        # currently being made while the mode is active. That way
        # right-3 / right-4 don't leak to other routers, but a neutral
        # right hand doesn't block Spotify/volume/etc from running
        # normally on the same frame.
        self._consume_other_routes = (
            self._mode_active and stable_label in self._ACTIONABLE
        )
        return self.snapshot()

    def _update_toggle(self, left_stable_label: str, now: float) -> None:
        """Left-hand activator. Same hold-then-fire mechanic as
        YouTube's `_update_forced_toggle` but reads the LEFT hand
        instead of the right (Discord activator is left, action
        poses are right)."""
        if left_stable_label not in self._ACTIVATOR_LABELS:
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

        self._toggle_latched = True
        self._toggle_cooldown_until = now + self.toggle_cooldown_seconds
        self._mode_active = not self._mode_active
        if self._mode_active:
            self._control_text = "discord mode on"
            self._info_text = "on"
            self._set_action("discord_mode_on")
        else:
            self._control_text = "discord mode off"
            self._info_text = "off"
            self._set_action("discord_mode_off")
            # Drop pose latches so a leftover right-3 doesn't fire
            # on the next mode-on.
            self._static_candidate = "neutral"
            self._static_candidate_since = 0.0
            self._static_latched_label = None

    def _update_static(
        self, stable_label: str, controller: DiscordController, now: float
    ) -> None:
        """Right-hand action handler. Same hold-then-fire mechanic
        as the Spotify/YouTube routers — pose must be held stable
        through `static_hold_seconds` and outside the cooldown."""
        if stable_label == self._static_latched_label:
            if stable_label not in self._ACTIONABLE:
                self._static_latched_label = None
            return
        if stable_label not in self._ACTIONABLE:
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
        if stable_label == "three":
            ok = controller.toggle_self_mute()
            self._control_text = controller.message
            self._set_action("discord_mute" if ok else "discord_mute_failed")
        elif stable_label == "four":
            ok = controller.toggle_self_deafen()
            self._control_text = controller.message
            self._set_action("discord_deafen" if ok else "discord_deafen_failed")
        elif stable_label == "fist":
            # Right fist while Discord mode is on = leave the current
            # voice channel. Overrides Spotify's fist=play/pause via
            # the consume_other_routes flag; Spotify still owns fist
            # when Discord mode is off.
            ok = controller.leave_voice_channel()
            self._control_text = controller.message
            self._set_action("discord_leave_call" if ok else "discord_leave_call_failed")

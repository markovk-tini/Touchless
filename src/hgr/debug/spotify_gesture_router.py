from __future__ import annotations

from dataclasses import dataclass

from .spotify_controller import SpotifyController


@dataclass(frozen=True)
class SpotifyGestureSnapshot:
    control_text: str
    info_text: str
    last_action: str
    action_counter: int = 0


class SpotifyGestureRouter:
    def __init__(
        self,
        *,
        static_hold_seconds: float = 1.0,
        static_cooldown_seconds: float = 1.5,
        dynamic_cooldown_seconds: float = 0.9,
    ) -> None:
        self.static_hold_seconds = float(static_hold_seconds)
        self.static_cooldown_seconds = float(static_cooldown_seconds)
        self.dynamic_cooldown_seconds = float(dynamic_cooldown_seconds)
        self.reset()

    def reset(self) -> None:
        self._static_candidate = "neutral"
        self._static_candidate_since = 0.0
        self._static_cooldown_until = 0.0
        self._static_latched_label: str | None = None
        self._dynamic_cooldown_until = 0.0
        self._dynamic_latched_label: str | None = None
        self._control_text = "spotify idle"
        self._info_text = "-"
        self._last_action = "-"
        self._action_counter = 0

    def _set_action(self, label: str) -> None:
        self._last_action = label
        self._action_counter += 1

    def _spotify_result_callback(self, verb: str):
        """Return a callback for `controller.dispatch_async` that
        overwrites the optimistic control_text with the REAL result
        message from the async HTTP call.

        v1.1.7 tester bug: the router used to set _control_text to
        e.g. "spotify play/pause" BEFORE the HTTP call ran; when the
        call failed (Premium user with no active device, wrong
        Spotify account, transfer 202, etc.), the wheel silently
        confirmed as if the action worked. This callback is fired
        AFTER the request settles. On failure, we replace the
        optimistic text with the controller's real message (which
        now routes through `_format_error_message` so the user sees
        e.g. "spotify play failed — no active Spotify device. Open
        the Spotify app on your PC or phone…"). We also bump
        _action_counter so noop_engine re-emits command_detected
        with the corrected text.
        """
        def _on_complete(result: bool, message: str) -> None:
            if result:
                # Success — leave the optimistic text alone. Nothing
                # to update; the user already saw the right label.
                return
            # Failure — overwrite with the real controller message.
            self._control_text = (
                message
                if message
                else f"spotify {verb} failed"
            )
            self._last_action = f"spotify_{verb.replace('/', '_').replace(' ', '_')}_failed"
            self._action_counter += 1
        return _on_complete

    def snapshot(self) -> SpotifyGestureSnapshot:
        return SpotifyGestureSnapshot(
            control_text=self._control_text,
            info_text=self._info_text,
            last_action=self._last_action,
            action_counter=self._action_counter,
        )

    def update(
        self,
        *,
        stable_label: str,
        dynamic_label: str,
        controller: SpotifyController,
        now: float,
    ) -> SpotifyGestureSnapshot:
        self._update_dynamic(dynamic_label, controller, now)
        self._update_static(stable_label, dynamic_label, controller, now)
        return self.snapshot()

    def _update_static(self, stable_label: str, dynamic_label: str, controller: SpotifyController, now: float) -> None:
        actionable = {"two", "fist", "ok"}
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
        required_hold = self.static_hold_seconds
        if now - self._static_candidate_since < required_hold:
            return

        self._static_cooldown_until = now + self.static_cooldown_seconds
        self._static_latched_label = stable_label
        if stable_label == "two":
            self._note_command_attempt(controller)
            self._control_text = "opening spotify"
            self._set_action("spotify_focus")
            controller.dispatch_async(
                controller.focus_or_open_window,
                on_complete=self._spotify_result_callback("focus"),
            )
        elif stable_label == "fist":
            if not self._can_control_without_focus(controller):
                self._control_text = "spotify inactive on device"
                self._set_action("spotify_toggle_idle")
                return
            self._note_command_attempt(controller)
            # Fire HTTP call on a background thread so the gesture
            # worker doesn't block on the 50-300 ms Spotify Web API
            # roundtrip. Action label is set OPTIMISTICALLY; the
            # `on_complete` callback overwrites _control_text with
            # the real failure message if the HTTP call errors.
            controller.dispatch_async(
                controller.toggle_playback,
                on_complete=self._spotify_result_callback("play/pause"),
            )
            self._control_text = "spotify play/pause"
            self._set_action("spotify_toggle")
        elif stable_label == "ok":
            if not self._can_control_without_focus(controller):
                self._control_text = "spotify inactive on device"
                self._set_action("spotify_shuffle_idle")
                return
            self._note_command_attempt(controller)
            controller.dispatch_async(
                controller.toggle_shuffle,
                on_complete=self._spotify_result_callback("shuffle"),
            )
            self._control_text = "spotify shuffle"
            self._set_action("spotify_shuffle")

    def _update_dynamic(self, dynamic_label: str, controller: SpotifyController, now: float) -> None:
        actionable = {"swipe_left", "swipe_right", "repeat_circle"}
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
        if not self._can_control_without_focus(controller):
            self._dynamic_cooldown_until = now + self.dynamic_cooldown_seconds
            self._dynamic_latched_label = dynamic_label
            self._control_text = "spotify inactive on device"
            if dynamic_label == "swipe_left":
                self._set_action("spotify_previous_idle")
            elif dynamic_label == "swipe_right":
                self._set_action("spotify_next_idle")
            else:
                self._set_action("spotify_repeat_idle")
            return

        self._note_command_attempt(controller)
        self._dynamic_cooldown_until = now + self.dynamic_cooldown_seconds
        self._dynamic_latched_label = dynamic_label
        # Fire HTTP/AppleScript on a background thread — dynamic
        # gestures used to spike the gesture worker during the
        # roundtrip. Cooldowns + latching above ensure we don't
        # double-fire while a dispatch is in flight.
        if dynamic_label == "swipe_left":
            controller.dispatch_async(
                controller.previous_track,
                on_complete=self._spotify_result_callback("previous"),
            )
            self._control_text = "spotify previous track"
            self._set_action("spotify_previous")
        elif dynamic_label == "swipe_right":
            controller.dispatch_async(
                controller.next_track,
                on_complete=self._spotify_result_callback("next"),
            )
            self._control_text = "spotify next track"
            self._set_action("spotify_next")
        else:
            controller.dispatch_async(
                controller.toggle_repeat_track,
                on_complete=self._spotify_result_callback("repeat"),
            )
            self._control_text = "spotify repeat toggle"
            self._set_action("spotify_repeat")

    @staticmethod
    def _note_command_attempt(controller: SpotifyController) -> None:
        # Signal that the user actually dispatched a Spotify control.
        # Idle fist/ok/swipe (Spotify closed) must NOT flip this —
        # MainWindow's connect toast keys off it.
        try:
            controller.record_command_attempt()
        except Exception:
            pass

    def _can_control_without_focus(self, controller: SpotifyController) -> bool:
        # Stricter than the old is_running() catch-all: a Spotify
        # protocol handler / helper process leaves is_running() True
        # even when there's no real Spotify to control, so fist
        # (toggle play/pause) used to attempt action and silently
        # auto-launch the app via play() → ensure_ready(open_if_needed=True).
        # Now we require either an active Web API device OR an
        # actual Spotify window. Right-hand 'two' and the voice
        # 'open spotify' command remain the ONLY paths that may
        # launch Spotify when it isn't running.
        #
        # r51 diag: log every branch decision so the next log tells
        # us EXACTLY which check let a gesture through when user
        # reported Spotify was fully closed. Small overhead — one
        # stderr write per gesture commit at most.
        import sys as _sys
        try:
            # Mac: AppleScript "Spotify is running", not helper
            # processes. Closed app must fail this gate so swipe/fist
            # stay silent and do not arm the connect overlay.
            if getattr(controller, "_mac", False) and bool(controller.is_running()):
                try:
                    _sys.stderr.write("[r51-spotify-gate] pass: mac is_running=True\n")
                    _sys.stderr.flush()
                except Exception:
                    pass
                return True
        except Exception:
            pass
        active = False
        try:
            active = controller.is_active_device_available()
        except Exception:
            pass
        if active:
            try:
                _sys.stderr.write("[r51-spotify-gate] pass: is_active_device_available=True\n")
                _sys.stderr.flush()
            except Exception:
                pass
            return True
        # r50: hard block for the "no live device AND no real Spotify"
        # case. Dad reported skip/pause/swipe silently auto-launching
        # Spotify. Root cause was is_window_open() / is_window_active()
        # returning True for phantom windows created by Spotify's
        # update-handler / protocol-handler helper processes even when
        # the interactive Spotify.exe wasn't running. Adding this
        # early-False when no Web API device AND no real Spotify.exe
        # process short-circuits before those phantom-window paths.
        # _has_real_spotify_process filters helper processes by
        # requiring the executable path + a >1 MB image size — see
        # spotify_controller.py. It carries a ~1 s TTL cache so this
        # extra call does not spike gesture-commit latency when the
        # real Spotify is running with its 10-20 helper procs.
        try:
            if not controller._has_real_spotify_process():
                try:
                    import time as _time
                    last = float(getattr(controller, "_r51_gate_log_at", 0.0) or 0.0)
                    now = _time.monotonic()
                    if now - last >= 2.0:
                        controller._r51_gate_log_at = now
                        _sys.stderr.write("[r51-spotify-gate] block: is_active=False AND _has_real_spotify_process=False\n")
                        _sys.stderr.flush()
                except Exception:
                    pass
                return False
        except Exception:
            # Absent method on older controller stubs → fall through to
            # the existing window checks (previous behavior).
            pass
        is_window_open = getattr(controller, "is_window_open", None)
        if callable(is_window_open) and is_window_open():
            try:
                _sys.stderr.write("[r51-spotify-gate] pass: is_window_open=True (real process present)\n")
                _sys.stderr.flush()
            except Exception:
                pass
            return True
        if controller.is_window_active():
            try:
                _sys.stderr.write("[r51-spotify-gate] pass: is_window_active=True (real process present)\n")
                _sys.stderr.flush()
            except Exception:
                pass
            return True
        try:
            _sys.stderr.write("[r51-spotify-gate] block: fell through all checks\n")
            _sys.stderr.flush()
        except Exception:
            pass
        return False

# Author: Konstantin Markov

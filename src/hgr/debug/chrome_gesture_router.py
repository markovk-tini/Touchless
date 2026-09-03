from __future__ import annotations

import threading
from dataclasses import dataclass

from .chrome_controller import ChromeController


@dataclass(frozen=True)
class ChromeGestureSnapshot:
    control_text: str
    info_text: str
    last_action: str
    mode_enabled: bool
    consume_other_routes: bool
    action_counter: int = 0


class ChromeGestureRouter:
    def __init__(
        self,
        *,
        static_hold_seconds: float = 1.0,
        static_cooldown_seconds: float = 1.5,
        dynamic_cooldown_seconds: float = 1.5,
    ) -> None:
        self.static_hold_seconds = float(static_hold_seconds)
        self.static_cooldown_seconds = float(static_cooldown_seconds)
        self.dynamic_cooldown_seconds = float(dynamic_cooldown_seconds)
        self.reset()

    def reset(self) -> None:
        self._mode_enabled = False
        self._static_candidate = "neutral"
        self._static_candidate_since = 0.0
        self._static_cooldown_until = 0.0
        self._static_latched_label: str | None = None
        self._dynamic_cooldown_until = 0.0
        self._dynamic_latched_label: str | None = None
        self._control_text = "chrome idle"
        self._info_text = "mode off"
        self._last_action = "-"
        self._consume_other_routes = False
        self._action_counter = 0

    def _set_action(self, label: str) -> None:
        self._last_action = label
        self._action_counter += 1

    def snapshot(self) -> ChromeGestureSnapshot:
        return ChromeGestureSnapshot(
            control_text=self._control_text,
            info_text=self._info_text,
            last_action=self._last_action,
            mode_enabled=self._mode_enabled,
            consume_other_routes=self._consume_other_routes,
            action_counter=self._action_counter,
        )

    def update(
        self,
        *,
        stable_label: str,
        dynamic_label: str,
        controller: ChromeController,
        now: float,
    ) -> ChromeGestureSnapshot:
        chrome_open = self._is_chrome_open(controller)
        if self._mode_enabled and not chrome_open:
            self._mode_enabled = False
            self._control_text = "chrome closed - mode off"
            self._set_action("chrome_mode_closed")
            self._dynamic_latched_label = None
        self._consume_other_routes = self._mode_enabled
        self._update_dynamic(dynamic_label, controller, now)
        self._update_static(stable_label, controller, now, chrome_open=chrome_open)
        self._info_text = "mode on" if self._mode_enabled else "mode off"
        return self.snapshot()

    def _update_static(self, stable_label: str, controller: ChromeController, now: float, *, chrome_open: bool) -> None:
        actionable = {"three", "three_together", "four", "four_together"}
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

        if stable_label == "three":
            self._control_text = "opening chrome"
            self._set_action("chrome_focus")
            self._consume_other_routes = True
            dispatch = getattr(controller, "dispatch_async", None)
            if callable(dispatch):
                dispatch(controller.focus_or_open_window)
            else:
                threading.Thread(
                    target=controller.focus_or_open_window,
                    name="chrome-focus",
                    daemon=True,
                ).start()
            return

        if stable_label == "three_together":
            if not self._mode_enabled:
                if chrome_open:
                    self._mode_enabled = True
                    self._control_text = "chrome mode on"
                    self._set_action("chrome_mode_on")
                else:
                    self._control_text = "chrome must be open for mode"
                    self._set_action("chrome_mode_requires_open")
            else:
                self._mode_enabled = False
                self._control_text = "chrome mode off"
                self._set_action("chrome_mode_off")
            self._consume_other_routes = True
            return

        if not self._mode_enabled:
            return

        if stable_label == "four":
            success = controller.new_tab()
            self._control_text = controller.message
            self._set_action("chrome_new_tab" if success else "chrome_new_tab_failed")
            self._consume_other_routes = True
        elif stable_label == "four_together":
            success = controller.new_incognito_tab()
            self._control_text = controller.message
            self._set_action("chrome_new_incognito" if success else "chrome_new_incognito_failed")
            self._consume_other_routes = True

    def _update_dynamic(self, dynamic_label: str, controller: ChromeController, now: float) -> None:
        actionable = {"swipe_left", "swipe_right", "repeat_circle"}
        if dynamic_label == self._dynamic_latched_label:
            if dynamic_label == "neutral":
                self._dynamic_latched_label = None
            return

        if dynamic_label not in actionable:
            if dynamic_label == "neutral":
                self._dynamic_latched_label = None
            return

        if not self._mode_enabled:
            return
        if now < self._dynamic_cooldown_until:
            return

        self._dynamic_cooldown_until = now + self.dynamic_cooldown_seconds
        self._dynamic_latched_label = dynamic_label
        self._consume_other_routes = True
        if dynamic_label == "swipe_left":
            success = controller.navigate_back()
            self._set_action("chrome_back" if success else "chrome_back_failed")
        elif dynamic_label == "swipe_right":
            success = controller.navigate_forward()
            self._set_action("chrome_forward" if success else "chrome_forward_failed")
        else:
            success = controller.refresh_page()
            self._set_action("chrome_refresh" if success else "chrome_refresh_failed")
        self._control_text = controller.message

    def _is_chrome_open(self, controller: ChromeController) -> bool:
        if hasattr(controller, "is_window_open"):
            try:
                return bool(controller.is_window_open())
            except Exception:
                return False
        if hasattr(controller, "is_window_active"):
            try:
                if controller.is_window_active():
                    return True
            except Exception:
                return False
        if hasattr(controller, "is_running"):
            try:
                return bool(controller.is_running())
            except Exception:
                return False
        return False

# Author: Konstantin Markov

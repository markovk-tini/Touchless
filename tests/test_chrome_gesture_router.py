from __future__ import annotations

import unittest

from hgr.debug.chrome_gesture_router import ChromeGestureRouter


class _FakeChromeController:
    def __init__(self) -> None:
        self.message = "chrome idle"
        self.focus_calls = 0
        self.back_calls = 0
        self.forward_calls = 0
        self.refresh_calls = 0
        self.new_tab_calls = 0
        self.new_incognito_calls = 0
        self._window_open = True

    def dispatch_async(self, callable_obj, *args, on_complete=None, **kwargs) -> None:
        result = callable_obj(*args, **kwargs)
        if on_complete is not None:
            on_complete(bool(result), str(self.message or ""))

    def focus_or_open_window(self) -> bool:
        self.focus_calls += 1
        self.message = "chrome focused"
        self._window_open = True
        return True

    def is_window_open(self) -> bool:
        return self._window_open

    def navigate_back(self) -> bool:
        self.back_calls += 1
        self.message = "chrome back"
        return True

    def navigate_forward(self) -> bool:
        self.forward_calls += 1
        self.message = "chrome forward"
        return True

    def refresh_page(self) -> bool:
        self.refresh_calls += 1
        self.message = "chrome refresh"
        return True

    def new_tab(self) -> bool:
        self.new_tab_calls += 1
        self.message = "chrome new tab"
        return True

    def new_incognito_tab(self) -> bool:
        self.new_incognito_calls += 1
        self.message = "chrome new incognito tab"
        return True


class ChromeGestureRouterTest(unittest.TestCase):
    def test_three_focuses_chrome_after_hold(self) -> None:
        router = ChromeGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=1.5)
        controller = _FakeChromeController()

        router.update(stable_label="three", dynamic_label="neutral", controller=controller, now=0.0)
        snapshot = router.update(stable_label="three", dynamic_label="neutral", controller=controller, now=0.6)

        self.assertEqual(controller.focus_calls, 1)
        self.assertEqual(snapshot.last_action, "chrome_focus")
        self.assertTrue(snapshot.consume_other_routes)

    def test_three_together_toggles_mode_on_and_off(self) -> None:
        router = ChromeGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=1.5)
        controller = _FakeChromeController()

        router.update(stable_label="three_together", dynamic_label="neutral", controller=controller, now=0.0)
        on_snapshot = router.update(stable_label="three_together", dynamic_label="neutral", controller=controller, now=0.6)
        self.assertTrue(on_snapshot.mode_enabled)
        self.assertEqual(on_snapshot.last_action, "chrome_mode_on")

        router.update(stable_label="neutral", dynamic_label="neutral", controller=controller, now=2.2)
        router.update(stable_label="three_together", dynamic_label="neutral", controller=controller, now=2.3)
        off_snapshot = router.update(stable_label="three_together", dynamic_label="neutral", controller=controller, now=2.9)
        self.assertFalse(off_snapshot.mode_enabled)
        self.assertEqual(off_snapshot.last_action, "chrome_mode_off")

    def test_three_together_requires_chrome_to_already_be_open(self) -> None:
        router = ChromeGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=1.5)
        controller = _FakeChromeController()
        controller._window_open = False

        router.update(stable_label="three_together", dynamic_label="neutral", controller=controller, now=0.0)
        snapshot = router.update(stable_label="three_together", dynamic_label="neutral", controller=controller, now=0.6)

        self.assertFalse(snapshot.mode_enabled)
        self.assertEqual(snapshot.last_action, "chrome_mode_requires_open")
        self.assertEqual(controller.focus_calls, 0)

    def test_mode_routes_swipes_refresh_and_tabs(self) -> None:
        router = ChromeGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=1.5)
        controller = _FakeChromeController()

        router.update(stable_label="three_together", dynamic_label="neutral", controller=controller, now=0.0)
        router.update(stable_label="three_together", dynamic_label="neutral", controller=controller, now=0.6)

        router.update(stable_label="neutral", dynamic_label="swipe_left", controller=controller, now=0.8)
        router.update(stable_label="neutral", dynamic_label="neutral", controller=controller, now=1.0)
        router.update(stable_label="neutral", dynamic_label="swipe_right", controller=controller, now=2.4)
        router.update(stable_label="neutral", dynamic_label="neutral", controller=controller, now=2.6)
        router.update(stable_label="neutral", dynamic_label="repeat_circle", controller=controller, now=4.0)

        router.update(stable_label="neutral", dynamic_label="neutral", controller=controller, now=4.2)
        router.update(stable_label="four", dynamic_label="neutral", controller=controller, now=5.8)
        router.update(stable_label="four", dynamic_label="neutral", controller=controller, now=6.4)

        router.update(stable_label="neutral", dynamic_label="neutral", controller=controller, now=7.4)
        router.update(stable_label="four_together", dynamic_label="neutral", controller=controller, now=8.1)
        router.update(stable_label="four_together", dynamic_label="neutral", controller=controller, now=8.7)

        self.assertEqual(controller.back_calls, 1)
        self.assertEqual(controller.forward_calls, 1)
        self.assertEqual(controller.refresh_calls, 1)
        self.assertEqual(controller.new_tab_calls, 1)
        self.assertEqual(controller.new_incognito_calls, 1)

    def test_dynamic_actions_do_not_fire_when_mode_is_off(self) -> None:
        router = ChromeGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=1.5)
        controller = _FakeChromeController()

        snapshot = router.update(stable_label="neutral", dynamic_label="swipe_left", controller=controller, now=0.0)

        self.assertEqual(controller.back_calls, 0)
        self.assertFalse(snapshot.consume_other_routes)

# Author: Konstantin Markov

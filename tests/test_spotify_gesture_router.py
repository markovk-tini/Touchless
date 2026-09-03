from __future__ import annotations

import unittest

from hgr.debug.spotify_controller import SpotifyTrackDetails
from hgr.debug.spotify_gesture_router import SpotifyGestureRouter


class _FakeSpotifyController:
    def __init__(self) -> None:
        self.message = "spotify idle"
        self.focus_calls = 0
        self.toggle_calls = 0
        self.previous_calls = 0
        self.next_calls = 0
        self.repeat_calls = 0
        self.shuffle_calls = 0
        self.details_calls = 0
        self._active = True
        self._running = False
        self._window_active = False

    def dispatch_async(self, callable_obj, *args, on_complete=None, **kwargs) -> None:
        result = callable_obj(*args, **kwargs)
        if on_complete is not None:
            on_complete(bool(result), str(self.message or ""))

    def focus_or_open_window(self) -> bool:
        self.focus_calls += 1
        self.message = "spotify focused"
        self._window_active = True
        return True

    def is_window_active(self) -> bool:
        return self._window_active

    def is_window_open(self) -> bool:
        return bool(self._running) or self._window_active

    def _has_real_spotify_process(self) -> bool:
        return bool(self._running)

    def is_active_device_available(self) -> bool:
        return self._active

    def is_running(self) -> bool:
        return self._running

    def toggle_playback(self) -> bool:
        self.toggle_calls += 1
        self.message = "spotify toggle"
        return True

    def previous_track(self) -> bool:
        self.previous_calls += 1
        self.message = "spotify previous track"
        return True

    def next_track(self) -> bool:
        self.next_calls += 1
        self.message = "spotify next track"
        return True

    def toggle_repeat_track(self) -> bool:
        self.repeat_calls += 1
        self.message = "spotify repeat track"
        return True

    def toggle_shuffle(self) -> bool:
        self.shuffle_calls += 1
        self.message = "spotify shuffle on"
        return True

    def get_current_track_details(self) -> SpotifyTrackDetails | None:
        self.details_calls += 1
        return SpotifyTrackDetails(
            song_name="Track",
            artist_names="Artist",
            album_name="Album",
            playlist_name="Playlist",
            device_name="Desktop",
            device_type="Computer",
            is_playing=True,
            shuffle_enabled=False,
            repeat_mode="off",
            progress_ms=12000,
            duration_ms=180000,
            context_type="playlist",
        )


class SpotifyGestureRouterTest(unittest.TestCase):
    def test_two_requires_hold_before_ready_action(self) -> None:
        router = SpotifyGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5)
        controller = _FakeSpotifyController()

        router.update(stable_label="two", dynamic_label="neutral", controller=controller, now=0.0)
        router.update(stable_label="two", dynamic_label="neutral", controller=controller, now=0.4)
        self.assertEqual(controller.focus_calls, 0)

        snapshot = router.update(stable_label="two", dynamic_label="neutral", controller=controller, now=0.6)
        self.assertEqual(controller.focus_calls, 1)
        self.assertEqual(snapshot.last_action, "spotify_focus")

    def test_two_does_nothing_when_spotify_is_already_active_window(self) -> None:
        router = SpotifyGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5)
        controller = _FakeSpotifyController()
        controller._window_active = True

        router.update(stable_label="two", dynamic_label="neutral", controller=controller, now=0.0)
        snapshot = router.update(stable_label="two", dynamic_label="neutral", controller=controller, now=1.05)

        self.assertEqual(controller.focus_calls, 1)
        self.assertEqual(snapshot.last_action, "spotify_focus")

    def test_static_actions_latch_and_respect_cooldown(self) -> None:
        router = SpotifyGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5)
        controller = _FakeSpotifyController()

        router.update(stable_label="fist", dynamic_label="neutral", controller=controller, now=0.0)
        router.update(stable_label="fist", dynamic_label="neutral", controller=controller, now=0.6)
        self.assertEqual(controller.toggle_calls, 1)

        router.update(stable_label="fist", dynamic_label="neutral", controller=controller, now=0.9)
        self.assertEqual(controller.toggle_calls, 1)

        router.update(stable_label="neutral", dynamic_label="neutral", controller=controller, now=1.0)
        router.update(stable_label="fist", dynamic_label="neutral", controller=controller, now=1.1)
        router.update(stable_label="fist", dynamic_label="neutral", controller=controller, now=1.7)
        self.assertEqual(controller.toggle_calls, 1)

        router.update(stable_label="neutral", dynamic_label="neutral", controller=controller, now=2.2)
        router.update(stable_label="fist", dynamic_label="neutral", controller=controller, now=2.3)
        router.update(stable_label="fist", dynamic_label="neutral", controller=controller, now=2.9)
        self.assertEqual(controller.toggle_calls, 2)

    def test_swipes_map_to_previous_and_next(self) -> None:
        router = SpotifyGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=0.9)
        controller = _FakeSpotifyController()

        router.update(stable_label="neutral", dynamic_label="swipe_left", controller=controller, now=0.0)
        router.update(stable_label="neutral", dynamic_label="neutral", controller=controller, now=0.2)
        router.update(stable_label="neutral", dynamic_label="swipe_right", controller=controller, now=1.1)

        self.assertEqual(controller.previous_calls, 1)
        self.assertEqual(controller.next_calls, 1)

    def test_repeat_circle_maps_to_repeat_toggle_and_suppresses_static_one(self) -> None:
        router = SpotifyGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=0.9)
        controller = _FakeSpotifyController()

        snapshot = router.update(stable_label="one", dynamic_label="repeat_circle", controller=controller, now=0.8)

        self.assertEqual(controller.repeat_calls, 1)
        self.assertEqual(controller.focus_calls, 0)
        self.assertEqual(snapshot.last_action, "spotify_repeat")

    def test_ok_toggles_shuffle(self) -> None:
        router = SpotifyGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5)
        controller = _FakeSpotifyController()

        router.update(stable_label="ok", dynamic_label="neutral", controller=controller, now=0.0)
        snapshot = router.update(stable_label="ok", dynamic_label="neutral", controller=controller, now=0.6)

        self.assertEqual(controller.shuffle_calls, 1)
        self.assertEqual(snapshot.last_action, "spotify_shuffle")

    def test_non_two_gestures_do_not_wake_inactive_spotify(self) -> None:
        router = SpotifyGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=0.9)
        controller = _FakeSpotifyController()
        controller._active = False

        router.update(stable_label="fist", dynamic_label="neutral", controller=controller, now=0.0)
        fist_snapshot = router.update(stable_label="fist", dynamic_label="neutral", controller=controller, now=0.6)
        swipe_snapshot = router.update(stable_label="neutral", dynamic_label="swipe_left", controller=controller, now=1.8)

        self.assertEqual(controller.toggle_calls, 0)
        self.assertEqual(controller.previous_calls, 0)
        self.assertEqual(fist_snapshot.last_action, "spotify_toggle_idle")
        self.assertEqual(swipe_snapshot.last_action, "spotify_previous_idle")

    def test_running_spotify_can_still_receive_controls_without_focus_gesture(self) -> None:
        router = SpotifyGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=0.9)
        controller = _FakeSpotifyController()
        controller._active = False
        controller._running = True

        router.update(stable_label="fist", dynamic_label="neutral", controller=controller, now=0.0)
        fist_snapshot = router.update(stable_label="fist", dynamic_label="neutral", controller=controller, now=0.6)
        swipe_snapshot = router.update(stable_label="neutral", dynamic_label="swipe_right", controller=controller, now=1.8)

        self.assertEqual(controller.toggle_calls, 1)
        self.assertEqual(controller.next_calls, 1)
        self.assertEqual(fist_snapshot.last_action, "spotify_toggle")
        self.assertEqual(swipe_snapshot.last_action, "spotify_next")

# Author: Konstantin Markov

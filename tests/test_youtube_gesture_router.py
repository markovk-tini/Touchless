from __future__ import annotations

import unittest

from hgr.debug.youtube_gesture_router import YouTubeGestureRouter


class _FakeYouTubeController:
    def __init__(self) -> None:
        self.message = "ok"
        self.like_calls = 0
        self.dislike_calls = 0
        self.toggle_calls = 0

    def has_youtube_tab(self) -> bool:
        return True

    def like_video(self) -> bool:
        self.like_calls += 1
        return True

    def dislike_video(self) -> bool:
        self.dislike_calls += 1
        return True

    def toggle_playback(self) -> bool:
        self.toggle_calls += 1
        self.message = "youtube toggle"
        return True


class YouTubeGestureRouterTest(unittest.TestCase):
    def test_thumb_poses_are_not_consumable(self) -> None:
        self.assertNotIn("thumb_up", YouTubeGestureRouter._CONSUMABLE_STATIC)
        self.assertNotIn("thumb_down", YouTubeGestureRouter._CONSUMABLE_STATIC)
        self.assertIn("fist", YouTubeGestureRouter._CONSUMABLE_STATIC)

    def test_thumb_up_does_not_like_in_forced_mode(self) -> None:
        router = YouTubeGestureRouter(
            static_hold_seconds=0.0,
            static_cooldown_seconds=0.0,
        )
        controller = _FakeYouTubeController()
        self.assertTrue(router.force_on(now=0.0, controller=controller))
        router.update(
            stable_label="thumb_up",
            dynamic_label="neutral",
            controller=controller,
            now=1.0,
        )
        router.update(
            stable_label="thumb_up",
            dynamic_label="neutral",
            controller=controller,
            now=2.0,
        )
        self.assertEqual(controller.like_calls, 0)
        self.assertEqual(controller.dislike_calls, 0)
        self.assertNotEqual(router.snapshot().last_action, "youtube_like")

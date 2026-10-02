from __future__ import annotations

import unittest
from unittest.mock import patch

from hgr.debug.chrome_controller import ChromeController


class ChromeControllerTest(unittest.TestCase):
    def test_voice_request_targets_chrome_when_phrase_mentions_chrome(self) -> None:
        controller = ChromeController(executable_paths=())

        self.assertTrue(controller.voice_request_targets_chrome("open indeed on google chrome please"))

    def test_voice_request_targets_chrome_when_mode_forces_chrome(self) -> None:
        controller = ChromeController(executable_paths=())

        self.assertTrue(controller.voice_request_targets_chrome("play lo fi beats", assume_chrome=True))

    def test_parse_voice_search_request_extracts_query(self) -> None:
        controller = ChromeController(executable_paths=())
        query = controller.parse_voice_search_request("search best tacos in seattle on google chrome")

        self.assertEqual(query, "best tacos in seattle")

    def test_parse_voice_search_request_strips_open_and_chrome_words(self) -> None:
        controller = ChromeController(executable_paths=())
        query = controller.parse_voice_search_request("open indeed on google chrome please")

        self.assertEqual(query, "indeed")

    def test_parse_voice_search_request_strips_search_up_phrase(self) -> None:
        controller = ChromeController(executable_paths=())
        query = controller.parse_voice_search_request("search up youtube on google chrome")

        self.assertEqual(query, "youtube")

    def test_parse_voice_search_request_assume_chrome_uses_phrase_without_target_words(self) -> None:
        controller = ChromeController(executable_paths=())
        query = controller.parse_voice_search_request("latest seattle weather", assume_chrome=True)

        self.assertEqual(query, "latest seattle weather")

    def test_normalize_spoken_target_repairs_known_domain(self) -> None:
        controller = ChromeController(executable_paths=())

        target = controller.normalize_spoken_target("indeedn.com")

        self.assertEqual(target, "indeed.com")

    def test_normalize_spoken_target_repairs_known_site_name(self) -> None:
        controller = ChromeController(executable_paths=())

        target = controller.normalize_spoken_target("chat gpt")

        self.assertEqual(target, "chatgpt")

    def test_search_google_launches_query_url(self) -> None:
        controller = ChromeController(executable_paths=())
        with patch.object(controller, "_launch_target", return_value=True) as launch_mock:
            with patch.object(controller, "_wait_for_window_handles", return_value=[123]):
                with patch.object(controller, "focus_or_open_window", return_value=True):
                    self.assertTrue(controller.search_google("best tacos"))

        launched_url = launch_mock.call_args.args[0]
        self.assertIn("google.com/search", launched_url)
        self.assertIn("best+tacos", launched_url)


    def test_normalize_spoken_target_strips_browser_context_tail(self) -> None:
        controller = ChromeController(executable_paths=())

        target = controller.normalize_spoken_target("youtube on google")

        self.assertEqual(target, "youtube")

    def test_parse_voice_search_request_strips_google_chrome_tail_from_open_phrase(self) -> None:
        controller = ChromeController(executable_paths=())
        query = controller.parse_voice_search_request("Can you open youtube on google chrome")

        self.assertEqual(query, "youtube")

    def test_stale_handle_cache_does_not_scan_on_hot_path(self) -> None:
        controller = ChromeController(executable_paths=())
        controller._available = True
        controller._handles_cache = [42]
        controller._handles_cache_until = 0.0
        with patch.object(controller, "_schedule_handles_refresh") as sched:
            got = controller._chrome_window_handles()
        self.assertEqual(got, [42])
        sched.assert_called_once()

    def test_refresh_scans_on_caller_thread(self) -> None:
        controller = ChromeController(executable_paths=())
        controller._available = True
        with patch.object(
            controller, "_scan_chrome_window_handles", return_value=[7]
        ) as scan:
            got = controller._chrome_window_handles(refresh=True)
        self.assertEqual(got, [7])
        scan.assert_called_once()
        self.assertEqual(controller._handles_cache, [7])


# Author: Konstantin Markov

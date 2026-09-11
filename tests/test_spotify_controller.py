from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from hgr.debug.spotify_controller import SpotifyController


class SpotifyControllerTest(unittest.TestCase):
    def test_ensure_ready_uses_loaded_token_and_devices(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        controller._client_id = "test-client"
        controller._client_secret = "test-secret"
        controller._redirect_uri = "http://localhost:5000/callback"
        controller._access_token = "access-token"
        controller._refresh_token = "refresh-token"
        controller._token_issue_time = time.time()
        with patch.object(
            controller,
            "_get_devices",
            return_value=[{"id": "device-1", "name": "Desktop", "type": "Computer", "is_active": True, "is_restricted": False}],
        ):
            self.assertTrue(controller.ensure_ready())
        self.assertIn("spotify ready", controller.message)

    def test_get_current_track_details_resolves_playlist_name(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(
            controller,
            "get_player_state",
            return_value={
                "is_playing": True,
                "shuffle_state": False,
                "repeat_state": "off",
                "progress_ms": 25000,
                "context": {
                    "type": "playlist",
                    "uri": "spotify:playlist:playlist-123",
                },
                "device": {
                    "name": "Desktop",
                    "type": "Computer",
                },
                "item": {
                    "name": "Song",
                    "duration_ms": 180000,
                    "artists": [{"name": "Artist One"}, {"name": "Artist Two"}],
                    "album": {"name": "Album"},
                },
            },
        ):
            with patch.object(controller, "_get_playlist_name", return_value="My Playlist"):
                details = controller.get_current_track_details()
        self.assertIsNotNone(details)
        assert details is not None
        self.assertEqual(details.song_name, "Song")
        self.assertEqual(details.artist_names, "Artist One, Artist Two")
        self.assertEqual(details.playlist_name, "My Playlist")

    def test_toggle_repeat_track_switches_track_on_and_off(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(controller, "get_player_state", side_effect=[{"repeat_state": "off"}, {"repeat_state": "track"}]):
            with patch.object(controller, "_request_json", side_effect=[(204, None), (204, None)]) as request_json_mock:
                self.assertTrue(controller.toggle_repeat_track())
                self.assertTrue(controller.toggle_repeat_track())

        first_call = request_json_mock.call_args_list[0]
        second_call = request_json_mock.call_args_list[1]
        self.assertEqual(first_call.kwargs["params"]["state"], "track")
        self.assertEqual(second_call.kwargs["params"]["state"], "off")

    def test_toggle_shuffle_flips_current_shuffle_state(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(controller, "get_player_state", side_effect=[{"shuffle_state": False}, {"shuffle_state": True}]):
            with patch.object(controller, "_request_json", side_effect=[(204, None), (204, None)]) as request_json_mock:
                self.assertTrue(controller.toggle_shuffle())
                self.assertTrue(controller.toggle_shuffle())

        first_call = request_json_mock.call_args_list[0]
        second_call = request_json_mock.call_args_list[1]
        self.assertEqual(first_call.kwargs["params"]["state"], "true")
        self.assertEqual(second_call.kwargs["params"]["state"], "false")

    def test_focus_or_open_window_is_noop_when_already_active(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(controller, "is_window_active", return_value=True):
            self.assertTrue(controller.focus_or_open_window())
        self.assertIn("already focused", controller.message)

    def test_focus_or_open_window_activates_existing_window(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(controller, "is_window_active", return_value=False):
            with patch.object(controller, "_spotify_window_handles", return_value=[12345]):
                with patch.object(controller, "_activate_window_handle", return_value=True) as activate_mock:
                    self.assertTrue(controller.focus_or_open_window())
        activate_mock.assert_called_once_with(12345)

    def test_parse_voice_play_request_prefers_playlist_when_named(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        request = controller.parse_voice_play_request("play the playlist chill mix on spotify")

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.query, "the chill mix")
        self.assertEqual(request.preferred_types[0], "playlist")

    def test_play_voice_request_plays_track_uri(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(controller, "ensure_ready", return_value=True):
            with patch.object(
                controller,
                "_search_best_playable",
                return_value={"kind": "track", "name": "Numb", "payload": {"uris": ["spotify:track:123"]}},
            ):
                with patch.object(controller, "_request_json", return_value=(204, None)) as request_mock:
                    with patch.object(controller, "focus_or_open_window", return_value=True) as focus_mock:
                        self.assertTrue(controller.play_voice_request("play numb on spotify"))
        focus_mock.assert_not_called()

        self.assertEqual(request_mock.call_args.kwargs["payload"], {"uris": ["spotify:track:123"]})

    def test_play_voice_request_plays_playlist_context(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(controller, "ensure_ready", return_value=True):
            with patch.object(
                controller,
                "_search_best_playable",
                return_value={"kind": "playlist", "name": "Focus", "payload": {"context_uri": "spotify:playlist:abc"}},
            ):
                with patch.object(controller, "_request_json", return_value=(204, None)) as request_mock:
                    with patch.object(controller, "focus_or_open_window", return_value=True) as focus_mock:
                        self.assertTrue(controller.play_voice_request("play focus playlist"))
        focus_mock.assert_not_called()

        self.assertEqual(request_mock.call_args.kwargs["payload"], {"context_uri": "spotify:playlist:abc"})

    def test_library_playlist_match_beats_generic_search(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(
            controller,
            "_request_json",
            side_effect=[
                (
                    200,
                    {
                        "items": [
                            {"name": "Chill Mix", "uri": "spotify:playlist:library-1"},
                            {"name": "Workout", "uri": "spotify:playlist:library-2"},
                        ],
                        "total": 2,
                        "next": None,
                    },
                )
            ],
        ):
            match = controller._find_library_playlist("chill mix")

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match["payload"]["context_uri"], "spotify:playlist:library-1")

    def test_add_current_track_to_queue_uses_track_uri(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(controller, "_current_track_uri", return_value="spotify:track:track-123"):
            with patch.object(controller, "ensure_ready", return_value=True):
                with patch.object(controller, "_request_json", return_value=(204, None)) as request_mock:
                    self.assertTrue(controller.add_current_track_to_queue())

        self.assertEqual(request_mock.call_args.args[1], "/me/player/queue")
        self.assertEqual(request_mock.call_args.kwargs["params"]["uri"], "spotify:track:track-123")

    def test_save_current_track_uses_track_id(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(controller, "_current_track_id", return_value="track-123"):
            with patch.object(controller, "_request_json", return_value=(204, None)) as request_mock:
                self.assertTrue(controller.save_current_track())

        self.assertEqual(request_mock.call_args.args[1], "/me/tracks")
        self.assertEqual(request_mock.call_args.kwargs["params"]["ids"], "track-123")

    def test_remove_current_track_from_liked_uses_track_id(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(controller, "_current_track_id", return_value="track-123"):
            with patch.object(controller, "_request_json", return_value=(200, None)) as request_mock:
                self.assertTrue(controller.remove_current_track_from_liked())

        self.assertEqual(request_mock.call_args.args[0], "DELETE")
        self.assertEqual(request_mock.call_args.args[1], "/me/tracks")
        self.assertEqual(request_mock.call_args.kwargs["params"]["ids"], "track-123")

    def test_add_current_track_to_playlist_prefers_library_match(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(
            controller,
            "_find_library_playlist",
            return_value={"name": "Chill Mix", "payload": {"context_uri": "spotify:playlist:library-1"}},
        ):
            with patch.object(controller, "_current_track_uri", return_value="spotify:track:track-123"):
                with patch.object(controller, "_request_json", return_value=(201, None)) as request_mock:
                    self.assertTrue(controller.add_current_track_to_playlist("chill mix"))

        self.assertEqual(request_mock.call_args.args[1], "/playlists/library-1/tracks")
        self.assertEqual(request_mock.call_args.kwargs["payload"], {"uris": ["spotify:track:track-123"]})

    def test_remove_current_track_from_playlist_uses_playlist_lookup(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        with patch.object(
            controller,
            "_find_library_playlist",
            return_value={"name": "Chill Mix", "payload": {"context_uri": "spotify:playlist:library-1"}},
        ):
            with patch.object(controller, "_current_track_uri", return_value="spotify:track:track-123"):
                with patch.object(controller, "_request_json", return_value=(200, {"snapshot_id": "snap"})) as request_mock:
                    self.assertTrue(controller.remove_current_track_from_playlist("chill mix"))

        self.assertEqual(request_mock.call_args.args[0], "DELETE")
        self.assertEqual(request_mock.call_args.args[1], "/playlists/library-1/tracks")
        self.assertEqual(
            request_mock.call_args.kwargs["payload"],
            {"tracks": [{"uri": "spotify:track:track-123"}]},
        )

    def test_remove_current_track_from_queue_reports_unsupported(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        self.assertFalse(controller.remove_current_track_from_queue())
        self.assertIn("not supported", controller.message)

    def test_mac_process_probe_matches_spotify_app_name(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        controller._mac = True

        class _Proc:
            info = {"name": "Spotify", "exe": "/Applications/Spotify.app/Contents/MacOS/Spotify"}

        with patch("hgr.debug.spotify_controller.psutil.process_iter", return_value=[_Proc()]):
            self.assertTrue(controller._probe_real_spotify_process())

    def test_windows_process_probe_still_requires_spotify_exe(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        controller._mac = False

        class _Proc:
            info = {"name": "Spotify", "exe": None}

        with patch("hgr.debug.spotify_controller.psutil.process_iter", return_value=[_Proc()]):
            self.assertFalse(controller._probe_real_spotify_process())

    def test_mac_applescript_readiness_without_oauth_tokens(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        controller._mac = True
        controller._access_token = None
        controller._refresh_token = None
        controller._client_id = "test-client"
        controller._needs_reauth = False
        controller._available = True
        # Connect toast is OAuth, not "Spotify.app is running".
        self.assertEqual(controller.readiness_state(), "NO_TOKENS")
        controller._available = False
        self.assertEqual(controller.readiness_state(), "NO_TOKENS")
        controller._access_token = "tok"
        self.assertEqual(controller.readiness_state(), "READY")

    def test_mac_transport_fails_when_spotify_is_not_running(self) -> None:
        controller = SpotifyController(env_paths=(), token_paths=(), executable_paths=())
        controller._mac = True
        with patch.object(controller, "_mac_osascript", return_value=(True, "not-running", "")):
            self.assertFalse(controller._mac_transport("playpause"))
        self.assertIsNone(controller.take_transient_failure())

# Author: Konstantin Markov

from __future__ import annotations

import contextlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hgr.debug.chrome_controller import ChromeController
from hgr.debug.desktop_controller import DesktopAppEntry, DesktopController
from hgr.debug.spotify_controller import SpotifyController
from hgr.voice.command_processor import VoiceCommandContext, VoiceCommandProcessor, VoiceProfileStore


class VoiceCommandProcessorTest(unittest.TestCase):
    @contextlib.contextmanager
    def _profile_dir(self):
        root = Path.home() / "Documents"
        tmp_dir = Path(tempfile.mkdtemp(dir=str(root)))
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _make_processor(self, profile_path: Path) -> VoiceCommandProcessor:
        return VoiceCommandProcessor(
            chrome_controller=ChromeController(executable_paths=()),
            spotify_controller=SpotifyController(env_paths=(), token_paths=(), executable_paths=()),
            desktop_controller=DesktopController(outlook_paths=()),
            profile_store=VoiceProfileStore(path=profile_path),
        )

    def _app_entry(
        self,
        processor: VoiceCommandProcessor,
        display_name: str,
        target: str,
        *,
        source: str = "start_apps",
        aliases: tuple[str, ...] = (),
        category: str = "gaming",
    ) -> DesktopAppEntry:
        controller = processor.desktop_controller
        return DesktopAppEntry(
            display_name=display_name,
            normalized_name=controller._normalize_application_name(display_name),
            target=target,
            source=source,
            aliases=controller._build_entry_aliases(display_name, aliases),
            category=category,
        )

    def test_parse_play_spotify_routes_to_resume(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("play spotify")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "spotify")
        self.assertEqual(intent.action, "resume")
        self.assertIsNone(intent.query)

    def test_execute_play_spotify_focuses_then_resumes(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")
            with patch.object(processor.spotify_controller, "focus_or_open_window", return_value=True) as focus_mock:
                with patch.object(processor.spotify_controller, "play", return_value=True) as play_mock:
                    with patch.object(processor.spotify_controller, "get_current_track_details", return_value=None):
                        result = processor.execute("play spotify")

        self.assertTrue(result.success)
        self.assertEqual(result.target, "spotify")
        focus_mock.assert_called_once_with()
        play_mock.assert_called_once_with()

    def test_parse_spotify_play_command_extracts_query(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("Play back in black by AC/DC on spotify")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "spotify")
        self.assertEqual(intent.action, "play")
        self.assertEqual(intent.query, "back in black by ac/dc")
        self.assertEqual(intent.slots["preferred_types"][0], "track")

    def test_parse_chrome_search_uses_context_preference(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse(
                "latest seattle weather",
                context=VoiceCommandContext(preferred_app="chrome"),
            )

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "chrome")
        self.assertEqual(intent.action, "search")
        self.assertEqual(intent.query, "latest seattle weather")

    def test_parse_chrome_search_up_phrase_extracts_clean_domain(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("search up indeed.com on google chrome")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "chrome")
        self.assertEqual(intent.action, "search")
        self.assertEqual(intent.query, "indeed.com")

    def test_parse_chrome_search_phrase_strips_full_browser_suffix(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("search up youtube on google chrome please")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "chrome")
        self.assertEqual(intent.action, "search")
        self.assertEqual(intent.query, "youtube")

    def test_parse_chrome_open_phrase_drops_trailing_on(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("can you open chatgpt on chrome?")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "chrome")
        self.assertEqual(intent.query, "chatgpt")

    def test_parse_settings_topic_intent(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("open bluetooth settings")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "settings")
        self.assertEqual(intent.query, "bluetooth")

    def test_parse_settings_casual_phrase_keeps_display_topic(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("can you open my pc's display settings")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "settings")
        self.assertEqual(intent.query, "display")

    def test_parse_file_explorer_folder_intent(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("open downloads folder")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "file_explorer")
        self.assertEqual(intent.action, "open")
        self.assertEqual(intent.query, "downloads")

    def test_parse_file_explorer_keeps_specific_file_request(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("open budget report pdf in documents folder")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "file_explorer")
        self.assertEqual(intent.action, "open")
        self.assertEqual(intent.query, "budget report pdf in documents")
        self.assertEqual(intent.slots.get("preferred_root"), "documents")

    def test_parse_plain_filename_request_routes_to_file_explorer(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            with patch.object(processor.desktop_controller, "rank_applications_in_text", return_value=[]):
                with patch.object(processor.desktop_controller, "can_resolve_application", return_value=False):
                    intent = processor.parse("open budget report pdf")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "file_explorer")
        self.assertEqual(intent.action, "open")
        self.assertEqual(intent.query, "budget report pdf")

    def test_parse_file_request_skips_expensive_catalog_lookup(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            with patch.object(
                processor.desktop_controller,
                "rank_applications_in_text",
                side_effect=AssertionError("catalog lookup should not run for file requests"),
            ):
                intent = processor.parse("open budget report text")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "file_explorer")
        self.assertEqual(intent.action, "open")
        self.assertEqual(intent.query, "budget report text")

    def test_parse_outlook_folder_intent(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("can you open my sent items from outlook please")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "outlook")
        self.assertEqual(intent.action, "open_folder")
        self.assertEqual(intent.query, "Sent Items")

    def test_parse_outlook_folder_without_saying_outlook(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("open sent items please")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "outlook")
        self.assertEqual(intent.action, "open_folder")
        self.assertEqual(intent.query, "Sent Items")

    def test_parse_generic_app_open_requires_resolvable_app(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("open steam")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "system")
        self.assertEqual(intent.query, "steam")

    def test_parse_generic_open_recovers_kicad_from_common_mishearing(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("open key card")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "system")
        self.assertEqual(intent.query, "kicad")

    def test_parse_generic_open_recovers_kicad_even_when_chrome_is_preferred(self) -> None:
        """A catalog match must beat `preferred_app`.

        "kkad" is not in the mishearing table (unlike "key card", which
        the sibling case covers), so recovery goes through
        `_best_fuzzy_launch_match` -> `rank_applications_in_text`, which
        reads the machine's REAL installed-app index. That made this case
        depend on whether KiCad happened to be installed on whoever ran
        it: with an empty catalog there is nothing to recover, the Chrome
        handler claims any utterance containing "open" while
        `preferred_app="chrome"`, and the assertion failed on a
        difference in the dev machine rather than in the code.

        Seeding one entry through the same `_app_entry` + `patch.object`
        pattern the ambiguous-prompt cases above use makes it
        deterministic and actually tests the precedence rule.

        `query` is compared case-insensitively on purpose: this path
        returns the catalog entry's display name ("KiCad"), whereas the
        mishearing-table path returns the normalized "kicad".
        """
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")
            kicad = self._app_entry(
                processor,
                "KiCad",
                "C:/Program Files/KiCad/bin/kicad.exe",
                aliases=("kicad",),
                category="engineering",
            )

            with patch.object(
                processor.desktop_controller,
                "rank_applications_in_text",
                return_value=[(kicad, 0.86, "kicad")],
            ):
                intent = processor.parse(
                    "open kkad app",
                    context=VoiceCommandContext(preferred_app="chrome"),
                )

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "system")
        assert intent.query is not None
        self.assertEqual(intent.query.lower(), "kicad")

    def test_parse_catalog_open_falls_back_to_best_matching_app(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("bring up my steam library")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "system")
        self.assertEqual(intent.query, "steam")

    def test_parse_browser_request_is_not_stolen_by_fuzzy_app_open(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("open youtube on chrome", context=VoiceCommandContext(preferred_app="chrome"))

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "chrome")
        self.assertEqual(intent.query, "youtube")

    def test_execute_ambiguous_app_prompt_uses_letters_and_opens_exact_selection(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")
            sequel = self._app_entry(processor, "Slay the Spire 2", "C:/Games/SlayTheSpire2.exe")
            original = self._app_entry(processor, "Slay the Spire", "C:/Games/SlayTheSpire.exe")

            with patch.object(
                processor.desktop_controller,
                "resolve_named_application_options",
                return_value=(sequel, [sequel, original]),
            ), patch.object(
                processor.desktop_controller,
                "can_resolve_application",
                return_value=True,
            ):
                prompt = processor.execute("open slay the spire")

            self.assertFalse(prompt.success)
            self.assertEqual(prompt.target, "voice_selection")
            self.assertIn("A. Slay the Spire 2", prompt.display_text or "")
            self.assertIn("B. Slay the Spire", prompt.display_text or "")

            with patch.object(processor.desktop_controller, "open_desktop_entry", return_value=True) as open_mock:
                followup = processor.execute("open option a")

        self.assertTrue(followup.success)
        open_mock.assert_called_once()
        self.assertEqual(open_mock.call_args.args[0].target, "C:/Games/SlayTheSpire2.exe")

    def test_execute_ambiguous_app_prompt_still_accepts_number_selection(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")
            fallout_3 = self._app_entry(processor, "Fallout 3", "C:/Games/Fallout3.exe")
            new_vegas = self._app_entry(processor, "Fallout New Vegas", "C:/Games/FalloutNV.exe")

            with patch.object(
                processor.desktop_controller,
                "resolve_named_application_options",
                return_value=(fallout_3, [fallout_3, new_vegas]),
            ), patch.object(
                processor.desktop_controller,
                "can_resolve_application",
                return_value=True,
            ):
                prompt = processor.execute("open fallout")

            self.assertFalse(prompt.success)
            with patch.object(processor.desktop_controller, "open_desktop_entry", return_value=True) as open_mock:
                followup = processor.execute("open one")

        self.assertTrue(followup.success)
        open_mock.assert_called_once()
        self.assertEqual(open_mock.call_args.args[0].target, "C:/Games/Fallout3.exe")

    def test_execute_routes_to_spotify_search_request(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")
            with patch.object(processor.spotify_controller, "play_search_request", return_value=True) as play_mock:
                with patch.object(processor.spotify_controller, "get_current_track_details", return_value=None):
                    result = processor.execute("play back in black by AC/DC on spotify")

        self.assertTrue(result.success)
        self.assertEqual(result.target, "spotify")
        play_mock.assert_called_once()
        self.assertEqual(play_mock.call_args.args[0], "back in black by ac/dc")

    def test_execute_routes_to_chrome_open_or_search(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")
            with patch.object(processor.chrome_controller, "open_or_search", return_value=True) as open_mock:
                result = processor.execute("open youtube on chrome")

        self.assertTrue(result.success)
        self.assertEqual(result.target, "chrome")
        open_mock.assert_called_once_with("youtube")

    def test_execute_routes_to_outlook_folder(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")
            with patch.object(processor.desktop_controller, "open_outlook_folder", return_value=True) as open_mock:
                result = processor.execute("open sent items from outlook")

        self.assertTrue(result.success)
        self.assertEqual(result.target, "outlook")
        open_mock.assert_called_once_with("Sent Items")

    def test_execute_file_request_uses_named_file_resolution(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")
            with patch.object(processor.desktop_controller, "open_named_file", return_value=True) as open_mock:
                result = processor.execute("open budget report pdf in documents folder")

        self.assertTrue(result.success)
        self.assertEqual(result.target, "file_explorer")
        open_mock.assert_called_once_with(
            "budget report pdf in documents",
            preferred_root="documents",
            folder_hint=None,
        )

    def test_successful_commands_are_learned(self) -> None:
        with self._profile_dir() as tmp_dir:
            profile_path = tmp_dir / "voice_profile.json"
            processor = self._make_processor(profile_path)
            with patch.object(processor.desktop_controller, "open_settings", return_value=True):
                processor.execute("open bluetooth settings")

            store = VoiceProfileStore(path=profile_path)
            learned = store.best_match("open bluetooth settings")

        self.assertIsNotNone(learned)
        assert learned is not None
        self.assertEqual(learned["app_name"], "settings")
        self.assertEqual(learned["action"], "open")

    def test_export_training_bundle_writes_expected_files(self) -> None:
        with self._profile_dir() as tmp_dir:
            profile_path = tmp_dir / "voice_profile.json"
            processor = self._make_processor(profile_path)
            paths = processor.export_training_bundle(output_dir=tmp_dir / "bundle")
            self.assertTrue(paths["train"].exists())
            self.assertTrue(paths["eval"].exists())
            self.assertTrue(paths["corrections"].exists())
            self.assertTrue(paths["summary"].exists())


    def test_parse_chrome_open_phrase_handles_google_chrome_context(self) -> None:
        with self._profile_dir() as tmp_dir:
            processor = self._make_processor(tmp_dir / "voice_profile.json")

            intent = processor.parse("Can you open youtube on google chrome")

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.app_name, "chrome")
        self.assertEqual(intent.action, "open")
        self.assertEqual(intent.query, "youtube")

# Author: Konstantin Markov

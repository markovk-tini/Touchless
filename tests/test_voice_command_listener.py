"""Unit tests for the voice-command listener's pure helpers.

Three cases used to live here that no longer can: two for
`_select_phrase` and one for `_parse_payload`. Both methods parsed the
JSON payload of a `powershell.exe` System.Speech.Recognition child
process, and `681b688` ("drop PowerShell shell-outs Defender ASR was
quarantining") deleted that whole backend -- the byte pattern of a
freshly-signed PyInstaller exe spawning hidden powershell.exe was
auto-quarantining the installed app. Whisper is now the only path, and
it returns one transcription rather than SAPI's ranked alternates, so
"pick the most complete phrase out of the alternate list" is not a
behaviour that exists to test. The cases are gone rather than skipped:
the backend is not coming back.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hgr.debug.voice_command_listener import VoiceCommandListener


class VoiceCommandListenerTest(unittest.TestCase):
    def test_normalize_text_applies_domain_corrections(self) -> None:
        listener = VoiceCommandListener()

        normalized = listener._normalize_text("Play Back In Black by AC DC on Google Chrome")

        self.assertEqual(normalized, "play back in black by ac/dc on chrome")

    def test_build_initial_prompt_includes_app_hints(self) -> None:
        listener = VoiceCommandListener()
        listener.set_app_hints(["KiCad", "Visual Studio Code"])

        prompt = listener._build_initial_prompt()

        self.assertIn("kicad", prompt)
        self.assertIn("visual studio code", prompt)
        # This case also asserted `assertIn("nested folders", prompt)`.
        # That string has only ever existed in this file -- no revision of
        # `_build_initial_prompt` has emitted it -- so the assertion was
        # red from the day it was written. What the prompt really does
        # carry for file targets is a folder-name vocabulary, so guard
        # that instead: it is the behaviour the removed line was reaching
        # for, and the one a regression would actually break.
        self.assertIn("folder names", prompt)
        self.assertIn("documents", prompt.lower())

    def test_build_initial_prompt_dictation_mentions_long_form_writing(self) -> None:
        listener = VoiceCommandListener()

        prompt = listener._build_initial_prompt(transcript_mode="dictation")

        self.assertIn("emails", prompt.lower())
        self.assertIn("new paragraph", prompt.lower())

    def test_build_initial_prompt_save_prompt_mentions_default_and_folders(self) -> None:
        listener = VoiceCommandListener()

        prompt = listener._build_initial_prompt(transcript_mode="save_prompt")

        self.assertIn("default", prompt.lower())
        self.assertIn("documents", prompt.lower())
        self.assertIn("absolute windows path", prompt.lower())

    def test_normalize_text_preserves_case_for_dictation_mode(self) -> None:
        listener = VoiceCommandListener()

        normalized = listener._normalize_text("ChatGPT in KiCad", transcript_mode="dictation")

        self.assertEqual(normalized, "ChatGPT in KiCad")

    # ---- whisper.cpp discovery -------------------------------------------
    #
    # These all patch `_candidate_whisper_roots`, because that -- not
    # `self._whisper_cpp_root` -- is what the command and model resolvers
    # actually walk. Two of them used to set `_whisper_cpp_root` to a fake
    # path and patch only `Path.exists`, which left the resolver walking
    # the REAL roots (`app_base_path()` and every parent of this file). The
    # model case then returned the repo's own
    # `whisper.cpp/models/ggml-medium.en.bin`, because the `ggml-*.bin`
    # glob fallback is not guarded by `exists()` and `glob` was never
    # patched -- a "fake path" test quietly reading the developer's real
    # checkout. `_whisper_cpp_root` is still live, but only for the VAD
    # lookup at the bottom of this file.

    def test_resolve_whisper_cpp_command_finds_local_build_when_not_on_path(self) -> None:
        """Renamed from ..._prefers_local_build: the resolver checks
        `shutil.which` BEFORE the local build dirs, so a whisper-cli on
        PATH wins. What this pins is the fallback -- nothing on PATH, so
        walk the roots -- and that `build/bin/Release` is tried first."""
        listener = VoiceCommandListener()
        fake_root = Path("C:/fake/whisper.cpp")
        command_path = fake_root / "build" / "bin" / "Release" / "whisper-cli.exe"

        with patch.dict(os.environ, {"HGR_WHISPER_CPP": ""}), \
                patch("shutil.which", return_value=None), \
                patch.object(VoiceCommandListener, "_candidate_whisper_roots",
                             return_value=[fake_root]), \
                patch.object(Path, "exists", autospec=True,
                             side_effect=lambda path: str(path) == str(command_path)):
            resolved = listener._resolve_whisper_cpp_command()

        self.assertEqual(resolved, (str(command_path),))

    def test_resolve_whisper_cpp_model_prefers_medium_en(self) -> None:
        listener = VoiceCommandListener()
        listener._model_root = Path("C:/fake/models")
        listener._whisper_cpp_model_path = None
        fake_root = Path("C:/fake/whisper.cpp")
        medium_path = fake_root / "models" / "ggml-medium.en.bin"

        with patch.dict(os.environ, {"HGR_WHISPER_CPP_MODEL": ""}), \
                patch.object(VoiceCommandListener, "_candidate_whisper_roots",
                             return_value=[fake_root]), \
                patch.object(Path, "glob", autospec=True,
                             side_effect=lambda path, pattern: iter(())), \
                patch.object(Path, "exists", autospec=True,
                             side_effect=lambda path: str(path) == str(medium_path)):
            resolved = listener._resolve_whisper_cpp_model_path()

        self.assertEqual(resolved, medium_path)

    def test_resolve_whisper_cpp_model_skips_for_tests_files(self) -> None:
        """Split out of the medium.en case, which claimed this in its name
        but could not reach it: the `for-tests-` filter lives in the
        `ggml-*.bin` glob fallback, and a run where medium.en exists
        returns before that fallback is ever consulted."""
        listener = VoiceCommandListener()
        listener._model_root = Path("C:/fake/models")
        listener._whisper_cpp_model_path = None
        fake_root = Path("C:/fake/whisper.cpp")
        models_dir = fake_root / "models"

        def _fake_glob(path, pattern):
            if str(path) == str(models_dir):
                return iter([path / "for-tests-tiny.en.bin",
                             path / "ggml-large-v3-turbo.bin"])
            return iter(())

        with patch.dict(os.environ, {"HGR_WHISPER_CPP_MODEL": ""}), \
                patch.object(VoiceCommandListener, "_candidate_whisper_roots",
                             return_value=[fake_root]), \
                patch.object(Path, "glob", autospec=True, side_effect=_fake_glob), \
                patch.object(Path, "exists", autospec=True,
                             side_effect=lambda path: False):
            resolved = listener._resolve_whisper_cpp_model_path()

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.name, "ggml-large-v3-turbo.bin")
        self.assertNotIn("for-tests", resolved.name)

    def test_resolve_whisper_cpp_vad_model_finds_downloaded_vad(self) -> None:
        listener = VoiceCommandListener()
        listener._model_root = Path("C:/fake/models")
        listener._whisper_cpp_root = Path("C:/fake/whisper.cpp")
        vad_path = listener._model_root / "ggml-silero-v5.1.2.bin"
        with patch.object(Path, "exists", autospec=True, side_effect=lambda path: str(path) == str(vad_path)):
            resolved = listener._resolve_whisper_cpp_vad_model_path()

        self.assertEqual(resolved, vad_path)


if __name__ == "__main__":
    unittest.main()

# Author: Konstantin Markov

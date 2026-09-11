from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from hgr.config.app_config import AppConfig, configured_save_directory
from hgr.voice.save_prompt import SavePromptProcessor


class SavePromptProcessorTest(unittest.TestCase):
    def _temp_dir(self) -> Path:
        return Path(tempfile.mkdtemp())

    def test_parse_silence_defaults_to_configured_folder(self) -> None:
        processor = SavePromptProcessor()

        decision = processor.parse("", success=False)

        self.assertEqual(decision.action, "default")
        self.assertEqual(decision.reason, "silence")

    def test_empty_cancel_payload_keeps_default_folder(self) -> None:
        """Left-fist cancel emits success=False and empty text.

        Spoken 'cancel/delete/nevermind' still discards; gesture cancel
        must not go through that path.
        """
        processor = SavePromptProcessor()
        decision = processor.parse("", success=False)
        self.assertEqual(decision.action, "default")
        self.assertNotEqual(decision.action, "discard")

    def test_parse_cancel_discards_output(self) -> None:
        processor = SavePromptProcessor()

        decision = processor.parse("nevermind delete it")

        self.assertEqual(decision.action, "discard")

    def test_parse_known_folder_moves_to_documents(self) -> None:
        processor = SavePromptProcessor()

        decision = processor.parse("save it in documents folder please")

        self.assertEqual(decision.action, "move")
        assert decision.folder is not None
        self.assertEqual(decision.folder.name.lower(), "documents")

    def test_parse_explicit_existing_path_moves_output(self) -> None:
        target_dir = self._temp_dir()
        try:
            processor = SavePromptProcessor()

            decision = processor.parse(str(target_dir))

            self.assertEqual(decision.action, "move")
            self.assertEqual(decision.folder, target_dir)
        finally:
            shutil.rmtree(target_dir, ignore_errors=True)


class ConfiguredSaveDirectoryTest(unittest.TestCase):
    def _temp_dir(self) -> Path:
        return Path(tempfile.mkdtemp())

    def test_configured_save_directory_creates_missing_folder(self) -> None:
        root = self._temp_dir()
        try:
            drawings_dir = root / "custom" / "drawings"
            config = AppConfig(drawings_save_dir=str(drawings_dir))

            resolved = configured_save_directory(config, "drawings")

            self.assertEqual(resolved, drawings_dir)
            self.assertTrue(drawings_dir.exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

# Author: Konstantin Markov

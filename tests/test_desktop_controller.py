from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hgr.debug.desktop_controller import DesktopAppEntry, DesktopController


class DesktopControllerTest(unittest.TestCase):
    def _temp_dir(self) -> Path:
        return Path(tempfile.mkdtemp())

    def _app_entry(
        self,
        controller: DesktopController,
        display_name: str,
        target: str,
        *,
        source: str = "start_apps",
        aliases: tuple[str, ...] = (),
        category: str = "gaming",
    ) -> DesktopAppEntry:
        return DesktopAppEntry(
            display_name=display_name,
            normalized_name=controller._normalize_application_name(display_name),
            target=target,
            source=source,
            aliases=controller._build_entry_aliases(display_name, aliases),
            category=category,
        )

    def test_can_resolve_known_app_aliases(self) -> None:
        controller = DesktopController(outlook_paths=())

        self.assertTrue(controller.can_resolve_application("steam"))
        self.assertTrue(controller.can_resolve_application("visual studios"))

    def test_rank_applications_in_text_prefers_known_alias(self) -> None:
        controller = DesktopController(outlook_paths=())

        ranked = controller.rank_applications_in_text("open a visual stdios window please")

        self.assertTrue(ranked)
        top_entry, score, matched_alias = ranked[0]
        self.assertEqual(top_entry.display_name, "visual studio code")
        self.assertGreaterEqual(score, 0.82)
        self.assertIn(matched_alias, {"visual studio", "visual studio code"})

    def test_rank_applications_in_text_ignores_generic_apps_alias_noise(self) -> None:
        """A generic "app"/"apps" alias must not outrank a real name match.

        The catalog is seeded rather than read off the machine.
        `_application_catalog` caches into a CLASS-level
        `_shared_app_catalog`, and under pytest
        `_ensure_background_catalog_build` deliberately sets that to `[]`
        -- the background Start Menu scan was surfacing a Windows access
        violation deep in the suite. So the FIRST call in a process falls
        back to the quick catalog while every later call gets the empty
        sentinel. "kkad" scores 0.373 against "kicad", below the 0.78
        quick-path short circuit, so this case reached the full catalog
        and passed only while it happened to be the first test in the
        process to ask for one. It went red the moment any other file ran
        first, which is what made it look order-dependent. Production
        never takes that branch ("pytest" is not in `sys.modules` there),
        so the fix belongs in the test.
        """
        controller = DesktopController(outlook_paths=())
        kicad = self._app_entry(
            controller, "kicad", "C:/Program Files/KiCad/bin/kicad.exe"
        )
        # The noise this case is named for: an entry whose aliases are the
        # bare words "app"/"apps", which the trailing "app" in the
        # utterance would otherwise match outright.
        noise = self._app_entry(
            controller,
            "Generic Apps Launcher",
            "C:/Apps/generic.exe",
            aliases=("app", "apps"),
        )

        with patch.object(
            controller, "_quick_application_catalog", return_value=[kicad, noise]
        ), patch.object(
            controller, "_application_catalog", return_value=[kicad, noise]
        ):
            ranked = controller.rank_applications_in_text("open kkad app")

        self.assertTrue(ranked)
        self.assertEqual(ranked[0][0].display_name, "kicad")

    def test_open_named_application_prefers_known_display_name(self) -> None:
        controller = DesktopController(outlook_paths=())

        with patch.object(controller, "_launch_path_or_command", return_value=True) as launch_mock:
            self.assertTrue(controller.open_named_application("visual studios"))

        launch_mock.assert_called_once()
        self.assertEqual(controller.message, "opened app: visual studio code")

    def test_open_outlook_folder_uses_classic_select_when_available(self) -> None:
        classic_path = Path("C:/Program Files/Microsoft Office/root/Office16/OUTLOOK.EXE")
        controller = DesktopController(outlook_paths=())

        with patch.object(controller, "_classic_outlook_path", return_value=classic_path):
            with patch("subprocess.Popen") as popen_mock:
                self.assertTrue(controller.open_outlook_folder("scent"))

        popen_mock.assert_called_once_with([str(classic_path), "/select", "outlook:Sent Items"], shell=False)
        self.assertEqual(controller.message, "opened outlook folder: Sent Items")

    def test_resolve_named_application_options_prefers_exact_spoken_number_title(self) -> None:
        controller = DesktopController(outlook_paths=())
        sequel = self._app_entry(controller, "Slay the Spire 2", "C:/Games/SlayTheSpire2.exe")
        original = self._app_entry(controller, "Slay the Spire", "C:/Games/SlayTheSpire.exe")

        with patch.object(controller, "_quick_application_catalog", return_value=[sequel, original]):
            with patch.object(controller, "_application_catalog", return_value=[sequel, original]):
                resolved, ambiguous = controller.resolve_named_application_options("slay the spire two")

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.display_name, "Slay the Spire 2")
        self.assertFalse(ambiguous)

    def test_resolve_named_application_options_keeps_shared_family_queries_ambiguous(self) -> None:
        controller = DesktopController(outlook_paths=())
        new_vegas = self._app_entry(controller, "Fallout New Vegas", "C:/Games/FalloutNV.exe")
        fallout_3 = self._app_entry(controller, "Fallout 3", "C:/Games/Fallout3.exe")

        with patch.object(controller, "_quick_application_catalog", return_value=[new_vegas, fallout_3]):
            with patch.object(controller, "_application_catalog", return_value=[new_vegas, fallout_3]):
                resolved, ambiguous = controller.resolve_named_application_options("fallout")

        self.assertIsNotNone(resolved)
        self.assertGreaterEqual(len(ambiguous), 2)
        self.assertEqual(
            {entry.display_name for entry in ambiguous[:2]},
            {"Fallout New Vegas", "Fallout 3"},
        )

    def test_open_named_application_prefers_exact_numbered_match(self) -> None:
        controller = DesktopController(outlook_paths=())
        new_vegas = self._app_entry(controller, "Fallout New Vegas", "C:/Games/FalloutNV.exe")
        fallout_3 = self._app_entry(controller, "Fallout 3", "C:/Games/Fallout3.exe")

        with patch.object(controller, "_quick_application_catalog", return_value=[new_vegas, fallout_3]):
            with patch.object(controller, "_application_catalog", return_value=[new_vegas, fallout_3]):
                with patch.object(controller, "_launch_path_or_command", return_value=True) as launch_mock:
                    self.assertTrue(controller.open_named_application("fallout 3"))

        launch_mock.assert_called_once_with("C:/Games/Fallout3.exe")
        self.assertEqual(controller.message, "opened app: Fallout 3")

    def test_open_outlook_folder_reports_partial_fallback_when_only_opening_outlook(self) -> None:
        """With no Classic Outlook to /select into, report partial success.

        `_classic_outlook_path()` scans the standard Office install
        locations and ignores the `outlook_paths=()` constructor argument,
        so on any machine that HAS Classic Outlook this case took the
        success branch instead: it ran
        `subprocess.Popen([OUTLOOK.EXE, "/select", "outlook:Sent Items"])`
        -- actually launching Outlook on the developer's desktop on every
        suite run -- and then failed, because `open_outlook_folder`
        correctly returned True. "scent" is a real alias of "sent items",
        so the mishearing path was never the problem.

        Patching the path lookup pins the branch this case is named for
        and stops the suite launching mail clients. The companion case
        below covers the other branch with `Popen` stubbed out.
        """
        controller = DesktopController(outlook_paths=())

        with patch.object(controller, "_classic_outlook_path", return_value=None),                 patch.object(controller, "open_outlook", return_value=True):
            self.assertFalse(controller.open_outlook_folder("scent"))

        self.assertEqual(controller.message, "opened outlook, but could not select Sent Items")

    def test_open_outlook_folder_selects_the_folder_when_classic_outlook_exists(self) -> None:
        """The success branch, with the launch stubbed.

        Kept separate from the fallback case so neither depends on
        whether Classic Outlook happens to be installed, and so nothing
        here spawns a real process. Asserts the /select argument too: the
        canonical display name is what Outlook needs, and a regression
        that passed the raw mishearing ("scent") would still have
        returned True.
        """
        controller = DesktopController(outlook_paths=())
        fake_exe = Path("C:/Program Files/Microsoft Office/root/Office16/OUTLOOK.EXE")

        with patch.object(controller, "_classic_outlook_path", return_value=fake_exe),                 patch("hgr.debug.desktop_controller.subprocess.Popen") as popen:
            self.assertTrue(controller.open_outlook_folder("scent"))

        popen.assert_called_once()
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], str(fake_exe))
        self.assertEqual(argv[1], "/select")
        self.assertEqual(argv[2], "outlook:Sent Items")
        self.assertEqual(controller.message, "opened outlook folder: Sent Items")

    def test_open_named_file_can_resolve_plain_filename_without_folder_hint(self) -> None:
        root = self._temp_dir()
        try:
            target = root / "Budget Report.pdf"
            target.write_text("budget", encoding="utf-8")
            controller = DesktopController(outlook_paths=())

            with patch.object(controller, "_file_search_roots", return_value=[root]):
                with patch.object(controller, "_launch_target", return_value=True) as launch_mock:
                    self.assertTrue(controller.open_named_file("budget report pdf"))

            launch_mock.assert_called_once_with(str(target))
            self.assertEqual(controller.message, "opened file: Budget Report.pdf")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_open_named_file_prefers_exact_extension_match(self) -> None:
        root = self._temp_dir()
        try:
            pdf_target = root / "Final Presentation.pdf"
            ppt_target = root / "Final Presentation.pptx"
            pdf_target.write_text("pdf", encoding="utf-8")
            ppt_target.write_text("ppt", encoding="utf-8")
            controller = DesktopController(outlook_paths=())

            with patch.object(controller, "_file_search_roots", return_value=[root]):
                with patch.object(controller, "_launch_target", return_value=True) as launch_mock:
                    self.assertTrue(controller.open_named_file("final presentation pdf"))

            launch_mock.assert_called_once_with(str(pdf_target))
            self.assertEqual(controller.message, "opened file: Final Presentation.pdf")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_open_named_file_prefers_indexed_results_before_fallback_scan(self) -> None:
        root = self._temp_dir()
        try:
            target = root / "Test Cases" / "notes.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("notes", encoding="utf-8")
            controller = DesktopController(outlook_paths=())

            with patch.object(controller, "_file_search_roots", return_value=[root]):
                with patch.object(controller, "_query_indexed_paths", return_value=[target]):
                    with patch.object(controller, "_scan_file_candidates", return_value=[]) as scan_mock:
                        with patch.object(controller, "_launch_target", return_value=True) as launch_mock:
                            self.assertTrue(controller.open_named_file("notes txt"))

            launch_mock.assert_called_once_with(str(target))
            scan_mock.assert_not_called()
            self.assertEqual(controller.message, "opened file: notes.txt")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_resolve_named_folder_returns_known_documents_path(self) -> None:
        controller = DesktopController(outlook_paths=())

        resolved, ambiguous = controller.resolve_named_folder("documents folder")

        self.assertFalse(ambiguous)
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.name.lower(), "documents")

# Author: Konstantin Markov

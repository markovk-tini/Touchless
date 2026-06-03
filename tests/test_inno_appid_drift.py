"""Compile-time guard: Updater._INNO_APP_ID must match the AppId in
installers/windows/hgr_app.iss.

Both files independently hardcode the same GUID. Today they agree
({2C4EE680-53F5-4D83-92A8-ADF4D2D8794E}). A future rename — say,
splitting the app into a beta-channel + stable-channel with separate
GUIDs — would silently desync. The Updater would write DisplayVersion
to a registry key Inno never created, and Microsoft Store would keep
re-prompting Store users on every update because the registry value
it reads at install path never changes.

Catch the drift here at PR time instead of at user-runtime.

The check is intentionally string-level (re.search against the .iss)
because hgr_app.iss isn't a Python module and we don't want to pull
in an Inno-script parser. The regex is loose enough to handle
whitespace variants but strict enough to extract the GUID.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

# Allow running this test stand-alone without an installed package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Match `AppId={{<guid>}` or `AppId={{<guid>}}` — Inno's odd
# double-brace syntax for embedded braces in constants.
_ISS_APP_ID_RE = re.compile(
    r"^\s*AppId\s*=\s*\{*(\{[0-9A-Fa-f-]{36}\})\}*\s*$",
    re.MULTILINE,
)


class InnoAppIdDriftTest(unittest.TestCase):
    def test_updater_inno_app_id_matches_iss_file(self) -> None:
        iss_path = _REPO_ROOT / "installers" / "windows" / "hgr_app.iss"
        self.assertTrue(
            iss_path.exists(),
            f"hgr_app.iss not found at {iss_path} — repository layout "
            "may have changed. Update this test to point at the new path.",
        )
        iss_text = iss_path.read_text(encoding="utf-8")
        match = _ISS_APP_ID_RE.search(iss_text)
        self.assertIsNotNone(
            match,
            "Couldn't find AppId={...} in hgr_app.iss. "
            "If the format changed, update _ISS_APP_ID_RE.",
        )
        iss_app_id = match.group(1)  # e.g. {2C4EE680-...-ADF4D2D8794E}

        from hgr.app.updater.updater import Updater
        expected_suffix = "_is1"
        py_app_id = Updater._INNO_APP_ID
        self.assertTrue(
            py_app_id.endswith(expected_suffix),
            f"Updater._INNO_APP_ID should end with {expected_suffix!r} "
            f"(Inno's uninstall-key naming convention); got {py_app_id!r}.",
        )
        py_guid = py_app_id[:-len(expected_suffix)]
        # Inno's AppId can be wrapped in extra braces; normalize for
        # comparison.
        self.assertEqual(
            py_guid.upper(),
            iss_app_id.upper(),
            "DRIFT DETECTED: Updater._INNO_APP_ID GUID does not match "
            "the AppId in hgr_app.iss. The registry self-heal AND the "
            "post-app-zip DisplayVersion writeback will both write to "
            "the wrong key, leaving Microsoft Store + Add/Remove "
            "Programs stuck on the prior version. Pick one as the "
            "source of truth and update the other to match — or, "
            "better, refactor so both read from a shared constant "
            "(e.g. inject via the .iss-generation step at build time).",
        )

    def test_inno_app_id_format_sanity(self) -> None:
        """Sanity check on Updater._INNO_APP_ID's shape — must be
        `{GUID}_is1` exactly. Catches a sloppy edit that drops the
        braces or the _is1 suffix."""
        from hgr.app.updater.updater import Updater
        self.assertRegex(
            Updater._INNO_APP_ID,
            r"^\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
            r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}_is1$",
            "Updater._INNO_APP_ID must be `{GUID}_is1` exactly.",
        )


if __name__ == "__main__":
    unittest.main()

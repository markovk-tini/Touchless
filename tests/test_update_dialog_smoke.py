"""Update-dialog construction smoke test.

Purpose: catch the specific bug class that shipped in 1.1.8 and
1.1.8.1 — a NameError inside `UpdateDialog.__init__` that made the
dialog impossible to construct at runtime. Because no test ever
instantiated the dialog, the crash was invisible in CI and only
surfaced when a user's app tried to show an update prompt.

This test does the minimum needed to detect that class of bug:

  1. Instantiate `UpdateDialog` with a realistic `ReleaseInfo`.
  2. Call `.show()` and drive one event-loop tick.
  3. Assert the widget is visible, has the expected internal state
     (specifically `self._body`, so a regression to the bare-`body`
     NameError is caught immediately).
  4. Assert the version comparator we depend on ranks releases the
     way we assume (dismissed 1.1.9rc1 does NOT swallow 1.1.9 final;
     1.1.8.1 > 1.1.8).

Referenced by PUBLISHING_POLICY.md as a STOP-SHIP rule.
"""
from __future__ import annotations

import os
import unittest

# Qt has to run headless on CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class UpdateDialogSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def _make_info(self, **overrides):
        from hgr.app.updater.release_checker import ReleaseInfo
        defaults = dict(
            version="1.1.9",
            body="## Test\n- item one\n- item two",
            download_url="https://example.com/x.exe",
            html_url="https://github.com/example/example/releases/tag/v1.1.9",
            size_bytes=140_000_000,
            update_kind="full-exe",
        )
        defaults.update(overrides)
        return ReleaseInfo(**defaults)

    def test_dialog_constructs_and_is_visible(self) -> None:
        """The 1.1.8/1.1.8.1 regression: bare `body` NameError in
        _build_ui crashed __init__ so the dialog never got past
        construction. If this test fails with NameError or AttributeError,
        the same class of bug has returned — do not ship."""
        from hgr.app.updater.update_dialog import UpdateDialog
        dlg = UpdateDialog(self._make_info(), parent=None)
        try:
            dlg.show()
            self._app.processEvents()
            self.assertTrue(dlg.isVisible(), "dialog must be visible after show()")
            self.assertTrue(
                hasattr(dlg, "_body"),
                "UpdateDialog._body must exist (regression sentinel for the "
                "1.1.8/1.1.8.1 bare-body NameError). Do not delete this "
                "assertion — it is the reason this file exists.",
            )
            self.assertEqual(dlg.windowTitle(), "Touchless Update Available")
        finally:
            dlg.close()
            dlg.deleteLater()

    def test_dialog_survives_app_zip_kind(self) -> None:
        """app-zip is the auto-update path — smaller size, different
        subtitle. Must construct without exception."""
        from hgr.app.updater.update_dialog import UpdateDialog
        info = self._make_info(update_kind="app-zip", size_bytes=140_000_000)
        dlg = UpdateDialog(info, parent=None)
        try:
            dlg.show()
            self._app.processEvents()
            self.assertTrue(dlg.isVisible())
        finally:
            dlg.close()
            dlg.deleteLater()

    def test_dialog_survives_empty_release_body(self) -> None:
        """A release with an empty changelog body used to blow up
        the QTextBrowser.setMarkdown call. Guarded now; keep the
        assertion so future refactors can't reintroduce."""
        from hgr.app.updater.update_dialog import UpdateDialog
        info = self._make_info(body="")
        dlg = UpdateDialog(info, parent=None)
        try:
            dlg.show()
            self._app.processEvents()
            self.assertTrue(dlg.isVisible())
        finally:
            dlg.close()
            dlg.deleteLater()

    def test_version_comparator_pep440_ordering(self) -> None:
        """PUBLISHING_POLICY.md stop-ship rule: _is_newer must rank
        pre-releases correctly so a 'Later' click on 1.1.9rc1 doesn't
        silently swallow 1.1.9 final. Also 1.1.8.1 > 1.1.8."""
        from hgr.app.updater.release_checker import _is_newer
        self.assertTrue(
            _is_newer("1.1.9", "1.1.9rc1"),
            "final release must be newer than its rc",
        )
        self.assertTrue(
            _is_newer("1.1.8.1", "1.1.8"),
            "1.1.8.1 must be newer than 1.1.8 (four-part vs three-part)",
        )
        self.assertFalse(
            _is_newer("1.1.8", "1.1.8"),
            "same version must NOT be considered newer",
        )
        self.assertFalse(
            _is_newer("1.1.7", "1.1.8"),
            "older version must NOT be considered newer",
        )


if __name__ == "__main__":
    unittest.main()

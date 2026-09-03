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
  5. Assert the 1.1.7 z-order flags (FramelessWindowHint AND
     WindowStaysOnTopHint) survive construction, including when the
     dialog is parented to a shown window.
  6. Assert the tray-balloon fallback is actually callable
     (`TouchlessTrayIcon.showMessage`) so `hasattr(tray, "showMessage")`
     in MainWindow cannot go dead again.

Offscreen Qt cannot prove compositor z-order on real Windows. That
remains a human eyeball stop-ship in docs/UPDATE_RELEASE_CHECKLIST.md.
These tests pin everything that *can* be pinned in CI.

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
            version="1.1.10",
            body="## Test\n- item one\n- item two",
            download_url="https://example.com/x.exe",
            html_url="https://github.com/example/example/releases/tag/v1.1.10",
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

    def test_dialog_keeps_stays_on_top_and_frameless_flags(self) -> None:
        """1.1.7 regression: r51 indigo chrome sets FramelessWindowHint,
        which on Windows hid the dialog behind the parent. The 1.1.8.1
        fix adds WindowStaysOnTopHint AFTER chrome. If either flag is
        missing after construction, do not ship — isVisible() on the
        offscreen platform would still pass."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QWidget
        from hgr.app.updater.update_dialog import UpdateDialog

        parent = QWidget()
        parent.show()
        self._app.processEvents()
        dlg = UpdateDialog(self._make_info(), parent=parent)
        try:
            dlg.show()
            self._app.processEvents()
            flags = dlg.windowFlags()
            self.assertTrue(
                bool(flags & Qt.FramelessWindowHint),
                "indigo chrome must still set FramelessWindowHint",
            )
            self.assertTrue(
                bool(flags & Qt.WindowStaysOnTopHint),
                "WindowStaysOnTopHint must survive install_indigo_chrome "
                "(1.1.7 / 1.1.8 invisible-popup regression).",
            )
            self.assertTrue(hasattr(dlg, "_force_to_front"))
            self.assertTrue(dlg.isVisible())
        finally:
            dlg.close()
            dlg.deleteLater()
            parent.close()
            parent.deleteLater()

    def test_dialog_download_button_is_usable(self) -> None:
        """Human smoke test also requires a clickable Download Update
        button with the new version in the title. Pin that in CI so a
        construction-only pass can't hide a missing primary action."""
        from hgr.app.updater.update_dialog import UpdateDialog
        dlg = UpdateDialog(self._make_info(version="1.1.10", size_bytes=140_000_000), parent=None)
        try:
            dlg.show()
            self._app.processEvents()
            self.assertTrue(hasattr(dlg, "download_button"))
            self.assertEqual(dlg.download_button.text(), "Download Update")
            self.assertTrue(dlg.download_button.isEnabled())
            from PySide6.QtWidgets import QLabel
            labels = [w.text() for w in dlg.findChildren(QLabel) if w.text()]
            self.assertTrue(
                any("1.1.10" in text for text in labels),
                f"version string must appear in the dialog labels, got {labels!r}",
            )
            self.assertTrue(
                any("MB" in text or "download" in text.lower() for text in labels),
                f"size/kind subtitle must appear in the dialog labels, got {labels!r}",
            )
        finally:
            dlg.close()
            dlg.deleteLater()

    def test_tray_icon_exposes_show_message(self) -> None:
        """The 1.1.7 / 1.1.8 safety net called showMessage on
        TouchlessTrayIcon. That wrapper had no such method, so
        hasattr(...) was False and the balloon never fired. This
        assertion is the pin — do not delete it."""
        from PySide6.QtGui import QIcon
        from hgr.app.ui.tray_icon import TouchlessTrayIcon

        tray = TouchlessTrayIcon(QIcon())
        self.assertTrue(
            hasattr(tray, "showMessage"),
            "TouchlessTrayIcon.showMessage must exist so MainWindow's "
            "update-popup balloon fallback actually runs.",
        )
        self.assertTrue(hasattr(tray, "message_clicked"))
        tray.showMessage("t", "m", msecs=1)

    def test_on_update_available_fires_tray_balloon(self) -> None:
        """Both the website and Store branches of _on_update_available
        must attempt a tray balloon. A refactor that drops showMessage
        from either path re-creates the 1.1.8.1 Store-cohort gap."""
        import inspect
        from hgr.app.ui.main_window import MainWindow

        src = inspect.getsource(MainWindow._on_update_available)
        self.assertGreaterEqual(
            src.count("showMessage"),
            2,
            "_on_update_available must call showMessage on both the "
            "Store branch and the website-installer branch",
        )
        inspect.getsource(MainWindow._on_tray_message_clicked)
        self.assertTrue(callable(MainWindow._on_tray_message_clicked))


if __name__ == "__main__":
    unittest.main()

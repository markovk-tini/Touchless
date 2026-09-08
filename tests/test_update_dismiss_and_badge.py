"""Dismiss semantics + pending-update badge sentinels.

Why this file exists: closing the update prompt with the title-bar X
used to emit the same `dismissed` signal as clicking Later, so both
paths persisted `last_dismissed_update_version` and the update went
permanently quiet. Users reach for the X to clear the prompt off the
screen right now, not to decline the release, so the only surviving
cue was a Settings panel they had no reason to open.

The split behaviour these tests pin:
  * Later  -> `dismissed`  -> suppressed for this version, badge stays.
  * X/Esc  -> `deferred`   -> NOT suppressed, prompt returns next launch.
  * mid-download close -> neither signal (the update is already being
    applied; treating that as a decline would be wrong).

Plus the nav badge toggle, so the green "!" pip on the Updates row
can't silently stop rendering.

Referenced by PUBLISHING_POLICY.md as a STOP-SHIP rule.
"""
from __future__ import annotations

import os
import unittest

# Qt has to run headless on CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _SignalSpy:
    """Minimal counter so the tests don't depend on QSignalSpy's
    version-dependent constructor overloads."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *_args) -> None:
        self.count += 1


class UpdateDismissSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def _make_dialog(self):
        from hgr.app.updater.release_checker import ReleaseInfo
        from hgr.app.updater.update_dialog import UpdateDialog
        info = ReleaseInfo(
            version="1.1.10",
            body="## Test\n- one",
            download_url="https://example.com/Touchless_App_Update_1.1.10.zip",
            html_url="https://example.com/releases/tag/v1.1.10",
            size_bytes=142_262_058,
            update_kind="app-zip",
        )
        return UpdateDialog(info, parent=None)

    def test_later_emits_dismissed_only(self) -> None:
        dlg = self._make_dialog()
        dismissed, deferred = _SignalSpy(), _SignalSpy()
        dlg.dismissed.connect(dismissed)
        dlg.deferred.connect(deferred)
        try:
            dlg.later_button.click()
            self._app.processEvents()
            self.assertEqual(dismissed.count, 1, "Later must emit dismissed()")
            self.assertEqual(
                deferred.count, 0,
                "Later must NOT emit deferred() — it is an explicit "
                "'stop asking about this version'.",
            )
        finally:
            dlg.deleteLater()

    def test_close_emits_deferred_not_dismissed(self) -> None:
        """The regression this file was created for. If `dismissed`
        fires here, closing the window silently suppresses the update
        again — do not ship."""
        dlg = self._make_dialog()
        dismissed, deferred = _SignalSpy(), _SignalSpy()
        dlg.dismissed.connect(dismissed)
        dlg.deferred.connect(deferred)
        try:
            dlg.close()
            self._app.processEvents()
            self.assertEqual(
                dismissed.count, 0,
                "Closing with the X must NOT emit dismissed() — that "
                "persists last_dismissed_update_version and kills the "
                "prompt for good.",
            )
            self.assertEqual(deferred.count, 1, "X must emit deferred() exactly once")
        finally:
            dlg.deleteLater()

    def test_reject_emits_deferred_exactly_once(self) -> None:
        """Esc routes through reject(), and the chrome X arrives as
        close() -> closeEvent -> Qt's own reject(). One user action must
        still produce exactly one signal."""
        dlg = self._make_dialog()
        deferred = _SignalSpy()
        dlg.deferred.connect(deferred)
        try:
            dlg.reject()
            self._app.processEvents()
            self.assertEqual(deferred.count, 1)
            # A second close must not re-fire it.
            dlg.close()
            self._app.processEvents()
            self.assertEqual(
                deferred.count, 1,
                "deferred() must be emitted at most once per dialog",
            )
        finally:
            dlg.deleteLater()

    def test_mid_download_close_emits_neither(self) -> None:
        dlg = self._make_dialog()
        dismissed, deferred = _SignalSpy(), _SignalSpy()
        dlg.dismissed.connect(dismissed)
        dlg.deferred.connect(deferred)
        try:
            dlg.download_button.click()   # disables the buttons
            self._app.processEvents()
            dlg.close()
            self._app.processEvents()
            self.assertEqual(dismissed.count, 0)
            self.assertEqual(
                deferred.count, 0,
                "Closing the window while the download runs must not be "
                "recorded as declining the update.",
            )
        finally:
            dlg.deleteLater()

    def test_download_click_emits_download_requested(self) -> None:
        """Non-regression: splitting the dismiss paths must not disturb
        the primary action."""
        dlg = self._make_dialog()
        requested = _SignalSpy()
        dlg.download_requested.connect(requested)
        try:
            dlg.download_button.click()
            self._app.processEvents()
            self.assertEqual(requested.count, 1)
        finally:
            dlg.deleteLater()


class UpdateNavBadgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def test_nav_button_badge_toggles(self) -> None:
        from hgr.app.ui.main_window import SettingsNavButton
        button = SettingsNavButton("Updates", 9, None)
        try:
            self.assertFalse(button.has_update_badge(), "badge starts hidden")
            button.set_update_badge(True)
            self.assertTrue(button.has_update_badge())
            button.set_update_badge(False)
            self.assertFalse(button.has_update_badge())
        finally:
            button.deleteLater()

    def test_nav_button_paints_with_badge_on(self) -> None:
        """The pip is painted, not a child widget, so a bad QPainter
        call would raise only at paint time."""
        from PySide6.QtGui import QPixmap
        from hgr.app.ui.main_window import SettingsNavButton
        button = SettingsNavButton("Updates", 9, None)
        try:
            button.set_update_badge(True)
            button.resize(180, 44)
            pixmap = QPixmap(button.size())
            button.render(pixmap)   # raises if paintEvent throws
        finally:
            button.deleteLater()

    def test_bounce_cue_stays_inside_shrunken_viewport(self) -> None:
        """Window shrink used to park the cue below the nav, so it
        vanished. After a resize it must still sit fully inside the
        viewport, and on the right (aligned with the Updates pip),
        not the old centered spot."""
        from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget
        from hgr.app.ui.main_window import (
            MainWindow,
            SettingsNavButton,
            _UpdateNavHintArrow,
        )

        class _Stub:
            _refresh_update_badges = MainWindow._refresh_update_badges
            _is_nav_button_in_view = MainWindow._is_nav_button_in_view
            _position_update_nav_hint = MainWindow._position_update_nav_hint

        page = QWidget()
        page.resize(900, 600)
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        buttons = [SettingsNavButton(f"Row {i}", i, None) for i in range(11)]
        for b in buttons:
            b.setFixedHeight(44)
            lay.addWidget(b)
        scroll.setWidget(inner)
        scroll.setGeometry(10, 10, 200, 280)
        page.show()
        self._app.processEvents()

        stub = _Stub()
        stub.settings_page = page
        stub._settings_nav_scroll = scroll
        stub._updates_nav_button = buttons[9]
        stub._update_nav_hint = _UpdateNavHintArrow(scroll.viewport())
        stub._pending_update_version = "1.1.9.2"
        stub._refresh_update_badges()
        self._app.processEvents()

        hint = stub._update_nav_hint
        vp = scroll.viewport()
        self.assertTrue(hint.isVisible(), "cue must show when Updates is off-screen")
        self.assertGreater(
            hint.x(),
            vp.width() // 2,
            "cue must sit on the right half of the nav, not centered",
        )
        hint_rect = hint.geometry()
        self.assertTrue(
            vp.rect().contains(hint_rect),
            f"cue {hint_rect} must start inside viewport {vp.rect()}",
        )

        # Shrink the nav the way a short window does.
        scroll.setGeometry(10, 10, 200, 90)
        self._app.processEvents()
        stub._refresh_update_badges()
        self._app.processEvents()
        hint_rect = hint.geometry()
        self.assertTrue(hint.isVisible(), "cue must survive a window shrink")
        self.assertTrue(
            vp.rect().contains(hint_rect),
            f"after shrink, cue {hint_rect} must stay inside viewport {vp.rect()}",
        )
        page.deleteLater()

    def test_simulated_version_bump(self) -> None:
        from hgr.app.ui.main_window import MainWindow
        self.assertEqual(MainWindow._next_simulated_version("1.1.9.1"), "1.1.9.2")
        self.assertEqual(MainWindow._next_simulated_version("1.1.9"), "1.1.10")
        self.assertEqual(MainWindow._next_simulated_version(""), "9.9.9")

    def test_source_run_defaults_to_simulated_prompt(self) -> None:
        """A plain `python run_app.py` must take the fake-prompt path.
        Frozen builds must not, even if the env var is set."""
        import sys
        from unittest.mock import patch
        from hgr.app.ui.main_window import MainWindow

        class _Stub:
            pass

        stub = _Stub()
        called = {"n": 0}
        stub._start_simulated_update_check = lambda _raw: called.__setitem__(
            "n", called["n"] + 1
        )

        os.environ.pop("TOUCHLESS_SIMULATE_UPDATE", None)
        with patch.object(sys, "frozen", False, create=True):
            MainWindow._kick_off_update_check(stub)
        self.assertEqual(called["n"], 1, "source run must simulate by default")

        called["n"] = 0
        with patch.object(sys, "frozen", True, create=True):
            with patch.dict("os.environ", {"TOUCHLESS_SIMULATE_UPDATE": "1"}):
                with patch(
                    "hgr.utils.runtime_paths.build_channel",
                    return_value="website",
                ):
                    with patch(
                        "hgr.app.updater.ReleaseChecker",
                        side_effect=RuntimeError("frozen must not simulate"),
                    ):
                        try:
                            MainWindow._kick_off_update_check(stub)
                        except RuntimeError:
                            pass
        self.assertEqual(
            called["n"], 0,
            "frozen builds must never take the simulated-update path",
        )


if __name__ == "__main__":
    unittest.main()

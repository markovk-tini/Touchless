"""Opaque settings snapshots must not be transparent.

Control Guide section widgets stay collapsed until the user opens them.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class SettingsPageSwapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_reveal_shows_hidden_body_of_stack_page(self) -> None:
        from PySide6.QtWidgets import QFrame, QLabel, QStackedWidget, QVBoxLayout

        from hgr.app.ui.main_window import MainWindow

        panel = QFrame()
        layout = QVBoxLayout(panel)
        title = QLabel("General")
        title.setObjectName("settingsPanelTitle")
        subtitle = QLabel("Tweak how Touchless behaves")
        subtitle.setObjectName("settingsPanelSubtitle")
        body = QLabel("heavy body")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(body)

        stack = QStackedWidget()
        stack.addWidget(QLabel("other"))
        stack.addWidget(panel)
        stack.setCurrentIndex(0)
        body.hide()

        self.assertTrue(body.isHidden())
        MainWindow._reveal_settings_panel_body(object(), panel)
        self.assertFalse(body.isHidden())

    def test_gpu_ffmpeg_defaults_to_720p_60(self) -> None:
        import inspect

        from hgr.app.camera.ffmpeg_capture import FfmpegMjpegCapture

        sig = inspect.signature(FfmpegMjpegCapture.__init__)
        self.assertEqual(sig.parameters["width"].default, 1280)
        self.assertEqual(sig.parameters["height"].default, 720)
        self.assertEqual(sig.parameters["fps"].default, 60)

    def test_gpu_video_widget_keeps_numpy_buffer(self) -> None:
        import numpy as np

        from hgr.app.ui.gpu_video_widget import GpuVideoWidget

        widget = GpuVideoWidget()
        try:
            widget.show()
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            widget.update_frame(frame)
            self.assertIsNotNone(widget._image)
            self.assertEqual(widget._image.width(), 64)
            self.assertEqual(widget._image.height(), 48)
            self.assertIs(widget._frame_keep, frame)
        finally:
            widget.close()
            widget.deleteLater()

    def test_live_scroll_policy_skips_layout(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock

        from hgr.app.ui.main_window import MainWindow

        mw = MainWindow.__new__(MainWindow)
        mw._worker = SimpleNamespace(is_running=True)
        mw._gesture_binds_pending_action = None
        mw._gesture_binds_pill = None
        mw._gesture_binds_pill_warning = None
        mw._settings_content_scroll = Mock()
        MainWindow._apply_settings_outer_scroll_policy(mw, 0)
        mw._settings_content_scroll.setVerticalScrollBarPolicy.assert_not_called()

    def test_opaque_pixmap_has_no_transparent_pixels(self) -> None:
        from PySide6.QtGui import QColor, QPixmap
        from PySide6.QtCore import Qt

        from hgr.app.ui.main_window import MainWindow

        class _Host:
            def _settings_overlay_fill_color(self):
                return QColor("#0F172A")

        src = QPixmap(120, 80)
        src.fill(Qt.transparent)
        out = MainWindow._settings_opaque_pixmap(_Host(), src)
        img = out.toImage()
        self.assertFalse(img.isNull())
        self.assertEqual(img.pixelColor(0, 0).alpha(), 255)
        self.assertEqual(img.pixelColor(60, 40).alpha(), 255)

    def test_control_guide_section_stays_collapsed_after_addwidget(self) -> None:
        from hgr.app.ui.main_window import GestureGuideSection

        section = GestureGuideSection("Static Gestures", [])
        try:
            self.assertTrue(section.content.isHidden())
        finally:
            section.close()
            section.deleteLater()

    def test_idle_snapshot_skips_while_engine_running(self) -> None:
        from types import SimpleNamespace

        from hgr.app.ui.main_window import MainWindow

        mw = MainWindow.__new__(MainWindow)
        mw._worker = SimpleNamespace(is_running=True)
        MainWindow._idle_snapshot_current_settings_page(mw)
        MainWindow._store_settings_page_snapshot(mw, 0, True)

    def test_live_engine_skips_accurate_height_walk(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from PySide6.QtWidgets import QLabel, QWidget

        from hgr.app.ui.main_window import _CurrentSizedStack

        class Host(QWidget):
            def __init__(self) -> None:
                super().__init__()
                self._worker = SimpleNamespace(is_running=True)

        host = Host()
        stack = _CurrentSizedStack(host)
        page = QLabel("camera")
        page.setProperty("useAccurateHeight", True)
        stack.addWidget(page)
        stack.setCurrentWidget(page)
        stack._cheap_hint = False
        stack._hint_cache.clear()
        try:
            with patch.object(stack, "_true_content_height") as walk:
                stack.sizeHint()
                walk.assert_not_called()
        finally:
            page.deleteLater()
            stack.deleteLater()
            host.deleteLater()

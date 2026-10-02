"""Control Guide video cards must not create QMediaPlayer during init.

Constructing ~16 QVideoWidget + setSource() at settings-build time
mapped native decoder surfaces when the stacked page was shown and
froze tab switches on a still frame of the previous page.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ControlGuideVideoDeferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_video_card_does_not_construct_player_in_init(self) -> None:
        from hgr.app.ui.main_window import GestureMediaWidget

        widget = GestureMediaWidget(
            video_name="Mouse Demo.mp4",
            gesture_key="open_hand",
        )
        try:
            self.assertIsNone(widget._player)
            self.assertIsNone(widget._video_widget)
            if widget._pending_video_path is None:
                self.skipTest("GestureGuide clip not found or Qt Multimedia missing")
            self.assertIsNotNone(widget._video_placeholder)
        finally:
            widget.close()
            widget.deleteLater()

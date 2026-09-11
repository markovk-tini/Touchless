from __future__ import annotations

import sys

from PySide6.QtCore import QPropertyAnimation, QTimer, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_AUTO_DISMISS_MS = 10_000
_FADE_MS = 450


class SpotifyConnectPromptOverlay(QWidget):
    """Top-right toast asking the user to connect / reconnect Spotify.

    Matches the LowFpsSuggestionOverlay pattern (same window flags, same
    top-right placement, same fade animation, same X-close, same
    dimensions family) so both toasts read as one system.

    Differences from LowFpsSuggestionOverlay:
      * Body is rich text with a clickable "here" link — the link IS
        the primary action, so no separate action button.
      * Adds a "Don't show me again this session" checkbox at the
        bottom. Checked state is surfaced on `dismissed(bool)` so the
        caller can latch a session-scoped suppression flag.
      * Body text is passed in via `show_prompt(body_html)` so the same
        widget instance can surface different reasons (needs reauth vs
        no active device vs generic).
    """

    linkClicked = Signal()
    dismissed = Signal(bool)  # payload: user checked "don't show this session"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet("background: transparent;")

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade.setDuration(_FADE_MS)
        self._fade.finished.connect(self._on_fade_finished)

        self._auto_dismiss = QTimer(self)
        self._auto_dismiss.setSingleShot(True)
        self._auto_dismiss.timeout.connect(self._begin_fade_out)

        self._fading_out = False

        self._build_ui()
        self.setFixedSize(420, 172)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        panel = QFrame()
        panel.setObjectName("spotifyPromptPanel")
        panel.setStyleSheet(
            """
            QFrame#spotifyPromptPanel {
                background: rgba(16, 70, 132, 220);
                border: 1px solid rgba(120, 190, 255, 180);
                border-radius: 14px;
            }
            QLabel#spotifyPromptTitle {
                color: #E8F3FF;
                font-size: 15px;
                font-weight: 700;
                background: transparent;
            }
            QLabel#spotifyPromptBody {
                color: #E8F3FF;
                font-size: 16px;
                background: transparent;
            }
            QPushButton#spotifyPromptClose {
                color: #D4E8FF;
                background: transparent;
                border: none;
                font-size: 22px;
                font-weight: 700;
                padding: 0;
            }
            QPushButton#spotifyPromptClose:hover {
                color: #FFFFFF;
            }
            QCheckBox#spotifyPromptSuppress {
                color: #D4E8FF;
                font-size: 12px;
                background: transparent;
                spacing: 8px;
            }
            QCheckBox#spotifyPromptSuppress::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid rgba(212, 232, 255, 180);
                border-radius: 3px;
                background: transparent;
            }
            QCheckBox#spotifyPromptSuppress::indicator:hover {
                border-color: #FFFFFF;
            }
            QCheckBox#spotifyPromptSuppress::indicator:checked {
                background: #1DE9B6;
                border-color: #1DE9B6;
            }
            """
        )
        outer.addWidget(panel)

        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 8, 10, 10)
        panel_layout.setSpacing(2)

        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        title = QLabel("Setup / Reconnect Spotify")
        title.setObjectName("spotifyPromptTitle")
        header_row.addWidget(title, 1)
        close_btn = QPushButton("×")
        close_btn.setObjectName("spotifyPromptClose")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setFixedSize(28, 28)
        close_btn.setToolTip("Close")
        close_btn.clicked.connect(self._on_close_clicked)
        header_row.addWidget(close_btn, 0, Qt.AlignRight | Qt.AlignTop)
        panel_layout.addLayout(header_row)

        self._body = QLabel(
            "Connect to Spotify to control your music with Touchless "
            "— click <a href='#connect'>here</a>."
        )
        self._body.setObjectName("spotifyPromptBody")
        self._body.setWordWrap(True)
        self._body.setTextFormat(Qt.RichText)
        self._body.setOpenExternalLinks(False)
        self._body.linkActivated.connect(self._on_link_clicked)
        panel_layout.addWidget(self._body, 1)

        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.setSpacing(6)
        self._suppress_checkbox = QCheckBox("Don't show me again this session")
        self._suppress_checkbox.setObjectName("spotifyPromptSuppress")
        self._suppress_checkbox.setCursor(Qt.PointingHandCursor)
        footer_row.addWidget(self._suppress_checkbox, 0, Qt.AlignLeft | Qt.AlignVCenter)
        footer_row.addStretch(1)
        panel_layout.addLayout(footer_row)

    def show_prompt(self, body_html: str) -> None:
        """Position in the top-right of the primary screen and fade in.
        `body_html` should include a `<a href='#connect'>...</a>` anchor
        somewhere — its click emits linkClicked and dismisses the pill.
        """
        self._fading_out = False
        try:
            self._body.setText(str(body_html or ""))
        except Exception:
            pass
        # Reset the suppress checkbox each show so a prior "checked
        # then closed" state doesn't leak into an intentional re-show
        # (the caller's own session flag handles suppression).
        try:
            self._suppress_checkbox.setChecked(False)
        except Exception:
            pass
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            margin = 24
            x = geo.right() - self.width() - margin
            y = geo.top() + margin
            self.move(x, y)
        self._opacity_effect.setOpacity(0.0)
        self.show()
        if sys.platform != "darwin":
            self.raise_()
        else:
            try:
                from ..app.ui.native_overlay import apply_overlay
                apply_overlay(self)
            except Exception:
                pass
        self._fade.stop()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()
        self._auto_dismiss.start(_AUTO_DISMISS_MS)

    def dismiss(self) -> None:
        if not self.isVisible() or self._fading_out:
            return
        self._begin_fade_out()

    def hide_immediately(self) -> None:
        """Drop the toast now (wizard is about to exec() and would
        freeze a fading Tool window on top of the dialog)."""
        self._auto_dismiss.stop()
        self._fade.stop()
        self._fading_out = False
        if self.isVisible():
            self.hide()
            try:
                suppress = bool(self._suppress_checkbox.isChecked())
            except Exception:
                suppress = False
            self.dismissed.emit(suppress)

    def _on_link_clicked(self, _href: str) -> None:
        # Hide before the wizard exec() so the toast isn't stuck on
        # top of a modal dialog with no way to dismiss it.
        self.hide_immediately()
        self.linkClicked.emit()

    def _on_close_clicked(self) -> None:
        self._auto_dismiss.stop()
        self._begin_fade_out()

    def _begin_fade_out(self) -> None:
        if self._fading_out:
            return
        self._fading_out = True
        self._auto_dismiss.stop()
        self._fade.stop()
        self._fade.setStartValue(float(self._opacity_effect.opacity()))
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _on_fade_finished(self) -> None:
        if self._fading_out:
            self.hide()
            self._fading_out = False
            try:
                suppress = bool(self._suppress_checkbox.isChecked())
            except Exception:
                suppress = False
            self.dismissed.emit(suppress)

# Author: Konstantin Markov

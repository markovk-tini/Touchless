"""Modeless "which drawing did you mean?" picker.

Fired when the `show_overlay_drawing` action's filesystem search
returns more than one match for a bare filename. Each match gets
a one-letter shortcut (A, B, C, ...) shown alongside the file's
parent directory so the user can tell which copy they want.

Selection sources:
  - Mouse / hand-cursor click on a row.
  - Programmatic `select_by_letter(letter)` — called by the voice
    listener when the utterance maps to a shortcut letter (e.g.
    "A", "show me B", "let's do C").

Constraints enforced (matches CLAUDE.md "secondary windows" rule):
  - Modeless: `show()`, never `exec()`. The engine and live camera
    keep running while this window is open.
  - Always-on-top so it sits above whatever foreground app the
    user is in.
  - Auto-dismisses after a configurable timeout so an accidental
    fire doesn't strand a UI element.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


DEFAULT_AUTO_DISMISS_MS = 15_000

# A..J — caps the visible options to the search helper's result
# limit. More than 10 should never happen.
_LETTERS = tuple("ABCDEFGHIJ")


class DrawingChooserPill(QWidget):
    """Single-instance disambiguation chooser. Built once per
    main window and re-populated on each fire."""

    # Emitted when the user picks a row (click or voice).
    # Argument: the absolute Path of the chosen file.
    choice_made = Signal(object)
    # Emitted on auto-dismiss timeout, explicit close, or Escape.
    cancelled = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._candidates: List[Path] = []
        self._buttons: List[QPushButton] = []
        self._auto_dismiss_timer = QTimer(self)
        self._auto_dismiss_timer.setSingleShot(True)
        self._auto_dismiss_timer.timeout.connect(self._on_timeout)

        # ---- Visual frame ----
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._card = QFrame(self)
        self._card.setObjectName("chooserCard")
        outer.addWidget(self._card)

        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(10)

        self._title_label = QLabel("Multiple drawings found")
        title_font = QFont("Segoe UI", 11)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._title_label.setStyleSheet("color: #FFFFFF;")
        card_layout.addWidget(self._title_label)

        self._subtitle_label = QLabel("")
        subtitle_font = QFont("Segoe UI", 9)
        self._subtitle_label.setFont(subtitle_font)
        self._subtitle_label.setStyleSheet("color: #B8C2D6;")
        self._subtitle_label.setWordWrap(True)
        card_layout.addWidget(self._subtitle_label)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        card_layout.addLayout(self._rows_layout)

        self._hint_label = QLabel(
            "Click a row, or say the letter (e.g. \"A\", \"let's do B\")."
        )
        hint_font = QFont("Segoe UI", 8)
        self._hint_label.setFont(hint_font)
        self._hint_label.setStyleSheet("color: #7F8EAB;")
        card_layout.addWidget(self._hint_label)

        # Dark card with rounded corners + subtle border. Buttons
        # use the same accent green as the rest of the app on hover.
        self.setStyleSheet(
            "QFrame#chooserCard {"
            "  background-color: #1B2236;"
            "  border: 1px solid #2F3A55;"
            "  border-radius: 12px;"
            "}"
            "QPushButton.choiceRow {"
            "  background-color: #232C44;"
            "  border: 1px solid #2F3A55;"
            "  border-radius: 8px;"
            "  color: #E5F6FF;"
            "  padding: 8px 12px;"
            "  text-align: left;"
            "  font-family: 'Segoe UI';"
            "  font-size: 10pt;"
            "}"
            "QPushButton.choiceRow:hover {"
            "  background-color: #2A3553;"
            "  border-color: #1DE9B6;"
            "}"
        )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def show_choices(
        self,
        filename: str,
        candidates: List[Path],
        *,
        auto_dismiss_ms: int = DEFAULT_AUTO_DISMISS_MS,
    ) -> None:
        """Populate the chooser with `candidates` and display it.
        Repopulates on every call so the pill can be reused across
        fires without leaking old buttons."""
        self._candidates = list(candidates)[: len(_LETTERS)]
        self._subtitle_label.setText(
            f"There are {len(self._candidates)} files called "
            f"\"{filename}\". Which one do you want to show?"
        )
        self._clear_rows()
        for i, path in enumerate(self._candidates):
            row = self._build_row(_LETTERS[i], path)
            self._buttons.append(row)
            self._rows_layout.addWidget(row)

        self._place_on_screen()
        self.show()
        self.raise_()
        if auto_dismiss_ms > 0:
            self._auto_dismiss_timer.start(int(auto_dismiss_ms))
        else:
            self._auto_dismiss_timer.stop()

    def select_by_letter(self, letter: str) -> bool:
        """Programmatic select (voice command). Returns True if the
        letter mapped to a visible row and the choice was emitted."""
        if not letter:
            return False
        ch = letter.strip().upper()[:1]
        if ch not in _LETTERS:
            return False
        idx = _LETTERS.index(ch)
        if idx >= len(self._candidates):
            return False
        self._fire_choice(idx)
        return True

    def is_active(self) -> bool:
        return self.isVisible() and len(self._candidates) > 0

    def visible_letters(self) -> List[str]:
        """Letters that map to live rows right now. Used by the
        voice listener to scope its choice-mode regex."""
        return [_LETTERS[i] for i in range(len(self._candidates))]

    def cancel(self) -> None:
        """Public counterpart of the auto-dismiss path. Closes the
        pill without picking anything and fires `cancelled`."""
        if not self.isVisible():
            return
        self._auto_dismiss_timer.stop()
        self.hide()
        self._candidates = []
        self.cancelled.emit()

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------

    def _clear_rows(self) -> None:
        for btn in self._buttons:
            btn.setParent(None)
            btn.deleteLater()
        self._buttons.clear()

    def _build_row(self, letter: str, path: Path) -> QPushButton:
        # Two-column layout inside the button: letter + filename on
        # the left, parent directory on the right.
        parent = str(path.parent)
        text = f"{letter}    {path.name}        {parent}"
        btn = QPushButton(text)
        btn.setProperty("class", "choiceRow")
        btn.setObjectName(f"choice_{letter}")
        # Style class is set via dynamic property → need to repolish
        # so the QSS .choiceRow rule actually applies.
        btn.setProperty("class", "choiceRow")
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        # Manually set the class via direct stylesheet because Qt's
        # dotted-class selector is unreliable on dynamic properties.
        btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #232C44;"
            "  border: 1px solid #2F3A55;"
            "  border-radius: 8px;"
            "  color: #E5F6FF;"
            "  padding: 8px 12px;"
            "  text-align: left;"
            "  font-family: 'Segoe UI';"
            "  font-size: 10pt;"
            "}"
            "QPushButton:hover {"
            "  background-color: #2A3553;"
            "  border-color: #1DE9B6;"
            "}"
        )
        btn.setCursor(Qt.PointingHandCursor)
        idx = len(self._buttons)
        btn.clicked.connect(lambda _checked=False, i=idx: self._fire_choice(i))
        return btn

    def _fire_choice(self, idx: int) -> None:
        if not (0 <= idx < len(self._candidates)):
            return
        chosen = self._candidates[idx]
        self._auto_dismiss_timer.stop()
        self.hide()
        # Capture before clearing so the signal carries the path
        # even after the chooser has reset.
        self._candidates = []
        self.choice_made.emit(chosen)

    def _on_timeout(self) -> None:
        if not self.isVisible():
            return
        self.hide()
        self._candidates = []
        self.cancelled.emit()

    def _place_on_screen(self) -> None:
        """Centre horizontally near the bottom of the primary
        screen, leaving room above the taskbar."""
        self.adjustSize()
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.center().x() - self.width() // 2
        y = geo.bottom() - self.height() - 80
        self.move(x, y)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.cancel()
            return
        # Keyboard letter shortcut for users who'd rather type than
        # voice / click.
        text = event.text().upper()
        if text and text in _LETTERS:
            if self.select_by_letter(text):
                return
        super().keyPressEvent(event)

# Author: Konstantin Markov

"""Modal A/B/C disambiguation chooser used at gesture-creation time.

Companion to DrawingChooserPill (modeless, used at fire time). This
variant is a proper QDialog so the wizard flow can `exec()` it
inline: the user picks (or cancels) before _on_start returns, and
the wizard then proceeds to recording with the resolved absolute
path baked into the action payload.

Why the wizard-time variant is separate from the fire-time pill:
  - At wizard time the engine is usually paused, so voice
    intercept and always-on-top behaviour are unnecessary.
  - Modal exec() keeps validation flow synchronous — no callback
    plumbing back into _on_start, no held-open wizard state.
  - The yellow banner already inside the wizard does the
    "multiple files" announce; the dialog itself just shows the
    list and waits for a click.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


_LETTERS = tuple("ABCDEFGHIJ")


class DrawingChooserDialog(QDialog):
    """Modal A/B/C chooser. On accept(), `chosen_path` holds the
    selected absolute path; on reject() it stays None."""

    def __init__(
        self,
        filename: str,
        candidates: List[Path],
        parent=None,
    ) -> None:
        super().__init__(parent)
        from .window_chrome import apply_touchless_chrome
        apply_touchless_chrome(self)
        self.setWindowTitle("Select a drawing")
        self.setModal(True)
        # No close/minimize chrome — Cancel button does it cleanly.
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        self.chosen_path: Optional[Path] = None
        self._candidates: List[Path] = list(candidates)[: len(_LETTERS)]

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(12)

        title = QLabel(
            f"There are {len(self._candidates)} files named "
            f"“{filename}” — pick the one to bind:"
        )
        title.setWordWrap(True)
        tf = QFont("Segoe UI", 10)
        tf.setBold(True)
        title.setFont(tf)
        outer.addWidget(title)

        for i, path in enumerate(self._candidates):
            outer.addWidget(self._build_row(i, _LETTERS[i], path))

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setAutoDefault(False)
        cancel.setDefault(False)
        cancel.clicked.connect(self.reject)
        button_row.addWidget(cancel)
        outer.addLayout(button_row)

        # Yellow-tinted styling so the dialog visually matches the
        # warning banner the wizard puts up at the same time.
        self.setStyleSheet(
            "QDialog {"
            "  background-color: #1B2236;"
            "}"
            "QLabel { color: #E5F6FF; }"
            "QPushButton {"
            "  background-color: #232C44;"
            "  border: 1px solid #2F3A55;"
            "  border-radius: 8px;"
            "  color: #E5F6FF;"
            "  padding: 8px 14px;"
            "  font-family: 'Segoe UI';"
            "  font-size: 10pt;"
            "}"
            "QPushButton:hover {"
            "  background-color: #2A3553;"
            "  border-color: #F5B450;"
            "}"
            "QPushButton.choiceRow {"
            "  text-align: left;"
            "  padding: 10px 14px;"
            "}"
        )

    def _build_row(self, index: int, letter: str, path: Path) -> QPushButton:
        parent_dir = str(path.parent)
        text = f"{letter}    {path.name}        {parent_dir}"
        btn = QPushButton(text)
        btn.setProperty("class", "choiceRow")
        btn.setObjectName(f"choice_{letter}")
        btn.setAutoDefault(False)
        btn.setDefault(False)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #232C44;"
            "  border: 1px solid #2F3A55;"
            "  border-radius: 8px;"
            "  color: #E5F6FF;"
            "  padding: 10px 14px;"
            "  text-align: left;"
            "  font-family: 'Segoe UI';"
            "  font-size: 10pt;"
            "}"
            "QPushButton:hover {"
            "  background-color: #2A3553;"
            "  border-color: #F5B450;"
            "}"
        )
        btn.clicked.connect(lambda _checked=False, i=index: self._pick(i))
        return btn

    def _pick(self, index: int) -> None:
        if not (0 <= index < len(self._candidates)):
            return
        self.chosen_path = self._candidates[index]
        self.accept()

    def keyPressEvent(self, event) -> None:
        text = event.text().upper()
        if text and text in _LETTERS:
            idx = _LETTERS.index(text)
            if idx < len(self._candidates):
                self._pick(idx)
                return
        super().keyPressEvent(event)

# Author: Konstantin Markov

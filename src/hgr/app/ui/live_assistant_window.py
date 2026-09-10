"""Public stub — Iris is not published."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

_OFFICIAL = (
    "Iris ships in the official Touchless app.\n"
    "https://touchless-control.com"
)


class LiveAssistantWindow(QWidget):
    def __init__(self, config=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_OFFICIAL))

    def show_window(self) -> None:
        self.show()
        try:
            self.raise_()
            self.activateWindow()
        except Exception:
            pass


# Author: Konstantin Markov

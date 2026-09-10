"""Public stub for Settings → Custom Gesture."""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

_OFFICIAL = (
    "Custom gestures ship in the official Touchless app.\n"
    "https://touchless-control.com"
)


class CustomGesturesPanel(QWidget):
    open_create_requested = Signal()
    open_sandbox_requested = Signal()
    open_edit_requested = Signal(str)
    import_requested = Signal()
    export_all_requested = Signal()
    export_one_requested = Signal(str)
    tutorial_requested = Signal()

    def __init__(
        self,
        config=None,
        accent_color: str = "",
        registry_path_provider: Optional[Callable] = None,
        worker_provider: Optional[Callable] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(_OFFICIAL)
        label.setWordWrap(True)
        layout.addWidget(label)

    def refresh_cards(self) -> None:
        return None

    def set_tutorial_complete(self, _done: bool) -> None:
        return None


# Author: Konstantin Markov

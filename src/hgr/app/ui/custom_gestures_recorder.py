"""Public stub."""
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

_OFFICIAL = (
    "This editor ships in the official Touchless app.\n"
    "https://touchless-control.com"
)


class RecordingWindow(QDialog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(kwargs.get("parent"))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_OFFICIAL))


class CreateGestureWizard(QDialog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(kwargs.get("parent"))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_OFFICIAL))


class SandboxWindow(QDialog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(kwargs.get("parent"))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_OFFICIAL))


class GestureInspectDialog(QDialog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(kwargs.get("parent"))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_OFFICIAL))


# Author: Konstantin Markov

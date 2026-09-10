"""Public stub for Settings → Profiles."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

_OFFICIAL = (
    "Profiles ship in the official Touchless app.\n"
    "https://touchless-control.com"
)


class ProfilesPanel(QWidget):
    profile_changed = Signal(str)
    inspect_requested = Signal(object)
    membership_changed = Signal()
    dirty_changed = Signal(bool)

    def __init__(
        self,
        accent_color: str = "",
        text_color: str = "",
        confirm: Optional[Callable] = None,
        warn: Optional[Callable] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(_OFFICIAL)
        label.setWordWrap(True)
        layout.addWidget(label)

    def refresh(self, selected_id: Optional[str] = None) -> None:
        return None

    def pending_inspect_for(self, _item) -> Optional[Dict[str, Any]]:
        return None

    def take_pending_inspect(self) -> List:
        return []

    def commit_draft(self) -> None:
        return None

    def discard_draft(self) -> None:
        return None


class ProfileBar(QWidget):
    profile_changed = Signal(str)

    def refresh(self, selected_id: Optional[str] = None) -> None:
        return None


# Author: Konstantin Markov

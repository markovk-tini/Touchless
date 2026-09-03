"""Dialog that shows the recorded path/pose diagram for a dynamic gesture."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hgr.custom_gestures.path_preview import render_dynamic_path_preview


def show_gesture_path_preview(gesture, parent: Optional[QWidget] = None) -> None:
    """Open a modeless dialog with the path diagram for `gesture`."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"Recorded paths — {getattr(gesture, 'name', 'gesture')}")
    dlg.setModal(False)
    dlg.setMinimumSize(760, 640)
    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(10)

    state = {"mode": "takes"}

    image_label = QLabel()
    image_label.setAlignment(Qt.AlignCenter)
    image_label.setMinimumSize(720, 520)
    layout.addWidget(image_label, 1)

    hint = QLabel()
    hint.setWordWrap(True)
    hint.setStyleSheet("color: #9FB3C2; font-size: 12px;")
    layout.addWidget(hint)

    btn_row = QHBoxLayout()
    btn_row.setSpacing(8)

    avg_btn = QPushButton("Show average path")
    avg_btn.setToolTip(
        "Thick white = mean wrist path across takes. "
        "Thick amber = mean palm-center path. "
        "Faint grey lines underneath are the individual takes."
    )
    btn_row.addWidget(avg_btn)

    btn_row.addStretch(1)

    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dlg.close)
    btn_row.addWidget(close_btn)
    layout.addLayout(btn_row)

    def _refresh() -> None:
        mode = state["mode"]
        image = render_dynamic_path_preview(gesture, mode=mode)
        _set_label_bgr(image_label, image)
        if mode == "average":
            avg_btn.setText("Show all takes")
            hint.setText(
                "Average view: thick white = mean wrist path, thick amber = "
                "mean palm-center path. Faint grey lines are the individual "
                "takes. Matching does NOT use this average — it compares your "
                "live motion to the closest individual take."
            )
        else:
            avg_btn.setText("Show average path")
            hint.setText(
                "Takes view: each colored trail is one take's wrist path. "
                "Thin tip trails are take-1 fingertips. Grey / orange / red "
                "skeletons are start / mid / end poses from take 1. "
                "A clean circle should look like a closed loop."
            )

    def _toggle_average() -> None:
        state["mode"] = "average" if state["mode"] == "takes" else "takes"
        _refresh()

    avg_btn.clicked.connect(_toggle_average)
    _refresh()

    # Keep a reference on the parent so a modeless dialog isn't GC'd.
    if parent is not None:
        holders = getattr(parent, "_path_preview_dialogs", None)
        if holders is None:
            holders = []
            setattr(parent, "_path_preview_dialogs", holders)
        holders.append(dlg)
        dlg.finished.connect(lambda *_: _drop_holder(parent, dlg))

    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


def _set_label_bgr(label: QLabel, bgr: np.ndarray) -> None:
    rgb = bgr[:, :, ::-1].copy()
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    label.setPixmap(QPixmap.fromImage(qimg.copy()))


def _drop_holder(parent: QWidget, dlg: QDialog) -> None:
    holders = getattr(parent, "_path_preview_dialogs", None)
    if not holders:
        return
    try:
        holders.remove(dlg)
    except ValueError:
        pass


# Author: Konstantin Markov

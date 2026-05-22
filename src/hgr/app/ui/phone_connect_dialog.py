"""Modeless popup for the pairing-code phone-camera flow.

Shows the 6-digit code, short instructions, and a QR code that links
straight to touchless-control.com/connect. The user opens that page on
their phone (scan the QR or type the URL), enters the code, and their
camera streams to the PC over WebRTC.

Modeless by design (show(), not exec()) so the live camera / gesture
pipeline keeps running while this is open — per the project's
secondary-window rule. Themed from AppConfig so it matches the rest of
the app (surface / text / accent colours).

Author: Konstantin Markov
"""
from __future__ import annotations

import io
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

CONNECT_URL = "https://touchless-control.com/connect"


def _qr_pixmap(data: str, size: int = 190) -> Optional[QPixmap]:
    """Render `data` to a QR-code QPixmap, or None if qrcode is missing."""
    try:
        import qrcode

        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=2,
        )
        qr.add_data(data)
        qr.make(fit=True)
        # Dark modules on white so any phone camera scans it cleanly.
        img = qr.make_image(fill_color="#101820", back_color="white").convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        pix = QPixmap()
        pix.loadFromData(buf.getvalue(), "PNG")
        if pix.isNull():
            return None
        return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    except Exception:
        return None


class PhoneConnectDialog(QDialog):
    """Displays the pairing code + QR; `set_status` updates the line as
    the phone connects. Themed from AppConfig (falls back to the website
    palette if no config is available)."""

    closed = Signal()

    def __init__(self, code: str, parent=None, connect_url: str = CONNECT_URL, config=None) -> None:
        super().__init__(parent)
        cfg = config if config is not None else getattr(parent, "config", None)
        self._surface = getattr(cfg, "surface_color", "#0c2331")
        self._text = getattr(cfg, "text_color", "#f2fbff")
        self._accent = getattr(cfg, "accent_color", "#1de9b6")
        self._muted = "rgba(255,255,255,0.62)"
        self._waiting = "#f4c97b"

        self.setWindowTitle("Connect Phone")
        self.setModal(False)
        self.setMinimumWidth(440)
        self.setStyleSheet(f"PhoneConnectDialog {{ background-color: {self._surface}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 26, 28, 24)
        root.setSpacing(18)

        title = QLabel("Use your phone as a camera")
        title.setStyleSheet(f"font-size: 18px; font-weight: 800; color: {self._text};")
        root.addWidget(title)

        cols = QHBoxLayout()
        cols.setSpacing(22)

        left = QVBoxLayout()
        left.setSpacing(12)

        steps = QLabel(
            "1.  On your phone, open\n"
            "     <b>touchless-control.com/connect</b>\n"
            "     (or scan the code →)\n\n"
            "2.  Enter this pairing code:"
        )
        steps.setTextFormat(Qt.RichText)
        steps.setStyleSheet(f"color: {self._muted}; font-size: 13px;")
        steps.setWordWrap(True)
        left.addWidget(steps)

        self._code_label = QLabel(" ".join(code))
        self._code_label.setAlignment(Qt.AlignCenter)
        self._code_label.setStyleSheet(
            "font-family: 'Cascadia Mono','Consolas',monospace;"
            "font-size: 34px; font-weight: 800; letter-spacing: 6px;"
            f"color: {self._accent}; background: rgba(255,255,255,0.06);"
            "border: 1px solid rgba(255,255,255,0.16); border-radius: 12px;"
            "padding: 12px 8px;"
        )
        left.addWidget(self._code_label)

        self._status = QLabel("Waiting for your phone…")
        self._status.setStyleSheet(f"color: {self._waiting}; font-size: 13px; font-weight: 700;")
        left.addWidget(self._status)
        left.addStretch(1)
        cols.addLayout(left, 1)

        qr_box = QVBoxLayout()
        qr_box.setSpacing(6)
        qr_label = QLabel()
        qr_label.setAlignment(Qt.AlignCenter)
        pix = _qr_pixmap(connect_url)
        if pix is not None:
            qr_label.setPixmap(pix)
        else:
            qr_label.setText("(QR unavailable)")
            qr_label.setStyleSheet(f"color: {self._muted};")
        qr_box.addWidget(qr_label)
        cap = QLabel("Scan to open the page")
        cap.setAlignment(Qt.AlignCenter)
        cap.setStyleSheet(f"color: {self._muted}; font-size: 11px;")
        qr_box.addWidget(cap)
        qr_box.addStretch(1)
        cols.addLayout(qr_box, 0)

        root.addLayout(cols)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: rgba(255,255,255,0.10);")
        root.addWidget(line)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._close_btn = QPushButton("Close")
        self._close_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: rgba(255,255,255,0.08);"
            f"  color: {self._text};"
            "  border: 1px solid rgba(255,255,255,0.20);"
            "  border-radius: 10px; padding: 8px 18px; min-width: 96px;"
            "}"
            "QPushButton:hover { background-color: rgba(255,255,255,0.15); }"
        )
        self._close_btn.clicked.connect(self.close)
        btn_row.addWidget(self._close_btn)
        root.addLayout(btn_row)

    def set_status(self, text: str, *, connected: bool = False) -> None:
        self._status.setText(text)
        color = self._accent if connected else self._waiting
        self._status.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: 700;")

    def closeEvent(self, event):  # noqa: N802 (Qt API)
        self.closed.emit()
        super().closeEvent(event)

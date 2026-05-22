"""Modal that owns the phone-camera server's lifecycle for the "Connect via QR" flow.

Opens a secondary window showing a scannable QR (and the URL as text for
typing), watches server status callbacks, and lets the user commit the
phone stream as the Touchless camera source once frames are flowing. On
cancel or window-close we tear the server down so nothing keeps running
unattended.
"""
from __future__ import annotations

import io
from typing import Optional

import qrcode
from PIL import Image
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ...config.app_config import AppConfig
from ...debug.phone_camera import PhoneCameraServer


def _pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    data = pil_img.tobytes("raw", "RGBA")
    qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def _build_qr_pixmap(url: str, pixel_size: int = 320) -> QPixmap:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = img.resize((pixel_size, pixel_size), Image.NEAREST)
    return _pil_to_qpixmap(img)


class PhoneCameraConnectDialog(QDialog):
    """Runs the phone-camera server for the duration of the dialog.

    Emits `camera_accepted` when the user confirms they want to use the
    running phone stream as the Touchless camera source. If the dialog is
    canceled or closed, the server is stopped and `camera_accepted` is
    NOT emitted — the caller should keep using whatever source was
    active before.
    """

    camera_accepted = Signal(object)  # PhoneCameraServer
    # Thread-safe marshaling from the server's asyncio worker thread
    # back onto the Qt GUI thread. Direct calls into Qt widgets from a
    # non-Qt thread (or QTimer.singleShot scheduled from such a thread)
    # can silently no-op — using a real Signal with Qt.QueuedConnection
    # is the one reliable path.
    _server_status = Signal(str, object)

    def __init__(self, config: AppConfig, parent=None, *, existing_server: Optional[PhoneCameraServer] = None) -> None:
        super().__init__(parent)
        from .window_chrome import apply_touchless_chrome
        apply_touchless_chrome(self)
        self.config = config
        self._server_status.connect(self._apply_server_status)
        # When the caller already has a running server (auto-started at
        # launch for a previously-paired phone), reuse it so clicking
        # Cancel doesn't tear down the background listener the user is
        # expecting to keep running. Hook up our own status callback so
        # UI updates still flow to this dialog.
        self._server_is_borrowed = existing_server is not None and existing_server.is_running
        if self._server_is_borrowed:
            self._server = existing_server
            self._server.set_status_callback(self._forward_server_status)
        else:
            self._server = PhoneCameraServer(port=8765, on_status=self._forward_server_status)
        self._server_info = None
        self._streaming_seen = False
        self._committed = False

        self.setWindowTitle("Connect Phone Camera")
        self.setModal(True)
        self.setMinimumWidth(460)
        self._apply_theme()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("Scan this QR code with your phone camera")
        title.setObjectName("phoneDialogTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        subtitle = QLabel(
            "Your phone must be on the same WiFi network as this PC. Open the QR in your phone's default browser."
        )
        subtitle.setObjectName("phoneDialogSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self.qr_label = QLabel()
        self.qr_label.setAlignment(Qt.AlignCenter)
        self.qr_label.setMinimumHeight(320)
        layout.addWidget(self.qr_label)

        self.url_label = QLabel("starting server...")
        self.url_label.setObjectName("phoneDialogUrl")
        self.url_label.setAlignment(Qt.AlignCenter)
        self.url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.url_label.setWordWrap(True)
        layout.addWidget(self.url_label)

        warn = QLabel(
            "Your browser will warn that the connection is not private because Touchless uses a self-signed "
            "certificate. Tap \"Show Details\" → \"visit this website\" (Safari) or \"Advanced\" → \"Proceed\" "
            "(Chrome) to continue. Then tap Allow when the browser asks to use your camera."
        )
        warn.setObjectName("phoneDialogWarn")
        warn.setWordWrap(True)
        layout.addWidget(warn)

        self.status_label = QLabel("Waiting for phone...")
        self.status_label.setObjectName("phoneDialogStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("phoneDialogButton")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)

        self.use_button = QPushButton("Use This Camera")
        self.use_button.setObjectName("phoneDialogPrimary")
        self.use_button.setEnabled(False)
        self.use_button.clicked.connect(self._on_use_clicked)
        buttons.addWidget(self.use_button)
        layout.addLayout(buttons)

        # Server start + QR render happen right after the dialog is shown
        # so UI paints first (QR generation is trivial but the cert file
        # IO can occasionally stall for a beat on first run).
        QTimer.singleShot(0, self._start_server)

        # Poll every second for "did we see a frame recently" — the
        # streaming callback only fires once per client, but we want to
        # reflect ongoing connectivity.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(1000)
        self._refresh_timer.timeout.connect(self._refresh_status)
        self._refresh_timer.start()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            f"""
            PhoneCameraConnectDialog {{
                background-color: {self.config.surface_color};
                color: {self.config.text_color};
            }}
            QLabel#phoneDialogTitle {{
                color: {self.config.text_color};
                font-size: 17px;
                font-weight: 600;
            }}
            QLabel#phoneDialogSubtitle {{
                color: {self.config.text_color};
                font-size: 13px;
                opacity: 0.85;
            }}
            QLabel#phoneDialogUrl {{
                color: {self.config.accent_color};
                font-size: 15px;
                font-weight: 600;
                padding: 8px 4px;
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
            }}
            QLabel#phoneDialogWarn {{
                color: {self.config.text_color};
                font-size: 12px;
                opacity: 0.75;
            }}
            QLabel#phoneDialogStatus {{
                color: {self.config.text_color};
                font-size: 13px;
                font-weight: 600;
                padding: 8px;
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
            }}
            QPushButton#phoneDialogButton {{
                background-color: rgba(255,255,255,0.08);
                color: {self.config.text_color};
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 10px;
                padding: 8px 18px;
                min-width: 96px;
            }}
            QPushButton#phoneDialogButton:hover {{
                background-color: rgba(255,255,255,0.15);
            }}
            QPushButton#phoneDialogPrimary {{
                background-color: {self.config.accent_color};
                color: #003d2a;
                border: none;
                border-radius: 10px;
                padding: 8px 18px;
                min-width: 140px;
                font-weight: 600;
            }}
            QPushButton#phoneDialogPrimary:disabled {{
                background-color: rgba(29,233,182,0.4);
                color: rgba(0,61,42,0.6);
            }}
            """
        )

    def _start_server(self) -> None:
        if self._server_is_borrowed:
            # Server was already running before this dialog opened; don't
            # re-start, just surface its info.
            info = self._server.info
            if info is None:
                self.status_label.setText("Server is starting... try again in a moment.")
                return
        else:
            try:
                info = self._server.start()
            except Exception as exc:
                self.url_label.setText(f"Failed to start server: {type(exc).__name__}: {exc}")
                self.status_label.setText("Server could not start. Close and try again.")
                return
        self._server_info = info
        self.url_label.setText(info.url)
        try:
            pix = _build_qr_pixmap(info.url, pixel_size=320)
            self.qr_label.setPixmap(pix)
        except Exception as exc:
            self.qr_label.setText(f"(could not render QR: {exc})")
        # Reflect any state the borrowed server is already in so the user
        # doesn't see "Waiting for phone..." when the phone is already
        # streaming.
        if self._server_is_borrowed:
            try:
                if self._server.seconds_since_last_frame < 4.0:
                    self._streaming_seen = True
                    self.status_label.setText("Streaming — phone camera is live.")
                    self.use_button.setEnabled(True)
                elif self._server.connected_clients > 0:
                    self.status_label.setText("Phone connected. Waiting for frames...")
                else:
                    self.status_label.setText("Server is running — waiting for phone to connect.")
            except Exception:
                pass

    def _forward_server_status(self, event: str, data: dict) -> None:
        # Server-thread entry point: re-emit as a Qt Signal which queues
        # the update onto the GUI thread via Qt.QueuedConnection. Never
        # touch widgets directly here.
        try:
            self._server_status.emit(event, data)
        except Exception:
            pass

    def _apply_server_status(self, event: str, data) -> None:
        if event == "phone_page_loaded":
            if not self._streaming_seen:
                self.status_label.setText(
                    "Phone loaded the Touchless page. Waiting for camera connection..."
                )
        elif event == "client_connected":
            self.status_label.setText("Phone browser opened the video stream. Waiting for frames...")
        elif event == "streaming":
            self._streaming_seen = True
            self.status_label.setText("Streaming — phone camera is live.")
            self.use_button.setEnabled(True)
        elif event == "client_disconnected":
            self.status_label.setText("Phone disconnected.")
            self._streaming_seen = False
            self.use_button.setEnabled(False)

    def _refresh_status(self) -> None:
        if self._committed:
            return
        if not self._streaming_seen:
            return
        stale = self._server.seconds_since_last_frame
        if stale > 4.0:
            self.status_label.setText(f"Stream appears stalled (no frame for {stale:.0f}s).")
            self.use_button.setEnabled(False)
        else:
            self.status_label.setText(f"Streaming — last frame {stale*1000:.0f} ms ago.")
            self.use_button.setEnabled(True)

    def _on_use_clicked(self) -> None:
        self._committed = True
        self.camera_accepted.emit(self._server)
        self.accept()

    def _detach_from_server(self) -> None:
        # Stop receiving status updates to this dialog (which is about to
        # be destroyed). The server may live on if it was borrowed.
        try:
            self._server.set_status_callback(None)
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        # If the user closed the dialog without clicking "Use This Camera",
        # tear the server down — UNLESS we borrowed it (auto-started by
        # MainWindow because a phone was previously paired), in which
        # case the user expects it to keep running.
        self._detach_from_server()
        if not self._committed and not self._server_is_borrowed:
            try:
                self._server.stop()
            except Exception:
                pass
        super().closeEvent(event)

    def reject(self) -> None:  # type: ignore[override]
        self._detach_from_server()
        if not self._committed and not self._server_is_borrowed:
            try:
                self._server.stop()
            except Exception:
                pass
        super().reject()

    @property
    def server(self) -> Optional[PhoneCameraServer]:
        return self._server

# Author: Konstantin Markov

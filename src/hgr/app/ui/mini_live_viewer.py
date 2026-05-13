from __future__ import annotations

from typing import Optional

import cv2
from PySide6.QtCore import QPoint, Qt, Signal, QEvent, QTimer
from PySide6.QtGui import QColor, QCursor, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ...config.app_config import AppConfig
from .gpu_video_widget import GpuVideoWidget


class MiniLiveViewer(QWidget):
    enlarge_requested = Signal()
    toggle_gestures_requested = Signal()
    close_requested = Signal()

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._worker: Optional[object] = None
        self._last_frame = None
        self._drag_offset: Optional[QPoint] = None
        self._user_positioned = False
        self._hover_active = False
        self._gestures_enabled = True
        # Frozen-pipeline state: while True, raw_frame_ready is
        # ignored (display holds its current frame, blurred), and
        # the overlay label below sits on top of the video panel.
        # Driven by GestureWorker.frozen_state_changed.
        self._frozen = False

        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.resize(360, 250)

        self._build_ui()
        self.apply_theme(config)

        # Always-on-top watchdog. Qt's WindowStaysOnTopHint alone
        # isn't reliable on Windows: opening a chrome page or the
        # main settings tab can grab focus in a way that drops the
        # mini viewer behind. Periodically re-applying SetWindowPos
        # with HWND_TOPMOST (via the existing native_overlay helper)
        # keeps it visibly pinned. Also re-raised on every show.
        from PySide6.QtCore import QTimer
        self._topmost_watchdog = QTimer(self)
        self._topmost_watchdog.setInterval(2000)
        self._topmost_watchdog.timeout.connect(self._reassert_topmost)
        self._topmost_watchdog.start()

    @staticmethod
    def _pin_topmost_native(widget) -> None:
        """Re-pin the widget as topmost via SetWindowPos. Unlike
        native_overlay.apply_overlay (used by the translucent wheel
        overlays), this DOES NOT add WS_EX_LAYERED / WS_EX_TOOLWINDOW
        — those flags require layered-window APIs to draw and would
        make a normal opaque widget like the mini viewer paint
        invisibly. We just call SetWindowPos with HWND_TOPMOST +
        SWP_NOMOVE / SWP_NOSIZE / SWP_NOACTIVATE so the window keeps
        its current position/size and doesn't steal focus."""
        import sys
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes
        except Exception:
            return
        try:
            hwnd = int(widget.winId())
        except Exception:
            return
        if not hwnd:
            return
        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOACTIVATE = 0x0010
        try:
            ctypes.windll.user32.SetWindowPos(
                wintypes.HWND(hwnd),
                ctypes.c_void_p(HWND_TOPMOST),
                0, 0, 0, 0,
                ctypes.c_uint(SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE),
            )
        except Exception:
            pass

    def _reassert_topmost(self) -> None:
        if not self.isVisible():
            return
        self._pin_topmost_native(self)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        # Re-raise + re-pin topmost on every show. Qt sometimes drops
        # the WindowStaysOnTopHint after hide/show cycles or when
        # another window grabs focus aggressively.
        try:
            self.raise_()
        except Exception:
            pass
        self._pin_topmost_native(self)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.header = QFrame()
        self.header.setObjectName("miniHeader")
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(6)

        self.title_label = QLabel("Touchless")
        self.title_label.setObjectName("miniTitle")
        header_layout.addWidget(self.title_label)

        # Light-blue Lite Mode badge — only visible while
        # config.lite_mode is on. Sits between the title and the
        # buttons so it's the first thing the user notices when the
        # mode flips. We use the same stylesheet hook as the gesture
        # chip so the colour pops against the dark header surface.
        self.lite_mode_badge = QLabel("Lite")
        self.lite_mode_badge.setObjectName("miniLiteBadge")
        self.lite_mode_badge.setAlignment(Qt.AlignCenter)
        self.lite_mode_badge.setVisible(bool(getattr(self.config, "lite_mode", False)))
        header_layout.addWidget(self.lite_mode_badge)

        header_layout.addStretch(1)

        self.enlarge_button = QPushButton("Enlarge")
        self.enlarge_button.setObjectName("miniButton")
        self.enlarge_button.clicked.connect(self.enlarge_requested.emit)
        header_layout.addWidget(self.enlarge_button)

        self.gesture_toggle_button = QPushButton("Gestures On")
        self.gesture_toggle_button.setObjectName("miniButton")
        self.gesture_toggle_button.clicked.connect(self.toggle_gestures_requested.emit)
        header_layout.addWidget(self.gesture_toggle_button)

        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("miniButton")
        self.close_button.clicked.connect(self._handle_close)
        header_layout.addWidget(self.close_button)
        layout.addWidget(self.header)

        # GPU-backed video panel. Replaces the previous QLabel +
        # cv2.cvtColor + cv2.resize + QImage + QPixmap.fromImage +
        # setPixmap path. Frames go from BGR numpy → QImage(BGR888)
        # → GL texture → fragment shader sampler → screen, with
        # only one CPU memcpy along the way.
        self.video_label = GpuVideoWidget()
        self.video_label.setMinimumSize(220, 140)
        self.video_label.clear_video("Press START to begin live gesture tracking.")
        layout.addWidget(self.video_label, 1)

        # Centered overlay label on top of the video panel — visible
        # only while the worker pipeline is frozen (custom-gesture
        # recorder or sandbox open). Stretched edge-to-edge with a transparent
        # background so the blurred frame underneath shows through;
        # the centered text + dark scrim sits in the middle.
        self._frozen_overlay = QLabel(self.video_label)
        self._frozen_overlay.setObjectName("miniFrozenOverlay")
        self._frozen_overlay.setAlignment(Qt.AlignCenter)
        self._frozen_overlay.setText("Paused\nRecording custom gesture…")
        self._frozen_overlay.setStyleSheet(
            "QLabel#miniFrozenOverlay {"
            "  background: rgba(0, 0, 0, 0.42);"
            "  color: #FFFFFF;"
            "  font-weight: 800;"
            "  font-size: 13px;"
            "  border-radius: 12px;"
            "}"
        )
        self._frozen_overlay.hide()
        self._frozen_overlay.setText("Paused\nCustom gesture tool active...")

        # Top-left diagnostic HUD: tiny green text showing
        # 'FPS: X.X', 'Latency: Y ms', or both depending on the
        # Settings → Camera → Live View Overlays toggles. Hidden by
        # default; opt-in via set_overlay_visibility().
        self._diag_hud = QLabel(self.video_label)
        self._diag_hud.setObjectName("miniDiagHud")
        self._diag_hud.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._diag_hud.setStyleSheet(
            "QLabel#miniDiagHud {"
            "  color: #1DE9B6;"
            "  font-family: 'Consolas', 'Segoe UI Mono', monospace;"
            "  font-size: 11px;"
            "  font-weight: 700;"
            "  background: rgba(0, 0, 0, 0.42);"
            "  padding: 2px 6px;"
            "  border-radius: 4px;"
            "}"
        )
        self._diag_hud.hide()
        self._diag_hud_fps_text = "--"
        self._diag_hud_lag_text = "--"
        self._diag_hud_show_fps = bool(getattr(self.config, "live_view_show_fps", False))
        self._diag_hud_show_latency = bool(getattr(self.config, "live_view_show_latency", False))
        # Per-paint EWMA state for the latency value.
        self._lag_ms_smoothed: float = 0.0

        self.gesture_chip = QLabel("Gesture: neutral")
        self.gesture_chip.setObjectName("miniChip")
        self.gesture_chip.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.gesture_chip)

        for widget in (
            self,
            self.header,
            self.title_label,
            self.enlarge_button,
            self.gesture_toggle_button,
            self.close_button,
            self.video_label,
            self.gesture_chip,
        ):
            widget.setAttribute(Qt.WA_Hover, True)
            widget.setMouseTracking(True)
            widget.installEventFilter(self)
        self._set_hover_active(False)

    def apply_theme(self, config: AppConfig) -> None:
        self.config = config
        hover = QColor(self.config.primary_color).lighter(118)
        hover.setAlpha(175)
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: rgba(7,19,29,0.98);
                color: {self.config.text_color};
                border: 1px solid rgba(29,233,182,0.22);
                border-radius: 14px;
            }}
            QFrame#miniHeader {{
                background-color: rgba(9,24,36,0.92);
                border: 1px solid rgba(255,255,255,0.04);
                border-radius: 10px;
            }}
            QLabel#miniTitle {{
                color: {self.config.text_color};
                font-weight: 800;
            }}
            QLabel#miniVideo {{
                background-color: rgba(0,0,0,0.18);
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.08);
                color: {self.config.text_color};
            }}
            QLabel#miniChip {{
                background-color: rgba(9,42,58,0.92);
                color: {self.config.accent_color};
                border-radius: 12px;
                padding: 6px 8px;
                font-weight: 800;
            }}
            QLabel#miniLiteBadge {{
                background-color: rgba(96,165,250,0.18);
                color: #93C5FD;
                border: 1px solid rgba(147,197,253,0.55);
                border-radius: 10px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 0.5px;
            }}
            QPushButton#miniButton {{
                background-color: rgba(255,255,255,0.07);
                color: {self.config.text_color};
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 10px;
                padding: 6px 10px;
                font-weight: 800;
            }}
            QPushButton#miniButton:hover {{
                background-color: {hover.name(QColor.HexArgb)};
                border: 1px solid {self.config.accent_color};
            }}
            QPushButton#miniButton:pressed {{
                background-color: {self.config.accent_color};
                color: #001B24;
                border: 1px solid {self.config.accent_color};
            }}
            """
        )

    def set_gestures_enabled(self, enabled: bool) -> None:
        self._gestures_enabled = bool(enabled)
        self.gesture_toggle_button.setText("Gestures On" if self._gestures_enabled else "Gestures Off")

    def set_lite_mode_active(self, active: bool) -> None:
        if hasattr(self, "lite_mode_badge"):
            self.lite_mode_badge.setVisible(bool(active))

    def attach_to_worker(self, worker: Optional[object]) -> None:
        if self._worker is worker:
            return
        # When transitioning between workers (camera hot-swap), keep
        # the previous frame on screen until the new worker emits one.
        # Calling detach_from_worker here would blank to the "Press
        # START" idle text, which is misleading mid-session — the user
        # just saved a camera change, didn't ask to start over.
        if self._worker is not None:
            try:
                self._worker.debug_frame_ready.disconnect(self._on_worker_debug_frame)
            except Exception:
                pass
            try:
                self._worker.raw_frame_ready.disconnect(self._on_worker_raw_frame)
            except Exception:
                pass
            try:
                self._worker.engine_landmarks_ready.disconnect(self._on_worker_landmarks)
            except Exception:
                pass
            if hasattr(self._worker, "frozen_state_changed"):
                try:
                    self._worker.frozen_state_changed.disconnect(self._on_worker_frozen_changed)
                except Exception:
                    pass
        self._worker = worker
        if self._worker is None:
            self._set_idle_state()
            return
        try:
            self._worker.raw_frame_ready.connect(self._on_worker_raw_frame)
            self._worker.engine_landmarks_ready.connect(self._on_worker_landmarks)
            self._worker.debug_frame_ready.connect(self._on_worker_debug_frame)
            if hasattr(self._worker, "frozen_state_changed"):
                try:
                    self._worker.frozen_state_changed.connect(self._on_worker_frozen_changed)
                except Exception:
                    pass
        except Exception:
            self._worker = None
            self._set_idle_state()

    def detach_from_worker(self) -> None:
        if self._worker is not None:
            try:
                self._worker.raw_frame_ready.disconnect(self._on_worker_raw_frame)
            except Exception:
                pass
            try:
                self._worker.debug_frame_ready.disconnect(self._on_worker_debug_frame)
            except Exception:
                pass
            if hasattr(self._worker, "frozen_state_changed"):
                try:
                    self._worker.frozen_state_changed.disconnect(self._on_worker_frozen_changed)
                except Exception:
                    pass
        self._worker = None
        self._set_idle_state()

    def show_overlay(self) -> None:
        if not self._user_positioned:
            self._move_to_default_corner()
        self.show()
        self.raise_()
        self._sync_hover_state()

    def _move_to_default_corner(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        margin = 22
        self.move(
            geometry.right() - self.width() - margin,
            geometry.bottom() - self.height() - margin,
        )

    def _on_worker_raw_frame(self, frame, capture_ts: float = 0.0) -> None:
        # Fast display path — paints the camera frame at camera fps,
        # before the engine has even started processing it.
        # `capture_ts` is the monotonic time the reader thread
        # decoded this frame.
        if not self.isVisible() or frame is None:
            return
        # While the pipeline is frozen, hold the blurred snapshot
        # already on screen and ignore fresh frames. The overlay
        # label tells the user we're paused.
        if self._frozen:
            return
        # Two-stage drop:
        #   (a) Backlog detection: drop this frame if the daemon
        #       reader has ALREADY decoded a strictly newer one
        #       (with 0.05 s tolerance for the normal one-frame
        #       race between emit and slot-fire). A fresher
        #       raw_frame_ready is queued behind us and will
        #       render better content.
        #   (b) Absolute age cap: drop if the frame is older than
        #       0.12 s regardless. (a) only fires when the daemon
        #       outpaces the slot — but if BOTH the daemon AND
        #       the slot are slowed by GPU contention, frames can
        #       pile in the queue each only marginally newer than
        #       the prior, all collectively stale. (b) is the
        #       safety net for that case. 0.12 s = ~3.5 frames at
        #       30 fps; tight enough to limit visible lag, loose
        #       enough that normal sub-100 ms pipeline jitter
        #       doesn't starve the display.
        if capture_ts > 0.0 and self._worker is not None:
            cap = getattr(self._worker, "_cap", None)
            if cap is not None:
                latest_ts = float(getattr(cap, "_latest_frame_ts", 0.0) or 0.0)
                if latest_ts > capture_ts + 0.05:
                    return
        if capture_ts > 0.0:
            import time as _time
            if (_time.monotonic() - capture_ts) > 0.12:
                return
        self._last_frame = frame
        self._render_frame()
        # Update top-left HUD latency EWMA when enabled. Same 0.8/0.2
        # smoothing as LiveViewWindow so the two readouts agree.
        if self._diag_hud_show_latency and capture_ts > 0.0:
            import time as _time
            instant_ms = max(0.0, (_time.monotonic() - capture_ts) * 1000.0)
            if self._lag_ms_smoothed <= 0.0:
                self._lag_ms_smoothed = instant_ms
            else:
                self._lag_ms_smoothed = 0.8 * self._lag_ms_smoothed + 0.2 * instant_ms
            self._diag_hud_lag_text = f"{self._lag_ms_smoothed:.0f}"
            self._update_diag_hud()

    def _on_worker_landmarks(self, hands_xy_norm) -> None:
        # Engine-completed landmark overlay. Goes to the GPU widget
        # which paints them via QPainter on top of the texture.
        if not self.isVisible():
            return
        try:
            self.video_label.update_landmarks(hands_xy_norm)
        except Exception:
            pass

    def _on_worker_debug_frame(self, frame, payload) -> None:
        # Engine-completion path. The frame here is the engine-
        # annotated version (with landmark overlays drawn by the
        # worker post-engine). We IGNORE the frame and use the
        # payload only — the live-view widget is painted from
        # raw_frame_ready, not from this signal, to keep display
        # latency at camera fps. Updating the gesture-chip text
        # from here keeps the chip in sync with what the engine
        # actually saw, even though the chip may lag the visible
        # hand position by ~1 frame's worth of engine time.
        if not self.isVisible():
            return
        self.gesture_chip.setText(str(payload.get("gesture_chip", "Gesture: neutral")))
        # Top-left HUD FPS readout (when enabled).
        if self._diag_hud_show_fps:
            try:
                fps_value = float(payload.get("fps", 0.0) or 0.0)
            except (TypeError, ValueError):
                fps_value = 0.0
            self._diag_hud_fps_text = "--" if fps_value <= 0.0 else f"{fps_value:.1f}"
            self._update_diag_hud()

    def _on_worker_frozen_changed(self, frozen: bool) -> None:
        self._frozen = bool(frozen)
        if self._frozen:
            # Apply a one-shot blur to the last frame so the user
            # sees a softened version of what the camera was last
            # showing, then ignore subsequent raw frames until the
            # recorder closes. cv2 GaussianBlur is fine on the GUI
            # thread — single ~360x250 BGR frame, sub-millisecond.
            if self._last_frame is not None:
                try:
                    blurred = cv2.GaussianBlur(self._last_frame, (0, 0), sigmaX=9.0)
                    self.video_label.update_frame(blurred)
                except Exception:
                    pass
            self._reposition_frozen_overlay()
            self._frozen_overlay.show()
            self._frozen_overlay.raise_()
        else:
            self._frozen_overlay.hide()
            # Push the most recent (sharp) frame back so the user
            # sees the live view immediately on unfreeze instead of
            # waiting for the next raw_frame_ready emit.
            if self._last_frame is not None:
                self.video_label.update_frame(self._last_frame)

    def _reposition_frozen_overlay(self) -> None:
        try:
            self._frozen_overlay.setGeometry(self.video_label.rect())
        except Exception:
            pass
        self._reposition_diag_hud()

    def _reposition_diag_hud(self) -> None:
        hud = getattr(self, "_diag_hud", None)
        if hud is None:
            return
        try:
            hud.adjustSize()
            hud.move(6, 6)
            hud.raise_()
        except Exception:
            pass

    def _update_diag_hud(self) -> None:
        hud = getattr(self, "_diag_hud", None)
        if hud is None:
            return
        show_fps = bool(getattr(self, "_diag_hud_show_fps", False))
        show_lat = bool(getattr(self, "_diag_hud_show_latency", False))
        if not show_fps and not show_lat:
            hud.hide()
            return
        parts: list[str] = []
        if show_fps:
            parts.append(f"FPS: {self._diag_hud_fps_text}")
        if show_lat:
            parts.append(f"Latency: {self._diag_hud_lag_text} ms")
        hud.setText("  ".join(parts))
        hud.adjustSize()
        hud.show()
        hud.raise_()

    def set_overlay_visibility(
        self,
        *,
        show_fps: bool,
        show_latency: bool,
        show_tracking_quality: bool = False,
    ) -> None:
        """Toggle the top-left diagnostic HUD. Mirrors the same
        Settings → Camera → Live View Overlays flags the enlarged
        LiveViewWindow uses. The tracking_quality argument is
        accepted for signature parity with LiveViewWindow but is
        not rendered in the mini viewer."""
        self._diag_hud_show_fps = bool(show_fps)
        self._diag_hud_show_latency = bool(show_latency)
        _ = show_tracking_quality
        self._update_diag_hud()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._reposition_frozen_overlay()

    def _set_idle_state(self) -> None:
        self._last_frame = None
        self.video_label.clear_video("Press START to begin live gesture tracking.")
        self.gesture_chip.setText("Gesture: neutral")

    def _render_frame(self) -> None:
        # GPU paint: hand the BGR frame straight to the OpenGL
        # widget. Texture upload + scaling + BGR→RGB swizzle all
        # happen on the GPU. No cv2.cvtColor / cv2.resize on the
        # main thread.
        if self._last_frame is None:
            return
        self.video_label.update_frame(self._last_frame)

    def _handle_close(self) -> None:
        self.hide()
        self.close_requested.emit()

    def _set_hover_active(self, active: bool) -> None:
        if self._hover_active == active:
            return
        self._hover_active = active
        self.header.setVisible(active)

    def _sync_hover_state(self) -> None:
        cursor_pos = QCursor.pos()
        local_pos = self.mapFromGlobal(cursor_pos)
        self._set_hover_active(self.rect().contains(local_pos))

    def eventFilter(self, obj, event):  # noqa: N802
        if obj in (
            self,
            self.header,
            self.title_label,
            self.enlarge_button,
            self.gesture_toggle_button,
            self.close_button,
            self.video_label,
            self.gesture_chip,
        ) and event.type() in {
            QEvent.Enter,
            QEvent.Leave,
            QEvent.HoverEnter,
            QEvent.HoverLeave,
            QEvent.HoverMove,
            QEvent.MouseMove,
            QEvent.Show,
        }:
            QTimer.singleShot(0, self._sync_hover_state)
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self._user_positioned = True
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_frame()

    def leaveEvent(self, event) -> None:  # noqa: N802
        super().leaveEvent(event)
        QTimer.singleShot(0, self._sync_hover_state)

# Author: Konstantin Markov

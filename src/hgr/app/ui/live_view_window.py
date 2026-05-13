from __future__ import annotations

import time
from typing import Optional

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from ...config.app_config import AppConfig
from .gpu_video_widget import GpuVideoWidget


class LiveViewWindow(QMainWindow):
    minimize_requested = Signal()
    toggle_gestures_requested = Signal()

    def __init__(self, config: AppConfig, worker: Optional[object] = None):
        super().__init__()
        self.config = config
        self._worker: Optional[object] = None
        self._last_frame = None
        self._volume_level = 0.0
        self._volume_muted = False
        self._volume_active = False
        self._gestures_enabled = True
        # Frozen-pipeline state: mirror MiniLiveViewer's behavior so
        # the enlarged view also indicates the worker is paused
        # while a custom-gesture recorder dialog is open.
        self._frozen = False

        self.setWindowTitle("Touchless Live View")
        self.setMinimumSize(980, 680)
        self.resize(1120, 780)

        self._build_ui()
        self.apply_theme(config)
        self.attach_to_worker(worker)

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("debugRoot")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = QFrame()
        self.header.setObjectName("debugHeader")
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 6, 10, 6)
        header_layout.setSpacing(8)

        self.title_label = QLabel("Touchless Gesture Live View")
        self.title_label.setObjectName("debugHeaderTitle")
        header_layout.addWidget(self.title_label)

        # Light-blue Lite Mode badge — visible only while
        # config.lite_mode is on, so the user can tell at a glance
        # that the speed boost is engaged.
        self.lite_mode_badge = QLabel("Lite")
        self.lite_mode_badge.setObjectName("debugLiteBadge")
        self.lite_mode_badge.setAlignment(Qt.AlignCenter)
        self.lite_mode_badge.setVisible(bool(getattr(self.config, "lite_mode", False)))
        header_layout.addWidget(self.lite_mode_badge)

        header_layout.addStretch(1)

        self.min_button = QPushButton("Minimize")
        self.min_button.setObjectName("debugHeaderButton")
        self.min_button.clicked.connect(self._handle_minimize)

        self.gesture_toggle_button = QPushButton("Gestures On")
        self.gesture_toggle_button.setObjectName("debugHeaderButton")
        self.gesture_toggle_button.clicked.connect(self.toggle_gestures_requested.emit)

        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("debugHeaderButton")
        self.close_button.clicked.connect(self._handle_close)

        header_layout.addWidget(self.min_button)
        header_layout.addWidget(self.gesture_toggle_button)
        header_layout.addWidget(self.close_button)
        outer.addWidget(self.header)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        outer.addWidget(body, 1)

        video_wrap = QFrame()
        video_wrap.setObjectName("debugVideoWrap")
        video_layout = QVBoxLayout(video_wrap)
        video_layout.setContentsMargins(10, 10, 10, 10)
        video_layout.setSpacing(8)

        # GPU-backed video display. See gpu_video_widget.py and
        # MiniLiveViewer for the rationale — this replaces the
        # previous QLabel + CPU pixmap pipeline.
        self.video_label = GpuVideoWidget()
        self.video_label.setMinimumSize(280, 180)
        self.video_label.clear_video("Press START in the app to begin live gesture tracking.")
        video_layout.addWidget(self.video_label, 1)

        # Centered "paused" overlay shown while the worker pipeline
        # is frozen for a custom-gesture recorder or sandbox session. Same
        # treatment as MiniLiveViewer; sized to the video panel via
        # _reposition_frozen_overlay() on every resize.
        self._frozen_overlay = QLabel(self.video_label)
        self._frozen_overlay.setObjectName("debugFrozenOverlay")
        self._frozen_overlay.setAlignment(Qt.AlignCenter)
        self._frozen_overlay.setText("Paused\nRecording custom gesture…")
        self._frozen_overlay.setStyleSheet(
            "QLabel#debugFrozenOverlay {"
            "  background: rgba(0, 0, 0, 0.42);"
            "  color: #FFFFFF;"
            "  font-weight: 800;"
            "  font-size: 18px;"
            "  border-radius: 14px;"
            "}"
        )
        self._frozen_overlay.hide()
        self._frozen_overlay.setText("Paused\nCustom gesture tool active...")

        # Engine-required pill: shown when the live view is opened
        # but the gesture engine isn't running, so the user knows
        # why they're seeing a blank canvas. Positioned by
        # _reposition_engine_required_pill() on every resize.
        self._engine_required_pill = QLabel(self.video_label)
        self._engine_required_pill.setObjectName("debugEngineRequiredPill")
        self._engine_required_pill.setAlignment(Qt.AlignCenter)
        self._engine_required_pill.setWordWrap(True)
        self._engine_required_pill.setText(
            "Start the engine from the home page to enable\n"
            "gesture controls and see the live camera feed."
        )
        self._engine_required_pill.setStyleSheet(
            "QLabel#debugEngineRequiredPill {"
            "  background: rgba(13, 28, 50, 0.86);"
            "  color: #FFFFFF;"
            "  font-weight: 700;"
            "  font-size: 14px;"
            "  padding: 12px 18px;"
            "  border: 1px solid rgba(29,233,182,0.55);"
            "  border-radius: 14px;"
            "}"
        )
        self._engine_required_pill.hide()

        # Top-left HUD: tiny always-on FPS + display-lag readout in
        # the accent green. Child of the video widget so it sits
        # ON TOP of the camera frame; positioned by _reposition_diag_hud()
        # on every resize. Mouse-transparent so it doesn't intercept
        # clicks meant for the underlying video / overlay widgets.
        self._diag_hud = QLabel(self.video_label)
        self._diag_hud.setObjectName("liveViewDiagHud")
        self._diag_hud.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._diag_hud.setStyleSheet(
            "QLabel#liveViewDiagHud {"
            "  color: #1DE9B6;"
            "  font-family: 'Consolas', 'Segoe UI Mono', monospace;"
            "  font-size: 11px;"
            "  font-weight: 700;"
            "  background: rgba(0, 0, 0, 0.42);"
            "  padding: 2px 6px;"
            "  border-radius: 4px;"
            "}"
        )
        self._diag_hud.hide()  # opt-in via fps / latency toggles
        self._diag_hud_fps_text = "--"
        self._diag_hud_lag_text = "--"
        self._diag_hud_show_fps = False
        self._diag_hud_show_latency = False

        # ---- Diagnostic overlay row (FPS · Latency · Tracking) ----
        # Three optional pills that sit between the video panel and
        # the gesture chip. Each one is hidden by default and shown
        # only when the user enables it in Settings → Camera → Live
        # View Overlays. The whole row hides when nothing is enabled
        # so a clean feed remains the default look.
        self.diagnostic_row = QWidget()
        diag_layout = QHBoxLayout(self.diagnostic_row)
        diag_layout.setContentsMargins(0, 0, 0, 0)
        diag_layout.setSpacing(8)
        diag_layout.addStretch(1)

        self.fps_chip = QLabel("FPS: —")
        self.fps_chip.setObjectName("fpsChip")
        self.fps_chip.setAlignment(Qt.AlignCenter)
        self.fps_chip.hide()
        diag_layout.addWidget(self.fps_chip)

        # Live latency readout: time from camera-frame decode to
        # this very paint, in milliseconds. EWMA-smoothed so the
        # display number doesn't strobe.
        self.latency_label = QLabel("Display lag: --")
        self.latency_label.setObjectName("latencyChip")
        self.latency_label.setAlignment(Qt.AlignCenter)
        self.latency_label.hide()
        diag_layout.addWidget(self.latency_label)

        self.tracking_quality_chip = QLabel("Tracking: —")
        self.tracking_quality_chip.setObjectName("trackingQualityChip")
        self.tracking_quality_chip.setAlignment(Qt.AlignCenter)
        self.tracking_quality_chip.hide()
        diag_layout.addWidget(self.tracking_quality_chip)

        diag_layout.addStretch(1)
        video_layout.addWidget(self.diagnostic_row, 0, Qt.AlignCenter)
        # Per-paint EWMA state for the latency chip.
        self._lag_ms_smoothed: float = 0.0
        # Tracking-quality state. last_hand_ts records the most
        # recent frame where `found=True` so the "No hand seen"
        # state only triggers after the hand has been absent for
        # >1.5 s. Throttle prevents the chip from churning at 30 Hz.
        self._tracking_quality_state = "idle"
        self._tracking_quality_last_hand_ts = 0.0
        self._tracking_quality_last_update_ts = 0.0
        # Apply initial visibility from config.
        self.set_overlay_visibility(
            show_fps=bool(getattr(self.config, "live_view_show_fps", False)),
            show_latency=bool(getattr(self.config, "live_view_show_latency", False)),
            show_tracking_quality=bool(getattr(self.config, "live_view_show_tracking_quality", False)),
        )

        self.gesture_chip = QLabel("Gesture: neutral")
        self.gesture_chip.setObjectName("gestureChip")
        self.gesture_chip.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.gesture_chip, 0, Qt.AlignCenter)
        body_layout.addWidget(video_wrap, 1)

        self.side_card = QFrame()
        self.side_card.setObjectName("debugSideCard")
        self.side_card.setMinimumWidth(380)
        side_layout = QVBoxLayout(self.side_card)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(8)

        self.info_labels: list[QLabel] = []
        for text in (
            "Camera: waiting",
            "Handedness: -",
            "Gesture raw/stable: neutral / neutral",
            "Confidence: 0.00",
            "FPS: 0.0",
            "Box: -",
            "Palm: -",
            "Dynamic: neutral",
            "Candidates: -",
            "Thumb: -",
            "Index: -",
            "Middle: -",
            "Ring: -",
            "Pinky: -",
            "Spreads: -",
            "Reasoning: no hand in frame",
            "Volume control: unavailable",
            "Volume level: -",
            "Spotify control: -",
            "Spotify info: -",
            "Chrome mode: off",
            "Chrome control: -",
            "YouTube mode: off",
            "YouTube control: -",
            "Voice mode: ready",
            "Voice control: -",
            "Voice heard: -",
            "Mouse mode: off",
            "Mouse control: -",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            label.setObjectName("debugInfoLabel")
            side_layout.addWidget(label)
            self.info_labels.append(label)

        self.volume_bar_card = QFrame()
        self.volume_bar_card.setObjectName("volumeCard")
        volume_card_layout = QVBoxLayout(self.volume_bar_card)
        volume_card_layout.setContentsMargins(10, 10, 10, 10)
        volume_card_layout.setSpacing(6)

        self.volume_bar_title = QLabel("System Volume")
        self.volume_bar_title.setObjectName("volumeTitle")
        volume_card_layout.addWidget(self.volume_bar_title)

        self.volume_bar_bg = QFrame()
        self.volume_bar_bg.setObjectName("volumeBarBg")
        self.volume_bar_bg.setFixedHeight(18)
        volume_bg_layout = QHBoxLayout(self.volume_bar_bg)
        volume_bg_layout.setContentsMargins(0, 0, 0, 0)
        volume_bg_layout.setSpacing(0)

        self.volume_bar_fill = QFrame(self.volume_bar_bg)
        self.volume_bar_fill.setObjectName("volumeBarFill")
        self.volume_bar_fill.setFixedWidth(0)
        volume_bg_layout.addWidget(self.volume_bar_fill, 0, Qt.AlignLeft)
        volume_bg_layout.addStretch(1)
        volume_card_layout.addWidget(self.volume_bar_bg)

        self.volume_bar_text = QLabel("Volume inactive")
        self.volume_bar_text.setObjectName("volumeText")
        volume_card_layout.addWidget(self.volume_bar_text)
        side_layout.addWidget(self.volume_bar_card)

        note = QLabel(
            "This app live view mirrors the active Touchless runtime. Use it to watch the live hand skeleton, finger-state reasoning, gesture status, system-volume feedback, app control state, and voice state while the app is running."
        )
        note.setWordWrap(True)
        note.setObjectName("debugNote")
        side_layout.addWidget(note)
        side_layout.addStretch(1)

        body_layout.addWidget(self.side_card)

    def apply_theme(self, config: AppConfig) -> None:
        self.config = config
        button_hover = QColor(self.config.primary_color).lighter(118)
        button_hover.setAlpha(175)
        self.setStyleSheet(
            f"""
            QWidget#debugRoot {{
                background-color: rgba(7, 19, 29, 0.98);
                color: {self.config.text_color};
            }}
            QFrame#debugHeader {{
                background-color: rgba(9, 24, 36, 0.95);
                border-bottom: 1px solid rgba(29, 233, 182, 0.22);
            }}
            QLabel#debugHeaderTitle {{
                color: {self.config.text_color};
                font-size: 15px;
                font-weight: 800;
            }}
            QPushButton#debugHeaderButton {{
                background-color: rgba(255,255,255,0.07);
                color: {self.config.text_color};
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 10px;
                padding: 6px 10px;
                min-width: 72px;
                font-weight: 800;
            }}
            QPushButton#debugHeaderButton:hover {{
                background-color: {button_hover.name(QColor.HexArgb)};
                border: 1px solid {self.config.accent_color};
            }}
            QPushButton#debugHeaderButton:pressed {{
                background-color: {self.config.accent_color};
                color: #001B24;
                border: 1px solid {self.config.accent_color};
            }}
            QFrame#debugVideoWrap, QFrame#debugSideCard, QFrame#volumeCard {{
                background-color: rgba(255,255,255,0.04);
                border: 1px solid rgba(29, 233, 182, 0.22);
                border-radius: 16px;
            }}
            QLabel#videoLabel {{
                background-color: rgba(0,0,0,0.16);
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.08);
                color: {self.config.text_color};
            }}
            QLabel#gestureChip {{
                background-color: rgba(9, 42, 58, 0.90);
                color: {self.config.accent_color};
                border-radius: 12px;
                padding: 6px 10px;
                font-weight: 800;
            }}
            QLabel#latencyChip {{
                background-color: rgba(255, 200, 60, 0.14);
                color: #FFCB66;
                border: 1px solid rgba(255, 200, 60, 0.40);
                border-radius: 10px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.3px;
            }}
            QLabel#debugLiteBadge {{
                background-color: rgba(96,165,250,0.18);
                color: #93C5FD;
                border: 1px solid rgba(147,197,253,0.55);
                border-radius: 10px;
                padding: 2px 10px;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 0.5px;
            }}
            QLabel#debugInfoLabel {{
                color: {self.config.text_color};
                background: transparent;
            }}
            QLabel#debugNote {{
                color: rgba(229,246,255,0.88);
                background: transparent;
            }}
            QLabel#volumeTitle {{
                color: {self.config.accent_color};
                font-weight: 800;
                background: transparent;
            }}
            QFrame#volumeBarBg {{
                background-color: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 9px;
            }}
            QFrame#volumeBarFill {{
                background-color: {self.config.accent_color};
                border-radius: 9px;
            }}
            QLabel#volumeText {{
                color: {self.config.text_color};
                background: transparent;
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
        # Hold the last visible frame across worker transitions
        # (camera hot-swap). detach_from_worker would blank to the
        # "Press START" idle text mid-session, which surprises users.
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

    def show_window(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _handle_minimize(self) -> None:
        self.hide()
        self.minimize_requested.emit()

    def _handle_close(self) -> None:
        self.hide()

    def _on_worker_raw_frame(self, frame, capture_ts: float = 0.0) -> None:
        # Decoupled-display paint path — see MiniLiveViewer for
        # rationale. Renders the camera frame at camera fps,
        # bypasses the engine pipeline.
        if not self.isVisible() or frame is None:
            return
        if self._frozen:
            # Hold the blurred snapshot already on screen while a
            # custom-gesture recorder is open.
            return
        # Two-stage drop (see MiniLiveViewer for full rationale):
        #   (a) Backlog detection: drop if daemon already decoded
        #       a strictly newer frame (with 0.05 s tolerance).
        #   (b) Absolute age cap at 0.12 s as a safety net for
        #       the case where daemon AND slot are both slowed by
        #       GPU contention so backlog detection alone doesn't
        #       fire.
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
        # Update the latency + display-rate readout. The label
        # shows BOTH so we can spot the case where lag looks low
        # (frames flow into the slot quickly) but display is
        # actually low fps (paint events are coalescing or driver
        # is back-pressuring). The two together = honest user-
        # visible truth.
        import time as _time
        now = _time.monotonic()
        # Lag (capture → here)
        if capture_ts > 0.0:
            instant_ms = max(0.0, (now - capture_ts) * 1000.0)
            if self._lag_ms_smoothed <= 0.0:
                self._lag_ms_smoothed = instant_ms
            else:
                self._lag_ms_smoothed = 0.8 * self._lag_ms_smoothed + 0.2 * instant_ms
        # Display rate (slot fires per second — note this is the
        # frame ARRIVAL rate, not the paint rate; the GpuVideoWidget
        # logs actual paint rate to stderr separately).
        if not hasattr(self, "_arrival_count"):
            self._arrival_count = 0
            self._arrival_window_start = now
            self._arrival_rate_smoothed = 0.0
        self._arrival_count += 1
        elapsed = now - self._arrival_window_start
        if elapsed >= 1.0:
            instant_rate = self._arrival_count / elapsed
            if self._arrival_rate_smoothed <= 0.0:
                self._arrival_rate_smoothed = instant_rate
            else:
                self._arrival_rate_smoothed = 0.7 * self._arrival_rate_smoothed + 0.3 * instant_rate
            self._arrival_count = 0
            self._arrival_window_start = now
        self.latency_label.setText(
            f"Display lag: {self._lag_ms_smoothed:.0f} ms  |  "
            f"Frames in: {self._arrival_rate_smoothed:.0f}/s"
        )
        # Always-on top-left HUD mirrors the lag value.
        self._diag_hud_lag_text = f"{self._lag_ms_smoothed:.0f}"
        self._update_diag_hud()

    def _on_worker_landmarks(self, hands_xy_norm) -> None:
        # GPU landmark overlay path. Same rationale as in
        # MiniLiveViewer.
        if not self.isVisible():
            return
        try:
            self.video_label.update_landmarks(hands_xy_norm)
        except Exception:
            pass

    def _on_worker_debug_frame(self, frame, payload) -> None:
        # Engine-result path — payload only. The live-view pixmap
        # is painted from raw_frame_ready above, not from here, so
        # the display stays at camera fps regardless of engine
        # processing time. We use this signal exclusively for
        # state widgets that mirror engine-derived information
        # (gesture chip, info lines, volume readout).
        if not self.isVisible():
            return
        self.gesture_chip.setText(str(payload.get("gesture_chip", "Gesture: neutral")))
        info_lines = list(payload.get("info_lines", []))
        for index, label in enumerate(self.info_labels):
            if index < len(info_lines):
                label.setText(str(info_lines[index]))
            else:
                label.setText("-")
        self._volume_level = float(payload.get("volume_level_scalar", 0.0) or 0.0)
        self._volume_muted = bool(payload.get("volume_muted", False))
        self._volume_active = bool(payload.get("volume_active", False))
        self._update_volume_widgets()
        self._update_diagnostic_chips(payload)

    def _update_diagnostic_chips(self, payload: dict) -> None:
        """Update FPS + tracking-quality chips from the worker
        payload. Throttled to ~4 Hz so the chips don't churn at the
        camera's full frame rate. Latency is updated separately
        from raw frames (different signal, different EWMA)."""
        if not (
            self.fps_chip.isVisibleTo(self)
            or self.tracking_quality_chip.isVisibleTo(self)
        ):
            # Nothing visible — still keep the last-hand timestamp
            # current so toggling on doesn't flash "No hand seen"
            # immediately.
            try:
                if bool(payload.get("found", False)):
                    self._tracking_quality_last_hand_ts = time.monotonic()
            except Exception:
                pass
            return
        now = time.monotonic()
        if (now - self._tracking_quality_last_update_ts) < 0.25:
            try:
                if bool(payload.get("found", False)):
                    self._tracking_quality_last_hand_ts = now
            except Exception:
                pass
            return
        self._tracking_quality_last_update_ts = now

        # ---- FPS ----
        try:
            fps_value = float(payload.get("fps", 0.0) or 0.0)
        except (TypeError, ValueError):
            fps_value = 0.0
        # Always feed the top-left HUD; chip update is gated on its
        # own visibility for the diagnostic-row pills.
        self._diag_hud_fps_text = "--" if fps_value <= 0.0 else f"{fps_value:.1f}"
        self._update_diag_hud()
        if self.fps_chip.isVisibleTo(self):
            if fps_value <= 0.0:
                self.fps_chip.setText("FPS: —")
                self._apply_fps_chip_color("neutral")
            else:
                self.fps_chip.setText(f"FPS: {fps_value:.1f}")
                if fps_value >= 22.0:
                    self._apply_fps_chip_color("good")
                elif fps_value >= 15.0:
                    self._apply_fps_chip_color("warn")
                else:
                    self._apply_fps_chip_color("bad")

        # ---- Tracking quality ----
        if self.tracking_quality_chip.isVisibleTo(self):
            try:
                found = bool(payload.get("found", False))
                confidence = float(payload.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                found = False
                confidence = 0.0
            if found:
                self._tracking_quality_last_hand_ts = now
            time_since_hand = now - self._tracking_quality_last_hand_ts
            # Key off `found` (MediaPipe hand-presence flag), NOT the
            # `confidence` value: `confidence` is the gesture-CLASSIFIER
            # score, so a neutral hand with no recognized gesture
            # would be `found=True, confidence≈0`, which previously
            # left the chip stuck in whatever the prior state was.
            # User-visible symptom: chip read 'No hand seen' even
            # while the skeleton + bbox were clearly drawn on top of
            # a clearly-visible hand.
            if found:
                self._set_tracking_quality_state("good")
            elif time_since_hand >= 0.6:
                self._set_tracking_quality_state("poor")
            # Between found=False and 0.6 s timeout, keep the prior
            # state to avoid flashing red on single-frame drops.

    def _apply_fps_chip_color(self, kind: str) -> None:
        """Recolor the FPS chip — green when comfortably high,
        amber when in the Low FPS Mode zone, red when very low."""
        palette = {
            "good":    ("#1DE9B6", "rgba(29,233,182,0.18)", "rgba(29,233,182,0.55)"),
            "warn":    ("#F5B450", "rgba(245,180,80,0.18)",  "rgba(245,180,80,0.55)"),
            "bad":     ("#FF8A8A", "rgba(255,107,107,0.18)", "rgba(255,107,107,0.55)"),
            "neutral": ("rgba(229,246,255,0.75)", "rgba(255,255,255,0.05)", "rgba(127,127,127,0.30)"),
        }
        fg, bg, border = palette.get(kind, palette["neutral"])
        self.fps_chip.setStyleSheet(
            "QLabel#fpsChip {"
            f"  background: {bg};"
            f"  color: {fg};"
            f"  border: 1px solid {border};"
            "  border-radius: 12px;"
            "  padding: 3px 12px;"
            "  font-size: 12px;"
            "  font-weight: 700;"
            "  letter-spacing: 0.3px;"
            "}"
        )

    def _set_tracking_quality_state(self, state: str) -> None:
        if state == self._tracking_quality_state:
            return
        self._tracking_quality_state = state
        states = {
            "good": ("Tracking: Good",       "#1DE9B6", "rgba(29,233,182,0.18)", "rgba(29,233,182,0.55)"),
            "fair": ("Tracking: Marginal",   "#F5B450", "rgba(245,180,80,0.18)", "rgba(245,180,80,0.55)"),
            "poor": ("Tracking: No hand seen","#FF8A8A","rgba(255,107,107,0.18)","rgba(255,107,107,0.55)"),
            "idle": ("Tracking: —",          "rgba(229,246,255,0.75)", "rgba(255,255,255,0.05)", "rgba(127,127,127,0.30)"),
        }
        text, fg, bg, border = states.get(state, states["idle"])
        self.tracking_quality_chip.setText(text)
        self.tracking_quality_chip.setStyleSheet(
            "QLabel#trackingQualityChip {"
            f"  background: {bg};"
            f"  color: {fg};"
            f"  border: 1px solid {border};"
            "  border-radius: 12px;"
            "  padding: 3px 12px;"
            "  font-size: 12px;"
            "  font-weight: 700;"
            "  letter-spacing: 0.3px;"
            "}"
        )

    def set_overlay_visibility(
        self,
        *,
        show_fps: bool,
        show_latency: bool,
        show_tracking_quality: bool,
    ) -> None:
        """Toggle individual diagnostic-pill visibility. Hides the
        row container entirely when nothing is enabled so the live
        view stays clean for users who don't want diagnostics."""
        self.fps_chip.setVisible(bool(show_fps))
        self.latency_label.setVisible(bool(show_latency))
        self.tracking_quality_chip.setVisible(bool(show_tracking_quality))
        # Top-left HUD mirrors the same toggles independently — text
        # composes as 'FPS: X.X', 'Latency: Y ms', or both with FPS
        # left-most.
        self._diag_hud_show_fps = bool(show_fps)
        self._diag_hud_show_latency = bool(show_latency)
        self._update_diag_hud()
        # When tracking quality is freshly enabled and the engine
        # isn't running, the chip would otherwise stay blank until
        # the first frame arrives. Seed the idle state so it reads
        # immediately.
        if show_tracking_quality and self._tracking_quality_state == "idle":
            self._set_tracking_quality_state("idle")
        # When all three are off, hide the wrapper row too.
        any_visible = bool(show_fps or show_latency or show_tracking_quality)
        self.diagnostic_row.setVisible(any_visible)

    def _on_worker_frozen_changed(self, frozen: bool) -> None:
        self._frozen = bool(frozen)
        if self._frozen:
            if self._last_frame is not None:
                try:
                    blurred = cv2.GaussianBlur(self._last_frame, (0, 0), sigmaX=11.0)
                    self.video_label.update_frame(blurred)
                except Exception:
                    pass
            self._reposition_frozen_overlay()
            self._frozen_overlay.show()
            self._frozen_overlay.raise_()
        else:
            self._frozen_overlay.hide()
            if self._last_frame is not None:
                self.video_label.update_frame(self._last_frame)

    def _reposition_frozen_overlay(self) -> None:
        try:
            self._frozen_overlay.setGeometry(self.video_label.rect())
        except Exception:
            pass
        self._reposition_engine_required_pill()
        self._reposition_diag_hud()

    def _reposition_diag_hud(self) -> None:
        hud = getattr(self, "_diag_hud", None)
        if hud is None:
            return
        try:
            hud.adjustSize()
            hud.move(8, 8)
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

    def _reposition_engine_required_pill(self) -> None:
        try:
            video_rect = self.video_label.rect()
            hint_w = min(420, max(220, int(video_rect.width() * 0.6)))
            hint_h = max(64, self._engine_required_pill.sizeHint().height())
            x = (video_rect.width() - hint_w) // 2
            y = (video_rect.height() - hint_h) // 2
            self._engine_required_pill.setGeometry(x, y, hint_w, hint_h)
        except Exception:
            pass

    def set_engine_required_pill_visible(self, visible: bool) -> None:
        """Toggle the centered 'Start the engine to use controls' pill.
        Shown when the live view is opened with no live worker so the
        user understands why the feed is empty."""
        try:
            if bool(visible):
                self._reposition_engine_required_pill()
                self._engine_required_pill.show()
                self._engine_required_pill.raise_()
            else:
                self._engine_required_pill.hide()
        except Exception:
            pass

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._reposition_frozen_overlay()

    def _set_idle_state(self) -> None:
        self._last_frame = None
        self.video_label.clear_video("Press START in the app to begin live gesture tracking.")
        self.gesture_chip.setText("Gesture: neutral")
        defaults = (
            "Camera: waiting",
            "Handedness: -",
            "Gesture raw/stable: neutral / neutral",
            "Confidence: 0.00",
            "FPS: 0.0",
            "Box: -",
            "Palm: -",
            "Dynamic: neutral",
            "Candidates: -",
            "Thumb: -",
            "Index: -",
            "Middle: -",
            "Ring: -",
            "Pinky: -",
            "Spreads: -",
            "Reasoning: no hand in frame",
            "Volume control: idle",
            "Volume level: -",
            "Spotify control: -",
            "Spotify info: -",
            "Chrome mode: off",
            "Chrome control: -",
            "YouTube mode: off",
            "YouTube control: -",
            "Voice mode: ready",
            "Voice control: -",
            "Voice heard: -",
            "Mouse mode: off",
            "Mouse control: -",
        )
        for index, label in enumerate(self.info_labels):
            label.setText(defaults[index] if index < len(defaults) else "-")
        self._volume_level = 0.0
        self._volume_muted = False
        self._volume_active = False
        self._gestures_enabled = True
        self._update_volume_widgets()

    def _render_frame(self) -> None:
        # GPU paint — hand the BGR frame to the OpenGL widget.
        if self._last_frame is None:
            return
        self.video_label.update_frame(self._last_frame)

    def _update_volume_widgets(self) -> None:
        level = max(0.0, min(1.0, float(self._volume_level)))
        bar_width = max(0, int(self.volume_bar_bg.contentsRect().width() * level))
        self.volume_bar_fill.setFixedWidth(bar_width)
        prefix = "Active" if self._volume_active else "Idle"
        mute_suffix = " [muted]" if self._volume_muted else ""
        self.volume_bar_text.setText(f"{prefix}: {int(round(level * 100))}%{mute_suffix}")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_frame()
        self._update_volume_widgets()

# Author: Konstantin Markov

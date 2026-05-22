from __future__ import annotations

import queue
import threading
import time

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..app.camera.camera_utils import open_preferred_or_first_available
from ..config.app_config import AppConfig
from ..voice.dictation import DictationProcessor
from ..core.pipeline.gesture_backend import GestureBackend
from ..voice.command_processor import VoiceCommandProcessor
from ..voice.live_dictation import LiveDictationStreamer
from .screen_volume_overlay import ScreenVolumeOverlay
from .text_input_controller import TextInputController
from .voice_command_listener import VoiceCommandListener
from .volume_controller import VolumeController
from .volume_gesture import VolumeGestureTracker


class DebugWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        from ..app.ui.window_chrome import apply_touchless_chrome
        apply_touchless_chrome(self)
        self.config = config
        self.backend: GestureBackend | None = None
        self.cap = None
        self.camera_info = None

        self._last_time = time.time()
        self._fps = 0.0

        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._tick)

        # Volume control
        self.volume_controller = VolumeController()
        self.volume_overlay = ScreenVolumeOverlay(config)
        self.volume_overlay.attach_controller(self.volume_controller)
        self.volume_tracker = VolumeGestureTracker()
        self._volume_message = self.volume_controller.message
        self._volume_mode_active = False
        self._volume_level: float | None = self.volume_controller.get_level()
        self._volume_status_text = "idle"
        self._volume_muted = self._read_system_mute()
        self._volume_overlay_visible = False
        self._mute_block_until = 0.0
        self._dynamic_status_text = "neutral"
        self._dynamic_summary_text = "-"
        self._dynamic_hold_until = 0.0
        self._horizontal_swipe_block_label: str | None = None
        self._horizontal_swipe_block_until = 0.0
        self.voice_listener = VoiceCommandListener()
        self.voice_processor = VoiceCommandProcessor()
        self.live_dictation_streamer = LiveDictationStreamer()
        self.text_input_controller = TextInputController()
        self.dictation_processor = DictationProcessor()
        try:
            self.voice_listener.set_app_hints(self.voice_processor.desktop_controller.application_hint_names())
        except Exception:
            pass
        self._voice_control_text = self.voice_listener.message
        self._voice_heard_text = "-"
        self._voice_candidate = "neutral"
        self._voice_candidate_since = 0.0
        self._voice_cooldown_until = 0.0
        self._voice_latched_label: str | None = None
        self._voice_queue: queue.Queue[tuple[int, object]] = queue.Queue()
        self._voice_thread: threading.Thread | None = None
        self._voice_request_id = 0
        self._voice_stop_event: threading.Event | None = None
        self._voice_listening = False
        self._dictation_active = False
        self._dictation_backend = "idle"
        self._voice_mode = "ready"
        self._voice_display_text = "-"

        self.setWindowTitle("Touchless Debug")
        self.setMinimumSize(980, 680)
        self.resize(1120, 780)

        self._build_ui()
        self.apply_theme(config)
        QTimer.singleShot(0, self.start_session)

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

        self.title_label = QLabel("Touchless Gesture Debugger")
        self.title_label.setObjectName("debugHeaderTitle")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)

        self.restart_button = QPushButton("Restart Camera")
        self.restart_button.setObjectName("debugHeaderButton")
        self.restart_button.clicked.connect(self.restart_session)

        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("debugHeaderButton")
        self.close_button.clicked.connect(self.close)

        header_layout.addWidget(self.restart_button)
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

        self.video_label = QLabel("Opening camera...")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(280, 180)
        video_layout.addWidget(self.video_label, 1)

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
            "Raw gesture: neutral",
            "Stable gesture: neutral",
            "Confidence: 0.00",
            "FPS: 0.0",
            "Fingers: T=- I=- M=- R=- P=-",
            "Finger count open: 0",
            "Open scores: T=0.00 I=0.00 M=0.00 R=0.00 P=0.00",
            "Thumb-Index: -",
            "Index-Middle: -",
            "Middle-Ring: -",
            "Ring-Pinky: -",
            "Top candidates: -",
            "Dynamic candidates: -",
            "Volume control: unavailable",
            "Volume level: -",
            "Voice mode: idle",
            "Voice control: -",
            "Voice heard: -",
        ):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setObjectName("debugInfoLabel")
            side_layout.addWidget(lbl)
            self.info_labels.append(lbl)

        self.volume_bar_card = QFrame()
        self.volume_bar_card.setMaximumWidth(128)
        self.volume_bar_card.setObjectName("volumeCard")
        volume_card_layout = QVBoxLayout(self.volume_bar_card)
        volume_card_layout.setContentsMargins(10, 10, 10, 10)
        volume_card_layout.setSpacing(6)

        self.volume_bar_title = QLabel("Vol.")
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
            "Gesture testing starts here. The app itself is currently UI-only, while this debugger uses the new 2D/2.5D recognizer.\n\n"
            "Volume gesture: index + middle open and together, thumb/ring/pinky closed. The volume overlay appears only while volume control is active, and opening the pinky briefly holds the current level.\n\n"
            "Dynamic gestures show up here too, including swipe left and swipe right.\n\n"
            "Voice: hold the left hand in the one gesture for commands, or hold left hand two to start and stop dictation."
        )
        note.setWordWrap(True)
        note.setObjectName("debugNote")
        side_layout.addWidget(note)
        side_layout.addStretch(1)

        body_layout.addWidget(self.side_card)

    def apply_theme(self, config: AppConfig) -> None:
        self.config = config
        self.volume_overlay.apply_theme(config)
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

    def start_session(self) -> None:
        self.stop_session()
        self._volume_message = self.volume_controller.message
        self._volume_level = self.volume_controller.get_level()
        self._volume_mode_active = False
        self._volume_status_text = "idle"
        self._volume_muted = self._read_system_mute()
        self.volume_tracker.reset(self._volume_level, self._volume_muted)
        self._mute_block_until = 0.0
        self._dynamic_status_text = "neutral"
        self._dynamic_summary_text = "-"
        self._dynamic_hold_until = 0.0
        self._horizontal_swipe_block_label = None
        self._horizontal_swipe_block_until = 0.0
        self._reset_voice_state()
        self._update_volume_widgets()
        self.volume_overlay.hide_overlay()
        self._volume_overlay_visible = False

        try:
            self.backend = GestureBackend()
        except Exception as exc:
            self.video_label.setText("Could not start gesture backend.")
            QMessageBox.critical(self, "Touchless Debug", str(exc))
            return

        camera_info, cap = open_preferred_or_first_available(
            self.config.preferred_camera_index,
            max_index=self.config.camera_scan_limit,
        )
        if cap is None or camera_info is None:
            self.video_label.setText("No camera found.")
            QMessageBox.warning(self, "Touchless Debug", "No available camera was found.")
            return

        self.cap = cap
        self.camera_info = camera_info
        self._set_info(0, f"Camera: {camera_info.display_name}")
        self.video_label.setText("")
        self._last_time = time.time()
        self.timer.start()

    def restart_session(self) -> None:
        self.start_session()

    def stop_session(self) -> None:
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.backend is not None:
            self.backend.close()
            self.backend = None
        self._volume_mode_active = False
        self.volume_tracker.reset(self._volume_level, self._volume_muted)
        self._mute_block_until = 0.0
        self._dynamic_status_text = "neutral"
        self._dynamic_summary_text = "-"
        self._dynamic_hold_until = 0.0
        self._horizontal_swipe_block_label = None
        self._horizontal_swipe_block_until = 0.0
        self._reset_voice_state()
        self.volume_overlay.hide_overlay()
        self._volume_overlay_visible = False

    def _tick(self) -> None:
        if self.cap is None or self.backend is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            return

        frame = cv2.flip(frame, 1)
        result = self.backend.process(frame)

        now = time.time()
        dt = max(now - self._last_time, 1e-6)
        self._fps = 0.85 * self._fps + 0.15 * (1.0 / dt) if self._fps else (1.0 / dt)
        self._last_time = now

        monotonic_now = time.monotonic()
        self._update_dynamic_status(result, monotonic_now)
        self._handle_volume_control(result, monotonic_now)
        self._drain_voice_results()
        if result.handedness == "Left":
            self._handle_left_hand_voice(result.stable_gesture, monotonic_now)
        else:
            self._reset_voice_candidate(monotonic_now)

        display_frame = result.annotated_frame.copy()
        gesture_text = self._dynamic_summary_text if self._dynamic_status_text != "neutral" else result.stable_gesture
        if self._dynamic_status_text != "neutral":
            self.gesture_chip.setText(f"Dynamic: {gesture_text}   (raw: {result.raw_gesture})")
        else:
            self.gesture_chip.setText(f"Gesture: {gesture_text}   (raw: {result.raw_gesture})")
        self._set_info(1, f"Handedness: {result.handedness or '-'}")
        self._set_info(2, f"Raw gesture: {result.raw_gesture}")
        self._set_info(3, f"Stable gesture: {result.stable_gesture} ({result.stable_count}/3)")
        self._set_info(4, f"Confidence: {result.confidence:.2f}")
        self._set_info(5, f"FPS: {self._fps:.1f}")

        if result.features is not None:
            states = result.features.states
            scores = result.features.open_scores
            spread_states = result.features.spread_states

            self._set_info(
                6,
                "Fingers: "
                f"T={states['thumb']} I={states['index']} M={states['middle']} R={states['ring']} P={states['pinky']}",
            )
            self._set_info(7, f"Finger count open: {result.features.finger_count_open}")
            self._set_info(
                8,
                "Open scores: "
                f"T={scores['thumb']:.2f} I={scores['index']:.2f} M={scores['middle']:.2f} "
                f"R={scores['ring']:.2f} P={scores['pinky']:.2f}",
            )
            self._set_info(9, f"Thumb-Index: {spread_states['thumb_index']}")
            self._set_info(10, f"Index-Middle: {spread_states['index_middle']}")
            self._set_info(11, f"Middle-Ring: {spread_states['middle_ring']}")
            self._set_info(12, f"Ring-Pinky: {spread_states['ring_pinky']}")

        if result.candidate_scores:
            ranked = sorted(result.candidate_scores.items(), key=lambda kv: kv[1], reverse=True)[:4]
            self._set_info(13, "Top candidates: " + ", ".join(f"{k}={v:.2f}" for k, v in ranked))
        else:
            self._set_info(13, "Top candidates: -")

        if self._dynamic_status_text != "neutral":
            self._set_info(14, f"Dynamic gesture: {self._dynamic_summary_text}")
        else:
            self._set_info(14, "Dynamic gesture: -")

        self._set_info(15, f"Volume control: {self._volume_message}")
        mute_suffix = " [muted]" if self._volume_muted else ""
        if self._volume_level is None:
            self._set_info(16, f"Volume level: -   ({self._volume_status_text}{mute_suffix})")
        else:
            self._set_info(16, f"Volume level: {int(self._volume_level * 100)}   ({self._volume_status_text}{mute_suffix})")
        self._set_info(17, f"Voice mode: {self._voice_mode_text()}")
        self._set_info(18, f"Voice control: {self._voice_control_text}")
        self._set_info(19, f"Voice heard: {self._voice_preview_text(self._voice_display_text)}")

        self._update_volume_widgets()

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _update_dynamic_status(self, result, now: float) -> None:
        dynamic_scores = result.dynamic_candidate_scores or {}
        direct_dynamic = result.raw_gesture if result.raw_gesture in {"swipe_left", "swipe_right"} else None
        if not dynamic_scores and direct_dynamic is None and now >= self._dynamic_hold_until:
            self._dynamic_status_text = "neutral"
            self._dynamic_summary_text = "-"
            return

        if now >= self._horizontal_swipe_block_until:
            self._horizontal_swipe_block_label = None
            self._horizontal_swipe_block_until = 0.0

        if direct_dynamic is not None:
            self._dynamic_status_text = direct_dynamic
            self._dynamic_summary_text = self._format_dynamic_label(direct_dynamic)
            self._dynamic_hold_until = now + 0.85
            if direct_dynamic == "swipe_left":
                self._horizontal_swipe_block_label = "swipe_right"
                self._horizontal_swipe_block_until = now + 1.2
                self._mute_block_until = max(self._mute_block_until, now + 0.5)
            elif direct_dynamic == "swipe_right":
                self._horizontal_swipe_block_label = "swipe_left"
                self._horizontal_swipe_block_until = now + 1.2
                self._mute_block_until = max(self._mute_block_until, now + 0.5)
            return

        ranked = sorted(dynamic_scores.items(), key=lambda kv: kv[1], reverse=True)
        thresholds = {
            "swipe_left": 0.60,
            "swipe_right": 0.68,
        }
        selected_label: str | None = None
        for label, score in ranked:
            if label not in thresholds or score < thresholds[label]:
                continue
            if (
                label in {"swipe_left", "swipe_right"}
                and self._horizontal_swipe_block_label == label
                and now < self._horizontal_swipe_block_until
            ):
                continue
            selected_label = label
            break

        if selected_label is None:
            if now >= self._dynamic_hold_until:
                self._dynamic_status_text = "neutral"
                self._dynamic_summary_text = "-"
            return

        self._dynamic_status_text = selected_label
        self._dynamic_summary_text = self._format_dynamic_label(selected_label)
        self._dynamic_hold_until = now + 0.85
        if selected_label == "swipe_left":
            self._horizontal_swipe_block_label = "swipe_right"
            self._horizontal_swipe_block_until = now + 1.2
            self._mute_block_until = max(self._mute_block_until, now + 0.5)
        elif selected_label == "swipe_right":
            self._horizontal_swipe_block_label = "swipe_left"
            self._horizontal_swipe_block_until = now + 1.2
            self._mute_block_until = max(self._mute_block_until, now + 0.5)

    def _handle_volume_control(self, result, now: float) -> None:
        if not self.volume_controller.available:
            self._volume_message = self.volume_controller.message
            self._volume_mode_active = False
            self._volume_status_text = 'unavailable'
            self._volume_overlay_visible = False
            self._update_volume_widgets()
            return

        current_level = self.volume_controller.get_level()
        current_muted = self._read_system_mute()
        update = self.volume_tracker.update(
            features=result.features,
            landmarks=result.landmarks,
            candidate_scores=result.candidate_scores,
            stable_gesture=result.stable_gesture,
            current_level=current_level,
            current_muted=current_muted,
            now=now,
            allow_mute_toggle=now >= self._mute_block_until,
        )

        controller_error_message: str | None = None
        controller_error_status: str | None = None
        if update.trigger_mute_toggle:
            toggled = self.volume_controller.toggle_mute()
            if toggled is not None:
                current_muted = toggled
            else:
                controller_error_message = self.volume_controller.message or 'mute failed'
                controller_error_status = 'error'

        if update.active and update.level is not None:
            if self.volume_controller.set_level(update.level):
                current_level = update.level
                read_back_level = self.volume_controller.get_level()
                if read_back_level is not None:
                    current_level = read_back_level
            else:
                controller_error_message = self.volume_controller.message or 'set_level failed'
                controller_error_status = 'error'

        self._volume_mode_active = update.active
        self._volume_level = current_level if current_level is not None else update.level
        self._volume_muted = current_muted
        self._volume_message = controller_error_message or update.message
        self._volume_status_text = controller_error_status or update.status
        self._volume_overlay_visible = update.overlay_visible
        self._update_volume_widgets()

    def _voice_mode_text(self) -> str:
        if self._dictation_active:
            return "dictation"
        if self._voice_listening:
            if self._voice_mode == "general":
                return "command"
            return self._voice_mode.replace("_", " ")
        return "ready"

    def _voice_preview_text(self, text: str, *, max_chars: int = 220) -> str:
        value = str(text or "").strip()
        if not value:
            return "-"
        if len(value) <= max_chars:
            return value
        return "..." + value[-(max_chars - 3):]

    def _handle_left_hand_voice(self, stable_label: str, now: float) -> None:
        trigger_labels = {"one", "two"}
        if stable_label == self._voice_latched_label:
            if stable_label not in trigger_labels:
                self._voice_latched_label = None
            return

        if stable_label not in trigger_labels:
            self._reset_voice_candidate(now)
            return

        if stable_label != self._voice_candidate:
            self._voice_candidate = stable_label
            self._voice_candidate_since = now
            return

        if stable_label == "one" and (self._voice_listening or self._dictation_active):
            return
        if now < self._voice_cooldown_until:
            return
        if now - self._voice_candidate_since < 0.5:
            return

        self._voice_latched_label = stable_label
        self._voice_cooldown_until = now + 1.25
        if stable_label == "two":
            if self._dictation_active:
                self._stop_dictation_capture()
            else:
                self._start_dictation_capture()
            return
        self._start_voice_command()

    def _reset_voice_candidate(self, now: float) -> None:
        self._voice_candidate = "neutral"
        self._voice_candidate_since = now
        if self._voice_latched_label is not None:
            self._voice_latched_label = None

    def _start_voice_command(self) -> None:
        self._start_voice_capture(mode="general")

    def _start_dictation_capture(self) -> None:
        if self._voice_listening:
            return
        if not self.text_input_controller.available:
            self._voice_control_text = self.text_input_controller.message
            return
        self.dictation_processor.reset()
        self._start_voice_capture(mode="dictation")

    def _stop_dictation_capture(self) -> None:
        if not self._dictation_active:
            return
        self._dictation_active = False
        self._voice_control_text = "ending dictation..."
        if self._voice_stop_event is not None:
            self._voice_stop_event.set()

    def _start_voice_capture(self, *, mode: str) -> None:
        if self._voice_listening:
            return
        self._voice_listening = True
        self._voice_mode = mode
        self._voice_request_id += 1
        request_id = self._voice_request_id
        self._voice_stop_event = threading.Event() if mode == "dictation" else None
        self._dictation_active = mode == "dictation"
        self._dictation_backend = "local_stream" if mode == "dictation" else "idle"
        self._voice_control_text = "dictation listening..." if mode == "dictation" else "voice listening..."
        self._voice_heard_text = "-"
        self._voice_display_text = "-"

        def _push_status(status: str) -> None:
            self._voice_queue.put((request_id, {"event": "status", "status": status}))

        def _worker() -> None:
            if mode == "dictation":
                stop_event = self._voice_stop_event
                final_display = "-"
                final_message = "dictation stopped"
                pending_raw = ""
                pending_display = ""

                def _combined_display() -> str:
                    combined = f"{self.dictation_processor.full_text}{pending_display}"
                    return combined or "-"

                if self.live_dictation_streamer.available:
                    def _handle_live_event(event) -> None:
                        nonlocal final_display, final_message, pending_display, pending_raw
                        event_name = str(event.event or "").strip().lower()
                        if event_name == "ready":
                            _push_status("listening")
                            return
                        if event_name == "hypothesis":
                            pending_raw = str(event.text or "").strip()
                            preview_text = self.dictation_processor.preview(pending_raw)
                            if preview_text == pending_display:
                                return
                            if not self.text_input_controller.replace_text(pending_display, preview_text):
                                final_message = self.text_input_controller.message
                                if stop_event is not None:
                                    stop_event.set()
                                return
                            pending_display = preview_text
                            final_display = _combined_display()
                            self._voice_queue.put(
                                (
                                    request_id,
                                    {
                                        "event": "dictation_chunk",
                                        "success": True,
                                        "heard_text": pending_raw,
                                        "control_text": "live dictating...",
                                        "display_text": final_display,
                                        "partial": True,
                                    },
                                )
                            )
                            return
                        if event_name == "final":
                            final_raw = str(event.text or "").strip()
                            if not final_raw:
                                return
                            final_rendered = self.dictation_processor.preview(final_raw)
                            if final_rendered != pending_display:
                                if not self.text_input_controller.replace_text(pending_display, final_rendered):
                                    final_message = self.text_input_controller.message
                                    if stop_event is not None:
                                        stop_event.set()
                                    return
                            update = self.dictation_processor.ingest(final_raw)
                            pending_raw = ""
                            pending_display = ""
                            final_display = update.display_text or final_display
                            self._voice_queue.put(
                                (
                                    request_id,
                                    {
                                        "event": "dictation_chunk",
                                        "success": True,
                                        "heard_text": update.raw_text,
                                        "control_text": "live dictation committed",
                                        "display_text": final_display,
                                        "partial": False,
                                    },
                                )
                            )
                            return
                        if event_name == "error":
                            final_message = str(event.text or self.live_dictation_streamer.message or "live dictation failed")
                            if stop_event is not None:
                                stop_event.set()

                    stream_ok = self.live_dictation_streamer.stream(
                        stop_event=stop_event,
                        event_callback=_handle_live_event,
                    )
                    if pending_raw:
                        update = self.dictation_processor.ingest(pending_raw)
                        final_display = update.display_text or final_display
                        pending_raw = ""
                        pending_display = ""
                    if not stream_ok and final_message == "dictation stopped":
                        final_message = self.live_dictation_streamer.message or final_message
                else:
                    while True:
                        result = self.voice_listener.listen(
                            max_seconds=18.0,
                            status_callback=_push_status,
                            stop_event=stop_event,
                            transcript_mode="dictation",
                        )
                        if result.success and result.heard_text:
                            update = self.dictation_processor.ingest(result.heard_text)
                            final_display = update.display_text or final_display
                            if update.text_to_insert:
                                inserted = self.text_input_controller.insert_text(update.text_to_insert)
                                control_text = self.text_input_controller.message
                            else:
                                inserted = False
                                control_text = "dictation heard no insertable text"
                            self._voice_queue.put(
                                (
                                    request_id,
                                    {
                                        "event": "dictation_chunk",
                                        "success": inserted,
                                        "heard_text": update.raw_text,
                                        "control_text": control_text,
                                        "display_text": update.display_text,
                                        "partial": False,
                                    },
                                )
                            )
                        elif result.message:
                            lowered = result.message.lower()
                            if "failed" in lowered or "unavailable" in lowered:
                                final_message = result.message
                                break
                        if stop_event is not None and stop_event.is_set():
                            break
                if self.dictation_processor.full_text:
                    final_display = self.dictation_processor.full_text
                self._voice_queue.put(
                    (
                        request_id,
                        {
                            "event": "dictation_complete",
                            "success": bool(self.dictation_processor.full_text),
                            "heard_text": "",
                            "control_text": final_message,
                            "display_text": final_display,
                        },
                    )
                )
                return

            result = self.voice_listener.listen(max_seconds=12.0, status_callback=_push_status)
            if result.success:
                execution = self.voice_processor.execute(result.heard_text)
                payload = {
                    "event": "result",
                    "success": execution.success,
                    "target": execution.target,
                    "heard_text": execution.heard_text,
                    "control_text": execution.control_text,
                    "info_text": execution.info_text,
                }
            else:
                payload = {
                    "event": "result",
                    "success": False,
                    "target": "voice",
                    "heard_text": result.heard_text,
                    "control_text": result.message,
                    "info_text": "-",
                }
            self._voice_queue.put((request_id, payload))

        self._voice_thread = threading.Thread(target=_worker, name="hgr-debug-voice-command", daemon=True)
        self._voice_thread.start()

    def _drain_voice_results(self) -> None:
        while True:
            try:
                request_id, payload = self._voice_queue.get_nowait()
            except queue.Empty:
                break
            if request_id != self._voice_request_id:
                continue
            event = str(payload.get("event", "result"))
            if event == "status":
                status = str(payload.get("status", "") or "").strip().lower()
                if status == "listening":
                    self._voice_control_text = "dictation listening..." if self._voice_mode == "dictation" else "voice listening..."
                elif status == "recognizing":
                    self._voice_control_text = "transcribing dictation..." if self._voice_mode == "dictation" else "recognizing..."
                continue
            if event == "dictation_chunk":
                heard_text = str(payload.get("heard_text", "") or "").strip()
                display_text = str(payload.get("display_text", "") or "").strip()
                partial = bool(payload.get("partial"))
                if heard_text:
                    self._voice_heard_text = heard_text
                if display_text:
                    self._voice_display_text = display_text
                self._voice_control_text = str(payload.get("control_text", "dictation updated"))
                if partial:
                    self._voice_control_text = "live dictating..."
                continue
            if event == "dictation_complete":
                self._voice_listening = False
                self._dictation_active = False
                self._dictation_backend = "idle"
                self._voice_mode = "ready"
                self._voice_stop_event = None
                display_text = str(payload.get("display_text", "") or "").strip()
                if display_text:
                    self._voice_display_text = display_text
                self._voice_control_text = str(payload.get("control_text", "dictation stopped"))
                continue
            self._voice_listening = False
            self._dictation_active = False
            self._dictation_backend = "idle"
            self._voice_mode = "ready"
            self._voice_stop_event = None
            heard_text = str(payload.get("heard_text", "") or "").strip()
            self._voice_heard_text = heard_text or "-"
            self._voice_display_text = heard_text or "-"
            self._voice_control_text = str(payload.get("control_text", "voice idle"))

    def _reset_voice_state(self) -> None:
        if self._voice_stop_event is not None:
            self._voice_stop_event.set()
            self._voice_stop_event = None
        self.dictation_processor.reset()
        self._voice_candidate = "neutral"
        self._voice_candidate_since = 0.0
        self._voice_cooldown_until = 0.0
        self._voice_latched_label = None
        self._voice_request_id += 1
        self._voice_listening = False
        self._dictation_active = False
        self._dictation_backend = "idle"
        self._voice_mode = "ready"
        self._voice_control_text = self.voice_listener.message
        self._voice_heard_text = "-"
        self._voice_display_text = "-"
        while True:
            try:
                self._voice_queue.get_nowait()
            except queue.Empty:
                break

    def _read_system_mute(self) -> bool:
        muted = self.volume_controller.get_mute()
        return bool(muted) if muted is not None else False

    def _update_volume_widgets(self) -> None:
        level = 0.0 if self._volume_level is None else max(0.0, min(1.0, float(self._volume_level)))
        bar_rect = self.volume_bar_bg.contentsRect()
        bar_width = max(0, int(bar_rect.width() * level))
        self.volume_bar_fill.setFixedWidth(bar_width)

        if not self.volume_controller.available:
            self.volume_bar_card.hide()
            self.volume_bar_text.setText("Volume unavailable")
            if self.volume_overlay.isVisible():
                self.volume_overlay.hide_overlay()
        else:
            self.volume_bar_card.show()
            prefix = "Active" if self._volume_mode_active else "Idle"
            mute_suffix = " [muted]" if self._volume_muted else ""
            self.volume_bar_text.setText(f"{prefix}: {int(round(level * 100))}{mute_suffix}")
            self.volume_overlay.set_level(
                self._volume_level,
                muted=self._volume_muted,
                active=self._volume_mode_active,
                message=self._volume_message if self._volume_mode_active else self._volume_status_text,
            )
            if self._volume_overlay_visible:
                if not self.volume_overlay.isVisible():
                    self.volume_overlay.show_overlay()
            elif self.volume_overlay.isVisible():
                self.volume_overlay.hide_overlay()

    def _format_dynamic_label(self, label: str) -> str:
        return label.replace('_', ' ')

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_volume_widgets()

    def _set_info(self, index: int, text: str) -> None:
        if 0 <= index < len(self.info_labels):
            self.info_labels[index].setText(text)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_session()
        super().closeEvent(event)

# Author: Konstantin Markov

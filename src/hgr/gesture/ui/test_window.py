from __future__ import annotations

import math
import queue
import re
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QPixmap, QRegion
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...app.camera.camera_utils import open_preferred_or_first_available
from ...debug.chrome_controller import ChromeController
from ...debug.chrome_gesture_router import ChromeGestureRouter
from ...debug.mouse_controller import MouseController
from ...debug.mouse_gesture import MouseGestureTracker
from ...debug.mouse_overlay import draw_mouse_control_box_overlay, draw_mouse_monitor_overlay
from ...config.app_config import AppConfig
from ...voice.live_dictation import LiveDictationStreamer
from ...debug.screen_volume_overlay import ScreenVolumeOverlay
from ...debug.spotify_controller import SpotifyController
from ...debug.spotify_gesture_router import SpotifyGestureRouter
from ...debug.text_input_controller import TextInputController
from ...debug.voice_command_listener import VoiceCommandListener
from ...debug.volume_controller import VolumeController
from ...debug.volume_gesture import VolumeGestureTracker
from ...voice.command_processor import VoiceCommandContext, VoiceCommandProcessor
from ...voice.dictation import DictationProcessor
from ...app.ui.native_overlay import apply_overlay
from ..recognition.engine import GestureRecognitionEngine
from ..rendering.overlay import draw_hand_overlay
from .voice_status_overlay import VoiceStatusOverlay


class SpotifyWheelOverlay(QWidget):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._items: tuple[tuple[str, str, float], ...] = ()
        self._selected_key: str | None = None
        self._selection_progress = 0.0
        self._status_text = "Gesture Wheel"
        self._cursor_offset: tuple[float, float] | None = None
        # Cache for the static layer (slices in neutral state, rings,
        # center disc, title). set_wheel() runs every 15 ms tick so
        # the per-frame paint cost dominates if we re-render text +
        # 8 sector paths every time. Static slices change only when
        # the wheel variant or theme changes â€” cache them in a
        # pixmap; per-frame paint just blits + draws the active
        # slice overlay, selection arc, cursor, and subtitle.
        self._static_pixmap: QPixmap | None = None
        self._static_signature: tuple = ()
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: none;")
        self.setFixedSize(472, 472)
        self._apply_circle_mask()

    def show_overlay(self) -> None:
        import sys
        self._place_on_screen()
        self._apply_circle_mask()
        self.show()
        if sys.platform != "darwin":
            # raise_() activates the whole app on macOS, stealing focus from
            # the app you're controlling (Chrome, etc.). apply_overlay orders
            # the wheel front NON-activating via the nonactivating NSPanel path.
            self.raise_()
        self.repaint()
        apply_overlay(self)

    def hide_overlay(self) -> None:
        self.hide()

    def apply_theme(self, config: AppConfig) -> None:
        self.config = config
        # Theme colors are baked into the static pixmap â€” invalidate
        # so the next paintEvent regenerates it.
        self._static_pixmap = None
        self._static_signature = ()
        self.update()

    def set_wheel(
        self,
        *,
        items: tuple[tuple[str, str, float], ...],
        selected_key: str | None,
        selection_progress: float,
        status_text: str,
        cursor_offset: tuple[float, float] | None = None,
    ) -> None:
        new_progress = max(0.0, min(1.0, float(selection_progress)))
        # Skip the update() when no visible change occurred. The
        # caller drives this method every engine tick (~30 Hz). When
        # the user is steady on a slice, selection_progress crawls
        # at ~1% / tick and cursor moves are sub-pixel. Each skipped
        # paintEvent saves ~1.5 ms (cached blit + dynamic redraw)
        # and the next real change always re-fires update().
        items_changed = items != self._items
        selected_changed = selected_key != self._selected_key
        progress_changed = abs(new_progress - self._selection_progress) >= 0.005
        status_changed = status_text != self._status_text
        cursor_changed = self._cursor_offset_changed(cursor_offset)
        self._items = items
        self._selected_key = selected_key
        self._selection_progress = new_progress
        self._status_text = status_text
        self._cursor_offset = cursor_offset
        if (
            items_changed
            or selected_changed
            or progress_changed
            or status_changed
            or cursor_changed
        ):
            self.update()

    def _cursor_offset_changed(
        self, new_offset: tuple[float, float] | None
    ) -> bool:
        prev = self._cursor_offset
        if (prev is None) != (new_offset is None):
            return True
        if prev is None or new_offset is None:
            return False
        # Threshold at ~1px on a 472px wheel (radius ~190 px). Below
        # this the cursor circle (18 px) does not visibly move.
        return (
            abs(prev[0] - new_offset[0]) >= 0.005
            or abs(prev[1] - new_offset[1]) >= 0.005
        )

    def _place_on_screen(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.move(40, 40)
            return
        geo = screen.availableGeometry()
        x = geo.center().x() - self.width() // 2
        y = geo.center().y() - self.height() // 2
        self.move(x, y)

    def _apply_circle_mask(self) -> None:
        bounds = self.rect().adjusted(7, 7, -7, -7)
        if bounds.width() <= 0 or bounds.height() <= 0:
            return
        self.setMask(QRegion(bounds, QRegion.Ellipse))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_circle_mask()

    def _hex_to_qcolor(self, color: str, fallback: str) -> QColor:
        value = str(color or "").strip()
        if not value.startswith("#") or len(value) != 7:
            value = fallback
        return QColor(value)

    def _sector_path(
        self,
        cx: float,
        cy: float,
        inner_radius: float,
        outer_radius: float,
        start_deg: float,
        end_deg: float,
        steps: int = 22,
    ) -> QPainterPath:
        path = QPainterPath()
        outer_points = []
        inner_points = []
        for i in range(steps + 1):
            t = i / steps
            angle = start_deg + (end_deg - start_deg) * t
            radians = math.radians(angle)
            outer_points.append((cx + math.cos(radians) * outer_radius, cy - math.sin(radians) * outer_radius))
        for i in range(steps + 1):
            t = i / steps
            angle = end_deg + (start_deg - end_deg) * t
            radians = math.radians(angle)
            inner_points.append((cx + math.cos(radians) * inner_radius, cy - math.sin(radians) * inner_radius))

        if outer_points:
            path.moveTo(*outer_points[0])
            for point in outer_points[1:]:
                path.lineTo(*point)
            for point in inner_points:
                path.lineTo(*point)
            path.closeSubpath()
        return path

    def _label_lines(self, text: str) -> str:
        cleaned = str(text).strip()
        explicit = {
            "Add to Playlist": "Add to\nPlaylist",
            "Add Playlist": "Add\nPlaylist",
            "Remove Playlist": "Remove\nPlaylist",
            "Add Queue": "Add\nQueue",
            "Remove Queue": "Remove\nQueue",
            "Remove from Liked": "Remove\nLiked",
            "Add to Liked": "Add\nLiked",
            "Remove from Queue": "Remove\nfrom Queue",
            "Remove from Playlist": "Remove\nfrom Playlist",
            "Bookmarks Manager": "Bookmarks\nManager",
            "Reopen Tab": "Reopen\nTab",
            "Gesture Wheel": "Gesture\nWheel",
        }
        if cleaned in explicit:
            return explicit[cleaned]

        words = cleaned.split()
        if len(words) <= 1:
            return cleaned
        if len(words) == 2:
            return "\n".join(words)
        if len(words) == 3:
            if len(words[0]) + len(words[1]) <= 9:
                return f"{words[0]} {words[1]}\n{words[2]}"
            return f"{words[0]}\n{' '.join(words[1:])}"
        if len(words) == 4:
            return f"{' '.join(words[:2])}\n{' '.join(words[2:])}"
        midpoint = (len(words) + 1) // 2
        return " ".join(words[:midpoint]) + "\n" + " ".join(words[midpoint:])

    def _label_font(self, label: str, *, active: bool = False, center: bool = False) -> QFont:
        lines = self._label_lines(label).splitlines()
        longest = max((len(line) for line in lines), default=0)
        if center:
            size = 16 if len(lines) == 1 else 15
        else:
            if len(lines) >= 3 or longest >= 13:
                size = 9
            elif len(lines) == 2 or longest >= 10:
                size = 10
            else:
                size = 11
            if active:
                size += 1
        font = QFont("Segoe UI", size)
        font.setBold(True)
        return font

    def _wheel_geometry(self) -> tuple[float, float, float, float, float, float, float, float]:
        rect = self.rect()
        cx = rect.center().x()
        cy = rect.center().y()
        outer_radius = min(rect.width(), rect.height()) * 0.40
        inner_radius = outer_radius * 0.50
        center_radius = inner_radius * 0.78
        slice_span = 360.0 / max(1, len(self._items))
        half_slice = slice_span / 2.0
        label_radius = inner_radius + (outer_radius - inner_radius) * 0.57
        label_box_w = max(128.0, min(168.0, 2.0 * label_radius * math.sin(math.radians(half_slice)) * 1.18))
        return cx, cy, outer_radius, inner_radius, center_radius, half_slice, label_radius, label_box_w

    def _paint_slice(
        self,
        painter: QPainter,
        *,
        target_angle: float,
        label: str,
        active: bool,
        cx: float,
        cy: float,
        inner_radius: float,
        outer_radius: float,
        half_slice: float,
        label_radius: float,
        label_box_w: float,
        label_box_h: float,
        outline: QColor,
        glow_slice: QColor,
        base_slice: QColor,
    ) -> None:
        start_deg = target_angle - half_slice
        end_deg = target_angle + half_slice
        path = self._sector_path(cx, cy, inner_radius, outer_radius, start_deg, end_deg)
        if active:
            # Active slice is drawn on top of the cached static layer
            # (which painted this slice neutral). Source composition
            # for the fill replaces those pixels exactly so no
            # alpha-blending of base under glow shows through.
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(glow_slice))
            painter.drawPath(path)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setPen(QPen(outline, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
            painter.setPen(QPen(QColor(255, 255, 255, 34), 1))
            painter.drawPath(path)
        else:
            painter.setPen(QPen(QColor(outline.red(), outline.green(), outline.blue(), 120), 2))
            painter.setBrush(QBrush(base_slice))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
            painter.drawPath(path)

        mid_angle = math.radians(target_angle)
        label_x = cx + math.cos(mid_angle) * label_radius
        label_y = cy - math.sin(mid_angle) * label_radius
        label_rect = QRectF(label_x - label_box_w / 2.0, label_y - label_box_h / 2.0, label_box_w, label_box_h)
        painter.setPen(QPen(QColor(8, 12, 18) if active else outline))
        painter.setFont(self._label_font(label, active=active))
        painter.drawText(label_rect, Qt.AlignCenter | Qt.TextWordWrap, self._label_lines(label))

    def _compute_static_signature(self) -> tuple:
        return (
            tuple((str(k), str(l), float(a)) for k, l, a in self._items),
            str(self.config.accent_color or ""),
            str(self.config.surface_color or ""),
            int(self.width()),
            int(self.height()),
        )

    def _build_static_pixmap(self) -> QPixmap | None:
        if not self._items or self.width() <= 0 or self.height() <= 0:
            return None
        pm = QPixmap(self.size())
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            cx, cy, outer_radius, inner_radius, center_radius, half_slice, label_radius, label_box_w = self._wheel_geometry()
            label_box_h = 88.0
            outline = QColor(235, 244, 255, 215)
            accent = self._hex_to_qcolor(self.config.accent_color, "#1DE9B6")
            surface = self._hex_to_qcolor(self.config.surface_color, "#0F172A")
            base_slice = QColor(24, 44, 78, 205)
            glow_slice = QColor(accent)
            glow_slice.setAlpha(235)

            for _key, label, target_angle in self._items:
                self._paint_slice(
                    painter,
                    target_angle=target_angle,
                    label=label,
                    active=False,
                    cx=cx, cy=cy,
                    inner_radius=inner_radius, outer_radius=outer_radius,
                    half_slice=half_slice,
                    label_radius=label_radius,
                    label_box_w=label_box_w, label_box_h=label_box_h,
                    outline=outline, glow_slice=glow_slice, base_slice=base_slice,
                )

            painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 235), 4))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRectF(cx - outer_radius, cy - outer_radius, 2 * outer_radius, 2 * outer_radius))

            painter.setPen(QPen(QColor(255, 255, 255, 88), 2))
            painter.drawEllipse(QRectF(cx - inner_radius, cy - inner_radius, 2 * inner_radius, 2 * inner_radius))

            painter.setPen(QPen(QColor(255, 255, 255, 105), 2))
            painter.setBrush(QBrush(surface))
            painter.drawEllipse(QRectF(cx - center_radius, cy - center_radius, 2 * center_radius, 2 * center_radius))

            painter.setPen(QPen(outline))
            title_font = QFont("Segoe UI", 14)
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.drawText(
                QRectF(cx - center_radius + 6, cy - 42, 2 * center_radius - 12, 44),
                Qt.AlignCenter,
                "Gesture Wheel",
            )
        finally:
            painter.end()
        return pm

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self._items:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        signature = self._compute_static_signature()
        if self._static_pixmap is None or signature != self._static_signature:
            self._static_pixmap = self._build_static_pixmap()
            self._static_signature = signature
        if self._static_pixmap is None:
            return
        painter.drawPixmap(0, 0, self._static_pixmap)

        cx, cy, outer_radius, inner_radius, center_radius, half_slice, label_radius, label_box_w = self._wheel_geometry()
        label_box_h = 88.0
        outline = QColor(235, 244, 255, 215)
        accent = self._hex_to_qcolor(self.config.accent_color, "#1DE9B6")
        glow_slice = QColor(accent)
        glow_slice.setAlpha(235)
        base_slice = QColor(24, 44, 78, 205)

        if self._selected_key is not None:
            for key, label, target_angle in self._items:
                if key != self._selected_key:
                    continue
                self._paint_slice(
                    painter,
                    target_angle=target_angle,
                    label=label,
                    active=True,
                    cx=cx, cy=cy,
                    inner_radius=inner_radius, outer_radius=outer_radius,
                    half_slice=half_slice,
                    label_radius=label_radius,
                    label_box_w=label_box_w, label_box_h=label_box_h,
                    outline=outline, glow_slice=glow_slice, base_slice=base_slice,
                )
                break
            # Redraw the outer + inner rings on top so the active
            # slice's outline doesn't overlap them. In the original
            # un-cached paint path, rings were drawn after every
            # slice so they masked all slice edges uniformly â€” we
            # have to replicate that layering here when we redraw
            # an active slice over the cached static layer.
            painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 235), 4))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRectF(cx - outer_radius, cy - outer_radius, 2 * outer_radius, 2 * outer_radius))
            painter.setPen(QPen(QColor(255, 255, 255, 88), 2))
            painter.drawEllipse(QRectF(cx - inner_radius, cy - inner_radius, 2 * inner_radius, 2 * inner_radius))

        if self._selected_key is not None and self._selection_progress > 0.0:
            progress_radius = center_radius + 16
            painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 245), 6))
            painter.setBrush(Qt.NoBrush)
            progress_rect = QRectF(cx - progress_radius, cy - progress_radius, 2 * progress_radius, 2 * progress_radius)
            painter.drawArc(progress_rect, 90 * 16, int(-360 * 16 * self._selection_progress))

        if self._cursor_offset is not None:
            dx, dy = self._cursor_offset
            cursor_radius = min(outer_radius * 0.78, math.hypot(dx, dy) * outer_radius * 0.85)
            cursor_angle = math.atan2(-dy, dx)
            cursor_x = cx + math.cos(cursor_angle) * cursor_radius
            cursor_y = cy - math.sin(cursor_angle) * cursor_radius
            painter.setPen(QPen(QColor(255, 255, 255, 240), 3))
            painter.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue(), 105)))
            painter.drawEllipse(QRectF(cursor_x - 9, cursor_y - 9, 18, 18))
            painter.setPen(QPen(QColor(255, 255, 255, 225), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRectF(cursor_x - 4, cursor_y - 4, 8, 8))

        subtitle_font = QFont("Segoe UI", 10)
        subtitle_font.setBold(False)
        painter.setFont(subtitle_font)
        painter.setPen(QPen(QColor(226, 235, 248, 210)))
        painter.drawText(
            QRectF(cx - center_radius + 10, cy + 6, 2 * center_radius - 20, 36),
            Qt.AlignCenter | Qt.TextWordWrap,
            self._status_text,
        )


class GestureTestWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.engine: GestureRecognitionEngine | None = None
        self.cap = None
        self.camera_info = None
        self._last_time = time.time()
        self._fps = 0.0
        self._dynamic_hold_label = "neutral"
        self._dynamic_hold_until = 0.0
        self.volume_controller = VolumeController()
        self.volume_overlay = ScreenVolumeOverlay(config)
        self.volume_overlay.attach_controller(self.volume_controller)
        self.volume_tracker = VolumeGestureTracker(confirm_frames=2, release_frames=3, smoothing=0.22, pose_grace_seconds=0.42, no_hand_grace_seconds=0.22)
        self._volume_message = self.volume_controller.message
        self._volume_mode_active = False
        self._volume_level: float | None = self.volume_controller.get_level()
        self._volume_status_text = "idle"
        self._volume_muted = self._read_system_mute()
        self._volume_overlay_visible = False
        self._mute_block_until = 0.0
        self.mouse_controller = MouseController()
        self.mouse_tracker = MouseGestureTracker(
            control_box_center_x=config.mouse_control_box_center_x,
            control_box_center_y=config.mouse_control_box_center_y,
            control_box_area=config.mouse_control_box_area,
            control_box_aspect_power=config.mouse_control_box_aspect_power,
        )
        self._mouse_mode_enabled = False
        self._mouse_control_text = "mouse mode off" if self.mouse_controller.available else self.mouse_controller.message
        self._mouse_status_text = "off" if self.mouse_controller.available else "unavailable"
        self.chrome_controller = ChromeController()
        self.chrome_router = ChromeGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=1.5)
        self._chrome_mode_enabled = False
        self._chrome_control_text = self.chrome_controller.message
        self._chrome_wheel_candidate = "neutral"
        self._chrome_wheel_candidate_since = 0.0
        self._chrome_wheel_visible = False
        self._chrome_wheel_anchor = None
        self._chrome_wheel_selected_key: str | None = None
        self._chrome_wheel_selected_since = 0.0
        self._chrome_wheel_pose_grace_until = 0.0
        self._chrome_wheel_cooldown_until = 0.0
        self._chrome_wheel_cursor_offset: tuple[float, float] | None = None
        self._chrome_active_cache = False
        self._chrome_active_cache_until = 0.0
        self.spotify_controller = SpotifyController()
        self.spotify_router = SpotifyGestureRouter(static_hold_seconds=0.5, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=1.5)
        self._spotify_control_text = self.spotify_controller.message
        self._spotify_info_text = "-"
        self._spotify_wheel_candidate = "neutral"
        self._spotify_wheel_candidate_since = 0.0
        self._spotify_wheel_visible = False
        self._spotify_wheel_anchor = None
        self._spotify_wheel_selected_key: str | None = None
        self._spotify_wheel_selected_since = 0.0
        self._spotify_wheel_pose_grace_until = 0.0
        self._spotify_wheel_cooldown_until = 0.0
        self._spotify_wheel_cursor_offset: tuple[float, float] | None = None
        self._spotify_active_cache = False
        self._spotify_active_cache_until = 0.0
        self._spotify_wheel_cursor_offset = None
        self.spotify_wheel_overlay = SpotifyWheelOverlay(config)
        self.chrome_wheel_overlay = SpotifyWheelOverlay(config)
        self.voice_listener = VoiceCommandListener()
        self.voice_processor = VoiceCommandProcessor(
            chrome_controller=self.chrome_controller,
            spotify_controller=self.spotify_controller,
        )
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
        self._selection_prompt_active = False
        self._selection_auto_listen_at = 0.0
        self.voice_status_overlay = VoiceStatusOverlay(config)
        try:
            self.voice_status_overlay.selectionChosen.connect(self._handle_voice_overlay_selection)
        except Exception:
            pass

        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._tick)

        self.setWindowTitle("Touchless Gesture Test")
        self.resize(1240, 820)
        self.setMinimumSize(1080, 720)
        self._build_ui()
        self._apply_theme()
        QTimer.singleShot(0, self.start_session)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        left = QFrame()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        self.video_label = QLabel("Opening camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(740, 560)
        left_layout.addWidget(self.video_label, 1)

        self.banner_label = QLabel("Gesture: neutral")
        self.banner_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.banner_label)
        layout.addWidget(left, 2)

        right = QFrame()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(6)

        self.info_labels: list[QLabel] = []
        for text in (
            "Camera: waiting",
            "Handedness: -",
            "Gesture raw/stable: -",
            "Confidence: 0.00",
            "FPS: 0.0",
            "Box: -",
            "Palm: -",
            "Dynamic: -",
            "Candidates: -",
            "Thumb: -",
            "Index: -",
            "Middle: -",
            "Ring: -",
            "Pinky: -",
            "Spreads: -",
            "Reasoning: -",
            "Volume control: unavailable",
            "Volume level: -",
            "Spotify control: -",
            "Spotify info: -",
            "Chrome mode: off",
            "Chrome control: -",
            "Voice mode: ready",
            "Voice control: -",
            "Voice heard: -",
            "Mouse mode: off",
            "Mouse control: -",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            right_layout.addWidget(label)
            self.info_labels.append(label)

        self.volume_bar_card = QFrame()
        self.volume_bar_card.setMaximumWidth(128)
        volume_card_layout = QVBoxLayout(self.volume_bar_card)
        volume_card_layout.setContentsMargins(10, 10, 10, 10)
        volume_card_layout.setSpacing(6)

        self.volume_bar_title = QLabel("Vol.")
        volume_card_layout.addWidget(self.volume_bar_title)

        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setTextVisible(False)
        self.volume_bar.setFixedHeight(8)
        volume_card_layout.addWidget(self.volume_bar)

        self.volume_bar_text = QLabel("Volume inactive")
        self.volume_bar_text.setAlignment(Qt.AlignCenter)
        volume_card_layout.addWidget(self.volume_bar_text)
        right_layout.addWidget(self.volume_bar_card)

        controls = QHBoxLayout()
        restart = QPushButton("Restart Camera")
        restart.clicked.connect(self.start_session)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        controls.addWidget(restart)
        controls.addWidget(close)
        right_layout.addLayout(controls)

        gesture_note = QLabel("Voice: hold left one for commands. Hold left two to start dictation, and hold left two again to stop.")
        gesture_note.setWordWrap(True)
        right_layout.addWidget(gesture_note)
        layout.addWidget(right, 1)

    def _apply_theme(self) -> None:
        self.spotify_wheel_overlay.apply_theme(self.config)
        self.chrome_wheel_overlay.apply_theme(self.config)
        self.voice_status_overlay.apply_theme(self.config)
        self.setStyleSheet(
            f"""
            QWidget {{
                background: {self.config.surface_color};
                color: {self.config.text_color};
            }}
            QFrame {{
                border: 1px solid rgba(29,233,182,0.24);
                border-radius: 14px;
                background: rgba(255,255,255,0.04);
            }}
            QLabel {{
                background: transparent;
            }}
            QPushButton {{
                background: rgba(255,255,255,0.08);
                color: {self.config.text_color};
                border-radius: 10px;
                padding: 6px 12px;
                border: 1px solid rgba(255,255,255,0.10);
            }}
            QPushButton:hover {{
                border: 1px solid {self.config.accent_color};
            }}
            QProgressBar {{
                background-color: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 9px;
            }}
            QProgressBar::chunk {{
                background-color: {self.config.accent_color};
                border-radius: 9px;
            }}
            """
        )

    def start_session(self) -> None:
        self.stop_session()
        self.engine = GestureRecognitionEngine(stable_frames_required=max(2, self.config.stable_frames_required // 2))
        self._volume_message = self.volume_controller.message
        self._volume_level = self.volume_controller.refresh_cache().level_scalar
        self._volume_mode_active = False
        self._volume_status_text = "idle"
        self._volume_muted = self._read_system_mute()
        self._volume_overlay_visible = False
        self._mute_block_until = 0.0
        self.mouse_controller.release_all()
        self.mouse_tracker.reset()
        self._mouse_mode_enabled = False
        self._mouse_control_text = "mouse mode off" if self.mouse_controller.available else self.mouse_controller.message
        self._mouse_status_text = "off" if self.mouse_controller.available else "unavailable"
        self.chrome_router.reset()
        self._chrome_mode_enabled = False
        self._chrome_control_text = self.chrome_controller.message
        self._reset_chrome_wheel(clear_cooldown=True)
        self.chrome_wheel_overlay.hide_overlay()
        self.spotify_router.reset()
        self._spotify_control_text = self.spotify_controller.message
        self._spotify_info_text = "-"
        self._reset_spotify_wheel()
        self.spotify_wheel_overlay.hide_overlay()
        self._reset_voice_state()
        self.volume_tracker.reset(self._volume_level, self._volume_muted)
        self.volume_overlay.hide_overlay()
        self.voice_status_overlay.hide_overlay()
        self._update_volume_widgets()
        self.camera_info, self.cap = open_preferred_or_first_available(self.config.preferred_camera_index, max_index=self.config.camera_scan_limit)
        if self.cap is None or self.camera_info is None:
            QMessageBox.warning(self, "Touchless Gesture Test", "No available camera was found.")
            return
        self._set_info(0, f"Camera: {self.camera_info.display_name}")
        self._last_time = time.time()
        self._dynamic_hold_label = "neutral"
        self._dynamic_hold_until = 0.0
        self.timer.start()

    def stop_session(self) -> None:
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        self._dynamic_hold_label = "neutral"
        self._dynamic_hold_until = 0.0
        self._volume_mode_active = False
        self._volume_overlay_visible = False
        self._mute_block_until = 0.0
        self.mouse_controller.release_all()
        self.mouse_tracker.reset()
        self._mouse_mode_enabled = False
        self._mouse_control_text = "mouse mode off" if self.mouse_controller.available else self.mouse_controller.message
        self._mouse_status_text = "off" if self.mouse_controller.available else "unavailable"
        self.chrome_router.reset()
        self._chrome_mode_enabled = False
        self._chrome_control_text = self.chrome_controller.message
        self._reset_chrome_wheel(clear_cooldown=True)
        self.chrome_wheel_overlay.hide_overlay()
        self.spotify_router.reset()
        self._reset_spotify_wheel()
        self.spotify_wheel_overlay.hide_overlay()
        self.volume_tracker.reset(self._volume_level, self._volume_muted)
        self._reset_voice_state()
        self.volume_overlay.hide_overlay()
        self.voice_status_overlay.hide_overlay()

    def _tick(self) -> None:
        if self.cap is None or self.engine is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        result = self.engine.process_frame(frame)

        now = time.time()
        dt = max(now - self._last_time, 1e-6)
        self._fps = 0.86 * self._fps + 0.14 * (1.0 / dt) if self._fps else (1.0 / dt)
        self._last_time = now

        prediction = result.prediction
        dynamic_display = prediction.dynamic_label
        monotonic_now = time.monotonic()
        if prediction.dynamic_label != "neutral":
            self._dynamic_hold_label = prediction.dynamic_label
            self._dynamic_hold_until = monotonic_now + 0.85
        elif monotonic_now < self._dynamic_hold_until:
            dynamic_display = self._dynamic_hold_label
        else:
            self._dynamic_hold_label = "neutral"
            self._dynamic_hold_until = 0.0
        banner_text = prediction.stable_label if prediction.stable_label != "neutral" else prediction.raw_label
        self.banner_label.setText(f"Gesture: {banner_text} | Dynamic: {dynamic_display}")

        self._set_info(2, f"Gesture raw/stable: {prediction.raw_label} / {prediction.stable_label}")
        self._set_info(3, f"Confidence: {prediction.confidence:.2f}")
        self._set_info(4, f"FPS: {self._fps:.1f}")
        self._drain_voice_results()
        self._update_selection_listening(monotonic_now if "monotonic_now" in locals() else time.monotonic())
        hand_handedness = result.tracked_hand.handedness if result.found and result.tracked_hand is not None else None
        self._handle_volume_control(result, monotonic_now, hand_handedness=hand_handedness)
        self._handle_app_controls(prediction, result.hand_reading, hand_handedness, monotonic_now)

        self.voice_status_overlay.tick(monotonic_now)
        self._update_chrome_wheel_overlay(monotonic_now)
        self._update_spotify_wheel_overlay(monotonic_now)
        annotated = draw_hand_overlay(result.annotated_frame, result)
        self._draw_mouse_control_overlay(annotated)
        self._draw_mouse_monitor_overlay(annotated)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        image = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if not result.found or result.tracked_hand is None or result.hand_reading is None:
            self._set_info(1, "Handedness: -")
            self._set_info(5, "Box: -")
            self._set_info(6, "Palm: -")
            self._set_info(7, "Dynamic: -")
            self._set_info(8, "Candidates: -")
            for index in range(9, 15):
                self._set_info(index, "-")
            self._set_info(15, "Reasoning: no hand in frame")
            self._set_info(16, f"Volume control: {self._volume_message}")
            if self._volume_level is None:
                self._set_info(17, f"Volume level: - ({self._volume_status_text})")
            else:
                mute_suffix = " [muted]" if self._volume_muted else ""
                self._set_info(17, f"Volume level: {int(round(self._volume_level * 100))} ({self._volume_status_text}{mute_suffix})")
            self._set_info(18, f"Spotify control: {self._spotify_control_text}")
            self._set_info(19, f"Spotify info: {self._spotify_info_text}")
            self._set_info(20, f"Chrome mode: {'on' if self._chrome_mode_enabled else 'off'}")
            self._set_info(21, f"Chrome control: {self._chrome_control_text}")
            self._set_info(22, f"Voice mode: {self._voice_mode_text()}")
            self._set_info(23, f"Voice control: {self._voice_control_text}")
            self._set_info(24, f"Voice heard: {self._voice_preview_text(self._voice_display_text)}")
            self._set_info(25, f"Mouse mode: {'on' if self._mouse_mode_enabled else 'off'} ({self._mouse_status_text})")
            self._set_info(26, f"Mouse control: {self._mouse_control_text}")
            return

        hand = result.tracked_hand
        reading = result.hand_reading
        self._set_info(1, f"Handedness: {hand.handedness} ({hand.handedness_confidence:.2f})")
        self._set_info(5, f"Box: x={hand.bbox.x:.2f} y={hand.bbox.y:.2f} w={hand.bbox.width:.2f} h={hand.bbox.height:.2f}")
        self._set_info(6, f"Palm: roll={reading.palm.roll_deg:.1f} pitch={reading.palm.pitch_deg:.1f} yaw={reading.palm.yaw_deg:.1f}")
        self._set_info(7, f"Dynamic: {dynamic_display}")
        candidate_text = ", ".join(f"{candidate.label}={candidate.score:.2f}" for candidate in prediction.candidates[:4]) or "-"
        self._set_info(8, f"Candidates: {candidate_text}")
        for row, name in zip(range(9, 14), ("thumb", "index", "middle", "ring", "pinky")):
            finger = reading.fingers[name]
            self._set_info(
                row,
                f"{name.title()}: {finger.state} | open={finger.openness:.2f} curl={finger.curl:.2f} conf={finger.confidence:.2f} occ={finger.occluded}",
            )
        spread_text = ", ".join(f"{name}={spread.state}:{spread.distance:.2f}" for name, spread in reading.spreads.items())
        self._set_info(14, f"Spreads: {spread_text}")
        self._set_info(
            15,
            f"Reasoning: extended={reading.finger_count_extended} occlusion={reading.occlusion_score:.2f} shape={reading.shape_confidence:.2f}",
        )
        self._set_info(16, f"Volume control: {self._volume_message}")
        if self._volume_level is None:
            self._set_info(17, f"Volume level: - ({self._volume_status_text})")
        else:
            mute_suffix = " [muted]" if self._volume_muted else ""
            self._set_info(17, f"Volume level: {int(round(self._volume_level * 100))} ({self._volume_status_text}{mute_suffix})")
        self._set_info(18, f"Spotify control: {self._spotify_control_text}")
        self._set_info(19, f"Spotify info: {self._spotify_info_text}")
        self._set_info(20, f"Chrome mode: {'on' if self._chrome_mode_enabled else 'off'}")
        self._set_info(21, f"Chrome control: {self._chrome_control_text}")
        self._set_info(22, f"Voice mode: {self._voice_mode_text()}")
        self._set_info(23, f"Voice control: {self._voice_control_text}")
        self._set_info(24, f"Voice heard: {self._voice_preview_text(self._voice_display_text)}")
        self._set_info(25, f"Mouse mode: {'on' if self._mouse_mode_enabled else 'off'} ({self._mouse_status_text})")
        self._set_info(26, f"Mouse control: {self._mouse_control_text}")

    def _set_info(self, index: int, text: str) -> None:
        if 0 <= index < len(self.info_labels):
            self.info_labels[index].setText(text)

    def _begin_voice_selection_prompt(self, title: str, items: list[tuple[str, str, str]], instruction: str) -> None:
        self._selection_prompt_active = True
        self._selection_auto_listen_at = time.monotonic() + 3.0
        self.voice_status_overlay.show_selection(title, items, instruction, status_text="Waiting for voice...")

    def _update_selection_listening(self, now: float) -> None:
        if not self._selection_prompt_active:
            return
        if getattr(self.voice_status_overlay, "_mode", "") != "selection":
            self._selection_prompt_active = False
            self._selection_auto_listen_at = 0.0
            return
        if self._voice_listening:
            return
        if self._selection_auto_listen_at > 0.0 and now >= self._selection_auto_listen_at:
            self._selection_auto_listen_at = 0.0
            self._start_voice_capture(mode="selection", preferred_app=None)

    def _handle_volume_control(self, result, now: float, *, hand_handedness: str | None) -> None:
        if not self.volume_controller.available:
            self._volume_message = self.volume_controller.message
            self._volume_mode_active = False
            self._volume_status_text = "unavailable"
            self._volume_overlay_visible = False
            self._update_volume_widgets()
            return

        current_level = self.volume_controller.get_level()
        current_muted = self._read_system_mute()
        if self.mouse_tracker.mode_enabled and hand_handedness == "Right":
            self._volume_level = current_level
            self._volume_muted = current_muted
            self._volume_message = "mouse mode active"
            self._volume_mode_active = False
            self._volume_status_text = "paused"
            self._volume_overlay_visible = False
            self._update_volume_widgets()
            return
        if hand_handedness == "Right" and result.prediction.dynamic_label in {"swipe_left", "swipe_right"}:
            self._mute_block_until = max(self._mute_block_until, now + 0.5)

        features = None
        candidate_scores = {}
        landmarks = None
        stable_gesture = "neutral"
        if (
            hand_handedness == "Right"
            and result.found
            and result.tracked_hand is not None
            and result.hand_reading is not None
            and self.engine is not None
        ):
            landmarks = result.tracked_hand.landmarks
            features = self._volume_features_from_hand_reading(result.hand_reading)
            candidate_scores = self.engine.last_static_scores
            stable_gesture = result.prediction.stable_label

        update = self.volume_tracker.update(
            features=features,
            landmarks=landmarks,
            candidate_scores=candidate_scores,
            stable_gesture=stable_gesture,
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
                controller_error_message = self.volume_controller.message or "mute failed"
                controller_error_status = "error"

        if update.active and update.level is not None:
            if self.volume_controller.set_level(update.level):
                current_level = update.level
                read_back_level = self.volume_controller.get_level()
                if read_back_level is not None:
                    current_level = read_back_level
            else:
                controller_error_message = self.volume_controller.message or "set_level failed"
                controller_error_status = "error"

        if update.overlay_visible:
            live_status = self.volume_controller.sync_live_state()
            if live_status.level_scalar is not None:
                current_level = live_status.level_scalar
            live_muted = self.volume_controller.get_mute(prefer_cached=False)
            if live_muted is not None:
                current_muted = bool(live_muted)

        self._volume_mode_active = update.active
        self._volume_level = current_level if current_level is not None else update.level
        self._volume_muted = current_muted
        self._volume_message = controller_error_message or update.message
        self._volume_status_text = controller_error_status or update.status
        self._volume_overlay_visible = update.overlay_visible
        self._update_volume_widgets()

    def _handle_mouse_control(self, prediction, hand_reading, hand_handedness: str | None, now: float) -> bool:
        if not self.mouse_controller.available:
            self.mouse_tracker.reset()
            self._mouse_mode_enabled = False
            self._mouse_status_text = "unavailable"
            self._mouse_control_text = self.mouse_controller.message
            return False

        self.mouse_tracker.set_desktop_bounds(self.mouse_controller.virtual_bounds())
        update = self.mouse_tracker.update(
            hand_reading=hand_reading,
            prediction=prediction,
            hand_handedness=hand_handedness,
            cursor_seed=self.mouse_controller.current_position_normalized(),
            now=now,
        )

        if update.cursor_position is not None:
            self.mouse_controller.move_normalized(*update.cursor_position)
        if update.left_press:
            self.mouse_controller.left_down()
        if update.left_release:
            self.mouse_controller.left_up()
        if update.left_click:
            self.mouse_controller.left_click()
        if update.right_click:
            self.mouse_controller.right_click()
        if update.scroll_steps:
            self.mouse_controller.scroll(update.scroll_steps)

        self._mouse_mode_enabled = update.mode_enabled
        self._mouse_status_text = update.status
        action_text = update.control_text
        if update.left_press or update.left_release or update.left_click or update.right_click or update.scroll_steps:
            action_text = self.mouse_controller.message
        self._mouse_control_text = action_text
        return update.consume_other_routes

    def _handle_app_controls(self, prediction, hand_reading, hand_handedness: str | None, now: float) -> None:
        if self._dictation_active:
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            if hand_handedness == "Left":
                self._handle_left_hand_voice(prediction, now)
            else:
                self._reset_voice_candidate(now)
            return

        mouse_consuming = self._handle_mouse_control(
            prediction=prediction,
            hand_reading=hand_reading if hand_handedness == "Right" else None,
            hand_handedness=hand_handedness,
            now=now,
        )
        if mouse_consuming:
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._reset_voice_candidate(now)
            return

        if hand_handedness == "Left":
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._handle_left_hand_voice(prediction, now)
            return

        self._reset_voice_candidate(now)
        chrome_wheel_consuming = self._update_chrome_wheel(
            prediction=prediction if hand_handedness == "Right" else None,
            hand_reading=hand_reading if hand_handedness == "Right" else None,
            now=now,
            active=hand_handedness == "Right" and hand_reading is not None,
        )
        if chrome_wheel_consuming:
            return
        wheel_consuming = self._update_spotify_wheel(
            prediction=prediction if hand_handedness == "Right" else None,
            hand_reading=hand_reading if hand_handedness == "Right" else None,
            now=now,
            active=hand_handedness == "Right" and hand_reading is not None,
        )
        if wheel_consuming:
            return

        app_static_label = self._derive_app_static_label(prediction, hand_reading)
        chrome_snapshot = self.chrome_router.update(
            stable_label=app_static_label,
            dynamic_label=prediction.dynamic_label,
            controller=self.chrome_controller,
            now=now,
        )
        self._chrome_mode_enabled = chrome_snapshot.mode_enabled
        self._chrome_control_text = chrome_snapshot.control_text

        if chrome_snapshot.consume_other_routes or app_static_label in {"three", "three_together"}:
            return

        snapshot = self.spotify_router.update(
            stable_label=prediction.stable_label,
            dynamic_label=prediction.dynamic_label,
            controller=self.spotify_controller,
            now=now,
        )
        self._spotify_control_text = snapshot.control_text
        self._spotify_info_text = snapshot.info_text

    def _read_system_mute(self) -> bool:
        muted = self.volume_controller.get_mute()
        return bool(muted) if muted is not None else False

    def _update_volume_widgets(self) -> None:
        if self.volume_controller.available and self._volume_overlay_visible:
            live_status = self.volume_controller.sync_live_state()
            if live_status.level_scalar is not None:
                self._volume_level = live_status.level_scalar
            live_muted = self.volume_controller.get_mute(prefer_cached=False)
            if live_muted is not None:
                self._volume_muted = bool(live_muted)

        level = 0 if self._volume_level is None else max(0, min(100, int(round(self._volume_level * 100))))
        self.volume_bar.setValue(level)
        if not self.volume_controller.available:
            self.volume_bar_card.hide()
            self.volume_bar_text.setText("Volume unavailable")
            if self.volume_overlay.isVisible():
                self.volume_overlay.hide_overlay()
            return

        self.volume_bar_card.show()
        prefix = "Active" if self._volume_mode_active else "Idle"
        mute_suffix = " [muted]" if self._volume_muted else ""
        self.volume_bar_text.setText(f"{prefix}: {level}%{mute_suffix}")
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

    def _volume_features_from_hand_reading(self, hand_reading):
        fine_states = {name: finger.state for name, finger in hand_reading.fingers.items()}
        states = {
            name: ("open" if finger.state in {"fully_open", "partially_curled"} else "closed")
            for name, finger in hand_reading.fingers.items()
        }
        open_scores = {name: float(finger.openness) for name, finger in hand_reading.fingers.items()}
        spread_states = {name: spread.state for name, spread in hand_reading.spreads.items()}
        spread_ratios = {name: float(spread.distance) for name, spread in hand_reading.spreads.items()}
        spread_together_strengths = {name: float(spread.together_strength) for name, spread in hand_reading.spreads.items()}
        spread_apart_strengths = {name: float(spread.apart_strength) for name, spread in hand_reading.spreads.items()}
        return SimpleNamespace(
            palm_scale=float(hand_reading.palm.scale),
            open_scores=open_scores,
            states=states,
            fine_states=fine_states,
            finger_count_open=sum(1 for finger in hand_reading.fingers.values() if finger.extended),
            spread_states=spread_states,
            spread_ratios=spread_ratios,
            spread_together_strengths=spread_together_strengths,
            spread_apart_strengths=spread_apart_strengths,
        )

    def _derive_app_static_label(self, prediction, hand_reading):
        stable_label = prediction.stable_label
        if hand_reading is None:
            return stable_label
        if stable_label in {"three", "neutral"} and self._is_three_apart(hand_reading):
            return "three_apart"
        if stable_label in {"three", "neutral"} and self._is_three_together(hand_reading):
            return "three_together"
        if stable_label in {"four", "neutral"} and self._is_four_together(hand_reading):
            return "four_together"
        return stable_label

    def _is_three_together(self, hand_reading) -> bool:
        fingers = hand_reading.fingers
        return (
            self._is_chrome_open_finger(fingers["index"])
            and self._is_chrome_open_finger(fingers["middle"])
            and self._is_chrome_open_finger(fingers["ring"])
            and self._is_folded_finger(fingers["thumb"], allow_partial=True)
            and self._is_folded_finger(fingers["pinky"])
            and self._spread_is_together(hand_reading, "index_middle", max_distance=0.38, min_strength=0.56)
            and self._spread_is_together(hand_reading, "middle_ring", max_distance=0.38, min_strength=0.54)
        )

    def _is_three_apart(self, hand_reading) -> bool:
        fingers = hand_reading.fingers
        return (
            self._is_chrome_open_finger(fingers["index"])
            and self._is_chrome_open_finger(fingers["middle"])
            and self._is_chrome_open_finger(fingers["ring"])
            and self._is_folded_finger(fingers["thumb"], allow_partial=True)
            and self._is_folded_finger(fingers["pinky"])
            and self._spread_is_apart(hand_reading, "index_middle", min_distance=0.50, min_strength=0.56)
            and self._spread_is_apart(hand_reading, "middle_ring", min_distance=0.48, min_strength=0.54)
        )

    def _is_four_together(self, hand_reading) -> bool:
        fingers = hand_reading.fingers
        return (
            all(self._is_chrome_open_finger(fingers[name]) for name in ("index", "middle", "ring", "pinky"))
            and self._is_folded_finger(fingers["thumb"], allow_partial=True)
            and self._spread_is_together(hand_reading, "index_middle", max_distance=0.40, min_strength=0.52)
            and self._spread_is_together(hand_reading, "middle_ring", max_distance=0.40, min_strength=0.52)
            and self._spread_is_together(hand_reading, "ring_pinky", max_distance=0.40, min_strength=0.52)
        )

    def _is_chrome_open_finger(self, finger) -> bool:
        return finger.state == "fully_open" or (
            finger.state == "partially_curled"
            and finger.openness >= 0.70
            and finger.curl <= 0.46
        )

    def _is_folded_finger(self, finger, *, allow_partial: bool = False) -> bool:
        allowed = {"mostly_curled", "closed"}
        if allow_partial:
            allowed.add("partially_curled")
        return finger.state in allowed

    def _spread_is_together(self, hand_reading, spread_name: str, *, max_distance: float, min_strength: float) -> bool:
        spread = hand_reading.spreads[spread_name]
        return (
            spread.state == "together"
            or spread.distance <= max_distance
            or (spread.together_strength >= min_strength and spread.apart_strength <= 0.42)
        )

    def _spread_is_apart(self, hand_reading, spread_name: str, *, min_distance: float, min_strength: float) -> bool:
        spread = hand_reading.spreads[spread_name]
        return (
            spread.state == "apart"
            or spread.distance >= min_distance
            or (spread.apart_strength >= min_strength and spread.together_strength <= 0.42)
        )

    def _reset_spotify_wheel(self, *, clear_cooldown: bool = False) -> None:
        self._spotify_wheel_candidate = "neutral"
        self._spotify_wheel_candidate_since = 0.0
        self._spotify_wheel_visible = False
        self._spotify_wheel_anchor = None
        self._spotify_wheel_selected_key = None
        self._spotify_wheel_selected_since = 0.0
        self._spotify_wheel_pose_grace_until = 0.0
        self._spotify_wheel_cursor_offset = None
        self._spotify_active_cache = False
        self._spotify_active_cache_until = 0.0
        if clear_cooldown:
            self._spotify_wheel_cooldown_until = 0.0
        if hasattr(self, 'spotify_wheel_overlay') and self.spotify_wheel_overlay.isVisible():
            self.spotify_wheel_overlay.hide_overlay()

    def _reset_chrome_wheel(self, *, clear_cooldown: bool = False) -> None:
        self._chrome_wheel_candidate = "neutral"
        self._chrome_wheel_candidate_since = 0.0
        self._chrome_wheel_visible = False
        self._chrome_wheel_anchor = None
        self._chrome_wheel_selected_key = None
        self._chrome_wheel_selected_since = 0.0
        self._chrome_wheel_pose_grace_until = 0.0
        self._chrome_wheel_cursor_offset = None
        self._chrome_active_cache = False
        self._chrome_active_cache_until = 0.0
        if clear_cooldown:
            self._chrome_wheel_cooldown_until = 0.0
        if hasattr(self, "chrome_wheel_overlay") and self.chrome_wheel_overlay.isVisible():
            self.chrome_wheel_overlay.hide_overlay()

    def _spotify_wheel_items(self) -> tuple[tuple[str, str, float], ...]:
        labels = (
            ("add_playlist", "Add Playlist"),
            ("remove_playlist", "Remove Playlist"),
            ("add_queue", "Add Queue"),
            ("remove_queue", "Remove Queue"),
            ("like", "Add to Liked"),
            ("remove_liked", "Remove from Liked"),
            ("shuffle", "Shuffle"),
        )
        slice_span = 360.0 / len(labels)
        return tuple(
            (key, label, (90.0 - index * slice_span) % 360.0)
            for index, (key, label) in enumerate(labels)
        )

    def _chrome_wheel_items(self) -> tuple[tuple[str, str, float], ...]:
        return (
            ("bookmark", "Bookmark This Tab", 90.0),
            ("history", "History", 30.0),
            ("downloads", "Downloads", 330.0),
            ("bookmarks", "Bookmarks Manager", 270.0),
            ("print", "Print Page", 210.0),
            ("reopen", "Reopen Tab", 150.0),
        )

    def _chrome_active_for_wheel(self, now: float) -> bool:
        if now < self._chrome_active_cache_until:
            return self._chrome_active_cache
        active = bool(self.chrome_controller.is_window_active())
        self._chrome_active_cache = active
        self._chrome_active_cache_until = now + 0.55
        return active

    def _chrome_wheel_pose_active(self, prediction, now: float) -> bool:
        if prediction is None:
            return False
        stable_label = str(getattr(prediction, "stable_label", "neutral") or "neutral")
        raw_label = str(getattr(prediction, "raw_label", "neutral") or "neutral")
        confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        if stable_label == "chrome_wheel_pose":
            return True
        if raw_label == "chrome_wheel_pose" and confidence >= 0.50:
            return True
        if self._chrome_mode_enabled and self._chrome_active_for_wheel(now):
            if stable_label == "wheel_pose":
                return True
            if raw_label == "wheel_pose" and confidence >= 0.48:
                return True
        return False

    def _update_chrome_wheel(self, prediction, hand_reading, now: float, *, active: bool) -> bool:
        if not active or prediction is None or hand_reading is None:
            if self._chrome_wheel_visible and now >= self._chrome_wheel_pose_grace_until:
                self._chrome_control_text = "chrome wheel closed"
                self._reset_chrome_wheel()
            else:
                self._chrome_wheel_candidate = "neutral"
                self._chrome_wheel_candidate_since = now
            return self._chrome_wheel_visible

        wheel_label = self._chrome_wheel_pose_active(prediction, now)
        if self._chrome_wheel_visible:
            if wheel_label:
                self._chrome_wheel_pose_grace_until = now + 0.25
                self._update_chrome_wheel_selection(hand_reading, now)
            elif now >= self._chrome_wheel_pose_grace_until:
                self._chrome_control_text = "chrome wheel closed"
                self._reset_chrome_wheel()
            return True

        if now < self._chrome_wheel_cooldown_until:
            if not wheel_label:
                self._chrome_wheel_candidate = "neutral"
            return False

        if not wheel_label:
            self._chrome_wheel_candidate = "neutral"
            self._chrome_wheel_candidate_since = now
            return False

        if not self._chrome_active_for_wheel(now):
            self._chrome_control_text = "chrome must be active for wheel"
            self._chrome_wheel_candidate = "neutral"
            self._chrome_wheel_candidate_since = now
            return True

        if self._chrome_wheel_candidate != "chrome_wheel_pose":
            self._chrome_wheel_candidate = "chrome_wheel_pose"
            self._chrome_wheel_candidate_since = now
            self._chrome_control_text = "hold chrome wheel pose"
            return True

        if now - self._chrome_wheel_candidate_since < 1.0:
            return True

        self._chrome_wheel_visible = True
        self._chrome_wheel_anchor = hand_reading.palm.center.copy()
        self._chrome_wheel_cursor_offset = (0.0, 0.0)
        self._chrome_wheel_selected_key = None
        self._chrome_wheel_selected_since = now
        self._chrome_wheel_pose_grace_until = now + 0.25
        self._chrome_control_text = "chrome wheel active"
        return True

    def _update_chrome_wheel_selection(self, hand_reading, now: float) -> None:
        if self._chrome_wheel_anchor is None:
            self._chrome_wheel_anchor = hand_reading.palm.center.copy()
        offset = (hand_reading.palm.center - self._chrome_wheel_anchor) / max(hand_reading.palm.scale, 1e-6)
        self._chrome_wheel_cursor_offset = (float(offset[0]), float(offset[1]))
        selection_key = self._wheel_selection_key(float(offset[0]), float(offset[1]), self._chrome_wheel_items())
        if selection_key is None:
            self._chrome_wheel_selected_key = None
            self._chrome_wheel_selected_since = now
            self._chrome_control_text = "chrome wheel active"
            return

        if selection_key != self._chrome_wheel_selected_key:
            self._chrome_wheel_selected_key = selection_key
            self._chrome_wheel_selected_since = now
            self._chrome_control_text = f"chrome wheel: {self._chrome_wheel_label(selection_key)}"
            return

        if now - self._chrome_wheel_selected_since < 1.0:
            return

        self._execute_chrome_wheel_action(selection_key)
        self._chrome_wheel_cooldown_until = now + 1.5
        self._reset_chrome_wheel()

    def _wheel_selection_key(self, dx: float, dy: float, items: tuple[tuple[str, str, float], ...]) -> str | None:
        radius = math.hypot(dx, dy)
        # No upper bound: once the cursor is outside the central
        # deadzone the angle alone picks the slice, even if the
        # user's hand has drifted well past the visible wheel ring.
        min_select_radius = 0.30
        if radius < min_select_radius:
            return None

        angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0
        slice_span = 360.0 / max(1, len(items))
        half_slice = slice_span / 2.0
        angular_margin = 1.5
        for key, _label, target_angle in items:
            delta = abs((angle - target_angle + 180.0) % 360.0 - 180.0)
            if delta <= max(0.0, half_slice - angular_margin):
                return key
        return None

    def _chrome_wheel_label(self, key: str) -> str:
        for item_key, label, _angle in self._chrome_wheel_items():
            if item_key == key:
                return label
        return key.replace("_", " ")

    def _execute_chrome_wheel_action(self, key: str) -> None:
        if key == "bookmark":
            success = self.chrome_controller.bookmark_current_tab()
        elif key == "history":
            success = self.chrome_controller.open_history()
        elif key == "downloads":
            success = self.chrome_controller.open_downloads()
        elif key == "bookmarks":
            success = self.chrome_controller.open_bookmarks_manager()
        elif key == "print":
            success = self.chrome_controller.print_page()
        else:
            success = self.chrome_controller.reopen_closed_tab()
        self._chrome_control_text = self.chrome_controller.message
        if success:
            self._chrome_mode_enabled = True

    def _spotify_active_for_wheel(self, now: float) -> bool:
        if now < self._spotify_active_cache_until:
            return self._spotify_active_cache

        if hasattr(self.spotify_controller, "is_active_for_wheel"):
            active = bool(self.spotify_controller.is_active_for_wheel())
        else:
            active = bool(self.spotify_controller.is_active_device_available())

        self._spotify_active_cache = active
        self._spotify_active_cache_until = now + 0.9
        return active

    def _update_spotify_wheel(self, prediction, hand_reading, now: float, *, active: bool) -> bool:
        if not active or prediction is None or hand_reading is None:
            if self._spotify_wheel_visible and now >= self._spotify_wheel_pose_grace_until:
                self._spotify_control_text = "spotify wheel closed"
                self._reset_spotify_wheel()
            else:
                self._spotify_wheel_candidate = "neutral"
                self._spotify_wheel_candidate_since = now
            return self._spotify_wheel_visible

        wheel_label = prediction.stable_label == "wheel_pose" or (
            prediction.raw_label == "wheel_pose" and prediction.confidence >= 0.56
        )
        if self._spotify_wheel_visible:
            if wheel_label:
                self._spotify_wheel_pose_grace_until = now + 0.25
                self._update_spotify_wheel_selection(hand_reading, now)
            elif now >= self._spotify_wheel_pose_grace_until:
                self._spotify_control_text = "spotify wheel closed"
                self._reset_spotify_wheel()
            return True

        if now < self._spotify_wheel_cooldown_until:
            if not wheel_label:
                self._spotify_wheel_candidate = "neutral"
            return False

        if self._chrome_mode_enabled and self._chrome_active_for_wheel(now):
            self._spotify_wheel_candidate = "neutral"
            self._spotify_wheel_candidate_since = now
            return False

        if not wheel_label:
            self._spotify_wheel_candidate = "neutral"
            self._spotify_wheel_candidate_since = now
            return False

        if not self._spotify_active_for_wheel(now):
            self._spotify_control_text = self.spotify_controller.message or "spotify inactive on device"
            self._spotify_wheel_candidate = "neutral"
            self._spotify_wheel_candidate_since = now
            return True

        if self._spotify_wheel_candidate != "wheel_pose":
            self._spotify_wheel_candidate = "wheel_pose"
            self._spotify_wheel_candidate_since = now
            self._spotify_control_text = "hold wheel pose for spotify wheel"
            return True

        if now - self._spotify_wheel_candidate_since < 1.0:
            return True

        self._spotify_wheel_visible = True
        self._spotify_wheel_anchor = hand_reading.palm.center.copy()
        self._spotify_wheel_cursor_offset = (0.0, 0.0)
        self._spotify_wheel_selected_key = None
        self._spotify_wheel_selected_since = now
        self._spotify_wheel_pose_grace_until = now + 0.25
        self._spotify_control_text = "spotify wheel active"
        return True

    def _update_spotify_wheel_selection(self, hand_reading, now: float) -> None:
        if self._spotify_wheel_anchor is None:
            self._spotify_wheel_anchor = hand_reading.palm.center.copy()
        offset = (hand_reading.palm.center - self._spotify_wheel_anchor) / max(hand_reading.palm.scale, 1e-6)
        self._spotify_wheel_cursor_offset = (float(offset[0]), float(offset[1]))
        selection_key = self._spotify_wheel_selection_key(float(offset[0]), float(offset[1]))
        if selection_key is None:
            self._spotify_wheel_selected_key = None
            self._spotify_wheel_selected_since = now
            self._spotify_control_text = "spotify wheel active"
            return

        if selection_key != self._spotify_wheel_selected_key:
            self._spotify_wheel_selected_key = selection_key
            self._spotify_wheel_selected_since = now
            self._spotify_control_text = f"spotify wheel: {self._spotify_wheel_label(selection_key)}"
            return

        if now - self._spotify_wheel_selected_since < 1.0:
            return

        self._execute_spotify_wheel_action(selection_key)
        self._spotify_wheel_cooldown_until = now + 1.5
        self._reset_spotify_wheel()

    def _spotify_wheel_selection_key(self, dx: float, dy: float) -> str | None:
        radius = math.hypot(dx, dy)
        # No upper bound: once the cursor is past the central
        # deadzone the angle alone picks the slice, even if the
        # user's hand has drifted well past the visible wheel ring.
        min_select_radius = 0.46
        if radius < min_select_radius:
            return None

        angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0
        slice_span = 360.0 / max(1, len(self._spotify_wheel_items()))
        half_slice = slice_span / 2.0
        angular_margin = 1.5

        for key, _label, target_angle in self._spotify_wheel_items():
            delta = abs((angle - target_angle + 180.0) % 360.0 - 180.0)
            if delta <= max(0.0, half_slice - angular_margin):
                return key
        return None

    def _spotify_wheel_label(self, key: str) -> str:
        for item_key, label, _angle in self._spotify_wheel_items():
            if item_key == key:
                return label
        return key.replace("_", " ")

    def _execute_spotify_wheel_action(self, key: str) -> None:
        if key == "add_playlist":
            self._spotify_control_text = "say what playlist you would like to add to"
            self._start_playlist_prompt("add_playlist")
            return
        if key == "remove_playlist":
            self._spotify_control_text = "say what playlist you would like to remove from"
            self._start_playlist_prompt("remove_playlist")
            return

        if key == "add_queue":
            success = self.spotify_controller.add_current_track_to_queue()
        elif key == "remove_queue":
            success = self.spotify_controller.remove_current_track_from_queue()
        elif key == "remove_liked":
            success = self.spotify_controller.remove_current_track_from_liked()
        elif key == "shuffle":
            success = self.spotify_controller.toggle_shuffle()
        else:
            success = self.spotify_controller.save_current_track()
        self._spotify_control_text = self.spotify_controller.message
        if success:
            details = self.spotify_controller.get_current_track_details()
            if details is not None:
                self._spotify_info_text = details.summary()

    def _start_playlist_prompt(self, action: str) -> None:
        if self._voice_listening:
            return
        self._voice_cooldown_until = time.monotonic() + 1.5
        self._start_voice_capture(mode=action, preferred_app=None)

    def _clean_playlist_reply(self, spoken_text: str) -> str:
        cleaned = str(spoken_text or "").lower()
        cleaned = re.sub(
            r"\b(add|remove|it|this|current|song|track|playlist|to|from|my|the|please|thanks|thank you|would like|like)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
        return cleaned

    def _hex_to_bgr(self, color: str, *, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
        value = str(color or "").strip().lstrip("#")
        if len(value) != 6:
            return fallback
        try:
            red = int(value[0:2], 16)
            green = int(value[2:4], 16)
            blue = int(value[4:6], 16)
        except ValueError:
            return fallback
        return blue, green, red

    def _wheel_label_lines(self, label: str, *, max_chars: int = 12) -> list[str]:
        words = str(label or "").split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines[:3]

    def _wheel_ring_segment_points(
        self,
        center: tuple[int, int],
        inner_radius: int,
        outer_radius: int,
        start_deg: float,
        end_deg: float,
        *,
        step_deg: float = 4.0,
    ) -> np.ndarray:
        cx, cy = center
        outer_points: list[tuple[int, int]] = []
        angle = start_deg
        while angle <= end_deg + 1e-6:
            radians = math.radians(angle)
            outer_points.append((
                int(round(cx + math.cos(radians) * outer_radius)),
                int(round(cy - math.sin(radians) * outer_radius)),
            ))
            angle += step_deg
        inner_points: list[tuple[int, int]] = []
        angle = end_deg
        while angle >= start_deg - 1e-6:
            radians = math.radians(angle)
            inner_points.append((
                int(round(cx + math.cos(radians) * inner_radius)),
                int(round(cy - math.sin(radians) * inner_radius)),
            ))
            angle -= step_deg
        points = np.array(outer_points + inner_points, dtype=np.int32)
        return points.reshape((-1, 1, 2))

    def _draw_wheel_label(
        self,
        frame,
        *,
        center: tuple[int, int],
        angle_deg: float,
        radius: float,
        lines: list[str],
        color: tuple[int, int, int],
        selected: bool,
    ) -> None:
        cx, cy = center
        radians = math.radians(angle_deg)
        anchor_x = int(round(cx + math.cos(radians) * radius))
        anchor_y = int(round(cy - math.sin(radians) * radius))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.54 if selected else 0.48
        thickness = 2 if selected else 1
        line_height = 18 if selected else 16
        block_top = anchor_y - (len(lines) - 1) * line_height // 2
        for index, line in enumerate(lines):
            size, _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_x = anchor_x - size[0] // 2
            text_y = block_top + index * line_height
            cv2.putText(frame, line, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    def _draw_mouse_control_overlay(self, frame) -> None:
        draw_mouse_control_box_overlay(
            frame,
            debug_state=self.mouse_tracker.debug_state,
            mode_enabled=self._mouse_mode_enabled,
            active_monitor_index=getattr(self.config, "mouse_active_monitor_index", None),
        )

    def _draw_mouse_monitor_overlay(self, frame) -> None:
        draw_mouse_monitor_overlay(
            frame,
            mouse_controller=self.mouse_controller,
            debug_state=self.mouse_tracker.debug_state,
            mode_enabled=self._mouse_mode_enabled,
        )

    def _draw_spotify_wheel(self, frame, hand_reading) -> None:
        if self._spotify_wheel_anchor is None:
            return

        items = self._spotify_wheel_items()
        if not items:
            return

        height, width = frame.shape[:2]
        center_x = int(self._spotify_wheel_anchor[0] * width)
        center_y = int(self._spotify_wheel_anchor[1] * height)
        center = (center_x, center_y)

        outer_radius = max(102, min(width, height) // 7)
        inner_radius = int(outer_radius * 0.44)
        center_radius = int(inner_radius * 0.84)

        accent = self._hex_to_bgr(self.config.accent_color, fallback=(182, 233, 29))
        surface = self._hex_to_bgr(self.config.surface_color, fallback=(22, 28, 44))
        outer_outline = tuple(min(255, c + 26) for c in accent)
        text_color = (240, 246, 255)
        muted_text = (222, 232, 244)
        segment_fill = (
            min(255, int(surface[0] * 0.92 + 42)),
            min(255, int(surface[1] * 0.82 + 24)),
            min(255, int(surface[2] * 0.76 + 10)),
        )
        selected_fill = (
            min(255, int(accent[0] * 0.70 + 55)),
            min(255, int(accent[1] * 0.84 + 34)),
            min(255, int(accent[2] * 0.96 + 18)),
        )
        center_fill = (
            max(0, int(surface[0] * 0.72)),
            max(0, int(surface[1] * 0.72)),
            max(0, int(surface[2] * 0.80)),
        )

        overlay = frame.copy()
        slice_span = 360.0 / max(1, len(items))
        half_slice = slice_span / 2.0

        # ring segments
        for key, label, angle_deg in items:
            start_deg = angle_deg - half_slice
            end_deg = angle_deg + half_slice
            points = self._wheel_ring_segment_points(center, inner_radius, outer_radius, start_deg, end_deg)
            fill = selected_fill if key == self._spotify_wheel_selected_key else segment_fill
            cv2.fillPoly(overlay, [points], fill)

        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0.0, dst=frame)

        # dividers and ring outlines
        for index in range(len(items)):
            boundary = index * slice_span
            radians = math.radians(boundary)
            inner_point = (
                int(round(center_x + math.cos(radians) * inner_radius)),
                int(round(center_y - math.sin(radians) * inner_radius)),
            )
            outer_point = (
                int(round(center_x + math.cos(radians) * outer_radius)),
                int(round(center_y - math.sin(radians) * outer_radius)),
            )
            cv2.line(frame, inner_point, outer_point, outer_outline, 2, cv2.LINE_AA)

        cv2.circle(frame, center, outer_radius, outer_outline, 3, cv2.LINE_AA)
        cv2.circle(frame, center, inner_radius, outer_outline, 2, cv2.LINE_AA)
        cv2.circle(frame, center, center_radius, center_fill, thickness=-1)
        cv2.circle(frame, center, center_radius, outer_outline, 2, cv2.LINE_AA)

        # highlight selected segment more strongly
        if self._spotify_wheel_selected_key is not None:
            selected = next((item for item in items if item[0] == self._spotify_wheel_selected_key), None)
            if selected is not None:
                _key, _label, angle_deg = selected
                selected_points = self._wheel_ring_segment_points(
                    center,
                    inner_radius,
                    outer_radius,
                    angle_deg - half_slice,
                    angle_deg + half_slice,
                )
                highlight = frame.copy()
                cv2.fillPoly(highlight, [selected_points], (255, 255, 255))
                cv2.addWeighted(highlight, 0.12, frame, 0.88, 0.0, dst=frame)
                cv2.polylines(frame, [selected_points], True, (255, 255, 255), 2, cv2.LINE_AA)

        # center title
        center_lines = ["Gesture", "Wheel"]
        center_font = cv2.FONT_HERSHEY_SIMPLEX
        block_top = center_y - 8
        for i, line in enumerate(center_lines):
            size, _ = cv2.getTextSize(line, center_font, 0.60, 2)
            cv2.putText(
                frame,
                line,
                (center_x - size[0] // 2, block_top + i * 22),
                center_font,
                0.60,
                text_color,
                2,
                cv2.LINE_AA,
            )

        # selection progress arc
        if self._spotify_wheel_selected_key is not None:
            progress = max(0.0, min(1.0, (time.monotonic() - self._spotify_wheel_selected_since) / 0.7))
            if progress > 0.0:
                cv2.ellipse(
                    frame,
                    center,
                    (center_radius + 8, center_radius + 8),
                    0.0,
                    -90.0,
                    -90.0 + 360.0 * progress,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

        # label text inside each wedge
        text_radius = inner_radius + (outer_radius - inner_radius) * 0.52
        for key, label, angle_deg in items:
            selected = key == self._spotify_wheel_selected_key
            lines = self._wheel_label_lines(label, max_chars=12 if selected else 11)
            self._draw_wheel_label(
                frame,
                center=center,
                angle_deg=angle_deg,
                radius=text_radius,
                lines=lines,
                color=text_color if selected else muted_text,
                selected=selected,
            )

        # pointer based on current palm offset
        if hand_reading is not None:
            offset = (hand_reading.palm.center - self._spotify_wheel_anchor) / max(hand_reading.palm.scale, 1e-6)
            dx = float(offset[0])
            dy = float(offset[1])
            magnitude = math.hypot(dx, dy)
            if magnitude > 0.04:
                max_pointer = outer_radius - 8
                scale = min(max_pointer, magnitude * outer_radius * 1.18) / max(magnitude, 1e-6)
                pointer_x = int(round(center_x + dx * scale))
                pointer_y = int(round(center_y + dy * scale))
                cv2.line(frame, center, (pointer_x, pointer_y), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (pointer_x, pointer_y), 7, (255, 255, 255), thickness=-1)
                cv2.circle(frame, (pointer_x, pointer_y), 9, outer_outline, thickness=2)



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

    def _handle_left_hand_voice(self, prediction, now: float) -> None:
        stable_label = prediction.stable_label
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
        chrome_mode_voice = self._chrome_mode_enabled and self.chrome_controller.is_window_open()
        self._start_voice_capture(mode="general", preferred_app="chrome" if chrome_mode_voice else None)

    def _start_dictation_capture(self) -> None:
        if self._voice_listening:
            return
        if not self.text_input_controller.available:
            self._voice_control_text = self.text_input_controller.message
            return
        self.dictation_processor.reset()
        self._start_voice_capture(mode="dictation", preferred_app=None)

    def _stop_dictation_capture(self) -> None:
        if not self._dictation_active:
            return
        self._dictation_active = False
        self._voice_control_text = "ending dictation..."
        self.voice_status_overlay.show_processing(
            "Ending dictation...",
            command_text="",
        )
        if self._voice_stop_event is not None:
            self._voice_stop_event.set()

    def _start_voice_capture(self, *, mode: str, preferred_app: str | None) -> None:
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
        self.voice_status_overlay.show_listening()

        chrome_mode_voice = self._chrome_mode_enabled and self.chrome_controller.is_window_open()
        preferred_target = "chrome" if chrome_mode_voice and mode == "general" else preferred_app

        def _push_status(status: str, *, command_text: str = "") -> None:
            self._voice_queue.put(
                (
                    request_id,
                    {
                        "event": "status",
                        "status": status,
                        "command_text": command_text,
                    },
                )
            )

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
                                        "mode": mode,
                                        "success": True,
                                        "target": "dictation",
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
                                        "mode": mode,
                                        "success": True,
                                        "target": "dictation",
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
                                        "mode": mode,
                                        "success": inserted,
                                        "target": "dictation",
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
                            "mode": mode,
                            "success": bool(self.dictation_processor.full_text),
                            "target": "dictation",
                            "heard_text": "",
                            "control_text": final_message,
                            "display_text": final_display,
                        },
                    )
                )
                return

            result = self.voice_listener.listen(max_seconds=12.5, status_callback=_push_status)
            if mode == "general" and result.success:
                _push_status("processing", command_text=result.heard_text)
                execution = self.voice_processor.execute(
                    result.heard_text,
                    context=VoiceCommandContext(preferred_app=preferred_target),
                )
                payload = {
                    "event": "result",
                    "mode": mode,
                    "success": execution.success,
                    "target": execution.target,
                    "heard_text": execution.heard_text,
                    "control_text": execution.control_text,
                    "info_text": execution.info_text,
                    "display_text": execution.display_text or execution.heard_text,
                }
            else:
                payload = {
                    "event": "result",
                    "mode": mode,
                    "success": result.success,
                    "target": "voice",
                    "heard_text": result.heard_text,
                    "control_text": result.message,
                    "info_text": self._spotify_info_text,
                    "display_text": result.heard_text,
                }
            self._voice_queue.put((request_id, payload))

        self._voice_thread = threading.Thread(target=_worker, name="hgr-voice-command", daemon=True)
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
                command_text = str(payload.get("command_text", "") or "").strip()
                if command_text:
                    self._voice_heard_text = command_text
                    self._voice_display_text = command_text
                if status == "listening":
                    self._voice_control_text = "dictation listening..." if self._voice_mode == "dictation" else "voice listening..."
                    if self._voice_mode == "dictation":
                        self.voice_status_overlay.show_processing("Dictation active", command_text="")
                    else:
                        if getattr(self.voice_status_overlay, "_mode", "") == "selection":
                            self.voice_status_overlay.update_selection_status("Listening...", listening=True)
                        else:
                            self.voice_status_overlay.show_listening()
                elif status == "recognizing":
                    if self._voice_mode == "dictation":
                        self._voice_control_text = "dictation active"
                        self.voice_status_overlay.show_processing("Dictation active", command_text="")
                    else:
                        self._voice_control_text = "recognizing..."
                        if getattr(self.voice_status_overlay, "_mode", "") == "selection":
                            self.voice_status_overlay.update_selection_status("Recognizing...", listening=False)
                        else:
                            self.voice_status_overlay.show_processing(
                                "Recognizing...",
                                command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                            )
                elif status == "processing":
                    self._voice_control_text = "processing command..."
                    if getattr(self.voice_status_overlay, "_mode", "") == "selection":
                        self.voice_status_overlay.update_selection_status("Processing command...", listening=False)
                    else:
                        self.voice_status_overlay.show_processing(
                            "Processing command...",
                            command_text=command_text or (self._voice_display_text if self._voice_display_text != "-" else ""),
                        )
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
                    self.voice_status_overlay.show_processing("Dictation active", command_text="")
                else:
                    self.voice_status_overlay.show_processing("Dictation active", command_text="")
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
                self.voice_status_overlay.show_result(
                    "Dictation stopped",
                    command_text="",
                    duration=1.8,
                )
                continue

            self._voice_listening = False
            self._dictation_active = False
            self._dictation_backend = "idle"
            self._voice_mode = "ready"
            self._voice_stop_event = None
            mode = str(payload.get("mode", "general"))
            target = str(payload.get("target", "") or "")
            heard_text = str(payload.get("heard_text", "") or "").strip()
            self._voice_heard_text = heard_text or "-"
            self._voice_display_text = str(payload.get("display_text", "") or heard_text or "-")
            self._voice_control_text = str(payload.get("control_text", "voice idle"))
            if mode == "selection":
                self._selection_prompt_active = False
                self._selection_auto_listen_at = 0.0
                self.command_detected.emit(self._voice_control_text)
                if payload.get("success") and target != "voice_selection":
                    self.voice_status_overlay.show_result(
                        "Executing command",
                        command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                        duration=1.8,
                    )
                else:
                    self.voice_status_overlay.hide_overlay()
                    self.voice_status_overlay.show_result(
                        "Selection failed",
                        command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                        duration=1.6,
                    )
                continue

            if mode == "general":
                if payload.get("target") == "spotify":
                    if payload.get("success"):
                        self._spotify_info_text = str(payload.get("info_text", self._spotify_info_text))
                    self._spotify_control_text = str(payload.get("control_text", self._spotify_control_text))
                elif payload.get("target") == "chrome":
                    self._chrome_control_text = str(payload.get("control_text", self._chrome_control_text))
                if payload.get("target") == "voice_selection":
                    title, items, instruction = self._parse_voice_selection_text(self._voice_display_text if self._voice_display_text != "-" else "")
                    self.voice_status_overlay.show_selection(title, items, instruction, status_text="Choose a file or app")
                    continue
                status_text = "Executing command" if payload.get("success") else "Command not understood"
                self.voice_status_overlay.show_result(
                    status_text,
                    command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                    duration=2.0,
                )
                continue

            if not payload.get("success"):
                self._spotify_control_text = str(payload.get("control_text", "voice command not understood"))
                self.voice_status_overlay.show_result(
                    "Playlist not understood",
                    command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                    duration=2.0,
                )
                continue

            playlist_name = self._clean_playlist_reply(heard_text)
            if not playlist_name:
                self._spotify_control_text = "playlist name not understood"
                self.voice_status_overlay.show_result(
                    "Playlist not understood",
                    command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                    duration=2.0,
                )
                continue
            if mode == "add_playlist":
                success = self.spotify_controller.add_current_track_to_playlist(playlist_name)
            else:
                success = self.spotify_controller.remove_current_track_from_playlist(playlist_name)
            self._spotify_control_text = self.spotify_controller.message
            if success:
                details = self.spotify_controller.get_current_track_details()
                if details is not None:
                    self._spotify_info_text = details.summary()
            self.voice_status_overlay.show_result(
                self.spotify_controller.message,
                command_text=playlist_name,
                duration=2.0,
            )

    def _parse_voice_selection_text(self, text: str) -> tuple[str, list[tuple[str, str, str]], str]:
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        title = "Which file/folder?"
        if lines and "app" in lines[0].lower():
            title = "Which app?"
        instruction = "Say the corresponding letter"
        items: list[tuple[str, str, str]] = []
        for line in lines:
            match = re.match(r"^([A-Za-z]|\d+)\.\s+(.*?)(?:\s+[—-]\s+(.*))?$", line)
            if match:
                selection_key = str(match.group(1) or "").strip().upper()
                label = str(match.group(2) or "").strip()
                path_text = str(match.group(3) or "").strip()
                items.append((selection_key, label, path_text))
                continue
            if line.lower().startswith("say "):
                instruction = line
            elif line.lower().startswith("which "):
                instruction = line
        return title, items, instruction

    def _handle_voice_overlay_selection(self, selection_key: str) -> None:
        self._selection_prompt_active = False
        self._selection_auto_listen_at = 0.0
        try:
            self.voice_status_overlay.update_selection_status("Processing command...", listening=False)
        except Exception:
            pass
        try:
            result = self.voice_processor.execute(str(selection_key or "").strip())
        except Exception as exc:
            self._voice_queue.put((self._voice_request_id, {
                "event": "result",
                "mode": "selection",
                "success": False,
                "target": "voice",
                "heard_text": str(number),
                "control_text": f"command failed: {type(exc).__name__}",
                "info_text": "-",
                "display_text": str(number),
            }))
            return
        self._voice_queue.put((self._voice_request_id, {
            "event": "result",
            "mode": "selection",
            "success": result.success,
            "target": result.target,
            "heard_text": result.heard_text,
            "control_text": result.control_text,
            "info_text": result.info_text,
            "display_text": getattr(result, "display_text", result.heard_text),
        }))

    def _update_spotify_wheel_overlay(self, now: float) -> None:
        if not self._spotify_wheel_visible:
            if self.spotify_wheel_overlay.isVisible():
                self.spotify_wheel_overlay.hide_overlay()
            return

        selected_label = "Hold a slice"
        if self._spotify_wheel_selected_key is not None:
            selected_label = self._spotify_wheel_label(self._spotify_wheel_selected_key)

        selection_progress = 0.0
        if self._spotify_wheel_selected_key is not None:
            selection_progress = min(1.0, max(0.0, (now - self._spotify_wheel_selected_since) / 1.0))

        self.spotify_wheel_overlay.set_wheel(
            items=self._spotify_wheel_items(),
            selected_key=self._spotify_wheel_selected_key,
            selection_progress=selection_progress,
            status_text=selected_label,
            cursor_offset=self._spotify_wheel_cursor_offset,
        )
        self.spotify_wheel_overlay.show_overlay()

    def _update_chrome_wheel_overlay(self, now: float) -> None:
        if not self._chrome_wheel_visible:
            if self.chrome_wheel_overlay.isVisible():
                self.chrome_wheel_overlay.hide_overlay()
            return

        selected_label = "Hold a slice"
        if self._chrome_wheel_selected_key is not None:
            selected_label = self._chrome_wheel_label(self._chrome_wheel_selected_key)

        selection_progress = 0.0
        if self._chrome_wheel_selected_key is not None:
            selection_progress = min(1.0, max(0.0, (now - self._chrome_wheel_selected_since) / 1.0))

        self.chrome_wheel_overlay.set_wheel(
            items=self._chrome_wheel_items(),
            selected_key=self._chrome_wheel_selected_key,
            selection_progress=selection_progress,
            status_text=selected_label,
            cursor_offset=self._chrome_wheel_cursor_offset,
        )
        self.chrome_wheel_overlay.show_overlay()


    def _reset_voice_state(self) -> None:
        if self._voice_stop_event is not None:
            self._voice_stop_event.set()
            self._voice_stop_event = None
        self.dictation_processor.reset()
        self._voice_candidate = "neutral"
        self._voice_candidate_since = 0.0
        self._voice_cooldown_until = 0.0
        self._voice_latched_label = None
        self._dictation_active = False
        self._dictation_backend = "idle"
        self._voice_mode = "ready"
        self._voice_request_id += 1
        self._voice_listening = False
        self._voice_control_text = self.voice_listener.message
        self._voice_heard_text = "-"
        self._voice_display_text = "-"
        self.voice_status_overlay.hide_overlay()
        while True:
            try:
                self._voice_queue.get_nowait()
            except queue.Empty:
                break

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_session()
        super().closeEvent(event)

# Author: Konstantin Markov


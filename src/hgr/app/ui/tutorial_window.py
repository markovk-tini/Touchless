from __future__ import annotations

import math
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QFont, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


from ...config.app_config import AppConfig
from ...debug.chrome_controller import ChromeController
from ...debug.mouse_gesture import MouseDebugState, MouseGestureTracker
from ...debug.mouse_overlay import (
    draw_mouse_control_box_overlay,
    draw_mouse_monitor_overlay,
)
from ...debug.voice_command_listener import VoiceCommandListener
from ...gesture.recognition.engine import GestureRecognitionEngine
from ...gesture.rendering.overlay import HAND_CONNECTIONS
from ...voice.command_processor import VoiceCommandProcessor
from ...debug.spotify_controller import SpotifyController
from ...gesture.ui.test_window import SpotifyWheelOverlay
# NOTE: camera helpers (open_camera_by_index / open_preferred_or_first_available)
# used to be imported here, but the tutorial window now reads frames from
# the parent app's shared GestureWorker — not from a fresh capture handle.
# The unused imports were cluttering the import list. See line ~1597 for
# the historical comment about why this window doesn't open its own
# camera anymore.
from ...voice.command_processor import VoiceCommandContext
from ..integration.noop_engine import GestureWorker


@dataclass(frozen=True)
class _StepDefinition:
    key: str
    title: str
    description: str
    progress_template: str


class SwipeInstructionWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._left_count = 0
        self._right_count = 0
        self._accent = QColor("#1DE9B6")
        self._text = QColor("#F4FAFF")
        self.setMinimumHeight(200)

    def apply_theme(self, accent: str, text: str) -> None:
        self._accent = QColor(accent)
        self._text = QColor(text)
        self.update()

    def set_counts(self, left_count: int, right_count: int) -> None:
        self._left_count = int(left_count)
        self._right_count = int(right_count)
        self.update()

    def _draw_hand(self, painter: QPainter, center: QPointF, scale: float, color: QColor) -> None:
        palm = [
            QPointF(center.x() - 18 * scale, center.y() + 28 * scale),
            QPointF(center.x() - 28 * scale, center.y() + 4 * scale),
            QPointF(center.x() - 18 * scale, center.y() - 30 * scale),
            QPointF(center.x() + 18 * scale, center.y() - 30 * scale),
            QPointF(center.x() + 28 * scale, center.y() + 4 * scale),
            QPointF(center.x() + 18 * scale, center.y() + 28 * scale),
        ]
        finger_bases = (-18, -6, 6, 18)
        tip_offsets = (-36, -48, -48, -36)

        pen = QPen(color, 2.6)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        path = QPainterPath()
        path.moveTo(palm[0])
        for point in palm[1:]:
            path.lineTo(point)
        path.closeSubpath()
        painter.drawPath(path)

        wrist = QPointF(center.x(), center.y() + 28 * scale)
        for base_x, tip_y in zip(finger_bases, tip_offsets):
            base = QPointF(center.x() + base_x * scale, center.y() - 6 * scale)
            tip = QPointF(center.x() + base_x * scale, center.y() + tip_y * scale)
            painter.drawLine(wrist, base)
            painter.drawLine(base, tip)
            painter.drawEllipse(tip, 2.2 * scale, 2.2 * scale)

        thumb_mid = QPointF(center.x() - 28 * scale, center.y() + 6 * scale)
        thumb_tip = QPointF(center.x() - 48 * scale, center.y() + 18 * scale)
        painter.drawLine(wrist, thumb_mid)
        painter.drawLine(thumb_mid, thumb_tip)
        painter.drawEllipse(thumb_tip, 2.2 * scale, 2.2 * scale)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), Qt.transparent)

        card = self.rect().adjusted(6, 6, -6, -6)
        painter.setBrush(QColor(255, 255, 255, 10))
        painter.setPen(QPen(QColor(self._accent.red(), self._accent.green(), self._accent.blue(), 50), 1.2))
        painter.drawRoundedRect(card, 18, 18)
        title_font = QFont("Segoe UI", 13)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(self._accent)
        painter.drawText(
            QRectF(card.left() + 18, card.top() + 20, card.width() - 36, 28),
            Qt.AlignCenter,
            "Swipe Practice",
        )

        # The Left:N/3 Right:N/3 line that used to live here was a
        # duplicate — the same counts are now shown big-and-bold under
        # the camera view via tutorial_camera_footer. Keep the hint
        # so the practice card still has guidance text.
        hint_font = QFont("Segoe UI", 10)
        painter.setFont(hint_font)
        painter.setPen(QColor(self._text.red(), self._text.green(), self._text.blue(), 185))
        painter.drawText(
            QRectF(card.left() + 24, card.top() + 64, card.width() - 48, 60),
            Qt.AlignCenter | Qt.TextWordWrap,
            "Use your right hand and match the swipe direction shown over the live camera view.",
        )


class WheelInstructionWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._accent = QColor("#1DE9B6")
        self._text = QColor("#F4FAFF")
        self.setMinimumHeight(220)

    def apply_theme(self, accent: str, text: str) -> None:
        self._accent = QColor(accent)
        self._text = QColor(text)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), Qt.transparent)

        bounds = QRectF(self.rect()).adjusted(16, 12, -16, -12)
        center = bounds.center()
        radius = min(bounds.width(), bounds.height()) * 0.34
        inner_radius = radius * 0.43

        for index in range(6):
            start_angle = 90 - index * 60
            painter.setPen(QPen(QColor(255, 255, 255, 110), 1.4))
            painter.setBrush(QColor(29, 233, 182, 22 if index % 2 == 0 else 12))
            path = QPainterPath()
            outer = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
            inner = QRectF(center.x() - inner_radius, center.y() - inner_radius, inner_radius * 2, inner_radius * 2)
            path.arcMoveTo(outer, start_angle)
            path.arcTo(outer, start_angle, -60)
            path.arcTo(inner, start_angle - 60, 60)
            path.closeSubpath()
            painter.drawPath(path)
            painter.fillPath(path, painter.brush())

        painter.setBrush(QColor(8, 20, 34, 230))
        painter.setPen(QPen(self._accent, 2.0))
        painter.drawEllipse(center, inner_radius, inner_radius)

        title_font = QFont("Segoe UI", 12)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(self._accent)
        painter.drawText(QRectF(center.x() - 90, center.y() - 26, 180, 26), Qt.AlignCenter, "Gesture Wheel")

        hint_font = QFont("Segoe UI", 9)
        painter.setFont(hint_font)
        painter.setPen(QColor(self._text.red(), self._text.green(), self._text.blue(), 210))
        painter.drawText(
            QRectF(center.x() - 100, center.y() + 4, 200, 34),
            Qt.AlignCenter | Qt.TextWordWrap,
            "Hold the pose, move toward a slice, and keep it there to confirm.",
        )


class MousePracticeWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._accent = QColor("#1DE9B6")
        self._text = QColor("#F4FAFF")
        self._targets = [
            (0.20, 0.28),
            (0.74, 0.22),
            (0.66, 0.64),
            (0.28, 0.72),
        ]
        self._active_index = 0
        self._cursor_position: tuple[float, float] | None = None
        self._mode_enabled = False
        self._status_text = "Hold left-hand three to turn mouse mode on."
        # Bumped from 240 → 360 per UX request: bigger click arena
        # makes the four numbered targets easier to see + clearer
        # space between them. The arena is also vertically centred
        # inside this height (see paintEvent) so the dots sit in the
        # middle of the right panel rather than crowding the top
        # edge.
        self.setMinimumHeight(360)

    def apply_theme(self, accent: str, text: str) -> None:
        self._accent = QColor(accent)
        self._text = QColor(text)
        self.update()

    def reset(self) -> None:
        self._active_index = 0
        self._cursor_position = None
        self._mode_enabled = False
        self._status_text = "Hold left-hand three to turn mouse mode on."
        self.update()

    @property
    def completed(self) -> bool:
        return self._active_index >= len(self._targets)

    @property
    def completed_targets(self) -> int:
        return int(self._active_index)

    def mark_all_targets_completed(self) -> None:
        self._active_index = len(self._targets)
        self._status_text = "Nice work!"
        self.update()

    def set_mode_enabled(self, enabled: bool) -> None:
        self._mode_enabled = bool(enabled)
        if self.completed:
            self._status_text = "Nice work!"
        elif self._mode_enabled:
            self._status_text = "Move the cursor and left-click each target in order."
        else:
            self._status_text = "Hold left-hand three to turn mouse mode on."
        self.update()

    def set_cursor_position(self, position: tuple[float, float] | None) -> None:
        self._cursor_position = position
        self.update()

    # Tutorial-only "aim assist": registration radius is wider than
    # the visible dot (drawn at 22 px in paintEvent). Pinch-driven
    # cursor is smoothed but still has a small tremor — beginners
    # would otherwise miss often, get frustrated, and conclude
    # "mouse mode doesn't work". Learning-mode forgiveness margin
    # only; live mouse mode hits real Windows controls where the OS
    # does its own click-target heuristics. Bumped 0.12 -> 0.16
    # (~33% larger hit zone) per Touchless to-do.md §4.1.6.
    _TUTORIAL_TARGET_HIT_RADIUS = 0.16

    def register_click(self, position: tuple[float, float] | None) -> bool:
        if not self._mode_enabled or position is None or self.completed:
            return False
        target = self._targets[self._active_index]
        if math.hypot(position[0] - target[0], position[1] - target[1]) <= self._TUTORIAL_TARGET_HIT_RADIUS:
            self._active_index += 1
            if self.completed:
                self._status_text = "Nice work!"
            else:
                self._status_text = "Great. Keep clicking the highlighted target."
            self.update()
            return True
        self._status_text = "Close, but click inside the glowing target."
        self.update()
        return False

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), Qt.transparent)

        card = self.rect().adjusted(8, 8, -8, -8)
        painter.setBrush(QColor(255, 255, 255, 10))
        painter.setPen(QPen(QColor(self._accent.red(), self._accent.green(), self._accent.blue(), 55), 1.2))
        painter.drawRoundedRect(card, 18, 18)

        # Centre the practice arena vertically inside the card. The
        # arena is sized to ~75 % of the card's available height
        # (height minus the 64 px status band reserved at the
        # bottom) so the remaining 25 % is split evenly as top +
        # bottom padding, putting the four target dots visually in
        # the middle of the right panel instead of pinned to the
        # top edge.
        status_band_h = 64
        available_h = max(160, card.height() - status_band_h)
        arena_h = max(180, int(available_h * 0.78))
        # Clamp to card width minus side padding so the arena stays
        # close to square / portrait on narrow panels.
        arena_w = card.width() - 24
        arena_top = card.top() + (available_h - arena_h) // 2
        arena = QRectF(
            card.left() + 12,
            arena_top,
            arena_w,
            arena_h,
        )
        painter.setBrush(QColor(255, 255, 255, 7))
        painter.setPen(QPen(QColor(255, 255, 255, 28), 1.0))
        painter.drawRoundedRect(arena, 16, 16)

        # Color tiers per user request: targets the user still has to
        # click are RED (active = bright, future = dimmer red), and
        # turn GREEN once clicked. The current target also gets a
        # thicker pen so it's clearly the next one to hit.
        red = QColor("#FF5252")
        for index, (tx, ty) in enumerate(self._targets):
            point = QPointF(arena.left() + tx * arena.width(), arena.top() + ty * arena.height())
            if index < self._active_index:
                # Already clicked — green (accent).
                fill = QColor(self._accent.red(), self._accent.green(), self._accent.blue(), 110)
                pen = QPen(self._accent, 2.4)
            elif index == self._active_index:
                # Current — bright red, thick outline.
                fill = QColor(red.red(), red.green(), red.blue(), 95)
                pen = QPen(red, 3.4)
            else:
                # Future — muted red.
                fill = QColor(red.red(), red.green(), red.blue(), 45)
                pen = QPen(QColor(red.red(), red.green(), red.blue(), 170), 1.8)
            painter.setBrush(fill)
            painter.setPen(pen)
            painter.drawEllipse(point, 22, 22)
            painter.setPen(QPen(QColor(self._text.red(), self._text.green(), self._text.blue(), 200), 1.0))
            painter.drawText(QRectF(point.x() - 18, point.y() - 12, 36, 24), Qt.AlignCenter, str(index + 1))

        if self._cursor_position is not None:
            cx = arena.left() + self._cursor_position[0] * arena.width()
            cy = arena.top() + self._cursor_position[1] * arena.height()
            painter.setPen(QPen(self._accent, 2.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), 9, 9)
            painter.drawLine(QPointF(cx - 15, cy), QPointF(cx + 15, cy))
            painter.drawLine(QPointF(cx, cy - 15), QPointF(cx, cy + 15))

        status_font = QFont("Segoe UI", 9)
        status_font.setBold(True)
        painter.setFont(status_font)
        painter.setPen(self._accent if self._mode_enabled else QColor(self._text.red(), self._text.green(), self._text.blue(), 210))
        painter.drawText(
            QRectF(card.left() + 14, card.bottom() - 38, card.width() - 28, 28),
            Qt.AlignCenter | Qt.TextWordWrap,
            self._status_text,
        )

        # When all targets are cleared AND mouse mode is back off,
        # paint a giant translucent ✓ centered over the arena so the
        # user gets clear "you're done" feedback without losing sight
        # of the targets they just clicked. Painted last so it sits
        # on top of everything else.
        if self.completed and not self._mode_enabled:
            check_color = QColor(
                self._accent.red(), self._accent.green(), self._accent.blue(), 200
            )
            check_size = int(min(arena.width(), arena.height()) * 0.95)
            check_font = QFont("Segoe UI", max(96, int(check_size * 0.78)))
            check_font.setBold(True)
            painter.setFont(check_font)
            painter.setPen(check_color)
            painter.drawText(arena, Qt.AlignCenter, "✓")
            label_font = QFont("Segoe UI", 16)
            label_font.setBold(True)
            painter.setFont(label_font)
            painter.drawText(
                QRectF(arena.left(), arena.bottom() - 30, arena.width(), 24),
                Qt.AlignCenter,
                "Completed!",
            )


class MouseScrollPracticeWidget(QScrollArea):
    """Scroll-phase practice arena. Active after the click practice is
    finished — the user has to scroll all the way to the TOP bar AND
    all the way to the BOTTOM bar to complete the step.

    Why a QScrollArea: Touchless's mouse mode emits real OS-level
    scroll wheel events when the user makes the scroll gesture, so
    a standard QScrollArea responds without us simulating anything.
    We listen to the vertical scrollbar's `valueChanged` signal,
    flip `_top_reached` / `_bottom_reached` when the scrollbar hits
    each extremity, and complete when both are seen.

    Top / bottom bars: small dyed banners glued to the start/end of
    the scrollable content. They're purely visual markers — the
    completion check is on the scrollbar's value, not collision
    with the bars (saves us a hit-test path).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._accent = QColor("#1DE9B6")
        self._text = QColor("#F4FAFF")
        self._top_reached = False
        self._bottom_reached = False
        self._status_text = "Use mouse mode to scroll up to the top bar."
        # Strip the QScrollArea's default chrome so it visually blends
        # with the tutorial card surface.
        self.setObjectName("mouseScrollPractice")
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setWidgetResizable(True)
        # Fill the entire right panel vertically while scroll phase
        # is active — Expanding size policy in both directions, plus
        # a generous minimum height so it doesn't shrink below a
        # usable scrollable region on smaller window sizes.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(360)
        # Custom-styled vertical scrollbar so the green track matches
        # the rest of the tutorial chrome (the stock Win11 scrollbar
        # clashes with the accent palette).
        self.setStyleSheet(
            "QScrollArea#mouseScrollPractice { background: transparent; }"
            "QScrollArea#mouseScrollPractice > QWidget > QWidget { background: transparent; }"
            "QScrollBar:vertical { background: rgba(255,255,255,0.06); "
            "  width: 10px; border-radius: 5px; margin: 2px 0; }"
            "QScrollBar::handle:vertical { background: #1DE9B6; "
            "  min-height: 28px; border-radius: 5px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )

        content = QWidget()
        content.setObjectName("mouseScrollContent")
        content.setStyleSheet(
            "#mouseScrollContent { background: transparent; }"
        )
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Top bar — the "you've reached the top" marker.
        self._top_bar = QLabel("▲  Top of the page")
        self._top_bar.setAlignment(Qt.AlignCenter)
        self._top_bar.setMinimumHeight(48)
        self._top_bar.setStyleSheet(self._bar_style(active=False))
        layout.addWidget(self._top_bar)

        # Upper-instruction text. The animated tutorial_scrolling.mp4
        # demo used to live to the right of this text — it's now in
        # the top-right of the LIVE CAMERA VIEW instead (the same
        # spot the left_three / mouse_clicks demos use), so users
        # see the pose without having to look away from the camera.
        upper_text = QLabel(
            "Scroll by holding two fingers up and drifting your hand "
            "<b>up</b> (to scroll up) or <b>down</b> (to scroll down).<br><br>"
            "Reach the top bar and the bottom bar to complete this step."
        )
        upper_text.setWordWrap(True)
        upper_text.setTextFormat(Qt.RichText)
        upper_text.setStyleSheet("color: #E8F6FF; font-size: 13px;")
        layout.addWidget(upper_text)

        # Tall middle spacer — pushes the bottom bar past the
        # viewport so the user has to actually scroll.
        spacer = QWidget()
        spacer.setMinimumHeight(640)
        spacer.setStyleSheet(
            "background: rgba(255,255,255,0.04);"
            " border: 1px dashed rgba(255,255,255,0.18);"
            " border-radius: 12px;"
        )
        spacer_layout = QVBoxLayout(spacer)
        spacer_layout.setContentsMargins(20, 20, 20, 20)
        spacer_label = QLabel(
            "Keep scrolling.\n\nWhen you reach the bar at the bottom, "
            "the bar turns green and you can head back to the top."
        )
        spacer_label.setAlignment(Qt.AlignCenter)
        spacer_label.setWordWrap(True)
        spacer_label.setStyleSheet("color: #A6C7D7; font-size: 12px;")
        spacer_layout.addStretch(1)
        spacer_layout.addWidget(spacer_label)
        spacer_layout.addStretch(1)
        layout.addWidget(spacer)

        # Footer text "below" the practice area.
        lower_text = QLabel(
            "Almost there — touch the bottom bar, then scroll back up "
            "to finish this part."
        )
        lower_text.setWordWrap(True)
        lower_text.setStyleSheet("color: #E8F6FF; font-size: 13px;")
        layout.addWidget(lower_text)

        # Bottom bar.
        self._bottom_bar = QLabel("▼  Bottom of the page")
        self._bottom_bar.setAlignment(Qt.AlignCenter)
        self._bottom_bar.setMinimumHeight(48)
        self._bottom_bar.setStyleSheet(self._bar_style(active=False))
        layout.addWidget(self._bottom_bar)

        # Status line that mirrors which extremity the user still
        # needs to reach. Lives inside the scroll content so it
        # follows the user as they navigate.
        self._status_label = QLabel(self._status_text)
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(
            "color: #1DE9B6; font-size: 13px; font-weight: 700;"
        )
        layout.addWidget(self._status_label)

        self.setWidget(content)
        # Listen for scrollbar moves so we can mark top / bottom
        # reached. A tolerance of 2 px on each extremity covers the
        # 1-px rounding the Qt scrollbar sometimes produces at the
        # very ends of the range.
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    @staticmethod
    def _bar_style(active: bool) -> str:
        if active:
            return (
                "background: rgba(29, 233, 182, 0.20);"
                " color: #1DE9B6;"
                " border: 1.5px solid #1DE9B6;"
                " border-radius: 10px;"
                " font-size: 15px; font-weight: 800;"
                " padding: 12px;"
            )
        return (
            "background: rgba(255, 82, 82, 0.16);"
            " color: #FF8A8A;"
            " border: 1.5px solid rgba(255, 82, 82, 0.55);"
            " border-radius: 10px;"
            " font-size: 15px; font-weight: 800;"
            " padding: 12px;"
        )

    def apply_theme(self, accent: str, text: str) -> None:
        self._accent = QColor(accent)
        self._text = QColor(text)
        self._refresh_bars()

    def reset(self) -> None:
        self._top_reached = False
        self._bottom_reached = False
        self._status_text = "Use mouse mode to scroll up to the top bar."
        self._refresh_bars()
        self._status_label.setText(self._status_text)
        # Reset scroll position to a neutral midway start so neither
        # extremity is auto-claimed when the widget shows.
        try:
            sb = self.verticalScrollBar()
            sb.setValue((sb.minimum() + sb.maximum()) // 2)
        except Exception:
            pass

    @property
    def completed(self) -> bool:
        return self._top_reached and self._bottom_reached

    @property
    def top_reached(self) -> bool:
        return self._top_reached

    @property
    def bottom_reached(self) -> bool:
        return self._bottom_reached

    def register_scroll_steps(self, steps: int) -> None:
        """Programmatically scroll the practice area by the number of
        scroll-wheel steps emitted by the tutorial's mouse-gesture
        tracker. Positive = up, negative = down. The widget would
        otherwise only respond to real OS-level wheel events landing
        on its viewport — but in tutorial mode the engine doesn't
        actually fire OS wheel events (those would scroll whatever
        the desktop cursor sits over, not the tutorial), so we
        translate steps → scrollbar deltas ourselves.

        Step → pixel mapping: one logical step = the QScrollArea's
        own SingleStep, which gives a per-frame scroll amount close
        to what a physical wheel notch would produce on a standard
        Qt scrollable widget.
        """
        if steps == 0:
            return
        try:
            sb = self.verticalScrollBar()
            step_pixels = max(20, int(sb.singleStep() or 20))
            # Negative `steps` means scroll DOWN (raise scrollbar
            # value); positive `steps` means scroll UP (lower
            # scrollbar value). Matches QAbstractSlider behaviour.
            sb.setValue(sb.value() - int(steps) * step_pixels)
        except Exception:
            pass

    def _on_scroll(self, value: int) -> None:
        sb = self.verticalScrollBar()
        if value <= sb.minimum() + 2:
            if not self._top_reached:
                self._top_reached = True
                self._refresh_bars()
        if value >= sb.maximum() - 2:
            if not self._bottom_reached:
                self._bottom_reached = True
                self._refresh_bars()
        # Status copy updates based on which extremes have been hit.
        if self.completed:
            new_status = "Scrolling done! Turn mouse mode off to finish."
        elif self._top_reached and not self._bottom_reached:
            new_status = "Top reached. Now scroll down to the bottom bar."
        elif self._bottom_reached and not self._top_reached:
            new_status = "Bottom reached. Now scroll up to the top bar."
        else:
            new_status = "Use mouse mode to scroll up to the top bar."
        if new_status != self._status_text:
            self._status_text = new_status
            self._status_label.setText(new_status)

    def _refresh_bars(self) -> None:
        self._top_bar.setStyleSheet(self._bar_style(active=self._top_reached))
        self._bottom_bar.setStyleSheet(self._bar_style(active=self._bottom_reached))


class _VoiceMicArrow(QWidget):
    """Bouncing arrow rendered as a click-through overlay on the
    tutorial window during the voice-command practice step. Points
    from a fixed anchor inside the tutorial window toward the
    absolute screen position where the VoiceStatusOverlay appears
    (bottom-center of the tutorial window's screen).

    Geometry tracking: TutorialWindow.moveEvent / resizeEvent call
    update_target_from_screen() so dragging the window keeps the
    arrow on-target. The arrow direction is recomputed every paint
    from the current global anchor position vs the cached target,
    so even cross-monitor drags re-aim correctly.

    The bounce is a sine-wave displacement along the arrow's own
    direction vector — gives a "look this way" pulse without the
    distracting wobble of a perpendicular bob.
    """

    def __init__(self, parent: QWidget, accent: QColor) -> None:
        super().__init__(parent)
        self._accent = QColor(accent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._target_global = QPoint(0, 0)
        self._bounce_phase = 0.0
        # ~30 FPS animation tick. Cheap; only runs while the arrow
        # is visible (started in show_pointing, stopped in hide_arrow).
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(33)
        self._tick_timer.timeout.connect(self._on_tick)
        self.hide()

    def _on_tick(self) -> None:
        # Increment phase so a full sine cycle takes ~600 ms.
        self._bounce_phase += 0.18
        if self._bounce_phase > 6.283185:
            self._bounce_phase -= 6.283185
        self.update()

    def update_target_from_screen(self) -> None:
        """Cache the global pixel coordinate the arrow should point
        at. Mirrors VoiceStatusOverlay._place_on_screen — bottom-
        center of the available screen geometry, ~50 px up from the
        bottom edge to land in the middle of the overlay's body
        rather than its bottom edge."""
        parent = self.parentWidget()
        if parent is None:
            return
        screen = parent.screen() if hasattr(parent, "screen") else None
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self._target_global = QPoint(geo.center().x(), geo.bottom() - 50)
        self.update()

    @staticmethod
    def _styled_pen(color: QColor, width: float) -> QPen:
        pen = QPen(color)
        pen.setWidthF(float(width))
        pen.setStyle(Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    @staticmethod
    def _draw_arrow_segments(
        painter: QPainter,
        start: QPointF,
        tip: QPointF,
        head_a: QPointF,
        head_b: QPointF,
    ) -> None:
        painter.drawLine(start, tip)
        painter.drawLine(tip, head_a)
        painter.drawLine(tip, head_b)

    @staticmethod
    def _arrow_path(
        start: QPointF,
        tip: QPointF,
        head_a: QPointF,
        head_b: QPointF,
    ) -> QPainterPath:
        path = QPainterPath(start)
        path.lineTo(tip)
        path.lineTo(head_a)
        path.moveTo(tip)
        path.lineTo(head_b)
        return path

    def paintEvent(self, ev) -> None:  # noqa: N802 — Qt naming
        # Anchor: bottom-center of this overlay (which is sized to
        # cover the tutorial window). Leave 90 px margin above the
        # bottom edge so the shaft + arrowhead doesn't run off.
        anchor_x = self.width() // 2
        anchor_y = max(60, self.height() - 90)
        anchor_global = self.mapToGlobal(QPoint(anchor_x, anchor_y))
        dx = self._target_global.x() - anchor_global.x()
        dy = self._target_global.y() - anchor_global.y()
        length = math.hypot(dx, dy)
        if length < 1.0:
            return
        ux = dx / length
        uy = dy / length

        # Bounce along arrow direction plus a slightly slower breathing
        # glow that fattens the halo. The glow is intentionally a cool
        # blue-white so it reads as a separate pulse layer around the
        # mint arrow instead of the whole arrow just getting brighter.
        bounce = math.sin(self._bounce_phase) * 12.0
        glow_pulse = 0.5 + 0.5 * math.sin(self._bounce_phase * 0.82 - 0.7)
        ax = anchor_x + ux * bounce
        ay = anchor_y + uy * bounce

        SHAFT_LEN = 132.0
        HEAD_LEN = 34.0
        HEAD_HALFWIDTH = 22.0

        sx = ax - ux * SHAFT_LEN * 0.5
        sy = ay - uy * SHAFT_LEN * 0.5
        ex = ax + ux * SHAFT_LEN * 0.5
        ey = ay + uy * SHAFT_LEN * 0.5

        # Perpendicular for arrowhead barbs
        px, py = -uy, ux
        h1x = ex - ux * HEAD_LEN + px * HEAD_HALFWIDTH
        h1y = ey - uy * HEAD_LEN + py * HEAD_HALFWIDTH
        h2x = ex - ux * HEAD_LEN - px * HEAD_HALFWIDTH
        h2y = ey - uy * HEAD_LEN - py * HEAD_HALFWIDTH

        start = QPointF(sx, sy)
        tip = QPointF(ex, ey)
        head_a = QPointF(h1x, h1y)
        head_b = QPointF(h2x, h2y)
        path = self._arrow_path(start, tip, head_a, head_b)
        shadow_path = self._arrow_path(
            QPointF(sx + 3.0, sy + 3.0),
            QPointF(ex + 3.0, ey + 3.0),
            QPointF(h1x + 3.0, h1y + 3.0),
            QPointF(h2x + 3.0, h2y + 3.0),
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        # Drop shadow first, then accent-colored shaft/head on top —
        # gives the arrow enough contrast to stay readable over the
        # camera view's dark background AND the lighter side panels.
        shadow = QColor(0, 0, 0, 145)
        painter.setPen(self._styled_pen(shadow, 15.0))
        painter.drawPath(shadow_path)

        outer_glow = QColor(92, 184, 255)
        outer_glow.setAlpha(int(54 + 44 * glow_pulse))
        painter.setPen(self._styled_pen(outer_glow, 24.0 + 7.0 * glow_pulse))
        painter.drawPath(path)

        inner_glow = QColor(64, 156, 255)
        inner_glow.setAlpha(int(88 + 48 * glow_pulse))
        painter.setPen(self._styled_pen(inner_glow, 15.0 + 4.0 * glow_pulse))
        painter.drawPath(path)

        core = QColor(self._accent)
        core.setAlpha(245)
        painter.setPen(self._styled_pen(core, 8.8))
        painter.drawPath(path)

        # Small callout badge that rides next to the arrow so the user
        # immediately understands what the cue is pointing at.
        label_text = "Look this way for mic!"
        label_font = QFont("Segoe UI", 11)
        label_font.setBold(True)
        painter.setFont(label_font)
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(label_text)
        box_w = float(text_rect.width() + 26)
        box_h = float(text_rect.height() + 18)
        mid_x = (sx + ex) * 0.5
        mid_y = (sy + ey) * 0.5
        label_cx = mid_x - px * 98.0
        label_cy = mid_y - py * 98.0
        box_x = max(8.0, min(float(self.width()) - box_w - 8.0, label_cx - box_w * 0.5))
        box_y = max(8.0, min(float(self.height()) - box_h - 8.0, label_cy - box_h * 0.5))
        bubble_rect = QRectF(box_x, box_y, box_w, box_h)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.drawRoundedRect(bubble_rect.adjusted(2.0, 2.0, 2.0, 2.0), 12.0, 12.0)
        painter.setBrush(QColor(10, 22, 34, 230))
        painter.setPen(QPen(QColor(92, 184, 255, 190), 1.4))
        painter.drawRoundedRect(bubble_rect, 12.0, 12.0)
        painter.setPen(QColor(229, 246, 255))
        painter.drawText(bubble_rect, int(Qt.AlignCenter), label_text)


    def show_pointing(self) -> None:
        self.update_target_from_screen()
        self.raise_()
        self.show()
        self._bounce_phase = 0.0
        self._tick_timer.start()

    def hide_arrow(self) -> None:
        self._tick_timer.stop()
        self.hide()


class _TutorialStartingPill(QWidget):
    """Bottom-centred pill shown over the tutorial dialog while the
    session is spinning up. Paints exactly like the desktop
    ProcessingOverlay (translucent blue panel, teal border, label
    on top, thin progress bar at the bottom) but as a child widget
    of the dialog -- no top-level translucency / DWM games needed.
    Per the app's loading-affordance policy: progress bar for
    starting app / tutorial / processing saves; dots reserved for
    voice and incidental status."""

    _WIDTH = 220
    _HEIGHT = 88

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self._label = "Starting tutorial"
        # Progress fields mirror the desktop ProcessingOverlay so the
        # bar visibly creeps forward even without explicit checkpoints
        # from the tutorial-spin-up path. Cap at 0.92 so the bar never
        # "completes" until hide_pill() is called.
        self._progress = 0.0
        self._progress_target = 0.0
        self._progress_idle_creep_rate = 0.05  # fraction/sec
        self._progress_idle_creep_cap = 0.92
        self._last_tick_time = time.monotonic()
        self._timer = QTimer(self)
        # PreciseTimer at 60 fps so the wave dots advance at a steady
        # cadence on Windows. The default CoarseTimer rounds to ~15.6
        # ms (the system tick), turning a 40 ms interval into uneven
        # 30 / 45 ms ticks that read as "freezing then jumping".
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.setFixedSize(self._WIDTH, self._HEIGHT)
        self.setVisible(False)

    def show_pill(self, label: str = "Starting tutorial") -> None:
        self._label = str(label or "Starting tutorial")
        # Reset the progress fields on every show so a re-opened tutorial
        # doesn't inherit the previous run's leftover bar position.
        self._progress = 0.0
        self._progress_target = 0.0
        self._last_tick_time = time.monotonic()
        self.setVisible(True)
        self.raise_()
        self._timer.start()
        self.update()

    def hide_pill(self) -> None:
        self._timer.stop()
        self.setVisible(False)

    def set_label(self, label: str) -> None:
        self._label = str(label or "")
        if self.isVisible():
            self.update()

    def _tick(self) -> None:
        # Idle creep + ease-toward-target, matching ProcessingOverlay so
        # the bar visibly advances during the tutorial spin-up even if
        # nothing calls set_progress() explicitly. dt is clamped so a
        # paused-by-debugger jump doesn't blow the bar past the cap.
        now = time.monotonic()
        dt = max(0.0, min(0.5, now - self._last_tick_time))
        self._last_tick_time = now
        self._progress_target = min(
            self._progress_idle_creep_cap,
            self._progress_target + self._progress_idle_creep_rate * dt,
        )
        # Ease current toward target by ~30 % each frame — smooth without
        # being floaty.
        self._progress += (self._progress_target - self._progress) * 0.30
        if self.isVisible():
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt API name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        panel = QColor(25, 73, 143, 164)
        border = QColor(29, 233, 182, 210)
        text_color = QColor(232, 246, 255, 238)
        accent = QColor(29, 233, 182)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(border, 1.2))
        painter.setBrush(panel)
        painter.drawRoundedRect(rect, 18.0, 18.0)

        font = QFont("Segoe UI", 12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(text_color))
        label_rect = QRectF(rect.left() + 12, rect.top() + 14, rect.width() - 24, 24)
        painter.drawText(label_rect, Qt.AlignCenter, self._label)

        # Progress bar — same geometry / palette as ProcessingOverlay so
        # the two pills read as one visual family.
        bar_h = 6.0
        bar_y = rect.bottom() - 20
        bar_left = rect.left() + 18
        bar_right = rect.right() - 18
        bar_w = bar_right - bar_left
        track = QColor(accent.red(), accent.green(), accent.blue(), 55)
        fill = QColor(accent.red(), accent.green(), accent.blue(), 235)
        painter.setPen(Qt.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(QRectF(bar_left, bar_y, bar_w, bar_h), bar_h / 2.0, bar_h / 2.0)
        fill_w = bar_w * float(self._progress)
        if fill_w > 0.5:
            painter.setBrush(fill)
            painter.drawRoundedRect(QRectF(bar_left, bar_y, fill_w, bar_h), bar_h / 2.0, bar_h / 2.0)


class TutorialWindow(QDialog):
    tutorial_closed = Signal(bool, bool, bool)
    gesture_guide_requested = Signal(bool)
    # Cross-thread bridge for the voice-command media-playing
    # check. The Spotify Web API call + chrome window scan run on
    # a daemon thread (they used to block the GUI thread for
    # 100-300 ms each, which made the live view lag and prevented
    # right-swipe-to-advance from being detected). The worker
    # emits (is_playing, is_final_pass); the slot runs on the GUI
    # thread and decides what to do (complete the step / schedule
    # the next attempt).
    _voice_media_check_signal = Signal(bool, bool)
    # Cross-thread bridge for the PERIODIC Spotify-playback poll
    # the voice_command step uses to recognise "the user is already
    # playing music" -- whether they voice-commanded it or hit space
    # in Spotify by hand. The slot completes the step iff playing=True;
    # it deliberately does NOT have a 'settle for voice success'
    # fallback (the one-shot _voice_media_check path owns that).
    _spotify_periodic_poll_signal = Signal(bool)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        # r51: install_indigo_chrome for Win10 + Win11 parity.
        from .window_chrome import install_indigo_chrome
        self._body = install_indigo_chrome(self, "Touchless Tutorial")
        # Connect the off-thread media-check bridges to their slots.
        self._voice_media_check_signal.connect(self._on_voice_media_check_result)
        self._spotify_periodic_poll_signal.connect(self._on_spotify_periodic_poll)
        # Throttle / single-flight state for the periodic poll.
        self._spotify_poll_inflight: bool = False
        self._spotify_poll_last_at: float = 0.0
        self.config = config
        self._camera_index: Optional[int] = None
        self._launched_from_settings = False
        self._auto_start_on_done = False
        self._camera_info = None
        self._cap = None
        self._engine: GestureRecognitionEngine | None = None
        self._worker: GestureWorker | None = None
        self._owns_worker = False
        self._shared_worker_connected = False
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        # Focus-guard timer for the voice_command step. Spotify (and
        # Chrome on the YouTube branch) steal foreground focus the
        # moment they start playing, leaving the tutorial behind
        # whichever window the user just summoned. The single
        # _reclaim_tutorial_focus call wired to step completion only
        # fires AFTER the engine confirms playback — between the
        # voice-command and the polling-detects-playback edge, the
        # user is stuck looking at Spotify. This timer polls the
        # foreground window every 500 ms while on the voice_command
        # step and force-foregrounds the tutorial whenever the
        # foreground HWND isn't ours. 500 ms is the sweet spot
        # between responsiveness (sub-second handback) and
        # background-cost (less than 0.1% CPU on idle ticks).
        self._focus_guard_timer = QTimer(self)
        self._focus_guard_timer.setInterval(500)
        self._focus_guard_timer.timeout.connect(self._focus_guard_tick)
        self._last_dynamic_label = "neutral"
        self._swipe_counts = {"swipe_left": 0, "swipe_right": 0}
        # Timestamp of the most-recent swipe-count increment. Used by
        # _refresh_swipe_camera_labels to show 'Completed right
        # swipes!' / 'Completed left swipes!' for a brief moment
        # before the next prompt.
        self._swipe_last_advance_at: float = 0.0
        self._hold_started: dict[str, float] = {}
        self._hold_last_fired: dict[str, float] = {}
        self._close_emitted = False
        self._closing_programmatically = False
        self._voice_listener = VoiceCommandListener(
            preferred_input_device=getattr(config, "preferred_microphone_name", None),
            input_gain=getattr(config, "mic_input_gain", 1.0),
        )
        # Phone-mic plumbing: if the user has the phone-as-mic toggle
        # on AND a paired phone QR server is running on the parent
        # MainWindow, route its PhoneAudioSource into the tutorial's
        # voice listener too. Without this, opening the tutorial cold
        # (before the main app's worker exists) on a phone-only mic
        # setup left the part-6 voice command listening on a missing
        # local sounddevice → instant "Command not understood".
        try:
            self._wire_phone_mic_to_voice_listener()
        except Exception:
            pass
        self._chrome_controller = ChromeController()
        self._spotify_controller = SpotifyController()
        self._voice_processor = VoiceCommandProcessor(
            chrome_controller=self._chrome_controller,
            spotify_controller=self._spotify_controller,
        )
        self._tutorial_wheel_overlay = SpotifyWheelOverlay(config)
        self._voice_overlay = None
        self._voice_queue: queue.Queue[tuple[int, dict]] = queue.Queue()
        self._voice_request_id = 0
        self._voice_listening = False
        self._voice_heard_text = ""
        self._voice_status = "ready"
        self._mouse_tracker = MouseGestureTracker()
        # When the tutorial runs against the shared parent worker,
        # the local _mouse_tracker is NOT updated — the parent
        # engine's mouse_tracker is. Mirror the latest payload's
        # mouse-overlay info here so _draw_demo_overlay can render
        # the same red control-area box the regular live view
        # renders.
        self._payload_mouse_state: dict | None = None
        self._spotify_open_hold_seconds = 1.0
        self._spotify_static_cooldown_seconds = 1.5
        self._spotify_play_pause_hold_seconds = 0.5
        self._wheel_hold_seconds = 1.0
        self._wheel_cooldown_seconds = 1.5
        self._tutorial_nav_cooldown_seconds = 1.5
        self._visual_hold_started: dict[str, float] = {}
        self._visual_green_until: dict[str, float] = {}
        self._visual_edge_active: dict[str, bool] = {}
        self._gesture_flash_seconds = 1.0
        self._completion_feedback_until = 0.0
        self._completion_feedback_duration = 2.0
        self._completion_feedback_step = -1
        # Encouragement-popup state. When a step's _visual_green_until is
        # newly pushed forward (i.e., the user just got a correct
        # detection and the hand turned green), pick a random message
        # and fade it out over 1.5 s at the bottom-center of the
        # camera view. Throttled to 1 per second so the same swipe
        # can't stack three popups.
        self._last_seen_green_until: dict[str, float] = {}
        self._encouragement_until: float = 0.0
        self._encouragement_last_at: float = 0.0
        self._play_pause_ready_for_next = True
        self._swipe_goal_index = 0
        self._nav_swipe_cooldown_until = 0.0
        self._spotify_toggle_count = 0
        self._mouse_stage = "enable"
        self._mouse_cursor_seen = False
        self._tutorial_wheel_anchor = None
        self._tutorial_wheel_selected_key: str | None = None
        self._tutorial_wheel_selected_since = 0.0
        self._tutorial_wheel_cursor_offset: tuple[float, float] | None = None
        self._last_spotify_tutorial_action = "-"
        self._last_tutorial_play_pause_text = ""
        self._last_voice_success_text = ""
        self._prime_voice_runtime_async()
        # Detect whether Spotify desktop is installed. Drives the
        # "play media" voice step (Spotify vs YouTube) and decides
        # whether play/pause + swipes route through Spotify Web API
        # or the OS media keys (which YouTube + every other media
        # app respect via Windows Media integration).
        self._has_spotify = self._detect_spotify_installed()
        # Spotify-first-active prompt forwarding. The main app's
        # per-frame Spotify-detection path doesn't fire while the
        # tutorial is in front (parent engine is frozen and the
        # tutorial owns the live debug-frame stream). If the user
        # already has Spotify running but never authorised Touchless,
        # we'd silently skip the prompt for the entire tutorial — so
        # ask the parent MainWindow to run its one-shot startup check
        # immediately on tutorial open. The check is idempotent
        # (gated on config.spotify_first_active_prompt_shown), so
        # it's a no-op for already-prompted users.
        try:
            parent = self.parent()
            check = getattr(parent, "_check_spotify_at_startup", None)
            if callable(check):
                from PySide6.QtCore import QTimer as _QT
                _QT.singleShot(800, check)
        except Exception:
            pass
        # Five-step practice flow. Swipes come first so users learn
        # left/right-swipe early — the same gesture the tutorial uses
        # to advance / go back between pages — then the audio-driven
        # steps follow once the user is fluent with navigation.
        self._practice_steps = (
            _StepDefinition("swipes", "Part 1/5: Swipe Gestures", "", ""),
            _StepDefinition("mouse_mode", "Part 2/5: Mouse Control", "", ""),
            _StepDefinition("voice_command", "Part 3/5: Voice Command", "", ""),
            _StepDefinition("volume", "Part 4/5: Volume Control", "", ""),
            _StepDefinition("play_pause", "Part 5/5: Pause/Play", "", ""),
        )
        self._step_index = 0
        self._completed_steps: set[int] = set()
        self._step_completed = self._step_index in self._completed_steps
        self._show_completion_page = False
        self._build_ui()
        self.apply_theme(config)
        self._reset_for_step()

    def _detect_spotify_installed(self) -> bool:
        """Cheap one-shot check at tutorial open: do any of the
        Spotify executable paths SpotifyController already knows
        about exist on disk? No process scan, no network call --
        just file-existence checks that take microseconds.

        Yes -> tutorial uses the existing Spotify Web API path for
        the media voice step + play/pause + swipes.
        No  -> tutorial swaps to a YouTube voice command and the
        OS media-key path (VK_MEDIA_PLAY_PAUSE / NEXT / PREV) so
        YouTube / Edge / VLC / etc. respond.
        """
        try:
            controller = SpotifyController()
            for candidate in getattr(controller, "_executable_paths", ()):
                try:
                    if Path(candidate).exists():
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    @staticmethod
    def _send_media_key(virtual_key: int) -> bool:
        """Press + release a Windows media virtual key.
        VK_MEDIA_PLAY_PAUSE = 0xB3, VK_MEDIA_NEXT_TRACK = 0xB0,
        VK_MEDIA_PREV_TRACK = 0xB1. Used in the no-Spotify tutorial
        path so play/pause + swipes drive YouTube (or any app that
        registers as a Windows media controller) without going
        through Spotify Web API.

        Uses keybd_event because it's the simplest path and works
        on Windows 7+; SendInput has the same effect with more
        boilerplate. Returns True on apparent success, False if
        we're not on Windows or the call failed."""
        if sys.platform != "win32":
            return False
        try:
            import ctypes
            KEYEVENTF_EXTENDEDKEY = 0x0001
            KEYEVENTF_KEYUP = 0x0002
            user32 = ctypes.windll.user32
            user32.keybd_event(virtual_key, 0, KEYEVENTF_EXTENDEDKEY, 0)
            user32.keybd_event(virtual_key, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    def _prime_voice_runtime_async(self) -> None:
        def _worker() -> None:
            try:
                self._voice_listener.prewarm()
            except Exception:
                pass

        threading.Thread(
            target=_worker,
            name="hgr-tutorial-voice-prewarm",
            daemon=True,
        ).start()

    def _build_ui(self) -> None:
        self.setWindowTitle("Touchless Tutorial")
        self.setModal(False)
        # Slightly taller than the main app (1020×820) per UX request
        # so the live camera view gets meaningfully larger without
        # widening the dialog. Min height bumped in parallel so
        # constrained displays still get the bigger camera region.
        self.resize(1020, 820)
        self.setMinimumSize(880, 740)

        # In-window "Starting..." pill. Floating QLabel anchored at
        # bottom-center of the tutorial dialog (NOT a desktop-level
        # overlay -- per user spec, keep tutorial's loading
        # indicator inside the dialog so it doesn't compete with the
        # main app's desktop-pill if they overlap). Visible during
        # _start_session() until the first camera frame arrives.
        self._starting_pill = _TutorialStartingPill(self)
        self._starting_pill.setVisible(False)

        root = QVBoxLayout(self._body)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("tutorialHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 16, 18, 16)
        header_layout.setSpacing(12)

        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(4)
        self.hero_label = QLabel("Touchless Tutorial")
        self.hero_label.setObjectName("tutorialHero")
        # Static subtitle removed per UX feedback — was always the
        # same "Practice the main controls..." string on every step
        # and added noise to the header. Step-specific copy is shown
        # inside the body card (step.title + step.description). Kept
        # as a hidden zero-height QLabel so existing show()/hide()
        # callers don't NameError.
        self.hero_subtitle = QLabel("")
        self.hero_subtitle.setObjectName("tutorialSubtitle")
        self.hero_subtitle.setWordWrap(True)
        self.hero_subtitle.setVisible(False)
        self.hero_subtitle.setMaximumHeight(0)
        title_wrap.addWidget(self.hero_label)
        title_wrap.addWidget(self.hero_subtitle)
        header_layout.addLayout(title_wrap, 1)

        self.progress_badge = QLabel("Step 1 of 6")
        self.progress_badge.setObjectName("tutorialBadge")
        header_layout.addWidget(self.progress_badge, 0, Qt.AlignTop)
        root.addWidget(header)

        self.body_stack = QStackedWidget()
        root.addWidget(self.body_stack, 1)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(18, 18, 18, 18)
        body_layout.setSpacing(18)
        self.body_stack.addWidget(body)

        # ---- LEFT CARD ----------------------------------------------
        # User-requested single-panel layout for all steps. Visual
        # order: Part X/5 title → short description (+ Show more
        # link) → Show Example button → live camera view → footer
        # words (camera_header above, tutorial_camera_footer below)
        # → remaining diagnostic / completion labels. The mouse step
        # additionally shows a right card with the practice arena;
        # all other steps show only this left card so the page looks
        # and feels like one panel.
        video_card = QFrame()
        video_card.setObjectName("tutorialCard")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(16, 16, 16, 16)
        video_layout.setSpacing(10)

        # --- Top row: title on the left, all step-action buttons on
        # the right. Mouse / Spotify / example buttons share the same
        # right cluster — only one or two are visible per step but
        # the layout keeps them top-right consistently.
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(10)

        self.step_title = QLabel("")
        self.step_title.setObjectName("tutorialStepTitle")
        # Bigger title font — overrides global QSS for #tutorialStepTitle.
        self.step_title.setStyleSheet(
            "QLabel#tutorialStepTitle {"
            "  color: #1DE9B6;"
            "  font-size: 26px;"
            "  font-weight: 800;"
            "  letter-spacing: -0.01em;"
            "}"
        )
        self.step_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # Don't let the title stretch — fix its size policy to its
        # text width so the description sits right next to it
        # without a giant gap.
        self.step_title.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        title_row.addWidget(self.step_title, 0, Qt.AlignLeft | Qt.AlignVCenter)
        title_row.addStretch(1)

        # step_desc (white inline description) removed per UX request.
        # The widget is still constructed and hidden so legacy
        # code paths that call self.step_desc.setText(...) keep
        # working without modification — they just write into a
        # dead widget. Avoids a wider sweep through _apply_step_content
        # and related callers.
        self.step_desc = QLabel("")
        self.step_desc.setObjectName("tutorialStepDesc")
        self.step_desc.hide()

        # Right-side button cluster. Spotify-setup first so it sits
        # leftmost in the cluster, mouse-instructions in the middle,
        # Show Example as the rightmost (most-used) action.
        self.spotify_setup_button = QPushButton("Set up Spotify")
        self.spotify_setup_button.setVisible(False)
        self.spotify_setup_button.clicked.connect(self._open_spotify_setup_from_tutorial)
        title_row.addWidget(self.spotify_setup_button, 0, Qt.AlignRight)

        # Mouse instructions button removed per UX request — Show
        # Example covers the same destination (the example popup
        # already carries the full mouse-step instructions text). We
        # still construct the widget hidden so any back-compat call
        # to setVisible(...) doesn't hit an AttributeError.
        self.mouse_instructions_button = QPushButton("Mouse instructions")
        self.mouse_instructions_button.setVisible(False)
        self.mouse_instructions_button.clicked.connect(self._open_step_example)
        # NOT added to title_row — kept as an orphan widget so it
        # never displays. setVisible(True) elsewhere is a no-op since
        # it has no parent layout.
        # Back-compat shim — older code still references the attribute.
        self.expand_instructions_button = self.mouse_instructions_button

        self.example_button = QPushButton("Show Example")
        self.example_button.clicked.connect(self._open_step_example)
        title_row.addWidget(self.example_button, 0, Qt.AlignRight)

        video_layout.addLayout(title_row)

        # Mouse-step long-form instructions (back-compat — still used
        # by _open_step_example to compose the popup's "Detailed
        # instructions" section). Kept invisible inline since the
        # text is now surfaced only through the popup.
        self.instruction_box = QLabel("")
        self.instruction_box.setObjectName("tutorialInstructionBox")
        self.instruction_box.setWordWrap(True)
        self.instruction_box.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.instruction_box.setVisible(False)

        # --- Camera + footer (was video_card content) -----------------
        # "Camera: waiting" meta label removed per UX request — the
        # camera_label widget is still constructed and hidden to keep
        # the rest of the engine's update path working (it sets
        # camera_label.setText(...) on every frame to report device
        # state).
        self.camera_label = QLabel("")
        self.camera_label.setObjectName("tutorialMeta")
        self.camera_label.hide()

        # Big bold accent-coloured header above the camera view. Used
        # by the swipes step to call out the current sub-task ("Let's
        # start with swiping right!" then "Now let's try swiping to
        # the left!"). Hidden for steps that don't set it.
        self.tutorial_camera_header = QLabel("")
        self.tutorial_camera_header.setObjectName("tutorialCameraHeader")
        self.tutorial_camera_header.setAlignment(Qt.AlignCenter)
        self.tutorial_camera_header.setWordWrap(True)
        # Rich-text + clickable so the "Show more…" link appended by
        # `_set_camera_header_text` can route to the example popup.
        # Routes to the same _on_step_desc_link handler the step_desc
        # link used before — opens the step's example dialog.
        self.tutorial_camera_header.setTextFormat(Qt.RichText)
        self.tutorial_camera_header.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.tutorial_camera_header.setOpenExternalLinks(False)
        self.tutorial_camera_header.linkActivated.connect(self._on_step_desc_link)
        self.tutorial_camera_header.hide()
        video_layout.addWidget(self.tutorial_camera_header)

        self.video_label = QLabel("The tutorial will show your live hand skeleton here.")
        self.video_label.setObjectName("tutorialVideo")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setWordWrap(True)
        # Floor + ceiling both bumped per UX request — bigger camera
        # view fills the new vertical space the resized window
        # provides. 440 min floor (was 360) so even minimum-window
        # users see a meaningfully bigger image; 640 ceiling (was 540)
        # so larger windows can grow further without the camera
        # stalling at a small ratio.
        self.video_label.setMinimumSize(480, 440)
        self.video_label.setMaximumHeight(640)
        video_layout.addWidget(self.video_label, 1)

        # Floating "✓ + Completed!" overlay child of video_label.
        # Stays visible at a fixed semi-transparent level whenever
        # the current step is in completed state — no fade animation
        # (the previous fade-in/fade-out kept restarting on every
        # _update_completion_feedback tick which manifested as a
        # flashing glyph). Alpha is baked into the stylesheet color
        # via rgba(...) so we don't need QGraphicsOpacityEffect.
        # Positioned + sized by an eventFilter on video_label so it
        # tracks the camera widget's geometry as the window resizes.
        self._fade_check_label = QLabel("✓", self.video_label)
        self._fade_check_label.setObjectName("tutorialFadeCheck")
        self._fade_check_label.setAlignment(Qt.AlignCenter)
        self._fade_check_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._fade_check_label.hide()
        # Sub-label below the check. Larger than a bare caption so
        # the "Completed!" reads at a glance, but smaller than the
        # main glyph.
        self._fade_check_caption = QLabel("Completed!", self.video_label)
        self._fade_check_caption.setObjectName("tutorialFadeCheckCaption")
        self._fade_check_caption.setAlignment(Qt.AlignCenter)
        self._fade_check_caption.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._fade_check_caption.hide()
        # Tracks resize events on video_label so the overlay stays
        # centred / sized as the window grows.
        self.video_label.installEventFilter(self)

        self.gesture_chip = QLabel("Gesture: neutral")
        self.gesture_chip.setObjectName("tutorialChip")
        self.gesture_chip.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.gesture_chip, 0, Qt.AlignCenter)

        # Big bold accent-coloured counter below the camera view —
        # paired with tutorial_camera_header. Swipes step uses it to
        # show "Completed N/3 right swipes" / "Completed N/3 left
        # swipes". Hidden for steps that don't set it.
        self.tutorial_camera_footer = QLabel("")
        self.tutorial_camera_footer.setObjectName("tutorialCameraFooter")
        self.tutorial_camera_footer.setAlignment(Qt.AlignCenter)
        self.tutorial_camera_footer.setWordWrap(True)
        self.tutorial_camera_footer.hide()
        video_layout.addWidget(self.tutorial_camera_footer)
        body_layout.addWidget(video_card, 7)

        # Encouragement popup is drawn directly onto the camera frame
        # via cv2.putText in _render_frame (see _draw_encouragement_overlay).
        # No QLabel — keeping the popup *inside* the live camera image
        # rather than overlapping it with a Qt widget so it reads as
        # part of the video, not a chrome-on-top decoration.
        self._encouragement_text = ""
        # Lazy RGBA sprite cache. Sprites are loaded on first
        # request via _sprite_for(name) and reused for the rest of
        # the session. "missing" sentinel distinguishes "not yet
        # tried" from "tried and failed" to avoid repeated disk
        # lookups.
        self._sprites: dict = {}

        # ---- RIGHT CARD ---------------------------------------------
        # Only visible on the mouse step (where the user practises
        # clicking targets) AND when the completion-overlay fires for
        # any step (so the "Completed ✓" check has somewhere to sit).
        # Every other step hides this card entirely, giving the
        # tutorial the user-requested single-panel feel.
        self.info_card = QFrame()
        self.info_card.setObjectName("tutorialCard")
        info_layout = QVBoxLayout(self.info_card)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setSpacing(12)
        body_layout.addWidget(self.info_card, 5)
        # Hide by default — _apply_step_content + completion-fire
        # paths toggle visibility back on when needed.
        self.info_card.setVisible(False)

        self.practice_stack = QStackedWidget()
        # Practice stack fills the right-side info_card vertically.
        # Without Expanding policy the QStackedWidget defaults to
        # Preferred sizing, which means it shrinks to the sizeHint of
        # whichever child is active and floats up to the top of the
        # info_card — leaving empty room at the bottom and making
        # the mouse-scroll arena look pinned to the top-right.
        self.practice_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.swipe_widget = SwipeInstructionWidget()
        self.wheel_widget = WheelInstructionWidget()
        self.mouse_widget = MousePracticeWidget()
        # Scroll-phase counterpart to mouse_widget. Becomes the
        # active page in practice_stack after the user clicks all 4
        # click-phase targets. User must scroll to the top bar AND
        # the bottom bar to complete; tracked via the QScrollArea's
        # valueChanged signal in the widget itself.
        self.mouse_scroll_widget = MouseScrollPracticeWidget()
        # Big check overlay used by every step EXCEPT mouse_mode
        # (mouse paints its own ✓ inside the practice arena). Lives
        # inside practice_stack so it occupies the stretch=1 region
        # of info_layout without compressing the instruction text
        # above. Switching to this page replaces whatever was being
        # shown (or empty space) when the user completes the step.
        self.completion_overlay = QFrame()
        self.completion_overlay.setObjectName("tutorialCompletionOverlay")
        # Avoid layout-driven cropping: the overlay sets the check label
        # to fill the frame minus a reserved band for "Completed!", and
        # we rescale the glyph in resizeEvent so it never gets clipped.
        # Margins are tight (top=0, bottom=0) so the check has the full
        # frame height to work with; left/right keep a small inset so it
        # doesn't kiss the card border.
        completion_layout = QVBoxLayout(self.completion_overlay)
        completion_layout.setContentsMargins(8, 0, 8, 0)
        completion_layout.setSpacing(0)
        self.completion_overlay_check = QLabel("✓")
        self.completion_overlay_check.setObjectName("tutorialCompletionOverlayCheck")
        self.completion_overlay_check.setAlignment(Qt.AlignCenter)
        self.completion_overlay_check.setStyleSheet(
            "color: rgb(29, 233, 182); font-weight: 900;"
        )
        self.completion_overlay_check.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Minimum height of 0 lets the check shrink without forcing the
        # overlay to grow when the practice_stack region is short.
        self.completion_overlay_check.setMinimumHeight(0)
        completion_layout.addWidget(self.completion_overlay_check, 1)
        self.completion_overlay_text = QLabel("Completed!")
        self.completion_overlay_text.setObjectName("tutorialCompletionOverlayText")
        self.completion_overlay_text.setAlignment(Qt.AlignCenter)
        self.completion_overlay_text.setStyleSheet(
            "color: rgb(29, 233, 182); font-size: 28px; font-weight: 800;"
        )
        self.completion_overlay_text.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        completion_layout.addWidget(self.completion_overlay_text, 0, Qt.AlignBottom | Qt.AlignHCenter)

        # Auto-fit the check glyph to whatever vertical space the
        # overlay has, minus the reserved bottom band for "Completed!".
        # Without this the 200 px check gets clipped on short dialogs
        # (especially since steps with long inline instructions now
        # squeeze the practice_stack region). Letting the check overlap
        # surrounding UI is acceptable — but it must never overlap the
        # "Completed!" word below it, which is why we subtract its
        # sizeHint from the available height before rescaling.
        def _rescale_completion_check(event, _frame=self.completion_overlay,
                                       _check=self.completion_overlay_check,
                                       _text=self.completion_overlay_text):
            QFrame.resizeEvent(_frame, event)
            reserved = _text.sizeHint().height() + 4  # 4 px breathing gap
            target = max(40, _frame.height() - reserved - 4)
            f = _check.font()
            f.setPixelSize(int(target))
            _check.setFont(f)
        self.completion_overlay.resizeEvent = _rescale_completion_check
        self.generic_practice = QLabel("")
        self.generic_practice.setObjectName("tutorialPracticeLabel")
        self.generic_practice.setWordWrap(True)
        self.generic_practice.setAlignment(Qt.AlignCenter)
        self.practice_stack.addWidget(self.swipe_widget)
        self.practice_stack.addWidget(self.generic_practice)
        self.practice_stack.addWidget(self.wheel_widget)
        self.practice_stack.addWidget(self.mouse_widget)
        self.practice_stack.addWidget(self.mouse_scroll_widget)
        self.practice_stack.addWidget(self.completion_overlay)
        # High stretch factor (99 vs. the trailing addStretch(1)
        # below) so the practice_stack absorbs essentially all of the
        # info_card's vertical room. Before this the QVBoxLayout's
        # trailing stretch claimed an equal share, pinning the
        # practice arena to the top half of the right panel and
        # leaving a dead band underneath. The other widgets (progress
        # label / completion labels / etc.) are hidden on every step
        # so giving practice_stack near-100% of the stretch budget
        # is safe.
        info_layout.addWidget(self.practice_stack, 99)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("tutorialProgress")
        self.progress_label.setWordWrap(True)
        info_layout.addWidget(self.progress_label)

        self.low_fps_warning_label = QLabel(
            "If swipes aren't being detected, your camera may be running at low FPS. "
            "Try turning on Low FPS Mode in Settings → Camera."
        )
        self.low_fps_warning_label.setObjectName("tutorialLowFpsWarning")
        self.low_fps_warning_label.setWordWrap(True)
        self.low_fps_warning_label.setStyleSheet(
            "background-color: rgba(255, 193, 7, 0.18);"
            "color: #FFC107;"
            "border: 1px solid #FFC107;"
            "border-radius: 8px;"
            "padding: 8px 12px;"
            "font-weight: 700;"
        )
        self.low_fps_warning_label.hide()
        info_layout.addWidget(self.low_fps_warning_label)
        self._low_fps_warning_since: float | None = None

        self.completion_check_label = QLabel("✓")
        self.completion_check_label.setObjectName("tutorialCompletionCheck")
        self.completion_check_label.setAlignment(Qt.AlignCenter)
        self.completion_check_label.hide()
        info_layout.addWidget(self.completion_check_label)

        self.completion_text_label = QLabel("Completed!")
        self.completion_text_label.setObjectName("tutorialCompletionText")
        self.completion_text_label.setAlignment(Qt.AlignCenter)
        self.completion_text_label.hide()
        info_layout.addWidget(self.completion_text_label)

        self.note_label = QLabel("")
        self.note_label.setObjectName("tutorialNote")
        self.note_label.setWordWrap(True)
        info_layout.addWidget(self.note_label)

        self.voice_preview_label = QLabel("")
        self.voice_preview_label.setObjectName("tutorialVoicePreview")
        self.voice_preview_label.setWordWrap(True)
        self.note_label.hide()
        self.voice_preview_label.hide()
        info_layout.addWidget(self.voice_preview_label)
        info_layout.addStretch(1)

        self.body_stack.addWidget(self._build_completion_guide_page())

        footer = QFrame()
        footer.setObjectName("tutorialFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(18, 14, 18, 14)
        footer_layout.setSpacing(10)

        self.guide_button = QPushButton("Open Control Guide")
        self.guide_button.clicked.connect(self._open_guide)
        self.leave_button = QPushButton("Leave Tutorial")
        self.leave_button.clicked.connect(lambda: self._finish_and_close(False))
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self._go_previous)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self._go_next)

        footer_layout.addWidget(self.guide_button)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.leave_button)
        footer_layout.addWidget(self.prev_button)
        footer_layout.addWidget(self.next_button)
        root.addWidget(footer)

        # Click-through overlay that draws the bouncing arrow toward
        # the mic during the voice-command practice step. Created
        # last so it lands on top of every other child by default;
        # show_pointing also calls raise_() to be safe across late
        # widget additions. Initial geometry is set by the first
        # resizeEvent that fires after show() — we don't size it
        # here because the dialog hasn't been laid out yet.
        accent = QColor(self.config.accent_color or "#1DE9B6")
        if not accent.isValid():
            accent = QColor("#1DE9B6")
        self._voice_mic_arrow = _VoiceMicArrow(self, accent)
        self._step_example_dialog: QDialog | None = None

    def _apply_example_dialog_theme(self, dialog: QDialog) -> None:
        dialog.setStyleSheet(
            self.styleSheet()
            + f"""
            QScrollArea#tutorialExampleScroll,
            QScrollArea#tutorialExampleScroll > QWidget,
            QScrollArea#tutorialExampleScroll QWidget#qt_scrollarea_viewport,
            QWidget#tutorialExampleContent {{
                background: transparent;
                border: none;
            }}
            QFrame#innerCard {{
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(29,233,182,0.22);
                border-radius: 18px;
            }}
            QFrame#innerCard QLabel {{
                color: {self.config.text_color};
                background: transparent;
            }}
            QLabel#gestureCardTitle {{
                color: {self.config.accent_color};
                font-size: 18px;
                font-weight: 900;
            }}
            QLabel#gestureCardSubtitle {{
                color: {self.config.text_color};
                font-size: 13px;
                font-weight: 800;
            }}
            QLabel#gestureCardBody {{
                color: {self.config.text_color};
                font-size: 13px;
            }}
            QScrollBar:vertical {{
                background: rgba(255,255,255,0.06);
                width: 14px;
                border-radius: 7px;
                margin: 2px 0;
            }}
            QScrollBar::handle:vertical {{
                background: {self.config.accent_color};
                min-height: 36px;
                border-radius: 7px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            """
        )

    @staticmethod
    def _guide_card_title(card: QWidget) -> str:
        title_label = card.findChild(QLabel, "gestureCardTitle")
        if title_label is None:
            return ""
        return str(title_label.text() or "").strip()

    def _close_step_example_dialog(self) -> None:
        dialog = getattr(self, "_step_example_dialog", None)
        if dialog is None:
            return
        self._step_example_dialog = None
        try:
            dialog.close()
        except Exception:
            pass

    @staticmethod
    def _voice_command_hold_warning_text() -> str:
        return (
            "You can drop your hand now. If you keep Left Hand One up you will "
            "trigger voice listening when current command is processed."
        )

    def _voice_command_intro_header_text(self) -> str:
        # Branch on Spotify detection (same flag the rest of the
        # voice step + play/pause uses) so the spoken example
        # matches what the completion check actually verifies.
        target = "Spotify" if getattr(self, "_has_spotify", False) else "YouTube"
        return (
            "Hold left-hand one until the microphone appears below, "
            f"then say: “Play [a song] on {target}”."
        )

    def _voice_command_header_text(self, hold_active: bool) -> str:
        if hold_active and not self._step_completed:
            return self._voice_command_hold_warning_text()
        return self._voice_command_intro_header_text()

    def _voice_command_progress_text(self, *, left_one_active: bool, voice_listening: bool) -> str:
        if self._step_completed:
            return "Completed! Swipe right to move on!"
        if voice_listening:
            return "Listening for your voice command..."
        if left_one_active:
            return "Detected left-hand one!"
        return "Waiting for left-hand one and the voice command."

    def _mouse_mode_header_text(
        self,
        *,
        mode_enabled: bool,
        cursor_seen: bool | None = None,
        completed_targets: int | None = None,
    ) -> str:
        # Progressive header for the mouse step. Each state corresponds
        # to the next action the user should take, so reading the text
        # alone walks them through the whole flow without needing the
        # side panel.
        seen = self._mouse_cursor_seen if cursor_seen is None else bool(cursor_seen)
        done = self.mouse_widget.completed_targets if completed_targets is None else int(completed_targets)
        if self._step_completed:
            return "Swipe right to continue!"
        # Three post-click stages mapped onto distinct headers:
        #   "scroll"  — clicks done, waiting on scroll completion
        #   "disable" — scroll done, waiting on left-hand-three to
        #               toggle mouse mode off
        # "scroll" must check before the broader `done >= 4` clause
        # so the scroll-pose hint isn't masked by the disable hint.
        if self._mouse_stage == "scroll":
            return "Use your middle and index fingers together to scroll!"
        if done >= 4 or self._mouse_stage == "disable":
            return "Use left hand three to turn off mouse mode!"
        if done >= 3:
            return "Now click on target 4"
        if done >= 2:
            return "Now click on target 3"
        if done >= 1:
            return "Now click on target 2"
        if mode_enabled:
            if seen:
                return (
                    "Now left click on target 1 by pinching index finger "
                    "and thumb together briefly"
                )
            return (
                "With right hand open facing towards the screen, "
                "move the cursor around"
            )
        return "Toggle mouse mode with left hand three"

    def _step_example_spec(self, step_key: str) -> tuple[str, tuple[str, ...]]:
        specs = {
            "swipes": (
                "These are the Control Guide examples for the two swipe directions used in this part.",
                ("Swipe Right", "Swipe Left"),
            ),
            "spotify_open": (
                "This is the Control Guide example for the pose that opens or focuses Spotify.",
                ("Right Hand Two",),
            ),
            "play_pause": (
                "This is the Control Guide example for the play or pause pose used in this part.",
                ("Right Hand Fist",),
            ),
            "volume": (
                "These are the Control Guide examples for the volume pose (raise / lower) and the "
                "shaka mute pose used in this part.",
                ("Volume Control", "Mute"),
            ),
            "gesture_wheel": (
                "This is the Control Guide example for the gesture wheel pose used in this part.",
                ("Gesture Wheel",),
            ),
            "mouse_mode": (
                "These Control Guide examples cover the left-hand toggle pose and the right-hand mouse actions "
                "(pinch clicks, scroll, full demo) used in this part.",
                ("Left Hand Three", "Mouse Clicks", "Mouse Scroll", "Mouse Demo"),
            ),
            "voice_command": (
                "These Control Guide examples show the trigger pose and the voice-listening behavior used in this part.",
                ("Left Hand One", "Triggering a voice command"),
            ),
        }
        return specs.get(
            step_key,
            ("Relevant Control Guide examples for this tutorial part.", tuple()),
        )

    def _build_step_example_cards(self, step_key: str) -> tuple[str, list[QWidget]]:
        from .main_window import (
            _build_gesture_guide_dynamic_cards,
            _build_gesture_guide_static_cards,
            _build_voice_command_cards,
        )

        intro_text, wanted_titles = self._step_example_spec(step_key)
        all_cards = [
            *_build_gesture_guide_static_cards(),
            *_build_gesture_guide_dynamic_cards(),
            *_build_voice_command_cards(),
        ]
        cards_by_title: dict[str, QWidget] = {}
        for card in all_cards:
            card_title = self._guide_card_title(card)
            if card_title and card_title not in cards_by_title:
                cards_by_title[card_title] = card

        selected_cards = [cards_by_title[title] for title in wanted_titles if title in cards_by_title]
        if not selected_cards:
            selected_cards = all_cards

        for card in all_cards:
            if card not in selected_cards:
                card.setParent(None)
                card.deleteLater()

        return intro_text, selected_cards

    def _open_spotify_setup_from_tutorial(self) -> None:
        """Open the Spotify setup wizard from the tutorial's
        voice_command step. This is a shortcut into the same flow the
        Set up your own Spotify app button in Settings → General runs,
        so users discovering "play X on Spotify" needs authorisation
        can fix it without leaving the tutorial."""
        try:
            from .spotify_setup_wizard import SpotifySetupWizard
        except Exception:
            return
        try:
            dialog = SpotifySetupWizard(self.config, parent=self)
            dialog.exec()
        except Exception:
            pass

    def _open_full_instructions_dialog(self) -> None:
        """Pop the current step's full instruction text into a tall
        modal dialog so users can read every step without fighting the
        inline scrollbox. Triggered by the "Show full instructions"
        button next to "Show Example". The dialog shows whatever
        text the inline instruction_box currently holds (so it tracks
        per-step content automatically) — no separate copy to maintain.
        """
        if self._show_completion_page:
            return
        try:
            step = self._practice_steps[self._step_index]
            title = step.title
        except Exception:
            title = "Instructions"
        instructions = ""
        try:
            instructions = self.instruction_box.text() or ""
        except Exception:
            pass
        if not instructions.strip():
            instructions = "No detailed instructions available for this step."
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{title} — Full Instructions")
        dialog.setModal(True)
        dialog.resize(680, 520)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        header = QLabel(f"{title} — Full Instructions")
        header.setStyleSheet(
            f"color: {self.config.accent_color or '#1DE9B6'}; "
            "font-size: 18px; font-weight: 700;"
        )
        layout.addWidget(header)
        body = QLabel(instructions)
        body.setWordWrap(True)
        body.setTextFormat(Qt.PlainText)
        body.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        body.setStyleSheet(
            "color: rgba(232,246,255,0.95); font-size: 14px; "
            "line-height: 160%; padding: 8px;"
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(body)
        scroll.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(scroll, 1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet(
            f"QPushButton {{ background: {self.config.accent_color or '#1DE9B6'}; "
            "color: #0a162b; border: none; border-radius: 8px; "
            "padding: 8px 18px; font-weight: 700; }}"
        )
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        dialog.exec()

    def _on_step_desc_link(self, link: str) -> None:
        """Route 'Show more…' clicks in step_desc to the existing
        step-example dialog. Same dialog the example button + mouse
        instructions button open — keeps one source of truth for the
        full instruction text per step."""
        if str(link or "").strip() == "show_more":
            self._open_step_example()

    def _open_step_example(self) -> None:
        if self._show_completion_page:
            return

        step = self._practice_steps[self._step_index]
        existing = getattr(self, "_step_example_dialog", None)
        if existing is not None and existing.isVisible() and existing.property("tutorialStepKey") == step.key:
            existing.raise_()
            existing.activateWindow()
            return

        self._close_step_example_dialog()
        intro_text, cards = self._build_step_example_cards(step.key)

        dialog = QDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.setModal(False)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setProperty("tutorialStepKey", step.key)
        dialog.setWindowTitle(f"{step.title} Example")
        dialog.resize(960, 740)
        # r51: install_indigo_chrome for Win10 + Win11 parity (was
        # apply_touchless_chrome, Win11 only).
        try:
            from .window_chrome import install_indigo_chrome
            example_body = install_indigo_chrome(dialog, f"{step.title} Example")
        except Exception:
            example_body = dialog
        self._apply_example_dialog_theme(dialog)

        outer = QVBoxLayout(example_body)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(12)

        card = QFrame()
        card.setObjectName("tutorialCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(12)

        title_label = QLabel(f"{step.title} Examples")
        title_label.setObjectName("tutorialStepTitle")
        title_label.setWordWrap(True)
        card_layout.addWidget(title_label)

        intro_label = QLabel(intro_text)
        intro_label.setObjectName("tutorialInstructionBox")
        intro_label.setWordWrap(True)
        intro_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        card_layout.addWidget(intro_label)

        # Detailed step instructions — same text the inline instruction
        # box on the main tutorial page shows. Folded into the Example
        # dialog so users get the full how-to + examples in one window
        # instead of clicking through two separate buttons.
        try:
            detailed_text = self.instruction_box.text() or ""
        except Exception:
            detailed_text = ""
        if detailed_text.strip():
            details_header = QLabel("Detailed instructions")
            details_header.setStyleSheet(
                f"color: {self.config.accent_color or '#1DE9B6'}; "
                "font-size: 14px; font-weight: 700; padding-top: 4px;"
            )
            card_layout.addWidget(details_header)
            details_label = QLabel(detailed_text)
            details_label.setWordWrap(True)
            details_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            details_label.setStyleSheet(
                "color: rgba(232,246,255,0.92); font-size: 13px; "
                "line-height: 155%;"
            )
            card_layout.addWidget(details_label)

        scroll = QScrollArea()
        scroll.setObjectName("tutorialExampleScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content.setObjectName("tutorialExampleContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        for guide_card in cards:
            # Per the tutorial-popup UX request, scale every media
            # widget (gesture image / clip / sketch) by 1.3× so the
            # examples are easier to see at a glance. set_scale_factor
            # operates on the original-resolution source pixmap so the
            # result stays crisp rather than blurry.
            try:
                from .main_window import GestureMediaWidget as _GMW
                for media in guide_card.findChildren(_GMW):
                    try:
                        media.set_scale_factor(1.3)
                    except Exception:
                        pass
            except Exception:
                pass
            content_layout.addWidget(guide_card)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        card_layout.addWidget(scroll, 1)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        button_row.addWidget(close_button)
        card_layout.addLayout(button_row)

        outer.addWidget(card, 1)

        dialog.destroyed.connect(lambda *_: setattr(self, "_step_example_dialog", None))
        self._step_example_dialog = dialog
        dialog.move(self.x() + 36, self.y() + 36)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _toggle_completion_more_tips(self) -> None:
        """Flip the secondary tips container between hidden and shown.
        Button label tracks state: 'More tips (N) …' when collapsed,
        'Fewer tips' when expanded."""
        container = getattr(self, "_completion_more_tips_container", None)
        button = getattr(self, "_completion_more_tips_button", None)
        if container is None or button is None:
            return
        now_visible = not container.isVisible()
        container.setVisible(now_visible)
        if now_visible:
            button.setText("Fewer tips")
        else:
            n = getattr(self, "_completion_more_tips_secondary_count", 0)
            button.setText(f"More tips ({n}) …")

    def _build_completion_guide_page(self) -> QWidget:
        """End-of-tutorial page. Title + a 'Quick Tips' card with
        discoverability hints the user wouldn't otherwise stumble
        across (where Connect Spotify lives, how to pause gestures
        without quitting, etc.), then the full gesture-guide
        widget below for reference. Renamed from a plain 'Tutorial
        Completed' confirmation to a tips page because first-pass
        users land here once and the gesture grid alone wasn't
        surfacing the useful 'oh, that's where THAT setting lives'
        moments.

        The whole page is wrapped in a QScrollArea so a short
        window doesn't clip the tips card or the gesture grid;
        the user can scroll through everything end-to-end —
        including expanded Static / Dynamic / Voice dropdowns
        that grow the gesture-grid section vertically. We use
        `build_gesture_guide_content_widget()` (raw widget, no
        inner scroll area) rather than the scroll-area variant so
        the outer page scrollbar can take over for the entire
        height instead of fighting an inner scrollbar that would
        only scroll the grid."""
        from .main_window import build_gesture_guide_content_widget

        outer = QWidget()
        outer.setObjectName("tutorialCompletionPage")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        completion_scroll = QScrollArea()
        completion_scroll.setObjectName("tutorialCompletionScroll")
        completion_scroll.setWidgetResizable(True)
        completion_scroll.setFrameShape(QFrame.NoFrame)
        completion_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        completion_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        completion_scroll.setStyleSheet(
            "QScrollArea#tutorialCompletionScroll, "
            "QScrollArea#tutorialCompletionScroll > QWidget, "
            "QScrollArea#tutorialCompletionScroll QWidget#qt_scrollarea_viewport "
            "{ background: transparent; border: none; }"
        )
        completion_scroll.viewport().setStyleSheet("background: transparent;")
        outer_layout.addWidget(completion_scroll, 1)

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(18, 18, 18, 18)
        page_layout.setSpacing(14)
        completion_scroll.setWidget(page)

        self.completion_title_label = QLabel("Tutorial Completed — Quick Tips")
        self.completion_title_label.setObjectName("tutorialStepTitle")
        page_layout.addWidget(self.completion_title_label)

        self.completion_disclaimer_label = QLabel(
            "Some things that aren't obvious from the UI. The full "
            "gesture reference is below.\n\n"
            "Swipe RIGHT to finish the tutorial, or swipe LEFT to "
            "go back to the previous step."
        )
        self.completion_disclaimer_label.setObjectName("tutorialInstructionBox")
        self.completion_disclaimer_label.setWordWrap(True)
        self.completion_disclaimer_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        page_layout.addWidget(self.completion_disclaimer_label)

        # ---- Tips card -----------------------------------------
        # The first three tips are the highest-leverage discoverability
        # hints (pause without quitting, connect Spotify, adjust mouse
        # sensitivity). The remaining five are hidden behind a "More
        # tips" toggle so the completion page reads as a quick three-
        # item primer; users who want the deep list expand it.
        primary_tips = [
            (
                "Connect Spotify",
                "Settings → General → Spotify → Set up your own Spotify app "
                "(one-time, ~1 minute), then click Connect Spotify. After that, "
                "voice and gesture controls for Spotify just work — re-click "
                "Connect anytime you need to re-authorise.",
            ),
            (
                "Pause gestures without quitting",
                "Open the Live View widget (the small camera thumbnail) and "
                "toggle gestures off. The app keeps running and you can flip "
                "them back on the same way.",
            ),
            (
                "App Performance",
                "If gestures feel slow or drop frames, Settings → General has "
                "four toggles that trade a feature or a bit of fidelity for "
                "speed. Tap Show more to see what each one does.",
                (
                    "<p style='margin:0 0 6px 0;'><b>Disable Clip Cache</b> "
                    "(Settings → General → Clip Presets) &mdash; stops the "
                    "rolling clip recorder from running in the background. "
                    "Saves CPU and disk if you never use &lsquo;clip that&rsquo;.</p>"
                    "<p style='margin:0 0 6px 0;'><b>Low FPS Mode</b> "
                    "(Settings → General → System Modes) &mdash; loosens "
                    "tracking thresholds so gestures still register when the "
                    "camera runs slow (around 10-17 FPS). Touchless can also "
                    "offer to turn this on automatically if your measured FPS "
                    "stays low for too long.</p>"
                    "<p style='margin:0 0 6px 0;'><b>Lite Mode</b> "
                    "(Settings → General → System Modes) &mdash; improves "
                    "processing by about 2.5&times; for simple gestures and "
                    "commands. For very extreme angles or heavy occlusion it "
                    "may be slightly less stable. A &lsquo;Lite&rsquo; badge "
                    "appears in live viewers when active.</p>"
                    "<p style='margin:0;'><b>GPU Mode</b> "
                    "(Settings → General → System Modes) &mdash; if your "
                    "machine can run it, Touchless uses the graphics card to "
                    "speed up hand tracking. If not, Touchless quietly falls "
                    "back to the regular path so gestures keep working.</p>"
                ),
            ),
            (
                "Adjust mouse sensitivity",
                "Settings → General → Mouse. Smaller control box = a tiny hand "
                "movement covers the whole screen (high sensitivity); larger "
                "box = more precision.",
            ),
        ]
        secondary_tips = [
            (
                "Use your phone as the camera",
                "Settings → Camera → scan the QR code with your phone. Streams "
                "video (and optionally the phone microphone) over your local "
                "Wi-Fi.",
            ),
            (
                "Save a clip of what just happened",
                "Hold left-hand 'one' to start the voice listener, then say "
                "“clip that” — saves the last minute to your Clips folder.",
            ),
            (
                "Auto-hide overlays while gaming",
                "Settings → General → Overlay → enable Gaming Overlay. Touchless "
                "auto-suppresses pop-ups when a known game is detected. Set "
                "games to Windowed Fullscreen / Borderless so the overlay can "
                "appear over them.",
            ),
            (
                "If gestures aren't detecting reliably",
                "Settings → General → System Modes → try Lite Mode or Low FPS "
                "Mode. They loosen tracking thresholds for slower hardware.",
            ),
            (
                "Replay this tutorial anytime",
                "You can come back to this tour whenever you want — Settings → "
                "Tutorial.",
            ),
        ]
        tips_card = QFrame()
        tips_card.setObjectName("tutorialCard")
        tips_card_layout = QVBoxLayout(tips_card)
        tips_card_layout.setContentsMargins(18, 18, 18, 18)
        tips_card_layout.setSpacing(0)
        accent = self.config.accent_color or "#1DE9B6"
        text_color = self.config.text_color or "#E5F6FF"
        divider_color = "rgba(29,233,182,0.18)"

        def _add_tip(parent_layout, headline, body, is_first, details=None):
            # Soft horizontal divider between tips so the eye can find
            # each tip without straining.
            if not is_first:
                divider = QFrame()
                divider.setFrameShape(QFrame.HLine)
                divider.setStyleSheet(
                    f"QFrame {{ background: {divider_color}; "
                    f"border: none; max-height: 1px; min-height: 1px; "
                    f"margin: 12px 0 12px 0; }}"
                )
                parent_layout.addWidget(divider)
            head_label = QLabel(headline)
            head_label.setWordWrap(True)
            head_label.setStyleSheet(
                f"color: {accent}; font-size: 15px; font-weight: 800; "
                f"letter-spacing: 0.1px; background: transparent;"
            )
            body_label = QLabel(body)
            body_label.setWordWrap(True)
            body_label.setStyleSheet(
                f"color: {text_color}; font-size: 13px; line-height: 150%; "
                f"background: transparent; margin-top: 4px;"
            )
            parent_layout.addWidget(head_label)
            parent_layout.addWidget(body_label)
            # v1.1.7: optional inline "Show more..." expander for tips
            # that carry a longer explanation. Mirrors the settings-
            # panel expander pattern inline (no cross-module import).
            if details:
                toggle_row = QHBoxLayout()
                toggle_row.setContentsMargins(0, 4, 0, 0)
                toggle_row.addStretch(1)
                more_btn = QPushButton("Show more...")
                more_btn.setCursor(Qt.PointingHandCursor)
                more_btn.setFlat(True)
                more_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                more_btn.setStyleSheet(
                    f"QPushButton {{ color: {accent}; "
                    f"background: transparent; border: none; "
                    f"padding: 0; font-size: 12px; font-weight: 700; "
                    f"text-decoration: underline; }}"
                    f"QPushButton:hover {{ color: {accent}; }}"
                )
                toggle_row.addWidget(more_btn)
                parent_layout.addLayout(toggle_row)
                detail_label = QLabel(details)
                detail_label.setTextFormat(Qt.RichText)
                detail_label.setWordWrap(True)
                detail_label.setStyleSheet(
                    f"color: {text_color}; font-size: 13px; "
                    f"line-height: 150%; background: transparent; "
                    f"margin-top: 6px;"
                )
                detail_label.hide()
                parent_layout.addWidget(detail_label)

                def _toggle_details():
                    expanded = not detail_label.isVisible()
                    detail_label.setVisible(expanded)
                    more_btn.setText("Show less..." if expanded else "Show more...")

                more_btn.clicked.connect(_toggle_details)

        for i, tip in enumerate(primary_tips):
            headline, body = tip[0], tip[1]
            details = tip[2] if len(tip) > 2 else None
            _add_tip(
                tips_card_layout, headline, body,
                is_first=(i == 0), details=details,
            )

        # Hidden-by-default container for secondary tips. Toggled by
        # the "More tips" / "Fewer tips" link below the card.
        self._completion_more_tips_container = QWidget()
        more_layout = QVBoxLayout(self._completion_more_tips_container)
        more_layout.setContentsMargins(0, 0, 0, 0)
        more_layout.setSpacing(0)
        for tip in secondary_tips:
            headline, body = tip[0], tip[1]
            details = tip[2] if len(tip) > 2 else None
            _add_tip(
                more_layout, headline, body,
                is_first=False, details=details,
            )
        self._completion_more_tips_container.setVisible(False)
        tips_card_layout.addWidget(self._completion_more_tips_container)

        page_layout.addWidget(tips_card)

        # Centered "More tips ..." toggle directly under the card.
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.addStretch(1)
        self._completion_more_tips_button = QPushButton(
            f"More tips ({len(secondary_tips)}) …"
        )
        self._completion_more_tips_button.setObjectName("completionMoreTips")
        self._completion_more_tips_button.setCursor(Qt.PointingHandCursor)
        # WA_StyledBackground forces Qt to use the QSS-painted
        # background instead of the platform's native button look,
        # which on Windows draws square-cornered chrome OVER the
        # QSS border-radius and made this button read as sharp box
        # edges even though border-radius was set.
        self._completion_more_tips_button.setAttribute(
            Qt.WA_StyledBackground, True
        )
        self._completion_more_tips_button.setStyleSheet(
            f"QPushButton#completionMoreTips {{ "
            f"  color: {accent}; "
            f"  background-color: transparent; "
            f"  border: 1.5px solid rgba(29,233,182,0.45); "
            # Half the button height ≈ pill cap. 18 px matches the
            # natural button height with the 8/22 padding below.
            f"  border-radius: 18px; "
            f"  padding: 8px 22px; "
            f"  font-size: 13px; "
            f"  font-weight: 700; "
            f"  outline: none; "
            f"}}"
            f"QPushButton#completionMoreTips:hover {{ "
            f"  border: 1.5px solid {accent}; "
            f"  background-color: rgba(29,233,182,0.10); "
            f"}}"
            f"QPushButton#completionMoreTips:pressed {{ "
            f"  background-color: rgba(29,233,182,0.16); "
            f"}}"
        )
        self._completion_more_tips_button.clicked.connect(
            self._toggle_completion_more_tips
        )
        self._completion_more_tips_secondary_count = len(secondary_tips)
        toggle_row.addWidget(self._completion_more_tips_button)
        toggle_row.addStretch(1)
        page_layout.addLayout(toggle_row)

        # ---- Gesture reference --------------------------------
        # Use the raw content widget (no inner scroll area). The
        # outer completion_scroll already provides scrolling, so
        # nesting another scroll area inside would trap the
        # dropdown expansions to a small inner viewport instead of
        # letting the page grow naturally as the user opens
        # Static / Dynamic / Voice. SizePolicy is set to
        # MinimumExpanding so the page can grow vertically with
        # the dropdowns without forcing the rest of the layout to
        # shrink.
        guide_card = QFrame()
        guide_card.setObjectName("tutorialCard")
        guide_card_layout = QVBoxLayout(guide_card)
        guide_card_layout.setContentsMargins(14, 14, 14, 14)
        guide_card_layout.setSpacing(10)
        guide_content = build_gesture_guide_content_widget()
        guide_content.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.MinimumExpanding
        )
        guide_card_layout.addWidget(guide_content)
        page_layout.addWidget(guide_card)
        page_layout.addStretch(0)
        return outer

    def apply_theme(self, config: AppConfig) -> None:
        self.config = config
        self.swipe_widget.apply_theme(config.accent_color, config.text_color)
        self.wheel_widget.apply_theme(config.accent_color, config.text_color)
        self.mouse_widget.apply_theme(config.accent_color, config.text_color)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {self.config.surface_color};
                color: {self.config.text_color};
            }}
            QFrame#tutorialHeader {{
                background-color: rgba(255,255,255,0.03);
                border-bottom: 1px solid rgba(29,233,182,0.22);
            }}
            QFrame#tutorialCard, QFrame#tutorialFooter {{
                background-color: rgba(255,255,255,0.04);
                border: 1px solid rgba(29,233,182,0.22);
                border-radius: 18px;
            }}
            QLabel#tutorialHero {{
                color: {self.config.accent_color};
                font-size: 28px;
                font-weight: 800;
                letter-spacing: -0.3px;
            }}
            QLabel#tutorialSubtitle {{
                color: {self.config.text_color};
                font-size: 16px;
                font-weight: 400;
                line-height: 150%;
                letter-spacing: 0.1px;
            }}
            QLabel#tutorialBadge {{
                background-color: rgba(9,42,58,0.92);
                color: {self.config.accent_color};
                border-radius: 13px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }}
            QLabel#tutorialMeta, QLabel#tutorialProgress, QLabel#tutorialVoicePreview {{
                color: {self.config.text_color};
                background: transparent;
            }}
            QLabel#tutorialStepTitle {{
                color: {self.config.accent_color};
                font-size: 22px;
                font-weight: 800;
                letter-spacing: -0.2px;
            }}
            QLabel#tutorialCameraHeader, QLabel#tutorialCameraFooter {{
                color: {self.config.accent_color};
                font-size: 20px;
                font-weight: 800;
                letter-spacing: -0.1px;
                background: transparent;
                padding: 6px 4px;
            }}
            QLabel#tutorialStepDesc {{
                color: {self.config.text_color};
                font-size: 14px;
                line-height: 150%;
            }}
            QLabel#tutorialInstructionBox {{
                background-color: rgba(255,255,255,0.04);
                border: 1px solid rgba(29,233,182,0.18);
                border-radius: 16px;
                color: {self.config.text_color};
                padding: 16px 18px;
                font-size: 14px;
                line-height: 150%;
            }}
            QLabel#tutorialVideo {{
                background-color: rgba(0,0,0,0.16);
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,0.08);
                color: {self.config.text_color};
                padding: 14px;
            }}
            QLabel#tutorialChip {{
                background-color: rgba(9,42,58,0.92);
                color: {self.config.accent_color};
                border-radius: 12px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }}
            QLabel#tutorialPracticeLabel {{
                background-color: rgba(255,255,255,0.04);
                border: 1px solid rgba(29,233,182,0.18);
                border-radius: 16px;
                color: {self.config.text_color};
                padding: 18px;
                font-size: 15px;
                line-height: 150%;
            }}
            QLabel#tutorialNote {{
                color: {self.config.accent_color};
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }}
            QPushButton {{
                background-color: {self.config.primary_color};
                color: {self.config.text_color};
                border: 1px solid rgba(29,233,182,0.35);
                border-radius: 12px;
                padding: 10px 16px;
                font-weight: 800;
            }}
            QPushButton:hover {{
                border: 1px solid {self.config.accent_color};
            }}
            QPushButton:pressed {{
                background-color: {self.config.accent_color};
                color: #001B24;
                border: 1px solid {self.config.accent_color};
            }}
            QPushButton:disabled {{
                color: rgba(229,246,255,0.45);
                border: 1px solid rgba(255,255,255,0.12);
                background-color: rgba(255,255,255,0.05);
            }}
            """
        )
        dialog = getattr(self, "_step_example_dialog", None)
        if dialog is not None:
            self._apply_example_dialog_theme(dialog)

    def _reset_tutorial_progress(self) -> None:
        self._completed_steps.clear()
        self._show_completion_page = False
        self._swipe_counts = {"swipe_left": 0, "swipe_right": 0}
        self._swipe_last_advance_at = 0.0
        self._swipe_goal_index = 0
        self._spotify_toggle_count = 0
        self._mouse_stage = "enable"
        self._step_completed = False
        self._completion_feedback_until = 0.0
        self._completion_feedback_step = -1
        self._play_pause_ready_for_next = True

    def _seed_prior_step_completion_state(self, start_step_index: int) -> None:
        clamped = max(0, min(int(start_step_index), len(self._practice_steps) - 1))
        self._completed_steps = set(range(clamped))
        if clamped > 0:
            self._swipe_counts = {"swipe_left": 3, "swipe_right": 3}
            self._swipe_goal_index = 6
        if clamped > 2:
            self._spotify_toggle_count = 2
        if clamped > 4:
            self._mouse_stage = "disable"
            self._mouse_cursor_seen = True

    def configure_session(
        self,
        *,
        camera_index: Optional[int],
        launched_from_settings: bool,
        auto_start_on_done: bool,
        start_step_index: int = 0,
    ) -> None:
        self._camera_index = camera_index
        self._launched_from_settings = bool(launched_from_settings)
        self._auto_start_on_done = bool(auto_start_on_done)
        self._reset_tutorial_progress()
        clamped_step_index = max(0, min(int(start_step_index), len(self._practice_steps) - 1))
        self._step_index = clamped_step_index
        self._seed_prior_step_completion_state(clamped_step_index)
        self._close_emitted = False
        self._closing_programmatically = False
        self._reset_for_step()
        self._start_session()

    def _resolve_parent_worker(self) -> GestureWorker | None:
        parent = self.parent()
        worker = getattr(parent, "_worker", None)
        if worker is not None and getattr(worker, "is_running", False):
            return worker
        return None

    def _connect_worker(self, worker: GestureWorker) -> None:
        if self._shared_worker_connected and self._worker is worker:
            return
        self._disconnect_worker()
        self._worker = worker
        try:
            worker.debug_frame_ready.connect(self._on_worker_debug_frame)
            worker.command_detected.connect(self._on_worker_command_detected)
            worker.camera_selected.connect(self._on_worker_camera_selected)
            worker.running_state_changed.connect(self._on_worker_running_state_changed)
            worker.error_occurred.connect(self._on_worker_error)
            self._shared_worker_connected = True
            self._sync_worker_tutorial_context()
        except Exception:
            self._worker = None
            self._shared_worker_connected = False

    def _disconnect_worker(self) -> None:
        if self._worker is not None and self._shared_worker_connected:
            try:
                self._worker.set_tutorial_context(False)
            except Exception:
                pass
            try:
                self._worker.debug_frame_ready.disconnect(self._on_worker_debug_frame)
            except Exception:
                pass
            try:
                self._worker.command_detected.disconnect(self._on_worker_command_detected)
            except Exception:
                pass
            try:
                self._worker.camera_selected.disconnect(self._on_worker_camera_selected)
            except Exception:
                pass
            try:
                self._worker.running_state_changed.disconnect(self._on_worker_running_state_changed)
            except Exception:
                pass
            try:
                self._worker.error_occurred.disconnect(self._on_worker_error)
            except Exception:
                pass
        self._shared_worker_connected = False
        self._worker = None

    def _sync_worker_tutorial_context(self) -> None:
        if self._worker is None:
            return
        tutorial_key = self._practice_steps[self._step_index].key
        if self._show_completion_page:
            tutorial_key = "voice_command"
        try:
            self._worker.set_tutorial_context(True, tutorial_key)
        except Exception:
            pass

    def _start_session(self) -> None:
        self._stop_session()
        self._show_starting_pill("Starting tutorial")
        shared_worker = self._resolve_parent_worker()
        if shared_worker is not None:
            self._owns_worker = False
            self._connect_worker(shared_worker)
            camera_info = getattr(shared_worker, "_camera_info", None)
            if camera_info is not None:
                self.camera_label.setText(f"Camera: {camera_info.display_name}")
            else:
                self.camera_label.setText("Camera: shared app session")
            self.video_label.setText("Waiting for the live app frame...")
            self.gesture_chip.setText("Gesture: neutral")
            return

        self._owns_worker = True
        # NOTE: a previous attempt at this line wrapped self.config with
        # `dataclasses.replace(..., gpu_mode=False, lite_mode=False,
        # low_fps_mode=False)` to skip the perf-mode init paths during
        # tutorial cold-start. That broke the tutorial outright on
        # configs where any of those modes were actually relied on at
        # the engine layer (the worker would init, then immediately
        # report running_state=False with a "Tutorial runtime stopped"
        # banner). Reverted until we have a safer way to get the same
        # speed-up — most likely either pre-spawning the worker before
        # the user clicks Tutorial, or wiring `set_tutorial_context`
        # to suppress the perf-mode paths from inside the engine where
        # we can guarantee state consistency. The ffmpeg-startup
        # timeout drop in ffmpeg_capture.py still applies and saves
        # ~11 s on the dead-camera path the user originally reported.
        owned_worker = GestureWorker(self.config, camera_index_override=self._camera_index, parent=self)
        # Plumb the parent app's phone-camera QR capture into the
        # owned worker BEFORE start() so phone-only users can still
        # take the tutorial. Without this, a user who has no local
        # webcam (only phone QR connected) would hit
        # open_preferred_or_first_available with no candidates and
        # the tutorial would stall on "Starting tutorial camera and
        # runtime...". The parent MainWindow holds the QR server +
        # the active flag.
        try:
            parent = self.parent()
            qr_server = getattr(parent, "_phone_camera_qr_server", None)
            qr_active = bool(getattr(self.config, "phone_camera_qr_active", False))
            if qr_server is not None and qr_active and self._camera_index is None:
                qr_capture = getattr(qr_server, "capture", None)
                if qr_capture is not None and hasattr(owned_worker, "set_phone_camera_capture"):
                    owned_worker.set_phone_camera_capture(qr_capture)
        except Exception:
            pass
        self._connect_worker(owned_worker)
        self.camera_label.setText("Camera: Starting tutorial runtime...")
        self.video_label.setText("Starting tutorial camera and runtime...")
        self.gesture_chip.setText("Gesture: Starting")
        owned_worker.start()

    def _stop_session(self) -> None:
        self._timer.stop()
        # Focus guard only ran during the voice_command step but
        # stop it defensively here too so a session that ends while
        # on that step doesn't leave the timer firing in the
        # background (the foreground check would still cost some
        # microseconds per tick even with the early-return when
        # step_key != voice_command).
        try:
            self._focus_guard_timer.stop()
        except Exception:
            pass
        owned_worker = self._worker if self._owns_worker else None
        self._disconnect_worker()
        if owned_worker is not None:
            try:
                owned_worker.stop()
            except Exception:
                pass
        if self._voice_overlay is not None:
            self._voice_overlay.hide_overlay()
        self._tutorial_wheel_overlay.hide_overlay()
        self._cap = None
        self._engine = None
        self._camera_info = None
        self._owns_worker = False

    def _voice_overlay_widget(self):
        if self._voice_overlay is None:
            from ...gesture.ui.voice_status_overlay import VoiceStatusOverlay

            self._voice_overlay = VoiceStatusOverlay(self.config)
        return self._voice_overlay

    def _on_worker_camera_selected(self, text: str) -> None:
        self.camera_label.setText(f"Camera: {text}")

    def _on_worker_running_state_changed(self, is_running: bool) -> None:
        if not is_running and self._owns_worker:
            self.video_label.setText("Tutorial runtime stopped.")
            self.gesture_chip.setText("Gesture: runtime stopped")

    def _on_worker_error(self, message: str) -> None:
        self.video_label.setText(str(message or "Tutorial runtime error"))
        self.gesture_chip.setText("Gesture: runtime error")

    def _on_worker_command_detected(self, command: str) -> None:
        text = str(command or "").strip().lower()
        if not text:
            return
        if "spotify" in text:
            self._last_spotify_tutorial_action = text
        if self._practice_steps[self._step_index].key == "voice_command":
            # The voice step's goal is simply "speak a command and have
            # it work." command_detected fires with the executed action's
            # status text, so ANY of these success markers means the
            # user's spoken command did something — complete the step.
            # Previously this was gated on _has_spotify and the Spotify
            # completion branch never checked _last_voice_success_text,
            # so a working "play X on Spotify" could leave the step stuck
            # on "waiting for left hand one." No app-presence gate now.
            success_markers = (
                "spotify", "play", "youtube", "chrome",
                "execut", "launch", "pause",
            )
            if any(marker in text for marker in success_markers):
                self._last_voice_success_text = text

    def _tutorial_nav_from_payload(self, payload: dict, now: float) -> None:
        if payload.get("mouse_mode_enabled"):
            return
        if self._swipe_goal_index < 6 or now < self._nav_swipe_cooldown_until:
            return
        label = str(payload.get("dynamic_label", "neutral") or "neutral")
        if label == "swipe_left" and (self._show_completion_page or self._step_index > 0):
            self._nav_swipe_cooldown_until = now + self._tutorial_nav_cooldown_seconds
            self._visual_green_until["tutorial_nav"] = now + self._gesture_flash_seconds
            self._go_previous()
        elif label == "swipe_right" and (self._show_completion_page or self._step_index < len(self._practice_steps) - 1 or (self._step_index == len(self._practice_steps) - 1 and self._step_completed)):
            self._nav_swipe_cooldown_until = now + self._tutorial_nav_cooldown_seconds
            self._visual_green_until["tutorial_nav"] = now + self._gesture_flash_seconds
            self._go_next()

    def _reset_for_step(self) -> None:
        try:
            from ...telemetry import track as _track
            _track(
                "tutorial_step_entered",
                {
                    "step_index": int(self._step_index),
                    "step_key": (
                        self._practice_steps[self._step_index].key
                        if 0 <= self._step_index < len(self._practice_steps)
                        else "completion"
                    ),
                    "completion_page": bool(self._show_completion_page),
                },
            )
        except Exception:
            pass
        self._step_completed = True if self._show_completion_page else (self._step_index in self._completed_steps)
        self._hold_started.clear()
        self._hold_last_fired.clear()
        # Hide the steady ✓ + "Completed!" overlay whenever we move
        # to a new step. _update_completion_feedback re-shows it the
        # moment _step_completed flips True again. Skipped on the
        # completion page where the overlay is irrelevant.
        if not self._step_completed:
            try:
                self._fade_check_set_visible(False)
            except Exception:
                pass
        self._visual_hold_started.clear()
        self._visual_edge_active.clear()
        self._last_dynamic_label = "neutral"
        self._voice_listening = False
        self._voice_status = "ready"
        self._voice_heard_text = ""
        self._mouse_cursor_seen = False
        # Always hide the voice-mic arrow when entering a new step:
        # if the previous step left it visible (rare race during a
        # forced advance) we don't want it bleeding into the next
        # screen. show_pointing on the next voice step re-shows.
        arrow = getattr(self, "_voice_mic_arrow", None)
        if arrow is not None:
            arrow.hide_arrow()
        self._close_step_example_dialog()
        self._last_spotify_tutorial_action = "-"
        self._last_tutorial_play_pause_text = ""
        self._last_voice_success_text = ""
        self._tutorial_wheel_overlay.hide_overlay()
        self._tutorial_wheel_anchor = None
        self._tutorial_wheel_selected_key = None
        self._tutorial_wheel_selected_since = 0.0
        self._tutorial_wheel_cursor_offset = None
        while True:
            try:
                self._voice_queue.get_nowait()
            except queue.Empty:
                break
        self._mouse_tracker.reset()
        self.mouse_widget.reset()
        self._play_pause_ready_for_next = True
        # Volume-step state: reset so up/down/mute progress doesn't
        # leak across navigation. Re-initialized lazily in the
        # volume handler the next time the step runs.
        self._volume_step_state = None
        # Volume-step gate on the engine: leaving the volume step
        # (in either direction) MUST lift the mute block so post-
        # tutorial dictation / normal-mode use can mute as usual.
        self._set_engine_tutorial_mute_block(False)
        if self._voice_overlay is not None:
            self._voice_overlay.hide_overlay()
        self._apply_step_content()
        self._sync_worker_tutorial_context()

    def _apply_step_content(self) -> None:
        step = self._practice_steps[self._step_index]
        # Focus-guard timer is only useful during the voice_command
        # step (where Spotify / Chrome steal focus). Start it here
        # whenever we land on that step, stop it on every other
        # step / on the completion page so we don't burn cycles
        # checking the foreground during silent steps.
        if step.key == "voice_command" and not self._show_completion_page:
            try:
                if not self._focus_guard_timer.isActive():
                    self._focus_guard_timer.start()
            except Exception:
                pass
        else:
            try:
                if self._focus_guard_timer.isActive():
                    self._focus_guard_timer.stop()
            except Exception:
                pass
        if self._show_completion_page:
            self.body_stack.setCurrentIndex(1)
            # Hide the per-page hero + subtitle + badge so the only
            # header on the completion page is the in-body
            # "Tutorial Completed — Quick Tips" title (no double
            # header).
            self.hero_label.hide()
            self.hero_subtitle.hide()
            self.progress_badge.hide()
            self.guide_button.hide()
            self.example_button.hide()
            try:
                self.spotify_setup_button.hide()
            except Exception:
                pass
            try:
                self.mouse_instructions_button.hide()
            except Exception:
                pass
            self.prev_button.setEnabled(True)
            self.prev_button.setText("Previous")
            self.next_button.setEnabled(True)
            self.next_button.setText("Finish")
            self.leave_button.show()
            self._update_completion_feedback(time.monotonic())
            return

        self.body_stack.setCurrentIndex(0)
        # Restore the header on regular step pages.
        self.hero_label.show()
        self.hero_subtitle.show()
        self.progress_badge.show()
        self.guide_button.show()
        self.example_button.show()
        # Right card (info_card / practice_stack) is only shown on
        # the mouse step (where the click-practice arena lives) and
        # whenever the completion overlay is the active page in the
        # stack. Every other step hides it so the tutorial reads as
        # a single panel per the layout refactor.
        try:
            current_practice = self.practice_stack.currentWidget()
        except Exception:
            current_practice = None
        show_right = (
            step.key == "mouse_mode"
            or current_practice is self.completion_overlay
        )
        try:
            self.info_card.setVisible(show_right)
        except Exception:
            pass
        # Spotify shortcut: only visible on the voice command step,
        # where it's most useful (the step where users first try
        # "play X on Spotify" and discover the connection needs
        # authorisation).
        try:
            self.spotify_setup_button.setVisible(step.key == "voice_command")
        except Exception:
            pass
        self.progress_badge.setText(f"Step {self._step_index + 1} of {len(self._practice_steps)}")
        # Once the step is finished, flip the big top title to a
        # completion message so the user gets a clear "done" signal
        # at the same spot they read the step name from earlier.
        # Per-step completion copy below; fallback is generic.
        _completion_title_map = {
            "mouse_mode":    "Mouse controls completed!",
            "voice_command": "Voice command completed!",
            "volume":        "Volume controls completed!",
            "play_pause":    "Play / pause completed!",
            "swipes":        "Swipes completed!",
            "gesture_wheel": "Gesture wheel completed!",
        }
        if self._step_completed:
            self.step_title.setText(
                _completion_title_map.get(step.key, f"{step.title} completed!")
            )
        else:
            self.step_title.setText(step.title)

        # Step copy branches on Spotify presence: with Spotify the
        # voice step says "play X on Spotify" and play/pause +
        # swipes route through Spotify Web API; without, the voice
        # step says "play X on YouTube" and play/pause + swipes
        # send Windows media keys (which YouTube / Edge / VLC /
        # etc. respect via Windows Media integration).
        media_target = "Spotify" if self._has_spotify else "YouTube"
        media_app_phrase = "Spotify" if self._has_spotify else "your YouTube video"
        # Lowercase phrase the user actually speaks, e.g. "spotify"
        # or "youtube" — kept distinct from media_target which is
        # used as a proper noun ("Recommended for Spotify users…").
        media_voice_phrase = "spotify" if self._has_spotify else "youtube"
        what_is_map = {
            "mouse_mode": "Move the cursor and click with your right hand. Toggle on/off with your left.",
            "voice_command": f"Speak a command like \u201cplay [a song] on {media_voice_phrase}\u201d.",
            "volume": "Adjust system volume with a hand pose. Pinch in to mute / unmute.",
            "play_pause": "Pause or play whatever's playing on your computer.",
            "swipes": "Skip to the next or previous track.",
        }
        instruction_map = {
            "mouse_mode": (
                "Steps:\n"
                "1. Toggle mouse mode ON \u2014 LEFT hand, three fingers up (index + middle + ring), thumb across, pinky curled.\n"
                "2. Move the cursor \u2014 RIGHT hand open, palm to the camera. The cursor mirrors your hand inside the red box.\n"
                "3. Left-click \u2014 pinch your RIGHT thumb tip to your index tip, then release.\n"
                "4. Click each of the 4 tutorial targets in order.\n"
                "5. Toggle mouse mode OFF \u2014 LEFT hand three fingers again.\n"
                "6. Swipe right to continue.\n\n"
                "Bonus: right-click = thumb-to-middle. Scroll = index + middle touching; hold briefly to set an anchor, then lift up to scroll up or drop down to scroll down."
            ),
            "voice_command": (
                "Steps:\n"
                "1. LEFT hand, only the index finger up (others curled, thumb tucked).\n"
                "2. Hold until the microphone appears at the bottom of the screen — that means Touchless is listening.\n"
                f"3. Speak clearly: \u201cplay [a song] on {media_voice_phrase}\u201d.\n"
                "4. Swipe right to continue once it starts playing.\n\n"
                + (
                    "To complete: trigger the listener and play something on Spotify."
                    if self._has_spotify else
                    "To complete: trigger the listener and start a YouTube video playing in your default browser."
                )
            ),
            "volume": (
                "How to do it:\n"
                "\u2022 RIGHT hand, peace sign with index + middle TOUCHING (closed peace sign), ring + pinky curled, thumb relaxed.\n"
                "\u2022 Hold the pose to enter volume mode — the volume bar appears on screen.\n"
                "\u2022 Move your hand UP to raise volume, DOWN to lower. Bigger moves = bigger jumps.\n"
                "\u2022 Mute / unmute: hold a SHAKA \u2014 thumb and pinky extended outwards while "
                "index, middle, and ring fingers are curled in. Palm facing towards "
                "the monitor. Release the pose between toggles \u2014 it has to fire "
                "twice (mute, then unmute) to count.\n\n"
                "To complete: raise the volume, lower it, then mute and unmute once."
            ),
            "play_pause": (
                "How to do it:\n"
                "\u2022 RIGHT hand, closed fist — all four fingers curled in, thumb across the front.\n"
                "\u2022 Knuckles face the camera.\n"
                "\u2022 Each clean fist toggles play/pause.\n\n"
                f"To complete: pause and resume {media_app_phrase} once each (fist, relax, fist again)."
            ),
            "swipes": (
                "How to do it:\n"
                "\u2022 RIGHT hand, open palm (all 5 fingers extended, palm to camera).\n"
                "\u2022 Swipe right: start on the LEFT, sweep to the right in one motion (next track).\n"
                "\u2022 Swipe left: start on the RIGHT, sweep to the left (previous track).\n"
                "\u2022 The motion triggers it, not the pose.\n\n"
                "To complete: 3 right swipes, then 3 left swipes. Bbox turns green on each one."
            ),
        }
        # Always set the instruction_box text first — _open_step_example
        # reads it back to compose the "Detailed instructions" section in
        # the popup, and the Show-more link on non-mouse steps relies on
        # the same popup as the source of truth.
        self.instruction_box.setText(instruction_map.get(step.key, step.description))
        short_desc = what_is_map.get(step.key, step.description)
        # White-text description sits inline next to the green title.
        # The "Show more…" link USED to be appended here; per the
        # latest UX request it now lives at the end of the green
        # camera-header text instead, surfaced via _set_camera_header_text.
        self.step_desc.setText(short_desc)
        self.instruction_box.setVisible(False)
        # NEVER setVisible(True) on mouse_instructions_button — it was
        # removed from the title row at construction time, which makes
        # it an orphan widget with no parent layout. Qt promotes
        # orphan-but-visible widgets to top-level windows, which was
        # popping the button up as a free-floating dialog whenever
        # the user navigated to the mouse step. Force it hidden here.
        self.mouse_instructions_button.setVisible(False)
        self.note_label.clear()
        self.voice_preview_label.clear()

        # The right-info-card "practice box" used to mirror what's now
        # shown big and bold above + below the camera view. For most
        # steps that's pure duplication, so we hide the practice_stack
        # entirely — only the mouse-mode step has an interactive
        # widget (target dots to clear) that the camera-view text
        # can't replicate. Header and footer reset before each step so
        # one step's text never bleeds into another.
        self.tutorial_camera_header.hide()
        self.tutorial_camera_footer.hide()

        if step.key == "swipes":
            self.practice_stack.hide()
            self.swipe_widget.set_counts(self._swipe_counts["swipe_left"], self._swipe_counts["swipe_right"])
            self._refresh_swipe_camera_labels()
            self.progress_label.clear()
        elif step.key == "gesture_wheel":
            self.practice_stack.hide()
            self._set_camera_step_labels(
                header="Hold the wheel pose to open the gesture wheel.",
                footer=(
                    "Completed! Swipe right to move on!"
                    if self._step_completed
                    else "Waiting for the wheel pose…"
                ),
            )
            self.progress_label.clear()
        elif step.key == "mouse_mode":
            self.practice_stack.show()
            self.practice_stack.setCurrentWidget(self.mouse_widget)
            if self._step_completed:
                self.mouse_widget.mark_all_targets_completed()
            if self._step_completed:
                footer_text = "Swipe right to continue!"
            elif self._mouse_stage == "enable":
                footer_text = "Toggle mouse mode on to begin."
            elif self._mouse_stage == "practice":
                footer_text = "Click each of the four targets."
            else:
                footer_text = "All targets done. Toggle mouse mode off."
            self._set_camera_step_labels(
                header=self._mouse_mode_header_text(mode_enabled=False),
                footer=footer_text,
            )
            self.progress_label.clear()
        else:
            self.practice_stack.hide()
            header_map = {
                "play_pause": "Pause / play with right-hand fist!",
                "voice_command": self._voice_command_header_text(False),
                "volume": "Hold the volume pose to adjust.",
            }
            self._set_camera_step_labels(
                header=header_map.get(step.key, step.description),
                footer=self._step_progress_footer(step),
            )
            self.progress_label.clear()

        self.prev_button.setEnabled(self._step_index > 0)
        self.next_button.setEnabled(self._step_index in self._completed_steps)
        self.next_button.setText("Next")
        self._update_completion_feedback(time.monotonic())

    _ENCOURAGEMENT_MESSAGES = (
        "Good job!",
        "Nice work!",
        "Yay!",
        "Awesome!",
        "Perfect!",
        "Got it!",
        "Nice!",
        "You got it!",
        "Way to go!",
    )

    def _check_encouragement_trigger(self, now: float) -> None:
        """No-op kept for backwards compatibility. The encouragement
        popup was previously fired on every newly-pushed
        _visual_green_until edge (which fires whenever the hand turns
        green for ANY reason — including detection-only flashes that
        don't actually trigger an action). The popup now fires from
        explicit `_trigger_encouragement(now)` calls placed at the
        actual action-completion sites: each individual swipe count
        bump, each play/pause toggle, each step completion, etc."""
        # Keep _last_seen_green_until in sync so we don't burst a flood
        # of popups if some other code path starts using it.
        for key, until in list(self._visual_green_until.items()):
            self._last_seen_green_until[key] = until

    def _trigger_encouragement(self, now: float) -> None:
        """Fire a random encouragement message at the bottom of the
        camera view for 1.5 s. Throttled to one per second so a
        rapid burst (e.g., spamming swipes) doesn't stack popups.
        Call this AFTER the action actually fires (a count was
        incremented, a step completed, etc.) — not on the gesture
        detection edge."""
        if now - self._encouragement_last_at < 1.0:
            return
        self._encouragement_last_at = now
        try:
            self._encouragement_text = random.choice(self._ENCOURAGEMENT_MESSAGES)
        except Exception:
            self._encouragement_text = "Nice!"
        self._encouragement_until = now + 1.5

    def _draw_encouragement_overlay(self, frame, now: float) -> None:
        """Render the active encouragement message centered in the
        MIDDLE of the camera frame in big bold accent-green text.
        Fades over the last 0.5 s of the 1.5 s lifetime via alpha-
        blended overlay."""
        if self._encouragement_until <= 0.0 or not self._encouragement_text:
            return
        remaining = self._encouragement_until - now
        if remaining <= 0.0:
            self._encouragement_until = 0.0
            self._encouragement_text = ""
            return
        text = self._encouragement_text
        try:
            h, w = frame.shape[:2]
        except Exception:
            return
        # FONT_HERSHEY_TRIPLEX has the cleanest serifed look of the
        # built-in cv2 fonts — pairs well with the heavy stroke we
        # paint below to fake a 'bold' weight (cv2 has no separate
        # bold variant; we layer two passes with different thickness
        # to get the same visual heft).
        font = cv2.FONT_HERSHEY_TRIPLEX
        scale = max(1.4, h / 280.0)
        thickness = max(3, int(round(scale * 1.8)))
        (tw, th), _baseline = cv2.getTextSize(text, font, scale, thickness)
        cx = w // 2
        cy = h // 2
        text_x = cx - tw // 2
        text_y = cy + th // 2
        # Fade: full opacity until the last 0.5 s, then linearly down
        # to zero.
        fade_window = 0.5
        alpha = 1.0 if remaining >= fade_window else max(0.0, remaining / fade_window)
        if alpha <= 0.01:
            return
        accent_bgr = (182, 233, 29)  # = #1DE9B6 → BGR
        outline_bgr = (10, 25, 35)
        overlay = frame.copy()
        # Heavier outline + double fill pass to fake bold.
        outline_thickness = max(thickness + 4, 7)
        cv2.putText(
            overlay, text, (text_x, text_y), font, scale,
            outline_bgr, outline_thickness, cv2.LINE_AA,
        )
        # Two passes of the fill at slightly different thickness give
        # the strokes a bolder look than a single pass would.
        cv2.putText(
            overlay, text, (text_x, text_y), font, scale,
            accent_bgr, thickness + 1, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, text, (text_x, text_y), font, scale,
            accent_bgr, thickness, cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)

    def _update_encouragement_visual(self, now: float) -> None:
        """Compatibility shim — drawing happens during _render_frame
        via _draw_encouragement_overlay; this just decays the timer
        when called from non-render code paths."""
        if self._encouragement_until > 0.0 and now >= self._encouragement_until:
            self._encouragement_until = 0.0
            self._encouragement_text = ""

    def _set_step_progress(self, text: str) -> None:
        """Single setter for per-frame progress strings. Updates the
        big-bold camera-view footer (new home for this info) AND the
        legacy right-side progress_label as a fallback. Swipes step
        owns its own footer text via _refresh_swipe_camera_labels and
        skips this — its progress is the count, not a sentence."""
        try:
            step_key = self._practice_steps[self._step_index].key
        except Exception:
            step_key = ""
        if step_key != "swipes":
            try:
                self.tutorial_camera_footer.setText(text)
                self.tutorial_camera_footer.show()
            except Exception:
                pass
        try:
            self.progress_label.setText(text)
        except Exception:
            pass

    def _set_camera_step_labels(self, header: str = "", footer: str = "") -> None:
        """Show / clear the big-bold accent-coloured header and footer
        that frame the camera view. Empty strings hide the widget so
        the layout doesn't reserve space for a blank line."""
        self._set_camera_header_text(header)
        if footer:
            self.tutorial_camera_footer.setText(footer)
            self.tutorial_camera_footer.show()
        else:
            self.tutorial_camera_footer.clear()
            self.tutorial_camera_footer.hide()

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API name)
        # Keep the floating ✓ overlay centred and sized to the
        # current video_label geometry. Re-position on every resize
        # because the QLabel parent-child relationship doesn't
        # auto-stretch the child.
        try:
            if obj is self.video_label and event.type() == event.Type.Resize:
                self._position_fade_check()
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _position_fade_check(self) -> None:
        """Re-position + re-style the ✓ glyph and the 'Completed!'
        caption to track the live video_label dimensions. Called on
        every resize via eventFilter and on every show via
        _fade_check_set_visible. Alpha is baked into the rgba(...)
        colour (no QGraphicsOpacityEffect) so the overlay holds at a
        constant transparency instead of fading in/out."""
        try:
            w = self.video_label.width()
            h = self.video_label.height()
        except Exception:
            return
        if w <= 0 or h <= 0:
            return
        # Check glyph fills ~58% of the smaller dimension. Slightly
        # smaller than before so there's room for the caption below
        # without crowding the camera frame.
        size = max(110, int(min(w, h) * 0.58))
        font_px = max(110, int(size * 0.85))
        caption_h = max(28, int(size * 0.18))
        caption_font_px = max(22, int(caption_h * 0.82))
        try:
            # Centre the (glyph + caption) pair vertically — total
            # height = size + small gap + caption_h. Bias the cluster
            # slightly upward (top_offset) so it doesn't sit on the
            # camera's bottom edge.
            gap = max(6, int(size * 0.04))
            cluster_h = size + gap + caption_h
            top = max(0, (h - cluster_h) // 2)
            self._fade_check_label.setGeometry((w - size) // 2, top, size, size)
            self._fade_check_caption.setGeometry(
                (w - int(size * 1.4)) // 2,
                top + size + gap,
                int(size * 1.4),
                caption_h,
            )
            # Both glyphs use the same rgba colour so they fade
            # together visually at a constant level (alpha 220/255 ≈
            # 86% — slightly less transparent than the prior fade
            # peak of 0.92 → 92%? actually the user asked for
            # "slightly less transparent" = "more opaque", so we
            # bump opacity. 220/255 reads as a confident overlay
            # without fully blocking the camera).
            self._fade_check_label.setStyleSheet(
                "QLabel#tutorialFadeCheck {"
                "  color: rgba(29, 233, 182, 220);"
                f"  font-size: {font_px}px;"
                "  font-weight: 900;"
                "  background: transparent;"
                "}"
            )
            self._fade_check_caption.setStyleSheet(
                "QLabel#tutorialFadeCheckCaption {"
                "  color: rgba(29, 233, 182, 220);"
                f"  font-size: {caption_font_px}px;"
                "  font-weight: 800;"
                "  letter-spacing: 0.02em;"
                "  background: transparent;"
                "}"
            )
        except Exception:
            pass

    def _fade_check_set_visible(self, visible: bool) -> None:
        """Show or hide the steady ✓ + 'Completed!' overlay. Idempotent
        — repeated calls don't trigger any animation or restart."""
        try:
            if visible:
                self._position_fade_check()
                self._fade_check_label.raise_()
                self._fade_check_caption.raise_()
                self._fade_check_label.show()
                self._fade_check_caption.show()
            else:
                self._fade_check_label.hide()
                self._fade_check_caption.hide()
        except Exception:
            pass

    def _set_camera_header_text(self, text: str) -> None:
        if text:
            # Per UX request, append the "Show more…" link to the
            # green camera-header text so it sits immediately after
            # the in-the-moment instruction (e.g. "Use example clip
            # for help. Show more…"). The header is rich-text +
            # link-interactive (configured at construction time).
            # Tutorial-completion-page texts ("All done, great work!"
            # etc.) skip the suffix — they're celebratory and not
            # backed by an example.
            try:
                step = self._practice_steps[self._step_index]
                allow_link = (
                    not self._show_completion_page
                    and not self._step_completed
                )
            except Exception:
                allow_link = False
            if allow_link:
                annotated = (
                    f"{text} "
                    "<a href='show_more' "
                    "style='color: #58E3FF; text-decoration: underline; "
                    "font-weight: 600;'>Show more…</a>"
                )
            else:
                annotated = text
            self.tutorial_camera_header.setText(annotated)
            self.tutorial_camera_header.show()
        else:
            self.tutorial_camera_header.clear()
            self.tutorial_camera_header.hide()

    def _step_progress_footer(self, step) -> str:
        """Compose the footer line shown under the camera view for a
        non-swipes step. Mirrors what the old progress_label said but
        rendered big-bold-accent below the live image."""
        if self._step_completed:
            return "Completed! Swipe right to move on!"
        if step.key == "play_pause":
            return self._fist_progress_html()
        if step.key == "spotify_open":
            return "Waiting for right-hand two…"
        if step.key == "voice_command":
            return "Waiting for left-hand one and the voice command."
        return step.progress_template or ""

    @staticmethod
    def _progress_color(n: int, target: int) -> str:
        """Color tiers for any "N/target" progress counter:
        0 = red (haven't started), 1..target-1 = orange (in progress),
        target = green (done). Used by both the swipes footer (target=3)
        and the play/pause-fist footer (target=2). Same palette so the
        visual feedback feels consistent across steps."""
        if n <= 0:
            return "#FF5252"
        if n < target:
            return "#FFA726"
        return "#1DE9B6"

    @staticmethod
    def _swipe_count_color(n: int) -> str:
        """Back-compat wrapper kept so any older code paths still
        compile. Forwards to the generic _progress_color with the
        swipes step's target of 3."""
        return TutorialWindow._progress_color(n, 3)

    def _fist_progress_html(self) -> str:
        """Footer text for the play/pause-fist step with the count
        colored on the same red/orange/green tier the swipes step
        uses. Same visual style across both steps so the user gets
        the same "where am I" cue everywhere it shows up."""
        n = int(self._spotify_toggle_count)
        color = self._progress_color(n, 2)
        return f'Fist detections <span style="color:{color};">{n}/2</span>'

    def _refresh_swipe_camera_labels(self) -> None:
        """Drive the big-bold accent-green header above the camera and
        the matching counter below it for the swipes step. Phase
        switches at 3 right swipes: header changes from 'right' to
        'left', and the counter swaps to track left swipes.

        Also handles the 1-second transition celebration after each
        sub-phase finishes: 'Completed right swipes!' between the
        right and left phases, and 'Completed left swipes!' before
        the overall step settles into 'All done, great work!' +
        'Swipe right to move on!'."""
        right_count = int(self._swipe_counts.get("swipe_right", 0))
        left_count = int(self._swipe_counts.get("swipe_left", 0))
        in_left_phase = right_count >= 3
        now = time.monotonic()
        # 1-second window after the most recent count advance: we use
        # this to show "Completed right swipes!" / "Completed left
        # swipes!" before transitioning to the next prompt.
        in_transition = now - self._swipe_last_advance_at < 1.0

        footer_html: str | None = None  # set when we want a fixed
        # post-completion sentence instead of the count line.
        if self._step_completed:
            if in_transition:
                header = "Completed left swipes!"
                count, label = 3, "left swipes"
            else:
                header = "All done, great work!"
                footer_html = "Swipe right to move on!"
                count, label = 3, "left swipes"
        elif not in_left_phase:
            # Use "example clip" for dynamic gestures (swipes), "example
            # image" for static. Swipes is dynamic so the link directs
            # the user to the looping clip in the Show Example popup.
            header = "Let's start with swiping right! Use the example clip for help."
            count, label = right_count, "right swipes"
        elif left_count == 0 and in_transition:
            header = "Completed right swipes!"
            count, label = right_count, "right swipes"
        elif left_count >= 3 and in_transition:
            header = "Completed left swipes!"
            count, label = left_count, "left swipes"
        else:
            header = "Now let's try swiping to the left!"
            count, label = left_count, "left swipes"

        if footer_html is None:
            color = self._swipe_count_color(count)
            footer_html = (
                f'Completed '
                f'<span style="color:{color};">{count}/3</span> '
                f'{label}'
            )
        # Route through _set_camera_header_text so the "Show more…"
        # link gets appended automatically (the swipe step used to
        # call setText directly and skip that path, which left the
        # link missing on every swipe-mode refresh).
        self._set_camera_header_text(header)
        self.tutorial_camera_footer.setText(footer_html)
        self.tutorial_camera_header.show()
        self.tutorial_camera_footer.show()

    def _complete_step(self, note: str | None = None) -> None:
        if self._step_completed and self._completion_feedback_step == self._step_index:
            return
        self._step_completed = True
        self._completed_steps.add(self._step_index)
        self.next_button.setEnabled(True)
        self._completion_feedback_until = time.monotonic() + self._completion_feedback_duration
        self._completion_feedback_step = self._step_index
        # Step completion is the canonical action-completion edge for
        # most parts of the tutorial; fire encouragement here.
        self._trigger_encouragement(time.monotonic())
        text = note or "Completed! Swipe right to move on!"
        self._set_step_progress(text)
        # Flip the big top title to the per-step completion copy so
        # the user gets a clear "done" signal at the same place they
        # read the step name from. Mirrors the same map used when the
        # step page is (re-)rendered with self._step_completed=True.
        try:
            step = self._practice_steps[self._step_index]
            _completion_title_map = {
                "mouse_mode":    "Mouse controls completed!",
                "voice_command": "Voice command completed!",
                "volume":        "Volume controls completed!",
                "play_pause":    "Play / pause completed!",
                "swipes":        "Swipes completed!",
                "gesture_wheel": "Gesture wheel completed!",
            }
            self.step_title.setText(
                _completion_title_map.get(step.key, f"{step.title} completed!")
            )
        except Exception:
            pass

    def _update_completion_feedback(self, now: float) -> None:
        visible = self._step_completed and not self._show_completion_page
        # Always-hidden — replaced by completion_overlay (in
        # practice_stack) which doesn't squeeze the instruction box.
        self.completion_check_label.hide()
        self.completion_text_label.hide()
        if not visible:
            return
        try:
            step_key = self._practice_steps[self._step_index].key
        except Exception:
            step_key = ""
        # Mouse mode paints its OWN big checkmark inside the practice
        # widget (over the target dots) — keep practice_stack on the
        # mouse widget so that paint stays visible.
        if step_key == "mouse_mode":
            try:
                self.mouse_widget.update()
            except Exception:
                pass
            return
        # Every other step: switch the practice_stack to the
        # completion overlay page. Stack absorbs the available
        # vertical space (stretch=1), so the giant ✓ fills the
        # empty area below the instruction text WITHOUT shrinking
        # it. Make sure the stack is visible — for swipes step
        # _apply_step_content explicitly hides it; we override that
        # here so the overlay can show.
        # Per UX: show the steady-state ✓ + "Completed!" overlay
        # over the live camera view. Constant transparency, no
        # animation — replaces the prior fade-in/fade-out which kept
        # restarting every tick and visually flashed.
        try:
            self._fade_check_set_visible(True)
        except Exception:
            pass

    def _go_previous(self) -> None:
        if self._show_completion_page:
            self._show_completion_page = False
            self._reset_for_step()
            return
        if self._step_index <= 0:
            return
        self._step_index -= 1
        self._reset_for_step()

    def _go_next(self) -> None:
        if self._show_completion_page:
            self._finish_and_close(True)
            return
        if not self._step_completed:
            return
        if self._step_index >= len(self._practice_steps) - 1:
            self._show_completion_page = True
            self._reset_for_step()
            return
        self._step_index += 1
        self._reset_for_step()

    def _open_guide(self) -> None:
        self._closing_programmatically = True
        self._stop_session()
        self.gesture_guide_requested.emit(self._launched_from_settings)
        self.close()

    def _finish_and_close(self, completed: bool) -> None:
        if self._close_emitted:
            self.close()
            return
        self._close_emitted = True
        self._closing_programmatically = True
        self._stop_session()
        self.tutorial_closed.emit(bool(completed), self._auto_start_on_done, self._launched_from_settings)
        self.close()

    def _render_frame(self, frame) -> None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        image = QImage(frame_rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _hold_ready(self, key: str, active: bool, threshold: float, now: float, cooldown: float = 0.55) -> bool:
        if not active:
            self._hold_started.pop(key, None)
            return False
        start = self._hold_started.setdefault(key, now)
        if now - self._hold_last_fired.get(key, -1e9) < cooldown:
            return False
        if now - start >= threshold:
            self._hold_last_fired[key] = now
            self._hold_started.pop(key, None)
            self._visual_green_until[key] = max(self._visual_green_until.get(key, 0.0), now + self._gesture_flash_seconds)
            return True
        return False

    def _gesture_active(self, result, label: str, *, handedness: str | None, min_confidence: float = 0.56) -> bool:
        if not result.found or result.tracked_hand is None:
            return False
        if handedness is not None and str(result.tracked_hand.handedness or "").lower() != handedness.lower():
            return False
        prediction = result.prediction
        return prediction.stable_label == label or (prediction.raw_label == label and prediction.confidence >= min_confidence)

    def _flash_on_edge(self, key: str, active: bool, now: float) -> None:
        previous = self._visual_edge_active.get(key, False)
        if active and not previous:
            self._visual_green_until[key] = max(self._visual_green_until.get(key, 0.0), now + self._gesture_flash_seconds)
        self._visual_edge_active[key] = bool(active)

    def _visual_ready(self, key: str, active: bool, now: float, threshold: float = 0.6) -> bool:
        self._flash_on_edge(key, active, now)
        return now < self._visual_green_until.get(key, 0.0)

    def _tutorial_nav_from_swipe(self, prediction, now: float) -> None:
        if self._mouse_tracker.mode_enabled:
            return
        if self._swipe_goal_index < 6 or now < self._nav_swipe_cooldown_until:
            return
        label = str(getattr(prediction, "dynamic_label", "neutral") or "neutral")
        if label == "swipe_left" and (self._show_completion_page or self._step_index > 0):
            self._nav_swipe_cooldown_until = now + self._tutorial_nav_cooldown_seconds
            self._visual_green_until["tutorial_nav"] = now + self._gesture_flash_seconds
            self._go_previous()
        elif label == "swipe_right" and (self._show_completion_page or self._step_index < len(self._practice_steps) - 1 or (self._step_index == len(self._practice_steps) - 1 and self._step_completed)):
            self._nav_swipe_cooldown_until = now + self._tutorial_nav_cooldown_seconds
            self._visual_green_until["tutorial_nav"] = now + self._gesture_flash_seconds
            self._go_next()

    def _draw_user_skeleton_overlay(self, frame, result, color) -> None:
        if not result.found or result.tracked_hand is None:
            return
        height, width = frame.shape[:2]
        pts = result.tracked_hand.landmarks
        for a, b in HAND_CONNECTIONS:
            pa = (int(pts[a][0] * width), int(pts[a][1] * height))
            pb = (int(pts[b][0] * width), int(pts[b][1] * height))
            cv2.line(frame, pa, pb, color, 3, cv2.LINE_AA)
        for index in range(len(pts)):
            px = int(pts[index][0] * width)
            py = int(pts[index][1] * height)
            cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)

    def _draw_top_label(self, frame, text: str, color_bgr: tuple) -> None:
        """Render a bold label horizontally centred near the TOP of the
        camera frame. Used by the swipes demo so the directional cue
        ('Swipe right' / 'Swipe left') sits above the demo hands instead
        of crowding the centre of the live view."""
        try:
            h, w = frame.shape[:2]
        except Exception:
            return
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = max(0.85, min(1.25, h / 540.0))
        thickness = max(2, int(round(scale * 2.0)))
        (tw, th), _baseline = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = max(int(h * 0.10), th + 12)
        # Dark outline first for legibility on busy camera content.
        cv2.putText(frame, text, (x, y), font, scale, (10, 25, 35),
                    thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, color_bgr,
                    thickness, cv2.LINE_AA)

    def _sprite_for(self, name: str):
        """Lazy-load and cache the RGBA sprite for `name`. Returns
        a StaticSprite, SequenceSprite, or None when no asset
        exists (in which case the demo overlay simply skips drawing
        a hand for that step)."""
        cached = self._sprites.get(name, "missing")
        if cached != "missing":
            return cached
        try:
            from .tutorial_demo_sprite import resolve_sprite
            sprite = resolve_sprite(name)
        except Exception:
            sprite = None
        self._sprites[name] = sprite
        return sprite

    def _static_demo_alpha(self, now: float) -> float:
        """Fade-in / hold / fade-out / hide cycle for static demos.
        Replaces the previous geometric morph (which produced
        anatomically nonsensical intermediate poses when
        interpolating MediaPipe landmarks). Now the recorded snapshot
        renders directly; the only animation is overall opacity."""
        cycle = 3.0
        ct = now % cycle
        if ct < 0.35:
            p = ct / 0.35
            return p * p * (3.0 - 2.0 * p)
        if ct < 2.0:
            return 1.0
        if ct < 2.45:
            p = 1.0 - (ct - 2.0) / 0.45
            return p * p * (3.0 - 2.0 * p)
        return 0.0

    def _draw_tutorial_mouse_overlays(self, frame, payload_state: dict) -> None:
        """Use the SAME mouse overlays the live view uses, fed from
        the engine's per-frame payload (no engine objects to share
        across processes — we reconstruct the small dataclass +
        adapter the overlays need from the payload's primitive
        fields).

        Renders both:
            - draw_mouse_control_box_overlay → the red control-area
              box + cursor dot in camera space.
            - draw_mouse_monitor_overlay → the corner Desktop Map
              panel with each monitor + the actual cursor position.

        This replaces the older bespoke `_draw_tutorial_mouse_box`
        so the tutorial's visualization stays in sync with whatever
        the live view evolves to."""
        bounds = payload_state.get("camera_control_bounds")
        cursor_norm = payload_state.get("cursor_position")
        virtual_bounds = payload_state.get("virtual_bounds")

        debug_state = MouseDebugState(
            mode_enabled=True,
            status="active",
            cursor_position=tuple(cursor_norm) if cursor_norm is not None else None,
            cursor_anchor_position=None,
            cursor_reach_bounds=None,
            camera_control_bounds=(
                tuple(float(v) for v in bounds) if bounds is not None else None
            ),
            camera_anchor_position=None,
            dragging=False,
            scrolling=False,
        )
        adapter = self._PayloadMouseControllerAdapter(virtual_bounds, cursor_norm)
        active_monitor = getattr(self.config, "mouse_active_monitor_index", None)
        try:
            draw_mouse_control_box_overlay(
                frame,
                debug_state=debug_state,
                mode_enabled=True,
                active_monitor_index=active_monitor,
            )
        except Exception:
            pass
        try:
            draw_mouse_monitor_overlay(
                frame,
                mouse_controller=adapter,
                debug_state=debug_state,
                mode_enabled=True,
            )
        except Exception:
            pass

    class _PayloadMouseControllerAdapter:
        """Minimal stand-in for MouseController, fed from the engine
        payload's `mouse_virtual_bounds` + normalized cursor pos.
        draw_mouse_monitor_overlay only needs `available`,
        `virtual_bounds()`, and `current_position()` — the rest of
        MouseController's interface (clicks, scrolling, etc.) isn't
        used by the overlay."""

        __slots__ = ("_vb", "_cp_norm", "available")

        def __init__(self, virtual_bounds, cursor_pos_norm) -> None:
            self._vb = tuple(virtual_bounds) if virtual_bounds is not None else None
            self._cp_norm = (
                tuple(cursor_pos_norm) if cursor_pos_norm is not None else None
            )
            self.available = self._vb is not None

        def virtual_bounds(self):
            return self._vb

        def current_position(self):
            if self._vb is None or self._cp_norm is None:
                return None
            v_left, v_top, v_w, v_h = self._vb
            x = float(v_left) + float(self._cp_norm[0]) * float(v_w)
            y = float(v_top) + float(self._cp_norm[1]) * float(v_h)
            return (x, y)

    # Inset geometry: pinned to the top-right corner of the live
    # view, sized to ~26 % of frame width with a tiny margin so
    # the visible clip content's top-right edge sits hard against
    # the frame's top-right edge.
    _INSET_W_FRAC = 0.26
    # No margin — the inset's outer edges sit flush against the
    # frame's top and right edges. Previous 6 px gap left a thin
    # band of camera content visible above and to the right of
    # the demo, which the user reported as a misalignment.
    _INSET_MARGIN = 0
    _INSET_MAX_H_FRAC = 0.55

    def _inset_rect(self, frame_w: int, frame_h: int,
                    clip_aspect: float = 0.75,
                    scale: float = 1.0) -> tuple:
        """Return (cx, cy, w, h) for the top-right inset, sized to
        match the clip's natural aspect (height / width) so we don't
        letterbox empty bars that visually pull the gesture away
        from the corner. Caps height to _INSET_MAX_H_FRAC of the
        frame so a tall portrait still doesn't dominate the live
        view.

        `scale` shrinks (or grows) the inset uniformly while
        keeping it pinned to the top-right corner — used per-step
        to size each gesture clip appropriately (e.g., the tall
        Two pose at part 2 is rendered at 2/3 so it doesn't
        dominate the live frame)."""
        scale = max(0.2, float(scale))
        w = max(80, int(round(frame_w * self._INSET_W_FRAC * scale)))
        h = max(60, int(round(w * clip_aspect)))
        max_h = int(frame_h * self._INSET_MAX_H_FRAC * scale)
        if h > max_h:
            h = max_h
            w = max(80, int(round(h / max(0.1, clip_aspect))))
        cx = frame_w - self._INSET_MARGIN - w // 2
        cy = self._INSET_MARGIN + h // 2
        return cx, cy, w, h

    def _draw_static_demo(
        self,
        frame,
        _main_center: tuple,
        demo_name: str,
        *,
        fallback_scale: float = 1.0,
        mirror: bool = False,
        now: float,
    ) -> None:
        """Play the Control Guide clip for `demo_name` in a small
        bordered inset at the top-right of the live frame. Static
        gestures stay visible for the entire step (no fade cycle —
        users were missing them during the off-phases). The
        `fallback_scale` arg now controls the inset size so steps
        with tall portrait clips (e.g. part 2's Two gesture) can
        shrink without bothering the layout."""
        clip = self._sprite_for(demo_name)
        if clip is None:
            return
        state = clip.at(now)
        if not state.visible or state.frame is None:
            return
        height, width = frame.shape[:2]
        ch, cw = state.frame.shape[:2]
        clip_aspect = ch / max(1.0, float(cw))
        cx, cy, rw, rh = self._inset_rect(width, height, clip_aspect,
                                           scale=fallback_scale)
        from .tutorial_demo_sprite import composite_sprite
        composite_sprite(frame, state.frame, (cx, cy, rw, rh),
                         alpha=1.0, mirror=mirror)

    def _draw_demo_overlay(self, frame, step_key: str) -> None:
        height, width = frame.shape[:2]
        accent = (182, 233, 29)
        try:
            now = time.monotonic()
        except Exception:
            now = 0.0
        if step_key == "swipes":
            right_phase = self._swipe_goal_index < 3
            demo_name = "swipe_right" if right_phase else "swipe_left"
            arrow_y = int(height * 0.46)
            if right_phase:
                arrow_start = (int(width * 0.20), arrow_y)
                arrow_end = (int(width * 0.80), arrow_y)
            else:
                arrow_start = (int(width * 0.80), arrow_y)
                arrow_end = (int(width * 0.20), arrow_y)
            cv2.arrowedLine(frame, arrow_start, arrow_end, accent, 4,
                             cv2.LINE_AA, tipLength=0.20)
            self._draw_top_label(frame,
                                  "Swipe right" if right_phase else "Swipe left",
                                  accent)
            clip = self._sprite_for(demo_name)
            if clip is not None:
                state = clip.at(now)
                if state.visible and state.frame is not None:
                    ch, cw = state.frame.shape[:2]
                    clip_aspect = ch / max(1.0, float(cw))
                    cx, cy, rw, rh = self._inset_rect(width, height, clip_aspect)
                    from .tutorial_demo_sprite import composite_sprite
                    composite_sprite(frame, state.frame, (cx, cy, rw, rh))
        elif step_key == "spotify_open":
            # Two.png is portrait — at full inset width it dominates
            # the corner, pushing past the edge-glow margins. 0.67
            # scale shrinks it to ~2/3 so the gesture stays clearly
            # visible without crowding the live frame.
            self._draw_static_demo(frame, None, "right_two",
                                    fallback_scale=0.67, now=now)
        elif step_key == "play_pause":
            self._draw_static_demo(frame, None, "right_fist",
                                    fallback_scale=1.00, now=now)
        elif step_key == "gesture_wheel":
            self._draw_static_demo(frame, None, "wheel_pose",
                                    fallback_scale=0.96, now=now)
        elif step_key == "mouse_mode":
            # Top-right helper inset:
            #   Mouse mode OFF                    → small "Left Three.png"
            #     pose so the user knows the activation gesture.
            #   Mouse mode ON, targets remaining  → looped "Mouse Clicks.mp4"
            #     demo so they see the pinch click mechanic to mirror.
            #   Mouse mode ON, all targets done   → small "Left Three.png"
            #     pose again, signalling that the next thing to do is
            #     toggle mouse mode off (same gesture as turning it on).
            # The Left Three image is rendered at a smaller scale than
            # the Mouse Clicks demo so it doesn't dominate the live view
            # (it's just a static reminder, not something the user has
            # to mirror frame-by-frame).
            # Plus, when mouse mode is ON, we ALSO render the SAME
            # mouse overlays the live view uses so the tutorial preview
            # matches the real experience.
            mouse_state = getattr(self, "_payload_mouse_state", None) or (
                {
                    "mode_enabled": True,
                    "camera_control_bounds": (
                        tuple(float(v) for v in self._mouse_tracker.debug_state.camera_control_bounds)
                        if self._mouse_tracker.debug_state.camera_control_bounds is not None
                        else None
                    ),
                    "cursor_position": self._mouse_tracker.debug_state.cursor_position,
                    "virtual_bounds": None,
                } if self._mouse_tracker.mode_enabled else None
            )
            mode_on = mouse_state is not None and bool(mouse_state.get("mode_enabled"))
            try:
                targets_done = bool(self.mouse_widget.completed)
            except Exception:
                targets_done = False
            try:
                scroll_done = bool(self.mouse_scroll_widget.completed)
            except Exception:
                scroll_done = False
            # Three-way branching across the mouse-mode sub-stages:
            #   click phase (targets pending, mouse mode on)
            #       → mouse_clicks demo (pinch-click motion)
            #   scroll phase (targets done, scroll incomplete)
            #       → tutorial_scrolling demo (two-finger scroll pose)
            #   pre-activation OR final disable phase
            #       → left_three pose (toggle mouse mode on / off)
            in_scroll_phase = targets_done and not scroll_done
            if mode_on and not targets_done:
                self._draw_static_demo(frame, None, "mouse_clicks",
                                        fallback_scale=1.00, now=now)
            elif mode_on and in_scroll_phase:
                self._draw_static_demo(frame, None, "tutorial_scrolling",
                                        fallback_scale=1.00, now=now)
            else:
                # Either pre-activation (mode_off) or post-completion
                # (targets + scroll both cleared, time to toggle off).
                # Both want the same Left Three reminder.
                self._draw_static_demo(frame, None, "left_three",
                                        fallback_scale=0.62, now=now)
            if mouse_state is not None and mouse_state.get("camera_control_bounds") is not None:
                self._draw_tutorial_mouse_overlays(frame, mouse_state)
        elif step_key == "volume":
            # Volume tutorial has two phases:
            #   Phase A (raise + lower volume): show the animated
            #     volume-pose clip so the user can mirror the pose.
            #   Phase B (mute / unmute): swap to the static Mute.png
            #     so the user knows they're now on the mute pinch
            #     part of the step.
            tracker = getattr(self, "_volume_step_state", None) or {}
            up_done = bool(tracker.get("up_done"))
            down_done = bool(tracker.get("down_done"))
            mute_done = int(tracker.get("mute_count", 0)) >= 2
            if up_done and down_done and not mute_done:
                # On the mute phase — show the mute pose.
                self._draw_static_demo(frame, None, "mute_pose",
                                        fallback_scale=0.85, now=now)
            else:
                # Default / adjust phase — show the volume-pose
                # demo. Animated VolControl.mp4.
                self._draw_static_demo(frame, None, "volume_pose",
                                        fallback_scale=1.00, now=now)
        elif step_key == "voice_command":
            # Smaller inset for the Left Hand One pose: it's a static
            # reminder, not an animated demo, so it doesn't need to
            # dominate the corner.
            self._draw_static_demo(frame, None, "left_one",
                                    fallback_scale=0.62, now=now)
            self._draw_top_label(frame, "Voice", accent)
    def _drain_voice_queue(self) -> None:
        while True:
            try:
                request_id, payload = self._voice_queue.get_nowait()
            except queue.Empty:
                break
            if request_id != self._voice_request_id:
                continue
            event = str(payload.get("event", ""))
            if event == "status":
                self._voice_status = str(payload.get("status", "ready"))
                command_text = str(payload.get("command_text", "") or "").strip()
                if command_text:
                    self._voice_heard_text = command_text
                if self._voice_status == "listening":
                    self._voice_overlay_widget().show_listening()
                elif self._voice_status == "recognizing":
                    self._voice_overlay_widget().show_processing("Recognizing...")
                elif self._voice_status == "processing":
                    self._voice_overlay_widget().show_processing(
                        "Processing command...",
                        command_text=command_text or self._voice_heard_text,
                    )
            elif event == "result":
                success = bool(payload.get("success", False))
                heard_text = str(payload.get("heard_text", "") or "").strip()
                self._voice_heard_text = heard_text
                self._voice_listening = False
                self._voice_status = "ready"
                # Voice round done — drop the arrow so the user
                # isn't stuck staring at a "look at the mic" hint
                # after the listening phase has already ended.
                arrow = getattr(self, "_voice_mic_arrow", None)
                if arrow is not None:
                    arrow.hide_arrow()
                if success:
                    self._voice_overlay_widget().show_result("Executing command", command_text=heard_text, duration=1.9)
                else:
                    self._voice_overlay_widget().show_result("Command not understood", command_text=heard_text, duration=1.9)
                normalized = heard_text.lower()
                if self._practice_steps[self._step_index].key == "voice_command":
                    spotify_present = bool(getattr(self, "_has_spotify", False))
                    media_keyword = (
                        (spotify_present and ("spotify" in normalized or "play" in normalized))
                        or (not spotify_present and ("youtube" in normalized or "you tube" in normalized))
                    )
                    if media_keyword:
                        # Wait briefly so Spotify has time to start
                        # the track before we poll its playback state.
                        QTimer.singleShot(3000, self._check_voice_media_playing)
                    elif success:
                        # The voice command was recognized AND executed,
                        # but the heard text didn't include a Spotify /
                        # YouTube keyword — e.g. user said "open
                        # Notepad". Treat that as a successful voice-
                        # command step too: the user has demonstrated
                        # the voice listener works.
                        if not self._step_completed:
                            self._complete_step(
                                f"Voice command executed. Part {self._step_index + 1}/5 completed!"
                            )
                            self._reclaim_tutorial_focus(delay_ms=300)

    def _check_voice_media_playing(self) -> None:
        """First-pass media-played check. Runs the Spotify Web API
        call on a BACKGROUND thread because get_playback_state()
        blocks for 100–300 ms while it does its HTTP round-trip —
        on the GUI thread that stalled the live view long enough
        that right-swipe-to-advance couldn't be detected and the
        user had to click Next manually.

        Result is posted back to the GUI thread via
        `_voice_media_check_signal`, which then completes the step
        or schedules the final-attempt timer."""
        if self._step_completed:
            return
        if self._practice_steps[self._step_index].key != "voice_command":
            return
        spotify_present = bool(getattr(self, "_has_spotify", False))

        def _worker(final_pass: bool) -> None:
            playing = False
            try:
                if spotify_present:
                    playing = self._is_spotify_playing_safe()
                else:
                    playing = self._chrome_controller.has_youtube_open()
            except Exception:
                playing = False
            try:
                self._voice_media_check_signal.emit(bool(playing), bool(final_pass))
            except Exception:
                pass

        threading.Thread(target=_worker, args=(False,), daemon=True, name="voice-media-check").start()

    def _check_voice_media_playing_final(self) -> None:
        """Same as the first pass but with `final_pass=True` so the
        receiving slot marks the step done unconditionally when the
        worker reports no media detected."""
        if self._step_completed:
            return
        if self._practice_steps[self._step_index].key != "voice_command":
            return
        spotify_present = bool(getattr(self, "_has_spotify", False))

        def _worker() -> None:
            playing = False
            try:
                if spotify_present:
                    playing = self._is_spotify_playing_safe()
                else:
                    playing = self._chrome_controller.has_youtube_open()
            except Exception:
                playing = False
            try:
                self._voice_media_check_signal.emit(bool(playing), True)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True, name="voice-media-check-final").start()

    def _on_voice_media_check_result(self, playing: bool, final_pass: bool) -> None:
        """Slot for the off-thread media-playing check. Runs on the
        GUI thread (signal/slot auto-marshals). Completes the step
        if media is playing; otherwise either schedules the
        second-pass check (final_pass=False) or settles for "voice
        command was heard" (final_pass=True)."""
        if self._step_completed:
            return
        if self._practice_steps[self._step_index].key != "voice_command":
            return
        spotify_present = bool(getattr(self, "_has_spotify", False))
        if playing:
            if spotify_present:
                self._complete_step(f"Spotify is playing! Part {self._step_index + 1}/5 completed!")
            else:
                # Tutorial-only: flip the engine into YouTube forced
                # mode now that the user has YouTube actually playing.
                # The next practice step (gesture wheel) and the
                # voice/swipe controls then behave the way the user
                # would normally only get after holding the 'four'
                # pose, which is one less thing for them to discover
                # mid-tutorial. The engine clears this in
                # set_tutorial_context(False) on tutorial close.
                worker = self._worker
                if worker is not None:
                    try:
                        worker.force_youtube_mode_for_tutorial()
                    except Exception:
                        pass
                self._complete_step(f"YouTube opened in Chrome! Part {self._step_index + 1}/5 completed!")
            # Spotify / Chrome just stole focus by launching or
            # waking up. Reclaim focus for the tutorial so the
            # user can see the completion + continue gesturing
            # without clicking back manually.
            self._reclaim_tutorial_focus(delay_ms=300)
            return
        if final_pass:
            self._complete_step(f"Voice command detected. Part {self._step_index + 1}/5 completed!")
            self._reclaim_tutorial_focus(delay_ms=300)
            return
        # First pass didn't see media yet — give it another 2 s
        # and try the final-pass check.
        QTimer.singleShot(2000, self._check_voice_media_playing_final)

    def _maybe_periodic_spotify_poll(self, now: float) -> None:
        """Per-frame callable: spawn a background Spotify-playback
        check at most every ~3 s while the user is on the
        voice_command step with Spotify present. The result slot
        only completes the step on playing=True, so polling can't
        time-out the user out of the step."""
        if self._step_completed:
            return
        if self._practice_steps[self._step_index].key != "voice_command":
            return
        if not bool(getattr(self, "_has_spotify", False)):
            return
        if self._spotify_poll_inflight:
            return
        if now - self._spotify_poll_last_at < 3.0:
            return
        self._spotify_poll_last_at = now
        self._spotify_poll_inflight = True

        def _worker() -> None:
            playing = False
            try:
                playing = self._is_spotify_playing_safe()
            except Exception:
                playing = False
            try:
                self._spotify_periodic_poll_signal.emit(bool(playing))
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True, name="spotify-periodic-poll").start()

    def _on_spotify_periodic_poll(self, playing: bool) -> None:
        """GUI-thread slot for the periodic Spotify poll. Completes
        the step ONLY when the playback API confirms a song is
        playing; otherwise just clears the in-flight latch so the
        next tick can fire."""
        self._spotify_poll_inflight = False
        if not playing:
            return
        if self._step_completed:
            return
        if self._practice_steps[self._step_index].key != "voice_command":
            return
        self._complete_step(f"Spotify is playing! Part {self._step_index + 1}/5 completed!")
        # Spotify just stole focus by launching / waking up. Reclaim
        # focus to the tutorial window so the user can see the step
        # completion + continue gesturing without having to click
        # back. A 250 ms delay gives Spotify time to finish its own
        # focus grab before we override it (otherwise the race goes
        # the other way and Spotify wins on slower machines).
        self._reclaim_tutorial_focus(delay_ms=250)

    def _focus_guard_tick(self) -> None:
        """Per-tick (500 ms) check: if the foreground window isn't
        the tutorial dialog, force it back. Runs only while the
        voice_command step is active and the step hasn't completed
        — Spotify / Chrome are the only windows that aggressively
        steal focus from inside this step's flow.

        Performance: GetForegroundWindow is a single CRITICAL_SECTION
        + atomic-read in user32 — order of microseconds. We compare
        an int against our own winId(), then bail. Only the
        actual-mismatch path calls SetForegroundWindow, so when the
        tutorial IS focused (the steady state) the tick is
        essentially free."""
        try:
            if self._step_completed:
                return
            if self._show_completion_page:
                return
            step_key = self._practice_steps[self._step_index].key
            if step_key != "voice_command":
                # Defensive: timer should have been stopped on step
                # change; if it wasn't, no-op rather than thrash.
                return
            if not self.isVisible():
                return
            import ctypes
            user32 = ctypes.windll.user32
            foreground_hwnd = int(user32.GetForegroundWindow())
            if foreground_hwnd == 0:
                return
            try:
                our_hwnd = int(self.winId())
            except Exception:
                return
            if foreground_hwnd == our_hwnd:
                return
            # Also tolerate other top-level Touchless windows holding
            # focus (e.g. the parent MainWindow). The user only cares
            # that some Touchless window is foreground while they're
            # on this step — not specifically the tutorial. Walking
            # the parent chain matches Qt's "we own this thread"
            # check without the cost.
            try:
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None:
                    for widget in app.topLevelWidgets():
                        try:
                            if int(widget.winId()) == foreground_hwnd:
                                # Foreground IS some Touchless
                                # window — fine, don't override.
                                return
                        except Exception:
                            continue
            except Exception:
                pass
            # Reclaim. Use SetForegroundWindow directly because Qt's
            # activateWindow can no-op when another app held focus
            # within the last second on Windows.
            user32.SetForegroundWindow(our_hwnd)
            try:
                self.raise_()
                self.activateWindow()
            except Exception:
                pass
        except Exception:
            # Never let the guard tick raise — it must stay cheap
            # and silent for the polling cadence to be acceptable.
            pass

    def _reclaim_tutorial_focus(self, *, delay_ms: int = 250) -> None:
        """Force the tutorial window back to the foreground after a
        focus-stealing operation (Spotify launch / playback toggle).
        Called from the periodic Spotify poll and the voice-command
        completion paths during the voice_command tutorial step.

        Uses Win32 SetForegroundWindow directly because Qt's
        activateWindow() is often a silent no-op when another app
        just stole focus — Windows' anti-focus-stealing protections
        require the foreground change to be authorised by the
        kernel, which happens here because we ARE the previous
        foreground (Touchless → Spotify → Touchless within seconds)."""
        def _do_reclaim() -> None:
            try:
                if self.isMinimized():
                    self.showNormal()
                else:
                    self.show()
                self.raise_()
                self.activateWindow()
                # Belt-and-suspenders Win32 SetForegroundWindow.
                # Qt's activateWindow can no-op on Windows when
                # another app held focus recently — the explicit
                # SetForegroundWindow call wins because Windows
                # remembers we were the foreground a moment ago.
                try:
                    import ctypes
                    hwnd = int(self.winId())
                    if hwnd:
                        ctypes.windll.user32.SetForegroundWindow(hwnd)
                except Exception:
                    pass
            except Exception:
                pass
        try:
            QTimer.singleShot(max(0, int(delay_ms)), _do_reclaim)
        except Exception:
            _do_reclaim()

    def _set_engine_tutorial_mute_block(self, blocked: bool) -> None:
        """Tell the gesture engine to drop the shaka-mute trigger
        while the tutorial volume step is in its 'practise up/down
        first' phase. Engine-side gate prevents the OS audio from
        actually muting if the user does the shaka pose early, so
        the experience matches the tutorial's progress display."""
        worker = self._resolve_parent_worker()
        if worker is None:
            return
        setter = getattr(worker, "set_tutorial_block_mute", None)
        if callable(setter):
            try:
                setter(bool(blocked))
            except Exception:
                pass

    def _is_spotify_playing_safe(self) -> bool:
        """Wrap SpotifyController.get_playback_state in a try/except
        so a missing token / offline network doesn't crash the
        tutorial — returns False in that case."""
        try:
            state = self._spotify_controller.get_playback_state()
            return state is True
        except Exception:
            return False

    def _try_adopt_parent_voice_listener(self) -> None:
        worker = self._resolve_parent_worker()
        if worker is None:
            return
        parent_listener = getattr(worker, "voice_listener", None)
        if parent_listener is not None and parent_listener is not self._voice_listener:
            self._voice_listener = parent_listener
        # Also re-wire phone mic on the (possibly newly-adopted)
        # listener — adopting the parent's listener doesn't itself
        # carry over set_external_audio_source state if the parent
        # configured it later.
        try:
            self._wire_phone_mic_to_voice_listener()
        except Exception:
            pass

    def _wire_phone_mic_to_voice_listener(self) -> None:
        """Plumb the parent MainWindow's PhoneCameraServer audio source
        into our voice listener when the user has the phone-as-mic
        toggle on. Safe to call multiple times — idempotent on the
        listener side."""
        listener = getattr(self, "_voice_listener", None)
        if listener is None or not hasattr(listener, "set_external_audio_source"):
            return
        # Need the parent MainWindow's QR server to read its
        # PhoneAudioSource. Walk up the parent chain.
        owner = self.parent()
        while owner is not None and not hasattr(owner, "_phone_camera_qr_server"):
            owner = owner.parent() if hasattr(owner, "parent") else None
        if owner is None:
            return
        use_phone_mic = (
            bool(getattr(self.config, "phone_camera_qr_use_mic", False))
            and bool(getattr(self.config, "phone_camera_qr_paired", False))
        )
        qr_server = getattr(owner, "_phone_camera_qr_server", None)
        if not use_phone_mic or qr_server is None:
            try:
                listener.set_external_audio_source(None)
            except Exception:
                pass
            return
        try:
            listener.set_external_audio_source(qr_server.audio_source)
        except Exception:
            pass

    def _update_voice_arrow(self, currently_listening: bool) -> None:
        """Show/hide the bouncing voice-mic arrow based on the current
        voice-listening state. Driven from BOTH the owned-worker
        voice path (which sets self._voice_listening directly) and
        the shared-worker path (which reads voice_listening from the
        worker's per-frame payload). Without this, the arrow only
        showed on the owned-worker path because _start_voice_practice
        is bypassed when the tutorial uses the parent app's worker
        — exactly the case the field user reported ("started
        tutorial with the app already running, never saw the arrow")."""
        arrow = getattr(self, "_voice_mic_arrow", None)
        if arrow is None:
            return
        prev = getattr(self, "_arrow_prev_voice_listening", False)
        self._arrow_prev_voice_listening = bool(currently_listening)
        if currently_listening and not prev:
            arrow.show_pointing()
        elif prev and not currently_listening:
            arrow.hide_arrow()

    def _start_voice_practice(self) -> None:
        self._try_adopt_parent_voice_listener()
        if self._voice_listening:
            return
        self._voice_listening = True
        self._voice_status = "listening"
        self._voice_heard_text = ""
        self._voice_request_id += 1
        request_id = self._voice_request_id
        self._voice_overlay_widget().show_listening()
        # Show the bouncing arrow on the tutorial window pointing to
        # the mic overlay. Only meaningful during the part-6 voice
        # step — the gesture that triggers _start_voice_practice
        # only fires inside that step, so no extra guard needed.
        arrow = getattr(self, "_voice_mic_arrow", None)
        if arrow is not None:
            arrow.show_pointing()

        def _status_callback(status: str) -> None:
            self._voice_queue.put((request_id, {"event": "status", "status": status}))

        def _worker() -> None:
            result = self._voice_listener.listen(
                max_seconds=6.0,
                status_callback=_status_callback,
                transcript_mode="command",
            )
            heard_text = result.heard_text
            if heard_text:
                self._voice_queue.put((request_id, {"event": "status", "status": "processing", "command_text": heard_text}))
                execution = self._voice_processor.execute(
                    heard_text,
                    context=VoiceCommandContext(preferred_app="chrome"),
                )
                display_text = execution.display_text or heard_text
            else:
                display_text = heard_text
            self._voice_queue.put(
                (
                    request_id,
                    {
                        "event": "result",
                        "success": bool(result.success),
                        "heard_text": display_text,
                    },
                )
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_worker_debug_frame(self, frame, payload) -> None:
        # First frame in -> camera is up -> kill the starting pill.
        # Hide on EVERY frame is a cheap no-op (just checks visible).
        self._hide_starting_pill()
        monotonic_now = time.monotonic()
        current_step_key = self._practice_steps[self._step_index].key
        visual_ready = self._update_step_progress_from_payload(payload, monotonic_now)
        self._tutorial_nav_from_payload(payload, monotonic_now)
        self._update_low_fps_warning(payload, monotonic_now, current_step_key)

        display = frame.copy() if frame is not None else None
        if display is not None:
            result = payload.get("result")
            if result is not None:
                skeleton_color = (80, 235, 120) if visual_ready else (70, 70, 255)
                self._draw_user_skeleton_overlay(display, result, skeleton_color)
            self._draw_demo_overlay(display, current_step_key)
            self._draw_encouragement_overlay(display, monotonic_now)
            self._render_frame(display)
        if current_step_key != "gesture_wheel":
            self._tutorial_wheel_overlay.hide_overlay()

        dynamic_label = str(payload.get("dynamic_label", "neutral") or "neutral")
        stable_label = str(payload.get("stable_label", "neutral") or "neutral")
        raw_label = str(payload.get("raw_label", "neutral") or "neutral")
        banner = stable_label if stable_label != "neutral" else raw_label
        if self._practice_steps[self._step_index].key == "mouse_mode" and payload.get("mouse_mode_enabled"):
            self.gesture_chip.setText("Mouse mode active")
        elif dynamic_label != "neutral":
            self.gesture_chip.setText(f"Dynamic: {dynamic_label.replace('_', ' ')}")
        else:
            self.gesture_chip.setText(f"Gesture: {banner}")

        info_lines = payload.get("info_lines") or []
        if info_lines:
            self.camera_label.setText(str(info_lines[0]))
        self._check_encouragement_trigger(monotonic_now)
        self._update_encouragement_visual(monotonic_now)
        self._update_completion_feedback(monotonic_now)

    def _update_low_fps_warning(self, payload: dict, now: float, step_key: str) -> None:
        if step_key != "swipes":
            self.low_fps_warning_label.hide()
            self._low_fps_warning_since = None
            return
        if bool(payload.get("low_fps_active")):
            self.low_fps_warning_label.hide()
            self._low_fps_warning_since = None
            return
        fps = float(payload.get("fps", 0.0) or 0.0)
        if 0.0 < fps < 20.0:
            if self._low_fps_warning_since is None:
                self._low_fps_warning_since = now
            elif (now - self._low_fps_warning_since) >= 2.5:
                self.low_fps_warning_label.show()
        else:
            self._low_fps_warning_since = None
            self.low_fps_warning_label.hide()

    def _update_step_progress_from_payload(self, payload: dict, now: float) -> bool:
        result = payload.get("result")
        if result is None:
            return False

        step = self._practice_steps[self._step_index]
        handedness = str(payload.get("handedness", "") or "").lower()
        dynamic_label = str(payload.get("dynamic_label", "neutral") or "neutral")
        stable_label = str(payload.get("stable_label", "neutral") or "neutral")
        raw_label = str(payload.get("raw_label", "neutral") or "neutral")
        confidence = float(payload.get("confidence", 0.0) or 0.0)

        if step.key == "swipes":
            if self._swipe_goal_index >= 6:
                self._swipe_counts["swipe_right"] = min(3, self._swipe_counts["swipe_right"])
                self._swipe_counts["swipe_left"] = min(3, self._swipe_counts["swipe_left"])
                self.swipe_widget.set_counts(self._swipe_counts["swipe_left"], self._swipe_counts["swipe_right"])
                self._refresh_swipe_camera_labels()
                return now < self._visual_green_until.get("swipes", 0.0)

            expected = "swipe_right" if self._swipe_goal_index < 3 else "swipe_left"
            accepted_swipe = False
            if dynamic_label != self._last_dynamic_label and handedness == "right" and dynamic_label == expected:
                self._swipe_goal_index = min(6, self._swipe_goal_index + 1)
                self._swipe_counts[dynamic_label] = min(3, self._swipe_counts[dynamic_label] + 1)
                self._swipe_last_advance_at = now
                self.swipe_widget.set_counts(self._swipe_counts["swipe_left"], self._swipe_counts["swipe_right"])
                self._visual_green_until["swipes"] = max(self._visual_green_until.get("swipes", 0.0), now + self._gesture_flash_seconds)
                accepted_swipe = True
                # No-Spotify path: send the OS media-track keys so
                # YouTube / Edge / VLC respond to the swipe as a
                # next/previous-track hint. Single videos won't
                # actually advance (no playlist context), but any
                # media app with a queue will, and the gesture fires
                # the visual confirmation regardless. Spotify users
                # don't need this -- the engine's spotify_router
                # already routes swipes through the Web API.
                if not self._has_spotify:
                    VK_MEDIA_NEXT_TRACK = 0xB0
                    VK_MEDIA_PREV_TRACK = 0xB1
                    self._send_media_key(
                        VK_MEDIA_NEXT_TRACK
                        if dynamic_label == "swipe_right"
                        else VK_MEDIA_PREV_TRACK
                    )
                # Each successful swipe is its own action-completion
                # event — fire encouragement now (throttle handles
                # the burst case).
                self._trigger_encouragement(now)
            self._last_dynamic_label = dynamic_label
            if self._swipe_goal_index >= 6:
                self._complete_step("Both swipes detected! Swipe right to move on!")
            self._refresh_swipe_camera_labels()
            visual_ready = accepted_swipe or now < self._visual_green_until.get("swipes", 0.0)
            return visual_ready

        if step.key == "spotify_open":
            active = handedness == "right" and (
                stable_label == "two" or (raw_label == "two" and confidence >= 0.56)
            )
            self._flash_on_edge("spotify_open", active, now)
            visual_ready = now < self._visual_green_until.get("spotify_open", 0.0)
            hold_fired = self._hold_ready(
                "tutorial_spotify_open",
                active,
                self._spotify_open_hold_seconds,
                now,
                cooldown=self._spotify_static_cooldown_seconds,
            )
            if hold_fired:
                visual_ready = True
                self._complete_step("Completed! Swipe right to move on!")
                # The engine launches Spotify in the BACKGROUND
                # (hidden=True) during this tutorial step so it
                # doesn't steal focus. Belt-and-suspenders: poll
                # the foreground window for the next ~3 s and
                # re-focus the tutorial instantly if Spotify
                # (or its splash) does pop to the front anyway.
                self._start_tutorial_refocus_guard()
            if self._step_completed:
                # Once Spotify opens, the user just needs to swipe to
                # advance — no value in repeating "Detected right-hand
                # two!" since they're already past that gate.
                self._set_step_progress("Swipe right to move on!")
            else:
                self._set_step_progress("Detected right-hand two!" if active else "Waiting for right-hand two.")
            return visual_ready

        if step.key == "play_pause":
            active = handedness == "right" and (
                stable_label == "fist" or (raw_label == "fist" and confidence >= 0.52)
            )
            self._flash_on_edge("play_pause", active, now)
            if not active:
                self._play_pause_ready_for_next = True
            visual_ready = now < self._visual_green_until.get("play_pause", 0.0)
            if self._play_pause_ready_for_next and self._hold_ready(
                "tutorial_play_pause_count",
                active,
                self._spotify_play_pause_hold_seconds,
                now,
                cooldown=self._spotify_static_cooldown_seconds,
            ):
                self._spotify_toggle_count = min(2, self._spotify_toggle_count + 1)
                self._play_pause_ready_for_next = False
                self._trigger_encouragement(now)
                visual_ready = True
                # No-Spotify path: send the OS media-play/pause key
                # so YouTube / Edge / VLC / anything that registers
                # as a Windows media controller toggles. The engine's
                # spotify_controller.toggle_playback() path is a
                # no-op without the desktop app installed; without
                # this fallback the user's fist would do nothing
                # visible.
                if not self._has_spotify:
                    VK_MEDIA_PLAY_PAUSE = 0xB3
                    self._send_media_key(VK_MEDIA_PLAY_PAUSE)
                # Toggling Spotify playback can pop the Spotify
                # desktop client to the foreground (its window
                # manager reacts to play/pause Web API commands by
                # raising the main window in some Spotify versions).
                # Kick the refocus guard so the tutorial stays in
                # front for the next ~3 s — same belt-and-suspenders
                # we already use for the spotify_open step.
                self._start_tutorial_refocus_guard()
            self._set_step_progress(self._fist_progress_html())
            if self._spotify_toggle_count >= 2:
                self._complete_step("Completed! Swipe right to move on!")
            return visual_ready

        if step.key == "volume":
            # Volume practice: detect mode entry + at least one up
            # swing AND down swing AND two mute toggles. The engine
            # emits volume_active, volume_level_scalar, and
            # volume_muted in the debug payload; we just watch for
            # transitions.
            volume_active = bool(payload.get("volume_active"))
            level_raw = payload.get("volume_level_scalar")
            try:
                level = float(level_raw) if level_raw is not None else None
            except Exception:
                level = None
            muted = bool(payload.get("volume_muted"))
            tracker = getattr(self, "_volume_step_state", None)
            if tracker is None:
                tracker = {
                    "engaged": False,
                    # Volume swing detection now tracks EXTREMES from
                    # the anchor level captured on first engagement,
                    # not per-frame delta. So a user who goes down and
                    # then back up gets credit for both directions even
                    # though their net change is zero. 5 % swing
                    # threshold in either direction.
                    "anchor_level": None,
                    "max_seen": None,
                    "min_seen": None,
                    "up_done": False,
                    "down_done": False,
                    "last_muted": muted,
                    # Two mute toggles required (mute, then unmute)
                    # so the user practises both directions — mirrors
                    # the play/pause step which requires two fists.
                    "mute_count": 0,
                }
                self._volume_step_state = tracker
                # Tell the engine to BLOCK system mute toggles until
                # up + down are both complete, so the user can't
                # accidentally hit shaka first and have the OS audio
                # mute before they've practised volume up / down.
                self._set_engine_tutorial_mute_block(True)
            mute_target = 2
            SWING_THRESHOLD = 0.05  # 5 % of full volume range
            if volume_active and not tracker["engaged"]:
                tracker["engaged"] = True
                if level is not None:
                    tracker["anchor_level"] = level
                    tracker["max_seen"] = level
                    tracker["min_seen"] = level
            if volume_active and level is not None and tracker["anchor_level"] is not None:
                # Track running max / min so a single take that goes
                # down THEN up scores both directions.
                if tracker["max_seen"] is None or level > tracker["max_seen"]:
                    tracker["max_seen"] = level
                if tracker["min_seen"] is None or level < tracker["min_seen"]:
                    tracker["min_seen"] = level
                swing_up = float(tracker["max_seen"]) - float(tracker["anchor_level"])
                swing_down = float(tracker["anchor_level"]) - float(tracker["min_seen"])
                if swing_up >= SWING_THRESHOLD and not tracker["up_done"]:
                    tracker["up_done"] = True
                    self._trigger_encouragement(now)
                if swing_down >= SWING_THRESHOLD and not tracker["down_done"]:
                    tracker["down_done"] = True
                    self._trigger_encouragement(now)
            # Mute toggle counting. ONLY counts when both directions
            # are complete, so a user holding the shaka before they've
            # practised volume up/down doesn't get to skip the swing
            # half of the step. Engine-side block (set above) prevents
            # the OS audio from actually muting too.
            volume_done = bool(tracker["up_done"] and tracker["down_done"])
            if volume_done and muted != tracker["last_muted"]:
                tracker["mute_count"] = min(mute_target, tracker.get("mute_count", 0) + 1)
                self._trigger_encouragement(now)
            # last_muted always tracks the current state so we don't
            # double-count a pre-swing toggle once the gate opens.
            tracker["last_muted"] = muted
            # Once swings are done, lift the engine-side mute block.
            if volume_done:
                self._set_engine_tutorial_mute_block(False)
            mute_count = int(tracker.get("mute_count", 0))
            mute_done = mute_count >= mute_target
            # Skeleton turns green when the user is doing either the
            # volume pose (raise/lower) OR the mute shaka — earlier
            # only volume_active triggered green, so during the shaka
            # half of the step the skeleton flashed red even though
            # the user was doing the correct gesture. Stable label
            # 'mute' is what the static recognizer fires for a held
            # shaka with palm-to-camera; checking raw_label too gives
            # us a frame or two of green during transitions before
            # stability latches.
            mute_pose_held = (
                stable_label == "mute"
                or (raw_label == "mute" and confidence >= 0.50)
            )
            visual_ready = volume_active or mute_pose_held
            # Header switches once both raise + lower are done: the
            # instruction at the top of the camera view changes from
            # 'hold the volume pose' to the shaka instruction so the
            # user knows the step has moved on to the mute half. We
            # also shorten the footer in the mute phase to just the
            # counter (the shaka instruction is now in the header).
            if tracker["up_done"] and tracker["down_done"]:
                self._set_camera_header_text(
                    "Do a shaka with palm towards the monitor to mute."
                )
            else:
                self._set_camera_header_text("Hold the volume pose to adjust.")
            if not tracker["engaged"]:
                self._set_step_progress("Hold the volume pose to begin.")
            elif not (tracker["up_done"] and tracker["down_done"]):
                progress_bits = []
                progress_bits.append("✓" if tracker["up_done"] else "—")
                progress_bits.append(" raise volume   ")
                progress_bits.append("✓" if tracker["down_done"] else "—")
                progress_bits.append(" lower volume")
                self._set_step_progress("".join(progress_bits))
            elif not mute_done:
                # Colored counter like the fist step: red 0/2 →
                # orange 1/2 → green 2/2. The shaka instruction lives
                # in the camera header now (set above) — keep the
                # footer to just the counter so the two pieces of
                # information don't duplicate each other.
                color = self._progress_color(mute_count, mute_target)
                self._set_step_progress(
                    f"Mute toggles <span style='color:{color};'>{mute_count}/{mute_target}</span>"
                )
            else:
                self._set_step_progress("All three! Swipe right to move on!")
            if (
                tracker["up_done"]
                and tracker["down_done"]
                and mute_done
                and not self._step_completed
            ):
                self._complete_step("Completed! Swipe right to move on!")
            return visual_ready

        if step.key == "gesture_wheel":
            active = handedness == "right" and (
                stable_label in {"wheel_pose", "chrome_wheel_pose"} or (raw_label in {"wheel_pose", "chrome_wheel_pose"} and confidence >= 0.50)
            )
            wheel_visible = bool(payload.get("wheel_visible"))
            self._flash_on_edge("gesture_wheel", active, now)
            visual_ready = now < self._visual_green_until.get("gesture_wheel", 0.0)
            if self._step_completed:
                self._set_step_progress("Wheel pose detected! Swipe right to move on.")
            else:
                self._set_step_progress(
                    "Wheel pose detected!" if (active or wheel_visible) else "Waiting for wheel pose."
                )
            if active and not self._step_completed:
                self._visual_green_until["gesture_wheel"] = max(self._visual_green_until.get("gesture_wheel", 0.0), now + self._gesture_flash_seconds)
                visual_ready = True
                self._complete_step("Completed! Swipe right to move on!")
            self._tutorial_wheel_overlay.hide_overlay()
            return visual_ready

        if step.key == "mouse_mode":
            mouse_mode_enabled = bool(payload.get("mouse_mode_enabled"))
            cursor_position = payload.get("mouse_cursor_position")
            # The pinch redesign moved the canonical click signal: a
            # pinch-down emits left_press (no synthetic left_click).
            # We treat either as "register a click" in the practice
            # arena so both pre- and post-redesign streamers work.
            left_click = bool(payload.get("mouse_left_click")) or bool(payload.get("mouse_left_press"))
            # Mirror the engine's mouse-overlay state into a tutorial-
            # level field so _draw_demo_overlay can render the red
            # control-area box (same overlay the regular live view
            # shows). Without this the tutorial's local _mouse_tracker
            # never sees an .update() call in the engine-driven path,
            # so its mode_enabled stays False and the box never
            # rendered.
            if mouse_mode_enabled:
                self._payload_mouse_state = {
                    "mode_enabled": True,
                    "camera_control_bounds": payload.get("mouse_camera_control_bounds"),
                    "cursor_position": cursor_position,
                    "virtual_bounds": payload.get("mouse_virtual_bounds"),
                }
            else:
                self._payload_mouse_state = None
            left_three_active = handedness == "left" and (
                stable_label == "three" or (raw_label == "three" and confidence >= 0.56)
            )
            self.mouse_widget.set_mode_enabled(mouse_mode_enabled)
            self.mouse_widget.set_cursor_position(cursor_position)
            if left_click:
                self.mouse_widget.register_click(cursor_position)
            self._set_camera_header_text(
                self._mouse_mode_header_text(
                    mode_enabled=mouse_mode_enabled,
                    completed_targets=self.mouse_widget.completed_targets,
                )
            )
            if self._mouse_stage == "enable":
                self._set_step_progress(
                    "Detected left-hand three!" if left_three_active else "Mouse mode off. Turn it on to begin."
                )
                self._flash_on_edge("mouse_enable", left_three_active, now)
                visual_ready = now < self._visual_green_until.get("mouse_enable", 0.0)
                if self._hold_ready(
                    "mouse_enable_hold",
                    left_three_active,
                    self._mouse_tracker.toggle_hold_seconds,
                    now,
                    cooldown=self._mouse_tracker.toggle_cooldown_seconds,
                ):
                    visual_ready = True
                if mouse_mode_enabled:
                    self._mouse_stage = "practice"
            elif self._mouse_stage == "practice":
                self._set_step_progress("Mouse mode on. Clear all tutorial targets.")
                visual_ready = False
                if self.mouse_widget.completed:
                    # Transition into the scroll phase. Swap the
                    # right-panel widget to the scroll practice arena
                    # and reset its state in case the user navigated
                    # away and came back.
                    self._mouse_stage = "scroll"
                    try:
                        self.mouse_scroll_widget.reset()
                        self.practice_stack.setCurrentWidget(self.mouse_scroll_widget)
                    except Exception:
                        pass
            elif self._mouse_stage == "scroll":
                # Progressive footer copy per UX spec:
                #   neither end reached → "Get to the top of the page on the right."
                #   top reached, bottom not → "Now get to the bottom!"
                #   both reached → handled by the disable-phase
                #     branch below (we'll have already transitioned)
                if self.mouse_scroll_widget.top_reached and not self.mouse_scroll_widget.bottom_reached:
                    self._set_step_progress("Now get to the bottom!")
                elif self.mouse_scroll_widget.bottom_reached and not self.mouse_scroll_widget.top_reached:
                    self._set_step_progress("Now scroll up to the top!")
                else:
                    self._set_step_progress(
                        "Get to the top of the page on the right."
                    )
                visual_ready = False
                # Translate scroll-pose steps emitted by the engine
                # into the practice arena's scrollbar movement.
                # mouse_scroll_steps is positive for scroll-up,
                # negative for scroll-down (matches the engine's
                # mouse_controller.scroll(steps) signature).
                steps = int(payload.get("mouse_scroll_steps", 0) or 0)
                if steps != 0:
                    try:
                        self.mouse_scroll_widget.register_scroll_steps(steps)
                    except Exception:
                        pass
                if self.mouse_scroll_widget.completed:
                    self._mouse_stage = "disable"
                    # Swap back to the mouse_widget so the disable-
                    # phase paints the standard "✓ targets cleared"
                    # overlay over the original click arena.
                    try:
                        self.practice_stack.setCurrentWidget(self.mouse_widget)
                    except Exception:
                        pass
            else:
                # Scroll phase is done by the time we hit disable;
                # tell the user the practice piece is complete and
                # the only thing left is to toggle mouse mode off.
                self._set_step_progress(
                    "Detected left-hand three!" if left_three_active
                    else "You've completed the mouse tutorial! Turn mouse mode off to finish."
                )
                self._flash_on_edge("mouse_disable", left_three_active, now)
                visual_ready = now < self._visual_green_until.get("mouse_disable", 0.0)
                if self._hold_ready(
                    "mouse_disable_hold",
                    left_three_active,
                    self._mouse_tracker.toggle_hold_seconds,
                    now,
                    cooldown=self._mouse_tracker.toggle_cooldown_seconds,
                ):
                    visual_ready = True
                if not mouse_mode_enabled and self.mouse_widget.completed and self.mouse_scroll_widget.completed:
                    self._complete_step("Completed! Swipe right to move on!")
            if mouse_mode_enabled and cursor_position is not None:
                self._mouse_cursor_seen = True
            return visual_ready

        if step.key == "voice_command":
            left_one_active = handedness == "left" and (
                stable_label == "one" or (raw_label == "one" and confidence >= 0.56)
            )
            voice_listening = bool(payload.get("voice_listening"))
            voice_heard = str(payload.get("voice_heard_text", "") or "").lower()
            voice_control = str(payload.get("voice_control_text", "") or "").lower()
            # Drive the bouncing arrow from the worker's per-frame
            # voice_listening flag so the cue appears even when the
            # tutorial is running against the parent app's worker
            # (the path that bypasses _start_voice_practice entirely).
            self._update_voice_arrow(voice_listening)
            self._flash_on_edge("voice_command", left_one_active, now)
            visual_ready = now < self._visual_green_until.get("voice_command", 0.0)
            self._set_camera_header_text(self._voice_command_header_text(left_one_active or voice_listening))
            self._set_step_progress(
                self._voice_command_progress_text(
                    left_one_active=left_one_active,
                    voice_listening=voice_listening,
                )
            )
            if (
                left_one_active
                and not voice_listening
                and self._worker is not None
                and self._hold_ready("tutorial_worker_voice_command", True, 0.6, now, cooldown=1.5)
            ):
                try:
                    self._worker._start_voice_command()
                except Exception:
                    pass
                visual_ready = True
            # Primary completion signal: _on_worker_command_detected set
            # _last_voice_success_text when a recognized voice command
            # executed. This is the reliable path (command_detected
            # always fires on a successful action) and is app-agnostic,
            # so it works whether the user said "play X on Spotify" or
            # opened YouTube on Chrome.
            if self._last_voice_success_text:
                self._complete_step("Completed! Swipe right to move on!")
                return visual_ready or voice_listening
            spotify_present = bool(getattr(self, "_has_spotify", False))
            if spotify_present:
                # Secondary signals when command_detected didn't carry a
                # marker: confirm via the playback poll, or the per-frame
                # heard/control text fallback.
                self._maybe_periodic_spotify_poll(now)
                if "spotify" in voice_heard and ("execut" in voice_control or "play" in voice_control):
                    self._complete_step("Completed! Swipe right to move on!")
            else:
                if "youtube" in voice_heard and "chrome" in voice_heard and ("execut" in voice_control or "chrome open" in voice_control):
                    self._complete_step("Completed! Swipe right to move on!")
            return visual_ready or voice_listening

        return False

    def _update_step_progress(self, result, now: float) -> bool:
        step = self._practice_steps[self._step_index]
        handedness = str(result.tracked_hand.handedness or "") if result.found and result.tracked_hand is not None else ""
        dynamic_label = result.prediction.dynamic_label
        visual_ready = False

        if step.key == "swipes":
            expected = "swipe_right" if self._swipe_goal_index < 3 else "swipe_left"
            accepted_swipe = False
            if dynamic_label != self._last_dynamic_label and handedness.lower() == "right" and dynamic_label == expected:
                self._swipe_goal_index += 1
                self._swipe_counts[dynamic_label] += 1
                self._swipe_last_advance_at = now
                self.swipe_widget.set_counts(self._swipe_counts["swipe_left"], self._swipe_counts["swipe_right"])
                self._visual_green_until["swipes"] = max(self._visual_green_until.get("swipes", 0.0), now + 1.0)
                accepted_swipe = True
            self._last_dynamic_label = dynamic_label
            if self._swipe_goal_index >= 6:
                self._complete_step("Completed! Swipe right to move on!")
            self._refresh_swipe_camera_labels()
            visual_ready = accepted_swipe or now < self._visual_green_until.get("swipes", 0.0)
            return visual_ready

        if step.key == "spotify_open":
            active = bool(result.found and result.tracked_hand is not None and str(result.tracked_hand.handedness or "").lower() == "right" and result.prediction.stable_label == "two")
            visual_ready = now < self._visual_green_until.get("spotify_open", 0.0)
            if self._hold_ready("spotify_open", active, self._spotify_open_hold_seconds, now, cooldown=self._spotify_static_cooldown_seconds):
                self._complete_step("Completed! Swipe right to move on!")
                visual_ready = True
            if self._step_completed:
                self._set_step_progress("Swipe right to move on!")
            else:
                self._set_step_progress("Detected right-hand two!" if active else "Waiting for right-hand two.")
            return visual_ready

        if step.key == "play_pause":
            active = bool(result.found and result.tracked_hand is not None and str(result.tracked_hand.handedness or "").lower() == "right" and result.prediction.stable_label == "fist")
            if not active:
                self._play_pause_ready_for_next = True
            visual_ready = now < self._visual_green_until.get("play_pause", 0.0)
            self._set_step_progress("Detected right-hand fist!" if active else self._fist_progress_html())
            if self._play_pause_ready_for_next and self._hold_ready("play_pause", active, self._spotify_play_pause_hold_seconds, now, cooldown=self._spotify_static_cooldown_seconds):
                self._spotify_toggle_count = min(2, self._spotify_toggle_count + 1)
                self._play_pause_ready_for_next = False
                self._trigger_encouragement(now)
                self._set_step_progress(self._fist_progress_html())
                if self._spotify_toggle_count >= 2:
                    self._complete_step("Completed! Swipe right to move on!")
                visual_ready = True
                # Mirrors the engine-driven path above (line ~2786):
                # Spotify can pop to the foreground on play/pause Web
                # API commands. Snap focus back to the tutorial.
                self._start_tutorial_refocus_guard()
            return visual_ready

        if step.key == "gesture_wheel":
            active = bool(
                result.found and result.tracked_hand is not None and str(result.tracked_hand.handedness or "").lower() == "right" and (
                    result.prediction.stable_label == "wheel_pose" or (result.prediction.raw_label == "wheel_pose" and result.prediction.confidence >= 0.56)
                )
            )
            visual_ready = self._visual_ready("gesture_wheel", active, now, self._wheel_hold_seconds)
            if self._step_completed:
                self._set_step_progress("Wheel pose detected! Swipe right to move on.")
            else:
                self._set_step_progress("Wheel pose detected!" if active else "Waiting for wheel pose.")
            items = (
                ("add_playlist", "Add Playlist", 90.0),
                ("remove_playlist", "Remove Playlist", 38.57),
                ("add_queue", "Add Queue", 347.14),
                ("remove_queue", "Remove Queue", 295.71),
                ("like", "Add to Liked", 244.29),
                ("remove_liked", "Remove from Liked", 192.86),
                ("shuffle", "Shuffle", 141.43),
            )
            if active:
                if self._tutorial_wheel_anchor is None:
                    self._tutorial_wheel_anchor = result.hand_reading.palm.center.copy()
                    self._tutorial_wheel_selected_key = None
                    self._tutorial_wheel_selected_since = now
                offset = (result.hand_reading.palm.center - self._tutorial_wheel_anchor) / max(result.hand_reading.palm.scale, 1e-6)
                self._tutorial_wheel_cursor_offset = (float(offset[0]), float(offset[1]))
                selection_key = self._tutorial_wheel_selection_key(float(offset[0]), float(offset[1]), items)
                if selection_key != self._tutorial_wheel_selected_key:
                    self._tutorial_wheel_selected_key = selection_key
                    self._tutorial_wheel_selected_since = now
                selected_label = "Move to a slice"
                if self._tutorial_wheel_selected_key is not None:
                    selected_label = self._tutorial_wheel_label(self._tutorial_wheel_selected_key, items)
                self._tutorial_wheel_overlay.set_wheel(
                    items=items,
                    selected_key=self._tutorial_wheel_selected_key,
                    selection_progress=(
                        min(1.0, max(0.0, (now - self._tutorial_wheel_selected_since) / 1.0))
                        if self._tutorial_wheel_selected_key is not None else 0.0
                    ),
                    status_text=selected_label,
                    cursor_offset=self._tutorial_wheel_cursor_offset,
                )
                self._tutorial_wheel_overlay.show_overlay()
                if self._tutorial_wheel_selected_key is not None and (now - self._tutorial_wheel_selected_since) >= 1.0:
                    self._complete_step("Completed! Swipe right to move on!")
            else:
                self._tutorial_wheel_anchor = None
                self._tutorial_wheel_selected_key = None
                self._tutorial_wheel_selected_since = 0.0
                self._tutorial_wheel_cursor_offset = None
                self._tutorial_wheel_overlay.hide_overlay()
            return visual_ready

        if step.key == "mouse_mode":
            if result.found and result.hand_reading is not None and result.tracked_hand is not None:
                update = self._mouse_tracker.update(
                    hand_reading=result.hand_reading,
                    prediction=result.prediction,
                    hand_handedness=result.tracked_hand.handedness,
                    cursor_seed=None,
                    now=now,
                )
            else:
                update = self._mouse_tracker.update(hand_reading=None, prediction=None, hand_handedness=None, cursor_seed=None, now=now)
            self.mouse_widget.set_mode_enabled(update.mode_enabled)
            self.mouse_widget.set_cursor_position(update.cursor_position)
            # The pinch-redesign moved the canonical "click" signal
            # from update.left_click to update.left_press: a pinch
            # start now fires left_press immediately (so a real-app
            # mouse-down lands cleanly), and a release fires
            # left_release. left_click is no longer emitted on the
            # release path — that was producing duplicate down+up
            # cycles in the live engine. The tutorial's practice
            # arena only cares about "click intent at this cursor",
            # so we use left_press here. Falling back to left_click
            # too keeps any stragglers (or future re-introduction of
            # the event) working without a code change.
            if update.left_press or update.left_click:
                self.mouse_widget.register_click(update.cursor_position)
            self._set_camera_header_text(
                self._mouse_mode_header_text(
                    mode_enabled=update.mode_enabled,
                    completed_targets=self.mouse_widget.completed_targets,
                )
            )
            if self._mouse_stage == "enable":
                mouse_enable_active = bool(result.found and result.tracked_hand is not None and str(result.tracked_hand.handedness or "").lower() == "left" and result.prediction.stable_label == "three")
                self._set_step_progress("Detected left-hand three!" if mouse_enable_active else "Mouse mode off. Turn it on to begin.")
                visual_ready = self._visual_ready("mouse_enable", mouse_enable_active, now, self._mouse_tracker.toggle_hold_seconds)
                if update.mode_enabled:
                    self._mouse_stage = "practice"
            elif self._mouse_stage == "practice":
                self._set_step_progress("Mouse mode on. Clear all tutorial targets.")
                visual_ready = update.mode_enabled
                if self.mouse_widget.completed:
                    # Move into scroll phase — see the parallel
                    # branch in the engine-driven handler for the
                    # same flow.
                    self._mouse_stage = "scroll"
                    try:
                        self.mouse_scroll_widget.reset()
                        self.practice_stack.setCurrentWidget(self.mouse_scroll_widget)
                    except Exception:
                        pass
            elif self._mouse_stage == "scroll":
                # Mirror the engine-driven branch's progressive
                # footer: which extremity has the user reached?
                if self.mouse_scroll_widget.top_reached and not self.mouse_scroll_widget.bottom_reached:
                    self._set_step_progress("Now get to the bottom!")
                elif self.mouse_scroll_widget.bottom_reached and not self.mouse_scroll_widget.top_reached:
                    self._set_step_progress("Now scroll up to the top!")
                else:
                    self._set_step_progress(
                        "Get to the top of the page on the right."
                    )
                visual_ready = update.mode_enabled
                # Pipe scroll_steps from the local mouse_tracker
                # update straight into the practice arena. Same
                # contract as the engine-driven path above; positive
                # = up, negative = down.
                steps = int(getattr(update, "scroll_steps", 0) or 0)
                if steps != 0:
                    try:
                        self.mouse_scroll_widget.register_scroll_steps(steps)
                    except Exception:
                        pass
                if self.mouse_scroll_widget.completed:
                    self._mouse_stage = "disable"
                    try:
                        self.practice_stack.setCurrentWidget(self.mouse_widget)
                    except Exception:
                        pass
            else:
                mouse_disable_active = bool(result.found and result.tracked_hand is not None and str(result.tracked_hand.handedness or "").lower() == "left" and result.prediction.stable_label == "three")
                self._set_step_progress(
                    "Detected left-hand three!" if mouse_disable_active
                    else "You've completed the mouse tutorial! Turn mouse mode off to finish."
                )
                visual_ready = self._visual_ready("mouse_disable", mouse_disable_active, now, self._mouse_tracker.toggle_hold_seconds)
                if not update.mode_enabled and self.mouse_widget.completed and self.mouse_scroll_widget.completed:
                    self._complete_step(f"Mouse mode practice completed. Part {self._step_index + 1}/6 completed!")
            if update.mode_enabled and update.cursor_position is not None:
                self._mouse_cursor_seen = True
            return visual_ready

        if step.key == "voice_command":
            left_one_active = bool(
                result.found and result.tracked_hand is not None
                and str(result.tracked_hand.handedness or "").lower() == "left"
                and result.prediction.stable_label == "one"
            )
            # Same arrow wiring as the worker-driven step handler
            # above — keeps the cue visible across both ownership
            # modes. self._voice_listening is the owned-worker source
            # of truth (set in _start_voice_practice and cleared in
            # the result handler).
            self._update_voice_arrow(self._voice_listening)
            self._flash_on_edge("voice_command", left_one_active, now)
            visual_ready = now < self._visual_green_until.get("voice_command", 0.0)
            self._set_camera_header_text(self._voice_command_header_text(left_one_active or self._voice_listening))
            self._set_step_progress(
                self._voice_command_progress_text(
                    left_one_active=left_one_active,
                    voice_listening=self._voice_listening,
                )
            )
            if self._hold_ready("voice_command", left_one_active and not self._voice_listening, 0.6, now, cooldown=1.5):
                if self._worker is not None:
                    try:
                        self._worker._start_voice_command()
                    except Exception:
                        self._start_voice_practice()
                else:
                    self._start_voice_practice()
                visual_ready = True
            return visual_ready or self._voice_listening

        return visual_ready

    def _tutorial_wheel_selection_key(self, dx: float, dy: float, items: tuple[tuple[str, str, float], ...]) -> str | None:
        radius = math.hypot(dx, dy)
        # No upper bound: once the cursor is past the central
        # deadzone the angle alone picks the slice, even if the
        # user's hand has drifted well past the visible wheel ring.
        if radius < 0.28:
            return None
        angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0
        slice_span = 360.0 / max(1, len(items))
        half_slice = slice_span / 2.0
        for key, _label, target_angle in items:
            delta = abs((angle - target_angle + 180.0) % 360.0 - 180.0)
            if delta <= max(0.0, half_slice - 1.5):
                return key
        return None

    def _tutorial_wheel_label(self, key: str, items: tuple[tuple[str, str, float], ...]) -> str:
        for item_key, label, _angle in items:
            if item_key == key:
                return label
        return key.replace("_", " ")

    def _tick(self) -> None:
        if self._cap is None or self._engine is None:
            return
        ok, frame = self._cap.read()
        if not ok:
            return
        # Always mirror to selfie view. The earlier camera_source_is_mirrored
        # branch caused tutorial / recorder cold-start paths to show
        # camera-perspective when the config flag was on, which felt
        # broken to users. Selfie view is what people expect across
        # every Touchless surface, so we flip unconditionally here.
        if not bool(getattr(self.config, "camera_source_is_mirrored", False)):
            frame = cv2.flip(frame, 1)
        result = self._engine.process_frame(frame)
        monotonic_now = time.monotonic()
        self._drain_voice_queue()
        visual_ready = self._update_step_progress(result, monotonic_now)
        self._tutorial_nav_from_swipe(result.prediction, monotonic_now)

        display = result.annotated_frame.copy()
        self._draw_demo_overlay(display, self._practice_steps[self._step_index].key)
        self._draw_user_skeleton_overlay(display, result, (80, 235, 120) if visual_ready else (70, 70, 255))
        self._draw_encouragement_overlay(display, monotonic_now)
        self._render_frame(display)

        step_key = self._practice_steps[self._step_index].key
        if step_key == "mouse_mode":
            mouse_mode_text = "On" if self._mouse_tracker.mode_enabled else "Off"
            cv2.putText(display, f"Mouse Mode: {mouse_mode_text}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (182, 233, 29), 2, cv2.LINE_AA)
        banner = result.prediction.stable_label if result.prediction.stable_label != "neutral" else result.prediction.raw_label
        if step_key == "mouse_mode" and self._mouse_tracker.mode_enabled:
            self.gesture_chip.setText("Mouse mode active")
        elif result.prediction.dynamic_label != "neutral":
            self.gesture_chip.setText(f"Dynamic: {result.prediction.dynamic_label.replace('_', ' ')}")
        else:
            self.gesture_chip.setText(f"Gesture: {banner}")
        if self._voice_overlay is not None:
            self._voice_overlay.tick(monotonic_now)
        self._check_encouragement_trigger(monotonic_now)
        self._update_encouragement_visual(monotonic_now)
        self._update_completion_feedback(monotonic_now)

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        super().resizeEvent(event)
        # Voice-mic arrow overlay covers the whole dialog so its
        # bottom-center anchor lines up with the bottom-center of
        # the tutorial window. Re-pin on every resize and refresh
        # the absolute target coord (the screen we're on may have
        # changed if we just got dragged across monitors).
        arrow = getattr(self, "_voice_mic_arrow", None)
        if arrow is not None:
            arrow.setGeometry(0, 0, self.width(), self.height())
            arrow.update_target_from_screen()
        # Re-anchor the starting-pill to bottom-center of the
        # dialog on resize so it stays put if the user drags the
        # window to a new size mid-load.
        self._reposition_starting_pill()

    # ----- Starting pill -----------------------------------------------
    def _reposition_starting_pill(self) -> None:
        pill = getattr(self, "_starting_pill", None)
        if pill is None or not pill.isVisible():
            return
        x = max(8, (self.width() - pill.width()) // 2)
        y = max(8, self.height() - pill.height() - 64)
        pill.move(x, y)

    def _show_starting_pill(self, label: str = "Starting tutorial") -> None:
        pill = getattr(self, "_starting_pill", None)
        if pill is None:
            return
        pill.show_pill(label)
        self._reposition_starting_pill()

    def _hide_starting_pill(self) -> None:
        pill = getattr(self, "_starting_pill", None)
        if pill is not None:
            pill.hide_pill()

    def moveEvent(self, event) -> None:  # noqa: N802 — Qt naming
        super().moveEvent(event)
        # Window dragged: the arrow's anchor moved, so its direction
        # to the absolute mic position needs to be recomputed.
        # Recompute the target too — a cross-monitor drag changes
        # which screen the mic overlay will appear on.
        arrow = getattr(self, "_voice_mic_arrow", None)
        if arrow is not None and arrow.isVisible():
            arrow.update_target_from_screen()

    def showEvent(self, event):  # noqa: N802 (Qt API name)
        super().showEvent(event)
        # Paint the OS title bar in Touchless blue so the tutorial
        # window matches the rest of the app's chrome (Custom
        # Gestures dialogs etc. all use the same helper).
        try:
            from .custom_gestures_chrome import apply_touchless_titlebar
            apply_touchless_titlebar(self)
        except Exception:
            pass

    def _start_tutorial_refocus_guard(self) -> None:
        """After triggering Spotify (or any other foreground-stealing
        action) during the tutorial, run a short polling loop that
        snaps focus back to the tutorial window if anything else
        becomes the foreground window.

        Polls every 120 ms for ~3 s. Spotify's launch sequence on
        Windows briefly throws a splash/main window even when we
        ask for hidden mode — this guarantees the user's keyboard
        focus stays on the tutorial regardless."""
        if sys.platform != "win32":
            return
        timer = getattr(self, "_tutorial_refocus_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(120)
            timer.timeout.connect(self._tutorial_refocus_tick)
            self._tutorial_refocus_timer = timer
        # Reset the deadline so consecutive triggers extend, not
        # overlap, the guard window.
        self._tutorial_refocus_deadline = time.monotonic() + 3.0
        # Snap focus immediately, then keep polling.
        self._refocus_tutorial_window()
        timer.start()

    def _tutorial_refocus_tick(self) -> None:
        deadline = getattr(self, "_tutorial_refocus_deadline", 0.0)
        if time.monotonic() >= deadline:
            timer = getattr(self, "_tutorial_refocus_timer", None)
            if timer is not None:
                timer.stop()
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            fg = int(user32.GetForegroundWindow() or 0)
            try:
                tutorial_hwnd = int(self.winId())
            except Exception:
                tutorial_hwnd = 0
            if fg == tutorial_hwnd or fg == 0:
                return
            # Foreground is something else (Spotify splash, etc.).
            # Pull the tutorial back to the front.
            self._refocus_tutorial_window()
        except Exception:
            pass

    def _refocus_tutorial_window(self) -> None:
        """Snap the tutorial window back to the foreground. Uses
        the standard Qt API plus a Win32 fallback because Windows
        prevents apps from stealing focus from another process
        unless they go through specific call sequences (the
        AttachThreadInput / AllowSetForegroundWindow trick is
        usually unnecessary because we're the parent process of
        whatever just stole focus, but keeping the SetForegroundWindow
        call is cheap defense)."""
        try:
            self.raise_()
            self.activateWindow()
        except Exception:
            pass
        if sys.platform != "win32":
            return
        try:
            import ctypes
            tutorial_hwnd = int(self.winId())
            if tutorial_hwnd:
                ctypes.windll.user32.SetForegroundWindow(tutorial_hwnd)
        except Exception:
            pass

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._stop_session()
        self._close_step_example_dialog()
        arrow = getattr(self, "_voice_mic_arrow", None)
        if arrow is not None:
            arrow.hide_arrow()
        if not self._closing_programmatically and not self._close_emitted:
            self._close_emitted = True
            self.tutorial_closed.emit(False, self._auto_start_on_done, self._launched_from_settings)
        super().closeEvent(event)

# Author: Konstantin Markov

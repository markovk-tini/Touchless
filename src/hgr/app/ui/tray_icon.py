"""System tray icon for Touchless.

The tray icon's BODY stays the regular Touchless hand icon at all
times -- the user mentioned the body shouldn't change so they can
still recognise it at a glance. Only the BORDER swaps colour to
indicate state:

  * green  -- engine running, gestures enabled
  * yellow -- engine running, gestures paused (manually or 30-min auto)
  * grey   -- engine off

A thicker border keeps the colour change noticeable at 16x16 / 22x22
tray sizes where a thin stroke would be hard to read.

Menu:
  * Pause Gestures (30 min)   -- one-click temporary pause that
                                 auto-resumes after a 30-minute timer
  * Settings                  -- bring the main window to front
  * Quit                      -- hard-exit the app
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon


# Tray-icon colour palette. Matches the rest of the app's accent
# colours so the tray is visually part of the family.
_BORDER_ACTIVE = QColor(29, 233, 182)    # mint (engine running, gestures on)
_BORDER_PAUSED = QColor(245, 180, 80)    # amber (gestures paused)
_BORDER_OFF = QColor(127, 127, 127, 200)  # grey (engine off)


def _state_border_color(state: str) -> QColor:
    if state == "active":
        return _BORDER_ACTIVE
    if state == "paused":
        return _BORDER_PAUSED
    return _BORDER_OFF


def _render_bordered_icon(base_icon: QIcon, border_color: QColor) -> QIcon:
    """Paint a thick coloured rectangle border on top of the app
    icon and return a fresh QIcon. The body of the icon is untouched
    so users still recognise the Touchless hand silhouette; the
    border is the state cue."""
    composed = QIcon()
    for size in (16, 22, 24, 32, 48, 64, 128, 256):
        src_pixmap = base_icon.pixmap(size, size)
        if src_pixmap.isNull():
            continue
        out = QPixmap(src_pixmap.size())
        out.fill(Qt.transparent)
        painter = QPainter(out)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.drawPixmap(0, 0, src_pixmap)
            # Border thickness scales with icon size so the cue is
            # readable on both 16 px tray and 128 px notification
            # popovers. Minimum 2 px so small sizes still show.
            stroke = max(2, int(round(src_pixmap.width() * 0.10)))
            pen = QPen(border_color, stroke)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            half = stroke / 2.0
            painter.drawRect(
                int(half),
                int(half),
                int(src_pixmap.width() - stroke),
                int(src_pixmap.height() - stroke),
            )
        finally:
            painter.end()
        composed.addPixmap(out)
    return composed


class TouchlessTrayIcon(QObject):
    """QSystemTrayIcon wrapper. Owns its menu, the 30-minute pause
    auto-resume timer, and the icon-recoloring on state change."""

    pause_requested = Signal()   # user clicked "Pause Gestures (30 min)"
    resume_requested = Signal()  # user clicked "Resume Gestures"
    settings_requested = Signal()
    quit_requested = Signal()

    PAUSE_SECONDS = 30 * 60  # 30 minutes

    def __init__(self, base_icon: QIcon, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._base_icon = base_icon
        self._state = "off"            # active / paused / off
        self._pause_active = False
        self._pause_started_at = 0.0
        self._tray = QSystemTrayIcon(parent if isinstance(parent, QObject) else None)
        self._tray.setIcon(_render_bordered_icon(base_icon, _BORDER_OFF))
        self._tray.setToolTip("Touchless — engine off")
        self._tray.activated.connect(self._on_activated)

        self._menu = QMenu()
        self._pause_action = QAction("Pause Gestures (30 min)", self._menu)
        self._pause_action.triggered.connect(self._on_pause_clicked)
        self._menu.addAction(self._pause_action)

        self._settings_action = QAction("Settings", self._menu)
        self._settings_action.triggered.connect(lambda: self.settings_requested.emit())
        self._menu.addAction(self._settings_action)

        self._menu.addSeparator()

        self._quit_action = QAction("Quit Touchless", self._menu)
        self._quit_action.triggered.connect(lambda: self.quit_requested.emit())
        self._menu.addAction(self._quit_action)

        self._tray.setContextMenu(self._menu)

        # Auto-resume timer for the 30-minute pause. Single-shot;
        # restarted on each fresh "Pause" click so a second click
        # mid-pause extends the window.
        self._auto_resume_timer = QTimer(self)
        self._auto_resume_timer.setSingleShot(True)
        self._auto_resume_timer.timeout.connect(self._on_auto_resume)

        # Tooltip-refresh timer so the user can see how much pause
        # time is left when hovering the tray icon.
        self._tooltip_refresh_timer = QTimer(self)
        self._tooltip_refresh_timer.setInterval(15_000)  # every 15s
        self._tooltip_refresh_timer.timeout.connect(self._refresh_tooltip)

    @staticmethod
    def is_supported() -> bool:
        return QSystemTrayIcon.isSystemTrayAvailable()

    def show(self) -> None:
        if self.is_supported():
            self._tray.show()

    def hide(self) -> None:
        self._tray.hide()
        self._auto_resume_timer.stop()
        self._tooltip_refresh_timer.stop()

    def set_engine_state(self, *, engine_running: bool, gestures_enabled: bool) -> None:
        """Public hook the MainWindow calls when running_state or
        gestures-enabled changes. Recomputes the state label and
        repaints the tray icon accordingly. Does NOT touch the
        manual-pause flag -- that's owned by this class."""
        if not engine_running:
            new_state = "off"
            self._cancel_pause()
        elif self._pause_active or not gestures_enabled:
            new_state = "paused"
        else:
            new_state = "active"
        if new_state == self._state:
            return
        self._state = new_state
        self._refresh_icon()
        self._refresh_tooltip()
        self._pause_action.setText(self._pause_action_label())

    # ----- internal ----------------------------------------------

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.DoubleClick:
            self.settings_requested.emit()

    def _on_pause_clicked(self) -> None:
        if self._pause_active:
            # User clicked "Resume Gestures" -- the menu entry's
            # label flips when paused (see _pause_action_label).
            self._cancel_pause()
            self.resume_requested.emit()
            return
        self._pause_active = True
        self._pause_started_at = time.monotonic()
        self._auto_resume_timer.start(self.PAUSE_SECONDS * 1000)
        self._tooltip_refresh_timer.start()
        self._pause_action.setText(self._pause_action_label())
        self.pause_requested.emit()

    def _cancel_pause(self) -> None:
        if not self._pause_active:
            return
        self._pause_active = False
        self._pause_started_at = 0.0
        self._auto_resume_timer.stop()
        self._tooltip_refresh_timer.stop()
        self._pause_action.setText(self._pause_action_label())

    def _on_auto_resume(self) -> None:
        if not self._pause_active:
            return
        self._cancel_pause()
        self.resume_requested.emit()

    def _pause_action_label(self) -> str:
        if self._pause_active:
            remaining = self._pause_remaining_seconds()
            return f"Resume Gestures ({_fmt_mmss(remaining)} left)"
        return "Pause Gestures (30 min)"

    def _pause_remaining_seconds(self) -> int:
        if not self._pause_active:
            return 0
        elapsed = max(0.0, time.monotonic() - self._pause_started_at)
        return max(0, int(round(self.PAUSE_SECONDS - elapsed)))

    def _refresh_icon(self) -> None:
        border = _state_border_color(self._state)
        self._tray.setIcon(_render_bordered_icon(self._base_icon, border))

    def _refresh_tooltip(self) -> None:
        if self._state == "off":
            self._tray.setToolTip("Touchless — engine off")
        elif self._state == "paused":
            remaining = self._pause_remaining_seconds()
            if remaining > 0:
                self._tray.setToolTip(f"Touchless — paused ({_fmt_mmss(remaining)} left)")
            else:
                self._tray.setToolTip("Touchless — gestures paused")
        else:
            self._tray.setToolTip("Touchless — active")


def _fmt_mmss(seconds: int) -> str:
    minutes, sec = divmod(max(0, int(seconds)), 60)
    return f"{minutes}:{sec:02d}"

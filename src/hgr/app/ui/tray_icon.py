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
_BORDER_ACTIVE = QColor(29, 233, 182)    # mint   (engine running, gestures on)
_BORDER_PAUSED = QColor(255, 138, 61)    # orange (gestures paused, mid-warning)
_BORDER_OFF = QColor(150, 150, 150, 230)  # grey  (app open but engine not running)


def _state_border_color(state: str) -> QColor:
    if state == "active":
        return _BORDER_ACTIVE
    if state == "paused":
        return _BORDER_PAUSED
    return _BORDER_OFF


def _render_bordered_icon(base_icon: QIcon, border_color: QColor) -> QIcon:
    """Render a state icon: thick coloured ring around the
    Touchless hand. The hand stays recognisable; the ring is
    the state cue.

    Approach: paint a filled rounded-rect of the state colour as
    the BACKGROUND, then paint the hand icon ON TOP, slightly
    inset so the colour shows as a thick ring. This is way more
    visible at 16-22 px tray sizes than a thin stroked border,
    which gets eaten by anti-aliasing on small icons."""
    composed = QIcon()
    for size in (16, 22, 24, 32, 48, 64, 128, 256):
        # Bypass Qt's icon-cache scaling by asking for the EXACT
        # pixel size we want -- otherwise QIcon may hand back a
        # pre-rendered cached pixmap that was already coloured
        # for a previous border state.
        src_pixmap = base_icon.pixmap(size, size)
        if src_pixmap.isNull():
            continue
        out = QPixmap(size, size)
        out.fill(Qt.transparent)
        painter = QPainter(out)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            # Ring thickness ~4 % of width (a notch thinner than the
            # previous 6 %). Hand body fills ~92 % of icon area --
            # the Touchless silhouette is overwhelmingly dominant
            # and the ring is just a slim frame. Min 2 px so the
            # cue survives at the smallest tray sizes.
            ring = max(2, int(round(size * 0.04)))
            radius = max(2, int(round(size * 0.18)))
            # Filled coloured rounded-rect background.
            painter.setBrush(border_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(0, 0, size, size, radius, radius)
            # Hand icon drawn ON TOP, inset by the ring thickness on
            # each side so the colour shows as a uniform ring around
            # the silhouette.
            inner_size = max(1, size - 2 * ring)
            painter.drawPixmap(ring, ring, inner_size, inner_size, src_pixmap)
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
    # Emitted whenever the rendered state icon changes. MainWindow
    # wires this to setWindowIcon so the TASKBAR entry mirrors the
    # tray's state cue -- user always sees the colour at a glance
    # without having to expand the hidden-tray popout.
    icon_changed = Signal(object)

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
        new_icon = _render_bordered_icon(self._base_icon, border)
        self._tray.setIcon(new_icon)
        # Force the Windows system tray (Shell_NotifyIcon path) to
        # actually repaint. Qt's QSystemTrayIcon.setIcon issues
        # NIM_MODIFY, but Explorer.exe sometimes ignores it and
        # keeps showing the previous icon until something else
        # triggers a refresh -- which is why the colour swap
        # appeared not to happen. A hide() + show() pair forces
        # NIM_DELETE + NIM_ADD, which always repaints. The flicker
        # is ~50 ms and only fires on state change.
        if self._tray.isVisible():
            try:
                self._tray.hide()
                self._tray.show()
            except Exception:
                pass
        # Push the new icon to listeners so the taskbar (window
        # icon) also reflects the state cue. Tray-only would force
        # the user to expand the hidden-icons popout to see state.
        try:
            self.icon_changed.emit(new_icon)
        except Exception:
            pass

    def current_icon(self) -> QIcon:
        """Return the currently-rendered bordered icon. Useful for
        wiring up the MainWindow's window icon at construction time
        before the first state-change signal fires."""
        border = _state_border_color(self._state)
        return _render_bordered_icon(self._base_icon, border)

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

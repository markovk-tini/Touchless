"""Consistent Touchless brand chrome for every user-visible pop-up.

Most dialogs in the app use Qt's native title bar (the OS draws it).
By default that produces white-on-light on Windows 11, very different
from the deep-indigo title bar the main window uses (`#1F2D6B` via
the custom frameless `TitleBar` widget).

This helper unifies both touch points without forcing every dialog
to go frameless:

  1. **Icon top-left** — calls `setWindowIcon(QApplication.windowIcon())`
     which Windows displays at the top-left of the native title bar.
     `QApplication.windowIcon()` is set once at app startup from
     `touchless_icon.ico`, so all callers share the same hand icon.

  2. **Title-bar color** — Windows 11 (build 22000+) exposes
     `DWMWA_CAPTION_COLOR` (35) and `DWMWA_TEXT_COLOR` (36) via
     `DwmSetWindowAttribute`. We set them to Touchless indigo / light
     text so the native bar matches the main window's custom bar.

Behavior on older Windows / other OSes:
  * Icon is set unconditionally (Qt handles it everywhere).
  * DWM calls silently no-op on pre-Win11 / non-Windows — the dialog
    keeps the OS default title color, which is fine.

The DWM attribute IDs are integer constants, so no SDK header
dependency — just ctypes against `dwmapi.dll`.
"""
from __future__ import annotations

import ctypes
import sys
from typing import Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget


# Touchless brand colors. Must MATCH the main window's title bar
# (which uses its own DwmSetWindowAttribute call in main_window.py)
# so every dialog and pop-up looks like part of the same app.
# COLORREF format is 0x00BBGGRR (little-endian RGB).
#   #1F2D6B (Touchless deep indigo)  =>  COLORREF 0x006B2D1F
#   #E5F6FF (Touchless light text)   =>  COLORREF 0x00FFF6E5
# The older #0B3D91 (brighter primary blue) lived here before --
# user noticed the custom-gesture creator's title bar didn't match
# the main window after the main window's title-bar repaint
# landed (#1F2D6B). Realigning them here propagates to every
# dialog that calls apply_touchless_chrome (recorder, sandbox,
# wizard, gesture pose picker, etc.).
_CAPTION_COLORREF = 0x006B2D1F
_TEXT_COLORREF = 0x00FFF6E5

# Windows DwmSetWindowAttribute attribute IDs (Win11 22000+).
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36


def apply_touchless_chrome(window: QWidget) -> None:
    """Apply the Touchless icon + title-bar color to `window`.

    Safe to call in any dialog's __init__. The DWM call defers via
    QTimer.singleShot so it lands AFTER Qt has created the native
    window handle (winId() before the window is shown sometimes
    returns 0, which would make the DWM call a no-op).
    """
    # Icon: works on any platform, on any title-bar style.
    icon = QApplication.windowIcon()
    if not icon.isNull():
        try:
            window.setWindowIcon(icon)
        except Exception:
            pass

    # DWM color: Windows only, and only meaningful on Win11. Defer
    # via singleShot so Qt has finished native-handle creation.
    if not sys.platform.startswith("win"):
        return
    QTimer.singleShot(0, lambda w=window: _apply_dwm_caption_color(w))


def touchless_message_box(
    parent: Optional[QWidget],
    title: str,
    text: str,
    *,
    icon: "QMessageBox.Icon" = QMessageBox.Warning,
    buttons: "QMessageBox.StandardButtons" = QMessageBox.Ok,
    default_button: "QMessageBox.StandardButton" = QMessageBox.NoButton,
) -> "QMessageBox.StandardButton":
    """Construct and show a QMessageBox with Touchless chrome applied.

    Use this anywhere you would otherwise call `QMessageBox.warning(...)`
    / `QMessageBox.critical(...)` / `QMessageBox.information(...)` —
    those static methods construct and show the box before we can
    chrome it, so they end up with white OS title bars. Building the
    box manually and applying chrome before `exec()` keeps the icon
    + caption color consistent with the rest of the app.

    Returns the StandardButton the user clicked.
    """
    box = QMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(buttons)
    if default_button != QMessageBox.NoButton:
        box.setDefaultButton(default_button)
    apply_touchless_chrome(box)
    return QMessageBox.StandardButton(box.exec())


def _apply_dwm_caption_color(window: QWidget) -> None:
    """Call DwmSetWindowAttribute for caption + text color.

    Silently no-op on pre-Win11. Safe to call multiple times.
    """
    try:
        hwnd = int(window.winId())
    except Exception:
        return
    if hwnd == 0:
        return
    try:
        dwmapi = ctypes.WinDLL("dwmapi")
    except OSError:
        return
    for attr, value in (
        (_DWMWA_CAPTION_COLOR, _CAPTION_COLORREF),
        (_DWMWA_TEXT_COLOR, _TEXT_COLORREF),
    ):
        try:
            colorref = ctypes.c_int(value)
            dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(hwnd),
                ctypes.c_uint(attr),
                ctypes.byref(colorref),
                ctypes.c_uint(ctypes.sizeof(colorref)),
            )
        except Exception:
            # Pre-Win11 returns an error HRESULT — that's fine, the
            # title bar just keeps its default color.
            continue

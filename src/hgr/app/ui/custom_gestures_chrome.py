"""Shared title-bar theming helper for the Custom Gestures dialogs.

Mirrors the DwmSetWindowAttribute call MainWindow uses on its own
notice/popup windows so my Custom Gestures dialogs paint a Touchless-
blue OS title bar on Windows 11 (silently no-ops on Win 10 / non-
Windows). Pull this in via apply_touchless_titlebar(self) inside the
showEvent of any QDialog/QWidget.
"""
from __future__ import annotations

import sys


# Windows 11 build 22000+ DWM attribute IDs.
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36

# COLORREF = 0x00BBGGRR.
# Touchless deep indigo #1F2D6B → 0x006B2D1F  (matches MainWindow titlebar)
# Touchless light text #E5F6FF → 0x00FFF6E5
_CAPTION_COLOR = 0x006B2D1F
_TEXT_COLOR = 0x00FFF6E5


def apply_touchless_titlebar(widget) -> None:
    """Paint the OS title bar with Touchless's primary blue + set the
    Touchless hand icon as the window icon (which Windows displays at
    the top-left of the native title bar). Safe to call from any
    showEvent — wraps every operation in try/except so a stale HWND or
    older Windows version just produces a no-op."""
    # Icon (works on any platform, any title-bar style).
    try:
        from PySide6.QtWidgets import QApplication
        app_icon = QApplication.windowIcon()
        if not app_icon.isNull():
            widget.setWindowIcon(app_icon)
    except Exception:
        pass

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
    caption = ctypes.c_uint32(_CAPTION_COLOR)
    text = ctypes.c_uint32(_TEXT_COLOR)
    try:
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            ctypes.c_uint32(_DWMWA_CAPTION_COLOR),
            ctypes.byref(caption),
            ctypes.sizeof(caption),
        )
    except Exception:
        pass
    try:
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            ctypes.c_uint32(_DWMWA_TEXT_COLOR),
            ctypes.byref(text),
            ctypes.sizeof(text),
        )
    except Exception:
        pass

# Author: Konstantin Markov

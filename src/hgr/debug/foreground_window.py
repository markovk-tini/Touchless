from __future__ import annotations

import ctypes
import os
import platform
from ctypes import wintypes
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    process_name: str


_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS:
    _user32 = ctypes.windll.user32
    _kernel32 = ctypes.windll.kernel32

    _EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    _user32.EnumWindows.argtypes = [_EnumWindowsProc, wintypes.LPARAM]
    _user32.EnumWindows.restype = wintypes.BOOL
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _user32.IsWindowVisible.restype = wintypes.BOOL
    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetDesktopWindow.restype = wintypes.HWND
    _user32.GetShellWindow.restype = wintypes.HWND
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.GetWindowRect.restype = wintypes.BOOL

    class _MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    _user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    _user32.MonitorFromWindow.restype = ctypes.c_void_p
    _user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MONITORINFO)]
    _user32.GetMonitorInfoW.restype = wintypes.BOOL
    _user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.GetWindowLongW.restype = ctypes.c_long

    _MONITOR_DEFAULTTONEAREST = 2
    _GWL_STYLE = -16
    _WS_CAPTION = 0x00C00000
    _WS_THICKFRAME = 0x00040000


def _process_name_for_pid(pid: int) -> str:
    try:
        import psutil
        return (psutil.Process(pid).name() or "").lower()
    except Exception:
        return ""


def _window_title(hwnd: int) -> str:
    try:
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value or ""
    except Exception:
        return ""


def _window_process_name(hwnd: int) -> str:
    try:
        pid = wintypes.DWORD(0)
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return _process_name_for_pid(int(pid.value))
    except Exception:
        return ""


def get_foreground_window_info() -> WindowInfo | None:
    if not _IS_WINDOWS:
        return None
    try:
        hwnd = int(_user32.GetForegroundWindow() or 0)
        if hwnd == 0:
            return None
        return WindowInfo(hwnd=hwnd, title=_window_title(hwnd), process_name=_window_process_name(hwnd))
    except Exception:
        return None


def enumerate_visible_windows() -> list[WindowInfo]:
    if not _IS_WINDOWS:
        return []
    results: list[WindowInfo] = []

    def _callback(hwnd: int, _lparam: int) -> bool:
        try:
            if not _user32.IsWindowVisible(hwnd):
                return True
            title = _window_title(hwnd)
            if not title:
                return True
            results.append(WindowInfo(hwnd=int(hwnd), title=title, process_name=_window_process_name(hwnd)))
        except Exception:
            pass
        return True

    try:
        _user32.EnumWindows(_EnumWindowsProc(_callback), 0)
    except Exception:
        pass
    return results


def find_chrome_youtube_windows() -> list[WindowInfo]:
    """Fast path: filter by title first, only resolve process name for YouTube-titled windows.

    psutil.Process(pid).name() is ~5-20ms on Windows, so doing it for every visible
    window (often 50+) is unacceptable on the hot frame loop. Title inspection is a
    pure user32 call — cheap.
    """
    if not _IS_WINDOWS:
        return []
    candidates: list[tuple[int, str]] = []

    def _callback(hwnd: int, _lparam: int) -> bool:
        try:
            if not _user32.IsWindowVisible(hwnd):
                return True
            title = _window_title(hwnd)
            if not title:
                return True
            if "youtube" not in title.lower():
                return True
            candidates.append((int(hwnd), title))
        except Exception:
            pass
        return True

    try:
        _user32.EnumWindows(_EnumWindowsProc(_callback), 0)
    except Exception:
        pass

    matches: list[WindowInfo] = []
    for hwnd, title in candidates:
        process_name = _window_process_name(hwnd)
        if "chrome" not in process_name:
            continue
        matches.append(WindowInfo(hwnd=hwnd, title=title, process_name=process_name))
    return matches


def is_foreground_fullscreen() -> bool:
    """True when the foreground window is a REAL borderless / exclusive
    fullscreen app (a game, a video player in fullscreen mode) — not a
    normal maximized Win32 window (Chrome, VS Code, Explorer, etc.).

    v1.1.9.2 fix: previously any window whose rect equalled-or-overshot
    the monitor rect returned True, which caught every maximized Win32
    window (their GetWindowRect commonly reports left=-8 / right=mon.right+8
    because of the invisible resize-border padding). That produced a
    latched `_gpu_suppressed_for_fullscreen = True` the first time a
    user maximized Chrome, pinning Touchless in CPU MediaPipe at
    ~22-25 fps until they un-maximized. Two guards fix it:
      * skip our own process (Touchless maximized should never suppress
        itself),
      * reject any window that still has a caption or resize frame —
        genuine borderless fullscreen has neither WS_CAPTION nor
        WS_THICKFRAME.
    Games + fullscreen video players stay detected correctly.
    """
    if not _IS_WINDOWS:
        return False
    try:
        hwnd = int(_user32.GetForegroundWindow() or 0)
        if hwnd == 0:
            return False
        desktop = int(_user32.GetDesktopWindow() or 0)
        shell = int(_user32.GetShellWindow() or 0)
        if hwnd == desktop or hwnd == shell:
            return False
        # Skip our own foreground window — Touchless-maximized should
        # never trigger fullscreen suppression on itself.
        try:
            pid_dw = wintypes.DWORD(0)
            _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_dw))
            if int(pid_dw.value) == os.getpid():
                return False
        except Exception:
            pass
        # Reject normal chrome-having windows: real borderless fullscreen
        # has neither WS_CAPTION (title bar) nor WS_THICKFRAME (resize
        # border). Every regular maximized Win32 window has one or both.
        try:
            style = int(_user32.GetWindowLongW(hwnd, _GWL_STYLE))
            if style & (_WS_CAPTION | _WS_THICKFRAME):
                return False
        except Exception:
            pass
        rect = wintypes.RECT()
        if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return False
        hmon = _user32.MonitorFromWindow(hwnd, _MONITOR_DEFAULTTONEAREST)
        if not hmon:
            return False
        mi = _MONITORINFO()
        mi.cbSize = ctypes.sizeof(_MONITORINFO)
        if not _user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            return False
        mon = mi.rcMonitor
        return (
            rect.left <= mon.left
            and rect.top <= mon.top
            and rect.right >= mon.right
            and rect.bottom >= mon.bottom
        )
    except Exception:
        return False

# Author: Konstantin Markov

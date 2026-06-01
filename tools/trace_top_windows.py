"""Trace EVERY top-level Win32 window that becomes visible while we
watch — so we can identify the mystery white popup by its actual title
and class name instead of guessing.

Usage:
    set PYTHONPATH=src
    python tools/trace_top_windows.py

Then in a SEPARATE terminal, run the app the way you normally do
(python -m hgr.app.main). Watch this terminal: every new visible
top-level window of the python.exe process gets a line printed with
its HWND, title, class, rect, and pid. Stop with Ctrl+C.
"""
from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

EnumWindows = user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
GetWindowText = user32.GetWindowTextW
GetWindowTextLength = user32.GetWindowTextLengthW
GetClassName = user32.GetClassNameW
IsWindowVisible = user32.IsWindowVisible
GetWindowThreadProcessId = user32.GetWindowThreadProcessId
GetWindowRect = user32.GetWindowRect


def _title(hwnd: int) -> str:
    n = GetWindowTextLength(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    GetWindowText(hwnd, buf, n + 1)
    return buf.value


def _cls(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    GetClassName(hwnd, buf, 256)
    return buf.value


def _pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _rect(hwnd: int):
    r = wintypes.RECT()
    if GetWindowRect(hwnd, ctypes.byref(r)):
        return (r.left, r.top, r.right - r.left, r.bottom - r.top)
    return (0, 0, 0, 0)


def snapshot(filter_pid_name: str = "python") -> dict:
    out = {}
    def cb(hwnd, lparam):
        if not IsWindowVisible(hwnd):
            return True
        pid = _pid(hwnd)
        try:
            with open(f"\\\\.\\pipe\\__never__{pid}", "rb"):
                pass
        except Exception:
            pass
        out[hwnd] = {
            "title": _title(hwnd),
            "class": _cls(hwnd),
            "pid": pid,
            "rect": _rect(hwnd),
        }
        return True
    EnumWindows(EnumWindowsProc(cb), 0)
    return out


def proc_name(pid: int) -> str:
    """Best-effort process name. PROCESS_QUERY_LIMITED_INFORMATION = 0x1000."""
    try:
        h = kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return ""
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value).lower()
    except Exception:
        return ""
    return ""


def main() -> None:
    print("Watching for new visible top-level python.exe windows. "
          "Run the app in another terminal. Ctrl+C to stop.\n")
    seen: set[int] = set()
    while True:
        snap = snapshot()
        for hwnd, info in snap.items():
            if hwnd in seen:
                continue
            name = proc_name(info["pid"]).lower()
            if "python" not in name and "touchless" not in name:
                continue
            seen.add(hwnd)
            print(f"NEW hwnd={hwnd:#x} pid={info['pid']} proc={name}"
                  f"\n    title={info['title']!r}"
                  f"\n    class={info['class']!r}"
                  f"\n    rect={info['rect']} (w x h = {info['rect'][2]}x{info['rect'][3]})\n",
                  flush=True)
        time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

from __future__ import annotations

import ctypes
import platform
import weakref
from ctypes import wintypes


MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
WHEEL_DELTA = 120


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class _Point(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class MouseController:
    # WeakSet of every live controller so main_window's topology-change
    # filter (WM_DISPLAYCHANGE / WM_DPICHANGED) can invalidate every
    # bounds cache without needing to reach through the engine chain.
    _instances: "weakref.WeakSet[MouseController]" = weakref.WeakSet()

    @classmethod
    def invalidate_all_bounds_caches(cls) -> None:
        """Called from main_window's WM_DISPLAYCHANGE / WM_DPICHANGED
        handler so every MouseController re-reads GetSystemMetrics
        after a monitor topology change."""
        for inst in list(cls._instances):
            try:
                inst.invalidate_bounds_cache()
            except Exception:
                pass

    def __init__(self) -> None:
        self._available = platform.system() == "Windows"
        self._message = "mouse mode off" if self._available else "mouse unavailable on this platform"
        self._left_down = False
        self._user32 = None
        # v1.1.9.2: virtual_bounds() was called up to 4x per tick from the
        # mouse-control hot path (16 GetSystemMetrics per tick). Under a
        # monitor topology change or GDI lock contention each of these
        # can stall waiting for the display-info lock. Cache the result
        # and invalidate via WM_DISPLAYCHANGE / WM_DPICHANGED in main_window.
        self._cached_bounds: tuple[int, int, int, int] | None = None
        # Skip SetCursorPos when the target pixel hasn't changed — one
        # less user32 syscall on frames where the hand is still.
        self._last_cursor_target: tuple[int, int] | None = None
        MouseController._instances.add(self)
        if not self._available:
            return
        try:
            self._user32 = ctypes.windll.user32
            self._user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
            self._user32.SetCursorPos.restype = wintypes.BOOL
            self._user32.GetCursorPos.argtypes = [ctypes.POINTER(_Point)]
            self._user32.GetCursorPos.restype = wintypes.BOOL
            self._user32.GetSystemMetrics.argtypes = [ctypes.c_int]
            self._user32.GetSystemMetrics.restype = ctypes.c_int
        except Exception:
            self._available = False
            self._user32 = None
            self._message = "mouse unavailable"

    def invalidate_bounds_cache(self) -> None:
        """Public hook: main_window's WM_DISPLAYCHANGE / WM_DPICHANGED
        filter calls this so the next virtual_bounds() re-reads
        GetSystemMetrics after a monitor topology change."""
        self._cached_bounds = None

    @property
    def available(self) -> bool:
        return self._available and self._user32 is not None

    @property
    def message(self) -> str:
        return self._message

    @property
    def left_pressed(self) -> bool:
        return self._left_down

    def virtual_bounds(self) -> tuple[int, int, int, int] | None:
        if not self.available:
            return None
        cached = self._cached_bounds
        if cached is not None:
            return cached
        assert self._user32 is not None
        left = int(self._user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
        top = int(self._user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
        width = max(1, int(self._user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)))
        height = max(1, int(self._user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)))
        bounds = (left, top, width, height)
        self._cached_bounds = bounds
        return bounds

    def current_position(self) -> tuple[int, int] | None:
        if not self.available:
            return None
        assert self._user32 is not None
        point = _Point()
        if not self._user32.GetCursorPos(ctypes.byref(point)):
            self._message = "mouse position unavailable"
            return None
        return int(point.x), int(point.y)

    def current_position_normalized(self) -> tuple[float, float] | None:
        bounds = self.virtual_bounds()
        position = self.current_position()
        if bounds is None or position is None:
            return None
        left, top, width, height = bounds
        x = _clamp01((position[0] - left) / max(width - 1, 1))
        y = _clamp01((position[1] - top) / max(height - 1, 1))
        return x, y

    def move_normalized(self, x: float, y: float) -> bool:
        bounds = self.virtual_bounds()
        if bounds is None:
            self._message = "mouse unavailable"
            return False
        left, top, width, height = bounds
        target_x = left + int(round(_clamp01(x) * max(width - 1, 1)))
        target_y = top + int(round(_clamp01(y) * max(height - 1, 1)))
        assert self._user32 is not None
        # Dedupe: if the target pixel matches the last one we set, the
        # cursor is already there — skip the syscall. Kiyo Pro at 60 Hz
        # + a still hand produces many identical targets in a row.
        if self._last_cursor_target == (target_x, target_y):
            self._message = f"mouse move {target_x}, {target_y}"
            return True
        if not self._user32.SetCursorPos(target_x, target_y):
            self._message = "mouse move failed"
            return False
        self._last_cursor_target = (target_x, target_y)
        self._message = f"mouse move {target_x}, {target_y}"
        return True

    def left_down(self) -> bool:
        if not self._send_mouse_event(MOUSEEVENTF_LEFTDOWN):
            return False
        self._left_down = True
        self._message = "mouse drag start"
        return True

    def left_up(self) -> bool:
        if not self.available:
            self._message = "mouse unavailable"
            return False
        if self._left_down:
            if not self._send_mouse_event(MOUSEEVENTF_LEFTUP):
                return False
        self._left_down = False
        self._message = "mouse drag release"
        return True

    def left_click(self) -> bool:
        if not self.available:
            self._message = "mouse unavailable"
            return False
        self._left_down = False
        if not self._send_mouse_event(MOUSEEVENTF_LEFTDOWN):
            return False
        if not self._send_mouse_event(MOUSEEVENTF_LEFTUP):
            return False
        self._message = "mouse left click"
        return True

    def right_click(self) -> bool:
        if not self.available:
            self._message = "mouse unavailable"
            return False
        if not self._send_mouse_event(MOUSEEVENTF_RIGHTDOWN):
            return False
        if not self._send_mouse_event(MOUSEEVENTF_RIGHTUP):
            return False
        self._message = "mouse right click"
        return True

    def scroll(self, steps: int) -> bool:
        steps = int(steps)
        if steps == 0:
            return True
        if not self._send_mouse_event(MOUSEEVENTF_WHEEL, data=steps * WHEEL_DELTA):
            return False
        direction = "up" if steps > 0 else "down"
        self._message = f"mouse scroll {direction} x{abs(steps)}"
        return True

    def release_all(self) -> bool:
        if not self.available:
            self._message = "mouse unavailable"
            return False
        return self.left_up()

    def _send_mouse_event(self, flags: int, *, data: int = 0) -> bool:
        if not self.available:
            self._message = "mouse unavailable"
            return False
        try:
            assert self._user32 is not None
            self._user32.mouse_event(int(flags), 0, 0, int(data), 0)
        except Exception:
            self._message = "mouse input failed"
            return False
        return True

# Author: Konstantin Markov

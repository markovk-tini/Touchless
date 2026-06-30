from __future__ import annotations

import ctypes
import platform
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
    """OS cursor + click + scroll driver for the gesture-mouse mode.

    Two backends, same public API:
      - Windows: user32 SetCursorPos / mouse_event (unchanged).
      - macOS:   Quartz CGWarpMouseCursorPosition + CGEvent* (pyobjc).

    macOS NOTE: synthetic clicks/drags/scroll go through CGEventPost, which
    macOS silently drops unless the app has the **Accessibility** permission
    (System Settings > Privacy & Security > Accessibility). Cursor *movement*
    via CGWarp works without it; clicks do not. There is no error from
    CGEventPost when blocked — the first-run onboarding must prompt for it.
    """

    def __init__(self) -> None:
        system = platform.system()
        self._backend: str | None = None
        self._message = "mouse mode off"
        self._left_down = False
        self._user32 = None
        self._q = None  # Quartz module on macOS

        if system == "Windows":
            try:
                self._user32 = ctypes.windll.user32
                self._user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
                self._user32.SetCursorPos.restype = wintypes.BOOL
                self._user32.GetCursorPos.argtypes = [ctypes.POINTER(_Point)]
                self._user32.GetCursorPos.restype = wintypes.BOOL
                self._user32.GetSystemMetrics.argtypes = [ctypes.c_int]
                self._user32.GetSystemMetrics.restype = ctypes.c_int
                self._backend = "win"
            except Exception:
                self._user32 = None
                self._message = "mouse unavailable"
        elif system == "Darwin":
            try:
                import Quartz  # type: ignore

                self._q = Quartz
                self._backend = "mac"
                # Register/prompt for Accessibility so the app appears under
                # System Settings > Privacy & Security > Accessibility and the
                # user can grant it (required for synthetic clicks/drags/scroll;
                # cursor movement via CGWarp works without it). Best-effort —
                # never let a permission probe break controller construction.
                try:
                    from ..platform_compat.capabilities import is_accessibility_trusted

                    if not is_accessibility_trusted(prompt=True):
                        self._message = "mouse mode off (grant Accessibility for clicks)"
                except Exception:
                    pass
            except Exception:
                self._message = "mouse unavailable (pyobjc Quartz missing)"
        else:
            self._message = "mouse unavailable on this platform"

        self._available = self._backend is not None

    @property
    def available(self) -> bool:
        if self._backend == "win":
            return self._user32 is not None
        if self._backend == "mac":
            return self._q is not None
        return False

    @property
    def message(self) -> str:
        return self._message

    @property
    def left_pressed(self) -> bool:
        return self._left_down

    def virtual_bounds(self) -> tuple[int, int, int, int] | None:
        if not self.available:
            return None
        if self._backend == "mac":
            # Primary display bounds in global (top-left origin) points — the
            # same coordinate space CGWarp/CGEvent use. Multi-monitor spanning
            # is a later refinement; the primary screen is the sane default for
            # gesture-cursor mapping.
            bounds = self._q.CGDisplayBounds(self._q.CGMainDisplayID())
            return (
                int(bounds.origin.x),
                int(bounds.origin.y),
                max(1, int(bounds.size.width)),
                max(1, int(bounds.size.height)),
            )
        assert self._user32 is not None
        left = int(self._user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
        top = int(self._user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
        width = max(1, int(self._user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)))
        height = max(1, int(self._user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)))
        return left, top, width, height

    def current_position(self) -> tuple[int, int] | None:
        if not self.available:
            return None
        if self._backend == "mac":
            try:
                loc = self._q.CGEventGetLocation(self._q.CGEventCreate(None))
                return int(loc.x), int(loc.y)
            except Exception:
                self._message = "mouse position unavailable"
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
        if self._backend == "mac":
            # While a drag is held, post LeftMouseDragged so apps see the drag;
            # otherwise a plain warp (the analog of SetCursorPos). CGWarp moves
            # the cursor without an event, which is what a plain follow wants.
            if self._left_down:
                if not self._mac_mouse_event(self._q.kCGEventLeftMouseDragged, (target_x, target_y), self._q.kCGMouseButtonLeft):
                    return False
            else:
                try:
                    self._q.CGWarpMouseCursorPosition((target_x, target_y))
                except Exception:
                    self._message = "mouse move failed"
                    return False
            self._message = f"mouse move {target_x}, {target_y}"
            return True
        assert self._user32 is not None
        if not self._user32.SetCursorPos(target_x, target_y):
            self._message = "mouse move failed"
            return False
        self._message = f"mouse move {target_x}, {target_y}"
        return True

    def left_down(self) -> bool:
        if self._backend == "mac":
            if not self.available:
                self._message = "mouse unavailable"
                return False
            pos = self.current_position() or (0, 0)
            if not self._mac_mouse_event(self._q.kCGEventLeftMouseDown, pos, self._q.kCGMouseButtonLeft):
                return False
            self._left_down = True
            self._message = "mouse drag start"
            return True
        if not self._send_mouse_event(MOUSEEVENTF_LEFTDOWN):
            return False
        self._left_down = True
        self._message = "mouse drag start"
        return True

    def left_up(self) -> bool:
        if not self.available:
            self._message = "mouse unavailable"
            return False
        if self._backend == "mac":
            if self._left_down:
                pos = self.current_position() or (0, 0)
                if not self._mac_mouse_event(self._q.kCGEventLeftMouseUp, pos, self._q.kCGMouseButtonLeft):
                    return False
            self._left_down = False
            self._message = "mouse drag release"
            return True
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
        if self._backend == "mac":
            pos = self.current_position() or (0, 0)
            if not self._mac_mouse_event(self._q.kCGEventLeftMouseDown, pos, self._q.kCGMouseButtonLeft):
                return False
            if not self._mac_mouse_event(self._q.kCGEventLeftMouseUp, pos, self._q.kCGMouseButtonLeft):
                return False
            self._message = "mouse left click"
            return True
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
        if self._backend == "mac":
            pos = self.current_position() or (0, 0)
            if not self._mac_mouse_event(self._q.kCGEventRightMouseDown, pos, self._q.kCGMouseButtonRight):
                return False
            if not self._mac_mouse_event(self._q.kCGEventRightMouseUp, pos, self._q.kCGMouseButtonRight):
                return False
            self._message = "mouse right click"
            return True
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
        if self._backend == "mac":
            if not self.available:
                self._message = "mouse unavailable"
                return False
            try:
                event = self._q.CGEventCreateScrollWheelEvent(
                    None, self._q.kCGScrollEventUnitLine, 1, steps
                )
                self._q.CGEventPost(self._q.kCGHIDEventTap, event)
            except Exception:
                self._message = "mouse input failed"
                return False
            direction = "up" if steps > 0 else "down"
            self._message = f"mouse scroll {direction} x{abs(steps)}"
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

    def _mac_mouse_event(self, event_type, pos, button) -> bool:
        """Create + post a CGEvent mouse event. Requires Accessibility — if not
        granted, CGEventPost is silently dropped (no exception)."""
        try:
            event = self._q.CGEventCreateMouseEvent(None, event_type, pos, button)
            self._q.CGEventPost(self._q.kCGHIDEventTap, event)
        except Exception:
            self._message = "mouse input failed"
            return False
        return True

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

from __future__ import annotations

import platform

_SYSTEM = platform.system()

# ---------- macOS native overlay ----------
if _SYSTEM == "Darwin":
    try:
        import ctypes
        import objc
        from AppKit import (
            NSFloatingWindowLevel,
            NSMainMenuWindowLevel,
            NSModalPanelWindowLevel,
            NSStatusWindowLevel,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowCollectionBehaviorStationary,
            NSWindowStyleMaskNonactivatingPanel,
        )
        try:
            from AppKit import NSWindowCollectionBehaviorCanJoinAllApplications
        except Exception:
            NSWindowCollectionBehaviorCanJoinAllApplications = None
        try:
            from AppKit import NSWindowCollectionBehaviorIgnoresCycle
        except Exception:
            NSWindowCollectionBehaviorIgnoresCycle = None
        _HAS_MAC = True
    except Exception:
        ctypes = None
        objc = None
        NSFloatingWindowLevel = None
        NSMainMenuWindowLevel = None
        NSModalPanelWindowLevel = None
        NSStatusWindowLevel = None
        NSWindowCollectionBehaviorCanJoinAllSpaces = None
        NSWindowCollectionBehaviorFullScreenAuxiliary = None
        NSWindowCollectionBehaviorCanJoinAllApplications = None
        NSWindowCollectionBehaviorStationary = None
        NSWindowCollectionBehaviorIgnoresCycle = None
        NSWindowStyleMaskNonactivatingPanel = None
        _HAS_MAC = False
else:
    ctypes = None
    objc = None
    NSFloatingWindowLevel = None
    NSMainMenuWindowLevel = None
    NSModalPanelWindowLevel = None
    NSStatusWindowLevel = None
    NSWindowCollectionBehaviorCanJoinAllSpaces = None
    NSWindowCollectionBehaviorFullScreenAuxiliary = None
    NSWindowCollectionBehaviorCanJoinAllApplications = None
    NSWindowCollectionBehaviorStationary = None
    NSWindowCollectionBehaviorIgnoresCycle = None
    NSWindowStyleMaskNonactivatingPanel = None
    _HAS_MAC = False

# ---------- Windows native overlay ----------
if _SYSTEM == "Windows":
    try:
        import ctypes
        from ctypes import wintypes

        _user32 = ctypes.WinDLL("user32", use_last_error=True)
        _dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)

        HWND_TOPMOST = -1
        GWL_EXSTYLE = -20

        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_APPWINDOW = 0x00040000
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_LAYERED = 0x00080000

        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOACTIVATE = 0x0010
        SWP_FRAMECHANGED = 0x0020
        SWP_SHOWWINDOW = 0x0040
        SWP_NOOWNERZORDER = 0x0200

        SW_SHOWNOACTIVATE = 4
        DWMWA_NCRENDERING_POLICY = 2
        DWMNCRP_DISABLED = 1
        DWMWA_WINDOW_CORNER_PREFERENCE = 33
        DWMWCP_DONOTROUND = 1
        DWMWA_BORDER_COLOR = 34
        DWM_COLOR_NONE = 0xFFFFFFFE

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        _user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        _user32.SetWindowPos.restype = wintypes.BOOL

        _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
        _user32.GetWindowRect.restype = wintypes.BOOL

        _user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        _user32.ShowWindow.restype = wintypes.BOOL

        _dwmapi.DwmSetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            wintypes.LPCVOID,
            wintypes.DWORD,
        ]
        _dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long

        if hasattr(_user32, "GetWindowLongPtrW"):
            _GetWindowLongPtr = _user32.GetWindowLongPtrW
            _SetWindowLongPtr = _user32.SetWindowLongPtrW
            _GetWindowLongPtr.argtypes = [wintypes.HWND, ctypes.c_int]
            _GetWindowLongPtr.restype = ctypes.c_longlong
            _SetWindowLongPtr.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_longlong]
            _SetWindowLongPtr.restype = ctypes.c_longlong
        else:
            _GetWindowLongPtr = _user32.GetWindowLongW
            _SetWindowLongPtr = _user32.SetWindowLongW
            _GetWindowLongPtr.argtypes = [wintypes.HWND, ctypes.c_int]
            _GetWindowLongPtr.restype = ctypes.c_long
            _SetWindowLongPtr.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
            _SetWindowLongPtr.restype = ctypes.c_long

        _HAS_WIN = True
    except Exception:
        _user32 = None
        _dwmapi = None
        HWND_TOPMOST = None
        GWL_EXSTYLE = None
        WS_EX_TOOLWINDOW = None
        WS_EX_APPWINDOW = None
        WS_EX_NOACTIVATE = None
        WS_EX_LAYERED = None
        SWP_NOMOVE = None
        SWP_NOSIZE = None
        SWP_NOACTIVATE = None
        SWP_FRAMECHANGED = None
        SWP_SHOWWINDOW = None
        SWP_NOOWNERZORDER = None
        SW_SHOWNOACTIVATE = None
        DWMWA_NCRENDERING_POLICY = None
        DWMNCRP_DISABLED = None
        DWMWA_WINDOW_CORNER_PREFERENCE = None
        DWMWCP_DONOTROUND = None
        DWMWA_BORDER_COLOR = None
        DWM_COLOR_NONE = None
        RECT = None
        _GetWindowLongPtr = None
        _SetWindowLongPtr = None
        _HAS_WIN = False
else:
    _user32 = None
    _dwmapi = None
    HWND_TOPMOST = None
    GWL_EXSTYLE = None
    WS_EX_TOOLWINDOW = None
    WS_EX_APPWINDOW = None
    WS_EX_NOACTIVATE = None
    WS_EX_LAYERED = None
    SWP_NOMOVE = None
    SWP_NOSIZE = None
    SWP_NOACTIVATE = None
    SWP_FRAMECHANGED = None
    SWP_SHOWWINDOW = None
    SWP_NOOWNERZORDER = None
    SW_SHOWNOACTIVATE = None
    DWMWA_NCRENDERING_POLICY = None
    DWMNCRP_DISABLED = None
    DWMWA_WINDOW_CORNER_PREFERENCE = None
    DWMWCP_DONOTROUND = None
    DWMWA_BORDER_COLOR = None
    DWM_COLOR_NONE = None
    RECT = None
    _GetWindowLongPtr = None
    _SetWindowLongPtr = None
    _HAS_WIN = False


def apply_overlay(widget) -> bool:
    if widget is None or not widget.isVisible():
        return False
    if _SYSTEM == "Darwin":
        return _apply_macos_overlay(widget)
    if _SYSTEM == "Windows":
        return _apply_windows_overlay(widget)
    return False


def _apply_macos_overlay(widget) -> bool:
    if not _HAS_MAC or objc is None or ctypes is None:
        return False

    try:
        widget.winId()
        ns_view = objc.objc_object(c_void_p=ctypes.c_void_p(int(widget.winId())))
        ns_window = ns_view.window()
        if ns_window is None:
            return False

        if NSWindowStyleMaskNonactivatingPanel is not None:
            try:
                ns_window.setStyleMask_(int(ns_window.styleMask()) | int(NSWindowStyleMaskNonactivatingPanel))
            except Exception:
                pass

        # Stay above ordinary windows, but avoid extremely aggressive system levels
        # that can disrupt fullscreen/maximized apps.
        level = None
        for candidate in (
            NSStatusWindowLevel,
            NSModalPanelWindowLevel,
            NSMainMenuWindowLevel,
            NSFloatingWindowLevel,
        ):
            if candidate is not None:
                level = int(candidate)
                break
        if level is not None:
            ns_window.setLevel_(level)

        behavior = int(ns_window.collectionBehavior())
        # CanJoinAllSpaces and MoveToActiveSpace are MUTUALLY EXCLUSIVE. Qt sets
        # MoveToActiveSpace (1<<1) on Tool windows, so OR-ing in CanJoinAllSpaces
        # made setCollectionBehavior_ raise NSInternalInconsistencyException and
        # abort the entire overlay setup (the window then stayed a plain window —
        # hiding on app switch and stealing focus). Clear it before requesting
        # all-spaces.
        behavior &= ~(1 << 1)  # NSWindowCollectionBehaviorMoveToActiveSpace
        if NSWindowCollectionBehaviorCanJoinAllSpaces is not None:
            behavior |= int(NSWindowCollectionBehaviorCanJoinAllSpaces)
        if NSWindowCollectionBehaviorStationary is not None:
            behavior |= int(NSWindowCollectionBehaviorStationary)
        if NSWindowCollectionBehaviorIgnoresCycle is not None:
            behavior |= int(NSWindowCollectionBehaviorIgnoresCycle)
        if NSWindowCollectionBehaviorCanJoinAllApplications is not None:
            behavior |= int(NSWindowCollectionBehaviorCanJoinAllApplications)
        elif NSWindowCollectionBehaviorFullScreenAuxiliary is not None:
            behavior |= int(NSWindowCollectionBehaviorFullScreenAuxiliary)
        ns_window.setCollectionBehavior_(behavior)

        for method_name, value in (
            ("setHidesOnDeactivate_", False),
            ("setFloatingPanel_", True),
            ("setBecomesKeyOnlyIfNeeded_", True),
            ("setCanHide_", False),
            ("setReleasedWhenClosed_", False),
            ("setMovableByWindowBackground_", False),
            ("setAcceptsMouseMovedEvents_", True),
            ("setExcludedFromWindowsMenu_", True),
        ):
            try:
                getattr(ns_window, method_name)(value)
            except Exception:
                pass

        # Avoid orderFrontRegardless() here. The mini viewer should remain non-activating.
        try:
            ns_window.orderFront_(None)
        except Exception:
            pass
        return True
    except Exception as exc:
        try:
            import sys as _s
            print(f"[overlay] macOS overlay FAILED: {type(exc).__name__}: {exc}", file=_s.stderr, flush=True)
        except Exception:
            pass
        return False


def _apply_windows_overlay(widget) -> bool:
    if not _HAS_WIN or _user32 is None:
        return False
    try:
        hwnd = int(widget.winId())
        if not hwnd:
            return False

        try:
            style = _GetWindowLongPtr(hwnd, GWL_EXSTYLE)
            style |= int(WS_EX_TOOLWINDOW)
            style |= int(WS_EX_NOACTIVATE)
            style |= int(WS_EX_LAYERED)
            style &= ~int(WS_EX_APPWINDOW)
            _SetWindowLongPtr(hwnd, GWL_EXSTYLE, style)
        except Exception:
            pass

        try:
            policy = ctypes.c_int(int(DWMNCRP_DISABLED))
            _dwmapi.DwmSetWindowAttribute(
                hwnd,
                int(DWMWA_NCRENDERING_POLICY),
                ctypes.byref(policy),
                ctypes.sizeof(policy),
            )
        except Exception:
            pass

        try:
            corner = ctypes.c_int(int(DWMWCP_DONOTROUND))
            _dwmapi.DwmSetWindowAttribute(
                hwnd,
                int(DWMWA_WINDOW_CORNER_PREFERENCE),
                ctypes.byref(corner),
                ctypes.sizeof(corner),
            )
        except Exception:
            pass

        try:
            border = ctypes.c_uint(int(DWM_COLOR_NONE))
            _dwmapi.DwmSetWindowAttribute(
                hwnd,
                int(DWMWA_BORDER_COLOR),
                ctypes.byref(border),
                ctypes.sizeof(border),
            )
        except Exception:
            pass

        try:
            _user32.ShowWindow(hwnd, int(SW_SHOWNOACTIVATE))
        except Exception:
            pass

        left = top = width = height = 0
        try:
            rect = RECT()
            if _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                left = int(rect.left)
                top = int(rect.top)
                width = max(0, int(rect.right - rect.left))
                height = max(0, int(rect.bottom - rect.top))
        except Exception:
            pass

        flags = (
            int(SWP_NOACTIVATE)
            | int(SWP_SHOWWINDOW)
            | int(SWP_FRAMECHANGED)
            | int(SWP_NOOWNERZORDER)
        )
        if width > 0 and height > 0:
            ok = _user32.SetWindowPos(
                hwnd,
                int(HWND_TOPMOST),
                left,
                top,
                width,
                height,
                flags,
            )
        else:
            ok = _user32.SetWindowPos(
                hwnd,
                int(HWND_TOPMOST),
                0,
                0,
                0,
                0,
                flags | int(SWP_NOMOVE) | int(SWP_NOSIZE),
            )
        return bool(ok)
    except Exception:
        return False

# Author: Konstantin Markov

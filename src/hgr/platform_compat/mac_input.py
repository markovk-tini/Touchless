"""macOS synthetic keyboard input via Quartz CGEvent.

Shared foundation for every macOS "type / press keys" feature: dictation text
insertion (text_input_controller), custom-gesture keystrokes, and app keyboard
shortcuts (Chrome mode, media, etc.).

REQUIRES the Accessibility permission — CGEventPost is silently dropped without
it (no exception), so callers should ensure the app is Accessibility-trusted
(see platform_compat.capabilities.is_accessibility_trusted).

Author: Konstantin Markov (macOS port)
"""
from __future__ import annotations

import sys
import time

_q = None
if sys.platform == "darwin":
    try:
        import Quartz as _q  # type: ignore
    except Exception:
        _q = None

# CGEventFlags modifier masks (kCGEventFlagMask*).
CMD = 1 << 20      # Command
SHIFT = 1 << 17    # Shift
OPTION = 1 << 19   # Option / Alt
CONTROL = 1 << 18  # Control

# ANSI virtual keycodes (kVK_*) for the keys we synthesize by name.
KEYCODES = {
    "a": 0x00, "c": 0x08, "v": 0x09, "x": 0x07, "z": 0x06, "s": 0x01,
    "t": 0x11, "r": 0x0F, "n": 0x2D, "w": 0x0D, "l": 0x25, "h": 0x04,
    "d": 0x02, "f": 0x03, "g": 0x05, "y": 0x10, "0": 0x1D, "9": 0x19,
    "left": 0x7B, "right": 0x7C, "up": 0x7E, "down": 0x7D,
    "return": 0x24, "enter": 0x24, "tab": 0x30, "delete": 0x33,
    "backspace": 0x33, "escape": 0x35, "space": 0x31,
    "[": 0x21, "]": 0x1E,
}


def available() -> bool:
    """True when CGEvent synthesis is usable (macOS + pyobjc Quartz present).
    Does NOT verify the Accessibility grant — events may still be dropped."""
    return _q is not None


def tap(key, *, cmd: bool = False, shift: bool = False, option: bool = False, control: bool = False) -> bool:
    """Press + release a key with optional modifiers.

    `key` may be a name in KEYCODES (e.g. "t", "left", "delete") or an int
    virtual keycode. Returns False if Quartz is unavailable or the key is
    unknown. Note: a True return only means the event was posted, not that the
    OS delivered it (Accessibility may be ungranted)."""
    if _q is None:
        return False
    keycode = KEYCODES.get(key) if isinstance(key, str) else int(key)
    if keycode is None:
        return False
    flags = 0
    if cmd:
        flags |= CMD
    if shift:
        flags |= SHIFT
    if option:
        flags |= OPTION
    if control:
        flags |= CONTROL
    try:
        down = _q.CGEventCreateKeyboardEvent(None, keycode, True)
        up = _q.CGEventCreateKeyboardEvent(None, keycode, False)
        if flags:
            _q.CGEventSetFlags(down, flags)
            _q.CGEventSetFlags(up, flags)
        _q.CGEventPost(_q.kCGHIDEventTap, down)
        _q.CGEventPost(_q.kCGHIDEventTap, up)
        return True
    except Exception:
        return False


def type_text(text: str) -> bool:
    """Type an arbitrary Unicode string (layout-independent) via
    CGEventKeyboardSetUnicodeString — works for any character/emoji."""
    if _q is None:
        return False
    if not text:
        return True
    try:
        down = _q.CGEventCreateKeyboardEvent(None, 0, True)
        _q.CGEventKeyboardSetUnicodeString(down, len(text), text)
        _q.CGEventPost(_q.kCGHIDEventTap, down)
        up = _q.CGEventCreateKeyboardEvent(None, 0, False)
        _q.CGEventKeyboardSetUnicodeString(up, len(text), text)
        _q.CGEventPost(_q.kCGHIDEventTap, up)
        return True
    except Exception:
        return False


def paste_text(text: str) -> bool:
    """Put `text` on the clipboard (NSPasteboard) and paste with Cmd+V.

    Preferred for dictation: instant for long text and inserts at the caret of
    the focused app. Overwrites the clipboard (matches the Windows behavior)."""
    if _q is None:
        return False
    try:
        from AppKit import NSPasteboard, NSPasteboardTypeString  # type: ignore

        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(str(text), NSPasteboardTypeString)
    except Exception:
        return False
    time.sleep(0.03)  # let the pasteboard settle before Cmd+V
    return tap("v", cmd=True)


def backspace(count: int = 1) -> bool:
    """Send `count` backspace (delete) key presses."""
    ok = _q is not None
    for _ in range(max(0, int(count))):
        ok = tap("delete") and ok
    return ok

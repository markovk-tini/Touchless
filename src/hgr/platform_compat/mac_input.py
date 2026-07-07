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
    # letters
    "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05,
    "z": 0x06, "x": 0x07, "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C,
    "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10, "t": 0x11, "o": 0x1F,
    "u": 0x20, "i": 0x22, "p": 0x23, "l": 0x25, "j": 0x26, "k": 0x28,
    "n": 0x2D, "m": 0x2E,
    # digits
    "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "5": 0x17, "6": 0x16,
    "7": 0x1A, "8": 0x1C, "9": 0x19, "0": 0x1D,
    # function keys
    "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60, "f6": 0x61,
    "f7": 0x62, "f8": 0x64, "f9": 0x65, "f10": 0x6D, "f11": 0x67, "f12": 0x6F,
    # navigation / editing
    "left": 0x7B, "right": 0x7C, "up": 0x7E, "down": 0x7D,
    "return": 0x24, "enter": 0x24, "tab": 0x30, "delete": 0x33,
    "backspace": 0x33, "escape": 0x35, "esc": 0x35, "space": 0x31,
    "home": 0x73, "end": 0x77, "pageup": 0x74, "pagedown": 0x79,
    "[": 0x21, "]": 0x1E, "-": 0x1B, "=": 0x18, ";": 0x29, "'": 0x27,
    ",": 0x2B, ".": 0x2F, "/": 0x2C, "\\": 0x2A, "`": 0x32,
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


# NX_KEYTYPE aux-control-button codes for system media control.
NX_KEYTYPE_PLAY = 16
NX_KEYTYPE_NEXT = 17
NX_KEYTYPE_PREVIOUS = 18
NX_KEYTYPE_FAST = 19
NX_KEYTYPE_REWIND = 20


def media_key(key_code: int) -> bool:
    """Post a system media key (play/pause, next, previous, …) via the
    NSSystemDefined aux-control-button event — the SAME event the physical
    media keys emit, so it controls whatever app currently owns Now Playing
    (Apple Music, the Spotify app, a browser video, …) rather than any one app.

    `key_code` is one of the NX_KEYTYPE_* constants. Requires Accessibility
    (like every CGEventPost here). Returns False if Quartz/AppKit is
    unavailable or posting fails."""
    if _q is None:
        return False
    try:
        from AppKit import NSEvent  # type: ignore

        ns_system_defined = 14  # NSEventType.systemDefined
        aux_subtype = 8         # NX_SUBTYPE_AUX_CONTROL_BUTTONS
        for down in (True, False):
            # data1 encodes the key + its up/down state (0xA down, 0xB up);
            # modifierFlags mirrors it in its high byte.
            data1 = (int(key_code) << 16) | ((0xA if down else 0xB) << 8)
            event = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                ns_system_defined,
                (0.0, 0.0),
                0xA00 if down else 0xB00,
                0,
                0,
                None,
                aux_subtype,
                data1,
                -1,
            )
            if event is None:
                return False
            cg_event = event.CGEvent()
            if cg_event is None:
                return False
            _q.CGEventPost(_q.kCGHIDEventTap, cg_event)
        return True
    except Exception:
        return False


def media_play_pause() -> bool:
    return media_key(NX_KEYTYPE_PLAY)


def media_next() -> bool:
    return media_key(NX_KEYTYPE_NEXT)


def media_previous() -> bool:
    return media_key(NX_KEYTYPE_PREVIOUS)

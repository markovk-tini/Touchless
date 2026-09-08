from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import time
import webbrowser
from ctypes import wintypes
from typing import Dict, Iterable, List, Optional

from .registry import Action


# Virtual-key codes we map from friendly names. Expanded on demand — this
# covers the common dictation/shortcut keys and everything that shows up in
# standard Ctrl/Alt/Shift/Win combos. Lowercase everywhere for matching.
_VK_NAME_MAP: Dict[str, int] = {
    "backspace": 0x08, "bksp": 0x08,
    "tab": 0x09,
    "enter": 0x0D, "return": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11, "control": 0x11,
    "alt": 0x12,
    "pause": 0x13,
    "capslock": 0x14, "caps": 0x14,
    "esc": 0x1B, "escape": 0x1B,
    "space": 0x20, " ": 0x20,
    "pageup": 0x21, "pagedown": 0x22,
    "end": 0x23, "home": 0x24,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "printscreen": 0x2C, "insert": 0x2D, "delete": 0x2E, "del": 0x2E,
    "win": 0x5B, "windows": 0x5B, "meta": 0x5B, "super": 0x5B,
    "apps": 0x5D,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74,
    "f6": 0x75, "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79,
    "f11": 0x7A, "f12": 0x7B,
    "numlock": 0x90, "scrolllock": 0x91,
    ";": 0xBA, "=": 0xBB, ",": 0xBC, "-": 0xBD, ".": 0xBE, "/": 0xBF,
    "`": 0xC0, "[": 0xDB, "\\": 0xDC, "]": 0xDD, "'": 0xDE,
}
for _ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    _VK_NAME_MAP[_ch.lower()] = ord(_ch)
for _d in "0123456789":
    _VK_NAME_MAP[_d] = ord(_d)


# --- Minimal SendInput bindings. We deliberately do NOT depend on the
# TextInputController module — this package stays self-contained. The bits
# we need are small and stable.

_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004

_ULONG_PTR = ctypes.c_size_t


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _MOUSEINPUT(ctypes.Structure):
    """Only here so the union below sizes correctly (28 bytes on 64-bit).
    SendInput rejects calls whose cbSize doesn't match Windows' INPUT
    struct size — and Windows' union covers all three input variants, of
    which MOUSEINPUT is the largest."""
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", _KEYBDINPUT),
        ("mi", _MOUSEINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _INPUT_UNION),
    ]


def _resolve_vk(name: str) -> int:
    key = name.strip().lower()
    if not key:
        raise ValueError("empty key name")
    if key in _VK_NAME_MAP:
        return _VK_NAME_MAP[key]
    raise ValueError(f"unknown key name: {name!r}")


# Load a PRIVATE handle to user32 so argtypes set on our SendInput
# function pointer don't collide with another module's bindings. The
# dictation path (text_input_controller) sets `SendInput.argtypes` on
# `ctypes.windll.user32.SendInput` with ITS own INPUT struct type —
# because windll caches one shared instance, that argtypes setting
# applies globally and ctypes then rejects our `_INPUT` array as
# "expected LP_INPUT instance instead of _INPUT_Array_N". Loading a
# fresh WinDLL gives us our own function-pointer slot to configure.
_USER32_SENDINPUT = None


def _resolve_send_input():
    global _USER32_SENDINPUT
    if _USER32_SENDINPUT is not None:
        return _USER32_SENDINPUT
    if platform.system() != "Windows":
        return None
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SendInput.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(_INPUT),
            ctypes.c_int,
        ]
        user32.SendInput.restype = ctypes.c_uint
        _USER32_SENDINPUT = user32.SendInput
    except Exception as exc:
        print(f"[custom-gestures] SendInput resolve failed: {exc}")
        _USER32_SENDINPUT = None
    return _USER32_SENDINPUT


def _send_inputs(events: List[_INPUT]) -> bool:
    if not events:
        return True
    send = _resolve_send_input()
    if send is None:
        return False
    n = len(events)
    arr_type = _INPUT * n
    arr = arr_type(*events)
    sent = send(
        n,
        ctypes.cast(arr, ctypes.POINTER(_INPUT)),
        ctypes.sizeof(_INPUT),
    )
    return int(sent) == n


def _vk_press(vk: int) -> List[_INPUT]:
    down = _INPUT(type=_INPUT_KEYBOARD, u=_INPUT_UNION(ki=_KEYBDINPUT(
        wVk=vk, wScan=0, dwFlags=0, time=0, dwExtraInfo=_ULONG_PTR(0),
    )))
    up = _INPUT(type=_INPUT_KEYBOARD, u=_INPUT_UNION(ki=_KEYBDINPUT(
        wVk=vk, wScan=0, dwFlags=_KEYEVENTF_KEYUP, time=0, dwExtraInfo=_ULONG_PTR(0),
    )))
    return [down, up]


def _unicode_press(ch: str) -> List[_INPUT]:
    code = ord(ch)
    down = _INPUT(type=_INPUT_KEYBOARD, u=_INPUT_UNION(ki=_KEYBDINPUT(
        wVk=0, wScan=code, dwFlags=_KEYEVENTF_UNICODE, time=0, dwExtraInfo=_ULONG_PTR(0),
    )))
    up = _INPUT(type=_INPUT_KEYBOARD, u=_INPUT_UNION(ki=_KEYBDINPUT(
        wVk=0, wScan=code,
        dwFlags=_KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP,
        time=0, dwExtraInfo=_ULONG_PTR(0),
    )))
    return [down, up]


# --- Public executors ---


def execute_keystroke(key: str) -> bool:
    """Press and release a single named key (e.g., 'enter', 'f5', 'a')."""
    if platform.system() == "Darwin":
        from ..platform_compat import mac_input

        return mac_input.tap(str(key or "").strip().lower())
    vk = _resolve_vk(key)
    return _send_inputs(_vk_press(vk))


def execute_hotkey(keys: Iterable[str]) -> bool:
    """Press a combo like Ctrl+Shift+T. Holds all modifiers while the final
    key fires, then releases in reverse order — standard chord semantics."""
    keys = list(keys)
    if platform.system() == "Darwin":
        from ..platform_compat import mac_input

        # Map Ctrl/Cmd/Win -> Cmd (the common cross-platform intent, e.g. a
        # user-bound "Ctrl+C" means Cmd+C on macOS). Shift/Alt map directly.
        mods = {"cmd": False, "shift": False, "option": False, "control": False}
        main_key = None
        for raw in keys:
            token = str(raw or "").strip().lower()
            if token in ("ctrl", "control", "cmd", "command", "win", "meta", "super"):
                mods["cmd"] = True
            elif token == "shift":
                mods["shift"] = True
            elif token in ("alt", "option", "opt"):
                mods["option"] = True
            elif token:
                main_key = token
        if main_key is None:
            return False
        return mac_input.tap(main_key, **mods)
    vks = [_resolve_vk(k) for k in keys]
    if not vks:
        return False
    # Press all down, then release all up in reverse.
    downs: List[_INPUT] = []
    ups: List[_INPUT] = []
    for vk in vks:
        downs.append(_INPUT(type=_INPUT_KEYBOARD, u=_INPUT_UNION(ki=_KEYBDINPUT(
            wVk=vk, wScan=0, dwFlags=0, time=0, dwExtraInfo=_ULONG_PTR(0),
        ))))
    for vk in reversed(vks):
        ups.append(_INPUT(type=_INPUT_KEYBOARD, u=_INPUT_UNION(ki=_KEYBDINPUT(
            wVk=vk, wScan=0, dwFlags=_KEYEVENTF_KEYUP, time=0, dwExtraInfo=_ULONG_PTR(0),
        ))))
    return _send_inputs(downs + ups)


def execute_text(text: str) -> bool:
    """Type a literal string via Unicode key events. Slower than clipboard
    paste but avoids clobbering the clipboard and works in more targets.

    Characters are sent exactly as stored — no URL-encoding, escaping,
    or other substitution (a space stays a space, never '%20')."""
    if platform.system() == "Darwin":
        from ..platform_compat import mac_input

        return mac_input.type_text(str(text or ""))
    events: List[_INPUT] = []
    for ch in text:
        events.extend(_unicode_press(ch))
    return _send_inputs(events)


def _mistaken_https_plain_text(url: str) -> Optional[str]:
    """Detect legacy payloads like 'https://my snap' where the wizard
    auto-prefixed https:// onto plain words. Those open in a browser as
    'https://my%20snap'. Returns the literal text to type instead, or
    None when `url` looks like a real URL."""
    raw = (url or "").strip()
    if not raw:
        return None
    lower = raw.lower()
    prefix = ""
    for candidate in ("https://", "http://"):
        if lower.startswith(candidate):
            prefix = candidate
            break
    if not prefix:
        return None
    rest = raw[len(prefix):]
    host = rest.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    # Real hosts have a dot (example.com) or no whitespace. Spaced
    # hosts with no dot were almost certainly meant as typed text.
    if " " in host and "." not in host and ":" not in host:
        return rest
    return None


def execute_open_url(url: str) -> bool:
    # Never percent-encode in our layer. If this was plain text that
    # accidentally got an https:// prefix, type it literally instead
    # of handing the browser a space that becomes %20.
    plain = _mistaken_https_plain_text(url)
    if plain is not None:
        return execute_text(plain)
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


def execute_run_command(command: str, *, shell: bool = True) -> bool:
    """Run a shell command detached from our process. Intentionally does not
    wait for the child or capture output — custom gestures should fire-and-
    forget to avoid blocking the gesture pipeline."""
    try:
        subprocess.Popen(
            command,
            shell=shell,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        return True
    except Exception:
        return False


def execute_open_file(path: str) -> bool:
    """Open a file with the OS's default handler. Word docs open in
    Word, PNGs in Photos, MP4s in the default player, .url files in
    the browser, etc. Equivalent to double-clicking in Explorer.

    Strips wrapping quotes from the path because Explorer's 'Copy as
    path' command surrounds the path with double-quotes — pasting
    that straight in shouldn't break the binding."""
    cleaned = path.strip().strip('"').strip("'")
    if not cleaned:
        return False
    try:
        if platform.system() == "Windows":
            os.startfile(cleaned)
            return True
        # Defensive cross-platform fallback. Touchless ships
        # Windows-only today, but keeping this branch means a
        # custom-gesture action authored on Windows still does the
        # right thing if a future macOS/Linux build picks it up.
        opener = "open" if platform.system() == "Darwin" else "xdg-open"
        subprocess.Popen(
            [opener, cleaned],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        return True
    except Exception:
        return False


def execute(action: Action) -> bool:
    """Dispatch an Action to the right executor. Returns True on success,
    False on failure. A 'noop' action always succeeds (useful for disabling
    a gesture without deleting it)."""
    kind = (action.kind or "").lower()
    payload = action.payload or {}

    if kind == "noop":
        return True
    if kind == "keystroke":
        key = str(payload.get("key", ""))
        if not key:
            return False
        return execute_keystroke(key)
    if kind == "hotkey":
        keys = payload.get("keys")
        if not isinstance(keys, (list, tuple)) or not keys:
            return False
        return execute_hotkey([str(k) for k in keys])
    if kind == "text":
        text = str(payload.get("text", ""))
        if not text:
            return False
        return execute_text(text)
    if kind == "open_url":
        url = str(payload.get("url", "")).strip()
        if not url:
            return False
        return execute_open_url(url)
    if kind == "run_command":
        cmd = str(payload.get("command", "")).strip()
        if not cmd:
            return False
        return execute_run_command(cmd, shell=bool(payload.get("shell", True)))
    if kind == "open_file":
        path = str(payload.get("path", "")).strip()
        if not path:
            return False
        return execute_open_file(path)
    if kind == "show_overlay_drawing":
        # Qt-coupled action — the runner intercepts this kind before
        # dispatching to execute() and routes it to a callback that
        # talks to the GUI thread (see CustomGestureRunner). We
        # never reach this branch in practice, but return True here
        # so that if the runner falls back to execute() for any
        # reason the cooldown still ticks (preventing a tight
        # repeat-fire loop) instead of looking like a failed action.
        return True
    return False


def describe(action: Action) -> str:
    """Human-readable one-liner for logs/UI."""
    kind = (action.kind or "").lower()
    p = action.payload or {}
    if kind == "noop":
        return "(no action)"
    if kind == "keystroke":
        return f"press key: {p.get('key', '?')}"
    if kind == "hotkey":
        keys = p.get("keys") or []
        return "hotkey: " + "+".join(str(k) for k in keys)
    if kind == "text":
        txt = str(p.get("text", ""))
        preview = txt if len(txt) <= 40 else txt[:37] + "..."
        return f"type text: {preview!r}"
    if kind == "open_url":
        return f"open URL: {p.get('url', '?')}"
    if kind == "run_command":
        cmd = str(p.get("command", ""))
        preview = cmd if len(cmd) <= 60 else cmd[:57] + "..."
        return f"run: {preview}"
    if kind == "open_file":
        path = str(p.get("path", ""))
        preview = path if len(path) <= 60 else "..." + path[-57:]
        return f"open file: {preview}"
    if kind == "show_overlay_drawing":
        return f"show drawing: {p.get('filename', '?')}"
    return f"unknown action: {kind}"


def cooldown_seconds(action: Action) -> float:
    """How long to suppress re-triggering the same gesture after an
    execution, in seconds. Default is 2.0s — long enough that the user has
    to deliberately re-engage the gesture rather than have it spam-fire
    while held. Overridable per action via payload['cooldown_s']."""
    raw = (action.payload or {}).get("cooldown_s")
    try:
        if raw is not None:
            return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    return 2.0


_LAST_FIRED_AT: Dict[str, float] = {}


def fire_once(gesture_name: str, action: Action) -> bool:
    """Execute the action, but only if the cooldown for this gesture name
    has elapsed. Returns True if the action was dispatched, False if it was
    suppressed by the cooldown OR if execution failed."""
    now = time.monotonic()
    last = _LAST_FIRED_AT.get(gesture_name, 0.0)
    if now - last < cooldown_seconds(action):
        return False
    _LAST_FIRED_AT[gesture_name] = now
    return execute(action)


def reset_cooldowns() -> None:
    _LAST_FIRED_AT.clear()

# Author: Konstantin Markov

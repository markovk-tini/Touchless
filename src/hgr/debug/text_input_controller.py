from __future__ import annotations

import ctypes
import platform
import subprocess
import threading
import time
from ctypes import wintypes


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_BACK = 0x08
VK_H = 0x48
VK_LWIN = 0x5B
VK_MENU = 0x12
VK_RETURN = 0x0D
VK_TAB = 0x09
VK_LEFT = 0x25
VK_RIGHT = 0x27
SW_RESTORE = 9
ASFW_ANY = 0xFFFFFFFF
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

ULONG_PTR = wintypes.WPARAM


def _compute_replace_edit(previous_text: str, new_text: str) -> tuple[int, int, str, int]:
    previous = str(previous_text or "")
    new = str(new_text or "")
    if previous == new:
        return 0, 0, "", 0

    prefix_len = 0
    max_prefix = min(len(previous), len(new))
    while prefix_len < max_prefix and previous[prefix_len] == new[prefix_len]:
        prefix_len += 1

    max_suffix = min(len(previous) - prefix_len, len(new) - prefix_len)
    suffix_len = 0
    while suffix_len < max_suffix and previous[-(suffix_len + 1)] == new[-(suffix_len + 1)]:
        suffix_len += 1

    old_mid_end = len(previous) - suffix_len if suffix_len else len(previous)
    new_mid_end = len(new) - suffix_len if suffix_len else len(new)
    old_mid = previous[prefix_len:old_mid_end]
    new_mid = new[prefix_len:new_mid_end]
    return suffix_len, len(old_mid), new_mid, suffix_len


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


# MOUSEINPUT and HARDWAREINPUT are declared purely so the INPUT union has
# the same size Windows expects (MOUSEINPUT is the largest member on x64
# at 28 bytes, which makes sizeof(INPUT) == 40). Without this, SendInput
# silently rejects events because cbSize doesn't match the OS definition.
class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class INPUT_UNION(ctypes.Union):
    _fields_ = (
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    )


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = (
        ("type", wintypes.DWORD),
        ("union", INPUT_UNION),
    )


class TextInputController:
    def __init__(self) -> None:
        self._win = platform.system() == "Windows"
        self._mac = platform.system() == "Darwin"
        # macOS types into the frontmost app via Quartz CGEvent (no HWND focus
        # tracking); see platform_compat.mac_input. Requires Accessibility.
        self._available = self._win or self._mac
        self._message = "text input ready" if self._available else "text input unavailable on this platform"
        self._user32 = ctypes.windll.user32 if self._win else None
        self._kernel32 = ctypes.windll.kernel32 if self._win else None
        self._target_hwnd: int | None = None
        self._last_external_hwnd: int | None = None
        self._own_pid: int = int(self._kernel32.GetCurrentProcessId()) if self._kernel32 is not None else 0
        self._io_lock = threading.RLock()
        if self._available and self._user32 is not None:
            try:
                self._user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
                self._user32.SendInput.restype = wintypes.UINT
                self._user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
                self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD
                self._user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
                self._user32.AttachThreadInput.restype = wintypes.BOOL
                self._user32.SetForegroundWindow.argtypes = [wintypes.HWND]
                self._user32.SetForegroundWindow.restype = wintypes.BOOL
                self._user32.SetFocus.argtypes = [wintypes.HWND]
                self._user32.SetFocus.restype = wintypes.HWND
                self._user32.SetActiveWindow.argtypes = [wintypes.HWND]
                self._user32.SetActiveWindow.restype = wintypes.HWND
                self._user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
                self._user32.ShowWindow.restype = wintypes.BOOL
                self._user32.BringWindowToTop.argtypes = [wintypes.HWND]
                self._user32.BringWindowToTop.restype = wintypes.BOOL
                self._user32.IsWindow.argtypes = [wintypes.HWND]
                self._user32.IsWindow.restype = wintypes.BOOL
                self._user32.AllowSetForegroundWindow.argtypes = [wintypes.DWORD]
                self._user32.AllowSetForegroundWindow.restype = wintypes.BOOL
                self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
                self._user32.IsWindowVisible.restype = wintypes.BOOL
                self._user32.IsIconic.argtypes = [wintypes.HWND]
                self._user32.IsIconic.restype = wintypes.BOOL
                self._user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
                self._user32.GetClassNameW.restype = ctypes.c_int
                self._user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
                self._user32.GetWindowTextW.restype = ctypes.c_int
                self._user32.EnumWindows.argtypes = [ctypes.c_void_p, wintypes.LPARAM]
                self._user32.EnumWindows.restype = wintypes.BOOL
                self._user32.OpenClipboard.argtypes = [wintypes.HWND]
                self._user32.OpenClipboard.restype = wintypes.BOOL
                self._user32.CloseClipboard.argtypes = []
                self._user32.CloseClipboard.restype = wintypes.BOOL
                self._user32.EmptyClipboard.argtypes = []
                self._user32.EmptyClipboard.restype = wintypes.BOOL
                self._user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
                self._user32.SetClipboardData.restype = wintypes.HANDLE
                self._kernel32.GetCurrentThreadId.argtypes = []
                self._kernel32.GetCurrentThreadId.restype = wintypes.DWORD
                self._kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
                self._kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
                self._kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
                self._kernel32.GlobalLock.restype = wintypes.LPVOID
                self._kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
                self._kernel32.GlobalUnlock.restype = wintypes.BOOL
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return self._available

    @property
    def message(self) -> str:
        return self._message

    def capture_target_window(self) -> bool:
        if self._mac:
            # macOS inserts into whatever app is frontmost at insert time, so
            # there's no window handle to capture — this is a no-op success.
            self._message = "dictation target captured"
            return True
        if not self._available or self._user32 is None:
            self._message = "text input unavailable on this platform"
            return False
        self.remember_active_window()
        hwnd = self._foreground_window()
        if hwnd > 0 and not self._is_own_window(hwnd):
            self._target_hwnd = hwnd
            self._last_external_hwnd = hwnd
            self._message = "dictation target captured"
            return True
        fallback = int(self._last_external_hwnd or 0)
        if fallback > 0:
            self._target_hwnd = fallback
            self._message = "dictation target captured"
            return True
        if int(self._target_hwnd or 0) > 0:
            self._message = "dictation target captured"
            return True
        self._message = "could not capture dictation target"
        return False

    def clear_target_window(self) -> None:
        self._target_hwnd = None

    def remember_active_window(self) -> bool:
        if not self._available or self._user32 is None:
            return False
        hwnd = self._foreground_window()
        if hwnd <= 0 or self._is_own_window(hwnd):
            return False
        self._last_external_hwnd = hwnd
        return True

    def focus_target_window(self) -> bool:
        ok = self._restore_target_window()
        if ok:
            self._message = "dictation target focused"
        else:
            self._message = "could not focus dictation target"
        return ok
    def insert_text(self, text: str, *, prefer_paste: bool = True) -> bool:
        with self._io_lock:
            if self._mac:
                from ..platform_compat import mac_input

                payload = str(text or "")
                if not payload:
                    self._message = "dictation text missing"
                    return False
                ok = mac_input.paste_text(payload) if prefer_paste else mac_input.type_text(payload)
                if not ok and prefer_paste:
                    ok = mac_input.type_text(payload)
                if ok:
                    self._message = f"inserted dictated text ({len(payload)} chars)"
                    self._publish_to_iris_bridge(payload)
                else:
                    self._message = "could not insert dictated text (grant Accessibility)"
                return ok
            if not self._available or self._user32 is None:
                self._message = "text input unavailable on this platform"
                return False
            payload = str(text or "")
            if not payload:
                self._message = "dictation text missing"
                return False
            if not self._restore_target_window():
                self._message = "could not focus dictation target"
                return False
            time.sleep(0.06)
            if prefer_paste and self._paste_text(payload):
                self._message = f"inserted dictated text ({len(payload)} chars)"
                self._publish_to_iris_bridge(payload)
                return True

            inputs = self._text_to_inputs(payload)
            if not inputs:
                self._message = "dictation text missing"
                return False

            if not self._send_inputs(inputs):
                self._message = "could not insert dictated text"
                return False
            self._message = f"inserted dictated text ({len(payload)} chars)"
            self._publish_to_iris_bridge(payload)
            return True

    def _publish_to_iris_bridge(self, text: str) -> None:
        """Phase-3 wiring: tell Iris what was just dictated so the
        manager / planner can answer 'summarize what I just typed'
        without copy-paste. Best-effort; never block the dictation
        path on a bridge failure."""
        if not text:
            return
        try:
            from hgr.live_api.dictation_bridge import global_bridge
            from hgr.live_api.incognito import is_incognito
            if is_incognito():
                return  # private mode: don't leak typed text
            window_title = self._get_target_window_title()
            global_bridge().publish_dictation_text(
                text, window_title=window_title)
        except Exception:
            pass  # bridge optional — never break dictation

    def _get_target_window_title(self) -> str:
        """Best-effort window title for the current dictation target.
        Empty string when unavailable."""
        try:
            if (self._user32 is None or self._target_hwnd is None
                    or int(self._target_hwnd) <= 0):
                return ""
            import ctypes
            buf = ctypes.create_unicode_buffer(256)
            self._user32.GetWindowTextW(self._target_hwnd, buf, 256)
            return buf.value or ""
        except Exception:
            return ""


    def remove_text(self, char_count: int) -> bool:
        with self._io_lock:
            if self._mac:
                from ..platform_compat import mac_input

                count = max(0, int(char_count))
                if count <= 0:
                    self._message = "nothing to remove"
                    return True
                ok = mac_input.backspace(count)
                self._message = (f"updated dictated text (-{count} chars)" if ok
                                 else "could not update dictated text")
                return ok
            if not self._available or self._user32 is None:
                self._message = "text input unavailable on this platform"
                return False
            count = max(0, int(char_count))
            if count <= 0:
                self._message = "nothing to remove"
                return True
            if not self._restore_target_window():
                self._message = "could not focus dictation target"
                return False

            inputs: list[INPUT] = []
            for _index in range(count):
                inputs.extend(self._vk_inputs(VK_BACK))
            if not self._send_inputs(inputs):
                self._message = "could not update dictated text"
                return False
            self._message = f"updated dictated text (-{count} chars)"
            return True

    def replace_text(self, previous_text: str, new_text: str) -> bool:
        with self._io_lock:
            prior = str(previous_text or "")
            replacement = str(new_text or "")
            if prior == replacement:
                self._message = "updated dictated text"
                return True
            if self._mac:
                from ..platform_compat import mac_input

                left_moves, backspaces, ins_text, right_moves = _compute_replace_edit(prior, replacement)
                ok = True
                for _ in range(left_moves):
                    ok = mac_input.tap("left") and ok
                if backspaces:
                    ok = mac_input.backspace(backspaces) and ok
                if ins_text:
                    ok = (mac_input.paste_text(ins_text) or mac_input.type_text(ins_text)) and ok
                for _ in range(right_moves):
                    ok = mac_input.tap("right") and ok
                self._message = "updated dictated text" if ok else "could not update dictated text"
                return ok
            if not self._available or self._user32 is None:
                self._message = "text input unavailable on this platform"
                return False
            if not self._restore_target_window():
                self._message = "could not focus dictation target"
                return False
            left_moves, backspaces, insert_text, right_moves = _compute_replace_edit(prior, replacement)
            inputs: list[INPUT] = []
            for _index in range(left_moves):
                inputs.extend(self._vk_inputs(VK_LEFT))
            for _index in range(backspaces):
                inputs.extend(self._vk_inputs(VK_BACK))
            if insert_text:
                inputs.extend(self._text_to_inputs(insert_text))
            for _index in range(right_moves):
                inputs.extend(self._vk_inputs(VK_RIGHT))
            if inputs and not self._send_inputs(inputs):
                self._message = "could not update dictated text"
                return False
            self._message = "updated dictated text"
            return True

    def toggle_windows_dictation(self) -> bool:
        if not self._available or self._user32 is None:
            self._message = "windows dictation unavailable on this platform"
            return False
        try:
            # Sending Win+H as separate events is more reliable for the OS voice typing flyout
            # than batching the whole shortcut into one SendInput array.
            if not self._send_key_down(VK_LWIN):
                return False
            time.sleep(0.03)
            if not self._send_key_down(VK_H):
                self._send_key_up(VK_LWIN)
                return False
            time.sleep(0.03)
            if not self._send_key_up(VK_H):
                self._send_key_up(VK_LWIN)
                return False
            time.sleep(0.03)
            if not self._send_key_up(VK_LWIN):
                return False
        except Exception:
            self._message = "could not send keyboard shortcut"
            return False
        self._message = "sent keyboard shortcut"
        return True

    def start_windows_dictation(self) -> bool:
        if not self._available or self._user32 is None:
            self._message = "windows dictation unavailable on this platform"
            return False

        # Windows' built-in dictation (Win+H) types into whichever control has
        # keyboard focus when the user speaks — so the goal here is just to make
        # sure *some* external text-editor window holds focus by the time we
        # press Win+H. We NEVER hard-fail: if nothing is available, we launch a
        # fresh Notepad so the user always gets to dictate somewhere.

        self.remember_active_window()
        foreground = self._foreground_window()

        # Fast path: user has a non-HGR window foreground (typically because
        # they clicked into Notepad / Word / a browser textbox).
        if foreground > 0 and not self._is_own_window(foreground):
            self._target_hwnd = foreground
            self._last_external_hwnd = foreground
            self._message = "dictation active at the current cursor"
            return self.toggle_windows_dictation()

        # HGR is foreground at the moment of the gesture. Poll briefly in case
        # the user is mid-switch, or our own overlay is transiently on top.
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            time.sleep(0.04)
            self.remember_active_window()
            fg = self._foreground_window()
            if fg > 0 and not self._is_own_window(fg):
                self._target_hwnd = fg
                self._last_external_hwnd = fg
                self._message = "dictation active at the current cursor"
                return self.toggle_windows_dictation()

        # Try to restore the most recent remembered external window.
        target_hwnd = int(self._last_external_hwnd or self._target_hwnd or 0)
        if target_hwnd > 0:
            try:
                if not bool(self._user32.IsWindow(wintypes.HWND(target_hwnd))):
                    target_hwnd = 0
            except Exception:
                target_hwnd = 0
        if target_hwnd > 0:
            self._target_hwnd = target_hwnd
            self._restore_target_window()
            time.sleep(0.12)
            self._message = "dictation active at the current cursor"
            return self.toggle_windows_dictation()

        # No remembered window — scan open top-level windows for a text editor.
        existing = self._find_external_text_window()
        if existing > 0:
            self._target_hwnd = existing
            self._last_external_hwnd = existing
            self._restore_target_window()
            time.sleep(0.12)
            self._message = "dictation active at the current cursor"
            return self.toggle_windows_dictation()

        # Nothing at all — spawn Notepad and dictate into it.
        if not self._launch_notepad_for_dictation(timeout=4.0):
            # Even if Notepad launch didn't confirm foreground, still try Win+H
            # so the user gets a chance to dictate if anything catches focus.
            self._message = "dictation started (no text field detected)"
            return self.toggle_windows_dictation()
        time.sleep(0.2)
        self._message = "dictation active in Notepad"
        return self.toggle_windows_dictation()

    def _find_external_text_window(self) -> int:
        if not self._available or self._user32 is None:
            return 0
        preferred_classes = {
            "notepad",
            "edit",
            "richedit",
            "richedit20w",
            "richeditd2dpt",
            "richedit50w",
            "opusapp",  # Word
            "wordpadclass",
            "chrome_widgetwin_1",  # Chromium-based (Chrome, Edge, VS Code)
            "mozillawindowclass",
            "applicationframewindow",  # Modern Win11 Notepad wrapper
        }
        found: list[tuple[int, int]] = []  # (priority, hwnd)

        EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _callback(hwnd, _lparam):
            try:
                if not bool(self._user32.IsWindowVisible(wintypes.HWND(hwnd))):
                    return True
                if bool(self._user32.IsIconic(wintypes.HWND(hwnd))):
                    return True
                if self._is_own_window(int(hwnd)):
                    return True
                class_name = self._window_class_name(int(hwnd)).lower()
                title = self._window_text(int(hwnd))
                if not title and class_name not in preferred_classes:
                    return True
                priority = 1 if class_name in preferred_classes else 3
                if class_name == "notepad" or "notepad" in title.lower():
                    priority = 0
                found.append((priority, int(hwnd)))
            except Exception:
                pass
            return True

        try:
            self._user32.EnumWindows(EnumProc(_callback), 0)
        except Exception:
            return 0
        if not found:
            return 0
        found.sort(key=lambda item: item[0])
        return found[0][1]

    def _window_class_name(self, hwnd: int) -> str:
        if not self._available or self._user32 is None or hwnd <= 0:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(256)
            self._user32.GetClassNameW(wintypes.HWND(hwnd), buf, 256)
            return str(buf.value or "")
        except Exception:
            return ""

    def _window_text(self, hwnd: int) -> str:
        if not self._available or self._user32 is None or hwnd <= 0:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(512)
            self._user32.GetWindowTextW(wintypes.HWND(hwnd), buf, 512)
            return str(buf.value or "")
        except Exception:
            return ""

    def _launch_notepad_for_dictation(self, *, timeout: float = 4.0) -> bool:
        launched = False
        # Try direct Popen first (fastest path); if that fails, fall back
        # to ShellExecuteW via our launch_external helper. Both avoid the
        # `cmd /c start` Norton-flagged pattern.
        try:
            subprocess.Popen(["notepad.exe"], shell=False)
            launched = True
        except Exception:
            pass
        if not launched:
            from ..utils.subprocess_utils import launch_external
            launched = launch_external("notepad.exe")
        if not launched:
            return False
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            time.sleep(0.1)
            hwnd = self._foreground_window()
            if hwnd > 0 and not self._is_own_window(hwnd):
                self._target_hwnd = hwnd
                self._last_external_hwnd = hwnd
                return True
            candidate = self._find_external_text_window()
            if candidate > 0:
                self._target_hwnd = candidate
                self._last_external_hwnd = candidate
                self._restore_target_window()
                time.sleep(0.15)
                return True
        return False

    def stop_windows_dictation(self) -> bool:
        if not self._available or self._user32 is None:
            self._message = "windows dictation unavailable on this platform"
            return False
        self._restore_target_window()
        time.sleep(0.12)
        if not self.toggle_windows_dictation():
            return False
        self._message = "dictation stopped"
        return True

    def _foreground_window(self) -> int:
        if not self._available or self._user32 is None:
            return 0
        try:
            return int(self._user32.GetForegroundWindow() or 0)
        except Exception:
            return 0

    def _restore_target_window(self) -> bool:
        if self._mac:
            return True  # frontmost app is the target; nothing to restore
        if not self._available or self._user32 is None or self._kernel32 is None:
            return False
        target_hwnd = int(self._target_hwnd or self._last_external_hwnd or 0)
        if target_hwnd <= 0:
            return False
        try:
            if not bool(self._user32.IsWindow(wintypes.HWND(target_hwnd))):
                self._message = "dictation target is no longer available"
                return False
        except Exception:
            pass

        foreground = self._foreground_window()
        if foreground == target_hwnd:
            return True

        current_tid = int(self._kernel32.GetCurrentThreadId())
        target_tid = self._window_thread_id(target_hwnd)
        foreground_tid = self._window_thread_id(foreground) if foreground > 0 else 0
        attached: list[tuple[int, int]] = []
        pairs = []
        for a, b in ((current_tid, foreground_tid), (current_tid, target_tid), (target_tid, foreground_tid)):
            if a and b and a != b and (a, b) not in pairs:
                pairs.append((a, b))
        try:
            try:
                self._user32.AllowSetForegroundWindow(ASFW_ANY)
            except Exception:
                pass
            for a, b in pairs:
                try:
                    if bool(self._user32.AttachThreadInput(a, b, True)):
                        attached.append((a, b))
                except Exception:
                    pass
            try:
                self._user32.ShowWindow(wintypes.HWND(target_hwnd), SW_RESTORE)
            except Exception:
                pass
            # Nudge foreground permission with an Alt tap before requesting focus.
            self._tap_alt()
            try:
                self._user32.BringWindowToTop(wintypes.HWND(target_hwnd))
            except Exception:
                pass
            try:
                self._user32.SetForegroundWindow(wintypes.HWND(target_hwnd))
            except Exception:
                pass
            try:
                self._user32.SetActiveWindow(wintypes.HWND(target_hwnd))
            except Exception:
                pass
            try:
                self._user32.SetFocus(wintypes.HWND(target_hwnd))
            except Exception:
                pass
        finally:
            for a, b in reversed(attached):
                try:
                    self._user32.AttachThreadInput(a, b, False)
                except Exception:
                    pass
        time.sleep(0.08)
        return self._foreground_window() == target_hwnd

    def _tap_alt(self) -> None:
        if not self._available or self._user32 is None:
            return
        inputs = [
            self._keyboard_input(virtual_key=VK_MENU, flags=0),
            self._keyboard_input(virtual_key=VK_MENU, flags=KEYEVENTF_KEYUP),
        ]
        array_type = INPUT * len(inputs)
        try:
            self._user32.SendInput(len(inputs), array_type(*inputs), ctypes.sizeof(INPUT))
        except Exception:
            pass

    def _send_shortcut(self, *virtual_keys: int) -> bool:
        if not self._available or self._user32 is None:
            self._message = "text input unavailable on this platform"
            return False
        keys = [int(key) for key in virtual_keys if int(key) > 0]
        if not keys:
            return False
        try:
            for key in keys:
                if not self._send_key_down(key):
                    return False
                time.sleep(0.015)
            for key in reversed(keys):
                if not self._send_key_up(key):
                    return False
                time.sleep(0.015)
        except Exception:
            self._message = "could not send keyboard shortcut"
            return False
        self._message = "sent keyboard shortcut"
        return True

    def _send_key_down(self, virtual_key: int) -> bool:
        if not self._available or self._user32 is None:
            return False
        inp = INPUT * 1
        sent = int(self._user32.SendInput(1, inp(self._keyboard_input(virtual_key=int(virtual_key), flags=0)), ctypes.sizeof(INPUT)))
        return sent == 1

    def _send_key_up(self, virtual_key: int) -> bool:
        if not self._available or self._user32 is None:
            return False
        inp = INPUT * 1
        sent = int(self._user32.SendInput(1, inp(self._keyboard_input(virtual_key=int(virtual_key), flags=KEYEVENTF_KEYUP)), ctypes.sizeof(INPUT)))
        return sent == 1

    def _window_thread_id(self, hwnd: int) -> int:
        if not self._available or self._user32 is None or hwnd <= 0:
            return 0
        try:
            pid = wintypes.DWORD()
            return int(self._user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid)) or 0)
        except Exception:
            return 0

    def _is_own_window(self, hwnd: int) -> bool:
        if not self._available or self._user32 is None or hwnd <= 0:
            return False
        try:
            pid = wintypes.DWORD()
            self._user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
            return int(pid.value) == int(self._own_pid)
        except Exception:
            return False

    def _paste_text(self, text: str) -> bool:
        if not self._available or self._user32 is None or self._kernel32 is None:
            return False
        data = (str(text) + "\0").encode("utf-16-le")
        handle = None
        locked = None
        try:
            if not self._user32.OpenClipboard(None):
                return False
            self._user32.EmptyClipboard()
            handle = self._kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            if not handle:
                self._user32.CloseClipboard()
                return False
            locked = self._kernel32.GlobalLock(handle)
            if not locked:
                self._user32.CloseClipboard()
                return False
            ctypes.memmove(locked, data, len(data))
            self._kernel32.GlobalUnlock(handle)
            locked = None
            if not self._user32.SetClipboardData(CF_UNICODETEXT, handle):
                self._user32.CloseClipboard()
                return False
            handle = None
            self._user32.CloseClipboard()
            time.sleep(0.03)
            return self._send_shortcut(0x11, 0x56)
        except Exception:
            try:
                self._user32.CloseClipboard()
            except Exception:
                pass
            return False
        finally:
            if locked is not None:
                try:
                    self._kernel32.GlobalUnlock(handle)
                except Exception:
                    pass

    def _unicode_inputs(self, text: str) -> list[INPUT]:
        data = text.encode("utf-16-le")
        units = [int.from_bytes(data[index:index + 2], "little") for index in range(0, len(data), 2)]
        payload: list[INPUT] = []
        for unit in units:
            payload.append(self._keyboard_input(scan_code=unit, flags=KEYEVENTF_UNICODE))
            payload.append(self._keyboard_input(scan_code=unit, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        return payload

    def _vk_inputs(self, virtual_key: int) -> list[INPUT]:
        return [
            self._keyboard_input(virtual_key=virtual_key, flags=0),
            self._keyboard_input(virtual_key=virtual_key, flags=KEYEVENTF_KEYUP),
        ]

    def _text_to_inputs(self, text: str) -> list[INPUT]:
        inputs: list[INPUT] = []
        for char in str(text or ""):
            if char == "\n":
                inputs.extend(self._vk_inputs(VK_RETURN))
            elif char == "\t":
                inputs.extend(self._vk_inputs(VK_TAB))
            else:
                inputs.extend(self._unicode_inputs(char))
        return inputs

    def _send_inputs(self, inputs: list[INPUT]) -> bool:
        if not inputs or not self._available or self._user32 is None:
            return not inputs
        array_type = INPUT * len(inputs)
        sent = int(self._user32.SendInput(len(inputs), array_type(*inputs), ctypes.sizeof(INPUT)))
        return sent == len(inputs)

    def _keyboard_input(self, *, virtual_key: int = 0, scan_code: int = 0, flags: int = 0) -> INPUT:
        return INPUT(
            type=INPUT_KEYBOARD,
            union=INPUT_UNION(
                ki=KEYBDINPUT(
                    wVk=int(virtual_key),
                    wScan=int(scan_code),
                    dwFlags=int(flags),
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        )

# Author: Konstantin Markov

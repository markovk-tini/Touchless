from __future__ import annotations

import ctypes
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

from .foreground_window import enumerate_visible_windows, find_chrome_youtube_windows
from .text_input_controller import TextInputController

_VK_MEDIA_NEXT_TRACK = 0xB0
_VK_MEDIA_PREV_TRACK = 0xB1
_VK_MEDIA_PLAY_PAUSE = 0xB3

_VK_SHIFT = 0x10
_VK_CONTROL = 0x11
_VK_KEY_L = 0x4C
_VK_F = 0x46
_VK_T = 0x54
_VK_I = 0x49
_VK_C = 0x43
_VK_J = 0x4A
_VK_K = 0x4B
_VK_L = 0x4C
_VK_N = 0x4E
_VK_P = 0x50
_VK_OEM_COMMA = 0xBC
_VK_OEM_PERIOD = 0xBE
_VK_TAB = 0x09
_VK_RETURN = 0x0D
_VK_UP = 0x26
_VK_DOWN = 0x28
_VK_9 = 0x39

_SW_RESTORE = 9
_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002

_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_ABSOLUTE = 0x8000
_MOUSEEVENTF_VIRTUALDESK = 0x4000

_INPUT_MOUSE = 0
_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79

_SKIP_TEMPLATE_DIRNAME = "youtube_skip"
_SKIP_MATCH_THRESHOLD = 0.75
_SKIP_CLICK_COOLDOWN_SECONDS = 0.8
_TAB_SWITCH_SETTLE_SECONDS = 0.12
_WINDOW_FOCUS_SETTLE_SECONDS = 0.08
_BACKGROUND_TAB_SEARCH_STEPS = 18
_RECENT_YOUTUBE_WINDOW_SECONDS = 6.0
_CAPTIONS_FEEDBACK_SETTLE_SECONDS = 0.28
_CAPTIONS_OCR_TIMEOUT_SECONDS = 3.0
_UIA_ACTION_TIMEOUT_SECONDS = 2.2
_YOUTUBE_SCRIPT_RESULT_PREFIX = "HGR_YT_ACTION_"
_YOUTUBE_SCRIPT_RESULT_PATTERN = re.compile(rf"{_YOUTUBE_SCRIPT_RESULT_PREFIX}([A-Z_]+)__", re.IGNORECASE)
_YOUTUBE_SCRIPT_TITLE_TIMEOUT_SECONDS = 1.2


class YouTubeController:
    """Controls the currently playing YouTube tab via Chrome.

    Detection is a title scan for Chrome windows whose caption contains
    'YouTube'. A VolumeController is used to distinguish 'YouTube tab open
    but paused' from 'YouTube currently emitting audio' when the caller
    wants the stricter signal.
    """

    def __init__(self, volume_controller=None) -> None:
        self._volume_controller = volume_controller
        self._is_windows = platform.system() == "Windows"
        self._message = "YouTube idle"
        self._tab_cache_until = 0.0
        self._tab_cache_value = False
        self._skip_click_cooldown_until = 0.0
        self._last_youtube_hwnd = 0
        self._last_youtube_seen_until = 0.0
        self._text_input = TextInputController() if self._is_windows else None

    @property
    def message(self) -> str:
        return self._message

    def has_youtube_tab(self) -> bool:
        now = time.time()
        if now < self._tab_cache_until:
            return self._tab_cache_value
        try:
            value = bool(find_chrome_youtube_windows())
        except Exception:
            value = False
        if not value:
            value = self._has_recent_youtube_window()
        self._tab_cache_value = value
        self._tab_cache_until = now + 1.0
        return value

    def _send_virtual_key(self, vk: int) -> bool:
        if not self._is_windows:
            return False
        try:
            user32 = ctypes.windll.user32
            user32.keybd_event(wintypes.BYTE(vk), 0, _KEYEVENTF_EXTENDEDKEY, 0)
            user32.keybd_event(wintypes.BYTE(vk), 0, _KEYEVENTF_EXTENDEDKEY | _KEYEVENTF_KEYUP, 0)
            return True
        except Exception as exc:
            self._message = f"YouTube key send failed: {type(exc).__name__}"
            return False

    def _focus_youtube_window(self) -> bool:
        return self._activate_youtube_tab() is not None

    def _window_title(self, hwnd: int) -> str:
        if not self._is_windows or hwnd <= 0:
            return ""
        try:
            user32 = ctypes.windll.user32
            length = int(user32.GetWindowTextLengthW(wintypes.HWND(hwnd)) or 0)
            if length <= 0:
                return ""
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(wintypes.HWND(hwnd), buf, length + 1)
            return str(buf.value or "")
        except Exception:
            return ""

    def _is_window_minimized(self, hwnd: int) -> bool:
        if not self._is_windows or hwnd <= 0:
            return False
        try:
            return bool(ctypes.windll.user32.IsIconic(wintypes.HWND(hwnd)))
        except Exception:
            return False

    def _restore_window(self, hwnd: int) -> bool:
        if not self._is_windows or hwnd <= 0:
            return False
        try:
            return bool(ctypes.windll.user32.ShowWindow(wintypes.HWND(hwnd), _SW_RESTORE))
        except Exception:
            return False

    def _bring_window_to_front(self, hwnd: int) -> bool:
        if not self._is_windows or hwnd <= 0:
            return False
        try:
            user32 = ctypes.windll.user32
            user32.BringWindowToTop(wintypes.HWND(hwnd))
            if user32.SetForegroundWindow(wintypes.HWND(hwnd)):
                time.sleep(_WINDOW_FOCUS_SETTLE_SECONDS)
                return True
            fg = user32.GetForegroundWindow()
            return int(fg) == hwnd if fg else False
        except Exception:
            return False

    def _focus_window_handle(self, hwnd: int, *, restore_if_minimized: bool = True) -> bool:
        if not self._is_windows or hwnd <= 0:
            return False
        if restore_if_minimized and self._is_window_minimized(hwnd):
            self._restore_window(hwnd)
        return self._bring_window_to_front(hwnd)

    def _cycle_chrome_tab(self) -> bool:
        if not self._is_windows:
            return False
        try:
            user32 = ctypes.windll.user32
            user32.keybd_event(wintypes.BYTE(_VK_CONTROL), 0, 0, 0)
            user32.keybd_event(wintypes.BYTE(_VK_TAB), 0, 0, 0)
            user32.keybd_event(wintypes.BYTE(_VK_TAB), 0, _KEYEVENTF_KEYUP, 0)
            user32.keybd_event(wintypes.BYTE(_VK_CONTROL), 0, _KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    def _chrome_window_handles(self) -> list[int]:
        try:
            windows = enumerate_visible_windows()
        except Exception:
            windows = []
        handles: list[int] = []
        seen: set[int] = set()
        for window in windows:
            try:
                hwnd = int(window.hwnd)
            except (AttributeError, TypeError, ValueError):
                continue
            process_name = str(getattr(window, "process_name", "") or "").lower()
            if "chrome" not in process_name or hwnd <= 0 or hwnd in seen:
                continue
            seen.add(hwnd)
            handles.append(hwnd)
        recent_hwnd = self._recent_youtube_window_handle()
        if recent_hwnd is not None and recent_hwnd in handles:
            handles.remove(recent_hwnd)
            handles.insert(0, recent_hwnd)
        return handles

    def _remember_youtube_window(self, hwnd: int) -> None:
        try:
            hwnd_value = int(hwnd)
        except (TypeError, ValueError):
            return
        if hwnd_value <= 0:
            return
        self._last_youtube_hwnd = hwnd_value
        self._last_youtube_seen_until = time.time() + _RECENT_YOUTUBE_WINDOW_SECONDS

    def _recent_youtube_window_handle(self) -> int | None:
        now = time.time()
        if now > self._last_youtube_seen_until or self._last_youtube_hwnd <= 0:
            return None
        recent_hwnd = int(self._last_youtube_hwnd)
        try:
            windows = enumerate_visible_windows()
        except Exception:
            return None
        for window in windows:
            try:
                hwnd = int(window.hwnd)
            except (AttributeError, TypeError, ValueError):
                continue
            if hwnd != recent_hwnd:
                continue
            process_name = str(getattr(window, "process_name", "") or "").lower()
            if "chrome" in process_name:
                return recent_hwnd
            break
        return None

    def _has_recent_youtube_window(self) -> bool:
        return self._recent_youtube_window_handle() is not None

    def _activate_background_youtube_tab(self) -> int | None:
        for hwnd in self._chrome_window_handles():
            if not self._focus_window_handle(hwnd, restore_if_minimized=True):
                continue
            title = self._window_title(hwnd)
            if "youtube" in title.lower():
                self._remember_youtube_window(hwnd)
                return hwnd
            seen_titles: set[str] = set()
            normalized = title.strip().lower()
            if normalized:
                seen_titles.add(normalized)
            for _index in range(_BACKGROUND_TAB_SEARCH_STEPS):
                if not self._cycle_chrome_tab():
                    break
                time.sleep(_TAB_SWITCH_SETTLE_SECONDS)
                title = self._window_title(hwnd)
                normalized = title.strip().lower()
                if "youtube" in normalized:
                    self._remember_youtube_window(hwnd)
                    return hwnd
                if normalized and normalized in seen_titles:
                    break
                if normalized:
                    seen_titles.add(normalized)
        return None

    def _activate_youtube_tab(self) -> int | None:
        if not self._is_windows:
            return None
        try:
            handles = find_chrome_youtube_windows()
        except Exception:
            handles = []
        for handle in handles:
            try:
                hwnd = int(handle.hwnd)
            except (AttributeError, TypeError, ValueError):
                continue
            if self._focus_window_handle(hwnd, restore_if_minimized=True):
                self._remember_youtube_window(hwnd)
                return hwnd
        return self._activate_background_youtube_tab()

    def _send_key_to_youtube(self, vk: int, *, shift: bool = False, hwnd: int | None = None) -> bool:
        target_hwnd = hwnd if hwnd is not None else self._activate_youtube_tab()
        if target_hwnd is None:
            self._message = "YouTube tab not focusable"
            return False
        self._remember_youtube_window(target_hwnd)
        try:
            user32 = ctypes.windll.user32
            if shift:
                user32.keybd_event(wintypes.BYTE(_VK_SHIFT), 0, 0, 0)
            user32.keybd_event(wintypes.BYTE(vk), 0, 0, 0)
            user32.keybd_event(wintypes.BYTE(vk), 0, _KEYEVENTF_KEYUP, 0)
            if shift:
                user32.keybd_event(wintypes.BYTE(_VK_SHIFT), 0, _KEYEVENTF_KEYUP, 0)
            return True
        except Exception as exc:
            self._message = f"YouTube key send failed: {type(exc).__name__}"
            return False

    def toggle_playback(self) -> bool:
        ok = self._send_key_to_youtube(_VK_K)
        self._message = "YouTube play/pause" if ok else self._message
        return ok

    def next_track(self) -> bool:
        ok = self._send_key_to_youtube(_VK_N, shift=True)
        self._message = "YouTube next" if ok else self._message
        return ok

    def previous_track(self) -> bool:
        ok = self._send_key_to_youtube(_VK_P, shift=True)
        self._message = "YouTube previous" if ok else self._message
        return ok

    def step_player_volume(self, direction: int, steps: int = 1) -> bool:
        if direction == 0 or steps <= 0:
            return False
        vk = _VK_UP if direction > 0 else _VK_DOWN
        hwnd = self._activate_youtube_tab()
        if hwnd is None:
            self._message = "YouTube tab not focusable"
            return False
        ok_any = False
        for _ in range(int(max(1, min(steps, 10)))):
            if not self._send_key_to_youtube(vk, hwnd=hwnd):
                break
            ok_any = True
        if ok_any:
            self._message = "YouTube volume up" if direction > 0 else "YouTube volume down"
        return ok_any

    def toggle_fullscreen(self) -> bool:
        ok = self._send_key_to_youtube(_VK_F)
        self._message = "YouTube fullscreen" if ok else self._message
        return ok

    def toggle_theater(self) -> bool:
        hwnd = self._activate_youtube_tab()
        if hwnd is None:
            self._message = "YouTube tab not focusable"
            return False
        ok = self._send_key_to_youtube(_VK_T, hwnd=hwnd)
        self._message = "YouTube theater" if ok else self._message
        return ok

    def toggle_mini_player(self) -> bool:
        ok = self._send_key_to_youtube(_VK_I)
        self._message = "YouTube mini-player" if ok else self._message
        return ok

    def toggle_captions(self) -> bool:
        hwnd = self._activate_youtube_tab()
        if hwnd is None:
            self._message = "YouTube tab not focusable"
            return False
        ok = self._invoke_uia_named_control(
            hwnd,
            ("subtitles", "captions", "closed captions", "cc"),
        )
        if not ok:
            ok = self._send_key_to_youtube(_VK_C, hwnd=hwnd)
        if not ok:
            return False
        feedback = self._detect_captions_feedback(hwnd)
        if feedback == "unavailable":
            self._message = "No captions available for this video"
            return False
        self._message = "YouTube captions"
        return True

    def seek_backward(self) -> bool:
        ok = self._send_key_to_youtube(_VK_J)
        self._message = "YouTube -10s" if ok else self._message
        return ok

    def seek_forward(self) -> bool:
        ok = self._send_key_to_youtube(_VK_L)
        self._message = "YouTube +10s" if ok else self._message
        return ok

    def speed_down(self) -> bool:
        hwnd = self._activate_youtube_tab()
        if hwnd is None:
            self._message = "YouTube tab not focusable"
            return False
        ok = self._send_key_to_youtube(_VK_OEM_COMMA, shift=True, hwnd=hwnd)
        self._message = "YouTube slower" if ok else self._message
        return ok

    def speed_up(self) -> bool:
        hwnd = self._activate_youtube_tab()
        if hwnd is None:
            self._message = "YouTube tab not focusable"
            return False
        ok = self._send_key_to_youtube(_VK_OEM_PERIOD, shift=True, hwnd=hwnd)
        self._message = "YouTube faster" if ok else self._message
        return ok

    def like_video(self) -> bool:
        return self._invoke_named_control_action(
            ("like this video", "undo like", "like"),
            success_message="YouTube like",
        )

    def dislike_video(self) -> bool:
        return self._invoke_named_control_action(
            ("dislike this video", "undo dislike", "dislike"),
            success_message="YouTube dislike",
        )

    def share_video(self) -> bool:
        return self._invoke_named_control_action(
            ("share", "share video"),
            success_message="YouTube share",
        )

    def toggle_subscribe(self) -> bool:
        # UIA name search covers both states: a not-yet-subscribed user
        # sees "Subscribe", a subscribed user sees "Unsubscribe" or
        # "Subscribed" depending on YouTube's A/B bucket. Try the
        # toggleable names first; YouTube's accessibility tree exposes
        # this control as a Button or ToggleButton depending on layout
        # (channel page vs watch page), so the UIA hunt picks whichever
        # is present.
        return self._invoke_named_control_action(
            ("subscribe", "unsubscribe", "subscribed", "subscribe to this channel"),
            success_message="YouTube subscribe toggled",
        )

    def auto_skip_ads_tick(self) -> bool:
        """Background-friendly variant of skip_ad.

        Called on a timer when the user has enabled the auto-skip-ads
        setting. Differs from skip_ad in two ways:
          1. Silently returns False when no YouTube tab is present
             (rather than setting an error message — the auto-loop
             would otherwise spam the UI with 'no youtube tab').
          2. Skips the focus-window step if the YouTube tab is not
             foreground; auto-skip should be invisible, not steal
             focus from whatever the user is doing. Template-match
             still works on the YouTube window even when it isn't
             the foreground window because GetWindowRect + ImageGrab
             with all_screens=True captures absolute screen coords.
        """
        if not self._is_windows:
            return False
        try:
            handles = find_chrome_youtube_windows()
        except Exception:
            handles = []
        if not handles:
            return False
        now = time.time()
        if now < self._skip_click_cooldown_until:
            return False
        # Pick the first YouTube-titled Chrome window. Don't focus it.
        target_hwnd = 0
        for handle in handles:
            try:
                hwnd = int(handle.hwnd)
            except (AttributeError, TypeError, ValueError):
                continue
            if hwnd > 0:
                target_hwnd = hwnd
                break
        if target_hwnd <= 0:
            return False
        rect = wintypes.RECT()
        try:
            ctypes.windll.user32.GetWindowRect(target_hwnd, ctypes.byref(rect))
        except Exception:
            return False
        if rect.right <= rect.left or rect.bottom <= rect.top:
            return False
        screen_bgr = self._capture_window_bgr(rect, context="auto-skip")
        if screen_bgr is None:
            return False
        templates = self._load_skip_templates()
        if not templates:
            return False
        try:
            import cv2
        except ImportError:
            return False
        best_score = 0.0
        best_loc = None
        best_size = None
        for tpl in templates:
            if tpl.shape[0] >= screen_bgr.shape[0] or tpl.shape[1] >= screen_bgr.shape[1]:
                continue
            try:
                result = cv2.matchTemplate(screen_bgr, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
            except Exception:
                continue
            if max_val > best_score:
                best_score = float(max_val)
                best_loc = max_loc
                best_size = (tpl.shape[1], tpl.shape[0])
        if best_loc is None or best_size is None or best_score < _SKIP_MATCH_THRESHOLD:
            return False
        # Auto-skip must briefly focus the window so the click actually
        # hits the YouTube tab. Save the current foreground first so we
        # can restore it — otherwise an auto-skip yanks focus away from
        # whatever the user is doing and they notice immediately.
        try:
            prior_fg = int(ctypes.windll.user32.GetForegroundWindow())
        except Exception:
            prior_fg = 0
        if not self._focus_window_handle(target_hwnd, restore_if_minimized=False):
            return False
        cx = rect.left + best_loc[0] + best_size[0] // 2
        cy = rect.top + best_loc[1] + best_size[1] // 2
        clicked = self._click_at(cx, cy)
        # Restore the user's previous focus immediately after the click.
        if prior_fg > 0 and prior_fg != target_hwnd:
            try:
                ctypes.windll.user32.SetForegroundWindow(wintypes.HWND(prior_fg))
            except Exception:
                pass
        if not clicked:
            return False
        self._skip_click_cooldown_until = now + _SKIP_CLICK_COOLDOWN_SECONDS
        self._message = f"auto-skip ad ({best_score:.2f})"
        return True

    def set_caption_language(self, language: str) -> bool:
        """Switch YouTube's auto-translate target language via UIA.

        Navigation: Settings cog -> Subtitles/CC -> Auto-translate ->
        <language>. Each step is a UIA invoke on an item whose Name
        contains a known string. The menu is built lazily by YouTube so
        we sleep briefly between steps to let each submenu render.

        Returns True only when the full chain completes. Failures are
        common because:
          - The Settings cog has no Name on older YouTube layouts
            (we try several aria-labels).
          - Auto-translate only appears for videos that HAVE captions.
          - The language list is keyed by the user's locale (English
            users see 'Spanish', Spanish users see 'Espanol').

        Caller should treat False as 'this video doesn't support it'
        rather than a hard error.
        """
        if not self._is_windows:
            return False
        normalized = str(language or "").strip()
        if not normalized:
            self._message = "translate captions: missing language"
            return False
        hwnd = self._activate_youtube_tab()
        if hwnd is None:
            self._message = "translate captions: no tab"
            return False
        # Captions must be ON for the translate submenu to appear. If
        # they're off, toggle them on first — the user wouldn't have
        # asked to translate if they didn't want captions visible.
        self._invoke_uia_named_control(
            hwnd,
            ("subtitles/closed captions (c)", "subtitles", "captions", "closed captions", "cc"),
        )
        time.sleep(0.25)
        # Step 1: open the settings gear.
        if not self._invoke_uia_named_control(
            hwnd,
            ("settings", "settings menu", "more settings"),
        ):
            self._message = "translate captions: settings menu not found"
            return False
        time.sleep(0.30)
        # Step 2: open the Subtitles/CC submenu.
        if not self._invoke_uia_named_control(
            hwnd,
            ("subtitles/cc", "subtitles", "captions"),
        ):
            self._message = "translate captions: subtitles submenu not found"
            return False
        time.sleep(0.30)
        # Step 3: open Auto-translate.
        if not self._invoke_uia_named_control(
            hwnd,
            ("auto-translate", "auto translate"),
        ):
            self._message = "translate captions: auto-translate not available for this video"
            return False
        time.sleep(0.30)
        # Step 4: pick the requested language.
        if not self._invoke_uia_named_control(hwnd, (normalized.lower(),)):
            self._message = f"translate captions: '{normalized}' not in the language list"
            return False
        self._message = f"YouTube captions -> {normalized}"
        return True

    def skip_ad(self) -> bool:
        if not self._is_windows:
            self._message = "skip ad: unsupported platform"
            return False
        now = time.time()
        if now < self._skip_click_cooldown_until:
            return False
        hwnd = self._activate_youtube_tab()
        if hwnd is None:
            self._message = "skip ad: no youtube tab"
            return False
        if hwnd <= 0:
            self._message = "skip ad: invalid window handle"
            return False
        if not self._focus_window_handle(hwnd, restore_if_minimized=True):
            self._message = "skip ad: tab not focusable"
            return False
        time.sleep(0.16)

        rect = wintypes.RECT()
        try:
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
        except Exception:
            self._message = "skip ad: window rect failed"
            return False
        if rect.right <= rect.left or rect.bottom <= rect.top:
            self._message = "skip ad: window rect empty"
            return False

        screen_bgr = self._capture_window_bgr(rect, context="skip ad")
        if screen_bgr is None:
            return False

        templates = self._load_skip_templates()
        if not templates:
            self._message = "skip ad: no templates in assets/youtube_skip/"
            return False

        try:
            import cv2
        except ImportError:
            self._message = "skip ad: cv2 unavailable"
            return False

        best_score = 0.0
        best_loc = None
        best_size = None
        for tpl in templates:
            if tpl.shape[0] >= screen_bgr.shape[0] or tpl.shape[1] >= screen_bgr.shape[1]:
                continue
            try:
                result = cv2.matchTemplate(screen_bgr, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
            except Exception:
                continue
            if max_val > best_score:
                best_score = float(max_val)
                best_loc = max_loc
                best_size = (tpl.shape[1], tpl.shape[0])

        if best_loc is None or best_size is None or best_score < _SKIP_MATCH_THRESHOLD:
            self._message = f"skip ad: no match ({best_score:.2f})"
            return False

        cx = rect.left + best_loc[0] + best_size[0] // 2
        cy = rect.top + best_loc[1] + best_size[1] // 2
        if not self._click_at(cx, cy):
            return False
        self._skip_click_cooldown_until = now + _SKIP_CLICK_COOLDOWN_SECONDS
        self._message = f"skip ad ({best_score:.2f})"
        return True

    def _capture_window_bgr(self, rect, *, context: str = "capture"):
        try:
            from PIL import ImageGrab
            import numpy as np
            import cv2
        except ImportError:
            self._message = f"{context}: capture deps missing"
            return None
        try:
            img = ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom), all_screens=True)
        except Exception as exc:
            self._message = f"{context}: capture failed {type(exc).__name__}"
            return None
        try:
            arr = np.array(img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            self._message = f"{context}: capture decode failed {type(exc).__name__}"
            return None

    def _send_ctrl_key(self, hwnd: int, vk: int) -> bool:
        """Send Ctrl+<vk> to the focused Chrome window. Used by the
        auto-play path to issue Ctrl+9 (jump to last tab) when the
        freshly-opened YouTube tab landed in the background."""
        if not self._is_windows or hwnd <= 0:
            return False
        if not self._focus_window_handle(hwnd, restore_if_minimized=True):
            return False
        try:
            user32 = ctypes.windll.user32
            user32.keybd_event(wintypes.BYTE(_VK_CONTROL), 0, 0, 0)
            user32.keybd_event(wintypes.BYTE(vk), 0, 0, 0)
            user32.keybd_event(wintypes.BYTE(vk), 0, _KEYEVENTF_KEYUP, 0)
            user32.keybd_event(wintypes.BYTE(_VK_CONTROL), 0, _KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    def _log_autoplay(self, stage: str, detail: str = "") -> None:
        """Single sink for YouTube auto-play diagnostics. Writes to
        stderr (visible in the same console log the user is already
        tailing) AND latches `self._message` so the in-app action
        history shows the same string. Tagged so a quick log filter
        on `[yt-autoplay]` shows the full lifecycle of one command."""
        suffix = f": {detail}" if detail else ""
        line = f"[yt-autoplay] {stage}{suffix}"
        try:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        self._message = line.lower()

    def play_first_search_result(self, query: str, *, search_url_was_just_opened: bool = True) -> bool:
        """Auto-click the first video card on a YouTube search results
        page. Used by the "play X on YouTube" voice command after
        `chrome_controller.search_youtube` opens the results page.

        Strategy:
          1. Poll for the Chrome window whose title contains the
             spoken query (YouTube sets the page title to
             "<query> - YouTube" once the results render).
          2. Bring it to the foreground.
          3. Run a UIA script that walks the page's accessibility
             tree, picks the first hyperlink whose name overlaps
             with the query keywords, and outputs its bounding
             rectangle.
          4. Synthesize a real mouse click at the center of that
             rectangle. We click via SetCursorPos+mouse_event
             rather than `InvokePattern.Invoke()` because YouTube's
             SPA video cards have JavaScript click handlers that
             respond to mouse events but ignore UIA Invoke (the
             previous version found the link successfully but the
             page didn't navigate).

        On any failure path we leave the user on the search-results
        page — the user-facing degradation is identical to the
        pre-auto-play behaviour, so this is purely additive."""
        if not self._is_windows:
            return False
        normalized = " ".join((query or "").split()).strip()
        if not normalized:
            self._log_autoplay("missing query")
            return False
        self._log_autoplay("start", normalized)
        if search_url_was_just_opened:
            time.sleep(0.55)
        target_hwnd = self._wait_for_search_results_window(normalized, timeout_seconds=10.0)
        if target_hwnd is None:
            self._log_autoplay("search window not found in time")
            return False
        self._log_autoplay("found window", f"hwnd={target_hwnd}")
        self._remember_youtube_window(target_hwnd)
        self._log_autoplay("focusing window")
        if not self._focus_window_handle(target_hwnd, restore_if_minimized=True):
            self._log_autoplay("search window not focusable")
            return False
        try:
            current_title = self._window_title(target_hwnd).lower()
        except Exception:
            current_title = ""
        if "youtube" not in current_title:
            self._log_autoplay("activating last tab", f"current_title={current_title!r}")
            self._send_ctrl_key(target_hwnd, _VK_9)
            time.sleep(0.40)
        else:
            self._log_autoplay("youtube already active", f"title={current_title!r}")
        # Bumped from 0.45 -> 1.20 because Chrome activates page
        # accessibility lazily and the first UIA query against an
        # un-activated tree returns empty.
        self._log_autoplay("waiting for page hydration", "1.20s")
        time.sleep(1.20)
        keywords = self._search_query_keywords(normalized)
        self._log_autoplay("UIA query", f"keywords={keywords}")
        rect = self._find_first_search_result_rect(target_hwnd, keywords)
        if rect is None:
            # _find_first_search_result_rect already wrote a specific
            # diagnostic via _log_autoplay (see the implementation).
            return False
        click_x = int(round((rect[0] + rect[2]) / 2.0))
        click_y = int(round((rect[1] + rect[3]) / 2.0))
        try:
            pre_title = self._window_title(target_hwnd) or ""
        except Exception:
            pre_title = ""
        for attempt in range(2):
            # Re-establish foreground right before the click — UIA
            # query took ~seconds, focus may have drifted, and a
            # background click is the difference between Chrome
            # routing it to the page vs swallowing it.
            self._focus_window_handle(target_hwnd, restore_if_minimized=True)
            time.sleep(0.05)
            self._log_autoplay("clicking", f"({click_x},{click_y}) rect={rect} attempt={attempt+1}")
            if not self._click_at(click_x, click_y):
                self._log_autoplay("click failed", f"attempt={attempt+1}")
                continue
            # Poll the window title for navigation. A successful
            # click navigates the search-results page → video
            # watch page, which changes the Chrome window title
            # from "<query> - YouTube …" to "<video title> - YouTube …".
            deadline = time.monotonic() + 2.5
            while time.monotonic() < deadline:
                time.sleep(0.20)
                try:
                    new_title = self._window_title(target_hwnd) or ""
                except Exception:
                    new_title = ""
                if new_title and new_title != pre_title:
                    self._log_autoplay("done", f"{normalized} (navigated)")
                    return True
            self._log_autoplay("no navigation after click", f"attempt={attempt+1} title={pre_title!r}")
        self._log_autoplay("click did not navigate after retry", normalized)
        return False

    def _wait_for_search_results_window(
        self,
        query: str,
        *,
        timeout_seconds: float,
    ) -> int | None:
        """Poll for a Chrome window holding the freshly-opened
        YouTube search-results tab. Three-pass strategy because
        Chrome may put the URL in any of:
          1. A new window (title becomes "<query> - YouTube").
          2. A new tab in an existing window where YouTube was
             ALREADY the active tab (title also becomes "<query> -
             YouTube").
          3. A new tab in an existing window where some OTHER tab
             is active — title stays as the previously-active tab,
             so neither "youtube" nor the query appear in it.

        Pass 1: any Chrome window whose title contains "youtube"
                (fast path — covers cases 1 and 2).
        Pass 2: any Chrome window whose title contains a query
                keyword (catches case 1 if Chrome decided to use a
                custom title format).
        Pass 3: ANY visible Chrome window — if the user only has
                one Chrome window open, that's where the new tab
                landed regardless of title. We then need to
                activate the YouTube tab inside it (handled
                separately by Ctrl+9 / Ctrl+Tab cycling)."""
        deadline = time.monotonic() + max(0.5, float(timeout_seconds))
        needle = " ".join(query.lower().split())
        keywords = [w for w in needle.split() if len(w) >= 4]

        def _hwnd_of(item) -> int:
            """find_chrome_youtube_windows() returns WindowInfo
            objects with a .hwnd attribute, while
            _chrome_window_handles() returns plain ints. Normalize
            either to an int."""
            value = getattr(item, "hwnd", item)
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        while time.monotonic() < deadline:
            # ---- Pass 1: YouTube-titled windows (fast path) ----
            try:
                yt_windows = list(find_chrome_youtube_windows())
            except Exception:
                yt_windows = []
            yt_hwnds = [h for h in (_hwnd_of(w) for w in yt_windows) if h > 0]
            for hwnd in yt_hwnds:
                try:
                    title = self._window_title(hwnd).lower()
                except Exception:
                    title = ""
                if needle and needle in title:
                    return hwnd
            if yt_hwnds:
                # YouTube tab exists somewhere; pick the first one
                # we see and let the rest of the pipeline activate it.
                self._log_autoplay("window match", "youtube-titled fallback")
                return yt_hwnds[0]
            # ---- Pass 2: any Chrome window whose title has a query keyword ----
            try:
                chrome_hwnds = list(self._chrome_window_handles())
            except Exception:
                chrome_hwnds = []
            chrome_hwnds = [h for h in (_hwnd_of(h) for h in chrome_hwnds) if h > 0]
            for hwnd in chrome_hwnds:
                try:
                    title = self._window_title(hwnd).lower()
                except Exception:
                    title = ""
                if keywords and any(word in title for word in keywords):
                    self._log_autoplay("window match", f"keyword in title={title!r}")
                    return hwnd
            # ---- Pass 3: any Chrome window at all (last resort) ----
            # Only after we've waited long enough for Chrome to
            # finish opening the URL — the very first ticks might
            # find a stale Chrome window before the new tab landed.
            if chrome_hwnds and time.monotonic() > deadline - max(0.5, timeout_seconds * 0.5):
                self._log_autoplay("window match", f"any-chrome fallback hwnd={chrome_hwnds[0]}")
                return chrome_hwnds[0]
            time.sleep(0.18)
        return None

    @staticmethod
    def _search_query_keywords(query: str) -> list[str]:
        """Pick the meaningful words from `query` for fuzzy-matching
        against video link names. Drops short stop words so common
        connectors like 'by', 'the', 'on' don't make every link
        match."""
        stop = {"a", "an", "the", "to", "of", "and", "or", "in", "on", "at", "for", "by", "with", "is", "it"}
        words = [word.lower() for word in re.split(r"[^A-Za-z0-9]+", query) if word]
        keep = [word for word in words if word not in stop and len(word) >= 3]
        return keep or words  # if everything was stop-word, keep originals

    def _find_first_search_result_rect(
        self, hwnd: int, keywords: list[str]
    ) -> tuple[int, int, int, int] | None:
        """Run a PowerShell UIA script that walks the YouTube search-
        results page tree and returns the bounding rectangle of the
        first `Hyperlink` whose name overlaps with the query
        keywords (filtered against an extensive nav-chrome blocklist).

        Returns (left, top, right, bottom) in screen coords or None.

        The previous version called `InvokePattern.Invoke()` directly
        from PowerShell. UIA accepted the call but YouTube's SPA
        components didn't trigger their JS handlers, so the page
        never navigated — the user reported "it searched correctly
        but didn't play a video". Returning the rect and letting the
        Python side click it via SetCursorPos+mouse_event triggers
        the real mouse-event pipeline, which YouTube's handlers DO
        respond to."""
        if hwnd <= 0 or not keywords:
            return None
        try:
            payload = json.dumps([str(k).lower() for k in keywords])
        except Exception:
            return None
        script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$hwnd = [IntPtr]({int(hwnd)})
$keywords = ConvertFrom-Json @'
{payload}
'@
$root = [System.Windows.Automation.AutomationElement]::FromHandle($hwnd)
if ($null -eq $root) {{
    [Console]::Out.Write('NOT_FOUND_ROOT')
    exit 0
}}

# CHROME ACCESSIBILITY ACTIVATION
# Chrome enables its accessibility tree lazily — the first FindAll
# against an un-activated Chrome window returns 0 elements even
# when the page is fully loaded. Walking the tree with TreeWalker
# and reading per-node properties is what screen readers do and is
# what reliably triggers Chrome to populate. Up to 400 nodes in
# BFS order to keep this bounded if the tree is huge.
$walker = [System.Windows.Automation.TreeWalker]::RawViewWalker
$queue = New-Object System.Collections.Generic.Queue[object]
$queue.Enqueue($root)
$processed = 0
while ($queue.Count -gt 0 -and $processed -lt 400) {{
    $current = $queue.Dequeue()
    $processed++
    try {{
        $null = $current.Current.Name
        $null = $current.Current.ControlType
    }} catch {{}}
    try {{
        $child = $walker.GetFirstChild($current)
    }} catch {{
        $child = $null
    }}
    while ($child -ne $null -and $queue.Count -lt 600) {{
        $queue.Enqueue($child)
        try {{
            $child = $walker.GetNextSibling($child)
        }} catch {{
            $child = $null
        }}
    }}
}}
Start-Sleep -Milliseconds 350

$linkType = [System.Windows.Automation.ControlType]::Hyperlink
$cond = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    $linkType
)
$links = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
if ($links.Count -eq 0) {{
    # Last-resort: try ANY element type. YouTube's video card
    # outer wrapper sometimes registers as Group / Image / Button
    # rather than Hyperlink, depending on Chrome version + page
    # state. Filter by name only.
    $links = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, [System.Windows.Automation.Condition]::TrueCondition)
}}
if ($links.Count -eq 0) {{
    [Console]::Out.Write('NO_LINKS')
    exit 0
}}
for ($i = 0; $i -lt $links.Count; $i++) {{
    $link = $links.Item($i)
    try {{
        $name = [string]$link.Current.Name
    }} catch {{
        $name = ''
    }}
    if ([string]::IsNullOrWhiteSpace($name)) {{ continue }}
    $nameLower = $name.ToLowerInvariant()
    # Reject nav/chrome links present on every YouTube page.
    if ($nameLower -match '^(home|shorts|subscriptions|library|history|trending|gaming|music|news|sports|learning|fashion|sign in|premium|youtube studio|your channel|search|skip navigation|skip to main content|guide|settings|create|notifications|filters|help|send feedback|about|press|copyright|contact us|creators|advertise|developers|terms|privacy|policy & safety|how youtube works|test new features|youtube)$') {{
        continue
    }}
    if ($nameLower.Length -lt 8) {{ continue }}
    $matched = $false
    foreach ($kw in $keywords) {{
        if (-not [string]::IsNullOrWhiteSpace([string]$kw) -and $nameLower.Contains(([string]$kw).ToLowerInvariant())) {{
            $matched = $true
            break
        }}
    }}
    if (-not $matched) {{ continue }}
    try {{
        $rect = $link.Current.BoundingRectangle
    }} catch {{
        continue
    }}
    # Min size 100x60 = real video card. The TrueCondition fallback
    # otherwise matches small tooltip / hidden overlay elements
    # whose names happen to share keywords with the query.
    if ($rect.Width -lt 100 -or $rect.Height -lt 60) {{ continue }}
    # Visible-on-screen check: element must have a non-degenerate rect.
    if ($rect.Width -le 0 -or $rect.Height -le 0) {{ continue }}
    [Console]::Out.Write(('RECT ' + [int]$rect.Left + ' ' + [int]$rect.Top + ' ' + [int]$rect.Right + ' ' + [int]$rect.Bottom + ' ' + $name))
    exit 0
}}
[Console]::Out.Write(('NOT_FOUND_LINKS=' + $links.Count))
"""
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    script,
                ],
                capture_output=True,
                text=True,
                # TreeWalker activation walk + per-node property
                # reads is slower than a bare FindAll; 12 s cap so
                # a slow Chrome accessibility activation doesn't
                # time out. Worker is on a daemon thread so the
                # voice pipeline isn't blocked.
                timeout=12.0,
                check=False,
            )
        except Exception as exc:
            self._log_autoplay("UIA exec failed", type(exc).__name__)
            return None
        if completed.returncode != 0:
            stderr_tail = (completed.stderr or "").strip().splitlines()[-1:] if completed.stderr else []
            self._log_autoplay(
                "UIA returncode",
                f"rc={completed.returncode} stderr={stderr_tail!r}",
            )
            return None
        outcome = str(completed.stdout or "").strip()
        if not outcome.startswith("RECT "):
            short = outcome.split()[0] if outcome else "EMPTY_OUTPUT"
            self._log_autoplay("UIA returned", short)
            return None
        try:
            parts = outcome.split(maxsplit=5)
            left = int(parts[1])
            top = int(parts[2])
            right = int(parts[3])
            bottom = int(parts[4])
            link_name = parts[5] if len(parts) > 5 else ""
        except (IndexError, ValueError):
            self._log_autoplay("bad UIA output", outcome[:120])
            return None
        self._log_autoplay("UIA matched link", f"name={link_name!r}")
        return (left, top, right, bottom)

    def _invoke_named_control_action(self, name_patterns: tuple[str, ...], *, success_message: str) -> bool:
        hwnd = self._activate_youtube_tab()
        if hwnd is None:
            self._message = "YouTube tab not focusable"
            return False
        if self._invoke_uia_named_control(hwnd, name_patterns):
            self._message = str(success_message)
            return True
        self._message = f"{success_message.lower()} unavailable"
        return False

    def _invoke_uia_named_control(self, hwnd: int, name_patterns: tuple[str, ...]) -> bool:
        if not self._is_windows or hwnd <= 0:
            return False
        needles = [str(pattern or "").strip().lower() for pattern in name_patterns if str(pattern or "").strip()]
        if not needles:
            return False
        if not self._focus_window_handle(hwnd, restore_if_minimized=True):
            return False
        try:
            payload = json.dumps(needles)
        except Exception:
            return False
        script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$hwnd = [IntPtr]({int(hwnd)})
$needles = ConvertFrom-Json @'
{payload}
'@
$root = [System.Windows.Automation.AutomationElement]::FromHandle($hwnd)
if ($null -eq $root) {{
    [Console]::Out.Write('NOT_FOUND')
    exit 0
}}
$elements = $root.FindAll(
    [System.Windows.Automation.TreeScope]::Descendants,
    [System.Windows.Automation.Condition]::TrueCondition
)
for ($i = 0; $i -lt $elements.Count; $i++) {{
    $element = $elements.Item($i)
    try {{
        $name = [string]$element.Current.Name
    }} catch {{
        $name = ''
    }}
    if ([string]::IsNullOrWhiteSpace($name)) {{
        continue
    }}
    $nameLower = $name.ToLowerInvariant()
    $matched = $false
    foreach ($needle in $needles) {{
        if (-not [string]::IsNullOrWhiteSpace([string]$needle) -and $nameLower.Contains(([string]$needle).ToLowerInvariant())) {{
            $matched = $true
            break
        }}
    }}
    if (-not $matched) {{
        continue
    }}
    try {{
        $invokePattern = $element.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
        if ($invokePattern -is [System.Windows.Automation.InvokePattern]) {{
            $invokePattern.Invoke()
            [Console]::Out.Write('INVOKED')
            exit 0
        }}
    }} catch {{}}
    try {{
        $togglePattern = $element.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
        if ($togglePattern -is [System.Windows.Automation.TogglePattern]) {{
            $togglePattern.Toggle()
            [Console]::Out.Write('TOGGLED')
            exit 0
        }}
    }} catch {{}}
    try {{
        $selectionPattern = $element.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern)
        if ($selectionPattern -is [System.Windows.Automation.SelectionItemPattern]) {{
            $selectionPattern.Select()
            [Console]::Out.Write('SELECTED')
            exit 0
        }}
    }} catch {{}}
    try {{
        $legacyPattern = $element.GetCurrentPattern([System.Windows.Automation.LegacyIAccessiblePattern]::Pattern)
        if ($legacyPattern -is [System.Windows.Automation.LegacyIAccessiblePattern]) {{
            $legacyPattern.DoDefaultAction()
            [Console]::Out.Write('DEFAULT')
            exit 0
        }}
    }} catch {{}}
}}
[Console]::Out.Write('NOT_FOUND')
"""
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    script,
                ],
                capture_output=True,
                text=True,
                timeout=_UIA_ACTION_TIMEOUT_SECONDS,
                check=False,
            )
        except Exception:
            return False
        if completed.returncode != 0:
            return False
        outcome = str(completed.stdout or "").strip().upper()
        return outcome in {"INVOKED", "TOGGLED", "SELECTED", "DEFAULT"}

    def _execute_youtube_script_action(self, hwnd: int, script: str) -> str | None:
        if not self._is_windows or hwnd <= 0 or self._text_input is None:
            return None
        payload = str(script or "").strip()
        if not payload:
            return None
        if not self._focus_window_handle(hwnd, restore_if_minimized=True):
            return None
        self._text_input._target_hwnd = int(hwnd)
        self._text_input._last_external_hwnd = int(hwnd)
        try:
            if not self._text_input._send_shortcut(_VK_CONTROL, _VK_KEY_L):
                return None
            time.sleep(0.04)
            if not self._text_input.insert_text(payload, prefer_paste=False):
                return None
            time.sleep(0.04)
            if not self._text_input._send_shortcut(_VK_RETURN):
                return None
        except Exception:
            return None
        return self._wait_for_script_result(hwnd)

    def _wait_for_script_result(self, hwnd: int) -> str | None:
        deadline = time.time() + _YOUTUBE_SCRIPT_TITLE_TIMEOUT_SECONDS
        while time.time() < deadline:
            title = self._window_title(hwnd)
            match = _YOUTUBE_SCRIPT_RESULT_PATTERN.search(title)
            if match is not None:
                return str(match.group(1) or "").upper()
            time.sleep(0.08)
        return None

    def _captions_script(self) -> str:
        body = (
            "const p=document.getElementById('movie_player');"
            "const tracks=p&&typeof p.getOption==='function'?p.getOption('captions','tracklist'):null;"
            "const b=document.querySelector('button.ytp-subtitles-button');"
            "if(!b||!tracks||!tracks.length){return 'NO_CAPTIONS';}"
            "b.click();"
            "return 'CAPTIONS';"
        )
        return self._wrap_youtube_script(body)

    def _theater_script(self) -> str:
        body = (
            "const b=document.querySelector('button.ytp-size-button');"
            "if(!b){return 'THEATER_FAILED';}"
            "b.click();"
            "return 'THEATER';"
        )
        return self._wrap_youtube_script(body)

    def _like_script(self) -> str:
        return self._button_action_script(
            result_name="LIKE",
            label_patterns=("like this video", "undo like", "like"),
        )

    def _dislike_script(self) -> str:
        return self._button_action_script(
            result_name="DISLIKE",
            label_patterns=("dislike this video", "undo dislike", "dislike"),
        )

    def _share_script(self) -> str:
        return self._button_action_script(
            result_name="SHARE",
            label_patterns=("share",),
        )

    def _button_action_script(self, *, result_name: str, label_patterns: tuple[str, ...]) -> str:
        filters = ",".join(f"'{pattern.lower()}'" for pattern in label_patterns if pattern)
        body = (
            "const buttons=Array.from(document.querySelectorAll('button[aria-label],button[title]'));"
            "const match=buttons.find((button)=>{"
            "const label=((button.getAttribute('aria-label')||button.getAttribute('title')||button.innerText||'')+'').toLowerCase();"
            f"return [{filters}].some((needle)=>label.includes(needle));"
            "});"
            f"if(!match){{return '{result_name}_FAILED';}}"
            "match.click();"
            f"return '{result_name}';"
        )
        return self._wrap_youtube_script(body)

    def _wrap_youtube_script(self, body: str) -> str:
        action_body = str(body or "").strip()
        if not action_body:
            return ""
        return (
            "javascript:(()=>{try{"
            "const __hgrTitle=document.title;"
            f"const __hgrMark=(value)=>{{document.title='{_YOUTUBE_SCRIPT_RESULT_PREFIX}'+value+'__'+__hgrTitle;"
            "setTimeout(()=>{document.title=__hgrTitle;},1600);};"
            f"const __hgrResult=(()=>{{{action_body}}})();"
            "__hgrMark(__hgrResult||'FAILED');"
            "}catch(_error){"
            "const __hgrTitle=document.title;"
            f"document.title='{_YOUTUBE_SCRIPT_RESULT_PREFIX}FAILED__'+__hgrTitle;"
            "setTimeout(()=>{document.title=__hgrTitle;},1600);"
            "}})()"
        )

    def _detect_captions_feedback(self, hwnd: int) -> str | None:
        if not self._is_windows or hwnd <= 0:
            return None
        time.sleep(_CAPTIONS_FEEDBACK_SETTLE_SECONDS)
        rect = wintypes.RECT()
        try:
            ctypes.windll.user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
        except Exception:
            return None
        if rect.right <= rect.left or rect.bottom <= rect.top:
            return None
        screen_bgr = self._capture_window_bgr(rect, context="captions")
        if screen_bgr is None:
            return None
        text = self._ocr_captions_feedback_text(screen_bgr)
        if not text:
            return None
        normalized = " ".join(str(text).lower().split())
        if "unavailable" in normalized and any(token in normalized for token in ("caption", "captions", "subtitle", "subtitles", "cc")):
            return "unavailable"
        return None

    def _ocr_captions_feedback_text(self, screen_bgr) -> str:
        try:
            import cv2
        except ImportError:
            return ""
        try:
            height, width = screen_bgr.shape[:2]
        except Exception:
            return ""
        if height <= 0 or width <= 0:
            return ""
        left = max(0, int(width * 0.18))
        right = min(width, int(width * 0.82))
        top = max(0, int(height * 0.54))
        bottom = min(height, int(height * 0.86))
        if right - left < 24 or bottom - top < 24:
            return ""
        crop = screen_bgr[top:bottom, left:right]
        if crop.size == 0:
            return ""
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = 255 - binary
        except Exception:
            return ""
        ok, encoded = cv2.imencode(".png", processed)
        if not ok:
            return ""
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="hgr_youtube_captions_", suffix=".png", delete=False) as temp_file:
                temp_file.write(encoded.tobytes())
                temp_path = Path(temp_file.name)
            return self._ocr_image_path(temp_path)
        except Exception:
            return ""
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _ocr_image_path(self, image_path: Path) -> str:
        if not self._is_windows:
            return ""
        script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType = WindowsRuntime]
function Await($asyncOp, $resultType) {
    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 } |
        Select-Object -First 1
    if ($null -eq $method) {
        return $null
    }
    $task = $method.MakeGenericMethod($resultType).Invoke($null, @($asyncOp))
    $task.Wait(-1)
    return $task.Result
}
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($args[0])) ([Windows.Storage.StorageFile])
if ($null -eq $file) { return }
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
if ($null -eq $stream) { return }
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
if ($null -eq $decoder) { return }
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
if ($null -eq $bitmap) { return }
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) { return }
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
if ($null -ne $result -and $null -ne $result.Text) {
    [Console]::Out.Write($result.Text)
}
"""
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    script,
                    str(image_path),
                ],
                capture_output=True,
                text=True,
                timeout=_CAPTIONS_OCR_TIMEOUT_SECONDS,
                check=False,
            )
        except Exception:
            return ""
        if completed.returncode != 0:
            return ""
        return str(completed.stdout or "").strip()

    def _load_skip_templates(self) -> list:
        try:
            import cv2
        except ImportError:
            return []
        templates: list = []
        seen: set[str] = set()
        for directory in self._skip_template_dirs():
            if not directory.exists() or not directory.is_dir():
                continue
            for png in sorted(directory.glob("*.png")):
                key = str(png.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                try:
                    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
                except Exception:
                    img = None
                if img is not None and img.size > 0:
                    templates.append(img)
        return templates

    def _skip_template_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            dirs.append(Path(meipass) / "assets" / _SKIP_TEMPLATE_DIRNAME)
        here = Path(__file__).resolve()
        for candidate in here.parents:
            assets = candidate / "assets" / _SKIP_TEMPLATE_DIRNAME
            dirs.append(assets)
            if (candidate / "assets").exists():
                break
        dirs.append(Path.cwd() / "assets" / _SKIP_TEMPLATE_DIRNAME)
        return dirs

    def _click_at(self, x: int, y: int) -> bool:
        """Synthesize a left-button click at screen coords (x, y).

        Uses SendInput with MOUSEEVENTF_VIRTUALDESK + MOUSEEVENTF_ABSOLUTE
        so multi-monitor setups with negative-X coordinates (e.g. a
        secondary display to the left of the primary) click the
        actual on-screen pixel rather than the legacy mouse_event
        path which was silently dropping clicks on negative-X.

        The move + down + up are submitted as a single SendInput batch
        so Windows treats them as one synchronous input sequence —
        prevents Chrome's hover/focus handler from firing between
        the move and the click."""
        try:
            self._log_autoplay("_click_at", f"entered with x={x} y={y}")
            user32 = ctypes.windll.user32

            class _MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", wintypes.LONG),
                    ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.c_void_p),
                ]

            class _INPUT_UNION(ctypes.Union):
                _fields_ = [("mi", _MOUSEINPUT)]

            class _INPUT(ctypes.Structure):
                _anonymous_ = ("u",)
                _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]

            # Explicit ctypes signatures — without these, the
            # default `c_int` argtypes truncate the pointer to 32
            # bits on x64 Windows, SendInput receives a garbage
            # buffer address, and silently returns 0.
            user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
            user32.SendInput.restype = wintypes.UINT
            user32.GetSystemMetrics.argtypes = [ctypes.c_int]
            user32.GetSystemMetrics.restype = ctypes.c_int

            virt_left   = user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)
            virt_top    = user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)
            virt_width  = user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)
            virt_height = user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)
            self._log_autoplay(
                "virt-desk metrics",
                f"left={virt_left} top={virt_top} w={virt_width} h={virt_height} "
                f"sizeof_INPUT={ctypes.sizeof(_INPUT)}",
            )
            if virt_width <= 0 or virt_height <= 0:
                # Fall back to legacy path if metrics look wrong.
                self._log_autoplay("falling back to legacy path", "")
                user32.SetCursorPos(int(x), int(y))
                time.sleep(0.04)
                user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.02)
                user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                return True

            # Normalize (x, y) into the virtual-desktop 0..65535 space.
            abs_x = int(round(((int(x) - virt_left) * 65535.0) / max(1, virt_width  - 1)))
            abs_y = int(round(((int(y) - virt_top)  * 65535.0) / max(1, virt_height - 1)))
            move_flags = _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE | _MOUSEEVENTF_VIRTUALDESK

            inputs = (_INPUT * 3)()
            for entry in inputs:
                entry.type = _INPUT_MOUSE
                entry.mi.time = 0
                entry.mi.dwExtraInfo = None
                entry.mi.mouseData = 0
            inputs[0].mi.dx = abs_x; inputs[0].mi.dy = abs_y; inputs[0].mi.dwFlags = move_flags
            inputs[1].mi.dx = 0;     inputs[1].mi.dy = 0;     inputs[1].mi.dwFlags = _MOUSEEVENTF_LEFTDOWN
            inputs[2].mi.dx = 0;     inputs[2].mi.dy = 0;     inputs[2].mi.dwFlags = _MOUSEEVENTF_LEFTUP

            self._log_autoplay("SendInput call", f"abs=({abs_x},{abs_y})")
            # Pass byref to the first element — type is LP__INPUT,
            # which matches the argtypes declaration. byref(inputs)
            # would be LP__INPUT_Array_3 and ctypes' type checker
            # rejects it even though the underlying address is
            # identical and SendInput reads count*sizeof(INPUT)
            # bytes from the start.
            sent = user32.SendInput(3, ctypes.byref(inputs[0]), ctypes.sizeof(_INPUT))
            self._log_autoplay("SendInput returned", f"sent={sent}/3")
            if sent != 3:
                last_err = ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else 0
                self._log_autoplay(
                    "SendInput partial",
                    f"sent={sent}/3 errno={last_err} virt=({virt_left},{virt_top},{virt_width}x{virt_height}) abs=({abs_x},{abs_y})",
                )
                return False
            return True
        except Exception as exc:
            import traceback
            self._log_autoplay("_click_at exception", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._message = f"skip ad: click failed {type(exc).__name__}"
            return False

    def get_volume(self) -> float | None:
        if self._volume_controller is None:
            return None
        target, level = self._volume_controller.get_app_audio_info(["chrome"])
        return level if target is not None else None

    def set_volume(self, scalar: float) -> bool:
        if self._volume_controller is None:
            return False
        return self._volume_controller.set_app_audio_level(["chrome"], scalar)

# Author: Konstantin Markov

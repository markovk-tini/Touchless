from __future__ import annotations

import ctypes
import platform
import re
import subprocess
import time
from ctypes import wintypes
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse
from urllib.parse import quote_plus

import psutil

from ..utils.subprocess_utils import launch_external


SW_RESTORE = 9
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_MENU = 0x12
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_R = 0x52
VK_T = 0x54
VK_N = 0x4E
VK_D = 0x44
VK_H = 0x48
VK_J = 0x4A
VK_P = 0x50

KNOWN_WEB_TARGETS = {
    "chatgpt": "https://chatgpt.com",
    "gmail": "https://mail.google.com",
    "google docs": "https://docs.google.com",
    "indeed": "https://www.indeed.com",
    "outlook": "https://outlook.office.com",
    "youtube": "https://www.youtube.com",
}


class ChromeController:
    def __init__(self, *, executable_paths: tuple[Path, ...] | None = None) -> None:
        self._available = platform.system() == "Windows"
        self._message = "chrome idle"
        self._executable_paths = executable_paths or self._default_executable_paths()
        self._handles_cache: list[int] = []
        self._handles_cache_until = 0.0

    @property
    def available(self) -> bool:
        return self._available

    @property
    def message(self) -> str:
        return self._message

    def voice_request_targets_chrome(self, spoken_text: str, *, assume_chrome: bool = False) -> bool:
        text = " ".join((spoken_text or "").strip().split()).lower()
        if not text:
            return False
        if assume_chrome:
            return True
        return "chrome" in text or "google" in text

    def is_running(self) -> bool:
        if not self._available:
            return False
        try:
            for proc in psutil.process_iter(["name"]):
                name = (proc.info.get("name") or "").lower()
                if name == "chrome.exe" or "chrome" in name:
                    return True
        except Exception:
            return False
        return False

    def is_window_active(self) -> bool:
        handles = self._chrome_window_handles()
        if not handles:
            return False
        return self._foreground_window_handle() in handles

    def is_window_open(self) -> bool:
        return bool(self._chrome_window_handles())

    def focus_or_open_window(self) -> bool:
        if not self._available:
            self._message = "chrome unavailable on this platform"
            return False
        if self.is_window_active():
            self._message = "chrome already focused"
            return True

        handles = self._chrome_window_handles()
        launched_fresh = False
        if not handles:
            if not self.launch_chrome():
                return False
            launched_fresh = True
            handles = self._wait_for_window_handles()
        # If we just successfully launched Chrome, treat the action as
        # success even if window-handle detection times out — Chrome IS
        # opening from the user's perspective. Otherwise the Live API
        # router (and the voice overlay) would falsely report failure.
        if not handles:
            if launched_fresh:
                self._message = "launched chrome (window not yet visible)"
                return True
            self._message = "chrome window not found"
            return False

        if self._activate_window_handle(handles[0]):
            time.sleep(0.05)
            self._message = "chrome focused"
            return True
        if launched_fresh:
            # Same logic as above: launch happened, focus failed — not
            # the user's problem.
            self._message = "launched chrome"
            return True
        self._message = "chrome focus failed"
        return False

    def launch_chrome(self) -> bool:
        if self._launch_target():
            self._message = "launching chrome"
            return True
        self._message = "chrome launch path not found"
        return False

    def navigate_back(self) -> bool:
        if not self.focus_or_open_window():
            return False
        if self._send_shortcut(VK_MENU, VK_LEFT):
            self._message = "chrome back"
            return True
        self._message = "chrome back failed"
        return False

    def navigate_forward(self) -> bool:
        if not self.focus_or_open_window():
            return False
        if self._send_shortcut(VK_MENU, VK_RIGHT):
            self._message = "chrome forward"
            return True
        self._message = "chrome forward failed"
        return False

    def refresh_page(self) -> bool:
        if not self.focus_or_open_window():
            return False
        if self._send_shortcut(VK_CONTROL, VK_R):
            self._message = "chrome refresh"
            return True
        self._message = "chrome refresh failed"
        return False

    def new_tab(self) -> bool:
        if not self.focus_or_open_window():
            return False
        if self._send_shortcut(VK_CONTROL, VK_T):
            self._message = "chrome new tab"
            return True
        self._message = "chrome new tab failed"
        return False

    def new_incognito_tab(self) -> bool:
        if not self.focus_or_open_window():
            return False
        if self._send_shortcut(VK_CONTROL, VK_SHIFT, VK_N):
            self._message = "chrome new incognito tab"
            return True
        self._message = "chrome new incognito tab failed"
        return False

    def bookmark_current_tab(self) -> bool:
        if not self.focus_or_open_window():
            return False
        if self._send_shortcut(VK_CONTROL, VK_D):
            self._message = "chrome bookmark current tab"
            return True
        self._message = "chrome bookmark failed"
        return False

    def open_history(self) -> bool:
        if not self.focus_or_open_window():
            return False
        if self._send_shortcut(VK_CONTROL, VK_H):
            self._message = "chrome history"
            return True
        self._message = "chrome history failed"
        return False

    def open_downloads(self) -> bool:
        if not self.focus_or_open_window():
            return False
        if self._send_shortcut(VK_CONTROL, VK_J):
            self._message = "chrome downloads"
            return True
        self._message = "chrome downloads failed"
        return False

    def open_bookmarks_manager(self) -> bool:
        if not self._available:
            self._message = "chrome unavailable on this platform"
            return False
        return self.open_url("chrome://bookmarks/", display_name="chrome bookmarks")

    def print_page(self) -> bool:
        if not self.focus_or_open_window():
            return False
        if self._send_shortcut(VK_CONTROL, VK_P):
            self._message = "chrome print page"
            return True
        self._message = "chrome print failed"
        return False

    def reopen_closed_tab(self) -> bool:
        if not self.focus_or_open_window():
            return False
        if self._send_shortcut(VK_CONTROL, VK_SHIFT, VK_T):
            self._message = "chrome reopen closed tab"
            return True
        self._message = "chrome reopen closed tab failed"
        return False

    def parse_voice_search_request(self, spoken_text: str, *, assume_chrome: bool = False) -> str | None:
        text = " ".join((spoken_text or "").strip().split())
        if not text:
            return None
        lowered = text.lower()
        if not self.voice_request_targets_chrome(text, assume_chrome=assume_chrome):
            return None
        if not assume_chrome and not any(token in lowered for token in ("search", "look up", "google", "find", "open", "go to", "navigate")):
            return None

        normalized = self._sanitize_spoken_target_text(
            lowered,
            strip_request_words=True,
            strip_browser_context=True,
        )
        normalized = self.normalize_spoken_target(normalized)
        return normalized or None

    def search_google(self, query: str) -> bool:
        if not self._available:
            self._message = "chrome unavailable on this platform"
            return False
        normalized = " ".join((query or "").split()).strip()
        if not normalized:
            self._message = "chrome search query missing"
            return False
        url = f"https://www.google.com/search?q={quote_plus(normalized)}"
        if self._launch_target(url):
            self._message = f"chrome search: {normalized}"
            return True
        self._message = "chrome search failed"
        return False

    def search_youtube(self, query: str, *, kind: str = "auto") -> bool:
        """Open YouTube search results for `query` in Chrome. Used by
        the voice command "play X on YouTube".

        `kind` selects the YouTube search filter:
          - "video"    -> sp=EgIQAQ%253D%253D (videos only — best for
                          single-song queries; first card is reliably
                          a playable video).
          - "playlist" -> sp=EgIQAw%253D%253D (playlists only — best
                          for "play the X playlist" / "play X mix" /
                          "play X songs" intents; first card is a
                          playlist that auto-starts its first video
                          when clicked).
          - "auto"     -> heuristic on the query text: contains
                          'playlist'/'mix'/'album'/'songs'/'compilation'
                          -> playlist, otherwise video.

        Auto-click of the first card is handled by the YouTube
        controller's `play_first_search_result` after the page loads;
        it works the same regardless of result kind."""
        if not self._available:
            self._message = "chrome unavailable on this platform"
            return False
        normalized = " ".join((query or "").split()).strip()
        if not normalized:
            self._message = "youtube search query missing"
            return False
        resolved_kind = str(kind or "auto").lower()
        if resolved_kind == "auto":
            lower = normalized.lower()
            playlist_markers = (
                "playlist", " mix", " album", " songs",
                "compilation", "soundtrack", "best of",
            )
            resolved_kind = "playlist" if any(
                m in (" " + lower) for m in playlist_markers
            ) else "video"
        if resolved_kind == "playlist":
            sp_param = "EgIQAw%253D%253D"
        else:
            sp_param = "EgIQAQ%253D%253D"
        url = (
            "https://www.youtube.com/results"
            f"?search_query={quote_plus(normalized)}"
            f"&sp={sp_param}"
        )
        if self._launch_target(url):
            self._message = f"youtube search ({resolved_kind}): {normalized}"
            return True
        self._message = "youtube search failed"
        return False

    def open_url(self, url: str, *, display_name: str | None = None) -> bool:
        if not self._available:
            self._message = "chrome unavailable on this platform"
            return False
        normalized = " ".join((url or "").split()).strip()
        if not normalized:
            self._message = "chrome target missing"
            return False
        if self._launch_target(normalized):
            self._message = f"chrome open: {display_name or normalized}"
            return True
        self._message = "chrome open failed"
        return False

    def open_or_search(self, target: str) -> bool:
        normalized = self.normalize_spoken_target(target)
        if not normalized:
            return self.focus_or_open_window()
        known_target = KNOWN_WEB_TARGETS.get(normalized.lower())
        if known_target is not None:
            return self.open_url(known_target, display_name=normalized)
        direct_url = self._normalize_target_url(normalized)
        if direct_url is not None:
            return self.open_url(direct_url, display_name=normalized)
        return self.search_google(normalized)

    def normalize_spoken_target(self, target: str) -> str:
        normalized = self._sanitize_spoken_target_text(
            target,
            strip_request_words=False,
            strip_browser_context=True,
        )
        if not normalized:
            return ""
        lowered = normalized.lower()
        best_target = normalized
        best_score = 0.0
        for label, url in KNOWN_WEB_TARGETS.items():
            display_domain = self._display_domain(url)
            for candidate in {
                label.lower(),
                display_domain,
                display_domain.removeprefix("www."),
            }:
                score = self._spoken_target_score(lowered, candidate)
                if score > best_score:
                    best_score = score
                    best_target = display_domain if "." in lowered or "." in candidate else label
        if best_score >= 0.80:
            return best_target
        return normalized


    def _sanitize_spoken_target_text(
        self,
        target: str,
        *,
        strip_request_words: bool,
        strip_browser_context: bool,
    ) -> str:
        normalized = " ".join((target or "").split()).strip().lower()
        if not normalized:
            return ""
        normalized = normalized.replace("you tube", "youtube")
        normalized = normalized.replace("chat gpt", "chatgpt")
        normalized = re.sub(r"[?!,]+", " ", normalized)

        if strip_request_words:
            normalized = re.sub(r"\b(can you|could you|would you|will you|please|for me)\b", " ", normalized)
            normalized = re.sub(
                r"\b(search up|search for|search|look up|look for|find|open|go to|navigate to|take me to|show me)\b",
                " ",
                normalized,
            )
            normalized = re.sub(r"\bup\b", " ", normalized)

        if strip_browser_context:
            normalized = re.sub(
                r"\b(on|in|using|with)\s+(google chrome|chrome browser|chrome|browser)\b",
                " ",
                normalized,
            )
            normalized = re.sub(
                r"\b(on|in|using|with)\s+google\b",
                " ",
                normalized,
            )
            normalized = re.sub(r"\b(google chrome|chrome browser|chrome)\b", " ", normalized)
            normalized = re.sub(r"\bbrowser\b", " ", normalized)

        normalized = re.sub(r"\s+", " ", normalized).strip(" .!?")

        # Remove lingering browser/context tails such as "youtube on google"
        normalized = re.sub(r"\b(on|in|using|with)\s+(google|chrome)\b$", "", normalized).strip(" .!?")
        normalized = re.sub(r"\b(google chrome|chrome browser|chrome)\b$", "", normalized).strip(" .!?")
        normalized = re.sub(r"\bgoogle\b$", "", normalized).strip(" .!?")

        return " ".join(normalized.split()).strip()

    def _default_executable_paths(self) -> tuple[Path, ...]:
        user_profile = Path.home()
        program_files = Path.home().anchor + "Program Files"
        program_files_x86 = Path.home().anchor + "Program Files (x86)"
        local_appdata = user_profile / "AppData" / "Local"
        return (
            Path(program_files) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path(program_files_x86) / "Google" / "Chrome" / "Application" / "chrome.exe",
            local_appdata / "Google" / "Chrome" / "Application" / "chrome.exe",
        )

    def _foreground_window_handle(self) -> int | None:
        if not self._available:
            return None
        try:
            foreground = ctypes.windll.user32.GetForegroundWindow()
        except Exception:
            return None
        return int(foreground) if foreground else None

    def _chrome_window_handles(self) -> list[int]:
        if not self._available:
            return []
        now = time.monotonic()
        if now < self._handles_cache_until:
            return list(self._handles_cache)
        try:
            chrome_pids = {
                int(proc.info["pid"])
                for proc in psutil.process_iter(["pid", "name"])
                if "chrome" in (proc.info.get("name") or "").lower()
            }
        except Exception:
            chrome_pids = set()
        if not chrome_pids:
            self._handles_cache = []
            self._handles_cache_until = now + 1.0
            return []

        user32 = ctypes.windll.user32
        handles: list[int] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def _enum_windows(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if int(pid.value) not in chrome_pids:
                return True
            title_length = user32.GetWindowTextLengthW(hwnd)
            if title_length <= 0:
                return True
            handles.append(int(hwnd))
            return True

        try:
            user32.EnumWindows(_enum_windows, 0)
        except Exception:
            handles = []
        self._handles_cache = list(handles)
        self._handles_cache_until = now + 1.0
        return handles

    def _invalidate_handles_cache(self) -> None:
        self._handles_cache_until = 0.0

    def has_youtube_open(self) -> bool:
        if not self._available:
            return False
        chrome_pids = {
            int(proc.info["pid"])
            for proc in psutil.process_iter(["pid", "name"])
            if "chrome" in (proc.info.get("name") or "").lower()
        }
        if not chrome_pids:
            return False
        user32 = ctypes.windll.user32
        found = [False]

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def _enum_cb(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if int(pid.value) not in chrome_pids:
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if "youtube" in buf.value.lower():
                found[0] = True
            return True

        try:
            user32.EnumWindows(_enum_cb, 0)
        except Exception:
            return False
        return found[0]

    def _wait_for_window_handles(self, timeout_seconds: float = 4.0) -> list[int]:
        deadline = time.monotonic() + timeout_seconds
        handles = self._chrome_window_handles()
        while not handles and time.monotonic() < deadline:
            time.sleep(0.2)
            handles = self._chrome_window_handles()
        return handles

    def _activate_window_handle(self, hwnd: int) -> bool:
        if not self._available:
            return False
        user32 = ctypes.windll.user32
        try:
            user32.ShowWindow(wintypes.HWND(hwnd), SW_RESTORE)
            user32.BringWindowToTop(wintypes.HWND(hwnd))
            success = bool(user32.SetForegroundWindow(wintypes.HWND(hwnd)))
            return success or self._foreground_window_handle() == hwnd
        except Exception:
            return False

    def _send_shortcut(self, *keys: int) -> bool:
        if not self._available:
            return False
        if not keys:
            return False
        user32 = ctypes.windll.user32
        modifiers = list(keys[:-1])
        main_key = keys[-1]
        try:
            for key in modifiers:
                user32.keybd_event(key, 0, 0, 0)
            user32.keybd_event(main_key, 0, 0, 0)
            user32.keybd_event(main_key, 0, KEYEVENTF_KEYUP, 0)
            for key in reversed(modifiers):
                user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0)
            return True
        except Exception:
            return False

    def _launch_target(self, *args: str) -> bool:
        for candidate in self._executable_paths:
            if not candidate.exists():
                continue
            try:
                subprocess.Popen([str(candidate), *args])
                return True
            except Exception:
                continue
        # Fallback via ShellExecuteW (App Paths registry lookup) — avoids
        # `cmd /c start` which Norton SONAR flags as a dropper pattern.
        return launch_external("chrome", args=args)

    def _normalize_target_url(self, target: str) -> str | None:
        normalized = target.strip()
        if not normalized:
            return None
        lowered = normalized.lower()
        if lowered.startswith("http://") or lowered.startswith("https://"):
            return normalized
        if " " in normalized:
            return None
        if "." not in normalized:
            return None
        return f"https://{normalized}"

    def _display_domain(self, url: str) -> str:
        parsed = urlparse(url)
        host = parsed.netloc.lower().removeprefix("www.")
        return host or url.lower()

    def _spoken_target_score(self, spoken: str, candidate: str) -> float:
        if not spoken or not candidate:
            return 0.0
        if spoken == candidate:
            return 1.0
        spoken_tokens = set(re.sub(r"[^a-z0-9.]+", " ", spoken).split())
        candidate_tokens = set(re.sub(r"[^a-z0-9.]+", " ", candidate).split())
        overlap = len(spoken_tokens & candidate_tokens) / max(len(spoken_tokens), len(candidate_tokens), 1)
        ratio = SequenceMatcher(None, spoken, candidate).ratio()
        compact_spoken = re.sub(r"[^a-z0-9]+", "", spoken)
        compact_candidate = re.sub(r"[^a-z0-9]+", "", candidate)
        compact_ratio = SequenceMatcher(None, compact_spoken, compact_candidate).ratio()
        edge_bonus = 0.10 if candidate.startswith(spoken) or spoken.startswith(candidate) else 0.0
        dot_bonus = 0.08 if "." in spoken and "." in candidate else 0.0
        return max(
            ratio * 0.66 + overlap * 0.34 + edge_bonus + dot_bonus,
            compact_ratio * 0.82 + dot_bonus,
        )

# Author: Konstantin Markov

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
import shlex
import shutil
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote
from typing import Any

import psutil

from ..utils.subprocess_utils import launch_external

try:
    import winreg
except ImportError:  # pragma: no cover - only unavailable off Windows
    winreg = None


APP_QUERY_EDGE_WORDS = {
    "a",
    "an",
    "app",
    "application",
    "bring",
    "focus",
    "launcher",
    "launch",
    "my",
    "new",
    "open",
    "please",
    "program",
    "show",
    "start",
    "switch",
    "the",
    "to",
    "up",
    "window",
}

DYNAMIC_ALIAS_STOP_WORDS = {
    "app",
    "apps",
    "application",
    "assistant",
    "beta",
    "browser",
    "community",
    "desktop",
    "edition",
    "experience",
    "for",
    "games",
    "launcher",
    "microsoft",
    "new",
    "openai",
    "preview",
    "program",
    "play",
    "riot",
    "sample",
    "samples",
    "store",
    "the",
    "tool",
    "tools",
    "utility",
    "utilities",
    "windows",
    "documentation",
}

VENDOR_PREFIXES = {
    "adobe",
    "epic",
    "github",
    "google",
    "microsoft",
    "mozilla",
    "nvidia",
    "openai",
    "oracle",
    "riot",
    "steam",
    "ubisoft",
}

FILE_SEARCH_ROOT_NAMES = ("Desktop", "Documents", "Downloads", "Music", "Pictures", "Videos", "OneDrive")

# Top-level dirs to skip when shallow-scanning a whole drive root (C:\) —
# the huge OS/app trees no user means by a folder name. Keeps the
# drive-root scan fast and the results relevant (so "HGR App v1.0.0" at
# C:\ is found without descending into Windows/Program Files).
_DRIVE_SCAN_SKIP_DIRS = frozenset({
    "windows", "program files", "program files (x86)", "programdata",
    "$recycle.bin", "$winreagent", "$sysreset", "system volume information",
    "recovery", "perflogs", "msocache", "config.msi", "intel", "appdata",
    "windows.old", "onedrivetemp",
})
FILE_EXTENSION_ALIASES = {
    "pdf": ".pdf",
    "doc": ".doc",
    "docx": ".docx",
    "word": ".docx",
    "spreadsheet": ".xlsx",
    "excel": ".xlsx",
    "xlsx": ".xlsx",
    "xls": ".xls",
    "ppt": ".ppt",
    "pptx": ".pptx",
    "powerpoint": ".pptx",
    "txt": ".txt",
    "text": ".txt",
    "csv": ".csv",
    "json": ".json",
    "py": ".py",
}

FILE_QUERY_ABBREVIATIONS = {
    "homework": ("hw",),
    "assignment": ("asg", "hw"),
    "project": ("proj",),
    "solution": ("sol", "soln"),
    "report": ("rpt",),
    "presentation": ("ppt", "slides"),
    "document": ("doc",),
    "spreadsheet": ("sheet",),
}

FILE_QUERY_HOMOPHONES = {
    # 0-10 number words and digits map to one another so "report
    # two" and "report 2" both surface "Report 2.pdf". The "two/
    # too/to" cluster keeps its homophones because all three are
    # commonly transcribed for the digit 2 — whisper sometimes
    # picks "to" for ambiguous audio.
    "0": ("0", "zero"),
    "zero": ("0", "zero"),
    "1": ("1", "one"),
    "one": ("1", "one"),
    "2": ("2", "two", "too", "to"),
    "two": ("2", "two", "too", "to"),
    "too": ("2", "two", "too", "to"),
    "to": ("2", "two", "too", "to"),
    "3": ("3", "three"),
    "three": ("3", "three"),
    "4": ("4", "four", "for"),
    "four": ("4", "four", "for"),
    "for": ("4", "four", "for"),
    "5": ("5", "five"),
    "five": ("5", "five"),
    "6": ("6", "six"),
    "six": ("6", "six"),
    "7": ("7", "seven"),
    "seven": ("7", "seven"),
    "8": ("8", "eight"),
    "eight": ("8", "eight"),
    "9": ("9", "nine"),
    "nine": ("9", "nine"),
    "10": ("10", "ten"),
    "ten": ("10", "ten"),
}
APPLICATION_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
APPLICATION_ROMAN_NUMERALS = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
    "vii": "7",
    "viii": "8",
    "ix": "9",
    "x": "10",
}

WM_CLOSE = 0x0010
SW_MAXIMIZE = 3
SW_MINIMIZE = 6
SW_RESTORE = 9


@dataclass(frozen=True)
class DesktopAppEntry:
    display_name: str
    normalized_name: str
    target: str
    source: str
    aliases: tuple[str, ...] = ()
    category: str = "generic"


class DesktopController:
    _shared_app_catalog: list[DesktopAppEntry] | None = None
    _shared_app_catalog_build_thread: "threading.Thread | None" = None
    _shared_app_catalog_build_lock = threading.Lock()

    SETTINGS_URIS = {
        "apps": "ms-settings:appsfeatures",
        "bluetooth": "ms-settings:bluetooth",
        "camera": "ms-settings:privacy-webcam",
        "display": "ms-settings:display",
        "email": "ms-settings:emailandaccounts",
        "network": "ms-settings:network",
        "privacy": "ms-settings:privacy",
        "sound": "ms-settings:sound",
        "storage": "ms-settings:storagesense",
        "update": "ms-settings:windowsupdate",
        "volume": "ms-settings:sound",
        "wifi": "ms-settings:network-wifi",
    }

    KNOWN_APPLICATIONS = {
        "visual studio code": {
            "aliases": ("visual studio code", "visual studio", "visual studios", "vs code", "vscode", "code"),
            "category": "editor",
            "targets": (
                str(Path.home() / "AppData" / "Local" / "Programs" / "Microsoft VS Code" / "Code.exe"),
                str(Path(Path.home().anchor + "Program Files") / "Microsoft VS Code" / "Code.exe"),
                str(Path(Path.home().anchor + "Program Files (x86)") / "Microsoft VS Code" / "Code.exe"),
                "code",
            ),
        },
        "steam": {
            "aliases": ("steam",),
            "category": "gaming",
            "targets": (
                "steam://open/main",
                "steam",
                str(Path(Path.home().anchor + "Program Files (x86)") / "Steam" / "steam.exe"),
                str(Path(Path.home().anchor + "Program Files") / "Steam" / "steam.exe"),
            ),
        },
        "google chrome": {
            "aliases": ("google chrome", "chrome"),
            "category": "browser",
            "targets": (
                str(Path(Path.home().anchor + "Program Files") / "Google" / "Chrome" / "Application" / "chrome.exe"),
                str(Path(Path.home().anchor + "Program Files (x86)") / "Google" / "Chrome" / "Application" / "chrome.exe"),
                str(Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "Application" / "chrome.exe"),
                "chrome",
            ),
        },
        "spotify": {
            "aliases": ("spotify",),
            "category": "music",
            "targets": (
                "spotify:",
                "spotify",
                str(Path.home() / "AppData" / "Roaming" / "Spotify" / "Spotify.exe"),
                str(Path.home() / "AppData" / "Local" / "Microsoft" / "WindowsApps" / "SpotifyAB.SpotifyMusic_zpdnekdrzrea0" / "Spotify.exe"),
            ),
        },
        "outlook": {
            "aliases": ("outlook", "mail", "email"),
            "category": "email",
            "targets": ("outlook", "olk.exe", "outlook.exe"),
        },
        "discord": {
            "aliases": ("discord",),
            "category": "chat",
            "targets": (
                "hgr:discord",
                str(Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Discord Inc" / "Discord.lnk"),
                str(Path.home() / "Desktop" / "Discord.lnk"),
            ),
        },
        "hoyoplay": {
            "aliases": ("hoyoplay", "hoyo play", "ho yo play"),
            "category": "gaming",
            "targets": (
                "hgr:hoyoplay",
                str(Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "HoYoPlay.lnk"),
                str(Path.home() / "Desktop" / "HoYoPlay.lnk"),
            ),
        },
        "genshin impact": {
            "aliases": ("genshin impact", "genshin"),
            "category": "gaming",
            "targets": (
                "hgr:genshin-impact",
                str(Path.home() / "Desktop" / "Genshin Impact.lnk"),
                str(Path(os.getenv("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Genshin Impact.lnk"),
            ),
        },
        "valorant": {
            "aliases": ("valorant", "valorent"),
            "category": "gaming",
            "targets": (
                "hgr:valorant",
                str(Path.home() / "Desktop" / "VALORANT.lnk"),
                str(Path(os.getenv("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "VALORANT.lnk"),
            ),
        },
        "riot client": {
            "aliases": ("riot client",),
            "category": "gaming",
            "targets": (
                "hgr:riot-client",
                str(Path.home() / "Desktop" / "Riot Client.lnk"),
                str(Path(os.getenv("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Riot Client.lnk"),
            ),
        },
        "matlab": {
            "aliases": ("matlab", "mat lab"),
            "category": "engineering",
            "targets": (
                "hgr:matlab",
                "matlab",
            ),
        },
        "kicad": {
            "aliases": ("kicad", "ki cad", "keycad", "keycard"),
            "category": "cad",
            "targets": (
                "hgr:kicad",
                "kicad",
            ),
        },
    }

    OUTLOOK_FOLDERS = {
        "calendar": ("calendar",),
        "deleted items": ("deleted items", "trash", "deleted"),
        "drafts": ("drafts", "draft"),
        "inbox": ("inbox",),
        "junk email": ("junk", "junk email", "spam"),
        "outbox": ("outbox",),
        "sent items": ("sent items", "sent item", "sent", "scent"),
    }
    OUTLOOK_FOLDER_DISPLAY_NAMES = {
        "calendar": "Calendar",
        "deleted items": "Deleted Items",
        "drafts": "Drafts",
        "inbox": "Inbox",
        "junk email": "Junk Email",
        "outbox": "Outbox",
        "sent items": "Sent Items",
    }

    SOURCE_PRIORITY = {
        "known": 6,
        "start_apps": 5,
        "app_paths": 4,
        "start_menu": 3,
        "uninstall": 2,
    }

    def __init__(self, *, outlook_paths: tuple[Path, ...] | None = None) -> None:
        system = platform.system()
        self._win = system == "Windows"
        self._mac = system == "Darwin"
        # macOS: open apps/files/folders via the `open` command (see _mac_open).
        # Window-control (min/max/close of the active window) stays Windows-only
        # for now and reports "unavailable" on macOS rather than running win32.
        self._available = self._win or self._mac
        self._message = "desktop idle"
        self._outlook_paths = outlook_paths or self._default_outlook_paths()
        self._app_catalog: list[DesktopAppEntry] | None = None
        self._quick_app_catalog: list[DesktopAppEntry] | None = None
        self._indexed_search_cache: dict[tuple[str, tuple[str, ...], str | None, str], tuple[float, list[Path]]] = {}
        self._indexed_search_supported: bool | None = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def message(self) -> str:
        return self._message

    def _subprocess_hidden_kwargs(self) -> dict:
        kwargs: dict = {}
        if platform.system() == "Windows":
            create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo_cls = getattr(subprocess, "STARTUPINFO", None)
            if create_no_window:
                kwargs["creationflags"] = create_no_window
            if startupinfo_cls is not None:
                startupinfo = startupinfo_cls()
                startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
                startupinfo.wShowWindow = 0
                kwargs["startupinfo"] = startupinfo
        return kwargs

    def refresh_application_catalog(self) -> int:
        self._app_catalog = None
        self._quick_app_catalog = None
        type(self)._shared_app_catalog = None
        return len(self._application_catalog())

    def can_resolve_application(self, app_name: str) -> bool:
        return self._resolve_application(app_name) is not None

    def application_catalog_snapshot(self) -> list[DesktopAppEntry]:
        return list(self._application_catalog())

    def application_hint_names(self, *, limit: int = 20) -> list[str]:
        names: list[str] = list(self.KNOWN_APPLICATIONS.keys())
        catalog = self._app_catalog or type(self)._shared_app_catalog or []
        for entry in catalog:
            display_name = str(entry.display_name or "").strip()
            if display_name:
                names.append(display_name)
        deduped: list[str] = []
        seen: set[str] = set()
        for name in names:
            key = self._normalize_application_name(name)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(name)
            if len(deduped) >= max(1, int(limit)):
                break
        return deduped

    def rank_applications_in_text(self, text: str, *, limit: int = 6) -> list[tuple[DesktopAppEntry, float, str | None]]:
        normalized = self._normalize_application_query(text)
        if not normalized:
            normalized = self._normalize_application_name(text)
        if not normalized:
            return []
        tokens = normalized.split()
        ranked: list[tuple[DesktopAppEntry, float, str | None]] = []
        quick_ranked: list[tuple[DesktopAppEntry, float, str | None]] = []
        for entry in self._quick_application_catalog():
            score, matched_alias = self._application_mention_score(normalized, tokens, entry)
            if score <= 0.0:
                continue
            quick_ranked.append((entry, score, matched_alias))
        quick_ranked.sort(
            key=lambda item: (
                item[1],
                self.SOURCE_PRIORITY.get(item[0].source, 0),
                len(item[0].aliases),
                -len(item[0].normalized_name),
            ),
            reverse=True,
        )
        if self._app_catalog is None and quick_ranked and quick_ranked[0][1] >= 0.78:
            return quick_ranked[: max(1, int(limit))]
        for entry in self._application_catalog():
            score, matched_alias = self._application_mention_score(normalized, tokens, entry)
            if score <= 0.0:
                continue
            ranked.append((entry, score, matched_alias))
        ranked.sort(
            key=lambda item: (
                item[1],
                self.SOURCE_PRIORITY.get(item[0].source, 0),
                len(item[0].aliases),
                -len(item[0].normalized_name),
            ),
            reverse=True,
        )
        return ranked[: max(1, int(limit))]

    def open_settings(self, topic: str | None = None) -> bool:
        if not self._available:
            self._message = "settings unavailable on this platform"
            return False
        canonical = self._canonical_settings_topic(topic)
        target = self.SETTINGS_URIS.get(canonical, "ms-settings:")
        if self._launch_target(target):
            self._message = f"opened settings: {canonical}" if canonical else "opened settings"
            return True
        self._message = "settings launch failed"
        return False

    def open_file_explorer(self, location: str | None = None) -> bool:
        if not self._available:
            self._message = "file explorer unavailable on this platform"
            return False
        target_path = self._known_folder_path(location)
        if target_path is not None:
            launched = self._launch_target(str(target_path))
            label = target_path.name
        elif location:
            launched = self._launch_target(location)
            label = location
        else:
            launched = self._launch_target("explorer")
            label = "home"
        self._message = f"opened files: {label}" if launched else "file explorer launch failed"
        return launched

    def search_file_explorer(self, query: str) -> bool:
        if not self._available:
            self._message = "file explorer unavailable on this platform"
            return False
        normalized = " ".join((query or "").split()).strip()
        if not normalized:
            self._message = "file explorer search query missing"
            return False
        target = f"search-ms:query={quote(normalized)}"
        if self._launch_target(target):
            self._message = f"file explorer search: {normalized}"
            return True
        self._message = "file explorer search failed"
        return False

    def resolve_named_file(
        self,
        query: str,
        *,
        preferred_root: str | None = None,
        folder_hint: str | None = None,
    ) -> tuple[Path | None, list[Path]]:
        if not self._available:
            self._message = "file open unavailable on this platform"
            return None, []
        known_folder = self._known_folder_path(query)
        if known_folder is not None and self._normalize_application_name(str(query or "")) in {"desktop", "documents", "downloads", "music", "pictures", "videos"}:
            return known_folder, []
        return self._resolve_path_query(query, preferred_root=preferred_root, folder_hint=folder_hint)

    def resolve_named_folder(
        self,
        query: str,
        *,
        preferred_root: str | None = None,
        folder_hint: str | None = None,
    ) -> tuple[Path | None, list[Path]]:
        if not self._available:
            self._message = "folder open unavailable on this platform"
            return None, []
        known_folder = self._known_folder_path(query)
        if known_folder is not None:
            return known_folder, []
        return self._resolve_folder_query(query, preferred_root=preferred_root, folder_hint=folder_hint)

    def open_resolved_path(self, path: Path) -> bool:
        if not self._available:
            self._message = "file open unavailable on this platform"
            return False
        resolved = Path(path)
        try:
            resolved_lower = str(resolved.resolve()).lower() if resolved.exists() else str(resolved).lower()
        except Exception:
            resolved_lower = str(resolved).lower()
        if resolved_lower.endswith("touchless.exe") or resolved_lower.endswith("hgr app.exe") or resolved_lower.endswith("hgr_app.exe"):
            self._message = "cannot reopen the running Touchless app"
            return False
        if resolved_lower.endswith("touchless.lnk") or resolved_lower.endswith("hgr app.lnk") or resolved_lower.endswith("hgr_app.lnk"):
            self._message = "cannot reopen the running Touchless app"
            return False
        resolved_str = str(resolved)
        if not resolved.exists():
            label = "folder" if resolved_str.endswith(("/", "\\")) else "file"
            self._message = f"could not open {label}: {resolved.name} (not found)"
            return False
        if self._launch_target(resolved_str):
            label = "folder" if resolved.is_dir() else "file"
            self._message = f"opened {label}: {resolved.name}"
            return True
        # Fallback: explorer.exe can open files/folders reliably when os.startfile
        # silently fails (e.g., file association quirks, network paths).
        try:
            subprocess.Popen(["explorer.exe", resolved_str], shell=False)
            label = "folder" if resolved.is_dir() else "file"
            self._message = f"opened {label}: {resolved.name}"
            return True
        except Exception:
            pass
        label = "folder" if resolved.is_dir() else "file"
        self._message = f"could not open {label}: {resolved.name}"
        return False

    def open_named_file(
        self,
        query: str,
        *,
        preferred_root: str | None = None,
        folder_hint: str | None = None,
    ) -> bool:
        resolved, ambiguous = self.resolve_named_file(query, preferred_root=preferred_root, folder_hint=folder_hint)
        if ambiguous:
            pretty = "; ".join(path.name for path in ambiguous[:3])
            self._message = f"multiple matching files found, please give more detail: {pretty}"
            return False
        if resolved is None:
            self._message = f"could not find file: {' '.join((query or '').split()).strip()}"
            return False
        return self.open_resolved_path(resolved)

    def deep_find(
        self,
        query: str,
        *,
        kind: str = "any",
        time_budget: float = 8.0,
        limit: int = 25,
    ) -> tuple[Path | None, list[Path]]:
        """Best-effort DEEP search across all fixed drives + the home tree,
        name-matching files/folders at any depth. Wall-clock time-budgeted so
        it can never hang. Returns (best_match, other_matches) like the normal
        resolvers, so callers handle it the same way. Used as an on-demand
        fallback when the fast (index + shallow) search finds nothing."""
        if not self._available:
            return None, []
        normalized = self._normalize_file_query(query)
        if not normalized:
            return None, []
        qtokens = [t for t in normalized.split() if t]
        # Exact-match target uses the SAME normalizer as candidate names
        # (so the early-exit actually fires; the two normalizers differ).
        exact_target = self._normalize_file_token_text(query)
        deadline = time.monotonic() + max(1.0, float(time_budget))
        want_dirs = kind in ("any", "folder")
        want_files = kind in ("any", "file")

        def _matches(name_norm: str) -> bool:
            if not name_norm:
                return False
            if normalized in name_norm:
                return True
            return bool(qtokens) and all(tok in name_norm for tok in qtokens)

        from collections import deque
        roots = self._fixed_drive_roots()
        home = Path.home()
        if home not in roots:
            roots.append(home)
        found: list[Path] = []
        seen: set[str] = set()
        visited: set[str] = set()
        max_deep_depth = 9
        # Breadth-first: visit shallow directories before deep ones, so a
        # top-level target (C:\HGR App v1.0.0) is found immediately instead
        # of the DFS diving into C:\Users first and burning the budget.
        queue: deque[tuple[Path, int]] = deque((r, 0) for r in roots)
        while queue:
            if time.monotonic() > deadline or len(found) >= limit:
                break
            current_path, depth = queue.popleft()
            dkey = str(current_path).lower()
            if dkey in visited:
                continue
            visited.add(dkey)
            try:
                entries = list(os.scandir(current_path))
            except Exception:
                continue
            subdirs: list[Path] = []
            for entry in entries:
                name = entry.name
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except Exception:
                    is_dir = False
                if is_dir:
                    if name.lower() in _DRIVE_SCAN_SKIP_DIRS or name.startswith("$"):
                        continue
                    p = Path(entry.path)
                    if want_dirs:
                        cn = self._normalize_file_token_text(name)
                        if _matches(cn):
                            if cn == exact_target:  # exact match - stop now
                                return p, []
                            key = str(p).lower()
                            if key not in seen:
                                seen.add(key)
                                found.append(p)
                    subdirs.append(p)
                elif want_files:
                    fn = self._normalize_file_token_text(Path(name).stem)
                    if _matches(fn):
                        p = Path(entry.path)
                        if fn == exact_target:  # exact match - stop now
                            return p, []
                        key = str(p).lower()
                        if key not in seen:
                            seen.add(key)
                            found.append(p)
                            if len(found) >= limit:
                                break
            if depth < max_deep_depth:
                subdirs.sort(key=lambda q: (len(q.name), q.name.lower()))
                for p in subdirs:
                    queue.append((p, depth + 1))
        if not found:
            return None, []
        # Rank: exact name match first, then shallower path, then shorter name.
        def _score(p: Path):
            base = self._normalize_file_token_text(p.stem if p.suffix else p.name)
            return (base == normalized, -len(p.parts), -len(p.name))
        found.sort(key=_score, reverse=True)
        return found[0], found[1:8]

    def resolve_named_application_options(
        self,
        app_name: str,
        *,
        limit: int = 6,
    ) -> tuple[DesktopAppEntry | None, list[DesktopAppEntry]]:
        normalized = self._normalize_application_query(app_name) or self._normalize_application_name(app_name)
        if not normalized:
            return None, []
        ranked = self.rank_applications_in_text(normalized, limit=max(3, int(limit)))
        # ---- Stem-disambiguation pass --------------------------------
        # When the (canonicalised) query tokens are a strict subset of
        # MULTIPLE installed app names, surface those entries directly
        # — bypasses the fuzzy-score floor that otherwise filters out
        # the longer-titled sibling. Real-world: "open fallout" should
        # offer both Fallout 3 and Fallout: New Vegas; the fuzzy score
        # for FNV's bare "fallout" stem lands ~0.5 because "fallout"
        # is only one of three tokens in the title, but the user
        # clearly wants to choose. Same logic also resolves "fallout
        # three" → "fallout 3" cleanly to a single match (because
        # `_exact_match_canonical_form` maps the number word to the
        # digit), short-circuiting any later file-request fallback.
        canonical_query = self._exact_match_canonical_form(normalized)
        query_tokens = set(canonical_query.split()) if canonical_query else set()
        if 1 <= len(query_tokens) <= 3:
            stem_matches: list[DesktopAppEntry] = []
            seen_displays: set[str] = set()
            for entry in self._application_catalog():
                candidate_canonical = self._exact_match_canonical_form(entry.normalized_name)
                candidate_tokens = set(candidate_canonical.split()) if candidate_canonical else set()
                if not candidate_tokens:
                    continue
                if query_tokens <= candidate_tokens and entry.display_name not in seen_displays:
                    stem_matches.append(entry)
                    seen_displays.add(entry.display_name)
            if len(stem_matches) >= 2:
                # Multiple titles contain the query tokens — return
                # them all as disambiguation. Prefer the top-ranked
                # entry as the "suggested default" if one of the
                # stem matches is in the ranked list, else fall back
                # to the first stem hit.
                top_in_stems = next(
                    (entry for entry, _, _ in (ranked or []) if entry.display_name in seen_displays),
                    stem_matches[0],
                )
                return top_in_stems, stem_matches[: max(2, int(limit))]
            if len(stem_matches) == 1:
                return stem_matches[0], []
        # ---- End stem pass -------------------------------------------
        if not ranked:
            resolved = self._resolve_application(app_name)
            if resolved is not None:
                return resolved, []
            return None, []
        top_entry, top_score, _matched_alias = ranked[0]
        if top_score < 0.78:
            return None, []
        top_is_exact = self._entry_exact_query_match(top_entry, normalized)
        # Exact-match short-circuit. When the user spoke (or typed) a
        # full title that exactly matches one entry — including the
        # numeric variant ("slay the spire two" → "Slay the Spire 2") —
        # always open that exact title without offering siblings, even
        # when a sibling shares the score (e.g., "Slay the Spire" ties
        # at 1.08 because every alias the query-string matches against
        # is also present in both entries' alias lists). Without this
        # short-circuit the close-entries ambiguity path below would
        # surface the sibling as a disambiguation choice and force the
        # user to pick again, which directly contradicts what they
        # asked for.
        if top_is_exact:
            return top_entry, []
        # Token-superset ambiguity. When the query is a strict subset
        # of multiple installed apps (e.g., "fallout" matches both
        # "Fallout 3" and "Fallout New Vegas"), surface every such
        # entry. Score gate is intentionally relaxed below the 0.78
        # main threshold here — partial-name matches against
        # multi-token titles often don't reach 0.78 because most of
        # the title is missing from the query, but the user clearly
        # intended a disambiguation. We keep a 0.55 floor so genuinely
        # unrelated catalog entries don't sneak into the prompt.
        token_superset_matches: list[DesktopAppEntry] = []
        for entry, score, _alias in ranked[: max(3, int(limit))]:
            if score < 0.55:
                continue
            if not self._entry_query_token_superset_match(entry, normalized):
                continue
            if any(existing.display_name == entry.display_name for existing in token_superset_matches):
                continue
            token_superset_matches.append(entry)
        if len(token_superset_matches) >= 2:
            return top_entry, token_superset_matches[: max(2, int(limit))]
        close_entries: list[DesktopAppEntry] = []
        top_category = str(getattr(top_entry, 'category', 'generic') or 'generic')
        top_norm = self._normalize_application_name(top_entry.display_name)
        for entry, score, _alias in ranked[: max(3, int(limit))]:
            entry_norm = self._normalize_application_name(entry.display_name)
            similar = (
                score >= top_score - 0.05
                or self._application_match_score(top_norm, entry_norm) >= 0.84
                or self._application_match_score(normalized, entry_norm) >= 0.84
            )
            if not similar:
                continue
            if close_entries and entry.display_name == close_entries[-1].display_name:
                continue
            if not close_entries or str(getattr(entry, 'category', 'generic') or 'generic') == top_category or score >= top_score - 0.02:
                close_entries.append(entry)
        if len(close_entries) <= 1:
            return top_entry, []
        return top_entry, close_entries[: max(2, int(limit))]

    def open_desktop_entry(self, entry: DesktopAppEntry) -> bool:
        if not self._available:
            self._message = "application launch unavailable on this platform"
            return False
        if self._launch_path_or_command(entry.target):
            self._message = f"opened app: {entry.display_name}"
            return True
        self._message = f"could not open app: {entry.display_name}"
        return False

    def open_outlook(self) -> bool:
        if not self._available:
            self._message = "outlook unavailable on this platform"
            return False
        for candidate in self._outlook_paths:
            if candidate.exists() and self._launch_path_or_command(str(candidate)):
                self._message = "opened outlook"
                return True
        resolved = self._resolve_application("outlook")
        if resolved is not None and self._launch_path_or_command(resolved.target):
            self._message = "opened outlook"
            return True
        self._message = "outlook launch failed"
        return False

    def open_outlook_folder(self, folder_name: str | None) -> bool:
        if not self._available:
            self._message = "outlook unavailable on this platform"
            return False
        canonical = self._canonical_outlook_folder(folder_name)
        if canonical is None:
            return self.open_outlook()
        display_name = self.outlook_folder_display_name(canonical) or canonical
        classic_path = self._classic_outlook_path()
        if classic_path is not None:
            try:
                subprocess.Popen([str(classic_path), "/select", f"outlook:{display_name}"], shell=False)
                self._message = f"opened outlook folder: {display_name}"
                return True
            except Exception:
                pass
        if self.open_outlook():
            self._message = f"opened outlook, but could not select {display_name}"
            return False
        self._message = "outlook folder launch failed"
        return False

    def compose_email(
        self,
        *,
        recipient: str | None = None,
        subject: str | None = None,
        body: str | None = None,
    ) -> bool:
        if not self._available:
            self._message = "email compose unavailable on this platform"
            return False
        mailto = self._build_mailto_link(recipient=recipient, subject=subject, body=body)
        if self._launch_target(mailto):
            self._message = "opened email composer"
            return True
        self._message = "email composer launch failed"
        return False

    def open_named_application(self, app_name: str) -> bool:
        if not self._available:
            self._message = "application launch unavailable on this platform"
            return False
        if self._mac:
            spoken = " ".join((app_name or "").split()).strip()
            if not spoken:
                self._message = "no app name given"
                return False
            # Prefer the scanned catalog (proper display name + alias match),
            # then fall back to LaunchServices fuzzy matching via `open -a`.
            resolved = self._resolve_application(spoken)
            if resolved is not None and self._launch_path_or_command(resolved.target):
                self._message = f"opened app: {resolved.display_name}"
                return True
            if self._mac_open(spoken, as_app=True):
                self._message = f"opened app: {spoken}"
                return True
            self._message = f"could not open app: {spoken}"
            return False
        resolved = self._resolve_application(app_name)
        if resolved is None:
            self._message = f"could not find app: {' '.join((app_name or '').split()).strip()}"
            return False
        if self._launch_path_or_command(resolved.target):
            self._message = f"opened app: {resolved.display_name}"
            return True
        self._message = f"could not open app: {resolved.display_name}"
        return False

    def _default_outlook_paths(self) -> tuple[Path, ...]:
        program_files = Path.home().anchor + "Program Files"
        program_files_x86 = Path.home().anchor + "Program Files (x86)"
        user_profile = Path.home()
        return (
            Path(program_files) / "Microsoft Office" / "root" / "Office16" / "OUTLOOK.EXE",
            Path(program_files_x86) / "Microsoft Office" / "root" / "Office16" / "OUTLOOK.EXE",
            user_profile / "AppData" / "Local" / "Microsoft" / "WindowsApps" / "olk.exe",
            user_profile / "AppData" / "Local" / "Microsoft" / "WindowsApps" / "outlook.exe",
        )

    def _build_mailto_link(
        self,
        *,
        recipient: str | None,
        subject: str | None,
        body: str | None,
    ) -> str:
        recipient_text = quote(" ".join((recipient or "").split()).strip())
        query_parts: list[str] = []
        if subject:
            query_parts.append(f"subject={quote(subject)}")
        if body:
            query_parts.append(f"body={quote(body)}")
        query_text = "&".join(query_parts)
        if query_text:
            return f"mailto:{recipient_text}?{query_text}"
        return f"mailto:{recipient_text}"

    def _canonical_settings_topic(self, topic: str | None) -> str | None:
        normalized = " ".join((topic or "").lower().split()).strip()
        if not normalized:
            return None
        for canonical, aliases in {
            "apps": ("apps", "applications", "installed apps", "programs"),
            "bluetooth": ("bluetooth",),
            "camera": ("camera", "webcam"),
            "display": ("display", "screen", "monitor"),
            "email": ("email", "mail", "accounts"),
            "network": ("network",),
            "privacy": ("privacy", "permissions"),
            "sound": ("sound", "audio"),
            "storage": ("storage", "disk", "drive"),
            "update": ("update", "updates", "windows update"),
            "volume": ("volume",),
            "wifi": ("wifi", "wi-fi", "wireless"),
        }.items():
            for alias in aliases:
                if alias in normalized:
                    return canonical
        return normalized if normalized in self.SETTINGS_URIS else None

    def _canonical_outlook_folder(self, folder_name: str | None) -> str | None:
        normalized = " ".join((folder_name or "").lower().split()).strip()
        if not normalized:
            return None
        for canonical, aliases in self.OUTLOOK_FOLDERS.items():
            for alias in aliases:
                if alias in normalized:
                    return canonical
        return None

    @classmethod
    def outlook_folder_display_name(cls, folder_name: str | None) -> str | None:
        normalized = " ".join((folder_name or "").lower().split()).strip()
        if not normalized:
            return None
        canonical = normalized if normalized in cls.OUTLOOK_FOLDERS else None
        if canonical is None:
            for candidate, aliases in cls.OUTLOOK_FOLDERS.items():
                if any(alias in normalized for alias in aliases):
                    canonical = candidate
                    break
        if canonical is None:
            return None
        return cls.OUTLOOK_FOLDER_DISPLAY_NAMES.get(canonical, canonical.title())

    def _known_folder_path(self, location: str | None) -> Path | None:
        normalized = " ".join((location or "").lower().split()).strip()
        if not normalized:
            return None
        home = Path.home()
        known_paths = {
            "desktop": home / "Desktop",
            "documents": home / "Documents",
            "downloads": home / "Downloads",
            "music": home / "Music",
            "pictures": home / "Pictures",
            # macOS uses ~/Movies for video; there is no ~/Videos by default.
            "videos": (home / "Movies") if self._mac else (home / "Videos"),
            "movies": home / "Movies",
            "onedrive": home / "OneDrive",
            "applications": Path("/Applications") if self._mac else home,
            "home": home,
        }
        for name, path in known_paths.items():
            if name in normalized:
                return path
        candidate = Path(location).expanduser()
        return candidate if candidate.exists() else None

    def _resolve_file_query(
        self,
        query: str,
        *,
        preferred_root: str | None = None,
        folder_hint: str | None = None,
    ) -> tuple[Path | None, list[Path]]:
        variants = self._generate_file_query_variants(query)
        if not variants:
            return None, []
        desired_extension = None
        primary_tokens: list[str] = []
        for token in variants[0].split():
            mapped = FILE_EXTENSION_ALIASES.get(token)
            if mapped is not None:
                desired_extension = mapped
                continue
            primary_tokens.append(token)
        preferred_roots, folder_hints, ordered_folder_hints = self._file_search_preferences(
            query,
            preferred_root=preferred_root,
            folder_hint=folder_hint,
        )

        scored: list[tuple[float, float, Path]] = []
        search_roots = self._file_search_roots(preferred_roots, folder_hints, ordered_folder_hints)
        indexed_candidates = self._query_indexed_paths(
            token_sets=self._file_query_token_sets(variants),
            search_roots=search_roots,
            desired_extension=desired_extension,
            item_kind="file",
            limit=60,
        )
        for path in indexed_candidates:
            scored.extend(
                self._score_file_candidate(
                    path,
                    variants=variants,
                    desired_extension=desired_extension,
                    preferred_roots=preferred_roots,
                    folder_hints=folder_hints,
                    ordered_folder_hints=ordered_folder_hints,
                )
            )
        if not scored:
            scored.extend(
                self._scan_file_candidates(
                    search_roots=search_roots,
                    variants=variants,
                    desired_extension=desired_extension,
                    preferred_roots=preferred_roots,
                    folder_hints=folder_hints,
                    ordered_folder_hints=ordered_folder_hints,
                )
            )

        deduped_scored: dict[str, tuple[float, float, Path]] = {}
        for score, modified_time, path in scored:
            key = str(path).lower()
            current = deduped_scored.get(key)
            if current is None or (score, modified_time) > (current[0], current[1]):
                deduped_scored[key] = (score, modified_time, path)
        scored = list(deduped_scored.values())

        if not scored:
            return None, []

        scored.sort(key=lambda item: (item[0], item[1], -len(item[2].name)), reverse=True)
        best_score, _best_mtime, best_path = scored[0]
        if best_score < 0.74:
            return None, []

        query_numbers_union = set()
        for variant in variants:
            query_numbers_union |= set(re.findall(r"\d+", variant))
        best_numbers = set(re.findall(r"\d+", self._normalize_file_token_text(best_path.name)))
        best_canonical = self._canonical_file_homophone_text(best_path.stem)
        # Treat any spoken-number-bearing query as homophone-aware so
        # the looser ambiguity gate kicks in when the user said
        # "report three" / "report 3" / "report iii" — all three
        # canonicalize to the same form, and we want sibling files
        # ("Report 3 v2.pdf") to surface as ambiguous candidates the
        # same way they do for the "two" cluster.
        homophone_tokens = (
            {"2", "two", "too", "to"}
            | set(APPLICATION_NUMBER_WORDS.keys())
            | set(APPLICATION_NUMBER_WORDS.values())
            | set(APPLICATION_ROMAN_NUMERALS.keys())
        )
        homophone_query = any(
            token in homophone_tokens for variant in variants for token in variant.split()
        )
        ambiguous: list[Path] = []
        for score, _mtime, path in scored[1:8]:
            same_parent = path.parent == best_path.parent
            similar_name = self._normalize_file_token_text(path.stem) == self._normalize_file_token_text(best_path.stem)
            canonical_match = self._canonical_file_homophone_text(path.stem) == best_canonical
            path_numbers = set(re.findall(r"\d+", self._normalize_file_token_text(path.name)))
            number_compatible = (
                homophone_query
                or not query_numbers_union
                or (query_numbers_union <= best_numbers and query_numbers_union <= path_numbers)
            )
            close_score = score >= best_score - (0.05 if homophone_query else 0.02)
            if number_compatible and (canonical_match or (close_score and (same_parent or similar_name or score >= best_score - 0.03))):
                ambiguous.append(path)

        return best_path, ambiguous

    def _resolve_path_query(
        self,
        query: str,
        *,
        preferred_root: str | None = None,
        folder_hint: str | None = None,
    ) -> tuple[Path | None, list[Path]]:
        resolved, ambiguous = self._resolve_file_query(query, preferred_root=preferred_root, folder_hint=folder_hint)
        if resolved is not None or ambiguous:
            return resolved, ambiguous
        return self._resolve_folder_query(query, preferred_root=preferred_root, folder_hint=folder_hint)

    def _resolve_folder_query(
        self,
        query: str,
        *,
        preferred_root: str | None = None,
        folder_hint: str | None = None,
    ) -> tuple[Path | None, list[Path]]:
        normalized = self._normalize_file_query(query)
        if not normalized:
            return None, []
        preferred_roots, folder_hints, ordered_folder_hints = self._file_search_preferences(
            query,
            preferred_root=preferred_root,
            folder_hint=folder_hint,
        )
        query_numbers = set(re.findall(r"\d+", normalized))
        query_tokens = [token for token in normalized.split() if token]
        scored: list[tuple[float, Path]] = []
        search_roots = self._file_search_roots(preferred_roots, folder_hints, ordered_folder_hints)
        indexed_candidates = self._query_indexed_paths(
            token_sets=self._folder_query_token_sets(normalized),
            search_roots=search_roots,
            desired_extension=None,
            item_kind="folder",
            limit=40,
        )
        for path in indexed_candidates:
            candidate_name = self._normalize_file_token_text(path.name)
            score = self._application_match_score(normalized, candidate_name)
            if normalized and normalized in candidate_name:
                score += 0.16
            if query_tokens:
                shared = len(set(candidate_name.split()) & set(query_tokens))
                score += min(0.18, shared * 0.04)
            candidate_numbers = set(re.findall(r"\d+", candidate_name))
            if query_numbers:
                if query_numbers <= candidate_numbers:
                    score += 0.20
                else:
                    score -= 0.18
            if preferred_roots and any(str(path).lower().startswith(str(root).lower()) for root in preferred_roots):
                score += 0.12
            if folder_hints:
                parent_names = {self._normalize_file_token_text(part) for part in path.parent.parts}
                if any(hint in parent_names for hint in folder_hints):
                    score += 0.18
            if ordered_folder_hints:
                ordered_parts = [self._normalize_file_token_text(part) for part in path.parts]
                cursor = 0
                matched_order = 0
                for hint in ordered_folder_hints:
                    for index in range(cursor, len(ordered_parts)):
                        if ordered_parts[index] == hint or hint in ordered_parts[index]:
                            matched_order += 1
                            cursor = index + 1
                            break
                if matched_order:
                    score += min(0.22, matched_order * 0.07)
            if score >= 0.62:
                scored.append((score, path))
        if not scored:
            for path in self._scan_folder_candidates(search_roots):
                candidate_name = self._normalize_file_token_text(path.name)
                score = self._application_match_score(normalized, candidate_name)
                if normalized and normalized in candidate_name:
                    score += 0.16
                if query_tokens:
                    shared = len(set(candidate_name.split()) & set(query_tokens))
                    score += min(0.18, shared * 0.04)
                candidate_numbers = set(re.findall(r"\d+", candidate_name))
                if query_numbers:
                    if query_numbers <= candidate_numbers:
                        score += 0.20
                    else:
                        score -= 0.18
                if preferred_roots and any(str(path).lower().startswith(str(root).lower()) for root in preferred_roots):
                    score += 0.12
                if folder_hints:
                    parent_names = {self._normalize_file_token_text(part) for part in path.parent.parts}
                    if any(hint in parent_names for hint in folder_hints):
                        score += 0.18
                if ordered_folder_hints:
                    ordered_parts = [self._normalize_file_token_text(part) for part in path.parts]
                    cursor = 0
                    matched_order = 0
                    for hint in ordered_folder_hints:
                        for index in range(cursor, len(ordered_parts)):
                            if ordered_parts[index] == hint or hint in ordered_parts[index]:
                                matched_order += 1
                                cursor = index + 1
                                break
                    if matched_order:
                        score += min(0.22, matched_order * 0.07)
                if score >= 0.62:
                    scored.append((score, path))
        deduped_scored: dict[str, tuple[float, Path]] = {}
        for score, path in scored:
            key = str(path).lower()
            current = deduped_scored.get(key)
            if current is None or score > current[0]:
                deduped_scored[key] = (score, path)
        scored = list(deduped_scored.values())

        if not scored:
            return None, []
        scored.sort(key=lambda item: (item[0], len(item[1].name)), reverse=True)
        best_score, best_path = scored[0]
        if best_score < 0.78:
            return None, []
        ambiguous: list[Path] = []
        for score, path in scored[1:6]:
            if score >= best_score - 0.02 and path.parent == best_path.parent:
                ambiguous.append(path)
        return best_path, ambiguous

    def _file_query_token_sets(self, variants: list[str]) -> list[tuple[str, ...]]:
        token_sets: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for variant in variants:
            tokens = tuple(
                dict.fromkeys(
                    token
                    for token in variant.split()
                    if FILE_EXTENSION_ALIASES.get(token) is None and (len(token) > 1 or token.isdigit())
                )
            )
            if not tokens or tokens in seen:
                continue
            seen.add(tokens)
            token_sets.append(tokens)
            if len(token_sets) >= 4:
                break
        return token_sets

    def _folder_query_token_sets(self, normalized_query: str) -> list[tuple[str, ...]]:
        tokens = tuple(token for token in normalized_query.split() if token and (len(token) > 1 or token.isdigit()))
        return [tokens] if tokens else []

    def _query_indexed_paths(
        self,
        *,
        token_sets: list[tuple[str, ...]],
        search_roots: list[Path],
        desired_extension: str | None,
        item_kind: str,
        limit: int,
    ) -> list[Path]:
        if not self._available or not token_sets or not search_roots:
            return []

        scope_values = tuple(str(path) for path in search_roots if path.exists())
        if not scope_values:
            return []
        cache_key = (
            json.dumps(token_sets, sort_keys=True),
            scope_values,
            desired_extension,
            item_kind,
        )
        now = time.monotonic()
        cached = self._indexed_search_cache.get(cache_key)
        if cached is not None and now - cached[0] <= 20.0:
            return list(cached[1])

        payload = {
            "token_sets": [list(tokens) for tokens in token_sets],
            "scopes": list(scope_values),
            "extension": desired_extension or "",
            "item_kind": item_kind,
            "limit": int(max(8, min(120, limit))),
        }
        script = r"""
$ErrorActionPreference = 'Stop'
$payload = $args[0] | ConvertFrom-Json
function SqlLiteral([string]$value) {
    if ($null -eq $value) {
        return ''
    }
    return ([string]$value).Replace("'", "''")
}
function LikeLiteral([string]$value) {
    $escaped = SqlLiteral($value)
    $escaped = $escaped.Replace('[', '[[]').Replace('%', '[%]').Replace('_', '[_]')
    return $escaped
}
$rows = New-Object System.Collections.ArrayList
$connection = $null
try {
    $connection = New-Object -ComObject ADODB.Connection
    $connection.Open("Provider=Search.CollatorDSO;Extended Properties='Application=Windows';")
    foreach ($tokenSet in @($payload.token_sets)) {
        if ($null -eq $tokenSet -or @($tokenSet).Count -eq 0) {
            continue
        }
        $conditions = New-Object System.Collections.Generic.List[string]
        $scopeConditions = New-Object System.Collections.Generic.List[string]
        foreach ($scope in @($payload.scopes)) {
            if ([string]::IsNullOrWhiteSpace($scope)) {
                continue
            }
            $normalizedScope = (SqlLiteral([string]$scope)).Replace('\', '/')
            $scopeConditions.Add("(SCOPE='file:$normalizedScope')")
        }
        if ($scopeConditions.Count -gt 0) {
            $conditions.Add("(" + ($scopeConditions -join " OR ") + ")")
        }
        foreach ($token in @($tokenSet)) {
            if ([string]::IsNullOrWhiteSpace($token)) {
                continue
            }
            $conditions.Add("System.FileName LIKE '%" + (LikeLiteral([string]$token)) + "%'")
        }
        if (-not [string]::IsNullOrWhiteSpace([string]$payload.extension)) {
            $conditions.Add("System.FileExtension = '" + (SqlLiteral([string]$payload.extension)) + "'")
        }
        $sql = "SELECT TOP " + [int]$payload.limit + " System.ItemPathDisplay, System.DateModified FROM SYSTEMINDEX"
        if ($conditions.Count -gt 0) {
            $sql += " WHERE " + ($conditions -join " AND ")
        }
        $sql += " ORDER BY System.DateModified DESC"
        $recordset = $connection.Execute($sql)
        try {
            while (-not $recordset.EOF) {
                $pathValue = [string]$recordset.Fields.Item('System.ItemPathDisplay').Value
                $modifiedValue = [string]$recordset.Fields.Item('System.DateModified').Value
                [void]$rows.Add([PSCustomObject]@{
                    path = $pathValue
                    modified = $modifiedValue
                })
                $recordset.MoveNext()
            }
        } finally {
            if ($null -ne $recordset) {
                $recordset.Close()
            }
        }
    }
    $rows | ConvertTo-Json -Compress
} finally {
    if ($null -ne $connection) {
        $connection.Close()
    }
}
"""
        try:
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    script,
                    json.dumps(payload),
                ],
                capture_output=True,
                text=True,
                timeout=8.0,
                check=False,
                **self._subprocess_hidden_kwargs(),
            )
        except Exception:
            self._indexed_search_supported = False
            return []
        stdout_text = completed.stdout.strip()
        if completed.returncode != 0 or not stdout_text:
            self._indexed_search_supported = False
            return []
        try:
            parsed = json.loads(stdout_text)
        except json.JSONDecodeError:
            self._indexed_search_supported = False
            return []

        self._indexed_search_supported = True
        if isinstance(parsed, dict):
            parsed_rows = [parsed]
        elif isinstance(parsed, list):
            parsed_rows = [item for item in parsed if isinstance(item, dict)]
        else:
            parsed_rows = []

        seen_paths: set[str] = set()
        candidates: list[Path] = []
        for row in parsed_rows:
            raw_path = str(row.get("path", "") or "").strip()
            if not raw_path:
                continue
            normalized = raw_path.lower()
            if normalized in seen_paths:
                continue
            path = Path(raw_path)
            try:
                if item_kind == "file" and not path.is_file():
                    continue
                if item_kind == "folder" and not path.is_dir():
                    continue
            except Exception:
                continue
            seen_paths.add(normalized)
            candidates.append(path)
            if len(candidates) >= limit:
                break

        self._indexed_search_cache[cache_key] = (now, list(candidates))
        return candidates

    def _scan_file_candidates(
        self,
        *,
        search_roots: list[Path],
        variants: list[str],
        desired_extension: str | None,
        preferred_roots: list[Path],
        folder_hints: list[str],
        ordered_folder_hints: list[str],
    ) -> list[tuple[float, float, Path]]:
        scored: list[tuple[float, float, Path]] = []
        for root in search_roots:
            for path in self._walk_search_candidates(root, want_files=True, want_dirs=False, max_items=1500):
                scored.extend(
                    self._score_file_candidate(
                        path,
                        variants=variants,
                        desired_extension=desired_extension,
                        preferred_roots=preferred_roots,
                        folder_hints=folder_hints,
                        ordered_folder_hints=ordered_folder_hints,
                    )
                )
        return scored

    def _scan_folder_candidates(self, search_roots: list[Path]) -> list[Path]:
        results: list[Path] = []
        seen: set[str] = set()
        for root in search_roots:
            for path in self._walk_search_candidates(root, want_files=False, want_dirs=True, max_items=900):
                key = str(path).lower()
                if key in seen:
                    continue
                seen.add(key)
                results.append(path)
        return results

    def _walk_search_candidates(
        self,
        root: Path,
        *,
        want_files: bool,
        want_dirs: bool,
        max_items: int,
    ) -> list[Path]:
        results: list[Path] = []
        if not root.exists():
            return results
        # A drive root (C:\) is enormous: scan it shallowly (depth 2) and
        # skip the big OS/app trees, so top-level folders are found fast
        # without descending into Windows/Program Files. Home folders keep
        # the full depth-5 walk.
        is_drive_root = root.parent == root
        max_depth = 2 if is_drive_root else 5
        try:
            root_depth = len(root.relative_to(root.anchor).parts)
        except Exception:
            root_depth = len(root.parts)
        for current_root, dirnames, filenames in os.walk(root, topdown=True):
            current_path = Path(current_root)
            try:
                current_depth = len(current_path.relative_to(root).parts)
            except Exception:
                current_depth = max(0, len(current_path.parts) - root_depth)
            # Prune giant/system + hidden ($...) dirs from the descent
            # (cheap everywhere; essential at a drive root).
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in _DRIVE_SCAN_SKIP_DIRS and not d.startswith("$")
            ]
            dirnames.sort(key=lambda name: (len(name), name.lower()))
            if current_depth >= max_depth:
                dirnames[:] = []
            elif len(dirnames) > 40:
                del dirnames[40:]
            # At a drive root, capture ALL (pruned) top-level folders up
            # front so a DFS budget can't exhaust before reaching them
            # (dedup downstream handles the re-visit). This guarantees e.g.
            # C:\HGR App v1.0.0 is a candidate.
            if want_dirs and is_drive_root and current_path == root:
                for d in dirnames:
                    results.append(root / d)
                    if len(results) >= max_items:
                        break
            if want_dirs and current_path != root:
                results.append(current_path)
                if len(results) >= max_items:
                    break
            if want_files:
                for filename in filenames[:120]:
                    results.append(current_path / filename)
                    if len(results) >= max_items:
                        break
            if len(results) >= max_items:
                break
        return results

    def _score_file_candidate(
        self,
        path: Path,
        *,
        variants: list[str],
        desired_extension: str | None,
        preferred_roots: list[Path],
        folder_hints: list[str],
        ordered_folder_hints: list[str],
    ) -> list[tuple[float, float, Path]]:
        if desired_extension is not None and path.suffix.lower() != desired_extension:
            return []
        name_lower = path.name.lower()
        if name_lower in {"hgr app.exe", "hgr_app.exe", "hgr app.lnk", "hgr_app.lnk"}:
            return []
        candidate_name = self._normalize_file_token_text(path.name)
        candidate_stem = self._normalize_file_token_text(path.stem)
        candidate_parent = self._normalize_file_token_text(str(path.parent))
        candidate_numbers = set(re.findall(r"\d+", candidate_name)) | set(re.findall(r"\d+", candidate_parent))
        try:
            modified_time = float(path.stat().st_mtime)
        except Exception:
            modified_time = 0.0

        best_variant_score = 0.0
        for variant in variants:
            tokens = []
            for token in variant.split():
                if FILE_EXTENSION_ALIASES.get(token) is not None:
                    continue
                tokens.append(token)
            core_query = " ".join(tokens).strip() or variant
            query_numbers = set(re.findall(r"\d+", core_query))
            query_tokens = [token for token in core_query.split() if token]
            score = max(
                self._application_match_score(core_query, candidate_name),
                self._application_match_score(core_query, candidate_stem),
            )
            if desired_extension is not None and path.suffix.lower() == desired_extension:
                score += 0.12
            if core_query and core_query in candidate_name:
                score += 0.14
            if core_query and core_query == candidate_stem:
                score += 0.28
            elif core_query and candidate_stem.startswith(core_query):
                score += 0.14
            if query_tokens:
                stem_tokens = set(candidate_stem.split())
                shared = len(stem_tokens & set(query_tokens))
                score += min(0.20, shared * 0.04)
                if len(query_tokens) >= 2 and set(query_tokens) <= stem_tokens:
                    score += 0.10
            if query_numbers:
                if query_numbers <= candidate_numbers:
                    score += 0.26
                else:
                    score -= 0.24
            if score > best_variant_score:
                best_variant_score = score

        score = best_variant_score
        if folder_hints:
            parent_names = {self._normalize_file_token_text(part) for part in path.parent.parts}
            if any(hint in parent_names for hint in folder_hints):
                score += 0.32
            elif any(hint in candidate_parent for hint in folder_hints):
                score += 0.22
        if ordered_folder_hints:
            ordered_parts = [self._normalize_file_token_text(part) for part in path.parent.parts]
            cursor = 0
            matched_order = 0
            for hint in ordered_folder_hints:
                for index in range(cursor, len(ordered_parts)):
                    if ordered_parts[index] == hint or hint in ordered_parts[index]:
                        matched_order += 1
                        cursor = index + 1
                        break
            if matched_order:
                score += min(0.24, matched_order * 0.08)
        if preferred_roots and any(str(path).lower().startswith(str(root).lower()) for root in preferred_roots):
            score += 0.10
        if score < 0.58:
            return []
        return [(score, modified_time, path)]

    def _generate_file_query_variants(self, query: str) -> list[str]:
        base = self._normalize_file_query(query)
        if not base:
            return []
        variants: set[str] = {base}
        tokens = base.split()

        # replace common long forms with abbreviations
        for index, token in enumerate(tokens):
            for short in FILE_QUERY_ABBREVIATIONS.get(token, ()):
                repl = list(tokens)
                repl[index] = short
                variants.add(" ".join(repl).strip())

        # expand common spoken homophones like two/too/to/2
        for index, token in enumerate(tokens):
            for alt in FILE_QUERY_HOMOPHONES.get(token, ()):
                repl = list(tokens)
                repl[index] = alt
                variants.add(" ".join(repl).strip())

        # merge common alnum pairs like "cs 579" -> "cs579" and "hw 2" -> "hw2"
        merged: set[str] = set()
        for value in list(variants):
            t = value.split()
            out: list[str] = []
            i = 0
            while i < len(t):
                if i + 1 < len(t) and t[i].isalpha() and t[i + 1].isdigit() and len(t[i]) <= 8:
                    out.append(f"{t[i]}{t[i + 1]}")
                    i += 2
                    continue
                out.append(t[i])
                i += 1
            merged.add(" ".join(out).strip())
        variants |= merged
        return [variant for variant in variants if variant]

    def _normalize_file_token_text(self, text: str) -> str:
        normalized = str(text or "")
        normalized = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", normalized)
        normalized = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", normalized)
        normalized = normalized.lower()
        normalized = re.sub(r'\b(hgr)([a-z])', r'\1 \2', normalized)
        replacements = (
            ("dot ", ""),
            ("you tube", "youtube"),
            ("key card", "kicad"),
            ("key cards", "kicad"),
            ("key cad", "kicad"),
            ("ki cad", "kicad"),
        )
        for source, target in replacements:
            normalized = normalized.replace(source, target)
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _canonical_file_homophone_text(self, text: str) -> str:
        """Canonicalize a file/folder name for ambiguity comparison.

        Maps every spoken-form number to its digit equivalent (zero-
        ten + roman numerals I-X), plus the "two/too/to" homophone
        cluster. Both sides of an ambiguity check go through this
        same pipeline so a user's spoken "report three" canonicalizes
        to "report 3" and matches the on-disk "Report 3.pdf"; the
        previous narrow mapping (only 2) left every other numbered
        sequel falling through to fuzzy scoring."""
        normalized = self._normalize_file_token_text(text)
        if not normalized:
            return ""
        raw_tokens = [token for token in normalized.split() if token]
        total_tokens = len(raw_tokens)
        tokens: list[str] = []
        for token in raw_tokens:
            mapped = APPLICATION_NUMBER_WORDS.get(token)
            if mapped is None and token in {"too", "to"}:
                mapped = "2"
            if mapped is None and total_tokens > 1:
                mapped = APPLICATION_ROMAN_NUMERALS.get(token)
            tokens.append(mapped or token)
        return " ".join(tokens).strip()

    def _normalize_file_query(self, text: str) -> str:
        normalized = self._normalize_file_token_text(text)
        normalized = f" {normalized} ".replace(" homework ", " hw ").strip()
        normalized = re.sub(r"\b(?:located|saved|stored) in\b.*$", " ", normalized)
        normalized = re.sub(r"\b(?:inside|under|within)\b.*$", " ", normalized)
        normalized = re.sub(r"\bin\s+[a-z0-9 ]+\s+folder\b.*$", " ", normalized)
        normalized = re.sub(r"^\s*(?:can you|could you|would you|will you|please|hey hgr|hey app)\s+", " ", normalized)
        tokens = [
            token for token in normalized.split()
            if token not in {
                "file", "files", "folder", "document", "documents", "open", "show", "me", "pull", "up", "my", "named", "called", "the",
                "located", "in", "inside", "under", "within", "from", "at", "path", "directory", "please",
                "can", "you", "could", "would", "will", "hey", "app", "dot",
            }
        ]
        return " ".join(tokens).strip()

    def _file_search_preferences(
        self,
        query: str,
        *,
        preferred_root: str | None = None,
        folder_hint: str | None = None,
    ) -> tuple[list[Path], list[str], list[str]]:
        normalized = self._normalize_application_name(query)
        preferred_roots: list[Path] = []
        folder_hints: list[str] = []
        ordered_folder_hints: list[str] = []

        if preferred_root:
            root_path = self._known_folder_path(preferred_root)
            if root_path is not None and root_path not in preferred_roots:
                preferred_roots.append(root_path)

        if folder_hint:
            hint = self._normalize_application_name(folder_hint)
            if hint and hint not in folder_hints:
                folder_hints.append(hint)

        for key in ("desktop", "documents", "downloads", "music", "pictures", "videos", "onedrive"):
            path = self._known_folder_path(key)
            if path is not None and re.search(rf"\b{re.escape(key)}\b", normalized):
                preferred_roots.append(path)

        for match in re.finditer(r"\b([a-z0-9][a-z0-9 ]{1,80}?)\s+folder\b", normalized):
            hint = self._normalize_application_name(match.group(1))
            if hint and hint not in {"the", "my"} and hint not in folder_hints:
                folder_hints.append(hint)

        # Also treat patterns like "in CS 579" as a possible folder hint when paired with a root.
        for match in re.finditer(r"\b(?:in|inside|under|within)\s+([a-z0-9][a-z0-9 ]{1,80}?)\b", normalized):
            hint = self._normalize_application_name(match.group(1))
            if hint and hint not in {"documents", "downloads", "desktop", "music", "pictures", "videos", "onedrive"}:
                if len(hint.split()) <= 5 and hint not in folder_hints:
                    folder_hints.append(hint)

        ordered_segments: list[str] = []
        for match in re.finditer(
            r"\b(?:in|inside|under|within)\s+([a-z0-9][a-z0-9 ]{0,80}?)(?=\s+(?:in|inside|under|within)\b|$)",
            normalized,
        ):
            hint = self._normalize_application_name(match.group(1))
            if not hint:
                continue
            hint = re.sub(r"\s+folder$", "", hint).strip()
            if hint:
                ordered_segments.append(hint)

        known_root_names = {"desktop", "documents", "downloads", "music", "pictures", "videos", "onedrive"}
        root_candidate: str | None = None
        for segment in ordered_segments:
            if segment in known_root_names:
                root_candidate = segment
        if root_candidate is not None:
            root_path = self._known_folder_path(root_candidate)
            if root_path is not None and root_path not in preferred_roots:
                preferred_roots.append(root_path)

        for segment in reversed(ordered_segments):
            if segment in known_root_names:
                continue
            if segment not in ordered_folder_hints:
                ordered_folder_hints.append(segment)
            if segment not in folder_hints:
                folder_hints.append(segment)

        return preferred_roots, folder_hints, ordered_folder_hints

    def _file_search_roots(
        self,
        preferred_roots: list[Path] | None = None,
        folder_hints: list[str] | None = None,
        ordered_folder_hints: list[str] | None = None,
    ) -> list[Path]:
        roots: list[Path] = []
        seen: set[str] = set()
        preferred_roots = preferred_roots or []
        folder_hints = folder_hints or []
        ordered_folder_hints = ordered_folder_hints or []

        def add_root(path: Path) -> None:
            key = str(path).lower()
            if key in seen or not path.exists():
                return
            seen.add(key)
            roots.append(path)

        for path in preferred_roots:
            add_root(path)
            current_candidates = [path]
            for hint in ordered_folder_hints:
                next_candidates: list[Path] = []
                for candidate_root in current_candidates[:4]:
                    try:
                        children = list(candidate_root.iterdir())
                    except Exception:
                        continue
                    matches: list[tuple[int, Path]] = []
                    for child in children:
                        if not child.is_dir():
                            continue
                        normalized_child = self._normalize_application_name(child.name)
                        if normalized_child == hint:
                            matches.append((2, child))
                        elif hint in normalized_child:
                            matches.append((1, child))
                    matches.sort(key=lambda item: (item[0], -len(item[1].name)), reverse=True)
                    next_candidates.extend(path for _score, path in matches[:3])
                if not next_candidates:
                    break
                for candidate in next_candidates:
                    add_root(candidate)
                current_candidates = next_candidates
            for hint in folder_hints:
                candidate = path / hint
                if candidate.exists():
                    add_root(candidate)
                else:
                    try:
                        for child in path.iterdir():
                            if child.is_dir() and self._normalize_application_name(child.name) == hint:
                                add_root(child)
                            elif child.is_dir() and hint in self._normalize_application_name(child.name):
                                add_root(child)
                    except Exception:
                        pass

        home = Path.home()
        for name in FILE_SEARCH_ROOT_NAMES:
            path = home / name
            add_root(path)
        if self._mac:
            # macOS: video saves live in ~/Movies (FILE_SEARCH_ROOT_NAMES uses
            # the Windows "Videos", which doesn't exist here), and Touchless
            # historically saved clips directly in the home dir. Include both so
            # the fast search finds ALL clips, plus /Applications for app files.
            add_root(home / "Movies")
            add_root(home)
            add_root(Path("/Applications"))
        # Also include fixed-drive roots (C:\, D:\, ...) so top-level
        # folders outside the home tree (e.g. C:\HGR App v1.0.0) are
        # findable. These are scanned shallowly + system-dir-pruned in
        # _walk_search_candidates, and the index path scopes to them too.
        for drive in self._fixed_drive_roots():
            add_root(drive)
        return roots

    def _fixed_drive_roots(self) -> list[Path]:
        """Existing drive roots (C:..Z:). Skips A:/B: to avoid floppy stalls."""
        import string
        roots: list[Path] = []
        for letter in string.ascii_uppercase[2:]:  # C..Z
            p = Path(f"{letter}:\\")
            try:
                if p.exists():
                    roots.append(p)
            except Exception:
                continue
        return roots

    def _classic_outlook_path(self) -> Path | None:
        for path in self._outlook_paths:
            if path.exists() and path.name.lower() == "outlook.exe":
                return path
        resolved = self._resolve_application("outlook")
        if resolved is None:
            return None
        target_path = Path(str(resolved.target))
        if target_path.name.lower() == "outlook.exe" and target_path.exists():
            return target_path
        return None

    def _resolve_application(self, app_name: str) -> DesktopAppEntry | None:
        query = self._normalize_application_query(app_name)
        if not query:
            return None

        explicit = self._resolve_known_application(query)
        if explicit is not None:
            return explicit

        for entry in self._quick_application_catalog():
            if self._entry_exact_query_match(entry, query):
                return entry
        for entry in self._application_catalog():
            if self._entry_exact_query_match(entry, query):
                return entry

        best_entry: DesktopAppEntry | None = None
        best_score = 0.0
        for entry in self._quick_application_catalog():
            score = self._application_match_score(query, entry.normalized_name)
            for alias in entry.aliases:
                score = max(score, self._application_match_score(query, alias))
            if score > best_score:
                best_entry = entry
                best_score = score
        if best_entry is not None and best_score >= 0.70:
            return best_entry
        for entry in self._application_catalog():
            score = self._application_match_score(query, entry.normalized_name)
            for alias in entry.aliases:
                score = max(score, self._application_match_score(query, alias))
            if score > best_score:
                best_entry = entry
                best_score = score
        if best_entry is None or best_score < 0.70:
            # Stem-subset fallback. When the canonicalised query
            # tokens are a strict subset of any catalog entry's
            # normalised-name tokens, accept that entry as a valid
            # resolution. This guarantees `can_resolve_application`
            # returns True for queries like "fallout three" (which
            # canonicalises to "fallout 3" and is a subset of "Fallout
            # 3 - Game of the Year Edition" tokens), preventing the
            # caller's file-request fallback from picking up some
            # unrelated "draft3.raw" file. Cheap to compute — same
            # canonical-form helper the disambiguation path uses.
            canonical_query = self._exact_match_canonical_form(query)
            stem_tokens = set(canonical_query.split()) if canonical_query else set()
            if 1 <= len(stem_tokens) <= 3:
                for catalog in (
                    self._quick_application_catalog(),
                    self._application_catalog(),
                ):
                    for entry in catalog:
                        cand_canonical = self._exact_match_canonical_form(entry.normalized_name)
                        cand_tokens = set(cand_canonical.split()) if cand_canonical else set()
                        if cand_tokens and stem_tokens <= cand_tokens:
                            return entry
            return None
        return best_entry

    def _resolve_known_application(self, query: str) -> DesktopAppEntry | None:
        for display_name, config in self.KNOWN_APPLICATIONS.items():
            aliases = config["aliases"]
            if not any(self._application_match_score(query, self._normalize_application_name(alias)) >= 0.82 for alias in aliases):
                continue
            for target in config["targets"]:
                if self._target_exists_or_is_launchable(target):
                    return DesktopAppEntry(
                        display_name=display_name,
                        normalized_name=self._normalize_application_name(display_name),
                        target=str(target),
                        source="known",
                        aliases=self._build_entry_aliases(display_name, aliases),
                        category=str(config.get("category", "generic")),
                    )
        return None

    def _application_catalog(self) -> list[DesktopAppEntry]:
        if self._app_catalog is not None:
            return self._app_catalog
        if self._mac:
            self._app_catalog = self._build_mac_catalog()
            return self._app_catalog
        shared = type(self)._shared_app_catalog
        if shared is not None:
            self._app_catalog = list(shared)
            return self._app_catalog
        self._ensure_background_catalog_build()
        return list(self._quick_application_catalog())

    def _ensure_background_catalog_build(self) -> None:
        cls = type(self)
        with cls._shared_app_catalog_build_lock:
            if cls._shared_app_catalog is not None:
                return
            existing = cls._shared_app_catalog_build_thread
            if existing is not None and existing.is_alive():
                return
            thread = threading.Thread(
                target=self._build_full_catalog_worker,
                name="hgr-desktop-catalog-build",
                daemon=True,
            )
            cls._shared_app_catalog_build_thread = thread
            thread.start()

    def _build_full_catalog_worker(self) -> None:
        entries: dict[str, DesktopAppEntry] = {}
        iterators = (
            self._iter_known_applications,
            self._iter_start_apps_entries,
            self._iter_start_menu_entries,
            self._iter_registry_app_paths,
            self._iter_uninstall_entries,
        )
        for factory in iterators:
            try:
                produced = factory()
            except Exception:
                continue
            for entry in produced:
                self._store_entry(entries, entry)
        catalog = sorted(entries.values(), key=lambda item: (item.normalized_name, item.display_name.lower()))
        type(self)._shared_app_catalog = catalog

    def _quick_application_catalog(self) -> list[DesktopAppEntry]:
        if self._quick_app_catalog is not None:
            return self._quick_app_catalog
        if self._mac:
            self._quick_app_catalog = self._build_mac_catalog()
            return self._quick_app_catalog
        entries: dict[str, DesktopAppEntry] = {}
        for iterator in (
            self._iter_known_applications(),
            self._iter_start_apps_entries(),
            self._iter_registry_app_paths(),
        ):
            for entry in iterator:
                self._store_entry(entries, entry)
        self._quick_app_catalog = sorted(entries.values(), key=lambda item: (item.normalized_name, item.display_name.lower()))
        return self._quick_app_catalog

    @staticmethod
    def _entry_is_running_hgr(entry: DesktopAppEntry) -> bool:
        target = str(getattr(entry, "target", "") or "").strip().lower()
        display = str(getattr(entry, "display_name", "") or "").strip().lower()
        normalized = str(getattr(entry, "normalized_name", "") or "").strip().lower()
        if target:
            base = target.replace("\\", "/").rsplit("/", 1)[-1]
            if base in {"hgr app.exe", "hgr_app.exe", "hgr app.lnk", "hgr_app.lnk"}:
                return True
            if target.endswith("hgr app.exe") or target.endswith("hgr_app.exe"):
                return True
            if target.endswith("hgr app.lnk") or target.endswith("hgr_app.lnk"):
                return True
        if display in {"hgr app", "hgr", "hgr application"}:
            return True
        if normalized in {"hgr app", "hgr", "hgr application"}:
            return True
        return False

    def _store_entry(self, entries: dict[str, DesktopAppEntry], entry: DesktopAppEntry) -> None:
        if not entry.normalized_name or not entry.target:
            return
        if entry.normalized_name.startswith("uninstall "):
            return
        if self._entry_is_running_hgr(entry):
            return
        existing = entries.get(entry.normalized_name)
        if existing is None:
            entries[entry.normalized_name] = entry
            return

        def quality(candidate: DesktopAppEntry) -> tuple[int, int, int]:
            target = str(candidate.target).lower()
            if target.startswith(("http://", "https://")) or target.endswith(".url"):
                target_rank = 0
            elif target.endswith(".lnk") or target.endswith(".appref-ms"):
                target_rank = 1
            elif target.startswith("shell:appsfolder"):
                target_rank = 2
            else:
                target_rank = 3
            return (
                self.SOURCE_PRIORITY.get(candidate.source, 0),
                target_rank,
                len(candidate.aliases),
            )

        if quality(entry) > quality(existing):
            entries[entry.normalized_name] = entry

    def _iter_known_applications(self) -> list[DesktopAppEntry]:
        entries: list[DesktopAppEntry] = []
        for display_name, config in self.KNOWN_APPLICATIONS.items():
            launch_target = next(
                (str(target) for target in config["targets"] if self._target_exists_or_is_launchable(str(target))),
                None,
            )
            if launch_target is None:
                continue
            entries.append(
                DesktopAppEntry(
                    display_name=display_name,
                    normalized_name=self._normalize_application_name(display_name),
                    target=launch_target,
                    source="known",
                    aliases=self._build_entry_aliases(display_name, config.get("aliases", ())),
                    category=str(config.get("category", "generic")),
                )
            )
        return entries

    def _iter_start_apps_entries(self) -> list[DesktopAppEntry]:
        if not self._available:
            return []
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress",
        ]
        try:
            raw = subprocess.check_output(
                command,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
                **self._subprocess_hidden_kwargs(),
            )
        except Exception:
            return []
        raw = raw.strip()
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except Exception:
            return []
        items = payload if isinstance(payload, list) else [payload]
        entries: list[DesktopAppEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("Name") or "").strip()
            app_id = str(item.get("AppID") or "").strip()
            if not name or not app_id:
                continue
            target = f"shell:AppsFolder\\{app_id}"
            entries.append(
                DesktopAppEntry(
                    display_name=name,
                    normalized_name=self._normalize_application_name(name),
                    target=target,
                    source="start_apps",
                    aliases=self._build_entry_aliases(name),
                    category=self._infer_category(name),
                )
            )
        return entries

    def _iter_start_menu_entries(self) -> list[DesktopAppEntry]:
        roots = (
            Path(os.getenv("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path(os.getenv("ProgramData", "C:\\ProgramData")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path.home() / "Desktop",
            Path("C:\\Users\\Public\\Desktop"),
        )
        entries: list[DesktopAppEntry] = []
        for root in roots:
            if not root.exists():
                continue
            for pattern in ("*.lnk", "*.appref-ms", "*.url", "*.exe"):
                try:
                    matches = list(root.rglob(pattern))
                except Exception:
                    continue
                for path in matches:
                    name = path.stem.strip()
                    normalized = self._normalize_application_name(name)
                    if not normalized:
                        continue
                    entries.append(
                        DesktopAppEntry(
                            display_name=name,
                            normalized_name=normalized,
                            target=str(path),
                            source="start_menu",
                            aliases=self._build_entry_aliases(name),
                            category=self._infer_category(name),
                        )
                    )
        return entries

    def _iter_registry_app_paths(self) -> list[DesktopAppEntry]:
        if winreg is None:
            return []
        hives = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
        subkeys = (
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths",
        )
        entries: list[DesktopAppEntry] = []
        for hive in hives:
            for subkey in subkeys:
                try:
                    with winreg.OpenKey(hive, subkey) as key:
                        index = 0
                        while True:
                            try:
                                child_name = winreg.EnumKey(key, index)
                            except OSError:
                                break
                            index += 1
                            try:
                                with winreg.OpenKey(key, child_name) as child:
                                    value, _ = winreg.QueryValueEx(child, "")
                            except OSError:
                                continue
                            normalized = self._normalize_application_name(Path(child_name).stem)
                            if not normalized:
                                continue
                            entries.append(
                                DesktopAppEntry(
                                    display_name=Path(child_name).stem,
                                    normalized_name=normalized,
                                    target=str(value),
                                    source="app_paths",
                                    aliases=self._build_entry_aliases(Path(child_name).stem),
                                    category=self._infer_category(child_name),
                                )
                            )
                except OSError:
                    continue
        return entries

    def _iter_uninstall_entries(self) -> list[DesktopAppEntry]:
        if winreg is None:
            return []
        hives = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
        subkeys = (
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        )
        entries: list[DesktopAppEntry] = []
        for hive in hives:
            for subkey in subkeys:
                try:
                    with winreg.OpenKey(hive, subkey) as key:
                        index = 0
                        while True:
                            try:
                                child_name = winreg.EnumKey(key, index)
                            except OSError:
                                break
                            index += 1
                            try:
                                with winreg.OpenKey(key, child_name) as child:
                                    display_name = self._reg_query(child, "DisplayName")
                                    display_icon = self._reg_query(child, "DisplayIcon")
                                    install_location = self._reg_query(child, "InstallLocation")
                                target = self._guess_uninstall_launch_target(display_icon, install_location, display_name)
                            except OSError:
                                continue
                            if not display_name or not target:
                                continue
                            entries.append(
                                DesktopAppEntry(
                                    display_name=display_name,
                                    normalized_name=self._normalize_application_name(display_name),
                                    target=target,
                                    source="uninstall",
                                    aliases=self._build_entry_aliases(display_name),
                                    category=self._infer_category(display_name),
                                )
                            )
                except OSError:
                    continue
        return entries

    def _reg_query(self, key: object, value_name: str) -> str | None:
        try:
            value, _ = winreg.QueryValueEx(key, value_name)
        except OSError:
            return None
        text = str(value).strip()
        return text or None

    def _guess_uninstall_launch_target(self, display_icon: str | None, install_location: str | None, display_name: str | None) -> str | None:
        candidate = self._parse_display_icon_target(display_icon)
        if candidate is not None and self._target_exists_or_is_launchable(candidate):
            return candidate
        location = Path(str(install_location or "")).expanduser()
        if location.exists() and location.is_dir():
            guessed = self._find_best_executable_in_directory(location, display_name or "")
            if guessed is not None:
                return str(guessed)
        return None

    def _parse_display_icon_target(self, value: str | None) -> str | None:
        text = str(value or "").strip().strip('"')
        if not text:
            return None
        text = re.sub(r",\s*-?\d+$", "", text).strip().strip('"')
        return text or None

    def _find_best_executable_in_directory(self, directory: Path, display_name: str) -> Path | None:
        normalized_display = self._normalize_application_name(display_name)
        best_path: Path | None = None
        best_score = 0.0
        try:
            candidates = list(directory.glob("*.exe"))
            candidates.extend(directory.glob("**/*.exe"))
        except Exception:
            return None
        for path in candidates[:120]:
            score = self._application_match_score(normalized_display, self._normalize_application_name(path.stem))
            if score > best_score:
                best_score = score
                best_path = path
        if best_path is None or best_score < 0.64:
            return None
        return best_path

    def _build_entry_aliases(self, display_name: str, extra_aliases: tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
        aliases: list[str] = []
        for item in (display_name, *(extra_aliases or ())):
            for variant in self._raw_alias_variants(str(item or "")):
                normalized = self._normalize_application_name(variant)
                if normalized:
                    aliases.append(normalized)
                    aliases.extend(self._derived_aliases(normalized))
        return tuple(
            dict.fromkeys(
                alias for alias in aliases
                if alias and self._is_helpful_alias(alias)
            )
        )

    def _raw_alias_variants(self, text: str) -> tuple[str, ...]:
        value = str(text or "").strip()
        if not value:
            return ()
        spaced_camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
        spaced_camel = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced_camel)
        collapsed = re.sub(r"[^A-Za-z0-9]+", "", value)
        variants = [value, spaced_camel]
        if collapsed:
            variants.append(collapsed)
        return tuple(dict.fromkeys(item for item in variants if item))

    def _derived_aliases(self, normalized_name: str) -> list[str]:
        aliases: list[str] = []
        tokens = [token for token in normalized_name.split() if token]
        if not tokens:
            return aliases

        joined = "".join(tokens)
        if joined and joined != normalized_name:
            aliases.append(joined)
        if len(tokens) >= 2:
            aliases.append("".join(tokens[:2]) + (" " + " ".join(tokens[2:]) if len(tokens) > 2 else ""))
            aliases.append(" ".join(tokens[:-1]) + tokens[-1])
        cleaned = [token for token in tokens if token not in DYNAMIC_ALIAS_STOP_WORDS and not re.fullmatch(r"v?\d+(?:\.\d+)?", token)]
        if cleaned and cleaned != tokens:
            aliases.append(" ".join(cleaned))
            cleaned_joined = "".join(cleaned)
            if cleaned_joined:
                aliases.append(cleaned_joined)
            if len(cleaned) >= 2:
                aliases.append("".join(cleaned[:2]) + (" " + " ".join(cleaned[2:]) if len(cleaned) > 2 else ""))
        while cleaned and cleaned[0] in VENDOR_PREFIXES:
            cleaned.pop(0)
        if cleaned:
            aliases.append(" ".join(cleaned))
            aliases.append("".join(cleaned))
        if len(cleaned) >= 2:
            aliases.append(" ".join(cleaned[-2:]))
            aliases.append("".join(cleaned[-2:]))
        if len(cleaned) >= 1 and cleaned[-1] not in DYNAMIC_ALIAS_STOP_WORDS and len(cleaned[-1]) >= 4:
            aliases.append(cleaned[-1])
        return [alias for alias in dict.fromkeys(alias for alias in aliases if alias and alias != normalized_name)]

    def _is_helpful_alias(self, alias: str) -> bool:
        tokens = [token for token in str(alias or "").split() if token]
        if not tokens:
            return False
        if all(token in DYNAMIC_ALIAS_STOP_WORDS for token in tokens):
            return False
        if len(tokens) == 1 and tokens[0] in DYNAMIC_ALIAS_STOP_WORDS:
            return False
        compact = "".join(tokens)
        if len(compact) <= 2:
            return False
        return True

    def _infer_category(self, name: str) -> str:
        normalized = self._normalize_application_name(name)
        if any(token in normalized for token in ("chrome", "browser", "firefox", "edge", "opera")):
            return "browser"
        if any(token in normalized for token in ("spotify", "music", "itunes", "vlc")):
            return "music"
        if any(token in normalized for token in ("mail", "outlook", "thunderbird")):
            return "email"
        if any(token in normalized for token in ("studio", "code", "pycharm", "intellij", "notepad")):
            return "editor"
        if any(token in normalized for token in ("steam", "game", "fortnite", "genshin", "riot", "epic")):
            return "gaming"
        return "generic"

    def _normalize_application_query(self, text: str) -> str:
        normalized = self._normalize_application_name(text)
        tokens = [token for token in normalized.split() if token not in APP_QUERY_EDGE_WORDS]
        while tokens and tokens[0] in APP_QUERY_EDGE_WORDS:
            tokens.pop(0)
        while tokens and tokens[-1] in APP_QUERY_EDGE_WORDS:
            tokens.pop()
        return " ".join(tokens).strip()

    def _normalize_application_name(self, text: str) -> str:
        normalized = str(text or "")
        normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
        normalized = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", normalized)
        normalized = normalized.lower()
        replacements = (
            ("chat gpt", "chatgpt"),
            ("visual stdios", "visual studio"),
            ("visual studios", "visual studio"),
            ("pc's", "pc"),
            ("pcs", "pc"),
            ("you tube", "youtube"),
            ("key card", "kicad"),
            ("key cards", "kicad"),
            ("key cad", "kicad"),
            ("ki cad", "kicad"),
            ("k i cad", "kicad"),
        )
        for source, target in replacements:
            normalized = normalized.replace(source, target)
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _canonical_application_numeric_text(self, text: str) -> str:
        normalized = self._normalize_application_name(text)
        if not normalized:
            return ""
        raw_tokens = [token for token in normalized.split() if token]
        total_tokens = len(raw_tokens)
        tokens: list[str] = []
        for token in raw_tokens:
            mapped = APPLICATION_NUMBER_WORDS.get(token)
            if mapped is None and total_tokens > 1:
                mapped = APPLICATION_ROMAN_NUMERALS.get(token)
            tokens.append(mapped or token)
        return " ".join(tokens).strip()

    def _exact_match_canonical_form(self, text: str) -> str:
        """Build the canonical form used for exact-match comparison
        between a query and a catalog entry's titles/aliases. Both
        sides go through the SAME pipeline so subtleties like stop-
        words ("the", "a") and number-word vs. digit don't cause
        false negatives. Without the stopword strip, a query like
        "slay the spire two" (which `_normalize_application_query`
        had already stripped to "slay spire two" upstream) would
        fail to match the entry "Slay the Spire 2" (whose stop-
        words were preserved in `_normalize_application_name`)."""
        no_stopwords = self._normalize_application_query(text)
        if not no_stopwords:
            no_stopwords = self._normalize_application_name(text)
        return self._canonical_application_numeric_text(no_stopwords)

    def _entry_exact_query_match(self, entry: DesktopAppEntry, query: str) -> bool:
        query_canonical = self._exact_match_canonical_form(query)
        if not query_canonical:
            return False
        # Only the display name and normalized name count as exact-
        # match candidates. Stem-style aliases like "fallout" for
        # "Fallout 3" must NOT trigger an exact-match short-circuit:
        # a user who said "open fallout" may have meant the bare-stem
        # disambiguation between "Fallout 3" and "Fallout New Vegas",
        # and accepting the stem alias as exact would silently open
        # whichever title happened to land first in the ranking and
        # skip the disambiguation prompt the user expected.
        canonical_targets = {
            self._exact_match_canonical_form(entry.display_name),
            self._exact_match_canonical_form(entry.normalized_name),
        }
        return query_canonical in (target for target in canonical_targets if target)

    def _entry_query_token_superset_match(self, entry: DesktopAppEntry, query: str) -> bool:
        query_tokens = set(self._exact_match_canonical_form(query).split())
        if not query_tokens:
            return False
        candidate_texts = [
            entry.display_name,
            entry.normalized_name,
            *(alias for alias in entry.aliases if alias),
        ]
        for candidate in candidate_texts:
            candidate_tokens = set(self._exact_match_canonical_form(candidate).split())
            if not candidate_tokens:
                continue
            # Strict superset: every query token appears in this
            # candidate. Best-case match.
            if query_tokens <= candidate_tokens:
                return True
            # Whisper-noise tolerance: allow up to one query token
            # that's NOT in the candidate, as long as the overlap
            # contains a distinctive token (≥4 chars, non-numeric).
            # Real-world: the user said "open fallout" but whisper
            # heard "open fallout out". The leftover token "out"
            # isn't in either Fallout title, but the meaningful
            # overlap "fallout" is — so both Fallout 3 and Fallout
            # New Vegas should still surface for disambiguation
            # rather than us silently picking one. The 4-char floor
            # prevents a stray short whisper artifact ("a", "the",
            # "an") from latching onto every catalog entry.
            overlap = query_tokens & candidate_tokens
            leftover = query_tokens - candidate_tokens
            if overlap and len(leftover) <= 1 and any(
                len(token) >= 4 and not token.isdigit() for token in overlap
            ):
                return True
        return False

    def _application_match_score(self, query: str, candidate: str) -> float:
        if not query or not candidate:
            return 0.0
        query = self._normalize_application_name(query)
        candidate = self._normalize_application_name(candidate)
        if query == candidate:
            return 1.0
        canonical_query = self._canonical_application_numeric_text(query)
        canonical_candidate = self._canonical_application_numeric_text(candidate)
        if canonical_query == canonical_candidate and canonical_query:
            return 1.08
        query_tokens = set(query.split())
        candidate_tokens = set(candidate.split())
        canonical_query_tokens = set(canonical_query.split())
        canonical_candidate_tokens = set(canonical_candidate.split())
        overlap = max(
            len(query_tokens & candidate_tokens) / max(len(query_tokens), len(candidate_tokens), 1),
            len(canonical_query_tokens & canonical_candidate_tokens) / max(len(canonical_query_tokens), len(canonical_candidate_tokens), 1),
        )
        ratio = max(
            SequenceMatcher(None, query, candidate).ratio(),
            SequenceMatcher(None, canonical_query, canonical_candidate).ratio(),
        )
        prefix_bonus = 0.12 if (
            candidate.startswith(query)
            or query.startswith(candidate)
            or canonical_candidate.startswith(canonical_query)
            or canonical_query.startswith(canonical_candidate)
        ) else 0.0
        subset_bonus = 0.08 if canonical_query_tokens and canonical_query_tokens <= canonical_candidate_tokens else 0.0
        query_numbers = {token for token in canonical_query_tokens if token.isdigit()}
        candidate_numbers = {token for token in canonical_candidate_tokens if token.isdigit()}
        number_bonus = 0.0
        if query_numbers:
            if query_numbers <= candidate_numbers:
                number_bonus += 0.24
            else:
                number_bonus -= 0.32
        elif candidate_numbers and canonical_query_tokens and canonical_query_tokens <= canonical_candidate_tokens:
            number_bonus -= 0.03
        score = ratio * 0.56 + overlap * 0.44 + prefix_bonus + subset_bonus + number_bonus
        return max(0.0, min(1.18, score))

    def _application_mention_score(
        self,
        normalized_text: str,
        tokens: list[str],
        entry: DesktopAppEntry,
    ) -> tuple[float, str | None]:
        best_score = 0.0
        best_alias: str | None = None
        aliases = entry.aliases or (entry.normalized_name,)
        haystack = f" {normalized_text} "
        canonical_haystack = f" {self._canonical_application_numeric_text(normalized_text)} "
        for alias in aliases:
            if not alias:
                continue
            if f" {alias} " in haystack:
                return 1.08, alias
            canonical_alias = self._canonical_application_numeric_text(alias)
            if canonical_alias and f" {canonical_alias} " in canonical_haystack:
                return 1.10, alias

            alias_tokens = alias.split()
            window_sizes = {
                max(1, len(alias_tokens) - 1),
                len(alias_tokens),
                min(len(tokens), len(alias_tokens) + 1),
            }
            for window_size in sorted(window_sizes):
                for window in self._sliding_windows(tokens, window_size):
                    score = self._application_match_score(alias, window)
                    if alias_tokens and set(alias_tokens).issubset(set(window.split())):
                        score += 0.10
                    if score > best_score:
                        best_score = score
                        best_alias = alias
        if entry.normalized_name and (
            entry.normalized_name in normalized_text
            or self._canonical_application_numeric_text(entry.normalized_name) in self._canonical_application_numeric_text(normalized_text)
        ):
            best_score = max(best_score, 0.96)
            best_alias = best_alias or entry.normalized_name
        return best_score, best_alias

    def _sliding_windows(self, tokens: list[str], size: int) -> list[str]:
        if size <= 0 or not tokens or size > len(tokens):
            return []
        return [" ".join(tokens[index:index + size]) for index in range(len(tokens) - size + 1)]


    def _resolve_url_shortcut(self, path: Path) -> str | None:
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.upper().startswith("URL="):
                    value = line.split("=", 1)[1].strip()
                    return value or None
        except Exception:
            return None
        return None

    def _resolve_windows_shortcut(self, path: Path) -> tuple[str | None, str]:
        if not self._available or path.suffix.lower() != ".lnk":
            return None, ""
        script = (
            "$WshShell = New-Object -ComObject WScript.Shell; "
            f"$Shortcut = $WshShell.CreateShortcut('{str(path).replace("'", "''")}'); "
            "[Console]::WriteLine($Shortcut.TargetPath); "
            "[Console]::WriteLine($Shortcut.Arguments)"
        )
        try:
            output = subprocess.check_output(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=8,
                **self._subprocess_hidden_kwargs(),
            )
        except Exception:
            return None, ""
        lines = output.splitlines()
        target_path = lines[0].strip() if lines else ""
        arguments = lines[1].strip() if len(lines) > 1 else ""
        return (target_path or None), arguments

    def _launch_resolved_shortcut(self, target_path: str, arguments: str = "") -> bool:
        try:
            if arguments:
                subprocess.Popen([target_path, *shlex.split(arguments, posix=False)], shell=False)
            else:
                subprocess.Popen([target_path], shell=False)
            return True
        except Exception:
            pass
        # Fallback via ShellExecuteW (matches Explorer's open behavior)
        # rather than `cmd /c start` which Norton SONAR flags.
        if arguments:
            parsed_args = shlex.split(arguments, posix=False)
            return launch_external(target_path, args=parsed_args)
        return launch_external(target_path)

    def _target_exists_or_is_launchable(self, target: str) -> bool:
        candidate = Path(target)
        if candidate.exists():
            return True
        lowered = str(target).lower()
        if lowered.startswith("hgr:"):
            return True
        if lowered.startswith("shell:appsfolder"):
            return True
        if lowered.endswith(".url") or lowered.endswith(".lnk") or lowered.endswith(".appref-ms"):
            return True
        if target.endswith(":") or "://" in target:
            return True
        return shutil.which(str(target)) is not None

    def _launch_path_or_command(self, target: str) -> bool:
        if self._mac:
            # An existing path (e.g. a .app bundle or a file) -> `open <path>`;
            # otherwise treat it as an app name -> `open -a <name>`.
            if Path(target).exists():
                return self._mac_open(target)
            return self._mac_open(target, as_app=True)
        if Path(target).exists():
            return self._launch_target(target)
        if self._launch_target(target):
            return True
        # ShellExecuteW fallback instead of `cmd /c start`.
        return launch_external(target)

    def _launch_hgr_target(self, target: str) -> bool:
        lowered = target.lower()
        local = Path.home() / "AppData" / "Local"
        appdata = Path(os.getenv("APPDATA", ""))
        desktop = Path.home() / "Desktop"

        def first_existing(paths: list[Path]) -> Path | None:
            for path in paths:
                if path.exists():
                    return path
            return None

        def newest_glob(pattern: str) -> Path | None:
            matches = sorted(local.glob(pattern), reverse=True)
            return matches[0] if matches else None

        if lowered == "hgr:discord":
            discord_exe = newest_glob("Discord/app-*/Discord.exe")
            if discord_exe and self._launch_resolved_shortcut(str(discord_exe)):
                return True
            update_exe = local / "Discord" / "Update.exe"
            if update_exe.exists():
                try:
                    subprocess.Popen([str(update_exe), "--processStart", "Discord.exe"], shell=False)
                    return True
                except Exception:
                    pass
            return False

        if lowered == "hgr:hoyoplay":
            candidates = [
                local / "Programs" / "HoYoPlay" / "launcher.exe",
                local / "Programs" / "HoYoPlay" / "HoYoPlay.exe",
                local / "HoYoPlay" / "launcher.exe",
                desktop / "HoYoPlay.lnk",
                appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "HoYoPlay.lnk",
            ]
            path = first_existing(candidates)
            return self._launch_target(str(path)) if path else False

        if lowered == "hgr:genshin-impact":
            candidates = [
                desktop / "Genshin Impact.lnk",
                appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Genshin Impact.lnk",
                local / "Programs" / "HoYoPlay" / "launcher.exe",
                local / "Programs" / "HoYoPlay" / "HoYoPlay.exe",
            ]
            path = first_existing(candidates)
            return self._launch_target(str(path)) if path else False

        if lowered == "hgr:riot-client":
            candidates = [
                Path("C:/Riot Games/Riot Client/RiotClientServices.exe"),
                local / "Riot Games" / "Riot Client" / "RiotClientServices.exe",
                desktop / "Riot Client.lnk",
                appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Riot Client.lnk",
            ]
            path = first_existing(candidates)
            return self._launch_target(str(path)) if path else False

        if lowered == "hgr:valorant":
            riot_services = first_existing([
                Path("C:/Riot Games/Riot Client/RiotClientServices.exe"),
                local / "Riot Games" / "Riot Client" / "RiotClientServices.exe",
            ])
            if riot_services is not None:
                try:
                    subprocess.Popen(
                        [str(riot_services), "--launch-product=valorant", "--launch-patchline=live"],
                        shell=False,
                    )
                    return True
                except Exception:
                    pass
            candidates = [
                desktop / "VALORANT.lnk",
                appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "VALORANT.lnk",
            ]
            path = first_existing(candidates)
            return self._launch_target(str(path)) if path else False

        if lowered == "hgr:matlab":
            matlab_candidates = sorted(Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"), reverse=True)
            path = matlab_candidates[0] if matlab_candidates else None
            if path and path.exists():
                try:
                    subprocess.Popen([str(path)], shell=False)
                    return True
                except Exception:
                    pass
            resolved = shutil.which("matlab")
            if resolved:
                try:
                    subprocess.Popen([resolved], shell=False)
                    return True
                except Exception:
                    pass
            return False

        if lowered == "hgr:kicad":
            kicad_candidates = sorted(Path("C:/Program Files/KiCad").glob("*/bin/kicad.exe"), reverse=True)
            path = kicad_candidates[0] if kicad_candidates else None
            if path and path.exists():
                try:
                    subprocess.Popen([str(path)], shell=False)
                    return True
                except Exception:
                    pass
            resolved = shutil.which("kicad")
            if resolved:
                try:
                    subprocess.Popen([resolved], shell=False)
                    return True
                except Exception:
                    pass
            return False

        return False

    def _mac_open(self, target: str, *, as_app: bool = False) -> bool:
        """Open a file / folder / URL / app on macOS via the `open` command.

        `open <path>`  -> opens a file in its default app, or a folder in Finder.
        `open <url>`   -> opens a URL / mailto: in the default handler.
        `open -a <name>` -> launches an app by name (LaunchServices fuzzy-matches,
                            so 'chrome' -> Google Chrome, 'spotify' -> Spotify, ...).
        Returns False (non-zero exit) when the target/app can't be found.
        """
        text = str(target or "").strip()
        if not text:
            return False
        if not as_app and text.lower() == "explorer":
            text = str(Path.home())  # Windows 'explorer' (home) -> Finder at home
        cmd = ["open", "-a", text] if as_app else ["open", text]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=12)
            return result.returncode == 0
        except Exception:
            return False

    def _build_mac_catalog(self) -> list[DesktopAppEntry]:
        """Scan the standard macOS application directories for *.app bundles and
        build the resolver catalog (used for spoken-name matching + hints)."""
        roots = (
            Path("/Applications"),
            Path("/Applications/Utilities"),
            Path.home() / "Applications",
            Path("/System/Applications"),
            Path("/System/Applications/Utilities"),
        )
        entries: dict[str, DesktopAppEntry] = {}
        for root in roots:
            try:
                if not root.exists():
                    continue
                for app in root.glob("*.app"):
                    name = app.stem
                    norm = self._normalize_application_name(name)
                    if not norm or norm in entries:
                        continue
                    entries[norm] = DesktopAppEntry(
                        display_name=name,
                        normalized_name=norm,
                        target=str(app),
                        source="mac",
                        category="generic",
                    )
            except Exception:
                continue
        return sorted(entries.values(), key=lambda e: (e.normalized_name, e.display_name.lower()))

    def _launch_target(self, target: str) -> bool:
        if self._mac:
            return self._mac_open(target)
        lowered = str(target).lower()
        path_target = Path(target)
        try:
            import sys as _sys
            _self_exe = str(Path(_sys.executable).resolve()).lower()
            _target_resolved = str(path_target.resolve()).lower() if path_target.exists() else ""
            if _target_resolved and (_target_resolved == _self_exe or _target_resolved.endswith("touchless.exe") or _target_resolved.endswith("hgr app.exe")):
                self._message = "cannot reopen the running Touchless app"
                return False
            if lowered.startswith("hgr:"):
                return self._launch_hgr_target(target)
            if lowered.startswith("shell:appsfolder"):
                subprocess.Popen(["explorer.exe", target], shell=False)
                return True
            if path_target.suffix.lower() == ".url" and path_target.exists():
                resolved_url = self._resolve_url_shortcut(path_target)
                if resolved_url:
                    return self._launch_target(resolved_url)
            if path_target.suffix.lower() == ".lnk" and path_target.exists():
                try:
                    if hasattr(os, "startfile"):
                        os.startfile(str(path_target))
                        return True
                except Exception:
                    pass
                resolved_target, arguments = self._resolve_windows_shortcut(path_target)
                if resolved_target:
                    return self._launch_resolved_shortcut(resolved_target, arguments)
            if hasattr(os, "startfile"):
                os.startfile(target)
            else:  # pragma: no cover
                if not launch_external(target):
                    return False
            return True
        except Exception:
            # ShellExecuteW fallback — os.startfile internally calls the
            # same API, but the ctypes wrapper bypasses whatever
            # exception path startfile hit above.
            return launch_external(target)


    def minimize_active_window(self) -> bool:
        if not self._win:
            self._message = "window minimize unavailable on this platform"
            return False
        hwnd = self._foreground_window_handle()
        if hwnd <= 0:
            self._message = "no active window to minimize"
            return False
        title = self._window_text(hwnd)
        if self._show_window(hwnd, SW_MINIMIZE):
            self._message = f"minimized window: {title}" if title else "minimized active window"
            return True
        self._message = "could not minimize active window"
        return False

    def maximize_active_window(self) -> bool:
        if not self._win:
            self._message = "window maximize unavailable on this platform"
            return False
        hwnd = self._foreground_window_handle()
        if hwnd <= 0:
            self._message = "no active window to maximize"
            return False
        title = self._window_text(hwnd)
        if self._show_window(hwnd, SW_MAXIMIZE):
            self._message = f"maximized window: {title}" if title else "maximized active window"
            return True
        self._message = "could not maximize active window"
        return False

    def restore_active_window(self) -> bool:
        if not self._win:
            self._message = "window restore unavailable on this platform"
            return False
        hwnd = self._foreground_window_handle()
        if hwnd <= 0:
            self._message = "no active window to restore"
            return False
        title = self._window_text(hwnd)
        if self._show_window(hwnd, SW_RESTORE):
            self._message = f"restored window: {title}" if title else "restored active window"
            return True
        self._message = "could not restore active window"
        return False

    def close_active_window(self) -> bool:
        if not self._win:
            self._message = "window close unavailable on this platform"
            return False
        hwnd = self._foreground_window_handle()
        if hwnd <= 0:
            self._message = "no active window to close"
            return False
        title = self._window_text(hwnd)
        if self._post_close(hwnd):
            self._message = f"closed window: {title}" if title else "closed active window"
            return True
        self._message = "could not close active window"
        return False

    def close_named_window(self, app_name: str) -> bool:
        if not self._win:
            self._message = "window close unavailable on this platform"
            return False
        normalized_query = self._normalize_application_query(app_name) or self._normalize_application_name(app_name)
        if not normalized_query:
            return self.close_active_window()

        close_terms: set[str] = {normalized_query}
        for display_name, metadata in self.KNOWN_APPLICATIONS.items():
            normalized_display = self._normalize_application_name(display_name)
            aliases = tuple(metadata.get("aliases", ()))
            alias_norms = {self._normalize_application_name(alias) for alias in aliases if alias}
            known_terms = {normalized_display, *alias_norms}
            if any(
                term and (
                    normalized_query == term
                    or bool(re.search(rf'\b{re.escape(term)}\b', normalized_query))
                    or bool(re.search(rf'\b{re.escape(normalized_query)}\b', term))
                    or self._application_match_score(normalized_query, term) >= 0.84
                )
                for term in known_terms
            ):
                close_terms |= {term for term in known_terms if term}
                for target in metadata.get("targets", ()): 
                    target_name = Path(str(target)).stem or str(target)
                    target_norm = self._normalize_application_name(target_name)
                    if target_norm:
                        close_terms.add(target_norm)

        matches: list[tuple[int, str, str]] = []
        seen_hwnds: set[int] = set()
        for hwnd, title, pid in self._enumerate_top_level_windows():
            process_name = self._process_name_from_pid(pid)
            title_norm = self._normalize_application_name(title)
            process_norm = self._normalize_application_name(Path(process_name).stem or process_name)
            haystacks = [value for value in (title_norm, process_norm) if value]
            if not haystacks:
                continue
            matched = False
            for term in close_terms:
                for haystack in haystacks:
                    if term == haystack or bool(re.search(rf'\b{re.escape(term)}\b', haystack)) or bool(re.search(rf'\b{re.escape(haystack)}\b', term)) or self._application_match_score(term, haystack) >= 0.82:
                        matched = True
                        break
                if matched:
                    break
            if not matched or hwnd in seen_hwnds:
                continue
            seen_hwnds.add(hwnd)
            matches.append((hwnd, title, process_name))

        if not matches:
            self._message = f"could not find window: {normalized_query}"
            return False

        closed = 0
        for hwnd, _title, _process_name in matches:
            if self._post_close(hwnd):
                closed += 1

        if closed <= 0:
            self._message = f"could not close window: {normalized_query}"
            return False
        self._message = f"closed window: {normalized_query}" if closed == 1 else f"closed {closed} windows: {normalized_query}"
        return True

    def _foreground_window_handle(self) -> int:
        try:
            user32 = ctypes.windll.user32
            return int(user32.GetForegroundWindow() or 0)
        except Exception:
            return 0

    def _show_window(self, hwnd: int, show_command: int) -> bool:
        try:
            user32 = ctypes.windll.user32
            return bool(user32.ShowWindow(wintypes.HWND(hwnd), int(show_command)))
        except Exception:
            return False

    def _post_close(self, hwnd: int) -> bool:
        try:
            user32 = ctypes.windll.user32
            return bool(user32.PostMessageW(wintypes.HWND(hwnd), WM_CLOSE, 0, 0))
        except Exception:
            return False

    def _window_text(self, hwnd: int) -> str:
        try:
            user32 = ctypes.windll.user32
            length = int(user32.GetWindowTextLengthW(wintypes.HWND(hwnd)) or 0)
            buffer = ctypes.create_unicode_buffer(max(1, length + 1))
            user32.GetWindowTextW(wintypes.HWND(hwnd), buffer, len(buffer))
            return str(buffer.value or '').strip()
        except Exception:
            return ''

    def _process_name_from_pid(self, pid: int) -> str:
        try:
            if pid <= 0:
                return ''
            return str(psutil.Process(pid).name() or '').strip()
        except Exception:
            return ''

    def _enumerate_top_level_windows(self) -> list[tuple[int, str, int]]:
        if not self._available:
            return []
        results: list[tuple[int, str, int]] = []
        try:
            user32 = ctypes.windll.user32
        except Exception:
            return results
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        get_pid = user32.GetWindowThreadProcessId
        get_pid.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        get_pid.restype = wintypes.DWORD
        is_visible = user32.IsWindowVisible
        is_visible.argtypes = [wintypes.HWND]
        is_visible.restype = wintypes.BOOL
        get_owner = getattr(user32, 'GetWindow', None)
        GW_OWNER = 4

        @EnumWindowsProc
        def _enum_proc(hwnd, _lparam):
            try:
                if not bool(is_visible(hwnd)):
                    return True
                if get_owner is not None and int(get_owner(hwnd, GW_OWNER) or 0) != 0:
                    return True
                title = self._window_text(int(hwnd))
                pid = wintypes.DWORD(0)
                get_pid(hwnd, ctypes.byref(pid))
                process_name = self._process_name_from_pid(int(pid.value))
                if not title and not process_name:
                    return True
                results.append((int(hwnd), title, int(pid.value)))
            except Exception:
                return True
            return True

        try:
            user32.EnumWindows(_enum_proc, 0)
        except Exception:
            return []
        return results

# Author: Konstantin Markov

"""Canonical per-platform application directories for Touchless.

Collapses the ~20 scattered `LOCALAPPDATA or ~/.local/share` idioms and the
hard-coded `Documents/Touchless` literals into one place. This is *additive*:
existing modules keep working unchanged; new macOS code (and, incrementally,
adopters) call these.

Platform mapping (see docs/MACOS_PORT.md "runtime-paths"):

  | helper               | Windows                              | macOS                                   |
  |----------------------|--------------------------------------|-----------------------------------------|
  | app_config_dir()     | ~/.touchless                         | ~/.touchless  (kept for parity; see Q6) |
  | app_data_dir()       | %LOCALAPPDATA%/Touchless             | ~/Library/Application Support/Touchless  |
  | app_logs_dir()       | %LOCALAPPDATA%/Touchless/logs        | ~/Library/Logs/Touchless                 |
  | app_cache_dir()      | %LOCALAPPDATA%/Touchless/Cache       | ~/Library/Caches/Touchless               |
  | models_dir()         | ~/Documents/TouchlessVoiceModels     | ~/Documents/TouchlessVoiceModels         |
  | user_movies_dir()    | ~/Videos                             | ~/Movies                                 |

`app_config_dir()` intentionally stays at `~/.touchless` on macOS so we never
strand an existing user's settings. Relocating macOS config into
~/Library/Application Support is a future, opt-in migration (docs Q6).

Author: Konstantin Markov (macOS port)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Touchless"

_IS_WIN = sys.platform == "win32"
_IS_MAC = sys.platform == "darwin"


def _home() -> Path:
    return Path.home()


def _localappdata() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(_home() / "AppData" / "Local")
    return Path(base)


def app_config_dir() -> Path:
    """User settings dir. Matches hgr.config.app_config.CONFIG_DIR on every OS."""
    return _home() / ".touchless"


def app_data_dir() -> Path:
    """Private app data (databases, downloaded runtimes, caches that must persist)."""
    if _IS_WIN:
        return _localappdata() / APP_NAME
    if _IS_MAC:
        return _home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME") or (_home() / ".local" / "share")) / APP_NAME


def app_logs_dir() -> Path:
    if _IS_MAC:
        return _home() / "Library" / "Logs" / APP_NAME
    return app_data_dir() / "logs"


def app_cache_dir() -> Path:
    if _IS_WIN:
        return _localappdata() / APP_NAME / "Cache"
    if _IS_MAC:
        return _home() / "Library" / "Caches" / APP_NAME
    return Path(os.environ.get("XDG_CACHE_HOME") or (_home() / ".cache")) / APP_NAME


def user_documents_dir() -> Path:
    return _home() / "Documents"


def user_pictures_dir() -> Path:
    return _home() / "Pictures"


def user_movies_dir() -> Path:
    """Default video output dir. macOS uses ~/Movies; Windows uses ~/Videos.

    (On macOS ~/Videos does not exist by default, so writing there silently
    creates a stray folder — use ~/Movies. See docs/MACOS_PORT.md.)
    """
    return _home() / ("Movies" if _IS_MAC else "Videos")


def models_dir() -> Path:
    """Voice/LLM model download dir (shared by dictation + grammar corrector)."""
    return _home() / "Documents" / "TouchlessVoiceModels"


def ensure(path: Path) -> Path:
    """mkdir -p the given path (best-effort) and return it."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path

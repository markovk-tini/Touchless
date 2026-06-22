"""Per-platform capability flags.

Lets feature code ask "is this available on the current OS?" without sprinkling
`sys.platform` checks (and their rationale) across the codebase. Each macOS gap
is documented in docs/MACOS_PORT.md ("Not-possible / degraded features").

Importing this module is side-effect-free and safe on every platform.

Author: Konstantin Markov (macOS port)
"""
from __future__ import annotations

import sys

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


# --- Capabilities that exist on Windows but are ABSENT / DEGRADED on macOS ---
# (see docs/MACOS_PORT.md). Feature code should gate on these and degrade
# gracefully rather than hard-coding `if sys.platform == "win32"`.

#: Per-application output volume + the AppSessionDucker. macOS CoreAudio is
#: device-level only — there is no public per-process output volume API.
PER_APP_AUDIO_VOLUME = IS_WINDOWS
AUDIO_DUCKING = IS_WINDOWS

#: Classic Outlook COM/MAPI (zero-auth inbox). No COM on macOS -> use Graph.
OUTLOOK_COM = IS_WINDOWS

#: Windows Phone Link SMS/iMessage bridge. macOS replacement is a separate
#: Messages.app/chat.db connector (not a port).
PHONE_LINK = IS_WINDOWS

#: "Maximize" a *foreign* app's window. No direct macOS concept (closest is the
#: AX zoom button or sizing to visibleFrame).
FOREIGN_WINDOW_MAXIMIZE = IS_WINDOWS

#: Microsoft Store update channel.
MS_STORE_UPDATES = IS_WINDOWS

#: System dictation hotkey (Win+H). No stable macOS analog.
SYSTEM_DICTATION_HOTKEY = IS_WINDOWS


# --- Capabilities requiring a runtime TCC permission grant on macOS ----------
# On Windows these "just work"; on macOS the user must grant the permission and
# it can reset on a signature change. UI should surface a "grant X to enable Y"
# state. These are *potentially available* (not blocked), hence True on macOS.

#: Synthetic mouse/keyboard, AXUIElement reads/actions, idle detection.
#: Requires Accessibility (cannot be auto-granted).
def needs_accessibility() -> bool:
    return IS_MACOS


#: Screen capture for OCR/vision and foreign-window titles. Requires Screen
#: Recording (cannot be auto-granted).
def needs_screen_recording() -> bool:
    return IS_MACOS


#: AppleScript control of Chrome/Spotify/Discord/Office/Messages. Requires a
#: per-target Automation grant on first Apple Event.
def needs_automation() -> bool:
    return IS_MACOS

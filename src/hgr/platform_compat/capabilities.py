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


def is_accessibility_trusted(prompt: bool = False) -> bool:
    """macOS: is this process trusted for the Accessibility permission?

    Accessibility gates ALL synthetic mouse/keyboard input (CGEventPost) and
    AXUIElement reads. If ``prompt`` is True and the process is untrusted,
    macOS shows the grant prompt and registers the app under System Settings >
    Privacy & Security > Accessibility — the user must then toggle it on AND
    RESTART the app for it to take effect (it cannot be granted
    programmatically). Always returns True off macOS.

    For an ad-hoc-signed build the grant is keyed to the (unstable) code hash,
    so it resets on every rebuild — a stable signing identity fixes that."""
    if not IS_MACOS:
        return True
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions  # type: ignore

        try:
            from ApplicationServices import kAXTrustedCheckOptionPrompt  # type: ignore

            key = kAXTrustedCheckOptionPrompt
        except Exception:
            key = "AXTrustedCheckOptionPrompt"
        return bool(AXIsProcessTrustedWithOptions({key: bool(prompt)}))
    except Exception:
        try:
            from ApplicationServices import AXIsProcessTrusted  # type: ignore

            return bool(AXIsProcessTrusted())
        except Exception:
            return False


def is_screen_recording_trusted(prompt: bool = False) -> bool:
    """macOS: is this process trusted for the Screen Recording permission?

    Screen Recording is a DISTINCT TCC permission (separate from Camera /
    Accessibility / Automation). It gates CGDisplayCreateImage / ScreenCaptureKit
    (used for the instant-clip rolling buffer, OCR/vision, foreign-window
    titles). ``CGPreflightScreenCaptureAccess`` returns the grant status WITHOUT
    prompting; ``CGRequestScreenCaptureAccess`` (prompt=True) shows the OS prompt
    once and registers the app under System Settings > Privacy & Security >
    Screen Recording. A fresh grant needs an app RESTART to take effect for the
    capture APIs, and (for ad-hoc-signed builds) resets on rebuild. Always True
    off macOS; True on pre-10.15 where the preflight symbol is absent."""
    if not IS_MACOS:
        return True
    try:
        import Quartz  # type: ignore

        if prompt and hasattr(Quartz, "CGRequestScreenCaptureAccess"):
            return bool(Quartz.CGRequestScreenCaptureAccess())
        if hasattr(Quartz, "CGPreflightScreenCaptureAccess"):
            return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        pass
    return True


# --- Camera / Microphone (AVFoundation) TCC status + prompt -------------------
# Camera and Microphone are their own TCC permissions, distinct from
# Accessibility / Screen Recording / Automation. Unlike those, macOS shows the
# grant prompt *in-process* the first time the app touches the device (or when
# we call AVCaptureDevice.requestAccessForMediaType_), and a fresh grant takes
# effect IMMEDIATELY — no app restart required. That's why the onboarding wizard
# can request these live and reflect the result on its next status poll.

# AVMediaType four-char codes (AVMediaTypeVideo / AVMediaTypeAudio).
_AV_VIDEO = "vide"
_AV_AUDIO = "soun"
# AVAuthorizationStatus: 0 notDetermined, 1 restricted, 2 denied, 3 authorized.
_AV_STATE_NAMES = {0: "notDetermined", 1: "restricted", 2: "denied", 3: "authorized"}


def _av_media_state(media_type: str) -> str:
    """AVFoundation authorization state for a media type as a string:
    'authorized' | 'denied' | 'notDetermined' | 'restricted' | 'unknown'.
    'unknown' off macOS or if AVFoundation can't be reached."""
    if not IS_MACOS:
        return "unknown"
    try:
        from AVFoundation import AVCaptureDevice  # type: ignore

        s = int(AVCaptureDevice.authorizationStatusForMediaType_(media_type))
        return _AV_STATE_NAMES.get(s, "unknown")
    except Exception:
        return "unknown"


def camera_permission_state() -> str:
    return _av_media_state(_AV_VIDEO)


def microphone_permission_state() -> str:
    return _av_media_state(_AV_AUDIO)


def is_camera_authorized() -> bool:
    """True when the camera is usable. Off macOS there is no TCC gate, so we
    treat it as authorized (Windows just opens the device)."""
    return True if not IS_MACOS else camera_permission_state() == "authorized"


def is_microphone_authorized() -> bool:
    """True when the microphone is usable. True off macOS (no TCC gate)."""
    return True if not IS_MACOS else microphone_permission_state() == "authorized"


def _request_av_media_access(media_type: str) -> None:
    """Fire the OS grant prompt for a media type (async, fire-and-forget).
    Only prompts while the status is notDetermined — once the user has decided,
    macOS won't re-prompt and they must use System Settings. No-op off macOS."""
    if not IS_MACOS:
        return
    try:
        from AVFoundation import AVCaptureDevice  # type: ignore

        # completionHandler is required by the API; a no-op block is fine — the
        # caller polls authorizationStatusForMediaType_ separately rather than
        # reacting inside the handler (which runs on an arbitrary GCD queue,
        # not the Qt/main thread).
        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            media_type, lambda granted: None
        )
    except Exception:
        pass


def request_camera_access() -> None:
    _request_av_media_access(_AV_VIDEO)


def request_microphone_access() -> None:
    _request_av_media_access(_AV_AUDIO)

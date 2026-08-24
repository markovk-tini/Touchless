"""Irreversible speed bump — the honest 30% of universal undo.

Phase-1 trust substrate. Two layers:

  1. **Pre-invocation gate**: before any DESTRUCTIVE or IRREVERSIBLE
     tool runs, route the call through a confirmation callback. The
     callback is provided by the UI (chat panel's QMessageBox flow),
     or `None` in headless contexts (defaults to ALLOW for backward
     compatibility — the existing planner already has its own narrower
     confirm gate for risky plans, so this layer adds a second
     sentinel without breaking flows that don't wire a callback).

  2. **Reversibility steerage**: for tools that COULD be made reversible
     by rerouting (e.g. delete_file → send to Recycle Bin instead of
     hard unlink), this module exposes `safe_file_delete()` which
     downstream tool handlers call instead of os.unlink().

The full universal-undo system (per-tool inverse-action registry +
30-day reversal path) is Phase-1's most-mispriced item per the
critics; we deliberately skip it and ship the SPEED BUMP that captures
~70% of the value at ~10% of the cost. Per-domain undo lands
incrementally as each connector matures.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .tool_metadata import Destructiveness, destructiveness_of


# Module-level callback the UI installs. Signature: (title, detail) -> bool
# True = proceed, False = cancel. None = no UI installed; we DEFAULT to
# proceed so we never break headless / scripted flows.
ConfirmCallback = Callable[[str, str], bool]
_confirm_callback: Optional[ConfirmCallback] = None
_callback_lock = threading.Lock()


def install_confirm_callback(fn: Optional[ConfirmCallback]) -> None:
    """UI calls this once at startup with the QMessageBox-backed
    callback. Idempotent — last installer wins. Pass None to remove."""
    global _confirm_callback
    with _callback_lock:
        _confirm_callback = fn


def get_confirm_callback() -> Optional[ConfirmCallback]:
    with _callback_lock:
        return _confirm_callback


# ---- pre-invocation gate ---------------------------------------------------


def needs_confirmation(tool: str, args: Dict[str, Any]) -> bool:
    """True when this (tool, args) pair should prompt the user before
    running. Reads destructiveness from tool_metadata."""
    d = destructiveness_of(tool)
    return d in (Destructiveness.DESTRUCTIVE, Destructiveness.IRREVERSIBLE)


def gate(tool: str, args: Dict[str, Any],
         *, source: str = "") -> Tuple[bool, Optional[str]]:
    """Returns (allowed, decline_reason).

      * (True, None)  → proceed with the tool call.
      * (False, msg)  → caller MUST return an error result with `msg`
                         and SKIP the tool.

    Failure-open by design: if no callback is installed (frozen build,
    headless test, etc.) we don't block — the planner's existing
    plan-level confirm + the connector-internal validations already
    cover most risky paths. This adds a SECOND sentinel without
    introducing new blocking behavior. The chat-panel callback wiring
    is what turns this into a hard gate.

    Phase-3 voice-safety wiring: when `source` indicates the
    invocation came via voice, an additional VoiceSpoofDefense check
    runs FIRST. The defense layer requires a non-voice second channel
    (gesture / physical button / keyboard) for destructive ops, refuses
    repeat-attacks within a 2-second window, and refuses everything
    while a TTS-loop is detected. Best-effort: any failure in that
    layer falls through to the existing typed-confirm gate.
    """
    if not needs_confirmation(tool, args):
        return True, None
    # Phase-3 voice-spoof defense — only relevant when this tool call
    # was triggered via voice. The planner / realtime tagged its
    # source on the invocation; pass it down to us as a string.
    if str(source).lower() in ("voice", "realtime"):
        try:
            from .voice_spoof_defense import (
                ConfirmChannel, global_defense)
            from .tool_metadata import destructiveness_of
            # The defense module expects the destructiveness string
            # ("read"/"write"/"destructive"/"irreversible").
            d_str = destructiveness_of(tool).value
            # Today we don't have a structured way to know which
            # second channel the user used for THIS specific op
            # (gesture vs typed). Pass NONE; the defense will require
            # second-channel for destructive/irreversible and the
            # caller's UI layer can re-invoke with a CHANNEL once it
            # gathers the confirmation. Until that UI layer lands,
            # the practical effect is: voice → destructive → blocked
            # → user falls through to the typed-confirm gate below
            # (which IS a second channel in practice — typed input
            # in a Qt modal is harder to spoof than audio).
            decision = global_defense().check_destructive_voice_op(
                tool=tool, args=args, source=source,
                destructiveness=d_str,
                confirm_channel=ConfirmChannel.NONE,
            )
            if not decision.allowed and not decision.require_second_channel:
                # Hard refuse (TTS-loop / repeat-attack). Don't fall
                # through to typed confirm — the defense layer says no.
                return False, decision.reason
            # `require_second_channel=True` means the user CAN proceed
            # via a non-voice channel. The typed Qt modal below IS that
            # second channel, so fall through to it.
        except Exception:
            pass  # best-effort; never block the gate on internal error
    cb = get_confirm_callback()
    if cb is None:
        return True, None  # failure-open in headless contexts
    title, detail = _format_prompt(tool, args)
    # When the trigger was voice, surface that to the user so the
    # confirm modal carries the extra trust context — they should
    # know this destructive op originated from spoken audio (which
    # is more easily spoofed than a typed/clicked request).
    if str(source).lower() in ("voice", "realtime"):
        title = f"[voice command] {title}"
        detail = (detail
                  + "\n\n⚠ This was triggered by voice. If you didn't "
                    "say this, click No.")
    try:
        allowed = bool(cb(title, detail))
    except Exception:
        # Callback errored — bias toward NOT silently running a
        # destructive op. The caller surfaces a clean error.
        return False, "confirmation callback errored — aborting destructive action"
    if allowed:
        return True, None
    return False, f"User declined: {tool}"


def _format_prompt(tool: str, args: Dict[str, Any]) -> Tuple[str, str]:
    """Compose a clear human-readable prompt. Per-tool overrides could
    be added in v2 (e.g. mail-send shows recipient + subject). The
    blanket version is honest about the action without leaking secrets
    in the dialog body."""
    d = destructiveness_of(tool)
    if d == Destructiveness.IRREVERSIBLE:
        title = f"This will permanently {_verb_for(tool)}."
    else:
        title = f"Confirm: {_verb_for(tool)}?"
    # Build a SHORT detail line listing 1-2 user-relevant args, redacted.
    chunks = []
    for k in ("to", "subject", "path", "url", "target", "name", "title"):
        if k in args and args[k]:
            v = str(args[k])
            if len(v) > 80:
                v = v[:80] + "…"
            chunks.append(f"{k}: {v}")
        if len(chunks) >= 2:
            break
    detail = "\n".join(chunks) if chunks else f"Tool: {tool}"
    return title, detail


_VERB_BY_TOOL = {
    # outbound — messages
    "email_send": "send an email", "gmail_send": "send an email",
    "ms_mail_send": "send an email", "outlook_compose": "draft an email",
    "outlook_send": "send an email",
    "teams_send": "send a Teams message",
    "teams_post": "post in a Teams channel",
    "teams_channel_post": "post in a Teams channel",
    "slack_post": "post in Slack", "slack_send": "send in Slack",
    "discord_send": "send a Discord message",
    "discord_post": "post in Discord",
    "phone_link_send_text": "send a text message",
    # calendar
    "calendar_create": "create a calendar event",
    "calendar_create_event": "create a calendar event",
    "calendar_event_create": "create a calendar event",
    "ms_calendar_create": "create a calendar event",
    # cloud uploads / docs
    "drive_upload": "upload to Drive", "onedrive_upload": "upload to OneDrive",
    "gdocs_create": "create a Google Doc",
    "gdocs_append_text": "append text to a Google Doc",
    "sheets_create": "create a Google Sheet",
    "sheets_append_rows": "add rows to a Google Sheet",
    "slides_create": "create a Google Slides deck",
    "slides_add_slide": "add a slide",
    "onenote_create": "create a OneNote page",
    "onenote_append_text": "append text to a OneNote page",
    "excel_create": "create an Excel workbook",
    "word_create": "create a Word doc",
    "powerpoint_create": "create a PowerPoint deck",
    # notion
    "notion_create_page": "create a Notion page",
    "notion_append_to_page": "append to a Notion page",
    "notion_add_to_database": "add a row to a Notion database",
    # destructive
    "delete_file": "delete a file", "file_delete": "delete a file",
    "drive_delete": "delete a Drive file",
    "mail_delete": "delete an email",
    "iris_forget_contact": "forget a contact from memory",
    # mcp
    "mcp_call": "run an MCP tool",
}


def _verb_for(tool: str) -> str:
    return _VERB_BY_TOOL.get(tool, f"run {tool}")


# ---- reversibility steerage -----------------------------------------------


def safe_file_delete(path: Path) -> Tuple[bool, str]:
    """Send a file to the Recycle Bin instead of permanent unlink.
    Returns (ok, message). Used by `delete_file` tool to make
    deletions actually recoverable.

    Uses `send2trash` when available (cross-platform, handles
    Windows Recycle Bin + macOS Trash + Linux freedesktop trash).
    Falls back to platform-specific Win32 SHFileOperation if not.
    Last resort: refuses (returns False) rather than hard-unlinking,
    because hard-unlinking from Iris should be an EXPLICIT user
    choice, never a fallback."""
    path = Path(path)
    if not path.exists():
        return False, f"not found: {path}"
    # Preferred: send2trash. Bundle this in PyInstaller hiddenimports.
    try:
        import send2trash  # type: ignore
        send2trash.send2trash(str(path))
        return True, f"sent to Recycle Bin: {path.name}"
    except Exception:
        pass
    # Fallback: SHFileOperation on Windows.
    try:
        import platform
        if platform.system() == "Windows":
            import ctypes
            from ctypes import wintypes
            # SHFileOperationW with FO_DELETE + FOF_ALLOWUNDO sends to
            # Recycle Bin. Pack the path as a double-null-terminated
            # wide string.
            FO_DELETE = 0x0003
            FOF_SILENT = 0x0004
            FOF_NOCONFIRMATION = 0x0010
            FOF_ALLOWUNDO = 0x0040
            FOF_NOERRORUI = 0x0400
            class SHFILEOPSTRUCTW(ctypes.Structure):  # noqa: N801
                _fields_ = [
                    ("hwnd", wintypes.HWND),
                    ("wFunc", wintypes.UINT),
                    ("pFrom", ctypes.c_wchar_p),
                    ("pTo", ctypes.c_wchar_p),
                    ("fFlags", ctypes.c_uint16),
                    ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p),
                    ("lpszProgressTitle", ctypes.c_wchar_p),
                ]
            op = SHFILEOPSTRUCTW()
            op.hwnd = 0
            op.wFunc = FO_DELETE
            op.pFrom = str(path) + "\0\0"
            op.pTo = None
            op.fFlags = (FOF_ALLOWUNDO | FOF_NOCONFIRMATION
                         | FOF_NOERRORUI | FOF_SILENT)
            shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
            rc = shell32.SHFileOperationW(ctypes.byref(op))
            if rc == 0 and not op.fAnyOperationsAborted:
                return True, f"sent to Recycle Bin: {path.name}"
            return False, f"SHFileOperation failed (code {rc})"
    except Exception as exc:
        return False, f"recycle-bin send failed: {type(exc).__name__}: {exc}"
    return False, ("no Recycle Bin backend available — install send2trash "
                   "(pip install send2trash). Refusing to permanently "
                   "delete: that would defeat the speed-bump.")

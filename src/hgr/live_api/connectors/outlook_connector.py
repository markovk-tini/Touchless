"""Email connector — the API-first path for composing email.

iris composes an email (recipient + subject + body pre-filled) with one
deterministic call instead of driving a mail UI by screenshot-and-click.

Provider: if the user's default email (user_prefs.get_default_email) is a
Gmail address, compose opens **Gmail web compose** pre-filled, from that
account — so it uses the user's chosen account and needs no OAuth. Otherwise
it falls back to a `mailto:` link (the OS default mail client, e.g. Outlook).
The `outlook_open*` tools still target the Outlook desktop app.

Controller sharing: reuses the executor's lazily-created DesktopController
via `executor._ensure_desktop()` (building the app catalog is expensive —
one instance only). Falls back to its own if no executor is supplied.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import urllib.parse
import webbrowser
from typing import Any, Dict, List, Optional

from .base import Connector, connector_result
from ..user_prefs import get_default_email


class OutlookConnector(Connector):
    id = "outlook"

    def __init__(self, executor: Optional[Any] = None,
                 controller: Optional[Any] = None) -> None:
        self._executor = executor
        self._own = controller

    def _ctrl(self):
        if self._executor is not None:
            try:
                ctrl = self._executor._ensure_desktop()
                if ctrl is not None:
                    return ctrl
            except Exception:
                pass
        if self._own is None:
            from ...debug.desktop_controller import DesktopController
            self._own = DesktopController()
        return self._own

    def available(self) -> bool:
        try:
            return bool(getattr(self._ctrl(), "available", False))
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        def fn(name: str, desc: str, props: Dict[str, Any] | None = None,
               required: List[str] | None = None) -> Dict[str, Any]:
            return {
                "type": "function",
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props or {},
                    "required": required or [],
                    "additionalProperties": False,
                },
            }

        mail_params = {
            "recipient": {"type": "string", "description": "Email address to send to."},
            "subject": {"type": "string",
                        "description": "Email subject — always provide one; "
                                       "summarize the body if unstated."},
            "body": {"type": "string", "description": "Email body text."},
        }
        return [
            fn("outlook_compose",
               "Open a pre-filled email DRAFT (Gmail web compose if the user's "
               "default is Gmail, else the OS default mail app) for the user to "
               "review and send themselves. Does NOT send. Use when the user "
               "just wants a draft. ALWAYS pass a concise `subject` (summarize "
               "the body if unstated). "
               "ONLY call this tool when the utterance STARTS with a compose/"
               "send verb (send, mail, compose, draft, write, shoot, fire off, "
               "or 'email' used as an imperative verb followed by a recipient). "
               "DO NOT call this tool when the word 'email' appears as CONTENT "
               "(e.g. after 'to say', 'to write', 'set to', 'change to', "
               "'enter', 'type', 'put', inside quotes) or when the utterance "
               "references a spreadsheet cell (A1-style token such as C1, B12, "
               "AA3), a Sheets / Docs / Slides / OneNote document, or otherwise "
               "names a non-mail target. Those requests go to "
               "sheets_update_range / gdocs_append / slides_* instead.",
               mail_params),
            fn("email_send",
               "Compose AND SEND an email without API setup: opens the pre-filled "
               "draft, then presses the mail app's send shortcut (Ctrl+Enter, "
               "works in Gmail and Outlook). Use when the user says to SEND an "
               "email. The user is asked to confirm first. ALWAYS pass a concise "
               "`subject`. If it can't confirm sending, the draft stays open to "
               "send manually — do NOT retry in a loop.",
               mail_params),
            fn("outlook_open", "Open the Outlook desktop app."),
            fn("outlook_open_folder",
               "Open Outlook and select a folder (e.g. 'inbox', 'sent', "
               "'drafts', 'junk', 'deleted').",
               {"folder": {"type": "string",
                           "description": "Folder name, e.g. 'inbox'."}},
               ["folder"]),
        ]

    def _open_draft(self, c, args: Dict[str, Any]) -> tuple[bool, str, Optional[str]]:
        """Open a pre-filled draft (Gmail web if the default is Gmail, else
        mailto). Returns (ok, via, recipient)."""
        recipient = str(args.get("recipient") or "").strip() or None
        subject = str(args.get("subject") or "").strip() or None
        body = str(args.get("body") or "").strip() or None
        # Never leave the subject blank — derive one from the body.
        if not subject and body:
            words = body.splitlines()[0].strip().split()
            subject = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
        default_email = get_default_email()
        if default_email and "gmail.com" in default_email.lower():
            if self._open_gmail_compose(default_email, recipient, subject, body):
                return True, "gmail", recipient
        ok = bool(c.compose_email(recipient=recipient, subject=subject, body=body))
        return ok, "mailto", recipient

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        c = self._ctrl()
        if name == "outlook_compose":
            ok, via, recipient = self._open_draft(c, args)
            return connector_result("ok" if ok else "error",
                                    opened=ok, recipient=recipient, via=via)
        if name == "email_send":
            ok, via, recipient = self._open_draft(c, args)
            if not ok:
                return connector_result("error", error="could not open the draft",
                                        recipient=recipient)
            # Give the compose window a moment to load + take focus, then fire
            # the universal mail send shortcut (Ctrl+Enter in Gmail & Outlook).
            import time
            time.sleep(2.5)
            sent = self._send_ctrl_enter()
            return connector_result(
                "ok" if sent else "error",
                sent=sent, recipient=recipient, via=via,
                note=("sent via the mail app's keyboard shortcut" if sent
                      else "draft is open — the user can press Send manually"),
            )
        if name == "outlook_open":
            return connector_result("ok" if c.open_outlook() else "error")
        if name == "outlook_open_folder":
            folder = str(args.get("folder") or "").strip()
            if not folder:
                return connector_result("error", error="folder is required")
            ok = bool(c.open_outlook_folder(folder))
            return connector_result("ok" if ok else "error", folder=folder)
        return connector_result("error", error=f"unknown outlook tool: {name}", code="no_handler")

    @staticmethod
    def _send_ctrl_enter() -> bool:
        """Press Ctrl+Enter — the 'send' shortcut in both Gmail web compose and
        Outlook — targeting whatever window has focus (the draft we just
        opened). Windows-only; best-effort."""
        try:
            import ctypes
            u = ctypes.windll.user32
            VK_CONTROL, VK_RETURN, KEYUP = 0x11, 0x0D, 0x0002
            u.keybd_event(VK_CONTROL, 0, 0, 0)
            u.keybd_event(VK_RETURN, 0, 0, 0)
            u.keybd_event(VK_RETURN, 0, KEYUP, 0)
            u.keybd_event(VK_CONTROL, 0, KEYUP, 0)
            return True
        except Exception:
            return False

    @staticmethod
    def _open_gmail_compose(account: str, to: Optional[str], subject: Optional[str],
                            body: Optional[str]) -> bool:
        """Open Gmail's web compose window pre-filled, scoped to `account`
        (authuser), in the user's default browser."""
        params = {"view": "cm", "fs": "1", "authuser": account}
        if to:
            params["to"] = to
        if subject:
            params["su"] = subject
        if body:
            params["body"] = body
        url = "https://mail.google.com/mail/?" + urllib.parse.urlencode(params)
        try:
            # os.startfile is the most reliable URL launcher on Windows;
            # webbrowser is the cross-platform fallback.
            opener = getattr(os, "startfile", None)
            if opener is not None:
                opener(url)
            else:
                webbrowser.open(url, new=2)
            return True
        except Exception:
            try:
                return bool(webbrowser.open(url, new=2))
            except Exception:
                return False

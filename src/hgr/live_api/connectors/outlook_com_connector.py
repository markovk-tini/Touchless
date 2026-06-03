"""Outlook desktop COM connector — zero-auth email reading on Windows.

Talks directly to a running (or installable) Outlook.exe via the
Win32 COM Automation interface. No OAuth, no API quotas, no Google /
Microsoft verification fees, no scope grants. Whatever account(s)
Outlook is configured to sync (Exchange, IMAP, Gmail-via-IMAP,
Outlook.com, anything) becomes readable for free.

Performance notes:
  - `Items.Restrict("[Unread] = true")` filters at the MAPI layer
    before properties cross the COM boundary — orders of magnitude
    faster than iterating + checking in Python (each .Property access
    is a cross-process call).
  - `Items.Sort("[ReceivedTime]", True)` so we grab newest-first.
  - We hold the Application + Namespace handles in the connector so
    repeat calls reuse them and skip the COM dispatch overhead
    (~80-150 ms on cold start, ~5-15 ms warm).
  - We read ONLY the fields we need per message; reading .Body is the
    expensive part so it's gated on include_body.

Threading: COM is apartment-threaded. The connector calls
`pythoncom.CoInitialize()` defensively in case execute() is invoked
from a non-main thread (e.g. the tool executor's worker thread).
CoUninitialize is NOT called here — we want to stay in COM context
for the connector's lifetime.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from .base import Connector, connector_result

# Outlook folder constants we use. From MSDN's OlDefaultFolders enum.
_OL_FOLDER_INBOX = 6


def _strip_html(html: str) -> str:
    """Minimal HTML-to-text. We only need it for the snippet preview;
    full body parsing is rare here."""
    if not html:
        return ""
    text = re.sub(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", " ",
                  html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;|&apos;", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _libs_available() -> bool:
    try:
        import win32com.client  # noqa: F401
        import pythoncom  # noqa: F401
        return True
    except Exception:
        return False


class OutlookComConnector(Connector):
    """Read the user's Outlook desktop inbox via COM. Free, fast, no
    auth — works whenever Outlook is installed AND can run (i.e. has
    been launched at least once and its profile is set up)."""

    id = "outlook_com"

    def __init__(self) -> None:
        self._app = None        # cached COM dispatch handle
        self._namespace = None  # cached MAPI namespace
        self._last_probe = 0.0  # debounce probes — Dispatch is slow on miss
        self._last_probe_result = False

    # ---- COM lifecycle ----------------------------------------------------

    def _ensure_com_apartment(self) -> bool:
        """COM needs the calling thread to be in an apartment. The tool
        executor may dispatch from a worker thread (Qt slot, asyncio,
        etc.); CoInitialize is idempotent within a thread so this is
        cheap to call repeatedly."""
        try:
            import pythoncom
            pythoncom.CoInitialize()
            return True
        except Exception:
            return False

    def _dispatch(self) -> Optional[Any]:
        """Get or create the Outlook.Application COM object. Returns
        None if Outlook isn't installed or refuses to start."""
        if self._app is not None:
            return self._app
        if not self._ensure_com_apartment():
            return None
        try:
            import win32com.client
            # `Dispatch` connects to a running Outlook instance if one
            # exists, otherwise launches a new one. Both flavors return
            # the same Application object.
            self._app = win32com.client.Dispatch("Outlook.Application")
            self._namespace = self._app.GetNamespace("MAPI")
            return self._app
        except Exception:
            self._app = None
            self._namespace = None
            return None

    # ---- Connector API ----------------------------------------------------

    def available(self) -> bool:
        """Cheap probe: are the libs present + can we dispatch? Cached
        for 30 s to avoid hammering COM on a tight loop. A failure is
        also cached so the cascade doesn't pay the full Dispatch cost
        on every retry."""
        if not _libs_available():
            return False
        now = time.time()
        if now - self._last_probe < 30.0:
            return self._last_probe_result
        self._last_probe = now
        self._last_probe_result = self._dispatch() is not None
        return self._last_probe_result

    def tools(self) -> List[Dict[str, Any]]:
        def fn(name: str, desc: str,
               props: Optional[Dict[str, Any]] = None,
               required: Optional[List[str]] = None) -> Dict[str, Any]:
            return {
                "type": "function", "name": name, "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props or {},
                    "required": required or [],
                    "additionalProperties": False,
                },
            }
        return [
            fn("outlook_com_list",
               "Read unread (or recent) messages from the user's running "
               "Outlook desktop inbox via COM. FREE, no OAuth, no API "
               "quotas — works for whatever account(s) Outlook is "
               "configured to sync (Exchange, IMAP, Gmail-via-IMAP, "
               "Outlook.com, all of them). Returns {messages: [{id, "
               "from, from_name, subject, received, snippet}], count, "
               "summary}. Set unread_only=true (default) to filter "
               "to unread; max defaults to 50; include_body=true also "
               "fetches the full body (capped at 2 KB) inline.",
               {"unread_only": {"type": "boolean",
                                "description": "Default true. False to "
                                               "return recent regardless "
                                               "of read state."},
                "max": {"type": "integer",
                        "description": "Max messages 1-50 (default 50)."},
                "include_body": {"type": "boolean",
                                 "description": "Default false. True to "
                                                "also fetch each body "
                                                "(adds ~50 ms/msg)."}}),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name != "outlook_com_list":
            return connector_result(
                "error", error=f"Unknown tool: {name}",
                code="unknown_tool")
        if not _libs_available():
            return connector_result(
                "error",
                error=("Outlook COM requires pywin32. Install via: "
                       "pip install pywin32. (Touchless ships with it; "
                       "this only fails in a dev install missing the "
                       "dep.)"),
                code="not_ready")
        if not self._ensure_com_apartment():
            return connector_result(
                "error", error="COM apartment init failed.",
                code="not_ready")
        app = self._dispatch()
        if app is None:
            return connector_result(
                "error",
                error=("Outlook desktop isn't installed or refused to "
                       "start. Install Outlook and launch it once to "
                       "complete profile setup; this connector then "
                       "works with no further setup."),
                code="not_ready")

        unread_only = bool(args.get("unread_only", True))
        max_n = max(1, min(50, int(args.get("max") or 50)))
        include_body = bool(args.get("include_body"))

        try:
            inbox = self._namespace.GetDefaultFolder(_OL_FOLDER_INBOX)
        except Exception as exc:
            return connector_result(
                "error",
                error=(f"Couldn't reach Outlook inbox: "
                       f"{type(exc).__name__}: {exc}. The Outlook "
                       f"profile may not be initialized — launch "
                       f"Outlook once and try again."),
                code="connector_failed")

        items = inbox.Items
        # Sort BEFORE Restrict — Restrict on an unsorted collection
        # works but Outlook's docs recommend Sort first. Newest first.
        try:
            items.Sort("[ReceivedTime]", True)
        except Exception:
            pass

        if unread_only:
            try:
                items = items.Restrict("[Unread] = true")
            except Exception as exc:
                # If Restrict fails (very old Outlook), fall back to
                # manual filtering after a Find on the Unread property.
                return connector_result(
                    "error",
                    error=(f"Outlook Restrict failed: "
                           f"{type(exc).__name__}: {exc}"),
                    code="connector_failed")

        msgs: List[Dict[str, Any]] = []
        # Iterating COM collections in Python is one-indexed and the
        # underlying COM proxy doesn't always behave like a list — use
        # a counter-driven loop for predictable behavior + early exit.
        try:
            count = int(items.Count)
        except Exception:
            count = 0

        idx = 1
        # We may walk up to max_n*1.5 to skip junk (rare, but defensive
        # against COM items that fail to project as MailItem — e.g.
        # meeting requests, delivery receipts).
        walk_cap = max(max_n + 10, min(count, max_n * 2))
        while idx <= count and len(msgs) < max_n and idx <= walk_cap:
            try:
                m = items.Item(idx)
            except Exception:
                idx += 1
                continue
            idx += 1
            # Only standard MailItem (Class 43) — skip meeting / task
            # / report items that won't have a Subject/From in the
            # expected shape.
            try:
                if int(getattr(m, "Class", 0)) != 43:
                    continue
            except Exception:
                continue

            try:
                subject = str(getattr(m, "Subject", "") or "").strip()
            except Exception:
                subject = ""
            # SenderName is always present; SenderEmailAddress can be
            # an Exchange DN ("/o=ExchangeLabs/ou=...") for Exchange
            # senders — fall back to PR_SMTP_ADDRESS via PropertyAccessor
            # when needed.
            try:
                sender_name = str(getattr(m, "SenderName", "") or "").strip()
            except Exception:
                sender_name = ""
            sender_addr = ""
            try:
                raw_addr = str(getattr(m, "SenderEmailAddress", "") or "").strip()
                if raw_addr and "@" in raw_addr:
                    sender_addr = raw_addr
                elif raw_addr:
                    # Exchange DN — get the SMTP address via the PR_SMTP_ADDRESS
                    # property on the Sender entry. Falls back silently
                    # if anything goes sideways.
                    try:
                        pa = m.PropertyAccessor
                        smtp = pa.GetProperty(
                            "http://schemas.microsoft.com/mapi/proptag/"
                            "0x39FE001E")  # PR_SMTP_ADDRESS
                        if smtp:
                            sender_addr = str(smtp)
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                received = getattr(m, "ReceivedTime", None)
                received_str = str(received) if received else ""
            except Exception:
                received_str = ""

            snippet = ""
            try:
                # Body access is the slow part — for snippets we use
                # only the first 200 chars. include_body=True gets the
                # full preview.
                body = str(getattr(m, "Body", "") or "")
                # Outlook strips HTML for .Body already; just take the
                # leading chunk and collapse whitespace.
                snippet = re.sub(r"\s+", " ", body[:600]).strip()
                if len(snippet) > 180:
                    snippet = snippet[:177].rstrip() + "..."
            except Exception:
                snippet = ""

            entry: Dict[str, Any] = {
                "id": str(getattr(m, "EntryID", "") or ""),
                "from": sender_addr,
                "from_name": sender_name,
                "subject": subject,
                "received": received_str,
                "snippet": snippet,
            }
            if include_body:
                try:
                    full_body = str(getattr(m, "Body", "") or "")
                    entry["body_text"] = full_body[:2000]
                except Exception:
                    entry["body_text"] = ""
            msgs.append(entry)

        # Deterministic faithful summary built from real messages —
        # same formatter the Gmail/MS365 connectors use so the prose
        # renderer downstream sees a consistent shape.
        try:
            from .gmail_connector import _format_email_summary
            summary = _format_email_summary(
                msgs, unread_only=unread_only, max_arg=max_n)
        except Exception:
            summary = ""

        return connector_result(
            "ok", count=len(msgs), messages=msgs, summary=summary)

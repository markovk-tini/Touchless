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
  - We cache the discovered inbox list for 60 s so repeat calls don't
    re-walk every Store's root folder (also a cross-process trip).

Multi-account note: `GetDefaultFolder(6)` only returns the *default*
profile's inbox. A user with Gmail (IMAP) added as a *secondary*
account in Outlook would have those messages invisible to us if we
relied on the default folder alone. We additionally iterate
`Namespace.Stores` and each store's root → "Inbox" subfolder so all
accounts contribute messages, then merge sorted by ReceivedTime.

Threading: COM is apartment-threaded. The connector calls
`pythoncom.CoInitialize()` defensively in case execute() is invoked
from a non-main thread (e.g. the tool executor's worker thread).
CoUninitialize is NOT called here — we want to stay in COM context
for the connector's lifetime.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import Connector, connector_result

# Outlook folder constants we use. From MSDN's OlDefaultFolders enum.
_OL_FOLDER_INBOX = 6

# How long to cache the discovered inbox list before re-walking Stores.
_INBOX_CACHE_TTL = 60.0


def _diag(msg: str) -> None:
    """Stderr diagnostic for dev visibility. Production users won't see
    this directly, but execute()'s returned error strings ALSO carry
    the same detail so the cascade can surface it."""
    try:
        sys.stderr.write(f"[outlook_com] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


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


def _process_is_elevated() -> Optional[bool]:
    """True if Touchless is running with admin/elevated rights, False if
    standard user, None if we can't tell (non-Windows, ctypes missing).
    Used to flag potential integrity-level mismatches with Outlook."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return None


class OutlookComConnector(Connector):
    """Read the user's Outlook desktop inbox via COM. Free, fast, no
    auth — works whenever Outlook is installed AND can run (i.e. has
    been launched at least once and its profile is set up).

    Reads from ALL configured accounts (default profile + every store
    with an Inbox subfolder) so Gmail-via-IMAP secondaries are visible."""

    id = "outlook_com"

    def __init__(self) -> None:
        self._app = None        # cached COM dispatch handle
        self._namespace = None  # cached MAPI namespace
        self._last_probe = 0.0  # debounce probes — Dispatch is slow on miss
        self._last_probe_result = False
        # Cached inbox folder COM proxies + the timestamp they were
        # discovered, so we don't pay the Stores-walk cost on each call.
        self._inbox_cache: List[Any] = []
        self._inbox_cache_at = 0.0

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
        except Exception as exc:
            _diag(f"dispatch failed: {type(exc).__name__}: {exc}")
            self._app = None
            self._namespace = None
            return None

    # ---- Multi-account inbox discovery -----------------------------------

    def _discover_inboxes(self) -> Tuple[List[Any], List[str]]:
        """Return (inboxes, diagnostics). Walks every Store the MAPI
        namespace knows about and collects its "Inbox" folder if one
        exists. Default inbox is always first. Skips duplicates by
        EntryID. Diagnostics is a list of short per-store status
        strings, useful for the user-facing error path.

        Cached for _INBOX_CACHE_TTL so a chatty caller doesn't re-walk
        every store on every email read."""
        now = time.time()
        if (self._inbox_cache and
                now - self._inbox_cache_at < _INBOX_CACHE_TTL):
            return self._inbox_cache, ["cache hit"]

        inboxes: List[Any] = []
        seen_ids: set = set()
        diags: List[str] = []

        # 1) Default profile inbox first — this is the historical path
        #    and it's the one most users expect to be primary.
        default_inbox = None
        try:
            default_inbox = self._namespace.GetDefaultFolder(_OL_FOLDER_INBOX)
            entry_id = str(getattr(default_inbox, "EntryID", "") or "")
            inboxes.append(default_inbox)
            if entry_id:
                seen_ids.add(entry_id)
            diags.append("default inbox ok")
        except Exception as exc:
            diags.append(
                f"default inbox failed: {type(exc).__name__}: {exc}")

        # 2) Walk each Store and try to find an "Inbox" subfolder under
        #    its root. Some stores (e.g. PST archives) have no Inbox at
        #    all — silently skip those. Stores is a 1-indexed COM
        #    collection.
        try:
            stores = self._namespace.Stores
            store_count = int(stores.Count)
        except Exception as exc:
            diags.append(
                f"stores enum failed: {type(exc).__name__}: {exc}")
            store_count = 0

        for i in range(1, store_count + 1):
            store_name = "?"
            try:
                store = stores.Item(i)
                try:
                    store_name = str(getattr(store, "DisplayName", "") or "?")
                except Exception:
                    pass
                root = store.GetRootFolder()
                # Try named "Inbox" subfolder. Outlook localizes folder
                # names in some installs (e.g. "Posteingang" in German),
                # so if "Inbox" misses, scan root subfolders for one
                # whose DefaultItemType is 0 (olMailItem) and whose
                # FolderClass is "IPF.Note" + a name that maps to inbox.
                inbox = None
                try:
                    inbox = root.Folders("Inbox")
                except Exception:
                    pass
                if inbox is None:
                    # Fallback: scan root folders for a mail folder
                    # named like an inbox in common locales.
                    try:
                        for j in range(1, int(root.Folders.Count) + 1):
                            sub = root.Folders.Item(j)
                            try:
                                cls = str(getattr(sub, "FolderClass", "") or "")
                                nm = str(getattr(sub, "Name", "") or "").lower()
                                if cls.startswith("IPF.Note") and (
                                    nm in ("inbox", "posteingang",
                                           "bandeja de entrada",
                                           "boîte de réception",
                                           "boite de reception",
                                           "posta in arrivo",
                                           "входящие", "受信トレイ",
                                           "收件箱", "수신함",
                                           "caixa de entrada")
                                ):
                                    inbox = sub
                                    break
                            except Exception:
                                continue
                    except Exception:
                        pass
                if inbox is None:
                    diags.append(f"store '{store_name}': no inbox")
                    continue
                entry_id = str(getattr(inbox, "EntryID", "") or "")
                if entry_id and entry_id in seen_ids:
                    diags.append(f"store '{store_name}': dupe (skipped)")
                    continue
                inboxes.append(inbox)
                if entry_id:
                    seen_ids.add(entry_id)
                diags.append(f"store '{store_name}': ok")
            except Exception as exc:
                diags.append(
                    f"store #{i} ('{store_name}'): "
                    f"{type(exc).__name__}: {exc}")
                continue

        self._inbox_cache = inboxes
        self._inbox_cache_at = now
        return inboxes, diags

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
               "Outlook.com, all of them — primary AND secondary). "
               "Returns {messages: [{id, from, from_name, subject, "
               "received, snippet}], count, summary}. Set "
               "unread_only=true (default) to filter to unread; max "
               "defaults to 50; include_body=true also fetches the full "
               "body (capped at 2 KB) inline.",
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
            _diag("libs=False — pywin32/pythoncom missing")
            return connector_result(
                "error",
                error=("Outlook desktop COM requires pywin32, which "
                       "isn't loaded. Install via: `pip install "
                       "pywin32`. (Touchless's installer build ships "
                       "with it — this only fails in a dev install "
                       "missing the dep.)"),
                code="not_ready")
        if not self._ensure_com_apartment():
            _diag("CoInitialize failed")
            return connector_result(
                "error",
                error=("Couldn't initialize the Windows COM apartment "
                       "for the calling thread. This usually means "
                       "pywin32 is installed but corrupt — try `pip "
                       "install --upgrade --force-reinstall pywin32`."),
                code="not_ready")
        app = self._dispatch()
        if app is None:
            elev = _process_is_elevated()
            extra = ""
            if elev is True:
                # Elevated Touchless can't see a non-elevated Outlook's
                # COM server (and vice versa) due to integrity-level
                # isolation. Calling Dispatch from elevated either
                # starts a NEW empty Outlook (visible-but-wrong) or
                # silently fails. Surface it as actionable.
                extra = (" Touchless is running ELEVATED (admin) — "
                         "Windows blocks elevated processes from "
                         "automating a normal-user Outlook. Relaunch "
                         "Touchless WITHOUT 'Run as administrator' "
                         "and try again.")
            _diag(f"dispatch=None elevated={elev}")
            return connector_result(
                "error",
                error=("Outlook desktop COM Dispatch returned None — "
                       "either Outlook isn't installed, the install is "
                       "broken, or its profile hasn't been initialized. "
                       "Launch Outlook once manually to complete profile "
                       "setup; this connector then works with no further "
                       "setup." + extra),
                code="not_ready")

        # Integrity-level sanity warning when we DID get a handle but
        # the user is elevated. The Dispatch may have started a *new*
        # empty Outlook instead of attaching to their running one.
        elev = _process_is_elevated()
        if elev is True:
            try:
                inspectors = int(app.Inspectors.Count)
                explorers = int(app.Explorers.Count)
                # A fresh Outlook with no UI windows usually means we
                # spawned our own ghost — warn the model so it can
                # tell the user.
                if inspectors == 0 and explorers == 0:
                    _diag("elevated dispatch — possible ghost Outlook "
                          f"(insp={inspectors} expl={explorers})")
            except Exception:
                pass

        unread_only = bool(args.get("unread_only", True))
        max_n = max(1, min(50, int(args.get("max") or 50)))
        include_body = bool(args.get("include_body"))

        # Discover every account's inbox we can reach.
        inboxes, walk_diags = self._discover_inboxes()
        _diag(
            f"libs=True dispatch=ok stores_walked={len(walk_diags)} "
            f"inboxes={len(inboxes)} (" + "; ".join(walk_diags) + ")")

        if not inboxes:
            return connector_result(
                "error",
                error=("Couldn't reach any Outlook inbox. Detail: " +
                       "; ".join(walk_diags) +
                       ". The Outlook profile may not be initialized — "
                       "launch Outlook once and try again."),
                code="connector_failed")

        # Per-inbox: filter to unread (if requested), sort newest-first,
        # then merge across all accounts and trim to max_n.
        merged: List[Tuple[Any, Dict[str, Any]]] = []
        per_inbox_errors: List[str] = []
        # Cap per inbox so a single huge mailbox doesn't starve others;
        # we then re-trim after the global sort.
        per_inbox_cap = max(max_n, 20)

        for inbox in inboxes:
            inbox_name = "?"
            try:
                inbox_name = str(getattr(inbox, "Name", "") or "?")
            except Exception:
                pass
            try:
                items = inbox.Items
                # Sort BEFORE Restrict — Restrict on an unsorted
                # collection works but Outlook's docs recommend Sort
                # first. Newest first.
                try:
                    items.Sort("[ReceivedTime]", True)
                except Exception:
                    pass
                if unread_only:
                    try:
                        items = items.Restrict("[Unread] = true")
                    except Exception as exc:
                        per_inbox_errors.append(
                            f"'{inbox_name}': Restrict failed "
                            f"({type(exc).__name__})")
                        continue

                try:
                    count = int(items.Count)
                except Exception:
                    count = 0
            except Exception as exc:
                per_inbox_errors.append(
                    f"'{inbox_name}': items failed "
                    f"({type(exc).__name__}: {exc})")
                continue

            idx = 1
            collected = 0
            walk_cap = max(per_inbox_cap + 10,
                           min(count, per_inbox_cap * 2))
            while (idx <= count and collected < per_inbox_cap and
                   idx <= walk_cap):
                try:
                    m = items.Item(idx)
                except Exception:
                    idx += 1
                    continue
                idx += 1
                # Only standard MailItem (Class 43) — skip meeting /
                # task / report items.
                try:
                    if int(getattr(m, "Class", 0)) != 43:
                        continue
                except Exception:
                    continue

                try:
                    subject = str(getattr(m, "Subject", "") or "").strip()
                except Exception:
                    subject = ""
                try:
                    sender_name = str(
                        getattr(m, "SenderName", "") or "").strip()
                except Exception:
                    sender_name = ""
                sender_addr = ""
                try:
                    raw_addr = str(
                        getattr(m, "SenderEmailAddress", "") or "").strip()
                    if raw_addr and "@" in raw_addr:
                        sender_addr = raw_addr
                    elif raw_addr:
                        # Exchange DN — get the SMTP address via the
                        # PR_SMTP_ADDRESS property.
                        try:
                            pa = m.PropertyAccessor
                            smtp = pa.GetProperty(
                                "http://schemas.microsoft.com/mapi/"
                                "proptag/0x39FE001E")  # PR_SMTP_ADDRESS
                            if smtp:
                                sender_addr = str(smtp)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Capture the raw COM ReceivedTime (a PyTime) for the
                # cross-account merge sort, and a string copy for the
                # result payload.
                received_raw = None
                received_str = ""
                try:
                    received_raw = getattr(m, "ReceivedTime", None)
                    received_str = (
                        str(received_raw) if received_raw else "")
                except Exception:
                    pass

                snippet = ""
                try:
                    body = str(getattr(m, "Body", "") or "")
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
                merged.append((received_raw, entry))
                collected += 1

        if per_inbox_errors:
            _diag("per-inbox issues: " + "; ".join(per_inbox_errors))

        # Cross-account merge sort: newest ReceivedTime first. Entries
        # with no timestamp go last so anything we couldn't timestamp
        # falls after real messages. str(PyTime) is ISO-ish
        # ('YYYY-MM-DD HH:MM:SS') so lexical compare matches chronology.
        valid = [p for p in merged if p[0] is not None]
        invalid = [p for p in merged if p[0] is None]
        valid.sort(key=lambda p: str(p[0]), reverse=True)
        merged_sorted = valid + invalid

        msgs: List[Dict[str, Any]] = [
            entry for _rt, entry in merged_sorted[:max_n]
        ]

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

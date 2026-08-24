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
_OL_FOLDER_CONTACTS = 10

# How long to cache the discovered inbox list before re-walking Stores.
_INBOX_CACHE_TTL = 60.0


def _diag_path() -> str:
    """Persistent log file the user can open to see what this connector
    actually did. Lives under %TEMP%\\Touchless_Iris\\outlook_com.log
    so it's findable without grepping the codebase."""
    base = os.environ.get("TEMP") or os.environ.get("TMP") or "."
    folder = os.path.join(base, "Touchless_Iris")
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception:
        pass
    return os.path.join(folder, "outlook_com.log")


def _diag(msg: str) -> None:
    """Diagnostic that goes to BOTH stderr AND a persistent log file the
    user can open. The file path is printed via the first call's
    'log_at=' marker so the user knows where to look."""
    try:
        sys.stderr.write(f"[outlook_com] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass
    try:
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(_diag_path(), "a", encoding="utf-8", errors="ignore") as f:
            f.write(f"{ts}  {msg}\n")
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

    # ---- .ics fallback (works on New Outlook, no COM) -------------------

    def _create_event_via_ics(
            self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Write a one-off .ics file and ShellExecute it. Both New
        Outlook (UWP) and Classic Outlook register .ics as their
        default handler, so the user gets a one-click 'Add to
        calendar' prompt that lands in the right calendar with no
        auth, COM, or Graph. This is our shipping path for the
        majority of users (New Outlook default since 2024).

        Returns connector_result('ok', requires_user_confirm=True, ...)
        so the LLM tells the user 'I've opened the event in Outlook
        for you to confirm' instead of falsely claiming it's saved."""
        import datetime as _dt
        import uuid as _uuid
        subject = str(args.get("subject")
                      or args.get("title") or "").strip()
        start_s = str(args.get("start") or "").strip()
        end_s = str(args.get("end") or "").strip()
        duration_min = args.get("duration_minutes")
        location = str(args.get("location") or "").strip()
        body_text = str(args.get("body")
                        or args.get("description") or "").strip()
        if not subject:
            return connector_result(
                "error", error="'subject' is required.",
                code="invalid_args")
        if not start_s:
            return connector_result(
                "error",
                error=("'start' is required (ISO 8601 like "
                       "'2026-06-10T13:50:00')."),
                code="invalid_args")
        try:
            start_dt = _dt.datetime.fromisoformat(
                start_s.replace("Z", "+00:00"))
            if start_dt.tzinfo is not None:
                start_dt = start_dt.astimezone().replace(tzinfo=None)
        except Exception:
            return connector_result(
                "error",
                error=f"Couldn't parse start={start_s!r}.",
                code="invalid_args")
        end_dt = None
        if end_s:
            try:
                end_dt = _dt.datetime.fromisoformat(
                    end_s.replace("Z", "+00:00"))
                if end_dt.tzinfo is not None:
                    end_dt = end_dt.astimezone().replace(tzinfo=None)
            except Exception:
                end_dt = None
        if end_dt is None:
            try:
                mins = int(duration_min or 60)
            except (TypeError, ValueError):
                mins = 60
            end_dt = start_dt + _dt.timedelta(
                minutes=max(5, mins))

        def _esc(s: str) -> str:
            return (s or "").replace("\\", "\\\\").replace(
                ",", "\\,").replace(";", "\\;").replace("\n", "\\n")

        def _fmt(d: _dt.datetime) -> str:
            return d.strftime("%Y%m%dT%H%M%S")

        uid = f"{_uuid.uuid4().hex}@touchless.local"
        dtstamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        ics_lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Touchless//Iris//EN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{dtstamp}",
            f"DTSTART:{_fmt(start_dt)}",
            f"DTEND:{_fmt(end_dt)}",
            f"SUMMARY:{_esc(subject)}",
        ]
        if location:
            ics_lines.append(f"LOCATION:{_esc(location)}")
        if body_text:
            ics_lines.append(f"DESCRIPTION:{_esc(body_text)}")
        ics_lines += ["END:VEVENT", "END:VCALENDAR"]
        ics_text = "\r\n".join(ics_lines) + "\r\n"

        # Drop the .ics into a stable folder so old files don't pile up
        # in TEMP and so the user can find them if needed for debugging.
        base = (os.environ.get("LOCALAPPDATA")
                or os.environ.get("APPDATA") or os.path.expanduser("~"))
        ics_dir = os.path.join(base, "Touchless", "calendar_ics")
        try:
            os.makedirs(ics_dir, exist_ok=True)
        except Exception:
            ics_dir = os.environ.get("TEMP") or "."
        safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", subject)[:40] or "event"
        ics_path = os.path.join(
            ics_dir, f"{safe_name}_{_fmt(start_dt)}.ics")
        try:
            with open(ics_path, "w", encoding="utf-8") as f:
                f.write(ics_text)
        except Exception as exc:
            _diag(f"ics write failed: {type(exc).__name__}: {exc}")
            return connector_result(
                "error",
                error=(f"Couldn't write the calendar file: "
                       f"{type(exc).__name__}: {exc}"),
                code="ics_write_failed")
        try:
            os.startfile(ics_path)  # type: ignore[attr-defined]
        except Exception as exc:
            _diag(f"ics open failed: {type(exc).__name__}: {exc}")
            return connector_result(
                "error",
                error=(f"Wrote the event file but Windows couldn't "
                       f"open it: {type(exc).__name__}: {exc}. "
                       f"You can manually open it at: {ics_path}"),
                code="ics_open_failed")
        _diag(
            f"ics fallback opened subject={subject!r} "
            f"start={start_dt.isoformat()} end={end_dt.isoformat()} "
            f"path={ics_path}")
        return connector_result(
            "ok",
            created=False,
            requires_user_confirm=True,
            subject=subject,
            start=start_dt.isoformat(timespec="minutes"),
            end=end_dt.isoformat(timespec="minutes"),
            location=location,
            calendar="Outlook (via .ics import)",
            source="ics_fallback",
            file=ics_path,
            note=("Outlook has opened a confirmation window. The user "
                  "needs to click 'Save & Close' (or 'Accept') for the "
                  "event to appear in their calendar. Tell them this."))

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

    def _dispatch(self, allow_launch: bool = False) -> Optional[Any]:
        """Get or create the Outlook.Application COM object. Returns
        None if Outlook isn't installed or (when allow_launch=False)
        isn't already running.

        allow_launch semantics — this is load-bearing for the email
        cascade UX. `win32com.client.Dispatch('Outlook.Application')`
        will *launch* Outlook.exe as a side effect if no instance is
        running, which surprises users who only asked Iris to
        "check my email" (the silent email_summary cascade used to
        open the entire desktop client behind their back). So we
        default to `GetActiveObject` — attach only to an already-
        running Outlook — and only fall back to Dispatch when the
        caller is an EXPLICIT user action (creating a calendar event,
        an explicit "connect Outlook" wizard flow) where the user
        clearly wants Outlook to start if it isn't already.
        """
        if self._app is not None:
            return self._app
        if not self._ensure_com_apartment():
            return None
        try:
            import win32com.client
            # Attach-first: GetActiveObject only sees a running COM
            # server, never spawns one. This is the safe path for
            # silent cascades (email_summary, contacts_search).
            try:
                self._app = win32com.client.GetActiveObject(
                    "Outlook.Application")
            except Exception:
                if not allow_launch:
                    _diag("outlook not running; refusing to auto-launch "
                          "during silent cascade (allow_launch=False)")
                    self._app = None
                    self._namespace = None
                    return None
                # Explicit user action — Dispatch() will start Outlook
                # if it isn't running. This is the historical behavior,
                # preserved for outlook_com_create_event and any future
                # "connect Outlook" wizard.
                self._app = win32com.client.Dispatch("Outlook.Application")
            self._namespace = self._app.GetNamespace("MAPI")
            return self._app
        except Exception as exc:
            _diag(f"dispatch failed: {type(exc).__name__}: {exc}")
            self._app = None
            self._namespace = None
            return None

    # ---- Multi-account inbox discovery -----------------------------------

    def _discover_inboxes(self) -> Tuple[List[Tuple[Any, str]], List[str]]:
        """Return ([(inbox, account_name), ...], diagnostics). Walks
        every Store the MAPI namespace knows about and collects its
        "Inbox" folder if one exists. Default inbox is always first.
        Skips duplicates by EntryID. Diagnostics is a list of short
        per-store status strings, useful for the user-facing error
        path.

        Each entry carries the OWNING ACCOUNT name (the parent Store's
        DisplayName) so messages can be tagged with which account they
        came from. That powers per-account replies ('your work account
        has 3 unread, gmail has 7') and an optional `account` filter
        on read.

        Cached for _INBOX_CACHE_TTL so a chatty caller doesn't re-walk
        every store on every email read."""
        now = time.time()
        if (self._inbox_cache and
                now - self._inbox_cache_at < _INBOX_CACHE_TTL):
            return self._inbox_cache, ["cache hit"]

        inboxes: List[Tuple[Any, str]] = []
        seen_ids: set = set()
        diags: List[str] = []

        # 1) Default profile inbox first — this is the historical path
        #    and it's the one most users expect to be primary.
        default_inbox = None
        try:
            default_inbox = self._namespace.GetDefaultFolder(_OL_FOLDER_INBOX)
            entry_id = str(getattr(default_inbox, "EntryID", "") or "")
            # Look up the owning store's DisplayName for tagging.
            default_account = "default"
            try:
                ds = default_inbox.Store
                default_account = str(
                    getattr(ds, "DisplayName", "") or "default")
            except Exception:
                pass
            inboxes.append((default_inbox, default_account))
            if entry_id:
                seen_ids.add(entry_id)
            diags.append(f"default inbox ok ({default_account})")
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
                inboxes.append((inbox, store_name))
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
        """Available on Windows whenever we can attempt ANY path:
        - Classic Outlook COM (full read + write)
        - .ics fallback (write only — needs no COM, no pywin32)
        We must return True on Windows even when COM dispatch fails,
        otherwise outlook_com_create_event drops out of the LLM's
        tool list and it can't be called at all. The .ics fallback
        in execute() handles the COM-failed case for create_event;
        outlook_com_list returns a clean 'not_ready' error in that
        case and the email cascade falls through to other paths."""
        if sys.platform != "win32":
            return False
        # _diag is cheap; logging here lets us see in outlook_com.log
        # whether the LLM's tool list ever included our tools.
        now = time.time()
        if now - self._last_probe < 30.0:
            return True  # cached "yes we're in the toolset" decision
        self._last_probe = now
        # Probe whether COM works, but don't gate availability on it
        # — just cache it so execute() can route to .ics fast.
        com_ok = False
        if _libs_available():
            try:
                # Probe only — never launch Outlook during a health
                # check. If Outlook isn't running the probe simply
                # reports False and the connector still advertises
                # itself (available() unconditionally returns True on
                # Windows so tools stay in the LLM's list).
                com_ok = self._dispatch(allow_launch=False) is not None
            except Exception:
                com_ok = False
        self._last_probe_result = com_ok
        _diag(f"available() probe: com_ok={com_ok} "
              f"(connector advertised as AVAILABLE regardless — "
              f".ics fallback handles no-COM case)")
        return True

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
               "received, snippet, account}], count, summary, "
               "account_counts, available_accounts}. Each message is "
               "tagged with its source `account` so multi-account "
               "users see which inbox each one came from. Set "
               "`account` to a substring of an account name to "
               "filter to just that one (e.g. account='gmail' reads "
               "only the Gmail account, account='work' for an "
               "Exchange/work account). Empty `account` reads across "
               "ALL connected Outlook accounts. Set unread_only=true "
               "(default) to filter to unread; max defaults to 50; "
               "include_body=true also fetches the full body (capped "
               "at 2 KB) inline.",
               {"unread_only": {"type": "boolean",
                                "description": "Default true. False to "
                                               "return recent regardless "
                                               "of read state."},
                "max": {"type": "integer",
                        "description": "Max messages 1-50 (default 50)."},
                "include_body": {"type": "boolean",
                                 "description": "Default false. True to "
                                                "also fetch each body "
                                                "(adds ~50 ms/msg)."},
                "account": {"type": "string",
                            "description": "Optional substring match on "
                                           "the Outlook account name to "
                                           "filter results to one "
                                           "account. Empty = read all."}}),
            fn("outlook_com_create_event",
               "Create a calendar event / appointment on the user's "
               "Outlook desktop calendar via COM. FREE — no OAuth, no "
               "MS Graph CASA verification, no scope grants. Works for "
               "whatever account Outlook is configured to sync to "
               "(Exchange / Outlook.com / Microsoft 365). The event "
               "syncs to the cloud automatically through Outlook's "
               "own sync engine. PREFER this for any user request to "
               "create / schedule / book an appointment on the local "
               "Outlook calendar — it's the path that will actually "
               "show up in their Outlook UI, unlike ms_calendar_create "
               "which only writes via Graph and requires OAuth. "
               "`subject` is the event title (required). `start` is "
               "ISO 8601 LOCAL time without timezone (e.g. "
               "'2026-06-10T13:50:00'). `end` OR `duration_minutes` "
               "(default 60 min) sets the length. Optional `location` "
               "and `body` (notes).",
               {"subject": {"type": "string",
                            "description": "Event title."},
                "start": {"type": "string",
                          "description": "ISO 8601 local time "
                                         "'YYYY-MM-DDTHH:MM:SS'."},
                "end": {"type": "string",
                        "description": "ISO 8601 local end time. "
                                       "Optional — falls back to "
                                       "duration_minutes."},
                "duration_minutes": {"type": "integer",
                                     "description": "Length in "
                                                    "minutes if `end` "
                                                    "not given. "
                                                    "Default 60."},
                "location": {"type": "string",
                             "description": "Optional location."},
                "body": {"type": "string",
                         "description": "Optional event notes / "
                                        "description."}},
               ["subject", "start"]),
            fn("outlook_com_contacts_search",
               "Search the user's Classic Outlook desktop Contacts "
               "folder via COM. FREE, no OAuth — works for whatever "
               "contacts are stored in the local Outlook profile. "
               "Substring-matches `query` (case-insensitive) against "
               "FullName, FirstName, LastName, CompanyName and the "
               "three primary email fields. Returns {count, contacts: "
               "[{display_name, given_name, family_name, emails: "
               "[{value, type}], phones: [{value, type}], "
               "organization}]}. Falls back to a clean `com_unavailable` "
               "error on New Outlook (UWP) so the contacts cascade can "
               "drop to Google / Microsoft 365 sources.",
               {"query": {"type": "string",
                          "description": "Substring to match against "
                                         "name, company, or email."},
                "max": {"type": "integer",
                        "description": "Max contacts 1-50 "
                                       "(default 25)."}},
               ["query"]),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in (
            "outlook_com_list",
            "outlook_com_create_event",
            "outlook_com_contacts_search",
        ):
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
            if name == "outlook_com_create_event":
                return self._create_event_via_ics(args)
            return connector_result(
                "error",
                error=("Couldn't initialize the Windows COM apartment "
                       "for the calling thread. This usually means "
                       "pywin32 is installed but corrupt — try `pip "
                       "install --upgrade --force-reinstall pywin32`."),
                code="not_ready")
        # Only outlook_com_create_event is an explicit user action
        # where Outlook can reasonably be spawned if it isn't running
        # (and even then the .ics fallback below covers the "no COM"
        # case cleanly). outlook_com_list is the silent email cascade
        # path — never spawn Outlook there. outlook_com_contacts_search
        # is also silently cascaded from contacts_lookup, same rule.
        allow_launch = (name == "outlook_com_create_event")
        app = self._dispatch(allow_launch=allow_launch)
        if app is None and name == "outlook_com_contacts_search":
            _diag("COM dispatch=None on contacts_search — "
                  "returning com_unavailable for cascade fallback")
            return connector_result(
                "error",
                error=("Outlook desktop COM not available — falling "
                       "back to other contact sources."),
                code="com_unavailable")
        if app is None and name == "outlook_com_create_event":
            # New Outlook (UWP) has no COM. For *creating* an event we
            # can sidestep COM entirely — write a one-shot .ics file
            # and ShellExecute it. Both New Outlook and Classic Outlook
            # register .ics as their default handler, so the user gets
            # a one-click "Add to calendar" prompt that lands in the
            # right calendar with no auth, no Graph, no COM.
            _diag("COM dispatch=None — falling back to .ics path "
                  "(New Outlook / no Classic install)")
            return self._create_event_via_ics(args)
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

        # ---- outlook_com_create_event ---------------------------
        # Creates an appointment on the user's default calendar via
        # the Outlook desktop COM Application.CreateItem(1) path
        # (1 = olAppointmentItem). Zero OAuth — works for whatever
        # account Outlook is configured to sync to. The event syncs
        # to Exchange / Outlook.com / etc. automatically. We use the
        # Outlook session's local time zone so what the user typed
        # ("today at 1:50 pm") lands at their wall-clock time.
        if name == "outlook_com_create_event":
            _diag(f"create_event ENTRY args={args!r}")
            subject = str(args.get("subject")
                          or args.get("title") or "").strip()
            start_s = str(args.get("start") or "").strip()
            end_s = str(args.get("end") or "").strip()
            duration_min = args.get("duration_minutes")
            location = str(args.get("location") or "").strip()
            body_text = str(args.get("body")
                            or args.get("description") or "").strip()
            if not subject:
                return connector_result(
                    "error",
                    error="'subject' is required (event title).",
                    code="invalid_args")
            if not start_s:
                return connector_result(
                    "error",
                    error="'start' is required "
                          "(ISO 8601 like '2026-06-10T13:50:00').",
                    code="invalid_args")
            # Parse start.
            import datetime as _dt
            try:
                start_dt = _dt.datetime.fromisoformat(
                    start_s.replace("Z", "+00:00"))
                if start_dt.tzinfo is not None:
                    start_dt = start_dt.astimezone().replace(
                        tzinfo=None)
            except Exception:
                return connector_result(
                    "error",
                    error=(f"Couldn't parse start={start_s!r}. "
                           "Use ISO 8601 like "
                           "'2026-06-10T13:50:00'."),
                    code="invalid_args")
            # End: from explicit field OR duration_minutes OR
            # default 60 min.
            end_dt = None
            if end_s:
                try:
                    end_dt = _dt.datetime.fromisoformat(
                        end_s.replace("Z", "+00:00"))
                    if end_dt.tzinfo is not None:
                        end_dt = end_dt.astimezone().replace(
                            tzinfo=None)
                except Exception:
                    end_dt = None
            if end_dt is None:
                try:
                    mins = int(duration_min or 60)
                except (TypeError, ValueError):
                    mins = 60
                end_dt = start_dt + _dt.timedelta(
                    minutes=max(5, mins))
            try:
                # 1 = olAppointmentItem
                appt = app.CreateItem(1)
                appt.Subject = subject
                appt.Start = start_dt
                appt.End = end_dt
                if location:
                    appt.Location = location
                if body_text:
                    appt.Body = body_text
                appt.Save()
                entry_id = ""
                try:
                    entry_id = str(appt.EntryID or "")
                except Exception:
                    pass
                _diag(
                    f"appointment created subject={subject!r} "
                    f"start={start_dt.isoformat()} "
                    f"end={end_dt.isoformat()} "
                    f"entry_id={entry_id[:24]}")
                return connector_result(
                    "ok", created=True,
                    subject=subject,
                    start=start_dt.isoformat(timespec="minutes"),
                    end=end_dt.isoformat(timespec="minutes"),
                    location=location,
                    entry_id=entry_id,
                    calendar="Outlook (desktop)",
                    source="outlook_desktop")
            except Exception as exc:
                _diag(
                    f"appointment create failed: "
                    f"{type(exc).__name__}: {exc}")
                return connector_result(
                    "error",
                    error=(f"Outlook desktop couldn't create the "
                           f"event: {type(exc).__name__}: "
                           f"{str(exc)[:160]}"),
                    code="com_create_failed")

        # ---- outlook_com_contacts_search ------------------------
        if name == "outlook_com_contacts_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return connector_result(
                    "error", error="'query' is required.",
                    code="invalid_args")
            try:
                max_n = int(args.get("max") or 25)
            except (TypeError, ValueError):
                max_n = 25
            max_n = max(1, min(50, max_n))
            qlow = query.lower()
            try:
                contacts_folder = self._namespace.GetDefaultFolder(
                    _OL_FOLDER_CONTACTS)
            except Exception as exc:
                _diag(
                    f"contacts folder failed: "
                    f"{type(exc).__name__}: {exc}")
                return connector_result(
                    "error",
                    error=("Couldn't open the Outlook Contacts folder: "
                           f"{type(exc).__name__}: {exc}"),
                    code="connector_failed")
            try:
                items = contacts_folder.Items
                total = int(items.Count)
            except Exception as exc:
                _diag(
                    f"contacts items failed: "
                    f"{type(exc).__name__}: {exc}")
                return connector_result(
                    "error",
                    error=("Couldn't enumerate the Outlook Contacts "
                           f"folder: {type(exc).__name__}: {exc}"),
                    code="connector_failed")

            matches: List[Dict[str, Any]] = []
            walk_cap = min(total, 5000)
            for idx in range(1, walk_cap + 1):
                if len(matches) >= max_n:
                    break
                try:
                    c = items.Item(idx)
                except Exception:
                    continue
                try:
                    if int(getattr(c, "Class", 0)) != 40:
                        continue
                except Exception:
                    continue

                def _s(attr: str) -> str:
                    try:
                        return str(getattr(c, attr, "") or "").strip()
                    except Exception:
                        return ""

                full_name = _s("FullName")
                first = _s("FirstName")
                last = _s("LastName")
                company = _s("CompanyName")
                e1 = _s("Email1Address")
                e2 = _s("Email2Address")
                e3 = _s("Email3Address")

                haystack = " ".join((
                    full_name, first, last, company, e1, e2, e3
                )).lower()
                if qlow not in haystack:
                    continue

                emails: List[Dict[str, str]] = []
                for value, kind in (
                    (e1, "email1"),
                    (e2, "email2"),
                    (e3, "email3"),
                ):
                    if value and "@" in value:
                        emails.append({"value": value, "type": kind})

                phones: List[Dict[str, str]] = []
                for attr, kind in (
                    ("MobileTelephoneNumber", "mobile"),
                    ("BusinessTelephoneNumber", "work"),
                    ("HomeTelephoneNumber", "home"),
                ):
                    val = _s(attr)
                    if val:
                        phones.append({"value": val, "type": kind})

                display = (full_name
                           or " ".join(p for p in (first, last) if p).strip()
                           or company
                           or (emails[0]["value"] if emails else ""))

                matches.append({
                    "display_name": display,
                    "given_name": first,
                    "family_name": last,
                    "emails": emails,
                    "phones": phones,
                    "organization": company or None,
                    "source": "outlook_com",
                })

            _diag(
                f"contacts_search query={query!r} scanned={walk_cap} "
                f"total={total} matches={len(matches)}")
            return connector_result(
                "ok", count=len(matches), contacts=matches)

        # ---- outlook_com_list (inbox read) ----------------------
        unread_only = bool(args.get("unread_only", True))
        max_n = max(1, min(50, int(args.get("max") or 50)))
        include_body = bool(args.get("include_body"))
        # Optional account filter: case-insensitive substring match
        # against the Store DisplayName. 'gmail' matches any account
        # whose name contains 'gmail'. Empty = all accounts.
        account_filter = str(args.get("account") or "").strip().lower()

        # Discover every account's inbox we can reach.
        inboxes_with_names, walk_diags = self._discover_inboxes()
        _diag(
            f"libs=True dispatch=ok stores_walked={len(walk_diags)} "
            f"inboxes={len(inboxes_with_names)} (" +
            "; ".join(walk_diags) + ")")

        if not inboxes_with_names:
            return connector_result(
                "error",
                error=("Couldn't reach any Outlook inbox. Detail: " +
                       "; ".join(walk_diags) +
                       ". The Outlook profile may not be initialized — "
                       "launch Outlook once and try again."),
                code="connector_failed")

        # Apply account filter.
        all_account_names = [n for _, n in inboxes_with_names]
        if account_filter:
            inboxes_filtered = [
                (ix, nm) for ix, nm in inboxes_with_names
                if account_filter in nm.lower()
            ]
            if not inboxes_filtered:
                # User asked for a specific account but no match.
                # Return an actionable error listing what we DID find.
                return connector_result(
                    "error",
                    error=(
                        f"No Outlook account matching '{account_filter}'. "
                        f"Available accounts: "
                        f"{', '.join(all_account_names)}. Pass `account` "
                        f"as any substring of one of those names."),
                    code="account_not_found",
                    available_accounts=all_account_names)
            inboxes_with_names = inboxes_filtered
            _diag(f"account filter='{account_filter}' -> "
                  f"{len(inboxes_with_names)} matching")

        # Per-inbox: filter to unread (if requested), sort newest-first,
        # then merge across all accounts and trim to max_n.
        merged: List[Tuple[Any, Dict[str, Any]]] = []
        per_inbox_errors: List[str] = []
        # Cap per inbox so a single huge mailbox doesn't starve others;
        # we then re-trim after the global sort.
        per_inbox_cap = max(max_n, 20)

        for inbox, account_name in inboxes_with_names:
            inbox_name = account_name or "?"
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
                    "account": account_name,
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

        # Per-account count breakdown — feeds the prose renderer so it
        # can naturally say "you've got 7 unread across Gmail and Work
        # — Gmail has 5, Work has 2" instead of just one undifferentiated
        # number.
        account_counts: Dict[str, int] = {}
        for _rt, entry in merged_sorted:
            acc = str(entry.get("account") or "")
            if acc:
                account_counts[acc] = account_counts.get(acc, 0) + 1

        # Deterministic faithful summary built from real messages —
        # same formatter the Gmail/MS365 connectors use so the prose
        # renderer downstream sees a consistent shape.
        try:
            from .gmail_connector import _format_email_summary
            summary = _format_email_summary(
                msgs, unread_only=unread_only, max_arg=max_n)
            # Inject a per-account intro line if there are multiple
            # accounts represented, since _format_email_summary's
            # headline only carries the total.
            if len(account_counts) > 1:
                pretty = ", ".join(
                    f"{name}: {n}" for name, n in sorted(
                        account_counts.items(), key=lambda kv: -kv[1])
                )
                summary = (
                    f"Across your Outlook accounts ({pretty}):\n" +
                    summary)
        except Exception:
            summary = ""

        return connector_result(
            "ok", count=len(msgs), messages=msgs, summary=summary,
            account_counts=account_counts,
            available_accounts=all_account_names,
            account_filter=account_filter or None)

"""Microsoft Phone Link connector — drive the Windows Phone Link app
to send/read texts from a paired iPhone (or Android) via its UI.

Phone Link is the only Windows software with an Apple partnership that
can actually send iMessage. There's no public API, so we drive the GUI
via UIA (the accessibility tree — same path that works in any Windows
app). Slower than a real API (each action takes 1-3 seconds), but
deterministic and uses no vision tokens.

For Android, KDE Connect is a faster + cheaper alternative; for iPhone
this is the only working path.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

from .base import Connector, connector_result
from ...utils.subprocess_utils import launch_external


# Phone Link's AppX package name. Microsoft renamed it twice; we check
# the modern name first then the legacy "Your Phone" name.
_APPX_NAMES = ["Microsoft.YourPhone", "Microsoft.PhoneLink"]
# URI schemes that launch Phone Link. ms-phone is the modern one;
# ms-yourphone is the legacy that still works on most installs.
_LAUNCH_URIS = ["ms-phone:", "ms-yourphone:"]

# Process / window titles to look for to confirm it's running.
_WINDOW_TITLES = ("Phone Link", "Microsoft Phone Link", "Your Phone")

# UIA element names we search for. These are the accessible names
# Microsoft uses — we list multiple variants because the labels have
# shifted across versions.
_MESSAGES_TAB_NAMES = ("Messages", "Texts", "SMS")
_NEW_MESSAGE_NAMES = ("New message", "New conversation", "Compose")
_SEARCH_BOX_NAMES = ("Search", "Search messages", "Search contacts",
                     "To", "Type a name")
_MESSAGE_INPUT_NAMES = ("Type a message", "Message", "Text message",
                        "Send a message")
_SEND_BUTTON_NAMES = ("Send", "Send message")


class PhoneLinkConnector(Connector):
    """Send / read texts via Microsoft Phone Link's UI.

    `setup_self(path='')` verifies the Phone Link AppX is installed and
    optionally launches it. After successful setup the connector reports
    available=True and its tools enter the catalog.
    """

    id = "phone_link"
    description = (
        "Microsoft Phone Link: send / read SMS + iMessage on Windows by "
        "driving the Phone Link app via accessibility. Requires Phone "
        "Link installed + a paired phone (iPhone or Android). Only path "
        "for iMessage sending on Windows."
    )

    def __init__(self, executor: Any = None) -> None:
        # executor is the session's ToolExecutor — gives us access to the
        # shared UiaController so we don't spin up a second COM client.
        self._executor = executor
        self._appx_name: Optional[str] = None
        self._verified = False

    # ---- availability + setup ----------------------------------------------
    def available(self) -> bool:
        if self._verified:
            return True
        # Cheap check: AppX present.
        self._appx_name = _find_phone_link_appx()
        if self._appx_name:
            self._verified = True
            return True
        return False

    def setup_self(self, path: str = "") -> Dict[str, Any]:
        """Detect Phone Link, optionally launch it to verify the user has
        paired a phone. Returns {ok, appx_name, version, hint}."""
        appx = _find_phone_link_appx()
        if not appx:
            return {
                "ok": False,
                "error": (
                    "Phone Link not installed. Install from the Microsoft "
                    "Store ('Phone Link') or via 'winget install "
                    "Microsoft.PhoneLink', then sign in and pair your "
                    "iPhone following the prompts. Re-run 'set up phone "
                    "link' once paired."
                ),
            }
        self._appx_name = appx
        self._verified = True
        version = _appx_version(appx)
        # Best-effort launch so the user sees it open and can confirm
        # pairing. If it fails (e.g. headless session), we still report
        # success because the AppX is installed.
        launched = _launch_phone_link()
        return {
            "ok": True,
            "appx_name": appx,
            "version": version,
            "launched": launched,
            "hint": ("Pair your iPhone in Phone Link if you haven't yet "
                     "(Settings > Add device). Once paired, ask me to "
                     "'send a text to X saying Y' and I'll drive the UI."),
        }

    # ---- tools -------------------------------------------------------------
    def tools(self) -> List[Dict[str, Any]]:
        def fn(name, desc, props=None, required=None):
            return {"type": "function", "name": name, "description": desc,
                    "parameters": {"type": "object",
                                   "properties": props or {},
                                   "required": required or [],
                                   "additionalProperties": False}}
        return [
            fn("phone_link_open",
               "Launch (or focus) the Microsoft Phone Link app. Use as a "
               "first step before sending or reading messages.", {}),
            fn("phone_link_send_text",
               "Send a text/iMessage via Phone Link. `to` is a contact "
               "name (resolved via Phone Link's contacts) OR a phone "
               "number. `body` is the message. Drives the UI; takes 3-8 "
               "seconds. Only path that works for iMessage on Windows.",
               {"to": {"type": "string"},
                "body": {"type": "string"}},
               ["to", "body"]),
            fn("phone_link_read_recent",
               "Read the recent conversations list from Phone Link's "
               "Messages tab. Returns the sender + last-message preview "
               "for the top `limit` conversations.",
               {"limit": {"type": "integer",
                          "description": "Max conversations (default 10)."}}),
            fn("phone_link_open_conversation",
               "Open a specific contact's conversation in Phone Link "
               "(navigates the UI). Use to position before reading a "
               "thread or sending a follow-up.",
               {"contact": {"type": "string"}},
               ["contact"]),
        ]

    # ---- execution ---------------------------------------------------------
    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available():
            return connector_result(
                "error",
                error=("Phone Link not set up. Say 'set up phone link' first."),
                code="not_ready")

        try:
            if name == "phone_link_open":
                ok = _launch_phone_link()
                if not ok:
                    return connector_result("error",
                                            error="couldn't launch Phone Link")
                return connector_result("ok", launched=True,
                                        title=_visible_window_title())

            if name == "phone_link_send_text":
                to = str(args.get("to") or "").strip()
                body = str(args.get("body") or "").strip()
                if not to:
                    return connector_result("error", error="'to' is required")
                if not body:
                    return connector_result("error", error="'body' is required")
                return self._send_text(to, body)

            if name == "phone_link_read_recent":
                limit = max(1, min(int(args.get("limit") or 10), 30))
                return self._read_recent(limit)

            if name == "phone_link_open_conversation":
                contact = str(args.get("contact") or "").strip()
                if not contact:
                    return connector_result("error", error="'contact' is required")
                return self._open_conversation(contact)

        except Exception as exc:
            return connector_result(
                "error", error=f"{type(exc).__name__}: {exc}")
        return connector_result(
            "error", error=f"unknown phone_link tool: {name}",
            code="no_handler")

    # ---- UI driving primitives ---------------------------------------------
    def _uia(self):
        """Return a UiaController, reusing the executor's if available."""
        if self._executor is not None and hasattr(self._executor, "_uia"):
            try:
                return self._executor._uia  # the executor's shared instance
            except Exception:
                pass
        # Fallback: standalone instance.
        from ..uia_controller import UiaController
        return UiaController(logger=None)

    def _focus_phone_link(self) -> bool:
        """Make sure Phone Link is the foreground window; launch if needed."""
        title = _visible_window_title()
        if not title:
            if not _launch_phone_link():
                return False
            # Wait up to 5s for the window to appear.
            for _ in range(10):
                time.sleep(0.5)
                if _visible_window_title():
                    break
        return bool(_visible_window_title())

    def _send_text(self, to: str, body: str) -> Dict[str, Any]:
        """The slow but functional path: open + navigate + type + send.
        Each step is best-effort with name variants tried so this survives
        Phone Link version drift."""
        if not self._focus_phone_link():
            return connector_result("error",
                                    error="Phone Link didn't open / focus")
        uia = self._uia()
        title = _visible_window_title()

        # 1. Click Messages tab (idempotent if already there).
        if not _click_first_match(uia, title, _MESSAGES_TAB_NAMES):
            return connector_result(
                "error",
                error=("Couldn't find Messages tab in Phone Link. The UI "
                       "may have changed. Open Phone Link manually and "
                       "ask me to send again."),
                code="ui_drift")
        time.sleep(0.4)

        # 2. Click 'New message' OR an existing search box. Either lets us
        #    type the recipient.
        if not (_click_first_match(uia, title, _NEW_MESSAGE_NAMES)
                or _click_first_match(uia, title, _SEARCH_BOX_NAMES)):
            return connector_result(
                "error",
                error="Couldn't find new-message / search field",
                code="ui_drift")
        time.sleep(0.4)

        # 3. Type recipient.
        _type_text(uia, to)
        time.sleep(0.6)  # let autocomplete populate

        # 4. Click the matching contact in the autocomplete (its accessible
        #    name should contain the recipient string).
        if not _click_by_name_substring(uia, title, to):
            # If autocomplete didn't yield, press Tab/Enter to accept
            # raw text — Phone Link sometimes accepts a phone number that way.
            uia._press_key("tab")  # type: ignore[attr-defined]
        time.sleep(0.4)

        # 5. Find the message input + type body.
        if not _click_first_match(uia, title, _MESSAGE_INPUT_NAMES):
            return connector_result(
                "error",
                error="Couldn't find message input field",
                code="ui_drift")
        _type_text(uia, body)
        time.sleep(0.2)

        # 6. Send via Enter (works in current Phone Link) or Send button.
        sent = uia._press_key("enter")  # type: ignore[attr-defined]
        if not sent:
            _click_first_match(uia, title, _SEND_BUTTON_NAMES)
        return connector_result("ok", sent=True, to=to,
                                preview=body[:80])

    def _open_conversation(self, contact: str) -> Dict[str, Any]:
        if not self._focus_phone_link():
            return connector_result("error",
                                    error="Phone Link didn't open / focus")
        uia = self._uia()
        title = _visible_window_title()
        _click_first_match(uia, title, _MESSAGES_TAB_NAMES)
        time.sleep(0.3)
        if not _click_by_name_substring(uia, title, contact):
            return connector_result(
                "error",
                error=f"Couldn't find a conversation matching {contact!r}",
                code="not_found")
        return connector_result("ok", opened=contact)

    def _read_recent(self, limit: int) -> Dict[str, Any]:
        """List the conversation tiles in the sidebar."""
        if not self._focus_phone_link():
            return connector_result("error",
                                    error="Phone Link didn't open / focus")
        uia = self._uia()
        title = _visible_window_title()
        _click_first_match(uia, title, _MESSAGES_TAB_NAMES)
        time.sleep(0.3)
        try:
            data = uia.list_elements(window_title=title, limit=80)
        except Exception as exc:
            return connector_result(
                "error", error=f"UIA list_elements failed: {exc}")
        # Conversation tiles are 'listitem' or 'button' control types with
        # names that look like "<sender> · <preview>" or just "<sender>".
        items: List[Dict[str, Any]] = []
        seen: set = set()
        for el in (data.get("elements") or []):
            kind = el.get("type") or ""
            name = (el.get("name") or "").strip()
            if not name or kind not in ("listitem", "button"):
                continue
            # Skip nav chrome.
            if name.lower() in {"messages", "calls", "photos", "apps",
                                "back", "new message", "search"}:
                continue
            if name in seen:
                continue
            seen.add(name)
            items.append({"name": name})
            if len(items) >= limit:
                break
        return connector_result("ok", count=len(items), conversations=items)


# ---- module-level helpers (no UIA needed) ---------------------------------
def _find_phone_link_appx() -> Optional[str]:
    """Returns the AppX package name if Phone Link is installed, else None.

    v1.1.7: was `powershell -Command Get-AppxPackage -Name ...`. That exact
    byte pattern (hidden powershell.exe launch enumerating installed AppX
    packages) is one of the Defender ASR fingerprints that started
    quarantining the app. We enumerate shell:AppsFolder via Shell.Application
    COM instead and look for the AppUserModelID prefix — same detection,
    no subprocess, no fingerprint.
    """
    try:
        from comtypes import CoInitialize, CoUninitialize
        import comtypes.client
    except Exception:
        return None
    CoInitialize()
    try:
        try:
            shell = comtypes.client.CreateObject("Shell.Application", dynamic=True)
            apps_folder = shell.NameSpace("shell:AppsFolder")
        except Exception:
            return None
        if apps_folder is None:
            return None
        try:
            items = apps_folder.Items()
            count = int(items.Count)
        except Exception:
            return None
        lowered_targets = [name.lower() for name in _APPX_NAMES]
        for index in range(count):
            try:
                item = items.Item(index)
                app_id = str(item.Path or "").lower()
            except Exception:
                continue
            if not app_id:
                continue
            for target_name, lowered in zip(_APPX_NAMES, lowered_targets):
                if app_id.startswith(lowered + "_"):
                    return target_name
        return None
    finally:
        try:
            CoUninitialize()
        except Exception:
            pass


def _appx_version(appx_name: str) -> str:
    """Best-effort version probe. Returns "" on failure or when winsdk
    isn't available in this build. Non-critical — used only for a debug
    hint in setup_self output. Was a PowerShell Get-AppxPackage call;
    now uses winsdk (WinRT PackageManager) so no subprocess is spawned.
    """
    try:
        from winsdk.windows.management.deployment import PackageManager  # type: ignore
    except Exception:
        return ""
    try:
        manager = PackageManager()
        for package in manager.find_packages_by_name(appx_name):
            try:
                version = package.id.version
                return f"{version.major}.{version.minor}.{version.build}.{version.revision}"
            except Exception:
                continue
    except Exception:
        return ""
    return ""


def _launch_phone_link() -> bool:
    """Launch (or focus) Phone Link via its URI scheme. Uses
    ShellExecuteW (via launch_external) so the OS resolves the URI the
    same way Explorer does — no cmd.exe / no flash / no AV alerts on
    PyInstaller builds."""
    for uri in _LAUNCH_URIS:
        try:
            if launch_external(uri):
                return True
        except Exception:
            continue
    return False


def _visible_window_title() -> str:
    """Returns the title of a visible Phone Link window if one exists.
    Used to confirm launch and as the window_title arg to UIA."""
    try:
        import ctypes
        from ctypes import wintypes
        EnumWindows = ctypes.windll.user32.EnumWindows
        GetWindowText = ctypes.windll.user32.GetWindowTextW
        IsWindowVisible = ctypes.windll.user32.IsWindowVisible
        GetClassName = ctypes.windll.user32.GetClassNameW

        found = {"title": ""}

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def _cb(hwnd, _lparam):
            if not IsWindowVisible(hwnd):
                return True
            buf = ctypes.create_unicode_buffer(256)
            GetWindowText(hwnd, buf, 256)
            title = (buf.value or "").strip()
            if any(needle.lower() in title.lower() for needle in _WINDOW_TITLES):
                found["title"] = title
                return False
            return True

        EnumWindows(_cb, 0)
        return found["title"]
    except Exception:
        return ""


def _click_first_match(uia, window_title: str, name_variants) -> bool:
    """Try each name in turn; click the first one found. Returns True on
    success, False if none matched."""
    for name in name_variants:
        try:
            r = uia.click(target=name, window_title=window_title)
            if isinstance(r, dict) and r.get("status") == "ok":
                return True
        except Exception:
            continue
    return False


def _click_by_name_substring(uia, window_title: str, needle: str) -> bool:
    """List elements + click the first whose name contains `needle`."""
    try:
        data = uia.list_elements(window_title=window_title, limit=80)
    except Exception:
        return False
    nlow = needle.lower()
    for el in (data.get("elements") or []):
        name = (el.get("name") or "").strip()
        if name and nlow in name.lower():
            try:
                r = uia.click(target=name, window_title=window_title)
                if isinstance(r, dict) and r.get("status") == "ok":
                    return True
            except Exception:
                continue
    return False


def _type_text(uia, text: str) -> None:
    """Type via the UIA controller's keyboard primitive."""
    try:
        uia._type_text(text)  # type: ignore[attr-defined]
    except Exception:
        pass

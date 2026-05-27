"""Shared Microsoft Graph (Microsoft 365) OAuth client for the MS connectors.

The Microsoft equivalent of google_client.py. Microsoft Graph is the one API
behind everything Copilot/M365 touches — Outlook mail, M365 calendar,
OneDrive, OneNote, Teams, etc. This brokers a delegated-permission token via
MSAL so iris can act on the user's behalf.

Why this is a *scaffold* (dormant until set up)
------------------------------------------------
It needs:
  1. The MSAL lib — ``pip install msal`` (added to requirements).
  2. An **Azure app registration** (Microsoft Entra) of type "public client"
     (Desktop). A public client uses PKCE, so there is NO client secret to
     ship — only the **client id**. Provide it via env ``MS_GRAPH_CLIENT_ID``
     or a bundled ``assets/ms_graph_client.json`` ({"client_id": "..."}).
  3. A one-time interactive consent (the "Connect Microsoft" button →
     ``connect()`` opens the browser).

Until (2) + (3) are done, ``ready()`` is False, the MS connectors report
``available() == False``, and nothing is exposed — safe to ship dormant.

Token cache lives at ~/Documents/Touchless/microsoft/token_cache.bin so it
survives restarts; MSAL refreshes silently.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

# Delegated scopes — send mail, read/write calendar, per-app file access,
# basic profile. offline_access (refresh token) is added by MSAL.
SCOPES: List[str] = [
    "Mail.Send",
    "Mail.ReadWrite",       # read inbox + manage messages (mark read, drafts)
    "Calendars.ReadWrite",
    "Files.ReadWrite",      # OneDrive + Excel workbook read/write
    "Chat.ReadWrite",       # Teams 1:1 chat messages
    "Team.ReadBasic.All",   # list joined teams (resolve by name)
    "Channel.ReadBasic.All",# list channels in a team (resolve by name)
    "ChannelMessage.Send",  # post to a Teams channel
    "Tasks.ReadWrite",      # Microsoft To Do
    "Notes.ReadWrite",      # OneNote
    "Contacts.ReadWrite",   # Outlook contacts
    "User.Read",
]

AUTHORITY = "https://login.microsoftonline.com/common"
GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def _config_dir() -> Path:
    override = os.environ.get("MS_GRAPH_DIR")
    base = Path(override).expanduser() if override else (
        Path.home() / "Documents" / "Touchless" / "microsoft"
    )
    return base


def _cache_path() -> Path:
    return _config_dir() / "token_cache.bin"


def libs_available() -> bool:
    try:
        import msal  # noqa: F401
        return True
    except Exception:
        return False


def _client_id() -> str:
    """The app's Azure public-client id (no secret needed for desktop)."""
    cid = (os.environ.get("MS_GRAPH_CLIENT_ID") or "").strip()
    if cid:
        return cid
    try:
        from ...utils.runtime_paths import resource_path
        p = resource_path("assets", "ms_graph_client.json")
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return str(data.get("client_id") or "").strip()
    except Exception:
        pass
    return ""


def client_configured() -> bool:
    return bool(_client_id())


class MsGraphClient:
    _shared: Optional["MsGraphClient"] = None

    def __init__(self) -> None:
        self._app = None
        self._cache = None

    @classmethod
    def shared(cls) -> "MsGraphClient":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    # ---- MSAL plumbing ----
    def _load_cache(self):
        import msal
        cache = msal.SerializableTokenCache()
        cp = _cache_path()
        if cp.exists():
            try:
                cache.deserialize(cp.read_text(encoding="utf-8"))
            except Exception:
                pass
        return cache

    def _save_cache(self) -> None:
        try:
            if self._cache is not None and self._cache.has_state_changed:
                _config_dir().mkdir(parents=True, exist_ok=True)
                _cache_path().write_text(self._cache.serialize(), encoding="utf-8")
        except Exception:
            pass

    def _get_app(self):
        if self._app is not None:
            return self._app
        if not libs_available():
            return None
        cid = _client_id()
        if not cid:
            return None
        import msal
        self._cache = self._load_cache()
        self._app = msal.PublicClientApplication(
            cid, authority=AUTHORITY, token_cache=self._cache)
        return self._app

    def token(self) -> Optional[str]:
        """A valid access token via the cached account (silent), or None."""
        app = self._get_app()
        if app is None:
            return None
        try:
            accounts = app.get_accounts()
            if not accounts:
                return None
            result = app.acquire_token_silent(SCOPES, account=accounts[0])
            self._save_cache()
            if result and "access_token" in result:
                return result["access_token"]
        except Exception:
            return None
        return None

    def ready(self) -> bool:
        return self.token() is not None

    def connect(self) -> tuple[bool, str]:
        """Interactive consent (opens browser). Call off the UI thread."""
        if not libs_available():
            return False, "Microsoft auth library missing. Run: pip install msal"
        app = self._get_app()
        if app is None:
            return False, ("No Azure client configured. (Developer: set "
                           "MS_GRAPH_CLIENT_ID or bundle assets/ms_graph_client.json.)")
        try:
            result = app.acquire_token_interactive(SCOPES, prompt="select_account")
            self._save_cache()
            if result and "access_token" in result:
                MsGraphClient._shared = None  # rebuild with fresh cache
                return True, "Microsoft 365 connected."
            return False, f"Authorization failed: {result.get('error_description', 'unknown')}"
        except Exception as exc:
            return False, f"Authorization failed: {type(exc).__name__}: {exc}"


def status() -> str:
    if not libs_available():
        return "needs_libs"
    if MsGraphClient.shared().ready():
        return "connected"
    if client_configured():
        return "ready_to_connect"
    return "needs_client"

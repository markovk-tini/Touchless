"""Shared Google OAuth client for the Gmail / Calendar / Docs connectors.

Why this is a *scaffold*
------------------------
Google connectors need three things that aren't in the repo yet:

  1. The client libraries — ``pip install google-api-python-client
     google-auth google-auth-oauthlib`` (added to requirements.txt, but
     install them in your env).
  2. An OAuth **client secret** from a Google Cloud project (Desktop-app
     credentials). Put the downloaded JSON at the path below.
  3. A one-time **consent** run that mints a refresh token.

Until (2) and (3) are done, ``GoogleClient.ready()`` is False, the
connectors report ``available() == False``, and nothing is exposed to the
model — so this is safe to ship dormant.

Interactive consent canNOT run inside the realtime worker (it opens a
browser and blocks), so it lives in a separate entry point:

    python -m hgr.live_api.connectors.google_client authorize

Run that once; it writes a token next to the client secret and from then
on the connectors silently refresh it.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

# SEND-only mail (gmail.send), full calendar, create docs, per-file Drive.
# gmail.send / calendar / documents / drive.file are all "sensitive" scopes
# — free public verification, NO paid CASA assessment.
#
# gmail.readonly is "restricted" and would require CASA for PUBLIC release,
# but it's fine for PERSONAL / DEV use (the OAuth consent screen just shows
# an "unverified app" warning you can click through). It unlocks 'read my
# emails' / morning-briefing flows where Microsoft Graph for personal MSA
# accounts is incomplete. Disable by setting TOUCHLESS_GMAIL_READONLY=0.
SCOPES: List[str] = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/drive.file",
]
if os.environ.get("TOUCHLESS_GMAIL_READONLY", "1") != "0":
    SCOPES.append("https://www.googleapis.com/auth/gmail.readonly")


def _config_dir() -> Path:
    """Where the client secret + token live. Mirrors LiveApiConfig's
    ~/Documents/Touchless home so it survives auto-updates, overridable
    via GOOGLE_OAUTH_DIR."""
    override = os.environ.get("GOOGLE_OAUTH_DIR")
    base = Path(override).expanduser() if override else (
        Path.home() / "Documents" / "Touchless" / "google"
    )
    return base


def _client_secret_path() -> Path:
    override = os.environ.get("GOOGLE_OAUTH_CLIENT_FILE")
    if override:
        return Path(override).expanduser()
    return _config_dir() / "client_secret.json"


def _embedded_client_config() -> Optional[dict]:
    """The app's *own* OAuth client, embedded so end users never create one.

    The developer ships it via env (GOOGLE_OAUTH_CLIENT_ID +
    GOOGLE_OAUTH_CLIENT_SECRET) or a bundled file (GOOGLE_OAUTH_CLIENT_JSON).
    For a Desktop OAuth client the secret is a public-client secret (PKCE),
    so shipping it is per Google's installed-app model. Returns a dict
    suitable for InstalledAppFlow.from_client_config, or None if not embedded.
    """
    cid = (os.environ.get("GOOGLE_OAUTH_CLIENT_ID") or "").strip()
    csec = (os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET") or "").strip()
    if cid and csec:
        return {"installed": {
            "client_id": cid,
            "client_secret": csec,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }}
    import json
    bundled = (os.environ.get("GOOGLE_OAUTH_CLIENT_JSON") or "").strip()
    if bundled:
        try:
            return json.loads(Path(bundled).expanduser().read_text(encoding="utf-8"))
        except Exception:
            return None
    # Shipped-with-the-app client: assets/google_oauth_client.json is bundled
    # by hgr_app.spec and resolves in both source and frozen runs. This is how
    # the client reaches end users — they never create one.
    try:
        from ...utils.runtime_paths import resource_path
        p = resource_path("assets", "google_oauth_client.json")
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def client_configured() -> bool:
    """True if an OAuth client is available to start the consent flow with
    (embedded env/bundled config, or a developer's client_secret.json)."""
    return bool(_embedded_client_config()) or _client_secret_path().exists()


def _token_path() -> Path:
    return _config_dir() / "token.json"


def libs_available() -> bool:
    """True if the Google client libraries are importable."""
    try:
        import google.oauth2.credentials  # noqa: F401
        import googleapiclient.discovery  # noqa: F401
        return True
    except Exception:
        return False


class GoogleClient:
    """Loads stored OAuth credentials and builds Google API service objects.
    A single instance is shared by all Google connectors so they reuse one
    token / refresh."""

    _shared: Optional["GoogleClient"] = None

    def __init__(self) -> None:
        self._creds = None
        self._services: dict = {}

    @classmethod
    def shared(cls) -> "GoogleClient":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def ready(self) -> bool:
        """True when we can build authenticated services right now: libs
        installed AND a valid (or refreshable) stored token exists."""
        if not libs_available():
            return False
        return self._load_creds() is not None

    def _load_creds(self):
        if self._creds is not None and getattr(self._creds, "valid", False):
            return self._creds
        if not libs_available():
            return None
        token_path = _token_path()
        if not token_path.exists():
            return None
        # Read whatever scopes the token was ACTUALLY granted (not the
        # SCOPES we'd LIKE). Stored in self._granted_scopes so per-
        # operation callers (gmail_list needs readonly, etc.) can check
        # exactly what's available without invalidating the whole grant.
        self._granted_scopes = set()
        try:
            token_doc = json.loads(token_path.read_text(encoding="utf-8"))
            self._granted_scopes = set(token_doc.get("scopes") or [])
        except Exception:
            pass
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request

            # Use the granted scope list to construct the Credentials so
            # refresh works even when the user granted only a subset.
            # Passing the full SCOPES list here causes
            # Credentials.from_authorized_user_file to mark every scope
            # as "current" — refresh succeeds but per-API calls fail at
            # the server for the missing ones. Using granted_scopes keeps
            # the credentials object truthful.
            scopes_for_creds = (sorted(self._granted_scopes)
                                if self._granted_scopes else SCOPES)
            creds = Credentials.from_authorized_user_file(
                str(token_path), scopes_for_creds)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                token_path.write_text(creds.to_json(), encoding="utf-8")
            if creds and creds.valid:
                self._creds = creds
                return creds
        except Exception:
            return None
        return None

    def has_scope(self, scope: str) -> bool:
        """True if the user's Google grant includes `scope`. Forces a
        creds reload so a freshly-reconnected token is reflected."""
        if self._load_creds() is None:
            return False
        return scope in getattr(self, "_granted_scopes", set())

    def service(self, api: str, version: str):
        """Return a cached googleapiclient service, or None if not ready."""
        creds = self._load_creds()
        if creds is None:
            return None
        key = f"{api}:{version}"
        svc = self._services.get(key)
        if svc is not None:
            return svc
        try:
            from googleapiclient.discovery import build
            svc = build(api, version, credentials=creds, cache_discovery=False)
            self._services[key] = svc
            return svc
        except Exception:
            return None


def status() -> str:
    """Coarse connection state for the UI button:
    'connected' | 'ready_to_connect' | 'needs_client' | 'needs_libs'."""
    if not libs_available():
        return "needs_libs"
    if GoogleClient.shared().ready():
        return "connected"
    if client_configured():
        return "ready_to_connect"
    return "needs_client"


def connect() -> tuple[bool, str]:
    """Run the interactive OAuth consent flow (opens the user's browser) and
    persist the token. Returns (ok, message). Safe to call from a background
    thread — it blocks on the local consent server, so never call it on the
    Qt/UI thread. This is what the 'Connect Gmail' button invokes.
    """
    if not libs_available():
        return False, ("Google libraries not installed. Run: pip install "
                       "google-api-python-client google-auth google-auth-oauthlib")
    from google_auth_oauthlib.flow import InstalledAppFlow

    config = _embedded_client_config()
    secret_file = _client_secret_path()
    try:
        if config is not None:
            flow = InstalledAppFlow.from_client_config(config, SCOPES)
        elif secret_file.exists():
            flow = InstalledAppFlow.from_client_secrets_file(str(secret_file), SCOPES)
        else:
            return False, ("No OAuth client configured. (Developer: set "
                           "GOOGLE_OAUTH_CLIENT_ID/SECRET or bundle the client "
                           "JSON via GOOGLE_OAUTH_CLIENT_JSON.)")
        # prompt="consent" always shows the grant screen (even if the account
        # already authorized) and guarantees a refresh token; access_type
        # offline keeps the refresh token across restarts.
        creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
        _config_dir().mkdir(parents=True, exist_ok=True)
        _token_path().write_text(creds.to_json(), encoding="utf-8")
        # Reset the shared client so services rebuild with the new token.
        GoogleClient._shared = None
        return True, "Gmail connected."
    except Exception as exc:
        return False, f"Authorization failed: {type(exc).__name__}: {exc}"


def authorize() -> int:
    """CLI entry: `python -m hgr.live_api.connectors.google_client authorize`."""
    ok, msg = connect()
    print(msg)
    return 0 if ok else 2


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "authorize":
        raise SystemExit(authorize())
    print("Usage: python -m hgr.live_api.connectors.google_client authorize")

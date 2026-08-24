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

# SEND-only mail (gmail.send), full calendar, per-file Drive.
# gmail.send / calendar / drive.file are all "sensitive" scopes
# — free public verification, NO paid CASA assessment.
#
# DOCS / SHEETS / SLIDES broad scopes removed 2026-07-29: Google
# rejected `documents`, `spreadsheets`, and `presentations` during
# OAuth verification (they read/write every file in the user's
# Drive and are restricted-tier). We now reach individual Docs,
# Sheets, and Slides files via the `drive.file` scope, which the
# Docs v1 / Sheets v4 / Slides v1 APIs accept for reads AND writes
# as long as the caller has a file_id that was either
#   (a) CREATED by this app (all *_connector.create paths satisfy
#       this automatically), or
#   (b) explicitly opened by the user via the Google Picker widget.
# See app/ui/google_picker_dialog.py + live_api/connectors/
# google_picker_cache.py for the name->file_id resolution flow
# that replaces the old drive.files().list(q=name) search path.
#
# gmail.readonly is "restricted" and would require CASA for PUBLIC release,
# but it's fine for PERSONAL / DEV use (the OAuth consent screen just shows
# an "unverified app" warning you can click through). It unlocks 'read my
# emails' / morning-briefing flows where Microsoft Graph for personal MSA
# accounts is incomplete. Disable by setting TOUCHLESS_GMAIL_READONLY=0.
SCOPES: List[str] = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.file",
    # Round-2 connectors. All sensitive-tier (free verification) EXCEPT
    # user.birthday.read which is restricted — flagged in OPEN_ISSUES.md
    # for OAuth review before public ship.
    "https://www.googleapis.com/auth/contacts",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/forms.responses.readonly",
    # NOTE: youtube.readonly CANNOT be requested in the same OAuth flow as
    # drive.file — Google returns 400 invalid_request "cannot be requested
    # together". YouTube would need its own OAuth client (separate
    # client_id + separate consent flow). Dropped for now; add via a
    # dedicated YouTubeClient later if the feature becomes must-have.
    "https://www.googleapis.com/auth/photoslibrary.appendonly",
    # Identity / profile reads. openid + userinfo.* are non-sensitive
    # basic scopes; user.birthday.read is sensitive and gated by Google.
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/user.birthday.read",
]
# gmail.readonly is a Google "restricted" scope that requires CASA Tier 2
# security assessment ($10-30k upfront + annual renewal) for verified
# distribution. NOT shippable for free, so we OPT-OUT by default and
# leave reads to Outlook desktop COM (free, no verification) and MS
# Graph Mail.Read ($99/yr Microsoft Partner, no CASA). Personal devs
# can flip it on via TOUCHLESS_GMAIL_READONLY=1 — but the shipped app
# never asks for it, so users never see the "unverified app" warning.
if os.environ.get("TOUCHLESS_GMAIL_READONLY", "0") == "1":
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


def _picker_api_key() -> Optional[str]:
    """API key for the Google Picker widget (server-issued, restricted to
    the Picker API by IP/referer in Cloud Console).

    Resolution mirrors _embedded_client_config():
      1) env override GOOGLE_PICKER_API_KEY,
      2) bundled asset assets/google_picker_api_key.txt (single line),
      3) None (Picker path degrades to "manually paste a file id").
    """
    override = (os.environ.get("GOOGLE_PICKER_API_KEY") or "").strip()
    if override:
        return override
    try:
        from ...utils.runtime_paths import resource_path
        p = resource_path("assets", "google_picker_api_key.txt")
        if p.exists():
            key = p.read_text(encoding="utf-8").strip()
            return key or None
    except Exception:
        pass
    return None


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
                if not getattr(self, "_logged_missing_scopes", False):
                    self._logged_missing_scopes = True
                    missing = set(SCOPES) - (self._granted_scopes or set())
                    if missing:
                        try:
                            print(f"[google_client] stored token is "
                                  f"missing {len(missing)} scope(s); "
                                  f"reconnect Google to grant: "
                                  f"{sorted(missing)}")
                        except Exception:
                            pass
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

    def missing_scopes(self) -> set:
        """Scopes the app declares in SCOPES that the stored token does
        NOT yet grant. Empty set when grant matches (or exceeds) SCOPES.
        Returns a copy of SCOPES if the token is unloadable / absent."""
        if self._load_creds() is None:
            return set(SCOPES)
        granted = getattr(self, "_granted_scopes", set()) or set()
        return set(SCOPES) - granted

    @staticmethod
    def scope_missing_result(scope: str, *, friendly: str = "") -> dict:
        """Uniform `scope_missing` connector_result. `friendly` is an
        optional short label for the capability (e.g. 'Google Tasks');
        the user-facing error always tells them to re-click Connect
        Google in Settings so the new consent screen re-grants the
        added scope."""
        from .base import connector_result
        label = friendly or scope
        return connector_result(
            "error",
            code="scope_missing",
            error=(f"I need {label} access (Google scope '{scope}') — "
                   f"please click 'Connect Google' in Settings to "
                   f"re-authorize. The consent screen will include it."),
            missing_scope=scope,
        )

    def service(self, api: str, version: str, *, http_timeout: float = 20.0):
        """Return a cached googleapiclient service, or None if not ready.

        `http_timeout` gives the underlying httplib2 socket a wall-clock
        ceiling so a wedged TLS handshake / dropped keep-alive / captive-
        portal stall cannot block a calling thread indefinitely. Without
        this googleapiclient builds a default httplib2.Http() whose
        socket timeout is None (see drive_connector hang trace).
        """
        creds = self._load_creds()
        if creds is None:
            return None
        key = f"{api}:{version}:{http_timeout}"
        svc = self._services.get(key)
        if svc is not None:
            return svc
        try:
            from googleapiclient.discovery import build
            try:
                import httplib2
                from google_auth_httplib2 import AuthorizedHttp
                http = AuthorizedHttp(creds, http=httplib2.Http(timeout=http_timeout))
                svc = build(api, version, http=http, cache_discovery=False)
            except Exception:
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


# Short human labels for each scope, used to surface a friendly
# "you declined: Google Tasks, Google Slides, Google Sheets" message
# when the user un-checks boxes on the consent screen. Anything not
# listed here falls back to the raw scope URL.
_SCOPE_FRIENDLY: dict = {
    "https://www.googleapis.com/auth/gmail.send": "Gmail (send)",
    "https://www.googleapis.com/auth/gmail.readonly": "Gmail (read)",
    "https://www.googleapis.com/auth/calendar": "Google Calendar",
    "https://www.googleapis.com/auth/drive.file": "Google Drive",
    "https://www.googleapis.com/auth/contacts": "Google Contacts",
    "https://www.googleapis.com/auth/tasks": "Google Tasks",
    "https://www.googleapis.com/auth/forms.body": "Google Forms",
    "https://www.googleapis.com/auth/forms.responses.readonly":
        "Google Forms responses",
    "https://www.googleapis.com/auth/youtube.readonly": "YouTube",
    "https://www.googleapis.com/auth/photoslibrary.appendonly":
        "Google Photos",
    "openid": "OpenID",
    "https://www.googleapis.com/auth/userinfo.email": "Email address",
    "https://www.googleapis.com/auth/userinfo.profile": "Profile",
    "https://www.googleapis.com/auth/user.birthday.read": "Birthday",
}


def _friendly_scope_labels(scopes) -> list:
    """Map a list/set of scope URLs to short human names, deduped while
    preserving order of the iterable for stable UI output."""
    out: list = []
    seen: set = set()
    for s in scopes:
        label = _SCOPE_FRIENDLY.get(s, s)
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def connect() -> tuple[bool, str]:
    """Run the interactive OAuth consent flow (opens the user's browser) and
    persist the token. Returns (ok, message). Safe to call from a background
    thread — it blocks on the local consent server, so never call it on the
    Qt/UI thread. This is what the 'Connect Gmail' button invokes.

    Post-connect we audit the freshly-written token against SCOPES; if the
    user un-checked any of the optional boxes on the consent screen we
    surface a clear "you declined N scopes" warning in the returned
    message so the UI can echo it (otherwise a partial grant is silent
    until a tool happens to need a missing scope).
    """
    if not libs_available():
        return False, ("Google libraries not installed. Run: pip install "
                       "google-api-python-client google-auth google-auth-oauthlib")
    # include_granted_scopes='true' below makes Google legitimately return a
    # SUPERSET token (union with previously-granted scopes, e.g. legacy
    # gmail.readonly). Without this env var oauthlib's token-response parser
    # raises Warning('Scope has changed from ... to ...') and aborts the flow
    # even though the response is a strict superset. The post-connect audit
    # further down still verifies every required SCOPE is present, so extras
    # are harmless. setdefault preserves any user override.
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
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
        #
        # include_granted_scopes='true' enables INCREMENTAL authorization:
        # Google merges the newly-approved scopes on top of the existing
        # grant rather than replacing it. So if the user un-checks one of
        # the optional sensitive-scope boxes on a second consent run, they
        # keep whatever they had granted before instead of regressing to
        # only the subset they re-approved this time. (Google's OAuth lib
        # forwards extra kwargs straight into authorization_url.)
        creds = flow.run_local_server(
            port=0,
            prompt="consent",
            access_type="offline",
            include_granted_scopes="true",
        )
        _config_dir().mkdir(parents=True, exist_ok=True)
        _token_path().write_text(creds.to_json(), encoding="utf-8")
        # Reset the shared client so services rebuild with the new token,
        # then immediately force a creds reload so missing_scopes() reads
        # from the just-written file rather than a stale in-memory grant.
        GoogleClient._shared = None
        shared = GoogleClient.shared()
        try:
            shared._load_creds()
        except Exception:
            pass
        granted = sorted(getattr(shared, "_granted_scopes", set()) or set())
        missing = sorted(set(SCOPES) - set(granted))
        # Always emit a diagnostic line so the console / log shows the
        # exact granted-vs-required set on every connect. Helpful for
        # the partial-grant bug report ("token has only N of M scopes").
        try:
            print(f"[google_client] connect complete: granted "
                  f"{len(granted)}/{len(SCOPES)} scope(s); "
                  f"missing={_friendly_scope_labels(missing)}")
        except Exception:
            pass
        if missing:
            friendly = _friendly_scope_labels(missing)
            return True, (
                f"Google connected, but you declined {len(missing)} "
                f"scope(s): {', '.join(friendly)}. Click "
                f"'Google ✓ (reconnect)' and on the consent screen "
                f"CHECK ALL CHECKBOXES (Google leaves new sensitive "
                f"scopes UNCHECKED by default)."
            )
        return True, "Google connected."
    except Exception as exc:
        return False, f"Authorization failed: {type(exc).__name__}: {exc}"


def authorize() -> int:
    """CLI entry: `python -m hgr.live_api.connectors.google_client authorize`."""
    ok, msg = connect()
    print(msg)
    return 0 if ok else 2


def scopes_report() -> int:
    """CLI entry: `python -m hgr.live_api.connectors.google_client scopes`.

    Prints the required-vs-granted scope diff for the current stored
    token so a user (or support agent) can see exactly which boxes the
    user un-checked on the consent screen, without running a fresh
    OAuth flow. Exits 0 if all SCOPES granted, 1 if any missing, 2 if
    no token present."""
    if not libs_available():
        print("[google_client] Google libraries not installed.")
        return 2
    shared = GoogleClient.shared()
    if not _token_path().exists():
        print(f"[google_client] No token at {_token_path()}.")
        return 2
    try:
        shared._load_creds()
    except Exception as exc:
        print(f"[google_client] Failed to load token: {exc}")
        return 2
    granted = sorted(getattr(shared, "_granted_scopes", set()) or set())
    missing = sorted(set(SCOPES) - set(granted))
    print(f"Token: {_token_path()}")
    print(f"Required scopes ({len(SCOPES)}):")
    for s in SCOPES:
        marker = "OK " if s in granted else "-- "
        print(f"  {marker}{_SCOPE_FRIENDLY.get(s, s)}  ({s})")
    if missing:
        print("")
        print(f"MISSING {len(missing)} scope(s): "
              f"{', '.join(_friendly_scope_labels(missing))}")
        print("Run the 'Connect Google' button in Settings and CHECK ALL "
              "CHECKBOXES on the consent screen.")
        return 1
    print("\nAll required scopes granted.")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "authorize":
        raise SystemExit(authorize())
    if len(sys.argv) > 1 and sys.argv[1] == "scopes":
        raise SystemExit(scopes_report())
    print("Usage: python -m hgr.live_api.connectors.google_client "
          "{authorize|scopes}")

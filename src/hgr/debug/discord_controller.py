"""Discord RPC controller — mute, deafen, and read voice state on the
local Discord desktop client.

Auth model mirrors [spotify_controller.py](spotify_controller.py)'s
overall shape (env-loaded credentials, persisted JSON token file,
`_needs_reauth` latch) but the transport is fundamentally different:

  * **Spotify** talks HTTPS to api.spotify.com.
  * **Discord** talks a binary RPC frame protocol over a *local Windows
    named pipe* (`\\\\.\\pipe\\discord-ipc-{0..9}`) that the Discord
    desktop client publishes. Only the token-exchange step uses HTTPS.

The pipe protocol is 8-byte header (uint32-LE opcode + uint32-LE length)
followed by UTF-8 JSON. Opcodes: 0 HANDSHAKE, 1 FRAME, 2 CLOSE, 3 PING,
4 PONG. Discord documents this at <https://discord.com/developers/docs/topics/rpc>.

Auth flow (v1 — `rpc` scope only):

  1. Open `\\\\.\\pipe\\discord-ipc-N` (try 0 → 9, take the first one
     `Discord.exe` is listening on).
  2. Send HANDSHAKE `{v:1, client_id}`. Discord replies with READY event
     carrying user info + the negotiated protocol version.
  3. Send `AUTHORIZE` frame with `scopes:["rpc"]`. The Discord client
     pops a modal: "Allow Touchless to control your Discord?" The user
     clicks Allow. Discord replies with `{data:{code}}`.
  4. POST that code to `https://discord.com/api/oauth2/token` along with
     client_id + client_secret + the registered redirect_uri. Discord
     returns `access_token` + `refresh_token` + `expires_in`.
  5. Send `AUTHENTICATE {access_token}`. The connection is now in
     authed state — `SET_VOICE_SETTINGS`, `SELECT_VOICE_CHANNEL`, etc.
     all work.

For development the app **owner** (the Discord account that registered
the app at <https://discord.com/developers/applications>) can step
through this flow without any additional Discord-side review. To roll
out to other users a separate one-time "Public Distribution" review is
required by Discord — we file that once the controller is proven.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import secrets
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


_LOG = logging.getLogger(__name__)


# --- Constants -------------------------------------------------------------

# Public client_id baked into every shipped build. Discord treats
# client_id as public per their docs (it's the OAuth2 `Application ID`
# you see on the General Information page of any app at
# https://discord.com/developers/applications). Override via env var or
# .env for dev work against a different Discord app.
_DEFAULT_DISCORD_CLIENT_ID = "1505345698126630972"

# Redirect URI registered against the Touchless Discord app. The IPC
# auth flow doesn't actually redirect a browser, but Discord's
# token-exchange endpoint still validates redirect_uri against the
# registered list, so this string MUST match a Redirect entry in the
# app's OAuth2 settings.
_DEFAULT_DISCORD_REDIRECT_URI = "http://127.0.0.1"

# No baked-in default for the client_secret. Loading order is env var
# (TOUCHLESS_DISCORD_CLIENT_SECRET, then DISCORD_CLIENT_SECRET) →
# .env file. Shipping the secret in source is too risky — even though
# the `rpc` scope can't do much without user consent, exposing the
# secret makes it easier for a third party to impersonate Touchless.
# When we ship the public release we'll either bake the secret in (low
# real risk — what an attacker can do is request `rpc` scope as
# "Touchless", and the user still has to click Allow) or front-end the
# token exchange with a Cloudflare Worker.

DISCORD_API_BASE = "https://discord.com/api"
DISCORD_TOKEN_URL = f"{DISCORD_API_BASE}/oauth2/token"

DISCORD_RPC_SCOPES = ("rpc",)

# Cloudflare sits in front of discord.com and rejects requests with
# the default Python `Python-urllib/3.x` User-Agent — returns HTTP 403
# with Cloudflare error code 1010 ("banned browser signature"). Setting
# a real UA matching Discord's official guidance (`AppName/Version
# (url)`) gets through cleanly. Discord's API docs actually require
# this header for bots; we follow the same convention for OAuth
# token requests so we don't trip Cloudflare's bot heuristics.
_DISCORD_USER_AGENT = "Touchless/1.1 (https://touchless-control.pages.dev)"

TOKEN_TTL_SECONDS = 600_000.0  # Discord access tokens are valid 7 days
                               # (604_800 s). 600_000 ≈ 6.9 days so we
                               # refresh slightly before expiry.

# Frame opcodes — see Discord IPC docs.
_OPCODE_HANDSHAKE = 0
_OPCODE_FRAME = 1
_OPCODE_CLOSE = 2
_OPCODE_PING = 3
_OPCODE_PONG = 4

_PIPE_NAME_TEMPLATE = r"\\.\pipe\discord-ipc-{slot}"

# Per Discord's docs, AUTHORIZE waits for the user to click Allow in
# the desktop client. Two minutes is plenty for the dialog; longer
# than the Spotify 3-minute callback because the user is staring
# directly at Discord, not browsing email.
_AUTHORIZE_RESPONSE_TIMEOUT_S = 120.0


# --- Errors ----------------------------------------------------------------


class DiscordRpcError(RuntimeError):
    """Generic failure interacting with the Discord IPC pipe."""


class DiscordPipeClosed(DiscordRpcError):
    """Raised when the IPC pipe is read-closed mid-frame (Discord
    desktop client exited, user crashed, etc.). Callers should drop
    their cached connection and re-open on the next command."""


# --- DTOs ------------------------------------------------------------------


@dataclass(frozen=True)
class DiscordVoiceState:
    """Snapshot of the local user's mute/deafen state. Populated from
    `GET_VOICE_SETTINGS`. Used by the gesture HUD to show whether the
    user is currently muted before issuing a toggle."""

    muted: bool
    deafened: bool
    input_volume: float | None
    output_volume: float | None


# --- Controller ------------------------------------------------------------


class DiscordController:
    """v1 capabilities — mute, deafen, read voice state.

    All public methods are safe to call from any thread; the
    `_pipe_lock` serialises frame send/recv on the single underlying
    pipe handle. `authorize_full_scopes` MUST be called from a worker
    thread — it blocks for up to ~2 minutes waiting for the user to
    click Allow in the Discord client."""

    def __init__(
        self,
        *,
        token_paths: tuple[Path, ...] | None = None,
        env_paths: tuple[Path, ...] | None = None,
        request_timeout_seconds: float = 5.0,
    ) -> None:
        self._request_timeout_seconds = float(request_timeout_seconds)
        # Same available-on-Windows-only check as Spotify. The Discord
        # IPC pipe path format `\\.\pipe\...` is Windows-specific;
        # macOS/Linux use a different socket path (we'll wire that in
        # when the macOS port lands).
        self._available = os.name == "nt"
        self._message = "discord idle"
        self._client_id: str | None = None
        self._client_secret: str | None = None
        self._redirect_uri: str | None = None
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._token_issue_time: float | None = None
        self._token_path: Path | None = None
        self._env_path: Path | None = None
        self._user_id: str | None = None
        self._user_name: str | None = None
        # Cache of the last text-channel ID we navigated to via
        # `select_text_channel`. Used as the second-tier fallback for
        # voice-command "send X" with no destination — when we don't
        # know which channel the user has manually focused (Discord
        # RPC doesn't expose that), we use the channel WE most
        # recently focused via a Touchless command. Cleared on
        # disconnect.
        self._last_focused_text_channel: dict[str, Any] | None = None

        self._repo_root = Path(__file__).resolve().parents[3]
        self._token_paths = token_paths or self._default_token_paths()
        self._env_paths = env_paths or self._default_env_paths()

        # Pipe state. `_pipe` is a buffering=0 binary file handle when
        # connected, None otherwise. `_pipe_slot` records which
        # discord-ipc-N slot we ended up on (useful when Discord
        # restarts and we need to rediscover).
        self._pipe = None  # type: ignore[var-annotated]
        self._pipe_slot: int | None = None
        self._pipe_lock = threading.Lock()
        self._authenticated = False

        # Mirrors the Spotify "needs reauth" latch — flipped True when
        # a refresh fails (Discord revoked the token because the user
        # un-authorised the app at https://discord.com/settings/authorized-apps).
        # The MainWindow polls this and surfaces a one-shot toast.
        self._needs_reauth = False

        self._load_credentials()
        self._load_tokens()

    # --- Properties --------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    @property
    def message(self) -> str:
        return self._message

    @property
    def has_authorization(self) -> bool:
        """True if a saved access or refresh token was loaded. Used by
        the first-time-Discord-active prompt to skip the connect modal
        when the user has already authorised in a previous run."""
        return bool(self._refresh_token) or bool(self._access_token)

    def needs_reauth(self) -> bool:
        return self._needs_reauth

    def clear_reauth_flag(self) -> None:
        self._needs_reauth = False

    # --- Credentials + token persistence -----------------------------------

    def _default_token_paths(self) -> tuple[Path, ...]:
        # Persist tokens beside the Spotify token in
        # `~/Documents/Touchless/discord_auth_token.json`. Survives
        # app updates (installer doesn't touch Documents) and is
        # per-user. Repo-root fallback is for dev runs from source.
        try:
            docs = Path.home() / "Documents" / "Touchless"
            primary = docs / "discord_auth_token.json"
        except Exception:
            primary = self._repo_root / "discord_auth_token.json"
        return (primary, self._repo_root / "discord_auth_token.json")

    def _default_env_paths(self) -> tuple[Path, ...]:
        # Match Spotify's env-file discovery order. Each path is
        # parsed lazily; first one with DISCORD_CLIENT_ID wins.
        candidates = [
            self._repo_root / ".env",
            Path.home() / "Documents" / "Touchless" / ".env",
        ]
        return tuple(candidates)

    def _resolve_persistent_token_path(self) -> Path:
        """Pick the first writable token-path candidate. Walks the
        configured `_token_paths` in order and tries to create the
        parent directory + touch a write probe. The Documents path
        is preferred; the repo-root fallback is for source dev runs
        where Documents may not exist (CI, fresh dev VMs)."""
        for candidate in self._token_paths:
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                # Probe write — opening for append creates if missing
                # without truncating anything we might have written
                # earlier this session.
                with candidate.open("a", encoding="utf-8"):
                    pass
                return candidate
            except Exception:
                continue
        # Last resort: return the first candidate even if creation
        # failed. _save_tokens will swallow the write error.
        return self._token_paths[0]

    def _load_credentials(self) -> None:
        """Resolve client_id / client_secret / redirect_uri for the
        Discord OAuth flow. Resolution order:

          1. Per-user setting (config.discord_client_id +
             config.discord_client_secret) supplied via the in-app
             Discord setup wizard. This is the path every shipped
             user takes — Discord's `rpc` scope only works for the
             OWNER of the registered app, so we need every user to
             register their OWN free Discord Dev app and paste both
             values into the wizard. Same per-user-app pattern as
             Spotify, different motivation (Spotify caps shared
             apps at 5-25 users; Discord rate-limits non-owner rpc
             to zero).
          2. Env vars (TOUCHLESS_DISCORD_CLIENT_ID + ...SECRET, or the
             plain DISCORD_* equivalents). For dev work.
          3. First .env file in `_env_paths` that defines
             DISCORD_CLIENT_ID + DISCORD_CLIENT_SECRET. Also for
             dev work — that's the developer's own dev-portal app
             baked in via repo-root .env, gitignored.
          4. Embedded default for client_id only. The secret has no
             embedded fallback — without it we can't do the
             token-exchange step and authorize_full_scopes will
             report "discord secret missing".
        """
        # Path 1: per-user credentials from the setup wizard.
        try:
            from ..config.app_config import load_config as _load_config
            user_cfg = _load_config()
            user_client_id = str(getattr(user_cfg, "discord_client_id", "") or "").strip()
            user_client_secret = str(getattr(user_cfg, "discord_client_secret", "") or "").strip()
        except Exception:
            user_client_id = ""
            user_client_secret = ""
        if user_client_id and user_client_secret:
            self._client_id = user_client_id
            self._client_secret = user_client_secret
            self._redirect_uri = _DEFAULT_DISCORD_REDIRECT_URI
            return

        env_client_id = (
            os.getenv("TOUCHLESS_DISCORD_CLIENT_ID")
            or os.getenv("DISCORD_CLIENT_ID")
        )
        env_client_secret = (
            os.getenv("TOUCHLESS_DISCORD_CLIENT_SECRET")
            or os.getenv("DISCORD_CLIENT_SECRET")
        )
        env_redirect_uri = (
            os.getenv("TOUCHLESS_DISCORD_REDIRECT_URI")
            or os.getenv("DISCORD_REDIRECT_URI")
        )
        if env_client_id and env_client_secret:
            self._client_id = env_client_id
            self._client_secret = env_client_secret
            self._redirect_uri = env_redirect_uri or _DEFAULT_DISCORD_REDIRECT_URI
            return

        for path in self._env_paths:
            values = self._parse_env_file(path)
            if values.get("DISCORD_CLIENT_ID") and values.get("DISCORD_CLIENT_SECRET"):
                self._env_path = path
                self._client_id = values["DISCORD_CLIENT_ID"]
                self._client_secret = values["DISCORD_CLIENT_SECRET"]
                self._redirect_uri = values.get(
                    "DISCORD_REDIRECT_URI", _DEFAULT_DISCORD_REDIRECT_URI
                )
                return

        # Fall through — public client_id only; no secret.
        # authorize_full_scopes will report the missing secret cleanly.
        self._client_id = _DEFAULT_DISCORD_CLIENT_ID
        self._client_secret = None
        self._redirect_uri = _DEFAULT_DISCORD_REDIRECT_URI

    @staticmethod
    def _parse_env_file(path: Path) -> dict[str, str]:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return {}
        result: dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip('"').strip("'")
        return result

    def _load_tokens(self) -> None:
        for path in self._token_paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            access_token = data.get("access_token")
            refresh_token = data.get("refresh_token")
            if not access_token and not refresh_token:
                continue
            self._token_path = path
            self._access_token = access_token
            self._refresh_token = refresh_token
            issue_time = data.get("issue_time")
            self._token_issue_time = (
                float(issue_time) if isinstance(issue_time, (int, float)) else None
            )
            self._needs_reauth = False
            return

    def _save_tokens(self) -> None:
        if self._token_path is None:
            self._token_path = self._resolve_persistent_token_path()
        payload = {
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "issue_time": self._token_issue_time or time.time(),
        }
        try:
            self._token_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            _LOG.exception("[discord-auth] failed to persist tokens to %s", self._token_path)

    # --- Pipe IO -----------------------------------------------------------

    def _open_ipc_pipe(self) -> None:
        """Probe `\\\\.\\pipe\\discord-ipc-0` through `-9` for an open
        Discord IPC pipe and bind `self._pipe`. The Discord desktop
        client claims slot 0 on launch and shifts to 1, 2, ... when
        slot 0 is taken (e.g. multiple Discord installs). 0 is the
        common case."""
        if self._pipe is not None:
            return
        last_err: Exception | None = None
        for slot in range(10):
            path = _PIPE_NAME_TEMPLATE.format(slot=slot)
            try:
                # buffering=0 — Discord IPC needs unbuffered writes;
                # any Python-side buffering would defer the frame
                # until the next read, which never comes because
                # we're waiting for a response.
                pipe = open(path, "rb+", buffering=0)
            except FileNotFoundError:
                continue
            except OSError as exc:
                last_err = exc
                continue
            self._pipe = pipe
            self._pipe_slot = slot
            self._message = f"discord pipe open (slot {slot})"
            return
        # No pipe found — Discord desktop client probably not running.
        self._pipe = None
        self._pipe_slot = None
        msg = "discord client not running (no IPC pipe found)"
        if last_err is not None:
            msg = f"{msg}: {last_err}"
        self._message = msg
        raise DiscordRpcError(msg)

    def _close_ipc_pipe(self) -> None:
        if self._pipe is None:
            return
        try:
            self._pipe.close()
        except Exception:
            pass
        self._pipe = None
        self._pipe_slot = None
        self._authenticated = False

    def _send_frame(self, opcode: int, payload: dict[str, Any]) -> None:
        """Frame a JSON payload + ship it. Caller must hold
        `_pipe_lock`."""
        if self._pipe is None:
            raise DiscordRpcError("pipe not open")
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = opcode.to_bytes(4, "little") + len(body).to_bytes(4, "little")
        try:
            self._pipe.write(header + body)
        except OSError as exc:
            self._close_ipc_pipe()
            raise DiscordPipeClosed(f"write failed: {exc}") from exc

    def _recv_frame(self) -> tuple[int, dict[str, Any]]:
        """Read one frame and return (opcode, parsed-JSON-payload).
        Blocking. Caller must hold `_pipe_lock`."""
        if self._pipe is None:
            raise DiscordRpcError("pipe not open")
        header = self._read_exact(8)
        opcode = int.from_bytes(header[:4], "little")
        length = int.from_bytes(header[4:], "little")
        body = self._read_exact(length) if length else b""
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError as exc:
            raise DiscordRpcError(f"malformed JSON payload: {exc}") from exc
        return opcode, data

    def _read_exact(self, n: int) -> bytes:
        if self._pipe is None:
            raise DiscordRpcError("pipe not open")
        if n == 0:
            return b""
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self._pipe.read(n - len(buf))
            except OSError as exc:
                self._close_ipc_pipe()
                raise DiscordPipeClosed(f"read failed: {exc}") from exc
            if not chunk:
                # EOF — Discord client closed the pipe. Surface as
                # DiscordPipeClosed so the caller can reconnect on
                # next command.
                self._close_ipc_pipe()
                raise DiscordPipeClosed("pipe closed by remote")
            buf.extend(chunk)
        return bytes(buf)

    # --- High-level handshake + command --------------------------------------

    def _handshake(self) -> dict[str, Any]:
        """Open the pipe (if not already) and send the handshake
        frame. Returns the READY payload (containing user info +
        version). Caller must hold `_pipe_lock`."""
        self._open_ipc_pipe()
        self._send_frame(_OPCODE_HANDSHAKE, {"v": 1, "client_id": self._client_id})
        opcode, payload = self._recv_frame()
        if opcode != _OPCODE_FRAME:
            raise DiscordRpcError(
                f"handshake: expected FRAME opcode (1), got {opcode}"
            )
        # READY event arrives as a regular FRAME with cmd="DISPATCH"
        # and evt="READY". Anything else is a protocol violation by
        # the client side (or we connected to a non-Discord pipe).
        if payload.get("cmd") != "DISPATCH" or payload.get("evt") != "READY":
            raise DiscordRpcError(f"handshake: unexpected READY frame: {payload}")
        user = (payload.get("data") or {}).get("user") or {}
        self._user_id = user.get("id")
        self._user_name = user.get("username")
        return payload

    def _send_command(
        self,
        cmd: str,
        args: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a command frame, wait for the matching nonce response.
        Caller must hold `_pipe_lock`. Returns the response payload
        (the full envelope, not just the `data` field). Raises
        `DiscordRpcError` on protocol errors or if the response carries
        an `evt:"ERROR"` field."""
        nonce = secrets.token_hex(8)
        frame = {"cmd": cmd, "args": args or {}, "nonce": nonce}
        self._send_frame(_OPCODE_FRAME, frame)

        deadline = time.monotonic() + (timeout if timeout is not None else self._request_timeout_seconds)
        while True:
            if time.monotonic() > deadline:
                raise DiscordRpcError(f"timed out waiting for {cmd!r} response")
            opcode, payload = self._recv_frame()
            if opcode == _OPCODE_PING:
                # Respect PING liveness probes from Discord. We don't
                # initiate PINGs but if we see one, mirror it back as
                # a PONG so Discord doesn't drop the pipe.
                self._send_frame(_OPCODE_PONG, payload)
                continue
            if opcode == _OPCODE_CLOSE:
                self._close_ipc_pipe()
                raise DiscordPipeClosed(f"client sent CLOSE: {payload}")
            if opcode != _OPCODE_FRAME:
                # Unknown opcode — keep reading; Discord may have
                # introduced a new event type that doesn't concern us.
                continue
            if payload.get("nonce") != nonce:
                # Out-of-band event (e.g. VOICE_STATE_UPDATE). Not the
                # response we're waiting for; ignore and keep reading.
                continue
            if payload.get("evt") == "ERROR":
                err_code = (payload.get("data") or {}).get("code")
                err_msg = (payload.get("data") or {}).get("message")
                raise DiscordRpcError(f"discord rpc error {err_code}: {err_msg}")
            return payload

    # --- Authorize / authenticate ------------------------------------------

    def authorize_full_scopes(
        self,
        *,
        timeout_seconds: float = _AUTHORIZE_RESPONSE_TIMEOUT_S,
    ) -> bool:
        """Run the one-time consent flow:
          - handshake the pipe
          - send AUTHORIZE → user clicks Allow in Discord client
          - exchange the returned code for access + refresh tokens via
            the HTTPS token endpoint
          - send AUTHENTICATE with the access token
          - persist tokens

        Long-blocking — call from a worker thread. Returns True on
        success."""
        if not self._available:
            self._message = "discord unavailable on this platform"
            return False
        if not self._client_id:
            self._message = "discord client_id missing"
            return False
        if not self._client_secret:
            self._message = "discord client_secret missing (set DISCORD_CLIENT_SECRET in .env)"
            return False

        with self._pipe_lock:
            try:
                self._handshake()
                _LOG.info(
                    "[discord-auth] handshake ok (slot %s, user %s / %s)",
                    self._pipe_slot, self._user_id, self._user_name,
                )
                response = self._send_command(
                    "AUTHORIZE",
                    {"client_id": self._client_id, "scopes": list(DISCORD_RPC_SCOPES)},
                    timeout=timeout_seconds,
                )
            except DiscordRpcError as exc:
                self._message = f"discord authorize failed: {exc}"
                _LOG.error("[discord-auth] authorize failed: %s", exc)
                self._close_ipc_pipe()
                return False

            code = (response.get("data") or {}).get("code")
            if not code:
                self._message = f"discord authorize: no code in response: {response}"
                _LOG.error("[discord-auth] authorize: no code in response: %s", response)
                self._close_ipc_pipe()
                return False

            # Token exchange happens OUTSIDE the pipe but we keep the
            # lock held — we don't want another thread sneaking in a
            # command between AUTHORIZE and AUTHENTICATE.
            try:
                token_data = self._exchange_code_for_token(code)
            except DiscordRpcError as exc:
                self._message = f"discord token exchange failed: {exc}"
                _LOG.error("[discord-auth] token exchange failed: %s", exc)
                self._close_ipc_pipe()
                return False

            self._access_token = token_data.get("access_token")
            self._refresh_token = token_data.get("refresh_token")
            self._token_issue_time = time.time()
            self._needs_reauth = False
            self._save_tokens()

            try:
                self._send_command(
                    "AUTHENTICATE", {"access_token": self._access_token}
                )
            except DiscordRpcError as exc:
                self._message = f"discord authenticate failed: {exc}"
                _LOG.error("[discord-auth] authenticate failed: %s", exc)
                self._close_ipc_pipe()
                return False

            self._authenticated = True
            self._message = f"discord ready: {self._user_name or 'connected'}"
            _LOG.info("[discord-auth] authenticated as %s", self._user_name)
            return True

    def _exchange_code_for_token(self, code: str) -> dict[str, Any]:
        body = urllib_parse.urlencode(
            {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "redirect_uri": self._redirect_uri,
            }
        ).encode("utf-8")
        req = urllib_request.Request(
            DISCORD_TOKEN_URL,
            data=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": _DISCORD_USER_AGENT,
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self._request_timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DiscordRpcError(
                f"token endpoint HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib_error.URLError, OSError) as exc:
            raise DiscordRpcError(f"token endpoint unreachable: {exc}") from exc
        if "access_token" not in payload:
            raise DiscordRpcError(f"token endpoint returned no access_token: {payload}")
        return payload

    def _refresh_access_token(self) -> bool:
        if not self._refresh_token or not self._client_id or not self._client_secret:
            return False
        body = urllib_parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            }
        ).encode("utf-8")
        req = urllib_request.Request(
            DISCORD_TOKEN_URL,
            data=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": _DISCORD_USER_AGENT,
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self._request_timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            # 400 / 401 here means Discord revoked the refresh token —
            # user un-authorised the app, or the app's permissions
            # changed, etc. Flip the latch so the UI can prompt for a
            # fresh consent.
            if exc.code in (400, 401):
                self._needs_reauth = True
            _LOG.warning("[discord-auth] refresh failed: HTTP %s", exc.code)
            return False
        except (urllib_error.URLError, OSError) as exc:
            _LOG.warning("[discord-auth] refresh transport error: %s", exc)
            return False
        new_access = payload.get("access_token")
        if not new_access:
            return False
        self._access_token = new_access
        # Discord rotates refresh tokens on refresh — keep the new one.
        new_refresh = payload.get("refresh_token")
        if new_refresh:
            self._refresh_token = new_refresh
        self._token_issue_time = time.time()
        self._save_tokens()
        return True

    def _ensure_authenticated(self) -> bool:
        """Bring the controller to authenticated state, refreshing or
        re-handshaking as needed. Returns True if ready to issue
        commands."""
        if not self._available:
            self._message = "discord unavailable on this platform"
            return False
        if not self._client_id:
            self._message = "discord client_id missing"
            return False
        if not self._access_token and not self._refresh_token:
            self._message = "discord not authorised (run setup)"
            return False

        # Refresh if the access token is older than TOKEN_TTL.
        if (
            self._refresh_token
            and self._token_issue_time
            and time.time() - self._token_issue_time > TOKEN_TTL_SECONDS
        ):
            self._refresh_access_token()

        with self._pipe_lock:
            if self._authenticated and self._pipe is not None:
                return True
            try:
                self._handshake()
                self._send_command(
                    "AUTHENTICATE", {"access_token": self._access_token}
                )
                self._authenticated = True
                self._message = f"discord ready: {self._user_name or 'connected'}"
                return True
            except DiscordRpcError as exc:
                self._message = f"discord reconnect failed: {exc}"
                _LOG.warning("[discord] reconnect failed: %s", exc)
                self._close_ipc_pipe()
                # If authenticate specifically failed, the token is
                # probably stale. Try a one-shot refresh + re-handshake.
                if "AUTHENTICATE" in str(exc) and self._refresh_access_token():
                    try:
                        self._handshake()
                        self._send_command(
                            "AUTHENTICATE", {"access_token": self._access_token}
                        )
                        self._authenticated = True
                        return True
                    except DiscordRpcError as inner:
                        _LOG.warning("[discord] post-refresh reconnect failed: %s", inner)
                        self._close_ipc_pipe()
                return False

    # --- Commands ----------------------------------------------------------

    def get_voice_settings(self) -> DiscordVoiceState | None:
        """Read the local user's current voice settings. Returns None
        on error."""
        if not self._ensure_authenticated():
            return None
        with self._pipe_lock:
            try:
                response = self._send_command("GET_VOICE_SETTINGS")
            except DiscordRpcError as exc:
                self._message = f"discord get_voice_settings failed: {exc}"
                return None
        data = response.get("data") or {}
        input_volume = (data.get("input") or {}).get("volume")
        output_volume = (data.get("output") or {}).get("volume")
        return DiscordVoiceState(
            muted=bool(data.get("mute", False)),
            deafened=bool(data.get("deaf", False)),
            input_volume=float(input_volume) if isinstance(input_volume, (int, float)) else None,
            output_volume=float(output_volume) if isinstance(output_volume, (int, float)) else None,
        )

    def set_self_mute(self, muted: bool) -> bool:
        if not self._ensure_authenticated():
            return False
        with self._pipe_lock:
            try:
                self._send_command("SET_VOICE_SETTINGS", {"mute": bool(muted)})
            except DiscordRpcError as exc:
                self._message = f"discord set_self_mute failed: {exc}"
                return False
        self._message = "discord mute on" if muted else "discord mute off"
        return True

    def set_self_deafen(self, deafened: bool) -> bool:
        if not self._ensure_authenticated():
            return False
        with self._pipe_lock:
            try:
                self._send_command("SET_VOICE_SETTINGS", {"deaf": bool(deafened)})
            except DiscordRpcError as exc:
                self._message = f"discord set_self_deafen failed: {exc}"
                return False
        self._message = "discord deafen on" if deafened else "discord deafen off"
        return True

    def toggle_self_mute(self) -> bool:
        state = self.get_voice_settings()
        if state is None:
            return False
        return self.set_self_mute(not state.muted)

    def toggle_self_deafen(self) -> bool:
        state = self.get_voice_settings()
        if state is None:
            return False
        return self.set_self_deafen(not state.deafened)

    def disconnect(self) -> None:
        """Drop the pipe cleanly. Safe to call from any thread."""
        with self._pipe_lock:
            self._close_ipc_pipe()

    # --- Channel / guild discovery + navigation ----------------------------

    # Channel type constants (Discord docs, OAuth2 channel types).
    CHANNEL_GUILD_TEXT = 0
    CHANNEL_DM = 1
    CHANNEL_GUILD_VOICE = 2
    CHANNEL_GROUP_DM = 3
    CHANNEL_GUILD_CATEGORY = 4
    CHANNEL_GUILD_ANNOUNCEMENT = 5
    CHANNEL_GUILD_STAGE_VOICE = 13

    _VOICE_CHANNEL_TYPES = frozenset({CHANNEL_GUILD_VOICE, CHANNEL_GUILD_STAGE_VOICE})
    _TEXT_CHANNEL_TYPES = frozenset({CHANNEL_GUILD_TEXT, CHANNEL_GUILD_ANNOUNCEMENT})

    def list_guilds(self) -> list[dict[str, Any]]:
        """Return the user's guild list. List of dicts with at least
        `id` (str snowflake) and `name`. Empty list on failure."""
        if not self._ensure_authenticated():
            return []
        with self._pipe_lock:
            try:
                resp = self._send_command("GET_GUILDS")
            except DiscordRpcError as exc:
                self._message = f"discord get_guilds failed: {exc}"
                _LOG.warning("[discord] get_guilds failed: %s", exc)
                return []
        data = resp.get("data") or {}
        guilds = data.get("guilds")
        return list(guilds) if isinstance(guilds, list) else []

    def get_guild_channels(self, guild_id: str) -> list[dict[str, Any]]:
        """Return the channels in a specific guild. Each dict carries
        at least `id`, `name`, `type` (int — see CHANNEL_* constants)."""
        if not self._ensure_authenticated():
            return []
        with self._pipe_lock:
            try:
                resp = self._send_command("GET_CHANNELS", {"guild_id": str(guild_id)})
            except DiscordRpcError as exc:
                self._message = f"discord get_channels failed: {exc}"
                _LOG.warning("[discord] get_channels(%s) failed: %s", guild_id, exc)
                return []
        data = resp.get("data") or {}
        channels = data.get("channels")
        return list(channels) if isinstance(channels, list) else []

    def find_voice_channel_by_name(self, name: str) -> dict[str, Any] | None:
        """Fuzzy-match `name` against every voice channel across the
        user's guilds. Match priority: exact (case-insensitive) >
        prefix > substring. Returns the first ranked dict or None."""
        return self._find_channel_by_name(name, types=self._VOICE_CHANNEL_TYPES)

    def find_text_channel_by_name(self, name: str) -> dict[str, Any] | None:
        return self._find_channel_by_name(name, types=self._TEXT_CHANNEL_TYPES)

    def find_guild_by_name(self, name: str) -> dict[str, Any] | None:
        """Fuzzy-match `name` against the user's guild list. Same
        ranking as `_find_channel_by_name` (exact > prefix > substring)."""
        target = (name or "").strip().lower()
        if not target:
            return None
        guilds = self.list_guilds()
        if not guilds:
            return None
        return self._best_name_match(target, guilds)

    def _find_channel_by_name(
        self, name: str, *, types: frozenset[int]
    ) -> dict[str, Any] | None:
        target = (name or "").strip().lower()
        if not target:
            return None
        guilds = self.list_guilds()
        if not guilds:
            return None
        candidates: list[dict[str, Any]] = []
        for guild in guilds:
            guild_id = guild.get("id")
            if not guild_id:
                continue
            for channel in self.get_guild_channels(guild_id):
                if channel.get("type") not in types:
                    continue
                # Decorate with guild context so the executor can
                # mention which server the match came from when there
                # are duplicate-named channels.
                enriched = dict(channel)
                enriched["_guild_id"] = guild_id
                enriched["_guild_name"] = guild.get("name", "")
                candidates.append(enriched)
        return self._best_name_match(target, candidates)

    @staticmethod
    def _best_name_match(
        target: str, items: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not items:
            return None
        # Priority: exact, then prefix, then substring. Within each
        # bucket, the first one encountered wins (preserves guild
        # ordering — which is the user's own sidebar ordering).
        exact: list[dict[str, Any]] = []
        prefix: list[dict[str, Any]] = []
        substr: list[dict[str, Any]] = []
        for item in items:
            name = str(item.get("name", "")).strip().lower()
            if not name:
                continue
            if name == target:
                exact.append(item)
            elif name.startswith(target):
                prefix.append(item)
            elif target in name:
                substr.append(item)
        for bucket in (exact, prefix, substr):
            if bucket:
                return bucket[0]
        return None

    def select_voice_channel(self, channel_id: str) -> bool:
        """Join the voice channel identified by snowflake ID."""
        if not self._ensure_authenticated():
            return False
        with self._pipe_lock:
            try:
                self._send_command(
                    "SELECT_VOICE_CHANNEL", {"channel_id": str(channel_id)}
                )
            except DiscordRpcError as exc:
                self._message = f"discord select_voice_channel failed: {exc}"
                _LOG.warning("[discord] select_voice_channel(%s) failed: %s", channel_id, exc)
                return False
        self._message = f"discord joined voice channel {channel_id}"
        return True

    def leave_voice_channel(self) -> bool:
        """Disconnect from the current voice channel (if any). RPC
        contract: `SELECT_VOICE_CHANNEL` with `channel_id: null`."""
        if not self._ensure_authenticated():
            return False
        with self._pipe_lock:
            try:
                self._send_command("SELECT_VOICE_CHANNEL", {"channel_id": None})
            except DiscordRpcError as exc:
                self._message = f"discord leave_voice_channel failed: {exc}"
                _LOG.warning("[discord] leave_voice_channel failed: %s", exc)
                return False
        self._message = "discord left voice channel"
        return True

    def select_text_channel(
        self, channel_id: str, *, channel_info: dict[str, Any] | None = None
    ) -> bool:
        """Focus the given text channel in the Discord UI. Same RPC
        contract as voice, with `SELECT_TEXT_CHANNEL`. Optional
        `channel_info` is the full dict from `find_text_channel_by_name`
        (or similar) — when supplied, we cache it as the "last focused"
        target so the voice-command `send X` fallback chain can route
        a destination-less message back to this channel."""
        if not self._ensure_authenticated():
            return False
        with self._pipe_lock:
            try:
                self._send_command(
                    "SELECT_TEXT_CHANNEL", {"channel_id": str(channel_id)}
                )
            except DiscordRpcError as exc:
                self._message = f"discord select_text_channel failed: {exc}"
                _LOG.warning("[discord] select_text_channel(%s) failed: %s", channel_id, exc)
                return False
        self._message = f"discord focused text channel {channel_id}"
        if channel_info is not None:
            self._last_focused_text_channel = dict(channel_info)
            self._last_focused_text_channel.setdefault("id", str(channel_id))
        else:
            self._last_focused_text_channel = {"id": str(channel_id)}
        return True

    def get_selected_voice_channel(self) -> dict[str, Any] | None:
        """Return the user's currently joined voice channel as a dict
        (id, name, guild_id, ...) or None if not in a call. Used by
        the voice-command 'send X' fallback chain — when the user is
        in a voice call, an in-call message is the most natural
        destination for a destination-less send."""
        if not self._ensure_authenticated():
            return None
        with self._pipe_lock:
            try:
                resp = self._send_command("GET_SELECTED_VOICE_CHANNEL")
            except DiscordRpcError as exc:
                _LOG.debug("[discord] get_selected_voice_channel failed: %s", exc)
                return None
        data = resp.get("data") or {}
        # Discord returns null for `data` when the user isn't in a
        # voice channel; treat that as "not in call".
        if not data or not data.get("id"):
            return None
        return data

    def default_send_target(self) -> dict[str, Any] | None:
        """Pick a fallback target for a destination-less 'send X' voice
        command. Priority chain:

          1. Currently joined voice channel — Discord lets you post
             messages into the voice channel itself, which appear in
             the in-call text chat. Most natural for "I'm in a call,
             type a quick message" use cases.
          2. Last text channel we focused via `select_text_channel` —
             survives between commands within a single session.
          3. None — caller should prompt the user.

        Returns a channel-info dict (carrying at least `id` and `name`),
        or None when nothing matches."""
        voice = self.get_selected_voice_channel()
        if voice is not None and voice.get("id"):
            # Decorate so the executor's HUD message can name the channel.
            return {
                "id": voice["id"],
                "name": voice.get("name", ""),
                "_guild_name": voice.get("guild_id") or "voice channel",
                "_via": "voice-call",
            }
        if self._last_focused_text_channel is not None and self._last_focused_text_channel.get("id"):
            target = dict(self._last_focused_text_channel)
            target.setdefault("_via", "last-focused")
            return target
        return None

    # --- Window focus (Win32) ---------------------------------------------

    def focus_discord_window(self) -> bool:
        """Bring the Discord desktop window to the foreground.

        Required before pasting a message via TextInputController:
        `SELECT_TEXT_CHANNEL` navigates Discord's UI internally but
        DOES NOT raise its window to the foreground. The user might
        still be looking at Touchless or any other app, in which case
        a paste operation would deliver the keystrokes to that wrong
        window instead of Discord's compose box.

        Windows-only. Enumerates top-level windows looking for
        Discord's signature: Chrome_WidgetWin_1 class (Electron) +
        title containing "Discord" (with optional notification-count
        prefix like "(3) Discord"). Returns True if any match was
        successfully raised to the foreground.
        """
        if not self._available:
            return False
        handles = self._enumerate_discord_window_handles()
        for hwnd in handles:
            if self._activate_window_handle(hwnd):
                self._message = "discord window focused"
                return True
        self._message = "discord window not found"
        return False

    @staticmethod
    def _enumerate_discord_window_handles() -> list[int]:
        """Walk top-level windows and return hwnds whose class +
        title match Discord. Same approach as
        spotify_controller._spotify_window_handles, simplified
        (Discord ships only one shell so we don't need the dual-class
        check Spotify uses for its old WinUI build)."""
        if os.name != "nt":
            return []
        try:
            user32 = ctypes.windll.user32
        except Exception:
            return []
        handles: list[int] = []
        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, wintypes.HWND, wintypes.LPARAM
        )

        def _enum(hwnd: wintypes.HWND, _lparam: wintypes.LPARAM) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            title_len = int(user32.GetWindowTextLengthW(hwnd))
            if title_len <= 0:
                return True
            title_buf = ctypes.create_unicode_buffer(title_len + 1)
            user32.GetWindowTextW(hwnd, title_buf, title_len + 1)
            title = (title_buf.value or "").lower()
            if "discord" not in title:
                return True
            class_buf = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, class_buf, 64)
            class_name = class_buf.value or ""
            # Discord is an Electron app — class name is
            # Chrome_WidgetWin_1 just like Spotify, but only one shell
            # per install.
            if not class_name.startswith("Chrome_WidgetWin"):
                return True
            handles.append(int(hwnd))
            return True

        try:
            user32.EnumWindows(EnumWindowsProc(_enum), 0)
        except Exception:
            return []
        return handles

    @staticmethod
    def _activate_window_handle(hwnd: int) -> bool:
        if os.name != "nt":
            return False
        try:
            user32 = ctypes.windll.user32
        except Exception:
            return False
        try:
            # SW_RESTORE so a minimised Discord window pops back open.
            user32.ShowWindow(wintypes.HWND(hwnd), 9)
            user32.BringWindowToTop(wintypes.HWND(hwnd))
            return bool(user32.SetForegroundWindow(wintypes.HWND(hwnd)))
        except Exception:
            return False

    def focus_guild_first_text_channel(self, guild_id: str) -> bool:
        """Convenience helper: switch the UI to the guild by focusing
        its first available text channel. Used by the "open server X"
        voice command — Discord doesn't have a direct 'select guild'
        RPC, but focusing any text channel in the guild navigates the
        sidebar to it."""
        channels = self.get_guild_channels(guild_id)
        for channel in channels:
            if channel.get("type") in self._TEXT_CHANNEL_TYPES:
                channel_id = channel.get("id")
                if channel_id:
                    return self.select_text_channel(channel_id)
        return False

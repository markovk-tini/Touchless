from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import platform
import re
import secrets
import subprocess
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import parse as urllib_parse

from ..utils.subprocess_utils import launch_external
from urllib import request as urllib_request

import psutil


SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_TTL_SECONDS = 3500.0
SW_RESTORE = 9

# Default Spotify app client_id, baked into every shipped Touchless
# build. Spotify treats client_id as PUBLIC — embedding it is the
# documented practice for desktop apps that can't keep a secret.
# Auth flow is PKCE (Proof Key for Code Exchange) so no client_secret
# is required: each user's auth dance uses a per-session
# code_verifier instead, and refresh tokens work the same way.
# Override via env var (TOUCHLESS_SPOTIFY_CLIENT_ID / CLIENT_ID) or
# a .env file in repo root / ~/Documents/Touchless/ for dev work
# against a different Spotify dev-console app.
_DEFAULT_SPOTIFY_CLIENT_ID = "7763a7c443604776b0060da01428686f"
_DEFAULT_SPOTIFY_REDIRECT_URI = "http://127.0.0.1:5000/callback"

SPOTIFY_SCOPES = (
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "user-library-read",
    "user-library-modify",
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-read-private",
    "user-read-email",
)


@dataclass(frozen=True)
class SpotifyTrackDetails:
    song_name: str
    artist_names: str
    album_name: str | None
    playlist_name: str | None
    device_name: str | None
    device_type: str | None
    is_playing: bool
    shuffle_enabled: bool
    repeat_mode: str | None
    progress_ms: int | None
    duration_ms: int | None
    context_type: str | None

    def summary(self) -> str:
        playlist_text = self.playlist_name if self.playlist_name else "not in playlist"
        playback_text = "playing" if self.is_playing else "paused"
        album_text = self.album_name or "unknown album"
        device_text = self.device_name or "unknown device"
        repeat_text = self.repeat_mode or "off"
        return (
            f"Song: {self.song_name} | Artist: {self.artist_names} | Album: {album_text} | "
            f"Playlist: {playlist_text} | Device: {device_text} | State: {playback_text} | "
            f"Shuffle: {'on' if self.shuffle_enabled else 'off'} | Repeat: {repeat_text}"
        )


@dataclass(frozen=True)
class SpotifyVoiceRequest:
    raw_text: str
    query: str
    preferred_types: tuple[str, ...]


class SpotifyController:
    def __init__(
        self,
        *,
        token_paths: tuple[Path, ...] | None = None,
        env_paths: tuple[Path, ...] | None = None,
        executable_paths: tuple[Path, ...] | None = None,
        request_timeout_seconds: float = 5.0,
    ) -> None:
        self._request_timeout_seconds = float(request_timeout_seconds)
        _system = platform.system()
        # macOS controls the Spotify DESKTOP APP directly via AppleScript
        # (no OAuth/Web API) — available whenever Spotify.app is installed.
        # Windows keeps the Web-API path. Everything else stays unavailable.
        self._mac = _system == "Darwin"
        self._available = self._detect_mac_spotify_installed() if self._mac else (_system == "Windows")
        # Short-TTL cache for is_running() so per-frame mac gates don't scan the
        # full process table every frame (busted on launch — see launch paths).
        self._mac_running_cache: bool | None = None
        self._mac_running_cache_until: float = 0.0
        self._mac_running_cache_ttl: float = 1.5
        self._message = "spotify idle"
        self._client_id: str | None = None
        self._client_secret: str | None = None
        self._redirect_uri: str | None = None
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._token_issue_time: float | None = None
        self._token_path: Path | None = None
        self._env_path: Path | None = None
        self._device_id: str | None = None
        self._device_name: str | None = None
        self._repo_root = Path(__file__).resolve().parents[3]
        self._token_paths = token_paths or self._default_token_paths()
        self._env_paths = env_paths or self._default_env_paths()
        self._executable_paths = executable_paths or self._default_executable_paths()
        self._handles_cache: list[int] = []
        self._handles_cache_until = 0.0
        self._launch_in_flight = False
        self._launch_lock = threading.Lock()
        # Cache the "is an active Spotify device available?" answer
        # for a few seconds. The gesture pipeline checks this on
        # every dynamic-gesture decision (and on each is_active_for_
        # _wheel call), and each check fires a /me/player HTTPS
        # request — 50-200 ms per call, blocking the gesture worker
        # thread. The answer only changes when the user opens /
        # closes Spotify or transfers playback to another device,
        # neither of which happens at frame rate. 3 s is short
        # enough to feel responsive after the user opens Spotify
        # via right-hand 'two', long enough to absorb the 60 fps
        # gesture loop without making 60 HTTPS requests per second.
        self._active_device_cache: bool | None = None
        self._active_device_cache_until: float = 0.0
        self._active_device_cache_seconds: float = 3.0
        # Stale-while-revalidate latch for is_active_device_available().
        # When the 3 s cache expires while a hand is in frame, we used
        # to fire the 50-300 ms /me/player HTTP call on the calling
        # (UI) thread inside the gesture router's hot path -- visible
        # camera lag spikes every 3 s once Spotify started playing.
        # Now the call kicks the refresh on a background thread and
        # returns the previous cached value; the next gesture frame
        # reads the fresh result.
        self._active_device_refresh_in_flight: bool = False
        # Flips True when a stored refresh token fails (Spotify
        # revoked it server-side, the user changed password, or the
        # token simply aged out). The MainWindow polls this in its
        # per-frame debug handler and surfaces a one-shot 'reconnect
        # Spotify' toast so the user knows controls have silently
        # stopped working.
        self._needs_reauth: bool = False
        self._load_credentials()
        self._load_tokens()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def message(self) -> str:
        return self._message

    @property
    def has_authorization(self) -> bool:
        """True if a saved access or refresh token was loaded for the
        current user. Used by the Touchless first-time-Spotify-active
        prompt to skip the Allow/Don't Allow modal when the user has
        already authorised in a previous run."""
        return bool(self._refresh_token) or bool(self._access_token)

    # ---- macOS: control the Spotify desktop app via AppleScript ----------
    # The macOS Spotify app is natively scriptable, so no OAuth/Web API is
    # needed. Every property read/tell auto-LAUNCHES Spotify, so each script
    # guards with `if application "Spotify" is running` first; only that query
    # and `open -a` are launch-safe. Fields are joined with ASCII unit
    # separator (0x1F) — it never appears in track metadata.
    _MAC_SEP = "\x1f"

    def _mac_osascript(self, script: str, timeout: float = 4.0) -> tuple[bool, str, str]:
        """Run one AppleScript. Returns (ok, stdout_stripped, stderr_stripped)."""
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
        except Exception as exc:
            return False, "", f"{type(exc).__name__}: {exc}"
        return (proc.returncode == 0), (proc.stdout or "").strip(), (proc.stderr or "").strip()

    def _detect_mac_spotify_installed(self) -> bool:
        """True iff the Spotify app is installed. Filesystem-first (no side
        effects); LaunchServices id lookup as a fallback (does NOT launch)."""
        for candidate in (Path("/Applications/Spotify.app"), Path.home() / "Applications" / "Spotify.app"):
            try:
                if candidate.exists():
                    return True
            except Exception:
                pass
        ok, out, _ = self._mac_osascript('id of application "Spotify"')
        return ok and out == "com.spotify.client"

    def _mac_error_message(self, prefix: str, detail: str) -> str:
        """Map the TCC Automation denial (-1743) to a one-time friendly hint."""
        if "-1743" in detail or "Not authorized" in detail or "not allowed assistive" in detail:
            return f"{prefix}: allow Touchless to control Spotify in System Settings > Privacy & Security > Automation"
        return prefix

    def _mac_transport(self, command: str, *, launch: bool = False) -> bool:
        """Send a transport verb (playpause/play/pause/next track/previous
        track). launch=True opens+focuses Spotify first (for explicit play)."""
        if launch and not self.is_running():
            try:
                subprocess.run(["open", "-a", "Spotify"], capture_output=True, text=True, timeout=8.0, check=False)
            except Exception:
                pass
            for _ in range(20):  # ~2s for Spotify to accept Apple Events
                ok, out, _ = self._mac_osascript('application "Spotify" is running')
                if ok and out == "true":
                    self._mac_running_cache = None  # force gates to re-check
                    break
                time.sleep(0.1)
        script = (
            'if application "Spotify" is running then\n'
            f'\ttell application "Spotify" to {command}\n'
            'end if'
        )
        ok, _out, err = self._mac_osascript(script)
        if not ok:
            self._message = self._mac_error_message("spotify control failed", err)
        return ok

    def _mac_player_state(self) -> dict[str, Any] | None:
        """Synthesize the Web-API-shaped /me/player dict from one guarded
        AppleScript call, so get_playback_state/get_current_track_details/
        get_volume all work unchanged. None when Spotify isn't running."""
        sep = self._MAC_SEP
        script = (
            'if application "Spotify" is not running then\n'
            '\treturn "not-running"\n'
            'end if\n'
            'tell application "Spotify"\n'
            f'\tset d to (character id 31)\n'
            '\tif player state is playing then\n'
            '\t\tset st to "playing"\n'
            '\telse if player state is paused then\n'
            '\t\tset st to "paused"\n'
            '\telse\n'
            '\t\tset st to "stopped"\n'
            '\tend if\n'
            '\tset shuf to (shuffling as text)\n'
            '\tset rep to (repeating as text)\n'
            '\tset vol to (sound volume as text)\n'
            '\tif player state is stopped then\n'
            '\t\treturn st & d & "" & d & "" & d & "" & d & "" & d & "" & d & shuf & d & rep & d & "0" & d & vol\n'
            '\tend if\n'
            '\tset t to current track\n'
            '\treturn st & d & (name of t) & d & (artist of t) & d & (album of t) & d & ((duration of t) as text) & d & (spotify url of t) & d & shuf & d & rep & d & ((player position) as text) & d & vol\n'
            'end tell'
        )
        ok, out, err = self._mac_osascript(script)
        if not ok:
            self._message = self._mac_error_message("spotify not reachable", err)
            return None
        if out == "not-running":
            self._message = "spotify not running"
            return None
        parts = out.split(sep)
        if len(parts) < 10:
            return None
        state, name, artist, album, dur, uri, shuf, rep, pos, vol = parts[:10]

        def _to_int(text: str):
            try:
                return int(float(text))
            except (ValueError, TypeError):
                return None

        volume_percent = _to_int(vol)
        player: dict[str, Any] = {
            "is_playing": state == "playing",
            "shuffle_state": (shuf == "true"),
            # AppleScript `repeating` is a plain bool; map to the Web-API vocab
            # (no track-vs-context distinction available on the desktop app).
            "repeat_state": ("context" if rep == "true" else "off"),
            "device": {"name": "This Mac", "type": "Computer", "volume_percent": volume_percent},
            "context": {},
            "progress_ms": (int(float(pos) * 1000) if pos not in ("", "0") and _to_int(pos) is not None else (0 if pos == "0" else None)),
            "item": (
                {
                    "name": name,
                    "artists": [{"name": artist}] if artist else [],
                    "album": {"name": album},
                    "duration_ms": _to_int(dur),
                    "uri": uri,
                }
                if name
                else {}
            ),
        }
        self._message = "spotify track info"
        return player

    def ensure_ready(self, *, open_if_needed: bool = False) -> bool:
        if self._mac:
            if self.is_running():
                return True
            if open_if_needed:
                try:
                    subprocess.run(["open", "-a", "Spotify"], capture_output=True, text=True, timeout=8.0, check=False)
                except Exception:
                    pass
                for _ in range(20):
                    if self._is_running_uncached():
                        self._mac_running_cache = None  # force gates to re-check
                        return True
                    time.sleep(0.1)
            self._message = "spotify not running"
            return False
        if not self._available:
            self._message = "spotify unavailable on this platform"
            return False
        if not self._ensure_authenticated():
            return False

        devices = self._get_devices()
        if not devices and open_if_needed:
            launched = self.launch_spotify(hidden=True)
            if launched:
                devices = self._wait_for_devices()
        if not devices:
            self._message = "spotify device not available"
            return False

        active_device = next((device for device in devices if device.get("is_active")), None)
        if active_device is None:
            active_device = self._pick_device(devices)
            if active_device is None:
                self._message = "spotify device not available"
                return False
            if not self._transfer_playback(active_device.get("id"), play=False):
                self._message = "spotify device activation failed"
                return False

        self._device_id = active_device.get("id")
        self._device_name = active_device.get("name")
        self._message = f"spotify ready: {self._device_name or 'device ready'}"
        return True

    def launch_spotify(self, *, hidden: bool) -> bool:
        if self._mac:
            # `open -g -a` launches in the background (no focus steal) for the
            # tutorial hidden path; `open -a` foregrounds it otherwise.
            args = ["open", "-g", "-a", "Spotify"] if hidden else ["open", "-a", "Spotify"]
            try:
                proc = subprocess.run(args, capture_output=True, text=True, timeout=8.0, check=False)
            except Exception:
                self._message = "spotify launch failed"
                return False
            if proc.returncode != 0:
                self._message = "spotify launch failed"
                return False
            for _ in range(20):
                if self._is_running_uncached():
                    self._mac_running_cache = None  # force gates to re-check
                    self._message = "launching spotify"
                    return True
                time.sleep(0.1)
            self._message = "launching spotify"
            return True
        # The launcher used to be a single-line subprocess.Popen on
        # whichever Spotify.exe path existed first — including the
        # 0-byte App Execution Alias stub at
        # %LOCALAPPDATA%\Microsoft\WindowsApps\<package>\Spotify.exe.
        # That worked: Windows resolves the alias on CreateProcess
        # and the real Store Spotify boots. A later attempt to skip
        # the stub by file size was wrong — Popen on the stub does
        # work, and removing it broke launching for users who only
        # have the Store install. Now we Popen every candidate that
        # exists, regardless of size, and verify each launch by
        # polling for a real Spotify process before declaring it
        # the winner.
        def _attempt(fire: Callable[[], None]) -> bool:
            try:
                fire()
            except Exception:
                return False
            return self._wait_for_spotify_process(timeout_seconds=4.0)

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        for candidate in self._executable_paths:
            try:
                if not candidate.exists():
                    continue
            except Exception:
                continue
            startupinfo = None
            if hidden and hasattr(subprocess, "STARTUPINFO"):
                try:
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = 0
                except Exception:
                    startupinfo = None
            if _attempt(
                lambda c=candidate, si=startupinfo: subprocess.Popen(
                    [str(c)], startupinfo=si, creationflags=creationflags
                )
            ):
                self._message = "launching spotify"
                return True

        # ShellExecute fallbacks — only reached when no candidate
        # path was launchable. os.startfile self-inits the COM
        # apartment so it works from background threads where raw
        # ShellExecuteW silently drops the request. PowerShell is a
        # last resort because it spawns a new process and reads
        # like a dropper to behavioural AV.
        if _attempt(lambda: os.startfile("spotify")):
            self._message = "launching spotify"
            return True
        if _attempt(lambda: os.startfile("spotify:")):
            self._message = "launching spotify"
            return True
        if _attempt(
            lambda: subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Start-Process spotify:",
                ],
                creationflags=creationflags,
            )
        ):
            self._message = "launching spotify"
            return True

        self._message = "spotify launch path not found"
        return False

    def _wait_for_spotify_process(self, *, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() < deadline:
            if self._spotify_window_handles():
                return True
            if self._has_real_spotify_process():
                return True
            time.sleep(0.25)
        return self._has_real_spotify_process()

    def _has_real_spotify_process(self) -> bool:
        # is_running() returns True for any process whose name
        # contains 'spotify' — including Spotify-WebHelper.exe,
        # SpotifyMigrator, Spotify update services, even the
        # store's Spotify.exe stub host before the real client
        # spins up. For verifying a fresh launch we want to know
        # the *interactive* client started, so we accept the match
        # only when the executable has a meaningful size (>1MB).
        try:
            for proc in psutil.process_iter(["name", "exe"]):
                name = (proc.info.get("name") or "").lower()
                if name != "spotify.exe":
                    continue
                exe_path = proc.info.get("exe")
                if not exe_path:
                    return True
                try:
                    if Path(exe_path).stat().st_size > 1024 * 1024:
                        return True
                except Exception:
                    return True
        except Exception:
            return False
        return False

    def _is_running_uncached(self) -> bool:
        try:
            for proc in psutil.process_iter(["name"]):
                name = (proc.info.get("name") or "").lower()
                if "spotify" in name:
                    return True
        except Exception:
            return False
        return False

    def is_running(self) -> bool:
        # macOS gates (is_window_open / is_active_device_available) call this
        # per gesture frame. psutil.process_iter over every process is ~10-30 ms
        # — enough to visibly drop fps — so cache it briefly on mac. Windows
        # keeps the uncached scan (its window/device gates have their own
        # caches). Cache is busted on launch so 'two'/open sees the app quickly.
        if not self._mac:
            return self._is_running_uncached()
        now = time.monotonic()
        if self._mac_running_cache is not None and now < self._mac_running_cache_until:
            return self._mac_running_cache
        value = self._is_running_uncached()
        self._mac_running_cache = value
        self._mac_running_cache_until = now + self._mac_running_cache_ttl
        return value

    def is_window_open(self) -> bool:
        """Stricter than is_running(): True only when Spotify has at
        least one visible top-level window. Used by wheel/fist gates
        so Spotify protocol handlers, web helpers, or update services
        running in the background don't count as 'Spotify is running'
        for control purposes — only a real interactive Spotify
        window counts. is_running() is kept as-is for places that
        care about ANY spotify process (e.g. avoiding redundant
        launches in focus_or_open_window)."""
        if self._mac:
            # No cheap "has a visible window" check without Accessibility;
            # a running Spotify app is the usable-for-control signal on mac.
            return self.is_running()
        return bool(self._spotify_window_handles())

    def get_playback_state(self) -> bool | None:
        player = self.get_player_state()
        if not player:
            return None
        return bool(player.get("is_playing"))

    def get_player_state(self) -> dict[str, Any] | None:
        if self._mac:
            return self._mac_player_state()
        if not self._ensure_authenticated():
            return None
        status, payload = self._request_json("GET", "/me/player")
        if status == 204:
            self._message = "spotify inactive on device"
            return None
        if status != 200 or not isinstance(payload, dict):
            return None
        device = payload.get("device") or {}
        self._device_id = device.get("id") or self._device_id
        self._device_name = device.get("name") or self._device_name
        return payload

    def toggle_playback(self) -> bool:
        if self._mac:
            # Atomic — no need to read state first.
            if self._mac_transport("playpause"):
                self._message = "spotify play/pause"
                return True
            return False
        playback_state = self.get_playback_state()
        if playback_state is True:
            return self.pause()
        return self.play()

    def _device_params(self) -> dict[str, Any] | None:
        """Build a `?device_id=...` param dict from the device we
        activated during ensure_ready. Spotify Web API returns 404
        NO_ACTIVE_DEVICE on /me/player/play and friends when no
        device is currently 'active' from Spotify's perspective —
        which can happen for a few seconds after transferring
        playback to a freshly-launched desktop client. Routing the
        command explicitly to the device_id we set up bypasses
        that race entirely."""
        if not self._device_id:
            return None
        return {"device_id": self._device_id}

    def play(self) -> bool:
        if self._mac:
            if self._mac_transport("play", launch=True):
                self._message = "spotify play"
                return True
            return False
        if not self.ensure_ready(open_if_needed=True):
            self._message = "spotify play failed (not ready)"
            return False
        status, body = self._request_json("PUT", "/me/player/play", params=self._device_params())
        # Spotify can answer with 200 (with playback-state body) when
        # the request lands on the desktop client, in addition to the
        # documented 202/204. Treat all three as success so the
        # toast on the phone doesn't lie about a working action.
        if status in {200, 202, 204}:
            self._message = "spotify play"
            return True
        # Diagnostic: record the actual status so we can see in the
        # console what unexpected code came back. User-visible message
        # stays terse so the toast doesn't blow out.
        try:
            import sys as _sys
            _sys.stderr.write(f"[spotify] play got HTTP {status} body={body!r}\n")
            _sys.stderr.flush()
        except Exception:
            pass
        self._message = f"spotify play failed (status {status})"
        return False

    def pause(self) -> bool:
        if self._mac:
            if self._mac_transport("pause"):
                self._message = "spotify pause"
                return True
            return False
        if not self.ensure_ready(open_if_needed=False):
            self._message = "spotify pause failed (not ready)"
            return False
        status, body = self._request_json("PUT", "/me/player/pause", params=self._device_params())
        if status in {200, 202, 204}:
            self._message = "spotify pause"
            return True
        try:
            import sys as _sys
            _sys.stderr.write(f"[spotify] pause got HTTP {status} body={body!r}\n")
            _sys.stderr.flush()
        except Exception:
            pass
        self._message = f"spotify pause failed (status {status})"
        return False

    def next_track(self) -> bool:
        if self._mac:
            if self._mac_transport("next track"):
                self._message = "spotify next track"
                return True
            return False
        if not self.ensure_ready(open_if_needed=True):
            return False
        status, _ = self._request_json("POST", "/me/player/next", params=self._device_params())
        if status in {200, 202, 204}:
            self._message = "spotify next track"
            return True
        self._message = f"spotify next failed (status {status})"
        return False

    def previous_track(self) -> bool:
        if self._mac:
            if self._mac_transport("previous track"):
                self._message = "spotify previous track"
                return True
            return False
        if not self.ensure_ready(open_if_needed=True):
            return False
        status, _ = self._request_json("POST", "/me/player/previous", params=self._device_params())
        if status in {200, 202, 204}:
            self._message = "spotify previous track"
            return True
        self._message = f"spotify previous failed (status {status})"
        return False

    def toggle_repeat_track(self) -> bool:
        if self._mac:
            # Desktop app has only on/off repeat (no track-vs-context).
            script = (
                'if application "Spotify" is not running then\n'
                '\treturn "not-running"\n'
                'end if\n'
                'tell application "Spotify"\n'
                '\tset repeating to (not repeating)\n'
                '\treturn (repeating as text)\n'
                'end tell'
            )
            ok, out, err = self._mac_osascript(script)
            if not ok or out == "not-running":
                self._message = self._mac_error_message("spotify not running", err) if not ok else "spotify not running"
                return False
            self._message = f"spotify repeat {'on' if out == 'true' else 'off'}"
            return True
        player = self.get_player_state()
        current_mode = (player or {}).get("repeat_state")
        target_mode = "off" if current_mode == "track" else "track"
        # Merge state with device_id so the request both targets the
        # right device and avoids NO_ACTIVE_DEVICE 404s.
        params = {"state": target_mode}
        if self._device_id:
            params["device_id"] = self._device_id
        status, _ = self._request_json("PUT", "/me/player/repeat", params=params)
        # Same Spotify quirk as play/pause/next/prev: status can come
        # back as 200 (with non-JSON body) when the request lands on
        # the desktop client, even though the docs only mention 204.
        if status in {200, 202, 204}:
            self._message = f"spotify repeat {target_mode}"
            return True
        self._message = f"spotify repeat failed (status {status})"
        return False

    def toggle_shuffle(self) -> bool:
        if self._mac:
            script = (
                'if application "Spotify" is not running then\n'
                '\treturn "not-running"\n'
                'end if\n'
                'tell application "Spotify"\n'
                '\tset shuffling to (not shuffling)\n'
                '\treturn (shuffling as text)\n'
                'end tell'
            )
            ok, out, err = self._mac_osascript(script)
            if not ok or out == "not-running":
                self._message = self._mac_error_message("spotify not running", err) if not ok else "spotify not running"
                return False
            self._message = f"spotify shuffle {'on' if out == 'true' else 'off'}"
            return True
        player = self.get_player_state()
        if player is None and not self.ensure_ready(open_if_needed=True):
            return False
        current_state = bool((player or {}).get("shuffle_state"))
        target_state = not current_state
        params = {"state": "true" if target_state else "false"}
        if self._device_id:
            params["device_id"] = self._device_id
        status, _ = self._request_json("PUT", "/me/player/shuffle", params=params)
        if status in {200, 202, 204}:
            self._message = f"spotify shuffle {'on' if target_state else 'off'}"
            return True
        self._message = f"spotify shuffle failed (status {status})"
        return False

    def get_volume(self) -> int | None:
        player = self.get_player_state()
        if player is None:
            return None
        device = player.get("device") or {}
        vol = device.get("volume_percent")
        return int(vol) if vol is not None else None

    def set_volume(self, volume_percent: int) -> bool:
        volume_percent = max(0, min(100, int(volume_percent)))
        if self._mac:
            script = (
                'if application "Spotify" is running then\n'
                f'\ttell application "Spotify" to set sound volume to {volume_percent}\n'
                'end if'
            )
            ok, _out, err = self._mac_osascript(script)
            self._message = (
                f"spotify volume {volume_percent}%" if ok
                else self._mac_error_message("spotify volume set failed", err)
            )
            return ok
        if not self._ensure_authenticated():
            return False
        status, _ = self._request_json(
            "PUT",
            "/me/player/volume",
            params={"volume_percent": volume_percent},
        )
        if status in {200, 202, 204}:
            self._message = f"spotify volume {volume_percent}%"
            return True
        self._message = "spotify volume set failed"
        return False

    def is_window_active(self) -> bool:
        if self._mac:
            # Frontmost detection would need Accessibility and isn't required:
            # every mac control verb is focus-independent.
            return False
        handles = self._spotify_window_handles()
        if not handles:
            return False
        return self._foreground_window_handle() in handles

    def focus_or_open_window(self) -> bool:
        if self._mac:
            # `open -a` launches Spotify if needed AND brings it to the front,
            # WITHOUT sending an Apple Event — so it works even before the
            # Automation grant (unlike `activate`, which needs the TCC prompt).
            try:
                proc = subprocess.run(["open", "-a", "Spotify"], capture_output=True, text=True, timeout=8.0, check=False)
                ok = proc.returncode == 0
            except Exception:
                ok = False
            self._mac_running_cache = None  # re-check running state after focus/launch
            self._message = "spotify focused" if ok else "spotify focus failed"
            return ok
        if not self._available:
            self._message = "spotify unavailable on this platform"
            return False
        if self.is_window_active():
            self._message = "spotify already focused"
            return True

        handles = self._spotify_window_handles()
        if handles:
            if self._activate_window_handle(handles[0]):
                self._message = "spotify focused"
                return True
            self._message = "spotify focus failed"
            return False

        # No window yet → kick off the launch on a background thread
        # and return immediately. The 10-15s wait_for_window_handles
        # poll used to run on whichever thread invoked focus, which
        # froze the gesture worker (right-hand 'two') and voice
        # command pipeline (open spotify) and made the camera feed
        # stutter for the duration. The bg thread will refresh
        # self._message when it knows the outcome; we report
        # "launching spotify" as the immediate optimistic result so
        # the caller gets a non-failure status to surface in the UI.
        with self._launch_lock:
            already_in_flight = self._launch_in_flight
            if not already_in_flight:
                self._launch_in_flight = True
        if already_in_flight:
            self._message = "launching spotify"
            return True

        worker = threading.Thread(
            target=self._async_launch_and_focus,
            name="spotify-launch",
            daemon=True,
        )
        self._message = "launching spotify"
        worker.start()
        return True

    def dispatch_async(self, callable_obj, *args, **kwargs) -> None:
        # Fire-and-forget runner for synchronous Spotify HTTP calls
        # (next_track, previous_track, toggle_playback, etc.). The
        # gesture worker calls these from inside its main loop;
        # each call is 50-300 ms of urllib_request roundtrip.
        # Without this dispatcher the camera/MediaPipe loop visibly
        # stalls during a swipe — diagnostic showed 200+ ms spikes
        # on slow network round-trips. NOTE: deliberately does NOT
        # invalidate the active-device cache here. Track-skip /
        # play-pause / shuffle / repeat don't change *whether* a
        # device is available, only what's playing on it. Forcing
        # a re-query after every action made the next gesture
        # decision pay a fresh HTTP call, undoing the cache's
        # whole reason to exist. focus_or_open_window already
        # invalidates separately when it actually opens Spotify.
        def _runner():
            try:
                callable_obj(*args, **kwargs)
            except Exception:
                pass

        worker = threading.Thread(target=_runner, name="spotify-action", daemon=True)
        worker.start()

    def _async_launch_and_focus(self) -> None:
        try:
            # Strict check: SpotifyLauncher.exe and
            # Spotify-WebHelper.exe loiter in the background long
            # after the user has closed Spotify, and is_running()
            # matches any process whose name contains 'spotify'.
            # Using is_running() here meant we skipped launch_spotify
            # entirely whenever those helpers were alive — leaving
            # the user with no window and our overlay stuck on
            # "launching spotify". Use the strict "real client"
            # check that requires an actual large Spotify.exe.
            if not self._has_real_spotify_process():
                try:
                    self.ensure_ready(open_if_needed=True)
                except Exception:
                    pass
                self.launch_spotify(hidden=False)
            else:
                try:
                    self.ensure_ready(open_if_needed=False)
                except Exception:
                    pass
            handles = self._wait_for_window_handles(timeout_seconds=15.0)
            if handles:
                if self._activate_window_handle(handles[0]):
                    self._message = "spotify focused"
                else:
                    self._message = "spotify focus failed"
            else:
                self._message = "spotify window not found"
        finally:
            with self._launch_lock:
                self._launch_in_flight = False

    def is_active_device_available(self) -> bool:
        if self._mac:
            # A running Spotify app IS the controllable device on mac. (Bypass
            # the has_authorization/Web-API device probe entirely.)
            return self.is_running()
        now = time.monotonic()
        if self._active_device_cache is not None and now < self._active_device_cache_until:
            return self._active_device_cache
        # Short-circuit when the user has no Spotify auth: every
        # subsequent get_player_state() call would return None +
        # spam self._message ("spotify token not found"), and the
        # background refresh thread would keep spinning up uselessly.
        # The per-frame poll from SpotifyGestureRouter would then make
        # the status overlay flash "spotify not authorized" on every
        # frame even when the user is doing nothing. False here is
        # the same answer is_active_device_available would have
        # eventually returned anyway — just skipping the wasted
        # HTTP round-trips and the noise that comes with them.
        if not self.has_authorization:
            self._active_device_cache = False
            self._active_device_cache_until = now + self._active_device_cache_seconds
            return False
        # Cache miss. get_player_state() is a 50-300 ms HTTP call --
        # this method is called per gesture frame from
        # SpotifyGestureRouter._can_control_without_focus while a
        # hand is in frame, and the gesture loop runs on the UI
        # thread, so a sync call here freezes the camera display for
        # the duration of the request. Push the cache fence forward
        # so we don't spawn a second refresh while the first is
        # still in flight, kick the refresh on a daemon thread, and
        # return the previous cached value (defaulting to False on
        # first call so we don't pretend a device is available
        # before the very first probe has finished).
        self._active_device_cache_until = now + self._active_device_cache_seconds
        if not self._active_device_refresh_in_flight:
            self._active_device_refresh_in_flight = True

            def _refresh() -> None:
                try:
                    fresh = self.get_player_state() is not None
                    self._active_device_cache = fresh
                except Exception:
                    pass
                finally:
                    self._active_device_refresh_in_flight = False

            try:
                threading.Thread(
                    target=_refresh,
                    name="spotify-active-device-probe",
                    daemon=True,
                ).start()
            except Exception:
                self._active_device_refresh_in_flight = False
        return bool(self._active_device_cache) if self._active_device_cache is not None else False

    def invalidate_active_device_cache(self) -> None:
        # Called from focus_or_open_window paths after we deliberately
        # change Spotify's run state, so the next is_active_device_
        # available query re-checks instead of returning a stale
        # "False" from before we launched the app.
        self._active_device_cache = None
        self._active_device_cache_until = 0.0

    def get_current_track_details(self) -> SpotifyTrackDetails | None:
        player = self.get_player_state()
        if not player:
            self._message = "spotify inactive on device"
            return None

        item = player.get("item") or {}
        if not item:
            self._message = "spotify track unavailable"
            return None
        artists = ", ".join(artist.get("name", "") for artist in item.get("artists") or [] if artist.get("name"))
        album = item.get("album") or {}
        playlist_name = None
        context = player.get("context") or {}
        context_type = context.get("type")
        context_uri = context.get("uri")
        if context_type == "playlist" and isinstance(context_uri, str):
            playlist_name = self._get_playlist_name(context_uri)

        details = SpotifyTrackDetails(
            song_name=item.get("name") or "unknown song",
            artist_names=artists or "unknown artist",
            album_name=album.get("name"),
            playlist_name=playlist_name,
            device_name=(player.get("device") or {}).get("name"),
            device_type=(player.get("device") or {}).get("type"),
            is_playing=bool(player.get("is_playing")),
            shuffle_enabled=bool(player.get("shuffle_state")),
            repeat_mode=player.get("repeat_state"),
            progress_ms=player.get("progress_ms"),
            duration_ms=item.get("duration_ms"),
            context_type=context_type,
        )
        self._message = "spotify track info"
        return details

    def parse_voice_play_request(self, spoken_text: str) -> SpotifyVoiceRequest | None:
        raw_text = " ".join((spoken_text or "").strip().split())
        if not raw_text:
            return None

        lowered = raw_text.lower()
        normalized = f" {lowered} "
        replacements = (
            "on spotify",
            "in spotify",
            "from spotify",
            "through spotify",
            "using spotify",
            "spotify",
            "please",
            "for me",
        )
        for token in replacements:
            normalized = normalized.replace(f" {token} ", " ")

        normalized = re.sub(r"\b(can you|could you|would you|will you)\b", " ", normalized)
        normalized = re.sub(r"\b(play|put on|start|listen to|queue up|queue)\b", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip(" .!?")
        normalized = re.sub(r"^(and|then|uh|um)\s+", "", normalized).strip(" .!?")
        if not normalized:
            return None

        preferred_types: tuple[str, ...]
        if "playlist" in lowered:
            normalized = re.sub(r"\bplaylist\b", " ", normalized).strip()
            preferred_types = ("playlist", "track", "album", "artist")
        elif "album" in lowered:
            normalized = re.sub(r"\balbum\b", " ", normalized).strip()
            preferred_types = ("album", "track", "playlist", "artist")
        elif "artist" in lowered or "songs by " in lowered:
            normalized = re.sub(r"\bartist\b", " ", normalized).strip()
            normalized = re.sub(r"\bsongs by\b", " ", normalized).strip()
            preferred_types = ("artist", "track", "playlist", "album")
        else:
            preferred_types = ("track", "playlist", "album", "artist")

        normalized = re.sub(r"\s+", " ", normalized).strip(" .!?")
        if len(normalized) < 2:
            return None
        return SpotifyVoiceRequest(
            raw_text=raw_text,
            query=normalized,
            preferred_types=preferred_types,
        )

    def play_voice_request(self, spoken_text: str) -> bool:
        request = self.parse_voice_play_request(spoken_text)
        if request is None:
            self._message = "spotify voice request not understood"
            return False
        return self.play_search_request(request.query, preferred_types=request.preferred_types)

    def play_search_request(self, query: str, *, preferred_types: tuple[str, ...] | None = None) -> bool:
        if self._mac:
            self._message = "search-and-play isn't available for the macOS Spotify app"
            return False
        normalized = re.sub(r"\s+", " ", str(query or "")).strip(" .!?")
        if len(normalized) < 2:
            self._message = "spotify play query missing"
            return False
        if not self.ensure_ready(open_if_needed=True):
            return False

        search_types = preferred_types or ("track", "playlist", "album", "artist")
        selection = self._search_best_playable(normalized, search_types)
        if selection is None:
            self._message = f"spotify could not find '{normalized}'"
            return False

        payload = selection["payload"]
        status, _ = self._request_json("PUT", "/me/player/play", payload=payload)
        if status not in {202, 204}:
            self._message = "spotify play request failed"
            return False

        self._message = f"spotify play {selection['kind']}: {selection['name']}"
        return True

    def is_active_for_wheel(self) -> bool:
        if self._mac:
            # Per-frame wheel gate: use the cached run check; never fall through
            # to the ~150 ms get_player_state osascript below.
            if self.is_running():
                return True
            self._message = "spotify not running"
            return False
        # Wheel only engages when Spotify is genuinely available for
        # control: either it has a visible window the user can
        # interact with, or the Spotify Web API confirms an active
        # device (might be playing on the user's phone). Stricter
        # than the previous is_running() check, which considered
        # Spotify protocol-handler / helper processes as "running"
        # and surfaced wheel slices that did nothing useful.
        if self.is_window_open():
            return True
        player = self.get_player_state()
        if player is not None:
            return True
        self._message = "spotify not running"
        return False

    def add_current_track_to_queue(self) -> bool:
        if self._mac:
            self._message = "add-to-queue isn't available for the macOS Spotify app"
            return False
        uri = self._current_track_uri()
        if not uri:
            self._message = "spotify track unavailable"
            return False
        if not self.ensure_ready(open_if_needed=False):
            return False
        status, _ = self._request_json("POST", "/me/player/queue", params={"uri": uri})
        if status in {202, 204}:
            self._message = "spotify add to queue"
            return True
        self._message = "spotify add to queue failed"
        return False

    def remove_current_track_from_queue(self) -> bool:
        self._message = "spotify queue removal is not supported by the Spotify API"
        return False

    def save_current_track(self) -> bool:
        if self._mac:
            self._message = "liking songs isn't available for the macOS Spotify app"
            return False
        track_id = self._current_track_id()
        if not track_id:
            self._message = "spotify track unavailable"
            return False
        status, payload = self._request_json("PUT", "/me/tracks", params={"ids": track_id})
        if status in {200, 201, 202, 204}:
            self._message = "spotify saved current track"
            return True
        self._message = self._format_error_message("spotify save track failed", status, payload)
        return False

    def remove_current_track_from_liked(self) -> bool:
        if self._mac:
            self._message = "unliking songs isn't available for the macOS Spotify app"
            return False
        track_id = self._current_track_id()
        if not track_id:
            self._message = "spotify track unavailable"
            return False
        status, _ = self._request_json("DELETE", "/me/tracks", params={"ids": track_id})
        if status in {200, 201, 202, 204}:
            self._message = "spotify removed current track from liked songs"
            return True
        self._message = "spotify remove liked track failed"
        return False

    def add_current_track_to_playlist(self, playlist_name: str) -> bool:
        if self._mac:
            self._message = "playlist editing isn't available for the macOS Spotify app"
            return False
        target = self._resolve_playlist_target(playlist_name)
        track_uri = self._current_track_uri()
        if target is None or not track_uri:
            if track_uri is None:
                self._message = "spotify track unavailable"
            return False
        playlist_id = self._playlist_id_from_uri(target["uri"])
        if playlist_id is None:
            self._message = "spotify playlist unavailable"
            return False
        status, payload = self._request_json(
            "POST",
            f"/playlists/{playlist_id}/tracks",
            payload={"uris": [track_uri]},
        )
        if status in {200, 201}:
            self._message = f"spotify added to playlist: {target['name']}"
            return True
        self._message = self._format_error_message(
            f"spotify add to playlist failed: {target['name']}", status, payload
        )
        return False

    def remove_current_track_from_current_playlist(self) -> bool:
        if self._mac:
            self._message = "playlist editing isn't available for the macOS Spotify app"
            return False
        player = self.get_player_state()
        track_uri = self._current_track_uri()
        if not player or not track_uri:
            if track_uri is None:
                self._message = "spotify track unavailable"
            return False
        context = player.get("context") or {}
        if str(context.get("type") or "") != "playlist":
            self._message = "spotify current track is not playing from a playlist"
            return False
        playlist_uri = str(context.get("uri") or "").strip()
        playlist_id = self._playlist_id_from_uri(playlist_uri)
        if playlist_id is None:
            self._message = "spotify current playlist unavailable"
            return False
        playlist_name = self._get_playlist_name(playlist_uri) or "current playlist"
        status, _ = self._request_json(
            "DELETE",
            f"/playlists/{playlist_id}/tracks",
            payload={"tracks": [{"uri": track_uri}]},
        )
        if status in {200, 201}:
            self._message = f"spotify removed from {playlist_name}"
            return True
        self._message = f"spotify remove from playlist failed: {playlist_name}"
        return False

    def remove_current_track_from_playlist(self, playlist_name: str) -> bool:
        if self._mac:
            self._message = "playlist editing isn't available for the macOS Spotify app"
            return False
        target = self._resolve_playlist_target(playlist_name)
        track_uri = self._current_track_uri()
        if target is None or not track_uri:
            if track_uri is None:
                self._message = "spotify track unavailable"
            return False
        playlist_id = self._playlist_id_from_uri(target["uri"])
        if playlist_id is None:
            self._message = "spotify playlist unavailable"
            return False
        status, _ = self._request_json(
            "DELETE",
            f"/playlists/{playlist_id}/tracks",
            payload={"tracks": [{"uri": track_uri}]},
        )
        if status in {200, 201}:
            self._message = f"spotify removed from playlist: {target['name']}"
            return True
        self._message = f"spotify remove from playlist failed: {target['name']}"
        return False

    def create_playlist(self, name: str, *, public: bool = False) -> bool:
        if self._mac:
            self._message = "creating playlists isn't available for the macOS Spotify app"
            return False
        clean = (name or "").strip()
        if not clean:
            self._message = "spotify playlist name missing"
            return False
        user_id = self._get_current_user_id()
        if not user_id:
            return False
        status, payload = self._request_json(
            "POST",
            f"/users/{user_id}/playlists",
            payload={"name": clean, "public": bool(public)},
        )
        if status in {200, 201}:
            self._message = f"spotify created playlist: {clean}"
            return True
        self._message = self._format_error_message(
            f"spotify create playlist failed: {clean}", status, payload
        )
        return False

    def _get_current_user_id(self) -> str | None:
        status, payload = self._request_json("GET", "/me")
        if status != 200 or not isinstance(payload, dict):
            self._message = self._format_error_message("spotify profile unavailable", status, payload)
            return None
        user_id = payload.get("id")
        if not isinstance(user_id, str) or not user_id:
            self._message = "spotify profile unavailable"
            return None
        return user_id

    def _format_error_message(self, prefix: str, status: int | None, payload: Any) -> str:
        detail = ""
        if isinstance(payload, dict):
            inner = payload.get("error")
            if isinstance(inner, dict):
                msg = inner.get("message")
                if isinstance(msg, str) and msg:
                    detail = msg
            elif isinstance(inner, str):
                detail = inner
        elif isinstance(payload, str):
            detail = payload.strip()
        if status == 403 and ("scope" in detail.lower() or not detail):
            return f"{prefix} (403 missing scope — re-authorize Spotify in Settings)"
        if status is None:
            return f"{prefix} (network error)"
        if detail:
            return f"{prefix} ({status}: {detail})"
        return f"{prefix} ({status})"

    def authorize_full_scopes(self, *, port: int = 5000, timeout_seconds: float = 180.0) -> bool:
        """Open Spotify's OAuth flow in the user's browser using
        PKCE (Proof Key for Code Exchange). PKCE eliminates the
        need for a client_secret — each authorization derives proof
        from a per-session code_verifier instead. This is the
        standard for desktop apps; the client_id alone is enough.

        On success, writes the resulting access + refresh tokens to
        the user's token file (`auth_token.json`) so subsequent
        launches skip the dance until the refresh token expires.

        Port-fallback: tries the requested port first, then
        5001..5004 if that one is already in use (Discord, OBS, dev
        servers, IIS, etc. commonly grab 5000). Each fallback port
        needs to be a registered redirect URI in the Spotify
        Developer Dashboard.

        Errors are logged to stderr with [spotify-auth] prefix so a
        user / tester running from a terminal can diagnose. Surfaces
        Spotify-specific error codes verbatim (user_not_listed,
        invalid_redirect_uri, etc.)."""
        import sys as _sys
        def _log(msg: str) -> None:
            try:
                _sys.stderr.write(f"[spotify-auth] {msg}\n")
                _sys.stderr.flush()
            except Exception:
                pass

        if not self._client_id:
            self._message = "spotify client id not configured"
            _log("FAIL: client_id missing (env var or embedded default both unset)")
            return False
        import http.server
        import socketserver
        import threading
        import webbrowser

        _log(f"starting PKCE flow with client_id={self._client_id[:8]}…")
        # Determine redirect URI + port list to try. If the caller
        # passed an explicit redirect_uri via .env, honour it exactly
        # (no port fallback — they're telling us they registered that
        # specific URI). Otherwise try a small range of localhost
        # ports so a busy 5000 (Discord RPC, OBS dock, dev server)
        # doesn't completely break OAuth for the user.
        fallback_ports = [port, 5001, 5002, 5003, 5004]
        explicit_redirect = bool(self._redirect_uri)
        redirect_uri = self._redirect_uri or f"http://127.0.0.1:{port}/callback"
        state = secrets.token_urlsafe(16)
        # PKCE: generate a high-entropy code_verifier (43-128 chars,
        # URL-safe) and derive the challenge as base64url(SHA256(verifier)).
        # The verifier is held in memory until the token exchange step,
        # at which point Spotify checks SHA256(verifier) == challenge
        # to prove the same client started + finished the flow.
        code_verifier = secrets.token_urlsafe(64)[:128]
        challenge_bytes = hashlib.sha256(code_verifier.encode("ascii")).digest()
        code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode("ascii").rstrip("=")
        auth_params = {
            "client_id": self._client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(SPOTIFY_SCOPES),
            "state": state,
            "show_dialog": "true",
            "code_challenge_method": "S256",
            "code_challenge": code_challenge,
        }
        auth_url = f"{SPOTIFY_AUTH_URL}?{urllib_parse.urlencode(auth_params)}"

        result: dict[str, str | None] = {"code": None, "error": None}
        done = threading.Event()

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self_inner):
                parsed = urllib_parse.urlparse(self_inner.path)
                params = dict(urllib_parse.parse_qsl(parsed.query))
                if params.get("state") != state:
                    result["error"] = "state mismatch"
                else:
                    result["code"] = params.get("code")
                    result["error"] = params.get("error")
                self_inner.send_response(200)
                self_inner.send_header("Content-Type", "text/html; charset=utf-8")
                self_inner.end_headers()
                body = (
                    "<html><body style='font-family:sans-serif;padding:32px;'>"
                    "<h2>Spotify authorization complete.</h2>"
                    "<p>You can close this tab and return to Touchless.</p>"
                    "</body></html>"
                )
                self_inner.wfile.write(body.encode("utf-8"))
                done.set()

        # Bind the local callback server. With explicit_redirect we
        # only try the one configured port; otherwise walk the
        # fallback list. Each attempt logs its outcome so a tester
        # can see WHICH port worked / which were busy.
        host = urllib_parse.urlparse(redirect_uri).hostname or "127.0.0.1"
        httpd = None
        bound_port = None
        ports_to_try = [port] if explicit_redirect else fallback_ports
        for candidate_port in ports_to_try:
            try:
                httpd = socketserver.TCPServer((host, candidate_port), _Handler)
                bound_port = candidate_port
                _log(f"callback server bound to {host}:{candidate_port}")
                break
            except OSError as exc:
                _log(f"port {candidate_port} busy ({exc}); trying next")
                continue
            except Exception as exc:
                _log(f"port {candidate_port} failed ({exc}); trying next")
                continue
        if httpd is None or bound_port is None:
            tried = ", ".join(str(p) for p in ports_to_try)
            self._message = (
                f"spotify auth: no callback port available (tried {tried}). "
                "Close apps using these ports (Discord, OBS, dev servers) and "
                "try again."
            )
            _log(f"FAIL: every candidate port busy ({tried})")
            return False
        # Rebuild the auth URL with whichever port actually bound.
        # The Spotify Dev Dashboard must list ALL fallback URIs as
        # registered redirect URIs (5000..5004) or Spotify rejects
        # the authorize call with invalid_redirect_uri.
        if not explicit_redirect and bound_port != port:
            redirect_uri = f"http://127.0.0.1:{bound_port}/callback"
            auth_params["redirect_uri"] = redirect_uri
            auth_url = f"{SPOTIFY_AUTH_URL}?{urllib_parse.urlencode(auth_params)}"
            _log(f"using fallback redirect_uri={redirect_uri}")

        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        try:
            _log("opening Spotify authorize URL in default browser")
            opened = webbrowser.open(auth_url)
            if not opened:
                _log("WARNING: webbrowser.open returned False — browser may not have launched")
            done.wait(timeout=timeout_seconds)
        finally:
            httpd.shutdown()
            httpd.server_close()

        if result.get("error") or not result.get("code"):
            err = result.get("error") or "no code (timed out or browser closed)"
            # Map common Spotify error codes to actionable guidance.
            guidance = ""
            if err == "access_denied":
                guidance = " — user clicked Cancel / Don't Allow in the browser."
            elif err == "invalid_redirect_uri":
                guidance = (
                    " — the redirect URI isn't registered in the Spotify "
                    "Developer Dashboard. Open your app at developer.spotify.com, "
                    "go to Edit Settings → Redirect URIs, and add "
                    f"{redirect_uri}."
                )
            elif err == "invalid_client":
                guidance = " — the embedded client_id is wrong or has been deleted in the Spotify Dashboard."
            elif err == "user_not_listed" or err == "user_not_registered":
                guidance = (
                    " — your Spotify app is still in Development Mode and "
                    "this user isn't on the allow-list. Either add them at "
                    "developer.spotify.com → app → Users and Access, or "
                    "submit your app for Extended Quota Mode review to "
                    "lift the 25-user cap."
                )
            elif err == "state mismatch":
                guidance = " — possible CSRF / browser-cache issue; clear cookies for accounts.spotify.com and try again."
            self._message = f"spotify auth failed: {err}{guidance}"
            _log(f"FAIL: {self._message}")
            return False

        # PKCE token exchange: send client_id + code_verifier in the
        # POST body. Spotify recomputes SHA256(code_verifier) and
        # checks it against the challenge it stored from the
        # authorize step. No Basic auth header (no client_secret).
        data = urllib_parse.urlencode(
            {
                "grant_type": "authorization_code",
                "code": result["code"],
                "redirect_uri": redirect_uri,
                "client_id": self._client_id,
                "code_verifier": code_verifier,
            }
        ).encode("utf-8")
        request = urllib_request.Request(
            SPOTIFY_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            _log("exchanging authorization code for access token (PKCE)")
            with urllib_request.urlopen(request, timeout=self._request_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            _log("token exchange OK — received access + refresh tokens")
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            # Parse Spotify's JSON error so we can surface the
            # specific code (invalid_grant, unauthorized_client, etc.)
            # in a user-friendly way.
            err_code = ""
            err_desc = ""
            try:
                parsed = json.loads(body)
                err_code = str(parsed.get("error") or "")
                err_desc = str(parsed.get("error_description") or "")
            except Exception:
                pass
            guidance = ""
            if err_code == "invalid_grant":
                guidance = " — authorization code expired or already used. Try connecting again."
            elif err_code == "invalid_client":
                guidance = " — client_id rejected by Spotify (deleted from dashboard? Wrong account?)."
            elif err_code == "unauthorized_client":
                guidance = (
                    " — your Spotify Dev app isn't authorised for this grant "
                    "type. Check the app's settings at developer.spotify.com."
                )
            elif exc.code == 429:
                guidance = " — rate-limited by Spotify. Wait a minute and try again."
            human = err_desc or body[:200] or str(exc)
            self._message = f"spotify token exchange failed ({exc.code}): {human}{guidance}"
            _log(f"FAIL: HTTPError {exc.code}: code={err_code!r} desc={err_desc!r}")
            return False
        except Exception as exc:
            self._message = f"spotify token exchange failed: {exc}"
            _log(f"FAIL: exception during token exchange: {exc}")
            return False

        self._access_token = payload.get("access_token")
        refresh = payload.get("refresh_token")
        if refresh:
            self._refresh_token = refresh
        self._token_issue_time = time.time()
        if self._token_path is None:
            # Save to the persistent per-user path
            # (~/Documents/Touchless/auth_token.json) rather than the
            # install-directory default. The install dir is wiped on
            # every auto-update, which would force every shipped user
            # to re-authorise Spotify after every release. The
            # Documents folder survives updates AND is per Windows
            # user (different users on the same machine get separate
            # tokens). _resolve_persistent_token_path picks that path,
            # creating the directory if needed.
            self._token_path = self._resolve_persistent_token_path()
        self._save_tokens()
        _log(f"tokens saved to {self._token_path}")
        self._message = "spotify authorized with full scopes"
        _log("SUCCESS: spotify authorized with full scopes")
        return bool(self._access_token)

    def _resolve_persistent_token_path(self) -> Path:
        """Return the per-user, update-survival path Spotify tokens
        should be written to. Always ~/Documents/Touchless/
        auth_token.json on Windows; the parent directory is
        created on demand so the very-first OAuth on a fresh
        install doesn't fail just because Documents/Touchless/
        doesn't exist yet."""
        home = Path.home()
        target = home / "Documents" / "Touchless" / "auth_token.json"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If we can't create the per-user folder for any reason
            # (locked-down corporate environment, etc.) fall back to
            # the legacy install-dir path. The user might lose tokens
            # on update but at least the OAuth flow completes.
            return self._token_paths[0]
        return target

    def _default_token_paths(self) -> tuple[Path, ...]:
        home = Path.home()
        return (
            self._repo_root / "auth_token.json",
            home / "Documents" / "Touchless" / "auth_token.json",
            home / "Documents" / "HandGestureControl" / "HGRApp" / "auth_token.json",
            home / "Documents" / "HandAI" / "HandMeshLive" / "src" / "auth_token.json",
        )

    def _default_env_paths(self) -> tuple[Path, ...]:
        home = Path.home()
        return (
            self._repo_root / ".env",
            home / "Documents" / "Touchless" / ".env",
            home / "Documents" / "HandGestureControl" / "HGRApp" / ".env",
            home / "Documents" / "HandAI" / "HandMeshLive" / "src" / ".env",
        )

    def _default_executable_paths(self) -> tuple[Path, ...]:
        home = Path.home()
        candidates: list[Path] = [
            # Per-user classic install (Spotify's default installer
            # location for the desktop .exe build off spotify.com).
            home / "AppData" / "Roaming" / "Spotify" / "Spotify.exe",
            home / "AppData" / "Local" / "Spotify" / "Spotify.exe",
            # Machine-wide installs (rare for Spotify but supported).
            Path("C:/Program Files/Spotify/Spotify.exe"),
            Path("C:/Program Files (x86)/Spotify/Spotify.exe"),
            Path("C:/ProgramData/Spotify/Spotify.exe"),
            # Microsoft Store stub last — these are 0-byte App
            # Execution Aliases that need ShellExecute, and we now
            # filter them by stat().st_size < 1024 in launch_spotify
            # so they don't masquerade as a launchable .exe.
            home / "AppData" / "Local" / "Microsoft" / "WindowsApps" / "SpotifyAB.SpotifyMusic_zpdnekdrzrea0" / "Spotify.exe",
        ]
        # Pull the App Paths registry key so we pick up unusual
        # install locations the user might have (portable installs,
        # custom paths, etc.) without hardcoding more guesses.
        try:
            import winreg

            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(
                        hive,
                        r"Software\Microsoft\Windows\CurrentVersion\App Paths\Spotify.exe",
                    ) as key:
                        raw, _ = winreg.QueryValueEx(key, None)
                        if raw:
                            candidates.insert(0, Path(str(raw).strip('"')))
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
        except Exception:
            pass
        # De-dup while preserving order.
        seen: set[str] = set()
        ordered: list[Path] = []
        for path in candidates:
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(path)
        return tuple(ordered)

    def _load_credentials(self) -> None:
        """Resolve client_id + redirect_uri for the Spotify OAuth
        flow. Resolution order:
          1. Per-user setting (config.spotify_client_id) set via the
             in-app Spotify setup wizard. This is the ONLY path that
             scales past Spotify's 5-user cap on shared dev apps —
             each user runs against their own Spotify Dev app, so
             the cap is per-Touchless-user instead of per-installation.
          2. Env vars (TOUCHLESS_SPOTIFY_CLIENT_ID / CLIENT_ID).
          3. First .env file in `_env_paths` that defines CLIENT_ID.
          4. Embedded defaults (`_DEFAULT_SPOTIFY_CLIENT_ID`) — this
             is what shipped builds use for the first 5 testers per
             release. After 5 unique Spotify users have authorised,
             Spotify rejects further connections with `user_not_listed`
             unless the user supplies their own client_id (path 1).

        client_secret is loaded for backwards compatibility (legacy
        non-PKCE refresh tokens from older Touchless installs can
        still be refreshed using the secret path) but is NOT
        required for new auth flows."""
        # Path 1: per-user client_id from in-app setup wizard.
        try:
            from ..config.app_config import load_config as _load_config
            user_cfg = _load_config()
            user_client_id = str(getattr(user_cfg, "spotify_client_id", "") or "").strip()
        except Exception:
            user_client_id = ""
        if user_client_id:
            self._client_id = user_client_id
            self._client_secret = None
            self._redirect_uri = _DEFAULT_SPOTIFY_REDIRECT_URI
            return
        env_client_id = (
            os.getenv("TOUCHLESS_SPOTIFY_CLIENT_ID")
            or os.getenv("CLIENT_ID")
            or os.getenv("SPOTIFY_CLIENT_ID")
        )
        env_client_secret = os.getenv("CLIENT_SECRET") or os.getenv("SPOTIFY_CLIENT_SECRET")
        env_redirect_uri = os.getenv("REDIRECT_URI") or os.getenv("SPOTIFY_REDIRECT_URI")
        if env_client_id:
            self._client_id = env_client_id
            self._client_secret = env_client_secret  # may be None — fine for PKCE
            self._redirect_uri = env_redirect_uri or _DEFAULT_SPOTIFY_REDIRECT_URI
            return

        for path in self._env_paths:
            values = self._parse_env_file(path)
            if values.get("CLIENT_ID"):
                self._env_path = path
                self._client_id = values["CLIENT_ID"]
                self._client_secret = values.get("CLIENT_SECRET")  # optional
                self._redirect_uri = values.get("REDIRECT_URI", _DEFAULT_SPOTIFY_REDIRECT_URI)
                return

        # Fall through to embedded defaults so shipped builds (with
        # no .env on disk) can still authenticate every user via
        # the PKCE flow. No secret is shipped — Spotify's PKCE
        # mode doesn't need one.
        self._client_id = _DEFAULT_SPOTIFY_CLIENT_ID
        self._client_secret = None
        self._redirect_uri = _DEFAULT_SPOTIFY_REDIRECT_URI

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
            self._token_issue_time = float(issue_time) if isinstance(issue_time, (int, float)) else None
            # Fresh tokens loaded — clear the 'needs reauth' latch in
            # case it was set by a prior dead refresh token. Otherwise
            # the toast keeps nagging the user to reconnect even though
            # their tokens are now valid.
            self._needs_reauth = False
            return

    def reload_tokens(self) -> bool:
        """Force-reload tokens from disk into this controller, blowing
        away whatever was in memory. Called after the OAuth flow when
        the controller doing the authorization is a DIFFERENT instance
        than the engine's running controller (race: user clicks
        Connect while the engine's controller is mid-init, or auth
        happens via a one-shot SpotifyController spawned before the
        engine started). Returns True if a token file was found and
        loaded, False otherwise.

        Also resets the cached device id so the next ensure_ready
        re-resolves the active device against the freshly authorised
        account (covers the case where the controller had a device
        cached against an older / wrong user)."""
        # Snapshot prior state so we can tell if the reload actually
        # changed anything (for diagnostic logging).
        before = (self._access_token, self._refresh_token)
        self._access_token = None
        self._refresh_token = None
        self._token_issue_time = None
        self._token_path = None
        self._device_id = None
        self._device_name = None
        self._needs_reauth = False
        self._load_tokens()
        loaded = bool(self._access_token) or bool(self._refresh_token)
        if loaded and before != (self._access_token, self._refresh_token):
            self._message = "spotify tokens reloaded"
        return loaded

    def _save_tokens(self) -> None:
        if self._token_path is None:
            return
        payload = {
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "issue_time": self._token_issue_time or time.time(),
        }
        try:
            self._token_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    def _ensure_authenticated(self) -> bool:
        # PKCE auth doesn't require a client_secret — client_id alone
        # is enough. Legacy installs that have both can still use the
        # secret path during refresh (_refresh_access_token handles it).
        if not self._client_id:
            self._message = "spotify client id not configured"
            return False
        if self._access_token and not self._token_expired():
            return True
        if self._refresh_token:
            return self._refresh_access_token()
        if self._access_token:
            return True
        self._message = "spotify token not found"
        return False

    def warm_up(self) -> bool:
        if not self._available:
            return False
        if not self._client_id or not self._client_secret:
            return False
        if not self._refresh_token:
            return bool(self._access_token)
        if self._access_token and not self._token_expired():
            return True
        return self._refresh_access_token()

    def _token_expired(self) -> bool:
        if not self._token_issue_time:
            return True
        return (time.time() - self._token_issue_time) >= TOKEN_TTL_SECONDS

    def _refresh_access_token(self) -> bool:
        """Refresh the access token using the stored refresh token.

        Tries PKCE-style refresh first (client_id-only POST body) —
        the standard for tokens minted by the PKCE auth flow. Falls
        back to the classic Basic-auth refresh path only when a
        client_secret is present (covers legacy tokens minted by
        older Touchless builds before the PKCE switch).
        """
        if not self._refresh_token or not self._client_id:
            self._message = "spotify refresh unavailable"
            return False
        data = urllib_parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": self._client_id,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        # Backwards-compat: if a client_secret IS configured (dev
        # .env with both keys), use the classic Basic-auth header.
        # Legacy refresh tokens minted by the pre-PKCE flow need it.
        if self._client_secret:
            token_pair = f"{self._client_id}:{self._client_secret}".encode("utf-8")
            encoded = base64.b64encode(token_pair).decode("utf-8")
            headers["Authorization"] = f"Basic {encoded}"
        request = urllib_request.Request(
            SPOTIFY_TOKEN_URL,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self._request_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            self._message = f"spotify auth refresh failed ({exc.code})"
            # 400 / 401 from /api/token = refresh token is dead
            # (revoked, expired, or the Spotify app's client_id
            # rotated). Surface this so the UI can prompt for
            # re-auth instead of failing silently every gesture.
            if exc.code in (400, 401):
                self._needs_reauth = True
            return False
        except Exception:
            self._message = "spotify auth refresh failed"
            return False

        self._access_token = payload.get("access_token")
        self._refresh_token = payload.get("refresh_token", self._refresh_token)
        self._token_issue_time = time.time()
        self._save_tokens()
        self._message = "spotify token refreshed"
        # A successful refresh clears the reauth-needed flag if it
        # was set; the user is back to good standing.
        self._needs_reauth = False
        return bool(self._access_token)

    @property
    def needs_reauth(self) -> bool:
        """True when a stored refresh token has been rejected. The
        UI uses this to surface a one-shot 'reconnect Spotify' toast.
        Clears automatically on the next successful auth."""
        return bool(self._needs_reauth)

    def clear_reauth_flag(self) -> None:
        """Called by the UI after surfacing the reauth toast so the
        same flag-flip doesn't re-fire on every subsequent gesture
        frame. Re-arms only on the next refresh failure."""
        self._needs_reauth = False

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        allow_refresh: bool = True,
    ) -> tuple[int | None, Any]:
        if not self._ensure_authenticated():
            return None, None
        url = f"{SPOTIFY_API_BASE}{path}"
        if params:
            query = urllib_parse.urlencode(params)
            url = f"{url}?{query}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._access_token}"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = urllib_request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib_request.urlopen(request, timeout=self._request_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                if not raw:
                    return response.status, None
                # Spotify's PUT /me/player/play and /pause endpoints
                # often respond 200 with a non-JSON body (sometimes
                # an empty-string-with-trailing-whitespace, sometimes
                # a short status line like "OK"). Treat that as a
                # successful response with no parsed payload — the
                # status code is what callers actually look at, and
                # parsing failure here was bubbling up as
                # JSONDecodeError → caught-as-Exception → returning
                # (None, None) → toast incorrectly said "failed".
                try:
                    return response.status, json.loads(raw)
                except json.JSONDecodeError:
                    return response.status, None
        except urllib_error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="ignore")
            payload_value = None
            if raw:
                try:
                    payload_value = json.loads(raw)
                except json.JSONDecodeError:
                    payload_value = raw
            if exc.code == 401 and allow_refresh and self._refresh_access_token():
                return self._request_json(method, path, params=params, payload=payload, allow_refresh=False)
            return exc.code, payload_value
        except Exception as exc:
            # Diagnostic: surface what actually went wrong instead of
            # collapsing every error to (None, None). Common cases:
            # urllib_error.URLError (DNS / connect refused), socket.timeout,
            # ssl.SSLError. The action may have succeeded server-side
            # (Spotify is fast enough that our timeout fires after the
            # server already executed the action), so we want to
            # distinguish "client gave up reading the response" from
            # "Spotify rejected it".
            #
            # Rate-limited: when the user isn't signed in to Spotify
            # OR has revoked the token, every per-frame is_active_
            # device probe lands here and prints the same exception,
            # which spams the console with hundreds of identical
            # "[spotify] request GET /me/player raised ..." lines per
            # minute. Cap to one print per (method, path, exception
            # type) every 30 seconds so the diagnostic survives but
            # the noise dies.
            try:
                import sys as _sys
                key = f"{method}:{path}:{type(exc).__name__}"
                now_mono = time.monotonic()
                last = getattr(self, "_request_log_last_at", None)
                if last is None:
                    last = {}
                    self._request_log_last_at = last
                if now_mono - last.get(key, 0.0) >= 30.0:
                    last[key] = now_mono
                    _sys.stderr.write(
                        f"[spotify] request {method} {path} raised "
                        f"{type(exc).__name__}: {exc!s}\n"
                    )
                    _sys.stderr.flush()
            except Exception:
                pass
            self._message = f"spotify request failed: {type(exc).__name__}"
            return None, None

    def _get_devices(self) -> list[dict[str, Any]]:
        status, payload = self._request_json("GET", "/me/player/devices")
        if status != 200 or not isinstance(payload, dict):
            return []
        devices = payload.get("devices") or []
        return [device for device in devices if not device.get("is_restricted")]

    def _wait_for_devices(self) -> list[dict[str, Any]]:
        deadline = time.monotonic() + 7.0
        devices: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            devices = self._get_devices()
            if devices:
                return devices
            time.sleep(0.6)
        return devices

    def _pick_device(self, devices: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not devices:
            return None
        active = next((device for device in devices if device.get("is_active")), None)
        if active is not None:
            return active
        for preferred_type in ("Computer", "Smartphone", "Speaker"):
            for device in devices:
                if device.get("type") == preferred_type:
                    return device
        return devices[0]

    def _transfer_playback(self, device_id: str | None, *, play: bool) -> bool:
        if not device_id:
            return False
        status, _ = self._request_json(
            "PUT",
            "/me/player",
            payload={
                "device_ids": [device_id],
                "play": bool(play),
            },
        )
        return status == 204

    def _search_best_playable(self, query: str, preferred_types: tuple[str, ...]) -> dict[str, Any] | None:
        if preferred_types and preferred_types[0] == "playlist":
            library_playlist = self._find_library_playlist(query)
            if library_playlist is not None:
                return library_playlist
        search_types = tuple(dict.fromkeys(preferred_types))
        status, payload = self._request_json(
            "GET",
            "/search",
            params={
                "q": query,
                "type": ",".join(search_types),
                "limit": 5,
            },
        )
        if status != 200 or not isinstance(payload, dict):
            return None

        for item_type in search_types:
            items = self._search_items_for_type(payload, item_type)
            if not items:
                continue
            top = items[0]
            uri = top.get("uri")
            if not isinstance(uri, str) or not uri:
                continue
            name = top.get("name") or query
            if item_type == "track":
                return {
                    "kind": "track",
                    "name": name,
                    "payload": {"uris": [uri]},
                }
            return {
                "kind": item_type,
                "name": name,
                "payload": {"context_uri": uri},
            }
        return None

    def _resolve_playlist_target(self, playlist_name: str) -> dict[str, str] | None:
        normalized = self._normalize_search_text(playlist_name)
        if not normalized:
            self._message = "spotify playlist name missing"
            return None
        library_match = self._find_library_playlist(normalized)
        if library_match is not None:
            return {
                "name": str(library_match["name"]),
                "uri": str(library_match["payload"]["context_uri"]),
            }
        status, payload = self._request_json(
            "GET",
            "/search",
            params={
                "q": playlist_name,
                "type": "playlist",
                "limit": 5,
            },
        )
        if status != 200 or not isinstance(payload, dict):
            self._message = f"spotify playlist not found: {playlist_name}"
            return None
        items = self._search_items_for_type(payload, "playlist")
        if not items:
            self._message = f"spotify playlist not found: {playlist_name}"
            return None
        best = items[0]
        uri = best.get("uri")
        if not isinstance(uri, str):
            self._message = f"spotify playlist not found: {playlist_name}"
            return None
        return {
            "name": str(best.get("name") or playlist_name),
            "uri": uri,
        }

    def _current_track_uri(self) -> str | None:
        player = self.get_player_state()
        if not player:
            return None
        item = player.get("item") or {}
        uri = item.get("uri")
        return uri if isinstance(uri, str) and uri else None

    def _current_track_id(self) -> str | None:
        uri = self._current_track_uri()
        if not uri:
            return None
        if uri.startswith("spotify:track:"):
            return uri.rsplit(":", 1)[-1]
        return None

    def _find_library_playlist(self, query: str) -> dict[str, Any] | None:
        normalized_query = self._normalize_search_text(query)
        if not normalized_query:
            return None

        offset = 0
        best_match: dict[str, Any] | None = None
        best_score = -1
        while offset < 200:
            status, payload = self._request_json(
                "GET",
                "/me/playlists",
                params={"limit": 50, "offset": offset},
            )
            if status != 200 or not isinstance(payload, dict):
                break
            items = payload.get("items") or []
            if not isinstance(items, list) or not items:
                break
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                uri = item.get("uri")
                if not name or not isinstance(uri, str):
                    continue
                score = self._playlist_match_score(normalized_query, self._normalize_search_text(name))
                if score > best_score:
                    best_score = score
                    best_match = {
                        "kind": "playlist",
                        "name": name,
                        "payload": {"context_uri": uri},
                    }
            total = int(payload.get("total") or 0)
            offset += len(items)
            if not payload.get("next") or offset >= total:
                break
        return best_match if best_score >= 2 else None

    def _search_items_for_type(self, payload: dict[str, Any], item_type: str) -> list[dict[str, Any]]:
        key = f"{item_type}s"
        bucket = payload.get(key)
        if not isinstance(bucket, dict):
            return []
        items = bucket.get("items")
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def _normalize_search_text(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
        return re.sub(r"\s+", " ", normalized)

    def _playlist_match_score(self, query: str, candidate: str) -> int:
        if not candidate:
            return -1
        if candidate == query:
            return 5
        if query in candidate:
            return 4
        query_words = query.split()
        candidate_words = candidate.split()
        overlap = len(set(query_words) & set(candidate_words))
        if overlap >= max(1, len(query_words) - 1):
            return 3
        if overlap >= max(1, len(query_words) // 2):
            return 2
        return overlap

    def _get_playlist_name(self, context_uri: str) -> str | None:
        playlist_id = self._playlist_id_from_uri(context_uri)
        if playlist_id is None:
            return None
        status, payload = self._request_json("GET", f"/playlists/{playlist_id}")
        if status != 200 or not isinstance(payload, dict):
            return None
        return payload.get("name")

    def _playlist_id_from_uri(self, context_uri: str) -> str | None:
        if context_uri.startswith("spotify:playlist:"):
            return context_uri.rsplit(":", 1)[-1]
        if "playlist/" in context_uri:
            return context_uri.rsplit("playlist/", 1)[-1].split("?", 1)[0]
        return None

    def _foreground_window_handle(self) -> int | None:
        if not self._available or self._mac:
            return None
        try:
            foreground = ctypes.windll.user32.GetForegroundWindow()
        except Exception:
            return None
        return int(foreground) if foreground else None

    def _spotify_window_handles(self) -> list[int]:
        if not self._available or self._mac:
            return []
        now = time.monotonic()
        if now < self._handles_cache_until:
            return list(self._handles_cache)
        try:
            spotify_pids = {
                int(proc.info["pid"])
                for proc in psutil.process_iter(["pid", "name"])
                if "spotify" in (proc.info.get("name") or "").lower()
            }
        except Exception:
            spotify_pids = set()

        user32 = ctypes.windll.user32
        handles: list[int] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def _enum_windows(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            title_length = user32.GetWindowTextLengthW(hwnd)
            if title_length <= 0:
                return True
            # PID-based match (preferred): catches the classic
            # AppData\Roaming\Spotify\Spotify.exe install where the
            # visible top-level window is owned by Spotify.exe itself.
            if spotify_pids:
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if int(pid.value) in spotify_pids:
                    handles.append(int(hwnd))
                    return True
            # Title + class fallback for the Microsoft Store install:
            # the visible top-level window is an ApplicationFrameWindow
            # owned by ApplicationFrameHost.exe — its PID is *not*
            # Spotify.exe, so the PID filter alone misses it. The
            # actual Spotify.exe lives one HWND down as a
            # Windows.UI.Core.CoreWindow child. Match the frame
            # window by class + title prefix so we surface a window
            # that focus_or_open_window can BringWindowToTop on, but
            # stay strict about the title to avoid grabbing a Discord
            # / browser tab that merely mentions Spotify.
            buf = ctypes.create_unicode_buffer(title_length + 1)
            user32.GetWindowTextW(hwnd, buf, title_length + 1)
            title = (buf.value or "").strip()
            if not title:
                return True
            class_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, class_buf, 256)
            class_name = class_buf.value or ""
            is_frame_class = (
                class_name == "ApplicationFrameWindow"
                or class_name.startswith("Chrome_WidgetWin")
            )
            if not is_frame_class:
                return True
            lowered_title = title.lower()
            looks_like_spotify = (
                lowered_title == "spotify"
                or lowered_title.startswith("spotify ")
                or lowered_title.startswith("spotify - ")
                or lowered_title.startswith("spotify premium")
                or lowered_title.startswith("spotify free")
                or lowered_title.endswith(" | spotify")
                or lowered_title.endswith(" - spotify")
            )
            if looks_like_spotify:
                handles.append(int(hwnd))
            return True

        try:
            user32.EnumWindows(_enum_windows, 0)
        except Exception:
            handles = []
        self._handles_cache = list(handles)
        self._handles_cache_until = now + 1.0
        return handles

    def _wait_for_window_handles(self, timeout_seconds: float = 15.0) -> list[int]:
        # Microsoft Store cold launches resolve the App Execution
        # Alias stub then spin up an AppContainer; on first launch
        # of a session that can take 5-12s. The 1s cache inside
        # _spotify_window_handles must be busted between polls or
        # we'd spend 4-5 of every 5s window getting the same cached
        # empty list back instead of actually re-enumerating.
        deadline = time.monotonic() + timeout_seconds
        self._handles_cache_until = 0.0
        handles = self._spotify_window_handles()
        while not handles and time.monotonic() < deadline:
            time.sleep(0.5)
            self._handles_cache_until = 0.0
            handles = self._spotify_window_handles()
        return handles

    def _activate_window_handle(self, hwnd: int) -> bool:
        if not self._available or self._mac:
            return False
        user32 = ctypes.windll.user32
        try:
            user32.ShowWindow(wintypes.HWND(hwnd), SW_RESTORE)
            user32.BringWindowToTop(wintypes.HWND(hwnd))
            success = bool(user32.SetForegroundWindow(wintypes.HWND(hwnd)))
            return success or self._foreground_window_handle() == hwnd
        except Exception:
            return False

    def _parse_env_file(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        values: dict[str, str] = {}
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
        except Exception:
            return {}
        return values

# Author: Konstantin Markov

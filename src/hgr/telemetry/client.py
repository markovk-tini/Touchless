"""Telemetry client: thread-safe event queue + background flush
to PostHog's `/batch/` endpoint. Designed to be a complete no-op
when no API key is configured so we can safely bundle the wire-up
in every build and only flip the switch (env var or
`config.POSTHOG_API_KEY`) when ready to actually collect data.

The client doesn't depend on the `posthog` package — uses
`urllib.request` directly so we don't pay for an extra dep just
for HTTP POST.
"""
from __future__ import annotations

import hashlib
import json
import queue
import sys
import threading
import time
import urllib.request
import urllib.error
from typing import Any

from . import config as _config


def _now_iso8601() -> str:
    # PostHog accepts ISO-8601 with millisecond precision + UTC Z.
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + (
        f".{int((time.time() % 1) * 1000):03d}Z"
    )


# Salt makes the derived install_id project-specific and one-way: even
# if someone else also reads MachineGuid, they can't compute the same
# hash without this string. Bump the salt (e.g. "touchless:v3") to
# force a global re-keying of every install_id (intentional break: all
# existing users would appear as new IDs going forward).
#
# v2 (current) — MachineGuid only. One ID per Windows installation.
# v1 (deprecated) — included the Windows username; two accounts on
#                   the same PC produced two IDs.
_INSTALL_ID_SALT = "touchless:v2"


def derive_stable_install_id() -> str:
    """Derive a per-DEVICE anonymous install_id.

    Returns a 32-char hex string derived from SHA-256(salt + Windows
    MachineGuid). One value per Windows installation — same hash for
    every user account on that machine, every reboot, every app
    update, every uninstall + reinstall. Only changes if the user
    reinstalls Windows itself (which creates a new MachineGuid).

    Why not include the username? Two accounts on the same physical
    PC would otherwise count as two distinct "users" in the dashboard,
    which is wrong for a per-device usage tracker. If you ever need
    per-Windows-user granularity, change the salt to a new version
    (e.g. "touchless:v3") and add `getpass.getuser()` back into the
    raw input — that's an intentional schema break that produces a
    fresh set of install_ids.

    Why not include IP / hardware serials? IP changes with the network
    (mobile, VPN, etc.) and the client can't read its own public IP
    without an external HTTP call. Hardware serials need WMI / admin
    in many Windows configurations and aren't more stable than the
    MachineGuid anyway.

    Anonymous: the input is hashed one-way and the hash includes a
    project-specific salt, so the value can't be reversed back to
    the MachineGuid.

    Returns the empty string when the input can't be read (non-Windows,
    registry locked, etc.) — callers fall back to uuid4() in that case.
    """
    try:
        if sys.platform != "win32":
            return ""
        import winreg  # type: ignore[import-not-found]
        # MachineGuid: stable per Windows install. Resets only on a
        # clean reinstall of Windows. Lives under HKLM so 64-bit access
        # is explicit to avoid the WoW6432 redirector on 32-bit Python.
        machine_guid = ""
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
                0,
                winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
            ) as key:
                machine_guid, _ = winreg.QueryValueEx(key, "MachineGuid")
        except Exception:
            return ""
        raw = f"{_INSTALL_ID_SALT}:{machine_guid}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]
    except Exception:
        return ""


class TelemetryClient:
    """Buffered, fire-and-forget telemetry poster.

    Public API:
        client = TelemetryClient(install_id="...", app_version="...")
        client.track("gesture_fired", {"gesture": "swipe_right"})
        client.shutdown()  # called on app exit; flushes pending events
    """

    # Sentinel pushed onto the queue from shutdown() so the
    # background thread wakes from its blocking queue.get and
    # checks _stop_event without having to wait the full 30 s
    # FLUSH_INTERVAL_SECONDS timeout.
    _SHUTDOWN_SENTINEL: object = object()

    def __init__(
        self,
        *,
        install_id: str,
        app_version: str | None = None,
        api_key: str | None = None,
        host: str | None = None,
        user_opt_in: bool = False,
    ) -> None:
        self._install_id = str(install_id) if install_id else ""
        self._app_version = str(app_version or "")
        self._api_key = api_key if api_key is not None else _config.resolve_api_key()
        self._host = host if host is not None else _config.resolve_host()
        # Two layers gate the worker: technical (have api key +
        # install id) and consent (user explicitly opted in via
        # the first-run dialog or Settings → About toggle). Both
        # must be true to actually emit events. Default
        # user_opt_in=False so a fresh install never sends a
        # single track() call until the user agrees.
        self._technical_enabled = bool(self._api_key) and bool(self._install_id)
        self._user_opt_in = bool(user_opt_in)
        self._enabled = self._technical_enabled and self._user_opt_in
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=_config.QUEUE_MAX_SIZE
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._dropped = 0
        self._sent = 0
        # Latched True the first time an `app_session_started` event
        # is successfully queued. set_user_opt_in() uses it to detect
        # the "init-time start was dropped because user hadn't opted
        # in yet" case and replay the start when consent arrives. Stops
        # later opt-in toggles in the same session from double-firing.
        self._session_started_emitted = False
        if self._enabled:
            self._thread = threading.Thread(
                target=self._run, name="telemetry-poster", daemon=True,
            )
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def technical_enabled(self) -> bool:
        """True if api key and install id are present, regardless of
        user consent. Useful for surfacing in Settings → About so we
        can tell the user "telemetry is configured but disabled" vs
        "telemetry not available on this build"."""
        return self._technical_enabled

    @property
    def user_opt_in(self) -> bool:
        return self._user_opt_in

    def set_user_opt_in(
        self,
        enabled: bool,
        replay_properties: dict[str, Any] | None = None,
    ) -> None:
        """Flip the user-consent gate at runtime (called when the
        user toggles the analytics switch in Settings → About).
        Lazily starts the poster thread on opt-in; signals stop on
        opt-out so any buffered events drain.

        Also: if MainWindow already tried to fire `app_session_started`
        during _init_telemetry (before the user had a chance to opt
        in via the first-run dialog), that event was silently dropped
        because `_enabled` was False at the time. We track whether
        any event has been emitted yet via `_sent` and replay
        `app_session_started` on the first opt-in flip so the
        dashboard sees a matching start for the session-end event
        that will fire at shutdown. Without this, an opt-in mid-
        session showed up as "N minutes total time, 0 sessions"."""
        new_value = bool(enabled)
        if new_value == self._user_opt_in:
            return
        self._user_opt_in = new_value
        was_enabled = self._enabled
        self._enabled = self._technical_enabled and self._user_opt_in
        if self._enabled and not was_enabled:
            if self._thread is None or not self._thread.is_alive():
                self._stop_event.clear()
                self._thread = threading.Thread(
                    target=self._run, name="telemetry-poster", daemon=True,
                )
                self._thread.start()
            # Replay the missed session_started exactly once. The
            # _session_started_emitted flag is set by track() whenever
            # an app_session_started event lands in the queue, so:
            # - if the init-time start succeeded (user had already
            #   opted in from a previous run), flag is True → skip
            #   replay; no double-count.
            # - if the init-time start was dropped (user hadn't opted
            #   in yet at __init__), flag is False → replay now.
            # - if the user toggles opt-in OFF and back ON inside the
            #   same session, flag is True from the earlier emit →
            #   skip; one start per process.
            if not self._session_started_emitted:
                # Merge any caller-supplied props (e.g. custom_gesture_count)
                # so the replayed start carries the same metadata the
                # init-time emit would have if the user had already opted
                # in. Without this, first-time opt-in sessions land in the
                # dashboard with NULL for properties the init path sets.
                props: dict[str, Any] = {"opt_in_path": "first_run_dialog"}
                if replay_properties:
                    props.update(replay_properties)
                self.track("app_session_started", props)
        elif was_enabled and not self._enabled:
            # User just opted out: don't kill the thread (it might
            # still flush queued events on a clean shutdown), but
            # subsequent track() calls will short-circuit at the
            # `if not self._enabled` check.
            pass

    @property
    def install_id(self) -> str:
        return self._install_id

    def track(self, event: str, properties: dict[str, Any] | None = None) -> None:
        """Enqueue an event. Silent no-op when telemetry is
        disabled or the queue is full (drops oldest semantics —
        actually drops the new event, since back-pressure on a
        recurring caller is preferable to losing the start-of-day
        events that anchor the funnel)."""
        if not self._enabled or self._stop_event.is_set():
            return
        if not event:
            return
        try:
            payload = {
                "event": str(event),
                "distinct_id": self._install_id,
                "timestamp": _now_iso8601(),
                "properties": self._normalise_properties(properties),
            }
            self._queue.put_nowait(payload)
            if event == "app_session_started":
                self._session_started_emitted = True
        except queue.Full:
            self._dropped += 1
        except Exception:
            pass

    def consolidate(self, *, target_install_id: str, merge_ids: list[str]) -> bool:
        """Re-tag historic events under a single install_id.

        Fires one synchronous POST to /api/consolidate so the migration
        cleanup happens in a single shot. Silent no-op when telemetry
        is technically disabled (no api key) or `merge_ids` is empty.
        Returns True on a 2xx response. Runs from whatever thread
        the caller invokes it on — typically a one-shot background
        thread spawned by _init_telemetry so the GUI doesn't stall on
        the HTTP round-trip.

        Auth: reuses the same shared secret the /batch/ endpoint uses
        (already configured on the client as `self._api_key`).
        """
        if not self._technical_enabled:
            return False
        cleaned = [str(s) for s in (merge_ids or []) if s and str(s) != str(target_install_id)]
        if not cleaned:
            return False
        try:
            payload = json.dumps({
                "api_key": self._api_key,
                "target_install_id": str(target_install_id),
                "merge_ids": cleaned,
            }).encode("utf-8")
            user_agent = f"Touchless-Telemetry/{self._app_version or '0'} (Mozilla/5.0)"
            req = urllib.request.Request(
                f"{self._host.rstrip('/')}/api/consolidate",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": user_agent,
                },
                method="POST",
            )
            with urllib.request.urlopen(
                req, timeout=_config.HTTP_TIMEOUT_SECONDS,
            ) as resp:
                body = resp.read()
            print(
                f"[telemetry] consolidate -> {resp.status} {body[:160]!r}",
                flush=True,
            )
            return 200 <= int(resp.status) < 300
        except Exception as exc:
            print(f"[telemetry] consolidate failed: {exc!r}", flush=True)
            return False

    def shutdown(self, timeout: float = 4.0) -> None:
        """Stop the background thread and flush any buffered
        events. Safe to call multiple times; safe to call from the
        GUI thread (won't block more than `timeout` seconds)."""
        if not self._enabled:
            return
        self._stop_event.set()
        # Wake the worker out of its blocking queue.get so it sees
        # the stop signal immediately and drains any pending
        # events before the daemon thread is killed at process
        # exit.
        try:
            self._queue.put_nowait(self._SHUTDOWN_SENTINEL)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            try:
                thread.join(timeout=timeout)
            except Exception:
                pass

    # --- internals ---------------------------------------------------

    def _normalise_properties(
        self, properties: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {
            "app": "Touchless",
            "platform": sys.platform,
        }
        if self._app_version:
            merged["app_version"] = self._app_version
        if properties:
            for k, v in properties.items():
                if not isinstance(k, str):
                    continue
                # Coerce non-JSON-serialisable values to strings so
                # the event isn't dropped by the JSON encoder.
                try:
                    json.dumps(v)
                    merged[k] = v
                except (TypeError, ValueError):
                    merged[k] = repr(v)
        return merged

    def _run(self) -> None:
        while not self._stop_event.is_set():
            batch = self._collect_batch(blocking_timeout=_config.FLUSH_INTERVAL_SECONDS)
            if batch:
                self._send_batch(batch)
        # Final drain on shutdown.
        final = self._collect_batch(blocking_timeout=0.0)
        if final:
            self._send_batch(final)

    def _collect_batch(
        self, *, blocking_timeout: float,
    ) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        try:
            first = self._queue.get(timeout=blocking_timeout)
            if first is not self._SHUTDOWN_SENTINEL:
                batch.append(first)
        except queue.Empty:
            return batch
        # Drain any additional events that arrived during the wait.
        while len(batch) < 50:
            try:
                evt = self._queue.get_nowait()
                if evt is not self._SHUTDOWN_SENTINEL:
                    batch.append(evt)
            except queue.Empty:
                break
        return batch

    def _send_batch(self, batch: list[dict[str, Any]]) -> None:
        try:
            payload = {
                "api_key": self._api_key,
                "batch": batch,
            }
            data = json.dumps(payload).encode("utf-8")
            # Cloudflare's Bot Fight Mode blocks the default
            # `Python-urllib/X.Y` User-Agent at the edge (responds
            # 403 before the Worker even runs). Send a real-looking
            # UA — tagged so you can identify Touchless traffic in
            # Cloudflare logs.
            user_agent = f"Touchless-Telemetry/{self._app_version or '0'} (Mozilla/5.0)"
            req = urllib.request.Request(
                f"{self._host.rstrip('/')}/batch/",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": user_agent,
                },
                method="POST",
            )
            with urllib.request.urlopen(
                req, timeout=_config.HTTP_TIMEOUT_SECONDS,
            ) as resp:
                body = resp.read()
            self._sent += len(batch)
            print(
                f"[telemetry] sent {len(batch)} event(s) -> "
                f"{resp.status} {body[:120]!r}",
                flush=True,
            )
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read()[:200]
            except Exception:
                err_body = b""
            print(
                f"[telemetry] HTTP error {exc.code} {exc.reason} body={err_body!r}",
                flush=True,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"[telemetry] network error: {exc!r}", flush=True)
        except Exception as exc:
            import traceback
            print(f"[telemetry] unexpected error: {exc!r}", flush=True)
            traceback.print_exc()

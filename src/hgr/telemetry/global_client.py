"""Singleton bridge so call sites scattered across the codebase
can fire `telemetry.track(...)` without each module needing a
reference to the live `TelemetryClient`.

MainWindow constructs the client at startup and registers it via
`set_client(client)`. Anywhere else in the app — engine, voice,
gesture worker, custom gestures runner — calls `track(event,
properties)` and the helper routes through the singleton (or
silently drops the call if no client is registered yet, e.g.
during early startup or unit tests).
"""
from __future__ import annotations

from typing import Any, Optional

from .client import TelemetryClient


_client: Optional[TelemetryClient] = None


def set_client(client: Optional[TelemetryClient]) -> None:
    global _client
    _client = client


def get_client() -> Optional[TelemetryClient]:
    return _client


def track(event: str, properties: dict[str, Any] | None = None) -> None:
    """Fire-and-forget event track. Safe to call from any thread
    and at any point in the app lifecycle — no-op when no client
    is registered (e.g. before MainWindow finishes init)."""
    client = _client
    if client is None:
        return
    try:
        client.track(event, properties)
    except Exception:
        # Telemetry must never raise into call sites.
        pass


def track_error(component: str, exc: BaseException, *, extra: dict[str, Any] | None = None) -> None:
    """Anonymous error-rate event. `component` is a short stable
    string (e.g. "engine_init", "voice_command", "telemetry_init")
    so the dashboard can group by subsystem. We send the exception
    type and a truncated message — no traceback, no file paths, no
    user input."""
    if _client is None:
        return
    try:
        message = str(exc)
        if len(message) > 200:
            message = message[:200] + "…"
        properties: dict[str, Any] = {
            "component": str(component) or "unknown",
            "exc_type": exc.__class__.__name__,
            "message": message,
        }
        if extra:
            for k, v in extra.items():
                if isinstance(k, str):
                    properties.setdefault(k, v)
        track("error_caught", properties)
    except Exception:
        pass

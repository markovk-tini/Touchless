"""Anonymous usage telemetry for Touchless.

Tracks app-level events (gesture fires, voice commands, settings
navigation, tutorial completion, etc.) without any personal data:
- Random install UUID (no name, email, or IP-derived ID).
- No gesture landmark data, no voice transcripts, no file paths,
  no save locations, no microphone audio.
- Background posting; never blocks the GUI thread.
- Silent no-op when the PostHog API key isn't configured (the
  `client.TelemetryClient` reads the key from
  `hgr.telemetry.config.POSTHOG_API_KEY` and from the
  `TOUCHLESS_TELEMETRY_API_KEY` env var, in that order).

Wire-up: MainWindow constructs a `TelemetryClient` at startup and
exposes it via `self._telemetry`. Call sites use the lightweight
`telemetry.track(event, properties)` helper which routes through
the singleton instance set by MainWindow.
"""
from .client import TelemetryClient, derive_stable_install_id
from .global_client import get_client, set_client, track, track_error

__all__ = [
    "TelemetryClient",
    "derive_stable_install_id",
    "get_client",
    "set_client",
    "track",
    "track_error",
]

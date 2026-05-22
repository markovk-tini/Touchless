"""Telemetry endpoint config. The API key is hardcoded so every
shipped build (and every source run) reports to the same Cloudflare
Worker. The DASHBOARD that visualizes those events is gated by the
same `SHARED_SECRET` via the `?token=` URL param — users can ingest
events but can never view aggregates. Override the key or host via
`TOUCHLESS_TELEMETRY_API_KEY` / `TOUCHLESS_TELEMETRY_HOST` env vars
when you need to point a build at a staging backend."""
from __future__ import annotations

import os

# Hardcoded so the bundled installer reports to the dev's worker.
# Users never see this string — they just send events to the worker.
# Dashboard access is gated separately by the same secret via the
# dashboard URL's `?token=` query param.
POSTHOG_API_KEY: str = "bec4297146174344837d8f483b5618ea"

# Default PostHog-compatible ingest host. The Cloudflare Worker
# accepts the same `/batch/` shape PostHog does.
POSTHOG_HOST: str = "https://touchless-telemetry.konstantinvmarkov.workers.dev"

# Max events buffered before drops happen. Realistic apps with
# the call sites we plumb in average ~3-15 events per minute, so
# 500 is ~30+ minutes of offline buffering before any drop.
QUEUE_MAX_SIZE: int = 500

# Background flush cadence in seconds. Short enough that events
# arrive in near-real-time on the dashboard; long enough that we
# aren't hitting the network on every gesture.
FLUSH_INTERVAL_SECONDS: float = 30.0

# Per-flush HTTP timeout. Generous because PostHog's ingest can
# be slow on cold-start regions. We never block the GUI on this —
# it's a background thread.
HTTP_TIMEOUT_SECONDS: float = 10.0


def resolve_api_key() -> str:
    return os.environ.get("TOUCHLESS_TELEMETRY_API_KEY") or POSTHOG_API_KEY


def resolve_host() -> str:
    return os.environ.get("TOUCHLESS_TELEMETRY_HOST") or POSTHOG_HOST

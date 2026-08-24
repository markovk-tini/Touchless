"""User-facing status surfaces for the Sentinel daemon.

Phase-3 polish. The Sentinel runs ~4-5 background watchers; users
have no way to see what's running or whether any have auto-disabled
after repeated failures. This module produces compact, structured
status views the UI can render:

  * `current_status()` — one-line text summary ("4/4 watchers
    running, 1 auto-disabled after errors").
  * `watcher_table()` — list of dicts with per-watcher state +
    badges, ready for a settings page or debug pane.
  * `format_status_pill()` — single short string for a chat-header
    pill (green / yellow / red color hint).

All pure functions over the Sentinel's existing `state()` and
`watcher_specs()` accessors — no new substrate.

Author: Konstantin Markov
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class SentinelStatusBadge(str, Enum):
    GREEN = "green"        # all watchers healthy
    YELLOW = "yellow"      # some recoverable failures
    RED = "red"            # ≥1 watcher auto-disabled
    GREY = "grey"          # daemon stopped / not started


@dataclass
class SentinelStatusView:
    badge: SentinelStatusBadge
    one_line: str
    watcher_count: int
    running_count: int
    disabled_count: int
    total_runs: int
    total_failures: int


def _sentinel() -> Optional[Any]:
    try:
        from .sentinel import global_sentinel
        return global_sentinel()
    except Exception:
        return None


def current_status(sentinel: Optional[Any] = None) -> SentinelStatusView:
    """Read the Sentinel's state + watchers and produce a compact
    user-facing view."""
    s = sentinel or _sentinel()
    if s is None:
        return SentinelStatusView(
            badge=SentinelStatusBadge.GREY,
            one_line="Background watchers: unavailable.",
            watcher_count=0, running_count=0, disabled_count=0,
            total_runs=0, total_failures=0,
        )
    try:
        stats = s.state()
        watchers = s.watcher_specs()
    except Exception:
        return SentinelStatusView(
            badge=SentinelStatusBadge.GREY,
            one_line="Background watchers: state unreadable.",
            watcher_count=0, running_count=0, disabled_count=0,
            total_runs=0, total_failures=0,
        )
    total = len(watchers)
    disabled = sum(1 for w in watchers if getattr(w, "disabled", False))
    running = total - disabled
    state_str = getattr(stats, "state", "stopped")
    if state_str != "running":
        return SentinelStatusView(
            badge=SentinelStatusBadge.GREY,
            one_line=f"Background watchers: {state_str}.",
            watcher_count=total, running_count=0,
            disabled_count=disabled,
            total_runs=getattr(stats, "total_runs", 0),
            total_failures=getattr(stats, "total_failures", 0),
        )
    if disabled > 0:
        badge = SentinelStatusBadge.RED
        text = (f"{running}/{total} watchers running — {disabled} "
                f"auto-disabled after repeated errors.")
    elif getattr(stats, "total_failures", 0) > 0:
        badge = SentinelStatusBadge.YELLOW
        text = (f"{running}/{total} watchers running — "
                f"{stats.total_failures} recoverable errors so far.")
    else:
        badge = SentinelStatusBadge.GREEN
        text = f"{running}/{total} watchers running."
    return SentinelStatusView(
        badge=badge, one_line=text,
        watcher_count=total, running_count=running,
        disabled_count=disabled,
        total_runs=getattr(stats, "total_runs", 0),
        total_failures=getattr(stats, "total_failures", 0),
    )


def watcher_table(sentinel: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Per-watcher rows for a settings page / debug pane. Each row
    carries (name, interval, runs, failures, consecutive_failures,
    last_duration_ms, disabled)."""
    s = sentinel or _sentinel()
    if s is None:
        return []
    try:
        specs = s.watcher_specs()
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for w in specs:
        rows.append({
            "name": getattr(w, "name", ""),
            "interval_sec": getattr(w, "interval_sec", 0.0),
            "runs": getattr(w, "runs", 0),
            "failure_count": getattr(w, "failure_count", 0),
            "consecutive_failures": getattr(
                w, "consecutive_failures", 0),
            "last_duration_ms": getattr(w, "last_duration_ms", 0),
            "last_run_at": getattr(w, "last_run_at", 0.0),
            "disabled": getattr(w, "disabled", False),
        })
    return rows


def format_status_pill(sentinel: Optional[Any] = None
                       ) -> Dict[str, str]:
    """Chat-header friendly status pill data. Returns a dict with
    {'text', 'tooltip', 'color'} the UI can splat into a QLabel."""
    view = current_status(sentinel)
    color_map = {
        SentinelStatusBadge.GREEN:  "#3a7d3a",
        SentinelStatusBadge.YELLOW: "#9c8a3a",
        SentinelStatusBadge.RED:    "#b04040",
        SentinelStatusBadge.GREY:   "#666",
    }
    text = {
        SentinelStatusBadge.GREEN:  "⊙",
        SentinelStatusBadge.YELLOW: "⊙ warn",
        SentinelStatusBadge.RED:    "⊗",
        SentinelStatusBadge.GREY:   "·",
    }.get(view.badge, "·")
    return {
        "text": text,
        "tooltip": view.one_line,
        "color": color_map.get(view.badge, "#666"),
    }

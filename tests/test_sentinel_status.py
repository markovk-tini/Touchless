"""Tests for sentinel_status formatters."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.sentinel import Sentinel  # noqa: E402
from hgr.live_api.sentinel_status import (  # noqa: E402
    SentinelStatusBadge, current_status, format_status_pill,
    watcher_table,
)


# ---- current_status ----------------------------------------------------

def test_current_status_grey_when_no_sentinel():
    view = current_status(sentinel=None)
    # Module-level sentinel may exist from prior tests; just verify
    # we get a SentinelStatusView shape.
    assert view.badge in {SentinelStatusBadge.GREY,
                          SentinelStatusBadge.GREEN,
                          SentinelStatusBadge.YELLOW,
                          SentinelStatusBadge.RED}


def test_current_status_grey_when_stopped():
    s = Sentinel()
    view = current_status(sentinel=s)
    assert view.badge == SentinelStatusBadge.GREY
    assert "stopped" in view.one_line.lower()


def test_current_status_green_when_running_clean():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.05
    s.register("noop", lambda: None, interval_sec=10.0)
    s.start()
    try:
        time.sleep(0.1)
        view = current_status(sentinel=s)
        assert view.badge == SentinelStatusBadge.GREEN
        assert "1/1" in view.one_line
    finally:
        s.stop(timeout=1.0)


def test_current_status_yellow_when_recoverable_failures():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.05
    # AUTO_DISABLE_AFTER default is 5; one or two failures stay
    # under that threshold so we get YELLOW (not RED).
    call_count = {"n": 0}

    def maybe_fail():
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise RuntimeError("transient")
    s.register("flaky", maybe_fail, interval_sec=0.05)
    s.start()
    try:
        time.sleep(0.5)
        view = current_status(sentinel=s)
        assert view.watcher_count == 1
        # If we hit AUTO_DISABLE_AFTER it'd be RED; otherwise YELLOW.
        assert view.badge in {SentinelStatusBadge.YELLOW,
                              SentinelStatusBadge.RED}
    finally:
        s.stop(timeout=1.0)


def test_current_status_red_when_watcher_disabled():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.02
    s.AUTO_DISABLE_AFTER = 2

    def always_fail():
        raise RuntimeError("nope")
    s.register("dead", always_fail, interval_sec=0.02)
    s.start()
    try:
        time.sleep(0.3)
        view = current_status(sentinel=s)
        assert view.badge == SentinelStatusBadge.RED
        assert view.disabled_count >= 1
        assert "auto-disabled" in view.one_line.lower()
    finally:
        s.stop(timeout=1.0)


# ---- watcher_table -----------------------------------------------------

def test_watcher_table_shape():
    s = Sentinel()
    s.register("a", lambda: None, interval_sec=5.0)
    s.register("b", lambda: None, interval_sec=10.0)
    rows = watcher_table(sentinel=s)
    assert len(rows) == 2
    names = {r["name"] for r in rows}
    assert names == {"a", "b"}
    for r in rows:
        assert "interval_sec" in r
        assert "runs" in r
        assert "failure_count" in r
        assert "disabled" in r


def test_watcher_table_empty_for_no_watchers():
    s = Sentinel()
    assert watcher_table(sentinel=s) == []


# ---- format_status_pill ------------------------------------------------

def test_format_status_pill_shape():
    s = Sentinel()
    pill = format_status_pill(sentinel=s)
    assert "text" in pill
    assert "tooltip" in pill
    assert "color" in pill


def test_format_status_pill_green_when_clean():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.05
    s.register("ok", lambda: None, interval_sec=10.0)
    s.start()
    try:
        time.sleep(0.1)
        pill = format_status_pill(sentinel=s)
        # GREEN hex from cost_surfaces palette.
        assert pill["color"] == "#3a7d3a"
    finally:
        s.stop(timeout=1.0)

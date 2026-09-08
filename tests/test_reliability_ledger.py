"""Tests for the ReliabilityLedger and ErrorClassifier (Phase 2)."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from hgr.live_api.reliability_ledger import (  # noqa: E402
    ReliabilityLedger, classify_error, _decay_weight,
)
from hgr.live_api.tool_invocation import (  # noqa: E402
    InvocationBus, InvocationSource, ToolInvocation,
)


def _fresh_ledger() -> ReliabilityLedger:
    d = Path(tempfile.mkdtemp())
    return ReliabilityLedger(db_path=d / "rel.db", auto_subscribe=False)


# ---- classify_error ----------------------------------------------------

def test_classify_error_known_classes():
    assert classify_error("Connection timed out") == "transient_network"
    assert classify_error("HTTP 503 Service Unavailable") == "upstream_5xx"
    assert classify_error("401 Unauthorized") == "auth_revoked"
    assert classify_error("invalid_grant") == "auth_revoked"
    assert classify_error("HTTP 404 not found") == "not_found"
    assert classify_error("HTTP 429 too many requests") == "rate_limited"
    assert classify_error("recipient address invalid") == "recipient_invalid"
    assert classify_error("unresolved reference {step:1.id}") == "ref_unresolved"
    assert classify_error("user declined the prompt") == "user_cancelled"
    assert classify_error("precondition_not_met: gmail not connected") == "not_connected"


def test_classify_error_falls_through_to_other():
    assert classify_error(None) == "other"
    assert classify_error("") == "other"
    assert classify_error("kernel panic at 0x000123") == "other"


def test_decay_weight_halves_each_day():
    assert _decay_weight(0) == pytest.approx(1.0)
    # 24h later → 0.5
    assert _decay_weight(24 * 3600) == pytest.approx(0.5)
    # 48h later → 0.25
    assert _decay_weight(48 * 3600) == pytest.approx(0.25)


# ---- record + health roll-up -------------------------------------------

def test_record_creates_health_and_history():
    led = _fresh_ledger()
    led.record(tool="weather_get", status="ok", duration_ms=120)
    led.record(tool="weather_get", status="ok", duration_ms=140)
    led.record(tool="weather_get", status="error",
               error_text="HTTP 503", duration_ms=300)
    health = led.tool_health("weather_get")
    assert health is not None
    assert health["samples"] == 3
    assert health["ok_count"] == 2
    assert health["error_count"] == 1
    assert health["last_error_text"] == "HTTP 503"


def test_record_unknown_tool_returns_none_health():
    led = _fresh_ledger()
    assert led.tool_health("never_called") is None


def test_recent_errors_filtering_and_ordering():
    led = _fresh_ledger()
    led.record(tool="gmail_send", status="error", error_text="HTTP 401")
    led.record(tool="gmail_send", status="ok")
    led.record(tool="gmail_send", status="error", error_text="HTTP 503")
    errs = led.recent_errors("gmail_send", window_sec=10)
    # only errors, newest first
    assert len(errs) == 2
    assert errs[0]["error_text"] == "HTTP 503"
    assert errs[0]["error_class"] == "upstream_5xx"
    assert errs[1]["error_class"] == "auth_revoked"


def test_error_rate_recent_for_clean_tool_is_zero():
    led = _fresh_ledger()
    for _ in range(5):
        led.record(tool="volume_set", status="ok")
    assert led.error_rate_recent("volume_set") == 0.0


def test_error_rate_recent_weights_fresh_higher_than_old():
    led = _fresh_ledger()
    # Force timestamps: poke history rows directly so we control ages.
    now = time.time()
    led._conn.execute(
        "INSERT INTO invocation_history(ts, tool, status, error_class) "
        "VALUES (?, ?, 'error', 'upstream_5xx')", (now, "x"))
    led._conn.execute(
        "INSERT INTO invocation_history(ts, tool, status) "
        "VALUES (?, ?, 'ok')", (now - 24 * 3600, "x"))
    # Fresh error (weight 1) vs day-old ok (weight 0.5)
    rate = led.error_rate_recent("x", window_sec=48 * 3600)
    assert rate > 0.5  # error dominates


def test_is_currently_flaky_requires_min_samples():
    led = _fresh_ledger()
    # 2 errors only → below the 3-sample floor
    led.record(tool="gmail_send", status="error", error_text="HTTP 500")
    led.record(tool="gmail_send", status="error", error_text="HTTP 500")
    assert led.is_currently_flaky("gmail_send") is False
    led.record(tool="gmail_send", status="error", error_text="HTTP 500")
    assert led.is_currently_flaky("gmail_send") is True


def test_error_class_counts_groups_by_class():
    led = _fresh_ledger()
    led.record(tool="ms_mail_send", status="error", error_text="HTTP 401")
    led.record(tool="ms_mail_send", status="error", error_text="invalid_grant")
    led.record(tool="ms_mail_send", status="error", error_text="HTTP 503")
    counts = led.error_class_counts("ms_mail_send", window_sec=60)
    assert counts["auth_revoked"] == 2
    assert counts["upstream_5xx"] == 1


def test_history_cap_rolls_off_oldest():
    led = _fresh_ledger()
    led.HISTORY_CAP_PER_TOOL = 5  # squeeze the cap for the test
    for i in range(10):
        led.record(tool="cap_test", status="ok", duration_ms=i)
    cur = led._conn.execute(
        "SELECT COUNT(*) FROM invocation_history WHERE tool='cap_test'")
    n = int(cur.fetchone()[0])
    assert n == 5


# ---- bus subscription end-to-end ---------------------------------------

def test_bus_subscriber_records_invocations():
    bus = InvocationBus()
    d = Path(tempfile.mkdtemp())
    led = ReliabilityLedger(db_path=d / "rel.db", bus=bus,
                            auto_subscribe=True)
    inv = ToolInvocation(
        invocation_id="t1", turn_id="turn-1", tool="weather_get",
        args={}, source=InvocationSource.PLANNER, status="ok")
    bus.publish(inv)
    health = led.tool_health("weather_get")
    assert health is not None and health["ok_count"] == 1


def _make_inv(tool: str, status: str = "ok", incognito: bool = False
              ) -> ToolInvocation:
    inv = ToolInvocation(
        invocation_id=f"inv-{tool}", turn_id="turn-x", tool=tool,
        args={}, source=InvocationSource.PLANNER, status=status)
    if incognito:
        inv.extra["incognito"] = True
    return inv


def test_bus_subscriber_honors_incognito_tag():
    bus = InvocationBus()
    d = Path(tempfile.mkdtemp())
    led = ReliabilityLedger(db_path=d / "rel.db", bus=bus,
                            auto_subscribe=True)
    bus.publish(_make_inv("weather_get", incognito=True))
    # Incognito → ledger should skip this entirely.
    assert led.tool_health("weather_get") is None


def test_wipe_clears_both_tables():
    led = _fresh_ledger()
    led.record(tool="weather_get", status="ok")
    assert led.tool_health("weather_get") is not None
    n = led.wipe()
    assert n == 1
    assert led.tool_health("weather_get") is None

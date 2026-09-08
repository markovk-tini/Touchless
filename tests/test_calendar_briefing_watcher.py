"""Tests for CalendarBriefingWatcher (Phase 3 wiring)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from hgr.live_api.calendar_briefing_watcher import (  # noqa: E402
    CalendarBriefingWatcher, _normalize_events, _parse_event_ts,
)
from hgr.live_api.interruption_gate import (  # noqa: E402
    InterruptionGate, InterruptSeverity, SignalKind,
)
import hgr.live_api.interruption_gate as gate_mod  # noqa: E402


class _StubRegistry:
    def __init__(self, events):
        self._events = events
        self.calls = []

    def call(self, tool, args):
        self.calls.append((tool, args))
        return {"status": "ok", "count": len(self._events),
                "events": self._events}


def _event(start_ts, *, id="e1", summary="Sync"):
    return {"id": id, "summary": summary, "start":
            __import__("datetime").datetime.fromtimestamp(
                start_ts,
                __import__("datetime").timezone.utc).isoformat()}


def _open_gate(gate=None):
    """Return a gate seeded so SEC-002 fail-closed doesn't pre-empt.
    Quiet-hours disabled so the test passes regardless of wall-clock
    time-of-day (would otherwise fail when run between 22:00 and 07:00
    local time)."""
    from hgr.live_api.interruption_gate import GateConfig
    g = gate or InterruptionGate(
        config=GateConfig(quiet_start_hour=0, quiet_end_hour=0))
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    g.set_signal(SignalKind.MIC_IN_USE, False)
    g.set_signal(SignalKind.CAMERA_IN_USE, False)
    return g


# ---- event time parsing ------------------------------------------------

def test_parse_event_ts_rfc3339_with_offset():
    ts = _parse_event_ts("2026-06-04T14:30:00-07:00")
    assert ts is not None
    assert ts > 0


def test_parse_event_ts_rfc3339_with_z():
    ts = _parse_event_ts("2026-06-04T14:30:00Z")
    assert ts is not None


def test_parse_event_ts_handles_garbage():
    assert _parse_event_ts("") is None
    assert _parse_event_ts(None) is None
    assert _parse_event_ts("not a date") is None


def test_normalize_events_handles_missing_fields():
    raw = [
        {"id": "e1", "summary": "Standup",
         "start": "2026-06-04T09:00:00Z"},
        {"id": "e2"},  # missing start — dropped
        "not a dict",  # dropped
    ]
    out = _normalize_events(raw)
    assert len(out) == 1
    assert out[0].id == "e1"
    assert out[0].title == "Standup"


# ---- watcher tick ------------------------------------------------------

def test_tick_no_registry_is_noop():
    watcher = CalendarBriefingWatcher(
        registry=None, notifier=lambda *a: None)
    watcher.tick()  # should not raise


def test_tick_fires_notifier_for_upcoming_event(monkeypatch):
    # Install a fresh gate that won't fail-closed.
    g = _open_gate()
    monkeypatch.setattr(gate_mod, "_gate", g)
    # Event starting in 2 minutes.
    soon = time.time() + 120
    reg = _StubRegistry([_event(soon, summary="Demo")])
    received = []
    watcher = CalendarBriefingWatcher(
        registry=reg, notifier=lambda b, d: received.append((b, d)),
        lead_minutes=5)
    watcher.tick()
    assert len(received) == 1
    briefing, decision = received[0]
    assert briefing.title == "Demo"
    assert decision.allow is True


def test_tick_respects_gate_when_screen_sharing(monkeypatch):
    g = _open_gate()
    g.set_signal(SignalKind.SCREEN_SHARING, True)
    monkeypatch.setattr(gate_mod, "_gate", g)
    soon = time.time() + 120
    reg = _StubRegistry([_event(soon)])
    received = []
    watcher = CalendarBriefingWatcher(
        registry=reg, notifier=lambda b, d: received.append((b, d)))
    watcher.tick()
    # Gate blocked → notifier never called.
    assert received == []


def test_tick_skips_already_briefed_event(monkeypatch):
    g = _open_gate()
    monkeypatch.setattr(gate_mod, "_gate", g)
    soon = time.time() + 120
    reg = _StubRegistry([_event(soon, id="repeat")])
    n = []
    watcher = CalendarBriefingWatcher(
        registry=reg, notifier=lambda b, d: n.append((b, d)))
    watcher.tick()
    watcher.tick()  # second tick — same event
    assert len(n) == 1  # cooldown blocked the repeat


def test_tick_does_not_mark_briefed_when_gate_blocks(monkeypatch):
    """Important: if the gate said NO, we MUST NOT mark briefed —
    otherwise the user permanently misses that event."""
    g = _open_gate()
    g.set_signal(SignalKind.SCREEN_SHARING, True)
    monkeypatch.setattr(gate_mod, "_gate", g)
    soon = time.time() + 120
    reg = _StubRegistry([_event(soon, id="blocked")])
    n = []
    watcher = CalendarBriefingWatcher(
        registry=reg, notifier=lambda b, d: n.append((b, d)))
    watcher.tick()
    assert n == []
    # Now stop sharing and tick again — should fire.
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    watcher.tick()
    assert len(n) == 1


def test_tick_handles_calendar_call_failure(monkeypatch):
    g = _open_gate()
    monkeypatch.setattr(gate_mod, "_gate", g)

    class BadReg:
        def call(self, *a, **kw):
            raise RuntimeError("Calendar service down")
    watcher = CalendarBriefingWatcher(
        registry=BadReg(), notifier=lambda *a: None)
    watcher.tick()  # should not raise


def test_tick_skips_far_future_events(monkeypatch):
    g = _open_gate()
    monkeypatch.setattr(gate_mod, "_gate", g)
    far = time.time() + 3600  # 1h from now
    reg = _StubRegistry([_event(far, summary="Later")])
    n = []
    watcher = CalendarBriefingWatcher(
        registry=reg, notifier=lambda b, d: n.append((b, d)),
        lead_minutes=5)
    watcher.tick()
    assert n == []

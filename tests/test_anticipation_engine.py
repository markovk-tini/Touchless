"""Tests for AnticipationEngine (Phase 6 B2)."""
from __future__ import annotations

import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.incognito as inc  # noqa: E402
from hgr.live_api.anticipation_engine import (  # noqa: E402
    Anticipation, AnticipationEngine, AnticipationKind,
    MEETING_IMMINENT_LEAD_MIN, _hour_bucket,
    _normalize_window, _parse_event_ts,
)
from hgr.live_api.interruption_gate import (  # noqa: E402
    GateConfig, InterruptionGate, SignalKind,
)


def setup_function():
    inc.set_incognito(False)
    # Phase-7: InterruptionGate now consults global affect state
    # for LOW severity. Reset to avoid pollution from earlier tests.
    try:
        from hgr.live_api.affect import reset_global as _aff_reset
        _aff_reset()
    except Exception:
        pass


def _open_gate():
    g = InterruptionGate(config=GateConfig(quiet_start_hour=0,
                                            quiet_end_hour=0))
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    g.set_signal(SignalKind.MIC_IN_USE, False)
    g.set_signal(SignalKind.CAMERA_IN_USE, False)
    return g


# ---- helpers ----------------------------------------------------------

def test_parse_event_ts_round_trip():
    iso = datetime.fromtimestamp(
        1_700_000_000, timezone.utc).isoformat()
    ts = _parse_event_ts(iso)
    assert ts is not None
    assert abs(ts - 1_700_000_000) < 1


def test_parse_event_ts_handles_garbage():
    assert _parse_event_ts("") is None
    assert _parse_event_ts(None) is None
    assert _parse_event_ts("not a date") is None


def test_normalize_window_strips_app_suffix():
    out = _normalize_window(
        "report.docx - Microsoft Word")
    # Last ' - ' separated segment is treated as the app.
    assert out == "Microsoft Word"


def test_normalize_window_passes_through_simple():
    assert _normalize_window("Notepad") == "Notepad"
    assert _normalize_window("") == ""


def test_hour_bucket_returns_local_hour():
    bucket = _hour_bucket(time.time())
    assert 0 <= bucket <= 23


# ---- meeting_imminent --------------------------------------------------

class _CalendarRegistry:
    def __init__(self, events):
        self._events = events
        self.calls = []

    def call(self, tool, args):
        self.calls.append((tool, args))
        return {"status": "ok", "events": self._events,
                "count": len(self._events)}


def _meeting_at(minutes_from_now, *, summary="Sync", id="m1",
                 attendees=None):
    ts = time.time() + minutes_from_now * 60
    return {
        "id": id, "summary": summary,
        "start": datetime.fromtimestamp(
            ts, timezone.utc).isoformat(),
        "attendees": attendees or [],
    }


def test_meeting_imminent_fires_inside_window():
    fired = []
    reg = _CalendarRegistry([_meeting_at(
        10, summary="Standup", id="m1",
        attendees=["Dani <dani@x>"])])
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=reg, gate=_open_gate())
    engine.tick()
    assert len(fired) == 1
    a = fired[0]
    assert a.kind == AnticipationKind.MEETING_IMMINENT
    assert "Standup" in a.headline
    assert "Dani" in a.headline


def test_meeting_imminent_skipped_outside_window():
    fired = []
    # 4 min away — too close (briefing watcher's job).
    reg = _CalendarRegistry([_meeting_at(4)])
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=reg, gate=_open_gate())
    engine.tick()
    assert all(a.kind != AnticipationKind.MEETING_IMMINENT
               for a in fired)


def test_meeting_imminent_skipped_when_too_far_future():
    fired = []
    # 60 min away — outside the imminent window.
    reg = _CalendarRegistry([_meeting_at(60)])
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=reg, gate=_open_gate())
    engine.tick()
    assert fired == []


def test_meeting_imminent_cooldown_blocks_repeat():
    fired = []
    reg = _CalendarRegistry([_meeting_at(10, id="m-repeat")])
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=reg, gate=_open_gate())
    engine.tick()
    engine.tick()
    # Same event → cooldown blocks.
    assert len(fired) == 1


def test_meeting_imminent_caps_per_tick():
    fired = []
    reg = _CalendarRegistry([
        _meeting_at(8, id="m1"),
        _meeting_at(10, id="m2"),
        _meeting_at(12, id="m3"),
    ])
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=reg, gate=_open_gate())
    engine.tick()
    # MAX_IMMINENT_PER_TICK = 1 → only the soonest fires.
    assert len(fired) == 1


def test_meeting_imminent_skipped_without_registry():
    fired = []
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=None, gate=_open_gate())
    engine.tick()
    assert fired == []


# ---- stale standing order ---------------------------------------------

def test_stale_standing_order_fires_after_12h(monkeypatch):
    """A standing order that fired 13h ago + is still ACTIVE +
    has fire_count > 0 → fire a 'still want it?' anticipation."""
    from hgr.live_api.standing_orders import (
        StandingOrdersStore, OrderState,
        make_inbox_watch_order)
    d = Path(tempfile.mkdtemp())
    store = StandingOrdersStore(db_path=d / "o.db")
    order = make_inbox_watch_order(
        user_text="watch", query="Q3")
    store.add(order)
    # Force a fire 13h ago.
    order.fire_count = 1
    order.last_fired_at = time.time() - 13 * 3600
    store.update_after_check(order, fired=False)
    # Patch the global store.
    import hgr.live_api.standing_orders as so
    monkeypatch.setattr(so, "_store", store)
    fired = []
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=None, gate=_open_gate())
    engine.tick()
    stale = [a for a in fired
             if a.kind == AnticipationKind.STALE_STANDING_ORDER]
    assert len(stale) == 1
    assert "12h" in stale[0].headline or "13h" in stale[0].headline


def test_stale_standing_order_skipped_when_recently_fired(monkeypatch):
    from hgr.live_api.standing_orders import (
        StandingOrdersStore, make_inbox_watch_order)
    d = Path(tempfile.mkdtemp())
    store = StandingOrdersStore(db_path=d / "o.db")
    order = make_inbox_watch_order(
        user_text="watch", query="Q3")
    store.add(order)
    # Fired only 1h ago.
    order.fire_count = 1
    order.last_fired_at = time.time() - 3600
    store.update_after_check(order, fired=False)
    import hgr.live_api.standing_orders as so
    monkeypatch.setattr(so, "_store", store)
    fired = []
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=None, gate=_open_gate())
    engine.tick()
    assert all(a.kind != AnticipationKind.STALE_STANDING_ORDER
               for a in fired)


# ---- recurring routine ------------------------------------------------

def test_recurring_routine_fires_after_3_distinct_days(monkeypatch):
    """Same (window, hour_bucket) on 3+ distinct days → routine."""
    fired = []
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=None, gate=_open_gate())
    # Pre-seed activity log: 3 prior days + today.
    import hgr.live_api.anticipation_engine as ae
    monkeypatch.setattr(ae, "_current_window_title",
                        lambda: "VS Code")
    now = time.time()
    bucket = _hour_bucket(now)
    today = datetime.fromtimestamp(now).date().isoformat()
    yest = datetime.fromtimestamp(now - 86400).date().isoformat()
    two = datetime.fromtimestamp(now - 2 * 86400).date().isoformat()
    key = ("VS Code", bucket)
    engine._activity_log[key] = {yest, two, today}
    engine.tick()
    routines = [a for a in fired
                if a.kind == AnticipationKind.RECURRING_ROUTINE]
    assert len(routines) == 1


def test_recurring_routine_skipped_below_threshold(monkeypatch):
    fired = []
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=None, gate=_open_gate())
    import hgr.live_api.anticipation_engine as ae
    monkeypatch.setattr(ae, "_current_window_title",
                        lambda: "VS Code")
    # Only 2 prior days — under threshold.
    now = time.time()
    bucket = _hour_bucket(now)
    today = datetime.fromtimestamp(now).date().isoformat()
    yest = datetime.fromtimestamp(now - 86400).date().isoformat()
    engine._activity_log[("VS Code", bucket)] = {yest, today}
    engine.tick()
    assert all(a.kind != AnticipationKind.RECURRING_ROUTINE
               for a in fired)


# ---- gating ----------------------------------------------------------

def test_engine_honors_incognito():
    fired = []
    reg = _CalendarRegistry([_meeting_at(10)])
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=reg, gate=_open_gate())
    inc.set_incognito(True)
    try:
        engine.tick()
    finally:
        inc.set_incognito(False)
    assert fired == []


def test_engine_blocked_when_gate_closes():
    fired = []
    gate = _open_gate()
    gate.set_signal(SignalKind.SCREEN_SHARING, True)
    reg = _CalendarRegistry([_meeting_at(10)])
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=reg, gate=gate)
    engine.tick()
    assert fired == []


# ---- reset / construction --------------------------------------------

def test_reset_clears_state():
    fired = []
    engine = AnticipationEngine(
        handler=lambda a: fired.append(a),
        registry=None, gate=_open_gate())
    engine._last_fire[(AnticipationKind.MEETING_IMMINENT,
                       "k")] = time.time()
    engine.reset()
    assert engine._last_fire == {}


def test_anticipation_dataclass_defaults():
    a = Anticipation(kind=AnticipationKind.MEETING_IMMINENT,
                     headline="x")
    assert a.severity == "low"
    assert a.payload == {}
    assert a.ts > 0

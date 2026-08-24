"""Tests for ProactiveNudges (Phase 3 polish)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.interruption_gate import (  # noqa: E402
    GateConfig, InterruptionGate, SignalKind,
)
from hgr.live_api.proactive_nudges import (  # noqa: E402
    LONG_IDLE_THRESHOLD_SEC, Nudge, NudgeKind,
    PER_KIND_COOLDOWN_SEC, ProactiveNudges,
)


def _open_gate() -> InterruptionGate:
    """Gate that won't fail-closed and won't block on quiet hours."""
    g = InterruptionGate(config=GateConfig(quiet_start_hour=0,
                                            quiet_end_hour=0))
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    g.set_signal(SignalKind.MIC_IN_USE, False)
    g.set_signal(SignalKind.CAMERA_IN_USE, False)
    return g


# ---- long-idle nudge ---------------------------------------------------

def test_long_idle_fires_when_threshold_crossed():
    gate = _open_gate()
    gate.set_signal(SignalKind.USER_IDLE_SEC,
                    LONG_IDLE_THRESHOLD_SEC + 60)
    received = []
    nudges = ProactiveNudges(
        handler=lambda n: received.append(n),
        gate=gate, cost_meter=_NoCapMeter(), ledger=_EmptyLedger())
    nudges.tick()
    # If we're outside the 1am-7am quiet window, expect a fire.
    import datetime as _dt
    h = _dt.datetime.now().hour
    if 1 <= h < 7:
        # Inside the explicit night-skip; should NOT fire.
        assert all(n.kind != NudgeKind.LONG_IDLE for n in received)
    else:
        assert any(n.kind == NudgeKind.LONG_IDLE for n in received)


def test_long_idle_does_not_fire_when_active():
    gate = _open_gate()
    gate.set_signal(SignalKind.USER_IDLE_SEC, 30)
    received = []
    nudges = ProactiveNudges(handler=lambda n: received.append(n),
                              gate=gate, cost_meter=_NoCapMeter(),
                              ledger=_EmptyLedger())
    nudges.tick()
    assert all(n.kind != NudgeKind.LONG_IDLE for n in received)


def test_long_idle_only_fires_once_per_session():
    gate = _open_gate()
    gate.set_signal(SignalKind.USER_IDLE_SEC,
                    LONG_IDLE_THRESHOLD_SEC + 60)
    received = []
    nudges = ProactiveNudges(handler=lambda n: received.append(n),
                              gate=gate, cost_meter=_NoCapMeter(),
                              ledger=_EmptyLedger())
    nudges.tick()
    nudges.tick()
    nudges.tick()
    long_idles = [n for n in received if n.kind == NudgeKind.LONG_IDLE]
    # Either zero (night skip) or exactly one.
    assert len(long_idles) <= 1


# ---- cost-cap nudges ---------------------------------------------------

class _Meter:
    def __init__(self, *, spent: float, cap: float = 5.0):
        self.daily_cap_usd = cap
        self._spent = spent

    def today_total(self):
        return self._spent

    def by_model_today(self):
        return {}

    def history_last_n_days(self, n):
        return []


class _NoCapMeter:
    daily_cap_usd = 0.0
    def today_total(self): return 0.0
    def by_model_today(self): return {}
    def history_last_n_days(self, n): return []


class _EmptyLedger:
    def all_tool_health(self): return []
    def is_currently_flaky(self, tool, threshold=0.5): return False
    def recent_errors(self, tool, window_sec=0, limit=0): return []
    def error_class_counts(self, tool, window_sec=0): return {}


def test_near_cap_fires_at_80_percent():
    gate = _open_gate()
    gate.set_signal(SignalKind.USER_IDLE_SEC, 10)  # active
    meter = _Meter(spent=4.2, cap=5.0)  # 84%
    received = []
    nudges = ProactiveNudges(handler=lambda n: received.append(n),
                              gate=gate, cost_meter=meter,
                              ledger=_EmptyLedger())
    nudges.tick()
    assert any(n.kind == NudgeKind.NEAR_COST_CAP for n in received)


def test_at_cap_fires_at_95_percent():
    gate = _open_gate()
    gate.set_signal(SignalKind.USER_IDLE_SEC, 10)
    meter = _Meter(spent=4.8, cap=5.0)  # 96%
    received = []
    nudges = ProactiveNudges(handler=lambda n: received.append(n),
                              gate=gate, cost_meter=meter,
                              ledger=_EmptyLedger())
    nudges.tick()
    kinds = [n.kind for n in received]
    # AT cap takes precedence; NEAR shouldn't ALSO fire on the same tick.
    assert NudgeKind.AT_COST_CAP in kinds
    assert NudgeKind.NEAR_COST_CAP not in kinds


def test_cost_nudges_skipped_when_no_cap():
    gate = _open_gate()
    gate.set_signal(SignalKind.USER_IDLE_SEC, 10)
    received = []
    nudges = ProactiveNudges(handler=lambda n: received.append(n),
                              gate=gate, cost_meter=_NoCapMeter(),
                              ledger=_EmptyLedger())
    nudges.tick()
    assert all(n.kind not in (NudgeKind.NEAR_COST_CAP,
                              NudgeKind.AT_COST_CAP)
               for n in received)


def test_at_cap_cooldown_blocks_repeat():
    gate = _open_gate()
    gate.set_signal(SignalKind.USER_IDLE_SEC, 10)
    meter = _Meter(spent=4.8, cap=5.0)
    received = []
    nudges = ProactiveNudges(handler=lambda n: received.append(n),
                              gate=gate, cost_meter=meter,
                              ledger=_EmptyLedger())
    nudges.tick()
    nudges.tick()
    at_cap = [n for n in received if n.kind == NudgeKind.AT_COST_CAP]
    assert len(at_cap) == 1


# ---- unhealthy-tool nudge ---------------------------------------------

class _FlakeyLedger:
    def __init__(self, tool="gmail_send", count=6):
        self._tool = tool
        self._count = count

    def all_tool_health(self):
        return [{"tool": self._tool, "samples": self._count,
                 "ok_count": 0, "error_count": self._count}]

    def is_currently_flaky(self, tool, threshold=0.5):
        return tool == self._tool

    def recent_errors(self, tool, window_sec, limit):
        return [{"error_class": "auth_revoked"}] * self._count

    def error_class_counts(self, tool, window_sec):
        return {"auth_revoked": self._count}


def test_unhealthy_tool_fires_with_top_error_class():
    gate = _open_gate()
    gate.set_signal(SignalKind.USER_IDLE_SEC, 10)
    received = []
    nudges = ProactiveNudges(handler=lambda n: received.append(n),
                              gate=gate, cost_meter=_NoCapMeter(),
                              ledger=_FlakeyLedger())
    nudges.tick()
    unhealthy = [n for n in received
                 if n.kind == NudgeKind.UNHEALTHY_TOOL]
    assert len(unhealthy) == 1
    assert "auth_revoked" in unhealthy[0].body
    assert unhealthy[0].suggested_action.startswith("pause_tool:")


def test_unhealthy_per_tool_cooldown_distinct():
    """Two different flaky tools each get their own cooldown bucket."""
    gate = _open_gate()
    gate.set_signal(SignalKind.USER_IDLE_SEC, 10)
    received = []

    class _TwoToolLedger:
        def all_tool_health(self):
            return [{"tool": "gmail_send"}, {"tool": "ms_mail_send"}]

        def is_currently_flaky(self, tool, threshold=0.5):
            return True

        def recent_errors(self, tool, window_sec, limit):
            return [{"error_class": "auth_revoked"}] * 6

        def error_class_counts(self, tool, window_sec):
            return {"auth_revoked": 6}

    nudges = ProactiveNudges(handler=lambda n: received.append(n),
                              gate=gate, cost_meter=_NoCapMeter(),
                              ledger=_TwoToolLedger())
    nudges.tick()
    unhealthy = [n for n in received
                 if n.kind == NudgeKind.UNHEALTHY_TOOL]
    # Both tools should have fired (distinct cooldown buckets).
    assert len(unhealthy) == 2


# ---- gate respect + reset ---------------------------------------------

def test_nudges_blocked_when_gate_says_no():
    gate = _open_gate()
    gate.set_signal(SignalKind.SCREEN_SHARING, True)  # hard block
    gate.set_signal(SignalKind.USER_IDLE_SEC,
                    LONG_IDLE_THRESHOLD_SEC + 60)
    received = []
    nudges = ProactiveNudges(handler=lambda n: received.append(n),
                              gate=gate, cost_meter=_NoCapMeter(),
                              ledger=_EmptyLedger())
    nudges.tick()
    assert received == []


def test_nudges_honor_incognito(monkeypatch):
    import hgr.live_api.incognito as inc
    gate = _open_gate()
    gate.set_signal(SignalKind.USER_IDLE_SEC,
                    LONG_IDLE_THRESHOLD_SEC + 60)
    received = []
    nudges = ProactiveNudges(handler=lambda n: received.append(n),
                              gate=gate, cost_meter=_NoCapMeter(),
                              ledger=_EmptyLedger())
    inc.set_incognito(True)
    try:
        nudges.tick()
    finally:
        inc.set_incognito(False)
    assert received == []


def test_reset_clears_cooldowns():
    gate = _open_gate()
    gate.set_signal(SignalKind.USER_IDLE_SEC, 10)
    meter = _Meter(spent=4.8, cap=5.0)
    received = []
    nudges = ProactiveNudges(handler=lambda n: received.append(n),
                              gate=gate, cost_meter=meter,
                              ledger=_EmptyLedger())
    nudges.tick()
    received.clear()
    nudges.reset()
    nudges.tick()
    assert any(n.kind == NudgeKind.AT_COST_CAP for n in received)

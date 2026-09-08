"""Tests for ReliabilityAdvisor (Phase 5 B4)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.incognito as inc  # noqa: E402
from hgr.live_api.reliability_advisor import (  # noqa: E402
    _KNOWN_ALTERNATES, DEFAULT_CHRONIC_ERROR_RATE,
    DEFAULT_MIN_SAMPLES, ReliabilityAdvisor,
    ReliabilityAdvice,
)
from hgr.live_api.interruption_gate import (  # noqa: E402
    GateConfig, InterruptionGate, SignalKind,
)


def setup_function():
    inc.set_incognito(False)


def _open_gate() -> InterruptionGate:
    """Quiet-hours-disabled gate seeded so the advisor's gate
    check always allows LOW interruptions."""
    g = InterruptionGate(config=GateConfig(quiet_start_hour=0,
                                            quiet_end_hour=0))
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    g.set_signal(SignalKind.MIC_IN_USE, False)
    g.set_signal(SignalKind.CAMERA_IN_USE, False)
    return g


class _FakeLedger:
    """Per-tool error-rate stub."""

    def __init__(self, tools):
        """tools: list of {tool, samples, error_rate}"""
        self._tools = tools

    def all_tool_health(self):
        return [{"tool": t["tool"], "samples": t["samples"],
                 "ok_count": int(t["samples"] * (1 - t["error_rate"])),
                 "error_count": int(t["samples"] * t["error_rate"])}
                for t in self._tools]

    def error_rate_recent(self, tool, window_sec=0):
        for t in self._tools:
            if t["tool"] == tool:
                return float(t["error_rate"])
        return 0.0


# ---- basic firing ------------------------------------------------------

def test_advisor_fires_for_chronically_unhealthy_tool():
    gate = _open_gate()
    ledger = _FakeLedger([{"tool": "gmail_send",
                            "samples": 50,
                            "error_rate": 0.45}])
    fired = []
    advisor = ReliabilityAdvisor(
        handler=lambda a: fired.append(a),
        ledger=ledger, gate=gate)
    advisor.tick()
    assert len(fired) == 1
    advice = fired[0]
    assert advice.tool == "gmail_send"
    assert advice.suggested_alternate == "ms_mail_send"
    assert advice.alternate_friendly_name == "Outlook"
    assert "45%" in advice.headline


def test_advisor_skips_below_min_samples():
    gate = _open_gate()
    ledger = _FakeLedger([{"tool": "gmail_send",
                            "samples": 5, "error_rate": 0.9}])
    fired = []
    advisor = ReliabilityAdvisor(
        handler=lambda a: fired.append(a),
        ledger=ledger, gate=gate)
    advisor.tick()
    assert fired == []


def test_advisor_skips_below_error_rate_threshold():
    gate = _open_gate()
    ledger = _FakeLedger([{"tool": "gmail_send",
                            "samples": 100,
                            "error_rate": 0.10}])  # 10% < 30%
    fired = []
    advisor = ReliabilityAdvisor(
        handler=lambda a: fired.append(a),
        ledger=ledger, gate=gate)
    advisor.tick()
    assert fired == []


def test_advisor_provides_investigation_action_when_no_alternate():
    gate = _open_gate()
    ledger = _FakeLedger([{"tool": "obscure_custom_connector",
                            "samples": 50,
                            "error_rate": 0.5}])
    fired = []
    advisor = ReliabilityAdvisor(
        handler=lambda a: fired.append(a),
        ledger=ledger, gate=gate)
    advisor.tick()
    assert len(fired) == 1
    assert fired[0].suggested_alternate == ""
    assert fired[0].suggested_action.startswith("investigate_tool:")


# ---- gating ------------------------------------------------------------

def test_advisor_honors_incognito():
    gate = _open_gate()
    ledger = _FakeLedger([{"tool": "gmail_send",
                            "samples": 50, "error_rate": 0.5}])
    fired = []
    advisor = ReliabilityAdvisor(
        handler=lambda a: fired.append(a),
        ledger=ledger, gate=gate)
    inc.set_incognito(True)
    try:
        advisor.tick()
    finally:
        inc.set_incognito(False)
    assert fired == []


def test_advisor_blocked_when_gate_says_no():
    gate = _open_gate()
    gate.set_signal(SignalKind.SCREEN_SHARING, True)
    ledger = _FakeLedger([{"tool": "gmail_send",
                            "samples": 50, "error_rate": 0.5}])
    fired = []
    advisor = ReliabilityAdvisor(
        handler=lambda a: fired.append(a),
        ledger=ledger, gate=gate)
    advisor.tick()
    assert fired == []


# ---- per-tool cooldown -------------------------------------------------

def test_advisor_cooldown_blocks_repeat_for_same_tool():
    gate = _open_gate()
    ledger = _FakeLedger([{"tool": "gmail_send",
                            "samples": 50, "error_rate": 0.5}])
    fired = []
    advisor = ReliabilityAdvisor(
        handler=lambda a: fired.append(a),
        ledger=ledger, gate=gate)
    advisor.tick()
    advisor.tick()
    advisor.tick()
    # Multiple ticks within the cooldown window → only one advice.
    assert len(fired) == 1


def test_distinct_tools_each_advise():
    gate = _open_gate()
    ledger = _FakeLedger([
        {"tool": "gmail_send", "samples": 50, "error_rate": 0.5},
        {"tool": "ms_mail_send", "samples": 50, "error_rate": 0.4},
    ])
    fired = []
    advisor = ReliabilityAdvisor(
        handler=lambda a: fired.append(a),
        ledger=ledger, gate=gate)
    advisor.tick()
    assert {a.tool for a in fired} == {"gmail_send", "ms_mail_send"}


# ---- alternates table sanity -----------------------------------------

def test_known_alternates_have_friendly_names():
    """Every entry should have a non-empty friendly name (used in
    the user-facing nudge: 'Want me to prefer <name>?'). Strict
    bidirectional symmetry isn't required because multiple keys
    can validly point to the same alternate destination
    (e.g., outlook_compose and ms_mail_send both → Gmail)."""
    assert _KNOWN_ALTERNATES, "alternates table should be non-empty"
    for tool, (alt, name) in _KNOWN_ALTERNATES.items():
        assert alt, f"{tool} has empty alternate tool name"
        assert name, f"{tool} -> {alt} has empty friendly name"


def test_reset_clears_cooldowns():
    gate = _open_gate()
    ledger = _FakeLedger([{"tool": "gmail_send",
                            "samples": 50, "error_rate": 0.5}])
    fired = []
    advisor = ReliabilityAdvisor(
        handler=lambda a: fired.append(a),
        ledger=ledger, gate=gate)
    advisor.tick()
    advisor.reset()
    advisor.tick()
    assert len(fired) == 2  # cooldown cleared

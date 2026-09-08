"""Tests for system_signals (Phase 3 wiring)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.interruption_gate import (  # noqa: E402
    InterruptionGate, SignalKind,
)
from hgr.live_api.system_signals import (  # noqa: E402
    _any_process_matches, _is_windows, _SCREEN_SHARE_PROCS,
    DEFAULT_INTERVAL_SEC, prime_signals_safely, tick_all_signals,
)


# ---- process matcher --------------------------------------------------

def test_any_process_matches_finds_screen_share_process():
    procs = {"chrome.exe", "ms-teams.exe", "explorer.exe"}
    assert _any_process_matches(procs, _SCREEN_SHARE_PROCS) is True


def test_any_process_matches_clean_set():
    procs = {"chrome.exe", "code.exe"}
    assert _any_process_matches(procs, _SCREEN_SHARE_PROCS) is False


def test_any_process_matches_empty():
    assert _any_process_matches(set(), _SCREEN_SHARE_PROCS) is False


# ---- prime_signals_safely seeds defaults -------------------------------

def test_prime_signals_seeds_privacy_signals_when_probes_fail(monkeypatch):
    """When Win32 probes can't run (non-Windows OR Windows with broken
    probes), prime_signals_safely must seed conservative defaults so
    the gate's SEC-002 fail-closed doesn't block the very first
    interruption. We force the fallback by monkeypatching _is_windows."""
    monkeypatch.setattr("hgr.live_api.system_signals._is_windows",
                        lambda: False)
    g = InterruptionGate()
    prime_signals_safely(g)
    # After priming, all privacy-critical signals should be FRESH-False.
    state, val = g._signal_state(SignalKind.SCREEN_SHARING.value)
    assert state == "fresh"
    assert val is False
    state, val = g._signal_state(SignalKind.MIC_IN_USE.value)
    assert state == "fresh"
    assert val is False
    state, val = g._signal_state(SignalKind.CAMERA_IN_USE.value)
    assert state == "fresh"
    assert val is False


def test_prime_signals_allows_normal_interrupt(monkeypatch):
    """When the system has no screen-share / mic / camera in use AND
    Focus Assist is off, a NORMAL interruption should be allowed."""
    monkeypatch.setattr("hgr.live_api.system_signals._is_windows",
                        lambda: False)
    from hgr.live_api.interruption_gate import (
        GateConfig, InterruptSeverity)
    g = InterruptionGate(config=GateConfig(quiet_start_hour=0,
                                            quiet_end_hour=0))
    prime_signals_safely(g)
    d = g.can_interrupt(InterruptSeverity.NORMAL)
    assert d.allow is True


def test_prime_signals_on_real_host_does_not_crash():
    """Smoke: on the actual host (whatever it is), the prime call
    must not raise. If we're on Windows and a screen-share IS in
    progress, the gate may correctly return allow=False — that's the
    intended SEC-002 behavior, not a bug. We just verify no crash."""
    g = InterruptionGate()
    prime_signals_safely(g)
    # At minimum SOMETHING was set in the snapshot.
    snap = g.snapshot()
    assert isinstance(snap, dict)


# ---- tick_all_signals doesn't crash on non-Windows ---------------------

def test_tick_all_signals_runs_clean_on_any_platform():
    """Whatever platform tests run on, tick_all_signals must not raise.
    On non-Windows the probes all return None and the gate's signals
    just stay as they were."""
    g = InterruptionGate()
    # Should never throw.
    tick_all_signals(g)


def test_default_interval_is_sane():
    # Sentinel watcher with this interval shouldn't peg CPU.
    assert 1.0 <= DEFAULT_INTERVAL_SEC <= 60.0


# ---- register_with_sentinel idempotency --------------------------------

def test_register_with_sentinel_idempotent():
    from hgr.live_api.sentinel import Sentinel
    from hgr.live_api.system_signals import register_with_sentinel
    s = Sentinel()
    register_with_sentinel(s)
    register_with_sentinel(s)  # second call should not duplicate
    specs = s.watcher_specs()
    names = [w.name for w in specs]
    assert names.count("system_signals") == 1

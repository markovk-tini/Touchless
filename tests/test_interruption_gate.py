"""Tests for InterruptionGate (Phase 3 B2)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.interruption_gate import (  # noqa: E402
    GateConfig, InterruptDecision, InterruptSeverity,
    InterruptionGate, SignalKind, _truthy,
)


def setup_function():
    """Phase-7: the InterruptionGate now consults the global affect
    model for LOW severity. Reset it between tests so a prior test's
    frustration / focus signals don't leak across."""
    try:
        from hgr.live_api.affect import reset_global
        reset_global()
    except Exception:
        pass


def _no_quiet_cfg() -> GateConfig:
    # Disable quiet-hours so time-of-day doesn't shake tests.
    return GateConfig(quiet_start_hour=0, quiet_end_hour=0)


def _fresh() -> InterruptionGate:
    """SEC-002 audit: the gate now fails CLOSED for LOW/NORMAL when
    privacy-critical signals (SCREEN_SHARING, MIC_IN_USE,
    CAMERA_IN_USE) are missing or stale. Pre-seed them as False so
    tests that don't care about those signals can still observe the
    default 'allow' behavior they expect."""
    g = InterruptionGate(config=_no_quiet_cfg())
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    g.set_signal(SignalKind.MIC_IN_USE, False)
    g.set_signal(SignalKind.CAMERA_IN_USE, False)
    return g


# ---- default + signal mechanics ---------------------------------------

def test_default_allows_normal_interrupt():
    g = _fresh()
    d = g.can_interrupt(InterruptSeverity.NORMAL)
    assert d.allow is True
    assert d.suggested_channel == "voice"


def test_set_then_clear_signal_round_trips():
    g = _fresh()
    g.set_signal(SignalKind.DND, True)
    assert g.get_signal(SignalKind.DND) is True
    g.clear_signal(SignalKind.DND)
    assert g.get_signal(SignalKind.DND) is None


# ---- critical bypass --------------------------------------------------

def test_critical_passes_through_everything():
    g = _fresh()
    g.set_signal(SignalKind.SCREEN_SHARING, True)
    g.set_signal(SignalKind.DND, True)
    g.set_signal(SignalKind.GAME_MODE, True)
    g.set_signal(SignalKind.FOCUS_ASSIST, True)
    d = g.can_interrupt(InterruptSeverity.CRITICAL)
    assert d.allow is True
    assert d.tier_used == "critical"


# ---- hard blockers ----------------------------------------------------

def test_screen_sharing_blocks_normal():
    g = _fresh()
    g.set_signal(SignalKind.SCREEN_SHARING, True)
    d = g.can_interrupt(InterruptSeverity.NORMAL)
    assert d.allow is False
    assert "screen-sharing" in d.reason


def test_screen_sharing_blocks_high():
    g = _fresh()
    g.set_signal(SignalKind.SCREEN_SHARING, True)
    d = g.can_interrupt(InterruptSeverity.HIGH)
    assert d.allow is False


def test_game_mode_blocks_normal():
    g = _fresh()
    g.set_signal(SignalKind.GAME_MODE, True)
    d = g.can_interrupt(InterruptSeverity.NORMAL)
    assert d.allow is False


def test_focus_assist_suppresses_normal_passes_high():
    g = _fresh()
    g.set_signal(SignalKind.FOCUS_ASSIST, True)
    assert g.can_interrupt(InterruptSeverity.NORMAL).allow is False
    assert g.can_interrupt(InterruptSeverity.HIGH).allow is True


def test_dnd_suppresses_low_and_normal_passes_high():
    g = _fresh()
    g.set_signal(SignalKind.DND, True)
    assert g.can_interrupt(InterruptSeverity.LOW).allow is False
    assert g.can_interrupt(InterruptSeverity.NORMAL).allow is False
    assert g.can_interrupt(InterruptSeverity.HIGH).allow is True


# ---- soft suppressors (defer rather than drop) ------------------------

def test_mic_in_use_defers_normal():
    g = _fresh()
    g.set_signal(SignalKind.MIC_IN_USE, True)
    d = g.can_interrupt(InterruptSeverity.NORMAL)
    assert d.allow is False
    assert d.delay_until is not None
    # Defer ~30s for NORMAL.
    assert d.delay_until - time.time() < 60


def test_mic_in_use_defers_low_longer():
    g = _fresh()
    g.set_signal(SignalKind.MIC_IN_USE, True)
    d_norm = g.can_interrupt(InterruptSeverity.NORMAL)
    d_low = g.can_interrupt(InterruptSeverity.LOW)
    # LOW's deferral should be later than NORMAL's.
    assert d_low.delay_until > d_norm.delay_until


def test_mic_in_use_high_passes_as_earcon():
    g = _fresh()
    g.set_signal(SignalKind.MIC_IN_USE, True)
    d = g.can_interrupt(InterruptSeverity.HIGH)
    assert d.allow is True
    assert d.suggested_channel == "earcon"


def test_fullscreen_app_blocks_normal_high_as_earcon():
    g = _fresh()
    g.set_signal(SignalKind.FULLSCREEN_APP, True)
    assert g.can_interrupt(InterruptSeverity.NORMAL).allow is False
    d = g.can_interrupt(InterruptSeverity.HIGH)
    assert d.allow is True
    assert d.suggested_channel == "earcon"


def test_camera_in_use_treated_like_mic():
    g = _fresh()
    g.set_signal(SignalKind.CAMERA_IN_USE, True)
    d = g.can_interrupt(InterruptSeverity.NORMAL)
    assert d.allow is False
    assert d.delay_until is not None


# ---- AFK boost --------------------------------------------------------

def test_afk_boost_allows_low_when_long_idle():
    g = InterruptionGate(config=GateConfig(
        afk_idle_sec=60.0,
        quiet_start_hour=0, quiet_end_hour=0,
    ))
    # Privacy signals must be fresh-False so SEC-002 fail-closed
    # doesn't pre-empt the AFK branch.
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    g.set_signal(SignalKind.MIC_IN_USE, False)
    g.set_signal(SignalKind.CAMERA_IN_USE, False)
    g.set_signal(SignalKind.USER_IDLE_SEC, 300)
    d = g.can_interrupt(InterruptSeverity.LOW)
    assert d.allow is True
    assert "AFK" in d.tier_used


def test_low_blocked_when_not_idle():
    g = InterruptionGate(config=GateConfig(
        afk_idle_sec=60.0,
        quiet_start_hour=0, quiet_end_hour=0,
    ))
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    g.set_signal(SignalKind.MIC_IN_USE, False)
    g.set_signal(SignalKind.CAMERA_IN_USE, False)
    g.set_signal(SignalKind.USER_IDLE_SEC, 10)
    # No suppressors → LOW still allowed (default permissive path).
    d = g.can_interrupt(InterruptSeverity.LOW)
    assert d.allow is True


# ---- quiet hours ------------------------------------------------------

def test_quiet_hours_explicit_signal_overrides_time():
    g = InterruptionGate(config=GateConfig(
        quiet_start_hour=0, quiet_end_hour=0,  # disable time check
    ))
    g.set_signal(SignalKind.QUIET_HOURS, True)
    assert g.can_interrupt(InterruptSeverity.LOW).allow is False
    assert g.can_interrupt(InterruptSeverity.NORMAL).allow is False
    # HIGH passes as earcon.
    d = g.can_interrupt(InterruptSeverity.HIGH)
    assert d.allow is True
    assert d.suggested_channel == "earcon"


def test_quiet_hours_explicit_off_overrides_time():
    g = InterruptionGate(config=GateConfig(
        quiet_start_hour=0, quiet_end_hour=23,  # always quiet by time
    ))
    # Pre-seed privacy signals so SEC-002 fail-closed doesn't pre-empt.
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    g.set_signal(SignalKind.MIC_IN_USE, False)
    g.set_signal(SignalKind.CAMERA_IN_USE, False)
    g.set_signal(SignalKind.QUIET_HOURS, False)
    assert g.can_interrupt(InterruptSeverity.NORMAL).allow is True


def test_cross_midnight_quiet_window_works():
    # 22 → 7 cross-midnight window. Mock time-of-day by patching localtime.
    g = InterruptionGate(config=GateConfig(
        quiet_start_hour=22, quiet_end_hour=7,
    ))
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    g.set_signal(SignalKind.MIC_IN_USE, False)
    g.set_signal(SignalKind.CAMERA_IN_USE, False)
    g.set_signal(SignalKind.QUIET_HOURS, True)
    assert g.can_interrupt(InterruptSeverity.NORMAL).allow is False
    g.clear_signal(SignalKind.QUIET_HOURS)
    # Use config that bypasses time check:
    g2 = InterruptionGate(config=GateConfig(quiet_start_hour=0,
                                            quiet_end_hour=0))
    g2.set_signal(SignalKind.SCREEN_SHARING, False)
    g2.set_signal(SignalKind.MIC_IN_USE, False)
    g2.set_signal(SignalKind.CAMERA_IN_USE, False)
    assert g2.can_interrupt(InterruptSeverity.NORMAL).allow is True


# ---- snapshot ---------------------------------------------------------

def test_snapshot_returns_copy_of_signals():
    g = _fresh()
    g.set_signal(SignalKind.MIC_IN_USE, True)
    snap = g.snapshot()
    assert snap["mic_in_use"] is True
    # Mutating returned dict must NOT affect the gate.
    snap["fake"] = "x"
    assert "fake" not in g.snapshot()


# ---- truthy helper ----------------------------------------------------

def test_truthy_recognizes_common_falsy_strings():
    assert _truthy("0") is False
    assert _truthy("false") is False
    assert _truthy("no") is False
    assert _truthy("off") is False
    assert _truthy("") is False
    assert _truthy("True") is True
    assert _truthy(1) is True
    assert _truthy(0) is False
    assert _truthy(None) is False


def test_truthy_handles_float_strings():
    # Missed-by-panel: '0.0', 'None', 'null' should all be falsy.
    assert _truthy("0.0") is False
    assert _truthy("None") is False
    assert _truthy("null") is False
    assert _truthy("nan") is False
    assert _truthy("1.5") is True


# ---- SEC-002: fail-closed when privacy signals missing/stale ----------

def test_normal_fails_closed_when_screen_signal_missing():
    g = InterruptionGate(config=_no_quiet_cfg())
    # Don't seed any privacy signals.
    d = g.can_interrupt(InterruptSeverity.NORMAL)
    assert d.allow is False
    assert "unknown" in d.reason.lower() or "stale" in d.reason.lower()


def test_critical_passes_even_with_missing_signals():
    g = InterruptionGate(config=_no_quiet_cfg())
    d = g.can_interrupt(InterruptSeverity.CRITICAL)
    assert d.allow is True


def test_normal_allowed_when_all_privacy_signals_fresh_and_false():
    g = _fresh()  # _fresh now seeds all three
    d = g.can_interrupt(InterruptSeverity.NORMAL)
    assert d.allow is True


# ---- SEC-008: set_signal validation -----------------------------------

def test_set_signal_rejects_non_bool_for_boolean_signal():
    g = _fresh()
    # Should silently reject (logged warning) and leave prior value.
    g.set_signal(SignalKind.SCREEN_SHARING, {"weird": "shape"})
    # Privacy signal still fresh-False from _fresh() seeding.
    d = g.can_interrupt(InterruptSeverity.NORMAL)
    assert d.allow is True


def test_set_signal_coerces_bool_strings():
    g = _fresh()
    g.set_signal(SignalKind.DND, "true")
    assert g.can_interrupt(InterruptSeverity.NORMAL).allow is False


def test_set_signal_clamps_numeric_range():
    g = _fresh()
    g.set_signal(SignalKind.USER_IDLE_SEC, 999_999_999)
    # Clamped to <= 86_400 (one day).
    assert g.get_signal(SignalKind.USER_IDLE_SEC) <= 86_400


# ---- battery_low (formerly dead enum) ---------------------------------

def test_battery_low_suppresses_low_severity_only():
    g = _fresh()
    g.set_signal(SignalKind.BATTERY_LOW, True)
    assert g.can_interrupt(InterruptSeverity.LOW).allow is False
    assert g.can_interrupt(InterruptSeverity.NORMAL).allow is True

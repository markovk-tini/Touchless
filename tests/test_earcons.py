"""Tests for QuietMode + EarconPlayer (Phase 2 B4)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.earcons import (  # noqa: E402
    EarconKind, EarconPlayer, QuietMode, QuietReason,
    _python_synth, _EARCON_SPECS, _FORCE_THROUGH_QUIET,
)


# ---- QuietMode ---------------------------------------------------------

def test_default_quiet_mode_is_off():
    q = QuietMode()
    st = q.state()
    assert st.is_quiet is False
    assert st.speech_allowed is True
    assert st.earcons_allowed is True


def test_adding_reason_silences_speech():
    q = QuietMode()
    q.add_reason(QuietReason.SCREEN_SHARING)
    st = q.state()
    assert st.is_quiet is True
    assert st.speech_allowed is False
    # Default: earcons stay on even when speech is silenced.
    assert st.earcons_allowed is True


def test_adding_reason_can_silence_earcons():
    q = QuietMode()
    q.add_reason(QuietReason.IN_MEETING, silences_earcons=True)
    st = q.state()
    assert st.earcons_allowed is False


def test_clearing_only_one_of_multiple_reasons_stays_quiet():
    q = QuietMode()
    q.add_reason(QuietReason.SCREEN_SHARING)
    q.add_reason(QuietReason.IN_MEETING)
    q.clear_reason(QuietReason.SCREEN_SHARING)
    assert q.state().is_quiet is True
    q.clear_reason(QuietReason.IN_MEETING)
    assert q.state().is_quiet is False


def test_reset_clears_all_reasons():
    q = QuietMode()
    q.add_reason(QuietReason.DND)
    q.add_reason(QuietReason.FOCUS_ASSIST, silences_earcons=True)
    q.reset()
    st = q.state()
    assert st.is_quiet is False
    assert st.speech_allowed is True
    assert st.earcons_allowed is True


def test_subscribe_receives_state_changes():
    q = QuietMode()
    received = []

    def cb(state):
        received.append(state.is_quiet)

    unsub = q.subscribe(cb)
    q.add_reason(QuietReason.DND)
    q.clear_reason(QuietReason.DND)
    unsub()
    q.add_reason(QuietReason.DND)
    # After unsub, no more notifications.
    assert received == [True, False]


def test_subscriber_exception_does_not_break_others():
    q = QuietMode()
    good = []

    def bad(state):
        raise RuntimeError("kaboom")

    def good_cb(state):
        good.append(1)

    q.subscribe(bad)
    q.subscribe(good_cb)
    q.add_reason(QuietReason.DND)
    assert good == [1]


# ---- EarconPlayer ------------------------------------------------------

def test_earcon_specs_cover_all_kinds():
    for k in EarconKind:
        assert k in _EARCON_SPECS


def test_play_returns_false_when_sounddevice_missing():
    # In CI / test env, sounddevice may be installed but no audio
    # backend is present. The player must NOT crash; it should
    # just return False from play().
    q = QuietMode()
    p = EarconPlayer(quiet_mode=q)
    # Force the no-sd fast path so the test is deterministic.
    p._sd_failed = True
    assert p.play(EarconKind.DONE) is False


def test_quiet_mode_suppresses_done_earcon():
    q = QuietMode()
    q.add_reason(QuietReason.DND, silences_earcons=True)
    p = EarconPlayer(quiet_mode=q)
    p._sd_failed = True
    # Even if audio backend WERE present, the policy check returns False.
    assert p.play(EarconKind.DONE) is False


def test_force_through_quiet_earcons_still_attempt_play():
    q = QuietMode()
    q.add_reason(QuietReason.DND, silences_earcons=True)
    p = EarconPlayer(quiet_mode=q)
    p._sd_failed = True
    # _allowed should return True for NEEDS_CONFIRM even in quiet mode.
    # But play still returns False because no sd backend exists.
    # We verify the policy directly:
    assert p._allowed(EarconKind.NEEDS_CONFIRM) is True
    assert p._allowed(EarconKind.ERROR) is True
    assert p._allowed(EarconKind.DONE) is False


def test_force_through_set_matches_expected_kinds():
    assert EarconKind.NEEDS_CONFIRM in _FORCE_THROUGH_QUIET
    assert EarconKind.ERROR in _FORCE_THROUGH_QUIET
    assert EarconKind.DONE not in _FORCE_THROUGH_QUIET


# ---- _python_synth fallback -------------------------------------------

def test_python_synth_returns_samples_in_range():
    samples = _python_synth(freq=440, duration_ms=10,
                            db_atten=18, sr=24_000)
    assert len(samples) > 0
    assert all(-1.0 <= s <= 1.0 for s in samples)


def test_python_synth_zero_duration_returns_empty():
    assert _python_synth(freq=440, duration_ms=0,
                         db_atten=18, sr=24_000) == []


def test_python_synth_attenuation_reduces_amplitude():
    loud = _python_synth(freq=440, duration_ms=10,
                         db_atten=0, sr=24_000)
    quiet = _python_synth(freq=440, duration_ms=10,
                          db_atten=24, sr=24_000)
    assert max(abs(s) for s in loud) > max(abs(s) for s in quiet)

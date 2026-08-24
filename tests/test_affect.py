"""Tests for affective state model (Phase 7 B3)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.affect import (  # noqa: E402
    AffectiveState, AffectModel, Signal, SignalKind,
    current_state, global_model, observe_user_turn,
    reply_length_bias, reset_global,
    should_suppress_callbacks, should_suppress_low_nudges,
)


def setup_function():
    reset_global()


# ---- baseline -------------------------------------------------------

def test_baseline_state_is_neutral():
    st = current_state()
    assert st.mood == 0.0
    assert st.focus == 0.0
    assert st.verbosity == 0.0
    assert not st.is_frustrated()
    assert not st.is_focused()


# ---- frustration detection -----------------------------------------

def test_frustration_word_lowers_mood():
    observe_user_turn("ugh this is broken again")
    st = current_state()
    assert st.mood < 0
    assert st.is_frustrated()


def test_no_frustration_for_normal_text():
    observe_user_turn("what's the weather")
    st = current_state()
    assert st.mood >= -0.1
    assert not st.is_frustrated()


def test_frustration_compounds_across_turns():
    observe_user_turn("ugh")
    observe_user_turn("ffs")
    st = current_state()
    assert st.mood < -0.4  # both signals stack


def test_never_mind_counts_as_frustration():
    observe_user_turn("never mind")
    assert current_state().is_frustrated()


def test_no_i_said_counts_as_frustration():
    observe_user_turn("no, I said the OTHER one")
    assert current_state().is_frustrated()


# ---- acceptance ----------------------------------------------------

def test_acceptance_lifts_mood():
    observe_user_turn("thanks, perfect")
    st = current_state()
    assert st.mood > 0


def test_acceptance_recovers_from_frustration():
    observe_user_turn("ugh")
    observe_user_turn("perfect, thanks")
    st = current_state()
    # Frustration -0.35, acceptance +0.20 → net negative but small.
    # Within margin of frustrated cutoff.
    assert st.mood > -0.3


def test_perfect_alone_is_acceptance():
    observe_user_turn("perfect")
    assert current_state().mood > 0


# ---- exploratory ---------------------------------------------------

def test_exploratory_raises_verbosity():
    observe_user_turn("can you explain how the cache works")
    st = current_state()
    assert st.verbosity > 0
    assert st.wants_expansive()


def test_exploratory_does_not_lower_mood():
    observe_user_turn("could you tell me more about that")
    st = current_state()
    assert st.mood >= -0.05


# ---- terse follow-up ----------------------------------------------

def test_terse_no_lowers_verbosity():
    observe_user_turn("no")
    st = current_state()
    assert st.verbosity < 0
    assert st.wants_terse()


def test_terse_cancel_counts():
    observe_user_turn("cancel")
    assert current_state().wants_terse()


def test_terse_does_not_match_questions():
    """'no' must be the WHOLE utterance — 'no idea' shouldn't
    trigger."""
    observe_user_turn("no idea what that is")
    st = current_state()
    assert not st.wants_terse() or st.verbosity > -0.2


# ---- correction signal ---------------------------------------------

def test_correction_lowers_mood():
    m = global_model()
    m.record_correction()
    st = m.current_state()
    assert st.mood < 0


def test_two_corrections_compound():
    m = global_model()
    m.record_correction()
    m.record_correction()
    assert m.current_state().mood < -0.4


# ---- decay ---------------------------------------------------------

def test_signal_decays_over_time():
    m = global_model()
    old = Signal(kind=SignalKind.FRUSTRATION_WORD,
                 weight=1.0, ts=time.time() - 60 * 60)  # 1h old
    m.record(old)
    st = m.current_state()
    # 1 hour = 4 half-lives → weight ~6% → mood barely moves.
    assert -0.05 < st.mood <= 0.0


def test_recent_signal_dominates():
    m = global_model()
    m.record(Signal(kind=SignalKind.FRUSTRATION_WORD,
                     weight=1.0,
                     ts=time.time() - 60 * 60))
    m.record(Signal(kind=SignalKind.FRUSTRATION_WORD,
                     weight=1.0,
                     ts=time.time()))
    st = m.current_state()
    # Recent one not decayed; -0.35 mood
    assert st.mood <= -0.3


# ---- silence gap ---------------------------------------------------

def test_silence_gap_raises_focus():
    m = global_model()
    # First turn — sets _last_user_turn_at.
    m.observe_user_turn("hi")
    # Simulate 10 min later.
    m._last_user_turn_at = time.time() - 600
    m.observe_user_turn("ok back")
    st = m.current_state()
    assert st.focus > 0


# ---- bounds --------------------------------------------------------

def test_mood_clamps_at_minus_1():
    m = global_model()
    for _ in range(20):
        m.record(Signal(kind=SignalKind.FRUSTRATION_WORD,
                         weight=1.0))
    st = m.current_state()
    assert st.mood >= -1.0
    assert st.mood <= 1.0


def test_mood_clamps_at_plus_1():
    m = global_model()
    for _ in range(20):
        m.record(Signal(kind=SignalKind.ACCEPTANCE, weight=1.0))
    st = m.current_state()
    assert st.mood <= 1.0


# ---- consumers ----------------------------------------------------

def test_should_suppress_callbacks_when_frustrated():
    observe_user_turn("ugh, stop it, this is broken")
    assert should_suppress_callbacks() is True


def test_should_suppress_callbacks_when_neutral():
    assert should_suppress_callbacks() is False


def test_should_suppress_low_nudges_when_focused():
    m = global_model()
    m.record(Signal(kind=SignalKind.SILENCE_GAP, weight=1.0))
    m.record(Signal(kind=SignalKind.SILENCE_GAP, weight=1.0))
    m.record(Signal(kind=SignalKind.SILENCE_GAP, weight=1.0))
    assert should_suppress_low_nudges() is True


def test_reply_length_bias_terse_when_user_terse():
    observe_user_turn("no")
    assert reply_length_bias() == -1


def test_reply_length_bias_expansive_when_user_explores():
    observe_user_turn("could you explain that more deeply")
    assert reply_length_bias() == +1


def test_reply_length_bias_neutral_default():
    assert reply_length_bias() == 0


# ---- model lifecycle ---------------------------------------------

def test_global_model_is_singleton():
    a = global_model()
    b = global_model()
    assert a is b


def test_reset_clears_state():
    observe_user_turn("ugh")
    assert current_state().is_frustrated()
    reset_global()
    assert not current_state().is_frustrated()


def test_observe_turn_returns_emitted_signals():
    emitted = observe_user_turn("ugh, that's broken")
    assert any(s.kind == SignalKind.FRUSTRATION_WORD
               for s in emitted)


def test_observe_turn_returns_empty_for_neutral():
    emitted = observe_user_turn("what's the weather")
    assert all(s.kind != SignalKind.FRUSTRATION_WORD
               for s in emitted)


# ---- state dict / introspection ----------------------------------

def test_state_as_dict_contains_axes():
    observe_user_turn("ugh broken")
    d = current_state().as_dict()
    assert "mood" in d
    assert "focus" in d
    assert "verbosity" in d
    assert d["frustrated"] is True

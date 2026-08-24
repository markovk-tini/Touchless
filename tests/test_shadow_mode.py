"""Tests for ShadowMode (Phase 3 B3)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.shadow_mode import (  # noqa: E402
    Divergence, ShadowMode, ShadowSummary, _decisions_equal,
)


# ---- equality helper --------------------------------------------------

def test_decisions_equal_primitives():
    assert _decisions_equal(1, 1) is True
    assert _decisions_equal("a", "a") is True
    assert _decisions_equal(1, 2) is False


def test_decisions_equal_objects_compare_by_dict():
    class A:
        def __init__(self, x):
            self.x = x

    a, b = A(1), A(1)
    c = A(2)
    assert _decisions_equal(a, b) is True
    assert _decisions_equal(a, c) is False


# ---- baseline + shadow round trip -------------------------------------

def test_matching_pair_returns_no_divergence():
    s = ShadowMode("test")
    s.record_baseline(turn_id="t1", user_text="hi", decision="A")
    div = s.record_shadow(turn_id="t1", decision="A")
    assert div is None
    assert s.summary().divergences == 0
    assert s.summary().pairs_compared == 1


def test_diverging_pair_returns_divergence():
    s = ShadowMode("test")
    s.record_baseline(turn_id="t2", user_text="hi", decision="A")
    div = s.record_shadow(turn_id="t2", decision="B")
    assert isinstance(div, Divergence)
    assert div.baseline == "A"
    assert div.shadow == "B"
    assert s.summary().divergences == 1


def test_shadow_without_baseline_counted_as_skipped():
    s = ShadowMode("test")
    div = s.record_shadow(turn_id="no-baseline", decision="X")
    assert div is None
    assert s.summary().shadow_skipped == 1


def test_shadow_error_counted_as_divergence_and_error():
    s = ShadowMode("test")
    s.record_baseline(turn_id="t3", user_text="hi", decision="ok")
    div = s.record_shadow(turn_id="t3", decision=None,
                          error="planner crashed")
    assert div is not None
    assert "ERROR" in str(div.shadow)
    summary = s.summary()
    assert summary.divergences == 1
    assert summary.shadow_errors == 1


# ---- sampling ---------------------------------------------------------

def test_should_run_shadow_fully_on_when_fraction_1():
    s = ShadowMode("test", shadow_fraction=1.0)
    assert all(s.should_run_shadow() for _ in range(10))


def test_should_run_shadow_fully_off_when_fraction_0():
    s = ShadowMode("test", shadow_fraction=0.0)
    assert not any(s.should_run_shadow() for _ in range(10))


def test_should_run_shadow_samples_when_fractional():
    s = ShadowMode("test", shadow_fraction=0.25)
    decisions = [s.should_run_shadow() for _ in range(20)]
    # 25% sampling → ~5 runs in 20 calls.
    on = sum(1 for d in decisions if d)
    assert 4 <= on <= 6


def test_set_fraction_clamps_input():
    s = ShadowMode("test")
    s.set_fraction(2.0)
    assert s.fraction() == 1.0
    s.set_fraction(-1.0)
    assert s.fraction() == 0.0


# ---- summary + divergence_rate ----------------------------------------

def test_divergence_rate_overall_and_windowed():
    s = ShadowMode("test")
    for i in range(5):
        s.record_baseline(turn_id=f"t{i}", user_text="x", decision="A")
        s.record_shadow(turn_id=f"t{i}", decision="B")
    overall = s.divergence_rate()
    assert overall == 1.0
    # Window covering everything still returns 1.0.
    windowed = s.divergence_rate(window_sec=1000)
    assert windowed == 1.0


def test_divergence_rate_zero_when_no_pairs():
    s = ShadowMode("test")
    assert s.divergence_rate() == 0.0


def test_recent_divergences_capped_by_limit():
    s = ShadowMode("test")
    for i in range(50):
        s.record_baseline(turn_id=f"t{i}", user_text="x", decision="A")
        s.record_shadow(turn_id=f"t{i}", decision="B")
    recent = s.recent_divergences(limit=10)
    assert len(recent) == 10
    # Should be the LAST 10.
    ids = [d.turn_id for d in recent]
    assert ids[-1] == "t49"


def test_ring_buffer_drops_oldest():
    s = ShadowMode("test", ring_size=5)
    for i in range(10):
        s.record_baseline(turn_id=f"t{i}", user_text="x", decision="A")
        s.record_shadow(turn_id=f"t{i}", decision="B")
    assert len(s.recent_divergences(limit=100)) == 5


def test_reset_clears_all_state():
    s = ShadowMode("test")
    s.record_baseline(turn_id="t", user_text="x", decision="A")
    s.record_shadow(turn_id="t", decision="B")
    s.reset()
    assert s.summary().pairs_compared == 0
    assert s.summary().divergences == 0
    assert s.recent_divergences() == []

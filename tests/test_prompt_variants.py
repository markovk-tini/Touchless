"""Tests for prompt_variants (Phase 5 B3)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.incognito as inc  # noqa: E402
from hgr.live_api.prompt_variants import (  # noqa: E402
    EXPLORE_FLOOR_PCT, PRIOR_ALPHA, PRIOR_BETA,
    PromptVariantTracker, Variant, WARMUP_SAMPLE_THRESHOLD,
)


def setup_function():
    inc.set_incognito(False)


def _fresh_tracker() -> PromptVariantTracker:
    d = Path(tempfile.mkdtemp())
    return PromptVariantTracker(db_path=d / "v.db")


# ---- registration -----------------------------------------------------

def test_register_loads_persisted_counters():
    d = Path(tempfile.mkdtemp())
    t1 = PromptVariantTracker(db_path=d / "v.db")
    t1.register("planner", [Variant(id="v1", text="A"),
                            Variant(id="v2", text="B")])
    t1.record_outcome("planner", "v1", success=True)
    t1.record_outcome("planner", "v1", success=True)
    t1.close()
    # Second tracker over the same DB should see those counters.
    t2 = PromptVariantTracker(db_path=d / "v.db")
    t2.register("planner", [Variant(id="v1", text="A"),
                            Variant(id="v2", text="B")])
    leaderboard = t2.leaderboard("planner")
    v1 = next(r for r in leaderboard if r["id"] == "v1")
    assert v1["successes"] == 2


def test_pick_returns_none_for_unregistered_group():
    t = _fresh_tracker()
    assert t.pick("nonexistent") is None


# ---- warm-up routing --------------------------------------------------

def test_warmup_routes_uniformly_across_variants():
    t = _fresh_tracker()
    t.register("g", [Variant(id="a", text="A"),
                     Variant(id="b", text="B")])
    picks = [t.pick("g").id for _ in range(WARMUP_SAMPLE_THRESHOLD * 2)]
    # Both variants should be picked during warm-up.
    assert "a" in picks
    assert "b" in picks


def test_warmup_clears_once_threshold_met():
    """Once every variant has WARMUP_SAMPLE_THRESHOLD samples,
    the bandit transitions to exploit/explore. Verify by feeding
    enough outcomes that warm-up exits, then check picks bias
    toward the leader."""
    t = _fresh_tracker()
    t.register("g", [Variant(id="winner", text="A"),
                     Variant(id="loser", text="B")])
    # Feed outcomes to clear warm-up + establish a clear winner.
    for _ in range(WARMUP_SAMPLE_THRESHOLD + 5):
        t.record_outcome("g", "winner", success=True)
        t.record_outcome("g", "loser", success=False)
    # Post-warmup: leader should dominate but the explore floor
    # still ensures the loser shows up roughly EXPLORE_FLOOR_PCT
    # of the time.
    picks = [t.pick("g").id for _ in range(100)]
    winner_count = picks.count("winner")
    loser_count = picks.count("loser")
    # Winner gets ~80% (100 - EXPLORE_FLOOR_PCT).
    assert winner_count >= 60
    # Loser still gets at least some traffic (explore floor).
    assert loser_count >= 5


# ---- record_outcome --------------------------------------------------

def test_record_outcome_increments_counters():
    t = _fresh_tracker()
    t.register("g", [Variant(id="v", text="A")])
    t.record_outcome("g", "v", success=True)
    t.record_outcome("g", "v", success=True)
    t.record_outcome("g", "v", success=False)
    row = t.leaderboard("g")[0]
    assert row["successes"] == 2
    assert row["failures"] == 1


def test_record_outcome_honors_incognito():
    t = _fresh_tracker()
    t.register("g", [Variant(id="v", text="A")])
    inc.set_incognito(True)
    try:
        t.record_outcome("g", "v", success=True)
    finally:
        inc.set_incognito(False)
    row = t.leaderboard("g")[0]
    assert row["successes"] == 0


def test_record_outcome_unknown_variant_is_noop():
    t = _fresh_tracker()
    t.register("g", [Variant(id="v", text="A")])
    # Recording for an unknown variant id shouldn't raise.
    t.record_outcome("g", "ghost", success=True)
    # Existing variant unaffected.
    row = next(r for r in t.leaderboard("g") if r["id"] == "v")
    assert row["successes"] == 0


# ---- smoothed score --------------------------------------------------

def test_smoothed_score_uses_prior_when_no_samples():
    v = Variant(id="x", text="...")
    # With prior (alpha=7, beta=3) the baseline is 0.7.
    assert abs(v.smoothed_success_rate() - 0.7) < 0.001


def test_smoothed_score_responds_to_data():
    v = Variant(id="x", text="...",
                successes=20, failures=0)
    # Strong evidence for success should push score above the prior.
    assert v.smoothed_success_rate() > 0.85


def test_smoothed_score_responds_to_failure():
    v = Variant(id="x", text="...",
                successes=0, failures=20)
    # Strong evidence for failure should pull score below the prior.
    assert v.smoothed_success_rate() < 0.3


# ---- leaderboard ------------------------------------------------------

def test_leaderboard_sorted_by_smoothed_score():
    t = _fresh_tracker()
    t.register("g", [
        Variant(id="winner", text="A"),
        Variant(id="loser", text="B"),
        Variant(id="middle", text="C"),
    ])
    for _ in range(50):
        t.record_outcome("g", "winner", success=True)
    for _ in range(50):
        t.record_outcome("g", "loser", success=False)
    for _ in range(50):
        t.record_outcome("g", "middle", success=True)
    for _ in range(50):
        t.record_outcome("g", "middle", success=False)
    rows = t.leaderboard("g")
    # winner > middle > loser.
    assert rows[0]["id"] == "winner"
    assert rows[1]["id"] == "middle"
    assert rows[2]["id"] == "loser"


def test_leaderboard_includes_notes_and_sample_counts():
    t = _fresh_tracker()
    t.register("g", [
        Variant(id="v1", text="A", notes="terse"),
        Variant(id="v2", text="B", notes="verbose"),
    ])
    t.record_outcome("g", "v1", success=True)
    rows = t.leaderboard("g")
    v1 = next(r for r in rows if r["id"] == "v1")
    assert v1["notes"] == "terse"
    assert v1["total_samples"] == 1


# ---- wipe -------------------------------------------------------------

def test_wipe_clears_persisted_and_in_memory():
    t = _fresh_tracker()
    t.register("g", [Variant(id="v", text="A")])
    for _ in range(5):
        t.record_outcome("g", "v", success=True)
    n = t.wipe()
    assert n >= 1
    rows = t.leaderboard("g")
    assert rows[0]["successes"] == 0
    assert rows[0]["failures"] == 0

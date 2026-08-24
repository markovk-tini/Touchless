"""Tests for speculative_planner (Phase 8 B2)."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.speculative_planner import (  # noqa: E402
    SpeculativeCache, SpeculativeEntry, SpeculativePlanner,
    TranscriptStabilizer, _token_overlap,
)


# ---- token_overlap helper ------------------------------------------

def test_token_overlap_identical():
    assert _token_overlap("hello world", "hello world") == 1.0


def test_token_overlap_disjoint():
    assert _token_overlap("hello world", "foo bar") == 0.0


def test_token_overlap_partial():
    score = _token_overlap("send the email to dani",
                            "send email to dani about q3")
    assert 0.4 < score < 0.9


def test_token_overlap_handles_empty():
    assert _token_overlap("", "") == 1.0
    assert _token_overlap("", "hi") == 0.0


def test_token_overlap_case_insensitive():
    assert _token_overlap("Hello", "hello") == 1.0


# ---- stabilizer ----------------------------------------------------

def test_stabilizer_holds_until_window():
    s = TranscriptStabilizer(stability_window_ms=100,
                              stability_hold_ms=300)
    # Several rapid changes — never declares stable.
    s.observe("send")
    s.observe("send email")
    s.observe("send email to dani")
    assert s._declared_partial is None


def test_stabilizer_fires_after_hold():
    s = TranscriptStabilizer(stability_window_ms=50,
                              stability_hold_ms=80)
    s.observe("send the email to dani")
    time.sleep(0.12)
    stable = s.observe("send the email to dani")
    assert stable == "send the email to dani"


def test_stabilizer_fires_once_per_stable_run():
    s = TranscriptStabilizer(stability_window_ms=50,
                              stability_hold_ms=80)
    s.observe("send the email to dani")
    time.sleep(0.12)
    first = s.observe("send the email to dani")
    second = s.observe("send the email to dani")
    assert first is not None
    assert second is None


def test_stabilizer_resets_on_change():
    s = TranscriptStabilizer(stability_window_ms=50,
                              stability_hold_ms=80)
    s.observe("send the email")
    time.sleep(0.1)
    s.observe("send the email to dani")
    # Reset — needs new hold window.
    assert s._declared_partial is None


def test_stabilizer_period_short_circuits():
    s = TranscriptStabilizer(stability_window_ms=50,
                              stability_hold_ms=500)
    s.observe("send the email to dani.")
    time.sleep(0.07)
    stable = s.observe("send the email to dani.")
    assert stable == "send the email to dani."


def test_stabilizer_requires_min_words():
    s = TranscriptStabilizer(stability_window_ms=10,
                              stability_hold_ms=20)
    s.observe("hi")
    time.sleep(0.05)
    stable = s.observe("hi")
    assert stable is None


def test_stabilizer_skips_empty():
    s = TranscriptStabilizer()
    assert s.observe("") is None
    assert s.observe(None) is None


def test_stabilizer_reset_clears():
    s = TranscriptStabilizer(stability_window_ms=10,
                              stability_hold_ms=20)
    s.observe("send the email to dani")
    time.sleep(0.05)
    s.observe("send the email to dani")
    s.reset()
    assert s._declared_partial is None


# ---- cache ----------------------------------------------------------

def test_cache_put_and_find():
    c = SpeculativeCache()
    c.put("send the email to dani", {"tool": "email_send"})
    entry = c.find("send the email to dani")
    assert entry is not None
    assert entry.plan == {"tool": "email_send"}
    assert entry.consumed is True


def test_cache_marks_consumed():
    c = SpeculativeCache()
    c.put("send", {"x": 1})
    e1 = c.find("send the email")
    # find always tries to consume — second call shouldn't match
    # the already-consumed entry.
    e2 = c.find("send the email")
    assert e1 is None or e2 is None


def test_cache_min_overlap_threshold():
    c = SpeculativeCache()
    c.put("send the email to dani", {"tool": "email_send"})
    # Final has very different words.
    entry = c.find("what is the weather like today")
    assert entry is None


def test_cache_finds_best_overlap():
    c = SpeculativeCache()
    c.put("send to dani", {"v": "a"})
    c.put("send the email to dani about q3", {"v": "b"})
    entry = c.find("send the email to dani about q3 schedule")
    assert entry is not None
    assert entry.plan["v"] == "b"


def test_cache_age_drops_old():
    c = SpeculativeCache()
    c.put("send the email to dani", {"x": 1})
    e = list(c._entries.values())[0]
    e.fired_at = time.time() - 1000
    entry = c.find("send the email to dani")
    assert entry is None


def test_cache_evicts_oldest_when_full():
    c = SpeculativeCache()
    c._MAX_ENTRIES = 3
    c.put("a one two three", {"v": 1})
    c.put("b one two three", {"v": 2})
    c.put("c one two three", {"v": 3})
    c.put("d one two three", {"v": 4})
    assert len(c._entries) == 3
    # 'a' should be gone.
    assert c.find("a one two three") is None


def test_cache_clear():
    c = SpeculativeCache()
    c.put("hi there friend", {"x": 1})
    c.clear()
    assert c.find("hi there friend") is None


# ---- speculative planner ----------------------------------------------

def test_planner_does_not_fire_on_unstable():
    planned = []
    p = SpeculativePlanner(
        plan_fn=lambda t: planned.append(t) or {"plan": t},
        stabilizer=TranscriptStabilizer(
            stability_window_ms=200, stability_hold_ms=500))
    p.on_partial("send")
    p.on_partial("send email")
    p.on_partial("send the email to dani")
    # No time for hold to elapse.
    assert planned == []


def test_planner_fires_when_stable():
    planned = []
    p = SpeculativePlanner(
        plan_fn=lambda t: (planned.append(t), {"plan": t})[1],
        stabilizer=TranscriptStabilizer(
            stability_window_ms=20, stability_hold_ms=40))
    p.on_partial("send the email to dani")
    time.sleep(0.08)
    p.on_partial("send the email to dani")
    # Wait for the background plan thread.
    deadline = time.time() + 2.0
    while time.time() < deadline and not planned:
        time.sleep(0.02)
    assert planned == ["send the email to dani"]


def test_planner_hit_when_final_matches():
    p = SpeculativePlanner(
        plan_fn=lambda t: {"plan": t},
        stabilizer=TranscriptStabilizer(
            stability_window_ms=20, stability_hold_ms=40))
    p.on_partial("send the email to dani")
    time.sleep(0.08)
    p.on_partial("send the email to dani")
    deadline = time.time() + 2.0
    while time.time() < deadline and (
            "send the email to dani" not in
            {e.partial for e in p._cache._entries.values()}):
        time.sleep(0.02)
    plan = p.find_plan("send the email to dani about q3")
    assert plan is not None
    assert plan["plan"] == "send the email to dani"
    assert p.stats()["hit"] == 1


def test_planner_miss_when_final_diverges():
    p = SpeculativePlanner(
        plan_fn=lambda t: {"plan": t},
        stabilizer=TranscriptStabilizer(
            stability_window_ms=20, stability_hold_ms=40))
    p.on_partial("send the email to dani")
    time.sleep(0.08)
    p.on_partial("send the email to dani")
    deadline = time.time() + 2.0
    while time.time() < deadline and not p._cache._entries:
        time.sleep(0.02)
    plan = p.find_plan("what is the weather like today")
    assert plan is None
    assert p.stats()["miss"] == 1


def test_planner_skips_duplicate_inflight():
    """Two stable hits for the same partial shouldn't fire the
    planner twice."""
    call_count = [0]
    def slow_plan(t):
        call_count[0] += 1
        time.sleep(0.2)
        return {"plan": t}
    p = SpeculativePlanner(
        plan_fn=slow_plan,
        stabilizer=TranscriptStabilizer(
            stability_window_ms=10, stability_hold_ms=20))
    # First stable call.
    p.on_partial("send the email to dani")
    time.sleep(0.05)
    p.on_partial("send the email to dani")
    # Second stable call before first completes.
    p.on_partial("send the email to dani")
    time.sleep(0.4)
    assert call_count[0] == 1


def test_planner_reset_clears_all():
    p = SpeculativePlanner(plan_fn=lambda t: {"plan": t})
    p._cache.put("x one two three", {"a": 1})
    p.reset()
    assert p._cache.find("x one two three") is None
    assert p.stats() == {"fired": 0, "hit": 0, "miss": 0,
                          "discarded": 0}


def test_planner_exception_in_plan_fn_is_swallowed():
    """Speculation must NEVER bubble exceptions — it's optional."""
    def bad_plan(t):
        raise RuntimeError("nope")
    p = SpeculativePlanner(
        plan_fn=bad_plan,
        stabilizer=TranscriptStabilizer(
            stability_window_ms=10, stability_hold_ms=20))
    p.on_partial("send the email to dani")
    time.sleep(0.05)
    # Should not raise.
    p.on_partial("send the email to dani")
    time.sleep(0.2)
    # Cache empty — planner failed silently.
    assert not p._cache._entries

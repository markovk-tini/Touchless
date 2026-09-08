"""Tests for ChainOfThoughtLayer (Phase 2 B6)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from hgr.live_api.cot_layer import (  # noqa: E402
    ChainOfThoughtLayer, Decision, DecisionStage, TurnTrail,
    MAX_TURN_PAYLOAD_BYTES,
)
import hgr.live_api.incognito as incognito  # noqa: E402


def _fresh() -> ChainOfThoughtLayer:
    d = Path(tempfile.mkdtemp())
    return ChainOfThoughtLayer(db_path=d / "cot.db")


# ---- TurnTrail helpers -------------------------------------------------

def test_turn_trail_add_decision_appends():
    trail = TurnTrail(turn_id="t1", user_text="hi")
    trail.add_decision(DecisionStage.CLASSIFY, "weather_get",
                       why="matched 'weather' pattern")
    assert len(trail.decisions) == 1
    assert trail.decisions[0].stage == "classify"
    assert trail.decisions[0].choice == "weather_get"


def test_turn_trail_add_tool_ref():
    trail = TurnTrail(turn_id="t1", user_text="x")
    trail.add_tool_ref("inv-abc")
    trail.add_tool_ref("inv-def")
    trail.add_tool_ref("")  # empty ref skipped
    assert trail.tool_call_refs == ["inv-abc", "inv-def"]


def test_turn_trail_to_json_round_trips():
    trail = TurnTrail(turn_id="t1", user_text="hello",
                      inferred_goal="say hi")
    trail.add_decision(DecisionStage.PLAN, "single_step",
                       alternatives=["multi_step"])
    payload = trail.to_json()
    import json as _json
    decoded = _json.loads(payload)
    assert decoded["turn_id"] == "t1"
    assert decoded["user_text"] == "hello"
    assert decoded["decisions"][0]["alternatives"] == ["multi_step"]


# ---- layer round-trip --------------------------------------------------

def test_start_then_finalize_persists():
    incognito.set_incognito(False)
    cot = _fresh()
    trail = cot.start_turn("what's the weather")
    trail.inferred_goal = "weather lookup"
    trail.add_decision(DecisionStage.CLASSIFY, "weather_get")
    assert cot.finalize(trail, final_message="It's sunny.") is True
    fetched = cot.explain_turn(trail.turn_id)
    assert fetched is not None
    assert fetched["user_text"] == "what's the weather"
    assert fetched["final_message"] == "It's sunny."


def test_recent_turns_returns_newest_first():
    incognito.set_incognito(False)
    cot = _fresh()
    t1 = cot.start_turn("first turn")
    cot.finalize(t1, final_message="ok 1")
    t2 = cot.start_turn("second turn")
    cot.finalize(t2, final_message="ok 2")
    recent = cot.recent_turns(limit=2)
    assert recent[0]["user_text"] == "second turn"
    assert recent[1]["user_text"] == "first turn"


def test_last_turn_returns_most_recent():
    incognito.set_incognito(False)
    cot = _fresh()
    cot.finalize(cot.start_turn("a"), final_message="A")
    cot.finalize(cot.start_turn("b"), final_message="B")
    last = cot.last_turn()
    assert last is not None
    assert last["user_text"] == "b"


def test_explain_unknown_turn_returns_none():
    cot = _fresh()
    assert cot.explain_turn("nonexistent") is None


# ---- incognito behavior ------------------------------------------------

def test_finalize_returns_false_when_incognito():
    cot = _fresh()
    incognito.set_incognito(True)
    try:
        trail = cot.start_turn("private question")
        assert cot.finalize(trail, final_message="private answer") is False
        # And nothing should have been written.
        assert cot.recent_turns() == []
    finally:
        incognito.set_incognito(False)


# ---- payload cap -------------------------------------------------------

def test_oversized_payload_gets_clamped():
    incognito.set_incognito(False)
    cot = _fresh()
    trail = cot.start_turn("x" * 100)
    for i in range(2000):
        trail.add_decision(DecisionStage.PLAN, "choice-" + str(i),
                           why="reason " * 50,
                           alternatives=["alt-" + str(i)] * 20)
    assert cot.finalize(trail,
                        final_message="y" * 5000) is True
    fetched = cot.explain_turn(trail.turn_id)
    # The clamp removes alternatives + truncates final_message.
    assert fetched is not None
    # Final message should be ≤200 chars after clamp.
    assert len(fetched["final_message"]) <= 200


# ---- wipe -------------------------------------------------------------

def test_wipe_clears_table():
    incognito.set_incognito(False)
    cot = _fresh()
    cot.finalize(cot.start_turn("a"), final_message="A")
    cot.finalize(cot.start_turn("b"), final_message="B")
    n = cot.wipe()
    assert n == 2
    assert cot.recent_turns() == []
    assert cot.last_turn() is None


# ---- error path -------------------------------------------------------

def test_finalize_records_error_text():
    incognito.set_incognito(False)
    cot = _fresh()
    trail = cot.start_turn("buggy turn")
    assert cot.finalize(trail, final_message="oops",
                        error="planner crashed") is True
    fetched = cot.explain_turn(trail.turn_id)
    assert fetched["error"] == "planner crashed"

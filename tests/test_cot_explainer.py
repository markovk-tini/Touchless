"""Tests for cot_explainer (Phase 3 polish)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.cot_layer as cot_layer_mod  # noqa: E402
from hgr.live_api.cot_explainer import (  # noqa: E402
    explain_by_turn_id, explain_last_turn, _render,
    looks_like_explain_request,
)
from hgr.live_api.cot_layer import (  # noqa: E402
    ChainOfThoughtLayer, DecisionStage,
)
import hgr.live_api.incognito as inc  # noqa: E402


# ---- intent classifier -------------------------------------------------

def test_looks_like_explain_recognizes_canonical_phrases():
    for q in (
        "why did you do that?",
        "Why did you pick gpt-4?",
        "How did you decide to use Notion?",
        "Explain that last call",
        "What were you thinking?",
        "Walk me through that reasoning",
        "Show me your work",
    ):
        assert looks_like_explain_request(q), q


def test_looks_like_explain_ignores_unrelated():
    for q in (
        "why is the sky blue?",
        "what's the weather?",
        "explain quantum mechanics",  # not "explain that last"
        "",
        "tell me a joke",
    ):
        assert not looks_like_explain_request(q), q


def test_looks_like_explain_caps_long_input():
    huge = "why did you " + ("x" * 300)
    assert looks_like_explain_request(huge) is False


# ---- explanation rendering --------------------------------------------

def _make_layer():
    d = Path(tempfile.mkdtemp())
    return ChainOfThoughtLayer(db_path=d / "cot.db")


def _seed_turn(layer, *, user_text, decisions, final_message,
               turn_id="seed-1"):
    inc.set_incognito(False)
    trail = layer.start_turn(user_text, turn_id=turn_id)
    for stage, choice, why in decisions:
        trail.add_decision(stage, choice, why=why)
    layer.finalize(trail, final_message=final_message)


def test_explain_last_turn_renders_trail(monkeypatch):
    layer = _make_layer()
    monkeypatch.setattr(cot_layer_mod, "_layer", layer)
    _seed_turn(layer,
               user_text="send Dani the Q3 report",
               decisions=[
                   (DecisionStage.PLAN, "single_step",
                    "single send via gmail_send"),
                   (DecisionStage.REVISE, "retry_once",
                    "auth_revoked → retry"),
               ],
               final_message="Sent to dani@x.com (via Gmail).")
    out = explain_last_turn()
    assert "send Dani the Q3 report" in out
    assert "planned" in out
    assert "revised the plan" in out
    assert "Sent to dani" in out


def test_explain_returns_empty_when_no_trail(monkeypatch):
    layer = _make_layer()
    monkeypatch.setattr(cot_layer_mod, "_layer", layer)
    assert explain_last_turn() == ""


def test_explain_by_turn_id_finds_specific_trail(monkeypatch):
    layer = _make_layer()
    monkeypatch.setattr(cot_layer_mod, "_layer", layer)
    _seed_turn(layer,
               user_text="check the weather",
               decisions=[(DecisionStage.CLASSIFY, "weather_get",
                           "matched 'weather'")],
               final_message="Sunny.",
               turn_id="t-weather")
    out = explain_by_turn_id("t-weather")
    assert "check the weather" in out
    assert "classified" in out


def test_explain_by_turn_id_unknown_returns_empty(monkeypatch):
    layer = _make_layer()
    monkeypatch.setattr(cot_layer_mod, "_layer", layer)
    assert explain_by_turn_id("nope") == ""
    assert explain_by_turn_id("") == ""


def test_render_caps_long_decision_chain():
    trail = {
        "user_text": "x",
        "final_message": "done",
        "decisions": [
            {"stage": "plan", "choice": f"step-{i}", "why": "loop"}
            for i in range(20)
        ],
    }
    out = _render(trail)
    assert "and 12 more steps" in out  # 20 - 8 shown


def test_render_handles_no_decisions():
    trail = {"user_text": "hi", "final_message": "hello",
             "decisions": []}
    out = _render(trail)
    assert "no multi-step plan" in out


def test_explain_includes_tool_refs(monkeypatch):
    """The explainer should mention WHAT tools actually ran, not
    just the planning decisions."""
    layer = _make_layer()
    monkeypatch.setattr(cot_layer_mod, "_layer", layer)
    inc.set_incognito(False)
    trail = layer.start_turn("send Dani the report",
                              turn_id="t-tools")
    trail.add_decision(DecisionStage.PLAN, "multi_step",
                       why="contacts_search then gmail_send")
    trail.add_tool_ref("contacts_search")
    trail.add_tool_ref("gmail_send")
    layer.finalize(trail, final_message="Sent to dani@x.")
    out = explain_by_turn_id("t-tools")
    assert "Tools called:" in out
    assert "contacts_search" in out
    assert "gmail_send" in out


def test_explain_skips_incognito_writes(monkeypatch):
    # Incognito turns are not persisted, so explain finds nothing.
    layer = _make_layer()
    monkeypatch.setattr(cot_layer_mod, "_layer", layer)
    inc.set_incognito(True)
    try:
        trail = layer.start_turn("private")
        layer.finalize(trail, final_message="secret")
    finally:
        inc.set_incognito(False)
    assert explain_last_turn() == ""

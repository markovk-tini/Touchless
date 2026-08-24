"""Tests for SelfCritique (Phase 2)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.self_critique import (  # noqa: E402
    SelfCritique, _looks_multi_action,
    _extract_recipient_from_goal, _extract_channel_from_goal,
)
from hgr.live_api.planner.plan import Plan, Step, StepResult  # noqa: E402


def _ok(step_id: int, tool: str, output=None) -> StepResult:
    return StepResult(step_id=step_id, tool=tool, status="ok",
                      output=output or {})


# ---- happy paths --------------------------------------------------------

def test_critique_done_when_goal_simple_and_plan_ok():
    c = SelfCritique()
    plan = Plan(goal="what's the weather",
                steps=[Step(tool="weather_get", id=1)])
    results = [_ok(1, "weather_get", {"summary": "Sunny."})]
    v = c.critique(goal="what's the weather", plan=plan, results=results)
    assert v.needs_augmentation is False


def test_critique_skipped_when_errors_present():
    c = SelfCritique()
    plan = Plan(goal="g", steps=[Step(tool="x", id=1)])
    results = [StepResult(step_id=1, tool="x", status="error",
                          error="boom")]
    v = c.critique(goal="g", plan=plan, results=results)
    assert v.needs_augmentation is False
    assert "errors present" in v.reason


# ---- fetch-only for action goal ----------------------------------------

def test_critique_adds_send_step_for_fetched_then_email_goal():
    c = SelfCritique()
    plan = Plan(goal="find Dani's latest message and email her",
                steps=[Step(tool="gmail_list", id=1,
                            args={"max": 5})])
    results = [_ok(1, "gmail_list", {"count": 1, "messages": [{}]})]
    v = c.critique(goal="email Dani the summary",
                   plan=plan, results=results)
    assert v.needs_augmentation is True
    assert any(s.tool == "gmail_send" for s in v.augment_steps)
    send = next(s for s in v.augment_steps if s.tool == "gmail_send")
    assert send.args.get("_critic_added") is True
    assert send.args.get("to") == "Dani"


def test_critique_no_augment_when_recipient_unextractable():
    c = SelfCritique()
    plan = Plan(goal="email them the summary",
                steps=[Step(tool="gmail_list", id=1)])
    results = [_ok(1, "gmail_list", {"count": 1})]
    v = c.critique(goal="email them the summary",
                   plan=plan, results=results)
    # 'them' is in the pronoun stop-list → can't infer args → no aug.
    assert v.needs_augmentation is False


def test_critique_handles_slack_action_goal():
    c = SelfCritique()
    plan = Plan(goal="check #ops and post about it",
                steps=[Step(tool="notion_search", id=1,
                            args={"query": "ops"})])
    results = [_ok(1, "notion_search", {"count": 0, "results": []})]
    v = c.critique(goal="check #ops and post a note",
                   plan=plan, results=results)
    assert v.needs_augmentation is True
    assert any(s.tool == "slack_post" for s in v.augment_steps)


# ---- multi-action under-planned ----------------------------------------

def test_critique_catches_single_step_for_multi_action_goal():
    c = SelfCritique()
    plan = Plan(goal="check weather and email Dani",
                steps=[Step(tool="weather_get", id=1,
                            args={"city": "Berlin"})])
    results = [_ok(1, "weather_get", {"summary": "Sunny."})]
    v = c.critique(goal="check weather and email Dani",
                   plan=plan, results=results)
    assert v.needs_augmentation is True
    assert any(s.tool == "gmail_send" for s in v.augment_steps)


# ---- helpers -----------------------------------------------------------

def test_looks_multi_action_patterns():
    assert _looks_multi_action("do X and do Y") is True
    assert _looks_multi_action("do X then do Y") is True
    assert _looks_multi_action("do X, then do Y") is True
    assert _looks_multi_action("just do X") is False
    assert _looks_multi_action("") is False


def test_extract_recipient_from_goal():
    assert _extract_recipient_from_goal("email Dani hi") == "Dani"
    assert _extract_recipient_from_goal("send a note to Alice") == "Alice"
    assert _extract_recipient_from_goal("message them") == ""
    assert _extract_recipient_from_goal("") == ""


def test_extract_channel_from_goal():
    assert _extract_channel_from_goal("post in #general") == "#general"
    assert _extract_channel_from_goal("post in #ops-room") == "#ops-room"
    assert _extract_channel_from_goal("post a note") == ""

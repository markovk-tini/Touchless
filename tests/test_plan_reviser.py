"""Tests for the PlanReviser (Phase 2)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from hgr.live_api.plan_reviser import (  # noqa: E402
    PlanReviser, ReviserConfig, ReviseAction,
    _DETERMINISTIC_FIXES, _connector_id_for_tool,
)
from hgr.live_api.planner.plan import Plan, Step, StepResult  # noqa: E402


# ---- helpers -----------------------------------------------------------

def _ok(step_id: int, tool: str, output=None) -> StepResult:
    return StepResult(step_id=step_id, tool=tool, status="ok",
                      output=output or {})


def _err(step_id: int, tool: str, error: str) -> StepResult:
    return StepResult(step_id=step_id, tool=tool, status="error",
                      error=error, output={"status": "error", "error": error})


def _reviser(cost_meter=None) -> PlanReviser:
    return PlanReviser(config=ReviserConfig(), llm_planner=None,
                       cost_meter=cost_meter)


# ---- DONE paths --------------------------------------------------------

def test_decide_done_when_all_steps_ok():
    r = _reviser()
    plan = Plan(goal="g", steps=[Step(tool="weather_get", id=1)])
    results = [_ok(1, "weather_get", {"summary": "Sunny."})]
    d = r.decide(goal="g", plan=plan, results=results, turn_id="t1")
    assert d.action == ReviseAction.DONE


def test_decide_done_when_user_cancelled():
    # User cancellation is NOT a recoverable error; reviser must not loop.
    r = _reviser()
    plan = Plan(goal="g", steps=[Step(tool="gmail_send", id=1)])
    cancelled = StepResult(step_id=1, tool="gmail_send",
                           status="cancelled",
                           output={"status": "cancelled",
                                   "code": "user_declined"})
    d = r.decide(goal="g", plan=plan, results=[cancelled], turn_id="t2")
    assert d.action == ReviseAction.DONE


# ---- deterministic revisions -------------------------------------------

def test_decide_retries_on_transient_network_error():
    r = _reviser()
    plan = Plan(goal="g", steps=[
        Step(tool="weather_get", id=1, layer="connector",
             args={"city": "Berlin"})])
    results = [_err(1, "weather_get", "Connection timed out")]
    d = r.decide(goal="g", plan=plan, results=results, turn_id="t3")
    assert d.action == ReviseAction.REVISE_AND_RETRY
    assert d.revised_plan is not None
    assert len(d.revised_plan.steps) == 1
    assert d.revised_plan.steps[0].tool == "weather_get"
    assert d.revised_plan.steps[0].args == {"city": "Berlin"}


def test_decide_proposes_contacts_search_on_recipient_invalid():
    r = _reviser()
    plan = Plan(goal="email Dani hi", steps=[
        Step(tool="gmail_send", id=1, args={"to": "Dani", "body": "hi"})])
    results = [_err(1, "gmail_send", "recipient address invalid")]
    d = r.decide(goal="email Dani hi", plan=plan,
                 results=results, turn_id="t4")
    assert d.action == ReviseAction.REVISE_AND_RETRY
    tools = [s.tool for s in d.revised_plan.steps]
    assert "contacts_search" in tools
    assert "gmail_send" in tools
    # The send step should reference the contacts_search output.
    send_step = next(s for s in d.revised_plan.steps if s.tool == "gmail_send")
    assert "{step:" in str(send_step.args.get("to"))


def test_decide_proposes_reauth_message_on_auth_revoked():
    # After SEC/P2-INT-03 audit: the reviser no longer emits an
    # iris_setup_tool step (the executor can't dispatch pseudo-tools).
    # It now emits an _iris_reauth_message step that the orchestrator
    # short-circuits into a user-visible message.
    r = _reviser()
    plan = Plan(goal="send slack", steps=[
        Step(tool="slack_post", id=1, args={"channel": "#general"})])
    results = [_err(1, "slack_post", "401 Unauthorized")]
    d = r.decide(goal="send slack", plan=plan,
                 results=results, turn_id="t5")
    assert d.action == ReviseAction.REVISE_AND_RETRY
    step = d.revised_plan.steps[0]
    assert step.tool == "_iris_reauth_message"
    assert step.args.get("connector") == "slack"
    assert "reconnect" in step.args.get("message", "").lower() \
        or "set up" in step.args.get("message", "").lower()


def test_decide_no_amendment_when_error_unmatched():
    r = _reviser()
    plan = Plan(goal="something", steps=[
        Step(tool="weird_tool", id=1)])
    # 'kernel panic' classifies as 'other' → no deterministic fix.
    results = [_err(1, "weird_tool", "kernel panic deep in libfoo")]
    d = r.decide(goal="something", plan=plan,
                 results=results, turn_id="t6")
    assert d.action == ReviseAction.DONE


# ---- caps --------------------------------------------------------------

def test_decide_caps_revisions_per_turn():
    r = PlanReviser(config=ReviserConfig(max_revisions=2),
                    llm_planner=None, cost_meter=None)
    plan = Plan(goal="g", steps=[
        Step(tool="weather_get", id=1, args={"city": "x"})])
    results = [_err(1, "weather_get", "HTTP 503")]
    # First two decisions should retry, third should DONE out.
    d1 = r.decide(goal="g", plan=plan, results=results, turn_id="cap")
    d2 = r.decide(goal="g", plan=plan, results=results, turn_id="cap")
    d3 = r.decide(goal="g", plan=plan, results=results, turn_id="cap")
    assert d1.action == ReviseAction.REVISE_AND_RETRY
    assert d2.action == ReviseAction.REVISE_AND_RETRY
    assert d3.action == ReviseAction.DONE


def test_decide_reset_turn_clears_caps():
    r = PlanReviser(config=ReviserConfig(max_revisions=1),
                    llm_planner=None, cost_meter=None)
    plan = Plan(goal="g", steps=[
        Step(tool="weather_get", id=1, args={"city": "x"})])
    results = [_err(1, "weather_get", "HTTP 503")]
    r.decide(goal="g", plan=plan, results=results, turn_id="reset")
    assert r.decide(goal="g", plan=plan, results=results,
                    turn_id="reset").action == ReviseAction.DONE
    r.reset_turn("reset")
    assert r.decide(goal="g", plan=plan, results=results,
                    turn_id="reset").action == ReviseAction.REVISE_AND_RETRY


# ---- cost gate ---------------------------------------------------------

class _FakeMeter:
    def __init__(self, over=False):
        self._over = over

    def is_over_cap(self):
        return self._over


def test_decide_bails_when_cost_meter_over_cap():
    r = PlanReviser(config=ReviserConfig(), llm_planner=None,
                    cost_meter=_FakeMeter(over=True))
    plan = Plan(goal="g", steps=[Step(tool="gmail_send", id=1)])
    results = [_err(1, "gmail_send", "HTTP 401")]
    d = r.decide(goal="g", plan=plan, results=results, turn_id="bail")
    assert d.action == ReviseAction.BAIL
    assert "budget" in d.bail_message.lower()


def test_decide_proceeds_when_cost_meter_under_cap():
    r = PlanReviser(config=ReviserConfig(), llm_planner=None,
                    cost_meter=_FakeMeter(over=False))
    plan = Plan(goal="g", steps=[
        Step(tool="weather_get", id=1, args={"city": "x"})])
    results = [_err(1, "weather_get", "HTTP 503")]
    d = r.decide(goal="g", plan=plan, results=results, turn_id="ok")
    assert d.action == ReviseAction.REVISE_AND_RETRY


# ---- disabled-via-config -----------------------------------------------

def test_decide_done_when_revisions_and_critique_disabled():
    cfg = ReviserConfig(enable_revisions=False, enable_critique=False)
    r = PlanReviser(config=cfg, llm_planner=None, cost_meter=None)
    plan = Plan(goal="g", steps=[Step(tool="weather_get", id=1)])
    results = [_err(1, "weather_get", "HTTP 503")]
    d = r.decide(goal="g", plan=plan, results=results, turn_id="off")
    assert d.action == ReviseAction.DONE


# ---- connector id mapping ---------------------------------------------

def test_connector_id_for_tool_recognizes_known_prefixes():
    assert _connector_id_for_tool("gmail_send") == "gmail"
    assert _connector_id_for_tool("ms_mail_send") == "ms365"
    assert _connector_id_for_tool("teams_post") == "ms365"
    assert _connector_id_for_tool("slack_post") == "slack"
    assert _connector_id_for_tool("notion_create_page") == "notion"
    assert _connector_id_for_tool("drive_upload") == "google"
    assert _connector_id_for_tool("spotify_play") == "spotify"


def test_connector_id_for_tool_unknown_returns_empty():
    assert _connector_id_for_tool("weather_get") == ""
    assert _connector_id_for_tool("") == ""


def test_deterministic_fixes_table_consistency():
    # Every class the classifier emits should have a defined fix
    # (even if it's 'no_action' / 'stop').
    expected = {"auth_revoked", "recipient_invalid", "ref_unresolved",
                "not_connected", "rate_limited", "transient_network",
                "upstream_5xx", "user_cancelled"}
    assert set(_DETERMINISTIC_FIXES.keys()) >= expected


def test_llm_revise_skipped_when_cheap_llm_throttled():
    """When the scheduler reports cheap-LLM is rate-limited, the
    reviser should NOT call the LLM planner — falls through to the
    deterministic fix instead. Prevents back-to-back 429s in the
    same turn."""
    from hgr.live_api.planner.scheduler import scheduler

    class _Spy:
        def __init__(self):
            self.calls = 0

        def plan(self, goal):
            self.calls += 1

            class P:
                steps = []
            return P()

    spy = _Spy()
    r = PlanReviser(config=ReviserConfig(), llm_planner=spy,
                    cost_meter=None)
    # Force the cheap-LLM lane into throttle.
    scheduler().record_rate_limit("cheap-llm")
    try:
        # Use a tool that DOES have a connector mapping
        # (gmail_send → "gmail") so the deterministic fallback
        # can emit a _iris_reauth_message step.
        plan = Plan(goal="g", steps=[
            Step(tool="gmail_send", id=1, args={"to": "x"})])
        results = [_err(1, "gmail_send", "HTTP 401")]
        d = r.decide(goal="g", plan=plan, results=results,
                     turn_id="throttled")
        assert spy.calls == 0  # LLM never called
        # Deterministic fallback emitted a reauth message step.
        assert d.action == ReviseAction.REVISE_AND_RETRY
        assert d.revised_plan.steps[0].tool == "_iris_reauth_message"
    finally:
        # Clean up scheduler state for other tests.
        scheduler()._events.clear()

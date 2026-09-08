"""Tests for SkillConsolidator (Phase 4 B2)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.incognito as inc  # noqa: E402
from hgr.live_api.skill_consolidator import (  # noqa: E402
    ConsolidationNudge, MIN_PLAN_LENGTH, MIN_RECURRENCE,
    SkillConsolidator, _suggest_skill_name,
)
from hgr.live_api.tool_invocation import (  # noqa: E402
    InvocationBus, InvocationSource, ToolInvocation,
)


def setup_function():
    inc.set_incognito(False)


def _fresh_consolidator(handler):
    d = Path(tempfile.mkdtemp())
    return SkillConsolidator(
        handler=handler, db_path=d / "sk.db",
        min_plan_length=2, min_recurrence=3)


def _inv(tool: str, *, turn_id="t1", status="ok",
         incognito=False):
    inv = ToolInvocation(
        invocation_id=f"inv-{tool}-{turn_id}",
        turn_id=turn_id, tool=tool, args={},
        source=InvocationSource.PLANNER, status=status)
    if incognito:
        inv.extra["incognito"] = True
    return inv


# ---- shape collection -------------------------------------------------

def test_records_ok_tools_for_turn():
    fired = []
    c = _fresh_consolidator(lambda n: fired.append(n))
    c._on_invocation(_inv("weather_get", turn_id="t1"))
    c._on_invocation(_inv("gmail_send", turn_id="t1"))
    c.complete_turn(turn_id="t1", user_text="check weather then send")
    assert c.shape_count("weather_get>gmail_send") == 1


def test_short_plans_skipped():
    fired = []
    c = _fresh_consolidator(lambda n: fired.append(n))
    c._on_invocation(_inv("weather_get", turn_id="t1"))
    c.complete_turn(turn_id="t1", user_text="just weather")
    assert c.shape_count("weather_get") == 0


def test_error_tools_not_counted_in_shape():
    fired = []
    c = _fresh_consolidator(lambda n: fired.append(n))
    c._on_invocation(_inv("weather_get", turn_id="t1"))
    c._on_invocation(_inv("gmail_send", turn_id="t1", status="error"))
    c._on_invocation(_inv("notion_create_page", turn_id="t1"))
    c.complete_turn(turn_id="t1", user_text="multi")
    # gmail_send (error) excluded; shape is weather_get>notion_create_page.
    assert c.shape_count("weather_get>notion_create_page") == 1
    assert c.shape_count("weather_get>gmail_send>notion_create_page") == 0


def test_pseudo_tools_skipped():
    fired = []
    c = _fresh_consolidator(lambda n: fired.append(n))
    c._on_invocation(_inv("_iris_reauth_message", turn_id="t1"))
    c._on_invocation(_inv("gmail_send", turn_id="t1"))
    c._on_invocation(_inv("notion_create_page", turn_id="t1"))
    c.complete_turn(turn_id="t1", user_text="multi")
    assert c.shape_count("gmail_send>notion_create_page") == 1


def test_incognito_drops_in_flight():
    fired = []
    c = _fresh_consolidator(lambda n: fired.append(n))
    inc.set_incognito(False)  # collect normally
    c._on_invocation(_inv("weather_get", turn_id="t1"))
    c._on_invocation(_inv("gmail_send", turn_id="t1"))
    # Flip incognito BEFORE complete_turn — turn must drop silently.
    inc.set_incognito(True)
    try:
        c.complete_turn(turn_id="t1", user_text="x")
    finally:
        inc.set_incognito(False)
    assert c.shape_count("weather_get>gmail_send") == 0


def test_incognito_invocations_not_recorded():
    fired = []
    c = _fresh_consolidator(lambda n: fired.append(n))
    c._on_invocation(_inv("weather_get", turn_id="t1", incognito=True))
    c._on_invocation(_inv("gmail_send", turn_id="t1", incognito=True))
    c.complete_turn(turn_id="t1", user_text="private")
    assert c.shape_count("weather_get>gmail_send") == 0


# ---- nudge emission ---------------------------------------------------

def test_nudge_fires_after_min_recurrence():
    fired = []
    c = _fresh_consolidator(lambda n: fired.append(n))
    for i in range(MIN_RECURRENCE):
        tid = f"turn-{i}"
        c._on_invocation(_inv("weather_get", turn_id=tid))
        c._on_invocation(_inv("gmail_send", turn_id=tid))
        c.complete_turn(turn_id=tid,
                        user_text="check weather email Dani")
    assert len(fired) == 1
    nudge = fired[0]
    assert isinstance(nudge, ConsolidationNudge)
    assert nudge.shape == "weather_get>gmail_send"
    assert nudge.occurrence_count == MIN_RECURRENCE
    assert "weather_get" in nudge.sample_tools


def test_nudge_skipped_below_threshold():
    fired = []
    c = _fresh_consolidator(lambda n: fired.append(n))
    for i in range(MIN_RECURRENCE - 1):
        tid = f"turn-{i}"
        c._on_invocation(_inv("weather_get", turn_id=tid))
        c._on_invocation(_inv("gmail_send", turn_id=tid))
        c.complete_turn(turn_id=tid, user_text="x")
    assert fired == []


def test_nudge_cooldown_blocks_repeat():
    fired = []
    c = _fresh_consolidator(lambda n: fired.append(n))
    # Run the same shape MIN_RECURRENCE * 2 times.
    for i in range(MIN_RECURRENCE * 2):
        tid = f"turn-{i}"
        c._on_invocation(_inv("weather_get", turn_id=tid))
        c._on_invocation(_inv("gmail_send", turn_id=tid))
        c.complete_turn(turn_id=tid, user_text="x")
    # Should only fire ONCE despite multiple threshold crossings.
    assert len(fired) == 1


def test_distinct_shapes_get_distinct_nudges():
    fired = []
    c = _fresh_consolidator(lambda n: fired.append(n))
    for i in range(MIN_RECURRENCE):
        c._on_invocation(_inv("weather_get",
                              turn_id=f"weather-{i}"))
        c._on_invocation(_inv("gmail_send",
                              turn_id=f"weather-{i}"))
        c.complete_turn(turn_id=f"weather-{i}", user_text="w")
    for i in range(MIN_RECURRENCE):
        c._on_invocation(_inv("notion_search",
                              turn_id=f"notion-{i}"))
        c._on_invocation(_inv("slack_post",
                              turn_id=f"notion-{i}"))
        c.complete_turn(turn_id=f"notion-{i}", user_text="n")
    shapes = {n.shape for n in fired}
    assert shapes == {"weather_get>gmail_send",
                      "notion_search>slack_post"}


# ---- attach idempotency ------------------------------------------------

def test_attach_to_bus_idempotent():
    fired = []
    c = _fresh_consolidator(lambda n: fired.append(n))
    bus = InvocationBus()
    c.attach_to_bus(bus)
    c.attach_to_bus(bus)  # second call replaces; no double delivery
    for i in range(MIN_RECURRENCE):
        tid = f"t-{i}"
        bus.publish(_inv("weather_get", turn_id=tid))
        bus.publish(_inv("gmail_send", turn_id=tid))
        c.complete_turn(turn_id=tid, user_text="x")
    # Exactly one nudge despite the double attach attempt.
    assert len(fired) == 1


# ---- helpers ----------------------------------------------------------

def test_suggest_skill_name_uses_short_user_text():
    assert (_suggest_skill_name(["a", "b"],
                                  "send the daily standup")
            == "send the daily standup")


def test_suggest_skill_name_fallback_uses_tool_head():
    out = _suggest_skill_name(["gmail_send", "notion_create_page"], "")
    assert "gmail" in out


def test_suggest_skill_name_strips_trailing_punct():
    out = _suggest_skill_name(["a", "b"], "Do the morning routine?")
    assert "?" not in out

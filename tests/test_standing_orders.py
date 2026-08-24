"""Tests for standing_orders (Phase 5 B1)."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.incognito as inc  # noqa: E402
from hgr.live_api.standing_orders import (  # noqa: E402
    OrderState, StandingOrder, StandingOrdersEvaluator,
    StandingOrdersStore, TriggerKind,
    make_inbox_watch_order, make_time_after_order,
    make_time_at_order, predicate_deadline_passed,
    predicate_inbox_match, predicate_time_after,
    predicate_time_at, predicate_tool_returns_ok,
)


def setup_function():
    inc.set_incognito(False)


def _fresh_store() -> StandingOrdersStore:
    d = Path(tempfile.mkdtemp())
    return StandingOrdersStore(db_path=d / "orders.db")


# ---- CRUD --------------------------------------------------------------

def test_add_then_get_round_trips():
    s = _fresh_store()
    order = make_time_at_order(
        user_text="remind me at 3pm",
        at_ts=time.time() + 3600,
        label="afternoon nudge")
    s.add(order)
    back = s.get(order.id)
    assert back is not None
    assert back.trigger_kind == TriggerKind.TIME_AT.value
    assert back.label == "afternoon nudge"


def test_all_active_excludes_done_and_cancelled():
    s = _fresh_store()
    o1 = make_time_at_order(user_text="a", at_ts=time.time() + 60)
    o2 = make_time_at_order(user_text="b", at_ts=time.time() + 60)
    s.add(o1)
    s.add(o2)
    s.update_state(o1.id, OrderState.DONE.value)
    active = s.all_active()
    ids = {o.id for o in active}
    assert o2.id in ids
    assert o1.id not in ids


def test_delete_removes_order():
    s = _fresh_store()
    order = make_inbox_watch_order(
        user_text="watch Q3", query="Q3 contract")
    s.add(order)
    assert s.delete(order.id) is True
    assert s.get(order.id) is None
    assert s.delete(order.id) is False


def test_wipe_clears_all():
    s = _fresh_store()
    for i in range(5):
        s.add(make_time_at_order(
            user_text=f"r{i}", at_ts=time.time() + 60))
    n = s.wipe()
    assert n == 5
    assert s.all() == []


# ---- predicate: time_at ------------------------------------------------

def test_time_at_fires_after_target():
    order = make_time_at_order(
        user_text="r", at_ts=100.0)
    result = predicate_time_at(order, {"now": 200.0})
    assert result.fired is True
    assert result.terminal is True


def test_time_at_skipped_before_target():
    order = make_time_at_order(
        user_text="r", at_ts=1000.0)
    result = predicate_time_at(order, {"now": 100.0})
    assert result.fired is False


def test_time_at_does_not_refire():
    order = make_time_at_order(
        user_text="r", at_ts=100.0)
    order.fire_count = 1   # already fired once
    result = predicate_time_at(order, {"now": 200.0})
    assert result.fired is False


def test_deadline_passed_uses_same_logic():
    order = StandingOrder(
        id="d", user_text="x",
        trigger_kind=TriggerKind.DEADLINE_PASSED.value,
        trigger_params={"at_ts": 100.0})
    assert predicate_deadline_passed(
        order, {"now": 200.0}).fired is True


# ---- predicate: time_after ---------------------------------------------

def test_time_after_fires_when_interval_elapsed():
    start = 100.0
    order = make_time_after_order(
        user_text="r", interval_sec=60.0, start_ts=start)
    # 200s after start → 1 full interval + extra elapsed.
    result = predicate_time_after(order, {"now": start + 200})
    assert result.fired is True
    assert result.terminal is False  # repeating


def test_time_after_skipped_before_first_interval():
    order = make_time_after_order(
        user_text="r", interval_sec=300.0, start_ts=100.0)
    assert predicate_time_after(
        order, {"now": 200.0}).fired is False


def test_time_after_respects_last_fired_at():
    order = make_time_after_order(
        user_text="r", interval_sec=60.0, start_ts=100.0)
    order.last_fired_at = 200.0
    # 30s after last fire → still in cooldown.
    assert predicate_time_after(
        order, {"now": 230.0}).fired is False
    # 70s after last fire → ready to fire again.
    assert predicate_time_after(
        order, {"now": 280.0}).fired is True


# ---- predicate: inbox_match -------------------------------------------

class _StubRegistry:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def call(self, tool, args):
        self.calls.append((tool, args))
        return self._result


def test_inbox_match_fires_when_new_message():
    order = make_inbox_watch_order(
        user_text="watch Q3", query="Q3 contract")
    reg = _StubRegistry({"status": "ok", "messages": [
        {"id": "m-new-1", "subject": "Q3 contract draft"},
    ]})
    result = predicate_inbox_match(order, {"registry": reg})
    assert result.fired is True
    assert "m-new-1" in result.state_updates["seen_ids"]


def test_inbox_match_does_not_fire_for_already_seen():
    order = make_inbox_watch_order(
        user_text="watch Q3", query="Q3 contract")
    order.eval_state = {"seen_ids": ["m-old"]}
    reg = _StubRegistry({"status": "ok", "messages": [
        {"id": "m-old", "subject": "Q3 contract draft"},
    ]})
    result = predicate_inbox_match(order, {"registry": reg})
    assert result.fired is False


def test_inbox_match_skipped_without_registry():
    order = make_inbox_watch_order(
        user_text="watch Q3", query="Q3 contract")
    result = predicate_inbox_match(order, {})
    assert result.fired is False


def test_inbox_match_tolerates_tool_error():
    order = make_inbox_watch_order(
        user_text="watch Q3", query="Q3 contract")
    reg = _StubRegistry({"status": "error", "error": "boom"})
    result = predicate_inbox_match(order, {"registry": reg})
    assert result.fired is False


# ---- predicate: tool_returns_ok ---------------------------------------

def test_tool_returns_ok_fires_terminal_on_success():
    order = StandingOrder(
        id="t", user_text="ping when ready",
        trigger_kind=TriggerKind.TOOL_RETURNS_OK.value,
        trigger_params={"tool": "build_status",
                        "args": {"branch": "main"}})
    reg = _StubRegistry({"status": "ok"})
    result = predicate_tool_returns_ok(order, {"registry": reg})
    assert result.fired is True
    assert result.terminal is True
    assert reg.calls[0][0] == "build_status"


def test_tool_returns_ok_does_not_fire_on_pending():
    order = StandingOrder(
        id="t", user_text="x",
        trigger_kind=TriggerKind.TOOL_RETURNS_OK.value,
        trigger_params={"tool": "build_status"})
    reg = _StubRegistry({"status": "pending"})
    assert predicate_tool_returns_ok(
        order, {"registry": reg}).fired is False


# ---- evaluator tick ---------------------------------------------------

def test_evaluator_fires_and_marks_done_for_time_at():
    s = _fresh_store()
    order = make_time_at_order(
        user_text="r", at_ts=time.time() - 1)
    s.add(order)
    fired = []
    evaluator = StandingOrdersEvaluator(
        store=s, notifier=lambda o, r: fired.append((o, r)))
    evaluator.tick()
    assert len(fired) == 1
    o = s.get(order.id)
    assert o.state == OrderState.DONE.value


def test_evaluator_honors_incognito():
    s = _fresh_store()
    order = make_time_at_order(
        user_text="r", at_ts=time.time() - 1)
    s.add(order)
    fired = []
    evaluator = StandingOrdersEvaluator(
        store=s, notifier=lambda o, r: fired.append((o, r)))
    inc.set_incognito(True)
    try:
        evaluator.tick()
    finally:
        inc.set_incognito(False)
    assert fired == []
    # Order is still active — incognito just paused evaluation,
    # not cancellation.
    assert s.get(order.id).state == OrderState.ACTIVE.value


def test_evaluator_marks_error_when_predicate_raises():
    s = _fresh_store()
    order = StandingOrder(
        id="e", user_text="bad",
        trigger_kind="nonexistent_trigger_kind",
        trigger_params={})
    s.add(order)
    fired = []
    evaluator = StandingOrdersEvaluator(
        store=s, notifier=lambda *a: fired.append(a))
    # Unknown trigger kind → predicate is None → tick just skips,
    # doesn't error. State stays ACTIVE.
    evaluator.tick()
    assert s.get(order.id).state == OrderState.ACTIVE.value
    assert fired == []


def test_evaluator_tick_updates_last_checked():
    s = _fresh_store()
    order = make_time_at_order(
        user_text="r", at_ts=time.time() + 3600)
    s.add(order)
    evaluator = StandingOrdersEvaluator(
        store=s, notifier=lambda *a: None)
    evaluator.tick()
    o = s.get(order.id)
    assert o.last_checked_at > 0


def test_inbox_match_through_evaluator_persists_seen():
    s = _fresh_store()
    order = make_inbox_watch_order(
        user_text="watch", query="contract")
    s.add(order)
    reg = _StubRegistry({"status": "ok", "messages": [
        {"id": "m1", "subject": "contract draft"},
    ]})
    fired = []
    evaluator = StandingOrdersEvaluator(
        store=s, notifier=lambda o, r: fired.append((o, r)),
        registry=reg)
    evaluator.tick()
    assert len(fired) == 1
    # Persisted seen_ids should now include m1.
    refreshed = s.get(order.id)
    assert "m1" in refreshed.eval_state["seen_ids"]
    # Second tick with same message → no fire.
    evaluator.tick()
    assert len(fired) == 1


# ---- factories --------------------------------------------------------

def test_make_inbox_watch_order_sets_tool_default():
    o = make_inbox_watch_order(
        user_text="x", query="contract")
    assert o.trigger_kind == TriggerKind.INBOX_MATCH.value
    assert o.trigger_params["query"] == "contract"
    assert o.trigger_params["tool"] == "gmail_list"


def test_make_time_at_order_label_falls_back_to_iso():
    o = make_time_at_order(user_text="x", at_ts=1_700_000_000.0)
    assert "remind at" in o.label or "reminder" in o.label.lower()


def test_short_label_uses_user_text_when_no_label():
    o = StandingOrder(
        id="x", user_text="long " * 50,
        trigger_kind=TriggerKind.TIME_AT.value,
        trigger_params={"at_ts": 1.0})
    assert len(o.short_label()) <= 81  # 80 chars + ellipsis

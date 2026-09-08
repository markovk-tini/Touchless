"""Standing orders — long-running goals Iris pursues in the background.

Phase-5 cognition. Today every Iris turn is reactive: user says X,
Iris does X, then goes idle until the next turn. For genuine
Jarvis behavior she needs persistent goals that survive restarts
and act on their own when conditions become true:

  * "watch my inbox for the Q3 contract from Dani and tell me
    when it arrives" → poll gmail every N minutes, fire a nudge
    when a matching message lands.
  * "ping me when the CI build for branch X finishes" → poll a
    URL or check tool status, fire on transition.
  * "remind me at 3pm to call my dentist" → simple time trigger.
  * "if I haven't sent an email to Vesko by Friday EOD, prompt
    me to draft one" → deadline + checkpoint.

This module owns the DATA MODEL + EVALUATOR. The actual condition
checks delegate to predicates so we can add new triggers without
touching the core loop.

Storage: SQLite at `%LOCALAPPDATA%\\Touchless\\private\\orders.db`.
Survives restarts (the whole point).

Condition types (initial set, extensible):
  * `inbox_match`     — gmail_list / ms_mail_list polled with a
                         query; fires when count > 0 on a fresh result.
  * `time_at`          — fire once at a specific epoch ts.
  * `time_after`       — fire repeatedly at intervals from a start.
  * `tool_returns_ok`  — fire when a specified tool call succeeds.
  * `deadline_passed`  — fire when wall clock crosses a deadline.

Honors incognito (orders ARE persisted, but the evaluator pauses
emission in private mode — orders resume on exit).

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class OrderState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DONE = "done"
    CANCELLED = "cancelled"
    ERROR = "error"


class TriggerKind(str, Enum):
    INBOX_MATCH = "inbox_match"
    TIME_AT = "time_at"
    TIME_AFTER = "time_after"
    TOOL_RETURNS_OK = "tool_returns_ok"
    DEADLINE_PASSED = "deadline_passed"


@dataclass
class StandingOrder:
    """One persistent goal Iris is watching for the user."""
    id: str
    user_text: str                  # original natural-language goal
    trigger_kind: str               # see TriggerKind
    trigger_params: Dict[str, Any] = field(default_factory=dict)
    action: str = "notify"          # 'notify' (default) | tool_name
    action_args: Dict[str, Any] = field(default_factory=dict)
    state: str = OrderState.ACTIVE.value
    created_at: float = field(default_factory=time.time)
    last_checked_at: float = 0.0
    last_fired_at: float = 0.0
    fire_count: int = 0
    # Per-trigger transient state (e.g., inbox-match: last-seen
    # message-id set so a still-present match doesn't re-fire).
    eval_state: Dict[str, Any] = field(default_factory=dict)
    # User-facing label the UI shows in /orders.
    label: str = ""

    def short_label(self) -> str:
        if self.label:
            return self.label
        t = (self.user_text or "").strip()
        return t[:80] + ("…" if len(t) > 80 else "") or "(no label)"


def _default_db() -> Path:
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(local) / "Touchless" / "private" / "orders.db"


class StandingOrdersStore:
    """SQLite-backed persistent store + thin CRUD."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS orders (
        id              TEXT PRIMARY KEY,
        user_text       TEXT NOT NULL,
        trigger_kind    TEXT NOT NULL,
        trigger_params  TEXT NOT NULL,
        action          TEXT NOT NULL DEFAULT 'notify',
        action_args     TEXT NOT NULL DEFAULT '{}',
        state           TEXT NOT NULL DEFAULT 'active',
        created_at      REAL NOT NULL,
        last_checked_at REAL DEFAULT 0,
        last_fired_at   REAL DEFAULT 0,
        fire_count      INTEGER DEFAULT 0,
        eval_state      TEXT NOT NULL DEFAULT '{}',
        label           TEXT DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_orders_state ON orders(state);
    """

    def __init__(self, *, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _default_db()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.executescript(self.SCHEMA)

    # ---- CRUD ---------------------------------------------------------

    def add(self, order: StandingOrder) -> str:
        with self._lock:
            self._conn.execute(
                "INSERT INTO orders"
                "(id, user_text, trigger_kind, trigger_params, "
                " action, action_args, state, created_at, "
                " eval_state, label) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (order.id, order.user_text, order.trigger_kind,
                 json.dumps(order.trigger_params),
                 order.action, json.dumps(order.action_args),
                 order.state, order.created_at,
                 json.dumps(order.eval_state),
                 order.label),
            )
        return order.id

    def get(self, order_id: str) -> Optional[StandingOrder]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM orders WHERE id=?", (order_id,))
            row = cur.fetchone()
        return self._row_to_order(row) if row else None

    def all_active(self) -> List[StandingOrder]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM orders WHERE state IN (?, ?) "
                "ORDER BY created_at ASC",
                (OrderState.ACTIVE.value, OrderState.PAUSED.value))
            rows = cur.fetchall()
        return [self._row_to_order(r) for r in rows]

    def all(self, *, include_done: bool = False
            ) -> List[StandingOrder]:
        with self._lock:
            if include_done:
                cur = self._conn.execute(
                    "SELECT * FROM orders ORDER BY created_at DESC")
            else:
                cur = self._conn.execute(
                    "SELECT * FROM orders WHERE state != ? "
                    "AND state != ? ORDER BY created_at DESC",
                    (OrderState.DONE.value, OrderState.CANCELLED.value))
            rows = cur.fetchall()
        return [self._row_to_order(r) for r in rows]

    def update_state(self, order_id: str,
                     state: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE orders SET state=? WHERE id=?",
                (state, order_id))
            return cur.rowcount > 0

    def update_after_check(self, order: StandingOrder,
                           *, fired: bool) -> None:
        with self._lock:
            now = time.time()
            order.last_checked_at = now
            if fired:
                order.last_fired_at = now
                order.fire_count += 1
            self._conn.execute(
                "UPDATE orders SET last_checked_at=?, last_fired_at=?, "
                "fire_count=?, eval_state=?, state=? WHERE id=?",
                (order.last_checked_at, order.last_fired_at,
                 order.fire_count,
                 json.dumps(order.eval_state),
                 order.state, order.id),
            )

    def delete(self, order_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM orders WHERE id=?", (order_id,))
            return cur.rowcount > 0

    def wipe(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM orders")
            n = int(cur.fetchone()[0])
            self._conn.execute("DELETE FROM orders")
            return n

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    @staticmethod
    def _row_to_order(row: Any) -> StandingOrder:
        if row is None:
            return None  # type: ignore[return-value]
        # SQLite row indexing: id, user_text, trigger_kind,
        # trigger_params, action, action_args, state, created_at,
        # last_checked_at, last_fired_at, fire_count, eval_state, label
        try:
            tp = json.loads(row[3] or "{}")
            aa = json.loads(row[5] or "{}")
            es = json.loads(row[11] or "{}")
        except Exception:
            tp, aa, es = {}, {}, {}
        return StandingOrder(
            id=row[0], user_text=row[1] or "",
            trigger_kind=row[2] or "",
            trigger_params=tp, action=row[4] or "notify",
            action_args=aa, state=row[6] or OrderState.ACTIVE.value,
            created_at=float(row[7] or 0.0),
            last_checked_at=float(row[8] or 0.0),
            last_fired_at=float(row[9] or 0.0),
            fire_count=int(row[10] or 0),
            eval_state=es, label=row[12] or "",
        )


# ---- Condition evaluator -----------------------------------------------

@dataclass
class FireResult:
    fired: bool
    detail: str = ""
    state_updates: Dict[str, Any] = field(default_factory=dict)
    # When True, the order is one-shot — mark DONE after fire.
    terminal: bool = False


# Each predicate signature: (order, ctx) -> FireResult.
# `ctx` is a free-form dict the evaluator passes through (current
# time, optional registry handle, etc.) so predicates stay pure.
Predicate = Callable[[StandingOrder, Dict[str, Any]], FireResult]


def predicate_time_at(order: StandingOrder,
                      ctx: Dict[str, Any]) -> FireResult:
    """Fire ONCE when wall clock crosses params['at_ts'] (epoch seconds)."""
    target = float(order.trigger_params.get("at_ts") or 0.0)
    if target <= 0:
        return FireResult(False, "no at_ts set")
    now = float(ctx.get("now") or time.time())
    if now < target:
        return FireResult(False, "not yet")
    # Already fired? Don't refire.
    if order.fire_count > 0:
        return FireResult(False, "already fired")
    return FireResult(True, "time_at reached", terminal=True)


def predicate_deadline_passed(order: StandingOrder,
                              ctx: Dict[str, Any]) -> FireResult:
    """Same shape as time_at but the user-facing semantic is 'deadline'
    rather than 'reminder' — different wording in the notification."""
    return predicate_time_at(order, ctx)


def predicate_time_after(order: StandingOrder,
                         ctx: Dict[str, Any]) -> FireResult:
    """Fire repeatedly every params['interval_sec'] from
    params['start_ts'] (defaults to created_at)."""
    interval = float(order.trigger_params.get("interval_sec") or 0.0)
    if interval <= 0:
        return FireResult(False, "no interval_sec set")
    start = float(order.trigger_params.get("start_ts")
                  or order.created_at)
    now = float(ctx.get("now") or time.time())
    if now < start + interval:
        return FireResult(False, "interval not elapsed yet")
    last_fired = order.last_fired_at
    elapsed = now - max(last_fired, start)
    if elapsed < interval:
        return FireResult(False, "cooldown active")
    return FireResult(True, "time_after interval elapsed",
                       terminal=False)


def predicate_inbox_match(order: StandingOrder,
                          ctx: Dict[str, Any]) -> FireResult:
    """Poll the inbox via a connector and fire when a NEW matching
    message arrives. `eval_state['seen_ids']` is the set of message
    IDs we've already seen — so the order doesn't re-fire on the
    same message every tick."""
    registry = ctx.get("registry")
    if registry is None:
        return FireResult(False, "no registry available")
    query = str(order.trigger_params.get("query") or "").strip()
    tool = str(order.trigger_params.get("tool") or "gmail_list")
    if not query:
        return FireResult(False, "no query set")
    try:
        result = registry.call(tool, {"max": 25, "query": query})
    except Exception as exc:
        return FireResult(False, f"call failed: {exc}")
    if not isinstance(result, dict) or result.get("status") != "ok":
        return FireResult(False, "tool error")
    messages = result.get("messages") or []
    seen = set(order.eval_state.get("seen_ids") or [])
    new_msgs = []
    for m in messages:
        mid = str(m.get("id") or m.get("message_id") or "").strip()
        if mid and mid not in seen:
            new_msgs.append(m)
            seen.add(mid)
    # Update seen set regardless (cap to recent 200 to bound size).
    new_state = {"seen_ids": list(seen)[-200:]}
    if not new_msgs:
        return FireResult(False, "no new matches",
                          state_updates=new_state)
    return FireResult(True,
                      f"{len(new_msgs)} new message(s) match '{query}'",
                      state_updates=new_state)


def predicate_tool_returns_ok(order: StandingOrder,
                              ctx: Dict[str, Any]) -> FireResult:
    """Fire when the configured tool returns status=='ok'. One-shot."""
    registry = ctx.get("registry")
    if registry is None:
        return FireResult(False, "no registry available")
    tool = str(order.trigger_params.get("tool") or "").strip()
    args = dict(order.trigger_params.get("args") or {})
    if not tool:
        return FireResult(False, "no tool set")
    try:
        result = registry.call(tool, args)
    except Exception as exc:
        return FireResult(False, f"call failed: {exc}")
    if isinstance(result, dict) and result.get("status") == "ok":
        return FireResult(True, "tool returned ok",
                          terminal=True)
    return FireResult(False, "tool not OK yet")


_PREDICATES: Dict[str, Predicate] = {
    TriggerKind.TIME_AT.value:        predicate_time_at,
    TriggerKind.TIME_AFTER.value:     predicate_time_after,
    TriggerKind.INBOX_MATCH.value:    predicate_inbox_match,
    TriggerKind.TOOL_RETURNS_OK.value: predicate_tool_returns_ok,
    TriggerKind.DEADLINE_PASSED.value: predicate_deadline_passed,
}


# ---- The evaluator + watcher -------------------------------------------

NotificationHandler = Callable[[StandingOrder, FireResult], None]


class StandingOrdersEvaluator:
    """Sentinel-tickable evaluator. On each tick, iterates active
    orders + runs the matching predicate + notifies on fire."""

    def __init__(self, *, store: StandingOrdersStore,
                 notifier: NotificationHandler,
                 registry: Any = None) -> None:
        self._store = store
        self._notifier = notifier
        self._registry = registry

    def set_registry(self, registry: Any) -> None:
        self._registry = registry

    def tick(self) -> None:
        # Incognito: pause evaluation, don't even read orders.
        try:
            from .incognito import is_incognito
            if is_incognito():
                return
        except Exception:
            pass
        try:
            active = self._store.all_active()
        except Exception:
            return
        ctx = {"now": time.time(), "registry": self._registry}
        for order in active:
            if order.state != OrderState.ACTIVE.value:
                continue
            predicate = _PREDICATES.get(order.trigger_kind)
            if predicate is None:
                continue
            try:
                result = predicate(order, ctx)
            except Exception as exc:
                # Order's predicate raised — flag it so the user
                # can see the failure rather than silent stalling.
                order.state = OrderState.ERROR.value
                order.eval_state["last_error"] = (
                    f"{type(exc).__name__}: {exc}")[:200]
                try:
                    self._store.update_after_check(order, fired=False)
                except Exception:
                    pass
                continue
            # Persist any state updates the predicate made.
            if result.state_updates:
                order.eval_state.update(result.state_updates)
            # Delivery first; only mark fired (and terminal->DONE) once
            # the notifier confirms it actually surfaced the reminder.
            # Otherwise a disconnected realtime client or full pending
            # queue silently incremented fire_count and the reminder
            # was lost forever — the regression the user hit when
            # "remind me to call dad in 1 minute" never spoke.
            delivered = False
            if result.fired:
                try:
                    ack = self._notifier(order, result)
                except Exception:
                    ack = False
                delivered = bool(ack) if ack is not None else True
            effective_fired = bool(result.fired and delivered)
            if effective_fired and result.terminal:
                order.state = OrderState.DONE.value
            try:
                self._store.update_after_check(
                    order, fired=effective_fired)
            except Exception:
                pass


# ---- Construction helpers ----------------------------------------------

def parse_when(text: str) -> Optional[float]:
    """Parse a casual time phrase ('in 30 min', '3pm',
    'tomorrow 9am', '5:30pm', 'in 2 hours') into an epoch
    seconds value. Returns None on failure so the caller can
    prompt for retry. Shared between the slash-command UI
    and the natural-language classifier path."""
    if not text:
        return None
    import re as _re
    import time as _time
    from datetime import datetime as _dt, timedelta as _td
    t = text.strip().lower()
    now = _dt.now()
    # 'in N (sec|s|min|m|hour|h)'
    m = _re.match(
        r"in\s+(\d+)\s*(s|sec|second|seconds|m|min|minute|minutes|"
        r"h|hr|hour|hours)\b", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("s"):
            return _time.time() + n
        if unit.startswith("h"):
            return _time.time() + n * 3600
        return _time.time() + n * 60   # default minutes
    # 'tomorrow [time]'
    tomorrow = "tomorrow" in t
    base = now + _td(days=1) if tomorrow else now
    if tomorrow:
        t = t.replace("tomorrow", "").strip()
    # 'Ham|Hpm' / 'H:MMam|pm'
    m = _re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2) or 0)
        ampm = m.group(3)
        if ampm == "pm" and h < 12:
            h += 12
        if ampm == "am" and h == 12:
            h = 0
        target = base.replace(hour=h, minute=mm, second=0,
                              microsecond=0)
        if not tomorrow and target <= now:
            target += _td(days=1)
        return target.timestamp()
    # 24-hour 'HH:MM'
    m = _re.match(r"(\d{1,2}):(\d{2})\b", t)
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
        target = base.replace(hour=h, minute=mm, second=0,
                              microsecond=0)
        if not tomorrow and target <= now:
            target += _td(days=1)
        return target.timestamp()
    return None


def make_inbox_watch_order(*, user_text: str, query: str,
                           tool: str = "gmail_list",
                           label: str = "") -> StandingOrder:
    return StandingOrder(
        id=uuid.uuid4().hex[:12],
        user_text=user_text,
        trigger_kind=TriggerKind.INBOX_MATCH.value,
        trigger_params={"query": query, "tool": tool},
        label=label or f"inbox: {query}",
    )


def make_time_at_order(*, user_text: str, at_ts: float,
                       label: str = "") -> StandingOrder:
    return StandingOrder(
        id=uuid.uuid4().hex[:12],
        user_text=user_text,
        trigger_kind=TriggerKind.TIME_AT.value,
        trigger_params={"at_ts": float(at_ts)},
        label=label or f"reminder at {datetime.fromtimestamp(at_ts).isoformat(timespec='minutes')}",
    )


def make_time_after_order(*, user_text: str, interval_sec: float,
                          start_ts: Optional[float] = None,
                          label: str = "") -> StandingOrder:
    params: Dict[str, Any] = {"interval_sec": float(interval_sec)}
    if start_ts is not None:
        params["start_ts"] = float(start_ts)
    return StandingOrder(
        id=uuid.uuid4().hex[:12],
        user_text=user_text,
        trigger_kind=TriggerKind.TIME_AFTER.value,
        trigger_params=params,
        label=label or f"every {int(interval_sec)}s",
    )


# ---- Module singletons -------------------------------------------------

_store: Optional[StandingOrdersStore] = None
_evaluator: Optional[StandingOrdersEvaluator] = None
_lock = threading.RLock()


def global_store() -> StandingOrdersStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = StandingOrdersStore()
    return _store


def global_evaluator(*, notifier: Optional[NotificationHandler] = None,
                     registry: Any = None
                     ) -> StandingOrdersEvaluator:
    global _evaluator
    with _lock:
        if _evaluator is None:
            if notifier is None:
                notifier = lambda order, res: None
            _evaluator = StandingOrdersEvaluator(
                store=global_store(), notifier=notifier,
                registry=registry)
        else:
            # Update injected refs without rebuilding.
            if notifier is not None:
                _evaluator._notifier = notifier
            if registry is not None:
                _evaluator._registry = registry
    return _evaluator


def _reset_for_tests() -> None:
    global _store, _evaluator
    with _lock:
        if _store is not None:
            try:
                _store.close()
            except Exception:
                pass
        _store = None
        _evaluator = None

"""Self-learning skill consolidator.

Phase-4 cognition. Iris records every successful multi-step plan
via the CoT layer + tool_invocation bus. Over time, the same
plan shape recurs (the user says "check my email and reply to
Dani" or "save this to Notion and post in Slack" five times in
a week). Each instance burns a fresh planner LLM call.

This module watches for those recurring shapes and proposes them
as SAVED SKILLS (`SkillStore`). Once saved, the Tier-0.5 skill
replay path returns the same plan in 0 tokens, 0 ms.

Detection algorithm:
  1. Subscribe to the InvocationBus.
  2. For each (turn_id, source) pair, collect the ordered list of
     tools that ran successfully.
  3. Build a "shape signature" = "tool1>tool2>tool3" (no args).
  4. After each completed turn, see if THIS signature has been
     seen N times before (counter persisted in a side table).
  5. If count crosses threshold, emit a Nudge: "Want me to save
     this as 'send the daily standup'?" — the user decides.

Conservative defaults:
  * Min plan length 2 (single-step plans are already cheap).
  * Min recurrence 3 (one-offs aren't worth saving).
  * Per-shape cooldown 1 day (don't nag for a re-prompt).
  * Honors incognito.

This is GROUND TRUTH self-learning — what the user actually does,
not what they say. Pattern-matching free, no LLM in the
detection loop.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# Defaults — tune via env once we have field data.
MIN_PLAN_LENGTH = 2
MIN_RECURRENCE = 3
PER_SHAPE_NUDGE_COOLDOWN_SEC = 24 * 3600.0

# Cap on persisted shape history so a chatty user doesn't bloat
# the DB. ~10k turn-shapes is enough for a multi-month history.
MAX_HISTORY_ROWS = 10_000


def _default_db() -> Path:
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(local) / "Touchless" / "private" / "skill_consolidator.db"


@dataclass
class ConsolidationNudge:
    """Payload emitted when a recurring shape crosses the threshold."""
    shape: str                 # "weather_get>gmail_send"
    occurrence_count: int
    sample_tools: List[str] = field(default_factory=list)
    last_user_text: str = ""
    suggested_skill_name: str = ""


NudgeHandler = Callable[[ConsolidationNudge], None]


class SkillConsolidator:
    """Bus subscriber + shape counter. Emits a ConsolidationNudge
    when a recurring multi-step pattern crosses MIN_RECURRENCE."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS turn_shapes (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ts           REAL NOT NULL,
        turn_id      TEXT NOT NULL,
        shape        TEXT NOT NULL,
        user_text    TEXT,
        sample_tools TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_shapes_shape
        ON turn_shapes(shape);
    CREATE TABLE IF NOT EXISTS shape_nudges (
        shape       TEXT PRIMARY KEY,
        last_at     REAL NOT NULL
    );
    """

    def __init__(self, *, handler: NudgeHandler,
                 db_path: Optional[Path] = None,
                 min_plan_length: int = MIN_PLAN_LENGTH,
                 min_recurrence: int = MIN_RECURRENCE) -> None:
        self._handler = handler
        self._min_len = int(min_plan_length)
        self._min_rec = int(min_recurrence)
        self._lock = threading.RLock()
        self._db_path = db_path or _default_db()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.executescript(self.SCHEMA)
        # Per-process in-flight turn aggregation: turn_id → list of
        # tool names. Cleared when the turn completes or is dropped.
        self._inflight: Dict[str, List[str]] = {}
        self._unsubscribe: Optional[Callable[[], None]] = None

    # ---- bus subscriber ----------------------------------------------

    def attach_to_bus(self, bus: Optional[Any] = None) -> None:
        """Idempotent — re-attach detaches first."""
        self.detach_from_bus()
        from .tool_invocation import global_bus
        target = bus if bus is not None else global_bus()
        self._unsubscribe = target.subscribe(self._on_invocation)

    def detach_from_bus(self) -> None:
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:
                pass
            self._unsubscribe = None

    def _on_invocation(self, inv) -> None:
        # Honor incognito.
        try:
            extra = getattr(inv, "extra", None) or {}
            if isinstance(extra, dict) and extra.get("incognito"):
                return
        except Exception:
            return
        # Only OK invocations contribute to a successful shape.
        if str(getattr(inv, "status", "")) != "ok":
            return
        tool = str(getattr(inv, "tool", "") or "")
        if not tool or tool.startswith("_"):
            # Skip orchestrator pseudo-tools.
            return
        turn_id = str(getattr(inv, "turn_id", "") or "")
        if not turn_id:
            return
        with self._lock:
            self._inflight.setdefault(turn_id, []).append(tool)

    # ---- turn completion ---------------------------------------------

    def complete_turn(self, *, turn_id: str,
                      user_text: str = "") -> None:
        """Called by the orchestrator after a turn finishes. Records
        the shape + checks for recurrence. Best-effort; never
        raises into the caller."""
        if not turn_id:
            return
        try:
            from .incognito import is_incognito
            if is_incognito():
                # Drop the in-flight; never record anything.
                with self._lock:
                    self._inflight.pop(turn_id, None)
                return
        except Exception:
            pass
        with self._lock:
            tools = self._inflight.pop(turn_id, [])
        if len(tools) < self._min_len:
            return
        shape = ">".join(tools)
        now = time.time()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO turn_shapes "
                    "(ts, turn_id, shape, user_text, sample_tools) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (now, turn_id, shape, (user_text or "")[:240],
                     ",".join(tools)),
                )
                self._maybe_prune_history()
                cur = self._conn.execute(
                    "SELECT COUNT(*) FROM turn_shapes WHERE shape=?",
                    (shape,))
                count = int(cur.fetchone()[0])
            except Exception:
                return
        if count >= self._min_rec:
            self._maybe_emit_nudge(shape, count, tools,
                                    user_text)

    def _maybe_emit_nudge(self, shape: str, count: int,
                          tools: List[str],
                          user_text: str) -> None:
        now = time.time()
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT last_at FROM shape_nudges WHERE shape=?",
                    (shape,))
                row = cur.fetchone()
            except Exception:
                row = None
            if row is not None:
                last_at = float(row[0] or 0.0)
                if (now - last_at) < PER_SHAPE_NUDGE_COOLDOWN_SEC:
                    return
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO shape_nudges "
                    "(shape, last_at) VALUES (?, ?)",
                    (shape, now),
                )
            except Exception:
                pass
        suggested = _suggest_skill_name(tools, user_text)
        nudge = ConsolidationNudge(
            shape=shape, occurrence_count=count,
            sample_tools=tools, last_user_text=user_text,
            suggested_skill_name=suggested,
        )
        try:
            self._handler(nudge)
        except Exception:
            pass

    # ---- inspection (used by tests) -----------------------------------

    def shape_count(self, shape: str) -> int:
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT COUNT(*) FROM turn_shapes WHERE shape=?",
                    (shape,))
                return int(cur.fetchone()[0])
            except Exception:
                return 0

    def reset(self) -> None:
        with self._lock:
            try:
                self._conn.execute("DELETE FROM turn_shapes")
                self._conn.execute("DELETE FROM shape_nudges")
            except Exception:
                pass
            self._inflight.clear()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ---- helpers ------------------------------------------------------

    def _maybe_prune_history(self) -> None:
        try:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM turn_shapes")
            n = int(cur.fetchone()[0])
        except Exception:
            return
        if n <= MAX_HISTORY_ROWS:
            return
        # Drop the oldest 10%.
        try:
            self._conn.execute(
                "DELETE FROM turn_shapes WHERE id IN "
                "(SELECT id FROM turn_shapes ORDER BY ts ASC LIMIT ?)",
                (max(1, n // 10),),
            )
        except Exception:
            pass


def _suggest_skill_name(tools: List[str], user_text: str) -> str:
    """Best-effort short name for the suggested skill. Used in the
    nudge body so the user can accept with one click."""
    if user_text:
        clean = user_text.strip().rstrip("?.!,").lower()
        if clean and len(clean) <= 60:
            return clean
    if not tools:
        return "custom skill"
    head = tools[0].split("_")[0]
    return f"{head} workflow"

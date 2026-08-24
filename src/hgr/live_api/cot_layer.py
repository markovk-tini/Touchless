"""Chain-of-thought data layer.

Phase-2. Right now Iris's reasoning is invisible after a turn —
the planner makes a plan, the executor runs it, and the user
sees just the final reply. That's fine when it works. When it
doesn't, the user has no way to ask "why did you call that
tool?" or "what did you consider?". This module persists a
structured reasoning trail for every turn so:

  * "why did you do X?" returns the actual decision steps.
  * "what did you skip?" can list rejected alternatives.
  * post-hoc audits can analyze planner quality without
    re-running.

What gets recorded per turn:

  * `intent`        — the user's literal text + the classifier
                      / planner's inferred goal.
  * `decisions`     — ordered list of `Decision(stage, choice,
                      alternatives, why)`. Stages: 'classify',
                      'route', 'plan', 'revise', 'critique',
                      'speak'.
  * `tool_calls`    — slim refs to ToolInvocations from the bus
                      (id + tool, NOT a full duplicate).
  * `final_message` — what the user actually saw.

Storage: SQLite at `%LOCALAPPDATA%\\Touchless\\private\\cot.db`.
Bounded — each turn caps at ~16KB, the DB itself caps at 250MB
total via a rolling per-turn limit and a wipe-oldest sweep when
near cap.

Privacy: honors incognito mode (no rows persisted when
is_incognito() is True at record time).

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db() -> Path:
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(local) / "Touchless" / "private" / "cot.db"


# Per-turn payload cap. Hard cap so a runaway reasoning loop can't
# write multi-MB rows.
MAX_TURN_PAYLOAD_BYTES = 16 * 1024
# Total DB size cap. When exceeded, oldest 10% of rows are pruned.
MAX_DB_BYTES = 250 * 1024 * 1024


class DecisionStage(str, Enum):
    CLASSIFY = "classify"
    ROUTE = "route"            # ModelRouter pick
    PLAN = "plan"
    REVISE = "revise"          # PlanReviser amendment
    CRITIQUE = "critique"      # SelfCritique augmentation
    SPEAK = "speak"            # synthesis / TTS decision
    REFUSE = "refuse"          # safety gate / spoof defense


@dataclass
class Decision:
    stage: str
    choice: str
    why: str = ""
    alternatives: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


@dataclass
class TurnTrail:
    """One reasoning trail for one user turn."""
    turn_id: str
    user_text: str
    inferred_goal: str = ""
    decisions: List[Decision] = field(default_factory=list)
    tool_call_refs: List[str] = field(default_factory=list)
    final_message: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    error: str = ""

    def add_decision(self, stage: DecisionStage, choice: str,
                     why: str = "", alternatives: Optional[List[str]]
                     = None) -> None:
        self.decisions.append(Decision(
            stage=stage.value if isinstance(stage, DecisionStage)
            else str(stage),
            choice=choice, why=why,
            alternatives=list(alternatives or []),
        ))

    def add_tool_ref(self, invocation_id: str) -> None:
        if invocation_id:
            self.tool_call_refs.append(str(invocation_id))

    def to_json(self) -> str:
        d = {
            "turn_id": self.turn_id,
            "user_text": self.user_text[:1000],
            "inferred_goal": self.inferred_goal[:500],
            "decisions": [asdict(d) for d in self.decisions],
            "tool_call_refs": self.tool_call_refs,
            "final_message": self.final_message[:1500],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": self.error[:500],
        }
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


class ChainOfThoughtLayer:
    """Append-only store for turn reasoning trails."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS turn_trails (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id       TEXT UNIQUE,
        started_at    REAL NOT NULL,
        ended_at      REAL,
        user_text     TEXT NOT NULL,
        inferred_goal TEXT,
        payload_json  TEXT NOT NULL,
        size_bytes    INTEGER NOT NULL,
        error         TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_trails_started_at
        ON turn_trails(started_at DESC);
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
        # F-007 audit: actually back the docstring's "runs at most
        # every N writes" claim with a counter. Without this, every
        # finalize() did a Path.stat() + SELECT COUNT(*) — ~50-200µs
        # per turn on cold filesystems, which adds up in the
        # conversational latency budget.
        self._writes_since_prune = 0

    def start_turn(self, user_text: str,
                   turn_id: Optional[str] = None) -> TurnTrail:
        return TurnTrail(turn_id=turn_id or uuid.uuid4().hex,
                         user_text=str(user_text or ""))

    def finalize(self, trail: TurnTrail, final_message: str = "",
                 error: str = "") -> bool:
        """Persist a completed trail. Honors incognito — returns
        False without writing when incognito is active."""
        from .incognito import is_incognito
        if is_incognito():
            return False
        trail.final_message = str(final_message or "")
        trail.error = str(error or "")
        trail.ended_at = time.time()
        payload = trail.to_json()
        if len(payload.encode("utf-8")) > MAX_TURN_PAYLOAD_BYTES:
            # Drop alternatives + clamp message + clip why text to fit
            # the cap. Reasoning trail must still record SOMETHING —
            # never silently drop, and the persisted payload MUST stay
            # valid JSON (no raw-byte slicing — that would corrupt it).
            for d in trail.decisions:
                d.alternatives = []
                d.why = (d.why or "")[:80]
            trail.final_message = trail.final_message[:200]
            payload = trail.to_json()
            # If STILL over cap, drop all but the first + last 8
            # decisions so we keep the start + end of reasoning.
            if len(payload.encode("utf-8")) > MAX_TURN_PAYLOAD_BYTES:
                if len(trail.decisions) > 16:
                    trail.decisions = (trail.decisions[:8]
                                       + trail.decisions[-8:])
                payload = trail.to_json()
            # Final safety: if the user_text itself was huge, shrink it.
            if len(payload.encode("utf-8")) > MAX_TURN_PAYLOAD_BYTES:
                trail.user_text = trail.user_text[:200]
                trail.decisions = trail.decisions[:4]
                payload = trail.to_json()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO turn_trails"
                "(turn_id, started_at, ended_at, user_text, "
                " inferred_goal, payload_json, size_bytes, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (trail.turn_id, trail.started_at, trail.ended_at,
                 trail.user_text[:1000],
                 trail.inferred_goal[:500],
                 payload, len(payload), trail.error[:500]),
            )
            # Light maintenance: occasionally check size + prune.
            self._maybe_prune()
        return True

    def recent_turns(self, *, limit: int = 20
                     ) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT turn_id, started_at, ended_at, user_text, "
                "inferred_goal, payload_json, error "
                "FROM turn_trails ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                payload = json.loads(r[5])
            except Exception:
                payload = {}
            out.append({
                "turn_id": r[0], "started_at": r[1], "ended_at": r[2],
                "user_text": r[3], "inferred_goal": r[4],
                "payload": payload, "error": r[6],
            })
        return out

    def explain_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """Pull a single turn for the user's 'why did you do X?'
        question. Returns None for unknown turn_id."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT payload_json FROM turn_trails WHERE turn_id=?",
                (turn_id,))
            row = cur.fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def last_turn(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT payload_json FROM turn_trails "
                "ORDER BY started_at DESC LIMIT 1")
            row = cur.fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def wipe(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM turn_trails")
            n = int(cur.fetchone()[0])
            self._conn.execute("DELETE FROM turn_trails")
            return n

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ---- internals ----------------------------------------------------

    PRUNE_CHECK_EVERY_N_WRITES = 100

    def _maybe_prune(self) -> None:
        """If DB has grown past MAX_DB_BYTES, drop the oldest 10% of
        rows. Runs at most once every PRUNE_CHECK_EVERY_N_WRITES
        finalize() calls — without that gate, each turn would pay
        a Path.stat() + SELECT COUNT(*) hit on the hot path."""
        self._writes_since_prune += 1
        if self._writes_since_prune < self.PRUNE_CHECK_EVERY_N_WRITES:
            return
        self._writes_since_prune = 0
        try:
            size = self._db_path.stat().st_size
        except OSError:
            return
        if size < MAX_DB_BYTES:
            return
        cur = self._conn.execute("SELECT COUNT(*) FROM turn_trails")
        n = int(cur.fetchone()[0])
        if n < 10:
            return
        prune_n = max(1, n // 10)
        self._conn.execute(
            "DELETE FROM turn_trails WHERE id IN "
            "(SELECT id FROM turn_trails ORDER BY started_at ASC LIMIT ?)",
            (prune_n,),
        )


# ---- module singleton --------------------------------------------------

_layer: Optional[ChainOfThoughtLayer] = None
_lock = threading.Lock()


def global_cot_layer() -> ChainOfThoughtLayer:
    global _layer
    if _layer is None:
        with _lock:
            if _layer is None:
                _layer = ChainOfThoughtLayer()
    return _layer


def _reset_for_tests() -> None:
    global _layer
    if _layer is not None:
        try:
            _layer.close()
        except Exception:
            pass
    _layer = None

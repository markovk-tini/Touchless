"""Iris skills — user-defined named procedures. A skill is just a saved
Plan with a trigger phrase (e.g. "morning briefing" → a fixed 3-step
plan). Tier 0.5 in the orchestrator: when the user's request matches a
skill trigger, the saved Plan runs directly through the Executor — NO
LLM call, NO classifier work, instant replay.

Backed by SQLite alongside the memory store. Free-text triggers are
matched first by exact-phrase then by token-overlap so "do my morning
briefing" still finds "morning briefing".

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .plan import Plan, Step


_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    trigger     TEXT    NOT NULL,
    plan_json   TEXT    NOT NULL,
    created_ts  REAL    NOT NULL,
    used_ts     REAL,
    use_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_skills_trigger ON skills(trigger);
"""

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def default_skills_path() -> Path:
    override = os.environ.get("TOUCHLESS_SKILLS_DB")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Touchless" / "skills.db"
    return Path.home() / ".touchless" / "skills.db"


class SkillStore:
    """SQLite-backed skill catalog. Same threadsafety model as MemoryStore."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path or default_skills_path())
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self._path), check_same_thread=False, timeout=5.0)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    # ---- writes ---------------------------------------------------------
    def save(self, name: str, trigger: str, plan: Plan) -> int:
        plan_json = json.dumps({
            "goal": plan.goal,
            "final": plan.final,
            "steps": [
                {"id": s.id, "tool": s.tool, "args": s.args,
                 "depends_on": s.depends_on, "description": s.description}
                for s in plan.steps
            ],
        })
        norm_trigger = _normalize(trigger or name)
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO skills (name, trigger, plan_json, created_ts) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "trigger=excluded.trigger, plan_json=excluded.plan_json",
                (name.strip(), norm_trigger, plan_json, time.time()),
            )
            return int(cur.lastrowid or 0)

    def delete(self, name: str) -> bool:
        with self._lock, self._conn() as c:
            cur = c.execute("DELETE FROM skills WHERE name = ?", (name.strip(),))
            return cur.rowcount > 0

    def mark_used(self, name: str) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE skills SET used_ts = ?, use_count = use_count + 1 "
                "WHERE name = ?",
                (time.time(), name.strip()),
            )

    # ---- reads ----------------------------------------------------------
    def list_all(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, trigger, plan_json, created_ts, used_ts, use_count "
                "FROM skills ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def find(self, text: str) -> Optional[Plan]:
        """Try to match `text` to a stored skill. Two passes:
          1. exact normalized match against any trigger
          2. all-trigger-words present in `text` (substring/token containment)
        Returns the most-recently-used match on ties."""
        norm = _normalize(text)
        if not norm:
            return None
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, trigger, plan_json, used_ts FROM skills"
            ).fetchall()
        if not rows:
            return None

        # Pass 1: exact trigger present as a contiguous substring.
        exact_hits = [r for r in rows if r["trigger"] and r["trigger"] in norm]
        if exact_hits:
            exact_hits.sort(key=lambda r: (r["used_ts"] or 0), reverse=True)
            return _row_to_plan(exact_hits[0])

        # Pass 2: every word of the trigger appears somewhere in the text.
        text_tokens = set(norm.split())
        for r in rows:
            tt = set((r["trigger"] or "").split())
            if tt and tt.issubset(text_tokens):
                return _row_to_plan(r)
        return None


def _row_to_plan(row: sqlite3.Row) -> Plan:
    raw = json.loads(row["plan_json"])
    steps = [
        Step(
            id=int(s.get("id") or i + 1),
            tool=str(s.get("tool") or ""),
            args=s.get("args") or {},
            depends_on=[int(x) for x in (s.get("depends_on") or [])],
            description=str(s.get("description") or ""),
        )
        for i, s in enumerate(raw.get("steps") or [])
    ]
    return Plan(goal=str(raw.get("goal") or row["name"]),
                steps=steps,
                final=str(raw.get("final") or "return"))

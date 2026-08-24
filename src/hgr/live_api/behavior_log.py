"""Behavior log — Iris learns from corrections across sessions.

Phase-9 self-improvement. `local_intent` adapts the CLASSIFIER
from raw (utterance → tool) examples. But the user often expresses
preferences AT A HIGHER LEVEL: "no I meant Edge", "use Slack
instead", "always send via Gmail". Those corrections shouldn't
just retune one classifier — they should surface as DEFAULTS the
planner respects on future turns.

This module captures `BehaviorEntry` rows: (utterance, chosen_tool,
chosen_args, outcome). Outcome is either ACCEPTED (no objection
within 2 turns) or CORRECTED (user said no/wrong/different/instead).
After N corrections on the same (utterance_pattern, alternate_tool)
pair, we surface a "switch default" nudge.

SQLite-backed at:
    %LOCALAPPDATA%/Touchless/private/behavior_log.db

Public:
  * `log_dispatch(utterance, tool, args)` — orchestrator calls on
    every successful dispatch.
  * `log_correction(utterance, original_tool, corrected_tool)` —
    when the user corrects an action mid-turn or in a follow-up.
  * `find_pattern_suggestions(min_corrections=3)` — returns a list
    of `SuggestedDefault` rows the UI / nudger can present.
  * `apply_suggestion(pattern, tool)` — persists "always use
    `tool` for `pattern`" as a preference.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _default_db_path() -> Path:
    override = os.environ.get("TOUCHLESS_BEHAVIOR_LOG_DIR")
    if override:
        d = Path(override)
    else:
        local = os.environ.get(
            "LOCALAPPDATA",
            str(Path.home() / "AppData" / "Local"))
        d = Path(local) / "Touchless" / "private"
    d.mkdir(parents=True, exist_ok=True)
    return d / "behavior_log.db"


# ---- patterns ----------------------------------------------------------

_TOKEN_RE = re.compile(r"\b[a-z0-9']+\b")


def _utterance_pattern(text: str) -> str:
    """Crude utterance fingerprint — first 3 lowercase content tokens
    joined. Stable enough to bucket "open chrome" vs "open the
    chrome browser" together."""
    if not text:
        return ""
    tokens = _TOKEN_RE.findall(text.lower())
    stop = {"the", "a", "an", "to", "for", "with", "in", "on",
            "of", "my", "your", "and", "or", "but", "i", "me",
            "you", "is", "are", "do"}
    keep = [t for t in tokens if t not in stop][:3]
    return " ".join(keep)


# ---- data ----------------------------------------------------------

@dataclass
class BehaviorEntry:
    utterance: str
    pattern: str
    tool: str
    args_json: str = "{}"
    outcome: str = "pending"          # 'accepted' / 'corrected'
    corrected_tool: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class SuggestedDefault:
    pattern: str
    original_tool: str
    suggested_tool: str
    correction_count: int
    last_seen: float
    headline: str = ""


# ---- store ----------------------------------------------------------

class BehaviorLog:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        utterance TEXT,
        pattern TEXT,
        tool TEXT,
        args_json TEXT,
        outcome TEXT,
        corrected_tool TEXT,
        ts REAL
    );
    CREATE INDEX IF NOT EXISTS idx_pattern_outcome
        ON entries(pattern, outcome);
    CREATE INDEX IF NOT EXISTS idx_ts ON entries(ts);
    CREATE TABLE IF NOT EXISTS preferences (
        pattern TEXT PRIMARY KEY,
        tool TEXT,
        set_at REAL
    );
    """

    def __init__(self, *, db_path: Optional[Path] = None) -> None:
        self._path = (Path(db_path) if db_path
                      else _default_db_path())
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path),
            isolation_level=None,
            check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(self._SCHEMA)

    def log_dispatch(self, utterance: str, tool: str,
                     args: Optional[Dict[str, Any]] = None) -> int:
        """Returns the row id."""
        pattern = _utterance_pattern(utterance)
        args_json = json.dumps(args or {}, default=str,
                                sort_keys=True)[:1000]
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO entries(utterance, pattern, tool, "
                "args_json, outcome, corrected_tool, ts) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (utterance, pattern, tool, args_json,
                 "pending", "", time.time()))
            return int(cur.lastrowid)

    def mark_accepted(self, entry_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE entries SET outcome='accepted' "
                "WHERE id=? AND outcome='pending'", (entry_id,))
            return cur.rowcount > 0

    def log_correction(self, utterance: str,
                       original_tool: str,
                       corrected_tool: str) -> None:
        pattern = _utterance_pattern(utterance)
        with self._lock:
            # Mark the most recent PENDING matching dispatch as
            # corrected. Don't touch already-corrected rows — that
            # would clobber distinct correction history.
            self._conn.execute(
                "UPDATE entries SET outcome='corrected', "
                "corrected_tool=? "
                "WHERE id = ("
                "  SELECT id FROM entries "
                "  WHERE pattern=? AND tool=? "
                "  AND outcome='pending' "
                "  ORDER BY id DESC LIMIT 1)",
                (corrected_tool, pattern, original_tool))
            # Insert a synthetic "corrected" entry directly so
            # the suggestion engine sees it without depending on
            # prior dispatch logs.
            self._conn.execute(
                "INSERT INTO entries(utterance, pattern, tool, "
                "args_json, outcome, corrected_tool, ts) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (utterance, pattern, original_tool,
                 "{}", "corrected", corrected_tool,
                 time.time()))

    def find_pattern_suggestions(self,
                                  *, min_corrections: int = 3
                                  ) -> List[SuggestedDefault]:
        """Return patterns where the user has corrected from
        original_tool→corrected_tool at least N times. Filters out
        patterns already covered by an explicit preference."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT pattern, tool AS original_tool, "
                "corrected_tool AS suggested_tool, "
                "COUNT(*) AS cnt, MAX(ts) AS last_seen "
                "FROM entries "
                "WHERE outcome = 'corrected' "
                "AND corrected_tool != '' "
                "AND corrected_tool != tool "
                "GROUP BY pattern, tool, corrected_tool "
                "HAVING cnt >= ?",
                (int(min_corrections),))
            rows = cur.fetchall()
            cur2 = self._conn.execute(
                "SELECT pattern, tool FROM preferences")
            applied = {r["pattern"]: r["tool"]
                       for r in cur2.fetchall()}
        out: List[SuggestedDefault] = []
        for r in rows:
            if applied.get(r["pattern"]) == r["suggested_tool"]:
                continue
            out.append(SuggestedDefault(
                pattern=r["pattern"],
                original_tool=r["original_tool"],
                suggested_tool=r["suggested_tool"],
                correction_count=int(r["cnt"]),
                last_seen=float(r["last_seen"]),
                headline=(
                    f"You've corrected '{r['pattern']}' to "
                    f"{r['suggested_tool']} {r['cnt']} times — "
                    "want me to use that as the default?"),
            ))
        return out

    def apply_suggestion(self, pattern: str,
                          tool: str) -> bool:
        if not pattern or not tool:
            return False
        with self._lock:
            self._conn.execute(
                "INSERT INTO preferences(pattern, tool, set_at) "
                "VALUES(?, ?, ?) "
                "ON CONFLICT(pattern) DO UPDATE SET "
                "tool=excluded.tool, set_at=excluded.set_at",
                (pattern, tool, time.time()))
        return True

    def preferred_tool(self, utterance: str) -> Optional[str]:
        """When the user has applied a 'use X for pattern Y'
        preference, return X for matching utterances."""
        pattern = _utterance_pattern(utterance)
        if not pattern:
            return None
        with self._lock:
            cur = self._conn.execute(
                "SELECT tool FROM preferences WHERE pattern = ?",
                (pattern,))
            row = cur.fetchone()
        return row["tool"] if row else None

    def stats(self) -> Dict[str, int]:
        with self._lock:
            d = {}
            cur = self._conn.execute(
                "SELECT outcome, COUNT(*) c FROM entries "
                "GROUP BY outcome")
            for r in cur.fetchall():
                d[r["outcome"]] = int(r["c"])
            cur2 = self._conn.execute(
                "SELECT COUNT(*) c FROM preferences")
            d["preferences"] = int(cur2.fetchone()["c"])
        return d

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ---- singleton ------------------------------------------------------

_lock = threading.Lock()
_singleton: Optional[BehaviorLog] = None


def global_log() -> BehaviorLog:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = BehaviorLog()
        return _singleton


def reset_global() -> None:
    global _singleton
    with _lock:
        if _singleton is not None:
            try:
                _singleton.close()
            except Exception:
                pass
        _singleton = None

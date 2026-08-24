"""Tool reliability ledger.

Phase-2 cognition substrate. Subscribes to the InvocationBus and
maintains a per-tool error history with time decay. The planner +
self-critique loops consult it before retrying or proposing
fallbacks:

  * "ms_mail_send errored 3 times in the last 30 min with 'recipient
    not found' — try the contacts_search dance before retrying."
  * "weather_get on wttr.in failed 4 times in the last 24h — bias
    to the Open-Meteo fallback path."
  * "gmail_send + drive_upload are both failing — the user's network
    is probably down; skip cloud tools for this turn."

Storage: SQLite `%LOCALAPPDATA%\\Touchless\\private\\reliability.db`
with two tables:

    invocation_history(ts, tool, status, error_class, error_text,
                       duration_ms, source)
    -- Capped: keep most recent ~10k per tool, oldest rolls off.

    tool_health(tool, samples, ok_count, error_count, p50_ms,
                last_seen_at, last_error_at, last_error_text)
    -- Materialized rollup. Updated on each invocation. Read-cheap.

We never block a call based on the ledger — it's advisory data for
the planner / critic, not a gate. Reliability data has well-known
false-positive modes (one user's broken Wi-Fi != tool is broken),
and the speed-bump gate is the right place for HARD blocking.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db() -> Path:
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(local) / "Touchless" / "private" / "reliability.db"


_HALF_LIFE_HOURS = 24.0  # decay weighting for "recent" errors


def _decay_weight(age_seconds: float) -> float:
    """Exponential decay so a 1-day-old error contributes half as much
    as a fresh one. Used by error_rate_recent()."""
    hours = age_seconds / 3600.0
    return 0.5 ** (hours / _HALF_LIFE_HOURS)


# Map of exception/error-text fragments to a short error_class string.
# The class is what the planner consults — "transient_network" vs
# "auth_revoked" vs "recipient_invalid" leads to very different
# retries. Add patterns as we see them; default is "other".
_ERROR_CLASSIFIERS = [
    (re.compile(r"\btimed?\s*out|timeout|connection (?:reset|aborted)",
                re.IGNORECASE), "transient_network"),
    (re.compile(r"\b(?:HTTP\s*5\d{2}|502|503|504)\b"), "upstream_5xx"),
    (re.compile(r"\b(?:HTTP\s*4(?:01|03))\b|unauthori[sz]ed|forbidden|"
                r"token (?:expired|revoked|invalid)|"
                r"invalid_grant|access denied",
                re.IGNORECASE), "auth_revoked"),
    (re.compile(r"\b(?:HTTP\s*404|not found)\b", re.IGNORECASE),
     "not_found"),
    (re.compile(r"\bHTTP\s*429|rate.?limit|too many requests",
                re.IGNORECASE), "rate_limited"),
    (re.compile(r"recipient|address.*invalid|placeholder domain",
                re.IGNORECASE), "recipient_invalid"),
    (re.compile(r"unresolved reference|prior step returned no matching",
                re.IGNORECASE), "ref_unresolved"),
    (re.compile(r"user declined|cancelled by user",
                re.IGNORECASE), "user_cancelled"),
    (re.compile(r"precondition_not_met|not_connected",
                re.IGNORECASE), "not_connected"),
    (re.compile(r"scope_missing|insufficient (?:authentication )?scope|"
                r"Google scope '|Connect Google' in Settings",
                re.IGNORECASE), "scope_missing"),
]


def classify_error(error_text: Optional[str]) -> str:
    if not error_text:
        return "other"
    for pat, klass in _ERROR_CLASSIFIERS:
        if pat.search(error_text):
            return klass
    return "other"


class ReliabilityLedger:
    """Reads from the InvocationBus, writes per-tool history + a rollup."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS invocation_history (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ts           REAL NOT NULL,
        tool         TEXT NOT NULL,
        status       TEXT NOT NULL,
        error_class  TEXT,
        error_text   TEXT,
        duration_ms  INTEGER,
        source       TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_hist_tool_ts
        ON invocation_history(tool, ts DESC);

    CREATE TABLE IF NOT EXISTS tool_health (
        tool          TEXT PRIMARY KEY,
        samples       INTEGER NOT NULL DEFAULT 0,
        ok_count      INTEGER NOT NULL DEFAULT 0,
        error_count   INTEGER NOT NULL DEFAULT 0,
        p50_ms        INTEGER,
        last_seen_at  REAL,
        last_error_at REAL,
        last_error_text TEXT
    );
    """

    # Per-tool history cap. ~10k rows / tool is plenty for the
    # planner's "what error did this throw recently" question; older
    # rows roll off so the table doesn't grow unbounded.
    HISTORY_CAP_PER_TOOL = 10_000

    def __init__(self, *, db_path: Optional[Path] = None,
                 bus: Optional[Any] = None,
                 auto_subscribe: bool = True) -> None:
        self._db_path = db_path or _default_db()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.executescript(self.SCHEMA)
        self._closed = False
        self._unsubscribe: Optional[Any] = None
        if auto_subscribe:
            from .tool_invocation import global_bus
            target = bus if bus is not None else global_bus()
            self._unsubscribe = target.subscribe(self._on_invocation)

    # ---- bus subscriber -----------------------------------------------

    def _on_invocation(self, inv) -> None:
        if self._closed:
            return
        # Skip incognito invocations (Phase-1 trust honored at substrate
        # level): no data about tool reliability persists when the user
        # explicitly opted into private mode.
        try:
            if inv.extra.get("incognito"):
                return
        except Exception:
            return
        try:
            self.record(
                tool=inv.tool,
                status=inv.status,
                error_text=inv.error,
                duration_ms=inv.duration_ms,
                source=(inv.source.value if hasattr(inv.source, "value")
                        else str(inv.source)),
            )
        except Exception as exc:  # pragma: no cover - defensive
            import sys
            print(f"[reliability] record failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

    # ---- writes -------------------------------------------------------

    def record(self, *, tool: str, status: str,
               error_text: Optional[str] = None,
               duration_ms: Optional[int] = None,
               source: Optional[str] = None) -> None:
        ts = time.time()
        ec = classify_error(error_text) if status == "error" else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO invocation_history"
                "(ts, tool, status, error_class, error_text, duration_ms, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, tool, status, ec,
                 (error_text or "")[:1000],
                 duration_ms, source),
            )
            # Update health rollup.
            self._conn.execute(
                "INSERT INTO tool_health"
                "(tool, samples, ok_count, error_count, last_seen_at, "
                " last_error_at, last_error_text) "
                "VALUES (?, 1, ?, ?, ?, ?, ?) "
                "ON CONFLICT(tool) DO UPDATE SET "
                "  samples = samples + 1, "
                "  ok_count = ok_count + excluded.ok_count, "
                "  error_count = error_count + excluded.error_count, "
                "  last_seen_at = excluded.last_seen_at, "
                "  last_error_at = COALESCE(excluded.last_error_at, last_error_at), "
                "  last_error_text = COALESCE(excluded.last_error_text, last_error_text)",
                (tool,
                 1 if status == "ok" else 0,
                 1 if status == "error" else 0,
                 ts,
                 ts if status == "error" else None,
                 (error_text or "")[:300] if status == "error" else None),
            )
            # Cap history per-tool.
            self._conn.execute(
                "DELETE FROM invocation_history "
                "WHERE tool=? AND id NOT IN "
                "  (SELECT id FROM invocation_history WHERE tool=? "
                "   ORDER BY ts DESC LIMIT ?)",
                (tool, tool, self.HISTORY_CAP_PER_TOOL),
            )

    # ---- queries used by planner / critic -----------------------------

    def tool_health(self, tool: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT tool, samples, ok_count, error_count, "
                "       last_seen_at, last_error_at, last_error_text "
                "FROM tool_health WHERE tool=?",
                (tool,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "tool": row[0], "samples": row[1],
            "ok_count": row[2], "error_count": row[3],
            "last_seen_at": row[4], "last_error_at": row[5],
            "last_error_text": row[6],
        }

    def recent_errors(self, tool: str, window_sec: float = 1800.0,
                      limit: int = 20) -> List[Dict[str, Any]]:
        """Errors in the last `window_sec` for `tool`. The planner uses
        this to decide retry strategy."""
        cutoff = time.time() - window_sec
        with self._lock:
            cur = self._conn.execute(
                "SELECT ts, status, error_class, error_text, duration_ms "
                "FROM invocation_history WHERE tool=? AND ts>=? "
                "AND status='error' ORDER BY ts DESC LIMIT ?",
                (tool, cutoff, limit),
            )
            rows = cur.fetchall()
        return [{"ts": r[0], "status": r[1], "error_class": r[2],
                 "error_text": r[3], "duration_ms": r[4]} for r in rows]

    def error_rate_recent(self, tool: str,
                          window_sec: float = 3600.0) -> float:
        """Decay-weighted error rate in the last `window_sec`. 0..1.
        Used by the planner to bias away from currently-flaky tools."""
        cutoff = time.time() - window_sec
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "SELECT ts, status FROM invocation_history "
                "WHERE tool=? AND ts>=?",
                (tool, cutoff),
            )
            rows = cur.fetchall()
        if not rows:
            return 0.0
        ok_w = err_w = 0.0
        for ts, status in rows:
            w = _decay_weight(now - ts)
            if status == "error":
                err_w += w
            else:
                ok_w += w
        total = ok_w + err_w
        return (err_w / total) if total > 0 else 0.0

    def error_class_counts(self, tool: str,
                           window_sec: float = 3600.0
                           ) -> Dict[str, int]:
        cutoff = time.time() - window_sec
        with self._lock:
            cur = self._conn.execute(
                "SELECT error_class, COUNT(*) FROM invocation_history "
                "WHERE tool=? AND ts>=? AND status='error' "
                "GROUP BY error_class",
                (tool, cutoff),
            )
            return {row[0] or "other": int(row[1]) for row in cur.fetchall()}

    def is_currently_flaky(self, tool: str,
                           threshold: float = 0.5) -> bool:
        """Convenience: is the recent error rate above `threshold`
        AND the tool has been called at least 3 times in the window?
        Used by 'should I retry vs propose a fallback' decisions."""
        recent_errs = self.recent_errors(tool, window_sec=1800)
        if len(recent_errs) < 3:
            return False
        return self.error_rate_recent(tool, window_sec=1800) >= threshold

    def all_tool_health(self) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT tool, samples, ok_count, error_count, "
                "       last_seen_at, last_error_at, last_error_text "
                "FROM tool_health ORDER BY samples DESC"
            )
            rows = cur.fetchall()
        return [
            {"tool": r[0], "samples": r[1],
             "ok_count": r[2], "error_count": r[3],
             "last_seen_at": r[4], "last_error_at": r[5],
             "last_error_text": r[6]}
            for r in rows
        ]

    def wipe(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM invocation_history")
            n = int(cur.fetchone()[0])
            self._conn.execute("DELETE FROM invocation_history")
            self._conn.execute("DELETE FROM tool_health")
            return n

    def close(self) -> None:
        self._closed = True
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:
                pass
        try:
            self._conn.close()
        except Exception:
            pass


_global_ledger: Optional[ReliabilityLedger] = None
_ledger_lock = threading.Lock()


def global_ledger() -> ReliabilityLedger:
    global _global_ledger
    if _global_ledger is None:
        with _ledger_lock:
            if _global_ledger is None:
                _global_ledger = ReliabilityLedger()
    return _global_ledger

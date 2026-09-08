"""Append-only audit log of every Iris tool invocation.

Phase-1 substrate. Subscribes to the InvocationBus and writes one row
per tool call to a small SQLite file. User-visible (Activity surface
can render it; export can dump it; delete-everything can wipe it).

Storage path: `%LOCALAPPDATA%\\Touchless\\private\\audit.db`. The
private folder convention (NOT under Documents) is per CLAUDE.md
rule #7 — private data must never sync to OneDrive / be readable by
family members browsing the user's Documents.

What's stored: timestamp, tool name, destructiveness, status,
duration, source, blanket-redacted args (known-sensitive arg names
emptied: api_key, token, password, secret, body, etc.).

What's NOT stored:
  * full tool output (potentially huge / contains user content)
  * any data while incognito mode is active (future Phase-1)
  * full args verbatim — secret-shaped fields are redacted

The v1 schema is intentionally minimal. HMAC chaining and per-tool
custom redaction rules are v2 features that should land after real
usage data shows what's needed — shipping them in v1 would be
gold-plating substrate.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tool_invocation import InvocationBus, ToolInvocation, global_bus


# Argument names whose values we BLANKET-REDACT before storing. Phase-1
# protective measure — Phase-2 can replace this with per-tool rules once
# we have signals about what users actually want logged vs not.
_SENSITIVE_ARG_NAMES = frozenset({
    # Auth
    "api_key", "apikey", "token", "auth", "authorization", "bearer",
    "password", "passwd", "secret", "credential", "credentials",
    "client_secret", "refresh_token", "access_token", "personal_access_token",
    "github_personal_access_token", "slack_bot_token", "openai_api_key",
    "anthropic_api_key", "notion_api_key", "linear_api_key",
    "sentry_auth_token", "brave_api_key", "google_maps_api_key",
    "atlassian_api_token",
    # Communication bodies — could contain PII the user doesn't want logged.
    # User can opt back in via a per-tool rule in v2.
    "body", "message", "text",
    # Connection URLs sometimes embed credentials.
    "connection_url", "connection_string", "url_or_query",
})

_REDACTED_PLACEHOLDER = "[redacted]"

# Bounds: args strings get capped so a 10MB clipboard_write doesn't
# bloat the audit log. The full text is still in the tool output (not
# stored here), so this only affects audit-row size.
_MAX_ARG_VALUE_LEN = 300


def _redact_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe copy of `args` with sensitive values replaced
    by `[redacted]` and overlong values truncated."""
    out: Dict[str, Any] = {}
    for k, v in (args or {}).items():
        key_low = str(k).lower()
        if key_low in _SENSITIVE_ARG_NAMES:
            out[k] = _REDACTED_PLACEHOLDER
            continue
        if isinstance(v, str) and len(v) > _MAX_ARG_VALUE_LEN:
            out[k] = v[:_MAX_ARG_VALUE_LEN] + "…[truncated]"
        elif isinstance(v, dict):
            out[k] = _redact_args(v)
        elif isinstance(v, list):
            out[k] = [_redact_args({"_": x})["_"] if isinstance(x, dict)
                      else (x[:_MAX_ARG_VALUE_LEN] + "…[truncated]"
                            if isinstance(x, str) and len(x) > _MAX_ARG_VALUE_LEN
                            else x)
                      for x in v]
        else:
            out[k] = v
    return out


def _default_db_path() -> Path:
    """Per CLAUDE.md rule #7: private data goes under %LOCALAPPDATA%
    (NOT Documents, which syncs to OneDrive). Falls back gracefully on
    non-Windows."""
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    base = Path(local) / "Touchless" / "private"
    return base / "audit.db"


class AuditLog:
    """Append-only SQLite log of ToolInvocations. Thread-safe (Python's
    sqlite3 module enforces single-thread access via check_same_thread;
    we wrap with a lock to make publish-from-any-thread safe).

    Subscribes to a global InvocationBus on construction; unsubscribe
    is handled by ``close()``. Designed to be a singleton per process —
    a second instance pointed at the same DB will fight on writes."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS invocations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        invocation_id   TEXT NOT NULL,
        turn_id         TEXT,
        ts              REAL NOT NULL,
        ended_ts        REAL,
        duration_ms     INTEGER,
        tool            TEXT NOT NULL,
        source          TEXT NOT NULL,
        destructiveness TEXT NOT NULL,
        reversible      INTEGER NOT NULL,
        was_confirmed   INTEGER NOT NULL,
        was_undone      INTEGER NOT NULL DEFAULT 0,
        status          TEXT NOT NULL,
        error           TEXT,
        args_json       TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_invocations_ts
        ON invocations(ts);
    CREATE INDEX IF NOT EXISTS idx_invocations_tool
        ON invocations(tool);
    CREATE INDEX IF NOT EXISTS idx_invocations_turn
        ON invocations(turn_id);
    """

    def __init__(self, *,
                 db_path: Optional[Path] = None,
                 bus: Optional[InvocationBus] = None,
                 auto_subscribe: bool = True) -> None:
        self._db_path = db_path or _default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._bus = bus or global_bus()
        # RLock (re-entrant) — wipe() calls count() under the lock,
        # which itself takes the lock. A plain Lock deadlocks.
        self._lock = threading.RLock()
        # check_same_thread=False because we serialize via _lock above;
        # the alternative (per-thread connections) is more state to
        # manage for no real win at this write rate.
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we group with explicit BEGIN
        )
        self._conn.executescript(self.SCHEMA)
        self._unsubscribe: Optional[Any] = None
        self._closed = False
        if auto_subscribe:
            self._unsubscribe = self._bus.subscribe(self._on_invocation)

    # ---- bus callback -------------------------------------------------

    def _on_invocation(self, inv: ToolInvocation) -> None:
        if self._closed:
            return
        # HARD privacy gate: incognito mode means NO row is written.
        # The bus tags invocations during a private session; we drop
        # them here before any sqlite call sees them. Integration test
        # in test_invocation.py asserts zero rows written during
        # incognito — this guard is what makes that pass.
        try:
            if inv.extra.get("incognito"):
                return
        except Exception:
            # If we can't even read .extra, bias safe: don't write.
            return
        try:
            self.record(inv)
        except Exception as exc:  # pragma: no cover - defensive
            import sys
            print(f"[audit_log] record failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    # ---- writes -------------------------------------------------------

    def record(self, inv: ToolInvocation) -> int:
        """Write one row. Returns the row's primary key."""
        args_redacted = _redact_args(inv.args)
        # JSON-serialize args defensively — exotic types (Path, bytes)
        # get str()'d so a single bad tool can't break the audit log.
        try:
            args_json = json.dumps(args_redacted, default=str,
                                   ensure_ascii=False)
        except Exception:
            args_json = json.dumps({"_serialize_error": str(args_redacted)})
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO invocations "
                "(invocation_id, turn_id, ts, ended_ts, duration_ms, "
                " tool, source, destructiveness, reversible, "
                " was_confirmed, was_undone, status, error, args_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    inv.invocation_id, inv.turn_id, inv.started_at,
                    inv.ended_at or None,
                    inv.duration_ms if inv.ended_at else None,
                    inv.tool,
                    inv.source.value if hasattr(inv.source, "value")
                    else str(inv.source),
                    (inv.destructiveness.value
                     if hasattr(inv.destructiveness, "value")
                     else str(inv.destructiveness)),
                    1 if inv.reversible else 0,
                    1 if inv.was_confirmed else 0,
                    1 if inv.was_undone else 0,
                    inv.status,
                    inv.error,
                    args_json,
                ),
            )
            return int(cur.lastrowid or 0)

    def mark_undone(self, invocation_id: str) -> bool:
        """Future-Phase-1 universal-undo speed bump hook. Called when
        the user reverses a prior action. Returns True if a row was
        updated."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE invocations SET was_undone = 1 "
                "WHERE invocation_id = ?",
                (invocation_id,),
            )
            return cur.rowcount > 0

    # ---- reads (for activity pill / export) ---------------------------

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Most recent N invocations, newest first."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, invocation_id, turn_id, ts, ended_ts, "
                "       duration_ms, tool, source, destructiveness, "
                "       reversible, was_confirmed, was_undone, status, "
                "       error, args_json "
                "FROM invocations ORDER BY ts DESC LIMIT ?",
                (int(limit),),
            )
            rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM invocations")
            return int(cur.fetchone()[0])

    def by_turn(self, turn_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, invocation_id, turn_id, ts, ended_ts, "
                "       duration_ms, tool, source, destructiveness, "
                "       reversible, was_confirmed, was_undone, status, "
                "       error, args_json "
                "FROM invocations WHERE turn_id = ? ORDER BY ts",
                (turn_id,),
            )
            rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: tuple) -> Dict[str, Any]:
        keys = ("id", "invocation_id", "turn_id", "ts", "ended_ts",
                "duration_ms", "tool", "source", "destructiveness",
                "reversible", "was_confirmed", "was_undone", "status",
                "error", "args_json")
        out = dict(zip(keys, row))
        try:
            out["args"] = json.loads(out.pop("args_json"))
        except Exception:
            out["args"] = {}
            out.pop("args_json", None)
        out["reversible"] = bool(out["reversible"])
        out["was_confirmed"] = bool(out["was_confirmed"])
        out["was_undone"] = bool(out["was_undone"])
        return out

    # ---- delete-everything kill-switch hook ---------------------------

    def wipe(self) -> int:
        """Delete every row. Returns the count deleted. For the
        delete-everything kill switch + tests.

        Deliberately omits VACUUM — on Windows sqlite under certain AV
        configurations VACUUM can stall for many seconds, and the
        space reclaim isn't worth blocking the user's privacy-reset
        button. Subsequent INSERTs will reuse the freed pages
        automatically."""
        with self._lock:
            before = self.count()
            self._conn.execute("DELETE FROM invocations")
            return before

    # ---- lifecycle ----------------------------------------------------

    def close(self) -> None:
        self._closed = True
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:
                pass
            self._unsubscribe = None
        try:
            self._conn.close()
        except Exception:
            pass

    @property
    def db_path(self) -> Path:
        return self._db_path

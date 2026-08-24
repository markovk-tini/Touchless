"""MCP server trust boundary.

Phase-1 substrate. Lets the user grant per-MCP-server consent BEFORE
its tools become callable, and pessimistically defaults every MCP
tool's destructiveness to DESTRUCTIVE (so the speed-bump gate
catches them) until the user explicitly grants the server "trusted"
status.

Why this matters: an MCP server can ship dozens of tools whose
intent isn't legible from the schema alone. Filesystem MCP can
delete files. GitHub MCP can publish gists. Slack MCP can post in
channels. The catalog has no notion of "this tool sends; that one
only reads". Treating every MCP tool as DESTRUCTIVE-by-default
means a user who connects a new MCP server doesn't surprise-execute
a destructive call on first invocation — the safety gate fires, the
user sees what's about to happen, accepts (or denies) it.

Trust granularity:
  * `pending`   — never used yet, NOT exposed in tool list.
                  First exposure path: user enables in MCP picker.
  * `trusted_read`   — user OK'd read-only access. Read tools exposed
                       at their declared destructiveness; write/delete
                       tools STILL flagged DESTRUCTIVE+confirm.
  * `trusted_write`  — user OK'd writes. Reads + writes exposed as
                       declared. Destructive tools still confirm.
  * `trusted_full`   — user OK'd everything. Destructive tools also
                       bypass the gate (still audited).

Storage: SQLite at `%LOCALAPPDATA%\\Touchless\\private\\mcp_trust.db`.
First-time consent UI lives in the picker module.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class TrustLevel(str, Enum):
    PENDING = "pending"
    TRUSTED_READ = "trusted_read"
    TRUSTED_WRITE = "trusted_write"
    TRUSTED_FULL = "trusted_full"
    REVOKED = "revoked"


def _default_db() -> Path:
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(local) / "Touchless" / "private" / "mcp_trust.db"


class McpTrustStore:
    """Per-server trust record. Thread-safe."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS mcp_servers (
        server_id    TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        trust_level  TEXT NOT NULL DEFAULT 'pending',
        granted_at   REAL,
        revoked_at   REAL,
        tool_count   INTEGER NOT NULL DEFAULT 0,
        notes        TEXT
    );
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

    def register(self, server_id: str, display_name: str,
                 tool_count: int = 0) -> str:
        """First-time registration. Returns current trust level. If
        already present, only refreshes display_name + tool_count."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT trust_level FROM mcp_servers WHERE server_id=?",
                (server_id,))
            row = cur.fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO mcp_servers"
                    "(server_id, display_name, trust_level, tool_count) "
                    "VALUES (?, ?, ?, ?)",
                    (server_id, display_name, TrustLevel.PENDING.value,
                     tool_count))
                return TrustLevel.PENDING.value
            self._conn.execute(
                "UPDATE mcp_servers SET display_name=?, tool_count=? "
                "WHERE server_id=?",
                (display_name, tool_count, server_id))
            return str(row[0])

    def grant(self, server_id: str, level: TrustLevel,
              notes: Optional[str] = None) -> bool:
        """User-side grant. Records granted_at and clears revoked_at."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE mcp_servers SET trust_level=?, granted_at=?, "
                "revoked_at=NULL, notes=? WHERE server_id=?",
                (level.value, time.time(), notes, server_id))
            return cur.rowcount > 0

    def revoke(self, server_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE mcp_servers SET trust_level=?, revoked_at=? "
                "WHERE server_id=?",
                (TrustLevel.REVOKED.value, time.time(), server_id))
            return cur.rowcount > 0

    def trust_level(self, server_id: str) -> TrustLevel:
        with self._lock:
            cur = self._conn.execute(
                "SELECT trust_level FROM mcp_servers WHERE server_id=?",
                (server_id,))
            row = cur.fetchone()
        if row is None:
            return TrustLevel.PENDING
        try:
            return TrustLevel(row[0])
        except ValueError:
            return TrustLevel.PENDING

    def is_callable(self, server_id: str) -> bool:
        """Can the planner / realtime call this server's tools right
        now? True for any trusted_*; False for pending / revoked."""
        lvl = self.trust_level(server_id)
        return lvl in (TrustLevel.TRUSTED_READ,
                       TrustLevel.TRUSTED_WRITE,
                       TrustLevel.TRUSTED_FULL)

    def all_servers(self) -> List[Dict[str, object]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT server_id, display_name, trust_level, "
                "granted_at, revoked_at, tool_count, notes "
                "FROM mcp_servers ORDER BY display_name")
            rows = cur.fetchall()
        return [
            {
                "server_id": r[0],
                "display_name": r[1],
                "trust_level": r[2],
                "granted_at": r[3],
                "revoked_at": r[4],
                "tool_count": r[5],
                "notes": r[6],
            }
            for r in rows
        ]

    def wipe_all(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM mcp_servers")
            n = int(cur.fetchone()[0])
            self._conn.execute("DELETE FROM mcp_servers")
            return n

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# Module-level singleton for the real app
_global_store: Optional[McpTrustStore] = None
_store_lock = threading.Lock()


def global_store() -> McpTrustStore:
    global _global_store
    if _global_store is None:
        with _store_lock:
            if _global_store is None:
                _global_store = McpTrustStore()
    return _global_store


# ---- gate -----------------------------------------------------------------


def gate_mcp_call(server_id: str, *, tool_destructiveness: str) -> bool:
    """Combined check used by the MCP connector wrapper before
    dispatching a tool call. Returns True when the call may proceed.

    Decision matrix:
        server=PENDING/REVOKED → False (always)
        server=TRUSTED_READ    → True only if tool is read-only
        server=TRUSTED_WRITE   → True for read + write; destructive
                                  still goes through the speed-bump
                                  gate (returns True here; the gate
                                  in safety_gate.py prompts)
        server=TRUSTED_FULL    → True
    """
    store = global_store()
    level = store.trust_level(server_id)
    if level in (TrustLevel.PENDING, TrustLevel.REVOKED):
        return False
    if level == TrustLevel.TRUSTED_READ:
        return tool_destructiveness in ("read",)
    return True


def required_destructiveness_for_mcp_tool(tool_destructiveness: str
                                          ) -> str:
    """Inheritance rule for MCP-registered tools: bump everything by
    one tier. Read → write, write → destructive, destructive →
    irreversible. Means the safety gate fires more aggressively on
    MCP-sourced tools than on first-party connectors until per-server
    overrides are added. Untrusted by default."""
    bump = {
        "read": "write",
        "write": "destructive",
        "destructive": "irreversible",
        "irreversible": "irreversible",
    }
    return bump.get(tool_destructiveness, "destructive")

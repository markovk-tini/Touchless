"""tool_call_log — lightweight SQLite log of tool dispatches grouped by
session, used to compute cross-tool co-occurrence edges for the cortex
visualization.

Design:

  - **One row per tool fire.** ``(id, session_id, tool_id, ts)``.
    Append-only; reads happen out-of-band when the cortex / simulator
    asks for the cooccurrence aggregate.
  - **Session_id = stable per-process UUID.** Derived once at module
    import; callers can override via ``set_session_id(...)`` (the live
    api manager passes its own session id at startup).
  - **Best-effort, never blocks tool dispatch.** Every public write
    wraps in try/except. A logging failure must NEVER bubble up to
    the tool executor.
  - **Lives at appdata/Touchless/tool_call_log.db** — same parent
    directory as ``cortex_world.json`` so all cortex persistence sits
    in one place. Override via ``TOUCHLESS_TOOL_CALL_LOG`` env var.

The aggregation step (``compute_cooccurrence``) is a separate read
pass — designed to be called once at simulator launch / cortex window
open. It is O(N²) per session but N is tiny (typical session = 5–30
tool calls), so even with hundreds of stored sessions it stays
sub-millisecond.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


# Process-stable default session id. Overridden by set_session_id()
# when LiveApiManager has its own id to thread through.
_session_id: str = uuid.uuid4().hex
_session_lock = threading.Lock()

# DB lock — sqlite3's own locking is fine for short writes, but we
# serialize on the Python side so concurrent threads don't open
# multiple connections in parallel + step on each other's cursors.
_db_lock = threading.Lock()
_db_ready = False  # one-time schema init flag


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[tool-call-log {time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def default_log_path() -> Path:
    """Mirror ``world_state.default_world_path()`` — same parent dir."""
    override = os.environ.get("TOUCHLESS_TOOL_CALL_LOG")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Touchless" / "tool_call_log.db"
    return Path.home() / ".touchless" / "tool_call_log.db"


def set_session_id(session_id: str) -> None:
    """Override the process default session id. Called by LiveApiManager
    once its own session id is known so co-occurrence buckets align
    with the session the user perceives."""
    global _session_id
    if not session_id:
        return
    with _session_lock:
        _session_id = str(session_id)


def current_session_id() -> str:
    with _session_lock:
        return _session_id


def _open(path: Optional[Path] = None) -> sqlite3.Connection:
    """Open the SQLite DB and lazily create the schema on first use."""
    global _db_ready
    db_path = path or default_log_path()
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(str(db_path), timeout=2.0)
    if not _db_ready:
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tool_call_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool_id    TEXT NOT NULL,
                    ts         REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_tool_call_session
                    ON tool_call_log(session_id);
                CREATE INDEX IF NOT EXISTS ix_tool_call_ts
                    ON tool_call_log(ts);
                """
            )
            conn.commit()
            _db_ready = True
        except Exception as exc:
            _log(f"schema init failed: {exc}")
    return conn


def record_tool_call(
    tool_id: str,
    *,
    session_id: Optional[str] = None,
    ts: Optional[float] = None,
    path: Optional[Path] = None,
) -> None:
    """Append a single tool-call row. Best-effort; swallows all errors."""
    if not tool_id:
        return
    sid = session_id or current_session_id()
    ts_val = float(ts) if ts is not None else time.time()
    try:
        with _db_lock:
            conn = _open(path)
            try:
                conn.execute(
                    "INSERT INTO tool_call_log (session_id, tool_id, ts) VALUES (?, ?, ?)",
                    (sid, str(tool_id), ts_val),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as exc:
        _log(f"record_tool_call failed ({tool_id}): {exc}")


def load_sessions(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read all rows, grouped by session_id.

    Returns ``[{session_id, tool_ids: [...]}, ...]`` ordered by the
    session's earliest ts. ``tool_ids`` preserves dispatch order.
    """
    db_path = path or default_log_path()
    if not db_path.exists():
        return []
    sessions: Dict[str, List[str]] = {}
    earliest: Dict[str, float] = {}
    try:
        with _db_lock:
            conn = _open(db_path)
            try:
                cursor = conn.execute(
                    "SELECT session_id, tool_id, ts FROM tool_call_log "
                    "ORDER BY ts ASC"
                )
                for row in cursor:
                    sid, tid, ts = row[0], row[1], float(row[2])
                    sessions.setdefault(sid, []).append(tid)
                    if sid not in earliest or ts < earliest[sid]:
                        earliest[sid] = ts
            finally:
                conn.close()
    except Exception as exc:
        _log(f"load_sessions failed: {exc}")
        return []
    out: List[Dict[str, Any]] = []
    for sid, tools in sessions.items():
        out.append({
            "session_id": sid,
            "earliest_ts": earliest.get(sid, 0.0),
            "tool_ids": tools,
        })
    out.sort(key=lambda s: s["earliest_ts"])
    return out


def compute_cooccurrence(
    *,
    min_weight: int = 2,
    max_edges: int = 60,
    path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Aggregate tool co-occurrence across all stored sessions.

    Weight = number of distinct sessions in which a (tool_a, tool_b)
    pair both fired. Pairs below ``min_weight`` are filtered out.
    Returns at most ``max_edges`` entries, sorted by weight desc, in
    the shape ``[{from, to, weight}, ...]``.

    Symmetric: pair (a, b) and (b, a) collapse to one edge with
    ``from < to`` lexicographically so the JS side can dedup by id.
    """
    sessions = load_sessions(path)
    if not sessions:
        return []
    counts: Dict[tuple, int] = {}
    for sess in sessions:
        # Dedup per-session (a tool repeated 50× in one session counts
        # once toward each pair). Preserves order for tie-break only.
        seen: List[str] = []
        seen_set: set = set()
        for tid in sess["tool_ids"]:
            if tid not in seen_set:
                seen_set.add(tid)
                seen.append(tid)
        for i, a in enumerate(seen):
            for b in seen[i + 1:]:
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                counts[key] = counts.get(key, 0) + 1
    links: List[Dict[str, Any]] = []
    for (a, b), w in counts.items():
        if w >= int(min_weight):
            links.append({"from": a, "to": b, "weight": int(w)})
    links.sort(key=lambda x: (-x["weight"], x["from"], x["to"]))
    if max_edges and len(links) > max_edges:
        links = links[: int(max_edges)]
    return links

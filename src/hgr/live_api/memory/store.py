"""SQLite-backed memory store. Two tables:

  episodic — one row per planner-handled turn:
      (id, ts, user_text, plan_json, steps_json, outcome, embedding)
  semantic — extracted facts, e.g. "Dani = dani@mangollc.org":
      (id, ts, kind, key, value, source); unique on (kind, key, value)

Embeddings stored as raw float32 bytes (numpy-compatible). The store is
embedding-shape-agnostic — whatever the embedder produces gets written
verbatim and compared with cosine similarity at recall time.

Thread-safe via a per-instance lock + per-call connection (sqlite3 is
fine across threads, but connections are not shareable). Cheap: each
write is one INSERT.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EpisodicRow:
    id: int
    ts: float
    user_text: str
    plan_json: Optional[str]
    steps_json: Optional[str]
    outcome: Optional[str]
    embedding: List[float] = field(default_factory=list)


@dataclass
class SemanticRow:
    id: int
    ts: float
    kind: str
    key: str
    value: str
    source: Optional[str] = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    user_text   TEXT    NOT NULL,
    plan_json   TEXT,
    steps_json  TEXT,
    outcome     TEXT,
    embedding   BLOB
);
CREATE INDEX IF NOT EXISTS idx_episodic_ts ON episodic(ts);

CREATE TABLE IF NOT EXISTS semantic (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     REAL    NOT NULL,
    kind   TEXT    NOT NULL,
    key    TEXT    NOT NULL,
    value  TEXT    NOT NULL,
    source TEXT,
    UNIQUE(kind, key, value) ON CONFLICT REPLACE
);
CREATE INDEX IF NOT EXISTS idx_semantic_kind_key ON semantic(kind, key);
"""


def _encode_vec(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_vec(blob: Optional[bytes]) -> List[float]:
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class MemoryStore:
    """SQLite memory store. All operations grab a fresh connection so the
    store can be used safely from multiple threads."""

    # Hard cap on episodic rows; oldest evicted on overflow. Keeps memory
    # bounded over years of use.
    MAX_EPISODIC_ROWS = 10_000

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        # check_same_thread=False so the store can be called from worker
        # threads (we serialize writes via _lock anyway).
        c = sqlite3.connect(str(self._path), check_same_thread=False, timeout=5.0)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    # ---- writes -----------------------------------------------------------
    def add_episodic(self, user_text: str, plan_json: Optional[str],
                     steps_json: Optional[str], outcome: Optional[str],
                     embedding: List[float]) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO episodic (ts, user_text, plan_json, steps_json, outcome, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), user_text, plan_json, steps_json, outcome,
                 _encode_vec(embedding)),
            )
            new_id = int(cur.lastrowid or 0)
            # Trim if we've exceeded the cap.
            c.execute(
                "DELETE FROM episodic WHERE id IN ("
                "  SELECT id FROM episodic ORDER BY ts ASC "
                "  LIMIT max(0, (SELECT COUNT(*) FROM episodic) - ?))",
                (self.MAX_EPISODIC_ROWS,),
            )
            return new_id

    # Kinds for which (kind, key) is unique — last value written wins.
    # Preferences are the obvious case ('default_send_via' should never have
    # two simultaneous values). Aliases too ('preferred_name' is a single
    # value at a time). Other kinds keep the existing UNIQUE(kind,key,value)
    # semantics so e.g. one person can have multiple email addresses.
    _SINGLE_VALUE_KINDS = frozenset({"preference", "alias"})

    def add_semantic(self, kind: str, key: str, value: str,
                     source: Optional[str] = None) -> None:
        if not (kind and key and value):
            return
        with self._lock, self._conn() as c:
            if kind in self._SINGLE_VALUE_KINDS:
                # Last-write-wins: delete any prior value for this key
                # before inserting the new one.
                c.execute("DELETE FROM semantic WHERE kind = ? AND key = ?",
                          (kind, key.lower()))
            c.execute(
                "INSERT OR REPLACE INTO semantic (ts, kind, key, value, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), kind, key.lower(), value, source),
            )

    def delete_facts(self, kind: Optional[str] = None,
                     key: Optional[str] = None,
                     value: Optional[str] = None) -> int:
        """Delete semantic rows matching the given filters. Returns the
        number of rows removed. At least one filter must be non-None to
        avoid wiping the whole table by accident — use clear() for that."""
        if kind is None and key is None and value is None:
            return 0
        sql = "DELETE FROM semantic WHERE 1=1"
        args: List[Any] = []
        if kind is not None:
            sql += " AND kind = ?"
            args.append(kind)
        if key is not None:
            sql += " AND key = ?"
            args.append(key.lower())
        if value is not None:
            sql += " AND value = ?"
            args.append(value)
        with self._lock, self._conn() as c:
            cur = c.execute(sql, args)
            return cur.rowcount or 0

    def add_facts(self, facts: List[Tuple[str, str, str, Optional[str]]]) -> None:
        """Bulk semantic insert. `facts` is [(kind, key, value, source), ...]."""
        if not facts:
            return
        now = time.time()
        with self._lock, self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO semantic (ts, kind, key, value, source) "
                "VALUES (?, ?, ?, ?, ?)",
                [(now, k, key.lower(), v, src) for (k, key, v, src) in facts],
            )

    # ---- reads ------------------------------------------------------------
    def list_episodic(self, limit: int = 200) -> List[EpisodicRow]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM episodic ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_episodic(r) for r in rows]

    def find_facts(self, kind: Optional[str] = None,
                   key: Optional[str] = None,
                   limit: int = 50) -> List[SemanticRow]:
        sql = "SELECT * FROM semantic WHERE 1=1"
        args: List[Any] = []
        if kind is not None:
            sql += " AND kind = ?"
            args.append(kind)
        if key is not None:
            sql += " AND key = ?"
            args.append(key.lower())
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [_row_to_semantic(r) for r in rows]

    def count(self) -> Dict[str, int]:
        with self._conn() as c:
            ep = c.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
            se = c.execute("SELECT COUNT(*) FROM semantic").fetchone()[0]
        return {"episodic": int(ep), "semantic": int(se)}

    def clear(self) -> None:
        """Wipe everything. User-callable via the (future) memory UI."""
        with self._lock, self._conn() as c:
            c.executescript("DELETE FROM episodic; DELETE FROM semantic;")


# ---- helpers ---------------------------------------------------------------
def _row_to_episodic(r: sqlite3.Row) -> EpisodicRow:
    return EpisodicRow(
        id=int(r["id"]),
        ts=float(r["ts"]),
        user_text=str(r["user_text"]),
        plan_json=r["plan_json"],
        steps_json=r["steps_json"],
        outcome=r["outcome"],
        embedding=_decode_vec(r["embedding"]),
    )


def _row_to_semantic(r: sqlite3.Row) -> SemanticRow:
    return SemanticRow(
        id=int(r["id"]),
        ts=float(r["ts"]),
        kind=str(r["kind"]),
        key=str(r["key"]),
        value=str(r["value"]),
        source=r["source"],
    )

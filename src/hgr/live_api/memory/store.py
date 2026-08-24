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
    # Provenance: where this fact came from and when it was extracted.
    # source_kind is a coarse enum ('user_said', 'conversation',
    # 'planner_step', 'legacy', ...). source_id is an optional finer
    # pointer (e.g. an episodic row id, a turn id, or a comma-separated
    # list after consolidation). extracted_at is the wall-clock when
    # extraction happened — distinct from `ts` which is the insert time.
    source_kind: str = "legacy"
    source_id: Optional[str] = None
    extracted_at: Optional[float] = None


# Provenance enum for source_kind. Keep this open — additions are fine,
# but stay within this vocabulary so the cortex grouping UI doesn't have
# to handle arbitrary strings.
SOURCE_KIND_USER_SAID = "user_said"
SOURCE_KIND_CONVERSATION = "conversation"
SOURCE_KIND_PLANNER_STEP = "planner_step"
SOURCE_KIND_LEGACY = "legacy"


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
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    kind         TEXT    NOT NULL,
    key          TEXT    NOT NULL,
    value        TEXT    NOT NULL,
    source       TEXT,
    source_kind  TEXT    DEFAULT 'legacy',
    source_id    TEXT,
    extracted_at REAL,
    UNIQUE(kind, key, value) ON CONFLICT REPLACE
);
CREATE INDEX IF NOT EXISTS idx_semantic_kind_key ON semantic(kind, key);
-- Note: idx_semantic_source_kind is created inside add_columns_if_missing()
-- AFTER the ALTER TABLE adds the column, so legacy DBs don't blow up here.
"""


# Columns added to the semantic table after v1 of the schema. Each is
# applied via ALTER TABLE only if not already present. Order matters
# only for readability — sqlite is happy either way.
_SEMANTIC_MIGRATIONS: List[Tuple[str, str]] = [
    ("source_kind", "ALTER TABLE semantic ADD COLUMN source_kind TEXT DEFAULT 'legacy'"),
    ("source_id", "ALTER TABLE semantic ADD COLUMN source_id TEXT"),
    ("extracted_at", "ALTER TABLE semantic ADD COLUMN extracted_at REAL"),
]


def add_columns_if_missing(conn: sqlite3.Connection) -> List[str]:
    """Idempotent schema migration. Adds the provenance columns to the
    semantic table when they don't already exist. Returns the list of
    columns actually added (empty when already migrated).

    Safe to call repeatedly; safe on both brand-new databases (where
    the CREATE TABLE already defined the columns) and legacy databases
    (where the columns are missing and need an ALTER TABLE)."""
    cur = conn.execute("PRAGMA table_info(semantic)")
    existing = {row[1] for row in cur.fetchall()}
    added: List[str] = []
    for col_name, ddl in _SEMANTIC_MIGRATIONS:
        if col_name in existing:
            continue
        try:
            conn.execute(ddl)
            added.append(col_name)
        except sqlite3.OperationalError:
            # Another thread / process raced us — column already exists.
            # Re-read the table info so we don't double-attempt.
            cur = conn.execute("PRAGMA table_info(semantic)")
            existing = {row[1] for row in cur.fetchall()}
    # Ensure the source_kind index exists on legacy DBs (CREATE INDEX
    # in _SCHEMA already covers fresh DBs; this no-ops when present).
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_source_kind "
            "ON semantic(source_kind)"
        )
    except sqlite3.OperationalError:
        pass
    # Backfill: legacy rows have NULL source_kind. Tag them as 'legacy'
    # so query paths can group by provenance without special-casing NULL.
    try:
        conn.execute(
            "UPDATE semantic SET source_kind = 'legacy' "
            "WHERE source_kind IS NULL"
        )
    except sqlite3.OperationalError:
        pass
    return added


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
            # Apply column-level migrations for legacy databases. New
            # databases pick up the columns from CREATE TABLE above;
            # the helper is a no-op in that case.
            add_columns_if_missing(c)

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
                     source: Optional[str] = None,
                     source_kind: Optional[str] = None,
                     source_id: Optional[str] = None,
                     extracted_at: Optional[float] = None) -> None:
        if not (kind and key and value):
            return
        now = time.time()
        # Default extracted_at to write-time so callers who don't know
        # the real extraction time still get a usable timestamp.
        ext_at = extracted_at if extracted_at is not None else now
        sk = source_kind or "unknown"
        with self._lock, self._conn() as c:
            if kind in self._SINGLE_VALUE_KINDS:
                # Last-write-wins: delete any prior value for this key
                # before inserting the new one.
                c.execute("DELETE FROM semantic WHERE kind = ? AND key = ?",
                          (kind, key.lower()))
            c.execute(
                "INSERT OR REPLACE INTO semantic "
                "(ts, kind, key, value, source, source_kind, source_id, extracted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (now, kind, key.lower(), value, source, sk, source_id, ext_at),
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

    def add_facts(self, facts: List[Tuple[str, str, str, Optional[str]]],
                  source_kind: Optional[str] = None,
                  source_id: Optional[str] = None,
                  extracted_at: Optional[float] = None) -> None:
        """Bulk semantic insert. `facts` is [(kind, key, value, source), ...].
        Optional ``source_kind`` / ``source_id`` / ``extracted_at`` apply
        to every fact in the batch — convenient when an entire turn
        shares the same provenance (e.g. one planner step)."""
        if not facts:
            return
        now = time.time()
        ext_at = extracted_at if extracted_at is not None else now
        sk = source_kind or "unknown"
        with self._lock, self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO semantic "
                "(ts, kind, key, value, source, source_kind, source_id, extracted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(now, k, key.lower(), v, src, sk, source_id, ext_at)
                 for (k, key, v, src) in facts],
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
        """Recall semantic facts matching (kind, key). Phase-3:
        rows whose `source_kind == 'auto_pin'` (written by
        `memory_pinning.maybe_pin_fact` after a fact has been
        independently mentioned ≥3 times) are returned FIRST so
        the planner's compose-rewrite + lookup paths get the
        user's most-reinforced value before any one-off mentions."""
        sql = "SELECT * FROM semantic WHERE 1=1"
        args: List[Any] = []
        if kind is not None:
            sql += " AND kind = ?"
            args.append(kind)
        if key is not None:
            sql += " AND key = ?"
            args.append(key.lower())
        # Sort: auto-pinned rows first (source_kind = 'auto_pin'),
        # then by recency desc. SQLite's CASE in ORDER BY is
        # deterministic and cheap.
        sql += (" ORDER BY CASE WHEN source_kind = 'auto_pin' "
                "THEN 0 ELSE 1 END, ts DESC LIMIT ?")
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [_row_to_semantic(r) for r in rows]

    def list_facts_by_source(self, source_kind: str,
                             limit: int = 200) -> List[SemanticRow]:
        """Return semantic rows whose provenance source_kind matches.
        Used by the cortex grouping UI ('show me everything you learned
        from conversation' vs '...from planner steps')."""
        if not source_kind:
            return []
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM semantic WHERE source_kind = ? "
                "ORDER BY ts DESC LIMIT ?",
                (source_kind, limit),
            ).fetchall()
        return [_row_to_semantic(r) for r in rows]

    def list_all_facts_with_embedding_text(self) -> List[SemanticRow]:
        """Return every semantic row. Used by consolidate_facts which
        needs to scan the whole table; keep separate from find_facts so
        the recall path can stay bounded."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM semantic ORDER BY id ASC"
            ).fetchall()
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
    # Provenance columns are present on migrated DBs but missing on a
    # row mapping pulled from an unmigrated test fixture. Guard each
    # access so old code paths still decode cleanly.
    keys = r.keys() if hasattr(r, "keys") else []
    def _opt(name: str, default: Any = None) -> Any:
        if name in keys:
            return r[name]
        return default
    sk = _opt("source_kind") or "legacy"
    return SemanticRow(
        id=int(r["id"]),
        ts=float(r["ts"]),
        kind=str(r["kind"]),
        key=str(r["key"]),
        value=str(r["value"]),
        source=r["source"],
        source_kind=str(sk),
        source_id=_opt("source_id"),
        extracted_at=(float(_opt("extracted_at"))
                      if _opt("extracted_at") is not None else None),
    )


# ---- consolidation ---------------------------------------------------------
def consolidate_facts(store: "MemoryStore",
                      embedder: Any,
                      min_similarity: float = 0.92,
                      max_pairs: int = 100) -> int:
    """Merge near-duplicate semantic facts by embedding cosine similarity.

    For each pair of facts within the same (kind, key) bucket whose
    value embeddings are at least ``min_similarity`` similar, keep the
    one with the newest ``extracted_at`` and fold the loser's
    provenance into the winner's ``source_id`` as a comma-separated
    list. The loser row is deleted.

    Idempotent: re-running on an already-consolidated store returns 0.
    Defensive: if embeddings can't be computed (no API key, network
    down) the function logs nothing and returns 0 — better to skip than
    to crash the background daemon.

    Returns the number of merged pairs.
    """
    # Local import to avoid a hard dependency at module-load time
    # (embedder.py imports cleanly but cosine_sim is the only thing we
    # need; keep the surface narrow).
    from .embedder import cosine_sim  # noqa: WPS433 (intentional local import)

    if min_similarity <= 0 or max_pairs <= 0:
        return 0
    try:
        rows = store.list_all_facts_with_embedding_text()
    except Exception:
        return 0
    if len(rows) < 2:
        return 0

    # Group by (kind, key) so we only compare apples-to-apples. Two
    # facts under different keys are semantically distinct even if
    # their values happen to embed similarly.
    buckets: Dict[Tuple[str, str], List[SemanticRow]] = {}
    for row in rows:
        buckets.setdefault((row.kind, row.key), []).append(row)

    merged = 0
    for bucket_rows in buckets.values():
        if len(bucket_rows) < 2 or merged >= max_pairs:
            continue
        # Embed each fact's value once; bail on any single failure so
        # one bad row doesn't poison the whole pass.
        embeds: Dict[int, List[float]] = {}
        for row in bucket_rows:
            try:
                vec = embedder.embed(row.value)
            except Exception:
                vec = []
            if vec:
                embeds[row.id] = vec
        if len(embeds) < 2:
            continue
        # Sort newest-first so the highest extracted_at wins. Ties
        # fall back to insert ts.
        bucket_rows.sort(
            key=lambda r: (r.extracted_at or r.ts, r.ts),
            reverse=True,
        )
        already_merged: set = set()
        for i, winner in enumerate(bucket_rows):
            if merged >= max_pairs:
                break
            if winner.id in already_merged or winner.id not in embeds:
                continue
            for loser in bucket_rows[i + 1:]:
                if merged >= max_pairs:
                    break
                if loser.id in already_merged or loser.id not in embeds:
                    continue
                sim = cosine_sim(embeds[winner.id], embeds[loser.id])
                if sim < min_similarity:
                    continue
                # Fold loser's provenance into winner's source_id, then
                # delete the loser. Use the store's lock so other writers
                # don't race the merge.
                try:
                    with store._lock, store._conn() as c:  # type: ignore[attr-defined]
                        # Append loser source_kind/source_id to winner.
                        existing_ids = (winner.source_id or "").strip()
                        addition_parts = []
                        if loser.source_kind:
                            addition_parts.append(f"{loser.source_kind}:{loser.id}")
                        if loser.source_id:
                            addition_parts.append(loser.source_id)
                        addition = ",".join(p for p in addition_parts if p)
                        new_source_id = (
                            f"{existing_ids},{addition}".strip(",")
                            if existing_ids else addition
                        )
                        c.execute(
                            "UPDATE semantic SET source_id = ? WHERE id = ?",
                            (new_source_id or None, winner.id),
                        )
                        c.execute(
                            "DELETE FROM semantic WHERE id = ?",
                            (loser.id,),
                        )
                except Exception:
                    # Best-effort; skip this pair on any error.
                    continue
                already_merged.add(loser.id)
                # Reflect the in-memory winner so subsequent iterations
                # see the updated source_id.
                winner.source_id = new_source_id or None
                merged += 1
    return merged

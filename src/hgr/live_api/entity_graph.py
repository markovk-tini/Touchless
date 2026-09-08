"""Entity graph — relationships, not just flat facts.

Phase-6 cognition. By Phase 5, memory was a SemanticRow table
keyed by (kind, key, value): "person.dani -> dani@x". Flat. Two
problems:

  1. **No relationships.** Memory knows Dani's email but not that
     Dani is "the lead on the Q3 contract" or "copied on the
     pricing thread." When the user says "send the Q3 stuff to
     Dani," we can't cross-reference.

  2. **No pronoun resolution.** "Send THAT to HIM" needs an
     entity store that tracks "the thing the user just looked at"
     + "the person the user just discussed."

This module adds a small relationship layer ON TOP of the
existing semantic store. It doesn't replace MemoryStore — that
keeps owning facts. EntityGraph owns:

  * **Entities** — typed nodes (person, project, document,
    artifact, location, app, file). Each entity has a canonical
    id, a display name, and a small attribute bag.
  * **Relations** — typed edges between entities ("works_on",
    "copied_on", "mentioned_in", "authored_by"). Plus a "last
    referenced" timestamp so pronoun resolution can pick "the
    most recently mentioned person."
  * **Aliases** — when the user says "Dani" but memory has
    "Daniel Markov", the graph knows they're the same entity.

SQLite-backed at `%LOCALAPPDATA%/Touchless/private/entities.db`.
Survives restarts. Honors incognito (no writes in private mode).

This module exposes the GRAPH. The PRONOUN RESOLVER and PROJECT
PROFILE substrates that consume it live in their own modules.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class EntityKind(str, Enum):
    PERSON = "person"
    PROJECT = "project"
    DOCUMENT = "document"
    ARTIFACT = "artifact"
    LOCATION = "location"
    APP = "app"
    FILE = "file"
    EMAIL_THREAD = "email_thread"
    EVENT = "event"
    TOPIC = "topic"


class RelationKind(str, Enum):
    WORKS_ON = "works_on"            # person -> project
    LEADS = "leads"                  # person -> project
    COPIED_ON = "copied_on"          # person -> email_thread/document
    AUTHORED_BY = "authored_by"      # document -> person
    MENTIONED_IN = "mentioned_in"    # any -> any (conversational)
    PART_OF = "part_of"              # any -> project
    ATTENDS = "attends"              # person -> event
    LOCATED_AT = "located_at"        # person -> location
    DUE_BY = "due_by"                # project/document -> event
    HAS_ALIAS = "has_alias"          # entity -> entity (canonical)
    RELATED_TO = "related_to"        # generic catchall


@dataclass
class Entity:
    id: str                            # uuid hex (12 chars)
    kind: str                          # EntityKind value
    name: str                          # canonical display name
    attrs: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = 0.0

    def matches_name(self, query: str) -> bool:
        if not query:
            return False
        q = query.strip().lower()
        if self.name.lower() == q:
            return True
        return any(a.lower() == q for a in self.aliases)


@dataclass
class Relation:
    src_id: str
    kind: str                          # RelationKind value
    dst_id: str
    ts: float = field(default_factory=time.time)
    attrs: Dict[str, Any] = field(default_factory=dict)


def _default_db() -> Path:
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(local) / "Touchless" / "private" / "entities.db"


class EntityGraph:
    """SQLite-backed entity + relation store. Thread-safe."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS entities (
        id            TEXT PRIMARY KEY,
        kind          TEXT NOT NULL,
        name          TEXT NOT NULL,
        attrs_json    TEXT NOT NULL DEFAULT '{}',
        aliases_json  TEXT NOT NULL DEFAULT '[]',
        created_at    REAL NOT NULL,
        last_seen_at  REAL NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);
    CREATE INDEX IF NOT EXISTS idx_entities_lastseen
        ON entities(last_seen_at DESC);

    CREATE TABLE IF NOT EXISTS relations (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        src_id       TEXT NOT NULL,
        kind         TEXT NOT NULL,
        dst_id       TEXT NOT NULL,
        ts           REAL NOT NULL,
        attrs_json   TEXT NOT NULL DEFAULT '{}',
        UNIQUE (src_id, kind, dst_id) ON CONFLICT REPLACE
    );
    CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src_id);
    CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst_id);
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

    # ---- entities -----------------------------------------------------

    def upsert_entity(self, *, kind: str, name: str,
                      attrs: Optional[Dict[str, Any]] = None,
                      aliases: Optional[List[str]] = None,
                      entity_id: Optional[str] = None) -> str:
        """Insert OR merge an entity. When `entity_id` is given,
        update that specific entity; otherwise look up by
        (kind, name lower-case match) first. Returns the entity id.
        Honors incognito — silent no-op in private mode."""
        try:
            from .incognito import is_incognito
            if is_incognito():
                return ""
        except Exception:
            pass
        attrs = attrs or {}
        aliases = aliases or []
        with self._lock:
            now = time.time()
            existing = None
            if entity_id is not None:
                existing = self.get(entity_id)
            else:
                existing = self._find_by_kind_and_name(kind, name)
            if existing is not None:
                # Merge attrs + aliases.
                merged_attrs = dict(existing.attrs)
                merged_attrs.update(attrs)
                merged_aliases = list({
                    *existing.aliases,
                    *aliases,
                })
                self._conn.execute(
                    "UPDATE entities SET name=?, attrs_json=?, "
                    "aliases_json=?, last_seen_at=? WHERE id=?",
                    (name, json.dumps(merged_attrs),
                     json.dumps(merged_aliases),
                     now, existing.id),
                )
                return existing.id
            new_id = uuid.uuid4().hex[:12]
            self._conn.execute(
                "INSERT INTO entities"
                "(id, kind, name, attrs_json, aliases_json, "
                " created_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (new_id, kind, name,
                 json.dumps(attrs), json.dumps(aliases),
                 now, now),
            )
            return new_id

    def get(self, entity_id: str) -> Optional[Entity]:
        if not entity_id:
            return None
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM entities WHERE id=?", (entity_id,))
            row = cur.fetchone()
        return self._row_to_entity(row)

    def _find_by_kind_and_name(self, kind: str,
                                name: str) -> Optional[Entity]:
        if not (kind and name):
            return None
        q = name.strip().lower()
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM entities WHERE kind=? "
                "ORDER BY last_seen_at DESC", (kind,))
            rows = cur.fetchall()
        for row in rows:
            e = self._row_to_entity(row)
            if e is None:
                continue
            if e.matches_name(q):
                return e
        return None

    def find_by_alias(self, alias: str,
                      *, kind: Optional[str] = None,
                      ) -> Optional[Entity]:
        """Lookup by display name OR any alias. When `kind` is
        given, restrict the search to that entity kind. Returns the
        most-recently-seen match."""
        if not alias:
            return None
        with self._lock:
            sql = "SELECT * FROM entities"
            args: List[Any] = []
            if kind is not None:
                sql += " WHERE kind=?"
                args.append(kind)
            sql += " ORDER BY last_seen_at DESC"
            cur = self._conn.execute(sql, args)
            rows = cur.fetchall()
        for row in rows:
            e = self._row_to_entity(row)
            if e is None:
                continue
            if e.matches_name(alias):
                return e
        return None

    def touch(self, entity_id: str) -> None:
        """Bump last_seen_at on this entity. Used by the pronoun
        resolver every time an entity is referenced — drives the
        'most-recent' tie-breaker."""
        if not entity_id:
            return
        try:
            from .incognito import is_incognito
            if is_incognito():
                return
        except Exception:
            pass
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE entities SET last_seen_at=? WHERE id=?",
                    (time.time(), entity_id))
            except Exception:
                pass

    def all_of_kind(self, kind: str,
                    *, limit: int = 100) -> List[Entity]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM entities WHERE kind=? "
                "ORDER BY last_seen_at DESC LIMIT ?",
                (kind, limit))
            rows = cur.fetchall()
        return [e for e in (self._row_to_entity(r) for r in rows)
                if e is not None]

    def recent_by_kind(self, kind: str, *,
                       max_age_sec: float = 86400.0,
                       limit: int = 5) -> List[Entity]:
        """Entities of this kind touched within `max_age_sec`. Used
        by the pronoun resolver — 'the most recent person mentioned'
        + 'the most recent document opened'."""
        cutoff = time.time() - max_age_sec
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM entities WHERE kind=? "
                "AND last_seen_at >= ? "
                "ORDER BY last_seen_at DESC LIMIT ?",
                (kind, cutoff, limit))
            rows = cur.fetchall()
        return [e for e in (self._row_to_entity(r) for r in rows)
                if e is not None]

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity + every relation touching it."""
        if not entity_id:
            return False
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM entities WHERE id=?", (entity_id,))
            self._conn.execute(
                "DELETE FROM relations WHERE src_id=? OR dst_id=?",
                (entity_id, entity_id))
            return cur.rowcount > 0

    # ---- relations ----------------------------------------------------

    def add_relation(self, *, src_id: str, kind: str,
                     dst_id: str,
                     attrs: Optional[Dict[str, Any]] = None) -> bool:
        """Insert or replace a (src, kind, dst) edge. Honors
        incognito."""
        if not (src_id and kind and dst_id):
            return False
        try:
            from .incognito import is_incognito
            if is_incognito():
                return False
        except Exception:
            pass
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO relations "
                    "(src_id, kind, dst_id, ts, attrs_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (src_id, kind, dst_id, time.time(),
                     json.dumps(attrs or {})),
                )
                return True
            except Exception:
                return False

    def relations_of(self, entity_id: str,
                     *, kind: Optional[str] = None,
                     direction: str = "both"
                     ) -> List[Relation]:
        """Return edges touching `entity_id`. `direction` is one of
        'out' (entity is src), 'in' (entity is dst), 'both'."""
        with self._lock:
            sql = "SELECT src_id, kind, dst_id, ts, attrs_json FROM relations WHERE "
            args: List[Any] = []
            if direction == "out":
                sql += "src_id=?"
                args.append(entity_id)
            elif direction == "in":
                sql += "dst_id=?"
                args.append(entity_id)
            else:
                sql += "(src_id=? OR dst_id=?)"
                args.extend([entity_id, entity_id])
            if kind is not None:
                sql += " AND kind=?"
                args.append(kind)
            sql += " ORDER BY ts DESC"
            cur = self._conn.execute(sql, args)
            rows = cur.fetchall()
        out: List[Relation] = []
        for r in rows:
            try:
                attrs = json.loads(r[4] or "{}")
            except Exception:
                attrs = {}
            out.append(Relation(
                src_id=r[0], kind=r[1], dst_id=r[2],
                ts=float(r[3] or 0.0), attrs=attrs,
            ))
        return out

    def neighbors(self, entity_id: str, *,
                  kind: Optional[str] = None,
                  ) -> List[Tuple[Relation, Entity]]:
        """Convenience: return (relation, other_entity) pairs."""
        rels = self.relations_of(entity_id, kind=kind)
        out: List[Tuple[Relation, Entity]] = []
        for r in rels:
            other_id = r.dst_id if r.src_id == entity_id else r.src_id
            other = self.get(other_id)
            if other is not None:
                out.append((r, other))
        return out

    # ---- maintenance --------------------------------------------------

    def wipe(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM entities")
            n = int(cur.fetchone()[0])
            self._conn.execute("DELETE FROM entities")
            self._conn.execute("DELETE FROM relations")
            return n

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ---- row → dataclass helpers --------------------------------------

    @staticmethod
    def _row_to_entity(row: Any) -> Optional[Entity]:
        if row is None:
            return None
        try:
            attrs = json.loads(row[3] or "{}")
            aliases = json.loads(row[4] or "[]")
        except Exception:
            attrs, aliases = {}, []
        return Entity(
            id=row[0], kind=row[1] or "", name=row[2] or "",
            attrs=attrs, aliases=aliases,
            created_at=float(row[5] or 0.0),
            last_seen_at=float(row[6] or 0.0),
        )


# ---- name extraction helpers (used by the pronoun resolver) ---------

_NAME_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]{1,30})\b")


def extract_name_mentions(text: str) -> List[str]:
    """Pull likely person-name tokens from free-text. Conservative:
    capitalized 2-30 char tokens; rejects common English words
    that often appear capitalized at sentence start."""
    if not text:
        return []
    seen: List[str] = []
    stop = {"I", "I'm", "I'll", "The", "A", "An", "This", "That",
            "These", "Those", "It", "Hey", "Hi", "Hello", "Ok",
            "Okay", "Yes", "No", "Maybe", "Today", "Tomorrow",
            "Yesterday", "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday", "January", "February",
            "March", "April", "May", "June", "July", "August",
            "September", "October", "November", "December"}
    for m in _NAME_RE.finditer(text):
        name = m.group(1)
        if name in stop:
            continue
        if name in seen:
            continue
        seen.append(name)
    return seen


# ---- module singleton --------------------------------------------------

_graph: Optional[EntityGraph] = None
_lock = threading.Lock()


def global_graph() -> EntityGraph:
    global _graph
    if _graph is None:
        with _lock:
            if _graph is None:
                _graph = EntityGraph()
    return _graph


def _reset_for_tests() -> None:
    global _graph
    with _lock:
        if _graph is not None:
            try:
                _graph.close()
            except Exception:
                pass
        _graph = None

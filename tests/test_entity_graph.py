"""Tests for entity_graph (Phase 6 B1)."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.incognito as inc  # noqa: E402
from hgr.live_api.entity_graph import (  # noqa: E402
    EntityGraph, EntityKind, RelationKind, extract_name_mentions,
)


def setup_function():
    inc.set_incognito(False)


def _fresh() -> EntityGraph:
    d = Path(tempfile.mkdtemp())
    return EntityGraph(db_path=d / "e.db")


# ---- upsert + get ------------------------------------------------------

def test_upsert_creates_new_entity():
    g = _fresh()
    eid = g.upsert_entity(
        kind=EntityKind.PERSON.value,
        name="Dani Markov",
        attrs={"email": "dani@x"},
        aliases=["Dani", "Daniel"])
    assert eid
    entity = g.get(eid)
    assert entity is not None
    assert entity.name == "Dani Markov"
    assert entity.attrs["email"] == "dani@x"
    assert "Dani" in entity.aliases


def test_upsert_merges_into_existing():
    g = _fresh()
    eid = g.upsert_entity(
        kind=EntityKind.PERSON.value,
        name="Dani Markov", attrs={"email": "dani@x"},
        aliases=["Dani"])
    g.upsert_entity(
        kind=EntityKind.PERSON.value,
        name="Dani Markov", attrs={"role": "lead"},
        aliases=["DM"])
    entity = g.get(eid)
    # Attrs merged.
    assert entity.attrs["email"] == "dani@x"
    assert entity.attrs["role"] == "lead"
    # Aliases merged + deduped.
    assert "Dani" in entity.aliases
    assert "DM" in entity.aliases


def test_upsert_skipped_in_incognito():
    g = _fresh()
    inc.set_incognito(True)
    try:
        eid = g.upsert_entity(
            kind=EntityKind.PERSON.value, name="Dani")
    finally:
        inc.set_incognito(False)
    assert eid == ""


# ---- alias lookup ------------------------------------------------------

def test_find_by_alias_canonical_name():
    g = _fresh()
    g.upsert_entity(
        kind=EntityKind.PERSON.value,
        name="Dani Markov", aliases=["Dani", "DM"])
    assert g.find_by_alias("Dani Markov") is not None
    assert g.find_by_alias("Dani") is not None
    assert g.find_by_alias("DM") is not None


def test_find_by_alias_case_insensitive():
    g = _fresh()
    g.upsert_entity(kind=EntityKind.PERSON.value, name="Dani")
    assert g.find_by_alias("dani") is not None
    assert g.find_by_alias("DANI") is not None


def test_find_by_alias_kind_filter():
    g = _fresh()
    g.upsert_entity(kind=EntityKind.PERSON.value, name="Q3")
    g.upsert_entity(kind=EntityKind.PROJECT.value, name="Q3")
    person = g.find_by_alias("Q3",
                              kind=EntityKind.PERSON.value)
    project = g.find_by_alias("Q3",
                                kind=EntityKind.PROJECT.value)
    assert person.kind == EntityKind.PERSON.value
    assert project.kind == EntityKind.PROJECT.value


def test_find_by_alias_returns_none_unknown():
    g = _fresh()
    assert g.find_by_alias("ghost") is None


# ---- last_seen + recent ordering --------------------------------------

def test_touch_bumps_last_seen():
    g = _fresh()
    eid = g.upsert_entity(
        kind=EntityKind.PERSON.value, name="Dani")
    initial = g.get(eid).last_seen_at
    time.sleep(0.01)
    g.touch(eid)
    bumped = g.get(eid).last_seen_at
    assert bumped > initial


def test_recent_by_kind_orders_newest_first():
    g = _fresh()
    e1 = g.upsert_entity(
        kind=EntityKind.PERSON.value, name="A")
    time.sleep(0.005)
    e2 = g.upsert_entity(
        kind=EntityKind.PERSON.value, name="B")
    time.sleep(0.005)
    g.touch(e1)  # A becomes most-recent
    recent = g.recent_by_kind(EntityKind.PERSON.value, limit=5)
    assert recent[0].name == "A"


def test_recent_by_kind_age_cutoff():
    g = _fresh()
    eid = g.upsert_entity(
        kind=EntityKind.PERSON.value, name="Old")
    # Force the last_seen_at to be ancient.
    g._conn.execute(
        "UPDATE entities SET last_seen_at=0 WHERE id=?", (eid,))
    recent = g.recent_by_kind(EntityKind.PERSON.value,
                                max_age_sec=60.0)
    assert recent == []


# ---- relations --------------------------------------------------------

def test_add_relation_and_query():
    g = _fresh()
    dani = g.upsert_entity(
        kind=EntityKind.PERSON.value, name="Dani")
    q3 = g.upsert_entity(
        kind=EntityKind.PROJECT.value, name="Q3 contract")
    g.add_relation(
        src_id=dani, kind=RelationKind.WORKS_ON.value, dst_id=q3)
    rels = g.relations_of(dani, direction="out")
    assert len(rels) == 1
    assert rels[0].kind == RelationKind.WORKS_ON.value
    assert rels[0].dst_id == q3


def test_relation_kind_filter():
    g = _fresh()
    dani = g.upsert_entity(
        kind=EntityKind.PERSON.value, name="Dani")
    q3 = g.upsert_entity(
        kind=EntityKind.PROJECT.value, name="Q3")
    g.add_relation(src_id=dani,
                    kind=RelationKind.WORKS_ON.value, dst_id=q3)
    g.add_relation(src_id=dani,
                    kind=RelationKind.LEADS.value, dst_id=q3)
    leads = g.relations_of(
        dani, kind=RelationKind.LEADS.value)
    assert len(leads) == 1


def test_relation_replace_on_duplicate():
    g = _fresh()
    a = g.upsert_entity(kind=EntityKind.PERSON.value, name="A")
    b = g.upsert_entity(kind=EntityKind.PROJECT.value, name="P")
    g.add_relation(src_id=a, kind=RelationKind.WORKS_ON.value,
                    dst_id=b, attrs={"v": 1})
    g.add_relation(src_id=a, kind=RelationKind.WORKS_ON.value,
                    dst_id=b, attrs={"v": 2})
    rels = g.relations_of(a, kind=RelationKind.WORKS_ON.value)
    # UNIQUE conflict → REPLACE; only one row.
    assert len(rels) == 1
    assert rels[0].attrs["v"] == 2


def test_neighbors_returns_pairs():
    g = _fresh()
    a = g.upsert_entity(kind=EntityKind.PERSON.value, name="A")
    b = g.upsert_entity(kind=EntityKind.PROJECT.value, name="B")
    g.add_relation(src_id=a, kind=RelationKind.WORKS_ON.value,
                    dst_id=b)
    pairs = g.neighbors(a)
    assert len(pairs) == 1
    rel, other = pairs[0]
    assert other.id == b


def test_delete_entity_cascades_relations():
    g = _fresh()
    a = g.upsert_entity(kind=EntityKind.PERSON.value, name="A")
    b = g.upsert_entity(kind=EntityKind.PROJECT.value, name="B")
    g.add_relation(src_id=a, kind=RelationKind.WORKS_ON.value,
                    dst_id=b)
    g.delete_entity(a)
    assert g.get(a) is None
    # Relation gone too.
    assert g.relations_of(b, direction="in") == []


def test_add_relation_skipped_in_incognito():
    g = _fresh()
    a = g.upsert_entity(kind=EntityKind.PERSON.value, name="A")
    b = g.upsert_entity(kind=EntityKind.PROJECT.value, name="B")
    inc.set_incognito(True)
    try:
        ok = g.add_relation(
            src_id=a, kind=RelationKind.WORKS_ON.value, dst_id=b)
    finally:
        inc.set_incognito(False)
    assert ok is False
    assert g.relations_of(a) == []


# ---- name extraction helper ------------------------------------------

def test_extract_name_mentions_picks_capitalized_tokens():
    out = extract_name_mentions(
        "Dani and Alice are working with Bob on Q3.")
    assert "Dani" in out
    assert "Alice" in out
    assert "Bob" in out
    assert "Q3" in out


def test_extract_name_mentions_skips_common_words():
    out = extract_name_mentions(
        "I'm working with Tomorrow on Monday.")
    # 'I'm', 'Tomorrow', 'Monday' all in stop-list.
    assert out == []


# ---- wipe -------------------------------------------------------------

def test_wipe_clears_entities_and_relations():
    g = _fresh()
    a = g.upsert_entity(kind=EntityKind.PERSON.value, name="A")
    b = g.upsert_entity(kind=EntityKind.PERSON.value, name="B")
    g.add_relation(src_id=a, kind=RelationKind.RELATED_TO.value,
                    dst_id=b)
    n = g.wipe()
    assert n >= 2
    assert g.get(a) is None
    assert g.relations_of(b) == []

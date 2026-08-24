"""Tests for memory pinning (Phase 3)."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.memory_pinning import (  # noqa: E402
    COUNT_WINDOW_SEC, PIN_THRESHOLD, PINNED_CONFIDENCE,
    evaluate_pinning_candidate, maybe_pin_fact,
)
import hgr.live_api.incognito as inc  # noqa: E402


@dataclass
class _Fact:
    value: str
    extracted_at: float


class _FakeStore:
    """In-memory fake matching the shape memory_pinning expects."""

    def __init__(self, facts=None):
        self._facts: List[_Fact] = list(facts or [])
        self.pinned_calls: List[tuple] = []
        self.added: List[tuple] = []

    def find_facts(self, *, kind, key):
        return list(self._facts)

    def pin_fact(self, *, kind, key, value):
        self.pinned_calls.append((kind, key, value))

    def add_semantic(self, kind, key, value, *,
                     source="", source_kind="", source_id=None,
                     extracted_at=None):
        self.added.append((kind, key, value, source_kind))


def setup_function():
    inc.set_incognito(False)


# ---- evaluate -----------------------------------------------------------

def test_below_threshold_does_not_pin():
    store = _FakeStore([_Fact("Berlin", time.time())])
    d = evaluate_pinning_candidate(
        store=store, kind="location", key="home")
    assert d.should_pin is False
    assert d.mention_count == 1


def test_threshold_crossed_with_same_value_pins():
    now = time.time()
    store = _FakeStore([_Fact("Berlin", now - 100),
                        _Fact("Berlin", now - 50),
                        _Fact("Berlin", now)])
    d = evaluate_pinning_candidate(
        store=store, kind="location", key="home")
    assert d.should_pin is True
    assert d.mention_count == 3


def test_threshold_crossed_with_different_values_does_not_pin():
    """Multiple distinct values for the same key = user changed mind;
    don't auto-pin a stale value."""
    now = time.time()
    store = _FakeStore([_Fact("Berlin", now - 100),
                        _Fact("Tokyo", now - 50),
                        _Fact("Berlin", now)])
    d = evaluate_pinning_candidate(
        store=store, kind="location", key="home")
    assert d.should_pin is False
    assert "ambiguous" in d.reason.lower()


def test_old_mentions_excluded_from_count():
    """Mentions older than COUNT_WINDOW_SEC don't count toward the
    pin threshold."""
    now = time.time()
    way_old = now - (COUNT_WINDOW_SEC + 86400)
    store = _FakeStore([_Fact("Berlin", way_old),
                        _Fact("Berlin", way_old),
                        _Fact("Berlin", way_old)])
    d = evaluate_pinning_candidate(
        store=store, kind="location", key="home")
    assert d.should_pin is False


def test_missing_keys_return_false():
    store = _FakeStore([])
    assert evaluate_pinning_candidate(
        store=store, kind="", key="x").should_pin is False
    assert evaluate_pinning_candidate(
        store=store, kind="x", key="").should_pin is False


def test_no_facts_returns_zero_count():
    d = evaluate_pinning_candidate(
        store=_FakeStore([]), kind="location", key="home")
    assert d.mention_count == 0
    assert d.should_pin is False


def test_undated_facts_count_as_recent():
    """A fact without an extracted_at timestamp shouldn't be silently
    dropped — treat it as recent."""
    store = _FakeStore([_Fact("x", None),
                        _Fact("x", None),
                        _Fact("x", None)])
    d = evaluate_pinning_candidate(
        store=store, kind="k", key="key")
    assert d.should_pin is True


# ---- maybe_pin_fact ---------------------------------------------------

def test_maybe_pin_calls_pin_fact_when_present():
    now = time.time()
    store = _FakeStore([_Fact("Berlin", now)] * 3)
    pinned = maybe_pin_fact(
        store=store, kind="location", key="home")
    assert pinned is True
    assert len(store.pinned_calls) == 1


def test_maybe_pin_falls_back_to_add_semantic():
    """When the store has no pin_fact API, fall back to add_semantic
    with elevated confidence."""
    class _NoPinStore(_FakeStore):
        pin_fact = None  # type: ignore[assignment]

        def __getattribute__(self, name):
            if name == "pin_fact":
                raise AttributeError("no pin_fact")
            return object.__getattribute__(self, name)

    store = _NoPinStore([_Fact("Berlin", time.time())] * 3)
    pinned = maybe_pin_fact(
        store=store, kind="location", key="home")
    assert pinned is True
    assert len(store.added) == 1
    assert store.added[0][3] == "auto_pin"


def test_maybe_pin_skips_when_below_threshold():
    store = _FakeStore([_Fact("Berlin", time.time())])
    assert maybe_pin_fact(
        store=store, kind="location", key="home") is False
    assert store.pinned_calls == []


def test_maybe_pin_honors_incognito():
    store = _FakeStore([_Fact("Berlin", time.time())] * 3)
    inc.set_incognito(True)
    try:
        pinned = maybe_pin_fact(
            store=store, kind="location", key="home")
    finally:
        inc.set_incognito(False)
    assert pinned is False
    assert store.pinned_calls == []


def test_constants_in_sane_range():
    assert 2 <= PIN_THRESHOLD <= 10
    assert 0.5 < PINNED_CONFIDENCE <= 1.0


# ---- read-side: pinned facts surface first in MemoryStore.find_facts --

def test_real_store_returns_pinned_facts_first():
    """End-to-end with the real MemoryStore: a pinned (auto_pin)
    row must come back BEFORE any later non-pinned mentions of
    the same (kind, key). Uses mkdtemp (no auto-cleanup) because
    Windows holds a lock on the SQLite file until the connection
    is closed."""
    import tempfile
    from pathlib import Path as _Path
    from hgr.live_api.memory.store import MemoryStore
    tmp = tempfile.mkdtemp()
    store = MemoryStore(_Path(tmp) / "mem.db")
    # Three implicit-pattern mentions of "Berlin" so the pin
    # rule fires.
    now = time.time()
    for i in range(3):
        store.add_semantic("location", "home", "Berlin",
                           source="user said",
                           source_kind="implicit_pattern",
                           source_id=None,
                           extracted_at=now - (10 - i))
    # Pin the most recent value.
    store.add_semantic("location", "home", "Berlin",
                       source="auto-pin: 3 mentions",
                       source_kind="auto_pin",
                       source_id=None,
                       extracted_at=now - 5)
    # A NEWER (one-off) mention with a different value — should
    # still come back AFTER the pinned row.
    store.add_semantic("location", "home", "Vacation Cabin",
                       source="user said",
                       source_kind="implicit_pattern",
                       source_id=None,
                       extracted_at=now)
    rows = store.find_facts(kind="location", key="home")
    assert rows, "expected at least one row"
    assert rows[0].source_kind == "auto_pin"
    assert rows[0].value == "Berlin"

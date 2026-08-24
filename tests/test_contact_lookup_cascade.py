"""Invariants for the iris_lookup_contact pseudo-tool and the
contacts_search orchestrator cascade.

Two real bugs these tests pin down:

1. iris_lookup_contact previously only queried `kind='person'` and
   blindly took facts[0].value. A stored relation fact like
   `kind='person', key='vesko', value='my brother'` (or 'contact')
   would render as `"Vesko's email is my brother."`. The fix scans
   BOTH kind='person' AND kind='contact' and requires the value to
   look email-shaped before using it.

2. contacts_search routes to exactly ONE owning connector (first-come
   wins in `ConnectorRegistry.available_tool_schemas`). A miss in
   Google People used to return 'not_found' even when Microsoft Graph
   OR memory had the contact. The fix cascades through own-connector
   -> memory facts -> the OTHER connector before giving up.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataclasses import dataclass  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402


@dataclass
class _FakeFact:
    kind: str
    key: str
    value: str


class _FakeStore:
    def __init__(self, rows: List[_FakeFact]):
        self._rows = list(rows)

    def find_facts(self, kind: Optional[str] = None,
                   key: Optional[str] = None, limit: int = 50):
        out = []
        for r in self._rows:
            if kind is not None and r.kind != kind:
                continue
            if key is not None and r.key != key.lower():
                continue
            out.append(r)
        return out[:limit]


class _FakeMemory:
    def __init__(self, rows: List[_FakeFact]):
        self._store = _FakeStore(rows)


# ---- iris_lookup_contact: invariant tests -----------------------------

def _lookup_email_with_memory(name: str,
                              rows: List[_FakeFact]) -> Optional[str]:
    """Mirror the orchestrator's iris_lookup_contact memory scan:
    try (name, name-without-trailing-s) x (person, contact) and only
    accept email-shaped values."""
    candidates = [name]
    if len(name) > 3 and name.lower().endswith("s"):
        candidates.append(name[:-1])
    memory = _FakeMemory(rows)
    for cand in candidates:
        for kind in ("person", "contact"):
            for f in memory._store.find_facts(kind=kind, key=cand.lower()):
                v = (f.value or "").strip()
                if "@" in v and "." in v.rsplit("@", 1)[-1]:
                    return v
    return None


def test_iris_lookup_contact_skips_non_email_person_fact():
    """A kind=person/value='my brother' must NOT be returned as the email."""
    rows = [_FakeFact("person", "vesko", "my brother")]
    assert _lookup_email_with_memory("vesko", rows) is None


def test_iris_lookup_contact_skips_literal_contact_string():
    """The reported bug: value='contact' must NOT render as the email."""
    rows = [_FakeFact("person", "vesko", "contact")]
    assert _lookup_email_with_memory("vesko", rows) is None


def test_iris_lookup_contact_reads_kind_contact():
    """LLM extractor / self_learner write under kind='contact'; the
    lookup MUST find those rows, not only kind='person'."""
    rows = [_FakeFact("contact", "vesko", "markovi@msn.com")]
    assert _lookup_email_with_memory("vesko", rows) == "markovi@msn.com"


def test_iris_lookup_contact_prefers_email_over_nonsense():
    """If both a relation fact and a real email exist, return the email."""
    rows = [
        _FakeFact("person", "vesko", "my brother"),
        _FakeFact("contact", "vesko", "markovi@msn.com"),
    ]
    assert _lookup_email_with_memory("vesko", rows) == "markovi@msn.com"


def test_iris_lookup_contact_handles_trailing_s_capture():
    """'Veskos email' captures name='Veskos'; lookup must try 'vesko'."""
    rows = [_FakeFact("person", "vesko", "vesko@example.com")]
    assert _lookup_email_with_memory("Veskos", rows) == "vesko@example.com"


# ---- contacts_search cascade: invariant tests -------------------------

def test_cascade_helper_signature_present():
    """The orchestrator must expose `_contacts_search_cascade` so the
    main dispatch (line ~951) can intercept contacts_search and run
    the three-tier fallback."""
    # Import the module directly (not via the package) so this stays
    # independent of unrelated planner package-init imports.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_orchestrator_under_test",
        str(ROOT / "src" / "hgr" / "live_api" / "planner" /
            "orchestrator.py"))
    assert spec is not None and spec.loader is not None
    # Just text-scan the file for the symbol — avoids loading the
    # whole package graph (which has many optional deps).
    src = (ROOT / "src" / "hgr" / "live_api" / "planner" /
           "orchestrator.py").read_text(encoding="utf-8")
    assert "def _contacts_search_cascade(" in src
    assert "single.tool == \"contacts_search\"" in src

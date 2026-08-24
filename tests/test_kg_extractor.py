"""Tests for kg_extractor (Phase 9 B3)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.incognito as inc  # noqa: E402
from hgr.live_api.kg_extractor import (  # noqa: E402
    ExtractedEntity, ExtractedKind, _extract_dates,
    _extract_emails, _extract_persons, _extract_projects,
    extract_from_text, seed_graph_from_text,
)


def setup_function():
    inc.set_incognito(False)


# ---- emails -------------------------------------------------------

def test_extract_emails_basic():
    text = "Reach out to dani@mangollc.org for details."
    out = _extract_emails(text, source="t")
    assert len(out) == 1
    assert out[0].name == "dani@mangollc.org"
    assert out[0].kind == ExtractedKind.EMAIL


def test_extract_emails_multiple():
    text = "Email a@b.com or c@d.io if needed."
    out = _extract_emails(text, source="t")
    assert len(out) == 2


def test_extract_emails_none():
    out = _extract_emails("no email here", source="t")
    assert out == []


# ---- persons ------------------------------------------------------

def test_extract_persons_via_email_proximity():
    text = "Dani (dani@mangollc.org) sent the Q3 deck."
    emails = _extract_emails(text, source="t")
    out = _extract_persons(text, "t", emails)
    names = {p.name for p in out}
    assert "Dani" in names


def test_extract_persons_via_recurrence():
    text = ("Dani opened the issue. Dani assigned it to me. "
            "Dani is following up on Friday.")
    emails = []
    out = _extract_persons(text, "t", emails)
    names = {p.name for p in out}
    assert "Dani" in names


def test_extract_persons_skips_iris_and_dates():
    text = "Iris worked with Monday and Thursday on the deck."
    emails = []
    out = _extract_persons(text, "t", emails)
    names = {p.name for p in out}
    assert "Iris" not in names
    assert "Monday" not in names


def test_extract_persons_no_email_no_recurrence_skipped():
    text = "Alice noted the bug."  # Only 1 mention.
    emails = []
    out = _extract_persons(text, "t", emails)
    names = {p.name for p in out}
    assert "Alice" not in names


# ---- projects -----------------------------------------------------

def test_extract_projects_kebab():
    text = "We shipped the auth-rewrite-2026 last week."
    out = _extract_projects(text, source="t")
    assert any(p.name == "auth-rewrite-2026" for p in out)


def test_extract_projects_code():
    text = "Done with PROJ123 and FY26 planning."
    out = _extract_projects(text, source="t")
    names = {p.name for p in out}
    assert "PROJ123" in names
    assert "FY26" in names


def test_extract_projects_skips_quarter_codes():
    text = "Need Q3 deliverables locked."
    out = _extract_projects(text, source="t")
    assert all(p.name != "Q3" for p in out)


def test_extract_projects_dedups():
    text = "auth-rewrite is in progress. The auth-rewrite ships Q3."
    out = _extract_projects(text, source="t")
    names = [p.name for p in out]
    assert names.count("auth-rewrite") == 1


# ---- dates --------------------------------------------------------

def test_extract_dates_iso():
    text = "Due 2026-06-15 and we ship after."
    out = _extract_dates(text, source="t")
    assert any("2026-06-15" in d.name for d in out)


def test_extract_dates_relative():
    text = "Let's regroup next Thursday."
    out = _extract_dates(text, source="t")
    names = [d.name.lower() for d in out]
    assert any("next thursday" in n for n in names)


def test_extract_dates_quarter():
    text = "Ships Q3 2026."
    out = _extract_dates(text, source="t")
    assert any("q3" in d.name.lower() for d in out)


# ---- top-level extract_from_text ---------------------------------

def test_extract_from_text_returns_all_kinds():
    text = ("Dani <dani@mangollc.org> emailed about "
            "auth-rewrite-2026, due next Thursday.")
    out = extract_from_text(text, source="email")
    kinds = {e.kind for e in out}
    assert ExtractedKind.EMAIL in kinds
    assert ExtractedKind.PERSON in kinds
    assert ExtractedKind.DATE_REF in kinds


def test_extract_from_text_empty_input():
    assert extract_from_text("") == []


def test_extract_from_text_incognito_blocks():
    inc.set_incognito(True)
    try:
        out = extract_from_text("Dani <dani@x.com>")
    finally:
        inc.set_incognito(False)
    assert out == []


def test_extract_caps_input_length():
    text = "Dani <dani@x.com> " * 5000   # very long
    # Should not OOM; should still extract from first chunk.
    out = extract_from_text(text, source="t")
    assert any(e.kind == ExtractedKind.EMAIL for e in out)


# ---- seed_graph integration --------------------------------------

class _FakeGraph:
    def __init__(self):
        self.calls = []

    def upsert_entity(self, **kwargs):
        self.calls.append(kwargs)
        return "id-" + kwargs.get("name", "")


def test_seed_graph_upserts_extracted():
    g = _FakeGraph()
    text = "Dani <dani@mangollc.org> sent the deck."
    seed_graph_from_text(text, source="email", graph=g)
    upserts = g.calls
    names = {c["name"] for c in upserts}
    assert "Dani" in names or "dani@mangollc.org" in names


def test_seed_graph_handles_empty():
    g = _FakeGraph()
    seed_graph_from_text("", source="email", graph=g)
    assert g.calls == []


def test_seed_graph_passes_source_in_attrs():
    g = _FakeGraph()
    seed_graph_from_text(
        "Dani <dani@mangollc.org> sent the deck.",
        source="my-email", graph=g)
    for c in g.calls:
        if "attrs" in c:
            assert c["attrs"].get("source") in (None, "my-email")
            break


def test_seed_graph_swallows_upsert_errors():
    class BoomGraph:
        def upsert_entity(self, **kwargs):
            raise RuntimeError("nope")
    # Should not raise.
    seed_graph_from_text(
        "Dani <dani@mangollc.org> sent the deck.",
        source="t", graph=BoomGraph())


# ---- ExtractedEntity dataclass ----------------------------------

def test_extracted_entity_to_dict():
    e = ExtractedEntity(
        kind=ExtractedKind.PERSON, name="Dani",
        attributes={"email": "x"})
    d = e.to_dict()
    assert d["kind"] == "person"
    assert d["name"] == "Dani"
    assert d["attributes"] == {"email": "x"}

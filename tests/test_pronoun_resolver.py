"""Tests for pronoun_resolver (Phase 6 B1)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.entity_graph import (  # noqa: E402
    EntityGraph, EntityKind,
)
from hgr.live_api.pronoun_resolver import (  # noqa: E402
    Resolution, ResolutionReport, resolve_references,
)


def _fresh_graph() -> EntityGraph:
    d = Path(tempfile.mkdtemp())
    return EntityGraph(db_path=d / "e.db")


class _StubTurn:
    def __init__(self, role, text):
        self.role = role
        self.text = text


class _StubSession:
    def __init__(self, turns):
        self._turns = list(turns)

    def recent(self, *, max_turns=4):
        return self._turns[-max_turns:]


class _StubScreen:
    def __init__(self, app="", title=""):
        self.active_app = app
        self.active_window_title = title


# ---- name → entity ----------------------------------------------------

def test_name_in_text_resolves_to_entity():
    g = _fresh_graph()
    g.upsert_entity(
        kind=EntityKind.PERSON.value, name="Dani Markov",
        aliases=["Dani"], attrs={"email": "dani@x"})
    report = resolve_references(
        "send Dani the contract", graph=g)
    assert any(r.token == "Dani" for r in report.resolutions)
    dani_res = next(r for r in report.resolutions if r.token == "Dani")
    assert dani_res.entity_kind == EntityKind.PERSON.value
    assert dani_res.detail == "dani@x"


def test_unknown_name_does_not_resolve():
    g = _fresh_graph()
    report = resolve_references(
        "send Tomato the contract", graph=g)
    assert all(r.token != "Tomato" for r in report.resolutions)


def test_stopwords_at_sentence_start_dont_resolve():
    g = _fresh_graph()
    # Even if a person called 'The' existed, the resolver should
    # skip the capitalized stop-word.
    g.upsert_entity(
        kind=EntityKind.PERSON.value, name="The")
    report = resolve_references("The dog ran", graph=g)
    assert report.resolutions == []


# ---- pronoun: him/her -------------------------------------------------

def test_him_resolves_to_most_recent_person_in_session():
    g = _fresh_graph()
    g.upsert_entity(
        kind=EntityKind.PERSON.value, name="Dani",
        attrs={"email": "dani@x"})
    session = _StubSession([
        _StubTurn("user", "what's Dani's email"),
        _StubTurn("iris", "dani@x"),
    ])
    report = resolve_references(
        "send him the doc", graph=g, session_buffer=session)
    him_res = next((r for r in report.resolutions
                    if r.token.lower() == "him"), None)
    assert him_res is not None
    assert him_res.display_name == "Dani"


def test_her_falls_back_to_graph_when_no_session_mention():
    g = _fresh_graph()
    g.upsert_entity(
        kind=EntityKind.PERSON.value, name="Alice",
        attrs={"email": "alice@x"})
    report = resolve_references(
        "send her the report", graph=g)
    her_res = next((r for r in report.resolutions
                    if r.token.lower() == "her"), None)
    assert her_res is not None
    assert her_res.display_name == "Alice"


def test_them_resolves_like_him():
    g = _fresh_graph()
    g.upsert_entity(
        kind=EntityKind.PERSON.value, name="Bob",
        attrs={"email": "bob@x"})
    report = resolve_references("ping them", graph=g)
    them_res = next((r for r in report.resolutions
                     if r.token.lower() == "them"), None)
    assert them_res is not None


def test_no_pronoun_no_person_returns_empty():
    g = _fresh_graph()
    g.upsert_entity(
        kind=EntityKind.PERSON.value, name="Dani")
    report = resolve_references(
        "what's the weather?", graph=g)
    assert report.resolutions == []


# ---- pronoun: it/that/this --------------------------------------------

def test_it_resolves_to_most_recent_document_in_session():
    g = _fresh_graph()
    g.upsert_entity(
        kind=EntityKind.DOCUMENT.value, name="Q3 Contract",
        attrs={"url": "https://docs/q3"})
    session = _StubSession([
        _StubTurn("user", "open the Q3 Contract"),
    ])
    report = resolve_references(
        "send it to Dani", graph=g, session_buffer=session)
    it_res = next((r for r in report.resolutions
                   if r.token.lower() == "it"), None)
    assert it_res is not None
    assert "Q3" in it_res.display_name


def test_it_falls_back_to_screen_when_no_session_thing():
    g = _fresh_graph()
    screen = _StubScreen(app="Chrome",
                          title="Gmail — Inbox")
    report = resolve_references(
        "what does it say?", graph=g, screen_summary=screen)
    it_res = next((r for r in report.resolutions
                   if r.token.lower() == "it"), None)
    assert it_res is not None
    assert "Chrome" in it_res.display_name


def test_that_resolves_like_it():
    g = _fresh_graph()
    screen = _StubScreen(app="Notepad",
                          title="todo.txt")
    report = resolve_references(
        "summarize that", graph=g, screen_summary=screen)
    that_res = next((r for r in report.resolutions
                     if r.token.lower() == "that"), None)
    assert that_res is not None


# ---- multi-word demonstratives ---------------------------------------

def test_the_one_resolves_to_thing():
    g = _fresh_graph()
    g.upsert_entity(
        kind=EntityKind.DOCUMENT.value, name="Pricing Memo")
    session = _StubSession([
        _StubTurn("user", "open the Pricing Memo"),
    ])
    report = resolve_references(
        "send the one we just opened", graph=g,
        session_buffer=session)
    res = next((r for r in report.resolutions
                if r.token.lower() == "the one"), None)
    assert res is not None


# ---- prompt block rendering ------------------------------------------

def test_prompt_block_contains_resolutions():
    g = _fresh_graph()
    g.upsert_entity(
        kind=EntityKind.PERSON.value, name="Dani",
        attrs={"email": "dani@x"})
    report = resolve_references(
        "send Dani the report", graph=g)
    assert "RESOLVED REFERENCES:" in report.prompt_block
    assert "Dani" in report.prompt_block
    assert "dani@x" in report.prompt_block


def test_empty_resolutions_empty_block():
    g = _fresh_graph()
    report = resolve_references("hello", graph=g)
    assert report.prompt_block == ""


def test_duplicate_resolutions_deduped_in_block():
    g = _fresh_graph()
    g.upsert_entity(
        kind=EntityKind.PERSON.value, name="Dani",
        attrs={"email": "dani@x"})
    session = _StubSession([_StubTurn("user", "Dani is on this")])
    report = resolve_references(
        "tell him and ping him later", graph=g,
        session_buffer=session)
    # 'him' appears twice in input but should render once.
    body = report.prompt_block.split("\n")
    him_lines = [ln for ln in body if "him" in ln.lower()]
    assert len(him_lines) <= 1


# ---- empty / safety ---------------------------------------------------

def test_empty_text_returns_empty_report():
    report = resolve_references("", graph=None)
    assert report.original_text == ""
    assert report.resolutions == []


def test_resolver_tolerates_no_graph():
    """If the global graph isn't available, the resolver should
    silently return empty rather than raising."""
    report = resolve_references("send Dani hi", graph=None)
    # No graph + no session + no screen → no resolutions, no crash.
    assert isinstance(report, ResolutionReport)

"""Tests for multimodal_fusion (Phase 5 B2)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.multimodal_fusion import (  # noqa: E402
    DEFAULT_BUDGET_CHARS, MIN_RESERVE_PER_SECTION,
    ContextSection, allocate_budget, build_unified_context,
    dedup_overlap, render, score_sections,
)


# ---- score_sections ----------------------------------------------------

def test_empty_parts_yields_no_sections():
    assert score_sections("hi", {}) == []
    assert score_sections("hi", {"memory": ""}) == []


def test_pronoun_request_boosts_session_relevance():
    sections = score_sections(
        "and that one?",
        {"session": "RECENT CONVERSATION:\nuser: x"})
    assert sections
    assert sections[0].name == "session"
    assert sections[0].relevance >= 0.8


def test_long_request_doesnt_boost_session():
    sections = score_sections(
        "Send Dani a long email about the Q3 roadmap that goes "
        "into detail on every single milestone and risk we're "
        "tracking for the rest of the year.",
        {"session": "RECENT CONVERSATION:\nuser: x"})
    assert sections
    # Long, self-contained → lower session score.
    assert sections[0].relevance < 0.7


def test_pinned_memory_gets_priority_bump():
    sections = score_sections(
        "what's Dani's email",
        {"memory": "FACTS:\n - auto-pin: dani -> dani@x",
         "repo": "REPO: touchless"})
    # Memory comes first not just by relevance but by priority
    # bump from 'auto-pin' marker.
    assert sections[0].name == "memory"
    assert sections[0].priority == 1


def test_repo_request_boosts_repo_relevance():
    sections = score_sections(
        "what's on this branch?",
        {"repo": "REPO: touchless\n  branch: main"})
    assert sections[0].name == "repo"
    assert sections[0].relevance >= 0.9


# ---- allocate_budget --------------------------------------------------

def test_budget_split_proportional_to_relevance():
    sections = [
        ContextSection(name="a", raw_text="x" * 1000,
                       relevance=1.0),
        ContextSection(name="b", raw_text="y" * 1000,
                       relevance=0.5),
    ]
    allocated = allocate_budget(sections, budget=600)
    a_cap = next(c for s, c in allocated if s.name == "a")
    b_cap = next(c for s, c in allocated if s.name == "b")
    # A should get more than B by roughly 2:1.
    assert a_cap > b_cap
    # Total within budget.
    assert a_cap + b_cap <= 600


def test_budget_reserves_minimum_per_section():
    sections = [
        ContextSection(name="a", raw_text="x" * 1000,
                       relevance=0.99),
        ContextSection(name="b", raw_text="y" * 1000,
                       relevance=0.01),
    ]
    allocated = allocate_budget(sections, budget=400)
    b_cap = next(c for s, c in allocated if s.name == "b")
    # B has near-zero relevance but should still get the floor.
    assert b_cap >= min(MIN_RESERVE_PER_SECTION, 400 // 2)


def test_zero_budget_returns_empty():
    sections = [ContextSection(name="a", raw_text="x", relevance=1.0)]
    assert allocate_budget(sections, budget=0) == []


def test_caps_never_exceed_raw_text_len():
    sections = [
        ContextSection(name="a", raw_text="short",
                       relevance=1.0),
    ]
    allocated = allocate_budget(sections, budget=10_000)
    cap = allocated[0][1]
    assert cap <= len("short")


# ---- dedup_overlap ----------------------------------------------------

def test_dedup_keeps_distinct_sections():
    sections = [
        (ContextSection(name="memory", raw_text="FACTS: foo",
                        relevance=1.0), 100),
        (ContextSection(name="screen", raw_text="ON SCREEN: bar",
                        relevance=1.0), 100),
    ]
    deduped = dedup_overlap(sections)
    assert len(deduped) == 2


def test_dedup_drops_substring_duplicate():
    sections = [
        (ContextSection(name="memory",
                        raw_text="FACTS: dani -> dani@x",
                        relevance=1.0), 200),
        (ContextSection(name="session",
                        raw_text="FACTS: dani -> dani@x (echoed)",
                        relevance=0.5), 200),
    ]
    deduped = dedup_overlap(sections)
    # Second is dropped because its head is a substring of the first.
    assert len(deduped) == 1
    assert deduped[0][0].name == "memory"


# ---- render ordering --------------------------------------------------

def test_render_orders_memory_first_then_session():
    parts = [
        (ContextSection(name="screen", raw_text="ON SCREEN: x"),
         "ON SCREEN: x"),
        (ContextSection(name="memory", raw_text="FACTS: a"),
         "FACTS: a"),
        (ContextSection(name="session", raw_text="RECENT: b"),
         "RECENT: b"),
    ]
    out = render(parts)
    # Stable order regardless of input order.
    assert out.find("FACTS:") < out.find("RECENT:") < out.find(
        "ON SCREEN:")


def test_render_empty_returns_empty():
    assert render([]) == ""


# ---- build_unified_context end-to-end ---------------------------------

def test_full_pipeline_returns_block_under_budget():
    text = "what does this say?"
    parts = {
        "memory": "FACTS:\n - dani -> dani@x" * 50,
        "session": "RECENT: 4 prior turns" * 30,
        "screen": "ON SCREEN: Gmail inbox; 24 elements" * 20,
        "repo": "",   # empty — should be dropped
        "dictation": "",
    }
    out = build_unified_context(text, parts, budget=600)
    assert out
    assert len(out) <= 800   # some overhead for separators
    # Vision-relevant request should include the screen block.
    assert "ON SCREEN" in out


def test_full_pipeline_skips_empty_modalities():
    out = build_unified_context("hi", {"memory": "", "screen": ""})
    assert out == ""


def test_full_pipeline_handles_single_modality():
    out = build_unified_context(
        "what's my workday start?",
        {"memory": "FACTS: workday_start = 9am"},
        budget=500)
    assert "9am" in out


def test_full_pipeline_priority_keeps_pinned_memory():
    """Even when budget is tight, pinned memory should survive."""
    text = "what's Dani's email?"
    parts = {
        "memory": "FACTS:\n - auto-pin: dani -> dani@x",
        "repo": "REPO: touchless\n  branch: main\n" + ("y" * 500),
    }
    out = build_unified_context(text, parts, budget=200)
    assert "dani" in out.lower()

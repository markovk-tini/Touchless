"""Tests for ContextCompressor (Phase 2 B3)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.context_compressor import (  # noqa: E402
    ContextChunk, ContextCompressor, KEEP_RECENT_TURNS,
    _heuristic_summarize, build_chunks_from_recall,
)


def _turn(text: str, age: int) -> ContextChunk:
    return ContextChunk(kind="turn", text=text, age_turns=age)


def _fact(text: str) -> ContextChunk:
    return ContextChunk(kind="fact", text=text, weight=2.0)


# ---- happy paths ------------------------------------------------------

def test_compress_empty_returns_empty_string():
    c = ContextCompressor()
    assert c.compress([]) == ""


def test_compress_keeps_recent_turns_verbatim():
    c = ContextCompressor(soft_limit_chars=200, hard_limit_chars=4000)
    chunks = [_turn(f"turn-{i}: " + "x" * 40, i) for i in range(8)]
    out = c.compress(chunks)
    # The KEEP_RECENT_TURNS most recent (lowest age) should appear.
    for i in range(KEEP_RECENT_TURNS):
        assert f"turn-{i}:" in out


def test_compress_summarizes_old_turns():
    c = ContextCompressor(soft_limit_chars=200, hard_limit_chars=4000)
    chunks = [_turn(f"turn-{i}: " + "y" * 100, i) for i in range(8)]
    out = c.compress(chunks)
    # Section header for older turns should appear.
    assert "EARLIER:" in out
    # And it should be shorter than raw text — each old turn must be
    # condensed.
    assert len(out) < sum(len(c.text) for c in chunks)


def test_compress_includes_facts_section():
    c = ContextCompressor()
    chunks = [
        _fact("user lives in Berlin"),
        _turn("turn-0: hi", 0),
    ]
    out = c.compress(chunks)
    assert "FACTS:" in out
    assert "user lives in Berlin" in out


def test_compress_drops_oldest_summaries_under_hard_limit():
    # Hard-limit must be respected even when many old turns exist.
    c = ContextCompressor(soft_limit_chars=100, hard_limit_chars=300)
    chunks = ([_fact("F1"), _fact("F2")]
              + [_turn(f"recent-{i}: " + "z" * 30, i)
                 for i in range(KEEP_RECENT_TURNS)]
              + [_turn(f"old-{i}: " + "w" * 30, i + KEEP_RECENT_TURNS)
                 for i in range(20)])
    out = c.compress(chunks)
    assert len(out) <= 300


def test_would_exceed_budget_returns_true_over_soft_limit():
    c = ContextCompressor(soft_limit_chars=100, hard_limit_chars=200)
    assert c.would_exceed_budget("x" * 150) is True
    assert c.would_exceed_budget("x" * 50) is False
    assert c.would_exceed_budget("") is False


# ---- ordering ---------------------------------------------------------

def test_compress_orders_recent_newest_first():
    c = ContextCompressor()
    chunks = [_turn("oldest-recent", 3),
              _turn("middle-recent", 2),
              _turn("newest-recent", 0)]
    out = c.compress(chunks)
    pos_newest = out.find("newest-recent")
    pos_oldest = out.find("oldest-recent")
    assert pos_newest != -1 and pos_oldest != -1
    # Newest should appear before oldest in the rendered block.
    assert pos_newest < pos_oldest


# ---- heuristic summarize ----------------------------------------------

def test_heuristic_summary_passes_through_short_text():
    assert _heuristic_summarize("hi") == "hi"


def test_heuristic_summary_truncates_long_text():
    long_text = "what is the weather " * 20  # ~400 chars
    s = _heuristic_summarize(long_text)
    assert len(s) <= 100  # 80 chars body + summary tag
    assert "…" in s or "checked weather" in s


def test_heuristic_summary_recognizes_tool_verb():
    text = ("I asked Iris to look something up. weather_get ran and "
            + "returned a forecast for next week with multiple cities listed.")
    s = _heuristic_summarize(text)
    assert "checked weather" in s


def test_heuristic_summary_strips_disfluencies():
    s = _heuristic_summarize("um like, weather please, you know")
    assert "um" not in s.lower()
    assert "you know" not in s.lower()


# ---- LLM backend ------------------------------------------------------

class _FakeLLM:
    def __init__(self, summary: str):
        self.summary = summary
        self.calls = 0

    def plan(self, goal: str):
        self.calls += 1

        class P:
            pass

        p = P()
        p.goal = self.summary
        return p


def test_llm_backend_used_when_enabled(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_CONTEXT_LLM", "1")
    fake = _FakeLLM(summary="user asked Iris to check Berlin weather")
    c = ContextCompressor(llm_planner=fake)
    chunks = [_turn(f"turn-{i}: something verbose " + "x" * 80, i)
              for i in range(8)]
    out = c.compress(chunks)
    assert fake.calls > 0
    assert "Berlin" in out or "weather" in out


def test_llm_backend_falls_back_to_heuristic_on_none(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_CONTEXT_LLM", "1")

    class BadLLM:
        def plan(self, goal):
            return None

    c = ContextCompressor(llm_planner=BadLLM())
    chunks = [_turn(f"old turn-{i} " + "x" * 100, i)
              for i in range(8)]
    out = c.compress(chunks)
    # Should still produce SOME compressed output.
    assert "EARLIER:" in out


# ---- build_chunks_from_recall -----------------------------------------

def test_build_chunks_from_recall_handles_facts_and_episodes():
    class EP:
        def __init__(self, text):
            self.text = text

    recall = {
        "facts": ["user lives in Berlin", "Dani's email is dani@x"],
        "episodes": [EP("turn-A"), EP("turn-B")],
    }
    chunks = build_chunks_from_recall(recall)
    kinds = [c.kind for c in chunks]
    assert kinds.count("fact") == 2
    assert kinds.count("turn") == 2


def test_build_chunks_from_recall_empty_returns_empty_list():
    assert build_chunks_from_recall({}) == []
    assert build_chunks_from_recall({"facts": [], "episodes": []}) == []

"""Tests for behavior_log (Phase 9 B2)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.behavior_log import (  # noqa: E402
    BehaviorLog, SuggestedDefault, _utterance_pattern,
)


def _new_log(tmp_path: Path) -> BehaviorLog:
    return BehaviorLog(db_path=tmp_path / "b.db")


# ---- pattern -------------------------------------------------------

def test_pattern_basic():
    assert _utterance_pattern("open chrome browser") == "open chrome browser"


def test_pattern_drops_stopwords():
    assert _utterance_pattern("open the chrome browser") == "open chrome browser"


def test_pattern_caps_at_three_words():
    out = _utterance_pattern("send email to dani about q3")
    # 5 content tokens → first 3.
    assert len(out.split()) == 3


def test_pattern_empty():
    assert _utterance_pattern("") == ""


def test_pattern_only_stopwords():
    assert _utterance_pattern("the and a") == ""


# ---- log_dispatch + mark_accepted ---------------------------------

def test_log_dispatch_creates_pending(tmp_path):
    log = _new_log(tmp_path)
    id_ = log.log_dispatch("open chrome", "browser_open",
                            {"target": "chrome"})
    assert id_ > 0
    stats = log.stats()
    assert stats.get("pending", 0) == 1


def test_mark_accepted_updates_outcome(tmp_path):
    log = _new_log(tmp_path)
    id_ = log.log_dispatch("open chrome", "browser_open")
    assert log.mark_accepted(id_) is True
    stats = log.stats()
    assert stats.get("accepted", 0) == 1


def test_mark_accepted_idempotent(tmp_path):
    log = _new_log(tmp_path)
    id_ = log.log_dispatch("x", "tool_x")
    log.mark_accepted(id_)
    # Second call — row already accepted, returns False.
    assert log.mark_accepted(id_) is False


def test_mark_accepted_missing_row(tmp_path):
    log = _new_log(tmp_path)
    assert log.mark_accepted(9999) is False


# ---- corrections ---------------------------------------------------

def test_log_correction_records_corrected(tmp_path):
    log = _new_log(tmp_path)
    log.log_dispatch("open chrome", "browser_open")
    log.log_correction("open chrome", "browser_open",
                        "edge_open")
    stats = log.stats()
    assert stats.get("corrected", 0) >= 1


# ---- suggestions ---------------------------------------------------

def test_suggestions_fire_above_threshold(tmp_path):
    log = _new_log(tmp_path)
    for _ in range(3):
        log.log_correction(
            "open chrome", "browser_open", "edge_open")
    suggestions = log.find_pattern_suggestions(
        min_corrections=3)
    assert len(suggestions) == 1
    s = suggestions[0]
    assert s.pattern == "open chrome"
    assert s.original_tool == "browser_open"
    assert s.suggested_tool == "edge_open"
    assert s.correction_count >= 3


def test_suggestions_skip_below_threshold(tmp_path):
    log = _new_log(tmp_path)
    log.log_correction("open chrome", "a", "b")
    log.log_correction("open chrome", "a", "b")
    # Only 2 → below default threshold of 3.
    assert log.find_pattern_suggestions(min_corrections=3) == []


def test_suggestions_grouped_by_corrected_tool(tmp_path):
    log = _new_log(tmp_path)
    for _ in range(3):
        log.log_correction("open chrome", "a", "b")
    for _ in range(3):
        log.log_correction("open chrome", "a", "c")
    suggestions = log.find_pattern_suggestions(
        min_corrections=3)
    # Two distinct suggested tools → two rows.
    assert len(suggestions) == 2


def test_suggestion_headline_mentions_count(tmp_path):
    log = _new_log(tmp_path)
    for _ in range(4):
        log.log_correction("open chrome", "a", "b")
    s = log.find_pattern_suggestions(min_corrections=3)[0]
    assert "4 times" in s.headline or "4" in s.headline


# ---- apply_suggestion + preferred_tool ---------------------------

def test_apply_suggestion_persists(tmp_path):
    log = _new_log(tmp_path)
    assert log.apply_suggestion("open chrome", "edge_open") is True
    assert log.preferred_tool("open chrome") == "edge_open"


def test_apply_suggestion_overrides(tmp_path):
    log = _new_log(tmp_path)
    log.apply_suggestion("open chrome", "a")
    log.apply_suggestion("open chrome", "b")
    assert log.preferred_tool("open chrome") == "b"


def test_apply_suggestion_rejects_empty(tmp_path):
    log = _new_log(tmp_path)
    assert log.apply_suggestion("", "x") is False
    assert log.apply_suggestion("pattern", "") is False


def test_preferred_tool_returns_none_when_no_match(tmp_path):
    log = _new_log(tmp_path)
    assert log.preferred_tool("anything") is None


def test_suggestions_skip_when_preference_applied(tmp_path):
    log = _new_log(tmp_path)
    for _ in range(3):
        log.log_correction("open chrome", "a", "b")
    log.apply_suggestion("open chrome", "b")
    # No longer suggests once user agreed.
    assert log.find_pattern_suggestions(min_corrections=3) == []


# ---- persistence ---------------------------------------------------

def test_persistence_across_reopen(tmp_path):
    log = BehaviorLog(db_path=tmp_path / "b.db")
    log.apply_suggestion("open chrome", "edge_open")
    log.close()
    log2 = BehaviorLog(db_path=tmp_path / "b.db")
    assert log2.preferred_tool("open chrome") == "edge_open"


# ---- stats ---------------------------------------------------------

def test_stats_reflect_all_outcomes(tmp_path):
    log = _new_log(tmp_path)
    id1 = log.log_dispatch("a one two", "tool_a")
    log.mark_accepted(id1)
    log.log_dispatch("b one two", "tool_b")  # stays pending
    log.log_correction("c one two", "tool_c", "tool_d")
    stats = log.stats()
    assert stats["accepted"] >= 1
    assert stats["pending"] >= 1
    assert stats["corrected"] >= 1

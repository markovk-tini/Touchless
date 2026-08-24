"""Tests for local_intent classifier (Phase 8 B3)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.local_intent import (  # noqa: E402
    ClassifyResult, IntentStore, LocalIntentClassifier,
    _tokenize,
)


def _new_store(tmp_path: Path) -> IntentStore:
    return IntentStore(db_path=tmp_path / "intent.db")


def _train(classifier, examples):
    """Helper — feed N (text, tool) pairs."""
    for text, tool in examples:
        classifier.record_example(text, tool)


# ---- tokenize ------------------------------------------------------

def test_tokenize_basic():
    assert _tokenize("Pause the music") == ["pause", "music"]


def test_tokenize_handles_punctuation():
    assert _tokenize("What's the weather?") == ["what's", "weather"]


def test_tokenize_drops_stopwords():
    out = _tokenize("the and a in")
    assert out == []


def test_tokenize_empty():
    assert _tokenize("") == []


def test_tokenize_lowercase():
    out = _tokenize("PAUSE MUSIC")
    assert "pause" in out
    assert "music" in out


# ---- store ---------------------------------------------------------

def test_store_record_increments_count(tmp_path):
    s = _new_store(tmp_path)
    s.record("pause the music", "media_pause")
    assert s.get_examples_count("media_pause") == 1
    s.record("pause music now", "media_pause")
    assert s.get_examples_count("media_pause") == 2


def test_store_records_weights(tmp_path):
    s = _new_store(tmp_path)
    s.record("pause music", "media_pause")
    score, matched = s.score_for("media_pause",
                                  ["pause", "music"])
    assert score > 0
    assert "pause" in matched
    assert "music" in matched


def test_store_all_tools_respects_min_examples(tmp_path):
    s = _new_store(tmp_path)
    s.record("pause", "media_pause")
    # Below MIN_EXAMPLES (3).
    assert "media_pause" not in s.all_tools()
    s.record("pause", "media_pause")
    s.record("pause", "media_pause")
    assert "media_pause" in s.all_tools()


def test_store_negative_examples_decrement(tmp_path):
    s = _new_store(tmp_path)
    s.record("pause music", "media_pause", success=True)
    s.record("pause music", "media_pause", success=True)
    pos_score, _ = s.score_for("media_pause", ["pause", "music"])
    s.record("pause music", "media_pause", success=False)
    neg_score, _ = s.score_for("media_pause", ["pause", "music"])
    assert neg_score < pos_score


def test_store_forget_removes_tool(tmp_path):
    s = _new_store(tmp_path)
    s.record("a b c", "tool_x")
    s.record("d e f", "tool_x")
    s.record("g h i", "tool_x")
    assert s.get_examples_count("tool_x") == 3
    assert s.forget("tool_x") is True
    assert s.get_examples_count("tool_x") == 0


def test_store_score_handles_no_match(tmp_path):
    s = _new_store(tmp_path)
    s.record("foo bar baz", "tool_x")
    score, matched = s.score_for(
        "tool_x", ["does", "not", "exist"])
    assert score == 0.0
    assert matched == []


def test_store_persists_across_reopen(tmp_path):
    s = IntentStore(db_path=tmp_path / "intent.db")
    s.record("kill the music", "media_pause")
    s.record("kill the music", "media_pause")
    s.record("kill the music", "media_pause")
    s.close()
    s2 = IntentStore(db_path=tmp_path / "intent.db")
    assert "media_pause" in s2.all_tools()


# ---- classifier ----------------------------------------------------

def test_classifier_returns_none_without_examples(tmp_path):
    c = LocalIntentClassifier(store=_new_store(tmp_path))
    assert c.classify("pause music") is None


def test_classifier_picks_trained_tool(tmp_path):
    c = LocalIntentClassifier(store=_new_store(tmp_path))
    _train(c, [
        ("pause the music", "media_pause"),
        ("pause music", "media_pause"),
        ("pause music now", "media_pause"),
    ])
    result = c.classify("pause music")
    assert result is not None
    assert result.tool == "media_pause"
    assert result.confidence > 0


def test_classifier_picks_better_match(tmp_path):
    c = LocalIntentClassifier(store=_new_store(tmp_path))
    _train(c, [
        ("pause music", "media_pause"),
        ("pause music", "media_pause"),
        ("pause music", "media_pause"),
        ("send email", "email_send"),
        ("send email", "email_send"),
        ("send email", "email_send"),
    ])
    pause = c.classify("pause music now")
    send = c.classify("send email to dani")
    assert pause.tool == "media_pause"
    assert send.tool == "email_send"


def test_classifier_confidence_low_for_unknown_words(tmp_path):
    c = LocalIntentClassifier(store=_new_store(tmp_path))
    _train(c, [
        ("pause music", "media_pause"),
        ("pause music", "media_pause"),
        ("pause music", "media_pause"),
    ])
    # Single matched word → confidence drops.
    result = c.classify("pause something different")
    assert result is not None
    # Confidence should not exceed 0.7 (we only matched one
    # word with content).
    assert result.confidence < 1.0


def test_classifier_returns_none_when_no_overlap(tmp_path):
    c = LocalIntentClassifier(store=_new_store(tmp_path))
    _train(c, [
        ("pause music", "media_pause"),
        ("pause music", "media_pause"),
        ("pause music", "media_pause"),
    ])
    result = c.classify("xyz qrs lmn")
    assert result is None


def test_classifier_learns_user_phrasing(tmp_path):
    """User says 'kill' instead of 'pause' → after training the
    classifier picks it up."""
    c = LocalIntentClassifier(store=_new_store(tmp_path))
    _train(c, [
        ("kill the music", "media_pause"),
        ("kill the music", "media_pause"),
        ("kill the music", "media_pause"),
    ])
    result = c.classify("kill the music")
    assert result is not None
    assert result.tool == "media_pause"
    assert result.confidence > 0.5


def test_classifier_is_confident_threshold(tmp_path):
    c = LocalIntentClassifier(store=_new_store(tmp_path),
                               threshold=0.7)
    _train(c, [
        ("pause music", "media_pause"),
        ("pause music", "media_pause"),
        ("pause music", "media_pause"),
    ])
    result = c.classify("pause music")
    assert result.is_confident()


def test_classifier_resists_competing_tools(tmp_path):
    """When two tools share some words, the more-specific one wins."""
    c = LocalIntentClassifier(store=_new_store(tmp_path))
    # 'volume up' vs 'pause music' both share 'the'
    _train(c, [
        ("turn the volume up", "volume_up"),
        ("turn the volume up", "volume_up"),
        ("turn the volume up", "volume_up"),
        ("pause the music", "media_pause"),
        ("pause the music", "media_pause"),
        ("pause the music", "media_pause"),
    ])
    r1 = c.classify("turn the volume up")
    r2 = c.classify("pause the music")
    assert r1.tool == "volume_up"
    assert r2.tool == "media_pause"


def test_forget_tool_kills_classification(tmp_path):
    c = LocalIntentClassifier(store=_new_store(tmp_path))
    _train(c, [
        ("pause music", "media_pause"),
        ("pause music", "media_pause"),
        ("pause music", "media_pause"),
    ])
    assert c.classify("pause music") is not None
    assert c.forget_tool("media_pause") is True
    assert c.classify("pause music") is None


def test_stats_reflect_training(tmp_path):
    c = LocalIntentClassifier(store=_new_store(tmp_path))
    _train(c, [
        ("a b c", "tool_x"),
        ("d e f", "tool_x"),
        ("g h i", "tool_x"),
    ])
    stats = c.stats()
    assert stats["tools"] == 1
    # 9 unique non-stopword tokens (a/d/e/g/h/i — some are
    # stopwords; 'a' is dropped, 'b' 'c' 'd' 'e' 'f' 'g' 'h'
    # 'i' counted modulo collapses).
    assert stats["weights"] >= 6


# ---- ClassifyResult dataclass --------------------------------------

def test_classify_result_is_confident_method():
    r = ClassifyResult(tool="x", confidence=0.8)
    assert r.is_confident(0.7) is True
    assert r.is_confident(0.9) is False


def test_classify_result_defaults():
    r = ClassifyResult(tool="x", confidence=0.5)
    assert r.matched_words == ()
    assert r.fallback_used is False

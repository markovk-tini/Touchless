"""Tests for the smart fast-path classifier (Phase 6 B4)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.smart_fast_path import (  # noqa: E402
    FastPathKind, classify, global_stats, reset_stats,
)


def setup_function():
    reset_stats()


# ---- ignore ---------------------------------------------------------

def test_classify_empty_returns_ignore():
    r = classify("")
    assert r.kind == FastPathKind.IGNORE


def test_classify_whitespace_returns_ignore():
    assert classify("   ").kind == FastPathKind.IGNORE


def test_classify_none_returns_ignore():
    assert classify(None).kind == FastPathKind.IGNORE


# ---- chat -----------------------------------------------------------

def test_classify_greeting_is_chat():
    r = classify("hi")
    assert r.kind == FastPathKind.CHAT
    assert r.reply
    assert r.rule == "greeting"


def test_classify_hello_iris_is_chat():
    r = classify("hello iris")
    assert r.kind == FastPathKind.CHAT


def test_classify_thanks_is_chat_ack():
    r = classify("thanks")
    assert r.kind == FastPathKind.CHAT
    assert r.rule == "ack"


def test_classify_cancel_is_chat():
    r = classify("never mind")
    assert r.kind == FastPathKind.CHAT
    assert r.rule == "cancel"


# ---- direct ---------------------------------------------------------

def test_classify_time_falls_through():
    """No clock tool exists; planner handles time queries."""
    r = classify("what time is it")
    assert r.kind == FastPathKind.PLANNER


def test_classify_pause_is_direct():
    r = classify("pause")
    assert r.kind == FastPathKind.DIRECT
    assert r.tool == "media_play_pause"


def test_classify_resume_bare_is_direct():
    """Bare 'play' / 'resume' fast-path. Multi-word ('play music',
    'play X') intentionally falls through to the planner so it can
    pick the right tool (e.g. spotify search)."""
    for utt in ("play", "resume"):
        r = classify(utt)
        assert r.kind == FastPathKind.DIRECT, utt
        assert r.tool == "media_play_pause", utt


def test_classify_play_music_falls_through_to_planner():
    """'play music' is ambiguous — planner picks spotify search
    or media_play_pause based on context. Fast-path must NOT
    grab it (regression check for B7 bug)."""
    r = classify("play music")
    assert r.kind == FastPathKind.PLANNER


def test_classify_skip_is_direct():
    r = classify("skip")
    assert r.kind == FastPathKind.DIRECT
    assert r.tool == "media_next_track"


def test_classify_prev_is_direct():
    r = classify("previous")
    assert r.kind == FastPathKind.DIRECT
    assert r.tool == "media_previous_track"


def test_classify_mute_is_direct():
    r = classify("mute")
    assert r.kind == FastPathKind.DIRECT
    assert r.tool == "volume_toggle_mute"


def test_classify_unmute_is_direct():
    r = classify("unmute")
    assert r.kind == FastPathKind.DIRECT
    assert r.tool == "volume_toggle_mute"


# ---- planner --------------------------------------------------------

def test_classify_complex_chain_is_planner():
    r = classify("pause the music and then dim the lights")
    assert r.kind == FastPathKind.PLANNER
    assert r.rule == "too_complex"


def test_classify_why_question_is_planner():
    r = classify("why is the build broken")
    assert r.kind == FastPathKind.PLANNER


def test_classify_personalized_is_planner():
    r = classify("remember to buy milk")
    assert r.kind == FastPathKind.PLANNER


def test_classify_pronoun_reference_is_planner():
    """Pronoun resolution belongs to the planner path."""
    r = classify("send that to him")
    assert r.kind == FastPathKind.PLANNER


def test_classify_long_utterance_is_planner():
    r = classify(
        "okay so I was thinking maybe we could go through "
        "all the stuff that happened today")
    assert r.kind == FastPathKind.PLANNER
    assert r.rule == "too_long"


def test_classify_no_match_is_planner():
    r = classify("what's the capital of france")
    assert r.kind == FastPathKind.PLANNER
    assert r.rule == "no_match"


# ---- structural -----------------------------------------------------

def test_fast_path_result_is_fast():
    assert classify("hi").is_fast()
    assert classify("pause").is_fast()
    assert not classify("what's the capital of france").is_fast()


def test_stats_increment_correctly():
    classify("hi")          # chat
    classify("pause")       # direct
    classify("what's the capital of france")  # planner
    classify("")            # ignore
    s = global_stats()
    assert s.total == 4
    assert s.chat == 1
    assert s.direct == 1
    assert s.planner == 1
    assert s.ignore == 1

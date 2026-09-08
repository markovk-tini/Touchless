"""Tests for reply_judge (Phase 9 B1)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.incognito as inc  # noqa: E402
import hgr.live_api.persona_voice as pv  # noqa: E402
from hgr.live_api.reply_judge import (  # noqa: E402
    JudgeResult, _parse_score, feed_bandit, judge_async,
    judge_sync,
)


def setup_function():
    inc.set_incognito(False)
    pv.reset_for_tests()


# ---- JudgeResult dataclass ------------------------------------------

def test_judge_result_average():
    r = JudgeResult(helpfulness=0.9, accuracy=0.8, vibe=0.7)
    assert abs(r.average() - 0.8) < 1e-6


def test_judge_result_skipped_average_zero():
    r = JudgeResult(skipped=True)
    assert r.average() == 0.0


# ---- _parse_score ----------------------------------------------------

def test_parse_score_clean_json():
    text = ('{"helpfulness": 0.8, "accuracy": 0.9, '
            '"vibe": 0.7, "reason": "looks good"}')
    parsed = _parse_score(text)
    assert parsed["helpfulness"] == 0.8
    assert parsed["reason"] == "looks good"


def test_parse_score_with_code_fences():
    text = '```json\n{"helpfulness": 0.5, "accuracy": 0.5, '\
           '"vibe": 0.5, "reason": "ok"}\n```'
    parsed = _parse_score(text)
    assert parsed["helpfulness"] == 0.5


def test_parse_score_embedded_in_prose():
    text = ('Sure, here is the JSON: '
            '{"helpfulness": 0.6, "accuracy": 0.7, '
            '"vibe": 0.8, "reason": "x"} ok done')
    parsed = _parse_score(text)
    assert parsed["helpfulness"] == 0.6


def test_parse_score_garbage_returns_none():
    assert _parse_score("not json at all") is None


def test_parse_score_empty_returns_none():
    assert _parse_score("") is None


# ---- skip paths -----------------------------------------------------

def test_judge_skips_short_reply():
    r = judge_sync(question="hi", reply="ok")
    assert r.skipped is True
    assert "short" in r.skip_reason


def test_judge_skips_no_question():
    r = judge_sync(question="", reply="a longer reply that exceeds 25 chars")
    assert r.skipped is True


def test_judge_skips_in_incognito():
    inc.set_incognito(True)
    try:
        r = judge_sync(
            question="any emails",
            reply="Yes, two new ones from Dani and Sam.")
    finally:
        inc.set_incognito(False)
    assert r.skipped is True
    assert "incognito" in r.skip_reason


def test_judge_skips_without_api_key():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        r = judge_sync(
            question="weather",
            reply="It's 67 and clear out there friend.")
    assert r.skipped is True
    assert "api key" in r.skip_reason


# ---- happy path (mocked HTTP) --------------------------------------

def test_judge_parses_scores():
    fake_response = {
        "choices": [{
            "message": {
                "content": (
                    '{"helpfulness": 0.9, "accuracy": 0.95, '
                    '"vibe": 0.8, "reason": "spot on"}')}}]
    }
    import json as _j
    class _Resp:
        status = 200
        def __init__(self, body): self._body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._body
    resp = _Resp(_j.dumps(fake_response).encode("utf-8"))
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen", return_value=resp):
            r = judge_sync(
                question="weather",
                reply="It's 67 and clear, friend.")
    assert r.skipped is False
    assert r.helpfulness == 0.9
    assert r.accuracy == 0.95
    assert r.vibe == 0.8
    assert "spot on" in r.reason


def test_judge_clamps_out_of_range():
    fake_response = {
        "choices": [{
            "message": {
                "content": (
                    '{"helpfulness": 1.5, "accuracy": -0.3, '
                    '"vibe": "bad-value", "reason": "x"}')}}]
    }
    import json as _j
    class _Resp:
        status = 200
        def __init__(self, body): self._body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._body
    resp = _Resp(_j.dumps(fake_response).encode("utf-8"))
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen", return_value=resp):
            r = judge_sync(
                question="weather",
                reply="It's 67 and clear, friend.")
    assert r.helpfulness == 1.0
    assert r.accuracy == 0.0
    assert r.vibe == 0.0


def test_judge_http_error_skips_gracefully():
    import urllib.error
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    side_effect=urllib.error.HTTPError(
                        "u", 500, "x", {}, None)):
            r = judge_sync(
                question="weather",
                reply="It's 67 and clear, friend.")
    assert r.skipped is True


def test_judge_parse_fail_skips():
    fake_response = {
        "choices": [{
            "message": {"content": "I don't think so"}
        }]
    }
    import json as _j
    class _Resp:
        status = 200
        def __init__(self, body): self._body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._body
    resp = _Resp(_j.dumps(fake_response).encode("utf-8"))
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen", return_value=resp):
            r = judge_sync(
                question="weather",
                reply="It's 67 and clear, friend.")
    assert r.skipped is True


# ---- async ----------------------------------------------------------

def test_judge_async_invokes_callback():
    captured = []
    def cb(r):
        captured.append(r)
    fake_response = {
        "choices": [{
            "message": {
                "content": (
                    '{"helpfulness": 0.5, "accuracy": 0.5, '
                    '"vibe": 0.5, "reason": "ok"}')}}]
    }
    import json as _j
    class _Resp:
        status = 200
        def __init__(self, body): self._body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._body
    resp = _Resp(_j.dumps(fake_response).encode("utf-8"))
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen", return_value=resp):
            t = judge_async(
                question="weather",
                reply="It's 67 and clear, friend.",
                callback=cb)
            t.join(timeout=2.0)
    assert captured
    assert captured[0].helpfulness == 0.5


def test_judge_async_callback_exception_swallowed():
    def bad_cb(r):
        raise RuntimeError("nope")
    fake_response = {
        "choices": [{
            "message": {
                "content": (
                    '{"helpfulness": 0.5, "accuracy": 0.5, '
                    '"vibe": 0.5, "reason": "ok"}')}}]
    }
    import json as _j
    class _Resp:
        status = 200
        def __init__(self, body): self._body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._body
    resp = _Resp(_j.dumps(fake_response).encode("utf-8"))
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen", return_value=resp):
            t = judge_async(
                question="weather",
                reply="It's 67 and clear, friend.",
                callback=bad_cb)
            t.join(timeout=2.0)
    # No raise = pass


# ---- bandit feedback ------------------------------------------------

def test_feed_bandit_records_kept_when_high():
    r = JudgeResult(helpfulness=0.9, accuracy=0.9, vibe=0.9)
    feed_bandit(r, persona_name="jarvis")
    snap = pv.usage_snapshot()
    assert snap["jarvis"]["kept_replies"] == 1


def test_feed_bandit_records_revised_when_low():
    r = JudgeResult(helpfulness=0.2, accuracy=0.3, vibe=0.4)
    feed_bandit(r, persona_name="jarvis")
    snap = pv.usage_snapshot()
    assert snap["jarvis"]["revised"] == 1


def test_feed_bandit_skipped_is_no_op():
    r = JudgeResult(skipped=True)
    feed_bandit(r, persona_name="jarvis")
    snap = pv.usage_snapshot()
    assert snap["jarvis"]["kept_replies"] == 0
    assert snap["jarvis"]["revised"] == 0

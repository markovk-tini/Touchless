"""Tests for streaming_renderer (Phase 8 B1)."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.prose_renderer as _pr  # noqa: E402
from hgr.live_api.streaming_renderer import (  # noqa: E402
    last_render_passed_guard, stream_jarvis,
)


def setup_function():
    _pr._CACHE.clear()


class _FakeResp:
    """Stand-in for urlopen()'s response — context manager + iterable
    over SSE-style lines."""
    def __init__(self, lines, status=200):
        self._lines = list(lines)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        return iter(self._lines)


def _make_response_iter(chunks):
    import json as _j
    lines = []
    for c in chunks:
        if c == "[DONE]":
            lines.append(b"data: [DONE]\n")
        else:
            lines.append(
                f"data: {_j.dumps(c)}\n".encode("utf-8"))
    return _FakeResp(lines)


# ---- happy path -----------------------------------------------------

def test_stream_yields_chunks_in_order():
    chunks = [
        {"choices": [{"delta": {"content": "Two emails, sir. "}}]},
        {"choices": [{"delta": {"content": "Both routine."}}]},
        "[DONE]",
    ]
    result = {"count": 2, "messages": [
        {"from_name": "Dani", "subject": "hi"},
        {"from_name": "Sam", "subject": "ok"}]}
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch(
                "urllib.request.urlopen",
                return_value=_make_response_iter(chunks)):
            tokens = list(stream_jarvis(
                "any emails", "gmail_list", result,
                "You've got 2 emails — Dani and Sam."))
    # Non-final tokens then a single final marker.
    contents = [t for t, _ in tokens]
    finals = [f for _, f in tokens]
    assert "Two emails, sir. " in contents
    assert "Both routine." in contents
    assert finals[-1] is True


def test_stream_uses_cache_on_repeat():
    chunks = [
        {"choices": [{"delta": {"content": "67 and clear."}}]},
        "[DONE]",
    ]
    result = {"temp_f": 67, "conditions": "clear"}
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    return_value=_make_response_iter(chunks)):
            list(stream_jarvis(
                "weather", "weather_get", result,
                "It's 67 and clear out there."))
    # Second call: should NOT hit the network — cache returns the
    # final chunk in one shot.
    with patch("urllib.request.urlopen") as mock_open:
        tokens = list(stream_jarvis(
            "weather", "weather_get", result,
            "It's 67 and clear out there."))
        assert mock_open.call_count == 0
    final_tokens = [t for t, f in tokens if f]
    assert any("67" in t or "clear" in t.lower()
               for t in final_tokens)


# ---- degradation paths ---------------------------------------------

def test_stream_returns_fallback_when_no_api_key():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        tokens = list(stream_jarvis(
            "weather", "weather_get",
            {"temp_f": 67}, "67 and clear."))
    assert tokens == [("67 and clear.", True)]


def test_stream_returns_fallback_when_empty_fallback():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        tokens = list(stream_jarvis(
            "x", "tool_x", {"a": 1}, ""))
    assert tokens == [("", True)]


def test_stream_returns_fallback_when_skip_render():
    # Short status confirmation — render is skipped.
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        tokens = list(stream_jarvis(
            "mute", "volume_mute", {"muted": True}, "Muted."))
    assert tokens == [("Muted.", True)]


def test_stream_returns_fallback_when_tool_result_empty():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        tokens = list(stream_jarvis(
            "x", "tool_x", {}, "Nothing to report here."))
    assert tokens == [("Nothing to report here.", True)]


# ---- error paths ----------------------------------------------------

def test_stream_returns_fallback_on_http_error():
    import urllib.error
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    side_effect=urllib.error.HTTPError(
                        "u", 500, "x", {}, None)):
            tokens = list(stream_jarvis(
                "x", "y", {"a": 1}, "Some fallback message."))
    assert tokens[-1] == ("Some fallback message.", True)


def test_stream_returns_fallback_on_url_error():
    import urllib.error
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    side_effect=urllib.error.URLError("no net")):
            tokens = list(stream_jarvis(
                "x", "y", {"a": 1}, "Some fallback message."))
    assert tokens[-1] == ("Some fallback message.", True)


# ---- guard ----------------------------------------------------------

def test_guard_passes_when_facts_preserved():
    chunks = [
        {"choices": [{"delta": {"content": "Dani sent two — "}}]},
        {"choices": [{"delta": {"content": "Q3 and the schedule."}}]},
        "[DONE]",
    ]
    result = {"count": 2, "messages": [
        {"from_name": "Dani", "subject": "Q3"},
        {"from_name": "Dani", "subject": "schedule"},
    ]}
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    return_value=_make_response_iter(chunks)):
            list(stream_jarvis(
                "any emails", "gmail_list", result,
                "Two emails from Dani."))
    assert last_render_passed_guard() is True


def test_guard_fails_when_facts_hallucinated():
    """Output mentions no real sender → guard rejects."""
    chunks = [
        {"choices": [{"delta": {"content": "One from "
                                            "Alice "}}]},
        {"choices": [{"delta": {"content": "about Bob."}}]},
        "[DONE]",
    ]
    result = {"count": 1, "messages": [
        {"from_name": "Dani", "subject": "Q3"},
    ]}
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    return_value=_make_response_iter(chunks)):
            list(stream_jarvis(
                "any emails", "gmail_list", result,
                "One email from Dani about Q3."))
    assert last_render_passed_guard() is False


# ---- token bias from affect ----------------------------------------

def test_affect_terse_reduces_max_tokens(monkeypatch):
    """When affect says wants_terse, stream_jarvis caps max_tokens
    lower. We can't observe the HTTP body easily, but we can spy on
    json.dumps to confirm the cap was applied."""
    chunks = [
        {"choices": [{"delta": {"content": "67."}}]},
        "[DONE]",
    ]
    # Inject affect.reply_length_bias = -1.
    import hgr.live_api.affect as aff_mod
    monkeypatch.setattr(aff_mod, "reply_length_bias", lambda: -1)
    captured = {}
    real_dumps = __import__("json").dumps
    def spy_dumps(obj, *a, **kw):
        if isinstance(obj, dict) and "max_tokens" in obj:
            captured["max_tokens"] = obj["max_tokens"]
        return real_dumps(obj, *a, **kw)
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    return_value=_make_response_iter(chunks)):
            with patch("hgr.live_api.streaming_renderer.json.dumps",
                        side_effect=spy_dumps):
                # Use gmail_list (NOT in prose_renderer SKIP_RENDER_TOOLS)
                # — weather_get is now skipped because weather.py
                # already produces conversational summaries.
                list(stream_jarvis(
                    "summarize emails", "gmail_list",
                    {"messages": [{"from_name": "Dani",
                                    "subject": "hi"}],
                     "count": 1},
                    "You've got one new email from Dani."))
    assert captured.get("max_tokens", 9999) <= 200


# ---- preamble stripping --------------------------------------------

def test_stream_strips_sure_preamble_in_final():
    """The buffered output gets the preamble stripped on the final
    flush — we can verify via the cache write."""
    chunks = [
        {"choices": [{"delta": {"content": "Sure! "}}]},
        {"choices": [{"delta": {"content": "Just one new email "}}]},
        {"choices": [{"delta": {"content": "from Dani."}}]},
        "[DONE]",
    ]
    result = {"count": 1, "messages": [
        {"from_name": "Dani", "subject": "x"}]}
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    return_value=_make_response_iter(chunks)):
            # gmail_list goes through render (not in SKIP_RENDER_TOOLS);
            # weather_get is skipped because weather.py already produces
            # conversational output.
            list(stream_jarvis(
                "any emails", "gmail_list", result,
                "Just one new email from Dani."))
    # Check cache value — should NOT start with "Sure!".
    cached_values = list(_pr._CACHE.values())
    assert cached_values
    assert not any(v.lower().startswith("sure")
                   for v in cached_values)

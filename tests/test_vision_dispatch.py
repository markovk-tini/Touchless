"""Tests for vision_dispatch (Phase 4 B1 optional)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.vision_dispatch import (  # noqa: E402
    _build_request, _wrap_question, ask_about_screen,
    vision_available,
)


# ---- env gating --------------------------------------------------------

def test_vision_available_off_by_default(monkeypatch):
    monkeypatch.delenv("TOUCHLESS_VISION", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert vision_available() is False


def test_vision_available_requires_both_flags(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_VISION", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert vision_available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    assert vision_available() is True


def test_vision_blocked_when_over_cap(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_VISION", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    # Patch the meter so is_over_cap returns True.
    import hgr.live_api.cost_meter as cm
    class _Over:
        def is_over_cap(self): return True
    monkeypatch.setattr(cm, "global_meter", lambda: _Over())
    assert vision_available() is False


# ---- payload shape -----------------------------------------------------

def test_build_request_includes_image_and_text_blocks():
    payload = _build_request("claude-sonnet-4-6",
                              "what does this say?",
                              b"\x89PNG\r\n\x1a\n")
    assert payload["model"] == "claude-sonnet-4-6"
    msg = payload["messages"][0]
    assert msg["role"] == "user"
    types = [block["type"] for block in msg["content"]]
    assert "image" in types
    assert "text" in types
    image_block = next(b for b in msg["content"]
                       if b["type"] == "image")
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/png"


def test_wrap_question_adds_conversational_framing():
    out = _wrap_question("what's the subject?")
    assert "screenshot" in out.lower()
    assert "what's the subject?" in out


def test_wrap_question_strips_whitespace():
    out = _wrap_question("  hello   ")
    assert "hello" in out
    assert "  hello  " not in out


# ---- early bail-outs ---------------------------------------------------

def test_ask_about_screen_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("TOUCHLESS_VISION", raising=False)
    assert ask_about_screen("what's that?") is None


def test_ask_about_screen_returns_none_for_empty_question(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_VISION", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    assert ask_about_screen("") is None
    assert ask_about_screen(None) is None


def test_ask_about_screen_returns_none_when_capture_fails(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_VISION", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    import hgr.live_api.vision_dispatch as vd
    monkeypatch.setattr(vd, "_capture_screen_png", lambda: None)
    assert ask_about_screen("what?") is None

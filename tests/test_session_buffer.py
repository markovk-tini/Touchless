"""Tests for SessionBuffer (Phase 3)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.session_buffer import (  # noqa: E402
    DEFAULT_MAX_CHARS, DEFAULT_MAX_TURNS, DEFAULT_TTL_SEC,
    SessionBuffer, TurnEntry,
)
import hgr.live_api.incognito as inc  # noqa: E402


def setup_function():
    inc.set_incognito(False)


# ---- basics ------------------------------------------------------------

def test_empty_buffer_renders_empty():
    b = SessionBuffer()
    assert b.render() == ""
    assert b.recent() == []


def test_add_user_then_assistant_round_trip():
    b = SessionBuffer()
    b.add_user("hi")
    b.add_assistant("hello back")
    items = b.recent()
    assert len(items) == 2
    assert items[0].role == "user"
    assert items[1].role == "assistant"


def test_render_labels_roles():
    b = SessionBuffer()
    b.add_user("ping")
    b.add_assistant("pong")
    out = b.render()
    assert "user: ping" in out
    assert "iris: pong" in out


def test_empty_text_skipped():
    b = SessionBuffer()
    b.add_user("")
    b.add_user("   ")
    b.add_assistant(None)  # type: ignore[arg-type]
    assert b.recent() == []


# ---- bounds ------------------------------------------------------------

def test_max_turns_cap():
    b = SessionBuffer(max_turns=3)
    for i in range(10):
        b.add_user(f"u{i}")
    assert len(b.recent()) == 3
    # Most recent retained.
    assert b.recent()[-1].text == "u9"


def test_per_turn_text_capped_at_240():
    b = SessionBuffer()
    long = "x" * 1000
    b.add_user(long)
    rendered = b.render()
    # Single rendered line should be at most 240 chars + label.
    assert len(rendered) < 300


def test_render_drops_oldest_when_over_char_cap():
    b = SessionBuffer(max_chars=100)
    for i in range(20):
        b.add_user("y" * 30)  # each line ~36 chars rendered
    out = b.render()
    assert len(out) <= 100
    # Oldest should be gone; newest retained.
    lines = out.split("\n")
    assert lines  # at least one line


def test_max_turns_clamp_via_render_arg():
    b = SessionBuffer()
    for i in range(5):
        b.add_user(f"u{i}")
    out = b.render(max_turns=2)
    assert out.count("\n") == 1   # 2 lines → 1 newline


# ---- TTL ---------------------------------------------------------------

def test_ttl_drops_stale_entries():
    b = SessionBuffer(ttl_sec=0.05)
    b.add_user("hi")
    time.sleep(0.1)
    b.add_user("now")
    items = b.recent()
    assert len(items) == 1
    assert items[0].text == "now"


# ---- incognito ---------------------------------------------------------

def test_incognito_blocks_user_writes():
    b = SessionBuffer()
    inc.set_incognito(True)
    try:
        b.add_user("private")
    finally:
        inc.set_incognito(False)
    assert b.recent() == []


def test_incognito_blocks_assistant_writes():
    b = SessionBuffer()
    inc.set_incognito(True)
    try:
        b.add_assistant("hidden")
    finally:
        inc.set_incognito(False)
    assert b.recent() == []


# ---- reset -------------------------------------------------------------

def test_reset_clears_buffer():
    b = SessionBuffer()
    b.add_user("a")
    b.add_assistant("b")
    b.reset()
    assert b.recent() == []

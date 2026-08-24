"""Tests for ScreenAwareness ambient capture (Phase 4 B1)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.screen_awareness import (  # noqa: E402
    DEFAULT_CONTEXT_TTL_SEC, DEFAULT_INTERVAL_SEC,
    ScreenAwareness, ScreenSummary, looks_vision_relevant,
)
import hgr.live_api.incognito as inc  # noqa: E402
import hgr.live_api.interruption_gate as gate_mod  # noqa: E402


def setup_function():
    inc.set_incognito(False)


# ---- vision-intent classifier ------------------------------------------

def test_looks_vision_relevant_catches_common_phrases():
    for q in ("what does this say", "summarize this email",
              "what's on my screen", "explain that one",
              "this tab", "current page"):
        assert looks_vision_relevant(q), q


def test_looks_vision_relevant_rejects_unrelated():
    for q in ("send Dani an email", "what's the weather",
              "set volume to 30", "play poker face"):
        assert not looks_vision_relevant(q), q


def test_looks_vision_relevant_skips_long_requests():
    # Self-contained long requests don't reference the screen.
    huge = ("Send Dani a long email about the Q3 roadmap that "
            "includes the next 90 days of milestones, the team "
            "headcount projections, and a section on risks " * 3)
    assert not looks_vision_relevant(huge)


# ---- context block rendering ------------------------------------------

def test_summary_as_context_block_includes_app_and_text():
    s = ScreenSummary(
        captured_at=time.time(),
        active_app="Chrome",
        active_window_title="Gmail Inbox",
        one_line="Chrome — Gmail Inbox; 24 elements",
        visible_text="From Dani — Q3 plan attached. Subject: Q3 plan.",
        elements_preview=["Compose", "Inbox (24)", "Sent"],
    )
    block = s.as_context_block()
    assert "ON SCREEN:" in block
    assert "Chrome" in block
    assert "Q3 plan" in block
    assert "Compose" in block


def test_summary_as_context_block_respects_max_chars():
    s = ScreenSummary(
        captured_at=time.time(),
        one_line="x" * 5000,
        visible_text="y" * 5000,
        elements_preview=["z" * 200] * 50,
    )
    block = s.as_context_block(max_chars=400)
    assert len(block) <= 400


# ---- tick gating -------------------------------------------------------

class _StubContext:
    def __init__(self, sig, app="Chrome", elements=None,
                 text_blocks=None):
        self.timestamp = time.time()
        self.active_app = app
        self.active_window_title = "Test"
        self.active_process = "test.exe"
        self.elements = elements or []
        self.text_blocks = text_blocks or []
        self.screen_hash = sig

    def summary(self):
        return f"{self.active_app} — {self.active_window_title}"


class _StubElement:
    def __init__(self, text):
        self.text = text


class _StubReader:
    def __init__(self, contexts):
        self._contexts = list(contexts)
        self.calls = 0

    def get_context(self, *, want_text=False, force=False):
        self.calls += 1
        if not self._contexts:
            return None
        return self._contexts.pop(0)


def test_tick_captures_and_caches_summary():
    reader = _StubReader([_StubContext(
        "sig1", elements=[_StubElement("Inbox")])])
    a = ScreenAwareness(reader=reader, interval_sec=0.0)
    a.tick()
    s = a.current_summary()
    assert s is not None
    assert "Chrome" in s.one_line
    assert "Inbox" in s.elements_preview


def test_tick_skipped_when_incognito():
    reader = _StubReader([_StubContext("sig1")])
    a = ScreenAwareness(reader=reader, interval_sec=0.0)
    inc.set_incognito(True)
    try:
        a.tick()
    finally:
        inc.set_incognito(False)
    assert a.current_summary() is None
    assert reader.calls == 0


def test_tick_skipped_when_screen_sharing(monkeypatch):
    # Inject a gate that says screen-sharing.
    from hgr.live_api.interruption_gate import (
        InterruptionGate, SignalKind)
    g = InterruptionGate()
    g.set_signal(SignalKind.SCREEN_SHARING, True)
    monkeypatch.setattr(gate_mod, "_gate", g)
    reader = _StubReader([_StubContext("sig1")])
    a = ScreenAwareness(reader=reader, interval_sec=0.0)
    a.tick()
    assert a.current_summary() is None
    assert reader.calls == 0


def test_tick_throttle_prevents_rapid_repeats():
    """Sentinel could fire ticks faster than the interval; the
    awareness layer should throttle defensively."""
    reader = _StubReader([_StubContext("a"), _StubContext("b")])
    a = ScreenAwareness(reader=reader, interval_sec=100.0)
    a.tick()
    a.tick()  # within interval — should NOT capture again
    assert reader.calls == 1


def test_current_summary_returns_none_after_ttl():
    reader = _StubReader([_StubContext("sig1")])
    a = ScreenAwareness(reader=reader, interval_sec=0.0,
                        context_ttl_sec=0.0)
    a.tick()
    # TTL=0 → next read sees it as expired.
    time.sleep(0.01)
    assert a.current_summary() is None


def test_force_capture_now_bypasses_throttle():
    reader = _StubReader([
        _StubContext("a"), _StubContext("b")])
    a = ScreenAwareness(reader=reader, interval_sec=999.0)
    a.tick()
    a.force_capture_now()
    assert reader.calls == 2


def test_reset_clears_summary():
    reader = _StubReader([_StubContext("sig1")])
    a = ScreenAwareness(reader=reader, interval_sec=0.0)
    a.tick()
    assert a.current_summary() is not None
    a.reset()
    assert a.current_summary() is None

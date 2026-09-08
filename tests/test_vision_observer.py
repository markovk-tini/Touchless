"""Tests for VisionObserver (Phase 6 B3)."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.incognito as inc  # noqa: E402
from hgr.live_api.vision_observer import (  # noqa: E402
    Observation, ObservationKind, VisionObserver,
    _recognize_error_dialog, _recognize_save_dialog,
    _recognize_stack_trace, _recognize_url_opened,
)
from hgr.live_api.interruption_gate import (  # noqa: E402
    GateConfig, InterruptionGate, SignalKind,
)


def setup_function():
    inc.set_incognito(False)


def _open_gate():
    g = InterruptionGate(config=GateConfig(quiet_start_hour=0,
                                            quiet_end_hour=0))
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    g.set_signal(SignalKind.MIC_IN_USE, False)
    g.set_signal(SignalKind.CAMERA_IN_USE, False)
    return g


@dataclass
class _FakeSummary:
    active_window_title: str = ""
    visible_text: str = ""
    screen_hash: str = "hash-1"


@dataclass
class _FakeTurn:
    text: str = ""


@dataclass
class _FakeBuffer:
    turns: List[_FakeTurn] = field(default_factory=list)

    def recent(self, max_turns: int = 6):
        return self.turns[-max_turns:]


class _FakeAwareness:
    def __init__(self, summary):
        self._summary = summary

    def current_summary(self):
        return self._summary


# ---- recognizers ------------------------------------------------------

def test_recognize_error_dialog_matches_title():
    s = _FakeSummary(active_window_title="Crash report - MyApp")
    hit = _recognize_error_dialog(s)
    assert hit is not None
    assert hit["kind"] == ObservationKind.ERROR_DIALOG
    assert "Crash report" in hit["headline"]


def test_recognize_error_dialog_skips_innocuous_title():
    s = _FakeSummary(active_window_title="Notepad - Untitled.txt")
    assert _recognize_error_dialog(s) is None


def test_recognize_error_dialog_handles_blank():
    assert _recognize_error_dialog(_FakeSummary()) is None


def test_recognize_stack_trace_python():
    s = _FakeSummary(visible_text=(
        'Traceback (most recent call last):\n'
        '  File "main.py", line 12, in <module>\n'
        '    raise ValueError("oops")\n'))
    hit = _recognize_stack_trace(s)
    assert hit is not None
    assert hit["kind"] == ObservationKind.STACK_TRACE
    assert "Traceback" in hit["payload"]["snippet"]


def test_recognize_stack_trace_java():
    s = _FakeSummary(visible_text=(
        "Exception in thread \"main\" "
        "java.lang.NullPointerException\n"
        "  at com.example.App.run(App.java:42)\n"))
    hit = _recognize_stack_trace(s)
    assert hit is not None
    assert hit["kind"] == ObservationKind.STACK_TRACE


def test_recognize_stack_trace_skips_plain_text():
    s = _FakeSummary(visible_text=(
        "Hello world, this is normal output."))
    assert _recognize_stack_trace(s) is None


def test_recognize_save_dialog_matches_title():
    s = _FakeSummary(active_window_title="Save As")
    hit = _recognize_save_dialog(s)
    assert hit is not None
    assert hit["kind"] == ObservationKind.SAVE_DIALOG


def test_recognize_save_dialog_open_file():
    s = _FakeSummary(active_window_title="Open File")
    hit = _recognize_save_dialog(s)
    assert hit is not None


def test_recognize_save_dialog_skips_unrelated():
    s = _FakeSummary(active_window_title="Slack - Channel")
    assert _recognize_save_dialog(s) is None


def test_recognize_url_opened_with_session_match():
    buf = _FakeBuffer(turns=[
        _FakeTurn(text="check stripe.com docs"),
    ])
    s = _FakeSummary(active_window_title=(
        "Stripe Docs - stripe.com/docs/api"))
    hit = _recognize_url_opened(s, session_buffer=buf)
    assert hit is not None
    assert hit["payload"]["domain"] == "stripe.com"


def test_recognize_url_opened_skips_when_not_mentioned():
    buf = _FakeBuffer(turns=[_FakeTurn(text="hello")])
    s = _FakeSummary(active_window_title="Some site - example.com")
    assert _recognize_url_opened(s, session_buffer=buf) is None


def test_recognize_url_opened_skips_common_domains():
    buf = _FakeBuffer(turns=[_FakeTurn(text="google.com")])
    s = _FakeSummary(active_window_title=(
        "Google - google.com"))
    assert _recognize_url_opened(s, session_buffer=buf) is None


def test_recognize_url_opened_handles_no_session_buffer():
    s = _FakeSummary(active_window_title="x - foo.com")
    assert _recognize_url_opened(s, session_buffer=None) is None


# ---- observer integration --------------------------------------------

def test_observer_emits_error_dialog_observation():
    summary = _FakeSummary(
        active_window_title="Application Error",
        screen_hash="h1")
    fired = []
    obs = VisionObserver(
        handler=lambda o: fired.append(o),
        awareness=_FakeAwareness(summary),
        gate=_open_gate())
    obs.tick()
    assert len(fired) == 1
    assert fired[0].kind == ObservationKind.ERROR_DIALOG


def test_observer_cooldown_blocks_repeat():
    summary = _FakeSummary(
        active_window_title="Crash dialog", screen_hash="h1")
    fired = []
    obs = VisionObserver(
        handler=lambda o: fired.append(o),
        awareness=_FakeAwareness(summary),
        gate=_open_gate())
    obs.tick()
    # Force the screen hash to change so the tick is processed
    # again — cooldown should still suppress.
    summary.screen_hash = "h2"
    obs.tick()
    assert len(fired) == 1


def test_observer_skips_unchanged_screen():
    summary = _FakeSummary(
        active_window_title="Some Error", screen_hash="h1")
    fired = []
    obs = VisionObserver(
        handler=lambda o: fired.append(o),
        awareness=_FakeAwareness(summary),
        gate=_open_gate())
    obs.tick()
    obs.tick()
    # Second tick has same hash → recognizers never run again,
    # so still exactly one fire.
    assert len(fired) == 1


def test_observer_honors_incognito():
    summary = _FakeSummary(
        active_window_title="Crash dialog", screen_hash="h1")
    fired = []
    obs = VisionObserver(
        handler=lambda o: fired.append(o),
        awareness=_FakeAwareness(summary),
        gate=_open_gate())
    inc.set_incognito(True)
    try:
        obs.tick()
    finally:
        inc.set_incognito(False)
    assert fired == []


def test_observer_blocked_when_gate_closed():
    summary = _FakeSummary(
        active_window_title="Crash dialog", screen_hash="h1")
    gate = _open_gate()
    gate.set_signal(SignalKind.SCREEN_SHARING, True)
    fired = []
    obs = VisionObserver(
        handler=lambda o: fired.append(o),
        awareness=_FakeAwareness(summary),
        gate=gate)
    obs.tick()
    assert fired == []


def test_observer_handles_none_summary():
    class _NoneAwareness:
        def current_summary(self):
            return None
    fired = []
    obs = VisionObserver(
        handler=lambda o: fired.append(o),
        awareness=_NoneAwareness(),
        gate=_open_gate())
    obs.tick()
    assert fired == []


def test_observer_handler_exception_does_not_break_tick():
    summary = _FakeSummary(
        active_window_title="Error popup", screen_hash="h1")
    def bad_handler(o):
        raise RuntimeError("boom")
    obs = VisionObserver(
        handler=bad_handler,
        awareness=_FakeAwareness(summary),
        gate=_open_gate())
    # Should not raise.
    obs.tick()


def test_observer_reset_clears_state():
    summary = _FakeSummary(
        active_window_title="Crash", screen_hash="h1")
    fired = []
    obs = VisionObserver(
        handler=lambda o: fired.append(o),
        awareness=_FakeAwareness(summary),
        gate=_open_gate())
    obs.tick()
    obs.reset()
    # After reset, the screen-hash cache is empty, so the same
    # summary registers as "new" — but cooldown still applies.
    summary.screen_hash = "h2"
    # Manually clear cooldown to confirm reset wiped it.
    assert obs._last_fire == {}


def test_observer_url_opened_uses_session_buffer():
    summary = _FakeSummary(
        active_window_title=(
            "Stripe Docs - stripe.com/docs"),
        screen_hash="h1")
    buf = _FakeBuffer(turns=[
        _FakeTurn(text="look up stripe.com"),
    ])
    fired = []
    obs = VisionObserver(
        handler=lambda o: fired.append(o),
        awareness=_FakeAwareness(summary),
        session_buffer=buf,
        gate=_open_gate())
    obs.tick()
    url_obs = [o for o in fired
               if o.kind == ObservationKind.URL_OPENED]
    assert len(url_obs) == 1


def test_observer_emits_distinct_kinds_in_one_tick():
    """Stack trace + error dialog can co-occur."""
    summary = _FakeSummary(
        active_window_title="Application Error",
        visible_text=(
            'Traceback (most recent call last):\n'
            '  File "x.py", line 1, in <module>\n'),
        screen_hash="h1")
    fired = []
    obs = VisionObserver(
        handler=lambda o: fired.append(o),
        awareness=_FakeAwareness(summary),
        gate=_open_gate())
    obs.tick()
    kinds = {o.kind for o in fired}
    assert ObservationKind.ERROR_DIALOG in kinds
    assert ObservationKind.STACK_TRACE in kinds


def test_observation_dataclass_defaults():
    o = Observation(kind=ObservationKind.ERROR_DIALOG, headline="x")
    assert o.severity == "low"
    assert o.payload == {}
    assert o.suggested_action == ""
    assert o.ts > 0

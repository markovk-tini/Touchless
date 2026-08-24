"""Tests for callback_engine (Phase 7 B2)."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.incognito as inc  # noqa: E402
import hgr.live_api.persona_voice as pv  # noqa: E402
from hgr.live_api.callback_engine import (  # noqa: E402
    CallbackHint, CallbackKind, maybe_callback,
    render_as_prompt_block, reset_state,
)


def setup_function():
    inc.set_incognito(False)
    pv.reset_for_tests()
    reset_state()
    # Phase-7: callback_engine consults affect.should_suppress_callbacks
    # which reads the global affect model. Reset between tests so a
    # prior test's frustration signals don't suppress callbacks here.
    try:
        from hgr.live_api.affect import reset_global as _aff_reset
        _aff_reset()
    except Exception:
        pass


@dataclass
class _Turn:
    role: str = "user"
    text: str = ""


@dataclass
class _Buffer:
    turns: List[_Turn] = field(default_factory=list)

    def recent(self, max_turns: int = 6):
        return self.turns[-max_turns:]


@dataclass
class _Profile:
    short: str = "myproj"
    recent_files: List[str] = field(default_factory=list)
    recent_activity: List[dict] = field(default_factory=list)
    last_seen_at: float = 0.0

    def short_label(self) -> str:
        return self.short


class _Entity:
    def __init__(self, attrs):
        self.attributes = attrs


class _Graph:
    def __init__(self, aliases):
        self._a = aliases

    def find_by_alias(self, name):
        return self._a.get(name)


def _warm_state():
    """Bypass cooldown by giving the engine a fresh state with
    turns_since_last_callback=999 (which is the default)."""
    reset_state()


# ---- detectors --------------------------------------------------------

def test_past_failure_fires_on_round_two():
    _warm_state()
    buf = _Buffer(turns=[
        _Turn(text="the test is failing"),
        _Turn(text="let me try running it again"),
    ])
    hint = maybe_callback(
        user_text="run the tests again",
        session_buffer=buf)
    assert hint is not None
    assert hint.kind == CallbackKind.PAST_FAILURE


def test_past_failure_skipped_without_failure_history():
    _warm_state()
    buf = _Buffer(turns=[_Turn(text="hello iris")])
    hint = maybe_callback(
        user_text="run the tests", session_buffer=buf)
    # No history of failure → no past-failure callback. Could still
    # fire prior_topic since "tests" matches, but in this short
    # convo it won't have 2 turns to scan.
    assert (hint is None
            or hint.kind != CallbackKind.PAST_FAILURE)


def test_callback_name_fires_when_entity_known():
    _warm_state()
    graph = _Graph(aliases={
        "Dani": _Entity({"email": "dani@x.com"}),
    })
    hint = maybe_callback(
        user_text="email Dani about Q3",
        entity_graph=graph)
    assert hint is not None
    assert hint.kind == CallbackKind.CALLBACK_NAME
    assert "dani" in hint.hook.lower()


def test_callback_name_skips_unknown_entity():
    _warm_state()
    graph = _Graph(aliases={})
    hint = maybe_callback(
        user_text="email Frodo about Q3",
        entity_graph=graph)
    assert hint is None or hint.kind != CallbackKind.CALLBACK_NAME


def test_callback_name_skips_common_words():
    _warm_state()
    # "Iris" should not trigger — it's the assistant.
    graph = _Graph(aliases={
        "Iris": _Entity({"role": "self"})
    })
    hint = maybe_callback(
        user_text="Iris what's up", entity_graph=graph)
    assert hint is None


def test_project_activity_fires_when_recent():
    _warm_state()
    prof = _Profile(
        short="touchless",
        recent_files=["main.py", "ui.py"],
        last_seen_at=time.time() - 600)  # 10 min ago
    hint = maybe_callback(
        user_text="what's the build status",
        project_profile=prof)
    assert hint is not None
    assert hint.kind == CallbackKind.PROJECT_ACTIVITY
    assert "main.py" in hint.hook


def test_project_activity_skips_stale():
    _warm_state()
    prof = _Profile(
        short="touchless",
        recent_files=["main.py"],
        last_seen_at=time.time() - 7200)  # 2h ago, too stale
    hint = maybe_callback(
        user_text="status", project_profile=prof)
    assert (hint is None
            or hint.kind != CallbackKind.PROJECT_ACTIVITY)


def test_repeated_app_fires():
    _warm_state()
    prof = _Profile(
        short="VSCode",
        recent_files=[],
        recent_activity=[
            {"kind": "focus", "ts": time.time() - 3000},
            {"kind": "focus", "ts": time.time() - 6000},
        ],
        last_seen_at=time.time() - 7200)
    hint = maybe_callback(
        user_text="what's going on",
        project_profile=prof)
    assert hint is not None


def test_prior_topic_fires_on_shared_keywords():
    _warm_state()
    buf = _Buffer(turns=[
        _Turn(text="I was looking at the database migration"),
        _Turn(text="okay moving on"),
        _Turn(text="forget it"),
    ])
    hint = maybe_callback(
        user_text="actually, the migration database thing",
        session_buffer=buf)
    assert hint is not None
    assert hint.kind == CallbackKind.PRIOR_TOPIC


def test_prior_topic_skips_no_overlap():
    _warm_state()
    buf = _Buffer(turns=[
        _Turn(text="hello there"),
        _Turn(text="how's the weather"),
    ])
    hint = maybe_callback(
        user_text="run the tests",
        session_buffer=buf)
    assert (hint is None
            or hint.kind != CallbackKind.PRIOR_TOPIC)


# ---- gating ----------------------------------------------------------

def test_cooldown_blocks_back_to_back():
    _warm_state()
    graph = _Graph(aliases={
        "Dani": _Entity({"email": "dani@x.com"}),
    })
    h1 = maybe_callback(
        user_text="email Dani", entity_graph=graph)
    h2 = maybe_callback(
        user_text="email Dani again", entity_graph=graph)
    assert h1 is not None
    assert h2 is None


def test_cooldown_clears_after_four_turns():
    _warm_state()
    graph = _Graph(aliases={
        "Dani": _Entity({"email": "dani@x.com"}),
    })
    h1 = maybe_callback(
        user_text="email Dani", entity_graph=graph)
    assert h1 is not None
    # Four blocked turns clear the cooldown.
    for _ in range(4):
        maybe_callback(user_text="hi", session_buffer=_Buffer())
    h2 = maybe_callback(
        user_text="email Dani again", entity_graph=graph)
    assert h2 is not None


def test_concise_preset_blocks_callbacks():
    _warm_state()
    pv.set_active("concise")
    graph = _Graph(aliases={
        "Dani": _Entity({"email": "dani@x.com"}),
    })
    hint = maybe_callback(
        user_text="email Dani", entity_graph=graph)
    assert hint is None


def test_incognito_blocks_callbacks():
    _warm_state()
    graph = _Graph(aliases={
        "Dani": _Entity({"email": "dani@x.com"})})
    inc.set_incognito(True)
    try:
        hint = maybe_callback(
            user_text="email Dani", entity_graph=graph)
    finally:
        inc.set_incognito(False)
    assert hint is None


def test_jarvis_preset_higher_likelihood():
    _warm_state()
    pv.set_active("jarvis")
    prof = _Profile(
        short="touchless",
        recent_files=["main.py"],
        last_seen_at=time.time() - 600)
    hint = maybe_callback(
        user_text="what's going on", project_profile=prof)
    assert hint is not None  # 0.7 confidence × 0.85 gain = 0.595 > 0.5


# ---- rendering -------------------------------------------------------

def test_render_as_prompt_block_includes_hook():
    h = CallbackHint(
        kind=CallbackKind.PRIOR_TOPIC,
        hook="coming back to auth", confidence=0.8)
    block = render_as_prompt_block(h)
    assert "auth" in block
    assert "CALLBACK" in block


def test_render_empty_for_none():
    assert render_as_prompt_block(None) == ""


# ---- dataclass ------------------------------------------------------

def test_callback_hint_defaults():
    h = CallbackHint(
        kind=CallbackKind.PRIOR_TOPIC, hook="x")
    assert h.confidence == 0.0
    assert h.payload == {}
    assert h.ts > 0


# ---- exception tolerance --------------------------------------------

def test_buggy_session_buffer_does_not_break():
    _warm_state()
    class Boom:
        def recent(self, max_turns):
            raise RuntimeError("nope")
    # Should silently return None.
    hint = maybe_callback(
        user_text="hi", session_buffer=Boom())
    assert hint is None or isinstance(hint, CallbackHint)


def test_buggy_entity_graph_does_not_break():
    _warm_state()
    class Boom:
        def find_by_alias(self, n):
            raise RuntimeError("nope")
    hint = maybe_callback(
        user_text="email Dani", entity_graph=Boom())
    # Detector silently fails, others may still produce hints.
    assert hint is None or isinstance(hint, CallbackHint)

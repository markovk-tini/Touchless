"""Tests for tool speculation (Phase 8 B4)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.incognito as inc  # noqa: E402
from hgr.live_api.tool_speculation import (  # noqa: E402
    ToolSpeculator, _CacheEntry, _SpeculationCache, _Trigger,
    maybe_speculate, reset_global,
)
import re


def setup_function():
    inc.set_incognito(False)
    reset_global()


def _wait(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ---- cache -----------------------------------------------------------

def test_cache_put_and_get():
    c = _SpeculationCache()
    c.put("clock_now", {}, {"hour": 14})
    assert c.get("clock_now", {}) == {"hour": 14}


def test_cache_get_missing():
    c = _SpeculationCache()
    assert c.get("clock_now", {}) is None


def test_cache_expires():
    c = _SpeculationCache()
    c.put("clock_now", {}, {"hour": 14})
    # Mutate the cached_at to past TTL.
    key = c._key("clock_now", {})
    c._entries[key].cached_at = time.time() - 300
    assert c.get("clock_now", {}) is None


def test_cache_cooldown():
    c = _SpeculationCache()
    assert c.on_cooldown("clock_now") is False
    c.mark_fired("clock_now")
    assert c.on_cooldown("clock_now") is True


def test_cache_args_disambiguate():
    c = _SpeculationCache()
    c.put("gmail_list", {"max": 5}, [1, 2, 3])
    c.put("gmail_list", {"max": 10}, [1, 2, 3, 4, 5])
    assert c.get("gmail_list", {"max": 5}) == [1, 2, 3]
    assert (c.get("gmail_list", {"max": 10})
            == [1, 2, 3, 4, 5])


def test_cache_clear():
    c = _SpeculationCache()
    c.put("clock_now", {}, {"hour": 14})
    c.clear()
    assert c.get("clock_now", {}) is None


def test_cache_turn_limit():
    c = _SpeculationCache()
    c.begin_turn()
    assert c.can_speculate_more() is True
    c.increment_turn()
    c.increment_turn()
    assert c.can_speculate_more() is False


# ---- speculator dispatch --------------------------------------------

def test_maybe_speculate_fires_on_keyword():
    called = []
    sp = ToolSpeculator(
        dispatcher=lambda t, a: called.append((t, a))
                                or {"ok": True})
    sp.begin_turn()
    fired = sp.maybe_speculate("what's the weather today")
    assert any(t == "weather_get" for t, _ in fired)
    _wait(lambda: any(t == "weather_get"
                       for t, _ in called))
    assert any(t == "weather_get" for t, _ in called)


def test_maybe_speculate_skips_no_keyword():
    called = []
    sp = ToolSpeculator(
        dispatcher=lambda t, a: called.append((t, a))
                                or {"ok": True})
    sp.begin_turn()
    fired = sp.maybe_speculate("hello there friend")
    assert fired == []


def test_maybe_speculate_honors_incognito():
    called = []
    sp = ToolSpeculator(
        dispatcher=lambda t, a: called.append((t, a))
                                or {"ok": True})
    sp.begin_turn()
    inc.set_incognito(True)
    try:
        fired = sp.maybe_speculate("what's the weather")
    finally:
        inc.set_incognito(False)
    assert fired == []


def test_maybe_speculate_respects_cooldown():
    called = []
    sp = ToolSpeculator(
        dispatcher=lambda t, a: called.append((t, a))
                                or {"ok": True})
    sp.begin_turn()
    sp.maybe_speculate("what's the weather")
    _wait(lambda: len(called) >= 1)
    sp.begin_turn()
    fired2 = sp.maybe_speculate("what's the weather today")
    # On cooldown — second call is a no-op for weather_get.
    assert all(t != "weather_get" for t, _ in fired2)


def test_maybe_speculate_caps_per_turn():
    called = []
    sp = ToolSpeculator(
        dispatcher=lambda t, a: called.append((t, a))
                                or {"ok": True})
    sp.begin_turn()
    fired = sp.maybe_speculate(
        "what's the time and the weather "
        "and my calendar today")
    # Capped at MAX_SPECULATIONS_PER_TURN (2).
    assert len(fired) <= 2


def test_maybe_speculate_no_dispatcher_no_fire():
    sp = ToolSpeculator(dispatcher=None)
    sp.begin_turn()
    fired = sp.maybe_speculate("what's the weather")
    assert fired == []


def test_get_cached_returns_result():
    called = []
    sp = ToolSpeculator(
        dispatcher=lambda t, a: (called.append((t, a)),
                                  {"temp_f": 67, "ok": True})[1])
    sp.begin_turn()
    sp.maybe_speculate("what's the weather")
    _wait(lambda: sp.get_cached("weather_get", {})
                  is not None)
    cached = sp.get_cached("weather_get", {})
    assert cached is not None
    assert cached.get("temp_f") == 67


def test_get_cached_returns_none_when_no_speculation():
    sp = ToolSpeculator(dispatcher=lambda t, a: {})
    assert sp.get_cached("clock_now", {}) is None


def test_dispatcher_exception_does_not_crash():
    def bad(t, a):
        raise RuntimeError("nope")
    sp = ToolSpeculator(dispatcher=bad)
    sp.begin_turn()
    # Must not raise.
    sp.maybe_speculate("what's the weather")
    # Cache stays empty.
    assert sp.get_cached("clock_now", {}) is None


def test_dispatcher_returning_none_does_not_cache():
    sp = ToolSpeculator(dispatcher=lambda t, a: None)
    sp.begin_turn()
    sp.maybe_speculate("what's the weather")
    time.sleep(0.1)
    assert sp.get_cached("clock_now", {}) is None


# ---- different triggers ---------------------------------------------

def test_weather_trigger():
    called = []
    sp = ToolSpeculator(
        dispatcher=lambda t, a: (called.append(t),
                                  {"temp_f": 67})[1])
    sp.begin_turn()
    sp.maybe_speculate("what's the weather today")
    _wait(lambda: "weather_get" in called)
    assert "weather_get" in called


def test_calendar_trigger():
    called = []
    sp = ToolSpeculator(
        dispatcher=lambda t, a: (called.append(t),
                                  {"events": []})[1])
    sp.begin_turn()
    sp.maybe_speculate("what meetings do I have")
    _wait(lambda: "calendar_list_events" in called)
    assert "calendar_list_events" in called


def test_email_trigger_uses_default_args():
    called = []
    sp = ToolSpeculator(
        dispatcher=lambda t, a: (called.append((t, a)),
                                  {"messages": []})[1])
    sp.begin_turn()
    sp.maybe_speculate("any new emails")
    _wait(lambda: any(t == "gmail_list" for t, _ in called))
    matching = [a for t, a in called if t == "gmail_list"]
    assert matching
    assert matching[0].get("unread_only") is True


def test_custom_triggers():
    """Caller can inject their own trigger list."""
    custom = [
        _Trigger(
            pattern=re.compile(r"\bnews\b"),
            tool="news_get",
            args={"limit": 5}),
    ]
    called = []
    sp = ToolSpeculator(
        dispatcher=lambda t, a: (called.append((t, a)),
                                  {"items": []})[1],
        triggers=custom)
    sp.begin_turn()
    sp.maybe_speculate("show me the latest news")
    _wait(lambda: any(t == "news_get" for t, _ in called))
    assert any(t == "news_get" for t, _ in called)


# ---- clear / reset --------------------------------------------------

def test_clear_drops_cache_and_cooldown():
    sp = ToolSpeculator(dispatcher=lambda t, a: {"ok": True})
    sp.begin_turn()
    sp.maybe_speculate("what's the weather")
    _wait(lambda: sp.get_cached("weather_get", {})
                  is not None)
    sp.clear()
    assert sp.get_cached("weather_get", {}) is None

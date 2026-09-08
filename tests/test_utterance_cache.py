"""Tests for UtteranceCache + prompt cache helpers (Phase 2 B2)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.utterance_cache import (  # noqa: E402
    UtteranceCache, build_anthropic_request_kwargs,
    make_anthropic_messages_with_cache, system_prompt_cache_hash,
)


# ---- UtteranceCache ----------------------------------------------------

def test_get_returns_none_for_missing_key():
    c = UtteranceCache()
    assert c.get("never asked") is None
    assert c.stats()["misses"] == 1


def test_put_then_get_round_trips():
    c = UtteranceCache()
    c.put("what time is it", "It's 3pm.", payload={"hour": 15})
    hit = c.get("what time is it")
    assert hit is not None
    assert hit.message == "It's 3pm."
    assert hit.payload["hour"] == 15
    assert c.stats()["hits"] == 1


def test_normalization_collapses_whitespace_and_case():
    c = UtteranceCache()
    c.put("Hey  Iris ,  what's UP?", "All good.")
    # All-caps + trailing punct + whitespace variants all hit.
    assert c.get("hey iris, what's up") is not None
    assert c.get("HEY IRIS, WHAT'S UP!") is not None
    assert c.get("hey  iris,  what's up?") is not None


def test_put_cacheable_false_does_not_store():
    c = UtteranceCache()
    assert c.put("weather in Berlin", "Sunny.", cacheable=False) is False
    assert c.get("weather in Berlin") is None


def test_lru_eviction_drops_oldest():
    c = UtteranceCache(max_entries=3)
    c.put("a", "1")
    c.put("b", "2")
    c.put("c", "3")
    c.put("d", "4")  # evicts 'a'
    assert c.get("a") is None
    assert c.get("b") is not None
    assert c.get("c") is not None
    assert c.get("d") is not None


def test_get_promotes_entry_to_most_recent():
    c = UtteranceCache(max_entries=3)
    c.put("a", "1")
    c.put("b", "2")
    c.put("c", "3")
    c.get("a")            # promote a to MRU
    c.put("d", "4")       # evicts the next-oldest, which is now b
    assert c.get("a") is not None
    assert c.get("b") is None


def test_ttl_expires_old_entries():
    c = UtteranceCache(ttl_seconds=0)  # immediate expiry
    c.put("x", "y")
    time.sleep(0.01)
    assert c.get("x") is None


def test_clear_resets_state():
    c = UtteranceCache()
    c.put("a", "1")
    c.put("b", "2")
    n = c.clear()
    assert n == 2
    assert c.get("a") is None
    assert c.stats() == {"size": 0, "hits": 0, "misses": 1}


def test_empty_text_ignored():
    c = UtteranceCache()
    assert c.put("", "x") is False
    assert c.get("") is None
    assert c.put("   ", "x") is False


# ---- prompt cache helpers ----------------------------------------------

def test_anthropic_kwargs_has_system_at_top_level():
    # F-001 audit: return shape is now a dict matching SDK kwargs.
    # `system` is top-level (NOT a role inside messages).
    kwargs = build_anthropic_request_kwargs(
        system_prompt="You are Iris.",
        tool_catalog_text="TOOLS: weather_get, gmail_send",
        per_turn_user_text="what's the weather",
    )
    assert "system" in kwargs
    assert isinstance(kwargs["system"], list)
    assert len(kwargs["system"]) == 2
    assert all(b["cache_control"]["type"] == "ephemeral"
               for b in kwargs["system"])
    assert kwargs["messages"][0]["role"] == "user"
    assert "what's the weather" in kwargs["messages"][0]["content"]
    # CRITICAL: no message with role=system inside messages.
    assert not any(m.get("role") == "system"
                   for m in kwargs["messages"])


def test_anthropic_kwargs_includes_memory_context_in_user():
    kwargs = build_anthropic_request_kwargs(
        system_prompt="You are Iris.",
        tool_catalog_text="TOOLS",
        per_turn_user_text="email Dani",
        memory_context="USER PREFS:\n- send via gmail",
    )
    user = kwargs["messages"][0]["content"]
    assert "USER PREFS" in user
    assert "email Dani" in user


def test_anthropic_kwargs_omits_system_when_empty():
    kwargs = build_anthropic_request_kwargs(
        system_prompt="",
        tool_catalog_text="",
        per_turn_user_text="hi",
    )
    assert "system" not in kwargs
    assert kwargs["messages"][0]["role"] == "user"


def test_legacy_make_anthropic_messages_with_cache_returns_dict():
    # Back-compat alias still works but returns the corrected dict shape.
    result = make_anthropic_messages_with_cache(
        system_prompt="Sys",
        tool_catalog_text="Tools",
        per_turn_user_text="hi",
    )
    assert isinstance(result, dict)
    assert "messages" in result
    assert "system" in result


def test_cache_hash_stable_across_calls():
    h1 = system_prompt_cache_hash(system_prompt="A", tool_catalog_text="B")
    h2 = system_prompt_cache_hash(system_prompt="A", tool_catalog_text="B")
    assert h1 == h2
    assert len(h1) == 12


def test_cache_hash_changes_with_input():
    h1 = system_prompt_cache_hash(system_prompt="A", tool_catalog_text="B")
    h2 = system_prompt_cache_hash(system_prompt="A2", tool_catalog_text="B")
    h3 = system_prompt_cache_hash(system_prompt="A", tool_catalog_text="B2")
    assert h1 != h2 != h3

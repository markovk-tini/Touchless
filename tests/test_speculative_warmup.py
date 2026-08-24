"""Tests for SpeculativeWarmup (Phase 6 B4)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.speculative_warmup import (  # noqa: E402
    SpeculativeWarmup, WarmupStats,
)


def _wait_for_thread(stats, key, expected, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if getattr(stats, key) >= expected:
            return True
        time.sleep(0.02)
    return False


def test_warmup_fires_when_provider_known():
    stats = WarmupStats()
    fired_flag = {"hit": False}

    def fake_ping():
        fired_flag["hit"] = True
        return True

    w = SpeculativeWarmup(
        stats=stats,
        ping_table={"openai": fake_ping})
    assert w.notify_speech_start(lambda: "openai") is True
    assert _wait_for_thread(stats, "fired", 1)
    assert fired_flag["hit"]


def test_warmup_cooldown_blocks_back_to_back():
    stats = WarmupStats()
    w = SpeculativeWarmup(
        stats=stats,
        ping_table={"openai": lambda: True},
        cooldown_sec=30.0)
    assert w.notify_speech_start(lambda: "openai") is True
    # Second call within cooldown is skipped.
    assert w.notify_speech_start(lambda: "openai") is False
    assert stats.skipped_cooldown >= 1


def test_warmup_skips_when_provider_unknown_and_no_env():
    stats = WarmupStats()
    w = SpeculativeWarmup(stats=stats,
                          ping_table={"openai": lambda: True})
    # No resolver target + no env vars → skipped.
    with patch.dict(os.environ, {}, clear=False):
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            os.environ.pop(k, None)
        assert w.notify_speech_start(lambda: None) is False


def test_warmup_falls_back_to_openai_env():
    stats = WarmupStats()
    w = SpeculativeWarmup(stats=stats,
                          ping_table={"openai": lambda: True})
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        assert w.notify_speech_start(lambda: None) is True
    assert _wait_for_thread(stats, "fired", 1)


def test_warmup_falls_back_to_anthropic_env():
    stats = WarmupStats()
    w = SpeculativeWarmup(stats=stats,
                          ping_table={"anthropic": lambda: True})
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant"}):
        # Make sure openai key isn't also set.
        os.environ.pop("OPENAI_API_KEY", None)
        assert w.notify_speech_start(lambda: None) is True
    assert _wait_for_thread(stats, "fired", 1)


def test_warmup_failed_ping_increments_failed():
    stats = WarmupStats()
    w = SpeculativeWarmup(
        stats=stats,
        ping_table={"openai": lambda: False})
    w.notify_speech_start(lambda: "openai")
    assert _wait_for_thread(stats, "failed", 1)


def test_warmup_exception_ping_increments_failed():
    stats = WarmupStats()
    def boom():
        raise RuntimeError("nope")
    w = SpeculativeWarmup(stats=stats,
                          ping_table={"openai": boom})
    w.notify_speech_start(lambda: "openai")
    assert _wait_for_thread(stats, "failed", 1)


def test_warmup_reset_clears_cooldown():
    stats = WarmupStats()
    w = SpeculativeWarmup(
        stats=stats,
        ping_table={"openai": lambda: True},
        cooldown_sec=30.0)
    w.notify_speech_start(lambda: "openai")
    w.reset()
    # After reset, the cooldown is cleared so a second call fires.
    assert w.notify_speech_start(lambda: "openai") is True


def test_warmup_stats_track_requested():
    stats = WarmupStats()
    w = SpeculativeWarmup(stats=stats,
                          ping_table={"openai": lambda: True})
    w.notify_speech_start(lambda: "openai")
    w.notify_speech_start(lambda: "openai")
    assert stats.requested == 2


def test_warmup_target_recorded():
    stats = WarmupStats()
    w = SpeculativeWarmup(stats=stats,
                          ping_table={"openai": lambda: True})
    w.notify_speech_start(lambda: "openai")
    assert stats.last_target == "openai"

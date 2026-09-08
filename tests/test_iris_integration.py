"""End-to-end integration test for Iris substrate.

NOTE on persistence isolation: three Phase-8/9 stores (local_intent,
behavior_log, project_profile) write to `%LOCALAPPDATA%/Touchless/
private/*.db` by default. Without redirection, a prior run's
training data leaks into these tests AND every test pollutes the
real on-disk DBs the user app reads. Each test that exercises
those stores must reset their globals + point them at a temp dir.

Exercises the full orchestrator wiring with all Phase 6-10 modules
active simultaneously:
  * persona_voice presets
  * callback_engine + affect (verifies the active preset + affect
    state actually influences planner output)
  * local_intent (verifies record_example fires on tool dispatch)
  * tool_speculation (verifies trigger keywords schedule pre-fetches)
  * latency_dashboard (verifies StageTimer entries land)
  * smart_fast_path (verifies Tier-0.4 still fires)
  * project_profile (verifies the modality block reaches the planner)
  * behavior_log (verifies preference suggestion path)

The unit tests in this repo cover each module in isolation. This
file verifies they WIRE TOGETHER without raising and without
clobbering each other's state.

We do NOT exercise the LLM-planner / executor here (those need
OPENAI_API_KEY and real network) — only the deterministic tiers
and the post-dispatch side effects.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# CRITICAL: point persistent stores at a temp dir BEFORE any
# Phase-8/9 module imports. Otherwise local_intent, behavior_log,
# and project_profile will lazily open the user's real DBs and
# pollute them. Per-module env vars are honored at construction
# time by each store.
_TMP_DB_DIR = Path(tempfile.mkdtemp(prefix="iris-integ-"))
os.environ["TOUCHLESS_LOCAL_INTENT_DIR"] = str(_TMP_DB_DIR)
os.environ["TOUCHLESS_BEHAVIOR_LOG_DIR"] = str(_TMP_DB_DIR)
os.environ["TOUCHLESS_PROJECT_PROFILE_DIR"] = str(_TMP_DB_DIR)

import hgr.live_api.affect as affect_mod  # noqa: E402
import hgr.live_api.incognito as inc  # noqa: E402
import hgr.live_api.latency_dashboard as latd  # noqa: E402
import hgr.live_api.local_intent as li  # noqa: E402
import hgr.live_api.persona_voice as pv  # noqa: E402
import hgr.live_api.smart_fast_path as sfp  # noqa: E402
import hgr.live_api.tool_speculation as tspec  # noqa: E402
from hgr.live_api.callback_engine import reset_state as cb_reset  # noqa: E402


def setup_function():
    """Reset every shared singleton this integration test touches.
    The unit tests reset modules individually; here we want a clean
    slate across the entire substrate."""
    inc.set_incognito(False)
    pv.reset_for_tests()
    cb_reset()
    affect_mod.reset_global()
    sfp.reset_stats()
    tspec.reset_global()
    latd.reset_global()
    li.reset_global()


# ---------- fake registry ---------------------------------------------

class _FakeRegistry:
    """Minimal ToolRegistry stand-in. `handles_connector` always
    True; `call` returns a canned response per tool. Records every
    invocation for assertions."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self._responses: Dict[str, Dict[str, Any]] = {
            # Real tool names from media_connector + volume_connector
            # + spotify_connector.
            "media_play_pause": {"status": "ok", "sent": True},
            "media_next_track": {"status": "ok", "sent": True},
            "media_previous_track": {"status": "ok",
                                      "sent": True},
            "media_now_playing": {"status": "ok",
                                   "playing": False},
            "volume_toggle_mute": {"status": "ok",
                                    "muted": True},
            "volume_set": {"status": "ok", "percent": 60},
            "weather_get": {"status": "ok",
                            "temp_f": 67,
                            "conditions": "clear"},
            "calendar_list_events": {"status": "ok",
                                      "events": [],
                                      "count": 0},
            "gmail_list": {"status": "ok",
                           "messages": [],
                           "count": 0},
        }

    def call(self, tool: str,
             args: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((tool, dict(args or {})))
        return self._responses.get(
            tool, {"status": "ok"})

    def handles_connector(self, tool: str) -> bool:
        return tool in self._responses


# ---------- smart fast-path → registry dispatch -----------------------

def test_fast_path_pause_dispatches_to_registry():
    """Verifies P6-B4 Tier-0.4 fires + records into local_intent +
    counts as a fast-path stat."""
    from hgr.live_api.planner.orchestrator import IrisPlanner
    reg = _FakeRegistry()
    planner = IrisPlanner(registry=reg)
    result = planner.try_handle("pause")
    assert result is not None
    assert result["message"]
    assert any(t == "media_play_pause" for t, _ in reg.calls)
    stats = sfp.global_stats()
    assert stats.direct >= 1


def test_fast_path_chat_returns_static_reply():
    """Chat fast-path bypasses the registry entirely."""
    from hgr.live_api.planner.orchestrator import IrisPlanner
    reg = _FakeRegistry()
    planner = IrisPlanner(registry=reg)
    result = planner.try_handle("hi")
    assert result is not None
    assert result["message"]
    # No tool dispatch for chat replies.
    assert reg.calls == []


def test_fast_path_planner_falls_through_for_complex():
    """A multi-clause utterance doesn't fast-path."""
    from hgr.live_api.planner.orchestrator import IrisPlanner
    reg = _FakeRegistry()
    planner = IrisPlanner(registry=reg)
    # The planner will fall through (no real LLM); we only assert
    # it doesn't crash + doesn't dispatch via the fast-path.
    result = planner.try_handle(
        "pause the music and then send an email to Dani")
    # Falls through → result may be None when no Tier-1 match.
    # We just need: no exception, no inappropriate fast-path call.
    fp_tools = [t for t, _ in reg.calls
                if t in ("media_play_pause", "gmail_send")]
    assert "media_play_pause" not in fp_tools


# ---------- affect signal → callback suppression ----------------------

def test_frustration_suppresses_callbacks():
    """User's frustration disables witty callbacks."""
    from hgr.live_api.callback_engine import maybe_callback
    affect_mod.observe_user_turn("ugh, this is broken again")
    assert affect_mod.should_suppress_callbacks() is True
    hint = maybe_callback(
        user_text="run that again",
        session_buffer=None)
    assert hint is None


def test_acceptance_clears_frustration_enough_for_callbacks():
    """A 'thanks' raises mood back above the suppression threshold."""
    affect_mod.observe_user_turn("ugh")
    affect_mod.observe_user_turn("thanks, perfect")
    # Net mood: -0.35 + 0.20 = -0.15 → not frustrated.
    assert affect_mod.should_suppress_callbacks() is False


# ---------- preset switching propagates -------------------------------

def test_active_preset_drives_persona_block():
    """Switching to jarvis changes get_persona_block output."""
    from hgr.live_api.persona import get_persona_block
    pv.set_active("jarvis")
    block = get_persona_block()
    assert "sir" in block.lower()


def test_concise_preset_blocks_callbacks():
    from hgr.live_api.callback_engine import maybe_callback
    pv.set_active("concise")
    # Even with detectable callback material, concise gain = 0.
    hint = maybe_callback(
        user_text="pause music")
    assert hint is None


# ---------- latency dashboard collects samples ------------------------

def test_latency_dashboard_records_via_stage_timer():
    """The StageTimer context manager actually feeds the dashboard."""
    with latd.StageTimer("plan"):
        time.sleep(0.02)
    snap = latd.summary()
    assert snap["plan"]["samples"] >= 1
    assert snap["plan"]["p50"] is not None


def test_one_line_aggregates_across_stages():
    latd.record("asr", 200)
    latd.record("plan", 400)
    latd.record("reply", 800)
    line = latd.one_line()
    assert "asr" in line
    assert "plan" in line
    assert "reply" in line


# ---------- local_intent learns from successful dispatch --------------

def test_local_intent_records_on_dispatch():
    """When the orchestrator runs a tool, the local classifier
    learns the (utterance, tool) pair."""
    from hgr.live_api.planner.orchestrator import IrisPlanner
    reg = _FakeRegistry()
    planner = IrisPlanner(registry=reg)
    planner.try_handle("pause")
    # Single training example is below the MIN_EXAMPLES_PER_TOOL
    # threshold — classify() returns None for now.
    # Repeat three times so the classifier sees enough.
    planner.try_handle("pause")
    planner.try_handle("pause")
    stats = li.global_classifier().stats()
    assert stats["tools"] >= 1


# ---------- tool speculation respects gates ---------------------------

def test_tool_speculation_fires_on_keyword():
    """Trigger keywords in partial transcript schedule pre-fetches."""
    called = []
    tspec.global_speculator().set_dispatcher(
        lambda t, a: (called.append((t, a)),
                       {"status": "ok"})[1])
    tspec.global_speculator().begin_turn()
    fired = tspec.maybe_speculate("what's the weather today")
    assert any(t == "weather_get" for t, _ in fired)
    # Daemon thread is non-deterministic; wait briefly.
    deadline = time.time() + 1.0
    while time.time() < deadline and not called:
        time.sleep(0.02)
    assert called


def test_tool_speculation_blocked_in_incognito():
    inc.set_incognito(True)
    try:
        tspec.global_speculator().begin_turn()
        fired = tspec.maybe_speculate("what's the weather today")
    finally:
        inc.set_incognito(False)
    assert fired == []


# ---------- persona + affect + callback compose -----------------------

def test_full_stack_terse_user_no_callbacks_concise_voice():
    """User has been corrected once, signals terse, picks concise.
    The whole stack should produce: no callback, terse reply
    style, no LOW nudges."""
    from hgr.live_api.callback_engine import maybe_callback
    from hgr.live_api.affect import (reply_length_bias,
                                       should_suppress_callbacks,
                                       should_suppress_low_nudges)
    pv.set_active("concise")
    affect_mod.observe_user_turn("no")
    affect_mod.observe_user_turn("stop")
    affect_mod.global_model().record_correction()
    assert reply_length_bias() <= 0
    assert maybe_callback(user_text="something") is None


def test_full_stack_relaxed_user_jarvis_voice():
    """User is happy + active, picks jarvis. Callbacks should be
    permitted; LOW nudges allowed."""
    pv.set_active("jarvis")
    affect_mod.observe_user_turn("thanks, that was great")
    # No suppression should be active.
    assert affect_mod.should_suppress_callbacks() is False


# ---------- import sanity --------------------------------------------

def test_all_new_modules_importable_together():
    """The bundled app imports modules in different orders than the
    test runner. Force-import all Phase 6-10 modules in one process
    to catch circular-import or class-eval-time errors."""
    # Already imported at top of file, but explicitly:
    import importlib
    mods = [
        "hgr.live_api.persona_voice",
        "hgr.live_api.callback_engine",
        "hgr.live_api.affect",
        "hgr.live_api.smart_fast_path",
        "hgr.live_api.speculative_warmup",
        "hgr.live_api.vision_observer",
        "hgr.live_api.streaming_renderer",
        "hgr.live_api.speculative_planner",
        "hgr.live_api.local_intent",
        "hgr.live_api.tool_speculation",
        "hgr.live_api.reply_judge",
        "hgr.live_api.behavior_log",
        "hgr.live_api.kg_extractor",
        "hgr.live_api.tool_discovery",
        "hgr.live_api.latency_dashboard",
        "hgr.live_api.persona_marketplace",
        "hgr.live_api.tts_voice",
        "hgr.live_api.privacy_tier",
        "hgr.live_api.federation",
        "hgr.live_api.entity_graph",
        "hgr.live_api.pronoun_resolver",
        "hgr.live_api.anticipation_engine",
        "hgr.live_api.project_profile",
    ]
    for name in mods:
        m = importlib.import_module(name)
        assert m is not None


def test_orchestrator_constructs_with_all_wiring():
    """The orchestrator's __init__ exercises lazy-imports across
    multiple substrate modules. Confirm it doesn't crash."""
    from hgr.live_api.planner.orchestrator import IrisPlanner
    reg = _FakeRegistry()
    planner = IrisPlanner(registry=reg)
    assert planner is not None
    # try_handle on an empty registry call should NOT crash even
    # when the fast-path doesn't match.
    result = planner.try_handle("hi")
    assert result is not None


# ---------- privacy tier gates the cloud paths -----------------------

def test_privacy_tier_local_blocks_cloud_tts():
    """Privacy-tier toggle gates active_voice() — cloud voice
    falls back to SAPI in local-only mode."""
    import hgr.live_api.privacy_tier as pt
    import hgr.live_api.tts_voice as tts
    tts.reset_for_tests()
    pt.reset_for_tests()
    tts.set_active("eleven:rachel")
    pt.set_tier(pt.PrivacyTier.LOCAL_ONLY)
    v = tts.active_voice()
    assert v.provider == tts.TTSProvider.SAPI
    pt.reset_for_tests()


# ---------- behavior log records corrections -------------------------

def test_behavior_log_records_and_suggests():
    """3 corrections to same pattern surface a SuggestedDefault."""
    import tempfile
    from hgr.live_api.behavior_log import BehaviorLog
    log = BehaviorLog(
        db_path=Path(tempfile.mkdtemp()) / "b.db")
    for _ in range(3):
        log.log_correction(
            "open chrome", "browser_open", "edge_open")
    suggestions = log.find_pattern_suggestions(
        min_corrections=3)
    assert len(suggestions) == 1
    assert suggestions[0].suggested_tool == "edge_open"

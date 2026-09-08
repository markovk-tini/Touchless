"""Tests for ModelRouter (Phase 2 B2)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from hgr.live_api.model_router import (  # noqa: E402
    LatencyBudget, ModelRouter, TaskKind, _TIER_MODEL,
)


class _FakeMeter:
    def __init__(self, *, over: bool = False, near: bool = False):
        self._over = over
        self._near = near

    def is_over_cap(self):
        return self._over

    def is_near_cap(self):
        return self._near


class _FakeLedger:
    def __init__(self, flaky: bool = False):
        self._flaky = flaky

    def is_currently_flaky(self, tool: str) -> bool:
        return self._flaky


# ---- default routing ---------------------------------------------------

def test_planner_defaults_to_haiku_when_anthropic_available():
    r = ModelRouter(has_anthropic_key=True, has_openai_key=True,
                    has_local_runtime=True)
    d = r.route(kind=TaskKind.PLANNER)
    assert d.tier == "haiku"
    assert d.provider == "anthropic"
    assert "claude-haiku" in d.model_id


def test_synthesis_defaults_to_sonnet_when_anthropic_available():
    r = ModelRouter(has_anthropic_key=True, has_openai_key=True,
                    has_local_runtime=True)
    d = r.route(kind=TaskKind.SYNTHESIS)
    assert d.tier == "sonnet"
    assert d.provider == "anthropic"


def test_realtime_convo_routes_to_gpt_realtime():
    r = ModelRouter(has_openai_key=True, has_anthropic_key=True)
    d = r.route(kind=TaskKind.REALTIME_CONVO)
    assert d.tier == "realtime"
    assert d.provider == "openai"


# ---- cascading ---------------------------------------------------------

def test_planner_cascades_to_local_when_no_keys():
    r = ModelRouter(has_anthropic_key=False, has_openai_key=False,
                    has_local_runtime=True)
    d = r.route(kind=TaskKind.PLANNER)
    assert d.tier == "local"
    assert d.provider == "local"


def test_planner_cascades_to_gpt5mini_when_no_anthropic():
    r = ModelRouter(has_anthropic_key=False, has_openai_key=True,
                    has_local_runtime=True)
    d = r.route(kind=TaskKind.PLANNER)
    # haiku unavailable → cascade down: gpt5mini is next.
    assert d.tier == "gpt5mini"
    assert d.provider == "openai"


# ---- cost-cap behavior -------------------------------------------------

def test_over_cap_forces_local():
    r = ModelRouter(has_anthropic_key=True, has_openai_key=True,
                    cost_meter=_FakeMeter(over=True))
    d = r.route(kind=TaskKind.SYNTHESIS)
    assert d.tier == "local"
    assert "over cap" in d.reason


def test_near_cap_steps_down_one_tier():
    r = ModelRouter(has_anthropic_key=True, has_openai_key=True,
                    cost_meter=_FakeMeter(near=True))
    # Synthesis normally sonnet → near-cap steps DOWN to gpt5mini.
    d = r.route(kind=TaskKind.SYNTHESIS)
    assert d.tier == "gpt5mini"


# ---- reliability bump --------------------------------------------------

def test_planner_flaky_bumps_one_tier_up():
    r = ModelRouter(has_anthropic_key=True, has_openai_key=True,
                    ledger=_FakeLedger(flaky=True))
    d = r.route(kind=TaskKind.PLANNER, tool_name="haiku")
    # haiku is currently flaky → bump up: next up from haiku is
    # gpt5mini in our cascade.
    assert d.tier == "gpt5mini"


def test_planner_not_flaky_stays_on_haiku():
    r = ModelRouter(has_anthropic_key=True, has_openai_key=True,
                    ledger=_FakeLedger(flaky=False))
    d = r.route(kind=TaskKind.PLANNER, tool_name="haiku")
    assert d.tier == "haiku"


# ---- latency budget ----------------------------------------------------

def test_instant_budget_demotes_sonnet_to_haiku():
    r = ModelRouter(has_anthropic_key=True, has_openai_key=True)
    d = r.route(kind=TaskKind.SYNTHESIS, latency=LatencyBudget.INSTANT)
    assert d.tier == "haiku"


# ---- env overrides -----------------------------------------------------

def test_env_override_routes_to_specific_tier(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_ROUTER_PLANNER", "sonnet")
    r = ModelRouter(has_anthropic_key=True, has_openai_key=True)
    d = r.route(kind=TaskKind.PLANNER)
    assert d.tier == "sonnet"
    assert "TOUCHLESS_ROUTER_PLANNER" in d.reason


def test_force_local_env_overrides_everything(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_FORCE_LOCAL_MODELS", "1")
    r = ModelRouter(has_anthropic_key=True, has_openai_key=True)
    d = r.route(kind=TaskKind.SYNTHESIS)
    assert d.tier == "local"


def test_force_tier_param_short_circuits():
    r = ModelRouter(has_anthropic_key=True, has_openai_key=True)
    d = r.route(kind=TaskKind.PLANNER, force_tier="local")
    assert d.tier == "local"
    assert "force_tier" in d.reason


# ---- known tiers -------------------------------------------------------

def test_all_tier_models_resolve_to_provider_and_id():
    for tier, (provider, model_id) in _TIER_MODEL.items():
        assert provider in ("openai", "anthropic", "local")
        assert model_id

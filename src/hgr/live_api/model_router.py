"""Cost-aware model router.

Picks the right "tier" of model for each kind of work. The router
NEVER calls a model itself — it returns a decision; callers wire
their own SDKs (openai, anthropic, llama-server). This stays a
zero-dep module so it can be imported anywhere safely.

Tiers (from cheapest to highest quality):
  * `local`    — bundled llama-server + Qwen / Llama (free, slow)
  * `haiku`    — Claude Haiku 4.5 (cheap, fast, good for planning,
                 critique, fact-extraction)
  * `gpt5mini` — OpenAI gpt-5-mini (current dev planner)
  * `sonnet`   — Claude Sonnet 4.6 (high-quality synthesis, hard
                 plans, code-touching)
  * `realtime` — gpt-realtime (only for conversational TTS-bound
                 turns; expensive)

Routing inputs:
  * task kind (planner / critique / synthesis / fact_extract / etc.)
  * estimated input tokens
  * user-visible latency budget ("instant" | "fast" | "background")
  * cost meter (today's spend, daily cap)
  * tool reliability ledger (when planner model has been flaky,
    prefer the next tier up for revisions)
  * "force_local" override env

The decision dataclass returns: (provider, model_id, reason). The
caller then dispatches through the matching SDK.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class TaskKind(str, Enum):
    PLANNER = "planner"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    FACT_EXTRACT = "fact_extract"
    CONTEXT_COMPRESS = "context_compress"
    PROSE_RENDER = "prose_render"
    CODE_TOUCH = "code_touch"
    REALTIME_CONVO = "realtime_convo"


class LatencyBudget(str, Enum):
    INSTANT = "instant"       # < 200ms preferred (UI-blocking)
    FAST = "fast"             # < 2s preferred
    BACKGROUND = "background" # whenever


@dataclass
class ModelDecision:
    tier: str                 # "local" | "haiku" | "gpt5mini" | "sonnet" | "realtime"
    provider: str             # "openai" | "anthropic" | "local"
    model_id: str             # full SDK model id
    reason: str = ""
    # Cost guardrails the caller should respect:
    max_input_tokens: int = 8_000
    max_output_tokens: int = 1_024


# Conservative defaults for each tier — kept here so callers don't
# have to chase model IDs through 12 files when they change.
_TIER_MODEL = {
    "local":    ("local", "qwen-2.5-3b-instruct-q4"),
    "haiku":    ("anthropic", "claude-haiku-4-5-20251001"),
    "gpt5mini": ("openai", "gpt-5-mini"),
    "sonnet":   ("anthropic", "claude-sonnet-4-6"),
    "realtime": ("openai", "gpt-realtime"),
}

# Default routing matrix: maps task kind to a preferred tier.
# Overridden by env (TOUCHLESS_ROUTER_<TASK_KIND>) and by runtime
# signals (cost cap, reliability, latency budget).
_DEFAULT_ROUTING = {
    TaskKind.PLANNER:           "haiku",
    TaskKind.CRITIQUE:          "haiku",
    TaskKind.SYNTHESIS:         "sonnet",
    TaskKind.FACT_EXTRACT:      "haiku",
    TaskKind.CONTEXT_COMPRESS:  "haiku",
    TaskKind.PROSE_RENDER:      "haiku",
    TaskKind.CODE_TOUCH:        "sonnet",
    TaskKind.REALTIME_CONVO:    "realtime",
}


# Per-tier rough cost in micro-dollars per 1k tokens. Used for cost-
# cap fallback selection ("which tier can I afford"). Conservative
# HIGH estimates so we never under-budget.
_TIER_COST_USD_PER_1K_IN = {
    "local":    0.0,
    "haiku":    0.001,
    "gpt5mini": 0.001,
    "sonnet":   0.003,
    "realtime": 0.020,
}


class ModelRouter:
    """Single source of truth for which model handles which task."""

    # Per-tier fallback order. Used when the preferred tier's
    # provider isn't available — fall to the next available tier in
    # the list. "local" is always the final stop because the bundled
    # runtime is provider-less.
    _FALLBACK_CHAIN = {
        "realtime": ("sonnet", "gpt5mini", "haiku", "local"),
        "sonnet":   ("gpt5mini", "haiku", "local"),
        "gpt5mini": ("haiku", "local"),
        "haiku":    ("gpt5mini", "local"),
        "local":    (),
    }
    # Strict cheap→expensive ordering used by cost-cap "step down" and
    # reliability "step up" decisions. Both directions need a single
    # canonical sort.
    _COST_ORDER = ("local", "haiku", "gpt5mini", "sonnet", "realtime")

    def __init__(self, *,
                 cost_meter: Optional[Any] = None,
                 ledger: Optional[Any] = None,
                 has_openai_key: Optional[bool] = None,
                 has_anthropic_key: Optional[bool] = None,
                 has_local_runtime: Optional[bool] = None) -> None:
        self._cost_meter = cost_meter
        self._ledger = ledger
        # Capability flags (None = autodetect from env at call-time).
        self._has_openai = has_openai_key
        self._has_anthropic = has_anthropic_key
        self._has_local = has_local_runtime

    # ---- main entry point ---------------------------------------------

    def route(self, *, kind: TaskKind,
              latency: LatencyBudget = LatencyBudget.FAST,
              est_input_tokens: int = 1_000,
              tool_name: Optional[str] = None,
              force_tier: Optional[str] = None
              ) -> ModelDecision:
        """Return the best tier for the requested task."""
        if force_tier:
            return self._build(force_tier,
                               reason=f"forced via force_tier={force_tier}")

        # Env override per task kind.
        env_key = f"TOUCHLESS_ROUTER_{kind.value.upper()}"
        env_tier = os.environ.get(env_key, "").strip().lower()
        if env_tier and env_tier in _TIER_MODEL:
            return self._build(env_tier,
                               reason=f"{env_key}={env_tier}")

        # Global force-local override (offline mode).
        if os.environ.get("TOUCHLESS_FORCE_LOCAL_MODELS", "0") == "1":
            return self._build("local",
                               reason="TOUCHLESS_FORCE_LOCAL_MODELS=1")

        base = _DEFAULT_ROUTING[kind]

        # Latency penalty: realtime/sonnet are slower-on-first-token
        # for short prompts; for INSTANT budget bias toward haiku.
        if latency == LatencyBudget.INSTANT and base in ("sonnet",
                                                          "realtime"):
            base = "haiku"

        # Reliability bump: if planner has been failing repeatedly,
        # bump to the next tier up. Caller's tool_name lets the
        # router consult the ledger.
        if (self._ledger is not None and tool_name
                and kind == TaskKind.PLANNER):
            try:
                if self._ledger.is_currently_flaky(tool_name):
                    base = self._next_tier_up(base)
            except Exception:
                pass

        # Cost cap fallback: cascade DOWN until we find an affordable
        # tier. If the meter says we're over cap, force local. If near
        # cap, skip the top tiers.
        if self._cost_meter is not None:
            try:
                if self._cost_meter.is_over_cap():
                    return self._build("local",
                                       reason="cost meter over cap")
                if hasattr(self._cost_meter, "is_near_cap") \
                        and self._cost_meter.is_near_cap():
                    # Step DOWN one tier when near cap, never UP.
                    base = self._next_tier_down(base)
            except Exception:
                pass

        # Capability fallback: if the chosen tier needs a key we
        # don't have, cascade down.
        base = self._available_or_cascade_down(base)
        return self._build(base, reason="default routing")

    # ---- helpers ------------------------------------------------------

    def _build(self, tier: str, reason: str) -> ModelDecision:
        provider, model_id = _TIER_MODEL[tier]
        return ModelDecision(
            tier=tier, provider=provider, model_id=model_id,
            reason=reason,
        )

    def _has(self, provider: str) -> bool:
        if provider == "openai":
            if self._has_openai is not None:
                return self._has_openai
            return bool(os.environ.get("OPENAI_API_KEY"))
        if provider == "anthropic":
            if self._has_anthropic is not None:
                return self._has_anthropic
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        if provider == "local":
            if self._has_local is not None:
                return self._has_local
            # Local always "available" in router terms; whether it's
            # *running* is the caller's check.
            return True
        return False

    def _available_or_cascade_down(self, tier: str) -> str:
        """If `tier`'s provider isn't available, walk the per-tier
        FALLBACK_CHAIN and return the first tier whose provider IS
        available. Local is always the final fallback."""
        provider, _ = _TIER_MODEL[tier]
        if self._has(provider):
            return tier
        for cand in self._FALLBACK_CHAIN.get(tier, ()):
            cand_provider, _ = _TIER_MODEL[cand]
            if self._has(cand_provider):
                return cand
        return "local"

    def _next_tier_up(self, tier: str) -> str:
        """One tier MORE expensive (= higher quality). For reliability
        bump when a cheaper tier has been flaky."""
        idx = self._COST_ORDER.index(tier)
        return self._COST_ORDER[min(len(self._COST_ORDER) - 1, idx + 1)]

    def _next_tier_down(self, tier: str) -> str:
        """One tier CHEAPER. For cost-cap near-limit step-down."""
        idx = self._COST_ORDER.index(tier)
        return self._COST_ORDER[max(0, idx - 1)]


# ---- module-level singleton --------------------------------------------

_global_router: Optional[ModelRouter] = None


def global_router() -> ModelRouter:
    global _global_router
    if _global_router is None:
        _global_router = ModelRouter()
    return _global_router


def _reset_for_tests() -> None:
    global _global_router
    _global_router = None

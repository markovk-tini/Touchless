"""Privacy tier toggle — one switch between cloud + local-only.

Phase-10 subscription polish. Today the choice between paid cloud
calls (OpenAI Realtime / Anthropic / OpenAI TTS) and local-only
inference (whisper.cpp + llama.cpp + SAPI TTS) is determined by:
  * Environment variables
  * Cost meter slow-mode
  * Per-feature flags

That's fine for power users — invisible to subscribers. The
privacy tier is a single sticky toggle the user picks in settings:

  * `cloud`      — best quality, best latency, billed via subscription
                    proxy. The default for paying users.
  * `local_only` — no network calls. Voice clones disabled,
                    Claude Vision off, planner routes to local LLM.

The toggle is stored as a memory preference + read by all the
downstream gates: `tts_voice.active_voice`, `vision_dispatch`,
`tool_speculation`, `kg_extractor`, etc.

Persistence:
  * `set_tier(t)` — sets it sticky for the session.
  * `get_tier()` — resolves from explicit → env → memory → default.
  * Default = `cloud` (subscribers get the good stuff by default).

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import threading
from enum import Enum
from typing import Any, Optional


class PrivacyTier(str, Enum):
    CLOUD      = "cloud"
    LOCAL_ONLY = "local_only"


_lock = threading.RLock()
_active: Optional[PrivacyTier] = None
_DEFAULT = PrivacyTier.CLOUD


def set_tier(tier: PrivacyTier | str) -> bool:
    """Set sticky for the session. Returns True when the value is
    a known tier."""
    global _active
    if isinstance(tier, PrivacyTier):
        with _lock:
            _active = tier
        return True
    try:
        slug = str(tier).strip().lower()
        chosen = PrivacyTier(slug)
        with _lock:
            _active = chosen
        return True
    except Exception:
        return False


def get_tier(*, memory: Optional[Any] = None) -> PrivacyTier:
    with _lock:
        if _active is not None:
            return _active
    env = (os.environ.get("TOUCHLESS_PRIVACY_TIER") or "").strip()
    if env:
        try:
            return PrivacyTier(env.lower())
        except Exception:
            pass
    if memory is not None:
        try:
            facts = memory._store.find_facts(
                kind="preference", key="privacy_tier")
            if facts:
                val = str(facts[0].value or "").strip().lower()
                try:
                    return PrivacyTier(val)
                except Exception:
                    pass
        except Exception:
            pass
    return _DEFAULT


def is_cloud(*, memory: Optional[Any] = None) -> bool:
    return get_tier(memory=memory) == PrivacyTier.CLOUD


def is_local_only(*, memory: Optional[Any] = None) -> bool:
    return get_tier(memory=memory) == PrivacyTier.LOCAL_ONLY


def persist_choice(memory: Any, tier: PrivacyTier) -> bool:
    if memory is None or tier is None:
        return False
    try:
        memory.write_preference("privacy_tier", tier.value)
        return True
    except Exception:
        pass
    try:
        memory._store.write_fact(
            kind="preference", key="privacy_tier",
            value=tier.value)
        return True
    except Exception:
        return False


def reset_for_tests() -> None:
    global _active
    with _lock:
        _active = None


def headline_label(tier: PrivacyTier) -> str:
    if tier == PrivacyTier.CLOUD:
        return "Cloud (best quality)"
    if tier == PrivacyTier.LOCAL_ONLY:
        return "Local only (private)"
    return str(tier.value)

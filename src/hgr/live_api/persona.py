"""Iris persona anchor — user-tweakable personality without code edits.

Phase-3 polish. The Iris system prompt has a fixed PERSONA ANCHOR
hardcoded in `live_api_manager`. Users want to tune it: shorter
replies, more formal tone, different name, etc. This module owns
that tunable layer.

Resolution order (first non-empty wins):
  1. Explicit `set_persona_block(text)` call (UI-set).
  2. Env var `TOUCHLESS_PERSONA` (string of free-form persona text).
  3. Env var `TOUCHLESS_PERSONA_FILE` → path to a UTF-8 text file
     whose contents become the persona.
  4. Per-user memory fact `preference.persona` (the user told Iris
     "be more concise", "speak British English", etc., and the
     fact extractor persisted it).
  5. Built-in DEFAULT persona ("warm, smart, slightly witty…").

The resolved block is injected at the END of the system prompt so
the user's preference overrides any model-baked tendencies. Caps
at 1500 chars so the prompt doesn't bloat.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional


DEFAULT_PERSONA = (
    "You are Iris. Speak warmly, conversationally, and with a "
    "touch of dry wit — the way a smart friend sitting next to "
    "the user would. Prefer contractions, vary phrasing, keep "
    "replies short unless the user asks for depth. Never read "
    "URLs aloud. Use 'tomorrow' / 'Thursday' for dates, not "
    "ISO timestamps. Don't restate identical facts ('it's 65, "
    "feels like 65' — just say 'it's 65'). When you don't know "
    "something, say so plainly and offer to look."
)

# Hard cap on the persona block so a misconfigured env-var can't
# blow up the system prompt.
MAX_PERSONA_CHARS = 1500


_lock = threading.RLock()
_explicit_block: Optional[str] = None


def set_persona_block(text: Optional[str]) -> None:
    """UI-side setter. Pass None to clear and fall through to env /
    memory / default."""
    global _explicit_block
    with _lock:
        _explicit_block = None if text is None else str(text)[:MAX_PERSONA_CHARS]


def get_persona_block(*, memory: Optional[Any] = None) -> str:
    """Resolve the active persona block. Best-effort across all
    sources; never raises. Returns a non-empty string.

    Phase-7: when a named preset is active (set via
    `persona_voice.set_active`, env, or memory), its style_block +
    few-shot examples win over the legacy DEFAULT. Free-form
    `_explicit_block` / TOUCHLESS_PERSONA env / file overrides
    still take precedence — they're explicit user intent."""
    with _lock:
        if _explicit_block:
            return _explicit_block
    env = os.environ.get("TOUCHLESS_PERSONA", "").strip()
    if env:
        return env[:MAX_PERSONA_CHARS]
    file_path = os.environ.get("TOUCHLESS_PERSONA_FILE", "").strip()
    if file_path:
        try:
            text = Path(file_path).read_text(
                encoding="utf-8", errors="ignore").strip()
            if text:
                return text[:MAX_PERSONA_CHARS]
        except Exception:
            pass
    # Memory-backed free-form persona (legacy — "be more concise").
    if memory is not None:
        try:
            facts = memory._store.find_facts(
                kind="preference", key="persona")
            if facts:
                value = str(facts[0].value or "").strip()
                if value:
                    return value[:MAX_PERSONA_CHARS]
        except Exception:
            pass
    # Phase-7: named preset wins over DEFAULT_PERSONA. Falls back
    # to "default" preset which is functionally equivalent to
    # DEFAULT_PERSONA but adds few-shot examples.
    try:
        from .persona_voice import style_block as _voice_block
        block = _voice_block(memory=memory)
        if block:
            return block[:MAX_PERSONA_CHARS]
    except Exception:
        pass
    return DEFAULT_PERSONA


def reset_for_tests() -> None:
    global _explicit_block
    with _lock:
        _explicit_block = None

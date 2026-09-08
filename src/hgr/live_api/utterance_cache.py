"""Utterance cache + Anthropic prompt-cache hint helper.

Two distinct caches with one shared theme: "we just said this; don't
spend tokens saying it again."

  1. UtteranceCache
     ----------------
     Maps a normalized user utterance text to the final user-facing
     reply produced for it. Persists in memory only; bounded LRU.
     Hits return the cached reply INSTANTLY (no model call at all)
     when the same question is asked within `ttl_seconds`.

     What's safe to cache: deterministic question→answer turns where
     the world hasn't changed (weather is NOT safe — different time).
     Caller decides cache-worthiness by passing `cacheable=True/False`
     on `put()`.

  2. PromptCacheHint
     ----------------
     For paid LLM calls (Anthropic Claude) the SDK supports prompt
     caching via `cache_control: {"type": "ephemeral"}` on a message
     block. This module produces the canonical "system prompt" block
     for our planner/critic/synthesizer prompts so Claude caches the
     fixed prefix and only bills for the variable suffix per request.

     Marks the system prompt + the tool catalog block as cacheable.
     Roughly cuts paid input tokens by 60-80% across an extended
     session.

Author: Konstantin Markov
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---- Utterance cache ---------------------------------------------------

@dataclass
class CachedReply:
    text: str
    message: str            # the original final user-facing message
    payload: Dict[str, Any] # whatever the orchestrator returned
    created_at: float
    hits: int = 0


class UtteranceCache:
    """LRU map of normalized-utterance → cached reply."""

    def __init__(self, *, max_entries: int = 256,
                 ttl_seconds: int = 600) -> None:
        self._max = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._d: "OrderedDict[str, CachedReply]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _normalize(text: str) -> str:
        """Collapse whitespace + lowercase + strip ALL punctuation
        runs to a single space, then re-collapse. Two utterances
        with different punctuation density should hit the same key
        ('Hey  Iris ,  what is UP?' == 'hey iris what is up'). Caps
        at 240 chars — anything longer is highly individualized and
        rarely a cache candidate."""
        if not text:
            return ""
        t = (text or "").strip().lower()
        # Drop apostrophes inside words so "what's" == "whats".
        t = re.sub(r"(\w)'(\w)", r"\1\2", t)
        # Replace any run of non-word non-space chars with a space.
        t = re.sub(r"[^\w\s]+", " ", t)
        # Collapse whitespace runs.
        t = re.sub(r"\s+", " ", t).strip()
        return t[:240]

    def get(self, text: str) -> Optional[CachedReply]:
        key = self._normalize(text)
        if not key:
            return None
        with self._lock:
            entry = self._d.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.time() - entry.created_at > self._ttl:
                # Expired — drop and miss.
                self._d.pop(key, None)
                self._misses += 1
                return None
            # LRU: move to end.
            self._d.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            return entry

    def put(self, text: str, message: str,
            payload: Optional[Dict[str, Any]] = None,
            *, cacheable: bool = True) -> bool:
        if not cacheable:
            return False
        key = self._normalize(text)
        if not key:
            return False
        with self._lock:
            self._d[key] = CachedReply(
                text=text, message=message, payload=payload or {},
                created_at=time.time())
            self._d.move_to_end(key)
            while len(self._d) > self._max:
                self._d.popitem(last=False)
            return True

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"size": len(self._d), "hits": self._hits,
                    "misses": self._misses}

    def clear(self) -> int:
        with self._lock:
            n = len(self._d)
            self._d.clear()
            self._hits = 0
            self._misses = 0
            return n


# ---- Prompt cache hint helpers -----------------------------------------

def build_anthropic_request_kwargs(
        *,
        system_prompt: str,
        tool_catalog_text: str,
        per_turn_user_text: str,
        memory_context: str = "",
) -> Dict[str, Any]:
    """Build an Anthropic Messages API payload as a dict of kwargs
    ready to splat into `client.messages.create(**kwargs, ...)`.

    Layout:
      {
        "system": [
           {"type": "text", "text": system_prompt,
            "cache_control": {"type": "ephemeral"}},
           {"type": "text", "text": tool_catalog_text,
            "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [{"role": "user", "content": <text>}],
      }

    Per F-001 audit: Anthropic's Messages API does NOT accept a
    `role="system"` message inside the `messages` array. System
    content goes as the top-level `system=` parameter, which itself
    can be a list of text blocks with `cache_control` set. This
    helper now returns the kwargs in that shape.
    """
    sys_blocks: List[Dict[str, Any]] = []
    if system_prompt.strip():
        sys_blocks.append({
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        })
    if tool_catalog_text.strip():
        sys_blocks.append({
            "type": "text",
            "text": tool_catalog_text,
            "cache_control": {"type": "ephemeral"},
        })
    user_text = ""
    if memory_context.strip():
        user_text = memory_context.strip() + "\n\n"
    user_text += per_turn_user_text
    kwargs: Dict[str, Any] = {
        "messages": [{"role": "user", "content": user_text}],
    }
    if sys_blocks:
        kwargs["system"] = sys_blocks
    return kwargs


# Legacy alias — back-compat for any caller using the old name.
# Returns the same dict shape; passing it through `**kwargs` works.
def make_anthropic_messages_with_cache(
        *,
        system_prompt: str,
        tool_catalog_text: str,
        per_turn_user_text: str,
        memory_context: str = "",
) -> Dict[str, Any]:
    """DEPRECATED — use build_anthropic_request_kwargs(...). Returns
    the same dict shape now; the old list-of-messages shape was a
    bug that violated the Anthropic API contract."""
    return build_anthropic_request_kwargs(
        system_prompt=system_prompt,
        tool_catalog_text=tool_catalog_text,
        per_turn_user_text=per_turn_user_text,
        memory_context=memory_context,
    )


def system_prompt_cache_hash(*, system_prompt: str,
                             tool_catalog_text: str) -> str:
    """Stable hash for the cached prefix. Useful for telemetry +
    cache-hit verification. SHA-256, first 12 hex chars."""
    h = hashlib.sha256()
    h.update((system_prompt or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((tool_catalog_text or "").encode("utf-8"))
    return h.hexdigest()[:12]


# ---- module-level singleton --------------------------------------------

_global_utterance_cache: Optional[UtteranceCache] = None
_cache_lock = threading.Lock()


def global_utterance_cache() -> UtteranceCache:
    global _global_utterance_cache
    if _global_utterance_cache is None:
        with _cache_lock:
            if _global_utterance_cache is None:
                _global_utterance_cache = UtteranceCache()
    return _global_utterance_cache


def _reset_for_tests() -> None:
    global _global_utterance_cache
    _global_utterance_cache = None

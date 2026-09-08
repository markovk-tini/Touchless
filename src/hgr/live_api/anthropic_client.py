"""Minimal Anthropic Messages API client.

Phase-4 multi-provider routing. The ModelRouter picks tiers across
both providers (haiku / sonnet / gpt5mini / etc.) — but the actual
LLM call sites (`planner_llm.plan`, `synthesizer.summarize`,
`plan_reviser._llm_revise`) were OpenAI-only. This module is a
zero-dep `urllib` wrapper for the Anthropic Messages API so those
sites can route to either provider.

Why not the official SDK? Two reasons:
  1. Zero extra dependency footprint. PyInstaller bundling cost
     stays flat.
  2. We only need a single call shape (one-shot completion +
     optional JSON-mode + optional prompt cache). The full SDK
     surfaces dozens of features we don't use.

The shape mirrors `planner_llm._call`:
  * Input: messages list (system + user blocks)
  * Output: parsed JSON dict (when json_mode=True) OR raw text

Cost accounting happens via the existing `cost_meter`; we record
input + output token estimates after each call.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


_API_URL = "https://api.anthropic.com/v1/messages"
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_TOKENS = 1024


def configured() -> bool:
    return bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip())


def call_messages(*,
                  model: str,
                  system: Optional[Any] = None,
                  messages: List[Dict[str, Any]],
                  max_tokens: int = _DEFAULT_MAX_TOKENS,
                  json_mode: bool = False,
                  timeout: float = _DEFAULT_TIMEOUT,
                  ) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
    """One-shot call. Returns (text, usage) — text is the model's
    reply (None on any failure); usage is `{input_tokens, output_tokens}`
    when the API returned a usage block, None otherwise.

    `system` may be a string (treated as a single text block) or a
    list of {type:"text", text:"..."} blocks — pass the second form
    when you want `cache_control` markers for prompt caching.

    `json_mode` doesn't have a direct Anthropic equivalent; we steer
    via system prompt + use a strict prefill ('{') in the assistant
    role to nudge JSON-only output.
    """
    if not configured():
        return None, None
    body: Dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "messages": list(messages),
    }
    if system is not None:
        body["system"] = system
    if json_mode:
        # Steer toward JSON: assistant prefill + reminder line.
        body["messages"] = list(messages) + [
            {"role": "assistant", "content": "{"}
        ]
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            try:
                from .planner.scheduler import scheduler
                scheduler().record_rate_limit("anthropic")
            except Exception:
                pass
        return None, None
    except Exception:
        return None, None
    # Extract text from the first text block. Anthropic returns
    # content as a list of {type, text/...} blocks.
    text_parts: List[str] = []
    try:
        for b in data.get("content") or []:
            if b.get("type") == "text":
                text_parts.append(b.get("text") or "")
    except Exception:
        pass
    text = "".join(text_parts).strip() or None
    if text and json_mode:
        # When we prefilled '{', the model continues from there.
        # The reply may not include the leading '{'; prepend if
        # missing.
        if not text.startswith("{"):
            text = "{" + text
    usage = None
    try:
        u = data.get("usage")
        if isinstance(u, dict):
            usage = {
                "input_tokens": int(u.get("input_tokens") or 0),
                "output_tokens": int(u.get("output_tokens") or 0),
            }
    except Exception:
        pass
    return text, usage


def messages_with_system_cache(*,
                                system_prompt: str,
                                tool_catalog_text: str,
                                user_text: str,
                                memory_context: str = ""
                                ) -> Dict[str, Any]:
    """Convenience: build the kwargs dict for `call_messages` with
    prompt caching marked on the system blocks. Use as:

        kwargs = messages_with_system_cache(...)
        text, usage = call_messages(model="haiku", **kwargs)
    """
    from .utterance_cache import build_anthropic_request_kwargs
    return build_anthropic_request_kwargs(
        system_prompt=system_prompt,
        tool_catalog_text=tool_catalog_text,
        per_turn_user_text=user_text,
        memory_context=memory_context,
    )


def record_spend(model: str,
                 usage: Optional[Dict[str, int]],
                 fallback_in_chars: int = 0,
                 fallback_out_chars: int = 0) -> None:
    """Charge the cost meter for this call. Uses the API's usage
    block when present; falls back to a char-based estimate
    otherwise. Best-effort, never raises."""
    try:
        from .cost_meter import global_meter
        if usage is not None:
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
        else:
            tokens_in = max(1, fallback_in_chars // 4)
            tokens_out = max(1, fallback_out_chars // 4)
        global_meter().record(model,
                              tokens_in=tokens_in,
                              tokens_out=tokens_out)
    except Exception:
        pass

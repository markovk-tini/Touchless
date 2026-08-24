"""Streaming-first prose renderer — Iris's first word reaches the
TTS before the LLM finishes its last.

Phase-8 latency. `prose_renderer.render_jarvis` waits for the
full HTTP response before returning. With ~500-token replies the
user feels every millisecond of generation time. This module
exposes `stream_jarvis` — same inputs, but yields incremental
`(token, is_final)` tuples as the SSE stream produces them.

Architecture mirrors `prose_renderer`:

  * Same `_SYSTEM_PROMPT` + persona preset injection + callback
    hint splice.
  * Same fact-preservation guard, applied on the FULL buffered
    output once the stream ends. When the guard rejects, the
    caller is told via a final tuple `("", False)` AND we expose
    a `last_render_passed_guard()` flag so the orchestrator can
    fall back to the deterministic `fallback` string.
  * Same SHA-256 cache. Streaming hits write into the same cache
    the non-streaming path uses, so a repeat of the exact same
    question returns instantly.

Caller pattern:

    tts_pause()
    for token, final in stream_jarvis(question, tool, result, fallback):
        if token:
            tts_buffer(token)
        if final:
            if last_render_passed_guard():
                tts_resume()      # commit
            else:
                tts_flush()
                tts_speak(fallback)

When OPENAI_API_KEY isn't set, we degrade gracefully: yield the
fallback as a single chunk + final=True.

Author: Konstantin Markov
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, Tuple

from . import prose_renderer as _pr


_API_URL = "https://api.openai.com/v1/chat/completions"
_MODEL = _pr._MODEL          # share the same model / config
_TIMEOUT_S = _pr._TIMEOUT_S
_MAX_TOKENS = _pr._MAX_TOKENS


_guard_lock = threading.RLock()
_LAST_GUARD_PASSED = True


def last_render_passed_guard() -> bool:
    with _guard_lock:
        return _LAST_GUARD_PASSED


def _set_guard_passed(value: bool) -> None:
    global _LAST_GUARD_PASSED
    with _guard_lock:
        _LAST_GUARD_PASSED = bool(value)


def _build_messages(question, tool_name, tool_result, fallback,
                     context_tail, preset_block, callback_hint):
    sys_prompt = _pr._SYSTEM_PROMPT
    if preset_block:
        sys_prompt = (
            sys_prompt
            + "\n\nACTIVE VOICE PRESET — follow this VOICE while "
            "keeping every fact verbatim:\n" + preset_block)
    if callback_hint:
        sys_prompt = sys_prompt + "\n\n" + callback_hint
    try:
        result_blob = json.dumps(tool_result, default=str,
                                  sort_keys=True)[:8000]
    except Exception:
        result_blob = "{}"
    parts = []
    if context_tail.strip():
        parts.append(
            "Recent conversation (most recent last):\n"
            + context_tail)
    parts.extend([
        f"User just asked: {question or '(no recent question)'}",
        f"Tool that ran: {tool_name}",
        f"Tool data (JSON):\n{result_blob}",
        f"Fallback phrasing if a better rewrite isn't possible:\n"
        f"{fallback}",
        "Compose ONE reply. Preserve every fact verbatim.",
    ])
    return sys_prompt, "\n\n".join(parts), result_blob


def stream_jarvis(
    question: str,
    tool_name: str,
    tool_result: Dict[str, Any],
    fallback: str,
    *,
    timeout: float = _TIMEOUT_S,
    context: str = "",
    callback_hint: str = "",
) -> Iterator[Tuple[str, bool]]:
    """Stream the Jarvis-voice rewrite of `tool_result` as
    incremental SSE chunks. Yields (token, is_final) tuples. The
    last tuple has `is_final=True`. On any failure the generator
    yields one fallback tuple and exits.

    After the generator exhausts, `last_render_passed_guard()`
    indicates whether the streamed output preserved facts. The
    caller MUST consult it before committing the streamed text
    to TTS — when it returns False, fall back to `fallback`.
    """
    if not fallback or not _pr.should_render(tool_name, fallback):
        _set_guard_passed(True)
        yield (fallback, True)
        return
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        _set_guard_passed(True)
        yield (fallback, True)
        return
    if not isinstance(tool_result, dict) or not tool_result:
        _set_guard_passed(True)
        yield (fallback, True)
        return

    # Phase-7 persona preset.
    try:
        from . import persona_voice
        preset_block = persona_voice.style_block(
            with_examples=True, max_examples=3)
        preset_temp = persona_voice.temperature()
    except Exception:
        preset_block = ""
        preset_temp = 0.6
    # Phase-7 affect: caller may want to bias terser/expansive
    # replies. Apply a soft cap by lowering max_tokens when terse.
    max_tokens = _MAX_TOKENS
    try:
        from . import affect
        bias = affect.reply_length_bias()
        if bias < 0:
            max_tokens = max(120, _MAX_TOKENS // 3)
        elif bias > 0:
            max_tokens = _MAX_TOKENS
    except Exception:
        pass

    context_tail = (context or "")[-400:]
    sys_prompt, user_msg, result_blob = _build_messages(
        question, tool_name, tool_result, fallback,
        context_tail, preset_block, callback_hint)

    # Cache: identical key returns the non-streaming cached string
    # in one shot (yielded as the single final chunk).
    try:
        preset_tag = persona_voice.active_preset().name
    except Exception:
        preset_tag = "default"
    cache_key = hashlib.sha256(
        f"{question}::{tool_name}::{result_blob}::"
        f"{context_tail}::{preset_tag}".encode("utf-8")).hexdigest()
    cached = _pr._CACHE.get(cache_key)
    if cached:
        _set_guard_passed(True)
        yield (cached, True)
        return

    try:
        body = json.dumps({
            "model": _MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": float(preset_temp),
            "max_tokens": int(max_tokens),
            "stream": True,
        }).encode("utf-8")
    except Exception:
        _set_guard_passed(True)
        yield (fallback, True)
        return

    req = urllib.request.Request(
        _API_URL, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        })

    buffer: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                _set_guard_passed(True)
                yield (fallback, True)
                return
            for line in resp:
                if not line:
                    continue
                try:
                    text = line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if not text or not text.startswith("data:"):
                    continue
                payload_str = text[len("data:"):].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    payload = json.loads(payload_str)
                except Exception:
                    continue
                try:
                    delta = ((payload.get("choices") or [{}])[0]
                              .get("delta", {})
                              .get("content"))
                except Exception:
                    delta = None
                if not delta:
                    continue
                buffer.append(delta)
                yield (delta, False)
    except urllib.error.HTTPError:
        _set_guard_passed(True)
        yield (fallback, True)
        return
    except urllib.error.URLError:
        _set_guard_passed(True)
        yield (fallback, True)
        return
    except Exception:
        _set_guard_passed(True)
        yield (fallback, True)
        return

    full = "".join(buffer).strip()
    if not full:
        _set_guard_passed(True)
        yield (fallback, True)
        return
    # Same preamble strip as render_jarvis.
    full = re.sub(
        r"^(sure[,!]?\s+|of course[,!]?\s+|here(?:'s| is| you go|'s the rundown)[:,]?\s+|"
        r"alright[,!]?\s+|got it[,!]?\s+|absolutely[,!]?\s+|okay[,!]?\s+)",
        "", full, flags=re.IGNORECASE).lstrip()

    guard_ok = _pr._facts_preserved(tool_result, full)
    _set_guard_passed(guard_ok)
    if guard_ok:
        if len(_pr._CACHE) >= _pr._CACHE_LIMIT:
            _pr._CACHE.clear()
        _pr._CACHE[cache_key] = full
    # Final tuple: signals end-of-stream + whether the buffered
    # text passed the guard (via last_render_passed_guard()).
    yield ("", True)

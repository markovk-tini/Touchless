"""Reply judge — grade each Iris turn after delivery, feed into
the bandit.

Phase-9 self-improvement. The substrate already has `prompt_variants`
(bandit-tuned A/B over prompt styles) but no signal-quality
producer drives it — it's wired to a few opt-in modules. The reply
judge is a generic producer: after each Iris reply lands, a tiny
LLM pass grades it on 3 axes:

  * helpfulness — did the reply actually answer the question?
  * accuracy   — did the facts cited match the tool data?
  * vibe       — was the style consistent with the active persona?

Scores 0..1. The orchestrator submits the average back to
`prompt_variants.record_outcome` so the winning preset gets
reinforced + losing one decays.

Cheap by design:
  * Uses gpt-4o-mini (~$0.0001 per judgement)
  * Bounded JSON output (3 floats + 1 sentence reason)
  * Skipped when reply is short status confirmation (already in
    prose_renderer's _SKIP_RENDER_BELOW_CHARS regime)
  * Async — fired on a daemon thread; never blocks the user
  * Cost-aware — skipped in slow mode
  * Incognito-aware

Public:
  * `judge_async(question, reply, persona_name, tool_results, callback)`
    — fire-and-forget. Calls callback with `JudgeResult` when done.
  * `judge_sync(...)` — blocking variant used in tests.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


_API_URL = "https://api.openai.com/v1/chat/completions"
_MODEL = "gpt-4o-mini"
_TIMEOUT_S = 6.0
_MAX_TOKENS = 220
_SKIP_BELOW_CHARS = 25


@dataclass
class JudgeResult:
    helpfulness: float = 0.0          # 0..1
    accuracy: float = 0.0             # 0..1
    vibe: float = 0.0                 # 0..1
    reason: str = ""
    cost_estimate_usd: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    def average(self) -> float:
        if self.skipped:
            return 0.0
        return (self.helpfulness + self.accuracy
                + self.vibe) / 3.0


_SYSTEM = (
    "You are an evaluator grading Iris (a voice assistant) replies. "
    "Output STRICT JSON with three floats 0..1 and a one-sentence "
    "reason:\n"
    "{\n"
    "  \"helpfulness\": 0.0..1.0,\n"
    "  \"accuracy\": 0.0..1.0,\n"
    "  \"vibe\": 0.0..1.0,\n"
    "  \"reason\": \"<1 short sentence>\"\n"
    "}\n"
    "Helpfulness: did the reply actually answer the question? "
    "(0 = irrelevant, 1 = direct hit)\n"
    "Accuracy: every fact mentioned has to be supported by the "
    "tool data. Invented names / counts / dates → 0.\n"
    "Vibe: does the style match the persona description provided? "
    "(0 = wrong tone, 1 = perfect fit)\n"
    "Output ONLY JSON. No preamble."
)


def _build_user_message(question: str, reply: str,
                        persona_name: str,
                        tool_blob: str) -> str:
    return (
        f"PERSONA: {persona_name}\n"
        f"USER ASKED: {question}\n"
        f"TOOL DATA AVAILABLE (JSON):\n{tool_blob[:2000]}\n"
        f"IRIS REPLY:\n{reply}\n"
    )


def _http_call(api_key: str, body: bytes,
               timeout: float) -> Optional[Dict[str, Any]]:
    req = urllib.request.Request(
        _API_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        return None
    except urllib.error.URLError:
        return None
    except Exception:
        return None


def _parse_score(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Strip code fences if model wrapped JSON.
    text = re.sub(
        r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text).strip()
    try:
        return json.loads(text)
    except Exception:
        # Try to find the first { ... } block.
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def judge_sync(*,
               question: str,
               reply: str,
               persona_name: str = "default",
               tool_results: Optional[List[Dict[str, Any]]] = None,
               timeout: float = _TIMEOUT_S) -> JudgeResult:
    """Blocking variant. Returns JudgeResult with skipped=True
    when not applicable."""
    if not reply or len(reply.strip()) < _SKIP_BELOW_CHARS:
        return JudgeResult(skipped=True,
                           skip_reason="reply too short")
    if not question:
        return JudgeResult(skipped=True,
                           skip_reason="no question")
    # Incognito → don't grade (avoid sending content out).
    try:
        from .incognito import is_incognito
        if is_incognito():
            return JudgeResult(skipped=True,
                               skip_reason="incognito")
    except Exception:
        pass
    # Cost-aware.
    try:
        from .cost_meter import global_meter
        m = global_meter()
        if getattr(m, "is_slow_mode", lambda: False)():
            return JudgeResult(skipped=True,
                               skip_reason="cost slow_mode")
    except Exception:
        pass
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return JudgeResult(skipped=True,
                           skip_reason="no api key")
    try:
        tool_blob = json.dumps(tool_results or [],
                                default=str, sort_keys=True)
    except Exception:
        tool_blob = "[]"
    user_msg = _build_user_message(
        question, reply, persona_name, tool_blob)
    try:
        body = json.dumps({
            "model": _MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": _MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
    except Exception:
        return JudgeResult(skipped=True,
                           skip_reason="body serialize fail")
    payload = _http_call(api_key, body, timeout)
    if not payload:
        return JudgeResult(skipped=True,
                           skip_reason="http error")
    try:
        content = ((payload.get("choices") or [{}])[0]
                   .get("message", {}).get("content") or "")
    except Exception:
        content = ""
    parsed = _parse_score(content)
    if not parsed:
        return JudgeResult(skipped=True,
                           skip_reason="parse fail")
    def _clamp(v):
        try:
            x = float(v)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, x))
    return JudgeResult(
        helpfulness=_clamp(parsed.get("helpfulness")),
        accuracy=_clamp(parsed.get("accuracy")),
        vibe=_clamp(parsed.get("vibe")),
        reason=str(parsed.get("reason", ""))[:300])


def judge_async(*,
                question: str,
                reply: str,
                persona_name: str = "default",
                tool_results: Optional[List[Dict[str, Any]]] = None,
                callback: Optional[
                    Callable[[JudgeResult], None]] = None,
                timeout: float = _TIMEOUT_S) -> threading.Thread:
    """Fire and forget. Returns the spawned thread (for tests).
    `callback(result)` runs on the worker thread when done."""
    def _run():
        result = judge_sync(
            question=question, reply=reply,
            persona_name=persona_name,
            tool_results=tool_results,
            timeout=timeout)
        if callback is not None:
            try:
                callback(result)
            except Exception:
                pass
    t = threading.Thread(target=_run, daemon=True,
                          name="reply-judge")
    t.start()
    return t


# ---- bandit hook -----------------------------------------------------

def feed_bandit(result: JudgeResult, *,
                persona_name: str) -> None:
    """Report the judge's average back to the persona_voice usage
    tracker so winning presets get reinforced."""
    if result.skipped:
        return
    try:
        from . import persona_voice
        kept = result.average() >= 0.6
        persona_voice.record_outcome(persona_name, kept=kept)
    except Exception:
        pass

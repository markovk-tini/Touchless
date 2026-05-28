"""Synthesizer — when a Plan specifies `final='synthesize'`, take the step
outputs and produce a short natural-language answer with ONE cheap-LLM
call. Falls back to None on failure / when not configured / when the
scheduler says cheap-LLM is currently throttled; the orchestrator then
uses the deterministic `_format_plan_message` instead.

Designed dormant: if OPENAI_API_KEY is absent, this just returns None and
the existing terse summary path stays in effect.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .plan import Plan, StepResult
from .scheduler import scheduler

DEFAULT_MODEL = "gpt-5-mini"
API_URL = "https://api.openai.com/v1/chat/completions"

# Don't push 50 KB of email body into the synthesizer prompt — keep each
# step output compact. The synthesizer cares about *what happened*, not
# every byte of raw payload.
_MAX_OUTPUT_CHARS = 1200
_MAX_USER_JSON = 8000


def configured() -> bool:
    return bool((os.environ.get("OPENAI_API_KEY") or "").strip())


class Synthesizer:
    """One cheap-LLM call: plan goal + step outputs → conversational reply."""

    def __init__(self, model: Optional[str] = None, logger: Any = None) -> None:
        self._model = (
            model
            or os.environ.get("TOUCHLESS_SYNTH_MODEL")
            or os.environ.get("TOUCHLESS_PLANNER_MODEL")
            or DEFAULT_MODEL
        )
        self._logger = logger

    def summarize(self, plan: Plan, results: List[StepResult]) -> Optional[str]:
        if not configured() or not results:
            return None
        try:
            messages = self._build_messages(plan, results)
            text = self._call(messages)
            return (text or "").strip() or None
        except urllib.error.HTTPError as exc:
            # 429 → tell the scheduler so the next request can route around
            # cheap-LLM, then fall back to the deterministic summary.
            if exc.code == 429:
                scheduler().record_rate_limit("cheap-llm")
            if self._logger:
                self._logger.event("synthesizer_http_error", code=exc.code)
            return None
        except Exception as exc:
            if self._logger:
                self._logger.exception("synthesizer_failed", exc)
            return None

    # ---- prompt -----------------------------------------------------------
    def _build_messages(self, plan: Plan, results: List[StepResult]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for r in results:
            rows.append({
                "step": r.step_id,
                "tool": r.tool,
                "status": r.status,
                "result": self._trim_output(r.output),
                "error": r.error,
            })
        system = (
            "You are Iris's response synthesizer. Given the user's goal and "
            "the outputs of the tools we just ran, write a concise, "
            "conversational 1-3 sentence reply that DIRECTLY answers the "
            "goal using real data from the results. Don't recite tool "
            "names, step ids, or JSON. Include a useful link if one exists. "
            "If a step failed and the overall goal failed, say so briefly."
        )
        payload = {"goal": plan.goal, "steps": rows}
        user = json.dumps(payload, default=str)[:_MAX_USER_JSON]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _trim_output(out: Any) -> Any:
        if not isinstance(out, dict):
            return str(out)[:_MAX_OUTPUT_CHARS] if out is not None else None
        trimmed: Dict[str, Any] = {}
        for k, v in out.items():
            if k == "raw":  # huge payloads we don't want in the prompt
                continue
            if isinstance(v, str):
                trimmed[k] = v[:_MAX_OUTPUT_CHARS]
            elif isinstance(v, (int, float, bool)) or v is None:
                trimmed[k] = v
            elif isinstance(v, list):
                trimmed[k] = v[:10]
            elif isinstance(v, dict):
                trimmed[k] = {kk: vv for kk, vv in list(v.items())[:10]}
            else:
                trimmed[k] = str(v)[:_MAX_OUTPUT_CHARS]
        return trimmed

    # ---- HTTP --------------------------------------------------------------
    def _call(self, messages: List[Dict[str, Any]]) -> str:
        key = os.environ["OPENAI_API_KEY"]
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 300,
        }
        req = urllib.request.Request(
            API_URL, data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        scheduler().record_call("cheap-llm")
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return (payload.get("choices") or [{}])[0].get("message", {}).get("content", "")

"""Cheap-LLM JSON planner — one Chat Completions call decomposes a request
into a structured Plan, so the Executor runs each step directly without
per-step model turns. Reserves gpt-realtime for voice; this uses a cheap text
model (mini-class) with much higher TPM.

Designed dormant: if OPENAI_API_KEY is absent or anything fails, plan()
returns None and the caller falls back to the existing LLM path.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .plan import Plan, Step
from .scheduler import scheduler

DEFAULT_MODEL = "gpt-5-mini"
API_URL = "https://api.openai.com/v1/chat/completions"


def configured() -> bool:
    """True if we have what we need to make a planning call."""
    return bool((os.environ.get("OPENAI_API_KEY") or "").strip())


class LLMPlanner:
    """Produces a JSON Plan from a natural-language request using one cheap-LLM
    Chat Completions call. The Plan is then executed locally by Executor."""

    def __init__(self, registry: Any, logger: Any = None,
                 model: Optional[str] = None) -> None:
        self._registry = registry
        self._logger = logger
        self._model = (model
                       or os.environ.get("TOUCHLESS_PLANNER_MODEL")
                       or DEFAULT_MODEL)

    def plan(self, goal: str) -> Optional[Plan]:
        if not configured() or self._registry is None:
            return None
        try:
            messages = self._build_messages(goal)
            data = self._call(messages)
            return self._parse(goal, data)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                scheduler().record_rate_limit("cheap-llm")
            if self._logger:
                self._logger.event("planner_llm_http_error", code=exc.code)
            return None
        except Exception as exc:
            if self._logger:
                self._logger.exception("planner_llm_failed", exc)
            return None

    # ---- prompt + tool catalog --------------------------------------------
    def _build_messages(self, goal: str) -> List[Dict[str, Any]]:
        catalog = self._tool_catalog()
        system = (
            "You are Iris's task planner. Given a user request, output a JSON "
            "Plan: a minimal ordered list of tool calls to accomplish it. "
            "Use only tools from the catalog. Prefer the cheapest layer: "
            "Touchless (deterministic) > connector API > screen/GUI. For "
            "multi-step tasks, decompose into the smallest set of steps; mark "
            "ordering with depends_on (list of earlier step ids). Pass "
            "earlier outputs forward via {step:N.field} string refs in args "
            "(e.g. \"to\":\"{step:1.email}\"). End with final='synthesize' if "
            "a natural-language answer over gathered data is needed.\n\n"
            "Available tools:\n" + catalog + "\n\n"
            "Output STRICT JSON, no commentary:\n"
            '{"goal": <string>, "steps": [{"id": <int starting at 1>, '
            '"tool": <string>, "args": <object>, "depends_on": [<int>]}], '
            '"final": "return" | "synthesize"}'
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": goal},
        ]

    def _tool_catalog(self) -> str:
        """Compact tool catalog for the planner — connector tools available
        right now + the highest-leverage built-ins. Keeps the prompt small."""
        lines: List[str] = []
        try:
            for entry in self._registry.connector_catalog():
                desc = entry.get("description") or ""
                for tool_name in entry.get("tools", []):
                    lines.append(f"- {tool_name}  [{entry.get('id')}] {desc}")
        except Exception:
            pass
        lines.extend([
            "- read_screen [iris]  Local OCR + UIA: returns active-window text "
            "AND clickable_elements with screen-pixel x,y. Use to read or "
            "summarize on-screen content (e.g. emails in Outlook).",
            "- click_screen [iris]  Click pixel coords (set coordinate_space="
            "'screen' to use read_screen's x,y directly).",
            "- click_type [iris]  Click + type + optional Enter in ONE call — "
            "use to fill a field and send.",
            "- type_text [iris]  Type text into the focused field.",
            "- press_hotkey [iris]  Press a key combination.",
            "- open_app [iris]  Launch any app by name (Outlook, Teams, etc.).",
            "- open_url [iris]  Open a URL in the default browser.",
            "- web_navigate / web_get_text / web_get_links [iris]  Drive the "
            "controllable Chrome (fresh, not the user's signed-in one).",
        ])
        return "\n".join(lines)

    # ---- HTTP --------------------------------------------------------------
    def _call(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        key = os.environ["OPENAI_API_KEY"]
        body = {
            "model": self._model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
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
        text = (payload.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return json.loads(text or "{}")

    # ---- parse + validate --------------------------------------------------
    @staticmethod
    def _parse(goal: str, data: Dict[str, Any]) -> Optional[Plan]:
        raw_steps = data.get("steps") or []
        if not isinstance(raw_steps, list) or not raw_steps:
            return None
        steps: List[Step] = []
        for s in raw_steps:
            if not isinstance(s, dict):
                continue
            tool = str(s.get("tool") or "").strip()
            if not tool:
                continue
            try:
                steps.append(Step(
                    id=int(s.get("id") or (len(steps) + 1)),
                    tool=tool,
                    args=s.get("args") if isinstance(s.get("args"), dict) else {},
                    depends_on=[int(x) for x in (s.get("depends_on") or [])],
                    description=str(s.get("description") or ""),
                ))
            except Exception:
                continue
        if not steps:
            return None
        final = str(data.get("final") or "return").lower()
        if final not in ("return", "synthesize", "speak"):
            final = "return"
        return Plan(goal=str(data.get("goal") or goal), steps=steps, final=final)

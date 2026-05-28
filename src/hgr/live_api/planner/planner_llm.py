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

    def plan(self, goal: str, memory_context: str = "") -> Optional[Plan]:
        if not configured() or self._registry is None:
            return None
        try:
            messages = self._build_messages(goal, memory_context=memory_context)
            data = self._call(messages)
            return self._parse(goal, data, self._known_tool_names())
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
    def _build_messages(self, goal: str,
                        memory_context: str = "") -> List[Dict[str, Any]]:
        catalog = self._tool_catalog()
        system = (
            "You are Iris's task planner. Given a user request, output a JSON "
            "Plan: a minimal ordered list of tool calls to accomplish it. "
            "Use only tools from the catalog. Prefer the cheapest layer: "
            "Touchless (deterministic) > connector API > screen/GUI. For "
            "multi-step tasks, decompose into the smallest set of steps; mark "
            "ordering with depends_on (list of earlier step ids). Pass "
            "earlier outputs forward via {step:N.field} string refs in args "
            "(e.g. \"to\":\"{step:1.email}\"). List indexing works too: "
            "{step:N.results[0].url}.\n\n"
            "IMPORTANT: 'synthesize' is NOT a tool — it's the plan-level "
            '"final" flag. Set "final":"synthesize" when the user wants a '
            "natural-language answer over what was gathered; the framework "
            "calls a cheap-LLM at the end automatically. Do NOT add a step "
            'named \"synthesize\" or \"summarize\" — those don\'t exist as '
            "tools.\n\n"
            "CROSS-PROVIDER CHAINS: universal data (email addresses, URLs, "
            "plain text, names) is fine to pass between providers — pulling "
            "a contact from Microsoft contacts_search and sending via "
            "gmail_send is a valid plan. ID-shaped data is NOT: a message "
            "ID from ms_mail_list / ms_mail_search is only valid in "
            "Microsoft Graph; do NOT feed it to gmail tools. Same for "
            "calendar event IDs and OneDrive item IDs.\n\n"
            "USER PREFERENCES: if the 'Context from prior turns' block "
            "lists a 'preference default_send_via = X', use that sender tool "
            "(X is the actual tool name, e.g. gmail_send or ms_mail_send). "
            "If the current request explicitly says 'from my gmail' / 'via "
            "outlook' / 'from my .edu account' etc., that one-shot override "
            "wins for THIS request (don't update the preference).\n\n"
            "OUTPUT SHAPES of common lookup tools (so you reference fields "
            "correctly with {step:N.field...}):\n"
            "  contacts_search → {contacts: [{name, emails:[str,...], "
            "source_account}, ...], searched_accounts:[str,...]}. Searches "
            "ALL connected Microsoft accounts by default. Reference the "
            "first hit as {step:N.contacts[0].emails[0]}. To restrict to "
            "one account pass account='edu' (or 'gmail', etc.).\n"
            "  ms_mail_list / ms_mail_search → {messages: [{id, subject, "
            "from, ...}, ...]}\n"
            "  web_search → {results: [{title, url, snippet}, ...]} → "
            "use {step:N.results[0].url}\n\n"
            "WEB CHAIN: to 'read and summarize an article from a search', the "
            "correct chain is: (1) web_search, (2) web_navigate the chosen "
            "result's URL, (3) web_get_text, then set final='synthesize'. "
            "Snippets from web_search alone are too short to summarize from.\n\n"
            "Example for 'search for AI news and summarize the top result':\n"
            '{"goal":"search for AI news and summarize the top result",'
            '"steps":['
            '{"id":1,"tool":"web_search","args":{"query":"latest AI news",'
            '"count":5,"recent_days":7}},'
            '{"id":2,"tool":"web_navigate","args":{"url_or_query":'
            '"{step:1.results[0].url}"},"depends_on":[1]},'
            '{"id":3,"tool":"web_get_text","args":{"max_chars":4000},'
            '"depends_on":[2]}'
            '],"final":"synthesize"}\n\n'
            "Example for 'find Dani's email and send him a quick hi' "
            "(contacts_search returns an email STRING — fine to feed into "
            "any *_send tool; pick whichever sender the user prefers, or "
            "the one most likely to be authenticated):\n"
            '{"goal":"find Dani\'s email and send him a quick hi",'
            '"steps":['
            '{"id":1,"tool":"contacts_search","args":{"query":"Dani"}},'
            '{"id":2,"tool":"gmail_send","args":{"to":'
            '"{step:1.contacts[0].emails[0]}","subject":"Hi",'
            '"body":"Hi Dani!"},"depends_on":[1]}'
            '],"final":"return"}\n\n'
            "Available tools:\n" + catalog + "\n\n"
            "Output STRICT JSON, no commentary:\n"
            '{"goal": <string>, "steps": [{"id": <int starting at 1>, '
            '"tool": <string>, "args": <object>, "depends_on": [<int>]}], '
            '"final": "return" | "synthesize"}'
        )
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        if memory_context:
            # Background facts from prior turns. Goes in as system so the model
            # treats it as established context rather than a new user turn.
            messages.append({"role": "system", "content": memory_context})
        messages.append({"role": "user", "content": goal})
        return messages

    def _known_tool_names(self) -> Optional[set]:
        """Set of tool names the registry actually exposes. Used by _parse
        to drop steps the LLM hallucinated. Returns None on failure so the
        parser falls back to its old non-validating behaviour rather than
        rejecting everything when introspection breaks."""
        try:
            names = {s.get("name") for s in self._registry.openai_tools()
                     if isinstance(s, dict) and s.get("name")}
            return names or None
        except Exception:
            return None

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
            "- web_search [iris]  Structured search results "
            "(title/url/snippet) WITHOUT spinning up Chrome — Google CSE if "
            "configured, else DuckDuckGo. Use first for any 'search the web' "
            "/ 'latest news' request. Returns {results: [{title, url, "
            "snippet}, ...]}; reference the first result's URL as "
            "{step:N.results[0].url}. Args: query, count (default 5), site "
            "(optional domain), recent_days (e.g. 7 for news).",
            "- web_navigate [iris]  Open a URL (or run a Google search for "
            "a plain query) in the controlled Chrome and wait for load. "
            "Args: url_or_query. MUST come BEFORE web_get_text / "
            "web_get_links — those two read whatever page is currently "
            "loaded; they do NOT accept a url arg themselves.",
            "- web_get_text [iris]  Return the current page's visible text. "
            "ONLY callable AFTER web_navigate has loaded the page you want. "
            "Args: max_chars (default 4000). No url arg.",
            "- web_get_links [iris]  Return the current page's links as "
            "{index, text, url}. Same rule: requires a prior web_navigate. "
            "Args: contains, limit.",
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
    def _parse(goal: str, data: Dict[str, Any],
               known_tools: Optional[set] = None) -> Optional[Plan]:
        raw_steps = data.get("steps") or []
        if not isinstance(raw_steps, list) or not raw_steps:
            return None
        steps: List[Step] = []
        seen_ids: set = set()
        for s in raw_steps:
            if not isinstance(s, dict):
                continue
            tool = str(s.get("tool") or "").strip()
            if not tool:
                continue
            # Defensive: drop steps whose tool isn't actually exposed by the
            # registry (e.g. the LLM emitting 'synthesize' as a step name
            # when it should have set final='synthesize'). When known_tools
            # is None we couldn't introspect, so we keep the old behaviour.
            if known_tools is not None and tool not in known_tools:
                continue
            try:
                step_id = int(s.get("id") or (len(steps) + 1))
                # The LLM occasionally repeats an id; if it does we'd silently
                # overwrite the earlier step's result in the executor. Renumber
                # duplicates to the next free id so both steps actually run.
                while step_id in seen_ids:
                    step_id += 1
                seen_ids.add(step_id)
                steps.append(Step(
                    id=step_id,
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

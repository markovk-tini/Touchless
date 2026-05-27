"""Cost policy + routing classification for the iris routing ladder.

Touchless already routes cheapest-first: Layer-0 deterministic router →
API connectors → built-in/GUI executor. This module makes that ladder
*explicit and loggable* so we can see (and test) which cost tier handled
each command, per the JARVIS-style cost policy.

Levels (lowest = cheapest/most reliable; prefer the lowest that works):
  0 local-free   deterministic router, app launch, hotkeys, media keys,
                 window control, UIA/DOM reads, file ops, cache
  1 local-cheap  local OCR, fuzzy search, local classifier
  2 connector    cheap API connector (Spotify/Gmail/Graph/etc.)
  3 text-llm     a text LLM reasoning call
  4 vision       screenshot / vision reasoning
  5 realtime     GPT-Realtime live multimodal session

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Tuple

LEVEL_LABELS = {
    0: "local-free",
    1: "local-cheap",
    2: "connector-api",
    3: "text-llm",
    4: "vision",
    5: "realtime",
}

# Built-in (executor) tools that actually use the screenshot/vision path.
_VISION_TOOLS = {
    "get_screen_context", "click_screen", "zoom_screen", "click_zoom",
    "draw_path", "draw_shape", "drag", "wait_for_screen_text",
}
# Built-in tools that lean on local OCR.
_OCR_TOOLS = {"click_text_on_screen"}
# Built-in tools that delegate to a separate LLM/agent.
_LLM_TOOLS = {"send_to_coding_agent", "follow_up_coding_agent"}


def classify(tool_name: str, source: str) -> Tuple[int, str]:
    """Map a (tool, source) to a cost level + short human label.

    `source` is the executing layer the manager already tags:
    'touchless' (Layer 0), 'connector' (API), or 'iris' (built-in/GUI).
    """
    if source == "touchless":
        return 0, "deterministic local command"
    if source == "connector":
        return 2, "API connector (no screenshots/clicks)"
    # source == "iris" -> built-in / computer-use executor; refine by tool.
    if tool_name in _VISION_TOOLS:
        return 4, "screenshot/vision"
    if tool_name in _OCR_TOOLS:
        return 1, "local OCR"
    if tool_name in _LLM_TOOLS:
        return 3, "delegated to coding agent (LLM)"
    return 0, "local OS/UI action"


def decision_record(*, raw: str = "", tool: str, source: str,
                    status: str = "", call_id: str = "") -> dict:
    """Build the structured routing-decision log entry."""
    level, why = classify(tool, source)
    return {
        "tool": tool,
        "source": source,
        "cost_level": level,
        "cost_label": LEVEL_LABELS.get(level, "?"),
        "why_short": why,
        "status": status,
        "raw_command": raw,
        "call_id": call_id,
    }

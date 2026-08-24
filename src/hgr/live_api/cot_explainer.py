"""Conversational "why did you do that?" explainer for the cot_layer.

Phase-3 polish. `cot_layer` records every turn's structured
decision trail. When the user asks Iris "why did you do that?" /
"how did you decide?" / "explain that last call", we need to turn
the raw trail into Iris-voice prose.

This module owns the translation:
  * `looks_like_explain_request(text)` — cheap heuristic so the
    planner can short-circuit to this path instead of routing the
    question through Tier-2 (which would just hallucinate).
  * `explain_last_turn()` — pulls the most recent TurnTrail from
    cot_layer and renders a conversational summary.
  * `explain_by_turn_id(turn_id)` — same but for a specific past
    turn.

Output is plain text — the orchestrator's prose-renderer pass
will polish it further before TTS. Empty string when no trail
exists ("I don't have a record of that turn").

Author: Konstantin Markov
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# Cheap heuristics for "this is an explain-my-reasoning question".
_EXPLAIN_PATTERNS = (
    re.compile(r"\bwhy\s+did\s+you\b", re.IGNORECASE),
    re.compile(r"\bhow\s+did\s+you\s+(?:decide|choose|pick|figure)\b",
               re.IGNORECASE),
    re.compile(r"\bexplain\s+(?:that|the\s+last|your\s+(?:choice|reasoning))\b",
               re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:was|were)\s+you\s+thinking\b", re.IGNORECASE),
    re.compile(r"\bwalk\s+me\s+through\s+(?:that|your\s+reasoning)\b",
               re.IGNORECASE),
    re.compile(r"\bshow\s+(?:me\s+)?your\s+work\b", re.IGNORECASE),
    re.compile(r"\bwhy\s+the\b.*\bcall\b", re.IGNORECASE),
)


def looks_like_explain_request(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if not t or len(t) > 200:
        return False
    return any(p.search(t) for p in _EXPLAIN_PATTERNS)


def explain_last_turn() -> str:
    """Return a conversational explanation of the most recent turn
    recorded by cot_layer. Empty string when no trail exists."""
    try:
        from .cot_layer import global_cot_layer
        trail = global_cot_layer().last_turn()
    except Exception:
        return ""
    if trail is None:
        return ""
    return _render(trail)


def explain_by_turn_id(turn_id: str) -> str:
    if not turn_id:
        return ""
    try:
        from .cot_layer import global_cot_layer
        trail = global_cot_layer().explain_turn(turn_id)
    except Exception:
        return ""
    if trail is None:
        return ""
    return _render(trail)


def _render(trail: Dict[str, Any]) -> str:
    """Turn a TurnTrail dict into a one-paragraph explanation."""
    user_text = (trail.get("user_text") or "").strip()
    final_msg = (trail.get("final_message") or "").strip()
    decisions: List[Dict[str, Any]] = trail.get("decisions") or []
    tool_refs: List[str] = trail.get("tool_call_refs") or []
    if not decisions and not tool_refs:
        if user_text:
            return (f"Last turn you asked: \"{user_text[:120]}\". "
                    "I went straight to the answer — no multi-step "
                    "plan, so there's no decision trail to walk you "
                    "through.")
        return "I don't have a record of recent reasoning."
    parts: List[str] = []
    if user_text:
        parts.append(f"For \"{user_text[:120]}\", here's what I did:")
    # Summarize the decisions in order.
    for i, d in enumerate(decisions[:8], start=1):
        stage = d.get("stage") or "step"
        choice = d.get("choice") or "(unspecified)"
        why = d.get("why") or ""
        line = f"  {i}. {_stage_verb(stage)} — {choice}"
        if why:
            line += f" ({why[:120]})"
        parts.append(line)
    if len(decisions) > 8:
        parts.append(f"  …and {len(decisions) - 8} more steps.")
    # Tools that actually ran, if any. Lifts the explanation from
    # "I planned to do X" → "I planned to do X and then actually
    # called weather_get, drive_upload".
    if tool_refs:
        snip = tool_refs[:8]
        suffix = (f" (and {len(tool_refs) - 8} more)"
                  if len(tool_refs) > 8 else "")
        parts.append(f"Tools called: {', '.join(snip)}{suffix}.")
    if final_msg:
        parts.append(f"Final reply: \"{final_msg[:180]}\".")
    return "\n".join(parts)


def _stage_verb(stage: str) -> str:
    """Map a Decision.stage to human-readable past-tense phrasing."""
    return {
        "classify": "classified",
        "route":    "routed",
        "plan":     "planned",
        "revise":   "revised the plan",
        "critique": "ran a critique pass",
        "speak":    "synthesized the reply",
        "refuse":   "refused at the safety gate",
    }.get(stage, stage)

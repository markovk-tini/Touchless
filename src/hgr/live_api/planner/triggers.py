"""Phase-2 trigger heuristic — decides if a request *looks* multi-step
enough to be worth spending a cheap-LLM planning call on. Cheap, regex-
only; when in doubt, returns False and lets the request fall through to
the existing realtime path. False positives just waste one ~1K-token
call; false negatives waste nothing (the realtime fallback still works).

Also exports the small "risky tool" set used by the orchestrator's whole-
plan confirm-gate, so a generated plan that touches sending email,
uploading files, GUI input, etc. surfaces ONE confirm dialog summarizing
the whole plan instead of running silently.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
from typing import Iterable

# Chain words: when these appear, the request almost always has multiple
# sequential actions ("find it then send it", "open Outlook and then read").
_CHAIN_RE = re.compile(
    r"\b(?:and\s+then|then|after\s+that|after\s+(?:you\s+)?(?:do|finish)|"
    r"once\s+(?:you|that)|next\b)\b",
    re.IGNORECASE,
)

# "and <verb>" — "...and send him hi", "...also create a doc"
_AND_VERB_RE = re.compile(
    r"\b(?:and|,)\s+(?:also\s+|then\s+)?(?:please\s+)?"
    r"(?:open|close|send|find|read|search|click|type|press|create|make|"
    r"delete|summarize|summarise|upload|download|navigate|copy|move|"
    r"email|message|draft|compose|schedule|post|share)\b",
    re.IGNORECASE,
)

# Verbs that strongly imply an action; two distinct ones in the same
# sentence usually means a multi-step plan.
_ACTION_VERBS = (
    "open", "close", "send", "find", "read", "search", "click", "type",
    "press", "create", "make", "delete", "summarize", "summarise",
    "upload", "download", "navigate", "copy", "move", "email", "message",
    "draft", "compose", "schedule", "post", "share",
)
_VERB_PATTERNS = [re.compile(r"\b" + v + r"\b", re.IGNORECASE) for v in _ACTION_VERBS]


def looks_multi_action(text: str) -> bool:
    """True if `text` looks like a multi-step request worth planning."""
    t = (text or "").strip()
    if len(t) < 12:
        return False
    if _CHAIN_RE.search(t):
        return True
    if _AND_VERB_RE.search(t):
        return True
    # Two distinct action verbs in the same sentence.
    found = 0
    for pat in _VERB_PATTERNS:
        if pat.search(t):
            found += 1
            if found >= 2:
                return True
    return False


# Tools whose effects are hard to undo or visible to other people. A plan
# containing any of these triggers the whole-plan confirm-gate.
RISKY_TOOLS: frozenset = frozenset({
    # Outbound communication
    "outlook_compose", "outlook_send", "ms_mail_send",
    "gmail_send", "gmail_compose",
    "teams_post", "teams_send", "teams_channel_post",
    "slack_post", "slack_send",
    "discord_send",
    # Created / uploaded artifacts
    "drive_upload",
    "gdocs_create", "sheets_create", "slides_create",
    "excel_create", "word_create", "powerpoint_create",
    "onenote_create", "todo_add",
    "calendar_create", "calendar_event_create",
    # GUI input — can do anything the focused app accepts
    "click_screen", "click_type", "type_text", "press_hotkey",
    # Destructive
    "file_delete", "file_move", "drive_delete", "mail_delete",
})


def plan_needs_confirm(step_tools: Iterable[str]) -> bool:
    return any(t in RISKY_TOOLS for t in step_tools)

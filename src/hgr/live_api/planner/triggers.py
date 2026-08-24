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
# CRITICAL: every continuation string here MUST be a raw string (r"...").
# A regular Python string at the end with `\b` becomes a literal BACKSPACE
# character (\x08), silently breaking the entire alternation — every search
# returns None and the verb branch below has to carry the whole detection.
_AND_VERB_RE = re.compile(
    r"\b(?:and|,)\s+(?:also\s+|then\s+)?(?:please\s+)?"
    r"(?:open|close|send|find|read|search|click|type|press|create|make|"
    r"delete|summarize|summarise|upload|download|navigate|copy|move|"
    r"email|message|draft|compose|schedule|post|share|"
    # Added after live tests C19/C20: 'and add a task' / 'and set volume'
    # / 'and remove X' / 'and remind me'. Without these, multi-action
    # prompts containing common command verbs slip past — Layer 0 then
    # mis-fires on ONE embedded sub-action (e.g. picks 'set volume to 30'
    # out of a 3-clause prompt and runs only that).
    r"add|set|remove|remind|toggle|mute|unmute|"
    r"translate|rephrase|generate|produce|fetch|"
    r"play|pause|skip|launch|start)\b",
    re.IGNORECASE,
)

# Verbs that strongly imply an action; two distinct ones in the same
# sentence usually means a multi-step plan.
_ACTION_VERBS = (
    "open", "close", "send", "find", "read", "search", "click", "type",
    "press", "create", "make", "delete", "summarize", "summarise",
    "upload", "download", "navigate", "copy", "move", "email", "message",
    "draft", "compose", "schedule", "post", "share",
    # Added after live test 'create doc and write debrief': clearly action-
    # shaped second-action verbs. Common verbs ('get', 'show', 'tell', 'list')
    # are deliberately excluded to avoid false-positive multi-action on
    # ordinary single-intent requests like 'tell me Dani's email'.
    "write", "save", "store", "append", "include", "fill", "insert", "record",
    # Added after live tests C19 ('add a task, set volume, and tell me X')
    # and C20 ('write a haiku and add a task'): these prompts have 2+ clearly
    # distinct actions but the two-verb branch counted only 'write' (1 verb)
    # because 'add' and 'set' were missing here. Same noun-form stripping
    # below protects against false positives like 'set the table'.
    "add", "set", "remove", "remind", "toggle", "mute", "unmute",
    "translate", "rephrase", "generate", "produce", "fetch",
    "play", "pause", "skip", "launch", "start",
)
_VERB_PATTERNS = [re.compile(r"\b" + v + r"\b", re.IGNORECASE) for v in _ACTION_VERBS]


_CONNECTOR_RE = re.compile(r"[,;]| \band\b ", re.IGNORECASE)

# Strip noun-form occurrences of action-shaped words so the two-verb branch
# doesn't false-positive on "Dani's email" / "the message" / "this post"
# (the word follows a possessive or determiner — it's a noun, not a verb).
_NOUN_FORM_RE = re.compile(
    r"\b(?:'s|s)?\s*(?:'s|the|a|an|this|that|my|your|his|her|their|our)\s+"
    r"(?:email|message|post|search|share|copy|move|read|write|draft)\b",
    re.IGNORECASE,
)
_POSSESSIVE_NOUN_RE = re.compile(
    r"\b[a-z]+'s\s+(?:email|message|post|address|number|name|file|"
    r"document|note|task|calendar)\b",
    re.IGNORECASE,
)


def looks_multi_action(text: str) -> bool:
    """True if `text` looks like a multi-step request worth planning."""
    t = (text or "").strip()
    if len(t) < 12:
        return False
    if _CHAIN_RE.search(t):
        return True
    if _AND_VERB_RE.search(t):
        return True
    # Two distinct action verbs ONLY counts when something actually joins
    # them — otherwise "send dani an email" (one action, two verb words)
    # would be a false positive.
    if not _CONNECTOR_RE.search(t):
        return False
    # Strip noun-form occurrences so we don't mis-count "Dani's email" /
    # "the message" as the verb "email"/"message".
    stripped = _POSSESSIVE_NOUN_RE.sub(" ", t)
    stripped = _NOUN_FORM_RE.sub(" ", stripped)
    found = 0
    for pat in _VERB_PATTERNS:
        if pat.search(stripped):
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
    "email_send",
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

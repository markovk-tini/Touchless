"""Implicit fact extraction.

Phase-2 cognition. Where `memory/extractor.py` extracts facts from
PLANNER STEP ARGS (the deterministic stuff: this tool was called
with `to=dani@x` so dani's email is x), `implicit_facts` extracts
facts from FREE-FORM CONVERSATION — what the user said in chat,
what Iris wrote back, what assistant tool-results contained.

Conservative by design. Patterns must be:
  * Unambiguous — "my birthday is March 5" is clear; "I went to
    the doctor on the 5th" is not.
  * Low-cost — pure regex / dict lookups by default; LLM-backed
    extraction is opt-in via `TOUCHLESS_IMPLICIT_LLM=1`.
  * Reversible — every fact carries source metadata so the user
    can ask "where did you learn X?" and Iris can answer truthfully.

Categories extracted (rule-based):
  * preferences:  "I prefer dark mode", "always send via gmail"
  * locations:    "I live in Berlin", "I'm in Tokyo right now"
  * names:        "my name is Dani", "call me K"
  * contacts:     "<name>'s email is <addr>", "<name> @ <addr>"
  * schedules:    "my workday starts at 9am", "I sleep at 11"
  * birthdays:    "my birthday is <date>"
  * dislikes:     "I don't like X", "stop X"
  * pronouns:     "use they/them for me"

Every fact extracted goes through a confidence check before write
(0.0–1.0). Below `MIN_CONFIDENCE` (default 0.7) it's discarded
silently. Set TOUCHLESS_IMPLICIT_DEBUG=1 to log discards.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


MIN_CONFIDENCE = 0.7


@dataclass
class FactCandidate:
    """One extracted fact, pre-write. The MemoryManager decides whether
    to persist based on `confidence` and any deduplication checks."""
    kind: str         # 'preference' | 'location' | 'person' | etc.
    key: str          # canonicalized key (lowercased, trimmed)
    value: str        # canonicalized value
    confidence: float # 0.0–1.0
    source: str       # short label: "user said: 'I live in Berlin'"
    source_kind: str = "implicit"
    raw_span: str = ""  # the exact span the extractor pulled from


# ---- pattern table ----------------------------------------------------

_PATTERNS: List[Tuple[str, str, "re.Pattern"]] = []


def _pat(kind: str, key_or_fmt: str, regex: str,
         case_sensitive: bool = False) -> None:
    """Register a rule-based pattern. `key_or_fmt` is either a fixed
    key string ('default_send_via') or a Python format that pulls
    capture groups by index ('{0}' = first group). Set
    case_sensitive=True for patterns whose capture group depends on
    `[A-Z]` strictly meaning uppercase (proper nouns, names)."""
    flags = 0 if case_sensitive else re.IGNORECASE
    _PATTERNS.append((kind, key_or_fmt, re.compile(regex, flags)))


# ---- preferences ------------------------------------------------------

_pat("preference", "default_send_via",
     r"\b(?:always|prefer(?:ably)?|by default|default to|prefer to)\s+"
     r"(?:send|email|message)(?:ing)?\s+(?:via|through|with|from|using)\s+"
     r"(gmail|outlook|teams|slack|discord|signal|whatsapp)\b")

_pat("preference", "ui_theme",
     r"\bI\s+(?:prefer|like|want|use)\s+(dark|light)\s+(?:mode|theme)\b")

_pat("preference", "timezone",
     r"\b(?:my|i'?m? in)\s+(?:time\s*zone|tz)\s+(?:is\s+)?"
     r"([A-Z]{2,5}|[+-]\d{1,2}(?::\d{2})?|UTC[+-]\d+)\b")

_pat("preference", "voice_style",
     r"\b(?:keep|make|use|prefer)\s+(?:your|the|my)?\s*"
     r"(?:responses|replies|answers|voice)\s+"
     r"(short|brief|terse|long|detailed|verbose|casual|formal)\b")

# Numeric defaults: "set my default volume to 60", "always start at 50%".
_pat("preference", "default_volume",
     r"\b(?:my\s+)?default\s+volume\s+(?:is|should be|=)\s+(\d{1,3})\b")


# ---- location --------------------------------------------------------

_pat("location", "home",
     r"\b(?i:i)\s+(?i:live\s+in)\s+"
     r"([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,3})\b",
     case_sensitive=True)

_pat("location", "current",
     r"\b(?i:i'?m\s+(?:in|at|located in))\s+"
     r"([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,3})"
     r"\s+(?i:right now|currently|at the moment|today)\b",
     case_sensitive=True)


# ---- pronouns / names -------------------------------------------------

_pat("preference", "pronouns",
     r"\b(?:my\s+pronouns\s+are\s+|use\s+|prefer\s+)"
     r"(he/him|she/her|they/them|he/they|she/they|xe/xem)\b")

# Name capture: ONLY the explicit "my name is" / "call me" phrases.
# The ambiguous "I'm ..." branch was removed because it false-positives
# on adjectives/participles ("I'm Sorry" → name=Sorry) and the
# confidence was high enough to persist (F-002 audit). Anchor end of
# capture to whitespace/punctuation/EOS to avoid eating trailing words.
_pat("user", "name",
     r"\b(?i:my\s+name\s+is|call\s+me)\s+([A-Z][a-zA-Z]{1,30})"
     r"(?=[\s.!,?;:]|$)",
     case_sensitive=True)


# ---- schedules -------------------------------------------------------

_pat("schedule", "workday_start",
     r"\b(?:my\s+)?(?:workday|work day|morning|day)\s+starts?\s+at\s+"
     r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b")

_pat("schedule", "sleep",
     r"\bI\s+(?:sleep|go to bed)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b")

_pat("schedule", "birthday",
     r"\bmy\s+birthday\s+is\s+(?:on\s+)?"
     r"((?:\d{1,2}(?:st|nd|rd|th)?\s+)?"
     r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
     r"[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?"
     r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b")


# ---- contacts --------------------------------------------------------

_NAME_EMAIL_RE = re.compile(
    r"\b([A-Z][a-zA-Z]{1,30})(?:'s)?\s+email\s+is\s+"
    r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b",
    re.IGNORECASE,
)
_NAME_AT_EMAIL_RE = re.compile(
    r"\b([A-Z][a-zA-Z]{1,30})\s+(?:is\s+)?at\s+"
    r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b"
)


# ---- dislikes / stop signals -----------------------------------------

_DISLIKE_RE = re.compile(
    r"\b(?:stop|don'?t|please don'?t|never|no more)\s+"
    r"(?:doing|saying|using|reading|playing)\s+"
    r"([a-zA-Z][\w\s]{2,40})\b",
    re.IGNORECASE,
)


# ---- main extractor --------------------------------------------------

def extract_implicit_facts(*, user_text: str,
                           assistant_text: str = "",
                           tool_results: Optional[List[Dict[str, Any]]] = None
                           ) -> List[FactCandidate]:
    """Return facts implied by the conversation, pre-filtered by
    confidence. Empty list when nothing reliable is in scope."""
    out: List[FactCandidate] = []
    text = (user_text or "").strip()
    if not text and not assistant_text:
        return out

    # 1. Run registered patterns over the user's text.
    for kind, key, pat in _PATTERNS:
        for m in pat.finditer(text):
            try:
                value = _canonical_value(kind, m.group(1))
            except IndexError:
                continue
            if not value:
                continue
            # F-002: name extractor must reject English participles
            # / adjectives ("Sorry", "Going", etc.) that look like
            # capitalized words but obviously aren't names.
            if kind == "user" and key.lower() == "name":
                if not _looks_like_a_name(value):
                    continue
            out.append(FactCandidate(
                kind=kind,
                key=key.lower(),
                value=value,
                confidence=_pattern_confidence(kind, m),
                source="user said: " + _trim(m.group(0)),
                source_kind="implicit_pattern",
                raw_span=m.group(0),
            ))

    # 2. Name → email patterns.
    for pat in (_NAME_EMAIL_RE, _NAME_AT_EMAIL_RE):
        for m in pat.finditer(text):
            name = m.group(1).strip()
            email = m.group(2).strip().lower()
            if not _looks_like_a_name(name):
                continue
            out.append(FactCandidate(
                kind="person",
                key=name.lower(),
                value=email,
                confidence=0.9,
                source=f"user said: {_trim(m.group(0))}",
                source_kind="implicit_pattern",
                raw_span=m.group(0),
            ))

    # 3. Dislikes.
    for m in _DISLIKE_RE.finditer(text):
        thing = m.group(1).strip().rstrip(".!,")
        if not thing or len(thing.split()) > 6:
            continue
        out.append(FactCandidate(
            kind="dislike",
            key=thing.lower(),
            value="user dislikes",
            confidence=0.75,
            source=f"user said: {_trim(m.group(0))}",
            source_kind="implicit_pattern",
            raw_span=m.group(0),
        ))

    # 4. Optional LLM extraction (env-gated to keep cost flat).
    if (os.environ.get("TOUCHLESS_IMPLICIT_LLM", "0") == "1"
            and (assistant_text or tool_results)):
        out.extend(_llm_extract(user_text, assistant_text, tool_results))

    return [c for c in out if c.confidence >= MIN_CONFIDENCE]


# ---- helpers ---------------------------------------------------------

def _trim(s: str, n: int = 80) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def _canonical_value(kind: str, raw: str) -> str:
    v = (raw or "").strip()
    if kind == "preference":
        v = v.lower()
        # Map common email-provider words to the actual tool name so
        # the planner's `default_send_via` preference can read it
        # without mapping again.
        m = {"gmail": "gmail_send", "outlook": "ms_mail_send",
             "teams": "teams_send"}
        if v in m:
            return m[v]
    if kind == "location":
        # Strip trailing words that often hitch a ride on the regex.
        v = re.sub(r"\s+(?:right now|currently|at the moment|today)$",
                   "", v, flags=re.IGNORECASE).strip(" ,.;")
    if kind == "schedule":
        v = v.lower().replace(" ", "")
    return v


def _pattern_confidence(kind: str, m: "re.Match") -> float:
    """Static confidence per kind. Subjects close to natural language
    are scored higher than schedule/birthday which can collide with
    other text."""
    base = {
        "preference": 0.9,
        "location": 0.85,
        "schedule": 0.75,
        "user": 0.85,
        "person": 0.9,
        "dislike": 0.75,
    }
    return base.get(kind, 0.7)


def _looks_like_a_name(s: str) -> bool:
    if not s:
        return False
    # Reject common nouns the regex would otherwise grab ("My email is..")
    # AND common English participles/adjectives that follow "I'm" and
    # would otherwise be misread as a personal name (F-002 audit:
    # "I'm Sorry" → user.name = "Sorry"). The "I'm" branch of the
    # name regex was removed too, but this stoplist is still the
    # second line of defense for the explicit "call me X" / "my name
    # is X" branches where the user really does say a participle as
    # a nickname (rare but possible).
    blacklist = {
        # Determiners + pronouns.
        "my", "your", "their", "the", "a", "an", "this",
        "that", "his", "her", "our", "us", "we", "you",
        # Common copular-completion adjectives/participles.
        "sorry", "sure", "fine", "ok", "okay", "busy", "late",
        "early", "here", "there", "hungry", "tired", "done",
        "ready", "glad", "happy", "sad", "bored", "scared",
        "going", "coming", "leaving", "trying", "waiting",
        "back", "home", "in", "out", "on", "off", "up", "down",
        "right", "wrong", "good", "bad", "great", "fine",
        "thinking", "wondering", "asking", "saying",
    }
    return s.lower() not in blacklist and s[0].isupper()


def _llm_extract(user_text: str, assistant_text: str,
                 tool_results: Optional[List[Dict[str, Any]]]
                 ) -> List[FactCandidate]:
    """Stub for env-gated LLM extraction. Returns []; the wiring to
    the cheap-LLM planner lives in memory.manager. Kept here so the
    interface is stable and tests can monkeypatch."""
    return []

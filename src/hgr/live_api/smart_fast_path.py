"""Smart fast-path classifier — bypass the planner for trivially
classifiable utterances.

Phase-6 latency push. Every voice utterance currently goes through:
  audio → ASR → planner LLM → (maybe) synthesizer LLM → tools

For a large fraction of utterances ("what time is it", "pause music",
"hi iris"), the planner round-trip is wasted overhead — a single
hand-written rule + the tool call gets us the answer in ~50ms
instead of ~600-1200ms.

This module exposes `classify(text) -> FastPathResult`. Result is
one of:
  * `DIRECT`  — call a specific tool with these args (skip planner)
  * `CHAT`    — small-talk reply with a static phrase (skip planner)
  * `IGNORE`  — empty / accidental wake (skip everything)
  * `PLANNER` — too complex, run the full pipeline

The orchestrator consults this BEFORE calling the planner. When the
result is anything other than PLANNER, it short-circuits.

Hard rules:
  * Conservative-by-default: when in doubt → PLANNER. A wrong
    fast-path classification is worse than the latency it saves.
  * No user-data leakage: classifier doesn't see memory, screen,
    or convo — only the bare utterance + flat lookups.
  * Bypasses are LOGGED so we can spot drift (a fast-path that's
    wrong 5% of the time is worse than no fast-path).

Author: Konstantin Markov
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class FastPathKind(str, Enum):
    DIRECT = "direct"
    CHAT = "chat"
    IGNORE = "ignore"
    PLANNER = "planner"


@dataclass
class FastPathResult:
    kind: FastPathKind
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    reply: str = ""
    rule: str = ""
    confidence: float = 0.0

    def is_fast(self) -> bool:
        return self.kind != FastPathKind.PLANNER


_GREETING_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|hi there|hey iris|hi iris|"
    r"hello iris|good morning|good evening|good afternoon)"
    r"[\s.!?]*$",
    re.IGNORECASE,
)

_PAUSE_RE = re.compile(
    r"^\s*(pause|stop|stop music|stop the music|"
    r"pause music|pause the music)[\s.?!]*$",
    re.IGNORECASE,
)

# NOTE: only bare "play" / "resume" match. Phrases like "play music"
# or "play <song name>" or "play <artist>" deliberately fall through
# to the planner so it can search Spotify / queue the right track
# rather than just sending the global play/pause toggle key.
_RESUME_RE = re.compile(
    r"^\s*(resume|play)[\s.?!]*$",
    re.IGNORECASE,
)

_SKIP_RE = re.compile(
    r"^\s*(skip|next|next track|next song|skip song|skip track)"
    r"[\s.?!]*$",
    re.IGNORECASE,
)

_PREV_RE = re.compile(
    r"^\s*(previous|back|previous track|previous song|"
    r"go back|last song)[\s.?!]*$",
    re.IGNORECASE,
)

_MUTE_RE = re.compile(
    r"^\s*(mute|mute it|silence)[\s.?!]*$",
    re.IGNORECASE,
)

_UNMUTE_RE = re.compile(
    r"^\s*(unmute|unmute it)[\s.?!]*$",
    re.IGNORECASE,
)

_ACK_RE = re.compile(
    r"^\s*(thanks|thank you|thx|cheers|got it|cool|nice|"
    r"okay|ok|alright|nice one)[\s.?!]*$",
    re.IGNORECASE,
)

_CANCEL_RE = re.compile(
    r"^\s*(cancel|nevermind|never mind|forget it|stop that|"
    r"abort|cancel that)[\s.?!]*$",
    re.IGNORECASE,
)


def _is_too_complex(text: str) -> bool:
    """Heuristics to stay OUT of fast-path when the utterance has
    structure the rules won't get right."""
    t = text.lower()
    # Multi-clause: 'and then', 'after that', 'also', 'but'.
    for needle in (" and then ", " after that ", " also ",
                   " but ", " then ", " followed by "):
        if needle in t:
            return True
    # Question requiring reasoning: 'why', 'how come', 'explain'.
    if re.search(r"\b(why|how come|explain|describe|"
                 r"summarize|recommend)\b", t):
        return True
    # Memory / personalization references.
    if re.search(r"\b(remember|forget|i said|earlier|yesterday|"
                 r"my)\b", t):
        return True
    # References to entities the resolver should handle. NOTE: "it"
    # is intentionally NOT here — it's idiomatic in many fast-path
    # commands ("turn it up", "what time is it"). Only catch the
    # unambiguous referent forms.
    if re.search(r"\b(him|her|them|that one|this one|the one)\b",
                 t):
        return True
    # Counts/lists with specifics.
    if re.search(r"\b(top \d+|first \d+|last \d+|how many)\b", t):
        return True
    return False


# Ordered (rule_name, regex, builder).  Earliest match wins.
def _build_chat(reply, rule):
    def b(_m):
        return FastPathResult(kind=FastPathKind.CHAT, reply=reply,
                              rule=rule, confidence=0.95)
    return b


def _build_direct(tool, args, rule):
    def b(_m):
        return FastPathResult(kind=FastPathKind.DIRECT, tool=tool,
                              args=dict(args), rule=rule,
                              confidence=0.9)
    return b


# Rules use REAL tool names from the connectors:
#   media: media_play_pause / media_next_track / media_previous_track
#   volume: volume_toggle_mute (no direct up/down — needs LLM for %)
# Time/date have NO underlying tool — the planner injects current
# time via the system prompt; fast-pathing them would 404. So they
# fall through to the LLM.
_RULES = (
    ("greeting", _GREETING_RE,
     _build_chat("Hey, what's up?", "greeting")),
    ("ack",      _ACK_RE,
     _build_chat("Anytime.", "ack")),
    ("cancel",   _CANCEL_RE,
     _build_chat("Okay, cancelled.", "cancel")),
    # Single play/pause toggle — both 'pause' and 'play' route to it.
    ("pause",    _PAUSE_RE,
     _build_direct("media_play_pause", {}, "pause")),
    ("resume",   _RESUME_RE,
     _build_direct("media_play_pause", {}, "resume")),
    ("skip",     _SKIP_RE,
     _build_direct("media_next_track", {}, "skip")),
    ("prev",     _PREV_RE,
     _build_direct("media_previous_track", {}, "prev")),
    ("mute",     _MUTE_RE,
     _build_direct("volume_toggle_mute", {}, "mute")),
    ("unmute",   _UNMUTE_RE,
     _build_direct("volume_toggle_mute", {}, "unmute")),
)


@dataclass
class FastPathStats:
    total: int = 0
    direct: int = 0
    chat: int = 0
    ignore: int = 0
    planner: int = 0
    last_at: float = 0.0


_GLOBAL_STATS = FastPathStats()


def classify(text: str) -> FastPathResult:
    """Top-level classifier. Returns FastPathResult."""
    stats = _GLOBAL_STATS
    stats.total += 1
    stats.last_at = time.time()
    if text is None:
        stats.ignore += 1
        return FastPathResult(kind=FastPathKind.IGNORE,
                              rule="empty", confidence=1.0)
    stripped = text.strip()
    if not stripped:
        stats.ignore += 1
        return FastPathResult(kind=FastPathKind.IGNORE,
                              rule="empty", confidence=1.0)
    # Anything longer than ~12 words is almost certainly not a
    # one-shot fast-path command.
    if len(stripped.split()) > 12:
        stats.planner += 1
        return FastPathResult(kind=FastPathKind.PLANNER,
                              rule="too_long", confidence=1.0)
    if _is_too_complex(stripped):
        stats.planner += 1
        return FastPathResult(kind=FastPathKind.PLANNER,
                              rule="too_complex", confidence=1.0)
    for rule_name, pattern, builder in _RULES:
        m = pattern.match(stripped)
        if m is not None:
            result = builder(m)
            if result.kind == FastPathKind.DIRECT:
                stats.direct += 1
            elif result.kind == FastPathKind.CHAT:
                stats.chat += 1
            return result
    stats.planner += 1
    return FastPathResult(kind=FastPathKind.PLANNER,
                          rule="no_match", confidence=0.0)


def global_stats() -> FastPathStats:
    return _GLOBAL_STATS


def reset_stats() -> None:
    global _GLOBAL_STATS
    _GLOBAL_STATS = FastPathStats()

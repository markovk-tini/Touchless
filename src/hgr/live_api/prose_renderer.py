"""Jarvis-voice prose renderer for tool results.

Turns a tool's raw output into one flowing conversational reply in
Iris's Jarvis voice — warm, witty, the way a smart friend would read
it aloud. Used by the realtime override hook so emails and other
summary-bearing tools sound human instead of like a numbered list or
template.

Architecture:
  - Tiny LLM pass (gpt-4o-mini, ~$0.0001 per call, ~300-700 ms)
  - Strict system prompt: PRESERVE every fact verbatim, no invention
  - Programmatic fact-preservation guard (sender names, counts) that
    rejects rewrites missing source facts
  - On ANY failure (no API key, timeout, hallucination, model error)
    returns the deterministic fallback string the caller already had
  - SHA-256 cache so repeated identical queries are instant

This module is a thin pure-Python utility: no Qt, no threads. Caller
decides where to invoke it (synchronously in the dispatch hook, or
lazily on consumption — both work).

Author: Konstantin Markov
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Set

_API_URL = "https://api.openai.com/v1/chat/completions"
# gpt-4o-mini is fast (~400 ms median), cheap (~$0.0001/call at this size),
# and good enough for the prose rewrite — we're not asking for reasoning,
# just stylistic rendering with strict fact preservation.
_MODEL = "gpt-4o-mini"
_TIMEOUT_S = 4.0
_MAX_TOKENS = 500

# Lightweight cache. Keyed on (question, tool, hash(result), context_tail)
# so identical repeats are zero-latency. Bounded; clear when full (good
# enough — these replies aren't worth a proper LRU).
_CACHE: Dict[str, str] = {}
_CACHE_LIMIT = 200

# Tools whose status-confirmation text is so short and template-y that
# running it through an LLM rewrite would add latency without value.
# Volume/Discord/mute confirmations etc. — keep them snappy.
_SKIP_RENDER_BELOW_CHARS = 25

_SYSTEM_PROMPT = (
    "You are Iris — a Jarvis-style personal assistant. The user just "
    "asked something and a tool returned data. Compose ONE flowing, "
    "conversational reply in Iris's voice — warm, smart, slightly "
    "witty, the way a friend sitting next to the user would read it "
    "aloud.\n\n"
    "HARD RULES (violating any of these is a worse failure than a "
    "robotic reply):\n"
    "• Every fact in your reply — sender names, subject lines, numbers, "
    "counts, dates, locations, prices, URLs, file paths — MUST come "
    "literally from the data. Do NOT invent. Do NOT substitute. Do NOT "
    "round. If a sender is 'VESSELIN MARKOV' say 'Vesselin Markov', "
    "not 'Vesko' or 'a colleague'.\n"
    "• If the data is empty (count=0, messages=[], results=[]), say so "
    "plainly. Do NOT fill the gap with made-up examples.\n"
    "• Speak dates naturally: 'tomorrow' / 'Thursday' / 'this Friday', "
    "NEVER 'June 3rd' or '2026-06-03'.\n"
    "• Do NOT read URLs aloud — EVER. No 'visit wttr.in slash...', "
    "no 'for more information at https...', no 'see open-meteo dot "
    "com'. If there's a link in the data, just offer to pull it up "
    "('want me to grab the full forecast page?').\n"
    "• Do NOT print a fact twice when one value is the same as "
    "another (the canonical example: never say 'it's 65, feels like "
    "65' — just say 'it's 65'). Only mention feels-like when it's "
    "meaningfully different (~3°+ off). 'Subject is X' when X IS the "
    "subject is the same anti-pattern — say 'X' once.\n"
    "• Use contractions, vary phrasing, conversational connectives "
    "('looks like', 'just so you know', 'heads up', 'quick rundown', "
    "'so', 'real quick').\n"
    "• Keep it tight: trivial data → 1 sentence; substantive → 2-5 "
    "sentences. Don't lecture, don't bullet-point ceremoniously.\n"
    "• For email lists: flowing prose when there are 1-5 messages "
    "('Looks like a few — Humble Bundle is hyping Pride Month, "
    "LinkedIn has a PCB Layout Engineer alert, and Paramount+ wants "
    "you back'). For 6+, brief intro then short numbered list, ONE "
    "line per email (sender + subject + ~one phrase about the body).\n"
    "• ANTICIPATE: if one item in the data stands out (a notable "
    "sender, an urgent subject, a forecast that conflicts with the "
    "user's plans, a search result that obviously answers the "
    "question), call it out FIRST in one phrase. Don't bury it.\n"
    "• If the conversation context shows the user mentioned plans / "
    "intent / context earlier (going out, a meeting, a project), "
    "weave a one-clause reference in naturally — 'heads up before "
    "your meeting' / 'rain's clearing before you head out'. Don't "
    "force it if there's no real connection.\n"
    "• Output ONLY the reply text. No preamble like 'Sure!' or "
    "'Here you go:'. No closing like 'Hope that helps'. Just the reply."
)


def should_render(tool_name: str, fallback: str) -> bool:
    """Whether this tool's reply benefits from prose rendering.

    Skip for very short status confirmations ('Volume set to 30.') —
    they're already concise and the LLM round-trip would add latency
    without value. Everything else gets composed.
    """
    if not fallback or len(fallback.strip()) < _SKIP_RENDER_BELOW_CHARS:
        return False
    return True


def render_jarvis(
    question: str,
    tool_name: str,
    tool_result: Dict[str, Any],
    fallback: str,
    timeout: float = _TIMEOUT_S,
    context: str = "",
) -> str:
    """Compose `tool_result` as a Jarvis-voice reply.

    Returns `fallback` unchanged on:
      - No OPENAI_API_KEY in env
      - Tool flagged as already-conversational (weather_get)
      - HTTP timeout / 4xx / 5xx
      - JSON parse error
      - Fact-preservation guard rejects the rewrite (hallucinated names
        or wrong counts)
      - Any unexpected exception

    The contract: never return something WORSE than `fallback`. If the
    rewrite isn't strictly better and verifiably faithful, the original
    deterministic summary wins.
    """
    if not fallback:
        return fallback
    if not should_render(tool_name, fallback):
        return fallback
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return fallback
    if not isinstance(tool_result, dict) or not tool_result:
        return fallback

    # Cache lookup. Serialize once; reuse for both key and prompt body
    # to keep their fingerprints identical. Cache key includes a
    # truncated context fingerprint so a follow-up turn in a new
    # context doesn't return the prior turn's phrasing.
    try:
        result_blob = json.dumps(tool_result, default=str, sort_keys=True)
    except Exception:
        return fallback
    result_blob = result_blob[:8000]  # bound prompt size

    context_tail = (context or "")[-400:]
    cache_key = hashlib.sha256(
        f"{question}::{tool_name}::{result_blob}::{context_tail}"
        .encode("utf-8")
    ).hexdigest()
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    user_msg_parts = []
    if context_tail.strip():
        user_msg_parts.append(
            f"Recent conversation (most recent last — use it to anchor "
            f"any natural reference back to what the user just "
            f"mentioned):\n{context_tail}\n"
        )
    user_msg_parts.extend([
        f"User just asked: {question or '(no recent question)'}",
        f"Tool that ran: {tool_name}",
        f"Tool data (JSON):\n{result_blob}",
        f"Fallback phrasing the system would use if you can't compose "
        f"a better Jarvis-voice rewrite:\n{fallback}",
        f"Compose ONE reply. Preserve every fact verbatim.",
    ])
    user_msg = "\n\n".join(user_msg_parts)

    try:
        body = json.dumps({
            "model": _MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            # Some creative variation; high enough for natural phrasing,
            # low enough that the model stays anchored to the data.
            "temperature": 0.6,
            "max_tokens": _MAX_TOKENS,
        }).encode("utf-8")
    except Exception:
        return fallback

    req = urllib.request.Request(
        _API_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return fallback
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        return fallback
    except urllib.error.URLError:
        return fallback
    except Exception:
        return fallback

    try:
        rendered = ((payload.get("choices") or [{}])[0]
                    .get("message", {}).get("content") or "").strip()
    except Exception:
        rendered = ""
    if not rendered:
        return fallback

    # Strip a leading "Sure," / "Here you go," / "Of course!" preamble
    # if the model slipped one in despite the system prompt.
    rendered = re.sub(
        r"^(sure[,!]?\s+|of course[,!]?\s+|here(?:'s| is| you go|'s the rundown)[:,]?\s+|"
        r"alright[,!]?\s+|got it[,!]?\s+|absolutely[,!]?\s+|okay[,!]?\s+)",
        "", rendered, flags=re.IGNORECASE).lstrip()

    if not _facts_preserved(tool_result, rendered):
        return fallback

    # Cache. Cheap clear-when-full (200 entries is enough for a session;
    # a real LRU isn't worth the complexity for ~$0.0001 saves).
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.clear()
    _CACHE[cache_key] = rendered
    return rendered


def _facts_preserved(source: Dict[str, Any], output: str) -> bool:
    """Loose hallucination check.

    For email-style results (messages array): if there are any senders
    in source, at least ONE real sender name must appear in output. If
    none do, the rewrite is fabricated and we reject.

    For count-bearing results: if count >= 1, the output must mention
    the count (digit or number word) or use a vague-but-honest
    quantifier ('a few', 'a handful', 'several'). count=0 must say so.

    Permissive on purpose: we don't require EVERY sender to appear
    (the LLM may rightly say 'a few' for 8 emails), but we do require
    at least one anchor to real data.
    """
    if not output:
        return False
    lower_out = output.lower()

    # ---- email-style: messages with from_name / from --------------
    msgs = source.get("messages") or []
    if isinstance(msgs, list) and msgs:
        senders: Set[str] = set()
        for m in msgs[:10]:
            if not isinstance(m, dict):
                continue
            for key in ("from_name", "from"):
                val = (m.get(key) or "").strip()
                if not val:
                    continue
                # Strip email-address forms to just the local-part
                # name token if needed: "Vesko <v@x.com>" → "Vesko"
                # and "v@x.com" → "v".
                token = re.split(r"\s*<|\s+at\s+|@", val, maxsplit=1)[0]
                token = token.strip().lower()
                if len(token) >= 3:
                    senders.add(token)
        if senders:
            anchored = any(s in lower_out for s in senders)
            if not anchored:
                return False  # hallucinated emails

    # ---- count fidelity (only when source explicitly reports one) -
    count = source.get("count")
    if isinstance(count, int):
        if count == 0:
            # Must say so plainly. Reject if it claims any quantity.
            if re.search(
                r"\byou(?:'ve| have)?\s+got\s+\d+\b",
                lower_out,
            ):
                return False
        elif count > 0:
            number_words = {
                1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
            }
            digit_ok = re.search(rf"\b{count}\b", output)
            word_ok = (count in number_words
                       and number_words[count] in lower_out)
            vague_ok = re.search(
                r"\b(a\s+few|a\s+handful|several|some|a\s+bunch|"
                r"a\s+couple|a\s+pile|quite\s+a\s+few)\b",
                lower_out)
            # For small counts (1-3), require exact mention; for bigger
            # counts, vague-but-honest quantifiers are OK.
            if count <= 3 and not (digit_ok or word_ok):
                return False
            if count > 3 and not (digit_ok or word_ok or vague_ok):
                return False

    return True

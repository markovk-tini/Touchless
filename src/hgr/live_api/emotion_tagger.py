"""Per-reply emotion classifier + TTS delivery instruction builder.

Phase-7 polish (JARVIS-grade voice). The persona system gives Iris
a CHARACTER (Jarvis butler, playful friend, etc.) — one voice ID +
one delivery instruction set per session. What it doesn't do is
adapt the SPOKEN TONE to what's actually being said. A real
person reading "build failed" sounds different from the same
person reading "all green, ready to ship!" — the words convey
content; the tone conveys meaning.

This module bridges that gap. For each reply about to be spoken,
we classify the emotion (neutral / happy / apologetic / urgent /
curious / playful / concerned / excited / thoughtful / dry), then
return a delivery hint to merge with the active persona's base
instructions. The combined instruction goes to gpt-4o-mini-tts'
`instructions` field per call.

Two classifier backends:
  1. LLM (gpt-4o-mini, JSON mode, ~150-400 ms): high accuracy,
     handles context, sarcasm, mixed signals.
  2. Heuristic (regex/keyword, ~1 ms): no cost, no latency, ~70%
     accurate on the easy cases.

Strategy: LLM first when OPENAI_API_KEY is set + not in slow mode;
heuristic fallback on any failure (no key, timeout, slow mode,
parse error). SHA-256 cache keyed on (text, persona) so the same
reply doesn't re-classify.

The delivery hints are SHORT (60-100 chars each) and tuned to
combine cleanly with persona base instructions — they amend,
not override.

Author: Konstantin Markov
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


_API_URL = "https://api.openai.com/v1/chat/completions"
_MODEL = "gpt-4o-mini"
_TIMEOUT_S = 1.5
_MAX_TOKENS = 60


class Emotion(str, Enum):
    NEUTRAL    = "neutral"      # informative, baseline
    HAPPY      = "happy"        # pleased, celebratory, success
    APOLOGETIC = "apologetic"   # error, can't do, sorry
    URGENT     = "urgent"       # warning, must-act, time-sensitive
    CURIOUS    = "curious"      # question back, clarifying ask
    PLAYFUL    = "playful"      # joke, riff, light moment
    CONCERNED  = "concerned"    # bad news that isn't urgent
    EXCITED    = "excited"      # big positive moment
    THOUGHTFUL = "thoughtful"   # explanatory, reflective
    DRY        = "dry"          # deadpan, sardonic, witty


@dataclass
class EmotionTag:
    emotion: Emotion
    confidence: float = 0.0
    source: str = ""            # "llm" | "heuristic" | "fallback"
    raw_reason: str = ""


# ---- per-emotion delivery hints (amend persona base instructions) ----

# These hints are MERGED with the persona's base instructions and
# sent to gpt-4o-mini-tts as the `instructions` parameter. Per
# OpenAI's TTS documentation + community findings, the model
# responds FAR better to REFERENCE-BASED language ("speak like a
# friend delivering bad news") than to acoustic abstractions
# ("drop pitch a hair"). Every hint here grounds the tone in a
# recognizable real-world delivery scenario.
_DELIVERY_HINTS: Dict[Emotion, str] = {
    Emotion.NEUTRAL: (
        "For this reply: speak matter-of-factly, like a news "
        "anchor reading routine information — calm, clear, "
        "neither rushed nor lingering."),
    Emotion.HAPPY: (
        "For this reply: speak like someone delivering good news "
        "to a friend — pleased, warm, smiling, lifted. "
        "Genuinely happy without overselling."),
    Emotion.APOLOGETIC: (
        "For this reply: speak like a friend gently delivering "
        "bad news — softer, slower, sincere. Wishing you had "
        "better news to share. Never stiff or formal."),
    Emotion.URGENT: (
        "For this reply: speak like a paramedic giving "
        "clear instructions — alert, focused, every word landing. "
        "Authority without panic."),
    Emotion.CURIOUS: (
        "For this reply: speak like a scientist asking a "
        "clarifying question — inquisitive, engaged, interested "
        "in the answer. Rising at the end."),
    Emotion.PLAYFUL: (
        "For this reply: speak like a podcast co-host riffing on "
        "a fun moment — bright, lively, leaning into the joke. "
        "Quick aside-y delivery on parentheticals. Suggest a smile."),
    Emotion.CONCERNED: (
        "For this reply: speak like a doctor sharing a serious "
        "but not-emergency finding — measured, sincere, slightly "
        "slower. Real without being alarming."),
    Emotion.EXCITED: (
        "For this reply: speak like a sports commentator calling "
        "a winning goal — energetic, punchy, celebratory. Energy, "
        "not volume."),
    Emotion.THOUGHTFUL: (
        "For this reply: speak like a mentor explaining a "
        "complex idea — measured, deliberate, pausing briefly "
        "before key ideas. Inviting reflection."),
    Emotion.DRY: (
        "For this reply: speak with deadpan understatement, like "
        "a British comedian delivering a flat observation — "
        "minimal variation, the humor in the dryness, brief beat "
        "before any punch line. No smile in the voice."),
}


# Persona-specific emotion biases. Some personas favor certain
# emotions: Jarvis trends DRY/NEUTRAL even when the content is
# happy; Playful trends PLAYFUL/EXCITED; Concise stays NEUTRAL.
# When the classifier picks an emotion the persona would never
# express, we snap to the closest persona-appropriate one.
_PERSONA_EMOTION_ALLOWLIST: Dict[str, set] = {
    "default": {Emotion.NEUTRAL, Emotion.HAPPY, Emotion.APOLOGETIC,
                Emotion.URGENT, Emotion.CURIOUS, Emotion.PLAYFUL,
                Emotion.CONCERNED, Emotion.EXCITED,
                Emotion.THOUGHTFUL},
    "jarvis":  {Emotion.NEUTRAL, Emotion.DRY, Emotion.APOLOGETIC,
                Emotion.URGENT, Emotion.CONCERNED,
                Emotion.THOUGHTFUL, Emotion.CURIOUS},
    "concise": {Emotion.NEUTRAL, Emotion.URGENT, Emotion.APOLOGETIC},
    "warm":    {Emotion.NEUTRAL, Emotion.HAPPY, Emotion.APOLOGETIC,
                Emotion.CURIOUS, Emotion.CONCERNED,
                Emotion.EXCITED, Emotion.THOUGHTFUL},
    "playful": {Emotion.PLAYFUL, Emotion.EXCITED, Emotion.HAPPY,
                Emotion.NEUTRAL, Emotion.DRY, Emotion.CURIOUS,
                Emotion.APOLOGETIC},
    "tutor":   {Emotion.NEUTRAL, Emotion.THOUGHTFUL,
                Emotion.HAPPY, Emotion.APOLOGETIC, Emotion.CURIOUS},
}


def _snap_to_persona(emotion: Emotion, persona: str) -> Emotion:
    """Snap a classified emotion to the closest one the persona
    would actually express. Concise + EXCITED → NEUTRAL. Jarvis +
    PLAYFUL → DRY. Etc."""
    allow = _PERSONA_EMOTION_ALLOWLIST.get(
        persona, _PERSONA_EMOTION_ALLOWLIST["default"])
    if emotion in allow:
        return emotion
    # Snap rules (persona-aware tonal mapping).
    if persona == "jarvis":
        if emotion in (Emotion.PLAYFUL, Emotion.HAPPY,
                        Emotion.EXCITED):
            return Emotion.DRY
        return Emotion.NEUTRAL
    if persona == "concise":
        if emotion in (Emotion.URGENT, Emotion.APOLOGETIC):
            return emotion
        return Emotion.NEUTRAL
    if persona == "playful":
        if emotion in (Emotion.CONCERNED, Emotion.THOUGHTFUL,
                        Emotion.URGENT):
            return Emotion.NEUTRAL  # playful softens to neutral
        return Emotion.PLAYFUL
    if persona == "warm":
        if emotion == Emotion.DRY:
            return Emotion.NEUTRAL
        return Emotion.HAPPY if emotion == Emotion.PLAYFUL \
            else emotion
    if persona == "tutor":
        return Emotion.THOUGHTFUL
    return Emotion.NEUTRAL


# ---- heuristic classifier (fallback) --------------------------------

_HEURISTIC_PATTERNS = [
    # (emotion, regex)
    (Emotion.APOLOGETIC, re.compile(
        r"\b(sorry|couldn'?t|can'?t|unable|failed|error|"
        r"didn'?t work|wasn'?t able|no luck|nothing matched)\b",
        re.IGNORECASE)),
    (Emotion.URGENT, re.compile(
        r"\b(urgent|critical|immediately|now|warning|caution|"
        r"about to|deadline|expires|expiring|battery low|"
        r"running out|stopped working)\b", re.IGNORECASE)),
    (Emotion.HAPPY, re.compile(
        r"\b(done|all set|got it|perfect|nice|great|love it|"
        r"all green|shipping|ready)\b|✓|✅", re.IGNORECASE)),
    (Emotion.EXCITED, re.compile(
        r"\b(awesome|incredible|huge|massive|finally|"
        r"absolutely|wow)\b|!{2,}", re.IGNORECASE)),
    (Emotion.PLAYFUL, re.compile(
        r"\b(haha|lol|joke|riff|kidding|cheeky|silly|"
        r"buckle up|enjoy)\b|—.*—", re.IGNORECASE)),
    (Emotion.CONCERNED, re.compile(
        r"\b(heads up|fyi|might want|consider|"
        r"may want to|something to watch)\b", re.IGNORECASE)),
    (Emotion.CURIOUS, re.compile(r"\?\s*$")),
    (Emotion.THOUGHTFUL, re.compile(
        r"\b(let me|let's see|interesting|hmm|"
        r"thinking about|consider that)\b", re.IGNORECASE)),
    (Emotion.DRY, re.compile(
        r"\b(of course|naturally|as expected|"
        r"unsurprisingly|predictably)\b", re.IGNORECASE)),
]


def _classify_heuristic(text: str) -> EmotionTag:
    """Cheap regex-based classifier. Returns NEUTRAL when no
    pattern matches."""
    if not text:
        return EmotionTag(Emotion.NEUTRAL, confidence=1.0,
                           source="heuristic", raw_reason="empty")
    # Run all patterns; the FIRST matching emotion wins (priority
    # order in the list). Confidence 0.7 for any heuristic hit.
    for emo, pat in _HEURISTIC_PATTERNS:
        if pat.search(text):
            return EmotionTag(
                emotion=emo, confidence=0.7,
                source="heuristic",
                raw_reason=f"regex match for {emo.value}")
    return EmotionTag(Emotion.NEUTRAL, confidence=0.6,
                       source="heuristic",
                       raw_reason="no pattern matched")


# ---- LLM classifier --------------------------------------------------

_SYSTEM_PROMPT = (
    "You classify the emotional tone of short replies that a "
    "voice assistant is about to speak aloud. Output STRICT JSON: "
    "{\"emotion\": <one of: neutral, happy, apologetic, urgent, "
    "curious, playful, concerned, excited, thoughtful, dry>, "
    "\"confidence\": <0.0-1.0>, \"reason\": \"<5-10 word "
    "justification>\"}. Pick the SINGLE dominant tone. "
    "Default to 'neutral' for plain informative output. Use "
    "'dry' for deadpan/sardonic/understatement, NOT for "
    "neutral facts. Use 'concerned' for soft bad news, "
    "'apologetic' for clear failures/errors. Use 'curious' "
    "when the reply itself asks the user a question. Use "
    "'urgent' for time-sensitive warnings only — not for "
    "ordinary action confirmations. Output JSON only, no "
    "preamble."
)


def _classify_llm(text: str, *,
                  api_key: str,
                  timeout: float = _TIMEOUT_S
                  ) -> Optional[EmotionTag]:
    """LLM call. Returns EmotionTag on success, None on any
    failure (caller falls back to heuristic)."""
    try:
        body = json.dumps({
            "model": _MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Reply text:\n{text}"},
            ],
            "temperature": 0.0,
            "max_tokens": _MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
    except Exception:
        return None
    req = urllib.request.Request(
        _API_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        return None
    except urllib.error.URLError:
        return None
    except Exception:
        return None
    try:
        content = ((payload.get("choices") or [{}])[0]
                   .get("message", {})
                   .get("content") or "")
    except Exception:
        return None
    if not content:
        return None
    try:
        parsed = json.loads(content)
    except Exception:
        return None
    raw = str(parsed.get("emotion") or "").strip().lower()
    try:
        emo = Emotion(raw)
    except ValueError:
        return None
    conf = parsed.get("confidence")
    try:
        conf_f = float(conf)
    except (TypeError, ValueError):
        conf_f = 0.7
    reason = str(parsed.get("reason") or "")[:120]
    return EmotionTag(emotion=emo, confidence=conf_f,
                       source="llm", raw_reason=reason)


# ---- cache + top-level -----------------------------------------------

_CACHE_LIMIT = 200
_cache_lock = threading.RLock()
_cache: Dict[str, EmotionTag] = {}


def _cache_key(text: str, persona: str) -> str:
    blob = f"{persona}|{text}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def classify(text: str, *,
             persona: str = "default",
             allow_llm: bool = True,
             timeout: float = _TIMEOUT_S
             ) -> EmotionTag:
    """Top-level. Returns an EmotionTag — never raises. LLM-first
    when an API key is available + cost-meter isn't in slow mode;
    heuristic fallback otherwise.

    The returned emotion is snapped to one the active persona
    would actually express (Jarvis + PLAYFUL → DRY, etc.)."""
    text = (text or "").strip()
    if not text:
        return EmotionTag(Emotion.NEUTRAL, source="empty")
    persona = (persona or "default").strip().lower() or "default"
    key = _cache_key(text, persona)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
    tag: Optional[EmotionTag] = None
    # Cost / cap gates — skip LLM in slow mode.
    if allow_llm:
        try:
            from .cost_meter import global_meter
            m = global_meter()
            if getattr(m, "is_slow_mode", lambda: False)():
                allow_llm = False
        except Exception:
            pass
    if allow_llm:
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if api_key:
            tag = _classify_llm(text, api_key=api_key,
                                 timeout=timeout)
    if tag is None:
        tag = _classify_heuristic(text)
    # Snap to persona-allowed emotion.
    final_emo = _snap_to_persona(tag.emotion, persona)
    if final_emo != tag.emotion:
        tag = EmotionTag(
            emotion=final_emo, confidence=tag.confidence,
            source=tag.source + "+snap",
            raw_reason=(f"snapped {tag.emotion.value}->"
                        f"{final_emo.value} for {persona}"))
    with _cache_lock:
        if len(_cache) >= _CACHE_LIMIT:
            _cache.clear()
        _cache[key] = tag
    return tag


def delivery_hint(emotion: Emotion) -> str:
    return _DELIVERY_HINTS.get(emotion, _DELIVERY_HINTS[Emotion.NEUTRAL])


def combine_instructions(persona_base: str, emotion: Emotion) -> str:
    """Merge persona's base instructions with the emotion-specific
    delivery hint. The persona owns the CHARACTER; the emotion owns
    the moment-to-moment TONE."""
    base = (persona_base or "").strip()
    hint = delivery_hint(emotion).strip()
    if not base:
        return hint
    if not hint:
        return base
    return f"{base}\n\nFor THIS reply: {hint}"


def reset_cache() -> None:
    with _cache_lock:
        _cache.clear()


def cache_size() -> int:
    with _cache_lock:
        return len(_cache)

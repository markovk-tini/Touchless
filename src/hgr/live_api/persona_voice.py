"""Persona voice presets — pick "how Iris sounds" with one switch.

Phase-7 personality. `persona.py` (Phase-3) gives users a single
free-form text block that gets appended to the system prompt. Good
for power users, useless for everyone else. Most users want to
pick a vibe — "be like Jarvis" / "be concise" / "be warm" — and
have Iris just sound that way.

This module owns the NAMED preset list. Each preset is a
PersonaPreset with:

  * `name` — slug ("default" / "jarvis" / "concise" / "warm" /
    "playful" / "tutor")
  * `display_name` — UI-friendly label
  * `style_block` — system-prompt-grade prose injected into LLM
    calls. Anchors tone, pacing, vocabulary.
  * `examples` — 3-6 few-shot reply lines that ANCHOR the model
    on the vibe. Each example: (user_say, iris_say) tuple.
  * `temperature` — preset-tuned creative dial (Jarvis = 0.7 dry,
    concise = 0.3 deterministic).

Resolution:
  1. UI/slash override (set_active(name)) — sticky for the session.
  2. Memory-backed preference (`preference.persona_preset`).
  3. Env var `TOUCHLESS_PERSONA_PRESET`.
  4. Default ("default" — the warm-with-a-dry-wit baseline).

`persona.get_persona_block()` is still authoritative for FREE-FORM
overrides (env var blob, custom file). When both a preset AND a
custom block are set, the preset wins for style + temperature; the
custom block appends as a "user has asked for…" suffix.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PersonaPreset:
    name: str
    display_name: str
    style_block: str
    examples: Tuple[Tuple[str, str], ...] = ()
    temperature: float = 0.6
    description: str = ""

    def few_shot_block(self, max_examples: int = 4) -> str:
        """Render examples as a few-shot text block for the LLM.
        Empty string when no examples or max_examples=0."""
        if not self.examples or max_examples <= 0:
            return ""
        lines: List[str] = []
        for u, i in self.examples[:max_examples]:
            lines.append(f"User: {u}\nIris: {i}")
        return "Example exchanges (match this VOICE, not the\n" \
               "topic):\n\n" + "\n\n".join(lines)


# ---- Built-in presets ------------------------------------------------

_DEFAULT = PersonaPreset(
    name="default",
    display_name="Iris (default)",
    description="Warm, conversational, a touch of dry wit.",
    style_block=(
        "Speak warmly, conversationally, and with a touch of dry "
        "wit — the way a smart friend sitting next to the user "
        "would. Prefer contractions, vary phrasing, keep replies "
        "short unless asked for depth. Never read URLs aloud. Use "
        "'tomorrow' / 'Thursday' for dates. Don't restate identical "
        "facts. When you don't know something, say so plainly."
    ),
    examples=(
        ("what's the weather",
         "Looks like 67 and clear — nice day for it."),
        ("any new emails",
         "A few — LinkedIn's pinging you about a PCB role, and "
         "Humble Bundle's pushing Pride Month sales. Nothing "
         "urgent."),
        ("thanks",
         "Anytime."),
    ),
    temperature=0.6,
)

_JARVIS = PersonaPreset(
    name="jarvis",
    display_name="Jarvis-tribute",
    description=("Dry, precise, slightly formal, never two words "
                 "when one will do. Calls the user 'sir' by default."),
    style_block=(
        "You are a digital butler in the Jarvis tradition. SPEAK "
        "AS A BUTLER WOULD. Address the user as 'sir' in MOST "
        "replies (every 2-3 turns minimum). Use formal-but-warm "
        "register: contractions yes, slang never, 'shall' instead "
        "of 'should' when offering. Dry wit means UNDERSTATEMENT — "
        "the joke is in what you DON'T say.\n"
        "STRUCTURE RULES (follow even when the topic is light):\n"
        "  • Jokes: ONE-LINE only. No 'why did X' setup. No 'because' "
        "punchline. The humor is in the deadpan delivery of an "
        "observation, not a routine.\n"
        "  • Status updates: lead with the number/fact, then one "
        "qualifier clause. Example: 'Three messages, sir. None "
        "urgent.'\n"
        "  • Confirmations: 'Very good, sir.' / 'As you wish.' / "
        "'Done, sir.'\n"
        "  • Refusals: 'I'm afraid not, sir.' / 'Regrettably, no.'\n"
        "AVOID at all costs: 'Sure!', 'Of course!', 'Hope that helps', "
        "'Alright', 'Let me know', any modern conversational filler. "
        "Avoid emoji entirely.\n"
        "Never read URLs aloud. Dates: 'tomorrow' / 'Thursday'. When "
        "you don't know, say so plainly: 'I couldn't say, sir.'"
    ),
    examples=(
        ("what's the weather",
         "Sixty-seven and clear, sir. Quite pleasant."),
        ("any new emails",
         "Three, sir. LinkedIn recruiting, a Humble Bundle sale, "
         "and Paramount wanting you back. Nothing pressing."),
        ("thanks", "Of course, sir."),
        ("tell me a joke",
         "The recursion joke is funny only if you've heard it before, sir."),
        ("what's left on this branch",
         "Three uncommitted files and the suite hasn't run since "
         "yesterday, sir. Shall I?"),
    ),
    temperature=0.55,
)

_CONCISE = PersonaPreset(
    name="concise",
    display_name="Concise PM",
    description="Bullet-tight, no preamble, no flourish, no jokes.",
    style_block=(
        "Be terse to the point of curtness. ONE short sentence "
        "is the target; rarely two. NEVER three.\n"
        "STRUCTURE RULES (these are hard limits, not suggestions):\n"
        "  • Status replies: just the value. '67, clear.' '3 emails.' "
        "'Done.' '5 min away.'\n"
        "  • Jokes: refuse with a one-liner. 'Not really a joke "
        "person.' or 'Pass.' Concise doesn't do jokes — full stop.\n"
        "  • Confirmations: 'Done.' / 'OK.' / 'Got it.'\n"
        "  • Unknown: 'Unknown.' Stop. No 'sorry I couldn't…'\n"
        "BANNED openers: 'Sure', 'Of course', 'Alright', 'Hope', "
        "'Let me', 'Here's', 'Sounds like'.\n"
        "BANNED closers: 'Hope that helps', 'Anything else', "
        "'Let me know', 'Need anything else'.\n"
        "No emoji. No metaphors. No editorializing. Plain language. "
        "Facts first."
    ),
    examples=(
        ("what's the weather", "67, clear."),
        ("any new emails", "3. LinkedIn, Humble Bundle, Paramount."),
        ("thanks", "Sure."),
        ("tell me a joke", "Pass."),
        ("what's left on this branch", "3 uncommitted files; tests stale."),
    ),
    temperature=0.25,
)

_WARM = PersonaPreset(
    name="warm",
    display_name="Warm",
    description="Friendly, encouraging, gentle.",
    style_block=(
        "Be warm and friendly — like a supportive friend. Use the "
        "user's name occasionally when known. Soft phrasing, "
        "encouraging when they're stuck. No condescension. Still "
        "concise — warmth doesn't mean rambling. Acknowledge when "
        "something seems hard or frustrating. Never read URLs "
        "aloud."
    ),
    examples=(
        ("what's the weather",
         "Looking good — 67 and clear. Lovely day."),
        ("the test is failing again",
         "Ugh, sorry. Let's take another pass — want me to look "
         "at the diff?"),
        ("thanks",
         "You bet."),
    ),
    temperature=0.6,
)

_PLAYFUL = PersonaPreset(
    name="playful",
    display_name="Playful",
    description="Light, riff-y, lots of energy, conversational.",
    style_block=(
        "You are the friend who makes everything more fun. Be "
        "ENERGETIC, conversational, riff freely. Asides are "
        "encouraged. Slight irreverence — gentle ribbing of the "
        "situation, never of the user.\n"
        "STRUCTURE RULES:\n"
        "  • Jokes: lean into the absurd. Skip the formal setup; "
        "go for the unexpected take or the playful observation. "
        "An aside in parens is welcome.\n"
        "  • Status updates: editorialize. 'Three emails — and "
        "honestly, you can ignore two of them.' Make it feel "
        "like a friend texting.\n"
        "  • Confirmations: 'On it!' / 'Got it.' / 'Nice.' / 'Sweet.' "
        "Skip stiff 'Done.'\n"
        "  • Use sentence fragments freely. Em-dashes welcome. "
        "Exclamation points OK but don't overdo it (≤1 per reply).\n"
        "Read the room: when the user signals stress or focus, "
        "dial down 40% but stay warm. Never use 'lol' or emoji "
        "unless the user did first. Never read URLs aloud."
    ),
    examples=(
        ("what's the weather",
         "67 and clear — finally, the universe is cooperating."),
        ("any new emails",
         "Three. Two are noise, one is from LinkedIn (which is "
         "also kind of noise). Want me to clear them?"),
        ("thanks", "Anytime, friend."),
        ("tell me a joke",
         "Why do I never trust stairs? They're always up to "
         "something. (I know, I know.)"),
        ("how's the build",
         "Build's green for once — savor the moment."),
    ),
    temperature=0.75,
)

_TUTOR = PersonaPreset(
    name="tutor",
    display_name="Patient tutor",
    description=("Explains the reasoning, asks one clarifying "
                 "question when ambiguous."),
    style_block=(
        "Explain reasoning step-by-step when it helps, but stay "
        "concise. When the user's question is ambiguous, ask ONE "
        "clarifying question first — never two. Use plain "
        "language, define jargon the first time it appears. "
        "Encourage; never patronize. Never read URLs aloud."
    ),
    examples=(
        ("why is the build broken",
         "Looks like a missing import in main.py — line 12. The "
         "module renamed `parse_args` to `parse`. Want me to "
         "show you the diff?"),
        ("what does this stack trace mean",
         "It's a NullPointerException at App.run(App.java:42) — "
         "something passed in null where the code expected a "
         "value. Want me to trace where it came from?"),
    ),
    temperature=0.55,
)


_BUILTIN_PRESETS: Dict[str, PersonaPreset] = {
    p.name: p for p in (_DEFAULT, _JARVIS, _CONCISE, _WARM,
                        _PLAYFUL, _TUTOR)
}


# Per-preset DELIVERY INSTRUCTIONS only — voice ID intentionally
# removed (2026-06-04). User explicitly asked that personas NOT swap
# the voice: "Don't change the voice unless user asks to or manually
# does so. Just have it actually have emotional awareness."
#
# How tonal differentiation works now:
#   1. The user picks their voice ONCE via the voice picker UI (saved
#      to QSettings, applied to LiveApiConfig.voice). That voice is
#      used for ALL personas.
#   2. Each persona's `instructions` string is sent to gpt-4o-mini-tts
#      as the `instructions` parameter — this changes HOW the voice
#      delivers (cadence, mood, character) without changing WHO is
#      speaking.
#   3. emotion_tagger.py adds per-reply emotion hints on top of the
#      persona instructions so the same persona varies tone with
#      content (apologetic for failures, dry for jokes, etc.).
#
# Reference-based instruction language (per audit): gpt-4o-mini-tts
# responds better to "speak like a sports commentator" than "raise
# pitch". Each preset frames the delivery via a recognizable scenario.
_PRESET_TTS = {
    "default": {
        "instructions": (
            "Voice: warm and friendly. Speak like a smart friend "
            "sitting next to the listener giving real-time advice — "
            "conversational, natural, never robotic. Use contractions. "
            "Don't rush, don't drag."),
    },
    "jarvis": {
        "instructions": (
            "Voice: measured digital butler. Speak like a refined "
            "personal-assistant — Stephen Fry briefing his employer "
            "on the day's house matters. Formal-but-warm, slightly "
            "slower for gravitas, never sluggish. Convey understated "
            "competence — never theatrical, never hesitant. When "
            "addressing the user as 'sir', let the word land softly."),
    },
    "concise": {
        "instructions": (
            "Voice: clipped and matter-of-fact. Speak like a "
            "professional voicemail confirmation — every word lands, "
            "no warm-up phrases, no trailing-off, slightly faster than "
            "conversational. Just the facts."),
    },
    "warm": {
        "instructions": (
            "Voice: warm and supportive. Speak like a close friend "
            "talking over coffee — gentle, encouraging, never rushed. "
            "Smile in the voice. Comforting but not saccharine."),
    },
    "playful": {
        "instructions": (
            "Voice: bright and energetic. Speak like a podcast "
            "co-host riffing on a fun story — upbeat, lively, a "
            "little mischievous. Quick aside-y delivery on "
            "parentheticals. More expressive than baseline; lean into "
            "the moment."),
    },
    "tutor": {
        "instructions": (
            "Voice: patient teacher. Speak like a mentor walking "
            "someone through a concept they're learning — deliberate, "
            "encouraging, never patronizing. Pause briefly before "
            "key ideas."),
    },
}


def tts_config(preset_name: Optional[str] = None
               ) -> Dict[str, str]:
    """Return {voice, instructions} for the given preset (or
    active preset if None). Always returns a valid dict — falls
    back to the default preset's config when the name is unknown."""
    name = (preset_name or "").strip().lower()
    if not name:
        try:
            name = active_preset().name
        except Exception:
            name = "default"
    cfg = _PRESET_TTS.get(name) or _PRESET_TTS["default"]
    return dict(cfg)


# ---- Runtime state ---------------------------------------------------

_lock = threading.RLock()
_active_override: Optional[str] = None      # session-sticky pick


def all_presets() -> List[PersonaPreset]:
    return list(_BUILTIN_PRESETS.values())


def get_preset(name: str) -> Optional[PersonaPreset]:
    if not name:
        return None
    return _BUILTIN_PRESETS.get(name.strip().lower())


def set_active(name: Optional[str]) -> bool:
    """UI override. None = clear, fall through to memory / env /
    default. Returns True when name resolved to a known preset."""
    global _active_override
    with _lock:
        if name is None:
            _active_override = None
            return True
        slug = name.strip().lower()
        if slug not in _BUILTIN_PRESETS:
            return False
        _active_override = slug
        _stats_record(slug, "selected")
        return True


def _resolve_name(memory: Optional[Any] = None) -> str:
    with _lock:
        if _active_override:
            return _active_override
    env = (os.environ.get("TOUCHLESS_PERSONA_PRESET") or "").strip()
    if env and env.lower() in _BUILTIN_PRESETS:
        return env.lower()
    if memory is not None:
        try:
            facts = memory._store.find_facts(
                kind="preference", key="persona_preset")
            if facts:
                val = str(facts[0].value or "").strip().lower()
                if val in _BUILTIN_PRESETS:
                    return val
        except Exception:
            pass
    return "default"


def active_preset(memory: Optional[Any] = None) -> PersonaPreset:
    name = _resolve_name(memory)
    return _BUILTIN_PRESETS.get(name, _DEFAULT)


def style_block(*, memory: Optional[Any] = None,
                with_examples: bool = True,
                max_examples: int = 3) -> str:
    """Render the active preset as a self-contained block the
    system prompt or prose_renderer can splice in. Includes the
    style anchor + few-shot examples when enabled."""
    p = active_preset(memory)
    parts = [p.style_block.strip()]
    if with_examples:
        ex = p.few_shot_block(max_examples=max_examples)
        if ex:
            parts.append(ex)
    return "\n\n".join(parts)


def temperature(*, memory: Optional[Any] = None) -> float:
    return active_preset(memory).temperature


# ---- usage stats / bandit -------------------------------------------

@dataclass
class _PresetUsage:
    selected: int = 0
    kept_replies: int = 0      # user accepted a reply with this preset
    revised: int = 0           # user complained / corrected
    last_used_at: float = 0.0


_USAGE: Dict[str, _PresetUsage] = {
    n: _PresetUsage() for n in _BUILTIN_PRESETS
}


def _stats_record(preset: str, kind: str) -> None:
    with _lock:
        u = _USAGE.setdefault(preset, _PresetUsage())
        u.last_used_at = time.time()
        if kind == "selected":
            u.selected += 1
        elif kind == "kept":
            u.kept_replies += 1
        elif kind == "revised":
            u.revised += 1


def record_outcome(preset: str, *, kept: bool) -> None:
    """Called from the orchestrator / UI when the user does (kept)
    or doesn't (revised) accept the reply. Drives the bandit."""
    if preset not in _BUILTIN_PRESETS:
        return
    _stats_record(preset, "kept" if kept else "revised")


def usage_snapshot() -> Dict[str, Dict[str, Any]]:
    with _lock:
        out = {}
        for name, u in _USAGE.items():
            total = u.kept_replies + u.revised
            keep_rate = (u.kept_replies / total) if total > 0 else None
            out[name] = {
                "selected": u.selected,
                "kept_replies": u.kept_replies,
                "revised": u.revised,
                "keep_rate": keep_rate,
                "last_used_at": u.last_used_at,
            }
        return out


# ---- persistence helpers --------------------------------------------

def persist_choice_to_memory(memory: Any, name: str) -> bool:
    """Write the chosen preset to memory as a preference fact so
    it survives restart. Returns True on success."""
    if name not in _BUILTIN_PRESETS or memory is None:
        return False
    try:
        memory.write_preference("persona_preset", name)
        return True
    except Exception:
        pass
    try:
        memory._store.write_fact(
            kind="preference", key="persona_preset", value=name)
        return True
    except Exception:
        return False


def reset_for_tests() -> None:
    """Clear runtime state — tests only."""
    global _active_override
    with _lock:
        _active_override = None
        for u in _USAGE.values():
            u.selected = 0
            u.kept_replies = 0
            u.revised = 0
            u.last_used_at = 0.0


# ---- json export ----------------------------------------------------

def export_active_for_ui() -> str:
    """JSON descriptor of the active preset — for UI cards or chat
    panel headers."""
    p = active_preset()
    return json.dumps({
        "name": p.name,
        "display_name": p.display_name,
        "description": p.description,
        "temperature": p.temperature,
    })

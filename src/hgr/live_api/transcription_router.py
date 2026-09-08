"""Two-tier transcription router.

Phase-2 voice. Iris hears the user three different ways today —
the live realtime model, the whisper-batch path in voice_command_
listener, and the streaming dictation pipeline. Each has wildly
different latency/cost/accuracy tradeoffs:

  * **fast**    — local whisper.cpp tiny.en / base.en streaming.
                  Sub-second; cheap; misses non-English and stutters.
                  Used to: detect a wake-word, decide which path
                  to send the full utterance to, AND surface a
                  preview transcription instantly while the more
                  accurate model is still running.

  * **accurate** — whisper.cpp medium.en OR OpenAI gpt-4o-transcribe
                  via the realtime model. 1-3s; slow but solid.
                  Used to: final transcript for commands that act
                  on the world.

The router decides which tier handles a given audio frame based on
the **classifier** decision: "this clip says a wake-word + a
trivial command → fast is good enough"; "this clip looks long /
ambiguous / contains an email / a destination → escalate".

Escalation rules (deterministic):
  1. Audio length > 6 seconds → ACCURATE.
  2. Fast transcript contains a domain-name, email, file path, or
     a sequence of >2 digits in a row → ACCURATE.
  3. Fast transcript starts with a destructive verb (delete, send,
     email, transfer, pay, post) → ACCURATE.
  4. Fast confidence < 0.55 (when the backend reports one) → ACCURATE.
  5. User asked for "verbatim mode" → ACCURATE.
  6. Otherwise → FAST is good enough.

Router doesn't run STT itself; it returns a decision (`tier`,
`reason`) and the caller invokes the matching engine. Keeps this
module dependency-free + testable.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TranscriptTier(str, Enum):
    FAST = "fast"
    ACCURATE = "accurate"


@dataclass
class TranscriptionDecision:
    tier: TranscriptTier
    reason: str = ""
    # When True, callers should ALSO run fast in parallel as a UI
    # preview while accurate's slower result is awaited.
    show_fast_preview: bool = True


# Verbs that imply real-world side effects. Word-boundary checked,
# not just prefix — so "please delete" still escalates. Cheap to
# maintain — add more as new connectors land. SEC-012 audit
# expansion: drop/discard/revoke/kill/terminate/format/erase/wipe.
_DESTRUCTIVE_VERBS = frozenset({
    "delete", "remove", "trash", "drop", "discard", "wipe",
    "erase", "format", "revoke", "kill", "terminate", "shutdown",
    "send", "email", "post", "publish", "tweet", "message",
    "transfer", "wire", "pay", "buy", "purchase", "subscribe",
    "schedule", "cancel", "decline",
    "restart", "reboot",
    "uninstall", "unsubscribe",
})

# Politeness / leading prefixes the STT often produces before the
# real verb. Stripped BEFORE the destructive-verb check.
_POLITENESS_PREFIXES = (
    "please ", "can you ", "could you ", "would you ",
    "go ahead and ", "now ", "just ", "kindly ",
    "i need you to ", "i want you to ", "i'd like you to ",
)

# Pre-compiled word-boundary regex for fast scanning.
_DESTRUCTIVE_RE = re.compile(
    r"\b(" + "|".join(sorted(_DESTRUCTIVE_VERBS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

# Cheap regexes for ambiguity signals.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b")
_URL_RE = re.compile(r"\bhttps?://\S+\b|\b\w+\.(?:com|net|org|io|app|ai|co)\b",
                     re.IGNORECASE)
_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s]+|/(?:home|usr|opt)/\S+")
_DIGIT_RUN_RE = re.compile(r"\d{3,}")

# Length threshold (seconds) beyond which short-tier is no longer
# trustworthy enough for an actionable command.
DEFAULT_LONG_CLIP_SEC = 6.0

# Conf threshold below which we escalate even if everything else
# looks fine. -1 = backend reports no confidence (treat as "ok").
DEFAULT_LOW_CONF = 0.55

# Verbatim mode forces accurate for everything (dictation use-case).
_VERBATIM_ENV = "TOUCHLESS_VERBATIM_TRANSCRIPTION"


def _verbatim_enabled() -> bool:
    return os.environ.get(_VERBATIM_ENV, "0") == "1"


class TranscriptionRouter:
    """Decides which transcription tier serves a given clip."""

    def __init__(self, *,
                 long_clip_sec: float = DEFAULT_LONG_CLIP_SEC,
                 low_conf: float = DEFAULT_LOW_CONF) -> None:
        self._long = float(long_clip_sec)
        self._low_conf = float(low_conf)

    def decide(self, *,
               audio_seconds: float = 0.0,
               fast_text: Optional[str] = None,
               fast_confidence: float = -1.0,
               user_requested_verbatim: bool = False,
               ) -> TranscriptionDecision:
        """Return the tier that should produce the FINAL transcript.

        `fast_text` is the preview transcript the fast tier already
        produced (when available); we read it for ambiguity signals.
        Passing None just means "we don't have it yet" — falls back
        to length-only routing.
        """
        if user_requested_verbatim or _verbatim_enabled():
            return TranscriptionDecision(
                TranscriptTier.ACCURATE,
                reason="verbatim mode",
                show_fast_preview=False)

        if audio_seconds > self._long:
            return TranscriptionDecision(
                TranscriptTier.ACCURATE,
                reason=f"long clip ({audio_seconds:.1f}s > {self._long:.1f}s)")

        if fast_text:
            ambiguity = _why_ambiguous(fast_text)
            if ambiguity:
                return TranscriptionDecision(
                    TranscriptTier.ACCURATE,
                    reason=f"ambiguous: {ambiguity}")
            if _starts_with_destructive(fast_text):
                return TranscriptionDecision(
                    TranscriptTier.ACCURATE,
                    reason="destructive verb detected")

        if 0 <= fast_confidence < self._low_conf:
            return TranscriptionDecision(
                TranscriptTier.ACCURATE,
                reason=f"low confidence {fast_confidence:.2f}")

        return TranscriptionDecision(
            TranscriptTier.FAST,
            reason="fast tier sufficient")


# ---- module helpers ----------------------------------------------------

def _why_ambiguous(text: str) -> str:
    """Return a short reason string if `text` contains tokens we
    know the fast tier mangles. Empty string when clean."""
    if _EMAIL_RE.search(text):
        return "contains email"
    if _URL_RE.search(text):
        return "contains URL/domain"
    if _PATH_RE.search(text):
        return "contains path"
    if _DIGIT_RUN_RE.search(text):
        return "contains digit run"
    return ""


def _starts_with_destructive(text: str) -> bool:
    """True when the utterance contains a destructive verb at word
    boundary, after stripping common politeness prefixes. The function
    name is historical — checks ANYWHERE in the leading clause, not
    just position 0 (SEC-012 hardening)."""
    if not text:
        return False
    t = (text or "").lower().lstrip()
    # Strip politeness prefixes iteratively (handles "please can you
    # delete...").
    changed = True
    while changed:
        changed = False
        for p in _POLITENESS_PREFIXES:
            if t.startswith(p):
                t = t[len(p):]
                changed = True
    # Word-boundary scan for a destructive verb in the first ~80 chars
    # — limits scope so a long monologue with "delete" buried mid-way
    # doesn't escalate.
    head = t[:80]
    return bool(_DESTRUCTIVE_RE.search(head))

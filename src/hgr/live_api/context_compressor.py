"""Session context compression.

Phase-2 cognition. As a session grows, the rolling memory context
block injected into the planner prompt grows too. Past ~1500 chars
it starts to crowd out the user's actual question; past ~3000 chars
the planner gets distracted from the goal entirely.

Strategy: keep recent turns verbatim, COMPRESS older turns into a
one-line summary preserved as a `summary` row, then drop the
verbose originals. Two compression backends:

  * `heuristic`  — pure rule-based; collapses each turn into a
                   one-line "user did X, assistant did Y" tag.
                   Zero token cost; always available.
  * `llm`        — Haiku-tier summarization via the existing
                   LLMPlanner / planner_llm path. Smarter but
                   billable; opt-in via TOUCHLESS_CONTEXT_LLM=1.

Triggers:
  * When `len(context_text) > soft_limit_chars` (default 1500).
  * When the configured budget for a given task tier would be
    exceeded by the next call (`would_exceed_budget`).

This module owns the COMPRESSION step only. Whoever calls it
(MemoryManager) decides when to substitute the compressed string
for the rolling buffer.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


@dataclass
class ContextChunk:
    """One unit of session context.

    `kind`: 'turn' | 'fact' | 'summary'
    `text`: the rendered form ready for the prompt.
    `weight`: relative importance — higher means more likely to be
    KEPT verbatim during compression. Facts > recent turns >
    old turns > summaries.
    `age_turns`: how many turns ago this happened (0 = current).
    """
    kind: str
    text: str
    weight: float = 1.0
    age_turns: int = 0


# Defaults. Override via env: TOUCHLESS_CONTEXT_SOFT, TOUCHLESS_CONTEXT_HARD.
DEFAULT_SOFT_LIMIT_CHARS = 1500
DEFAULT_HARD_LIMIT_CHARS = 3000
# Recent turns to ALWAYS keep verbatim, regardless of size.
KEEP_RECENT_TURNS = 4


def _soft_limit() -> int:
    try:
        return int(os.environ.get("TOUCHLESS_CONTEXT_SOFT",
                                  str(DEFAULT_SOFT_LIMIT_CHARS)))
    except Exception:
        return DEFAULT_SOFT_LIMIT_CHARS


def _hard_limit() -> int:
    try:
        return int(os.environ.get("TOUCHLESS_CONTEXT_HARD",
                                  str(DEFAULT_HARD_LIMIT_CHARS)))
    except Exception:
        return DEFAULT_HARD_LIMIT_CHARS


class ContextCompressor:
    """Compresses a sequence of ContextChunks into a string that
    fits within the configured char limit."""

    def __init__(self, *, llm_planner: Optional[Any] = None,
                 soft_limit_chars: Optional[int] = None,
                 hard_limit_chars: Optional[int] = None) -> None:
        self._llm = llm_planner
        self._soft = soft_limit_chars or _soft_limit()
        self._hard = hard_limit_chars or _hard_limit()

    def compress(self, chunks: Sequence[ContextChunk]) -> str:
        """Return a context string fitting under the hard limit.
        Strategy:
          1. Always emit facts (they're already compact).
          2. Always emit the most recent KEEP_RECENT_TURNS turns
             verbatim.
          3. For older turns, replace with one-line summaries
             produced by the chosen backend.
          4. If we're still over the hard limit, drop summaries
             from the oldest end."""
        if not chunks:
            return ""
        facts = [c for c in chunks if c.kind == "fact"]
        turns = [c for c in chunks if c.kind == "turn"]
        summaries = [c for c in chunks if c.kind == "summary"]

        # Order turns by age — newest first. Then split into kept-verbatim
        # vs compress-this.
        turns_sorted = sorted(turns, key=lambda c: c.age_turns)
        verbatim = turns_sorted[:KEEP_RECENT_TURNS]
        to_compress = turns_sorted[KEEP_RECENT_TURNS:]

        compressed_old: List[ContextChunk] = list(summaries)
        if to_compress:
            backend = self._pick_backend()
            for chunk in to_compress:
                summary_text = self._summarize_one(chunk.text, backend)
                if summary_text:
                    compressed_old.append(ContextChunk(
                        kind="summary",
                        text=summary_text,
                        weight=chunk.weight * 0.5,
                        age_turns=chunk.age_turns,
                    ))

        rendered = self._render(facts, verbatim, compressed_old)
        if len(rendered) <= self._hard:
            return rendered

        # Hard-limit fallback: drop oldest summaries until we fit.
        compressed_old.sort(key=lambda c: -c.age_turns)
        while compressed_old and len(rendered) > self._hard:
            compressed_old.pop()
            rendered = self._render(facts, verbatim, compressed_old)
        return rendered[:self._hard]

    def would_exceed_budget(self, current_text: str) -> bool:
        return len(current_text or "") > self._soft

    # ---- internals ----------------------------------------------------

    def _pick_backend(self) -> str:
        if (os.environ.get("TOUCHLESS_CONTEXT_LLM", "0") == "1"
                and self._llm is not None):
            return "llm"
        return "heuristic"

    def _summarize_one(self, turn_text: str, backend: str) -> str:
        if backend == "llm":
            try:
                return self._llm_summarize(turn_text)
            except Exception:
                pass  # fall through to heuristic
        return _heuristic_summarize(turn_text)

    def _llm_summarize(self, turn_text: str) -> str:
        # Delegate to the planner LLM with a one-line prompt. The
        # planner returns a Plan; we expect the model to put the
        # summary in `goal`. Cheap, single shot.
        #
        # SEC-009: wrap turn_text in content_quarantine envelope so
        # any prior assistant/tool content embedded in the turn (which
        # may itself include external attacker text) cannot escape
        # the summarizer prompt as an instruction.
        try:
            from .content_quarantine import wrap as _quarantine_wrap
            wrapped_turn = _quarantine_wrap((turn_text or "")[:1200],
                                            source="earlier_turn")
        except Exception:
            wrapped_turn = (turn_text or "")[:1200]
        goal = (
            "Summarize the following conversation turn in ONE sentence "
            "of at most 80 characters. Keep the user's intent + outcome. "
            "Drop greetings and filler. Reply with PLAIN TEXT only, no "
            "tools.\n\n" + wrapped_turn
        )
        plan = self._llm.plan(goal)
        if plan is None:
            return _heuristic_summarize(turn_text)
        return (plan.goal or "")[:120].strip() or _heuristic_summarize(turn_text)

    def _render(self, facts: List[ContextChunk],
                verbatim: List[ContextChunk],
                compressed_old: List[ContextChunk]) -> str:
        sections: List[str] = []
        if facts:
            facts_block = "FACTS:\n" + "\n".join(
                f"  - {c.text.strip()}" for c in facts)
            sections.append(facts_block[:600])
        if compressed_old:
            old_block = "EARLIER:\n" + "\n".join(
                f"  - {c.text.strip()}"
                for c in sorted(compressed_old, key=lambda x: -x.age_turns))
            sections.append(old_block[:800])
        if verbatim:
            # Newest first: low age_turns is the most recent turn.
            recent_block = "RECENT:\n" + "\n".join(
                f"  • {c.text.strip()}"
                for c in sorted(verbatim, key=lambda x: x.age_turns))
            sections.append(recent_block[:1200])
        return "\n\n".join(sections)


# ---- module helpers --------------------------------------------------

_NOISE_RE = re.compile(
    r"\b(?:um+|uh+|hmm+|like|you know|i mean|kinda|sorta)\b",
    re.IGNORECASE,
)

_TOOL_VERB = {
    "weather_get": "checked weather",
    "gmail_send": "sent an email via gmail",
    "ms_mail_send": "sent an email via outlook",
    "gmail_list": "listed gmail inbox",
    "ms_mail_list": "listed outlook inbox",
    "notion_search": "searched notion",
    "notion_create_page": "created a notion page",
    "drive_upload": "uploaded a file to drive",
    "todo_add": "added a task",
    "volume_set": "set volume",
    "volume_get": "checked volume",
    "spotify_play": "played music on spotify",
    "calendar_list": "listed calendar events",
    "iris_lookup_contact": "looked up a contact",
    "iris_remember_contact": "remembered a contact",
    "iris_set_preference": "saved a preference",
}


def _heuristic_summarize(turn_text: str) -> str:
    """Cheap one-liner summary: extract the user's first verb phrase
    and any tool name we recognize. No LLM."""
    t = (turn_text or "").strip()
    if not t:
        return ""
    # Drop disfluencies for clarity.
    t = _NOISE_RE.sub("", t).strip()
    # If the turn is itself short, the heuristic is just "use it".
    if len(t) <= 80:
        return t
    # Try to find a tool name in the text and a verb phrase.
    tool_hits = [v for k, v in _TOOL_VERB.items() if k in t]
    if tool_hits:
        # Most distinctive: the first matched tool verb + the user's
        # first 50 chars of intent.
        intent = re.sub(r"\s+", " ", t.split(".")[0])[:50]
        return f"{intent}… ({tool_hits[0]})"
    return re.sub(r"\s+", " ", t.split(".")[0])[:80] + "…"


def build_chunks_from_recall(memory_recall: Any) -> List[ContextChunk]:
    """Convenience: convert a MemoryManager.recall() result into
    ContextChunks the compressor can ingest. Mostly used by the
    integration layer in memory.manager — kept here so the data
    shape is documented in one place."""
    chunks: List[ContextChunk] = []
    facts = memory_recall.get("facts", []) or []
    for i, fact in enumerate(facts):
        chunks.append(ContextChunk(
            kind="fact",
            text=str(fact),
            weight=2.0,
            age_turns=0,
        ))
    episodes = memory_recall.get("episodes", []) or []
    for i, ep in enumerate(episodes):
        text = str(getattr(ep, "text", ep))
        chunks.append(ContextChunk(
            kind="turn",
            text=text,
            weight=1.0,
            age_turns=i + 1,
        ))
    return chunks

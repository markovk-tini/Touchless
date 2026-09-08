"""Multi-modal context fusion — unify ambient context into one block.

Phase-5 cognition. By Phase 4 the orchestrator's `_recall_context`
was injecting up to FIVE separate sections into the planner prompt:

  * memory recall (facts + episodes)
  * recent dictation
  * repo context (when in IDE)
  * session buffer (recent chat turns)
  * screen awareness (when vision-relevant)

Stacked, these often crossed 2-3 KB even for a 20-char request.
That bloats the planner prompt + makes the LLM less focused on
the actual task.

This module owns:
  * **Relevance scoring** — for each modality, rate how relevant
    it is to THIS user request (cheap heuristic: keyword overlap,
    presence of vision/dictation triggers, etc.).
  * **Budget allocation** — split a hard char budget (default
    2000) across modalities by relevance, dropping the least-
    relevant when over.
  * **Section dedup** — when two modalities reference the same
    fact (e.g., memory says "Dani's email is dani@x" AND a
    recent session turn quoted the same email), merge.
  * **Compact rendering** — single unified block the orchestrator
    can splat into the prompt.

The orchestrator delegates to `build_unified_context(text)` and
gets back a single string. Per-modality fetch logic stays in the
modality modules (memory.manager, repo_focus_watcher, etc.) —
this is the FUSER, not the source.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# Default hard char budget for the whole unified block.
DEFAULT_BUDGET_CHARS = 2000
# Per-section minimum reserve so a low-relevance modality with
# important content (e.g., a pinned memory fact) still gets at
# least a tiny slice.
MIN_RESERVE_PER_SECTION = 80


@dataclass
class ContextSection:
    """One modality's contribution to the unified block."""
    name: str                      # 'memory', 'dictation', etc.
    raw_text: str                  # the modality's full rendered block
    relevance: float = 0.5         # 0..1; 1 = always include
    always_include: bool = False   # bypass budget when True
    # Optional priority bias: when two sections compete for the
    # same budget slice, higher priority wins.
    priority: int = 0


# Modality-specific relevance heuristics. Each takes the user text
# and the modality's raw block, returns a score 0..1. Pure
# functions — easy to test, easy to tune.
def _rel_memory(text: str, block: str) -> float:
    if not block:
        return 0.0
    # Memory recall is always at least mildly relevant — it filters
    # to facts whose keys appeared in the request.
    return 0.7


def _rel_repo(text: str, block: str) -> float:
    if not block:
        return 0.0
    t = (text or "").lower()
    # Boost when the request mentions code-related keywords or
    # references "this project" / "this repo" / "this branch".
    if any(w in t for w in ("repo", "branch", "commit", "code",
                             "project", "this codebase", "main")):
        return 0.95
    # Default: keep it but don't blow budget.
    return 0.45


def _rel_dictation(text: str, block: str) -> float:
    if not block:
        return 0.0
    # `_wants_recent_dictation` already gated this — if it made it
    # into the block, it's HIGHLY relevant. Else 0.
    return 0.9


def _rel_session(text: str, block: str) -> float:
    if not block:
        return 0.0
    # Short follow-ups + pronouns benefit most from session context.
    t = (text or "").lower()
    is_short = len(t) <= 60
    has_pronoun = bool(re.search(
        r"\b(it|that|this|them|those|these|him|her|the one)\b", t))
    if is_short and has_pronoun:
        return 0.85
    if is_short:
        return 0.65
    return 0.4


def _rel_screen(text: str, block: str) -> float:
    if not block:
        return 0.0
    # `looks_vision_relevant` already gated this — present = highly
    # relevant.
    return 0.9


def _rel_resolved_refs(text: str, block: str) -> float:
    """Phase-6: pronoun resolver output. When present, it's highly
    relevant — the resolver already gated on whether pronouns or
    names matched."""
    if not block:
        return 0.0
    return 0.95


_RELEVANCE_FUNCS: Dict[str, Callable[[str, str], float]] = {
    "memory": _rel_memory,
    "repo": _rel_repo,
    "dictation": _rel_dictation,
    "session": _rel_session,
    "screen": _rel_screen,
    "resolved_refs": _rel_resolved_refs,
}


def score_sections(text: str, parts: Dict[str, str]
                   ) -> List[ContextSection]:
    """Score each non-empty part. Returns a list of ContextSection
    objects sorted by relevance desc."""
    sections: List[ContextSection] = []
    for name, block in (parts or {}).items():
        if not block:
            continue
        func = _RELEVANCE_FUNCS.get(name)
        rel = func(text, block) if func else 0.5
        if rel <= 0.0:
            continue
        # Memory facts that include the word "pinned" / "auto-pin"
        # are user-reinforced; bump priority so they survive budget
        # cuts.
        priority = 0
        if name == "memory" and "auto-pin" in block.lower():
            priority = 1
        sections.append(ContextSection(
            name=name, raw_text=block, relevance=rel,
            priority=priority))
    sections.sort(key=lambda s: (-s.priority, -s.relevance))
    return sections


def allocate_budget(sections: List[ContextSection],
                    budget: int = DEFAULT_BUDGET_CHARS
                    ) -> List[Tuple[ContextSection, int]]:
    """Given scored sections + a total char budget, decide how many
    chars each section gets. High-relevance + high-priority sections
    get more; everyone gets at least MIN_RESERVE_PER_SECTION when
    possible. Returns [(section, allowed_chars), ...] in the same
    order."""
    if not sections or budget <= 0:
        return []
    # Reserve floor first.
    n = len(sections)
    floor = min(MIN_RESERVE_PER_SECTION, budget // max(n, 1))
    remaining = budget - floor * n
    # Distribute remaining proportional to relevance.
    total_rel = sum(s.relevance for s in sections) or 1.0
    out: List[Tuple[ContextSection, int]] = []
    for s in sections:
        extra = int(remaining * (s.relevance / total_rel))
        cap = floor + extra
        cap = min(cap, len(s.raw_text))
        out.append((s, cap))
    return out


def dedup_overlap(sections: List[Tuple[ContextSection, int]]
                  ) -> List[Tuple[ContextSection, str]]:
    """Drop sections whose first 80 chars substantially overlap
    with a previous (higher-relevance) section. Returns
    [(section, trimmed_text)] where trimmed_text fits the allocated
    char cap."""
    out: List[Tuple[ContextSection, str]] = []
    seen_heads: List[str] = []
    for s, cap in sections:
        text = s.raw_text[:cap]
        head = re.sub(r"\s+", " ", text[:80].lower())
        # Skip when this head is a substring of any already-kept
        # section's head (or vice versa). Conservative — only kills
        # near-exact duplicates.
        if any(head and (head in h or h in head) for h in seen_heads):
            continue
        seen_heads.append(head)
        out.append((s, text))
    return out


def render(parts: List[Tuple[ContextSection, str]]) -> str:
    """Render the surviving sections into a single block, ordered
    by section priority (memory first, then session, then screen,
    then repo, then dictation)."""
    if not parts:
        return ""
    # Resolved refs come FIRST so the planner sees the binding
    # hints before the raw context blocks.
    order = ["resolved_refs", "memory", "session", "screen", "repo",
             "dictation"]
    indexed = {name: i for i, name in enumerate(order)}
    parts = sorted(parts, key=lambda kv: indexed.get(kv[0].name, 99))
    lines: List[str] = []
    for section, text in parts:
        text = text.rstrip()
        if not text:
            continue
        lines.append(text)
    return "\n\n".join(lines)


def build_unified_context(text: str, parts: Dict[str, str],
                          *, budget: int = DEFAULT_BUDGET_CHARS
                          ) -> str:
    """Top-level fuser. Takes the user's request + a dict of
    per-modality blocks and produces a single unified context
    string. Empty string when no modality contributed anything."""
    scored = score_sections(text, parts)
    allocated = allocate_budget(scored, budget=budget)
    deduped = dedup_overlap(allocated)
    return render(deduped)

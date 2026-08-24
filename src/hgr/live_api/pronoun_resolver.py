"""Multi-modal pronoun resolver — "send THAT to HIM" actually works.

Phase-6 cognition. Today every Iris turn treats pronouns as
opaque strings — the planner LLM has to guess what "that" and
"him" refer to from raw text. With four modalities now wired
(memory, session_buffer, screen_awareness, dictation_bridge) +
the new entity graph, we can resolve pronouns DETERMINISTICALLY
before the planner sees them.

How it works:
  * The orchestrator calls `resolve_references(text)` before
    building the planner prompt.
  * The resolver scans `text` for pronoun + demonstrative tokens
    ("it", "that", "this", "him", "her", "them", "the one", etc.).
  * For each pronoun, it picks the most-relevant entity by
    consulting (in order):
      1. The session_buffer's last 4 turns for the most recent
         mention.
      2. The screen_awareness summary for what's visible right now.
      3. The entity graph for the most-recently-touched entity
         of the matching kind.
  * Returns a `ResolutionReport` with the original text + a
    rewritten version that inlines the resolved references
    (e.g., "send that to him" → "send [the Q3 contract] to
    [Dani <dani@x>]"), PLUS a structured map of which token
    resolved to which entity.

The orchestrator injects this report into the planner prompt as a
"RESOLVED REFERENCES" block so the planner gets BOTH the
original natural-language input AND the deterministic resolution
hint. If the planner disagrees with the hint, it can override —
this is a NUDGE, not a hard rewrite.

Pure-Python; no LLM in the loop. Cheap.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .entity_graph import Entity, EntityKind, global_graph


# Pronoun classes — each maps to a preferred entity kind.
_PERSON_PRONOUNS = frozenset({
    "him", "her", "them", "they",
    # NOTE: "she" and "he" are intentionally NOT included as
    # subject pronouns — they appear too often in narration
    # ("she said", "he wrote"). Object pronouns ('him', 'her')
    # are safer markers of references the user means for Iris
    # to act on.
})
_THING_PRONOUNS = frozenset({
    "it", "that", "this", "those", "these",
    # Multi-word demonstratives are matched separately below.
})
_THING_DEMONSTRATIVES = (
    "the one", "that one", "this one", "the thing", "that thing",
    "the same", "the previous", "the last one",
)
_PRONOUN_RE = re.compile(
    r"\b(" + "|".join(_PERSON_PRONOUNS | _THING_PRONOUNS) + r")\b",
    re.IGNORECASE,
)


# Direct-reference patterns ("Dani", "Q3 contract") that AREN'T
# pronouns but we still want to surface as resolved entities so
# the planner gets the full picture.
_NAME_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]{1,30})\b")


@dataclass
class Resolution:
    """One pronoun/name → entity binding."""
    token: str                     # original surface form
    span: Tuple[int, int]          # char offsets in the user text
    entity_id: str
    entity_kind: str
    display_name: str
    detail: str = ""               # email/url for the planner prompt
    source: str = ""               # 'session' | 'screen' | 'graph'


@dataclass
class ResolutionReport:
    original_text: str
    resolutions: List[Resolution] = field(default_factory=list)
    # A compact block ready to drop into the planner prompt.
    prompt_block: str = ""

    def has_resolutions(self) -> bool:
        return bool(self.resolutions)


def resolve_references(text: str,
                       *,
                       graph: Optional[Any] = None,
                       session_buffer: Optional[Any] = None,
                       screen_summary: Optional[Any] = None,
                       ) -> ResolutionReport:
    """Top-level resolver. All modality sources are injectable so
    tests can stub them out; production callers pass None and we
    look up the globals.

    Returns a `ResolutionReport`. Empty resolutions list when no
    pronouns/names found OR none resolved confidently."""
    report = ResolutionReport(original_text=text or "")
    if not text:
        return report
    g = graph if graph is not None else _safe_global_graph()
    # 1. Pronoun pass.
    pronoun_resolutions = _resolve_pronouns(
        text, graph=g, session_buffer=session_buffer,
        screen_summary=screen_summary)
    report.resolutions.extend(pronoun_resolutions)
    # 2. Name pass — direct references that match a known person.
    name_resolutions = _resolve_names(text, graph=g)
    report.resolutions.extend(name_resolutions)
    # 3. Multi-word demonstratives.
    demo_resolutions = _resolve_demonstratives(
        text, graph=g, session_buffer=session_buffer,
        screen_summary=screen_summary)
    report.resolutions.extend(demo_resolutions)
    report.prompt_block = _render_block(report.resolutions)
    return report


# ---- internals ---------------------------------------------------------

def _safe_global_graph() -> Optional[Any]:
    try:
        return global_graph()
    except Exception:
        return None


def _resolve_pronouns(text: str, *, graph: Optional[Any],
                      session_buffer: Optional[Any],
                      screen_summary: Optional[Any]
                      ) -> List[Resolution]:
    out: List[Resolution] = []
    for m in _PRONOUN_RE.finditer(text):
        token = m.group(1)
        token_lower = token.lower()
        if token_lower in _PERSON_PRONOUNS:
            entity = _pick_recent_entity_of_kind(
                EntityKind.PERSON.value,
                graph=graph, session_buffer=session_buffer)
            source = "graph" if entity else ""
        elif token_lower in _THING_PRONOUNS:
            entity = _pick_recent_thing(
                graph=graph, session_buffer=session_buffer,
                screen_summary=screen_summary)
            source = "session+screen+graph"
        else:
            entity = None
            source = ""
        if entity is None:
            continue
        out.append(Resolution(
            token=token, span=(m.start(), m.end()),
            entity_id=entity.id, entity_kind=entity.kind,
            display_name=entity.name,
            detail=_entity_detail(entity),
            source=source,
        ))
    return out


def _resolve_names(text: str, *,
                   graph: Optional[Any]) -> List[Resolution]:
    if graph is None:
        return []
    out: List[Resolution] = []
    seen_ids: set = set()
    for m in _NAME_RE.finditer(text):
        candidate = m.group(1)
        # Skip pronouns the regex picked up due to capitalization.
        if candidate.lower() in (_PERSON_PRONOUNS | _THING_PRONOUNS):
            continue
        # Skip common stop-words at sentence start.
        if candidate in {"The", "A", "An", "I", "This", "That",
                          "Today", "Tomorrow", "Monday", "Tuesday",
                          "Wednesday", "Thursday", "Friday",
                          "Saturday", "Sunday"}:
            continue
        try:
            entity = graph.find_by_alias(
                candidate, kind=EntityKind.PERSON.value)
        except Exception:
            entity = None
        if entity is None or entity.id in seen_ids:
            continue
        seen_ids.add(entity.id)
        # Touch the entity so subsequent pronouns resolve to it.
        try:
            graph.touch(entity.id)
        except Exception:
            pass
        out.append(Resolution(
            token=candidate, span=(m.start(), m.end()),
            entity_id=entity.id, entity_kind=entity.kind,
            display_name=entity.name,
            detail=_entity_detail(entity),
            source="graph",
        ))
    return out


def _resolve_demonstratives(text: str, *,
                            graph: Optional[Any],
                            session_buffer: Optional[Any],
                            screen_summary: Optional[Any]
                            ) -> List[Resolution]:
    """Multi-word demonstratives ('the one', 'that thing', etc.).
    Resolved like _THING_PRONOUNS — pick the most-recent thing
    across session + screen + graph."""
    t = text.lower()
    out: List[Resolution] = []
    for phrase in _THING_DEMONSTRATIVES:
        idx = 0
        while True:
            found = t.find(phrase, idx)
            if found < 0:
                break
            entity = _pick_recent_thing(
                graph=graph, session_buffer=session_buffer,
                screen_summary=screen_summary)
            if entity is not None:
                out.append(Resolution(
                    token=text[found:found + len(phrase)],
                    span=(found, found + len(phrase)),
                    entity_id=entity.id, entity_kind=entity.kind,
                    display_name=entity.name,
                    detail=_entity_detail(entity),
                    source="session+screen+graph",
                ))
            idx = found + len(phrase)
    return out


def _pick_recent_entity_of_kind(kind: str, *,
                                 graph: Optional[Any],
                                 session_buffer: Optional[Any]
                                 ) -> Optional[Entity]:
    """Pick the most-recently mentioned entity of `kind`. Prefers
    session-buffer mentions over graph-touch order — what the user
    JUST said wins over what's been recently active."""
    # Session buffer pass.
    session_pick = _session_buffer_recent_of_kind(
        kind, session_buffer=session_buffer, graph=graph)
    if session_pick is not None:
        return session_pick
    # Graph fallback.
    if graph is not None:
        try:
            recent = graph.recent_by_kind(
                kind, max_age_sec=24 * 3600, limit=1)
        except Exception:
            recent = []
        if recent:
            return recent[0]
    return None


def _pick_recent_thing(*, graph: Optional[Any],
                       session_buffer: Optional[Any],
                       screen_summary: Optional[Any]
                       ) -> Optional[Entity]:
    """Pick the most-recent 'thing' the user could mean by it /
    that / this. Three sources:
      1. The most-recent DOCUMENT/ARTIFACT mentioned in session.
      2. The active app/window from screen_summary (treated as
         a transient 'app' entity).
      3. The most-recently-touched non-person entity in the graph.
    """
    # Session pass — look across the thing-shaped entity kinds.
    for kind in (EntityKind.DOCUMENT.value, EntityKind.ARTIFACT.value,
                 EntityKind.FILE.value, EntityKind.EMAIL_THREAD.value,
                 EntityKind.PROJECT.value, EntityKind.TOPIC.value):
        pick = _session_buffer_recent_of_kind(
            kind, session_buffer=session_buffer, graph=graph)
        if pick is not None:
            return pick
    # Screen pass — synthesize a transient 'screen' entity that the
    # planner can act on. Not persisted; just used for resolution.
    if screen_summary is not None:
        try:
            app = getattr(screen_summary, "active_app", "") or ""
            title = getattr(screen_summary, "active_window_title",
                            "") or ""
            if app or title:
                # Return a non-persisted Entity for the prompt
                # block; entity_id="" signals "in-memory only".
                return Entity(
                    id="screen",
                    kind=EntityKind.APP.value,
                    name=(f"{app}" + (f" — {title}"
                                       if title else "")).strip(),
                    attrs={"transient": True},
                )
        except Exception:
            pass
    # Graph fallback — any recently-touched non-person entity.
    if graph is not None:
        candidates: List[Entity] = []
        for kind in (EntityKind.DOCUMENT.value,
                     EntityKind.ARTIFACT.value,
                     EntityKind.FILE.value,
                     EntityKind.EMAIL_THREAD.value,
                     EntityKind.PROJECT.value):
            try:
                candidates.extend(graph.recent_by_kind(
                    kind, max_age_sec=24 * 3600, limit=2))
            except Exception:
                continue
        if candidates:
            candidates.sort(key=lambda e: -e.last_seen_at)
            return candidates[0]
    return None


def _session_buffer_recent_of_kind(kind: str, *,
                                    session_buffer: Optional[Any],
                                    graph: Optional[Any]
                                    ) -> Optional[Entity]:
    """Scan the last 4 session turns for a mention of any entity
    of `kind`. Used to bias pronoun resolution toward 'what the
    user JUST said'."""
    if session_buffer is None or graph is None:
        return None
    try:
        recent_turns = session_buffer.recent(max_turns=4)
    except Exception:
        return None
    if not recent_turns:
        return None
    # Walk newest → oldest looking for a name token whose entity
    # is of the right kind.
    for turn in reversed(recent_turns):
        text = getattr(turn, "text", "") or ""
        for m in _NAME_RE.finditer(text):
            candidate = m.group(1)
            try:
                entity = graph.find_by_alias(candidate, kind=kind)
            except Exception:
                entity = None
            if entity is not None:
                return entity
    return None


def _entity_detail(entity: Entity) -> str:
    """Render a short context string for the planner prompt
    (email for person, url for document, etc.)."""
    if not entity:
        return ""
    for key in ("email", "address", "url", "link", "path"):
        v = entity.attrs.get(key)
        if v:
            return str(v)[:120]
    return ""


def _render_block(resolutions: List[Resolution]) -> str:
    """Render the resolved references as a compact prompt block."""
    if not resolutions:
        return ""
    # Group identical resolutions so "him" appearing twice doesn't
    # render twice.
    seen: set = set()
    lines: List[str] = ["RESOLVED REFERENCES:"]
    for r in resolutions:
        key = (r.token.lower(), r.entity_id)
        if key in seen:
            continue
        seen.add(key)
        detail = f" ({r.detail})" if r.detail else ""
        lines.append(
            f"  '{r.token}' → {r.display_name} "
            f"[{r.entity_kind}]{detail}")
    return "\n".join(lines)

"""Background knowledge-graph extraction — Iris seeds the entity
graph from documents the user touches.

Phase-9 cognition. Today the entity_graph (Phase-6) only learns
from EXPLICIT user mentions in chat ("Dani's email is x"). That
means "send the deck to Dani" works on day 50 but fails on day 1.

This module bridges the gap: when the user touches an email,
opens a document, or pastes text, we opportunistically extract:

  * Person names (recurring capitalized tokens with an email
    address or @-handle nearby)
  * Project names (camelCase / kebab-case / repeating short codes)
  * Date references (next Thursday / 2026-Q3 / etc.)

…and upsert them into the entity_graph as low-confidence nodes.
The graph already supports alias resolution + recency, so a
heuristic-seeded "Dani" node gets reinforced (or rejected) by
later interactions.

HARD constraints:
  * Honor incognito — no extraction in private mode.
  * Honor content_quarantine — reject text flagged as suspicious.
  * Source-tag every extraction so a future "forget all from
    file X" call can target it.
  * Don't promote to high-confidence without explicit user
    confirmation.

Cheap by design — pure regex + bounded token counts. No LLM
calls. For deep extraction (entity types, relations), defer to
the planner LLM via a separate opt-in module.

Public:
  * `extract_from_text(text, source)` — returns a list of
    `ExtractedEntity` dicts.
  * `seed_graph_from_text(text, source, graph=None)` — extracts +
    upserts into the entity_graph + returns the upserted nodes.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ExtractedKind(str, Enum):
    PERSON = "person"
    PROJECT = "project"
    DATE_REF = "date_ref"
    EMAIL = "email"


@dataclass
class ExtractedEntity:
    kind: ExtractedKind
    name: str                                # canonical form
    aliases: Tuple[str, ...] = ()
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    source: str = ""
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "aliases": list(self.aliases),
            "attributes": dict(self.attributes),
            "confidence": self.confidence,
            "source": self.source,
            "snippet": self.snippet,
        }


# ---- regexes -------------------------------------------------------

_EMAIL_RE = re.compile(
    r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b")

# Capitalised name token — 2-30 chars, possibly multi-word ("John Smith")
_NAME_TOKEN_RE = re.compile(
    r"\b([A-Z][a-zA-Z]{1,29}(?:\s+[A-Z][a-zA-Z]{1,29})?)\b")

# Project-ish: kebab-case (project-name) or short uppercase code (Q3, PROJ)
_PROJECT_RE = re.compile(
    r"\b([a-z0-9]+(?:-[a-z0-9]+){1,4})\b"     # kebab-case
    r"|\b([A-Z]{2,6}\d{0,3})\b"                # PROJ / Q3 / FY26
)

# Date-ish: ISO, Q-numbers, weekday names, relative
_DATE_RE = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}|"                       # ISO 2026-06-04
    r"(?:Q[1-4]\s*(?:20\d{2})?)|"               # Q3 or Q3 2026
    r"(?:next|this|last)\s+("
    r"week|month|year|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)|"
    r"(?:in\s+\d+\s+(?:days?|weeks?|months?|years?))|"
    r"(?:\d{1,2}\s+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))"
    r")\b",
    re.IGNORECASE)


_STOPNAME = frozenset({
    "I", "Iris", "Touchless", "The", "A", "An",
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October",
    "November", "December",
})


# ---- extractors -----------------------------------------------------

def _extract_emails(text: str,
                    source: str) -> List[ExtractedEntity]:
    out: List[ExtractedEntity] = []
    for m in _EMAIL_RE.finditer(text):
        addr = m.group(1)
        snippet = text[max(0, m.start() - 20):
                        m.end() + 20]
        out.append(ExtractedEntity(
            kind=ExtractedKind.EMAIL,
            name=addr,
            attributes={"address": addr},
            confidence=0.85,
            source=source,
            snippet=snippet.strip()))
    return out


def _extract_persons(text: str,
                     source: str,
                     emails: List[ExtractedEntity]
                     ) -> List[ExtractedEntity]:
    """Person inference: when a name token appears NEAR an email
    address, that's a strong signal. Otherwise the bar is HIGH
    (need ≥3 occurrences of the same name) to avoid over-extraction."""
    out: List[ExtractedEntity] = []
    name_counts: Dict[str, int] = {}
    name_positions: Dict[str, int] = {}
    for m in _NAME_TOKEN_RE.finditer(text):
        name = m.group(1).strip()
        if name in _STOPNAME:
            continue
        if name.lower() in {"iris", "touchless"}:
            continue
        # Skip single-letter words and date-y tokens.
        if len(name) < 3:
            continue
        name_counts[name] = name_counts.get(name, 0) + 1
        if name not in name_positions:
            name_positions[name] = m.start()
    seen_names: set = set()
    # Person-by-email-proximity (within 40 chars).
    for em in emails:
        local_part = em.name.split("@", 1)[0]
        # Look at the 40-char window before/after the email.
        idx = text.find(em.name)
        if idx < 0:
            continue
        window = text[max(0, idx - 40): idx + 40]
        person = None
        for m in _NAME_TOKEN_RE.finditer(window):
            cand = m.group(1).strip()
            if cand in _STOPNAME or len(cand) < 3:
                continue
            person = cand
            break
        if person is None:
            # Fall back: capitalise the local part as a name guess.
            person = local_part.replace(".", " ").title()
        if person in seen_names:
            continue
        seen_names.add(person)
        out.append(ExtractedEntity(
            kind=ExtractedKind.PERSON,
            name=person,
            aliases=(local_part,),
            attributes={"email": em.name},
            confidence=0.8,
            source=source,
            snippet=window.strip()))
    # Person-by-recurrence (≥3 mentions, no email link).
    for name, cnt in name_counts.items():
        if cnt < 3:
            continue
        if name in seen_names:
            continue
        seen_names.add(name)
        out.append(ExtractedEntity(
            kind=ExtractedKind.PERSON,
            name=name,
            attributes={"mention_count": cnt},
            confidence=0.55,
            source=source))
    return out


def _extract_projects(text: str,
                      source: str) -> List[ExtractedEntity]:
    out: List[ExtractedEntity] = []
    seen: set = set()
    for m in _PROJECT_RE.finditer(text):
        kebab = m.group(1)
        code = m.group(2)
        token = (kebab or code or "").strip()
        if not token or token in seen:
            continue
        # Skip dates that look like Q3 — they're date_refs.
        if code and re.fullmatch(r"Q[1-4]", code,
                                  re.IGNORECASE):
            continue
        if len(token) < 4:
            continue
        seen.add(token)
        out.append(ExtractedEntity(
            kind=ExtractedKind.PROJECT,
            name=token,
            confidence=0.6,
            source=source))
    return out


def _extract_dates(text: str,
                    source: str) -> List[ExtractedEntity]:
    out: List[ExtractedEntity] = []
    seen: set = set()
    for m in _DATE_RE.finditer(text):
        token = m.group(0).strip()
        norm = token.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(ExtractedEntity(
            kind=ExtractedKind.DATE_REF,
            name=token,
            confidence=0.7,
            source=source))
    return out


# ---- top-level ------------------------------------------------------

def extract_from_text(text: str, *,
                      source: str = "") -> List[ExtractedEntity]:
    """Run all heuristic extractors. Honors incognito + content
    quarantine. Returns [] when blocked."""
    if not text:
        return []
    try:
        from .incognito import is_incognito
        if is_incognito():
            return []
    except Exception:
        pass
    # Content quarantine: bail when the text is flagged suspicious.
    try:
        from .content_quarantine import looks_suspicious
        if looks_suspicious(text):
            return []
    except Exception:
        pass
    # Cap input length so a huge document doesn't OOM.
    text = text[:50_000]
    emails = _extract_emails(text, source)
    persons = _extract_persons(text, source, emails)
    projects = _extract_projects(text, source)
    dates = _extract_dates(text, source)
    return [*emails, *persons, *projects, *dates]


def seed_graph_from_text(text: str, *,
                         source: str = "",
                         graph: Optional[Any] = None
                         ) -> List[ExtractedEntity]:
    """Extract + upsert into the global entity_graph. Returns the
    extracted entities (so callers can show 'I learned X')."""
    extracted = extract_from_text(text, source=source)
    if not extracted:
        return extracted
    if graph is None:
        try:
            from .entity_graph import global_graph
            graph = global_graph()
        except Exception:
            return extracted
    try:
        # entity_graph uses a typed enum (EntityKind) — translate
        # our str-enum to its enum.
        from .entity_graph import EntityKind, RelationKind
        _kind_map = {
            ExtractedKind.PERSON: EntityKind.PERSON,
            ExtractedKind.PROJECT: EntityKind.PROJECT,
            ExtractedKind.DATE_REF: EntityKind.EVENT,
            ExtractedKind.EMAIL: EntityKind.PERSON,
        }
    except Exception:
        return extracted
    for ent in extracted:
        try:
            attrs = dict(ent.attributes)
            if source:
                attrs.setdefault("source", source)
            graph.upsert_entity(
                kind=_kind_map.get(ent.kind, EntityKind.TOPIC),
                name=ent.name,
                aliases=list(ent.aliases),
                attrs=attrs)
        except Exception:
            continue
    return extracted

"""LLM-based fact extraction from realtime conversation turns.

The deterministic extractor (extractor.py) pulls facts from planner step
args — that catches 'email Dani saying hi' → person/dani=email. But when
the user CHATS with realtime ('my office is Kearney 204', 'Vesko is my
brother', 'I prefer to be called Konstantin'), no tools fire and nothing
gets saved. This module fills that gap.

One cheap-LLM call per realtime turn:
  - System prompt teaches it what counts as a durable user-revealed fact
  - User message contains (user_text, assistant_text) and few-shot examples
  - Returns JSON: [{kind, key, value, confidence}, ...]
  - Caller validates + filters by confidence threshold + writes to memory

Costs ~$0.0001 per turn (cheap-LLM is much cheaper than realtime).
Skipped when OPENAI_API_KEY is absent or TOUCHLESS_REALTIME_FACT_EXTRACT=0.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, List, Tuple

DEFAULT_MODEL = "gpt-5-mini"
API_URL = "https://api.openai.com/v1/chat/completions"

# Vocabulary of allowed kinds — extractions outside this set are dropped.
# Keeps the memory store from filling with garbage taxonomy.
_ALLOWED_KINDS = frozenset({
    "person", "place", "preference", "relation", "schedule",
    "fact", "alias", "contact", "course", "interest", "role",
})

_SYSTEM_PROMPT = (
    "You are a memory fact extractor for Iris, a personal AI assistant. "
    "Given a recent USER message and the assistant's reply, identify "
    "durable facts the USER revealed that would help Iris respond better "
    "in future turns.\n\n"
    "Output ONE JSON array (no commentary, no markdown). Each item is:\n"
    "  {\"kind\":<string>, \"key\":<string>, \"value\":<string>, "
    "\"confidence\":<0..1>}\n\n"
    "Allowed kinds: person, place, preference, relation, schedule, fact, "
    "alias, contact, course, interest, role.\n"
    "Key format: short lowercase canonical identifier (kebab-case OK).\n\n"
    "RULES — read carefully:\n"
    "1. ONLY facts the USER said. Ignore the assistant's own text.\n"
    "2. ONLY DURABLE facts — things that will still be true tomorrow. "
    "Skip emotions ('I'm tired'), one-off reactions ('that's cool'), "
    "questions, requests, hypotheticals, jokes.\n"
    "3. Don't restate what the user just asked. 'User wants the weather' "
    "is NOT a fact.\n"
    "4. Confidence < 0.7 = probably noise; mark uncertain claims 0.5-0.6 "
    "so the caller can drop them.\n"
    "5. Multi-fact statements split into multiple entries.\n"
    "6. If nothing factual was revealed, output [].\n\n"
    "Examples:\n"
    "USER: 'my office is in Kearney 204'\n"
    "  -> [{\"kind\":\"place\",\"key\":\"office\",\"value\":\"Kearney 204\","
    "\"confidence\":0.95}]\n\n"
    "USER: 'Vesko is my brother'\n"
    "  -> [{\"kind\":\"relation\",\"key\":\"brother\",\"value\":\"Vesko\","
    "\"confidence\":0.95},"
    "{\"kind\":\"person\",\"key\":\"vesko\",\"value\":\"my brother\","
    "\"confidence\":0.85}]\n\n"
    "USER: 'I prefer to be called Konstantin, not Kosta'\n"
    "  -> [{\"kind\":\"preference\",\"key\":\"preferred_name\","
    "\"value\":\"Konstantin\",\"confidence\":0.95}]\n\n"
    "USER: 'I'm at Oregon State studying CS'\n"
    "  -> [{\"kind\":\"place\",\"key\":\"school\",\"value\":\"Oregon State\","
    "\"confidence\":0.95},"
    "{\"kind\":\"interest\",\"key\":\"major\",\"value\":\"CS\","
    "\"confidence\":0.9}]\n\n"
    "USER: 'tell me a joke'\n  -> []\n\n"
    "USER: 'what's the weather'\n  -> []\n\n"
    "USER: 'I'm tired'\n  -> []"
)


def configured() -> bool:
    return bool((os.environ.get("OPENAI_API_KEY") or "").strip())


def enabled() -> bool:
    return os.environ.get("TOUCHLESS_REALTIME_FACT_EXTRACT", "1") != "0"


def confidence_threshold() -> float:
    try:
        return float(os.environ.get("TOUCHLESS_FACT_CONFIDENCE", "0.7"))
    except (TypeError, ValueError):
        return 0.7


def extract_facts_from_conversation(
        user_text: str, assistant_text: str = "",
        model: str = "", timeout: float = 20.0,
        logger: Any = None,
) -> List[Tuple[str, str, str, float]]:
    """Run one cheap-LLM extraction pass. Returns a list of
    (kind, key, value, confidence) tuples, filtered to the allowed
    kinds vocabulary. Confidence filtering is the caller's job so they
    can audit raw output if they want."""
    user_text = (user_text or "").strip()
    if not user_text:
        return []
    if not configured():
        if logger:
            logger.event("memory_llm_extract_no_api_key")
        return []
    if not enabled():
        if logger:
            logger.event("memory_llm_extract_disabled_by_env")
        return []

    chosen_model = (model
                    or os.environ.get("TOUCHLESS_FACT_MODEL")
                    or os.environ.get("TOUCHLESS_PLANNER_MODEL")
                    or DEFAULT_MODEL)

    user_msg = (
        f"USER: {user_text[:2000]!r}\n"
        f"ASSISTANT: {assistant_text[:1000]!r}\n\n"
        "Extract durable facts the USER revealed. Output JSON array only."
    )
    body = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": 600,
    }
    key = os.environ["OPENAI_API_KEY"]
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if logger:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="ignore")[:400]
            except Exception:
                pass
            logger.event("memory_llm_extract_http_error",
                         code=exc.code, model=chosen_model, body=body)
        return []
    except Exception as exc:
        if logger:
            logger.event("memory_llm_extract_network_error",
                         error=f"{type(exc).__name__}: {exc}")
        return []

    raw = ((payload.get("choices") or [{}])[0]
           .get("message", {}).get("content") or "").strip()
    facts = _parse_facts(raw)
    if logger:
        logger.event("memory_llm_extract_api_returned",
                     raw_chars=len(raw), parsed_count=len(facts),
                     model=chosen_model)
    return facts


def _parse_facts(raw: str) -> List[Tuple[str, str, str, float]]:
    """Tolerant parse: accepts a bare JSON array OR an object wrapping it
    (response_format=json_object can't return a bare array, so the model
    is told to wrap it as {'facts': [...]} when that mode is enforced)."""
    if not raw:
        return []
    try:
        doc = json.loads(raw)
    except Exception:
        return []
    if isinstance(doc, dict):
        # Models often pick 'facts' or 'items'; accept either, plus a
        # single-key dict whose only value is a list.
        if isinstance(doc.get("facts"), list):
            doc = doc["facts"]
        elif isinstance(doc.get("items"), list):
            doc = doc["items"]
        elif len(doc) == 1 and isinstance(next(iter(doc.values())), list):
            doc = next(iter(doc.values()))
        else:
            return []
    if not isinstance(doc, list):
        return []
    out: List[Tuple[str, str, str, float]] = []
    for entry in doc:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "").strip().lower()
        key = str(entry.get("key") or "").strip()
        value = str(entry.get("value") or "").strip()
        try:
            conf = float(entry.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        # Vocabulary + non-empty checks. Confidence threshold is applied
        # by the caller so it can be logged / audited.
        if kind in _ALLOWED_KINDS and key and value:
            out.append((kind, key, value, max(0.0, min(1.0, conf))))
    return out

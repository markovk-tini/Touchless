"""Extract semantic facts from a planner-handled turn.

A "fact" is a 4-tuple: (kind, key, value, source). Examples:
  ("person", "dani", "dani@mangollc.org", "outlook_compose arg")
  ("file",   "budget.xlsx", "C:/Users/k/Documents/budget.xlsx", "drive_upload arg")
  ("preference", "default_email", "dani@x", "user said")

This pass is intentionally conservative — only fact patterns we KNOW
are reliable get extracted, so memory doesn't fill up with garbage.
Future passes can add fuzzier extractors (LLM-based) layered on top.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Tools whose args carry a recipient identifier we want to remember.
_RECIPIENT_TOOLS = {
    "outlook_compose": "recipient",
    "outlook_send": "recipient",
    "ms_mail_send": "to",
    "gmail_send": "to",
    "gmail_compose": "to",
    "teams_send": "user",
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# "email <name>" / "tell <name>" / "for <name>" — best-effort name capture.
_NAME_FROM_TEXT_RE = re.compile(
    r"\b(?:email|to|for|tell|message|send)\s+([A-Z][a-zA-Z]+)\b"
)


def extract_facts(user_text: str, steps: List[Any],
                  results: List[Any]) -> List[Tuple[str, str, str, Optional[str]]]:
    """Return the list of (kind, key, value, source) facts for this turn.

    Inputs are duck-typed — `step.tool`, `step.args`, `result.output` are
    enough. Designed so the orchestrator can call this without importing
    the planner package.
    """
    facts: List[Tuple[str, str, str, Optional[str]]] = []
    text_names = _names_from_text(user_text or "")

    for step, result in zip(steps or [], results or []):
        tool = getattr(step, "tool", "") or ""
        args = getattr(step, "args", {}) or {}
        out = getattr(result, "output", {}) or {}

        # 1. Person -> email from recipient-style tools.
        recipient_field = _RECIPIENT_TOOLS.get(tool)
        if recipient_field:
            recipient = str(args.get(recipient_field, "")).strip()
            email = _first_email(recipient) or _first_email(str(args.get("to", "")))
            if email:
                # Prefer a name captured from the user's original text (so
                # "email Dani saying hi" learns Dani -> dani@x), else use
                # the local-part as a fallback handle.
                names_to_link = list(text_names) or [email.split("@", 1)[0]]
                for name in names_to_link:
                    facts.append(("person", name.lower(), email, f"{tool} arg"))

        # 2. Person -> email when a lookup-style tool RETURNS an email.
        if isinstance(out, dict):
            ret_email = _first_email(str(out.get("email", "")))
            ret_name = str(out.get("name", "")).strip()
            if ret_email and (ret_name or text_names):
                for name in [ret_name] if ret_name else list(text_names):
                    if name:
                        facts.append(
                            ("person", name.lower(), ret_email, f"{tool} result"))

        # 3. Created-artifact links (docs, sheets, slides, drive uploads).
        link = (out or {}).get("link") if isinstance(out, dict) else None
        title = args.get("title") or args.get("name") or args.get("filename")
        if link and title and isinstance(title, str):
            facts.append(("artifact", title.lower(), str(link), f"{tool} result"))

    # Dedup while preserving order.
    seen: set = set()
    dedup: List[Tuple[str, str, str, Optional[str]]] = []
    for f in facts:
        sig = (f[0], f[1], f[2])
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append(f)
    return dedup


# ---- helpers --------------------------------------------------------------
def _first_email(s: str) -> Optional[str]:
    m = _EMAIL_RE.search(s or "")
    return m.group(0) if m else None


def _names_from_text(text: str) -> List[str]:
    """Best-effort proper-name extraction from the user's request. Catches
    'email Dani', 'message Sarah', etc. Misses lowercase names (acceptable
    — they look like common nouns and would be too noisy)."""
    seen = set()
    out: List[str] = []
    for m in _NAME_FROM_TEXT_RE.finditer(text):
        name = m.group(1)
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(name)
    return out

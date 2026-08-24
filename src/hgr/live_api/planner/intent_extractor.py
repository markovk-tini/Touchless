"""Iris planner Tier-1.5 — local-LLM intent extractor.

Runs AFTER `Classifier.classify_chain()` returns None (or after
`looks_multi_action` nulls out a single-tool hit) and BEFORE the Tier-2
LLM planner. Uses the local Ollama model to convert one natural-
language utterance into ONE strictly-shaped `Step(tool, args)` from a
tight allow-list of connector tools. Returns None whenever it isn't
100% confident, so the request falls through to Tier-2 unchanged.

Design goals:
- Zero cost when Ollama isn't running (fast-fail on `is_available`).
- Never dispatch a mangled call: strict allow-list + arg-shape gate +
  per-tool bounds + substring sanity vs. the raw utterance.
- Cache identical utterances (positive AND negative) so command
  repetition inside a session pays no LLM cost the second time.

Public surface:
    class ExtractedIntent
    class IntentExtractor:
        __init__(registry, logger=None, cache_size=256, model=None)
        try_extract(text: str) -> Optional[Step]

Author: Konstantin Markov
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .plan import Step


# ---------- tool corpus (single source of truth) ----------------------

# Each entry declares:
#   hint          - one-line human-readable purpose (fed to the model)
#   required_args - {arg_name: type_key}
#   optional_args - {arg_name: type_key}
# Type keys are validated by `_coerce_value` below. Keeping the corpus
# small on purpose: only "power CRUD" tools where the whole intent
# lives in the args and where regex Tier-1 has phrasing-drift holes.
_TOOL_CORPUS: Dict[str, Dict[str, Any]] = {
    "sheets_create": {
        "hint": "create a new Google Sheet.",
        "required_args": {"title": "string"},
        "optional_args": {"rows": "rows"},
    },
    "gdocs_create": {
        "hint": "create a new Google Doc.",
        "required_args": {"title": "string"},
        "optional_args": {"text": "string"},
    },
    "slides_create": {
        "hint": "create a new Google Slides deck.",
        "required_args": {"title": "string"},
        "optional_args": {"heading": "string"},
    },
    "forms_create": {
        "hint": "create a Google Form.",
        "required_args": {"title": "string", "questions": "list"},
        "optional_args": {},
    },
    "drive_upload": {
        "hint": ("upload a LOCAL file path (stated by the user) to "
                 "Google Drive."),
        "required_args": {"path": "string"},
        "optional_args": {},
    },
    "photos_upload": {
        "hint": "upload a LOCAL file path to Google Photos.",
        "required_args": {"path": "string"},
        "optional_args": {},
    },
    "tasks_add": {
        "hint": "add an item to Google Tasks.",
        "required_args": {"title": "string"},
        "optional_args": {},
    },
    "tasks_complete": {
        "hint": "mark a Google Task as complete.",
        "required_args": {"title_match": "string"},
        "optional_args": {},
    },
    "tasks_delete": {
        "hint": "delete a Google Task.",
        "required_args": {"title_match": "string"},
        "optional_args": {},
    },
    "outlook_compose": {
        "hint": ("open a draft email in Outlook. recipient MUST be an "
                 "@ email address, not a name."),
        "required_args": {"recipient": "string", "body": "string"},
        "optional_args": {"subject": "string"},
    },
    "discord_mute": {
        "hint": "set Discord mic mute state.",
        "required_args": {"muted": "boolean"},
        "optional_args": {},
    },
    "discord_deafen": {
        "hint": "set Discord deafen state.",
        "required_args": {"deafened": "boolean"},
        "optional_args": {},
    },
    "volume_set": {
        "hint": "set system audio volume 0-100.",
        "required_args": {"percent": "integer"},
        "optional_args": {},
    },
    "contacts_create": {
        "hint": "create a Google contact.",
        "required_args": {"given_name": "string"},
        "optional_args": {
            "family_name": "string",
            "email": "string",
            "phone": "string",
        },
    },
    "sheets_update_range": {
        "hint": "write cells into an existing sheet.",
        "required_args": {
            "sheet_name": "string",
            "range": "string",
            "values": "rows",
        },
        "optional_args": {},
    },
}


# Tools whose extracted args must include at least one value that
# appears in the raw utterance (case-insensitive substring). Guards
# against the model wholesale-hallucinating a plausible-looking call.
_SUBSTRING_SANITY_TOOLS = frozenset({
    "outlook_compose", "drive_upload", "photos_upload",
    "contacts_create",
})


_A1_RANGE_RE = re.compile(r"^[A-Z]+\d+(:[A-Z]+\d+)?$")
_EMAIL_HINT_RE = re.compile(r"@")
_FENCE_OPEN_RE = re.compile(r"^```(?:[a-zA-Z0-9_-]*)?\s*\n?")
_FENCE_CLOSE_RE = re.compile(r"\n?```\s*$")


# ---------- prompt assembly -------------------------------------------

def _render_tool_catalog() -> str:
    """Render `_TOOL_CORPUS` into the AVAILABLE TOOLS block the model
    sees. One source of truth: adding/removing a corpus entry updates
    both the validator AND the prompt in lockstep."""
    lines: List[str] = []
    for name, spec in _TOOL_CORPUS.items():
        lines.append(f"- {name}: {spec['hint']}")
        req = spec.get("required_args") or {}
        opt = spec.get("optional_args") or {}
        if req:
            parts = ", ".join(f"{k}({_type_label(v)})"
                              for k, v in req.items())
            lines.append(f"    required: {parts}")
        if opt:
            parts = ", ".join(f"{k}({_type_label(v)})"
                              for k, v in opt.items())
            lines.append(f"    optional: {parts}")
    return "\n".join(lines)


def _type_label(t: str) -> str:
    if t == "rows":
        return "array of arrays of strings"
    if t == "list":
        return "array"
    return t


_SYSTEM_PROMPT = """You extract ONE structured tool call from the user's utterance. You do
NOT converse and do NOT explain. Output STRICT JSON matching the
response schema below, with NO markdown fences and NO prose.

Rules:
1. Choose EXACTLY ONE tool from AVAILABLE TOOLS, OR return
   {"tool": null} when the utterance doesn't cleanly map to a listed
   tool. Prefer {"tool": null} over guessing.
2. Fill args ONLY from information PRESENT in the utterance. Do NOT
   invent titles, recipients, times, file paths, or numbers.
3. If a required arg for the tool isn't stated, return {"tool": null}.
4. Match arg TYPES exactly:
     - string  -> a JSON string
     - integer -> a JSON number (no quotes)
     - boolean -> true / false (no quotes)
     - rows    -> array of arrays of strings
5. Preserve the user's original casing for titles / names / cell
   contents ("Q4 Budget", not "q4 budget").

AVAILABLE TOOLS:
__CATALOG__

RESPONSE SCHEMA:
{"tool": "<one of the names above>" | null, "args": {<key>: <value>, ...}}

EXAMPLES:
User: spin up a google sheet called Q4 Budget with a header row for name amount and date
JSON: {"tool":"sheets_create","args":{"title":"Q4 Budget","rows":[["name","amount","date"]]}}

User: toss a doc called Meeting Notes with hey everyone welcome as the opening line
JSON: {"tool":"gdocs_create","args":{"title":"Meeting Notes","text":"hey everyone welcome"}}

User: crank the volume up to 40
JSON: {"tool":"volume_set","args":{"percent":40}}

User: silence me on discord
JSON: {"tool":"discord_mute","args":{"muted":true}}

User: draft an email to dani@mangollc.org saying the Q3 report is done
JSON: {"tool":"outlook_compose","args":{"recipient":"dani@mangollc.org","body":"the Q3 report is done"}}

User: in the google sheet called testing change C1 to say Email
JSON: {"tool":"sheets_update_range","args":{"sheet_name":"testing","range":"C1","values":[["Email"]]}}

User: in my Q4 Budget sheet write Total in cell B10
JSON: {"tool":"sheets_update_range","args":{"sheet_name":"Q4 Budget","range":"B10","values":[["Total"]]}}

User: what's the weather in Boston tomorrow
JSON: {"tool":null}

User: email dani that Q3 report is done
JSON: {"tool":null}

User: summarize my unread emails
JSON: {"tool":null}
"""


def _build_system_prompt() -> str:
    """Assemble the system prompt (catalog + rules + examples). Runs
    once at module import via a module-level constant below."""
    return _SYSTEM_PROMPT.replace("__CATALOG__", _render_tool_catalog())


_SYSTEM_PROMPT_RENDERED = _build_system_prompt()


def _build_user_prompt(raw_utterance: str) -> str:
    return f"User: {raw_utterance}\nJSON:"


# ---------- normalization + cache key ---------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = _WHITESPACE_RE.sub(" ", t)
    return t.rstrip(".!?,;: ")


def _cache_key(text: str) -> str:
    return hashlib.sha1(
        _normalize(text).encode("utf-8", errors="replace")).hexdigest()


# ---------- type coercion --------------------------------------------

_BOOL_TRUE = {"true", "yes", "y", "1", "on"}
_BOOL_FALSE = {"false", "no", "n", "0", "off"}


def _coerce_value(value: Any, type_key: str) -> Tuple[bool, Any]:
    """Return (ok, coerced_value). ok=False means the value cannot
    reasonably be coerced to the requested type — validator treats
    this as a miss for required args, drops the arg for optional."""
    if value is None:
        return False, None
    if type_key == "string":
        try:
            if isinstance(value, bool):
                return True, ("true" if value else "false")
            if isinstance(value, (int, float)):
                return True, str(value)
            if isinstance(value, str):
                v = value.strip()
                return (bool(v), v)
        except Exception:
            return False, None
        return False, None
    if type_key == "integer":
        try:
            if isinstance(value, bool):
                return True, int(value)
            if isinstance(value, int):
                return True, value
            if isinstance(value, float):
                return True, int(value)
            if isinstance(value, str):
                s = value.strip()
                if s.startswith(("-", "+")):
                    body = s[1:]
                else:
                    body = s
                if body.isdigit():
                    return True, int(s)
                # accept decimals like "40.0"
                try:
                    return True, int(float(s))
                except Exception:
                    return False, None
        except Exception:
            return False, None
        return False, None
    if type_key == "boolean":
        if isinstance(value, bool):
            return True, value
        if isinstance(value, (int, float)):
            return True, bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in _BOOL_TRUE:
                return True, True
            if v in _BOOL_FALSE:
                return True, False
        return False, None
    if type_key == "list":
        if isinstance(value, list):
            return True, value
        return False, None
    if type_key == "rows":
        if not isinstance(value, list):
            return False, None
        out_rows: List[List[str]] = []
        for row in value:
            if not isinstance(row, list):
                return False, None
            out_row: List[str] = []
            for cell in row:
                if isinstance(cell, str):
                    out_row.append(cell)
                elif isinstance(cell, (int, float, bool)):
                    out_row.append(str(cell))
                else:
                    return False, None
            out_rows.append(out_row)
        return True, out_rows
    return False, None


# ---------- fence stripping + JSON parsing ---------------------------

def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` (or any language tag) if the model
    wrapped its answer against instructions."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = _FENCE_OPEN_RE.sub("", s, count=1)
        s = _FENCE_CLOSE_RE.sub("", s, count=1)
    return s.strip()


# ---------- data model -----------------------------------------------

@dataclass
class ExtractedIntent:
    """Debug/telemetry record of a single extractor call. NOT returned
    to the orchestrator on the happy path — the orchestrator wants a
    `Step`. This exists for tests and future observability."""
    tool: Optional[str]
    args: Dict[str, Any] = field(default_factory=dict)
    raw_json: str = ""
    source: str = "extractor"  # "extractor" | "cache"


# ---------- extractor ------------------------------------------------

class IntentExtractor:
    """Tier-1.5 local-LLM intent extractor. See module docstring."""

    def __init__(self, registry: Any, logger: Any = None,
                 cache_size: int = 256,
                 model: Optional[str] = None) -> None:
        self._registry = registry
        self._logger = logger
        self._cache_size = max(1, int(cache_size))
        self._model = model  # None -> connector's default
        self._cache: "OrderedDict[str, Optional[Step]]" = OrderedDict()

    def _cache_get(self, key: str) -> Tuple[bool, Optional[Step]]:
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return True, value
        return False, None

    def _cache_put(self, key: str, value: Optional[Step]) -> None:
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _log_hit(self, tool: str, elapsed_ms: int,
                 cache_hit: bool) -> None:
        if self._logger is not None:
            try:
                self._logger.event("intent_extractor_hit",
                                   tool=tool,
                                   elapsed_ms=elapsed_ms,
                                   cache_hit=cache_hit)
            except Exception:
                pass

    def _log_miss(self, reason: str) -> None:
        if self._logger is not None:
            try:
                self._logger.event("intent_extractor_miss",
                                   reason=reason)
            except Exception:
                pass

    def try_extract(self, text: str) -> Optional[Step]:
        """Return a validated `Step` or None. Fully optional — every
        failure mode short-circuits to None so Tier-2 sees the request
        unchanged."""
        t = (text or "").strip()
        if not t:
            return None

        registry = self._registry
        if registry is None:
            return None

        # 1) Fast-fail if Ollama isn't runnable right now.
        try:
            reason = registry.is_available("ollama_generate")
        except Exception:
            reason = "ollama not available"
        if reason:
            self._log_miss("ollama_unavailable")
            return None
        try:
            if not registry.handles_connector("ollama_generate"):
                self._log_miss("ollama_unavailable")
                return None
        except Exception:
            self._log_miss("ollama_unavailable")
            return None

        # 2) Cache probe.
        key = _cache_key(t)
        hit, cached = self._cache_get(key)
        if hit:
            if cached is None:
                self._log_miss("cache_negative")
            else:
                self._log_hit(cached.tool, elapsed_ms=0, cache_hit=True)
            return cached

        # 3) Build prompt + call the model.
        started = time.time()
        try:
            args = {
                "prompt": _build_user_prompt(t),
                "system": _SYSTEM_PROMPT_RENDERED,
            }
            if self._model:
                args["model"] = self._model
            result = registry.call("ollama_generate", args)
        except Exception as exc:
            if self._logger is not None:
                try:
                    self._logger.exception(
                        "intent_extractor_call_failed", exc)
                except Exception:
                    pass
            self._cache_put(key, None)
            self._log_miss("ollama_unavailable")
            return None

        if not isinstance(result, dict) or \
                str(result.get("status") or "").lower() != "ok":
            self._cache_put(key, None)
            self._log_miss("ollama_unavailable")
            return None

        text_out = result.get("text") or result.get("response") or ""
        if not isinstance(text_out, str) or not text_out.strip():
            self._cache_put(key, None)
            self._log_miss("json_parse")
            return None

        # 4) JSON parse (tolerate stray fences).
        raw = _strip_fences(text_out)
        try:
            payload = json.loads(raw)
        except Exception:
            self._cache_put(key, None)
            self._log_miss("json_parse")
            return None

        if not isinstance(payload, dict):
            self._cache_put(key, None)
            self._log_miss("json_parse")
            return None

        tool = payload.get("tool")
        if tool is None:
            self._cache_put(key, None)
            self._log_miss("tool_none")
            return None
        if not isinstance(tool, str):
            self._cache_put(key, None)
            self._log_miss("json_parse")
            return None

        # 5) Allowlist gate.
        spec = _TOOL_CORPUS.get(tool)
        if spec is None:
            self._cache_put(key, None)
            self._log_miss("not_in_allowlist")
            return None

        # 6) Runtime gate — the connector for this tool must still be
        # registered right now (guards against "extract for a tool the
        # user hasn't authed").
        try:
            handles = registry.handles_connector(tool)
        except Exception:
            handles = False
        if not handles:
            self._cache_put(key, None)
            self._log_miss("tool_unregistered")
            return None

        raw_args = payload.get("args") or {}
        if not isinstance(raw_args, dict):
            self._cache_put(key, None)
            self._log_miss("json_parse")
            return None

        # 7) Arg-shape gate.
        validated: Dict[str, Any] = {}
        for name, tkey in (spec.get("required_args") or {}).items():
            if name not in raw_args:
                self._cache_put(key, None)
                self._log_miss(f"missing_required_arg:{name}")
                return None
            ok, coerced = _coerce_value(raw_args[name], tkey)
            if not ok:
                self._cache_put(key, None)
                self._log_miss(f"missing_required_arg:{name}")
                return None
            # empty string / empty list still counts as missing
            if coerced == "" or coerced == []:
                self._cache_put(key, None)
                self._log_miss(f"missing_required_arg:{name}")
                return None
            validated[name] = coerced
        for name, tkey in (spec.get("optional_args") or {}).items():
            if name not in raw_args:
                continue
            ok, coerced = _coerce_value(raw_args[name], tkey)
            if not ok:
                continue
            if coerced == "" or coerced == []:
                continue
            validated[name] = coerced

        # 8) Per-tool bounds sanity.
        bounds_reason = _bounds_check(tool, validated)
        if bounds_reason:
            self._cache_put(key, None)
            self._log_miss(f"bounds:{bounds_reason}")
            return None

        # 9) Substring sanity vs. raw utterance.
        if tool in _SUBSTRING_SANITY_TOOLS:
            lower_utt = t.lower()
            if not _any_required_in_utterance(
                    tool, validated, lower_utt):
                self._cache_put(key, None)
                self._log_miss("sanity")
                return None

        step = Step(
            tool=tool,
            args=validated,
            layer="connector",
            description=f"extracted: {tool}",
        )
        self._cache_put(key, step)
        elapsed_ms = int((time.time() - started) * 1000)
        self._log_hit(tool, elapsed_ms=elapsed_ms, cache_hit=False)
        return step


# ---------- helpers used by try_extract ------------------------------

def _bounds_check(tool: str,
                  args: Dict[str, Any]) -> Optional[str]:
    """Return None when bounds pass, else a short reason string
    (the arg name that failed)."""
    if tool == "volume_set":
        pct = args.get("percent")
        if not isinstance(pct, int):
            return "percent"
        if pct < 0 or pct > 100:
            return "percent"
    if tool == "outlook_compose":
        rcpt = args.get("recipient") or ""
        if not isinstance(rcpt, str) or "@" not in rcpt \
                or "." not in rcpt.rsplit("@", 1)[-1]:
            return "recipient"
    if tool == "sheets_update_range":
        rng = args.get("range") or ""
        if not isinstance(rng, str) or \
                not _A1_RANGE_RE.match(rng.upper()):
            return "range"
    return None


def _any_required_in_utterance(tool: str, args: Dict[str, Any],
                                lower_utt: str) -> bool:
    """True when at least one required-arg VALUE appears (case-insens.)
    in the raw utterance. Protects against wholesale hallucination for
    tools that name external targets (recipients, file paths, contact
    names). For values that are structured (lists, dicts) we recurse
    into strings inside them."""
    spec = _TOOL_CORPUS.get(tool)
    if spec is None:
        return False
    required = spec.get("required_args") or {}
    for name in required:
        val = args.get(name)
        if _value_appears_in(val, lower_utt):
            return True
    return False


def _value_appears_in(val: Any, lower_utt: str) -> bool:
    if isinstance(val, str):
        v = val.strip().lower()
        if not v:
            return False
        # for email addresses, the local part alone is enough evidence.
        if "@" in v:
            local = v.split("@", 1)[0]
            if local and local in lower_utt:
                return True
        return v in lower_utt
    if isinstance(val, (int, float, bool)):
        return str(val).lower() in lower_utt
    if isinstance(val, list):
        for item in val:
            if _value_appears_in(item, lower_utt):
                return True
    return False

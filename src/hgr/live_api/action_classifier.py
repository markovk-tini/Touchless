"""Action classifier — pure rule-based categorisation of agent actions.

Two entry points, both deterministic / idempotent and **side-effect free**:

  * :func:`classify_tool_action` — takes ``(tool_name, args, result)`` from
    :class:`hgr.live_api.tool_executor.ToolExecutor.execute` and decides what
    *kind* of action just happened (file touch, app launch, project open, …).
  * :func:`classify_user_text` — takes raw user utterance and decides whether
    it carries a long-lived fact / preference / new project mention.

Both return a dict::

    {
        "kind":       <one of ACTION_KINDS>,
        "target":     str | None,   # primary subject (path, url, app, fact)
        "payload":    dict,         # extra structured context for downstream
        "confidence": float,        # 0..1, downstream may threshold @ 0.5
    }

Designed to live BELOW the memory + tool-executor layers — no I/O, no LLM,
no Qt, no logging. Safe to call from any thread.

Constraints
-----------
* Standalone module — only stdlib imports.
* Idempotent: same inputs always yield the same output.
* Conservative: when in doubt, return ``kind="unknown"`` with confidence 0.0
  rather than guessing.
* The whitelist of recognised kinds is intentionally narrow; downstream
  consumers (memory router, pattern learner) can branch on ``kind``.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Tuple


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

ACTION_KINDS: Tuple[str, ...] = (
    "file_touch",
    "fact",
    "preference",
    "episode",
    "tool_use",
    "new_project",
    "pattern_seed",
    "unknown",
)


# Tool-name groupings — keep these as frozensets so membership checks are O(1)
# and the values are immutable / hashable.

# Tools that read or mutate a single file path.
_FILE_TOUCH_TOOLS: frozenset = frozenset({
    "write_file",
    "append_file",
    "create_file",
    "read_file",
    "open_file",
    "move_file",
    "rename_file",
    "delete_file",
    "copy_file",
    "edit_file",
})

# Tools that open something in an external app — treated as "new_project" when
# a folder is involved (IDE-style open), otherwise plain tool_use.
_EDITOR_TOOLS: frozenset = frozenset({
    "open_in_editor",
    "open_folder",
    "open_project",
})

# Tools that launch an external app or URL — recorded as tool_use unless the
# target matches a pattern_seed heuristic (learning content).
_APP_LAUNCH_TOOLS: frozenset = frozenset({
    "open_app",
    "launch_app",
    "run_app",
})
_URL_TOOLS: frozenset = frozenset({
    "open_url",
    "open_browser",
    "browse",
})

# Candidate arg keys (checked in order) that hold the primary path/target.
_PATH_KEYS: Tuple[str, ...] = (
    "path",
    "file_path",
    "filepath",
    "relative_path",
    "full_path",
    "target_path",
    "source_path",
    "destination_path",
    "dest_path",
    "src",
    "dst",
)
_FOLDER_KEYS: Tuple[str, ...] = (
    "folder_path",
    "folder",
    "directory",
    "dir",
    "project_path",
    "project_dir",
    "root",
)
_APP_KEYS: Tuple[str, ...] = ("app", "app_name", "name", "program", "executable")
_URL_KEYS: Tuple[str, ...] = ("url", "link", "href", "url_or_query", "query")


# Default confidence floors per kind — calibrated conservatively so the
# downstream router can keep its single 0.5 threshold without per-kind tuning.
_CONF_FILE_TOUCH = 0.95
_CONF_NEW_PROJECT_TOOL = 0.9
_CONF_APP_LAUNCH = 0.85
_CONF_URL_OPEN = 0.8
_CONF_PATTERN_SEED = 0.75
_CONF_USER_FACT = 0.85
_CONF_USER_PREFERENCE = 0.9
_CONF_USER_NEW_PROJECT = 0.8
_CONF_USER_EPISODE = 0.6


# Regexes for user-text routing. Compiled once at import.
_RE_REMEMBER = re.compile(r"^\s*(?:please\s+)?remember(?:\s+that)?\s+(.+)$", re.IGNORECASE)
_RE_MY_X_IS_Y = re.compile(r"\bmy\s+([a-z][a-z0-9 _-]{0,40}?)\s+is\s+(.+)$", re.IGNORECASE)
_RE_I_PREFER = re.compile(r"^\s*i\s+prefer\b(.+)$", re.IGNORECASE)
_RE_I_ALWAYS = re.compile(r"^\s*i\s+always\b(.+)$", re.IGNORECASE)
_RE_I_NEVER = re.compile(r"^\s*i\s+never\b(.+)$", re.IGNORECASE)
_RE_I_DONT_LIKE = re.compile(r"^\s*i\s+(?:don'?t|do\s+not)\s+like\b(.+)$", re.IGNORECASE)
# Windows-style absolute path: c:\foo, D:/bar, etc. Greedy enough to absorb
# spaces inside the path (e.g. ``c:/HGR App v1.0.0/src/...``) by anchoring on
# subsequent ``/`` / ``\`` / alphanumeric / dash / dot / underscore segments
# separated by single spaces. Trailing punctuation is stripped downstream.
_RE_WIN_PATH = re.compile(
    r"\b([a-zA-Z]:[\\/](?:[^\\/\s\"'<>|?*]+(?:\s+[^\\/\s\"'<>|?*]+)*)"
    r"(?:[\\/](?:[^\\/\s\"'<>|?*]+(?:\s+[^\\/\s\"'<>|?*]+)*))*)"
)

# Heuristic substrings hinting that an opened URL is "learning content" —
# downstream pattern-seed consumers want to keep an eye on these even
# though we are NOT certain.
_PATTERN_SEED_HOSTS: Tuple[str, ...] = (
    "youtube.com/watch",
    "youtu.be/",
    "coursera.org",
    "udemy.com",
    "edx.org",
    "khanacademy.org",
    "docs.python.org",
    "stackoverflow.com",
    "github.com",
    "developer.mozilla.org",
    "arxiv.org",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_tool_action(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Categorise a tool invocation.

    Parameters
    ----------
    tool_name:
        Canonical tool name as registered with the ToolExecutor.
    args:
        Validated arg dict the tool was called with (post ``validate_args``).
    result:
        The tool's result dict. If ``status`` is not ``"ok"`` we always
        return ``unknown`` — failed actions should not feed memory.

    Returns
    -------
    dict
        ``{kind, target, payload, confidence}``. Never raises.
    """
    args = args or {}
    result = result or {}

    name = (tool_name or "").strip()
    if not name:
        return _unknown(target=None, reason="empty_tool_name")

    # Refuse to classify failed / pending / unknown-status tool calls.
    status = str(result.get("status", "") or "").lower()
    if status and status != "ok":
        return _unknown(
            target=name,
            reason=f"status_{status}",
            payload={"tool": name, "status": status},
        )

    # ---- file-touch tools -----------------------------------------------
    if name in _FILE_TOUCH_TOOLS:
        path = _first_present(args, _PATH_KEYS)
        if path:
            payload: Dict[str, Any] = {
                "tool": name,
                "base_dir": args.get("base_dir"),
            }
            size = _safe_int(result.get("bytes") or result.get("size"))
            if size is not None:
                payload["size_bytes"] = size
            return {
                "kind": "file_touch",
                "target": str(path),
                "payload": payload,
                "confidence": _CONF_FILE_TOUCH,
            }
        # File-tool with no recognisable path arg — still a tool_use so the
        # caller can log it, but at low confidence.
        return {
            "kind": "tool_use",
            "target": name,
            "payload": {"tool": name, "reason": "no_path_arg"},
            "confidence": 0.4,
        }

    # ---- editor / project-open tools ------------------------------------
    if name in _EDITOR_TOOLS:
        folder = _first_present(args, _FOLDER_KEYS)
        if folder:
            return {
                "kind": "new_project",
                "target": str(folder),
                "payload": {
                    "tool": name,
                    "editor": args.get("editor"),
                    "file": args.get("file_to_open") or _first_present(args, _PATH_KEYS),
                },
                "confidence": _CONF_NEW_PROJECT_TOOL,
            }
        # Editor opened on a bare file — file_touch is the right kind.
        path = _first_present(args, _PATH_KEYS)
        if path:
            return {
                "kind": "file_touch",
                "target": str(path),
                "payload": {"tool": name, "editor": args.get("editor")},
                "confidence": _CONF_FILE_TOUCH,
            }
        return {
            "kind": "tool_use",
            "target": name,
            "payload": {"tool": name},
            "confidence": 0.4,
        }

    # ---- app launches ---------------------------------------------------
    if name in _APP_LAUNCH_TOOLS:
        app = _first_present(args, _APP_KEYS)
        target = str(app) if app else name
        return {
            "kind": "tool_use",
            "target": target,
            "payload": {"tool": name, "app": app},
            "confidence": _CONF_APP_LAUNCH if app else 0.5,
        }

    # ---- URL opens ------------------------------------------------------
    if name in _URL_TOOLS:
        url = _first_present(args, _URL_KEYS)
        if url and _looks_like_pattern_seed(str(url)):
            return {
                "kind": "pattern_seed",
                "target": str(url),
                "payload": {"tool": name},
                "confidence": _CONF_PATTERN_SEED,
            }
        return {
            "kind": "tool_use",
            "target": str(url) if url else name,
            "payload": {"tool": name, "url": url},
            "confidence": _CONF_URL_OPEN if url else 0.5,
        }

    # ---- everything else: still useful as tool_use ----------------------
    return {
        "kind": "tool_use",
        "target": name,
        "payload": {"tool": name},
        "confidence": 0.5,
    }


def classify_user_text(
    user_text: Optional[str],
    known_project_roots: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Categorise a raw user utterance.

    Parameters
    ----------
    user_text:
        The exact transcript or typed text the user produced. Empty / None
        inputs always classify as ``unknown`` with confidence 0.
    known_project_roots:
        Optional iterable of absolute paths that already correspond to
        registered projects. When a path mentioned by the user is NOT
        in this set we tentatively classify the utterance as a
        ``new_project`` mention. Comparison is case-insensitive with
        trailing-slash insensitivity (Windows convention).

    Returns
    -------
    dict
        ``{kind, target, payload, confidence}``. Never raises.
    """
    if not user_text or not isinstance(user_text, str):
        return _unknown(target=None, reason="empty_text")
    text = user_text.strip()
    if not text:
        return _unknown(target=None, reason="empty_text")

    # 1. Preferences win over generic facts ("I prefer dark mode" should not
    #    be classified as 'fact' just because it contains "is").
    for rx, polarity in (
        (_RE_I_PREFER, "prefer"),
        (_RE_I_ALWAYS, "always"),
        (_RE_I_NEVER, "never"),
        (_RE_I_DONT_LIKE, "dislike"),
    ):
        m = rx.match(text)
        if m:
            tail = m.group(1).strip(" .,!?:;")
            return {
                "kind": "preference",
                "target": tail or None,
                "payload": {"polarity": polarity, "text": text},
                "confidence": _CONF_USER_PREFERENCE,
            }

    # 2. Explicit "remember ..." command.
    m = _RE_REMEMBER.match(text)
    if m:
        return {
            "kind": "fact",
            "target": m.group(1).strip(" .,!?:;") or None,
            "payload": {"source": "remember", "text": text},
            "confidence": _CONF_USER_FACT,
        }

    # 3. New-project mention via filesystem path.
    path_match = _RE_WIN_PATH.search(text)
    if path_match:
        path = _normalize_path(path_match.group(1))
        roots_norm = {
            _normalize_path(r)
            for r in (known_project_roots or [])
            if isinstance(r, str) and r
        }
        if path not in roots_norm and not any(
            path.startswith(r.rstrip("/") + "/") for r in roots_norm
        ):
            return {
                "kind": "new_project",
                "target": path,
                "payload": {"text": text},
                "confidence": _CONF_USER_NEW_PROJECT,
            }
        # Path mentions an EXISTING project — treat as episodic context.
        return {
            "kind": "episode",
            "target": path,
            "payload": {"text": text, "known_project": True},
            "confidence": _CONF_USER_EPISODE,
        }

    # 4. "My X is Y" — generic fact extraction. Check AFTER preferences so
    #    "I prefer ..." does not fall in here, and AFTER 'remember' so the
    #    explicit form wins on confidence.
    m = _RE_MY_X_IS_Y.search(text)
    if m:
        attr = m.group(1).strip().lower()
        value = m.group(2).strip(" .,!?:;")
        return {
            "kind": "fact",
            "target": attr or None,
            "payload": {"attribute": attr, "value": value, "text": text},
            "confidence": _CONF_USER_FACT,
        }

    return _unknown(target=None, reason="no_rule_matched")


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------

def _unknown(
    *,
    target: Optional[str],
    reason: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "kind": "unknown",
        "target": target,
        "payload": dict(payload or {}, reason=reason),
        "confidence": 0.0,
    }


def _first_present(args: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        v = args.get(k)
        if v not in (None, "", []):
            return v
    return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _looks_like_pattern_seed(url: str) -> bool:
    lower = url.lower()
    return any(host in lower for host in _PATTERN_SEED_HOSTS)


def _normalize_path(p: str) -> str:
    """Lowercase, forward-slash, trailing-slash trimmed.

    Used to compare user-typed paths against registered project roots
    without false negatives from Windows path-style differences.
    """
    s = (p or "").strip().strip("\"'")
    s = s.replace("\\", "/")
    s = s.rstrip("/")
    return s.lower()

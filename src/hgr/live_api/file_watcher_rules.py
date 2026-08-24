"""File-watcher rules engine.

Phase-3. Users want Iris to react to file-system changes:
  * "When I drop a PDF in ~/Downloads/Invoices, file it in Notion."
  * "When a screenshot lands in ~/Pictures/Screenshots, OCR it."
  * "When ~/Desktop/inbox.txt is touched, parse it as a journal."
  * "When my CSV in ~/Documents/Sales is overwritten, refresh
    the dashboard sheet."

This module owns the RULES — what to match + what to do — and the
EVENT NORMALIZATION layer that abstracts over Win32 / inotify /
FSEvents. The actual watcher loop (watchdog / win32 ReadDirectory
ChangesW) is plugged in by the Sentinel watcher.

Design:
  * `Rule` is a dataclass with match criteria and an "action" id
    that maps to a registered Iris tool / pseudo-tool.
  * `RulesEngine.match(event)` returns the list of rules that fire
    for a given event.
  * `RulesEngine.dispatch(event, dispatcher)` matches AND calls the
    dispatcher with each matched rule.
  * Persistence is via SQLite at %LOCALAPPDATA%\\Touchless\\
    private\\file_rules.db so rules survive restart.
  * Rate-limiting per rule: a "thundering rename" (IDE saving
    file 50x in 1s) won't fire 50 actions.

Author: Konstantin Markov
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class FsEventKind(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"
    RENAMED = "renamed"


@dataclass
class FsEvent:
    """One normalized file-system event."""
    kind: FsEventKind
    path: str                  # absolute path
    src_path: str = ""         # for MOVED/RENAMED, the old path
    ts: float = field(default_factory=time.time)
    size_bytes: int = 0


@dataclass
class Rule:
    """One user-defined file-watcher rule.

    `path_glob`: shell-style glob ('**/*.pdf' matches recursively).
    `kinds`: list of FsEventKind values; empty means all.
    `min_size_bytes` / `max_size_bytes`: 0 = unbounded.
    `action`: tool name to invoke when this rule fires
              (e.g., 'notion_create_page').
    `action_args`: arguments forwarded to the tool. Placeholders
                   `{path}` / `{filename}` / `{ext}` / `{kind}`
                   are substituted from the event.
    `rate_limit_sec`: minimum seconds between fires for THIS rule.
    `enabled`: rules can be paused without deletion.
    """
    id: str
    name: str
    path_glob: str
    action: str
    action_args: Dict[str, Any] = field(default_factory=dict)
    kinds: List[str] = field(default_factory=list)
    min_size_bytes: int = 0
    max_size_bytes: int = 0
    rate_limit_sec: float = 2.0
    require_path_contains: str = ""
    enabled: bool = True
    created_at: float = field(default_factory=time.time)


def _default_db() -> Path:
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(local) / "Touchless" / "private" / "file_rules.db"


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


class RulesEngine:
    """Matches events against rules + persists rules to SQLite."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS rules (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        path_glob    TEXT NOT NULL,
        kinds_json   TEXT NOT NULL DEFAULT '[]',
        action       TEXT NOT NULL,
        action_args  TEXT NOT NULL DEFAULT '{}',
        min_size     INTEGER NOT NULL DEFAULT 0,
        max_size     INTEGER NOT NULL DEFAULT 0,
        rate_limit   REAL NOT NULL DEFAULT 2.0,
        require_path_contains TEXT NOT NULL DEFAULT '',
        enabled      INTEGER NOT NULL DEFAULT 1,
        created_at   REAL NOT NULL
    );
    """

    def __init__(self, *, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _default_db()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.executescript(self.SCHEMA)
        # In-memory last-fire timestamps for rate limiting. Read +
        # write are now under self._lock (fwr-1 audit).
        self._last_fire: Dict[str, float] = {}
        # fwr-2 audit: cache parsed rules + compiled glob regex so
        # match() doesn't re-query SQLite + JSON.loads on every event
        # in a burst. Invalidated on add/remove/set_enabled/wipe.
        self._rules_cache: Optional[List[Rule]] = None
        self._compiled_glob_cache: Dict[str, "re.Pattern"] = {}

    # ---- CRUD ---------------------------------------------------------

    def add_rule(self, *, name: str, path_glob: str, action: str,
                 kinds: Optional[List[FsEventKind]] = None,
                 action_args: Optional[Dict[str, Any]] = None,
                 min_size_bytes: int = 0,
                 max_size_bytes: int = 0,
                 rate_limit_sec: float = 2.0,
                 require_path_contains: str = "",
                 enabled: bool = True) -> Rule:
        rule = Rule(
            id=uuid.uuid4().hex[:12],
            name=name, path_glob=path_glob, action=action,
            action_args=dict(action_args or {}),
            kinds=[(k.value if isinstance(k, FsEventKind) else str(k))
                   for k in (kinds or [])],
            min_size_bytes=int(min_size_bytes),
            max_size_bytes=int(max_size_bytes),
            rate_limit_sec=float(rate_limit_sec),
            require_path_contains=str(require_path_contains or ""),
            enabled=bool(enabled),
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO rules"
                "(id, name, path_glob, kinds_json, action, action_args, "
                " min_size, max_size, rate_limit, "
                " require_path_contains, enabled, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rule.id, rule.name, rule.path_glob,
                 json.dumps(rule.kinds),
                 rule.action, json.dumps(rule.action_args),
                 rule.min_size_bytes, rule.max_size_bytes,
                 rule.rate_limit_sec, rule.require_path_contains,
                 1 if rule.enabled else 0, rule.created_at),
            )
            self._invalidate_cache()
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM rules WHERE id=?", (rule_id,))
            self._invalidate_cache()
            return cur.rowcount > 0

    def set_enabled(self, rule_id: str, enabled: bool) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE rules SET enabled=? WHERE id=?",
                (1 if enabled else 0, rule_id))
            self._invalidate_cache()
            return cur.rowcount > 0

    def _invalidate_cache(self) -> None:
        self._rules_cache = None
        self._compiled_glob_cache.clear()

    def rules(self, *, include_disabled: bool = True) -> List[Rule]:
        with self._lock:
            # fwr-2 audit: serve from cache when fresh — avoids SQL
            # query + JSON.loads per file event during save storms.
            all_rules = self._rules_cache
            if all_rules is None:
                cur = self._conn.execute(
                    "SELECT id, name, path_glob, kinds_json, action, "
                    "       action_args, min_size, max_size, rate_limit, "
                    "       require_path_contains, enabled, created_at "
                    "FROM rules ORDER BY created_at DESC")
                rows = cur.fetchall()
                all_rules = []
                for r in rows:
                    try:
                        kinds = json.loads(r[3])
                        args = json.loads(r[5])
                    except Exception:
                        kinds, args = [], {}
                    all_rules.append(Rule(
                        id=r[0], name=r[1], path_glob=r[2],
                        action=r[4], action_args=args, kinds=kinds,
                        min_size_bytes=r[6], max_size_bytes=r[7],
                        rate_limit_sec=r[8],
                        require_path_contains=r[9],
                        enabled=bool(r[10]), created_at=r[11],
                    ))
                self._rules_cache = all_rules
        if include_disabled:
            return list(all_rules)
        return [r for r in all_rules if r.enabled]

    # ---- matching + dispatch -----------------------------------------

    def match(self, event: FsEvent) -> List[Rule]:
        """Return the rules that fire for this event, AFTER applying
        rate limits. The caller can then dispatch via its own
        side-effect path.

        fwr-1 audit: rate-limit read+write is now under the engine's
        RLock so two concurrent dispatcher threads can't both observe
        last=old and both fire. Without this lock the rate limiter
        was effectively a no-op under burst load."""
        now = event.ts or time.time()
        matched: List[Rule] = []
        with self._lock:
            for rule in self.rules(include_disabled=False):
                if not _kind_matches(rule, event):
                    continue
                if not _path_matches(rule, event):
                    continue
                if not _size_matches(rule, event):
                    continue
                if not _path_contains_matches(rule, event):
                    continue
                last = self._last_fire.get(rule.id, 0.0)
                if (now - last) < rule.rate_limit_sec:
                    continue
                self._last_fire[rule.id] = now
                matched.append(rule)
        return matched

    def dispatch(self, event: FsEvent,
                 dispatcher: Callable[[str, Dict[str, Any]], Any]
                 ) -> List[Dict[str, Any]]:
        """Match `event` against rules, then call `dispatcher(action,
        args)` for each. Returns a list of {rule_id, dispatched, error}
        records the caller can persist."""
        results: List[Dict[str, Any]] = []
        for rule in self.match(event):
            args = _resolve_placeholders(rule.action_args, event)
            try:
                out = dispatcher(rule.action, args)
                results.append({"rule_id": rule.id, "dispatched": True,
                                "output": out})
            except Exception as exc:
                results.append({"rule_id": rule.id, "dispatched": False,
                                "error": f"{type(exc).__name__}: {exc}"})
        return results

    # ---- maintenance --------------------------------------------------

    def wipe(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM rules")
            n = int(cur.fetchone()[0])
            self._conn.execute("DELETE FROM rules")
            self._last_fire.clear()
            self._invalidate_cache()
            return n

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---- match helpers -----------------------------------------------------

def _kind_matches(rule: Rule, event: FsEvent) -> bool:
    if not rule.kinds:
        return True
    return event.kind.value in rule.kinds


def _path_matches(rule: Rule, event: FsEvent) -> bool:
    """fnmatch with '**' as a recursive directory marker.

    `**/*.pdf` matches at any depth. `~/Downloads/*` matches one
    level. `*.png` matches by basename in any dir.

    Both sides are forward-slash normalized so a Windows-native
    glob like `C:\\Downloads\\**\\*.pdf` (which file pickers
    produce) still matches `C:/Downloads/x.pdf` (missed-by-panel
    Windows-paths finding).
    """
    glob = rule.path_glob
    if not glob:
        return False
    # Expand `~`.
    if glob.startswith("~"):
        glob = str(Path(glob).expanduser())
    # Normalize BOTH sides to forward slashes so Windows-native
    # paths in the user-supplied glob still match the event path.
    glob = glob.replace("\\", "/")
    path = (event.path or "").replace("\\", "/")
    # If there's no slash in the pattern, match against basename.
    if "/" not in glob:
        return fnmatch.fnmatch(os.path.basename(path).lower(),
                               glob.lower())
    # Handle the `**` recursive marker by translating to a regex
    # that allows nested slashes.
    pat_regex = _glob_to_regex(glob)
    return bool(pat_regex.match(path))


def _glob_to_regex(glob: str) -> "re.Pattern":
    """Build a regex that matches:
       - `**/`   → optional any-depth prefix (so `**/*.pdf` also
                   matches `a.pdf` at the root). Only recognized
                   when `**` is at a path-segment boundary (start of
                   string OR preceded by `/`) — `src/**foo` does NOT
                   collapse to `src/.*foo` (fwr-4 audit).
       - `**`    → recursive marker only as its own segment; mid-
                   segment `**` is treated as two `*`s (each = any
                   non-slash chars).
       - `*`     → any non-slash chars
       - `?`     → single non-slash
    Otherwise standard literal matching."""
    out_chars: List[str] = []
    i = 0
    while i < len(glob):
        c = glob[i]
        if c == "*":
            if i + 1 < len(glob) and glob[i + 1] == "*":
                # `**` must be at a path-segment boundary to be a
                # recursive marker. Prev char must be '/' or start.
                prev_ok = i == 0 or glob[i - 1] == "/"
                if not prev_ok:
                    # Treat both stars as ordinary `*` (no slash).
                    out_chars.append("[^/]*[^/]*")
                    i += 2
                    continue
                # `**/` → optional any-depth prefix.
                if i + 2 < len(glob) and glob[i + 2] == "/":
                    out_chars.append("(?:.*/)?")
                    i += 3
                    continue
                # `**` at end of pattern or before non-slash → any
                # path (including slashes).
                out_chars.append(".*")
                i += 2
                continue
            out_chars.append("[^/]*")
            i += 1
            continue
        if c == "?":
            out_chars.append("[^/]")
            i += 1
            continue
        out_chars.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(out_chars) + "$", re.IGNORECASE)


def _size_matches(rule: Rule, event: FsEvent) -> bool:
    if rule.min_size_bytes and event.size_bytes < rule.min_size_bytes:
        return False
    if rule.max_size_bytes and event.size_bytes > rule.max_size_bytes:
        return False
    return True


def _path_contains_matches(rule: Rule, event: FsEvent) -> bool:
    if not rule.require_path_contains:
        return True
    return rule.require_path_contains.lower() in (event.path or "").lower()


def _resolve_placeholders(template_args: Dict[str, Any],
                          event: FsEvent) -> Dict[str, Any]:
    """Substitute {path}/{filename}/{ext}/{kind} placeholders in
    string-valued args. Non-string args pass through unchanged."""
    if not template_args:
        return {}
    filename = os.path.basename(event.path)
    name, ext = os.path.splitext(filename)
    repl = {
        "path": event.path,
        "filename": filename,
        "name": name,
        "ext": ext.lstrip("."),
        "kind": event.kind.value,
        "src_path": event.src_path or event.path,
    }
    out: Dict[str, Any] = {}
    for k, v in template_args.items():
        if isinstance(v, str):
            out[k] = _PLACEHOLDER_RE.sub(
                lambda m: str(repl.get(m.group(1), m.group(0))), v)
        else:
            out[k] = v
    return out

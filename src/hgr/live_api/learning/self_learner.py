"""Background self-learning daemon.

Iris already learns from live realtime conversation (via
``MemoryManager.observe_conversation``) and from planner-handled turns
(via the deterministic ``extract_facts``). The self-learning daemon
fills in the other sources: project-file metadata, git config, tool
call patterns, and email/calendar contacts pulled from connectors that
already have working OAuth in this session.

Design contract — keep it boring so the daemon doesn't surprise anyone:

  * **Never blocks realtime.** Runs on its own daemon thread; every
    source scanner is wrapped in try/except and silently degrades on
    auth/API failure (offline = no-op, missing connector = no-op).
  * **Throttled.** Each source has its own minimum interval (project
    files: 6 h, git config: 6 h, tool patterns: 1 h, email/calendar
    contacts: 24 h). Per-source ``last_run_at`` is persisted to
    ``%LOCALAPPDATA%/Touchless/self_learner_state.json`` so a restart
    doesn't re-scan everything from scratch.
  * **Dedup before LLM.** When a candidate fact lands on a source
    scanner, the daemon checks the existing memory store BEFORE paying
    for any LLM extraction round-trip — the LLM is the expensive
    surface and the most common scenario (re-running on the same
    project files) should be free.
  * **Provenance.** Every write tags ``source_kind`` ∈ {project_file,
    git_config, tool_pattern, email, calendar, conversation} plus a
    ``source_id`` pointer (file path, project id, etc.) so the cortex
    UI can group "things Iris learned from your projects" separately
    from chat-derived facts.
  * **No new dependencies.** All connector / store imports are
    in-process; everything is stdlib + the existing memory machinery.

Public surface:

  ``SelfLearner(memory_manager, project_store, tool_call_log_path,
                connector_registry=None, state_path=None, logger=None)``
  ``run_once() -> Dict[str, int]``     — orchestrates all source scans
  ``self_learn_now() -> Dict[str, int]`` — alias for the manual trigger

The daemon thread itself lives in ``LiveApiManager`` (see
``_init_self_learner``). This module is intentionally importable
without Qt so unit tests + CLI smoke runs work in any env.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..memory.llm_extractor import (
    confidence_threshold as _llm_confidence_threshold,
    enabled as _llm_extraction_enabled,
    extract_facts_from_conversation,
)
from ..memory.store import (
    SOURCE_KIND_CONVERSATION,
    MemoryStore,
    SemanticRow,
)


# Source ids — match the ``source_kind`` vocabulary in store.py plus
# the daemon-specific tags. Kept here as constants so cortex / tests
# can compare without typo-prone string literals.
SOURCE_PROJECT_FILE = "project_file"
SOURCE_GIT_CONFIG = "git_config"
SOURCE_TOOL_PATTERN = "tool_pattern"
SOURCE_EMAIL = "email"
SOURCE_CALENDAR = "calendar"


# Per-source minimum interval between scans. Picked to balance freshness
# against API-cost / disk churn: tool patterns are cheap (local DB) so
# hourly is fine; project files don't change every minute so 6 h is
# plenty; email/calendar hit external APIs so daily by default.
_CADENCE_SECONDS: Dict[str, float] = {
    "recent_episodes": 60 * 10,       # 10 min — episodic memory only
    SOURCE_PROJECT_FILE: 60 * 60 * 6,   # 6 h
    SOURCE_GIT_CONFIG: 60 * 60 * 6,     # 6 h
    SOURCE_TOOL_PATTERN: 60 * 60 * 1,   # 1 h
    SOURCE_EMAIL: 60 * 60 * 24,         # 24 h
    SOURCE_CALENDAR: 60 * 60 * 24,      # 24 h
}


# LLM throttling — never make more than N extraction calls per minute
# across the whole daemon, regardless of how many sources fire in the
# same cycle. Keeps the spend bounded even when first-time-pass
# discovers many new episodes at once.
_LLM_CALLS_PER_MINUTE = 6
# Per-cycle hard cap on facts emitted (irrespective of LLM rate limit).
_MAX_FACTS_PER_CYCLE = 100


# Regexes for project-file metadata. Cheap, predictable: each captures
# one ``(label, value)`` pair we can write as a fact. Multi-line scan
# is fine — these are short files (README, CLAUDE.md, package.json).
_AUTHOR_RE = re.compile(
    r"(?im)^\s*(?:author|maintained\s*by|maintainer)\s*[:\-]\s*([^\n]+?)\s*$"
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def default_state_path() -> Path:
    """Where the per-source last_run timestamps live.

    Mirrors ``default_memory_path`` so all of Iris's persistence lands
    under one folder. Override via ``TOUCHLESS_SELF_LEARN_STATE``.
    """
    override = os.environ.get("TOUCHLESS_SELF_LEARN_STATE")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Touchless" / "self_learner_state.json"
    return Path.home() / ".touchless" / "self_learner_state.json"


class SelfLearner:
    """Background daemon that mines user facts from external sources.

    Stateless across instances except for the on-disk
    ``self_learner_state.json`` (last_run_at per source). Designed so
    multiple ``run_once()`` calls in a row are cheap when nothing
    changed — cadence guards + dedup-before-LLM do the work.
    """

    def __init__(self,
                 memory_manager: Any,
                 project_store: Any = None,
                 tool_call_log_path: Optional[Path] = None,
                 connector_registry: Optional[Any] = None,
                 state_path: Optional[Path] = None,
                 logger: Any = None) -> None:
        if memory_manager is None:
            raise ValueError("memory_manager is required")
        self._memory = memory_manager
        self._project_store = project_store
        self._registry = connector_registry
        self._tool_call_log_path = (Path(tool_call_log_path)
                                    if tool_call_log_path else None)
        self._state_path = Path(state_path) if state_path else default_state_path()
        self._logger = logger
        self._state_lock = threading.Lock()
        # In-memory mirror of the persisted state; loaded lazily on first
        # access so import is side-effect-free.
        self._state: Optional[Dict[str, Any]] = None
        # LLM rate-limit window — list of timestamps of recent calls;
        # callers check ``_acquire_llm_slot()`` before paying.
        self._llm_call_times: List[float] = []
        self._llm_lock = threading.Lock()

    # ----------------------------------------------------------------- public

    def run_once(self) -> Dict[str, int]:
        """Single learning pass across every available source.

        Returns ``{source_id: facts_written, ...}``. Sources skipped
        (cadence not reached, auth missing, store unavailable) are
        omitted from the result so callers can tell signal from noise.
        """
        report: Dict[str, int] = {}
        cycle_budget = _MAX_FACTS_PER_CYCLE

        # Order matters only for the cycle-budget: cheaper / more-reliable
        # sources first so high-cost LLM sources don't starve them.
        scanners: List[tuple] = [
            ("recent_episodes", self._mine_recent_episodes),
            (SOURCE_TOOL_PATTERN, self._mine_tool_patterns),
            (SOURCE_PROJECT_FILE, self._mine_project_metadata),
            (SOURCE_GIT_CONFIG, self._mine_git_configs),
            (SOURCE_EMAIL, self._mine_email_contacts),
            (SOURCE_CALENDAR, self._mine_calendar_contacts),
        ]

        for source_id, scanner in scanners:
            if cycle_budget <= 0:
                break
            cadence = _CADENCE_SECONDS.get(source_id, 0)
            if cadence and not self._should_run(source_id, cadence):
                continue
            try:
                written = scanner(budget=cycle_budget)
            except Exception as exc:  # pragma: no cover - defensive
                if self._logger:
                    self._logger.exception(
                        "self_learner_source_failed",
                        exc,
                        source=source_id,
                    )
                continue
            if written:
                report[source_id] = int(written)
                cycle_budget = max(0, cycle_budget - int(written))
            # Always mark the source as run, even when 0 facts were
            # found — re-running is wasted work until cadence elapses.
            self._mark_run(source_id)

        # Persist last_run state once per cycle (single write instead of
        # per-source) so a crash mid-cycle doesn't leave the JSON in a
        # half-updated state.
        try:
            self._save_state()
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("self_learner_state_save_failed", exc)

        if self._logger:
            self._logger.event("self_learner_cycle",
                               sources=list(report.keys()),
                               counts=report,
                               total=sum(report.values()))
        return report

    # Manual trigger alias — semantically identical to run_once(); kept
    # as a separate name so CLI / debug callers can find it easily.
    def self_learn_now(self) -> Dict[str, int]:
        return self.run_once()

    # ----------------------------------------------------------------- sources

    def _mine_recent_episodes(self, budget: int = 50) -> int:
        """Re-extract facts from episodes added since the last run.

        ``MemoryManager.observe_conversation`` already extracts facts
        from realtime turns. This pass picks up episodes the planner
        recorded that DIDN'T go through observe_conversation (e.g.
        Tier-1 deterministic handles) and runs the same LLM extractor
        over their user_text + outcome, so nothing slips through.
        """
        store = self._semantic_store()
        if store is None or not _llm_extraction_enabled():
            return 0

        since = self._get_last_run_at("recent_episodes")
        try:
            rows = store.list_episodic(limit=400)
        except Exception:
            return 0
        # Newest first from list_episodic; filter to anything written
        # since the last successful pass.
        fresh = [r for r in rows if r.ts > since]
        if not fresh:
            return 0
        # Process oldest-first so per-LLM logs read chronologically.
        fresh.reverse()

        written = 0
        threshold = _llm_confidence_threshold()
        for row in fresh:
            if written >= budget:
                break
            user_text = (row.user_text or "").strip()
            if len(user_text) < 8:
                continue
            if not self._acquire_llm_slot():
                # Rate-limited — leave the rest for the next cycle. The
                # last_run_at update happens at the cycle level, so the
                # next run will start exactly where this one stopped.
                break
            try:
                candidates = extract_facts_from_conversation(
                    user_text, (row.outcome or ""), logger=self._logger)
            except Exception:
                continue
            for kind, key, value, conf in candidates:
                if conf < threshold:
                    continue
                if self._fact_already_present(store, kind, key, value):
                    continue
                self._write_fact(
                    kind=kind,
                    key=key,
                    value=value,
                    source=f"episode:{row.id}",
                    source_kind=SOURCE_KIND_CONVERSATION,
                    source_id=f"episode:{row.id}",
                )
                written += 1
                if written >= budget:
                    break
        return written

    def _mine_project_metadata(self, budget: int = 100) -> int:
        """Scan registered project roots for author / maintainer info.

        Cheap: each file is opened once, regexed for a handful of
        well-known patterns. We don't try to LLM-extract the README's
        body — that's project-RAG's job. The goal here is to learn
        ``person/author -> <name>`` style identity facts.
        """
        if self._project_store is None:
            return 0
        store = self._semantic_store()
        if store is None:
            return 0

        try:
            projects = self._project_store.list_projects()
        except Exception:
            return 0

        written = 0
        targets = ("CLAUDE.md", "README.md", "package.json", "pyproject.toml")
        for proj in projects:
            if written >= budget:
                break
            root_path = proj.get("root_path")
            if not root_path:
                continue
            root = Path(root_path)
            if not root.is_dir():
                continue
            for name in targets:
                if written >= budget:
                    break
                candidate = root / name
                if not candidate.is_file():
                    continue
                try:
                    text = candidate.read_text(
                        encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if not text:
                    continue
                for fact in self._extract_project_file_facts(
                        text, candidate, proj):
                    if written >= budget:
                        break
                    if self._fact_already_present(
                            store, fact["kind"], fact["key"], fact["value"]):
                        continue
                    self._write_fact(
                        kind=fact["kind"],
                        key=fact["key"],
                        value=fact["value"],
                        source=f"{SOURCE_PROJECT_FILE}:{candidate}",
                        source_kind=SOURCE_PROJECT_FILE,
                        source_id=str(candidate),
                    )
                    written += 1
        return written

    def _mine_git_configs(self, budget: int = 20) -> int:
        """Per-project ``git config user.name`` + ``user.email`` capture.

        Failures (git not installed, not a repo, no config set) silently
        skip. Useful for picking up identity facts the project files
        don't surface explicitly.
        """
        if self._project_store is None:
            return 0
        store = self._semantic_store()
        if store is None:
            return 0
        try:
            projects = self._project_store.list_projects()
        except Exception:
            return 0

        written = 0
        for proj in projects:
            if written >= budget:
                break
            root_path = proj.get("root_path")
            if not root_path:
                continue
            root = Path(root_path)
            if not (root / ".git").exists():
                continue
            name = self._git_config_value(root, "user.name")
            email = self._git_config_value(root, "user.email")
            if name and not self._fact_already_present(
                    store, "person", "git-user", name):
                self._write_fact(
                    kind="person",
                    key="git-user",
                    value=name,
                    source=f"{SOURCE_GIT_CONFIG}:{root}",
                    source_kind=SOURCE_GIT_CONFIG,
                    source_id=str(root),
                )
                written += 1
            if email and not self._fact_already_present(
                    store, "contact", "git-email", email):
                self._write_fact(
                    kind="contact",
                    key="git-email",
                    value=email,
                    source=f"{SOURCE_GIT_CONFIG}:{root}",
                    source_kind=SOURCE_GIT_CONFIG,
                    source_id=str(root),
                )
                written += 1
        return written

    def _mine_tool_patterns(self, budget: int = 20) -> int:
        """Count tool fires across the last 7 days; emit a 'preference'
        fact when any tool crosses a usage threshold (>=5 sessions or
        >=20 total fires).

        Read-only against tool_call_log.db; no LLM cost. Idempotent —
        the dedup check prevents writing the same preference twice.
        """
        store = self._semantic_store()
        if store is None:
            return 0
        try:
            from ..cortex.tool_call_log import load_sessions
        except Exception:
            return 0

        try:
            sessions = load_sessions(self._tool_call_log_path)
        except Exception:
            return 0
        if not sessions:
            return 0

        # Window: last 7 days. The session record carries earliest_ts so
        # we don't need to re-scan rows.
        cutoff = time.time() - 7 * 86400
        recent = [s for s in sessions
                  if float(s.get("earliest_ts", 0.0)) >= cutoff]
        if not recent:
            return 0

        per_tool_sessions: Counter = Counter()
        per_tool_total: Counter = Counter()
        for sess in recent:
            seen_in_session: set = set()
            for tid in sess.get("tool_ids") or []:
                if not tid:
                    continue
                per_tool_total[tid] += 1
                if tid not in seen_in_session:
                    seen_in_session.add(tid)
                    per_tool_sessions[tid] += 1

        written = 0
        for tool_id, sess_count in per_tool_sessions.most_common():
            if written >= budget:
                break
            total = per_tool_total[tool_id]
            # Only mark as a habit when usage is non-trivial. Single-fire
            # tools are noise; we want the things the user actually
            # leans on.
            if sess_count < 5 and total < 20:
                continue
            key = f"frequent-tool:{tool_id}"
            value = (f"used {total} times across {sess_count} sessions "
                     "(last 7 days)")
            # Skip if we already wrote the same usage tier — using the
            # exact same value text is the cheapest way to dedup since
            # the underlying UNIQUE(kind,key,value) constraint will
            # silently no-op writes that match.
            if self._fact_already_present(store, "preference", key, value):
                continue
            self._write_fact(
                kind="preference",
                key=key,
                value=value,
                source=f"{SOURCE_TOOL_PATTERN}:{tool_id}",
                source_kind=SOURCE_TOOL_PATTERN,
                source_id=tool_id,
            )
            written += 1
        return written

    def _mine_email_contacts(self, budget: int = 30) -> int:
        """Pull person -> email facts from the Gmail / MS365 connectors.

        Both connectors expose ``available()``; missing auth is a
        no-op. Batched: one Gmail list call returns up to 50 messages,
        from which we extract ``(from_name, from_addr)`` pairs and write
        the unseen ones. No LLM is required — these are structured
        header values.
        """
        if self._registry is None:
            return 0
        store = self._semantic_store()
        if store is None:
            return 0

        written = 0
        # ---- Gmail ----
        gmail = self._find_connector("gmail")
        if gmail is not None and self._is_available(gmail):
            try:
                # Reuse the existing tool surface so we don't re-implement
                # auth / quoted-printable / header parsing. include_body
                # off keeps this to a single metadata round-trip.
                result = gmail.execute(
                    "gmail_list",
                    {"max": 50, "include_body": False, "unread_only": False},
                )
            except Exception:
                result = None
            if isinstance(result, dict) and result.get("status") == "ok":
                for msg in result.get("messages") or []:
                    if written >= budget:
                        break
                    addr = (msg.get("from") or "").strip()
                    name = (msg.get("from_name") or "").strip()
                    if not addr or "@" not in addr:
                        continue
                    contact_key = name.lower() if name else addr.split("@", 1)[0].lower()
                    if not contact_key:
                        continue
                    if self._fact_already_present(
                            store, "person", contact_key, addr):
                        continue
                    self._write_fact(
                        kind="person",
                        key=contact_key,
                        value=addr,
                        source=f"{SOURCE_EMAIL}:gmail",
                        source_kind=SOURCE_EMAIL,
                        source_id=f"gmail:{msg.get('id')}",
                    )
                    written += 1

        # ---- MS365 mail ----
        ms = self._find_connector("ms365")
        if ms is not None and self._is_available(ms) and written < budget:
            try:
                result = ms.execute(
                    "ms_mail_list",
                    {"max": 50, "include_body": False, "unread_only": False},
                )
            except Exception:
                result = None
            if isinstance(result, dict) and result.get("status") == "ok":
                for msg in result.get("messages") or []:
                    if written >= budget:
                        break
                    addr = (msg.get("from") or "").strip()
                    name = (msg.get("from_name") or "").strip()
                    if not addr or "@" not in addr:
                        continue
                    contact_key = name.lower() if name else addr.split("@", 1)[0].lower()
                    if not contact_key:
                        continue
                    if self._fact_already_present(
                            store, "person", contact_key, addr):
                        continue
                    self._write_fact(
                        kind="person",
                        key=contact_key,
                        value=addr,
                        source=f"{SOURCE_EMAIL}:ms365",
                        source_kind=SOURCE_EMAIL,
                        source_id=f"ms365:{msg.get('id')}",
                    )
                    written += 1
        return written

    def _mine_calendar_contacts(self, budget: int = 20) -> int:
        """Pull person facts from upcoming MS365 calendar attendees.

        Google Calendar's connector also exposes ``list``-style tools
        but the shape differs; iterate the result defensively.
        """
        if self._registry is None:
            return 0
        store = self._semantic_store()
        if store is None:
            return 0

        written = 0
        ms = self._find_connector("ms365")
        if ms is not None and self._is_available(ms):
            try:
                result = ms.execute("ms_calendar_list", {"max": 20})
            except Exception:
                result = None
            if isinstance(result, dict) and result.get("status") == "ok":
                for evt in result.get("events") or []:
                    if written >= budget:
                        break
                    for att in (evt.get("attendees") or []):
                        if written >= budget:
                            break
                        # Graph shape: {emailAddress: {name, address}} or
                        # the flatter {name, email} we sometimes pass
                        # through downstream tools.
                        addr = ""
                        name = ""
                        ea = att.get("emailAddress") if isinstance(att, dict) else None
                        if isinstance(ea, dict):
                            addr = (ea.get("address") or "").strip()
                            name = (ea.get("name") or "").strip()
                        if not addr and isinstance(att, dict):
                            addr = (att.get("email") or "").strip()
                            name = (att.get("name") or name).strip()
                        if not addr or "@" not in addr:
                            continue
                        contact_key = (
                            name.lower() if name
                            else addr.split("@", 1)[0].lower())
                        if not contact_key:
                            continue
                        if self._fact_already_present(
                                store, "person", contact_key, addr):
                            continue
                        self._write_fact(
                            kind="person",
                            key=contact_key,
                            value=addr,
                            source=f"{SOURCE_CALENDAR}:ms365",
                            source_kind=SOURCE_CALENDAR,
                            source_id=f"ms365_event:{evt.get('id')}",
                        )
                        written += 1
        return written

    # --------------------------------------------------------------- helpers

    def _extract_project_file_facts(self, text: str, path: Path,
                                    project: Dict[str, Any]
                                    ) -> List[Dict[str, str]]:
        """Run the cheap project-file regexes and return a list of
        ``{kind, key, value}`` dicts ready for dedup + write."""
        out: List[Dict[str, str]] = []
        suffix = path.suffix.lower()

        # 1. CLAUDE.md / README.md "Author: <name>" line.
        if suffix == ".md":
            for m in _AUTHOR_RE.finditer(text):
                value = m.group(1).strip()
                if not value or len(value) > 200:
                    continue
                # Strip a trailing parenthetical / role suffix so the
                # canonical key is just the name.
                cleaned = re.sub(r"\s*[\(\[].*$", "", value).strip()
                if not cleaned:
                    continue
                out.append({
                    "kind": "person",
                    "key": "author",
                    "value": cleaned,
                })
                # Pick up any email in the same line for free.
                email_match = _EMAIL_RE.search(value)
                if email_match:
                    out.append({
                        "kind": "contact",
                        "key": cleaned.lower(),
                        "value": email_match.group(0),
                    })

        # 2. package.json author field (either string or {name, email}).
        if path.name == "package.json":
            try:
                data = json.loads(text)
            except Exception:
                data = None
            if isinstance(data, dict):
                author = data.get("author")
                name = None
                email = None
                if isinstance(author, str):
                    em = _EMAIL_RE.search(author)
                    email = em.group(0) if em else None
                    # "Name <email>" → split at the bracket.
                    cleaned = re.sub(r"\s*<[^>]*>", "", author).strip()
                    name = cleaned or None
                elif isinstance(author, dict):
                    name = (author.get("name") or "").strip() or None
                    email = (author.get("email") or "").strip() or None
                if name:
                    out.append({
                        "kind": "person",
                        "key": "author",
                        "value": name,
                    })
                if email:
                    out.append({
                        "kind": "contact",
                        "key": (name or "author").lower(),
                        "value": email,
                    })

        # 3. pyproject.toml authors. Plain regex — TOML parsing adds a
        # dependency and we only need the obvious cases.
        if path.name == "pyproject.toml":
            # authors = [{name = "X", email = "y"}]
            for m in re.finditer(
                    r'name\s*=\s*"([^"]+)"(?:\s*,\s*email\s*=\s*"([^"]+)")?',
                    text):
                name = m.group(1).strip()
                email = (m.group(2) or "").strip()
                if name:
                    out.append({
                        "kind": "person",
                        "key": "author",
                        "value": name,
                    })
                if email:
                    out.append({
                        "kind": "contact",
                        "key": (name or "author").lower(),
                        "value": email,
                    })

        return out

    def _git_config_value(self, root: Path, key: str) -> str:
        """Best-effort ``git -C <root> config --get <key>``. Returns ''
        on any failure (git missing, not a repo, key unset)."""
        try:
            # hidden_subprocess_kwargs equivalent — wrap in CREATE_NO_WINDOW
            # if available, otherwise plain subprocess. Don't import the
            # touchless helper to keep this module dependency-light.
            extra: Dict[str, Any] = {}
            if os.name == "nt":
                extra["creationflags"] = getattr(
                    subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(
                ["git", "-C", str(root), "config", "--get", key],
                capture_output=True, text=True, timeout=5.0, **extra,
            )
            if proc.returncode != 0:
                return ""
            return (proc.stdout or "").strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    def _find_connector(self, connector_id: str) -> Optional[Any]:
        """Pull a connector from the registry by id; returns None if the
        registry is missing or the id isn't registered. Handles both
        dict-shaped and list-shaped registries to stay tolerant of
        whichever surface ``build_connector_registry`` exposes today.
        """
        reg = self._registry
        if reg is None:
            return None
        # Preferred path: ``ConnectorRegistry.find_by_id(id)`` if present
        # (the canonical accessor in connectors/base.py).
        for accessor in ("find_by_id", "find"):
            method = getattr(reg, accessor, None)
            if method is None:
                continue
            try:
                got = method(connector_id)
                if got is not None:
                    return got
            except Exception:
                pass
        # Fallback: scan the underlying list. Use the private attr
        # because ConnectorRegistry doesn't expose a public iterator
        # today (see connectors/base.py).
        try:
            for conn in getattr(reg, "_connectors", []) or []:
                if getattr(conn, "id", "") == connector_id:
                    return conn
        except Exception:
            pass
        return None

    @staticmethod
    def _is_available(connector: Any) -> bool:
        try:
            return bool(connector.available())
        except Exception:
            return False

    # -------------------------------------------------------- memory access

    def _semantic_store(self) -> Optional[MemoryStore]:
        """Pull the underlying ``MemoryStore`` from the manager.

        ``MemoryManager._store`` is the private accessor; this is the
        same path other in-tree helpers (consolidate_facts, patterns)
        use, so the contract is stable. Returns None if the manager
        doesn't expose a store (custom test stubs).
        """
        try:
            return self._memory._store  # type: ignore[attr-defined]
        except AttributeError:
            return None

    def _fact_already_present(self, store: Any, kind: str, key: str,
                              value: str) -> bool:
        """True when (kind, key, value) is already in the store.

        Used to skip work BEFORE paying for an LLM round-trip. The
        underlying SQLite UNIQUE constraint already protects against
        duplicate INSERTs, but knowing in advance lets us avoid the
        extraction call entirely.
        """
        try:
            rows = store.find_facts(kind=kind, key=key.lower(), limit=50)
        except Exception:
            return False
        for row in rows:
            if getattr(row, "value", None) == value:
                return True
        return False

    def _write_fact(self, *, kind: str, key: str, value: str, source: str,
                    source_kind: str, source_id: Optional[str]) -> None:
        """Single fact write through MemoryManager.set_fact, with the
        right provenance tag. Best-effort — failures are logged via the
        manager and not re-raised."""
        try:
            self._memory.set_fact(
                kind, key, value,
                source=source,
                source_kind=source_kind,
                source_id=source_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("self_learner_write_failed", exc,
                                       kind=kind, key=key)

    # ---------------------------------------------------------- rate limit

    def _acquire_llm_slot(self) -> bool:
        """Token-bucket-ish per-minute throttle on LLM calls.

        Returns True (and reserves the slot) when at least one of the
        last 60 s slots is free, False otherwise. Cheap O(N) where N
        is the bucket cap (~6).
        """
        with self._llm_lock:
            now = time.time()
            cutoff = now - 60.0
            self._llm_call_times = [t for t in self._llm_call_times
                                    if t >= cutoff]
            if len(self._llm_call_times) >= _LLM_CALLS_PER_MINUTE:
                return False
            self._llm_call_times.append(now)
            return True

    # ---------------------------------------------------------- state mgmt

    def _ensure_state(self) -> Dict[str, Any]:
        with self._state_lock:
            if self._state is None:
                self._state = self._load_state()
            return self._state

    def _load_state(self) -> Dict[str, Any]:
        path = self._state_path
        try:
            if not path.exists():
                return {"last_run_at": {}}
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return {"last_run_at": {}}
        if not isinstance(data, dict):
            return {"last_run_at": {}}
        # Backward-compat: older state files might just have a flat dict
        # of source -> ts. Normalize to the wrapped shape.
        if "last_run_at" not in data:
            normalized = {
                k: float(v) for k, v in data.items()
                if isinstance(v, (int, float))
            }
            return {"last_run_at": normalized}
        return data

    def _save_state(self) -> None:
        with self._state_lock:
            state = self._state if self._state is not None else self._load_state()
            try:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                # Write atomically via temp + replace so a crash mid-write
                # can't leave a half-written JSON file.
                tmp = self._state_path.with_suffix(
                    self._state_path.suffix + ".tmp")
                with tmp.open("w", encoding="utf-8") as fh:
                    json.dump(state, fh, indent=2, sort_keys=True)
                os.replace(tmp, self._state_path)
            except Exception:
                # Best-effort persistence — failure to save means we
                # might re-scan a source next cycle, which is harmless.
                pass

    def _should_run(self, source_id: str, cadence_sec: float) -> bool:
        """True when enough time has elapsed since the last successful
        scan of ``source_id``. Cadence is enforced per-source, not
        per-cycle, so a cheap source can fire on a 1 h schedule even
        when expensive sources only fire daily."""
        if cadence_sec <= 0:
            return True
        last = self._get_last_run_at(source_id)
        return (time.time() - last) >= cadence_sec

    def _get_last_run_at(self, source_id: str) -> float:
        state = self._ensure_state()
        try:
            return float(state.get("last_run_at", {}).get(source_id, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _mark_run(self, source_id: str) -> None:
        state = self._ensure_state()
        bucket = state.setdefault("last_run_at", {})
        bucket[source_id] = time.time()

"""Iris Live Simulator — wired to your real cortex_world.json.

Reads %LOCALAPPDATA%\\Touchless\\cortex_world.json, converts the
projects + their touched files into the simulator's node hierarchy,
and injects the data via a QWebEngineScript that runs BEFORE the
page's own module script — so `window.IRIS_WORLD` is already set
when the simulator boots and the demo can use it synchronously.

If the world is empty (first run / Iris hasn't touched anything yet)
or the file is missing, the simulator falls back to its hardcoded
demo projects.

  python run_iris_simulator.py

Override the world file location via TOUCHLESS_CORTEX_WORLD.

Author: Konstantin Markov
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── Load-time fast-path cache + timeout helpers ────────────────────────
# A tiny JSON cache that memoizes the expensive parts of load_world_payload
# (patterns + suggestions) so the simulator doesn't burn 1-2 minutes on
# every restart re-embedding the same recent episodes. TTL is short (60s)
# so a real change to memory.db is reflected on the next launch within a
# minute, but rapid re-opens (testing, accidental double-clicks) are
# instant. Invalidated automatically when memory.db mtime is newer than
# the cache file. Long TTL (4h) keeps typical multi-session-per-day
# usage on the fast 1-2s cached path; only the first open of a brand
# new day re-pays the 60s+ cold-load cost.
_CACHE_TTL_S = 4 * 60 * 60  # 4 hours
# Hard per-source timeout used for HTTP-backed connector calls (MS365,
# Gmail). Keeps a single dead-quiet connector from holding up the whole
# simulator launch.
_CONNECTOR_TIMEOUT_S = 3.0
# Skip embedder-heavy pattern work if the embedder would block for too
# long. The 10s gate covers "first launch, slow uplink, 10+ episodes."
_PATTERN_FAST_PATH_BUDGET_S = 10.0
# Drastic caps — the cortex constellation only renders the top few
# anyway; recomputing 30 patterns/8 suggestions on every launch costs
# real wall-clock time for zero visual benefit.
_MAX_PATTERNS = 10
_MAX_SUGGESTIONS = 4


def _cache_path() -> Path:
    """Location of the load-time payload cache file. Mirrors the world
    state file's parent dir so installs that override LOCALAPPDATA also
    pick up the override cache."""
    override = os.environ.get("TOUCHLESS_IRIS_PAYLOAD_CACHE")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Touchless" / "iris_payload_cache.json"
    return Path.home() / ".touchless" / "iris_payload_cache.json"


def _memory_db_mtime() -> float:
    """mtime of the real memory.db so the cache can self-invalidate when
    the user actually adds new episodes/facts between launches.
    Returns 0.0 if the DB doesn't exist yet (fresh install)."""
    try:
        from hgr.live_api.memory import default_memory_path
        p = default_memory_path()
        if p.exists():
            return float(p.stat().st_mtime)
    except Exception:
        pass
    return 0.0


def _load_cache() -> Dict[str, Any]:
    """Read the payload cache, returning an empty dict on any failure.
    Caller is responsible for TTL/mtime validation."""
    try:
        p = _cache_path()
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Write the payload cache. Best-effort — never crashes the launcher
    if %LOCALAPPDATA% is read-only or the disk is full."""
    try:
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"WARN: payload cache write failed: {exc}", file=sys.stderr)


def _cache_entry_fresh(entry: Optional[Dict[str, Any]],
                       memory_mtime: float,
                       ttl_s: float = _CACHE_TTL_S) -> bool:
    """A cache entry is fresh iff (a) it exists, (b) its age is within
    the TTL, and (c) the memory.db hasn't changed since the entry was
    written. Returning False forces a recompute."""
    if not entry or not isinstance(entry, dict):
        return False
    try:
        age = time.time() - float(entry.get("written_at") or 0.0)
        if age < 0 or age > ttl_s:
            return False
        if float(entry.get("memory_mtime") or 0.0) < memory_mtime:
            # Real changes since cache write — let the recompute happen.
            return False
        return True
    except Exception:
        return False


def _run_with_timeout(fn: Callable[[], Any],
                      timeout_s: float,
                      label: str = "task") -> Tuple[bool, Any]:
    """Run ``fn`` on a worker thread and return (ok, result). On
    timeout / exception, returns (False, None). Used to bound connector
    HTTP calls so one slow connector can't stall load_world_payload."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn)
            try:
                return True, fut.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                print(
                    f"WARN: {label} timed out after {timeout_s:.1f}s; "
                    f"continuing with empty result",
                    file=sys.stderr,
                )
                # Note: the worker thread keeps running until its blocking
                # call returns — Python has no safe way to kill it. We
                # just stop waiting and move on. Acceptable: the thread
                # is a daemon, exits with the process.
                return False, None
            except Exception as exc:
                print(f"WARN: {label} crashed: {exc}", file=sys.stderr)
                return False, None
    except Exception as exc:
        print(f"WARN: {label} executor failed: {exc}", file=sys.stderr)
        return False, None


# Tier-1 project tint colors — cycled per project so the cortex still
# reads as "different projects, different teal shades."
_PROJECT_COLORS = [
    {"base": 0x1de9b6, "hot": 0xc8ffea, "halo": 0x1de9b6},
    {"base": 0x4be0d4, "hot": 0xcdf6f0, "halo": 0x4be0d4},
    {"base": 0x6ed8a9, "hot": 0xd3f4dd, "halo": 0x6ed8a9},
    {"base": 0x33c39b, "hot": 0xc4f0d8, "halo": 0x33c39b},
    {"base": 0x58cda0, "hot": 0xc8efd6, "halo": 0x58cda0},
    {"base": 0x42d4b6, "hot": 0xc7f0e2, "halo": 0x42d4b6},
]


def _default_world_path() -> Path:
    """Mirror world_state.default_world_path() — same location the
    real cortex writes to / reads from."""
    override = os.environ.get("TOUCHLESS_CORTEX_WORLD")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Touchless" / "cortex_world.json"
    return Path.home() / ".touchless" / "cortex_world.json"


def _slug(value: str) -> str:
    """Stable JS-safe id slug (no spaces, dots, slashes)."""
    out = []
    for ch in (value or "").lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in "-_":
            out.append(ch)
        else:
            out.append("-")
    s = "".join(out).strip("-") or "x"
    # collapse runs of dashes
    while "--" in s:
        s = s.replace("--", "-")
    return s


def _group_files_by_folder(files: List[Dict[str, Any]], root_path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Group files by their top-level subfolder under the project root.

    Files at the project root land under '(root)'. Files outside the
    detected root (rare — happens when world_state's auto-detect
    couldn't place them) land under '(other)'.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for f in files:
        path = f.get("path") or ""
        folder = "(other)"
        try:
            if root_path:
                rel = os.path.relpath(path, root_path)
                # If relpath returned something starting with ".." the
                # file is outside the root — drop into (other).
                if not rel.startswith(".."):
                    parts = rel.replace("\\", "/").split("/")
                    folder = parts[0] if len(parts) > 1 else "(root)"
            else:
                folder = "(root)"
        except Exception:
            folder = "(other)"
        groups.setdefault(folder, []).append(f)
    return groups


# Known sibling project roots — scanned when the cortex_world.json
# is empty or doesn't list them. Lets the simulator show real folder
# structure on first run, before Iris has actually populated the world.
# Shared with LiveApiManager (project-RAG indexer) via the
# ``hgr.live_api.known_projects`` module so both sides stay in sync.
from hgr.live_api.known_projects import KNOWN_PROJECT_ROOTS as _KNOWN_PROJECT_ROOTS

# Files we DON'T want to surface as leaves (test pollution, build
# artifacts, internal noise).
_PATH_BLACKLIST_FRAGMENTS = (
    r"\AppData\Local\Temp\\",
    "/AppData/Local/Temp/",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".git/",
    r"\.git\\",
    "build/",
    r"build\\",
    "dist/",
    r"dist\\",
)


def _path_is_blacklisted(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    for frag in _PATH_BLACKLIST_FRAGMENTS:
        if frag.lower().replace("\\\\", "/").replace("\\", "/") in p:
            return True
    return False


def _scan_project_folder(root: Path, *, max_files: int = 300) -> List[Dict[str, Any]]:
    """List interesting files under a project root — markdown docs,
    Python source, top-level READMEs, the OPEN_ISSUES + TODO files.
    Capped so a giant repo doesn't dump thousands of leaves."""
    if not root.exists() or not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    # Priority — top-level marker files first.
    priority_globs = [
        "OPEN_ISSUES.md", "README.md", "CLAUDE.md", "Touchless to-do.md",
        "package.json", "pyproject.toml",
        "docs/*.md", "*.md",
        "index.html", "*.html",
    ]
    seen: set = set()
    def maybe_add(fp):
        try:
            rp = str(fp.resolve())
            if rp in seen or _path_is_blacklisted(rp):
                return False
            seen.add(rp)
            out.append({"path": rp, "touches": 0})
            return True
        except Exception:
            return False

    for pat in priority_globs:
        for fp in sorted(root.glob(pat)):
            maybe_add(fp)
            if len(out) >= max_files:
                return out

    # Subfolder sweep — covers Python sources, web/JS source, docs,
    # styles, configs. Broad enough that every real project surfaces
    # something useful as satellite leaves.
    SUBFOLDER_GLOBS = [
        ("src",      ["*.py", "*.ts", "*.tsx", "*.js", "*.jsx", "*.html", "*.css", "*.json", "*.md"]),
        ("scripts",  ["*.py", "*.js", "*.sh", "*.bat", "*.ps1"]),
        ("docs",     ["*.md", "*.html", "*.pdf"]),
        ("pages",    ["*.html", "*.tsx", "*.jsx", "*.md", "*.vue"]),
        ("app",      ["*.tsx", "*.jsx", "*.html", "*.ts"]),
        ("public",   ["*.html", "*.png", "*.svg", "*.ico"]),
        ("content",  ["*.md", "*.html", "*.json"]),
        ("styles",   ["*.css", "*.scss"]),
        ("config",   ["*.yml", "*.yaml", "*.toml", "*.json"]),
        ("builder",  ["*.py", "*.spec", "*.bat", "*.ps1"]),
        ("assets",   ["*.png", "*.jpg", "*.svg"]),
        ("tests",    ["*.py", "*.js", "*.ts"]),
    ]
    PER_SUBFOLDER_CAP = 60   # max leaves to take from one subfolder
    for sub, patterns in SUBFOLDER_GLOBS:
        sub_path = root / sub
        if not sub_path.exists():
            continue
        added = 0
        for pat in patterns:
            for fp in sorted(sub_path.rglob(pat)):
                if maybe_add(fp):
                    added += 1
                if added >= PER_SUBFOLDER_CAP or len(out) >= max_files:
                    break
            if added >= PER_SUBFOLDER_CAP or len(out) >= max_files:
                break
        if len(out) >= max_files:
            return out
    return out


def world_to_projects(world: Dict[str, Any], *, max_leaves_per_branch: int = 60) -> List[Dict[str, Any]]:
    """Convert cortex_world.json into the simulator's project hierarchy.

    Each project becomes a tier-1 node; its files are grouped by
    top-level subfolder into tier-2 sub-nodes; individual file
    basenames become tier-3 leaves under their folder sub-node.

    Truncates to `max_leaves_per_branch` per folder so a project
    with thousands of files doesn't blow out the constellation.
    """
    projects_raw = dict((world or {}).get("projects") or {})
    files_raw    = (world or {}).get("files") or {}

    # Index real world files by project_id (skipping blacklisted noise).
    files_by_project: Dict[str, List[Dict[str, Any]]] = {}
    for path, fdata in files_raw.items():
        pid = fdata.get("project_id")
        if not pid:
            continue
        if _path_is_blacklisted(path):
            continue
        files_by_project.setdefault(pid, []).append({
            "path": path,
            "touches": int(fdata.get("touches", 0)),
            "last_touched_at": fdata.get("last_touched_at"),
        })

    # Supplement with scanned filesystem data for known sibling
    # project roots — gives the simulator real structure even before
    # Iris has touched anything.
    for known in _KNOWN_PROJECT_ROOTS:
        pid = known["id"]
        if pid not in projects_raw and known["root"].exists():
            projects_raw[pid] = {
                "label": known["label"],
                "root_path": str(known["root"]).replace("\\", "/"),
                "touch_count": 0,
            }
        if pid in projects_raw and not files_by_project.get(pid):
            # World knows the project but has no files for it — pull
            # the filesystem listing instead.
            scanned = _scan_project_folder(known["root"])
            if scanned:
                files_by_project[pid] = scanned

    out: List[Dict[str, Any]] = []
    color_idx = 0
    # Sort projects by touch_count desc so the most-used appear first.
    sorted_pids = sorted(
        projects_raw.keys(),
        key=lambda k: int((projects_raw[k] or {}).get("touch_count", 0)),
        reverse=True,
    )
    for pid in sorted_pids:
        pdata = projects_raw[pid] or {}
        label = pdata.get("label") or pid
        root = pdata.get("root_path")
        files = files_by_project.get(pid, [])

        # Group + truncate.
        groups = _group_files_by_folder(files, root)
        branches: List[Dict[str, Any]] = []
        for folder, fs in sorted(groups.items()):
            fs.sort(key=lambda f: -(f["touches"] or 0))
            leaves = []
            for f in fs[:max_leaves_per_branch]:
                leaves.append({
                    "id": f"proj-{_slug(pid)}::{_slug(folder)}::{_slug(os.path.basename(f['path']))}",
                    "label": os.path.basename(f["path"]),
                    "path": f["path"],
                })
            if not leaves:
                continue
            branches.append({
                "id": f"proj-{_slug(pid)}::{_slug(folder)}",
                "label": folder,
                "leaves": leaves,
            })

        # Synthetic channel sub-nodes for the socials/marketing project
        # — the actual folder is sparse (only docs/), but the project
        # mentally has many social channels worth surfacing. Each gets
        # a stub leaf or two so the sub-cloud actually has content.
        if pid == "touchless-marketing":
            for ch in _SOCIALS_CHANNELS:
                branches.append({
                    "id": f"proj-{_slug(pid)}::ch-{_slug(ch['label'])}",
                    "label": ch["label"],
                    "leaves": [
                        {
                            "id": f"proj-{_slug(pid)}::ch-{_slug(ch['label'])}::{_slug(leaf)}",
                            "label": leaf,
                        }
                        for leaf in ch["seeds"]
                    ],
                })

        # If a project has no detected files yet, still surface it as a
        # bare top-level node — it'll be the parent of just the core
        # halo, useful for "yes Iris knows about this project, but
        # hasn't touched anything inside yet."
        color = _PROJECT_COLORS[color_idx % len(_PROJECT_COLORS)]
        color_idx += 1
        out.append({
            "id": f"proj-{_slug(pid)}",
            "label": label,
            "color": color,
            "touch_count": int(pdata.get("touch_count", 0)),
            "branches": branches,
        })
    return out


# Curated channel structure for the socials/marketing project — the
# actual folder is just docs/ at the moment but Iris should see the
# real shape of the channels work happens across. Replace seeds with
# real content as it appears on disk / in a tracker.
_SOCIALS_CHANNELS = [
    {"label": "Instagram",  "seeds": ["posts", "reels", "stories", "carousels", "captions"]},
    {"label": "X (Twitter)", "seeds": ["threads", "single posts", "replies", "hooks"]},
    {"label": "TikTok",     "seeds": ["hooks", "scripts", "trends", "duets"]},
    {"label": "YouTube",    "seeds": ["long-form", "shorts", "thumbnails", "titles"]},
    {"label": "Video ideas", "seeds": ["unboxing", "before/after", "feature demo", "comparison", "tutorial"]},
    {"label": "Hooks",      "seeds": ["pain-point", "curiosity", "contrarian", "result", "scroll-stop"]},
    {"label": "Schedule",   "seeds": ["weekly cadence", "best post times", "campaigns"]},
]


def gather_memory_data(max_per_category: int = 40) -> Dict[str, List[Dict[str, Any]]]:
    """Read live memory (semantic facts + episodic interactions + an
    'active context' projection) and shape each row as a lightweight
    {id, label, text} dict for the cortex tier-3 leaves.

    Wrapped in try/except so a missing DB or broken import never
    crashes the simulator launcher — we just return empty lists.
    """
    empty = {"facts": [], "episodes": [], "active_context": []}
    try:
        from hgr.live_api.memory import MemoryManager, MemoryStore, default_memory_path
    except Exception as exc:
        print(f"WARN: memory import failed: {exc}", file=sys.stderr)
        return empty

    facts_out: List[Dict[str, Any]] = []
    episodes_out: List[Dict[str, Any]] = []
    active_out: List[Dict[str, Any]] = []

    try:
        db_path = default_memory_path()
        if not db_path.exists():
            # Fresh install — no DB yet. Don't try to create it just to read.
            return empty
        store = MemoryStore(db_path)

        try:
            facts = store.find_facts(limit=max_per_category)
            for f in facts:
                label = f"{f.kind}: {f.key}" if f.kind else f.key
                facts_out.append({
                    "id": f"mem-facts::fact-{int(f.id)}",
                    "label": label[:48],
                    "text": f"{f.kind} · {f.key} = {f.value}",
                })
        except Exception as exc:
            print(f"WARN: memory facts read failed: {exc}", file=sys.stderr)

        try:
            episodes = store.list_episodic(limit=max_per_category)
            for ep in episodes:
                txt = (ep.user_text or "").strip().replace("\n", " ")
                short = txt[:48] if txt else f"episode #{ep.id}"
                outcome = (ep.outcome or "").strip().replace("\n", " ")
                full = txt
                if outcome:
                    full = f"{txt}  →  {outcome[:160]}"
                episodes_out.append({
                    "id": f"mem-episodes::ep-{int(ep.id)}",
                    "label": short,
                    "text": full[:400],
                })
        except Exception as exc:
            print(f"WARN: memory episodes read failed: {exc}", file=sys.stderr)

        try:
            mgr = MemoryManager(store=store, async_writes=False)
            summary = mgr.summary_for_session(max_facts=max_per_category,
                                              max_chars=2000)
            if summary:
                # Split the rendered " | "-delimited block back into
                # individual leaves so the constellation shows each
                # active context fragment as its own node.
                pieces = [p.strip() for p in summary.split(" | ") if p.strip()]
                # Drop the outer "(Background context …)" wrapper if it
                # came back as a single piece.
                if pieces and pieces[0].startswith("("):
                    pieces[0] = pieces[0].lstrip("(")
                if pieces and pieces[-1].endswith(")"):
                    pieces[-1] = pieces[-1].rstrip(")")
                for i, piece in enumerate(pieces[:max_per_category]):
                    active_out.append({
                        "id": f"mem-active::ctx-{i}",
                        "label": piece[:48],
                        "text": piece[:400],
                    })
        except Exception as exc:
            print(f"WARN: memory active-context read failed: {exc}",
                  file=sys.stderr)

    except Exception as exc:
        print(f"WARN: gather_memory_data failed: {exc}", file=sys.stderr)
        return empty

    return {
        "facts": facts_out,
        "episodes": episodes_out,
        "active_context": active_out,
    }


def gather_contacts_data(max_contacts: int = 80) -> Dict[str, List[Dict[str, Any]]]:
    """Read every contact source iris currently has access to, classify
    each as 'person' or 'transactional', and return a SPLIT dict:

      {
        "people":        [...],   # real people
        "transactional": [...],   # newsletters, alerts, brand senders
      }

    Independently wrapped — a missing DB / missing connector / git not
    installed never crashes the launcher. Both lists default to ``[]``.
    """
    empty = {"people": [], "transactional": []}
    try:
        from hgr.live_api.memory import MemoryStore, default_memory_path
        from hgr.live_api.memory.contacts import gather_contacts
    except Exception as exc:
        print(f"WARN: contacts import failed: {exc}", file=sys.stderr)
        return empty

    store = None
    try:
        db_path = default_memory_path()
        if db_path.exists():
            store = MemoryStore(db_path)
    except Exception as exc:
        print(f"WARN: contacts memory open failed: {exc}", file=sys.stderr)
        store = None

    # Hand the gatherer the known sibling project roots so per-project
    # git log mining works on first run before any explicit projects
    # list is wired through.
    projects = None
    try:
        from hgr.live_api.known_projects import KNOWN_PROJECT_ROOTS
        projects = list(KNOWN_PROJECT_ROOTS)
    except Exception:
        projects = None

    try:
        contacts = gather_contacts(store, projects=projects,
                                   max_contacts=max_contacts)
    except Exception as exc:
        print(f"WARN: gather_contacts failed: {exc}", file=sys.stderr)
        return empty

    # Split by classify_contact() kind (set in gather_contacts).
    people: List[Dict[str, Any]] = []
    transactional: List[Dict[str, Any]] = []
    for c in contacts:
        if c.get("kind") == "transactional":
            transactional.append(c)
        else:
            people.append(c)

    # Enrich real people with relevance tags (typical_hours / days /
    # related tools / related projects / mention_count). Transactional
    # senders don't need this — they're newsletters/alerts and their
    # mention pattern is meaningless. Wrapped: a broken enrich call
    # must not strip the people list itself.
    try:
        from hgr.live_api.memory.contacts import enrich_contacts_with_relevance
        from hgr.live_api.cortex.tool_call_log import default_log_path as _tool_log_path
        try:
            tool_log = _tool_log_path()
        except Exception:
            tool_log = None
        enrich_contacts_with_relevance(
            people,
            store,
            tool_call_log_path=tool_log,
            projects=projects,
        )
    except Exception as exc:
        print(f"WARN: enrich_contacts_with_relevance failed: {exc}",
              file=sys.stderr)

    # Log aggregates only — never raw addresses (PII).
    try:
        src_counts: Dict[str, int] = {}
        for c in contacts:
            for s in (c.get("sources") or []):
                base = s.split("[")[0]
                src_counts[base] = src_counts.get(base, 0) + 1
        print("Loaded {} contact(s) - {} people, {} transactional [{}]".format(
            len(contacts), len(people), len(transactional),
            ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items())),
        ))
    except Exception:
        pass
    return {"people": people, "transactional": transactional}


def gather_patterns(max_patterns: int = _MAX_PATTERNS,
                    days: int = 7,
                    budget_s: float = 2.0,
                    skip_embedder: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Surface repeated user-query patterns + tool-sequence patterns
    from the live MemoryStore, shaped as flat ``{id, label, count,
    kind}`` dicts for the cortex tier-3 leaves under a new mem-patterns
    sub-node.

    ``kind`` is ``'query'`` for clustered query habits and
    ``'tool_seq'`` for repeated tool-pair transitions.

    Wrapped in try/except so a missing DB / unloaded embedder / slow
    embed call never crashes or blocks the simulator launcher. If the
    analysis takes longer than ``budget_s`` seconds, we log a warning,
    return what we have, and skip the rest.
    """
    out: List[Dict[str, Any]] = []
    started = time.monotonic()
    try:
        from hgr.live_api.memory import (
            MemoryStore,
            default_embedder,
            default_memory_path,
            find_repeated_patterns,
            find_tool_sequences,
        )
    except Exception as exc:
        print(f"WARN: patterns import failed: {exc}", file=sys.stderr)
        return []

    try:
        db_path = default_memory_path()
        if not db_path.exists():
            # Fresh install — no DB yet. Quietly return [].
            return []
        store = MemoryStore(db_path)
    except Exception as exc:
        print(f"WARN: pattern store open failed: {exc}", file=sys.stderr)
        return []

    # ---- Query-habit clustering (embedding-based). May make HTTPS
    # calls to OpenAI; tolerated to fail individually. We bail early
    # if we've already burned more than half the budget here.
    # Skip entirely on fast-path / first launch — the embedder block is
    # 10+ HTTPS calls and the background daemon will fill this in.
    if not skip_embedder:
        try:
            embedder = default_embedder()
            # Bound the whole find_repeated_patterns call with a timeout
            # so a slow OpenAI connection doesn't stall the launcher.
            ok, result = _run_with_timeout(
                lambda: find_repeated_patterns(store, embedder, days=days),
                timeout_s=_PATTERN_FAST_PATH_BUDGET_S,
                label="find_repeated_patterns",
            )
            if ok and result and not result.get("cold_start"):
                for p in result.get("patterns") or []:
                    lbl = (p.get("label") or "").strip() or "(query)"
                    count = int(p.get("count") or 0)
                    last_seen = float(p.get("last_seen_at") or 0.0)
                    # Stable id from label + count + ts so repeated runs
                    # don't churn the constellation.
                    pid = f"mem-patterns::q-{abs(hash((lbl, count, int(last_seen)))) % 10_000_000}"
                    out.append({
                        "id": pid,
                        "label": lbl[:48],
                        "count": count,
                        "kind": "query",
                    })
        except Exception as exc:
            print(f"WARN: find_repeated_patterns failed: {exc}", file=sys.stderr)

    if time.monotonic() - started > budget_s:
        print(
            f"WARN: pattern detection exceeded {budget_s}s budget; "
            f"skipping tool-sequence pass",
            file=sys.stderr,
        )
        return out[:max_patterns]

    # ---- Tool-sequence pattern (steps-json scan, no embedder needed).
    try:
        seq_result = find_tool_sequences(store, days=days)
        for s in seq_result.get("sequences") or []:
            from_tool = str(s.get("from_tool") or "?")
            to_tool = str(s.get("to_tool") or "?")
            count = int(s.get("count") or 0)
            last_seen = float(s.get("last_seen_at") or 0.0)
            lbl = f"{from_tool} -> {to_tool}"
            pid = f"mem-patterns::seq-{abs(hash((from_tool, to_tool, count))) % 10_000_000}"
            out.append({
                "id": pid,
                "label": lbl[:48],
                "count": count,
                "kind": "tool_seq",
            })
    except Exception as exc:
        print(f"WARN: find_tool_sequences failed: {exc}", file=sys.stderr)

    elapsed = time.monotonic() - started
    if elapsed > budget_s:
        print(
            f"WARN: pattern detection took {elapsed:.2f}s (budget {budget_s}s)",
            file=sys.stderr,
        )

    # Cap total leaves to keep the constellation legible.
    return out[:max_patterns]


def gather_suggestions(contacts: Optional[List[Dict[str, Any]]] = None,
                       max_suggestions: int = _MAX_SUGGESTIONS) -> List[Dict[str, Any]]:
    """Run the SuggestionEngine and return its ranked next-action list.

    Wrapped in try/except so a missing module / broken DB / unloaded
    embedder never crashes the launcher — we just return [] and the
    simulator's suggestions sub-node renders empty.

    Args:
        contacts: optional list of enriched contact dicts (from
            ``gather_contacts_data().get('people')``). Lets the engine
            surface ``follow_up_person`` suggestions whose typical_hours
            match now. When omitted the person path is silently skipped.
        max_suggestions: hard cap on the returned list.
    """
    try:
        from hgr.live_api.learning import SuggestionEngine
        from hgr.live_api.memory import (
            MemoryManager,
            MemoryStore,
            default_memory_path,
        )
    except Exception as exc:
        print(f"WARN: suggestions import failed: {exc}", file=sys.stderr)
        return []

    try:
        db_path = default_memory_path()
        if not db_path.exists():
            # Fresh install — no memory yet. Quietly return [].
            return []
        store = MemoryStore(db_path)
        manager = MemoryManager(store=store, async_writes=False)
    except Exception as exc:
        print(f"WARN: suggestions memory open failed: {exc}",
              file=sys.stderr)
        return []

    tool_log_path = None
    try:
        from hgr.live_api.cortex.tool_call_log import default_log_path
        tool_log_path = default_log_path()
    except Exception:
        tool_log_path = None

    try:
        engine = SuggestionEngine(
            memory_manager=manager,
            tool_call_log_path=tool_log_path,
            contacts=contacts,
        )
        return engine.suggest_now(max_suggestions=max_suggestions)
    except Exception as exc:
        print(f"WARN: SuggestionEngine.suggest_now failed: {exc}",
              file=sys.stderr)
        return []


def gather_tool_cooccurrence(min_weight: int = 2,
                             max_edges: int = 60) -> List[Dict[str, Any]]:
    """Read the persistent tool_call_log.db and return cross-tool
    co-occurrence edges shaped for ``world.tool_links``.

    Each edge is ``{from, to, weight}`` where ``weight`` is the number
    of distinct sessions both tools fired in. Pairs below ``min_weight``
    are filtered out; result is capped at ``max_edges`` to keep the
    visualization legible.

    Wrapped in try/except so a missing module / unreadable DB never
    crashes the launcher — returns [] and the simulator simply
    renders no cross-edges.
    """
    try:
        from hgr.live_api.cortex.tool_call_log import compute_cooccurrence
    except Exception as exc:
        print(f"WARN: tool_call_log import failed: {exc}", file=sys.stderr)
        return []
    try:
        return compute_cooccurrence(min_weight=min_weight, max_edges=max_edges)
    except Exception as exc:
        print(f"WARN: compute_cooccurrence failed: {exc}", file=sys.stderr)
        return []


def list_all_tools(max_tools: int = 30) -> List[Dict[str, Any]]:
    """Gather the real Iris tool registry (built-in + connector schemas)
    so the cortex visualization can show actual capabilities instead of
    the 5 hardcoded aliases (router/weather/open-app/messages/calendar).

    Returns up to ``max_tools`` entries as
    ``[{id, label, description, usage_count}, ...]``. Sorted by
    ``usage_count`` desc when available (currently 0 for all — episode
    log aggregation is a future enhancement); ties broken by id.

    Wrapped in try/except so a missing import never crashes the
    launcher — we just return [] and the simulator keeps its hardcoded
    fallback tools.
    """
    try:
        from hgr.live_api.schemas import all_tool_schemas
    except Exception as exc:
        print(f"WARN: tool schemas import failed: {exc}", file=sys.stderr)
        return []

    # Built-in tools — always available.
    try:
        builtin = list(all_tool_schemas() or [])
    except Exception as exc:
        print(f"WARN: all_tool_schemas() failed: {exc}", file=sys.stderr)
        builtin = []

    # Connector tools — currently-available connectors only (Gmail
    # disappears when google auth is missing, etc.). Failure to build
    # the registry is non-fatal; we just skip the connector half.
    connector_tools: List[Dict[str, Any]] = []
    try:
        from hgr.live_api.connectors import build_connector_registry
        registry = build_connector_registry()
        connector_tools = list(registry.available_tool_schemas() or [])
    except Exception as exc:
        print(f"WARN: connector registry build failed: {exc}", file=sys.stderr)

    all_schemas = builtin + connector_tools
    seen: set = set()
    result: List[Dict[str, Any]] = []
    for schema in all_schemas:
        tool_id = (schema or {}).get("name") or ""
        if not tool_id or tool_id in seen:
            continue
        seen.add(tool_id)
        result.append({
            "id": tool_id,
            "label": tool_id.replace("_", " ").title(),
            "description": (schema.get("description") or "")[:280],
            # usage_count — would require parsing live_api.memory's
            # episode log; left at 0 for now so all tools render at the
            # same priority. Sort key still works (stable on id).
            "usage_count": 0,
        })

    result.sort(key=lambda x: (-int(x.get("usage_count", 0)), x["id"]))
    if max_tools and len(result) > max_tools:
        result = result[:max_tools]
    return result


def _run_auto_discovery() -> List[Dict[str, Any]]:
    """Best-effort: scan the filesystem for project folders not already
    in KNOWN_PROJECT_ROOTS, cache the result, and return the merged
    cache. Wrapped in try/except — never crashes the launcher.

    Returns a list of ``{id, label, root}`` dicts; each carries an
    ``auto: True`` marker so downstream renderers can tell them apart
    from the hardcoded sibling projects.

    Applies the user's exclusion list (project_exclusions.json) via
    the underlying discover_and_cache / load_cached_auto_discovered
    helpers — no per-call wiring needed here.
    """
    try:
        from hgr.live_api.project_autodetect import (
            discover_and_cache,
            load_cached_auto_discovered,
        )
        exclude = {Path(e["root"]) for e in _KNOWN_PROJECT_ROOTS}
        fresh = discover_and_cache(exclude_roots=exclude)
        if fresh:
            print(f"Auto-discovered {len(fresh)} candidate project(s)")
            return [{**p, "auto": True} for p in fresh]
        # Scan returned nothing this run — fall back to last cache so
        # we still surface previously-found projects.
        cached = load_cached_auto_discovered()
        return [{**p, "auto": True} for p in cached]
    except Exception as exc:
        print(f"WARN: project auto-discovery failed: {exc}", file=sys.stderr)
        return []


def _filter_world_projects_by_exclusions(world: Dict[str, Any]) -> Dict[str, Any]:
    """Strip projects/files whose ids or labels match the user's
    exclusion list from the in-memory ``world`` dict. Returns the same
    dict (mutated in place) for fluent use.

    Wrapped in try/except — exclusion filtering is decorative and
    must NEVER block the simulator from loading the rest of the world.

    KNOWN_PROJECT_ROOTS slug ids are protected: they're never filtered
    even if an exclusion happens to match their id/label, so the four
    hardcoded sibling projects can't be hidden by accident.
    """
    try:
        from hgr.live_api.project_autodetect import (
            load_project_exclusions,
            is_project_excluded,
        )
    except Exception:
        return world
    try:
        exclusions = load_project_exclusions()
    except Exception:
        exclusions = []
    if not exclusions:
        return world
    # Protected ids (hardcoded roots — never excludable via the user list).
    protected: set = {p["id"] for p in _KNOWN_PROJECT_ROOTS}
    projects = dict((world or {}).get("projects") or {})
    if not projects:
        return world
    removed: set = set()
    kept: Dict[str, Any] = {}
    for pid, pdata in projects.items():
        if pid in protected:
            kept[pid] = pdata
            continue
        label = (pdata or {}).get("label") or pid
        if is_project_excluded(pid, label, exclusions=exclusions):
            removed.add(pid)
            continue
        kept[pid] = pdata
    if not removed:
        return world
    world["projects"] = kept
    # Also strip files whose project_id points at a now-removed
    # project so the simulator doesn't render orphaned leaves under
    # a missing parent. Files without a project_id (catch-all bucket)
    # are left intact.
    files = (world or {}).get("files") or {}
    if isinstance(files, dict) and files:
        world["files"] = {
            path: fdata
            for path, fdata in files.items()
            if (fdata or {}).get("project_id") not in removed
        }
    print(
        f"Filtered {len(removed)} excluded project(s) from world: "
        f"{sorted(removed)}",
        file=sys.stderr,
    )
    return world


def load_world_payload() -> Dict[str, Any]:
    """Read cortex_world.json (or fall back to known-folder scan) and
    return the simulator-shaped payload. Safe to call from any
    launcher that wants to feed real data into iris_simulator.html.

    Also runs a one-shot, best-effort filesystem scan for project
    folders not already in ``KNOWN_PROJECT_ROOTS``; results are cached
    so subsequent startups don't re-walk the disk.
    """
    world_path = _default_world_path()
    world: Dict[str, Any] = {}
    source = "missing"
    if world_path.exists():
        try:
            world = json.loads(world_path.read_text(encoding="utf-8"))
            source = str(world_path)
        except Exception as exc:
            print(f"WARN: failed to parse {world_path}: {exc}", file=sys.stderr)
            world = {}

    # User-driven exclusion list — drop projects (and their files)
    # whose ids/labels match before we hand the world off to
    # world_to_projects. Wrapped in try/except inside the helper;
    # KNOWN_PROJECT_ROOTS are protected.
    try:
        world = _filter_world_projects_by_exclusions(world)
    except Exception as exc:
        print(f"WARN: exclusion filter failed: {exc}", file=sys.stderr)

    projects = world_to_projects(world)

    # Auto-discovered siblings — folded into the simulator's project
    # list with their own tier-2/3 sub-branches generated by the same
    # ``_scan_project_folder`` + ``_group_files_by_folder`` pipeline
    # used for hardcoded roots. Marked ``auto: True`` so the cortex UI
    # can tint or label them differently later if desired.
    try:
        auto_projects = _run_auto_discovery()
        if auto_projects:
            known_ids = {p.get("id") for p in projects}
            known_roots: set = set()
            for p in _KNOWN_PROJECT_ROOTS:
                try:
                    known_roots.add(str(p["root"].resolve()).lower())
                except Exception:
                    pass

            start_idx = len(projects)
            for i, ap in enumerate(auto_projects):
                pid = f"auto-{ap['id']}"
                if pid in known_ids:
                    continue
                root = ap["root"]
                try:
                    root_norm = str(root.resolve()).lower()
                except Exception:
                    root_norm = None
                if root_norm and root_norm in known_roots:
                    continue

                scanned = _scan_project_folder(root)
                groups = _group_files_by_folder(scanned, str(root))
                branches: List[Dict[str, Any]] = []
                for folder, fs in sorted(groups.items()):
                    fs.sort(key=lambda f: -(f.get("touches") or 0))
                    leaves = []
                    for f in fs[:60]:
                        leaves.append({
                            "id": (
                                f"proj-{_slug(pid)}::{_slug(folder)}"
                                f"::{_slug(os.path.basename(f['path']))}"
                            ),
                            "label": os.path.basename(f["path"]),
                            "path": f["path"],
                        })
                    if not leaves:
                        continue
                    branches.append({
                        "id": f"proj-{_slug(pid)}::{_slug(folder)}",
                        "label": folder,
                        "leaves": leaves,
                    })

                color = _PROJECT_COLORS[(start_idx + i) % len(_PROJECT_COLORS)]
                projects.append({
                    "id": f"proj-{_slug(pid)}",
                    "label": ap.get("label") or ap["id"],
                    "color": color,
                    "touch_count": 0,
                    "branches": branches,
                    "auto_discovered": True,
                })
                known_ids.add(pid)
    except Exception as exc:
        # Catch-all: auto-discovery is decorative — never let it block
        # the simulator from rendering the cortex_world.json projects.
        print(f"WARN: auto-discovery merge failed: {exc}", file=sys.stderr)
    # ── Cache + fast-path planning ─────────────────────────────────
    # Memoize the expensive gather_patterns + gather_suggestions output
    # at %LOCALAPPDATA%/Touchless/iris_payload_cache.json with a short
    # TTL. Re-launches within the TTL (and with no memory.db change)
    # skip the embedder entirely. Cache only covers slow sources; the
    # cheap parts (memory/contacts/tools) are always recomputed so the
    # cortex picks up new episodes / contacts immediately.
    mem_mtime = _memory_db_mtime()
    _cache = _load_cache()
    _patterns_entry = _cache.get("patterns")
    _suggestions_entry = _cache.get("suggestions")
    _contacts_entry = _cache.get("contacts")
    fresh_contacts = (
        _contacts_entry.get("data")
        if _cache_entry_fresh(_contacts_entry, mem_mtime) else None
    )
    fresh_patterns = (
        _patterns_entry.get("data")
        if _cache_entry_fresh(_patterns_entry, mem_mtime) else None
    )
    fresh_suggestions = (
        _suggestions_entry.get("data")
        if _cache_entry_fresh(_suggestions_entry, mem_mtime) else None
    )
    # Fresh install (no memory.db) → fast-path: skip embedder entirely.
    # The cortex Patterns/Suggestions nodes render empty initially;
    # the background daemon (or the next launch within TTL) fills them.
    first_launch = (mem_mtime == 0.0)
    skip_embedder = first_launch and (fresh_patterns is None)

    # ── Parallel gather_* execution ───────────────────────────────
    # Run the five independent gathers concurrently. Previously they
    # were serial and the slowest source dominated wall time. Each one
    # is already individually try/excepted, so failures land in the
    # default value below instead of bubbling out.
    memory: Dict[str, Any] = {"facts": [], "episodes": [], "active_context": []}
    tools: List[Dict[str, Any]] = []
    tool_links: List[Dict[str, Any]] = []
    contacts_split: Dict[str, List[Dict[str, Any]]] = {"people": [], "transactional": []}
    patterns_out: List[Dict[str, Any]] = []

    def _safe(label, fn, default):
        try:
            return fn()
        except Exception as exc:
            print(f"WARN: {label} crashed: {exc}", file=sys.stderr)
            return default

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as _ex:
        fut_memory = _ex.submit(
            _safe, "gather_memory_data", gather_memory_data,
            {"facts": [], "episodes": [], "active_context": []},
        )
        # Contacts: reuse cache when fresh — avoids gmail/ms365 HTTPS
        # timeouts on every launch. Gmail timeouts alone add 6s+ per
        # call (2x sources × 3s each). Cache TTL bypasses them.
        if fresh_contacts is not None:
            fut_contacts = _ex.submit(lambda: dict(fresh_contacts))
        else:
            fut_contacts = _ex.submit(
                _safe, "gather_contacts_data", gather_contacts_data,
                {"people": [], "transactional": []},
            )
        fut_tools = _ex.submit(
            _safe, "list_all_tools", list_all_tools, [],
        )
        fut_tool_links = _ex.submit(
            _safe, "gather_tool_cooccurrence", gather_tool_cooccurrence, [],
        )
        # Patterns: reuse the cache when fresh; otherwise compute (skipping
        # the embedder block on first launch). Either way bounded to
        # _MAX_PATTERNS.
        if fresh_patterns is not None:
            fut_patterns = _ex.submit(lambda: list(fresh_patterns)[:_MAX_PATTERNS])
        else:
            fut_patterns = _ex.submit(
                _safe, "gather_patterns",
                lambda: gather_patterns(
                    max_patterns=_MAX_PATTERNS,
                    skip_embedder=skip_embedder,
                ),
                [],
            )

        memory = fut_memory.result() or memory
        contacts_split = fut_contacts.result() or contacts_split
        tools = fut_tools.result() or []
        tool_links = fut_tool_links.result() or []
        patterns_out = fut_patterns.result() or []

    memory["patterns"] = patterns_out[:_MAX_PATTERNS]
    memory["contacts"] = contacts_split.get("people", [])
    memory["transactional"] = contacts_split.get("transactional", [])

    # SuggestionEngine — ranked next-action recommendations. Depends on
    # the enriched contacts list (so it can surface follow_up_person
    # suggestions whose typical_hours match now), so runs AFTER the
    # parallel block. Uses the cache when fresh; embedder-skipped on
    # first launch (returns []).
    if fresh_suggestions is not None:
        suggestions = list(fresh_suggestions)[:_MAX_SUGGESTIONS]
    elif skip_embedder:
        # Background daemon / next launch will fill these in.
        suggestions = []
    else:
        try:
            ok, suggestions = _run_with_timeout(
                lambda: gather_suggestions(
                    contacts=memory.get("contacts") or [],
                    max_suggestions=_MAX_SUGGESTIONS,
                ),
                timeout_s=_PATTERN_FAST_PATH_BUDGET_S,
                label="gather_suggestions",
            )
            if not ok or not suggestions:
                suggestions = []
        except Exception as exc:
            print(f"WARN: gather_suggestions crashed: {exc}", file=sys.stderr)
            suggestions = []

    # Persist the recomputed slow values into the cache so the next
    # launch within the TTL skips them entirely. Mtime is captured at
    # write time so future launches can detect "memory.db changed since".
    try:
        now_ts = time.time()
        new_cache = dict(_cache)
        if fresh_patterns is None and not skip_embedder:
            new_cache["patterns"] = {
                "data": memory.get("patterns") or [],
                "written_at": now_ts,
                "memory_mtime": mem_mtime,
            }
        if fresh_suggestions is None and not skip_embedder:
            new_cache["suggestions"] = {
                "data": suggestions or [],
                "written_at": now_ts,
                "memory_mtime": mem_mtime,
            }
        if fresh_contacts is None and contacts_split:
            new_cache["contacts"] = {
                "data": contacts_split,
                "written_at": now_ts,
                "memory_mtime": mem_mtime,
            }
        # Only write if something actually changed.
        if new_cache != _cache:
            _save_cache(new_cache)
    except Exception as exc:
        print(f"WARN: payload cache update failed: {exc}", file=sys.stderr)

    # ---- Demo Project (re-injected every launch) ----
    # Curated tier-1 entry so the user can practice the
    # ``iris_remove_project`` flow: removal appends to the exclusion
    # list for the current session, but since this block injects the
    # project AFTER exclusion filtering, it always re-appears on the
    # next launch. Distinct purple tint flags it as a test fixture so
    # the user knows it isn't a real project. Id ``demo-project`` is
    # NOT in KNOWN_PROJECT_ROOTS, so the protected-project guard in
    # ``iris_remove_project`` lets it through.
    try:
        # Avoid double-injection if some upstream change adds a
        # collision (defensive — current code never injects this id
        # from any other path).
        if not any(p.get("id") == "proj-demo-project" for p in projects):
            projects.append({
                "id": "proj-demo-project",
                "label": "Demo Project",
                # Purple — visually distinct from the teal/green
                # `_PROJECT_COLORS` palette so it's obviously a test
                # fixture, not a real project.
                "color": {
                    "base": 0xb47cff,
                    "hot":  0xe9d6ff,
                    "halo": 0xb47cff,
                },
                "touch_count": 0,
                "branches": [
                    {
                        "id": "proj-demo-project::tasks",
                        "label": "Tasks",
                        "leaves": [
                            {
                                "id": "proj-demo-project::tasks::task-1",
                                "label": "task-1.md",
                                "path": "demo/tasks/task-1.md",
                            },
                            {
                                "id": "proj-demo-project::tasks::task-2",
                                "label": "task-2.md",
                                "path": "demo/tasks/task-2.md",
                            },
                            {
                                "id": "proj-demo-project::tasks::task-3",
                                "label": "task-3.md",
                                "path": "demo/tasks/task-3.md",
                            },
                        ],
                    },
                    {
                        "id": "proj-demo-project::notes",
                        "label": "Notes",
                        "leaves": [
                            {
                                "id": "proj-demo-project::notes::note-1",
                                "label": "note-1.md",
                                "path": "demo/notes/note-1.md",
                            },
                            {
                                "id": "proj-demo-project::notes::note-2",
                                "label": "note-2.md",
                                "path": "demo/notes/note-2.md",
                            },
                        ],
                    },
                ],
                # Marks this as a runtime-injected demo so any later
                # tooling can distinguish it from real auto-discovered
                # projects. Not protected — the iris_remove_project
                # protection list is hardcoded to KNOWN_PROJECT_ROOTS,
                # so the user can remove this fixture for the current
                # session; it will re-appear on next launch.
                "demo_fixture": True,
            })
    except Exception as exc:
        # Demo fixture is decorative — never let it block the payload.
        print(f"WARN: demo project injection failed: {exc}", file=sys.stderr)

    # Test-fill: append 5 stub projects so we can eyeball how the layout
    # handles a denser tier-1 ring. OFF by default; opt-in by setting
    # TOUCHLESS_IRIS_TEST_PROJECTS=1 (used during layout testing only).
    if os.environ.get("TOUCHLESS_IRIS_TEST_PROJECTS", "0") == "1":
        projects.extend(_build_test_projects(len(projects)))

    return {
        "source": source,
        "project_count": len(projects),
        "projects": projects,
        "memory": memory,
        "tools": tools,
        "tool_links": tool_links,
        "suggestions": suggestions,
    }


# 5 stub projects for layout testing — generic names so it's obvious
# they're placeholders. Each has 3-4 tier-2 branches and 4-6 tier-3
# leaves per branch. Color palette continues from _PROJECT_COLORS.
_TEST_PROJECTS = [
    {
        "label": "Test A",
        "branches": [
            ("Backend",  ["api", "db", "auth", "queue", "cache"]),
            ("Frontend", ["pages", "components", "store", "router"]),
            ("Ops",      ["ci", "deploy", "monitoring"]),
        ],
    },
    {
        "label": "Test B",
        "branches": [
            ("Research", ["spec", "prototype", "benchmarks"]),
            ("Design",   ["wireframes", "mockups", "tokens", "icons"]),
            ("Docs",     ["readme", "guides", "api-ref", "changelog", "examples"]),
            ("Tests",    ["unit", "integration", "e2e"]),
        ],
    },
    {
        "label": "Test C",
        "branches": [
            ("Engine",  ["loop", "renderer", "physics", "audio", "input"]),
            ("Assets",  ["models", "textures", "shaders", "sfx"]),
            ("Build",   ["pack", "sign", "ship"]),
        ],
    },
    {
        "label": "Test D",
        "branches": [
            ("Data",       ["ingest", "transform", "store"]),
            ("Pipeline",   ["schedule", "validate", "publish", "alert"]),
            ("Dashboards", ["overview", "drilldown", "alerts", "ad-hoc"]),
        ],
    },
    {
        "label": "Test E",
        "branches": [
            ("Mobile",  ["ios", "android", "shared"]),
            ("Server",  ["sync", "push", "billing", "analytics"]),
            ("Web",     ["landing", "dashboard", "settings"]),
            ("Support", ["faq", "ticket-flow", "macros"]),
        ],
    },
]


def _build_test_projects(start_color_idx: int) -> List[Dict[str, Any]]:
    """Materialize the test project stubs into the simulator's project
    schema. Cycles through the project color palette starting at the
    given offset so test nodes get distinct tints from real ones.
    """
    out: List[Dict[str, Any]] = []
    for i, proj in enumerate(_TEST_PROJECTS):
        color = _PROJECT_COLORS[(start_color_idx + i) % len(_PROJECT_COLORS)]
        pslug = _slug(proj["label"])
        branches: List[Dict[str, Any]] = []
        for branch_name, leaves in proj["branches"]:
            bslug = _slug(branch_name)
            branches.append({
                "id": f"proj-{pslug}::{bslug}",
                "label": branch_name,
                "leaves": [
                    {
                        "id": f"proj-{pslug}::{bslug}::{_slug(leaf)}",
                        "label": leaf,
                    }
                    for leaf in leaves
                ],
            })
        out.append({
            "id": f"proj-{pslug}",
            "label": proj["label"],
            "color": color,
            "touch_count": 0,
            "branches": branches,
        })
    return out


def install_world_injection(view, payload: Dict[str, Any]) -> None:
    """Attach a DocumentCreation QWebEngineScript to ``view`` so
    ``window.IRIS_WORLD`` is set BEFORE any page's own module script
    runs. Survives in-page navigations (e.g. demos/index.html →
    iris_simulator.html), so the gallery launcher can hand off real
    data to the simulator without re-wiring anything.
    """
    from PySide6.QtWebEngineCore import QWebEngineScript

    payload_json = json.dumps(payload, ensure_ascii=False)
    js_literal = json.dumps(payload_json)

    script = QWebEngineScript()
    script.setName("IrisWorldInjection")
    script.setSourceCode(f"window.IRIS_WORLD = JSON.parse({js_literal});")
    script.setInjectionPoint(QWebEngineScript.DocumentCreation)
    script.setWorldId(QWebEngineScript.MainWorld)
    script.setRunsOnSubFrames(False)
    view.page().scripts().insert(script)


def install_qwebchannel_bootstrap(view) -> None:
    """Inject Qt's bundled ``qwebchannel.js`` at DocumentCreation so
    the iris_simulator's bridge-setup code can reference the
    ``QWebChannel`` constructor synchronously. Without this, the
    bridge handshake would race the page load.

    Qt ships qwebchannel.js as a Qt resource at
    ``qrc:///qtwebchannel/qwebchannel.js``. We can't reference a qrc
    URL from a file:// page via a <script src>, so we read the file
    out of the Qt resource system and inline it into a
    QWebEngineScript instead.
    """
    from PySide6.QtCore import QFile, QIODevice
    from PySide6.QtWebEngineCore import QWebEngineScript

    qf = QFile(":/qtwebchannel/qwebchannel.js")
    if not qf.open(QIODevice.ReadOnly):
        print(
            "WARN: qwebchannel.js not found in Qt resources — live cortex "
            "updates will be disabled (page falls back to static IRIS_WORLD).",
            file=sys.stderr,
        )
        return
    try:
        src_bytes = qf.readAll().data()
        src_text = src_bytes.decode("utf-8", errors="replace")
    finally:
        qf.close()

    script = QWebEngineScript()
    script.setName("QWebChannelBootstrap")
    script.setSourceCode(src_text)
    script.setInjectionPoint(QWebEngineScript.DocumentCreation)
    script.setWorldId(QWebEngineScript.MainWorld)
    script.setRunsOnSubFrames(False)
    view.page().scripts().insert(script)


def wire_live_cortex_bridge(view):
    """Build the live-update bridge: CortexBridge → world_state observer.

    Returns (bridge, channel) — the caller MUST hold a reference for the
    lifetime of the QWebEngineView, otherwise QWebChannel garbage-collects
    the registered object and silently drops events.

    Threading: world_state observer callbacks fire from whatever thread
    called touch_file/add_project. Signal.emit() auto-queues to the GUI
    thread for QWebChannel marshaling — safe to call from anywhere.

    Page-reload safety: this function is idempotent enough to be called
    again from a ``loadFinished`` handler if the page reloads. The
    channel registration survives reload (QWebChannel re-handshakes
    automatically once the JS-side QWebChannel constructor runs again).
    """
    from PySide6.QtWebChannel import QWebChannel
    from hgr.live_api.cortex.bridge import CortexBridge, set_active_bridge

    bridge = CortexBridge(view)
    channel = QWebChannel(view.page())
    channel.registerObject("cortex", bridge)
    view.page().setWebChannel(channel)

    # JS-injection fallback for emit_project_added / emit_project_removed
    # so the visual fade happens even if the QWebChannel signal drops.
    try:
        _page = view.page()
        def _run_js(src: str, _p=_page) -> None:
            try:
                _p.runJavaScript(src)
            except Exception:
                pass
        bridge.set_js_runner(_run_js)
    except Exception as exc:
        print(f"WARN: bridge.set_js_runner failed: {exc}", file=sys.stderr)

    # Register the bridge module-globally so non-GUI subsystems
    # (tool_executor → emit_tool_used, world_state → emit_project_added /
    # emit_leaf_added, memory.manager → emit_pattern_added) can push
    # discrete signals without taking a hard import dependency on this
    # launcher. They all look the bridge up via get_active_bridge() and
    # no-op when it's None — so non-simulator runs are unaffected.
    #
    # Signal.emit() auto-queues to the GUI thread for safe QWebChannel
    # marshaling, so callers from any thread (touch worker, fact-extract
    # background thread, …) are safe.
    try:
        set_active_bridge(bridge)
    except Exception as exc:
        print(f"WARN: set_active_bridge failed: {exc}", file=sys.stderr)

    return bridge, channel


def main() -> int:
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QApplication, QMainWindow
    from PySide6.QtWebEngineWidgets import QWebEngineView

    payload = load_world_payload()
    projects = payload["projects"]
    tools = payload.get("tools", [])
    print(f"Loaded {len(projects)} project(s) from {payload['source']}")
    for p in projects[:6]:
        n_leaves = sum(len(b["leaves"]) for b in p["branches"])
        print(f"  - {p['label']}: {len(p['branches'])} branch(es), {n_leaves} leaf(ves)")
    print(f"Loaded {len(tools)} tool(s) into the cortex tools branch")

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Iris Cortex — Live Simulator (real data)")
    win.resize(1320, 880)

    view = QWebEngineView()
    # Static world payload injection (existing behavior).
    install_world_injection(view, payload)
    # QWebChannel bootstrap script — MUST be inserted BEFORE the page
    # loads so the simulator's bridge-setup code can reference
    # `QWebChannel` synchronously.
    install_qwebchannel_bootstrap(view)
    # Live update bridge — kept on `win` so it outlives the function
    # scope and the QWebChannel registration survives until the window
    # closes. Without holding a reference, QWebChannel GCs the bridge
    # and drops every event silently.
    win._cortex_bridge, win._cortex_channel = wire_live_cortex_bridge(view)

    html_path = (
        SRC / "hgr" / "live_api" / "cortex" / "web" / "demos" / "iris_simulator.html"
    )
    if not html_path.exists():
        print(f"ERROR: simulator HTML missing at {html_path}", file=sys.stderr)
        return 2
    view.load(QUrl.fromLocalFile(str(html_path)))
    win.setCentralWidget(view)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())

# Author: Konstantin Markov

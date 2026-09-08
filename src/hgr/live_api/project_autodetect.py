"""Auto-detect candidate project folders for the cortex / RAG indexer.

Best-effort scanner that looks under a small handful of common
parent directories for folders that look like real projects — README,
package.json, pyproject.toml, or .git/ markers. Results are cached
on disk so we don't re-walk the filesystem every startup.

Designed to never crash the caller: every entry-point is wrapped in
try/except and degrades to an empty list / dict on any failure.

Two consumers today:

  * ``run_iris_simulator.py`` — merges auto-discovered projects into
    the simulator's project list (alongside ``KNOWN_PROJECT_ROOTS``)
    so the cortex visualization shows folders the user actually has
    on disk, not just the four hardcoded sibling roots.
  * ``known_projects.combined_project_roots()`` — exposes the merged
    list for any other consumer that wants both sources.

Cache location: ``%LOCALAPPDATA%\\Touchless\\auto_discovered_projects.json``
on Windows, ``~/.touchless/auto_discovered_projects.json`` elsewhere.
Override via ``TOUCHLESS_AUTO_DISCOVER_CONFIG``.

Exclusion list: ``%LOCALAPPDATA%\\Touchless\\project_exclusions.json``
(override via ``TOUCHLESS_PROJECT_EXCLUSIONS``). Stores a list of
slug IDs / labels (case-insensitive substring match) that should be
suppressed from auto-discovery results. Pre-seeded with a handful of
known-noise projects on first read so the user doesn't see them by
default — see ``_DEFAULT_EXCLUSIONS`` below.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# --- Caps + skip-lists ----------------------------------------------------

# Hard depth cap: each scan root descends at most this many levels.
_MAX_DEPTH = 2

# Hard total cap on candidate folders examined across the whole scan,
# so a deep ~/Documents tree can't lock us up.
_MAX_CANDIDATES_EXAMINED = 200

# Project markers — at least one must exist in the folder for it to
# count as a "project."
_PROJECT_MARKERS = ("README.md", "package.json", "pyproject.toml", ".git")

# Folder name fragments we never recurse into. Lowercased for
# case-insensitive compare on Windows.
_NAME_BLOCKLIST = {
    "appdata", "windows", "program files", "program files (x86)",
    "programdata", "system32", "$recycle.bin",
    "system volume information", "recovery",
    "node_modules", "__pycache__", ".pytest_cache",
    "venv", ".venv", "env",
    "build", "dist", "out", "target",
}


# --- Cache path -----------------------------------------------------------

def _cache_path() -> Path:
    """Return the on-disk cache location for auto-discovered projects.

    Mirrors the env-override + AppData pattern used elsewhere in the
    codebase (cortex_world.json, mcp_servers.json).
    """
    override = os.environ.get("TOUCHLESS_AUTO_DISCOVER_CONFIG")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Touchless" / "auto_discovered_projects.json"
    return Path.home() / ".touchless" / "auto_discovered_projects.json"


# --- Project exclusion list ----------------------------------------------
#
# The exclusion list is a simple JSON file with a list of IDs/labels
# the user wants suppressed from auto-discovery + the cortex world.
# Matched case-insensitively against BOTH the slug id and the display
# label (substring match), so the user can type either a folder name
# or the slug. Hardcoded KNOWN_PROJECT_ROOTS entries are NEVER
# suppressed via this list (the consumer logic in
# known_projects.combined_project_roots applies exclusions only to the
# auto-discovered branch).
#
# Pre-seeded on first read with a small set of known-noise projects
# the user has asked to keep out by default.

# Default entries pre-seeded into a freshly-created exclusion file.
# Each entry is matched case-insensitively against BOTH the slug id
# and the display label as a substring — so "vcpkg" hides any folder
# whose label or id contains "vcpkg". The four entries below match
# the user's known-noise list.
_DEFAULT_EXCLUSIONS: List[str] = [
    "hgr-app-v1-0-0-jarvis-test",
    "HGR App v1.0.0 - jarvis test",
    "jarvis-assistant",
    "jarvis_assistant",
    "hgrui",
    "HGRui",
    "vcpkg",
    "sources",
    "tsc",
]


def _exclusions_path() -> Path:
    """On-disk location of the project exclusion list. Mirrors
    ``_cache_path()`` (LOCALAPPDATA on Windows, ~/.touchless elsewhere).
    Override via ``TOUCHLESS_PROJECT_EXCLUSIONS`` for tests.
    """
    override = os.environ.get("TOUCHLESS_PROJECT_EXCLUSIONS")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Touchless" / "project_exclusions.json"
    return Path.home() / ".touchless" / "project_exclusions.json"


# Lock so concurrent readers/writers don't tear the file. The list is
# small (handful of strings) so we just hold the lock across read/write.
_exclusions_lock = threading.RLock()


def load_project_exclusions() -> List[str]:
    """Return the raw list of excluded ID / label strings.

    Auto-creates the exclusion file on first read, pre-seeded with
    ``_DEFAULT_EXCLUSIONS`` so the four known-noise projects are
    hidden by default. Always returns a list (empty on any error) —
    never raises.
    """
    path = _exclusions_path()
    with _exclusions_lock:
        if not path.exists():
            # First run: write the default-seeded list so the user
            # sees their preferred exclusions immediately, and so the
            # file exists for the iris_remove_project tool to mutate.
            try:
                _save_project_exclusions_unlocked(list(_DEFAULT_EXCLUSIONS))
            except Exception:
                pass
            return list(_DEFAULT_EXCLUSIONS)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(data, dict):
            return []
        ids = data.get("excluded_ids")
        if not isinstance(ids, list):
            return []
        out: List[str] = []
        seen = set()
        for entry in ids:
            if isinstance(entry, str) and entry.strip():
                cleaned = entry.strip()
                if cleaned.lower() not in seen:
                    seen.add(cleaned.lower())
                    out.append(cleaned)
        # Retroactive seed: union with current _DEFAULT_EXCLUSIONS so
        # new defaults (added in later releases) reach existing users
        # without requiring them to delete their file. Idempotent — if
        # an existing user already has every default entry, nothing
        # changes. If any are missing, we add them and persist.
        added_any = False
        for default in _DEFAULT_EXCLUSIONS:
            if default.lower() not in seen:
                seen.add(default.lower())
                out.append(default)
                added_any = True
        if added_any:
            try:
                _save_project_exclusions_unlocked(out)
            except Exception:
                pass
        return out


def _save_project_exclusions_unlocked(exclusions: List[str]) -> bool:
    """Persist the exclusion list to disk. Caller must hold the lock
    (or be in a single-threaded path like first-read seeding).
    Returns ``True`` on success."""
    try:
        path = _exclusions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "excluded_ids": [str(e) for e in exclusions if str(e).strip()],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def save_project_exclusions(exclusions: List[str]) -> bool:
    """Persist the exclusion list to disk (thread-safe)."""
    with _exclusions_lock:
        return _save_project_exclusions_unlocked(exclusions)


def add_project_exclusion(identifier: str) -> bool:
    """Append ``identifier`` to the exclusion list (idempotent).

    Returns ``True`` if the identifier was added (or already present)
    and the file was saved successfully, ``False`` on any I/O failure.
    """
    ident = (identifier or "").strip()
    if not ident:
        return False
    with _exclusions_lock:
        current = load_project_exclusions()
        # Case-insensitive idempotency: don't add "VCPKG" if "vcpkg"
        # is already there.
        existing_lower = {e.lower() for e in current}
        if ident.lower() not in existing_lower:
            current.append(ident)
            return _save_project_exclusions_unlocked(current)
        return True


def is_project_excluded(
    project_id: Optional[str],
    label: Optional[str] = None,
    exclusions: Optional[List[str]] = None,
) -> bool:
    """True if ``project_id`` or ``label`` matches any entry in the
    exclusion list (case-insensitive substring match in either
    direction).

    If ``exclusions`` is None, loads the on-disk list. Pass an
    already-loaded list when filtering many candidates in a tight
    loop to avoid repeated disk reads.
    """
    if exclusions is None:
        try:
            exclusions = load_project_exclusions()
        except Exception:
            exclusions = []
    if not exclusions:
        return False
    pid = (project_id or "").strip().lower()
    lab = (label or "").strip().lower()
    for excl in exclusions:
        e = (excl or "").strip().lower()
        if not e:
            continue
        # Bi-directional substring: "vcpkg" matches both "vcpkg" and
        # "vcpkg-tools"; "hgrui" matches "HGRui" and vice-versa.
        if pid and (e in pid or pid in e):
            return True
        if lab and (e in lab or lab in e):
            return True
    return False


# --- Scanning -------------------------------------------------------------

def _is_blocked_name(name: str) -> bool:
    """Skip system dirs, build artifacts, dot-folders."""
    if not name:
        return True
    if name.startswith("."):
        return True
    return name.lower() in _NAME_BLOCKLIST


def _has_marker(folder: Path) -> bool:
    """True if at least one project-marker file/dir exists in ``folder``."""
    for m in _PROJECT_MARKERS:
        try:
            if (folder / m).exists():
                return True
        except OSError:
            continue
    return False


def _folder_is_empty_or_hidden_only(folder: Path) -> bool:
    """True if folder is empty OR contains only hidden entries.

    Cheap pre-filter so we don't bother registering a project root that
    has nothing useful in it.
    """
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if not entry.name.startswith("."):
                    return False
        return True
    except OSError:
        # Permission denied / vanished — treat as empty so we skip it.
        return True


def _slug_id(label: str) -> str:
    """Stable JS/SQL-safe id from a folder name."""
    out = []
    for ch in (label or "").lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in "-_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_-") or "x"
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _scan_roots() -> List[Path]:
    """The handful of common parent directories we sweep for projects."""
    home = Path.home()
    return [
        Path("c:/"),
        home / "Documents",
        home / "projects",
        home / "code",
        home / "source",
    ]


def _resolve_lower(path: Path) -> Optional[str]:
    """Resolve a path to its absolute lowercase form. ``None`` on failure
    (broken symlink, permission denied)."""
    try:
        return str(path.resolve()).lower()
    except (OSError, RuntimeError):
        return None


def discover_candidate_projects(
    exclude_roots: Optional[Set[Path]] = None,
    exclude_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Sweep common parent directories for folders that look like projects.

    Returns a list of ``{id, label, root}`` dicts matching the shape of
    ``KNOWN_PROJECT_ROOTS`` entries. Anything in ``exclude_roots`` (by
    resolved path) is filtered out so we don't double-register projects
    already in the hardcoded list.

    ``exclude_ids`` is the user-managed exclusion list (slug ids OR
    display labels — matched case-insensitively as substrings in
    either direction). When ``None``, the persisted exclusion list at
    ``%LOCALAPPDATA%\\Touchless\\project_exclusions.json`` is loaded
    automatically.

    Hard caps:
      * Depth: never deeper than ``_MAX_DEPTH`` from each scan root.
      * Total folders examined: ``_MAX_CANDIDATES_EXAMINED``.
      * System / dotfile / build folders are skipped by name.

    Never raises — wrap in try/except is unnecessary at the call site.
    """
    if exclude_roots is None:
        exclude_roots = set()
    if exclude_ids is None:
        try:
            exclude_ids = load_project_exclusions()
        except Exception:
            exclude_ids = []

    # Normalize exclusion set to resolved lowercase strings for compare.
    excluded_norm: Set[str] = set()
    for r in exclude_roots:
        try:
            excluded_norm.add(str(Path(r).resolve()).lower())
        except Exception:
            continue

    discovered: Dict[str, Dict[str, Any]] = {}  # keyed by resolved lower path
    examined = 0

    def _walk(folder: Path, depth: int) -> None:
        nonlocal examined
        if examined >= _MAX_CANDIDATES_EXAMINED:
            return
        if depth > _MAX_DEPTH:
            return

        # Permission-safe listing.
        try:
            entries = list(os.scandir(folder))
        except (OSError, PermissionError):
            return

        for entry in entries:
            if examined >= _MAX_CANDIDATES_EXAMINED:
                return
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue

            name = entry.name
            if _is_blocked_name(name):
                continue

            child = Path(entry.path)
            examined += 1

            # Skip symlinks/junctions to avoid loops + duplicate visits.
            try:
                if entry.is_symlink():
                    continue
            except OSError:
                continue
            # Path.is_junction() is Python 3.12+; guard for older runtimes.
            is_junction = getattr(child, "is_junction", None)
            try:
                if callable(is_junction) and is_junction():
                    continue
            except OSError:
                continue

            resolved = _resolve_lower(child)
            if not resolved:
                continue
            if resolved in excluded_norm:
                continue
            if resolved in discovered:
                continue

            try:
                if _has_marker(child):
                    if _folder_is_empty_or_hidden_only(child):
                        continue
                    slug = _slug_id(name)
                    # User-driven exclusions: matched against either
                    # slug id OR display label, case-insensitive
                    # substring in either direction. Suppresses noise
                    # the user has told us they don't want surfaced.
                    if exclude_ids and is_project_excluded(
                        slug, name, exclusions=exclude_ids
                    ):
                        continue
                    discovered[resolved] = {
                        "id": slug,
                        "label": name,
                        "root": child,
                    }
                    # Don't descend into a recognized project.
                    continue
            except Exception:
                continue

            # No marker — keep descending if depth allows.
            if depth < _MAX_DEPTH:
                _walk(child, depth + 1)

    for scan_root in _scan_roots():
        if examined >= _MAX_CANDIDATES_EXAMINED:
            break
        try:
            if not scan_root.exists():
                continue
            # Depth 1 means: look at immediate children of scan_root.
            _walk(scan_root, depth=1)
        except Exception:
            # Any unexpected scan-root failure: silently move on.
            continue

    # Deterministic ordering so cache writes don't churn on every scan.
    return sorted(discovered.values(), key=lambda p: str(p["root"]).lower())


# --- Cache I/O ------------------------------------------------------------

def load_cached_auto_discovered() -> List[Dict[str, Any]]:
    """Read previously-cached auto-discovered projects. Returns ``[]``
    if the cache is missing, malformed, or any entry's root no longer
    exists on disk.

    Honors the user's exclusion list — entries whose id/label matches
    an exclusion are filtered out before returning. This way, even a
    stale cache file (written before an exclusion was added) doesn't
    leak hidden projects into ``known_projects.combined_project_roots``.
    """
    path = _cache_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    projects = data.get("projects")
    if not isinstance(projects, list):
        return []
    # Load the exclusion list once so we don't hit disk per-entry.
    try:
        exclusions = load_project_exclusions()
    except Exception:
        exclusions = []
    out: List[Dict[str, Any]] = []
    for p in projects:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        root = p.get("root")
        if not pid or not root:
            continue
        try:
            root_path = Path(root)
            if not root_path.exists():
                continue
        except Exception:
            continue
        label = p.get("label") or pid
        # User-driven exclusion filter (defense in depth: also applied
        # at discover_candidate_projects(); applied again here so an
        # exclusion added AFTER the last scan still hides the project
        # without forcing a fresh disk walk).
        if exclusions and is_project_excluded(pid, label, exclusions=exclusions):
            continue
        out.append({
            "id": pid,
            "label": label,
            "root": root_path,
        })
    return out


def save_cached_auto_discovered(projects: List[Dict[str, Any]]) -> bool:
    """Persist the auto-discovered projects to the cache file. Returns
    ``True`` on success, ``False`` on any I/O failure (silently swallowed)."""
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "last_scanned_at": time.time(),
            "projects": [
                {
                    "id": p["id"],
                    "label": p.get("label") or p["id"],
                    "root": str(Path(p["root"]).resolve()).replace("\\", "/"),
                }
                for p in projects
                if p.get("id") and p.get("root")
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def discover_and_cache(
    exclude_roots: Optional[Set[Path]] = None,
    exclude_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run a fresh scan and persist the result to the cache file.

    Convenience wrapper around ``discover_candidate_projects`` +
    ``save_cached_auto_discovered`` — every failure mode (scan crash,
    cache write failure) is swallowed and returns the best-effort
    in-memory list.

    ``exclude_ids`` is forwarded to ``discover_candidate_projects`` —
    when ``None``, the persisted user exclusion list is loaded and
    applied automatically.
    """
    try:
        found = discover_candidate_projects(
            exclude_roots=exclude_roots,
            exclude_ids=exclude_ids,
        )
    except Exception:
        found = []
    if found:
        save_cached_auto_discovered(found)
    return found


def purge_from_cache(project_id: str) -> bool:
    """Remove ``project_id`` from the auto-discovered cache file
    (idempotent — no-op if the cache doesn't exist or the id isn't
    present). Returns ``True`` if the cache was rewritten (or didn't
    need to be), ``False`` only on I/O failure.

    Used by ``iris_remove_project`` so a freshly-excluded project is
    physically removed from the cache without waiting for the next
    discovery sweep.
    """
    try:
        cached = load_cached_auto_discovered()
    except Exception:
        return False
    pid_lower = (project_id or "").strip().lower()
    if not pid_lower:
        return False
    filtered = [p for p in cached if str(p.get("id", "")).lower() != pid_lower]
    if len(filtered) == len(cached):
        # Nothing to remove — still a success.
        return True
    try:
        return save_cached_auto_discovered(filtered)
    except Exception:
        return False


# Author: Konstantin Markov

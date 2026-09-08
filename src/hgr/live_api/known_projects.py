"""Shared registry of known sibling project roots.

Two consumers today:

  * ``run_iris_simulator.py`` — supplements the cortex_world.json with
    filesystem listings for these roots so the simulator shows real
    folder structure even before Iris has touched anything.
  * ``LiveApiManager`` (start path) — registers each existing root into
    the ProjectMemoryStore and spawns a background indexer thread so
    ``MemoryManager.recall`` can surface project chunks during planner /
    realtime turns.

Single source of truth: edit the list here, both callers pick it up.

Author: Konstantin Markov
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


KNOWN_PROJECT_ROOTS: List[Dict[str, Any]] = [
    {"id": "touchless-dev",       "label": "Touchless dev",       "root": Path("c:/HGR App v1.0.0")},
    {"id": "touchless-website",   "label": "Touchless website",   "root": Path("c:/touchless-website")},
    {"id": "touchless-marketing", "label": "Touchless socials",   "root": Path("c:/touchless-marketing")},
    {"id": "touchless-tracking",  "label": "Touchless tracking",  "root": Path("c:/touchless-tracking")},
]


def existing_known_roots() -> List[Dict[str, Any]]:
    """Return only the project entries whose ``root`` exists on disk.

    Callers that scan or index the filesystem can iterate the result
    directly without re-checking ``Path.exists()`` per entry.
    """
    return [p for p in KNOWN_PROJECT_ROOTS if p["root"].exists()]


def combined_project_roots(include_auto: bool = True) -> List[Dict[str, Any]]:
    """Return hardcoded ``KNOWN_PROJECT_ROOTS`` merged with cached
    auto-discovered projects (from
    ``project_autodetect.load_cached_auto_discovered``).

    Hardcoded entries win on id-collision. Auto-discovered entries
    carry an ``auto`` boolean flag so consumers can tell them apart
    (e.g. for UI labeling or different indexer policies).

    Best-effort: any failure to import / load the auto-discovery cache
    returns just the hardcoded list — never raises.
    """
    out: List[Dict[str, Any]] = []
    seen_ids: set = set()
    seen_roots: set = set()

    for entry in KNOWN_PROJECT_ROOTS:
        out.append({
            "id": entry["id"],
            "label": entry.get("label") or entry["id"],
            "root": entry["root"],
            "auto": False,
        })
        seen_ids.add(entry["id"])
        try:
            seen_roots.add(str(entry["root"].resolve()).lower())
        except Exception:
            pass

    if not include_auto:
        return out

    try:
        from .project_autodetect import (
            load_cached_auto_discovered,
            load_project_exclusions,
            is_project_excluded,
        )
        # Load the exclusion list ONCE for this call so we don't hit
        # the disk per-entry. load_cached_auto_discovered() already
        # applies the same filter on its end, but we re-apply here as
        # a defense-in-depth layer (a stale cache containing an
        # excluded project must NEVER leak through).
        try:
            exclusions = load_project_exclusions()
        except Exception:
            exclusions = []
        for entry in load_cached_auto_discovered():
            pid = entry.get("id")
            root = entry.get("root")
            if not pid or not root:
                continue
            label = entry.get("label") or pid
            # User-driven exclusion. Only auto-discovered entries are
            # filterable — the hardcoded loop above is already done,
            # so KNOWN_PROJECT_ROOTS can never be hidden via this
            # path. That's the intended contract.
            if exclusions and is_project_excluded(
                pid, label, exclusions=exclusions
            ):
                continue
            try:
                root_norm = str(root.resolve()).lower()
            except Exception:
                root_norm = None
            if pid in seen_ids:
                continue
            if root_norm and root_norm in seen_roots:
                continue
            out.append({
                "id": pid,
                "label": label,
                "root": root,
                "auto": True,
            })
            seen_ids.add(pid)
            if root_norm:
                seen_roots.add(root_norm)
    except Exception:
        # Auto-discovery is best-effort — never let it break the
        # hardcoded-list consumers.
        pass

    return out

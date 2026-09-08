"""Pattern surfacing — cluster recent episodic memory to find repeated
user habits and tool-sequence patterns.

Two analyses:

  find_repeated_patterns(store, embedder, days=7) — greedy single-pass
    agglomerative clustering over episode user_text embeddings (cosine
    sim). Surfaces "this user keeps asking variations of the same
    thing." Cold-start safe (returns empty + cold_start=True when fewer
    than min_cluster_size episodes exist in the window).

  find_tool_sequences(store, days=7) — pairwise (toolA -> toolB)
    transition counter over steps_json across recent episodes. Surfaces
    "these two tools frequently fire together" without requiring an
    embedder.

Both functions are read-only against ``MemoryStore`` and silent on
errors — return empty results when the DB is missing, the embedder
isn't loaded, or any row is malformed.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

from .embedder import Embedder, cosine_sim
from .store import MemoryStore


def find_repeated_patterns(
    store: MemoryStore,
    embedder: Embedder,
    days: int = 7,
    sim_threshold: float = 0.85,
    min_cluster_size: int = 2,
    max_scan: int = 10_000,
) -> Dict[str, Any]:
    """Cluster recent episodes by query-embedding cosine similarity.

    Returns:
        {
            "patterns": [
                {
                    "label": str (first 60 chars of most recent query),
                    "kind": "query",
                    "count": int,
                    "example_user_text": str,
                    "last_seen_at": float (ts),
                    "examples": [str, ...] (up to 5 user_text strings),
                }, ...
            ],
            "cold_start": bool (True when too few episodes to cluster),
            "scanned_episodes": int,
        }
    """
    empty: Dict[str, Any] = {
        "patterns": [],
        "cold_start": True,
        "scanned_episodes": 0,
    }
    try:
        cutoff_ts = time.time() - (days * 86400)
        all_rows = store.list_episodic(limit=max_scan)
    except Exception:
        return empty
    recent = [r for r in all_rows if r.ts >= cutoff_ts]

    if len(recent) < max(min_cluster_size, 10):
        # Cold-start: too few episodes to make any cluster decision
        # meaningful. Caller (simulator) should hide the Patterns node.
        return {
            "patterns": [],
            "cold_start": True,
            "scanned_episodes": len(recent),
        }

    # Embed all recent user_text queries. Tolerate per-row failures.
    embeddings: List[List[float]] = []
    for r in recent:
        try:
            vec = embedder.embed(r.user_text or "")
            embeddings.append(list(vec) if vec else [])
        except Exception:
            embeddings.append([])

    # Greedy single-pass agglomerative: for each embedding, find the
    # existing cluster (by its representative — the earliest-added
    # member) with the highest cosine sim above the threshold. Merge if
    # found, otherwise start a new cluster. O(n * k) where k is the
    # number of clusters — fine for the 10k episode cap.
    clusters: List[List[int]] = []
    for i, vec_i in enumerate(embeddings):
        if not vec_i:
            continue
        best_cluster_idx = None
        best_sim = sim_threshold
        for cidx, cluster in enumerate(clusters):
            rep_idx = cluster[0]
            vec_rep = embeddings[rep_idx]
            if not vec_rep:
                continue
            sim = cosine_sim(vec_i, vec_rep)
            if sim > best_sim:
                best_sim = sim
                best_cluster_idx = cidx
        if best_cluster_idx is not None:
            clusters[best_cluster_idx].append(i)
        else:
            clusters.append([i])

    # Convert clusters to pattern dicts. Sort within each cluster by ts
    # desc so the "most recent" representative drives the label.
    patterns: List[Dict[str, Any]] = []
    for cluster_indices in clusters:
        if len(cluster_indices) < min_cluster_size:
            continue
        cluster_indices.sort(key=lambda idx: recent[idx].ts, reverse=True)
        most_recent_row = recent[cluster_indices[0]]
        label = (most_recent_row.user_text or "").strip()[:60]
        if not label:
            label = "(empty query)"
        patterns.append({
            "label": label,
            "kind": "query",
            "count": len(cluster_indices),
            "example_user_text": most_recent_row.user_text or "",
            "last_seen_at": float(most_recent_row.ts),
            "examples": [
                (recent[i].user_text or "")[:120]
                for i in cluster_indices[:5]
            ],
        })

    # Most-frequent + most-recent first.
    patterns.sort(key=lambda p: (-int(p["count"]), -float(p["last_seen_at"])))

    return {
        "patterns": patterns,
        "cold_start": False,
        "scanned_episodes": len(recent),
    }


def find_tool_sequences(
    store: MemoryStore,
    days: int = 7,
    min_frequency: int = 3,
    max_scan: int = 10_000,
) -> Dict[str, Any]:
    """Find repeated (toolA -> toolB) sequences across recent episodes.

    Returns:
        {
            "sequences": [
                {
                    "from_tool": str,
                    "to_tool": str,
                    "count": int,
                    "kind": "tool_sequence",
                    "last_seen_at": float (ts),
                }, ...
            ],
            "scanned_episodes": int,
        }
    """
    empty: Dict[str, Any] = {"sequences": [], "scanned_episodes": 0}
    try:
        cutoff_ts = time.time() - (days * 86400)
        all_rows = store.list_episodic(limit=max_scan)
    except Exception:
        return empty
    recent = [r for r in all_rows if r.ts >= cutoff_ts]

    counts: Counter = Counter()
    last_seen: Dict[Tuple[str, str], float] = {}

    for row in recent:
        if not row.steps_json:
            continue
        try:
            steps = json.loads(row.steps_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(steps, list) or len(steps) < 2:
            continue
        tools = [
            s.get("tool") for s in steps
            if isinstance(s, dict) and s.get("tool")
        ]
        for i in range(len(tools) - 1):
            from_tool = tools[i]
            to_tool = tools[i + 1]
            if not (from_tool and to_tool):
                continue
            key = (from_tool, to_tool)
            counts[key] += 1
            # Episodes come back DESC by ts — only set last_seen the
            # first time we encounter the pair (= most recent occurrence).
            if key not in last_seen:
                last_seen[key] = float(row.ts)

    result: List[Dict[str, Any]] = []
    for (from_tool, to_tool), count in counts.items():
        if count >= min_frequency:
            result.append({
                "from_tool": from_tool,
                "to_tool": to_tool,
                "count": int(count),
                "kind": "tool_sequence",
                "last_seen_at": last_seen[(from_tool, to_tool)],
            })
    result.sort(key=lambda x: (-int(x["count"]), -float(x["last_seen_at"])))

    return {
        "sequences": result,
        "scanned_episodes": len(recent),
    }

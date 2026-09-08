"""Self-improving prompt variant framework.

Phase-5 cognition. Today the planner / synthesizer / reviser each
use a single hard-coded system prompt. There's no way to know
whether a prompt change ACTUALLY helps until a human reads the
diff and guesses.

This module gives Iris a small A/B framework for prompt variants:

  * Register N variants of a prompt under a name (e.g., "planner_v1",
    "planner_v2_more_compact").
  * `pick(name)` returns one variant per call, biased by current
    success rate but always allocating at least `EXPLORE_FLOOR_PCT`
    to non-leaders so we don't get stuck on a local maximum.
  * `record_outcome(name, variant_id, success)` updates the
    tracker. Persisted to SQLite so it survives restarts.
  * `leaderboard(name)` shows which variant is winning + the
    sample sizes so a human can decide when to retire losers.

The "success" signal is intentionally caller-defined: the planner
treats a turn as successful when all steps returned status=ok,
the synthesizer when its output was non-empty + the orchestrator
returned a plausible final message, etc. Modules call
`record_outcome` from their post-result paths.

This is bandit-style learning — Thompson sampling would be ideal
but cost too much complexity. A simple epsilon-greedy with a 20%
explore floor + Bayesian smoothing covers the practical use case.

Honors incognito (no writes from private mode).

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Bayesian smoothing prior: assume each variant has a baseline
# success rate of ~70% before any data. Prevents a single early
# failure from killing a variant before it's been sampled enough.
PRIOR_ALPHA = 7.0   # successes
PRIOR_BETA = 3.0    # failures
# Minimum % of traffic that goes to non-leader variants so we
# don't get stuck on a local maximum.
EXPLORE_FLOOR_PCT = 20
# Below this many total samples, treat a variant's score as
# "unknown" and route it uniformly (warm-up phase).
WARMUP_SAMPLE_THRESHOLD = 10


def _default_db() -> Path:
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(local) / "Touchless" / "private" / "prompt_variants.db"


@dataclass
class Variant:
    """One prompt variant under a named group."""
    id: str              # short stable id ("v1", "compact", etc.)
    text: str            # the actual prompt text
    notes: str = ""      # human-readable description
    # Counters; updated from the store on each pick + record.
    successes: int = 0
    failures: int = 0

    @property
    def total(self) -> int:
        return self.successes + self.failures

    def smoothed_success_rate(self) -> float:
        a = self.successes + PRIOR_ALPHA
        b = self.failures + PRIOR_BETA
        return a / (a + b)


@dataclass
class _PickState:
    """Per-group rolling counter so we can do deterministic
    explore/exploit allocation without random()."""
    counter: int = 0


class PromptVariantTracker:
    """Per-group bandit over registered variants. Thread-safe."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS prompt_variants (
        group_name  TEXT NOT NULL,
        variant_id  TEXT NOT NULL,
        successes   INTEGER NOT NULL DEFAULT 0,
        failures    INTEGER NOT NULL DEFAULT 0,
        last_used_at REAL DEFAULT 0,
        PRIMARY KEY (group_name, variant_id)
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
        # In-memory variant registry: group_name -> [Variant].
        self._registry: Dict[str, List[Variant]] = {}
        # Per-group pick counters for deterministic allocation.
        self._pick_state: Dict[str, _PickState] = {}

    # ---- registration -------------------------------------------------

    def register(self, group_name: str,
                 variants: List[Variant]) -> None:
        """Register the set of variants for a group. Replaces any
        prior registration for that group."""
        if not group_name or not variants:
            return
        with self._lock:
            # Load any persisted counters so leaderboard + pick see
            # the cumulative history.
            loaded: List[Variant] = []
            for v in variants:
                successes, failures = self._read_counters(
                    group_name, v.id)
                loaded.append(Variant(
                    id=v.id, text=v.text, notes=v.notes,
                    successes=successes, failures=failures,
                ))
            self._registry[group_name] = loaded
            self._pick_state.setdefault(
                group_name, _PickState())

    # ---- pick ---------------------------------------------------------

    def pick(self, group_name: str) -> Optional[Variant]:
        """Return one variant. None when the group isn't registered."""
        with self._lock:
            variants = self._registry.get(group_name)
            if not variants:
                return None
            state = self._pick_state.setdefault(
                group_name, _PickState())
            state.counter += 1
            # Warm-up: if any variant has fewer than the threshold
            # samples, route there to gather data.
            warming = [v for v in variants
                       if v.total < WARMUP_SAMPLE_THRESHOLD]
            if warming:
                idx = state.counter % len(warming)
                return warming[idx]
            # Past warm-up: epsilon-greedy with an EXPLORE_FLOOR_PCT
            # explore slot. Every Nth call (N=100/explore_pct), route
            # to a non-leader; otherwise route to the leader.
            n_explore = max(1, 100 // EXPLORE_FLOOR_PCT)
            if state.counter % n_explore == 0:
                # Explore: round-robin across non-leaders.
                sorted_vs = sorted(
                    variants,
                    key=lambda v: -v.smoothed_success_rate())
                non_leaders = sorted_vs[1:] or sorted_vs
                idx = (state.counter // n_explore) % len(non_leaders)
                return non_leaders[idx]
            # Exploit: pick the leader by smoothed score.
            return max(variants,
                       key=lambda v: v.smoothed_success_rate())

    # ---- outcome ------------------------------------------------------

    def record_outcome(self, group_name: str, variant_id: str,
                       *, success: bool) -> None:
        """Record one sample. Honors incognito (no writes in
        private mode)."""
        try:
            from .incognito import is_incognito
            if is_incognito():
                return
        except Exception:
            pass
        with self._lock:
            variants = self._registry.get(group_name) or []
            for v in variants:
                if v.id == variant_id:
                    if success:
                        v.successes += 1
                    else:
                        v.failures += 1
                    break
            try:
                self._conn.execute(
                    "INSERT INTO prompt_variants "
                    "(group_name, variant_id, successes, failures, "
                    " last_used_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(group_name, variant_id) DO UPDATE "
                    "SET successes = successes + ?, "
                    "    failures = failures + ?, "
                    "    last_used_at = ?",
                    (group_name, variant_id,
                     1 if success else 0,
                     0 if success else 1,
                     time.time(),
                     1 if success else 0,
                     0 if success else 1,
                     time.time()),
                )
            except Exception:
                pass

    # ---- introspection ------------------------------------------------

    def leaderboard(self, group_name: str
                    ) -> List[Dict[str, object]]:
        """Per-variant snapshot: id, score, samples, notes.
        Ordered by smoothed score descending."""
        with self._lock:
            variants = self._registry.get(group_name) or []
            rows = sorted(
                variants,
                key=lambda v: -v.smoothed_success_rate())
        return [
            {"id": v.id, "notes": v.notes,
             "successes": v.successes, "failures": v.failures,
             "total_samples": v.total,
             "smoothed_score": v.smoothed_success_rate()}
            for v in rows
        ]

    def wipe(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM prompt_variants")
            n = int(cur.fetchone()[0])
            self._conn.execute("DELETE FROM prompt_variants")
            # Reset in-memory counters too.
            for variants in self._registry.values():
                for v in variants:
                    v.successes = 0
                    v.failures = 0
            return n

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ---- internals ----------------------------------------------------

    def _read_counters(self, group_name: str,
                       variant_id: str) -> tuple:
        try:
            cur = self._conn.execute(
                "SELECT successes, failures FROM prompt_variants "
                "WHERE group_name=? AND variant_id=?",
                (group_name, variant_id))
            row = cur.fetchone()
        except Exception:
            return (0, 0)
        if row is None:
            return (0, 0)
        return (int(row[0] or 0), int(row[1] or 0))


# ---- module singleton --------------------------------------------------

_tracker: Optional[PromptVariantTracker] = None
_lock = threading.Lock()


def global_tracker() -> PromptVariantTracker:
    global _tracker
    if _tracker is None:
        with _lock:
            if _tracker is None:
                _tracker = PromptVariantTracker()
    return _tracker


def _reset_for_tests() -> None:
    global _tracker
    with _lock:
        if _tracker is not None:
            try:
                _tracker.close()
            except Exception:
                pass
        _tracker = None

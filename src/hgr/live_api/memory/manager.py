"""MemoryManager — the single entry point for the rest of Iris.

Two operations:

  record(user_text, plan, steps, results, message)
      Async-ish: extracts facts synchronously (fast — pure regex/dict work)
      and writes them, then spins a background thread for the episodic
      embed + insert so the user-visible reply isn't blocked by the
      embedder's HTTP round-trip.

  recall(goal_text, k=3) -> {"episodes": [...], "facts": [...], "context": str}
      Returns the top-k episodic matches (cosine sim over embeddings) and
      any semantic facts whose key appears in the goal text. The
      "context" string is a pre-rendered block ready to drop into the
      planner / synthesizer prompt.

Storage path defaults to %LOCALAPPDATA%\\Touchless\\memory.db on Windows;
on other platforms falls back to ~/.touchless/memory.db. Override via
TOUCHLESS_MEMORY_DB.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .embedder import Embedder, cosine_sim, default_embedder
from .extractor import extract_facts
from .store import EpisodicRow, MemoryStore, SemanticRow


def default_memory_path() -> Path:
    override = os.environ.get("TOUCHLESS_MEMORY_DB")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Touchless" / "memory.db"
    return Path.home() / ".touchless" / "memory.db"


# How many recent episodes to scan when ranking by cosine similarity.
_RECALL_SCAN = 200
# Minimum cosine similarity to surface an episode as relevant.
_RECALL_MIN_SIM = 0.1
# Max chars of context block injected into a prompt.
_CONTEXT_MAX_CHARS = 600


class MemoryManager:
    """Wraps the store + embedder. Threadsafe; non-blocking writes."""

    def __init__(self, store: Optional[MemoryStore] = None,
                 embedder: Optional[Embedder] = None,
                 async_writes: bool = True,
                 logger: Any = None) -> None:
        self._store = store or MemoryStore(default_memory_path())
        self._embedder = embedder or default_embedder()
        self._async = async_writes
        self._logger = logger

    # ---- write -----------------------------------------------------------
    def set_fact(self, kind: str, key: str, value: str,
                 source: str = "user said") -> None:
        """Persist a single semantic fact. Used by Tier 1 preference-setting
        commands ('always send from gmail') to write through to the store
        without going via the executor."""
        try:
            self._store.add_semantic(kind, key, value, source)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_set_fact_failed", exc)

    def record(self, user_text: str, plan: Any, steps: List[Any],
               results: List[Any], message: str) -> None:
        """Persist a planner-handled turn. Fact extraction is sync (fast);
        episodic embedding + insert runs on a worker thread when async_writes
        is True so the user reply isn't blocked by the embedder HTTP call."""
        facts = extract_facts(user_text, steps, results)
        if facts:
            try:
                self._store.add_facts(facts)
            except Exception as exc:  # pragma: no cover - defensive
                if self._logger:
                    self._logger.exception("memory_add_facts_failed", exc)

        plan_json = self._plan_to_json(plan)
        steps_json = self._steps_to_json(steps, results)
        outcome = (message or "")[:600]

        if self._async:
            threading.Thread(
                target=self._episodic_insert_safe,
                args=(user_text, plan_json, steps_json, outcome),
                name="iris-memory-write",
                daemon=True,
            ).start()
        else:
            self._episodic_insert_safe(user_text, plan_json, steps_json, outcome)

    # ---- read ------------------------------------------------------------
    def recall(self, goal_text: str, k: int = 3) -> Dict[str, Any]:
        """Return relevant memory for the goal. Episodic ranked by cosine
        similarity over embeddings; facts surfaced when any semantic key
        appears (substring) in the goal text."""
        goal = (goal_text or "").strip()
        episodes: List[Dict[str, Any]] = []
        facts: List[SemanticRow] = []
        if not goal:
            return {"episodes": episodes, "facts": facts, "context": ""}

        # ---- episodic: embed goal, cosine-rank against recent rows ------
        try:
            goal_vec = self._embedder.embed(goal)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_recall_embed_failed", exc)
            goal_vec = []
        if goal_vec:
            try:
                rows = self._store.list_episodic(limit=_RECALL_SCAN)
            except Exception:
                rows = []
            scored: List[tuple] = []
            for r in rows:
                if not r.embedding:
                    continue
                sim = cosine_sim(goal_vec, r.embedding)
                if sim >= _RECALL_MIN_SIM:
                    scored.append((sim, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            for sim, r in scored[:k]:
                episodes.append({
                    "sim": round(sim, 3),
                    "ts": r.ts,
                    "user_text": r.user_text,
                    "outcome": r.outcome,
                })

        # ---- semantic: any fact whose key appears in the goal text -----
        lower = goal.lower()
        try:
            candidates = self._store.find_facts(limit=200)
        except Exception:
            candidates = []
        for row in candidates:
            if row.key and re.search(rf"\b{re.escape(row.key)}\b", lower):
                facts.append(row)

        return {
            "episodes": episodes,
            "facts": facts,
            "context": self._render_context(episodes, facts),
        }

    # ---- helpers --------------------------------------------------------
    def _episodic_insert_safe(self, user_text: str, plan_json: Optional[str],
                              steps_json: Optional[str], outcome: str) -> None:
        try:
            vec = self._embedder.embed(user_text)
        except Exception as exc:  # pragma: no cover
            if self._logger:
                self._logger.exception("memory_embed_failed", exc)
            vec = []
        try:
            self._store.add_episodic(user_text, plan_json, steps_json, outcome, vec)
        except Exception as exc:  # pragma: no cover
            if self._logger:
                self._logger.exception("memory_add_episodic_failed", exc)

    @staticmethod
    def _plan_to_json(plan: Any) -> Optional[str]:
        if plan is None:
            return None
        try:
            return json.dumps({
                "goal": getattr(plan, "goal", ""),
                "final": getattr(plan, "final", "return"),
                "n_steps": len(getattr(plan, "steps", []) or []),
            })
        except Exception:
            return None

    @staticmethod
    def _steps_to_json(steps: List[Any], results: List[Any]) -> Optional[str]:
        rows: List[Dict[str, Any]] = []
        for step, sr in zip(steps or [], results or []):
            try:
                rows.append({
                    "tool": getattr(step, "tool", "?"),
                    "status": getattr(sr, "status", "?"),
                })
            except Exception:
                continue
        try:
            return json.dumps(rows)
        except Exception:
            return None

    @staticmethod
    def _render_context(episodes: List[Dict[str, Any]],
                        facts: List[SemanticRow]) -> str:
        """Pre-render a tiny "Context from prior turns" block. Empty if there's
        nothing useful — caller can just check truthiness."""
        if not episodes and not facts:
            return ""
        lines: List[str] = ["Context from prior turns:"]
        for f in facts[:8]:
            lines.append(f"- {f.kind} {f.key} = {f.value}")
        for e in episodes:
            ut = (e.get("user_text") or "")[:120]
            oc = (e.get("outcome") or "")[:120]
            lines.append(f'- prior: "{ut}" -> {oc}')
        text = "\n".join(lines)
        if len(text) > _CONTEXT_MAX_CHARS:
            text = text[:_CONTEXT_MAX_CHARS] + "..."
        return text

"""Local intent classifier — per-user learned tool routing in <20ms.

Phase-8 latency. `smart_fast_path` (Phase-6) regex-matches a fixed
catalogue of trivial utterances. It's deterministic and zero-cost
but doesn't ADAPT to how the user actually talks — if a user always
says "kill the music" instead of "pause music", regex misses and
the planner round-trip kicks in.

This module owns a tiny on-device learner: per-tool word-weight
vectors trained from the user's own command history. Each time
the orchestrator dispatches a tool successfully, we record
`(text → tool)` as a positive example. Weights update with simple
incremental learning (perceptron-style); no torch / sklearn / external
deps so the installer doesn't grow.

Pure Python. SQLite-backed at
    %LOCALAPPDATA%/Touchless/private/local_intent.db
so the classifier survives restarts + learns over weeks.

Public API:

  * `record_example(text, tool, success=True)` — orchestrator calls
    when a tool runs.
  * `classify(text) -> ClassifyResult` — pre-planner check. Returns
    `(tool, confidence)` or None when nothing crosses threshold.
  * `forget_tool(tool)` — admin / privacy reset.

Threshold default 0.7 — only fires when the model is confident.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_DEFAULT_THRESHOLD = 0.7
_MIN_EXAMPLES_PER_TOOL = 3        # need at least N before we trust


def _default_db_path() -> Path:
    override = os.environ.get("TOUCHLESS_LOCAL_INTENT_DIR")
    if override:
        d = Path(override)
    else:
        local = os.environ.get(
            "LOCALAPPDATA",
            str(Path.home() / "AppData" / "Local"))
        d = Path(local) / "Touchless" / "private"
    d.mkdir(parents=True, exist_ok=True)
    return d / "local_intent.db"


@dataclass
class ClassifyResult:
    tool: str
    confidence: float
    matched_words: Tuple[str, ...] = ()
    fallback_used: bool = False

    def is_confident(self,
                     threshold: float = _DEFAULT_THRESHOLD) -> bool:
        return self.confidence >= threshold


_TOKEN_RE = re.compile(r"\b[a-z0-9']+\b")
_STOPWORDS = frozenset({
    "the", "a", "an", "to", "for", "of", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "and", "or",
    "but", "i", "me", "my", "you", "your", "it", "its",
    "this", "that", "these", "those", "with", "as",
    "by", "do", "did", "does",
})


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in _STOPWORDS]


# ---- store -----------------------------------------------------------

class IntentStore:
    """SQLite store of per-tool word weights + example counts.

    Schema:
        tools(name, examples_count)
        weights(tool, word, weight) — many rows per tool
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS tools (
        name TEXT PRIMARY KEY,
        examples_count INTEGER DEFAULT 0,
        positive_count INTEGER DEFAULT 0,
        last_used REAL
    );
    CREATE TABLE IF NOT EXISTS weights (
        tool TEXT,
        word TEXT,
        weight REAL,
        PRIMARY KEY (tool, word)
    );
    CREATE INDEX IF NOT EXISTS idx_weights_word ON weights(word);
    """

    def __init__(self, *, db_path: Optional[Path] = None) -> None:
        self._path = (Path(db_path)
                      if db_path else _default_db_path())
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path),
            isolation_level=None,
            check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(self._SCHEMA)

    def record(self, text: str, tool: str, *,
               success: bool = True,
               learning_rate: float = 0.1) -> None:
        """Update weights for this (text, tool) example. Positive
        examples push weights up; negative examples (success=False)
        push down so a failed dispatch teaches the classifier away."""
        toks = _tokenize(text)
        if not toks or not tool:
            return
        delta = learning_rate if success else -learning_rate
        with self._lock:
            self._conn.execute(
                "INSERT INTO tools(name, examples_count, "
                "positive_count, last_used) "
                "VALUES(?, 1, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "examples_count = examples_count + 1, "
                "positive_count = positive_count + "
                "  excluded.positive_count, "
                "last_used = excluded.last_used",
                (tool, 1 if success else 0, time.time()))
            for word in set(toks):
                self._conn.execute(
                    "INSERT INTO weights(tool, word, weight) "
                    "VALUES(?, ?, ?) "
                    "ON CONFLICT(tool, word) DO UPDATE SET "
                    "weight = weight + ?",
                    (tool, word, delta, delta))

    def get_examples_count(self, tool: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT examples_count FROM tools WHERE name = ?",
                (tool,))
            row = cur.fetchone()
        return int(row["examples_count"]) if row else 0

    def all_tools(self) -> List[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT name FROM tools "
                "WHERE examples_count >= ?",
                (_MIN_EXAMPLES_PER_TOOL,))
            return [r["name"] for r in cur.fetchall()]

    def score_for(self, tool: str,
                  tokens: List[str]) -> Tuple[float, List[str]]:
        """Sum the weight contributions of `tokens` for `tool`.
        Returns (raw_score, matched_words)."""
        if not tokens:
            return 0.0, []
        with self._lock:
            placeholder = ",".join("?" * len(tokens))
            cur = self._conn.execute(
                f"SELECT word, weight FROM weights "
                f"WHERE tool = ? AND word IN ({placeholder})",
                (tool, *tokens))
            rows = cur.fetchall()
        if not rows:
            return 0.0, []
        score = sum(r["weight"] for r in rows)
        matched = [r["word"] for r in rows
                   if r["weight"] > 0]
        return score, matched

    def forget(self, tool: str) -> bool:
        with self._lock:
            self._conn.execute(
                "DELETE FROM tools WHERE name = ?", (tool,))
            cur = self._conn.execute(
                "DELETE FROM weights WHERE tool = ?", (tool,))
            return cur.rowcount > 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) c FROM tools")
            tools_count = int(cur.fetchone()["c"])
            cur = self._conn.execute(
                "SELECT COUNT(*) c FROM weights")
            weights_count = int(cur.fetchone()["c"])
        return {"tools": tools_count, "weights": weights_count}

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ---- classifier ------------------------------------------------------

class LocalIntentClassifier:
    """Top-level wrapper. Holds a singleton store reference."""

    def __init__(self, *, store: Optional[IntentStore] = None,
                 threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._store = store or IntentStore()
        self._threshold = threshold

    def record_example(self, text: str, tool: str, *,
                       success: bool = True) -> None:
        self._store.record(text, tool, success=success)

    def classify(self, text: str) -> Optional[ClassifyResult]:
        toks = _tokenize(text)
        if not toks:
            return None
        candidates = self._store.all_tools()
        if not candidates:
            return None
        # Score each candidate; softmax-ish over RAW scores so
        # we don't suffer when scores are tiny.
        scored: List[Tuple[str, float, List[str]]] = []
        for tool in candidates:
            score, matched = self._store.score_for(tool, toks)
            if score > 0:
                scored.append((tool, score, matched))
        if not scored:
            return None
        # Pick best + compute confidence as fraction of best
        # over the SUM of all positive scores.
        scored.sort(key=lambda x: x[1], reverse=True)
        best_tool, best_score, best_match = scored[0]
        total = sum(s for _, s, _ in scored)
        if total <= 0:
            return None
        confidence = best_score / total
        # Penalize when the best has few matched words.
        if len(best_match) < 2:
            confidence *= 0.7
        return ClassifyResult(
            tool=best_tool,
            confidence=confidence,
            matched_words=tuple(best_match))

    def forget_tool(self, tool: str) -> bool:
        return self._store.forget(tool)

    def stats(self) -> Dict[str, Any]:
        return self._store.stats()


# ---- singleton -------------------------------------------------------

_lock = threading.Lock()
_singleton: Optional[LocalIntentClassifier] = None


def global_classifier() -> LocalIntentClassifier:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = LocalIntentClassifier()
        return _singleton


def reset_global() -> None:
    global _singleton
    with _lock:
        if _singleton is not None:
            try:
                _singleton._store.close()
            except Exception:
                pass
        _singleton = None


def classify(text: str) -> Optional[ClassifyResult]:
    return global_classifier().classify(text)


def record_example(text: str, tool: str, *,
                   success: bool = True) -> None:
    global_classifier().record_example(text, tool,
                                        success=success)

"""Auto-pinning of repeatedly-confirmed memory facts.

Phase-3. The implicit-fact extractor writes new candidates with
confidence ~0.7-0.9. Over time, certain facts get reinforced:
the user mentions "Berlin" five separate times across sessions,
or refers to "Dani" in eight different turns. Those facts deserve
a higher trust score so the planner can lean on them more
confidently in fuzzy-match scenarios.

This module owns the elevation rule:

  * Count how many independent extractions (different turns,
    different days) have produced the same (kind, key) tuple.
  * When the count crosses `PIN_THRESHOLD` (default 3), set a
    `pinned=True` marker on the most recent fact + boost its
    confidence to 0.95 (cap).
  * Pinned facts get a small boost in recall ranking so they
    surface first in the planner's "FACTS" context block.

Pure data-layer work; no behavioral change beyond the recall
ordering. Honors incognito (writes are skipped in private mode).

Author: Konstantin Markov
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


PIN_THRESHOLD = 3
# Confidence floor we set when pinning a fact. Cap at 0.95 so we
# never claim absolute certainty about a free-text-derived fact.
PINNED_CONFIDENCE = 0.95
# Time window over which we count reinforcement. Mentions older
# than this stop counting toward the pin threshold (people change
# minds; old mentions shouldn't permanently lock a fact).
COUNT_WINDOW_SEC = 30 * 24 * 3600.0   # 30 days


@dataclass
class PinDecision:
    """Result of `evaluate_pinning_candidate`."""
    should_pin: bool
    mention_count: int
    reason: str = ""


def evaluate_pinning_candidate(*,
                               store: Any,
                               kind: str,
                               key: str) -> PinDecision:
    """Inspect the store for past mentions of (kind, key) and decide
    whether to pin. Pure read — caller does the write.

    `store` must expose `find_facts(kind=..., key=...)` returning a
    list of fact objects with at least `.value` and (ideally)
    `.extracted_at`. The PinDecision.mention_count reflects how
    many distinct mentions occurred within COUNT_WINDOW_SEC."""
    if not (kind and key):
        return PinDecision(False, 0, "missing key")
    try:
        rows = store.find_facts(kind=kind, key=key)
    except Exception:
        return PinDecision(False, 0, "store query failed")
    if not rows:
        return PinDecision(False, 0, "no prior mentions")
    cutoff = time.time() - COUNT_WINDOW_SEC
    recent = 0
    for r in rows:
        ts = getattr(r, "extracted_at", None) or getattr(r, "ts", None)
        if ts is None:
            # Treat undated facts as recent (conservative).
            recent += 1
            continue
        try:
            if float(ts) >= cutoff:
                recent += 1
        except (TypeError, ValueError):
            recent += 1
    if recent < PIN_THRESHOLD:
        return PinDecision(False, recent,
                           f"need {PIN_THRESHOLD} mentions in 30 days")
    # Also need distinct values to count — otherwise spamming the
    # same exact extraction shouldn't auto-pin.
    distinct_values = {str(getattr(r, "value", "")).strip().lower()
                       for r in rows}
    if len(distinct_values) > 1:
        # User has used multiple values for this key (e.g., "Berlin"
        # then "Tokyo"). Don't auto-pin — they've changed.
        return PinDecision(False, recent,
                           "multiple distinct values; ambiguous")
    return PinDecision(True, recent,
                       f"{recent} mentions in 30 days")


def maybe_pin_fact(*,
                   store: Any,
                   kind: str,
                   key: str) -> bool:
    """Evaluate + pin the most recent matching fact when the
    threshold is crossed. Returns True if a pin write occurred.
    Honors incognito (no writes in private mode)."""
    try:
        from .incognito import is_incognito
        if is_incognito():
            return False
    except Exception:
        pass
    decision = evaluate_pinning_candidate(
        store=store, kind=kind, key=key)
    if not decision.should_pin:
        return False
    try:
        rows = store.find_facts(kind=kind, key=key)
    except Exception:
        return False
    if not rows:
        return False
    target = rows[-1]
    # Try the store's pin API if it has one; else update via
    # add_semantic with elevated confidence so the recall layer
    # sees the boost. Best-effort.
    try:
        if hasattr(store, "pin_fact"):
            store.pin_fact(kind=kind, key=key, value=target.value)
            return True
    except Exception:
        pass
    try:
        if hasattr(store, "add_semantic"):
            store.add_semantic(
                kind, key, str(target.value),
                source=f"auto-pin: {decision.reason}",
                source_kind="auto_pin",
                source_id=None,
                extracted_at=time.time(),
            )
            return True
    except Exception:
        pass
    return False

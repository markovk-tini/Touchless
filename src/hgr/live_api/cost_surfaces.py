"""Cost UI surfaces.

Phase-3. The CostMeter (Phase-1) tracks daily LLM spend. This
module produces user-facing READOUTS — strings + structured data
the UI can show in the chat header, settings page, and an
optional in-app receipt at end-of-day.

What it does:
  * `format_today_summary(meter)` — "$0.42 of $5 today (8%)."
  * `format_breakdown(meter)` — per-model bar-chart style breakdown.
  * `format_history(meter, days)` — last N days, one line each.
  * `cost_badge_state(meter)` — small status enum + color hint
    so the chat header pill can render "green / yellow / red".
  * `over_cap_message(meter)` — the user-facing reason why a paid
    request just got blocked.

Stays a pure formatter. No I/O of its own; the caller passes the
CostMeter and any extra context (preferred currency, locale).

Author: Konstantin Markov
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class CostBadge(str, Enum):
    GREEN = "green"      # < 50% of cap
    YELLOW = "yellow"    # 50-79%
    ORANGE = "orange"    # 80-99%
    RED = "red"          # at/over cap


@dataclass
class CostBadgeState:
    badge: CostBadge
    percent_of_cap: float
    spent_usd: float
    cap_usd: float


def cost_badge_state(meter: Any) -> CostBadgeState:
    """Read the meter and map to a badge state for the UI pill.

    cost-1 audit: previously called getattr twice with an eagerly-
    evaluated default that fired even when the outer lookup hit —
    extra DB query per render on properties backed by SQLite."""
    cap = float(_meter_attr(meter, "daily_cap_usd", 5.0))
    spent = _today_total(meter)
    # cost-2 audit: cap<=0 means user disabled the cap. Treat as
    # neutral "GREEN" but reflect "no cap" in pct/cap rendering.
    if cap <= 0:
        return CostBadgeState(badge=CostBadge.GREEN,
                              percent_of_cap=0.0,
                              spent_usd=spent, cap_usd=0.0)
    pct = (spent / cap * 100.0)
    if pct >= 100.0:
        badge = CostBadge.RED
    elif pct >= 80.0:
        badge = CostBadge.ORANGE
    elif pct >= 50.0:
        badge = CostBadge.YELLOW
    else:
        badge = CostBadge.GREEN
    return CostBadgeState(badge=badge, percent_of_cap=pct,
                          spent_usd=spent, cap_usd=cap)


def format_today_summary(meter: Any) -> str:
    st = cost_badge_state(meter)
    if st.cap_usd <= 0:
        return f"${st.spent_usd:.2f} today (no cap set)"
    return (f"${st.spent_usd:.2f} of ${st.cap_usd:.2f} today "
            f"({st.percent_of_cap:.0f}%)")


def format_breakdown(meter: Any, *,
                     top_n: int = 5) -> str:
    """One line per model — sorted by spend desc.

    Missed-by-panel audit: CostMeter.by_model_today() returns a LIST
    of dicts, not a dict — calling .items() raised silently. This
    accepts both shapes: dict {model: cost} OR list of dicts with
    'model' + 'cost_usd' keys."""
    try:
        rows = meter.by_model_today()
    except Exception:
        return ""
    if not rows:
        return "No paid model calls today."
    items: List[tuple] = []
    if isinstance(rows, dict):
        items = sorted(rows.items(), key=lambda kv: -_safe_float(kv[1]))
    elif isinstance(rows, list):
        # Accept {"model": ..., "cost_usd": ...} dicts; tolerate
        # legacy "usd" key.
        for r in rows:
            if not isinstance(r, dict):
                continue
            model = str(r.get("model") or r.get("name") or "?")
            cost = r.get("cost_usd")
            if cost is None:
                cost = r.get("usd")
            items.append((model, _safe_float(cost)))
        items.sort(key=lambda kv: -kv[1])
    items = items[:top_n]
    if not items:
        return "No paid model calls today."
    out_lines = [f"  - {model}: ${cost:.4f}"
                 for model, cost in items]
    return "Spend by model today:\n" + "\n".join(out_lines)


def format_history(meter: Any, *, days: int = 7) -> str:
    """Missed-by-panel audit: real CostMeter.history_last_n_days
    returns dicts with `cost_usd`, not `usd` — the formatter was
    silently printing $0.00 against the real meter. Accept both keys
    for back-compat AND skip malformed rows."""
    try:
        rows = meter.history_last_n_days(days) or []
    except Exception:
        return ""
    if not rows:
        return ""
    out = ["Recent spend:"]
    for r in rows:
        try:
            if isinstance(r, dict):
                date = r.get("date")
                spent = r.get("cost_usd")
                if spent is None:
                    spent = r.get("usd")
            else:
                # tuple / list with (date, cost) at positions 0/1.
                date = r[0]
                spent = r[1] if len(r) > 1 else None
            if date is None or spent is None:
                continue
            out.append(f"  {date}: ${_safe_float(spent):.2f}")
        except Exception:
            continue
    return "\n".join(out) if len(out) > 1 else ""


def over_cap_message(meter: Any) -> str:
    """Missed-by-panel audit: the previous 'until tomorrow' wording
    is misleading for users near a UTC/DST boundary. Use a concrete
    relative window when possible — the cap resets at local midnight."""
    st = cost_badge_state(meter)
    if st.cap_usd <= 0:
        # User disabled the cap; this message shouldn't reach them
        # but be honest about why.
        return ("Daily budget reached — but the cap is set to 0. "
                "Check Iris settings if this is unexpected.")
    return (f"Daily budget reached — ${st.spent_usd:.2f} of "
            f"${st.cap_usd:.2f}. Paid model calls are blocked until "
            f"local midnight. Local model still works.")


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---- helpers ----------------------------------------------------------

def _today_total(meter: Any) -> float:
    fn = getattr(meter, "today_total", None)
    if callable(fn):
        try:
            return float(fn())
        except Exception:
            return 0.0
    val = _meter_attr(meter, "today_total_usd", 0.0)
    try:
        return float(val)
    except Exception:
        return 0.0


def _meter_attr(meter: Any, name: str, default: Any) -> Any:
    return getattr(meter, name, default)

"""Cost + budget watcher with hard caps.

Phase-1 trust substrate. Tracks per-day cumulative LLM spend across
all backends (planner Tier-2 LLM, realtime, future cheap-LLM
synthesizer, MCP-server callouts that hit paid endpoints). Enforces
a HARD ceiling — once the daily cap is hit, every further paid call
short-circuits with a "budget exhausted, falling back to local"
error.

Why this is substrate, not a nice-to-have:
  * A single plan-revision loop bug (Phase-2 work) could fire 1000
    realtime turns in 10 minutes. Without a hard cap, that bankrupts
    a paying user overnight.
  * Self-critique loops + agentic retries (also Phase-2) silently
    multiply per-turn spend. The meter is the only thing that turns
    runaway-loop bugs from "incident" into "user gets a warning".

Storage shape: SQLite at
`%LOCALAPPDATA%\\Touchless\\private\\cost.db` with one table:
    daily_spend(date TEXT, provider TEXT, model TEXT, tokens_in INT,
                tokens_out INT, cost_usd REAL, PRIMARY KEY(date,
                provider, model))

The dollar values are estimates — accurate per-model pricing tables
need to track every OpenAI/Anthropic price drop. We use conservative
ceilings (assume MORE cost than actual) so the warning fires early.
That's the design: never let a runaway-loop bug surprise the user
with a $200 bill at month-end.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import date as _date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _default_cost_db() -> Path:
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(local) / "Touchless" / "private" / "cost.db"


# Per-1M-token USD price ceilings (input, output). Conservative — bias
# HIGH so the meter over-counts and the warning fires early. Update
# when official pricing falls; never count down toward zero based on
# discounts (a billing surprise from a runaway loop is way worse than
# a 10% overcounted estimate).
#
# Sources:
#   * Realtime: ~$5/1M in, ~$20/1M out (audio + text combined estimate).
#   * gpt-5-mini class: ~$0.40/1M in, ~$1.60/1M out.
#   * Sonnet class: ~$3/1M in, ~$15/1M out.
#   * Haiku class: ~$0.25/1M in, ~$1.25/1M out.
#   * Unknown / fallback: assume Sonnet pricing.
_PRICES: Dict[str, Tuple[float, float]] = {
    # OpenAI realtime
    "gpt-realtime":           (5.00, 20.00),
    "gpt-4o-realtime":        (5.00, 20.00),
    "gpt-4o-mini-realtime":   (0.60, 2.40),
    # OpenAI text
    "gpt-5":                  (10.00, 30.00),
    "gpt-5-mini":             (0.40, 1.60),
    "gpt-4o":                 (2.50, 10.00),
    "gpt-4o-mini":            (0.15, 0.60),
    # Anthropic
    "claude-opus":            (15.00, 75.00),
    "claude-sonnet":          (3.00, 15.00),
    "claude-haiku":           (0.25, 1.25),
    # Local — free
    "local":                  (0.0, 0.0),
    "ollama":                 (0.0, 0.0),
    "qwen":                   (0.0, 0.0),
}


def _price_for(model: str) -> Tuple[float, float]:
    """Cheap key-lowercase lookup with sensible fallback."""
    if not model:
        return _PRICES["claude-sonnet"]  # conservative
    m = model.lower()
    if m in _PRICES:
        return _PRICES[m]
    # Prefix matching for versioned model names.
    for key, price in _PRICES.items():
        if m.startswith(key):
            return price
    return _PRICES["claude-sonnet"]


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = _price_for(model)
    return (tokens_in / 1_000_000.0) * pin + (tokens_out / 1_000_000.0) * pout


class CostMeter:
    """Per-process cost meter. Thread-safe. Tracks per-day spend and
    enforces a hard cap.

    Default daily cap: $5.00 (gentle for a free-tier user, doesn't
    surprise a paying subscriber). Override via env
    `TOUCHLESS_DAILY_BUDGET_USD` or `set_daily_cap()`.

    API:
        meter = CostMeter()
        meter.record("gpt-realtime", tokens_in=1500, tokens_out=400)
        meter.today_total()          # → float USD
        meter.is_over_cap()           # → bool
        meter.remaining_today()       # → float USD
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS daily_spend (
        date        TEXT NOT NULL,
        provider    TEXT NOT NULL,
        model       TEXT NOT NULL,
        tokens_in   INTEGER NOT NULL DEFAULT 0,
        tokens_out  INTEGER NOT NULL DEFAULT 0,
        cost_usd    REAL NOT NULL DEFAULT 0,
        updated     REAL NOT NULL,
        PRIMARY KEY (date, provider, model)
    );
    """

    DEFAULT_CAP_USD = 5.00

    def __init__(self, *, db_path: Optional[Path] = None,
                 daily_cap_usd: Optional[float] = None) -> None:
        self._db_path = db_path or _default_cost_db()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.executescript(self.SCHEMA)
        env_cap = os.environ.get("TOUCHLESS_DAILY_BUDGET_USD", "").strip()
        if daily_cap_usd is not None:
            self._cap = float(daily_cap_usd)
        elif env_cap:
            try:
                self._cap = float(env_cap)
            except ValueError:
                self._cap = self.DEFAULT_CAP_USD
        else:
            self._cap = self.DEFAULT_CAP_USD

    # ---- config -------------------------------------------------------

    def set_daily_cap(self, usd: float) -> None:
        with self._lock:
            self._cap = max(0.0, float(usd))

    @property
    def daily_cap_usd(self) -> float:
        return self._cap

    # ---- recording ----------------------------------------------------

    def record(self, model: str, *, tokens_in: int = 0,
               tokens_out: int = 0,
               provider: Optional[str] = None,
               cost_usd: Optional[float] = None) -> float:
        """Add usage to today's tally. Returns the cost added.
        `cost_usd=None` triggers estimate_cost() — pass an explicit
        value when you know the exact billed amount (e.g. from a
        usage callback)."""
        today = _date.today().isoformat()
        prov = (provider or self._provider_for(model)).lower()
        cost = cost_usd if cost_usd is not None else estimate_cost(
            model, tokens_in, tokens_out)
        with self._lock:
            self._conn.execute(
                "INSERT INTO daily_spend"
                "(date, provider, model, tokens_in, tokens_out, cost_usd, updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(date, provider, model) DO UPDATE SET "
                "  tokens_in = tokens_in + excluded.tokens_in, "
                "  tokens_out = tokens_out + excluded.tokens_out, "
                "  cost_usd = cost_usd + excluded.cost_usd, "
                "  updated = excluded.updated",
                (today, prov, model, tokens_in, tokens_out, cost,
                 time.time()),
            )
        return cost

    @staticmethod
    def _provider_for(model: str) -> str:
        m = (model or "").lower()
        if m.startswith("gpt") or "realtime" in m:
            return "openai"
        if m.startswith("claude"):
            return "anthropic"
        if m.startswith("qwen") or m in {"local", "ollama"}:
            return "local"
        return "unknown"

    # ---- queries ------------------------------------------------------

    def today_total(self) -> float:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM daily_spend "
                "WHERE date = ?",
                (_date.today().isoformat(),),
            )
            return float(cur.fetchone()[0])

    def remaining_today(self) -> float:
        return max(0.0, self._cap - self.today_total())

    def is_over_cap(self) -> bool:
        return self.today_total() >= self._cap

    def is_near_cap(self, warn_at: float = 0.70) -> bool:
        """True when spend ≥ warn_at fraction of cap (default 70%).
        Use for soft warnings BEFORE the hard cap kicks in."""
        return self.today_total() >= self._cap * warn_at

    def by_model_today(self) -> List[Dict[str, Any]]:
        """Per-model breakdown for the UI."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT model, provider, tokens_in, tokens_out, cost_usd "
                "FROM daily_spend WHERE date=? ORDER BY cost_usd DESC",
                (_date.today().isoformat(),),
            )
            return [
                {"model": r[0], "provider": r[1],
                 "tokens_in": r[2], "tokens_out": r[3], "cost_usd": r[4]}
                for r in cur.fetchall()
            ]

    def history_last_n_days(self, n: int = 30) -> List[Dict[str, Any]]:
        """Daily rollup for the last N days, oldest first. Includes
        days with zero spend? No — only days where something was
        recorded (sparse table)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT date, SUM(cost_usd) FROM daily_spend "
                "GROUP BY date ORDER BY date DESC LIMIT ?",
                (int(n),),
            )
            rows = [(r[0], float(r[1])) for r in cur.fetchall()]
        rows.reverse()
        return [{"date": d, "cost_usd": c} for d, c in rows]

    # ---- gate ---------------------------------------------------------

    def gate_or_fallback(self, model: str,
                         estimated_tokens_in: int = 0,
                         estimated_tokens_out: int = 0) -> Tuple[bool, str]:
        """Pre-call check: should this call proceed at full spend, or
        be redirected to a free/local fallback?

        Returns (allow, reason). When False, the caller MUST NOT make
        the paid call — fall back to local model or refuse with a
        clear message including `reason`.

        Local-model calls are always allowed (cost=0).
        """
        if estimate_cost(model, estimated_tokens_in,
                         estimated_tokens_out) <= 0:
            return True, "local model (free)"
        if self.is_over_cap():
            return (False,
                    f"daily LLM budget exhausted "
                    f"(${self.today_total():.2f} of ${self._cap:.2f}). "
                    "Falling back to local model. Reset at midnight "
                    "or raise TOUCHLESS_DAILY_BUDGET_USD.")
        if self.is_near_cap():
            return (True,
                    f"approaching daily budget "
                    f"(${self.today_total():.2f} of ${self._cap:.2f}).")
        return True, ""

    # ---- maintenance --------------------------------------------------

    def wipe_all(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM daily_spend")
            before = int(cur.fetchone()[0])
            self._conn.execute("DELETE FROM daily_spend")
            return before

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


_global_meter: Optional[CostMeter] = None
_meter_lock = threading.Lock()


def global_meter() -> CostMeter:
    global _global_meter
    if _global_meter is None:
        with _meter_lock:
            if _global_meter is None:
                _global_meter = CostMeter()
    return _global_meter

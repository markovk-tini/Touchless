"""temporal_patterns — time-of-day pattern detection over the
``tool_call_log.db`` SQLite log.

Scans the append-only tool-call log over a sliding window (default 21
days), buckets each fire by ``(tool_id, hour_of_day, weekday/weekend)``,
and surfaces patterns where a tool repeatedly fires in the same time
slot with low intra-slot variance.

Design notes:

  - **Pure module.** Only I/O is the SQLite read against the supplied
    log path. No network, no Qt, no LLM, no embedder.
  - **Local time only.** All bucketing uses ``datetime.fromtimestamp``
    with no ``tz`` arg so the user's wall-clock hour drives the
    grouping. UTC would surface patterns at the wrong displayed hour.
  - **DST-safe predictions.** ``next_predicted_iso`` advances by
    ``datetime.timedelta(days=N)`` rather than ``+86400`` seconds, so
    spring-forward / fall-back transitions don't drift the predicted
    hour.
  - **Deterministic + idempotent.** Same input rows → same output (no
    randomness, no clock-dependent labels). The only clock dependency
    is ``next_predicted_iso`` which is purely derived from the pattern
    + the supplied ``now_ts``.
  - **Cold-start safe.** Returns ``[]`` when fewer than
    ``MIN_TOTAL_CALLS`` rows in the window. No "learning" placeholder.
  - **Noise filter.** Patterns with hour std-dev > 1.5 hours are
    rejected as too erratic to be useful predictions.
  - **Cross-session.** Aggregates across all ``session_id`` values —
    we want habits, not per-session repetition.

Author: Konstantin Markov
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Module-level tunables (also exposed as kwargs on the public function).
MAX_PATTERNS = 25
MIN_TOTAL_CALLS = 10           # cold-start floor: don't surface below this
MAX_HOUR_STDDEV_HOURS = 1.5    # reject buckets whose firing hour is too noisy
DEFAULT_MIN_OCCURRENCES = 4
DEFAULT_WINDOW_DAYS = 21


# day_of_week → human label. Python: Mon=0..Sun=6.
_DOW_NAMES = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)


def find_time_of_day_patterns(
    tool_call_log_path: Path,
    *,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
    window_days: int = DEFAULT_WINDOW_DAYS,
    max_patterns: int = MAX_PATTERNS,
    min_total_calls: int = MIN_TOTAL_CALLS,
    max_hour_stddev_hours: float = MAX_HOUR_STDDEV_HOURS,
    now_ts: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Detect time-of-day usage patterns in ``tool_call_log.db``.

    Args:
        tool_call_log_path: Path to ``tool_call_log.db`` (the
            append-only SQLite log written by
            ``cortex.tool_call_log.record_tool_call``).
        min_occurrences: Minimum fires within a single
            ``(tool, hour, day_class)`` bucket to surface as a pattern.
        window_days: How many days back to scan (default 21 = three
            weeks of history).
        max_patterns: Cap returned patterns; sorted by confidence desc.
        min_total_calls: Cold-start floor — if fewer total rows fall
            in the window, return ``[]``.
        max_hour_stddev_hours: Reject buckets whose firing-hour std-dev
            exceeds this threshold (default 1.5 hours).
        now_ts: Override "now" for deterministic testing. Defaults to
            ``time.time()``.

    Returns:
        List of pattern dicts (max ``max_patterns`` entries), each:
            {
                "id": str,                       # stable: tool|hour|dayclass
                "tool_id": str,
                "hour": int (0-23),
                "day_of_week": str (e.g. "Monday" | "Saturday" | "any"),
                "day_class": "weekday" | "weekend" | "daily",
                "occurrences": int,
                "confidence": float (0.0-1.0),
                "last_seen_iso": str (local time ISO 8601),
                "next_predicted_iso": str (local time ISO 8601),
            }

        Returns ``[]`` on any of: missing DB, fewer than
        ``min_total_calls`` rows in the window, query failure.
    """
    path = Path(tool_call_log_path)
    if not path.exists():
        return []

    if now_ts is None:
        now_ts = time.time()
    cutoff_ts = now_ts - (window_days * 86400.0)

    try:
        conn = sqlite3.connect(str(path), timeout=2.0)
        try:
            cursor = conn.execute(
                "SELECT tool_id, ts FROM tool_call_log "
                "WHERE ts >= ? ORDER BY ts ASC",
                (cutoff_ts,),
            )
            rows: List[Tuple[str, float]] = [
                (str(r[0]), float(r[1])) for r in cursor.fetchall()
            ]
        finally:
            conn.close()
    except Exception:
        return []

    if len(rows) < int(min_total_calls):
        return []

    # ---- Bucket by (tool_id, hour, day_class) ----
    # day_class is "weekday" (Mon-Fri) or "weekend" (Sat-Sun). We bucket
    # at the day-class level so patterns generalize ("every weekday
    # 9am") rather than fragmenting into seven per-weekday rows. The
    # specific weekday is recoverable later when the bucket happens to
    # be dominated by a single weekday.
    Bucket = Tuple[str, int, str]  # (tool_id, hour, day_class)
    bucket_ts: Dict[Bucket, List[float]] = defaultdict(list)
    bucket_weekdays: Dict[Bucket, List[int]] = defaultdict(list)
    bucket_minutes: Dict[Bucket, List[float]] = defaultdict(list)

    for tool_id, ts in rows:
        try:
            local = _dt.datetime.fromtimestamp(ts)  # local time, no tz arg
        except (ValueError, OSError, OverflowError):
            continue
        hour = local.hour
        weekday = local.weekday()  # 0=Mon..6=Sun
        day_class = "weekend" if weekday >= 5 else "weekday"
        key: Bucket = (tool_id, hour, day_class)
        bucket_ts[key].append(ts)
        bucket_weekdays[key].append(weekday)
        # Hour-as-float (hour + minute/60) for std-dev filtering. This
        # captures how tightly clustered the actual firings are within
        # the chosen hour bucket; a 9am-bucket with firings at 8:55,
        # 9:02, 9:05 has near-zero std-dev, whereas one spread across
        # 9:00, 9:45, 9:58 has noticeably higher spread.
        bucket_minutes[key].append(hour + (local.minute / 60.0))

    # ---- Score each bucket ----
    # confidence = occurrences / window_days, capped at 1.0. So a tool
    # that fires daily for 21 days = 1.0; one that fires 4× in 21 days
    # = ~0.19. We don't divide by "applicable days" (e.g. only weekdays
    # for weekday patterns) because the absolute frequency over the
    # full window is more comparable across pattern types.
    candidates: List[Dict[str, Any]] = []
    denom = max(int(window_days), 1)

    for key, timestamps in bucket_ts.items():
        occurrences = len(timestamps)
        if occurrences < int(min_occurrences):
            continue

        tool_id, hour, day_class = key

        # std-dev filter: drop noisy buckets. With <2 samples std-dev
        # is undefined / zero, but min_occurrences guards us above.
        minute_vals = bucket_minutes[key]
        stddev_hours = _stddev(minute_vals)
        if stddev_hours > float(max_hour_stddev_hours):
            continue

        # day_of_week labeling. If >=80% of the firings fall on a
        # single weekday, label with that day; otherwise generic.
        weekdays = bucket_weekdays[key]
        dow_label, refined_class = _dominant_weekday_label(
            weekdays, day_class
        )

        confidence = min(1.0, occurrences / denom)

        last_seen_ts = max(timestamps)
        last_seen_iso = _to_local_iso(last_seen_ts)

        next_ts = _next_predicted_ts(
            now_ts=now_ts,
            hour=hour,
            weekday_label=dow_label,
            day_class=refined_class,
        )
        next_iso = _to_local_iso(next_ts) if next_ts is not None else ""

        pattern_id = f"{tool_id}|{hour:02d}|{refined_class}"
        if dow_label not in ("any",):
            pattern_id = f"{tool_id}|{hour:02d}|{dow_label}"

        candidates.append({
            "id": pattern_id,
            "tool_id": tool_id,
            "hour": int(hour),
            "day_of_week": dow_label,
            "day_class": refined_class,
            "occurrences": int(occurrences),
            "confidence": round(float(confidence), 4),
            "last_seen_iso": last_seen_iso,
            "next_predicted_iso": next_iso,
        })

    # Deterministic sort: confidence desc, occurrences desc, then by
    # pattern id ascending for stable tie-breaks.
    candidates.sort(
        key=lambda p: (
            -float(p["confidence"]),
            -int(p["occurrences"]),
            str(p["id"]),
        )
    )

    if max_patterns and len(candidates) > int(max_patterns):
        candidates = candidates[: int(max_patterns)]

    return candidates


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stddev(values: List[float]) -> float:
    """Population std-dev. Returns 0.0 for <2 samples."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return variance ** 0.5


def _dominant_weekday_label(
    weekdays: List[int],
    fallback_day_class: str,
) -> Tuple[str, str]:
    """Pick a weekday name if >=80% of fires fall on one day.

    Returns ``(label, refined_class)`` where:

      - ``label`` is a weekday name ("Monday".."Sunday") when one day
        dominates, else ``"any"`` (the bucket spans multiple days
        within the class).
      - ``refined_class`` is "weekday", "weekend", or "daily"; daily
        applies only when fires span BOTH weekday + weekend cleanly,
        which shouldn't happen here because we already split by
        day_class — kept for forward compatibility.
    """
    if not weekdays:
        return ("any", fallback_day_class)
    counts: Dict[int, int] = defaultdict(int)
    for w in weekdays:
        counts[w] += 1
    top_day, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count / len(weekdays) >= 0.8:
        return (_DOW_NAMES[top_day], fallback_day_class)
    return ("any", fallback_day_class)


def _to_local_iso(ts: float) -> str:
    """Format a Unix ts as local-time ISO 8601 (seconds precision)."""
    try:
        return _dt.datetime.fromtimestamp(ts).replace(microsecond=0).isoformat()
    except (ValueError, OSError, OverflowError):
        return ""


def _next_predicted_ts(
    *,
    now_ts: float,
    hour: int,
    weekday_label: str,
    day_class: str,
) -> Optional[float]:
    """Compute the next Unix ts the pattern will match.

    Strategy:

      - If ``weekday_label`` is a specific day name, the next occurrence
        is that day's next instance at ``hour:00`` (today if today is
        that weekday and ``hour`` hasn't passed; otherwise +1..7 days
        ahead).
      - If ``weekday_label`` is ``"any"`` and ``day_class`` is
        "weekday", the next occurrence is the next weekday (Mon-Fri)
        at ``hour:00``, starting today if it qualifies.
      - For ``"weekend"`` day_class, the next Saturday or Sunday at
        ``hour:00``, starting today if it qualifies.

    Uses ``datetime.timedelta(days=N)`` so DST transitions don't drift
    the clock-hour.
    """
    try:
        now_local = _dt.datetime.fromtimestamp(now_ts)
    except (ValueError, OSError, OverflowError):
        return None

    target_today = now_local.replace(
        hour=int(hour), minute=0, second=0, microsecond=0
    )

    def _qualifies(dt: _dt.datetime) -> bool:
        wd = dt.weekday()
        if weekday_label in _DOW_NAMES:
            return _DOW_NAMES[wd] == weekday_label
        if day_class == "weekday":
            return wd < 5
        if day_class == "weekend":
            return wd >= 5
        # "daily" or unknown — every day qualifies.
        return True

    # Walk forward day-by-day (max 8 days to cover one full week + today).
    for offset in range(0, 8):
        candidate = target_today + _dt.timedelta(days=offset)
        if offset == 0 and candidate <= now_local:
            # Today's hour already passed.
            continue
        if _qualifies(candidate):
            return candidate.timestamp()

    return None

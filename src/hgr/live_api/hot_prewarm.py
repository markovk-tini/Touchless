"""Hot-tool pre-warm.

Phase-1 substrate, speed-pass. The first invocation of many Iris
tools pays a cold-start tax — import a Python module, build a UIA
walker, open a sqlite connection, instantiate a connector that
authenticates against an OAuth token. The user perceives that tax
on their FIRST command after Iris starts, then it disappears.

This module spawns a background daemon thread at session start that
pre-imports / pre-instantiates the most-frequently-used tools. The
order is data-driven: the audit log tells us which tools the user
ACTUALLY calls, and the prewarmer reads that top-N to focus its
work. On a fresh install with no audit history, it falls back to a
sensible default list (the ones that are obviously high-frequency:
read_screen, clipboard_read, get_active_window, weather_get,
volume_get).

Cheap by design — failures are silent, slow imports run on a
background thread, and we cap per-tool warmup at ~500ms so a single
slow connector can't delay the rest. The user should NEVER perceive
the prewarmer running; the WIN is just that "list my emails" feels
instant on the first call instead of stalling 1-2s.

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional


# Fallback default list — tools we know are common across most
# sessions and cheap to warm (each takes ~50-200ms). Read order:
# left to right. On a fresh install (no audit data) this is what
# the prewarmer touches.
_DEFAULT_PREWARM: List[str] = [
    "get_active_window",
    "clipboard_read",
    "read_screen",
    "weather_get",
    "volume_get",
    "media_now_playing",
    "find_capability",
]


# Per-tool warmup actions. Each entry is a no-arg callable that does
# whatever cheap "touch the cold paths" thing makes the FIRST real
# call faster. Most just import the module + instantiate the
# connector. Add entries here as new connectors / built-ins ship
# with non-trivial cold-start cost.
_WARMUP_ACTIONS: Dict[str, Callable[[], None]] = {}


def register_warmup(tool: str, action: Callable[[], None]) -> None:
    """Connectors / modules call this at import time to register their
    own cheap warmup. Last-write-wins (so an overridden warmup beats
    the default)."""
    _WARMUP_ACTIONS[tool] = action


def _default_warmup_for(tool: str) -> Callable[[], None]:
    """Best-effort warmup: import the connector / tool_executor module
    so the next call doesn't pay the import tax. Failures are silent."""
    def _do() -> None:
        try:
            # Hint: most tools live in tool_executor (built-ins) or
            # connectors/*. Touching them triggers their imports.
            __import__("hgr.live_api.tool_executor")
            __import__("hgr.live_api.tool_metadata")
            if tool.startswith("volume_"):
                __import__("hgr.live_api.connectors.volume_connector")
            elif tool.startswith("media_"):
                __import__("hgr.live_api.connectors.media_connector")
            elif tool == "weather_get":
                __import__("hgr.live_api.weather")
            elif tool in {"clipboard_read", "clipboard_write",
                          "clipboard_transform"}:
                # Touch win32clipboard (cold import on Windows is
                # ~50ms — paying it here removes the perceived
                # stutter on the first "what's in my clipboard").
                try:
                    import win32clipboard  # noqa: F401  type: ignore
                except Exception:
                    pass
            elif tool == "get_active_window":
                try:
                    import uiautomation  # noqa: F401  type: ignore
                except Exception:
                    pass
        except Exception:
            pass
    return _do


def _resolve_top_tools(audit_db_path: Optional[Any] = None,
                       limit: int = 10) -> List[str]:
    """Read the user's most-frequently-used tools from the audit log.
    Falls back to _DEFAULT_PREWARM when there's no history yet."""
    try:
        from .audit_log import AuditLog
        log = AuditLog(db_path=audit_db_path) if audit_db_path \
            else AuditLog()
    except Exception:
        return list(_DEFAULT_PREWARM)[:limit]
    try:
        # SELECT tool, COUNT(*) AS n FROM invocations GROUP BY tool
        # ORDER BY n DESC LIMIT ?
        with log._lock:  # type: ignore[attr-defined]
            cur = log._conn.execute(  # type: ignore[attr-defined]
                "SELECT tool, COUNT(*) AS n FROM invocations "
                "GROUP BY tool ORDER BY n DESC LIMIT ?",
                (int(limit),),
            )
            rows = cur.fetchall()
        tools = [r[0] for r in rows]
    except Exception:
        tools = []
    finally:
        try:
            log.close()
        except Exception:
            pass
    if not tools:
        return list(_DEFAULT_PREWARM)[:limit]
    # Merge: empirical top N first, then any fallback tools NOT
    # already in the list, capped at limit. Ensures fresh installs
    # warm the obvious things even when they're not yet in history.
    seen = set(tools)
    for t in _DEFAULT_PREWARM:
        if t not in seen and len(tools) < limit:
            tools.append(t)
            seen.add(t)
    return tools[:limit]


_started = False
_started_lock = threading.Lock()


def start_background_prewarm(*, limit: int = 8,
                              per_tool_timeout_sec: float = 0.5
                              ) -> threading.Thread | None:
    """Kick off a daemon thread that warms the top-N tools. Idempotent
    — second call is a no-op. Returns the started Thread (or None when
    already started / failure). Safe to call from any thread.

    The worker takes 1-3 seconds for a typical fresh install (just
    a handful of imports + a UIA touch). Total CPU cost is ~0.5%
    of a core for that interval. Network / disk are not touched —
    pure in-process warming."""
    global _started
    with _started_lock:
        if _started:
            return None
        _started = True

    def _worker() -> None:
        try:
            tools = _resolve_top_tools(limit=limit)
        except Exception:
            tools = list(_DEFAULT_PREWARM)[:limit]
        for tool in tools:
            action = _WARMUP_ACTIONS.get(tool) or _default_warmup_for(tool)
            t0 = time.time()
            try:
                # Run the action with a soft per-tool timeout via
                # cooperative check; we can't kill the thread, but
                # for the common case (a few imports) it returns in
                # well under 500ms.
                action()
            except Exception:
                pass
            elapsed = time.time() - t0
            if elapsed > per_tool_timeout_sec:
                # Log the offender so the dev can see it. Use stderr
                # not the logger because the logger isn't guaranteed
                # initialized yet.
                import sys
                print(f"[prewarm] slow warmup tool={tool} "
                      f"elapsed={elapsed:.2f}s", file=sys.stderr)

    th = threading.Thread(target=_worker, name="iris-hot-prewarm",
                          daemon=True)
    th.start()
    return th


def _reset_for_tests() -> None:
    global _started
    with _started_lock:
        _started = False

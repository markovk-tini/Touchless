"""Speculative warmup — pre-warm the planner provider so the first
real call after ASR doesn't pay TLS + cold-start latency.

Phase-6 latency push. Whisper finalizes a transcript ~300-800ms
after the user stops speaking. The planner LLM call on a cold TLS
connection adds another 150-400ms before the first token. When
those two pieces overlap, the user feels "instant"; when they're
serial, the user feels "thinking".

Strategy: when audio activity is FIRST detected (mic open, VAD
sees speech), fire a tiny no-op request at the planner provider
so the TLS handshake + DNS + edge-warmup happen DURING the user
talking, not after.

Guards:
  * Cooldown so back-to-back utterances don't spam (warmup every
    ~30s max).
  * Provider-aware: ask the ModelRouter who the planner WOULD
    route to, only warm THAT provider.
  * Cost-aware: a warmup is cheap (~100 tokens) but not free.
    Skip when cost-cap is in slow mode.
  * Failure-tolerant: a 500 from the warmup MUST NOT be surfaced
    to the user.

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


_WARMUP_COOLDOWN_SEC = 30.0
_WARMUP_MIN_INTERVAL_SEC = 5.0


@dataclass
class WarmupStats:
    requested: int = 0
    fired: int = 0
    skipped_cooldown: int = 0
    skipped_cost: int = 0
    failed: int = 0
    last_fired_at: float = 0.0
    last_target: str = ""


_GLOBAL_STATS = WarmupStats()


def _resolve_target_provider() -> Optional[str]:
    """Ask the ModelRouter which provider the next planner call
    would route to. None = unknown / no router."""
    try:
        from .model_router import global_router
        r = global_router()
        choice = r.choose_for_planner()
        return getattr(choice, "provider", None)
    except Exception:
        return None


def _provider_in_slow_mode() -> bool:
    """When cost-cap put us into slow mode, skip warmups to save
    pennies."""
    try:
        from .cost_meter import global_meter
        m = global_meter()
        return bool(getattr(m, "is_slow_mode", lambda: False)())
    except Exception:
        return False


def _fire_warmup_openai() -> bool:
    """Tiny ping to OpenAI so TLS + edge warm up. Best-effort."""
    try:
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        # Lightest possible request — just a models list HEAD.
        import urllib.request
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization":
                     f"Bearer {os.environ['OPENAI_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status < 500
    except Exception:
        return False


def _fire_warmup_anthropic() -> bool:
    """Tiny no-op against Anthropic. We can't HEAD the chat
    endpoint, so just open the TLS connection."""
    try:
        import os
        import socket
        import ssl
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        sock = socket.create_connection(
            ("api.anthropic.com", 443), timeout=2.0)
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(sock,
                             server_hostname="api.anthropic.com"):
            return True
    except Exception:
        return False


_PROVIDER_PING = {
    "openai":    _fire_warmup_openai,
    "anthropic": _fire_warmup_anthropic,
}


class SpeculativeWarmup:
    """Tracks last-warm time + provider. Call `notify_speech_start`
    when VAD/ASR sees speech begin; that schedules a warmup on a
    daemon thread."""

    def __init__(self, *,
                 stats: WarmupStats = _GLOBAL_STATS,
                 ping_table: Optional[dict] = None,
                 cooldown_sec: float = _WARMUP_COOLDOWN_SEC):
        self._stats = stats
        self._ping = (
            dict(ping_table) if ping_table is not None
            else dict(_PROVIDER_PING))
        self._cooldown = cooldown_sec
        self._last_fired: float = 0.0
        self._lock = threading.RLock()

    def notify_speech_start(self,
                            target_resolver: Optional[
                                Callable[[], Optional[str]]] = None
                            ) -> bool:
        """Schedule a warmup if cooldown has elapsed + we know who
        the planner would route to. Returns True if fired, False if
        skipped."""
        with self._lock:
            self._stats.requested += 1
            now = time.time()
            if (now - self._last_fired) < self._cooldown:
                self._stats.skipped_cooldown += 1
                return False
            if _provider_in_slow_mode():
                self._stats.skipped_cost += 1
                return False
            resolver = target_resolver or _resolve_target_provider
            target = resolver()
            if not target or target not in self._ping:
                # Fallback — warm whichever provider key is set.
                import os
                if os.environ.get("OPENAI_API_KEY") and \
                        "openai" in self._ping:
                    target = "openai"
                elif os.environ.get("ANTHROPIC_API_KEY") and \
                        "anthropic" in self._ping:
                    target = "anthropic"
                else:
                    self._stats.skipped_cooldown += 1
                    return False
            self._last_fired = now
            self._stats.last_target = target
            t = threading.Thread(
                target=self._do_fire, args=(target,),
                daemon=True, name=f"warmup-{target}")
            t.start()
            return True

    def _do_fire(self, target: str) -> None:
        try:
            fn = self._ping.get(target)
            if fn is None:
                return
            ok = bool(fn())
            with self._lock:
                if ok:
                    self._stats.fired += 1
                else:
                    self._stats.failed += 1
        except Exception:
            with self._lock:
                self._stats.failed += 1

    def reset(self) -> None:
        with self._lock:
            self._last_fired = 0.0


_singleton_lock = threading.Lock()
_singleton: Optional[SpeculativeWarmup] = None


def global_warmup() -> SpeculativeWarmup:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = SpeculativeWarmup()
    return _singleton


def reset_global() -> None:
    global _singleton, _GLOBAL_STATS
    with _singleton_lock:
        _singleton = None
        _GLOBAL_STATS = WarmupStats()


def warmup_stats() -> WarmupStats:
    return _GLOBAL_STATS

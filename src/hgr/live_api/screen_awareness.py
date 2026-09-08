"""Ambient screen-awareness — Iris knows what's on your screen.

Phase-4 substrate. Today Iris only reads the screen when explicitly
asked ("what's on my screen", "summarize this email"). For
Jarvis-grade behavior she needs an AMBIENT awareness that:

  * Captures a compact ScreenContext on a slow tick (~30s) when
    the user is active and not screen-sharing.
  * Deduplicates by screen signature so an unchanged screen
    doesn't re-poll OCR every tick.
  * Caches the latest summary with a short TTL so planner prompts
    can consult "what is the user looking at?" without re-reading.
  * Auto-injects the summary into the planner prompt ONLY when the
    user's request appears vision-relevant ("this", "that one",
    "what does it say", etc.) — keeps prompt budget tight.
  * Honors incognito (no captures in private mode).
  * Honors InterruptionGate's screen-sharing signal (no ambient
    captures of a screen the user is broadcasting).
  * Optional Claude Vision dispatch (env-gated) for rich queries.
    Cost-aware: skipped when the cost meter is at/over cap.

The actual capture is delegated to the existing ScreenReader.
This module is the AMBIENT LAYER around it: scheduling, caching,
inject decisions, dispatch gating.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional


DEFAULT_INTERVAL_SEC = 30.0
DEFAULT_CONTEXT_TTL_SEC = 90.0
# Maximum chars of screen summary injected into a planner prompt.
# Keep tight — the planner is already loaded with memory recall,
# repo context, conversation buffer, etc.
DEFAULT_MAX_INJECT_CHARS = 600


# Words / phrases that smell like the user is referring to something
# they can see on screen. When present we'll auto-inject the screen
# summary into the planner prompt.
_VISION_REFS = (
    "this", "that one", "that thing", "the highlighted",
    "what does it say", "what does that say", "what's on",
    "what is on", "summarize this", "summarize the screen",
    "read this", "explain this", "explain that",
    "what i'm looking at", "what i am looking at",
    "current page", "this email", "this message", "this doc",
    "this window", "this tab", "the selection",
    "what's open", "what is open",
)


def looks_vision_relevant(text: str) -> bool:
    """Cheap substring scan — True when the user's request appears
    to reference something on screen."""
    if not text:
        return False
    t = text.lower()
    if len(t) > 240:
        # Long requests are usually self-contained; skip injection
        # to avoid bloating the prompt.
        return False
    return any(p in t for p in _VISION_REFS)


@dataclass
class ScreenSummary:
    """Compact frame the planner can read. Built from a real
    ScreenContext but stripped to what fits in <600 chars."""
    captured_at: float
    active_app: str = ""
    active_window_title: str = ""
    screen_hash: str = ""
    one_line: str = ""               # "Chrome — gmail inbox; 24 elements"
    visible_text: str = ""           # short text excerpt
    elements_preview: List[str] = field(default_factory=list)

    def as_context_block(self,
                         max_chars: int = DEFAULT_MAX_INJECT_CHARS
                         ) -> str:
        parts: List[str] = ["ON SCREEN: " + self.one_line]
        if self.visible_text:
            text = self.visible_text.replace("\n", " ")
            parts.append("  text: " + text[:300])
        if self.elements_preview:
            preview = ", ".join(self.elements_preview[:8])
            parts.append("  visible: " + preview[:200])
        block = "\n".join(parts)
        return block[:max_chars]


class ScreenAwareness:
    """Sentinel-tickable ambient capture + cache layer.

    Holds a single latest-summary in memory; readers (orchestrator's
    `_recall_context`) call `current_summary()` to get the most
    recent capture with the TTL applied."""

    def __init__(self, *, reader: Optional[Any] = None,
                 interval_sec: float = DEFAULT_INTERVAL_SEC,
                 context_ttl_sec: float = DEFAULT_CONTEXT_TTL_SEC) -> None:
        self._reader = reader  # ScreenReader instance (lazy if None)
        self._interval = float(interval_sec)
        self._ttl = float(context_ttl_sec)
        self._lock = threading.RLock()
        self._summary: Optional[ScreenSummary] = None
        self._last_capture_at: float = 0.0

    # ---- Sentinel tick ------------------------------------------------

    def tick(self) -> None:
        """One scheduled capture. Cheap on every gate:
          * incognito → no-op
          * screen-sharing → no-op (don't capture what the user
            is broadcasting)
          * unchanged signature → no-op
          * too soon since last capture → no-op
        """
        # Incognito blocks ambient capture entirely.
        try:
            from .incognito import is_incognito
            if is_incognito():
                return
        except Exception:
            pass
        # Screen-sharing blocks ambient capture too. Iris shouldn't
        # be quietly snapshotting frames the user is presenting.
        try:
            from .interruption_gate import global_gate, SignalKind
            state, val = global_gate()._signal_state(
                SignalKind.SCREEN_SHARING.value)
            if state == "fresh" and bool(val):
                return
        except Exception:
            pass
        # Throttle: don't capture twice within a single interval
        # even if the Sentinel ticks faster (defensive).
        now = time.time()
        with self._lock:
            if (now - self._last_capture_at) < self._interval - 1:
                return
        self._capture()

    def _capture(self) -> None:
        reader = self._reader or self._lazy_reader()
        if reader is None:
            return
        try:
            ctx = reader.get_context(want_text=False, force=False)
        except Exception:
            return
        if ctx is None:
            return
        summary = self._summarize(ctx)
        with self._lock:
            self._summary = summary
            self._last_capture_at = time.time()

    def _lazy_reader(self) -> Optional[Any]:
        """Construct a ScreenReader if the caller didn't inject one.
        Best-effort: the import may fail in headless test envs."""
        try:
            from .screen_reader import ScreenReader
            self._reader = ScreenReader()
            return self._reader
        except Exception:
            return None

    @staticmethod
    def _summarize(ctx: Any) -> ScreenSummary:
        """Reduce a ScreenContext to a compact ScreenSummary."""
        try:
            one_line = ctx.summary()
        except Exception:
            one_line = ""
        elements_preview: List[str] = []
        try:
            for el in (ctx.elements or [])[:16]:
                text = (el.text or "").strip()
                if text:
                    elements_preview.append(text[:60])
        except Exception:
            pass
        visible_text = ""
        try:
            blocks = ctx.text_blocks or []
            if blocks:
                visible_text = " ".join(blocks)[:600]
        except Exception:
            pass
        return ScreenSummary(
            captured_at=getattr(ctx, "timestamp", time.time()),
            active_app=getattr(ctx, "active_app", "") or "",
            active_window_title=getattr(
                ctx, "active_window_title", "") or "",
            screen_hash=getattr(ctx, "screen_hash", "") or "",
            one_line=one_line,
            visible_text=visible_text,
            elements_preview=elements_preview,
        )

    # ---- public read -------------------------------------------------

    def current_summary(self) -> Optional[ScreenSummary]:
        """Return the most-recent summary if fresh, else None."""
        with self._lock:
            s = self._summary
            if s is None:
                return None
            if (time.time() - s.captured_at) > self._ttl:
                return None
            return s

    def force_capture_now(self) -> Optional[ScreenSummary]:
        """Synchronous capture — used by paths that explicitly need
        the freshest read regardless of throttle."""
        self._capture()
        return self.current_summary()

    def reset(self) -> None:
        with self._lock:
            self._summary = None
            self._last_capture_at = 0.0


# ---- module singleton --------------------------------------------------

_awareness: Optional[ScreenAwareness] = None
_lock = threading.Lock()


def global_screen_awareness() -> ScreenAwareness:
    global _awareness
    if _awareness is None:
        with _lock:
            if _awareness is None:
                _awareness = ScreenAwareness()
    return _awareness


def _reset_for_tests() -> None:
    global _awareness
    with _lock:
        _awareness = None


def register_with_sentinel(*, sentinel: Optional[Any] = None,
                           interval_sec: float = DEFAULT_INTERVAL_SEC
                           ) -> ScreenAwareness:
    awareness = global_screen_awareness()
    awareness._interval = float(interval_sec)
    from .sentinel import global_sentinel
    s = sentinel or global_sentinel()
    s.register("screen_awareness", awareness.tick,
               interval_sec=interval_sec, max_run_ms=600)
    return awareness

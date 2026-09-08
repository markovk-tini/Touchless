"""Vision observer — Iris reasons about what just appeared on screen.

Phase-6 cognition. `screen_awareness.py` (Phase-4) captures the
current screen state — passive snapshot. That's good for "what's
on my screen right now?" The vision observer is DELTA-aware: it
spots TRANSITIONS that probably warrant Iris commenting.

Examples:

  * Stack trace appears in a terminal / IDE → "want me to look
    up this error?"
  * A modal error dialog pops up → "I noticed an error popup —
    read it for you?"
  * A "Save As" dialog appears → "save it where?"
  * A new tab title contains an URL the user just talked about →
    "looks like you opened the Q3 doc — want me to summarize?"

The observer is INTENTIONALLY conservative: it only fires when a
recognized PATTERN appears + 60s have passed since the same
pattern fired. Honors incognito + InterruptionGate + screen-share
(piggybacks on screen_awareness's gate).

Pattern catalogue (initial set):
  - error dialog (window title contains 'error' / 'exception' /
    'crash' / 'failed' / etc.)
  - stack trace (visible text contains 'Traceback' / 'at line' /
    'Exception in thread' patterns)
  - save dialog (title contains 'Save As' / 'Save File')
  - URL link clicked (window title contains a domain mentioned
    in recent conversation)

Future: when `vision_dispatch` is enabled, the observer can
escalate to Claude Vision for richer interpretation ("what's the
error message say exactly?").

Author: Konstantin Markov
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ObservationKind(str, Enum):
    ERROR_DIALOG = "error_dialog"
    STACK_TRACE = "stack_trace"
    SAVE_DIALOG = "save_dialog"
    URL_OPENED = "url_opened"


@dataclass
class Observation:
    kind: ObservationKind
    headline: str
    suggested_action: str = ""
    severity: str = "low"
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


ObservationHandler = Callable[[Observation], None]


# Per-pattern cooldown so seeing the same error twice in a row
# doesn't double-fire.
_PER_KIND_COOLDOWN_SEC: Dict[ObservationKind, float] = {
    ObservationKind.ERROR_DIALOG:  60.0,
    ObservationKind.STACK_TRACE:   120.0,
    ObservationKind.SAVE_DIALOG:   30.0,
    ObservationKind.URL_OPENED:    180.0,
}


# Pattern recognizers — each takes the current ScreenSummary and
# returns either None (no match) or a partial Observation dict
# (kind + payload + headline) the engine wraps + emits.
PatternRecognizer = Callable[[Any], Optional[Dict[str, Any]]]


_ERROR_TITLE_RE = re.compile(
    r"\b(error|exception|crash|failed|failure|warning)\b",
    re.IGNORECASE,
)


def _recognize_error_dialog(summary: Any) -> Optional[Dict[str, Any]]:
    title = (getattr(summary, "active_window_title", "") or "").strip()
    if not title:
        return None
    if not _ERROR_TITLE_RE.search(title):
        return None
    return {
        "kind": ObservationKind.ERROR_DIALOG,
        "headline": (
            f"Noticed an error dialog: \"{title[:80]}\". "
            "Want me to read it and look up what it means?"),
        "suggested_action": "explain_error_dialog",
        "payload": {"title": title},
    }


_STACK_TRACE_RE = re.compile(
    r"(Traceback \(most recent call last\)|"
    r"\bException in thread\b|"
    r"\bat [\w\.$<>]+\([\w.:]+\:\d+\)|"     # Java/JS stack frame
    r"^\s*File \"[^\"]+\", line \d+)",
    re.MULTILINE,
)


def _recognize_stack_trace(summary: Any) -> Optional[Dict[str, Any]]:
    visible = (getattr(summary, "visible_text", "") or "")
    if not visible:
        return None
    m = _STACK_TRACE_RE.search(visible)
    if m is None:
        return None
    # Pull a short excerpt so the headline stays specific.
    snippet = visible[max(0, m.start()):m.start() + 120].replace(
        "\n", " ").strip()
    return {
        "kind": ObservationKind.STACK_TRACE,
        "headline": (
            f"Looks like a stack trace on screen: \"{snippet[:80]}…\". "
            "Want me to look up this error?"),
        "suggested_action": "explain_stack_trace",
        "payload": {"snippet": snippet[:300]},
    }


_SAVE_TITLE_RE = re.compile(
    r"\b(save\s+as|save\s+file|export|open\s+file)\b",
    re.IGNORECASE,
)


def _recognize_save_dialog(summary: Any) -> Optional[Dict[str, Any]]:
    title = (getattr(summary, "active_window_title", "") or "").strip()
    if not title or not _SAVE_TITLE_RE.search(title):
        return None
    return {
        "kind": ObservationKind.SAVE_DIALOG,
        "headline": (
            f"Save dialog open ({title[:60]}). "
            "Where do you want this — Documents, Desktop, or "
            "somewhere else?"),
        "suggested_action": "suggest_save_location",
        "payload": {"title": title},
    }


_DOMAIN_RE = re.compile(
    r"\b((?:[a-z0-9-]+\.)+[a-z]{2,})\b", re.IGNORECASE)


def _recognize_url_opened(summary: Any,
                          *, session_buffer: Optional[Any] = None
                          ) -> Optional[Dict[str, Any]]:
    """When the active window title contains a domain we just
    talked about, surface an offer to help with it."""
    title = (getattr(summary, "active_window_title", "") or "")
    if not title:
        return None
    m = _DOMAIN_RE.search(title)
    if m is None:
        return None
    domain = m.group(1).lower()
    if domain in ("google.com", "youtube.com", "github.com"):
        # These are too common to be useful signals on their own.
        return None
    # Was this domain mentioned in recent conversation?
    if session_buffer is None:
        return None
    try:
        turns = session_buffer.recent(max_turns=6)
    except Exception:
        return None
    if not any(domain in (getattr(t, "text", "") or "").lower()
               for t in turns):
        return None
    return {
        "kind": ObservationKind.URL_OPENED,
        "headline": (
            f"Looks like you just opened {domain} (which you "
            "mentioned a moment ago). Want me to summarize what's "
            "on the page?"),
        "suggested_action": f"summarize_page:{domain}",
        "payload": {"domain": domain, "title": title},
    }


_DEFAULT_RECOGNIZERS: List[PatternRecognizer] = [
    _recognize_error_dialog,
    _recognize_stack_trace,
    _recognize_save_dialog,
    # url_opened is wrapped at tick-time so it can see session_buffer.
]


class VisionObserver:
    """Sentinel-tickable. Compares the current ScreenSummary against
    the last-seen signature; when a recognized pattern matches AND
    the per-pattern cooldown has elapsed, emits an Observation."""

    def __init__(self, *,
                 handler: ObservationHandler,
                 awareness: Optional[Any] = None,
                 session_buffer: Optional[Any] = None,
                 gate: Optional[Any] = None,
                 recognizers: Optional[List[PatternRecognizer]] = None
                 ) -> None:
        self._handler = handler
        self._awareness = awareness
        self._session = session_buffer
        self._gate = gate
        self._recognizers = (
            list(recognizers) if recognizers is not None
            else list(_DEFAULT_RECOGNIZERS))
        self._last_screen_hash: Optional[str] = None
        self._last_fire: Dict[ObservationKind, float] = {}

    def tick(self) -> None:
        try:
            from .incognito import is_incognito
            if is_incognito():
                return
        except Exception:
            pass
        # Respect the gate — vision observer is LOW severity.
        try:
            from .interruption_gate import (global_gate,
                                             InterruptSeverity)
            gate = self._gate or global_gate()
            decision = gate.can_interrupt(InterruptSeverity.LOW)
            if not decision.allow:
                return
        except Exception:
            pass
        summary = self._read_summary()
        if summary is None:
            return
        # Skip when the screen hasn't changed since the last tick.
        screen_hash = getattr(summary, "screen_hash", "")
        if screen_hash and screen_hash == self._last_screen_hash:
            return
        self._last_screen_hash = screen_hash
        now = time.time()
        for recognizer in self._recognizers:
            try:
                hit = recognizer(summary)
            except Exception:
                continue
            if hit is None:
                continue
            self._maybe_emit(hit, now)
        # URL-opened recognizer needs the session buffer, so it
        # runs separately.
        try:
            hit = _recognize_url_opened(
                summary, session_buffer=self._session)
            if hit is not None:
                self._maybe_emit(hit, now)
        except Exception:
            pass

    def _maybe_emit(self, hit: Dict[str, Any], now: float) -> None:
        kind = hit.get("kind")
        if kind is None:
            return
        cooldown = _PER_KIND_COOLDOWN_SEC.get(kind, 60.0)
        last = self._last_fire.get(kind, 0.0)
        if (now - last) < cooldown:
            return
        observation = Observation(
            kind=kind,
            headline=hit.get("headline", ""),
            suggested_action=hit.get("suggested_action", ""),
            severity=hit.get("severity", "low"),
            payload=hit.get("payload", {}),
        )
        try:
            self._handler(observation)
        except Exception:
            return
        self._last_fire[kind] = now

    def _read_summary(self) -> Optional[Any]:
        if self._awareness is not None:
            try:
                return self._awareness.current_summary()
            except Exception:
                return None
        try:
            from .screen_awareness import global_screen_awareness
            return global_screen_awareness().current_summary()
        except Exception:
            return None

    def reset(self) -> None:
        self._last_screen_hash = None
        self._last_fire.clear()


def register_with_sentinel(*, handler: ObservationHandler,
                           sentinel: Optional[Any] = None,
                           interval_sec: float = 8.0):
    observer = VisionObserver(handler=handler)
    from .sentinel import global_sentinel
    s = sentinel or global_sentinel()
    s.register("vision_observer", observer.tick,
               interval_sec=interval_sec, max_run_ms=400)
    return observer

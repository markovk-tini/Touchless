"""Unified screen reader — one ScreenContext from cheap structured signals.

The routing-ladder's screen layer: instead of "screenshot → ask the model"
by default, merge the cheap, structured sources into a single element model,
cache it by screen signature (so an unchanged screen isn't re-read), and
resolve click targets locally before ever paying for vision.

Source priority (cheapest first), all best-effort and guarded:
  1. active window  (foreground_window) — title/process/app
  2. UIA            (UiaController.list_elements) — native control names/types
  3. DOM            (WebController.get_links) — when a browser is foreground
  4. OCR            (ScreenOcr) — only as a fallback / on demand

Vision/screenshots are NOT part of this module — they stay the last resort,
handled by the existing screen_context.py + the model.

What's unit-tested here is the genuinely new logic: the element model, the
hash-based cache (reuse when the active window is unchanged and fresh), and
the find_element match ladder (exact → fuzzy → role+text). The live source
adapters degrade to empty on any error, so this never crashes a session.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

_BROWSER_PROCS = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe"}


@dataclass
class ScreenElement:
    """One interactive/visible element, normalized across sources."""
    text: str
    role: str = ""
    source: str = ""               # 'uia' | 'dom' | 'ocr'
    clickable: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None  # (l,t,r,b) if known
    confidence: float = 1.0


@dataclass
class ScreenContext:
    timestamp: float
    active_app: str = ""
    active_window_title: str = ""
    active_process: str = ""
    elements: List[ScreenElement] = field(default_factory=list)
    text_blocks: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    screen_hash: str = ""
    stale: bool = False

    @property
    def clickable_elements(self) -> List[ScreenElement]:
        return [e for e in self.elements if e.clickable]

    @property
    def visible_elements(self) -> List[ScreenElement]:
        return self.elements

    def summary(self) -> str:
        names = [e.text for e in self.elements][:25]
        return (f"{self.active_app or self.active_process} — "
                f"{self.active_window_title}; {len(self.elements)} elements"
                + (f": {', '.join(names)}" if names else ""))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


class _NullLogger:
    def event(self, *a, **k): pass
    def exception(self, *a, **k): pass


class ScreenReader:
    """Builds + caches a unified ScreenContext from the cheap sources."""

    def __init__(self, *, logger: Any = None, uia: Any = None,
                 web: Any = None, ocr: Any = None, freshness_sec: float = 2.0) -> None:
        self._logger = logger or _NullLogger()
        self._uia = uia
        self._web = web
        self._ocr = ocr
        self._freshness = float(freshness_sec)
        self._cache: Optional[ScreenContext] = None
        self._dirty = True  # an action changed the screen -> rebuild next time

    def invalidate(self) -> None:
        """Call after any tool that changes the screen so the next read is fresh."""
        self._dirty = True

    # ---- public ----
    def get_context(self, *, want_text: bool = False, force: bool = False) -> ScreenContext:
        """Return a unified ScreenContext, reusing the cache when the active
        window is unchanged and the cache is fresh (and nothing dirtied it)."""
        app, title, proc = self._active_window()
        sig = f"{proc}|{title}"
        now = time.time()
        cached = self._cache
        if (not force and not self._dirty and cached is not None
                and cached.screen_hash == sig
                and (now - cached.timestamp) <= self._freshness):
            cached.stale = False
            return cached
        ctx = self._build(app, title, proc, sig, want_text=want_text)
        self._cache = ctx
        self._dirty = False
        return ctx

    def find_element(self, query: str, *, limit: int = 5) -> List[ScreenElement]:
        """Resolve a click target locally: exact → fuzzy → role+text, with an
        OCR fallback. Returns candidates sorted by confidence (highest first)."""
        ctx = self.get_context()
        return self._match(query, ctx.elements, ocr_fallback=True)[:limit]

    # ---- matching ladder (pure; unit-tested) ----
    def _match(self, query: str, elements: List[ScreenElement],
               *, ocr_fallback: bool = False) -> List[ScreenElement]:
        q = _norm(query)
        if not q:
            return []
        scored: List[Tuple[float, ScreenElement]] = []
        q_tokens = set(q.split())
        for e in elements:
            t = _norm(e.text)
            if not t:
                continue
            score = 0.0
            if t == q:
                score = 1.0                               # exact
            elif q in t or t in q:
                score = 0.8                               # substring
            else:
                overlap = q_tokens & set(t.split())
                if overlap:
                    score = 0.5 * (len(overlap) / max(1, len(q_tokens)))  # token overlap
            if score > 0:
                if e.clickable:
                    score += 0.05                         # prefer actionable
                scored.append((min(1.0, score) * float(e.confidence or 1.0), e))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [ScreenElement(**{**e.__dict__, "confidence": s}) for s, e in scored]
        if results or not ocr_fallback or self._ocr is None:
            return results
        # OCR fallback — only when structured sources had no match.
        try:
            res = self._ocr.find_text(query)
            if isinstance(res, dict) and res.get("status") == "ok":
                return [ScreenElement(text=query, role="text", source="ocr",
                                      clickable=True, confidence=0.6)]
        except Exception:
            pass
        return []

    # ---- source adapters (best-effort; never raise) ----
    def _active_window(self) -> Tuple[str, str, str]:
        try:
            from ..debug.foreground_window import get_foreground_window_info
            info = get_foreground_window_info()
            if info is None:
                return "", "", ""
            proc = (info.process_name or "")
            return proc.replace(".exe", ""), (info.title or ""), proc
        except Exception:
            return "", "", ""

    def _build(self, app: str, title: str, proc: str, sig: str,
               *, want_text: bool) -> ScreenContext:
        ctx = ScreenContext(timestamp=time.time(), active_app=app,
                            active_window_title=title, active_process=proc,
                            screen_hash=sig)
        # UIA native controls.
        for el in self._uia_elements(title):
            ctx.elements.append(el)
        if self._uia_elements_used:
            ctx.sources.append("uia")
        # DOM when a browser is foreground.
        if proc.lower() in _BROWSER_PROCS:
            dom = self._dom_elements()
            if dom:
                ctx.elements.extend(dom)
                ctx.sources.append("dom")
        # OCR text only on demand or when nothing structured was found.
        if want_text or not ctx.elements:
            text = self._ocr_text()
            if text:
                ctx.text_blocks = text
                ctx.sources.append("ocr")
        return ctx

    _uia_elements_used = False

    def _uia_elements(self, title: str) -> List[ScreenElement]:
        self._uia_elements_used = False
        if self._uia is None:
            try:
                from .uia_controller import UiaController
                self._uia = UiaController(self._logger)
            except Exception:
                return []
        try:
            res = self._uia.list_elements(limit=40)
        except Exception:
            return []
        if not isinstance(res, dict) or res.get("status") != "ok":
            return []
        self._uia_elements_used = True
        out: List[ScreenElement] = []
        for e in (res.get("elements") or []):
            name = str(e.get("name") or "").strip()
            if not name:
                continue
            out.append(ScreenElement(
                text=name, role=str(e.get("type") or "control"),
                source="uia", clickable=bool(e.get("enabled", True))))
        return out

    def _dom_elements(self) -> List[ScreenElement]:
        if self._web is None:
            return []  # web controller needs app config; only used if injected
        try:
            res = self._web.get_links(limit=30)
        except Exception:
            return []
        if not isinstance(res, dict) or res.get("status") != "ok":
            return []
        out: List[ScreenElement] = []
        for link in (res.get("links") or []):
            text = str((link.get("text") if isinstance(link, dict) else link) or "").strip()
            if not text:
                continue
            out.append(ScreenElement(text=text, role="link", source="dom", clickable=True))
        return out

    def _ocr_text(self) -> List[str]:
        if self._ocr is None:
            try:
                from .screen_ocr import ScreenOcr
                self._ocr = ScreenOcr(self._logger)
            except Exception:
                return []
        try:
            text = self._ocr.read_all_text()
            return [t for t in (text or "").splitlines() if t.strip()][:60]
        except Exception:
            return []

"""Stuck-pattern detector.

Phase-3 Sentinel watcher. Spots the moment when the user has
retried-and-failed-the-same-thing N times in a row — a signal
that they're stuck and Iris should offer help (a clarifying
question, a suggested alternative, a "give me a hint" prompt).

Concrete examples:
  * User asked 3 times "send email to dani" and every send failed
    with `recipient_invalid`. → Suggest contacts_search.
  * User opened the same file 4 times in 10 min. → "Want me to
    pin it to your dock?"
  * User issued the same volume_set request 5 times. → "I think
    your Bluetooth headset is the volume sink. Want me to switch?"

Inputs come from the InvocationBus (Phase-1 substrate) so this
module subscribes rather than touching execution paths. Output is
a callback fire — the UI layer decides what to do with the signal.

Heuristics (deterministic, no LLM):
  * `repeated_error` — same (tool, error_class) ≥ N times in
    `window_sec`.
  * `repeated_action` — same (tool, args_hash) ≥ N times in
    `window_sec` even when all succeed (user is grinding).
  * `circular_dialogue` — same user_text repeats ≥ 3 times in
    a row (Iris isn't understanding).

All thresholds are env-tunable for production calibration.

Author: Konstantin Markov
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional


class StuckKind(str, Enum):
    REPEATED_ERROR = "repeated_error"
    REPEATED_ACTION = "repeated_action"
    CIRCULAR_DIALOGUE = "circular_dialogue"


@dataclass
class StuckSignal:
    kind: StuckKind
    tool: str = ""
    error_class: str = ""
    count: int = 0
    window_sec: float = 0.0
    last_user_text: str = ""
    ts: float = field(default_factory=time.time)
    suggested_action: str = ""


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


REPEATED_ERROR_THRESHOLD = _envi("TOUCHLESS_STUCK_ERR_N", 3)
REPEATED_ERROR_WINDOW = _envf("TOUCHLESS_STUCK_ERR_SEC", 120.0)
REPEATED_ACTION_THRESHOLD = _envi("TOUCHLESS_STUCK_ACT_N", 5)
REPEATED_ACTION_WINDOW = _envf("TOUCHLESS_STUCK_ACT_SEC", 300.0)
CIRCULAR_DIALOGUE_THRESHOLD = _envi("TOUCHLESS_STUCK_DIA_N", 3)


@dataclass
class _ToolEvent:
    tool: str
    args_hash: str
    status: str
    error_class: str
    ts: float


class StuckPatternDetector:
    """Subscribes to the InvocationBus + a text-event hook from
    the UI, emits StuckSignal callbacks when a heuristic fires."""

    HISTORY_CAP = 200

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tool_events: Deque[_ToolEvent] = deque(maxlen=self.HISTORY_CAP)
        self._user_text_history: Deque[str] = deque(maxlen=20)
        self._subs: List[Callable[[StuckSignal], None]] = []
        self._already_fired_for: Dict[str, float] = {}
        self._unsubscribe_bus: Optional[Callable[[], None]] = None

    # ---- bus subscription --------------------------------------------

    def attach_to_bus(self, bus: Optional[Any] = None) -> None:
        # stuck-2 audit: idempotent. Without this, every re-attach
        # leaked the previous subscriber AND doubled invocation
        # delivery, producing false REPEATED_* signals.
        self.detach_from_bus()
        from .tool_invocation import global_bus
        target = bus if bus is not None else global_bus()
        self._unsubscribe_bus = target.subscribe(self._on_invocation)

    def detach_from_bus(self) -> None:
        if self._unsubscribe_bus is not None:
            try:
                self._unsubscribe_bus()
            except Exception:
                pass
            self._unsubscribe_bus = None

    def _on_invocation(self, inv) -> None:
        # stuck-1 audit: narrow the try-block to ONLY the incognito
        # check. Previously a bare except dropped the whole event on
        # any AttributeError, which would silently kill detection
        # the day someone evolved the invocation type.
        extra = getattr(inv, "extra", None)
        if isinstance(extra, dict) and extra.get("incognito"):
            return
        # Classify error if available — reuse reliability_ledger's
        # classifier so error_class strings stay consistent across
        # modules.
        try:
            from .reliability_ledger import classify_error
        except Exception:
            classify_error = lambda x: "other"
        ec = classify_error(inv.error) if inv.status == "error" else ""
        args_hash = _stable_args_hash(getattr(inv, "args", {}) or {})
        with self._lock:
            self._tool_events.append(_ToolEvent(
                tool=inv.tool, args_hash=args_hash,
                status=str(inv.status), error_class=ec,
                ts=time.time(),
            ))
        self._check_tool_patterns()

    # ---- dialog hook --------------------------------------------------

    def record_user_text(self, text: str) -> None:
        """Called by the UI/voice path with the raw user utterance.
        We DON'T persist the text itself, only the rolling history.

        SEC-005 audit: honor incognito. Emitted StuckSignals include
        last_user_text on the signal payload, which would leak the
        utterance to every subscriber (UI, telemetry, cortex). When
        the user is in private mode, skip recording entirely."""
        try:
            from .incognito import is_incognito
            if is_incognito():
                return
        except Exception:
            pass
        normalized = (text or "").strip().lower()
        if not normalized:
            return
        with self._lock:
            self._user_text_history.append(normalized)
        self._check_dialogue_pattern()

    # ---- subscribe ---------------------------------------------------

    def subscribe(self, cb: Callable[[StuckSignal], None]
                  ) -> Callable[[], None]:
        with self._lock:
            self._subs.append(cb)
        def _unsub():
            with self._lock:
                try:
                    self._subs.remove(cb)
                except ValueError:
                    pass
        return _unsub

    def _emit(self, signal: StuckSignal) -> None:
        # Cool-down: don't fire the same signal more than once per
        # 30 seconds to avoid spamming the UI.
        cool_key = f"{signal.kind.value}:{signal.tool}:{signal.error_class}"
        with self._lock:
            now = time.time()
            last = self._already_fired_for.get(cool_key, 0.0)
            if now - last < 30.0:
                return
            self._already_fired_for[cool_key] = now
            subs = list(self._subs)
        for s in subs:
            try:
                s(signal)
            except Exception:
                pass

    # ---- detection logic ---------------------------------------------

    def _check_tool_patterns(self) -> None:
        with self._lock:
            now = time.time()
            err_cutoff = now - REPEATED_ERROR_WINDOW
            act_cutoff = now - REPEATED_ACTION_WINDOW
            err_counts: Dict[tuple, int] = {}
            act_counts: Dict[tuple, int] = {}
            for ev in self._tool_events:
                if ev.status == "error" and ev.ts >= err_cutoff:
                    key = (ev.tool, ev.error_class)
                    err_counts[key] = err_counts.get(key, 0) + 1
                if ev.ts >= act_cutoff:
                    akey = (ev.tool, ev.args_hash)
                    act_counts[akey] = act_counts.get(akey, 0) + 1

        for (tool, ec), n in err_counts.items():
            if n >= REPEATED_ERROR_THRESHOLD:
                self._emit(StuckSignal(
                    kind=StuckKind.REPEATED_ERROR,
                    tool=tool, error_class=ec, count=n,
                    window_sec=REPEATED_ERROR_WINDOW,
                    suggested_action=_suggest_for_error(tool, ec),
                ))
        for (tool, _), n in act_counts.items():
            if n >= REPEATED_ACTION_THRESHOLD:
                self._emit(StuckSignal(
                    kind=StuckKind.REPEATED_ACTION,
                    tool=tool, count=n,
                    window_sec=REPEATED_ACTION_WINDOW,
                    suggested_action=_suggest_for_action(tool),
                ))

    def _check_dialogue_pattern(self) -> None:
        with self._lock:
            if len(self._user_text_history) < CIRCULAR_DIALOGUE_THRESHOLD:
                return
            recent = list(self._user_text_history)[-CIRCULAR_DIALOGUE_THRESHOLD:]
        if all(t == recent[0] for t in recent):
            self._emit(StuckSignal(
                kind=StuckKind.CIRCULAR_DIALOGUE,
                count=len(recent),
                last_user_text=recent[0],
                suggested_action=("ask the user to rephrase or describe "
                                  "what they expected to happen"),
            ))

    # ---- maintenance --------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._tool_events.clear()
            self._user_text_history.clear()
            self._already_fired_for.clear()


# ---- helpers ----------------------------------------------------------

def _stable_args_hash(args: Dict[str, Any]) -> str:
    """Stable hash of args for repeat-action detection. Sorts keys +
    hashes JSON repr."""
    try:
        import json
        text = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        text = str(sorted(args.items())) if isinstance(args, dict) else str(args)
    h = hashlib.sha1(text.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def _suggest_for_error(tool: str, error_class: str) -> str:
    if error_class == "recipient_invalid":
        return f"try contacts_search before {tool}"
    if error_class == "auth_revoked":
        return f"re-run iris_setup_tool for {tool}"
    if error_class == "rate_limited":
        return f"wait + retry {tool} with backoff"
    if error_class == "not_connected":
        return f"prompt user to connect the upstream connector for {tool}"
    return f"offer the user a different path; {tool} keeps failing"


def _suggest_for_action(tool: str) -> str:
    if tool == "volume_set":
        return "ask if the user wants to switch audio output device"
    if tool == "open_url":
        return "ask if the user wants to pin/bookmark this URL"
    if tool.endswith("_compose") or tool.endswith("_send"):
        return "ask if the user wants to draft a template for this recipient"
    return f"ask the user if they want a shortcut for {tool}"


# ---- module singleton --------------------------------------------------
# Mirrors the global_*() pattern used by sentinel, cost_meter,
# audit_log, etc. so the Phase-3 wiring patch in OPEN_ISSUES.md §4.5.1
# can do `global_stuck_detector().attach_to_bus()` consistently.

_detector: Optional["StuckPatternDetector"] = None
_lock = threading.Lock()


def global_stuck_detector() -> "StuckPatternDetector":
    global _detector
    if _detector is None:
        with _lock:
            if _detector is None:
                _detector = StuckPatternDetector()
    return _detector


def _reset_for_tests() -> None:
    global _detector
    with _lock:
        target = _detector
        _detector = None
    if target is not None:
        try:
            target.detach_from_bus()
        except Exception:
            pass

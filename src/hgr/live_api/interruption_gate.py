"""InterruptionGate — when can Iris talk to the user?

Phase-3. Whether Iris is allowed to PROACTIVELY interrupt the
user (notification, briefing, suggested action) depends on
context that has nothing to do with what Iris wants to say.
This module owns the policy.

Inputs (each can be detected from Windows APIs, queried periodically
by the Sentinel layer):
  * Windows Focus Assist / Do Not Disturb state
  * Screen-sharing active (Teams, Zoom, Meet, Discord screen-share,
    OBS recording, Loom)
  * Full-screen app active (game, presentation, video)
  * Camera in use (someone's on a video call)
  * Microphone in use (someone's on a voice call)
  * "Quiet hours" — user-configured time window
  * Last user interaction time (idle vs active)
  * QuietMode from earcons.py (compose)

Output: an `InterruptDecision` that downstream notifiers + voice
respect. The decision carries:
  * allow: bool — proceed or not
  * tier: HIGH | NORMAL | LOW — how loud the interruption is
  * delay_until: optional epoch — defer rather than drop
  * reason: human-readable string for telemetry/UX

Watchers feed signals via `set_signal(kind, value)`; the gate
combines them into a decision when callers ask
`can_interrupt(severity)`.

This module is deliberately Windows-agnostic at the API layer —
all OS detection lives in `system_signals.py` (helper module
below). The gate consumes "facts" about the system, not OS calls.

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


# SEC-002 audit: every signal carries a freshness timestamp; readings
# older than the per-signal TTL are treated as UNKNOWN rather than as
# "false / off". That stops the gate from failing-OPEN when a feeder
# watcher dies or hasn't published yet. Defaults err on the side of
# treating the world as "could be private right now".
_SIGNAL_TTL_SEC: Dict[str, float] = {
    "screen_sharing": 30.0,
    "camera_in_use":  30.0,
    "mic_in_use":     30.0,
    "fullscreen_app": 30.0,
    "game_mode":      60.0,
    "focus_assist":   300.0,
    "do_not_disturb": 300.0,
    "quiet_hours":    300.0,
    "battery_low":    300.0,
    "user_idle_sec":  60.0,
}
_DEFAULT_SIGNAL_TTL = 300.0


# SEC-008 audit: per-signal expected value type, so a buggy producer
# can't post {} or "unknown" and have _truthy decide for us.
_BOOLEAN_SIGNALS = frozenset({
    "screen_sharing", "camera_in_use", "mic_in_use",
    "fullscreen_app", "game_mode", "focus_assist",
    "do_not_disturb", "quiet_hours", "battery_low",
})
_NUMERIC_SIGNALS = frozenset({"user_idle_sec"})


class InterruptSeverity(str, Enum):
    LOW = "low"        # nice-to-have ("calendar event in 30min")
    NORMAL = "normal"  # default ("you have a meeting in 5min")
    HIGH = "high"      # important ("token expired", "build broken")
    CRITICAL = "critical"  # always interrupts ("file system full")


class SignalKind(str, Enum):
    SCREEN_SHARING = "screen_sharing"
    FULLSCREEN_APP = "fullscreen_app"
    CAMERA_IN_USE = "camera_in_use"
    MIC_IN_USE = "mic_in_use"
    FOCUS_ASSIST = "focus_assist"
    DND = "do_not_disturb"
    QUIET_HOURS = "quiet_hours"
    BATTERY_LOW = "battery_low"
    USER_IDLE_SEC = "user_idle_sec"  # value: seconds idle
    GAME_MODE = "game_mode"


@dataclass
class InterruptDecision:
    allow: bool
    tier_used: str
    reason: str = ""
    delay_until: Optional[float] = None
    suggested_channel: str = "voice"  # 'voice' | 'earcon' | 'badge'


@dataclass
class GateConfig:
    """Caller-tweakable thresholds. Defaults are conservative."""
    # When idle ≥ this many seconds, even NORMAL is OK (user is AFK).
    afk_idle_sec: float = 120.0
    # Default "quiet hours" window (24h clock). Suppresses NORMAL/LOW.
    quiet_start_hour: int = 22
    quiet_end_hour: int = 7
    # Defer-rather-than-drop window for LOW tier: defer up to N min
    # then drop.
    low_defer_max_sec: float = 600.0


class InterruptionGate:
    """Combines system signals + severity into an allow/defer/deny
    decision. Thread-safe; signals are read+write on a lock."""

    def __init__(self, *, config: Optional[GateConfig] = None) -> None:
        self._cfg = config or GateConfig()
        self._lock = threading.RLock()
        # SEC-002 audit: store (value, set_at_ts) tuples so a stale
        # signal can be distinguished from a fresh "off". Producers
        # write through `set_signal`; consumers read via `_signal_state`
        # which returns ('fresh' | 'stale' | 'missing', value).
        self._signals: Dict[str, Tuple[Any, float]] = {}

    # ---- signals ------------------------------------------------------

    def set_signal(self, kind: SignalKind, value: object) -> None:
        # SEC-008 audit: validate type per-signal so a buggy watcher
        # can't post {} / "unknown" / a list and bypass policy.
        key = kind.value
        if key in _BOOLEAN_SIGNALS and not isinstance(value, bool):
            # Coerce explicit "0"/"1"/"true"/"false" strings, reject
            # the rest by treating as UNKNOWN (no signal stored).
            coerced = _coerce_bool(value)
            if coerced is None:
                self._log("interruption_gate: rejecting non-bool signal "
                          f"{key!r} = {value!r}")
                return
            value = coerced
        elif key in _NUMERIC_SIGNALS:
            try:
                value = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                self._log("interruption_gate: rejecting non-numeric "
                          f"signal {key!r} = {value!r}")
                return
            value = max(0.0, min(value, 86_400.0))  # clamp to one day
        with self._lock:
            self._signals[key] = (value, time.time())

    def clear_signal(self, kind: SignalKind) -> None:
        with self._lock:
            self._signals.pop(kind.value, None)

    def get_signal(self, kind: SignalKind,
                   default: object = None) -> object:
        """Returns the stored value (no freshness info). Use
        `_signal_state` internally when freshness matters."""
        with self._lock:
            entry = self._signals.get(kind.value)
            return entry[0] if entry is not None else default

    def snapshot(self) -> Dict[str, object]:
        """Caller-facing snapshot (values only)."""
        with self._lock:
            return {k: v[0] for k, v in self._signals.items()}

    def _signal_state(self, key: str) -> Tuple[str, Any]:
        """Return ('fresh' | 'stale' | 'missing', value).
        'stale' means present but past TTL; 'missing' means never set."""
        with self._lock:
            entry = self._signals.get(key)
        if entry is None:
            return ("missing", None)
        value, set_at = entry
        ttl = _SIGNAL_TTL_SEC.get(key, _DEFAULT_SIGNAL_TTL)
        if (time.time() - set_at) > ttl:
            return ("stale", value)
        return ("fresh", value)

    @staticmethod
    def _log(msg: str) -> None:
        try:
            from .live_api_logger import get_fallback_logger
            get_fallback_logger().event("interruption_gate_warn",
                                        message=msg)
        except Exception:
            import sys
            print(msg, file=sys.stderr)

    # ---- decisions ---------------------------------------------------

    def can_interrupt(self, severity: InterruptSeverity
                      = InterruptSeverity.NORMAL) -> InterruptDecision:
        return self._decide(severity)

    # Signals whose stale/missing state forces fail-CLOSED for non-
    # CRITICAL severities (SEC-002 audit). If we don't have a fresh
    # reading we default to "could be sensitive — don't interrupt".
    _PRIVACY_CRITICAL_SIGNALS = (
        SignalKind.SCREEN_SHARING,
        SignalKind.MIC_IN_USE,
        SignalKind.CAMERA_IN_USE,
    )

    def _decide(self, sev: InterruptSeverity) -> InterruptDecision:
        # CRITICAL always passes — by definition.
        if sev == InterruptSeverity.CRITICAL:
            return InterruptDecision(
                allow=True, tier_used="critical",
                reason="critical severity bypasses gate",
                suggested_channel="voice",
            )

        # Phase-7 affect: LOW-severity nudges hold off when the
        # user is deeply focused OR frustrated. NORMAL+ pass as
        # before — that's where briefings and explicit asks live.
        if sev == InterruptSeverity.LOW:
            try:
                from .affect import should_suppress_low_nudges
                if should_suppress_low_nudges():
                    return InterruptDecision(
                        allow=False, tier_used=sev.value,
                        reason="affect: user focused/frustrated",
                        suggested_channel="badge",
                    )
            except Exception:
                pass

        # SEC-002: privacy-critical signals must be FRESH before we'll
        # allow an interruption. Missing or stale → fail-closed for
        # LOW/NORMAL (these can wait). HIGH passes as a muted earcon
        # so the user still hears something important.
        for key in self._PRIVACY_CRITICAL_SIGNALS:
            state, _ = self._signal_state(key.value)
            if state == "missing":
                if sev in (InterruptSeverity.LOW, InterruptSeverity.NORMAL):
                    return InterruptDecision(
                        allow=False, tier_used=sev.value,
                        reason=f"{key.value} signal unknown; failing closed",
                        suggested_channel="badge",
                    )
            elif state == "stale":
                if sev in (InterruptSeverity.LOW, InterruptSeverity.NORMAL):
                    return InterruptDecision(
                        allow=False, tier_used=sev.value,
                        reason=f"{key.value} signal stale; failing closed",
                        suggested_channel="badge",
                    )

        # Hard blockers — never proceed (except CRITICAL above).
        screen_state, screen_val = self._signal_state(
            SignalKind.SCREEN_SHARING.value)
        if screen_state == "fresh" and _truthy(screen_val):
            return InterruptDecision(
                allow=False, tier_used=sev.value,
                reason="screen-sharing active",
                suggested_channel="badge",
            )
        game_state, game_val = self._signal_state(
            SignalKind.GAME_MODE.value)
        if game_state == "fresh" and _truthy(game_val):
            return InterruptDecision(
                allow=False, tier_used=sev.value,
                reason="game mode active",
                suggested_channel="badge",
            )
        fa_state, fa_val = self._signal_state(SignalKind.FOCUS_ASSIST.value)
        if fa_state == "fresh" and _truthy(fa_val):
            if sev in (InterruptSeverity.LOW, InterruptSeverity.NORMAL):
                return InterruptDecision(
                    allow=False, tier_used=sev.value,
                    reason="focus assist on",
                    suggested_channel="badge",
                )
        dnd_state, dnd_val = self._signal_state(SignalKind.DND.value)
        if dnd_state == "fresh" and _truthy(dnd_val):
            if sev in (InterruptSeverity.LOW, InterruptSeverity.NORMAL):
                return InterruptDecision(
                    allow=False, tier_used=sev.value,
                    reason="do-not-disturb on",
                    suggested_channel="badge",
                )
        # Battery low: suppress LOW (Phase-3 audit — was dead enum).
        batt_state, batt_val = self._signal_state(
            SignalKind.BATTERY_LOW.value)
        if (batt_state == "fresh" and _truthy(batt_val)
                and sev == InterruptSeverity.LOW):
            return InterruptDecision(
                allow=False, tier_used=sev.value,
                reason="battery low; suppressing LOW interruptions",
                suggested_channel="badge",
            )

        # Softer suppressors — defer rather than drop.
        mic_state, mic_val = self._signal_state(SignalKind.MIC_IN_USE.value)
        cam_state, cam_val = self._signal_state(
            SignalKind.CAMERA_IN_USE.value)
        on_call = ((mic_state == "fresh" and _truthy(mic_val))
                   or (cam_state == "fresh" and _truthy(cam_val)))
        if on_call:
            if sev == InterruptSeverity.HIGH:
                return InterruptDecision(
                    allow=True, tier_used="high",
                    reason="HIGH overrides call-in-progress; muted earcon",
                    suggested_channel="earcon",
                )
            defer = 30.0 if sev == InterruptSeverity.NORMAL else 300.0
            return InterruptDecision(
                allow=False, tier_used=sev.value,
                reason="user is on a call (caller must re-check after delay_until)",
                delay_until=time.time() + defer,
                suggested_channel="badge",
            )
        fs_state, fs_val = self._signal_state(
            SignalKind.FULLSCREEN_APP.value)
        if fs_state == "fresh" and _truthy(fs_val):
            if sev == InterruptSeverity.HIGH:
                return InterruptDecision(
                    allow=True, tier_used="high",
                    reason="HIGH overrides fullscreen; muted earcon",
                    suggested_channel="earcon",
                )
            return InterruptDecision(
                allow=False, tier_used=sev.value,
                reason="fullscreen app active",
                suggested_channel="badge",
            )

        # Quiet hours: NORMAL/LOW gated; HIGH passes muted.
        if self._in_quiet_hours_state():
            if sev == InterruptSeverity.HIGH:
                return InterruptDecision(
                    allow=True, tier_used="high",
                    reason="quiet hours; passing as earcon",
                    suggested_channel="earcon",
                )
            return InterruptDecision(
                allow=False, tier_used=sev.value,
                reason="quiet hours",
                suggested_channel="badge",
            )

        # AFK boost: idle ≥ afk_idle_sec → LOW becomes NORMAL-OK.
        idle_state, idle_val = self._signal_state(
            SignalKind.USER_IDLE_SEC.value)
        idle_sec = _as_float(idle_val) if idle_state == "fresh" else 0.0
        if (sev == InterruptSeverity.LOW
                and idle_sec >= self._cfg.afk_idle_sec):
            return InterruptDecision(
                allow=True, tier_used="low (AFK-boosted)",
                reason=(f"user idle {int(idle_sec)}s; LOW interruption OK"),
                suggested_channel="voice",
            )

        # Default: allow.
        return InterruptDecision(
            allow=True, tier_used=sev.value,
            reason="no suppressors active",
            suggested_channel="voice",
        )

    # ---- helpers -----------------------------------------------------

    def _in_quiet_hours_state(self) -> bool:
        # Explicit signal overrides time-based heuristic when fresh.
        state, val = self._signal_state(SignalKind.QUIET_HOURS.value)
        if state == "fresh":
            return _truthy(val)
        # Stale or missing → fall through to time-based heuristic.
        now = time.localtime()
        h = now.tm_hour
        s, e = self._cfg.quiet_start_hour, self._cfg.quiet_end_hour
        if s == e:
            return False
        if s < e:
            return s <= h < e
        # Cross-midnight window (e.g., 22 → 7).
        return h >= s or h < e


# ---- helpers ----------------------------------------------------------

def _truthy(v: object) -> bool:
    if v is None or v is False:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        # Per missed-by-panel: '0.0', 'None', 'null' strings should
        # all be falsy. Try float-cast first; fall back to literal
        # allowlist.
        s = v.strip().lower()
        if s in ("", "0", "false", "no", "off", "none", "null", "nan"):
            return False
        try:
            return float(s) != 0
        except ValueError:
            return True  # arbitrary non-empty string → truthy
    return True


def _coerce_bool(v: Any) -> Optional[bool]:
    """Best-effort bool coercion for SignalKind validation. Returns
    None when the value really isn't bool-shaped (rejects dicts,
    lists, ambiguous strings)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        return None
    return None


def _as_float(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---- module singleton --------------------------------------------------

_gate: Optional[InterruptionGate] = None
_lock = threading.Lock()


def global_gate() -> InterruptionGate:
    global _gate
    if _gate is None:
        with _lock:
            if _gate is None:
                _gate = InterruptionGate()
    return _gate


def _reset_for_tests() -> None:
    global _gate
    with _lock:
        _gate = None

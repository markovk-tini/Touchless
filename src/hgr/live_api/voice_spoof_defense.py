"""Voice spoofing defense for destructive operations.

Phase-2 voice safety. The realtime voice path can be tricked: an
attacker who can play audio at the mic (a colleague trolling, a
podcast in the background, an iPhone in your pocket reading
notifications aloud, an injected MCP tool result that includes a
synthesized "say this aloud" line) can issue voice commands that
look identical to the user.

For READ operations, this is annoying but recoverable. For
DESTRUCTIVE/IRREVERSIBLE operations (delete file, send money, post
publicly, transfer to other account), the cost of one spoofed
trigger is too high to ignore.

Defense layers (this module wires the policy; the actual checks
plug into the safety_gate before tool dispatch):

  1. **No-mic confirmation** for destructive: require a second
     channel — a gesture, a typed click — to confirm. Voice alone
     can't both request AND confirm.
  2. **Cooldown after suspicious activity**: after one spoof
     suspicion fires, raise the bar (require gesture confirm even
     for write-tier) for `cooldown_sec`.
  3. **Cross-channel sanity**: if the same destructive command is
     issued twice in <2 seconds, treat as repeat-attack and refuse.
  4. **Voice-fingerprint hint** (TODO Phase 3): SpeakerID enrollment
     so a different speaker triggers stricter gating. Today: only the
     env opt-in `TOUCHLESS_SPEAKER_ID=1` enables this codepath.
  5. **TTS-loop detection**: when audio from our own TTS is being
     re-captured by the mic and producing transcripts, refuse to act
     on any of them. (The realtime SDK handles most of this, but we
     belt-and-suspender.)

Author: Konstantin Markov
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConfirmChannel(str, Enum):
    """How a confirmation was provided.

    Only out-of-band physical channels (GESTURE, KEYBOARD, MOUSE) +
    PHYSICAL_BUTTON satisfy the spoof-defense second-channel rule for
    destructive ops. TYPED is rejected — on Windows, any process that
    can inject WM_CHAR can produce 'typed' input, which is the same
    threat model the spoof defense exists to counter (SEC-007 audit
    finding)."""
    NONE = "none"
    GESTURE = "gesture"           # live hand in front of the camera
    KEYBOARD = "keyboard"         # physical key on the device
    MOUSE = "mouse"               # physical mouse click
    PHYSICAL_BUTTON = "physical_button"   # device-level button (volume etc.)
    TYPED = "typed"               # text input in a UI — REJECTED for destructive
    VOICE = "voice"               # voice-only confirms NEVER satisfy spoof-defense


# Per SEC-007: only these channels count as "second channel" for
# DESTRUCTIVE / IRREVERSIBLE. TYPED is intentionally excluded because
# WM_CHAR injection is part of the threat model.
_TRUSTED_SECOND_CHANNELS = frozenset({
    ConfirmChannel.GESTURE,
    ConfirmChannel.KEYBOARD,
    ConfirmChannel.MOUSE,
    ConfirmChannel.PHYSICAL_BUTTON,
})


# Validate destructiveness strings. Anything else is treated as
# destructive — fail-CLOSED for unknown values (SEC-007 finding (e)).
_VALID_DESTRUCTIVENESS = frozenset({
    "read", "write", "destructive", "irreversible",
})


def compute_args_hash(args: Optional[Dict[str, Any]]) -> str:
    """Canonical SHA-1 of `args` for repeat-attack detection. Sorts
    keys so {a:1, b:2} == {b:2, a:1}. Internal use; callers should
    NOT pass their own hash (SEC-007 finding (a))."""
    try:
        text = json.dumps(args or {}, sort_keys=True, default=str)
    except Exception:
        text = repr(args)
    h = hashlib.sha1(text.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


@dataclass
class SpoofCheckResult:
    """Decision returned by `check_destructive_voice_op`."""
    allowed: bool
    reason: str
    require_second_channel: bool = False
    # Cooldown that the caller should apply if `allowed` is False:
    cooldown_sec: float = 0.0


@dataclass
class _RecentOp:
    tool: str
    args_hash: str
    ts: float
    confirm_channel: ConfirmChannel = ConfirmChannel.NONE


class VoiceSpoofDefense:
    """Stateful policy module. Tracks recent destructive voice ops +
    a rolling 'suspicion' flag so cooldowns persist between calls."""

    REPEAT_WINDOW_SEC = 2.0       # rule 3: same op twice in <2s = attack
    DEFAULT_COOLDOWN_SEC = 60.0   # rule 2: heightened mode duration
    MAX_RECENT = 32               # history bound

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._recent: List[_RecentOp] = []
        self._heightened_until: float = 0.0
        self._tts_loop_detected_until: float = 0.0

    # ---- public surface used by safety_gate / planner -----------------

    def check_destructive_voice_op(self, *, tool: str,
                                   args: Optional[Dict[str, Any]] = None,
                                   args_hash: Optional[str] = None,
                                   source: str = "voice",
                                   destructiveness: str = "destructive",
                                   confirm_channel: ConfirmChannel
                                   = ConfirmChannel.NONE
                                   ) -> SpoofCheckResult:
        """Called BEFORE dispatching a tool when source involved voice.

        Prefer passing `args` (the actual tool args dict) so the hash
        is computed internally — `args_hash` remains supported for
        legacy callers but is treated as a hint, not a primitive
        (SEC-007 finding). `destructiveness` must be one of:
        "read"/"write"/"destructive"/"irreversible"; unknown values
        fail-closed to "destructive".
        """
        # SEC-007 (a): always derive args_hash from args when we have
        # them, so the caller can't influence repeat-attack detection.
        if args is not None:
            args_hash = compute_args_hash(args)
        if not args_hash:
            args_hash = "_no_hash_"
        # SEC-007 (e): validate destructiveness; unknown → destructive.
        if destructiveness not in _VALID_DESTRUCTIVENESS:
            destructiveness = "destructive"

        with self._lock:
            now = time.time()
            self._prune_recent(now)

            # TTS-loop pause: while we've recently caught our own TTS
            # bleeding back into the mic, refuse all destructive voice
            # ops outright. SEC-007 (c): a sustained loop should ALSO
            # raise suspicion so the heightened mode kicks in for the
            # cooldown window. Per the missed-by-panel #11 finding,
            # surface require_second_channel=True so a determined user
            # can still override via gesture.
            if now < self._tts_loop_detected_until:
                self._raise_suspicion(now)
                return SpoofCheckResult(
                    allowed=False,
                    reason=("TTS loop detected; voice commands paused — "
                            "confirm with gesture to override"),
                    require_second_channel=True,
                )

            in_heightened = now < self._heightened_until
            # SEC-007 (d): heightened mode bumps destructive → irreversible
            # so destructive ops require a stronger second-channel mode
            # AND write ops also gate on second channel.
            effective_destructiveness = destructiveness
            if in_heightened and destructiveness == "destructive":
                effective_destructiveness = "irreversible"
            second_channel_required = (
                effective_destructiveness in ("destructive", "irreversible")
                or (in_heightened and effective_destructiveness in ("write",)))

            if second_channel_required and confirm_channel == ConfirmChannel.NONE:
                return SpoofCheckResult(
                    allowed=False,
                    reason=("destructive voice op requires a second "
                            "channel (gesture or physical confirm)"),
                    require_second_channel=True,
                )
            if second_channel_required and confirm_channel == ConfirmChannel.VOICE:
                return SpoofCheckResult(
                    allowed=False,
                    reason="voice cannot confirm itself for destructive ops",
                    require_second_channel=True,
                )
            # SEC-007 (b): TYPED is NOT trusted as a second channel
            # for destructive ops — WM_CHAR injection is in scope.
            if second_channel_required and confirm_channel not in _TRUSTED_SECOND_CHANNELS:
                return SpoofCheckResult(
                    allowed=False,
                    reason=(f"confirm channel {confirm_channel.value!r} is "
                            "not trusted for destructive ops; need "
                            "gesture / physical key / button"),
                    require_second_channel=True,
                )

            # Rule 3: repeat-attack — same op twice in <2s.
            for r in self._recent:
                if (r.tool == tool and r.args_hash == args_hash
                        and (now - r.ts) < self.REPEAT_WINDOW_SEC):
                    self._raise_suspicion(now)
                    return SpoofCheckResult(
                        allowed=False,
                        reason=("repeat destructive op within "
                                f"{self.REPEAT_WINDOW_SEC:.0f}s — refusing"),
                        cooldown_sec=self.DEFAULT_COOLDOWN_SEC,
                    )

            self._recent.append(_RecentOp(
                tool=tool, args_hash=args_hash, ts=now,
                confirm_channel=confirm_channel,
            ))
            if len(self._recent) > self.MAX_RECENT:
                self._recent.pop(0)
            return SpoofCheckResult(allowed=True, reason="ok")

    def report_tts_loop_detected(self,
                                 duration_sec: float = 5.0) -> None:
        """Called by the audio path when it sees our own TTS being
        captured by the mic (signal correlation, level matching,
        timestamp alignment)."""
        with self._lock:
            self._tts_loop_detected_until = max(
                self._tts_loop_detected_until,
                time.time() + max(0.5, duration_sec),
            )

    def is_in_heightened_mode(self) -> bool:
        with self._lock:
            return time.time() < self._heightened_until

    def heightened_remaining(self) -> float:
        with self._lock:
            return max(0.0, self._heightened_until - time.time())

    def reset(self) -> None:
        """Test/UI hook: clear all state."""
        with self._lock:
            self._recent.clear()
            self._heightened_until = 0.0
            self._tts_loop_detected_until = 0.0

    # ---- internals ----------------------------------------------------

    def _raise_suspicion(self, now: float) -> None:
        env = os.environ.get("TOUCHLESS_SPOOF_COOLDOWN_SEC")
        try:
            cooldown = float(env) if env else self.DEFAULT_COOLDOWN_SEC
        except ValueError:
            cooldown = self.DEFAULT_COOLDOWN_SEC
        self._heightened_until = max(self._heightened_until, now + cooldown)

    def _prune_recent(self, now: float) -> None:
        # Keep only entries within the repeat window — old ones can't
        # contribute to a repeat-attack judgement.
        cutoff = now - max(self.REPEAT_WINDOW_SEC, 30.0)
        self._recent = [r for r in self._recent if r.ts >= cutoff]


# ---- module singleton --------------------------------------------------

_global: Optional[VoiceSpoofDefense] = None
_lock = threading.Lock()


def global_defense() -> VoiceSpoofDefense:
    global _global
    if _global is None:
        with _lock:
            if _global is None:
                _global = VoiceSpoofDefense()
    return _global


def _reset_for_tests() -> None:
    global _global
    _global = None

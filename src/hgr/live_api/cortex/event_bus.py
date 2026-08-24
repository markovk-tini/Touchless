"""CortexEventBus — buffer, throttle, and coalesce events for the
Cortex visualization.

Iris subsystems (realtime_client, memory/manager, tool_registry,
tool_executor, connectors, planner/orchestrator) call methods on
this bus. The bus enforces three rules from
docs/IRIS_VISUALIZATION.md:

  - Cap output to ~30 events/sec (1 frame at 30fps)
  - Coalesce rapid `core.audio` samples (keep latest within 33 ms)
  - Coalesce rapid `edge.pulse` on the same edge (within 100 ms)
  - Coalesce `node.activity` on the same node (keep max intensity
    within 100 ms)

Buffered events flush either on a QTimer tick or when explicitly
asked (e.g. when the JS scene finishes loading).

The bus knows nothing about the bridge transport; it just calls
a writer callable. CortexWindow wires it to CortexBridge.emit_event.

Author: Konstantin Markov
"""
from __future__ import annotations

import sys
import time
from collections import OrderedDict
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[cortex-bus {time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


# Flush cadence — 30 Hz matches the spec and is comfortable for the
# JS scene running at 60 fps (gives it two render frames per event tick).
_FLUSH_INTERVAL_MS = 33

# Coalescing windows.
_AUDIO_COALESCE_MS = 33   # drop intermediate samples within one tick
_PULSE_COALESCE_MS = 100  # merge rapid pulses on the same edge
_ACT_COALESCE_MS = 100    # merge rapid activity on the same node


class CortexEventBus(QObject):
    """Throttle + coalesce gateway between Iris and the Cortex window."""

    # Max events to keep buffered in the passthrough queue while suspended
    # (no writer attached / view not loaded). Bounded so a long pre-open
    # window can't OOM the process; the oldest events drop FIFO.
    #
    # Bumped from 50 → 500 because the prior cap silently FIFO-dropped
    # neuron events fired by long pre-open conversations (each user prompt
    # can produce 10+ leaf.add / node.spawn / state events). At 500 we
    # comfortably survive a full warm-up + several prompts before the JS
    # scene attaches.
    _MAX_BUFFERED_PASSTHROUGH = 500

    def __init__(
        self,
        writer: Optional[Callable[[dict], None]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        # Writer is optional at construction so the bus can be created
        # ahead of the view (LiveAssistantWindow pre-creates the bus at
        # __init__ so cortex_emit calls don't drop before the user clicks
        # Iris). set_writer() attaches the real writer once the bridge
        # is ready. When writer is None, _send() drops the event.
        self._writer = writer
        # Last `core.audio` payload buffered for this tick.
        self._pending_audio: Optional[dict] = None
        # node_id -> the activity dict to flush this tick (max intensity wins)
        self._pending_activity: "OrderedDict[str, dict]" = OrderedDict()
        # (from, to) -> the pulse dict to flush this tick
        self._pending_pulse: "OrderedDict[tuple, dict]" = OrderedDict()
        # Everything else is sent straight through into this queue.
        self._passthrough: list[dict] = []
        # Drop counters (visible via .stats() for the debug overlay)
        self._dropped_audio = 0
        self._dropped_pulse = 0
        self._dropped_activity = 0
        self._dropped_passthrough = 0  # FIFO drops when buffer full
        # Sent counter (total events delivered to JS)
        self._sent = 0
        # Last-state tracker for `core.state` so we don't spam unchanged states.
        self._last_state: Optional[str] = None
        # Most-recent activity timestamp per node, for the inter-event window.
        self._last_activity_ts_ms: dict[str, float] = {}
        self._last_pulse_ts_ms: dict[tuple, float] = {}
        # Whether to write through immediately or wait for the next flush.
        self._suspended = False

        self._timer = QTimer(self)
        self._timer.setInterval(_FLUSH_INTERVAL_MS)
        self._timer.timeout.connect(self._flush)
        self._timer.start()

    # ---- public ingress methods (Iris subsystems call these) ----

    def _append_passthrough(self, event: dict) -> None:
        """Append to the passthrough queue, dropping the oldest entry
        when the queue would exceed _MAX_BUFFERED_PASSTHROUGH. Keeps
        memory bounded while suspended (no writer / view collapsed)."""
        self._passthrough.append(event)
        if len(self._passthrough) > self._MAX_BUFFERED_PASSTHROUGH:
            # FIFO drop — preserves the freshest events for replay.
            overflow = len(self._passthrough) - self._MAX_BUFFERED_PASSTHROUGH
            self._dropped_passthrough += overflow
            self._passthrough = self._passthrough[-self._MAX_BUFFERED_PASSTHROUGH:]

    def core_state(self, state: str, intensity: float = 1.0) -> None:
        """Iris core transitioned to a new state."""
        if state == self._last_state:
            return
        self._last_state = state
        self._append_passthrough(
            {"type": "core.state", "state": state, "intensity": float(intensity)}
        )

    def core_audio(self, rms: float) -> None:
        """Voice level sample. Coalesced to one per flush tick."""
        if self._pending_audio is not None:
            self._dropped_audio += 1
        self._pending_audio = {"type": "core.audio", "rms": float(rms)}

    def node_spawn(
        self,
        node_id: str,
        label: str,
        category: str = "project",
        weight: float = 0.6,
        parent_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Add a new active-context or sub-context node.

        ``extra`` keys (e.g. ``preview=True``, ``path="..."``) are
        passed through verbatim to the JS scene — Phase 2 uses
        ``preview`` to mark tiny constellation-dot children that
        expand to full-size nodes when their parent is focused;
        ``path`` carries a file path for double-click-to-open.
        """
        event: Dict[str, Any] = {
            "type": "node.spawn",
            "id": node_id,
            "label": label,
            "category": category,
            "weight": float(weight),
            "parent_id": parent_id,
        }
        if extra:
            event.update(extra)
        self._append_passthrough(event)

    def node_activity(
        self,
        node_id: str,
        intensity: float = 1.0,
        duration_ms: int = 600,
    ) -> None:
        """Light a node up (memory retrieval, tool fire, etc.)."""
        now = time.monotonic() * 1000.0
        last = self._last_activity_ts_ms.get(node_id, 0.0)
        if now - last < _ACT_COALESCE_MS and node_id in self._pending_activity:
            # Within window — merge by keeping highest intensity.
            prev = self._pending_activity[node_id]
            if float(intensity) > float(prev.get("intensity", 0.0)):
                prev["intensity"] = float(intensity)
                prev["duration_ms"] = max(int(duration_ms), int(prev.get("duration_ms", duration_ms)))
            self._dropped_activity += 1
            return
        self._last_activity_ts_ms[node_id] = now
        self._pending_activity[node_id] = {
            "type": "node.activity",
            "id": node_id,
            "intensity": float(intensity),
            "duration_ms": int(duration_ms),
        }

    def edge_pulse(
        self,
        from_id: str,
        to_id: str,
        color: str = "cyan",
        duration_ms: int = 500,
    ) -> None:
        """Send a traveling pulse along an edge."""
        key = (from_id, to_id)
        now = time.monotonic() * 1000.0
        last = self._last_pulse_ts_ms.get(key, 0.0)
        if now - last < _PULSE_COALESCE_MS and key in self._pending_pulse:
            self._dropped_pulse += 1
            return
        self._last_pulse_ts_ms[key] = now
        self._pending_pulse[key] = {
            "type": "edge.pulse",
            "from": from_id,
            "to": to_id,
            "color": color,
            "duration_ms": int(duration_ms),
        }

    def leaf_add(
        self,
        parent_id: str,
        label: str,
        ttl_ms: int = 8000,
    ) -> None:
        """Attach a transient leaf node to a context node."""
        self._append_passthrough({
            "type": "leaf.add",
            "parent_id": parent_id,
            "label": label,
            "ttl_ms": int(ttl_ms),
        })

    def node_fade(self, node_id: str, duration_ms: int = 1500) -> None:
        """Smoothly remove a context node."""
        self._append_passthrough({
            "type": "node.fade",
            "id": node_id,
            "duration_ms": int(duration_ms),
        })

    # ---- introspection (debug overlay can read these) ----

    def stats(self) -> dict:
        return {
            "sent": self._sent,
            "dropped_audio": self._dropped_audio,
            "dropped_pulse": self._dropped_pulse,
            "dropped_activity": self._dropped_activity,
            "dropped_passthrough": self._dropped_passthrough,
            "buffered_passthrough": len(self._passthrough),
            "buffered_activity": len(self._pending_activity),
            "buffered_pulse": len(self._pending_pulse),
            "suspended": self._suspended,
            "has_writer": self._writer is not None,
        }

    def is_suspended(self) -> bool:
        return self._suspended

    # ---- flushing ----

    def suspend(self) -> None:
        """Hold events until resume() (used while the JS page is loading,
        or while the embedded cortex panel is collapsed). Events keep
        accumulating in pending_*; the passthrough list is bounded to
        _MAX_BUFFERED_PASSTHROUGH entries so a long suspension can't
        grow unbounded."""
        self._suspended = True

    def resume(self) -> None:
        """Stop suspending and flush whatever buffered events we have
        into the writer. Safe to call multiple times."""
        was_suspended = self._suspended
        self._suspended = False
        # INSTRUMENTATION: always log resume() with full state so we can
        # diagnose whether events ever drain. Previously only logged when
        # there were buffered events; that hid the failure mode where the
        # writer was None at resume time and events were silently dropped
        # on subsequent _flush ticks.
        buffered = (
            len(self._passthrough)
            + len(self._pending_activity)
            + len(self._pending_pulse)
        )
        _log(
            f"resume() called: was_suspended={was_suspended}, "
            f"buffered={buffered} "
            f"(passthrough={len(self._passthrough)}, "
            f"activity={len(self._pending_activity)}, "
            f"pulse={len(self._pending_pulse)}), "
            f"dropped_passthrough={self._dropped_passthrough}, "
            f"writer={'set' if self._writer is not None else 'NONE - events will be silently dropped'}"
        )
        if self._writer is None and buffered > 0:
            _log(
                f"WARNING: resume() called with {buffered} buffered events "
                f"but no writer attached — these events will be lost on flush"
            )
        self._flush()

    def set_writer(self, writer: Optional[Callable[[dict], None]]) -> None:
        """Swap the writer at runtime — used when the embedded cortex
        view loads (attach bridge.emit_event) or unloads (back to None).
        Buffered events fire through the next writer on resume()."""
        self._writer = writer

    def has_writer(self) -> bool:
        return self._writer is not None

    def _flush(self) -> None:
        if self._suspended:
            return
        # Order: state-y events first, then activity/pulse, then audio.
        # Passthrough drains all state, spawn, fade, leaf events.
        if self._passthrough:
            batch = self._passthrough
            self._passthrough = []
            for ev in batch:
                self._send(ev)
        if self._pending_activity:
            batch = list(self._pending_activity.values())
            self._pending_activity.clear()
            for ev in batch:
                self._send(ev)
        if self._pending_pulse:
            batch = list(self._pending_pulse.values())
            self._pending_pulse.clear()
            for ev in batch:
                self._send(ev)
        if self._pending_audio is not None:
            self._send(self._pending_audio)
            self._pending_audio = None

    def _send(self, ev: dict) -> None:
        writer = self._writer
        if writer is None:
            # No view attached — drop silently. Suspended-mode buffering
            # in pending_* / _passthrough already absorbed it; _flush
            # only reaches _send when not suspended, so this branch fires
            # only if a writer was cleared between suspend() ticks.
            return
        try:
            writer(ev)
            self._sent += 1
        except Exception as exc:
            _log(f"writer failed for {ev.get('type', '?')}: {exc}")

    # ---- lifecycle ----

    def shutdown(self) -> None:
        try:
            self._timer.stop()
        except Exception:
            pass

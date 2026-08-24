"""Python <-> JavaScript bridge for the Cortex visualization.

A QObject registered on a QWebChannel that the page's JS code can
talk to (and that we push events into). One signal (`event`) carries
JSON-encoded events outward to JS; two `@Slot`s receive interaction
events back from the user (node focus, leaf open).

This object is created on the GUI thread by CortexWindow and lives
exactly as long as the window. It is the only Python object the JS
side knows about — all other Iris subsystems push through
CortexEventBus, which then calls .emit_event() here.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import sys
import time
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal, Slot


def _log(msg: str) -> None:
    """Light stderr logger; matches the rest of the live_api package."""
    try:
        sys.stderr.write(f"[cortex-bridge {time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


class CortexBridge(QObject):
    """QWebChannel-exposed object — the only seam between Iris and JS.

    JS side reads `event(jsonStr)` and listens to it as a signal.
    Python side calls `emit_event(payload)` to push events.
    """

    # Signal fired toward JS. Payload is a JSON string (already
    # encoded so JS just JSON.parses once).
    event = Signal(str)

    # ---- discrete signals (OPEN_ISSUES #10) ------------------------------
    # Convenience signals that mirror the world_state observer events so
    # the iris_simulator JS can connect handlers per event type instead of
    # filtering inside a single "event" router. Each payload is a JSON
    # string with the relevant fields (project_id, label, root_path, …).
    # Strictly additive — the generic `event` signal above is still the
    # primary transport for cortex_emit / CortexEventBus events.
    projectAdded = Signal(str)
    projectRemoved = Signal(str)
    leafAdded = Signal(str)
    toolUsed = Signal(str)
    patternAdded = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._on_node_focused = None
        self._on_node_unfocused = None
        self._on_leaf_opened = None
        self._on_open_file = None
        self._on_ready = None
        self._ready = False
        # Optional direct JS-injection callback. When set, this is called
        # in ADDITION to the QWebChannel signal emit so we have a second
        # path for events even if the signal path is dropping. The window
        # / launcher wires this to ``view.page().runJavaScript(...)``.
        # Signature: callable(js_source: str) -> None.
        self._js_runner: Optional[Callable[[str], None]] = None  # type: ignore[name-defined]

    # ---- inbound to Python (called from JS) ----

    @Slot()
    def jsReady(self) -> None:  # noqa: N802 (JS-style for the JS side)
        """JS calls this once the page + scene are fully initialized.

        We use it to flush any buffered events that arrived while the
        page was still loading (cold-start safety)."""
        self._ready = True
        _log("JS ready")
        if self._on_ready is not None:
            try:
                self._on_ready()
            except Exception as exc:
                _log(f"on_ready callback failed: {exc}")

    @Slot(str)
    def nodeFocused(self, node_id: str) -> None:  # noqa: N802
        """User clicked a node in the visualization."""
        if self._on_node_focused is not None:
            try:
                self._on_node_focused(node_id)
            except Exception as exc:
                _log(f"on_node_focused callback failed: {exc}")

    @Slot()
    def nodeUnfocused(self) -> None:  # noqa: N802
        """User pressed Esc / clicked background — return to overview."""
        if self._on_node_unfocused is not None:
            try:
                self._on_node_unfocused()
            except Exception as exc:
                _log(f"on_node_unfocused callback failed: {exc}")

    @Slot(str)
    def leafOpened(self, leaf_id: str) -> None:  # noqa: N802
        """User double-clicked a leaf — Iris should open the artifact."""
        if self._on_leaf_opened is not None:
            try:
                self._on_leaf_opened(leaf_id)
            except Exception as exc:
                _log(f"on_leaf_opened callback failed: {exc}")

    @Slot(str)
    def openFile(self, path: str) -> None:  # noqa: N802
        """User double-clicked a node carrying a file path — open it
        in the OS default handler."""
        if self._on_open_file is not None:
            try:
                self._on_open_file(path)
            except Exception as exc:
                _log(f"on_open_file callback failed: {exc}")

    # ---- outbound to JS ----

    def emit_event(self, payload: dict) -> None:
        """Push one event to the JS scene.

        Payload is the dict described in docs/IRIS_VISUALIZATION.md
        ("Data model — events Iris must emit"). We JSON-encode here
        so the JS side parses exactly once."""
        try:
            kind = payload.get("type", "?")
            # INSTRUMENTATION: sample-log the first 20 events so we can
            # confirm Python is actually pushing dotted-form types over
            # the bridge. After that throttle to 1-in-50 to keep logs sane.
            self._emit_count = getattr(self, "_emit_count", 0) + 1
            if self._emit_count <= 20 or self._emit_count % 50 == 0:
                _log(f"emit_event #{self._emit_count} type={kind}")
            self.event.emit(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            _log(f"emit_event failed for {payload.get('type', '?')}: {exc}")

    def is_ready(self) -> bool:
        """True once the JS scene has finished loading and called jsReady."""
        return self._ready

    # ---- discrete signal helpers (OPEN_ISSUES #10) ----------------------
    # These wrap the discrete Signal(str) emitters so callers don't have
    # to know JSON encoding rules. Safe to call from any thread —
    # Signal.emit() auto-queues to the GUI thread for QWebChannel marshal.

    def emit_project_added(self, payload: dict) -> None:
        """Push a project-added event to JS via QWebChannel signal AND
        as a JS-injection fallback. Same belt-and-braces rationale as
        emit_project_removed — the signal is normally reliable but the
        injection guarantees the visual lands."""
        try:
            self.projectAdded.emit(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            _log(f"emit_project_added signal emit failed: {exc}")
        # Fallback: directly call window.liveAddProject via runJavaScript.
        try:
            pid = str(payload.get("project_id", "") or "")
            label = str(payload.get("label", "") or pid)
            if pid and self._js_runner is not None:
                escaped_pid = json.dumps(pid)
                escaped_label = json.dumps(label)
                js = (
                    "(function(){"
                    "try{"
                    f"console.log('[iris-sim] fallback inject liveAddProject', {escaped_pid});"
                    f"if(typeof window.liveAddProject==='function')"
                    f"  window.liveAddProject({escaped_pid},{escaped_label});"
                    f"else if(typeof liveAddProject==='function')"
                    f"  liveAddProject({escaped_pid},{escaped_label});"
                    "else console.warn('[iris-sim] liveAddProject unavailable');"
                    "}catch(e){console.warn('[iris-sim] fallback inject failed:',e);}"
                    "})();"
                )
                self._js_runner(js)
        except Exception as exc:
            _log(f"emit_project_added JS fallback failed: {exc}")

    def set_js_runner(self, runner: Optional[Callable[[str], None]]) -> None:
        """Register a callable that runs a JS source string in the
        attached page (typically ``view.page().runJavaScript``). Used
        for the direct-inject fallback in emit_project_added /
        emit_project_removed. None disables the fallback."""
        self._js_runner = runner

    def has_js_runner(self) -> bool:
        return self._js_runner is not None

    def emit_project_removed(self, payload: dict) -> None:
        """Push a project-removal event to JS. Symmetric counterpart
        to emit_project_added — fired by world_state.remove_project()
        and by the iris_remove_project tool so the cortex viz fades
        out the project node + its leaves.

        Belt-and-braces: in ADDITION to the QWebChannel signal emit,
        also invokes the registered JS runner to call
        ``window.handleProjectRemoved(<project_id>)`` directly. The
        signal path is normally reliable but we've seen it drop under
        load — the direct injection guarantees the fade still happens.
        """
        try:
            self.projectRemoved.emit(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            _log(f"emit_project_removed signal emit failed: {exc}")
        # Fallback: directly call the JS handler via runJavaScript.
        try:
            pid = str(payload.get("project_id", "") or "")
            if pid and self._js_runner is not None:
                # JSON-escape the id to handle anything weird (quotes,
                # backslashes). Triple-fallback chain on the JS side:
                # window.handleProjectRemoved → handleProjectRemoved →
                # no-op + warn (so we always see in console whether the
                # function name was reachable).
                escaped = json.dumps(pid)
                js = (
                    "(function(){"
                    "try{"
                    f"console.log('[iris-sim] fallback inject handleProjectRemoved', {escaped});"
                    f"if(typeof window.handleProjectRemoved==='function')"
                    f"  window.handleProjectRemoved({escaped});"
                    f"else if(typeof handleProjectRemoved==='function')"
                    f"  handleProjectRemoved({escaped});"
                    "else console.warn('[iris-sim] handleProjectRemoved unavailable');"
                    "}catch(e){console.warn('[iris-sim] fallback inject failed:',e);}"
                    "})();"
                )
                self._js_runner(js)
        except Exception as exc:
            _log(f"emit_project_removed JS fallback failed: {exc}")

    def emit_leaf_added(self, payload: dict) -> None:
        try:
            self.leafAdded.emit(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            _log(f"emit_leaf_added failed: {exc}")

    def emit_tool_used(self, payload: dict) -> None:
        try:
            self.toolUsed.emit(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            _log(f"emit_tool_used failed: {exc}")

    def emit_pattern_added(self, payload: dict) -> None:
        try:
            self.patternAdded.emit(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            _log(f"emit_pattern_added failed: {exc}")

    # ---- callbacks (Iris side wires these) ----

    def set_node_focused_callback(self, cb) -> None:
        self._on_node_focused = cb

    def set_node_unfocused_callback(self, cb) -> None:
        self._on_node_unfocused = cb

    def set_leaf_opened_callback(self, cb) -> None:
        self._on_leaf_opened = cb

    def set_open_file_callback(self, cb) -> None:
        self._on_open_file = cb

    def set_ready_callback(self, cb) -> None:
        self._on_ready = cb


# ---- module-level active-bridge registry (OPEN_ISSUES #10) ----------------
# Lets non-GUI subsystems (tool_executor, world_state, memory.manager) push
# discrete signals without taking a hard import dependency on the cortex
# window. The simulator launcher (run_iris_simulator.main) calls
# ``set_active_bridge(bridge)`` after constructing CortexBridge, and clears
# it when the window closes. Every emit_* call site uses ``get_active_bridge()``
# and no-ops if it returns None — so non-simulator runs are unaffected.
#
# Single writer (the GUI thread at simulator startup / window close); many
# readers (any subsystem firing signals). Plain module-global is fine — the
# worst case is a few signals dropped right at open/close, which is harmless
# (the cortex visual is best-effort by design, same contract as cortex_emit).
_ACTIVE_BRIDGE: Optional["CortexBridge"] = None


def set_active_bridge(bridge: Optional["CortexBridge"]) -> None:
    """Register the live CortexBridge. Called once by run_iris_simulator after
    the bridge is constructed; passed ``None`` when the simulator shuts down.
    Safe to call from any thread (single-load atomic write)."""
    global _ACTIVE_BRIDGE
    _ACTIVE_BRIDGE = bridge


def get_active_bridge() -> Optional["CortexBridge"]:
    """Return the currently-registered bridge, or None when no cortex
    window/simulator is active. All call sites MUST None-check and swallow
    exceptions — bridge calls are decorative and must never break Iris."""
    return _ACTIVE_BRIDGE

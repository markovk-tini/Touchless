"""cortex_emit — module-level shim Iris subsystems use to push events
to the Cortex visualization without taking a hard dependency on the
cortex package.

When the Cortex window is open, it registers its CortexEventBus via
``set_bus(bus)``. When it closes, it calls ``clear_bus()``. All emit
functions below silently no-op when no bus is registered — so every
Iris subsystem can sprinkle ``cortex_emit.node_activity(...)`` calls
freely without conditional checks. Zero overhead when the window is
closed (one dict lookup + None check).

This intentionally mirrors CortexEventBus's public surface so the
two stay one-to-one. If you're adding a new event type, add it in
both places.

Author: Konstantin Markov
"""
from __future__ import annotations

import sys
import time
from typing import Optional

# The bus is held module-globally and accessed without locks. Single
# writer (the GUI thread when the cortex window opens/closes); many
# readers (any subsystem emitting events). On Python, dict / attribute
# loads of a module global are atomic enough for this use; the worst
# case is a few events dropped right at open/close, which is harmless
# (the cortex visual is best-effort by design).
_bus = None  # type: Optional[object]


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[cortex-emit {time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def set_bus(bus) -> None:
    """Register the live CortexEventBus. Called by CortexWindow."""
    global _bus
    _bus = bus


def clear_bus(bus=None) -> None:
    """Unregister the bus. If a specific bus is passed, only clears
    when it matches (prevents a stale close from clobbering a fresh
    open). No-op if nothing is registered."""
    global _bus
    if bus is None or _bus is bus:
        _bus = None


def is_active() -> bool:
    """True when a CortexEventBus is currently registered."""
    return _bus is not None


# ---- emit functions (mirror CortexEventBus) ----
# Each function reads the bus once into a local to avoid races with
# clear_bus() and to keep the hot path branch-free past the None check.

def core_state(state: str, intensity: float = 1.0) -> None:
    bus = _bus
    if bus is None:
        return
    try:
        bus.core_state(state, intensity)
    except Exception as exc:
        _log(f"core_state failed: {exc}")


def core_audio(rms: float) -> None:
    bus = _bus
    if bus is None:
        return
    try:
        bus.core_audio(rms)
    except Exception as exc:
        _log(f"core_audio failed: {exc}")


def node_spawn(
    node_id: str,
    label: str,
    category: str = "project",
    weight: float = 0.6,
    parent_id: Optional[str] = None,
) -> None:
    bus = _bus
    if bus is None:
        return
    try:
        bus.node_spawn(node_id, label, category, weight, parent_id)
    except Exception as exc:
        _log(f"node_spawn failed: {exc}")


def node_activity(node_id: str, intensity: float = 1.0, duration_ms: int = 150) -> None:
    bus = _bus
    if bus is None:
        return
    try:
        bus.node_activity(node_id, intensity, duration_ms)
    except Exception as exc:
        _log(f"node_activity failed: {exc}")


def edge_pulse(
    from_id: str,
    to_id: str,
    color: str = "cyan",
    duration_ms: int = 125,
) -> None:
    bus = _bus
    if bus is None:
        return
    try:
        bus.edge_pulse(from_id, to_id, color, duration_ms)
    except Exception as exc:
        _log(f"edge_pulse failed: {exc}")


def leaf_add(parent_id: str, label: str, ttl_ms: int = 8000) -> None:
    bus = _bus
    if bus is None:
        return
    try:
        bus.leaf_add(parent_id, label, ttl_ms)
    except Exception as exc:
        _log(f"leaf_add failed: {exc}")


def node_fade(node_id: str, duration_ms: int = 1500) -> None:
    bus = _bus
    if bus is None:
        return
    try:
        bus.node_fade(node_id, duration_ms)
    except Exception as exc:
        _log(f"node_fade failed: {exc}")


def project_added(payload: dict) -> None:
    """Convenience: push the discrete projectAdded bridge signal so the
    iris simulator splats the new project node into its scene graph.

    Symmetric to ``project_remove`` — best-effort, never raises, no-op
    when no cortex window / bridge is active. ``payload`` should carry
    ``project_id``, ``label``, ``root_path`` and (optionally) ``color``
    — the same shape WorldState's ``project_added`` observer emits, so
    the JS handler sees a uniform schema regardless of source.

    Typically redundant when WorldState.add_project() runs (its
    observer chain already calls bridge.emit_project_added), but kept
    so direct callers (tests, future bus-only mode) can still light
    up the live viz.
    """
    if not isinstance(payload, dict):
        return
    try:
        from .cortex.bridge import get_active_bridge
        bridge = get_active_bridge()
        if bridge is not None:
            emitter = getattr(bridge, "emit_project_added", None)
            if callable(emitter):
                emitter(payload)
    except Exception as exc:
        _log(f"project_added bridge emit failed: {exc}")


def project_remove(project_id: str, duration_ms: int = 2000) -> None:
    """Convenience: fade out a project node and (best-effort) push the
    discrete projectRemoved bridge signal so the iris simulator can
    drop the project from its scene.

    Symmetric to ``leaf_add`` / ``node_fade`` — best-effort, never
    raises, no-op when no cortex window/bridge is active.
    """
    # Decorative fade in the bus-style scene.
    bus = _bus
    if bus is not None:
        try:
            bus.node_fade(project_id, duration_ms)
        except Exception as exc:
            _log(f"project_remove fade failed: {exc}")
    # Discrete bridge signal (powers the iris_simulator JS scene).
    # Log path taken so silent fade failures are debuggable: the prior
    # version no-op'd quietly when no bridge was registered and the
    # user would never know whether (a) the simulator never registered,
    # (b) the id never landed, or (c) JS dropped it.
    try:
        from .cortex.bridge import get_active_bridge
        bridge = get_active_bridge()
        if bridge is None:
            _log(
                f"project_remove({project_id!r}): no active bridge — "
                "fade signal not emitted (simulator window likely closed)"
            )
            return
        emitter = getattr(bridge, "emit_project_removed", None)
        if not callable(emitter):
            _log(
                f"project_remove({project_id!r}): bridge has no "
                "emit_project_removed — fade signal not emitted"
            )
            return
        emitter({"project_id": project_id})
        _log(f"project_remove({project_id!r}): emitted to bridge")
    except Exception as exc:
        _log(f"project_remove bridge emit failed: {exc}")


# ---- convenience helpers for higher-level subsystems ----

def tool_call(tool_name: str, source: str = "iris") -> None:
    """Convenience: light up the tool capability node + pulse from core.

    Iris subsystems call this when a tool dispatches so we don't make
    them choose category/intensity. Maps to the capability node IDs
    seeded by the cortex web layer (cap-tools)."""
    edge_pulse("core", "cap-tools", color="amber", duration_ms=110)
    node_activity("cap-tools", intensity=0.9, duration_ms=150)


def memory_retrieve(label: Optional[str] = None) -> None:
    """Convenience: pulse memory → core and optionally add a leaf for
    what was retrieved. Used by the memory manager."""
    edge_pulse("cap-memory", "core", color="magenta", duration_ms=110)
    node_activity("cap-memory", intensity=0.85, duration_ms=125)
    if label:
        # Drop the retrieved-memory snippet as a short-lived leaf so
        # users can see "what Iris just looked at."
        leaf_add("cap-memory", label, ttl_ms=6000)


def realtime_state(state: str, intensity: float = 0.7) -> None:
    """Convenience: combine core.state with a Realtime-capability pulse."""
    core_state(state, intensity=intensity)
    if state in ("listening", "thinking", "speaking", "retrieving"):
        node_activity("cap-realtime", intensity=intensity, duration_ms=500)


def voice_audio(rms: float) -> None:
    """Convenience: drive core audio + light the Voice capability node."""
    core_audio(rms)
    if rms > 0.04:
        node_activity("cap-voice", intensity=min(1.0, rms * 4.0), duration_ms=250)


# ---- world-touch (persistent) -----------------------------------------
# Touch() is the general-purpose entry point for "Iris just interacted
# with X". It does two things:
#
#   1) records the touch into the persistent world state (cortex_world.json)
#   2) emits a live visualization event so the cortex viz reflects it
#
# Importing world_state is lazy: the import (which constructs the
# WorldState singleton on first use) is deferred until the first
# touch() call, so cortex_emit stays cheap-to-import for code paths
# that never push anything.

_world = None  # type: Optional[object]
_world_import_failed = False


def _get_world():
    """Return the WorldState singleton, importing lazily. Returns None
    on import failure so touch() degrades gracefully."""
    global _world, _world_import_failed
    if _world is not None:
        return _world
    if _world_import_failed:
        return None
    try:
        from .cortex.world_state import get_world as _gw
        _world = _gw()
    except Exception as exc:
        _world_import_failed = True
        _log(f"world_state import failed: {exc}")
        return None
    return _world


def touch(
    kind: str,
    label: str,
    *,
    path: Optional[str] = None,
    project_hint: Optional[str] = None,
    parent_id: Optional[str] = None,
    ttl_ms: Optional[int] = None,
) -> Optional[str]:
    """Record that Iris just interacted with something.

    ``kind`` is one of ``"file" | "app" | "project" | "url" | "note"``.
    ``label`` is what the user sees on the leaf/node.
    ``path`` is an absolute path (when applicable) used for both
    auto-project-detection in the world state AND for the eventual
    "double-click leaf opens file" wiring.
    ``project_hint`` lets callers override auto-detection with a known
    project id.
    ``parent_id`` overrides which node the leaf attaches to. Default
    routing:
      - file/url/note → 'cap-tools' (or detected project if path-based)
      - app           → 'cap-tools'
      - project       → core (parent_id=None)
    ``ttl_ms`` overrides the leaf's auto-fade time (only relevant
    when the touch produces a leaf, not a project).

    Returns the project_id the touch was attached to, when applicable.
    """
    world = _get_world()
    project_id: Optional[str] = None

    # ---- 1) persistent world update ----
    if world is not None:
        try:
            if kind == "file" and path:
                project_id = world.touch_file(path)
            elif kind == "app":
                world.touch_app(label)
            elif kind == "project":
                pid = project_hint or label
                world.touch_project(pid)
                project_id = pid
            elif kind in ("url", "note"):
                # No persistent log yet for raw URLs/notes; future:
                # tag them under the active project context.
                pass
        except Exception as exc:
            _log(f"touch persistence failed ({kind}/{label}): {exc}")

    # ---- 2) live visualization update ----
    bus = _bus
    if bus is None:
        return project_id
    try:
        if kind == "project":
            # New project on first touch — spawn at the core.
            pid = project_hint or label
            bus.node_spawn(pid, label, category="project", weight=0.6)
            bus.edge_pulse("core", pid, color="cyan", duration_ms=600)
        elif kind == "file":
            # Leaf attached to either the detected project (if known)
            # or cap-tools (default catch-all).
            target_parent = parent_id or project_id or "cap-tools"
            bus.leaf_add(target_parent, label, ttl_ms=ttl_ms or 8000)
            bus.edge_pulse("core", target_parent, color="amber", duration_ms=380)
            bus.node_activity(target_parent, intensity=0.85, duration_ms=500)
        elif kind == "app":
            bus.leaf_add(parent_id or "cap-tools", f"📂 {label}", ttl_ms=ttl_ms or 6000)
            bus.node_activity("cap-tools", intensity=0.8, duration_ms=500)
        elif kind in ("url", "note"):
            bus.leaf_add(parent_id or "cap-tools", label, ttl_ms=ttl_ms or 6000)
    except Exception as exc:
        _log(f"touch viz emit failed ({kind}/{label}): {exc}")

    return project_id

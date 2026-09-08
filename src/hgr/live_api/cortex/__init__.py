"""Iris Cortex — 3D cinematic visualization of Iris's live thinking.

A modeless QMainWindow hosting a QWebEngineView. The browser-side
renderer (Three.js) draws a glowing central core with capability
nodes in fixed orbit and a dynamic outer ring of active-context
nodes spawned by events from the Iris subsystems.

Public surface:
  - CortexWindow:  the modeless visualization window
  - CortexBridge:  QObject exposed to JS via QWebChannel
  - CortexEventBus: throttle/coalesce gateway Iris subsystems push to

See docs/IRIS_VISUALIZATION.md for the full spec.

Author: Konstantin Markov
"""

from .bridge import CortexBridge
from .event_bus import CortexEventBus
from .window import CortexWindow

__all__ = ["CortexBridge", "CortexEventBus", "CortexWindow"]

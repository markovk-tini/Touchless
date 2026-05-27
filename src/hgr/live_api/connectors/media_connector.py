"""Media connector — the API-first path for the global media key.

Wraps MediaController, which sends the OS media play/pause virtual key.
This is the universal toggle that controls whatever app currently owns
media (Spotify, a browser tab, a player). Prefer it over hunting for and
clicking a play button on screen when the target is just "whatever is
playing right now".

Note: this overlaps the built-in `press_hotkey` tool, but a named tool
is cheaper for the model to choose correctly than reasoning out the
right virtual-key chord.

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Connector, connector_result


class MediaConnector(Connector):
    id = "media"

    def __init__(self, controller: Optional[Any] = None) -> None:
        self._controller = controller

    def _ctrl(self):
        if self._controller is None:
            from ...debug.media_controller import MediaController
            self._controller = MediaController()
        return self._controller

    def available(self) -> bool:
        try:
            return bool(getattr(self._ctrl(), "available", False))
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        return [{
            "type": "function",
            "name": "media_play_pause",
            "description": (
                "Send the global media play/pause key — toggles whatever app "
                "currently owns media playback. Use when the user says "
                "'play'/'pause'/'stop' without naming a specific app."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "media_play_pause":
            return connector_result("ok", sent=bool(self._ctrl().play_pause()))
        return connector_result("error", error=f"unknown media tool: {name}", code="no_handler")

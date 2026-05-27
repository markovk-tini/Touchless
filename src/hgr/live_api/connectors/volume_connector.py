"""Volume connector — the API-first path for system audio.

Wraps the existing VolumeController (pycaw / Core Audio). When this
connector is available, iris sets the system volume / mute with one
deterministic call instead of nudging the volume keys or clicking the
tray mixer — no screenshots, no guessing the current level.

VolumeController works with a 0..1 scalar internally; the model speaks
in 0-100 percent, so we convert at the boundary.

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Connector, connector_result


class VolumeConnector(Connector):
    id = "volume"

    def __init__(self, controller: Optional[Any] = None) -> None:
        self._controller = controller

    def _ctrl(self):
        if self._controller is None:
            from ...debug.volume_controller import VolumeController
            self._controller = VolumeController()
        return self._controller

    def available(self) -> bool:
        try:
            return bool(getattr(self._ctrl(), "available", False))
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        def fn(name: str, desc: str, props: Dict[str, Any] | None = None,
               required: List[str] | None = None) -> Dict[str, Any]:
            return {
                "type": "function",
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props or {},
                    "required": required or [],
                    "additionalProperties": False,
                },
            }

        return [
            fn("volume_get",
               "Get the current system volume (0-100 percent) and mute state. "
               "Prefer this over reading the volume from a screenshot."),
            fn("volume_set",
               "Set the system volume to a percent (0-100). Prefer this over "
               "pressing the volume keys repeatedly.",
               {"percent": {"type": "integer", "description": "Volume 0-100."}},
               ["percent"]),
            fn("volume_mute",
               "Mute or unmute the system audio.",
               {"muted": {"type": "boolean",
                          "description": "True to mute, False to unmute."}},
               ["muted"]),
            fn("volume_toggle_mute", "Toggle the system mute state."),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        c = self._ctrl()
        if name == "volume_get":
            level = c.get_level()
            mute = c.get_mute()
            return connector_result(
                "ok",
                percent=(round(level * 100) if isinstance(level, (int, float)) else None),
                muted=mute,
            )
        if name == "volume_set":
            try:
                pct = int(args.get("percent"))
            except (TypeError, ValueError):
                return connector_result("error", error="percent must be an integer 0-100")
            pct = max(0, min(100, pct))
            return connector_result("ok" if c.set_level(pct / 100.0) else "error", percent=pct)
        if name == "volume_mute":
            muted = bool(args.get("muted"))
            return connector_result("ok" if c.set_mute(muted) else "error", muted=muted)
        if name == "volume_toggle_mute":
            result = c.toggle_mute()
            return connector_result("ok" if result is not None else "error", muted=result)
        return connector_result("error", error=f"unknown volume tool: {name}", code="no_handler")

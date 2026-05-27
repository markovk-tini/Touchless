"""YouTube connector — the API-first path for YouTube playback control.

Wraps the existing YouTubeController, which drives the live YouTube tab
via CDP / injected JS and keyboard shortcuts. When a YouTube tab is open
iris controls playback with one deterministic call instead of locating
and clicking on-page controls (which move, hide on idle, and break OCR).

Controller sharing: the ToolExecutor already lazily owns a
YouTubeController (for `skip_youtube_ad`). We reuse *that* instance via
`executor._ensure_youtube()` so there's exactly one controller — a second
one would spin up a duplicate ad-watcher. If no executor is supplied
(e.g. unit tests) we lazy-create our own.

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Connector, connector_result


class YouTubeConnector(Connector):
    id = "youtube"

    def __init__(self, executor: Optional[Any] = None,
                 controller: Optional[Any] = None) -> None:
        self._executor = executor
        self._own = controller

    def _ctrl(self):
        # Prefer the executor's shared, lazily-created controller.
        if self._executor is not None:
            try:
                ctrl = self._executor._ensure_youtube()
                if ctrl is not None:
                    return ctrl
            except Exception:
                pass
        if self._own is None:
            from ...debug.youtube_controller import YouTubeController
            self._own = YouTubeController()
        return self._own

    def available(self) -> bool:
        try:
            return bool(self._ctrl().has_youtube_tab())
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
            fn("youtube_play_pause",
               "Toggle YouTube playback in the open YouTube tab. Prefer this "
               "over clicking the video."),
            fn("youtube_next", "Skip to the next YouTube video / item in the playlist."),
            fn("youtube_previous", "Go to the previous YouTube video."),
            fn("youtube_seek",
               "Seek the YouTube video forward or backward by the short step.",
               {"direction": {"type": "string", "enum": ["forward", "backward"]}},
               ["direction"]),
            fn("youtube_set_volume",
               "Set the YouTube player volume to a percent (0-100).",
               {"percent": {"type": "integer", "description": "Volume 0-100."}},
               ["percent"]),
            fn("youtube_speed",
               "Increase or decrease the YouTube playback speed by one step.",
               {"direction": {"type": "string", "enum": ["up", "down"]}},
               ["direction"]),
            fn("youtube_fullscreen", "Toggle YouTube fullscreen."),
            fn("youtube_captions", "Toggle YouTube captions (subtitles)."),
            fn("youtube_like", "Like the current YouTube video."),
            fn("youtube_play",
               "Search YouTube for a query and play the first result.",
               {"query": {"type": "string", "description": "What to search and play."}},
               ["query"]),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        c = self._ctrl()
        if name == "youtube_play_pause":
            return connector_result("ok", changed=bool(c.toggle_playback()))
        if name == "youtube_next":
            return connector_result("ok", skipped=bool(c.next_track()))
        if name == "youtube_previous":
            return connector_result("ok", went_back=bool(c.previous_track()))
        if name == "youtube_seek":
            direction = str(args.get("direction") or "").strip().lower()
            if direction == "forward":
                return connector_result("ok", sought=bool(c.seek_forward()))
            if direction == "backward":
                return connector_result("ok", sought=bool(c.seek_backward()))
            return connector_result("error", error="direction must be 'forward' or 'backward'")
        if name == "youtube_set_volume":
            try:
                pct = int(args.get("percent"))
            except (TypeError, ValueError):
                return connector_result("error", error="percent must be an integer 0-100")
            pct = max(0, min(100, pct))
            return connector_result("ok" if c.set_volume(pct / 100.0) else "error", percent=pct)
        if name == "youtube_speed":
            direction = str(args.get("direction") or "").strip().lower()
            if direction == "up":
                return connector_result("ok", changed=bool(c.speed_up()))
            if direction == "down":
                return connector_result("ok", changed=bool(c.speed_down()))
            return connector_result("error", error="direction must be 'up' or 'down'")
        if name == "youtube_fullscreen":
            return connector_result("ok", toggled=bool(c.toggle_fullscreen()))
        if name == "youtube_captions":
            return connector_result("ok", toggled=bool(c.toggle_captions()))
        if name == "youtube_like":
            return connector_result("ok", liked=bool(c.like_video()))
        if name == "youtube_play":
            query = str(args.get("query") or "").strip()
            if not query:
                return connector_result("error", error="query is required")
            ok = bool(c.play_first_search_result(query))
            return connector_result("ok" if ok else "error", query=query, started=ok)
        return connector_result("error", error=f"unknown youtube tool: {name}", code="no_handler")

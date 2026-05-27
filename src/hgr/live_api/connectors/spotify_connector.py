"""Spotify connector — the API-first path for Spotify control.

Wraps the existing SpotifyController (Spotify Web API). When this
connector is available (Spotify is set up + authorized), iris drives
Spotify with single deterministic API calls instead of clicking the
Spotify UI: one round-trip, no screenshots/OCR, no broken clicks when
the layout changes — and the Realtime session spends far less time
(= less cost / concurrency) than a GUI dance would.

If Spotify isn't authorized, available() returns False, these tools are
not exposed, and iris falls back to GUI computer-use like before.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Connector, connector_result


class SpotifyConnector(Connector):
    id = "spotify"

    def __init__(self, controller: Optional[Any] = None) -> None:
        # Prefer an injected, app-shared controller; otherwise lazy-create
        # one (reads the same stored tokens). Sharing avoids duplicate auth
        # state in production — pass the app's controller when wiring this.
        self._controller = controller

    def _ctrl(self):
        if self._controller is None:
            from ...debug.spotify_controller import SpotifyController
            self._controller = SpotifyController()
        return self._controller

    def available(self) -> bool:
        try:
            c = self._ctrl()
            # `available` and `has_authorization` are both @property — read
            # them, don't call them (calling a bool raises TypeError, which
            # would make this connector silently never activate).
            return bool(getattr(c, "available", False) and getattr(c, "has_authorization", False))
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
            fn("spotify_play_pause",
               "Toggle Spotify playback (play if paused, pause if playing). "
               "Prefer this over clicking the Spotify window."),
            fn("spotify_play",
               "Start Spotify playback. With 'query', search for and play that "
               "song/artist/album; without it, resume the current track.",
               {"query": {"type": "string",
                          "description": "Optional: what to play, e.g. 'Sade'."}}),
            fn("spotify_pause", "Pause Spotify playback."),
            fn("spotify_next", "Skip to the next track in Spotify."),
            fn("spotify_previous", "Go to the previous track in Spotify."),
            fn("spotify_set_volume",
               "Set Spotify volume to a percent (0-100).",
               {"percent": {"type": "integer",
                            "description": "Volume 0-100."}},
               ["percent"]),
            fn("spotify_now_playing",
               "Get the currently playing Spotify track (title + artist)."),
            fn("spotify_add_to_playlist",
               "Add the currently playing track to a named playlist.",
               {"playlist_name": {"type": "string"}},
               ["playlist_name"]),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        c = self._ctrl()
        if name == "spotify_play_pause":
            return connector_result("ok", changed=bool(c.toggle_playback()))
        if name == "spotify_play":
            query = str(args.get("query") or "").strip()
            if query:
                ok = bool(c.play_search_request(query))
                return connector_result("ok" if ok else "error", query=query, started=ok)
            return connector_result("ok", resumed=bool(c.play()))
        if name == "spotify_pause":
            return connector_result("ok", paused=bool(c.pause()))
        if name == "spotify_next":
            return connector_result("ok", skipped=bool(c.next_track()))
        if name == "spotify_previous":
            return connector_result("ok", went_back=bool(c.previous_track()))
        if name == "spotify_set_volume":
            try:
                pct = int(args.get("percent"))
            except (TypeError, ValueError):
                return connector_result("error", error="percent must be an integer 0-100")
            pct = max(0, min(100, pct))
            return connector_result("ok" if c.set_volume(pct) else "error", percent=pct)
        if name == "spotify_now_playing":
            details = c.get_current_track_details()
            if not details:
                return connector_result("ok", playing=False)
            return connector_result(
                "ok", playing=True,
                title=getattr(details, "title", None),
                artist=getattr(details, "artist", None),
            )
        if name == "spotify_add_to_playlist":
            playlist = str(args.get("playlist_name") or "").strip()
            if not playlist:
                return connector_result("error", error="playlist_name is required")
            ok = bool(c.add_current_track_to_playlist(playlist))
            return connector_result("ok" if ok else "error", playlist=playlist, added=ok)
        return connector_result("error", error=f"unknown spotify tool: {name}", code="no_handler")

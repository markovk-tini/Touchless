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
        def fn(name: str, desc: str, props=None, required=None):
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
            fn("media_play_pause",
               "BARE TOGGLE — send the OS media play/pause key. Use "
               "ONLY when the user said an unqualified 'pause' / "
               "'resume' / 'play' (with NO song/artist/playlist "
               "after) AND you want to flip whatever's already "
               "loaded. For 'play <something>' (a song, artist, "
               "playlist, or album), use spotify_play(query=...) "
               "INSTEAD — this toggle won't search; it just flips "
               "the current track's play/pause state."),
            fn("media_next_track",
               "Skip to the next track in whatever app currently owns "
               "media (Spotify, YouTube Music, etc.). Uses the OS "
               "global media key — works regardless of focus."),
            fn("media_previous_track",
               "Go back to the previous track in whatever app currently "
               "owns media playback."),
            fn("media_now_playing",
               "Return what's currently playing: {title, artist, album, "
               "app, playback_status} from the Windows System Media "
               "Transport Controls (SMTC). Works for any app that "
               "registers with SMTC — Spotify, Edge / Chrome media "
               "tabs, Groove, browser-based YouTube, etc. Use to "
               "answer 'what's playing?' or to verify before doing "
               "media_next_track."),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "media_play_pause":
            return connector_result("ok", sent=bool(self._ctrl().play_pause()))
        if name == "media_next_track":
            ok = _send_media_key("next")
            return connector_result("ok" if ok else "error", sent=ok)
        if name == "media_previous_track":
            ok = _send_media_key("previous")
            return connector_result("ok" if ok else "error", sent=ok)
        if name == "media_now_playing":
            data, err = _smtc_now_playing()
            if err:
                return connector_result("error", error=err,
                                        code="smtc_unavailable")
            if data is None:
                return connector_result(
                    "ok", playing=False,
                    note="Nothing is currently registered with the "
                         "Windows media transport.")
            return connector_result("ok", playing=True, **data)
        return connector_result("error", error=f"unknown media tool: {name}", code="no_handler")


# ---- helpers --------------------------------------------------------------


def _send_media_key(action: str) -> bool:
    """Send a virtual-key media event. Falls back to None when pywin32
    isn't available — caller surfaces an error."""
    try:
        import win32api  # type: ignore
        import win32con  # type: ignore
    except Exception:
        return False
    vk_map = {
        "play_pause": 0xB3,   # VK_MEDIA_PLAY_PAUSE
        "next": 0xB0,         # VK_MEDIA_NEXT_TRACK
        "previous": 0xB1,     # VK_MEDIA_PREV_TRACK
        "stop": 0xB2,         # VK_MEDIA_STOP
    }
    vk = vk_map.get(action)
    if vk is None:
        return False
    try:
        win32api.keybd_event(vk, 0, 0, 0)
        win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
        return True
    except Exception:
        return False


def _smtc_now_playing():
    """Returns ({title, artist, album, app, playback_status}, err) or
    (None, None) when no session is registered. (None, err) on backend
    failure (winsdk / winrt missing)."""
    try:
        # winsdk is the bundled name; some installs ship winrt.
        try:
            from winsdk.windows.media.control import (  # type: ignore
                GlobalSystemMediaTransportControlsSessionManager as _Mgr,
            )
        except Exception:
            from winrt.windows.media.control import (  # type: ignore
                GlobalSystemMediaTransportControlsSessionManager as _Mgr,
            )
    except Exception:
        return None, ("SMTC backend missing — pip install winsdk")
    import asyncio
    try:
        # The SMTC API is async; run a tiny loop synchronously.
        async def _query():
            mgr = await _Mgr.request_async()
            session = mgr.get_current_session()
            if session is None:
                return None
            props = await session.try_get_media_properties_async()
            playback = session.get_playback_info()
            status_int = int(playback.playback_status)
            status_names = {
                0: "closed", 1: "opened", 2: "changing",
                3: "stopped", 4: "playing", 5: "paused",
            }
            return {
                "title": (props.title or "").strip(),
                "artist": (props.artist or "").strip(),
                "album": (props.album_title or "").strip(),
                "app": (session.source_app_user_model_id or "").strip(),
                "playback_status": status_names.get(status_int, str(status_int)),
            }
        # Use a private loop — we may be called from any thread.
        try:
            loop = asyncio.new_event_loop()
            try:
                data = loop.run_until_complete(_query())
            finally:
                loop.close()
        except Exception as exc:
            return None, f"SMTC query failed: {type(exc).__name__}: {exc}"
        return data, None
    except Exception as exc:
        return None, f"SMTC error: {type(exc).__name__}: {exc}"

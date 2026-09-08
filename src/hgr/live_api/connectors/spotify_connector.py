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

    def __init__(self, controller: Optional[Any] = None, *,
                 setup_only: bool = False) -> None:
        # Prefer an injected, app-shared controller; otherwise lazy-create
        # one (reads the same stored tokens). Sharing avoids duplicate auth
        # state in production — pass the app's controller when wiring this.
        self._controller = controller
        # When True, expose ONLY setup_self() — tools() returns [] so the
        # 8 playback verbs stay hidden and the Layer-0 Touchless router
        # keeps owning Spotify control. Keeps the OAuth-setup pathway open
        # without re-introducing the playlist-parsing regression that got
        # this connector un-registered originally. Flip to False the day
        # the Web-API playlist parsing is reliable.
        self._setup_only = bool(setup_only)

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

    def setup_self(self) -> Dict[str, Any]:
        """Walk the Spotify PKCE OAuth flow so 'set up spotify' works from
        Iris. Returns the kicad-style {ok, ...} dict the orchestrator
        surfaces to the user. Idempotent — if tokens are already valid,
        returns ok with already_authorized=True so the model can tell the
        user it's already connected."""
        try:
            c = self._ctrl()
        except Exception as exc:
            return {"ok": False,
                    "error": f"Couldn't initialise Spotify controller: "
                             f"{type(exc).__name__}: {exc}"}
        if not bool(getattr(c, "available", False)):
            return {"ok": False,
                    "error": "Spotify Web API support isn't available in "
                             "this build (the controller didn't initialise)."}
        if bool(getattr(c, "has_authorization", False)):
            return {"ok": True, "already_authorized": True,
                    "message": "Spotify is already connected."}
        # Opens the browser, spins a local redirect listener, exchanges the
        # code for tokens via PKCE, saves them. Blocks up to ~3 minutes for
        # the user to approve. Returns True on success.
        ok = False
        try:
            ok = bool(c.authorize_full_scopes())
        except Exception as exc:
            return {"ok": False,
                    "error": f"Spotify authorization failed: "
                             f"{type(exc).__name__}: {exc}"}
        if ok and bool(getattr(c, "has_authorization", False)):
            return {"ok": True,
                    "message": ("Spotify is connected. You can now ask "
                                "me to play music, change tracks, etc.")}
        msg = (getattr(c, "message", None) or "").strip()
        return {"ok": False,
                "error": msg or ("Spotify authorization didn't complete "
                                 "(closed the browser, denied access, or "
                                 "the redirect couldn't bind a localhost "
                                 "port). Try 'set up spotify' again.")}

    def tools(self) -> List[Dict[str, Any]]:
        # Setup-only mode: hide the playback verbs so Layer-0 keeps owning
        # them. setup_self() is still reachable via the connector instance.
        if self._setup_only:
            return []
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
                # Capture BEFORE-state so we can tell if Spotify
                # actually swapped tracks (vs. polling too early and
                # returning the old track's title).
                before_title = None
                try:
                    before = c.get_current_track_details()
                    if before is not None:
                        # SpotifyTrackDetails uses song_name/artist_names
                        # field names, NOT title/artist. Earlier code
                        # used getattr(..., "title", None) which always
                        # returned None and silently broke the track-
                        # change confirmation — every reply fell back
                        # to the empty-subject persona variant.
                        before_title = getattr(before, "song_name", None)
                except Exception:
                    before_title = None
                api_ok = bool(c.play_search_request(query))
                # Spotify Web API returns 202 (accepted) even when no
                # device is active — nothing actually plays. Poll for
                # an ACTUAL track change to confirm. If nothing comes
                # up within ~4s, treat as failure so the user gets
                # an honest error instead of a fake "playing now".
                title = artist = None
                actually_playing = False
                if api_ok:
                    import time as _t
                    # Wait past Spotify's track-transition window
                    # (~1.5-2.5s) then double-read for stability.
                    # Polling and breaking on first different title
                    # catches intermediate "queued recommendation"
                    # state ("Kokomo instead of Poker Face" bug).
                    _t.sleep(2.5)
                    first = None
                    second = None
                    try:
                        first = c.get_current_track_details()
                    except Exception:
                        first = None
                    _t.sleep(0.4)
                    try:
                        second = c.get_current_track_details()
                    except Exception:
                        second = None
                    first_title = (getattr(first, "song_name", None)
                                    if first else None)
                    second_title = (getattr(second, "song_name", None)
                                     if second else None)
                    chosen = None
                    if (first_title and second_title
                            and first_title == second_title):
                        chosen = second
                    elif second_title:
                        chosen = second
                    elif first_title:
                        chosen = first
                    if chosen is not None:
                        new_title = getattr(chosen, "song_name",
                                             None)
                        new_artist = getattr(chosen, "artist_names",
                                              None)
                        if new_title and (
                                before_title is None
                                or new_title != before_title):
                            title = new_title
                            artist = new_artist
                            actually_playing = True
                # Three outcomes:
                #   1. Track CHANGED during poll → confident success
                #      with title/artist for the reply layer.
                #   2. API accepted but no change seen in 1.2s →
                #      could be same-song re-request OR Spotify was
                #      closed. Report ok (without title) — the
                #      persona reply uses the query as fallback.
                #   3. API outright rejected → error.
                if actually_playing:
                    return connector_result(
                        "ok", query=query, started=True,
                        now_playing_title=title,
                        now_playing_artist=artist)
                if api_ok:
                    return connector_result(
                        "ok", query=query, started=True)
                # api_ok=False — propagate the controller's specific
                # message so the user sees a useful reason
                # ("Spotify isn't available — make sure Spotify is
                # installed and logged in", "spotify could not find
                # 'X'", etc.) instead of a generic failure.
                #
                # SANITIZED: c.message can include unbounded user
                # input (playlist names, search queries) and raw
                # exception strings that may leak URLs / token
                # fragments / credentials. We:
                #   - cap length (160 chars)
                #   - collapse whitespace + strip newlines
                #   - scrub URLs (https?://…) — they shouldn't be
                #     read aloud by TTS anyway
                #   - fall back to a generic if scrubbing leaves
                #     nothing useful
                import re as _re
                raw = (getattr(c, "message", "") or "").strip()
                if raw:
                    clean = _re.sub(r"\s+", " ", raw)
                    clean = _re.sub(
                        r"https?://\S+", "[link]", clean)
                    # Drop common credential-shaped tokens just in
                    # case (Spotify access tokens, JWTs, etc.).
                    clean = _re.sub(
                        r"\b[A-Za-z0-9_\-]{32,}\b",
                        "[token]", clean)
                    clean = clean[:160].rstrip()
                    err_text = clean or "spotify play request failed"
                else:
                    err_text = "spotify play request failed"
                return connector_result(
                    "error", query=query, started=False,
                    error=err_text)
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
                title=getattr(details, "song_name", None),
                artist=getattr(details, "artist_names", None),
            )
        if name == "spotify_add_to_playlist":
            playlist = str(args.get("playlist_name") or "").strip()
            if not playlist:
                return connector_result("error", error="playlist_name is required")
            ok = bool(c.add_current_track_to_playlist(playlist))
            return connector_result("ok" if ok else "error", playlist=playlist, added=ok)
        return connector_result("error", error=f"unknown spotify tool: {name}", code="no_handler")

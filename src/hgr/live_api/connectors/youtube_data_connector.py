"""YouTube connector — API-first readonly access to playlists, items, subs.

"What are my playlists", "what's in my watch later", "who am I subscribed
to on youtube" as single API calls. Dormant until the shared GoogleClient
is authorized with the youtube.readonly scope.

Watch Later ('WL') and Liked Videos ('LL') are not accessible to third-
party OAuth apps via the YouTube Data API v3 — connector returns a
friendly fallback URL instead of a bare 404.

Author: Konstantin Markov
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List

from .base import Connector, connector_result, friendly_api_error
from .google_client import GoogleClient

_YOUTUBE_CALL_TIMEOUT_SEC = 25.0
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="youtube")


class YouTubeDataConnector(Connector):
    id = "youtube_data"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _svc(self):
        return self._client.service("youtube", "v3")

    def available(self) -> bool:
        try:
            return self._client.ready()
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "youtube_my_playlists",
                "description": (
                    "List the signed-in user's own YouTube playlists "
                    "(title, id, video count, privacy). Use for 'what are "
                    "my playlists', 'list my YouTube playlists'. Does NOT "
                    "include Watch Later or Liked Videos — those system "
                    "lists are not exposed by the API for OAuth users and "
                    "must be opened via web URL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max": {"type": "integer",
                                "description": "Max playlists (default 25, max 50)."},
                        "page_token": {"type": "string",
                                       "description": "nextPageToken from a prior call."},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "youtube_playlist_items",
                "description": (
                    "List videos inside a YouTube playlist by id. Use "
                    "after youtube_my_playlists to fetch a playlist's "
                    "contents. Also accepts 'WL' (Watch Later) and 'LL' "
                    "(Liked Videos) — those typically 404 for third-party "
                    "OAuth and the connector returns a fallback URL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "playlist_id": {"type": "string",
                                        "description": "YouTube playlist id, or 'WL'/'LL'."},
                        "max": {"type": "integer",
                                "description": "Max items (default 25, max 50)."},
                        "page_token": {"type": "string",
                                       "description": "nextPageToken from a prior call."},
                    },
                    "required": ["playlist_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "youtube_subscriptions",
                "description": (
                    "List channels the signed-in user is subscribed to on "
                    "YouTube. Use for 'who am I subscribed to on YouTube', "
                    "'list my YouTube subscriptions', 'what channels do I "
                    "follow'. Returns channel title, id, and a link."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max": {"type": "integer",
                                "description": "Max subscriptions (default 25, max 50)."},
                        "page_token": {"type": "string",
                                       "description": "nextPageToken from a prior call."},
                        "order": {"type": "string",
                                  "description": "'alphabetical' (default), 'relevance', or 'unread'."},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _pool.submit(self._execute_locked, name, args).result(
                timeout=_YOUTUBE_CALL_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            return connector_result(
                "error",
                error=f"{name} timed out after {_YOUTUBE_CALL_TIMEOUT_SEC}s",
                code="timeout",
            )

    def _execute_locked(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="YouTube not authorized",
                                    code="not_ready")
        if not self._client.has_scope(
                "https://www.googleapis.com/auth/youtube.readonly"):
            return self._client.scope_missing_result(
                "https://www.googleapis.com/auth/youtube.readonly",
                friendly="YouTube")
        try:
            from googleapiclient.errors import HttpError
        except Exception:
            HttpError = Exception  # type: ignore

        if name == "youtube_my_playlists":
            max_n = max(1, min(50, int(args.get("max") or 25)))
            page_token = str(args.get("page_token") or "").strip() or None
            try:
                kwargs = {"part": "snippet,contentDetails,status",
                          "mine": True, "maxResults": max_n}
                if page_token:
                    kwargs["pageToken"] = page_token
                resp = svc.playlists().list(**kwargs).execute()
                out = []
                for p in resp.get("items", []) or []:
                    snippet = p.get("snippet", {}) or {}
                    content = p.get("contentDetails", {}) or {}
                    status = p.get("status", {}) or {}
                    pid = p.get("id")
                    out.append({
                        "id": pid,
                        "title": snippet.get("title"),
                        "description": snippet.get("description"),
                        "item_count": content.get("itemCount"),
                        "privacy": status.get("privacyStatus"),
                        "link": (f"https://www.youtube.com/playlist?list={pid}"
                                 if pid else None),
                    })
                return connector_result("ok", count=len(out), playlists=out,
                                        next_page_token=resp.get("nextPageToken"))
            except HttpError as exc:
                return connector_result(
                    "error", error=f"{exc.resp.status}: {exc}", code="api_error")
            except Exception as exc:
                return connector_result(
                    "error", error=friendly_api_error(exc, api_label="YouTube"))

        if name == "youtube_playlist_items":
            playlist_id = str(args.get("playlist_id") or "").strip()
            if not playlist_id:
                return connector_result("error", error="playlist_id is required")
            max_n = max(1, min(50, int(args.get("max") or 25)))
            page_token = str(args.get("page_token") or "").strip() or None
            try:
                kwargs = {"part": "snippet,contentDetails",
                          "playlistId": playlist_id, "maxResults": max_n}
                if page_token:
                    kwargs["pageToken"] = page_token
                resp = svc.playlistItems().list(**kwargs).execute()
                items = []
                for it in resp.get("items", []) or []:
                    snippet = it.get("snippet", {}) or {}
                    content = it.get("contentDetails", {}) or {}
                    video_id = (content.get("videoId")
                                or (snippet.get("resourceId") or {}).get("videoId"))
                    items.append({
                        "title": snippet.get("title"),
                        "video_id": video_id,
                        "channel_title": snippet.get("videoOwnerChannelTitle")
                                          or snippet.get("channelTitle"),
                        "position": snippet.get("position"),
                        "published_at": content.get("videoPublishedAt")
                                         or snippet.get("publishedAt"),
                        "link": (f"https://www.youtube.com/watch?v={video_id}"
                                 f"&list={playlist_id}" if video_id else None),
                    })
                return connector_result("ok", playlist_id=playlist_id,
                                        count=len(items), items=items,
                                        next_page_token=resp.get("nextPageToken"))
            except HttpError as exc:
                status = getattr(getattr(exc, "resp", None), "status", None)
                if status == 404 and playlist_id.upper() in ("WL", "LL"):
                    label = ("Watch Later" if playlist_id.upper() == "WL"
                             else "Liked Videos")
                    fallback = (f"https://www.youtube.com/playlist?list="
                                f"{playlist_id.upper()}")
                    return connector_result(
                        "error",
                        error=(f"{label} is not accessible via the YouTube "
                               f"API for third-party apps. Open {fallback} "
                               f"in browser."),
                        code="watch_later_unavailable",
                        fallback_url=fallback,
                    )
                return connector_result(
                    "error", error=f"{status}: {exc}", code="api_error")
            except Exception as exc:
                return connector_result(
                    "error", error=friendly_api_error(exc, api_label="YouTube"))

        if name == "youtube_subscriptions":
            max_n = max(1, min(50, int(args.get("max") or 25)))
            page_token = str(args.get("page_token") or "").strip() or None
            order = str(args.get("order") or "alphabetical").strip().lower()
            if order not in ("alphabetical", "relevance", "unread"):
                order = "alphabetical"
            try:
                kwargs = {"part": "snippet", "mine": True,
                          "maxResults": max_n, "order": order}
                if page_token:
                    kwargs["pageToken"] = page_token
                resp = svc.subscriptions().list(**kwargs).execute()
                subs = []
                for it in resp.get("items", []) or []:
                    snippet = it.get("snippet", {}) or {}
                    resource = snippet.get("resourceId", {}) or {}
                    channel_id = resource.get("channelId")
                    subs.append({
                        "channel_id": channel_id,
                        "title": snippet.get("title"),
                        "description": snippet.get("description"),
                        "link": (f"https://www.youtube.com/channel/{channel_id}"
                                 if channel_id else None),
                    })
                return connector_result("ok", count=len(subs),
                                        subscriptions=subs,
                                        next_page_token=resp.get("nextPageToken"))
            except HttpError as exc:
                return connector_result(
                    "error", error=f"{exc.resp.status}: {exc}", code="api_error")
            except Exception as exc:
                return connector_result(
                    "error", error=friendly_api_error(exc, api_label="YouTube"))

        return connector_result("error",
                                error=f"unknown youtube tool: {name}",
                                code="no_handler")

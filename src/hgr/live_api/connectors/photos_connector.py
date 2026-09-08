"""Google Photos connector — append-only upload to the user's library.

"Save this screenshot to my Google Photos", "upload C:/path/clip.mp4 to
my photos" as a single two-step call (raw bytes -> upload token ->
batchCreate media item). Dormant until the shared GoogleClient is
authorized with the photoslibrary.appendonly scope.

Append-only by design: this connector can upload + (optionally) attach
to an app-created album — it CANNOT list, read, or delete existing
library items. That keeps the OAuth scope sensitive (not restricted),
so no paid CASA assessment is required to ship.

Author: Konstantin Markov
"""
from __future__ import annotations

import concurrent.futures
import mimetypes
import os
from typing import Any, Dict, List

from .base import Connector, connector_result, friendly_api_error
from .google_client import GoogleClient

_PHOTOS_CALL_TIMEOUT_SEC = 25.0
_PHOTOS_UPLOAD_TIMEOUT_SEC = 60.0
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="photos")


def _timeout_for(name: str) -> float:
    """photos_upload needs headroom for the raw bytes POST + batchCreate;
    other (future) tools keep the tighter 25s watchdog. Error responses
    still come back sub-second because they fail BEFORE the timeout fires."""
    return (_PHOTOS_UPLOAD_TIMEOUT_SEC if name == "photos_upload"
            else _PHOTOS_CALL_TIMEOUT_SEC)


class GooglePhotosConnector(Connector):
    id = "photos"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _svc(self):
        # photoslibrary was removed from the static discovery cache in
        # google-api-python-client (March 2024); static_discovery MUST
        # be False or build() raises UnknownApiNameOrVersion. Bypass
        # GoogleClient.service() to keep that change local to this file.
        creds = self._client._load_creds()
        if creds is None:
            return None
        try:
            from googleapiclient.discovery import build
            try:
                import httplib2
                from google_auth_httplib2 import AuthorizedHttp
                http = AuthorizedHttp(creds, http=httplib2.Http(timeout=20.0))
                return build("photoslibrary", "v1", http=http,
                             cache_discovery=False, static_discovery=False)
            except Exception:
                return build("photoslibrary", "v1", credentials=creds,
                             cache_discovery=False, static_discovery=False)
        except Exception:
            return None

    def available(self) -> bool:
        try:
            return self._client.ready()
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "photos_upload",
                "description": (
                    "Upload a LOCAL image or video file to the user's Google "
                    "Photos library and return the new media item id/link. "
                    "Use for 'save/upload X to my Google Photos', 'put this "
                    "screenshot in my photos'. `path` MUST be an absolute "
                    "local filesystem path — if the user names the file by "
                    "description ('my latest screenshot', 'the newest video "
                    "on my desktop'), FIRST resolve it (drive_list / "
                    "list_files / etc.) and pass the concrete path here. "
                    "APPEND-ONLY: this connector cannot read, list, or "
                    "delete existing library items. Optional `description` "
                    "becomes the media item's caption. Optional `album_id` "
                    "attaches the upload to an existing app-created album — "
                    "omit unless you already have an album id from a prior "
                    "call. Best for images and short clips."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "Absolute path to the local file."},
                        "description": {"type": "string",
                                        "description": "User-visible caption."},
                        "album_id": {"type": "string",
                                     "description": "App-created album id to attach to."},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        timeout = _timeout_for(name)
        try:
            return _pool.submit(self._execute_locked, name, args).result(
                timeout=timeout)
        except concurrent.futures.TimeoutError:
            return connector_result(
                "error",
                error=f"{name} timed out after {timeout}s",
                code="timeout",
            )

    def _execute_locked(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Google Photos not authorized",
                                    code="not_ready")
        if not self._client.has_scope(
                "https://www.googleapis.com/auth/photoslibrary.appendonly"):
            return self._client.scope_missing_result(
                "https://www.googleapis.com/auth/photoslibrary.appendonly",
                friendly="Google Photos")
        try:
            if name == "photos_upload":
                path = str(args.get("path") or "").strip()
                if not path:
                    return connector_result("error", error="path is required")
                if not os.path.isfile(path):
                    return connector_result("error",
                                            error=f"no such file: {path}",
                                            code="not_found")
                basename = os.path.basename(path)
                mime = mimetypes.guess_type(path)[0] or "image/jpeg"

                creds = self._client._load_creds()
                if creds is None or not getattr(creds, "token", None):
                    return connector_result("error",
                                            error="Google Photos not authorized",
                                            code="not_ready")
                try:
                    with open(path, "rb") as fh:
                        data = fh.read()
                except Exception as exc:
                    return connector_result("error",
                                            error=friendly_api_error(exc, api_label="Google Photos"))

                headers = {
                    "Authorization": f"Bearer {creds.token}",
                    "Content-type": "application/octet-stream",
                    "X-Goog-Upload-Content-Type": mime,
                    "X-Goog-Upload-Protocol": "raw",
                    "X-Goog-Upload-File-Name": basename,
                }
                try:
                    import requests
                    resp = requests.post(
                        "https://photoslibrary.googleapis.com/v1/uploads",
                        data=data, headers=headers, timeout=20.0)
                    if resp.status_code != 200 or not (resp.text or "").strip():
                        return connector_result(
                            "error",
                            error=f"upload failed: {resp.status_code}",
                            code="photos_upload_failed",
                        )
                    upload_token = resp.text.strip()
                except Exception as exc:
                    return connector_result("error",
                                            error=friendly_api_error(exc, api_label="Google Photos"),
                                            code="photos_upload_failed")

                description = str(args.get("description") or "").strip()
                album_id = str(args.get("album_id") or "").strip()
                new_item: Dict[str, Any] = {
                    "simpleMediaItem": {"fileName": basename,
                                        "uploadToken": upload_token}
                }
                if description:
                    new_item["description"] = description
                body: Dict[str, Any] = {"newMediaItems": [new_item]}
                if album_id:
                    body["albumId"] = album_id

                created = svc.mediaItems().batchCreate(body=body).execute()
                results = created.get("newMediaItemResults") or []
                if not results:
                    return connector_result("error",
                                            error="no result from batchCreate",
                                            code="photos_create_failed")
                first = results[0]
                status = first.get("status") or {}
                if int(status.get("code") or 0) != 0:
                    return connector_result(
                        "error",
                        error=str(status.get("message") or "batchCreate failed"),
                        code="photos_create_failed",
                    )
                item = first.get("mediaItem") or {}
                return connector_result(
                    "ok", uploaded=True, id=item.get("id"), name=basename,
                    link=item.get("productUrl"), mime=item.get("mimeType"),
                    album_id=album_id or None,
                    summary=f"Uploaded {basename} to Google Photos.",
                )
        except Exception as exc:
            return connector_result("error", error=friendly_api_error(exc, api_label="Google Photos"))
        return connector_result("error", error=f"unknown photos tool: {name}",
                                code="no_handler")

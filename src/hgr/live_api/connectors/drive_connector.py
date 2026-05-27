"""Google Drive connector — upload/list files via the drive.file scope.

`drive.file` is the **narrow** Drive scope: it only touches files this app
creates or that the user explicitly opens with it — never the user's whole
Drive. That keeps it a non-restricted (free-to-verify) scope while still
letting iris "save this file to my Drive" or list what it has created.

Dormant until the shared GoogleClient is authorized.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from .base import Connector, connector_result
from .google_client import GoogleClient


class DriveConnector(Connector):
    id = "drive"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _svc(self):
        return self._client.service("drive", "v3")

    def available(self) -> bool:
        try:
            return self._client.ready()
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "drive_upload",
                "description": ("Upload a LOCAL file to the user's Google Drive and "
                                "return its link. Use for 'save/upload X to my "
                                "Google Drive'. `path` must be an absolute local "
                                "file path — if the user names a file by "
                                "description ('my latest screenshot', 'the newest "
                                "file on my desktop'), FIRST call list_files to "
                                "find its path, then pass that path here. Do NOT "
                                "web-search for it."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "Absolute path to the local file."},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "drive_list",
                "description": ("List files this app has created/opened in the "
                                "user's Google Drive (name + link)."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max": {"type": "integer", "description": "Max files (default 20)."},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Google Drive not authorized", code="not_ready")
        try:
            if name == "drive_upload":
                path = str(args.get("path") or "").strip()
                if not path:
                    return connector_result("error", error="path is required")
                if not os.path.isfile(path):
                    return connector_result("error", error=f"no such file: {path}", code="not_found")
                from googleapiclient.http import MediaFileUpload
                media = MediaFileUpload(path, resumable=False)
                created = svc.files().create(
                    body={"name": os.path.basename(path)},
                    media_body=media,
                    fields="id,name,webViewLink",
                ).execute()
                return connector_result("ok", uploaded=True, id=created.get("id"),
                                        name=created.get("name"), link=created.get("webViewLink"))
            if name == "drive_list":
                max_n = max(1, min(100, int(args.get("max") or 20)))
                listing = svc.files().list(
                    pageSize=max_n, fields="files(id,name,webViewLink)").execute()
                files = [{"id": f.get("id"), "name": f.get("name"), "link": f.get("webViewLink")}
                         for f in (listing.get("files") or [])]
                return connector_result("ok", count=len(files), files=files)
        except Exception as exc:
            return connector_result("error", error=f"{type(exc).__name__}: {exc}")
        return connector_result("error", error=f"unknown drive tool: {name}", code="no_handler")

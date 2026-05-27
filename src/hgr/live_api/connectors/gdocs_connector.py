"""Google Docs connector — API-first document creation.

"Make a Google Doc titled X with this text" as one call (create + insert),
returning the shareable link. Dormant until the shared GoogleClient is
authorized. Editing arbitrary existing docs is left to the GUI/web path
for now — creation is the high-value, low-risk task.

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import Connector, connector_result
from .google_client import GoogleClient


class GoogleDocsConnector(Connector):
    id = "gdocs"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _svc(self):
        return self._client.service("docs", "v1")

    def available(self) -> bool:
        try:
            return self._client.ready()
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        return [{
            "type": "function",
            "name": "gdocs_create",
            "description": ("Create a new Google Doc with a title and optional "
                            "body text. Returns the document link."),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "text": {"type": "string", "description": "Optional body text."},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        }]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name != "gdocs_create":
            return connector_result("error", error=f"unknown gdocs tool: {name}", code="no_handler")
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Google Docs not authorized", code="not_ready")
        title = str(args.get("title") or "").strip()
        if not title:
            return connector_result("error", error="title is required")
        text = str(args.get("text") or "")
        try:
            doc = svc.documents().create(body={"title": title}).execute()
            doc_id = doc.get("documentId")
            if text and doc_id:
                svc.documents().batchUpdate(
                    documentId=doc_id,
                    body={"requests": [
                        {"insertText": {"location": {"index": 1}, "text": text}}
                    ]}).execute()
            link = f"https://docs.google.com/document/d/{doc_id}/edit" if doc_id else None
            return connector_result("ok", created=True, id=doc_id, title=title, link=link)
        except Exception as exc:
            return connector_result("error", error=f"{type(exc).__name__}: {exc}")

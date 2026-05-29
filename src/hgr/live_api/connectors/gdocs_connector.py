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
        return [
            {
                "type": "function",
                "name": "gdocs_create",
                "description": (
                    "Create a new Google Doc with a title and optional "
                    "body text. PREFER passing the body via `text` in this "
                    "single call when possible (1-step path). Returns "
                    "{created, id, title, link}; chain id into "
                    "gdocs_append_text for additional content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "text": {"type": "string",
                                 "description": "Optional body text."},
                    },
                    "required": ["title"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "gdocs_append_text",
                "description": (
                    "Append text to the END of an existing Google Doc. "
                    "Use this when content needs to be added AFTER the doc "
                    "was created (e.g. 'paste the summary I just made into "
                    "the doc I just created'). Get `doc_id` from "
                    "gdocs_create's `id` field via {step:N.id}. Supports "
                    "multi-paragraph plain text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["doc_id", "text"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Google Docs not authorized",
                                    code="not_ready")

        if name == "gdocs_create":
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
                link = (f"https://docs.google.com/document/d/{doc_id}/edit"
                        if doc_id else None)
                return connector_result("ok", created=True, id=doc_id,
                                        title=title, link=link)
            except Exception as exc:
                return connector_result("error",
                                        error=f"{type(exc).__name__}: {exc}")

        if name == "gdocs_append_text":
            doc_id = str(args.get("doc_id") or "").strip()
            text = str(args.get("text") or "")
            if not doc_id:
                return connector_result("error", error="doc_id is required")
            if not text:
                return connector_result("error", error="text is required")
            try:
                # endOfSegmentLocation with no segmentId targets the main body
                # — inserts at the very end of the doc, after all existing
                # content. Newer-than-insertText API but well-supported.
                svc.documents().batchUpdate(
                    documentId=doc_id,
                    body={"requests": [
                        {"insertText": {"endOfSegmentLocation": {},
                                         "text": text}}
                    ]}).execute()
                link = f"https://docs.google.com/document/d/{doc_id}/edit"
                return connector_result("ok", appended=True, id=doc_id,
                                        chars=len(text), link=link)
            except Exception as exc:
                return connector_result("error",
                                        error=f"{type(exc).__name__}: {exc}")

        return connector_result("error",
                                error=f"unknown gdocs tool: {name}",
                                code="no_handler")

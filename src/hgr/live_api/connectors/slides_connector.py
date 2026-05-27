"""Google Slides connector — create presentations via the Slides API.

"Make a slideshow titled …" as one call, returning the link. Optionally
sets the title-slide text. Dormant until the shared GoogleClient is
authorized with the presentations scope.

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import Connector, connector_result
from .google_client import GoogleClient


class GoogleSlidesConnector(Connector):
    id = "gslides"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _svc(self):
        return self._client.service("slides", "v1")

    def available(self) -> bool:
        try:
            return self._client.ready()
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        return [{
            "type": "function",
            "name": "slides_create",
            "description": ("Create a new Google Slides presentation with a title. "
                            "Optionally set the first slide's heading text. Returns "
                            "the presentation link."),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Presentation title."},
                    "heading": {"type": "string",
                                "description": "Optional text for the title slide."},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        }]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name != "slides_create":
            return connector_result("error", error=f"unknown slides tool: {name}", code="no_handler")
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Google Slides not authorized", code="not_ready")
        title = str(args.get("title") or "").strip()
        if not title:
            return connector_result("error", error="title is required")
        try:
            pres = svc.presentations().create(body={"title": title}).execute()
            pid = pres.get("presentationId")
            heading = str(args.get("heading") or "").strip()
            if pid and heading:
                # The default first slide's title placeholder gets the heading.
                slides = pres.get("slides") or []
                title_id = None
                if slides:
                    for el in (slides[0].get("pageElements") or []):
                        shape = el.get("shape") or {}
                        if shape.get("placeholder", {}).get("type") in ("TITLE", "CENTERED_TITLE"):
                            title_id = el.get("objectId")
                            break
                if title_id:
                    svc.presentations().batchUpdate(
                        presentationId=pid,
                        body={"requests": [
                            {"insertText": {"objectId": title_id, "text": heading}}
                        ]},
                    ).execute()
            link = f"https://docs.google.com/presentation/d/{pid}/edit" if pid else None
            return connector_result("ok", created=True, id=pid, title=title, link=link)
        except Exception as exc:
            return connector_result("error", error=f"{type(exc).__name__}: {exc}")

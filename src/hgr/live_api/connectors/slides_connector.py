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
        return [
            {
                "type": "function",
                "name": "slides_create",
                "description": (
                    "Create a new Google Slides presentation with a title. "
                    "Optionally set the first slide's heading text. Returns "
                    "{created, id, title, link}; chain id into "
                    "slides_add_slide for additional slides."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string",
                                  "description": "Presentation title."},
                        "heading": {"type": "string",
                                    "description": "Optional text for the title slide."},
                    },
                    "required": ["title"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "slides_add_slide",
                "description": (
                    "Add a new slide to an existing Google Slides presentation. "
                    "Default layout is TITLE_AND_BODY: pass `title` (heading) "
                    "and optional `body` (bullet/paragraph text — \\n separates "
                    "lines). Returns the new slide's id. Get `presentation_id` "
                    "from slides_create's `id` field via {step:N.id}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["presentation_id", "title"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Google Slides not authorized",
                                    code="not_ready")
        if name == "slides_add_slide":
            return self._add_slide(svc, args)
        if name != "slides_create":
            return connector_result("error", error=f"unknown slides tool: {name}",
                                    code="no_handler")
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

    def _add_slide(self, svc: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        pid = str(args.get("presentation_id") or "").strip()
        title = str(args.get("title") or "").strip()
        body = str(args.get("body") or "")
        if not pid:
            return connector_result("error", error="presentation_id is required")
        if not title:
            return connector_result("error", error="title is required")
        # Pre-name the new slide + its title/body placeholders so we can
        # immediately insert text without an extra fetch round-trip.
        import uuid
        suffix = uuid.uuid4().hex[:8]
        slide_id = f"slide_{suffix}"
        title_id = f"title_{suffix}"
        body_id = f"body_{suffix}"
        requests: List[Dict[str, Any]] = [
            {"createSlide": {
                "objectId": slide_id,
                "slideLayoutReference": {"predefinedLayout": "TITLE_AND_BODY"},
                "placeholderIdMappings": [
                    {"layoutPlaceholder": {"type": "TITLE", "index": 0},
                     "objectId": title_id},
                    {"layoutPlaceholder": {"type": "BODY", "index": 0},
                     "objectId": body_id},
                ],
            }},
            {"insertText": {"objectId": title_id, "text": title}},
        ]
        if body:
            requests.append({"insertText": {"objectId": body_id, "text": body}})
        try:
            svc.presentations().batchUpdate(
                presentationId=pid, body={"requests": requests}).execute()
            link = f"https://docs.google.com/presentation/d/{pid}/edit#slide=id.{slide_id}"
            return connector_result("ok", added=True, id=pid,
                                    slide_id=slide_id, title=title,
                                    link=link)
        except Exception as exc:
            return connector_result("error",
                                    error=f"{type(exc).__name__}: {exc}")

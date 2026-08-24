"""Google Slides connector — create presentations via the Slides API.

"Make a slideshow titled …" as one call, returning the link. Optionally
sets the title-slide text. Dormant until the shared GoogleClient is
authorized with the presentations scope.

Author: Konstantin Markov
"""
from __future__ import annotations

import concurrent.futures
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import Connector, connector_result, friendly_api_error
from .google_client import GoogleClient

_SLIDES_CALL_TIMEOUT_SEC = 25.0
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="gslides")

# Extract the presentation id from a docs.google.com URL.
_PRES_ID_RE = re.compile(r"/presentation/d/([A-Za-z0-9_\-]+)")


def _extract_pid_from_link(link: str) -> Optional[str]:
    """Pull the presentation id out of a docs.google.com URL. None when
    the link isn't a recognised Slides URL."""
    if not link:
        return None
    m = _PRES_ID_RE.search(str(link))
    return m.group(1) if m else None


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
                    "slides_add_slide for additional slides, or "
                    "slides_set_slide_text / slides_replace_text to edit "
                    "existing slides afterward (do NOT fall back to UI "
                    "automation for editing)."
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
                    "lines). Returns the new slide's id. Identify the target "
                    "with EITHER `presentation_id` (chain from slides_create's "
                    "`id` via {step:N.id}) OR `presentation_name` (friendly "
                    "name like 'Q3 review'; resolved via this-session artifact "
                    "tracker then a Drive search). Exactly one is required. "
                    "To edit an existing slide later, use slides_set_slide_text "
                    "(targeted) or slides_replace_text (find/replace)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "presentation_name": {
                            "type": "string",
                            "description": "Optional friendly name of the "
                                           "presentation (e.g. 'Q3 review'). "
                                           "Used to resolve presentation_id "
                                           "when not provided — falls back to "
                                           "this-session artifact tracker, "
                                           "then a Drive search.",
                        },
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["title"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "slides_set_slide_text",
                "description": (
                    "Set or append text on an EXISTING slide's TITLE and/or "
                    "BODY placeholders via the Slides API. Use this to edit a "
                    "slide that was already created — never fall back to "
                    "keyboard/mouse automation. `mode='replace'` (default) "
                    "clears the placeholder first; `mode='append'` adds to the "
                    "end. Pass `title` and/or `body`; omit either to leave it "
                    "untouched. Get `slide_id` from slides_add_slide's "
                    "`slide_id` field via {step:N.slide_id}. Identify the "
                    "target presentation with EITHER `presentation_id` OR "
                    "`presentation_name`."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "presentation_name": {
                            "type": "string",
                            "description": "Optional friendly name of the "
                                           "presentation (e.g. 'Q3 review'). "
                                           "Used to resolve presentation_id "
                                           "when not provided — falls back to "
                                           "this-session artifact tracker, "
                                           "then a Drive search.",
                        },
                        "slide_id": {"type": "string"},
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "mode": {"type": "string",
                                 "enum": ["replace", "append"],
                                 "description": "replace clears existing text first; append adds to end. Default replace."},
                    },
                    "required": ["slide_id"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "slides_replace_text",
                "description": (
                    "Find-and-replace text across ALL slides in a Google "
                    "Slides presentation via Slides API replaceAllText. Use "
                    "this to swap a placeholder string (e.g. replace "
                    "\"[name]\" with \"Konstantin\") without knowing which "
                    "slide it's on. Returns {replaced: occurrences_count}. "
                    "For targeted edits to one slide's title/body, prefer "
                    "slides_set_slide_text. Identify the target presentation "
                    "with EITHER `presentation_id` OR `presentation_name`."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "presentation_name": {
                            "type": "string",
                            "description": "Optional friendly name of the "
                                           "presentation (e.g. 'Q3 review'). "
                                           "Used to resolve presentation_id "
                                           "when not provided — falls back to "
                                           "this-session artifact tracker, "
                                           "then a Drive search.",
                        },
                        "find": {"type": "string",
                                 "description": "Text to search for."},
                        "replace": {"type": "string",
                                    "description": "Replacement text."},
                        "match_case": {"type": "boolean",
                                       "description": "Case-sensitive match. Default false."},
                    },
                    "required": ["find", "replace"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _pool.submit(self._execute_locked, name, args).result(
                timeout=_SLIDES_CALL_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            return connector_result(
                "error",
                error=f"{name} timed out after {_SLIDES_CALL_TIMEOUT_SEC}s",
                code="timeout",
            )

    def _execute_locked(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Google Slides not authorized",
                                    code="not_ready")
        if name == "slides_add_slide":
            return self._add_slide(svc, args)
        if name == "slides_set_slide_text":
            return self._set_slide_text(svc, args)
        if name == "slides_replace_text":
            return self._replace_text(svc, args)
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
            # Warm the picker cache so future "add slide to X" by title
            # skips the Picker (drive.file scope hides user-picked files
            # from Drive.list, so cache is the primary cross-session
            # resolver).
            if pid:
                try:
                    from .google_picker_cache import shared as _picker_cache
                    _picker_cache().remember(
                        title, "slide", pid, title,
                        "application/vnd.google-apps.presentation")
                except Exception:
                    pass
            return connector_result("ok", created=True, id=pid, title=title, link=link)
        except Exception as exc:
            return connector_result("error", error=friendly_api_error(exc, api_label="Google Slides"))

    def _add_slide(self, svc: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        pid = str(args.get("presentation_id") or "").strip()
        pres_name = str(args.get("presentation_name") or "").strip()
        title = str(args.get("title") or "").strip()
        body = str(args.get("body") or "")
        # Resolve friendly presentation_name → presentation_id when the
        # caller only gave the name (phase-2 classifier route).
        if not pid and pres_name:
            pid, resolve_err = self._resolve_pid_by_name(pres_name)
            if resolve_err and not pid:
                return connector_result(
                    "error", error=resolve_err, code="need_presentation")
        if not pid:
            hint = (f" (couldn't find a presentation matching '{pres_name}')"
                    if pres_name else "")
            return connector_result(
                "error",
                error=("I need the presentation — say its name or paste "
                       "the URL" + hint),
                code="need_presentation")
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
                                    error=friendly_api_error(exc, api_label="Google Slides"))

    def _set_slide_text(self, svc: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        pid = str(args.get("presentation_id") or "").strip()
        pres_name = str(args.get("presentation_name") or "").strip()
        slide_id = str(args.get("slide_id") or "").strip()
        title = args.get("title")
        body = args.get("body")
        mode = str(args.get("mode") or "replace").strip().lower()
        if not pid and pres_name:
            pid, resolve_err = self._resolve_pid_by_name(pres_name)
            if resolve_err and not pid:
                return connector_result(
                    "error", error=resolve_err, code="need_presentation")
        if not pid:
            hint = (f" (couldn't find a presentation matching '{pres_name}')"
                    if pres_name else "")
            return connector_result(
                "error",
                error=("I need the presentation — say its name or paste "
                       "the URL" + hint),
                code="need_presentation")
        if not slide_id:
            return connector_result("error", error="slide_id is required")
        if title is None and body is None:
            return connector_result("error", error="title or body is required")
        if mode not in ("replace", "append"):
            return connector_result("error", error="mode must be 'replace' or 'append'")
        try:
            pres = svc.presentations().get(
                presentationId=pid,
                fields="slides(objectId,pageElements(objectId,shape(placeholder(type),text(textElements(endIndex)))))",
            ).execute()
            target = None
            for slide in (pres.get("slides") or []):
                if slide.get("objectId") == slide_id:
                    target = slide
                    break
            if target is None:
                return connector_result("error",
                                        error=f"slide {slide_id} not found",
                                        code="not_found")
            title_id = None
            body_id = None
            title_end = 0
            body_end = 0
            for el in (target.get("pageElements") or []):
                shape = el.get("shape") or {}
                ptype = shape.get("placeholder", {}).get("type")
                obj_id = el.get("objectId")
                text_elements = (shape.get("text") or {}).get("textElements") or []
                end_index = 0
                for te in text_elements:
                    ei = te.get("endIndex")
                    if isinstance(ei, int) and ei > end_index:
                        end_index = ei
                if ptype in ("TITLE", "CENTERED_TITLE") and title_id is None:
                    title_id = obj_id
                    title_end = end_index
                elif ptype == "BODY" and body_id is None:
                    body_id = obj_id
                    body_end = end_index
            requests: List[Dict[str, Any]] = []
            if title is not None:
                if title_id is None:
                    return connector_result("error",
                                            error="slide has no TITLE placeholder",
                                            code="no_placeholder")
                self._build_text_requests(requests, title_id, str(title),
                                          title_end, mode)
            if body is not None:
                if body_id is None:
                    return connector_result("error",
                                            error="slide has no BODY placeholder",
                                            code="no_placeholder")
                self._build_text_requests(requests, body_id, str(body),
                                          body_end, mode)
            if requests:
                svc.presentations().batchUpdate(
                    presentationId=pid, body={"requests": requests}).execute()
            link = f"https://docs.google.com/presentation/d/{pid}/edit#slide=id.{slide_id}"
            return connector_result("ok", updated=True, id=pid,
                                    slide_id=slide_id, link=link)
        except Exception as exc:
            return connector_result("error",
                                    error=friendly_api_error(exc, api_label="Google Slides"))

    @staticmethod
    def _build_text_requests(requests: List[Dict[str, Any]], object_id: str,
                             text: str, end_index: int, mode: str) -> None:
        if mode == "replace":
            if end_index > 1:
                requests.append({"deleteText": {
                    "objectId": object_id,
                    "textRange": {"type": "ALL"},
                }})
            if text:
                requests.append({"insertText": {
                    "objectId": object_id, "text": text}})
        else:
            if text:
                insertion_index = max(end_index - 1, 0)
                requests.append({"insertText": {
                    "objectId": object_id,
                    "insertionIndex": insertion_index,
                    "text": text,
                }})

    # ---- name → presentation_id resolver ------------------------------
    def _resolve_pid_by_name(self, name: str
                              ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve a friendly presentation name to a presentation id.

        Ladder mirrors sheets_connector._resolve_sid_by_name:
          (1) orchestrator's per-session artifact tracker — a slideshow
              just created this session is almost always what the user
              means,
          (2) PickerCache — file the user picked via Google Picker in a
              prior session,
          (3) Drive search by name — still works for files THIS APP
              created (drive.file grants self-created access).
        Returns (pid, error_message). When multiple Drive matches are
        equally plausible, returns (None, ambiguity-message)."""
        # (1) Artifact tracker (this-session create).
        try:
            from ..planner import current_planner_artifact_lookup
            lookup = current_planner_artifact_lookup()
            if callable(lookup):
                hit = lookup(name, kind="slide")
                if hit:
                    pid = _extract_pid_from_link(str(hit.get("link") or ""))
                    if pid:
                        return pid, None
        except Exception:
            pass
        # (2) PickerCache — prior-session picks keyed by slug.
        try:
            from .google_picker_cache import shared as _picker_cache
            cached = _picker_cache().lookup(name, kind="slide")
            if cached and cached.get("file_id"):
                return cached["file_id"], None
        except Exception:
            pass
        # (3) Drive search.
        try:
            drive = self._client.service("drive", "v3")
        except Exception:
            drive = None
        if drive is None:
            return None, None
        try:
            escaped = name.replace("\\", "\\\\").replace("'", "\\'")
            resp = drive.files().list(
                q=(f"mimeType='application/vnd.google-apps.presentation' "
                   f"and name contains '{escaped}' and trashed=false"),
                orderBy="modifiedTime desc",
                pageSize=5,
                fields="files(id,name,modifiedTime)",
            ).execute()
        except Exception:
            return None, None
        files = (resp.get("files") or [])
        if not files:
            return None, None
        if len(files) == 1:
            return files[0].get("id"), None
        # Multiple matches — prefer exact (case-insensitive) name match.
        exact = [f for f in files
                 if str(f.get("name") or "").lower() == name.lower()]
        if len(exact) == 1:
            return exact[0].get("id"), None
        top = ", ".join(f"'{f.get('name')}'" for f in files[:3])
        return None, (f"Multiple presentations match '{name}': {top}. "
                      "Say a more specific name.")

    def _replace_text(self, svc: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        pid = str(args.get("presentation_id") or "").strip()
        pres_name = str(args.get("presentation_name") or "").strip()
        find = str(args.get("find") or "")
        replace = str(args.get("replace") or "")
        match_case = bool(args.get("match_case") or False)
        if not pid and pres_name:
            pid, resolve_err = self._resolve_pid_by_name(pres_name)
            if resolve_err and not pid:
                return connector_result(
                    "error", error=resolve_err, code="need_presentation")
        if not pid:
            hint = (f" (couldn't find a presentation matching '{pres_name}')"
                    if pres_name else "")
            return connector_result(
                "error",
                error=("I need the presentation — say its name or paste "
                       "the URL" + hint),
                code="need_presentation")
        if not find:
            return connector_result("error", error="find is required")
        try:
            resp = svc.presentations().batchUpdate(
                presentationId=pid,
                body={"requests": [{
                    "replaceAllText": {
                        "containsText": {"text": find, "matchCase": match_case},
                        "replaceText": replace,
                    }
                }]},
            ).execute()
            replies = resp.get("replies") or []
            occurrences = 0
            if replies:
                occurrences = (replies[0].get("replaceAllText") or {}).get(
                    "occurrencesChanged", 0)
            link = f"https://docs.google.com/presentation/d/{pid}/edit"
            return connector_result("ok", replaced=occurrences, id=pid, link=link)
        except Exception as exc:
            return connector_result("error",
                                    error=friendly_api_error(exc, api_label="Google Slides"))

"""Google Forms connector — API-first poll/survey creation + response read.

"Create a poll asking X", "how many responses on the beta signup form" as
single API calls. Dormant until the shared GoogleClient is authorized.

Forms API v1 requires a two-step dance: forms().create() accepts only
the 'info' field; questions must be added via a follow-up batchUpdate
with createItem requests. Mirrors the gdocs_create pattern.

Author: Konstantin Markov
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List

from .base import Connector, connector_result, friendly_api_error
from .google_client import GoogleClient

_FORMS_CALL_TIMEOUT_SEC = 25.0
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="forms")

_CHOICE_KINDS = {"multiple_choice", "checkbox", "dropdown"}
_CHOICE_TYPE_MAP = {
    "multiple_choice": "RADIO",
    "checkbox": "CHECKBOX",
    "dropdown": "DROP_DOWN",
}


class FormsConnector(Connector):
    id = "forms"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _svc(self):
        return self._client.service("forms", "v1")

    def available(self) -> bool:
        try:
            return self._client.ready()
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "forms_create",
                "description": (
                    "Create a new Google Form (poll/survey/quiz) with a "
                    "title and one or more questions. Use this for 'make a "
                    "poll', 'create a survey', 'build a form'. Each "
                    "question becomes a required item on the form. "
                    "`questions` is a list of objects with `title` (the "
                    "question text) and optional `type` ('short_answer'|"
                    "'paragraph'|'multiple_choice'|'checkbox'|'dropdown' — "
                    "defaults to 'short_answer') and `options` (list of "
                    "strings, only used for multiple_choice/checkbox/"
                    "dropdown). The form's title is set on creation; "
                    "questions are added via a follow-up batchUpdate "
                    "because the Forms API does NOT accept items in the "
                    "initial create() call. Returns {created, id, title, "
                    "link, responder_link, question_count}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string",
                                  "description": "Form title shown to respondents."},
                        "description": {"type": "string",
                                        "description": "Blurb under the title."},
                        "questions": {
                            "type": "array",
                            "description": "List of question objects.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "type": {"type": "string",
                                             "description": ("short_answer|paragraph|"
                                                             "multiple_choice|checkbox|dropdown")},
                                    "options": {"type": "array",
                                                "items": {"type": "string"}},
                                    "required": {"type": "boolean"},
                                },
                                "required": ["title"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["title", "questions"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "forms_responses",
                "description": (
                    "Read responses for an existing Google Form — total "
                    "count plus the most recent responses (with answers "
                    "per question when `include_answers` is true). Use "
                    "this for 'how many responses on the beta signup "
                    "form', 'show me what people answered on the survey', "
                    "'check the poll results'. `form_id` is required (get "
                    "it from forms_create's `id`, or the user pastes a "
                    "form URL — strip the /d/<ID>/edit segment). Returns "
                    "{count, recent: [{response_id, submitted_at, "
                    "answers?: {question_title: value}}], form_title}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "form_id": {"type": "string"},
                        "max": {"type": "integer",
                                "description": "Max recent responses (default 20, cap 100)."},
                        "include_answers": {"type": "boolean",
                                            "description": "Denormalize answers by question title."},
                    },
                    "required": ["form_id"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _pool.submit(self._execute_locked, name, args).result(
                timeout=_FORMS_CALL_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            return connector_result(
                "error",
                error=f"{name} timed out after {_FORMS_CALL_TIMEOUT_SEC}s",
                code="timeout",
            )

    def _execute_locked(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Google Forms not authorized",
                                    code="not_ready")

        if name == "forms_create":
            return self._forms_create(svc, args)

        if name == "forms_responses":
            return self._forms_responses(svc, args)

        return connector_result("error",
                                error=f"unknown forms tool: {name}",
                                code="no_handler")

    def _forms_create(self, svc, args: Dict[str, Any]) -> Dict[str, Any]:
        title = str(args.get("title") or "").strip()
        if not title:
            return connector_result("error", error="title is required")
        questions = args.get("questions") or []
        if not isinstance(questions, list):
            return connector_result("error", error="questions must be a list")

        try:
            info: Dict[str, Any] = {"title": title, "documentTitle": title}
            form = svc.forms().create(body={"info": info}).execute()
            form_id = form.get("formId")
            if not form_id:
                return connector_result("error", error="Forms API returned no formId")

            desc = str(args.get("description") or "").strip()
            update_requests: List[Dict[str, Any]] = []
            if desc:
                update_requests.append({
                    "updateFormInfo": {
                        "info": {"description": desc},
                        "updateMask": "description",
                    }
                })

            for i, q in enumerate(questions):
                req = self._build_create_item_request(q, i)
                if req is not None:
                    update_requests.append(req)

            if update_requests:
                svc.forms().batchUpdate(
                    formId=form_id,
                    body={"requests": update_requests,
                          "includeFormInResponse": False}).execute()

            responder_link = (form.get("responderUri")
                              or f"https://docs.google.com/forms/d/e/{form_id}/viewform")
            link = f"https://docs.google.com/forms/d/{form_id}/edit"
            return connector_result("ok", created=True, id=form_id,
                                    title=title, link=link,
                                    responder_link=responder_link,
                                    question_count=len(questions),
                                    summary=f"Created form: {title}.")
        except Exception as exc:
            return connector_result("error",
                                    error=friendly_api_error(exc, api_label="Google Forms"))

    @staticmethod
    def _build_create_item_request(q: Any, index: int) -> Dict[str, Any] | None:
        if not isinstance(q, dict):
            return None
        q_title = str(q.get("title") or "").strip()
        if not q_title:
            return None
        kind = str(q.get("type") or "short_answer").strip().lower()
        required = q.get("required")
        if required is None:
            required = True

        question_payload: Dict[str, Any] = {"required": bool(required)}

        if kind in _CHOICE_KINDS:
            options = q.get("options") or []
            opts = [{"value": str(o)} for o in options if str(o).strip()]
            if not opts:
                # Choice questions without options would 400 from the API;
                # degrade to a short-answer so the form still builds.
                question_payload["textQuestion"] = {"paragraph": False}
            else:
                question_payload["choiceQuestion"] = {
                    "type": _CHOICE_TYPE_MAP[kind],
                    "options": opts,
                    "shuffle": False,
                }
        elif kind == "paragraph":
            question_payload["textQuestion"] = {"paragraph": True}
        else:
            question_payload["textQuestion"] = {"paragraph": False}

        return {
            "createItem": {
                "item": {
                    "title": q_title,
                    "questionItem": {"question": question_payload},
                },
                "location": {"index": index},
            }
        }

    def _forms_responses(self, svc, args: Dict[str, Any]) -> Dict[str, Any]:
        form_id = str(args.get("form_id") or "").strip()
        if not form_id:
            return connector_result("error", error="form_id is required")
        max_n = max(1, min(100, int(args.get("max") or 20)))
        include_answers = bool(args.get("include_answers"))

        try:
            resp = svc.forms().responses().list(
                formId=form_id, pageSize=max_n).execute()
            items = resp.get("responses", []) or []

            qid_to_title: Dict[str, str] = {}
            form_title: str | None = None
            if include_answers:
                form_meta = svc.forms().get(formId=form_id).execute()
                form_title = (form_meta.get("info", {}) or {}).get("title")
                for item in form_meta.get("items", []) or []:
                    qi = item.get("questionItem")
                    if not qi:
                        continue
                    qid = (qi.get("question") or {}).get("questionId")
                    if qid:
                        qid_to_title[qid] = item.get("title") or qid

            recent: List[Dict[str, Any]] = []
            for r in items:
                entry: Dict[str, Any] = {
                    "response_id": r.get("responseId"),
                    "submitted_at": r.get("lastSubmittedTime"),
                }
                if include_answers:
                    answers_out: Dict[str, str] = {}
                    for qid, answer in (r.get("answers") or {}).items():
                        key = qid_to_title.get(qid, qid)
                        answers_out[key] = _flatten_answer(answer)
                    entry["answers"] = answers_out
                else:
                    entry["answers"] = None
                recent.append(entry)

            return connector_result("ok", count=len(items),
                                    form_title=form_title,
                                    recent=recent)
        except Exception as exc:
            return connector_result("error",
                                    error=friendly_api_error(exc, api_label="Google Forms"))


def _flatten_answer(answer: Any) -> str:
    if not isinstance(answer, dict):
        return ""
    text = answer.get("textAnswers") or {}
    values = text.get("answers") or []
    out: List[str] = []
    for v in values:
        if isinstance(v, dict):
            val = v.get("value")
            if val is not None:
                out.append(str(val))
    return ", ".join(out)

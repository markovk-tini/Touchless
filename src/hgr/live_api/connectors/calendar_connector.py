"""Google Calendar connector — API-first agenda read/create.

"What's on my calendar today", "add a meeting tomorrow at 3pm" as single
API calls. Dormant until the shared GoogleClient is authorized.

Datetimes are RFC3339 strings (e.g. '2026-05-27T15:00:00-07:00'); the
model is told to pass them that way.

Author: Konstantin Markov
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
from typing import Any, Dict, List

from .base import Connector, connector_result, friendly_api_error
from .google_client import GoogleClient

_CALENDAR_CALL_TIMEOUT_SEC = 25.0
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="calendar")


class CalendarConnector(Connector):
    id = "calendar"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _svc(self):
        return self._client.service("calendar", "v3")

    def available(self) -> bool:
        try:
            return self._client.ready()
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        def fn(name, desc, props=None, required=None):
            return {"type": "function", "name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props or {},
                                   "required": required or [],
                                   "additionalProperties": False}}
        return [
            fn("calendar_list_events",
               "List upcoming Google Calendar events (soonest first).",
               {"max": {"type": "integer", "description": "Max events (default 10)."}}),
            fn("calendar_create_event",
               "Create a Google Calendar event. start/end are RFC3339 datetimes "
               "with timezone offset, e.g. '2026-05-27T15:00:00-07:00'.",
               {"summary": {"type": "string", "description": "Event title."},
                "start": {"type": "string"},
                "end": {"type": "string"},
                "description": {"type": "string"}},
               ["summary", "start", "end"]),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _pool.submit(self._execute_locked, name, args).result(
                timeout=_CALENDAR_CALL_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            return connector_result(
                "error",
                error=f"{name} timed out after {_CALENDAR_CALL_TIMEOUT_SEC}s",
                code="timeout",
            )

    def _execute_locked(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Calendar not authorized", code="not_ready")
        try:
            if name == "calendar_list_events":
                max_n = max(1, min(50, int(args.get("max") or 10)))
                now = datetime.now(timezone.utc).isoformat()
                events = svc.events().list(
                    calendarId="primary", timeMin=now, maxResults=max_n,
                    singleEvents=True, orderBy="startTime").execute()
                out = []
                for e in events.get("items", []) or []:
                    start = e.get("start", {})
                    out.append({"summary": e.get("summary"),
                                "start": start.get("dateTime") or start.get("date"),
                                "id": e.get("id")})
                return connector_result("ok", count=len(out), events=out)

            if name == "calendar_create_event":
                try:
                    from .outlook_com_connector import _diag as _ocl_diag
                    _ocl_diag(f"calendar_create_event (Google) PICKED "
                              f"args={args!r}")
                except Exception:
                    pass
                summary = str(args.get("summary") or "").strip()
                start = str(args.get("start") or "").strip()
                end = str(args.get("end") or "").strip()
                if not (summary and start and end):
                    return connector_result("error", error="summary, start, end are required")
                body = {"summary": summary,
                        "start": {"dateTime": start},
                        "end": {"dateTime": end}}
                desc = str(args.get("description") or "").strip()
                if desc:
                    body["description"] = desc
                created = svc.events().insert(calendarId="primary", body=body).execute()
                try:
                    _dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    try:
                        start_label = _dt.strftime("%a %b %-d at %-I:%M %p")
                    except ValueError:
                        start_label = _dt.strftime("%a %b %#d at %#I:%M %p")
                except Exception:
                    start_label = start
                return connector_result("ok", created=True, id=created.get("id"),
                                        link=created.get("htmlLink"),
                                        calendar="Google Calendar",
                                        summary=f"Added '{summary}' to your "
                                                f"Google Calendar on "
                                                f"{start_label}.")
        except Exception as exc:
            return connector_result("error", error=friendly_api_error(exc, api_label="Google Calendar"))
        return connector_result("error", error=f"unknown calendar tool: {name}", code="no_handler")

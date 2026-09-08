"""Google Tasks connector — API-first to-do list read/write.

"What's on my task list", "add buy milk to my tasks", "mark buy milk as
done", "remove buy milk from my tasks" as single API calls. Dormant
until the shared GoogleClient is authorized.

The Tasks API has a two-level hierarchy (tasklists contain tasks); the
'@default' alias targets the user's primary list without a lookup
round-trip. Due dates are RFC3339 datetimes but the API stores only
the date portion — pass 'YYYY-MM-DDT00:00:00Z' for 'due that day'.

Author: Konstantin Markov
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List

from .base import Connector, connector_result, friendly_api_error
from .google_client import GoogleClient

_CALL_TIMEOUT_SEC = 25.0
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="tasks")


class TasksConnector(Connector):
    id = "tasks"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _svc(self):
        return self._client.service("tasks", "v1")

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
            fn("tasks_list",
               "List open tasks from the user's default Google Tasks list "
               "(most recent first). Returns task titles, due dates, "
               "completion status, and ids — use the id to complete or "
               "delete a specific task. Skips already-completed tasks "
               "unless include_completed=true.",
               {"max": {"type": "integer",
                        "description": "Max tasks to return (default 20, 1-100)."},
                "include_completed": {"type": "boolean",
                                      "description": "Include completed tasks (default false)."},
                "tasklist_id": {"type": "string",
                                "description": "Id of a specific task list; defaults to '@default'."}}),
            fn("tasks_add",
               "Add a new task to the user's Google Tasks list. Pass the "
               "task title (e.g. 'buy milk'); optionally include a due date "
               "as RFC3339 datetime ('2026-06-10T00:00:00Z') and notes. "
               "Returns the new task id so a follow-up tasks_complete or "
               "tasks_delete can target it.",
               {"title": {"type": "string",
                          "description": "Task title text."},
                "notes": {"type": "string",
                          "description": "Longer description body."},
                "due": {"type": "string",
                        "description": "RFC3339 datetime for the due date "
                                       "(date-only 'YYYY-MM-DD' also accepted)."},
                "tasklist_id": {"type": "string",
                                "description": "Tasklist id; defaults to '@default'."}},
               ["title"]),
            fn("tasks_complete",
               "Mark a Google Tasks item as done. Pass either the exact "
               "task id (from tasks_list / tasks_add) OR a free-text "
               "title_match — if title_match is given, the connector finds "
               "the most recent open task whose title contains that "
               "substring (case-insensitive) and completes it. Useful for "
               "'mark buy milk as done'.",
               {"task_id": {"type": "string",
                            "description": "Exact Tasks API id."},
                "title_match": {"type": "string",
                                "description": "Substring to fuzzy-match against open task titles."},
                "tasklist_id": {"type": "string",
                                "description": "Tasklist id; defaults to '@default'."}}),
            fn("tasks_delete",
               "Permanently remove a task from the user's Google Tasks "
               "list. Same resolution semantics as tasks_complete: pass "
               "either task_id (exact) or title_match (fuzzy substring on "
               "open tasks). Useful for 'remove buy milk from my tasks'.",
               {"task_id": {"type": "string",
                            "description": "Exact Tasks API id."},
                "title_match": {"type": "string",
                                "description": "Substring to fuzzy-match against open task titles."},
                "tasklist_id": {"type": "string",
                                "description": "Tasklist id; defaults to '@default'."}}),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _pool.submit(self._execute_locked, name, args).result(
                timeout=_CALL_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            return connector_result(
                "error",
                error=f"{name} timed out after {_CALL_TIMEOUT_SEC}s",
                code="timeout",
            )

    def _execute_locked(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Google Tasks not authorized",
                                    code="not_ready")
        if not self._client.has_scope("https://www.googleapis.com/auth/tasks"):
            return self._client.scope_missing_result(
                "https://www.googleapis.com/auth/tasks",
                friendly="Google Tasks")
        try:
            tasklist_id = str(args.get("tasklist_id") or "@default").strip() or "@default"

            if name == "tasks_list":
                max_n = max(1, min(100, int(args.get("max") or 20)))
                include_completed = bool(args.get("include_completed") or False)
                result = svc.tasks().list(
                    tasklist=tasklist_id, maxResults=max_n,
                    showCompleted=include_completed, showHidden=False).execute()
                out = []
                for t in result.get("items", []) or []:
                    out.append({"id": t.get("id"),
                                "title": t.get("title"),
                                "due": t.get("due"),
                                "status": t.get("status"),
                                "notes": t.get("notes")})
                if not out:
                    list_summary = "No open tasks on your Google Tasks list."
                else:
                    titles = ", ".join(
                        ((t.get("title") or "").strip() or "(untitled)")
                        for t in out[:5])
                    more = f" (+{len(out) - 5} more)" if len(out) > 5 else ""
                    plural = "s" if len(out) != 1 else ""
                    list_summary = (
                        f"{len(out)} task{plural}: {titles}{more}.")
                return connector_result("ok", count=len(out), tasks=out,
                                        tasklist=self._tasklist_title(svc, tasklist_id),
                                        summary=list_summary)

            if name == "tasks_add":
                title = str(args.get("title") or "").strip()
                if not title:
                    return connector_result("error", error="title is required")
                body: Dict[str, Any] = {"title": title}
                notes = str(args.get("notes") or "").strip()
                if notes:
                    body["notes"] = notes
                due = str(args.get("due") or "").strip()
                if due:
                    if "T" not in due:
                        due = f"{due}T00:00:00Z"
                    body["due"] = due
                created = svc.tasks().insert(tasklist=tasklist_id, body=body).execute()
                created_title = created.get("title") or title
                return connector_result("ok", created=True, id=created.get("id"),
                                        title=created_title,
                                        link="https://tasks.google.com/",
                                        tasklist=self._tasklist_title(svc, tasklist_id),
                                        summary=f"Added task: {created_title}.")

            if name == "tasks_complete":
                task_id, resolved_title, err = self._resolve_task(svc, tasklist_id, args)
                if err is not None:
                    return err
                svc.tasks().patch(tasklist=tasklist_id, task=task_id,
                                  body={"status": "completed"}).execute()
                return connector_result("ok", completed=True, id=task_id,
                                        title=resolved_title,
                                        tasklist=self._tasklist_title(svc, tasklist_id),
                                        summary=f"Marked done: {resolved_title}.")

            if name == "tasks_delete":
                task_id, resolved_title, err = self._resolve_task(svc, tasklist_id, args)
                if err is not None:
                    return err
                svc.tasks().delete(tasklist=tasklist_id, task=task_id).execute()
                return connector_result("ok", deleted=True, id=task_id,
                                        title=resolved_title,
                                        tasklist=self._tasklist_title(svc, tasklist_id),
                                        summary=f"Removed task: {resolved_title}.")
        except Exception as exc:
            code = ""
            try:
                from googleapiclient.errors import HttpError
                if isinstance(exc, HttpError):
                    status_code = int(getattr(exc.resp, "status", 0) or 0)
                    if status_code == 403:
                        code = "http_403"
                    elif status_code == 401:
                        code = "http_401"
                    elif status_code == 404:
                        code = "http_404"
                    elif status_code == 429:
                        code = "http_429"
            except Exception:
                pass
            return connector_result("error",
                                    error=friendly_api_error(exc, api_label="Google Tasks"),
                                    code=code)
        return connector_result("error", error=f"unknown tasks tool: {name}",
                                code="no_handler")

    def _resolve_task(self, svc, tasklist_id: str, args: Dict[str, Any]):
        task_id = str(args.get("task_id") or "").strip()
        title_match = str(args.get("title_match") or "").strip()
        if not task_id and not title_match:
            return None, None, connector_result(
                "error", error="task_id or title_match is required")
        if task_id:
            try:
                t = svc.tasks().get(tasklist=tasklist_id, task=task_id).execute()
                return task_id, t.get("title"), None
            except Exception as exc:
                return None, None, connector_result(
                    "error", error=friendly_api_error(exc, api_label="Google Tasks"))
        needle = title_match.lower()
        result = svc.tasks().list(tasklist=tasklist_id,
                                  showCompleted=False, showHidden=False).execute()
        items = result.get("items", []) or []
        matches = [t for t in items
                   if needle in (t.get("title") or "").lower()]
        if not matches:
            return None, None, connector_result(
                "error", error=f"no open task matching '{title_match}'",
                code="not_found")
        picked = matches[0]
        out = (picked.get("id"), picked.get("title"), None)
        if len(matches) > 1:
            return picked.get("id"), picked.get("title"), None
        return out

    def _tasklist_title(self, svc, tasklist_id: str) -> str:
        try:
            tl = svc.tasklists().get(tasklist=tasklist_id).execute()
            return tl.get("title") or tasklist_id
        except Exception:
            return tasklist_id


# Author: Konstantin Markov

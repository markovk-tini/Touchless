"""Microsoft 365 connector — Outlook mail, M365 calendar, OneDrive via Graph.

One connector covering the high-value Microsoft Graph actions, mirroring the
Google connectors. Sending mail (ms_mail_send) is confirm-gated like
gmail_send. Dormant until the shared MsGraphClient is authorized.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from .base import Connector, connector_result
from .ms_graph_client import MsGraphClient, GRAPH_BASE


class Microsoft365Connector(Connector):
    id = "ms365"
    description = ("Microsoft 365 Outlook email send, Microsoft calendar events, "
                   "OneDrive upload files — Office 365 / Copilot apps")

    def __init__(self, client: MsGraphClient | None = None) -> None:
        self._client = client or MsGraphClient.shared()

    def available(self) -> bool:
        try:
            return self._client.ready()
        except Exception:
            return False

    # ---- Graph REST helper ----
    def _graph(self, method: str, path: str, body: Dict[str, Any] | None = None,
               raw: bytes | None = None, content_type: str | None = None):
        token = self._client.token()
        if not token:
            return None, "not_connected"
        url = GRAPH_BASE + path
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {token}")
        if content_type:
            req.add_header("Content-Type", content_type)
        elif body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                text = resp.read().decode("utf-8") if resp.length != 0 else ""
                return (json.loads(text) if text else {}), None
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:
                pass
            return None, f"HTTP {exc.code}: {detail}"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def tools(self) -> List[Dict[str, Any]]:
        def fn(name, desc, props=None, required=None):
            return {"type": "function", "name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props or {},
                                   "required": required or [],
                                   "additionalProperties": False}}
        return [
            fn("ms_mail_send",
               "Send an email via Outlook / Microsoft 365. Sends immediately — "
               "confirm intent first. Always include a concise subject.",
               {"to": {"type": "string"}, "subject": {"type": "string"},
                "body": {"type": "string"}}, ["to", "subject", "body"]),
            fn("ms_calendar_list",
               "List upcoming Microsoft 365 calendar events (soonest first).",
               {"max": {"type": "integer", "description": "Max events (default 10)."}}),
            fn("ms_calendar_create",
               "Create a Microsoft 365 calendar event. start/end are ISO 8601 "
               "datetimes, e.g. '2026-05-27T15:00:00'.",
               {"subject": {"type": "string"}, "start": {"type": "string"},
                "end": {"type": "string"}}, ["subject", "start", "end"]),
            fn("onedrive_upload",
               "Upload a local file to the user's OneDrive; returns the link.",
               {"path": {"type": "string", "description": "Absolute local file path."}},
               ["path"]),
            fn("onedrive_list",
               "List recent files in the user's OneDrive root.",
               {"max": {"type": "integer", "description": "Max files (default 20)."}}),
            fn("teams_send",
               "Send a 1:1 Microsoft Teams chat message to a person by email. "
               "Sends immediately — confirm first.",
               {"to": {"type": "string", "description": "Recipient's email."},
                "text": {"type": "string"}}, ["to", "text"]),
            fn("excel_set_cell",
               "Set a cell value in a OneDrive Excel workbook (found by file name).",
               {"file": {"type": "string", "description": "Workbook name, e.g. 'Budget.xlsx'."},
                "cell": {"type": "string", "description": "A1-style cell, e.g. 'B2'."},
                "value": {"type": "string"},
                "sheet": {"type": "string", "description": "Worksheet name (default Sheet1)."}},
               ["file", "cell", "value"]),
            fn("excel_read_range",
               "Read a range from a OneDrive Excel workbook (found by file name).",
               {"file": {"type": "string"},
                "range": {"type": "string", "description": "A1-style range, e.g. 'A1:C5'."},
                "sheet": {"type": "string", "description": "Worksheet name (default Sheet1)."}},
               ["file", "range"]),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "ms_mail_send":
            to = str(args.get("to") or "").strip()
            if not to:
                return connector_result("error", error="'to' is required")
            msg = {
                "message": {
                    "subject": str(args.get("subject") or ""),
                    "body": {"contentType": "Text", "content": str(args.get("body") or "")},
                    "toRecipients": [{"emailAddress": {"address": to}}],
                },
                "saveToSentItems": True,
            }
            _, err = self._graph("POST", "/me/sendMail", body=msg)
            return connector_result("error" if err else "ok",
                                    error=err, sent=(err is None), to=to)

        if name == "ms_calendar_list":
            max_n = max(1, min(50, int(args.get("max") or 10)))
            data, err = self._graph(
                "GET", f"/me/events?$top={max_n}&$orderby=start/dateTime&"
                       f"$select=subject,start,end,webLink")
            if err:
                return connector_result("error", error=err)
            events = [{"subject": e.get("subject"),
                       "start": (e.get("start") or {}).get("dateTime"),
                       "link": e.get("webLink")} for e in (data.get("value") or [])]
            return connector_result("ok", count=len(events), events=events)

        if name == "ms_calendar_create":
            subject = str(args.get("subject") or "").strip()
            start = str(args.get("start") or "").strip()
            end = str(args.get("end") or "").strip()
            if not (subject and start and end):
                return connector_result("error", error="subject, start, end are required")
            body = {"subject": subject,
                    "start": {"dateTime": start, "timeZone": "UTC"},
                    "end": {"dateTime": end, "timeZone": "UTC"}}
            data, err = self._graph("POST", "/me/events", body=body)
            if err:
                return connector_result("error", error=err)
            return connector_result("ok", created=True, id=data.get("id"),
                                    link=data.get("webLink"))

        if name == "onedrive_upload":
            path = str(args.get("path") or "").strip()
            if not os.path.isfile(path):
                return connector_result("error", error=f"no such file: {path}", code="not_found")
            with open(path, "rb") as fh:
                content = fh.read()
            fname = os.path.basename(path)
            data, err = self._graph(
                "PUT", f"/me/drive/root:/{urllib.request.quote(fname)}:/content",
                raw=content, content_type="application/octet-stream")
            if err:
                return connector_result("error", error=err)
            return connector_result("ok", uploaded=True, name=data.get("name"),
                                    link=data.get("webUrl"))

        if name == "onedrive_list":
            max_n = max(1, min(100, int(args.get("max") or 20)))
            data, err = self._graph("GET", f"/me/drive/root/children?$top={max_n}&$select=name,webUrl")
            if err:
                return connector_result("error", error=err)
            files = [{"name": f.get("name"), "link": f.get("webUrl")}
                     for f in (data.get("value") or [])]
            return connector_result("ok", count=len(files), files=files)

        if name == "teams_send":
            to = str(args.get("to") or "").strip()
            text = str(args.get("text") or "")
            if not to:
                return connector_result("error", error="'to' is required")
            # oneOnOne chats are unique per member pair — POST returns the
            # existing chat if it already exists, so this is safe to repeat.
            chat_body = {
                "chatType": "oneOnOne",
                "members": [
                    {"@odata.type": "#microsoft.graph.aadUserConversationMember",
                     "roles": ["owner"],
                     "user@odata.bind": "https://graph.microsoft.com/v1.0/me"},
                    {"@odata.type": "#microsoft.graph.aadUserConversationMember",
                     "roles": ["owner"],
                     "user@odata.bind": f"https://graph.microsoft.com/v1.0/users('{to}')"},
                ],
            }
            chat, err = self._graph("POST", "/chats", body=chat_body)
            if err:
                return connector_result("error", error=err)
            cid = chat.get("id")
            _, err = self._graph("POST", f"/chats/{cid}/messages",
                                 body={"body": {"content": text}})
            return connector_result("error" if err else "ok", error=err,
                                    sent=(err is None), to=to)

        if name in ("excel_set_cell", "excel_read_range"):
            file = str(args.get("file") or "").strip()
            if not file:
                return connector_result("error", error="file is required")
            item_id, err = self._excel_item(file)
            if err:
                return connector_result("error", error=err, code="not_found")
            sheet = str(args.get("sheet") or "Sheet1").strip() or "Sheet1"
            base = f"/me/drive/items/{item_id}/workbook/worksheets('{sheet}')"
            if name == "excel_set_cell":
                cell = str(args.get("cell") or "").strip()
                if not cell:
                    return connector_result("error", error="cell is required")
                _, err = self._graph(
                    "PATCH", f"{base}/range(address='{cell}')",
                    body={"values": [[str(args.get("value") or "")]]})
                return connector_result("error" if err else "ok", error=err,
                                        file=file, cell=cell)
            rng = str(args.get("range") or "").strip()
            if not rng:
                return connector_result("error", error="range is required")
            data, err = self._graph("GET", f"{base}/range(address='{rng}')?$select=values")
            if err:
                return connector_result("error", error=err)
            return connector_result("ok", file=file, range=rng,
                                    values=(data or {}).get("values"))

        return connector_result("error", error=f"unknown ms365 tool: {name}", code="no_handler")

    def _excel_item(self, file: str):
        """Resolve a OneDrive workbook by name → (item_id, None) or (None, err)."""
        q = urllib.parse.quote(file)
        res, err = self._graph("GET", f"/me/drive/root/search(q='{q}')?$top=10&$select=id,name")
        if err:
            return None, err
        items = res.get("value") or []
        for it in items:
            if str(it.get("name", "")).lower().endswith((".xlsx", ".xlsm")):
                return it.get("id"), None
        if items:
            return items[0].get("id"), None
        return None, f"no workbook found named '{file}'"

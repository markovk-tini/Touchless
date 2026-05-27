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
        # Encode spaces (OData $filter like "isRead eq false" has them) — a raw
        # space makes urllib reject the URL as containing control characters.
        url = (GRAPH_BASE + path).replace(" ", "%20")
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
            fn("ms_mail_list",
               "List recent Outlook inbox messages (sender, subject, preview). "
               "Set unread_only=true for just unread.",
               {"max": {"type": "integer", "description": "Max messages (default 10)."},
                "unread_only": {"type": "boolean", "default": False}}),
            fn("ms_mail_search",
               "Search Outlook mail for a query; returns matching messages.",
               {"query": {"type": "string"},
                "max": {"type": "integer", "description": "Max results (default 10)."}},
               ["query"]),
            fn("ms_mail_read",
               "Read the full body of one Outlook message by its id (from "
               "ms_mail_list / ms_mail_search).",
               {"id": {"type": "string"}}, ["id"]),
            fn("ms_mail_mark_read",
               "Mark an Outlook message as read (or unread) by id.",
               {"id": {"type": "string"},
                "read": {"type": "boolean", "default": True}}, ["id"]),
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
               "Send a 1:1 Microsoft Teams chat message. `to` can be the "
               "person's email OR their name (matched against your existing "
               "Teams chats). Sends immediately — confirm first.",
               {"to": {"type": "string", "description": "Recipient email or name."},
                "text": {"type": "string"}}, ["to", "text"]),
            fn("teams_channel_post",
               "Post a message to a Microsoft Teams channel (resolved by team "
               "name; defaults to the team's General channel). Confirm first.",
               {"team": {"type": "string", "description": "Team name."},
                "text": {"type": "string"},
                "channel": {"type": "string",
                            "description": "Channel name (default: General)."}},
               ["team", "text"]),
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
            fn("todo_add",
               "Add a task to Microsoft To Do (the user's default task list).",
               {"title": {"type": "string"}, "note": {"type": "string"}}, ["title"]),
            fn("todo_list",
               "List tasks from Microsoft To Do (default list).",
               {"max": {"type": "integer", "description": "Max tasks (default 20)."}}),
            fn("onenote_create",
               "Create a OneNote page with a title and text in the default section.",
               {"title": {"type": "string"}, "text": {"type": "string"}}, ["title"]),
            fn("contacts_search",
               "Search the user's Outlook contacts; returns names + emails.",
               {"query": {"type": "string"},
                "max": {"type": "integer", "description": "Max results (default 10)."}},
               ["query"]),
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

        if name == "ms_mail_list":
            max_n = max(1, min(50, int(args.get("max") or 10)))
            flt = "&$filter=isRead eq false" if args.get("unread_only") else ""
            data, err = self._graph(
                "GET", f"/me/mailFolders/inbox/messages?$top={max_n}{flt}"
                       "&$select=id,subject,from,receivedDateTime,bodyPreview"
                       "&$orderby=receivedDateTime desc")  # newest first
            if err:
                return connector_result("error", error=err)
            msgs = [{"id": m.get("id"),
                     "from": (m.get("from") or {}).get("emailAddress", {}).get("address"),
                     "subject": m.get("subject"),
                     "received": m.get("receivedDateTime"),
                     "preview": m.get("bodyPreview")} for m in (data.get("value") or [])]
            return connector_result("ok", count=len(msgs), messages=msgs)

        if name == "ms_mail_search":
            q = str(args.get("query") or "").strip()
            if not q:
                return connector_result("error", error="query is required")
            max_n = max(1, min(50, int(args.get("max") or 10)))
            sq = urllib.parse.quote(f'"{q}"')
            data, err = self._graph(
                "GET", f"/me/messages?$search={sq}&$top={max_n}"
                       "&$select=id,subject,from,receivedDateTime")
            if err:
                return connector_result("error", error=err)
            msgs = [{"id": m.get("id"),
                     "from": (m.get("from") or {}).get("emailAddress", {}).get("address"),
                     "subject": m.get("subject"),
                     "received": m.get("receivedDateTime")} for m in (data.get("value") or [])]
            return connector_result("ok", count=len(msgs), messages=msgs)

        if name == "ms_mail_read":
            mid = str(args.get("id") or "").strip()
            if not mid:
                return connector_result("error", error="id is required")
            data, err = self._graph(
                "GET", f"/me/messages/{mid}?$select=subject,from,body,receivedDateTime")
            if err:
                return connector_result("error", error=err)
            body = (data.get("body") or {}).get("content") or ""
            if (data.get("body") or {}).get("contentType", "").lower() == "html":
                import re
                body = re.sub(r"<[^>]+>", " ", body)
                body = re.sub(r"\s+", " ", body).strip()
            return connector_result(
                "ok", subject=data.get("subject"),
                **{"from": (data.get("from") or {}).get("emailAddress", {}).get("address")},
                body=body[:5000])

        if name == "ms_mail_mark_read":
            mid = str(args.get("id") or "").strip()
            if not mid:
                return connector_result("error", error="id is required")
            read = bool(args.get("read", True))
            _, err = self._graph("PATCH", f"/me/messages/{mid}", body={"isRead": read})
            return connector_result("error" if err else "ok", error=err, id=mid, read=read)

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
                return connector_result("error", error="'to' (email or name) is required")
            cid, err = self._resolve_chat(to)
            if err:
                return connector_result("error", error=err, code="not_found")
            _, err = self._graph("POST", f"/chats/{cid}/messages",
                                 body={"body": {"content": text}})
            return connector_result("error" if err else "ok", error=err,
                                    sent=(err is None), to=to)

        if name == "teams_channel_post":
            team = str(args.get("team") or "").strip()
            text = str(args.get("text") or "")
            if not team:
                return connector_result("error", error="team is required")
            tid, cid, err = self._teams_channel(team, str(args.get("channel") or "").strip())
            if err:
                return connector_result("error", error=err, code="not_found")
            _, err = self._graph("POST", f"/teams/{tid}/channels/{cid}/messages",
                                 body={"body": {"content": text}})
            return connector_result("error" if err else "ok", error=err,
                                    posted=(err is None), team=team)

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

        if name in ("todo_add", "todo_list"):
            list_id, err = self._todo_default_list()
            if err:
                return connector_result("error", error=err)
            if name == "todo_add":
                title = str(args.get("title") or "").strip()
                if not title:
                    return connector_result("error", error="title is required")
                body = {"title": title}
                note = str(args.get("note") or "").strip()
                if note:
                    body["body"] = {"content": note, "contentType": "text"}
                data, err = self._graph("POST", f"/me/todo/lists/{list_id}/tasks", body=body)
                return connector_result("error" if err else "ok", error=err,
                                        added=(err is None), title=title, id=(data or {}).get("id"))
            max_n = max(1, min(50, int(args.get("max") or 20)))
            data, err = self._graph(
                "GET", f"/me/todo/lists/{list_id}/tasks?$top={max_n}&$select=title,status,id")
            if err:
                return connector_result("error", error=err)
            tasks = [{"title": t.get("title"), "status": t.get("status"), "id": t.get("id")}
                     for t in (data.get("value") or [])]
            return connector_result("ok", count=len(tasks), tasks=tasks)

        if name == "onenote_create":
            title = str(args.get("title") or "").strip()
            if not title:
                return connector_result("error", error="title is required")
            text = str(args.get("text") or "")
            html = (f"<!DOCTYPE html><html><head><title>{title}</title></head>"
                    f"<body><p>{text}</p></body></html>")
            data, err = self._graph("POST", "/me/onenote/pages",
                                    raw=html.encode("utf-8"), content_type="text/html")
            if err:
                return connector_result("error", error=err)
            links = (data or {}).get("links", {}) or {}
            return connector_result("ok", created=True, title=title,
                                    link=(links.get("oneNoteWebUrl") or {}).get("href"))

        if name == "contacts_search":
            q = str(args.get("query") or "").strip()
            if not q:
                return connector_result("error", error="query is required")
            max_n = max(1, min(50, int(args.get("max") or 10)))
            sq = urllib.parse.quote(f'"{q}"')
            data, err = self._graph(
                "GET", f"/me/contacts?$search={sq}&$top={max_n}"
                       "&$select=displayName,emailAddresses")
            if err:
                # $search on contacts can 400 on some mailboxes; fall back to filter.
                fq = urllib.parse.quote(q)
                data, err = self._graph(
                    "GET", f"/me/contacts?$top={max_n}&$select=displayName,emailAddresses"
                           f"&$filter=startswith(displayName,'{fq}')")
                if err:
                    return connector_result("error", error=err)
            out = []
            for c in (data.get("value") or []):
                emails = [e.get("address") for e in (c.get("emailAddresses") or []) if e.get("address")]
                out.append({"name": c.get("displayName"), "emails": emails})
            return connector_result("ok", count=len(out), contacts=out)

        return connector_result("error", error=f"unknown ms365 tool: {name}", code="no_handler")

    def _resolve_chat(self, to: str):
        """Resolve a Teams chat id from an email (create/find a 1:1 chat) OR a
        person's name (match a member in your existing chats). Returns
        (chat_id, None) or (None, err)."""
        if "@" in to:
            # oneOnOne chats are unique per member pair — POST returns the
            # existing one if it exists, so this is safe to repeat.
            body = {
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
            chat, err = self._graph("POST", "/chats", body=body)
            if err:
                return None, err
            return chat.get("id"), None
        # By name: find an existing chat whose members include that name.
        data, err = self._graph(
            "GET", "/me/chats?$expand=members&$top=50&$select=id,chatType,topic")
        if err:
            return None, err
        tn = to.lower()
        fallback = None
        for chat in (data.get("value") or []):
            names = [str(m.get("displayName") or "") for m in (chat.get("members") or [])]
            if any(tn in n.lower() for n in names):
                if chat.get("chatType") == "oneOnOne":
                    return chat.get("id"), None  # prefer a direct 1:1
                fallback = fallback or chat.get("id")
        if fallback:
            return fallback, None
        return None, (f"no existing Teams chat with '{to}'. Provide their email "
                      f"to start a new chat.")

    def _teams_channel(self, team_name: str, channel_name: str):
        """Resolve (team_id, channel_id) by names → (tid, cid, None) or (None, None, err).
        Defaults to the team's primary (General) channel."""
        teams, err = self._graph("GET", "/me/joinedTeams?$select=id,displayName")
        if err:
            return None, None, err
        tn = team_name.lower()
        tid = None
        for t in (teams.get("value") or []):
            if tn in str(t.get("displayName", "")).lower():
                tid = t.get("id")
                break
        if not tid:
            return None, None, f"no joined team matches '{team_name}'"
        if channel_name:
            chans, err = self._graph("GET", f"/teams/{tid}/channels?$select=id,displayName")
            if err:
                return None, None, err
            cn = channel_name.lower()
            for c in (chans.get("value") or []):
                if cn in str(c.get("displayName", "")).lower():
                    return tid, c.get("id"), None
            return None, None, f"no channel '{channel_name}' in team '{team_name}'"
        prim, err = self._graph("GET", f"/teams/{tid}/primaryChannel?$select=id")
        if err:
            return None, None, err
        return tid, (prim or {}).get("id"), None

    def _todo_default_list(self):
        """Resolve the user's default To Do list id → (id, None) or (None, err)."""
        data, err = self._graph("GET", "/me/todo/lists?$select=id,wellknownListName,displayName")
        if err:
            return None, err
        lists = data.get("value") or []
        for lst in lists:
            if lst.get("wellknownListName") == "defaultList":
                return lst.get("id"), None
        return (lists[0].get("id"), None) if lists else (None, "no To Do lists found")

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

"""Google Sheets connector — create spreadsheets via the Sheets API.

"Make a spreadsheet of …" as one call (create + optional initial rows),
returning the link. Dormant until the shared GoogleClient is authorized
with the spreadsheets scope.

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import Connector, connector_result
from .google_client import GoogleClient


class GoogleSheetsConnector(Connector):
    id = "gsheets"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _svc(self):
        return self._client.service("sheets", "v4")

    def available(self) -> bool:
        try:
            return self._client.ready()
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "sheets_create",
                "description": (
                    "Create a new Google Sheet with a title and optional "
                    "initial rows. Returns {created, id, title, link}; "
                    "chain id into sheets_append_rows to add more rows later."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "rows": {
                            "type": "array",
                            "description": "Optional rows; each row is a list "
                                           "of cell values, written starting "
                                           "at A1.",
                            "items": {"type": "array",
                                      "items": {"type": "string"}},
                        },
                    },
                    "required": ["title"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "sheets_append_rows",
                "description": (
                    "Append rows to the bottom of an existing Google Sheet. "
                    "Use after sheets_create (chain id via {step:N.id}) or "
                    "with any known spreadsheet_id. Rows are inserted after "
                    "the last row with data — no manual range math needed. "
                    "Use `sheet` arg to target a specific tab name (default: "
                    "first sheet)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "rows": {
                            "type": "array",
                            "items": {"type": "array",
                                      "items": {"type": "string"}},
                        },
                        "sheet": {"type": "string",
                                  "description": "Sheet/tab name. Default: "
                                                 "first sheet."},
                    },
                    "required": ["spreadsheet_id", "rows"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Google Sheets not authorized",
                                    code="not_ready")

        if name == "sheets_create":
            title = str(args.get("title") or "").strip()
            if not title:
                return connector_result("error", error="title is required")
            try:
                created = svc.spreadsheets().create(
                    body={"properties": {"title": title}},
                    fields="spreadsheetId,spreadsheetUrl",
                ).execute()
                sid = created.get("spreadsheetId")
                rows = args.get("rows")
                if sid and isinstance(rows, list) and rows:
                    svc.spreadsheets().values().update(
                        spreadsheetId=sid, range="A1", valueInputOption="RAW",
                        body={"values": rows},
                    ).execute()
                return connector_result("ok", created=True, id=sid, title=title,
                                        link=created.get("spreadsheetUrl"))
            except Exception as exc:
                return connector_result("error",
                                        error=f"{type(exc).__name__}: {exc}")

        if name == "sheets_append_rows":
            sid = str(args.get("spreadsheet_id") or "").strip()
            rows = args.get("rows")
            if not sid:
                return connector_result("error", error="spreadsheet_id is required")
            if not isinstance(rows, list) or not rows:
                return connector_result("error", error="rows is required")
            sheet = str(args.get("sheet") or "").strip()
            # values().append with a range like 'Sheet1!A1' tells the API
            # 'find the table starting near here and append below the last
            # data row'. INSERT_ROWS prevents overwriting; USER_ENTERED
            # lets formulas evaluate as the user would expect.
            target_range = f"{sheet}!A1" if sheet else "A1"
            try:
                resp = svc.spreadsheets().values().append(
                    spreadsheetId=sid, range=target_range,
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body={"values": rows},
                ).execute()
                updates = resp.get("updates") or {}
                return connector_result(
                    "ok", appended=True, id=sid,
                    rows_added=updates.get("updatedRows") or len(rows),
                    range=updates.get("updatedRange"),
                    link=(f"https://docs.google.com/spreadsheets/d/{sid}/edit"))
            except Exception as exc:
                return connector_result("error",
                                        error=f"{type(exc).__name__}: {exc}")

        return connector_result("error",
                                error=f"unknown sheets tool: {name}",
                                code="no_handler")

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
        return [{
            "type": "function",
            "name": "sheets_create",
            "description": ("Create a new Google Sheet with a title and optional "
                            "initial rows. Returns the spreadsheet link."),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "rows": {
                        "type": "array",
                        "description": "Optional rows; each row is a list of cell "
                                       "values, written starting at A1.",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        }]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name != "sheets_create":
            return connector_result("error", error=f"unknown sheets tool: {name}", code="no_handler")
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Google Sheets not authorized", code="not_ready")
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
            return connector_result("error", error=f"{type(exc).__name__}: {exc}")

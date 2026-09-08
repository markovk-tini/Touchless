"""Google Sheets connector — create spreadsheets via the Sheets API.

"Make a spreadsheet of …" as one call (create + optional initial rows),
returning the link. Dormant until the shared GoogleClient is authorized
with the spreadsheets scope.

Author: Konstantin Markov
"""
from __future__ import annotations

import concurrent.futures
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import Connector, connector_result, friendly_api_error
from .google_client import GoogleClient

_SHEETS_CALL_TIMEOUT_SEC = 25.0
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="gsheets")

# A1-notation: optional `Sheet!` prefix + LetterRow[:LetterRow].
_A1_RE = re.compile(
    r"^(?:[^!]+!)?[A-Z]+\d+(?::[A-Z]+\d+)?$",
    re.IGNORECASE,
)
# Extract the spreadsheet id from a docs.google.com URL.
_SHEET_ID_RE = re.compile(r"/spreadsheets/d/([A-Za-z0-9_\-]+)")


def _extract_sid_from_link(link: str) -> Optional[str]:
    """Pull the spreadsheet id out of a docs.google.com URL. None when
    the link isn't a recognised Sheets URL."""
    if not link:
        return None
    m = _SHEET_ID_RE.search(str(link))
    return m.group(1) if m else None


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
                    "Create a new Google Sheet. The `title` is the "
                    "spreadsheet NAME ONLY — strip phrases like 'with "
                    "header X,Y,Z' or 'with columns ...' from the title "
                    "and put those column names as the FIRST row in "
                    "`rows` (e.g. user says 'titled iris budget test "
                    "with header name, amount, date' -> title='iris "
                    "budget test', rows=[['name','amount','date']]). "
                    "Use `rows` for any initial header row or seed "
                    "data; cells go starting at A1. Returns "
                    "{created, id, title, link}; chain id into "
                    "sheets_append_rows to add more rows later."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Spreadsheet name only. Do "
                                           "NOT include 'with header "
                                           "...' / 'with columns ...' "
                                           "clauses.",
                        },
                        "rows": {
                            "type": "array",
                            "description": "Optional rows; each row is a list "
                                           "of cell values, written starting "
                                           "at A1. First row is typically the "
                                           "header row.",
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
            {
                "type": "function",
                "name": "sheets_update_range",
                "description": (
                    "Write values to a SPECIFIC cell or range in an "
                    "existing Google Sheet (e.g. A1, B2:C5, "
                    "Sheet2!A1). Use this — not sheets_append_rows — "
                    "when the user names a cell like 'put X in A1' or "
                    "'write to B1'. append always goes after the last "
                    "data row; this overwrites the exact range you "
                    "specify."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "sheet_name": {
                            "type": "string",
                            "description": "Optional friendly name of "
                                           "the spreadsheet (e.g. 'Q4 "
                                           "plan'). Used to resolve "
                                           "spreadsheet_id when not "
                                           "provided — falls back to "
                                           "this-session artifact "
                                           "tracker, then a Drive "
                                           "search.",
                        },
                        "range": {
                            "type": "string",
                            "description": "A1 notation, e.g. 'A1', "
                                           "'B1:C3', 'Sheet1!A1'.",
                        },
                        "values": {
                            "type": "array",
                            "description": "2-D list of cell values. "
                                           "For a single cell A1=Test1: "
                                           "[['Test1']]. For A1=Test1, "
                                           "B1=Test2: [['Test1','Test2']].",
                            "items": {"type": "array",
                                      "items": {"type": "string"}},
                        },
                    },
                    "required": ["range", "values"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "sheets_read_range",
                "description": (
                    "Read values from a SPECIFIC cell or range in an "
                    "existing Google Sheet (e.g. A2, A2:C5, "
                    "Sheet2!A1:B10). Use this when the user asks 'what's "
                    "in A2?' / 'read column A' / 'show me the header row'. "
                    "Returns the 2-D values grid plus a short human "
                    "summary."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "sheet_name": {
                            "type": "string",
                            "description": "Optional friendly name of "
                                           "the spreadsheet (e.g. 'Q4 "
                                           "plan'). Used to resolve "
                                           "spreadsheet_id when not "
                                           "provided — falls back to "
                                           "this-session artifact "
                                           "tracker, then a Drive "
                                           "search.",
                        },
                        "range": {
                            "type": "string",
                            "description": "A1 notation, e.g. 'A2', "
                                           "'A2:C5', 'Sheet2!A1:B10'.",
                        },
                    },
                    "required": ["range"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "sheets_clear_range",
                "description": (
                    "Clear values from a SPECIFIC cell or range in an "
                    "existing Google Sheet (e.g. A2, A2:C5, "
                    "Sheet2!A1:B10). Writes empty content — does NOT "
                    "delete rows or columns. Use when the user asks "
                    "'clear A2' / 'wipe column B' / 'empty the header "
                    "row'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "sheet_name": {
                            "type": "string",
                            "description": "Optional friendly name of "
                                           "the spreadsheet (e.g. 'Q4 "
                                           "plan'). Used to resolve "
                                           "spreadsheet_id when not "
                                           "provided — falls back to "
                                           "this-session artifact "
                                           "tracker, then a Drive "
                                           "search.",
                        },
                        "range": {
                            "type": "string",
                            "description": "A1 notation, e.g. 'A2', "
                                           "'A2:C5', 'Sheet2!A1:B10'.",
                        },
                    },
                    "required": ["range"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _pool.submit(self._execute_locked, name, args).result(
                timeout=_SHEETS_CALL_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            return connector_result(
                "error",
                error=f"{name} timed out after {_SHEETS_CALL_TIMEOUT_SEC}s",
                code="timeout",
            )

    def _execute_locked(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
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
                # Warm the picker cache so a subsequent 'append to X' by
                # title hits without triggering the Picker fallback.
                if sid:
                    try:
                        from .google_picker_cache import shared as _picker_cache
                        _picker_cache().remember(
                            title, "sheet", sid, title,
                            "application/vnd.google-apps.spreadsheet")
                    except Exception:
                        pass
                return connector_result("ok", created=True, id=sid, title=title,
                                        link=created.get("spreadsheetUrl"))
            except Exception as exc:
                return connector_result("error",
                                        error=friendly_api_error(exc, api_label="Google Sheets"))

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
                                        error=friendly_api_error(exc, api_label="Google Sheets"))

        if name == "sheets_update_range":
            sid = str(args.get("spreadsheet_id") or "").strip()
            sheet_name = str(args.get("sheet_name") or "").strip()
            rng = str(args.get("range") or "").strip()
            values = args.get("values")
            # When sid is missing, try to resolve from sheet_name via the
            # artifact tracker (this-session create) and then Drive search.
            if not sid and sheet_name:
                sid, resolve_err = self._resolve_sid_by_name(sheet_name)
                if resolve_err and not sid:
                    return connector_result(
                        "error", error=resolve_err, code="need_spreadsheet")
            if not sid:
                hint = (f" (couldn't find a sheet matching '{sheet_name}')"
                        if sheet_name else "")
                return connector_result(
                    "error",
                    error=("I need the spreadsheet — say its name or paste "
                           "the URL" + hint),
                    code="need_spreadsheet")
            if not rng:
                return connector_result(
                    "error",
                    error=("range is required (e.g. 'A2' or 'Sheet1!A2:B2')"),
                    code="bad_range")
            if not _A1_RE.match(rng):
                return connector_result(
                    "error",
                    error=(f"range '{rng}' isn't a valid A1 reference "
                           "(e.g. A2 or Sheet1!A2:B2)"),
                    code="bad_range")
            # Common LLM mistake: a flat list ['Test 1','Test 2'] for a
            # single row. Coerce rather than reject so the user's intent
            # still wins.
            if (isinstance(values, list) and values
                    and all(not isinstance(v, list) for v in values)):
                values = [list(values)]
            if (not isinstance(values, list) or not values
                    or not all(isinstance(row, list) for row in values)):
                return connector_result(
                    "error",
                    error=("values must be a 2-D list, e.g. "
                           "[['Test1','Test2']] for a single row"),
                    code="bad_values")
            try:
                resp = svc.spreadsheets().values().update(
                    spreadsheetId=sid, range=rng,
                    valueInputOption="USER_ENTERED",
                    body={"values": values},
                ).execute()
                return connector_result(
                    "ok", updated=True, id=sid,
                    range=resp.get("updatedRange"),
                    cells=resp.get("updatedCells"),
                    link=(f"https://docs.google.com/spreadsheets/d/{sid}/edit"))
            except Exception as exc:
                return connector_result(
                    "error", error=self._friendly_http_error(exc, sid))

        if name in ("sheets_read_range", "sheets_clear_range"):
            sid = str(args.get("spreadsheet_id") or "").strip()
            sheet_name = str(args.get("sheet_name") or "").strip()
            rng = str(args.get("range") or "").strip()
            if not sid and sheet_name:
                sid, resolve_err = self._resolve_sid_by_name(sheet_name)
                if resolve_err and not sid:
                    return connector_result(
                        "error", error=resolve_err, code="need_spreadsheet")
            if not sid:
                hint = (f" (couldn't find a sheet matching '{sheet_name}')"
                        if sheet_name else "")
                return connector_result(
                    "error",
                    error=("I need the spreadsheet — say its name or paste "
                           "the URL" + hint),
                    code="need_spreadsheet")
            if not rng:
                return connector_result(
                    "error",
                    error=("range is required (e.g. 'A2' or 'Sheet1!A2:B2')"),
                    code="bad_range")
            if not _A1_RE.match(rng):
                return connector_result(
                    "error",
                    error=(f"range '{rng}' isn't a valid A1 reference "
                           "(e.g. A2 or Sheet1!A2:B2)"),
                    code="bad_range")
            where = f" in '{sheet_name}'" if sheet_name else ""
            if name == "sheets_read_range":
                try:
                    resp = svc.spreadsheets().values().get(
                        spreadsheetId=sid, range=rng,
                    ).execute()
                    values = resp.get("values") or []
                    rng_used = resp.get("range") or rng
                    summary = self._summarize_read(rng_used, values, where)
                    return connector_result(
                        "ok", id=sid, range=rng_used, values=values,
                        summary=summary,
                        link=(f"https://docs.google.com/spreadsheets/d/{sid}"
                              "/edit"))
                except Exception as exc:
                    return connector_result(
                        "error", error=self._friendly_http_error(exc, sid))
            # sheets_clear_range
            try:
                resp = svc.spreadsheets().values().clear(
                    spreadsheetId=sid, range=rng, body={},
                ).execute()
                cleared_range = resp.get("clearedRange") or rng
                return connector_result(
                    "ok", cleared=True, id=sid, range=cleared_range,
                    summary=f"Cleared {cleared_range}{where}",
                    link=(f"https://docs.google.com/spreadsheets/d/{sid}/edit"))
            except Exception as exc:
                return connector_result(
                    "error", error=self._friendly_http_error(exc, sid))

        return connector_result("error",
                                error=f"unknown sheets tool: {name}",
                                code="no_handler")

    # ---- name → spreadsheet_id resolver --------------------------------
    def _resolve_sid_by_name(self, name: str
                              ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve a friendly sheet name to a spreadsheet id.

        Ladder:
          (1) orchestrator's per-session artifact tracker — a sheet just
              created this session is almost always what the user means,
          (2) PickerCache — file the user picked via Google Picker in a
              prior session (drive.file scope makes this the primary
              cross-session recall path since Drive.list only returns
              app-created files under that scope),
          (3) Drive search by name — still works for files THIS APP
              created (drive.file grants self-created access); returns
              [] for the user's other Sheets under drive.file.
        Returns (sid, error_message). When multiple Drive matches are
        equally plausible, returns (None, ambiguity-message)."""
        # (1) Artifact tracker (this-session create) — quick, free, and
        # most likely to be right immediately after sheets_create.
        try:
            from ..planner import current_planner_artifact_lookup
            lookup = current_planner_artifact_lookup()
            if callable(lookup):
                hit = lookup(name, kind="sheet")
                if hit:
                    sid = _extract_sid_from_link(str(hit.get("link") or ""))
                    if sid:
                        return sid, None
        except Exception:
            pass
        # (2) PickerCache — prior-session picks keyed by slug. Cache
        # miss is silent (returns None); worst case we fall through to
        # the Drive search below. Import lazily so a dormant install
        # with no Google dep still loads this module.
        try:
            from .google_picker_cache import shared as _picker_cache
            cached = _picker_cache().lookup(name, kind="sheet")
            if cached and cached.get("file_id"):
                return cached["file_id"], None
        except Exception:
            pass
        # (3) Drive search. Quote the name conservatively so a sheet
        # called "Q4 plan" doesn't break the query.
        try:
            drive = self._client.service("drive", "v3")
        except Exception:
            drive = None
        if drive is None:
            return None, None
        try:
            escaped = name.replace("\\", "\\\\").replace("'", "\\'")
            resp = drive.files().list(
                q=(f"mimeType='application/vnd.google-apps.spreadsheet' "
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
        # Multiple matches — prefer an exact (case-insensitive) name
        # match if one exists, else surface the ambiguity to the user.
        exact = [f for f in files
                 if str(f.get("name") or "").lower() == name.lower()]
        if len(exact) == 1:
            return exact[0].get("id"), None
        top = ", ".join(f"'{f.get('name')}'" for f in files[:3])
        return None, (f"Multiple sheets match '{name}': {top}. "
                      "Say a more specific name.")

    @staticmethod
    def _shift_col(col_letters: str, offset: int) -> str:
        """Return the column letters `offset` columns right of `col_letters`.
        _shift_col('A', 0) == 'A'; _shift_col('A', 1) == 'B';
        _shift_col('Z', 1) == 'AA'."""
        n = 0
        for ch in col_letters.upper():
            n = n * 26 + (ord(ch) - ord("A") + 1)
        n = (n - 1) + offset  # zero-based, then shift
        if n < 0:
            n = 0
        out = ""
        while True:
            out = chr(ord("A") + (n % 26)) + out
            n = n // 26 - 1
            if n < 0:
                break
        return out

    @classmethod
    def _summarize_read(cls, rng_used: str,
                        values: List[List[Any]], where: str = "") -> str:
        """Build a short 'A2 = 'John' | A3 = 'Dani' | ...' summary of a
        values.get() response. Falls back gracefully when the range has
        an unexpected shape."""
        if not values or all(not r for r in values):
            return f"{rng_used}{where} is empty"
        cell_part = rng_used.split("!", 1)[-1]
        start_cell = cell_part.split(":", 1)[0]
        m = re.match(r"^([A-Za-z]+)(\d+)$", start_cell)
        if not m:
            flat = [str(c) for row in values for c in row]
            head = " | ".join(flat[:8])
            tail = " | ..." if len(flat) > 8 else ""
            return head + tail
        start_col = m.group(1).upper()
        start_row = int(m.group(2))
        parts: List[str] = []
        total = 0
        for r_i, row in enumerate(values):
            for c_i, val in enumerate(row):
                total += 1
                if len(parts) < 8:
                    addr = cls._shift_col(start_col, c_i) + str(start_row + r_i)
                    parts.append(f"{addr} = {val!r}")
        tail = " | ..." if total > len(parts) else ""
        return " | ".join(parts) + tail

    @staticmethod
    def _friendly_http_error(exc: Exception, sid: str) -> str:
        """Translate a googleapiclient HttpError into a user-readable
        message. Falls back to the standard type+message for non-HTTP
        exceptions."""
        status: Optional[int] = None
        try:
            resp = getattr(exc, "resp", None)
            if resp is not None:
                status = int(getattr(resp, "status", 0)) or None
        except Exception:
            status = None
        if status == 403:
            return ("Sheets API denied write access — re-auth with the "
                    "sheets scope")
        if status == 404:
            return ("That spreadsheet no longer exists or isn't shared "
                    "with the signed-in account")
        return f"{type(exc).__name__}: {exc}"

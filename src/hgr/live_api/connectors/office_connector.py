"""Office connector — API-first Word / Excel / PowerPoint tasks via COM.

Wraps OfficeController. These are *specific in-document tasks* (create,
type, set a cell, add a slide, save as PDF) done with one deterministic
COM call instead of clicking Office's ribbon — the deep-control upgrade
over the GUI computer-use fallback.

Availability is per-app: a tool set is only exposed for an Office app that
is actually installed (resolved from its ProgID, without launching it). So
on a machine with only Word, the Excel/PowerPoint tools simply don't appear.

OneNote is intentionally omitted — its COM model is XML-based and brittle;
deep OneNote tasks belong on the Microsoft Graph API path (future), and
`open_app('onenote')` already covers launching it.

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Connector, connector_result


class OfficeConnector(Connector):
    id = "office"

    def __init__(self, controller: Optional[Any] = None) -> None:
        self._controller = controller

    def _ctrl(self):
        if self._controller is None:
            from ...debug.office_controller import OfficeController
            self._controller = OfficeController()
        return self._controller

    def available(self) -> bool:
        try:
            return bool(self._ctrl().any_available())
        except Exception:
            return False

    @staticmethod
    def _fn(name: str, desc: str, props: Dict[str, Any] | None = None,
            required: List[str] | None = None) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props or {},
                "required": required or [],
                "additionalProperties": False,
            },
        }

    def tools(self) -> List[Dict[str, Any]]:
        c = self._ctrl()
        fn = self._fn
        out: List[Dict[str, Any]] = []

        # Word — only if installed.
        if c.app_available("word"):
            out += [
                fn("word_new_document",
                   "Create a new Word document, optionally pre-filled with text.",
                   {"text": {"type": "string", "description": "Initial body text."}}),
                fn("word_append_text",
                   "Append text to the end of the active Word document.",
                   {"text": {"type": "string"}}, ["text"]),
                fn("word_open", "Open an existing Word document by file path.",
                   {"path": {"type": "string"}}, ["path"]),
                fn("word_save_as",
                   "Save the active Word document to a path. Set pdf=true to "
                   "export as PDF.",
                   {"path": {"type": "string"},
                    "pdf": {"type": "boolean", "default": False}}, ["path"]),
            ]

        # Excel — only if installed.
        if c.app_available("excel"):
            out += [
                fn("excel_new_workbook", "Create a new Excel workbook."),
                fn("excel_set_cell",
                   "Set a cell value in the active Excel sheet (e.g. cell 'B2').",
                   {"cell": {"type": "string", "description": "A1-style ref, e.g. 'B2'."},
                    "value": {"type": "string"}}, ["cell", "value"]),
                fn("excel_save_as",
                   "Save the active Excel workbook to a path. Set pdf=true to "
                   "export as PDF.",
                   {"path": {"type": "string"},
                    "pdf": {"type": "boolean", "default": False}}, ["path"]),
            ]

        # PowerPoint — only if installed.
        if c.app_available("powerpoint"):
            out += [
                fn("ppt_new_presentation", "Create a new PowerPoint presentation."),
                fn("ppt_add_slide",
                   "Add a slide (title + body) to the active presentation.",
                   {"title": {"type": "string"}, "body": {"type": "string"}}),
                fn("ppt_save_as",
                   "Save the active presentation to a path. Set pdf=true to "
                   "export as PDF.",
                   {"path": {"type": "string"},
                    "pdf": {"type": "boolean", "default": False}}, ["path"]),
            ]
        return out

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        c = self._ctrl()

        def _txt(key: str) -> Optional[str]:
            v = args.get(key)
            return None if v is None else str(v)

        def _ok(result: bool, **extra: Any) -> Dict[str, Any]:
            return connector_result(
                "ok" if result else "error",
                error=None if result else c.message,
                **extra,
            )

        if name == "word_new_document":
            return _ok(c.word_new_document(_txt("text")))
        if name == "word_append_text":
            text = _txt("text")
            if not text:
                return connector_result("error", error="text is required")
            return _ok(c.word_append_text(text))
        if name == "word_open":
            path = (_txt("path") or "").strip()
            if not path:
                return connector_result("error", error="path is required")
            return _ok(c.word_open(path), path=path)
        if name == "word_save_as":
            path = (_txt("path") or "").strip()
            if not path:
                return connector_result("error", error="path is required")
            return _ok(c.word_save_as(path, as_pdf=bool(args.get("pdf"))), path=path)

        if name == "excel_new_workbook":
            return _ok(c.excel_new_workbook())
        if name == "excel_set_cell":
            cell = (_txt("cell") or "").strip()
            if not cell:
                return connector_result("error", error="cell is required")
            return _ok(c.excel_set_cell(cell, _txt("value") or ""), cell=cell)
        if name == "excel_save_as":
            path = (_txt("path") or "").strip()
            if not path:
                return connector_result("error", error="path is required")
            return _ok(c.excel_save_as(path, as_pdf=bool(args.get("pdf"))), path=path)

        if name == "ppt_new_presentation":
            return _ok(c.ppt_new_presentation())
        if name == "ppt_add_slide":
            return _ok(c.ppt_add_slide(_txt("title"), _txt("body")))
        if name == "ppt_save_as":
            path = (_txt("path") or "").strip()
            if not path:
                return connector_result("error", error="path is required")
            return _ok(c.ppt_save_as(path, as_pdf=bool(args.get("pdf"))), path=path)

        return connector_result("error", error=f"unknown office tool: {name}", code="no_handler")

"""Microsoft Office automation via COM (Word / Excel / PowerPoint).

The deterministic, API-first path for in-document tasks — create a doc,
type text, set a cell, add a slide, save as PDF — without driving the
Office UI by screenshot-and-click.

Implementation notes
--------------------
* Uses ``comtypes`` (already a dependency; pywin32 is intentionally NOT
  added) with **dynamic** IDispatch binding, so no typelib generation is
  needed and it tolerates differing Office versions.
* COM is apartment-threaded: every public method calls ``_com()`` first,
  which runs ``CoInitialize()`` on the *calling* thread (the Live API
  worker), mirroring VolumeController / UiaController.
* Availability is resolved from the **ProgID** (``Word.Application`` etc.)
  via the registry — we never launch an Office app just to answer
  ``available()``.
* Apps are launched ``Visible = True`` so the user sees and reviews the
  result; nothing is silently sent/printed.

Author: Konstantin Markov
"""
from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

# Office COM format constants (stable across versions).
_WD_FORMAT_PDF = 17        # wdFormatPDF
_WD_FORMAT_DOCX = 16       # wdFormatDocumentDefault
_XL_FORMAT_XLSX = 51       # xlOpenXMLWorkbook
_PP_SAVE_AS_PDF = 32       # ppSaveAsPDF
_PP_SAVE_AS_DEFAULT = 11   # ppSaveAsDefault (pptx)
_PP_LAYOUT_TEXT = 2        # ppLayoutText (title + body)


def _progid_registered(progid: str) -> bool:
    """True if a COM ProgID resolves to a CLSID (i.e. the app is installed).
    Does not instantiate the server."""
    if platform.system() != "Windows":
        return False
    try:
        from comtypes import GUID
        GUID.from_progid(progid)
        return True
    except Exception:
        return False


class OfficeController:
    # ProgID per app — used for both availability and CreateObject.
    PROGIDS = {
        "word": "Word.Application",
        "excel": "Excel.Application",
        "powerpoint": "PowerPoint.Application",
    }

    def __init__(self) -> None:
        self._available = platform.system() == "Windows"
        self._message = "office idle"
        # Cached per-app Application objects (bound on this thread).
        self._apps: dict[str, object] = {}

    @property
    def message(self) -> str:
        return self._message

    def app_available(self, app: str) -> bool:
        """Whether a specific Office app is installed."""
        progid = self.PROGIDS.get(app)
        return bool(self._available and progid and _progid_registered(progid))

    def any_available(self) -> bool:
        return any(self.app_available(a) for a in self.PROGIDS)

    # ---- COM plumbing -------------------------------------------------
    def _com(self) -> bool:
        if not self._available:
            return False
        try:
            from comtypes import CoInitialize
            CoInitialize()
            return True
        except Exception:
            return False

    def _get_app(self, app: str):
        """Connect to a running Office app instance or start a fresh one.
        Returns the Application COM object or None."""
        if not self._com():
            return None
        progid = self.PROGIDS.get(app)
        if not progid:
            return None
        cached = self._apps.get(app)
        if cached is not None:
            try:
                _ = cached.Name  # liveness probe; raises if it was closed
                return cached
            except Exception:
                self._apps.pop(app, None)
        try:
            import comtypes.client
            try:
                obj = comtypes.client.GetActiveObject(progid, dynamic=True)
            except Exception:
                obj = comtypes.client.CreateObject(progid, dynamic=True)
            try:
                obj.Visible = True
            except Exception:
                pass
            self._apps[app] = obj
            return obj
        except Exception as exc:
            self._message = f"could not start {app}: {type(exc).__name__}: {exc}"
            return None

    @staticmethod
    def _norm_path(path: str) -> str:
        return str(Path(path).expanduser())

    # ---- Word ---------------------------------------------------------
    def word_new_document(self, text: Optional[str] = None) -> bool:
        app = self._get_app("word")
        if app is None:
            return False
        try:
            doc = app.Documents.Add()
            if text:
                doc.Content.Text = text
            self._message = "created Word document"
            return True
        except Exception as exc:
            self._message = f"word_new_document failed: {type(exc).__name__}: {exc}"
            return False

    def word_append_text(self, text: str) -> bool:
        app = self._get_app("word")
        if app is None:
            return False
        try:
            doc = app.ActiveDocument
            rng = doc.Content
            rng.Collapse(0)  # wdCollapseEnd
            rng.InsertAfter(text)
            self._message = "appended text to Word document"
            return True
        except Exception as exc:
            self._message = f"word_append_text failed: {type(exc).__name__}: {exc}"
            return False

    def word_save_as(self, path: str, *, as_pdf: bool = False) -> bool:
        app = self._get_app("word")
        if app is None:
            return False
        try:
            fmt = _WD_FORMAT_PDF if as_pdf else _WD_FORMAT_DOCX
            app.ActiveDocument.SaveAs2(self._norm_path(path), fmt)
            self._message = f"saved Word document to {path}"
            return True
        except Exception as exc:
            self._message = f"word_save_as failed: {type(exc).__name__}: {exc}"
            return False

    def word_open(self, path: str) -> bool:
        app = self._get_app("word")
        if app is None:
            return False
        try:
            app.Documents.Open(self._norm_path(path))
            self._message = f"opened Word document {path}"
            return True
        except Exception as exc:
            self._message = f"word_open failed: {type(exc).__name__}: {exc}"
            return False

    # ---- Excel --------------------------------------------------------
    def excel_new_workbook(self) -> bool:
        app = self._get_app("excel")
        if app is None:
            return False
        try:
            app.Workbooks.Add()
            self._message = "created Excel workbook"
            return True
        except Exception as exc:
            self._message = f"excel_new_workbook failed: {type(exc).__name__}: {exc}"
            return False

    def excel_set_cell(self, cell: str, value: str) -> bool:
        app = self._get_app("excel")
        if app is None:
            return False
        try:
            if not app.Workbooks.Count:
                app.Workbooks.Add()
            app.ActiveSheet.Range(cell).Value = value
            self._message = f"set {cell} = {value}"
            return True
        except Exception as exc:
            self._message = f"excel_set_cell failed: {type(exc).__name__}: {exc}"
            return False

    def excel_save_as(self, path: str, *, as_pdf: bool = False) -> bool:
        app = self._get_app("excel")
        if app is None:
            return False
        try:
            wb = app.ActiveWorkbook
            if as_pdf:
                wb.ExportAsFixedFormat(0, self._norm_path(path))  # xlTypePDF=0
            else:
                wb.SaveAs(self._norm_path(path), _XL_FORMAT_XLSX)
            self._message = f"saved Excel workbook to {path}"
            return True
        except Exception as exc:
            self._message = f"excel_save_as failed: {type(exc).__name__}: {exc}"
            return False

    # ---- PowerPoint ---------------------------------------------------
    def ppt_new_presentation(self) -> bool:
        app = self._get_app("powerpoint")
        if app is None:
            return False
        try:
            app.Presentations.Add()
            self._message = "created PowerPoint presentation"
            return True
        except Exception as exc:
            self._message = f"ppt_new_presentation failed: {type(exc).__name__}: {exc}"
            return False

    def ppt_add_slide(self, title: Optional[str] = None,
                      body: Optional[str] = None) -> bool:
        app = self._get_app("powerpoint")
        if app is None:
            return False
        try:
            if not app.Presentations.Count:
                app.Presentations.Add()
            pres = app.ActivePresentation
            index = pres.Slides.Count + 1
            slide = pres.Slides.Add(index, _PP_LAYOUT_TEXT)
            if title is not None:
                try:
                    slide.Shapes.Title.TextFrame.TextRange.Text = title
                except Exception:
                    pass
            if body is not None:
                try:
                    slide.Shapes.Placeholders(2).TextFrame.TextRange.Text = body
                except Exception:
                    pass
            self._message = f"added slide {index}"
            return True
        except Exception as exc:
            self._message = f"ppt_add_slide failed: {type(exc).__name__}: {exc}"
            return False

    def ppt_save_as(self, path: str, *, as_pdf: bool = False) -> bool:
        app = self._get_app("powerpoint")
        if app is None:
            return False
        try:
            pres = app.ActivePresentation
            fmt = _PP_SAVE_AS_PDF if as_pdf else _PP_SAVE_AS_DEFAULT
            pres.SaveAs(self._norm_path(path), fmt)
            self._message = f"saved presentation to {path}"
            return True
        except Exception as exc:
            self._message = f"ppt_save_as failed: {type(exc).__name__}: {exc}"
            return False

# Author: Konstantin Markov

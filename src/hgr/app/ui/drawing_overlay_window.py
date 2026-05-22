"""Frameless, click-through overlay that shows a saved drawing PNG on
top of every other window.

Used by the "show_overlay_drawing" custom-gesture action: the user
binds a gesture to a saved PNG filename, and firing the gesture
toggles this overlay over their current screen content. Strokes
appear "in mid-air" because the saved drawing is transparent (see
DrawOverlay.save_canvas_snapshot) and this window is itself
WA_TranslucentBackground + WA_TransparentForMouseEvents — clicks
pass straight through to whatever app is underneath.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QPainter, QPixmap
from PySide6.QtWidgets import QWidget


# Image extensions the show_overlay_drawing action knows how to
# render. Restricting the filesystem search to these stops the
# disambiguation prompt from offering, say, a Word document that
# happens to share a basename.
_DRAWING_SEARCH_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Hard ceiling on filesystem-search results. Beyond this either the
# filename is too generic (user should rename) or something is wrong
# with their indexed scope. Either way listing 50 paths in a chooser
# UI is worse than listing 10.
_DRAWING_SEARCH_MAX_RESULTS = 10


class DrawingOverlayWindow(QWidget):
    """Always-on-top, click-through PNG viewer.

    Single instance per app — main_window owns one and toggles it
    via show_image / hide. Mirrors the HelloOverlay flag pattern so
    the OS treats it the same way (no taskbar entry, no focus steal).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        flags = Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        # WindowTransparentForInput is the OS-level click-through flag
        # (added in Qt 5.6+). Combined with WA_TransparentForMouseEvents
        # below, mouse events never reach the overlay AND the OS routes
        # input straight to the window underneath — no flicker or
        # input-loss when the cursor passes over the overlay's region.
        transparent_flag = getattr(Qt, "WindowTransparentForInput", None)
        if transparent_flag is not None:
            flags |= transparent_flag
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._pixmap: Optional[QPixmap] = None
        self._current_path: Optional[str] = None
        # Live grab transform applied during pinch-drag. Stored in
        # NORMALISED screen units so a delta computed in
        # [0,1] palm coords lines up with the same delta on screen
        # regardless of which monitor the overlay is stretched
        # across. Scale is a multiplier on the auto-fit base size.
        self._translate_x_norm: float = 0.0
        self._translate_y_norm: float = 0.0
        self._scale: float = 1.0

    def show_image(self, path: str) -> bool:
        """Load `path` and show the overlay sized + positioned to
        cover the union of all screens. The drawing's transparent
        pixels stay transparent — only the user's strokes are
        visible. Returns False if the file couldn't be loaded.
        Resets any live grab transform so a freshly-shown drawing
        starts centered at its natural fit."""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        self._pixmap = pixmap
        self._current_path = path
        self.reset_transform()
        self._fit_to_virtual_screen()
        self.show()
        self.raise_()
        self.update()
        return True

    @property
    def current_path(self) -> Optional[str]:
        return self._current_path

    def set_grab_transform(self, dx_norm: float, dy_norm: float, scale: float) -> None:
        """Apply a translate + scale transform to the displayed
        drawing. dx/dy are in normalised screen units ([-1, 1] = full
        screen width/height); scale is a multiplier on the auto-fit
        base size. Engine pinch-grab handler calls this every frame
        while a pinch is held; the overlay locks in whatever value
        was last set when the gesture releases."""
        self._translate_x_norm = float(dx_norm)
        self._translate_y_norm = float(dy_norm)
        # Clamp scale to a sensible range so accidental rapid
        # bimanual movement can't shrink the drawing into a single
        # pixel or blow it up off screen.
        self._scale = max(0.1, min(10.0, float(scale)))
        self.update()

    def reset_transform(self) -> None:
        self._translate_x_norm = 0.0
        self._translate_y_norm = 0.0
        self._scale = 1.0
        self.update()

    def _fit_to_virtual_screen(self) -> None:
        """Stretch over every connected display so the strokes
        appear at the same screen coordinates the user drew them
        at, regardless of which monitor they were on."""
        screens = [s for s in QGuiApplication.screens() if s is not None]
        if not screens:
            self.resize(1200, 800)
            return
        union = screens[0].geometry()
        for screen in screens[1:]:
            union = union.united(screen.geometry())
        self.setGeometry(union)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        if self._pixmap is None or self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        # Auto-fit base size: scale the pixmap to fill the overlay
        # while preserving aspect ratio. The grab transform then
        # multiplies that base size and shifts it; we never re-scale
        # the source pixmap to avoid compounding interpolation
        # artefacts as the user drags + scales repeatedly.
        target = self.rect()
        base = self._pixmap.scaled(
            target.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        # Apply live scale on top of the auto-fit. scale=1.0 means
        # "show at the auto-fit size"; >1 grows, <1 shrinks. The
        # multiply happens at draw time on the already-scaled
        # `base`, so it costs one extra interpolated blit per paint
        # but never permanently degrades the source.
        if self._scale != 1.0:
            display = base.scaled(
                int(base.width() * self._scale),
                int(base.height() * self._scale),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        else:
            display = base
        # Translate offsets are normalised — convert to pixels
        # against the overlay's screen geometry so a swipe across
        # half the camera frame moves the drawing by half a screen
        # too, regardless of which monitor it's painted on.
        tx_px = int(self._translate_x_norm * target.width())
        ty_px = int(self._translate_y_norm * target.height())
        x = (target.width() - display.width()) // 2 + tx_px
        y = (target.height() - display.height()) // 2 + ty_px
        painter.drawPixmap(x, y, display)


def resolve_drawing_path(filename: str, drawings_dir: str | Path) -> Optional[Path]:
    """Resolve a user-typed drawing filename against the configured
    drawings directory. Accepts:
      - bare filename ('my_drawing.png') → joined with drawings_dir
      - relative path ('subdir/foo.png') → joined with drawings_dir
      - absolute path → used as-is
    Returns the resolved Path if the file exists, or None."""
    if not filename:
        return None
    candidate = Path(filename).expanduser()
    if not candidate.is_absolute():
        try:
            candidate = Path(drawings_dir).expanduser() / candidate
        except Exception:
            return None
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        return None
    return None


def search_drawings_by_filename(
    filename: str,
    *,
    max_results: int = _DRAWING_SEARCH_MAX_RESULTS,
) -> List[Path]:
    """Find every file on the system whose basename matches
    `filename` exactly (case-insensitive), restricted to image
    extensions. Used by the show_overlay_drawing action's
    "search anywhere" fallback when the configured
    `drawings_save_dir` doesn't contain the requested file.

    Resolution strategy:
      1. Windows Search Index (`SystemIndex`, the same backend File
         Explorer's search box uses) — fast (~50 ms) and covers every
         indexed location including OneDrive, Pictures, Documents,
         Desktop, Downloads, plus any user-added indexed folders.
      2. Bounded fallback walk of common user dirs — only runs if
         the index is unavailable (rare; the service is on by
         default) or the query raises.

    Returns absolute paths, capped at `max_results`. Empty list if
    nothing matched anywhere reachable.
    """
    if not filename:
        return []
    target = Path(filename).name
    if Path(target).suffix.lower() not in _DRAWING_SEARCH_EXTENSIONS:
        return []

    indexed = _search_via_system_index(target, max_results=max_results)
    if indexed is not None:
        return indexed
    return _fallback_walk_for_filename(target, max_results=max_results)


def _search_via_system_index(
    filename: str,
    *,
    max_results: int,
) -> Optional[List[Path]]:
    """Query the Windows Search Index for files named `filename`.
    Returns the (possibly empty) list of absolute paths on success;
    returns None if the index isn't reachable so the caller falls
    back to a directory walk."""
    try:
        import win32com.client  # type: ignore
    except Exception:
        return None

    safe = filename.replace("'", "''")
    sql = (
        "SELECT TOP %d System.ItemPathDisplay "
        "FROM SystemIndex "
        "WHERE System.FileName = '%s' AND System.Kind = 'picture'"
    ) % (max_results, safe)

    results: List[Path] = []
    conn = None
    rs = None
    try:
        conn = win32com.client.Dispatch("ADODB.Connection")
        conn.Open(
            "Provider=Search.CollatorDSO;"
            "Extended Properties='Application=Windows'"
        )
        rs = win32com.client.Dispatch("ADODB.Recordset")
        rs.Open(sql, conn)
        while not rs.EOF and len(results) < max_results:
            try:
                raw = rs.Fields.Item("System.ItemPathDisplay").Value
                if raw:
                    p = Path(str(raw))
                    if p.is_file():
                        results.append(p)
            except Exception:
                pass
            rs.MoveNext()
    except Exception:
        # Index service unavailable / SQL rejected / COM error.
        return None
    finally:
        for handle in (rs, conn):
            if handle is None:
                continue
            try:
                handle.Close()
            except Exception:
                pass
    return results


def _fallback_walk_for_filename(
    filename: str,
    *,
    max_results: int,
) -> List[Path]:
    """Bounded directory walk used when SystemIndex isn't available.
    Walks the most common user-content directories with a depth cap
    so a stray symlink loop or selectively-synced OneDrive tree
    can't run forever."""
    home = Path.home()
    bases = (
        home / "Pictures",
        home / "OneDrive" / "Pictures",
        home / "OneDrive - Personal" / "Pictures",
        home / "Documents",
        home / "Desktop",
        home / "Downloads",
    )
    target_lower = filename.lower()
    seen_roots: set[str] = set()
    found: List[Path] = []
    skip_dirs = {"appdata", "node_modules", "__pycache__", "$recycle.bin"}

    for base in bases:
        try:
            if not base.exists():
                continue
            root_key = str(base.resolve()).lower()
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)
            base_parts = len(base.parts)
            for root, dirs, files in os.walk(base, followlinks=False):
                depth = len(Path(root).parts) - base_parts
                if depth > 4:
                    dirs[:] = []
                    continue
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".") and d.lower() not in skip_dirs
                ]
                for f in files:
                    if f.lower() == target_lower:
                        candidate = Path(root) / f
                        if (candidate.is_file()
                                and candidate.suffix.lower() in _DRAWING_SEARCH_EXTENSIONS):
                            found.append(candidate)
                            if len(found) >= max_results:
                                return found
        except (OSError, PermissionError):
            continue
    return found

# Author: Konstantin Markov

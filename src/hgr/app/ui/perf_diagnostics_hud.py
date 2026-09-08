"""Perf diagnostics HUD — v1.1.7 round-32.

Small, read-only, semi-transparent overlay that surfaces the
Round-30 GestureWorker diagnostic scalars plus fps/mode flags in
the top-right corner of the home page. Pure read-side: the poll
callback touches only cheap Python attributes on the worker (no
cap.get(), no signal round-trips) so it cannot regress the
pipeline.

Round-32 delta: HUD-visibility fix. On initial-open sessions that
landed on the ffmpeg-MJPG path via _upgrade_to_ffmpeg_capture_if_
lite (Lite/GPU/Low-FPS persisted in config → PATH A), the round-31
HUD rendered "cap NonexNone@None" / "back " (empty) / "swap (none)"
because PATH A never stashed the round-30 scalars (its method-local
_log only wrote stderr). GestureWorker._stash_ffmpeg_engaged_
diagnostics now populates target + backend + swap-outcome on both
engage sites (PATH A and PATH B). A new "back {backend}" line and a
leading "PERF HUD r{round}" header surface the fix.

Design contract:
- 340x112 semi-transparent widget floating over the home `page`.
- Transparent to mouse events (never eats START/END/SETTINGS clicks).
- Parented to the home `page` widget: hides automatically when the
  QStackedWidget switches to Settings/etc.
- QTimer(500 ms) only runs while the HUD is visible — start on
  showEvent, stop on hideEvent — so a hidden home page pays zero
  polling cost.
- Every attribute read uses getattr() with a default, so a future
  rename on the worker degrades to the idle placeholder rather than
  raising AttributeError on the main thread.
- raise_() every poll so later-added siblings of `page` can never
  occlude the HUD's top-right 340x112 anchor.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from hgr import BUILD_ROUND


_HUD_W = 340
_HUD_H = 112
_HUD_MARGIN = 12
_POLL_MS = 500


class PerfDiagnosticsHud(QWidget):
    """Read-side diagnostics overlay for the home page."""

    def __init__(self, main_window, page: QWidget) -> None:
        super().__init__(page)
        self._mw = main_window
        self._page = page
        self.setObjectName("perfDiagnosticsHud")
        # Never eat clicks: START / END / SETTINGS sit under the HUD's
        # top-right anchor on narrow window widths and must remain
        # clickable.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.resize(_HUD_W, _HUD_H)
        # Fixed-width font so columns don't jitter as scalars update.
        f = QFont()
        f.setStyleHint(QFont.TypeWriter)
        f.setFamily("Consolas")
        f.setPointSize(8)
        self.setFont(f)
        # Reposition on parent resize. eventFilter guards on
        # event.type() == Resize on the FIRST line so the flood of
        # Paint/Mouse/Enter/Leave/Focus events Qt dispatches to
        # `page` costs one comparison per event.
        try:
            page.installEventFilter(self)
        except Exception:
            pass
        self._reposition()
        # Timer starts stopped; start/stop is driven by showEvent /
        # hideEvent so a hidden home page (stack on Settings) pays
        # zero polling cost.
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._on_poll)

    # -- geometry --------------------------------------------------

    def _reposition(self) -> None:
        try:
            pw = self._page.width()
            x = max(0, pw - _HUD_W - _HUD_MARGIN)
            y = _HUD_MARGIN
            self.move(x, y)
        except Exception:
            pass

    def eventFilter(self, obj: Any, event: Any) -> bool:  # type: ignore[override]
        # First-line type guard — everything else short-circuits.
        try:
            if event.type() == QEvent.Resize and obj is self._page:
                self._reposition()
        except Exception:
            pass
        return False

    # -- lifecycle -------------------------------------------------

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()
        # First tick immediately so the HUD isn't blank for 500 ms.
        self._on_poll()

    def hideEvent(self, event) -> None:  # type: ignore[override]
        # QStackedWidget hides `page` when the user goes to Settings;
        # Qt propagates hide to children, so this fires. Stop the
        # poll so we don't burn CPU on an invisible widget.
        try:
            self._timer.stop()
        except Exception:
            pass
        super().hideEvent(event)

    # -- poll ------------------------------------------------------

    def _on_poll(self) -> None:
        # raise_() every tick so later siblings of `page` added after
        # the HUD (info_card etc.) can't paint over the top-right box.
        try:
            self.raise_()
        except Exception:
            pass
        self.update()

    # -- paint -----------------------------------------------------

    def _read_worker(self) -> dict:
        """Defensive read of the Round-30 diagnostic scalars.

        Top-level `worker is None` short-circuits BEFORE any
        attribute access — this is the branch main_window.py:22175
        (`self._worker = None`) hits after start_engine fails, and
        also the pre-start_engine window before the user clicks
        START. Every field uses getattr with a default so a future
        rename on the worker degrades to the idle placeholder
        rather than raising on the main thread.
        """
        worker = getattr(self._mw, "_worker", None)
        if worker is None:
            return {"idle": True}
        return {
            "idle": False,
            "fps": getattr(worker, "_fps", 0.0),
            "target": getattr(worker, "_current_capture_target", None),
            "device": getattr(worker, "_resolved_dshow_device_name", "") or "",
            "swap": getattr(worker, "_last_perf_swap_outcome", "") or "",
            "tune_ms": getattr(worker, "_last_tune_elapsed_ms", 0.0),
            "tune_calls": getattr(worker, "_tune_call_count", 0),
            "low_fps": bool(getattr(worker, "_low_fps_active", False)),
            # Round-32: which capture backend is actually live —
            # populated by _stash_ffmpeg_engaged_diagnostics on ffmpeg
            # engage (both PATH A + PATH B) and by every OpenCV-branch
            # write in _apply_perf_camera_path + _upgrade_to_ffmpeg_
            # capture_if_lite. Empty string until the worker's cap
            # path finishes resolving — paint fallback shows "(pending)"
            # rather than the misleading "opencv" default.
            "backend": getattr(worker, "_camera_backend", "") or "",
        }

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            p.fillRect(self.rect(), QColor(12, 12, 18, 200))
            p.setPen(QPen(QColor(255, 255, 255, 60), 1))
            p.drawRect(self.rect().adjusted(0, 0, -1, -1))

            state = self._read_worker()
            text_color = QColor(220, 225, 235, 255)
            dim_color = QColor(160, 165, 180, 255)
            p.setPen(text_color)

            x = 8
            y = 14
            line_h = 12

            if state.get("idle"):
                p.setPen(dim_color)
                p.drawText(x, y, "PERF HUD  (worker idle)")
                p.drawText(x, y + line_h, "start engine to populate")
                return

            target = state.get("target") or (None, None, None, None)
            try:
                w, h, fps_req, fourcc = (
                    target[0], target[1], target[2], target[3],
                )
            except Exception:
                w = h = fps_req = fourcc = None

            device = state.get("device", "")
            if len(device) > 34:
                device = device[:33] + "..."
            swap = state.get("swap", "") or ""
            # Round-32 paint fallback: key on empty-string EXACTLY (the
            # __init__ default in noop_engine.py). Do NOT substring-
            # match on "ffmpeg-MJPG" or condition on "_apply_perf_
            # camera_path has ever fired" — the helper's synthesized
            # string on PATH A is legible on its own, and a substring
            # test would misclassify the PATH A initial-open engage
            # into "(none since start)" even though a real swap DID
            # happen. Nuance called out explicitly in the review.
            if not swap:
                swap_render = "(none since start)"
            elif len(swap) > 34:
                swap_render = swap[:33] + "..."
            else:
                swap_render = swap
            backend = state.get("backend", "") or ""
            if not backend:
                backend_render = "(pending)"
            elif len(backend) > 34:
                backend_render = backend[:33] + "..."
            else:
                backend_render = backend

            lines = [
                # Round-32 header: BUILD_ROUND surfaces so screenshot
                # forensics can tell whether the fix landed without
                # reading the About dialog.
                f"PERF HUD r{int(BUILD_ROUND)}",
                f"fps {float(state.get('fps', 0.0)):5.1f}  "
                f"lite {'on' if state.get('low_fps') else 'off'}",
                f"cap  {w}x{h} @ {fps_req} {fourcc}",
                # Round-32 new line: which capture backend is actually
                # live. Never lies across a reverse toggle because
                # noop_engine.py writes _camera_backend on every
                # OpenCV-branch return path in both engage-site methods.
                f"back {backend_render}",
                f"dev  {device}",
                f"swap {swap_render}",
                f"tune {float(state.get('tune_ms', 0.0)):6.1f}ms  "
                f"n={state.get('tune_calls', 0)}",
            ]
            for i, ln in enumerate(lines):
                p.drawText(x, y + i * line_h, ln)
        finally:
            p.end()

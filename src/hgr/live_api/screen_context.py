"""Screen-context capture for the Live API session.

Reuses the existing PIL ImageGrab pipeline already in
`src/hgr/debug/youtube_controller.py` (Pillow is in requirements.txt).
TODO: extend to multi-monitor — for the prototype we capture the
primary virtual screen and crop on demand later.

Image data is base64-encoded JPEG bytes ready to be sent as a
content part to the Realtime model. The capture uses a worker
thread loop scheduled from `LiveApiManager`, so the UI never
blocks on a screenshot.
"""
from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .live_api_logger import LiveApiLogger
from ..debug.foreground_window import get_foreground_window_info


def describe_monitor_layout() -> str:
    """Human-readable map of the physical monitors as they appear, left to
    right, in an all-screens stitched capture — so the model labels
    'primary'/'secondary' from ground truth instead of guessing by image
    position. Returns "" for single-monitor / non-Windows / on any failure.

    The stitched image's x-origin is the virtual desktop's min-left, so
    ordering monitors by their left edge matches their left→right order in
    the image. The PRIMARY monitor is the one whose top-left is (0,0).
    """
    try:
        import ctypes
        from ctypes import wintypes

        # Fresh WinDLL — the shared user32 can have argtypes polluted by
        # other modules, which breaks the WINFUNCTYPE callback (same reason
        # tool_executor._list_monitors does this).
        user32 = ctypes.WinDLL("user32")
        rects: list[tuple[int, int, int, int]] = []
        MonEnumProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(wintypes.RECT), ctypes.c_void_p,
        )

        def _cb(_hmon, _hdc, lprc, _data):
            try:
                r = lprc.contents
                rects.append((int(r.left), int(r.top), int(r.right), int(r.bottom)))
            except Exception:
                pass
            return True

        user32.EnumDisplayMonitors(0, 0, MonEnumProc(_cb), 0)
        if len(rects) < 2:
            return ""
        ordered = sorted(rects, key=lambda m: m[0])  # left edge = image x
        labels = []
        for left, top, right, bottom in ordered:
            is_primary = (left == 0 and top == 0)
            labels.append(f"{'PRIMARY' if is_primary else 'secondary'} "
                          f"({right - left}x{bottom - top})")
        return (
            f"This image stitches {len(rects)} monitors side by side. "
            f"Left to right in the image: {'; '.join(labels)}. "
            "When the user says 'primary'/'secondary' or 'main'/'second' "
            "monitor, use THESE labels — do not infer from left/right position."
        )
    except Exception:
        return ""


def describe_open_windows(limit: int = 12) -> str:
    """List the visible top-level windows (title + process) as ground truth
    so the model identifies apps from their real window titles instead of
    guessing from the pixels (e.g. calling VS Code 'Jupyter Notebook').
    Returns "" on non-Windows or any failure."""
    try:
        from ..debug.foreground_window import enumerate_visible_windows
        wins = enumerate_visible_windows()
    except Exception:
        return ""
    if not wins:
        return ""
    # Shell/system windows that are visible but meaningless to the user.
    skip = {
        "program manager", "windows input experience",
        "windows shell experience host", "microsoft text input application",
        "settings", "",
    }
    seen: set = set()
    items: list[str] = []
    for w in wins:
        title = (w.title or "").strip()
        proc = (w.process_name or "").strip()
        if title.lower() in skip:
            continue
        key = (title.lower(), proc)
        if key in seen:
            continue
        seen.add(key)
        shown = title if len(title) <= 60 else title[:57] + "..."
        items.append(f"'{shown}'" + (f" [{proc}]" if proc else ""))
        if len(items) >= limit:
            break
    if not items:
        return ""
    return ("Open windows (ground truth — name apps from these, not from the "
            "pixels): " + "; ".join(items) + ".")


@dataclass
class ScreenFrame:
    captured_at: float
    width: int
    height: int
    jpeg_bytes: bytes
    active_window_title: str
    active_window_process: str
    monitor_layout: str = ""
    open_windows: str = ""

    @property
    def b64(self) -> str:
        return base64.b64encode(self.jpeg_bytes).decode("ascii")


class ScreenContext:
    """Captures + compresses screenshots on demand."""

    def __init__(
        self,
        *,
        max_width: int,
        jpeg_quality: int,
        logger: LiveApiLogger,
        debug_save_dir: Optional[Path] = None,
    ) -> None:
        self._max_width = max(320, int(max_width))
        self._jpeg_quality = max(20, min(95, int(jpeg_quality)))
        self._logger = logger
        self._debug_save_dir = debug_save_dir
        self._capture_count = 0

    def capture(self) -> Optional[ScreenFrame]:
        started = time.time()
        try:
            from PIL import ImageGrab
        except Exception as exc:
            self._logger.exception("screen_capture_pillow_missing", exc)
            return None

        try:
            # all_screens=True captures the full virtual desktop on
            # multi-monitor setups so the model sees windows on any
            # monitor, not just the one with the active window.
            img = ImageGrab.grab(all_screens=True)
        except Exception as exc:
            self._logger.exception("screen_capture_grab_failed", exc)
            return None

        try:
            if img.width > self._max_width:
                ratio = self._max_width / float(img.width)
                new_size = (self._max_width, max(1, int(round(img.height * ratio))))
                img = img.resize(new_size)
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self._jpeg_quality, optimize=False)
            jpeg_bytes = buf.getvalue()
        except Exception as exc:
            self._logger.exception("screen_capture_encode_failed", exc)
            return None

        info = get_foreground_window_info()
        title = "" if info is None else (info.title or "")
        process = "" if info is None else (info.process_name or "")

        frame = ScreenFrame(
            captured_at=started,
            width=img.width,
            height=img.height,
            jpeg_bytes=jpeg_bytes,
            active_window_title=title,
            active_window_process=process,
            monitor_layout=describe_monitor_layout(),
            open_windows=describe_open_windows(),
        )
        self._capture_count += 1
        elapsed_ms = round((time.time() - started) * 1000.0, 2)
        self._logger.event(
            "screen_capture",
            seq=self._capture_count,
            width=frame.width,
            height=frame.height,
            jpeg_kb=round(len(jpeg_bytes) / 1024.0, 2),
            window_title=title,
            window_process=process,
            elapsed_ms=elapsed_ms,
        )

        if self._debug_save_dir is not None:
            try:
                self._debug_save_dir.mkdir(parents=True, exist_ok=True)
                out = self._debug_save_dir / f"screen_{int(started)}_{self._capture_count}.jpg"
                out.write_bytes(jpeg_bytes)
            except Exception as exc:
                self._logger.exception("screen_capture_debug_save_failed", exc)

        return frame

# Author: Konstantin Markov

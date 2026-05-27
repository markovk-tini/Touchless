"""On-screen OCR → exact-pixel clicking for the Live API agent.

The model's normal click_screen makes IT estimate a 0-1 fraction on a
DOWNSCALED screenshot, which is imprecise for small/custom targets. This
module instead MEASURES where a button is: it captures the full-resolution
desktop, runs OCR (rapidocr-onnxruntime — pure ONNX, offline), finds the box
whose text matches a target word, and clicks the center of THAT box in real
screen pixels. Works on anything with visible text — Chromium/webview buttons
(Valorant's PLAY, Claude Code's Yes), native dialogs, etc. — including UIs that
UIA can't see.

Self-contained and guarded: if OCR can't init, every call returns a clean
error and the agent can fall back to click_screen.
"""
from __future__ import annotations

import ctypes
import time
from typing import Any, Dict, List, Optional, Tuple


# SendInput structures. Defined at MODULE level (one canonical type) so the
# fresh WinDLL below can declare SendInput's argtypes against THIS exact type.
# Other modules set their own SendInput.argtypes on the SHARED ctypes.windll,
# which then rejects our byref() ("expected LP_INPUT instead of pointer to
# _INPUT") — so we never touch the shared handle for input injection.
_ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long), ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong), ("dwExtraInfo", _ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]


_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1


def _fresh_user32():
    """A PRIVATE user32 handle with SendInput argtypes bound to our _INPUT, so
    it's immune to argtypes pollution from other modules on ctypes.windll."""
    u = ctypes.WinDLL("user32")
    u.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(_INPUT), ctypes.c_int]
    u.SendInput.restype = ctypes.c_uint
    return u


def _norm(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())


class ScreenOcr:
    def __init__(self, logger) -> None:
        self._logger = logger
        self._engine = None
        self._init_failed = False
        # Private, un-polluted user32 for SendInput (mouse/keyboard injection).
        try:
            self._u32 = _fresh_user32()
        except Exception:
            self._u32 = ctypes.windll.user32  # last resort

    def _ensure(self) -> bool:
        if self._engine is not None:
            return True
        if self._init_failed:
            return False
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            self._logger.event("screen_ocr_ready")
            return True
        except Exception as exc:
            self._init_failed = True
            self._logger.exception("screen_ocr_init_failed", exc)
            return False

    def _virtual_origin(self) -> Tuple[int, int]:
        """Top-left of the virtual desktop (can be negative on multi-monitor).
        ImageGrab(all_screens=True) uses this as its (0,0)."""
        try:
            user32 = ctypes.windll.user32
            # SM_XVIRTUALSCREEN=76, SM_YVIRTUALSCREEN=77
            return int(user32.GetSystemMetrics(76)), int(user32.GetSystemMetrics(77))
        except Exception:
            return 0, 0

    def _capture(self):
        """Full-resolution grab of the WHOLE virtual desktop as a numpy array,
        plus the virtual-screen origin so OCR pixels map to absolute coords."""
        from PIL import ImageGrab
        import numpy as np
        img = ImageGrab.grab(all_screens=True)
        return np.array(img), self._virtual_origin()

    def find_text(self, needle: str) -> Dict[str, Any]:
        """OCR the screen and return every text box matching `needle`
        (case-insensitive substring), each with its absolute-pixel center,
        sorted by reading order (top→bottom, left→right)."""
        if not self._ensure():
            return {"status": "error", "error": "OCR unavailable", "code": "no_ocr"}
        needle_n = _norm(needle)
        if not needle_n:
            return {"status": "error", "error": "empty text", "code": "invalid_arguments"}
        try:
            arr, (ox, oy) = self._capture()
        except Exception as exc:
            self._logger.exception("screen_ocr_capture_failed", exc)
            return {"status": "error", "error": f"capture failed: {exc}", "code": "capture_failed"}
        try:
            result, _elapse = self._engine(arr)
        except Exception as exc:
            self._logger.exception("screen_ocr_run_failed", exc)
            return {"status": "error", "error": f"ocr failed: {exc}", "code": "ocr_failed"}
        matches: List[Dict[str, Any]] = []
        for item in (result or []):
            try:
                box, text, score = item
                t = _norm(text)
                if not t:
                    continue
                # Match only when the on-screen text CONTAINS the target. (The
                # reverse — target contains text — falsely matched short tokens
                # like 'on' inside a longer query, clicking the wrong thing.)
                if needle_n in t:
                    xs = [float(p[0]) for p in box]
                    ys = [float(p[1]) for p in box]
                    cx = int(sum(xs) / len(xs)) + ox
                    cy = int(sum(ys) / len(ys)) + oy
                    matches.append({
                        "text": str(text),
                        "score": round(float(score), 3),
                        "x": cx,
                        "y": cy,
                        "exact": t == needle_n,
                    })
            except Exception:
                continue
        # Reading order: top→bottom then left→right.
        matches.sort(key=lambda m: (m["y"] // 20, m["x"]))
        return {"status": "ok", "query": needle, "matches": matches, "count": len(matches)}

    def read_text_boxes(self) -> Dict[str, Any]:
        """OCR the screen and return EVERY text item with its absolute-pixel
        center, in reading order — a clickable text map for read_screen."""
        if not self._ensure():
            return {"status": "error", "error": "OCR unavailable", "code": "no_ocr"}
        try:
            arr, (ox, oy) = self._capture()
        except Exception as exc:
            self._logger.exception("screen_ocr_capture_failed", exc)
            return {"status": "error", "error": f"capture failed: {exc}", "code": "capture_failed"}
        try:
            result, _elapse = self._engine(arr)
        except Exception as exc:
            self._logger.exception("screen_ocr_run_failed", exc)
            return {"status": "error", "error": f"ocr failed: {exc}", "code": "ocr_failed"}
        items: List[Dict[str, Any]] = []
        for entry in (result or []):
            try:
                box, text, score = entry
                t = str(text).strip()
                if not t:
                    continue
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                items.append({
                    "text": t,
                    "x": int(sum(xs) / len(xs)) + ox,
                    "y": int(sum(ys) / len(ys)) + oy,
                    "score": round(float(score), 3),
                })
            except Exception:
                continue
        items.sort(key=lambda m: (m["y"] // 20, m["x"]))
        return {"status": "ok", "items": items, "count": len(items)}

    def capture_crop(self, cx: float, cy: float, size: float = 0.25):
        """Capture a FULL-RESOLUTION crop centered at normalized (cx,cy) of the
        virtual desktop, spanning `size` fraction of it. Returns
        (b64_jpeg, (abs_left, abs_top, crop_w, crop_h)) — the bounds are in
        absolute screen pixels so a later crop-relative click maps back exactly.
        Lets the model SEE a small target zoomed in (far more accurate than
        guessing on a downscaled full-screen image)."""
        import io
        import base64
        from PIL import ImageGrab
        img = ImageGrab.grab(all_screens=True)  # full native resolution
        ox, oy = self._virtual_origin()
        W, H = img.width, img.height
        size = max(0.05, min(1.0, float(size)))
        # Square crop sized off the screen HEIGHT so the zoom is uniform (a
        # width-based crop is wide-and-short on multi-monitor desktops).
        side = max(80, int(min(W, H) * size))
        cw = min(W, side)
        ch = min(H, side)
        cxp = int(max(0.0, min(1.0, cx)) * W)
        cyp = int(max(0.0, min(1.0, cy)) * H)
        left = max(0, min(W - cw, cxp - cw // 2))
        top = max(0, min(H - ch, cyp - ch // 2))
        crop = img.crop((left, top, left + cw, top + ch)).convert("RGB")
        # Cap the SENT size (bounds stay in original full-res pixels, so the
        # resize doesn't affect click mapping).
        max_w = 1024
        if crop.width > max_w:
            r = max_w / float(crop.width)
            crop = crop.resize((max_w, max(1, int(crop.height * r))))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, (ox + left, oy + top, cw, ch)

    def _click_xy(self, x: int, y: int, clicks: int = 1) -> bool:
        """Move to (x,y) and click via SendInput (the modern API — Chromium /
        Electron launchers like HoYoPlay / Riot often IGNORE the legacy
        mouse_event with no down→up gap). Focuses the window under the point
        first so an inactive launcher actually accepts the click, and supports
        double-click."""
        try:
            user32 = ctypes.windll.user32
            x = int(x)
            y = int(y)
            # Bring the top-level window under the cursor to the front so the
            # click activates a control. Chromium/CEF launchers IGNORE clicks
            # while not the active window, and a plain SetForegroundWindow from
            # a background process is blocked by Windows' foreground lock — so
            # we use the AttachThreadInput trick to genuinely force it active.
            try:
                from ctypes import wintypes
                user32.WindowFromPoint.argtypes = [wintypes.POINT]
                user32.WindowFromPoint.restype = wintypes.HWND
                # GetAncestor MUST declare HWND restype — the default c_int
                # TRUNCATES the 64-bit handle, so _force_foreground would get a
                # garbage hwnd and focus the wrong window (clicks land nowhere).
                user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
                user32.GetAncestor.restype = wintypes.HWND
                hwnd = user32.WindowFromPoint(wintypes.POINT(x, y))
                if hwnd:
                    top = user32.GetAncestor(hwnd, 2)  # GA_ROOT
                    if top:
                        self._force_foreground(top)
                        time.sleep(0.12)
            except Exception:
                pass
            self._move_abs(x, y)  # injected position so the button lands here
            user32.SetCursorPos(x, y)
            time.sleep(0.06)
            n = max(1, int(clicks))
            for _ in range(n):
                self._send_left_click()
                time.sleep(0.07)
            self._logger.event("ocr_click", x=x, y=y, clicks=n)
            return True
        except Exception as exc:
            self._logger.exception("screen_ocr_click_failed", exc)
            return False

    def _force_foreground(self, hwnd) -> None:
        """Genuinely bring `hwnd` to the foreground. A bare SetForegroundWindow
        from a background process is blocked by Windows' foreground lock, so we
        attach to the current foreground thread's input queue first (the
        standard workaround) — this is what lets Chromium/CEF launchers accept
        our subsequent click."""
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            fg = user32.GetForegroundWindow()
            if fg == hwnd:
                user32.BringWindowToTop(hwnd)
                return
            fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
            our_tid = kernel32.GetCurrentThreadId()
            attached = False
            if fg_tid and fg_tid != our_tid:
                attached = bool(user32.AttachThreadInput(our_tid, fg_tid, True))
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            if attached:
                user32.AttachThreadInput(our_tid, fg_tid, False)
        except Exception as exc:
            self._logger.exception("force_foreground_failed", exc)

    _MOUSEEVENTF_MOVE = 0x0001
    _MOUSEEVENTF_LEFTDOWN = 0x0002
    _MOUSEEVENTF_LEFTUP = 0x0004
    _MOUSEEVENTF_ABSOLUTE = 0x8000
    _MOUSEEVENTF_VIRTUALDESK = 0x4000

    def _send_mouse(self, flags: int, nx: int = 0, ny: int = 0) -> int:
        """Inject one mouse event via SendInput through our PRIVATE user32 handle
        (immune to argtypes pollution). Returns events injected (0 = failed)."""
        inp = _INPUT()
        inp.type = _INPUT_MOUSE
        inp.mi.dx = int(nx)
        inp.mi.dy = int(ny)
        inp.mi.dwFlags = flags  # anonymous union → inp.mi is the MOUSEINPUT
        try:
            return int(self._u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT)))
        except Exception:
            return 0

    def _mouse_event(self, flags: int) -> None:
        """One button event at the current position. Falls back to the legacy
        mouse_event API if SendInput injects nothing (normal Win32 apps)."""
        if not self._send_mouse(flags):
            try:
                self._u32.mouse_event(flags, 0, 0, 0, 0)
            except Exception:
                pass

    def _move_abs(self, x: int, y: int) -> None:
        """Move the cursor by INJECTING an absolute SendInput move event (mapped
        0..65535 over the whole virtual desktop). Critical for DRAGS: apps treat
        a SetCursorPos-warped cursor as teleporting, not dragging — only injected
        MOVE events with the button held register as a real drag. Falls back to
        SetCursorPos if the injected move fails."""
        vl, vt, vw, vh = self._virtual_rect()
        nx = int(round((x - vl) * 65535.0 / max(1, vw - 1)))
        ny = int(round((y - vt) * 65535.0 / max(1, vh - 1)))
        flags = self._MOUSEEVENTF_MOVE | self._MOUSEEVENTF_ABSOLUTE | self._MOUSEEVENTF_VIRTUALDESK
        if not self._send_mouse(flags, nx, ny):
            try:
                self._u32.SetCursorPos(int(x), int(y))
            except Exception:
                pass

    def _send_left_click(self) -> None:
        """One left down→up at the current cursor position with a short hold."""
        self._mouse_event(self._MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.04)
        self._mouse_event(self._MOUSEEVENTF_LEFTUP)

    def drag(self, x1: int, y1: int, x2: int, y2: int, steps: int = 24, hold: float = 0.12) -> Dict[str, Any]:
        """Press at (x1,y1), glide to (x2,y2) in small steps, release — a real
        click-and-drag (apps need the intermediate moves to register a drag).
        Absolute screen pixels. Focuses the window under the start point."""
        try:
            user32 = ctypes.windll.user32
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            try:
                from ctypes import wintypes
                user32.WindowFromPoint.argtypes = [wintypes.POINT]
                user32.WindowFromPoint.restype = wintypes.HWND
                user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
                user32.GetAncestor.restype = wintypes.HWND  # HWND, not truncated c_int
                hwnd = user32.WindowFromPoint(wintypes.POINT(x1, y1))
                top = user32.GetAncestor(hwnd, 2) if hwnd else 0
                if top:
                    self._force_foreground(top)
                    time.sleep(0.12)
            except Exception:
                pass
            self._move_abs(x1, y1)
            user32.SetCursorPos(x1, y1)
            time.sleep(0.08)
            self._mouse_event(self._MOUSEEVENTF_LEFTDOWN)
            time.sleep(max(0.05, hold))
            steps = max(2, int(steps))
            for i in range(1, steps + 1):
                ix = int(x1 + (x2 - x1) * i / steps)
                iy = int(y1 + (y2 - y1) * i / steps)
                self._move_abs(ix, iy)  # injected move (registers as a drag)
                time.sleep(0.012)
            time.sleep(max(0.05, hold))
            self._mouse_event(self._MOUSEEVENTF_LEFTUP)
            self._logger.event("ocr_drag", x1=x1, y1=y1, x2=x2, y2=y2)
            return {"status": "ok", "from": [x1, y1], "to": [x2, y2]}
        except Exception as exc:
            self._logger.exception("drag_failed", exc)
            return {"status": "error", "error": str(exc), "code": "drag_failed"}

    def draw_path(self, points, hold: float = 0.1) -> Dict[str, Any]:
        """Draw a CONNECTED stroke through `points` (a list of (x,y) absolute
        pixels): press at the first point, glide through every point in order,
        release once at the end. One continuous pen stroke — so a closed shape
        (square = 5 points back to start, triangle = 4) comes out connected.
        Far more reliable than the model issuing many separate drags."""
        try:
            pts = [(int(x), int(y)) for x, y in points]
            if len(pts) < 2:
                return {"status": "error", "error": "need at least 2 points", "code": "too_few_points"}
            # Use the SAME handle/sequence as drag() (proven to draw): shared
            # windll for cursor moves, _mouse_event for the button. The fresh
            # handle is only needed for SendInput (button), which _mouse_event
            # already uses.
            user32 = ctypes.windll.user32
            x0, y0 = pts[0]
            self._logger.event("ocr_draw_path_begin", points=len(pts), abs_pts=pts)
            # Focus the window under the start point so the stroke lands there.
            try:
                from ctypes import wintypes
                user32.WindowFromPoint.argtypes = [wintypes.POINT]
                user32.WindowFromPoint.restype = wintypes.HWND
                user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
                user32.GetAncestor.restype = wintypes.HWND
                hwnd = user32.WindowFromPoint(wintypes.POINT(x0, y0))
                top = user32.GetAncestor(hwnd, 2) if hwnd else 0
                if top:
                    self._force_foreground(top)
                    time.sleep(0.12)
            except Exception:
                pass
            self._move_abs(x0, y0)
            user32.SetCursorPos(x0, y0)
            time.sleep(0.08)
            self._mouse_event(self._MOUSEEVENTF_LEFTDOWN)
            time.sleep(max(0.05, hold))
            px, py = x0, y0
            for (tx, ty) in pts[1:]:
                dist = max(abs(tx - px), abs(ty - py))
                steps = max(10, min(120, dist // 6))
                for i in range(1, steps + 1):
                    ix = int(px + (tx - px) * i / steps)
                    iy = int(py + (ty - py) * i / steps)
                    self._move_abs(ix, iy)  # injected move (registers as a drag)
                    time.sleep(0.012)
                px, py = tx, ty
                time.sleep(0.03)  # small settle at each vertex
            time.sleep(max(0.05, hold))
            self._mouse_event(self._MOUSEEVENTF_LEFTUP)
            self._logger.event("ocr_draw_path", points=len(pts))
            return {"status": "ok", "points": len(pts)}
        except Exception as exc:
            self._logger.exception("draw_path_failed", exc)
            return {"status": "error", "error": str(exc), "code": "draw_path_failed"}

    def _virtual_rect(self):
        """(left, top, width, height) of the whole virtual desktop, physical px."""
        u = ctypes.windll.user32
        return (int(u.GetSystemMetrics(76)), int(u.GetSystemMetrics(77)),
                int(u.GetSystemMetrics(78)), int(u.GetSystemMetrics(79)))

    def read_all_text(self) -> str:
        """OCR the whole desktop and return all detected text joined into one
        string. Used to DETECT prompts (e.g. Claude Code's Yes/No) that UIA
        can't read because they're rendered in a webview/terminal canvas."""
        if not self._ensure():
            return ""
        try:
            arr, _origin = self._capture()
            result, _elapse = self._engine(arr)
        except Exception as exc:
            self._logger.exception("screen_ocr_readall_failed", exc)
            return ""
        parts: List[str] = []
        for item in (result or []):
            try:
                _box, text, _score = item
                if text:
                    parts.append(str(text))
            except Exception:
                continue
        return " ".join(parts)

    def press_key(self, name: str) -> bool:
        """Press a single key globally (goes to the focused window). Used after
        clicking a prompt to FOCUS its panel, to send the Yes key (1/y/Enter)."""
        vk_map = {
            "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B,
            "escape": 0x1B, "space": 0x20, "y": 0x59, "n": 0x4E,
        }
        nm = str(name or "").strip().lower()
        vk = vk_map.get(nm)
        if vk is None and len(nm) == 1:
            try:
                r = ctypes.windll.user32.VkKeyScanW(ord(nm))
                if r != -1:
                    vk = r & 0xFF
            except Exception:
                vk = None
        if vk is None:
            return False
        try:
            u = ctypes.windll.user32
            u.keybd_event(vk, 0, 0, 0)
            time.sleep(0.03)
            u.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP
            return True
        except Exception:
            return False

    def wait_for_text(self, needle: str, timeout_sec: float = 20.0, poll: float = 0.8) -> Dict[str, Any]:
        """Poll the screen (OCR) until `needle` is visible, up to timeout. The
        general 'is this ready / has the page loaded?' check — e.g. wait for a
        game's PLAY button, an app's title, a 'Sign in' link — before acting."""
        if not self._ensure():
            return {"status": "error", "error": "OCR unavailable", "code": "no_ocr"}
        deadline = time.monotonic() + max(1.0, min(120.0, float(timeout_sec)))
        last = {"count": 0}
        while time.monotonic() < deadline:
            r = self.find_text(needle)
            if r.get("status") == "ok" and r.get("matches"):
                return {"status": "ok", "appeared": True, "matches": r["matches"]}
            if r.get("status") == "ok":
                last = r
            time.sleep(max(0.3, poll))
        return {
            "status": "timeout",
            "appeared": False,
            "error": f"'{needle}' not visible within {int(timeout_sec)}s",
        }

    def _pick_target(self, matches, occurrence):
        exact = [m for m in matches if m.get("exact")]
        pool = exact or matches
        idx = max(1, int(occurrence or 1)) - 1
        if idx >= len(pool):
            idx = 0
        return pool[idx], pool, idx

    def click_text(self, needle: str, occurrence: int = 1, timeout_sec: float = 0.0,
                   clicks: int = 1, retries: int = 0, verify_delay: float = 3.0) -> Dict[str, Any]:
        """Find `needle` on screen and click the center of its text box. Prefers
        an EXACT word match; `occurrence` (1-based) picks among multiple hits in
        reading order. If `timeout_sec` > 0, POLL until it appears first.

        retries > 0 enables VERIFY-AND-RETRY: after clicking, wait `verify_delay`
        and re-OCR — if the text is GONE the click worked (success); if it's
        still there (e.g. a slow launcher whose layout shifted, so the first
        click hit a stale spot) re-capture FRESH coordinates and click again,
        up to `retries` times. Use for launcher buttons (Start Game / PLAY)."""
        if timeout_sec and float(timeout_sec) > 0:
            waited = self.wait_for_text(needle, timeout_sec=float(timeout_sec))
            if waited.get("status") != "ok":
                return {"status": "error", "error": waited.get("error", f"'{needle}' did not appear"), "code": "timeout"}

        attempts = max(1, int(retries) + 1)
        last_target = None
        for attempt in range(attempts):
            found = self.find_text(needle)
            if found.get("status") != "ok":
                return found
            matches = found.get("matches") or []
            if not matches:
                # Text no longer on screen. If we've already clicked, it
                # disappearing means the click worked (the launcher launched /
                # dialog closed).
                if attempt > 0:
                    return {"status": "ok", "clicked": needle, "verified": True, "attempts": attempt}
                return {"status": "error", "error": f"no on-screen text matching '{needle}'", "code": "no_match"}
            target, pool, idx = self._pick_target(matches, occurrence)
            last_target = target
            if not self._click_xy(target["x"], target["y"], clicks=max(1, int(clicks))):
                return {"status": "error", "error": "found text but click failed", "code": "click_failed", "target": target}
            if int(retries) <= 0:
                return {"status": "ok", "clicked": target["text"], "x": target["x"], "y": target["y"],
                        "score": target["score"], "occurrence": idx + 1, "total_matches": len(pool)}
            # Verify: wait, then loop to re-check. If still present next pass,
            # we re-click at fresh (settled) coordinates.
            time.sleep(max(0.5, float(verify_delay)))
        # Exhausted retries and the text is still visible after the last click.
        return {
            "status": "ok",
            "clicked": (last_target or {}).get("text", needle),
            "x": (last_target or {}).get("x"),
            "y": (last_target or {}).get("y"),
            "verified": False,
            "note": "Clicked but the target is still on screen — it may not have activated; the user might need to click it.",
        }

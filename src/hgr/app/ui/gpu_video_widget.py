"""GPU-backed video display widget.

Why this exists:
The QLabel.setPixmap path used to display the camera feed ran on
every frame:

    cv2.cvtColor(BGR→RGB)   ~3 ms at 720p
    cv2.resize(INTER_AREA)   ~1-2 ms
    QImage.copy()            ~1-2 ms
    QPixmap.fromImage()      ~1 ms (allocates GPU texture)
    setPixmap + paint event  ~1-2 ms

That's ~7-10 ms of CPU work per frame just to put a camera image on
screen. This widget skips all of it: it constructs a QImage with
Format_BGR888 (no CPU colour conversion — the GPU's texture sampler
handles BGR natively) and uses Qt's raster paint engine (D3D-backed
on Windows, GPU-accelerated by default) to draw image + landmarks.

We do NOT use QOpenGLWidget — that adds an OpenGL/ANGLE driver path
that on some Windows setups silently coalesces frames and produces
~9 fps perceived display rate even when the worker is running at
30+ fps. The default QWidget raster engine is already GPU-backed
through Qt's D3D11 paint backend, with much less driver-stack risk.

Includes a paint-rate counter that logs `[gpu_video] paint rate: N
fps` every 2 s, so display rate can be observed independently of
the worker's `actual self._fps`.
"""
from __future__ import annotations

import sys
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# macOS-only: QWidget paint uses the CPU software raster engine (no D3D
# acceleration like Windows), so scaling a full 720p frame in paintEvent is
# expensive. We pre-downscale to the on-screen size with cv2 there.
if sys.platform == "darwin":
    try:
        import cv2 as _cv2
    except Exception:
        _cv2 = None
else:
    _cv2 = None

from PySide6.QtCore import QLineF, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QGuiApplication, QImage, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

# MediaPipe's 21-landmark hand connections (pairs of indices).
# Same topology the cv2-based draw_hand_overlay used to draw on the
# BGR frame — we now draw it on the GPU instead.
_HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # palm base to pinky
)


class GpuVideoWidget(QWidget):
    """Drop-in replacement for the QLabel video panel.

    Public API (used by mini_live_viewer / live_view_window):
      - update_frame(bgr_numpy)  → schedule GPU repaint with new frame
      - update_landmarks(hands)  → list of per-hand normalized [(x,y), ...]
      - clear_video(idle_text)   → drop the frame, optionally show idle text
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._image_w: int = 0
        self._image_h: int = 0
        # Per-hand display info — list of dicts with keys:
        #   landmarks: list[(x, y)] normalized
        #   bbox:      (x, y, w, h) normalized | None
        #   handedness: "Left" / "Right" / ""
        #   label:     gesture name string (empty when inactive)
        #   active:    bool — toggles bbox to green
        # Both hands are equal — the engine produces a separate
        # prediction per hand and either can drive its own active
        # state independently. There is no "primary" / "secondary"
        # distinction in the display.
        self._hands_info: List[dict] = []
        # Mouse-mode "control area" overlay. Set by update_landmarks
        # when the engine emits a payload containing mouse_overlay
        # data. Dict with keys "bounds" (x1, y1, x2, y2 normalized
        # in [0, 1] image coords) and optional "anchor" (ax, ay).
        self._mouse_overlay: Optional[dict] = None
        # Cached pre-rendered monitor-layout pixmap. The layout is
        # static (screen geometry doesn't change frame-to-frame) but
        # we used to redraw N rectangles + a backdrop per paint. Now
        # we draw it once into a pixmap keyed on (bounds + screen
        # geometry) and just blit the pixmap each frame. The cursor
        # dot still draws on top live.
        self._mouse_box_pixmap: Optional[QPixmap] = None
        self._mouse_box_signature: tuple = ()
        self._idle_text: str = ""
        self._idle_font = QFont("Segoe UI", 10)
        self._banner_font = QFont("Segoe UI", 9)
        self._banner_font.setBold(True)
        # RoundCap on the landmark pen so each drawPoint renders as
        # a round disc instead of a square — at 5 px it's visibly
        # circular and reads as a "joint dot" rather than a pixel
        # cluster. Connection pen stays square-cap (default) and
        # bumps from 2 px to 3 px for a slightly chunkier skeleton.
        self._landmark_pen = QPen(QColor(29, 233, 182), 5)
        self._landmark_pen.setCapStyle(Qt.RoundCap)
        self._connection_pen = QPen(QColor(29, 233, 182, 200), 3)
        # Bbox colors — red default, green when that hand has a
        # recognized gesture.
        self._bbox_inactive_color = QColor(232, 72, 72, 235)
        self._bbox_active_color = QColor(70, 220, 130, 235)
        # Mouse-mode control box: red border so the user
        # immediately reads "this is the active region — keep your
        # hand in here." Faint red fill for the area itself.
        self._mouse_box_color = QColor(255, 64, 56, 235)
        self._mouse_box_fill_color = QColor(255, 64, 56, 24)
        self._mouse_anchor_color = QColor(255, 248, 212, 230)
        self._banner_bg_color = QColor(0, 0, 0, 150)
        self._banner_text_color = QColor(248, 250, 252, 250)
        self._idle_color = QColor(180, 200, 220)
        self._background = QColor(7, 19, 29)
        # Paint-rate diagnostic — prints `[gpu_video] paint rate:`
        # every 2 s so we can see whether the actual on-screen
        # update rate matches the worker's emit rate. If they
        # diverge, paint events are coalescing somewhere.
        self._paint_count = 0
        self._paint_log_at = 0.0
        # C1 (v1.1.7 diagnostic): per-section paintEvent timing +
        # paint-to-paint wall-clock delta. Every paint measures
        # drawImage cost + overlay cost + total wall time; the
        # 2 s summary log emits averages so we can distinguish
        # "paint is genuinely expensive" (delta ≈ paint duration)
        # from "paint is cheap but events coalesce" (delta ≫ paint
        # duration). Reset every log emission so the 2 s window is
        # independent — no cross-window contamination.
        self._paint_timing_total_us = 0
        self._paint_timing_drawimage_us = 0
        self._paint_timing_overlay_us = 0
        self._paint_timing_max_us = 0
        self._paint_delta_total_us = 0
        self._paint_delta_max_us = 0
        self._paint_delta_samples = 0
        self._paint_last_end_monotonic = 0.0
        # C3 (v1.1.7 diagnostic): slot-to-paint latency. When
        # update_frame is called from _on_worker_raw_frame in the
        # receiver, we timestamp it. The next paintEvent computes
        # the delta from that timestamp to paint completion — this
        # measures how quickly Qt actually services the update()
        # request. A large slot→paint delta with a small paint
        # duration indicates event-loop coalescing (main thread
        # busy handling other queued work between slot-fire and
        # paint). Reset to 0 after the paintEvent consumes it so a
        # subsequent paint triggered by something other than
        # update_frame (resize, expose) doesn't record a stale delta.
        self._last_slot_fire_perf = 0.0
        self._slot_to_paint_total_us = 0
        self._slot_to_paint_max_us = 0
        self._slot_to_paint_samples = 0
        # Fullscreen-aware lite paint mode. When True, paintEvent
        # skips _draw_landmarks entirely -- the skeleton, bbox,
        # banner, and mouse-overlay strokes are the expensive part
        # of each paint (anti-aliased QPen polylines on the GPU),
        # and during a fullscreen game DWM's GPU compositor is busy
        # with the game so we want each paint to be as cheap as
        # possible. The camera frame itself still draws -- detection
        # is unaffected because it runs on the worker thread off
        # raw frames, not off this widget's paint.
        self._lite_paint_mode = False
        # Switch-view (camera-target) drawing: hide the hand skeleton / bbox /
        # gesture-label overlay so the user's drawing (baked into the frame,
        # with its fingertip cursor) is the top layer and the camera doesn't
        # visibly "read the hand". Set per-frame from the engine payload.
        self._hide_hand_overlay = False
        # C8 (v1.1.7): explicit overlay-level tier for cheap paint-cost
        # scaling per mode. Levels:
        #   3 = full: skeleton + bbox + text banner (default; matches
        #       Normal Mode behaviour)
        #   2 = skeleton + bbox (banner text elided — saves per-hand
        #       text metrics + drawText call chain per paint)
        #   1 = skeleton only (matches the pre-C8 _lite_paint_mode
        #       behaviour used during fullscreen games)
        #   0 = no skeleton, no bbox, no banner (matches drawing-mode
        #       behaviour when _hide_hand_overlay is True)
        # Precedence at paint time is layered in _draw_landmarks:
        #   drawing mode (_hide_hand_overlay=True) forces effective 0
        #   fullscreen game (_lite_paint_mode=True) caps at 1
        #   otherwise, this field controls what draws
        # Set by the presentation-tier caller (see C9) — NOT plumbed
        # through the engine payload to avoid a 1-frame lag race
        # between the overlay-level change and the frame update.
        self._overlay_level = 3
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(220, 140)
        # Disable Qt's automatic background fill — we paint the
        # whole rect ourselves in paintEvent.
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

    def set_lite_paint_mode(self, enabled: bool) -> None:
        """Toggle the fullscreen-aware lite paint path. Called from
        the live-view receivers when a game / fullscreen app appears
        and disappears. Idempotent."""
        new_value = bool(enabled)
        if new_value == self._lite_paint_mode:
            return
        self._lite_paint_mode = new_value
        # Force one paint so the change is visible immediately
        # (otherwise the next paint waits for the next frame).
        self.update()

    def set_overlay_level(self, level: int) -> None:
        """Set the overlay-tier cap. See __init__ docstring for level
        semantics. Called by the presentation-tier caller (live-view
        receiver + engine mode-toggle handler); NOT plumbed through
        the engine payload to avoid a 1-frame race between the level
        change and the frame update. Clamped to [0, 3]. Idempotent."""
        try:
            new_level = max(0, min(3, int(level)))
        except (TypeError, ValueError):
            return
        if new_level == self._overlay_level:
            return
        self._overlay_level = new_level
        self.update()

    # ----- public API used by the receivers ------------------

    def update_frame(self, bgr_frame: np.ndarray) -> None:
        """Hand the widget a new BGR frame. The GPU paint will pick
        it up on the next paintGL. We `.copy()` so the worker's
        reader thread can safely overwrite its source buffer."""
        if bgr_frame is None or bgr_frame.size == 0:
            return
        # v1.1.7 event-loop optimization (Step 4): skip the QImage
        # construction + .copy() when the widget isn't currently
        # visible. Camera frames arrive at ~30-60 fps regardless of
        # whether the mini viewer is on screen — copying 2.76 MB of
        # pixel data every frame for a widget the user can't see is
        # pure waste. Qt's update() is a no-op on hidden widgets
        # anyway, so nothing user-visible changes when we bail early.
        # Restarts cleanly when the widget becomes visible again
        # because the next raw_frame_ready emit runs this method
        # with a fresh frame.
        try:
            visible = self.isVisible()
        except Exception:
            visible = True
        if not visible:
            return
        try:
            h, w = bgr_frame.shape[:2]
        except Exception:
            return
        if h <= 0 or w <= 0:
            return
        # macOS: paint is CPU raster, so scaling a 720p frame in paintEvent
        # costs ~60 ms (a hard ~15 fps cap). Pre-downscale to the widget's
        # on-screen (physical, Retina-aware) size with cv2 INTER_AREA — fast
        # (~0.5 ms) and good quality — so paintEvent draws ~1:1. Only ever
        # downscales; if the widget is bigger than the frame we leave it.
        if _cv2 is not None:
            try:
                dpr = float(self.devicePixelRatioF() or 1.0)
                aw = max(1, self.width())
                ah = max(1, self.height())
                fit = min(aw / float(w), ah / float(h))
                if fit > 0:
                    tw = max(1, int(round(w * fit * dpr)))
                    th = max(1, int(round(h * fit * dpr)))
                    if tw < w:
                        bgr_frame = _cv2.resize(bgr_frame, (tw, th), interpolation=_cv2.INTER_AREA)
                        h, w = th, tw
            except Exception:
                pass
        # Format_BGR888 stores 3 bytes/pixel B,G,R in that order
        # — same as the cv2 numpy buffer. Qt's GL paint engine
        # handles the BGR-vs-RGB sampler swizzle on the GPU, so
        # we skip the CPU cv2.cvtColor pass entirely.
        self._image = QImage(
            bgr_frame.data, w, h, 3 * w, QImage.Format_BGR888
        ).copy()
        self._image_w = w
        self._image_h = h
        self._idle_text = ""
        # C3: stamp slot-fire time BEFORE update() so paintEvent's
        # slot→paint delta measures from the moment this widget was
        # asked for a new frame to the moment Qt got around to
        # actually painting it.
        self._last_slot_fire_perf = time.perf_counter()
        self.update()

    def update_landmarks(self, payload: Optional[object]) -> None:
        """Store per-hand display info for the next paintEvent.

        Accepted payload shapes:
          - dict {"hands": [...], "mouse_overlay": {...}|None} — full
            payload from the engine when mouse mode is on (or any
            other future overlay we layer on the camera frame)
          - iterable of per-hand dicts with keys landmarks, bbox,
            handedness, label, active — the bare hands list when
            no mouse overlay is needed
          - iterable of plain list-of-(x,y) tuples — legacy shape
            from before the bbox/banner additions

        `update()` is NOT called here — the next `update_frame` will
        schedule the repaint, which keeps the overlay in sync with
        the frame it belongs to."""
        if payload is None:
            self._hands_info = []
            self._mouse_overlay = None
            self._hide_hand_overlay = False
            return
        if isinstance(payload, dict):
            hands_info = payload.get("hands") or []
            self._hide_hand_overlay = bool(payload.get("hide_hand_overlay"))
            mouse_overlay_raw = payload.get("mouse_overlay")
            if isinstance(mouse_overlay_raw, dict):
                bounds = mouse_overlay_raw.get("bounds")
                anchor = mouse_overlay_raw.get("anchor")
                cursor = mouse_overlay_raw.get("cursor")
                if bounds is not None and len(bounds) == 4:
                    raw_active = mouse_overlay_raw.get("active_monitor_index")
                    active_idx = raw_active if isinstance(raw_active, int) else None
                    self._mouse_overlay = {
                        "bounds": tuple(float(v) for v in bounds),
                        "anchor": (
                            tuple(float(v) for v in anchor)
                            if anchor is not None and len(anchor) == 2
                            else None
                        ),
                        "cursor": (
                            tuple(float(v) for v in cursor)
                            if cursor is not None and len(cursor) == 2
                            else None
                        ),
                        "active_monitor_index": active_idx,
                    }
                else:
                    self._mouse_overlay = None
            else:
                self._mouse_overlay = None
        else:
            hands_info = payload
            self._mouse_overlay = None
            self._hide_hand_overlay = False
        normalised: List[dict] = []
        for entry in hands_info:
            if entry is None:
                continue
            if isinstance(entry, dict):
                pts_raw = entry.get("landmarks") or []
                bbox = entry.get("bbox")
                handedness = str(entry.get("handedness") or "")
                label = str(entry.get("label") or "")
                active = bool(entry.get("active"))
            else:
                # Legacy: bare list of (x, y) tuples.
                pts_raw = entry
                bbox = None
                handedness = ""
                label = ""
                active = False
            pts: List[Tuple[float, float]] = []
            for pt in pts_raw:
                if pt is None:
                    continue
                try:
                    pts.append((float(pt[0]), float(pt[1])))
                except Exception:
                    continue
            if not pts and bbox is None:
                continue
            normalised.append({
                "landmarks": pts,
                "bbox": bbox,
                "handedness": handedness,
                "label": label,
                "active": active,
            })
        self._hands_info = normalised

    def clear_video(self, idle_text: str = "") -> None:
        self._image = None
        self._image_w = 0
        self._image_h = 0
        self._hands_info = []
        self._mouse_overlay = None
        self._idle_text = str(idle_text or "")
        self.update()

    # ----- paint -------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        # Qt's raster paint engine — D3D11-backed on Windows, so
        # already GPU-accelerated. drawImage with Format_BGR888
        # uploads to a texture and samples on the GPU; no CPU
        # colour conversion needed.
        # C1 (v1.1.7 diagnostic): time each section so the 2 s log
        # emission below can distinguish drawImage cost from overlay
        # cost from total, and can compare paint duration vs paint-
        # to-paint wall delta.
        t_start = time.perf_counter()
        painter = QPainter(self)
        # macOS: the frame is pre-scaled to ~1:1 in update_frame, so smooth
        # transform buys nothing and the CPU raster engine makes it costly —
        # disable it there. Windows keeps smooth (D3D-accelerated, free).
        painter.setRenderHint(QPainter.SmoothPixmapTransform, sys.platform != "darwin")
        painter.fillRect(self.rect(), self._background)
        target = self._aspect_target()
        drawimage_us = 0
        overlay_us = 0
        if self._image is not None and not self._image.isNull():
            t_before_drawimage = time.perf_counter()
            painter.drawImage(target, self._image)
            t_after_drawimage = time.perf_counter()
            drawimage_us = int((t_after_drawimage - t_before_drawimage) * 1_000_000)
            # In lite paint mode (a fullscreen game is foreground)
            # we keep the cheap parts of the overlay — the hand
            # skeleton + joint dots, which are 2 batched draw calls
            # total — and skip only the expensive parts (mouse-mode
            # control box pixmap blit, per-hand bbox/banner text).
            # The skeleton is what the user actually needs to see
            # while gaming to know whether their hand is being
            # tracked; the bbox/banners are diagnostic noise that's
            # OK to drop. _draw_landmarks now honours
            # _lite_paint_mode internally and gates its own heavy
            # sections.
            # In switch-view drawing mode, skip the hand overlay ENTIRELY
            # (skeleton/bbox/label) so only the camera + the user's drawing
            # (baked into the frame, with its fingertip cursor) show — the
            # drawing is the top layer and the hand-reading graphics vanish.
            if not self._hide_hand_overlay:
                t_before_overlay = time.perf_counter()
                self._draw_landmarks(painter, target)
                overlay_us = int((time.perf_counter() - t_before_overlay) * 1_000_000)
        elif self._idle_text:
            painter.setPen(QPen(self._idle_color, 1))
            painter.setFont(self._idle_font)
            painter.drawText(
                self.rect(),
                Qt.AlignCenter | Qt.TextWordWrap,
                self._idle_text,
            )
        painter.end()
        t_end = time.perf_counter()
        total_us = int((t_end - t_start) * 1_000_000)
        # Paint-rate diagnostic. Prints actual on-screen update
        # rate every 2 s so we can confirm whether the display is
        # tracking the worker's emit rate or coalescing. Also reports
        # avg paintEvent cost + widget/backing size + dpr so a slow
        # macOS paint can be attributed to destination size.
        self._paint_count += 1
        # C1 accumulation for the 2 s summary. Paint-to-paint wall
        # delta uses the interval between successive paintEvent
        # completions — if that delta is close to `total_us` the cap
        # is truly paint cost; if delta is much larger, event
        # coalescing is happening between paints.
        self._paint_timing_total_us += total_us
        self._paint_timing_drawimage_us += drawimage_us
        self._paint_timing_overlay_us += overlay_us
        if total_us > self._paint_timing_max_us:
            self._paint_timing_max_us = total_us
        if self._paint_last_end_monotonic > 0.0:
            delta_us = int((t_end - self._paint_last_end_monotonic) * 1_000_000)
            self._paint_delta_total_us += delta_us
            self._paint_delta_samples += 1
            if delta_us > self._paint_delta_max_us:
                self._paint_delta_max_us = delta_us
        self._paint_last_end_monotonic = t_end
        # C3: slot→paint delta from update_frame timestamp (if this
        # paint was triggered by update_frame; resize / expose paints
        # leave _last_slot_fire_perf==0 and are skipped).
        if self._last_slot_fire_perf > 0.0:
            slot_delta_us = int((t_end - self._last_slot_fire_perf) * 1_000_000)
            self._slot_to_paint_total_us += slot_delta_us
            self._slot_to_paint_samples += 1
            if slot_delta_us > self._slot_to_paint_max_us:
                self._slot_to_paint_max_us = slot_delta_us
            self._last_slot_fire_perf = 0.0
        now = time.monotonic()
        if self._paint_log_at == 0.0:
            self._paint_log_at = now
        elif now - self._paint_log_at >= 2.0:
            rate = self._paint_count / (now - self._paint_log_at)
            avg_total_us = (
                self._paint_timing_total_us // self._paint_count
                if self._paint_count > 0 else 0
            )
            avg_drawimage_us = (
                self._paint_timing_drawimage_us // self._paint_count
                if self._paint_count > 0 else 0
            )
            avg_overlay_us = (
                self._paint_timing_overlay_us // self._paint_count
                if self._paint_count > 0 else 0
            )
            avg_delta_us = (
                self._paint_delta_total_us // self._paint_delta_samples
                if self._paint_delta_samples > 0 else 0
            )
            avg_slot_us = (
                self._slot_to_paint_total_us // self._slot_to_paint_samples
                if self._slot_to_paint_samples > 0 else 0
            )
            src_w = self._image_w if self._image is not None else 0
            src_h = self._image_h if self._image is not None else 0
            tgt_w = target.width() if self._image is not None else 0
            tgt_h = target.height() if self._image is not None else 0
            try:
                sys.stderr.write(
                    f"[gpu_video] paint rate: {rate:.1f} fps "
                    f"(widget={self.objectName() or type(self).__name__}) | "
                    f"avg paint={avg_total_us / 1000:.2f}ms "
                    f"(drawImage={avg_drawimage_us / 1000:.2f}ms, "
                    f"overlay={avg_overlay_us / 1000:.2f}ms) "
                    f"max={self._paint_timing_max_us / 1000:.2f}ms | "
                    f"paint→paint delta avg={avg_delta_us / 1000:.2f}ms "
                    f"max={self._paint_delta_max_us / 1000:.2f}ms | "
                    f"slot→paint avg={avg_slot_us / 1000:.2f}ms "
                    f"max={self._slot_to_paint_max_us / 1000:.2f}ms "
                    f"(n={self._slot_to_paint_samples}) | "
                    f"src={src_w}x{src_h} target={tgt_w}x{tgt_h}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            self._paint_count = 0
            self._paint_ms_accum = 0.0
            self._paint_log_at = now
            self._paint_timing_total_us = 0
            self._paint_timing_drawimage_us = 0
            self._paint_timing_overlay_us = 0
            self._paint_timing_max_us = 0
            self._paint_delta_total_us = 0
            self._paint_delta_max_us = 0
            self._paint_delta_samples = 0
            self._slot_to_paint_total_us = 0
            self._slot_to_paint_max_us = 0
            self._slot_to_paint_samples = 0

    def _aspect_target(self) -> QRect:
        if self._image_w <= 0 or self._image_h <= 0:
            return self.rect()
        wa = max(1, self.width())
        ha = max(1, self.height())
        scale = min(wa / float(self._image_w), ha / float(self._image_h))
        if scale <= 0:
            return self.rect()
        tw = max(1, int(self._image_w * scale))
        th = max(1, int(self._image_h * scale))
        x = (wa - tw) // 2
        y = (ha - th) // 2
        return QRect(x, y, tw, th)

    def _draw_landmarks(self, painter: QPainter, target: QRect) -> None:
        tx = target.x()
        ty = target.y()
        tw = target.width()
        th = target.height()

        # Mouse-mode control area. Painted first so the hand
        # skeleton/bbox overlay on top — keeps the box visually
        # behind the hand. Faint red fill + bold red border + a
        # small "Mouse control area" label so the user
        # immediately knows where to keep their hand.
        # Heavy: monitor-layout pixmap blit. Skipped in lite paint
        # mode (fullscreen game) so each paint stays cheap.
        if self._mouse_overlay is not None and not self._lite_paint_mode:
            bx1, by1, bx2, by2 = self._mouse_overlay["bounds"]
            rx1 = bx1 * tw + tx
            ry1 = by1 * th + ty
            rx2 = bx2 * tw + tx
            ry2 = by2 * th + ty
            box_rect = QRectF(rx1, ry1, max(0.0, rx2 - rx1), max(0.0, ry2 - ry1))
            painter.fillRect(box_rect, self._mouse_box_fill_color)
            painter.setPen(QPen(self._mouse_box_color, 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(box_rect)
            label = "Mouse control area"
            painter.setFont(self._banner_font)
            metrics = QFontMetrics(self._banner_font)
            text_w = metrics.horizontalAdvance(label)
            label_h = metrics.height()
            if ry1 - label_h - 6 >= ty:
                bg_y = ry1 - label_h - 4
            else:
                bg_y = ry1 + 2
            bg_rect = QRectF(
                max(tx, rx1),
                bg_y,
                min(text_w + 10.0, tw - (max(tx, rx1) - tx)),
                label_h + 2.0,
            )
            painter.fillRect(bg_rect, self._banner_bg_color)
            painter.setPen(QPen(self._mouse_box_color, 1))
            painter.drawRect(bg_rect)
            painter.setPen(QPen(self._mouse_box_color, 1))
            painter.drawText(QPointF(bg_rect.x() + 5.0, bg_rect.y() + label_h - 3.0), label)
            # Anchor crosshair removed — it was a static "+" in
            # the box while the cursor dot moved, which read as a
            # stale duplicate cursor. The box itself + moving
            # cursor are enough.
            # Monitor layout: cached. The layout depends only on
            # the box rect + screen geometry, both of which change
            # rarely. Build a pixmap once when the signature
            # changes, then blit every frame. Was the largest
            # per-frame paint cost in mouse mode — saves ~1-2 ms
            # under steady operation.
            cursor = self._mouse_overlay.get("cursor")
            screens = QGuiApplication.screens()
            map_origin = None  # (mx, my, scale, v_left, v_top, v_w, v_h)
            if screens:
                primary = QGuiApplication.primaryScreen()
                active_idx = self._mouse_overlay.get("active_monitor_index")
                # Single-monitor mode: the chosen monitor IS the
                # whole map (cursor mapping is constrained to it,
                # so visually it makes sense to fill the box with
                # just that monitor). Compute layout from only the
                # chosen screen's geometry so the rest of the
                # function — cache signature, pixmap rebuild, blit,
                # cursor dot — operates on a single rectangle the
                # same way it would for an all-monitors layout.
                #
                # Multi-monitor / All-Monitors mode: keep the
                # virtual-desktop union with each monitor in
                # proportional position.
                single_geo = None
                if isinstance(active_idx, int) and 0 <= active_idx < len(screens):
                    try:
                        single_geo = screens[active_idx].geometry()
                    except Exception:
                        single_geo = None
                if single_geo is not None:
                    v_left = single_geo.x()
                    v_top = single_geo.y()
                    v_w = max(1, single_geo.width())
                    v_h = max(1, single_geo.height())
                else:
                    v_left = min(s.geometry().x() for s in screens)
                    v_top = min(s.geometry().y() for s in screens)
                    v_right = max(s.geometry().x() + s.geometry().width() for s in screens)
                    v_bottom = max(s.geometry().y() + s.geometry().height() for s in screens)
                    v_w = max(1, v_right - v_left)
                    v_h = max(1, v_bottom - v_top)
                inset = 12.0
                inner_w = max(40.0, box_rect.width() - 2 * inset)
                inner_h = max(40.0, box_rect.height() - 2 * inset)
                scale = min(inner_w / float(v_w), inner_h / float(v_h))
                map_w = float(v_w) * scale
                map_h = float(v_h) * scale
                mx = box_rect.x() + (box_rect.width() - map_w) / 2.0
                my = box_rect.y() + (box_rect.height() - map_h) / 2.0
                map_origin = (mx, my, scale, v_left, v_top, v_w, v_h)
                # Cache key. Box position + size, computed v-rect,
                # screen list, and active_idx — covers every input
                # that changes the rendered layout including
                # switching from "All Monitors" to a single screen.
                signature = (
                    int(round(box_rect.x())),
                    int(round(box_rect.y())),
                    int(round(box_rect.width())),
                    int(round(box_rect.height())),
                    int(v_left), int(v_top), int(v_w), int(v_h),
                    tuple(
                        (int(s.geometry().x()), int(s.geometry().y()),
                         int(s.geometry().width()), int(s.geometry().height()),
                         s == primary)
                        for s in screens
                    ),
                    active_idx if isinstance(active_idx, int) else -1,
                )
                if signature != self._mouse_box_signature or self._mouse_box_pixmap is None:
                    pm_w = max(1, int(round(box_rect.width())))
                    pm_h = max(1, int(round(box_rect.height())))
                    pm = QPixmap(pm_w, pm_h)
                    pm.fill(Qt.transparent)
                    pm_painter = QPainter(pm)
                    try:
                        pm_painter.setRenderHint(QPainter.Antialiasing, True)
                        # Monitor map sits centered inside the
                        # pixmap. mx/my computed above are in widget
                        # coords; convert to pixmap-local coords by
                        # subtracting box_rect origin.
                        local_mx = mx - box_rect.x()
                        local_my = my - box_rect.y()
                        pm_painter.fillRect(QRectF(local_mx, local_my, map_w, map_h), QColor(8, 14, 26, 140))
                        # Per-monitor fill rule (mirrors the OpenCV
                        # overlay in mouse_overlay.py and the Save
                        # Locations preview):
                        #   active_idx is None -> historical look
                        #     (primary in green, others in blue)
                        #   active_idx == this screen's index -> bright
                        #     accent green; non-active screens get a
                        #     dim slate so the user can see at a
                        #     glance which display the cursor is
                        #     constrained to.
                        accent_fill = QColor(140, 220, 184, 220)
                        accent_border = QColor(228, 240, 232, 240)
                        neutral_primary = QColor(58, 122, 96, 200)
                        neutral_secondary = QColor(39, 72, 108, 200)
                        neutral_border = QColor(228, 236, 243, 220)
                        if single_geo is not None:
                            # Single-monitor mode: that one screen IS
                            # the whole map. Fill the entire area
                            # (which equals single_geo at this point)
                            # with the accent color so the user sees
                            # one bold rectangle, not a tiny green
                            # speck inside a multi-screen layout.
                            pm_painter.fillRect(QRectF(local_mx, local_my, map_w, map_h), accent_fill)
                            pm_painter.setPen(QPen(accent_border, 2))
                            pm_painter.setBrush(Qt.NoBrush)
                            pm_painter.drawRect(QRectF(local_mx, local_my, map_w, map_h))
                            label = f"Monitor {active_idx + 1}"
                            pm_painter.setPen(QPen(QColor(245, 250, 252, 240)))
                            pm_painter.drawText(
                                QRectF(local_mx, local_my, map_w, map_h),
                                int(Qt.AlignCenter),
                                label,
                            )
                        else:
                            for idx, screen in enumerate(screens):
                                geo = screen.geometry()
                                sx = local_mx + (geo.x() - v_left) * scale
                                sy = local_my + (geo.y() - v_top) * scale
                                sw = geo.width() * scale
                                sh = geo.height() * scale
                                fill = neutral_primary if screen == primary else neutral_secondary
                                border = neutral_border
                                pm_painter.fillRect(QRectF(sx, sy, sw, sh), fill)
                                pm_painter.setPen(QPen(border, 1))
                                pm_painter.setBrush(Qt.NoBrush)
                                pm_painter.drawRect(QRectF(sx, sy, sw, sh))
                    finally:
                        pm_painter.end()
                    self._mouse_box_pixmap = pm
                    self._mouse_box_signature = signature
                # Blit cached layout.
                if self._mouse_box_pixmap is not None:
                    painter.drawPixmap(int(round(box_rect.x())), int(round(box_rect.y())), self._mouse_box_pixmap)
            if cursor is not None:
                if map_origin is not None:
                    mx, my, scale, v_left, v_top, v_w, v_h = map_origin
                    # cursor is normalized [0, 1] across the full
                    # virtual desktop. Project into the monitor
                    # layout coordinate space.
                    cx = mx + cursor[0] * float(v_w) * scale
                    cy = my + cursor[1] * float(v_h) * scale
                else:
                    cx = box_rect.x() + cursor[0] * box_rect.width()
                    cy = box_rect.y() + cursor[1] * box_rect.height()
                # Two-layer dot: white core for visibility on any
                # background, mint ring for the Touchless theme.
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(255, 255, 255, 250)))
                painter.drawEllipse(QPointF(cx, cy), 6.0, 6.0)
                painter.setPen(QPen(QColor(36, 220, 184, 240), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(cx, cy), 10.0, 10.0)

        if not self._hands_info:
            return

        # C8 effective overlay level computed with layered precedence.
        # See __init__ docstring for level semantics. Called after the
        # mouse-overlay drawing above so mouse control-box overlay
        # remains independent of the skeleton/bbox/banner level.
        if self._lite_paint_mode:
            # Fullscreen game: cap at 1 (skeleton only). Preserves the
            # pre-C8 lite-paint behaviour (fullscreen games drop the
            # bbox + banner cost).
            effective_level = min(self._overlay_level, 1)
        else:
            effective_level = self._overlay_level

        if effective_level == 0:
            return

        # Per-hand bbox + banner drawn at level 2 (bbox rects only,
        # text banners elided) and level 3 (default: bbox + banner
        # text). Drawn first so the skeleton + joints paint over
        # them (avoids the bbox edge cutting through a fingertip).
        if effective_level >= 2:
            painter.save()
            painter.setFont(self._banner_font)
            metrics = QFontMetrics(self._banner_font)
            banner_h = metrics.height()
            self._draw_hand_banners(
                painter, tx, ty, tw, th, metrics, banner_h,
                include_text=(effective_level >= 3),
            )
            painter.restore()

        # Skeleton + joints — cheap (2 batched paint ops total),
        # drawn at every level >= 1.
        self._draw_hand_skeleton(painter, tx, ty, tw, th)

    def _draw_hand_banners(
        self, painter, tx, ty, tw, th, metrics, banner_h, include_text: bool = True,
    ) -> None:
        for hand in self._hands_info:
            bbox = hand.get("bbox")
            if bbox is None:
                continue
            color = self._bbox_active_color if hand.get("active") else self._bbox_inactive_color
            bx, by, bw, bh = bbox
            rx = bx * tw + tx
            ry = by * th + ty
            rw = bw * tw
            rh = bh * th
            rect = QRectF(rx, ry, rw, rh)
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)

            # Text banner elided at overlay level 2 (bbox rect kept,
            # text + text-background rect skipped). Saves the per-hand
            # text-metrics + QPainter.drawText chain per paint.
            if not include_text:
                continue

            # Banner: "Right | gesture" when the hand has a
            # recognized gesture, "Right" when neutral. Empty
            # handedness falls back to just the label (or nothing).
            handedness = hand.get("handedness", "") or ""
            label = hand.get("label", "") or ""
            if label:
                banner = f"{handedness} | {label}" if handedness else label
            else:
                banner = handedness
            if not banner:
                continue
            text_w = metrics.horizontalAdvance(banner)
            # Sit the banner just above the bbox; if the box is at
            # the top of the frame, drop the banner inside the box
            # instead so it never gets clipped off-screen.
            if ry - banner_h - 6 >= ty:
                bg_y = ry - banner_h - 4
            else:
                bg_y = ry + 2
            bg_rect = QRectF(
                max(tx, rx),
                bg_y,
                min(text_w + 10.0, tw - (max(tx, rx) - tx)),
                banner_h + 2.0,
            )
            painter.fillRect(bg_rect, self._banner_bg_color)
            painter.setPen(QPen(color, 1))
            painter.drawRect(bg_rect)
            painter.setPen(QPen(self._banner_text_color, 1))
            text_pt = QPointF(bg_rect.x() + 5.0, bg_rect.y() + banner_h - 3.0)
            painter.drawText(text_pt, banner)

    def _draw_hand_skeleton(self, painter, tx, ty, tw, th) -> None:
        # Batch every connection across every hand into one drawLines
        # call and every joint into one drawPoints call. Replaces what
        # used to be ~84 individual painter.draw* calls per paint
        # (2 hands × (21 connections + 21 joints)) with 2 batched
        # paint ops + 2 pen swaps total. Each Qt paint call has
        # per-call overhead (transform, pen state, antialias setup);
        # batching collapses that overhead to constant.
        all_lines: list[QLineF] = []
        all_points: list[QPointF] = []
        for hand in self._hands_info:
            pts = hand.get("landmarks") or []
            n = len(pts)
            if n == 0:
                continue
            for a, b in _HAND_CONNECTIONS:
                if a >= n or b >= n:
                    continue
                ax, ay = pts[a][0], pts[a][1]
                bx, by = pts[b][0], pts[b][1]
                all_lines.append(QLineF(
                    ax * tw + tx, ay * th + ty,
                    bx * tw + tx, by * th + ty,
                ))
            for pt in pts:
                all_points.append(QPointF(pt[0] * tw + tx, pt[1] * th + ty))
        if all_lines:
            painter.setPen(self._connection_pen)
            painter.drawLines(all_lines)
        if all_points:
            painter.setPen(self._landmark_pen)
            painter.drawPoints(all_points)

# Author: Konstantin Markov

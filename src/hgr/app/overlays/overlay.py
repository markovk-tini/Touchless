from __future__ import annotations

import math
import sys
import time
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QColorDialog, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from ..ui.native_overlay import apply_overlay


def _front_overlay(widget) -> None:
    """Order a HUD overlay to the front WITHOUT stealing focus.

    On macOS, raise_() activates the whole app, pulling focus off whatever the
    user is actually controlling (Chrome, etc.); the nonactivating-NSPanel path
    in apply_overlay() orders the window front without activating. On Windows a
    Qt.Tool + WA_ShowWithoutActivating window's raise_() is a plain Z-order
    raise that does NOT steal focus, so keep the existing behavior there.
    """
    if sys.platform == "darwin":
        apply_overlay(widget)
    else:
        widget.raise_()


class HelloOverlay(QWidget):
    def __init__(self, font_size: int = 72, parent=None):
        super().__init__(parent)
        flags = Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        transparent_flag = getattr(Qt, "WindowTransparentForInput", None)
        if transparent_flag is not None:
            flags |= transparent_flag
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.message = "HELLO USER!"
        self.font_size = font_size
        self.current_color = QColor("#1DE9B6")
        self._color_cycle = cycle([
            "#1DE9B6",
            "#7C4DFF",
            "#FF5252",
            "#FFD740",
            "#40C4FF",
            "#69F0AE",
            "#FF6E40",
        ])
        self._resize_to_primary_screen()

    def _resize_to_primary_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 800)
            return
        self.setGeometry(screen.availableGeometry())

    def show_message(self) -> None:
        self.current_color = QColor(next(self._color_cycle))
        self._resize_to_primary_screen()
        self.show()
        _front_overlay(self)
        self.update()

    def hide_message(self) -> None:
        self.hide()

    def set_font_size(self, font_size: int) -> None:
        self.font_size = font_size
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(self.current_color)
        font = QFont("Arial", self.font_size, QFont.Bold)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self.message)


class DrawingSettingsDialog(QDialog):
    def __init__(self, color: QColor, thickness: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Drawing Settings")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.resize(420, 260)
        self._selected_color = QColor(color)
        self._auto_color_opened = False

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        title = QLabel("Drawing Settings")
        title.setStyleSheet("font-size: 20px; font-weight: 800;")
        root.addWidget(title)

        preview_row = QHBoxLayout()
        preview_label = QLabel("Current color")
        self.preview_chip = QFrame()
        self.preview_chip.setFixedSize(64, 28)
        preview_row.addWidget(preview_label)
        preview_row.addWidget(self.preview_chip)
        preview_row.addStretch(1)
        root.addLayout(preview_row)

        self.color_button = QPushButton("Open Color Wheel")
        self.color_button.clicked.connect(self._open_color_picker)
        root.addWidget(self.color_button, 0, Qt.AlignLeft)

        thickness_row = QHBoxLayout()
        thickness_label = QLabel("Brush thickness")
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setRange(2, 48)
        self.thickness_slider.setValue(max(2, thickness))
        self.thickness_value = QLabel(str(self.thickness_slider.value()))
        self.thickness_slider.valueChanged.connect(lambda v: self.thickness_value.setText(str(v)))
        thickness_row.addWidget(thickness_label)
        thickness_row.addWidget(self.thickness_slider, 1)
        thickness_row.addWidget(self.thickness_value)
        root.addLayout(thickness_row)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(apply_btn)
        root.addLayout(buttons)

        self.setStyleSheet(
            """
            QDialog {
                background: #0F172A;
                color: #E5F6FF;
                border: 1px solid rgba(29,233,182,0.35);
            }
            QLabel { color: #E5F6FF; }
            QPushButton {
                background-color: #0B3D91;
                color: #E5F6FF;
                border: 1px solid rgba(29,233,182,0.35);
                border-radius: 12px;
                padding: 9px 14px;
                font-weight: 700;
            }
            QPushButton:hover { border: 1px solid #1DE9B6; }
            QSlider::groove:horizontal {
                height: 6px;
                border-radius: 3px;
                background: rgba(255,255,255,0.14);
            }
            QSlider::handle:horizontal {
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
                background: #1DE9B6;
            }
            """
        )
        self._refresh_preview()

    @property
    def selected_color(self) -> QColor:
        return QColor(self._selected_color)

    @property
    def selected_thickness(self) -> int:
        return int(self.thickness_slider.value())

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._auto_color_opened:
            self._auto_color_opened = True
            QTimer.singleShot(0, self._open_color_picker)

    def _refresh_preview(self) -> None:
        self.preview_chip.setStyleSheet(
            f"background: {self._selected_color.name()}; border-radius: 8px; border: 1px solid rgba(255,255,255,0.22);"
        )

    def _open_color_picker(self) -> None:
        picker = QColorDialog(self._selected_color, self)
        picker.setWindowTitle("Choose Drawing Color")
        picker.setOption(QColorDialog.DontUseNativeDialog, False)
        if picker.exec() == QDialog.Accepted:
            chosen = picker.currentColor()
            if chosen.isValid():
                self._selected_color = QColor(chosen)
                self._refresh_preview()


class ScreenDrawOverlay(QWidget):
    def __init__(self, color: str = "#FFFFFF", thickness: int = 6, parent=None):
        super().__init__(parent)
        flags = Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        transparent_flag = getattr(Qt, "WindowTransparentForInput", None)
        if transparent_flag is not None:
            flags |= transparent_flag
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        self.brush_color = QColor(color)
        self.brush_thickness = int(max(2, thickness))
        self.eraser_thickness = int(max(6, thickness * 2))
        self.eraser_mode = "normal"
        self._canvas = QImage()
        self._cursor_pos: Optional[QPointF] = None
        self._cursor_mode = "hidden"  # hidden / hover / draw / erase
        self._last_draw_point: Optional[QPointF] = None
        self._history: list[tuple[QImage, list[dict], bool]] = []
        self._history_limit = 24
        self._strokes: list[dict] = []
        self._active_stroke_points: list[tuple[float, float]] = []
        self._raster_dirty = False
        self.shape_mode = False
        # Pinch-grab live transform. Translation + scale: the user
        # can move strokes around with a one-hand pinch and stretch
        # / squish them with a two-hand pinch (distance change
        # between the palms drives scale). Per-stroke movement is
        # still Phase 2 — for now the whole canvas transforms as a
        # unit. The rasteriser is set _raster_dirty after a bake so
        # it knows the stroke list no longer matches the pixels.
        self._grab_dx_norm: float = 0.0
        self._grab_dy_norm: float = 0.0
        self._grab_scale: float = 1.0
        self._resize_to_screen()

    def set_shape_mode(self, enabled: bool) -> None:
        self.shape_mode = bool(enabled)

    def _resize_to_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.setGeometry(0, 0, 1280, 720)
        else:
            self.setGeometry(screen.geometry())
        self._ensure_canvas_size()

    def _ensure_canvas_size(self) -> None:
        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            return
        if self._canvas.size() == size:
            return
        new_canvas = QImage(size, QImage.Format_ARGB32_Premultiplied)
        new_canvas.fill(Qt.transparent)
        if not self._canvas.isNull():
            painter = QPainter(new_canvas)
            painter.drawImage(0, 0, self._canvas)
            painter.end()
        self._canvas = new_canvas

    def _clone_canvas(self) -> QImage:
        self._ensure_canvas_size()
        return self._canvas.copy() if not self._canvas.isNull() else QImage()

    def _clone_strokes(self) -> list[dict]:
        clones: list[dict] = []
        for stroke in self._strokes:
            clones.append(
                {
                    "color": QColor(stroke["color"]),
                    "thickness": int(stroke["thickness"]),
                    "points": [(float(x), float(y)) for x, y in stroke["points"]],
                }
            )
        return clones

    def _rerender_from_strokes(self) -> None:
        self._ensure_canvas_size()
        self._canvas.fill(Qt.transparent)
        painter = QPainter(self._canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        for stroke in self._strokes:
            points = stroke.get("points") or []
            if len(points) < 2:
                continue
            pen = QPen(QColor(stroke["color"]))
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setWidth(int(stroke["thickness"]))
            painter.setPen(pen)
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        painter.end()

    def push_undo_state(self) -> None:
        snapshot = self._clone_canvas()
        self._history.append((snapshot, self._clone_strokes(), bool(self._raster_dirty)))
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    def undo_last_action(self) -> bool:
        if not self._history:
            return False
        canvas, strokes, raster_dirty = self._history.pop()
        self._canvas = canvas
        self._strokes = strokes
        self._raster_dirty = bool(raster_dirty)
        self._active_stroke_points = []
        self._last_draw_point = None
        self.update()
        return True

    def show_overlay(self) -> None:
        self._resize_to_screen()
        self.show()
        _front_overlay(self)
        self.update()

    def hide_overlay(self) -> None:
        self.hide()

    def set_brush(self, color: QColor | str, thickness: int) -> None:
        self.brush_color = QColor(color)
        self.brush_thickness = int(max(2, thickness))
        self.update()

    def set_eraser(self, thickness: int, mode: str = "normal") -> None:
        self.eraser_thickness = int(max(6, thickness))
        new_mode = "stroke" if str(mode).strip().lower() == "stroke" else "normal"
        if new_mode == "stroke" and self._raster_dirty and self._strokes:
            self._raster_dirty = False
            self._rerender_from_strokes()
        self.eraser_mode = new_mode
        self.update()

    def set_eraser_settings(self, thickness: int, mode: str = "normal") -> None:
        self.set_eraser(thickness, mode)

    def clear_canvas(self) -> None:
        self._ensure_canvas_size()
        self._canvas.fill(Qt.transparent)
        self._strokes = []
        self._active_stroke_points = []
        self._raster_dirty = False
        self._last_draw_point = None
        self.update()

    def set_cursor(self, pos: Optional[QPointF], mode: str) -> None:
        self._cursor_pos = QPointF(pos) if pos is not None else None
        self._cursor_mode = mode
        if mode != "draw":
            self._last_draw_point = None
        self.update()

    def begin_draw(self, pos: QPointF) -> None:
        self._last_draw_point = QPointF(pos)
        self._active_stroke_points = [(float(pos.x()), float(pos.y()))]
        self.set_cursor(pos, "draw")

    def draw_to(self, pos: QPointF) -> None:
        self._ensure_canvas_size()
        if self._last_draw_point is None:
            self._last_draw_point = QPointF(pos)
            if not self._active_stroke_points:
                self._active_stroke_points = [(float(pos.x()), float(pos.y()))]
        painter = QPainter(self._canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.brush_color)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setWidth(self.brush_thickness)
        painter.setPen(pen)
        painter.drawLine(self._last_draw_point, QPointF(pos))
        painter.end()
        self._active_stroke_points.append((float(pos.x()), float(pos.y())))
        self._last_draw_point = QPointF(pos)
        self._cursor_pos = QPointF(pos)
        self._cursor_mode = "draw"
        self.update()

    @staticmethod
    def _point_to_segment_distance_sq(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        abx = bx - ax
        aby = by - ay
        if abs(abx) < 1e-9 and abs(aby) < 1e-9:
            dx = px - ax
            dy = py - ay
            return dx * dx + dy * dy
        apx = px - ax
        apy = py - ay
        denom = abx * abx + aby * aby
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        cx = ax + t * abx
        cy = ay + t * aby
        dx = px - cx
        dy = py - cy
        return dx * dx + dy * dy

    def _stroke_hits_position(self, stroke: dict, px: float, py: float, radius: float) -> bool:
        points = stroke.get("points") or []
        if not points:
            return False
        threshold = max(float(radius), float(stroke.get("thickness", 0)) * 0.5 + 2.0)
        limit_sq = threshold * threshold
        if len(points) == 1:
            sx, sy = points[0]
            dx = sx - px
            dy = sy - py
            return dx * dx + dy * dy <= limit_sq
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            if self._point_to_segment_distance_sq(px, py, float(ax), float(ay), float(bx), float(by)) <= limit_sq:
                return True
        return False

    def erase_at(self, pos: QPointF) -> None:
        self._ensure_canvas_size()
        radius = max(8, int(self.eraser_thickness * 0.5))
        if self.eraser_mode == "stroke":
            if self._raster_dirty and self._strokes:
                self._raster_dirty = False
                self._rerender_from_strokes()
            px = float(pos.x())
            py = float(pos.y())
            hit_index = None
            for idx in range(len(self._strokes) - 1, -1, -1):
                stroke = self._strokes[idx]
                if self._stroke_hits_position(stroke, px, py, float(radius)):
                    hit_index = idx
                    break
            if hit_index is not None:
                self._strokes.pop(hit_index)
                self._rerender_from_strokes()
            else:
                self._cursor_pos = QPointF(pos)
                self._cursor_mode = "erase"
                self.update()
                return
        else:
            painter = QPainter(self._canvas)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.setPen(Qt.NoPen)
            painter.setBrush(Qt.transparent)
            painter.drawEllipse(pos, radius, radius)
            painter.end()
            self._raster_dirty = True
        self._cursor_pos = QPointF(pos)
        self._cursor_mode = "erase"
        self.update()

    def end_stroke(self) -> None:
        if self._active_stroke_points:
            if len(self._active_stroke_points) == 1:
                x, y = self._active_stroke_points[0]
                self._active_stroke_points.append((x + 0.01, y + 0.01))
            points = list(self._active_stroke_points)
            if self.shape_mode:
                snapped = self._snap_stroke_to_shape(points)
                if snapped and len(snapped) >= 2:
                    points = snapped
            self._strokes.append(
                {
                    "color": QColor(self.brush_color),
                    "thickness": int(self.brush_thickness),
                    "points": points,
                }
            )
            self._active_stroke_points = []
            if self.shape_mode:
                self._rerender_from_strokes()
        self._last_draw_point = None
        if self._cursor_mode == "draw":
            self._cursor_mode = "hover"
        self.update()

    def _snap_stroke_to_shape(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        # Strategy: score each shape independently against a resampled
        # + smoothed copy of the stroke, then snap to the best fit
        # only if its residual is acceptable. If everything fits
        # poorly we keep the original freehand stroke — that's the
        # main accuracy win over the previous version, which always
        # snapped closed strokes to a bbox rectangle regardless of
        # whether the stroke actually looked like one.
        pts = [(float(x), float(y)) for x, y in points]
        n = len(pts)
        if n < 3:
            return pts
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        span = max(width, height, 1.0)
        if span < 14.0:
            return pts

        # Resample to uniform arc-length spacing so dense slow segments
        # don't bias residuals, then smooth out fingertip jitter.
        sampled = self._resample_uniform(pts, count=128)
        if len(sampled) < 8:
            sampled = pts
        smooth = self._smooth_points(sampled, window=3)

        aspect = min(width, height) / max(width, height, 1.0)
        start_end_dist = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
        # Relaxed closure threshold: real users almost never close
        # perfectly. Match anything within ~40% of span.
        closed = start_end_dist < span * 0.40

        # --- Line score: mean perpendicular distance to chord(P0, Pn). ---
        line_residual = self._line_residual_norm(smooth, pts[0], pts[-1]) / span

        if not closed:
            if line_residual < 0.045:
                return [pts[0], pts[-1]]
            return pts

        # --- Centroid + radii (used by circle + ellipse). ---
        n_s = len(smooth)
        ctr_x = sum(p[0] for p in smooth) / n_s
        ctr_y = sum(p[1] for p in smooth) / n_s
        radii = [math.hypot(p[0] - ctr_x, p[1] - ctr_y) for p in smooth]
        avg_r = sum(radii) / n_s
        if avg_r > 0:
            circle_residual = sum(abs(r - avg_r) for r in radii) / n_s / avg_r
        else:
            circle_residual = 1.0

        # --- Step 1: try polygon detection first. ---
        # Corner-based polygons take priority over residual ranking
        # because a clean triangle's "rect residual" is moderate
        # (two edges align with the bbox), but its corner residual
        # against the actual 3 vertices is tiny. We try k=4 and k=3
        # and pick the better fit among polygons whose corners are
        # genuine sharp turns.
        POLY_RESIDUAL_GATE = 0.05  # mean point-to-edge distance / span
        MIN_CORNER_TURN = 0.55     # ~31° interior turn — eliminates circles
        poly_choice: tuple[str, list[tuple[float, float]], float] | None = None
        for k in (4, 3):
            picked = self._dominant_corners(smooth, k=k, span=span)
            if picked is None or len(picked) != k:
                continue
            min_turn = self._min_turn_score(smooth, picked, span)
            if min_turn < MIN_CORNER_TURN:
                continue
            verts = self._order_polygon_ccw([smooth[i] for i in picked], ctr_x, ctr_y)
            residual = self._polygon_edge_residual(smooth, verts) / span
            if residual > POLY_RESIDUAL_GATE:
                continue
            kind = "quad" if k == 4 else "triangle"
            if poly_choice is None or residual < poly_choice[2]:
                poly_choice = (kind, verts, residual)

        # --- Step 2: circle / ellipse via radial std-dev. ---
        var_r = sum((r - avg_r) * (r - avg_r) for r in radii) / n_s
        std_r = math.sqrt(var_r)
        # Std-dev normalized by SPAN, not avg_r, so this is directly
        # comparable to rect residuals. A clean circle has std ~ noise
        # level; a square has std ~ (corner radius - edge radius)/3 ~
        # several percent of span.
        circle_radial_std = std_r / span if span > 0 else 1.0

        # Ellipse via PCA-rotated bbox (only used when aspect < ~0.88).
        pca_angle = self._principal_axis_angle(smooth, ctr_x, ctr_y)
        cos_a = math.cos(-pca_angle)
        sin_a = math.sin(-pca_angle)
        rot_pts = [
            ((p[0] - ctr_x) * cos_a - (p[1] - ctr_y) * sin_a,
             (p[0] - ctr_x) * sin_a + (p[1] - ctr_y) * cos_a)
            for p in smooth
        ]
        r_min_x = min(p[0] for p in rot_pts)
        r_max_x = max(p[0] for p in rot_pts)
        r_min_y = min(p[1] for p in rot_pts)
        r_max_y = max(p[1] for p in rot_pts)
        rx_half = (r_max_x - r_min_x) / 2.0
        ry_half = (r_max_y - r_min_y) / 2.0
        ec_x = (r_min_x + r_max_x) / 2.0
        ec_y = (r_min_y + r_max_y) / 2.0
        if rx_half > 1.0 and ry_half > 1.0:
            ellipse_sum = 0.0
            for px, py in rot_pts:
                lx = (px - ec_x) / rx_half
                ly = (py - ec_y) / ry_half
                r_local = math.hypot(lx, ly)
                # Approximate: scale point to ellipse boundary, take
                # remaining radial difference in world units.
                ellipse_sum += abs(r_local - 1.0) * min(rx_half, ry_half)
            ellipse_residual = ellipse_sum / n_s / span
        else:
            ellipse_residual = 1.0

        CURVE_RESIDUAL_GATE = 0.03

        # --- Step 3: decide. ---
        # Polygon wins outright when present (we already filtered by
        # the polygon residual gate). When both quad and triangle
        # qualify we kept the lower-residual one.
        if poly_choice is not None:
            kind, verts, _ = poly_choice
            return verts + [verts[0]]

        if aspect > 0.88 and circle_radial_std < CURVE_RESIDUAL_GATE:
            steps = 96
            result: list[tuple[float, float]] = []
            for i in range(steps + 1):
                t = 2.0 * math.pi * i / steps
                result.append((ctr_x + avg_r * math.cos(t), ctr_y + avg_r * math.sin(t)))
            return result

        if aspect < 0.88 and ellipse_residual < CURVE_RESIDUAL_GATE:
            steps = 96
            cos_p = math.cos(pca_angle)
            sin_p = math.sin(pca_angle)
            result = []
            for i in range(steps + 1):
                t = 2.0 * math.pi * i / steps
                lx = ec_x + rx_half * math.cos(t)
                ly = ec_y + ry_half * math.sin(t)
                wx = ctr_x + lx * cos_p - ly * sin_p
                wy = ctr_y + lx * sin_p + ly * cos_p
                result.append((wx, wy))
            return result

        # No confident snap → keep freehand. (The old version always
        # snapped to a bbox rect here, which was the main source of
        # "why did my squiggle turn into a rectangle?" complaints.)
        return pts

    def _resample_uniform(self, pts: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
        if len(pts) < 2 or count < 2:
            return list(pts)
        seg_lens = []
        total = 0.0
        for i in range(1, len(pts)):
            d = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            seg_lens.append(d)
            total += d
        if total <= 0.0:
            return list(pts)
        step = total / (count - 1)
        out: list[tuple[float, float]] = [pts[0]]
        traveled = 0.0
        target = step
        i = 1
        # Walk segments, emitting points at every `step` of accumulated
        # arc length. Linear interp inside each segment.
        while i < len(pts) and len(out) < count - 1:
            seg = seg_lens[i - 1]
            if seg <= 0.0:
                i += 1
                continue
            while target <= traveled + seg and len(out) < count - 1:
                t = (target - traveled) / seg
                ax, ay = pts[i - 1]
                bx, by = pts[i]
                out.append((ax + (bx - ax) * t, ay + (by - ay) * t))
                target += step
            traveled += seg
            i += 1
        out.append(pts[-1])
        return out

    def _smooth_points(self, pts: list[tuple[float, float]], window: int) -> list[tuple[float, float]]:
        if window <= 1 or len(pts) < 3:
            return list(pts)
        half = window // 2
        n = len(pts)
        out: list[tuple[float, float]] = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            sx = 0.0
            sy = 0.0
            for j in range(lo, hi):
                sx += pts[j][0]
                sy += pts[j][1]
            denom = float(hi - lo)
            out.append((sx / denom, sy / denom))
        return out

    def _line_residual_norm(
        self,
        pts: list[tuple[float, float]],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        L = math.hypot(dx, dy)
        if L <= 0.0:
            return 0.0
        total = 0.0
        for px, py in pts:
            # Perpendicular distance from P to line AB.
            total += abs((dy * px) - (dx * py) + (bx * ay) - (by * ax)) / L
        return total / len(pts)

    def _principal_axis_angle(
        self,
        pts: list[tuple[float, float]],
        cx: float,
        cy: float,
    ) -> float:
        # 2x2 covariance matrix and closed-form dominant eigenvector.
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for x, y in pts:
            dx = x - cx
            dy = y - cy
            sxx += dx * dx
            syy += dy * dy
            sxy += dx * dy
        return 0.5 * math.atan2(2.0 * sxy, sxx - syy)

    def _dominant_corners(
        self,
        pts: list[tuple[float, float]],
        k: int,
        span: float,
    ) -> list[int] | None:
        # Score every point by the supplement of its turning angle in
        # a sliding window. Treats the stroke as cyclic (corners near
        # the start/end of a closed shape would otherwise be missed
        # entirely). NMS suppresses two scores collapsing onto the
        # same physical corner, with cyclic distance so the wrap-around
        # is handled correctly there too.
        n = len(pts)
        if n < k + 4:
            return None
        w = max(3, n // 16)
        scores = self._turn_scores(pts, w)
        min_sep = max(w, n // 8)
        ordered = sorted(range(n), key=lambda i: scores[i], reverse=True)
        picked: list[int] = []
        for idx in ordered:
            if scores[idx] <= 0.0:
                break
            ok = True
            for p in picked:
                d = abs(idx - p)
                d = min(d, n - d)  # cyclic
                if d < min_sep:
                    ok = False
                    break
            if ok:
                picked.append(idx)
            if len(picked) >= k:
                break
        if len(picked) < k:
            return None
        return sorted(picked)

    def _turn_scores(self, pts: list[tuple[float, float]], w: int) -> list[float]:
        # Exterior turn angle at each point, measured between
        # v1 = (P[i] - P[i-w]) and v2 = (P[i+w] - P[i]). A straight
        # stroke segment has v1 ∥ v2 → angle 0 (score 0). A sharp
        # 90° corner has perpendicular v1, v2 → score π/2. A full
        # about-face has v1 anti-parallel to v2 → score π.
        n = len(pts)
        scores = [0.0] * n
        for i in range(n):
            ax, ay = pts[(i - w) % n]
            bx, by = pts[i]
            cx, cy = pts[(i + w) % n]
            v1x, v1y = bx - ax, by - ay
            v2x, v2y = cx - bx, cy - by
            n1 = math.hypot(v1x, v1y)
            n2 = math.hypot(v2x, v2y)
            if n1 <= 0.0 or n2 <= 0.0:
                continue
            dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
            dot = max(-1.0, min(1.0, dot))
            scores[i] = math.acos(dot)
        return scores

    def _min_turn_score(
        self,
        pts: list[tuple[float, float]],
        idxs: list[int],
        span: float,
    ) -> float:
        n = len(pts)
        w = max(3, n // 16)
        scores = self._turn_scores(pts, w)
        return min(scores[i] for i in idxs)

    def _polygon_edge_residual(
        self,
        pts: list[tuple[float, float]],
        polygon: list[tuple[float, float]],
    ) -> float:
        total = 0.0
        m = len(polygon)
        if m < 2:
            return 1e9
        for px, py in pts:
            best = float("inf")
            for j in range(m):
                ax, ay = polygon[j]
                bx, by = polygon[(j + 1) % m]
                d = self._point_to_segment_distance_sq(px, py, ax, ay, bx, by)
                if d < best:
                    best = d
            total += math.sqrt(best)
        return total / len(pts)

    def _order_polygon_ccw(
        self,
        verts: list[tuple[float, float]],
        cx: float,
        cy: float,
    ) -> list[tuple[float, float]]:
        return sorted(verts, key=lambda v: math.atan2(v[1] - cy, v[0] - cx))

    def map_normalized_to_screen(self, x: float, y: float) -> QPointF:
        geo = self.geometry()
        return QPointF(geo.left() + x * geo.width(), geo.top() + y * geo.height())

    def save_canvas_snapshot(self, *, target_dir: Path | None = None, target_path: Path | None = None) -> Optional[Path]:
        self._ensure_canvas_size()
        # Save with a transparent background so the resulting PNG is
        # just the strokes — no solid black/white rectangle around
        # them. The custom-gesture "show_overlay_drawing" action
        # depends on this so a saved drawing can be re-displayed as
        # a click-through overlay on top of any app. Backwards-
        # compatible for users who just want the file: a transparent
        # PNG opens fine in every viewer / editor and shows the
        # stroke colors against whatever the viewer's background is.
        output = QImage(self._canvas.size(), QImage.Format_ARGB32_Premultiplied)
        output.fill(Qt.transparent)
        painter = QPainter(output)
        painter.drawImage(0, 0, self._canvas)
        painter.end()

        path = Path(target_path) if target_path is not None else None
        if path is None:
            base_dir = Path(target_dir) if target_dir is not None else (Path.home() / "Pictures")
            if not base_dir.exists():
                base_dir = Path.home()
            base_dir.mkdir(parents=True, exist_ok=True)
            path = base_dir / f"hgr_drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        saved = output.save(str(path), "PNG")
        return path if saved else None

    def build_canvas_image(self) -> "QImage | None":
        """Return a QImage copy of the current canvas WITHOUT saving
        to disk. Used by the async save path so the UI thread only
        builds the image (fast) and a worker thread does the PNG
        encode. Returns None if the canvas isn't ready."""
        try:
            self._ensure_canvas_size()
        except Exception:
            return None
        try:
            output = QImage(self._canvas.size(), QImage.Format_ARGB32_Premultiplied)
            output.fill(Qt.transparent)
            painter = QPainter(output)
            painter.drawImage(0, 0, self._canvas)
            painter.end()
            return output
        except Exception:
            return None

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._ensure_canvas_size()

    def set_grab_transform(self, dx_norm: float, dy_norm: float, scale: float) -> None:
        """Apply a live translate + scale transform to the displayed
        canvas during a pinch-grab. Translation is in normalised
        screen units (1.0 = full width / height). Scale is a
        multiplier on the canvas's natural size — driven by the
        distance between the two pinching palms when the user is
        bimanual-pinching. Clamped to [0.1, 10.0] so a fast
        accidental two-hand pinch can't shrink the canvas to
        nothing or blow it off the screen. apply_grab_to_canvas()
        bakes both translate AND scale into the canvas pixels at
        grab-end so subsequent strokes / saves reflect the new
        position + size."""
        self._grab_dx_norm = float(dx_norm)
        self._grab_dy_norm = float(dy_norm)
        self._grab_scale = max(0.1, min(10.0, float(scale)))
        self.update()

    def reset_grab_transform(self) -> None:
        self._grab_dx_norm = 0.0
        self._grab_dy_norm = 0.0
        self._grab_scale = 1.0
        self.update()

    def apply_grab_to_canvas(self) -> None:
        """Bake the current live grab transform (translate + scale)
        into the canvas pixels so subsequent strokes draw on top of
        the moved/stretched content and saving captures it. No-op
        when nothing has changed. Pushes a history entry first so
        a left-swipe undo restores the pre-grab canvas (revert any
        movement AND any stretching in one step)."""
        if (
            self._grab_dx_norm == 0.0
            and self._grab_dy_norm == 0.0
            and self._grab_scale == 1.0
        ):
            return
        self._ensure_canvas_size()
        if self._canvas.isNull():
            self._grab_dx_norm = 0.0
            self._grab_dy_norm = 0.0
            self._grab_scale = 1.0
            return
        # History entry: snapshot of the canvas BEFORE the move so
        # an undo restores the pre-grab position + size in one
        # step. Goes through push_undo_state so the snapshot uses
        # the same _clone_canvas / _clone_strokes helpers the rest
        # of the undo machinery does — keeps the entry shape
        # identical to a stroke commit, which means the existing
        # undo_last_action path restores it without any changes.
        try:
            self.push_undo_state()
        except Exception:
            pass
        dx_px = int(self._grab_dx_norm * self.width())
        dy_px = int(self._grab_dy_norm * self.height())
        new_canvas = QImage(self._canvas.size(), QImage.Format_ARGB32_Premultiplied)
        new_canvas.fill(Qt.transparent)
        # Rasterise the source canvas at the new scale, then blit
        # it onto the same-size new_canvas at the translated
        # position. Centering the scaled blit on the canvas
        # midpoint (rather than the top-left) means a pure scale
        # change keeps the strokes anchored where they already
        # were instead of pushing everything down-and-right as it
        # grew — that matched the user's mental model in testing.
        painter = QPainter(new_canvas)
        if self._grab_scale != 1.0:
            scaled_w = max(1, int(self._canvas.width() * self._grab_scale))
            scaled_h = max(1, int(self._canvas.height() * self._grab_scale))
            scaled = self._canvas.scaled(
                scaled_w,
                scaled_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            cx = self._canvas.width() // 2
            cy = self._canvas.height() // 2
            blit_x = cx - scaled.width() // 2 + dx_px
            blit_y = cy - scaled.height() // 2 + dy_px
            painter.drawImage(blit_x, blit_y, scaled)
        else:
            painter.drawImage(dx_px, dy_px, self._canvas)
        painter.end()
        self._canvas = new_canvas
        # Mark stored strokes as out of sync — they still have the
        # pre-translation coordinates because Phase 1 only moves
        # the rasterised pixels. Per-stroke point updates land in
        # Phase 2 along with the sidecar stroke storage. Setting
        # _raster_dirty here means the rasteriser knows it can't
        # rebuild from strokes alone without also re-applying the
        # baked offset.
        self._raster_dirty = True
        self._grab_dx_norm = 0.0
        self._grab_dy_norm = 0.0
        self._grab_scale = 1.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if not self._canvas.isNull():
            # Live grab applies translate + scale on every paint
            # while the user is pinching. Both clear back to
            # identity on apply_grab_to_canvas() (bakes into the
            # canvas pixels) or reset_grab_transform() (cancels
            # without baking). Scale is centred on the canvas
            # midpoint so a pure stretch grows outward in all
            # directions instead of pushing everything down-right.
            tx_px = int(self._grab_dx_norm * self.width())
            ty_px = int(self._grab_dy_norm * self.height())
            if self._grab_scale != 1.0:
                painter.setRenderHint(QPainter.SmoothPixmapTransform)
                scaled_w = max(1, int(self._canvas.width() * self._grab_scale))
                scaled_h = max(1, int(self._canvas.height() * self._grab_scale))
                scaled = self._canvas.scaled(
                    scaled_w,
                    scaled_h,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                cx = self.width() // 2
                cy = self.height() // 2
                bx = cx - scaled.width() // 2 + tx_px
                by = cy - scaled.height() // 2 + ty_px
                painter.drawImage(bx, by, scaled)
            else:
                painter.drawImage(tx_px, ty_px, self._canvas)

        if self._cursor_pos is None or self._cursor_mode == "hidden":
            return

        radius = max(6, int(self.brush_thickness if self._cursor_mode == "draw" else self.eraser_thickness * 0.5))
        outline = QPen(QColor("#FFFFFF"))
        outline.setWidth(2)
        painter.setPen(outline)

        if self._cursor_mode == "draw":
            painter.setBrush(self.brush_color)
            painter.drawEllipse(self._cursor_pos, radius, radius)
        else:
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self._cursor_pos, radius, radius)


class CountdownOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        flags = Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        transparent_flag = getattr(Qt, "WindowTransparentForInput", None)
        if transparent_flag is not None:
            flags |= transparent_flag
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._value = "3"
        self._resize_to_primary_screen()

    def _resize_to_primary_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.setGeometry(0, 0, 1280, 720)
            return
        # macOS: availableGeometry excludes the menu bar / Dock so top-anchored
        # HUD content isn't clipped behind the menu bar (screen.geometry()
        # starts at y=0 under it).
        self.setGeometry(screen.availableGeometry() if sys.platform == "darwin" else screen.geometry())

    def show_countdown(self, value: int | str) -> None:
        self._value = str(value)
        self._resize_to_primary_screen()
        self.show()
        _front_overlay(self)
        self.update()

    def hide_countdown(self) -> None:
        self.hide()

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self._value:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        box_width = 136
        box_height = 94
        rect = QRect(0, 0, box_width, box_height)
        rect.moveCenter(QPoint(self.rect().center().x(), self.rect().bottom() - 82))
        painter.setPen(QPen(QColor(255, 255, 255, 80), 1.4))
        painter.setBrush(QColor(10, 18, 26, 170))
        painter.drawRoundedRect(rect, 18, 18)
        painter.setPen(QColor('#F4FAFF'))
        font = QFont('Arial', 34, QFont.Bold)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, self._value)


class RecordingIndicatorOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        flags = Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        transparent_flag = getattr(Qt, "WindowTransparentForInput", None)
        if transparent_flag is not None:
            flags |= transparent_flag
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._pulse_on = True
        self._timer = QTimer(self)
        self._timer.setInterval(520)
        self._timer.timeout.connect(self._toggle_pulse)
        self._resize_to_primary_screen()

    def _resize_to_primary_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.setGeometry(0, 0, 1280, 720)
            return
        # macOS: availableGeometry so the top-anchored "Recording" pill clears
        # the menu bar (screen.geometry() put its top behind the menu bar and
        # clipped it — user report).
        self.setGeometry(screen.availableGeometry() if sys.platform == "darwin" else screen.geometry())

    def _toggle_pulse(self) -> None:
        self._pulse_on = not self._pulse_on
        if self.isVisible():
            self.update()

    def show_indicator(self) -> None:
        self._resize_to_primary_screen()
        self._pulse_on = True
        self.show()
        _front_overlay(self)
        self._timer.start()
        self.update()

    def hide_indicator(self) -> None:
        self._timer.stop()
        self.hide()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        box_width = 196
        box_height = 56
        rect = QRect(0, 0, box_width, box_height)
        rect.moveCenter(QPoint(self.rect().center().x(), self.rect().top() + 42))
        painter.setPen(QPen(QColor(255, 255, 255, 68), 1.2))
        painter.setBrush(QColor(10, 18, 26, 148))
        painter.drawRoundedRect(rect, 16, 16)
        dot_color = QColor(255, 62, 62, 245 if self._pulse_on else 120)
        painter.setPen(Qt.NoPen)
        painter.setBrush(dot_color)
        painter.drawEllipse(QPoint(rect.left() + 28, rect.center().y()), 8, 8)
        painter.setPen(QColor('#F4FAFF'))
        font = QFont('Arial', 18, QFont.Bold)
        painter.setFont(font)
        painter.drawText(rect.adjusted(44, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, 'Recording')


class ProcessingOverlay(QWidget):
    """Bottom-center "Processing ..." pill with animated dots.

    Visual idiom matches VoiceStatusOverlay (blue translucent
    panel, teal border, light text), so the user reads them as
    the same family. Mouse-transparent.

        overlay = ProcessingOverlay()
        overlay.show_processing("Processing 60s clip")
        # ... do work on a worker thread ...
        overlay.hide_processing()
    """

    # Pill geometry. The overlay WINDOW is sized exactly to the
    # pill (no extra transparent margin around it), otherwise the
    # transparent margin reads on screen as a faint rectangular
    # halo around the pill — what the user reported as a
    # "transparent border." Qt's WA_TranslucentBackground keeps
    # the transparent area invisible *in theory*, but some
    # compositors / GPU drivers leave a 1-pixel residue at the
    # window edge.
    _PILL_WIDTH = 220
    _PILL_HEIGHT = 88
    _SCREEN_BOTTOM_GAP = 64

    def __init__(self, parent=None):
        super().__init__(parent)
        # Window flags mirror VoiceStatusOverlay (which renders
        # correctly on this user's stack). We deliberately do NOT
        # use Qt.WindowTransparentForInput: that flag interacts
        # badly with translucent layered windows on some Win32/GPU
        # combinations and was blocking paint events from landing.
        # WA_TransparentForMouseEvents already covers the
        # "ignore mouse input" goal.
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: none;")
        self._label = "Processing"
        # Optional reference to the voice status overlay. When set,
        # this pill positions itself ABOVE the voice pill (with a
        # small gap) whenever the voice pill is visible and rendering
        # a result / processing / listening state. When the voice
        # pill hides, this pill smoothly slides DOWN to its default
        # bottom-of-screen position. Wired via
        # set_voice_status_anchor() — None until set, in which case
        # the pill always uses its default placement.
        self._voice_status_anchor = None
        self._y_current: float | None = None  # animated Y position
        self._y_target: float | None = None   # goal Y from _compute_target_y
        # Gap between this pill's bottom and the voice pill's top
        # when stacked. Small enough to read them as related, large
        # enough to not look fused.
        self._STACK_GAP = 12
        # Two progress fields: _progress_target is the goal pushed
        # by set_progress() callers at each init checkpoint, and
        # _progress is the currently-rendered fraction. _tick eases
        # _progress toward _progress_target every frame so the bar
        # animates SMOOTHLY between checkpoints instead of jumping
        # in one big step. Also _progress keeps creeping forward at
        # a slow idle rate so the bar never looks stuck even if no
        # new checkpoint arrives for a while.
        self._progress = 0.0
        self._progress_target = 0.0
        # Idle creep: when target hasn't advanced recently, push
        # the target up slowly so the bar visibly moves even
        # between checkpoints. Capped at 0.92 so we never overrun
        # the "real work done" signal.
        self._progress_idle_creep_rate = 0.05  # fraction/sec
        self._progress_idle_creep_cap = 0.92
        self._last_tick_time = time.monotonic()
        # When True, the timer will hide the pill once _progress
        # has eased to ~1.0. Set by complete_and_hide(); lets the
        # bar visibly fill before disappearing.
        self._hide_when_complete = False
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.resize(self._PILL_WIDTH, self._PILL_HEIGHT)

    def set_voice_status_anchor(self, overlay) -> None:
        """Tell this pill about the voice status overlay so it can
        stack ABOVE it when both are visible. Pass None to disable
        stacking. Called by main_window after both overlays exist."""
        self._voice_status_anchor = overlay

    def _voice_pill_top_y(self) -> int | None:
        """Top-y of the voice status pill if it's currently visible
        in a state that should push us up (result / processing /
        listening / warming). Returns None if not stacking."""
        v = self._voice_status_anchor
        if v is None:
            return None
        try:
            if not v.isVisible():
                return None
            mode = getattr(v, "_mode", "")
            # Stack only for modes that occupy bottom-center space.
            # "selection" (large picker) and "hidden" don't trigger.
            if mode not in ("result", "processing", "listening", "warming", "info_hint"):
                return None
            geo = v.geometry()
            return int(geo.top())
        except Exception:
            return None

    def _compute_target_y(self) -> int:
        """Compute the goal Y for this pill. Default: just above
        the screen-bottom gap. Stacked: above the voice pill's top
        with _STACK_GAP between."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return 40
        geo = screen.availableGeometry()
        default_y = geo.bottom() - self._PILL_HEIGHT - self._SCREEN_BOTTOM_GAP
        v_top = self._voice_pill_top_y()
        if v_top is None:
            return int(default_y)
        # Stack above the voice pill. Clamp so we don't go off-screen
        # if the voice pill is unusually tall.
        stacked_y = v_top - self._STACK_GAP - self._PILL_HEIGHT
        return int(max(geo.top() + 8, stacked_y))

    def _place_on_screen(self) -> None:
        # Bottom-center of the primary screen — same pattern the
        # voice status overlay uses, so the user reads them as
        # related. When the voice pill is also visible, stack above
        # it (see _compute_target_y + _tick animation).
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.move(40, 40)
            return
        geo = screen.availableGeometry()
        self.resize(self._PILL_WIDTH, self._PILL_HEIGHT)
        x = geo.center().x() - self._PILL_WIDTH // 2
        target_y = self._compute_target_y()
        # Initialize animation state on first placement so the
        # initial show doesn't slide in from elsewhere.
        if self._y_current is None:
            self._y_current = float(target_y)
        self._y_target = float(target_y)
        self.move(x, int(round(self._y_current)))

    def _tick(self) -> None:
        # Smooth-easing tick: nudge _progress toward _progress_target,
        # and (when no new checkpoint has arrived) creep _progress_target
        # forward slowly so the bar never sits still long enough to
        # look broken.
        if not self.isVisible():
            return
        now = time.monotonic()
        dt = max(0.0, min(0.1, now - self._last_tick_time))
        self._last_tick_time = now
        # Idle creep: lifts the target toward the cap when not at
        # the hide-on-complete phase. Stops once a real checkpoint
        # pushes the target above the cap.
        if not self._hide_when_complete and self._progress_target < self._progress_idle_creep_cap:
            self._progress_target = min(
                self._progress_idle_creep_cap,
                self._progress_target + self._progress_idle_creep_rate * dt,
            )
        # Ease toward target. Cap dt for the easing factor to 40 ms
        # so that when the UI thread comes back from a long block
        # the bar doesn't snap straight to the target in one frame —
        # it catches up over several 16 ms ticks instead, which the
        # eye reads as smooth motion rather than a jump.
        delta = self._progress_target - self._progress
        if abs(delta) > 0.0005:
            dt_capped = min(dt, 0.04)
            factor = 1.0 - math.exp(-10.0 * dt_capped)
            self._progress += delta * factor
            if self._progress > 1.0:
                self._progress = 1.0
        if self._hide_when_complete and self._progress >= 0.998:
            self._progress = 1.0
            self._hide_when_complete = False
            self.repaint()  # final 100 % frame
            self._timer.stop()
            self.hide()
            return
        # Y-position animation: recompute target each tick (so when
        # the voice pill hides we slide back down smoothly), then
        # ease _y_current toward it. Same exponential easing as the
        # progress bar so the motion looks consistent.
        target_y = float(self._compute_target_y())
        if self._y_current is None:
            self._y_current = target_y
        else:
            y_delta = target_y - self._y_current
            if abs(y_delta) > 0.5:
                dt_capped = min(dt, 0.04)
                # Faster easing for position than progress — the eye
                # picks up on bar drag; we want the slide to read as
                # decisive but not jumpy. e^(-12 * 0.016) ≈ 0.82, so
                # 18% of the gap is closed every 16ms.
                factor = 1.0 - math.exp(-12.0 * dt_capped)
                self._y_current += y_delta * factor
                screen = self.screen() or QGuiApplication.primaryScreen()
                if screen is not None:
                    geo = screen.availableGeometry()
                    x = geo.center().x() - self._PILL_WIDTH // 2
                    self.move(int(x), int(round(self._y_current)))
            elif abs(y_delta) > 0.0:
                # Snap the last fractional pixel to avoid jitter.
                self._y_current = target_y
                screen = self.screen() or QGuiApplication.primaryScreen()
                if screen is not None:
                    geo = screen.availableGeometry()
                    x = geo.center().x() - self._PILL_WIDTH // 2
                    self.move(int(x), int(round(self._y_current)))
        self._y_target = target_y
        self.repaint()

    def set_progress(self, fraction: float) -> None:
        """Bump the progress TARGET. The displayed bar eases toward
        it in _tick() so jumps between checkpoints look smooth
        instead of stepped."""
        try:
            new_target = max(0.0, min(1.0, float(fraction)))
        except Exception:
            return
        # Don't shrink the target — protects against out-of-order updates.
        if new_target < self._progress_target:
            return
        self._progress_target = new_target

    def complete_and_hide(self) -> None:
        """Smoothly fill the bar to 100 %, then hide. Use this on
        the success path so the user sees the bar actually finish
        before the pill disappears (otherwise we hide while the bar
        is mid-fill and the eye doesn't register the completion)."""
        self._progress_target = 1.0
        self._hide_when_complete = True
        # Make sure the timer is running so _tick can do the
        # easing. If the pill was hidden somehow without stopping
        # the timer, the isVisible() guard in _tick is a no-op.
        if not self._timer.isActive():
            self._timer.start()

    def show_processing(self, label: str = "Processing") -> None:
        self._label = str(label or "Processing")
        self._progress = 0.0
        self._progress_target = 0.0
        self._hide_when_complete = False
        self._last_tick_time = time.monotonic()
        self._place_on_screen()
        self.show()
        _front_overlay(self)
        # Force a synchronous paint + event flush BEFORE returning.
        # show_processing("Starting Touchless") is called immediately
        # before start_engine blocks the UI thread with worker spin-
        # up; without forcing the paint here, Qt schedules it async,
        # the UI thread is then busy for ~200 ms+, and DWM ends up
        # composting an empty frame (which the user sees as a
        # transparent rectangle with no pill content).
        self.repaint()
        try:
            QApplication.processEvents()
        except Exception:
            pass
        # apply_overlay() strips the DWM rectangle halo around the
        # layered window (DwmSetWindowAttribute disables the system
        # border-color + non-client rendering). Must be called AFTER
        # show() and AFTER the first repaint -- before that the HWND
        # isn't fully realized and DwmSetWindowAttribute returns
        # E_HANDLE silently.
        apply_overlay(self)
        self._timer.start()

    def hide_processing(self) -> None:
        self._timer.stop()
        self.hide()

    def _draw_loading_dots(self, painter: QPainter, cx: float, cy: float, accent: QColor) -> None:
        # Verbatim copy of VoiceStatusOverlay._draw_loading_dots so
        # the indicator reads exactly like the voice "recognising"
        # dots.
        painter.setPen(Qt.NoPen)
        phase = time.monotonic() * 6.0
        for index in range(5):
            wave = max(0.0, math.sin(phase - index * 0.48))
            pulse = 0.38 + 0.62 * wave
            dot = QColor(accent)
            dot.setAlpha(int(90 + 150 * pulse))
            painter.setBrush(dot)
            x = cx + (index - 2) * 14
            y = cy - 2 - 6 * wave
            size = 8.0 + 3.0 * pulse
            painter.drawEllipse(QRectF(x - size / 2.0, y - size / 2.0, size, size))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # Same palette as VoiceStatusOverlay's command panel —
        # translucent blue body, teal border, light foreground.
        panel = QColor(25, 73, 143, 164)
        border = QColor(29, 233, 182, 210)
        text_color = QColor(232, 246, 255, 238)
        accent = QColor(29, 233, 182)

        # Inset the rect by 0.5 px to keep the antialiased border
        # fully inside the window bounds (otherwise the outermost
        # half-pixel of the border would clip).
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(border, 1.2))
        painter.setBrush(panel)
        painter.drawRoundedRect(rect, 18.0, 18.0)

        # Stacked layout: label on top, progress bar below. Bar
        # steps forward as init checkpoints complete (volume API
        # bound, voice listener ready, etc.) -- on a stepped bar
        # the discontinuities read as real progress instead of a
        # broken animation, which is what we get for free even when
        # the UI thread is intermittently blocked during startup.
        font = QFont("Segoe UI", 12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(text_color))
        label_rect = QRectF(rect.left() + 12, rect.top() + 14, rect.width() - 24, 24)
        painter.drawText(label_rect, Qt.AlignCenter, self._label)

        # Progress bar: thin rounded track + accent-coloured fill
        # whose width = progress * track_width.
        bar_h = 6.0
        bar_y = rect.bottom() - 20
        bar_left = rect.left() + 18
        bar_right = rect.right() - 18
        bar_w = bar_right - bar_left
        track = QColor(accent.red(), accent.green(), accent.blue(), 55)
        fill = QColor(accent.red(), accent.green(), accent.blue(), 235)
        painter.setPen(Qt.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(QRectF(bar_left, bar_y, bar_w, bar_h), bar_h / 2.0, bar_h / 2.0)
        fill_w = bar_w * float(self._progress)
        if fill_w > 0.5:
            painter.setBrush(fill)
            painter.drawRoundedRect(QRectF(bar_left, bar_y, fill_w, bar_h), bar_h / 2.0, bar_h / 2.0)


class SavedLocationOverlay(QWidget):
    """Bottom-center pill that briefly shows where a file was just
    saved, then fades away.

    Same blue/teal palette as ProcessingOverlay and
    VoiceStatusOverlay so the user reads them as the same family.
    Width auto-fits the path text, capped at 80 % of screen width;
    paths longer than that are middle-elided so the user still
    sees the leading drive letter and the filename.

    Lifecycle:
        overlay.show_saved("Saved in: C:/.../foo.mp4", fade_after_ms=3000)
    The overlay then animates its windowOpacity from 1.0 → 0.0
    over the last ~600 ms of the visible window and hides itself
    when the animation finishes.
    """

    _PILL_HEIGHT = 56
    _PILL_PADDING_X = 28
    _PILL_PADDING_Y = 14
    _SCREEN_BOTTOM_GAP = 64
    # Vertical offset so this pill stacks ABOVE the standard
    # processing / voice-status pill row instead of overlapping with
    # them. The other two pills sit at y = bottom - height - 64.
    # 110 px = the 88-px ProcessingOverlay + 22-px gap (matches the
    # 56-px voice overlay's similar clearance).
    _STACK_ABOVE_OFFSET = 110
    _MIN_WIDTH = 280
    _MAX_WIDTH_FRAC = 0.80  # of screen width
    _MAX_HEIGHT = 240  # vertical growth cap for very long paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        # Mouse-RECEIVING (was Mouse-Transparent): clicking the pill
        # opens the saved file (or its folder if the file is gone).
        # The cursor flips to PointingHandCursor on enter so the
        # affordance is discoverable.
        self.setCursor(Qt.PointingHandCursor)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: none;")
        self._text = ""
        self._displayed_text = ""
        # Path the click handler will open. None disables the click
        # behaviour (falls back to a regular informational pill).
        self._click_target: Path | None = None
        # Hold-then-fade timers. Hold duration = total_ms - fade_ms.
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._begin_fade)
        # Fade animation tick (16 ms ~ 60 fps).
        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(16)
        self._fade_timer.timeout.connect(self._tick_fade)
        self._fade_total_ms = 600
        self._fade_remaining_ms = 0
        # Slide-down entrance animation. The pill starts 60 px ABOVE
        # its resting y, slides down to rest over _SLIDE_TOTAL_MS with
        # ease-out cubic so it lands smoothly instead of snapping.
        self._slide_total_ms = 280
        self._slide_elapsed_ms = 0
        self._slide_start_y = 0
        self._slide_target_y = 0
        self._slide_x = 0
        self._slide_timer = QTimer(self)
        self._slide_timer.setInterval(16)
        self._slide_timer.timeout.connect(self._tick_slide)
        self.resize(self._MIN_WIDTH, self._PILL_HEIGHT)

    def show_saved(self, text: str, *, total_ms: int = 3000, fade_ms: int = 600, click_target: Path | None = None) -> None:
        """Show the saved-location pill above the standard pill row.

        Animates a slide-down entrance: pill starts ~60 px above its
        resting position and slides down with ease-out cubic over
        ~280 ms so users see it ARRIVE instead of just popping in.
        After total_ms - fade_ms hold, fades out over fade_ms.

        If `click_target` is supplied (recommended for every save
        outcome), the user can click the pill to open the file in
        its native handler. When the file no longer exists at click
        time (deleted between save and click), we fall back to
        opening the containing folder via Explorer. None target =
        informational pill only, click does nothing.
        """
        self._text = str(text or "")
        self._click_target = Path(click_target) if click_target else None
        # Stop any prior cycle so a new save replaces the old pill
        # cleanly.
        self._hold_timer.stop()
        self._fade_timer.stop()
        self._slide_timer.stop()
        self._fade_remaining_ms = 0
        self._slide_elapsed_ms = 0
        self._fit_to_text()
        # _place_on_screen now sets self._slide_target_y / _slide_x
        # but DOES NOT move the widget — _tick_slide handles the
        # frame-by-frame position from start_y to target_y.
        self._place_on_screen()
        # Position the pill at the START of the slide (above rest).
        self._slide_start_y = self._slide_target_y - 60
        self.move(self._slide_x, self._slide_start_y)
        self.setWindowOpacity(1.0)
        self.show()
        _front_overlay(self)
        self.repaint()
        try:
            QApplication.processEvents()
        except Exception:
            pass
        apply_overlay(self)
        self._fade_total_ms = max(50, int(fade_ms))
        hold_ms = max(0, int(total_ms) - self._fade_total_ms)
        self._hold_timer.start(hold_ms)
        # Kick off the slide. Runs in parallel with the hold timer —
        # slide finishes in ~280 ms, hold typically lasts ~2400 ms,
        # so the pill rests for a long beat at the bottom before fading.
        self._slide_elapsed_ms = 0
        self._slide_timer.start()

    def mousePressEvent(self, event):  # noqa: N802
        """Left-click → open the saved file (or its folder if the
        file's been moved / deleted in the interim)."""
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        target = self._click_target
        if target is None:
            event.accept()
            return
        try:
            import os
            if target.exists():
                os.startfile(str(target))
            elif target.parent.exists():
                # File gone — open the folder so the user can see
                # where it WAS / find a renamed version.
                os.startfile(str(target.parent))
        except Exception:
            pass
        # Hide immediately on click so the pill doesn't linger
        # while the OS opens the file.
        self._hold_timer.stop()
        self._fade_timer.stop()
        self._slide_timer.stop()
        self.hide()
        event.accept()

    def _begin_fade(self) -> None:
        self._fade_remaining_ms = self._fade_total_ms
        self._fade_timer.start()

    def _tick_fade(self) -> None:
        self._fade_remaining_ms -= self._fade_timer.interval()
        if self._fade_remaining_ms <= 0:
            self._fade_timer.stop()
            self.hide()
            self.setWindowOpacity(1.0)
            return
        self.setWindowOpacity(max(0.0, self._fade_remaining_ms / float(self._fade_total_ms)))

    def _fit_to_text(self) -> None:
        # Compute pill size from the rendered text. Width grows up
        # to ~80% of the screen; if the text STILL doesn't fit on
        # one line, the pill grows VERTICALLY by word-wrapping
        # instead of middle-eliding so the user can read the whole
        # path / error / message. Height is capped at _MAX_HEIGHT so
        # a runaway string can't render off-screen.
        screen = self.screen() or QGuiApplication.primaryScreen()
        screen_w = screen.availableGeometry().width() if screen is not None else 1280
        max_pill_w = max(self._MIN_WIDTH, int(screen_w * self._MAX_WIDTH_FRAC))
        font = QFont("Segoe UI", 12)
        font.setBold(True)
        metrics = QFontMetrics(font)
        self._displayed_text = self._text
        single_line_w = metrics.horizontalAdvance(self._text) + 2 * self._PILL_PADDING_X
        if single_line_w <= max_pill_w:
            self.resize(max(self._MIN_WIDTH, single_line_w), self._PILL_HEIGHT)
            return
        # Doesn't fit single-line — wrap to multiple lines at max
        # width and grow the pill vertically to match.
        target_text_w = max_pill_w - 2 * self._PILL_PADDING_X
        wrap_rect = metrics.boundingRect(
            0, 0, target_text_w, 10_000,
            int(Qt.TextWordWrap), self._text,
        )
        wrapped_h = wrap_rect.height() + 2 * self._PILL_PADDING_Y
        height = min(self._MAX_HEIGHT, max(self._PILL_HEIGHT, wrapped_h))
        self.resize(max_pill_w, int(height))

    def _place_on_screen(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            # Degenerate path — no screen detected. Store sentinel
            # and rely on caller to skip animation.
            self._slide_x = 40
            self._slide_target_y = 40
            return
        geo = screen.availableGeometry()
        x = geo.center().x() - self.width() // 2
        # Stack ABOVE the standard processing / voice-status pill row.
        # Without the offset, this overlay landed at the same y as
        # ProcessingOverlay's "Processing clip" and VoiceStatusOverlay's
        # "Executing command", producing a visible overlap on every
        # save flow that involved either of those pills.
        y = geo.bottom() - self.height() - self._SCREEN_BOTTOM_GAP - self._STACK_ABOVE_OFFSET
        # Store the resting position; the slide animation will
        # interpolate the y from start_y down to this target.
        self._slide_x = x
        self._slide_target_y = y

    def _tick_slide(self) -> None:
        """Per-frame slide-down position update with ease-out cubic.
        Stops the timer when the pill reaches its resting y."""
        self._slide_elapsed_ms += self._slide_timer.interval()
        if self._slide_elapsed_ms >= self._slide_total_ms:
            # Snap to final position to avoid sub-pixel drift, then stop.
            self.move(self._slide_x, self._slide_target_y)
            self._slide_timer.stop()
            return
        # Ease-out cubic: t in [0, 1], y_offset = 1 - (1-t)^3.
        t = max(0.0, min(1.0, self._slide_elapsed_ms / float(self._slide_total_ms)))
        eased = 1.0 - (1.0 - t) ** 3
        cur_y = int(round(
            self._slide_start_y + (self._slide_target_y - self._slide_start_y) * eased
        ))
        self.move(self._slide_x, cur_y)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        panel = QColor(25, 73, 143, 164)
        border = QColor(29, 233, 182, 210)
        text_color = QColor(232, 246, 255, 238)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(border, 1.2))
        painter.setBrush(panel)
        painter.drawRoundedRect(rect, 18.0, 18.0)

        font = QFont("Segoe UI", 12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(text_color))
        # Word-wrap so multi-line text (paths too long for a single
        # line) renders all lines instead of clipping. AlignCenter
        # both vertically and horizontally keeps short text centered
        # in the larger pill envelope.
        painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, self._displayed_text)


class TrackingQualityPill(QWidget):
    """Bottom-centre desktop pill mirroring the diagnostic
    'Tracking: ...' chip from LiveViewWindow. Visible whenever
    the user has enabled the 'show tracking quality' overlay AND
    the engine is running, so the readout sticks with the user
    even when the live-view window is closed."""

    _PILL_HEIGHT = 32
    _SCREEN_BOTTOM_GAP = 28
    _PADDING_X = 18

    _STATES = {
        "good": ("Tracking: Good",          QColor(29, 233, 182),  QColor(29, 233, 182, 60), QColor(29, 233, 182, 200)),
        "fair": ("Tracking: Marginal",       QColor(245, 180, 80),  QColor(245, 180, 80, 60), QColor(245, 180, 80, 200)),
        "poor": ("Tracking: No hand seen",   QColor(255, 138, 138), QColor(255, 107, 107, 60), QColor(255, 107, 107, 200)),
        "idle": ("Tracking: —",         QColor(229, 246, 255, 200), QColor(20, 30, 50, 160), QColor(127, 127, 127, 120)),
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: none;")
        self._state = "idle"
        self._last_hand_ts = 0.0
        self._fit_size_for_state()

    def _fit_size_for_state(self) -> None:
        text, _fg, _bg, _border = self._STATES.get(self._state, self._STATES["idle"])
        font = QFont("Segoe UI", 10)
        font.setBold(True)
        metrics = QFontMetrics(font)
        w = metrics.horizontalAdvance(text) + 2 * self._PADDING_X
        self.resize(max(170, w), self._PILL_HEIGHT)

    def _place_on_screen(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.center().x() - self.width() // 2
        y = geo.bottom() - self.height() - self._SCREEN_BOTTOM_GAP
        self.move(x, y)

    def update_state(self, *, found: bool, confidence: float) -> None:
        """Feed a per-engine-frame tracking observation. Keys off
        the MediaPipe `found` flag, not the gesture-classifier
        confidence — a neutral hand has confidence≈0 even though
        the hand is fully tracked. State decays to 'poor' if no
        hand has been seen for 0.6 s."""
        del confidence  # accepted for API parity; unused
        now = time.monotonic()
        if found:
            self._last_hand_ts = now
            new_state = "good"
        elif now - self._last_hand_ts >= 0.6:
            new_state = "poor"
        else:
            new_state = self._state
        if new_state != self._state:
            self._state = new_state
            self._fit_size_for_state()
            if self.isVisible():
                self._place_on_screen()
                self.repaint()

    def show_pill(self) -> None:
        self._fit_size_for_state()
        self._place_on_screen()
        self.show()
        _front_overlay(self)
        self.repaint()
        try:
            apply_overlay(self)
        except Exception:
            pass

    def hide_pill(self) -> None:
        self.hide()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt API name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        text, fg, bg, border_color = self._STATES.get(self._state, self._STATES["idle"])
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(border_color, 1.2))
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, rect.height() / 2.0, rect.height() / 2.0)
        font = QFont("Segoe UI", 10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(fg))
        painter.drawText(rect, Qt.AlignCenter, text)


class CaptureRegionOverlay(QWidget):
    selection_finished = Signal(QRect)
    selection_canceled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self._origin_global: QPoint | None = None
        self._current_global: QPoint | None = None
        self._selection_global = QRect()
        self._hand_control = False
        self._cursor_global: QPoint | None = None
        self._last_left_down = False
        self._last_right_down = False

    def _desktop_geometry(self) -> QRect:
        screens = [screen for screen in QGuiApplication.screens() if screen is not None]
        if not screens:
            return QRect(0, 0, 1280, 720)
        geometry = screens[0].geometry()
        for screen in screens[1:]:
            geometry = geometry.united(screen.geometry())
        return geometry

    def begin_selection(self, *, hand_control: bool = False) -> None:
        self._origin_global = None
        self._current_global = None
        self._selection_global = QRect()
        self._cursor_global = None
        self._last_left_down = False
        self._last_right_down = False
        self._hand_control = bool(hand_control)
        self.setGeometry(self._desktop_geometry())
        if self._hand_control:
            self.unsetCursor()
        else:
            self.setCursor(Qt.CrossCursor)
        self.show()
        self.raise_()
        if not self._hand_control:
            self.activateWindow()
        self.update()

    def _local_from_global(self, point: QPoint) -> QPoint:
        origin = self.geometry().topLeft()
        return QPoint(point.x() - origin.x(), point.y() - origin.y())

    def _finish_selection(self) -> None:
        rect = QRect(self._origin_global, self._current_global).normalized() if self._origin_global is not None and self._current_global is not None else QRect()
        self._origin_global = None
        self._current_global = None
        self.hide()
        self.unsetCursor()
        if rect.width() < 8 or rect.height() < 8:
            self.selection_canceled.emit()
            return
        self.selection_finished.emit(rect)

    def _cancel_selection(self) -> None:
        self._origin_global = None
        self._current_global = None
        self._selection_global = QRect()
        self.hide()
        self.unsetCursor()
        self.selection_canceled.emit()

    def update_hand_control(self, global_point: QPoint | None, *, left_down: bool, right_down: bool) -> None:
        if not self._hand_control or not self.isVisible():
            return
        if global_point is not None:
            self._cursor_global = QPoint(global_point)
        if right_down and not self._last_right_down:
            self._last_right_down = True
            self._last_left_down = bool(left_down)
            self._cancel_selection()
            return
        self._last_right_down = bool(right_down)
        if self._cursor_global is None:
            self._last_left_down = bool(left_down)
            self.update()
            return
        if left_down and self._origin_global is None and not self._last_left_down:
            self._origin_global = QPoint(self._cursor_global)
            self._current_global = QPoint(self._cursor_global)
            self._selection_global = QRect(self._origin_global, self._current_global).normalized()
        elif left_down and self._origin_global is not None:
            self._current_global = QPoint(self._cursor_global)
            self._selection_global = QRect(self._origin_global, self._current_global).normalized()
        elif not left_down and self._origin_global is not None and self._last_left_down:
            self._current_global = QPoint(self._cursor_global)
            self._selection_global = QRect(self._origin_global, self._current_global).normalized()
            self._last_left_down = False
            self._finish_selection()
            return
        self._last_left_down = bool(left_down)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._hand_control:
            return
        if event.button() != Qt.LeftButton:
            return
        point = event.globalPosition().toPoint()
        self._origin_global = point
        self._current_global = point
        self._selection_global = QRect(point, point).normalized()
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._hand_control:
            return
        if self._origin_global is None:
            return
        self._current_global = event.globalPosition().toPoint()
        self._selection_global = QRect(self._origin_global, self._current_global).normalized()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._hand_control:
            return
        if event.button() != Qt.LeftButton or self._origin_global is None:
            return
        self._current_global = event.globalPosition().toPoint()
        self._selection_global = QRect(self._origin_global, self._current_global).normalized()
        self._finish_selection()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self._cancel_selection()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(4, 10, 16, 88))
        if not self._selection_global.isNull():
            local_rect = QRect(self._local_from_global(self._selection_global.topLeft()), self._local_from_global(self._selection_global.bottomRight())).normalized()
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(local_rect, Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor('#F4FAFF'), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(local_rect)
        if self._hand_control and self._cursor_global is not None:
            local = self._local_from_global(self._cursor_global)
            cursor_pen = QPen(QColor('#F4FAFF'), 2)
            painter.setPen(cursor_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(local, 10, 10)
            painter.drawLine(local.x() - 15, local.y(), local.x() + 15, local.y())
            painter.drawLine(local.x(), local.y() - 15, local.x(), local.y() + 15)

# Author: Konstantin Markov

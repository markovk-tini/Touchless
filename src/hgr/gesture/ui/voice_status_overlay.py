from __future__ import annotations

import math
import time
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ...app.ui.native_overlay import apply_overlay
from ...config.app_config import AppConfig


class VoiceStatusOverlay(QWidget):
    selectionChosen = Signal(str)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._mode = "hidden"
        self._status_text = ""
        self._command_text = ""
        self._visible_until = 0.0
        self._result_started = 0.0
        self._command_started = 0.0

        self._selection_title = ""
        self._selection_instruction = ""
        self._selection_items: list[tuple[str, str, str]] = []
        self._selection_scroll = 0
        self._selection_hover: int | None = None
        self._selection_path_scroll_ratio = 0.0
        self._selection_drag_mode = "none"
        self._selection_drag_origin = 0.0
        self._selection_drag_start_ratio = 0.0
        self._selection_drag_start_scroll = 0
        self._hint_text = ""
        self._backend_label = ""
        self._info_text = ""
        self._info_hint_hold_seconds = 0.0
        self._info_hint_fade_seconds = 0.65

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
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: none;")
        self._base_size = (356, 142)
        self.setFixedSize(*self._base_size)

    def apply_theme(self, config: AppConfig) -> None:
        self.config = config
        self.update()

    def show_listening(
        self,
        status_text: str = "Listening...",
        *,
        command_text: str = "",
        hint_text: str = "",
        backend_label: str = "",
    ) -> None:
        self._mode = "listening"
        self._status_text = str(status_text or "Listening...")
        self._command_text = str(command_text or "")
        self._hint_text = str(hint_text or "")
        self._backend_label = str(backend_label or "")
        self._info_text = ""
        self._visible_until = 0.0
        self._command_started = time.monotonic() if self._command_text else 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        if self._hint_text:
            self.setFixedSize(380, 156)
        elif self._backend_label:
            self.setFixedSize(148, 138)
        else:
            self.setFixedSize(128, 124)
        self._show_now()

    def set_backend_label(self, backend_label: str) -> None:
        self._backend_label = str(backend_label or "")
        if self._mode == "listening" and not self._hint_text:
            if self._backend_label:
                self.setFixedSize(148, 138)
            else:
                self.setFixedSize(128, 124)
            self._place_on_screen()
        self.update()

    def show_warming(self, status_text: str = "Get ready...") -> None:
        """Mic warmup phase. Shows a STATIC microphone icon (no
        animated sound waves) above the status text — visually
        distinct from the listening mode so the user knows not to
        speak yet. Auto-transitions to listening mode when the
        voice command listener finishes its warmup phase and
        emits status='listening'."""
        self._mode = "warming"
        self._status_text = str(status_text or "Get ready...")
        self._command_text = ""
        self._hint_text = ""
        self._backend_label = ""
        self._info_text = ""
        self._visible_until = 0.0
        self._command_started = 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFixedSize(140, 124)
        self._show_now()

    def show_processing(self, status_text: str, *, command_text: str = "") -> None:
        self._mode = "processing"
        self._status_text = str(status_text or "")
        self._command_text = str(command_text or "")
        self._info_text = ""
        self._visible_until = 0.0
        self._command_started = time.monotonic() if command_text else 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._resize_for_status(self._status_text, self._command_text)
        self._show_now()

    def show_result(self, status_text: str, *, command_text: str = "", duration: float = 1.9) -> None:
        self._mode = "result"
        self._status_text = str(status_text or "")
        self._command_text = str(command_text or "")
        self._info_text = ""
        self._result_started = time.monotonic()
        self._visible_until = self._result_started + max(0.9, float(duration))
        self._command_started = self._result_started if command_text else 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._resize_for_status(self._status_text, self._command_text)
        self._show_now()

    def show_info_hint(self, text: str, *, duration: float = 3.0) -> None:
        self._mode = "info_hint"
        self._status_text = ""
        self._command_text = ""
        self._hint_text = ""
        self._backend_label = ""
        self._info_text = str(text or "").strip()
        self._result_started = time.monotonic()
        self._info_hint_hold_seconds = max(0.8, float(duration))
        self._visible_until = self._result_started + self._info_hint_hold_seconds + self._info_hint_fade_seconds
        self._command_started = 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._resize_for_info_hint(self._info_text)
        self._show_now()

    def show_selection(self, title: str, items: list[tuple[str, str, str]], instruction: str, *, status_text: str = "") -> None:
        self._mode = "selection"
        self._status_text = str(status_text or "")
        self._command_text = ""
        self._selection_title = str(title or "Which file/folder?")
        self._selection_instruction = str(instruction or "Say the corresponding letter.")
        self._selection_items = list(items)
        self._selection_scroll = 0
        self._selection_hover = None
        self._selection_path_scroll_ratio = 0.0
        self._selection_drag_mode = "none"
        self._visible_until = 0.0
        self._command_started = 0.0
        self._info_text = ""
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        height = 336 if len(self._selection_items) > 5 else 308
        self.setFixedSize(760, height)
        self._show_now()

    def update_selection_status(self, status_text: str, *, listening: bool | None = None) -> None:
        if self._mode != "selection":
            return
        self._status_text = str(status_text or self._status_text)
        self.update()

    def hide_overlay(self) -> None:
        self._mode = "hidden"
        self._selection_items = []
        self._selection_hover = None
        self._selection_drag_mode = "none"
        self._info_text = ""
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.hide()

    def tick(self, now: float | None = None) -> None:
        if self._mode == "hidden":
            return
        now = time.monotonic() if now is None else float(now)
        self._place_on_screen()
        if self._visible_until and now >= self._visible_until:
            self.hide_overlay()
            return
        self.update()

    def _show_now(self) -> None:
        import sys
        self._place_on_screen()
        self.show()
        if sys.platform != "darwin":
            # raise_() activates the app on macOS, stealing focus from whatever
            # the command just opened/controlled (e.g. Chrome after "open
            # YouTube"). apply_overlay below orders this panel front
            # non-activating instead.
            self.raise_()
        self.repaint()
        apply_overlay(self)

    # Pill size envelope. Width is bounded so the pill doesn't run
    # screen-edge to screen-edge on wide monitors; height grows
    # without a hard cap so error/heard-text never gets truncated
    # — the user explicitly asked for "stretch vertically up" when
    # text doesn't fit horizontally. _MAX_HEIGHT is large enough to
    # cover a multi-sentence error message but still bounded so a
    # runaway string can't render off-screen.
    _MAX_WIDTH = 540
    _MIN_WIDTH = 160
    _MAX_HEIGHT = 420
    _MIN_HEIGHT_STATUS_ONLY = 60

    def _resize_for_status(self, status_text: str, command_text: str) -> None:
        status = str(status_text or "").strip()
        command = str(command_text or "").strip()
        status_font = QFont("Segoe UI", 12)
        status_font.setBold(True)
        status_metrics = QFontMetrics(status_font)
        command_font = QFont("Segoe UI", 11)
        command_metrics = QFontMetrics(command_font)
        # Layout chrome (must match paint exactly). Paint insets the
        # widget rect by (8, 4, -8, -4), then text rects are inset by
        # another 12 on each side. Net:
        #   H padding (widget edge -> text):  8 + 12 = 20 per side = 40
        #   V padding above dots (widget top -> dots): 4 (rect inset)
        #   V padding below command (widget bot -> text bot): 4 + 12 = 16
        # Tightened from the previous 52/18 to bring the pill snugly
        # around its text — user reported too much empty space.
        H_CHROME = 40
        DOTS_AREA = 36                # was 48
        BOTTOM_MARGIN = 12            # was 18
        TEXT_GAP = 8                  # was 10
        status_width = status_metrics.horizontalAdvance(status or " ")
        width = min(self._MAX_WIDTH, max(self._MIN_WIDTH, status_width + H_CHROME))
        command_height = 0
        if command:
            wrapped = command if "\n" in command else f'"{command}"'
            command_width_single = command_metrics.horizontalAdvance(wrapped)
            width = min(
                self._MAX_WIDTH,
                max(width, command_width_single + H_CHROME),
            )
            available = max(140, width - H_CHROME)
            command_rect = command_metrics.boundingRect(
                0, 0, available, 10_000, int(Qt.TextWordWrap), wrapped,
            )
            command_height = max(18, command_rect.height())
        available_status = max(140, width - H_CHROME)
        status_rect = status_metrics.boundingRect(
            0, 0, available_status, 10_000, int(Qt.TextWordWrap), status or " ",
        )
        status_height = max(20, status_rect.height())
        # Vertical: 4 (rect top inset) + DOTS_AREA + status_height +
        # TEXT_GAP + command_height + BOTTOM_MARGIN + 4 (rect bottom).
        gap = TEXT_GAP if command else 0
        height = 4 + DOTS_AREA + status_height + gap + command_height + BOTTOM_MARGIN + 4
        self.setFixedSize(
            int(width),
            int(min(self._MAX_HEIGHT, max(self._MIN_HEIGHT_STATUS_ONLY, height))),
        )

    def _resize_for_info_hint(self, text: str) -> None:
        content = str(text or "").strip() or "Info"
        font = QFont("Segoe UI", 11)
        font.setBold(True)
        metrics = QFontMetrics(font)
        # Layout chrome must match paint exactly. paintEvent insets
        # widget.rect() by (8, 4, -8, -4), then hint_box is rect
        # adjusted (8, 8, -8, -8), then text is hint_box adjusted
        # (10, 8, -10, -8). Net chrome from widget edge to text:
        #   Horizontal: 8 + 8 + 10 = 26 per side = 52 total
        #   Vertical:   4 + 8 + 8 = 20 per side = 40 total
        # Earlier resize budgeted 32/28 which under-counted by 20/12 —
        # paint wrapped text tighter than expected and clipped
        # "Mouse Mode: ON/OFF" mid-letter. Aligned now.
        H_CHROME = 52
        V_CHROME = 40
        rect = metrics.boundingRect(
            0, 0, self._MAX_WIDTH - H_CHROME, 10_000, int(Qt.TextWordWrap), content,
        )
        width = min(
            self._MAX_WIDTH,
            max(self._MIN_WIDTH, rect.width() + H_CHROME),
        )
        # Re-measure at the chosen width so the height reflects the
        # final wrap shape.
        rect = metrics.boundingRect(
            0, 0, width - H_CHROME, 10_000, int(Qt.TextWordWrap), content,
        )
        height = min(self._MAX_HEIGHT, max(60, rect.height() + V_CHROME))
        self.setFixedSize(int(width), int(height))

    def _place_on_screen(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.move(40, 40)
            return
        geo = screen.availableGeometry()
        x = geo.center().x() - self.width() // 2
        y = geo.bottom() - self.height() - 18
        self.move(x, y)

    def _accent(self) -> QColor:
        color = QColor(str(self.config.accent_color or "#1DE9B6"))
        if not color.isValid():
            color = QColor("#1DE9B6")
        return color

    def _selection_phase(self) -> str:
        lowered = self._status_text.lower()
        if "listen" in lowered:
            return "listening"
        if lowered:
            return "recognizing"
        return "idle"

    def _selection_layout(self) -> tuple[QRectF, QRectF, float, float, int, int]:
        rect = QRectF(self.rect()).adjusted(8, 4, -8, -4)
        indicator_gap = 54.0
        box_rect = QRectF(rect.left(), rect.top() + indicator_gap, rect.width(), rect.height() - indicator_gap - 6)
        list_top = box_rect.top() + 44.0
        list_bottom = box_rect.bottom() - 48.0
        row_h = 30
        visible_rows = max(1, int((list_bottom - list_top) // row_h))
        return rect, box_rect, list_top, list_bottom, row_h, visible_rows

    def _selection_visible_rows(self) -> list[tuple[int, tuple[str, str, str]]]:
        _outer, _box, _top, _bottom, _row_h, visible_rows = self._selection_layout()
        max_scroll = max(0, len(self._selection_items) - visible_rows)
        self._selection_scroll = max(0, min(self._selection_scroll, max_scroll))
        return list(enumerate(self._selection_items[self._selection_scroll:self._selection_scroll + visible_rows], start=self._selection_scroll))

    def _selection_max_path_overflow(self, path_width: float) -> float:
        if path_width <= 0:
            return 0.0
        font = QFont("Segoe UI", 10)
        metrics = QFontMetrics(font)
        max_width = 0.0
        for _number, _label, path_text in self._selection_items:
            max_width = max(max_width, float(metrics.horizontalAdvance(str(path_text or ""))))
        return max(0.0, max_width - path_width + 12.0)

    def _selection_horizontal_scrollbar_rect(self) -> QRectF:
        _outer, box_rect, _top, _bottom, _row_h, _visible_rows = self._selection_layout()
        return QRectF(box_rect.left() + 18, box_rect.bottom() - 22, box_rect.width() - 50, 10)

    def _selection_vertical_scrollbar_rect(self) -> QRectF:
        _outer, box_rect, list_top, list_bottom, _row_h, _visible_rows = self._selection_layout()
        return QRectF(box_rect.right() - 14, list_top + 2, 8, max(40.0, list_bottom - list_top - 4))

    def _selection_horizontal_thumb_rect(self) -> QRectF:
        track = self._selection_horizontal_scrollbar_rect()
        path_width = max(10.0, track.width() - 4)
        overflow = self._selection_max_path_overflow(path_width)
        if overflow <= 1e-3:
            return QRectF(track.left() + 1, track.top() + 1, track.width() - 2, track.height() - 2)
        thumb_w = max(64.0, track.width() * 0.28)
        max_x = max(0.0, track.width() - thumb_w)
        x = track.left() + max_x * max(0.0, min(1.0, self._selection_path_scroll_ratio))
        return QRectF(x, track.top() + 1, thumb_w, track.height() - 2)

    def _selection_vertical_thumb_rect(self) -> QRectF:
        track = self._selection_vertical_scrollbar_rect()
        _outer, _box, _top, _bottom, _row_h, visible_rows = self._selection_layout()
        total = max(1, len(self._selection_items))
        if total <= visible_rows:
            return QRectF(track.left() + 1, track.top() + 1, track.width() - 2, track.height() - 2)
        ratio = visible_rows / total
        thumb_h = max(26.0, track.height() * ratio)
        max_scroll = max(1, total - visible_rows)
        progress = self._selection_scroll / max_scroll
        y = track.top() + (track.height() - thumb_h) * progress
        return QRectF(track.left() + 1, y, track.width() - 2, thumb_h)

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._mode == "hidden":
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        rect = QRectF(self.rect()).adjusted(8, 4, -8, -4)
        now = time.monotonic()
        accent = self._accent()
        muted = QColor(232, 246, 255, 238)
        soft = QColor(214, 235, 255, 214)
        border = QColor(29, 233, 182, 210)
        panel = QColor(25, 73, 143, 164)

        if self._mode == "selection":
            _outer_rect, box_rect, list_top, list_bottom, row_h, visible_rows = self._selection_layout()
            phase = self._selection_phase()
            if phase == "listening":
                mic_cx = box_rect.center().x()
                mic_cy = box_rect.top() - 22.0
                painter.setPen(Qt.NoPen)
                self._draw_microphone(painter, mic_cx, mic_cy, accent)
                self._draw_sound_waves(painter, mic_cx, mic_cy, accent)
            elif phase == "recognizing":
                cx = box_rect.center().x()
                cy = box_rect.top() - 28.0
                painter.setPen(Qt.NoPen)
                self._draw_loading_dots(painter, cx, cy, accent)
                font = QFont("Segoe UI", 10)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QPen(muted))
                painter.drawText(QRectF(box_rect.left() + 16, box_rect.top() - 14, box_rect.width() - 32, 14), Qt.AlignCenter, self._status_text)

            painter.setPen(QPen(border, 1.2))
            painter.setBrush(panel)
            painter.drawRoundedRect(box_rect, 18.0, 18.0)

            title_font = QFont("Segoe UI", 12)
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.setPen(QPen(muted))
            painter.drawText(QRectF(box_rect.left() + 18, box_rect.top() + 14, box_rect.width() - 36, 18), Qt.AlignCenter, self._selection_title)

            row_font = QFont("Segoe UI", 10)
            painter.setFont(row_font)
            v_track = self._selection_vertical_scrollbar_rect()
            reserved_right = 18.0 if len(self._selection_items) > visible_rows else 0.0
            path_width = max(80.0, box_rect.width() - 28 - reserved_right)
            for idx, (selection_key, label, path_text) in self._selection_visible_rows():
                row_index = idx - self._selection_scroll
                row_rect = QRectF(box_rect.left() + 14, list_top + row_index * row_h, box_rect.width() - 28 - reserved_right, row_h - 2)
                hover = self._selection_hover == idx
                if hover:
                    painter.setPen(QPen(QColor(255, 255, 255, 22), 1.0))
                    painter.setBrush(QColor(255, 255, 255, 20))
                    painter.drawRoundedRect(row_rect, 8.0, 8.0)
                number_rect = QRectF(row_rect.left() + 8, row_rect.top(), 34, row_rect.height())
                label_rect = QRectF(row_rect.left() + 38, row_rect.top(), row_rect.width() * 0.33, row_rect.height())
                path_rect = QRectF(row_rect.left() + row_rect.width() * 0.38, row_rect.top(), row_rect.width() * 0.58, row_rect.height())
                painter.setPen(QPen(QColor(245, 250, 255, 235)))
                painter.drawText(number_rect, Qt.AlignVCenter | Qt.AlignLeft, f"{selection_key}.")
                painter.setPen(QPen(QColor(236, 246, 255, 230)))
                painter.drawText(label_rect, Qt.AlignVCenter | Qt.AlignLeft, label)
                painter.setPen(QPen(QColor(204, 226, 244, 195)))
                self._draw_path_text(painter, path_rect, path_text)

            instruction_font = QFont("Segoe UI", 10)
            instruction_font.setBold(True)
            painter.setFont(instruction_font)
            painter.setPen(QPen(QColor(230, 241, 250, 220)))
            instruction = self._selection_instruction or "Say the corresponding letter."
            painter.drawText(QRectF(box_rect.left() + 18, box_rect.bottom() - 42.0, box_rect.width() - 36, 14), Qt.AlignCenter, instruction)

            h_track = self._selection_horizontal_scrollbar_rect()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 22))
            painter.drawRoundedRect(h_track, 5.0, 5.0)
            painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 210))
            painter.drawRoundedRect(self._selection_horizontal_thumb_rect(), 5.0, 5.0)

            if len(self._selection_items) > visible_rows:
                painter.setBrush(QColor(255, 255, 255, 22))
                painter.drawRoundedRect(v_track, 4.0, 4.0)
                painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 210))
                painter.drawRoundedRect(self._selection_vertical_thumb_rect(), 4.0, 4.0)
            return

        rect = rect
        if self._mode == "info_hint":
            info_alpha = self._info_hint_alpha(now)
            if info_alpha <= 0:
                return
            hint_box = rect.adjusted(8, 8, -8, -8)
            hint_fill = QColor(29, 100, 233, int(92 * info_alpha / 255))
            hint_border = QColor(29, 150, 233, int(178 * info_alpha / 255))
            painter.setPen(QPen(hint_border, 1.2))
            painter.setBrush(hint_fill)
            painter.drawRoundedRect(hint_box, 12.0, 12.0)
            hint_font = QFont("Segoe UI", 11)
            hint_font.setBold(True)
            painter.setFont(hint_font)
            painter.setPen(QPen(QColor(240, 250, 255, int(248 * info_alpha / 255))))
            painter.drawText(hint_box.adjusted(10, 8, -10, -8), Qt.AlignCenter | Qt.TextWordWrap, self._info_text)
            return
        if self._mode == "warming":
            # Mic warmup phase. Same layout as the no-hint listening
            # pill (mic icon + text pill below) but WITHOUT the
            # animated sound waves — visually distinct so the user
            # knows not to speak yet.
            center_x = rect.center().x()
            icon_center_y = rect.center().y() - 14
            painter.setPen(Qt.NoPen)
            self._draw_microphone(painter, center_x, icon_center_y, accent)
            # NO _draw_sound_waves — that's the visible "listening" cue.
            warming_font = QFont("Segoe UI", 9)
            warming_font.setBold(True)
            painter.setFont(warming_font)
            warming_metrics = QFontMetrics(warming_font)
            warming_text = self._status_text or "Get ready..."
            warming_text_w = warming_metrics.horizontalAdvance(warming_text)
            warming_w = min(rect.width() - 12, warming_text_w + 22)
            warming_h = 20.0
            warming_left = center_x - warming_w / 2.0
            warming_top = icon_center_y + 38
            warming_rect = QRectF(warming_left, warming_top, warming_w, warming_h)
            # Amber-ish tint instead of the blue listening tint so the
            # user can tell the two modes apart at a glance.
            warming_fill = QColor(233, 145, 29, 92)
            warming_border = QColor(233, 168, 29, 178)
            painter.setPen(QPen(warming_border, 1.2))
            painter.setBrush(warming_fill)
            painter.drawRoundedRect(warming_rect, 9.0, 9.0)
            painter.setPen(QPen(QColor(255, 248, 240, 248)))
            painter.drawText(warming_rect, Qt.AlignCenter, warming_text)
            return
        if self._mode == "listening":
            center_x = rect.center().x()
            if self._hint_text:
                icon_center_y = rect.top() + 36
                painter.setPen(Qt.NoPen)
                self._draw_microphone(painter, center_x, icon_center_y, accent)
                self._draw_sound_waves(painter, center_x, icon_center_y, accent)
                hint_box = QRectF(rect.left() + 8, icon_center_y + 44, rect.width() - 16, 60)
                hint_fill = QColor(29, 100, 233, 80)
                hint_border = QColor(29, 150, 233, 160)
                painter.setPen(QPen(hint_border, 1.2))
                painter.setBrush(hint_fill)
                painter.drawRoundedRect(hint_box, 10.0, 10.0)
                hint_font = QFont("Segoe UI", 11)
                hint_font.setBold(True)
                painter.setFont(hint_font)
                painter.setPen(QPen(QColor(240, 250, 255, 248)))
                painter.drawText(hint_box.adjusted(8, 6, -8, -6), Qt.AlignCenter | Qt.TextWordWrap, self._hint_text)
            else:
                if self._backend_label:
                    icon_center_y = rect.center().y() - 22
                else:
                    icon_center_y = rect.center().y() - 14
                painter.setPen(Qt.NoPen)
                self._draw_microphone(painter, center_x, icon_center_y, accent)
                self._draw_sound_waves(painter, center_x, icon_center_y, accent)
                listening_font = QFont("Segoe UI", 9)
                listening_font.setBold(True)
                painter.setFont(listening_font)
                listening_metrics = QFontMetrics(listening_font)
                listening_text = "Listening"
                listening_text_w = listening_metrics.horizontalAdvance(listening_text)
                listening_w = min(rect.width() - 12, listening_text_w + 22)
                listening_h = 20.0
                listening_left = center_x - listening_w / 2.0
                listening_top = icon_center_y + 38
                listening_rect = QRectF(listening_left, listening_top, listening_w, listening_h)
                listening_fill = QColor(29, 100, 233, 92)
                listening_border = QColor(29, 150, 233, 178)
                painter.setPen(QPen(listening_border, 1.2))
                painter.setBrush(listening_fill)
                painter.drawRoundedRect(listening_rect, 9.0, 9.0)
                painter.setPen(QPen(QColor(240, 250, 255, 248)))
                painter.drawText(listening_rect, Qt.AlignCenter, listening_text)
                if self._backend_label:
                    badge_font = QFont("Segoe UI", 9)
                    badge_font.setBold(True)
                    painter.setFont(badge_font)
                    metrics = QFontMetrics(badge_font)
                    text = self._backend_label
                    text_width = metrics.horizontalAdvance(text)
                    badge_w = min(rect.width() - 12, text_width + 18)
                    badge_h = 18.0
                    badge_left = center_x - badge_w / 2.0
                    badge_top = icon_center_y + 62
                    badge_rect = QRectF(badge_left, badge_top, badge_w, badge_h)
                    badge_fill = QColor(29, 100, 233, 120)
                    badge_border = QColor(accent.red(), accent.green(), accent.blue(), 200)
                    painter.setPen(QPen(badge_border, 1.0))
                    painter.setBrush(badge_fill)
                    painter.drawRoundedRect(badge_rect, 8.0, 8.0)
                    painter.setPen(QPen(QColor(240, 250, 255, 248)))
                    painter.drawText(badge_rect, Qt.AlignCenter, text)
            return

        painter.setPen(QPen(border, 1.2))
        painter.setBrush(panel)
        painter.drawRoundedRect(rect, 18.0, 18.0)

        center_x = rect.center().x()
        # Tightened layout (must match _resize_for_status constants):
        #   DOTS_AREA = 36 (dots vertical center at rect.top()+18)
        #   Text rects use 12 px horizontal inset (was 18)
        #   BOTTOM_MARGIN = 12 (was 18)
        icon_center_y = rect.top() + 18
        painter.setPen(Qt.NoPen)
        self._draw_loading_dots(painter, center_x, icon_center_y, accent)

        status_font = QFont("Segoe UI", 12)
        status_font.setBold(True)
        painter.setFont(status_font)
        painter.setPen(QPen(muted))
        status_metrics = QFontMetrics(status_font)
        status_available = rect.width() - 24
        status_natural = status_metrics.boundingRect(
            0, 0, int(status_available), 10_000,
            int(Qt.TextWordWrap), self._status_text or " ",
        )
        status_height = max(20, status_natural.height())
        # When there is NO command_text, vertically center the status
        # text in the space below the dots.
        dot_area_bottom = rect.top() + 24
        if not self._command_text:
            available_top = dot_area_bottom + 4
            available_bottom = rect.bottom() - 8
            status_y = available_top + max(
                0.0,
                (available_bottom - available_top - status_height) / 2.0,
            )
        else:
            status_y = rect.top() + 36
        status_rect = QRectF(rect.left() + 12, status_y, status_available, status_height)
        painter.drawText(
            status_rect,
            Qt.AlignCenter | Qt.TextWordWrap,
            self._status_text,
        )

        if self._command_text:
            command_alpha = self._command_alpha(now)
            quote_font = QFont("Segoe UI", 11)
            painter.setFont(quote_font)
            quote_pen = QPen(soft)
            quote_pen.setColor(QColor(soft.red(), soft.green(), soft.blue(), command_alpha))
            painter.setPen(quote_pen)
            # 8-px gap between status and command (matches resize TEXT_GAP).
            command_top = status_rect.bottom() + 8
            command_rect = QRectF(
                rect.left() + 12,
                command_top,
                rect.width() - 24,
                rect.bottom() - command_top - 12,
            )
            painter.drawText(
                command_rect,
                Qt.AlignCenter | Qt.TextWordWrap,
                self._command_text if "\n" in self._command_text else f'"{self._command_text}"',
            )

    def _draw_path_text(self, painter: QPainter, rect: QRectF, text: str) -> None:
        content = str(text or "").strip()
        if not content:
            return
        metrics = QFontMetrics(painter.font())
        text_width = metrics.horizontalAdvance(content)
        baseline = rect.top() + (rect.height() + metrics.ascent() - metrics.descent()) / 2.0
        if text_width <= rect.width() - 2.0:
            painter.drawText(QRectF(rect), Qt.AlignVCenter | Qt.AlignLeft, content)
            return
        overflow = max(0.0, text_width - rect.width() + 12.0)
        offset = overflow * max(0.0, min(1.0, self._selection_path_scroll_ratio))
        painter.save()
        painter.setClipRect(rect)
        painter.drawText(QPointF(rect.left() - offset, baseline), content)
        painter.restore()

    def _command_alpha(self, now: float) -> int:
        if self._mode == "selection":
            return 255
        if not self._command_text:
            return 0
        if self._mode == "result" and self._status_text.lower().startswith("execut"):
            elapsed = max(0.0, now - self._result_started)
            if elapsed >= 0.55:
                return 0
            return int(255 * max(0.0, 1.0 - elapsed / 0.55))
        if self._command_started and now - self._command_started > 3.0:
            fade = (now - self._command_started - 3.0) / 0.55
            return int(255 * max(0.0, 1.0 - fade))
        if self._mode == "result" and self._visible_until > 0.0:
            remaining = max(0.0, self._visible_until - now)
            if remaining < 0.6:
                return int(255 * (remaining / 0.6))
        return 255

    def _info_hint_alpha(self, now: float) -> int:
        if self._mode != "info_hint":
            return 255
        if self._result_started <= 0.0:
            return 255
        fade_seconds = max(0.2, float(self._info_hint_fade_seconds or 0.65))
        hold_seconds = max(0.0, float(self._info_hint_hold_seconds or 0.0))
        elapsed = max(0.0, float(now) - float(self._result_started))
        if elapsed <= hold_seconds:
            return 255
        fade = (elapsed - hold_seconds) / fade_seconds
        return int(255 * max(0.0, 1.0 - fade))

    def _draw_microphone(self, painter: QPainter, cx: float, cy: float, accent: QColor) -> None:
        stroke = QColor(255, 255, 255, 242)
        fill = QColor(255, 255, 255, 18)
        head_rect = QRectF(cx - 10.0, cy - 20.0, 20.0, 28.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(head_rect, 10.0, 10.0)
        pen = QPen(stroke, 4.0)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(head_rect, 10.0, 10.0)
        cradle = QPainterPath()
        cradle.moveTo(cx - 15.0, cy - 3.0)
        cradle.lineTo(cx - 15.0, cy + 4.0)
        cradle.quadTo(cx - 15.0, cy + 17.0, cx, cy + 17.0)
        cradle.quadTo(cx + 15.0, cy + 17.0, cx + 15.0, cy + 4.0)
        cradle.lineTo(cx + 15.0, cy - 3.0)
        painter.drawPath(cradle)
        painter.drawLine(QPointF(cx, cy + 17.0), QPointF(cx, cy + 31.0))
        painter.drawLine(QPointF(cx - 12.0, cy + 31.0), QPointF(cx + 12.0, cy + 31.0))

    def _draw_sound_waves(self, painter: QPainter, cx: float, cy: float, accent: QColor) -> None:
        phase = int((time.monotonic() * 4.2) % 4)
        active_index = phase if phase < 3 else None
        base = QColor(accent.red(), accent.green(), accent.blue(), 84)
        wave_specs = ((21.0, 28.0), (28.0, 38.0), (36.0, 50.0))
        arc_center_y = cy - 6.0
        for index, (radius_x, radius_y) in enumerate(wave_specs):
            active = active_index == index
            color = QColor(accent.red(), accent.green(), accent.blue(), 248 if active else base.alpha())
            pen = QPen(color, 4.2 if active else 2.4)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            bounds = QRectF(cx - radius_x, arc_center_y - radius_y, radius_x * 2.0, radius_y * 2.0)
            left_path = QPainterPath()
            left_path.arcMoveTo(bounds, 132.0)
            left_path.arcTo(bounds, 132.0, 96.0)
            painter.drawPath(left_path)
            right_path = QPainterPath()
            right_path.arcMoveTo(bounds, -48.0)
            right_path.arcTo(bounds, -48.0, 96.0)
            painter.drawPath(right_path)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._mode != "selection":
            return super().mousePressEvent(event)
        pos = event.position()
        h_thumb = self._selection_horizontal_thumb_rect()
        h_track = self._selection_horizontal_scrollbar_rect()
        v_thumb = self._selection_vertical_thumb_rect()
        v_track = self._selection_vertical_scrollbar_rect()
        if h_thumb.contains(pos):
            self._selection_drag_mode = "horizontal"
            self._selection_drag_origin = pos.x()
            self._selection_drag_start_ratio = self._selection_path_scroll_ratio
            return
        if h_track.contains(pos):
            self._selection_drag_mode = "horizontal"
            self._selection_drag_origin = pos.x()
            self._selection_drag_start_ratio = self._selection_path_scroll_ratio
            return
        if v_thumb.contains(pos) or v_track.contains(pos):
            self._selection_drag_mode = "vertical"
            self._selection_drag_origin = pos.y()
            self._selection_drag_start_scroll = self._selection_scroll
            return
        index = self._selection_index_at(pos.x(), pos.y())
        if index is not None and 0 <= index < len(self._selection_items):
            self.selectionChosen.emit(str(self._selection_items[index][0]))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._mode != "selection":
            return super().mouseMoveEvent(event)
        pos = event.position()
        if self._selection_drag_mode == "horizontal":
            track = self._selection_horizontal_scrollbar_rect()
            thumb = self._selection_horizontal_thumb_rect()
            travel = max(1.0, track.width() - thumb.width())
            delta = pos.x() - self._selection_drag_origin
            self._selection_path_scroll_ratio = max(0.0, min(1.0, self._selection_drag_start_ratio + delta / travel))
            self.update()
            return
        if self._selection_drag_mode == "vertical":
            track = self._selection_vertical_scrollbar_rect()
            thumb = self._selection_vertical_thumb_rect()
            total = max(1, len(self._selection_items))
            _outer, _box, _top, _bottom, _row_h, visible_rows = self._selection_layout()
            max_scroll = max(0, total - visible_rows)
            travel = max(1.0, track.height() - thumb.height())
            delta = pos.y() - self._selection_drag_origin
            progress = max(0.0, min(1.0, (self._selection_drag_start_scroll / max(1, max_scroll)) + delta / travel))
            self._selection_scroll = int(round(progress * max_scroll))
            self.update()
            return
        index = self._selection_index_at(pos.x(), pos.y())
        if index != self._selection_hover:
            self._selection_hover = index
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._mode == "selection":
            self._selection_drag_mode = "none"
            return
        return super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._mode != "selection":
            return super().wheelEvent(event)
        if not self._selection_items:
            return
        delta = event.angleDelta().y()
        step = -1 if delta > 0 else 1
        _outer, _box, _top, _bottom, _row_h, visible_rows = self._selection_layout()
        max_scroll = max(0, len(self._selection_items) - visible_rows)
        self._selection_scroll = max(0, min(max_scroll, self._selection_scroll + step))
        self.update()

    def _selection_index_at(self, x: float, y: float) -> int | None:
        if self._mode != "selection" or not self._selection_items:
            return None
        _outer_rect, box_rect, list_top, list_bottom, row_h, _visible_rows = self._selection_layout()
        if x < box_rect.left() + 12 or x > box_rect.right() - 20:
            return None
        if y < list_top or y > list_bottom:
            return None
        relative = int((y - list_top) // row_h)
        index = self._selection_scroll + relative
        if index >= len(self._selection_items):
            return None
        return index

    def _draw_loading_dots(self, painter: QPainter, cx: float, cy: float, accent: QColor) -> None:
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

# Author: Konstantin Markov

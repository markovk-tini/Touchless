from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..app.ui.native_overlay import apply_overlay
from ..config.app_config import AppConfig


class ScreenVolumeOverlay(QWidget):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._controller = None
        self._current_fraction = 0.0
        self._muted = False
        self._active = False
        self._message = "Idle"
        self._title = "Vol"
        self._dual_mode = False
        self._app_fraction = 0.0
        self._app_label = ""
        self._bar_selected = "sys"
        self._accent = QColor("#1DE9B6")
        self._surface = QColor(15, 23, 42, 235)
        self._text = QColor(227, 237, 246, 240)
        self._border = QColor(29, 233, 182, 66)
        self._track = QColor(255, 255, 255, 22)
        self._track_border = QColor(255, 255, 255, 34)
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
        self.setFixedSize(100, 320)
        self.apply_theme(config)

    def attach_controller(self, controller) -> None:
        self._controller = controller

    def apply_theme(self, config: AppConfig) -> None:
        self.config = config
        accent = QColor(str(config.accent_color or "#1DE9B6"))
        text = QColor(str(config.text_color or "#E3EDF6"))
        surface = QColor(str(config.surface_color or "#0F172A"))
        if not accent.isValid():
            accent = QColor("#1DE9B6")
        if not text.isValid():
            text = QColor("#E3EDF6")
        if not surface.isValid():
            surface = QColor("#0F172A")
        self._accent = accent
        self._text = text
        self._surface = QColor(surface.red(), surface.green(), surface.blue(), 235)
        self._border = QColor(accent.red(), accent.green(), accent.blue(), 66)
        self._track = QColor(255, 255, 255, 22)
        self._track_border = QColor(255, 255, 255, 34)
        self.update()

    def show_overlay(self) -> None:
        self._place_on_screen()
        self.show()
        self.raise_()
        self.repaint()
        apply_overlay(self)

    def hide_overlay(self) -> None:
        self.hide()

    def set_level(self, level: float | None, *, muted: bool, active: bool, message: str) -> None:
        self._muted = bool(muted)
        self._active = bool(active)
        self._message = str(message)
        self._dual_mode = False
        self._set_fraction_from_level(level)
        if self.width() != 100:
            self.setFixedSize(100, 320)
        self.update()

    def set_dual_level(
        self,
        app_level: float | None,
        sys_level: float | None,
        app_label: str,
        *,
        muted: bool,
        active: bool,
        selected_bar: str,
        message: str,
    ) -> None:
        self._muted = bool(muted)
        self._active = bool(active)
        self._message = str(message)
        self._dual_mode = True
        self._app_label = str(app_label)
        self._bar_selected = str(selected_bar)
        self._app_fraction = max(0.0, min(1.0, float(app_level))) if app_level is not None else 0.0
        self._set_fraction_from_level(sys_level)
        if self.width() != 150:
            self.setFixedSize(150, 350)
        self.update()

    def sync_visual_state(self) -> None:
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.sync_visual_state()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        card_rect = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        painter.setPen(QPen(self._border, 1.2))
        painter.setBrush(self._surface)
        painter.drawRoundedRect(card_rect, 18, 18)

        if self._dual_mode:
            self._paint_dual(painter, card_rect)
        else:
            self._paint_single(painter, card_rect)

    def _paint_single(self, painter: QPainter, card_rect: QRectF) -> None:
        title_font = QFont("Segoe UI", 15)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QPen(self._accent))
        painter.drawText(
            QRectF(card_rect.left() + 14, card_rect.top() + 10, card_rect.width() - 28, 24),
            Qt.AlignCenter,
            self._title,
        )

        bar_rect = QRectF(card_rect.center().x() - 16, card_rect.top() + 42, 32, 182)
        painter.setPen(QPen(self._track_border, 1.2))
        painter.setBrush(self._track)
        painter.drawRoundedRect(bar_rect, 14, 14)

        if self._current_fraction > 0.005:
            inner = bar_rect.adjusted(4, 4, -4, -4)
            fill_height = inner.height() * self._current_fraction
            fill_rect = QRectF(inner.left(), inner.bottom() - fill_height, inner.width(), fill_height)
            fill_radius = min(fill_rect.width() / 2.0, fill_height / 2.0, 10.0)
            glow = QColor(self._accent.red(), self._accent.green(), self._accent.blue(), 84)
            painter.setPen(Qt.NoPen)
            painter.setBrush(glow)
            painter.drawRoundedRect(fill_rect.adjusted(-2, -2, 2, 2), fill_radius + 2.0, fill_radius + 2.0)
            painter.setBrush(self._accent)
            painter.drawRoundedRect(fill_rect, fill_radius, fill_radius)

        level_font = QFont("Segoe UI", 16)
        level_font.setBold(True)
        painter.setFont(level_font)
        painter.setPen(QPen(self._text))
        percent = int(round(self._current_fraction * 100))
        percent_text = "--" if percent < 0 else f"{percent}%"
        painter.drawText(
            QRectF(card_rect.left() + 10, bar_rect.bottom() + 14, card_rect.width() - 20, 32),
            Qt.AlignCenter,
            percent_text,
        )

        # State badge: only rendered when muted. "Adjusting" / "Idle"
        # were both removed — they were visual noise.
        if self._muted:
            status_font = QFont("Segoe UI", 11)
            status_font.setBold(True)
            painter.setFont(status_font)
            painter.drawText(
                QRectF(card_rect.left() + 16, bar_rect.bottom() + 48, card_rect.width() - 32, 18),
                Qt.AlignCenter,
                "Muted",
            )

    def _paint_dual(self, painter: QPainter, card_rect: QRectF) -> None:
        bar_w = 36.0
        bar_h = 168.0
        label_h = 34.0
        label_top = card_rect.top() + 12.0
        bar_top = label_top + label_h + 4.0
        pct_h = 28.0
        pct_top = bar_top + bar_h + 6.0
        state_top = pct_top + pct_h + 4.0

        cx = card_rect.center().x()
        gap = 28.0
        left_bar_x = cx - gap / 2.0 - bar_w
        right_bar_x = cx + gap / 2.0

        app_selected = self._bar_selected == "app"
        sys_selected = not app_selected
        dim_alpha = 80

        label_font = QFont("Segoe UI", 10)
        label_font.setBold(True)
        painter.setFont(label_font)

        app_accent = self._accent if app_selected else QColor(self._accent.red(), self._accent.green(), self._accent.blue(), dim_alpha)
        sys_accent = self._accent if sys_selected else QColor(self._accent.red(), self._accent.green(), self._accent.blue(), dim_alpha)

        painter.setPen(QPen(app_accent))
        painter.drawText(
            QRectF(left_bar_x - 10, label_top, bar_w + 20, label_h),
            Qt.AlignCenter | Qt.TextWordWrap,
            f"{self._app_label}\nVol",
        )
        painter.setPen(QPen(sys_accent))
        painter.drawText(
            QRectF(right_bar_x - 10, label_top, bar_w + 20, label_h),
            Qt.AlignCenter | Qt.TextWordWrap,
            "Sys\nVol",
        )

        for bar_x, fraction, accent, selected in (
            (left_bar_x, self._app_fraction, app_accent, app_selected),
            (right_bar_x, self._current_fraction, sys_accent, sys_selected),
        ):
            bar_rect = QRectF(bar_x, bar_top, bar_w, bar_h)
            track_color = self._track if selected else QColor(self._track.red(), self._track.green(), self._track.blue(), dim_alpha)
            border_color = self._track_border if selected else QColor(self._track_border.red(), self._track_border.green(), self._track_border.blue(), dim_alpha)
            painter.setPen(QPen(border_color, 1.2))
            painter.setBrush(track_color)
            painter.drawRoundedRect(bar_rect, 13, 13)
            if fraction > 0.005:
                inner = bar_rect.adjusted(4, 4, -4, -4)
                fill_height = inner.height() * fraction
                fill_rect = QRectF(inner.left(), inner.bottom() - fill_height, inner.width(), fill_height)
                fill_radius = min(fill_rect.width() / 2.0, fill_height / 2.0, 10.0)
                glow = QColor(accent.red(), accent.green(), accent.blue(), 84 if selected else 30)
                painter.setPen(Qt.NoPen)
                painter.setBrush(glow)
                painter.drawRoundedRect(fill_rect.adjusted(-2, -2, 2, 2), fill_radius + 2.0, fill_radius + 2.0)
                painter.setBrush(accent)
                painter.drawRoundedRect(fill_rect, fill_radius, fill_radius)

        pct_font = QFont("Segoe UI", 11)
        pct_font.setBold(True)
        painter.setFont(pct_font)
        for bar_x, fraction, accent in (
            (left_bar_x, self._app_fraction, app_accent),
            (right_bar_x, self._current_fraction, sys_accent),
        ):
            painter.setPen(QPen(accent))
            pct = int(round(fraction * 100))
            painter.drawText(
                QRectF(bar_x - 10, pct_top, bar_w + 20, pct_h),
                Qt.AlignCenter,
                f"{pct}%",
            )

        if self._muted:
            status_font = QFont("Segoe UI", 10)
            status_font.setBold(True)
            painter.setFont(status_font)
            painter.setPen(QPen(self._text))
            painter.drawText(
                QRectF(card_rect.left() + 16, state_top, card_rect.width() - 32, 20),
                Qt.AlignCenter,
                "Muted",
            )

    def _place_on_screen(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.move(20, 20)
            return
        geo = screen.availableGeometry()
        x = geo.right() - self.width() - 18
        y = geo.center().y() - self.height() // 2
        y = max(geo.top() + 18, y)
        self.move(x, y)

    def _set_fraction_from_level(self, level: float | None) -> None:
        if level is None:
            self._current_fraction = 0.0
            return
        clamped = max(0.0, min(1.0, float(level)))
        self._current_fraction = clamped

# Author: Konstantin Markov

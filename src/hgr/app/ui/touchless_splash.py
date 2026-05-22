from __future__ import annotations

import platform

from PySide6.QtCore import (
    QEasingCurve,
    Qt,
    QTimer,
    QVariantAnimation,
)
from PySide6.QtGui import (
    QBitmap,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QGuiApplication,
    QPainter,
)
from PySide6.QtWidgets import QWidget


class TouchlessSplash(QWidget):
    _WORD = "Touchless"
    _LETTER_STAGGER_MS = 140
    _LETTER_DURATION_MS = 560
    _WAVE_OFFSET_PX = 24
    _SIDE_PADDING = 16
    _VERTICAL_PADDING = 12

    def __init__(self, accent_color: str, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.FramelessWindowHint
            | Qt.Tool
            | Qt.WindowStaysOnTopHint
            | Qt.NoDropShadowWindowHint,
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAutoFillBackground(False)

        self._accent_color = QColor(accent_color)
        self._font = self._pick_display_font()
        self._metrics = QFontMetrics(self._font)

        self._letter_positions: list[int] = []
        cursor_x = self._SIDE_PADDING
        extra_spacing = int(round(self._metrics.averageCharWidth() * 0.08))
        for char in self._WORD:
            self._letter_positions.append(cursor_x)
            cursor_x += self._metrics.horizontalAdvance(char) + extra_spacing
        total_width = cursor_x + self._SIDE_PADDING
        total_height = (
            self._metrics.height()
            + self._WAVE_OFFSET_PX
            + self._VERTICAL_PADDING * 2
        )

        self._letter_opacities: list[float] = [0.0] * len(self._WORD)
        self._letter_offsets: list[float] = [float(self._WAVE_OFFSET_PX)] * len(self._WORD)

        self._finished = False
        self._animations: list[QVariantAnimation] = []
        self._dwm_tweaked = False

        self.resize(total_width, total_height)
        self._center_on_screen()
        self._apply_text_shape_mask()

    @staticmethod
    def _pick_display_font() -> QFont:
        preferred = (
            "Corbel",
            "Candara",
            "Calibri",
            "Segoe UI Variable Display",
            "Segoe UI",
        )
        available = set(QFontDatabase.families())
        family = next((name for name in preferred if name in available), "Segoe UI")
        font = QFont(family)
        font.setPointSize(96)
        font.setWeight(QFont.DemiBold)
        font.setItalic(True)
        font.setLetterSpacing(QFont.PercentageSpacing, 108)
        font.setStyleStrategy(QFont.PreferAntialias | QFont.PreferQuality)
        return font

    def _center_on_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geom = screen.availableGeometry()
        self.move(
            geom.center().x() - self.width() // 2,
            geom.center().y() - self.height() // 2,
        )

    def _apply_text_shape_mask(self) -> None:
        # Build a bitmap mask from the letter glyphs so the window region only
        # contains the text shape. Anything outside the glyphs (including any
        # stray compositor fill that would read as a border) is clipped out.
        bitmap = QBitmap(self.size())
        bitmap.clear()
        painter = QPainter(bitmap)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setFont(self._font)
        painter.setPen(Qt.color1)
        baseline_y = (
            self.height() + self._metrics.ascent() - self._metrics.descent()
        ) // 2
        # Draw each letter at every possible wave offset so the mask covers
        # the full vertical travel of the animation. Slight horizontal
        # dilation keeps anti-aliased edges inside the mask.
        travel_steps = 6
        for index, char in enumerate(self._WORD):
            x = self._letter_positions[index]
            for step in range(travel_steps + 1):
                frac = step / travel_steps
                offset_y = int(round(self._WAVE_OFFSET_PX * (1.0 - frac)))
                for dx in (-2, -1, 0, 1, 2):
                    painter.drawText(x + dx, baseline_y + offset_y, char)
        painter.end()
        self.setMask(bitmap)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._dwm_tweaked and platform.system() == "Windows":
            self._dwm_tweaked = True
            self._apply_windows_dwm_tweaks()

    def _apply_windows_dwm_tweaks(self) -> None:
        # Strip out the Windows 11 rounded-corner preference and disable DWM
        # non-client rendering for this HWND. Either of those can otherwise
        # produce a faint 1-px frame around a translucent tool window.
        try:
            import ctypes

            hwnd = int(self.winId())
            dwmapi = ctypes.windll.dwmapi

            DWMWA_NCRENDERING_POLICY = 2
            DWMNCRP_DISABLED = 1
            policy = ctypes.c_int(DWMNCRP_DISABLED)
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_NCRENDERING_POLICY,
                ctypes.byref(policy),
                ctypes.sizeof(policy),
            )

            DWMWA_WINDOW_CORNER_PREFERENCE = 33
            DWMWCP_DONOTROUND = 1
            corner = ctypes.c_int(DWMWCP_DONOTROUND)
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_WINDOW_CORNER_PREFERENCE,
                ctypes.byref(corner),
                ctypes.sizeof(corner),
            )
        except Exception:
            pass

    def start_animation(self) -> None:
        for index in range(len(self._WORD)):
            delay = index * self._LETTER_STAGGER_MS

            opacity_anim = QVariantAnimation(self)
            opacity_anim.setDuration(self._LETTER_DURATION_MS)
            opacity_anim.setStartValue(0.0)
            opacity_anim.setEndValue(1.0)
            opacity_anim.setEasingCurve(QEasingCurve.OutCubic)
            opacity_anim.valueChanged.connect(
                lambda value, i=index: self._apply_opacity(i, value)
            )

            offset_anim = QVariantAnimation(self)
            offset_anim.setDuration(self._LETTER_DURATION_MS)
            offset_anim.setStartValue(float(self._WAVE_OFFSET_PX))
            offset_anim.setEndValue(0.0)
            offset_anim.setEasingCurve(QEasingCurve.OutBack)
            offset_anim.valueChanged.connect(
                lambda value, i=index: self._apply_offset(i, value)
            )

            QTimer.singleShot(delay, opacity_anim.start)
            QTimer.singleShot(delay, offset_anim.start)
            self._animations.extend([opacity_anim, offset_anim])

        QTimer.singleShot(self.total_animation_ms(), self._on_finished)

    def _apply_opacity(self, index: int, value) -> None:
        self._letter_opacities[index] = float(value)
        self.update()

    def _apply_offset(self, index: int, value) -> None:
        self._letter_offsets[index] = float(value)
        self.update()

    def _on_finished(self) -> None:
        self._finished = True

    def is_finished(self) -> bool:
        return self._finished

    def total_animation_ms(self) -> int:
        return (len(self._WORD) - 1) * self._LETTER_STAGGER_MS + self._LETTER_DURATION_MS

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setFont(self._font)

        baseline_y = (
            self.height() + self._metrics.ascent() - self._metrics.descent()
        ) // 2

        for index, char in enumerate(self._WORD):
            opacity = max(0.0, min(1.0, self._letter_opacities[index]))
            if opacity <= 0.001:
                continue
            offset_y = int(round(self._letter_offsets[index]))
            color = QColor(self._accent_color)
            color.setAlphaF(opacity)
            painter.setPen(color)
            painter.drawText(
                self._letter_positions[index],
                baseline_y + offset_y,
                char,
            )

        painter.end()

    @staticmethod
    def run_with(callback_build_window, accent_color: str, app) -> QWidget:
        """Show splash, play the reveal, then build + map the main window
        fully transparent so its first paint cycle is invisible, close the
        splash, and reveal the main window at full opacity. Mapping it
        transparent (rather than the old off-screen WA_DontShowOnScreen
        toggle) avoids the hollow framed-white flash that toggle leaked on
        Windows."""
        splash = TouchlessSplash(accent_color)
        # Kill the startup flash: a Qt.Tool translucent window's FIRST
        # mapped frame briefly appears at the default top-left corner —
        # a small blank box — before the move-to-center, the glyph mask,
        # and the DWM non-client tweaks take effect. That corner flash
        # was the "little window that pops up right before the logo".
        #
        # Two-part guard:
        #   1. Park the window far off-screen for the initial map so the
        #      raw first frame is painted where nobody can see it.
        #      (setWindowOpacity(0) BEFORE show() doesn't help — there's
        #      no native HWND yet, so the layered-alpha call is a no-op
        #      and the first frame still shows at full opacity.)
        #   2. Hold opacity at 0 too, then after the window is mapped +
        #      masked (showEvent has fired, DWM tweaks applied), move it
        #      to center and reveal. Moving an already-mapped window
        #      doesn't flash. The per-letter content starts at 0 opacity,
        #      so revealing shows nothing until the animation begins.
        splash.move(-10000, -10000)
        splash.setWindowOpacity(0.0)
        splash.show()
        splash.raise_()
        for _ in range(4):
            app.processEvents()
        splash._center_on_screen()
        splash.setWindowOpacity(1.0)

        splash.start_animation()
        while not splash.is_finished():
            app.processEvents()

        # Build the main window, then map it FULLY TRANSPARENT for its
        # first paint so the user never sees the hollow framed white box
        # that flashes on a frameless window's first show.
        #
        # The previous approach toggled WA_DontShowOnScreen around an
        # off-screen show()/hide(). On Windows that leaks: the platform
        # plugin briefly maps a small default-framed window before the
        # frameless hint + real geometry apply — that was the little
        # white "Touchless" window popping up at launch.
        #
        # Instead: poke winId() to force native-handle creation, set
        # opacity to 0 on that real HWND (setWindowOpacity is a no-op
        # before the handle exists), then show(). The window maps
        # invisibly, paints its real content during the processEvents
        # below, and only after the splash closes do we reveal it at
        # full opacity. Position is left untouched, so saved/centered
        # geometry is preserved.
        window = callback_build_window()
        window.setWindowOpacity(0.0)
        window.winId()  # force native window creation so opacity applies
        window.setWindowOpacity(0.0)
        window.show()
        for _ in range(6):
            app.processEvents()

        splash.close()
        for _ in range(2):
            app.processEvents()

        window.setWindowOpacity(1.0)
        window.raise_()
        return window

# Author: Konstantin Markov

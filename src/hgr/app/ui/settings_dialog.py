from __future__ import annotations

import ctypes
import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...config.app_config import AppConfig


def _is_elevated() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


class ColorButton(QPushButton):
    color_changed = Signal(str)

    def __init__(self, label: str, color: str):
        super().__init__(label)
        self._color = color
        self.clicked.connect(self._pick_color)
        self._refresh_style()

    @property
    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color
        self._refresh_style()

    def _pick_color(self) -> None:
        color = QColorDialog.getColor()
        if color.isValid():
            self._color = color.name()
            self._refresh_style()
            self.color_changed.emit(self._color)

    def _refresh_style(self) -> None:
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self._color};
                color: white;
                border: 1px solid rgba(255,255,255,0.25);
                border-radius: 12px;
                padding: 10px 12px;
                font-weight: 700;
            }}
            """
        )


class SettingsDialog(QDialog):
    settings_applied = Signal(object)

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        # r51: install_indigo_chrome (Win10+Win11) replaces the old
        # apply_touchless_chrome (Win11 only). _body is the QWidget
        # under the indigo bar — _build_ui parents its layout to it.
        from .window_chrome import install_indigo_chrome
        self.setWindowTitle("Settings")
        self.setModal(False)
        self.setMinimumWidth(420)
        self._body = install_indigo_chrome(self, "Settings")
        self.config = AppConfig(**config.__dict__)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self._body)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        title = QLabel("Settings")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: #E5F6FF;")
        root.addWidget(title)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)

        self.primary_button = ColorButton("Primary", self.config.primary_color)
        self.primary_button.color_changed.connect(lambda c: setattr(self.config, "primary_color", c))
        form.addRow("Main color", self.primary_button)

        self.accent_button = ColorButton("Accent", self.config.accent_color)
        self.accent_button.color_changed.connect(lambda c: setattr(self.config, "accent_color", c))
        form.addRow("Accent color", self.accent_button)

        self.surface_button = ColorButton("Surface", self.config.surface_color)
        self.surface_button.color_changed.connect(lambda c: setattr(self.config, "surface_color", c))
        form.addRow("Surface color", self.surface_button)

        self.text_button = ColorButton("Text", self.config.text_color)
        self.text_button.color_changed.connect(lambda c: setattr(self.config, "text_color", c))
        form.addRow("Text color", self.text_button)

        # Admin row: opt-in elevation. Touchless runs unprivileged by
        # default; some features (clip recording of higher-IL games,
        # gesture control of Task Manager / regedit / anti-cheat games)
        # need a fresh elevated instance to bypass UIPI.
        admin_row = QWidget()
        admin_layout = QHBoxLayout(admin_row)
        admin_layout.setContentsMargins(0, 0, 0, 0)
        admin_layout.setSpacing(8)
        if _is_elevated():
            admin_state = QLabel("Running as administrator")
            admin_state.setStyleSheet("color: #1DE9B6; font-weight: 700;")
            admin_layout.addWidget(admin_state)
            admin_layout.addStretch(1)
        else:
            elevate_button = QPushButton("Restart as administrator")
            elevate_button.setToolTip("Needed for elevated-app clipping (UAC prompt).")
            elevate_button.clicked.connect(self._restart_as_admin)
            admin_layout.addWidget(elevate_button)
            admin_layout.addStretch(1)
        form.addRow("Admin privileges", admin_row)

        # YouTube section: auto-pause on absence, auto-skip ads,
        # caption-translate target language. Drives engine-side
        # behavior (see noop_engine absence-tracking + the auto-skip
        # background timer) so these toggles take effect on the next
        # frame after Apply is clicked.
        section_youtube = QLabel("YouTube")
        section_youtube.setStyleSheet(
            "font-size: 13px; font-weight: 800; letter-spacing: 0.05em; "
            "color: #58E3FF; padding-top: 6px;"
        )
        form.addRow(section_youtube)

        self.pause_on_absence_check = QCheckBox(
            "Pause YouTube when I walk away"
        )
        self.pause_on_absence_check.setChecked(
            bool(getattr(self.config, "youtube_pause_when_user_leaves", False))
        )
        self.pause_on_absence_check.stateChanged.connect(
            lambda s: setattr(
                self.config, "youtube_pause_when_user_leaves", bool(s)
            )
        )
        form.addRow("Auto-pause", self.pause_on_absence_check)

        self.pause_absence_seconds = QSpinBox()
        self.pause_absence_seconds.setRange(2, 60)
        self.pause_absence_seconds.setSuffix(" s")
        self.pause_absence_seconds.setValue(
            int(getattr(self.config, "youtube_pause_when_user_leaves_seconds", 6))
        )
        self.pause_absence_seconds.valueChanged.connect(
            lambda v: setattr(
                self.config, "youtube_pause_when_user_leaves_seconds", int(v)
            )
        )
        form.addRow("Absence threshold", self.pause_absence_seconds)

        self.auto_skip_ads_check = QCheckBox(
            "Automatically skip ads when they appear"
        )
        self.auto_skip_ads_check.setChecked(
            bool(getattr(self.config, "youtube_auto_skip_ads", False))
        )
        self.auto_skip_ads_check.stateChanged.connect(
            lambda s: setattr(self.config, "youtube_auto_skip_ads", bool(s))
        )
        form.addRow("Skip ads", self.auto_skip_ads_check)

        self.caption_lang_combo = QComboBox()
        self.caption_lang_combo.addItem("Off (don't translate)", "")
        for _lang in (
            "English", "Spanish", "French", "German", "Italian",
            "Portuguese", "Japanese", "Korean", "Chinese", "Hindi",
            "Russian", "Arabic", "Dutch", "Polish", "Turkish",
        ):
            self.caption_lang_combo.addItem(_lang, _lang)
        _current_lang = str(
            getattr(self.config, "youtube_caption_target_language", "") or ""
        )
        _idx = self.caption_lang_combo.findData(_current_lang)
        if _idx >= 0:
            self.caption_lang_combo.setCurrentIndex(_idx)
        self.caption_lang_combo.currentIndexChanged.connect(
            lambda _i: setattr(
                self.config,
                "youtube_caption_target_language",
                str(self.caption_lang_combo.currentData() or ""),
            )
        )
        form.addRow("Translate captions to", self.caption_lang_combo)

        self.font_slider = QSlider(Qt.Horizontal)
        self.font_slider.setMinimum(42)
        self.font_slider.setMaximum(140)
        self.font_slider.setValue(self.config.hello_font_size)
        self.font_value = QLabel(str(self.config.hello_font_size))
        self.font_value.setStyleSheet("font-weight: 700; color: #E5F6FF;")
        self.font_slider.valueChanged.connect(self._font_size_changed)
        font_row = QWidget()
        font_layout = QHBoxLayout(font_row)
        font_layout.setContentsMargins(0, 0, 0, 0)
        font_layout.addWidget(self.font_slider)
        font_layout.addWidget(self.font_value)
        form.addRow("HELLO size", font_row)

        root.addWidget(form_widget)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._apply)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_row.addWidget(apply_button)
        button_row.addWidget(close_button)
        root.addLayout(button_row)

        self.setStyleSheet(
            """
            QDialog {
                background-color: #0F172A;
                color: #E5F6FF;
                border: 1px solid rgba(29, 233, 182, 0.35);
            }
            QLabel {
                color: #E5F6FF;
                font-size: 14px;
            }
            QPushButton {
                background-color: #0B3D91;
                color: #E5F6FF;
                border: 1px solid rgba(29,233,182,0.35);
                border-radius: 12px;
                padding: 10px 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                border: 1px solid #1DE9B6;
            }
            """
        )

    def _font_size_changed(self, value: int) -> None:
        self.config.hello_font_size = value
        self.font_value.setText(str(value))

    def _restart_as_admin(self) -> None:
        # Source-run has no exe to relaunch; bail silently so devs aren't
        # surprised by the UAC prompt firing on a python.exe in the venv.
        if not getattr(sys, "frozen", False):
            return
        ret = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, "", None, 1
        )
        # ret <= 32 means ShellExecuteW failed — most often the user
        # declined the UAC prompt. No-op in that case; they're already
        # aware they cancelled.
        if ret <= 32:
            return
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _apply(self) -> None:
        self.settings_applied.emit(self.config)

# Author: Konstantin Markov

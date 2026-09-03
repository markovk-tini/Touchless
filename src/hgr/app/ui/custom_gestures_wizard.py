"""Survey-style wizard for creating a new custom gesture.

Collects: name, description, hold/cooldown timing, action kind + value.
On Start, launches the recorder dialog (caller wires that in). Conflict
detection on name happens here; pose-similarity conflicts are checked
later by the recorder once samples exist.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QPainter, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from hgr.custom_gestures.registry import Action, GestureRegistry

from .custom_gestures_chrome import apply_touchless_titlebar
from .window_chrome import touchless_message_box


# Cached PNG paths for spinbox / combo arrows. Qt stylesheet border
# triangles often paint as solid rectangles on Win11 + PySide6; real
# PNGs (same pattern as the app checkmark) render reliably.
_SPIN_ARROW_PNG_CACHE: dict[str, str] = {}


def _spin_arrow_image_path(direction: str) -> str:
    """Write a small white triangle PNG and return a QSS-safe path.

    direction: "up" | "down" | "combo" (chevron for QComboBox).
    """
    key = str(direction or "").lower()
    cached = _SPIN_ARROW_PNG_CACHE.get(key)
    if cached:
        try:
            if Path(cached).exists():
                return cached
        except Exception:
            pass
    try:
        from PySide6.QtGui import QPixmap

        pix = QPixmap(16, 16)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#FFFFFF"))
        if key == "up":
            poly = QPolygonF(
                [QPointF(8, 3), QPointF(13, 12), QPointF(3, 12)]
            )
        elif key == "combo":
            poly = QPolygonF(
                [QPointF(3, 6), QPointF(13, 6), QPointF(8, 12)]
            )
        else:  # down
            poly = QPolygonF(
                [QPointF(3, 4), QPointF(13, 4), QPointF(8, 13)]
            )
        p.drawPolygon(poly)
        p.end()
        tmp = Path(tempfile.gettempdir()) / f"touchless_spin_arrow_{key}.png"
        pix.save(str(tmp), "PNG")
        path = str(tmp).replace("\\", "/")
        _SPIN_ARROW_PNG_CACHE[key] = path
        return path
    except Exception:
        return ""


# (label, kind, value-prompt, placeholder)
_ACTION_KINDS = (
    ("Press a single key", "keystroke", "Key name", "e.g. enter, f12, space"),
    ("Press a hotkey combo", "hotkey", "Keys (joined by +)", "e.g. ctrl+shift+t"),
    ("Type a text snippet", "text", "Text to type", "e.g. test@example.com"),
    ("Open a URL in browser", "open_url", "URL", "e.g. https://example.com"),
    ("Run a shell command", "run_command", "Command", "e.g. start spotify"),
    ("Open any file", "open_file", "Full file path", r"e.g. C:\Users\you\Documents\notes.docx"),
    ("Show a saved drawing as overlay", "show_overlay_drawing", "Drawing filename (in your drawings folder)", "e.g. Touchless_Drawing_001.png"),
)


@dataclass(frozen=True)
class WizardResult:
    name: str
    description: str
    hold_seconds: float
    cooldown_seconds: float
    action: Action
    # Dynamic-only: one of "until_stopped", "fixed_1s", "fixed_2s",
    # "fixed_3s". Ignored for static gestures; left empty for them.
    duration_mode: str = ""


# Mock keyboard layout. (display_label, action_key_name, width_units).
# 1.0 width-unit ≈ 28 px. Width-units roughly match a real US QWERTY
# keyboard so the visual reads as familiar. Right-side modifier
# duplicates (R-Shift / R-Ctrl etc.) collapse to the same key name on
# selection — clicking 'Shift' twice on either side toggles the same
# entry, which matches what the underlying SendInput layer does.
_VK_LAYOUT_ROWS: tuple[tuple[tuple[str, str, float], ...], ...] = (
    (
        ("Esc", "esc", 1.5),
        ("F1", "f1", 1.0), ("F2", "f2", 1.0), ("F3", "f3", 1.0), ("F4", "f4", 1.0),
        ("F5", "f5", 1.0), ("F6", "f6", 1.0), ("F7", "f7", 1.0), ("F8", "f8", 1.0),
        ("F9", "f9", 1.0), ("F10", "f10", 1.0), ("F11", "f11", 1.0), ("F12", "f12", 1.0),
    ),
    (
        ("`", "`", 1.0),
        ("1", "1", 1.0), ("2", "2", 1.0), ("3", "3", 1.0), ("4", "4", 1.0),
        ("5", "5", 1.0), ("6", "6", 1.0), ("7", "7", 1.0), ("8", "8", 1.0),
        ("9", "9", 1.0), ("0", "0", 1.0),
        ("-", "-", 1.0), ("=", "=", 1.0),
        ("Backspace", "backspace", 2.0),
    ),
    (
        ("Tab", "tab", 1.5),
        ("Q", "q", 1.0), ("W", "w", 1.0), ("E", "e", 1.0), ("R", "r", 1.0), ("T", "t", 1.0),
        ("Y", "y", 1.0), ("U", "u", 1.0), ("I", "i", 1.0), ("O", "o", 1.0), ("P", "p", 1.0),
        ("[", "[", 1.0), ("]", "]", 1.0), ("\\", "\\", 1.5),
    ),
    (
        ("Caps", "caps", 1.75),
        ("A", "a", 1.0), ("S", "s", 1.0), ("D", "d", 1.0), ("F", "f", 1.0), ("G", "g", 1.0),
        ("H", "h", 1.0), ("J", "j", 1.0), ("K", "k", 1.0), ("L", "l", 1.0),
        (";", ";", 1.0), ("'", "'", 1.0),
        ("Enter", "enter", 2.25),
    ),
    (
        ("Shift", "shift", 2.25),
        ("Z", "z", 1.0), ("X", "x", 1.0), ("C", "c", 1.0), ("V", "v", 1.0), ("B", "b", 1.0),
        ("N", "n", 1.0), ("M", "m", 1.0),
        (",", ",", 1.0), (".", ".", 1.0), ("/", "/", 1.0),
        ("Shift", "shift", 2.75),
    ),
    (
        ("Ctrl", "ctrl", 1.5),
        ("Win", "win", 1.25),
        ("Alt", "alt", 1.25),
        ("Space", "space", 6.25),
        ("Alt", "alt", 1.25),
        ("Win", "win", 1.25),
        ("Ctrl", "ctrl", 1.5),
    ),
)

_VK_UNIT_WIDTH = 22
_VK_KEY_HEIGHT = 22
_VK_KEY_GAP = 2


class _VirtualKeyboard(QWidget):
    """Compact clickable keyboard for the gesture wizard. Two modes:
    'single' (only one key may be selected) and 'combo' (multi-key
    chord). Emits keys_changed with the formatted string ('a',
    'enter', 'ctrl+shift+t', etc.) so the wizard's QLineEdit can
    follow along.

    Future: detect the user's actual keyboard layout via Win32
    GetKeyboardLayoutName and swap rows. For now we ship US QWERTY,
    which covers the vast majority of bindable shortcuts the user is
    likely to want."""

    keys_changed = Signal(str)

    def __init__(self, accent_color: str, parent=None):
        super().__init__(parent)
        self._accent_color = accent_color
        self._mode = "single"
        self._selected: list[str] = []
        # Each action_key_name may be on multiple buttons (left + right
        # Shift / Ctrl / Win / Alt). Keep them all so we can highlight
        # both sides when one is clicked.
        self._buttons_by_key: dict[str, list[QPushButton]] = {}
        self._build()

    def set_mode(self, mode: str) -> None:
        """'single' = one key only (replaces on click); 'combo' = chord
        (toggles each key on click). Clears the current selection on
        mode change so a stale combo doesn't leak into a freshly-
        selected single-key action."""
        if mode == self._mode:
            return
        self._mode = mode
        self._selected = []
        self._refresh_visual()
        self.keys_changed.emit("")

    def set_value(self, value: str) -> None:
        """Sync from an external string so manual typing in the wizard's
        QLineEdit reflects on the keyboard's highlighted keys."""
        text = (value or "").strip().lower()
        if not text:
            self._selected = []
        elif self._mode == "single":
            self._selected = [text]
        else:
            self._selected = [p.strip() for p in text.split("+") if p.strip()]
        self._refresh_visual()

    def selection(self) -> str:
        return self._format()

    # --- internal -----------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(_VK_KEY_GAP)
        # Stretches on BOTH sides of each row so the row's content
        # sits centred horizontally inside the keyboard widget,
        # which itself can grow to fill whatever width the parent
        # layout gave it. The previous version only had a trailing
        # stretch and left-aligned the keys.
        for row_cells in _VK_LAYOUT_ROWS:
            row = QHBoxLayout()
            row.setSpacing(_VK_KEY_GAP)
            row.setContentsMargins(0, 0, 0, 0)
            row.addStretch(1)
            for label, key, width_units in row_cells:
                btn = self._make_key(label, key, width_units)
                row.addWidget(btn)
            row.addStretch(1)
            outer.addLayout(row)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def _make_key(self, label: str, key: str, width_units: float) -> QPushButton:
        btn = QPushButton(label)
        btn.setObjectName("vkKey")
        btn.setFocusPolicy(Qt.NoFocus)
        btn.setAutoDefault(False)
        btn.setDefault(False)
        # Stash the target pixel size on the button so
        # _apply_button_style can bake it into the stylesheet.
        # setFixedSize alone wasn't enforcing the size — the parent
        # wizard's QPushButton rule (padding 8 18) was cascading to
        # these buttons and Qt was honouring that padding's implied
        # min content size, so each key rendered ~100 px wide
        # regardless of what we asked for. Putting min-width and
        # max-width inside the per-button CSS forces the size.
        pixel_w = int(width_units * _VK_UNIT_WIDTH + max(0, width_units - 1) * _VK_KEY_GAP)
        pixel_h = _VK_KEY_HEIGHT
        btn._vk_w = pixel_w  # type: ignore[attr-defined]
        btn._vk_h = pixel_h  # type: ignore[attr-defined]
        btn.setFixedSize(pixel_w, pixel_h)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda _checked=False, k=key: self._on_clicked(k))
        self._buttons_by_key.setdefault(key, []).append(btn)
        self._apply_button_style(btn, selected=False)
        return btn

    def _on_clicked(self, key: str) -> None:
        if self._mode == "single":
            # Toggle: clicking the already-selected key clears it,
            # clicking any other replaces.
            self._selected = [] if self._selected == [key] else [key]
        else:
            if key in self._selected:
                self._selected.remove(key)
            else:
                self._selected.append(key)
        self._refresh_visual()
        self.keys_changed.emit(self._format())

    def _format(self) -> str:
        if not self._selected:
            return ""
        if self._mode == "single":
            return self._selected[0]
        return "+".join(self._selected)

    def _refresh_visual(self) -> None:
        for key, buttons in self._buttons_by_key.items():
            sel = key in self._selected
            for btn in buttons:
                self._apply_button_style(btn, sel)

    def _apply_button_style(self, btn: QPushButton, selected: bool) -> None:
        # Bake explicit min/max width + height into the stylesheet
        # so the size sticks even with the wizard's general
        # QPushButton rule (which has padding 8 18) cascaded down.
        # padding: 0 also has to be set explicitly here so the parent
        # rule doesn't reintroduce horizontal slack.
        w = int(getattr(btn, "_vk_w", _VK_UNIT_WIDTH))
        h = int(getattr(btn, "_vk_h", _VK_KEY_HEIGHT))
        size_css = (
            f"  min-width: {w}px; max-width: {w}px;"
            f"  min-height: {h}px; max-height: {h}px;"
            f"  padding: 0;"
        )
        if selected:
            btn.setStyleSheet(
                f"QPushButton#vkKey {{"
                f"  background: {self._accent_color};"
                f"  color: #0B1620;"
                f"  border: 1px solid {self._accent_color};"
                f"  border-radius: 3px;"
                f"  font-weight: 700;"
                f"  font-size: 10px;"
                f"{size_css}"
                f"}}"
                # Qt QSS doesn't support `filter:`. Hover uses a subtle
                # outline highlight instead of CSS brightness.
                f"QPushButton#vkKey:hover {{ border: 1px solid rgba(255,255,255,0.45); }}"
            )
        else:
            btn.setStyleSheet(
                "QPushButton#vkKey {"
                "  background: rgba(255,255,255,0.05);"
                "  color: #DCE9F2;"
                "  border: 1px solid rgba(255,255,255,0.15);"
                "  border-radius: 3px;"
                "  font-weight: 600;"
                "  font-size: 10px;"
                f"{size_css}"
                "}"
                "QPushButton#vkKey:hover {"
                "  background: rgba(255,255,255,0.12);"
                "  border-color: rgba(255,255,255,0.30);"
                "}"
            )


class _GreenDotRadio(QRadioButton):
    """QRadioButton with a custom-painted indicator: a white outlined
    circle with a green dot in the middle when checked. Qt's default
    indicator follows the OS theme (often filled-blue on Win11) which
    clashes with the Touchless dark surface. Subclassing to paint the
    indicator ourselves gives a consistent look without fighting QSS
    quirks across Qt versions / OS styles.
    """

    _OUTER_RADIUS = 7
    _DOT_RADIUS = 3
    _GAP = 8  # space between indicator and text

    def __init__(self, text: str = "", color: str = "#1DE9B6", parent=None) -> None:
        super().__init__(text, parent)
        self._dot_color = color
        # Hide Qt's native indicator entirely; we paint our own.
        # Stylesheet on the *widget* avoids polluting the dialog-wide
        # QRadioButton rule.
        self.setStyleSheet("QRadioButton::indicator { width: 0; height: 0; }")
        # Reserve room on the left for our painted indicator + gap.
        self.setContentsMargins(self._OUTER_RADIUS * 2 + self._GAP, 0, 0, 0)

    def paintEvent(self, event) -> None:  # noqa: N802
        from PySide6.QtGui import QPainter, QPen, QColor

        # Let QRadioButton's default paintEvent draw the text + focus
        # rect; the contentsMargins above kept the text away from the
        # indicator slot. Then paint our circle over the cleared slot.
        super().paintEvent(event)

        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing)
            cx = self._OUTER_RADIUS + 1
            cy = self.height() // 2
            # Outer ring: white outline, transparent fill.
            pen = QPen(QColor("#FFFFFF"))
            pen.setWidthF(1.4)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(
                cx - self._OUTER_RADIUS, cy - self._OUTER_RADIUS,
                self._OUTER_RADIUS * 2, self._OUTER_RADIUS * 2,
            )
            # Inner dot: green when checked, nothing otherwise.
            if self.isChecked():
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(self._dot_color))
                p.drawEllipse(
                    cx - self._DOT_RADIUS, cy - self._DOT_RADIUS,
                    self._DOT_RADIUS * 2, self._DOT_RADIUS * 2,
                )
        finally:
            p.end()


class GestureTypeToggle(QWidget):
    """Three-segment pill: Static | Sequence | Dynamic.

    Equal-width checkable buttons in a single rounded frame. Defaults
    to "static". Emits `selection_changed(str)` on user clicks.
    """

    selection_changed = Signal(str)

    _OPTIONS = (
        ("static", "Static (pose)"),
        ("pose_sequence", "Sequence"),
        ("dynamic", "Dynamic (motion)"),
    )

    def __init__(self, accent_color: str, parent=None) -> None:
        super().__init__(parent)
        self._accent = accent_color or "#1DE9B6"
        self._selection = "static"
        self.setObjectName("gestureTypeToggle")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        self._buttons = {}
        for key, label in self._OPTIONS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(40)
            btn.clicked.connect(lambda checked=False, k=key: self._on_clicked(k))
            layout.addWidget(btn, 1)
            self._buttons[key] = btn
        self._buttons["static"].setChecked(True)
        self._apply_styles()

    def selection(self) -> str:
        return self._selection

    def set_selection(self, value: str, *, emit: bool = True) -> None:
        if value not in self._buttons:
            return
        if value == self._selection:
            # Still refresh checked state / styles; only skip the signal
            # when nothing changed.
            for key, btn in self._buttons.items():
                btn.setChecked(key == value)
            self._apply_styles()
            return
        self._selection = value
        for key, btn in self._buttons.items():
            btn.setChecked(key == value)
        self._apply_styles()
        if emit:
            self.selection_changed.emit(self._selection)

    def set_interactive(self, enabled: bool) -> None:
        """Enable/disable user clicks. Used in edit mode so the gesture
        type stays locked to what was recorded."""
        for btn in self._buttons.values():
            btn.setEnabled(bool(enabled))

    def _on_clicked(self, key: str) -> None:
        if key == self._selection:
            self._buttons[key].setChecked(True)
            return
        self.set_selection(key, emit=True)

    def _apply_styles(self) -> None:
        accent = self._accent
        for key, btn in self._buttons.items():
            active = key == self._selection
            if active:
                btn.setStyleSheet(
                    f"QPushButton {{ background: {accent}; color: #0b1220; "
                    f"border: none; border-radius: 14px; font-weight: 600; "
                    f"padding: 6px 10px; }}"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { background: rgba(255,255,255,0.06); "
                    "color: rgba(255,255,255,0.72); border: none; "
                    "border-radius: 14px; padding: 6px 10px; }"
                    "QPushButton:hover { background: rgba(255,255,255,0.12); }"
                )

class _WrappingInfoBanner(QLabel):
    """Word-wrapped info banner that reports a correct height for its
    current width so Static ↔ Dynamic ↔ Sequence switches don't leave
    the green box crushed or over-tall."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        # Account for stylesheet padding roughly; Qt's QLabel
        # heightForWidth already includes font metrics for wrap.
        return max(0, super().heightForWidth(max(1, width)))

    def sizeHint(self):  # noqa: N802
        from PySide6.QtCore import QSize

        w = self.width() if self.width() > 1 else 480
        return QSize(w, self.heightForWidth(w))


class CreateGestureWizard(QDialog):
    """Modal-ish dialog. Caller calls exec(); on accept, .result_payload
    is a WizardResult; on reject, None."""

    def __init__(
        self,
        accent_color: str,
        parent: Optional[QWidget] = None,
        *,
        edit_mode: bool = False,
        initial_name: str = "",
        initial_description: str = "",
        initial_hold: float = 1.0,
        initial_cooldown: float = 2.0,
        initial_action_kind: Optional[str] = None,
        initial_action_value: str = "",
        initial_gesture_type: str = "static",
        initial_duration_mode: str = "",
        original_name: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        # r51: install_indigo_chrome for Win10 + Win11 parity.
        from .window_chrome import install_indigo_chrome
        self._body = install_indigo_chrome(self, "Create custom gesture")
        self._edit_mode = bool(edit_mode)
        self.setWindowTitle("Edit Custom Gesture" if self._edit_mode else "Create Custom Gesture")
        self.setModal(True)
        # Keyboard's natural width with 22-px unit keys is ~334 px,
        # so the original 520-wide dialog has plenty of room. Kept
        # the modest bump to 540 for a small safety margin without
        # making the dialog feel oversized for users with the other
        # action kinds (text / URL / command / file / overlay).
        self.setMinimumWidth(540)
        # Minimum height kept small so the default state ("Choose an
        # action" — value cluster hidden) doesn't leave empty space
        # below the form. _fit_action_value_into_view() grows the
        # dialog when the user picks an action whose value UI needs
        # more room (keyboard, hotkey) and shrinks it back when they
        # switch away. 380 is enough for the basic fields (name,
        # description, hold, cooldown, action combo, buttons) with
        # no value cluster shown.
        self.setMinimumHeight(380)
        self._accent_color = accent_color
        self._original_name = original_name if self._edit_mode else None
        self._initial_name = initial_name
        self._initial_description = initial_description
        self._initial_hold = float(initial_hold)
        self._initial_cooldown = float(initial_cooldown)
        self._initial_action_kind = initial_action_kind
        self._initial_action_value = initial_action_value
        gtype = str(initial_gesture_type or "static").lower()
        if gtype not in ("static", "dynamic", "pose_sequence"):
            gtype = "static"
        self._initial_gesture_type = gtype
        self._initial_duration_mode = str(initial_duration_mode or "")
        self.result_payload: Optional[WizardResult] = None
        self._build()
        self._populate_initial_values()

    def showEvent(self, event):  # noqa: N802 (Qt API name)
        super().showEvent(event)
        try:
            apply_touchless_titlebar(self)
        except Exception:
            pass

    # --- UI -------------------------------------------------------------

    def _build(self) -> None:
        up_img = _spin_arrow_image_path("up")
        down_img = _spin_arrow_image_path("down")
        combo_img = _spin_arrow_image_path("combo")
        up_url = f'url("{up_img}")' if up_img else "none"
        down_url = f'url("{down_img}")' if down_img else "none"
        combo_url = f'url("{combo_img}")' if combo_img else "none"
        self.setStyleSheet(
            f"""
            QDialog {{ background: #0E1822; }}
            QLabel {{ color: #DCE9F2; font-size: 13px; }}
            QLabel#sectionTitle {{ color: #E5F6FF; font-weight: 700; font-size: 16px; }}
            QLineEdit, QComboBox {{
                background: rgba(255,255,255,0.05);
                color: #E5F6FF;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 13px;
            }}
            /* Spinbox: same face as before. Arrows are side-by-side on
               the right (the prior look); each button gets its own
               non-overlapping hit box so up/down both click. Real PNG
               triangles — CSS border-triangles paint as rectangles on
               this Qt build. */
            QDoubleSpinBox {{
                background: rgba(255,255,255,0.05);
                color: #E5F6FF;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 6px;
                padding: 6px 36px 6px 8px;
                font-size: 13px;
            }}
            QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: center right;
                width: 16px;
                height: 20px;
                right: 18px;
                border: none;
                background: transparent;
            }}
            QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: center right;
                width: 16px;
                height: 20px;
                right: 2px;
                border: none;
                background: transparent;
            }}
            QDoubleSpinBox::up-arrow {{
                image: {up_url};
                width: 10px;
                height: 10px;
            }}
            QDoubleSpinBox::down-arrow {{
                image: {down_url};
                width: 10px;
                height: 10px;
            }}
            QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid {self._accent_color};
            }}
            /* The dropdown POPUP — Qt renders it as a separate native widget,
               so its colors must be styled explicitly or it inherits the OS
               theme (white text on white background on some Win 11 setups). */
            QComboBox QAbstractItemView {{
                background: #0E1822;
                color: #E5F6FF;
                selection-background-color: {self._accent_color};
                selection-color: #0B1620;
                border: 1px solid rgba(255,255,255,0.18);
                outline: none;
                padding: 4px 0;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 6px 12px;
                color: #E5F6FF;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: {combo_url};
                width: 10px;
                height: 10px;
                margin-right: 8px;
            }}
            QPushButton {{
                background: rgba(255,255,255,0.06);
                color: #E5F6FF;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 8px;
                padding: 8px 18px;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton:hover {{ background: rgba(255,255,255,0.12); }}
            QPushButton#startBtn {{
                background: {self._accent_color};
                color: #0B1620;
                font-weight: 800;
            }}
            QPushButton#startBtn:hover {{ background: #FFFFFF; }}
            QPushButton#startBtn:disabled {{
                background: rgba(255,255,255,0.06);
                color: #5C6F7E;
            }}
            /* Thin green scrollbar for the form's QScrollArea — the
               default Win11 scrollbar is wide and grey, which clashes
               with the dark Touchless dialog. */
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                margin: 4px 0 4px 0;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {self._accent_color};
                min-height: 24px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #FFFFFF;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: transparent;
                height: 0;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            """
        )

        outer = QVBoxLayout(self._body)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(14)

        title = QLabel("Create Custom Gesture")
        title.setObjectName("sectionTitle")
        outer.addWidget(title)

        # Form lives inside a scroll area so smaller windows still let
        # the user reach every field.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # CSS-only viewport selectors don't always beat the OS default
        # on Win 11 — set the viewport background explicitly.
        scroll.viewport().setStyleSheet("background: transparent;")
        # Stash a reference so the banner methods can scroll the banner
        # into view and grow the dialog when it appears.
        self._form_scroll = scroll
        outer.addWidget(scroll, 1)

        form_widget = QWidget()
        form_widget.setStyleSheet("background: transparent;")
        scroll.setWidget(form_widget)
        root = QVBoxLayout(form_widget)
        root.setContentsMargins(0, 0, 8, 0)
        root.setSpacing(14)

        # Gesture type — Static (pose) vs Dynamic (motion). Sits at
        # the top of the form because changing this changes what the
        # rest of the dialog means. Wrapped in a tighter sub-layout
        # so the label + toggle + banner cluster reads as one
        # compact control instead of three fields with the form's
        # default 14 px gap between them.
        gesture_type_block = QVBoxLayout()
        gesture_type_block.setContentsMargins(0, 0, 0, 0)
        gesture_type_block.setSpacing(4)

        type_label = QLabel("Gesture type")
        type_label.setStyleSheet("padding: 0; margin: 0;")
        gesture_type_block.addWidget(type_label)

        self.gesture_type_toggle = GestureTypeToggle(self._accent_color)
        self.gesture_type_toggle.selection_changed.connect(
            self._on_gesture_type_changed
        )
        gesture_type_block.addWidget(self.gesture_type_toggle)

        # One shared type-info banner (text swapped per selection). Two
        # separate show/hide labels fought the layout when hopping
        # Static → Dynamic → Sequence and left the green box looking
        # crushed / over-wide.
        self._type_banner = _WrappingInfoBanner()
        self._type_banner.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(29,233,182,0.10);"
            "  border: 1px solid rgba(29,233,182,0.45);"
            "  border-radius: 8px;"
            "  padding: 8px 12px;"
            "  color: #BAE6FD;"
            "  font-size: 12px;"
            "  margin-top: 4px;"
            "}"
        )
        self._type_banner.setVisible(False)
        gesture_type_block.addWidget(self._type_banner)

        # Dynamic-only: duration-per-take picker. Lives in the wizard
        # so the recorder window opens straight into the camera with a
        # mode already chosen (no mode toggle mid-flow).
        self._duration_block = QWidget()
        dur_v = QVBoxLayout(self._duration_block)
        dur_v.setContentsMargins(0, 6, 0, 0)
        dur_v.setSpacing(4)
        dur_label = QLabel("Duration per take")
        dur_label.setStyleSheet("padding: 0; margin: 0; color: #94A3B8;")
        dur_v.addWidget(dur_label)
        dur_row = QHBoxLayout()
        dur_row.setSpacing(14)
        self._dur_until = _GreenDotRadio("Until stopped", color=self._accent_color)
        self._dur_1s = _GreenDotRadio("1 s", color=self._accent_color)
        self._dur_2s = _GreenDotRadio("2 s", color=self._accent_color)
        self._dur_3s = _GreenDotRadio("3 s", color=self._accent_color)
        self._dur_until.setChecked(True)
        for rb in (self._dur_until, self._dur_1s, self._dur_2s, self._dur_3s):
            rb.setStyleSheet("color: #E5F6FF; background: transparent;")
            dur_row.addWidget(rb)
        dur_row.addStretch(1)
        self._dur_group = QButtonGroup(self)
        self._dur_group.addButton(self._dur_until, 0)
        self._dur_group.addButton(self._dur_1s, 1)
        self._dur_group.addButton(self._dur_2s, 2)
        self._dur_group.addButton(self._dur_3s, 3)
        dur_v.addLayout(dur_row)
        self._duration_block.setVisible(False)
        gesture_type_block.addWidget(self._duration_block)

        root.addLayout(gesture_type_block)

        # Name
        root.addWidget(QLabel("Name *"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Open Inbox")
        root.addWidget(self.name_edit)

        # Description
        root.addWidget(QLabel("Description (optional)"))
        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("What the gesture does, in your own words")
        root.addWidget(self.desc_edit)

        # Timing
        timing_row = QHBoxLayout()
        timing_row.setSpacing(14)
        # v1.1.8.2: hold-to-activate wrapped in a QWidget so
        # _on_gesture_type_changed can hide it as a unit on the Dynamic
        # tab. Dynamic gestures fire on motion match, not on a hold — a
        # hold slider is meaningless there and was confusing users.
        self._hold_block = QWidget()
        timing_box1 = QVBoxLayout(self._hold_block)
        timing_box1.setContentsMargins(0, 0, 0, 0)
        timing_box1.addWidget(QLabel("Hold to activate (seconds)"))
        self.hold_spin = QDoubleSpinBox()
        self.hold_spin.setRange(0.2, 5.0)
        self.hold_spin.setSingleStep(0.1)
        self.hold_spin.setDecimals(1)
        self.hold_spin.setValue(1.0)
        self.hold_spin.setButtonSymbols(QDoubleSpinBox.UpDownArrows)
        self.hold_spin.setKeyboardTracking(False)
        timing_box1.addWidget(self.hold_spin)
        timing_row.addWidget(self._hold_block)

        timing_box2 = QVBoxLayout()
        timing_box2.addWidget(QLabel("Cooldown after fire (seconds)"))
        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.0, 30.0)
        self.cooldown_spin.setSingleStep(0.5)
        self.cooldown_spin.setDecimals(1)
        self.cooldown_spin.setValue(2.0)
        self.cooldown_spin.setButtonSymbols(QDoubleSpinBox.UpDownArrows)
        self.cooldown_spin.setKeyboardTracking(False)
        timing_box2.addWidget(self.cooldown_spin)
        timing_row.addLayout(timing_box2)

        root.addLayout(timing_row)

        # Action kind. Starts unselected so the user has to deliberately
        # pick — the value-input row below stays hidden until they do.
        root.addWidget(QLabel("Action *"))
        self.action_combo = QComboBox()
        self.action_combo.addItem("— Choose an action —", None)
        for label, kind, *_ in _ACTION_KINDS:
            self.action_combo.addItem(label, kind)
        self.action_combo.setCurrentIndex(0)
        self.action_combo.currentIndexChanged.connect(self._refresh_action_value)
        root.addWidget(self.action_combo)

        # Action value (label + line edit, prompt changes with combo).
        # Hidden until the user picks a non-placeholder action.
        self.action_value_label = QLabel("")
        self.action_value_label.hide()
        root.addWidget(self.action_value_label)
        self.action_value_edit = QLineEdit()
        self.action_value_edit.hide()
        # Manual typing in the line edit updates the keyboard's
        # highlighted keys so the two views stay in sync. Use
        # textEdited (not textChanged) — textEdited fires only on
        # actual user input, so the keyboard's own setText calls
        # don't loop back through here.
        self.action_value_edit.textEdited.connect(self._on_value_text_edited)
        root.addWidget(self.action_value_edit)

        # Inline status banner — hidden by default. Pinned directly
        # below the action value line edit so error / warning text
        # appears in context with the field that caused it. Used by
        # the show_overlay_drawing validation flow to surface a red
        # "no such file" pill or a yellow "multiple matches" pill
        # without popping a separate QMessageBox.
        self._status_banner = QLabel("")
        self._status_banner.setWordWrap(True)
        self._status_banner.setAlignment(Qt.AlignCenter)
        self._status_banner.hide()
        root.addWidget(self._status_banner)

        # Mock keyboard, shown only for keystroke / hotkey actions.
        # Single-mode for keystroke (one key replaces another),
        # combo-mode for hotkey (chord). The user can either type
        # in the line edit above OR click keys here; both paths
        # stay in sync.
        self.action_value_keyboard = _VirtualKeyboard(self._accent_color, parent=self)
        self.action_value_keyboard.hide()
        self.action_value_keyboard.keys_changed.connect(self._on_keyboard_keys_changed)
        root.addWidget(self.action_value_keyboard)

        root.addStretch(1)

        # Buttons pinned outside the scroll area so they stay reachable.
        # autoDefault=False on every button so pressing Enter in any
        # text field doesn't accidentally activate Back/Save/Start. Enter
        # is handled explicitly in keyPressEvent below — it advances to
        # Start ONLY when Start is actually enabled.
        button_row = QHBoxLayout()
        back_button = QPushButton("Back")
        back_button.setAutoDefault(False)
        back_button.setDefault(False)
        back_button.clicked.connect(self.reject)
        button_row.addWidget(back_button)
        button_row.addStretch(1)
        self._start_button = QPushButton(
            "Save" if self._edit_mode else "Start"
        )
        self._start_button.setObjectName("startBtn")
        self._start_button.setAutoDefault(False)
        self._start_button.setDefault(False)
        self._start_button.clicked.connect(self._on_start)
        # Disabled until the user picks an action — keeps the "next part
        # only appears after action chosen" promise even via shortcut.
        self._start_button.setEnabled(False)
        button_row.addWidget(self._start_button)
        outer.addLayout(button_row)

    def keyPressEvent(self, event):  # noqa: N802 (Qt API name)
        """Swallow Enter/Return so it never closes the dialog or
        triggers Back. Only forward to Start when the form is fully
        valid (Start button enabled). Escape uses the QDialog default
        (reject) — handled by super(). Every other key passes through
        normally so text editing works."""
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self._start_button.isEnabled():
                self._on_start()
            event.accept()
            return
        super().keyPressEvent(event)

    def _populate_initial_values(self) -> None:
        """For edit mode: pre-fill the form with the existing gesture's
        values so the user can tweak instead of typing everything fresh."""
        # Gesture type first so banners / hold / duration visibility
        # match the recorded kind before other fields fill in.
        try:
            self.gesture_type_toggle.set_selection(
                self._initial_gesture_type, emit=True
            )
            # Edit = metadata + action only; type can't change without
            # re-recording samples / trajectories / sequence steps.
            if self._edit_mode:
                self.gesture_type_toggle.set_interactive(False)
        except Exception:
            pass
        self._apply_duration_mode(self._initial_duration_mode)
        if self._edit_mode:
            try:
                for rb in (
                    self._dur_until, self._dur_1s, self._dur_2s, self._dur_3s
                ):
                    rb.setEnabled(False)
            except Exception:
                pass
        if self._initial_name:
            self.name_edit.setText(self._initial_name)
        if self._initial_description:
            self.desc_edit.setText(self._initial_description)
        try:
            self.hold_spin.setValue(self._initial_hold)
        except Exception:
            pass
        try:
            self.cooldown_spin.setValue(self._initial_cooldown)
        except Exception:
            pass
        if self._initial_action_kind:
            for i in range(self.action_combo.count()):
                if self.action_combo.itemData(i) == self._initial_action_kind:
                    self.action_combo.setCurrentIndex(i)
                    break
            if self._initial_action_value:
                self.action_value_edit.setText(self._initial_action_value)
                # Mirror to the mock keyboard for keystroke / hotkey
                # so editing an existing gesture shows the saved keys
                # already highlighted. The keyboard's set_mode call
                # in _refresh_action_value already ran via the
                # currentIndexChanged trigger above, so the mode is
                # set; we only need to push the value in.
                if self._initial_action_kind in ("keystroke", "hotkey"):
                    self.action_value_keyboard.set_value(self._initial_action_value)

    def _apply_duration_mode(self, mode: str) -> None:
        """Select the matching Duration radio (dynamic edit prefill)."""
        m = (mode or "").strip().lower()
        try:
            if m == "fixed_1s" or m == "fixed_short":
                self._dur_1s.setChecked(True)
            elif m == "fixed_2s":
                self._dur_2s.setChecked(True)
            elif m in ("fixed_3s", "fixed_long"):
                self._dur_3s.setChecked(True)
            else:
                self._dur_until.setChecked(True)
        except Exception:
            pass

    _TYPE_BANNER_TEXT = {
        "dynamic": (
            "Dynamic captures your motion as keypoint paths over time — "
            "shape changes, movement, and pacing included. Record "
            "several takes; the app selects moving landmarks and matches "
            "live motion against those paths. Prefer Until stopped for "
            "longer sequences."
        ),
        "pose_sequence": (
            "Sequence: record an ordered chain of held poses you invent "
            "in one continuous take. Touchless learns each pose, timing, "
            "order, and hand — and saves a video clip of your take."
        ),
    }

    def _on_gesture_type_changed(self, value: str) -> None:
        """Show/hide mode-specific banners + controls."""
        text = self._TYPE_BANNER_TEXT.get(value, "")
        try:
            if text:
                self._type_banner.setText(text)
                self._type_banner.setVisible(True)
            else:
                self._type_banner.clear()
                self._type_banner.setVisible(False)
        except Exception:
            pass
        try:
            self._duration_block.setVisible(value == "dynamic")
        except Exception:
            pass
        # Hold only applies to single static poses.
        try:
            self._hold_block.setVisible(value == "static")
        except Exception:
            pass
        # Sequence / Dynamic banners (and Dynamic duration radios) add
        # height; Static hides them and shows Hold instead. Grow or
        # shrink so Action * stays visible without scrolling.
        QTimer.singleShot(0, self._fit_action_value_into_view)

    def duration_mode(self) -> str:
        """Return the selected duration mode for Dynamic gestures.

        One of "until_stopped", "fixed_1s", "fixed_2s", "fixed_3s".
        Defaults to "until_stopped". Meaningful only when
        gesture_type() == "dynamic".
        """
        try:
            if self._dur_1s.isChecked():
                return "fixed_1s"
            if self._dur_2s.isChecked():
                return "fixed_2s"
            if self._dur_3s.isChecked():
                return "fixed_3s"
        except Exception:
            pass
        return "until_stopped"

    def gesture_type(self) -> str:
        """'static', 'dynamic', or 'pose_sequence'."""
        toggle = getattr(self, "gesture_type_toggle", None)
        return toggle.selection() if toggle is not None else "static"

    def _refresh_action_value(self) -> None:
        kind = self.action_combo.currentData()
        if kind is None:
            # Placeholder ("Choose an action") selected — keep value row
            # hidden and the Start button disabled.
            self.action_value_label.hide()
            self.action_value_edit.hide()
            self.action_value_keyboard.hide()
            self.action_value_edit.setText("")
            self._start_button.setEnabled(False)
            # Shrink the dialog back to its compact default-state
            # height so there's no empty space below the form when
            # the user resets the action picker.
            QTimer.singleShot(0, self._fit_action_value_into_view)
            return
        # Look up the matching prompt + placeholder for the chosen kind.
        for _label, k, prompt, placeholder in _ACTION_KINDS:
            if k == kind:
                # Keystroke / hotkey actions get the mock keyboard
                # below the input plus a clearer prompt that mentions
                # both input methods. Other action kinds keep their
                # original "Key name" / "Keys (joined by +)" / "URL"
                # / etc. prompt.
                if kind in ("keystroke", "hotkey"):
                    self.action_value_label.setText("Type or select key(s) below")
                else:
                    self.action_value_label.setText(prompt)
                self.action_value_edit.setPlaceholderText(placeholder)
                self.action_value_edit.setText("")
                self.action_value_label.show()
                self.action_value_edit.show()
                if kind == "keystroke":
                    self.action_value_keyboard.set_mode("single")
                    self.action_value_keyboard.set_value("")
                    self.action_value_keyboard.show()
                elif kind == "hotkey":
                    self.action_value_keyboard.set_mode("combo")
                    self.action_value_keyboard.set_value("")
                    self.action_value_keyboard.show()
                else:
                    self.action_value_keyboard.hide()
                break
        self._start_button.setEnabled(True)
        # Auto-grow the dialog so the keyboard (and any other newly-
        # revealed action-value widgets) stay visible without forcing
        # the user to scroll. Then scroll the value section into view
        # in case the dialog was already at its max height. Deferred
        # via singleShot so Qt finishes laying out the now-visible
        # widgets before we measure their preferred height.
        QTimer.singleShot(0, self._fit_action_value_into_view)

    def _fit_action_value_into_view(self) -> None:
        """Resize the dialog vertically (up to a screen-aware cap) so
        the action-value cluster fits exactly, then scroll it into
        view. Resizes in BOTH directions: grows when the user picks
        an action whose value UI needs more space (e.g. keyboard
        appears), and shrinks back when they switch to a smaller
        action (e.g. text snippet) so the dialog isn't left with a
        big empty area below the form."""
        scroll = getattr(self, "_form_scroll", None)
        if scroll is None:
            return
        try:
            screen = self.screen() or QApplication.primaryScreen()
            cap_h = int(screen.availableGeometry().height() * 0.92) if screen else 1080
        except Exception:
            cap_h = 1080
        try:
            inner = scroll.widget()
            banner = getattr(self, "_type_banner", None)
            if banner is not None and banner.isVisible():
                # Recompute wrap height for the current viewport without
                # permanently pinning minWidth (that crushed the green
                # box across Static/Dynamic/Sequence switches).
                banner.setMinimumHeight(0)
                banner.setMaximumHeight(16777215)
                banner.updateGeometry()
            if inner is not None and inner.layout() is not None:
                inner.layout().invalidate()
                inner.layout().activate()
                inner.adjustSize()
            needed = (
                (inner.sizeHint().height() if inner is not None else 0) + 200
            )  # chrome + padding
            target = min(cap_h, max(self.minimumHeight(), needed))
            # Only resize when the delta is meaningful — avoids a
            # one-pixel jitter from Qt's layout rounding on every
            # combo change.
            if abs(target - self.height()) >= 8:
                self.resize(self.width(), target)
            # Prefer the deepest visible action-value widget; fall back
            # to the Action combo so type-toggle / banner growth still
            # keeps "Action *" on screen before an action is chosen.
            if self.action_value_keyboard.isVisible():
                target_widget = self.action_value_keyboard
            elif self.action_value_edit.isVisible():
                target_widget = self.action_value_edit
            else:
                target_widget = self.action_combo
            scroll.ensureWidgetVisible(target_widget, 0, 24)
        except Exception:
            pass

    def _on_keyboard_keys_changed(self, value: str) -> None:
        """User clicked / unclicked a key on the mock keyboard.
        Push the formatted string into the line edit. Setting via
        setText doesn't fire textEdited, so this won't loop back
        through _on_value_text_edited."""
        self.action_value_edit.setText(value)

    def _on_value_text_edited(self, text: str) -> None:
        """User typed in the line edit. Mirror to the keyboard's
        highlighted keys so clicks-vs-typing stay consistent. No-op
        when the keyboard isn't visible (non keystroke/hotkey
        action), so other action kinds aren't affected."""
        if not self.action_value_keyboard.isVisible():
            return
        self.action_value_keyboard.set_value(text)

    # --- validation + accept --------------------------------------------

    def _show_banner(self, message: str, *, kind: str = "error") -> None:
        """Show the inline status banner with one of three styles:
          - 'error'   → red pill
          - 'warning' → yellow pill
          - 'info'    → neutral blue pill
        kind controls the colours but layout is identical.

        The banner lives inside the form's scroll area; on show we
        grow the dialog vertically so the banner is visible without
        forcing the user to scroll down to it. Restored on
        _clear_banner."""
        palette = {
            "error":   ("#3B1A1F", "#FF6B6B", "#FFD2D2"),
            "warning": ("#3A2E12", "#F5B450", "#FFE7C0"),
            "info":    ("#15273E", "#4FB3FF", "#D2E7FF"),
        }
        bg, border, fg = palette.get(kind, palette["error"])
        self._status_banner.setText(message)
        self._status_banner.setStyleSheet(
            "QLabel {"
            f"  background-color: {bg};"
            f"  color: {fg};"
            f"  border: 1px solid {border};"
            "  border-radius: 10px;"
            "  padding: 8px 14px;"
            "  font-size: 11pt;"
            "  font-weight: 600;"
            "}"
        )
        self._status_banner.show()

        # Grow the dialog to fit the banner so it sits inside the
        # visible viewport. Baseline height captured the first time
        # so subsequent shows don't compound.
        if getattr(self, "_banner_height_baseline", None) is None:
            self._banner_height_baseline = self.height()
        self._status_banner.adjustSize()
        banner_h = self._status_banner.sizeHint().height()
        target_h = max(
            self._banner_height_baseline,
            self._banner_height_baseline + banner_h + 24,
        )
        if self.height() < target_h:
            self.resize(self.width(), target_h)
        # Scroll the banner into the viewport in case the form was
        # already mid-scroll when the user clicked Start.
        scroll = getattr(self, "_form_scroll", None)
        if scroll is not None:
            try:
                scroll.ensureWidgetVisible(self._status_banner)
            except Exception:
                pass

    def _clear_banner(self) -> None:
        self._status_banner.hide()
        self._status_banner.setText("")
        baseline = getattr(self, "_banner_height_baseline", None)
        if baseline is not None and self.height() > baseline:
            self.resize(self.width(), baseline)
        self._banner_height_baseline = None

    def _validate_show_overlay_drawing(self, raw_value: str) -> Optional[str]:
        """Validate a show_overlay_drawing filename at gesture-creation
        time. Returns the absolute path string the gesture should bind
        to, or None if validation can't complete (banner has already
        been set, caller should abort).

        Flow:
          0 matches  → red error banner, abort.
          1 match    → return absolute path.
          >1 matches → yellow banner + modal chooser; on pick
                       return the chosen absolute path, on cancel
                       leave banner up and abort."""
        from .drawing_overlay_window import (
            resolve_drawing_path,
            search_drawings_by_filename,
        )

        filename = raw_value.strip()
        if filename and not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filename = filename + ".png"
        if not filename:
            self._show_banner("Error: please enter a drawing filename.", kind="error")
            return None

        # Always do a full filesystem search, even if the configured
        # drawings_save_dir has a match — the whole point of doing
        # validation at creation time is to catch shadow files in
        # other folders BEFORE the gesture is bound.
        matches = search_drawings_by_filename(filename)

        # Pull the configured drawings dir so a save-dir hit on a
        # non-indexed folder is still included. Loaded lazily here
        # because the wizard doesn't take a config in its constructor.
        configured_dir = ""
        try:
            from hgr.config.app_config import load_config
            configured_dir = str(getattr(load_config(), "drawings_save_dir", "") or "")
        except Exception:
            configured_dir = ""
        configured = resolve_drawing_path(filename, configured_dir)
        if configured is not None:
            already = {str(p).lower() for p in matches}
            if str(configured).lower() not in already:
                matches.insert(0, configured)

        if len(matches) == 0:
            self._show_banner(
                f"Error: no file named “{filename}” exists on this system.",
                kind="error",
            )
            return None
        if len(matches) == 1:
            self._clear_banner()
            return str(matches[0])

        # Multiple matches → yellow banner + modal chooser.
        self._show_banner(
            f"There are multiple files called “{filename}” — please select one.",
            kind="warning",
        )
        from .drawing_chooser_dialog import DrawingChooserDialog
        chooser = DrawingChooserDialog(filename, matches, parent=self)
        if chooser.exec() != QDialog.Accepted:
            return None
        chosen = chooser.chosen_path
        if chosen is None:
            return None
        self._clear_banner()
        return str(chosen)

    def _on_start(self) -> None:
        # Clear any prior banner before we re-validate the form.
        self._clear_banner()
        name = self.name_edit.text().strip()
        if not name:
            self._error("Please enter a gesture name.")
            return
        # Spaces are now allowed (e.g., "open chrome", "my wave"); the
        # registry stores the literal name so display reflects what
        # the user typed. The thumbnail filename is sanitised
        # separately in custom_gestures_recorder._save_thumbnail_to_disk.

        # Name-conflict check against the registry. In edit mode, the
        # gesture's own existing name is fine (no warning needed) — only
        # warn if the user changed it to collide with a DIFFERENT
        # gesture.
        registry = GestureRegistry()
        registry.load()
        existing = registry.get(name)
        unchanged_in_edit = (
            self._edit_mode
            and self._original_name is not None
            and name == self._original_name
        )
        if existing is not None and not unchanged_in_edit:
            answer = touchless_message_box(
                self,
                "Gesture name already exists",
                f"A gesture named '{name}' already exists.\n\n"
                f"Continuing will overwrite it. Or click Cancel to pick a "
                f"different name.",
                icon=QMessageBox.Warning,
                buttons=QMessageBox.Ok | QMessageBox.Cancel,
                default_button=QMessageBox.Cancel,
            )
            if answer != QMessageBox.Ok:
                return

        action_kind = self.action_combo.currentData()
        if action_kind is None:
            self._error("Please choose an action from the dropdown.")
            return
        action_value = self.action_value_edit.text().strip()
        if action_kind != "noop" and not action_value:
            self._error("Please fill in the action value.")
            return

        # show_overlay_drawing: resolve the filename to an absolute
        # path right here so the binding is unambiguous. Validation
        # surfaces a banner on miss / multi-match and the user fixes
        # the form without losing what they typed.
        resolved_drawing_path: Optional[str] = None
        if action_kind == "show_overlay_drawing":
            resolved_drawing_path = self._validate_show_overlay_drawing(action_value)
            if resolved_drawing_path is None:
                return  # banner already shown / chooser cancelled

        try:
            action = self._build_action(
                action_kind, action_value,
                resolved_drawing_path=resolved_drawing_path,
            )
        except ValueError as exc:
            self._error(str(exc))
            return

        # v1.1.8.2: dynamic gestures fire on motion match — no hold
        # semantics — so force hold_seconds = 0 regardless of what the
        # (now-hidden) hold spinbox holds. Keeps saved payloads honest
        # for anything that reads hold_s at run-time.
        gtype = self.gesture_type()
        self.result_payload = WizardResult(
            name=name,
            description=self.desc_edit.text().strip(),
            hold_seconds=(0.0 if gtype in ("dynamic", "pose_sequence")
                          else float(self.hold_spin.value())),
            cooldown_seconds=float(self.cooldown_spin.value()),
            action=action,
            duration_mode=(
                self.duration_mode() if gtype == "dynamic" else ""
            ),
        )
        self.accept()

    def _build_action(
        self,
        kind: str,
        value: str,
        *,
        resolved_drawing_path: Optional[str] = None,
    ) -> Action:
        # Both cooldown AND hold-to-activate are stored in the payload so
        # the live runner reads per-gesture timing back at run-time.
        # action.cooldown_seconds() already reads cooldown_s; the runner
        # reads hold_s.
        # Dynamic / pose_sequence fire without a static hold_s.
        _is_instant = self.gesture_type() in ("dynamic", "pose_sequence")
        timing_payload = {
            "cooldown_s": float(self.cooldown_spin.value()),
            "hold_s": 0.0 if _is_instant else float(self.hold_spin.value()),
        }
        if kind == "keystroke":
            return Action(kind=kind, payload={"key": value, **timing_payload})
        if kind == "hotkey":
            keys = [k.strip() for k in value.split("+") if k.strip()]
            if not keys:
                raise ValueError("Hotkey combo must include at least one key.")
            return Action(kind=kind, payload={"keys": keys, **timing_payload})
        if kind == "text":
            # Store exactly what the user typed — no escaping / encoding.
            return Action(kind=kind, payload={"text": value, **timing_payload})
        if kind == "open_url":
            raw = value.strip()
            # Never percent-encode here. Also don't auto-prefix https://
            # onto spaced plain text ("my snap") — that used to become
            # https://my snap and the browser then showed my%20snap.
            if "://" not in raw and not raw.startswith("/"):
                if any(ch.isspace() for ch in raw):
                    raise ValueError(
                        "That looks like plain text, not a URL. "
                        "Choose “Type a text snippet” to type it exactly "
                        "as written (spaces stay spaces), or enter a full "
                        "URL like https://example.com."
                    )
                raw = "https://" + raw
            else:
                # Reject already-schemed URLs whose host is spaced words
                # with no domain dot — same foot-gun as above.
                try:
                    from urllib.parse import urlparse

                    host = (urlparse(raw).netloc or "").strip()
                    if (
                        " " in host
                        and "." not in host
                        and ":" not in host
                    ):
                        raise ValueError(
                            "That URL host has spaces and isn't a website. "
                            "Choose “Type a text snippet” to type the text "
                            "exactly, or enter a real URL."
                        )
                except ValueError:
                    raise
                except Exception:
                    pass
            return Action(kind=kind, payload={"url": raw, **timing_payload})
        if kind == "run_command":
            return Action(kind=kind, payload={"command": value, "shell": True, **timing_payload})
        if kind == "open_file":
            # Strip wrapping quotes — Explorer's "Copy as path"
            # context-menu entry surrounds paths with double-quotes,
            # and pasting that straight in shouldn't break the
            # action. The executor strips them too as a defence in
            # depth, but normalising here keeps the stored payload
            # clean for display in Recent Actions / edit-mode.
            cleaned = value.strip().strip('"').strip("'")
            return Action(kind=kind, payload={"path": cleaned, **timing_payload})
        if kind == "show_overlay_drawing":
            # Filename is stored for display + as a fallback if the
            # absolute path ever goes stale (file moved/deleted).
            # `path` is the absolute resolution captured at creation
            # time after the disambiguation flow — fire-time
            # resolution prefers `path` so we never re-prompt the
            # user about a binding they already decided.
            filename = value
            if filename and not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filename = filename + ".png"
            payload: dict = {"filename": filename, **timing_payload}
            if resolved_drawing_path:
                payload["path"] = resolved_drawing_path
            return Action(kind=kind, payload=payload)
        return Action(kind="noop", payload=timing_payload)

    def _error(self, message: str) -> None:
        touchless_message_box(
            self, "Cannot create gesture", message,
            icon=QMessageBox.Warning, buttons=QMessageBox.Ok,
        )

# Author: Konstantin Markov

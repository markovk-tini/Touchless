"""Consistent Touchless brand chrome for every user-visible pop-up.

Most dialogs in the app use Qt's native title bar (the OS draws it).
By default that produces white-on-light on Windows 11, very different
from the deep-indigo title bar the main window uses (`#1F2D6B` via
the custom frameless `TitleBar` widget).

This helper unifies both touch points without forcing every dialog
to go frameless:

  1. **Icon top-left** — calls `setWindowIcon(QApplication.windowIcon())`
     which Windows displays at the top-left of the native title bar.
     `QApplication.windowIcon()` is set once at app startup from
     `touchless_icon.ico`, so all callers share the same hand icon.

  2. **Title-bar color** — Windows 11 (build 22000+) exposes
     `DWMWA_CAPTION_COLOR` (35) and `DWMWA_TEXT_COLOR` (36) via
     `DwmSetWindowAttribute`. We set them to Touchless indigo / light
     text so the native bar matches the main window's custom bar.

Behavior on older Windows / other OSes:
  * Icon is set unconditionally (Qt handles it everywhere).
  * DWM calls silently no-op on pre-Win11 / non-Windows — the dialog
    keeps the OS default title color, which is fine.

The DWM attribute IDs are integer constants, so no SDK header
dependency — just ctypes against `dwmapi.dll`.
"""
from __future__ import annotations

import ctypes
import sys
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)


# Touchless brand colors. Must MATCH the main window's title bar
# (which uses its own DwmSetWindowAttribute call in main_window.py)
# so every dialog and pop-up looks like part of the same app.
# COLORREF format is 0x00BBGGRR (little-endian RGB).
#   #1F2D6B (Touchless deep indigo)  =>  COLORREF 0x006B2D1F
#   #E5F6FF (Touchless light text)   =>  COLORREF 0x00FFF6E5
# The older #0B3D91 (brighter primary blue) lived here before --
# user noticed the custom-gesture creator's title bar didn't match
# the main window after the main window's title-bar repaint
# landed (#1F2D6B). Realigning them here propagates to every
# dialog that calls apply_touchless_chrome (recorder, sandbox,
# wizard, gesture pose picker, etc.).
_CAPTION_COLORREF = 0x006B2D1F
_TEXT_COLORREF = 0x00FFF6E5

# Windows DwmSetWindowAttribute attribute IDs (Win11 22000+).
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36


def apply_touchless_chrome(window: QWidget) -> None:
    """Apply the Touchless icon + title-bar color to `window`.

    Safe to call in any dialog's __init__. The DWM call defers via
    QTimer.singleShot so it lands AFTER Qt has created the native
    window handle (winId() before the window is shown sometimes
    returns 0, which would make the DWM call a no-op).
    """
    # Icon: works on any platform, on any title-bar style.
    icon = QApplication.windowIcon()
    if not icon.isNull():
        try:
            window.setWindowIcon(icon)
        except Exception:
            pass

    # DWM color: Windows only, and only meaningful on Win11. Defer
    # via singleShot so Qt has finished native-handle creation.
    if not sys.platform.startswith("win"):
        return
    QTimer.singleShot(0, lambda w=window: _apply_dwm_caption_color(w))


def touchless_message_box(
    parent: Optional[QWidget],
    title: str,
    text: str,
    *,
    icon: "QMessageBox.Icon" = QMessageBox.Warning,
    buttons: "QMessageBox.StandardButtons" = QMessageBox.Ok,
    default_button: "QMessageBox.StandardButton" = QMessageBox.NoButton,
) -> "QMessageBox.StandardButton":
    """Show a Touchless-chromed message box. Signature-compatible
    with the old QMessageBox-wrapping version so no callers change.

    r51: rewritten from `QMessageBox + apply_touchless_chrome` to a
    frameless QDialog + `_IndigoTitleBar` so it renders indigo on
    both Windows 10 AND 11. The old wrapper worked only on Win11
    (DWMWA_CAPTION_COLOR); on Win10 dad saw white/black OS chrome.

    Returns the StandardButton the user clicked (or Cancel/Close on
    escape / X-button dismiss, matching QMessageBox semantics).
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setWindowFlag(Qt.FramelessWindowHint, True)
    dialog.setModal(True)
    # Same dark surface as the rest of the app so the body doesn't
    # look like a floating white pill under the indigo bar.
    dialog.setStyleSheet(
        f"QDialog {{ background-color: {_INDIGO_MSG_BG}; }}"
        f"QLabel {{ color: {_INDIGO_FG}; font-size: 13px; }}"
        f"QDialogButtonBox QPushButton {{"
        f"  background-color: {_INDIGO_MSG_BTN};"
        f"  color: {_INDIGO_FG};"
        f"  border: 1px solid {_INDIGO_MSG_BTN_BORDER};"
        f"  padding: 6px 16px;"
        f"  min-width: 76px;"
        f"  border-radius: 4px;"
        f"  font-size: 13px;"
        f"}}"
        f"QDialogButtonBox QPushButton:default {{"
        f"  background-color: {_INDIGO_ACCENT};"
        f"  border-color: {_INDIGO_ACCENT};"
        f"}}"
        f"QDialogButtonBox QPushButton:hover {{"
        f"  background-color: {_INDIGO_MSG_BTN_HOVER};"
        f"}}"
    )
    outer = QVBoxLayout(dialog)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    outer.addWidget(_IndigoTitleBar(dialog, title))

    body = QWidget(dialog)
    body_layout = QGridLayout(body)
    body_layout.setContentsMargins(20, 20, 20, 16)
    body_layout.setHorizontalSpacing(14)
    body_layout.setVerticalSpacing(14)

    # Icon column. Map QMessageBox.Icon -> QStyle SP pixmap so we
    # get the same iconography QMessageBox would have used.
    icon_map = {
        QMessageBox.Warning: QStyle.SP_MessageBoxWarning,
        QMessageBox.Critical: QStyle.SP_MessageBoxCritical,
        QMessageBox.Information: QStyle.SP_MessageBoxInformation,
        QMessageBox.Question: QStyle.SP_MessageBoxQuestion,
    }
    sp_icon = icon_map.get(icon)
    if sp_icon is not None:
        style = QApplication.style()
        icon_pixmap = style.standardIcon(sp_icon).pixmap(32, 32)
        icon_label = QLabel(body)
        icon_label.setPixmap(icon_pixmap)
        icon_label.setFixedSize(32, 32)
        body_layout.addWidget(icon_label, 0, 0, Qt.AlignTop | Qt.AlignLeft)

    text_label = QLabel(text, body)
    text_label.setWordWrap(True)
    text_label.setTextInteractionFlags(
        Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
    )
    text_label.setMinimumWidth(320)
    text_label.setMaximumWidth(520)
    body_layout.addWidget(text_label, 0, 1, Qt.AlignTop | Qt.AlignLeft)
    body_layout.setColumnStretch(1, 1)

    # Custom button row: skips QDialogButtonBox entirely to avoid two
    # PySide6 6.4+ hazards on Windows — (a) StandardButton(int(flag))
    # bitmask cast produces a broken enum value, (b) int(role) inside
    # the clicked handler throws TypeError on the Qt6 strict enum,
    # crashing the handler so clicks silently fail to close the dialog.
    # Full manual layout also lets us put Yes on the right and style
    # only Yes green, which the platform-native button-box order
    # wouldn't do.
    button_row = QWidget(body)
    row_layout = QHBoxLayout(button_row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(8)
    row_layout.addStretch(1)

    _base_btn_qss = (
        f"QPushButton {{"
        f"  background-color: {_INDIGO_MSG_BTN};"
        f"  color: {_INDIGO_FG};"
        f"  border: 1px solid {_INDIGO_MSG_BTN_BORDER};"
        f"  padding: 6px 16px;"
        f"  min-width: 76px;"
        f"  border-radius: 4px;"
        f"  font-size: 13px;"
        f"}}"
        f"QPushButton:hover {{"
        f"  background-color: {_INDIGO_MSG_BTN_HOVER};"
        f"}}"
    )
    _yes_btn_qss = (
        f"QPushButton {{"
        f"  background-color: {_INDIGO_ACCENT};"
        f"  color: white;"
        f"  border: 1px solid {_INDIGO_ACCENT};"
        f"  padding: 6px 16px;"
        f"  min-width: 76px;"
        f"  border-radius: 4px;"
        f"  font-size: 13px;"
        f"  font-weight: 600;"
        f"}}"
        f"QPushButton:hover {{"
        f"  background-color: #4CB89F;"
        f"}}"
    )

    # Order: No / Cancel / Close on the LEFT, Yes / OK / Accept on the
    # RIGHT (Windows-preferred "safe on left, action on right"). We
    # keep the sole primary green style on Yes/OK.
    button_specs = []  # (label, QMessageBox.StandardButton, is_primary)
    flags = int(buttons)
    if flags & int(QMessageBox.No):
        button_specs.append(("No", QMessageBox.No, False))
    if flags & int(QMessageBox.Cancel):
        button_specs.append(("Cancel", QMessageBox.Cancel, False))
    if flags & int(QMessageBox.Close):
        button_specs.append(("Close", QMessageBox.Close, False))
    if flags & int(QMessageBox.Ok):
        button_specs.append(("OK", QMessageBox.Ok, True))
    if flags & int(QMessageBox.Yes):
        button_specs.append(("Yes", QMessageBox.Yes, True))

    result_holder = {"button": QMessageBox.Cancel}

    def _make_click_handler(chosen_btn: "QMessageBox.StandardButton"):
        def _fire():
            result_holder["button"] = chosen_btn
            dialog.accept()
        return _fire

    for label, msg_btn, is_primary in button_specs:
        pbtn = QPushButton(label, button_row)
        pbtn.setCursor(Qt.PointingHandCursor)
        pbtn.setStyleSheet(_yes_btn_qss if is_primary else _base_btn_qss)
        if default_button == msg_btn:
            pbtn.setDefault(True)
            pbtn.setAutoDefault(True)
        pbtn.clicked.connect(_make_click_handler(msg_btn))
        row_layout.addWidget(pbtn)

    body_layout.addWidget(button_row, 1, 0, 1, 2, Qt.AlignRight)

    outer.addWidget(body)

    # Esc / X-close → treat as No if present, else Cancel, else Close.
    def _on_reject_or_close():
        if any(spec[1] == QMessageBox.No for spec in button_specs):
            result_holder["button"] = QMessageBox.No
        elif any(spec[1] == QMessageBox.Cancel for spec in button_specs):
            result_holder["button"] = QMessageBox.Cancel
        elif any(spec[1] == QMessageBox.Close for spec in button_specs):
            result_holder["button"] = QMessageBox.Close
        dialog.accept()
    dialog.rejected.connect(_on_reject_or_close)

    dialog.exec()
    return result_holder["button"]


# Colors used by touchless_message_box body. Kept module-level so
# the QDialog stylesheet strings above stay readable and the Tier A
# body_container wrappers can pull the same surface color if they
# want to match visually.
_INDIGO_MSG_BG = "#111935"          # dark indigo body, matches app surface
_INDIGO_MSG_BTN = "#1F2D6B"         # button bg, matches title bar
_INDIGO_MSG_BTN_BORDER = "#2A3B85"  # slightly lighter for definition
_INDIGO_MSG_BTN_HOVER = "#2A3B85"
_INDIGO_ACCENT = "#3A9D8B"          # brand green for default button


def _apply_dwm_caption_color(window: QWidget) -> None:
    """Call DwmSetWindowAttribute for caption + text color.

    On Windows 11 22000+ the caption + text colour attrs (35 / 36)
    land the Touchless indigo bar directly. On Windows 10 those two
    attrs are silently rejected but DWMWA_USE_IMMERSIVE_DARK_MODE
    (attribute 20 on 20H1+, attribute 19 earlier) forces a dark
    caption which is much closer to the app theme than the OS
    default white. We try all three: the wrong ones no-op on their
    respective OS versions.
    """
    try:
        hwnd = int(window.winId())
    except Exception:
        return
    if hwnd == 0:
        return
    try:
        dwmapi = ctypes.WinDLL("dwmapi")
    except OSError:
        return
    # v1.1.7 r35: force dark caption first via IMMERSIVE_DARK_MODE.
    # On Win10 this is the ONLY attribute that changes the caption
    # colour — 35/36 return E_INVALIDARG. On Win11 setting it is
    # harmless; 35/36 below override with the actual indigo hex so
    # both OS versions land dark, with Win11 getting the exact
    # Touchless colour and Win10 getting a matching dark caption.
    for dark_attr in (20, 19):
        try:
            dark = ctypes.c_int(1)
            dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(hwnd),
                ctypes.c_uint(dark_attr),
                ctypes.byref(dark),
                ctypes.c_uint(ctypes.sizeof(dark)),
            )
        except Exception:
            continue
    for attr, value in (
        (_DWMWA_CAPTION_COLOR, _CAPTION_COLORREF),
        (_DWMWA_TEXT_COLOR, _TEXT_COLORREF),
    ):
        try:
            colorref = ctypes.c_int(value)
            dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(hwnd),
                ctypes.c_uint(attr),
                ctypes.byref(colorref),
                ctypes.c_uint(ctypes.sizeof(colorref)),
            )
        except Exception:
            # Pre-Win11 returns an error HRESULT — that's fine, the
            # title bar just keeps its default color.
            continue


# ---------------------------------------------------------------
# Frameless indigo title bar for dialogs that MUST look the same
# on Win10 and Win11.
#
# apply_touchless_chrome above can only paint the caption on
# Win11 22000+ because DWMWA_CAPTION_COLOR / _TEXT_COLOR are
# Win11-only. On Win10 the best we get is a dark caption via
# DWMWA_USE_IMMERSIVE_DARK_MODE — never Touchless indigo — and
# the singleShot(0) defer races with the first native paint so
# users report a white bar. For the small handful of dialogs
# where visual identity matters (start / walk-through / privacy
# prompts) we go frameless and draw our OWN title bar so every
# platform lands on the exact same #1F2D6B.
#
# Scope note: everything else (settings, wizard, recorder,
# sandbox, Spotify/Discord setup, phone-camera, tutorial,
# dynamic recorder, drawing chooser, every QMessageBox) is
# UNTOUCHED — those keep the native title bar via
# apply_touchless_chrome.
# ---------------------------------------------------------------
_INDIGO_BG = "#1F2D6B"
_INDIGO_FG = "#E5F6FF"
_INDIGO_CLOSE_HOVER = "#E81123"


class _IndigoTitleBar(QFrame):
    """Small custom title bar for frameless Touchless dialogs.

    Renders app-icon + title + close (X) on a solid Touchless-
    indigo strip, and lets the user drag the parent dialog by
    pressing anywhere on the bar (except the close button).
    Matches main_window.py TitleBar's drag pattern (offset =
    globalPos - dialog.frameGeometry().topLeft()); on a frameless
    dialog frameGeometry() == geometry() so the offset stays
    stable across the whole drag.
    """

    def __init__(self, dialog: QWidget, title_text: str) -> None:
        super().__init__(dialog)
        self._dialog = dialog
        self._drag_offset: Optional["QPoint"] = None  # noqa: F821
        self.setObjectName("indigoTitleBar")
        self.setFixedHeight(32)
        # r51: was 46x32 (Windows Explorer title-bar spec). On the
        # narrow indigo popups (permission modal, tutorial prompt,
        # confirm dialogs) the wide button made the red hover rect
        # extend well past where the X glyph is drawn (glyph is
        # ~12px). Iterations 46x32 -> 32x32 -> 24x24 all still
        # showed too much red-padding around the X per user's
        # sketch. Now 18x18 with the glyph font bumped to 14px so
        # the X visually fills the box — the red hover rect hugs
        # the X the way VS Code / Chrome tab-close buttons do.
        # Flush-top-right alignment puts it right into the corner
        # instead of vertically centered in the 32-tall bar.
        # Prefer the em-box-sized, baseline-centered ChromeClose glyph
        # (U+E8BB) from Segoe MDL2 Assets / Segoe Fluent Icons — the
        # exact icon Explorer / Notepad / Settings render. On hosts
        # where neither icon font is installed (Wine, LTSC minimal,
        # some N-editions) U+E8BB is a Private-Use codepoint with no
        # fallback and would draw tofu, so we detect at construction
        # time and fall back to U+00D7 MULTIPLICATION SIGN in Segoe
        # UI / Arial — smaller and math-baseline-high vs the icon
        # font, but guaranteed to render as an X on every install.
        _families = QFontDatabase.families()
        if (
            "Segoe Fluent Icons" in _families
            or "Segoe MDL2 Assets" in _families
        ):
            close_glyph = ""
            close_font_family = "'Segoe Fluent Icons','Segoe MDL2 Assets'"
            # 12px ~= Windows caption-glyph 10pt at 96 DPI — the size
            # Explorer's close button renders at. Restored from a
            # brief r51 experiment that bumped this to 13px for a
            # snug 18x18 hover box; final r51 spec keeps the standard
            # Windows glyph size and shrinks only the box width.
            close_font_size = "12px"
        else:
            close_glyph = "×"
            close_font_family = "'Segoe UI','Arial'"
            close_font_size = "14px"
        self.setStyleSheet(
            f"QFrame#indigoTitleBar {{ background-color: {_INDIGO_BG}; }}"
            f"QLabel#indigoTitleText {{"
            f"  color: {_INDIGO_FG};"
            f"  font-size: 13px;"
            f"  font-weight: 600;"
            f"  background: transparent;"
            f"}}"
            f"QPushButton#indigoCloseBtn {{"
            f"  background: transparent;"
            f"  color: {_INDIGO_FG};"
            f"  border: none;"
            f"  border-radius: 0px;"
            f"  font-family: {close_font_family};"
            f"  font-size: {close_font_size};"
            f"  font-weight: 400;"
            f"  padding: 0px;"
            f"  margin: 0px;"
            f"  min-width: 32px;"
            f"  max-width: 32px;"
            f"  min-height: 32px;"
            f"  max-height: 32px;"
            f"  text-align: center;"
            f"}}"
            f"QPushButton#indigoCloseBtn:hover {{"
            f"  background: {_INDIGO_CLOSE_HOVER};"
            f"  color: #FFFFFF;"
            f"  border-radius: 0px;"
            f"}}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(6)
        icon = QApplication.windowIcon()
        if not icon.isNull():
            icon_label = QLabel(self)
            icon_label.setPixmap(icon.pixmap(16, 16))
            icon_label.setFixedSize(20, 20)
            icon_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            layout.addWidget(icon_label, 0, Qt.AlignVCenter)
        title_label = QLabel(title_text, self)
        title_label.setObjectName("indigoTitleText")
        title_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(title_label, 1, Qt.AlignVCenter | Qt.AlignLeft)
        # Glyph resolved above based on installed fonts: U+E8BB
        # (ChromeClose from Segoe MDL2 Assets / Segoe Fluent Icons)
        # on shipping Win10 1809+ / Win11, U+00D7 (MULTIPLICATION
        # SIGN in Segoe UI / Arial) on hosts missing both icon fonts.
        # r51: 32x32 square (was 46x32 Windows-Explorer rectangle).
        # Same height as the title bar (setFixedHeight(32) above),
        # flush right, X centered inside. Per user: square whose
        # dimensions match the title bar's vertical length, X in
        # the middle. Only the excess horizontal padding of the old
        # 46-wide button is gone; the standard Windows glyph size
        # (12px icon-font / 14px ASCII fallback) stays.
        close_btn = QPushButton(close_glyph, self)
        close_btn.setObjectName("indigoCloseBtn")
        close_btn.setFixedSize(32, 32)
        close_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        close_btn.setFlat(True)
        close_btn.setToolTip("Close")
        close_btn.setCursor(Qt.ArrowCursor)
        # r51: was dialog.reject — only works on QDialog. Several
        # QMainWindow subclasses (TutorialWindow, DynamicGesture-
        # RecorderWindow, RecordingWindow, SandboxWindow) also use
        # this custom title bar; close() works on both QDialog and
        # QMainWindow while reject() would AttributeError on the
        # QMainWindow branch.
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn, 0, Qt.AlignVCenter | Qt.AlignRight)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint()
                - self._dialog.frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_offset is not None and (event.buttons() & Qt.LeftButton):
            self._dialog.move(
                event.globalPosition().toPoint() - self._drag_offset
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._drag_offset = None
        super().mouseReleaseEvent(event)


def apply_indigo_title_bar(dialog: QWidget, title_text: str) -> "_IndigoTitleBar":
    """Return a Touchless-indigo custom title bar for `dialog`.

    Side effects:
      * Sets `Qt.FramelessWindowHint` on `dialog` (must be called
        before the dialog is shown).
      * Sets the window icon to the app icon.

    Caller responsibility: insert the returned QFrame at row 0 of
    the dialog's top-level layout, zero the layout's contents
    margins, and wrap the rest of the content in a body_container
    widget with the ORIGINAL padding so the bar stays flush with
    the dialog edges while the body keeps its own inner padding.
    See call sites in main_window.py (WalkthroughStartDialog /
    StartTutorialDialog / TouchlessPrivacyDialog) for the exact
    wrapper pattern.

    Works identically on Win10, Win11, macOS, and Linux — all
    platforms get the exact #1F2D6B. Safe to leave
    apply_touchless_chrome on the same dialog: its deferred DWM
    calls silently no-op on the frameless HWND (attributes 20 /
    35 / 36 all fail) and its setWindowIcon call is idempotent
    with the one this helper makes.
    """
    icon = QApplication.windowIcon()
    if not icon.isNull():
        try:
            dialog.setWindowIcon(icon)
        except Exception:
            pass
    dialog.setWindowFlag(Qt.FramelessWindowHint, True)
    return _IndigoTitleBar(dialog, title_text)


def install_indigo_chrome(dialog: QWidget, title_text: str) -> QWidget:
    """r51 one-shot converter: give `dialog` a frameless indigo
    title bar and return a `body` QWidget that the caller should
    use as the parent for all their existing content.

    Typical conversion pattern (Tier A) — replace three lines:

        # BEFORE
        layout = QVBoxLayout(self)
        layout.addWidget(...)
        apply_touchless_chrome(self)

        # AFTER
        body = install_indigo_chrome(self, "Dialog Title")
        layout = QVBoxLayout(body)
        layout.addWidget(...)

    This helper wraps the caller's whole content in a QWidget below
    the indigo title bar, so the caller's original QVBoxLayout /
    QGridLayout / whatever stays byte-identical — only the parent
    changes from `self` to `body`. Works on QDialog AND QMainWindow
    subclasses (the close button connects to `dialog.close`, which
    both classes have).

    Renders identically on Windows 10 AND 11 (no DWM dependency).
    """
    dialog.setWindowFlag(Qt.FramelessWindowHint, True)
    icon = QApplication.windowIcon()
    if not icon.isNull():
        try:
            dialog.setWindowIcon(icon)
        except Exception:
            pass
    outer = QVBoxLayout(dialog)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    outer.addWidget(_IndigoTitleBar(dialog, title_text))
    body = QWidget(dialog)
    body.setObjectName("indigoChromeBody")
    body.setAttribute(Qt.WA_StyledBackground, True)
    body.setAutoFillBackground(True)
    body.setStyleSheet(
        f"QWidget#indigoChromeBody {{ background-color: {_INDIGO_MSG_BG}; }}"
    )
    outer.addWidget(body)
    return body


def install_indigo_chrome_main_window(mw: QWidget, title_text: str) -> QWidget:
    """QMainWindow variant of install_indigo_chrome. Call AFTER the
    caller has assembled their central widget content. Returns the
    central widget's new parent — but for typical usage the caller
    doesn't need to touch the return value, they just call this once
    at the end of __init__.

    Conversion pattern:

        # BEFORE
        central = QWidget()
        # ... build central content ...
        self.setCentralWidget(central)
        apply_touchless_chrome(self)

        # AFTER
        central = QWidget()
        # ... build central content unchanged ...
        self.setCentralWidget(central)
        install_indigo_chrome_main_window(self, "Window title")
    """
    mw.setWindowFlag(Qt.FramelessWindowHint, True)
    icon = QApplication.windowIcon()
    if not icon.isNull():
        try:
            mw.setWindowIcon(icon)
        except Exception:
            pass
    existing_central = None
    try:
        existing_central = mw.centralWidget()
    except Exception:
        pass
    container = QWidget()
    outer = QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    outer.addWidget(_IndigoTitleBar(mw, title_text))
    if existing_central is not None:
        outer.addWidget(existing_central)
    try:
        mw.setCentralWidget(container)
    except Exception:
        pass
    return container

"""Guided wizard for setting up a user's own Spotify Developer client_id.

Why this exists
---------------
Spotify caps shared developer apps at 5 unique users in Development Mode
and removed the indie-friendly Extended Quota Mode in 2024. The remaining
"Partner" extension requires 250k+ Monthly Active Users + a registered
business, which a free indie desktop app cannot meet.

The only way to ship Spotify integration to more than 5 users is to have
each user create their own free Spotify Developer app (cap: 5 of THEIR own
users — but they only have one, themselves). The user supplies the
resulting Client ID to Touchless and Touchless uses it for OAuth instead
of the embedded default. PKCE means no client_secret has to change hands.

Fully auto-filling Spotify's Developer Dashboard isn't possible without
browser automation (which would ship Chromium + violate Spotify's TOS for
automated dashboard interaction). What IS possible is a guided wizard:
each value the user needs to type is shown next to a Copy button, the
Dashboard page is opened in their browser at the right step, and they
paste their resulting Client ID back into Touchless when done.

Public API
----------
Open the wizard from settings:

    from .spotify_setup_wizard import SpotifySetupWizard
    SpotifySetupWizard(config, parent=main_window).exec()

The wizard writes `config.spotify_client_id` and calls save_config on
success. Caller should refresh anything that holds a SpotifyController
afterwards (engine controller pickup happens automatically on the next
auth attempt because _load_credentials reads config fresh each call).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QSize, QTimer, QUrl
from PySide6.QtGui import QClipboard, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...config.app_config import save_config
from .window_chrome import apply_touchless_chrome


class _PageStack(QStackedWidget):
    """QStackedWidget that reports the CURRENT page's sizeHint instead
    of the stock max-of-all behaviour. Lets the wizard dialog
    self.adjustSize() to hug just the visible page's content height,
    so the intro / collect pages stay compact and only page 2
    (paste-values, taller because of the URI list) grows the window."""

    def sizeHint(self) -> QSize:  # type: ignore[override]
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        w = self.currentWidget()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()


class _TightWrapLabel(QLabel):
    """QLabel that reports its actual wrapped height (heightForWidth)
    as both sizeHint and minimumSizeHint.

    Why: a default QLabel with setWordWrap(True) reports
    sizeHint().height() = single-line height but
    minimumSizeHint().height() = the worst-case wrapped height
    (one word per line). Qt treats minimumSizeHint as a hard floor,
    so the label widget gets allocated that worst-case height even
    when the actual rendered text only needs a few lines — which on
    the wizard's intro page meant the dialog couldn't be shrunk
    smaller than ~700 px tall regardless of setFixedSize.
    Overriding both hints to heightForWidth(currentWidth) returns
    the actual rendered pixel height instead."""

    def sizeHint(self) -> QSize:  # type: ignore[override]
        base = super().sizeHint()
        try:
            w = self.width() if self.width() > 0 else base.width()
            if self.wordWrap() and w > 0:
                h = self.heightForWidth(w)
                if h > 0:
                    return QSize(base.width(), h)
        except Exception:
            pass
        return base

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return self.sizeHint()


_SPOTIFY_DASHBOARD_URL = "https://developer.spotify.com/dashboard/applications"
_SPOTIFY_CREATE_APP_URL = "https://developer.spotify.com/dashboard/create"
_TOUCHLESS_SETUP_HELP_URL = "https://touchless-control.pages.dev/spotify-setup.html"

# Five callback URIs the user pastes into the Spotify Dashboard's
# "Redirect URIs" field. Matches the port-fallback list in
# spotify_controller.authorize_full_scopes so any of these can be
# bound at runtime.
_REDIRECT_URIS = (
    "http://127.0.0.1:5000/callback",
    "http://127.0.0.1:5001/callback",
    "http://127.0.0.1:5002/callback",
    "http://127.0.0.1:5003/callback",
    "http://127.0.0.1:5004/callback",
)


class _CopyRow(QWidget):
    """One row inside the wizard: label + read-only value + Copy button."""

    def __init__(self, label_text: str, value_text: str, accent: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._value_text = value_text
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        label = QLabel(label_text)
        label.setMinimumWidth(110)
        label.setStyleSheet("color: #B4D7FF; font-size: 12px; font-weight: 700;")
        layout.addWidget(label)

        value = QLineEdit(value_text)
        value.setReadOnly(True)
        value.setStyleSheet(
            "QLineEdit {"
            " background: rgba(255,255,255,0.04);"
            " color: #E5F6FF;"
            " border: 1px solid rgba(255,255,255,0.18);"
            " border-radius: 8px;"
            " padding: 6px 10px;"
            " font-size: 12px;"
            "}"
        )
        value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(value, 1)

        copy_btn = QPushButton("Copy")
        copy_btn.setCursor(Qt.PointingHandCursor)
        copy_btn.setMinimumWidth(70)
        # NOTE: no `filter:` CSS — Qt's QSS doesn't support it and
        # logs an "Unknown property filter" warning for every label/
        # button using it. Hover gets a subtle outline change instead.
        copy_btn.setStyleSheet(
            "QPushButton {"
            f" background: {accent};"
            " color: #001B24;"
            " border: 1px solid transparent;"
            " border-radius: 8px;"
            " padding: 7px 14px;"
            " font-weight: 800;"
            "}"
            "QPushButton:hover { border: 1px solid rgba(255,255,255,0.35); }"
        )
        copy_btn.clicked.connect(self._on_copy)
        self._copy_btn = copy_btn
        layout.addWidget(copy_btn)

    def _on_copy(self) -> None:
        clip = QGuiApplication.clipboard()
        if clip is None:
            return
        clip.setText(self._value_text, QClipboard.Clipboard)
        # Flash the button text so the user gets visible feedback.
        self._copy_btn.setText("Copied!")
        try:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(900, lambda: self._copy_btn.setText("Copy"))
        except Exception:
            pass


class SpotifySetupWizard(QDialog):
    """Three-page guided wizard.

    Page 1 — explains what we're doing and why, opens the Dashboard.
    Page 2 — shows every value the user needs to paste, with Copy buttons.
    Page 3 — collects the resulting Client ID, saves to config.
    """

    def __init__(self, config, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._accent = str(getattr(config, "accent_color", "") or "#1DE9B6")
        self._surface = str(getattr(config, "surface_color", "") or "#0F172A")
        self._text_color = str(getattr(config, "text_color", "") or "#E5F6FF")
        self.setWindowTitle("Set up your own Spotify app")
        self.setModal(True)
        # Explicit per-page dialog sizes. Qt's layout-driven sizing
        # (adjustSize, SetFixedSize) was leaving the dialog far taller
        # than the visible page needed — word-wrap QLabels report a
        # tall minimumSizeHint (one-word-per-line height) which
        # propagates up through QStackedWidget into the dialog's own
        # minimum size, regardless of how the layout signals are
        # configured. Hardcoded setFixedSize() per page sidesteps all
        # of that: the dialog IS this size on each page, end of story.
        self._page_sizes = {
            0: (560, 340),   # intro — trimmed body (4 lines + spacer)
            1: (680, 900),   # paste values — Step 1 fallback nav, URI list, Steps 2+3
            2: (560, 400),   # collect Client ID — title + body + input + tip
        }
        # Touchless-themed title bar (Win11 DWM caption color) + the
        # surface color from app config so the wizard matches the rest
        # of the settings UI.
        apply_touchless_chrome(self)
        self.setStyleSheet(
            f"QDialog {{ background: {self._surface}; }}"
            f"QLabel {{ color: {self._text_color}; font-size: 13px; }}"
            f"QLabel#wizardTitle {{ font-size: 22px; font-weight: 800; color: {self._accent}; }}"
            # Qt QSS doesn't support `line-height`; QLabel uses the
            # font's default leading and wraps via setWordWrap(True).
            "QPushButton#wizardPrimary {"
            f" background: {self._accent};"
            " color: #001B24;"
            " border: 1px solid transparent;"
            " border-radius: 10px;"
            " padding: 10px 18px;"
            " font-weight: 800;"
            " min-width: 120px;"
            "}"
            # Hover uses outline change because Qt QSS doesn't support
            # CSS `filter:` — `filter: brightness(...)` logs an
            # "Unknown property filter" warning on every hover paint.
            "QPushButton#wizardPrimary:hover { border: 1px solid rgba(255,255,255,0.40); }"
            "QPushButton#wizardSecondary {"
            " background: transparent;"
            " color: #B4D7FF;"
            " border: 1px solid rgba(255,255,255,0.20);"
            " border-radius: 10px;"
            " padding: 10px 18px;"
            " font-weight: 700;"
            " min-width: 100px;"
            "}"
            "QPushButton#wizardSecondary:hover { color: #E5F6FF; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        # Pages built once and stashed. We swap them in/out of a
        # single QFrame container instead of using QStackedWidget —
        # QStackedWidget's internal layout aggregates ALL children's
        # minimum sizes regardless of which is visible, which was
        # forcing the dialog to be as tall as the TALLEST page
        # (page 2's URI list) even when showing page 1. With a manual
        # swap, only the current page contributes to layout sizing.
        self._pages: list[QWidget] = [
            self._build_page_intro(),
            self._build_page_values(),
            self._build_page_collect(),
        ]
        self._current_page_index = 0
        self._content_frame = QFrame()
        self._content_frame.setObjectName("wizardContent")
        self._content_frame.setStyleSheet("QFrame#wizardContent { background: transparent; }")
        self._content_layout = QVBoxLayout(self._content_frame)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        # Mount the first page now so it actually has a parent before
        # the dialog is shown.
        self._content_layout.addWidget(self._pages[0])
        root.addWidget(self._content_frame, 0, Qt.AlignTop)
        # Slack absorber between the content area and the nav row.
        # Any vertical room left after the page sits at its natural
        # sizeHint goes here, so the nav row stays pinned at the
        # bottom and content stays pinned at the top.
        root.addStretch(1)

        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(10)

        self._back_btn = QPushButton("Back")
        self._back_btn.setObjectName("wizardSecondary")
        self._back_btn.setCursor(Qt.PointingHandCursor)
        self._back_btn.clicked.connect(self._on_back)
        nav_row.addWidget(self._back_btn)

        nav_row.addStretch(1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("wizardSecondary")
        self._cancel_btn.setCursor(Qt.PointingHandCursor)
        self._cancel_btn.clicked.connect(self.reject)
        nav_row.addWidget(self._cancel_btn)

        self._next_btn = QPushButton("Next")
        self._next_btn.setObjectName("wizardPrimary")
        self._next_btn.setCursor(Qt.PointingHandCursor)
        self._next_btn.clicked.connect(self._on_next)
        nav_row.addWidget(self._next_btn)

        root.addLayout(nav_row)

        self._refresh_nav()
        # Apply initial-page fixed size so the dialog opens compact.
        self._apply_fixed_size_for_page(self._current_page_index)

    def _apply_fixed_size_for_page(self, page_index: int) -> None:
        """Lock the dialog to this page's target size."""
        target = self._page_sizes.get(page_index)
        if target is None:
            return
        w, h = target
        self.setFixedSize(w, h)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        """Re-apply the page's fixed size every time the dialog is
        shown. Qt's initial show logic computes its own preferred
        size from the layout's sizeHint chain BEFORE honouring
        setFixedSize set in __init__, which is why earlier attempts
        left a tall empty band on page 1 — the dialog grabbed Qt's
        computed sizeHint at show time and never shrank again.
        Calling setFixedSize after the show event runs forces Qt to
        re-clamp the geometry to our target.

        Also re-applies the Touchless caption-bar color: the DWM call
        is most reliable when the HWND is already created (which it
        is by the time showEvent fires). The __init__-time call works
        too but can occasionally miss on first show — calling here
        belt-and-braces guarantees the indigo title bar."""
        super().showEvent(event)
        QTimer.singleShot(0, lambda: self._apply_fixed_size_for_page(self._current_page_index))
        # Re-apply title-bar theme now that HWND is definitely valid.
        try:
            apply_touchless_chrome(self)
        except Exception:
            pass

    # ---- page builders ------------------------------------------------

    def _build_page_intro(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        title = QLabel("Connect your Spotify account")
        title.setObjectName("wizardTitle")
        # Top-align the text within the label widget. Default QLabel
        # is AlignLeft | AlignVCenter, which paints text in the middle
        # of the widget when the widget gets stretched — that's what
        # made the title float in the middle of empty space on
        # page 1. AlignTop keeps text at the widget's top edge no
        # matter what size the widget ends up.
        title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(title)

        body = _TightWrapLabel(
            "Three quick steps:<br>"
            "<b>1.</b> Open the Spotify Developer Dashboard.<br>"
            "<b>2.</b> Paste the values shown on the next page.<br>"
            "<b>3.</b> Paste the resulting Client ID back here.<br><br>"
            "Your Spotify password never leaves Spotify."
        )
        body.setObjectName("wizardBody")
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(body)

        help_btn = QPushButton("Need more help? Open the setup guide on the Touchless website")
        help_btn.setObjectName("wizardSecondary")
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(_TOUCHLESS_SETUP_HELP_URL)))
        lay.addWidget(help_btn, 0, Qt.AlignLeft)
        # No trailing addStretch — under SetFixedSize we want each page's
        # sizeHint to be its actual content height so the dialog hugs
        # the visible page.
        return page

    def _build_page_values(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        title = QLabel("Paste these into the Spotify Dashboard")
        title.setObjectName("wizardTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(title)

        # Step 1: open the Dashboard.
        step1 = _TightWrapLabel(
            "<b>Step 1.</b>  Click the button below to open Spotify's "
            "Create-App page in your browser.<br><br>"
            "If you land on the marketing page instead (not signed in "
            "yet): click the <b>≡</b> menu top-right → <b>Log in</b> → "
            "<b>Dashboard</b> → <b>Create app</b>."
        )
        step1.setObjectName("wizardBody")
        step1.setWordWrap(True)
        step1.setTextFormat(Qt.RichText)
        step1.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(step1)

        open_btn = QPushButton("Open Spotify Create-App page in browser")
        open_btn.setObjectName("wizardPrimary")
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(self._open_create_app_page)
        lay.addWidget(open_btn, 0, Qt.AlignLeft)

        # Spacer line
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background: rgba(255,255,255,0.08); max-height: 1px;")
        lay.addWidget(sep)

        # Step 2: paste the values.
        step2 = _TightWrapLabel(
            "<b>Step 2.</b>  Paste the following values into the form. "
            "App name can be anything you want — \"Touchless Personal\" "
            "is just a suggestion."
        )
        step2.setObjectName("wizardBody")
        step2.setWordWrap(True)
        step2.setTextFormat(Qt.RichText)
        step2.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(step2)

        lay.addWidget(_CopyRow("App name", "Touchless Personal", self._accent))
        lay.addWidget(_CopyRow("Description", "Personal Spotify control via Touchless", self._accent))
        lay.addWidget(_CopyRow("Website", "https://touchless-control.pages.dev/", self._accent))

        # Redirect URIs section. Explain what a callback is + that all
        # 5 are recommended (port-fallback) but not strictly required.
        redirect_intro = _TightWrapLabel(
            "<b>Redirect URIs (callbacks)</b>  —  Spotify sends your "
            "browser to one of these local addresses when authorisation "
            "finishes. Touchless tries ports 5000 → 5004 in order in "
            "case 5000 is busy on your machine, so registering all five "
            "is recommended. At minimum, register 5000. Click <b>Add</b> "
            "in the Dashboard after each one:"
        )
        redirect_intro.setStyleSheet("color: #B4D7FF; font-size: 12px; font-weight: 600;")
        redirect_intro.setWordWrap(True)
        redirect_intro.setTextFormat(Qt.RichText)
        redirect_intro.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(redirect_intro)

        for uri in _REDIRECT_URIS:
            lay.addWidget(_CopyRow("", uri, self._accent))

        # Web API + app type guidance.
        api_label = _TightWrapLabel(
            "<b>Step 3.</b>  Under <b>\"Which API/SDKs are you planning "
            "to use?\"</b>, check the box next to <b>Web API</b>. Don't "
            "tick any of the others — Touchless only uses the Web API to "
            "send playback commands (play, pause, skip, volume) to your "
            "own Spotify account.<br><br>"
            "<b>App type:</b> <code>Desktop App</code>.<br>"
            "<b>Commercial application:</b> <code>No</code>."
        )
        api_label.setObjectName("wizardBody")
        api_label.setWordWrap(True)
        api_label.setTextFormat(Qt.RichText)
        api_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(api_label)
        return page

    def _build_page_collect(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        title = QLabel("Paste your Client ID")
        title.setObjectName("wizardTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(title)

        body = _TightWrapLabel(
            "After you click Save on the Dashboard, your app's page will "
            "show a <b>Client ID</b>. Click the Copy icon next to it, then "
            "paste it below.<br><br>"
            "The Client ID is a 32-character string that looks like:<br>"
            "<code style='background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px;'>7763a7c443604776b0060da01428686f</code>"
        )
        body.setObjectName("wizardBody")
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(body)

        self._client_id_input = QLineEdit()
        self._client_id_input.setPlaceholderText("32-character Client ID")
        self._client_id_input.setStyleSheet(
            "QLineEdit {"
            " background: rgba(255,255,255,0.05);"
            " color: #E5F6FF;"
            " border: 1px solid rgba(255,255,255,0.22);"
            " border-radius: 10px;"
            " padding: 12px 14px;"
            " font-size: 14px;"
            " font-family: 'Consolas', 'Menlo', monospace;"
            "}"
            "QLineEdit:focus {"
            f" border: 1px solid {self._accent};"
            "}"
        )
        # Preload from existing config so the user can edit / re-paste
        # without losing their previous value.
        existing = str(getattr(self._config, "spotify_client_id", "") or "")
        if existing:
            self._client_id_input.setText(existing)
        lay.addWidget(self._client_id_input)

        self._validation_msg = QLabel("")
        self._validation_msg.setWordWrap(True)
        self._validation_msg.setStyleSheet("color: #FFB86B; font-size: 12px;")
        lay.addWidget(self._validation_msg)

        info = _TightWrapLabel(
            "Tip: keep the Spotify app page open. If the OAuth callback "
            "step fails after we save this, you can come back here and "
            "double-check the Redirect URIs are all listed."
        )
        info.setStyleSheet("color: #98B8D6; font-size: 12px;")
        info.setWordWrap(True)
        lay.addWidget(info)
        return page

    # ---- nav handlers ------------------------------------------------

    def _refresh_nav(self) -> None:
        idx = self._current_page_index
        last = len(self._pages) - 1
        self._back_btn.setVisible(idx > 0)
        if idx == last:
            self._next_btn.setText("Finish")
        else:
            self._next_btn.setText("Next")

    def _swap_to_page(self, new_idx: int) -> None:
        """Detach current page widget, attach the target page widget,
        and resize the dialog. Manual swap (instead of QStackedWidget)
        is what finally lets setFixedSize work — only ONE page widget
        is in the layout at a time, so the layout's minimum size
        reflects only the visible page, not the union of all pages."""
        if new_idx == self._current_page_index:
            return
        if new_idx < 0 or new_idx >= len(self._pages):
            return
        old_widget = self._pages[self._current_page_index]
        new_widget = self._pages[new_idx]
        self._content_layout.removeWidget(old_widget)
        old_widget.setParent(None)
        old_widget.hide()
        self._content_layout.addWidget(new_widget)
        new_widget.show()
        self._current_page_index = new_idx
        self._refresh_nav()
        self._apply_fixed_size_for_page(new_idx)

    def _on_back(self) -> None:
        idx = self._current_page_index
        if idx <= 0:
            return
        self._swap_to_page(idx - 1)

    def _on_next(self) -> None:
        idx = self._current_page_index
        last = len(self._pages) - 1
        if idx < last:
            self._swap_to_page(idx + 1)
            return
        # Final page: validate + save.
        self._try_save_and_close()

    def _open_create_app_page(self) -> None:
        QDesktopServices.openUrl(QUrl(_SPOTIFY_CREATE_APP_URL))

    def _try_save_and_close(self) -> None:
        raw = self._client_id_input.text().strip()
        if not raw:
            self._validation_msg.setText("Paste your Client ID before saving.")
            self._client_id_input.setFocus()
            return
        # Spotify client_ids are 32 lowercase-hex characters. Don't reject
        # anything that doesn't match exactly (Spotify may change format),
        # but warn if it looks obviously wrong.
        sanitized = raw.replace(" ", "")
        if len(sanitized) < 16 or " " in raw:
            self._validation_msg.setText(
                "That doesn't look like a Spotify Client ID — it should be a "
                "single 32-character string with no spaces. Double-check what "
                "you copied from the Dashboard."
            )
            self._client_id_input.setFocus()
            return
        try:
            self._config.spotify_client_id = sanitized
            save_config(self._config)
        except Exception as exc:
            self._validation_msg.setText(
                f"Couldn't save: {exc}. Check the app has write access "
                "to your settings folder."
            )
            return
        self.accept()


# Author: Konstantin Markov

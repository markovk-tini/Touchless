"""Guided wizard for setting up a user's own Discord Developer app.

Why this exists
---------------
Discord's `rpc` OAuth scope (which Touchless uses to mute / deafen /
read voice state on the local Discord desktop client) is restricted:
only the OWNER of the registered dev-portal app can authorise non-owner
accounts to use it. Public-distribution approval is a one-time review
process from Discord, but Discord has been actively winding the `rpc`
scope down since ~2022 — new approvals are rare and the dedicated
self-serve form has been removed from the developer support portal.

The only way to ship Discord integration to more than one user is to
have each user create their own free Discord Developer app. Because
THEY are the owner of THEIR app, the `rpc` scope works on day one
for THEIR account, with no approval needed. The user supplies the
resulting Client ID + Client Secret to Touchless via this wizard;
Touchless uses them for the OAuth token-exchange step in
[discord_controller.py](../../debug/discord_controller.py).

Unlike Spotify (which uses PKCE and only needs a public Client ID),
Discord's RPC OAuth flow requires the Client Secret for the
token-exchange POST — there's no PKCE flow for the `rpc` scope. So
we collect both values in the wizard.

Public API
----------
Open the wizard from settings:

    from .discord_setup_wizard import DiscordSetupWizard
    DiscordSetupWizard(config, parent=main_window).exec()

The wizard writes `config.discord_client_id` +
`config.discord_client_secret` and calls save_config on success.
Caller should refresh anything that holds a DiscordController
afterwards (engine controller pickup happens automatically on the
next auth attempt because _load_credentials reads config fresh).
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
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...config.app_config import save_config
from .window_chrome import apply_touchless_chrome


class _TightWrapLabel(QLabel):
    """QLabel that reports its actual wrapped height (heightForWidth)
    as both sizeHint and minimumSizeHint.

    Why: a default QLabel with setWordWrap(True) reports
    sizeHint().height() = single-line height but
    minimumSizeHint().height() = the worst-case wrapped height
    (one word per line). Qt treats minimumSizeHint as a hard floor,
    so the label widget gets allocated that worst-case height even
    when the actual rendered text only needs a few lines — which
    breaks per-page setFixedSize on the wizard. Overriding both
    hints to heightForWidth(currentWidth) returns the actual
    rendered pixel height instead.

    Duplicated from spotify_setup_wizard.py (same name, same body).
    Worth keeping per-file rather than introducing a shared widgets
    module just for two small classes — both wizards are self-
    contained UI files and the friction of an import-chain isn't
    paid back yet."""

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


_DISCORD_DEV_DASHBOARD_URL = "https://discord.com/developers/applications"
_TOUCHLESS_DISCORD_SETUP_HELP_URL = "https://touchless-control.pages.dev/discord.html"

# Discord-side redirect URI. Unlike Spotify's port-fallback list,
# the IPC auth flow doesn't actually open a browser callback —
# Discord just validates the URI server-side against the registered
# list. A single bare `http://127.0.0.1` entry is sufficient and
# matches what discord_controller's `_DEFAULT_DISCORD_REDIRECT_URI`
# uses for the token-exchange POST.
_REDIRECT_URI = "http://127.0.0.1"

# Discord Blurple. Overrides the global accent_color for THIS
# wizard so it's visually distinct from the Spotify wizard (which
# uses the green-ish app accent). Same idea as why discord.html on
# the website is purple-tinted instead of the default cyan.
_DISCORD_ACCENT = "#5865F2"


class _CopyRow(QWidget):
    """One row inside the wizard: label + read-only value + Copy button.

    Duplicated from spotify_setup_wizard.py (see _TightWrapLabel for
    the same reasoning)."""

    def __init__(
        self,
        label_text: str,
        value_text: str,
        accent: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._value_text = value_text
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        label = QLabel(label_text)
        label.setMinimumWidth(110)
        label.setStyleSheet("color: #B6BFFF; font-size: 12px; font-weight: 700;")
        layout.addWidget(label)

        value = QLineEdit(value_text)
        value.setReadOnly(True)
        value.setStyleSheet(
            "QLineEdit {"
            " background: rgba(255,255,255,0.04);"
            " color: #E5F0FF;"
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
        copy_btn.setStyleSheet(
            "QPushButton {"
            f" background: {accent};"
            " color: #FFFFFF;"
            " border: 1px solid transparent;"
            " border-radius: 8px;"
            " padding: 7px 14px;"
            " font-weight: 800;"
            "}"
            "QPushButton:hover { border: 1px solid rgba(255,255,255,0.45); }"
        )
        copy_btn.clicked.connect(self._on_copy)
        self._copy_btn = copy_btn
        layout.addWidget(copy_btn)

    def _on_copy(self) -> None:
        clip = QGuiApplication.clipboard()
        if clip is None:
            return
        clip.setText(self._value_text, QClipboard.Clipboard)
        self._copy_btn.setText("Copied!")
        try:
            QTimer.singleShot(900, lambda: self._copy_btn.setText("Copy"))
        except Exception:
            pass


class DiscordSetupWizard(QDialog):
    """Three-page guided wizard.

    Page 1 — explains what we're doing and why, link to the website
             setup guide.
    Page 2 — walks through the Discord Developer Portal steps with
             Copy rows for every value the user has to paste.
    Page 3 — collects the resulting Client ID + Client Secret,
             saves to config.
    """

    def __init__(self, config, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        # Discord wizard uses Blurple regardless of the app's global
        # accent so it's immediately recognisable as "the Discord
        # setup screen" vs "the Spotify setup screen".
        self._accent = _DISCORD_ACCENT
        self._surface = str(getattr(config, "surface_color", "") or "#0F172A")
        self._text_color = str(getattr(config, "text_color", "") or "#E5F0FF")
        self.setWindowTitle("Set up your own Discord app")
        self.setModal(True)
        # Per-page dialog sizes. See spotify_setup_wizard for the
        # full reasoning — short version: word-wrap QLabels report a
        # tall minimumSizeHint (one-word-per-line) that propagates up
        # through any layout/stack and inflates the dialog. Hardcoded
        # setFixedSize per page sidesteps it.
        self._page_sizes = {
            0: (560, 340),   # intro
            1: (640, 720),   # paste values — shorter than Spotify
                              # (single redirect URI, fewer steps)
            2: (580, 460),   # collect — two input fields
        }
        apply_touchless_chrome(self)
        self.setStyleSheet(
            f"QDialog {{ background: {self._surface}; }}"
            f"QLabel {{ color: {self._text_color}; font-size: 13px; }}"
            f"QLabel#wizardTitle {{ font-size: 22px; font-weight: 800; color: {self._accent}; }}"
            "QPushButton#wizardPrimary {"
            f" background: {self._accent};"
            " color: #FFFFFF;"
            " border: 1px solid transparent;"
            " border-radius: 10px;"
            " padding: 10px 18px;"
            " font-weight: 800;"
            " min-width: 120px;"
            "}"
            "QPushButton#wizardPrimary:hover { border: 1px solid rgba(255,255,255,0.45); }"
            "QPushButton#wizardSecondary {"
            " background: transparent;"
            " color: #B6BFFF;"
            " border: 1px solid rgba(255,255,255,0.20);"
            " border-radius: 10px;"
            " padding: 10px 18px;"
            " font-weight: 700;"
            " min-width: 100px;"
            "}"
            "QPushButton#wizardSecondary:hover { color: #FFFFFF; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        # Manual page swap (no QStackedWidget) so only the current page
        # contributes to layout sizing — same approach as the Spotify
        # wizard, see its _swap_to_page docstring for the why.
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
        self._content_layout.addWidget(self._pages[0])
        root.addWidget(self._content_frame, 0, Qt.AlignTop)
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
        self._apply_fixed_size_for_page(self._current_page_index)

    def _apply_fixed_size_for_page(self, page_index: int) -> None:
        target = self._page_sizes.get(page_index)
        if target is None:
            return
        w, h = target
        self.setFixedSize(w, h)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        super().showEvent(event)
        QTimer.singleShot(
            0, lambda: self._apply_fixed_size_for_page(self._current_page_index)
        )

    # ---- page builders ------------------------------------------------

    def _build_page_intro(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        title = QLabel("Connect your Discord account")
        title.setObjectName("wizardTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(title)

        body = _TightWrapLabel(
            "Three quick steps:<br>"
            "<b>1.</b> Open the Discord Developer Portal and create a free app.<br>"
            "<b>2.</b> Copy two values (Application ID + Client Secret).<br>"
            "<b>3.</b> Paste both back here.<br><br>"
            "Your Discord password never leaves Discord."
        )
        body.setObjectName("wizardBody")
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(body)

        help_btn = QPushButton("Need more help? Open the setup guide on the Touchless website")
        help_btn.setObjectName("wizardSecondary")
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(_TOUCHLESS_DISCORD_SETUP_HELP_URL))
        )
        lay.addWidget(help_btn, 0, Qt.AlignLeft)
        return page

    def _build_page_values(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        title = QLabel("Set up your Discord app")
        title.setObjectName("wizardTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(title)

        # Step 1: open the dev portal + create app
        step1 = _TightWrapLabel(
            "<b>Step 1.</b>  Click below to open the Discord Developer Portal "
            "in your browser. If a survey pops up (\"What brings you to the "
            "Developer Portal?\"), click <b>Skip</b>.<br><br>"
            "Then click <b>New Application</b> top-right, name it anything "
            "(\"Touchless Personal\" is just a suggestion), accept the Terms, "
            "and click <b>Create</b>."
        )
        step1.setObjectName("wizardBody")
        step1.setWordWrap(True)
        step1.setTextFormat(Qt.RichText)
        step1.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(step1)

        open_btn = QPushButton("Open Discord Developer Portal in browser")
        open_btn.setObjectName("wizardPrimary")
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(self._open_dev_portal_page)
        lay.addWidget(open_btn, 0, Qt.AlignLeft)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background: rgba(255,255,255,0.08); max-height: 1px;")
        lay.addWidget(sep)

        # Step 2: add redirect URI
        step2 = _TightWrapLabel(
            "<b>Step 2.</b>  In the left sidebar of your new app, click "
            "<b>OAuth2</b>. Scroll to <b>Redirects</b> → <b>Add Another</b> "
            "and paste the URI below. Click <b>Save Changes</b> at the bottom "
            "of the page when done."
        )
        step2.setObjectName("wizardBody")
        step2.setWordWrap(True)
        step2.setTextFormat(Qt.RichText)
        step2.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(step2)

        lay.addWidget(_CopyRow("Redirect URI", _REDIRECT_URI, self._accent))

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("background: rgba(255,255,255,0.08); max-height: 1px;")
        lay.addWidget(sep2)

        # Step 3: grab Application ID + reset Client Secret
        step3 = _TightWrapLabel(
            "<b>Step 3.</b>  Still on the OAuth2 page, find <b>Client "
            "Information</b> at the top. You'll see two values to copy:<br>"
            "&nbsp;&nbsp;•&nbsp;<b>Application ID</b> (also called Client ID) — "
            "click the small <i>Copy</i> button.<br>"
            "&nbsp;&nbsp;•&nbsp;<b>Client Secret</b> — click <b>Reset Secret</b>, "
            "confirm, then copy the value Discord shows you. Discord only shows "
            "the secret <i>once</i> — copy it immediately.<br><br>"
            "Then click <b>Next</b> below and paste both values into the "
            "fields on the last page."
        )
        step3.setObjectName("wizardBody")
        step3.setWordWrap(True)
        step3.setTextFormat(Qt.RichText)
        step3.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(step3)
        return page

    def _build_page_collect(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        title = QLabel("Paste your Discord credentials")
        title.setObjectName("wizardTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(title)

        body = _TightWrapLabel(
            "Paste the two values you copied from the Discord Developer "
            "Portal. The Application ID is public; the Client Secret is "
            "sensitive — Touchless stores both on your machine and never "
            "sends them to any server except Discord's own OAuth endpoint."
        )
        body.setObjectName("wizardBody")
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(body)

        # Application ID input
        id_label = QLabel("Application ID (Client ID)")
        id_label.setStyleSheet("color: #B6BFFF; font-size: 12px; font-weight: 700;")
        lay.addWidget(id_label)

        self._client_id_input = QLineEdit()
        self._client_id_input.setPlaceholderText("18-19 digit numeric ID")
        self._client_id_input.setStyleSheet(self._monospace_input_style())
        existing_id = str(getattr(self._config, "discord_client_id", "") or "")
        if existing_id:
            self._client_id_input.setText(existing_id)
        lay.addWidget(self._client_id_input)

        # Client Secret input
        secret_label = QLabel("Client Secret")
        secret_label.setStyleSheet("color: #B6BFFF; font-size: 12px; font-weight: 700;")
        lay.addWidget(secret_label)

        self._client_secret_input = QLineEdit()
        self._client_secret_input.setPlaceholderText("32-character secret")
        self._client_secret_input.setStyleSheet(self._monospace_input_style())
        existing_secret = str(getattr(self._config, "discord_client_secret", "") or "")
        if existing_secret:
            self._client_secret_input.setText(existing_secret)
        lay.addWidget(self._client_secret_input)

        self._validation_msg = QLabel("")
        self._validation_msg.setWordWrap(True)
        self._validation_msg.setStyleSheet("color: #FFB86B; font-size: 12px;")
        lay.addWidget(self._validation_msg)
        return page

    def _monospace_input_style(self) -> str:
        return (
            "QLineEdit {"
            " background: rgba(255,255,255,0.05);"
            " color: #E5F0FF;"
            " border: 1px solid rgba(255,255,255,0.22);"
            " border-radius: 10px;"
            " padding: 11px 14px;"
            " font-size: 13px;"
            " font-family: 'Consolas', 'Menlo', monospace;"
            "}"
            "QLineEdit:focus {"
            f" border: 1px solid {self._accent};"
            "}"
        )

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
        self._try_save_and_close()

    def _open_dev_portal_page(self) -> None:
        QDesktopServices.openUrl(QUrl(_DISCORD_DEV_DASHBOARD_URL))

    def _try_save_and_close(self) -> None:
        client_id = self._client_id_input.text().strip()
        client_secret = self._client_secret_input.text().strip()

        if not client_id or not client_secret:
            self._validation_msg.setText(
                "Paste both values — Application ID and Client Secret — before "
                "saving."
            )
            if not client_id:
                self._client_id_input.setFocus()
            else:
                self._client_secret_input.setFocus()
            return

        # Discord application IDs are 18-20 digit decimal numbers
        # (Discord snowflake format). Client secrets are 32-char
        # alphanumerics. Don't reject hard if format drifts in the
        # future — just warn if something looks obviously off.
        sanitized_id = client_id.replace(" ", "")
        sanitized_secret = client_secret.replace(" ", "")
        if not sanitized_id.isdigit() or len(sanitized_id) < 10:
            self._validation_msg.setText(
                "Application ID should be a single 18-20 digit number with no "
                "spaces. Double-check what you copied from the Developer Portal."
            )
            self._client_id_input.setFocus()
            return
        if len(sanitized_secret) < 16:
            self._validation_msg.setText(
                "Client Secret should be a ~32-character alphanumeric string. "
                "If yours looks short, click Reset Secret on the Discord page "
                "and copy the new value immediately — Discord only shows it once."
            )
            self._client_secret_input.setFocus()
            return

        try:
            self._config.discord_client_id = sanitized_id
            self._config.discord_client_secret = sanitized_secret
            save_config(self._config)
        except Exception as exc:
            self._validation_msg.setText(
                f"Couldn't save: {exc}. Check the app has write access to "
                "your settings folder."
            )
            return
        self.accept()


# Author: Konstantin Markov

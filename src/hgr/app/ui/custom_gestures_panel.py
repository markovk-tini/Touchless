"""Settings panel for the Custom Gestures Beta feature.

Drops into the existing settings nav at SECTION_CUSTOM_GESTURE. Calls
out to dedicated wizard / recorder / sandbox dialogs in sibling modules.

Camera handling: the main GestureWorker owns the webcam. Recording and
sandbox dialogs subscribe to its `raw_frame_ready` signal — no camera
contention, just frame-stream sharing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# Stylesheet for the scroll bars used inside the panel + the cards list.
# Mirrors the camera-panel scroll styling. The triple-selector for
# QScrollArea / its child QWidget / the qt_scrollarea_viewport is
# important — Qt creates an internal viewport widget that doesn't
# inherit a transparent background otherwise, and the page ends up
# painted with the OS default (white on Win 11).
_SCROLLBAR_STYLE = """
QScrollArea, QScrollArea > QWidget, QScrollArea QWidget#qt_scrollarea_viewport {{
    background: transparent;
    border: none;
}}
QScrollArea QScrollBar:vertical {{
    background: rgba(255,255,255,0.04);
    width: 10px;
    margin: 6px 3px 6px 3px;
    border-radius: 5px;
}}
QScrollArea QScrollBar::handle:vertical {{
    background: {accent};
    border-radius: 5px;
    min-height: 32px;
}}
QScrollArea QScrollBar::handle:vertical:hover {{
    background: {accent};
    border: 1px solid rgba(255,255,255,0.25);
}}
QScrollArea QScrollBar::add-line:vertical,
QScrollArea QScrollBar::sub-line:vertical {{
    height: 0px;
    background: transparent;
}}
"""

# Allow imports from the standalone custom_gestures module.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from hgr.custom_gestures.action import describe as describe_action
from hgr.custom_gestures.description import format_gesture_summary
from hgr.custom_gestures.registry import CustomGesture, GestureRegistry


_HOW_IT_WORKS_HTML = (
    "<p style='margin-top:0;'>Custom Gestures lets you record your own hand "
    "pose and bind it to any keystroke, hotkey, text snippet, URL, or shell "
    "command. Once saved, the gesture works alongside the built-in Touchless "
    "controls.</p>"
    "<p><b>How to use it:</b></p>"
    "<ol style='margin-top:2px; padding-left:18px;'>"
    "<li><b>Click Create New Gesture.</b> Give it a name and (optional) "
    "description.</li>"
    "<li><b>Set timing.</b> Hold-to-activate (default 1s) is how long you "
    "must keep the pose before the action fires. Cooldown (default 2s) "
    "prevents back-to-back firing while you keep holding.</li>"
    "<li><b>Pick an action.</b> Press a key, fire a hotkey combo, type "
    "text, open a URL, or run a shell command. The form will ask for the "
    "specific value once you choose.</li>"
    "<li><b>Click Start to record.</b> The camera opens. Hold your pose "
    "and click <b>Begin Recording</b>. Touchless captures 100 frames "
    "(~10 seconds) — let your hand drift naturally during this so the "
    "classifier learns your real range. The live finger-state readout "
    "in the top-right shows what the system is perceiving.</li>"
    "<li><b>Save when done.</b> Touchless prints a Hand Pose summary so "
    "you can verify the recording captured what you intended.</li>"
    "</ol>"
    "<p><b>Limitations of the current model (Beta):</b></p>"
    "<ul style='margin-top:2px; padding-left:18px;'>"
    "<li><b>Static poses only.</b> The classifier matches single-frame "
    "shapes. Swipes, waves, and motion-based gestures aren't supported "
    "yet.</li>"
    "<li><b>Avoid heavy occlusion.</b> Poses where one finger is hidden "
    "behind another (interlocked fingers, pinky tucked behind thumb) "
    "produce noisy landmark predictions and unreliable matches. Pick "
    "poses where every fingertip is visible to the camera.</li>"
    "<li><b>One hand at a time.</b> Custom gestures are matched on the "
    "primary tracked hand only.</li>"
    "</ul>"
)


class CustomGesturesPanel(QWidget):
    """Container widget holding the description, the create button, the
    saved-gesture cards list, and the sandbox button. Emits a signal
    when it needs the parent window to open one of the dialogs that
    consumes the live camera stream."""

    open_create_requested = Signal()
    open_sandbox_requested = Signal()
    open_edit_requested = Signal(str)  # emits gesture name to edit
    # Import / export bundle requests. The panel doesn't own a
    # parent QWidget for QFileDialog (or for conflict-resolution
    # message boxes) reliably — main_window does — so the panel
    # emits and main_window opens the file picker, runs the
    # bundle helper, and pings refresh_cards on success.
    import_requested = Signal()
    export_all_requested = Signal()
    export_one_requested = Signal(str)  # gesture name

    def __init__(
        self,
        config,
        accent_color: str,
        registry_path_provider: Optional[Callable[[], Path]] = None,
        worker_provider: Optional[Callable[[], object]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        # Default Preferred sizePolicy so the widget reports its
        # natural content sizeHint to the outer settingsContentScroll,
        # which engages when sizeHint > viewport (same as Save
        # Locations).
        self._config = config
        self._accent_color = accent_color
        self._registry_path_provider = registry_path_provider
        self._worker_provider = worker_provider
        self._registry = GestureRegistry()
        self._cards: List["GestureCard"] = []

        # Direct layout — NO inner page_scroll. The outer
        # settingsContentScroll (wraps the whole settings page)
        # handles overflow. This mirrors the Save Locations panel
        # architecture which the user confirmed has the layout they
        # want: one green scrollbar at the page level, content sizes
        # naturally.
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        root.addWidget(self._build_how_it_works_card())
        root.addWidget(self._build_actions_card())
        self._cards_card = self._build_cards_card()
        # Preferred vertical — sizes to its content. No stretch needed
        # because there's no scroll viewport to fill; the outer scroll
        # handles the page.
        self._cards_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        root.addWidget(self._cards_card)

        self.refresh_cards()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        # Install an event filter on every descendant so wheel events
        # are forwarded to the enclosing outer settingsContentScroll —
        # by default Qt only auto-scrolls when the wheel event reaches
        # the QScrollArea, but some descendants (especially QFrames
        # with WA_StyledBackground) end up consuming wheel events
        # silently. We catch them and re-send to the outer scroll.
        self._install_wheel_forwarder()

    def _install_wheel_forwarder(self) -> None:
        try:
            if getattr(self, "_wheel_forwarder_installed", False):
                return
            self.installEventFilter(self)
            for child in self.findChildren(QWidget):
                try:
                    child.installEventFilter(self)
                except Exception:
                    pass
            self._wheel_forwarder_installed = True
        except Exception:
            pass

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        try:
            from PySide6.QtCore import QEvent
            if event.type() == QEvent.Wheel:
                # Directly manipulate the outer scrollbar — sendEvent
                # to viewport wasn't propagating through some Qt
                # internal handling. setValue is the authoritative API.
                from PySide6.QtWidgets import QScrollArea
                walker = self.parent()
                for _ in range(8):
                    if walker is None:
                        break
                    if isinstance(walker, QScrollArea):
                        sb = walker.verticalScrollBar()
                        if sb is not None and sb.maximum() > 0:
                            # angleDelta().y() is 120 per "click" of
                            # the wheel; * 0.5 gives ~60 px per click,
                            # roughly matching Qt's default 3-line step.
                            delta = event.angleDelta().y()
                            new_val = sb.value() - int(delta * 0.5)
                            sb.setValue(max(0, min(sb.maximum(), new_val)))
                            return True
                    walker = walker.parent()
        except Exception:
            pass
        return super().eventFilter(obj, event)

    # --- card builders ---------------------------------------------------

    def _inner_card_stylesheet(self) -> str:
        try:
            c = QColor(getattr(self._config, "surface_color", "") or "#0F172A")
            r, g, b, _ = c.getRgb()
            is_light = (0.299 * r + 0.587 * g + 0.114 * b) > 160
        except Exception:
            is_light = False
        card_bg = "rgba(0,0,0,0.05)" if is_light else "rgba(255,255,255,0.04)"
        card_border = "rgba(11,61,145,0.18)" if is_light else "rgba(29,233,182,0.22)"
        return (
            f"QFrame#innerCard {{"
            f"  background-color: {card_bg};"
            f"  border: 1px solid {card_border};"
            f"  border-radius: 18px;"
            f"  color: {getattr(self._config, 'text_color', '#E5F6FF')};"
            f"}}"
        )

    def _inner_card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        box = QFrame()
        box.setObjectName("innerCard")
        box.setAttribute(Qt.WA_StyledBackground, True)
        box.setStyleSheet(self._inner_card_stylesheet())
        layout = QVBoxLayout(box)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        if title:
            t = QLabel(title)
            t.setObjectName("cardTitle")
            t.setStyleSheet("font-size: 16px; font-weight: 700;")
            layout.addWidget(t)
        return box, layout

    def _build_how_it_works_card(self) -> QFrame:
        box, layout = self._inner_card("How it works")
        body = QLabel(_HOW_IT_WORKS_HTML)
        body.setTextFormat(Qt.RichText)
        body.setWordWrap(True)
        body.setStyleSheet(
            f"color: {self._config.text_color}; font-size: 13px; line-height: 1.4;"
        )
        body.setOpenExternalLinks(False)
        layout.addWidget(body)
        return box

    def _build_actions_card(self) -> QFrame:
        box, layout = self._inner_card("")
        row = QHBoxLayout()
        row.setSpacing(10)

        self.create_button = QPushButton("+  Create New Gesture")
        self.create_button.setMinimumHeight(38)
        self.create_button.setStyleSheet(
            f"QPushButton {{"
            f"  background: {self._accent_color};"
            f"  color: #0B1620;"
            f"  font-weight: 700;"
            f"  font-size: 14px;"
            f"  border: none;"
            f"  border-radius: 8px;"
            f"  padding: 8px 18px;"
            f"}}"
            f"QPushButton:hover {{ background: #FFFFFF; }}"
        )
        self.create_button.clicked.connect(self.open_create_requested)
        row.addWidget(self.create_button)

        self.sandbox_button = QPushButton("Sandbox")
        self.sandbox_button.setMinimumHeight(38)
        self.sandbox_button.setStyleSheet(
            "QPushButton {"
            "  background: rgba(127,127,127,0.10);"
            f"  color: {self._config.text_color};"
            "  font-weight: 600;"
            "  border: 1px solid rgba(127,127,127,0.22);"
            "  border-radius: 8px;"
            "  padding: 8px 18px;"
            "}"
            "QPushButton:hover { background: rgba(255,255,255,0.10); }"
        )
        self.sandbox_button.clicked.connect(self.open_sandbox_requested)
        row.addWidget(self.sandbox_button)

        row.addStretch(1)

        # Import / Export. Compact tertiary buttons — shorter labels,
        # tighter padding, and a smaller min-height than Sandbox so
        # they don't crowd the primary actions on the left side of
        # the row. Right-aligned via the stretch above so they sit
        # at the far end of the row.
        secondary_btn_style = (
            "QPushButton {"
            "  background: rgba(127,127,127,0.10);"
            f"  color: {self._config.text_color};"
            "  font-weight: 600;"
            "  font-size: 12px;"
            "  border: 1px solid rgba(127,127,127,0.22);"
            "  border-radius: 8px;"
            "  padding: 4px 10px;"
            "}"
            "QPushButton:hover { background: rgba(255,255,255,0.10); }"
            "QPushButton:disabled {"
            f"  color: rgba(127,127,127,0.55);"
            "  border: 1px solid rgba(127,127,127,0.12);"
            "}"
        )
        self.import_button = QPushButton("Import")
        self.import_button.setMinimumHeight(30)
        self.import_button.setMaximumHeight(34)
        self.import_button.setStyleSheet(secondary_btn_style)
        self.import_button.clicked.connect(self.import_requested)
        row.addWidget(self.import_button)

        self.export_all_button = QPushButton("Export All")
        self.export_all_button.setMinimumHeight(30)
        self.export_all_button.setMaximumHeight(34)
        self.export_all_button.setStyleSheet(secondary_btn_style)
        self.export_all_button.clicked.connect(self.export_all_requested)
        row.addWidget(self.export_all_button)

        layout.addLayout(row)
        return box

    def _build_cards_card(self) -> QFrame:
        box, self._cards_layout = self._inner_card("Saved gestures")
        self._empty_label = QLabel(
            "No custom gestures yet. Click <b>Create New Gesture</b> to "
            "record your first one."
        )
        self._empty_label.setStyleSheet("color: #9FB3C2; font-style: italic;")
        self._empty_label.setWordWrap(True)
        self._cards_layout.addWidget(self._empty_label)

        # Cards stack directly in cards_card. The outer page_scroll
        # in CustomGesturesPanel.__init__ provides the single
        # scrollbar for the whole page.
        self._cards_container = QWidget()
        self._cards_container.setStyleSheet("background: transparent;")
        self._cards_container_layout = QVBoxLayout(self._cards_container)
        self._cards_container_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_container_layout.setSpacing(8)
        self._cards_container_layout.addStretch(1)
        self._cards_layout.addWidget(self._cards_container, 1)
        return box

    # --- public API ------------------------------------------------------

    def _ping_worker_reload(self) -> None:
        """Tell the running GestureWorker to re-read custom gestures so
        new / edited / deleted gestures take effect live without an app
        restart. Safe no-op when no worker is running."""
        if self._worker_provider is None:
            return
        try:
            worker = self._worker_provider()
        except Exception:
            return
        if worker is None:
            return
        try:
            worker.reload_custom_gestures()
        except Exception:
            pass

    def refresh_cards(self) -> None:
        """Reload the registry from disk and rebuild the cards list.
        Also pings the live GestureWorker so any add / edit / delete
        propagates to the running pipeline immediately."""
        self._ping_worker_reload()
        self._registry = GestureRegistry()
        self._registry.load()
        # Clear existing cards.
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        gestures = self._registry.list()
        # Export-All is meaningless with an empty registry; toggle it
        # so users get a clear "nothing to export" affordance instead
        # of an empty error dialog.
        if hasattr(self, "export_all_button"):
            self.export_all_button.setEnabled(bool(gestures))
        if not gestures:
            self._empty_label.show()
            return
        self._empty_label.hide()
        # Insert each new card BEFORE the trailing stretch so cards
        # stack from the top of the scroll area instead of expanding.
        insert_index = max(0, self._cards_container_layout.count() - 1)
        for g in gestures:
            card = GestureCard(
                g,
                on_delete=self._on_delete_gesture,
                on_edit=self._on_edit_gesture,
                on_export=lambda name: self.export_one_requested.emit(name),
                parent=self._cards_container,
                text_color=str(getattr(self._config, "text_color", "") or "#E5F6FF"),
            )
            self._cards.append(card)
            self._cards_container_layout.insertWidget(insert_index, card)
            insert_index += 1
        # Newly created gesture cards aren't covered by the wheel
        # forwarder yet — reinstall so wheel events anywhere on them
        # (and their internal QPushButton / QLabel children) get
        # forwarded to the outer settings scroll.
        try:
            for child in self.findChildren(QWidget):
                child.removeEventFilter(self)
                child.installEventFilter(self)
        except Exception:
            pass

    def _on_edit_gesture(self, name: str) -> None:
        self.open_edit_requested.emit(name)

    def _on_delete_gesture(self, name: str) -> None:
        # TouchlessNotice.show_confirm renders the dialog with the
        # app's themed title bar + surface color background, matching
        # the rest of Touchless instead of the OS-native QMessageBox.
        # Lazy import keeps the panel module independent of main_window
        # at import time (main_window already imports this panel).
        from .main_window import TouchlessNotice
        confirmed = TouchlessNotice.show_confirm(
            self,
            "Delete gesture",
            f"Delete custom gesture '{name}'? This cannot be undone.",
            confirm_label="Delete",
            cancel_label="Cancel",
        )
        if not confirmed:
            return
        self._registry.load()
        self._registry.remove(name)
        self._registry.save()
        self.refresh_cards()


class GestureCard(QFrame):
    """Expandable card for one saved gesture: header with name + action +
    delete; click expands to show description, how-to summary, action
    detail. Video-clip preview is a future addition."""

    def __init__(
        self,
        gesture: CustomGesture,
        on_delete: Callable[[str], None],
        on_edit: Optional[Callable[[str], None]] = None,
        on_export: Optional[Callable[[str], None]] = None,
        parent: Optional[QWidget] = None,
        text_color: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._gesture = gesture
        self._on_delete = on_delete
        self._on_edit = on_edit
        self._on_export = on_export
        self._expanded = False
        # Cache the active text colour so this card (which is built
        # outside the main MainWindow.apply_theme path) reads in both
        # light and dark mode. Falls back to the dark-mode default
        # when no colour is provided.
        self._text_color = text_color or "#E5F6FF"

        self.setObjectName("gestureCard")
        # Neutral-grey backgrounds + borders so the card stays
        # visible on both light and dark surfaces. The previous
        # rgba(255,255,255,0.04) was a near-invisible whitewash on
        # light mode.
        self.setStyleSheet(
            "QFrame#gestureCard {"
            "  background: rgba(127,127,127,0.10);"
            "  border: 1px solid rgba(127,127,127,0.22);"
            "  border-radius: 8px;"
            "}"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)

        # Top row: name (expanding) + edit/delete (fixed). The thumbnail
        # the user picked while recording lives inside the expanded
        # details below — the collapsed card stays compact so a list
        # of many custom gestures scrolls cleanly.
        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        self._toggle_button = QPushButton(f"▶  {gesture.name}")
        self._toggle_button.setStyleSheet(
            "QPushButton {"
            "  background: transparent;"
            f"  color: {self._text_color};"
            "  font-weight: 700;"
            "  font-size: 14px;"
            "  text-align: left;"
            "  padding: 4px 0;"
            "  border: none;"
            "}"
        )
        self._toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle_button.clicked.connect(self._toggle)
        top_row.addWidget(self._toggle_button, 1)

        edit_button = QPushButton("Edit")
        edit_button.setFixedWidth(64)
        edit_button.setStyleSheet(
            "QPushButton {"
            "  background: transparent;"
            f"  color: {self._text_color};"
            "  border: 1px solid rgba(127,127,127,0.45);"
            "  border-radius: 6px;"
            "  padding: 4px 10px;"
            "  font-size: 12px;"
            "}"
            "QPushButton:hover {"
            "  background: rgba(127,127,127,0.18);"
            "}"
        )
        if self._on_edit is not None:
            edit_button.clicked.connect(lambda: self._on_edit(self._gesture.name))
        else:
            edit_button.setEnabled(False)
        top_row.addWidget(edit_button, 0)

        export_button = QPushButton("Export")
        export_button.setFixedWidth(72)
        export_button.setStyleSheet(
            "QPushButton {"
            "  background: transparent;"
            f"  color: {self._text_color};"
            "  border: 1px solid rgba(127,127,127,0.45);"
            "  border-radius: 6px;"
            "  padding: 4px 10px;"
            "  font-size: 12px;"
            "}"
            "QPushButton:hover {"
            "  background: rgba(127,127,127,0.18);"
            "}"
        )
        if self._on_export is not None:
            export_button.clicked.connect(lambda: self._on_export(self._gesture.name))
        else:
            export_button.setEnabled(False)
        top_row.addWidget(export_button, 0)

        delete_button = QPushButton("Delete")
        delete_button.setFixedWidth(78)
        delete_button.setStyleSheet(
            "QPushButton {"
            "  background: transparent;"
            "  color: #C9818D;"
            "  border: 1px solid rgba(201,129,141,0.35);"
            "  border-radius: 6px;"
            "  padding: 4px 10px;"
            "  font-size: 12px;"
            "}"
            "QPushButton:hover {"
            "  background: rgba(201,129,141,0.15);"
            "  color: #FFFFFF;"
            "}"
        )
        delete_button.clicked.connect(lambda: self._on_delete(self._gesture.name))
        top_row.addWidget(delete_button, 0)

        root.addLayout(top_row)

        # Second row: the action description, full width but small text.
        # Prefix with the bound hand so the user sees at-a-glance which
        # hand fires this custom gesture in the live pipeline.
        if gesture.handedness in ("Left", "Right"):
            hand_prefix = f"[{gesture.handedness} hand] "
        else:
            hand_prefix = "[Either hand] "
        action_label = QLabel(hand_prefix + describe_action(gesture.action))
        action_label.setStyleSheet("color: #9FB3C2; font-size: 12px;")
        action_label.setWordWrap(True)
        root.addWidget(action_label)

        # Expanded body: description on the left, thumbnail on the
        # right. Was previously two stacked widgets (description above,
        # thumbnail below); side-by-side reads more like a "details
        # card" — the user can scan the summary text while seeing the
        # captured pose without having to scroll.
        expanded_row = QHBoxLayout()
        expanded_row.setSpacing(10)

        self._details = QLabel()
        self._details.setWordWrap(True)
        self._details.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._details.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._details.setStyleSheet(
            f"color: {self._text_color};"
            " font-family: Consolas, 'Courier New', monospace;"
            " font-size: 12px;"
            " background: rgba(127,127,127,0.18);"
            " padding: 10px;"
            " border-radius: 6px;"
        )
        self._details.setText(format_gesture_summary(gesture))
        self._details.hide()
        expanded_row.addWidget(self._details, 1)

        # Thumbnail: fixed width so it doesn't squeeze the description
        # away when the card is wide. Shown only when expanded.
        self._expanded_thumb_label = QLabel()
        self._expanded_thumb_label.setObjectName("gestureCardExpandedThumb")
        self._expanded_thumb_label.setAlignment(Qt.AlignCenter)
        self._expanded_thumb_label.setFixedSize(240, 180)
        self._expanded_thumb_label.setStyleSheet(
            "QLabel#gestureCardExpandedThumb {"
            "  background: rgba(0,0,0,0.30);"
            "  border: 1px solid rgba(255,255,255,0.10);"
            "  border-radius: 8px;"
            "  color: rgba(229,246,255,0.55);"
            "  font-size: 12px;"
            "  padding: 8px;"
            "}"
        )
        self._expanded_thumb_label.hide()
        expanded_row.addWidget(self._expanded_thumb_label, 0, Qt.AlignTop)

        root.addLayout(expanded_row)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._toggle_button.setText(
            f"{'▼' if self._expanded else '▶'}  {self._gesture.name}"
        )
        self._details.setVisible(self._expanded)
        if self._expanded:
            # Lazy-load the thumbnail the first time the card is
            # expanded so a list of many custom gestures doesn't pay
            # the disk cost up-front.
            self._refresh_expanded_thumbnail()
        self._expanded_thumb_label.setVisible(self._expanded)

    def _refresh_expanded_thumbnail(self) -> None:
        """Load the user-picked thumbnail (or animated motion clip)
        into the expanded slot. Falls back to a friendly placeholder
        when the gesture has no image. Animated GIFs (saved by the
        dynamic-gesture flow) are played via QMovie so the card shows
        the recorded motion instead of a frozen frame."""
        try:
            from hgr.custom_gestures.registry import GestureRegistry
            registry = GestureRegistry()
            registry.load()
            path = registry.thumbnail_path(self._gesture)
        except Exception:
            path = None
        # Stop any previous QMovie so switching gestures doesn't leak
        # animations.
        prev_movie = getattr(self, "_expanded_thumb_movie", None)
        if prev_movie is not None:
            try:
                prev_movie.stop()
            except Exception:
                pass
            self._expanded_thumb_movie = None
        if path is not None:
            spath = str(path)
            if spath.lower().endswith(".gif"):
                from PySide6.QtGui import QMovie
                from PySide6.QtCore import QSize
                movie = QMovie(spath)
                movie.setScaledSize(QSize(224, 164))
                self._expanded_thumb_label.setMovie(movie)
                movie.start()
                self._expanded_thumb_movie = movie
                return
            pix = QPixmap(spath)
            if not pix.isNull():
                # Subtract the label's CSS padding (8px each side) so
                # the scaled pixmap doesn't render past the rounded
                # border. The fixed label size is 240×180.
                self._expanded_thumb_label.setPixmap(
                    pix.scaled(
                        224,
                        164,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
                return
        self._expanded_thumb_label.setText("No image picked for this gesture.")

# Author: Konstantin Markov

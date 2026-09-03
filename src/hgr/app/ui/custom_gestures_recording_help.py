"""Shared help UI + timed GIF helpers for custom-gesture recorders."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPropertyAnimation,
    Qt,
    QTimer,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


FrameStamp = Tuple[np.ndarray, float]  # (bgr, monotonic_ts)


_CONSISTENCY_TIP_TEXT = (
    "Keep each take as similar as you can. If you mess up, restart — "
    "consistent recordings help Touchless learn and detect your gesture "
    "more accurately."
)


class RecordingConsistencyTip(QFrame):
    """Tip card overlaid on the live camera view.

    Green outline, translucent blue fill, dismiss control bottom-right.
    Auto-fades out after 7 seconds (or sooner if Dismiss is clicked).
    """

    _AUTO_HIDE_MS = 7000
    _FADE_MS = 650

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("recordingConsistencyTip")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            "QFrame#recordingConsistencyTip {"
            "  background: rgba(30, 90, 170, 0.72);"
            "  border: 2px solid #1DE9B6;"
            "  border-radius: 12px;"
            "}"
            "QLabel#recordingConsistencyTipText {"
            "  color: #E8F7FF;"
            "  font-size: 13px;"
            "  font-weight: 600;"
            "  background: transparent;"
            "  border: none;"
            "}"
            "QPushButton#recordingConsistencyTipDismiss {"
            "  background: rgba(255,255,255,0.12);"
            "  color: #E8F7FF;"
            "  border: 1px solid rgba(29,233,182,0.55);"
            "  border-radius: 6px;"
            "  padding: 4px 12px;"
            "  font-size: 12px;"
            "  font-weight: 700;"
            "}"
            "QPushButton#recordingConsistencyTipDismiss:hover {"
            "  background: rgba(29,233,182,0.28);"
            "  color: #0B1620;"
            "}"
        )
        self.setMaximumWidth(420)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(10)

        self._text = QLabel(_CONSISTENCY_TIP_TEXT)
        self._text.setObjectName("recordingConsistencyTipText")
        self._text.setWordWrap(True)
        self._text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        root.addWidget(self._text)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addStretch(1)
        self._dismiss = QPushButton("Dismiss")
        self._dismiss.setObjectName("recordingConsistencyTipDismiss")
        self._dismiss.setCursor(Qt.PointingHandCursor)
        self._dismiss.setFocusPolicy(Qt.NoFocus)
        self._dismiss.clicked.connect(self.dismiss)
        btn_row.addWidget(self._dismiss, 0, Qt.AlignRight)
        root.addLayout(btn_row)

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)

        self._fade: Optional[QPropertyAnimation] = None
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._start_fade)
        self._host: Optional[QWidget] = None
        self.hide()

    def attach(self, host: QWidget) -> None:
        """Parent onto the live-view widget and show the tip once."""
        if host is None:
            return
        if self._host is not None and self._host is not host:
            try:
                self._host.removeEventFilter(self)
            except Exception:
                pass
        self._host = host
        self.setParent(host)
        host.installEventFilter(self)
        self.show_tip()

    def show_tip(self) -> None:
        self._hide_timer.stop()
        if self._fade is not None:
            self._fade.stop()
            self._fade = None
        self._opacity.setOpacity(1.0)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        self._hide_timer.start(self._AUTO_HIDE_MS)

    def dismiss(self) -> None:
        self._hide_timer.stop()
        if self._fade is not None:
            self._fade.stop()
            self._fade = None
        self.hide()
        self._opacity.setOpacity(1.0)

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self._host and event.type() == QEvent.Resize:
            self._reposition()
        return super().eventFilter(obj, event)

    def _reposition(self) -> None:
        host = self._host
        if host is None:
            return
        # Sit below the green on-frame prompt
        # ("Click Begin … or press Spacebar") drawn near the top of
        # the live view (~y=28–40 in frame coords).
        side_margin = 16
        top_margin = 56
        max_w = min(420, max(200, host.width() - 2 * side_margin))
        self.setFixedWidth(max_w)
        self.adjustSize()
        w = self.width()
        h = self.height()
        x = max(side_margin, (host.width() - w) // 2)
        y = top_margin
        # Keep fully inside the live view if the host is short.
        if y + h + side_margin > host.height() > 0:
            y = max(side_margin, host.height() - h - side_margin)
        self.setGeometry(x, y, w, h)

    def _start_fade(self) -> None:
        if not self.isVisible():
            return
        anim = QPropertyAnimation(self._opacity, b"opacity", self)
        anim.setDuration(self._FADE_MS)
        anim.setStartValue(self._opacity.opacity())
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.finished.connect(self._on_fade_finished)
        self._fade = anim
        anim.start()

    def _on_fade_finished(self) -> None:
        self.hide()
        self._opacity.setOpacity(1.0)
        self._fade = None


class ExpandableHelpPanel(QWidget):
    """Short blurb + optional Show more details for recorder pages."""

    def __init__(
        self,
        *,
        summary_html: str,
        details_html: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._summary = QLabel(summary_html)
        self._summary.setWordWrap(True)
        self._summary.setTextFormat(Qt.RichText)
        self._summary.setStyleSheet("color: #BAE6FD; font-size: 13px;")
        layout.addWidget(self._summary)

        self._details = QLabel(details_html)
        self._details.setWordWrap(True)
        self._details.setTextFormat(Qt.RichText)
        self._details.setStyleSheet("color: #94A3B8; font-size: 12px;")
        self._details.hide()
        layout.addWidget(self._details)

        self._toggle = QPushButton("Show more")
        self._toggle.setFlat(True)
        self._toggle.setCursor(Qt.PointingHandCursor)
        self._toggle.setStyleSheet(
            "QPushButton {"
            "  color: #1DE9B6; background: transparent; border: none;"
            "  text-align: left; padding: 0; font-size: 12px; font-weight: 600;"
            "}"
            "QPushButton:hover { color: #FFFFFF; }"
        )
        self._toggle.clicked.connect(self._on_toggle)
        layout.addWidget(self._toggle, 0, Qt.AlignLeft)
        self._expanded = False

    def _on_toggle(self) -> None:
        self._expanded = not self._expanded
        self._details.setVisible(self._expanded)
        self._toggle.setText("Show less" if self._expanded else "Show more")


def durations_ms_from_timestamps(
    timestamps: Sequence[float],
    *,
    freeze_end_ms: int = 1500,
    min_ms: int = 20,
    max_ms: int = 5000,
) -> List[int]:
    """Per-frame GIF durations from capture timestamps + end freeze."""
    n = len(timestamps)
    if n <= 0:
        return []
    if n == 1:
        return [max(min_ms, int(freeze_end_ms))]
    out: List[int] = []
    for i in range(n - 1):
        dt = float(timestamps[i + 1] - timestamps[i])
        ms = int(round(dt * 1000.0))
        out.append(max(min_ms, min(max_ms, ms)))
    # Hold the final frame so the loop doesn't snap/black-flash.
    out.append(max(min_ms, int(freeze_end_ms)))
    return out


def save_motion_gif(
    registry,
    frames: Sequence[Union[np.ndarray, FrameStamp]],
    gesture_name: str,
    *,
    freeze_end_ms: int = 1500,
    max_width: int = 320,
) -> str:
    """Save a looping GIF at roughly real capture speed.

    `frames` may be raw BGR arrays or `(bgr, timestamp)` pairs. When
    timestamps are present, frame delays follow wall-clock spacing.
    The last frame is held for `freeze_end_ms` before looping. Uses
    disposal=1 (leave in place) so loops don't flash black.
    """
    if not frames:
        return ""
    try:
        from PIL import Image
    except Exception as exc:
        print(f"[custom-gestures] GIF save needs Pillow: {exc}")
        return ""

    bgr_list: List[np.ndarray] = []
    stamps: List[float] = []
    for item in frames:
        if isinstance(item, tuple) and len(item) == 2:
            bgr_list.append(item[0])
            stamps.append(float(item[1]))
        else:
            bgr_list.append(item)  # type: ignore[arg-type]

    if not bgr_list:
        return ""

    if len(stamps) == len(bgr_list) and len(stamps) >= 2:
        durations = durations_ms_from_timestamps(
            stamps, freeze_end_ms=freeze_end_ms,
        )
    else:
        # Fallback ~real-time 30 fps assumption.
        durations = [33] * (len(bgr_list) - 1) + [max(33, int(freeze_end_ms))]

    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in gesture_name
    ).strip("_") or "gesture"
    filename = f"{safe}.gif"
    try:
        target: Path = registry.thumbnails_dir() / filename
    except Exception:
        return ""

    pil_frames: List[Image.Image] = []
    for f in bgr_list:
        try:
            h, w = f.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                f = cv2.resize(
                    f,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))
        except Exception:
            continue
    if not pil_frames:
        return ""
    if len(durations) != len(pil_frames):
        durations = [33] * (len(pil_frames) - 1) + [max(33, int(freeze_end_ms))]

    try:
        pil_frames[0].save(
            str(target),
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
            disposal=1,  # leave previous frame — avoids black flash
        )
        return filename
    except Exception as exc:
        print(f"[custom-gestures] GIF save failed: {exc}")
        return ""

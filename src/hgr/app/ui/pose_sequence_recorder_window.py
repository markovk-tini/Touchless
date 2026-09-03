"""Pose-sequence recorder: one continuous take, then auto-analysis.

User presses Begin, performs the full sequence (e.g. 3→2→1) in one go,
then Stop. The analyzer learns each pose, hold durations, gaps, and
handedness — no preset timing knobs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QEvent
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from hgr.custom_gestures.pose_sequence_analysis import (
    SequenceAnalysis,
    SequenceFrame,
    analyze_pose_sequence,
)
from hgr.custom_gestures.recorder import landmarks_from_mediapipe
from hgr.custom_gestures.registry import Action, GestureRegistry

from .custom_gestures_chrome import apply_touchless_titlebar
from .custom_gestures_recording_help import (
    ExpandableHelpPanel,
    RecordingConsistencyTip,
    save_motion_gif,
)


_DEFAULT_STEPS = 3
_CAPTURE_INTERVAL_FRAMES = 1

_SEQUENCE_SUMMARY = (
    "Use <b>Sequence</b> for an ordered chain of held poses you invent. "
    "Record the full chain in one continuous take — Touchless learns each "
    "shape, hold/gap timing, order, and which hand."
)
_SEQUENCE_DETAILS = (
    "<p style='margin:0 0 6px 0;'><b>Good for:</b> multi-step sign-like "
    "holds, any series of distinct static shapes that don’t exist as "
    "built-in gestures yet.</p>"
    "<p style='margin:0 0 6px 0;'><b>Don’t use for:</b> swipes, circles, waves, "
    "or other continuous motion — pick <b>Dynamic</b> for those. Don’t use for "
    "a single held pose — pick <b>Static</b>.</p>"
    "<p style='margin:0 0 6px 0;'><b>How to record:</b> set pose count (2–5), "
    "press Space / Begin, perform every pose in order with a brief hold on each, "
    "then Stop. Wrong order, wrong hand, or holding/waiting too long vs your "
    "demo will fail live matching.</p>"
    "<p style='margin:0;'><b>Limits:</b> 2–5 poses; needs clearly different "
    "shapes between steps; one hand; timing is learned from this take with a "
    "small buffer.</p>"
)


def _default_step_names(n: int) -> list:
    return [f"Pose {i + 1}" for i in range(max(2, min(5, int(n))))]


class _SpacePassthroughFilter(QObject):
    """Forward Space to the recorder even when a child control has focus."""

    def __init__(self, on_space, parent=None) -> None:
        super().__init__(parent)
        self._on_space = on_space

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Space:
            self._on_space()
            return True
        return False


class PoseSequenceRecorderWindow(QDialog):
    """One-take capture + analysis for pose_sequence gestures."""

    saved = Signal(str)

    def __init__(
        self,
        worker,
        accent_color: str,
        name: str,
        description: str,
        action: Action,
        parent: Optional[QWidget] = None,
        config=None,
        *,
        step_count: int = _DEFAULT_STEPS,
    ) -> None:
        super().__init__(parent)
        from .window_chrome import install_indigo_chrome

        self.setWindowTitle(f"Sequence: {name}")
        self.setModal(True)
        self.setMinimumSize(820, 600)
        self._body = install_indigo_chrome(self, f"Sequence: {name}")
        self._worker = worker
        self._accent = accent_color or "#1DE9B6"
        self._name = name
        self._description = description
        self._action = action
        self._config = config

        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._mp_drawer = mp.solutions.drawing_utils
        self._mp_hand_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
        self._mp_conn_style = mp.solutions.drawing_styles.get_default_hand_connections_style()

        n = max(2, min(5, int(step_count)))
        self._step_count = n
        self._step_names = _default_step_names(n)
        self._frames: List[SequenceFrame] = []
        self._video_clip_frames: List[np.ndarray] = []
        self._analysis: Optional[SequenceAnalysis] = None
        # idle | recording | review
        self._state = "idle"
        self._frames_since_capture = 0
        self._owns_camera = False
        self._cap = None
        self._poll_timer: Optional[QTimer] = None
        self._using_worker = False
        self._camera_connect_attempted = False
        self._record_started_at = 0.0

        self._build()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        try:
            apply_touchless_titlebar(self)
        except Exception:
            pass
        if not self._camera_connect_attempted:
            self._camera_connect_attempted = True
            QTimer.singleShot(0, self._connect_camera)
        if not getattr(self, "_consistency_tip_shown", False):
            self._consistency_tip_shown = True
            tip = getattr(self, "_consistency_tip", None)
            if tip is not None:
                QTimer.singleShot(0, lambda: tip.attach(self._video_label))

    def closeEvent(self, event):  # noqa: N802
        self._teardown_camera()
        try:
            self._mp_hands.close()
        except Exception:
            pass
        super().closeEvent(event)

    def _build(self) -> None:
        root = QVBoxLayout(self._body)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(10)

        self._title = QLabel()
        self._title.setStyleSheet(
            "color: #E5F6FF; font-size: 16px; font-weight: 700;"
        )
        root.addWidget(self._title)

        self._help = ExpandableHelpPanel(
            summary_html=_SEQUENCE_SUMMARY,
            details_html=_SEQUENCE_DETAILS,
        )
        root.addWidget(self._help)

        self._instructions = QLabel()
        self._instructions.setWordWrap(True)
        self._instructions.setStyleSheet(
            "color: #1DE9B6; font-size: 15px; font-weight: 700;"
        )
        root.addWidget(self._instructions)

        count_row = QHBoxLayout()
        count_row.setSpacing(4)
        count_lbl = QLabel("Poses in sequence")
        count_lbl.setStyleSheet("color: #DCE9F2;")
        count_row.addWidget(count_lbl)
        self._count_minus = QPushButton("−")
        self._count_minus.setObjectName("countBtn")
        self._count_minus.setFixedSize(18, 18)
        self._count_minus.setFocusPolicy(Qt.NoFocus)
        self._count_minus.clicked.connect(lambda: self._nudge_count(-1))
        count_row.addWidget(self._count_minus)
        self._count_value = QLabel(str(self._step_count))
        self._count_value.setAlignment(Qt.AlignCenter)
        self._count_value.setFixedWidth(16)
        self._count_value.setStyleSheet(
            "color: #E5F6FF; font-size: 13px; font-weight: 700;"
        )
        count_row.addWidget(self._count_value)
        self._count_plus = QPushButton("+")
        self._count_plus.setObjectName("countBtn")
        self._count_plus.setFixedSize(18, 18)
        self._count_plus.setFocusPolicy(Qt.NoFocus)
        self._count_plus.clicked.connect(lambda: self._nudge_count(1))
        count_row.addWidget(self._count_plus)
        count_row.addStretch(1)
        root.addLayout(count_row)

        self._video_label = QLabel()
        self._video_label.setMinimumSize(640, 360)
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet(
            "background: #0B1422; border-radius: 10px; color: #94A3B8;"
        )
        self._video_label.setText("Connecting camera…")
        root.addWidget(self._video_label, 1)

        self._consistency_tip = RecordingConsistencyTip(self._video_label)

        self._progress_label = QLabel()
        self._progress_label.setWordWrap(True)
        self._progress_label.setStyleSheet("color: #94A3B8;")
        root.addWidget(self._progress_label)

        buttons = QHBoxLayout()
        self._exit_btn = QPushButton("Exit")
        self._exit_btn.setFocusPolicy(Qt.NoFocus)
        self._exit_btn.clicked.connect(self.reject)
        buttons.addWidget(self._exit_btn)
        buttons.addStretch(1)
        self._retry_btn = QPushButton("Retry")
        self._retry_btn.setFocusPolicy(Qt.NoFocus)
        self._retry_btn.clicked.connect(self._on_retry)
        self._retry_btn.hide()
        buttons.addWidget(self._retry_btn)
        self._begin_btn = QPushButton("Begin sequence")
        self._begin_btn.setObjectName("primaryBtn")
        self._begin_btn.setFocusPolicy(Qt.NoFocus)
        self._begin_btn.clicked.connect(self._on_begin)
        buttons.addWidget(self._begin_btn)
        self._stop_btn = QPushButton("Stop & analyze")
        self._stop_btn.setObjectName("primaryBtn")
        self._stop_btn.setFocusPolicy(Qt.NoFocus)
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.hide()
        buttons.addWidget(self._stop_btn)
        self._save_btn = QPushButton("Save sequence")
        self._save_btn.setFocusPolicy(Qt.NoFocus)
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.hide()
        buttons.addWidget(self._save_btn)
        root.addLayout(buttons)

        self.setStyleSheet(
            f"""
            QDialog {{ background: #0E1822; }}
            QLabel {{ color: #DCE9F2; }}
            QPushButton {{
                background: rgba(255,255,255,0.06); color: #E5F6FF;
                border: 1px solid rgba(255,255,255,0.12); border-radius: 8px;
                padding: 8px 16px;
            }}
            QPushButton#countBtn {{
                background: rgba(255,255,255,0.08); color: #E5F6FF;
                border: 1px solid rgba(255,255,255,0.18); border-radius: 3px;
                padding: 0px; margin: 0px;
                min-width: 18px; max-width: 18px;
                min-height: 18px; max-height: 18px;
                font-size: 12px; font-weight: 700;
            }}
            QPushButton#countBtn:hover {{
                background: rgba(255,255,255,0.16);
            }}
            QPushButton#primaryBtn {{
                background: {self._accent}; color: #0B1620; font-weight: 700;
                border: none;
            }}
            """
        )
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._space_shortcut.setContext(Qt.WindowShortcut)
        self._space_shortcut.activated.connect(self._on_space)
        self._space_filter = _SpacePassthroughFilter(self._on_space, self)
        self.installEventFilter(self._space_filter)
        self._refresh_labels()

    def _nudge_count(self, delta: int) -> None:
        if self._state != "idle":
            return
        self._set_step_count(self._step_count + int(delta))

    def _set_step_count(self, value: int) -> None:
        n = max(2, min(5, int(value)))
        self._step_count = n
        self._step_names = _default_step_names(n)
        self._count_value.setText(str(n))
        self._count_minus.setEnabled(n > 2 and self._state == "idle")
        self._count_plus.setEnabled(n < 5 and self._state == "idle")
        self._refresh_labels()

    def _format_analysis(self, a: SequenceAnalysis) -> str:
        holds = " / ".join(f"{ms} ms" for ms in a.hold_ms)
        gaps = " / ".join(f"{ms} ms" for ms in a.gap_ms) if a.gap_ms else "—"
        hand = a.handedness or "either"
        return (
            f"{len(a.step_names)} poses · "
            f"holds {holds} · gaps {gaps} · "
            f"match window hold {a.dwell_ms}–{a.max_hold_ms} ms, "
            f"gap ≤ {a.max_gap_ms} ms · hand {hand}"
        )

    def _refresh_labels(self) -> None:
        if self._state == "idle":
            self._title.setText(
                f"Record sequence  ·  {self._step_count} poses"
            )
            self._instructions.setText(
                "Click <b>Begin sequence</b> or press <b>Spacebar</b>, "
                "perform your poses in order, then "
                "<b>Stop &amp; analyze</b> (Space again)."
            )
            self._progress_label.setText("")
            return
        if self._state == "recording":
            self._title.setText(
                f"Recording…  ·  {self._step_count} poses"
            )
            self._instructions.setText(
                "Perform the full sequence now. When finished, click "
                "<b>Stop &amp; analyze</b> or press <b>Spacebar</b>."
            )
            elapsed = max(0.0, time.monotonic() - self._record_started_at)
            self._progress_label.setText(
                f"Frames {len(self._frames)} · {elapsed:.1f}s"
            )
            return
        # review
        self._title.setText("Sequence learned")
        if self._analysis is not None:
            self._instructions.setText(
                f"Learned <b>{self._step_count} poses</b> from your take. "
                "Click <b>Save sequence</b> or press <b>Spacebar</b>, or Retry."
            )
            self._progress_label.setText(self._format_analysis(self._analysis))
        else:
            self._instructions.setText("Analysis failed — tap Retry.")
            self._progress_label.setText("")

    def _set_count_controls_enabled(self, enabled: bool) -> None:
        n = self._step_count
        self._count_minus.setEnabled(enabled and n > 2)
        self._count_plus.setEnabled(enabled and n < 5)

    def _on_space(self) -> None:
        if self._state == "idle":
            self._on_begin()
        elif self._state == "recording":
            self._on_stop()
        elif self._state == "review" and self._analysis is not None:
            self._on_save()

    def _on_begin(self) -> None:
        if self._state != "idle":
            return
        self._set_count_controls_enabled(False)
        self._frames = []
        self._video_clip_frames = []
        self._analysis = None
        self._frames_since_capture = 0
        self._record_started_at = time.monotonic()
        self._state = "recording"
        self._begin_btn.hide()
        self._stop_btn.show()
        self._save_btn.hide()
        self._retry_btn.hide()
        self._refresh_labels()

    def _on_retry(self) -> None:
        self._frames = []
        self._video_clip_frames = []
        self._analysis = None
        self._state = "idle"
        self._set_count_controls_enabled(True)
        self._begin_btn.show()
        self._stop_btn.hide()
        self._save_btn.hide()
        self._retry_btn.hide()
        self._refresh_labels()

    def _on_stop(self) -> None:
        from .window_chrome import touchless_message_box

        if self._state != "recording":
            return
        self._state = "review"
        self._stop_btn.hide()
        try:
            self._analysis = analyze_pose_sequence(
                self._frames,
                expected_steps=self._step_count,
                step_names=self._step_names,
            )
        except Exception as exc:
            self._analysis = None
            self._retry_btn.show()
            self._refresh_labels()
            touchless_message_box(
                self, "Couldn’t learn sequence", str(exc),
                icon=QMessageBox.Warning, buttons=QMessageBox.Ok,
            )
            return
        self._save_btn.show()
        self._retry_btn.show()
        self._refresh_labels()

    def _on_save(self) -> None:
        from .window_chrome import touchless_message_box

        if self._analysis is None:
            return
        a = self._analysis
        registry = GestureRegistry()
        registry.load()
        # Persist the one-take camera clip as an animated GIF so the
        # gesture card can play back what the user recorded.
        image_filename = save_motion_gif(
            registry, self._video_clip_frames, self._name,
        )
        try:
            registry.add_pose_sequence(
                self._name,
                a.steps,
                self._action,
                description=self._description,
                overwrite=False,
                handedness=a.handedness,
                image_filename=image_filename,
                dwell_ms=a.dwell_ms,
                max_hold_ms=a.max_hold_ms,
                max_gap_ms=a.max_gap_ms,
            )
            registry.save()
        except Exception as exc:
            touchless_message_box(
                self, "Save failed", str(exc),
                icon=QMessageBox.Critical, buttons=QMessageBox.Ok,
            )
            return
        self.saved.emit(self._name)
        self.accept()

    # ---- camera ----

    def _connect_camera(self) -> None:
        if self._worker is not None and hasattr(self._worker, "raw_frame_ready"):
            try:
                self._worker.raw_frame_ready.connect(self._on_worker_frame)
                self._using_worker = True
                self._video_label.setText("")
                return
            except Exception:
                pass
        self._owns_camera = True
        self._cap = cv2.VideoCapture(0)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_camera)
        self._poll_timer.start(33)

    def _teardown_camera(self) -> None:
        if self._using_worker and self._worker is not None:
            try:
                self._worker.raw_frame_ready.disconnect(self._on_worker_frame)
            except Exception:
                pass
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _poll_camera(self) -> None:
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if ok:
            self._handle_frame(frame)

    def _on_worker_frame(self, frame) -> None:
        if frame is None:
            return
        try:
            self._handle_frame(np.asarray(frame))
        except Exception:
            pass

    def _handle_frame(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None or frame_bgr.size == 0:
            return
        mirrored = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
        result = self._mp_hands.process(rgb)
        hand_present = bool(result.multi_hand_landmarks)
        if hand_present:
            hand_lms = result.multi_hand_landmarks[0]
            self._mp_drawer.draw_landmarks(
                mirrored, hand_lms,
                mp.solutions.hands.HAND_CONNECTIONS,
                self._mp_hand_style, self._mp_conn_style,
            )
            try:
                lm = landmarks_from_mediapipe(hand_lms.landmark)
            except Exception:
                lm = None
            handed = ""
            try:
                if result.multi_handedness:
                    handed = str(
                        result.multi_handedness[0].classification[0].label or ""
                    )
            except Exception:
                handed = ""
            if lm is not None and self._state == "recording":
                self._frames_since_capture += 1
                if self._frames_since_capture >= _CAPTURE_INTERVAL_FRAMES:
                    self._frames_since_capture = 0
                    self._frames.append(
                        SequenceFrame(
                            t=time.monotonic(),
                            landmarks=np.asarray(lm, dtype=np.float32).copy(),
                            handedness=handed,
                        )
                    )
                    self._refresh_labels()

        if self._state == "recording":
            badge = f"RECORDING  {len(self._frames)} frames"
            color = (40, 220, 40)
        elif self._state == "review" and self._analysis is not None:
            badge = "LEARNED — Save or Retry"
            color = (60, 220, 220)
        else:
            badge = "Click Begin sequence or press Spacebar"
            color = (40, 220, 40)  # bold green prompt
        cv2.putText(
            mirrored, badge, (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
        if self._state == "recording":
            # Timed clip for card GIF — ~25 fps max to keep size sane,
            # durations still follow wall clock between kept frames.
            try:
                now = time.monotonic()
                if (
                    not self._video_clip_frames
                    or (now - self._video_clip_frames[-1][1]) >= 0.04
                ):
                    h, w = mirrored.shape[:2]
                    scale = min(1.0, 320.0 / max(1, w))
                    small = cv2.resize(
                        mirrored,
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                    self._video_clip_frames.append((small.copy(), now))
            except Exception:
                pass
        self._show_frame(mirrored)

    def _show_frame(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._video_label.setPixmap(pix)

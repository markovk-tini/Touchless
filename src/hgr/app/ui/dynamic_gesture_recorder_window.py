"""Recording dialog for dynamic (motion-based) custom gestures.

Companion to `custom_gestures_recorder.py::RecordingWindow` which
handles static poses. The two are kept as separate dialog classes so
each can have a UI tuned to its workflow:

  * Static: hold one pose, capture ~100 sample frames automatically
    with stability gating.
  * Dynamic (this file): perform the gesture N takes (default 10),
    each take is one full motion sweep; the duration mode is set in
    the wizard and passed in.

UI mirrors the static recorder so users see the same shell:

  - Instruction line at top
  - Big video panel with the live hand skeleton drawn
  - "Recording complete!" overlay that appears once all takes are in
  - Progress label below the camera
  - Button row: Exit · Restart · Begin Recording / Stop / Save

After Save, a clip-picker dialog (the same one the static recorder
uses for thumbnails) lets the user choose which of the N takes
becomes the canonical example image shown on the gesture card.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import mediapipe as mp
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from hgr.custom_gestures.action import Action
from hgr.custom_gestures.dynamic_recorder import (
    DEFAULT_TARGET_TAKES,
    DurationMode,
    DynamicGestureRecorder,
    RecorderState,
)
from hgr.custom_gestures.dynamic_recording import palm_scale_from_landmarks
from hgr.custom_gestures.registry import GestureRegistry
from .custom_gestures_chrome import apply_touchless_titlebar
from .custom_gestures_recording_help import (
    ExpandableHelpPanel,
    RecordingConsistencyTip,
    durations_ms_from_timestamps,
    save_motion_gif,
)
from .window_chrome import apply_touchless_chrome, install_indigo_chrome, touchless_message_box


_DYNAMIC_SUMMARY = (
    "Use <b>Dynamic</b> for continuous motion — swipes, circles, waves, or "
    "any path where keypoint movement over time matters. Record several takes; "
    "Touchless learns the moving landmarks and matches live motion to those paths."
)
_DYNAMIC_DETAILS = (
    "<p style='margin:0 0 6px 0;'><b>Good for:</b> strokes and loops in space, "
    "finger wiggles with clear motion, countdown-like motions if the hand "
    "moves as one continuous gesture.</p>"
    "<p style='margin:0 0 6px 0;'><b>Don’t use for:</b> a single held pose "
    "(use <b>Static</b>) or an ordered chain of distinct held shapes with "
    "pauses (use <b>Sequence</b>).</p>"
    "<p style='margin:0 0 6px 0;'><b>How to record:</b> prefer <b>Until stopped</b> "
    "for longer takes. Do the same motion ~10 times at a natural pace. Keep the "
    "hand clearly in frame; vary speed a little across takes.</p>"
    "<p style='margin:0;'><b>Limits:</b> one hand; needs visible motion; "
    "very subtle pose-only changes without path may match poorly.</p>"
)


def _landmarks_array_from_mediapipe(mp_landmarks) -> np.ndarray:
    """Convert MediaPipe HandLandmarks → (21, 3) float32 numpy."""
    out = np.empty((21, 3), dtype=np.float32)
    for i, lm in enumerate(mp_landmarks):
        out[i, 0] = lm.x
        out[i, 1] = lm.y
        out[i, 2] = lm.z
    return out


def _duration_mode_from_string(value: str) -> DurationMode:
    v = (value or "").strip().lower()
    mapping = {
        "until_stopped": DurationMode.UNTIL_STOPPED,
        "fixed_1s": DurationMode.FIXED_1S,
        "fixed_2s": DurationMode.FIXED_2S,
        "fixed_3s": DurationMode.FIXED_3S,
        "fixed_short": DurationMode.FIXED_SHORT,
        "fixed_long": DurationMode.FIXED_LONG,
    }
    return mapping.get(v, DurationMode.UNTIL_STOPPED)


class DynamicGestureRecorderWindow(QDialog):
    """Dialog that captures N motion takes + saves them as a dynamic
    gesture in the registry. Duration-per-take is fixed at construction
    time (chosen earlier in the wizard)."""

    saved = Signal(str)  # emits gesture name on successful save

    def __init__(
        self,
        worker,
        accent_color: str,
        name: str,
        description: str,
        action: Action,
        duration_mode: str = "until_stopped",
        parent: Optional[QWidget] = None,
        config=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Recording: {name}")
        self.setModal(True)
        self.setMinimumSize(820, 560)
        # r51: install_indigo_chrome for Win10 + Win11 parity.
        self._body = install_indigo_chrome(self, f"Recording: {name}")

        self._worker = worker
        self._accent_color = accent_color or "#1DE9B6"
        self._name = name
        self._description = description
        self._action = action
        self._config = config
        self._duration_mode = _duration_mode_from_string(duration_mode)

        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._mp_drawer = mp.solutions.drawing_utils
        self._mp_hand_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
        self._mp_conn_style = mp.solutions.drawing_styles.get_default_hand_connections_style()

        self._recorder = self._build_recorder()
        self._latest_handedness: Optional[str] = None
        self._handedness_votes: List[str] = []

        # Per-take frame capture as (bgr, timestamp) for real-speed GIF /
        # clip-picker playback.
        self._active_take_frames: list = []
        self._take_clips: list = []

        # Deferred camera open (see static recorder for the same
        # pattern — prevents the dialog from freezing on a slow phone
        # IP-camera handshake).
        self._cap = None
        self._owns_camera = False
        self._using_worker = False
        self._poll_timer: Optional[QTimer] = None
        self._init_error: Optional[str] = None
        self._camera_connect_attempted = False

        self._build_ui()

        # Periodic progress refresh so the "0.4 / 1.5 s" elapsed-time
        # display updates even when no new frame arrives this tick.
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(80)
        self._progress_timer.timeout.connect(self._refresh_status)
        self._progress_timer.start()

    def _build_recorder(self) -> DynamicGestureRecorder:
        return DynamicGestureRecorder(
            self._duration_mode,
            target_takes=DEFAULT_TARGET_TAKES,
            on_state_changed=self._on_recorder_state_changed,
            on_take_completed=self._on_take_completed,
        )

    def showEvent(self, event):  # noqa: N802 (Qt API name)
        super().showEvent(event)
        try:
            apply_touchless_titlebar(self)
        except Exception:
            pass
        if not self._camera_connect_attempted:
            self._camera_connect_attempted = True
            QTimer.singleShot(0, self._deferred_connect)
        if not getattr(self, "_consistency_tip_shown", False):
            self._consistency_tip_shown = True
            tip = getattr(self, "_consistency_tip", None)
            if tip is not None:
                QTimer.singleShot(0, lambda: tip.attach(self._video_label))

    def _deferred_connect(self) -> None:
        self._video_label.setText("Connecting to camera...")
        self._video_label.repaint()
        self._connect_worker()
        if self._init_error:
            err = self._init_error
            self._init_error = None
            touchless_message_box(
                self, "Camera unavailable", err,
                icon=QMessageBox.Critical, buttons=QMessageBox.Ok,
            )

    # ---- UI build ----

    def _build_ui(self) -> None:
        self.setStyleSheet(
            f"""
            QDialog {{ background: #0E1822; }}
            QLabel {{ color: #DCE9F2; }}
            QPushButton {{
                background: rgba(255,255,255,0.08);
                color: #E5F6FF;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 8px;
                padding: 8px 18px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background: rgba(255,255,255,0.14); }}
            QPushButton#beginBtn, QPushButton#saveBtn {{
                background: {self._accent_color};
                color: #0B1620;
                font-weight: 800;
            }}
            QPushButton#beginBtn:hover, QPushButton#saveBtn:hover {{
                background: #FFFFFF;
            }}
            QPushButton#stopBtn {{
                background: #DC2626;
                color: #FFFFFF;
                font-weight: 800;
            }}
            QPushButton#stopBtn:hover {{ background: #EF4444; }}
            """
        )

        root = QVBoxLayout(self._body)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        cap = self._duration_mode.auto_stop_seconds
        if cap is None:
            cap_hint = "press <b>Stop</b> to finish a take"
        else:
            cap_hint = f"each take auto-stops after <b>{cap:g} s</b>"
        self._help = ExpandableHelpPanel(
            summary_html=(
                f"{_DYNAMIC_SUMMARY} Perform the gesture "
                f"<b>{DEFAULT_TARGET_TAKES}</b> times — {cap_hint}. "
                f"Click <b>Begin Recording</b> or press <b>Spacebar</b>."
            ),
            details_html=_DYNAMIC_DETAILS,
        )
        root.addWidget(self._help)

        self._instructions = QLabel("")
        self._instructions.hide()
        root.addWidget(self._instructions)

        self._video_label = QLabel("Waiting for camera frames...")
        self._video_label.setMinimumHeight(420)
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet(
            "QLabel { background: #050A0F; color: #5C6F7E; border-radius: 8px; }"
        )
        root.addWidget(self._video_label, 1)

        # "Recording complete!" overlay — mirrors the static recorder's
        # unmissable green pill once all takes have been captured.
        self._complete_overlay = QLabel("Recording complete!", self._video_label)
        self._complete_overlay.setObjectName("recordingCompleteOverlay")
        self._complete_overlay.setAlignment(Qt.AlignCenter)
        self._complete_overlay.setStyleSheet(
            "QLabel#recordingCompleteOverlay {"
            "  background: rgba(7, 30, 22, 0.78);"
            "  color: #1DE9B6;"
            "  border: 2px solid rgba(29, 233, 182, 0.55);"
            "  border-radius: 18px;"
            "  font-size: 38px;"
            "  font-weight: 900;"
            "  padding: 22px 36px;"
            "  letter-spacing: 0.5px;"
            "}"
        )
        self._complete_overlay.hide()

        self._consistency_tip = RecordingConsistencyTip(self._video_label)

        self._progress_label = QLabel(
            f"Take 0 / {DEFAULT_TARGET_TAKES}"
        )
        self._progress_label.setStyleSheet("color: #9FB3C2; font-size: 12px;")
        root.addWidget(self._progress_label)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        self._exit_button = QPushButton("Exit")
        self._exit_button.clicked.connect(self.reject)
        self._exit_button.setFocusPolicy(Qt.NoFocus)
        button_row.addWidget(self._exit_button)

        self._restart_button = QPushButton("Restart")
        self._restart_button.clicked.connect(self._on_restart)
        self._restart_button.setFocusPolicy(Qt.NoFocus)
        self._restart_button.hide()
        button_row.addWidget(self._restart_button)

        self._begin_button = QPushButton("Begin Recording")
        self._begin_button.setObjectName("beginBtn")
        self._begin_button.clicked.connect(self._on_begin_or_stop_clicked)
        self._begin_button.setFocusPolicy(Qt.NoFocus)
        button_row.addWidget(self._begin_button)

        self._save_button = QPushButton("Save")
        self._save_button.setObjectName("saveBtn")
        self._save_button.clicked.connect(self._on_save)
        self._save_button.setFocusPolicy(Qt.NoFocus)
        self._save_button.hide()
        button_row.addWidget(self._save_button)

        root.addLayout(button_row)

        self._space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._space_shortcut.setContext(Qt.WindowShortcut)
        self._space_shortcut.activated.connect(self._on_begin_or_stop_clicked)

        self._refresh_status()

    def _position_complete_overlay(self) -> None:
        try:
            ow = max(self._complete_overlay.sizeHint().width(), 360)
            oh = max(self._complete_overlay.sizeHint().height(), 80)
            x = max(0, (self._video_label.width() - ow) // 2)
            y = max(0, (self._video_label.height() - oh) // 2)
            self._complete_overlay.setGeometry(x, y, ow, oh)
        except Exception:
            pass

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_complete_overlay()

    # ---- worker / camera ----

    def _connect_worker(self) -> None:
        if self._worker is not None and bool(getattr(self._worker, "is_running", False)):
            try:
                self._worker.raw_frame_ready.connect(self._on_frame)
                self._using_worker = True
                return
            except Exception:
                self._using_worker = False
        try:
            from .custom_gestures_recorder import RecordingWindow
            cap = RecordingWindow._open_configured_camera(self)  # noqa: SLF001
        except Exception as exc:
            self._init_error = f"Camera open failed: {exc}"
            return
        if cap is None:
            self._init_error = (
                "Could not open the camera for recording. If you're using a "
                "phone camera, make sure your phone is on the same Wi-Fi "
                "network and the QR pairing is still active. Otherwise make "
                "sure no other app is holding the webcam."
            )
            return
        self._cap = cap
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(33)
        self._poll_timer.timeout.connect(self._pull_own_frame)
        self._poll_timer.start()

    def _pull_own_frame(self) -> None:
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        self._on_frame(frame)

    def _lookup_main_window_qr_capture(self):
        # Mirror of RecordingWindow._lookup_main_window_qr_capture so
        # the borrowed _open_configured_camera path resolves cleanly
        # when invoked with self = DynamicGestureRecorderWindow.
        if self._config is not None and not bool(
            getattr(self._config, "phone_camera_qr_active", False)
        ):
            return None
        node = self.parent()
        for _ in range(8):
            if node is None:
                return None
            server = getattr(node, "_phone_camera_qr_server", None)
            if server is not None:
                try:
                    cap = server.capture
                except Exception:
                    cap = None
                if cap is not None:
                    try:
                        if cap.isOpened():
                            return cap
                    except Exception:
                        return None
                return None
            node = node.parent() if hasattr(node, "parent") else None
        return None

    # ---- frame intake ----

    def _on_frame(self, frame, capture_ts: float = 0.0) -> None:
        if frame is None:
            return
        try:
            np_frame = frame if isinstance(frame, np.ndarray) else np.asarray(frame)
            if np_frame.ndim != 3 or np_frame.shape[2] not in (3, 4):
                return
            should_flip = self._owns_camera and not bool(
                getattr(self._config, "camera_source_is_mirrored", False)
            )
            mirrored = cv2.flip(np_frame, 1) if should_flip else np_frame
            rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB) if mirrored.shape[2] == 3 else mirrored[:, :, :3]
            result = self._mp_hands.process(rgb)
            display_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            landmarks = None
            if result.multi_hand_landmarks:
                self._mp_drawer.draw_landmarks(
                    display_bgr,
                    result.multi_hand_landmarks[0],
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self._mp_hand_style,
                    self._mp_conn_style,
                )
                landmarks = _landmarks_array_from_mediapipe(
                    result.multi_hand_landmarks[0].landmark
                )
                self._latest_handedness = None
                try:
                    if result.multi_handedness:
                        self._latest_handedness = str(
                            result.multi_handedness[0].classification[0].label
                        )
                except Exception:
                    self._latest_handedness = None
                if self._recorder.state == RecorderState.ACTIVE:
                    palm = palm_scale_from_landmarks(landmarks)
                    self._recorder.feed_frame(
                        landmarks,
                        palm,
                        timestamp=time.monotonic(),
                        handedness=self._latest_handedness,
                    )
                    if self._latest_handedness in ("Left", "Right"):
                        self._handedness_votes.append(self._latest_handedness)
                    # Snapshot the frame (with skeleton drawn) so we
                    # can pick a representative thumbnail when the
                    # take ends. Copy because OpenCV may reuse buffers.
                    try:
                        now = time.monotonic()
                        if (
                            not self._active_take_frames
                            or (now - self._active_take_frames[-1][1]) >= 0.04
                        ):
                            self._active_take_frames.append(
                                (display_bgr.copy(), now)
                            )
                    except Exception:
                        pass
            else:
                self._latest_handedness = None

            self._draw_state_badge(display_bgr)
            self._update_video_pixmap(display_bgr)
        except Exception as exc:
            print(f"[dynamic-recorder] frame error: {exc}")

    def _draw_state_badge(self, frame: np.ndarray) -> None:
        state = self._recorder.state
        if state == RecorderState.ACTIVE:
            cap = self._recorder.duration_mode.auto_stop_seconds
            elapsed = self._recorder.active_seconds
            if cap is None:
                badge = (
                    f"RECORDING take {self._recorder.completed_takes + 1}/"
                    f"{self._recorder.target_takes}  {elapsed:.1f}s"
                )
            else:
                badge = (
                    f"RECORDING take {self._recorder.completed_takes + 1}/"
                    f"{self._recorder.target_takes}  {elapsed:.1f}/{cap:.1f}s"
                )
            color = (40, 220, 40)
        elif state == RecorderState.COMPLETE:
            badge = "ALL TAKES RECORDED"
            color = (60, 220, 220)
        else:
            done = self._recorder.completed_takes
            total = self._recorder.target_takes
            if done == 0:
                badge = "Click Begin Recording or press Spacebar"
            else:
                badge = (
                    f"Take {done}/{total} captured — "
                    "Click Begin Recording or press Spacebar"
                )
            color = (40, 220, 40)  # bold green prompt
        try:
            cv2.putText(
                frame, badge, (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )
        except Exception:
            pass

    def _update_video_pixmap(self, bgr_frame: np.ndarray) -> None:
        try:
            h, w = bgr_frame.shape[:2]
            label_w = max(1, self._video_label.width())
            label_h = max(1, self._video_label.height())
            scale = min(label_w / float(w), label_h / float(h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(bgr_frame, (new_w, new_h)) if scale != 1.0 else bgr_frame
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
            self._video_label.setPixmap(QPixmap.fromImage(img.copy()))
        except Exception:
            pass

    # ---- UI handlers ----

    def _on_begin_or_stop_clicked(self) -> None:
        state = self._recorder.state
        if state == RecorderState.IDLE:
            self._active_take_frames = []
            if self._recorder.begin_take():
                self._refresh_status()
                self._refresh_buttons()
        elif state == RecorderState.ACTIVE:
            self._recorder.end_take()
            # state-change callback refreshes UI
        # COMPLETE — Begin button is hidden, nothing to do here.

    def _on_restart(self) -> None:
        # Drop everything and start over from take 1.
        self._recorder = self._build_recorder()
        self._handedness_votes.clear()
        self._take_clips.clear()
        self._active_take_frames = []
        self._complete_overlay.hide()
        self._refresh_status()
        self._refresh_buttons()

    def _on_recorder_state_changed(self, state: RecorderState) -> None:
        self._refresh_status()
        self._refresh_buttons()
        if state == RecorderState.COMPLETE:
            self._complete_overlay.show()
            self._position_complete_overlay()
            self._complete_overlay.raise_()

    def _on_take_completed(self, take_number: int, take) -> None:
        # Stash the entire BGR frame sequence for this take so the
        # clip-picker can play it back and so the eventually-chosen
        # take can be saved as the gesture's animated clip on disk.
        frames = self._active_take_frames
        self._active_take_frames = []
        if frames:
            self._take_clips.append(list(frames))

    def _refresh_status(self) -> None:
        state = self._recorder.state
        done = self._recorder.completed_takes
        total = self._recorder.target_takes
        if state == RecorderState.ACTIVE:
            cap = self._recorder.duration_mode.auto_stop_seconds
            elapsed = self._recorder.active_seconds
            if cap is None:
                self._progress_label.setText(
                    f"Recording take {done + 1} / {total} — {elapsed:.1f} s "
                    f"(press Stop / Space to finish)"
                )
            else:
                self._progress_label.setText(
                    f"Recording take {done + 1} / {total} — "
                    f"{elapsed:.1f} / {cap:.1f} s"
                )
        elif state == RecorderState.COMPLETE:
            self._progress_label.setText(
                f"All {total} takes captured. Click Save to finish."
            )
        else:
            if done == 0:
                self._progress_label.setText(
                    f"Take 0 / {total} — ready to record."
                )
            else:
                self._progress_label.setText(
                    f"Take {done} / {total} captured — ready for next take."
                )

    def _refresh_buttons(self) -> None:
        state = self._recorder.state
        if state == RecorderState.ACTIVE:
            self._begin_button.show()
            self._begin_button.setEnabled(True)
            self._begin_button.setObjectName("stopBtn")
            self._begin_button.setText("Stop (Space)")
            self._begin_button.setStyleSheet("")  # let CSS re-apply via objectName
            self._begin_button.style().unpolish(self._begin_button)
            self._begin_button.style().polish(self._begin_button)
            self._save_button.hide()
            self._restart_button.hide()
        elif state == RecorderState.COMPLETE:
            self._begin_button.hide()
            self._save_button.show()
            self._restart_button.show()
        else:
            self._begin_button.show()
            self._begin_button.setEnabled(True)
            self._begin_button.setObjectName("beginBtn")
            next_take = self._recorder.completed_takes + 1
            self._begin_button.setText(
                f"Begin take {next_take} / {self._recorder.target_takes}"
            )
            self._begin_button.setStyleSheet("")
            self._begin_button.style().unpolish(self._begin_button)
            self._begin_button.style().polish(self._begin_button)
            self._save_button.hide()
            self._restart_button.setVisible(
                self._recorder.completed_takes > 0
            )

    # ---- save ----

    def _on_save(self) -> None:
        if self._recorder.completed_takes < self._recorder.target_takes:
            touchless_message_box(
                self, "Not enough takes",
                f"Record all {self._recorder.target_takes} takes before saving.",
                icon=QMessageBox.Warning, buttons=QMessageBox.Ok,
            )
            return
        try:
            artifacts = self._recorder.build_artifacts(gesture_name=self._name)
        except Exception as exc:
            touchless_message_box(
                self, "Could not build gesture",
                f"Key-point selection failed: {exc}",
                icon=QMessageBox.Critical, buttons=QMessageBox.Ok,
            )
            return

        # Clip picker: each tile plays a take back as a small loop so
        # the user picks which TAKE becomes the gesture's canonical
        # motion clip on disk. We save it as an animated GIF (small,
        # native QMovie support) — the gesture card renders the
        # motion preview from this file.
        chosen_image_filename = ""
        registry = GestureRegistry()
        registry.load()
        if self._take_clips:
            accent = self._accent_color
            text = "#E5F6FF"
            surface = "#0F172A"
            try:
                if self._config is not None:
                    accent = self._config.accent_color or accent
                    text = self._config.text_color or text
                    surface = self._config.surface_color or surface
            except Exception:
                pass
            picker = DynamicClipPickerDialog(
                gesture_name=self._name,
                description=self._description or "",
                clips=self._take_clips,
                accent_color=accent,
                text_color=text,
                surface_color=surface,
                parent=self,
            )
            picker.exec()
            chosen_idx = picker.selected_clip_index
            if chosen_idx is not None:
                chosen_image_filename = save_motion_gif(
                    registry, self._take_clips[chosen_idx], self._name
                )

        # Majority vote on handedness.
        handedness = None
        if self._handedness_votes:
            from collections import Counter
            handedness = Counter(self._handedness_votes).most_common(1)[0][0]
            if handedness not in ("Left", "Right"):
                handedness = None

        try:
            registry.add_dynamic(
                name=self._name,
                key_point_indices=artifacts.key_points.indices,
                sample_trajectories=artifacts.template.sample_trajectories,
                # Persist the wrist channel + auto-computed strength so
                # the runtime classifier can do its weighted DTW without
                # rebuilding the template from raw takes. Critical for
                # swipes (high strength → wrist trajectory disambiguates
                # from "hand entered view") and pure-finger gestures
                # (strength ≈ 0 → finger-only matching, no false reject).
                wrist_trajectories=artifacts.template.wrist_trajectories,
                wrist_motion_strength=artifacts.template.wrist_motion_strength,
                # v1.1.8.1: the template's per-gesture match_threshold
                # (derived from intra-take pairwise DTW) rides along
                # so the runtime doesn't have to re-derive it on every
                # startup. wrist_schema=2 marks displacement semantics
                # (the recorder's build_template_from_takes stored
                # displacement, not absolute position).
                match_threshold=artifacts.template.match_threshold,
                wrist_schema=2,
                # v1.1.8.2: SPRING streaming features so the new
                # template fires at motion peak instead of settle.
                sample_features=artifacts.template.sample_features,
                # v1.1.8.2 (post-audit r2) intent signature — "the
                # essential motion" (direction + magnitude across all
                # keypoints). Fires reliably when SPRING's exact-shape
                # matching is too strict.
                intent_direction=artifacts.template.intent_direction,
                intent_magnitude=artifacts.template.intent_magnitude,
                intent_window_seconds=artifacts.template.intent_window_seconds,
                intent_fingertip_extension=(
                    artifacts.template.intent_fingertip_extension
                ),
                action=self._action,
                description=self._description,
                handedness=handedness,
                duration_mode=self._recorder.duration_mode.value,
                image_filename=chosen_image_filename,
                overwrite=True,
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

    # ---- cleanup ----

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            if self._using_worker and self._worker is not None:
                try:
                    self._worker.raw_frame_ready.disconnect(self._on_frame)
                except Exception:
                    pass
            if self._cap is not None and self._owns_camera:
                try:
                    self._cap.release()
                except Exception:
                    pass
            if self._poll_timer is not None:
                self._poll_timer.stop()
            self._progress_timer.stop()
            try:
                self._mp_hands.close()
            except Exception:
                pass
        finally:
            super().closeEvent(event)


# ---------------------------------------------------------------------------
# Clip picker for the post-save flow.


class _ClipTile(QWidget):
    """One clip-picker tile. Plays take frames at real capture speed,
    freezes 2s on the last frame, then loops (no blank flash)."""

    clicked = Signal(int)  # emits the tile index

    _SIZE = 200
    _FREEZE_END_MS = 1500

    def __init__(self, index: int, frames, accent: str, parent=None) -> None:
        super().__init__(parent)
        self._index = index
        # Accept raw BGR or (bgr, ts) pairs.
        self._frames_bgr = []
        stamps = []
        for item in frames or []:
            if isinstance(item, tuple) and len(item) == 2:
                self._frames_bgr.append(item[0])
                stamps.append(float(item[1]))
            else:
                self._frames_bgr.append(item)
        if len(stamps) == len(self._frames_bgr) and len(stamps) >= 2:
            self._durations = durations_ms_from_timestamps(
                stamps, freeze_end_ms=self._FREEZE_END_MS,
            )
        else:
            n = len(self._frames_bgr)
            self._durations = (
                [33] * max(0, n - 1) + [self._FREEZE_END_MS]
            ) if n else []
        self._accent = accent
        self._frame_idx = 0
        self._selected = False
        self.setFixedSize(self._SIZE, self._SIZE + 24)
        self.setCursor(Qt.PointingHandCursor)
        self._image_label = QLabel(self)
        self._image_label.setGeometry(0, 0, self._SIZE, self._SIZE)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet(
            "QLabel {"
            "  background: rgba(0,0,0,0.4);"
            "  border-radius: 10px;"
            "  border: 2px solid rgba(255,255,255,0.10);"
            "}"
        )
        self._caption = QLabel(f"Take {index + 1}", self)
        self._caption.setGeometry(0, self._SIZE + 2, self._SIZE, 20)
        self._caption.setAlignment(Qt.AlignCenter)
        self._caption.setStyleSheet("color: #94A3B8; font-size: 11px;")

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)
        self._render_current_frame()
        self._schedule_next()

    def set_selected(self, value: bool) -> None:
        if value == self._selected:
            return
        self._selected = value
        if value:
            self._image_label.setStyleSheet(
                "QLabel {"
                "  background: rgba(0,0,0,0.4);"
                "  border-radius: 10px;"
                f"  border: 2px solid {self._accent};"
                "}"
            )
            self._caption.setStyleSheet(
                f"color: {self._accent}; font-size: 11px; font-weight: 700;"
            )
        else:
            self._image_label.setStyleSheet(
                "QLabel {"
                "  background: rgba(0,0,0,0.4);"
                "  border-radius: 10px;"
                "  border: 2px solid rgba(255,255,255,0.10);"
                "}"
            )
            self._caption.setStyleSheet("color: #94A3B8; font-size: 11px;")

    def stop(self) -> None:
        self._timer.stop()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._index)
            event.accept()
        else:
            super().mousePressEvent(event)

    def _schedule_next(self) -> None:
        if not self._frames_bgr or not self._durations:
            return
        ms = self._durations[min(self._frame_idx, len(self._durations) - 1)]
        self._timer.start(max(20, int(ms)))

    def _advance(self) -> None:
        if not self._frames_bgr:
            return
        self._frame_idx = (self._frame_idx + 1) % len(self._frames_bgr)
        self._render_current_frame()
        self._schedule_next()

    def _render_current_frame(self) -> None:
        if not self._frames_bgr:
            return
        try:
            bgr = self._frames_bgr[self._frame_idx]
            h, w = bgr.shape[:2]
            scale = min((self._SIZE - 8) / float(w), (self._SIZE - 8) / float(h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(bgr, (new_w, new_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
            self._image_label.setPixmap(QPixmap.fromImage(img.copy()))
        except Exception:
            pass


class DynamicClipPickerDialog(QDialog):
    """Grid of looping take-clips. The user picks the take whose
    motion they want stored on disk as the gesture's preview clip.
    Skip is allowed — the gesture still saves, just without a clip
    image. Keeps chrome consistent with the rest of the app via
    `apply_touchless_chrome`."""

    def __init__(
        self,
        *,
        gesture_name: str,
        description: str,
        clips: list,
        accent_color: str,
        text_color: str,
        surface_color: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick a clip for this gesture")
        self.setObjectName("dynamicClipPicker")
        self.setModal(True)
        self.setMinimumWidth(720)
        # r51: install_indigo_chrome for Win10 + Win11 parity.
        self._body = install_indigo_chrome(self, "Pick a clip for this gesture")
        self._clips = list(clips)
        self._accent = accent_color or "#1DE9B6"
        self._text = text_color or "#E5F6FF"
        self._surface = surface_color or "#0F172A"
        self._tiles: list[_ClipTile] = []
        self._selected_index: Optional[int] = None
        self._build(gesture_name, description)
        self._apply_theme()

    @property
    def selected_clip_index(self) -> Optional[int]:
        return self._selected_index

    def _build(self, gesture_name: str, description: str) -> None:
        from PySide6.QtWidgets import QGridLayout

        root = QVBoxLayout(self._body)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        title = QLabel(f"Pick a clip for “{gesture_name}”")
        title.setObjectName("clipPickerTitle")
        title.setWordWrap(True)
        root.addWidget(title)

        if description:
            desc = QLabel(description)
            desc.setObjectName("clipPickerDesc")
            desc.setWordWrap(True)
            root.addWidget(desc)

        hint = QLabel(
            "Each tile loops one take. Pick the take you want saved as "
            "the gesture's motion preview."
        )
        hint.setObjectName("clipPickerHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        grid_host = QWidget()
        grid_host.setObjectName("clipPickerGrid")
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        root.addWidget(grid_host)

        if not self._clips:
            empty = QLabel(
                "No clips were captured during this recording. You can "
                "still save without a preview."
            )
            empty.setWordWrap(True)
            empty.setObjectName("clipPickerHint")
            grid.addWidget(empty, 0, 0)
        for i, frames in enumerate(self._clips):
            tile = _ClipTile(i, frames, accent=self._accent, parent=grid_host)
            tile.clicked.connect(self._on_tile_clicked)
            grid.addWidget(tile, i // 3, i % 3)
            self._tiles.append(tile)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self._skip_button = QPushButton("Skip — no clip")
        self._skip_button.setObjectName("clipPickerSecondary")
        self._skip_button.clicked.connect(self.reject)
        button_row.addWidget(self._skip_button)
        self._confirm_button = QPushButton("Use Selected Clip")
        self._confirm_button.setObjectName("clipPickerPrimary")
        self._confirm_button.setEnabled(False)
        self._confirm_button.clicked.connect(self.accept)
        button_row.addWidget(self._confirm_button)
        root.addLayout(button_row)

    def _on_tile_clicked(self, idx: int) -> None:
        self._selected_index = idx
        for i, tile in enumerate(self._tiles):
            tile.set_selected(i == idx)
        self._confirm_button.setEnabled(True)

    def closeEvent(self, event) -> None:  # noqa: N802
        for tile in self._tiles:
            tile.stop()
        super().closeEvent(event)

    def _apply_theme(self) -> None:
        accent = self._accent
        text = self._text
        surface = self._surface
        self.setStyleSheet(
            f"""
            QDialog#dynamicClipPicker {{
                background-color: {surface};
                color: {text};
                border: 1px solid rgba(29, 233, 182, 0.30);
            }}
            QLabel#clipPickerTitle {{
                color: {accent};
                font-size: 18px;
                font-weight: 800;
                background: transparent;
            }}
            QLabel#clipPickerDesc {{
                color: {text};
                font-size: 13px;
                background: rgba(255, 255, 255, 0.04);
                border-radius: 8px;
                padding: 8px 12px;
            }}
            QLabel#clipPickerHint {{
                color: rgba(229, 246, 255, 0.75);
                font-size: 12px;
                background: transparent;
            }}
            QPushButton#clipPickerPrimary {{
                background-color: rgba(29, 233, 182, 0.18);
                color: {accent};
                border: 1px solid {accent};
                border-radius: 10px;
                padding: 9px 22px;
                font-weight: 700;
                min-width: 160px;
            }}
            QPushButton#clipPickerPrimary:disabled {{
                background-color: rgba(255, 255, 255, 0.04);
                color: rgba(229, 246, 255, 0.30);
                border: 1px solid rgba(255, 255, 255, 0.10);
            }}
            QPushButton#clipPickerSecondary {{
                background-color: rgba(255, 255, 255, 0.06);
                color: {text};
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 10px;
                padding: 9px 18px;
                min-width: 130px;
            }}
            """
        )

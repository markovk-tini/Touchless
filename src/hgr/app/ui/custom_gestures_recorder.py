"""Recording window for Custom Gestures.

Receives raw camera frames from the running GestureWorker via its
`raw_frame_ready` signal — no camera contention with the main pipeline.
Runs MediaPipe locally on each frame (cheap; the pipeline is already
processing frames anyway, this just extracts landmarks for our purposes).

Layout:
  - Camera feed (mirrored, with hand landmarks drawn)
  - Top-right: live finger-state overlay (curl / spread classes)
  - Bottom-right buttons:
      Before recording: [Exit]  [Begin Recording]
      After recording:  [Exit]  [Restart]  [Save]

Capture parameters mirror the standalone trainer: 100 samples, paced
~3 frames apart so recording stretches over ~10 seconds at 30 fps.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
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

from hgr.custom_gestures.builtin_profiles import find_matching_builtin
from hgr.custom_gestures.classifier import GestureClassifier
from hgr.custom_gestures.description import (
    format_gesture_summary,
    live_signature,
    short_curl_label,
    short_spread_label,
)
from hgr.custom_gestures.recorder import (
    GestureRecorder,
    augment_samples,
    landmarks_from_mediapipe,
    normalize_landmarks,
)
from hgr.custom_gestures.registry import (
    Action,
    CustomGesture,
    GestureRegistry,
)

from .custom_gestures_chrome import apply_touchless_titlebar


_TARGET_SAMPLES = 100
_CAPTURE_INTERVAL_FRAMES = 3
_STABILITY_FRAMES = 3
_POSE_CONFLICT_THRESHOLD = 0.85
_POSE_CONFLICT_FRACTION = 0.30  # >30% of samples score above threshold for an existing gesture → conflict


class RecordingWindow(QDialog):
    """Dialog that captures landmark samples for a new custom gesture
    using frames from the running GestureWorker.
    """

    saved = Signal(str)  # emits gesture name on successful save

    def __init__(
        self,
        worker,
        accent_color: str,
        name: str,
        description: str,
        action: Action,
        parent: Optional[QWidget] = None,
        config=None,
    ) -> None:
        super().__init__(parent)
        # r51: install_indigo_chrome for Win10 + Win11 parity.
        from .window_chrome import install_indigo_chrome
        self.setWindowTitle(f"Recording: {name}")
        self.setModal(True)
        self.setMinimumSize(820, 560)
        self._body = install_indigo_chrome(self, f"Recording: {name}")
        self._worker = worker
        self._accent_color = accent_color
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

        self._recorder = GestureRecorder(target_samples=_TARGET_SAMPLES)
        self._state = "idle"  # "idle" | "recording" | "complete"
        self._frames_since_capture = 0
        self._stable_frames = 0
        self._latest_feats: Optional[np.ndarray] = None
        self._latest_sig: dict = {}
        # Per-sample handedness votes during recording. The frame is
        # cv2.flipped before MediaPipe (project convention), so MP's
        # labels are already user-perspective — store directly. The
        # MAJORITY vote at finalize-time becomes the gesture's stored
        # handedness, used by both the conflict check and the live
        # runner so it only fires on the matching hand.
        self._handedness_votes: List[str] = []
        self._latest_handedness: Optional[str] = None
        # Candidate thumbnail frames captured at sample indices 10, 20,
        # 30, 40, 50, 60, 70, 80, 90 — cropped to ~2x the hand bbox so
        # the user gets a clean, hand-centered preview to pick from.
        # Each entry is (sample_index, BGR ndarray). Reset whenever a
        # new recording starts.
        self._candidate_thumbnails: list[tuple[int, np.ndarray]] = []
        self._thumbnail_capture_indices = (10, 20, 30, 40, 50, 60, 70, 80, 90)
        # Selected thumbnail filename after the picker dialog runs.
        # Empty string if the user skipped the picker.
        self._selected_thumbnail_filename: str = ""

        # Set up empty state — actual camera open is deferred to
        # showEvent so the dialog renders immediately. Otherwise a slow
        # phone-camera connection (cv2.VideoCapture(url) can block for
        # several seconds, longer if unreachable) prevents the dialog
        # from ever appearing.
        self._owns_camera = False
        self._cap = None
        self._poll_timer: Optional[QTimer] = None
        self._using_worker = False
        self._init_error: Optional[str] = None
        self._camera_connect_attempted = False

        self._build()

    def showEvent(self, event):  # noqa: N802 (Qt API name)
        super().showEvent(event)
        try:
            apply_touchless_titlebar(self)
        except Exception:
            pass
        # Connect to the camera ONCE the dialog is on screen so the user
        # sees "Connecting to camera..." instead of a frozen UI while
        # cv2.VideoCapture(url) does its network handshake.
        if not self._camera_connect_attempted:
            self._camera_connect_attempted = True
            QTimer.singleShot(0, self._deferred_connect)

    def _deferred_connect(self) -> None:
        self._video_label.setText("Connecting to camera...")
        # Repaint immediately so the user sees the message before the
        # potentially-blocking camera open.
        self._video_label.repaint()
        self._connect_worker()
        if self._init_error:
            err = self._init_error
            self._init_error = None
            from .window_chrome import touchless_message_box
            touchless_message_box(
                self, "Camera unavailable", err,
                icon=QMessageBox.Critical, buttons=QMessageBox.Ok,
            )

    # --- UI -------------------------------------------------------------

    def _build(self) -> None:
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
            """
        )

        root = QVBoxLayout(self._body)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        self._instructions = QLabel(
            "Hold your gesture in front of the camera, then click "
            "<b>Begin Recording</b> or press <b>Spacebar</b>. Let your "
            "hand drift naturally during the ~10-second capture so the "
            "classifier learns your real range."
        )
        self._instructions.setWordWrap(True)
        root.addWidget(self._instructions)

        self._video_label = QLabel("Waiting for camera frames...")
        self._video_label.setMinimumHeight(420)
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet(
            "QLabel { background: #050A0F; color: #5C6F7E; border-radius: 8px; }"
        )
        root.addWidget(self._video_label, 1)

        # Big "Recording complete!" overlay that sits centered on top
        # of the camera view once 100 samples have been captured. The
        # smaller instruction line below the camera kept the success
        # state easy to miss; this gives the user an unmissable,
        # immediate confirmation. Auto-hides on Restart and never
        # carries over to a fresh recording session.
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

        self._progress_label = QLabel(
            f"Samples captured: 0 / {_TARGET_SAMPLES}"
        )
        self._progress_label.setStyleSheet("color: #9FB3C2; font-size: 12px;")
        root.addWidget(self._progress_label)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        self._exit_button = QPushButton("Exit")
        self._exit_button.clicked.connect(self.reject)
        # Drop focus on the Exit / Restart / Save buttons so a press
        # of Space-as-shortcut doesn't first fire whichever button
        # happens to have keyboard focus (Exit is the natural first
        # tab target, and pressing Space on a focused QPushButton
        # activates it via Qt's default keyboard click handling).
        self._exit_button.setFocusPolicy(Qt.NoFocus)
        button_row.addWidget(self._exit_button)

        self._restart_button = QPushButton("Restart")
        self._restart_button.clicked.connect(self._on_restart)
        self._restart_button.setFocusPolicy(Qt.NoFocus)
        self._restart_button.hide()
        button_row.addWidget(self._restart_button)

        self._begin_button = QPushButton("Begin Recording")
        self._begin_button.setObjectName("beginBtn")
        self._begin_button.clicked.connect(self._on_begin_recording)
        self._begin_button.setFocusPolicy(Qt.NoFocus)
        button_row.addWidget(self._begin_button)

        self._save_button = QPushButton("Save")
        self._save_button.setObjectName("saveBtn")
        self._save_button.clicked.connect(self._on_save)
        self._save_button.setFocusPolicy(Qt.NoFocus)
        self._save_button.hide()
        button_row.addWidget(self._save_button)

        root.addLayout(button_row)

        # Window-scoped Space shortcut that fires regardless of which
        # widget currently has focus inside the dialog. The earlier
        # keyPressEvent override didn't fire when a button (e.g.,
        # Exit) had focus, because Qt's default handler activates the
        # focused button on Space before the dialog's keyPressEvent
        # runs. The shortcut sits above per-widget handling, so it
        # always reaches our handler first.
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._space_shortcut.setContext(Qt.WindowShortcut)
        self._space_shortcut.activated.connect(self._on_space_pressed)

    # --- worker connection ----------------------------------------------

    def _connect_worker(self) -> None:
        """Two-mode frame source:
          1. If the running GestureWorker exposes raw_frame_ready and is
             currently emitting frames, subscribe to it (no camera
             contention with the main pipeline).
          2. Otherwise open our own cv2.VideoCapture and pump frames via
             a QTimer. This makes the dialog usable even when the user
             hasn't started the main live viewer yet.

        Init errors (camera open failures, unexpected exceptions) are
        captured and surfaced after the dialog is shown rather than
        before — showing a QMessageBox during __init__ before the parent
        QDialog has been show()n can swallow the dialog completely.
        """
        self._owns_camera = False
        self._cap = None
        self._poll_timer: Optional[QTimer] = None
        self._using_worker = False
        self._init_error: Optional[str] = None

        # Only piggy-back on the worker's frame stream if the worker is
        # ACTUALLY running (camera open + producing frames). The
        # GestureWorker instance exists from app startup but
        # `is_running` is False until the user presses Start in the
        # main window — in that idle state we have to open our own
        # camera, otherwise the dialog hangs on "Connecting...".
        if self._worker is not None and bool(getattr(self._worker, "is_running", False)):
            try:
                self._worker.raw_frame_ready.connect(self._on_frame)
                self._using_worker = True
                return
            except Exception:
                self._using_worker = False

        # Worker not available — open whatever camera Touchless is
        # configured to use (phone IP camera if the user has paired
        # their phone, otherwise their preferred webcam index).
        try:
            cap = self._open_configured_camera()
        except Exception as exc:
            print(f"[custom-gestures] camera open threw: {exc}")
            self._init_error = (
                f"Camera open raised an exception ({type(exc).__name__}: "
                f"{exc}). Check the terminal for the traceback."
            )
            return
        if cap is None:
            self._init_error = (
                "Could not open the camera for recording. If you're using "
                "a phone camera, make sure your phone is on the same Wi-Fi "
                "network and the QR pairing is still active. If you're "
                "using a webcam, make sure no other app (or the main "
                "Touchless live view) is holding it."
            )
            return
        self._cap = cap
        # _owns_camera was set by _open_configured_camera (True if we
        # opened it ourselves, False if we're borrowing the worker's
        # phone QR capture). Don't overwrite it here.
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(33)  # ~30 FPS
        self._poll_timer.timeout.connect(self._pull_own_frame)
        self._poll_timer.start()

    def _pull_own_frame(self) -> None:
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        # _on_frame expects a BGR numpy array — that's what cv2.read
        # returns directly, so pass through.
        self._on_frame(frame)

    def _open_configured_camera(self):
        """Open the camera Touchless is configured to use, mirroring the
        priority order GestureWorker._open_camera uses:

          1. Worker's QR-paired phone capture (engine running with QR).
          2. Main window's QR server capture (QR paired + active, but
             engine not running yet — capture is still live because the
             phone camera server auto-starts at app launch when paired).
          3. URL-based phone camera (`phone_camera_enabled` + URL set).
          4. Preferred / first-available webcam.

        Priorities 1 & 2 are BORROWED — `_owns_camera` stays False so
        teardown doesn't release a capture another component owns.
        """
        try:
            from hgr.app.camera.camera_utils import (
                open_phone_camera_url,
                open_preferred_or_first_available,
            )
        except Exception:
            return None

        # 1. Borrow the worker's QR-paired phone capture if present.
        if self._worker is not None:
            phone_qr = getattr(self._worker, "_phone_camera_capture", None)
            if phone_qr is not None:
                try:
                    if phone_qr.isOpened():
                        return phone_qr
                except Exception:
                    pass

        # 2. Borrow the main-window's QR server capture if the user has
        # paired their phone but hasn't started the live engine yet. The
        # PhoneCameraServer is auto-started at app launch when
        # phone_camera_qr_paired is True, so this capture is already
        # receiving frames from the phone WebSocket.
        qr_capture = self._lookup_main_window_qr_capture()
        if qr_capture is not None:
            return qr_capture

        # 3. URL-based phone camera (legacy/manual phone-URL setup).
        if self._config is not None:
            phone_enabled = bool(getattr(self._config, "phone_camera_enabled", False))
            phone_url = str(getattr(self._config, "phone_camera_url", "") or "").strip()
            if phone_enabled and phone_url:
                try:
                    _info, cap = open_phone_camera_url(phone_url)
                except Exception:
                    cap = None
                if cap is not None:
                    self._owns_camera = True
                    return cap

        # 4. Webcam fallback.
        preferred_index = None
        max_idx = 8
        if self._config is not None:
            preferred_index = getattr(self._config, "preferred_camera_index", None)
            try:
                max_idx = int(getattr(self._config, "camera_scan_limit", 8))
            except Exception:
                max_idx = 8
        try:
            _info, cap = open_preferred_or_first_available(preferred_index, max_index=max_idx)
        except Exception:
            cap = None
        if cap is not None:
            self._owns_camera = True
        return cap

    def _lookup_main_window_qr_capture(self):
        """Walk the parent chain looking for a main_window-style object
        that exposes `_phone_camera_qr_server`. Returns the running
        capture if `phone_camera_qr_active` is enabled, else None."""
        # Only honor the QR capture if the user has it actively selected.
        if self._config is not None and not bool(
            getattr(self._config, "phone_camera_qr_active", False)
        ):
            return None
        node = self.parent()
        for _ in range(8):  # bounded walk; main_window is usually 1 hop up
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

    def _disconnect_worker(self) -> None:
        if self._using_worker and self._worker is not None:
            try:
                self._worker.raw_frame_ready.disconnect(self._on_frame)
            except Exception:
                pass
            self._using_worker = False
        if self._poll_timer is not None:
            try:
                self._poll_timer.stop()
            except Exception:
                pass
            self._poll_timer = None
        if self._cap is not None:
            # Only release if we own it — borrowed worker captures stay alive.
            if self._owns_camera:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = None
        self._owns_camera = False

    # --- frame handling -------------------------------------------------

    def _on_frame(self, frame, capture_ts: float = 0.0) -> None:
        if frame is None:
            return
        try:
            np_frame = frame
            if not isinstance(np_frame, np.ndarray):
                # Some pipelines emit a special wrapper; convert defensively.
                try:
                    np_frame = np.asarray(np_frame)
                except Exception:
                    return
            if np_frame.ndim != 3 or np_frame.shape[2] not in (3, 4):
                return
            # When the recorder owns its own VideoCapture we mirror the
            # raw frame to selfie view (what users expect). When we're
            # borrowing the worker's frames they've already been mirrored
            # by the engine, so flipping again would flip them back to
            # camera-perspective — skip in that case.
            should_flip = self._owns_camera and not bool(
                getattr(self._config, "camera_source_is_mirrored", False)
            )
            mirrored = cv2.flip(np_frame, 1) if should_flip else np_frame
            # Worker may emit BGR; MediaPipe wants RGB.
            rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB) if mirrored.shape[2] == 3 else mirrored[:, :, :3]
            result = self._mp_hands.process(rgb)
            display_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            hand_present = bool(result.multi_hand_landmarks)
            if hand_present:
                self._mp_drawer.draw_landmarks(
                    display_bgr,
                    result.multi_hand_landmarks[0],
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self._mp_hand_style,
                    self._mp_conn_style,
                )
                lm = landmarks_from_mediapipe(
                    result.multi_hand_landmarks[0].landmark
                )
                self._latest_feats = normalize_landmarks(lm)
                self._latest_sig = live_signature(self._latest_feats)
                self._stable_frames = min(self._stable_frames + 1, _STABILITY_FRAMES * 2)
                # Pull per-frame handedness from MediaPipe. Frame was
                # cv2.flipped before MP.process(), so the label is
                # already user-perspective ("Left"/"Right") — never
                # invert it. (See project memory: handedness convention.)
                self._latest_handedness = None
                try:
                    if result.multi_handedness:
                        self._latest_handedness = str(
                            result.multi_handedness[0].classification[0].label
                        )
                except Exception:
                    pass
            else:
                self._latest_sig = {}
                self._latest_feats = None
                self._stable_frames = 0
                self._latest_handedness = None

            # Capture sample if we're recording AND stable AND interval elapsed.
            if (
                self._state == "recording"
                and hand_present
                and self._stable_frames >= _STABILITY_FRAMES
                and self._frames_since_capture >= _CAPTURE_INTERVAL_FRAMES
            ):
                self._recorder.capture(lm)
                self._frames_since_capture = 0
                # Vote this sample's handedness if we got a label this
                # frame. Majority wins at finalize.
                if self._latest_handedness in ("Left", "Right"):
                    self._handedness_votes.append(self._latest_handedness)
                # Snapshot a hand-centered thumbnail at preset sample
                # indices so the user can later pick one to represent
                # the gesture in the dropdown / Gesture Binds preview.
                # Crop from display_bgr (which already has the live
                # skeleton drawn on it) and overlay a green bbox so the
                # thumbnails match the look of the preset gesture
                # cards in the Control Guide.
                if self._recorder.count in self._thumbnail_capture_indices:
                    crop = self._crop_hand_thumbnail(
                        display_bgr,
                        lm,
                        draw_bbox=True,
                        handedness=self._latest_handedness,
                        gesture_name=self._name,
                    )
                    if crop is not None:
                        self._candidate_thumbnails.append(
                            (int(self._recorder.count), crop)
                        )
                self._progress_label.setText(
                    f"Samples captured: {self._recorder.count} / {_TARGET_SAMPLES}"
                )
                if self._recorder.count >= _TARGET_SAMPLES:
                    self._on_recording_complete()
            else:
                self._frames_since_capture += 1

            self._draw_overlay(display_bgr)
            self._render(display_bgr)
        except Exception:
            # A bad frame should not kill the recording loop.
            pass

    def _draw_overlay(self, frame: np.ndarray) -> None:
        # State badge top-left.
        if self._state == "recording":
            badge = f"RECORDING  {self._recorder.count}/{_TARGET_SAMPLES}"
            color = (40, 220, 40)
        elif self._state == "complete":
            badge = "RECORDING COMPLETE"
            color = (60, 220, 220)
        else:
            badge = "Ready — click Begin Recording or press Spacebar"
            color = (220, 220, 220)
        cv2.putText(
            frame, badge, (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )

        # Live hand badge under the state badge — shows the user
        # exactly which hand the system is detecting RIGHT NOW, plus
        # (during/after recording) the running majority of votes so
        # they can see the value that'll be persisted on save.
        if self._latest_handedness in ("Left", "Right") or self._handedness_votes:
            live = self._latest_handedness or "?"
            running = self._running_majority_label()
            if running and self._handedness_votes:
                hand_text = f"Hand: {live}    Recorded: {running}"
            else:
                hand_text = f"Hand: {live}"
            hand_color = (60, 200, 240) if live == "Left" else (240, 180, 60) if live == "Right" else (180, 180, 180)
            cv2.putText(
                frame, hand_text, (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_color, 2, cv2.LINE_AA,
            )

        # Live finger-state overlay top-right.
        sig = self._latest_sig
        if sig and self._latest_feats is not None:
            ext = self._latest_feats[66:71]
            h, w = frame.shape[:2]
            box_x = w - 240
            box_y = 8
            line_h = 18
            cv2.rectangle(
                frame, (box_x - 4, box_y - 2),
                (w - 4, box_y + line_h * 7 + 6),
                (0, 0, 0), -1,
            )
            cv2.putText(
                frame, "Live finger state:",
                (box_x, box_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
            )
            fingers = [
                ("Thumb", sig["thumb_curl"], float(ext[0])),
                ("Index", sig["index_curl"], float(ext[1])),
                ("Mid", sig["middle_curl"], float(ext[2])),
                ("Ring", sig["ring_curl"], float(ext[3])),
                ("Pinky", sig["pinky_curl"], float(ext[4])),
            ]
            for i, (fname, c, dist) in enumerate(fingers):
                col = self._color_for_curl(c)
                cv2.putText(
                    frame,
                    f"{fname:<5} {short_curl_label(c)[:9]:<9} c{c} d={dist:.2f}",
                    (box_x, box_y + 12 + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA,
                )
            spread_c = sig["spread"]
            cv2.putText(
                frame,
                f"Spread {short_spread_label(spread_c)} ({spread_c})",
                (box_x, box_y + 12 + 6 * line_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 80), 1, cv2.LINE_AA,
            )

    def _color_for_curl(self, c: int) -> tuple:
        return (
            (60, 220, 60),
            (60, 220, 180),
            (60, 180, 220),
            (140, 120, 220),
            (200, 80, 200),
        )[max(0, min(4, int(c)))]

    def _render(self, frame_bgr: np.ndarray) -> None:
        # Convert BGR -> RGB and into a QImage scaled to the label size.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self._video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    # --- state transitions ----------------------------------------------

    def _on_begin_recording(self) -> None:
        self._state = "recording"
        self._recorder.reset()
        self._handedness_votes = []
        self._candidate_thumbnails = []
        self._selected_thumbnail_filename = ""
        self._frames_since_capture = 0
        self._begin_button.hide()
        self._exit_button.show()
        self._progress_label.setText(
            f"Samples captured: 0 / {_TARGET_SAMPLES}"
        )

    @staticmethod
    def _crop_hand_thumbnail(
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        draw_bbox: bool = False,
        handedness: Optional[str] = None,
        gesture_name: str = "",
    ) -> Optional[np.ndarray]:
        """Crop a hand-centered thumbnail from a BGR frame using the
        MediaPipe landmarks as a guide. Bbox is expanded slightly
        beyond the natural hand bounds (1.4× side length) — tight
        enough that the hand fills the frame, loose enough that
        fingertips at the edge of the bbox don't get clipped by
        per-frame landmark jitter.

        When draw_bbox=True a green rectangle is drawn around the
        tight hand bounds inside the crop, so the thumbnail mimics
        the look of the preset Control Guide gesture cards (which
        all show a green bbox over the live skeleton). If a
        `handedness` and/or `gesture_name` is provided, a banner is
        drawn above the bbox in the same `Right | gesture_name`
        format the live app uses for preset gestures."""
        try:
            h, w = frame_bgr.shape[:2]
        except Exception:
            return None
        if landmarks is None or h <= 0 or w <= 0:
            return None
        try:
            xs = landmarks[:, 0]
            ys = landmarks[:, 1]
            x_min = float(xs.min()) * w
            x_max = float(xs.max()) * w
            y_min = float(ys.min()) * h
            y_max = float(ys.max()) * h
        except Exception:
            return None
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        side = max(x_max - x_min, y_max - y_min) * 1.4
        if side < 40:
            return None
        half = side * 0.5
        x1 = int(round(max(0, cx - half)))
        x2 = int(round(min(w, cx + half)))
        # Reserve ~32 frame-pixels of headroom ABOVE the hand bbox so
        # the "Left/Right | name" banner always fits cleanly above the
        # green box (instead of being forced inside it on poses where
        # the hand sits high in the frame). Extends y1 upward when the
        # natural square-centered crop wouldn't already provide that
        # gap; never shrinks below the natural top so the bbox itself
        # is never clipped.
        _BANNER_HEADROOM_PX = 36
        natural_y1 = max(0, int(round(cy - half)))
        banner_y1 = max(0, int(round(y_min - _BANNER_HEADROOM_PX)))
        y1 = min(natural_y1, banner_y1)
        y2 = int(round(min(h, cy + half)))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None
        crop = frame_bgr[y1:y2, x1:x2].copy()
        if draw_bbox:
            try:
                pad_w = int(round((x2 - x1) * 0.03))
                pad_h = int(round((y2 - y1) * 0.03))
                rect_x1 = int(round(max(0, x_min - x1 - pad_w)))
                rect_y1 = int(round(max(0, y_min - y1 - pad_h)))
                rect_x2 = int(round(min(crop.shape[1] - 1, x_max - x1 + pad_w)))
                rect_y2 = int(round(min(crop.shape[0] - 1, y_max - y1 + pad_h)))
                if rect_x2 > rect_x1 + 4 and rect_y2 > rect_y1 + 4:
                    # Match the live app's per-hand bbox style:
                    # active-green BGR (130, 220, 70) and a 3-pixel
                    # border. The live view paints in widget-pixel
                    # space at a generous size, so its 2-pixel
                    # painter pen reads thicker than 2 pixels in our
                    # cropped, smaller-resolution thumbnail. Bumping
                    # to 3 px here matches the visual weight of the
                    # preset Control Guide cards so custom-gesture
                    # thumbnails look at home next to them.
                    bbox_color = (130, 220, 70)  # BGR — matches gpu_video_widget active green
                    cv2.rectangle(
                        crop,
                        (rect_x1, rect_y1),
                        (rect_x2, rect_y2),
                        bbox_color,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

                    # Banner: "Right | gesture_name" — mirrors the
                    # live view's per-hand label for preset gestures.
                    # Only the gesture name on its own when there's
                    # no handedness vote yet (rare; usually MP gives
                    # us a label every frame).
                    banner_text = ""
                    if handedness and gesture_name:
                        banner_text = f"{handedness} | {gesture_name}"
                    elif gesture_name:
                        banner_text = gesture_name
                    elif handedness:
                        banner_text = handedness
                    if banner_text:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        font_thickness = 1
                        (text_w, text_h), _ = cv2.getTextSize(
                            banner_text, font, font_scale, font_thickness
                        )
                        # No background or border — just the green
                        # text drawn directly on the camera image.
                        # Place above the bbox; the crop is sized
                        # with explicit headroom so the banner fits
                        # without ever overlapping the box. Falls
                        # back to inside-the-box only on the rare
                        # frame where the hand is right at the top
                        # edge AND headroom couldn't be reserved.
                        text_x = max(2, rect_x1)
                        if rect_y1 - 6 >= text_h:
                            text_y = rect_y1 - 6
                        else:
                            text_y = rect_y1 + text_h + 4
                        # Thin black outline first, then green text
                        # on top — keeps the label readable against
                        # bright backgrounds without bulking up the
                        # stroke.
                        cv2.putText(
                            crop, banner_text, (text_x, text_y),
                            font, font_scale, (0, 0, 0),
                            font_thickness + 1, cv2.LINE_AA,
                        )
                        cv2.putText(
                            crop, banner_text, (text_x, text_y),
                            font, font_scale, bbox_color,
                            font_thickness, cv2.LINE_AA,
                        )
            except Exception:
                pass
        return crop

    def _on_recording_complete(self) -> None:
        self._state = "complete"
        self._instructions.setText(
            "Recording complete. Click <b>Save</b> to keep it, "
            "<b>Restart</b> to re-record, or <b>Exit</b> to discard."
        )
        self._begin_button.hide()
        self._restart_button.show()
        self._save_button.show()
        self._show_complete_overlay()

    def _on_restart(self) -> None:
        self._state = "idle"
        self._recorder.reset()
        self._handedness_votes = []
        self._restart_button.hide()
        self._save_button.hide()
        self._begin_button.show()
        self._complete_overlay.hide()
        self._instructions.setText(
            "Hold your gesture in front of the camera, then click "
            "<b>Begin Recording</b> or press <b>Spacebar</b>."
        )
        self._progress_label.setText(
            f"Samples captured: 0 / {_TARGET_SAMPLES}"
        )

    def _show_complete_overlay(self) -> None:
        self._complete_overlay.adjustSize()
        self._reposition_complete_overlay()
        self._complete_overlay.show()
        self._complete_overlay.raise_()

    def _reposition_complete_overlay(self) -> None:
        if not self._complete_overlay.isVisible() and self._state != "complete":
            return
        try:
            parent_rect = self._video_label.rect()
            label_size = self._complete_overlay.size()
            x = (parent_rect.width() - label_size.width()) // 2
            y = (parent_rect.height() - label_size.height()) // 2
            self._complete_overlay.move(max(0, x), max(0, y))
        except Exception:
            pass

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._reposition_complete_overlay()

    def _on_save(self) -> None:
        from .window_chrome import touchless_message_box
        try:
            originals = self._recorder.finalize()
        except ValueError:
            touchless_message_box(
                self, "No samples", "No samples were captured.",
                icon=QMessageBox.Warning, buttons=QMessageBox.Ok,
            )
            return

        # Majority hand from per-sample MediaPipe labels. With ~100
        # samples a 60% threshold is overwhelmingly clear; below that
        # the user kept switching hands or MP couldn't decide, and we
        # store None so the gesture fires on either hand.
        recorded_hand = self._majority_handedness()

        # Built-in pose conflict — does this recording match the
        # categorical signature of one of Touchless's built-in static
        # gestures wired to THIS hand? Built-ins on the other hand
        # are not a conflict. Built-ins CANNOT be overridden (the
        # app's hardcoded recognizer fires no matter what), so the
        # only sensible response is "pick a different pose".
        builtin = find_matching_builtin(originals, handedness=recorded_hand)
        if builtin is not None:
            touchless_message_box(
                self,
                "Pose already in use",
                f"This gesture pose already exists as <b>{builtin.name}</b> "
                f"({builtin.description}).<br><br>"
                f"Please try using a different pose.",
                icon=QMessageBox.Warning,
                buttons=QMessageBox.Ok,
            )
            return

        # Pose-conflict check: do these new samples overlap with any
        # existing custom gesture's pose ON THE SAME HAND? Build a
        # temp classifier from the OTHER gestures (excluding the one
        # we're saving in case of overwrite). Same-shape gestures on
        # opposite hands are NOT a conflict.
        registry = GestureRegistry()
        registry.load()
        others = [
            g for g in registry.list()
            if g.name != self._name
            # Skip if both have a definite hand and they differ.
            and not (
                g.handedness in ("Left", "Right")
                and recorded_hand in ("Left", "Right")
                and g.handedness != recorded_hand
            )
        ]
        if others:
            clf = GestureClassifier(
                gestures=others,
                threshold=_POSE_CONFLICT_THRESHOLD,
                confidence_margin=0.0,
            )
            clf.reload()
            hits: dict = {}
            for sample in originals:
                m = clf.classify_raw(sample.features)
                if m is not None:
                    prev = hits.get(m.gesture.name, (0, 0.0))
                    hits[m.gesture.name] = (prev[0] + 1, max(prev[1], m.score))
            offenders = [
                (name, count, peak)
                for name, (count, peak) in hits.items()
                if count >= int(len(originals) * _POSE_CONFLICT_FRACTION)
            ]
            if offenders:
                # Take the top match — that's the gesture the user
                # would conflict with most strongly.
                top_name = sorted(offenders, key=lambda x: -x[1])[0][0]
                # Custom gestures CAN be overridden — offer that as the
                # only positive choice. Override deletes the existing
                # gesture so live use only fires the new one.
                # r51: use touchless_message_box (frameless indigo) so
                # the popup renders identically on Win10 and Win11.
                from PySide6.QtWidgets import QMessageBox as _QMB
                from .window_chrome import touchless_message_box
                result = touchless_message_box(
                    self,
                    "Pose already in use",
                    f"This gesture pose already exists as <b>{top_name}</b>.<br><br>"
                    f"Override it? The existing <b>{top_name}</b> will be "
                    f"deleted so this new gesture becomes the sole owner "
                    f"of this pose.",
                    icon=_QMB.Warning,
                    buttons=_QMB.Yes | _QMB.Cancel,
                    default_button=_QMB.Cancel,
                )
                if result != _QMB.Yes:
                    return
                # User chose Override — drop the old gesture so the
                # new one is the sole owner of this pose.
                registry.remove(top_name)

        # Pose picker: present the candidate frames captured during
        # recording so the user can choose which one represents this
        # gesture in the dropdown / Gesture Binds preview. If they
        # skip, image_filename stays empty and the UI falls back to
        # a placeholder. We run this BEFORE registry.add so the saved
        # entry already has its image_filename.
        chosen_image_filename = ""
        if self._candidate_thumbnails:
            accent = "#1DE9B6"
            text = "#E5F6FF"
            surface = "#0F172A"
            try:
                if self._config is not None:
                    accent = self._config.accent_color or accent
                    text = self._config.text_color or text
                    surface = self._config.surface_color or surface
            except Exception:
                pass
            picker = GesturePosePickerDialog(
                gesture_name=self._name,
                description=self._description or "",
                thumbnails=self._candidate_thumbnails,
                accent_color=accent,
                text_color=text,
                surface_color=surface,
                parent=self,
            )
            picker.exec()
            chosen = picker.selected_thumbnail
            if chosen is not None:
                chosen_image_filename = self._save_thumbnail_to_disk(
                    registry, chosen[1], self._name
                )

        augmented = augment_samples(originals)
        try:
            registry.add(
                name=self._name,
                samples=augmented,
                action=self._action,
                description=self._description,
                overwrite=True,
                handedness=recorded_hand,
                image_filename=chosen_image_filename,
            )
        except ValueError as exc:
            touchless_message_box(
                self, "Save failed", str(exc),
                icon=QMessageBox.Critical, buttons=QMessageBox.Ok,
            )
            return
        registry.save()
        self._show_saved_summary(format_gesture_summary(registry.get(self._name)))
        self.saved.emit(self._name)
        self.accept()

    @staticmethod
    def _save_thumbnail_to_disk(registry, frame_bgr: np.ndarray, gesture_name: str) -> str:
        """Persist the chosen thumbnail to <registry_dir>/gesture_thumbnails/
        as a PNG. Returns the relative filename to store in the gesture
        record (or "" on failure)."""
        if frame_bgr is None:
            return ""
        try:
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in gesture_name).strip("_") or "gesture"
            filename = f"{safe}.png"
            target = registry.thumbnails_dir() / filename
            ok = cv2.imwrite(str(target), frame_bgr)
            if not ok:
                return ""
            return filename
        except Exception:
            return ""

    def _majority_handedness(self) -> Optional[str]:
        """Compute the majority hand label across recording samples.
        Returns "Left", "Right", or None when the votes are too mixed
        (< 60% agreement) — None means "fire on either hand"."""
        if not self._handedness_votes:
            return None
        total = len(self._handedness_votes)
        left = sum(1 for h in self._handedness_votes if h == "Left")
        right = total - left
        if right >= total * 0.60:
            return "Right"
        if left >= total * 0.60:
            return "Left"
        return None

    def _running_majority_label(self) -> Optional[str]:
        """Pretty label for the live overlay during recording. Same
        rule as _majority_handedness but returns 'mixed (Left/Right)'
        instead of None so the user sees what's happening."""
        if not self._handedness_votes:
            return None
        total = len(self._handedness_votes)
        left = sum(1 for h in self._handedness_votes if h == "Left")
        right = total - left
        if right >= total * 0.60:
            return f"Right ({right}/{total})"
        if left >= total * 0.60:
            return f"Left ({left}/{total})"
        return f"mixed (L{left}/R{right})"

    def _show_saved_summary(self, summary: str) -> None:
        """Custom popup so the summary uses the full dialog width
        (QMessageBox reserves the left side for the icon + indents the
        text past it, which truncates the monospace alignment)."""
        from PySide6.QtWidgets import QDialog
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Saved: {self._name}")
        dlg.setModal(True)
        dlg.setMinimumWidth(560)
        try:
            apply_touchless_titlebar(dlg)
        except Exception:
            pass
        dlg.setStyleSheet(
            f"""
            QDialog {{ background: #0E1822; }}
            QLabel#summaryHeader {{
                color: {self._accent_color};
                font-size: 16px;
                font-weight: 700;
            }}
            QLabel#summaryBody {{
                color: #DCE9F2;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 13px;
                background: rgba(0,0,0,0.25);
                border-radius: 6px;
                padding: 12px;
            }}
            QPushButton {{
                background: {self._accent_color};
                color: #0B1620;
                font-weight: 800;
                border: none;
                border-radius: 8px;
                padding: 8px 22px;
            }}
            QPushButton:hover {{ background: #FFFFFF; }}
            """
        )
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        header = QLabel(f"Saved gesture '{self._name}'")
        header.setObjectName("summaryHeader")
        layout.addWidget(header)

        body = QLabel(summary)
        body.setObjectName("summaryBody")
        body.setWordWrap(False)  # preserve monospace alignment
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(body)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        ok_button = QPushButton("OK")
        ok_button.setMinimumWidth(120)
        ok_button.clicked.connect(dlg.accept)
        button_row.addWidget(ok_button)
        layout.addLayout(button_row)

        dlg.exec()

    # --- close handler --------------------------------------------------

    def _on_space_pressed(self) -> None:
        """Fired by the dialog-scoped Space QShortcut. Starts recording
        only when we're in the idle state with the Begin button
        visible + enabled. No-op during a recording in progress or
        the post-recording save flow so the shortcut can't restart
        from the wrong state."""
        if self._state != "idle":
            return
        begin = getattr(self, "_begin_button", None)
        if begin is None or not begin.isVisible() or not begin.isEnabled():
            return
        self._on_begin_recording()

    def closeEvent(self, event) -> None:
        self._disconnect_worker()
        try:
            self._mp_hands.close()
        except Exception:
            pass
        super().closeEvent(event)

    def reject(self) -> None:
        self._disconnect_worker()
        try:
            self._mp_hands.close()
        except Exception:
            pass
        super().reject()


class GesturePosePickerDialog(QDialog):
    """3×3 grid of candidate thumbnails captured during recording.

    User clicks one to select it as the gesture's representative
    image. The selected pixmap is rendered in the Custom Gestures
    panel cards and in the Gesture Binds hover preview. Cancel /
    close keeps the gesture but leaves the image unset (the rest of
    the UI falls back to a placeholder)."""

    def __init__(
        self,
        *,
        gesture_name: str,
        description: str,
        thumbnails: list,
        accent_color: str,
        text_color: str,
        surface_color: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        # r51: install_indigo_chrome for Win10 + Win11 parity.
        from .window_chrome import install_indigo_chrome
        self._thumbnails = list(thumbnails)
        self._selected_index: Optional[int] = None
        self._accent = accent_color or "#1DE9B6"
        self._text = text_color or "#E5F6FF"
        self._surface = surface_color or "#0F172A"
        self.setWindowTitle("Pick a gesture image")
        self.setObjectName("gesturePosePicker")
        self.setModal(True)
        self.setMinimumWidth(720)
        self._body = install_indigo_chrome(self, "Pick a gesture image")
        self._build(gesture_name, description)
        self._apply_theme()

    @property
    def selected_thumbnail(self):
        if self._selected_index is None:
            return None
        try:
            return self._thumbnails[self._selected_index]
        except (IndexError, TypeError):
            return None

    def _build(self, gesture_name: str, description: str) -> None:
        root = QVBoxLayout(self._body)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        title = QLabel(f"Pick an image for “{gesture_name}”")
        title.setObjectName("posePickerTitle")
        title.setWordWrap(True)
        root.addWidget(title)

        if description:
            desc_box = QLabel(description)
            desc_box.setObjectName("posePickerDesc")
            desc_box.setWordWrap(True)
            root.addWidget(desc_box)

        hint = QLabel("Click the snapshot you want to represent this gesture.")
        hint.setObjectName("posePickerHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        grid_host = QFrame()
        grid_host.setObjectName("posePickerGrid")
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        root.addWidget(grid_host)

        # Up to 9 thumbs in a 3×3 grid. Each thumb is a clickable
        # button-shaped QFrame that emits its index on click via a
        # capturing lambda.
        cell_size = 180
        if not self._thumbnails:
            empty = QLabel(
                "No clean snapshots were captured during this recording. "
                "You can still save without an image."
            )
            empty.setWordWrap(True)
            empty.setObjectName("posePickerHint")
            grid.addWidget(empty, 0, 0)
        for i, (sample_idx, frame_bgr) in enumerate(self._thumbnails):
            r = i // 3
            c = i % 3
            cell = QPushButton()
            cell.setObjectName("posePickerCell")
            cell.setCheckable(True)
            cell.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            cell.setFixedSize(cell_size, cell_size)
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
                pix = QPixmap.fromImage(qimg).scaled(
                    cell_size - 8, cell_size - 8,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                cell.setIcon(pix)
                cell.setIconSize(pix.size())
            except Exception:
                cell.setText(f"#{sample_idx}")
            cell.clicked.connect(lambda _checked=False, idx=i: self._on_cell_clicked(idx))
            grid.addWidget(cell, r, c)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self._skip_button = QPushButton("Skip — no image")
        self._skip_button.setObjectName("posePickerSecondary")
        self._skip_button.clicked.connect(self.reject)
        button_row.addWidget(self._skip_button)
        self._confirm_button = QPushButton("Use Selected Image")
        self._confirm_button.setObjectName("posePickerPrimary")
        self._confirm_button.setEnabled(False)
        self._confirm_button.clicked.connect(self.accept)
        button_row.addWidget(self._confirm_button)
        root.addLayout(button_row)

        self._cells = [grid.itemAt(i).widget() for i in range(grid.count())]

    def _on_cell_clicked(self, idx: int) -> None:
        self._selected_index = idx
        for w in self._cells:
            if isinstance(w, QPushButton):
                w.setChecked(False)
        try:
            cell = self._cells[idx]
            if isinstance(cell, QPushButton):
                cell.setChecked(True)
        except IndexError:
            pass
        self._confirm_button.setEnabled(True)

    def _apply_theme(self) -> None:
        accent = self._accent
        text = self._text
        surface = self._surface
        self.setStyleSheet(
            f"""
            QDialog#gesturePosePicker {{
                background-color: {surface};
                color: {text};
                border: 1px solid rgba(29, 233, 182, 0.30);
            }}
            QLabel#posePickerTitle {{
                color: {accent};
                font-size: 18px;
                font-weight: 800;
                background: transparent;
            }}
            QLabel#posePickerDesc {{
                color: {text};
                font-size: 13px;
                background: rgba(255, 255, 255, 0.04);
                border-radius: 8px;
                padding: 8px 12px;
            }}
            QLabel#posePickerHint {{
                color: rgba(229, 246, 255, 0.75);
                font-size: 12px;
                background: transparent;
            }}
            QPushButton#posePickerCell {{
                background-color: rgba(0, 0, 0, 0.35);
                border: 2px solid rgba(255, 255, 255, 0.10);
                border-radius: 12px;
                padding: 4px;
            }}
            QPushButton#posePickerCell:hover {{
                border: 2px solid rgba(29, 233, 182, 0.55);
            }}
            QPushButton#posePickerCell:checked {{
                border: 2px solid {accent};
                background-color: rgba(29, 233, 182, 0.12);
            }}
            QPushButton#posePickerPrimary {{
                background-color: rgba(29, 233, 182, 0.18);
                color: {accent};
                border: 1px solid {accent};
                border-radius: 10px;
                padding: 9px 22px;
                font-weight: 700;
                min-width: 160px;
            }}
            QPushButton#posePickerPrimary:disabled {{
                background-color: rgba(255, 255, 255, 0.04);
                color: rgba(229, 246, 255, 0.30);
                border: 1px solid rgba(255, 255, 255, 0.10);
            }}
            QPushButton#posePickerSecondary {{
                background-color: rgba(255, 255, 255, 0.06);
                color: {text};
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 10px;
                padding: 9px 18px;
                min-width: 130px;
            }}
            """
        )

# Author: Konstantin Markov

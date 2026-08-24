"""Sandbox window: live test custom gestures only, with hold-to-activate
and cooldowns just like the live integration. Subscribes to the running
GestureWorker's frame stream, runs MediaPipe + the custom-gesture
classifier locally, and fires actions when matches complete a hold.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
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

from hgr.custom_gestures.action import describe as describe_action, fire_once
from hgr.custom_gestures.classifier import GestureClassifier
from hgr.custom_gestures.description import (
    live_signature,
    short_curl_label,
    short_spread_label,
)
from hgr.custom_gestures.dynamic_recording import palm_scale_from_landmarks
from hgr.custom_gestures.dynamic_runtime import DynamicGestureRuntime
from hgr.custom_gestures.recorder import (
    landmarks_from_mediapipe,
    normalize_landmarks,
)
from hgr.custom_gestures.registry import GestureRegistry

from .custom_gestures_chrome import apply_touchless_titlebar


_DEFAULT_HOLD = 1.0
_DEFAULT_GRACE = 0.2
_DEFAULT_COOLDOWN = 2.0


class SandboxWindow(QDialog):
    """Live tester. Read-only by default (shows matches but doesn't fire);
    toggle 'Fire actions' to enable real keystroke / hotkey delivery.
    """

    def __init__(
        self,
        worker,
        accent_color: str,
        parent: Optional[QWidget] = None,
        config=None,
    ) -> None:
        super().__init__(parent)
        # r51: install_indigo_chrome for Win10 + Win11 parity.
        from .window_chrome import install_indigo_chrome
        self._body = install_indigo_chrome(self, "Custom gesture sandbox")
        self.setWindowTitle("Custom Gestures Sandbox")
        self.setModal(False)
        self.setMinimumSize(820, 560)
        self._worker = worker
        self._accent_color = accent_color
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

        self._registry = GestureRegistry()
        self._registry.load()
        # Match the live runner's threshold (0.78) instead of the
        # classifier's default 0.88. Sandbox testing was rejecting
        # legit gestures whose live-runner scores hovered around
        # 0.80–0.85 — the user reported a gesture in their list
        # that 'isn't even being detected' here, while it fires
        # fine in the actual app. Aligning the threshold so the
        # sandbox is a faithful preview of live behaviour.
        self._classifier = GestureClassifier(self._registry, threshold=0.78)
        self._classifier.reload()
        # Dynamic-gesture runtime — same registry, separate classifier.
        # Sandbox calls process_frame with dispatch=False unless the
        # "Fire actions" checkbox is on, so detection feedback shows
        # up even in read-only mode.
        self._dynamic_runtime = DynamicGestureRuntime()
        self._dynamic_runtime.reload()

        # Hold-to-activate state.
        self._hold_name: Optional[str] = None
        self._hold_started_at = 0.0
        self._last_match_at = 0.0
        self._fired_for_hold = False
        self._latest_feats: Optional[np.ndarray] = None
        self._latest_sig: dict = {}
        self._last_match_label = "no hand"
        self._last_fire_name: Optional[str] = None
        self._last_fire_at = 0.0
        self._cooldown_until = 0.0  # absolute monotonic time at which cooldown ends

        # Defer camera open until after the dialog is shown so the
        # user sees the window immediately even if cv2.VideoCapture
        # blocks on a slow phone-camera handshake.
        self._owns_camera = False
        self._cap = None
        self._poll_timer: Optional[QTimer] = None
        self._using_worker = False
        self._camera_connect_attempted = False

        # Sandbox-owned drawing overlay window for the
        # show_overlay_drawing action, lazily constructed on
        # first use. Self-contained so 'fire on' tests work even
        # when the sandbox is opened standalone or when the
        # main-window signal glue isn't wired up yet — the user
        # gets the same toggle-show / toggle-hide behaviour the
        # live engine produces, without any cross-window plumbing.
        self._sandbox_drawing_overlay = None

        self._build()

    def showEvent(self, event):  # noqa: N802 (Qt API name)
        super().showEvent(event)
        try:
            apply_touchless_titlebar(self)
        except Exception:
            pass
        if not self._camera_connect_attempted:
            self._camera_connect_attempted = True
            QTimer.singleShot(0, self._deferred_connect)

    def _deferred_connect(self) -> None:
        self._video_label.setText("Connecting to camera...")
        self._video_label.repaint()
        self._connect_worker()

    def _build(self) -> None:
        self.setStyleSheet(
            f"""
            QDialog {{ background: #0E1822; }}
            QLabel {{ color: #DCE9F2; }}
            QPushButton, QCheckBox {{
                color: #E5F6FF;
                font-weight: 600;
            }}
            /* Checkbox indicator: explicit dark/light contrast so the
               outline is readable on the Touchless dark background.
               The OS default renders nearly black-on-dark-blue. */
            QCheckBox {{
                spacing: 8px;
                padding: 4px 0;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #B7C5D1;
                background: rgba(255,255,255,0.05);
            }}
            QCheckBox::indicator:hover {{
                border: 2px solid #E5F6FF;
                background: rgba(255,255,255,0.10);
            }}
            QCheckBox::indicator:checked {{
                border: 2px solid {self._accent_color};
                background: {self._accent_color};
            }}
            QCheckBox::indicator:checked:hover {{
                border: 2px solid #FFFFFF;
                background: {self._accent_color};
            }}
            QPushButton {{
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 8px;
                padding: 8px 18px;
            }}
            QPushButton:hover {{ background: rgba(255,255,255,0.14); }}
            QPushButton#closeBtn {{
                background: {self._accent_color};
                color: #0B1620;
                font-weight: 800;
            }}
            """
        )
        root = QVBoxLayout(self._body)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        info = QLabel(
            "Hold any of your saved custom gestures. Live matches are "
            "shown over the camera. Toggle <b>Fire actions</b> to actually "
            "trigger the bound keystrokes / commands."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        self._video_label = QLabel("Waiting for camera frames...")
        self._video_label.setMinimumHeight(420)
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet(
            "QLabel { background: #050A0F; color: #5C6F7E; border-radius: 8px; }"
        )
        root.addWidget(self._video_label, 1)

        bottom = QHBoxLayout()
        self._fire_checkbox = QCheckBox("Fire actions")
        # Default ON — sandbox is for testing real activation.
        self._fire_checkbox.setChecked(True)
        bottom.addWidget(self._fire_checkbox)
        bottom.addStretch(1)
        close_button = QPushButton("Close")
        close_button.setObjectName("closeBtn")
        close_button.clicked.connect(self.accept)
        bottom.addWidget(close_button)
        root.addLayout(bottom)

    # --- worker connection ----------------------------------------------

    def _connect_worker(self) -> None:
        """Dual-mode frame source: subscribe to the live worker if
        present, otherwise open a private cv2 camera so the sandbox
        works without the main pipeline running."""
        self._owns_camera = False
        self._cap = None
        self._poll_timer: Optional[QTimer] = None
        self._using_worker = False

        # Only subscribe to the worker's frames if it's actually pumping
        # them. The GestureWorker exists from app startup, but
        # is_running is False until the user starts the live view —
        # subscribing in that idle state would just sit on a silent
        # signal forever.
        if self._worker is not None and bool(getattr(self._worker, "is_running", False)):
            try:
                self._worker.raw_frame_ready.connect(self._on_frame)
                self._using_worker = True
                return
            except Exception:
                self._using_worker = False

        cap = self._open_configured_camera()
        if cap is None:
            # r51: touchless_message_box (indigo chrome, Win10+Win11 parity).
            from .window_chrome import touchless_message_box
            touchless_message_box(
                self,
                "Camera unavailable",
                "Could not open the camera. If you're using a phone "
                "camera, make sure your phone is on the same Wi-Fi "
                "network and the QR pairing is still active.",
                icon=QMessageBox.Critical,
                buttons=QMessageBox.Ok,
            )
            return
        self._cap = cap
        # _owns_camera was set inside _open_configured_camera (True if we
        # opened it ourselves, False if we're borrowing the worker's
        # QR-paired phone capture). Don't overwrite it here — releasing
        # a borrowed capture would break the main pipeline.
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

    def _open_configured_camera(self):
        """Open the camera Touchless is configured to use, mirroring the
        priority order GestureWorker._open_camera uses:

          1. Worker's QR-paired phone capture (engine running with QR).
          2. Main window's QR server capture (QR paired + active, but
             engine not running yet — server auto-starts at launch).
          3. URL-based phone camera (`phone_camera_enabled` + URL set).
          4. Preferred / first-available webcam.

        Priorities 1 & 2 are BORROWED — `_owns_camera` stays False.
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

        # 2. Borrow the main-window's QR server capture if QR-paired but
        # the live engine isn't running yet.
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
        """Walk the parent chain to find main_window's
        _phone_camera_qr_server.capture. Returns the running capture if
        `phone_camera_qr_active` is enabled, else None."""
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
                try:
                    np_frame = np.asarray(np_frame)
                except Exception:
                    return
            if np_frame.ndim != 3 or np_frame.shape[2] not in (3, 4):
                return
            # Mirror to selfie view when we own the camera; skip when
            # borrowing the worker's frames (already mirrored upstream).
            should_flip = self._owns_camera and not bool(
                getattr(self._config, "camera_source_is_mirrored", False)
            )
            mirrored = cv2.flip(np_frame, 1) if should_flip else np_frame
            rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB) if mirrored.shape[2] == 3 else mirrored[:, :, :3]
            result = self._mp_hands.process(rgb)
            display_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            now = time.monotonic()
            match = None
            if result.multi_hand_landmarks:
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
                # Pull MediaPipe handedness so we drop matches whose
                # gesture is bound to a different hand. Mirrors the
                # live runner's gating in CustomGestureRunner.process.
                live_hand: Optional[str] = None
                try:
                    if result.multi_handedness:
                        label = str(result.multi_handedness[0].classification[0].label)
                        if label in ("Left", "Right"):
                            live_hand = label
                except Exception:
                    pass
                match = self._classifier.classify_raw(
                    self._latest_feats, sticky_name=self._hold_name
                )
                # Apply the same handedness gate as the live runner —
                # if the matched gesture is bound to a different hand,
                # treat as no match.
                if match is not None:
                    g_hand = match.gesture.handedness
                    if (
                        g_hand in ("Left", "Right")
                        and live_hand in ("Left", "Right")
                        and g_hand != live_hand
                    ):
                        match = None
                if match is not None:
                    self._last_match_at = now
                    if self._hold_name != match.gesture.name:
                        self._hold_name = match.gesture.name
                        self._hold_started_at = now
                        self._fired_for_hold = False
                    held = now - self._hold_started_at
                    if self._fired_for_hold:
                        self._last_match_label = (
                            f"{match.gesture.name} ({match.score:.2f})  fired"
                        )
                    else:
                        # Hold timer is shown in the progress bar below;
                        # the label just identifies the matched gesture.
                        self._last_match_label = (
                            f"{match.gesture.name} ({match.score:.2f})"
                        )
                    if (
                        self._fire_checkbox.isChecked()
                        and not self._fired_for_hold
                        and held >= _DEFAULT_HOLD
                    ):
                        # Special-case the Qt-coupled `show_overlay_drawing`
                        # action. fire_once / execute() are no-ops for that
                        # kind because the actual overlay is a QWidget that
                        # has to be mutated on the GUI thread — the live
                        # engine handles it via a Qt signal on the worker.
                        # Sandbox handles it directly: own a
                        # DrawingOverlayWindow and toggle it here. Self-
                        # contained on purpose — the previous attempt
                        # routed through worker.drawing_overlay_toggle_
                        # requested.emit, but that only worked when
                        # main_window's signal glue was already wired up
                        # (and the user reported overlays still not
                        # appearing). Doing it locally guarantees the
                        # sandbox's 'fire on' visibly works regardless
                        # of how / whether the host main_window is
                        # listening.
                        action_kind = (match.gesture.action.kind or "").lower()
                        fired = False
                        if action_kind == "show_overlay_drawing":
                            filename = str(
                                (match.gesture.action.payload or {}).get("filename", "")
                            )
                            fired = bool(self._sandbox_toggle_drawing_overlay(filename))
                        else:
                            fired = bool(fire_once(match.gesture.name, match.gesture.action))
                        if fired:
                            self._fired_for_hold = True
                            self._last_fire_name = match.gesture.name
                            self._last_fire_at = now
                            self._cooldown_until = now + _DEFAULT_COOLDOWN
                else:
                    self._last_match_label = "hand present, no match"
                    if (
                        self._hold_name is not None
                        and now - self._last_match_at >= _DEFAULT_GRACE
                    ):
                        self._hold_name = None
                        self._fired_for_hold = False

                # Dynamic gesture matching runs in parallel to the
                # static classifier above. Detection (segment close +
                # DTW match below threshold) is independent of the
                # sandbox "Fire actions" checkbox — but we only call
                # the gesture's action when the checkbox is on, via
                # the runtime's dispatch=True path. Otherwise the
                # runtime returns the matched name without firing.
                try:
                    fire_now = bool(self._fire_checkbox.isChecked())
                    fired_dyn = self._dynamic_runtime.process_frame(
                        lm,
                        palm_scale=palm_scale_from_landmarks(lm),
                        handedness=live_hand,
                        timestamp=now,
                        dispatch=fire_now,
                    )
                except Exception as exc:
                    fired_dyn = None
                    print(f"[sandbox] dynamic runtime error: {exc}")
                if fired_dyn:
                    self._last_match_label = (
                        f"dynamic: {fired_dyn}"
                        + ("  fired" if fire_now else "  (detected, fire-off)")
                    )
                    self._last_fire_name = fired_dyn
                    self._last_fire_at = now
                    if fire_now:
                        self._cooldown_until = now + _DEFAULT_COOLDOWN
            else:
                self._latest_sig = {}
                self._latest_feats = None
                self._last_match_label = "no hand"
                if (
                    self._hold_name is not None
                    and now - self._last_match_at >= _DEFAULT_GRACE
                ):
                    self._hold_name = None
                    self._fired_for_hold = False
                try:
                    self._dynamic_runtime.hand_lost()
                except Exception:
                    pass

            self._draw_overlay(display_bgr, match, now)
            self._render(display_bgr)
        except Exception:
            pass

    def _draw_overlay(self, frame, match, now) -> None:
        cv2.putText(
            frame, self._last_match_label, (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (40, 220, 40) if match is not None else (180, 180, 180),
            2, cv2.LINE_AA,
        )

        # Three states for the progress bar below the label:
        #   1. cooldown active (just fired) — orange bar fills 0→1 over
        #      the cooldown window
        #   2. holding a match (not yet fired) — green bar fills 0→1 over
        #      the hold-to-activate window
        #   3. otherwise — no bar
        bar_x, bar_y = 12, 42
        bar_w, bar_h = 320, 18

        in_cooldown = now < self._cooldown_until
        if in_cooldown:
            cooldown_total = max(0.001, _DEFAULT_COOLDOWN)
            elapsed = max(0.0, _DEFAULT_COOLDOWN - (self._cooldown_until - now))
            progress = min(1.0, elapsed / cooldown_total)
            label = f"cooldown  {elapsed:.1f}s / {_DEFAULT_COOLDOWN:.1f}s"
            fill_color = (40, 140, 240)  # warm orange (BGR)
            track_color = (45, 50, 60)
            self._draw_progress_bar(
                frame, bar_x, bar_y, bar_w, bar_h,
                progress, label, fill_color, track_color,
            )
        elif match is not None and not self._fired_for_hold:
            held = now - self._hold_started_at
            progress = min(1.0, held / _DEFAULT_HOLD) if _DEFAULT_HOLD > 0 else 1.0
            label = f"hold  {min(held, _DEFAULT_HOLD):.1f}s / {_DEFAULT_HOLD:.1f}s"
            fill_color = (40, 220, 40)  # green
            track_color = (45, 60, 50)
            self._draw_progress_bar(
                frame, bar_x, bar_y, bar_w, bar_h,
                progress, label, fill_color, track_color,
            )

        if self._last_fire_name and now - self._last_fire_at < 1.0:
            cv2.putText(
                frame, f"FIRED: {self._last_fire_name}", (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA,
            )

    def _draw_progress_bar(
        self,
        frame,
        x: int,
        y: int,
        w: int,
        h: int,
        progress: float,
        label: str,
        fill_color: tuple,
        track_color: tuple,
    ) -> None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), track_color, -1)
        fill_w = int(w * max(0.0, min(1.0, progress)))
        if fill_w > 0:
            cv2.rectangle(frame, (x, y), (x + fill_w, y + h), fill_color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
        cv2.putText(
            frame, label, (x + 8, y + h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

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
                frame, "Live finger state:", (box_x, box_y + 12),
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
                col = (
                    (60, 220, 60),
                    (60, 220, 180),
                    (60, 180, 220),
                    (140, 120, 220),
                    (200, 80, 200),
                )[max(0, min(4, int(c)))]
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

    def _render(self, frame_bgr) -> None:
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

    # --- close handlers -------------------------------------------------

    def _sandbox_toggle_drawing_overlay(self, filename: str) -> bool:
        """Toggle the sandbox's own DrawingOverlayWindow for a fired
        show_overlay_drawing action. Returns True on a successful
        show OR successful hide; False if the file doesn't resolve
        (so the caller knows the fire effectively failed).

        Same toggle semantics as the live engine: fire once → show;
        fire again with the same filename → hide; fire with a
        different filename while one is showing → swap to the new
        drawing.
        """
        from .drawing_overlay_window import (
            DrawingOverlayWindow,
            resolve_drawing_path,
        )

        # Lazy-construct the overlay so the sandbox doesn't pay the
        # widget construction cost up-front for users who never test
        # an overlay-drawing gesture.
        if self._sandbox_drawing_overlay is None:
            try:
                self._sandbox_drawing_overlay = DrawingOverlayWindow(parent=self)
            except Exception:
                return False
        overlay = self._sandbox_drawing_overlay

        drawings_dir = ""
        if self._config is not None:
            drawings_dir = str(getattr(self._config, "drawings_save_dir", "") or "")
        resolved = resolve_drawing_path(filename, drawings_dir)
        if resolved is None:
            return False

        currently_showing_same = (
            overlay.isVisible()
            and overlay.current_path is not None
            and Path(overlay.current_path) == resolved
        )
        if currently_showing_same:
            overlay.hide()
            return True
        return bool(overlay.show_image(str(resolved)))

    def closeEvent(self, event) -> None:
        # Hide the sandbox's drawing overlay if it's still up so it
        # doesn't outlive the dialog — leaving an always-on-top
        # transparent window after the sandbox closes is confusing.
        if self._sandbox_drawing_overlay is not None:
            try:
                self._sandbox_drawing_overlay.hide()
            except Exception:
                pass
        self._disconnect_worker()
        try:
            self._mp_hands.close()
        except Exception:
            pass
        super().closeEvent(event)

    def accept(self) -> None:
        self._disconnect_worker()
        try:
            self._mp_hands.close()
        except Exception:
            pass
        super().accept()

# Author: Konstantin Markov

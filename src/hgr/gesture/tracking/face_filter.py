"""Conservative face-exclusion filter for the hand detector.

Problem this solves: the hand model occasionally latches onto a face
(or, more rarely, another non-hand object) and emits a full 21-landmark
"hand" skeleton on it. That false detection then drives gestures.

Two-part decision, designed so a real hand is *never* dropped:
  1. Hand-shape check (the deciding signal). Geometric face-overlap
     alone cannot tell a real hand held over a face from a face-latch,
     so we check the SHAPE. A detection that looks like a hand (finger
     bone-path length and/or fingertip spread) is kept unconditionally —
     even dead-centre on the face.
  2. Face overlap (only consulted for things that DON'T look like a
     hand). A non-hand-shaped detection sitting on a confident face box
     is the face-latch signature, so it is dropped.

Performance: the face DETECTION (BlazeFace, ~5 ms) runs on a background
worker thread, throttled, so the per-frame hand pipeline never blocks on
it — the hot path only does microsecond shape/containment math against
the most recent cached face boxes. When no hands are present, no frames
are submitted and the worker idles (zero cost).

Guarantees so the existing pipeline is never degraded:
  * Fail-safe: any error, an unavailable/unbundled model, or a dead
    worker makes filtering a silent no-op — hands pass through untouched.
  * Conservative: only ever removes detections that both fail the hand
    shape test AND sit on a face.
  * Kill switch: set env TOUCHLESS_FACE_FILTER=0 to disable entirely.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

# (xmin, ymin, xmax, ymax) in normalized [0, 1] image coordinates.
_Box = Tuple[float, float, float, float]


class FaceExclusionFilter:
    # Finger landmark chains (MediaPipe topology): MCP->PIP->DIP->TIP.
    _INDEX = (5, 6, 7, 8)
    _MIDDLE = (9, 10, 11, 12)
    _TIPS = (4, 8, 12, 16, 20)

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_face_confidence: float = 0.7,
        face_box_inflation: float = 0.15,
        containment_threshold: float = 0.5,
        min_detect_interval: float = 0.08,
        idle_exit_seconds: float = 5.0,
    ) -> None:
        # Env kill-switch takes precedence so a problematic build can be
        # reverted to the pre-filter behaviour without a code change.
        if os.getenv("TOUCHLESS_FACE_FILTER", "").strip() == "0":
            enabled = False
        self._enabled = bool(enabled)
        self._min_face_conf = float(min_face_confidence)
        self._inflation = max(0.0, float(face_box_inflation))
        self._containment = float(containment_threshold)
        self._min_detect_interval = max(0.0, float(min_detect_interval))
        self._idle_exit_seconds = max(0.5, float(idle_exit_seconds))

        self._detector = None
        self._init_failed = False

        # Shared state between the hot path and the worker thread.
        self._lock = threading.Lock()
        self._cached_faces: List[_Box] = []
        self._pending_frame: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._worker_started = False

    # ------------------------------------------------------------------
    # Detector lifecycle
    # ------------------------------------------------------------------
    def _ensure_detector(self) -> None:
        if self._detector is not None or self._init_failed:
            return
        try:
            import mediapipe as mp  # already a hard dependency

            # model_selection=1 = full-range (~5 m): covers a close
            # controlling user and a person standing further back.
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self._min_face_conf,
            )
        except Exception:
            self._init_failed = True
            self._detector = None

    def _detect_faces(self, rgb_frame: np.ndarray) -> List[_Box]:
        """Pure inference: return inflated face boxes for a frame."""
        self._ensure_detector()
        if self._detector is None:
            return []
        try:
            result = self._detector.process(rgb_frame)
        except Exception:
            return []
        faces: List[_Box] = []
        for det in (getattr(result, "detections", None) or []):
            try:
                score = float(det.score[0]) if getattr(det, "score", None) else 0.0
                if score < self._min_face_conf:
                    continue
                rbb = det.location_data.relative_bounding_box
                xmin = float(rbb.xmin)
                ymin = float(rbb.ymin)
                xmax = xmin + float(rbb.width)
                ymax = ymin + float(rbb.height)
                pad_w = (xmax - xmin) * self._inflation
                pad_h = (ymax - ymin) * self._inflation
                faces.append((xmin - pad_w, ymin - pad_h, xmax + pad_w, ymax + pad_h))
            except Exception:
                continue
        return faces

    def _refresh_faces(self, rgb_frame: np.ndarray) -> None:
        """Synchronous detect + store. Used by the diagnostic demo; the
        live pipeline goes through the background worker instead."""
        faces = self._detect_faces(rgb_frame)
        with self._lock:
            self._cached_faces = faces

    def _start_worker(self) -> None:
        if not self._enabled:
            return
        # Start a worker only if one isn't already running. The worker
        # clears this flag when it self-terminates on idle, so a detector
        # that goes quiet (e.g. replaced on a GPU/Lite toggle) lets its
        # thread exit and a resumed detector spins a fresh one — no
        # accumulation even if close() is never called.
        with self._lock:
            if self._worker_started:
                return
            self._worker_started = True
        self._worker = threading.Thread(
            target=self._worker_loop, name="face-filter", daemon=True
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        try:
            self._ensure_detector()
            if self._detector is None:
                return  # no model -> filter stays a no-op
            last_run = 0.0
            last_frame_at = time.monotonic()
            while not self._stop.is_set():
                frame = None
                with self._lock:
                    frame = self._pending_frame
                    self._pending_frame = None
                now = time.monotonic()
                if frame is None:
                    # Self-terminate after a quiet spell so a detector
                    # that stopped being used can't leave a thread idling.
                    if (now - last_frame_at) > self._idle_exit_seconds:
                        return
                    self._stop.wait(0.01)
                    continue
                last_frame_at = now
                if (now - last_run) < self._min_detect_interval:
                    self._stop.wait(0.01)
                    continue
                last_run = now
                try:
                    faces = self._detect_faces(frame)
                except Exception:
                    faces = None
                if faces is not None:
                    with self._lock:
                        self._cached_faces = faces
        finally:
            # Allow a future filter() call to start a fresh worker.
            with self._lock:
                self._worker_started = False

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hand_bbox(raw: np.ndarray) -> _Box:
        xs = raw[:, 0]
        ys = raw[:, 1]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    @staticmethod
    def _containment_fraction(hand: _Box, face: _Box) -> float:
        """Fraction of the hand box's area that lies inside the face box."""
        ix0 = max(hand[0], face[0])
        iy0 = max(hand[1], face[1])
        ix1 = min(hand[2], face[2])
        iy1 = min(hand[3], face[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        hand_area = max(1e-9, (hand[2] - hand[0]) * (hand[3] - hand[1]))
        return inter / hand_area

    @classmethod
    def _looks_like_hand(cls, raw: np.ndarray) -> bool:
        """Lenient, pose-invariant test that the 21 landmarks form a real
        hand rather than the model latching onto a face.

        Finger bone-path length (sum of segment lengths) is roughly
        pose-invariant — folding a finger bends the joints but doesn't
        shorten the bones — so this passes open hands, fists, and points
        alike. Fingertip spread catches open hands even more strongly.
        We OR several lenient cues and only declare "not a hand" when ALL
        of them fail, so a real hand is essentially never rejected. On any
        uncertainty it returns True (keep)."""
        try:
            pts = raw[:, :2].astype(float)
            palm_w = float(np.linalg.norm(pts[5] - pts[17]))  # index_mcp..pinky_mcp
            if palm_w < 1e-6:
                return True

            def path_len(chain):
                return sum(
                    float(np.linalg.norm(pts[chain[i + 1]] - pts[chain[i]]))
                    for i in range(len(chain) - 1)
                )

            index_ratio = path_len(cls._INDEX) / palm_w
            middle_ratio = path_len(cls._MIDDLE) / palm_w
            tips = pts[list(cls._TIPS)]
            spread = 0.0
            for i in range(len(tips)):
                for j in range(i + 1, len(tips)):
                    spread = max(spread, float(np.linalg.norm(tips[i] - tips[j])))
            tip_spread_ratio = spread / palm_w

            return (
                middle_ratio >= 0.55
                or index_ratio >= 0.55
                or tip_spread_ratio >= 1.0
            )
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------
    def filter(self, rgb_frame: np.ndarray, hand_entries: list) -> list:
        """Return hand_entries with face-latched entries removed.

        hand_entries: list of (raw_landmarks ndarray (N,3), label, score).
        Anything that goes wrong returns the input list unchanged. The
        only real work here is microsecond geometry math; face detection
        runs on the worker thread."""
        if not self._enabled or not hand_entries:
            return hand_entries
        try:
            self._start_worker()
            # Hand the worker the most recent frame (last-value-wins) and
            # snapshot the latest face boxes — both under one short lock.
            with self._lock:
                self._pending_frame = rgb_frame
                faces = list(self._cached_faces)
            if not faces:
                return hand_entries

            kept = []
            for entry in hand_entries:
                raw = entry[0]
                # The hand always wins: if it looks like a hand, keep it,
                # even when held directly over the face.
                if self._looks_like_hand(raw):
                    kept.append(entry)
                    continue
                hb = self._hand_bbox(raw)
                cx = (hb[0] + hb[2]) * 0.5
                cy = (hb[1] + hb[3]) * 0.5
                drop = False
                for fb in faces:
                    centre_inside = fb[0] <= cx <= fb[2] and fb[1] <= cy <= fb[3]
                    if centre_inside and self._containment_fraction(hb, fb) >= self._containment:
                        drop = True
                        break
                if not drop:
                    kept.append(entry)
            return kept
        except Exception:
            return hand_entries

    def reset(self) -> None:
        # A hand-detection gap doesn't mean the face left; keep the cache.
        pass

    def close(self) -> None:
        self._stop.set()
        worker = self._worker
        if worker is not None:
            try:
                worker.join(timeout=1.0)
            except Exception:
                pass
        self._worker = None
        try:
            if self._detector is not None:
                self._detector.close()
        except Exception:
            pass
        self._detector = None

"""OpenCV-compatible shim that yields frames received over the WebSocket.

The engine's existing camera path consumes a `cv2.VideoCapture`: it calls
`.read()` repeatedly, expects `(ok, frame)`, and releases when done. This
class implements that surface on top of a thread-safe "latest frame"
slot maintained by `PhoneCameraServer`. Consuming the latest frame (as
opposed to draining a queue) keeps gesture tracking at the server's
push rate without lag accumulating when the PC briefly stalls.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


class PhoneCameraCapture:
    def __init__(self, wait_seconds: float = 0.05) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_age_hint = 0.0  # monotonic ts of latest frame
        self._last_read_stamp = 0.0
        # Pipeline-latency exposure for the engine. Set by read()
        # to the monotonic ts at which the most-recently-consumed
        # frame was decoded, so callers can compute display lag.
        self._last_consumed_ts: float = 0.0
        self._closed = False
        self._opened = True
        self._wait_seconds = float(wait_seconds)
        self._wait_event = threading.Event()

    def push_jpeg(self, jpeg_bytes: bytes) -> None:
        """Server-side hook: decode a JPEG payload and publish as the latest frame."""
        if self._closed:
            return
        if not jpeg_bytes:
            return
        try:
            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            return
        if frame is None or frame.size == 0:
            return
        with self._lock:
            self._frame = frame
            self._frame_age_hint = time.monotonic()
        # Wake any consumer that's currently waiting on read(). The
        # event is consume-and-clear (cleared by read() once it has
        # taken the frame) so the next read() call blocks until the
        # NEXT push, instead of returning the same frame repeatedly.
        self._wait_event.set()

    def push_frame(self, frame: np.ndarray) -> None:
        """Publish an already-decoded BGR frame as the latest frame.

        The WebRTC receiver hands us decoded ndarrays directly (PyAV
        gives BGR via VideoFrame.to_ndarray), so there's no JPEG to
        decode. Mirrors push_jpeg's publish-and-wake otherwise.
        """
        if self._closed or frame is None or getattr(frame, "size", 0) == 0:
            return
        with self._lock:
            self._frame = frame
            self._frame_age_hint = time.monotonic()
        self._wait_event.set()

    def has_fresh_frame(self) -> bool:
        with self._lock:
            return self._frame is not None

    def isOpened(self) -> bool:  # noqa: N802 (cv2 API parity)
        return not self._closed

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:  # noqa: D401
        """Return the newest *fresh* frame, or (False, None) if no
        push lands within a brief wait window.

        Each phone-pushed frame is returned EXACTLY ONCE.
        Without consume-once semantics, a fast gesture loop
        (singleShot pacing) re-processes the same frame many times
        between phone pushes, surfacing as content-duplication lag.

        Brief 5 ms wait when no frame is buffered — gives the
        push-handler thread GIL scheduling time so it doesn't get
        starved by the main thread's tight-loop polling.
        """
        if self._closed:
            return False, None
        with self._lock:
            frame = self._frame
            self._frame = None
            self._wait_event.clear()
            if frame is not None:
                self._last_read_stamp = self._frame_age_hint
                self._last_consumed_ts = self._frame_age_hint
        if frame is not None:
            return True, frame
        if not self._wait_event.wait(timeout=0.005):
            return False, None
        with self._lock:
            frame = self._frame
            self._frame = None
            self._wait_event.clear()
            if frame is not None:
                self._last_read_stamp = self._frame_age_hint
                self._last_consumed_ts = self._frame_age_hint
        if frame is None:
            return False, None
        return True, frame

    def set(self, *_args, **_kwargs) -> bool:  # cv2.VideoCapture.set no-op for us
        return True

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            with self._lock:
                return float(self._frame.shape[1]) if self._frame is not None else 0.0
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            with self._lock:
                return float(self._frame.shape[0]) if self._frame is not None else 0.0
        return 0.0

    def grab(self) -> bool:
        return self.has_fresh_frame()

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.read()

    def release(self) -> None:
        self._closed = True
        self._wait_event.set()
        with self._lock:
            self._frame = None

# Author: Konstantin Markov

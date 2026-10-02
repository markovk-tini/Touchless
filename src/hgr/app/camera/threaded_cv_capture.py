"""Reader-thread wrapper around cv2.VideoCapture.

Why this exists:
`cv2.VideoCapture.read()` is synchronous — it blocks until the next
camera frame arrives, which on a 30 fps camera is up to 33 ms of dead
main-thread time per call. With the gesture loop's per-cycle work at
~5-7 ms, the cap.read blocking caps the loop at the camera's frame
rate AND blocks every other Qt event from firing during the wait.
With heavy main-thread paint pressure, FPS collapses below the
camera rate.

The fix mirrors the pattern in FfmpegMjpegCapture: a daemon thread
loops cv2 reads and stashes the latest frame; main-thread `.read()`
returns the latest fresh frame, blocking only briefly via an event
when no fresh frame has arrived since the last consume.

This module is a drop-in stand-in for cv2.VideoCapture wherever the
engine consumes one — same `read() / isOpened() / release() / get() /
set()` surface.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np

# v1.1.9.2 (r18): process-wide DirectShow graph lock.
#
# OpenCV's DirectShow backend keeps ONE global videoInput instance
# (g_VI) with a per-index device table shared by every VideoCapture in
# the process. Constructing a capture on thread A while thread B
# releases (or rebuilds via FOURCC/FPS set()) another capture corrupts
# that shared state — reproduced on the reference rig 2026-09-24 as
# STATUS_HEAP_CORRUPTION (0xC0000374) on the main thread inside
# ThreadedCvCapture.release() while the startup camera warmup scan was
# mid-construction on a background thread (the same START-click shape
# as the dad-rig r17 access violation). Everything that constructs,
# releases or writes properties on a cv2 capture takes this lock.
# read() never does, so the hot path is untouched.
DSHOW_GRAPH_LOCK = threading.RLock()

# r18 review: every wait on the lock is BOUNDED. An abandoned
# _cv2_open_with_timeout worker can sit inside cv2.VideoCapture for the
# driver's 60-120 s when another process holds the camera; nothing may
# block on it indefinitely, least of all the GUI thread.
_LOCK_WAIT_RELEASE_S = 1.0
_LOCK_WAIT_SET_S = 2.0

# Captures whose inner release was handed to their reader thread
# (release() join timed out). A NEW cv2.VideoCapture on the same
# DirectShow index while one of these still owns the graph shares
# OpenCV's global device slot -> use-after-free. Constructors wait
# for this list to drain (bounded) before building a graph.
_PENDING_RELEASES: list = []
_PENDING_LOCK = threading.Lock()


def release_capture_serialised(cap, wait: float = _LOCK_WAIT_RELEASE_S) -> None:
    """Release a cv2 capture under the graph lock with a bounded wait.
    If the lock is busy longer than `wait`, the release is handed to a
    daemon thread that will take the lock when it frees up, so the
    caller (often the GUI thread) never blocks on a stuck driver."""
    if cap is None:
        return
    if DSHOW_GRAPH_LOCK.acquire(timeout=float(wait)):
        try:
            cap.release()
        except Exception:
            pass
        finally:
            DSHOW_GRAPH_LOCK.release()
        return
    try:
        import sys as _sys
        _sys.stderr.write("[camera] DirectShow graph busy; deferring capture release to a helper thread\n")
        _sys.stderr.flush()
    except Exception:
        pass

    def _later() -> None:
        try:
            with DSHOW_GRAPH_LOCK:
                cap.release()
        except Exception:
            pass

    threading.Thread(target=_later, name="cv2-deferred-release", daemon=True).start()


def wait_for_pending_releases(timeout: float = 5.0) -> bool:
    """Block (bounded) until every deferred inner release has run.
    Returns True when the list is empty."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        with _PENDING_LOCK:
            pending = list(_PENDING_RELEASES)
        if not pending:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        for c in pending:
            t = getattr(c, "_reader_thread_for_join", None)
            if t is not None and t.is_alive():
                t.join(timeout=max(0.0, min(remaining, 0.25)))
        time.sleep(0.02)


class ThreadedCvCapture:
    """Async wrapper for cv2.VideoCapture. Drops blocking-read latency
    from main thread. Same API surface the engine consumes."""

    def __init__(self, inner: cv2.VideoCapture, *, warmup_frames: int = 12) -> None:
        self._inner = inner
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        # See FfmpegMjpegCapture for the same fields — used for
        # end-to-end pipeline latency measurement.
        self._latest_frame_ts: float = 0.0
        self._last_consumed_ts: float = 0.0
        self._fresh_frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._read_error = False
        self._closed = False
        self._reader_thread: Optional[threading.Thread] = None
        # r18: deferred-release handoff state (see release()).
        self._release_inner_on_exit = False
        self._reader_exited = False
        self._handoff_lock = threading.Lock()
        self._reader_thread_for_join: Optional[threading.Thread] = None
        # Drop this many OK frames at the start of the reader loop
        # before publishing any to consumers. Replaces the previous
        # synchronous `warmup_capture(cap)` that ran on the main
        # thread before the reader started — that approach blocked
        # the UI for up to ~2 s during camera open AND used a
        # brightness heuristic that misclassified corrupted-decode
        # frames (random noise can read as bright) and dim-room
        # frames (clean output can read as dark) in opposite ways.
        # Doing the discard here is non-blocking (the UI is free
        # while these frames pass through the reader thread) and
        # works on every camera regardless of lighting.
        self._warmup_remaining = max(0, int(warmup_frames))
        # Secondary safety net for the "live view is mainly black
        # with glitchy dots, camera didn't load" symptom: after the
        # fixed-prefix discard, ALSO keep discarding while the next
        # arriving frame is OVERWHELMINGLY black/zero (>= 98 % of
        # pixels at value 0). This is NOT the standard brightness
        # gate that the comment above rules out — that one passes
        # noisy/corrupted frames through. THIS one only catches the
        # case where the camera hasn't produced any real signal at
        # all and is still returning zero buffers. Bounded so a
        # genuinely dark scene doesn't stall the pipeline forever:
        # at most 30 extra frames (~1 s at 30 fps) past the fixed
        # prefix get the black-discard treatment. Empty / dim rooms
        # have noise (sensor read noise alone is > 2 % non-zero), so
        # this threshold doesn't false-positive on real content.
        self._black_discard_remaining = 60
        if self._inner.isOpened():
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name="ThreadedCvCapture",
                daemon=True,
            )
            self._reader_thread.start()

    def _reader_loop(self) -> None:
        try:
            self._reader_loop_impl()
        finally:
            # r18: deferred release handed over by release() when its
            # join timed out while we were blocked inside read(). The
            # handoff is decided under _handoff_lock so the token can
            # never be set after this thread already checked it.
            with self._handoff_lock:
                self._reader_exited = True
                do_release = bool(self._release_inner_on_exit)
                self._release_inner_on_exit = False
            if do_release:
                try:
                    self._release_inner_locked()
                finally:
                    with _PENDING_LOCK:
                        try:
                            _PENDING_RELEASES.remove(self)
                        except ValueError:
                            pass

    def _reader_loop_impl(self) -> None:
        consecutive_failures = 0
        # Tolerance window for transient cap.read() failures before we
        # mark the capture as dead. The previous 30 (~150 ms at the
        # 5 ms inter-attempt sleep below) was too tight: many USB
        # webcams go through a SECOND warm-up phase after their first
        # ok frame (auto-exposure + white balance settling) where
        # cap.read() returns ok=False for 200-800 ms. Hitting the old
        # ceiling during that window made isOpened() report False
        # permanently — which silently broke the engine's `_tick`
        # (no frames ever emitted, "Press START" placeholder
        # persists). 300 ≈ 1.5 s, generous enough to cover any
        # reasonable camera stall and still bound truly dead
        # captures (unplugged USB, app stole the device) within ~2 s.
        FAILURE_TOLERANCE = 300
        while not self._stop_event.is_set():
            try:
                ok, frame = self._inner.read()
            except Exception:
                self._read_error = True
                # Wake any waiter so they can observe the error
                # promptly instead of hitting the 100 ms timeout.
                self._fresh_frame_event.set()
                return
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= FAILURE_TOLERANCE:
                    self._read_error = True
                    self._fresh_frame_event.set()
                    return
                # Brief pause so we don't busy-spin if the camera
                # stalls for a moment.
                time.sleep(0.005)
                continue
            consecutive_failures = 0
            # Initial warm-up discard. Many USB / virtual cameras emit
            # the first few decoded frames in a partial / mostly-black
            # state (the symptom was "tutorial shows black with pixel
            # artifacts on first open"). We discard a fixed prefix here
            # rather than gating on brightness, because brightness-
            # based gates misclassify in both directions: corrupted
            # frames with random noise can read as bright (passed
            # through), and clean frames in a dim room can read as
            # dark (incorrectly drained). Discarding ~6 frames burns
            # ~200 ms of natural camera time, which is invisible to
            # the user (the placeholder text simply transitions to
            # the live feed slightly later).
            if self._warmup_remaining > 0:
                self._warmup_remaining -= 1
                continue
            if self._black_discard_remaining > 0:
                # Two-pronged check on the raw frame buffer:
                # (1) non-zero-pixel ratio < 2 % = mostly-black
                #     (sensor hasn't produced real signal yet).
                # (2) standard deviation < 8 = "glitchy dots" failure
                #     mode the user reported — frames have some
                #     non-zero pixels but no actual scene structure,
                #     just sparse uncorrelated noise.
                # Real scenes — even dim ones — have std > 15 due to
                # natural variance across pixels. < 8 is well below
                # any legitimate scene and catches the dot-pattern
                # camera-not-ready state.
                try:
                    if frame is not None and frame.size > 0:
                        nonzero = int(np.count_nonzero(frame))
                        ratio = nonzero / float(frame.size)
                        std = float(frame.std()) if frame.dtype.kind in ("u", "i", "f") else 0.0
                        if ratio < 0.02 or std < 8.0:
                            self._black_discard_remaining -= 1
                            continue
                except Exception:
                    pass
                self._black_discard_remaining = 0
            decoded_at = time.monotonic()
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_ts = decoded_at
            self._fresh_frame_event.set()

    def isOpened(self) -> bool:  # noqa: N802 (cv2 API parity)
        if self._closed or self._read_error:
            return False
        return bool(self._inner.isOpened())

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._closed:
            return False, None
        with self._frame_lock:
            frame = self._latest_frame
            ts = self._latest_frame_ts
            self._latest_frame = None
            self._fresh_frame_event.clear()
        if frame is not None:
            self._last_consumed_ts = ts
            return True, frame
        if not self._fresh_frame_event.wait(timeout=0.002):
            return False, None
        with self._frame_lock:
            frame = self._latest_frame
            ts = self._latest_frame_ts
            self._latest_frame = None
            self._fresh_frame_event.clear()
        if frame is None:
            return False, None
        self._last_consumed_ts = ts
        return True, frame

    def get(self, prop_id: int) -> float:
        try:
            return float(self._inner.get(prop_id))
        except Exception:
            return 0.0

    def set(self, prop_id: int, value: Any) -> bool:
        # r18: property writes on DirectShow rebuild the filter graph
        # (FOURCC / FPS) — serialise them against constructions and
        # releases on other threads. Rare (open-time tuning only).
        # Bounded wait: a False return means "keep current format".
        if not DSHOW_GRAPH_LOCK.acquire(timeout=_LOCK_WAIT_SET_S):
            return False
        try:
            return bool(self._inner.set(prop_id, value))
        except Exception:
            return False
        finally:
            DSHOW_GRAPH_LOCK.release()

    def grab(self) -> bool:
        with self._frame_lock:
            return self._latest_frame is not None

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.read()

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        self._fresh_frame_event.set()
        thread = self._reader_thread
        self._reader_thread = None
        if thread is not None:
            # DirectShow getPixels waits at most ~1000 ms per read, so
            # 2.5 s makes the deferred path exceptional, not routine.
            thread.join(timeout=2.5)
            with self._handoff_lock:
                still_running = not self._reader_exited and thread.is_alive()
                if still_running:
                    # r18: the reader is still inside _inner.read().
                    # Releasing the inner capture underneath it is a
                    # use-after-free. Hand the release to the reader
                    # thread and register it so new constructions wait.
                    self._release_inner_on_exit = True
                    self._reader_thread_for_join = thread
            if still_running:
                with _PENDING_LOCK:
                    _PENDING_RELEASES.append(self)
                return
        self._release_inner_locked()

    def _release_inner_locked(self) -> None:
        release_capture_serialised(self._inner)

# Author: Konstantin Markov

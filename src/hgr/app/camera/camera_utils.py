
from __future__ import annotations

import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from .threaded_cv_capture import ThreadedCvCapture


def _cam_log(msg: str) -> None:
    """Lightweight stderr log for camera diagnostics (tees into the
    Touchless debug log via main._StderrTee). Camera enumeration is
    infrequent, and the macOS port needs this visibility to tell a
    permission denial apart from a too-short open timeout."""
    try:
        sys.stderr.write(f"[camera] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _macos_camera_auth_status() -> str:
    """Best-effort AVFoundation camera authorization status string (macOS).

    AVAuthorizationStatus: 0=notDetermined, 1=restricted, 2=denied,
    3=authorized. (Note this is a DIFFERENT scale from TCC.db.auth_value,
    where 2 means allowed.)"""
    try:
        from AVFoundation import AVCaptureDevice  # type: ignore

        names = {0: "notDetermined", 1: "restricted", 2: "denied", 3: "authorized"}
        s = int(AVCaptureDevice.authorizationStatusForMediaType_("vide"))
        return f"AVFoundation auth={s} ({names.get(s, '?')})"
    except Exception as exc:  # noqa: BLE001
        return f"AVFoundation auth unavailable ({type(exc).__name__})"


@dataclass(frozen=True)
class CameraInfo:
    index: int
    backend: int
    backend_name: str
    display_name: str


_BACKEND_NAMES = {
    getattr(cv2, "CAP_AVFOUNDATION", -99999): "AVFoundation",
    getattr(cv2, "CAP_DSHOW", -99998): "DirectShow",
    getattr(cv2, "CAP_MSMF", -99997): "Media Foundation",
    getattr(cv2, "CAP_ANY", 0): "Default",
}


def backend_name(backend: int) -> str:
    return _BACKEND_NAMES.get(backend, f"Backend {backend}")


def _qt_video_device_names() -> List[str]:
    """Return the friendly names Qt reports for video capture devices.

    PySide6's QMediaDevices exposes the real device labels (e.g. "Iriun
    Webcam", "Integrated Camera", "USB Video Device") that OpenCV's
    VideoCapture does not. On Windows + DirectShow these typically
    enumerate in the same order as OpenCV's integer indices, so we can
    zip them together when the counts match. Falls back silently to an
    empty list if Qt's multimedia module is unavailable.
    """
    try:
        from PySide6.QtMultimedia import QMediaDevices
    except Exception:
        return []
    try:
        return [str(dev.description() or "").strip() for dev in QMediaDevices.videoInputs()]
    except Exception:
        return []


def _backend_candidates() -> List[int]:
    system = platform.system()

    if system == "Darwin":
        # On macOS, avoid CAP_ANY fallback because it tends to duplicate AVFoundation probing
        # and produces extra invalid-index noise/crashes in this app flow.
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            return [cv2.CAP_AVFOUNDATION]
        return [cv2.CAP_ANY]

    if system == "Windows":
        # DirectShow first — most consumer webcams negotiate frames
        # noticeably faster under DSHOW than MSMF. The EOS Webcam
        # Utility crash bug is per-filter, not per-machine: opening
        # cv2.VideoCapture(0, CAP_DSHOW) only instantiates index-0's
        # filter graph, NOT the EOS filter at index N. So routine
        # webcam opens on a system that happens to have EOS Webcam
        # Utility installed are still safe; only opening the EOS
        # index itself triggers the buggy filter. The
        # `_is_eos_camera_at_index` check in `open_camera_by_index`
        # handles that case by skipping DSHOW for EOS indices.
        backends: List[int] = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        unique: List[int] = []
        for backend in backends:
            if backend not in unique:
                unique.append(backend)
        return unique

    return [cv2.CAP_ANY]


def _candidate_indices(max_index: int) -> List[int]:
    if max_index <= 0:
        return []
    if platform.system() == "Darwin":
        # macOS/OpenCV AVFoundation has been unstable here when probing out-of-range indices.
        # Probe only index 0 during app discovery/preflight.
        return [0]
    return list(range(max_index))


@contextmanager
def _quiet_opencv_probe():
    get_level = getattr(cv2, "getLogLevel", None)
    set_level = getattr(cv2, "setLogLevel", None)
    previous_level = None
    if callable(get_level) and callable(set_level):
        try:
            previous_level = int(get_level())
            set_level(0)
        except Exception:
            previous_level = None
    try:
        yield
    finally:
        if previous_level is not None and callable(set_level):
            try:
                set_level(previous_level)
            except Exception:
                pass


def try_open_camera(
    index: int,
    backend: int,
    read_attempts: int = 10,
    read_interval: float = 0.03,
) -> Optional[cv2.VideoCapture]:
    with _quiet_opencv_probe():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            if platform.system() == "Darwin":
                _cam_log(f"open idx={index} backend={backend_name(backend)} -> NOT opened")
            return None

        for attempt in range(read_attempts):
            ok, _ = cap.read()
            if ok:
                if platform.system() == "Darwin":
                    _cam_log(f"open idx={index} backend={backend_name(backend)} -> OK (frame after {attempt + 1} reads)")
                return cap
            time.sleep(read_interval)

        cap.release()
        if platform.system() == "Darwin":
            _cam_log(f"open idx={index} backend={backend_name(backend)} -> opened but NO frame in {read_attempts} reads")
        return None


def request_camera_access_main_thread(max_index: int = 4) -> tuple[bool, str]:
    system = platform.system()
    if system != "Darwin":
        return True, "Camera permission prompt is not required on this platform."

    _cam_log(f"request_camera_access (main thread): {_macos_camera_auth_status()}")
    # Patience matters on macOS: the built-in MacBook camera can take well over
    # a second to deliver its first frame on a cold open (the green LED warm-up).
    # 120 reads x 0.03s ~= 3.6s matches the engine-open cold-start budget so a
    # working camera isn't misreported as "no camera detected".
    for backend in _backend_candidates():
        cap = try_open_camera(0, backend, read_attempts=120)
        if cap is not None:
            cap.release()
            _cam_log(f"request_camera_access -> OK via {backend_name(backend)}")
            return True, "Camera access confirmed on camera 0."

    _cam_log("request_camera_access -> FAILED on all backends")
    return False, (
        "macOS camera access was not granted yet. Approve camera access when prompted, "
        "or enable it in System Settings > Privacy & Security > Camera for Terminal or your packaged app, then try again."
    )


def is_eos_or_canon_name(display_name: str) -> bool:
    """Return True if the camera display name looks like Canon EOS
    Webcam Utility or another Canon EOS-style virtual camera.

    Used to special-case the Windows backend selection: EOS Webcam
    Utility's DirectShow filter has a documented segfault bug during
    cv2.VideoCapture(idx, CAP_DSHOW) construction on cold camera state.
    Routing EOS-named cameras through CAP_MSMF only — without the
    DirectShow fallback the rest of the codebase relies on — is enough
    to avoid the crash on EOS Webcam Utility v2.0+ (which registers
    a Media Foundation Frame Source). Older v1.x EOS Webcam Utility
    is DirectShow-only; for those installs the camera won't open via
    MSMF and we return None instead of falling through to a crash."""
    name = str(display_name or "").upper()
    return "EOS" in name or "CANON" in name


def _is_eos_camera_at_index(index: int) -> bool:
    """Look up the Qt-reported name at this index and return True if
    it matches the EOS / Canon detection rule. Returns False on
    non-Windows hosts and when Qt can't enumerate."""
    if platform.system() != "Windows":
        return False
    if index < 0:
        return False
    try:
        names = _qt_video_device_names()
    except Exception:
        return False
    if index >= len(names):
        return False
    return is_eos_or_canon_name(names[index])


def list_cameras_qt_only() -> List[CameraInfo]:
    """Enumerate cameras using Qt's QMediaDevices ONLY — no cv2 probe.

    Qt's QMediaDevices.videoInputs() asks the OS for the list of
    registered video devices without instantiating their underlying
    capture pipelines. cv2.VideoCapture(i, CAP_DSHOW), by contrast,
    builds a full DirectShow filter graph during construction —
    which on Windows touches every registered video filter on the
    system. A buggy third-party filter (notably some Canon EOS
    Webcam Utility releases when the camera isn't fully initialised
    yet) can segfault inside that graph instantiation, taking the
    whole Touchless process down with no error dialog.

    This Qt-only path is safe to run at app launch and on every
    "show me cameras" UI moment. The full cv2-probe path
    (`list_available_cameras`) is still available for explicit
    deep-refresh actions; that one verifies frames actually arrive,
    but it's also the one that can crash on bad filters.

    The CameraInfo entries returned here use CAP_DSHOW as a default
    backend hint — when the user actually picks one and starts the
    engine, `open_camera_by_index` walks the real backend list and
    picks whichever opens.
    """
    if platform.system() != "Windows":
        # Mac/Linux don't have the same "any registered filter can
        # crash enumeration" failure mode, so callers there are fine
        # using the cv2 path. We return [] here so callers fall back
        # to it explicitly rather than silently skipping enumeration.
        return []
    qt_names = _qt_video_device_names()
    if not qt_names:
        return []
    default_backend = getattr(cv2, "CAP_DSHOW", getattr(cv2, "CAP_ANY", 0))
    cameras: List[CameraInfo] = []
    for index, raw_name in enumerate(qt_names):
        name = str(raw_name).strip()
        if not name:
            display = f"Camera {index} ({backend_name(default_backend)})"
        else:
            display = f"{name} (Camera {index})"
        cameras.append(
            CameraInfo(
                index=index,
                backend=default_backend,
                backend_name=backend_name(default_backend),
                display_name=display,
            )
        )
    return cameras


def list_available_cameras(max_index: int = 8) -> List[CameraInfo]:
    discovered: List[CameraInfo] = []
    consecutive_misses = 0
    stop_after_misses = 2 if platform.system() == "Windows" else max_index
    qt_names = _qt_video_device_names()

    # macOS: same cold-start patience as the access check (built-in camera is
    # slow to first-frame); Windows keeps its short probe to stay snappy.
    probe_read_attempts = 120 if platform.system() == "Darwin" else 10
    if platform.system() == "Darwin":
        _cam_log(f"list_available_cameras start: {_macos_camera_auth_status()}")
    for index in _candidate_indices(max_index):
        found_for_index = False
        for backend in _backend_candidates():
            cap = try_open_camera(index, backend, read_attempts=probe_read_attempts)
            if cap is None:
                continue
            cap.release()
            # Prefer Qt's friendly device name at the matching index (e.g.
            # "Iriun Webcam"), and only fall back to "Camera N (Backend)"
            # when Qt either couldn't enumerate or returned fewer entries.
            friendly = qt_names[index] if index < len(qt_names) else ""
            if friendly:
                display = f"{friendly} (Camera {index})"
            else:
                display = f"Camera {index} ({backend_name(backend)})"
            discovered.append(
                CameraInfo(
                    index=index,
                    backend=backend,
                    backend_name=backend_name(backend),
                    display_name=display,
                )
            )
            found_for_index = True
            break
        if platform.system() == "Windows":
            if found_for_index:
                consecutive_misses = 0
            else:
                consecutive_misses += 1
                if discovered and consecutive_misses >= stop_after_misses:
                    break

    if platform.system() == "Darwin":
        _cam_log(f"list_available_cameras -> {len(discovered)} found ({[c.display_name for c in discovered]})")
    return discovered


def find_first_available_camera(max_index: int = 8) -> Tuple[Optional[int], Optional[cv2.VideoCapture]]:
    cameras = list_available_cameras(max_index)
    if not cameras:
        return None, None
    selected = cameras[0]
    cap = try_open_camera(selected.index, selected.backend)
    if cap is None:
        return None, None
    return selected.index, cap


def open_camera_by_index(index: int, max_index: int = 8) -> Tuple[Optional[CameraInfo], Optional[cv2.VideoCapture]]:
    # On macOS this app only supports index 0 for direct camera access in order to avoid
    # unstable AVFoundation probing of non-existent indices.
    if platform.system() == "Darwin" and index != 0:
        return None, None

    # Slow-start virtual cameras (Canon EOS Webcam Utility, OBS
    # Virtual Camera with no source bound yet, NDI Tools) routinely
    # take 1-3 s to deliver their first frame after cv2.VideoCapture
    # opens — the underlying USB / driver pipeline isn't ready
    # synchronously like a built-in webcam's. With the default 10
    # × 30 ms = 300 ms wait, both DSHOW and MSMF would mark the open
    # as failed and the user's saved-preferred camera never connects
    # even though Windows Camera app eventually does. Bumping the
    # wait to ~3 s per backend covers EOS Webcam's typical cold-start
    # without slowing the working-webcam case (try_open_camera
    # returns the moment ANY frame arrives, not the full timeout).
    cold_start_attempts = 100  # 100 × 30 ms = 3 s per backend
    # EOS Webcam Utility safeguard. cv2.VideoCapture(idx, CAP_DSHOW)
    # for EOS constructs a DirectShow filter graph that loads EOS's
    # filter — and that filter has a documented segfault path on
    # cold camera state, which kills the whole Touchless process
    # with no error dialog. For EOS cameras we skip CAP_DSHOW
    # entirely: try MSMF first (in-process, fast when it works),
    # and if MSMF can't deliver, fall through to the ffmpeg
    # subprocess path AFTER this loop (handled below). The MSMF
    # window is doubled here vs. non-EOS paths because EOS Webcam
    # Utility's MSMF Frame Source can take 3-6 s to negotiate its
    # first frame — the previous 3 s cap was timing out before MSMF
    # could deliver, dropping the friend's tutorial into the
    # "couldn't open camera" error path even though MSMF did
    # ultimately work for them in b2. 6 s gives the slow path room
    # to succeed without dragging down healthy webcams (we still
    # return the moment any frame arrives).
    if _is_eos_camera_at_index(index):
        msmf_backend = getattr(cv2, "CAP_MSMF", None)
        if msmf_backend is None:
            backends_to_try: List[int] = []
        else:
            backends_to_try = [msmf_backend]
        eos_attempts = 200  # 200 × 30 ms = 6 s
    else:
        backends_to_try = _backend_candidates()
        eos_attempts = cold_start_attempts
    for backend in backends_to_try:
        cap = try_open_camera(index, backend, read_attempts=eos_attempts)
        if cap is not None:
            info = CameraInfo(
                index=index,
                backend=backend,
                backend_name=backend_name(backend),
                display_name=f"Camera {index} ({backend_name(backend)})",
            )
            # Wrap the synchronous cv2.VideoCapture in a reader-thread
            # shim. The wrapper does two things:
            #
            # 1) cap.read() returns immediately with the latest buffered
            #    frame instead of blocking ~33 ms (a 30 fps frame
            #    interval) on the main thread. Without this the gesture
            #    loop's main-thread cap.read call was the dominant
            #    cycle cost on the OpenCV fallback path and starved
            #    Qt's paint events.
            #
            # 2) Drops the first few decoded frames internally so
            #    consumers never see the partial / mostly-black frames
            #    many cameras emit immediately after open (the symptom
            #    was "tutorial shows black with pixel artifacts" on
            #    first launch). The old synchronous warmup_capture(cap)
            #    here did the same job but blocked the UI thread for
            #    up to ~2 s during open AND mis-classified frames in
            #    dim rooms vs. corrupted-noise frames — fixed-prefix
            #    discard in the reader thread is simpler and works on
            #    every camera regardless of lighting.
            return info, ThreadedCvCapture(cap)

    # EOS subprocess-isolated fallback. cv2.VideoCapture failed for
    # every in-process backend we're willing to try (MSMF only for
    # EOS — DSHOW is deliberately skipped). For EOS specifically we
    # hand the open off to ffmpeg.exe, which builds the DirectShow
    # filter graph in a CHILD process. If the EOS Webcam Utility
    # filter then segfaults inside graph construction, only ffmpeg
    # dies — Touchless keeps running and just sees ffmpeg's stdout
    # pipe go quiet. If ffmpeg can construct the graph (it usually
    # can, because ffmpeg's DSHOW handling is more robust than
    # OpenCV's), we get a full-rate MJPEG capture back over a pipe,
    # wrapped in the same shape as a cv2.VideoCapture for the rest
    # of the codebase to consume transparently.
    if _is_eos_camera_at_index(index):
        try:
            qt_names = _qt_video_device_names()
        except Exception:
            qt_names = []
        if 0 <= index < len(qt_names):
            device_name = str(qt_names[index] or "").strip()
            if device_name:
                try:
                    from .ffmpeg_capture import open_ffmpeg_cap_with_fps_fallback
                    ffmpeg_cap = open_ffmpeg_cap_with_fps_fallback(
                        device_name, width=1280, height=720
                    )
                except Exception:
                    ffmpeg_cap = None
                if ffmpeg_cap is not None and ffmpeg_cap.isOpened():
                    info = CameraInfo(
                        index=index,
                        backend=-1,
                        backend_name="ffmpeg-dshow",
                        display_name=f"{device_name} (Camera {index}, ffmpeg)",
                    )
                    # ffmpeg_cap is already async-buffered internally
                    # (ffmpeg pipes raw BGR24 into our reader thread),
                    # so no ThreadedCvCapture wrapper needed here.
                    return info, ffmpeg_cap
    return None, None


def try_open_camera_url(url: str, read_attempts: int = 12) -> Optional[cv2.VideoCapture]:
    """Open an IP-webcam-style stream URL (MJPEG / RTSP / HTTP) and verify a frame arrives.

    Returns a `cv2.VideoCapture` on success, or None. Blocks for up to a few
    seconds while waiting for the first frame — callers that need
    responsiveness (e.g. a Test button in Settings) should run this on a
    worker thread.
    """
    clean = str(url or "").strip()
    if not clean:
        return None
    with _quiet_opencv_probe():
        try:
            cap = cv2.VideoCapture(clean)
        except Exception:
            return None
        if not cap.isOpened():
            cap.release()
            return None
        for _ in range(read_attempts):
            ok, _ = cap.read()
            if ok:
                # Initial garbage-frame discard happens inside
                # ThreadedCvCapture when the caller wraps this cap
                # (open_phone_camera_url does). No synchronous drain
                # here so the Test-Phone-URL settings button stays
                # responsive while the open path succeeds.
                return cap
            time.sleep(0.08)
        cap.release()
        return None


def open_phone_camera_url(url: str) -> Tuple[Optional[CameraInfo], Optional[cv2.VideoCapture]]:
    cap = try_open_camera_url(url)
    if cap is None:
        return None, None
    info = CameraInfo(
        index=-1,
        backend=0,
        backend_name="Phone",
        display_name=f"Phone Camera ({url})",
    )
    # Same async-reader wrap as open_camera_by_index — phone-URL
    # captures are over the network and read() can block well past
    # one frame interval if the phone hiccups. Off-main-thread.
    return info, ThreadedCvCapture(cap)


def open_preferred_or_first_available(preferred_index: Optional[int], max_index: int = 8) -> Tuple[Optional[CameraInfo], Optional[cv2.VideoCapture]]:
    if preferred_index is not None:
        info, cap = open_camera_by_index(preferred_index, max_index=max_index)
        if info is not None and cap is not None:
            return info, cap

    cameras = list_available_cameras(max_index)
    if not cameras:
        return None, None

    selected = cameras[0]
    # Same cold-start tolerance as the saved-preferred path. Once
    # we've decided this is the camera to bring up, give it the
    # full ~3 s window so a slow virtual camera (EOS Webcam Utility,
    # OBS Virtual Camera) doesn't time out on the connect step
    # after surviving the faster enumeration probe.
    cap = try_open_camera(selected.index, selected.backend, read_attempts=100)
    if cap is None:
        return None, None
    # Wrap in the threaded reader so behavior matches the preferred-
    # index path (open_camera_by_index also wraps): non-blocking
    # cap.read() AND internal first-frames-discard for the open-time
    # garbage every camera emits.
    return selected, ThreadedCvCapture(cap)

# Author: Konstantin Markov

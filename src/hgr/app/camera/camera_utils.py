
from __future__ import annotations

import platform
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from .threaded_cv_capture import ThreadedCvCapture


def _cv2_open_with_timeout(
    index: int,
    backend: int,
    timeout_seconds: float = 6.0,
) -> Optional[cv2.VideoCapture]:
    """Construct a `cv2.VideoCapture(index, backend)` in a background
    daemon thread, abandoning the thread (NOT killing it — Python has
    no portable way to interrupt a native call) if construction blocks
    beyond `timeout_seconds`.

    Why this exists:
    On Windows, `cv2.VideoCapture(idx, CAP_DSHOW)` builds a full
    DirectShow filter graph during construction. When another process
    (Razer Synapse, Windows Camera app, a crashed Touchless test
    session that didn't clean up, etc.) is holding the camera handle,
    Windows' DSHOW infrastructure can block this constructor for
    60-120 seconds before timing out. Without a wrapper, Touchless's
    cold-start camera scan would freeze the splash for several
    minutes — the user perceives this as "the app hung", quits with
    Ctrl+C, and tries again, often making the device-state worse.

    With this wrapper:
      * Healthy cameras open in <1 s — well under the timeout.
      * Locked cameras let the calling thread give up after 6 s and
        try the next backend / fall through to the OpenCV-fallback
        path or, eventually, "no camera" UI. The leaked background
        thread is a daemon, so it dies when Python exits; while
        Python is still alive it stays blocked on the OS call until
        Windows times out internally and the thread cleanly returns.

    Returns the cv2.VideoCapture on success, or None on timeout /
    construction exception. Caller is responsible for the rest of
    the open dance (cap.isOpened(), the read_attempts warmup loop).
    """
    result: list[Optional[cv2.VideoCapture]] = [None]

    def _worker() -> None:
        try:
            result[0] = cv2.VideoCapture(index, backend)
        except Exception:
            result[0] = None

    thread = threading.Thread(
        target=_worker,
        name=f"cv2-open-idx{index}-bk{backend}",
        daemon=True,
    )
    thread.start()
    thread.join(timeout=float(timeout_seconds))
    if thread.is_alive():
        try:
            sys.stderr.write(
                f"[camera_utils] cv2.VideoCapture(index={index}, backend={backend}) "
                f"blocked beyond {timeout_seconds:.1f}s timeout — likely the camera "
                f"is held by another process (Razer Synapse, Windows Camera, OBS, "
                f"a crashed Touchless test session, etc.). Skipping this backend; "
                f"the leaked background thread will resolve when Windows DSHOW "
                f"times out internally (no user impact, daemon thread dies with "
                f"the process).\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        return None
    return result[0]


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
        # Construct via timeout-wrapped helper. Healthy cameras open
        # in <1 s; a locked DSHOW device would otherwise block this
        # constructor for 60-120 s while Windows times out the
        # contended handle, freezing the whole Touchless splash.
        # 6 s is plenty for cold-start virtual cameras (EOS Webcam
        # Utility, OBS Virtual Camera) — those deliver their first
        # frame slowly inside the read_attempts loop below, not
        # during the VideoCapture() constructor itself.
        cap = _cv2_open_with_timeout(index, backend, timeout_seconds=6.0)
        if cap is None:
            return None
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None

        for _ in range(read_attempts):
            ok, _ = cap.read()
            if ok:
                return cap
            time.sleep(read_interval)

        try:
            cap.release()
        except Exception:
            pass
        return None


def request_camera_access_main_thread(max_index: int = 4) -> tuple[bool, str]:
    system = platform.system()
    if system != "Darwin":
        return True, "Camera permission prompt is not required on this platform."

    for backend in _backend_candidates():
        cap = try_open_camera(0, backend, read_attempts=12)
        if cap is not None:
            cap.release()
            return True, "Camera access confirmed on camera 0."

    return False, (
        "macOS camera access was not granted yet. Approve camera access when prompted, "
        "or enable it in System Settings > Privacy & Security > Camera for Terminal or your packaged app, then try again."
    )


# r50: auto-detect classifier for the r49 short-shutter camera hint.
#
# Allowlist is checked BEFORE the denylist so a Kiyo Pro (which contains
# 'kiyo') can never be resolved to 'generic' even if a partial substring
# also matched a denylist term. Both lists are conservative — the
# denylist is trimmed to high-confidence keyword substrings ('realtek',
# 'sonix', 'chicony', literal generic UVC names). Broad terms like
# 'integrated camera' and 'hd webcam' were intentionally excluded to
# avoid false-positives on premium built-in laptop cameras.
_R50_PREMIUM_CAMERA_KEYWORDS = (
    "kiyo",
    "brio",
    "c920",
    "c922",
    "c930",
    "streamcam",
    "elgato facecam",
    "insta360 link",
    "opal",
    "poly studio",
    "logitech mx brio",
    "sony imx",
)
_R50_GENERIC_UVC_KEYWORDS = (
    "full hd 1080p webcam",
    "usb2.0 camera",
    "usb camera",
    "uvc camera",
    "general webcam",
    "realtek",
    "sonix",
    "chicony",
)


def classify_camera_shutter_hint(display_name: str) -> Optional[bool]:
    """Return True if the camera's display name matches a known generic
    UVC driver family that benefits from the short-shutter hint;
    False if it matches a known premium camera family that must NOT
    receive the hint; None if the name is unknown (caller should
    fall through to the user's explicit config value).

    Runtime-only classifier — the caller does NOT persist the result
    to config. Every camera-open re-evaluates from the current
    display_name, which stays deterministic across restarts and can
    never diverge from a user's explicit Settings choice.
    """
    name = str(display_name or "").lower().strip()
    if not name:
        return None
    for premium in _R50_PREMIUM_CAMERA_KEYWORDS:
        if premium in name:
            return False
    for generic in _R50_GENERIC_UVC_KEYWORDS:
        if generic in name:
            return True
    return None


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

    for index in _candidate_indices(max_index):
        found_for_index = False
        for backend in _backend_candidates():
            cap = try_open_camera(index, backend)
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

    # v1.1.7 Windows ffmpeg-DShow enumeration fallback. On some
    # driver/OS combos (post Razer Synapse install, WMF-hidden UVC
    # devices, or a camera briefly held by a background app while
    # we probed), QMediaDevices returns empty AND every OpenCV
    # backend fails to open — so the user sees "no camera" even
    # though ffmpeg's DShow enumeration can see the device fine.
    # If ffmpeg's list is non-empty when our probe found nothing,
    # synthesize CameraInfo entries so the app can still open via
    # the ffmpeg-subprocess path (open_camera_by_index has the
    # matching fallback). Zero cost when the OpenCV probe already
    # found the camera — this branch only runs when discovered is
    # empty AND we're on Windows.
    if not discovered and platform.system() == "Windows":
        import sys as _sys
        try:
            _sys.stderr.write(
                "[camera-enum] OpenCV probe returned 0 cameras; trying ffmpeg-DShow fallback\n"
            )
            _sys.stderr.flush()
        except Exception:
            pass
        dshow_devices: list[str] = []
        try:
            from .ffmpeg_capture import list_dshow_video_devices
            dshow_devices = list_dshow_video_devices()
        except Exception as _exc:
            try:
                _sys.stderr.write(
                    f"[camera-enum] list_dshow_video_devices raised: {_exc!r}\n"
                )
                _sys.stderr.flush()
            except Exception:
                pass
            dshow_devices = []
        try:
            _sys.stderr.write(
                f"[camera-enum] ffmpeg-DShow fallback returned {len(dshow_devices)} device(s): {dshow_devices!r}\n"
            )
            _sys.stderr.flush()
        except Exception:
            pass
        for idx, name in enumerate(dshow_devices):
            discovered.append(
                CameraInfo(
                    index=idx,
                    backend=-1,
                    backend_name="ffmpeg-dshow",
                    display_name=f"{name} (Camera {idx})",
                )
            )

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
                    # C27: 640x480 to match Default's OpenCV cap res.
                    # Higher res introduced a persistent 1-2 s live-
                    # viewer lag through the frame-copy pipeline.
                    # v1.1.7.1: luma_min_threshold=50 auto-downshifts
                    # 60→30 fps when the driver responds to the higher
                    # rate by cutting shutter below usable brightness
                    # (HP HD Camera and similar built-in webcams that
                    # can't sustain 60 fps in indoor lighting).
                    ffmpeg_cap = open_ffmpeg_cap_with_fps_fallback(
                        device_name, width=640, height=480,
                        luma_min_threshold=50.0,
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
    # v1.1.7 general Windows ffmpeg-DShow open fallback. Every
    # cv2.VideoCapture backend failed above — usually because Qt/
    # OpenCV rely on Windows Media Foundation which sometimes
    # doesn't see UVC devices that DirectShow does (post Synapse
    # install for Kiyo Pro, some Razer/Discord/Teams driver states
    # that briefly park the camera). ffmpeg's DShow enumeration
    # runs in a child process and typically succeeds where the
    # in-process cv2.VideoCapture doesn't. Same shape ffmpeg cap
    # the perf-mode path already uses, so downstream code is
    # transparent to which path we came from.
    if platform.system() == "Windows":
        try:
            from .ffmpeg_capture import list_dshow_video_devices, open_ffmpeg_cap_with_fps_fallback
            dshow_devices = list_dshow_video_devices()
        except Exception:
            dshow_devices = []
        if 0 <= index < len(dshow_devices):
            device_name = str(dshow_devices[index] or "").strip()
            if device_name:
                try:
                    # C27: 640x480 to match Default's OpenCV cap res.
                    # Higher res introduced a persistent 1-2 s live-
                    # viewer lag through the frame-copy pipeline.
                    # v1.1.7.1: luma_min_threshold=50 auto-downshifts
                    # 60→30 fps if the driver responds by cutting
                    # shutter too aggressively (see camera_utils.py
                    # EOS path for the full rationale).
                    ffmpeg_cap = open_ffmpeg_cap_with_fps_fallback(
                        device_name, width=640, height=480,
                        luma_min_threshold=50.0,
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

"""FFmpeg-backed camera capture for Lite Mode on Windows.

Why this module exists:
OpenCV's `cv2.VideoCapture` on Windows uses DirectShow / Media Foundation
through a thin wrapper that does not reliably honour `CAP_PROP_FOURCC`
requests on common consumer webcams (Razer Kiyo, many cheap 1080p
webcams, etc.). Those drivers latch the negotiated stream format on
the first frame read and silently stay on YUY2 (uncompressed) — even
when we use OpenCV 4.5+'s 3-arg constructor that's supposed to apply
params before init. The result is a hard ~30 fps ceiling at 720p over
USB 2.0 because YUY2's bandwidth saturates the bus, plus a soft ~10 fps
ceiling at 1080p for the same reason.

ffmpeg's DirectShow input *does* honour `-vcodec mjpeg` reliably, so a
capture pipeline of:

    ffmpeg -f dshow -vcodec mjpeg -framerate <fps> -video_size <wxh> \\
           -i video=<friendly_name> \\
           -f rawvideo -pix_fmt bgr24 pipe:1

talks to the camera in MJPG, decodes the JPEG frames inside ffmpeg's
process (free CPU core), and pipes raw BGR24 bytes back to us at a
fixed frame size we can read deterministically. This unlocks 60 fps at
720p on cameras that advertise it but were stuck at 30 fps under
OpenCV, and 30 fps at 1080p on cameras that were stuck at ~10 fps.

The class below exposes the subset of the `cv2.VideoCapture` interface
the rest of this codebase touches: `isOpened()`, `read()`,
`release()`, `get(prop_id)`, `set(prop_id, value)`. The engine code
treats us as a drop-in replacement for the OpenCV capture object.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def locate_ffmpeg() -> str | None:
    """Find the ffmpeg executable. Prefers a copy that lives next to
    the running interpreter / packaged exe (PyInstaller layout) over
    PATH, since the installer ships ffmpeg there and we want the
    bundled one to win on user machines that have an older system
    ffmpeg installed.

    PyInstaller onedir layout (current): the main Touchless.exe sits
    at dist/Touchless/Touchless.exe but bundled binaries (added via
    the spec's `binaries` list) land in dist/Touchless/_internal/.
    So `Path(sys.executable).resolve().with_name(exe_name)` looks
    next to the exe and misses the bundled copy in `_internal/`. We
    check both locations explicitly here so the bundled ffmpeg is
    always discoverable in the packaged build, regardless of which
    PyInstaller layout convention is current.
    """
    exe_name = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    candidates: list[Path] = []
    try:
        exe_path = Path(sys.executable).resolve()
        # Same directory as the main executable (covers older
        # PyInstaller onefile-extracted layouts and any future
        # PyInstaller change that moves bundled binaries to the
        # bundle root).
        candidates.append(exe_path.with_name(exe_name))
        # Sibling _internal/ directory (PyInstaller 5.x+ onedir
        # default — this is where the bundled ffmpeg actually lands
        # for our current spec).
        candidates.append(exe_path.parent / "_internal" / exe_name)
    except Exception:
        pass
    try:
        candidates.append(Path.cwd() / exe_name)
    except Exception:
        pass
    for candidate in candidates:
        try:
            if candidate.exists():
                return str(candidate)
        except Exception:
            pass
    return shutil.which("ffmpeg") or shutil.which(exe_name)


def list_dshow_video_devices(ffmpeg_path: str | None = None) -> list[str]:
    """Ask ffmpeg for the DirectShow video device list. Returns the
    friendly names ffmpeg expects after `-i video=`. The names are
    case- and whitespace-sensitive, so we surface them verbatim."""
    if not sys.platform.startswith("win"):
        return []
    path = ffmpeg_path or locate_ffmpeg()
    if not path:
        return []
    try:
        completed = subprocess.run(
            [path, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=6.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return []
    # ffmpeg writes the device list to stderr. Lines look like:
    #   [dshow @ 0x...] "Razer Kiyo" (video)
    #   [dshow @ 0x...]   Alternative name "@device_pnp_..."
    # We want only the friendly-name rows tagged (video).
    text = (completed.stderr or "") + (completed.stdout or "")
    pattern = re.compile(r'"([^"]+)"\s*\(video\)', re.IGNORECASE)
    devices: list[str] = []
    for match in pattern.finditer(text):
        name = match.group(1).strip()
        if name and name not in devices:
            devices.append(name)
    return devices


def resolve_dshow_device_for_index(index: int, qt_name_hint: str = "") -> str | None:
    """Best-effort mapping from a CameraInfo.index to the DirectShow
    friendly name ffmpeg wants. Tries an exact / case-insensitive match
    against the Qt-reported display name first, then falls back to the
    Nth device in ffmpeg's enumerated list. Returns None when nothing
    plausibly maps — caller should then fall back to the OpenCV path."""
    devices = list_dshow_video_devices()
    if not devices:
        return None
    hint = (qt_name_hint or "").strip()
    if hint:
        # Strip our "(Camera N)" suffix from the Qt-side display name
        # before comparing — DirectShow doesn't include that bracket.
        hint_clean = re.sub(r"\s*\(Camera\s+\d+\)\s*$", "", hint).strip()
        for candidate in devices:
            if candidate.lower() == hint_clean.lower():
                return candidate
        for candidate in devices:
            if candidate and (candidate.lower() in hint_clean.lower() or hint_clean.lower() in candidate.lower()):
                return candidate
    if 0 <= int(index) < len(devices):
        return devices[int(index)]
    return None


def open_ffmpeg_cap_with_fps_fallback(
    device_name: str,
    *,
    width: int = 1280,
    height: int = 720,
    fps_candidates: tuple[int, ...] = (60, 30),
) -> "FfmpegMjpegCapture | None":
    """Try opening the ffmpeg MJPG cap at decreasing frame rates.

    Many consumer webcams advertise 1280x720 MJPG but only at 30 fps —
    requesting 60 fps causes ffmpeg to emit "Could not set video
    options" and exit before producing any frame. Rather than failing
    over to OpenCV (which forces YUY2 and adds main-thread copy cost),
    we just retry with the next fps in the list. 30 fps MJPG is still
    a major win over OpenCV's YUY2 — the hand-tracking loop sees fresh
    decompressed BGR frames without the per-frame uncompress cost
    YUY2 carries.

    Returns the opened capture, or None if every candidate failed.
    The caller should fall through to the OpenCV path on None.
    """
    for fps in fps_candidates:
        cap = FfmpegMjpegCapture(
            device_name,
            width=width,
            height=height,
            fps=fps,
        )
        if cap.isOpened():
            try:
                print(
                    f"[ffmpeg_capture] engaged at {width}x{height} @ {fps} fps MJPG",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
            return cap
        try:
            cap.release()
        except Exception:
            pass
        try:
            print(
                f"[ffmpeg_capture] {fps} fps unsupported by camera — trying next candidate",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
    return None


class FfmpegMjpegCapture:
    """A `cv2.VideoCapture`-shaped wrapper around an ffmpeg subprocess.

    Lifecycle:
      cap = FfmpegMjpegCapture(device_name="Razer Kiyo")
      if cap.isOpened():
          ok, frame = cap.read()  # frame is HxWx3 BGR np.ndarray
      cap.release()

    Threading model: ffmpeg writes raw BGR frames to stdout. A daemon
    reader thread on our side reads exactly W*H*3 bytes per frame off
    the pipe and stores the latest one under a lock; `read()` returns
    that latest frame. We keep just the latest — same single-frame
    "drop stale" behaviour we'd get from `cap.set(BUFFERSIZE, 1)` on
    the OpenCV path. This matches what every other consumer in the
    engine expects (gesture loop wants the freshest frame, not a
    queue of pending ones)."""

    def __init__(
        self,
        device_name: str,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        ffmpeg_path: str | None = None,
        startup_timeout_seconds: float = 2.5,
    ) -> None:
        self._device_name = device_name
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._ffmpeg_path = ffmpeg_path or locate_ffmpeg()
        # Lowered startup_timeout from 8s to 2.5s. The 8s value was
        # there to forgive a DSHOW driver that hadn't yet released
        # the device after an in-app camera-restart — but on the
        # cold-start path (e.g. tutorial launched from Settings while
        # the main app is closed) there's no lingering driver lock,
        # so 8s just means an extra 16 s of dead time when ffmpeg
        # genuinely can't open the device (failure × two fps
        # candidates). 2.5 s is plenty for the legitimate cold-open
        # case (typical first-frame latency is <500 ms when ffmpeg
        # works at all). The rare DSHOW-still-busy-after-restart
        # case now falls through to OpenCV faster — the user
        # perceives "camera came up" instead of "camera frozen for
        # 16 s then came up".
        self._startup_timeout = float(startup_timeout_seconds)
        # Known-fatal stderr substrings that mean ffmpeg has decided
        # it can't open this device: when any of these appear in the
        # stderr_loop's captured output we wake the startup waiter
        # immediately instead of sitting through the timeout. Without
        # this, ffmpeg can print "Error opening input file" within
        # the first 100 ms but stay in a half-alive state that holds
        # the pipes open until the OS reaps it, blocking us for the
        # full timeout. Patterns are case-insensitive substring
        # matches against the joined stderr buffer.
        self._fatal_stderr_patterns = (
            "could not set video options",
            "error opening input",
            "could not find video device",
            "i/o error",
        )
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        # Wall-clock time (time.monotonic) at which the reader
        # thread finished decoding _latest_frame. Used by callers
        # to measure end-to-end pipeline latency (capture → display).
        # Set every time _latest_frame is overwritten.
        self._latest_frame_ts: float = 0.0
        # Snapshot of _latest_frame_ts at the moment of the most
        # recent successful read(). The caller can query this AFTER
        # read() returns to know "when was THIS frame decoded".
        self._last_consumed_ts: float = 0.0
        # Set when ffmpeg has produced any frame at all — used by
        # _start to know "the pipe is alive". Stays set forever
        # after the first frame.
        self._first_frame_event = threading.Event()
        # Set every time a NEW frame lands in _latest_frame, cleared
        # by read() before it waits. Lets read() block exactly until
        # the next fresh frame arrives (capping at camera fps)
        # instead of busy-returning False whenever the consumer
        # outpaces the producer. Without this, a fast main-thread
        # loop (singleShot pacing post engine-async) would call
        # read() faster than 60 fps, see None, return False, and
        # exit _tick early — capping the gesture loop at the
        # 15-ms periodic timer fallback (~27 fps).
        self._fresh_frame_event = threading.Event()
        self._opened = False
        self._read_error = False
        # Fixed-prefix warmup discard. EOS Webcam Utility and some
        # other DSHOW filters emit a few placeholder frames immediately
        # after open (cached single-color frame, or auto-exposure
        # still settling) before real content starts. Drop the first
        # few decoded frames internally so consumers never see them
        # — matches the same approach used in ThreadedCvCapture for
        # cv2-based opens.
        self._warmup_remaining = 6
        # Capture ffmpeg's stderr to a memory buffer + a stderr
        # mirror so we can surface the actual failure reason on
        # startup. The previous configuration discarded stderr to
        # DEVNULL, which made "ffmpeg cap startup failed" a black
        # box — the user couldn't see whether the camera was busy,
        # the format was rejected, or the device name was wrong.
        self._stderr_log: list[str] = []
        self._stderr_thread: Optional[threading.Thread] = None
        self._opened = self._start()

    @property
    def device_name(self) -> str:
        return self._device_name

    def _start(self) -> bool:
        if not self._ffmpeg_path:
            print(
                "[ffmpeg_capture] _start aborted: no ffmpeg binary found "
                "(locate_ffmpeg returned None and no override passed)",
                file=sys.stderr,
                flush=True,
            )
            return False
        if not sys.platform.startswith("win"):
            print(
                f"[ffmpeg_capture] _start aborted: platform is "
                f"{sys.platform!r} (dshow input is Windows-only)",
                file=sys.stderr,
                flush=True,
            )
            return False
        if not self._device_name:
            print(
                "[ffmpeg_capture] _start aborted: empty device name",
                file=sys.stderr,
                flush=True,
            )
            return False
        # rtbufsize=32M: ffmpeg keeps a large input MJPG buffer so it
        # can decode + write BGR to pipe at full camera rate without
        # being stalled by the consumer. Smaller values (3M, 12M)
        # tested measurably worse — the OS pipe between ffmpeg's
        # stdout and our reader is tiny (~64KB on Windows, less than
        # one BGR frame), so without enough rtbufsize ffmpeg's BGR
        # write back-pressures the camera capture and the consumer
        # ends up waiting ~22 ms per cap.read() call. The latency
        # buffering this introduces (~200 ms of pipelined frames) is
        # hidden by the consumer always reading the freshest frame
        # from latest_frame and dropping older ones — see
        # FfmpegMjpegCapture.read.
        # rtbufsize 512K: bounds input queue at ~5 frames worst
        # case (1280x720 MJPG at typical compression). Combined
        # with +discardcorrupt, frames that pile up while the
        # reader is briefly behind get dropped instead of waiting
        # in queue — which is exactly what we want for low-latency
        # display. Tighter than 1M (better latency on transient
        # consumer slowdowns) but loose enough that ffmpeg doesn't
        # backpressure the camera into stalling.
        cmd = [
            self._ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            # Demuxer-side: don't buffer or hold back corrupt
            # packets. `+flush_packets` forces the muxer to flush
            # after each frame so packets don't sit in muxer queue.
            "-fflags", "nobuffer+discardcorrupt+flush_packets",
            "-flags", "low_delay",
            # Capture-side input buffer (bounded — see comments
            # above the cmd assignment).
            "-rtbufsize", "512K",
            # `-thread_queue_size 1` shrinks the input-thread queue
            # so frames don't pile up between the dshow read and
            # the encoder; combined with `-vsync 0` (don't normalize
            # framerate, just pass frames through as captured) we
            # eliminate ffmpeg's internal frame queue almost
            # entirely. Without these, a brief consumer slowdown
            # (Spotify launch, focus change, etc.) lets ffmpeg
            # accumulate seconds of frames internally and the user
            # sees a persistent 2-3 s lag for several seconds
            # after the slowdown ends.
            "-thread_queue_size", "1",
            "-f", "dshow",
            "-vcodec", "mjpeg",
            "-framerate", str(self._fps),
            "-video_size", f"{self._width}x{self._height}",
            "-i", f"video={self._device_name}",
            "-vsync", "0",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # Echo the resolved ffmpeg path + full command at startup so
        # we can rule out path-resolution issues (silently using a
        # different ffmpeg.exe than expected) and command-line typos
        # when investigating "startup failed" reports.
        try:
            print(
                f"[ffmpeg_capture] launching ffmpeg: path={self._ffmpeg_path!r}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                bufsize=0,
            )
        except Exception as exc:
            try:
                print(
                    f"[ffmpeg_capture] Popen failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
            self._stderr_log.append(f"Popen failed: {exc!s}")
            self._proc = None
            return False
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"ffmpeg-cap-{self._device_name[:24]}",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            name=f"ffmpeg-stderr-{self._device_name[:18]}",
            daemon=True,
        )
        self._stderr_thread.start()
        # Give ffmpeg a few seconds to negotiate format and emit the
        # first frame. If we don't see anything by the deadline, give
        # up so the caller can fall back to OpenCV without leaving the
        # user staring at a blank live view. We surface ffmpeg's own
        # stderr on failure so the user can see whether the device
        # was busy, the format was rejected, etc.
        # Wait for the reader thread to either decode the first
        # frame OR signal a fatal read error / EOF on the pipe (the
        # reader sets `_first_frame_event` in both cases so this
        # call wakes up promptly instead of always sitting through
        # the full 8-second timeout). If the wakeup came from the
        # error path, fall through to the diagnostic block below —
        # NOT return True — so the caller's OpenCV fallback fires
        # AND we surface why ffmpeg's pipe died (which is what was
        # being silently masked before).
        woke = self._first_frame_event.wait(timeout=self._startup_timeout)
        if woke and not self._read_error:
            return True
        if woke and self._read_error:
            try:
                print(
                    "[ffmpeg_capture] reader signalled error/EOF before any "
                    "frame decoded — see stderr tail below for the actual cause",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
        # Build the diagnostic FIRST so even if anything below it
        # raises (teardown errors, etc.), we still surface the
        # information about why startup failed.
        proc_alive: bool | None = None
        proc_returncode: int | None = None
        try:
            proc = self._proc
            if proc is not None:
                proc_returncode = proc.poll()
                proc_alive = proc_returncode is None
        except Exception:
            pass
        try:
            tail_pre = "".join(self._stderr_log[-12:]).strip()
        except Exception:
            tail_pre = ""
        try:
            print(
                f"[ffmpeg_capture] startup failed after {self._startup_timeout:.1f}s: "
                f"proc_alive={proc_alive} returncode={proc_returncode} "
                f"stderr_lines={len(self._stderr_log)}",
                file=sys.stderr,
                flush=True,
            )
            if tail_pre:
                print(f"[ffmpeg_capture] stderr tail: {tail_pre}", file=sys.stderr, flush=True)
            else:
                print(
                    "[ffmpeg_capture] stderr was EMPTY — ffmpeg subprocess "
                    "produced no diagnostic output. Common cause: camera "
                    "driver held by another process and DSHOW open hung.",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            try:
                print(f"[ffmpeg_capture] diagnostic-print failed: {exc!r}", file=sys.stderr, flush=True)
            except Exception:
                pass
        self._teardown_proc()
        return False

    def _stderr_loop(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            for line in iter(self._proc.stderr.readline, b""):
                if not line:
                    break
                try:
                    text = line.decode("utf-8", errors="replace")
                except Exception:
                    continue
                self._stderr_log.append(text)
                if len(self._stderr_log) > 200:
                    del self._stderr_log[:100]
                # NB: we used to early-bailout here when stderr
                # contained "could not set video options" / "error
                # opening input" / etc., to fail in <500 ms instead
                # of waiting for the full timeout. That worked but
                # exposed a Windows DSHOW race: when ffmpeg fails
                # too fast, the OpenCV cap we released a moment
                # earlier hasn't finished tearing down on the driver
                # side, so the recovery reopen returns a half-dead
                # capture and the engine emits running_state=False.
                # Reverted to the natural timeout so the driver gets
                # ~2.5 s to release before we try to reopen — matches
                # what slower webcams (Razer Kiyo Pro, Logitech
                # BRIO at high resolution) actually need.
        except Exception:
            return

    def _reader_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        frame_bytes = self._width * self._height * 3
        # `_read_exact` for a 2.76 MB frame at 30 fps takes ~10 ms
        # at steady state (camera-paced). If a read returns in
        # significantly less time than the camera frame interval,
        # the pipe was already full when we asked — i.e. ffmpeg's
        # internal queue had a backlog. In that case we discard
        # the frame and read the next one, walking the queue down
        # to a single buffered frame as fast as the pipe lets us.
        # That's what kills the "persistent 2-3 s lag after
        # Spotify launch" symptom: instead of processing every
        # backed-up frame in order (which the user perceives as
        # the live view crawling forward through old content for
        # seconds), we jump straight to the most recent frame.
        camera_interval_s = 1.0 / max(1.0, float(self._fps))
        # Threshold: reads completing in <40% of the frame
        # interval are flagged as "pipe was already full."
        backlog_threshold = camera_interval_s * 0.4
        # Bound the drain so we always make progress even on a
        # weird steady-state where reads consistently come back
        # too fast (shouldn't happen but defensive).
        drain_limit_per_iter = 16
        last_read_done = 0.0
        while not self._stop_event.is_set():
            drained = 0
            chunk: bytes | None = None
            while True:
                try:
                    raw = self._read_exact(self._proc.stdout, frame_bytes)
                except Exception:
                    self._read_error = True
                    self._first_frame_event.set()
                    return
                if raw is None:
                    self._read_error = True
                    self._first_frame_event.set()
                    return
                now = time.monotonic()
                # If this is the very first read, just take it.
                # Otherwise check whether the pipe was prefilled.
                if last_read_done > 0.0 and (now - last_read_done) < backlog_threshold and drained < drain_limit_per_iter:
                    # Pipe still has more data ready — discard
                    # this frame and grab the next one.
                    last_read_done = now
                    drained += 1
                    continue
                last_read_done = now
                chunk = raw
                break
            try:
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    (self._height, self._width, 3)
                ).copy()
            except Exception:
                continue
            # Discard the first few decoded frames (placeholder / pre-
            # exposure-lock output some DSHOW filters emit). Still
            # set _first_frame_event so the startup waiter in _start
            # knows the pipe is alive — we just don't publish the
            # frame to consumers yet.
            if self._warmup_remaining > 0:
                self._warmup_remaining -= 1
                self._first_frame_event.set()
                continue
            decoded_at = time.monotonic()
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_ts = decoded_at
            self._first_frame_event.set()
            self._fresh_frame_event.set()

    @staticmethod
    def _read_exact(stream, size: int) -> bytes | None:
        """Read exactly `size` bytes from a binary stream. Returns
        None on EOF (process exit). Required because pipe reads
        otherwise return short blocks at high frame rates."""
        buf = bytearray()
        while len(buf) < size:
            chunk = stream.read(size - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def isOpened(self) -> bool:  # cv2.VideoCapture API parity
        if not self._opened:
            return False
        if self._read_error:
            return False
        if self._proc is None or self._proc.poll() is not None:
            return False
        return True

    def read(self):  # cv2.VideoCapture API parity
        if not self.isOpened():
            return False, None
        with self._frame_lock:
            frame = self._latest_frame
            ts = self._latest_frame_ts
            self._latest_frame = None
            self._fresh_frame_event.clear()
        if frame is not None:
            self._last_consumed_ts = ts
            return True, frame
        # Wait briefly for the reader thread to produce a fresh
        # frame. 5 ms timeout — enough that the reader gets
        # guaranteed GIL scheduling time (otherwise the main
        # thread's singleShot(0) tight-loop polling pattern starves
        # the reader, frames pile up in ffmpeg's pipe, and the
        # user-visible pipeline lag grows unbounded into the
        # multi-second range). 5 ms is short enough that the main
        # thread's other Qt events still fire smoothly.
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

    def release(self) -> None:  # cv2.VideoCapture API parity
        self._stop_event.set()
        self._teardown_proc()
        thread = self._reader_thread
        self._reader_thread = None
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass
        with self._frame_lock:
            self._latest_frame = None
        self._opened = False

    def _teardown_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        except Exception:
            pass
        for stream_name in ("stdout", "stderr", "stdin"):
            stream = getattr(proc, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    # cv2.VideoCapture-style get/set so the rest of the engine can
    # query frame size & fps without knowing whether we're a real
    # cv2 cap or an ffmpeg-backed one.

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        if prop_id == cv2.CAP_PROP_FOURCC:
            try:
                return float(cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                return 0.0
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE") and prop_id == cv2.CAP_PROP_BUFFERSIZE:
            return 1.0
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:  # noqa: ARG002
        # Settings cannot be changed after the ffmpeg process is
        # started; the engine's other code paths set BUFFERSIZE etc.
        # which we already enforce ourselves, so silently no-op
        # rather than raising.
        return False

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

# Author: Konstantin Markov

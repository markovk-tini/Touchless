"""macOS speaker / system-audio tap via ScreenCaptureKit.

This is the Apple-supported replacement for a virtual audio driver
(BlackHole, Loopback, Soundflower). Those drivers install into the HAL,
hijack the default output, and need admin + codesign. Touchless does
not ship one.

ScreenCaptureKit (macOS 13+) can capture the same mix the speakers are
playing, gated on the Screen Recording permission the app already asks
for. The microphone stays on the existing shared sounddevice ring —
this module never opens a CoreAudio input device.

Off macOS every public function is a no-op.

Author: Konstantin Markov (macOS port)
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np

_FS = 48000


def mac_system_audio_available() -> bool:
    """True when this process can *try* ScreenCaptureKit audio (macOS 13+)."""
    if sys.platform != "darwin":
        return False
    try:
        mac = __import__("platform").mac_ver()[0]
        major = int(str(mac).split(".")[0] or "0")
        return major >= 13
    except Exception:
        return False


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[mac-sys-audio] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _window_pcm_ring(
    chunks_ref, lock, fs: int, left: float, right: float
) -> Optional[np.ndarray]:
    """Same contiguous-concat window as the mic ring. See main_window
    `_extract_mac_clip_audio` — do not stamp each chunk independently."""
    if chunks_ref is None or right <= left:
        return None
    total = int(round((right - left) * fs))
    if total <= 0:
        return None
    try:
        if lock is not None:
            with lock:
                chunks = list(chunks_ref)
        else:
            chunks = list(chunks_ref)
    except Exception:
        return None
    if not chunks:
        return None
    first_t_end, first_arr = chunks[0]
    first_start = first_t_end - len(first_arr) / fs
    pieces: list = []
    prev_t_end = None
    for (t_end, arr) in chunks:
        n = len(arr)
        if n == 0:
            continue
        if prev_t_end is not None:
            gap = (t_end - n / fs) - prev_t_end
            if gap > 0.020:
                pieces.append(np.zeros(int(round(gap * fs)), dtype=np.float32))
        pieces.append(arr)
        prev_t_end = t_end
    if not pieces:
        return None
    try:
        full = np.concatenate(pieces).astype(np.float32)
    except Exception:
        return None
    if full.size == 0:
        return None
    out = np.zeros(total, dtype=np.float32)
    start_idx = int(round((left - first_start) * fs))
    src0 = max(0, start_idx)
    src1 = min(full.size, start_idx + total)
    if src1 <= src0:
        return None
    dst0 = src0 - start_idx
    dst1 = dst0 + (src1 - src0)
    if dst0 < 0 or dst1 > total or dst1 <= dst0:
        return None
    out[dst0:dst1] = full[src0:src1]
    return out


def mix_mac_pcm(
    mic: Optional[np.ndarray], sys_a: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Mix mic + system rings. Either side may be None / too short."""
    def _ok(buf) -> bool:
        return buf is not None and getattr(buf, "size", 0) >= int(0.05 * _FS)

    mic_ok = _ok(mic)
    sys_ok = _ok(sys_a)
    if mic_ok and not sys_ok:
        return mic
    if sys_ok and not mic_ok:
        return sys_a
    if not mic_ok and not sys_ok:
        return None
    n = max(len(mic), len(sys_a))
    out = np.zeros(n, dtype=np.float32)
    out[: len(mic)] += mic
    out[: len(sys_a)] += sys_a
    np.clip(out, -1.0, 1.0, out=out)
    return out


def _sbuf_to_mono_f32(sbuf) -> Optional[np.ndarray]:
    """Best-effort PCM extract from an SCK audio CMSampleBuffer."""
    try:
        from CoreMedia import (  # type: ignore
            CMSampleBufferGetDataBuffer,
            CMSampleBufferGetNumSamples,
            CMSampleBufferGetFormatDescription,
            CMAudioFormatDescriptionGetStreamBasicDescription,
            CMBlockBufferGetDataLength,
            CMBlockBufferCopyDataBytes,
        )
    except Exception:
        return None
    try:
        n = int(CMSampleBufferGetNumSamples(sbuf) or 0)
    except Exception:
        n = 0
    channels = 2
    try:
        fmt = CMSampleBufferGetFormatDescription(sbuf)
        asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fmt)
        if asbd is not None:
            ch = getattr(asbd, "mChannelsPerFrame", None)
            if ch is None and isinstance(asbd, (tuple, list)) and len(asbd) > 3:
                ch = asbd[3]
            if ch is not None:
                channels = max(1, int(ch))
    except Exception:
        channels = 2
    raw = None
    try:
        block = CMSampleBufferGetDataBuffer(sbuf)
        if block is not None:
            length = int(CMBlockBufferGetDataLength(block) or 0)
            if length > 0:
                copied = CMBlockBufferCopyDataBytes(block, 0, length)
                if isinstance(copied, tuple):
                    status, data = copied[0], copied[1] if len(copied) > 1 else None
                    if int(status or 0) == 0 and data:
                        raw = bytes(data)
                elif copied not in (None, 0):
                    try:
                        raw = bytes(copied)
                    except Exception:
                        raw = None
                if raw is None:
                    buf = bytearray(length)
                    try:
                        status = CMBlockBufferCopyDataBytes(block, 0, length, buf)
                        if int(status or 0) == 0:
                            raw = bytes(buf)
                    except Exception:
                        raw = None
    except Exception:
        raw = None
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size == 0:
        return None
    if channels > 1 and arr.size % channels == 0:
        arr = arr.reshape(-1, channels).mean(axis=1)
    elif n > 0 and arr.size >= n:
        arr = arr[:n]
    return np.ascontiguousarray(arr, dtype=np.float32)


class MacSystemAudioTap:
    """Rolling wall-clock ring of speaker audio. One SCStream per process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: deque = deque()
        self._max_seconds = 310.0
        self._unbounded = False
        self._stream = None
        self._sink = None
        self._running = False
        self._fs = _FS
        self._start_error = ""
        self._thread: Optional[threading.Thread] = None
        self._starting = False

    @property
    def running(self) -> bool:
        return bool(self._running)

    @property
    def sample_rate(self) -> int:
        return int(self._fs)

    @property
    def last_error(self) -> str:
        return str(self._start_error or "")

    def set_unbounded(self, flag: bool) -> None:
        self._unbounded = bool(flag)

    def start(self, *, max_seconds: float = 310.0) -> bool:
        if sys.platform != "darwin":
            return False
        if not mac_system_audio_available():
            self._start_error = "needs macOS 13+"
            return False
        if self._running or self._starting:
            self._max_seconds = float(max_seconds)
            return True
        self._max_seconds = float(max_seconds)
        self._start_error = ""
        self._starting = True
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._start_on_thread,
            args=(ready,),
            name="mac-sck-audio",
            daemon=True,
        )
        self._thread.start()
        # Do not block the Qt GUI thread waiting for ScreenCaptureKit.
        return True

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        self._sink = None
        self._running = False
        self._starting = False
        if stream is not None:
            try:
                stream.stopCaptureWithCompletionHandler_(lambda _err: None)
            except Exception:
                pass
        with self._lock:
            self._chunks.clear()

    def extract(self, left: float, right: float) -> Optional[np.ndarray]:
        return _window_pcm_ring(self._chunks, self._lock, self._fs, left, right)

    def _push(self, mono: np.ndarray) -> None:
        if mono is None or mono.size == 0:
            return
        t_end = time.time()
        with self._lock:
            self._chunks.append((t_end, mono.copy()))
            if not self._unbounded:
                cutoff = t_end - self._max_seconds
                while self._chunks and self._chunks[0][0] < cutoff:
                    self._chunks.popleft()

    def _start_on_thread(self, ready: threading.Event) -> None:
        try:
            self._start_sck(ready)
        except Exception as exc:
            self._start_error = f"{type(exc).__name__}: {exc}"
            _log(f"start failed: {self._start_error}")
            self._running = False
            self._starting = False
            try:
                ready.set()
            except Exception:
                pass

    def _start_sck(self, ready: threading.Event) -> None:
        import objc  # type: ignore
        from Foundation import NSObject, NSDate, NSRunLoop, NSDefaultRunLoopMode  # type: ignore
        from ScreenCaptureKit import (  # type: ignore
            SCShareableContent,
            SCContentFilter,
            SCStream,
            SCStreamConfiguration,
        )
        try:
            from ScreenCaptureKit import SCStreamOutputTypeAudio  # type: ignore
            audio_type: Any = SCStreamOutputTypeAudio
        except Exception:
            audio_type = 1
        try:
            from CoreMedia import CMTimeMake  # type: ignore
        except Exception:
            CMTimeMake = None

        sink_cls = getattr(MacSystemAudioTap, "_sink_cls", None)
        if sink_cls is None:
            audio_type_holder = [audio_type]

            class _Sink(NSObject):
                def stream_didOutputSampleBuffer_ofType_(self, _stream, sbuf, otype):
                    try:
                        at = int(audio_type_holder[0])
                        if int(otype) not in (at, 1):
                            return
                        owner = getattr(self, "_tap", None)
                        if owner is None:
                            return
                        mono = _sbuf_to_mono_f32(sbuf)
                        if mono is not None:
                            owner._push(mono)
                    except Exception:
                        pass

            MacSystemAudioTap._sink_cls = _Sink
            sink_cls = _Sink

        holder: dict = {}
        got = threading.Event()

        def _on_content(content, error) -> None:
            holder["content"] = content
            holder["error"] = error
            got.set()

        SCShareableContent.getShareableContentWithCompletionHandler_(_on_content)
        deadline = time.time() + 6.0
        while not got.is_set() and time.time() < deadline:
            try:
                NSRunLoop.currentRunLoop().runMode_beforeDate_(
                    NSDefaultRunLoopMode,
                    NSDate.dateWithTimeIntervalSinceNow_(0.05),
                )
            except Exception:
                if got.wait(0.05):
                    break
        content = holder.get("content")
        error = holder.get("error")
        if content is None:
            self._start_error = f"no shareable content ({error})"
            _log(self._start_error)
            self._starting = False
            ready.set()
            return
        displays = list(content.displays() or [])
        if not displays:
            self._start_error = "no displays"
            _log(self._start_error)
            self._starting = False
            ready.set()
            return
        display = displays[0]
        try:
            filt = SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                display, []
            )
        except Exception:
            filt = SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                display, None
            )
        cfg = SCStreamConfiguration.alloc().init()
        try:
            cfg.setCapturesAudio_(True)
        except Exception:
            cfg.capturesAudio = True
        try:
            cfg.setExcludesCurrentProcessAudio_(True)
        except Exception:
            pass
        try:
            cfg.setSampleRate_(self._fs)
            cfg.setChannelCount_(2)
        except Exception:
            pass
        try:
            cfg.setWidth_(2)
            cfg.setHeight_(2)
        except Exception:
            pass
        if CMTimeMake is not None:
            try:
                cfg.setMinimumFrameInterval_(CMTimeMake(1, 1))
            except Exception:
                pass
        sink = sink_cls.alloc().init()
        sink._tap = self
        stream = SCStream.alloc().initWithFilter_configuration_delegate_(
            filt, cfg, None
        )
        err = None
        try:
            stream.addStreamOutput_type_sampleHandlerQueue_error_(
                sink, audio_type, None, None
            )
        except TypeError:
            try:
                ok, err = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                    sink, audio_type, None, None
                )
                if ok is False:
                    self._start_error = f"addStreamOutput failed ({err})"
                    _log(self._start_error)
                    self._starting = False
                    ready.set()
                    return
            except Exception as exc:
                self._start_error = f"addStreamOutput: {exc}"
                _log(self._start_error)
                self._starting = False
                ready.set()
                return
        started = threading.Event()
        start_err: dict = {}

        def _on_start(error) -> None:
            start_err["e"] = error
            started.set()

        stream.startCaptureWithCompletionHandler_(_on_start)
        start_deadline = time.time() + 6.0
        while not started.is_set() and time.time() < start_deadline:
            try:
                NSRunLoop.currentRunLoop().runMode_beforeDate_(
                    NSDefaultRunLoopMode,
                    NSDate.dateWithTimeIntervalSinceNow_(0.05),
                )
            except Exception:
                started.wait(0.05)
        if start_err.get("e"):
            self._start_error = f"startCapture: {start_err['e']}"
            _log(self._start_error)
            self._starting = False
            ready.set()
            return
        self._sink = sink
        self._stream = stream
        self._running = True
        self._starting = False
        _log(f"ScreenCaptureKit system-audio tap started fs={self._fs}")
        ready.set()
        # Keep this thread's run loop alive so the stream's callbacks
        # have a place to land if they were bound here.
        while self._running and self._stream is stream:
            try:
                NSRunLoop.currentRunLoop().runMode_beforeDate_(
                    NSDefaultRunLoopMode,
                    NSDate.dateWithTimeIntervalSinceNow_(0.25),
                )
            except Exception:
                time.sleep(0.25)
        _ = objc  # keep import used
        _log("system-audio tap thread exit")


# Author: Konstantin Markov

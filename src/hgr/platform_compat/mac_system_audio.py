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
from pathlib import Path
from typing import Any, Optional

import numpy as np

_FS = 48000

# Positive stamp_lead skips newer ring samples and makes the
# soundtrack *earlier* (clip t=0 plays a later wall time).
# +2.5 s with clear concat was ~4 s early ≈ 1.5 s inherent
# first-start lead + 2.5 s of this term. Negative delays the
# track. Screen recordings pass 0.
MAC_CLIP_STAMP_LEAD_S = -1.5
# Room mic on top of SCK re-records speakers (comb / static).
# Duck only when both tracks mix; mic-only is unchanged.
MAC_MIX_MIC_GAIN = 0.12


def looks_like_mp4(path) -> bool:
    """True when `path` is an ISO-BMFF / mp4 file, not a text log.

    ffmpeg screen-record used to write `<output>.mp4.ffmpeg.log` into
    the user save folder. A size check alone treated that log as a
    saved recording. MP4 always starts with a 4-byte size then `ftyp`.
    """
    try:
        p = Path(path)
        if not p.is_file() or p.stat().st_size < 64:
            return False
        with p.open("rb") as fh:
            head = fh.read(12)
        return len(head) >= 8 and head[4:8] == b"ftyp"
    except Exception:
        return False


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


def assemble_pcm_ring(
    chunks,
    fs: int,
    left: float,
    right: float,
    *,
    stamp_lead_s: float = 0.0,
    pad_to_window: bool = True,
) -> Optional[np.ndarray]:
    """Contiguous PCM for wall window [left, right], first-start sliced.

    Callbacks deliver sequential unique PCM. Stamp overlap is jitter,
    not duplicate samples — crossfading it mixed different audio
    (static) and shortened the ring vs wall-clock (~5 s early).
    Concatenate in arrival order. Ignore sub-quarter-second gaps.
    Then slice from the first block's implied start plus optional
    `stamp_lead_s`. Positive lead skips into the ring (soundtrack
    plays early); negative includes older samples (delays it).
    Clip mux only; screen recordings pass 0.

    `pad_to_window=False` returns only the concat samples that fall in
    the window (no leading/trailing zeros). Recording mux stretches
    that true sample count onto probed video duration so packed
    dropouts don't play future audio (~1 s early by the end).
    """
    if chunks is None or right <= left or int(fs) <= 0:
        return None
    total = int(round((float(right) - float(left)) * float(fs)))
    if total <= 0:
        return None
    pieces: list = []
    first_start = None
    prev_t_end = None
    fs_f = float(fs)
    dropout_s = 0.25
    for (t_end, arr) in chunks:
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        n = int(arr.size)
        if n == 0:
            continue
        t_end_f = float(t_end)
        t_start = t_end_f - n / fs_f
        if first_start is None:
            first_start = t_start
        elif prev_t_end is not None:
            gap = t_start - float(prev_t_end)
            # Negative / small positive gaps are stamp jitter. Do not
            # trim, crossfade, or insert silence — those desync and
            # crackle. Only pad a real dropout.
            if gap > dropout_s:
                n_pad = int(round(gap * fs_f))
                n_f = min(int(round(0.005 * fs_f)), max(1, n_pad // 4))
                # Hard splice of audio|zeros|audio is a click.
                # Fade the adjacent 5 ms; keep the gap as silence.
                if pieces and n_f > 1:
                    prev = np.array(pieces[-1], dtype=np.float32, copy=True)
                    k = min(n_f, int(prev.size))
                    if k > 1:
                        prev[-k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
                    pieces[-1] = prev
                pieces.append(np.zeros(n_pad, dtype=np.float32))
                arr = np.array(arr, dtype=np.float32, copy=True)
                k = min(n_f, int(arr.size))
                if k > 1:
                    arr[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)
        pieces.append(arr)
        prev_t_end = t_end_f
    if not pieces or first_start is None:
        return None
    try:
        full = np.concatenate(pieces).astype(np.float32, copy=False)
    except Exception:
        return None
    if full.size == 0:
        return None
    start_idx = int(
        round((float(left) - float(first_start) + float(stamp_lead_s or 0.0)) * fs_f)
    )
    end_idx = start_idx + total
    out = np.zeros(total, dtype=np.float32)
    src0 = max(0, start_idx)
    src1 = min(full.size, end_idx)
    if src1 <= src0:
        return None
    if not pad_to_window:
        return np.array(full[src0:src1], dtype=np.float32, copy=True)
    dst0 = src0 - start_idx
    dst1 = dst0 + (src1 - src0)
    if dst0 < 0 or dst1 > total or dst1 <= dst0:
        return None
    out[dst0:dst1] = full[src0:src1]
    return out


def _window_pcm_ring(
    chunks_ref, lock, fs: int, left: float, right: float,
    *,
    stamp_lead_s: float = 0.0,
    pad_to_window: bool = True,
) -> Optional[np.ndarray]:
    """Lock + copy, then `assemble_pcm_ring`. Same helper as the mic ring."""
    if chunks_ref is None or right <= left:
        return None
    try:
        if lock is not None:
            with lock:
                chunks = list(chunks_ref)
        else:
            chunks = list(chunks_ref)
    except Exception:
        return None
    return assemble_pcm_ring(
        chunks, fs, left, right,
        stamp_lead_s=stamp_lead_s,
        pad_to_window=pad_to_window,
    )


def mac_mux_atempo_factor(audio_dur: float, video_dur: float) -> float:
    """atempo so PCM lasting `audio_dur` fills `video_dur` (pitch kept).

    Packed rings (ignored callback gaps / xruns) play future content:
    start is on time, then audio leads by ~1 s. Factor < 1 slows it
    back onto the picture. Huge mismatches are left alone.
    """
    try:
        a = float(audio_dur)
        v = float(video_dur)
    except (TypeError, ValueError):
        return 1.0
    if a < 0.25 or v < 0.25:
        return 1.0
    tempo = a / v
    if abs(tempo - 1.0) < 0.004:
        return 1.0
    if tempo < 0.88 or tempo > 1.12:
        return 1.0
    return min(2.0, max(0.5, tempo))


def mix_mac_pcm(
    mic: Optional[np.ndarray],
    sys_a: Optional[np.ndarray],
    *,
    mic_gain: float = 1.0,
) -> Optional[np.ndarray]:
    """Mix mic + system rings. Either side may be None / too short.

    End-aligned: both tracks share the same right edge (the clip/record
    stop). Start-aligning a longer system buffer put old speaker audio
    at the front; ffmpeg -shortest then kept that early half.

    `mic_gain` applies only when BOTH tracks are present. The room mic
    re-records speakers on top of the SCK tap; equal mix comb-filters
    into static. Ducking the mic keeps voice faintly and the tap clean.
    """
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
    fade_n = int(0.008 * _FS)

    def _add(buf, gain: float) -> None:
        b = np.asarray(buf, dtype=np.float32).reshape(-1)
        start = n - int(b.size)
        need_fade = start > 0 and fade_n > 1
        g = float(gain)
        if g != 1.0 or need_fade:
            b = np.multiply(b, np.float32(g), dtype=np.float32)
            if need_fade:
                fn = min(fade_n, int(b.size))
                b[:fn] *= np.linspace(0.0, 1.0, fn, dtype=np.float32)
        out[start : start + int(b.size)] += b

    _add(mic, float(mic_gain))
    _add(sys_a, 1.0)
    np.clip(out, -1.0, 1.0, out=out)
    return out


def boost_quiet_mac_pcm(
    buf: Optional[np.ndarray],
    *,
    target_peak: float = 0.55,
    max_gain: float = 4.0,
    already_loud: float = 0.28,
) -> Optional[np.ndarray]:
    """Raise a quiet capture (typical SCK tap) without touching a loud mix."""
    if buf is None or getattr(buf, "size", 0) == 0:
        return buf
    peak = float(np.max(np.abs(buf)))
    if peak < 1e-5 or peak >= float(already_loud):
        return buf
    gain = min(float(max_gain), float(target_peak) / peak)
    if gain <= 1.01:
        return buf
    out = np.asarray(buf, dtype=np.float32) * np.float32(gain)
    return _soft_clip_mac_pcm(out)


def _soft_clip_mac_pcm(buf: np.ndarray, threshold: float = 0.90) -> np.ndarray:
    """Compress only the last 10% of full scale — no hard-clip tick."""
    out = np.asarray(buf, dtype=np.float32)
    if out.size == 0:
        return out
    thr = np.float32(threshold)
    ax = np.abs(out)
    over = ax > thr
    if not np.any(over):
        np.clip(out, -1.0, 1.0, out=out)
        return out
    if out.base is not None:
        out = np.array(out, dtype=np.float32, copy=True)
        ax = np.abs(out)
        over = ax > thr
    sign = np.sign(out)
    headroom = np.float32(1.0) - thr
    excess = ax - thr
    out[over] = sign[over] * (thr + headroom * np.tanh(excess[over] / headroom))
    np.clip(out, -1.0, 1.0, out=out)
    return out


def polish_mac_pcm(
    buf: Optional[np.ndarray],
    *,
    fs: int = _FS,
    fade_s: float = 0.010,
) -> Optional[np.ndarray]:
    """Fade edges, kill 1-sample clicks, and soft-clip rails."""
    if buf is None or getattr(buf, "size", 0) == 0:
        return buf
    out = np.array(buf, dtype=np.float32, copy=True)
    if out.size >= 5:
        d = np.diff(out)
        # Opposite-sign jumps on both sides of a sample = a tick, not
        # a musical transient (those last several samples).
        mag = np.minimum(np.abs(d[:-1]), np.abs(d[1:]))
        clicks = (d[:-1] * d[1:] < 0.0) & (mag > np.float32(0.35))
        idx = np.where(clicks)[0] + 1
        if idx.size:
            out[idx] = 0.5 * (out[idx - 1] + out[idx + 1])
    n_fade = int(round(float(fs) * float(fade_s)))
    if n_fade > 1 and out.size > 2 * n_fade:
        ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
        out[:n_fade] *= ramp
        out[-n_fade:] *= ramp[::-1]
    return _soft_clip_mac_pcm(out)


def mac_clip_video_timescale(video_dur: float, wall_span: float) -> float:
    """Timestamp scale so a short OpenCV clip plays in wall-clock time.

    Mac MJPG/mp4v writers often tag ~20 fps while Quartz capture is
    slower, so ffprobe duration is ~10 s short of the audio window.
    End-trimming audio to that probe made the soundtrack lead picture.
    """
    try:
        v = float(video_dur)
        w = float(wall_span)
    except (TypeError, ValueError):
        return 1.0
    if v <= 0.05 or w <= 0.05:
        return 1.0
    scale = w / v
    if 0.97 <= scale <= 1.03:
        return 1.0
    return min(max(scale, 0.5), 2.5)


def _sbuf_unix_end(sbuf, n_samples: int, fs: int) -> float:
    """Presentation time of the last sample, else wall-clock arrival."""
    now = time.time()
    dur = max(0, int(n_samples)) / float(fs) if fs else 0.0
    try:
        from CoreMedia import (  # type: ignore
            CMSampleBufferGetPresentationTimeStamp,
            CMTimeGetSeconds,
        )
        pts = CMSampleBufferGetPresentationTimeStamp(sbuf)
        sec = float(CMTimeGetSeconds(pts) or 0.0)
    except Exception:
        return now
    if sec <= 0.0:
        return now
    # Host/uptime clocks are small; unix seconds are ~1e9.
    if sec > 1.0e9:
        return sec + dur
    if sec < 1.0e8:
        return (now - time.monotonic()) + sec + dur
    return now


def _asbd_field(asbd, name: str, index: int, default):
    try:
        val = getattr(asbd, name, None)
        if val is not None:
            return val
    except Exception:
        pass
    try:
        if isinstance(asbd, (tuple, list)) and len(asbd) > index:
            return asbd[index]
    except Exception:
        pass
    try:
        if isinstance(asbd, dict):
            return asbd.get(name, default)
    except Exception:
        pass
    return default


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
    bits = 32
    flags = 1  # kAudioFormatFlagIsFloat
    sample_rate = float(_FS)
    try:
        fmt = CMSampleBufferGetFormatDescription(sbuf)
        asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fmt)
        if asbd is not None:
            # AudioStreamBasicDescription layout:
            # 0 mSampleRate, 1 mFormatID, 2 mFormatFlags,
            # 3 mBytesPerPacket, 4 mFramesPerPacket, 5 mBytesPerFrame,
            # 6 mChannelsPerFrame, 7 mBitsPerChannel
            sample_rate = float(_asbd_field(asbd, "mSampleRate", 0, _FS) or _FS)
            flags = int(_asbd_field(asbd, "mFormatFlags", 2, 1) or 0)
            channels = max(1, int(_asbd_field(asbd, "mChannelsPerFrame", 6, 2) or 2))
            bits = int(_asbd_field(asbd, "mBitsPerChannel", 7, 32) or 32)
    except Exception:
        channels = 2
        bits = 32
        flags = 1
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
    is_float = bool(flags & 1) or bits == 32
    try:
        if is_float and bits >= 32:
            arr = np.frombuffer(raw, dtype=np.float32)
        elif bits == 16:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif bits == 32 and not is_float:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            arr = np.frombuffer(raw, dtype=np.float32)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if n > 0 and channels > 1 and arr.size == n * channels:
        # Mean fold of a one-sided / low SCK buffer is ~half level.
        # Mid * sqrt(n) keeps mono-compatible level; clip later.
        folded = arr.reshape(n, channels).mean(axis=1)
        arr = folded * (min(int(channels), 2) ** 0.5)
    elif n > 0 and arr.size >= n:
        arr = arr[:n]
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if sample_rate > 1.0 and abs(sample_rate - float(_FS)) > 1.0:
        try:
            ratio = float(_FS) / float(sample_rate)
            new_n = max(1, int(round(arr.size * ratio)))
            x_old = np.linspace(0.0, 1.0, arr.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, new_n, endpoint=False)
            arr = np.interp(x_new, x_old, arr).astype(np.float32)
        except Exception:
            pass
    return _soft_clip_mac_pcm(arr)


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

    def extract(
        self,
        left: float,
        right: float,
        *,
        stamp_lead_s: float = 0.0,
        pad_to_window: bool = True,
    ) -> Optional[np.ndarray]:
        return _window_pcm_ring(
            self._chunks, self._lock, self._fs, left, right,
            stamp_lead_s=stamp_lead_s,
            pad_to_window=pad_to_window,
        )

    def _push(self, mono: np.ndarray, t_end: Optional[float] = None) -> None:
        if mono is None or mono.size == 0:
            return
        stamp = time.time() if t_end is None else float(t_end)
        with self._lock:
            self._chunks.append((stamp, mono.copy()))
            if not self._unbounded:
                cutoff = stamp - self._max_seconds
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
                            owner._push(
                                mono,
                                t_end=_sbuf_unix_end(
                                    sbuf, int(mono.size), int(owner._fs)
                                ),
                            )
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
                # 1 fps video interval batched audio into huge chunks and
                # made timestamps drift. 30 fps keeps the tap realtime.
                cfg.setMinimumFrameInterval_(CMTimeMake(1, 30))
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

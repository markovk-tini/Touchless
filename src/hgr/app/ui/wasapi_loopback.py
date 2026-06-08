"""WASAPI loopback + mic bridges for the clip-cache ffmpeg subprocess.

The bundled ffmpeg (gyan.dev 7.0.2-full_build) is built without the
`wasapi` indev — no released ffmpeg has one (trac #9408). We capture
the Windows render-endpoint output in Python via PyAudioWPatch (a
Windows-only PortAudio fork with native WASAPI loopback, MIT, ships
prebuilt wheels) and pipe the raw PCM bytes into ffmpeg through one
of two routes:

  - `pipe:0` for the system-audio (loopback) stream — single subprocess
    stdin, owned by `_run_clip_cache_audio` in main_window.
  - A localhost TCP connection (`tcp://127.0.0.1:PORT`) for the mic
    stream when system audio is ALSO enabled (ffmpeg can only have
    one `pipe:0`). When system audio is OFF, the mic uses `pipe:0`
    directly — same low-overhead route as loopback.

The DirectShow mic input was removed because device naming between
WASAPI (sounddevice / PortAudio) and DirectShow is incompatible —
`USB Audio Device` (WASAPI) vs `Microphone (USB Audio Device)`
(DirectShow). DirectShow silently produces a stalled audio stream
when the name doesn't match exactly, and the user sees clips with
no mic audio AND no error in the log. Going through PortAudio for
BOTH endpoints (system + mic) means we resolve by device index, not
by name-format-translated string, so resolution is robust.

Lifecycle:
- `_start_clip_cache_audio` probes the endpoints, spawns ffmpeg with
  `stdin=PIPE` (and optionally a TCP-accept thread for the mic),
  then constructs + starts one or two `WasapiLoopbackWriter`s. Each
  writer owns its PortAudio stream and a daemon thread that copies
  bytes into its destination (stdin OR an accepted socket) until
  either stopped or the pipe breaks.
- `_stop_clip_cache_audio` calls `writer.stop()` on each writer
  BEFORE `_stop_ffmpeg_process(...)` so the threads exit cleanly
  instead of spinning on a broken pipe.

If `pyaudiowpatch` isn't installed, the probe returns None and that
side of the bridge is never spawned — ffmpeg runs without the input
and the clip cache falls through to the other-source path (or
video-only when both fail).
"""

from __future__ import annotations

import queue
import socket
import sys
import threading
import time
from typing import Callable, Optional


class WasapiLoopbackWriter:
    """Owns a single PyAudioWPatch loopback stream and pipes its
    output into a subprocess stdin handle. Lock-step lifecycle with
    the clip-cache ffmpeg process: parent calls `start()` once after
    spawning ffmpeg, then `stop()` once before reaping it.

    The (device_index, rate, channels) tuple is provided by the
    caller (it was learned by `_probe_wasapi_loopback_format`
    earlier, so ffmpeg's `-ar`/`-ac` args already match). The writer
    does NOT re-query the default endpoint; that would race against
    a device-switch between probe and open.
    """

    def __init__(
        self,
        ffmpeg_stdin,
        *,
        device_index: int,
        rate: int,
        channels: int,
        on_error: Optional[Callable[[str], None]] = None,
        is_loopback: bool = True,
        label: str = "WasapiLoopback",
        close_stdin_on_exit: bool = True,
        align_to_wall_time: Optional[float] = None,
        use_callback_mode: bool = False,
    ) -> None:
        self._stdin = ffmpeg_stdin
        self._device_index = int(device_index)
        self.rate = int(rate)
        self.channels = max(1, int(channels))
        self._on_error = on_error or (lambda _msg: None)
        # When two writers share a single ffmpeg subprocess (one per
        # input pipe), only ONE of them should close the stdin handle
        # at teardown. If both close, the second close() may crash on
        # an already-closed handle, and worse — closing stdin signals
        # end-of-stream to ffmpeg, which may stop muxing both inputs
        # before the second writer has flushed. Set this False on all
        # but one writer when sharing a pipe.
        self._close_stdin_on_exit = bool(close_stdin_on_exit)
        # is_loopback distinguishes a render-endpoint LOOPBACK capture
        # (system audio — what the user hears) from an INPUT-endpoint
        # capture (microphone). PortAudio/PyAudioWPatch handles both
        # through the same `input=True, input_device_index=N` open call;
        # the only difference is which device-index the caller supplied
        # (a loopback wrapper index vs. a regular input device index).
        # We keep the flag so the lifecycle thread can label its log
        # messages and so future divergence (e.g. exclusive-mode
        # toggles) can branch cleanly without breaking call sites.
        self._is_loopback = bool(is_loopback)
        self._label = str(label)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pa = None
        self._stream = None
        # Wall-clock timestamp of the FIRST audio chunk actually returned
        # from stream.read(). The clip-export aligner needs this to anchor
        # audio time correctly: the ffmpeg-process-spawned-at timestamp
        # is ~200-500ms (sometimes >1s) before WASAPI loopback actually
        # produces samples, and using the wrong anchor makes audio drift
        # later than video in the exported clip. None until the bridge
        # has actually read its first sample.
        self.first_sample_at: Optional[float] = None
        # Optional wall time this bridge should ALIGN ITS STREAM TO.
        # When set (typically to the SYSTEM-loopback bridge's
        # first_sample_at when starting the MIC bridge), the mic
        # bridge pre-pads silence equal to (start_t - align_to_wall_time)
        # before sending real samples. This compensates for the TCP
        # accept wait (~4 s on Windows) — without the pad, mic byte 0
        # ends up at amix output PTS=0 alongside sys byte 0, so when
        # the user clips, mic content at any output PTS is captured at
        # a wall time ~4 s LATER than sys content at the same PTS.
        # The user perceives this as "mic plays N seconds ahead of
        # the video".
        self._align_to_wall_time: Optional[float] = (
            float(align_to_wall_time)
            if align_to_wall_time is not None
            else None
        )
        # When True, capture audio via PortAudio CALLBACK mode
        # (pa.open(stream_callback=...)) instead of polling
        # get_read_available() + non-blocking read. Memory note
        # `project_wasapi_callback_vs_read.md` documents that
        # polling-mode on WASAPI shared mode produces corrupted
        # audio on Kiyo Pro and other UVC mics — the "static /
        # garbled" symptom the user keeps reporting. Callback
        # mode goes through a different PortAudio code path that
        # is clean on the same hardware. Default False keeps the
        # polling path for sys loopback (the loopback wrapper has
        # historically been finicky in callback mode); main_window
        # passes True for the mic bridge.
        self._use_callback_mode = bool(use_callback_mode)
        # Bounded queue from PortAudio callback to the writer
        # thread. Callback must return fast (PortAudio realtime
        # thread); writer thread does the IO to stdin/socket.
        # maxsize 2000 chunks × 21 ms = ~42 s of buffering before
        # drops — large because ffmpeg's pipe can briefly back up
        # while it's flushing a segment file, and dropped mic
        # callback chunks sound like white-noise glitches in the
        # final clip (input is unsigned-PCM; a dropped 21 ms chunk
        # leaves the stream out of frame-alignment until the next
        # full chunk arrives).
        self._callback_queue: Optional[queue.Queue] = (
            queue.Queue(maxsize=2000) if self._use_callback_mode else None
        )
        self._drain_thread: Optional[threading.Thread] = None
        # Externally-observable liveness counter for the liveness
        # watchdog. Updated each time real (non-silence) audio data
        # arrives from PortAudio. Main thread's liveness QTimer
        # compares this against time.time(); if it hasn't moved in
        # >N seconds the bridge is silently dead and the watchdog
        # triggers auto-repair via swap_device(). Stays 0.0 until
        # the first real chunk so a startup race doesn't trigger
        # false-positive repair.
        self.last_real_data_at: float = 0.0
        # Device hot-swap state. The watchdog calls swap_device() to
        # follow a Windows-default-playback / preferred-mic change
        # mid-session WITHOUT restarting ffmpeg (the segment ring
        # would be wiped). Two transports:
        # * callback mode: stop old stream, open new stream sharing
        #   the SAME self._stream_callback so chunks land in the
        #   SAME _callback_queue, swap self._stream/self._pa under
        #   _swap_lock, close old outside the lock.
        # * polling mode: open the new stream (start=True implicit
        #   via pa.open), store as _swap_pending tuple; the _run
        #   loop picks it up at the top of its next iteration,
        #   swaps the local `stream` ref, and closes the old
        #   stream outside the lock.
        # _resample_src_* are set when the new device has a
        # different rate/channel config from the originally-opened
        # one; _maybe_resample() then linear-interp-resamples each
        # chunk before it reaches stdin so ffmpeg keeps seeing
        # bytes at its original -ar/-ac.
        self._swap_lock: threading.Lock = threading.Lock()
        self._swap_pending: Optional[tuple] = None
        # Background-thread close-then-open state. When the polling
        # _run loop detects a swap request it spawns a worker that
        # closes OLD pa+stream and opens NEW. Meanwhile _run keeps
        # paced silence chunks flowing so ffmpeg's input pipe never
        # stalls — segment mtimes stay on schedule so the chain
        # math in the export aligner doesn't drift. When the worker
        # finishes it writes its result here for _run to pick up.
        self._swap_worker_thread: Optional[threading.Thread] = None
        self._swap_worker_result: Optional[tuple] = None
        self._swap_worker_busy: bool = False
        self._resample_src_rate: Optional[int] = None
        self._resample_src_channels: Optional[int] = None

    def swap_device(
        self,
        new_device_index: int,
        new_rate: int,
        new_channels: int,
    ) -> bool:
        """Replace the underlying PortAudio capture device WITHOUT
        touching the downstream stdin/socket pipe, the callback
        queue, the drain thread, or the ffmpeg subprocess on the
        other end. The segment ring keeps rotating with continuous
        filenames; first_sample_at stays anchored to the original
        startup wall time so mic-alignment math is preserved.

        Returns True on successful swap, False on any open/start
        failure (the OLD stream is left intact in that case so the
        user keeps hearing audio in the next clip).

        Format-mismatch case: if the new device's rate/channels
        differ from the original we opened with, install the
        _resample_src_* fields so _maybe_resample() linear-interps
        each chunk back to the original format. PortAudio's WASAPI
        shared-mode resampler usually handles this automatically
        when we ask for the original rate/channels — that's the
        first attempt; numpy fallback is the safety net."""
        try:
            new_device_index = int(new_device_index)
        except Exception:
            return False
        if new_device_index < 0:
            return False
        if new_device_index == int(self._device_index):
            return True  # no-op: already on this device
        try:
            import pyaudiowpatch as pa  # type: ignore
        except Exception:
            return False
        # Build a NEW PyAudio instance + stream. We use a fresh PA
        # instance for the new device because PortAudio's stream
        # handles are tied to the PA context they were opened in;
        # closing the old PA context after the swap is also cleaner.
        new_pa = None
        new_stream = None
        # Polling mode: DO NOT pre-open the new stream here. PortAudio's
        # WASAPI shared-mode loopback typically refuses to deliver to
        # two concurrent loopback streams in the same process — the
        # second open "succeeds" but silently produces zero data,
        # which the user reported as "switched output to headset, no
        # audio captured". Instead, stage only the swap REQUEST
        # (device params); _run does close-old-then-open-new inline.
        # The bridge's silence backfill (already implemented) keeps
        # ffmpeg fed during the brief gap (~50-200 ms typical).
        if not self._use_callback_mode:
            with self._swap_lock:
                self._swap_pending = (
                    "request",
                    int(new_device_index),
                    int(new_rate) or 48000,
                    max(1, int(new_channels) or 1),
                )
            try:
                import sys as _sys
                _sys.stderr.write(
                    f"[wasapi-bridge] {self._label}: polling-mode swap "
                    f"request staged old_idx={self._device_index} -> "
                    f"new_idx={new_device_index} (rate={new_rate}, "
                    f"ch={new_channels})\n"
                )
                _sys.stderr.flush()
            except Exception:
                pass
            return True
        # Callback mode: PA happily delivers to multiple concurrent
        # callback streams (each gets its own host buffer). Open the
        # new stream first, atomic-swap refs, then close old.
        try:
            new_pa = pa.PyAudio()
            common_kwargs = dict(
                format=pa.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=new_device_index,
                frames_per_buffer=1024,
                start=False,
                stream_callback=self._stream_callback,
            )
            try:
                new_stream = new_pa.open(**common_kwargs)
                self._resample_src_rate = None
                self._resample_src_channels = None
            except Exception:
                # PA-side resampler refused — fall back to native rate
                # + software resample in _maybe_resample.
                common_kwargs["rate"] = int(new_rate) or 48000
                common_kwargs["channels"] = max(1, int(new_channels) or 1)
                new_stream = new_pa.open(**common_kwargs)
                self._resample_src_rate = int(common_kwargs["rate"])
                self._resample_src_channels = int(common_kwargs["channels"])
        except Exception:
            try:
                if new_pa is not None:
                    new_pa.terminate()
            except Exception:
                pass
            return False
        # Callback mode: start the new stream first (callbacks
        # begin firing into our shared _callback_queue), then swap
        # references atomically, then stop+close the old stream
        # OUTSIDE the lock so its final callbacks don't deadlock.
        try:
            new_stream.start_stream()
        except Exception:
            try:
                new_stream.close()
            except Exception:
                pass
            try:
                new_pa.terminate()
            except Exception:
                pass
            return False
        with self._swap_lock:
            old_stream = self._stream
            old_pa = self._pa
            self._stream = new_stream
            self._pa = new_pa
            self._device_index = int(new_device_index)
        # Now drain + close old outside the lock.
        try:
            if old_stream is not None:
                try:
                    old_stream.stop_stream()
                except Exception:
                    pass
                try:
                    old_stream.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if old_pa is not None:
                old_pa.terminate()
        except Exception:
            pass
        return True

    def _maybe_resample(self, data: bytes) -> bytes:
        """Linear-interp resample `data` from
        (_resample_src_rate, _resample_src_channels) to
        (self.rate, self.channels). No-op when no resampler is
        installed (the common case after a same-format swap).
        Cheap: ~0.5 ms per 21 ms chunk via numpy."""
        src_rate = self._resample_src_rate
        src_channels = self._resample_src_channels
        if src_rate is None or src_channels is None:
            return data
        if src_rate == self.rate and src_channels == self.channels:
            return data
        try:
            import numpy as np
            # Reinterpret bytes as int16 samples, reshape to
            # (frames, src_channels).
            arr = np.frombuffer(data, dtype=np.int16)
            if arr.size == 0:
                return data
            if src_channels > 1:
                if arr.size % src_channels != 0:
                    return data  # ragged - skip
                arr = arr.reshape(-1, src_channels)
            else:
                arr = arr.reshape(-1, 1)
            # Channel adapt first.
            if src_channels != self.channels:
                if self.channels == 1:
                    # Down-mix to mono via average.
                    arr = arr.mean(axis=1, keepdims=True).astype(np.int16)
                elif self.channels == 2 and src_channels == 1:
                    # Up-mix to stereo by duplicating.
                    arr = np.repeat(arr, 2, axis=1)
                else:
                    # Other channel mappings: just truncate / pad.
                    if arr.shape[1] > self.channels:
                        arr = arr[:, :self.channels]
                    else:
                        pad = np.zeros(
                            (arr.shape[0], self.channels - arr.shape[1]),
                            dtype=np.int16,
                        )
                        arr = np.concatenate([arr, pad], axis=1)
            # Rate adapt via linear interp.
            if src_rate != self.rate:
                ratio = float(self.rate) / float(src_rate)
                new_len = int(arr.shape[0] * ratio)
                if new_len <= 0:
                    return data
                x_old = np.linspace(0.0, 1.0, num=arr.shape[0], endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
                out = np.empty((new_len, arr.shape[1]), dtype=np.int16)
                for c in range(arr.shape[1]):
                    out[:, c] = np.interp(x_new, x_old, arr[:, c]).astype(np.int16)
                arr = out
            return arr.tobytes()
        except Exception:
            # On any failure, return the original bytes — better to
            # have slightly-mistimed audio than no audio at all.
            return data

    def start(self) -> bool:
        """Open the loopback stream + spawn the writer thread.
        Returns True on success, False if PyAudioWPatch is missing,
        the device went away, or the open failed for any other
        reason. On False the parent should kill ffmpeg and fall
        back to a no-system-audio command."""
        try:
            import pyaudiowpatch as pa  # type: ignore
        except Exception as exc:
            self._on_error(f"pyaudiowpatch import failed: {exc}")
            return False
        try:
            self._pa = pa.PyAudio()
        except Exception as exc:
            self._on_error(f"PyAudio init failed: {exc}")
            return False
        # paInt16 is the format ffmpeg expects (-f s16le). Some
        # exotic render endpoints (Atmos, audiophile DACs) expose
        # float32 mix-format only; PortAudio's resampler/converter
        # handles the conversion transparently in shared mode. If
        # the open still fails (exclusive-mode lock, device gone),
        # we return False and the caller falls back cleanly.
        try:
            if self._use_callback_mode:
                # Open with stream_callback. start=False so we can
                # write primer + pre-pad to stdin before audio
                # samples start flowing (callback fires the moment
                # the stream starts).
                self._stream = self._pa.open(
                    format=pa.paInt16,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    input_device_index=self._device_index,
                    frames_per_buffer=1024,
                    start=False,
                    stream_callback=self._stream_callback,
                )
            else:
                self._stream = self._pa.open(
                    format=pa.paInt16,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    input_device_index=self._device_index,
                    frames_per_buffer=1024,
                )
        except Exception as exc:
            self._on_error(f"WASAPI loopback open failed: {exc}")
            try:
                if self._pa is not None:
                    self._pa.terminate()
            except Exception:
                pass
            self._pa = None
            return False
        if self._use_callback_mode:
            # CALLBACK MODE: write primer + pre-pad to the pipe
            # FIRST so ffmpeg's input #0 probe completes before
            # any callback samples arrive. Then spawn the drain
            # thread (writes queued callback data to stdin) and
            # start the stream — callbacks begin firing.
            try:
                self._write_init_silence()
            except Exception as exc:
                self._on_error(f"primer write failed: {exc}")
                # Best-effort: continue anyway, the drain thread
                # will write whatever comes in.
            self._drain_thread = threading.Thread(
                target=self._drain_callback_queue,
                name=f"{self._label}_drain",
                daemon=True,
            )
            self._drain_thread.start()
            try:
                self._stream.start_stream()
            except Exception as exc:
                self._on_error(f"stream.start_stream failed: {exc}")
                self._stop.set()
                return False
        else:
            self._thread = threading.Thread(
                target=self._run, name=self._label, daemon=True
            )
            self._thread.start()
        return True

    def _write_init_silence(self) -> None:
        """Write the primer + anchor pre-pad silence to stdin.
        Used by callback mode (where primer/pre-pad can't happen
        inside the read loop because there is no read loop)."""
        bytes_per_chunk = 1024 * int(self.channels) * 2
        silence_chunk = b"\x00" * bytes_per_chunk
        start_t = time.time()
        anchor_pad_seconds = 0.0
        if self._align_to_wall_time is not None and self._align_to_wall_time > 0:
            anchor_pad_seconds = max(0.0, start_t - self._align_to_wall_time)
        anchor_pad_seconds = min(10.0, anchor_pad_seconds)
        # Whole 1024-frame chunks only — see polling-mode comment
        # about odd-byte LOUD STATIC for the gory details.
        anchor_pad_chunks = int(
            anchor_pad_seconds * float(self.rate) / 1024.0
        )
        total_init_chunks = max(1, anchor_pad_chunks)
        for _ in range(total_init_chunks):
            self._stdin.write(silence_chunk)
        try:
            self._stdin.flush()
        except Exception:
            pass
        if anchor_pad_chunks > 0 and self._align_to_wall_time is not None:
            self.first_sample_at = self._align_to_wall_time
        else:
            self.first_sample_at = start_t

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """PortAudio realtime callback — MUST return fast. Enqueues
        the captured bytes for the drain thread to write to the
        pipe/socket. Returning paContinue keeps the stream alive."""
        if self._stop.is_set():
            try:
                import pyaudiowpatch as _pa
                return (None, _pa.paComplete)
            except Exception:
                return (None, 0)
        if in_data and self._callback_queue is not None:
            # If a swap installed a software resampler (different
            # rate or channels on the new device), bring the chunk
            # back to the original (rate, channels) before
            # enqueueing so ffmpeg keeps seeing bytes at its
            # original -ar/-ac.
            if self._resample_src_rate is not None:
                in_data = self._maybe_resample(in_data)
            # Liveness signal for the main-thread watchdog.
            try:
                import time as _t
                self.last_real_data_at = _t.time()
            except Exception:
                pass
            try:
                self._callback_queue.put_nowait(in_data)
            except queue.Full:
                # Drop the chunk silently. ffmpeg is back-pressuring
                # the drain thread; better to lose a few ms of audio
                # than back up the realtime audio thread.
                pass
        try:
            import pyaudiowpatch as _pa
            return (None, _pa.paContinue)
        except Exception:
            return (None, 0)

    def _drain_callback_queue(self) -> None:
        """Pull captured chunks off the callback queue and write
        them to stdin/socket. This is the only thread that writes
        to the pipe in callback mode, so no IO sync needed."""
        import time as _time
        bytes_written = 0
        chunks_dropped = 0
        start_t = _time.time()
        next_log_at = start_t + 0.5
        q = self._callback_queue
        if q is None:
            return
        while not self._stop.is_set():
            try:
                data = q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._stdin.write(data)
            except (BrokenPipeError, OSError, ValueError):
                break
            bytes_written += len(data)
            now_t = _time.time()
            if now_t >= next_log_at:
                try:
                    kb = bytes_written // 1024
                    qsize = q.qsize()
                    sys.stderr.write(
                        f"[wasapi-bridge] {self._label} (callback): "
                        f"{kb} KB written, queue depth={qsize} "
                        f"(elapsed={now_t - start_t:.1f}s)\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                next_log_at = now_t + 5.0

    def _run(self) -> None:
        stream = self._stream
        stdin = self._stdin
        import time as _time
        # The bridge writes 1024-frame chunks to ffmpeg's stdin (or
        # to the TCP socket) at the device rate. Why we don't just
        # blocking-read from PortAudio and pipe through:
        #
        # WASAPI loopback in shared mode only produces samples while
        # an application is actively rendering audio. If the system
        # is idle (no music, no game, no notification sound), the
        # loopback stream returns NOTHING — `stream.read(1024)` blocks
        # for as long as the device is silent. Observed in production:
        # 66 seconds of zero samples while the user wasn't playing
        # anything.
        #
        # That stall cascades catastrophically: ffmpeg opens its
        # inputs in order, and `-f s16le -i pipe:0` doesn't return
        # from avformat_open_input until SOME data arrives. While
        # ffmpeg is blocked opening input #0, it can't open input #1
        # (the mic's `tcp://127.0.0.1:PORT`). Our TCP acceptor times
        # out after 30 seconds with "mic TCP listener never received
        # an ffmpeg connection," closes the listener, and when ffmpeg
        # finally gets unblocked and tries to open the TCP input it
        # hits ECONNREFUSED (-138). The whole audio cache subprocess
        # exits and the clip has NO audio.
        #
        # Fix: use `get_read_available()` to NEVER block on read.
        # When the device hasn't produced anything, write a 1024-
        # frame silence chunk paced at the device rate to keep
        # ffmpeg's pipe fed. ffmpeg opens both inputs immediately,
        # the TCP accept succeeds, and the clip captures real audio
        # the moment the device starts producing it.
        bytes_per_chunk = 1024 * int(self.channels) * 2  # paInt16
        silence_chunk = b"\x00" * bytes_per_chunk
        # Two operating modes:
        #
        # ACTIVE — the device is producing real audio. We read what
        # it gives us and write it out as-is. We do NOT silence-fill
        # the small per-tick gaps that show up when the device is
        # slightly under nominal rate; doing so produced an audible
        # ~5 Hz stutter in the user's clips (the "behind a fan"
        # symptom). Audio sounds smooth, exactly as the OS captured
        # it. The audio FILE may be a few % shorter than wall time
        # over a long cache run; the export reads each segment's
        # wall time from file mtime so the segment selection stays
        # correct regardless.
        #
        # SILENCE — entered only when the device hasn't produced
        # real audio for more than 500 ms straight (a truly idle /
        # muted loopback, e.g. no music or game playing). In this
        # mode we generate silence at the NOMINAL device rate so
        # ffmpeg's input never starves — segments keep getting
        # written, the manifest keeps getting entries, the export
        # has audio to anchor against. The first time real audio
        # reappears we drop back into ACTIVE mode immediately.
        #
        # Previous attempt tried writing one silence chunk per
        # 500 ms stall detection, which works out to ~4 % of
        # nominal rate — ffmpeg starved, no segments completed,
        # the clip had no audio at all (the user-reported "no
        # audio from anything" symptom).
        long_stall_threshold = 0.5
        tick_seconds = 1024.0 / max(1.0, float(self.rate))
        last_real_at = _time.time()
        next_silence_due = 0.0
        silence_mode = False
        real_bytes = 0
        silence_bytes = 0
        bytes_total = 0
        start_t = _time.time()
        next_log_at = start_t + 0.5  # first heartbeat after 500ms
        silence_warned = False
        # ALIGN PRE-PAD: when we were given an `align_to_wall_time`
        # (typically the sys-loopback bridge's first_sample_at when
        # we're the mic bridge), pre-pad silence equal to the gap
        # between that wall time and our own start_t. This shifts
        # our stream's effective PTS=0 to the align target so amix
        # mixes us with the other input at matching wall times.
        # Without this, the mic input lands at amix output PTS=0
        # alongside sys input PTS=0 even though mic byte 0 was
        # captured ~4 seconds after sys byte 0 (TCP accept wait) —
        # the user-reported "mic plays N seconds ahead of video"
        # symptom. The pre-pad bytes count toward bytes_total so
        # heartbeat numbers stay honest.
        anchor_pad_seconds = 0.0
        if self._align_to_wall_time is not None and self._align_to_wall_time > 0:
            anchor_pad_seconds = max(0.0, start_t - self._align_to_wall_time)
        # Cap at 10 s of silence — beyond that we've lost a bridge
        # entirely and padding 30 s of silence is worse than just
        # letting the offset stand.
        anchor_pad_seconds = min(10.0, anchor_pad_seconds)
        # Compute pre-pad in WHOLE 1024-frame chunks. Partial trailing
        # blocks (e.g. `b"\x00" * 1` when the seconds-to-bytes math
        # rounds to an odd byte count) destroy PCM alignment — ffmpeg
        # then logs 'Invalid PCM packet, data has size 1 but at least
        # a size of 2 was expected' and interprets every subsequent
        # sample at the wrong byte boundary, producing LOUD STATIC
        # across the whole clip. Whole 1024-frame chunks are sample-
        # aligned by construction.
        anchor_pad_chunks = int(
            anchor_pad_seconds * float(self.rate) / 1024.0
        )
        # Always write AT LEAST ONE silence chunk (the primer) so
        # ffmpeg's `-f s16le -i pipe:0` avformat_open_input probe
        # completes within milliseconds — without it ffmpeg waits up
        # to 30+ seconds on a silent endpoint and the mic TCP
        # acceptor times out before ffmpeg dials in.
        total_init_chunks = max(1, anchor_pad_chunks)
        try:
            for _ in range(total_init_chunks):
                stdin.write(silence_chunk)
                silence_bytes += bytes_per_chunk
                bytes_total += bytes_per_chunk
            try:
                stdin.flush()
            except Exception:
                pass
            # When we wrote real pre-pad (more than just the primer),
            # anchor first_sample_at to the ALIGN target so downstream
            # consumers treat our stream's PTS=0 as that wall moment.
            # Otherwise anchor to start_t (the bridge's own start).
            if anchor_pad_chunks > 0 and self._align_to_wall_time is not None:
                self.first_sample_at = self._align_to_wall_time
            else:
                self.first_sample_at = start_t
        except (BrokenPipeError, OSError, ValueError):
            # ffmpeg already exited (unlikely this early but possible
            # if the spawn failed). Nothing to do — fall through to
            # the loop which will see the broken pipe on its first
            # write and exit cleanly.
            pass
        try:
            while not self._stop.is_set():
                now_t = _time.time()
                data = None
                # Hot-swap pickup (polling mode). PortAudio's WASAPI
                # shared-mode loopback refuses to deliver to two
                # concurrent loopback streams in the same process,
                # so we MUST close OLD before opening NEW. Both
                # operations can each take 100-500 ms on Windows
                # (Pa_Initialize enumerates devices, WASAPI client
                # allocation negotiates with the audio engine).
                # Doing the full close-then-open synchronously inside
                # the _run loop pauses the bridge for 200-1000 ms,
                # during which silence isn't written either — the
                # downstream segment muxer's mtime cadence gets
                # delayed by the gap duration, the chain math
                # treats post-swap segments as if their wall_start
                # was the delayed mtime, and the export aligner
                # ends up reading audio that's the gap duration
                # EARLIER than it should be.
                #
                # Fix: spawn a worker thread for the close-then-open.
                # _run keeps emitting paced silence chunks at the
                # device rate (the `stream = None` + silence-backfill
                # path), so ffmpeg's input pipe never stalls and
                # segment mtimes stay on schedule. When the worker
                # signals done we atomic-publish the new stream and
                # real-audio capture resumes seamlessly.
                pending = None
                with self._swap_lock:
                    if self._swap_pending is not None:
                        pending = self._swap_pending
                        self._swap_pending = None
                if pending is not None:
                    try:
                        tag = pending[0] if isinstance(pending, tuple) and len(pending) >= 1 else None
                        if tag == "request":
                            _, new_idx, new_rate, new_ch = pending
                            # Reject if a previous swap worker is
                            # still in flight — keep the prior swap
                            # going rather than racing two.
                            if self._swap_worker_busy:
                                try:
                                    import sys as _sys
                                    _sys.stderr.write(
                                        f"[wasapi-bridge] {self._label}: swap "
                                        f"request to idx={new_idx} dropped "
                                        f"(previous swap still in flight)\n"
                                    )
                                    _sys.stderr.flush()
                                except Exception:
                                    pass
                            else:
                                # Detach OLD from the loop FIRST so
                                # silence backfill takes over and the
                                # bridge stops competing with the
                                # worker's close. Worker now owns
                                # OLD's teardown + NEW's open.
                                old_stream = stream
                                old_pa = self._pa
                                stream = None
                                self._stream = None
                                self._pa = None
                                self._swap_worker_busy = True
                                self._swap_worker_result = None

                                def _do_swap(
                                    _old_stream=old_stream,
                                    _old_pa=old_pa,
                                    _new_idx=int(new_idx),
                                    _new_rate=int(new_rate),
                                    _new_ch=int(new_ch),
                                    _label=self._label,
                                    _channels=self.channels,
                                    _rate=self.rate,
                                ):
                                    """Close OLD, open NEW. Runs on
                                    a worker thread; result is
                                    delivered via
                                    self._swap_worker_result."""
                                    err = None
                                    new_pa_local = None
                                    new_stream_local = None
                                    new_resample_src_rate = None
                                    new_resample_src_channels = None
                                    try:
                                        if _old_stream is not None:
                                            try:
                                                _old_stream.stop_stream()
                                            except Exception:
                                                pass
                                            try:
                                                _old_stream.close()
                                            except Exception:
                                                pass
                                        if _old_pa is not None:
                                            try:
                                                _old_pa.terminate()
                                            except Exception:
                                                pass
                                        import pyaudiowpatch as _pa_w  # type: ignore
                                        new_pa_local = _pa_w.PyAudio()
                                        open_kwargs = dict(
                                            format=_pa_w.paInt16,
                                            channels=_channels,
                                            rate=_rate,
                                            input=True,
                                            input_device_index=_new_idx,
                                            frames_per_buffer=1024,
                                        )
                                        try:
                                            new_stream_local = new_pa_local.open(**open_kwargs)
                                        except Exception:
                                            open_kwargs["rate"] = _new_rate or 48000
                                            open_kwargs["channels"] = max(1, _new_ch or 1)
                                            new_stream_local = new_pa_local.open(**open_kwargs)
                                            new_resample_src_rate = int(open_kwargs["rate"])
                                            new_resample_src_channels = int(open_kwargs["channels"])
                                    except Exception as _exc:
                                        err = _exc
                                        if new_pa_local is not None:
                                            try:
                                                new_pa_local.terminate()
                                            except Exception:
                                                pass
                                        new_pa_local = None
                                        new_stream_local = None
                                    self._swap_worker_result = (
                                        new_stream_local, new_pa_local, _new_idx,
                                        new_resample_src_rate,
                                        new_resample_src_channels,
                                        err,
                                    )

                                try:
                                    self._swap_worker_thread = threading.Thread(
                                        target=_do_swap,
                                        name=f"{self._label}-swap",
                                        daemon=True,
                                    )
                                    self._swap_worker_thread.start()
                                except Exception:
                                    # Worker thread couldn't start —
                                    # release the busy flag so the
                                    # next watchdog tick can retry,
                                    # and don't leave us silently
                                    # dropping future swap requests.
                                    self._swap_worker_busy = False
                                    self._swap_worker_thread = None
                                # Force silence backfill ON immediately
                                # so ffmpeg's pipe doesn't stall for the
                                # ~500 ms long_stall_threshold while the
                                # worker is still doing PA close+open.
                                # next_silence_due = now_t makes the
                                # first chunk write happen this tick.
                                silence_mode = True
                                next_silence_due = now_t
                                try:
                                    import sys as _sys
                                    _sys.stderr.write(
                                        f"[wasapi-bridge] {self._label}: "
                                        f"swap worker started -> new_idx={new_idx} "
                                        f"(loop emitting silence while close+open run)\n"
                                    )
                                    _sys.stderr.flush()
                                except Exception:
                                    pass
                        elif isinstance(pending, tuple) and len(pending) == 5:
                            # Legacy pre-opened-stream protocol (callback
                            # mode used to stage here in earlier
                            # builds; kept for the unlikely case of a
                            # cross-version handoff after auto-update).
                            new_stream_l, new_pa_l, new_idx_l, old_stream_l, old_pa_l = pending
                            stream = new_stream_l
                            self._stream = new_stream_l
                            self._pa = new_pa_l
                            self._device_index = int(new_idx_l)
                            try:
                                if old_stream_l is not None:
                                    try:
                                        old_stream_l.stop_stream()
                                    except Exception:
                                        pass
                                    try:
                                        old_stream_l.close()
                                    except Exception:
                                        pass
                                if old_pa_l is not None:
                                    try:
                                        old_pa_l.terminate()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        # Swap dispatch failed — bridge keeps running
                        # on whatever stream state it had before the
                        # request was picked up. Silence backfill will
                        # take over if stream is now None.
                        pass
                # Worker-result harvest. The swap worker thread runs
                # close-OLD + open-NEW off-loop so that this _run loop
                # can keep emitting paced silence chunks and ffmpeg's
                # input pipe never stalls. When the worker finishes,
                # it writes the result tuple to _swap_worker_result;
                # we atomic-publish on the next iteration.
                if self._swap_worker_busy:
                    result = self._swap_worker_result
                    if result is not None:
                        self._swap_worker_result = None
                        self._swap_worker_busy = False
                        try:
                            (new_stream_w, new_pa_w, new_idx_w,
                             new_resample_src_rate, new_resample_src_channels,
                             err_w) = result
                        except Exception:
                            new_stream_w = None
                            new_pa_w = None
                            new_idx_w = self._device_index
                            new_resample_src_rate = None
                            new_resample_src_channels = None
                            err_w = None
                        if err_w is not None or new_stream_w is None:
                            try:
                                import sys as _sys
                                _sys.stderr.write(
                                    f"[wasapi-bridge] {self._label}: "
                                    f"swap worker FAILED on idx={new_idx_w}: "
                                    f"{type(err_w).__name__ if err_w else 'no stream'}"
                                    f"{(': ' + str(err_w)) if err_w else ''} "
                                    f"— bridge stays in silence-only mode "
                                    f"(watchdog will retry)\n"
                                )
                                _sys.stderr.flush()
                            except Exception:
                                pass
                            # Already detached at dispatch time.
                            stream = None
                            self._stream = None
                            self._pa = None
                        else:
                            stream = new_stream_w
                            self._stream = new_stream_w
                            self._pa = new_pa_w
                            self._device_index = int(new_idx_w)
                            self._resample_src_rate = new_resample_src_rate
                            self._resample_src_channels = new_resample_src_channels
                            # Drain the new stream's WASAPI client buffer.
                            # The PA stream has been open and capturing
                            # the entire time the worker was running
                            # (~200-1000 ms typical, up to several
                            # seconds on slow systems). Without drain,
                            # the next reads return that backlog AS
                            # FAST AS THE LOOP CAN ITERATE — bursting
                            # several seconds of file content into
                            # ffmpeg in ~50 ms of wall, delaying
                            # segment mtimes by the burst duration and
                            # shifting subsequent chain math earlier by
                            # the same amount. User observed this as
                            # the post-swap audio leading video by 4-6 s.
                            # The silence-fill the loop just emitted
                            # already covered the wall gap during the
                            # swap — the buffered real samples would
                            # duplicate that coverage, so discarding is
                            # correct.
                            drained_chunks = 0
                            try:
                                # Drain in 256-sample increments so we
                                # leave at most 255 frames in the buffer
                                # (~5 ms at 48 k) instead of 1023 frames
                                # (~21 ms). The wider <1024 boundary
                                # combined with silence_mode staying
                                # True caused the post-swap loop to
                                # emit an extra silence chunk before
                                # falling through to a real read,
                                # duplicating wall coverage and
                                # presenting as a 2 s drift after a
                                # swap (verified via workflow's
                                # adversarial pass on this code path).
                                while True:
                                    avail_drain = int(new_stream_w.get_read_available())
                                    if avail_drain < 256:
                                        break
                                    take = 1024 if avail_drain >= 1024 else avail_drain
                                    new_stream_w.read(take, exception_on_overflow=False)
                                    drained_chunks += 1
                                    if drained_chunks > 4000:
                                        break  # safety
                            except Exception:
                                pass
                            # Exit silence mode now that the new stream
                            # is published AND drained. The next loop
                            # iteration reads real audio if the device
                            # is producing; if it isn't, the 500 ms
                            # stall threshold re-engages silence mode
                            # naturally — same as a cold bridge start.
                            silence_mode = False
                            last_real_at = now_t
                            try:
                                import sys as _sys
                                _sys.stderr.write(
                                    f"[wasapi-bridge] {self._label}: "
                                    f"swap worker landed OK -> dev_idx={new_idx_w} "
                                    f"(drained {drained_chunks} pre-buffered chunks "
                                    f"= {drained_chunks * 1024 / max(1.0, float(self.rate)):.2f}s)"
                                    f"{' (resampling)' if new_resample_src_rate else ''}\n"
                                )
                                _sys.stderr.flush()
                            except Exception:
                                pass
                avail = 0
                if stream is not None:
                    try:
                        avail = int(stream.get_read_available())
                    except Exception:
                        avail = 0
                if avail >= 1024 and stream is not None:
                    # Real audio available — read and write it.
                    # Always exit silence mode the moment real data
                    # comes back so we don't double-write.
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                    except Exception as exc:
                        self._on_error(f"WASAPI read error: {exc}")
                        break
                    if data:
                        real_bytes += len(data)
                        last_real_at = now_t
                        silence_mode = False
                        # Liveness signal for main-thread watchdog.
                        self.last_real_data_at = now_t
                else:
                    # No real data this tick.
                    if not silence_mode:
                        if (now_t - last_real_at) >= long_stall_threshold:
                            # Crossed the long-stall threshold —
                            # enter silence mode. Schedule the next
                            # silence chunk for "now" so we start
                            # writing it immediately.
                            silence_mode = True
                            next_silence_due = now_t
                        else:
                            # Brief gap; wait for real data without
                            # silence-filling.
                            _time.sleep(0.005)
                            continue
                    # In silence mode — pace silence chunks at the
                    # nominal device rate so ffmpeg's input never
                    # starves and segments keep getting written.
                    if now_t < next_silence_due:
                        _time.sleep(min(0.020, next_silence_due - now_t))
                        continue
                    data = silence_chunk
                    silence_bytes += len(data)
                    next_silence_due += tick_seconds
                    # If we fell so far behind the silence schedule
                    # that the next chunk is already overdue, snap
                    # to "now" — avoids a burst when waking from a
                    # long sleep.
                    if next_silence_due < now_t:
                        next_silence_due = now_t + tick_seconds
                if not data:
                    # Defensive: avail >= 1024 but read returned empty.
                    _time.sleep(0.005)
                    continue
                if self.first_sample_at is None:
                    # Stamp wall-clock of the FIRST chunk written
                    # (silence or real). The clip-export aligner uses
                    # this as the wall time of file_offset 0; with
                    # silence-fill the audio file's t=0 is the moment
                    # this bridge started writing, NOT when the device
                    # eventually produced real samples — exactly what
                    # we want for sync against video.
                    self.first_sample_at = _time.time()
                bytes_total += len(data)
                if (not silence_warned
                        and (now_t - start_t) > 2.0
                        and real_bytes == 0):
                    self._on_error(
                        f"{self._label}: device opened OK but produced "
                        "ZERO real samples in the first 2.0s — endpoint "
                        "may be muted, locked by another app, or not "
                        "playing audio. Writing silence to keep ffmpeg's "
                        "pipe fed; real samples will be captured the "
                        "moment the device starts producing them."
                    )
                    silence_warned = True
                if now_t >= next_log_at:
                    # Heartbeat: first one at 500ms, then every 5s.
                    # Split real vs silence so a misbehaving bridge
                    # (or muted/idle device) is obvious in the log.
                    try:
                        import sys as _sys
                        kb_r = real_bytes // 1024
                        kb_s = silence_bytes // 1024
                        _sys.stderr.write(
                            f"[wasapi-bridge] {self._label}: "
                            f"real={kb_r} KB silence={kb_s} KB "
                            f"dev_idx={self._device_index} "
                            f"(elapsed={now_t - start_t:.1f}s, "
                            f"first_chunk_at_offset="
                            f"{(self.first_sample_at - start_t) * 1000:.0f}ms)\n"
                        )
                        _sys.stderr.flush()
                    except Exception:
                        pass
                    # Schedule next heartbeat 5s out.
                    next_log_at = now_t + 5.0
                # If a swap installed a software resampler, bring the
                # chunk back to original (rate, channels) before write.
                # silence_chunk is already at native format (bytes_per_chunk
                # is computed from self.channels) so no resample needed
                # for the silence path.
                if (data is not silence_chunk
                        and self._resample_src_rate is not None):
                    data = self._maybe_resample(data)
                try:
                    stdin.write(data)
                except (BrokenPipeError, OSError, ValueError):
                    # ffmpeg exited / pipe closed — normal shutdown
                    # path, no logging. ValueError comes from Python's
                    # io.BufferedWriter when stdin was closed
                    # between the check and the write (race with the
                    # ffmpeg-exit handler).
                    break
        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            try:
                if self._pa is not None:
                    self._pa.terminate()
            except Exception:
                pass
            if self._close_stdin_on_exit:
                try:
                    stdin.close()
                except Exception:
                    pass

    def stop(self, timeout: float = 1.5) -> None:
        """Signal the writer to exit and wait briefly for the
        thread to drain. Safe to call multiple times; safe to call
        when start() returned False (no-op)."""
        self._stop.set()
        # Reap any unconsumed swap_device() handoff so we don't
        # leak a new PA stream that the _run loop never picked up.
        try:
            with self._swap_lock:
                pending = self._swap_pending
                self._swap_pending = None
        except Exception:
            pending = None
        if pending is not None:
            try:
                # Two pending shapes coexist:
                #   * ("request", new_idx, new_rate, new_ch) — polling-
                #     mode params-only handoff. Nothing to reap (no
                #     pre-opened resources).
                #   * (new_stream, new_pa, new_idx, old_stream, old_pa)
                #     — legacy pre-opened-stream protocol. Close both
                #     ends to avoid leaking a PA stream.
                tag = pending[0] if isinstance(pending, tuple) and len(pending) >= 1 else None
                if tag == "request":
                    pass  # nothing to reap
                elif isinstance(pending, tuple) and len(pending) >= 2:
                    new_stream = pending[0]
                    new_pa = pending[1]
                    try:
                        if new_stream is not None:
                            try:
                                new_stream.stop_stream()
                            except Exception:
                                pass
                            try:
                                new_stream.close()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        if new_pa is not None:
                            new_pa.terminate()
                    except Exception:
                        pass
            except Exception:
                pass
        if self._use_callback_mode:
            # Order matters here. The crash on clip-save was from
            # tearing PortAudio down while the realtime callback
            # was still in flight. Sequence:
            # 1. Stop the stream — callback stops firing.
            # 2. Join the drain thread — finishes whatever was in
            #    the queue (drain checks self._stop on its 100 ms
            #    timeout cycle so this is bounded).
            # 3. Close stdin so the next test's spawn doesn't see
            #    a leaked pipe.
            # 4. THEN close the stream and terminate PyAudio.
            #    Doing this LAST avoids the realtime-thread vs
            #    main-thread race that tripped the previous
            #    teardown.
            stream = self._stream
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
            dt = self._drain_thread
            if dt is not None:
                try:
                    dt.join(timeout=timeout)
                except Exception:
                    pass
            if self._close_stdin_on_exit:
                try:
                    self._stdin.close()
                except Exception:
                    pass
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None
        # Polling-mode tear-down (existing _run thread joins).
        t = self._thread
        if t is not None:
            try:
                t.join(timeout=timeout)
            except Exception:
                pass


def probe_input_device_format(
    device_name: Optional[str] = None,
    *,
    device_index: Optional[int] = None,
    fallback_rate: int = 48000,
    max_channels: int = 1,
) -> Optional[tuple[int, int, int]]:
    """Resolve a microphone (WASAPI input endpoint) to its
    (device_index, rate, channels) format using PyAudioWPatch.

    If `device_name` is given, find the WASAPI input device whose
    name matches (case-insensitive substring is allowed — Windows
    sometimes prefixes the friendly name with the channel label or
    appends `(N- ...)`). If not given (or no match), fall back to
    the PortAudio default WASAPI input.

    Returns None if PyAudioWPatch is missing, no input endpoint is
    available, or every candidate raised.

    This is the mic counterpart to `probe_default_loopback_format`.
    Both targets are then opened by `WasapiLoopbackWriter` with
    `input=True, input_device_index=N` — PyAudio uses the same call
    shape for loopback and real-input devices; the difference is
    purely which device index we hand it.
    """
    try:
        import pyaudiowpatch as pa  # type: ignore
    except Exception:
        return None
    p = None
    try:
        p = pa.PyAudio()
        # PRIORITY 1: caller passed an explicit device_index (typically
        # the voice listener's resolved sounddevice index). Both
        # sounddevice and PyAudioWPatch wrap the SAME PortAudio backend
        # and enumerate devices in the SAME order, so the index is
        # cross-compatible — the safest resolution path because it
        # bypasses every name-format quirk between the two libraries.
        # PortAudio's "default input" can also be wrong on Windows
        # (PyAudioWPatch returns MME default, not WASAPI default — they
        # can be different physical devices), so this path also avoids
        # the bug where the fallback picks an unplugged headset.
        if device_index is not None:
            try:
                info = p.get_device_info_by_index(int(device_index))
                if int(info.get("maxInputChannels", 0) or 0) > 0:
                    rate = int(info.get("defaultSampleRate", fallback_rate)
                               or fallback_rate)
                    ch = min(max_channels, int(
                        info.get("maxInputChannels", 1) or 1))
                    return (int(device_index), rate, max(1, ch))
            except Exception:
                pass  # fall through to name / default resolution
        # Resolve the WASAPI host API index. Inputs from other host
        # APIs (MME, DirectSound) have different latency and worse
        # driver behavior; we want WASAPI to match the rest of the
        # voice pipeline.
        wasapi_index: Optional[int] = None
        try:
            host_count = p.get_host_api_count()
            for hi in range(host_count):
                info = p.get_host_api_info_by_index(hi)
                name = str(info.get("name", "") or "").strip().lower()
                if "wasapi" in name:
                    wasapi_index = int(info.get("index", hi))
                    break
        except Exception:
            wasapi_index = None
        # Walk every input device and pick the best match.
        wanted = (device_name or "").strip().lower()
        chosen_idx: Optional[int] = None
        chosen_rate: int = fallback_rate
        chosen_channels: int = 1
        try:
            device_count = p.get_device_count()
        except Exception:
            device_count = 0
        candidates: list[tuple[int, dict]] = []
        for di in range(device_count):
            try:
                info = p.get_device_info_by_index(di)
            except Exception:
                continue
            try:
                max_input_channels = int(info.get("maxInputChannels", 0) or 0)
            except Exception:
                max_input_channels = 0
            if max_input_channels <= 0:
                continue
            # Skip render-endpoint loopback wrappers (they expose
            # maxInputChannels>0 but they're SYSTEM audio, not mic
            # input). Their name typically ends in " [Loopback]".
            name = str(info.get("name", "") or "")
            if name.lower().endswith("[loopback]"):
                continue
            if wasapi_index is not None:
                try:
                    if int(info.get("hostApi", -1)) != wasapi_index:
                        continue
                except Exception:
                    continue
            candidates.append((di, dict(info)))
        # Exact-name match first, then substring, then default fallback.
        if wanted:
            for di, info in candidates:
                if str(info.get("name", "") or "").strip().lower() == wanted:
                    chosen_idx = di
                    break
            if chosen_idx is None:
                for di, info in candidates:
                    nm = str(info.get("name", "") or "").strip().lower()
                    if wanted in nm or nm in wanted:
                        chosen_idx = di
                        break
        if chosen_idx is None:
            # Fall back to PortAudio's default WASAPI input. If the
            # default API isn't WASAPI (it usually IS on Windows but
            # not guaranteed), pick the first WASAPI input we saw.
            try:
                if wasapi_index is not None:
                    host_info = p.get_host_api_info_by_index(wasapi_index)
                    default_in = host_info.get("defaultInputDevice", -1)
                    if isinstance(default_in, int) and default_in >= 0:
                        chosen_idx = default_in
            except Exception:
                pass
            if chosen_idx is None and candidates:
                chosen_idx = candidates[0][0]
        if chosen_idx is None:
            return None
        # Pull the canonical format off the chosen device.
        try:
            info = p.get_device_info_by_index(chosen_idx)
            chosen_rate = int(info.get("defaultSampleRate") or fallback_rate) or fallback_rate
            mic_chs = int(info.get("maxInputChannels") or 1) or 1
            chosen_channels = max(1, min(int(max_channels), mic_chs))
        except Exception:
            pass
        return (int(chosen_idx), int(chosen_rate), int(chosen_channels))
    except Exception:
        return None
    finally:
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass


def _query_default_render_friendly_name_via_com() -> Optional[str]:
    """Query Windows' IMMDeviceEnumerator for the CURRENT default
    render endpoint's friendly name. Bypasses PortAudio entirely —
    PortAudio (and PyAudioWPatch's get_default_wasapi_loopback)
    caches the default at PA_Initialize time and DOES NOT re-resolve
    after Windows fires OnDefaultDeviceChanged. The user reported
    this concrete symptom: switching default playback in Windows
    Sound settings did not change the watchdog's fingerprint until
    the app was restarted.

    Pycaw.AudioUtilities.GetSpeakers() re-resolves on every call
    (it calls IMMDeviceEnumerator::GetDefaultAudioEndpoint freshly)
    so it tracks Windows state correctly. We only need the friendly
    name — the matching PA loopback wrapper is then found by name
    suffix match in the identity probe.

    Returns None if pycaw or COM init failed. Logs are silent so
    this can be called in a tight watchdog poll loop without noise.
    """
    try:
        # comtypes auto-inits COM on the calling thread. Qt's main
        # thread already runs CoInitialize(STA) via the Qt event loop
        # init, so this is free in the watchdog tick.
        from pycaw.pycaw import AudioUtilities  # type: ignore
        spk = AudioUtilities.GetSpeakers()
        if spk is None:
            return None
        name = getattr(spk, "FriendlyName", None)
        if not name:
            return None
        return str(name)
    except Exception:
        return None


def probe_default_loopback_identity() -> Optional[tuple[int, str]]:
    """(index, name) probe of the CURRENT Windows default playback
    endpoint that WASAPI loopback would capture. Used by the audio-
    endpoint watchdog to detect mid-session output-device changes
    (user swaps speakers ↔ headset) without paying the full open-
    stream cost of probe_default_loopback_format().

    Resolution strategy (in order):
      1. Live COM query → IMMDeviceEnumerator returns the current
         default render device's friendly name. THIS is the only
         path that survives a Windows default-output change at
         runtime; PyAudioWPatch's get_default_wasapi_loopback()
         caches at PA_Initialize even across PA instance recreation,
         so a watchdog driven by it can NEVER see the switch.
      2. Match the friendly name against PA's loopback wrappers.
         PyAudioWPatch names loopback wrappers '<friendly> [Loopback]',
         so the friendly prefix + the '[Loopback]' suffix uniquely
         identifies the wrapper index we'd open for capture.
      3. Fallback to legacy get_default_wasapi_loopback() if either
         the COM query or the PA enumeration leg fails. Keeps the
         existing behavior on machines where pycaw isn't available
         (e.g. partial Windows installs); the watchdog will still
         work for swaps the cached path happens to catch.

    Returns None if no default playback endpoint can be identified
    by either path. ~5-10 ms in practice."""
    try:
        import pyaudiowpatch as pa  # type: ignore
    except Exception:
        return None
    friendly = _query_default_render_friendly_name_via_com()
    p = None
    try:
        p = pa.PyAudio()
        # Path 1+2: COM friendly name -> PA loopback wrapper.
        if friendly:
            target_loopback = f"{friendly} [Loopback]"
            try:
                count = int(p.get_device_count())
            except Exception:
                count = 0
            for i in range(count):
                try:
                    info = p.get_device_info_by_index(i)
                except Exception:
                    continue
                name = str(info.get("name", ""))
                if name == target_loopback:
                    return (int(i), name)
            # Fuzzier match: some friendly names may differ in
            # punctuation/casing between PA's enumeration and COM's
            # response. Try a prefix + suffix match.
            for i in range(count):
                try:
                    info = p.get_device_info_by_index(i)
                except Exception:
                    continue
                name = str(info.get("name", ""))
                if name.startswith(friendly) and name.endswith("[Loopback]"):
                    return (int(i), name)
        # Path 3 (legacy fallback): PA's cached default lookup.
        try:
            info = p.get_default_wasapi_loopback()
            return (int(info.get("index", -1)), str(info.get("name", "")))
        except Exception:
            return None
    except Exception:
        return None
    finally:
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass


def probe_input_device_identity(
    name_hint: Optional[str] = None,
    *,
    device_index: Optional[int] = None,
) -> Optional[tuple[int, str]]:
    """Cheap (index, name) probe of the user's preferred mic.
    Mirror of probe_input_device_format but without the format
    fields. Resolution priority matches the full probe so the
    watchdog sees the SAME device the cache writer actually opens:

      1. device_index (typically voice listener's resolved index)
      2. case-insensitive substring match on name_hint
      3. PortAudio default WASAPI input
    """
    try:
        import pyaudiowpatch as pa  # type: ignore
    except Exception:
        return None
    p = None
    try:
        p = pa.PyAudio()
        if device_index is not None and int(device_index) >= 0:
            try:
                info = p.get_device_info_by_index(int(device_index))
                if int(info.get("maxInputChannels", 0) or 0) > 0:
                    return (int(device_index), str(info.get("name", "")))
            except Exception:
                pass
        if name_hint:
            name_lower = str(name_hint).lower().strip()
            if name_lower:
                for i in range(p.get_device_count()):
                    try:
                        info = p.get_device_info_by_index(i)
                        if int(info.get("maxInputChannels", 0) or 0) <= 0:
                            continue
                        nm = str(info.get("name", "")).lower()
                        if name_lower in nm or nm in name_lower:
                            return (int(i), str(info.get("name", "")))
                    except Exception:
                        continue
        try:
            info = p.get_default_input_device_info()
            return (int(info.get("index", -1)), str(info.get("name", "")))
        except Exception:
            return None
    except Exception:
        return None
    finally:
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass


def probe_default_loopback_format() -> Optional[tuple[int, int, int]]:
    """Open + immediately close a PyAudio instance to discover the
    default WASAPI loopback endpoint's (device_index, rate, channels).
    Returns None if PyAudioWPatch is missing, no default playback
    endpoint exists, or any step raises.

    Done BEFORE ffmpeg is spawned so the `-ar`/`-ac` args for the
    pipe input match exactly what the writer thread will produce —
    a mismatch causes pitched / fast / slow playback even though
    the bytes flow fine.

    Resolution: uses the same COM-first strategy as
    probe_default_loopback_identity() so a watchdog-detected
    endpoint change yields a swap target that matches the detection.
    """
    try:
        import pyaudiowpatch as pa  # type: ignore
    except Exception:
        return None
    friendly = _query_default_render_friendly_name_via_com()
    p = None
    try:
        p = pa.PyAudio()
        target_info = None
        # Path 1+2: COM friendly name -> PA loopback wrapper, then
        # pull rate/channels from the wrapper's PA device info.
        if friendly:
            target_loopback = f"{friendly} [Loopback]"
            try:
                count = int(p.get_device_count())
            except Exception:
                count = 0
            for i in range(count):
                try:
                    info = p.get_device_info_by_index(i)
                except Exception:
                    continue
                name = str(info.get("name", ""))
                if name == target_loopback or (
                    name.startswith(friendly) and name.endswith("[Loopback]")
                ):
                    target_info = info
                    break
        # Path 3 (fallback): PA's cached default lookup.
        if target_info is None:
            try:
                target_info = p.get_default_wasapi_loopback()
            except Exception:
                return None
        idx = int(target_info.get("index"))
        rate = int(target_info.get("defaultSampleRate") or 48000) or 48000
        channels = int(target_info.get("maxInputChannels") or 2) or 2
        return (idx, rate, max(1, channels))
    except Exception:
        return None
    finally:
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass


class TcpPcmAcceptor:
    """Listen on an ephemeral localhost TCP port, accept the FIRST
    inbound connection (ffmpeg dialing back), and expose the accepted
    socket as a file-like handle suitable for `WasapiLoopbackWriter`.

    Why TCP and not a second `pipe:`? ffmpeg's child stdin is a single
    `pipe:0`, and inheriting additional anonymous-pipe FDs through
    Popen on Windows is awkward (no native `pass_fds` for arbitrary
    FDs; child-side FDs are renumbered). A loopback TCP listener
    lets ffmpeg connect to us via `-i tcp://127.0.0.1:PORT` — a
    well-trodden ffmpeg input route, supported on every platform,
    no extra dependencies. Localhost-only, ephemeral port, accepts
    exactly one connection before closing the listener, so it's
    safe against external connection attempts.

    Lifecycle:
        acceptor = TcpPcmAcceptor()
        if not acceptor.bind(): ... (bail)
        port = acceptor.port
        # Spawn ffmpeg with `-i tcp://127.0.0.1:{port}` somewhere in
        # its input args.
        sock_file = acceptor.accept(timeout=5.0)
        if sock_file is None: ... (ffmpeg never connected, bail)
        # Pass sock_file to WasapiLoopbackWriter as its `ffmpeg_stdin`.
        writer = WasapiLoopbackWriter(sock_file, ..., close_stdin_on_exit=True)
        writer.start()
    """

    def __init__(self) -> None:
        self._listener: Optional[socket.socket] = None
        self._accepted: Optional[socket.socket] = None
        self.port: int = 0

    def bind(self) -> bool:
        """Open a listening socket on 127.0.0.1 with an OS-assigned
        ephemeral port. Returns True on success; sets `self.port`."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            s.settimeout(5.0)
            self._listener = s
            self.port = int(s.getsockname()[1])
            return True
        except Exception:
            self.close()
            return False

    def accept(self, *, timeout: float = 5.0) -> Optional[object]:
        """Block until ffmpeg connects to the listening socket, then
        return a write-capable file object (`sock.makefile("wb")`).
        Returns None if no client connected within `timeout`.

        Closes the listener after the first successful accept so the
        port is freed (we only ever expect one connection per audio
        cache session).
        """
        listener = self._listener
        if listener is None:
            return None
        try:
            listener.settimeout(max(0.1, float(timeout)))
            conn, _addr = listener.accept()
        except Exception:
            return None
        finally:
            try:
                listener.close()
            except Exception:
                pass
            self._listener = None
        try:
            # Disable Nagle so small raw-PCM chunks ship immediately.
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        self._accepted = conn
        try:
            return conn.makefile("wb", buffering=0)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            self._accepted = None
            return None

    def close(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass
        accepted = self._accepted
        self._accepted = None
        if accepted is not None:
            try:
                accepted.close()
            except Exception:
                pass

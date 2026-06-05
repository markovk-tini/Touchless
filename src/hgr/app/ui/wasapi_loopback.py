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
        self._thread = threading.Thread(
            target=self._run, name=self._label, daemon=True
        )
        self._thread.start()
        return True

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
        # READ THE REAL AUDIO AT DEVICE RATE. Earlier iterations of
        # this loop tried to wall-clock-pace by silence-filling any
        # gap, but the user heard the inserted silence chunks as a
        # constant fan-like stutter (~5 silence chunks per second
        # when the WASAPI loopback under-delivered at ~89% of
        # nominal). Solution: don't silence-fill in the middle.
        # Just write whatever the device gives us. The audio file's
        # duration may be slightly shorter than wall time but it
        # sounds SMOOTH — and the export now reads segment wall
        # times from file mtime (not from `a_anchor + file_time`),
        # so under-delivery no longer mis-aligns the segments
        # selected for the clip window.
        #
        # The ONLY safety net we keep is a "long stall" silence-
        # fill: if `stream.read()` produces no real data for more
        # than 500 ms straight, we write one silence chunk to keep
        # ffmpeg's pipe alive (the original cascade-failure fix:
        # without ANY bytes flowing, ffmpeg's avformat_open_input
        # blocks indefinitely on input #0 and never gets to opening
        # input #1, the mic TCP acceptor times out, ffmpeg dies).
        # A 500 ms silence is well below the threshold where a
        # listener perceives a stutter, and only fires when the
        # device is genuinely idle for that long.
        long_stall_threshold = 0.5
        last_real_at = _time.time()
        real_bytes = 0
        silence_bytes = 0
        bytes_total = 0
        start_t = _time.time()
        next_log_at = start_t + 0.5  # first heartbeat after 500ms
        silence_warned = False
        # PRIMER: write one chunk of silence to stdin BEFORE the read
        # loop starts. Unblocks ffmpeg's `-f s16le -i pipe:0`
        # avformat_open_input probe within milliseconds so it can
        # move on to open input #1 (mic TCP) — without the primer
        # ffmpeg waited up to 30+ seconds on a silent endpoint and
        # the mic TCP acceptor timed out before ffmpeg dialed in.
        try:
            stdin.write(silence_chunk)
            try:
                stdin.flush()
            except Exception:
                pass
            silence_bytes += len(silence_chunk)
            bytes_total += len(silence_chunk)
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
                # Non-blocking check for available frames. If the
                # device hasn't produced 1024 yet, don't read (would
                # block) — just sleep briefly and re-check. The
                # 500 ms long-stall watchdog below covers truly idle
                # endpoints (silent loopback) so ffmpeg's pipe never
                # underruns long enough to kill the cache.
                avail = 0
                try:
                    avail = int(stream.get_read_available())
                except Exception:
                    avail = 0
                if avail >= 1024:
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                    except Exception as exc:
                        self._on_error(f"WASAPI read error: {exc}")
                        break
                    if data:
                        real_bytes += len(data)
                        last_real_at = now_t
                elif (now_t - last_real_at) >= long_stall_threshold:
                    # Long stall — device idle / muted for > 500 ms.
                    # Write one silence chunk to keep ffmpeg fed and
                    # reset the stall timer so we don't burst silence.
                    data = silence_chunk
                    silence_bytes += len(data)
                    last_real_at = now_t
                else:
                    # Brief gap (< 500 ms since last real chunk).
                    # Just sleep one tick and retry; don't silence-
                    # fill — silence-fill in this band is exactly
                    # what produced the user-perceived stutter.
                    _time.sleep(0.005)
                    continue
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
                            f"(elapsed={now_t - start_t:.1f}s, "
                            f"first_chunk_at_offset="
                            f"{(self.first_sample_at - start_t) * 1000:.0f}ms)\n"
                        )
                        _sys.stderr.flush()
                    except Exception:
                        pass
                    # Schedule next heartbeat 5s out.
                    next_log_at = now_t + 5.0
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


def probe_default_loopback_format() -> Optional[tuple[int, int, int]]:
    """Open + immediately close a PyAudio instance to discover the
    default WASAPI loopback endpoint's (device_index, rate, channels).
    Returns None if PyAudioWPatch is missing, no default playback
    endpoint exists, or any step raises.

    Done BEFORE ffmpeg is spawned so the `-ar`/`-ac` args for the
    pipe input match exactly what the writer thread will produce —
    a mismatch causes pitched / fast / slow playback even though
    the bytes flow fine.
    """
    try:
        import pyaudiowpatch as pa  # type: ignore
    except Exception:
        return None
    p = None
    try:
        p = pa.PyAudio()
        info = p.get_default_wasapi_loopback()
        idx = int(info.get("index"))
        rate = int(info.get("defaultSampleRate") or 48000) or 48000
        channels = int(info.get("maxInputChannels") or 2) or 2
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

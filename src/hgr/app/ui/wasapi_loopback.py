"""WASAPI loopback bridge for the clip-cache ffmpeg subprocess.

The bundled ffmpeg (gyan.dev 7.0.2-full_build) is built without the
`wasapi` indev — no released ffmpeg has one (trac #9408). We capture
the Windows render-endpoint output in Python via PyAudioWPatch (a
Windows-only PortAudio fork with native WASAPI loopback, MIT, ships
prebuilt wheels) and pipe the raw PCM bytes into ffmpeg's stdin via
`-f s16le -i pipe:0`.

Lifecycle:
- `_start_clip_cache_ffmpeg` probes the default loopback endpoint,
  spawns ffmpeg with `stdin=PIPE`, then constructs + starts a
  `WasapiLoopbackWriter`. The writer owns its PortAudio stream and a
  daemon thread that copies bytes into ffmpeg's stdin until either
  stopped or the pipe breaks.
- `_stop_clip_cache_ffmpeg` calls `writer.stop()` BEFORE
  `_stop_ffmpeg_process(...)` so the thread exits its read loop
  cleanly instead of spinning on a broken pipe and logging a stack.

If `pyaudiowpatch` isn't installed, or no default loopback endpoint
exists, the probe returns None and the bridge is never spawned —
ffmpeg runs without `-f s16le -i pipe:0` and the clip cache falls
through to mic-only or video-only paths.
"""

from __future__ import annotations

import sys
import threading
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
    ) -> None:
        self._stdin = ffmpeg_stdin
        self._device_index = int(device_index)
        self.rate = int(rate)
        self.channels = max(1, int(channels))
        self._on_error = on_error or (lambda _msg: None)
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
            target=self._run, name="WasapiLoopback", daemon=True
        )
        self._thread.start()
        return True

    def _run(self) -> None:
        stream = self._stream
        stdin = self._stdin
        import time as _time
        try:
            while not self._stop.is_set():
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                except Exception as exc:
                    self._on_error(f"WASAPI read error: {exc}")
                    break
                if not data:
                    continue
                if self.first_sample_at is None:
                    # Stamp first-sample arrival exactly once. Used by
                    # the clip-export aligner so audio time anchors at
                    # the moment audio actually started flowing, not at
                    # the (earlier) moment ffmpeg's process spawned.
                    self.first_sample_at = _time.time()
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

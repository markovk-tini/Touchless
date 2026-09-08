"""Streaming PCM16 audio player for the Realtime API.

The Realtime API emits `response.output_audio.delta` events with base64-
encoded PCM16 chunks at the model's fixed output rate of 24 kHz mono.
This module exposes a tiny wrapper around `sounddevice.RawOutputStream`
that:

  * runs on the sounddevice audio thread (NOT the Qt thread)
  * locks the playback rate to 24 kHz so Windows can't silently
    upsample to 48 kHz and play our 24 kHz data at 2x speed (the
    chipmunk bug from the first QAudioSink-based version)
  * pulls chunks from a thread-safe Queue in the sd callback so the
    websocket reader thread never blocks
  * falls silent (no exception, no crash) if the device is busy or
    sounddevice isn't importable

Author: Konstantin Markov
"""
from __future__ import annotations

import queue
import threading
from typing import Optional

# OpenAI gpt-realtime ALWAYS emits PCM16 at 24 kHz mono regardless of the
# `rate` we requested in session.audio.output.format — locking the player
# to 24000 matches the wire and avoids resampling-by-mismatch.
PLAYBACK_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (int16)


class AudioPlayer:
    """Queue PCM16 audio bytes for streaming playback. Thread-safe.

    Construction never raises — failures (no sounddevice, no output
    device) leave the player in is_enabled()==False, and write() becomes
    a no-op."""

    def __init__(self, *, sample_rate: int = PLAYBACK_RATE,
                 volume: float = 0.9, parent=None) -> None:
        # `parent` is accepted for API parity with the QObject-based
        # version; unused here.
        del parent
        # IGNORE the requested sample_rate — see PLAYBACK_RATE comment
        # above. Kept in the signature for backwards compat.
        del sample_rate
        self._volume = max(0.0, min(1.0, float(volume)))
        # Bounded queue so a runaway producer can't OOM the process if
        # playback stalls. ~30 seconds at 24kHz / typical chunk sizes.
        self._buf: queue.Queue[bytes] = queue.Queue(maxsize=512)
        # Residual bytes left over from a queue chunk that was larger
        # than the audio callback's requested frame block.
        self._tail = b""
        self._stream = None
        self._enabled = False
        # Init-failure breadcrumb so the manager can log WHY playback
        # never started (sounddevice missing vs device busy vs format
        # rejected). Stays None on success.
        self._init_error: Optional[str] = None
        self._lock = threading.Lock()
        self._init_stream()

    # ---- init / teardown ---------------------------------------------------

    def _init_stream(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            self._init_error = f"sounddevice_import: {exc!r}"
            return
        try:
            # blocksize=0 lets PortAudio pick a sensible size; 24kHz mono
            # int16 is uncontroversial on Windows / Linux / macOS so we
            # don't need to negotiate.
            self._stream = sd.RawOutputStream(
                samplerate=PLAYBACK_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=0,
                latency="low",
                callback=self._callback,
            )
            self._stream.start()
            self._enabled = True
        except Exception as exc:
            self._stream = None
            self._enabled = False
            self._init_error = f"stream_open: {exc!r}"

    def is_enabled(self) -> bool:
        return self._enabled

    def init_error(self) -> Optional[str]:
        """Returns a short string describing why init_stream failed, or
        None on success. Used by the manager to surface silent failures
        in logs ('no sounddevice', 'no output device', etc.)."""
        return self._init_error

    def restart_stream(self) -> bool:
        """Close + reopen the output stream so it picks up the user's
        current Windows default audio device. sounddevice binds the
        device at stream-open; without this, swapping speakers ↔
        headset mid-session keeps audio on the original device.

        Returns True if the new stream came up enabled, False otherwise.
        Best-effort: drops queued audio for the swap to take effect."""
        with self._lock:
            old = self._stream
            self._stream = None
            self._enabled = False
            self._tail = b""
            # Drain pending audio so the old device doesn't drain
            # those samples before the swap.
            try:
                while True:
                    self._buf.get_nowait()
            except queue.Empty:
                pass
        if old is not None:
            try:
                old.stop(ignore_errors=True)
            except TypeError:
                # Older sounddevice API didn't take ignore_errors.
                try:
                    old.stop()
                except Exception:
                    pass
            try:
                old.close(ignore_errors=True)
            except TypeError:
                try:
                    old.close()
                except Exception:
                    pass
            except Exception:
                pass
        self._init_stream()
        return self._enabled

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, float(volume)))

    # ---- writes ------------------------------------------------------------

    def write(self, pcm16_bytes: bytes) -> None:
        """Public, thread-safe enqueue. Safe to call from any thread."""
        if not pcm16_bytes or not self._enabled:
            return
        try:
            self._buf.put_nowait(bytes(pcm16_bytes))
        except queue.Full:
            # Drop the oldest chunk and try again — losing 20-100ms of
            # audio is better than blocking the websocket thread.
            try:
                self._buf.get_nowait()
            except queue.Empty:
                pass
            try:
                self._buf.put_nowait(bytes(pcm16_bytes))
            except queue.Full:
                pass

    def _callback(self, outdata, frames: int, time, status) -> None:
        """sounddevice pull callback — fills `outdata` with exactly
        `frames * CHANNELS * SAMPLE_WIDTH` bytes. Runs on the audio
        thread; must not block. Underflows fill with zeros (silence)."""
        # noinspection PyUnusedLocal
        del time, status
        needed = frames * CHANNELS * SAMPLE_WIDTH
        chunks: list = []
        have = 0
        # Drain residual tail from a previous oversize chunk first.
        if self._tail:
            chunks.append(self._tail)
            have += len(self._tail)
            self._tail = b""
        # Pull from the queue until we have enough or it runs dry.
        while have < needed:
            try:
                c = self._buf.get_nowait()
            except queue.Empty:
                break
            chunks.append(c)
            have += len(c)
        if have == 0:
            # Underflow — write silence so the stream stays alive without
            # ticks/pops. Common at end-of-utterance.
            outdata[:] = b"\x00" * needed
            return
        joined = b"".join(chunks)
        if len(joined) >= needed:
            # Apply soft volume by clipping if not at 1.0 — keep CPU low
            # by skipping when volume==1.0.
            data = joined[:needed]
            if self._volume < 0.999:
                data = self._scale_pcm16(data, self._volume)
            outdata[:needed] = data
            # Stash whatever's left over for the next callback.
            self._tail = joined[needed:]
        else:
            # Have some data but not enough; write what we have, pad
            # the rest with silence.
            if self._volume < 0.999:
                joined = self._scale_pcm16(joined, self._volume)
            outdata[:len(joined)] = joined
            outdata[len(joined):needed] = b"\x00" * (needed - len(joined))

    @staticmethod
    def _scale_pcm16(data: bytes, volume: float) -> bytes:
        # Simple int-domain scale. Use audioop where available (built-in,
        # fast); fall back to manual scaling. audioop was removed in
        # Python 3.13; covering both keeps installs working.
        try:
            import audioop
            return audioop.mul(data, SAMPLE_WIDTH, volume)
        except Exception:
            import array
            arr = array.array("h")
            arr.frombytes(data)
            for i, v in enumerate(arr):
                arr[i] = max(-32768, min(32767, int(v * volume)))
            return arr.tobytes()

    def stop(self) -> None:
        """Stop playback and release the stream. Safe to call multiple
        times."""
        with self._lock:
            self._enabled = False
            stream = self._stream
            self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        # Drain any pending chunks so the queue doesn't hold memory.
        while True:
            try:
                self._buf.get_nowait()
            except queue.Empty:
                break
        self._tail = b""

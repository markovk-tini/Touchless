from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .llama_server import LlamaServer


_SENTENCE_ENDINGS = (".", "!", "?")
_MIN_CORRECTION_CHARS = 20
_DEFAULT_INTERVAL_SECONDS = 4.0
_DEFAULT_IDLE_SECONDS = 0.5
_LONG_IDLE_SECONDS = 2.0
# Idle-trigger mode: correction only fires after the user has been
# silent (no append() in this long) for this many seconds. 0 = old
# interval-based behaviour. Set to a real value to switch modes.
_DEFAULT_IDLE_TRIGGER_SECONDS = 10.0


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    tail: str = ""


class GrammarCorrector:
    def __init__(
        self,
        *,
        server: LlamaServer,
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        min_idle_seconds: float = _DEFAULT_IDLE_SECONDS,
        idle_trigger_seconds: float = _DEFAULT_IDLE_TRIGGER_SECONDS,
        on_correction: Optional[Callable[[CorrectionResult], None]] = None,
    ) -> None:
        self._server = server
        self._interval = max(2.0, float(interval_seconds))
        self._min_idle = max(0.0, float(min_idle_seconds))
        # When > 0 the corrector waits for this much silence (no
        # append() activity) before pulling a chunk. Each new
        # append() resets _last_append, so a fresh utterance during
        # the wait cancels the trigger and the timer restarts. The
        # idle-triggered chunk takes EVERYTHING in the buffer in
        # one shot -- the user has clearly paused so a sentence
        # boundary isn't required.
        self._idle_trigger = max(0.0, float(idle_trigger_seconds))
        self._on_correction = on_correction
        self._lock = threading.Lock()
        self._buffer = ""
        self._chunk_in_flight: Optional[str] = None
        self._tail_since_chunk = ""
        self._chunk_stale = False
        self._last_run = time.monotonic()
        self._last_append = time.monotonic()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._busy = False

    @property
    def backend(self) -> Optional[str]:
        return self._server.backend

    @property
    def available(self) -> bool:
        return self._server.available

    def set_callback(self, on_correction: Optional[Callable[[CorrectionResult], None]]) -> None:
        self._on_correction = on_correction

    def reset(self) -> None:
        with self._lock:
            self._buffer = ""
            self._chunk_in_flight = None
            self._tail_since_chunk = ""
            self._chunk_stale = False
            now = time.monotonic()
            self._last_run = now
            self._last_append = now

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._buffer += text
            self._last_append = time.monotonic()
            if self._chunk_in_flight is not None:
                self._tail_since_chunk += text
                # Idle-trigger mode: any new audio mid-correction
                # means the user kept talking before the LLM
                # finished. Abort the in-flight correction so we
                # don't replace text on top of what they're still
                # typing. The buffer's growing content will be
                # picked up on the next idle-trigger fire.
                if self._idle_trigger > 0:
                    self._chunk_stale = True

    def sync_replace(self, chars_to_remove: int, new_tail: str) -> None:
        with self._lock:
            if self._chunk_in_flight is not None:
                self._chunk_stale = True
            if chars_to_remove > 0:
                if chars_to_remove >= len(self._buffer):
                    self._buffer = ""
                else:
                    self._buffer = self._buffer[:-chars_to_remove]
                if self._chunk_in_flight is not None and self._tail_since_chunk:
                    if chars_to_remove >= len(self._tail_since_chunk):
                        self._tail_since_chunk = ""
                    else:
                        self._tail_since_chunk = self._tail_since_chunk[:-chars_to_remove]
            if new_tail:
                self._buffer += new_tail
                if self._chunk_in_flight is not None:
                    self._tail_since_chunk += new_tail

    def snapshot_tail(self) -> str:
        with self._lock:
            return self._tail_since_chunk

    def mark_correction_done(self) -> None:
        with self._lock:
            self._chunk_in_flight = None
            self._tail_since_chunk = ""
            self._chunk_stale = False

    def is_chunk_stale(self) -> bool:
        with self._lock:
            return self._chunk_stale

    def start(self) -> bool:
        if not self._server.available:
            print(f"[grammar] start skipped: server not available ({self._server.message})")
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        if not self._server.running:
            print(f"[grammar] launching llama-server ({self._server.backend})...")
            if not self._server.start():
                print(f"[grammar] llama-server failed to start: {self._server.message}")
                return False
        self._stop_event.clear()
        self.reset()
        self._thread = threading.Thread(target=self._run_loop, name="hgr-grammar-corrector", daemon=True)
        self._thread.start()
        if self._idle_trigger > 0:
            print(f"[grammar] corrector started (idle-trigger={self._idle_trigger:.1f}s, abort-on-resume=on)")
        else:
            print(f"[grammar] corrector started (interval={self._interval:.1f}s)")
        return True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        # Final pass: if there's still uncorrected text in the
        # buffer when dictation ends, kick a one-off correction so
        # the user's last thought gets cleaned up too. Async on a
        # daemon thread so stop() doesn't block its caller on the
        # 1-3 s LLM round-trip; if the app shuts down before the
        # thread finishes, nothing depends on it.
        self._spawn_final_pass()

    def _spawn_final_pass(self) -> None:
        if not self._server.available:
            return
        with self._lock:
            if self._chunk_in_flight is not None:
                # An idle-triggered correction is already in flight;
                # let _run_loop's normal apply path handle it. Don't
                # double-fire.
                return
            if not self._buffer or len(self._buffer) < _MIN_CORRECTION_CHARS:
                return
            chunk = self._buffer
            self._buffer = ""
            self._chunk_in_flight = chunk
            self._tail_since_chunk = ""
            self._chunk_stale = False
        callback = self._on_correction

        def _worker() -> None:
            corrected: Optional[str] = None
            try:
                corrected = self._server.correct(chunk)
            except Exception as exc:
                print(f"[grammar] final-pass correct() raised: {exc}")
            if not corrected or corrected.strip() == chunk.strip():
                print(f"[grammar] final-pass: no change ({len(chunk)} chars)")
                self.mark_correction_done()
                return
            if self.is_chunk_stale():
                # User started dictating again after we kicked off
                # the final pass -- their typing wins.
                self.mark_correction_done()
                return
            if callback is None:
                self.mark_correction_done()
                return
            try:
                callback(CorrectionResult(original=chunk, corrected=corrected))
            except Exception as exc:
                print(f"[grammar] final-pass callback raised: {exc}")
            self.mark_correction_done()

        try:
            threading.Thread(target=_worker, name="hgr-grammar-final-pass", daemon=True).start()
        except Exception:
            self.mark_correction_done()

    def _run_loop(self) -> None:
        last_idle_log = 0.0
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=1.0):
                break
            # In idle-trigger mode the gating is purely "has the
            # user been silent long enough?"; the per-interval
            # throttle would just delay the first correction by up
            # to _interval seconds past the idle threshold for no
            # benefit. _take_chunk_to_boundary still won't return
            # a chunk while one is in flight, so we can't spin.
            if self._idle_trigger <= 0:
                elapsed = time.monotonic() - self._last_run
                if elapsed < self._interval:
                    continue
            chunk = self._take_chunk_to_boundary()
            if not chunk:
                now = time.monotonic()
                with self._lock:
                    buf_len = len(self._buffer)
                    in_flight = self._chunk_in_flight is not None
                if (buf_len > 0 or in_flight) and now - last_idle_log > 15.0:
                    print(f"[grammar] idle: buffer={buf_len} chars, in_flight={in_flight}")
                    last_idle_log = now
                self._last_run = time.monotonic()
                continue
            preview = chunk if len(chunk) <= 80 else chunk[:77] + "..."
            print(f"[grammar] submitting chunk ({len(chunk)} chars): {preview!r}")
            self._busy = True
            t0 = time.monotonic()
            try:
                corrected = self._server.correct(chunk)
            except Exception as exc:
                print(f"[grammar] correct() raised: {exc}")
                corrected = None
            self._busy = False
            self._last_run = time.monotonic()
            latency = self._last_run - t0
            if not corrected:
                print(f"[grammar] no correction returned ({latency:.1f}s)")
                self.mark_correction_done()
                continue
            if corrected.strip() == chunk.strip():
                print(f"[grammar] no change ({latency:.1f}s)")
                self.mark_correction_done()
                continue
            if self.is_chunk_stale():
                print(f"[grammar] chunk stale (re-decode overlap), discarding correction")
                self.mark_correction_done()
                continue
            corr_preview = corrected if len(corrected) <= 80 else corrected[:77] + "..."
            print(f"[grammar] corrected ({latency:.1f}s): {corr_preview!r}")
            callback = self._on_correction
            if callback is None:
                print(f"[grammar] no callback set, discarding correction")
                self.mark_correction_done()
                continue
            try:
                callback(CorrectionResult(original=chunk, corrected=corrected))
            except Exception as exc:
                print(f"[grammar] callback raised: {exc}")
            self.mark_correction_done()

    def _take_chunk_to_boundary(self) -> str:
        with self._lock:
            if self._chunk_in_flight is not None:
                return ""
            if len(self._buffer) < _MIN_CORRECTION_CHARS:
                return ""
            idle = time.monotonic() - self._last_append
            # Idle-trigger mode: only run after the configured
            # silence threshold. Each new append() resets the
            # timer, so the user dictating again mid-wait cancels
            # the trigger naturally. When fired, take the whole
            # buffer in one shot -- a long pause means "I'm done
            # with this thought," not "wait for a period."
            if self._idle_trigger > 0:
                if idle < self._idle_trigger:
                    return ""
                chunk = self._buffer
                self._buffer = ""
                self._chunk_in_flight = chunk
                self._tail_since_chunk = ""
                self._chunk_stale = False
                return chunk
            # --- Interval mode (legacy) below ---
            if idle < self._min_idle:
                return ""
            boundary = _find_last_sentence_boundary(self._buffer)
            if boundary <= 0:
                # No sentence-ending punctuation in buffer. If the user has
                # been quiet long enough, take everything — whisper-stream
                # often emits finals without trailing periods on short
                # utterances, and we shouldn't stall forever waiting for one.
                if idle >= _LONG_IDLE_SECONDS:
                    boundary = len(self._buffer)
                elif len(self._buffer) < 120:
                    return ""
                else:
                    boundary = _find_last_space(self._buffer, start=80) or len(self._buffer)
            chunk = self._buffer[:boundary]
            self._buffer = self._buffer[boundary:]
            self._chunk_in_flight = chunk
            self._tail_since_chunk = ""
            return chunk


def _find_last_sentence_boundary(text: str) -> int:
    if not text:
        return 0
    n = len(text)
    last = -1
    for idx, char in enumerate(text):
        if char not in _SENTENCE_ENDINGS:
            continue
        # Only count a period/!/? as a sentence boundary when it's not
        # embedded in a token. A "." followed directly by a letter or digit
        # (e.g., "v2.1.3", "example.com", "e.g.") stays inside the token.
        if idx + 1 < n:
            nxt = text[idx + 1]
            if nxt.isalnum():
                continue
        last = idx
    if last < 0:
        return 0
    end = last + 1
    while end < n and text[end] in {' ', '\n', '\t', '"', "'", ')', ']'}:
        end += 1
    return end


def _find_last_space(text: str, *, start: int = 0) -> int:
    if not text:
        return 0
    idx = text.rfind(" ", start)
    return idx + 1 if idx >= 0 else 0

# Author: Konstantin Markov

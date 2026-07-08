from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .whisper_stream import (
    _ANSI_ESCAPE_RE,
    _candidate_whisper_roots,
    _first_existing_model,
    _is_meta_line,
    _resolve_backend_executable,
)


@dataclass
class RefinementResult:
    text: str
    start_time: float
    end_time: float
    duration_seconds: float


_CLI_NAME = "whisper-cli" + (".exe" if sys.platform == "win32" else "")  # extensionless Mach-O on macOS
_TIMESTAMP_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")


def _clean_cli_output(stdout: str) -> str:
    cleaned_lines: list[str] = []
    for raw in (stdout or "").splitlines():
        line = _ANSI_ESCAPE_RE.sub("", raw).strip()
        if not line:
            continue
        if _is_meta_line(line):
            continue
        line = _TIMESTAMP_PREFIX_RE.sub("", line)
        if line:
            cleaned_lines.append(line)
    return " ".join(cleaned_lines).strip()


def _resolve_cli_executable() -> Optional[tuple[str, Path]]:
    resolution = _resolve_backend_executable()
    if resolution is None:
        return None
    backend, stream_exe = resolution
    # whisper-cli.exe lives in the same bin/Release folder as whisper-stream.exe
    candidate = stream_exe.parent / _CLI_NAME
    if candidate.exists():
        return backend, candidate
    # fall back: scan build dirs (build_metal = macOS Apple-GPU build)
    build_dirs = ("build_metal", "build_cuda", "build_vulkan", "build_stream")
    for root in _candidate_whisper_roots():
        for build in build_dirs:
            for sub in ("bin/Release", "bin"):
                candidate = root / build / sub / _CLI_NAME
                if candidate.exists():
                    return backend, candidate
    return None


class WhisperRefiner:
    """Parallel mic capture + VAD + full-context whisper-cli transcription.

    Runs alongside WhisperStreamer. On each VAD-detected utterance boundary,
    writes the buffered audio to a temp WAV and invokes whisper-cli with a
    full 30-sec decode (no -nf, larger beam). The result is delivered via
    the on_refinement callback with the utterance's wall-clock start/end
    timestamps so the caller can replace the streamed text for that span.
    """

    def __init__(
        self,
        *,
        on_refinement: Optional[Callable[[RefinementResult], None]] = None,
        language: str = "en",
        sample_rate: int = 16000,
        vad_energy_threshold: float = 0.006,
        vad_speech_min_ms: int = 180,
        vad_silence_end_ms: int = 600,
        pre_roll_ms: int = 200,
        max_utterance_seconds: float = 30.0,
        beam_size: int = 8,
        threads: Optional[int] = None,
    ) -> None:
        self._on_refinement = on_refinement
        self._language = language
        self._sample_rate = int(sample_rate)
        self._energy_threshold = float(vad_energy_threshold)
        self._speech_min_samples = max(1, int(sample_rate * vad_speech_min_ms / 1000))
        self._silence_end_samples = max(1, int(sample_rate * vad_silence_end_ms / 1000))
        self._pre_roll_samples = max(0, int(sample_rate * pre_roll_ms / 1000))
        self._max_utterance_samples = int(sample_rate * max_utterance_seconds)
        self._beam_size = max(1, int(beam_size))
        self._threads = threads or max(2, min(8, os.cpu_count() or 4))

        self._audio_queue: "queue.Queue[tuple[np.ndarray, float]]" = queue.Queue()
        self._utterance_queue: "queue.Queue[tuple[np.ndarray, float, float]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._stream = None
        self._vad_thread: Optional[threading.Thread] = None
        self._transcribe_thread: Optional[threading.Thread] = None

        self._backend: Optional[str] = None
        self._cli_path: Optional[Path] = None
        self._model_path: Optional[Path] = None
        self._available = False
        self._message = "refiner not started"

        resolution = _resolve_cli_executable()
        if resolution is None:
            self._message = "whisper-cli.exe not found"
            return
        self._backend, self._cli_path = resolution
        self._model_path = _first_existing_model(
            (
                "ggml-medium.en.bin",
                "ggml-small.en.bin",
                "ggml-base.en.bin",
                "ggml-medium.bin",
                "ggml-small.bin",
                "ggml-base.bin",
            )
        )
        if self._model_path is None:
            self._message = "whisper model not found for refiner"
            return
        try:
            import sounddevice  # noqa: F401
        except Exception as exc:
            self._message = f"sounddevice unavailable: {exc}"
            return
        self._available = True
        self._message = f"refiner ready ({self._backend}, {self._model_path.name})"

    @property
    def available(self) -> bool:
        return self._available

    @property
    def message(self) -> str:
        return self._message

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def set_callback(self, callback: Optional[Callable[[RefinementResult], None]]) -> None:
        self._on_refinement = callback

    def start(self) -> bool:
        if not self._available:
            return False
        if self._stream is not None:
            return True
        try:
            import sounddevice as sd
        except Exception as exc:
            self._message = f"sounddevice import failed: {exc}"
            return False
        self._stop_event.clear()
        self._drain_queues()
        try:
            blocksize = int(self._sample_rate * 0.02)  # 20ms frames
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            self._message = f"refiner mic open failed: {exc}"
            return False
        self._vad_thread = threading.Thread(target=self._vad_loop, name="hgr-refiner-vad", daemon=True)
        self._vad_thread.start()
        self._transcribe_thread = threading.Thread(
            target=self._transcribe_loop, name="hgr-refiner-xcribe", daemon=True
        )
        self._transcribe_thread.start()
        self._message = f"refiner active ({self._backend})"
        return True

    def stop(self) -> None:
        self._stop_event.set()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        for thread in (self._vad_thread, self._transcribe_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        self._vad_thread = None
        self._transcribe_thread = None
        self._drain_queues()
        self._message = "refiner stopped"

    def _drain_queues(self) -> None:
        for q in (self._audio_queue, self._utterance_queue):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            pass
        try:
            arr = np.ascontiguousarray(indata[:, 0], dtype=np.float32).copy()
        except Exception:
            return
        self._audio_queue.put((arr, time.monotonic()))

    def _vad_loop(self) -> None:
        speech_buffer: list[np.ndarray] = []
        pre_roll: list[np.ndarray] = []
        pre_roll_samples = 0
        speech_samples = 0
        silence_samples = 0
        in_speech = False
        utterance_start_ts: Optional[float] = None

        while not self._stop_event.is_set():
            try:
                frame, ts = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
            is_speech = rms > self._energy_threshold
            frame_len = int(frame.shape[0])

            if not in_speech:
                # pre-roll ring buffer
                pre_roll.append(frame)
                pre_roll_samples += frame_len
                while pre_roll_samples > self._pre_roll_samples and len(pre_roll) > 1:
                    dropped = pre_roll.pop(0)
                    pre_roll_samples -= int(dropped.shape[0])
                if is_speech:
                    speech_samples += frame_len
                    if speech_samples >= self._speech_min_samples:
                        in_speech = True
                        utterance_start_ts = ts
                        speech_buffer = list(pre_roll)
                        pre_roll = []
                        pre_roll_samples = 0
                        silence_samples = 0
                else:
                    speech_samples = 0
                continue

            # in_speech == True
            speech_buffer.append(frame)
            if is_speech:
                silence_samples = 0
            else:
                silence_samples += frame_len

            total_samples = sum(int(x.shape[0]) for x in speech_buffer)
            end_utterance = (
                silence_samples >= self._silence_end_samples
                or total_samples >= self._max_utterance_samples
            )
            if end_utterance:
                audio = np.concatenate(speech_buffer)
                start = utterance_start_ts if utterance_start_ts is not None else ts
                end = ts
                self._utterance_queue.put((audio, start, end))
                speech_buffer = []
                pre_roll = []
                pre_roll_samples = 0
                speech_samples = 0
                silence_samples = 0
                in_speech = False
                utterance_start_ts = None

    def _transcribe_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio, start_ts, end_ts = self._utterance_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if audio.size < self._sample_rate // 4:  # <0.25s, skip
                continue
            text = self._transcribe(audio)
            if not text:
                continue
            callback = self._on_refinement
            if callback is None:
                continue
            try:
                callback(
                    RefinementResult(
                        text=text,
                        start_time=start_ts,
                        end_time=end_ts,
                        duration_seconds=float(audio.size) / float(self._sample_rate),
                    )
                )
            except Exception as exc:
                print(f"[refiner] callback error: {exc}")

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._cli_path is None or self._model_path is None:
            return ""
        try:
            tmp = tempfile.NamedTemporaryFile(prefix="hgr_refiner_", suffix=".wav", delete=False)
            tmp.close()
            path = Path(tmp.name)
        except Exception as exc:
            print(f"[refiner] temp wav create failed: {exc}")
            return ""
        try:
            self._write_wav(path, audio)
            args = [
                str(self._cli_path),
                "-m",
                str(self._model_path),
                "-l",
                self._language,
                "-t",
                str(self._threads),
                "-bs",
                str(self._beam_size),
                "-f",
                str(path),
                "-nt",  # suppress timestamps
                "-np",  # no prints
            ]
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            try:
                proc = subprocess.run(
                    args,
                    cwd=str(self._cli_path.parent),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30.0,
                    creationflags=creation_flags,
                )
            except subprocess.TimeoutExpired:
                print("[refiner] whisper-cli timed out")
                return ""
            if proc.returncode != 0:
                tail = (proc.stderr or "").strip().splitlines()[-3:]
                print(f"[refiner] whisper-cli rc={proc.returncode}: {' | '.join(tail)}")
                return ""
            return _clean_cli_output(proc.stdout)
        finally:
            try:
                path.unlink()
            except Exception:
                pass

    def _write_wav(self, path: Path, audio: np.ndarray) -> None:
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        import wave

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm.tobytes())

# Author: Konstantin Markov

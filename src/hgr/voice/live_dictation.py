from __future__ import annotations

from typing import Callable, Optional

from .whisper_stream import DictationEvent, WhisperStreamer


LiveDictationEvent = DictationEvent


class LiveDictationStreamer:
    """Live dictation via whisper.cpp streaming only.

    v1.1.7: SAPI fallback removed. Rationale: the fallback ran
    `powershell.exe -EncodedCommand <base64>` to invoke
    System.Speech.Recognition — a byte pattern indistinguishable
    from a malware-dropper fingerprint, which caused Windows
    Defender / ASR to auto-quarantine the app for at least one
    user. Whisper's fast+accurate two-tier pipeline is now the
    exclusive path; there's no runtime fallback if whisper.cpp
    can't start (e.g. GPU driver missing all three backends), just
    a status message. That's acceptable because the packaged spec
    ships every whisper backend variant (CUDA / Vulkan / CPU) so
    the "no backend engages" case only happens on a broken install.

    The old HGR_DICTATION_BACKEND=sapi env override is dropped.
    """

    def __init__(self, *, preferred_microphone_name: Optional[str] = None) -> None:
        self._preferred_mic_name = (preferred_microphone_name or "").strip() or None
        self._whisper: Optional[WhisperStreamer] = WhisperStreamer(
            preferred_microphone_name=self._preferred_mic_name
        )
        self._active_backend = self._whisper.backend if self._whisper is not None else None

    @property
    def available(self) -> bool:
        return self._whisper is not None and self._whisper.available

    @property
    def message(self) -> str:
        if self._whisper is not None:
            return self._whisper.message
        return "dictation unavailable"

    @property
    def backend(self) -> Optional[str]:
        return self._active_backend

    def stream(
        self,
        *,
        stop_event,
        event_callback: Callable[[DictationEvent], None],
    ) -> bool:
        if self._whisper is not None:
            return self._whisper.stream(stop_event=stop_event, event_callback=event_callback)
        return False

# Author: Konstantin Markov

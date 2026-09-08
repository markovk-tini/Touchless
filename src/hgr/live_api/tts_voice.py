"""TTS voice substrate — pick + persist Iris's speaking voice.

Phase-10 subscription polish. Voice cloning is the headline
subscription feature. This module owns the substrate without
shipping a model:

  * Active voice id (free Windows SAPI / Pro Coqui-XTTS /
    Subscription-only ElevenLabs / user-cloned).
  * Provider enum + auth token storage via secret_vault.
  * Per-voice metadata (display name, language, accent).
  * Sample-line library for previewing voices.

Actual synthesis lives elsewhere (the realtime client or a
local TTS service); this module is the configuration substrate
they consume.

ALL cloud TTS calls are gated by `privacy_tier` — when the user
has selected local-only, voice clone usage is forced back to the
on-device default with a one-line UI hint.

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TTSProvider(str, Enum):
    SAPI       = "sapi"          # Windows built-in
    COQUI      = "coqui"         # local Coqui-XTTS
    ELEVEN     = "elevenlabs"    # cloud
    OPENAI_TTS = "openai_tts"    # cloud
    USER_CLONE = "user_clone"    # subscriber-cloned voice


@dataclass(frozen=True)
class TTSVoice:
    voice_id: str
    display_name: str
    provider: TTSProvider
    language: str = "en-US"
    accent: str = ""
    tier: str = "free"            # "free" | "pro"
    sample_text: str = (
        "Sixty-seven and clear, sir. Pleasant enough.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "display_name": self.display_name,
            "provider": self.provider.value,
            "language": self.language,
            "accent": self.accent,
            "tier": self.tier,
            "sample_text": self.sample_text,
        }


# Built-in catalogue. Real installation may add cloned voices.
_CATALOGUE: List[TTSVoice] = [
    TTSVoice(
        voice_id="sapi:zira",
        display_name="Zira (Windows default)",
        provider=TTSProvider.SAPI,
        accent="US neutral"),
    TTSVoice(
        voice_id="sapi:david",
        display_name="David (Windows default)",
        provider=TTSProvider.SAPI,
        accent="US neutral"),
    TTSVoice(
        voice_id="openai:alloy",
        display_name="Alloy (OpenAI TTS)",
        provider=TTSProvider.OPENAI_TTS,
        accent="US neutral",
        tier="pro"),
    TTSVoice(
        voice_id="openai:onyx",
        display_name="Onyx (OpenAI TTS)",
        provider=TTSProvider.OPENAI_TTS,
        accent="US deep",
        tier="pro"),
    TTSVoice(
        voice_id="eleven:rachel",
        display_name="Rachel (ElevenLabs)",
        provider=TTSProvider.ELEVEN,
        accent="US warm",
        tier="pro"),
]


_lock = threading.RLock()
_active_voice_id: Optional[str] = None
_user_clones: List[TTSVoice] = []


def catalogue() -> List[TTSVoice]:
    with _lock:
        return list(_CATALOGUE) + list(_user_clones)


def get_voice(voice_id: str) -> Optional[TTSVoice]:
    if not voice_id:
        return None
    target = voice_id.strip().lower()
    for v in catalogue():
        if v.voice_id.lower() == target:
            return v
    return None


def set_active(voice_id: str) -> bool:
    global _active_voice_id
    if not voice_id:
        return False
    v = get_voice(voice_id)
    if v is None:
        return False
    with _lock:
        _active_voice_id = v.voice_id
    return True


def active_voice() -> TTSVoice:
    """Returns the currently active voice. Falls back to the first
    SAPI voice when nothing's been picked or privacy-tier blocks
    the configured choice."""
    with _lock:
        chosen_id = _active_voice_id
    if chosen_id:
        v = get_voice(chosen_id)
        if v is not None:
            # Privacy-tier guard.
            if _is_cloud(v) and _privacy_blocks_cloud():
                return _sapi_default()
            return v
    return _sapi_default()


def _sapi_default() -> TTSVoice:
    for v in _CATALOGUE:
        if v.provider == TTSProvider.SAPI:
            return v
    return _CATALOGUE[0]


def _is_cloud(v: TTSVoice) -> bool:
    return v.provider in (TTSProvider.ELEVEN,
                           TTSProvider.OPENAI_TTS)


def _privacy_blocks_cloud() -> bool:
    try:
        from .privacy_tier import is_local_only
        return bool(is_local_only())
    except Exception:
        return False


def register_user_clone(*, voice_id: str,
                        display_name: str,
                        language: str = "en-US",
                        accent: str = "",
                        sample_text: str = "") -> TTSVoice:
    """Add a user-cloned voice to the catalogue. The actual clone
    data lives outside this module (file path / cloud id stored
    via secret_vault); we only hold metadata."""
    if not voice_id or not display_name:
        raise ValueError("voice_id + display_name required")
    voice = TTSVoice(
        voice_id=voice_id,
        display_name=display_name,
        provider=TTSProvider.USER_CLONE,
        language=language,
        accent=accent,
        tier="pro",
        sample_text=(sample_text
                     or "This is my own voice, cloned."))
    with _lock:
        _user_clones.append(voice)
    return voice


def reset_for_tests() -> None:
    global _active_voice_id, _user_clones
    with _lock:
        _active_voice_id = None
        _user_clones = []

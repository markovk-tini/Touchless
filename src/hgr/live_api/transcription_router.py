"""Public stub — transcription router lives with Iris."""
from __future__ import annotations

import enum


class TranscriptTier(enum.Enum):
    FAST = "fast"
    ACCURATE = "accurate"


class _Decision:
    def __init__(self, tier: TranscriptTier = TranscriptTier.FAST, reason: str = "stub"):
        self.tier = tier
        self.reason = reason


class TranscriptionRouter:
    def decide(self, *args, **kwargs) -> _Decision:
        return _Decision()


# Author: Konstantin Markov

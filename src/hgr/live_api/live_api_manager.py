"""Public stub — Iris runtime is not published."""
from __future__ import annotations

import enum


class LiveApiState(enum.Enum):
    OFF = "off"
    CONNECTING = "connecting"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    ERROR = "error"


class LiveApiManager:
    def __init__(self, *args, **kwargs) -> None:
        self.state = LiveApiState.OFF

    def start(self, *args, **kwargs) -> None:
        return None

    def stop(self, *args, **kwargs) -> None:
        return None

    def join(self, *args, **kwargs) -> None:
        return None

    def send_audio_chunk(self, *args, **kwargs) -> None:
        return None

    def send_tool_result(self, *args, **kwargs) -> None:
        return None


# Author: Konstantin Markov

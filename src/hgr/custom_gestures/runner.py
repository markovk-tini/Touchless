"""Public stub."""
from __future__ import annotations

from typing import Optional


class CustomGestureRunner:
    def __init__(self, *args, **kwargs) -> None:
        self.current_match = None
        self.has_gestures = False

    def reload(self) -> None:
        return None

    def maybe_reload_if_changed(self, _now: float) -> None:
        return None

    def process_engine_hands(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def hand_lost(self, *_args, **_kwargs) -> None:
        return None


# Author: Konstantin Markov

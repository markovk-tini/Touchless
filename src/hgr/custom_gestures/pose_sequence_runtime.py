"""Public stub."""
from __future__ import annotations

from typing import Optional


class PoseSequenceRuntime:
    def reload(self) -> None:
        return None

    def maybe_reload_if_changed(self, _now: float) -> None:
        return None

    def has_sequences(self) -> bool:
        return False

    def is_in_progress(self) -> bool:
        return False

    def current_banner(self):
        return None

    def hand_lost(self, *_args, **_kwargs) -> None:
        return None

    def process_landmarks(self, *_args, **_kwargs) -> Optional[str]:
        return None


# Author: Konstantin Markov

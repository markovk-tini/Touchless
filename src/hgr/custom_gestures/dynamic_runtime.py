"""Public stub."""
from __future__ import annotations

from typing import Optional


class DynamicGestureRuntime:
    def reload(self) -> None:
        return None

    def maybe_reload_if_changed(self, _now: float) -> None:
        return None

    def has_dynamic_gestures(self) -> bool:
        return False

    def has_loop_or_complex_templates(self) -> bool:
        return False

    def current_match(self, _now: float):
        return None

    def hand_lost(self) -> None:
        return None

    def process_frame(self, *_args, **_kwargs) -> Optional[str]:
        return None


# Author: Konstantin Markov

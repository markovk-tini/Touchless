"""Public stub."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

KIND_BUILTIN = "builtin"
KIND_CUSTOM = "custom"


@dataclass
class SelectableGesture:
    kind: str = KIND_BUILTIN
    id: str = ""
    label: str = ""


def all_selectable_gestures(*_args, **_kwargs):
    return []


def detail_for_profile_gesture(*_args, **_kwargs) -> str:
    return ""


def find_selectable(*_args, **_kwargs) -> Optional[SelectableGesture]:
    return None


# Author: Konstantin Markov

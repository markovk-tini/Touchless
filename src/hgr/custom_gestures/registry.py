"""Public stub."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SHIPPED_CUSTOM_GESTURE_SLOTS = 0
SOURCE_CUSTOM_GESTURE_SLOTS = 0
MAX_CUSTOM_GESTURES = 0


def slot_limit() -> int:
    return 0


def registry_path() -> Path:
    return Path.home() / ".hgr_app" / "custom_gestures.json"


@dataclass
class Action:
    kind: str = "noop"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        return cls(
            kind=str((data or {}).get("kind", "noop")),
            payload=dict((data or {}).get("payload") or {}),
        )


@dataclass
class CustomGesture:
    name: str = ""
    action: Action = field(default_factory=Action)


class GestureRegistry:
    def load(self) -> None:
        return None

    def list(self) -> List[CustomGesture]:
        return []

    def get(self, _name: str) -> Optional[CustomGesture]:
        return None


# Author: Konstantin Markov

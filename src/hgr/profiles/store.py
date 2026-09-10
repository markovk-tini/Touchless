"""Public stub — fail-open so built-in gestures still fire."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_PROFILE_ID = "default"


@dataclass
class GestureProfile:
    id: str = DEFAULT_PROFILE_ID
    name: str = "Default"
    pose_actions: Dict[str, Any] = field(default_factory=dict)


class ProfileStore:
    def __init__(self) -> None:
        self.active_profile_id = DEFAULT_PROFILE_ID
        self.profiles: Dict[str, GestureProfile] = {
            DEFAULT_PROFILE_ID: GestureProfile()
        }

    def list_profiles(self) -> List[GestureProfile]:
        return list(self.profiles.values())

    def write_bindings(self, _bindings: Dict[str, str]) -> None:
        return None

    def apply_to_config(self, _config) -> bool:
        return False

    def pose_action_override(self, _pose_id: str) -> Optional[Dict[str, Any]]:
        return None

    def drop_pose_from_bindings(self, _profile_id: str, _pose_id: str) -> None:
        return None

    def enable_custom_on_active(self, _name: str) -> None:
        return None


_STORE = ProfileStore()


def get_store() -> ProfileStore:
    return _STORE


def ensure_profiles(_config=None) -> None:
    return None


def custom_is_enabled(_name: str) -> bool:
    return True


def pose_is_enabled(_pose_id: str) -> bool:
    return True


def action_is_enabled(_action_id: str, config=None) -> bool:
    return True


# Author: Konstantin Markov

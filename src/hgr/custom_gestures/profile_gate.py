"""Public stub — fail-open so built-in gestures still fire."""


def custom_allowed_in_active_profile(_name: str) -> bool:
    return True


def pose_allowed_in_active_profile(_pose_id: str) -> bool:
    return True


def action_allowed_in_active_profile(_action_id: str, config=None) -> bool:
    return True


# Author: Konstantin Markov

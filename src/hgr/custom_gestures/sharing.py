"""Public stub — gesture packs are not published."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

RESOLVE_OVERWRITE = "overwrite"
RESOLVE_SKIP = "skip"


class BundleError(Exception):
    pass


def export_bundle(*_args, **_kwargs) -> int:
    raise BundleError("Gesture packs are not included in the public source tree.")


def import_bundle(*_args, **_kwargs):
    raise BundleError("Gesture packs are not included in the public source tree.")


def gestures_in_bundle(_path: Path) -> List[str]:
    return []


# Author: Konstantin Markov

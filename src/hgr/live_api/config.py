"""Public stub — Iris config is not published."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LiveApiConfig:
    backend: str = "auto"
    enabled: bool = False


def load_config() -> LiveApiConfig:
    return LiveApiConfig()


# Author: Konstantin Markov

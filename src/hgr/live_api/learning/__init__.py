"""Self-learning daemon — discovers user facts from project files,
external connectors, and behavior logs.

Lives outside ``memory/`` because it ORCHESTRATES the existing memory
machinery rather than implementing storage itself — it reads from the
project store / connector registry / tool-call log and writes through
``MemoryManager.set_fact`` with provenance tags.

Author: Konstantin Markov
"""
from __future__ import annotations

from .self_learner import SelfLearner, default_state_path
from .suggestion_engine import (
    KIND_CONTINUE_TASK,
    KIND_FOLLOW_UP_PERSON,
    KIND_IDEA,
    KIND_OPEN_APP,
    KIND_PROJECT_FOCUS,
    KIND_REMINDER,
    SuggestionEngine,
)

__all__ = [
    "SelfLearner",
    "default_state_path",
    "SuggestionEngine",
    "KIND_OPEN_APP",
    "KIND_CONTINUE_TASK",
    "KIND_FOLLOW_UP_PERSON",
    "KIND_PROJECT_FOCUS",
    "KIND_IDEA",
    "KIND_REMINDER",
]

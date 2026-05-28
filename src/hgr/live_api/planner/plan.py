"""Plan / Step data model for the iris planner.

Designed so a Plan is either built by the deterministic classifier (one step,
Phase 1) or produced as JSON by the cheap-LLM planner (multi-step, Phase 2)
and executed by the same executor either way.

Author: Konstantin Markov
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    """One unit of work in a Plan."""
    tool: str                                 # tool name (connector or built-in)
    args: Dict[str, Any] = field(default_factory=dict)
    id: int = 0                               # 0 for single-step plans
    layer: str = "connector"                  # 'touchless' | 'connector' | 'iris' | 'cheap-llm' | 'realtime'
    depends_on: List[int] = field(default_factory=list)
    needs_confirm: bool = False
    description: str = ""                     # short human-readable label


@dataclass
class StepResult:
    step_id: int
    tool: str
    status: str                               # 'ok' | 'error' | 'cancelled' | 'needs_confirmation'
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    cost_level: int = 0                       # 0..5 (cost_policy)


@dataclass
class Plan:
    """An ordered (with optional parallel) set of Steps for one request."""
    goal: str                                 # the raw / normalized user goal
    steps: List[Step] = field(default_factory=list)
    final: str = "return"                     # 'return' | 'synthesize' | 'speak'

    @classmethod
    def single(cls, goal: str, step: Step) -> "Plan":
        return cls(goal=goal, steps=[step], final="return")

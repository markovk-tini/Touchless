"""Iris planner — the JARVIS-level decision engine.

Picks the cheapest reliable path per request: deterministic-first routing
(0 tokens), then a deterministic classifier for obvious single intents
(connectors called directly — still 0 tokens), and in later phases a cheap-
LLM Plan-and-Execute path for multi-step / ambiguous requests. gpt-realtime
stays reserved for live voice conversation.

See docs/IRIS_PLANNER_DESIGN.md for the full architecture.

Phase 1 (this module): plan/step data model + a deterministic classifier
that maps common single-intent commands straight to a connector + args, so
"set volume to 30", "add a task: buy milk", "make a google doc titled Demo",
etc. never touch a model.

Author: Konstantin Markov
"""
from .plan import Step, StepResult, Plan
from .plan_cache import PlanCache
from .classifier import Classifier
from .executor import Executor
from .planner_llm import LLMPlanner
from .scheduler import RateScheduler, scheduler
from .synthesizer import Synthesizer
from .triggers import looks_multi_action, plan_needs_confirm, RISKY_TOOLS
from .orchestrator import IrisPlanner

__all__ = ["Step", "StepResult", "Plan", "PlanCache", "Classifier",
           "Executor", "LLMPlanner", "Synthesizer",
           "RateScheduler", "scheduler",
           "looks_multi_action", "plan_needs_confirm", "RISKY_TOOLS",
           "IrisPlanner"]

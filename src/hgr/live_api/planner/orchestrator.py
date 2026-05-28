"""Iris planner orchestrator (Phase 1) — try to handle a request entirely
without the model. If the deterministic Classifier identifies a single-intent
command, run the corresponding connector/built-in directly through the
ToolRegistry, emit the same tool_event + cost_policy log the rest of the app
uses, and return a short user-facing message. If nothing matches, return None
so the manager falls through to the LLM path as before.

Later phases will add: a cheap-LLM Plan-and-Execute path for multi-step /
ambiguous requests, a synthesis step, and a rate-aware scheduler.

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import os

from .classifier import Classifier
from .executor import Executor
from .plan import Plan, Step, StepResult
from .plan_cache import PlanCache
from .planner_llm import LLMPlanner, configured as llm_planner_configured
from .scheduler import scheduler
from .synthesizer import Synthesizer, configured as synth_configured
from .triggers import looks_multi_action, plan_needs_confirm


class IrisPlanner:
    """Public entrypoint for the planner. The manager calls `try_handle(text)`
    after Layer 0 (command_router) declines and BEFORE handing the request to
    the LLM. Returns a dict on success, None to fall through."""

    def __init__(self, registry: Any, logger: Any = None,
                 confirm: Optional[Callable[[str, str], bool]] = None) -> None:
        self._registry = registry
        self._logger = logger
        self._confirm = confirm
        self._classifier = Classifier()
        self._llm_planner = LLMPlanner(registry, logger) if registry is not None else None
        self._executor = Executor(registry, logger) if registry is not None else None
        self._synthesizer = Synthesizer(logger=logger)
        self._plan_cache = PlanCache()

    def try_handle(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to fully handle a request without the realtime model.

        Returns a uniform dict on success (single step OR multi-step plan):
            {steps: [Step], results: [StepResult], message: str, plan: Plan?}
        and None to fall through to the LLM. Both Phase 1 (classifier) and
        Phase 2 (LLM plan + executor) produce the same shape so the manager
        can iterate steps and emit per-step badges either way.
        """
        if self._registry is None:
            return None

        # --- Phase 1: deterministic classifier -> single connector step ---
        single = self._classifier.classify(text)
        if single is not None and self._registry.handles_connector(single.tool):
            if single.needs_confirm and self._confirm is not None and \
                    not self._confirm(f"Run {single.tool}?", single.description):
                return {
                    "steps": [single],
                    "results": [StepResult(step_id=0, tool=single.tool,
                                           status="cancelled",
                                           output={"status": "cancelled"})],
                    "message": "Cancelled."}
            try:
                out = self._registry.call(single.tool, single.args)
            except Exception as exc:
                if self._logger:
                    self._logger.exception("iris_planner_call_failed", exc, tool=single.tool)
                return None
            sr = StepResult(step_id=0, tool=single.tool,
                            status=str((out or {}).get("status") or "ok"),
                            output=out or {}, error=(out or {}).get("error"))
            return {"steps": [single], "results": [sr],
                    "message": self._format_message(single, out or {})}

        # --- Phase 2: cheap-LLM JSON plan -> Executor ---
        # Fires when ANY of:
        #   (a) explicit opt-in flag (TOUCHLESS_IRIS_PLAN_LLM=1)
        #   (b) the request looks multi-step (heuristic)
        #   (c) realtime is rate-limited and cheap-LLM is healthy
        # — and OPENAI_API_KEY is available. Phase 2 stays disabled when none
        # of those apply, so single-step questions never burn a planner call.
        flag_on = os.environ.get("TOUCHLESS_IRIS_PLAN_LLM", "0") == "1"
        sched = scheduler()
        scheduler_prefers_cheap = sched.prefer_cheap_planner()
        heuristic_open = looks_multi_action(text)
        if ((flag_on or heuristic_open or scheduler_prefers_cheap)
                and llm_planner_configured()
                and self._llm_planner is not None
                and self._executor is not None):
            # Cache: same goal twice in ~10 min reuses the prior Plan and
            # skips the LLM call entirely.
            plan = self._plan_cache.get(text)
            if plan is None:
                plan = self._llm_planner.plan(text)
                self._plan_cache.put(text, plan)
            if plan is not None and plan.steps:
                # Whole-plan confirm-gate: if any step is "risky" (sends email,
                # uploads files, types into focused window, etc.), surface one
                # summary dialog before running.
                step_tools = [s.tool for s in plan.steps]
                if (plan_needs_confirm(step_tools)
                        and self._confirm is not None
                        and not self._confirm("Run this plan?",
                                              self._plan_confirm_summary(plan))):
                    return {
                        "steps": plan.steps,
                        "results": [
                            StepResult(step_id=s.id, tool=s.tool,
                                       status="cancelled",
                                       output={"status": "cancelled"})
                            for s in plan.steps
                        ],
                        "plan": plan,
                        "message": "Cancelled.",
                    }
                results = self._executor.run(plan)
                message = self._summarize(plan, results)
                return {
                    "steps": plan.steps,
                    "results": results,
                    "plan": plan,
                    "message": message,
                }
        return None

    @staticmethod
    def _plan_confirm_summary(plan: "Plan") -> str:
        """Compact human-readable summary of a Plan for the confirm dialog."""
        bullets = []
        for s in plan.steps:
            desc = (s.description or s.tool).strip()
            bullets.append(f"  • {desc}")
        return f"Goal: {plan.goal}\nSteps:\n" + "\n".join(bullets)

    # ---- summarize a multi-step plan --------------------------------------
    def _summarize(self, plan: "Plan", results: list) -> str:
        """Pick the right summary path: synthesizer when the plan asks for
        it AND cheap-LLM is healthy, deterministic format otherwise."""
        wants_synth = (plan.final or "").lower() in ("synthesize", "speak")
        if (wants_synth
                and synth_configured()
                and scheduler().allow_cheap_synthesis()):
            text = self._synthesizer.summarize(plan, results)
            if text:
                return text
        return self._format_plan_message(plan, results)

    def _format_plan_message(self, plan: "Plan", results: list) -> str:
        ok = sum(1 for r in results if r.status == "ok")
        if ok == len(results) and results:
            # If a "useful" final result exists, surface its message.
            last = results[-1].output if results else {}
            if isinstance(last, dict):
                for key in ("link", "message", "result", "summary"):
                    val = last.get(key)
                    if val:
                        return str(val)[:600]
            tools = ", ".join(r.tool for r in results)
            return f"Done ({ok}/{len(results)} steps: {tools})."
        errs = [r for r in results if r.status != "ok"]
        first = errs[0] if errs else None
        return f"Plan ran {ok}/{len(results)} steps" + (
            f"; {first.tool} failed: {first.error}" if first else ".")

    # ---- friendly result text ----
    @staticmethod
    def _format_message(step: Step, result: Dict[str, Any]) -> str:
        if (result.get("status") or "").lower() != "ok":
            err = result.get("error") or "error"
            return f"Couldn't run {step.tool}: {err}"
        tool = step.tool
        args = step.args or {}
        if tool == "volume_set":
            return f"Volume set to {args.get('percent')}%."
        if tool == "volume_get":
            pct = result.get("percent")
            muted = result.get("muted")
            return f"Volume is {pct}%{' (muted)' if muted else ''}."
        if tool in ("volume_mute", "volume_toggle_mute"):
            m = result.get("muted")
            if m is True:
                return "Muted."
            if m is False:
                return "Unmuted."
            return "Toggled mute."
        if tool == "discord_mute":
            return "Discord muted." if args.get("muted") else "Discord unmuted."
        if tool == "discord_toggle_mute":
            return "Toggled Discord mute."
        if tool == "discord_deafen":
            return "Discord deafened." if args.get("deafened") else "Discord undeafened."
        if tool == "discord_toggle_deafen":
            return "Toggled Discord deafen."
        if tool == "todo_add":
            return f"Added task: {args.get('title')}."
        if tool == "gdocs_create":
            link = result.get("link")
            return f"Created Google Doc \"{args.get('title')}\"" + (f": {link}" if link else ".")
        if tool == "sheets_create":
            link = result.get("link")
            return f"Created Google Sheet \"{args.get('title')}\"" + (f": {link}" if link else ".")
        if tool == "slides_create":
            link = result.get("link")
            return f"Created Slides \"{args.get('title')}\"" + (f": {link}" if link else ".")
        if tool == "drive_upload":
            link = result.get("link")
            return f"Uploaded to Drive" + (f": {link}" if link else ".")
        if tool == "outlook_compose":
            return f"Drafted email to {args.get('recipient')}."
        return f"Done ({tool})."

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
from .skills import SkillStore
from .synthesizer import Synthesizer, configured as synth_configured
from .triggers import looks_multi_action, plan_needs_confirm


class IrisPlanner:
    """Public entrypoint for the planner. The manager calls `try_handle(text)`
    after Layer 0 (command_router) declines and BEFORE handing the request to
    the LLM. Returns a dict on success, None to fall through."""

    def __init__(self, registry: Any, logger: Any = None,
                 confirm: Optional[Callable[[str, str], bool]] = None,
                 memory: Any = None) -> None:
        self._registry = registry
        self._logger = logger
        self._confirm = confirm
        self._classifier = Classifier()
        self._llm_planner = LLMPlanner(registry, logger) if registry is not None else None
        self._executor = Executor(registry, logger) if registry is not None else None
        self._synthesizer = Synthesizer(logger=logger)
        self._plan_cache = PlanCache()
        # Memory is optional — when None, recall/record are no-ops. The
        # manager wires a real MemoryManager when available; tests can
        # leave it unset to keep them offline.
        self._memory = memory
        # Skills catalog (Tier 0.5: user-saved Plans replayed without an
        # LLM call). Lazily opened; failures just disable the tier.
        try:
            self._skills: Optional[SkillStore] = SkillStore()
        except Exception:
            self._skills = None

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

        # --- Tier 0.5: user-defined skills (replay a saved Plan, 0 tokens) ---
        if self._skills is not None and self._executor is not None:
            try:
                skill_plan = self._skills.find(text)
            except Exception as exc:
                if self._logger:
                    self._logger.exception("skill_find_failed", exc)
                skill_plan = None
            if skill_plan is not None and skill_plan.steps:
                # Same confirm-gate behaviour as Phase 2.
                step_tools = [s.tool for s in skill_plan.steps]
                if (plan_needs_confirm(step_tools)
                        and self._confirm is not None
                        and not self._confirm("Run this skill?",
                                              self._plan_confirm_summary(skill_plan))):
                    return {
                        "steps": skill_plan.steps,
                        "results": [
                            StepResult(step_id=s.id, tool=s.tool,
                                       status="cancelled",
                                       output={"status": "cancelled"})
                            for s in skill_plan.steps
                        ],
                        "plan": skill_plan,
                        "message": "Cancelled.",
                    }
                results = self._executor.run(skill_plan)
                message = self._summarize(skill_plan, results)
                try:
                    self._skills.mark_used(skill_plan.goal)
                except Exception:
                    pass
                self._record_turn(text, skill_plan, skill_plan.steps,
                                  results, message)
                return {
                    "steps": skill_plan.steps,
                    "results": results,
                    "plan": skill_plan,
                    "message": message,
                }

        # --- Phase 1: deterministic classifier -> single connector step ---
        # IMPORTANT: when the input looks multi-action ("set volume to 30 AND
        # email dani"), the classifier would silently swallow only the first
        # intent. Skip Phase 1 in that case so the request can be decomposed
        # properly by Phase 2 (or fall through to realtime).
        multi = looks_multi_action(text)
        # Always run the classifier so high-confidence pseudo-tool patterns
        # (lookup / preference-set) can fire even when looks_multi_action
        # triggers — their regexes are very specific so false positives are
        # rare, and they answer cheaper than Tier 2 ever could.
        single = self._classifier.classify(text)
        _BYPASS_MULTI = {"iris_lookup_contact", "iris_set_preference",
                         "iris_remember_contact"}
        if multi and single is not None and single.tool not in _BYPASS_MULTI:
            single = None

        # --- Pseudo-tool: contact remember. Writes one or many
        # (person, name, email) facts straight to memory so subsequent
        # 'send Vesko hi' / 'what's Vesko's email' resolve cleanly.
        if single is not None and single.tool == "iris_remember_contact":
            names = list((single.args or {}).get("names") or [])
            email = str((single.args or {}).get("email") or "").strip()
            if self._memory is not None and email and names:
                for n in names:
                    try:
                        self._memory.set_fact("person", n, email)
                    except Exception as exc:  # pragma: no cover
                        if self._logger:
                            self._logger.exception("memory_set_fact_failed", exc)
            sr = StepResult(step_id=0, tool=single.tool, status="ok",
                            output={"status": "ok", "names": names,
                                    "email": email})
            if len(names) == 1:
                message = f"Got it — {names[0]} = {email}."
            else:
                message = (f"Got it — {', '.join(names[:-1])} and "
                           f"{names[-1]} all share {email}.")
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr], "message": message}

        # --- Pseudo-tool: contact lookup. Reads person/<name> from memory
        # and returns the email as the user-facing message.
        if single is not None and single.tool == "iris_lookup_contact":
            name = str((single.args or {}).get("name") or "").strip()
            email: Optional[str] = None
            resolved_name = name
            if self._memory is not None and name:
                # Try the exact captured name first; if not found, try
                # stripping a trailing 's' (handles "Veskos email" capture
                # that should match the stored "vesko" fact).
                candidates = [name]
                if len(name) > 3 and name.lower().endswith("s"):
                    candidates.append(name[:-1])
                for cand in candidates:
                    try:
                        facts = self._memory._store.find_facts(  # type: ignore[attr-defined]
                            kind="person", key=cand.lower())
                    except Exception:
                        facts = []
                    if facts:
                        email = facts[0].value
                        resolved_name = cand
                        break
            sr = StepResult(
                step_id=0, tool=single.tool,
                status="ok" if email else "not_found",
                output={"name": resolved_name, "email": email,
                        "status": "ok" if email else "not_found"})
            if email:
                message = f"{resolved_name}'s email is {email}."
            else:
                message = (f"I don't have {name}'s email in memory yet. "
                           f"Once you email them once, I'll remember.")
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr], "message": message}

        # --- Pseudo-tool: preference-setting commands. Not a registry tool —
        # writes directly to memory so the planner's recall layer injects the
        # preference into every future Phase-2 prompt.
        if single is not None and single.tool == "iris_set_preference":
            args = single.args or {}
            kind = str(args.get("kind") or "preference")
            key = str(args.get("key") or "")
            value = str(args.get("value") or "")
            if self._memory is not None and key and value:
                self._memory.set_fact(kind, key, value)
            sr = StepResult(step_id=0, tool=single.tool,
                            status="ok",
                            output={"status": "ok", "kind": kind,
                                    "key": key, "value": value})
            message = f"Got it — I'll remember {key} = {value}."
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr], "message": message}

        # Email-compose rewrite: respect default_send_via preference and
        # resolve recipient names via memory. Turns "email Dani saying hi"
        # into a direct gmail_send (or ms_mail_send) call when the user
        # has expressed a preference, instead of always opening the
        # Outlook draft window.
        if single is not None and single.tool == "outlook_compose":
            single = self._maybe_rewrite_compose(single)

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
            message = self._format_message(single, out or {})
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr], "message": message}

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
        if ((flag_on or multi or scheduler_prefers_cheap)
                and llm_planner_configured()
                and self._llm_planner is not None
                and self._executor is not None):
            # Cache: same goal twice in ~10 min reuses the prior Plan and
            # skips the LLM call entirely.
            plan = self._plan_cache.get(text)
            if plan is None:
                memory_ctx = self._recall_context(text)
                plan = self._llm_planner.plan(text, memory_context=memory_ctx)
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
                self._record_turn(text, plan, plan.steps, results, message)
                return {
                    "steps": plan.steps,
                    "results": results,
                    "plan": plan,
                    "message": message,
                }
        return None

    # ---- preference-aware compose rewrite --------------------------------
    def _maybe_rewrite_compose(self, step: "Step") -> "Step":
        """Apply two memory-driven transforms to an outlook_compose Step:

        1. Resolve recipient name → email via memory. If the user said
           'email Dani saying hi', recipient='Dani' isn't a valid address;
           look up person/dani in memory and substitute the email.
        2. Respect default_send_via preference. If memory says the user
           prefers gmail_send / ms_mail_send, rewrite the tool + adapt
           the args shape (those tools need {to, subject, body} vs
           outlook_compose's {recipient, body}). Default subject is the
           first line of the body or 'Hello'.

        Best-effort: failures fall through and the original outlook_compose
        Step runs unchanged."""
        if self._memory is None:
            return step
        args = dict(step.args or {})
        recipient = str(args.get("recipient", "")).strip()

        # 1. Name → email lookup if recipient isn't already an address.
        if "@" not in recipient and recipient:
            try:
                facts = self._memory._store.find_facts(  # type: ignore[attr-defined]
                    kind="person", key=recipient.lower())
            except Exception:
                facts = []
            if facts:
                recipient = facts[0].value
                args["recipient"] = recipient
                if self._logger:
                    self._logger.event("compose_recipient_resolved",
                                       name=step.args.get("recipient"),
                                       email=recipient)

        # If we still don't have a valid email, leave the step alone — the
        # connector will fail with a clear error rather than us silently
        # producing the wrong thing.
        if "@" not in recipient:
            return step

        # 2. Sender preference.
        try:
            prefs = self._memory._store.find_facts(  # type: ignore[attr-defined]
                kind="preference", key="default_send_via")
        except Exception:
            prefs = []
        if not prefs:
            args["recipient"] = recipient
            return Step(tool=step.tool, args=args, layer=step.layer,
                        description=step.description,
                        needs_confirm=step.needs_confirm)
        preferred = prefs[0].value
        if preferred not in ("gmail_send", "ms_mail_send"):
            args["recipient"] = recipient
            return Step(tool=step.tool, args=args, layer=step.layer,
                        description=step.description,
                        needs_confirm=step.needs_confirm)

        # Rewrite to the API send tool. Adapt args shape — outlook_compose
        # uses {recipient, body} but gmail_send / ms_mail_send require
        # {to, subject, body}. Default subject = first line of body, or
        # "Hello" if the body is empty.
        body = str(args.get("body") or "").strip()
        subject_default = body.split("\n", 1)[0][:60] if body else "Hello"
        new_args = {
            "to": recipient,
            "subject": str(args.get("subject") or subject_default),
            "body": body,
        }
        if self._logger:
            self._logger.event("compose_rewritten_to_api_send",
                               from_tool=step.tool, to_tool=preferred,
                               recipient=recipient)
        return Step(tool=preferred, args=new_args, layer="connector",
                    description=f"send email via {preferred}")

    # ---- memory bridge ----------------------------------------------------
    def _recall_context(self, text: str) -> str:
        """Pull a tiny memory-context block for the planner prompt. Empty
        string when no memory or nothing relevant."""
        if self._memory is None:
            return ""
        try:
            recall = self._memory.recall(text, k=3)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_recall_failed", exc)
            return ""
        return recall.get("context") or ""

    def _record_turn(self, user_text: str, plan: Any,
                     steps: list, results: list, message: str) -> None:
        """Persist this turn into memory. Best-effort; failures never
        bubble up to the user."""
        if self._memory is None:
            return
        try:
            self._memory.record(user_text, plan, steps, results, message)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_record_failed", exc)

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

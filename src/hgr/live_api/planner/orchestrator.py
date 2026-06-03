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


_ARTIFACT_KIND_BY_TOOL = {
    "gdocs_create": "doc",
    "gdocs_append_text": "doc",
    "sheets_create": "sheet",
    "sheets_append_rows": "sheet",
    "slides_create": "slideshow",
    "slides_add_slide": "slideshow",
    "onenote_create": "OneNote page",
    "onenote_append_text": "OneNote page",
    "drive_upload": "file",
    "onedrive_upload": "file",
    "excel_create": "Excel workbook",
}


def _artifact_kind_from_tool(tool: str) -> str:
    """Friendly name for the 'kind' of thing a tool created — used in the
    'Opening the doc \"X\"' reply when the user says 'open it'."""
    return _ARTIFACT_KIND_BY_TOOL.get(tool, "page")


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
        # Tracks the most recent artifact created in this session so
        # 'open it' / 'show me that' (Tier 1 iris_open_last) can resolve
        # without realtime's bad pronoun resolution. {kind, title, link}.
        self._last_artifact: Optional[Dict[str, Any]] = None
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
                         "iris_remember_contact", "iris_forget_contact",
                         "iris_open_last", "iris_setup_tool"}
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

        # --- Pseudo-tool: forget a contact (delete person/<name> from memory).
        if single is not None and single.tool == "iris_forget_contact":
            name = str((single.args or {}).get("name") or "").strip()
            removed = 0
            if self._memory is not None and name:
                try:
                    removed = self._memory.forget_fact(
                        kind="person", key=name.lower())
                    # Also try the trailing-s variant ("Veskos") so users
                    # don't have to know the exact stored form.
                    if removed == 0 and len(name) > 3 and name.lower().endswith("s"):
                        removed = self._memory.forget_fact(
                            kind="person", key=name[:-1].lower())
                except Exception as exc:
                    if self._logger:
                        self._logger.exception("forget_fact_failed", exc)
            sr = StepResult(step_id=0, tool=single.tool,
                            status="ok" if removed else "not_found",
                            output={"status": "ok" if removed else "not_found",
                                    "name": name, "removed": removed})
            if removed:
                message = (f"Forgotten — removed {removed} fact"
                           f"{'s' if removed != 1 else ''} about {name}.")
            else:
                message = (f"I didn't have anything stored for {name}, "
                           f"so nothing to forget.")
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr], "message": message}

        # --- Pseudo-tool: self-setup. 'set up kicad' / 'install spotify'
        # / 'connect notion'. Looks up the connector by id substring and
        # calls its setup_self() — auto-discovers binaries on PATH, walks
        # OAuth flows, etc. Connectors without setup_self() just report
        # availability.
        if single is not None and single.tool == "iris_setup_tool":
            name = str((single.args or {}).get("name") or "").strip()
            path_arg = str((single.args or {}).get("path") or "").strip()
            connector = (self._registry.find_connector(name)
                         if hasattr(self._registry, "find_connector") else None)
            if connector is None:
                sr = StepResult(step_id=0, tool=single.tool,
                                status="not_found",
                                output={"status": "not_found", "name": name})
                message = (f"I don't have a connector named '{name}'. "
                           f"Common ones I can set up: kicad. (Others come "
                           f"pre-wired and just need auth.)")
                self._record_turn(text, None, [single], [sr], message)
                return {"steps": [single], "results": [sr], "message": message}
            setup = getattr(connector, "setup_self", None)
            if callable(setup):
                try:
                    result = setup(path=path_arg) if path_arg else setup()
                except Exception as exc:
                    if self._logger:
                        self._logger.exception("iris_setup_tool_failed", exc)
                    result = {"ok": False,
                              "error": f"{type(exc).__name__}: {exc}"}
            else:
                # No custom setup; just check whether it's already available.
                try:
                    ok = bool(connector.available())
                except Exception:
                    ok = False
                result = {"ok": ok,
                          "error": ("" if ok else
                                    f"{name} has no automated setup. "
                                    "Likely needs OAuth from the UI.")}
            status = "ok" if result.get("ok") else "error"
            sr = StepResult(step_id=0, tool=single.tool,
                            status=status,
                            output={"status": status, "name": name, **result})
            if status == "ok":
                version = (str(result.get("version") or "")[:80]
                           if result.get("version") else "")
                where = result.get("cli_path") or result.get("path") or ""
                message = (f"Set up — {name} ready"
                           + (f" ({version})" if version else "")
                           + (f" at {where}" if where else "")
                           + ". Tools added to the next session.")
            else:
                message = f"Couldn't set up {name}: {result.get('error') or 'unknown error'}"
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr], "message": message}

        # --- Pseudo-tool: open the last-created artifact ('open it',
        # 'show me that', etc.). Resolves to whatever artifact (doc /
        # sheet / slide / OneNote / etc.) the planner most recently
        # produced in this session. Without this, 'open it' falls to
        # realtime which confabulates random plans.
        if single is not None and single.tool == "iris_open_last":
            last = self._last_artifact
            if not last or not last.get("link"):
                sr = StepResult(step_id=0, tool=single.tool,
                                status="not_found",
                                output={"status": "not_found"})
                message = ("I don't have anything recent to open yet. "
                           "Make a doc, sheet, slide, OneNote page, etc., "
                           "and I'll be able to open it.")
                self._record_turn(text, None, [single], [sr], message)
                return {"steps": [single], "results": [sr], "message": message}
            link = str(last.get("link") or "")
            title = str(last.get("title") or "the last one")
            # Use the iris open_url tool to launch it in the browser.
            try:
                out = self._registry.call("open_url", {"url_or_query": link})
            except Exception as exc:
                if self._logger:
                    self._logger.exception("iris_open_last_failed", exc)
                out = {"status": "error",
                       "error": f"{type(exc).__name__}: {exc}"}
            status = str((out or {}).get("status") or "ok")
            sr = StepResult(step_id=0, tool=single.tool,
                            status=status,
                            output={"link": link, "title": title,
                                    "status": status,
                                    "kind": last.get("kind")})
            kind = last.get("kind") or "page"
            message = (f"Opening the {kind} \"{title}\"."
                       if status == "ok"
                       else f"Couldn't open it: {out.get('error') if out else 'error'}")
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
            base_message = self._format_message(single, out or {})
            # Jarvis prose pass: rewrite the deterministic message as
            # conversational Iris-voice prose, with fact-preservation
            # guard. Falls back to base_message on any failure. The
            # renderer auto-skips very short status confirmations
            # ('Volume set to 30%.') so it never adds latency for
            # trivial replies.
            try:
                from ..prose_renderer import render_jarvis
                message = render_jarvis(
                    question=text,
                    tool_name=single.tool,
                    tool_result=out or {},
                    fallback=base_message,
                    context="",  # Layer-1 path doesn't carry convo buffer
                )
            except Exception:
                message = base_message
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr], "message": message}

        # Clarifying follow-up guard: short questions like 'and on my other
        # screen?' / 'what about that?' / 'how about now?' don't carry enough
        # signal for Tier 2 to plan against, but the planner has been
        # confabulating Discord/etc. plans from them. Route to realtime
        # which has the actual conversation context to resolve the pronoun
        # references.
        if self._is_clarifying_followup(text):
            if self._logger:
                self._logger.event("planner_skip_clarifying_followup",
                                   text_len=len(text))
            return None

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
        """Persist this turn into memory + update the 'last artifact'
        pointer so 'open it' / 'show me that' work. Best-effort;
        failures never bubble up to the user."""
        # Walk results for any link — the LAST one wins (the planner
        # tends to create-then-fill, so the final artifact is the one
        # the user means).
        for sr in results or []:
            out = getattr(sr, "output", None)
            if not isinstance(out, dict):
                continue
            link = out.get("link")
            if not link:
                continue
            tool = getattr(sr, "tool", "") or ""
            kind = _artifact_kind_from_tool(tool)
            title = out.get("title") or out.get("name") or kind
            self._last_artifact = {
                "kind": kind, "title": title, "link": link, "tool": tool,
            }
        if self._memory is None:
            return
        try:
            self._memory.record(user_text, plan, steps, results, message)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_record_failed", exc)

    @staticmethod
    def _is_clarifying_followup(text: str) -> bool:
        """True when the text reads like a short clarifying follow-up
        question that needs prior conversation context to make sense.
        Sending these to Tier 2 produces confabulated plans (the LLM
        invents tools to call from nothing); realtime can resolve them
        from session memory instead.

        Conservative: only catches the obvious cases. Long requests with
        clear actions pass through unaffected."""
        t = (text or "").strip()
        if not t or len(t) > 40:
            return False
        lower = t.lower()
        # Common continuation/clarification openers — almost always
        # reference the prior turn.
        _OPENERS = (
            "and ", "or ", "but ", "so ",
            "what about", "how about", "what if", "and what", "and how",
            "and on", "and in", "and the", "and is", "and are",
            "what else", "anything else", "any other",
        )
        if lower.startswith(_OPENERS):
            return True
        return False

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
            last_result = results[-1]
            last_tool = getattr(last_result, "tool", "") or ""
            last = last_result.output if isinstance(last_result.output, dict) else {}
            # Tool-specific surface FIRST — for connectors whose useful output
            # isn't in one of the generic keys below (e.g. notion_search has a
            # results list, ollama_list_models has a models list). Returns None
            # to fall through to the generic scan.
            specific = self._surface_connector_result(last_tool, last)
            if specific is not None:
                return specific
            # Generic key scan. `text` covers LLM-generated and read-back
            # content (ollama_generate, notion_read_page, etc.) — without it,
            # the model's actual answer got swallowed into "Done (1/1 steps:
            # ollama_generate)" which is what made testing feel broken.
            if isinstance(last, dict):
                for key in ("link", "message", "result", "summary", "text"):
                    val = last.get(key)
                    if val:
                        return str(val)[:2000]
            tools = ", ".join(r.tool for r in results)
            return f"Done ({ok}/{len(results)} steps: {tools})."
        errs = [r for r in results if r.status != "ok"]
        first = errs[0] if errs else None
        return f"Plan ran {ok}/{len(results)} steps" + (
            f"; {first.tool} failed: {first.error}" if first else ".")

    @staticmethod
    def _surface_connector_result(tool: str,
                                  r: Dict[str, Any]) -> Optional[str]:
        """Format a single-tool plan result for tools whose useful output
        isn't a single text field. Returns None to let the caller fall
        through to the generic key scan (link/message/result/summary/text)."""
        if not isinstance(r, dict) or not tool:
            return None
        # Notion: a search returns a hit list + count, not a "message" string.
        if tool == "notion_search":
            hits = r.get("results") or []
            count = int(r.get("count") or 0)
            if not hits:
                return ("Nothing in your Notion workspace matched. If you "
                        "expected a hit, check the page is shared with the "
                        "Iris integration (Share -> Add connections).")
            lines = []
            for h in hits[:8]:
                title = (h.get("title") or "(untitled)").strip()
                kind = (h.get("object") or "").strip()
                lines.append(f"- {title}" + (f"  [{kind}]" if kind else ""))
            more = "" if count <= 8 else f"\n...and {count - 8} more."
            return (f"Found {count} result"
                    f"{'s' if count != 1 else ''}:\n"
                    + "\n".join(lines) + more)
        if tool == "notion_append_to_page":
            n = int(r.get("blocks_appended") or 0)
            return (f"Appended {n} block{'s' if n != 1 else ''} "
                    "to the page.")
        if tool == "notion_create_page":
            url = (r.get("url") or "").strip()
            return (f"Created the page: {url}" if url
                    else "Created the page.")
        if tool == "notion_add_to_database":
            url = (r.get("url") or "").strip()
            return (f"Added the row to your database: {url}" if url
                    else "Added the row.")
        # Mail send results — show which account ACTUALLY sent + the
        # recipient + a verify link when ms_mail_send confirmed via Sent
        # folder lookup. Without this the user gets a generic 'Done.' and
        # has to guess why no message appeared in their Sent folder.
        if tool in ("ms_mail_send", "gmail_send"):
            to = (r.get("to") or "").strip()
            from_acct = (r.get("from_account") or "").strip()
            sender = (r.get("sender_address") or "").strip()
            web_link = (r.get("web_link") or "").strip()
            sent = r.get("sent")
            if sent is True or (sent is None and r.get("status") == "ok"):
                base = f"Sent to {to or 'recipient'}"
                # Prefer the actual SMTP sender (from Graph) over the
                # login account name — clearer when the login is e.g.
                # a gmail address that aliases to an outlook mailbox.
                shown_from = sender or from_acct
                if shown_from:
                    base += f" (from {shown_from}"
                    if sender and from_acct and sender != from_acct:
                        base += f", login {from_acct}"
                    base += ")"
                base += "."
                if web_link:
                    base += f"\nVerify: {web_link}"
                return base
            err = (r.get("error") or "").strip()
            return f"Couldn't send: {err or 'unknown error'}"
        # Ollama list: friendly inventory.
        if tool == "ollama_list_models":
            models = r.get("models") or []
            default = (r.get("default") or "").strip()
            if not models:
                return "No Ollama models installed yet."
            listing = ", ".join(models)
            return (f"Installed Ollama models: {listing}."
                    + (f" Default for quick tasks: {default}." if default
                       else ""))
        return None

    # ---- email list formatter (deterministic, never LLM-rephrased) ----
    @staticmethod
    def _format_email_list(result: Dict[str, Any],
                           args: Dict[str, Any]) -> str:
        """Render gmail_list / ms_mail_list output as a casual but
        FAITHFUL summary. No LLM in the loop — every sender, subject,
        and snippet comes straight from the tool response. The model
        was caught fabricating demo emails when given freedom to
        'write the reply itself', so this path bypasses it entirely."""
        msgs = result.get("messages") or []
        count = int(result.get("count") or len(msgs))
        unread_only = bool(args.get("unread_only"))
        kind = "unread email" if unread_only else "email"
        if count == 0:
            return (f"You're all caught up — no {kind}s in your inbox."
                    if unread_only else f"No {kind}s in your inbox.")
        # Headline. Be honest about truncation.
        max_n = int(args.get("max") or 0)
        truncated = max_n and len(msgs) < count
        if count == 1:
            head = f"You've got one {kind}:"
        else:
            head = f"You've got {count} {kind}s — here they are:"
            if truncated:
                head = (f"You've got {count} {kind}s; here are the "
                        f"first {len(msgs)}:")
        lines = [head]
        for i, m in enumerate(msgs, start=1):
            sender = (str(m.get("from_name") or "").strip()
                      or str(m.get("from") or "").strip()
                      or "unknown sender")
            subject = str(m.get("subject") or "").strip() or "(no subject)"
            snippet = str(m.get("snippet") or m.get("body_text") or "").strip()
            # Trim snippet to roughly one sentence so the spoken
            # version doesn't drone on for 30 emails.
            if snippet:
                snippet = snippet.replace("\r", " ").replace("\n", " ")
                snippet = " ".join(snippet.split())
                if len(snippet) > 140:
                    snippet = snippet[:137].rstrip() + "..."
                lines.append(f"{i}. {sender} — \"{subject}\". {snippet}")
            else:
                lines.append(f"{i}. {sender} — \"{subject}\".")
        lines.append("Want me to open any of them or dig deeper?")
        return "\n".join(lines)

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
        if tool == "weather_get":
            # weather.py already pre-renders the human-readable summary.
            return str(result.get("summary") or "").strip() or f"Done ({tool})."
        if tool in ("gmail_list", "ms_mail_list", "email_summary"):
            # Format the reply DETERMINISTICALLY from the actual tool
            # result so the LLM never gets a chance to hallucinate
            # sender names, subjects, or counts. The model has been
            # caught inventing canonical-looking demo emails (Carl,
            # Sarah, Mark, etc.) instead of reading the real array.
            # If the connector pre-rendered a summary (cascading
            # email_summary, gmail_list with summary), prefer that
            # — it carries source-specific context like "no unread in
            # your Gmail account; check Outlook".
            pre = str(result.get("summary") or "").strip()
            if pre:
                return pre
            return Orchestrator._format_email_list(result, args)
        if tool == "ollama_generate":
            # The generated text IS the user-facing response. Without this,
            # the haiku/poem/regex/etc. would get hidden behind
            # "Done (ollama_generate)." and the call would feel broken.
            text = (str(result.get("text") or result.get("response")
                        or result.get("output") or "")).strip()
            return text[:4000] if text else f"Done ({tool})."
        return f"Done ({tool})."

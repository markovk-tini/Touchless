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

from typing import Any, Callable, Dict, List, Optional

import os
import re
import time

from .classifier import Classifier
from .executor import Executor
from .intent_extractor import IntentExtractor
from .plan import Plan, Step, StepResult
from .plan_cache import PlanCache
from .planner_llm import LLMPlanner, configured as llm_planner_configured
from .scheduler import scheduler
from .skills import SkillStore
from .synthesizer import Synthesizer, configured as synth_configured
from .triggers import looks_multi_action, plan_needs_confirm


def _peek_manager():
    """Best-effort lookup of the most-recently-constructed
    LiveApiManager. Used by the skill consolidator wiring so the
    orchestrator can notify the consolidator without holding a
    direct reference (manager constructs orchestrator, not vice
    versa)."""
    try:
        import sys as _sys
        from .. import live_api_manager  # type: ignore
        return getattr(live_api_manager, "_last_constructed", None)
    except Exception:
        return None


_DICTATION_REFS = (
    "just dictated", "i just typed", "what i typed", "what i wrote",
    "what i just dictated", "the paragraph", "this paragraph",
    "this draft", "what i just said", "summarize what i", "reword that",
    "rephrase that", "fix what i", "rewrite what i", "clean that up",
    "tighten that", "polish that",
)


def _wants_recent_dictation(text: str) -> bool:
    """True when the user's request looks like a reference to text
    they just dictated. Cheap substring scan."""
    t = (text or "").lower()
    return any(p in t for p in _DICTATION_REFS)


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
    # Tasks — without these, 'add a task called X' recorded the artifact
    # with kind='page' (the fallback), which then made
    # find_artifact_by_name(name, kind='task') invisible AND caused
    # 'delete X' to silently pick whichever artifact happened to be
    # newest when a sheet and a task shared a name.
    "tasks_add": "task",
    "tasks_complete": "task",
    "tasks_delete": "task",
    # Calendar events — same class of bug: the create-tool result
    # should be addressable as kind='calendar event' later.
    "calendar_create_event": "calendar event",
    "ms_calendar_create": "calendar event",
    "outlook_com_create_event": "calendar event",
    # Contacts.
    "contacts_create": "contact",
    # Email sends — the produced artifact is the sent message.
    "gmail_send": "email",
    "ms_mail_send": "email",
    "email_send": "email",
}


# Delete-shaped tools whose only argument is a NAME (not an id / range),
# so 'delete scratch-notes' is ambiguous when multiple artifact kinds
# share that name. The backstop in _dispatch_connector_step consults
# find_all_artifacts_by_name for tools in this set BEFORE dispatching
# and refuses the delete when 2+ kinds match. Extend as new generic
# name-based deletes are added.
_DELETE_TOOLS_NEEDING_DISAMBIG = {
    "tasks_delete",
    "iris_forget_contact",
    "drive_trash",
}


# Arg keys that carry a plain-name target on a delete-shaped tool.
# Order matters: 'title_match' wins over 'name' when both are present
# because the tasks connector prefers the former. Note: keys that
# unambiguously identify a specific object (task_id, contact_id,
# file_id, sheet_id, range) are DELIBERATELY not listed — their
# presence means the tool call is already unambiguous and the
# backstop should not fire.
_DELETE_NAME_ARG_KEYS = (
    "title_match", "title", "name", "sheet_name", "query", "match",
)


def _artifact_kind_from_tool(tool: str) -> str:
    """Friendly name for the 'kind' of thing a tool created — used in the
    'Opening the doc \"X\"' reply when the user says 'open it'."""
    return _ARTIFACT_KIND_BY_TOOL.get(tool, "page")


def _extract_delete_name_arg(args: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the first plain-name arg on a delete-shaped tool call, or
    None when the args instead identify the target by id / range (in
    which case the ambiguity backstop should NOT fire — the caller
    already picked a specific object)."""
    if not isinstance(args, dict) or not args:
        return None
    # If the args already carry an explicit id / range, skip: this
    # call is unambiguous and doesn't need name-based disambiguation.
    for id_key in ("task_id", "contact_id", "file_id", "id",
                   "sheet_id", "range"):
        v = args.get(id_key)
        if isinstance(v, str) and v.strip():
            return None
    for k in _DELETE_NAME_ARG_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


# Track the most recently constructed IrisPlanner so connectors can
# fuzzy-match artifact names (e.g. sheets_update_range resolving
# 'Q4 plan' to the spreadsheet_id of the sheet just created this
# session) without holding a direct reference to the planner.
_LAST_PLANNER: Any = None


def current_planner_artifact_lookup() -> Optional[Callable[..., Optional[Dict[str, Any]]]]:
    """Return a callable `(name, kind=None) -> artifact dict | None` bound
    to the most recently constructed IrisPlanner. Returns None when no
    planner has been built yet (e.g. unit tests). Used by connectors that
    need name→id resolution without a hard import cycle."""
    p = _LAST_PLANNER
    if p is None:
        return None
    fn = getattr(p, "find_artifact_by_name", None)
    return fn if callable(fn) else None


# ---------- Tier-1 ambiguity gate ------------------------------------
# Constants + regex used by IrisPlanner._utterance_has_contradictory_signals
# to detect when the deterministic classifier's pick contradicts strong
# opposing signals in the raw utterance (cell refs, sheet context words,
# cell-edit verbs). When triggered, try_handle consults the intent
# extractor as a peer reviewer BEFORE dispatching the classifier's step.

# Words that indicate the utterance is about a spreadsheet — Google
# Sheets, Excel, etc. Lowercase, substring match against the lowered
# utterance.
_SHEET_CONTEXT_WORDS = (
    "google sheet", "google sheets", "the sheet", "spreadsheet",
    "in the sheet", "the tab called", "sheet called", "workbook",
    "excel sheet", "excel file",
)

# Verbs meaning "edit a cell". Only fire ambiguity when co-occurring
# with 'cell' or an A1-style ref so unrelated 'change my volume' etc.
# don't trigger.
_CELL_EDIT_VERBS = (
    "change", "set", "write", "put", "update", "fill", "type",
)

# Classifier tools whose dispatch is deferred for peer-review when
# the utterance ALSO names a sheet or cell — the classic cross-domain
# collision the classifier's verbless-email regex creates.
_EMAIL_TOOLS_AMBIG = frozenset({
    "outlook_compose", "outlook_send", "email_send",
    "gmail_send", "gmail_compose", "ms_mail_send",
    "teams_send", "slack_post", "discord_post",
})

# Match an A1-style cell reference in the RAW utterance (uppercase
# required — so bare 'a1' / 'c3' in prose doesn't trigger).
_A1_CELL_RE = re.compile(r"\b[A-Z]{1,2}[1-9]\d{0,2}\b")


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
        self._intent_extractor = IntentExtractor(registry, logger=logger) \
            if registry is not None else None
        self._synthesizer = Synthesizer(logger=logger)
        self._plan_cache = PlanCache()
        # Plan reviser (Phase-2 cognition): inspects failed/incomplete
        # plan executions and produces an amended Plan. Lazy-imported
        # to dodge a circular import: plan_reviser pulls from
        # planner.plan, and planner.__init__ pulls orchestrator.
        # Wire the global CostMeter so the BAIL-on-budget-exhausted
        # branch isn't dead code (without this the reviser would
        # cheerfully blow past the daily budget cap).
        from ..plan_reviser import PlanReviser
        try:
            from ..cost_meter import global_meter as _global_cost_meter
            _cm = _global_cost_meter()
        except Exception:
            _cm = None
        self._reviser = PlanReviser(llm_planner=self._llm_planner,
                                    cost_meter=_cm)
        self._turn_counter = 0
        # Memory is optional — when None, recall/record are no-ops. The
        # manager wires a real MemoryManager when available; tests can
        # leave it unset to keep them offline.
        self._memory = memory
        # Tracks the most recent artifact(s) created in this session so
        # 'open it' / 'show me that' / 'open the sheet' (Tier 1
        # iris_open_last) can resolve without realtime's bad pronoun
        # resolution. Per-kind so 'open the sheet' doesn't grab a doc
        # just because the doc was created earlier; recency list backs
        # the unqualified 'open it' case.
        self._last_artifacts: Dict[str, Dict[str, Any]] = {}
        self._artifact_order: List[str] = []
        # Clear the global utterance cache when a new planner is
        # constructed — each planner is a new user session in
        # production (one per process lifetime) but a new fresh test
        # in test mode. Either way the prior session's cached replies
        # shouldn't surface in this one. Best-effort; never blocks
        # construction.
        try:
            from ..utterance_cache import global_utterance_cache
            global_utterance_cache().clear()
        except Exception:
            pass
        # Skills catalog (Tier 0.5: user-saved Plans replayed without an
        # LLM call). Lazily opened; failures just disable the tier.
        try:
            self._skills: Optional[SkillStore] = SkillStore()
        except Exception:
            self._skills = None
        # One-time cleanup: forget the Spotify play tools from the
        # local_intent learner. Earlier buggy dispatches trained
        # local_intent to associate "play <song>" utterances with
        # tool names that either don't exist in the registry
        # (spotify_play is setup_only=True) or that drop the song
        # arg (media_play_pause). Both produce visible failures.
        # Drop these poisoned mappings so the orchestrator's
        # arg-aware Tier-1 classifier wins. Idempotent + cheap;
        # tools forgotten here can be re-learned only via bare-
        # command utterances per the new training guard.
        try:
            from ..local_intent import global_classifier
            li = global_classifier()
            for tool in ("spotify_play", "spotify_pause",
                          "spotify_next", "spotify_previous",
                          "spotify_now_playing",
                          "media_play_pause",
                          "media_next_track",
                          "media_previous_track",
                          "run_quick_command"):
                try:
                    li.forget_tool(tool)
                except Exception:
                    pass
        except Exception:
            pass
        # Expose this planner to connectors that need name→id resolution
        # (e.g. GoogleSheetsConnector resolving 'the Q4 plan sheet' to
        # the spreadsheet_id of the sheet just created this session).
        global _LAST_PLANNER
        _LAST_PLANNER = self

    # Tools whose replies must NEVER be cached. Two distinct reasons:
    #   1) Time-sensitive answers — weather, inbox, calendar.
    #   2) Side-effect tools — each invocation must run, including
    #      the confirm-gate; otherwise a cached "Sent to Dani"
    #      message would short-circuit the send the second time.
    _NEVER_CACHE_TOOLS = frozenset({
        # Time-sensitive:
        "weather_get", "calendar_list", "gmail_list", "ms_mail_list",
        "email_summary", "spotify_now_playing", "volume_get",
        "iris_open_last", "phone_link_read_recent", "notion_search",
        "calendar_list_events", "gmail_read", "ms_mail_read",
        # Side-effect / destructive (must always re-execute, never
        # serve the cached "Done" reply):
        "gmail_send", "ms_mail_send", "outlook_compose",
        "outlook_send", "gmail_compose", "teams_send",
        "slack_post", "discord_post", "phone_link_send_text",
        "gdocs_create", "sheets_create", "slides_create",
        "drive_upload", "onedrive_upload", "notion_create_page",
        "notion_append_to_page", "notion_add_to_database",
        "onenote_create", "onenote_append_text",
        "todo_add", "volume_set", "volume_mute", "volume_toggle_mute",
        "discord_mute", "discord_toggle_mute",
        "discord_deafen", "discord_toggle_deafen",
        "spotify_play", "spotify_pause", "spotify_next",
        "spotify_previous", "spotify_shuffle",
        "iris_remember_contact", "iris_forget_contact",
        "iris_set_preference", "iris_setup_tool",
        "iris_remove_project",
        "file_delete", "open_url",
        "_iris_reauth_message",
    })

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
        # Phase-3: record the user's turn into the session buffer so
        # follow-ups ("do it again with Alice", "and on my other
        # monitor") see prior context. Incognito-honored inside the
        # buffer module.
        try:
            from ..session_buffer import global_session_buffer
            global_session_buffer().add_user(text)
        except Exception:
            pass
        # Phase-7 affect: scan this turn for mood/focus/verbosity
        # signals. Best-effort; downstream consumers (callback
        # engine, nudges, prose renderer) read the resulting
        # state when deciding behavior.
        try:
            from ..affect import observe_user_turn as _affect_obs
            _affect_obs(text)
        except Exception:
            pass

        # --- Tier 0a: "why did you do that?" — explain the most
        # recent turn from the CoT layer instead of asking the
        # planner to confabulate. Cheap, deterministic.
        try:
            from ..cot_explainer import (explain_last_turn,
                                          looks_like_explain_request)
            if looks_like_explain_request(text):
                explanation = explain_last_turn()
                if explanation:
                    return {
                        "steps": [], "results": [],
                        "message": explanation,
                    }
        except Exception:
            pass

        # --- Tier 0: utterance cache (0 tokens, 0 ms) ---
        # If we've answered THIS exact question recently with a reply
        # that isn't time-sensitive, return the cached reply. Saves a
        # full planner + LLM round-trip on repeats.
        try:
            from ..utterance_cache import global_utterance_cache
            from ..incognito import is_incognito
            if not is_incognito():
                cache = global_utterance_cache()
                hit = cache.get(text)
                if hit is not None:
                    return hit.payload or {
                        "steps": [], "results": [], "message": hit.message,
                    }
        except Exception:
            pass

        # --- Tier 0.3: local learned intent classifier (Phase 8 B3) ---
        # Per-user perceptron-style classifier that adapts to how
        # the user actually talks.
        #
        # GUARD: if the utterance contains a NAMED ENTITY (a noun
        # past the verb — "play poker face", "email Dani about Q3",
        # "open the Q3 deck") we MUST NOT dispatch via local_intent
        # because it would call the learned tool with EMPTY args,
        # dropping the entity the user named. The orchestrator's
        # Tier-1 classifier handles those patterns with proper
        # arg extraction. We only let local_intent fire when the
        # text reads as a "bare command" (no obvious object past the
        # first verb) — that's the regime where the user really did
        # mean the bare action.
        # Verbs that almost ALWAYS take an object — even when the
        # utterance is short, "play X" / "email X" / "open X" / etc.
        # need a tool with args, which local_intent (always-empty-args)
        # gets wrong. Skip local_intent entirely for these.
        _OBJECT_VERBS = (
            "play", "queue", "put", "listen",
            "email", "send", "text", "message", "post",
            "open", "launch", "start", "run", "execute",
            "search", "find", "look", "show", "pull",
            "remind", "watch", "remember", "forget",
            "set", "change", "switch",
            "tell", "ask", "explain", "summarize",
            "write", "draft", "compose", "create", "make",
        )

        def _looks_bare_command(t: str) -> bool:
            # Strip courtesy framing.
            clean = t.strip().lower()
            for prefix in ("please ", "can you ", "could you ",
                            "would you ", "hey iris ", "iris "):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):].lstrip()
            words = clean.rstrip("?.! ").split()
            # Tightened: ≤2 words AND first word not an
            # object-taking verb. Examples that pass: "pause",
            # "mute", "skip", "next", "stop". Examples that
            # DON'T pass: "play poker face" (3 words + 'play'
            # is object-verb), "email dani" (object-verb),
            # "open chrome" (object-verb).
            if len(words) > 2:
                return False
            if words and words[0] in _OBJECT_VERBS:
                return False
            return True
        try:
            from ..local_intent import classify as _li_classify
            if _looks_bare_command(text):
                hit = _li_classify(text)
                if hit is not None and hit.is_confident():
                    try:
                        out = self._registry.call(hit.tool, {})
                    except Exception:
                        out = None
                    if out is not None:
                        sr = StepResult(
                            step_id=0, tool=hit.tool,
                            status=str((out or {}).get("status")
                                        or "ok"),
                            output=out or {},
                            error=(out or {}).get("error"))
                        shim_step = Step(tool=hit.tool, args={},
                                         id=0,
                                         description="learned")
                        base_message = self._format_message(
                            shim_step, out)
                        message = self._conversationalize(
                            question=text,
                            base_message=base_message,
                            tool_name=hit.tool,
                            tool_result=out)
                        if self._logger:
                            self._logger.event(
                                "local_intent_hit",
                                tool=hit.tool,
                                confidence=round(
                                    hit.confidence, 3))
                        self._record_turn(text, None, [], [sr],
                                          message)
                        return {"steps": [], "results": [sr],
                                "message": message}
        except Exception:
            pass

        # --- Tier 0.4: smart fast-path (Phase 6 B4) ---
        # Pattern-matches trivial utterances ('what time is it', 'pause',
        # 'thanks') and either returns a static chat reply or dispatches
        # a single tool — bypassing the planner round-trip entirely.
        # Conservative-by-default: unknown / multi-clause / pronoun-laden
        # text falls through to the regular planner.
        try:
            from ..smart_fast_path import classify, FastPathKind
            fp = classify(text)
            if fp.kind == FastPathKind.CHAT and fp.reply:
                if self._logger:
                    self._logger.event(
                        "fastpath_chat", rule=fp.rule)
                self._record_turn(text, None, [], [], fp.reply)
                return {"steps": [], "results": [],
                        "message": fp.reply}
            if fp.kind == FastPathKind.DIRECT and fp.tool:
                try:
                    out = self._registry.call(fp.tool, dict(fp.args))
                except Exception as exc:
                    if self._logger:
                        self._logger.exception(
                            "fastpath_call_failed", exc,
                            tool=fp.tool)
                    out = None
                if out is not None:
                    sr = StepResult(
                        step_id=0, tool=fp.tool,
                        status=str((out or {}).get("status") or "ok"),
                        output=out or {},
                        error=(out or {}).get("error"))
                    shim_step = Step(tool=fp.tool,
                                     args=dict(fp.args),
                                     id=0,
                                     description=fp.rule)
                    base_message = self._format_message(
                        shim_step, out)
                    message = self._conversationalize(
                        question=text, base_message=base_message,
                        tool_name=fp.tool, tool_result=out)
                    if self._logger:
                        self._logger.event(
                            "fastpath_direct", rule=fp.rule,
                            tool=fp.tool)
                    self._record_turn(text, None, [], [sr], message)
                    return {"steps": [], "results": [sr],
                            "message": message}
        except Exception:
            # Never break the planner on a fast-path failure —
            # just fall through.
            pass

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
                merged_plan, results = self._run_with_revision(
                    skill_plan.goal, skill_plan)
                base_message = self._summarize(merged_plan, results)
                # Multi-step plans get the same conversational pass
                # as Tier-1 — UNLESS the plan asked for synthesis
                # (in which case the synthesizer already produced
                # an LLM-rendered conversational summary; re-running
                # render_jarvis on it is wasteful and can rewrite
                # facts).
                if (merged_plan.final or "").lower() in ("synthesize",
                                                          "speak"):
                    message = base_message
                else:
                    last_tool, last_output = self._last_tool_output(results)
                    message = self._conversationalize(
                        question=text, base_message=base_message,
                        tool_name=last_tool, tool_result=last_output)
                try:
                    self._skills.mark_used(skill_plan.goal)
                except Exception:
                    pass
                self._record_turn(text, merged_plan, merged_plan.steps,
                                  results, message)
                self._finalize_cot(message)
                payload = {
                    "steps": merged_plan.steps,
                    "results": results,
                    "plan": merged_plan,
                    "message": message,
                }
                self._maybe_cache_reply(
                    text, payload,
                    tools_used=[s.tool for s in merged_plan.steps])
                return payload

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
        chain_steps = self._classifier.classify_chain(text)
        single = chain_steps[0] if chain_steps else None
        chain_open_step = (chain_steps[1]
                           if chain_steps and len(chain_steps) > 1
                           else None)
        _BYPASS_MULTI = {"iris_lookup_contact", "iris_set_preference",
                         "iris_remember_contact", "iris_forget_contact",
                         "iris_open_last", "iris_setup_tool",
                         "gdocs_create", "sheets_create", "slides_create",
                         "gdocs_append_text",
                         "onenote_create", "onenote_append_text",
                         # forms_create extracts a title + questions
                         # tail; treat like gdocs_create.
                         "forms_create"}
        if multi and single is not None and single.tool not in _BYPASS_MULTI:
            single = None
            chain_open_step = None

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
                base_message = f"Got it — {names[0]} = {email}."
            else:
                base_message = (f"Got it — {', '.join(names[:-1])} and "
                                f"{names[-1]} all share {email}.")
            message = self._conversationalize(
                question=text, base_message=base_message,
                tool_name=single.tool, tool_result=sr.output)
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
                base_message = (f"Forgotten — removed {removed} fact"
                                f"{'s' if removed != 1 else ''} about {name}.")
            else:
                base_message = (f"I didn't have anything stored for {name}, "
                                f"so nothing to forget.")
            message = self._conversationalize(
                question=text, base_message=base_message,
                tool_name=single.tool, tool_result=sr.output)
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
                base_message = (f"I don't have a connector named '{name}'. "
                                f"Common ones I can set up: kicad. (Others "
                                f"come pre-wired and just need auth.)")
                message = self._conversationalize(
                    question=text, base_message=base_message,
                    tool_name=single.tool, tool_result=sr.output)
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
                base_message = (f"Set up — {name} ready"
                                + (f" ({version})" if version else "")
                                + (f" at {where}" if where else "")
                                + ". Tools added to the next session.")
            else:
                base_message = (f"Couldn't set up {name}: "
                                f"{result.get('error') or 'unknown error'}")
            message = self._conversationalize(
                question=text, base_message=base_message,
                tool_name=single.tool, tool_result=sr.output)
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr], "message": message}

        # --- Pseudo-tool: open the last-created artifact ('open it',
        # 'show me that', etc.). Resolves to whatever artifact (doc /
        # sheet / slide / OneNote / etc.) the planner most recently
        # produced in this session. Without this, 'open it' falls to
        # realtime which confabulates random plans.
        if single is not None and single.tool == "iris_open_last":
            hint = str((single.args or {}).get("kind_hint") or "").strip()
            last = None
            if hint:
                last = self._last_artifacts.get(hint)
            if not last and self._artifact_order:
                last = self._last_artifacts.get(self._artifact_order[0])
            if not last or not last.get("link"):
                sr = StepResult(step_id=0, tool=single.tool,
                                status="not_found",
                                output={"status": "not_found"})
                base_message = ("I don't have anything recent to open "
                                "yet. Make a doc, sheet, slide, OneNote "
                                "page, etc., and I'll be able to open it.")
                message = self._conversationalize(
                    question=text, base_message=base_message,
                    tool_name=single.tool, tool_result=sr.output)
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
            if status == "ok":
                base_message = (
                    f"Opening the {kind} \"{title}\" in your browser now.")
                message = base_message
            else:
                base_message = (
                    f"Couldn't open it: "
                    f"{out.get('error') if out else 'error'}")
                message = self._conversationalize(
                    question=text, base_message=base_message,
                    tool_name=single.tool, tool_result=sr.output)
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
                # Search BOTH kinds; first valid email wins. Skip facts
                # whose value isn't email-shaped (the LLM extractor /
                # self_learner can land relation facts like
                # kind=person/value='my brother' or value='contact' under
                # the same key — those must not flow into "X's email is Y").
                for cand in candidates:
                    for kind in ("person", "contact"):
                        try:
                            facts = self._memory._store.find_facts(  # type: ignore[attr-defined]
                                kind=kind, key=cand.lower())
                        except Exception:
                            facts = []
                        for f in facts:
                            v = (getattr(f, "value", "") or "").strip()
                            if "@" in v and "." in v.rsplit("@", 1)[-1]:
                                email = v
                                resolved_name = cand
                                break
                        if email:
                            break
                    if email:
                        break
            sr = StepResult(
                step_id=0, tool=single.tool,
                status="ok" if email else "not_found",
                output={"name": resolved_name, "email": email,
                        "status": "ok" if email else "not_found"})
            if email:
                base_message = f"{resolved_name}'s email is {email}."
            else:
                base_message = (f"I don't have {name}'s email in memory "
                                f"yet. Once you email them once, I'll "
                                f"remember.")
            message = self._conversationalize(
                question=text, base_message=base_message,
                tool_name=single.tool, tool_result=sr.output)
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
            base_message = f"Got it — I'll remember {key} = {value}."
            message = self._conversationalize(
                question=text, base_message=base_message,
                tool_name=single.tool, tool_result=sr.output)
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr], "message": message}

        # --- Pseudo-tool: natural-language reminder ('remind me in
        # 5 minutes to walk the dog'). Creates a time-triggered
        # standing order so the existing Sentinel-driven evaluator
        # fires it at the right moment + queues a notification.
        if single is not None and single.tool == "iris_remind":
            args = single.args or {}
            when_text = str(args.get("when_text") or "").strip()
            what = str(args.get("what") or "reminder").strip()
            base_message = ""
            order_id = ""
            try:
                from ..standing_orders import (
                    global_store, parse_when,
                    make_time_at_order)
                at_ts = parse_when(when_text)
                if at_ts is None:
                    base_message = (
                        f"Sorry — I couldn't parse \"{when_text}\". "
                        "Try \"in 30 minutes\", \"at 3pm\", or "
                        "\"tomorrow at 9am\".")
                else:
                    order = make_time_at_order(
                        user_text=text, at_ts=at_ts,
                        label=what[:80])
                    global_store().add(order)
                    order_id = order.id
                    import datetime as _dt
                    when_h = _dt.datetime.fromtimestamp(
                        at_ts).strftime("%I:%M %p").lstrip("0")
                    if at_ts - __import__("time").time() < 43200:
                        base_message = (
                            f"Got it — I'll remind you to "
                            f"{what} at {when_h}.")
                    else:
                        when_full = _dt.datetime.fromtimestamp(
                            at_ts).strftime(
                            "%A %I:%M %p").lstrip("0")
                        base_message = (
                            f"Got it — I'll remind you to "
                            f"{what} {when_full}.")
            except Exception as exc:
                base_message = (
                    f"Sorry — couldn't set that reminder: "
                    f"{type(exc).__name__}.")
            sr = StepResult(
                step_id=0, tool=single.tool,
                status="ok" if order_id else "error",
                output={"status": "ok" if order_id else "error",
                        "order_id": order_id,
                        "when_text": when_text,
                        "what": what})
            message = self._conversationalize(
                question=text, base_message=base_message,
                tool_name=single.tool, tool_result=sr.output)
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr],
                    "message": message}

        # --- Pseudo-tool: cancel standing order(s) by natural
        # language ('remove that reminder', 'cancel all reminders',
        # 'delete the bacon reminder'). Args:
        #   scope: 'all' / 'last' / 'match'
        #   query: substring to match against order label/user_text
        if single is not None and single.tool == "iris_cancel_order":
            args = single.args or {}
            scope = str(args.get("scope") or "match").lower()
            query = str(args.get("query") or "").strip().lower()
            base_message = ""
            removed_count = 0
            removed_labels = []
            try:
                from ..standing_orders import (
                    global_store, OrderState)
                store = global_store()
                active = store.all_active()
                to_remove = []
                if scope == "all":
                    to_remove = list(active)
                elif scope == "last":
                    if active:
                        to_remove = [
                            max(active, key=lambda o: o.created_at)]
                else:  # match
                    for o in active:
                        haystack = (
                            (o.label or "") + " "
                            + (o.user_text or "")
                        ).lower()
                        if query and query in haystack:
                            to_remove.append(o)
                for o in to_remove:
                    if store.update_state(
                            o.id, OrderState.CANCELLED.value):
                        removed_count += 1
                        removed_labels.append(
                            o.short_label())
                if removed_count == 0:
                    if scope == "all":
                        base_message = (
                            "Nothing to cancel — no active "
                            "reminders right now.")
                    elif scope == "match" and query:
                        base_message = (
                            f"Couldn't find a reminder matching "
                            f"\"{query}\". Say `/orders` to see "
                            "what's active.")
                    else:
                        base_message = (
                            "Couldn't find that reminder.")
                elif removed_count == 1:
                    base_message = (
                        f"Cancelled — \"{removed_labels[0]}\".")
                else:
                    if scope == "all":
                        base_message = (
                            f"Cancelled all {removed_count} "
                            "reminders.")
                    else:
                        base_message = (
                            f"Cancelled {removed_count}: "
                            + ", ".join(
                                f'"{lbl}"'
                                for lbl in removed_labels[:5])
                            + ("…" if removed_count > 5 else "."))
            except Exception as exc:
                base_message = (
                    f"Sorry — couldn't cancel: "
                    f"{type(exc).__name__}.")
            sr = StepResult(
                step_id=0, tool=single.tool,
                status="ok",
                output={"status": "ok",
                        "removed_count": removed_count,
                        "removed_labels": removed_labels,
                        "scope": scope, "query": query})
            message = self._conversationalize(
                question=text, base_message=base_message,
                tool_name=single.tool, tool_result=sr.output)
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr],
                    "message": message}

        # --- Pseudo-tool: natural-language inbox watcher ('watch
        # my inbox for the Q3 contract from Dani'). Creates a
        # standing order with inbox_match trigger.
        if single is not None and single.tool == "iris_watch_inbox":
            args = single.args or {}
            query = str(args.get("query") or "").strip()
            base_message = ""
            order_id = ""
            if not query:
                base_message = (
                    "Need something specific to watch for — "
                    "try \"watch my inbox for Q3 contract from "
                    "Dani\".")
            else:
                try:
                    from ..standing_orders import (
                        global_store, make_inbox_watch_order)
                    order = make_inbox_watch_order(
                        user_text=text, query=query)
                    global_store().add(order)
                    order_id = order.id
                    base_message = (
                        f"Watching your inbox for \"{query}\" — "
                        "I'll ping you when something matches.")
                except Exception as exc:
                    base_message = (
                        f"Sorry — couldn't set that watcher: "
                        f"{type(exc).__name__}.")
            sr = StepResult(
                step_id=0, tool=single.tool,
                status="ok" if order_id else "error",
                output={"status": "ok" if order_id else "error",
                        "order_id": order_id, "query": query})
            message = self._conversationalize(
                question=text, base_message=base_message,
                tool_name=single.tool, tool_result=sr.output)
            self._record_turn(text, None, [single], [sr], message)
            return {"steps": [single], "results": [sr],
                    "message": message}

        # Email-compose rewrite: respect default_send_via preference and
        # resolve recipient names via memory. Turns "email Dani saying hi"
        # into a direct gmail_send (or ms_mail_send) call when the user
        # has expressed a preference, instead of always opening the
        # Outlook draft window.
        if single is not None and single.tool in ("outlook_compose", "email_send"):
            single = self._maybe_rewrite_compose(single)

        # Dispatch the classifier's Step when it targets EITHER a
        # registered connector tool OR a built-in tool the executor
        # owns (run_quick_command, volume_set, etc.). Previously this
        # gated only on handles_connector(...) which silently dropped
        # built-in dispatches like run_quick_command — the classifier's
        # Step was discarded and the request fell through to Layer 2
        # LLM, which produced the wrong tool dispatch.
        if single is not None and (
                self._registry.handles_connector(single.tool)
                or single.layer == "touchless"):
            # Tier-1 ambiguity gate: when the raw utterance carries strong
            # signals that contradict the classifier's pick (cell ref /
            # 'google sheet' co-occurring with an email/messaging tool,
            # etc.), consult the Tier-1.5 intent extractor as a peer
            # reviewer BEFORE dispatching. If the extractor produces a
            # different registry-handled tool, dispatch THAT instead —
            # the classifier's verbless-email regex latches onto phrases
            # like 'change C1 to say Email' and would otherwise send the
            # wrong tool call. See bug: "Tier 1.5 never runs when Tier 1
            # fires" (2026-07).
            override_step: Optional[Step] = None
            if (self._intent_extractor is not None
                    and self._utterance_has_contradictory_signals(
                        text, single.tool)):
                try:
                    candidate = self._intent_extractor.try_extract(text)
                except Exception as exc:
                    if self._logger:
                        try:
                            self._logger.exception(
                                "intent_extractor_ambiguity_failed", exc)
                        except Exception:
                            pass
                    candidate = None
                if (candidate is not None
                        and candidate.tool != single.tool
                        and self._registry.handles_connector(candidate.tool)):
                    override_step = candidate
                    if self._logger:
                        try:
                            self._logger.event(
                                "planner_ambiguity_override",
                                classifier_tool=single.tool,
                                extractor_tool=candidate.tool)
                        except Exception:
                            pass
            step_to_run = override_step if override_step is not None else single
            dispatched = self._dispatch_connector_step(
                step_to_run, text,
                chain_open_step=(chain_open_step
                                 if override_step is None else None))
            if dispatched is not None:
                return dispatched
            return None

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

        # --- Tier 1.5: local-LLM intent extractor -------------------------
        # Fires only when Tier-1 declined and the request is a single
        # utterance mapping to an allow-listed connector CRUD tool. Fully
        # optional: if Ollama isn't running / model unavailable / JSON
        # invalid / arg-shape wrong, returns None and Tier-2 handles it.
        if (self._intent_extractor is not None
                and not multi):
            try:
                extracted = self._intent_extractor.try_extract(text)
            except Exception as exc:
                if self._logger:
                    self._logger.exception(
                        "intent_extractor_failed", exc)
                extracted = None
            if extracted is not None:
                dispatched = self._dispatch_connector_step(
                    extracted, text, chain_open_step=None)
                if dispatched is not None:
                    return dispatched

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
                try:
                    from ..latency_dashboard import StageTimer
                    with StageTimer("plan"):
                        plan = self._llm_planner.plan(
                            text, memory_context=memory_ctx)
                except Exception:
                    plan = self._llm_planner.plan(
                        text, memory_context=memory_ctx)
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
                try:
                    from ..latency_dashboard import StageTimer
                    with StageTimer("tool"):
                        merged_plan, results = self._run_with_revision(
                            text, plan)
                except Exception:
                    merged_plan, results = self._run_with_revision(
                        text, plan)
                base_message = self._summarize(merged_plan, results)
                if (merged_plan.final or "").lower() in ("synthesize",
                                                          "speak"):
                    message = base_message
                else:
                    last_tool, last_output = self._last_tool_output(results)
                    message = self._conversationalize(
                        question=text, base_message=base_message,
                        tool_name=last_tool, tool_result=last_output)
                self._record_turn(text, merged_plan, merged_plan.steps,
                                  results, message)
                self._finalize_cot(message)
                payload = {
                    "steps": merged_plan.steps,
                    "results": results,
                    "plan": merged_plan,
                    "message": message,
                }
                self._maybe_cache_reply(
                    text, payload,
                    tools_used=[s.tool for s in merged_plan.steps])
                return payload
        return None

    def _dispatch_connector_step(
        self, step: "Step", text: str,
        chain_open_step: Optional["Step"] = None,
    ) -> Optional[Dict[str, Any]]:
        """Shared Tier-1 / Tier-1.5 dispatch: run one connector-owned
        Step, wrap the result, optionally chain an 'open the artifact'
        step, run the conversational pass, and record the turn. Returns
        the same payload shape both tiers produce ({steps, results,
        message}); returns None on a registry-level exception so the
        caller can fall through to the next tier."""
        # Close the confirm-gate hole for classifier-produced steps:
        # the per-Step needs_confirm bool from the deterministic path
        # is often False even for RISKY_TOOLS (outlook_compose,
        # email_send, gmail_send, etc.), so also consult the tool-name
        # allowlist — same signal Phase-2/Skills already use.
        needs_confirm_now = (step.needs_confirm
                             or plan_needs_confirm([step.tool]))
        if needs_confirm_now and self._confirm is not None and \
                not self._confirm(f"Run {step.tool}?", step.description):
            return {
                "steps": [step],
                "results": [StepResult(step_id=0, tool=step.tool,
                                       status="cancelled",
                                       output={"status": "cancelled"})],
                "message": "Cancelled."}
        # Ambiguous-name backstop. Delete tools that identify their
        # target by NAME only (tasks_delete, iris_forget_contact,
        # drive_trash, ...) silently used to grab the newest artifact
        # regardless of kind — so 'delete scratch-notes' would wipe
        # the task when the user meant the sheet. If the name matches
        # 2+ artifacts spanning DIFFERENT kinds, refuse the delete
        # and surface a needs_clarification result the caller can
        # use to ask the user which one they meant. When the args
        # already carry an id / range the helper returns None here
        # and dispatch proceeds normally (unambiguous).
        if step.tool in _DELETE_TOOLS_NEEDING_DISAMBIG:
            name_arg = _extract_delete_name_arg(step.args)
            if name_arg:
                matches = self.find_all_artifacts_by_name(name_arg)
                distinct_kinds = {
                    str(m.get("kind") or "") for m in matches
                }
                distinct_kinds.discard("")
                if len(matches) > 1 and len(distinct_kinds) >= 2:
                    candidates = [
                        {"kind": m.get("kind"),
                         "title": m.get("title")}
                        for m in matches
                    ]
                    kinds_phrase = " and ".join(
                        f"a {k}" for k in sorted(distinct_kinds))
                    msg = (
                        f"I'd be happy to delete '{name_arg}' — but I "
                        f"see {kinds_phrase} with that name. Which one "
                        f"did you mean?"
                    )
                    sr = StepResult(
                        step_id=0, tool=step.tool,
                        status="needs_clarification",
                        output={"status": "needs_clarification",
                                "message": msg,
                                "candidates": candidates},
                        error=None)
                    if self._logger:
                        try:
                            self._logger.event(
                                "planner_delete_ambiguous",
                                tool=step.tool, name=name_arg,
                                kinds=sorted(distinct_kinds))
                        except Exception:
                            pass
                    return {
                        "steps": [step],
                        "results": [sr],
                        "message": msg,
                        "needs_clarification": True,
                        "candidates": candidates,
                    }
        try:
            if step.tool == "contacts_search":
                out = self._contacts_search_cascade(step.args)
            else:
                out = self._registry.call(step.tool, step.args)
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "iris_planner_call_failed", exc, tool=step.tool)
            return None
        sr = StepResult(step_id=0, tool=step.tool,
                        status=str((out or {}).get("status") or "ok"),
                        output=out or {}, error=(out or {}).get("error"))
        base_message = self._format_message(step, out or {})

        chain_sr: Optional[StepResult] = None
        chain_finalized = False
        if (chain_open_step is not None
                and sr.status == "ok"
                and isinstance(out, dict)
                and out.get("link")):
            kind = _artifact_kind_from_tool(step.tool)
            title = str((out or {}).get("title")
                        or (out or {}).get("name") or kind)
            link = str(out.get("link") or "")
            self._record_artifact(kind=kind, title=title,
                                  link=link, tool=step.tool)
            try:
                open_out = self._registry.call(
                    "open_url", {"url_or_query": link})
            except Exception as exc:
                if self._logger:
                    self._logger.exception(
                        "iris_chain_open_failed", exc, tool=step.tool)
                open_out = {"status": "error",
                            "error": f"{type(exc).__name__}: {exc}"}
            open_status = str((open_out or {}).get("status") or "ok")
            chain_sr = StepResult(
                step_id=2, tool="iris_open_last",
                status=open_status,
                output={"link": link, "title": title,
                        "status": open_status, "kind": kind},
                error=(open_out or {}).get("error"))
            if open_status == "ok":
                base_message = (
                    f"Done — created \"{title}\" and opening "
                    f"the {kind} in your browser now.")
                chain_finalized = True
            else:
                base_message = base_message.rstrip(".") + (
                    f" — but I couldn't open the {kind}: "
                    f"{(open_out or {}).get('error') or 'unknown error'}")

        if chain_finalized:
            message = base_message
        else:
            message = self._conversationalize(
                question=text, base_message=base_message,
                tool_name=step.tool, tool_result=out or {})
        steps_out = [step] + ([chain_open_step]
                                if chain_open_step is not None
                                and chain_sr is not None else [])
        results_out = [sr] + ([chain_sr] if chain_sr is not None else [])
        self._record_turn(text, None, steps_out, results_out, message)
        return {"steps": steps_out, "results": results_out,
                "message": message}

    def _maybe_cache_reply(self, text: str, payload: Dict[str, Any],
                           tools_used: list) -> None:
        """Cache the reply iff none of the tools used produce time-
        sensitive output (weather, inbox, etc.) AND incognito is off.
        Idempotent, best-effort, never raises into the call path."""
        try:
            from ..incognito import is_incognito
            if is_incognito():
                return
            tool_names = {str(t) for t in (tools_used or [])}
            if tool_names & self._NEVER_CACHE_TOOLS:
                return
            from ..utterance_cache import global_utterance_cache
            global_utterance_cache().put(
                text, str(payload.get("message") or ""),
                payload=payload, cacheable=True)
        except Exception:
            pass

    @staticmethod
    def _last_tool_output(results: list) -> tuple:
        """Return (tool_name, output_dict) for the last OK result in
        `results` — used so the conversational renderer gets the most
        relevant structured data to draw facts from when summarizing
        a multi-step plan. Falls back to the final result regardless
        of status if no OK result exists."""
        for r in reversed(results or []):
            if getattr(r, "status", None) == "ok":
                out = getattr(r, "output", None)
                return (getattr(r, "tool", ""),
                        out if isinstance(out, dict) else {})
        if results:
            r = results[-1]
            out = getattr(r, "output", None)
            return (getattr(r, "tool", ""),
                    out if isinstance(out, dict) else {})
        return ("", {})

    # ---- conversational reply layer --------------------------------------
    def _conversationalize(self, *, question: str, base_message: str,
                           tool_name: str = "",
                           tool_result: Optional[Dict[str, Any]] = None
                           ) -> str:
        """Pass `base_message` through the Jarvis prose renderer so
        EVERY reply path produces conversational Iris-voice output —
        not just the Tier-1 connector path.

        The renderer auto-skips very short status confirmations
        ("Got it." / "Muted.") so the latency overhead is paid only
        where it materially changes the reply. On any failure
        (no API key, timeout, fact-preservation guard tripped) the
        original `base_message` is returned unchanged.
        """
        if not base_message:
            return base_message
        # Phase-7 callback hint: ask the engine whether there's a
        # clean reference worth weaving in. Cheap, returns None
        # most of the time.
        callback_hook = ""
        try:
            from ..callback_engine import (maybe_callback,
                                            render_as_prompt_block)
            from ..session_buffer import global_session_buffer
            from ..entity_graph import global_graph
            from ..project_profile import current_profile
            from ..repo_focus_watcher import current_repo_context
            focus_root = None
            try:
                rc = current_repo_context()
                if rc is not None:
                    focus_root = getattr(rc, "root", None)
            except Exception:
                pass
            hint = maybe_callback(
                user_text=question or "",
                session_buffer=global_session_buffer(),
                entity_graph=global_graph(),
                project_profile=current_profile(focus_root))
            if hint is not None:
                callback_hook = render_as_prompt_block(hint)
        except Exception:
            callback_hook = ""
        try:
            from ..prose_renderer import render_jarvis
            return render_jarvis(
                question=question or "",
                tool_name=tool_name,
                tool_result=tool_result or {},
                fallback=base_message,
                context="",
                callback_hint=callback_hook,
            )
        except Exception:
            return base_message

    # ---- execute-with-revision -------------------------------------------
    def _run_with_revision(self, goal: str, plan: "Plan") -> tuple:
        """Run a plan via the executor, then consult the PlanReviser. On
        REVISE_AND_RETRY, run the amended plan and merge results.
        Returns (final_plan, merged_results). Capped at max 2 revisions
        per turn by the reviser itself.

        Records the reasoning trail (CoT layer) so the user can later
        ask "why did you do that?" — incognito-honoring."""
        self._turn_counter += 1
        turn_id = f"turn-{self._turn_counter}"
        # Start a CoT trail. Honors incognito at finalize time.
        cot_trail = None
        try:
            from ..cot_layer import global_cot_layer, DecisionStage
            cot_layer = global_cot_layer()
            cot_trail = cot_layer.start_turn(goal, turn_id=turn_id)
            cot_trail.inferred_goal = (plan.goal or "")[:200]
            cot_trail.add_decision(
                DecisionStage.PLAN,
                choice=f"plan({len(plan.steps)} steps)",
                why=f"initial steps: {','.join(s.tool for s in plan.steps)}")
        except Exception:
            cot_layer = None
        results = self._executor.run(plan)
        # The realtime path may have stitched extra structured outputs;
        # we use the executor's raw results for revision decisions.
        all_steps = list(plan.steps)
        all_results = list(results)
        active_plan = plan
        from ..plan_reviser import ReviseAction
        while True:
            try:
                decision = self._reviser.decide(
                    goal=goal, plan=active_plan,
                    results=results, turn_id=turn_id)
            except Exception as exc:
                if self._logger:
                    self._logger.exception("reviser_decide_failed", exc)
                break
            if decision.action in (ReviseAction.DONE, ReviseAction.BAIL):
                break
            if decision.revised_plan is None or not decision.revised_plan.steps:
                break
            # P2-INT-03: if every step in the revised plan is the
            # `_iris_reauth_message` pseudo-tool, we DON'T touch the
            # executor (it can't dispatch pseudo-tools). Synthesize
            # StepResults directly so the orchestrator's summarizer
            # surfaces the reauth message to the user.
            revised = decision.revised_plan
            if all(s.tool == "_iris_reauth_message" for s in revised.steps):
                synthetic = []
                for s in revised.steps:
                    msg = str((s.args or {}).get("message") or
                              "Please reconnect the related account.")
                    synthetic.append(StepResult(
                        step_id=s.id, tool=s.tool, status="ok",
                        output={"status": "ok", "message": msg,
                                **(s.args or {})}))
                all_steps.extend(revised.steps)
                all_results.extend(synthetic)
                active_plan = revised
                break  # reauth message is terminal — stop revising.
            # Re-execute. Step IDs in the amended plan are local; the
            # executor doesn't care about cross-plan id collisions
            # because each run is independent.
            try:
                results = self._executor.run(revised)
            except Exception as exc:
                if self._logger:
                    self._logger.exception("reviser_rerun_failed", exc)
                break
            all_steps.extend(revised.steps)
            all_results.extend(results)
            active_plan = revised
        # Build a synthetic merged Plan so the caller's summarizer +
        # record_turn see every executed step.
        merged = Plan(goal=goal, steps=all_steps, final=plan.final)
        # Free per-turn state so cap counters don't leak across requests.
        self._reviser.reset_turn(turn_id)
        # Phase-4: notify the skill consolidator that this turn has
        # completed. The consolidator's `complete_turn` rolls the
        # tools it already saw on the bus into a "shape" signature
        # and emits a "save this as a skill?" nudge if the shape has
        # recurred N times.
        try:
            mgr = getattr(self, "_manager", None) or _peek_manager()
            consolidator = (
                getattr(mgr, "_skill_consolidator", None)
                if mgr is not None else None)
            if consolidator is not None:
                consolidator.complete_turn(
                    turn_id=turn_id, user_text=goal)
        except Exception:
            pass
        # Finalize CoT trail. The caller fills in the user-facing
        # message via record_cot_final(turn_id, message) after the
        # summarizer runs — we just stash the in-flight trail for now.
        if cot_trail is not None and cot_layer is not None:
            try:
                ok = sum(1 for r in all_results
                         if getattr(r, "status", "") == "ok")
                cot_trail.add_decision(
                    DecisionStage.PLAN,
                    choice=f"executed {len(all_results)} step(s)",
                    why=f"ok={ok}/{len(all_results)}")
                # Record the tool names that actually ran. cot_explainer
                # surfaces these in the "Tools called: ..." line.
                for s in all_steps:
                    tool = getattr(s, "tool", "") or ""
                    if tool:
                        cot_trail.add_tool_ref(tool)
                # Stash on the orchestrator so try_handle can finalize
                # with the message text once it's built.
                self._pending_cot_trail = (cot_layer, cot_trail)
            except Exception:
                pass
        return merged, all_results

    def _finalize_cot(self, message: str, error: str = "") -> None:
        """Called by the reply paths after they have the final
        user-facing message. Closes the CoT trail. No-op if no
        trail was opened (e.g. classifier-only Tier-1 path)."""
        pending = getattr(self, "_pending_cot_trail", None)
        if pending is None:
            return
        cot_layer, cot_trail = pending
        try:
            cot_layer.finalize(cot_trail, final_message=message,
                               error=error)
        except Exception:
            pass
        self._pending_cot_trail = None

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
                    description=f"send email via {preferred}",
                    needs_confirm=True)

    def _contacts_search_cascade(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """contacts_search with a three-tier cascade: own connector ->
        memory (kind person/contact) -> the OTHER connector
        (Microsoft365 <-> Google Contacts). Merges + dedupes by primary
        email so the same person across sources doesn't show up twice.

        Fixes the regression where the first-registered connector silently
        owned `contacts_search` and a miss returned "not found" even when
        the other connector OR memory had the contact."""
        args = dict(args or {})
        query = str(args.get("query") or "").strip()
        if not query:
            return {"status": "error", "error": "query is required"}
        merged: List[Dict[str, Any]] = []
        seen: set = set()
        errors: List[str] = []

        def _key(contact: Dict[str, Any]) -> str:
            emails = contact.get("emails") or []
            for e in emails:
                v = (e.get("value") if isinstance(e, dict) else None) or ""
                v = v.strip().lower()
                if v:
                    return v
            return (contact.get("display_name") or contact.get("name")
                    or "").strip().lower()

        def _merge(contacts: List[Dict[str, Any]]) -> None:
            for c in contacts or []:
                k = _key(c)
                if not k or k in seen:
                    continue
                seen.add(k)
                merged.append(c)

        # 1) Try the owning connector (whichever registered contacts_search
        # first — usually Microsoft365 when MS Graph is authed, else Google).
        primary_owner = None
        try:
            primary_owner = getattr(
                self._registry, "_connectors", None)
            if primary_owner is not None:
                primary_owner = primary_owner._owner.get("contacts_search")  # type: ignore[attr-defined]
        except Exception:
            primary_owner = None
        primary_id = (getattr(primary_owner, "id", "") or "").lower() \
            if primary_owner is not None else ""
        try:
            out = self._registry.call("contacts_search", args)
        except Exception as exc:
            out = {"status": "error",
                   "error": f"{type(exc).__name__}: {exc}"}
        if isinstance(out, dict):
            if out.get("status") == "ok":
                _merge(out.get("contacts") or [])
            elif out.get("error"):
                errors.append(f"{primary_id or 'primary'}: {out['error']}")

        # 2) Memory fallback: scan stored person/contact facts whose
        # value is email-shaped. Match either exact key OR substring
        # ('find vesko' should hit stored key='vesko' and also 'vesko-m').
        if self._memory is not None:
            qlow = query.lower()
            try:
                mem_rows = []
                for kind in ("person", "contact"):
                    try:
                        mem_rows.extend(
                            self._memory._store.find_facts(  # type: ignore[attr-defined]
                                kind=kind, key=qlow))
                    except Exception:
                        pass
                # Substring scan across ALL person/contact facts so
                # 'find Vesko' matches a stored key like 'vesko-m' or
                # 'vesko brother'. Bounded by find_facts' default limit.
                if not mem_rows:
                    try:
                        for kind in ("person", "contact"):
                            for r in self._memory._store.find_facts(  # type: ignore[attr-defined]
                                    kind=kind):
                                key_l = (getattr(r, "key", "") or "").lower()
                                if qlow and qlow in key_l:
                                    mem_rows.append(r)
                    except Exception:
                        pass
                for r in mem_rows:
                    v = (getattr(r, "value", "") or "").strip()
                    if "@" not in v or "." not in v.rsplit("@", 1)[-1]:
                        continue
                    name = (getattr(r, "key", "") or "").strip() or v
                    _merge([{
                        "name": name,
                        "display_name": name,
                        "emails": [{"value": v, "type": "memory"}],
                        "phones": [],
                        "organization": None,
                        "source": "memory",
                    }])
            except Exception:
                pass

        # 3) Other connector fallback: if MS owns the tool, try Google
        # ContactsConnector directly; if Google owns it, try MS365.
        other = None
        try:
            registry_inner = getattr(self._registry, "_connectors", None)
            if registry_inner is not None:
                if primary_id == "ms365":
                    other = registry_inner.find_by_id("contacts")
                elif primary_id == "contacts":
                    other = registry_inner.find_by_id("ms365")
        except Exception:
            other = None
        if other is not None and other is not primary_owner:
            try:
                if other.available():
                    other_out = other.execute("contacts_search", args)
                    if isinstance(other_out, dict):
                        if other_out.get("status") == "ok":
                            _merge(other_out.get("contacts") or [])
                        elif other_out.get("error"):
                            errors.append(
                                f"{getattr(other, 'id', 'other')}: "
                                f"{other_out['error']}")
            except Exception as exc:
                errors.append(
                    f"{getattr(other, 'id', 'other')}: "
                    f"{type(exc).__name__}: {exc}")

        # 4) Classic Outlook desktop COM fallback. Uses a distinct tool
        # name (outlook_com_contacts_search) so the registry doesn't
        # collide with the Microsoft365/Google contacts_search owner.
        # Falls through gracefully on New Outlook (no COM) via the
        # com_unavailable code.
        try:
            registry_inner = getattr(self._registry, "_connectors", None)
            outlook_com = (
                registry_inner.find_by_id("outlook_com")
                if registry_inner is not None else None)
        except Exception:
            outlook_com = None
        if outlook_com is not None and outlook_com is not primary_owner:
            try:
                if outlook_com.available():
                    oc_out = outlook_com.execute(
                        "outlook_com_contacts_search", args)
                    if isinstance(oc_out, dict):
                        if oc_out.get("status") == "ok":
                            _merge(oc_out.get("contacts") or [])
                        elif (oc_out.get("error")
                              and oc_out.get("code") != "com_unavailable"):
                            errors.append(
                                f"outlook_com: {oc_out['error']}")
            except Exception as exc:
                errors.append(
                    f"outlook_com: {type(exc).__name__}: {exc}")

        if merged:
            return {"status": "ok", "count": len(merged),
                    "contacts": merged,
                    "errors": errors or None}
        if errors:
            return {"status": "error", "error": "; ".join(errors[:3])}
        return {"status": "ok", "count": 0, "contacts": [],
                "message": f"No one matching {query!r} in contacts or memory."}

    # ---- memory bridge ----------------------------------------------------
    def _recall_context(self, text: str) -> str:
        """Pull a unified context block for the planner prompt. Empty
        string when no modality contributed anything.

        Phase-5 fusion: the per-modality fetch logic still lives in
        each modality module (memory, dictation, repo, session,
        screen). This method gathers them into a dict, hands them to
        `multimodal_fusion.build_unified_context` which scores
        relevance, allocates a 2KB total budget across modalities,
        dedupes overlap, and renders a single block in stable
        priority order."""
        # Build a per-modality block dict for the fuser.
        modality_parts: Dict[str, str] = {}
        # Memory recall.
        if self._memory is not None:
            try:
                recall = self._memory.recall(text, k=3)
                ctx = recall.get("context") or ""
                if ctx:
                    modality_parts["memory"] = ctx
            except Exception as exc:  # pragma: no cover - defensive
                if self._logger:
                    self._logger.exception("memory_recall_failed", exc)
        # Dictation bridge — only when the user's request appears to
        # reference recently-dictated text. Cheap heuristic; the bridge
        # call is also gated by max_age_sec inside the bridge.
        try:
            from ..dictation_bridge import global_bridge
            if _wants_recent_dictation(text):
                last = global_bridge().last_dictation_text(
                    max_age_sec=300.0)
                if last:
                    window = (global_bridge().last_dictation_window()
                              or "the focused window")
                    modality_parts["dictation"] = (
                        "RECENT DICTATION (just typed into "
                        f"{window}):\n---\n{last[:1500]}\n---")
        except Exception:
            pass
        # Repo context — when the user is currently focused in an IDE
        # with a known project open, fold its summary into the planner
        # prompt. Cheap: cached by the watcher; we just read.
        try:
            from ..repo_focus_watcher import current_repo_context
            ctx = current_repo_context()
            if ctx is not None:
                block = ctx.as_context_block()
                if block:
                    modality_parts["repo"] = block
        except Exception:
            pass
        # Session buffer — recent conversation turns so the planner
        # sees the immediate context for follow-ups. Bounded to ~12
        # turns / 4000 chars by the buffer itself.
        try:
            from ..session_buffer import global_session_buffer
            recent_turns = global_session_buffer().render(max_turns=8)
            # Skip the most recent (the user message that just kicked
            # off this turn — already the planner's primary input).
            if recent_turns:
                # Strip the last line (current user text).
                lines = recent_turns.split("\n")
                if lines and lines[-1].lower().startswith("user:"):
                    lines = lines[:-1]
                if lines:
                    modality_parts["session"] = (
                        "RECENT CONVERSATION:\n" + "\n".join(lines))
        except Exception:
            pass
        # Phase-4: ambient screen awareness — when the user's
        # request looks vision-relevant ("what does this say",
        # "summarize this", "that one"), fold in the most recent
        # screen summary captured by the Sentinel watcher. Cheap:
        # cached by the awareness layer; we just read.
        try:
            from ..screen_awareness import (global_screen_awareness,
                                              looks_vision_relevant)
            if looks_vision_relevant(text):
                summary = global_screen_awareness().current_summary()
                if summary is not None:
                    block = summary.as_context_block()
                    if block:
                        modality_parts["screen"] = block
        except Exception:
            pass
        # Phase-6: project profile — when the user is in a known
        # project (resolved from repo_focus_watcher or the most
        # recently-touched profile), include its name, branch,
        # recent files, and last activity. Lets Iris answer "what
        # was I doing here" without a planner round-trip.
        try:
            from ..project_profile import (current_profile,
                                             render_for_planner)
            from ..repo_focus_watcher import current_repo_context
            focus_root = None
            try:
                rc = current_repo_context()
                if rc is not None:
                    focus_root = getattr(rc, "root", None)
            except Exception:
                focus_root = None
            profile = current_profile(focus_root)
            if profile is not None:
                block = render_for_planner(profile)
                if block:
                    modality_parts["project"] = (
                        "PROJECT CONTEXT: " + block)
        except Exception:
            pass
        # Phase-6: pronoun + name resolver. Cross-references the
        # session buffer, screen summary, and entity graph to bind
        # "it" / "that" / "him" / "Dani" to concrete entities BEFORE
        # the planner sees them. Surfaces as a RESOLVED REFERENCES
        # block — the planner gets both the original text and the
        # deterministic resolution as a hint.
        try:
            from ..pronoun_resolver import resolve_references
            from ..session_buffer import global_session_buffer
            from ..screen_awareness import global_screen_awareness
            report = resolve_references(
                text,
                session_buffer=global_session_buffer(),
                screen_summary=(
                    global_screen_awareness().current_summary()),
            )
            if report.has_resolutions() and report.prompt_block:
                modality_parts["resolved_refs"] = report.prompt_block
        except Exception:
            pass
        # Recent-artifact tracker — surface the spreadsheet_id /
        # doc id / slideshow id for artifacts created this session
        # so the planner LLM can pass them straight into the next
        # tool call (e.g. sheets_update_range against the sheet just
        # created). Without this the planner sees only the URL in the
        # session buffer and has no way to extract the bare id.
        try:
            if self._last_artifacts:
                lines: List[str] = []
                seen_kinds: List[str] = []
                # Render in recency order: most recent of each kind
                # first (up to 3 kinds — sheet/doc/slideshow is
                # plenty for typical creation chains).
                for kind in self._artifact_order[:3]:
                    art = self._last_artifacts.get(kind)
                    if not art or kind in seen_kinds:
                        continue
                    seen_kinds.append(kind)
                    link = str(art.get("link") or "")
                    title = str(art.get("title") or kind)
                    # Pull the bare id out of the link when we can,
                    # so the planner doesn't have to URL-parse.
                    art_id: Optional[str] = None
                    try:
                        import re as _re
                        m = _re.search(
                            r"/(?:spreadsheets|document|"
                            r"presentation)/d/([A-Za-z0-9_\-]+)",
                            link)
                        if m:
                            art_id = m.group(1)
                    except Exception:
                        art_id = None
                    id_part = f" id={art_id}" if art_id else ""
                    lines.append(f"- {kind} '{title}'{id_part} "
                                  f"(tool={art.get('tool')})")
                if lines:
                    modality_parts["artifacts"] = (
                        "RECENT ARTIFACTS (this session — pass the "
                        "id directly into the matching tool):\n"
                        + "\n".join(lines))
        except Exception:
            pass
        # Phase-5: hand the modality blocks to the fuser for
        # relevance scoring + budget allocation + dedup. Falls
        # back to plain concatenation if the fuser is unavailable.
        try:
            from ..multimodal_fusion import build_unified_context
            return build_unified_context(text, modality_parts)
        except Exception:
            return "\n\n".join(modality_parts.values())

    def _record_artifact(self, *, kind: str, title: str,
                          link: str, tool: str) -> None:
        """Store an artifact in the per-kind map and bump it to the front
        of the recency list. Shared by the planner-result walker AND the
        realtime tool-call recorder so artifacts created via either path
        stay addressable by 'open it' / 'open the sheet' etc."""
        if not link or not kind:
            return
        self._last_artifacts[kind] = {
            "kind": kind, "title": title or kind, "link": link,
            "tool": tool, "ts": time.time(),
        }
        try:
            if kind in self._artifact_order:
                self._artifact_order.remove(kind)
        except Exception:
            pass
        self._artifact_order.insert(0, kind)

    def record_artifact_from_result(self, tool: str,
                                     output: Any) -> None:
        """Public hook for the realtime tool-call path. Mirrors the
        write-site in _record_turn so artifacts created via the LLM
        tool-calling path also populate _last_artifacts. No-op when the
        output lacks a usable link."""
        if not isinstance(output, dict):
            return
        link = output.get("link")
        if not link:
            return
        tool = str(tool or "")
        kind = _artifact_kind_from_tool(tool)
        title = output.get("title") or output.get("name") or kind
        self._record_artifact(kind=kind, title=str(title),
                               link=str(link), tool=tool)

    def find_artifact_by_name(self, name: str,
                               kind: Optional[str] = None
                               ) -> Optional[Dict[str, Any]]:
        """Public name-resolver for connectors. Looks up `name` (case-
        and substring-tolerant) against artifacts created this session.
        When `kind` is set, restricts the search to that kind (e.g.
        'sheet' so 'open the X doc' doesn't grab a sheet of similar
        title). Returns the artifact dict (`kind/title/link/tool/ts`)
        or None when nothing matches."""
        if not name:
            return None
        needle = name.strip().lower()
        candidates: List[Dict[str, Any]] = []
        for k, art in self._last_artifacts.items():
            if kind and k != kind:
                continue
            candidates.append(art)
        if not candidates:
            return None
        # Sort newest-first so a fresh "Q4 plan" wins over an older
        # one with the same name.
        candidates.sort(key=lambda a: float(a.get("ts") or 0.0),
                         reverse=True)
        # Exact (case-insensitive) match wins.
        for art in candidates:
            if str(art.get("title") or "").lower() == needle:
                return art
        # Then containment either way (handles "the Q4 plan sheet"
        # vs stored "Q4 plan" and vice versa).
        for art in candidates:
            title = str(art.get("title") or "").lower()
            if needle in title or title in needle:
                return art
        return None

    def find_all_artifacts_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Same match rules as find_artifact_by_name but returns EVERY
        match across kinds — for ambiguity detection ('delete
        scratch-notes' when both a sheet and a task have that name).
        Newest-first. Returns [] when nothing matches.

        find_artifact_by_name is intentionally kept as-is (single-return
        shape) because external callers depend on it — this is a
        separate helper for the ambiguity backstop."""
        if not name:
            return []
        needle = name.strip().lower()
        if not needle:
            return []
        # Walk every kind. _last_artifacts stores at most one artifact
        # per kind so cross-kind duplicates come from DIFFERENT slots
        # (kind='sheet' with title X, kind='task' with title X).
        exact: List[Dict[str, Any]] = []
        contains: List[Dict[str, Any]] = []
        for k, art in self._last_artifacts.items():
            title = str(art.get("title") or "").lower()
            if title == needle:
                exact.append(art)
            elif title and (needle in title or title in needle):
                contains.append(art)
        # Exact matches first, then containment matches — the caller
        # only cares about "how many DISTINCT kinds matched", but we
        # preserve the priority for deterministic ordering when
        # multiple exact matches exist (impossible today since kinds
        # are unique, but defensive against future storage changes).
        combined = exact + contains
        # Dedupe by (kind, title, ts) so the same entry doesn't appear
        # twice when a future storage change lets a single artifact
        # match both exact and substring passes.
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for art in combined:
            key = (str(art.get("kind") or ""),
                   str(art.get("title") or ""),
                   float(art.get("ts") or 0.0))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(art)
        deduped.sort(key=lambda a: float(a.get("ts") or 0.0),
                     reverse=True)
        return deduped

    def _record_turn(self, user_text: str, plan: Any,
                     steps: list, results: list, message: str) -> None:
        """Persist this turn into memory + update the 'last artifact'
        pointer so 'open it' / 'show me that' work. Best-effort;
        failures never bubble up to the user."""
        # Phase-3: append the assistant reply to the session buffer
        # so the next turn's _recall_context can include it as
        # immediate context. Incognito-honored inside the buffer.
        try:
            from ..session_buffer import global_session_buffer
            global_session_buffer().add_assistant(message)
        except Exception:
            pass
        # Phase-8 B3: record each successful step into the local
        # intent classifier so it learns the user's phrasing.
        try:
            from ..local_intent import record_example
            for sr in results or []:
                tool = getattr(sr, "tool", "") or ""
                status = (getattr(sr, "status", "") or "").lower()
                if not (tool and status == "ok" and user_text):
                    continue
                # GUARD: only train local_intent when the dispatched
                # step had NO args. local_intent always calls tools
                # with empty args; training it on a tool that needs
                # args (spotify_play(query=X), email_send(to=Y),
                # volume_set(percent=Z)) would later cause the wrong
                # behavior — Tier-0.3 would fire the tool empty,
                # dropping the user's intent.
                step_args = (getattr(
                    sr, "step", None) and
                    getattr(sr.step, "args", None)) or {}
                # Best-effort: results may not carry a back-pointer
                # to step. Iterate steps list to find the match.
                if not step_args:
                    for s in (steps or []):
                        if getattr(s, "tool", "") == tool:
                            step_args = (getattr(s, "args", {})
                                          or {})
                            break
                if step_args:
                    # Tool had args — don't train. The Tier-1
                    # classifier (with proper arg-extracting regex)
                    # is the right place to handle this pattern.
                    break
                record_example(user_text, tool, success=True)
                break  # one example per turn — first tool wins
        except Exception:
            pass
        # Walk results for any link — the LAST one wins per kind (the
        # planner tends to create-then-fill, so the final artifact of a
        # given kind is the one the user means). Per-kind storage lets
        # 'open the sheet' find the sheet even when a doc was created
        # earlier in the same session; recency list backs 'open it'.
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
            self._record_artifact(kind=kind, title=str(title),
                                  link=str(link), tool=str(tool))
        if self._memory is None:
            return
        try:
            self._memory.record(user_text, plan, steps, results, message)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_record_failed", exc)

    @staticmethod
    def _utterance_has_contradictory_signals(
            text: str, classifier_tool: str) -> bool:
        """True when the raw utterance carries strong signals that
        contradict the deterministic classifier's chosen tool.

        Triggers the Tier-1.5 'peer reviewer' path in try_handle so the
        local-LLM intent extractor gets to weigh in BEFORE Tier-1
        dispatches — this closes the bug where 'In the google sheet
        called testing and can you change C1 to say Email' was routed
        to outlook_compose (the classifier's verbless-email regex
        latches onto the trailing 'Email' as an intent verb) instead of
        sheets_update_range.

        Conservative on purpose: only flags obvious cross-domain
        collisions so we don't pay an Ollama round-trip on every
        single-domain utterance. Safe to call — pure string / regex,
        no I/O.
        """
        if not text or not classifier_tool:
            return False
        # Already a sheets/excel tool - no contradiction possible.
        if (classifier_tool.startswith("sheets_")
                or classifier_tool.startswith("excel_")):
            return False
        raw = str(text)
        lower = raw.lower()
        has_sheet_word = any(w in lower for w in _SHEET_CONTEXT_WORDS)
        # Uppercase A1 ref in the RAW utterance (bare lowercase 'a1'
        # doesn't count).
        has_a1_ref = bool(_A1_CELL_RE.search(raw))
        padded = f" {lower} "
        has_cell_noun = " cell " in padded or " cells " in padded
        a1_with_context = has_a1_ref and (has_sheet_word or has_cell_noun)
        has_cell_edit_verb = any(v in lower for v in _CELL_EDIT_VERBS)
        verb_with_ref = has_cell_edit_verb and (has_a1_ref or has_cell_noun)
        # (a) Classifier picked email/messaging AND utterance names a
        # sheet or cell — the canonical cross-domain collision.
        if (classifier_tool in _EMAIL_TOOLS_AMBIG
                and (has_sheet_word or has_a1_ref or has_cell_noun)):
            return True
        # (b) Utterance contains an A1 ref together with a sheet /
        # cell context word, regardless of which non-sheets tool was
        # picked (guards against Bach's 'C3' / paper's 'A4' false
        # positives by requiring the co-occurring context).
        if a1_with_context:
            return True
        # (c) Explicit cell-edit verb ('change', 'set', 'write', ...)
        # co-occurring with 'cell' or an A1 ref AND a sheet context
        # word — 'change cell C1 to Email in my testing sheet'.
        if verb_with_ref and has_sheet_word:
            return True
        return False

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
    # Persona-tinted confirmation templates for Tier-1 deterministic
    # replies. Tier-1 has NO LLM in the loop, so we apply the active
    # voice locally via a small lookup. Each preset has MULTIPLE
    # phrasings so the same prompt twice doesn't return the same
    # canned line — sounds like a person, not a chatbot.
    # Each phrasing is "(with_subject_template, without_subject_text)".
    # {x} is the song/playlist/whatever subject.
    _PERSONA_PLAY_VARIANTS = {
        "default": [
            ("Playing {x} now.",         "Got it — playing now."),
            ("Putting on {x}.",          "Playing it."),
            ("Here's {x}.",              "Playing."),
            ("{x} coming up.",           "On the way."),
            ("Queued {x} for you.",      "Queued."),
        ],
        "jarvis": [
            ("Playing {x}, sir.",        "Pulling that up, sir."),
            ("Putting on {x}, sir.",     "Of course, sir."),
            ("{x}, sir.",                "Very good, sir."),
            ("As you wish — {x}.",       "As you wish, sir."),
            ("{x} it is, sir.",          "Done, sir."),
        ],
        "concise": [
            ("Playing {x}.",             "Playing."),
            ("{x}.",                     "Done."),
            ("Queued {x}.",              "Queued."),
            ("On — {x}.",                "On."),
        ],
        "warm": [
            ("Got {x} going for you.",   "Playing it now."),
            ("Here you go — {x}.",       "On it, friend."),
            ("Putting on {x} for you.",  "Got it."),
            ("Coming right up — {x}.",   "Coming right up."),
            ("{x}, enjoy.",              "Enjoy."),
        ],
        "playful": [
            ("Cueing up {x} — enjoy!",   "On it!"),
            ("{x} incoming — buckle up.", "Done and done!"),
            ("Spinning {x}.",            "Spinning it up."),
            ("Alright, {x} it is.",      "Sweet."),
            ("Pulling up {x} — fun pick.", "Nice — on it!"),
        ],
        "tutor": [
            ("Playing {x}.",             "Playing now."),
            ("Now playing {x}.",         "Now playing."),
            ("Starting {x} for you.",    "Starting now."),
        ],
    }

    @staticmethod
    def _persona_play_reply(subject: str = "") -> str:
        """Persona-tinted 'now playing X' confirmation for Tier-1
        media replies. Picks ONE of the preset's variants at random
        so back-to-back commands don't produce the same line.

        subject is e.g. 'Poker Face by Lady Gaga' or the user's
        query — empty string falls back to the no-subject variant."""
        try:
            from .. import persona_voice
            name = persona_voice.active_preset().name
        except Exception:
            name = "default"
        variants = IrisPlanner._PERSONA_PLAY_VARIANTS.get(
            name, IrisPlanner._PERSONA_PLAY_VARIANTS["default"])
        import random as _r
        with_t, without_t = _r.choice(variants)
        if subject:
            return with_t.format(x=subject)
        return without_t

    @staticmethod
    def _format_message(step: Step, result: Dict[str, Any]) -> str:
        if (result.get("status") or "").lower() != "ok":
            err = result.get("error") or ""
            # Conversational error — the bare "Couldn't run X: error"
            # reads like a log line. When we have a real message use
            # it; otherwise apologize and offer to retry.
            if err:
                return f"Sorry — {err}"
            return ("I hit a snag with that. Want me to "
                    "try again?")
        tool = step.tool
        args = step.args or {}
        # ---- media: persona-tinted reply ---------------------------
        if tool == "spotify_play":
            # Prefer the fresh now_playing_title from the connector
            # (it polls + only fills this when track actually changed).
            title = (result.get("now_playing_title") or "").strip()
            artist = (result.get("now_playing_artist") or "").strip()
            if title and artist:
                subject = f"{title} by {artist}"
            elif title:
                subject = title
            else:
                # Fall back to echoing the user's query — safer than
                # naming a song we can't confirm.
                subject = (args.get("query") or "").strip()
                # Strip generic words so we don't say "Playing music".
                if subject.lower() in ("", "music", "song", "track"):
                    subject = ""
            return IrisPlanner._persona_play_reply(subject)
        if tool in ("spotify_pause", "media_play_pause"):
            return IrisPlanner._persona_play_reply("")  # safe ack
        if tool in ("spotify_next", "media_next_track"):
            return "Skipping."
        if tool in ("spotify_previous", "media_previous_track"):
            return "Going back."
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
            # P2-INT-02 audit: was `Orchestrator._format_email_list`
            # — class doesn't exist; the actual class is IrisPlanner.
            # That NameError silently bypassed deterministic email
            # formatting and let the LLM hallucinate sender names.
            return IrisPlanner._format_email_list(result, args)
        if tool == "ollama_generate":
            # The generated text IS the user-facing response. Without this,
            # the haiku/poem/regex/etc. would get hidden behind
            # "Done (ollama_generate)." and the call would feel broken.
            text = (str(result.get("text") or result.get("response")
                        or result.get("output") or "")).strip()
            return text[:4000] if text else f"Done ({tool})."
        if tool == "google_whoami":
            nm = (result.get("name") or "").strip()
            em = (result.get("email") or "").strip()
            if nm and em:
                return f"You're signed in as {nm} ({em})."
            if em:
                return f"You're signed in as {em}."
            if nm:
                return f"You're signed in as {nm}."
            return f"Done ({tool})."
        if tool == "google_my_birthday":
            if result.get("set") is False:
                return str(result.get("message")
                           or "No birthday on your Google profile.")
            txt = (result.get("text") or "").strip()
            if txt:
                return f"Your birthday is {txt}."
            return f"Done ({tool})."
        if tool == "contacts_search":
            # Prefer the connector's pre-rendered summary if it exists —
            # avoids double-rendering when a connector has already produced
            # a good spoken summary.
            pre = str(result.get("summary") or "").strip()
            if pre:
                return pre
            contacts = result.get("contacts") or []
            count = result.get("count") or len(contacts)
            if not contacts:
                return f"No contacts matching \"{args.get('query')}\"."
            parts = []
            for c in contacts[:3]:
                nm = (c.get("display_name")
                      or " ".join(p for p in [c.get("given_name"),
                                              c.get("family_name")] if p)
                      or "Unnamed")
                emails = c.get("emails") or []
                phones = c.get("phones") or []
                detail = ""
                if emails and emails[0].get("value"):
                    detail = f" ({emails[0]['value']})"
                elif phones and phones[0].get("value"):
                    detail = f" ({phones[0]['value']})"
                parts.append(f"{nm}{detail}")
            label = "contact" if count == 1 else "contacts"
            return f"Found {count} {label}: {', '.join(parts)}."
        if tool == "contacts_list":
            # Always prefer the connector's deterministic summary. The
            # connector produces "You have N contacts: Alpha (alice@x.com,
            # +1-555-...); Beta (no email, +1-555-...); ..." with the
            # ACTUAL data — the LLM must never get a chance to fabricate
            # names/emails from a bland "Done (contacts_list)." reply
            # (previously observed: LLM invented john@example.com /
            # mariya@example.com placeholder pattern when asked to
            # elaborate on a follow-up turn).
            pre = str(result.get("summary") or "").strip()
            if pre:
                return pre
            contacts = result.get("contacts") or []
            count = result.get("count") or len(contacts)
            total = int(result.get("total") or count or 0)
            # If a prefix filter was applied, include it in the fallback
            # message so users don't hear "you have 3 contacts" when they
            # asked for "contacts starting with A" (with no letter → all
            # the LLM has to interpret is the number).
            prefix = str(result.get("filter_applied")
                         or args.get("starts_with") or "").strip()
            if not contacts:
                if prefix:
                    return f"No contacts starting with '{prefix}'."
                return "Your contacts list is empty."
            # Fallback path: mirror the connector's self-grounding format
            # so even when the summary field is somehow missing we still
            # emit explicit 'no email' / 'no phone' rather than a bare
            # name list the model can dress up with fabricated fields.
            shown = contacts[:30]
            parts = []
            for c in shown:
                nm = (c.get("display_name")
                      or " ".join(p for p in [c.get("given_name"),
                                              c.get("family_name")] if p)
                      or "(unnamed)")
                emails = c.get("emails") or []
                phones = c.get("phones") or []
                email_part = ((emails[0].get("value") if emails else None)
                              or "no email")
                phone_part = ((phones[0].get("value") if phones else None)
                              or "no phone")
                parts.append(f"{nm} ({email_part}, {phone_part})")
            window_note = (f" (showing {len(shown)} of {total})"
                           if total > len(shown) else "")
            label = "contact" if total == 1 else "contacts"
            if prefix:
                return (f"You have {total} {label} starting with "
                        f"'{prefix}'{window_note}: {'; '.join(parts)}.")
            return (f"You have {total} {label}{window_note}: "
                    f"{'; '.join(parts)}.")
        if tool == "contacts_create":
            return f"Added {result.get('display_name') or args.get('given_name')} to your contacts."
        if tool == "tasks_list":
            tasks = result.get("tasks") or []
            count = result.get("count") or len(tasks)
            if not tasks:
                return "No open tasks."
            titles = [str(t.get("title") or "").strip() or "(untitled)"
                      for t in tasks[:5]]
            label = "task" if count == 1 else "tasks"
            more = f" (+{count - len(titles)} more)" if count > len(titles) else ""
            return f"{count} {label}: {', '.join(titles)}{more}."
        if tool == "tasks_add":
            return f"Added task: {result.get('title') or args.get('title')}."
        if tool == "tasks_complete":
            return f"Marked done: {result.get('title') or args.get('title_match') or args.get('task_id')}."
        if tool == "tasks_delete":
            return f"Removed task: {result.get('title') or args.get('title_match') or args.get('task_id')}."
        if tool == "forms_create":
            link = result.get("link")
            title = result.get("title") or args.get("title")
            return f"Created form \"{title}\"" + (f": {link}" if link else ".")
        if tool == "forms_responses":
            count = result.get("count") or 0
            label = "response" if count == 1 else "responses"
            return f"{count} {label} received."
        if tool == "youtube_my_playlists":
            playlists = result.get("playlists") or []
            count = result.get("count") or len(playlists)
            if not playlists:
                return "You have no YouTube playlists."
            titles = [str(p.get("title") or "").strip() or "(untitled)"
                      for p in playlists[:3]]
            label = "playlist" if count == 1 else "playlists"
            more = f" (+{count - len(titles)} more)" if count > len(titles) else ""
            return f"You have {count} {label}: {', '.join(titles)}{more}."
        if tool == "youtube_playlist_items":
            items = result.get("items") or []
            count = result.get("count") or len(items)
            if not items:
                return "That playlist is empty."
            titles = [str(it.get("title") or "").strip() or "(untitled)"
                      for it in items[:3]]
            label = "video" if count == 1 else "videos"
            more = f" (+{count - len(titles)} more)" if count > len(titles) else ""
            return f"{count} {label}: {', '.join(titles)}{more}."
        if tool == "youtube_subscriptions":
            subs = result.get("subscriptions") or []
            count = result.get("count") or len(subs)
            if not subs:
                return "You have no YouTube subscriptions."
            titles = [str(s.get("title") or "").strip() or "(unnamed)"
                      for s in subs[:3]]
            label = "channel" if count == 1 else "channels"
            more = f" (+{count - len(titles)} more)" if count > len(titles) else ""
            return f"Subscribed to {count} {label}: {', '.join(titles)}{more}."
        if tool == "photos_upload":
            return f"Uploaded {result.get('name') or args.get('path')} to Google Photos."
        if tool == "gmail_send":
            summary = result.get("summary")
            if summary:
                return summary
            to = result.get("to") or args.get("to") or args.get("recipient")
            return f"Sent to {to}." if to else "Sent."
        # Generic safety net: if a tool isn't explicitly cased above but
        # produced a deterministic `summary` field, surface it verbatim
        # rather than the bland "Done (tool)." fallback. Two reasons:
        # (1) prevents "Done (contacts_list)." — style regressions where a
        # new tool's summary field is silently dropped, and (2) the LLM
        # never sees a bland reply and so can't fabricate elaborated
        # content on the next turn ("write them out" → invented emails).
        pre = str(result.get("summary") or "").strip()
        if pre:
            return pre
        return f"Done ({tool})."

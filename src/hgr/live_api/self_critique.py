"""Self-critique loop.

Phase-2 cognition. The reviser handles the "a step errored" case;
the critic handles the subtler "every step succeeded but the goal
wasn't actually met" case. Common shapes:

  * User asked "summarize my unread emails" — plan listed inbox,
    returned 0 unread. Critic should say: "user wants the answer,
    that IS the answer; speak it" (DONE).
  * User asked "find Alice's last message and reply" — read step
    found Alice's thread but the reply step never ran (deps got
    confused). Critic should append a reply step.
  * User asked "create a doc and email it to Dani" — doc created,
    email NOT sent because the planner forgot the send tool.
    Critic should append the email_send step.

Heuristic-only by default (no token cost). When OPENAI_API_KEY is
set AND the request actually merits an LLM critique (TOUCHLESS_
SELF_CRITIQUE_LLM=1 or the goal contains "and"/"then" verbs that
imply multi-action), the critic falls back to a cheap-LLM call
that proposes additional Steps.

Hard-capped to ONE critique pass per turn — set at the reviser
level via ReviserConfig.max_critiques.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from .planner.plan import Plan, Step, StepResult


# Goal-verbs that indicate a multi-action request the planner often
# under-plans. If we see these AND the executed plan has only one
# step, the critic should look closer.
_MULTI_ACTION_PATTERNS = [
    re.compile(r"\b(?:and|then|also|after that|next)\b\s+\w+",
               re.IGNORECASE),
    re.compile(r",\s*(?:then|and)\s+\w+", re.IGNORECASE),
]


# Verbs that imply an "ack" or "side-effect" step on top of a
# data-fetch step. If the goal contains these AND the plan has
# fetched but not acted, the critic suggests the corresponding tool.
_GOAL_VERBS_TO_TOOLS: dict = {
    "email": "gmail_send",
    "send": "gmail_send",
    "message": "slack_post",
    "post": "slack_post",
    "save": "drive_upload",
    "note": "notion_create_page",
    "log": "notion_append_to_page",
    "remind": "todo_add",
    "draft": "outlook_compose",
    "create doc": "gdocs_create",
    "create sheet": "sheets_create",
    "create slide": "slides_create",
    # Sheets-append fallback: catches the very common "create sheet X
    # and add/append headers Y, Z" phrasing where gpt-5-mini drops the
    # second step. `sheets_create` matches earlier and is skipped if
    # already ran, so these entries only fire when the append is the
    # missing action. Order matters: longer phrases first so "add
    # header" wins over "add" (which isn't listed) and doesn't false-
    # match "and". "append" is last for the bare form.
    "add headers": "sheets_append_rows",
    "add header": "sheets_append_rows",
    "add columns": "sheets_append_rows",
    "add column": "sheets_append_rows",
    "append": "sheets_append_rows",
}


@dataclass
class CritiqueResult:
    """What the critic decided."""
    # True if extra steps are needed and `augment_steps` lists them.
    needs_augmentation: bool = False
    augment_steps: List[Step] = None
    reason: str = ""

    def __post_init__(self):
        if self.augment_steps is None:
            self.augment_steps = []


class SelfCritique:
    """Decides whether the executed plan actually meets the goal."""

    def __init__(self, *, llm_planner: Optional[Any] = None) -> None:
        self._llm = llm_planner

    def critique(self, *, goal: str, plan: Plan,
                 results: List[StepResult]) -> CritiqueResult:
        if not goal or not results:
            return CritiqueResult(reason="no goal or no results")

        # All steps must be OK to merit critique (errors are the
        # reviser's job, not the critic's).
        if any(r.status != "ok" for r in results):
            return CritiqueResult(
                reason="critique skipped: errors present")

        # Heuristic: 1-step plan + multi-action goal → likely
        # under-planned.
        if (len(plan.steps) == 1
                and _looks_multi_action(goal)):
            inferred = self._infer_missing_step_from_goal(goal, plan, results)
            if inferred is not None:
                return CritiqueResult(
                    needs_augmentation=True,
                    augment_steps=[inferred],
                    reason="goal mentions multiple actions; one step ran",
                )

        # Heuristic: plan ran a search/list/get tool and returned
        # results, but the goal includes a side-effect verb. Suggest
        # the side-effect.
        if self._is_fetch_only_for_action_goal(goal, plan):
            inferred = self._infer_missing_step_from_goal(goal, plan, results)
            if inferred is not None:
                return CritiqueResult(
                    needs_augmentation=True,
                    augment_steps=[inferred],
                    reason="fetched data but missing the action step",
                )

        # Optional LLM critique (env-gated to keep cost flat by default).
        if (os.environ.get("TOUCHLESS_SELF_CRITIQUE_LLM", "0") == "1"
                and self._llm is not None):
            llm_aug = self._llm_critique(goal, plan, results)
            if llm_aug:
                return CritiqueResult(
                    needs_augmentation=True,
                    augment_steps=llm_aug,
                    reason="LLM critic proposed augmentation",
                )

        return CritiqueResult(reason="goal appears satisfied")

    # ---- helpers ------------------------------------------------------

    def _is_fetch_only_for_action_goal(self, goal: str, plan: Plan) -> bool:
        if len(plan.steps) > 2:
            return False
        tools = {s.tool for s in plan.steps}
        # Heuristic set of "fetch-only" tool prefixes.
        fetchy = ("gmail_list", "ms_mail_list", "notion_search",
                  "weather_get", "calendar_list", "drive_list",
                  "contacts_search", "iris_lookup")
        if not any(any(t.startswith(p) for p in fetchy) for t in tools):
            return False
        # Goal must contain an action verb.
        g = (goal or "").lower()
        return any(v in g for v in _GOAL_VERBS_TO_TOOLS.keys())

    def _infer_missing_step_from_goal(self, goal: str, plan: Plan,
                                      results: List[StepResult]
                                      ) -> Optional[Step]:
        """Cheap rule-based inference: which action tool is the
        goal asking for that the plan didn't run."""
        g = (goal or "").lower()
        ran_tools = {s.tool for s in plan.steps}
        for verb, tool in _GOAL_VERBS_TO_TOOLS.items():
            if verb in g and tool not in ran_tools:
                # Build a minimal step shell. Args are deliberately
                # under-specified; the executor will route it through
                # the same arg-resolver that handles {step:N.x} refs.
                args = self._infer_args_for_tool(tool, goal, results)
                if args is None:
                    # First matched verb couldn't be filled — try the
                    # next one instead of giving up on the whole goal.
                    # Fixes the case where an incidental substring
                    # like 'Email' in a header list matches 'email'
                    # first and shadows the real intent ('add headers').
                    continue
                return Step(
                    tool=tool,
                    args=args,
                    id=len(plan.steps) + 1,
                    layer="connector",
                    description=f"critic-added: {tool} for goal '{goal[:50]}'",
                )
        return None

    def _infer_args_for_tool(self, tool: str, goal: str,
                             results: List[StepResult]
                             ) -> Optional[dict]:
        """Best-effort args for a critic-suggested step. Returns
        None if we can't infer enough to make the step safe to
        run (the critic only adds steps when args can be filled)."""
        if tool in ("gmail_send", "ms_mail_send"):
            recipient = _extract_recipient_from_goal(goal)
            if not recipient:
                return None
            return {"to": recipient, "subject": "(automated follow-up)",
                    "body": "Forwarding the requested information.",
                    "_critic_added": True}
        if tool == "slack_post":
            channel = _extract_channel_from_goal(goal)
            if not channel:
                return None
            return {"channel": channel, "text": "(automated note)",
                    "_critic_added": True}
        if tool == "todo_add":
            return {"title": goal[:120], "_critic_added": True}
        if tool == "sheets_append_rows":
            # Pull the spreadsheet id from a prior sheets_create result
            # in this same plan; sheets_create returns it under 'id'
            # (see sheets_connector.py:216). Also accept the alternate
            # 'spreadsheet_id' key for robustness.
            sid = ""
            for r in results:
                if r.tool == "sheets_create" and isinstance(r.output, dict):
                    cand = r.output.get("id") or r.output.get("spreadsheet_id")
                    if cand:
                        sid = str(cand)
                        break
            if not sid:
                return None
            headers = _extract_headers_from_goal(goal)
            if not headers:
                return None
            return {"spreadsheet_id": sid, "rows": [headers],
                    "_critic_added": True}
        return None

    def _llm_critique(self, goal: str, plan: Plan,
                      results: List[StepResult]) -> List[Step]:
        """Ask the LLM planner: 'given this goal + executed plan,
        what step(s) should be appended to actually satisfy the goal?
        Or empty list if it's already satisfied.' Best-effort; on any
        failure return [].

        Tool outputs are wrapped in the content_quarantine envelope
        so attacker-controlled text in a tool result doesn't get
        treated as an instruction by the critic LLM (SEC-009 audit
        finding)."""
        try:
            from .content_quarantine import wrap as _quarantine_wrap
        except Exception:
            _quarantine_wrap = lambda text, source="", note="": text
        try:
            lines = []
            for r in results:
                short = _short(r.output if isinstance(r.output, dict) else {})
                wrapped = _quarantine_wrap(
                    short, source=f"tool_output:{r.tool}")
                lines.append(f"  - step {r.step_id} ({r.tool}):\n{wrapped}")
            results_summary = "\n".join(lines)
            critique_goal = (
                f"ORIGINAL GOAL: {goal}\n\n"
                f"PLAN EXECUTED (all OK):\n{results_summary}\n\n"
                "Question: do the executed steps actually satisfy the "
                "original goal? If NOT, produce a Plan with ONLY the "
                "additional steps needed to finish it. If yes, produce "
                "an empty plan (no steps)."
            )
            plan_or_none = self._llm.plan(critique_goal)
            if plan_or_none is None:
                return []
            return list(plan_or_none.steps or [])
        except Exception:
            return []


# ---- module helpers ----------------------------------------------------

def _looks_multi_action(goal: str) -> bool:
    if not goal:
        return False
    return any(p.search(goal) for p in _MULTI_ACTION_PATTERNS)


def _extract_recipient_from_goal(goal: str) -> str:
    if not goal:
        return ""
    # Try "email/send/message X" patterns.
    m = re.search(r"\b(?:email|send|message|to)\s+(\w[\w._-]{1,40})",
                  goal, re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        if cand.lower() in ("a", "an", "the", "to", "him", "her", "them"):
            return ""
        return cand
    return ""


def _extract_channel_from_goal(goal: str) -> str:
    if not goal:
        return ""
    m = re.search(r"#([\w-]{2,40})", goal)
    return f"#{m.group(1)}" if m else ""


def _extract_headers_from_goal(goal: str) -> List[str]:
    """Best-effort: pull a list of header/column names out of phrases
    like 'add headers Name, Email, Phone', 'append columns X and Y',
    'with headers foo, bar, baz'. Returns [] if nothing recognisable
    is present so the critic falls through (safer than guessing)."""
    if not goal:
        return []
    m = re.search(
        r"\b(?:headers?|columns?|fields?)\s+"
        r"([A-Za-z][A-Za-z0-9 ,_/&+.-]*?)"
        r"(?:[.!?]|$)",
        goal, re.IGNORECASE)
    if not m:
        return []
    raw = m.group(1).strip().rstrip(".,")
    # Split on commas and the trailing 'and X' / 'and Y' patterns.
    parts = re.split(r"\s*,\s*|\s+and\s+", raw)
    headers = [p.strip() for p in parts if p.strip()]
    # Clamp so a runaway match doesn't produce a 200-column header row.
    return [h for h in headers if len(h) <= 40][:20]


def _short(d: dict, limit: int = 160) -> str:
    if not isinstance(d, dict):
        return str(d)[:limit]
    keys = list(d.keys())[:6]
    # F-012 audit fix: parenthesize so `[:limit]` clamps the whole
    # concatenation. Previously it only sliced the trailing "}",
    # making the cap a no-op for many-keyed dicts.
    return ("{" + ", ".join(f"{k}=..." for k in keys) + "}")[:limit]

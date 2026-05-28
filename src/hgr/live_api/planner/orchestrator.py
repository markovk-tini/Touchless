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

from .classifier import Classifier
from .plan import Step


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

    def try_handle(self, text: str) -> Optional[Dict[str, Any]]:
        """Return {'step', 'result', 'source', 'message'} if handled, else None."""
        step = self._classifier.classify(text)
        if step is None:
            return None
        if self._registry is None:
            return None
        # Only fire when the tool is actually owned by an *available* connector
        # (or is a built-in). If a connector isn't ready, fall through to the
        # LLM rather than throwing a confusing error.
        if not self._registry.handles_connector(step.tool):
            # Don't try built-in tools at this stage — they often need richer
            # arg shapes the regex parser doesn't produce. Phase 2 covers them.
            return None
        # Confirmation hook for sends/destructive — not needed in Phase 1's
        # current intent set, but kept for forward-compat.
        if step.needs_confirm and self._confirm is not None:
            if not self._confirm(f"Run {step.tool}?", step.description):
                return {
                    "step": step, "source": "connector",
                    "result": {"status": "cancelled", "code": "user_declined"},
                    "message": "Cancelled."}
        try:
            result = self._registry.call(step.tool, step.args)
        except Exception as exc:
            if self._logger:
                self._logger.exception("iris_planner_call_failed", exc, tool=step.tool)
            return None  # let the LLM try
        source = "connector" if self._registry.handles_connector(step.tool) else "iris"
        return {
            "step": step,
            "result": result or {"status": "error", "error": "no result"},
            "source": source,
            "message": self._format_message(step, result or {}),
        }

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

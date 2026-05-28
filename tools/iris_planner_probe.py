"""Offline probe for the iris planner. Takes a prompt and shows what would
happen — which tier handles it, what the classifier matched, whether the
multi-action heuristic fired, what tools would be called — WITHOUT
spinning up the full Touchless app, the voice stack, or burning real
OpenAI tokens (no live LLM call by default).

Usage:
    python tools/iris_planner_probe.py "set volume to 30"
    python tools/iris_planner_probe.py "find Dani's email and send him hi"

    # Interactive mode — paste prompts one per line:
    python tools/iris_planner_probe.py

    # Live mode — actually calls the cheap-LLM planner (needs
    # OPENAI_API_KEY). Costs a fraction of a cent per prompt.
    python tools/iris_planner_probe.py --live "search for AI news and summarize"

Author: Konstantin Markov
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make 'hgr' importable when run from the repo root.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from hgr.live_api.planner.classifier import Classifier  # noqa: E402
from hgr.live_api.planner.triggers import (  # noqa: E402
    looks_multi_action, plan_needs_confirm, RISKY_TOOLS,
)


# ---- A tiny fake registry that just records calls --------------------------
class _ProbeRegistry:
    """Pretends to be a ToolRegistry. Returns canned 'ok' outputs and
    records every call so we can show the trail."""

    # The Phase 1 classifier checks handles_connector before firing —
    # accept the full known set so the probe doesn't drop matches.
    _KNOWN_CONNECTOR_TOOLS = frozenset({
        "volume_set", "volume_get", "volume_mute", "volume_toggle_mute",
        "discord_mute", "discord_toggle_mute", "discord_deafen",
        "discord_toggle_deafen",
        "todo_add",
        "gdocs_create", "sheets_create", "slides_create", "drive_upload",
        "outlook_compose", "ms_mail_send",
    })

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def handles_connector(self, tool: str) -> bool:
        return tool in self._KNOWN_CONNECTOR_TOOLS

    def call(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((tool, dict(args or {})))
        # Canned "ok" so downstream summary formatters work.
        if tool == "volume_get":
            return {"status": "ok", "percent": 42, "muted": False}
        if tool.endswith("_create"):
            return {"status": "ok", "link": "https://example/preview"}
        return {"status": "ok"}


# ---- Probe -----------------------------------------------------------------
def _probe(text: str, live: bool) -> None:
    classifier = Classifier()
    single = classifier.classify(text)
    multi = looks_multi_action(text)

    print("=" * 72)
    print(f"PROMPT: {text!r}")
    print(f"  classifier match : {single.tool if single else '— (no match)'}"
          + (f"  args={single.args}" if single else ""))
    print(f"  multi-action     : {multi}")
    print()

    # ---- Decide the tier the orchestrator WOULD pick. ----
    if single is not None and not multi:
        # Phase 1 wins.
        print("  ROUTING -> Tier 1 (deterministic classifier, 0 tokens)")
        reg = _ProbeRegistry()
        out = reg.call(single.tool, single.args)
        print(f"  Tool call         : {single.tool}({single.args})")
        print(f"  Mock result       : {out}")
        print(f"  User-visible reply: (Tier 1 _format_message would render here)")
        return

    # Phase 1 declined. Tier 2 gate.
    flag_on = os.environ.get("TOUCHLESS_IRIS_PLAN_LLM") == "1"
    if not (flag_on or multi):
        print("  ROUTING -> Tier R (realtime LLM)")
        print("  Why: classifier didn't match AND heuristic didn't trigger.")
        print("  No cheap-LLM call would be made. Realtime would handle it.")
        return

    print("  ROUTING -> Tier 2 (cheap-LLM plan + Executor)")
    print(f"  Why: classifier={'matched but multi-action' if single else 'no match'}"
          f", multi-action={multi}, opt-in flag={flag_on}")

    if not live:
        print("  (use --live to actually call the LLM and see the JSON plan)")
        return

    # Live mode: do the real cheap-LLM call.
    if not os.environ.get("OPENAI_API_KEY"):
        print("  --live requested but OPENAI_API_KEY is unset; skipping LLM call.")
        return

    from hgr.live_api.planner.planner_llm import LLMPlanner
    reg = _ProbeRegistry()
    planner = LLMPlanner(reg)
    plan = planner.plan(text)
    if plan is None:
        print("  LLM returned no usable plan (parse failure / API error / empty).")
        return

    print(f"  Plan goal         : {plan.goal!r}")
    print(f"  Plan.final        : {plan.final}")
    print(f"  Steps ({len(plan.steps)}):")
    for s in plan.steps:
        depinfo = f" deps={s.depends_on}" if s.depends_on else ""
        risk = "  ! RISKY" if s.tool in RISKY_TOOLS else ""
        print(f"    {s.id}. {s.tool}{depinfo}  args={s.args}{risk}")
    if plan_needs_confirm([s.tool for s in plan.steps]):
        print("  -> would surface ONE whole-plan confirm dialog before executing.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", help="prompt to probe (omit for stdin)")
    ap.add_argument("--live", action="store_true",
                    help="actually call cheap-LLM planner (needs OPENAI_API_KEY)")
    args = ap.parse_args()

    if args.prompt:
        _probe(" ".join(args.prompt), args.live)
        return

    # Interactive: one prompt per line.
    print("iris planner probe — type prompts (blank line / Ctrl-C to exit)")
    try:
        while True:
            line = input("> ").strip()
            if not line:
                break
            _probe(line, args.live)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()

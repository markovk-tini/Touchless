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


# ---- Synthetic catalog matching what a fully-connected user sees ----------
# In production the planner asks the real ToolRegistry for connector_catalog()
# (Microsoft Graph, Google, Discord, etc.). The probe doesn't run with those
# connectors authenticated, so we hand-mirror what the LLM would see in the
# live app — otherwise it falls back to GUI computer-use because no API tools
# show up.
_SYNTHETIC_CATALOG = [
    {"id": "ms_graph", "description": "Microsoft 365 (Outlook/Calendar/OneDrive/Teams/To Do/OneNote/Contacts)",
     "tools": ["ms_mail_send", "ms_mail_list", "ms_mail_search", "ms_mail_read",
               "ms_mail_mark_read", "ms_calendar_list", "ms_calendar_create",
               "onedrive_upload", "onedrive_list", "teams_send",
               "teams_channel_post", "excel_create", "excel_set_cell",
               "todo_add", "onenote_create", "contacts_search",
               "ms_list_accounts", "ms_use_account"]},
    {"id": "outlook", "description": "Outlook desktop quick-actions",
     "tools": ["outlook_compose", "email_send", "outlook_open"]},
    {"id": "gmail", "description": "Gmail send-only API",
     "tools": ["gmail_send"]},
    {"id": "drive", "description": "Google Drive / Docs / Sheets / Slides",
     "tools": ["drive_upload", "drive_list", "gdocs_create", "sheets_create",
               "slides_create"]},
    {"id": "google_calendar", "description": "Google Calendar",
     "tools": ["calendar_list_events", "calendar_create_event"]},
    {"id": "discord", "description": "Discord voice/system controls",
     "tools": ["discord_voice_status", "discord_mute", "discord_toggle_mute",
               "discord_deafen", "discord_toggle_deafen",
               "discord_join_voice", "discord_leave_voice"]},
    {"id": "volume", "description": "System audio (Windows Core Audio)",
     "tools": ["volume_set", "volume_get", "volume_mute", "volume_toggle_mute"]},
    {"id": "media", "description": "Media playback (play/pause/skip)",
     "tools": ["media_play_pause"]},
]
_SYNTHETIC_CONNECTOR_TOOLS = {t for c in _SYNTHETIC_CATALOG for t in c["tools"]}


# ---- A tiny fake registry that just records calls --------------------------
class _ProbeRegistry:
    """Pretends to be a ToolRegistry. Returns canned 'ok' outputs and
    records every call so we can show the trail."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def handles_connector(self, tool: str) -> bool:
        return tool in _SYNTHETIC_CONNECTOR_TOOLS

    def connector_catalog(self) -> List[Dict[str, Any]]:
        """What LLMPlanner._tool_catalog uses to render the prompt's
        'Available tools:' section. Mirrors the live app's connector list."""
        return _SYNTHETIC_CATALOG

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
    # Bypass the silent error swallow in LLMPlanner.plan so we can SEE what
    # actually went wrong (wrong model id, 401, JSON shape mismatch, etc.).
    # Also pass known_tools=None so the probe doesn't drop steps just because
    # the probe registry is a stub — we want to see the LLM's raw plan shape.
    try:
        messages = planner._build_messages(text)
        data = planner._call(messages)
        print(f"  Raw response keys : {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
        if isinstance(data, dict) and "steps" not in data:
            print(f"  Raw response     : {data}")
        plan = planner._parse(text, data, known_tools=None)
    except Exception as exc:
        print(f"  LLM call FAILED  : {type(exc).__name__}: {exc}")
        # HTTPError carries the response body — surface it.
        try:
            body = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else None  # type: ignore[attr-defined]
            if body:
                print(f"  Response body    : {body[:500]}")
        except Exception:
            pass
        print(f"  Using model      : {planner._model}")
        print(f"  Tip              : if model name is wrong, try setting "
              f"TOUCHLESS_PLANNER_MODEL=gpt-4o-mini")
        return
    if plan is None:
        print("  Parsed response had no usable steps. Raw keys:",
              list(data.keys()) if isinstance(data, dict) else type(data).__name__)
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

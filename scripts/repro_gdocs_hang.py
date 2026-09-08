"""Repro script for the 'create google doc' planner hang.

Measures wall-clock for three phrases through:
  - Stage A: deterministic Classifier (no LLM, no network)
  - Stage B: looks_multi_action() gate
  - Stage C: full IrisPlanner.try_handle() with a mocked registry (so the
             gdocs_create call returns instantly with predictable results)

The registry boundary is the cleanest mock point — IrisPlanner only calls
self._registry.call(...), so we don't need to patch GoogleClient.service()
itself. A fake registry that returns instantly for gdocs_create is enough
to expose any per-phrase orchestrator overhead (Tier 0 / 0.3 / 0.4 / 0.5
fast-path checks, classifier, conversationalize / prose_renderer, etc.).

If IrisPlanner construction fails (the orchestrator pulls in many singleton
modules — memory, scheduler, cot_layer, callback_engine, ...), Stage C is
skipped and Stage A alone still answers the bottleneck question.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

# --- Pre-flight environment hardening so the orchestrator's optional
#     features stay quiet: never call any real LLM, skip prose rendering
#     fallback paths that hit OpenAI, etc.
os.environ.pop("OPENAI_API_KEY", None)
os.environ["TOUCHLESS_IRIS_PLAN_LLM"] = "0"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


PHRASES: List[str] = [
    "make a google doc titled iris budget demo with body hello from iris",
    "make a google doc and write hello from iris in it",
    "make a google doc titled X with body Y and open it",
]


# ---------- Stage A: classifier only --------------------------------------
def time_classifier_only() -> List[Dict[str, Any]]:
    from hgr.live_api.planner.classifier import Classifier
    clf = Classifier()
    out: List[Dict[str, Any]] = []
    # warm-up so first-call import cost doesn't pollute the timing
    clf.classify("hello world")
    for phrase in PHRASES:
        # repeat 5x and take min to be robust to GC/jitter
        best = float("inf")
        step = None
        for _ in range(5):
            t0 = time.perf_counter()
            step = clf.classify(phrase)
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt
        out.append({
            "phrase": phrase,
            "classifier_ms": round(best * 1000, 3),
            "tool": getattr(step, "tool", None),
            "args": getattr(step, "args", None),
        })
    return out


# ---------- Stage B: multi-action gate -----------------------------------
def time_multi_action_gate() -> List[Dict[str, Any]]:
    from hgr.live_api.planner.triggers import looks_multi_action
    out: List[Dict[str, Any]] = []
    looks_multi_action("warm up")
    for phrase in PHRASES:
        best = float("inf")
        flag = None
        for _ in range(5):
            t0 = time.perf_counter()
            flag = looks_multi_action(phrase)
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt
        out.append({
            "phrase": phrase,
            "multi_action_ms": round(best * 1000, 3),
            "looks_multi": flag,
        })
    return out


# ---------- Stage C: full IrisPlanner.try_handle() ------------------------
class _FakeConnectors:
    """Stub for the ConnectorRegistry shape used by IrisPlanner via
    ToolRegistry.handles_connector / find_connector. We pretend to own
    only gdocs_create / gdocs_append_text / open_url."""
    _OWNED = {"gdocs_create", "gdocs_append_text", "open_url"}

    def handles(self, name: str) -> bool:
        return name in self._OWNED

    def find_by_id(self, name: str):  # iris_setup_tool fallback
        return None

    def is_available_for(self, name: str) -> Optional[str]:
        return None

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "gdocs_create":
            title = str(args.get("title") or "Untitled")
            text = str(args.get("text") or args.get("body") or "")
            return {
                "status": "ok",
                "created": True,
                "id": "FAKE_DOC_ID",
                "title": title,
                "link": "https://docs.google.com/document/d/FAKE_DOC_ID/edit",
                "chars": len(text),
            }
        if name == "gdocs_append_text":
            return {
                "status": "ok", "appended": True,
                "id": str(args.get("doc_id") or "FAKE_DOC_ID"),
                "chars": len(str(args.get("text") or "")),
            }
        if name == "open_url":
            return {"status": "ok", "opened": True,
                    "url": str(args.get("url_or_query") or "")}
        return {"status": "error", "error": f"no fake for {name}"}

    def available_tool_schemas(self, *, exclude_lazy: bool = False):
        return []

    def search(self, query, limit=3):
        return []

    def catalog(self):
        return []


class _FakeRegistry:
    """Minimal ToolRegistry stand-in that IrisPlanner uses through:
        handles_connector / call / find_connector / is_available"""

    def __init__(self) -> None:
        self._connectors = _FakeConnectors()
        self.calls: List[Tuple[str, Dict[str, Any], float]] = []

    def handles_connector(self, name: str) -> bool:
        return self._connectors.handles(name)

    def find_connector(self, name: str):
        return self._connectors.find_by_id(name)

    def is_available(self, name: str) -> Optional[str]:
        return None

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = self._connectors.execute(name, args)
        self.calls.append((name, dict(args or {}), time.perf_counter() - t0))
        return result


def _try_build_planner():
    """Construct an IrisPlanner with the fake registry, returning
    (planner, build_error). On any import / construction failure returns
    (None, traceback-string)."""
    try:
        from hgr.live_api.planner.orchestrator import IrisPlanner
        planner = IrisPlanner(registry=_FakeRegistry(), logger=None,
                              confirm=None, memory=None)
        return planner, None
    except Exception:
        return None, traceback.format_exc()


def time_full_orchestrator() -> Tuple[Optional[List[Dict[str, Any]]],
                                       Optional[str]]:
    planner, build_err = _try_build_planner()
    if planner is None:
        return None, build_err
    out: List[Dict[str, Any]] = []
    for phrase in PHRASES:
        # one timed run (try_handle isn't idempotent because it records
        # turns into memory / session buffer / etc.)
        t0 = time.perf_counter()
        try:
            reply = planner.try_handle(phrase)
            err = None
        except Exception:
            reply = None
            err = traceback.format_exc()
        wall = time.perf_counter() - t0
        # Snapshot of what the planner did
        reg = planner._registry  # type: ignore[attr-defined]
        last_call = reg.calls[-1] if reg.calls else None
        out.append({
            "phrase": phrase,
            "orchestrator_ms": round(wall * 1000, 3),
            "reply_msg": (str((reply or {}).get("message") or "")[:160]
                          if reply else None),
            "reply_steps": [getattr(s, "tool", None)
                            for s in (reply or {}).get("steps", [])],
            "reply_results": [
                {"tool": getattr(r, "tool", None),
                 "status": getattr(r, "status", None)}
                for r in (reply or {}).get("results", [])
            ],
            "fake_registry_calls": [
                {"tool": c[0], "args_keys": sorted(list(c[1].keys())),
                 "fake_call_ms": round(c[2] * 1000, 3)}
                for c in reg.calls
            ],
            "error": err,
        })
        # Reset the registry call log between phrases so each row only
        # shows the calls THIS phrase triggered.
        reg.calls.clear()
    return out, None


# ---------- main ----------------------------------------------------------
def _fmt(d: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in d.items() if k != "phrase")


def main() -> int:
    print("=" * 78)
    print("REPRO: gdocs planner hang")
    print("=" * 78)

    # Stage A
    try:
        a = time_classifier_only()
    except Exception:
        print("STAGE A (classifier) failed to import / run:")
        traceback.print_exc()
        return 1
    print("\n--- Stage A: Classifier (regex-only, no LLM, no network) ---")
    for row in a:
        flag = " <-- >2s" if row["classifier_ms"] > 2000 else ""
        print(f"  phrase: {row['phrase']!r}")
        print(f"    classifier_ms: {row['classifier_ms']}  "
              f"tool={row['tool']!r}  args={row['args']!r}{flag}")

    # Stage B
    try:
        b = time_multi_action_gate()
    except Exception:
        print("STAGE B (looks_multi_action) failed:")
        traceback.print_exc()
    else:
        print("\n--- Stage B: looks_multi_action() gate ---")
        for row in b:
            print(f"  phrase: {row['phrase']!r}")
            print(f"    multi_action_ms: {row['multi_action_ms']}  "
                  f"looks_multi={row['looks_multi']}")

    # Stage C
    c, build_err = time_full_orchestrator()
    if c is None:
        print("\n--- Stage C: IrisPlanner.try_handle() ---")
        print("  SKIPPED — could not construct IrisPlanner. Traceback:")
        print(build_err or "(no traceback)")
    else:
        print("\n--- Stage C: IrisPlanner.try_handle() with mock registry ---")
        for row in c:
            flag = " <-- >2s" if row["orchestrator_ms"] > 2000 else ""
            print(f"  phrase: {row['phrase']!r}")
            print(f"    orchestrator_ms: {row['orchestrator_ms']}{flag}")
            print(f"    reply_steps: {row['reply_steps']}")
            print(f"    reply_results: {row['reply_results']}")
            print(f"    fake_registry_calls: {row['fake_registry_calls']}")
            if row.get("reply_msg"):
                print(f"    reply_msg: {row['reply_msg']!r}")
            if row["error"]:
                print(f"    ERROR during try_handle:\n{row['error']}")

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("Phrases that exceed 2000 ms in any stage:")
    flagged: List[str] = []
    for row in a:
        if row["classifier_ms"] > 2000:
            flagged.append(f"  - {row['phrase']!r} (classifier "
                           f"{row['classifier_ms']} ms)")
    if c:
        for row in c:
            if row["orchestrator_ms"] > 2000:
                flagged.append(f"  - {row['phrase']!r} (orchestrator "
                               f"{row['orchestrator_ms']} ms)")
    if not flagged:
        print("  (none — every stage finished well under 2s)")
    else:
        for line in flagged:
            print(line)
    print()
    # Highest-cost stage
    longest_classifier = max(a, key=lambda r: r["classifier_ms"])
    print(f"Slowest classifier phrase: "
          f"{longest_classifier['classifier_ms']} ms  "
          f"-> {longest_classifier['phrase']!r}")
    if c:
        longest_orch = max(c, key=lambda r: r["orchestrator_ms"])
        print(f"Slowest orchestrator phrase: "
              f"{longest_orch['orchestrator_ms']} ms  "
              f"-> {longest_orch['phrase']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

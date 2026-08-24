"""Plan executor — runs a Plan's Steps in dependency order WITHOUT a model
call per step. Each step is executed by ToolRegistry (so connectors and
built-ins both work), with `{step:N.field}` references in args resolved from
earlier step outputs.

Phase 2 is sequential-but-dependency-aware (a step runs as soon as its deps
finish). Parallel execution of independent steps is a straightforward
extension; left for a follow-up.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .. import cortex_emit
from ..tool_invocation import (
    InvocationSource, ToolInvocation, publish as _publish_invocation,
)
from .plan import Plan, StepResult


# Map a tool name to a Cortex capability node id. The cortex web layer
# seeds three capability nodes around the core (cap-tools / cap-memory
# / cap-voice); pre-dispatch we pulse core -> the relevant capability so
# the user sees which "branch" of Iris is about to fire. Anything not
# listed defaults to cap-tools (the catch-all).
_TOOL_TO_CAP: Dict[str, str] = {
    # memory / contact / fact tools
    "iris_lookup_contact": "cap-memory",
    "iris_remember_contact": "cap-memory",
    "iris_forget_contact": "cap-memory",
    "iris_set_preference": "cap-memory",
    # voice / tts tools
    "tts_speak": "cap-voice",
    "voice_speak": "cap-voice",
    "speak": "cap-voice",
}


def _capability_for_tool(tool: str) -> str:
    """Resolve a tool name to its Cortex capability node id.

    Exact match wins; otherwise common name fragments route by category
    (memory / fact / contact -> cap-memory; voice / tts / speak ->
    cap-voice). Everything else falls back to cap-tools."""
    if not tool:
        return "cap-tools"
    if tool in _TOOL_TO_CAP:
        return _TOOL_TO_CAP[tool]
    name = tool.lower()
    if ("memory" in name or "fact" in name or "contact" in name
            or name.startswith("memory_")):
        return "cap-memory"
    if ("tts" in name or "voice" in name or name.startswith("speak")
            or name.endswith("_speak")):
        return "cap-voice"
    return "cap-tools"

# {step:N.field.subfield[i].leaf} → look up results[N].output.field.subfield[i].leaf
# Each segment is a word optionally followed by [N]; segments joined by dots.
_REF_RE = re.compile(
    r"\{step:(\d+)\.(?:output\.)?"
    r"(\w+(?:\[\d+\])?(?:\.\w+(?:\[\d+\])?)*)"
    r"\}"
)
_SEG_RE = re.compile(r"(\w+)(?:\[(\d+)\])?")

# Key aliases used when a {step:N.X} reference asks for a field that doesn't
# exist on the prior step's output. Common across connectors: gdocs_create
# returns 'link' but the planner LLM frequently writes '{step:N.url}'; the
# strict resolver was failing the second step ('open_url' got "" and erred
# 'unresolved reference {step:1.url}'). Tries aliases in order, stops at
# first non-None value. Bidirectional pairs are listed both ways so e.g.
# url->link AND link->url both work.
_KEY_ALIASES: Dict[str, tuple] = {
    "url": ("link", "web_url", "webViewLink", "share_url", "href"),
    "link": ("url", "web_url", "webViewLink", "share_url", "href"),
    "href": ("url", "link"),
    "id": ("key", "uid", "_id", "page_id", "doc_id", "file_id"),
    "key": ("id", "uid", "_id"),
    "text": ("response", "output", "content", "message", "body", "summary"),
    "response": ("text", "output", "content"),
    "content": ("text", "body", "response", "output"),
    "body": ("text", "content", "message", "summary"),
    "title": ("name", "subject", "label"),
    "name": ("title", "label"),
    "subject": ("title",),
    # Weather aliases — LLMs love to guess temp_f / temp / weather /
    # conditions / report which weather_get doesn't actually return.
    # Fall back to the closest real field (summary covers most cases
    # since it's already a full sentence).
    "temp_f": ("temperature", "summary"),
    "temp_c": ("temperature", "summary"),
    "temp": ("temperature", "summary"),
    "temperature_f": ("temperature",),
    "temperature_c": ("temperature",),
    "weather": ("summary", "description"),
    "conditions": ("description", "summary"),
    "condition": ("description", "summary"),
    "report": ("summary",),
    "forecast_summary": ("summary",),
}


class Executor:
    def __init__(self, registry: Any, logger: Any = None) -> None:
        self._registry = registry
        self._logger = logger

    # Max concurrent steps PER PLAN. Bounded so a 20-step plan doesn't
    # spawn 20 threads racing for the GIL + network. 4 is a sweet spot:
    # enough parallelism for typical multi-source plans (weather +
    # email + calendar in one turn → 3 steps in parallel), low enough
    # not to thrash. Override with TOUCHLESS_PLAN_PARALLELISM.
    _MAX_PARALLEL_STEPS = max(1, int(
        __import__("os").environ.get("TOUCHLESS_PLAN_PARALLELISM", "4")
        or "4"))

    def run(self, plan: Plan) -> List[StepResult]:
        results: Dict[int, StepResult] = {}
        remaining = list(plan.steps)
        order: List[StepResult] = []
        # Stable turn_id for this whole plan execution. All
        # ToolInvocations from this plan share it so the audit log /
        # activity pill can group them as "one user turn".
        import uuid as _uuid
        turn_id = f"plan:{_uuid.uuid4().hex[:12]}"
        while remaining:
            # Phase-1 speed pass: pick ALL ready steps (deps satisfied)
            # and run them concurrently via a bounded ThreadPoolExecutor.
            # The dep graph is already represented in step.depends_on;
            # we just stop pretending steps must run one at a time.
            #
            # Destructive steps (DESTRUCTIVE / IRREVERSIBLE per
            # tool_metadata) are bucketed separately and run
            # sequentially — racing two confirm dialogs would look broken,
            # and the confirms are rare enough that sequential is fine.
            ready_steps = [s for s in remaining
                           if all(d in results for d in s.depends_on)]
            if not ready_steps:
                # Unresolvable dep / cycle — fail the rest and stop.
                for s in remaining:
                    r = StepResult(step_id=s.id, tool=s.tool, status="error",
                                   error="unresolved dependency")
                    results[s.id] = r
                    order.append(r)
                break
            for s in ready_steps:
                remaining.remove(s)
            # Split: destructive → sequential, safe → parallel.
            from ..tool_metadata import is_destructive_or_worse
            destructive = [s for s in ready_steps
                           if is_destructive_or_worse(s.tool)]
            safe = [s for s in ready_steps
                    if not is_destructive_or_worse(s.tool)]
            # Run destructive sequentially first so their confirms
            # happen one-at-a-time and their results feed into the
            # parallel batch (deps are honored implicitly because both
            # buckets came from the same ready-set).
            for s in destructive:
                r = self._run_one_step(s, results, turn_id)
                results[s.id] = r
                order.append(r)
                if r.status == "error":
                    self._fail_dependents(s.id, remaining, results, order)
            # Now the parallel batch. ThreadPoolExecutor handles GIL
            # release on I/O-bound work (the typical Iris tool — HTTP,
            # disk, Win32). For purely-CPU tools (rare) we still serialize
            # naturally via the GIL but at least don't add overhead.
            if safe:
                self._run_parallel(safe, results, order, turn_id, remaining)
        return order

    def _run_one_step(self, ready, results: Dict[int, "StepResult"],
                      turn_id: str) -> "StepResult":
        """Encapsulated per-step execution: resolve refs, precondition,
        safety gate, tool dispatch, invocation emit. Returns the
        StepResult; caller is responsible for adding to results+order
        and for failing dependents on error. Thread-safe IFF the tool
        impls are thread-safe (most are — see _run_parallel)."""
        args, unresolved = self._resolve_tracked(ready.args, results)
        # Ref-resolution failure.
        if unresolved:
            out = {"status": "error",
                   "error": f"unresolved reference {unresolved[0]} "
                            f"(prior step returned no matching data)",
                   "code": "ref_unresolved"}
            r = StepResult(step_id=ready.id, tool=ready.tool,
                           status="error", output=out, error=out["error"])
            self._emit_invocation(r, ready, args, turn_id)
            return r
        # Precondition check.
        unavail_reason: Optional[str] = None
        checker = getattr(self._registry, "is_available", None)
        if callable(checker):
            try:
                unavail_reason = checker(ready.tool)
            except Exception:
                unavail_reason = None
        if unavail_reason:
            out = {"status": "error", "error": unavail_reason,
                   "code": "precondition_not_met"}
        else:
            from ..safety_gate import gate as _safety_gate
            # Planner-path source: this tool is being run by the
            # deterministic plan executor, not the realtime model.
            # The spoof-defense layer in safety_gate only fires for
            # source="voice"/"realtime"; planner-path destructive ops
            # still hit the typed-confirm gate via needs_confirmation.
            allowed, decline = _safety_gate(ready.tool, args,
                                            source="planner")
            if not allowed:
                out = {"status": "cancelled",
                       "code": "user_declined_speed_bump",
                       "error": decline or "user declined"}
            else:
                # Cortex viz: pulse core -> capability node.
                try:
                    cap = _capability_for_tool(ready.tool)
                    cortex_emit.edge_pulse("core", cap,
                                           color="magenta", duration_ms=110)
                except Exception:
                    pass
                try:
                    out = self._registry.call(ready.tool, args)
                    if not isinstance(out, dict):
                        out = {"status": "error",
                               "error": "tool returned non-dict"}
                except Exception as exc:
                    if self._logger:
                        self._logger.exception("planner_exec_failed", exc,
                                               tool=ready.tool)
                    out = {"status": "error",
                           "error": f"{type(exc).__name__}: {exc}"}
        r = StepResult(step_id=ready.id, tool=ready.tool,
                       status=str(out.get("status") or "ok"),
                       output=out, error=out.get("error"))
        self._emit_invocation(r, ready, args, turn_id)
        return r

    def _emit_invocation(self, r, ready, args, turn_id: str) -> None:
        try:
            inv = ToolInvocation.starting(
                tool=ready.tool, args=args,
                source=InvocationSource.PLANNER,
                turn_id=turn_id,
                was_confirmed=bool(ready.needs_confirm),
            ).complete(status=r.status, output=r.output, error=r.error)
            _publish_invocation(inv)
        except Exception as exc:
            if self._logger:
                self._logger.exception("invocation_publish_failed", exc)

    def _fail_dependents(self, failed_id: int, remaining: List,
                         results: Dict[int, "StepResult"],
                         order: List) -> None:
        """Transitively mark every still-pending step that depends on a
        failed step as error. Mutates `remaining` (removes the failed
        ones) and appends to `results`/`order`."""
        failed = {failed_id}
        changed = True
        while changed:
            changed = False
            still: List = []
            for s in remaining:
                if any(d in failed for d in s.depends_on):
                    er = StepResult(step_id=s.id, tool=s.tool,
                                    status="error",
                                    error="upstream step failed")
                    results[s.id] = er
                    order.append(er)
                    failed.add(s.id)
                    changed = True
                else:
                    still.append(s)
            remaining[:] = still

    def _run_parallel(self, steps: List, results: Dict[int, "StepResult"],
                      order: List, turn_id: str, remaining: List) -> None:
        """Run multiple ready steps concurrently using a bounded thread
        pool. Each step's _run_one_step is run on a worker; results
        are collected and merged into `results`/`order` in COMPLETION
        ORDER (not submission order — preserves the user-observable
        execution-completed-at sequence in the audit log, which is
        what the activity pill renders)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = min(self._MAX_PARALLEL_STEPS, len(steps))
        if max_workers <= 1:
            # Avoid spinning up a pool for a single step — saves
            # ~ms of overhead.
            for s in steps:
                r = self._run_one_step(s, results, turn_id)
                results[s.id] = r
                order.append(r)
                if r.status == "error":
                    self._fail_dependents(s.id, remaining, results, order)
            return
        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="iris-plan-step"
                                ) as pool:
            futures = {pool.submit(self._run_one_step, s, results, turn_id): s
                       for s in steps}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # pragma: no cover - defensive
                    r = StepResult(step_id=s.id, tool=s.tool,
                                   status="error",
                                   error=f"worker raised: "
                                         f"{type(exc).__name__}: {exc}")
                    self._emit_invocation(r, s, s.args, turn_id)
                results[s.id] = r
                order.append(r)
                if r.status == "error":
                    self._fail_dependents(s.id, remaining, results, order)

    @classmethod
    def _resolve_tracked(cls, args: Any,
                         results: Dict[int, StepResult]) -> tuple:
        """Same as _resolve but ALSO returns a list of refs that didn't
        resolve. Used by run() to fail a step with a clear error instead
        of letting the downstream tool see a silently-empty arg."""
        unresolved: List[str] = []
        resolved = cls._resolve(args, results, unresolved)
        return resolved, unresolved

    # ---- {step:N.field} reference resolution ------------------------------
    @classmethod
    def _resolve(cls, args: Any, results: Dict[int, StepResult],
                 unresolved: Optional[List[str]] = None) -> Any:
        if isinstance(args, dict):
            return {k: cls._resolve(v, results, unresolved) for k, v in args.items()}
        if isinstance(args, list):
            return [cls._resolve(v, results, unresolved) for v in args]
        if isinstance(args, str):
            def sub(m):
                ref_text = m.group(0)
                step_id = int(m.group(1))
                path = m.group(2)
                def fail() -> str:
                    if unresolved is not None and ref_text not in unresolved:
                        unresolved.append(ref_text)
                    return ""
                r = results.get(step_id)
                if r is None:
                    return fail()
                val: Any = r.output
                for segment in path.split("."):
                    sm = _SEG_RE.match(segment)
                    if sm is None:
                        return fail()
                    key, idx = sm.group(1), sm.group(2)
                    if isinstance(val, dict):
                        next_val = val.get(key)
                        if next_val is None:
                            # Try aliases — e.g. {step:N.url} when the
                            # output actually has 'link'. Keeps the strict
                            # behavior when the key DOES exist (None values
                            # still fail below).
                            for alias in _KEY_ALIASES.get(key, ()):
                                if alias in val and val[alias] is not None:
                                    next_val = val[alias]
                                    break
                        val = next_val
                    else:
                        return fail()
                    if val is None:
                        return fail()
                    if idx is not None:
                        i = int(idx)
                        if not isinstance(val, list) or not (0 <= i < len(val)):
                            return fail()
                        val = val[i]
                # Lists / dicts get JSON-stringified so downstream tools (esp.
                # compose_text) receive a parseable structure instead of
                # Python's repr ('[{'id': ...}]' with single quotes).
                if isinstance(val, (list, dict)):
                    try:
                        return json.dumps(val, default=str)
                    except Exception:
                        pass
                return str(val)
            return _REF_RE.sub(sub, args)
        return args

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

from .plan import Plan, StepResult

# {step:N.field.subfield[i].leaf} → look up results[N].output.field.subfield[i].leaf
# Each segment is a word optionally followed by [N]; segments joined by dots.
_REF_RE = re.compile(
    r"\{step:(\d+)\.(?:output\.)?"
    r"(\w+(?:\[\d+\])?(?:\.\w+(?:\[\d+\])?)*)"
    r"\}"
)
_SEG_RE = re.compile(r"(\w+)(?:\[(\d+)\])?")


class Executor:
    def __init__(self, registry: Any, logger: Any = None) -> None:
        self._registry = registry
        self._logger = logger

    def run(self, plan: Plan) -> List[StepResult]:
        results: Dict[int, StepResult] = {}
        remaining = list(plan.steps)
        order: List[StepResult] = []
        while remaining:
            ready = next(
                (s for s in remaining if all(d in results for d in s.depends_on)),
                None,
            )
            if ready is None:
                # Unresolvable dep / cycle — fail the rest and stop.
                for s in remaining:
                    r = StepResult(step_id=s.id, tool=s.tool, status="error",
                                   error="unresolved dependency")
                    results[s.id] = r
                    order.append(r)
                break
            remaining.remove(ready)
            args, unresolved = self._resolve_tracked(ready.args, results)
            # Ref-resolution failure: a {step:N.field} pointed at nothing
            # (e.g. step 1 returned 0 contacts; downstream {step:1.contacts
            # [0].emails[0]} → ""). Fail the step here with the SPECIFIC
            # ref that didn't resolve, instead of letting the connector
            # error out with a cryptic '"to" is required' message.
            if unresolved:
                out = {"status": "error",
                       "error": f"unresolved reference {unresolved[0]} "
                                f"(prior step returned no matching data)",
                       "code": "ref_unresolved"}
                r = StepResult(step_id=ready.id, tool=ready.tool,
                               status="error", output=out,
                               error=out["error"])
                results[ready.id] = r
                order.append(r)
                # Fail dependents too (same logic as below).
                failed = {ready.id}
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
                    remaining = still
                continue
            # Precondition check (Phase 6): short-circuit unavailable tools
            # with a clear reason instead of letting the connector fail mid-
            # call. Registries that don't implement is_available are skipped.
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
                try:
                    out = self._registry.call(ready.tool, args)
                    if not isinstance(out, dict):
                        out = {"status": "error", "error": "tool returned non-dict"}
                except Exception as exc:
                    if self._logger:
                        self._logger.exception("planner_exec_failed", exc, tool=ready.tool)
                    out = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            r = StepResult(step_id=ready.id, tool=ready.tool,
                           status=str(out.get("status") or "ok"),
                           output=out, error=out.get("error"))
            results[ready.id] = r
            order.append(r)
            # Hard-fail short-circuit: when a step errors, transitively fail
            # every step (still pending) that depends on it. Independent steps
            # keep running.
            if r.status == "error":
                failed = {ready.id}
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
                    remaining = still
        return order

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
                        val = val.get(key)
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

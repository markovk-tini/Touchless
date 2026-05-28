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

import re
from typing import Any, Dict, List, Optional

from .plan import Plan, StepResult

# {step:N.field.subfield} → look up results[N].output.field.subfield (str).
_REF_RE = re.compile(r"\{step:(\d+)\.(?:output\.)?([\w.]+)\}")


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
            args = self._resolve(ready.args, results)
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

    # ---- {step:N.field} reference resolution ------------------------------
    @classmethod
    def _resolve(cls, args: Any, results: Dict[int, StepResult]) -> Any:
        if isinstance(args, dict):
            return {k: cls._resolve(v, results) for k, v in args.items()}
        if isinstance(args, list):
            return [cls._resolve(v, results) for v in args]
        if isinstance(args, str):
            def sub(m):
                step_id = int(m.group(1))
                path = m.group(2)
                r = results.get(step_id)
                if r is None:
                    return ""
                val: Any = r.output
                for key in path.split("."):
                    if isinstance(val, dict):
                        val = val.get(key)
                    else:
                        return ""
                    if val is None:
                        return ""
                return str(val)
            return _REF_RE.sub(sub, args)
        return args

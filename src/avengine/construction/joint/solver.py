"""Bounded, deterministic joint candidate search with explicit failure accounting."""
from collections import Counter
import math
import time
from .types import Certificate, CheckResult, PreparedCandidate, SolveResult
from .certification import certify_plan


def solve(context, request, plugin, *, budget_s=120., max_candidates=512, prepare_and_verify=None,
          max_native_preparations=8):
    if not math.isfinite(budget_s) or budget_s <= 0 or max_candidates < 1:
        raise ValueError("positive finite search budget required")
    if not request.family or not request.branch:
        raise ValueError("family and desired answer branch must be fixed before search")
    if request.options.get("require_native_preflight") and prepare_and_verify is None:
        return SolveResult("unavailable", reasons=("required_native_preflight_not_supplied",))
    started = time.monotonic()
    native_preparations = 0
    native_seconds = 0.
    failures = Counter()
    last = None
    count = 0
    status = "unavailable"
    candidates = iter(plugin.candidates(context, request))
    while count < max_candidates:
        if time.monotonic()-started >= budget_s:
            status = "budget_exhausted"
            break
        try:
            plan = next(candidates)
        except StopIteration:
            break
        count += 1
        if time.monotonic()-started >= budget_s:
            status = "budget_exhausted"
            break
        common = certify_plan(context, plan)
        requested = CheckResult("requested_targets", "pass" if set(request.target_ids) <= set(plan.actors) else "fail")
        family = tuple(plugin.verify(context, request, plan)) if common.passed else ()
        if common.passed and not family:
            family = (CheckResult("family_conditions", "unknown", "plugin_did_not_supply_answer_checks"),)
        last = Certificate(common.checks + (requested,) + family)
        prepared = None
        if last.passed and prepare_and_verify is not None:
            if native_preparations >= max_native_preparations:
                status = "budget_exhausted"
                failures["native_preparation_budget"] += 1
                break
            native_started = time.monotonic()
            prepared = prepare_and_verify(context, request, plan)
            native_seconds += time.monotonic()-native_started
            native_preparations += 1
            if not isinstance(prepared, PreparedCandidate):
                raise TypeError("native callback must return PreparedCandidate with explicit checks")
            native_checks = tuple(prepared.checks) or (CheckResult("native_preflight", "unknown", "no_native_checks"),)
            last = Certificate(last.checks+native_checks, "planning_and_native_preflight")
            if not last.passed and prepared.close is not None:
                prepared.close()
                prepared = None
        for check in last.checks:
            if check.status != "pass":
                failures[check.name+":"+(check.reason or check.status)] += 1
        if last.passed:
            plan.metadata["joint_solution"] = {
                "schema": "avengine_joint_solution_v1", "solver_entry": "avengine.construction.joint.solve",
                "family": request.family, "branch": request.branch, "target_ids": list(request.target_ids),
                "geometry_certificate_status": "pass",
                "native_preflight_status": "pass" if prepared is not None else "not_run",
                "same_prepared_payload_required": prepared is not None,
                "native_full_render_and_qa_verified": False,
                "candidates_evaluated": count, "native_preparations": native_preparations}
            if prepared is not None and hasattr(prepared.payload,"bind_solution"):
                try:
                    prepared.payload.bind_solution(plan.metadata["joint_solution"])
                except Exception:
                    if prepared.close is not None:
                        prepared.close()
                    raise
            return SolveResult("certified", plan, last, diagnostics={"candidates": count,
                "elapsed_s": time.monotonic()-started, "family": request.family, "branch": request.branch,
                "native_render_verified": False, "failure_counts": dict(failures),
                "native_preparations": native_preparations, "native_preparation_s": native_seconds},
                prepared=prepared)
    else:
        status = "budget_exhausted"
    return SolveResult(status, certificate=last,
        reasons=tuple(failures) or ("no_candidates_in_current_domain",),
        diagnostics={"candidates": count, "elapsed_s": time.monotonic()-started, "family": request.family,
                     "branch": request.branch, "failure_counts": dict(failures),
                     "continuous_room_infeasibility_proved": False,
                     "native_preparations": native_preparations, "native_preparation_s": native_seconds})

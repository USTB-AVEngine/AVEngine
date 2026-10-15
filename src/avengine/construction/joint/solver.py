"""Bounded, deterministic joint candidate search with explicit failure accounting."""
from collections import Counter
import math
import time
from .types import Certificate, CheckResult, SolveResult
from .certification import certify_plan


def solve(context, request, plugin, *, budget_s=120., max_candidates=512):
    if not math.isfinite(budget_s) or budget_s <= 0 or max_candidates < 1:
        raise ValueError("positive finite search budget required")
    if not request.family or not request.branch:
        raise ValueError("family and desired answer branch must be fixed before search")
    started = time.monotonic()
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
        for check in last.checks:
            if check.status != "pass":
                failures[check.name+":"+(check.reason or check.status)] += 1
        if last.passed:
            return SolveResult("certified", plan, last, diagnostics={"candidates": count,
                "elapsed_s": time.monotonic()-started, "family": request.family, "branch": request.branch,
                "native_render_verified": False, "failure_counts": dict(failures)})
    else:
        status = "budget_exhausted"
    return SolveResult(status, certificate=last,
        reasons=tuple(failures) or ("no_candidates_in_current_domain",),
        diagnostics={"candidates": count, "elapsed_s": time.monotonic()-started, "family": request.family,
                     "branch": request.branch, "failure_counts": dict(failures),
                     "continuous_room_infeasibility_proved": False})

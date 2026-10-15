"""Bounded joint search, filled after the public protocol handoff."""
from .types import SolveResult


def solve(context, request, plugin, *, budget_s=120., max_candidates=512):
    """Return a certified plan or an explicit failure; never random retries."""
    return SolveResult("unavailable", reasons=("joint_core_implementation_pending",),
                       diagnostics={"family": request.family, "branch": request.branch})

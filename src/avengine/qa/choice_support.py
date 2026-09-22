"""Keep useful questions without manufacturing MCQ distractors."""
from copy import deepcopy

INTRINSIC_DOMAINS = {
    frozenset(("yes", "no")), frozenset(("left", "right")),
    frozenset(("moving", "still")), frozenset(("nearer", "farther")),
    # QA-13 asks which of three declared angular bands contains the source.
    # A fourth band would change its meaning, so report the 1/3 baseline.
    frozenset(("fov_band_0", "fov_band_1", "fov_band_2")),
}


def apply_choice_support(item, *, minimum=4):
    """At bank export retain open answers when fewer than four real foils exist.

    Intrinsic finite domains retain their explicit chance baseline. Historical
    generator callers can still emit their original forms; this policy is
    applied to new exported banks, never to old artifacts in place.
    """
    result = deepcopy(item)
    form = result.get("forms", {}).get("mcq")
    if not form:
        return result
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        raise ValueError("minimum MCQ option count must be at least two")
    options = form.get("options", [])
    n = len(options)
    intrinsic = frozenset(str(o.get("value")) for o in options) in INTRINSIC_DOMAINS
    report = {"minimum_extensional_options": minimum, "real_candidate_count": n,
              "intrinsic_domain": intrinsic, "chance_baseline": 1/n if n else None,
              "action": "retained"}
    if n < minimum and not intrinsic:
        if "open" in result.get("forms", {}):
            result["forms"].pop("mcq", None)
            result.get("model_input", {}).pop("mcq", None)
            result.setdefault("form_status", {})["mcq"] = {
                "status": "deferred", "code": "insufficient_real_mcq_candidates",
                "detail": f"{n} real candidates; retain the valid open form instead of padding to {minimum}",
            }
            report["action"] = "open_only"
        else:
            # Do not throw away a usable sole form merely to meet a reporting
            # threshold. The bank receipt still marks its limited support.
            report["action"] = "retained_only_form_under_powered"
    result.setdefault("evidence", {})["mcq_support"] = report
    if isinstance(result.get("truth"), dict):
        result["truth"].setdefault("evidence", {})["mcq_support"] = deepcopy(report)
    return result

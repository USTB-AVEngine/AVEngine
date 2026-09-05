"""Reject a mesh preparation that damaged the animal, before it is rigged.

Only two readings here are route-independent enough to decide on, and both
describe damage the reduction itself did:

  head-end survival      how much of the front third's triangle share survived,
                         normalised so 1.0 is "lost the same share as the mesh
                         overall". A face collapsed into flat facets reads 0.53
                         where three good assets read 1.18, 1.23 and 1.39. It
                         only works because the reviewed forward direction says
                         which third is the head - an axis-blind version scored
                         0.51 on the ruined mesh and 0.507 on one that had
                         merely thinned its tail.
  face target reached    a mesh whose non-manifold edges block collapse never
                         gets there, and that is worth knowing before rigging.

Everything else is reported and not judged, because it turned out to depend on
which route produced the mesh rather than on whether the result is good:

  faceting               punishes a low face count. The coarsest asset in this
                         batch scored worst on it (0.657) and looked best; a
                         denser one scored 0.383 and looked speckled. Face count
                         and dihedral angle are not independent.
  boundary, non-manifold a remesh drives both to zero, so there is nothing to
                         check; a plain weld-and-collapse inherits whatever the
                         reconstruction had, which is a property of the source
                         and not a defect this step introduced. Two assets that
                         look right carry 1,052 and 1,986 non-manifold edges.
  fidelity               spans fivefold across assets that all look right.

Whether the result actually looks right is decided after rigging, from how the
surface deforms - see gate_rigged_asset.py. That is what lets a pipeline try the
cheap reduction first and stop when it is already good enough.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from collections.abc import Mapping, Sequence

try:
    from animal_review_policy import (
        STRATEGY_NAMES,
        AnimalReviewPolicyError,
        load_review_policy,
    )
except ModuleNotFoundError:  # imported as tools.assets.gate_retopology
    from tools.assets.animal_review_policy import (  # type: ignore[no-redef]
        STRATEGY_NAMES,
        AnimalReviewPolicyError,
        load_review_policy,
    )



class RetopologyGateError(ValueError):
    """The report cannot provide a structurally complete prepared mesh."""


def _finite_tree(value, owner="report"):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RetopologyGateError(f"{owner} contains NaN or Inf")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{owner}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            _finite_tree(child, f"{owner}[{index}]")
        return
    raise RetopologyGateError(f"{owner} contains an unsupported value")


def _number(value, owner, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetopologyGateError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise RetopologyGateError(f"{owner} must be a finite number")
    return result


def _read_report(path):
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise RetopologyGateError(f"report is unreadable: {candidate}")
    try:
        report = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RetopologyGateError(f"report is unreadable: {candidate}") from error
    if not isinstance(report, dict):
        raise RetopologyGateError("report must contain a JSON object")
    _finite_tree(report)
    return candidate.resolve(), report


def _readable_mesh(path, report_path):
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = report_path.parent / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise RetopologyGateError(f"mesh output is unreadable: {candidate}")
    try:
        if candidate.stat().st_size <= 0:
            raise RetopologyGateError(f"mesh output is empty: {candidate}")
        with candidate.open("rb") as handle:
            handle.read(1)
    except OSError as error:
        raise RetopologyGateError(f"mesh output is unreadable: {candidate}") from error
    return candidate.resolve()


def _structural(report_path, report):
    stages = report.get("stages")
    if not isinstance(stages, Mapping):
        raise RetopologyGateError("report lacks mesh stages")
    final = stages.get("decimated")
    if not isinstance(final, Mapping):
        raise RetopologyGateError("report lacks decimated mesh stage")
    faces = _number(final.get("faces"), "stages.decimated.faces", positive=True)
    if "verts" in final:
        _number(final.get("verts"), "stages.decimated.verts", positive=True)
    target = _number(report.get("target_faces"), "target_faces", positive=True)
    mesh = report.get("mesh")
    if isinstance(mesh, Mapping):
        if mesh.get("valid") is False:
            raise RetopologyGateError("report marks the prepared mesh invalid")
        if mesh.get("finite_coordinates") is False:
            raise RetopologyGateError("prepared mesh coordinates are not finite")
    if report.get("mesh_valid") is False:
        raise RetopologyGateError("report marks the prepared mesh invalid")
    output = report.get("output")
    if isinstance(output, Mapping):
        output = output.get("path")
    if not isinstance(output, str) or not output.strip():
        raise RetopologyGateError("report lacks an output mesh path")
    output_path = _readable_mesh(output, report_path)
    return {
        "faces": faces,
        "target_faces": target,
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--policy-config", "--config", dest="policy_config", type=Path)
    parser.add_argument("--strategy", choices=STRATEGY_NAMES)
    parser.add_argument("--min-head-survival", type=float)
    parser.add_argument("--face-tolerance", type=float)
    args = parser.parse_args(argv)

    try:
        policy = load_review_policy(args.policy_config, strategy=args.strategy)
        report_path, report = _read_report(args.report)
        structural = _structural(report_path, report)
    except (AnimalReviewPolicyError, RetopologyGateError) as error:
        print("RETOPOLOGY_GATE_HARD_FAIL " + json.dumps(
            {"failures": [str(error)]}, ensure_ascii=False))
        return 1
    final = report["stages"]["decimated"]
    bands = report.get("band_survival")
    target = structural["target_faces"]
    metrics = policy["metrics"]["retopology"]
    minimum_head = (
        metrics["min_head_survival"]
        if args.min_head_survival is None else args.min_head_survival
    )
    face_tolerance = (
        metrics["face_tolerance"]
        if args.face_tolerance is None else args.face_tolerance
    )
    try:
        minimum_head = _number(minimum_head, "minimum head survival")
        face_tolerance = _number(face_tolerance, "face tolerance")
        if minimum_head < 0.0 or face_tolerance < 0.0:
            raise RetopologyGateError(
                "metric thresholds must be finite and non-negative"
            )
    except RetopologyGateError as error:
        print("RETOPOLOGY_GATE_HARD_FAIL " + json.dumps(
            {"failures": [str(error)]}, ensure_ascii=False))
        return 1

    failures = []
    metric_failures = []
    if not isinstance(bands, Mapping):
        metric_failures.append("no band survival in the report")
    else:
        try:
            front_survival = _number(
                bands.get("front"), "band_survival.front"
            )
        except RetopologyGateError as error:
            metric_failures.append(str(error))
        else:
            if front_survival < minimum_head:
                metric_failures.append(
                    f"head-end survival {front_survival} < {minimum_head}: the "
                    "front third paid for the rest of the body")
    if abs(final["faces"] - target) > target * face_tolerance:
        metric_failures.append(
            f"faces {final['faces']} missed target {target}: the reduction was "
            "blocked, most likely by non-manifold edges collapse cannot touch")
    if policy["gate_metrics"]:
        failures.extend(metric_failures)

    verdict = {
        "report": args.report,
        "faces": final["faces"],
        "band_survival": bands,
        "policy_id": policy["policy_id"],
        "strategy": policy["strategy"],
        "status": (
            "accepted_by_metrics"
            if policy["gate_metrics"] and not failures
            else "needs_visual_review"
            if not policy["gate_metrics"]
            else "rejected_by_metrics"
        ),
        "structural": structural,
        "advisory": {
            "faceting_share_over_30deg": report.get("faceting", {}).get(
                "share_over_30deg"),
            "boundary": final.get("boundary"),
            "nonmanifold": final.get("nonmanifold"),
            "fidelity_p99": (report.get("fidelity_over_diagonal") or {}).get("p99"),
            "source_debris_share": report.get("source_debris_share"),
            "relief_ratio": report.get("relief_ratio"),
            "relief_smooth_iterations": report.get("relief_smooth_iterations"),
        },
        "metric_failures": metric_failures,
        "advisory_failures": [] if policy["gate_metrics"] else metric_failures,
        "failures": failures,
    }
    if failures:
        print("RETOPOLOGY_GATE_REJECT " + json.dumps(verdict, ensure_ascii=False))
        return 1
    print("RETOPOLOGY_GATE_OK " + json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Accept or reject a rigged animal from how its surface tears during the walk.

This is the reading that decides whether an asset looks right, and it is separate
from the pre-rig gate on purpose: the two failure modes do not show up in the
same number. A standard Burmese whose face collapsed into flat facets tears less
than a Jack Russell that looks fine (1.07 against 1.27 percent of posed area in
shards), so no tearing threshold will ever catch it - that one is caught before
rigging, by head-third survival. This gate is only responsible for tearing and
speckle.

Two earlier versions of this measurement were wrong in ways worth recording.
One sampled a single pose at 35 percent through the action, which understated the
worst frame by ten to thirteen times. The other weighted everything by area,
while the artifact that dominates what a viewer sees is a shard - a triangle
stretched into a long thin sliver that fans open at an armpit or a hip and
carries almost no area. Shards are now found by how far a face's longest edge
grew, over every sampled frame of the cycle, and the worst frame decides.

Both thresholds are calibrated against owner judgement on fifteen rigged versions
at ordinary viewing distance, which is the scale these assets are used at.
Magnified four times every version in the batch shows some tearing, so a
threshold argued from magnified stills would reject everything and mean nothing.
Recalibrate if the delivery scale changes.

The versions called acceptable top out at 1.92 percent of posed area in shards
and the one called unacceptable sits at 3.71, so any threshold between them fits
the labels. Rigging the same prepared mesh four times moves this reading by about
ten percent, so the threshold needs that much clearance on the accepted side:
2.5 leaves 30 percent above the highest accepted version and 33 percent below the
rejected one.

Splitting shards by where they sit was tried and did not earn its place. Owner
judgement is explicit that tearing under the belly matters less than tearing on a
flank, so the measurement reports downward-facing low shards separately - but on
these labels the visible-only reading separates *worse* than the total, 1.56x
between accepted and rejected against 1.94x. It stays a diagnostic.

Only the shard share is gated. The ten-times area share drew the same line across
those fifteen versions and looked like an independent confirmation, but rigging
the *same* prepared mesh four times moves it from 0.54 to 0.97 percent, a 1.8x
spread on identical input, while the shard share stays inside 2.16 to 2.63. So it
was measuring the draw, not the asset, and it had already produced one false
reject on a route the owner accepted. Worst edge growth is worse still, 11.5 to
25.6 on the same input. Both are reported and neither decides.

The rigger being stochastic at all is worth remembering when a verdict looks
marginal: a reading within ten percent of the threshold is inside the noise, and
the ladder's other rungs are the cheap way to get a second opinion.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys

try:
    from animal_review_policy import (
        STRATEGY_NAMES,
        AnimalReviewPolicyError,
        load_review_policy,
    )
except ModuleNotFoundError:  # imported as tools.assets.gate_rigged_asset
    from tools.assets.animal_review_policy import (  # type: ignore[no-redef]
        STRATEGY_NAMES,
        AnimalReviewPolicyError,
        load_review_policy,
    )



class RiggedAssetGateError(ValueError):
    """The deformation report cannot prove a structurally complete asset."""


def _finite_tree(value, owner="deformation report"):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RiggedAssetGateError(f"{owner} contains NaN or Inf")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{owner}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            _finite_tree(child, f"{owner}[{index}]")
        return
    raise RiggedAssetGateError(f"{owner} contains an unsupported value")


def _number(value, owner, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RiggedAssetGateError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise RiggedAssetGateError(f"{owner} must be a finite number")
    return result


def _read_report(path):
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise RiggedAssetGateError(f"deformation report is unreadable: {candidate}")
    try:
        report = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RiggedAssetGateError(
            f"deformation report is unreadable: {candidate}"
        ) from error
    if not isinstance(report, dict):
        raise RiggedAssetGateError("deformation report must contain a JSON object")
    _finite_tree(report)
    return candidate.resolve(), report


def _readable_input(path, report_path):
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = report_path.parent / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise RiggedAssetGateError(f"animated GLB is unreadable: {candidate}")
    try:
        if candidate.stat().st_size <= 0:
            raise RiggedAssetGateError(f"animated GLB is empty: {candidate}")
        with candidate.open("rb") as handle:
            handle.read(1)
    except OSError as error:
        raise RiggedAssetGateError(f"animated GLB is unreadable: {candidate}") from error
    return candidate.resolve()


def _structural(report_path, report):
    source = report.get("input")
    if not isinstance(source, str) or not source.strip():
        raise RiggedAssetGateError("deformation report lacks an input GLB path")
    source_path = _readable_input(source, report_path)
    action = report.get("action")
    if not isinstance(action, str) or not action.strip():
        raise RiggedAssetGateError("deformation report lacks a sampled action")
    frames = report.get("frames_sampled")
    if (
        not isinstance(frames, Sequence)
        or isinstance(frames, (str, bytes))
        or not frames
    ):
        raise RiggedAssetGateError(
            "required animation action is missing or cannot be sampled"
        )

    structural_keys = (
        "mesh", "armature", "skinning", "animation_numeric_bounds"
    )
    if not any(key in report for key in structural_keys):
        faces = _number(report.get("faces"), "faces", positive=True)
        return {
            "compatibility": "legacy_deformation_report",
            "input": str(source_path),
            "action": action,
            "frames_sampled": len(frames),
            "faces": faces,
            "structural_fields_unavailable": list(structural_keys),
        }

    mesh = report.get("mesh")
    if not isinstance(mesh, Mapping) or mesh.get("valid") is not True:
        raise RiggedAssetGateError("deformation report lacks a valid mesh")
    if mesh.get("finite_coordinates") is not True:
        raise RiggedAssetGateError("mesh coordinates are not finite")
    _number(mesh.get("faces"), "mesh.faces", positive=True)

    armature = report.get("armature")
    if not isinstance(armature, Mapping) or armature.get("present") is not True:
        raise RiggedAssetGateError("deformation report lacks an armature")
    _number(armature.get("bones"), "armature.bones", positive=True)

    skinning = report.get("skinning")
    if not isinstance(skinning, Mapping) or skinning.get("valid") is not True:
        raise RiggedAssetGateError("deformation report lacks valid skinning")
    if skinning.get("finite_weights") is not True:
        raise RiggedAssetGateError("skinning weights are not finite")
    _number(skinning.get("skinned_meshes"), "skinning.skinned_meshes", positive=True)

    numeric = report.get("animation_numeric_bounds")
    if not isinstance(numeric, Mapping):
        raise RiggedAssetGateError("deformation report lacks animation numeric bounds")
    if numeric.get("exploded") is True:
        raise RiggedAssetGateError(
            "animation scale or position exceeds configured finite bounds"
        )
    position = _number(numeric.get("max_abs_position"), "animation max position")
    scale = _number(numeric.get("max_abs_scale"), "animation max scale")
    limits = numeric.get("limits")
    if not isinstance(limits, Mapping):
        raise RiggedAssetGateError("animation numeric bounds lack limits")
    max_position = _number(
        limits.get("maximum_abs_position"),
        "maximum_abs_position",
        positive=True,
    )
    max_scale = _number(
        limits.get("maximum_abs_scale"),
        "maximum_abs_scale",
        positive=True,
    )
    if position > max_position or scale > max_scale:
        raise RiggedAssetGateError("animation scale or position exceeds configured bounds")
    return {
        "input": str(source_path),
        "action": action,
        "frames_sampled": len(frames),
        "faces": mesh["faces"],
        "bones": armature["bones"],
        "vertex_groups": skinning.get("vertex_groups"),
        "max_abs_position": position,
        "max_abs_scale": scale,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walk_report")
    parser.add_argument("--policy-config", "--config", dest="policy_config", type=Path)
    parser.add_argument("--strategy", choices=STRATEGY_NAMES)
    parser.add_argument("--max-shard-share", type=float)
    parser.add_argument("--max-share-over-10x", type=float)
    args = parser.parse_args(argv)

    try:
        policy = load_review_policy(args.policy_config, strategy=args.strategy)
        report_path, report = _read_report(args.walk_report)
        structural = _structural(report_path, report)
    except (AnimalReviewPolicyError, RiggedAssetGateError) as error:
        print("RIGGED_GATE_HARD_FAIL " + json.dumps(
            {"failures": [str(error)]}, ensure_ascii=False))
        return 1
    shards = report.get("worst_share_area_shards")
    over10 = report.get("worst_share_area_over_10x")
    metrics = policy["metrics"]["rigged"]
    shard_limit = (
        metrics["max_shard_share"]
        if args.max_shard_share is None else args.max_shard_share
    )
    over10_limit = (
        metrics.get("max_share_over_10x")
        if args.max_share_over_10x is None else args.max_share_over_10x
    )
    try:
        shard_limit = _number(shard_limit, "maximum shard share")
        if shard_limit < 0.0:
            raise RiggedAssetGateError(
                "maximum shard share must be finite and non-negative"
            )
        if over10_limit is not None:
            over10_limit = _number(
                over10_limit, "maximum 10x stretch share"
            )
            if over10_limit < 0.0:
                raise RiggedAssetGateError(
                    "maximum 10x stretch share must be non-negative"
                )
    except RiggedAssetGateError as error:
        print("RIGGED_GATE_HARD_FAIL " + json.dumps(
            {"failures": [str(error)]}, ensure_ascii=False))
        return 1

    failures = []
    metric_failures = []
    try:
        shard_value = _number(shards, "shard share")
    except RiggedAssetGateError as error:
        metric_failures.append(str(error))
    else:
        if shard_value > shard_limit:
            metric_failures.append(
                f"shards cover {shard_value} of posed area at frame "
                f"{report.get('worst_frame_by_shards')} > {shard_limit}: the "
                "surface tears open visibly during the walk"
            )
    if over10_limit is not None:
        try:
            over10_value = _number(over10, "area stretched past 10x")
        except RiggedAssetGateError as error:
            metric_failures.append(str(error))
        else:
            if over10_value > over10_limit:
                metric_failures.append(
                    f"area stretched past 10x is {over10_value} > {over10_limit}"
                )
    if policy["gate_metrics"]:
        failures.extend(metric_failures)

    verdict = {
        "report": args.walk_report,
        "faces": structural.get("faces"),
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
        "worst_share_area_shards": shards,
        "worst_share_area_over_10x_reported_only": over10,
        "worst_share_area_shards_visible_reported_only":
            report.get("worst_share_area_shards_visible"),
        "worst_share_area_shards_underside_reported_only":
            report.get("worst_share_area_shards_underside"),
        "worst_share_area_over_4x": report.get("worst_share_area_over_4x"),
        "worst_frame": report.get("worst_frame_by_shards"),
        "frames_sampled": len(report.get("frames_sampled", [])),
        # Max growth is reported and never gated: it is one outlier triangle and
        # it ranks assets wrongly.
        "worst_max_edge_growth_reported_only": report.get("worst_max_edge_growth"),
        "metric_failures": metric_failures,
        "advisory_failures": [] if policy["gate_metrics"] else metric_failures,
        "failures": failures,
    }
    if failures:
        print("RIGGED_GATE_REJECT " + json.dumps(verdict, ensure_ascii=False))
        return 1
    print("RIGGED_GATE_OK " + json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

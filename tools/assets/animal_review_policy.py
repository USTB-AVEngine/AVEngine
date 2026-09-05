#!/usr/bin/env python3
"""Load the generated-animal review policy and record human decisions.

The animal chain deliberately keeps its search ladder and all review settings in
one small JSON file.  This module is the only reader used by the shell driver
and the metric gates, so a normal dataset run and an explicitly requested
strict-metrics run cannot silently drift apart.

The policy is a workflow setting.  It does not bind an asset to a byte hash:
the human review record contains an ordinary review id, an artifact path, a
decision, and notes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "avengine_generated_animal_review_policy_v1"
DECISION_SCHEMA = "avengine_generated_animal_visual_review_decision_v1"
REVIEW_SCHEMA = "avengine_generated_animal_visual_review_v1"
STRATEGY_NAMES = ("visual_review", "strict_metrics")

HARD_FAILURES = (
    "file_unreadable",
    "no_valid_mesh",
    "non_finite_value",
    "missing_armature",
    "invalid_skinning",
    "required_action_missing_or_un_sampleable",
    "blender_import_export_failure",
    "retarget_failure",
    "review_render_failure",
    "animation_scale_or_position_explosion",
)

ADVISORY_METRICS = (
    "face_target",
    "head_survival",
    "nonmanifold",
    "watertight",
    "shard_share",
    "stretch_share",
    "worst_triangle",
    "fixed_bone_count",
    "faceting",
    "fidelity",
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class AnimalReviewPolicyError(ValueError):
    """A review policy or human review record is malformed."""


def default_policy_path() -> Path:
    """Return the checked-in default policy path."""

    return Path(__file__).resolve().parents[2] / (
        "examples/assets/generated_animal_review_policy_v1.json"
    )


def _nonempty_text(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnimalReviewPolicyError(f"{owner} must be a non-empty string")
    return value


def _finite_number(value: Any, *, owner: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnimalReviewPolicyError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        suffix = " and non-negative" if nonnegative else ""
        raise AnimalReviewPolicyError(f"{owner} must be a finite number{suffix}")
    return result


def _positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AnimalReviewPolicyError(f"{owner} must be a positive integer")
    return value


def _optional_positive_int(value: Any, *, owner: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, owner=owner)


def _finite_tree(value: Any, *, owner: str) -> None:
    """Reject JSON's non-standard NaN/Infinity values at the policy boundary."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise AnimalReviewPolicyError(f"{owner} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, owner=f"{owner}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            _finite_tree(child, owner=f"{owner}[{index}]")
        return
    raise AnimalReviewPolicyError(f"{owner} contains an unsupported value")


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise AnimalReviewPolicyError(f"policy config is not a regular file: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnimalReviewPolicyError(f"could not read policy config: {candidate}") from error
    if not isinstance(value, dict):
        raise AnimalReviewPolicyError("policy config must contain a JSON object")
    _finite_tree(value, owner="policy")
    return value


def _validate_actions(value: Any, *, owner: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise AnimalReviewPolicyError(f"{owner} must be a non-empty list")
    actions = [_nonempty_text(item, owner=f"{owner}[{index}]") for index, item in enumerate(value)]
    if len(set(actions)) != len(actions):
        raise AnimalReviewPolicyError(f"{owner} must contain unique action names")
    return actions


def _validate_ladder(value: Any) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise AnimalReviewPolicyError("ladder must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        owner = f"ladder[{index}]"
        if not isinstance(item, Mapping):
            raise AnimalReviewPolicyError(f"{owner} must be an object")
        rung_id = _nonempty_text(item.get("id"), owner=f"{owner}.id")
        if not _SAFE_ID.fullmatch(rung_id) or rung_id in seen:
            raise AnimalReviewPolicyError(f"{owner}.id must be unique and shell-safe")
        seen.add(rung_id)
        mode = _nonempty_text(item.get("mode"), owner=f"{owner}.mode")
        if mode not in {"plain", "remesh"}:
            raise AnimalReviewPolicyError(f"{owner}.mode must be plain or remesh")
        target_faces = _positive_int(item.get("target_faces"), owner=f"{owner}.target_faces")
        divisor = _finite_number(
            item.get("voxel_divisor", 0.0),
            owner=f"{owner}.voxel_divisor",
            nonnegative=True,
        )
        if mode == "remesh" and divisor <= 0.0:
            raise AnimalReviewPolicyError(f"{owner}.voxel_divisor must be positive for remesh")
        smoothing = item.get("relief_smooth_iterations", -1)
        if isinstance(smoothing, bool) or not isinstance(smoothing, int) or smoothing < -1:
            raise AnimalReviewPolicyError(
                f"{owner}.relief_smooth_iterations must be an integer >= -1"
            )
        result.append(
            {
                "id": rung_id,
                "mode": mode,
                "target_faces": target_faces,
                "voxel_divisor": divisor,
                "relief_smooth_iterations": smoothing,
            }
        )
    return result


def _validate_policy_document(document: Mapping[str, Any]) -> None:
    schema = document.get("schema")
    if schema is not None and (not isinstance(schema, str) or not schema.strip()):
        raise AnimalReviewPolicyError("policy schema must be text when present")
    _nonempty_text(document.get("policy_id"), owner="policy_id")
    default_strategy = document.get("default_strategy")
    if default_strategy not in STRATEGY_NAMES:
        raise AnimalReviewPolicyError(
            f"default_strategy must be one of {STRATEGY_NAMES}"
        )
    common = document.get("common")
    if not isinstance(common, Mapping):
        raise AnimalReviewPolicyError("common must be an object")

    _validate_ladder(common.get("ladder"))

    runner = common.get("runner")
    if not isinstance(runner, Mapping):
        raise AnimalReviewPolicyError("common.runner must be an object")
    pick = _nonempty_text(runner.get("pick"), owner="common.runner.pick")
    if pick not in {"first", "best"}:
        raise AnimalReviewPolicyError("common.runner.pick must be first or best")
    _finite_number(runner.get("retry_band"), owner="common.runner.retry_band", nonnegative=True)
    retries = runner.get("rig_retries")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise AnimalReviewPolicyError(
            "common.runner.rig_retries must be a non-negative integer"
        )
    _finite_number(
        runner.get("relief_smooth"),
        owner="common.runner.relief_smooth",
    )

    render = common.get("render")
    if not isinstance(render, Mapping):
        raise AnimalReviewPolicyError("common.render must be an object")
    for key in ("walking_action", "turntable_action"):
        _nonempty_text(render.get(key), owner=f"common.render.{key}")
    for key in ("walking_frames", "turntable_frames"):
        _positive_int(render.get(key), owner=f"common.render.{key}")
    _finite_number(common["render"].get("walking_zoom"), owner="common.render.walking_zoom")
    pose_ratio = _finite_number(
        common["render"].get("turntable_pose_ratio"),
        owner="common.render.turntable_pose_ratio",
        nonnegative=True,
    )
    if pose_ratio > 1.0:
        raise AnimalReviewPolicyError("common.render.turntable_pose_ratio must be <= 1")
    for key in ("walking_dir", "turntable_dir"):
        directory = _nonempty_text(render.get(key), owner=f"common.render.{key}")
        component = Path(directory)
        if (
            component.is_absolute()
            or len(component.parts) != 1
            or component.name in {"", ".", ".."}
        ):
            raise AnimalReviewPolicyError(
                f"common.render.{key} must be one relative child directory"
            )

    retarget = common.get("retarget")
    if not isinstance(retarget, Mapping):
        raise AnimalReviewPolicyError("common.retarget must be an object")
    target_axis = _nonempty_text(
        retarget.get("target_front_axis"),
        owner="common.retarget.target_front_axis",
    )
    if target_axis not in {"positive-x", "negative-x", "positive-y", "negative-y"}:
        raise AnimalReviewPolicyError("common.retarget.target_front_axis is invalid")
    yaw = retarget.get("motion_basis_yaw_deg")
    _finite_number(yaw, owner="common.retarget.motion_basis_yaw_deg")
    side_mode = _nonempty_text(
        retarget.get("side_chain_mode"),
        owner="common.retarget.side_chain_mode",
    )
    if side_mode not in {"matched", "swapped"}:
        raise AnimalReviewPolicyError("common.retarget.side_chain_mode is invalid")

    measurement = common.get("measurement")
    if not isinstance(measurement, Mapping):
        raise AnimalReviewPolicyError("common.measurement must be an object")
    _nonempty_text(
        measurement.get("deformation_action"),
        owner="common.measurement.deformation_action",
    )
    _positive_int(measurement.get("sample_count"), owner="common.measurement.sample_count")
    _finite_number(
        measurement.get("shard_edge_growth_threshold"),
        owner="common.measurement.shard_edge_growth_threshold",
        nonnegative=True,
    )
    underside = _finite_number(
        measurement.get("underside_height_fraction"),
        owner="common.measurement.underside_height_fraction",
        nonnegative=True,
    )
    if underside > 1.0:
        raise AnimalReviewPolicyError(
            "common.measurement.underside_height_fraction must be <= 1"
        )

    closure = common.get("closure")
    if not isinstance(closure, Mapping):
        raise AnimalReviewPolicyError("common.closure must be an object")
    _validate_actions(closure.get("required_actions"), owner="common.closure.required_actions")
    for key in (
        "weight_tolerance",
        "minimum_pose_translation_delta",
        "minimum_pose_rotation_delta_deg",
        "maximum_cycle_translation_delta",
        "maximum_cycle_rotation_delta_deg",
        "maximum_abs_position",
        "maximum_abs_scale",
    ):
        _finite_number(
            closure.get(key),
            owner=f"common.closure.{key}",
            nonnegative=True,
        )

    metrics = common.get("metrics")
    if not isinstance(metrics, Mapping):
        raise AnimalReviewPolicyError("common.metrics must be an object")
    retopo = metrics.get("retopology")
    rigged = metrics.get("rigged")
    if not isinstance(retopo, Mapping) or not isinstance(rigged, Mapping):
        raise AnimalReviewPolicyError("common.metrics.retopology and .rigged are required")
    _finite_number(
        retopo.get("min_head_survival"),
        owner="common.metrics.retopology.min_head_survival",
        nonnegative=True,
    )
    face_tolerance = _finite_number(
        retopo.get("face_tolerance"),
        owner="common.metrics.retopology.face_tolerance",
        nonnegative=True,
    )
    if face_tolerance > 1.0:
        raise AnimalReviewPolicyError(
            "common.metrics.retopology.face_tolerance must be <= 1"
        )
    _finite_number(
        rigged.get("max_shard_share"),
        owner="common.metrics.rigged.max_shard_share",
        nonnegative=True,
    )
    over10 = rigged.get("max_share_over_10x")
    if over10 is not None:
        _finite_number(
            over10,
            owner="common.metrics.rigged.max_share_over_10x",
            nonnegative=True,
        )

    strategies = document.get("strategies")
    if not isinstance(strategies, Mapping):
        raise AnimalReviewPolicyError("strategies must be an object")
    missing_strategies = sorted(set(STRATEGY_NAMES) - set(strategies))
    if missing_strategies:
        raise AnimalReviewPolicyError(
            f"strategies are missing {missing_strategies}"
        )
    for name in STRATEGY_NAMES:
        strategy = strategies.get(name)
        if not isinstance(strategy, Mapping):
            raise AnimalReviewPolicyError(f"strategies.{name} must be an object")
        if not isinstance(strategy.get("gate_metrics"), bool):
            raise AnimalReviewPolicyError(f"strategies.{name}.gate_metrics must be boolean")
        if not isinstance(strategy.get("require_closed_cycle"), bool):
            raise AnimalReviewPolicyError(
                f"strategies.{name}.require_closed_cycle must be boolean"
            )
        _optional_positive_int(
            strategy.get("expected_bone_count"),
            owner=f"strategies.{name}.expected_bone_count",
        )
        _optional_positive_int(
            strategy.get("expected_vertex_group_count"),
            owner=f"strategies.{name}.expected_vertex_group_count",
        )
        advisory = strategy.get("advisory_metrics")
        if (
            not isinstance(advisory, Sequence)
            or isinstance(advisory, (str, bytes))
            or any(item not in ADVISORY_METRICS for item in advisory)
        ):
            raise AnimalReviewPolicyError(
                f"strategies.{name}.advisory_metrics contains an unknown metric"
            )

    decision = document.get("decision")
    if not isinstance(decision, Mapping):
        raise AnimalReviewPolicyError("decision must be an object")
    _nonempty_text(decision.get("accept_value"), owner="decision.accept_value")
    _nonempty_text(decision.get("reject_value"), owner="decision.reject_value")


def load_review_policy(
    config_path: str | Path | None = None,
    *,
    strategy: str | None = None,
) -> dict[str, Any]:
    """Load and normalize one strategy from the checked-in JSON policy."""

    source = Path(config_path).expanduser() if config_path is not None else default_policy_path()
    document = _read_json(source)
    _validate_policy_document(document)
    name = strategy or str(document["default_strategy"])
    if name not in STRATEGY_NAMES:
        raise AnimalReviewPolicyError(
            f"unknown review strategy {name!r}; choose one of {STRATEGY_NAMES}"
        )
    selected = document["strategies"][name]
    assert isinstance(selected, Mapping)
    common = deepcopy(document["common"])
    return {
        "schema": document["schema"],
        "policy_id": document["policy_id"],
        "config_path": str(source.resolve()),
        "strategy": name,
        "gate_metrics": bool(selected["gate_metrics"]),
        "require_closed_cycle": bool(selected["require_closed_cycle"]),
        "expected_bone_count": selected.get("expected_bone_count"),
        "expected_vertex_group_count": selected.get("expected_vertex_group_count"),
        "advisory_metrics": list(selected["advisory_metrics"]),
        "hard_failures": list(HARD_FAILURES),
        "ladder": common["ladder"],
        "runner": common["runner"],
        "render": common["render"],
        "measurement": common["measurement"],
        "retarget": common["retarget"],
        "closure": common["closure"],
        "metrics": common["metrics"],
        "decision": document["decision"],
    }


def resolve_policy(
    config_path: str | Path | Mapping[str, Any] | None = None,
    *,
    strategy: str | None = None,
) -> dict[str, Any]:
    """Resolve a file path or already-loaded policy for callers and tests."""

    if isinstance(config_path, Mapping):
        document = dict(config_path)
        _finite_tree(document, owner="policy")
        _validate_policy_document(document)
        temporary = deepcopy(document)
        name = strategy or str(temporary["default_strategy"])
        if name not in STRATEGY_NAMES:
            raise AnimalReviewPolicyError(
                f"unknown review strategy {name!r}; choose one of {STRATEGY_NAMES}"
            )
        selected = temporary["strategies"][name]
        common = deepcopy(temporary["common"])
        return {
            "schema": temporary["schema"],
            "policy_id": temporary["policy_id"],
            "config_path": None,
            "strategy": name,
            "gate_metrics": bool(selected["gate_metrics"]),
            "require_closed_cycle": bool(selected["require_closed_cycle"]),
            "expected_bone_count": selected.get("expected_bone_count"),
            "expected_vertex_group_count": selected.get("expected_vertex_group_count"),
            "advisory_metrics": list(selected["advisory_metrics"]),
            "hard_failures": list(HARD_FAILURES),
            "ladder": common["ladder"],
            "runner": common["runner"],
            "render": common["render"],
            "measurement": common["measurement"],
            "retarget": common["retarget"],
            "closure": common["closure"],
            "metrics": common["metrics"],
            "decision": temporary["decision"],
        }
    return load_review_policy(config_path, strategy=strategy)


def _new_output(path: str | Path, *, owner: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists() or candidate.is_symlink():
        raise AnimalReviewPolicyError(
            f"{owner} already exists; refusing to overwrite: {candidate}"
        )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _review_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.exists():
        raise AnimalReviewPolicyError(f"review path is missing or unsafe: {candidate}")
    if not candidate.is_file() and not candidate.is_dir():
        raise AnimalReviewPolicyError(f"review path is not readable: {candidate}")
    return candidate.resolve()


def write_visual_review_manifest(
    path: str | Path,
    *,
    review_id: str,
    asset_id: str | None,
    review_path: str | Path,
    policy: Mapping[str, Any],
    accepted_rung: str,
    walking_render: str,
    turntable_render: str,
) -> Path:
    """Write the ordinary, human-facing review handoff without a hash binding."""

    review_id = _nonempty_text(review_id, owner="review_id")
    asset_id = (
        _nonempty_text(asset_id, owner="asset_id") if asset_id is not None else None
    )
    accepted_rung = _nonempty_text(accepted_rung, owner="accepted_rung")
    walking_render = _nonempty_text(walking_render, owner="walking_render")
    turntable_render = _nonempty_text(turntable_render, owner="turntable_render")
    review = _review_path(review_path)
    output = _new_output(path, owner="visual review manifest")
    payload = {
        "schema": REVIEW_SCHEMA,
        "status": "needs_visual_review",
        "review_id": review_id,
        "asset_id": asset_id,
        "review_path": str(review),
        "policy_id": policy.get("policy_id"),
        "strategy": policy.get("strategy"),
        "accepted_rung": accepted_rung,
        "renders": {
            "walking": walking_render,
            "turntable": turntable_render,
        },
        "decision": None,
        "qualification_claim": False,
        "dataset_asset_registration_authorized": False,
        "formal_dataset_registration_authorized": False,
    }
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as error:
        raise AnimalReviewPolicyError(
            f"could not write visual review manifest: {output}"
        ) from error
    return output


def write_visual_review_decision(
    path: str | Path,
    *,
    review_id: str,
    asset_id: str,
    review_path: str | Path,
    decision: str,
    notes: str = "",
) -> Path:
    """Record accept/reject from a reviewer using ordinary ids and paths only."""

    review_id = _nonempty_text(review_id, owner="review_id")
    asset_id = _nonempty_text(asset_id, owner="asset_id")
    decision = _nonempty_text(decision, owner="decision").lower()
    if decision not in {"accept", "reject"}:
        raise AnimalReviewPolicyError("decision must be accept or reject")
    if not isinstance(notes, str):
        raise AnimalReviewPolicyError("notes must be text")
    review = _review_path(review_path)
    output = _new_output(path, owner="visual review decision")
    status = (
        "accepted_for_dataset_asset"
        if decision == "accept"
        else "rejected_for_dataset_asset"
    )
    payload = {
        "schema": DECISION_SCHEMA,
        "review_id": review_id,
        "asset_id": asset_id,
        "review_path": str(review),
        "decision": decision,
        "notes": notes,
        "status": status,
        "qualification_claim": False,
        "dataset_asset_registration_authorized": decision == "accept",
        "formal_dataset_registration_authorized": False,
    }
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as error:
        raise AnimalReviewPolicyError(
            f"could not write visual review decision: {output}"
        ) from error
    return output


def _emit_policy_json(policy: Mapping[str, Any]) -> None:
    print(json.dumps(policy, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def _emit_ladder(policy: Mapping[str, Any]) -> None:
    for rung in policy["ladder"]:
        print(
            "\t".join(
                (
                    str(rung["id"]),
                    str(rung["mode"]),
                    str(rung["target_faces"]),
                    str(rung["voxel_divisor"]),
                    str(rung["relief_smooth_iterations"]),
                )
            )
        )


def _policy_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--policy-config", dest="config", type=Path)
    parser.add_argument("--strategy", choices=STRATEGY_NAMES)
    parser.add_argument("--emit-json", action="store_true")
    parser.add_argument("--emit-ladder", action="store_true")
    args = parser.parse_args(list(argv))
    try:
        policy = load_review_policy(args.config, strategy=args.strategy)
        if args.emit_ladder:
            _emit_ladder(policy)
        else:
            _emit_policy_json(policy)
    except AnimalReviewPolicyError as error:
        print(f"animal review policy error: {error}", file=sys.stderr)
        return 2
    return 0


def _decision_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=write_visual_review_decision.__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--review-path", type=Path, required=True)
    parser.add_argument("--decision", choices=("accept", "reject"), required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args(list(argv))
    try:
        output = write_visual_review_decision(
            args.output,
            review_id=args.review_id,
            asset_id=args.asset_id,
            review_path=args.review_path,
            decision=args.decision,
            notes=args.notes,
        )
    except AnimalReviewPolicyError as error:
        print(f"animal review decision error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "VISUAL_REVIEW_DECISION_OK", "path": str(output)}))
    return 0


def _manifest_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=write_visual_review_manifest.__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--asset-id")
    parser.add_argument("--review-path", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, required=True)
    parser.add_argument("--accepted-rung", required=True)
    parser.add_argument("--walking-render", required=True)
    parser.add_argument("--turntable-render", required=True)
    args = parser.parse_args(list(argv))
    try:
        output = write_visual_review_manifest(
            args.output,
            review_id=args.review_id,
            asset_id=args.asset_id,
            review_path=args.review_path,
            policy={
                "policy_id": args.policy_id,
                "strategy": args.strategy,
            },
            accepted_rung=args.accepted_rung,
            walking_render=args.walking_render,
            turntable_render=args.turntable_render,
        )
    except AnimalReviewPolicyError as error:
        print(f"animal review manifest error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "VISUAL_REVIEW_MANIFEST_OK", "path": str(output)}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "decision":
        return _decision_command(values[1:])
    if values and values[0] == "review-manifest":
        return _manifest_command(values[1:])
    return _policy_command(values)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate or deterministically replay the camera-pan motion audit receipt."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BUILDER_PATH = Path(__file__).with_name(
    "audit_strict_two_human_camera_pan_motion_realism.py"
)
SPEC = importlib.util.spec_from_file_location("camera_pan_motion_auditor", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import auditor: {BUILDER_PATH}")
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _close(left: object, right: float, tolerance: float = 1.0e-9) -> bool:
    if not isinstance(left, (int, float)):
        return False
    return math.isfinite(float(left)) and abs(float(left) - right) <= tolerance


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    _require(receipt.get("schema") == AUDITOR.RECEIPT_SCHEMA, "receipt schema drift")
    _require(
        receipt.get("status") == "reject_release_missing_camera_motion_authority",
        "receipt is not the expected fail-closed rejection",
    )
    _require(
        receipt.get("candidate_decision") == "REJECT_RELEASE_KEEP_PIPELINE_CANARY"
        and receipt.get("release_qualified") is False
        and receipt.get("release_classification")
        == "nonrelease_pipeline_evidence_only",
        "release classification drift",
    )
    blocker = receipt.get("first_blocker")
    _require(isinstance(blocker, Mapping), "first blocker is missing")
    _require(
        blocker.get("code") == "missing_native_or_approved_camera_motion_profile",
        "first blocker drift",
    )

    source = receipt.get("source_authority")
    _require(isinstance(source, Mapping), "source authority is missing")
    _require(
        source.get("suite_schema") == AUDITOR.EXPECTED_SOURCE_SCHEMA
        and source.get("scope") == "all scenarios and all plan frames",
        "source authority boundary drift",
    )
    inventory = source.get("inventory")
    _require(isinstance(inventory, Mapping), "inventory audit is missing")
    _require(
        inventory.get("status") == "pass_inventory_readback"
        and inventory.get("scenario_count") == 1000
        and inventory.get("total_plan_frame_count") == 75_000,
        "full unique1000 inventory closure drift",
    )
    frame_counts = inventory.get("plan_frame_count_distribution")
    rates = inventory.get("frame_rate_hz_distribution")
    _require(
        frame_counts == [{"frame_count": 75, "scenario_count": 1000}],
        "full75 inventory distribution drift",
    )
    _require(
        rates == [{"frame_rate_hz": 15.0, "scenario_count": 1000}],
        "15 Hz inventory distribution drift",
    )
    static = inventory.get("static_camera_inventory")
    dynamic = inventory.get("dynamic_camera_inventory")
    _require(isinstance(static, Mapping), "static camera inventory is missing")
    _require(isinstance(dynamic, Mapping), "dynamic camera inventory is missing")
    _require(
        static.get("episode_count") == 1000
        and static.get("unique_position_count") == 1
        and static.get("unique_habitat_yaw_count") == 1
        and static.get("positions") == [[[-0.7, 1.471, 0.65], 1000]]
        and static.get("habitat_yaws_deg") == [[55.0, 1000]]
        and static.get("cross_episode_pose_differences_are_motion_samples") is False,
        "static camera inventory facts drift",
    )
    _require(
        dynamic.get("scenario_count") == 0
        and dynamic.get("frame_count") == 0
        and dynamic.get("authority_field_distribution") == []
        and dynamic.get("positive_angular_speed_sample_count") == 0
        and dynamic.get("continuous_pan_segment_count") == 0,
        "empty dynamic-camera inventory closure drift",
    )
    for key in (
        "absolute_angular_speed_deg_s_distribution",
        "continuous_pan_duration_s_distribution",
    ):
        distribution = dynamic.get(key)
        _require(
            isinstance(distribution, Mapping)
            and distribution.get("status") == "undefined_empty"
            and distribution.get("sample_count") == 0,
            f"{key} must remain undefined without samples",
        )

    candidate = receipt.get("current_candidate")
    _require(isinstance(candidate, Mapping), "candidate audit is missing")
    _require(
        candidate.get("episode_id") == AUDITOR.EXPECTED_CANDIDATE_EPISODE
        and candidate.get("target_side") == "right"
        and candidate.get("frame_count") == 75
        and _close(candidate.get("frame_rate_hz"), 15.0)
        and _close(candidate.get("yaw_start_deg"), 52.0)
        and _close(candidate.get("yaw_end_deg"), 58.0)
        and _close(candidate.get("yaw_span_deg"), 6.0)
        and candidate.get("unique_yaw_count") == 75
        and candidate.get("nonzero_interframe_step_count") == 74
        and candidate.get("uniform_interframe_step") is True
        and _close(candidate.get("nominal_clip_duration_s"), 5.0)
        and _close(candidate.get("sampled_interval_duration_s"), 74.0 / 15.0)
        and _close(candidate.get("nominal_clip_angular_velocity_deg_s"), 1.2)
        and _close(
            candidate.get("interframe_slope_angular_velocity_deg_s"),
            6.0 / (74.0 / 15.0),
        ),
        "candidate angular-motion arithmetic drift",
    )
    _require(
        candidate.get("full_clip_linear_interpolation") is True
        and candidate.get("declared_active_interval") is None
        and candidate.get("outside_active_interval_hold_frame_count") == 0
        and candidate.get("recognized_motion_authority_fields") == []
        and candidate.get("native_or_approved_motion_authority_present") is False,
        "candidate incorrectly gained motion authority",
    )

    stretch = receipt.get("time_stretch_assessment")
    search = receipt.get("native_rate_candidate_search")
    _require(isinstance(stretch, Mapping), "time-stretch assessment is missing")
    _require(isinstance(search, Mapping), "candidate search is missing")
    _require(
        stretch.get("status") == "not_computable_no_native_camera_motion_source"
        and stretch.get("is_proven_resampling_of_a_native_pan") is False,
        "unsupported time-stretch claim drift",
    )
    _require(
        search.get("status") == "no_candidate_under_audited_authority"
        and search.get("candidate_count") == 0
        and search.get("candidates") == [],
        "an unauthorized camera candidate was invented",
    )

    replacement = receipt.get("required_replacement_contract")
    _require(isinstance(replacement, Mapping), "replacement contract is missing")
    _require(
        replacement.get("global_time_stretch_allowed") is False
        and replacement.get("cross_episode_static_pose_interpolation_allowed") is False
        and replacement.get("outside_active_interval_policy")
        == "HOLD first/last authorized yaw",
        "native-rate replacement contract drift",
    )

    machine = receipt.get("preserved_machine_evidence")
    _require(isinstance(machine, Mapping), "machine evidence is missing")
    _require(
        machine.get("preserved_not_recomputed") is True
        and machine.get("strict_finalization_status") == "pass"
        and machine.get("dynamic_full75_canary_pass") is True
        and machine.get("captured_frame_count") == 75
        and machine.get("camera_transform_readback_status")
        == "pass_exact_all_normal_and_target_only_frames"
        and machine.get("camera_readback_count") == 225
        and _close(machine.get("normal_camera_yaw_span_deg"), 6.0)
        and machine.get("normal_distinct_camera_yaw_count") == 75
        and machine.get("visibility_status") == "pass"
        and machine.get("visible_floor_clearance_gap_beneath_both_characters") is True
        and machine.get("evidence_classification")
        == "nonformal_pipeline_canary_evidence_only",
        "preserved machine evidence drift",
    )
    gates = receipt.get("independent_release_gates")
    _require(isinstance(gates, Mapping), "independent release gates are missing")
    _require(
        gates.get("motion_realism") == "reject_missing_authority"
        and gates.get("ground_contact")
        == "blocked_unqualified_visible_floor_clearance_gap",
        "independent release-gate status drift",
    )
    execution = receipt.get("audit_execution")
    _require(isinstance(execution, Mapping), "execution boundary is missing")
    _require(
        execution.get("cpu_only") is True
        and execution.get("gpu_used") is False
        and execution.get("files_mutated_in_source_repo") is False
        and execution.get("other_gate_results_recomputed") is False
        and receipt.get("formal") is False
        and receipt.get("formal_episode_count") == 0
        and receipt.get("qualification_claim") is False,
        "CPU/formal/scope boundary drift",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-suite", type=Path)
    parser.add_argument("--candidate-preflight", type=Path)
    parser.add_argument("--strict-finalization", type=Path)
    parser.add_argument("--visual-receipt", type=Path)
    args = parser.parse_args()
    receipt = _load(args.receipt.resolve())
    validate_receipt(receipt)
    replay_inputs = (
        args.source_suite,
        args.candidate_preflight,
        args.strict_finalization,
        args.visual_receipt,
    )
    if any(replay_inputs):
        _require(all(replay_inputs), "all four replay inputs are required together")
        assert args.source_suite is not None
        assert args.candidate_preflight is not None
        assert args.strict_finalization is not None
        assert args.visual_receipt is not None
        suite_path = args.source_suite.resolve()
        replay = AUDITOR.build_receipt(
            suite=_load(suite_path),
            source_suite_path=str(suite_path),
            preflight=_load(args.candidate_preflight.resolve()),
            finalization=_load(args.strict_finalization.resolve()),
            visual_receipt=_load(args.visual_receipt.resolve()),
        )
        _require(receipt == replay, "receipt differs from deterministic replay")
    print(
        "STRICT_TWO_HUMAN_CAMERA_PAN_MOTION_REALISM_AUDIT_VALID "
        f"status={receipt['status']} receipt={args.receipt.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

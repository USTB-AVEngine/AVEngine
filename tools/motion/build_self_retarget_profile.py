#!/usr/bin/env python3
"""Build a hash-bound identity-skeleton retarget probe profile.

This utility is intentionally for research-candidate normalization probes.  It
maps every joint in one target GLB to the identically named source joint and
requires callers to declare the species/family-specific locomotion chains.
The emitted QA thresholds are broad smoke gates, not admission thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from avengine.assets.glb import extract_skins, load_glb
from avengine.motion.profiles import load_motion_retarget_profile


def _chain(value: str) -> dict[str, Any]:
    fields = value.split("|")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "chain must be ID|SIDE|JOINT1,JOINT2,...|END_JOINT|END_ROLE"
        )
    chain_id, side, joint_text, end_joint, end_role = fields
    joints = joint_text.split(",")
    if (
        not chain_id
        or side not in {"left", "right", "center"}
        or any(not joint for joint in joints)
        or not end_joint
        or not end_role
    ):
        raise argparse.ArgumentTypeError("chain fields must be non-empty and valid")
    return {
        "chain_id": chain_id,
        "chain_kind": "locomotion_limb",
        "side": side,
        "semantic_joint_ids": joints,
        "end_effector_role": end_role,
        "target_end_effector_joint_id": end_joint,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--body-plan-id", required=True)
    parser.add_argument("--motion-family-id", required=True)
    parser.add_argument("--skeleton-id", required=True)
    parser.add_argument("--template-id", required=True)
    parser.add_argument("--root-joint", required=True)
    parser.add_argument(
        "--forward-axis", choices=("+X", "-X", "+Y", "-Y", "+Z", "-Z"), required=True
    )
    parser.add_argument(
        "--lateral-axis", choices=("+X", "-X", "+Y", "-Y", "+Z", "-Z"), required=True
    )
    parser.add_argument(
        "--vertical-axis", choices=("+X", "-X", "+Y", "-Y", "+Z", "-Z"), required=True
    )
    parser.add_argument("--coat-profile-id", required=True)
    parser.add_argument("--coat-value", action="append", required=True)
    parser.add_argument("--chain", type=_chain, action="append", required=True)
    parser.add_argument("--output-sample-rate-hz", type=int, default=30)
    parser.add_argument("--walk-action-hint", default="walking")
    return parser


def _pair_chains(
    chains: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_prefix: dict[str, dict[str, dict[str, Any]]] = {}
    for chain in chains:
        side = chain["side"]
        if side not in {"left", "right"}:
            continue
        identifier = chain["chain_id"]
        prefix = identifier.removesuffix("_left").removesuffix("_right")
        by_prefix.setdefault(prefix, {})[side] = chain
    paired = [value for value in by_prefix.values() if set(value) == {"left", "right"}]
    if not paired:
        raise ValueError("at least one left/right chain pair is required")
    groups = [
        {
            "group_id": f"{prefix}_pair",
            "chain_ids": [value["left"]["chain_id"], value["right"]["chain_id"]],
        }
        for prefix, value in sorted(by_prefix.items())
        if set(value) == {"left", "right"}
    ]
    symmetry = [
        {
            "symmetry_id": f"{prefix}_left_right",
            "first_chain_id": value["left"]["chain_id"],
            "second_chain_id": value["right"]["chain_id"],
            "maximum_relative_difference": 1.0,
            "axes": ["forward", "vertical"],
            "metric_space": "rest_length_normalized",
        }
        for prefix, value in sorted(by_prefix.items())
        if set(value) == {"left", "right"}
    ]
    return groups, symmetry


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.input_glb.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"refusing to replace profile: {output}")
    document = load_glb(source)
    skins = extract_skins(document)
    if len(skins) != 1:
        raise ValueError(f"expected one target skin, found {len(skins)}")
    joint_names = [joint.name for joint in skins[0].joints]
    if any(not name for name in joint_names) or len(set(joint_names)) != len(
        joint_names
    ):
        raise ValueError("target skin joints must have unique non-empty names")
    names = {str(name) for name in joint_names}
    if args.root_joint not in names:
        raise ValueError(f"root joint not found: {args.root_joint!r}")
    root_joint = next(
        joint for joint in skins[0].joints if joint.name == args.root_joint
    )
    child_name_by_node = {
        joint.node_index: str(joint.name) for joint in skins[0].joints
    }
    root_children = [
        child_name_by_node[node]
        for node in root_joint.child_joint_node_indices
        if node in child_name_by_node
    ]
    if not root_children:
        raise ValueError("root joint must have a direct child for the axial anchor")
    if len(args.coat_value) != 3 or len(set(args.coat_value)) != 3:
        raise ValueError("exactly three unique breed/species coat values are required")
    chains: list[dict[str, Any]] = args.chain
    referenced = {
        joint
        for chain in chains
        for joint in [
            *chain["semantic_joint_ids"],
            chain["target_end_effector_joint_id"],
        ]
    }
    missing = sorted(referenced - names)
    if missing:
        raise ValueError(f"chain joints are absent from the target skin: {missing}")
    groups, symmetry = _pair_chains(chains)
    first_group = groups[0]
    primary_mapped_names = [
        args.root_joint,
        *[joint for chain in chains for joint in chain["semantic_joint_ids"]],
    ]
    auxiliary_names = [
        str(name) for name in joint_names if name not in set(primary_mapped_names)
    ]
    profile = {
        "schema": "avengine_motion_retarget_profile_v1",
        "profile_id": args.profile_id,
        "adapter_id": "quadruped_mammal_locomotion_v1",
        "body_plan_id": args.body_plan_id,
        "motion_family_id": args.motion_family_id,
        "source_skeleton_id": args.skeleton_id,
        "target_template_id": args.template_id,
        "solver": {
            "solver_id": "world_left_delta_v2",
            "motion_basis_xyzw": [0.0, 0.0, 0.0, 1.0],
            "motion_amplitude": 1.0,
            "output_sample_rate_hz": args.output_sample_rate_hz,
            "time_mapping": "preserve_source_seconds",
            "root_joint_semantic_id": args.root_joint,
            "root_rotation_policy": "target_rest",
            "root_translation_policy": "target_rest",
            "unmapped_target_joint_policy": "target_rest_local",
        },
        "semantic_chains": [
            {
                "chain_id": "axial_root_anchor",
                "chain_kind": "axial",
                "side": "center",
                "semantic_joint_ids": [args.root_joint],
                "end_effector_role": "body_anchor",
                "target_end_effector_joint_id": root_children[0],
            },
            *chains,
            *[
                {
                    "chain_id": f"aux_joint_{index:03d}",
                    "chain_kind": "auxiliary",
                    "side": "center",
                    "semantic_joint_ids": [name],
                    "end_effector_role": f"auxiliary_joint_{index:03d}",
                    "target_end_effector_joint_id": name,
                }
                for index, name in enumerate(auxiliary_names)
            ],
        ],
        "joint_mappings": [
            {
                "semantic_joint_id": name,
                "source_joint_id": name,
                "target_joint_id": name,
            }
            for name in joint_names
        ],
        "actions": [
            {
                "semantic_action_id": "idle",
                "source_action_hint": "idle",
                "output_action_name": "Idle",
            },
            {
                "semantic_action_id": "walk",
                "source_action_hint": args.walk_action_hint,
                "output_action_name": "Walking",
            },
        ],
        "qa_contract": {
            "semantic_action_id": "walk",
            "coordinate_frame": {
                "forward_axis": args.forward_axis,
                "lateral_axis": args.lateral_axis,
                "vertical_axis": args.vertical_axis,
            },
            "sample_rate_hz": 15,
            "minimum_sample_count": 3,
            "cyclic": True,
            "required_chain_ids": [chain["chain_id"] for chain in chains],
            "chain_thresholds": {
                chain["chain_id"]: {
                    "minimum_forward_excursion_normalized": 0.01,
                    "maximum_lateral_to_forward_ratio": 10.0,
                }
                for chain in chains
            },
            "joint_thresholds_by_chain": {
                chain["chain_id"]: {
                    joint: {
                        "minimum_angular_excursion_degrees": 0.0,
                        "maximum_angular_speed_degrees_per_second": 3600.0,
                    }
                    for joint in chain["semantic_joint_ids"]
                }
                for chain in chains
            },
            "chain_groups": groups,
            "group_ratio_thresholds": [
                {
                    "ratio_id": "smoke_self_ratio",
                    "numerator_group_id": first_group["group_id"],
                    "numerator_axis": "forward",
                    "denominator_group_id": first_group["group_id"],
                    "denominator_axis": "forward",
                    "metric_space": "rest_length_normalized",
                    "minimum_ratio": 0.999999,
                    "maximum_ratio": 1.000001,
                }
            ],
            "symmetry_thresholds": symmetry,
        },
        "attribute_domain": {
            "size": ["small", "medium", "large"],
            "body_build": ["slim", "standard", "stocky"],
            "coat_profile_id": args.coat_profile_id,
            "coat_values": args.coat_value,
            "life_stage": ["young", "adult", "senior"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(profile, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    load_motion_retarget_profile(output)
    print(json.dumps({"status": "pass", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

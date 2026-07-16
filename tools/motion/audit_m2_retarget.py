#!/usr/bin/env python3
"""Audit one profile-bound retargeted M2 action with body-plan-neutral QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from avengine.contracts.json_io import load_json  # noqa: E402
from avengine.m2.actions import read_baked_actions_npz  # noqa: E402
from avengine.m2.glb import load_glb  # noqa: E402
from avengine.m2.habitat import build_habitat_asset_mapping  # noqa: E402
from avengine.m2.kinematics import forward_kinematics  # noqa: E402
from avengine.motion.profiles import (  # noqa: E402
    MotionRetargetProfile,
    SemanticChain,
    load_motion_retarget_profile,
)
from avengine.motion.qa import SemanticChainSamples, evaluate_motion_qa  # noqa: E402


REPORT_SCHEMA = "avengine_motion_retarget_audit_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input(path: Path, *, owner: str, suffix: str) -> Path:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{owner} path must not contain a symbolic link")
    if (
        not absolute.is_file()
        or absolute.stat().st_size <= 0
        or absolute.suffix.lower() != suffix
    ):
        raise ValueError(f"{owner} must be a non-empty {suffix} regular file")
    return absolute


def _output(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    if absolute.exists() or absolute.is_symlink():
        raise ValueError(f"refusing to replace output report: {absolute}")
    if absolute.suffix.lower() != ".json":
        raise ValueError("output report must use the .json suffix")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _axis_vector(axis: str) -> np.ndarray:
    sign = 1.0 if axis[0] == "+" else -1.0
    vector = np.zeros(3, dtype=np.float64)
    vector[{"X": 0, "Y": 1, "Z": 2}[axis[1]]] = sign
    return vector


def _mapping_by_semantic(profile: MotionRetargetProfile) -> dict[str, str]:
    return {
        mapping.semantic_joint_id: mapping.target_joint_id
        for mapping in profile.joint_mappings
    }


def _chain_rest_length_m(
    chain: SemanticChain,
    *,
    semantic_targets: dict[str, str],
    joint_by_id: dict[str, Any],
) -> float:
    target_chain = [semantic_targets[joint_id] for joint_id in chain.semantic_joint_ids]
    target_chain.append(chain.target_end_effector_joint_id)
    if target_chain[-1] == target_chain[-2]:
        target_chain.pop()
    length = 0.0
    for parent_id, child_id in zip(target_chain, target_chain[1:]):
        child = joint_by_id.get(child_id)
        if child is None:
            raise ValueError(
                f"chain {chain.chain_id!r} end-effector path has unknown joint "
                f"{child_id!r}"
            )
        if child.parent_joint_id != parent_id:
            raise ValueError(
                f"chain {chain.chain_id!r} is not a direct target hierarchy path: "
                f"{parent_id!r} -> {child_id!r}"
            )
        length += float(np.linalg.norm(child.local_translation_m))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"chain {chain.chain_id!r} has invalid rest arc length")
    return length


def _samples(
    profile: MotionRetargetProfile,
    mapping: Any,
    action: Any,
) -> tuple[SemanticChainSamples, ...]:
    semantic_targets = _mapping_by_semantic(profile)
    joint_by_id = {joint.joint_id: joint for joint in mapping.joints}
    runtime_index = {
        joint_id: index for index, joint_id in enumerate(mapping.runtime_joint_order)
    }
    frame = profile.qa_coordinate_frame
    basis = np.stack(
        (
            _axis_vector(frame.forward_axis),
            _axis_vector(frame.lateral_axis),
            _axis_vector(frame.vertical_axis),
        )
    )
    chain_by_id = {chain.chain_id: chain for chain in profile.semantic_chains}
    positions_by_chain = {
        chain_id: [] for chain_id in profile.qa_contract.required_chain_ids
    }
    for pose in action.rotations_xyzw:
        solved = forward_kinematics(mapping, pose)
        for chain_id in profile.qa_contract.required_chain_ids:
            chain = chain_by_id[chain_id]
            xyz = np.asarray(
                solved.joint_transform(
                    chain.target_end_effector_joint_id
                ).translation_m,
                dtype=np.float64,
            )
            positions_by_chain[chain_id].append(basis @ xyz)

    result: list[SemanticChainSamples] = []
    rotations = np.asarray(action.rotations_xyzw, dtype=np.float64)
    for chain_id in profile.qa_contract.required_chain_ids:
        chain = chain_by_id[chain_id]
        target_joint_ids = {
            semantic_id: semantic_targets[semantic_id]
            for semantic_id in chain.semantic_joint_ids
        }
        missing_runtime = sorted(set(target_joint_ids.values()) - set(runtime_index))
        if missing_runtime:
            raise ValueError(
                f"chain {chain_id!r} QA joints are absent from runtime order: "
                f"{missing_runtime}"
            )
        result.append(
            SemanticChainSamples(
                chain_id=chain_id,
                rest_length_m=_chain_rest_length_m(
                    chain,
                    semantic_targets=semantic_targets,
                    joint_by_id=joint_by_id,
                ),
                terminal_positions_flv_m=np.asarray(
                    positions_by_chain[chain_id], dtype=np.float64
                ),
                joint_rotations_xyzw={
                    semantic_id: rotations[:, runtime_index[target_joint_id], :]
                    for semantic_id, target_joint_id in target_joint_ids.items()
                },
            )
        )
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--actions-npz", type=Path, required=True)
    parser.add_argument("--joint-mapping", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    visual = _input(args.visual_glb, owner="visual GLB", suffix=".glb")
    actions_path = _input(args.actions_npz, owner="baked actions", suffix=".npz")
    mapping_path = _input(
        args.joint_mapping, owner="Habitat joint mapping", suffix=".json"
    )
    profile_path = _input(args.profile, owner="motion profile", suffix=".json")
    output = _output(args.output)

    profile = load_motion_retarget_profile(profile_path)
    document = load_glb(visual)
    actions = read_baked_actions_npz(actions_path)
    if actions.source_glb_sha256 != document.sha256:
        raise ValueError("baked actions are not bound to visual GLB")
    if actions.sample_rate_hz != profile.qa_contract.sample_rate_hz:
        raise ValueError("baked action sample rate differs from QA profile")
    mapping_value = load_json(mapping_path)
    if mapping_value.get("source_glb_sha256") != document.sha256:
        raise ValueError("Habitat joint mapping is not bound to visual GLB")
    mapping = build_habitat_asset_mapping(
        document,
        actor_from_skin_root=mapping_value["actor_from_skin_root"],
        actor_from_skin_root_source=mapping_value["actor_from_skin_root_source"],
    )
    if mapping.joint_mapping_data() != mapping_value:
        raise ValueError("Habitat joint mapping differs from reconstructed mapping")
    if mapping.runtime_joint_order != actions.runtime_joint_order:
        raise ValueError("Habitat mapping and baked action joint orders differ")

    action = actions.action(profile.qa_semantic_action_id)
    report = evaluate_motion_qa(_samples(profile, mapping, action), profile.qa_contract)
    payload = {
        "schema": REPORT_SCHEMA,
        "status": report.status,
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "bindings": {
            "visual_glb_sha256": document.sha256,
            "baked_actions_sha256": _sha256(actions_path),
            "joint_mapping_sha256": _sha256(mapping_path),
            "motion_profile_sha256": _sha256(profile_path),
            "profile_id": profile.profile_id,
            "adapter_id": profile.adapter_id,
            "body_plan_id": profile.body_plan_id,
            "motion_family_id": profile.motion_family_id,
            "semantic_action_id": profile.qa_semantic_action_id,
        },
        "qa": report.to_dict(),
    }
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with output.open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": report.status,
                "output": str(output),
                "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "issue_count": len(report.issues),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

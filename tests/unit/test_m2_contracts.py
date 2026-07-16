from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from avengine.m2.contracts import (
    ANIMAL_SCHEMA,
    APPLIED_STATE_HASH_ALGORITHM,
    CAPTURE_SCHEMA,
    CONTACT_ORDER,
    POSE_HASH_ALGORITHM,
    REQUIRED_FILE_ROLES,
    ContractError,
    compute_applied_state_hash,
    compute_pose_hash,
    load_and_validate_inputs,
    validate_animal_asset_package,
    validate_capture_request,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
IDENTITY_TRANSFORM = {
    "translation_m": [0.0, 0.0, 0.0],
    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
}
JOINT_ORDER = [
    "root",
    "spine",
    "head",
    "paw_front_left",
    "paw_front_right",
    "paw_hind_left",
    "paw_hind_right",
]
RUNTIME_JOINT_ORDER = [joint_id for joint_id in JOINT_ORDER if joint_id != "root"]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _file_record(package_root: Path, role: str) -> dict:
    suffix = ".json"
    if role in {"visual", "collision_proxy"}:
        suffix = ".glb"
    elif role in {"idle_poses", "walk_poses"}:
        suffix = ".npz"
    elif role == "habitat_urdf":
        suffix = ".urdf"
    relative_path = Path("payload") / f"{role}{suffix}"
    path = package_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"m2 fixture {role}\n".encode()
    path.write_bytes(payload)
    return {
        "role": role,
        "path": relative_path.as_posix(),
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _asset_fixture(tmp_path: Path) -> tuple[Path, dict]:
    package_root = tmp_path / "animal_package"
    package_root.mkdir()
    roles = sorted(REQUIRED_FILE_ROLES | {"human_visual_review"})
    files = [_file_record(package_root, role) for role in roles]
    review_record = next(
        record for record in files if record["role"] == "human_visual_review"
    )
    anchors = [
        {
            "anchor_id": "body",
            "joint_id": "root",
            "joint_from_anchor": copy.deepcopy(IDENTITY_TRANSFORM),
        },
        {
            "anchor_id": "head",
            "joint_id": "head",
            "joint_from_anchor": copy.deepcopy(IDENTITY_TRANSFORM),
        },
        {
            "anchor_id": "muzzle",
            "joint_id": "head",
            "joint_from_anchor": {
                "translation_m": [0.0, 0.0, -0.1],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
    ]
    anchors.extend(
        {
            "anchor_id": contact_id,
            "joint_id": contact_id,
            "joint_from_anchor": copy.deepcopy(IDENTITY_TRANSFORM),
        }
        for contact_id in CONTACT_ORDER
    )
    asset = {
        "schema": ANIMAL_SCHEMA,
        "asset_id": "dog_canary_v1",
        "template_id": "audited_dog_template_v1",
        "body_plan_id": "quadruped_dog",
        "morphotype_id": "medium_dog",
        "admission_state": "canary_qualified",
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "-Z",
            "linear_unit": "meter",
            "quaternion_order": "xyzw",
        },
        "revisions": {
            "topology_sha256": "1" * 64,
            "uv_sha256": "2" * 64,
            "skeleton_revision": "dog_skeleton_v1",
            "weights_revision": "dog_weights_v1",
            "collision_revision": "dog_collision_v1",
            "action_revision": "walk_idle_v1",
        },
        "skeleton": {
            "root_joint_id": "root",
            "joint_order": JOINT_ORDER,
            "runtime_joint_order": RUNTIME_JOINT_ORDER,
            "joint_pose_encoding": "ordered_local_rotation_xyzw_float64",
        },
        "contacts": {"contact_order": CONTACT_ORDER},
        "anchors": anchors,
        "actions": [
            {
                "action_id": "idle",
                "poses_file_role": "idle_poses",
                "source_action_revision": "idle_native_v1",
                "sample_count": 41,
            },
            {
                "action_id": "walk",
                "poses_file_role": "walk_poses",
                "source_action_revision": "walk_native_v1",
                "sample_count": 41,
            },
        ],
        "files": files,
        "qualification": {
            "automatic_qa_status": "pass",
            "human_visual_review_status": "pass",
            "human_review_binding_sha256": review_record["sha256"],
            "decision_reason": "All bounded M2 asset gates passed.",
        },
        "provenance": {
            "source": "fixture://audited-dog",
            "source_revision": "fixture-v1",
            "source_sha256": "3" * 64,
            "license": "CC0-1.0",
            "allowed_use": "research_canary",
            "redistribution": "allowed",
        },
    }
    asset_path = package_root / "asset_manifest.json"
    _write_json(asset_path, asset)
    return asset_path, asset


def _request_fixture(
    tmp_path: Path,
    asset: dict,
    *,
    asset_manifest_sha256: str,
) -> tuple[Path, dict]:
    states = []
    for frame_index in range(75):
        action_id = "idle" if frame_index < 15 else "walk"
        action_frame = frame_index if action_id == "idle" else frame_index - 15
        state = {
            "frame_index": frame_index,
            "pts_ticks": frame_index * 3200,
            "action_id": action_id,
            "action_time_ticks": action_frame * 3200,
            "root_transform": {
                "translation_m": [frame_index * 0.01, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "joint_states": [
                {
                    "joint_id": joint_id,
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
                for joint_id in RUNTIME_JOINT_ORDER
            ],
            "contact_states": [
                {
                    "contact_id": contact_id,
                    "in_contact": (frame_index + contact_index) % 2 == 0,
                }
                for contact_index, contact_id in enumerate(CONTACT_ORDER)
            ],
            "mouth_state": {"open_ratio": 0.0, "vocalizing": False},
        }
        state["pose_hash"] = compute_pose_hash(asset, state)
        state["applied_state_hash"] = compute_applied_state_hash(
            asset,
            state,
            asset_manifest_sha256=asset_manifest_sha256,
        )
        states.append(state)
    request = {
        "schema": CAPTURE_SCHEMA,
        "request_id": "m2_dog_canary_request_v1",
        "room_id": "blender_custom_two_zone_v1",
        "asset_id": asset["asset_id"],
        "asset_manifest_sha256": asset_manifest_sha256,
        "seed": 17,
        "camera_rig_id": "camera_rig_0",
        "listener_id": "listener0",
        "view_ids": ["view0"],
        "modalities": ["rgb", "depth", "semantic"],
        "runtime_joint_order": RUNTIME_JOINT_ORDER,
        "contact_order": CONTACT_ORDER,
        "pose_hash_algorithm": POSE_HASH_ALGORITHM,
        "applied_state_hash_algorithm": APPLIED_STATE_HASH_ALGORITHM,
        "capture_policy": {
            "state_evaluation": "explicit_fixed_state",
            "advance_clock_between_modalities": False,
            "free_running_animation": False,
        },
        "states": states,
    }
    request_path = tmp_path / "capture_request.json"
    _write_json(request_path, request)
    return request_path, request


def _valid_fixture(tmp_path: Path) -> tuple[Path, dict, Path, dict]:
    asset_path, asset = _asset_fixture(tmp_path)
    asset_manifest_sha256 = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    request_path, request = _request_fixture(
        tmp_path,
        asset,
        asset_manifest_sha256=asset_manifest_sha256,
    )
    return asset_path, asset, request_path, request


def _validate_request(asset_path: Path, asset: dict, request: dict) -> list[str]:
    return validate_capture_request(
        request,
        asset=asset,
        asset_manifest_sha256=hashlib.sha256(asset_path.read_bytes()).hexdigest(),
    )


def test_m2_json_schemas_are_valid_draft_2020_12() -> None:
    for filename in (
        "animal_asset_package_v1.schema.json",
        "m2_articulated_capture_request_v1.schema.json",
    ):
        schema = json.loads((REPOSITORY_ROOT / "schemas" / filename).read_text())
        Draft202012Validator.check_schema(schema)


def test_valid_package_and_exact_75_state_request_load(tmp_path: Path) -> None:
    asset_path, _, request_path, _ = _valid_fixture(tmp_path)

    validated = load_and_validate_inputs(asset_path, request_path)

    assert validated.asset["admission_state"] == "canary_qualified"
    assert len(validated.request["states"]) == 75
    assert validated.request["view_ids"] == ["view0"]
    assert validated.asset["skeleton"]["joint_order"] == JOINT_ORDER
    assert validated.asset["skeleton"]["runtime_joint_order"] == RUNTIME_JOINT_ORDER
    assert validated.request["runtime_joint_order"] == RUNTIME_JOINT_ORDER
    assert all(
        [joint["joint_id"] for joint in state["joint_states"]] == RUNTIME_JOINT_ORDER
        and "root" not in {joint["joint_id"] for joint in state["joint_states"]}
        for state in validated.request["states"]
    )


def test_approved_for_dataset_is_forbidden_in_m2(tmp_path: Path) -> None:
    asset_path, asset, _, _ = _valid_fixture(tmp_path)
    asset["admission_state"] = "approved_for_dataset"

    errors = validate_animal_asset_package(asset, manifest_path=asset_path)

    assert any("approved_for_dataset is M6-only" in error for error in errors)


@pytest.mark.parametrize(
    "admission_state",
    ["research_candidate", "rejected", "admission_blocked"],
)
def test_capture_rejects_every_nonqualified_asset(
    tmp_path: Path, admission_state: str
) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    asset["admission_state"] = admission_state

    errors = _validate_request(asset_path, asset, request)

    assert "M2 capture accepts only a canary_qualified animal package" in errors


def test_canary_qualified_requires_both_qa_gates_and_bound_review(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, _ = _valid_fixture(tmp_path)
    asset["qualification"]["human_visual_review_status"] = "not_run"
    asset["qualification"]["human_review_binding_sha256"] = "0" * 64

    errors = validate_animal_asset_package(asset, manifest_path=asset_path)

    assert any("automatic QA and human visual review pass" in error for error in errors)
    assert any("hash-bound human_visual_review" in error for error in errors)


def test_rejected_and_blocked_states_require_truthful_gate_statuses(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, _ = _valid_fixture(tmp_path)
    asset["admission_state"] = "rejected"
    rejected_errors = validate_animal_asset_package(asset, manifest_path=asset_path)
    asset["admission_state"] = "admission_blocked"
    blocked_errors = validate_animal_asset_package(asset, manifest_path=asset_path)

    assert any("rejected requires" in error for error in rejected_errors)
    assert any("admission_blocked requires" in error for error in blocked_errors)


def test_research_candidate_cannot_conceal_a_hard_failure(tmp_path: Path) -> None:
    asset_path, asset, _, _ = _valid_fixture(tmp_path)
    asset["admission_state"] = "research_candidate"
    asset["qualification"]["automatic_qa_status"] = "fail"

    errors = validate_animal_asset_package(asset, manifest_path=asset_path)

    assert any("cannot conceal a failed" in error for error in errors)


def test_package_rehashes_every_declared_file(tmp_path: Path) -> None:
    asset_path, asset, _, _ = _valid_fixture(tmp_path)
    visual = next(record for record in asset["files"] if record["role"] == "visual")
    (asset_path.parent / visual["path"]).write_bytes(b"tampered")

    errors = validate_animal_asset_package(asset, manifest_path=asset_path)

    assert any(
        "files" in error and "byte_size does not match" in error for error in errors
    )
    assert any(
        "files" in error and "sha256 does not match" in error for error in errors
    )


def test_package_requires_complete_unique_role_and_path_closure(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, _ = _valid_fixture(tmp_path)
    asset["files"] = [
        record for record in asset["files"] if record["role"] != "collision_proxy"
    ]
    asset["files"][1]["path"] = asset["files"][0]["path"]

    errors = validate_animal_asset_package(asset, manifest_path=asset_path)

    assert any("missing required M2 roles" in error for error in errors)
    assert any("same path" in error for error in errors)


def test_package_rejects_path_escape(tmp_path: Path) -> None:
    asset_path, asset, _, _ = _valid_fixture(tmp_path)
    outside = tmp_path / "outside.glb"
    outside.write_bytes(b"outside")
    visual = next(record for record in asset["files"] if record["role"] == "visual")
    visual.update(
        {
            "path": "../outside.glb",
            "byte_size": outside.stat().st_size,
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        }
    )

    errors = validate_animal_asset_package(asset, manifest_path=asset_path)

    assert any("must be relative to the package manifest" in error for error in errors)


@pytest.mark.parametrize("raw_path", ["$HOME/visual.glb", "~/visual.glb"])
def test_package_paths_cannot_depend_on_environment_or_home_expansion(
    tmp_path: Path, raw_path: str
) -> None:
    asset_path, asset, _, _ = _valid_fixture(tmp_path)
    visual = next(record for record in asset["files"] if record["role"] == "visual")
    visual["path"] = raw_path

    errors = validate_animal_asset_package(asset, manifest_path=asset_path)

    assert any("must be relative to the package manifest" in error for error in errors)


def test_package_requires_ordered_skeleton_contacts_and_anchor_joint_mapping(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, _ = _valid_fixture(tmp_path)
    asset["skeleton"]["root_joint_id"] = "unknown"
    asset["contacts"]["contact_order"] = list(reversed(CONTACT_ORDER))
    asset["anchors"][0]["joint_id"] = "unknown"

    errors = validate_animal_asset_package(asset, manifest_path=asset_path)

    assert "skeleton.root_joint_id must occur in joint_order" in errors
    assert any("contacts.contact_order must be exactly" in error for error in errors)
    assert any("joint_id is not in skeleton.joint_order" in error for error in errors)


def test_runtime_joint_order_is_exact_skeleton_order_without_root(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, _ = _valid_fixture(tmp_path)
    asset["skeleton"]["runtime_joint_order"] = JOINT_ORDER

    errors = validate_animal_asset_package(asset, manifest_path=asset_path)

    assert any(
        "runtime_joint_order must equal joint_order with root_joint_id removed" in error
        for error in errors
    )


def test_request_requires_exactly_75_sequential_states(tmp_path: Path) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    request["states"] = request["states"][:-1]
    request["states"][3]["frame_index"] = 9
    request["states"][4]["pts_ticks"] = 1

    errors = _validate_request(asset_path, asset, request)

    assert "M2 capture requires exactly 75 states" in errors
    assert "states[3].frame_index must equal 3" in errors
    assert "states[4].pts_ticks must equal 12800" in errors


def test_request_requires_single_view0_and_three_ordered_modalities(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    request["view_ids"] = ["view0", "view1"]
    request["modalities"] = ["depth", "rgb", "semantic"]

    errors = _validate_request(asset_path, asset, request)

    assert "M2 formal view_ids must be exactly ['view0']" in errors
    assert any("modalities must be ordered exactly" in error for error in errors)


def test_each_state_must_follow_the_declared_joint_and_contact_order(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    request["states"][0]["joint_states"][0:2] = reversed(
        request["states"][0]["joint_states"][0:2]
    )
    request["states"][1]["contact_states"] = list(
        reversed(request["states"][1]["contact_states"])
    )

    errors = _validate_request(asset_path, asset, request)

    assert any("joint_states must follow" in error for error in errors)
    assert any("contact_states must follow" in error for error in errors)


def test_frame_joint_states_cannot_reintroduce_the_runtime_root(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    request["states"][0]["joint_states"].insert(
        0,
        {"joint_id": "root", "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
    )

    errors = _validate_request(asset_path, asset, request)

    assert any(
        "states[0].joint_states must follow the package runtime_joint_order exactly"
        in error
        for error in errors
    )


def test_request_runtime_joint_order_must_match_the_rootless_asset_order(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    request["runtime_joint_order"] = list(reversed(RUNTIME_JOINT_ORDER))

    errors = _validate_request(asset_path, asset, request)

    assert "request runtime_joint_order must exactly match the animal package" in errors


def test_noncanonical_quaternion_is_rejected_even_if_hashes_are_recomputed(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    state = request["states"][0]
    state["joint_states"][0]["rotation_xyzw"] = [0.0, 0.0, 0.0, -1.0]
    state["pose_hash"] = compute_pose_hash(asset, state)
    state["applied_state_hash"] = compute_applied_state_hash(
        asset,
        state,
        asset_manifest_sha256=request["asset_manifest_sha256"],
    )

    errors = _validate_request(asset_path, asset, request)

    assert any("canonical quaternion hemisphere" in error for error in errors)


def test_visual_mouth_open_ratio_must_remain_zero(tmp_path: Path) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    state = request["states"][12]
    state["mouth_state"]["open_ratio"] = 0.1
    state["applied_state_hash"] = compute_applied_state_hash(
        asset,
        state,
        asset_manifest_sha256=request["asset_manifest_sha256"],
    )

    errors = _validate_request(asset_path, asset, request)

    assert "states[12].mouth_state.open_ratio must be exactly 0.0" in errors


def test_pose_and_applied_state_hashes_are_independent_and_recomputed(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    state = request["states"][20]
    assert state["pose_hash"] != state["applied_state_hash"]
    state["pose_hash"] = state["applied_state_hash"]

    errors = _validate_request(asset_path, asset, request)

    assert "states[20].pose_hash does not match canonical joint pose" in errors
    assert "states[20] must keep pose_hash and applied_state_hash separate" in errors


def test_applied_state_hash_binds_root_contact_action_and_tick(tmp_path: Path) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    state = request["states"][30]
    original_pose_hash = state["pose_hash"]
    state["root_transform"]["translation_m"][0] += 1.0

    errors = _validate_request(asset_path, asset, request)

    assert state["pose_hash"] == original_pose_hash
    assert any("applied_state_hash does not match" in error for error in errors)


def test_manifest_hash_binds_request_and_applied_state_but_not_pose(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    state = request["states"][0]
    actual_manifest_hash = request["asset_manifest_sha256"]
    other_manifest_hash = "0" * 64 if actual_manifest_hash != "0" * 64 else "1" * 64

    pose_hash = compute_pose_hash(asset, state)
    actual_applied_hash = compute_applied_state_hash(
        asset,
        state,
        asset_manifest_sha256=actual_manifest_hash,
    )
    other_applied_hash = compute_applied_state_hash(
        asset,
        state,
        asset_manifest_sha256=other_manifest_hash,
    )

    assert compute_pose_hash(asset, state) == pose_hash
    assert actual_applied_hash != other_applied_hash

    request["asset_manifest_sha256"] = other_manifest_hash
    state["applied_state_hash"] = other_applied_hash
    errors = _validate_request(asset_path, asset, request)

    assert "request asset_manifest_sha256 does not match the animal package" in errors
    assert any("applied_state_hash does not match" in error for error in errors)


def test_hashes_canonicalize_signed_zero_and_quaternion_hemisphere(
    tmp_path: Path,
) -> None:
    _, asset, _, request = _valid_fixture(tmp_path)
    state = request["states"][0]
    equivalent = copy.deepcopy(state)
    equivalent["joint_states"][0]["rotation_xyzw"] = [
        -0.0,
        -0.0,
        -0.0,
        -1.0,
    ]
    equivalent["root_transform"] = {
        "translation_m": [-0.0, -0.0, -0.0],
        "rotation_xyzw": [-0.0, -0.0, -0.0, -1.0],
    }
    equivalent["mouth_state"]["open_ratio"] = -0.0

    assert compute_pose_hash(asset, equivalent) == compute_pose_hash(asset, state)
    assert compute_applied_state_hash(
        asset,
        equivalent,
        asset_manifest_sha256=request["asset_manifest_sha256"],
    ) == compute_applied_state_hash(
        asset,
        state,
        asset_manifest_sha256=request["asset_manifest_sha256"],
    )


def test_the_formal_sequence_must_exercise_walk_and_idle(tmp_path: Path) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    for state in request["states"]:
        state["action_id"] = "idle"
        state["applied_state_hash"] = compute_applied_state_hash(
            asset,
            state,
            asset_manifest_sha256=request["asset_manifest_sha256"],
        )

    errors = _validate_request(asset_path, asset, request)

    assert "the 75-state M2 capture must exercise both idle and walk" in errors


def test_capture_policy_forbids_free_running_or_modality_clock_advance(
    tmp_path: Path,
) -> None:
    asset_path, asset, _, request = _valid_fixture(tmp_path)
    request["capture_policy"]["free_running_animation"] = True
    request["capture_policy"]["advance_clock_between_modalities"] = True

    errors = _validate_request(asset_path, asset, request)

    assert any("explicit fixed states" in error for error in errors)


def test_load_reports_both_asset_and_request_failures(tmp_path: Path) -> None:
    asset_path, asset, request_path, request = _valid_fixture(tmp_path)
    asset["admission_state"] = "research_candidate"
    _write_json(asset_path, asset)
    request["view_ids"] = ["view1"]
    _write_json(request_path, request)

    with pytest.raises(ContractError) as caught:
        load_and_validate_inputs(asset_path, request_path)

    assert any(error.startswith("request:") for error in caught.value.errors)
    assert any("canary_qualified" in error for error in caught.value.errors)

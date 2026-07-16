from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    resolve_declared_path,
    sha256_file,
)
from avengine.contracts.transforms import validate_transform


ANIMAL_SCHEMA = "avengine_animal_asset_package_v1"
CAPTURE_SCHEMA = "avengine_m2_articulated_capture_request_v1"
POSE_HASH_ALGORITHM = "avengine_m2_pose_hash_v1"
APPLIED_STATE_HASH_ALGORITHM = "avengine_m2_applied_state_hash_v1"

ADMISSION_STATES = {
    "research_candidate",
    "canary_qualified",
    "rejected",
    "admission_blocked",
}
CHECK_STATUSES = {"pass", "fail", "blocked", "not_run"}
FORMAL_VIEW_IDS = ["view0"]
FORMAL_MODALITIES = ["rgb", "depth", "semantic"]
CONTACT_ORDER = [
    "paw_front_left",
    "paw_front_right",
    "paw_hind_left",
    "paw_hind_right",
]
REQUIRED_ANCHORS = {
    "head",
    "muzzle",
    "body",
    *CONTACT_ORDER,
}
REQUIRED_FILE_ROLES = {
    "visual",
    "collision_proxy",
    "skeleton_manifest",
    "skinning_manifest",
    "emitter_anchors",
    "action_manifest",
    "idle_poses",
    "walk_poses",
    "contact_phases",
    "static_geometry_qa",
    "deformation_qa",
    "animation_qa",
    "provenance_manifest",
    "habitat_urdf",
    "habitat_ao_config",
    "habitat_joint_mapping",
}
OPTIONAL_FILE_ROLES = {"human_visual_review"}
ALLOWED_FILE_ROLES = REQUIRED_FILE_ROLES | OPTIONAL_FILE_ROLES

_SCHEMA_FILES = {
    ANIMAL_SCHEMA: "animal_asset_package_v1.schema.json",
    CAPTURE_SCHEMA: "m2_articulated_capture_request_v1.schema.json",
}


class ContractError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class ValidatedM2Inputs:
    asset_path: Path
    request_path: Path
    asset: dict[str, Any]
    request: dict[str, Any]


def _json_schema_errors(value: Any, schema_name: str) -> list[str]:
    filename = _SCHEMA_FILES[schema_name]
    source_path = Path(__file__).resolve().parents[3] / "schemas" / filename
    installed_path = Path(sys.prefix) / "share" / "avengine" / "schemas" / filename
    schema_path = source_path if source_path.is_file() else installed_path
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_float(value: Any) -> float:
    """Return a JSON-hash float with both signed zeros represented as +0.0."""

    number = float(value)
    return 0.0 if number == 0.0 else number


def _quaternion_sign_component(quaternion: np.ndarray) -> float:
    """Select the component that defines the canonical quaternion hemisphere."""

    sign_component = float(quaternion[3])
    if math.isclose(sign_component, 0.0, rel_tol=0.0, abs_tol=1e-15):
        for component in quaternion[:3]:
            if not math.isclose(float(component), 0.0, rel_tol=0.0, abs_tol=1e-15):
                return float(component)
    return sign_component


def _canonical_quaternion(value: Any) -> list[float]:
    quaternion = np.asarray([float(component) for component in value], dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("quaternion must contain four components")
    if _quaternion_sign_component(quaternion) < 0.0:
        quaternion = -quaternion
    return [_canonical_float(component) for component in quaternion]


def _canonical_quaternion_errors(value: Any, name: str) -> list[str]:
    if not (
        isinstance(value, list)
        and len(value) == 4
        and all(_is_number(component) for component in value)
    ):
        return [f"{name} must contain four finite numbers"]
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    errors: list[str] = []
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"{name} must already be unit normalized")
        return errors

    # q and -q encode the same rotation. A fixed hemisphere makes pose hashes
    # independent of an exporter's sign choice.
    if _quaternion_sign_component(quaternion) < 0.0:
        errors.append(f"{name} must use the canonical quaternion hemisphere")
    return errors


def _transform_errors(value: Any, name: str) -> list[str]:
    errors = validate_transform(value, name=name)
    if isinstance(value, dict):
        errors.extend(
            _canonical_quaternion_errors(
                value.get("rotation_xyzw"), f"{name}.rotation_xyzw"
            )
        )
    return errors


def _canonical_transform(value: dict[str, Any]) -> dict[str, list[float]]:
    return {
        "translation_m": [
            _canonical_float(component) for component in value["translation_m"]
        ],
        "rotation_xyzw": _canonical_quaternion(value["rotation_xyzw"]),
    }


def _canonical_joint_states(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "joint_id": joint["joint_id"],
            "rotation_xyzw": _canonical_quaternion(joint["rotation_xyzw"]),
        }
        for joint in state["joint_states"]
    ]


def _canonical_contact_states(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "contact_id": contact["contact_id"],
            "in_contact": bool(contact["in_contact"]),
        }
        for contact in state["contact_states"]
    ]


def compute_pose_hash(asset: dict[str, Any], state: dict[str, Any]) -> str:
    """Hash only the canonical local skeletal pose, not the world root state."""

    payload = {
        "schema": POSE_HASH_ALGORITHM,
        "asset_id": asset["asset_id"],
        "skeleton_revision": asset["revisions"]["skeleton_revision"],
        "joint_states": _canonical_joint_states(state),
    }
    return canonical_json_sha256(payload)


def compute_applied_state_hash(
    asset: dict[str, Any],
    state: dict[str, Any],
    *,
    asset_manifest_sha256: str,
) -> str:
    """Hash the full state applied for one formal frame.

    This intentionally nests the independently recomputed pose hash so callers
    cannot make the world-state hash depend on an unverified declared hash.
    """

    if not _is_lower_sha256(asset_manifest_sha256):
        raise ValueError("asset_manifest_sha256 must be lowercase SHA-256")

    mouth_state = state["mouth_state"]
    payload = {
        "schema": APPLIED_STATE_HASH_ALGORITHM,
        "asset_id": asset["asset_id"],
        "asset_manifest_sha256": asset_manifest_sha256,
        "frame_index": int(state["frame_index"]),
        "pts_ticks": int(state["pts_ticks"]),
        "action_id": state["action_id"],
        "action_time_ticks": int(state["action_time_ticks"]),
        "root_transform": _canonical_transform(state["root_transform"]),
        "pose_hash": compute_pose_hash(asset, state),
        "contact_states": _canonical_contact_states(state),
        "mouth_state": {
            "open_ratio": _canonical_float(mouth_state["open_ratio"]),
            "vocalizing": bool(mouth_state["vocalizing"]),
        },
    }
    return canonical_json_sha256(payload)


def _validate_file_closure(
    asset: dict[str, Any], manifest_path: str | Path | None
) -> list[str]:
    if manifest_path is None:
        return ["manifest_path is required for M2 package path/hash closure"]

    resolved_manifest = Path(manifest_path).resolve()
    package_root = resolved_manifest.parent
    records = asset.get("files")
    if not isinstance(records, list):
        return ["files must be an array"]

    errors: list[str] = []
    roles: set[str] = set()
    resolved_paths: set[Path] = set()
    for index, record in enumerate(records):
        prefix = f"files[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix} must be an object")
            continue
        role = record.get("role")
        raw_path = record.get("path")
        if not isinstance(role, str) or not role:
            errors.append(f"{prefix}.role must be a non-empty string")
        elif role not in ALLOWED_FILE_ROLES:
            errors.append(f"{prefix}.role is not an M2 package role: {role!r}")
        elif role in roles:
            errors.append(f"file role is duplicated: {role}")
        else:
            roles.add(role)

        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"{prefix}.path must be a non-empty relative path")
            continue
        path_parts = Path(raw_path).parts
        if (
            Path(raw_path).is_absolute()
            or raw_path.startswith("~")
            or "$" in raw_path
            or any(part in {".", ".."} for part in path_parts)
        ):
            errors.append(f"{prefix}.path must be relative to the package manifest")
            continue
        path_cursor = package_root
        has_symlink = False
        for part in path_parts:
            path_cursor /= part
            if path_cursor.is_symlink():
                has_symlink = True
                break
        if has_symlink:
            errors.append(f"{prefix}.path must not be a symbolic link")
            continue
        try:
            resolved = resolve_declared_path(raw_path, manifest_dir=package_root)
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"{prefix}.path cannot be resolved: {error}")
            continue
        if resolved == resolved_manifest:
            errors.append(f"{prefix}.path cannot refer to the package manifest itself")
        if resolved in resolved_paths:
            errors.append(f"multiple file roles resolve to the same path: {raw_path}")
        resolved_paths.add(resolved)
        if not resolved.is_file():
            errors.append(f"{prefix}.path is not a regular file: {raw_path}")
            continue

        expected_size = record.get("byte_size")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or resolved.stat().st_size != expected_size
        ):
            errors.append(f"{prefix}.byte_size does not match {raw_path}")
        expected_hash = record.get("sha256")
        if (
            not _is_lower_sha256(expected_hash)
            or sha256_file(resolved) != expected_hash
        ):
            errors.append(f"{prefix}.sha256 does not match {raw_path}")

    missing_roles = sorted(REQUIRED_FILE_ROLES - roles)
    if missing_roles:
        errors.append(f"files are missing required M2 roles: {missing_roles}")
    return errors


def validate_animal_asset_package(
    asset: dict[str, Any], *, manifest_path: str | Path | None = None
) -> list[str]:
    errors = _json_schema_errors(asset, ANIMAL_SCHEMA)
    if asset.get("schema") != ANIMAL_SCHEMA:
        errors.append(f"schema must be {ANIMAL_SCHEMA!r}")
    if asset.get("admission_state") not in ADMISSION_STATES:
        errors.append(
            "admission_state must be research_candidate, canary_qualified, "
            "rejected, or admission_blocked; approved_for_dataset is M6-only"
        )

    skeleton = asset.get("skeleton")
    skeleton_joint_order: list[str] = []
    runtime_joint_order: list[str] = []
    if not isinstance(skeleton, dict):
        errors.append("skeleton must be an object")
    else:
        raw_joint_order = skeleton.get("joint_order")
        if isinstance(raw_joint_order, list) and all(
            isinstance(joint, str) and joint for joint in raw_joint_order
        ):
            skeleton_joint_order = raw_joint_order
            if len(set(skeleton_joint_order)) != len(skeleton_joint_order):
                errors.append("skeleton.joint_order must contain unique joint IDs")
        else:
            errors.append("skeleton.joint_order must contain ordered joint IDs")
        root_joint_id = skeleton.get("root_joint_id")
        if root_joint_id not in skeleton_joint_order:
            errors.append("skeleton.root_joint_id must occur in joint_order")
        raw_runtime_joint_order = skeleton.get("runtime_joint_order")
        if isinstance(raw_runtime_joint_order, list) and all(
            isinstance(joint, str) and joint for joint in raw_runtime_joint_order
        ):
            runtime_joint_order = raw_runtime_joint_order
            if len(set(runtime_joint_order)) != len(runtime_joint_order):
                errors.append(
                    "skeleton.runtime_joint_order must contain unique joint IDs"
                )
        else:
            errors.append("skeleton.runtime_joint_order must contain ordered joint IDs")
        expected_runtime_joint_order = [
            joint for joint in skeleton_joint_order if joint != root_joint_id
        ]
        if runtime_joint_order != expected_runtime_joint_order:
            errors.append(
                "skeleton.runtime_joint_order must equal joint_order with "
                "root_joint_id removed"
            )
        if skeleton.get("joint_pose_encoding") != (
            "ordered_local_rotation_xyzw_float64"
        ):
            errors.append("skeleton.joint_pose_encoding is not the M2 encoding")

    contacts = asset.get("contacts")
    contact_order = (
        contacts.get("contact_order") if isinstance(contacts, dict) else None
    )
    if contact_order != CONTACT_ORDER:
        errors.append(f"contacts.contact_order must be exactly {CONTACT_ORDER}")

    anchors = asset.get("anchors")
    anchor_ids: set[str] = set()
    if not isinstance(anchors, list):
        errors.append("anchors must be an array")
    else:
        for index, anchor in enumerate(anchors):
            prefix = f"anchors[{index}]"
            if not isinstance(anchor, dict):
                errors.append(f"{prefix} must be an object")
                continue
            anchor_id = anchor.get("anchor_id")
            if isinstance(anchor_id, str):
                if anchor_id in anchor_ids:
                    errors.append(f"anchor_id is duplicated: {anchor_id}")
                anchor_ids.add(anchor_id)
            if anchor.get("joint_id") not in skeleton_joint_order:
                errors.append(f"{prefix}.joint_id is not in skeleton.joint_order")
            errors.extend(
                _transform_errors(
                    anchor.get("joint_from_anchor"), f"{prefix}.joint_from_anchor"
                )
            )
    missing_anchors = sorted(REQUIRED_ANCHORS - anchor_ids)
    if missing_anchors:
        errors.append(f"anchors are missing required semantic IDs: {missing_anchors}")

    actions = asset.get("actions")
    action_ids: set[str] = set()
    if not isinstance(actions, list):
        errors.append("actions must be an array")
    else:
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                errors.append(f"actions[{index}] must be an object")
                continue
            action_id = action.get("action_id")
            role = action.get("poses_file_role")
            if isinstance(action_id, str):
                if action_id in action_ids:
                    errors.append(f"action_id is duplicated: {action_id}")
                action_ids.add(action_id)
            expected_role = {"idle": "idle_poses", "walk": "walk_poses"}.get(action_id)
            if expected_role is not None and role != expected_role:
                errors.append(
                    f"actions[{index}].poses_file_role must be {expected_role!r}"
                )
    if action_ids != {"idle", "walk"}:
        errors.append("actions must contain exactly idle and walk")

    records = asset.get("files")
    records_by_role = (
        {
            record.get("role"): record
            for record in records
            if isinstance(record, dict) and isinstance(record.get("role"), str)
        }
        if isinstance(records, list)
        else {}
    )

    qualification = asset.get("qualification")
    if not isinstance(qualification, dict):
        errors.append("qualification must be an object")
        qualification = {}
    automatic_status = qualification.get("automatic_qa_status")
    human_status = qualification.get("human_visual_review_status")
    state = asset.get("admission_state")
    if state == "canary_qualified":
        if automatic_status != "pass" or human_status != "pass":
            errors.append(
                "canary_qualified requires automatic QA and human visual review pass"
            )
        review_record = records_by_role.get("human_visual_review")
        binding = qualification.get("human_review_binding_sha256")
        if not isinstance(review_record, dict) or binding != review_record.get(
            "sha256"
        ):
            errors.append(
                "canary_qualified requires a hash-bound human_visual_review file"
            )
        provenance = asset.get("provenance")
        if (
            isinstance(provenance, dict)
            and provenance.get("allowed_use") == "review_required"
        ):
            errors.append("canary_qualified cannot have review_required allowed_use")
    elif state == "rejected":
        if "fail" not in {automatic_status, human_status}:
            errors.append("rejected requires at least one failed qualification gate")
    elif state == "admission_blocked":
        statuses = {automatic_status, human_status}
        if "fail" in statuses or not statuses.intersection({"blocked", "not_run"}):
            errors.append(
                "admission_blocked requires a blocked/not_run gate and no failed gate"
            )
    elif state == "research_candidate" and "fail" in {
        automatic_status,
        human_status,
    }:
        errors.append("research_candidate cannot conceal a failed qualification gate")

    errors.extend(_validate_file_closure(asset, manifest_path))
    return errors


def validate_capture_request(
    request: dict[str, Any],
    *,
    asset: dict[str, Any],
    asset_manifest_sha256: str,
) -> list[str]:
    errors = _json_schema_errors(request, CAPTURE_SCHEMA)
    if request.get("schema") != CAPTURE_SCHEMA:
        errors.append(f"schema must be {CAPTURE_SCHEMA!r}")
    if request.get("view_ids") != FORMAL_VIEW_IDS:
        errors.append("M2 formal view_ids must be exactly ['view0']")
    if request.get("modalities") != FORMAL_MODALITIES:
        errors.append(
            "M2 modalities must be ordered exactly ['rgb', 'depth', 'semantic']"
        )
    if request.get("pose_hash_algorithm") != POSE_HASH_ALGORITHM:
        errors.append(f"pose_hash_algorithm must be {POSE_HASH_ALGORITHM!r}")
    if request.get("applied_state_hash_algorithm") != APPLIED_STATE_HASH_ALGORITHM:
        errors.append(
            f"applied_state_hash_algorithm must be {APPLIED_STATE_HASH_ALGORITHM!r}"
        )

    if asset.get("admission_state") != "canary_qualified":
        errors.append("M2 capture accepts only a canary_qualified animal package")
    if request.get("asset_id") != asset.get("asset_id"):
        errors.append("request asset_id does not match the animal package")
    declared_asset_manifest_sha256 = request.get("asset_manifest_sha256")
    valid_asset_manifest_sha256 = _is_lower_sha256(asset_manifest_sha256)
    if not valid_asset_manifest_sha256:
        errors.append("expected asset_manifest_sha256 must be lowercase SHA-256")
    if not _is_lower_sha256(declared_asset_manifest_sha256):
        errors.append("request asset_manifest_sha256 must be lowercase SHA-256")
    elif declared_asset_manifest_sha256 != asset_manifest_sha256:
        errors.append("request asset_manifest_sha256 does not match the animal package")

    asset_skeleton = asset.get("skeleton")
    expected_runtime_joint_order: list[str] | None = None
    declared_runtime_joint_order: Any = None
    if isinstance(asset_skeleton, dict):
        skeleton_joint_order = asset_skeleton.get("joint_order")
        root_joint_id = asset_skeleton.get("root_joint_id")
        declared_runtime_joint_order = asset_skeleton.get("runtime_joint_order")
        if isinstance(skeleton_joint_order, list) and isinstance(root_joint_id, str):
            expected_runtime_joint_order = [
                joint for joint in skeleton_joint_order if joint != root_joint_id
            ]
    if declared_runtime_joint_order != expected_runtime_joint_order:
        errors.append(
            "animal package runtime_joint_order must equal its skeleton order "
            "with root removed"
        )
    if request.get("runtime_joint_order") != expected_runtime_joint_order:
        errors.append(
            "request runtime_joint_order must exactly match the animal package"
        )
    asset_contacts = asset.get("contacts")
    expected_contact_order = (
        asset_contacts.get("contact_order")
        if isinstance(asset_contacts, dict)
        else None
    )
    if request.get("contact_order") != expected_contact_order:
        errors.append("request contact_order must exactly match the animal package")

    policy = request.get("capture_policy")
    expected_policy = {
        "state_evaluation": "explicit_fixed_state",
        "advance_clock_between_modalities": False,
        "free_running_animation": False,
    }
    if policy != expected_policy:
        errors.append(
            "capture_policy must use explicit fixed states without modality clock "
            "advancement or free-running animation"
        )

    states = request.get("states")
    if not isinstance(states, list):
        errors.append("states must be an array")
        return errors
    if len(states) != 75:
        errors.append("M2 capture requires exactly 75 states")

    observed_actions: set[str] = set()
    for index, state in enumerate(states):
        prefix = f"states[{index}]"
        if not isinstance(state, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if state.get("frame_index") != index:
            errors.append(f"{prefix}.frame_index must equal {index}")
        if state.get("pts_ticks") != index * 3200:
            errors.append(f"{prefix}.pts_ticks must equal {index * 3200}")
        action_id = state.get("action_id")
        if isinstance(action_id, str):
            observed_actions.add(action_id)

        errors.extend(
            _transform_errors(state.get("root_transform"), f"{prefix}.root_transform")
        )

        joint_states = state.get("joint_states")
        if not isinstance(joint_states, list):
            errors.append(f"{prefix}.joint_states must be an array")
        else:
            actual_joint_order = [
                joint.get("joint_id") if isinstance(joint, dict) else None
                for joint in joint_states
            ]
            if actual_joint_order != expected_runtime_joint_order:
                errors.append(
                    f"{prefix}.joint_states must follow the package "
                    "runtime_joint_order exactly"
                )
            for joint_index, joint in enumerate(joint_states):
                if not isinstance(joint, dict):
                    errors.append(
                        f"{prefix}.joint_states[{joint_index}] must be an object"
                    )
                    continue
                errors.extend(
                    _canonical_quaternion_errors(
                        joint.get("rotation_xyzw"),
                        f"{prefix}.joint_states[{joint_index}].rotation_xyzw",
                    )
                )

        contact_states = state.get("contact_states")
        if not isinstance(contact_states, list):
            errors.append(f"{prefix}.contact_states must be an array")
        else:
            actual_contact_order = [
                contact.get("contact_id") if isinstance(contact, dict) else None
                for contact in contact_states
            ]
            if actual_contact_order != expected_contact_order:
                errors.append(
                    f"{prefix}.contact_states must follow package contact_order exactly"
                )
            for contact_index, contact in enumerate(contact_states):
                if not isinstance(contact, dict) or not isinstance(
                    contact.get("in_contact"), bool
                ):
                    errors.append(
                        f"{prefix}.contact_states[{contact_index}].in_contact "
                        "must be a boolean"
                    )

        mouth_state = state.get("mouth_state")
        open_ratio = (
            mouth_state.get("open_ratio") if isinstance(mouth_state, dict) else None
        )
        if not _is_number(open_ratio) or float(open_ratio) != 0.0:
            errors.append(f"{prefix}.mouth_state.open_ratio must be exactly 0.0")

        declared_pose_hash = state.get("pose_hash")
        declared_applied_hash = state.get("applied_state_hash")
        if not _is_lower_sha256(declared_pose_hash):
            errors.append(f"{prefix}.pose_hash must be lowercase SHA-256")
        if not _is_lower_sha256(declared_applied_hash):
            errors.append(f"{prefix}.applied_state_hash must be lowercase SHA-256")
        try:
            expected_pose_hash = compute_pose_hash(asset, state)
            expected_applied_hash = compute_applied_state_hash(
                asset,
                state,
                asset_manifest_sha256=asset_manifest_sha256,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if declared_pose_hash != expected_pose_hash:
            errors.append(f"{prefix}.pose_hash does not match canonical joint pose")
        if declared_applied_hash != expected_applied_hash:
            errors.append(
                f"{prefix}.applied_state_hash does not match the full applied state"
            )
        if declared_pose_hash == declared_applied_hash:
            errors.append(
                f"{prefix} must keep pose_hash and applied_state_hash separate"
            )

    if observed_actions != {"idle", "walk"}:
        errors.append("the 75-state M2 capture must exercise both idle and walk")
    return errors


def load_and_validate_inputs(
    asset_path: str | Path, request_path: str | Path
) -> ValidatedM2Inputs:
    resolved_asset = Path(asset_path).resolve()
    resolved_request = Path(request_path).resolve()
    asset = load_json(resolved_asset)
    request = load_json(resolved_request)
    asset_manifest_sha256 = sha256_file(resolved_asset)
    errors = [
        f"asset: {error}"
        for error in validate_animal_asset_package(asset, manifest_path=resolved_asset)
    ]
    errors.extend(
        f"request: {error}"
        for error in validate_capture_request(
            request,
            asset=asset,
            asset_manifest_sha256=asset_manifest_sha256,
        )
    )
    if errors:
        raise ContractError(errors)
    return ValidatedM2Inputs(
        asset_path=resolved_asset,
        request_path=resolved_request,
        asset=asset,
        request=request,
    )

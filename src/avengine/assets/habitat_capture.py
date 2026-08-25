"""Formal fixed-state M2 capture on the pinned Habitat runtime.

The module keeps the formal boundary deliberately narrow:

* callers provide the immutable results of the M1 and M2
  ``load_and_validate_inputs`` functions;
* the M1 room and its single ``camera_rig_0/view0`` calibration are reused
  without creating another formal viewpoint;
* a fresh Simulator owns one kinematic articulated object;
* every frame explicitly writes the actor/skin-root transform and name-bound
  spherical joint positions, renders all three co-located modalities once,
  and proves that neither the clock nor applied state changed; and
* the returned evidence binds declared/recomputed state hashes to readable
  array artifacts and per-frame payload hashes.

The formal entrypoint emits no QA media or second viewpoint.  A separately
named research-review entrypoint may encode the same ``view0`` arrays as
review-only videos, but labels them with ``formal_view_ids=[]`` and an explicit
negative qualification claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.rooms.contracts import ValidatedM1Inputs
from avengine.rooms.evidence import array_sha256
from avengine.assets.actions import BakedActionSet, read_baked_actions_npz
from avengine.assets.contracts import (
    FORMAL_MODALITIES,
    FORMAL_VIEW_IDS,
    ValidatedM2Inputs,
    compute_applied_state_hash,
    compute_pose_hash,
    validate_animal_asset_package,
    validate_capture_request,
)
from avengine.assets.habitat import (
    AVENGINE_NATIVE_GLTF_SKIN_FRAME_KEY,
    HabitatJointBinding,
    HabitatLinkJointBlock,
    bind_habitat_link_layout,
)
from avengine.runtime_lock import RuntimeLockError, resolve_runtime_profile


EVIDENCE_SCHEMA = "avengine_m2_habitat_capture_evidence_v1"
READBACK_HASH_SCHEMA = "avengine_m2_habitat_readback_v1"
ARRAY_HASH_ALGORITHM = "avengine_array_sha256_v1"
FORMAL_RUNTIME_ROLES = (
    "visual",
    "habitat_urdf",
    "habitat_ao_config",
    "habitat_joint_mapping",
    "idle_poses",
    "walk_poses",
)
_ROOT_READBACK_ATOL = 2.0e-6
_JOINT_READBACK_ATOL = 2.0e-6
_FORMAL_ADMISSION_ERROR = "M2 capture accepts only a canary_qualified animal package"


Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


class HabitatCaptureError(RuntimeError):
    """The formal runtime could not prove an exact fixed-state capture."""


def _git_value(repository: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _locked_runtime_commit(repository_root: Path) -> str | None:
    try:
        lock_path = resolve_runtime_profile(repository_root, "m2")
    except RuntimeLockError:
        return None
    match = re.search(
        r"^habitat_runtime:\s*$.*?^\s+fork_governance_commit:\s+([0-9a-f]{40})\s*$",
        lock_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(1) if match else None


def _locked_native_binding_sha256(repository_root: Path) -> str | None:
    try:
        lock_path = resolve_runtime_profile(repository_root, "m2")
    except RuntimeLockError:
        return None
    match = re.search(
        r"^\s+required_m2_native_binding_sha256:\s+([0-9a-f]{64})\s*$",
        lock_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def _runtime_identity(
    *,
    runtime: Path,
    habitat_sim: Any,
    habitat_sim_bindings: Any,
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    runtime_commit = _git_value(runtime, "rev-parse", "HEAD")
    runtime_status = _git_value(runtime, "status", "--porcelain")
    avengine_commit = _git_value(repository_root, "rev-parse", "HEAD")
    avengine_status = _git_value(repository_root, "status", "--porcelain")
    locked_commit = _locked_runtime_commit(repository_root)
    locked_binding_sha256 = _locked_native_binding_sha256(repository_root)
    module_path = Path(habitat_sim.__file__).resolve()
    binding_path = Path(habitat_sim_bindings.__file__).resolve()
    native_binding_sha256 = sha256_file(binding_path)
    try:
        module_path.relative_to(runtime)
        binding_path.relative_to(runtime)
        binary_origin_matches = True
    except ValueError:
        binary_origin_matches = False
    lock_path = resolve_runtime_profile(repository_root, "m2")
    return {
        "avengine_commit": avengine_commit,
        "avengine_worktree_dirty": avengine_status != "",
        "avengine_git_status": avengine_status,
        "habitat_runtime_root": str(runtime),
        "habitat_runtime_commit": runtime_commit,
        "habitat_runtime_worktree_dirty": runtime_status != "",
        "habitat_runtime_git_status": runtime_status,
        "locked_habitat_runtime_commit": locked_commit,
        "runtime_commit_matches_lock": runtime_commit == locked_commit,
        "runtime_lock_sha256": sha256_file(lock_path),
        "habitat_module_path": str(module_path),
        "native_binding_path": str(binding_path),
        "native_binding_sha256": native_binding_sha256,
        "locked_native_binding_sha256": locked_binding_sha256,
        "native_binding_matches_lock": (native_binding_sha256 == locked_binding_sha256),
        "runtime_binary_origin_matches": binary_origin_matches,
        "habitat_python_version": getattr(habitat_sim, "__version__", None),
        "habitat_audio_enabled": bool(habitat_sim.audio_enabled),
        "habitat_bullet_enabled": bool(habitat_sim.built_with_bullet),
        "habitat_cuda_enabled": bool(habitat_sim.cuda_enabled),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _formal_runtime_identity_errors(identity: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if identity.get("habitat_runtime_worktree_dirty"):
        errors.append("Habitat runtime worktree is dirty")
    if identity.get("avengine_worktree_dirty"):
        errors.append("AVEngine worktree is dirty")
    if not identity.get("runtime_commit_matches_lock"):
        errors.append("Habitat runtime commit differs from lock")
    if not identity.get("runtime_binary_origin_matches"):
        errors.append("imported Habitat module/binding is outside runtime root")
    if not identity.get("native_binding_matches_lock"):
        errors.append("native Habitat binding SHA-256 differs from lock")
    return errors


@dataclass(frozen=True)
class RuntimeAssetBundle:
    """Hash-checked runtime files selected from M2 package roles."""

    paths_by_role: Mapping[str, Path]
    records_by_role: Mapping[str, Mapping[str, Any]]
    joint_mapping: Mapping[str, Any]
    actor_from_skin_root: Matrix4
    action_sets_by_role: Mapping[str, BakedActionSet]
    action_roles_by_id: Mapping[str, str]
    semantic_id: int


@dataclass(frozen=True)
class FrameApplication:
    """One validated request state resolved to exact runtime inputs."""

    frame_index: int
    pts_ticks: int
    action_id: str
    action_time_ticks: int
    effective_action_tick: int
    action_sample_index: int
    world_from_actor: Matrix4
    world_from_skin_root: Matrix4
    joint_rotations_xyzw: tuple[tuple[float, float, float, float], ...]
    declared_pose_hash: str
    recomputed_pose_hash: str
    declared_applied_state_hash: str
    recomputed_applied_state_hash: str


@dataclass(frozen=True)
class CapturedFrame:
    """Detached evidence and arrays from one fixed-state observation call."""

    record: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


def _matrix_tuple(value: np.ndarray) -> Matrix4:
    return tuple(tuple(float(component) for component in row) for row in value)  # type: ignore[return-value]


def quaternion_xyzw_to_matrix(value: Sequence[float]) -> np.ndarray:
    """Return a deterministic 3x3 matrix from a unit xyzw quaternion."""

    x, y, z, w = normalized_quaternion_xyzw(value)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_to_matrix(value: Mapping[str, Any]) -> np.ndarray:
    """Convert an already validated AVEngine transform to a 4x4 matrix."""

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_xyzw_to_matrix(value["rotation_xyzw"])
    translation = np.asarray(value["translation_m"], dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise HabitatCaptureError("root transform translation must be finite vec3")
    result[:3, 3] = translation
    return result


def _rigid_matrix(value: Any, *, owner: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HabitatCaptureError(f"{owner} must be a finite rigid 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise HabitatCaptureError(f"{owner} must be a finite rigid 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1.0e-7):
        raise HabitatCaptureError(f"{owner} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-7):
        raise HabitatCaptureError(f"{owner} rotation is not orthonormal")
    if not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-7
    ):
        raise HabitatCaptureError(f"{owner} rotation is not proper")
    return matrix


def validate_capture_context(
    inputs: Any,
    room_inputs: Any,
) -> list[str]:
    """Validate the cross-milestone fields that M1/M2 schemas cannot join."""

    errors: list[str] = []
    if not isinstance(inputs, ValidatedM2Inputs):
        errors.append(
            "inputs must be ValidatedM2Inputs from M2 load_and_validate_inputs"
        )
        return errors
    if not isinstance(room_inputs, ValidatedM1Inputs):
        errors.append(
            "room_inputs must be ValidatedM1Inputs from M1 load_and_validate_inputs"
        )
        return errors

    asset = inputs.asset
    request = inputs.request
    room = room_inputs.room
    room_request = room_inputs.request
    rig = room_request.get("primary_camera_rig", {})
    listener = room_request.get("listener", {})
    if asset.get("admission_state") != "canary_qualified":
        errors.append("formal Habitat capture accepts only canary_qualified assets")
    if request.get("view_ids") != FORMAL_VIEW_IDS:
        errors.append("M2 formal view_ids must remain exactly ['view0']")
    if request.get("modalities") != FORMAL_MODALITIES:
        errors.append("M2 formal modalities must remain rgb/depth/semantic")
    if request.get("room_id") != room.get("room_id"):
        errors.append("M2 request room_id differs from the validated M1 room")
    if request.get("camera_rig_id") != rig.get("rig_id"):
        errors.append("M2 camera_rig_id differs from the validated M1 rig")
    if rig.get("rig_id") != "camera_rig_0" or rig.get("view_id") != "view0":
        errors.append("M1 runtime input must provide only camera_rig_0/view0")
    if request.get("listener_id") != listener.get("listener_id"):
        errors.append("M2 listener_id differs from the validated M1 listener")
    if request.get("seed") != room_request.get("seed"):
        errors.append("M1 and M2 seeds must match for the reused room configuration")
    modality_names = [
        item.get("modality")
        for item in rig.get("modalities", [])
        if isinstance(item, dict)
    ]
    if sorted(modality_names) != sorted(FORMAL_MODALITIES):
        errors.append("M1 view0 must expose exactly co-located rgb/depth/semantic")
    return errors


def validate_research_review_context(
    inputs: Any,
    room_inputs: Any,
) -> list[str]:
    """Join M1/M2 inputs while requiring the non-formal research admission."""

    errors = validate_capture_context(inputs, room_inputs)
    if not isinstance(inputs, ValidatedM2Inputs) or not isinstance(
        room_inputs, ValidatedM1Inputs
    ):
        return errors
    try:
        errors.remove("formal Habitat capture accepts only canary_qualified assets")
    except ValueError:
        pass
    if inputs.asset.get("admission_state") != "research_candidate":
        errors.append("research review accepts only research_candidate assets")
    return errors


def load_research_review_inputs(
    asset_path: str | Path,
    request_path: str | Path,
) -> ValidatedM2Inputs:
    """Load a fully valid 75-state request for explicit review-only execution.

    The normal M2 request validator is still the authority.  This loader
    accepts exactly one otherwise-failing condition: the formal canary gate.
    Every schema, file closure, pose hash, applied-state hash, state count, and
    single-view requirement remains mandatory.
    """

    resolved_asset = Path(asset_path).resolve()
    resolved_request = Path(request_path).resolve()
    asset = load_json(resolved_asset)
    request = load_json(resolved_request)
    errors = [
        f"asset: {error}"
        for error in validate_animal_asset_package(asset, manifest_path=resolved_asset)
    ]
    if asset.get("admission_state") != "research_candidate":
        errors.append("asset: review loader requires research_candidate")
    request_errors = validate_capture_request(
        request,
        asset=asset,
        asset_manifest_sha256=sha256_file(resolved_asset),
    )
    if _FORMAL_ADMISSION_ERROR not in request_errors:
        errors.append("request: formal canary gate was not the sole admission blocker")
    errors.extend(
        f"request: {error}"
        for error in request_errors
        if error != _FORMAL_ADMISSION_ERROR
    )
    if errors:
        raise HabitatCaptureError("; ".join(errors))
    return ValidatedM2Inputs(
        asset_path=resolved_asset,
        request_path=resolved_request,
        asset=asset,
        request=request,
    )


def _reload_validated_context(
    inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
) -> tuple[ValidatedM2Inputs, ValidatedM1Inputs]:
    errors = validate_capture_context(inputs, room_inputs)
    if errors:
        raise HabitatCaptureError("; ".join(errors))

    # A dataclass can be constructed directly and its dictionaries are mutable.
    # Reloading both sources proves that the capture consumes the exact current
    # bytes accepted by the contract entrypoints, not a caller-mutated snapshot.
    from avengine.rooms.contracts import load_and_validate_inputs as load_m1_inputs
    from avengine.assets.contracts import load_and_validate_inputs as load_m2_inputs

    reloaded_m2 = load_m2_inputs(inputs.asset_path, inputs.request_path)
    reloaded_m1 = load_m1_inputs(room_inputs.room_path, room_inputs.request_path)
    if reloaded_m2 != inputs:
        raise HabitatCaptureError(
            "M2 validated inputs differ from current manifest bytes"
        )
    if reloaded_m1 != room_inputs:
        raise HabitatCaptureError(
            "M1 validated inputs differ from current manifest bytes"
        )
    errors = validate_capture_context(reloaded_m2, reloaded_m1)
    if errors:
        raise HabitatCaptureError("; ".join(errors))
    return reloaded_m2, reloaded_m1


def _reload_research_review_context(
    inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
) -> tuple[ValidatedM2Inputs, ValidatedM1Inputs]:
    errors = validate_research_review_context(inputs, room_inputs)
    if errors:
        raise HabitatCaptureError("; ".join(errors))
    from avengine.rooms.contracts import load_and_validate_inputs as load_m1_inputs

    reloaded_m2 = load_research_review_inputs(inputs.asset_path, inputs.request_path)
    reloaded_m1 = load_m1_inputs(room_inputs.room_path, room_inputs.request_path)
    if reloaded_m2 != inputs:
        raise HabitatCaptureError(
            "research M2 inputs differ from current manifest bytes"
        )
    if reloaded_m1 != room_inputs:
        raise HabitatCaptureError(
            "M1 validated inputs differ from current manifest bytes"
        )
    errors = validate_research_review_context(reloaded_m2, reloaded_m1)
    if errors:
        raise HabitatCaptureError("; ".join(errors))
    return reloaded_m2, reloaded_m1


def resolve_runtime_asset_paths(inputs: ValidatedM2Inputs) -> dict[str, Path]:
    """Resolve and rehash the six package roles used by formal Habitat capture."""

    records_by_role: dict[str, Mapping[str, Any]] = {}
    for record in inputs.asset.get("files", []):
        if isinstance(record, Mapping) and isinstance(record.get("role"), str):
            role = str(record["role"])
            if role in records_by_role:
                raise HabitatCaptureError(f"duplicate M2 file role: {role}")
            records_by_role[role] = record
    missing = [role for role in FORMAL_RUNTIME_ROLES if role not in records_by_role]
    if missing:
        raise HabitatCaptureError(f"M2 package lacks formal runtime roles: {missing}")

    package_root = inputs.asset_path.parent.resolve()
    paths: dict[str, Path] = {}
    for role in FORMAL_RUNTIME_ROLES:
        record = records_by_role[role]
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise HabitatCaptureError(f"M2 role {role} has no relative path")
        path = (package_root / raw_path).resolve()
        try:
            path.relative_to(package_root)
        except ValueError as exc:
            raise HabitatCaptureError(f"M2 role {role} escapes the package") from exc
        if path.is_symlink() or not path.is_file():
            raise HabitatCaptureError(f"M2 role {role} is not a regular package file")
        if path.stat().st_size != record.get("byte_size") or sha256_file(
            path
        ) != record.get("sha256"):
            raise HabitatCaptureError(f"M2 role {role} bytes changed after validation")
        paths[role] = path
    return paths


def _validate_joint_mapping(
    mapping: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
    visual_sha256: str,
) -> np.ndarray:
    skeleton = asset["skeleton"]
    if mapping.get("schema") != "avengine_m2_habitat_joint_mapping_v1":
        raise HabitatCaptureError("habitat_joint_mapping has an unsupported schema")
    if mapping.get("source_glb_sha256") != visual_sha256:
        raise HabitatCaptureError(
            "habitat_joint_mapping source GLB hash differs from visual"
        )
    if mapping.get("root_joint_id") != skeleton["root_joint_id"]:
        raise HabitatCaptureError(
            "habitat_joint_mapping root differs from asset skeleton"
        )
    if mapping.get("joint_order") != skeleton["joint_order"]:
        raise HabitatCaptureError(
            "habitat_joint_mapping joint_order differs from asset"
        )
    if mapping.get("runtime_joint_order") != skeleton["runtime_joint_order"]:
        raise HabitatCaptureError(
            "habitat_joint_mapping runtime_joint_order differs from asset"
        )
    coordinate = mapping.get("coordinate_system")
    if coordinate != asset.get("coordinate_system"):
        raise HabitatCaptureError(
            "habitat_joint_mapping coordinate system differs from asset"
        )
    if mapping.get("runtime_root_formula") != (
        "world_from_skin_root = world_from_actor @ actor_from_skin_root"
    ):
        raise HabitatCaptureError("habitat_joint_mapping root formula is unsupported")
    source = mapping.get("actor_from_skin_root_source")
    if not isinstance(source, str) or not source.strip():
        raise HabitatCaptureError(
            "habitat_joint_mapping lacks root-transform provenance"
        )
    layout = mapping.get("habitat_layout")
    if not isinstance(layout, Mapping):
        raise HabitatCaptureError("habitat_joint_mapping lacks habitat_layout")
    expected_count = 4 * len(skeleton["runtime_joint_order"])
    if (
        layout.get("base_link") != skeleton["root_joint_id"]
        or layout.get("runtime_joint_type") != "spherical"
        or layout.get("runtime_joint_position_count") != expected_count
        or layout.get("runtime_joint_position_encoding") != "xyzw"
        or layout.get("render_mode") != "skin"
    ):
        raise HabitatCaptureError(
            "habitat_joint_mapping layout is not the M2 skin layout"
        )
    return _rigid_matrix(
        mapping.get("actor_from_skin_root"), owner="actor_from_skin_root"
    )


def _resolve_ao_config(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    visual_path: Path,
    urdf_path: Path,
) -> int:
    if config.get("render_mode") != "skin":
        raise HabitatCaptureError("habitat_ao_config render_mode must be skin")
    user_defined = config.get("user_defined")
    if (
        not isinstance(user_defined, Mapping)
        or user_defined.get(AVENGINE_NATIVE_GLTF_SKIN_FRAME_KEY) is not True
    ):
        raise HabitatCaptureError(
            "habitat_ao_config must explicitly opt in to the native glTF skin frame"
        )
    render_asset = config.get("render_asset")
    urdf_filepath = config.get("urdf_filepath")
    if not isinstance(render_asset, str) or not isinstance(urdf_filepath, str):
        raise HabitatCaptureError("habitat_ao_config must name render_asset and URDF")
    if (config_path.parent / render_asset).resolve() != visual_path:
        raise HabitatCaptureError(
            "habitat_ao_config render_asset differs from visual role"
        )
    if (config_path.parent / urdf_filepath).resolve() != urdf_path:
        raise HabitatCaptureError(
            "habitat_ao_config URDF differs from habitat_urdf role"
        )
    semantic_id = config.get("semantic_id")
    if (
        isinstance(semantic_id, bool)
        or not isinstance(semantic_id, int)
        or semantic_id < 0
    ):
        raise HabitatCaptureError("habitat_ao_config semantic_id must be non-negative")
    return semantic_id


def load_runtime_asset_bundle(inputs: ValidatedM2Inputs) -> RuntimeAssetBundle:
    """Load the exact role-bound mapping, AO config, and baked action sets."""

    paths = resolve_runtime_asset_paths(inputs)
    records = {
        record["role"]: record
        for record in inputs.asset["files"]
        if isinstance(record, Mapping) and record.get("role") in FORMAL_RUNTIME_ROLES
    }
    visual_sha256 = str(records["visual"]["sha256"])
    mapping = load_json(paths["habitat_joint_mapping"])
    actor_from_skin_root = _validate_joint_mapping(
        mapping, asset=inputs.asset, visual_sha256=visual_sha256
    )
    config = load_json(paths["habitat_ao_config"])
    semantic_id = _resolve_ao_config(
        config,
        config_path=paths["habitat_ao_config"],
        visual_path=paths["visual"],
        urdf_path=paths["habitat_urdf"],
    )

    action_roles_by_id = {
        str(action["action_id"]): str(action["poses_file_role"])
        for action in inputs.asset["actions"]
    }
    if action_roles_by_id != {"idle": "idle_poses", "walk": "walk_poses"}:
        raise HabitatCaptureError("asset action-to-role mapping is not canonical")
    action_sets = {
        role: read_baked_actions_npz(paths[role])
        for role in ("idle_poses", "walk_poses")
    }
    runtime_order = tuple(inputs.request["runtime_joint_order"])
    action_records = {
        str(action["action_id"]): action for action in inputs.asset["actions"]
    }
    for action_id, role in action_roles_by_id.items():
        action_set = action_sets[role]
        if action_set.source_glb_sha256 != visual_sha256:
            raise HabitatCaptureError(f"{role} source GLB hash differs from visual")
        if action_set.runtime_joint_order != runtime_order:
            raise HabitatCaptureError(
                f"{role} runtime_joint_order differs from request"
            )
        clip = action_set.action(action_id)
        if clip.sample_count != action_records[action_id]["sample_count"]:
            raise HabitatCaptureError(
                f"{action_id} baked sample count differs from asset manifest"
            )
    return RuntimeAssetBundle(
        paths_by_role=paths,
        records_by_role=records,
        joint_mapping=mapping,
        actor_from_skin_root=_matrix_tuple(actor_from_skin_root),
        action_sets_by_role=action_sets,
        action_roles_by_id=action_roles_by_id,
        semantic_id=semantic_id,
    )


def _pose_array(state: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [joint["rotation_xyzw"] for joint in state["joint_states"]],
        dtype=np.float64,
    )


def compile_frame_applications(
    inputs: ValidatedM2Inputs,
    bundle: RuntimeAssetBundle,
) -> tuple[FrameApplication, ...]:
    """Resolve all declared frames to exact loop samples before Simulator start."""

    actor_from_skin_root = _rigid_matrix(
        bundle.actor_from_skin_root, owner="actor_from_skin_root"
    )
    asset_manifest_sha256 = sha256_file(inputs.asset_path)
    runtime_order = tuple(inputs.request["runtime_joint_order"])
    frames: list[FrameApplication] = []
    for ordinal, state in enumerate(inputs.request["states"]):
        if state["frame_index"] != ordinal:
            raise HabitatCaptureError("capture states are not sequential")
        action_id = str(state["action_id"])
        try:
            role = bundle.action_roles_by_id[action_id]
            clip = bundle.action_sets_by_role[role].action(action_id)
        except (KeyError, ValueError) as exc:
            raise HabitatCaptureError(
                f"frame {ordinal} has no role-bound action"
            ) from exc
        action_time_ticks = int(state["action_time_ticks"])
        effective_tick = action_time_ticks % clip.loop_duration_ticks
        try:
            sample_index = clip.sample_ticks.index(effective_tick)
        except ValueError as exc:
            raise HabitatCaptureError(
                f"frame {ordinal} action tick is not on the baked sample grid"
            ) from exc
        baked_pose = np.asarray(clip.rotations_xyzw[sample_index], dtype=np.float64)
        declared_pose = _pose_array(state)
        expected_shape = (len(runtime_order), 4)
        if baked_pose.shape != expected_shape or declared_pose.shape != expected_shape:
            raise HabitatCaptureError(
                f"frame {ordinal} pose shape differs from runtime order"
            )
        if not np.allclose(baked_pose, declared_pose, rtol=0.0, atol=1.0e-9):
            maximum_error = float(np.max(np.abs(baked_pose - declared_pose)))
            raise HabitatCaptureError(
                f"frame {ordinal} declared pose differs from {role} sample "
                f"(maximum error {maximum_error:.9g})"
            )
        recomputed_pose_hash = compute_pose_hash(inputs.asset, state)
        recomputed_applied_hash = compute_applied_state_hash(
            inputs.asset,
            state,
            asset_manifest_sha256=asset_manifest_sha256,
        )
        if state["pose_hash"] != recomputed_pose_hash:
            raise HabitatCaptureError(
                f"frame {ordinal} pose_hash changed after validation"
            )
        if state["applied_state_hash"] != recomputed_applied_hash:
            raise HabitatCaptureError(
                f"frame {ordinal} applied_state_hash changed after validation"
            )
        world_from_actor = transform_to_matrix(state["root_transform"])
        world_from_skin_root = world_from_actor @ actor_from_skin_root
        frames.append(
            FrameApplication(
                frame_index=ordinal,
                pts_ticks=int(state["pts_ticks"]),
                action_id=action_id,
                action_time_ticks=action_time_ticks,
                effective_action_tick=effective_tick,
                action_sample_index=sample_index,
                world_from_actor=_matrix_tuple(world_from_actor),
                world_from_skin_root=_matrix_tuple(world_from_skin_root),
                joint_rotations_xyzw=tuple(
                    tuple(float(component) for component in quaternion)
                    for quaternion in baked_pose
                ),
                declared_pose_hash=str(state["pose_hash"]),
                recomputed_pose_hash=recomputed_pose_hash,
                declared_applied_state_hash=str(state["applied_state_hash"]),
                recomputed_applied_state_hash=recomputed_applied_hash,
            )
        )
    if len(frames) != 75:
        raise HabitatCaptureError(
            "formal M2 capture plan must contain exactly 75 frames"
        )
    return tuple(frames)


def _runtime_snapshot(simulator: Any, articulated_object: Any) -> dict[str, Any]:
    root = np.asarray(
        articulated_object.root_scene_node.absolute_transformation(), dtype=np.float64
    )
    joints = np.asarray(articulated_object.joint_positions, dtype=np.float64).reshape(
        -1
    )
    if root.shape != (4, 4) or not np.all(np.isfinite(root)):
        raise HabitatCaptureError("Habitat root readback is not a finite 4x4 matrix")
    if not np.all(np.isfinite(joints)):
        raise HabitatCaptureError("Habitat joint-position readback is not finite")
    core = {
        "schema": READBACK_HASH_SCHEMA,
        "world_time_seconds": float(simulator.get_world_time()),
        "world_from_skin_root": root.tolist(),
        "joint_positions_xyzw": joints.tolist(),
    }
    return {**core, "sha256": canonical_json_sha256(core)}


def _quaternion_block_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape or actual.ndim != 1 or actual.size % 4:
        return math.inf
    maximum = 0.0
    for offset in range(0, actual.size, 4):
        left = actual[offset : offset + 4]
        right = expected[offset : offset + 4]
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return math.inf
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm <= 1.0e-15 or right_norm <= 1.0e-15:
            return math.inf
        left = left / left_norm
        right = right / right_norm
        maximum = max(
            maximum,
            min(
                float(np.max(np.abs(left - right))),
                float(np.max(np.abs(left + right))),
            ),
        )
    return maximum


def _validate_observation_arrays(
    observation: Mapping[str, Any],
    modality_to_uuid: Mapping[str, str],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for modality in FORMAL_MODALITIES:
        uuid = modality_to_uuid[modality]
        if uuid not in observation:
            raise HabitatCaptureError(f"Habitat observation lacks {modality} sensor")
        arrays[modality] = np.ascontiguousarray(np.asarray(observation[uuid])).copy()
    rgb = arrays["rgb"]
    depth = arrays["depth"]
    semantic = arrays["semantic"]
    if rgb.ndim != 3 or rgb.shape[-1] not in {3, 4}:
        raise HabitatCaptureError(f"RGB frame must be HxWx3/4, got {rgb.shape}")
    if depth.ndim != 2 or semantic.ndim != 2:
        raise HabitatCaptureError("depth and semantic frames must be HxW")
    if rgb.shape[:2] != depth.shape or depth.shape != semantic.shape:
        raise HabitatCaptureError(
            "formal RGB/depth/semantic frames are not co-registered"
        )
    return arrays


def apply_and_capture_fixed_frame(
    *,
    simulator: Any,
    articulated_object: Any,
    frame: FrameApplication,
    joint_binding: HabitatJointBinding,
    modality_to_uuid: Mapping[str, str],
    sensor_wrappers: Sequence[Any],
    apply_root_transform: Callable[[Any, np.ndarray], None],
    required_semantic_id: int | None = None,
) -> CapturedFrame:
    """Apply, read back, capture once, and prove one fixed formal frame."""

    if list(modality_to_uuid) != FORMAL_MODALITIES:
        raise HabitatCaptureError(
            "formal modality mapping must be ordered rgb/depth/semantic"
        )
    if len(sensor_wrappers) != 3:
        raise HabitatCaptureError(
            "formal capture requires exactly three sensor wrappers"
        )
    expected_root = np.asarray(frame.world_from_skin_root, dtype=np.float64)
    expected_joints = np.asarray(
        joint_binding.map_pose(frame.joint_rotations_xyzw), dtype=np.float64
    )
    apply_root_transform(articulated_object, expected_root)
    articulated_object.joint_positions = expected_joints.copy()

    before = _runtime_snapshot(simulator, articulated_object)
    before_root = np.asarray(before["world_from_skin_root"], dtype=np.float64)
    before_joints = np.asarray(before["joint_positions_xyzw"], dtype=np.float64)
    root_error = float(np.max(np.abs(before_root - expected_root)))
    joint_error = _quaternion_block_error(before_joints, expected_joints)
    if root_error > _ROOT_READBACK_ATOL:
        raise HabitatCaptureError(
            f"frame {frame.frame_index} root readback error {root_error:.9g}"
        )
    if joint_error > _JOINT_READBACK_ATOL:
        raise HabitatCaptureError(
            f"frame {frame.frame_index} joint readback error {joint_error:.9g}"
        )

    # This is the only observation call for the frame.  No physics or animation
    # clock method is invoked anywhere in the formal loop.
    observation = simulator.render_sensors(list(sensor_wrappers))
    arrays = _validate_observation_arrays(observation, modality_to_uuid)
    semantic_pixel_count = (
        int(np.count_nonzero(arrays["semantic"] == required_semantic_id))
        if required_semantic_id is not None
        else None
    )
    if required_semantic_id is not None and semantic_pixel_count == 0:
        raise HabitatCaptureError(
            f"frame {frame.frame_index} has no pixels for animal semantic ID "
            f"{required_semantic_id}"
        )
    after = _runtime_snapshot(simulator, articulated_object)
    if before["world_time_seconds"] != after["world_time_seconds"]:
        raise HabitatCaptureError(
            f"frame {frame.frame_index} advanced Habitat world time"
        )
    if before["sha256"] != after["sha256"]:
        raise HabitatCaptureError(
            f"frame {frame.frame_index} changed root or joint state"
        )

    modality_records = {
        modality: {
            "sensor_uuid": modality_to_uuid[modality],
            "dtype": arrays[modality].dtype.str,
            "shape": list(arrays[modality].shape),
            "payload_sha256": array_sha256(
                modality_to_uuid[modality], arrays[modality]
            ),
        }
        for modality in FORMAL_MODALITIES
    }
    record = {
        "frame_index": frame.frame_index,
        "pts_ticks": frame.pts_ticks,
        "action_id": frame.action_id,
        "action_time_ticks": frame.action_time_ticks,
        "effective_action_tick": frame.effective_action_tick,
        "action_sample_index": frame.action_sample_index,
        "world_from_actor": [list(row) for row in frame.world_from_actor],
        "world_from_skin_root": [list(row) for row in frame.world_from_skin_root],
        "hashes": {
            "declared_pose_hash": frame.declared_pose_hash,
            "recomputed_pose_hash": frame.recomputed_pose_hash,
            "declared_applied_state_hash": frame.declared_applied_state_hash,
            "recomputed_applied_state_hash": frame.recomputed_applied_state_hash,
        },
        "runtime_application": {
            "expected_joint_positions_sha256": canonical_json_sha256(
                {
                    "runtime_joint_order": list(joint_binding.runtime_joint_order),
                    "joint_positions_xyzw": expected_joints.tolist(),
                }
            ),
            "maximum_root_readback_error": root_error,
            "maximum_joint_quaternion_readback_error": joint_error,
            "before": before,
            "after": after,
            "world_time_advance_seconds": float(
                after["world_time_seconds"] - before["world_time_seconds"]
            ),
        },
        "modalities": modality_records,
        "animal_semantic_visibility": {
            "semantic_id": required_semantic_id,
            "pixel_count": semantic_pixel_count,
            "visible": semantic_pixel_count is None or semantic_pixel_count > 0,
        },
    }
    return CapturedFrame(record=record, arrays=arrays)


def save_capture_arrays(
    captures: Sequence[CapturedFrame],
    output_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Save three readable frame stacks and verify every payload on readback."""

    if not captures:
        raise HabitatCaptureError("cannot save an empty formal capture")
    output = Path(output_dir).resolve()
    array_dir = output / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for modality in FORMAL_MODALITIES:
        arrays = [
            np.ascontiguousarray(capture.arrays[modality]) for capture in captures
        ]
        first = arrays[0]
        if any(
            array.dtype != first.dtype or array.shape != first.shape for array in arrays
        ):
            raise HabitatCaptureError(f"{modality} frames changed dtype or shape")
        stack = np.ascontiguousarray(np.stack(arrays, axis=0))
        path = array_dir / f"{modality}.npy"
        np.save(path, stack, allow_pickle=False)
        reloaded = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_hashes = [
            str(capture.record["modalities"][modality]["payload_sha256"])
            for capture in captures
        ]
        sensor_uuid = str(captures[0].record["modalities"][modality]["sensor_uuid"])
        readback_hashes = [
            array_sha256(sensor_uuid, np.asarray(reloaded[index]))
            for index in range(len(captures))
        ]
        if readback_hashes != expected_hashes:
            raise HabitatCaptureError(
                f"{modality} array artifact failed payload readback"
            )
        artifacts[modality] = {
            "sensor_uuid": sensor_uuid,
            "dtype": stack.dtype.str,
            "shape": list(stack.shape),
            "frame_payload_sha256": expected_hashes,
            "artifact": file_record(path, relative_to=output),
            "readback_verified": True,
        }
    return artifacts


def verify_saved_capture_arrays(
    evidence: Mapping[str, Any], output_dir: str | Path
) -> list[str]:
    """Rehash saved stacks and their declared per-frame payloads."""

    output = Path(output_dir).resolve()
    errors: list[str] = []
    artifacts = evidence.get("array_artifacts")
    if not isinstance(artifacts, Mapping):
        return ["evidence lacks array_artifacts"]
    for modality in FORMAL_MODALITIES:
        record = artifacts.get(modality)
        if not isinstance(record, Mapping):
            errors.append(f"array_artifacts lacks {modality}")
            continue
        artifact = record.get("artifact")
        if not isinstance(artifact, Mapping) or not isinstance(
            artifact.get("path"), str
        ):
            errors.append(f"{modality} artifact record is invalid")
            continue
        path = (output / str(artifact["path"])).resolve()
        try:
            path.relative_to(output)
        except ValueError:
            errors.append(f"{modality} artifact escapes output directory")
            continue
        if not path.is_file():
            errors.append(f"{modality} artifact is missing")
            continue
        if path.stat().st_size != artifact.get("byte_size") or sha256_file(
            path
        ) != artifact.get("sha256"):
            errors.append(f"{modality} artifact bytes changed")
            continue
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            errors.append(f"{modality} artifact is not a readable NPY: {exc}")
            continue
        if list(array.shape) != record.get("shape") or array.dtype.str != record.get(
            "dtype"
        ):
            errors.append(f"{modality} artifact dtype/shape changed")
            continue
        expected = record.get("frame_payload_sha256")
        if not isinstance(expected, list) or len(expected) != len(array):
            errors.append(f"{modality} frame hash list is invalid")
            continue
        actual = [
            array_sha256(str(record["sensor_uuid"]), np.asarray(array[index]))
            for index in range(len(array))
        ]
        if actual != expected:
            errors.append(f"{modality} frame payload hash mismatch")
    return errors


def _semantic_color(semantic_id: int) -> tuple[int, int, int]:
    if semantic_id == 0:
        return (0, 0, 0)
    digest = hashlib.sha256(str(semantic_id).encode("ascii")).digest()
    return tuple(64 + channel % 192 for channel in digest[:3])  # type: ignore[return-value]


def _encode_review_video(frame_dir: Path, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        "15",
        "-start_number",
        "0",
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        "-threads",
        "1",
        str(destination),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise HabitatCaptureError(
            f"ffmpeg review encoding failed: {completed.stderr.strip()}"
        )


def write_research_review_media(
    captures: Sequence[CapturedFrame],
    output_dir: str | Path,
    *,
    encode_video: Callable[[Path, Path], None] = _encode_review_video,
) -> dict[str, Any]:
    """Encode same-view RGB/depth/semantic review videos.

    This helper never labels the videos as formal artifacts.  The readable NPY
    stacks remain the authoritative payloads; PNG frames are temporary encoder
    inputs and are removed after all three videos are hash-bound.
    """

    if not captures:
        raise HabitatCaptureError("cannot encode review media for an empty capture")
    output = Path(output_dir).resolve()
    media_dir = output / "review_media"
    frame_root = media_dir / "frames"
    for modality in FORMAL_MODALITIES:
        (frame_root / modality).mkdir(parents=True, exist_ok=True)

    positive_depth = [
        np.asarray(capture.arrays["depth"])[
            np.isfinite(capture.arrays["depth"])
            & (np.asarray(capture.arrays["depth"]) > 0)
        ]
        for capture in captures
    ]
    nonempty_depth = [value for value in positive_depth if value.size]
    depth_high = (
        max(float(np.percentile(np.concatenate(nonempty_depth), 99.0)), 1.0e-6)
        if nonempty_depth
        else 1.0
    )
    try:
        for frame_index, capture in enumerate(captures):
            rgb = np.asarray(capture.arrays["rgb"])
            Image.fromarray(rgb[..., :3].astype(np.uint8), mode="RGB").save(
                frame_root / "rgb" / f"frame_{frame_index:04d}.png"
            )

            depth = np.asarray(capture.arrays["depth"], dtype=np.float64)
            finite_positive = np.isfinite(depth) & (depth > 0.0)
            depth_preview = np.zeros((*depth.shape, 3), dtype=np.uint8)
            grayscale = np.zeros(depth.shape, dtype=np.uint8)
            grayscale[finite_positive] = np.asarray(
                (1.0 - np.clip(depth[finite_positive] / depth_high, 0.0, 1.0)) * 255.0,
                dtype=np.uint8,
            )
            depth_preview[finite_positive] = grayscale[finite_positive, None]
            Image.fromarray(depth_preview, mode="RGB").save(
                frame_root / "depth" / f"frame_{frame_index:04d}.png"
            )

            semantic = np.asarray(capture.arrays["semantic"])
            semantic_preview = np.zeros((*semantic.shape, 3), dtype=np.uint8)
            for semantic_id in np.unique(semantic):
                semantic_preview[semantic == semantic_id] = _semantic_color(
                    int(semantic_id)
                )
            Image.fromarray(semantic_preview, mode="RGB").save(
                frame_root / "semantic" / f"frame_{frame_index:04d}.png"
            )

        videos: dict[str, Any] = {}
        for modality in FORMAL_MODALITIES:
            destination = media_dir / f"view0_{modality}_review.mp4"
            encode_video(frame_root / modality, destination)
            if not destination.is_file():
                raise HabitatCaptureError(
                    f"review video encoder did not create {destination.name}"
                )
            videos[modality] = {
                "view_id": "view0",
                "review_only": True,
                "qualification_claim": False,
                "frame_count": len(captures),
                "frame_rate_hz": 15,
                "artifact": file_record(destination, relative_to=output),
            }
    finally:
        if frame_root.is_dir():
            shutil.rmtree(frame_root)
    return {
        "review_only": True,
        "qualification_claim": False,
        "view_ids": ["view0"],
        "formal_view_ids": [],
        "videos": videos,
    }


def _apply_root_with_habitat(
    articulated_object: Any,
    world_from_skin_root: np.ndarray,
    *,
    qt: Any,
    mn: Any,
) -> None:
    rotation = world_from_skin_root[:3, :3]
    quaternion_wxyz = qt.as_float_array(qt.from_rotation_matrix(rotation))
    articulated_object.translation = mn.Vector3(world_from_skin_root[:3, 3])
    articulated_object.rotation = mn.Quaternion(
        mn.Vector3(quaternion_wxyz[1:]), float(quaternion_wxyz[0])
    )


def _instantiate_articulated_object(
    simulator: Any,
    *,
    bundle: RuntimeAssetBundle,
    habitat_sim: Any,
) -> tuple[Any, HabitatJointBinding, list[int]]:
    config_path = bundle.paths_by_role["habitat_ao_config"]
    manager = simulator.metadata_mediator.ao_template_manager
    loaded_ids = manager.load_configs(str(config_path))
    handle_prefix = config_path.stem.removesuffix(".ao_config")
    handles = manager.get_template_handles(handle_prefix)
    if len(loaded_ids) != 1 or len(handles) != 1:
        raise HabitatCaptureError(
            f"expected one formal AO template, got ids={loaded_ids}, handles={handles}"
        )
    articulated_object = simulator.get_articulated_object_manager().add_articulated_object_by_template_handle(
        handles[0]
    )
    if articulated_object is None:
        raise HabitatCaptureError(
            "Habitat failed to instantiate the M2 articulated object"
        )
    articulated_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    link_ids = list(articulated_object.get_link_ids())
    expected_names = set(bundle.joint_mapping["joint_order"])
    actual_names = {articulated_object.get_link_name(-1)} | {
        articulated_object.get_link_name(link_id) for link_id in link_ids
    }
    if actual_names != expected_names:
        raise HabitatCaptureError(
            "Habitat AO link names differ from mapping: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    blocks = [
        HabitatLinkJointBlock(
            link_name=articulated_object.get_link_name(link_id),
            joint_position_offset=int(
                articulated_object.get_link_joint_pos_offset(link_id)
            ),
            joint_position_count=int(
                articulated_object.get_link_num_joint_pos(link_id)
            ),
        )
        for link_id in link_ids
    ]
    binding = bind_habitat_link_layout(
        bundle.joint_mapping["runtime_joint_order"],
        blocks,
        joint_position_count=len(articulated_object.joint_positions),
    )
    return articulated_object, binding, link_ids


def _prepare_output_directory(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise HabitatCaptureError(f"capture output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _installed_research_receipt(
    *,
    inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
    installed_runtime: Any,
    array_artifacts: Mapping[str, Any],
    review_media: Mapping[str, Any],
    initial_world_time: float,
    final_world_time: float,
    frame_count: int,
) -> dict[str, Any]:
    """Describe one current installed-prefix research run without v1 evidence.

    The old M2 evidence reader expects a schema, a content hash, a Git runtime
    identity, and formal/review v1 fields.  This receipt deliberately contains
    none of those bindings: it is an ordinary research-only result locator.
    Array write/readback and review-video production happen before this helper,
    but their hashes remain implementation-local rather than a new M2 contract.
    """

    array_paths: dict[str, str] = {}
    review_paths: dict[str, str] = {}
    videos = review_media.get("videos")
    if not isinstance(videos, Mapping):
        raise HabitatCaptureError("installed research review lacks video records")
    for modality in FORMAL_MODALITIES:
        array_record = array_artifacts.get(modality)
        video_record = videos.get(modality)
        if not isinstance(array_record, Mapping) or not isinstance(
            video_record, Mapping
        ):
            raise HabitatCaptureError(
                f"installed research review lacks {modality} artifacts"
            )
        array_artifact = array_record.get("artifact")
        video_artifact = video_record.get("artifact")
        if not isinstance(array_artifact, Mapping) or not isinstance(
            array_artifact.get("path"), str
        ):
            raise HabitatCaptureError(
                f"installed research review has invalid {modality} array path"
            )
        if not isinstance(video_artifact, Mapping) or not isinstance(
            video_artifact.get("path"), str
        ):
            raise HabitatCaptureError(
                f"installed research review has invalid {modality} video path"
            )
        array_paths[modality] = str(array_artifact["path"])
        review_paths[modality] = str(video_artifact["path"])

    habitat_module = getattr(installed_runtime.habitat_sim, "__file__", None)
    if not isinstance(habitat_module, str) or not habitat_module:
        raise HabitatCaptureError("installed Habitat runtime has no module path")
    return {
        "status": "research_only",
        "research_only": True,
        "qualification_claim": False,
        "formal_admission": False,
        "runtime": {
            "mode": "non_git_installed_prefix",
            "prefix": str(installed_runtime.prefix),
            "habitat_sim_module": str(Path(habitat_module).resolve()),
            "magnum_python_site": str(installed_runtime.magnum_python_site),
            "physics_config_path": str(installed_runtime.physics_config_path),
        },
        "inputs": {
            "animal_asset_package": str(inputs.asset_path),
            "m2_capture_request": str(inputs.request_path),
            "m1_room_manifest": str(room_inputs.room_path),
            "m1_camera_request": str(room_inputs.request_path),
        },
        "room": {
            "room_id": room_inputs.room["room_id"],
            "room_kind": room_inputs.room["room_kind"],
        },
        "capture": {
            "frame_count": frame_count,
            "review_view_ids": ["view0"],
            "modalities": list(FORMAL_MODALITIES),
            "state_evaluation": "explicit_fixed_state",
            "physics_steps": 0,
            "world_time_seconds": [initial_world_time, final_world_time],
        },
        "artifacts": {
            "arrays": array_paths,
            "review_media": review_paths,
        },
    }


def _capture_m2_states(
    inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
    output_dir: str | Path,
    *,
    runtime_root: str | Path | None = None,
    review_only: bool,
    installed_runtime: Any | None = None,
) -> dict[str, Any]:
    """Execute the shared 75-state runtime after admission-specific loading."""

    bundle = load_runtime_asset_bundle(inputs)
    frames = compile_frame_applications(inputs, bundle)
    output = _prepare_output_directory(output_dir)

    # M1 owns room resolution and the one formal camera calibration.  Import
    # lazily so unit tests for the fixed-state core never import Habitat.
    from avengine.rooms.contracts import (
        validate_loaded_scene_asset_graph,
        validate_scene_asset_graph,
    )
    from avengine.rooms.habitat_capture import (
        _make_configuration,
        _resolved_assets,
        discover_runtime_root,
    )

    if installed_runtime is None:
        runtime = discover_runtime_root(runtime_root)
        mp3d_root = None
        physics_config_path = None
    else:
        if runtime_root is not None:
            raise HabitatCaptureError(
                "installed-prefix research capture does not accept runtime_root"
            )
        if not review_only:
            raise HabitatCaptureError(
                "installed-prefix runtime is limited to research review"
            )
        runtime = None
        mp3d_root = installed_runtime.mp3d_root
        physics_config_path = installed_runtime.physics_config_path
    room_assets = _resolved_assets(room_inputs, runtime, mp3d_root=mp3d_root)
    missing_room_assets = [record for record in room_assets if not record["exists"]]
    if missing_room_assets:
        raise HabitatCaptureError("validated M1 room has missing runtime assets")
    visual_path = bundle.paths_by_role["visual"].resolve()
    room_asset_paths = {
        Path(record["resolved_path"]).resolve()
        for record in room_assets
        if isinstance(record, Mapping) and isinstance(record.get("resolved_path"), str)
    }
    if visual_path in room_asset_paths:
        raise HabitatCaptureError(
            "M2 visual asset path must be unique within the fresh Simulator cache"
        )
    static_scene_errors = validate_scene_asset_graph(
        room_inputs, runtime, mp3d_root=mp3d_root
    )
    if static_scene_errors:
        raise HabitatCaptureError(
            "M1 room graph failed before Simulator: " + "; ".join(static_scene_errors)
        )

    if installed_runtime is None:
        # The pinned build must import numpy-quaternion before habitat_sim.
        import quaternion as qt

        import habitat_sim
        import magnum as mn
        from habitat_sim._ext import habitat_sim_bindings

        runtime_identity = _runtime_identity(
            runtime=runtime,
            habitat_sim=habitat_sim,
            habitat_sim_bindings=habitat_sim_bindings,
        )
    else:
        qt = installed_runtime.quaternion
        habitat_sim = installed_runtime.habitat_sim
        mn = installed_runtime.magnum
        runtime_identity = None
    if not review_only:
        formal_identity_errors = _formal_runtime_identity_errors(runtime_identity)
        if formal_identity_errors:
            raise HabitatCaptureError(
                "formal runtime identity is not admissible: "
                + "; ".join(formal_identity_errors)
            )

    configuration, modality_to_uuid, _listener_uuid, resolved_scene = (
        _make_configuration(
            room_inputs,
            runtime,
            output,
            mp3d_root=mp3d_root,
            include_audio_sensor=installed_runtime is None,
            physics_config_path=physics_config_path,
        )
    )
    if list(modality_to_uuid) != FORMAL_MODALITIES:
        raise HabitatCaptureError("M1 configuration changed formal modality ordering")
    captures: list[CapturedFrame] = []
    loaded_graph: Mapping[str, Any] | None = None
    runtime_binding: Mapping[str, Any] | None = None
    with habitat_sim.Simulator(configuration) as simulator:
        navmesh_path = resolved_scene.get("navmesh")
        navmesh_loaded = False
        if navmesh_path is not None and Path(navmesh_path).is_file():
            navmesh_loaded = bool(simulator.pathfinder.load_nav_mesh(str(navmesh_path)))
        loaded_errors, loaded_graph_value = validate_loaded_scene_asset_graph(
            room_inputs,
            runtime,
            simulator,
            declared_navmesh_loaded=navmesh_loaded,
            mp3d_root=mp3d_root,
        )
        if loaded_errors:
            raise HabitatCaptureError(
                "fresh Simulator room graph differs from M1 declaration: "
                + "; ".join(loaded_errors)
            )
        loaded_graph = loaded_graph_value
        simulator.seed(int(inputs.request["seed"]))

        rig = room_inputs.request["primary_camera_rig"]
        camera_state = habitat_sim.AgentState()
        camera_state.position = np.asarray(
            rig["world_from_rig"]["translation_m"], dtype=np.float64
        )
        x, y, z, w = normalized_quaternion_xyzw(rig["world_from_rig"]["rotation_xyzw"])
        camera_state.rotation = qt.quaternion(w, x, y, z)
        simulator.initialize_agent(0, camera_state)

        articulated_object, joint_binding, link_ids = _instantiate_articulated_object(
            simulator, bundle=bundle, habitat_sim=habitat_sim
        )
        runtime_binding = {
            **joint_binding.to_json_data(),
            "base_link_id": -1,
            "base_link_name": articulated_object.get_link_name(-1),
            "link_ids": [int(link_id) for link_id in link_ids],
            "motion_type": "KINEMATIC",
        }
        sensor_wrappers = [
            simulator.sensors[modality_to_uuid[modality]]
            for modality in FORMAL_MODALITIES
        ]

        def apply_root(ao: Any, matrix: np.ndarray) -> None:
            _apply_root_with_habitat(ao, matrix, qt=qt, mn=mn)

        initial_world_time = float(simulator.get_world_time())
        for frame in frames:
            captures.append(
                apply_and_capture_fixed_frame(
                    simulator=simulator,
                    articulated_object=articulated_object,
                    frame=frame,
                    joint_binding=joint_binding,
                    modality_to_uuid=modality_to_uuid,
                    sensor_wrappers=sensor_wrappers,
                    apply_root_transform=apply_root,
                    required_semantic_id=bundle.semantic_id,
                )
            )
        final_world_time = float(simulator.get_world_time())
        if final_world_time != initial_world_time:
            raise HabitatCaptureError("75-state fixed loop advanced Habitat world time")

    array_artifacts = save_capture_arrays(captures, output)
    review_media = (
        write_research_review_media(captures, output) if review_only else None
    )
    if installed_runtime is not None:
        if review_media is None:
            raise HabitatCaptureError(
                "installed-prefix research capture lacks review media"
            )
        receipt = _installed_research_receipt(
            inputs=inputs,
            room_inputs=room_inputs,
            installed_runtime=installed_runtime,
            array_artifacts=array_artifacts,
            review_media=review_media,
            initial_world_time=initial_world_time,
            final_world_time=final_world_time,
            frame_count=len(captures),
        )
        write_json(output / "research_receipt.json", receipt)
        return receipt

    role_evidence = {
        role: {
            "path": str(bundle.paths_by_role[role]),
            "byte_size": int(bundle.records_by_role[role]["byte_size"]),
            "sha256": str(bundle.records_by_role[role]["sha256"]),
        }
        for role in FORMAL_RUNTIME_ROLES
    }
    qa_ids = [
        view["qa_id"]
        for view in room_inputs.request.get("qa_views", [])
        if isinstance(view, Mapping) and isinstance(view.get("qa_id"), str)
    ]
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "status": "review_only" if review_only else "pass",
        "evidence_kind": (
            "research_candidate_habitat_review"
            if review_only
            else "completed_formal_habitat_capture"
        ),
        "review_only": review_only,
        "request_id": inputs.request["request_id"],
        "asset_id": inputs.asset["asset_id"],
        "asset_admission_state": inputs.asset["admission_state"],
        "room_id": room_inputs.room["room_id"],
        "formal_view_ids": [] if review_only else ["view0"],
        "review_view_ids": ["view0"] if review_only else [],
        "formal_modalities": [] if review_only else list(FORMAL_MODALITIES),
        "review_modalities": list(FORMAL_MODALITIES) if review_only else [],
        "qa_policy": {
            "declared_m1_qa_ids": qa_ids,
            "qa_media_generated": False,
            "research_review_media_generated": review_only,
            "qa_ids_are_not_formal_view_ids": True,
            "review_view0_is_not_a_formal_capture": review_only,
        },
        "inputs": {
            "animal_asset_package": {
                "path": str(inputs.asset_path),
                "sha256": sha256_file(inputs.asset_path),
            },
            "m2_capture_request": {
                "path": str(inputs.request_path),
                "sha256": sha256_file(inputs.request_path),
            },
            "m1_room_manifest": {
                "path": str(room_inputs.room_path),
                "sha256": sha256_file(room_inputs.room_path),
            },
            "m1_camera_request": {
                "path": str(room_inputs.request_path),
                "sha256": sha256_file(room_inputs.request_path),
            },
        },
        "runtime_assets": role_evidence,
        "runtime_identity": runtime_identity,
        "room_assets": room_assets,
        "loaded_room_graph": loaded_graph,
        "sensor_contract": {
            "rig_id": room_inputs.request["primary_camera_rig"]["rig_id"],
            "view_id": room_inputs.request["primary_camera_rig"]["view_id"],
            "world_from_rig": room_inputs.request["primary_camera_rig"][
                "world_from_rig"
            ],
            "shared_calibration": room_inputs.request["primary_camera_rig"][
                "shared_calibration"
            ],
            "modality_to_sensor_uuid": modality_to_uuid,
        },
        "runtime_application": {
            "simulator_lifetime": "fresh_per_capture_call",
            "motion_type": "KINEMATIC",
            "root_formula": (
                "world_from_skin_root = world_from_actor @ actor_from_skin_root"
            ),
            "joint_binding": runtime_binding,
            "state_evaluation": "explicit_fixed_state",
            "observation_calls_per_frame": 1,
            "physics_steps": 0,
            "initial_world_time_seconds": initial_world_time,
            "final_world_time_seconds": final_world_time,
        },
        "hash_algorithms": {
            "pose_hash": inputs.request["pose_hash_algorithm"],
            "applied_state_hash": inputs.request["applied_state_hash_algorithm"],
            "modality_payload_hash": ARRAY_HASH_ALGORITHM,
            "runtime_readback_hash": READBACK_HASH_SCHEMA,
        },
        "frames": [dict(capture.record) for capture in captures],
        "array_artifacts": array_artifacts,
    }
    if review_only:
        evidence["qualification_claim"] = False
    if review_media is not None:
        evidence["review_media"] = review_media
    verification_errors = verify_saved_capture_arrays(evidence, output)
    if verification_errors:
        raise HabitatCaptureError("; ".join(verification_errors))
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    write_json(output / "evidence.json", evidence)
    return evidence


def capture_m2_habitat(
    inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
    output_dir: str | Path,
    *,
    runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute formal capture, fail-closed on every non-canary admission."""

    inputs, room_inputs = _reload_validated_context(inputs, room_inputs)
    return _capture_m2_states(
        inputs,
        room_inputs,
        output_dir,
        runtime_root=runtime_root,
        review_only=False,
    )


def capture_m2_research_review(
    inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
    output_dir: str | Path,
    *,
    runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    """Render research-candidate review media without making a formal claim."""

    inputs, room_inputs = _reload_research_review_context(inputs, room_inputs)
    evidence = _capture_m2_states(
        inputs,
        room_inputs,
        output_dir,
        runtime_root=runtime_root,
        review_only=True,
    )
    if (
        evidence.get("status") != "review_only"
        or evidence.get("review_only") is not True
        or evidence.get("qualification_claim") is not False
        or evidence.get("formal_view_ids") != []
    ):
        raise HabitatCaptureError("research review attempted to emit a formal claim")
    return evidence


def _require_bullet_enabled_installed_runtime(installed_runtime: Any) -> None:
    """Fail before configuration when the current M2 route cannot create an AO."""

    habitat_sim = getattr(installed_runtime, "habitat_sim", None)
    if not bool(getattr(habitat_sim, "built_with_bullet", False)):
        raise HabitatCaptureError(
            "installed-prefix M2 Blender-custom research requires a Bullet-enabled "
            "Habitat runtime (habitat_sim.built_with_bullet=True)"
        )


def capture_m2_installed_research_review(
    inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
    output_dir: str | Path,
    *,
    runtime_prefix: str | Path,
    magnum_python_site: str | Path,
) -> dict[str, Any]:
    """Run the current Blender-custom research path on an installed prefix.

    This entry intentionally has no ``runtime_root`` compatibility alias.  It
    cannot emit old M2 formal v1 evidence and cannot be used for admission.
    """

    if runtime_prefix is None or magnum_python_site is None:
        raise HabitatCaptureError(
            "installed-prefix research requires explicit runtime_prefix and "
            "magnum_python_site"
        )
    inputs, room_inputs = _reload_research_review_context(inputs, room_inputs)
    if room_inputs.room.get("room_kind") != "blender_custom":
        raise HabitatCaptureError(
            "installed-prefix M2 research currently supports only blender_custom rooms"
        )
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

    try:
        installed_runtime = prepare_installed_habitat_runtime(
            runtime_prefix=runtime_prefix,
            magnum_python_site=magnum_python_site,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise HabitatCaptureError(
            f"installed Habitat runtime is unavailable: {error}"
        ) from error
    _require_bullet_enabled_installed_runtime(installed_runtime)
    return _capture_m2_states(
        inputs,
        room_inputs,
        output_dir,
        runtime_root=None,
        review_only=True,
        installed_runtime=installed_runtime,
    )


__all__ = [
    "ARRAY_HASH_ALGORITHM",
    "CapturedFrame",
    "EVIDENCE_SCHEMA",
    "FrameApplication",
    "HabitatCaptureError",
    "RuntimeAssetBundle",
    "apply_and_capture_fixed_frame",
    "capture_m2_installed_research_review",
    "capture_m2_habitat",
    "capture_m2_research_review",
    "compile_frame_applications",
    "load_research_review_inputs",
    "load_runtime_asset_bundle",
    "quaternion_xyzw_to_matrix",
    "resolve_runtime_asset_paths",
    "save_capture_arrays",
    "transform_to_matrix",
    "validate_capture_context",
    "validate_research_review_context",
    "verify_saved_capture_arrays",
    "write_research_review_media",
]

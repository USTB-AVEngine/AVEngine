"""Fixed-state Habitat capture for the M5.1 Rocketbox-human + Beagle route.

The public path entrypoint consumes two 270-point actor trajectories.  It
compiles the Rocketbox Walking clip to 15 fps, repeats the caller-provided M2
Beagle states, writes both articulated states explicitly for every frame, and
makes exactly one co-located RGB/depth/semantic observation call.  RGB and
semantic are retained; depth is observed only to preserve the M1 shared-view
contract.

The legacy-route wrapper reads the two paths verbatim from the committed M5.1
route manifest.  Neither entrypoint advances Habitat physics or uses a static
sliding human fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    resolve_declared_path,
    sha256_file,
)
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.evidence import array_sha256
from avengine.m1.habitat_capture import (
    _make_configuration,
    _resolved_assets,
    discover_runtime_root,
)
from avengine.m2.contracts import (
    FORMAL_MODALITIES,
    load_and_validate_inputs as load_m2_inputs,
)
from avengine.m2.habitat import HabitatLinkJointBlock
from avengine.m2.habitat_capture import (
    HabitatCaptureError,
    _apply_root_with_habitat,
    _quaternion_block_error,
    _runtime_snapshot as dog_runtime_snapshot,
    _validate_observation_arrays,
    compile_frame_applications,
    load_runtime_asset_bundle,
    quaternion_xyzw_to_matrix,
)
from avengine.m2.local_tr_habitat import bind_local_tr_habitat_layout
from avengine.m2.local_tr_review import (
    _runtime_snapshot as human_runtime_snapshot,
    mixed_joint_readback_errors,
)
from avengine.m5.visual import (
    _instantiate_actor_with_semantic_template,
    _link_id_by_name,
    _node_world_position,
    _set_scene_node_semantic_readback,
)
from avengine.m5_1.human_runtime import (
    HEAD_LINK_NAME,
    MOUTH_LINK_NAME,
    HumanRuntimePackage,
    prepare_rocketbox_habitat_runtime,
)
from avengine.m5_1.legacy_route import (
    FRAME_COUNT,
    FRAME_RATE_HZ,
    assert_valid_route_manifest,
)


MIXED_CAPTURE_SCHEMA = "avengine_m5_1_human_beagle_capture_v1"
HEADING_ALIGNMENT_SCHEMA = "avengine_m5_1_actor_heading_gate_v1"
HUMAN_SEMANTIC_ID = 220
BEAGLE_SEMANTIC_ID = 221
BEAGLE_MOUTH_LINK_NAME = "beagle Xtra Mouth"
TIME_BASE_HZ = 48_000
TICKS_PER_FRAME = TIME_BASE_HZ // FRAME_RATE_HZ
LEGACY_CAMERA_POSITION_M = (-0.7, 1.471, 0.65)
LEGACY_CAMERA_YAW_DEG = 55.0
LEGACY_CAMERA_HFOV_DEG = 105.0
M5_1_LIGHT_SETUP_KEY = "avengine_m5_1_room_lighting"
M5_1_ACTOR_SHADER_TYPE = "pbr"
_ROOT_READBACK_ATOL = 2.0e-6
_JOINT_READBACK_ATOL = 2.0e-6
_LINK_MATRIX_READBACK_ATOL = 2.0e-5
_HEADING_ALIGNMENT_MAX_ERROR_DEG = 1.0e-6


class MixedCaptureError(RuntimeError):
    """The mixed M5.1 fixed-state capture or readback failed."""


@dataclass(frozen=True)
class MixedCaptureResult:
    """Retained arrays and evidence for one 270-frame mixed capture."""

    output_dir: Path
    rgb: np.ndarray
    semantic: np.ndarray
    actor_world_matrices: np.ndarray
    skin_root_world_matrices: np.ndarray
    anchor_positions_m: np.ndarray
    semantic_visibility_pixels: np.ndarray
    records: tuple[Mapping[str, Any], ...]
    evidence: Mapping[str, Any]


def _points(value: Any, *, owner: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MixedCaptureError(f"{owner} must be a finite [270,3] array") from exc
    if array.shape != (FRAME_COUNT, 3) or not np.all(np.isfinite(array)):
        raise MixedCaptureError(f"{owner} must be a finite [270,3] array")
    return np.ascontiguousarray(array)


def _axis_label_vector(value: Any, *, owner: str) -> tuple[float, float, float]:
    axes = {
        "+X": (1.0, 0.0, 0.0),
        "-X": (-1.0, 0.0, 0.0),
        "+Z": (0.0, 0.0, 1.0),
        "-Z": (0.0, 0.0, -1.0),
    }
    try:
        return axes[value]
    except (KeyError, TypeError) as exc:
        raise MixedCaptureError(
            f"{owner} must be one of {sorted(axes)} in the +Y-up actor frame"
        ) from exc


def _horizontal_unit_axis(value: Any, *, owner: str) -> np.ndarray:
    try:
        axis = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MixedCaptureError(
            f"{owner} must be a finite horizontal 3-vector"
        ) from exc
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise MixedCaptureError(f"{owner} must be a finite horizontal 3-vector")
    if not math.isclose(float(axis[1]), 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise MixedCaptureError(f"{owner} must be horizontal in the +Y-up actor frame")
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise MixedCaptureError(f"{owner} must be nonzero")
    return np.ascontiguousarray(axis / norm)


def _trajectory_tangents(points_m: Any) -> np.ndarray:
    points = _points(points_m, owner="actor trajectory")
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[:, 1] = 0.0
    norms = np.linalg.norm(tangents, axis=1)
    moving_indices = np.flatnonzero(norms > 1.0e-12)
    if not len(moving_indices):
        raise MixedCaptureError("actor trajectory has no horizontal path tangent")
    tangents[moving_indices] /= norms[moving_indices, None]
    for index in np.flatnonzero(norms <= 1.0e-12):
        nearest = moving_indices[
            int(np.argmin(np.abs(moving_indices - int(index))))
        ]
        tangents[index] = tangents[nearest]
    return np.ascontiguousarray(tangents)


def trajectory_world_matrices(
    points_m: Any, *, local_forward_axis: Any
) -> np.ndarray:
    """Create actor transforms that align an asset's forward axis to its path."""

    points = _points(points_m, owner="actor trajectory")
    tangents = _trajectory_tangents(points)
    local_forward = _horizontal_unit_axis(
        local_forward_axis, owner="local anatomical forward axis"
    )
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    local_right = np.cross(local_forward, up)
    local_right /= np.linalg.norm(local_right)
    local_basis = np.stack((local_right, up, -local_forward), axis=1)
    if not math.isclose(float(np.linalg.det(local_basis)), 1.0, abs_tol=1.0e-12):
        raise MixedCaptureError("local anatomical frame is not a proper rotation")
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], FRAME_COUNT, axis=0)
    for index, forward in enumerate(tangents):
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        world_basis = np.stack((right, up, -forward), axis=1)
        rotation = world_basis @ local_basis.T
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-12):
            raise MixedCaptureError("trajectory tangent produced an improper rotation")
        matrices[index, :3, :3] = rotation
        matrices[index, :3, 3] = points[index]
    return np.ascontiguousarray(matrices)


def _actor_heading_evidence(
    *,
    actor_id: str,
    points_m: Any,
    actor_world_matrices: Any,
    local_forward_axis: Any,
    binding_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain and gate every frame's anatomical-forward/path alignment."""

    points = _points(points_m, owner=f"{actor_id} root path")
    matrices = np.asarray(actor_world_matrices, dtype=np.float64)
    if matrices.shape != (FRAME_COUNT, 4, 4) or not np.all(np.isfinite(matrices)):
        raise MixedCaptureError(
            f"{actor_id} actor_world_matrices must be finite [270,4,4]"
        )
    local_forward = _horizontal_unit_axis(
        local_forward_axis, owner=f"{actor_id} local anatomical forward axis"
    )
    tangents = _trajectory_tangents(points)
    world_forwards = np.einsum(
        "nij,j->ni", matrices[:, :3, :3], local_forward
    )
    forward_norms = np.linalg.norm(world_forwards, axis=1)
    if np.any(forward_norms <= 1.0e-12):
        raise MixedCaptureError(f"{actor_id} produced a zero world anatomical forward")
    world_forwards /= forward_norms[:, None]
    dots = np.clip(np.sum(world_forwards * tangents, axis=1), -1.0, 1.0)
    cross_norms = np.linalg.norm(np.cross(world_forwards, tangents), axis=1)
    errors_deg = np.degrees(np.arctan2(cross_norms, dots))
    passed = errors_deg <= _HEADING_ALIGNMENT_MAX_ERROR_DEG
    maximum_index = int(np.argmax(errors_deg))
    if not bool(np.all(passed)):
        raise MixedCaptureError(
            f"{actor_id} anatomical forward/path heading gate failed at frame "
            f"{maximum_index}: {float(errors_deg[maximum_index]):.9f} degrees"
        )

    frames = [
        {
            "frame_index": index,
            "path_tangent_world": [float(value) for value in tangents[index]],
            "anatomical_forward_world": [
                float(value) for value in world_forwards[index]
            ],
            "alignment_dot": float(dots[index]),
            "heading_error_degrees": float(errors_deg[index]),
            "passed": bool(passed[index]),
        }
        for index in range(FRAME_COUNT)
    ]
    return {
        "actor_id": actor_id,
        "status": "pass",
        "local_anatomical_forward_axis": [float(value) for value in local_forward],
        "binding_source": dict(binding_source),
        "gate": {
            "maximum_allowed_error_degrees": _HEADING_ALIGNMENT_MAX_ERROR_DEG,
            "maximum_observed_error_degrees": float(errors_deg[maximum_index]),
            "maximum_error_frame_index": maximum_index,
            "minimum_alignment_dot": float(np.min(dots)),
            "all_frames_passed": True,
        },
        "frames": frames,
    }


def _human_anatomical_forward_binding(
    manifest_path: str | Path,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    path = Path(manifest_path).resolve()
    manifest = load_json(path)
    if (
        manifest.get("schema") != "avengine_m5_1_rocketbox_human_runtime_v1"
        or manifest.get("status") != "pass"
    ):
        raise MixedCaptureError("human runtime manifest must be a passing M5.1 report")
    content = dict(manifest)
    declared_content_hash = content.pop("manifest_content_sha256", None)
    if declared_content_hash != canonical_json_sha256(content):
        raise MixedCaptureError("human runtime manifest content hash differs")
    anatomical_frame = manifest.get("anatomical_frame")
    if not isinstance(anatomical_frame, Mapping):
        raise MixedCaptureError("human runtime manifest lacks anatomical_frame")
    if anatomical_frame.get("actor_up_axis") != "+Y":
        raise MixedCaptureError("human anatomical frame must declare actor_up_axis +Y")
    source = anatomical_frame.get("source")
    if not isinstance(source, str) or not source.strip():
        raise MixedCaptureError("human anatomical forward source must be explicit")
    axis_label = anatomical_frame.get("actor_forward_axis")
    axis = _axis_label_vector(axis_label, owner="human actor_forward_axis")
    return axis, {
        "kind": "generated_human_runtime_manifest",
        "path": str(path),
        "sha256": sha256_file(path),
        "json_field": "anatomical_frame.actor_forward_axis",
        "declared_axis": axis_label,
        "declared_source": source,
    }


def _beagle_anatomical_forward_binding(
    *, asset: Mapping[str, Any], asset_manifest_path: str | Path
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    records = [
        record
        for record in asset.get("files", [])
        if isinstance(record, Mapping) and record.get("role") == "animation_qa"
    ]
    if len(records) != 1:
        raise MixedCaptureError(
            "Beagle package must bind exactly one animation_qa file"
        )
    record = records[0]
    raw_path = record.get("path")
    try:
        path = resolve_declared_path(
            raw_path,
            manifest_dir=Path(asset_manifest_path).resolve().parent,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise MixedCaptureError(f"Beagle animation_qa path is invalid: {exc}") from exc
    if path.is_symlink() or not path.is_file():
        raise MixedCaptureError("Beagle animation_qa is not a regular package file")
    actual_sha256 = sha256_file(path)
    if path.stat().st_size != record.get("byte_size") or actual_sha256 != record.get(
        "sha256"
    ):
        raise MixedCaptureError("Beagle animation_qa bytes differ from its manifest")
    animation_qa = load_json(path)
    if (
        animation_qa.get("schema") != "avengine_m2_animation_qa_v1"
        or animation_qa.get("status") != "pass"
    ):
        raise MixedCaptureError("Beagle animation_qa must be a passing M2 report")
    terminal_motion = animation_qa.get("semantic_terminal_motion")
    if not isinstance(terminal_motion, Mapping):
        raise MixedCaptureError("Beagle animation_qa lacks semantic_terminal_motion")
    if terminal_motion.get("actor_up_axis") != "+Y":
        raise MixedCaptureError("Beagle animation QA must declare actor_up_axis +Y")
    axis_label = terminal_motion.get("source_facing_axis_in_actor_frame")
    axis = _axis_label_vector(
        axis_label,
        owner="Beagle semantic_terminal_motion.source_facing_axis_in_actor_frame",
    )
    return axis, {
        "kind": "m2_animation_qa_declared_axis",
        "path": str(path),
        "sha256": actual_sha256,
        "json_field": (
            "semantic_terminal_motion.source_facing_axis_in_actor_frame"
        ),
        "declared_axis": axis_label,
    }


def _continuous_beagle_walk_states(
    states: Sequence[Any],
) -> tuple[tuple[int, ...], tuple[Any, ...]]:
    indices = tuple(
        index for index, state in enumerate(states) if state.action_id == "walk"
    )
    if (
        len(indices) != 45
        or indices
        != tuple(range(indices[0], indices[0] + len(indices)))
    ):
        raise MixedCaptureError(
            "Beagle M2 request must contain one continuous 45-state walk block"
        )
    return indices, tuple(states[index] for index in indices)


def _validate_legacy_camera(room_inputs: Any) -> None:
    rig = room_inputs.request.get("primary_camera_rig", {})
    calibration = rig.get("shared_calibration", {})
    transform = rig.get("world_from_rig", {})
    position = np.asarray(transform.get("translation_m"), dtype=np.float64)
    if position.shape != (3,) or not np.allclose(
        position, LEGACY_CAMERA_POSITION_M, rtol=0.0, atol=1.0e-12
    ):
        raise MixedCaptureError(
            f"legacy camera position must be {list(LEGACY_CAMERA_POSITION_M)}"
        )
    if not math.isclose(
        float(calibration.get("hfov_degrees", math.nan)),
        LEGACY_CAMERA_HFOV_DEG,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise MixedCaptureError("legacy camera horizontal FOV must be 105 degrees")
    actual = np.asarray(
        normalized_quaternion_xyzw(transform.get("rotation_xyzw")),
        dtype=np.float64,
    )
    half = math.radians(LEGACY_CAMERA_YAW_DEG) / 2.0
    expected = np.asarray([0.0, math.sin(half), 0.0, math.cos(half)])
    error = min(
        float(np.max(np.abs(actual - expected))),
        float(np.max(np.abs(actual + expected))),
    )
    if error > 1.0e-12:
        raise MixedCaptureError("legacy camera yaw must be +55 degrees about Habitat +Y")


def _shader_type_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.lower()
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _bind_m5_1_scene_lighting(
    simulator: Any,
    configuration: Any,
    *,
    light_setup_key: str = M5_1_LIGHT_SETUP_KEY,
) -> dict[str, Any]:
    """Copy the stage's effective setup to the AO-specific M5.1 key."""

    if not isinstance(light_setup_key, str) or not light_setup_key:
        raise MixedCaptureError("M5.1 light setup key must be a non-empty string")
    configured_hbao = bool(configuration.sim_cfg.enable_hbao)
    simulator_hbao = bool(simulator.config.sim_cfg.enable_hbao)
    if not configured_hbao or not simulator_hbao:
        raise MixedCaptureError("M5.1 HBAO configuration did not read back enabled")

    current_setup = list(simulator.get_current_light_setup())
    simulator.set_light_setup(current_setup, light_setup_key)
    registered_setup = list(simulator.get_light_setup(light_setup_key))
    registered_matches_current = registered_setup == current_setup
    if not registered_matches_current:
        raise MixedCaptureError(
            "M5.1 registered actor light setup differs from the scene setup"
        )
    return {
        "status": "pass",
        "hbao": {
            "requested": True,
            "configuration_readback": configured_hbao,
            "simulator_readback": simulator_hbao,
            "effect_scope": (
                "screen-space ambient occlusion; not dynamic shadow-map evidence"
            ),
        },
        "scene_lighting": {
            "actor_light_setup_key": light_setup_key,
            "source_api": "Simulator.get_current_light_setup",
            "registration_api": "Simulator.set_light_setup",
            "current_light_count": len(current_setup),
            "registered_light_count": len(registered_setup),
            "registered_setup_matches_current": registered_matches_current,
        },
    }


def _actor_render_creation_evidence(
    actor: Any,
    *,
    actor_id: str,
    requested_shader_type: str,
    light_setup_key: str,
) -> dict[str, Any]:
    creation_shader_type = _shader_type_name(actor.creation_attributes.shader_type)
    if creation_shader_type != requested_shader_type:
        raise MixedCaptureError(
            f"{actor_id} creation shader type is {creation_shader_type!r}, "
            f"not {requested_shader_type!r}"
        )
    return {
        "status": "pass",
        "requested_shader_type": requested_shader_type,
        "creation_shader_type_readback": creation_shader_type,
        "creation_light_setup_key_argument": light_setup_key,
        "creation_light_setup_binding_api": (
            "ArticulatedObjectManager."
            "add_articulated_object_by_template_handle(light_setup_key=...)"
        ),
        "native_per_actor_light_key_readback": (
            "not_exposed_by_pinned_habitat_binding"
        ),
    }


def _instantiate_human(
    simulator: Any,
    *,
    package: HumanRuntimePackage,
    habitat_sim: Any,
    semantic_id: int,
    light_setup_key: str | None = None,
    shader_type: str | None = None,
) -> tuple[Any, Any, tuple[HabitatLinkJointBlock, ...]]:
    if light_setup_key is not None and (
        not isinstance(light_setup_key, str) or not light_setup_key
    ):
        raise MixedCaptureError(
            "human light_setup_key must be a non-empty string when provided"
        )
    if shader_type is not None and shader_type not in {
        "material",
        "flat",
        "phong",
        "pbr",
    }:
        raise MixedCaptureError(
            "human shader_type must be material, flat, phong, or pbr"
        )
    manager = simulator.metadata_mediator.ao_template_manager
    loaded = manager.load_configs(str(package.habitat_ao_config))
    prefix = package.habitat_ao_config.stem.removesuffix(".ao_config")
    handles = manager.get_template_handles(prefix)
    if len(loaded) != 1 or len(handles) != 1:
        raise MixedCaptureError(
            f"expected one human AO template, got ids={loaded}, handles={handles}"
        )
    attributes = manager.get_template_by_handle(handles[0])
    if attributes is None:
        raise MixedCaptureError("cannot retrieve the loaded human AO template")
    attributes.semantic_id = int(semantic_id)
    if shader_type is not None:
        attributes.shader_type = shader_type
    handle = f"{handles[0]}.m5_1_semantic{semantic_id}"
    if int(manager.register_template(attributes, handle)) < 0:
        raise MixedCaptureError("failed to register the semantic human AO template")
    object_manager = simulator.get_articulated_object_manager()
    if light_setup_key is None:
        actor = object_manager.add_articulated_object_by_template_handle(handle)
    else:
        actor = object_manager.add_articulated_object_by_template_handle(
            handle,
            light_setup_key=light_setup_key,
        )
    if actor is None:
        raise MixedCaptureError("Habitat failed to instantiate the Rocketbox human")
    actor.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    blocks = tuple(
        HabitatLinkJointBlock(
            link_name=actor.get_link_name(link_id),
            joint_position_offset=int(actor.get_link_joint_pos_offset(link_id)),
            joint_position_count=int(actor.get_link_num_joint_pos(link_id)),
        )
        for link_id in actor.get_link_ids()
    )
    try:
        binding = bind_local_tr_habitat_layout(
            package.mapping,
            blocks,
            joint_position_count=len(actor.joint_positions),
        )
    except (TypeError, ValueError) as exc:
        raise MixedCaptureError(f"human AO link layout differs: {exc}") from exc
    if actor.get_link_name(-1) != package.mapping.root_joint_id:
        raise MixedCaptureError("human AO base link differs from the synthetic root")
    if int(actor.creation_attributes.semantic_id) != semantic_id:
        raise MixedCaptureError("human AO creation semantic ID differs")
    if shader_type is not None and _shader_type_name(
        actor.creation_attributes.shader_type
    ) != shader_type:
        raise MixedCaptureError("human AO creation shader type differs")
    _set_scene_node_semantic_readback(actor, semantic_id)
    return actor, binding, blocks


def _save_array(output: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = output / "arrays" / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(value)
    np.save(path, array, allow_pickle=False)
    readback = np.load(path, mmap_mode="r", allow_pickle=False)
    if readback.dtype != array.dtype or readback.shape != array.shape or not np.array_equal(
        readback, array
    ):
        raise MixedCaptureError(f"saved {name} array differs on readback")
    return {
        **file_record(path, relative_to=output),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "readback_verified": True,
    }


def _human_skin_link_readback_error(
    articulated_object: Any,
    package: HumanRuntimePackage,
    *,
    world_from_skin_root: np.ndarray,
    translations_m: np.ndarray,
    rotations_xyzw: np.ndarray,
) -> float:
    """Compare skin links with FK without assuming joint-list topological order.

    The two zero-weight ancestors are deliberately appended so original
    JOINTS_0 ordinals remain unchanged.  Their list order is consequently not
    parent-first even though the hierarchy itself is one valid tree.
    """

    mapping = package.mapping
    translations = dict(
        zip(mapping.runtime_joint_order, translations_m, strict=True)
    )
    rotations = dict(zip(mapping.runtime_joint_order, rotations_xyzw, strict=True))
    joint_by_name = {joint.joint_id: joint for joint in mapping.joints}
    link_by_name = {
        articulated_object.get_link_name(link_id): int(link_id)
        for link_id in articulated_object.get_link_ids()
    }
    world_by_name: dict[str, np.ndarray] = {
        mapping.root_joint_id: np.asarray(world_from_skin_root, dtype=np.float64)
    }
    visiting: set[str] = set()

    def world(joint_id: str) -> np.ndarray:
        retained = world_by_name.get(joint_id)
        if retained is not None:
            return retained
        if joint_id in visiting:
            raise MixedCaptureError("human skin hierarchy contains a cycle")
        visiting.add(joint_id)
        try:
            joint = joint_by_name[joint_id]
            if joint.parent_joint_id is None:
                raise MixedCaptureError("human skin contains a second root")
            local = np.eye(4, dtype=np.float64)
            local[:3, :3] = quaternion_xyzw_to_matrix(rotations[joint_id])
            local[:3, 3] = translations[joint_id]
            value = world(joint.parent_joint_id) @ local
            world_by_name[joint_id] = value
            return value
        except KeyError as exc:
            raise MixedCaptureError(
                f"human FK lacks joint/link pose for {joint_id!r}"
            ) from exc
        finally:
            visiting.remove(joint_id)

    maximum = 0.0
    for joint_id in mapping.runtime_joint_order:
        try:
            actual = np.asarray(
                articulated_object.get_link_scene_node(
                    link_by_name[joint_id]
                ).absolute_transformation(),
                dtype=np.float64,
            )
        except KeyError as exc:
            raise MixedCaptureError(
                f"Habitat lacks human skin link {joint_id!r}"
            ) from exc
        expected = world(joint_id)
        if actual.shape != (4, 4) or not np.all(np.isfinite(actual)):
            raise MixedCaptureError(
                f"Habitat human skin link {joint_id!r} is not finite 4x4"
            )
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    return maximum


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def capture_human_beagle_paths(
    *,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    human_runtime_glb_path: str | Path,
    beagle_animal_manifest_path: str | Path,
    beagle_m2_request_path: str | Path,
    human_root_path_m: Any,
    beagle_root_path_m: Any,
    output_dir: str | Path,
    runtime_root: str | Path | None = None,
    route_provenance: Mapping[str, Any] | None = None,
    require_legacy_camera: bool = False,
    human_semantic_id: int = HUMAN_SEMANTIC_ID,
    beagle_semantic_id: int = BEAGLE_SEMANTIC_ID,
) -> MixedCaptureResult:
    """Capture explicit Walking-human + declared-state Beagle on two paths."""

    if (
        isinstance(human_semantic_id, bool)
        or not isinstance(human_semantic_id, int)
        or human_semantic_id < 0
        or isinstance(beagle_semantic_id, bool)
        or not isinstance(beagle_semantic_id, int)
        or beagle_semantic_id < 0
        or human_semantic_id == beagle_semantic_id
    ):
        raise MixedCaptureError("human and Beagle semantic IDs must be distinct nonnegative integers")
    human_points = _points(human_root_path_m, owner="human root path")
    beagle_points = _points(beagle_root_path_m, owner="Beagle root path")
    output = Path(output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise MixedCaptureError(f"refusing to replace capture output: {output}")
    output.mkdir(parents=True)

    try:
        human_package = prepare_rocketbox_habitat_runtime(
            human_runtime_glb_path, output / "runtime" / "human"
        )
        room_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
        if require_legacy_camera:
            _validate_legacy_camera(room_inputs)
        m2_inputs = load_m2_inputs(
            beagle_animal_manifest_path, beagle_m2_request_path
        )
        beagle_bundle = load_runtime_asset_bundle(m2_inputs)
        human_forward_axis, human_forward_source = (
            _human_anatomical_forward_binding(human_package.package_manifest)
        )
        beagle_forward_axis, beagle_forward_source = (
            _beagle_anatomical_forward_binding(
                asset=m2_inputs.asset,
                asset_manifest_path=m2_inputs.asset_path,
            )
        )
        human_world = trajectory_world_matrices(
            human_points, local_forward_axis=human_forward_axis
        )
        beagle_world = trajectory_world_matrices(
            beagle_points, local_forward_axis=beagle_forward_axis
        )
        heading_records = [
            _actor_heading_evidence(
                actor_id="human0",
                points_m=human_points,
                actor_world_matrices=human_world,
                local_forward_axis=human_forward_axis,
                binding_source=human_forward_source,
            ),
            _actor_heading_evidence(
                actor_id="dog0",
                points_m=beagle_points,
                actor_world_matrices=beagle_world,
                local_forward_axis=beagle_forward_axis,
                binding_source=beagle_forward_source,
            ),
        ]
        beagle_states = compile_frame_applications(m2_inputs, beagle_bundle)
        if len(beagle_states) != 75:
            raise MixedCaptureError("Beagle M2 request must provide 75 validated states")
        beagle_walking_indices, beagle_walking_states = (
            _continuous_beagle_walk_states(beagle_states)
        )
        walking = human_package.actions.action("walk")
        if walking.sample_count != 16:
            raise MixedCaptureError("human Walking loop must contain exactly 16 samples")
        runtime = discover_runtime_root(runtime_root)
        missing = [
            record
            for record in _resolved_assets(room_inputs, runtime)
            if not record["exists"]
        ]
        if missing:
            raise MixedCaptureError("validated legacy room has missing runtime assets")

        # The pinned Habitat build requires numpy-quaternion before habitat_sim.
        import quaternion as qt

        import habitat_sim
        import magnum as mn

        configuration, modality_to_uuid, _listener_uuid, resolved_scene = (
            _make_configuration(room_inputs, runtime, output / "scene_scratch")
        )
        configuration.sim_cfg.enable_hbao = True
        if not bool(configuration.sim_cfg.enable_hbao):
            raise MixedCaptureError("M5.1 HBAO could not be enabled in configuration")
        room_declared_physics = bool(resolved_scene.get("enable_physics", False))
        physics_enabled_for_articulation = not room_declared_physics
        if physics_enabled_for_articulation:
            # Articulated-object creation requires Bullet even for KINEMATIC
            # actors.  The loop still performs zero physics steps.
            configuration.sim_cfg.enable_physics = True
        if list(modality_to_uuid) != list(FORMAL_MODALITIES):
            raise MixedCaptureError("M1 modality order changed from rgb/depth/semantic")

        rgb_frames: list[np.ndarray] = []
        semantic_frames: list[np.ndarray] = []
        actor_matrices: list[np.ndarray] = []
        skin_root_matrices: list[np.ndarray] = []
        anchors: list[np.ndarray] = []
        visibility: list[tuple[int, int]] = []
        records: list[dict[str, Any]] = []
        human_pose_hashes: set[str] = set()
        maximum_errors = {
            "human_root": 0.0,
            "human_prismatic": 0.0,
            "human_spherical": 0.0,
            "human_skin_link_fk": 0.0,
            "beagle_root": 0.0,
            "beagle_spherical": 0.0,
        }

        with habitat_sim.Simulator(configuration) as simulator:
            navmesh_path = resolved_scene.get("navmesh")
            if navmesh_path is not None and Path(navmesh_path).is_file():
                simulator.pathfinder.load_nav_mesh(str(navmesh_path))
            rendering_evidence = _bind_m5_1_scene_lighting(
                simulator,
                configuration,
            )
            simulator.seed(int(m2_inputs.request["seed"]))
            rig = room_inputs.request["primary_camera_rig"]
            camera_state = habitat_sim.AgentState()
            camera_state.position = np.asarray(
                rig["world_from_rig"]["translation_m"], dtype=np.float64
            )
            x, y, z, w = normalized_quaternion_xyzw(
                rig["world_from_rig"]["rotation_xyzw"]
            )
            camera_state.rotation = qt.quaternion(w, x, y, z)
            simulator.initialize_agent(0, camera_state)

            human, human_binding, human_blocks = _instantiate_human(
                simulator,
                package=human_package,
                habitat_sim=habitat_sim,
                semantic_id=human_semantic_id,
                light_setup_key=M5_1_LIGHT_SETUP_KEY,
                shader_type=M5_1_ACTOR_SHADER_TYPE,
            )
            human_render_evidence = _actor_render_creation_evidence(
                human,
                actor_id="human0",
                requested_shader_type=M5_1_ACTOR_SHADER_TYPE,
                light_setup_key=M5_1_LIGHT_SETUP_KEY,
            )
            template_manager = simulator.metadata_mediator.ao_template_manager
            dog_config = beagle_bundle.paths_by_role["habitat_ao_config"]
            loaded_dog = template_manager.load_configs(str(dog_config))
            dog_prefix = dog_config.stem.removesuffix(".ao_config")
            dog_handles = template_manager.get_template_handles(dog_prefix)
            if len(loaded_dog) != 1 or len(dog_handles) != 1:
                raise MixedCaptureError("expected exactly one Beagle AO template")
            beagle, beagle_binding = _instantiate_actor_with_semantic_template(
                simulator,
                bundle=beagle_bundle,
                habitat_sim=habitat_sim,
                base_handle=dog_handles[0],
                semantic_id=beagle_semantic_id,
                actor_index=1,
                light_setup_key=M5_1_LIGHT_SETUP_KEY,
                shader_type=M5_1_ACTOR_SHADER_TYPE,
            )
            beagle_render_evidence = _actor_render_creation_evidence(
                beagle,
                actor_id="dog0",
                requested_shader_type=M5_1_ACTOR_SHADER_TYPE,
                light_setup_key=M5_1_LIGHT_SETUP_KEY,
            )
            human_head_link = _link_id_by_name(human, HEAD_LINK_NAME)
            human_mouth_link = _link_id_by_name(human, MOUTH_LINK_NAME)
            beagle_mouth_link = _link_id_by_name(beagle, BEAGLE_MOUTH_LINK_NAME)
            sensors = [
                simulator.sensors[modality_to_uuid[modality]]
                for modality in FORMAL_MODALITIES
            ]
            human_actor_from_skin = np.asarray(
                human_package.actor_from_skin_root, dtype=np.float64
            )
            beagle_actor_from_skin = np.asarray(
                beagle_bundle.actor_from_skin_root, dtype=np.float64
            )
            initial_world_time = float(simulator.get_world_time())

            for frame_index in range(FRAME_COUNT):
                human_sample_index = frame_index % walking.sample_count
                human_translations = np.asarray(
                    walking.translations_m[human_sample_index], dtype=np.float64
                )
                human_rotations = np.asarray(
                    walking.rotations_xyzw[human_sample_index], dtype=np.float64
                )
                human_skin = human_world[frame_index] @ human_actor_from_skin
                human_joints = np.asarray(
                    human_binding.map_pose(human_translations, human_rotations),
                    dtype=np.float64,
                )
                walking_state_index = frame_index % len(beagle_walking_states)
                beagle_state = beagle_walking_states[walking_state_index]
                beagle_state_index = beagle_walking_indices[walking_state_index]
                beagle_skin = beagle_world[frame_index] @ beagle_actor_from_skin
                beagle_joints = np.asarray(
                    beagle_binding.map_pose(beagle_state.joint_rotations_xyzw),
                    dtype=np.float64,
                )

                _apply_root_with_habitat(human, human_skin, qt=qt, mn=mn)
                human.joint_positions = human_joints.copy()
                _apply_root_with_habitat(beagle, beagle_skin, qt=qt, mn=mn)
                beagle.joint_positions = beagle_joints.copy()
                human_before = human_runtime_snapshot(simulator, human)
                beagle_before = dog_runtime_snapshot(simulator, beagle)

                human_root_error = float(
                    np.max(
                        np.abs(
                            np.asarray(human_before["world_from_skin_root"])
                            - human_skin
                        )
                    )
                )
                human_prismatic_error, human_spherical_error = (
                    mixed_joint_readback_errors(
                        np.asarray(human_before["mixed_joint_positions"]),
                        human_joints,
                        human_blocks,
                    )
                )
                human_fk_error = _human_skin_link_readback_error(
                    human,
                    human_package,
                    world_from_skin_root=human_skin,
                    translations_m=human_translations,
                    rotations_xyzw=human_rotations,
                )
                beagle_root_error = float(
                    np.max(
                        np.abs(
                            np.asarray(beagle_before["world_from_skin_root"])
                            - beagle_skin
                        )
                    )
                )
                beagle_joint_error = _quaternion_block_error(
                    np.asarray(beagle_before["joint_positions_xyzw"]),
                    beagle_joints,
                )
                frame_errors = {
                    "human_root": human_root_error,
                    "human_prismatic": human_prismatic_error,
                    "human_spherical": human_spherical_error,
                    "human_skin_link_fk": human_fk_error,
                    "beagle_root": beagle_root_error,
                    "beagle_spherical": beagle_joint_error,
                }
                for key, value in frame_errors.items():
                    maximum_errors[key] = max(maximum_errors[key], float(value))
                if (
                    human_root_error > _ROOT_READBACK_ATOL
                    or human_prismatic_error > _JOINT_READBACK_ATOL
                    or human_spherical_error > _JOINT_READBACK_ATOL
                    or human_fk_error > _LINK_MATRIX_READBACK_ATOL
                    or beagle_root_error > _ROOT_READBACK_ATOL
                    or beagle_joint_error > _JOINT_READBACK_ATOL
                ):
                    raise MixedCaptureError(
                        f"frame {frame_index} articulated readback failed: {frame_errors}"
                    )

                observation = simulator.render_sensors(sensors)
                arrays = _validate_observation_arrays(
                    observation, modality_to_uuid
                )
                frame_visibility = (
                    int(np.count_nonzero(arrays["semantic"] == human_semantic_id)),
                    int(np.count_nonzero(arrays["semantic"] == beagle_semantic_id)),
                )
                frame_anchors = np.stack(
                    (
                        _node_world_position(human, human_head_link),
                        _node_world_position(human, human_mouth_link),
                        _node_world_position(beagle, beagle_mouth_link),
                    ),
                    axis=0,
                )
                human_after = human_runtime_snapshot(simulator, human)
                beagle_after = dog_runtime_snapshot(simulator, beagle)
                if (
                    human_before["sha256"] != human_after["sha256"]
                    or beagle_before["sha256"] != beagle_after["sha256"]
                ):
                    raise MixedCaptureError(
                        f"frame {frame_index} render changed an articulated state"
                    )
                if float(simulator.get_world_time()) != initial_world_time:
                    raise MixedCaptureError(
                        f"frame {frame_index} advanced Habitat world time"
                    )

                pose_hash = canonical_json_sha256(
                    {
                        "sample_index": human_sample_index,
                        "translations_m": human_translations.tolist(),
                        "rotations_xyzw": human_rotations.tolist(),
                    }
                )
                human_pose_hashes.add(pose_hash)
                rgb = np.asarray(arrays["rgb"])[..., :3].astype(np.uint8, copy=True)
                semantic = np.asarray(arrays["semantic"]).copy()
                rgb_frames.append(rgb)
                semantic_frames.append(semantic)
                actor_matrices.append(
                    np.stack((human_world[frame_index], beagle_world[frame_index]))
                )
                skin_root_matrices.append(
                    np.stack(
                        (
                            np.asarray(human_before["world_from_skin_root"]),
                            np.asarray(beagle_before["world_from_skin_root"]),
                        )
                    )
                )
                anchors.append(frame_anchors)
                visibility.append(frame_visibility)
                records.append(
                    {
                        "frame_index": frame_index,
                        "pts_ticks": frame_index * TICKS_PER_FRAME,
                        "human": {
                            "action_id": "walk",
                            "source_action_name": walking.source_action_name,
                            "action_sample_index": human_sample_index,
                            "pose_sha256": pose_hash,
                            "actor_root_position_m": human_points[
                                frame_index
                            ].tolist(),
                            "skin_root_readback_position_m": np.asarray(
                                human_before["world_from_skin_root"]
                            )[:3, 3].tolist(),
                            "semantic_id": human_semantic_id,
                            "semantic_visible_pixels": frame_visibility[0],
                            "head_anchor_m": frame_anchors[0].tolist(),
                            "mouth_emitter_anchor_m": frame_anchors[1].tolist(),
                            "readback": {
                                "maximum_root_error": human_root_error,
                                "maximum_prismatic_error": human_prismatic_error,
                                "maximum_spherical_error": human_spherical_error,
                                "maximum_skin_link_fk_error": human_fk_error,
                                "state_sha256": human_before["sha256"],
                            },
                        },
                        "beagle": {
                            "m2_state_index": beagle_state_index,
                            "action_id": beagle_state.action_id,
                            "action_sample_index": beagle_state.action_sample_index,
                            "actor_root_position_m": beagle_points[
                                frame_index
                            ].tolist(),
                            "skin_root_readback_position_m": np.asarray(
                                beagle_before["world_from_skin_root"]
                            )[:3, 3].tolist(),
                            "semantic_id": beagle_semantic_id,
                            "semantic_visible_pixels": frame_visibility[1],
                            "mouth_emitter_anchor_m": frame_anchors[2].tolist(),
                            "readback": {
                                "maximum_root_error": beagle_root_error,
                                "maximum_spherical_error": beagle_joint_error,
                                "state_sha256": beagle_before["sha256"],
                            },
                        },
                        "observation": {
                            "calls": 1,
                            "rgb_payload_sha256": array_sha256(
                                modality_to_uuid["rgb"], arrays["rgb"]
                            ),
                            "semantic_payload_sha256": array_sha256(
                                modality_to_uuid["semantic"], arrays["semantic"]
                            ),
                        },
                    }
                )

        rgb_array = np.ascontiguousarray(np.stack(rgb_frames, axis=0))
        semantic_array = np.ascontiguousarray(np.stack(semantic_frames, axis=0))
        actor_array = np.ascontiguousarray(np.stack(actor_matrices, axis=0))
        skin_root_array = np.ascontiguousarray(
            np.stack(skin_root_matrices, axis=0)
        )
        anchor_array = np.ascontiguousarray(np.stack(anchors, axis=0))
        visibility_array = np.asarray(visibility, dtype=np.int64)
        if len(human_pose_hashes) != walking.sample_count:
            raise MixedCaptureError(
                "270-frame capture did not exercise every human Walking sample"
            )
        if np.max(visibility_array, axis=0).min() <= 0:
            raise MixedCaptureError(
                "the fixed legacy camera never observed one of the two semantic IDs"
            )
        if not np.array_equal(actor_array[:, 0, :3, 3], human_points) or not np.array_equal(
            actor_array[:, 1, :3, 3], beagle_points
        ):
            raise MixedCaptureError("retained actor roots differ from input paths")

        artifacts = {
            "rgb": _save_array(output, "rgb", rgb_array),
            "semantic": _save_array(output, "semantic", semantic_array),
            "actor_world_matrices": _save_array(
                output, "actor_world_matrices", actor_array
            ),
            "skin_root_world_matrices": _save_array(
                output, "skin_root_world_matrices", skin_root_array
            ),
            "anchor_positions_m": _save_array(
                output, "anchor_positions_m", anchor_array
            ),
            "semantic_visibility_pixels": _save_array(
                output, "semantic_visibility_pixels", visibility_array
            ),
        }
        records_path = output / "frame_readback.json"
        _write_json(records_path, records)
        retained_rig = room_inputs.request["primary_camera_rig"]
        evidence: dict[str, Any] = {
            "schema": MIXED_CAPTURE_SCHEMA,
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FRAME_RATE_HZ,
            "time_base_hz": TIME_BASE_HZ,
            "physics_steps": 0,
            "physics_configuration": {
                "room_declared_enable_physics": room_declared_physics,
                "enabled_for_articulated_object_creation": (
                    physics_enabled_for_articulation
                ),
                "effective_enable_physics": True,
            },
            "observation_calls_per_frame": 1,
            "camera": {
                "position_m": list(
                    retained_rig["world_from_rig"]["translation_m"]
                ),
                "rotation_xyzw": list(
                    retained_rig["world_from_rig"]["rotation_xyzw"]
                ),
                "horizontal_fov_deg": retained_rig["shared_calibration"][
                    "hfov_degrees"
                ],
                "legacy_camera_contract_required": require_legacy_camera,
            },
            "actors": [
                {
                    "actor_id": "human0",
                    "actor_class": "human",
                    "semantic_id": human_semantic_id,
                    "action": "Walking",
                    "fixed_state_playback": True,
                    "head_link": HEAD_LINK_NAME,
                    "emitter_link": MOUTH_LINK_NAME,
                    "local_anatomical_forward_axis": list(human_forward_axis),
                    "rendering": human_render_evidence,
                },
                {
                    "actor_id": "dog0",
                    "actor_class": "dog",
                    "semantic_id": beagle_semantic_id,
                    "state_source": (
                        "only continuous 45-state validated walk block repeated "
                        "modulo 45"
                    ),
                    "emitter_link": BEAGLE_MOUTH_LINK_NAME,
                    "local_anatomical_forward_axis": list(beagle_forward_axis),
                    "rendering": beagle_render_evidence,
                },
            ],
            "rendering": rendering_evidence,
            "heading_alignment": {
                "schema": HEADING_ALIGNMENT_SCHEMA,
                "status": "pass",
                "gate": {
                    "required_actor_ids": ["human0", "dog0"],
                    "all_actors_all_frames_passed": True,
                },
                "actors": heading_records,
            },
            "inputs": {
                "room_manifest": {
                    "path": str(Path(room_manifest_path).resolve()),
                    "sha256": sha256_file(room_manifest_path),
                },
                "m1_request": {
                    "path": str(Path(m1_request_path).resolve()),
                    "sha256": sha256_file(m1_request_path),
                },
                "human_runtime_glb": {
                    "path": str(Path(human_runtime_glb_path).resolve()),
                    "sha256": sha256_file(human_runtime_glb_path),
                },
                "beagle_manifest": {
                    "path": str(Path(beagle_animal_manifest_path).resolve()),
                    "sha256": sha256_file(beagle_animal_manifest_path),
                },
                "beagle_m2_request": {
                    "path": str(Path(beagle_m2_request_path).resolve()),
                    "sha256": sha256_file(beagle_m2_request_path),
                },
                "route_provenance": dict(route_provenance or {}),
            },
            "runtime": {
                "human_package_manifest": file_record(
                    human_package.package_manifest, relative_to=output
                ),
                "human_walking_sample_count": walking.sample_count,
                "human_distinct_pose_count": len(human_pose_hashes),
                "beagle_declared_state_count": len(beagle_states),
                "beagle_selected_walking_state_count": len(
                    beagle_walking_states
                ),
                "beagle_selected_walking_state_indices": list(
                    beagle_walking_indices
                ),
                "beagle_selected_walking_applied_state_hashes": [
                    state.declared_applied_state_hash
                    for state in beagle_walking_states
                ],
                "beagle_selected_walking_source_request_sha256": sha256_file(
                    beagle_m2_request_path
                ),
            },
            "readback": {
                "maximum_errors": maximum_errors,
                "semantic_visible_frame_count": {
                    "human0": int(np.count_nonzero(visibility_array[:, 0] > 0)),
                    "dog0": int(np.count_nonzero(visibility_array[:, 1] > 0)),
                },
                "semantic_maximum_visible_pixels": {
                    "human0": int(np.max(visibility_array[:, 0])),
                    "dog0": int(np.max(visibility_array[:, 1])),
                },
                "frame_records": file_record(records_path, relative_to=output),
            },
            "array_artifacts": artifacts,
            "anchor_order": [
                "human0.head",
                "human0.mouth_emitter",
                "dog0.mouth_emitter",
            ],
            "claim_boundary": (
                "M5.1 fixed-state mixed visual research canary; no asset, room, "
                "episode, or dataset admission claim"
            ),
        }
        evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
        evidence_path = output / "evidence.json"
        _write_json(evidence_path, evidence)
        return MixedCaptureResult(
            output_dir=output,
            rgb=rgb_array,
            semantic=semantic_array,
            actor_world_matrices=actor_array,
            skin_root_world_matrices=skin_root_array,
            anchor_positions_m=anchor_array,
            semantic_visibility_pixels=visibility_array,
            records=tuple(records),
            evidence=evidence,
        )
    except (HabitatCaptureError, OSError, ValueError) as exc:
        if isinstance(exc, MixedCaptureError):
            raise
        raise MixedCaptureError(str(exc)) from exc


def capture_legacy_route(
    *,
    route_manifest_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    human_runtime_glb_path: str | Path,
    beagle_animal_manifest_path: str | Path,
    beagle_m2_request_path: str | Path,
    output_dir: str | Path,
    runtime_root: str | Path | None = None,
) -> MixedCaptureResult:
    """Read and capture the committed legacy route without re-deriving paths."""

    route_path = Path(route_manifest_path).resolve()
    route = load_json(route_path)
    assert_valid_route_manifest(route)
    routes = route["routes"]
    human_path = routes["human_path"]["habitat_trajectory_m"]
    beagle_path = routes["dog_path"]["habitat_trajectory_m"]
    provenance = {
        "route_manifest_path": str(route_path),
        "route_manifest_sha256": sha256_file(route_path),
        "route_id": route["route_id"],
        "human_habitat_trajectory_sha256": routes["human_path"][
            "habitat_trajectory_sha256"
        ],
        "dog_habitat_trajectory_sha256": routes["dog_path"][
            "habitat_trajectory_sha256"
        ],
        "path_consumption": "verbatim_manifest_routes_habitat_trajectory_m",
    }
    return capture_human_beagle_paths(
        room_manifest_path=room_manifest_path,
        m1_request_path=m1_request_path,
        human_runtime_glb_path=human_runtime_glb_path,
        beagle_animal_manifest_path=beagle_animal_manifest_path,
        beagle_m2_request_path=beagle_m2_request_path,
        human_root_path_m=human_path,
        beagle_root_path_m=beagle_path,
        output_dir=output_dir,
        runtime_root=runtime_root,
        route_provenance=provenance,
        require_legacy_camera=True,
    )


__all__ = [
    "BEAGLE_MOUTH_LINK_NAME",
    "BEAGLE_SEMANTIC_ID",
    "HUMAN_SEMANTIC_ID",
    "MIXED_CAPTURE_SCHEMA",
    "MixedCaptureError",
    "MixedCaptureResult",
    "capture_human_beagle_paths",
    "capture_legacy_route",
    "trajectory_world_matrices",
]

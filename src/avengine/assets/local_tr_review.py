"""Research-only Habitat review capture for local-translation actions.

This module is intentionally separate from :mod:`avengine.assets.habitat_capture`.
The formal M2 v1 path remains rotation-only and cannot consume the local-TR v2
action or mixed prismatic/spherical Habitat layout used here.

An existing M2 capture request may be reused only as a 75-frame schedule: its
root transforms, action ids, action ticks, presentation ticks, and room/camera
bindings are retained, while its rotation-only joint states and v1 hashes are
explicitly ignored.  The actual local pose always comes from one hash-bound
``avengine_m2_local_tr_actions_v2`` artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.m1.evidence import array_sha256


# Keep the schema and all negative claims distinct from formal M2 v1 evidence.
EVIDENCE_SCHEMA = "avengine_m2_habitat_local_tr_review_evidence_v2"
READBACK_SCHEMA = "avengine_m2_habitat_local_tr_readback_v2"
REBASE_REPORT_SCHEMA = "avengine_m2_skin_root_rebase_local_tr_v2"
FORMAL_MODALITIES = ("rgb", "depth", "semantic")
DEFAULT_SEMANTIC_ID = 200
_ROOT_READBACK_ATOL = 2.0e-6
_JOINT_READBACK_ATOL = 2.0e-6
_LINK_MATRIX_READBACK_ATOL = 5.0e-5


class LocalTRReviewError(RuntimeError):
    """The research review could not prove its fixed-state contract."""


Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class LocalTRReviewFrame:
    """One request schedule state resolved to a local-TR v2 sample."""

    frame_index: int
    pts_ticks: int
    action_id: str
    action_time_ticks: int
    effective_action_tick: int
    action_sample_index: int
    source_action_name: str
    world_from_actor: Matrix4
    world_from_skin_root: Matrix4
    translations_m: tuple[tuple[float, float, float], ...]
    rotations_xyzw: tuple[tuple[float, float, float, float], ...]


def _matrix_tuple(value: np.ndarray) -> Matrix4:
    return tuple(tuple(float(component) for component in row) for row in value)  # type: ignore[return-value]


def _rigid_matrix(value: Any, *, owner: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LocalTRReviewError(f"{owner} must be a finite rigid 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise LocalTRReviewError(f"{owner} must be a finite rigid 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-7):
        raise LocalTRReviewError(f"{owner} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-7):
        raise LocalTRReviewError(f"{owner} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7):
        raise LocalTRReviewError(f"{owner} rotation is not proper")
    return matrix


def _transform_matrix(value: Any, *, owner: str) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise LocalTRReviewError(f"{owner} must be a transform object")
    try:
        translation = np.asarray(value["translation_m"], dtype=np.float64)
        x, y, z, w = normalized_quaternion_xyzw(value["rotation_xyzw"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalTRReviewError(f"{owner} is not a valid rigid transform") from exc
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise LocalTRReviewError(f"{owner}.translation_m must be a finite vec3")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = translation
    return result


def _integer(value: Any, *, owner: str, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LocalTRReviewError(f"{owner} must be an integer")
    if nonnegative and value < 0:
        raise LocalTRReviewError(f"{owner} must be non-negative")
    return value


def validate_review_schedule(
    schedule: Mapping[str, Any],
    *,
    runtime_joint_order: Sequence[str],
    room_inputs: Any | None = None,
) -> list[str]:
    """Validate the bounded subset of a v1 request reused as a schedule."""

    errors: list[str] = []
    if schedule.get("schema") != "avengine_m2_articulated_capture_request_v1":
        errors.append("schedule must use the M2 articulated capture request v1 schema")
    if schedule.get("modalities") != list(FORMAL_MODALITIES):
        errors.append("schedule modalities must remain ordered rgb/depth/semantic")
    if not runtime_joint_order:
        errors.append("local-TR runtime_joint_order must not be empty")
    # The reused request belongs to a historical rotation-only projection and
    # may legitimately have fewer joints.  Its runtime_joint_order and every
    # declared joint pose/hash are outside this schedule-only contract.
    declared_order = schedule.get("runtime_joint_order")
    if (
        not isinstance(declared_order, list)
        or not declared_order
        or any(not isinstance(name, str) or not name for name in declared_order)
        or len(set(declared_order)) != len(declared_order)
    ):
        errors.append("schedule must retain its historical runtime_joint_order")
    capture_policy = schedule.get("capture_policy")
    if not isinstance(capture_policy, Mapping) or (
        capture_policy.get("state_evaluation") != "explicit_fixed_state"
        or capture_policy.get("advance_clock_between_modalities") is not False
        or capture_policy.get("free_running_animation") is not False
    ):
        errors.append("schedule must require explicit fixed-state capture")
    states = schedule.get("states")
    if not isinstance(states, list) or len(states) != 75:
        errors.append("schedule must contain exactly 75 states")
    else:
        previous_pts = -1
        for ordinal, state in enumerate(states):
            if not isinstance(state, Mapping):
                errors.append(f"schedule state {ordinal} is not an object")
                continue
            if state.get("frame_index") != ordinal:
                errors.append(f"schedule state {ordinal} frame_index is not sequential")
            pts = state.get("pts_ticks")
            if isinstance(pts, bool) or not isinstance(pts, int) or pts < 0:
                errors.append(f"schedule state {ordinal} pts_ticks is invalid")
            elif pts <= previous_pts and ordinal:
                errors.append(f"schedule state {ordinal} pts_ticks is not increasing")
            else:
                previous_pts = int(pts)
            if not isinstance(state.get("action_id"), str) or not state.get(
                "action_id"
            ):
                errors.append(f"schedule state {ordinal} action_id is invalid")
            action_tick = state.get("action_time_ticks")
            if (
                isinstance(action_tick, bool)
                or not isinstance(action_tick, int)
                or action_tick < 0
            ):
                errors.append(f"schedule state {ordinal} action_time_ticks is invalid")
            try:
                _transform_matrix(
                    state.get("root_transform"), owner=f"schedule state {ordinal} root"
                )
            except LocalTRReviewError as exc:
                errors.append(str(exc))
    if room_inputs is not None:
        room = room_inputs.room
        request = room_inputs.request
        rig = request.get("primary_camera_rig", {})
        listener = request.get("listener", {})
        if schedule.get("room_id") != room.get("room_id"):
            errors.append("schedule room_id differs from M1 room")
        if schedule.get("camera_rig_id") != rig.get("rig_id"):
            errors.append("schedule camera_rig_id differs from M1 camera rig")
        if rig.get("rig_id") != "camera_rig_0" or rig.get("view_id") != "view0":
            errors.append("M1 review input must provide camera_rig_0/view0")
        if schedule.get("listener_id") != listener.get("listener_id"):
            errors.append("schedule listener_id differs from M1 listener")
        if schedule.get("seed") != request.get("seed"):
            errors.append("schedule and M1 request seeds differ")
    return errors


def compile_review_frames(
    schedule: Mapping[str, Any],
    actions: Any,
    mapping: Any,
) -> tuple[LocalTRReviewFrame, ...]:
    """Resolve all 75 schedule states against local-TR v2 action samples."""

    action_order = tuple(actions.runtime_joint_order)
    mapping_order = tuple(mapping.runtime_joint_order)
    if action_order != mapping_order:
        raise LocalTRReviewError("local-TR actions and Habitat mapping orders differ")
    errors = validate_review_schedule(
        schedule, runtime_joint_order=action_order, room_inputs=None
    )
    if errors:
        raise LocalTRReviewError("; ".join(errors))
    actor_from_skin_root = _rigid_matrix(
        mapping.actor_from_skin_root, owner="actor_from_skin_root"
    )
    frames: list[LocalTRReviewFrame] = []
    for ordinal, state in enumerate(schedule["states"]):
        action_id = str(state["action_id"])
        try:
            clip = actions.action(action_id)
        except (KeyError, ValueError) as exc:
            raise LocalTRReviewError(
                f"schedule state {ordinal} refers to unknown action {action_id!r}"
            ) from exc
        action_time_ticks = _integer(
            state["action_time_ticks"],
            owner=f"state {ordinal} action_time_ticks",
            nonnegative=True,
        )
        effective_tick = action_time_ticks % int(clip.loop_duration_ticks)
        try:
            sample_index = tuple(clip.sample_ticks).index(effective_tick)
        except ValueError as exc:
            raise LocalTRReviewError(
                f"schedule state {ordinal} action tick is off the local-TR sample grid"
            ) from exc
        translations = np.asarray(clip.translations_m[sample_index], dtype=np.float64)
        rotations = np.asarray(clip.rotations_xyzw[sample_index], dtype=np.float64)
        if translations.shape != (len(action_order), 3) or not np.all(
            np.isfinite(translations)
        ):
            raise LocalTRReviewError(
                f"schedule state {ordinal} has an invalid local translation sample"
            )
        if rotations.shape != (len(action_order), 4) or not np.all(
            np.isfinite(rotations)
        ):
            raise LocalTRReviewError(
                f"schedule state {ordinal} has an invalid local rotation sample"
            )
        norms = np.linalg.norm(rotations, axis=1)
        if float(np.max(np.abs(norms - 1.0))) > 1e-9:
            raise LocalTRReviewError(
                f"schedule state {ordinal} contains non-unit local rotations"
            )
        world_from_actor = _transform_matrix(
            state["root_transform"], owner=f"schedule state {ordinal} root"
        )
        world_from_skin_root = world_from_actor @ actor_from_skin_root
        frames.append(
            LocalTRReviewFrame(
                frame_index=ordinal,
                pts_ticks=int(state["pts_ticks"]),
                action_id=action_id,
                action_time_ticks=action_time_ticks,
                effective_action_tick=effective_tick,
                action_sample_index=sample_index,
                source_action_name=str(clip.source_action_name),
                world_from_actor=_matrix_tuple(world_from_actor),
                world_from_skin_root=_matrix_tuple(world_from_skin_root),
                translations_m=tuple(
                    tuple(float(component) for component in row) for row in translations
                ),
                rotations_xyzw=tuple(
                    tuple(float(component) for component in row) for row in rotations
                ),
            )
        )
    return tuple(frames)


def actor_from_skin_root_from_rebase_report(
    report: Mapping[str, Any],
    *,
    visual_sha256: str,
    visual_path: str | Path,
    visual_byte_size: int,
    report_path: str | Path,
    report_sha256: str,
) -> tuple[np.ndarray, str]:
    """Validate the local-TR rebase lineage and return its explicit root map."""

    if report.get("schema") != REBASE_REPORT_SCHEMA:
        raise LocalTRReviewError("rebase report is not the local-TR v2 schema")
    if report.get("status") != "pass":
        raise LocalTRReviewError("local-TR rebase report status is not pass")
    if report.get("qualification_claim") is not False:
        raise LocalTRReviewError("local-TR rebase report makes a qualification claim")
    output = report.get("output")
    skin = report.get("skin")
    runtime_contract = report.get("runtime_contract")
    if not isinstance(output, Mapping) or output.get("sha256") != visual_sha256:
        raise LocalTRReviewError("local-TR rebase output hash differs from visual GLB")
    if (
        output.get("byte_size") != visual_byte_size
        or not isinstance(output.get("path"), str)
        or Path(str(output["path"])).resolve() != Path(visual_path).resolve()
    ):
        raise LocalTRReviewError(
            "local-TR rebase output path/size differs from visual GLB"
        )
    if not isinstance(skin, Mapping) or "actor_from_canonical_root" not in skin:
        raise LocalTRReviewError(
            "local-TR rebase report lacks its actor/root transform"
        )
    if (
        not isinstance(runtime_contract, Mapping)
        or runtime_contract.get("schema") != "avengine_m2_local_tr_runtime_v2"
        or runtime_contract.get("per_bone_dynamic_translation") is not True
    ):
        raise LocalTRReviewError("rebase report lacks the local-TR v2 runtime contract")
    transform = _rigid_matrix(
        skin["actor_from_canonical_root"],
        owner="rebase_report.skin.actor_from_canonical_root",
    )
    resolved_report = Path(report_path).resolve()
    source = f"{resolved_report}#sha256={report_sha256}:skin.actor_from_canonical_root"
    return transform, source


def mixed_joint_readback_errors(
    actual: Any,
    expected: Any,
    link_blocks: Sequence[Any],
) -> tuple[float, float]:
    """Return maximum prismatic and sign-invariant spherical readback errors."""

    actual_array = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected_array = np.asarray(expected, dtype=np.float64).reshape(-1)
    if actual_array.shape != expected_array.shape or not np.all(
        np.isfinite(actual_array)
    ):
        return math.inf, math.inf
    covered = np.zeros(actual_array.shape, dtype=bool)
    maximum_prismatic = 0.0
    maximum_spherical = 0.0
    for block in link_blocks:
        offset = int(block.joint_position_offset)
        count = int(block.joint_position_count)
        if count not in {1, 4} or offset < 0 or offset + count > actual_array.size:
            return math.inf, math.inf
        if np.any(covered[offset : offset + count]):
            return math.inf, math.inf
        covered[offset : offset + count] = True
        left = actual_array[offset : offset + count]
        right = expected_array[offset : offset + count]
        if count == 1:
            maximum_prismatic = max(maximum_prismatic, float(abs(left[0] - right[0])))
            continue
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm < 1e-15 or right_norm < 1e-15:
            return math.inf, math.inf
        left /= left_norm
        right /= right_norm
        maximum_spherical = max(
            maximum_spherical,
            min(
                float(np.max(np.abs(left - right))),
                float(np.max(np.abs(left + right))),
            ),
        )
    if not np.all(covered):
        return math.inf, math.inf
    return maximum_prismatic, maximum_spherical


def _local_tr_matrix(
    translation: Sequence[float], quaternion: Sequence[float]
) -> np.ndarray:
    return _transform_matrix(
        {"translation_m": list(translation), "rotation_xyzw": list(quaternion)},
        owner="local joint transform",
    )


def skin_link_matrix_readback_error(
    articulated_object: Any,
    mapping: Any,
    frame: LocalTRReviewFrame,
) -> float:
    """Compare every same-named Habitat skin link with CPU glTF hierarchy FK."""

    translation_by_name = dict(
        zip(mapping.runtime_joint_order, frame.translations_m, strict=True)
    )
    rotation_by_name = dict(
        zip(mapping.runtime_joint_order, frame.rotations_xyzw, strict=True)
    )
    world_by_name: dict[str, np.ndarray] = {
        mapping.root_joint_id: np.asarray(frame.world_from_skin_root, dtype=np.float64)
    }
    link_id_by_name = {
        articulated_object.get_link_name(link_id): int(link_id)
        for link_id in articulated_object.get_link_ids()
    }
    maximum = 0.0
    for joint in mapping.joints:
        if joint.parent_joint_id is None:
            continue
        if joint.parent_joint_id not in world_by_name:
            raise LocalTRReviewError(
                f"joint hierarchy is not parent-first at {joint.joint_id!r}"
            )
        try:
            local = _local_tr_matrix(
                translation_by_name[joint.joint_id],
                rotation_by_name[joint.joint_id],
            )
            link_id = link_id_by_name[joint.joint_id]
        except KeyError as exc:
            raise LocalTRReviewError(
                f"missing local pose or Habitat link for {joint.joint_id!r}"
            ) from exc
        expected = world_by_name[joint.parent_joint_id] @ local
        world_by_name[joint.joint_id] = expected
        actual = np.asarray(
            articulated_object.get_link_scene_node(link_id).absolute_transformation(),
            dtype=np.float64,
        )
        if actual.shape != (4, 4) or not np.all(np.isfinite(actual)):
            raise LocalTRReviewError(
                f"Habitat link {joint.joint_id!r} matrix is not finite 4x4"
            )
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    return maximum


def _runtime_snapshot(simulator: Any, articulated_object: Any) -> dict[str, Any]:
    root = np.asarray(
        articulated_object.root_scene_node.absolute_transformation(), dtype=np.float64
    )
    joints = np.asarray(articulated_object.joint_positions, dtype=np.float64).reshape(
        -1
    )
    if root.shape != (4, 4) or not np.all(np.isfinite(root)):
        raise LocalTRReviewError("Habitat root readback is not a finite 4x4 matrix")
    if not np.all(np.isfinite(joints)):
        raise LocalTRReviewError("Habitat mixed joint readback is not finite")
    core = {
        "schema": READBACK_SCHEMA,
        "world_time_seconds": float(simulator.get_world_time()),
        "world_from_skin_root": root.tolist(),
        "mixed_joint_positions": joints.tolist(),
    }
    return {**core, "sha256": canonical_json_sha256(core)}


def _input_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise LocalTRReviewError(f"missing or unsafe review input: {resolved}")
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _safe_input_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise LocalTRReviewError(f"review input must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise LocalTRReviewError(f"missing review input: {resolved}")
    return resolved


def _prepare_output(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise LocalTRReviewError(f"review output is not an empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _apply_root(
    articulated_object: Any, matrix: np.ndarray, *, qt: Any, mn: Any
) -> None:
    quaternion_wxyz = qt.as_float_array(qt.from_rotation_matrix(matrix[:3, :3]))
    articulated_object.translation = mn.Vector3(matrix[:3, 3])
    articulated_object.rotation = mn.Quaternion(
        mn.Vector3(quaternion_wxyz[1:]), float(quaternion_wxyz[0])
    )


def _instantiate(
    simulator: Any,
    *,
    config_path: Path,
    mapping: Any,
    habitat_sim: Any,
) -> tuple[Any, Any, tuple[Any, ...]]:
    from avengine.assets.habitat import HabitatLinkJointBlock
    from avengine.assets.local_tr_habitat import bind_local_tr_habitat_layout

    manager = simulator.metadata_mediator.ao_template_manager
    loaded_ids = manager.load_configs(str(config_path))
    handle_prefix = config_path.stem.removesuffix(".ao_config")
    handles = manager.get_template_handles(handle_prefix)
    if len(loaded_ids) != 1 or len(handles) != 1:
        raise LocalTRReviewError(
            f"expected one local-TR AO template, got ids={loaded_ids}, handles={handles}"
        )
    articulated_object = simulator.get_articulated_object_manager().add_articulated_object_by_template_handle(
        handles[0]
    )
    if articulated_object is None:
        raise LocalTRReviewError("Habitat failed to instantiate the local-TR animal")
    articulated_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    blocks = tuple(
        HabitatLinkJointBlock(
            link_name=articulated_object.get_link_name(link_id),
            joint_position_offset=int(
                articulated_object.get_link_joint_pos_offset(link_id)
            ),
            joint_position_count=int(
                articulated_object.get_link_num_joint_pos(link_id)
            ),
        )
        for link_id in articulated_object.get_link_ids()
    )
    try:
        binding = bind_local_tr_habitat_layout(
            mapping,
            blocks,
            joint_position_count=len(articulated_object.joint_positions),
        )
    except (TypeError, ValueError) as exc:
        raise LocalTRReviewError(
            f"invalid measured local-TR link layout: {exc}"
        ) from exc
    return articulated_object, binding, blocks


def capture_local_tr_habitat_review(
    *,
    visual_glb: str | Path,
    actions_npz: str | Path,
    rebase_report: str | Path,
    schedule_path: str | Path,
    room_manifest: str | Path,
    room_request: str | Path,
    output_dir: str | Path,
    runtime_root: str | Path | None = None,
    semantic_id: int = DEFAULT_SEMANTIC_ID,
    shader_type: str = "pbr",
) -> dict[str, Any]:
    """Capture one fresh 75-frame local-TR Habitat review run."""

    if (
        isinstance(semantic_id, bool)
        or not isinstance(semantic_id, int)
        or semantic_id < 0
    ):
        raise LocalTRReviewError("semantic_id must be a non-negative integer")
    if shader_type not in {"phong", "pbr"}:
        raise LocalTRReviewError("shader_type must be exactly 'phong' or 'pbr'")
    visual_path = _safe_input_path(visual_glb)
    actions_path = _safe_input_path(actions_npz)
    rebase_report_path = _safe_input_path(rebase_report)
    schedule_source = _safe_input_path(schedule_path)
    room_manifest_path = _safe_input_path(room_manifest)
    room_request_path = _safe_input_path(room_request)
    input_records = {
        "visual_glb": _input_record(visual_path),
        "local_tr_actions": _input_record(actions_path),
        "local_tr_preserving_rebase_report": _input_record(rebase_report_path),
        "schedule_request": _input_record(schedule_source),
        "m1_room_manifest": _input_record(room_manifest_path),
        "m1_camera_request": _input_record(room_request_path),
    }

    from avengine.m1.contracts import (
        load_and_validate_inputs as load_m1_inputs,
        validate_loaded_scene_asset_graph,
        validate_scene_asset_graph,
    )
    from avengine.m1.habitat_capture import (
        _make_configuration,
        _resolved_assets,
        discover_runtime_root,
    )
    from avengine.assets.glb import load_glb
    from avengine.assets.habitat import build_habitat_ao_config_data
    from avengine.assets.habitat_capture import (
        CapturedFrame,
        _runtime_identity,
        _validate_observation_arrays,
        save_capture_arrays,
        write_research_review_media,
    )
    from avengine.assets.local_tr_actions import read_local_tr_actions_npz
    from avengine.assets.local_tr_habitat import build_local_tr_habitat_mapping

    room_inputs = load_m1_inputs(room_manifest_path, room_request_path)
    schedule = load_json(schedule_source)
    rebase = load_json(rebase_report_path)
    document = load_glb(visual_path)
    actions = read_local_tr_actions_npz(actions_path)
    if actions.source_glb_sha256 != document.sha256:
        raise LocalTRReviewError("local-TR actions source hash differs from visual GLB")
    actor_from_skin_root, actor_from_skin_root_source = (
        actor_from_skin_root_from_rebase_report(
            rebase,
            visual_sha256=document.sha256,
            visual_path=visual_path,
            visual_byte_size=visual_path.stat().st_size,
            report_path=rebase_report_path,
            report_sha256=sha256_file(rebase_report_path),
        )
    )
    mapping = build_local_tr_habitat_mapping(
        document,
        actions,
        actor_from_skin_root=actor_from_skin_root,
        actor_from_skin_root_source=actor_from_skin_root_source,
    )
    schedule_errors = validate_review_schedule(
        schedule,
        runtime_joint_order=actions.runtime_joint_order,
        room_inputs=room_inputs,
    )
    if schedule_errors:
        raise LocalTRReviewError("; ".join(schedule_errors))
    frames = compile_review_frames(schedule, actions, mapping)

    output = _prepare_output(output_dir)
    runtime_dir = output / "runtime"
    runtime_dir.mkdir()
    visual_copy = runtime_dir / "visual.glb"
    urdf_path = runtime_dir / "animal_local_tr.urdf"
    config_path = runtime_dir / "animal_local_tr.ao_config.json"
    mapping_path = runtime_dir / "local_tr_joint_mapping.json"
    shutil.copyfile(visual_path, visual_copy)
    urdf_path.write_text(
        mapping.render_urdf(robot_name="avengine_m2_local_tr_review"),
        encoding="utf-8",
        newline="\n",
    )
    write_json(mapping_path, mapping.joint_mapping_data())
    ao_config = build_habitat_ao_config_data(
        render_asset=visual_copy.name,
        urdf_filepath=urdf_path.name,
        semantic_id=semantic_id,
        shader_type=shader_type,
    )
    write_json(config_path, ao_config)

    runtime = discover_runtime_root(runtime_root)
    room_assets = _resolved_assets(room_inputs, runtime)
    missing_room_assets = [record for record in room_assets if not record["exists"]]
    if missing_room_assets:
        raise LocalTRReviewError("validated M1 room has missing runtime assets")
    room_asset_paths = {
        Path(record["resolved_path"]).resolve()
        for record in room_assets
        if isinstance(record, Mapping) and isinstance(record.get("resolved_path"), str)
    }
    if visual_path in room_asset_paths or visual_copy in room_asset_paths:
        raise LocalTRReviewError("animal visual collides with an M1 room asset path")
    scene_errors = validate_scene_asset_graph(room_inputs, runtime)
    if scene_errors:
        raise LocalTRReviewError(
            "M1 room graph failed before Simulator: " + "; ".join(scene_errors)
        )

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
    configuration, modality_to_uuid, _listener_uuid, resolved_scene = (
        _make_configuration(room_inputs, runtime, output)
    )
    if list(modality_to_uuid) != list(FORMAL_MODALITIES):
        raise LocalTRReviewError("M1 modality order changed from rgb/depth/semantic")

    captures: list[Any] = []
    runtime_binding: dict[str, Any] | None = None
    loaded_graph: Mapping[str, Any] | None = None
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
        )
        if loaded_errors:
            raise LocalTRReviewError(
                "fresh Simulator room graph differs from M1 declaration: "
                + "; ".join(loaded_errors)
            )
        loaded_graph = loaded_graph_value
        simulator.seed(int(schedule["seed"]))
        rig = room_inputs.request["primary_camera_rig"]
        camera_state = habitat_sim.AgentState()
        camera_state.position = np.asarray(
            rig["world_from_rig"]["translation_m"], dtype=np.float64
        )
        x, y, z, w = normalized_quaternion_xyzw(rig["world_from_rig"]["rotation_xyzw"])
        camera_state.rotation = qt.quaternion(w, x, y, z)
        simulator.initialize_agent(0, camera_state)

        articulated_object, binding, link_blocks = _instantiate(
            simulator,
            config_path=config_path,
            mapping=mapping,
            habitat_sim=habitat_sim,
        )
        runtime_binding = {
            **binding.to_json_data(),
            "base_link_id": -1,
            "base_link_name": articulated_object.get_link_name(-1),
            "motion_type": "KINEMATIC",
        }
        sensors = [
            simulator.sensors[modality_to_uuid[modality]]
            for modality in FORMAL_MODALITIES
        ]
        initial_world_time = float(simulator.get_world_time())
        for frame in frames:
            expected_root = np.asarray(frame.world_from_skin_root, dtype=np.float64)
            expected_joints = np.asarray(
                binding.map_pose(frame.translations_m, frame.rotations_xyzw),
                dtype=np.float64,
            )
            _apply_root(articulated_object, expected_root, qt=qt, mn=mn)
            articulated_object.joint_positions = expected_joints.copy()
            before = _runtime_snapshot(simulator, articulated_object)
            actual_root = np.asarray(before["world_from_skin_root"], dtype=np.float64)
            actual_joints = np.asarray(
                before["mixed_joint_positions"], dtype=np.float64
            )
            root_error = float(np.max(np.abs(actual_root - expected_root)))
            prismatic_error, spherical_error = mixed_joint_readback_errors(
                actual_joints, expected_joints, link_blocks
            )
            link_matrix_error = skin_link_matrix_readback_error(
                articulated_object, mapping, frame
            )
            if root_error > _ROOT_READBACK_ATOL:
                raise LocalTRReviewError(
                    f"frame {frame.frame_index} root readback error {root_error:.9g}"
                )
            if max(prismatic_error, spherical_error) > _JOINT_READBACK_ATOL:
                raise LocalTRReviewError(
                    f"frame {frame.frame_index} mixed readback error "
                    f"prismatic={prismatic_error:.9g}, spherical={spherical_error:.9g}"
                )
            if link_matrix_error > _LINK_MATRIX_READBACK_ATOL:
                raise LocalTRReviewError(
                    f"frame {frame.frame_index} skin-link FK readback error "
                    f"{link_matrix_error:.9g}"
                )

            observation = simulator.render_sensors(sensors)
            arrays = _validate_observation_arrays(observation, modality_to_uuid)
            semantic_pixel_count = int(
                np.count_nonzero(arrays["semantic"] == semantic_id)
            )
            if semantic_pixel_count == 0:
                raise LocalTRReviewError(
                    f"frame {frame.frame_index} has no animal semantic pixels"
                )
            after = _runtime_snapshot(simulator, articulated_object)
            if before["world_time_seconds"] != after["world_time_seconds"]:
                raise LocalTRReviewError(
                    f"frame {frame.frame_index} advanced Habitat world time"
                )
            if before["sha256"] != after["sha256"]:
                raise LocalTRReviewError(
                    f"frame {frame.frame_index} changed applied root/joint state"
                )
            modalities = {
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
            captures.append(
                CapturedFrame(
                    record={
                        "frame_index": frame.frame_index,
                        "pts_ticks": frame.pts_ticks,
                        "action_id": frame.action_id,
                        "source_action_name": frame.source_action_name,
                        "action_time_ticks": frame.action_time_ticks,
                        "effective_action_tick": frame.effective_action_tick,
                        "action_sample_index": frame.action_sample_index,
                        "world_from_actor": [
                            list(row) for row in frame.world_from_actor
                        ],
                        "world_from_skin_root": [
                            list(row) for row in frame.world_from_skin_root
                        ],
                        "local_tr_pose_sha256": canonical_json_sha256(
                            {
                                "runtime_joint_order": list(
                                    actions.runtime_joint_order
                                ),
                                "translations_m": [
                                    list(value) for value in frame.translations_m
                                ],
                                "rotations_xyzw": [
                                    list(value) for value in frame.rotations_xyzw
                                ],
                            }
                        ),
                        "runtime_application": {
                            "expected_mixed_joint_positions_sha256": (
                                canonical_json_sha256(expected_joints.tolist())
                            ),
                            "maximum_root_readback_error": root_error,
                            "maximum_prismatic_readback_error": prismatic_error,
                            "maximum_spherical_readback_error": spherical_error,
                            "maximum_skin_link_matrix_readback_error": (
                                link_matrix_error
                            ),
                            "before": before,
                            "after": after,
                            "world_time_advance_seconds": float(
                                after["world_time_seconds"]
                                - before["world_time_seconds"]
                            ),
                        },
                        "modalities": modalities,
                        "animal_semantic_visibility": {
                            "semantic_id": semantic_id,
                            "pixel_count": semantic_pixel_count,
                            "visible": True,
                        },
                    },
                    arrays=arrays,
                )
            )
        final_world_time = float(simulator.get_world_time())
        if final_world_time != initial_world_time:
            raise LocalTRReviewError("75-state review loop advanced Habitat world time")

    array_artifacts = save_capture_arrays(captures, output)
    review_media = write_research_review_media(captures, output)
    runtime_artifacts = {
        "visual_copy": file_record(visual_copy, relative_to=output),
        "habitat_urdf": file_record(urdf_path, relative_to=output),
        "habitat_ao_config": file_record(config_path, relative_to=output),
        "habitat_joint_mapping": file_record(mapping_path, relative_to=output),
    }
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "status": "review_only",
        "evidence_kind": "local_tr_v2_habitat_research_review",
        "review_only": True,
        "formal_capture": False,
        "qualification_claim": False,
        "formal_view_ids": [],
        "formal_modalities": [],
        "review_view_ids": ["view0"],
        "review_modalities": list(FORMAL_MODALITIES),
        "sensor_view_created": False,
        "request_id": schedule.get("request_id"),
        "room_id": room_inputs.room["room_id"],
        "inputs": input_records,
        "schedule_usage": {
            "retained_fields": [
                "room_id",
                "camera_rig_id",
                "listener_id",
                "seed",
                "states[].frame_index",
                "states[].pts_ticks",
                "states[].action_id",
                "states[].action_time_ticks",
                "states[].root_transform",
            ],
            "ignored_rotation_only_fields": [
                "asset_id",
                "asset_manifest_sha256",
                "runtime_joint_order",
                "pose_hash_algorithm",
                "applied_state_hash_algorithm",
                "states[].joint_states",
                "states[].pose_hash",
                "states[].applied_state_hash",
            ],
            "pose_authority": "local_tr_actions_npz",
        },
        "runtime_assets": runtime_artifacts,
        "runtime_identity": runtime_identity,
        "room_assets": room_assets,
        "loaded_room_graph": loaded_graph,
        "sensor_contract": {
            "rig_id": room_inputs.request["primary_camera_rig"]["rig_id"],
            "view_id": "view0",
            "world_from_rig": room_inputs.request["primary_camera_rig"][
                "world_from_rig"
            ],
            "shared_calibration": room_inputs.request["primary_camera_rig"][
                "shared_calibration"
            ],
            "modality_to_sensor_uuid": modality_to_uuid,
        },
        "local_tr_contract": {
            "schema": "avengine_m2_local_tr_actions_v2",
            "runtime_joint_order": list(actions.runtime_joint_order),
            "translation_driven_joint_ids": list(actions.translation_driven_joint_ids),
            "joint_position_count": runtime_binding["joint_position_count"],
            "layout": mapping.joint_mapping_data()["habitat_layout"],
            "runtime_binding": runtime_binding,
            "root_formula": (
                "world_from_skin_root = world_from_actor @ actor_from_skin_root"
            ),
        },
        "runtime_application": {
            "simulator_lifetime": "fresh_per_capture_call",
            "motion_type": "KINEMATIC",
            "state_evaluation": "explicit_fixed_state",
            "observation_calls_per_frame": 1,
            "observation_call_count": len(captures),
            "physics_steps": 0,
            "initial_world_time_seconds": initial_world_time,
            "final_world_time_seconds": final_world_time,
        },
        "frames": [dict(capture.record) for capture in captures],
        "array_artifacts": array_artifacts,
        "review_media": review_media,
        "notes": [
            "This is a non-qualifying local-TR v2 research review.",
            "It does not widen formal M2 v1 beyond rotation-only actions.",
            "The reused v1 request supplies schedule/root data only.",
        ],
    }
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    verification_errors = verify_local_tr_review_evidence(evidence, output)
    if verification_errors:
        raise LocalTRReviewError("; ".join(verification_errors))
    evidence_path = output / "evidence.json"
    write_json(evidence_path, evidence)
    readback_errors = verify_local_tr_review_evidence(evidence_path)
    if readback_errors:
        raise LocalTRReviewError(
            "written evidence failed readback: " + "; ".join(readback_errors)
        )
    return evidence


def _verify_file_record(
    record: Any,
    *,
    root: Path | None,
    owner: str,
) -> tuple[list[str], Path | None]:
    errors: list[str] = []
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        return [f"{owner} is not a file record"], None
    path = Path(str(record["path"]))
    if root is not None:
        candidate = root / path
        if candidate.is_symlink():
            return [f"{owner} is missing or symlinked"], candidate
        path = candidate.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return [f"{owner} escapes the review output"], None
    else:
        if path.is_symlink():
            return [f"{owner} is missing or symlinked"], path
        path = path.resolve()
    if not path.is_file():
        return [f"{owner} is missing or symlinked"], path
    if path.stat().st_size != record.get("byte_size"):
        errors.append(f"{owner} byte size changed")
    if sha256_file(path) != record.get("sha256"):
        errors.append(f"{owner} SHA-256 changed")
    return errors, path


def verify_local_tr_review_evidence(
    evidence_or_path: Mapping[str, Any] | str | Path,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Fail closed on formal claims, state drift, or artifact/hash changes."""

    evidence_path: Path | None = None
    if isinstance(evidence_or_path, Mapping):
        evidence = dict(evidence_or_path)
        if output_dir is None:
            raise ValueError("output_dir is required when verifying an in-memory value")
        output = Path(output_dir).resolve()
    else:
        evidence_path = Path(evidence_or_path).resolve()
        evidence = load_json(evidence_path)
        output = evidence_path.parent
        if output_dir is not None and Path(output_dir).resolve() != output:
            return ["explicit output_dir differs from evidence parent"]
    errors: list[str] = []
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        errors.append("unsupported local-TR review evidence schema")
    if evidence.get("status") != "review_only":
        errors.append("local-TR evidence status must be review_only")
    if evidence.get("review_only") is not True:
        errors.append("local-TR evidence must declare review_only=true")
    if evidence.get("formal_capture") is not False:
        errors.append("local-TR evidence must declare formal_capture=false")
    if evidence.get("qualification_claim") is not False:
        errors.append("local-TR evidence must declare qualification_claim=false")
    if evidence.get("formal_view_ids") != [] or evidence.get("formal_modalities") != []:
        errors.append("local-TR evidence must not claim formal views/modalities")
    if evidence.get("review_view_ids") != ["view0"]:
        errors.append("local-TR evidence must contain only review view0")
    if evidence.get("sensor_view_created") is not False:
        errors.append("local-TR review must not claim a newly created sensor view")
    declared_content_hash = evidence.get("evidence_content_sha256")
    unhashed = dict(evidence)
    unhashed.pop("evidence_content_sha256", None)
    if declared_content_hash != canonical_json_sha256(unhashed):
        errors.append("evidence content SHA-256 mismatch")

    inputs = evidence.get("inputs")
    if not isinstance(inputs, Mapping):
        errors.append("evidence lacks input file records")
    else:
        expected_inputs = {
            "visual_glb",
            "local_tr_actions",
            "local_tr_preserving_rebase_report",
            "schedule_request",
            "m1_room_manifest",
            "m1_camera_request",
        }
        if set(inputs) != expected_inputs:
            errors.append("evidence input role set changed")
        for role in sorted(expected_inputs & set(inputs)):
            record_errors, _ = _verify_file_record(
                inputs[role], root=None, owner=f"input {role}"
            )
            errors.extend(record_errors)

    runtime_assets = evidence.get("runtime_assets")
    if not isinstance(runtime_assets, Mapping):
        errors.append("evidence lacks runtime artifacts")
    else:
        expected_runtime = {
            "visual_copy",
            "habitat_urdf",
            "habitat_ao_config",
            "habitat_joint_mapping",
        }
        if set(runtime_assets) != expected_runtime:
            errors.append("runtime artifact role set changed")
        for role in sorted(expected_runtime & set(runtime_assets)):
            record_errors, _ = _verify_file_record(
                runtime_assets[role], root=output, owner=f"runtime artifact {role}"
            )
            errors.extend(record_errors)

    application = evidence.get("runtime_application")
    if not isinstance(application, Mapping):
        errors.append("evidence lacks runtime_application")
    else:
        if application.get("simulator_lifetime") != "fresh_per_capture_call":
            errors.append("Simulator lifetime is not fresh_per_capture_call")
        if application.get("state_evaluation") != "explicit_fixed_state":
            errors.append("runtime state evaluation is not explicit fixed state")
        if application.get("observation_calls_per_frame") != 1:
            errors.append("review did not make exactly one observation call per frame")
        if application.get("observation_call_count") != 75:
            errors.append("review observation call count is not 75")
        if application.get("physics_steps") != 0:
            errors.append("review declares physics steps")
        if application.get("initial_world_time_seconds") != application.get(
            "final_world_time_seconds"
        ):
            errors.append("Habitat world time advanced across the review")

    frames = evidence.get("frames")
    if not isinstance(frames, list) or len(frames) != 75:
        errors.append("evidence must contain exactly 75 frame records")
        frames = []
    for ordinal, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frame_index") != ordinal:
            errors.append(f"frame record {ordinal} is invalid")
            continue
        runtime_record = frame.get("runtime_application")
        if not isinstance(runtime_record, Mapping):
            errors.append(f"frame {ordinal} lacks runtime readback")
            continue
        if runtime_record.get("world_time_advance_seconds") != 0.0:
            errors.append(f"frame {ordinal} advanced world time")
        for key in (
            "maximum_root_readback_error",
            "maximum_prismatic_readback_error",
            "maximum_spherical_readback_error",
        ):
            value = runtime_record.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) > _JOINT_READBACK_ATOL
            ):
                errors.append(f"frame {ordinal} {key} exceeds tolerance")
        link_error = runtime_record.get("maximum_skin_link_matrix_readback_error")
        if (
            isinstance(link_error, bool)
            or not isinstance(link_error, (int, float))
            or not math.isfinite(float(link_error))
            or float(link_error) > _LINK_MATRIX_READBACK_ATOL
        ):
            errors.append(
                f"frame {ordinal} maximum_skin_link_matrix_readback_error "
                "exceeds tolerance"
            )
        visibility = frame.get("animal_semantic_visibility")
        if (
            not isinstance(visibility, Mapping)
            or visibility.get("visible") is not True
            or not isinstance(visibility.get("pixel_count"), int)
            or visibility.get("pixel_count", 0) <= 0
        ):
            errors.append(f"frame {ordinal} has no animal semantic visibility")

    from avengine.assets.habitat_capture import verify_saved_capture_arrays

    errors.extend(verify_saved_capture_arrays(evidence, output))
    review_media = evidence.get("review_media")
    if (
        not isinstance(review_media, Mapping)
        or review_media.get("review_only") is not True
        or review_media.get("qualification_claim") is not False
        or review_media.get("formal_view_ids") != []
    ):
        errors.append("review media claim is not strictly non-formal")
    else:
        videos = review_media.get("videos")
        if not isinstance(videos, Mapping) or set(videos) != set(FORMAL_MODALITIES):
            errors.append("review media video role set changed")
        else:
            for modality in FORMAL_MODALITIES:
                video = videos[modality]
                if (
                    not isinstance(video, Mapping)
                    or video.get("review_only") is not True
                    or video.get("qualification_claim") is not False
                    or video.get("view_id") != "view0"
                    or video.get("frame_count") != 75
                ):
                    errors.append(f"{modality} review video claim is invalid")
                    continue
                record_errors, _ = _verify_file_record(
                    video.get("artifact"),
                    root=output,
                    owner=f"{modality} review video",
                )
                errors.extend(record_errors)
    if evidence_path is not None and evidence_path.is_symlink():
        errors.append("evidence file is symlinked")
    return errors


__all__ = [
    "DEFAULT_SEMANTIC_ID",
    "EVIDENCE_SCHEMA",
    "LocalTRReviewError",
    "LocalTRReviewFrame",
    "REBASE_REPORT_SCHEMA",
    "actor_from_skin_root_from_rebase_report",
    "capture_local_tr_habitat_review",
    "compile_review_frames",
    "mixed_joint_readback_errors",
    "skin_link_matrix_readback_error",
    "validate_review_schedule",
    "verify_local_tr_review_evidence",
]

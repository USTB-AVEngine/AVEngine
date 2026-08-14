"""Habitat-native two-human MP3D capture authority and runtime.

The retained SPEAR suite is consumed only as a frozen timeline/state source.
It remains comparison evidence; this module never promotes its visual role.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json
from avengine.contracts.transforms import transform_error
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.habitat_capture import _resolved_scene, discover_runtime_root
from avengine.m2.habitat_capture import quaternion_xyzw_to_matrix
from avengine.m5_1.mixed_capture import (
    _JOINT_READBACK_ATOL,
    _LINK_MATRIX_READBACK_ATOL,
    _ROOT_READBACK_ATOL,
)
from avengine.m6x.rir_cache import RIRCacheError, validate_semantic_rir_job_plan
from avengine.runtime_profiles import (
    RuntimeProfileError,
    load_source_asset_runtime_registry,
    resolve_source_asset_runtime_profile,
)


FRAME_COUNT = 75
FRAME_RATE_HZ = 15
TIME_BASE_HZ = 48_000
TICKS_PER_FRAME = 3_200
DURATION_TICKS = 240_000
ACTOR_IDS = ("source1_actor", "source2_actor")
SOURCE_SLOTS = ("source1", "source2")
SEMANTIC_IDS = (62_000, 62_001)
PACKAGE_STEMS = ("human0", "human1")
COMPARISON_VISUAL_ROLE = "comparison_visual"


class TwoHumanCaptureError(RuntimeError):
    """The two-human request, state join, or Habitat capture is invalid."""


@dataclass(frozen=True)
class HumanActorAuthority:
    actor_id: str
    source_slot_id: str
    asset_id: str
    asset_revision: str
    source_glb: Path
    semantic_id: int
    package_stem: str
    walking_profile_sample_count: int | None
    emitter_offset_m: tuple[float, float, float]
    emitter_offset_space: str
    anatomical_forward_axis: tuple[float, float, float]
    anatomical_forward_source: str


@dataclass(frozen=True)
class PlannedHumanFrame:
    frame_index: int
    pts_ticks: int
    action_id: str
    action_time_ticks: int
    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class TwoHumanCaptureAuthority:
    episode_id: str
    room_id: str
    room_revision: str
    actors: tuple[HumanActorAuthority, HumanActorAuthority]
    actor_frames: tuple[tuple[PlannedHumanFrame, ...], tuple[PlannedHumanFrame, ...]]
    rig_frames: tuple[Mapping[str, Any], ...]
    resolution_hw: tuple[int, int]
    horizontal_fov_deg: float
    suite_visual_role: str
    qualification_claim: bool
    formal_dataset_count: int


@dataclass(frozen=True)
class _HumanFrameBinding:
    authority: HumanActorAuthority
    package: Any
    articulated_object: Any
    joint_binding: Any
    link_blocks: tuple[Any, ...]
    head_link_id: Any
    mouth_link_id: Any


@dataclass(frozen=True)
class _CapturedTwoHumanFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    semantic: np.ndarray
    actor_root_world_matrices: np.ndarray
    skin_root_world_matrices: np.ndarray
    anchor_positions_m: np.ndarray
    semantic_visibility_pixels: np.ndarray
    record: Mapping[str, Any]


def _action_sample_index(action: Any, action_time_ticks: int) -> int:
    """Resolve a Timeline tick only when it lands on an authored action sample."""

    _require(
        isinstance(action_time_ticks, int)
        and not isinstance(action_time_ticks, bool)
        and action_time_ticks >= 0,
        "action_time_ticks must be a nonnegative integer",
    )
    loop_duration = getattr(action, "loop_duration_ticks", None)
    sample_ticks = tuple(getattr(action, "sample_ticks", ()))
    _require(
        isinstance(loop_duration, int)
        and not isinstance(loop_duration, bool)
        and loop_duration > 0
        and sample_ticks
        and len(set(sample_ticks)) == len(sample_ticks)
        and all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and 0 <= item < loop_duration
            for item in sample_ticks
        ),
        "runtime action sample ticks are invalid",
    )
    requested_tick = action_time_ticks % loop_duration
    try:
        return sample_ticks.index(requested_tick)
    except ValueError as exc:
        raise TwoHumanCaptureError(
            f"action_time_ticks resolves to unauthored sample tick {requested_tick}"
        ) from exc


def _planned_actor_world_matrix(frame: PlannedHumanFrame) -> np.ndarray:
    """Use the frozen suite quaternion directly; do not derive trajectory heading."""

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_xyzw_to_matrix(frame.rotation_xyzw)
    result[:3, 3] = np.asarray(frame.translation_m, dtype=np.float64)
    return result


def _planned_emitter_world_position(
    actor: HumanActorAuthority, frame: PlannedHumanFrame
) -> np.ndarray:
    _require(
        actor.emitter_offset_space == "final_scaled_asset_root",
        f"{actor.actor_id} emitter offset space is unsupported",
    )
    local = np.asarray((*actor.emitter_offset_m, 1.0), dtype=np.float64)
    return (_planned_actor_world_matrix(frame) @ local)[:3]


def _validate_formal_capture_arrays(
    arrays: Mapping[str, Any], *, resolution_hw: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Normalize one co-located RGB/metric-depth/semantic observation."""

    _require(set(arrays) == {"rgb", "depth", "semantic"}, "formal modalities drift")
    rgb = np.asarray(arrays["rgb"])
    depth = np.asarray(arrays["depth"])
    semantic = np.asarray(arrays["semantic"])
    _require(
        rgb.ndim == 3
        and rgb.shape[-1] in {3, 4}
        and rgb.shape[:2] == resolution_hw
        and rgb.dtype == np.dtype(np.uint8),
        "RGB observation must be uint8 HxWx3/4 at the selected resolution",
    )
    _require(
        depth.shape == resolution_hw
        and np.issubdtype(depth.dtype, np.floating)
        and np.all(np.isfinite(depth)),
        "depth observation must be finite floating-point metric depth",
    )
    _require(
        semantic.shape == resolution_hw and np.issubdtype(semantic.dtype, np.integer),
        "semantic observation must be an integer image at the selected resolution",
    )
    return {
        "rgb": np.ascontiguousarray(rgb[..., :3]).copy(),
        "depth": np.ascontiguousarray(depth).copy(),
        "semantic": np.ascontiguousarray(semantic).copy(),
    }


def _semantic_absence_record(
    semantic: Any,
    *,
    resolution_hw: tuple[int, int],
    semantic_ids: Mapping[str, int],
) -> dict[str, Any]:
    image = np.asarray(semantic)
    _require(
        image.shape == resolution_hw and np.issubdtype(image.dtype, np.integer),
        "preflight semantic observation differs from the selected sensor",
    )
    counts = {
        actor_id: int(np.count_nonzero(image == semantic_id))
        for actor_id, semantic_id in semantic_ids.items()
    }
    _require(
        not any(counts.values()),
        "two-human semantic IDs collide with the no-actor MP3D observation",
    )
    return {
        "observation_calls": 1,
        "shape": list(image.shape),
        "dtype": image.dtype.str,
        "semantic_ids": dict(semantic_ids),
        "pixel_counts": counts,
        "all_absent": True,
    }


def _prepare_fresh_output(output_dir: str | Path) -> Path:
    requested = Path(output_dir).expanduser()
    _require(
        not requested.exists() and not requested.is_symlink(),
        f"refusing to replace capture output: {requested}",
    )
    output = requested.resolve()
    _require(
        not output.exists() and not output.is_symlink(),
        f"refusing to replace capture output: {output}",
    )
    output.mkdir(parents=True, exist_ok=False)
    return output


def _save_plain_array(output: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = output / "arrays" / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(value)
    np.save(path, array, allow_pickle=False)
    readback = np.load(path, mmap_mode="r", allow_pickle=False)
    _require(
        readback.dtype == array.dtype
        and readback.shape == array.shape
        and np.array_equal(readback, array),
        f"saved {name} array differs on readback",
    )
    return {
        "path": path.relative_to(output).as_posix(),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "readback_verified": True,
    }


def _camera_readback_record(
    snapshot: Mapping[str, Any],
    *,
    planned_world_from_rig: Mapping[str, Any],
    sensor_uuids: Sequence[str],
) -> dict[str, Any]:
    agent_pose = _mapping(snapshot.get("agent"), owner="camera agent readback")
    sensor_poses = _mapping(snapshot.get("sensors"), owner="camera sensor readback")
    _require(
        set(sensor_poses) == set(sensor_uuids),
        "camera readback does not contain every selected sensor and listener",
    )
    errors = {
        "agent": float(transform_error(planned_world_from_rig, agent_pose)),
        **{
            sensor_uuid: float(
                transform_error(
                    planned_world_from_rig,
                    _mapping(sensor_poses[sensor_uuid], owner=f"sensor {sensor_uuid}"),
                )
            )
            for sensor_uuid in sensor_uuids
        },
    }
    maximum_error = max(errors.values())
    _require(
        maximum_error <= _ROOT_READBACK_ATOL,
        "camera/listener planned-live transform readback failed",
    )
    return {
        "planned_world_from_rig": {
            "translation_m": list(planned_world_from_rig["translation_m"]),
            "rotation_xyzw": list(planned_world_from_rig["rotation_xyzw"]),
        },
        "live_agent": dict(agent_pose),
        "live_sensors": {
            sensor_uuid: dict(sensor_poses[sensor_uuid]) for sensor_uuid in sensor_uuids
        },
        "transform_errors": errors,
        "maximum_transform_error": maximum_error,
    }


def _capture_two_human_frame(
    *,
    authority: TwoHumanCaptureAuthority,
    frame_index: int,
    simulator: Any,
    runtimes: Sequence[_HumanFrameBinding],
    modality_to_uuid: Mapping[str, str],
    sensor_wrappers: Sequence[Any],
    camera_sensor_uuids: Sequence[str],
    camera_snapshot: Callable[[], Mapping[str, Any]],
    apply_root: Callable[[Any, np.ndarray], None],
    runtime_snapshot: Callable[[Any, Any], Mapping[str, Any]],
    joint_readback_errors: Callable[[Any, Any, Sequence[Any]], tuple[float, float]],
    fk_readback_error: Callable[..., float],
    node_world_position: Callable[[Any, Any], np.ndarray],
    observation_validator: Callable[
        [Mapping[str, Any], Mapping[str, str]], Mapping[str, Any]
    ],
) -> _CapturedTwoHumanFrame:
    """Apply both humans, read back both, and issue exactly one formal render."""

    _require(0 <= frame_index < FRAME_COUNT, "capture frame index is invalid")
    _require(
        len(runtimes) == 2
        and tuple(item.authority for item in runtimes) == authority.actors,
        "capture runtimes must match both authority actors in order",
    )
    _require(
        tuple(modality_to_uuid) == ("rgb", "depth", "semantic")
        and len(sensor_wrappers) == 3,
        "formal capture requires one ordered RGB/depth/semantic sensor set",
    )
    initial_world_time = float(simulator.get_world_time())
    prepared: list[dict[str, Any]] = []
    for actor_index, runtime in enumerate(runtimes):
        planned = authority.actor_frames[actor_index][frame_index]
        action = runtime.package.actions.action(planned.action_id)
        sample_index = _action_sample_index(action, planned.action_time_ticks)
        translations = np.asarray(action.translations_m[sample_index], dtype=np.float64)
        rotations = np.asarray(action.rotations_xyzw[sample_index], dtype=np.float64)
        joints = np.asarray(
            runtime.joint_binding.map_pose(translations, rotations), dtype=np.float64
        ).reshape(-1)
        actor_world = _planned_actor_world_matrix(planned)
        actor_from_skin = np.asarray(
            runtime.package.actor_from_skin_root, dtype=np.float64
        )
        _require(
            actor_from_skin.shape == (4, 4)
            and np.all(np.isfinite(actor_from_skin))
            and np.all(np.isfinite(joints)),
            f"{runtime.authority.actor_id} runtime pose is invalid",
        )
        skin_world = actor_world @ actor_from_skin
        apply_root(runtime.articulated_object, skin_world)
        runtime.articulated_object.joint_positions = joints.copy()
        prepared.append(
            {
                "runtime": runtime,
                "planned": planned,
                "action": action,
                "sample_index": sample_index,
                "translations": translations,
                "rotations": rotations,
                "joints": joints,
                "actor_world": actor_world,
                "actor_from_skin": actor_from_skin,
                "skin_world": skin_world,
            }
        )

    before: list[Mapping[str, Any]] = []
    actor_roots: list[np.ndarray] = []
    skin_roots: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    actor_records: list[dict[str, Any]] = []
    for item in prepared:
        runtime = item["runtime"]
        snapshot = runtime_snapshot(simulator, runtime.articulated_object)
        actual_skin = np.asarray(snapshot["world_from_skin_root"], dtype=np.float64)
        actual_joints = np.asarray(
            snapshot["mixed_joint_positions"], dtype=np.float64
        ).reshape(-1)
        _require(
            actual_skin.shape == (4, 4)
            and np.all(np.isfinite(actual_skin))
            and np.all(np.isfinite(actual_joints)),
            f"{runtime.authority.actor_id} runtime readback is invalid",
        )
        actual_actor = actual_skin @ np.linalg.inv(item["actor_from_skin"])
        skin_error = float(np.max(np.abs(actual_skin - item["skin_world"])))
        actor_error = float(np.max(np.abs(actual_actor - item["actor_world"])))
        prismatic_error, spherical_error = joint_readback_errors(
            actual_joints, item["joints"], runtime.link_blocks
        )
        fk_error = float(
            fk_readback_error(
                runtime.articulated_object,
                runtime.package,
                world_from_skin_root=item["skin_world"],
                translations_m=item["translations"],
                rotations_xyzw=item["rotations"],
            )
        )
        _require(
            max(skin_error, actor_error) <= _ROOT_READBACK_ATOL
            and prismatic_error <= _JOINT_READBACK_ATOL
            and spherical_error <= _JOINT_READBACK_ATOL
            and fk_error <= _LINK_MATRIX_READBACK_ATOL,
            f"{runtime.authority.actor_id} articulated readback failed",
        )
        head = np.asarray(
            node_world_position(runtime.articulated_object, runtime.head_link_id),
            dtype=np.float64,
        )
        mouth = np.asarray(
            node_world_position(runtime.articulated_object, runtime.mouth_link_id),
            dtype=np.float64,
        )
        _require(
            head.shape == mouth.shape == (3,)
            and np.all(np.isfinite(head))
            and np.all(np.isfinite(mouth)),
            f"{runtime.authority.actor_id} head/mouth readback is invalid",
        )
        planned_emitter = _planned_emitter_world_position(
            runtime.authority, item["planned"]
        )
        mouth_delta = mouth - planned_emitter
        before.append(
            {
                "world_from_skin_root": actual_skin.copy(),
                "mixed_joint_positions": actual_joints.copy(),
            }
        )
        actor_roots.append(actual_actor)
        skin_roots.append(actual_skin)
        anchors.append(np.stack((head, mouth)))
        actor_records.append(
            {
                "actor_id": runtime.authority.actor_id,
                "package_stem": runtime.authority.package_stem,
                "action_id": item["planned"].action_id,
                "action_time_ticks": item["planned"].action_time_ticks,
                "action_sample_index": item["sample_index"],
                "planned_actor_world_matrix": item["actor_world"].tolist(),
                "live_actor_world_matrix": actual_actor.tolist(),
                "live_skin_root_world_matrix": actual_skin.tolist(),
                "head_link_origin_m": head.tolist(),
                "planned_emitter_world_position_m": planned_emitter.tolist(),
                "planned_emitter_authority": (
                    "final_scaled_asset_root offset joined to trajectory/RIR source"
                ),
                "live_mouth_link_origin_diagnostic_m": mouth.tolist(),
                "live_mouth_minus_planned_emitter_diagnostic_m": mouth_delta.tolist(),
                "live_mouth_is_authoritative": False,
                "readback_errors": {
                    "actor_root": actor_error,
                    "skin_root": skin_error,
                    "joint_prismatic": float(prismatic_error),
                    "joint_spherical": float(spherical_error),
                    "skin_link_fk": fk_error,
                },
            }
        )

    rig_frame = authority.rig_frames[frame_index]
    _require(
        rig_frame.get("frame_index") == frame_index
        and rig_frame.get("pts_ticks") == frame_index * TICKS_PER_FRAME,
        "capture rig frame differs from Timeline",
    )
    planned_rig = _mapping(rig_frame.get("world_from_rig"), owner="planned rig")
    camera_before = _camera_readback_record(
        camera_snapshot(),
        planned_world_from_rig=planned_rig,
        sensor_uuids=camera_sensor_uuids,
    )

    observation = simulator.render_sensors(list(sensor_wrappers))
    arrays = _validate_formal_capture_arrays(
        observation_validator(observation, modality_to_uuid),
        resolution_hw=authority.resolution_hw,
    )
    visibility = np.asarray(
        [
            np.count_nonzero(arrays["semantic"] == runtime.authority.semantic_id)
            for runtime in runtimes
        ],
        dtype=np.int64,
    )
    _require(
        bool(np.all(visibility > 0)),
        "formal semantic frame must contain both human semantic IDs",
    )

    for item, retained in zip(prepared, before, strict=True):
        runtime = item["runtime"]
        after = runtime_snapshot(simulator, runtime.articulated_object)
        root_error = float(
            np.max(
                np.abs(
                    np.asarray(after["world_from_skin_root"], dtype=np.float64)
                    - np.asarray(retained["world_from_skin_root"], dtype=np.float64)
                )
            )
        )
        prismatic_error, spherical_error = joint_readback_errors(
            after["mixed_joint_positions"],
            retained["mixed_joint_positions"],
            runtime.link_blocks,
        )
        _require(
            root_error <= _ROOT_READBACK_ATOL
            and prismatic_error <= _JOINT_READBACK_ATOL
            and spherical_error <= _JOINT_READBACK_ATOL,
            f"frame {frame_index} render changed {runtime.authority.actor_id} state",
        )
    camera_after = _camera_readback_record(
        camera_snapshot(),
        planned_world_from_rig=planned_rig,
        sensor_uuids=camera_sensor_uuids,
    )
    final_world_time = float(simulator.get_world_time())
    _require(
        final_world_time == initial_world_time,
        f"frame {frame_index} advanced Habitat world time",
    )
    return _CapturedTwoHumanFrame(
        rgb=arrays["rgb"],
        depth_m=arrays["depth"],
        semantic=arrays["semantic"],
        actor_root_world_matrices=np.stack(actor_roots),
        skin_root_world_matrices=np.stack(skin_roots),
        anchor_positions_m=np.stack(anchors),
        semantic_visibility_pixels=visibility,
        record={
            "frame_index": frame_index,
            "pts_ticks": frame_index * TICKS_PER_FRAME,
            "physics_steps": 0,
            "formal_render_calls": 1,
            "world_time_seconds_before": initial_world_time,
            "world_time_seconds_after": final_world_time,
            "actors": actor_records,
            "camera": {"before": camera_before, "after": camera_after},
        },
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TwoHumanCaptureError(message)


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{owner} must be an object")
    return value


def _sequence(value: Any, *, owner: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{owner} must be an array",
    )
    return value


def _finite_vector(value: Any, length: int, *, owner: str) -> tuple[float, ...]:
    items = _sequence(value, owner=owner)
    _require(len(items) == length, f"{owner} must contain {length} numbers")
    result: list[float] = []
    for item in items:
        _require(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item)),
            f"{owner} must contain finite numbers",
        )
        result.append(float(item))
    return tuple(result)


def _vec3(value: Any, *, owner: str) -> tuple[float, float, float]:
    return _finite_vector(value, 3, owner=owner)  # type: ignore[return-value]


def _quat(value: Any, *, owner: str) -> tuple[float, float, float, float]:
    result = _finite_vector(value, 4, owner=owner)
    norm = math.sqrt(sum(item * item for item in result))
    _require(math.isclose(norm, 1.0, abs_tol=1.0e-7), f"{owner} must be unit")
    return result  # type: ignore[return-value]


def _transform(value: Any, *, owner: str) -> dict[str, list[float]]:
    transform = _mapping(value, owner=owner)
    return {
        "translation_m": list(
            _vec3(transform.get("translation_m"), owner=f"{owner}.translation_m")
        ),
        "rotation_xyzw": list(
            _quat(transform.get("rotation_xyzw"), owner=f"{owner}.rotation_xyzw")
        ),
    }


def _same_transform(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return bool(
        np.allclose(
            first["translation_m"], second["translation_m"], rtol=0.0, atol=1.0e-9
        )
        and np.allclose(
            first["rotation_xyzw"], second["rotation_xyzw"], rtol=0.0, atol=1.0e-9
        )
    )


def _identity_transform(value: Any, *, owner: str) -> bool:
    return _same_transform(
        _transform(value, owner=owner),
        {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )


def _safe_regular_path(value: Any, *, owner: str) -> Path:
    raw = Path(str(value))
    _require(raw.is_absolute(), f"{owner} must be absolute")
    candidates = (raw, *raw.parents)
    _require(
        not any(candidate.is_symlink() for candidate in candidates),
        f"{owner} cannot contain symlinks",
    )
    _require(raw.is_file(), f"{owner} must be a regular file")
    return raw.resolve(strict=True)


def _resolved_regular_path(value: Any, *, owner: str) -> Path:
    """Resolve a controlled runtime alias to its regular-file target."""

    raw = Path(str(value))
    _require(raw.is_absolute(), f"{owner} must be absolute")
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TwoHumanCaptureError(f"{owner} must resolve to a regular file") from exc
    _require(resolved.is_file(), f"{owner} must resolve to a regular file")
    return resolved


def _require_same_runtime_file(declared: Any, expected: Any, *, owner: str) -> None:
    declared_path = _resolved_regular_path(declared, owner=f"atom {owner}")
    expected_path = _resolved_regular_path(expected, owner=f"resolved M1 {owner}")
    _require(
        declared_path.samefile(expected_path),
        f"atom {owner} differs from resolved M1 runtime",
    )


def _validate_camera_runtime_navigation(
    camera_runtime: Mapping[str, Any], navigation: Mapping[str, Any]
) -> None:
    atom_height = float(camera_runtime.get("agent_height_m", math.nan))
    atom_radius = float(camera_runtime.get("agent_radius_m", math.nan))
    m1_height = float(navigation.get("agent_height_m", math.nan))
    m1_radius = float(navigation.get("agent_radius_m", math.nan))
    # The atom radius belongs to the camera-candidate clearance solver.  The
    # M1 radius configures the render agent.  They are independent positive
    # parameters; only their common physical height is expected to agree.
    _require(
        all(
            math.isfinite(value) and value > 0.0
            for value in (atom_height, atom_radius, m1_height, m1_radius)
        )
        and atom_height == m1_height,
        "atom solver and M1 render-agent navigation values are invalid",
    )


def _build_actor_authorities(
    atom: Mapping[str, Any], plan: Mapping[str, Any], registry: Mapping[str, Any]
) -> tuple[HumanActorAuthority, HumanActorAuthority]:
    declarations = _sequence(plan.get("actors"), owner="suite plan actors")
    _require(
        [
            item.get("actor_id") if isinstance(item, Mapping) else None
            for item in declarations
        ]
        == list(ACTOR_IDS),
        "suite actors must be source1_actor then source2_actor",
    )
    framing = _mapping(atom.get("actor_framing"), owner="atom actor_framing")
    bindings = _sequence(framing.get("actor_bindings"), owner="actor bindings")
    by_id = {
        item.get("actor_id"): item for item in bindings if isinstance(item, Mapping)
    }
    _require(
        set(by_id) == set(ACTOR_IDS) and len(bindings) == 2,
        "atom must bind exactly both human actors",
    )
    result: list[HumanActorAuthority] = []
    for index, (actor_id, slot, semantic_id, package_stem) in enumerate(
        zip(ACTOR_IDS, SOURCE_SLOTS, SEMANTIC_IDS, PACKAGE_STEMS, strict=True)
    ):
        declaration = _mapping(declarations[index], owner=f"{actor_id} declaration")
        binding = _mapping(by_id[actor_id], owner=f"{actor_id} binding")
        asset_id = str(declaration.get("asset_id", ""))
        revision = str(declaration.get("asset_revision", ""))
        _require(
            asset_id
            and revision
            and binding.get("asset_id") == asset_id
            and binding.get("asset_revision") == revision,
            f"{actor_id} asset identity differs across atom and suite",
        )
        try:
            profile = resolve_source_asset_runtime_profile(registry, asset_id, revision)
        except RuntimeProfileError as exc:
            raise TwoHumanCaptureError(str(exc)) from exc
        identity = _mapping(profile.get("identity"), owner=f"{actor_id} identity")
        _require(
            profile.get("entity_class") == "articulated_human"
            and identity.get("species_id") == "human",
            f"{actor_id} is not an articulated human",
        )
        timeline = _mapping(profile.get("timeline"), owner=f"{actor_id} timeline")
        period = timeline.get("walk_phase_period_frames")
        _require(period in {16, 19}, f"{actor_id} walking period must be 16 or 19")
        _require(
            timeline.get("body_plan_id")
            == declaration.get("body_plan_id")
            == "biped_human"
            and timeline.get("template_id") == declaration.get("template_id")
            and timeline.get("idle_action_id") == "idle"
            and timeline.get("walking_action_id") == "walk"
            and binding.get("skin_index") == 0,
            f"{actor_id} human skin/body/timeline binding drift",
        )
        _require(
            binding.get("action_name_by_action_id")
            == {"idle": "Standing_Idle", "walk": "Walking"},
            f"{actor_id} action mapping drift",
        )
        anchors = _sequence(
            profile.get("emitter_anchors"), owner=f"{actor_id} emitter anchors"
        )
        anchor = [
            item
            for item in anchors
            if isinstance(item, Mapping)
            and item.get("anchor_id") == profile.get("default_emitter_anchor_id")
        ]
        _require(len(anchor) == 1, f"{actor_id} default emitter anchor is not unique")
        emitter = _vec3(anchor[0].get("offset_m"), owner=f"{actor_id} emitter offset")
        _require(
            tuple(declaration.get("emitter_offset_m", ())) == emitter
            and anchor[0].get("offset_space") == "final_scaled_asset_root",
            f"{actor_id} suite emitter differs from runtime profile",
        )
        source_glb = _safe_regular_path(
            binding.get("source_asset_path"), owner=f"{actor_id} source GLB"
        )
        axis = _vec3(
            timeline.get("local_anatomical_forward_axis"),
            owner=f"{actor_id} forward axis",
        )
        _require(
            axis == (0.0, 0.0, 1.0), f"{actor_id} anatomical forward must remain +Z"
        )
        result.append(
            HumanActorAuthority(
                actor_id=actor_id,
                source_slot_id=slot,
                asset_id=asset_id,
                asset_revision=revision,
                source_glb=source_glb,
                semantic_id=semantic_id,
                package_stem=package_stem,
                walking_profile_sample_count=19 if period == 19 else None,
                emitter_offset_m=emitter,
                emitter_offset_space="final_scaled_asset_root",
                anatomical_forward_axis=axis,
                anatomical_forward_source=(
                    f"runtime_profile:{registry.get('registry_id')}/{asset_id}@{revision}"
                    "/timeline.local_anatomical_forward_axis"
                ),
            )
        )
    _require(
        result[0].semantic_id != result[1].semantic_id
        and result[0].asset_id != result[1].asset_id
        and result[0].source_glb != result[1].source_glb,
        "two humans must have distinct semantics, assets, and source paths",
    )
    return result[0], result[1]


def _build_actor_frames(
    plan: Mapping[str, Any],
) -> tuple[tuple[PlannedHumanFrame, ...], tuple[PlannedHumanFrame, ...]]:
    frames = _sequence(plan.get("frames"), owner="suite plan frames")
    _require(len(frames) == FRAME_COUNT, "suite plan must contain exactly 75 frames")
    collected: list[list[PlannedHumanFrame]] = [[], []]
    for ordinal, raw_frame in enumerate(frames):
        frame = _mapping(raw_frame, owner=f"suite frame {ordinal}")
        _require(
            frame.get("frame_index") == ordinal
            and frame.get("pts_ticks") == ordinal * TICKS_PER_FRAME,
            f"suite frame {ordinal} Timeline drift",
        )
        states = _sequence(
            frame.get("actor_states"), owner=f"suite frame {ordinal} actors"
        )
        _require(
            [
                item.get("actor_id") if isinstance(item, Mapping) else None
                for item in states
            ]
            == list(ACTOR_IDS),
            f"suite frame {ordinal} actor order drift",
        )
        for actor_index, raw_state in enumerate(states):
            state = _mapping(
                raw_state, owner=f"suite frame {ordinal} actor {actor_index}"
            )
            action_id = state.get("action_id")
            action_ticks = state.get("action_time_ticks")
            action_phase = state.get("action_phase")
            _require(
                action_id in {"idle", "walk"}, f"frame {ordinal} action is invalid"
            )
            _require(
                state.get("frame_index") == ordinal
                and state.get("asset_id") == plan["actors"][actor_index].get("asset_id")
                and not isinstance(action_phase, bool)
                and isinstance(action_phase, (int, float))
                and math.isfinite(float(action_phase))
                and 0.0 <= float(action_phase) < 1.0
                and isinstance(action_ticks, int)
                and not isinstance(action_ticks, bool)
                and action_ticks >= 0
                and action_ticks % TICKS_PER_FRAME == 0,
                f"frame {ordinal} action_time_ticks is off the 15 Hz grid",
            )
            _require(
                action_id == "idle"
                and float(action_phase) == 0.0
                and action_ticks == ordinal * TICKS_PER_FRAME,
                f"strict static actor action drift at frame {ordinal}",
            )
            collected[actor_index].append(
                PlannedHumanFrame(
                    frame_index=ordinal,
                    pts_ticks=ordinal * TICKS_PER_FRAME,
                    action_id=str(action_id),
                    action_time_ticks=action_ticks,
                    translation_m=_vec3(
                        state.get("translation_m"), owner=f"frame {ordinal} translation"
                    ),
                    rotation_xyzw=_quat(
                        state.get("rotation_xyzw"), owner=f"frame {ordinal} rotation"
                    ),
                )
            )
    for actor_index, actor_frames in enumerate(collected):
        first = actor_frames[0]
        _require(
            all(
                frame.translation_m == first.translation_m
                and frame.rotation_xyzw == first.rotation_xyzw
                for frame in actor_frames
            ),
            f"{ACTOR_IDS[actor_index]} strict static root/rotation must remain frozen",
        )
    return tuple(collected[0]), tuple(collected[1])


def validate_two_human_authority_documents(
    *,
    atom: Mapping[str, Any],
    suite: Mapping[str, Any],
    sensor_rig: Mapping[str, Any],
    trajectory_bank: Mapping[str, Any],
    rir_plan: Mapping[str, Any],
    runtime_profiles: Mapping[str, Any],
    room: Mapping[str, Any],
    m1_request: Mapping[str, Any],
) -> TwoHumanCaptureAuthority:
    """Join all pre-existing authorities without promoting UE visual evidence."""

    try:
        normalized_rir_jobs = validate_semantic_rir_job_plan(rir_plan)
    except RIRCacheError as exc:
        raise TwoHumanCaptureError(str(exc)) from exc

    _require(
        atom.get("schema")
        == "avengine_native_strict_two_human_mp3d_room_atom_request_v2",
        "atom request schema drift",
    )
    _require(
        atom.get("qualification_claim") is False
        and atom.get("formal_dataset_count") == 0,
        "atom must remain non-formal",
    )
    episode_id = str(atom.get("episode_id", ""))
    _require(bool(episode_id), "atom episode_id is missing")
    scenarios = _sequence(suite.get("scenarios"), owner="suite scenarios")
    _require(len(scenarios) == 1, "suite must contain one scenario")
    scenario = _mapping(scenarios[0], owner="suite scenario")
    plan = _mapping(scenario.get("plan"), owner="suite plan")
    _require(
        suite.get("schema") == "avengine_optional_spear_imported_glb_suite_v1"
        and scenario.get("schema") == "avengine_optional_spear_imported_glb_scenario_v1"
        and plan.get("schema") == "avengine_optional_spear_visual_plan_v1"
        and suite.get("backend_role")
        == scenario.get("backend_role")
        == plan.get("backend_role")
        == COMPARISON_VISUAL_ROLE,
        "retained UE suite schemas/roles must remain exact comparison evidence",
    )
    scenario_render = _mapping(scenario.get("render"), owner="scenario render")
    plan_render = _mapping(plan.get("render"), owner="plan render")
    qualification = _mapping(plan.get("qualification"), owner="plan qualification")
    _require(
        suite.get("qualification_claim") is False
        and suite.get("formal_dataset_count") == 0
        and scenario_render.get("frame_count") == FRAME_COUNT
        and scenario_render.get("frame_rate_hz") == FRAME_RATE_HZ
        and plan_render
        == {
            "fps_den": 1,
            "fps_num": FRAME_RATE_HZ,
            "frame_count": FRAME_COUNT,
            "ticks_per_frame": TICKS_PER_FRAME,
        }
        and qualification.get("qualification_claim") is False
        and qualification.get("formal_dataset_count") == 0,
        "suite/scenario/plan Timeline or non-formal boundary drift",
    )
    _require(
        scenario.get("scenario_id") == episode_id, "suite and atom episode IDs differ"
    )
    atom_room = _mapping(atom.get("room"), owner="atom room")
    plan_room = _mapping(plan.get("room"), owner="suite plan room")
    _require(
        plan_room.get("room_id") == atom_room.get("room_id")
        and plan_room.get("room_revision") == atom_room.get("room_revision")
        and plan_room.get("scene_id") == atom_room.get("scene_id"),
        "suite room/revision/scene differs from atom",
    )
    actors = _build_actor_authorities(atom, plan, runtime_profiles)
    actor_frames = _build_actor_frames(plan)

    rig_frames = _sequence(sensor_rig.get("frames"), owner="sensor rig frames")
    _require(
        sensor_rig.get("schema") == "avengine_sensor_rig_trajectory_v1"
        and sensor_rig.get("trajectory_id")
        == plan.get("camera", {}).get("sensor_rig_trajectory_id")
        and sensor_rig.get("rig_id") == "camera_rig_0"
        and sensor_rig.get("listener_id")
        == plan.get("camera", {}).get("listener_id")
        == "listener0"
        and sensor_rig.get("formal_view_id") == "view0"
        and sensor_rig.get("camera_listener_coupling") == "rigid_colocated_cooriented"
        and sensor_rig.get("time_base_hz") == TIME_BASE_HZ
        and sensor_rig.get("ticks_per_frame") == TICKS_PER_FRAME
        and sensor_rig.get("coordinate_frame") == "avengine_world_right_handed_y_up_m"
        and sensor_rig.get("pose_model") == "yaw_only_about_world_positive_y"
        and sensor_rig.get("frame_count") == FRAME_COUNT
        and sensor_rig.get("frame_rate_hz") == FRAME_RATE_HZ
        and sensor_rig.get("duration_ticks") == DURATION_TICKS
        and len(rig_frames) == FRAME_COUNT,
        "sensor rig identity or Timeline drift",
    )
    _require(
        _identity_transform(sensor_rig.get("rig_from_camera"), owner="rig_from_camera")
        and _identity_transform(
            sensor_rig.get("rig_from_listener"), owner="rig_from_listener"
        ),
        "sensor rig camera/listener offsets must remain identity",
    )
    normalized_rig: list[Mapping[str, Any]] = []
    for ordinal, raw_rig in enumerate(rig_frames):
        rig_frame = _mapping(raw_rig, owner=f"rig frame {ordinal}")
        expected_transform = _transform(
            rig_frame.get("world_from_rig"), owner=f"rig frame {ordinal}"
        )
        plan_frame = _mapping(plan["frames"][ordinal], owner=f"plan frame {ordinal}")
        camera_state = _mapping(
            plan_frame.get("camera_state"), owner=f"camera state {ordinal}"
        )
        _require(
            rig_frame.get("frame_index") == ordinal
            and rig_frame.get("pts_ticks") == ordinal * TICKS_PER_FRAME
            and camera_state.get("frame_index") == ordinal
            and camera_state.get("pts_ticks") == ordinal * TICKS_PER_FRAME
            and _same_transform(
                expected_transform,
                _transform(
                    camera_state.get("world_from_rig"), owner=f"camera state {ordinal}"
                ),
            ),
            f"camera and sensor rig differ at frame {ordinal}",
        )
        normalized_rig.append({**rig_frame, "world_from_rig": expected_transform})
    first_rig = normalized_rig[0]["world_from_rig"]
    _require(
        all(
            _same_transform(first_rig, item["world_from_rig"])
            for item in normalized_rig
        ),
        "current two-human capture requires the selected HOLD rig",
    )
    program = _mapping(sensor_rig.get("program"), owner="sensor rig program")
    yaw_degrees = float(program.get("yaw_deg", math.nan))
    half_yaw = math.radians(yaw_degrees) / 2.0
    _require(
        program.get("kind") == "HOLD"
        and _vec3(program.get("position_m"), owner="HOLD position")
        == tuple(first_rig["translation_m"])
        and np.allclose(
            first_rig["rotation_xyzw"],
            [0.0, math.sin(half_yaw), 0.0, math.cos(half_yaw)],
            rtol=0.0,
            atol=1.0e-9,
        ),
        "sensor rig HOLD program differs from its 75 poses",
    )

    episodes = _sequence(trajectory_bank.get("episodes"), owner="trajectory episodes")
    _require(
        trajectory_bank.get("schema") == "avengine_room_trajectory_bank_v2"
        and trajectory_bank.get("frame_count") == FRAME_COUNT
        and trajectory_bank.get("frame_rate_hz") == FRAME_RATE_HZ
        and trajectory_bank.get("seconds_per_episode") == 5.0
        and trajectory_bank.get("source_slots") == list(SOURCE_SLOTS)
        and len(episodes) == 1
        and isinstance(episodes[0], Mapping)
        and episodes[0].get("episode_id") == episode_id,
        "trajectory bank identity or Timeline drift",
    )
    episode = episodes[0]
    _require(
        episode.get("motion_case") == "strict_two_human_static_mp3d",
        "trajectory motion case drift",
    )
    roots = _mapping(episode.get("source_root_paths_m"), owner="trajectory roots")
    centers = _mapping(episode.get("source_center_paths_m"), owner="trajectory centers")
    expected_centers: dict[str, list[list[float]]] = {}
    for index, actor in enumerate(actors):
        expected_roots = [list(frame.translation_m) for frame in actor_frames[index]]
        expected_centers[actor.source_slot_id] = [
            _planned_emitter_world_position(actor, frame).tolist()
            for frame in actor_frames[index]
        ]
        _require(
            roots.get(actor.source_slot_id) == expected_roots,
            f"{actor.source_slot_id} trajectory roots differ from suite",
        )
        _require(
            centers.get(actor.source_slot_id) == expected_centers[actor.source_slot_id],
            f"{actor.source_slot_id} trajectory centers differ from suite",
        )

    jobs = normalized_rir_jobs
    _require(
        rir_plan.get("schema") == "avengine_room_rir_job_plan_v2"
        and rir_plan.get("status") == "planned_not_run"
        and rir_plan.get("producer_backend") == "RLR Audio Propagation"
        and rir_plan.get("listener_pose_mode") == "per_episode_frame"
        and rir_plan.get("dry_audio_independent") is True
        and rir_plan.get("slot_identity_affects_cache_key") is False
        and rir_plan.get("cache_key_fields")
        == [
            "source_position_m",
            "listener_position_m",
            "listener_orientation_wxyz",
        ]
        and rir_plan.get("stride_frames") == 1
        and rir_plan.get("requested_pair_state_count") == 2 * FRAME_COUNT
        and rir_plan.get("unique_rir_job_count") == 2
        and rir_plan.get("cache_reuse_count") == 148
        and rir_plan.get("unique_listener_pose_count") == 1,
        "RIR plan Timeline drift",
    )
    _require(len(jobs) == 2, "static two-human RIR plan must contain exactly two jobs")
    use_keys: list[tuple[str, int]] = []
    for raw_job in jobs:
        job = _mapping(raw_job, owner="RIR job")
        listener_position = _vec3(
            job.get("listener_position_m"), owner="RIR listener position"
        )
        listener_rotation = _finite_vector(
            job.get("listener_orientation_wxyz"), 4, owner="RIR listener orientation"
        )
        for raw_use in _sequence(job.get("uses"), owner="RIR uses"):
            use = _mapping(raw_use, owner="RIR use")
            slot = str(use.get("source_slot_id", ""))
            frame_index = use.get("frame_index")
            _require(
                slot in SOURCE_SLOTS
                and isinstance(frame_index, int)
                and 0 <= frame_index < FRAME_COUNT,
                "RIR use identity is invalid",
            )
            rig_transform = normalized_rig[frame_index]["world_from_rig"]
            xyzw = rig_transform["rotation_xyzw"]
            expected_wxyz = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
            _require(
                use.get("episode_id") == episode_id
                and listener_position == tuple(rig_transform["translation_m"])
                and np.allclose(listener_rotation, expected_wxyz, rtol=0.0, atol=1.0e-9)
                and tuple(job.get("source_position_m", ()))
                == tuple(expected_centers[slot][frame_index]),
                f"RIR state differs from suite/rig at {slot} frame {frame_index}",
            )
            use_keys.append((slot, frame_index))
    expected_use_keys = [
        (slot, frame) for slot in SOURCE_SLOTS for frame in range(FRAME_COUNT)
    ]
    _require(
        len(use_keys) == 2 * FRAME_COUNT
        and len(set(use_keys)) == len(use_keys)
        and sorted(use_keys) == sorted(expected_use_keys)
        and all(
            sum(slot == owner for slot, _ in use_keys) == FRAME_COUNT
            for owner in SOURCE_SLOTS
        ),
        "RIR uses must uniquely cover exactly 75 frames for each source",
    )

    room_id = str(atom_room.get("room_id", ""))
    room_revision = str(atom_room.get("room_revision", ""))
    scene_id = str(atom_room.get("scene_id", ""))
    room_scene = _mapping(room.get("scene"), owner="M1 room scene")
    coordinate_system = _mapping(
        room.get("coordinate_system"), owner="M1 room coordinate system"
    )
    _require(
        room.get("room_id") == room_id
        and room.get("room_kind") == "habitat_native"
        and scene_id
        and Path(str(room_scene.get("scene_id", ""))).stem == scene_id,
        "M1 room/scene differs from atom MP3D room",
    )
    _require(
        room.get("geometry_representation") == "real_surface_mesh"
        and coordinate_system.get("handedness") == "right"
        and coordinate_system.get("up_axis") == "+Y"
        and coordinate_system.get("forward_axis") == "-Z"
        and coordinate_system.get("linear_unit") == "meter"
        and coordinate_system.get("quaternion_order") == "xyzw"
        and room_scene.get("navmesh_policy") == "load_declared"
        and room_scene.get("load_semantic_mesh") is True
        and room_scene.get("enable_physics") is True,
        "M1 room coordinate/scene production semantics drift",
    )
    _require(m1_request.get("room_id") == room_id, "M1 request room differs from atom")
    m1_rig = _mapping(m1_request.get("primary_camera_rig"), owner="M1 camera rig")
    calibration = _mapping(m1_rig.get("shared_calibration"), owner="M1 calibration")
    atom_camera_framing = _mapping(
        atom.get("camera_framing"), owner="atom camera framing"
    )
    atom_calibration = _mapping(
        atom_camera_framing.get("calibration"), owner="atom camera calibration"
    )
    render = _mapping(scenario.get("render"), owner="suite render")
    height, width = calibration.get("resolution_hw", (None, None))
    _require(
        (height, width) == (render.get("height"), render.get("width"))
        and (height, width) == (720, 1280)
        and float(calibration.get("hfov_degrees"))
        == float(render.get("horizontal_fov_deg"))
        and calibration.get("projection") == "pinhole"
        and calibration.get("near_m") == atom_calibration.get("near_m") == 0.05
        and calibration.get("far_m") == 100.0
        and _same_transform(
            _transform(m1_rig.get("world_from_rig"), owner="M1 rig pose"), first_rig
        ),
        "M1 camera does not match selected suite rig/calibration",
    )
    modalities = _sequence(m1_rig.get("modalities"), owner="M1 modalities")
    _require(
        [item.get("modality") for item in modalities if isinstance(item, Mapping)]
        == ["rgb", "depth", "semantic"],
        "M1 modalities must remain ordered rgb/depth/semantic",
    )
    listener = _mapping(m1_request.get("listener"), owner="M1 listener")
    plan_camera = _mapping(plan.get("camera"), owner="suite plan camera")
    _require(
        m1_rig.get("rig_id") == "camera_rig_0"
        and m1_rig.get("view_id") == "view0"
        and listener.get("listener_id") == plan_camera.get("listener_id") == "listener0"
        and listener.get("attached_to") == m1_rig.get("rig_id"),
        "M1/suite listener or camera rig identity drift",
    )
    _require(
        _same_transform(
            _transform(calibration.get("rig_from_sensor"), owner="M1 rig_from_sensor"),
            {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        )
        and _same_transform(
            _transform(listener.get("rig_from_listener"), owner="M1 rig_from_listener"),
            {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        ),
        "camera sensors and listener must be rigidly co-located/co-oriented",
    )
    m1_sources = _sequence(m1_request.get("sources"), owner="M1 sources")
    _require(
        len(m1_sources) == 2
        and [item.get("source_id") for item in m1_sources if isinstance(item, Mapping)]
        == list(SOURCE_SLOTS),
        "M1 sources must be source1 then source2",
    )
    for index, source in enumerate(m1_sources):
        source_transform = _mapping(
            _mapping(source, owner=f"M1 source {index}").get("world_from_source"),
            owner=f"M1 source {index} transform",
        )
        _require(
            _vec3(
                source_transform.get("translation_m"),
                owner=f"M1 source {index} position",
            )
            == tuple(expected_centers[SOURCE_SLOTS[index]][0]),
            f"M1 {SOURCE_SLOTS[index]} position differs from trajectory/RIR",
        )
        _require(
            _quat(
                source_transform.get("rotation_xyzw"),
                owner=f"M1 source {index} rotation",
            )
            == actor_frames[index][0].rotation_xyzw,
            f"M1 {SOURCE_SLOTS[index]} rotation differs from frozen suite actor",
        )
    return TwoHumanCaptureAuthority(
        episode_id=episode_id,
        room_id=room_id,
        room_revision=room_revision,
        actors=actors,
        actor_frames=actor_frames,
        rig_frames=tuple(normalized_rig),
        resolution_hw=(int(height), int(width)),
        horizontal_fov_deg=float(calibration["hfov_degrees"]),
        suite_visual_role=COMPARISON_VISUAL_ROLE,
        qualification_claim=False,
        formal_dataset_count=0,
    )


def load_two_human_capture_authority(
    *,
    atom_request_path: str | Path,
    suite_plan_path: str | Path,
    sensor_rig_path: str | Path,
    trajectory_bank_path: str | Path,
    rir_plan_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    runtime_root: str | Path | None = None,
) -> TwoHumanCaptureAuthority:
    atom = load_json(atom_request_path)
    registry_path = _safe_regular_path(
        atom.get("actor_framing", {}).get("runtime_profile_registry"),
        owner="runtime profile registry",
    )
    m1_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
    authority = validate_two_human_authority_documents(
        atom=atom,
        suite=load_json(suite_plan_path),
        sensor_rig=load_json(sensor_rig_path),
        trajectory_bank=load_json(trajectory_bank_path),
        rir_plan=load_json(rir_plan_path),
        runtime_profiles=load_source_asset_runtime_registry(registry_path),
        room=m1_inputs.room,
        m1_request=m1_inputs.request,
    )
    selected_runtime = discover_runtime_root(runtime_root)
    resolved = _resolved_scene(m1_inputs, selected_runtime)
    camera_runtime = _mapping(atom.get("camera_runtime"), owner="atom camera runtime")
    atom_room = _mapping(atom.get("room"), owner="atom room")
    path_pairs = {
        "scene": (camera_runtime.get("scene_path"), resolved.get("scene_id")),
        "dataset": (
            camera_runtime.get("dataset_config_path"),
            resolved.get("dataset_config"),
        ),
        "navmesh": (atom_room.get("navmesh_path"), resolved.get("navmesh")),
        "physics": (
            camera_runtime.get("physics_config_path"),
            selected_runtime / "data/default.physics_config.json",
        ),
    }
    for owner, (declared, expected) in path_pairs.items():
        _require_same_runtime_file(declared, expected, owner=owner)
    navigation = _mapping(m1_inputs.room.get("navigation"), owner="M1 navigation")
    _require(
        camera_runtime.get("loaded_scene_id") == atom_room.get("scene_id"),
        "atom camera runtime scene identity differs from M1 room",
    )
    _validate_camera_runtime_navigation(camera_runtime, navigation)
    return authority


__all__ = [
    "HumanActorAuthority",
    "PlannedHumanFrame",
    "TwoHumanCaptureAuthority",
    "TwoHumanCaptureError",
    "load_two_human_capture_authority",
    "validate_two_human_authority_documents",
]

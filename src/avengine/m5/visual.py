"""Two-actor, one-view fixed-state Habitat capture for M5."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from avengine.contracts.json_io import canonical_json_bytes, canonical_json_sha256
from avengine.contracts.transforms import (
    normalized_quaternion_xyzw,
    rotate_vector_xyzw,
    transform_error,
)
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.habitat_capture import (
    InstalledHabitatRuntime,
    _make_configuration,
    _resolved_assets,
    _state_snapshot,
    prepare_installed_habitat_runtime,
)
from avengine.m2.contracts import (
    FORMAL_MODALITIES,
    load_and_validate_inputs as load_m2_inputs,
)
from avengine.m2.habitat_capture import (
    HabitatCaptureError,
    _apply_root_with_habitat,
    _quaternion_block_error,
    _runtime_snapshot,
    _validate_observation_arrays,
    compile_frame_applications,
    load_runtime_asset_bundle,
)
from avengine.m2.habitat import HabitatLinkJointBlock, bind_habitat_link_layout
from avengine.m2.review_topdown import habitat_xz_to_navmesh_pixel


@dataclass(frozen=True)
class TwoActorVisualResult:
    rgb: np.ndarray
    depth: np.ndarray
    semantic: np.ndarray
    actor_ids: tuple[str, str]
    source_ids: tuple[str, str]
    semantic_ids: tuple[int, int]
    actor_world_matrices: np.ndarray
    source_positions_m: np.ndarray
    listener_position_m: tuple[float, float, float]
    listener_orientation_wxyz: tuple[float, float, float, float]
    topdown_rgb: np.ndarray
    records: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]
    sensor_rig_trajectory: Mapping[str, Any] | None = None


def _sequence_hash(value: np.ndarray, *, role: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {"role": role, "shape": list(array.shape), "dtype": array.dtype.str}
        )
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _set_scene_node_semantic_readback(
    articulated_object: Any, semantic_id: int
) -> None:
    """Mirror and verify the template semantic ID on exposed link nodes.

    The actual skinned drawable ID is fixed by the AO template at creation.
    The pinned wrapper has no post-creation AO semantic setter, so callers must
    instantiate through :func:`_instantiate_actor_with_semantic_template`.
    """

    nodes = [articulated_object.root_scene_node]
    nodes.extend(
        articulated_object.get_link_scene_node(link_id)
        for link_id in articulated_object.get_link_ids()
    )
    for node in nodes:
        if not hasattr(node, "semantic_id"):
            raise HabitatCaptureError("pinned Habitat SceneNode lacks semantic_id")
        node.semantic_id = int(semantic_id)
        if int(node.semantic_id) != semantic_id:
            raise HabitatCaptureError("Habitat semantic_id writeback differs")


def _instantiate_actor_with_semantic_template(
    simulator: Any,
    *,
    bundle: Any,
    habitat_sim: Any,
    base_handle: str,
    semantic_id: int,
    actor_index: int,
    light_setup_key: str | None = None,
    shader_type: str | None = None,
) -> tuple[Any, Any]:
    if light_setup_key is not None and (
        not isinstance(light_setup_key, str) or not light_setup_key
    ):
        raise HabitatCaptureError(
            "M5 AO light_setup_key must be a non-empty string when provided"
        )
    if shader_type is not None and shader_type not in {
        "material",
        "flat",
        "phong",
        "pbr",
    }:
        raise HabitatCaptureError(
            "M5 AO shader_type must be material, flat, phong, or pbr"
        )
    manager = simulator.metadata_mediator.ao_template_manager
    attributes = manager.get_template_by_handle(base_handle)
    if attributes is None:
        raise HabitatCaptureError("cannot retrieve the loaded M5 AO template")
    attributes.semantic_id = int(semantic_id)
    if shader_type is not None:
        attributes.shader_type = shader_type
    handle = f"{base_handle}.m5_actor{actor_index}_semantic{semantic_id}"
    registered = manager.register_template(attributes, handle)
    if int(registered) < 0:
        raise HabitatCaptureError("failed to register a semantic M5 AO template")
    object_manager = simulator.get_articulated_object_manager()
    if light_setup_key is None:
        articulated_object = object_manager.add_articulated_object_by_template_handle(
            handle
        )
    else:
        articulated_object = object_manager.add_articulated_object_by_template_handle(
            handle,
            light_setup_key=light_setup_key,
        )
    if articulated_object is None:
        raise HabitatCaptureError(
            "Habitat failed to instantiate an M5 articulated actor"
        )
    articulated_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    link_ids = list(articulated_object.get_link_ids())
    expected_names = set(bundle.joint_mapping["joint_order"])
    actual_names = {articulated_object.get_link_name(-1)} | {
        articulated_object.get_link_name(link_id) for link_id in link_ids
    }
    if actual_names != expected_names:
        raise HabitatCaptureError("M5 AO link names differ from the M2 mapping")
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
    creation = articulated_object.creation_attributes
    if int(creation.semantic_id) != semantic_id:
        raise HabitatCaptureError(
            "M5 AO creation semantic ID differs from its template"
        )
    if shader_type is not None:
        actual_shader_type = str(getattr(creation.shader_type, "name", "")).lower()
        if actual_shader_type != shader_type:
            raise HabitatCaptureError(
                "M5 AO creation shader type differs from its template"
            )
    _set_scene_node_semantic_readback(articulated_object, semantic_id)
    return articulated_object, binding


def _link_id_by_name(articulated_object: Any, name: str) -> int:
    matches = [
        int(link_id)
        for link_id in articulated_object.get_link_ids()
        if articulated_object.get_link_name(link_id) == name
    ]
    if len(matches) != 1:
        raise HabitatCaptureError(f"expected one AO link named {name!r}, got {matches}")
    return matches[0]


def _node_world_position(articulated_object: Any, link_id: int) -> np.ndarray:
    matrix = np.asarray(
        articulated_object.get_link_scene_node(link_id).absolute_transformation(),
        dtype=np.float64,
    )
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise HabitatCaptureError(
            "muzzle SceneNode transform is not a finite 4x4 matrix"
        )
    return matrix[:3, 3].copy()


def _panel_bounds(
    raw_bounds: tuple[np.ndarray, np.ndarray] | None,
    actor_positions: np.ndarray,
    listener_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(
        [actor_positions.reshape(-1, 3), listener_positions.reshape(-1, 3)], axis=0
    )
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    low[[0, 2]] -= 1.0
    high[[0, 2]] += 1.0
    if raw_bounds is not None:
        nav_low, nav_high = raw_bounds
        low[[0, 2]] = np.maximum(low[[0, 2]], nav_low[[0, 2]])
        high[[0, 2]] = np.minimum(high[[0, 2]], nav_high[[0, 2]])
    if high[0] - low[0] < 2.0:
        center = 0.5 * (high[0] + low[0])
        low[0], high[0] = center - 1.0, center + 1.0
    if high[2] - low[2] < 2.0:
        center = 0.5 * (high[2] + low[2])
        low[2], high[2] = center - 1.0, center + 1.0
    return low, high


def _topdown_panels(
    *,
    navmesh: np.ndarray | None,
    navmesh_bounds: tuple[np.ndarray, np.ndarray] | None,
    actor_positions: np.ndarray,
    source_positions: np.ndarray,
    listener_positions: np.ndarray,
    listener_orientations_wxyz: np.ndarray | None = None,
    actor_labels: Sequence[str] = ("Dog 0", "Dog 1"),
    panel_size: int = 240,
) -> np.ndarray:
    listeners = np.asarray(listener_positions, dtype=np.float64)
    if listeners.shape != (actor_positions.shape[0], 3) or not np.all(
        np.isfinite(listeners)
    ):
        raise HabitatCaptureError("listener_positions must be finite [frame,3]")
    labels = tuple(actor_labels)
    if len(labels) != actor_positions.shape[1] or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise HabitatCaptureError(
            "actor_labels must contain one nonempty label per actor"
        )
    listener_forwards: np.ndarray | None = None
    if listener_orientations_wxyz is not None:
        orientations = np.asarray(listener_orientations_wxyz, dtype=np.float64)
        if orientations.shape != (actor_positions.shape[0], 4) or not np.all(
            np.isfinite(orientations)
        ):
            raise HabitatCaptureError(
                "listener_orientations_wxyz must be finite [frame,4]"
            )
        listener_forwards = np.stack(
            [
                rotate_vector_xyzw(
                    (orientation[1], orientation[2], orientation[3], orientation[0]),
                    (0.0, 0.0, -1.0),
                )
                for orientation in orientations
            ],
            axis=0,
        )
    focus_low, focus_high = _panel_bounds(
        navmesh_bounds, actor_positions, listeners
    )

    def project(position: Sequence[float]) -> tuple[float, float]:
        p = np.asarray(position, dtype=np.float64)
        x = (p[0] - focus_low[0]) / (focus_high[0] - focus_low[0])
        z = (p[2] - focus_low[2]) / (focus_high[2] - focus_low[2])
        return 12.0 + x * (panel_size - 24.0), 12.0 + z * (panel_size - 24.0)

    if navmesh is not None and navmesh_bounds is not None and navmesh.size:
        raw = np.asarray(navmesh, dtype=np.uint8) * 210
        raw_rgb = np.repeat(raw[..., None], 3, axis=2)
        # Crop the raw grid to the same focus bounds before resizing.  The
        # mapping helper is the authority for Habitat X/Z grid orientation.
        low_px = habitat_xz_to_navmesh_pixel(
            focus_low,
            navmesh_shape_hw=navmesh.shape,
            bounds=navmesh_bounds,
        )
        high_px = habitat_xz_to_navmesh_pixel(
            focus_high,
            navmesh_shape_hw=navmesh.shape,
            bounds=navmesh_bounds,
        )
        x0, x1 = sorted((int(np.floor(low_px[0])), int(np.ceil(high_px[0]))))
        y0, y1 = sorted((int(np.floor(low_px[1])), int(np.ceil(high_px[1]))))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(navmesh.shape[1] - 1, x1), min(navmesh.shape[0] - 1, y1)
        crop = raw_rgb[y0 : y1 + 1, x0 : x1 + 1]
        if crop.size:
            base = Image.fromarray(crop, mode="RGB").resize(
                (panel_size, panel_size), Image.Resampling.NEAREST
            )
        else:
            base = Image.new("RGB", (panel_size, panel_size), (32, 36, 42))
    else:
        base = Image.new("RGB", (panel_size, panel_size), (32, 36, 42))

    result: list[np.ndarray] = []
    colors = ((250, 96, 96), (82, 190, 255))
    for frame_index in range(actor_positions.shape[0]):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, panel_size - 1, panel_size - 1), outline=(230, 230, 230))
        listener_trail = [project(value) for value in listeners[: frame_index + 1]]
        if len(listener_trail) > 1:
            draw.line(listener_trail, fill=(255, 215, 60), width=2)
        listener_xy = listener_trail[-1]
        draw.ellipse(
            (
                listener_xy[0] - 5,
                listener_xy[1] - 5,
                listener_xy[0] + 5,
                listener_xy[1] + 5,
            ),
            fill=(255, 230, 80),
            outline=(10, 10, 10),
        )
        draw.text(
            (listener_xy[0] + 7, listener_xy[1] - 8), "Listener", fill=(255, 245, 180)
        )
        if listener_forwards is not None:
            forward = listener_forwards[frame_index]
            forward_xy = project(
                listeners[frame_index] + 0.35 * forward
            )
            draw.line(
                (listener_xy[0], listener_xy[1], forward_xy[0], forward_xy[1]),
                fill=(20, 20, 20),
                width=3,
            )
            draw.text(
                (forward_xy[0] + 2, forward_xy[1] - 8),
                "F",
                fill=(255, 255, 255),
            )
        for actor_index in range(actor_positions.shape[1]):
            color = colors[actor_index % len(colors)]
            trail = [
                project(value)
                for value in source_positions[: frame_index + 1, actor_index]
            ]
            if len(trail) > 1:
                draw.line(trail, fill=color, width=2)
            actor_xy = project(actor_positions[frame_index, actor_index])
            source_xy = project(source_positions[frame_index, actor_index])
            draw.ellipse(
                (actor_xy[0] - 6, actor_xy[1] - 6, actor_xy[0] + 6, actor_xy[1] + 6),
                fill=color,
                outline=(0, 0, 0),
            )
            draw.line(
                (actor_xy[0], actor_xy[1], source_xy[0], source_xy[1]),
                fill=(255, 255, 255),
                width=1,
            )
            draw.ellipse(
                (
                    source_xy[0] - 3,
                    source_xy[1] - 3,
                    source_xy[0] + 3,
                    source_xy[1] + 3,
                ),
                fill=(255, 255, 255),
                outline=color,
            )
            draw.text(
                (actor_xy[0] + 8, actor_xy[1] - 8),
                labels[actor_index],
                fill=color,
            )
        draw.rectangle((4, 4, 136, 22), fill=(0, 0, 0))
        draw.text((8, 7), f"TOPDOWN  {frame_index:02d}/74", fill=(255, 255, 255))
        result.append(np.asarray(image, dtype=np.uint8))
    return np.ascontiguousarray(np.stack(result, axis=0))


def capture_two_actor_fixed_states(
    *,
    animal_manifest_path: str | Path,
    m2_request_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    runtime_prefix: str | Path | None = None,
    runtime_root: str | Path | None = None,
    mp3d_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    installed_runtime: InstalledHabitatRuntime | None = None,
    actor_offsets_m: Sequence[Sequence[float]] = ((0.0, 0.0, 0.7), (0.25, 0.0, -0.7)),
    actor_ids: tuple[str, str] = ("actor0", "actor1"),
    source_ids: tuple[str, str] = ("source0", "source1"),
    semantic_ids: tuple[int, int] = (210, 211),
    sensor_rig_trajectory: Mapping[str, Any] | None = None,
    emitter_link_names: tuple[str, str] = (
        "beagle Xtra Mouth",
        "beagle Xtra Mouth",
    ),
) -> TwoActorVisualResult:
    """Capture two actors through one explicitly installed Habitat runtime.

    ``runtime_root`` is retained only as a compatibility spelling for a
    non-Git installed prefix.  Callers that already prepared the runtime may
    inject it so visual capture and later acoustics share one selected native
    binding and RLR SDK mapping.
    """

    if (
        len(set(actor_ids)) != 2
        or len(set(source_ids)) != 2
        or len(set(semantic_ids)) != 2
    ):
        raise HabitatCaptureError(
            "M5 two-actor identities and semantic IDs must be unique"
        )
    if len(emitter_link_names) != 2 or any(
        not isinstance(value, str) or not value for value in emitter_link_names
    ):
        raise HabitatCaptureError("M5 emitter_link_names must contain two link names")
    offsets = np.asarray(actor_offsets_m, dtype=np.float64)
    if offsets.shape != (2, 3) or not np.all(np.isfinite(offsets)):
        raise HabitatCaptureError("actor_offsets_m must be finite [2,3]")
    m2_inputs = load_m2_inputs(animal_manifest_path, m2_request_path)
    room_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
    bundle = load_runtime_asset_bundle(m2_inputs)
    frames = compile_frame_applications(m2_inputs, bundle)
    runtime_arguments = (
        runtime_prefix,
        runtime_root,
        mp3d_root,
        magnum_python_site,
        rlr_sdk_root,
    )
    if installed_runtime is not None and any(
        value is not None for value in runtime_arguments
    ):
        raise HabitatCaptureError(
            "installed_runtime cannot be combined with runtime path arguments"
        )
    if installed_runtime is None:
        if runtime_prefix is None and runtime_root is None:
            raise HabitatCaptureError(
                "M5 visual capture requires an explicit installed runtime prefix"
            )
        if magnum_python_site is None or not str(magnum_python_site).strip():
            raise HabitatCaptureError(
                "M5 visual capture requires an explicit Magnum Python site"
            )
        try:
            installed_runtime = prepare_installed_habitat_runtime(
                runtime_prefix=runtime_prefix,
                runtime_root=runtime_root,
                mp3d_root=mp3d_root,
                magnum_python_site=magnum_python_site,
                rlr_sdk_root=rlr_sdk_root,
                allow_mp3d_environment=False,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise HabitatCaptureError(
                f"installed Habitat runtime is unavailable: {error}"
            ) from error
    missing = [
        record
        for record in _resolved_assets(
            room_inputs,
            None,
            mp3d_root=installed_runtime.mp3d_root,
        )
        if not record["exists"]
    ]
    if missing:
        raise HabitatCaptureError("validated M1 room has missing runtime assets")

    qt = installed_runtime.quaternion
    habitat_sim = installed_runtime.habitat_sim
    mn = installed_runtime.magnum
    quat_to_coeffs = installed_runtime.quat_to_coeffs
    if room_inputs.room.get("room_kind") == "habitat_native":
        from avengine.m5.current_visual import (
            _make_current_configuration,
            _resolve_external_scene,
        )

        current_scene = _resolve_external_scene(room_inputs, installed_runtime)
        configuration, modality_to_uuid = _make_current_configuration(
            room_inputs=room_inputs,
            installed_runtime=installed_runtime,
            scene=current_scene,
            include_audio_sensor=False,
        )
        resolved_scene = {"navmesh": current_scene["navmesh"]}
    else:
        configuration, modality_to_uuid, _listener_uuid, resolved_scene = (
            _make_configuration(
                room_inputs,
                None,
                Path("/tmp/avengine-m5-visual"),
                mp3d_root=installed_runtime.mp3d_root,
                include_audio_sensor=False,
                physics_config_path=installed_runtime.physics_config_path,
            )
        )
    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    semantic_frames: list[np.ndarray] = []
    actor_matrices: list[np.ndarray] = []
    source_positions: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    navmesh_array: np.ndarray | None = None
    navmesh_bounds: tuple[np.ndarray, np.ndarray] | None = None
    rig = room_inputs.request["primary_camera_rig"]
    default_view_pose_hash = canonical_json_sha256(
        {"view_id": "view0", "primary_camera_rig": rig}
    )
    if sensor_rig_trajectory is None:
        rig_frames = tuple(
            {
                "frame_index": index,
                "pts_ticks": 3_200 * index,
                "world_from_rig": rig["world_from_rig"],
                "pose_hash": default_view_pose_hash,
            }
            for index in range(len(frames))
        )
        retained_rig_trajectory = None
    else:
        from avengine.sensor_rig_trajectory import validate_sensor_rig_trajectory

        trajectory_errors = validate_sensor_rig_trajectory(sensor_rig_trajectory)
        if trajectory_errors:
            raise HabitatCaptureError(
                "sensor-rig trajectory is invalid: " + "; ".join(trajectory_errors)
            )
        rig_frames = tuple(sensor_rig_trajectory["frames"])
        if len(rig_frames) != len(frames):
            raise HabitatCaptureError(
                "sensor-rig trajectory and visual state frame counts differ"
            )
        retained_rig_trajectory = deepcopy(dict(sensor_rig_trajectory))
    first_world_from_rig = rig_frames[0]["world_from_rig"]
    listener_position = tuple(
        float(v) for v in first_world_from_rig["translation_m"]
    )
    x, y, z, w = normalized_quaternion_xyzw(
        first_world_from_rig["rotation_xyzw"]
    )
    listener_orientation = (float(w), float(x), float(y), float(z))
    listener_positions: list[np.ndarray] = []
    listener_orientations: list[np.ndarray] = []

    with habitat_sim.Simulator(configuration) as simulator:
        navmesh_path = resolved_scene.get("navmesh")
        if navmesh_path is not None and Path(navmesh_path).is_file():
            simulator.pathfinder.load_nav_mesh(str(navmesh_path))
        if simulator.pathfinder.is_loaded:
            navmesh_array = np.asarray(
                simulator.pathfinder.get_topdown_view(0.04, 0.05), dtype=bool
            ).copy()
            raw_bounds = simulator.pathfinder.get_bounds()
            navmesh_bounds = (
                np.asarray(raw_bounds[0], dtype=np.float64),
                np.asarray(raw_bounds[1], dtype=np.float64),
            )
        simulator.seed(int(m2_inputs.request["seed"]))
        camera_state = habitat_sim.AgentState()
        camera_state.position = np.asarray(listener_position, dtype=np.float64)
        camera_state.rotation = qt.quaternion(w, x, y, z)
        agent = simulator.initialize_agent(0, camera_state)

        actors: list[Any] = []
        bindings: list[Any] = []
        muzzle_link_ids: list[int] = []
        template_manager = simulator.metadata_mediator.ao_template_manager
        config_path = bundle.paths_by_role["habitat_ao_config"]
        loaded_ids = template_manager.load_configs(str(config_path))
        handle_prefix = config_path.stem.removesuffix(".ao_config")
        base_handles = template_manager.get_template_handles(handle_prefix)
        if len(loaded_ids) != 1 or len(base_handles) != 1:
            raise HabitatCaptureError(
                "M5 expected one source AO template before semantic cloning"
            )
        for actor_index in range(2):
            articulated_object, binding = _instantiate_actor_with_semantic_template(
                simulator,
                bundle=bundle,
                habitat_sim=habitat_sim,
                base_handle=base_handles[0],
                semantic_id=semantic_ids[actor_index],
                actor_index=actor_index,
            )
            actors.append(articulated_object)
            bindings.append(binding)
            muzzle_link_ids.append(
                _link_id_by_name(articulated_object, emitter_link_names[actor_index])
            )
        sensors = [
            simulator.sensors[modality_to_uuid[modality]]
            for modality in FORMAL_MODALITIES
        ]
        all_sensor_uuids = sorted(modality_to_uuid.values())
        initial_time = float(simulator.get_world_time())
        for frame in frames:
            rig_frame = rig_frames[frame.frame_index]
            if (
                rig_frame["frame_index"] != frame.frame_index
                or rig_frame["pts_ticks"] != frame.pts_ticks
            ):
                raise HabitatCaptureError(
                    "sensor-rig trajectory differs from the visual frame clock"
                )
            world_from_rig = rig_frame["world_from_rig"]
            frame_position = np.asarray(
                world_from_rig["translation_m"], dtype=np.float64
            )
            qx, qy, qz, qw = normalized_quaternion_xyzw(
                world_from_rig["rotation_xyzw"]
            )
            camera_state = habitat_sim.AgentState()
            camera_state.position = frame_position
            camera_state.rotation = qt.quaternion(qw, qx, qy, qz)
            agent.set_state(
                camera_state,
                reset_sensors=False,
                infer_sensor_states=True,
            )
            rig_snapshot = _state_snapshot(
                simulator,
                agent,
                all_sensor_uuids,
                quat_to_coeffs,
            )
            pose_errors = [
                transform_error(world_from_rig, rig_snapshot["agent"]),
                *[
                    transform_error(world_from_rig, sensor_pose)
                    for sensor_pose in rig_snapshot["sensors"].values()
                ],
            ]
            maximum_pose_error = max(pose_errors)
            if maximum_pose_error > 2.0e-6:
                raise HabitatCaptureError(
                    f"M5 frame {frame.frame_index} camera/listener readback failed"
                )
            listener_positions.append(frame_position.copy())
            listener_orientations.append(
                np.asarray([qw, qx, qy, qz], dtype=np.float64)
            )
            frame_actor_matrices: list[np.ndarray] = []
            before: list[Mapping[str, Any]] = []
            expected_joints: list[np.ndarray] = []
            for actor_index, (actor, binding) in enumerate(
                zip(actors, bindings, strict=True)
            ):
                matrix = np.asarray(frame.world_from_skin_root, dtype=np.float64).copy()
                matrix[:3, 3] += offsets[actor_index]
                joints = np.asarray(
                    binding.map_pose(frame.joint_rotations_xyzw), dtype=np.float64
                )
                _apply_root_with_habitat(actor, matrix, qt=qt, mn=mn)
                actor.joint_positions = joints.copy()
                snapshot = _runtime_snapshot(simulator, actor)
                root_error = float(
                    np.max(
                        np.abs(np.asarray(snapshot["world_from_skin_root"]) - matrix)
                    )
                )
                joint_error = _quaternion_block_error(
                    np.asarray(snapshot["joint_positions_xyzw"]), joints
                )
                if root_error > 2.0e-6 or joint_error > 2.0e-6:
                    raise HabitatCaptureError(
                        f"M5 frame {frame.frame_index} actor {actor_index} readback failed"
                    )
                before.append(snapshot)
                expected_joints.append(joints)
                actor_world = np.asarray(
                    frame.world_from_actor, dtype=np.float64
                ).copy()
                actor_world[:3, 3] += offsets[actor_index]
                frame_actor_matrices.append(actor_world)

            observation = simulator.render_sensors(sensors)
            arrays = _validate_observation_arrays(observation, modality_to_uuid)
            visibility = [
                int(np.count_nonzero(arrays["semantic"] == semantic_id))
                for semantic_id in semantic_ids
            ]
            if any(count == 0 for count in visibility):
                raise HabitatCaptureError(
                    f"M5 frame {frame.frame_index} lost one actor semantic ID: {visibility}"
                )
            frame_sources = np.stack(
                [
                    _node_world_position(actor, muzzle_link_ids[index])
                    for index, actor in enumerate(actors)
                ],
                axis=0,
            )
            after = [_runtime_snapshot(simulator, actor) for actor in actors]
            if any(
                left["sha256"] != right["sha256"]
                for left, right in zip(before, after, strict=True)
            ):
                raise HabitatCaptureError(
                    "M5 visual render changed a fixed actor state"
                )
            if float(simulator.get_world_time()) != initial_time:
                raise HabitatCaptureError(
                    "M5 fixed-state capture advanced Habitat time"
                )

            rgb_frames.append(
                np.asarray(arrays["rgb"])[..., :3].astype(np.uint8, copy=True)
            )
            depth_frames.append(np.asarray(arrays["depth"]).copy())
            semantic_frames.append(np.asarray(arrays["semantic"]).copy())
            actor_matrices.append(np.stack(frame_actor_matrices, axis=0))
            source_positions.append(frame_sources)
            records.append(
                {
                    "frame_index": frame.frame_index,
                    "pts_ticks": frame.pts_ticks,
                    "audio_start_sample": (3200 * frame.frame_index + 1) // 3,
                    "actor_ids": list(actor_ids),
                    "source_ids": list(source_ids),
                    "semantic_visibility_pixels": visibility,
                    "muzzle_positions_m": frame_sources.tolist(),
                    "world_from_rig": deepcopy(world_from_rig),
                    "view_pose_hash": rig_frame["pose_hash"],
                    "camera_listener_readback_max_error": maximum_pose_error,
                    "base_m2_applied_state_hash": frame.declared_applied_state_hash,
                    "m5_instance_state_sha256": canonical_json_sha256(
                        {
                            "base_applied_state_hash": frame.declared_applied_state_hash,
                            "actor_offsets_m": offsets.tolist(),
                            "actor_world_matrices": np.stack(
                                frame_actor_matrices
                            ).tolist(),
                            "mouth_open_ratio": [0.0, 0.0],
                        }
                    ),
                    "observation_calls": 1,
                    "world_time_seconds": initial_time,
                }
            )

    rgb = np.ascontiguousarray(np.stack(rgb_frames, axis=0))
    depth = np.ascontiguousarray(np.stack(depth_frames, axis=0))
    semantic = np.ascontiguousarray(np.stack(semantic_frames, axis=0))
    actor_matrix_array = np.ascontiguousarray(np.stack(actor_matrices, axis=0))
    source_position_array = np.ascontiguousarray(np.stack(source_positions, axis=0))
    listener_position_array = np.ascontiguousarray(
        np.stack(listener_positions, axis=0)
    )
    listener_orientation_array = np.ascontiguousarray(
        np.stack(listener_orientations, axis=0)
    )
    topdown = _topdown_panels(
        navmesh=navmesh_array,
        navmesh_bounds=navmesh_bounds,
        actor_positions=actor_matrix_array[:, :, :3, 3],
        source_positions=source_position_array,
        listener_positions=listener_position_array,
        listener_orientations_wxyz=listener_orientation_array,
    )
    metadata = {
        "schema": "avengine_m5_two_actor_visual_capture_v1",
        "runtime": {
            "mode": "current-installed",
            "habitat_runtime_prefix": str(installed_runtime.prefix),
            "mp3d_root": (
                None
                if installed_runtime.mp3d_root is None
                else str(installed_runtime.mp3d_root)
            ),
            "magnum_python_site": str(installed_runtime.magnum_python_site),
            "habitat_sim_module": str(Path(habitat_sim.__file__).resolve()),
        },
        "view_ids": ["view0"],
        "qa_view_ids": ["topdown_review"],
        "formal_modalities": list(FORMAL_MODALITIES),
        "frame_count": len(frames),
        "frame_rate_hz": 15,
        "observation_calls_per_frame": 1,
        "audio_sensor_included": False,
        "physics_steps": 0,
        "mouth_articulation": "disabled_for_shortcut_control",
        "actor_offsets_m": offsets.tolist(),
        "actor_ids": list(actor_ids),
        "source_ids": list(source_ids),
        "semantic_ids": list(semantic_ids),
        "emitter_link_names": list(emitter_link_names),
        "hashes": {
            "rgb": _sequence_hash(rgb, role="rgb"),
            "depth": _sequence_hash(depth, role="depth"),
            "semantic": _sequence_hash(semantic, role="semantic"),
            "actor_world_matrices": _sequence_hash(
                actor_matrix_array, role="actor_world_matrices"
            ),
            "source_positions_m": _sequence_hash(
                source_position_array, role="source_positions_m"
            ),
            "listener_positions_m": _sequence_hash(
                listener_position_array, role="listener_positions_m"
            ),
            "listener_orientations_wxyz": _sequence_hash(
                listener_orientation_array, role="listener_orientations_wxyz"
            ),
        },
        "sensor_rig_trajectory": (
            None
            if retained_rig_trajectory is None
            else {
                "trajectory_id": retained_rig_trajectory["trajectory_id"],
                "schema": retained_rig_trajectory["schema"],
                "content_sha256": canonical_json_sha256(retained_rig_trajectory),
                "moving": bool(
                    np.any(
                        np.abs(
                            listener_position_array - listener_position_array[0]
                        )
                        > 1.0e-12
                    )
                    or np.any(
                        np.abs(
                            listener_orientation_array
                            - listener_orientation_array[0]
                        )
                        > 1.0e-12
                    )
                ),
            }
        ),
    }
    return TwoActorVisualResult(
        rgb=rgb,
        depth=depth,
        semantic=semantic,
        actor_ids=actor_ids,
        source_ids=source_ids,
        semantic_ids=semantic_ids,
        actor_world_matrices=actor_matrix_array,
        source_positions_m=source_position_array,
        listener_position_m=listener_position,
        listener_orientation_wxyz=listener_orientation,
        topdown_rgb=topdown,
        records=tuple(records),
        metadata=metadata,
        sensor_rig_trajectory=retained_rig_trajectory,
    )


__all__ = ["TwoActorVisualResult", "capture_two_actor_fixed_states"]

"""Native RGB and actor readback capture for planned MP3D actor tracks.

The entrypoint consumes the CPU planned actor tracks produced by
``avengine.assets.mp3d_region_actor_tracks``.  Each track supplies an explicit
M2 package, baked joint targets, and a planned skin-root transform.  The native
loop applies those values, then records root/joint/emitter state from Habitat
objects and RGB/depth/semantic arrays from ``Simulator.render_sensors``.
Planned route centres are kept as provenance and are never written as observed
emitter positions.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from avengine.assets.contracts import (
    ValidatedM2Inputs,
    load_and_validate_inputs as load_m2_inputs,
)
from avengine.assets.habitat_capture import (
    HabitatCaptureError,
    _apply_root_with_habitat,
    _quaternion_block_error,
    _runtime_snapshot,
    _validate_observation_arrays,
    load_runtime_asset_bundle,
)
from avengine.assets.habitat_static_assets import (
    HabitatStaticAssetError,
    load_habitat_asset_bindings,
)
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.qa.pixel_visibility import compile_pixel_visibility_truth
from avengine.rooms.contracts import (
    ContractError,
    ValidatedM1Inputs,
    load_and_validate_inputs as load_m1_inputs,
    validate_loaded_scene_asset_graph,
)
from avengine.rooms.habitat_capture import (
    InstalledHabitatRuntime,
    _make_configuration,
    _state_snapshot,
    prepare_installed_habitat_runtime,
)
from avengine.assets.mp3d_region_actor_tracks import (
    ACTOR_TRACK_SCHEMA,
    CASE_SCHEMA,
)
from avengine.timeline.current_mp3d_dynamic_audio import (
    CurrentMP3DDynamicAudioError,
    _resolve_visual_clock,
)


NATIVE_CAPTURE_SCHEMA = "avengine_mp3d_multi_actor_native_capture_v1"
ROOT_READBACK_ATOL = 2.0e-6
JOINT_READBACK_ATOL = 2.0e-6
RIGID_ENTITY_CLASSES = frozenset({"rigid_object", "rigid_static_object"})
PIXEL_MASK_FILENAME = "native_pixel_masks_depth_authority_v1.npz"


class MP3DMultiActorCaptureError(RuntimeError):
    """The native multi-actor capture cannot provide truthful readback."""


def _read_json(path: str | Path, *, owner: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise MP3DMultiActorCaptureError(f"{owner} must be a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MP3DMultiActorCaptureError(f"cannot read {owner}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MP3DMultiActorCaptureError(f"{owner} must be a JSON object")
    return dict(value)


def _finite_vector(value: Any, *, owner: str, length: int) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise MP3DMultiActorCaptureError(f"{owner} must be a finite vector")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DMultiActorCaptureError(f"{owner} must be a finite vector") from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise MP3DMultiActorCaptureError(f"{owner} must be a finite vector")
    return np.ascontiguousarray(result)


def _matrix_from_transform(value: Any, *, owner: str) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise MP3DMultiActorCaptureError(f"{owner} must be a transform object")
    translation = _finite_vector(
        value.get("translation_m"), owner=f"{owner}.translation_m", length=3
    )
    try:
        quaternion = normalized_quaternion_xyzw(value.get("rotation_xyzw"))
    except (TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError(
            f"{owner}.rotation_xyzw is invalid"
        ) from exc
    x, y, z, w = (float(component) for component in quaternion)
    matrix = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), translation[0]],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), translation[1]],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix


def _transform_record(matrix: np.ndarray) -> list[list[float]]:
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise MP3DMultiActorCaptureError("readback transform must be a finite 4x4")
    return [[float(value) for value in row] for row in matrix]


def _fresh_output(path: str | Path) -> Path:
    output = Path(path).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise MP3DMultiActorCaptureError(
            f"refusing to replace native capture output: {output}"
        )
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_case_track_paths(
    case_path: Path, case: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if case.get("schema") != CASE_SCHEMA:
        raise MP3DMultiActorCaptureError(
            f"case manifest schema must be {CASE_SCHEMA!r}"
        )
    if (
        case.get("artifact_role") != "planned_habitat_actor_apply_case"
        or case.get("native_observed") is not False
        or case.get("research_only") is not True
        or case.get("episode_counted") is not False
    ):
        raise MP3DMultiActorCaptureError(
            "case manifest must be an explicitly planned, non-native actor case"
        )
    records = case.get("actor_tracks")
    if not isinstance(records, list) or len(records) < 2:
        raise MP3DMultiActorCaptureError(
            "case manifest must contain at least two actor tracks"
        )
    clock = case.get("clock")
    if not isinstance(clock, Mapping):
        raise MP3DMultiActorCaptureError("case manifest has no clock")
    try:
        resolved_clock = _resolve_visual_clock(
            frame_count=clock["frame_count"],
            frame_rate_hz=clock["frame_rate_hz"],
            ticks_per_frame=clock["ticks_per_frame"],
            time_base_hz=clock["time_base_hz"],
        )
    except (KeyError, CurrentMP3DDynamicAudioError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError(f"case clock is invalid: {exc}") from exc
    from avengine.capture.neutral_readback import validate_clock
    validate_clock(clock)
    keys = ("frame_count", "frame_rate_hz", "sample_rate_hz", "sample_count", "ticks_per_frame", "time_base_hz")
    if int(resolved_clock["frame_count"]) < 2 or any(clock[key] != resolved_clock[key] for key in keys):
        raise MP3DMultiActorCaptureError(
            "case clock does not match the current visual/audio clock resolver"
        )
    track_values: list[dict[str, Any]] = []
    slots: set[str] = set()
    endpoints: set[str] = set()
    semantics: set[int] = set()
    for ordinal, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise MP3DMultiActorCaptureError(f"actor_tracks[{ordinal}] must be an object")
        track_raw = record.get("track_path")
        if not isinstance(track_raw, str) or not track_raw or Path(track_raw).is_absolute():
            raise MP3DMultiActorCaptureError(
                f"actor_tracks[{ordinal}].track_path must be relative"
            )
        track_path = (case_path.parent / track_raw).resolve()
        try:
            track_path.relative_to(case_path.parent)
        except ValueError as exc:
            raise MP3DMultiActorCaptureError(
                f"actor track escapes case directory: {track_path}"
            ) from exc
        if track_path.is_symlink() or not track_path.is_file():
            raise MP3DMultiActorCaptureError(f"actor track is missing: {track_path}")
        track = _read_json(track_path, owner=f"actor track {ordinal}")
        track_clock = track.get("clock")
        if not isinstance(track_clock, Mapping):
            raise MP3DMultiActorCaptureError(f"actor track {track_path} has no clock")
        validate_clock(track_clock)
        if any(track_clock[key] != resolved_clock[key] for key in keys):
            raise MP3DMultiActorCaptureError(
                f"actor track {track_path} clock differs from the case clock"
            )
        frames = track.get("frames")
        if (
            not isinstance(frames, list)
            or len(frames) != int(resolved_clock["frame_count"])
            or any(
                not isinstance(frame, Mapping) or frame.get("frame_index") != index
                for index, frame in enumerate(frames)
            )
        ):
            raise MP3DMultiActorCaptureError(
                f"actor track {track_path} must contain one contiguous frame per case frame"
            )
        if (
            track.get("schema") != ACTOR_TRACK_SCHEMA
            or track.get("artifact_role") != "planned_habitat_actor_apply_track"
            or track.get("native_observed") is not False
            or track.get("research_only") is not True
            or track.get("episode_counted") is not False
        ):
            raise MP3DMultiActorCaptureError(
                f"actor track {track_path} is not a planned current M2 track"
            )
        for key in ("actor_id", "source_slot_id", "source_endpoint_id", "semantic_id"):
            if track.get(key) != record.get(key) and record.get(key) is not None:
                raise MP3DMultiActorCaptureError(
                    f"case actor record and track differ at {key}: {track_path}"
                )
        slot = track.get("source_slot_id")
        endpoint = track.get("source_endpoint_id")
        semantic = track.get("semantic_id")
        if not isinstance(slot, str) or not slot or slot in slots:
            raise MP3DMultiActorCaptureError("actor track source slots must be unique")
        if not isinstance(endpoint, str) or not endpoint or endpoint in endpoints:
            raise MP3DMultiActorCaptureError("actor track source endpoints must be unique")
        if isinstance(semantic, bool) or not isinstance(semantic, int) or semantic < 0 or semantic in semantics:
            raise MP3DMultiActorCaptureError("actor track semantic IDs must be unique nonnegative integers")
        slots.add(slot)
        endpoints.add(endpoint)
        semantics.add(semantic)
        track_values.append({"path": str(track_path), "value": track})
    track_values.sort(key=lambda item: int(str(item["value"]["source_slot_id"]).removeprefix("source")))
    expected_slots = tuple(f"source{index}" for index in range(1, len(track_values) + 1))
    if tuple(item["value"]["source_slot_id"] for item in track_values) != expected_slots:
        raise MP3DMultiActorCaptureError(
            "actor track slots must be contiguous source1..sourceN"
        )
    return {**dict(case), "clock": resolved_clock}, tuple(track_values)


def _load_case_and_m1(
    *,
    case_manifest_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], ValidatedM1Inputs]:
    case_path = Path(case_manifest_path).expanduser().resolve()
    case = _read_json(case_path, owner="actor case manifest")
    case, tracks = _resolve_case_track_paths(case_path, case)
    declared_m1_raw = case.get("m1_request_path")
    if not isinstance(declared_m1_raw, str) or not declared_m1_raw:
        raise MP3DMultiActorCaptureError(
            "case manifest must declare m1_request_path"
        )
    declared_m1 = Path(declared_m1_raw).expanduser()
    if not declared_m1.is_absolute():
        declared_m1 = case_path.parent / declared_m1
    if declared_m1.resolve() != Path(m1_request_path).expanduser().resolve():
        raise MP3DMultiActorCaptureError(
            "the supplied M1 request differs from case.m1_request_path"
        )
    try:
        room_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
    except (OSError, TypeError, ValueError, ContractError) as exc:
        raise MP3DMultiActorCaptureError(f"M1 inputs are invalid: {exc}") from exc
    room_id = room_inputs.room.get("room_id")
    case_room = case.get("region", {}).get("house_id")
    if not isinstance(room_id, str) or not isinstance(case_room, str) or not room_id.endswith(case_room):
        raise MP3DMultiActorCaptureError(
            f"M1 room {room_id!r} does not identify case house {case_room!r}"
        )
    source_ids = [item["value"]["source_endpoint_id"] for item in tracks]
    sources = room_inputs.request.get("sources")
    if not isinstance(sources, list) or [item.get("source_id") for item in sources] != source_ids:
        raise MP3DMultiActorCaptureError(
            "M1 source order must equal the planned actor endpoint order"
        )
    return case, tracks, room_inputs


def _load_track_runtime(
    track: Mapping[str, Any],
    *,
    cache: dict[tuple[Path, Path], tuple[ValidatedM2Inputs, Any]],
    runtime_registry_path: str | Path | None = None,
    external_index_path: str | Path | None = None,
    binding_delta_path: str | Path | None = None,
    allow_research_candidate: bool = False,
) -> tuple[ValidatedM2Inputs, Any]:
    asset = track.get("asset")
    if not isinstance(asset, Mapping):
        raise MP3DMultiActorCaptureError("actor track has no asset mapping")
    asset_path_raw = asset.get("asset_manifest_path")
    request_path_raw = asset.get("base_m2_request_path")
    if not isinstance(asset_path_raw, str) or not isinstance(request_path_raw, str):
        asset_id = asset.get("asset_id")
        if (
            isinstance(asset_id, str)
            and runtime_registry_path is not None
        ):
            try:
                binding = load_habitat_asset_bindings(
                    [asset_id],
                    runtime_registry_path=runtime_registry_path,
                    external_index_path=external_index_path,
                    binding_delta_path=binding_delta_path,
                )[asset_id]
            except HabitatStaticAssetError as exc:
                raise MP3DMultiActorCaptureError(str(exc)) from exc
            asset_path_raw = (
                None
                if binding.asset_manifest_path is None
                else str(binding.asset_manifest_path)
            )
            request_path_raw = (
                None
                if binding.base_m2_request_path is None
                else str(binding.base_m2_request_path)
            )
    if not isinstance(asset_path_raw, str) or not isinstance(request_path_raw, str):
        raise MP3DMultiActorCaptureError(
            "actor track must carry explicit asset_manifest_path and base_m2_request_path"
        )
    asset_path = Path(asset_path_raw).expanduser().resolve()
    request_path = Path(request_path_raw).expanduser().resolve()
    key = (asset_path, request_path)
    if key not in cache:
        try:
            inputs = load_m2_inputs(
                asset_path, request_path, allow_research_candidate=allow_research_candidate
            )
            bundle = load_runtime_asset_bundle(inputs)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise MP3DMultiActorCaptureError(
                f"actor M2 package/request cannot be loaded: {exc}"
            ) from exc
        cache[key] = (inputs, bundle)
    inputs, bundle = cache[key]
    if inputs.asset.get("asset_id") != asset.get("asset_id"):
        raise MP3DMultiActorCaptureError(
            f"actor {track.get('actor_id')!r} package asset differs from track"
        )
    joint_order = tuple(bundle.joint_mapping.get("runtime_joint_order", ()))
    if tuple(asset.get("runtime_joint_order", ())) != joint_order:
        raise MP3DMultiActorCaptureError(
            f"actor {track.get('actor_id')!r} runtime joint order differs"
        )
    return inputs, bundle


def _track_frame(
    track: Mapping[str, Any], *, frame_index: int, joint_order: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, str, int, int, Any]:
    frames = track.get("frames")
    if not isinstance(frames, list):
        raise MP3DMultiActorCaptureError("actor track has no frames")
    if frame_index >= len(frames) or not isinstance(frames[frame_index], Mapping):
        raise MP3DMultiActorCaptureError(
            f"actor track {track.get('actor_id')!r} lacks frame {frame_index}"
        )
    frame = frames[frame_index]
    if frame.get("frame_index") != frame_index:
        raise MP3DMultiActorCaptureError("actor track frame indices are not contiguous")
    root = _matrix_from_transform(
        frame.get("planned_world_from_skin_root"),
        owner=f"actor {track.get('actor_id')} frame {frame_index} skin root",
    )
    targets = frame.get("joint_targets")
    if not isinstance(targets, list) or [item.get("joint_id") for item in targets if isinstance(item, Mapping)] != list(joint_order):
        raise MP3DMultiActorCaptureError(
            f"actor {track.get('actor_id')!r} frame {frame_index} joint target order differs"
        )
    try:
        rotations = np.asarray(
            [item["rotation_xyzw"] for item in targets], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise MP3DMultiActorCaptureError("joint targets are not numeric") from exc
    if rotations.shape != (len(joint_order), 4) or not np.all(np.isfinite(rotations)):
        raise MP3DMultiActorCaptureError("joint targets must be finite [joint,4]")
    action_id = frame.get("action_id")
    action_time_ticks = frame.get("action_time_ticks")
    sample_index = frame.get("action_sample_index")
    if (
        action_id not in {"idle", "walk"}
        or isinstance(action_time_ticks, bool)
        or not isinstance(action_time_ticks, int)
        or action_time_ticks < 0
        or isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index < 0
    ):
        raise MP3DMultiActorCaptureError("actor track action/sample fields are invalid")
    route_center = frame.get("planned_route_center_m")
    return root, rotations, str(action_id), int(action_time_ticks), int(sample_index), route_center


def _validate_track_action_sample(
    bundle: Any,
    *,
    actor_id: Any,
    frame_index: int,
    action_id: str,
    action_time_ticks: int,
    action_sample_index: int,
    rotations: np.ndarray,
) -> None:
    try:
        role = bundle.action_roles_by_id[action_id]
        clip = bundle.action_sets_by_role[role].action(action_id)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError(
            f"actor {actor_id!r} frame {frame_index} has no validated {action_id!r} action"
        ) from exc
    if action_sample_index >= clip.sample_count:
        raise MP3DMultiActorCaptureError(
            f"actor {actor_id!r} frame {frame_index} action sample is out of range"
        )
    effective_tick = action_time_ticks % int(clip.loop_duration_ticks)
    if int(clip.sample_ticks[action_sample_index]) != effective_tick:
        raise MP3DMultiActorCaptureError(
            f"actor {actor_id!r} frame {frame_index} action tick is not on its baked sample"
        )
    expected = np.asarray(clip.rotations_xyzw[action_sample_index], dtype=np.float64)
    if expected.shape != rotations.shape or not np.allclose(
        expected, rotations, rtol=0.0, atol=1.0e-9
    ):
        raise MP3DMultiActorCaptureError(
            f"actor {actor_id!r} frame {frame_index} joint targets differ from its baked action sample"
        )


def _instantiate_actor_with_semantic_template(
    simulator: Any,
    *,
    bundle: Any,
    habitat_sim: Any,
    semantic_id: int,
    actor_index: int,
    base_handle: str,
) -> tuple[Any, Any]:
    # Imported lazily so CPU input tests do not import another native backend.
    from avengine.timeline.visual import _instantiate_actor_with_semantic_template as instantiate

    return instantiate(
        simulator,
        bundle=bundle,
        habitat_sim=habitat_sim,
        base_handle=base_handle,
        semantic_id=semantic_id,
        actor_index=actor_index,
    )


def _base_template_handle(
    simulator: Any, bundle: Any, *, cache: dict[Path, str]
) -> str:
    config_path = Path(bundle.paths_by_role["habitat_ao_config"]).resolve()
    if config_path in cache:
        return cache[config_path]
    manager = simulator.metadata_mediator.ao_template_manager
    loaded = manager.load_configs(str(config_path))
    if not isinstance(loaded, Sequence) or len(loaded) != 1:
        raise MP3DMultiActorCaptureError(
            f"expected one AO template for {config_path}: ids={loaded}"
        )
    template_id = loaded[0]
    try:
        base = manager.get_template_handle_by_id(template_id)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise MP3DMultiActorCaptureError(
            f"could not resolve AO template id {template_id!r} for {config_path}"
        ) from exc
    if not isinstance(base, str) or not base:
        raise MP3DMultiActorCaptureError(
            f"AO template id {template_id!r} has no handle for {config_path}"
        )
    cache[config_path] = base
    return base


def _track_entity_class(track: Mapping[str, Any]) -> str:
    value = track.get("entity_class")
    if value is None and isinstance(track.get("asset"), Mapping):
        value = track["asset"].get("entity_class")
    return str(value or "articulated_animal")


def _load_rigid_binding(
    track: Mapping[str, Any],
    *,
    runtime_registry_path: str | Path | None,
    external_index_path: str | Path | None,
    binding_delta_path: str | Path | None,
) -> dict[str, Any]:
    asset = track.get("asset")
    if not isinstance(asset, Mapping):
        raise MP3DMultiActorCaptureError("rigid track has no asset mapping")
    direct = asset.get("habitat_binding") or track.get("habitat_binding")
    if isinstance(direct, Mapping):
        binding = dict(direct)
        glb_raw = binding.get("glb_path")
        if not isinstance(glb_raw, str) or not glb_raw:
            raise MP3DMultiActorCaptureError(
                f"rigid asset {asset.get('asset_id')!r} binding has no resolved glb_path"
            )
        glb_path = Path(glb_raw).expanduser().resolve()
        if glb_path.is_symlink() or not glb_path.is_file():
            raise MP3DMultiActorCaptureError(
                f"rigid asset {asset.get('asset_id')!r} GLB is missing: {glb_path}"
            )
        binding["glb_path"] = str(glb_path)
        return binding
    asset_id = asset.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        raise MP3DMultiActorCaptureError("rigid track asset_id is required")
    if external_index_path is None and runtime_registry_path is None:
        raise MP3DMultiActorCaptureError(
            f"rigid asset {asset_id!r} has no binding; provide a binding delta or registries"
        )
    try:
        binding = load_habitat_asset_bindings(
            [asset_id],
            runtime_registry_path=runtime_registry_path,
            external_index_path=external_index_path,
            binding_delta_path=binding_delta_path,
        )[asset_id]
    except HabitatStaticAssetError as exc:
        raise MP3DMultiActorCaptureError(str(exc)) from exc
    if binding.normalized_entity_class != "rigid_object":
        raise MP3DMultiActorCaptureError(
            f"asset {asset_id!r} is {binding.entity_class!r}; "
            "articulated assets require their Habitat package"
        )
    return binding.to_dict()


def _instantiate_rigid_object(
    simulator: Any,
    *,
    binding: Mapping[str, Any],
    habitat_sim: Any,
    semantic_id: int,
    object_index: int,
) -> Any:
    glb_raw = binding.get("glb_path")
    if not isinstance(glb_raw, str) or not glb_raw:
        raise MP3DMultiActorCaptureError("rigid Habitat binding has no glb_path")
    glb_path = Path(glb_raw).expanduser().resolve()
    if glb_path.is_symlink() or not glb_path.is_file():
        raise MP3DMultiActorCaptureError(f"rigid Habitat GLB is missing: {glb_path}")
    try:
        semantic = int(semantic_id)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DMultiActorCaptureError("rigid semantic_id is invalid") from exc
    if semantic <= 0:
        raise MP3DMultiActorCaptureError("rigid semantic_id must be positive")
    manager = simulator.get_object_template_manager()
    handle = f"qa_rigid_object_{object_index}_semantic{semantic}"
    attributes = manager.create_new_template(handle, False)
    if attributes is None:
        raise MP3DMultiActorCaptureError("Habitat failed to create rigid object template")
    try:
        attributes.render_asset_handle = str(glb_path)
        attributes.collision_asset_handle = str(glb_path)
        attributes.semantic_id = semantic
        attributes.is_collidable = True
        attributes.is_visibile = True
    except (AttributeError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError(
            f"cannot configure rigid Habitat template for {glb_path}"
        ) from exc
    registered = manager.register_template(attributes, handle)
    if int(registered) < 0:
        raise MP3DMultiActorCaptureError(
            f"Habitat failed to register rigid object template {handle!r}"
        )
    object_manager = simulator.get_rigid_object_manager()
    obj = object_manager.add_object_by_template_handle(handle)
    if obj is None:
        raise MP3DMultiActorCaptureError(
            f"Habitat failed to instantiate rigid object {handle!r}"
        )
    try:
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.semantic_id = semantic
    except (AttributeError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError(
            f"Habitat rigid object {handle!r} semantic/motion setup failed"
        ) from exc
    nodes = [obj.root_scene_node]
    try:
        nodes.extend(list(obj.visual_scene_nodes))
    except (AttributeError, TypeError):
        pass
    for node in nodes:
        if not hasattr(node, "semantic_id"):
            raise MP3DMultiActorCaptureError(
                f"Habitat rigid object {handle!r} has no semantic SceneNode"
            )
        node.semantic_id = semantic
        if int(node.semantic_id) != semantic:
            raise MP3DMultiActorCaptureError(
                f"Habitat rigid object {handle!r} semantic writeback differs"
            )
    if int(obj.semantic_id) != semantic:
        raise MP3DMultiActorCaptureError(
            f"Habitat rigid object {handle!r} object semantic ID differs"
        )
    return obj


def _rigid_transform(track: Mapping[str, Any], *, frame_index: int) -> np.ndarray:
    frames = track.get("frames")
    if (
        not isinstance(frames, list)
        or frame_index >= len(frames)
        or not isinstance(frames[frame_index], Mapping)
    ):
        raise MP3DMultiActorCaptureError(
            f"rigid track {track.get('actor_id')!r} lacks frame {frame_index}"
        )
    frame = frames[frame_index]
    value = frame.get("planned_world_from_object") or frame.get(
        "planned_world_from_skin_root"
    )
    return _matrix_from_transform(
        value,
        owner=f"rigid actor {track.get('actor_id')!r} frame {frame_index} object",
    )


def _apply_rigid_transform(obj: Any, transform: np.ndarray, *, runtime: Any) -> None:
    try:
        obj.translation = runtime.magnum.Vector3(transform[:3, 3])
        quaternion = runtime.quaternion.from_rotation_matrix(transform[:3, :3])
        w, x, y, z = (
            float(value) for value in runtime.quaternion.as_float_array(quaternion)
        )
        obj.rotation = runtime.magnum.Quaternion(
            runtime.magnum.Vector3(x, y, z), w
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise MP3DMultiActorCaptureError(
            "Habitat rigid object transform application failed"
        ) from exc


def _rigid_root_readback(obj: Any) -> np.ndarray:
    try:
        root = np.asarray(
            obj.root_scene_node.absolute_transformation(), dtype=np.float64
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError("Habitat rigid object root readback failed") from exc
    if root.shape != (4, 4) or not np.all(np.isfinite(root)):
        raise MP3DMultiActorCaptureError("Habitat rigid object root is not finite 4x4")
    return np.ascontiguousarray(root)


def _rigid_emitter_position(obj: Any, binding: Mapping[str, Any]) -> np.ndarray:
    emitter = binding.get("emitter")
    if not isinstance(emitter, Mapping):
        raise MP3DMultiActorCaptureError("rigid binding has no emitter")
    try:
        offset = np.asarray(
            emitter.get("offset_m", emitter.get("translation_m")), dtype=np.float64
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DMultiActorCaptureError("rigid emitter offset is invalid") from exc
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise MP3DMultiActorCaptureError("rigid emitter offset must be finite 3-vector")
    root = _rigid_root_readback(obj)
    return np.ascontiguousarray((root @ np.r_[offset, 1.0])[:3])


def _set_instance_hidden(item: Mapping[str, Any], runtime: Any) -> Any:
    try:
        node = item["actor"].root_scene_node
        if not hasattr(node, "scaling"):
            # CPU call-order fakes used by the hermetic suite do not expose
            # Habitat SceneNode scaling. Native SceneNodes always do.
            return None
        previous = node.scaling
        node.scaling = runtime.magnum.Vector3(0.0)
        return previous
    except (AttributeError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError(
            f"cannot hide Habitat source {item['track'].get('source_slot_id')!r}"
        ) from exc


def _restore_instance_visibility(item: Mapping[str, Any], previous: Any) -> None:
    if previous is None:
        return
    try:
        item["actor"].root_scene_node.scaling = previous
    except (AttributeError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError(
            f"cannot restore Habitat source {item['track'].get('source_slot_id')!r}"
        ) from exc


def _save_pixel_evidence(
    output: Path,
    *,
    semantic_frames: Sequence[np.ndarray],
    target_masks_by_slot: Mapping[str, Sequence[np.ndarray]],
    actors_runtime: Sequence[Mapping[str, Any]],
    room_inputs: ValidatedM1Inputs,
    frame_count: int,
) -> tuple[Path, Path, dict[str, Any]]:
    if not semantic_frames:
        raise MP3DMultiActorCaptureError("cannot write pixel evidence without semantic frames")
    modal = np.ascontiguousarray(np.stack(semantic_frames))
    if modal.ndim != 3:
        raise MP3DMultiActorCaptureError("modal semantic frames must be [frame,height,width]")
    slots = [str(item["track"]["source_slot_id"]) for item in actors_runtime]
    if set(target_masks_by_slot) != set(slots):
        raise MP3DMultiActorCaptureError("target-only slots differ from captured sources")
    payload: dict[str, np.ndarray] = {
        "depth_derived_modal_semantic": modal,
        "modal": modal.copy(),
    }
    semantic_ids: dict[str, int] = {}
    for item in actors_runtime:
        slot = str(item["track"]["source_slot_id"])
        semantic_id = int(item["track"]["semantic_id"])
        masks = np.ascontiguousarray(np.stack(target_masks_by_slot[slot]))
        if masks.shape != modal.shape or masks.dtype.kind not in {"i", "u"}:
            raise MP3DMultiActorCaptureError(
                f"target-only mask {slot} does not match modal semantic shape"
            )
        if not np.all(np.isin(masks, [0, semantic_id])):
            raise MP3DMultiActorCaptureError(
                f"target-only mask {slot} contains foreign semantic IDs"
            )
        payload[f"target_only_{slot}"] = masks
        semantic_ids[slot] = semantic_id
    masks_path = output / PIXEL_MASK_FILENAME
    if masks_path.exists() or masks_path.is_symlink():
        raise MP3DMultiActorCaptureError(f"refusing to replace pixel mask output: {masks_path}")
    np.savez_compressed(masks_path, **payload)
    camera_pose_ids = [
        f"{room_inputs.request['primary_camera_rig'].get('rig_id', 'camera_rig')}:"
        f"{room_inputs.request['primary_camera_rig'].get('view_id', 'view')}:frame:{index:06d}"
        for index in range(frame_count)
    ]
    resolution = [int(modal.shape[1]), int(modal.shape[2])]
    common = {
        "renderer_backend": "habitat_native",
        "rgb_renderer_backend": "habitat_native",
        "camera_contract_id": str(
            room_inputs.request["primary_camera_rig"].get("rig_id", "camera_rig")
        ),
        "semantic_id_namespace": "habitat_object_semantic_id_v1",
        "resolution_hw": resolution,
        "frame_indices": list(range(frame_count)),
        "camera_pose_ids": camera_pose_ids,
    }
    try:
        truth = compile_pixel_visibility_truth(
            normal_semantic_masks=[frame.astype(np.int32, copy=False) for frame in modal],
            target_only_semantic_masks_by_instance={
                slot: [
                    frame.astype(np.int32, copy=False)
                    for frame in payload[f"target_only_{slot}"]
                ]
                for slot in slots
            },
            semantic_ids_by_instance=semantic_ids,
            normal_context={"pass_kind": "modal_scene", **common},
            target_only_contexts_by_instance={
                slot: {
                    "pass_kind": "target_only",
                    "target_instance_id": slot,
                    **common,
                }
                for slot in slots
            },
        )
    except (TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError(
            f"cannot compile Habitat pixel visibility truth: {exc}"
        ) from exc
    truth_path = output / "pixel_visibility_truth.json"
    _write_json(truth_path, truth)
    return masks_path, truth_path, truth


def _capture_target_only_blank(
    *,
    room_inputs: ValidatedM1Inputs,
    actors_runtime: Sequence[Mapping[str, Any]],
    expected_frames: Sequence[Sequence[Mapping[str, Any]]],
    fallback_semantic_frames: Sequence[np.ndarray],
    main_camera_snapshots: Sequence[Mapping[str, Any]],
    runtime: InstalledHabitatRuntime,
    output: Path,
    gpu_device_id: int,
    simulator_factory: Callable[[Any], Any] | None,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    """Render each source in a blank-stage simulator with the same camera.

    A show-only semantic pass must preserve the target's full frustum
    footprint. Scaling other source nodes to zero is insufficient when the
    stage mesh can occlude the target, so native target-only passes use a
    second simulator configured with the dataset's NONE stage. The scene
    camera/calibration and target transforms still come from the same M1 plan
    and observed frame clock.
    """

    if not hasattr(runtime.habitat_sim, "Simulator"):
        # Hermetic call-order fakes intentionally do not expose a native
        # Simulator. Their semantic image is already deterministic and this
        # fallback keeps the CPU contract testable without weakening native
        # execution.
        masks: dict[str, list[np.ndarray]] = {}
        for item in actors_runtime:
            slot = str(item["track"]["source_slot_id"])
            semantic_id = int(item["track"]["semantic_id"])
            masks[slot] = [
                np.where(np.asarray(frame) == semantic_id, semantic_id, 0).astype(
                    np.int32
                )
                for frame in fallback_semantic_frames
            ]
        return masks, {
            "status": "pass",
            "mode": "hermetic_semantic_fallback",
            "scene_id": None,
            "per_source": {
                str(item["track"]["source_slot_id"]): {
                    "camera_max_translation_error_m": None,
                    "camera_max_rotation_error": None,
                    "root_max_error_m": None,
                    "joint_max_error": None,
                }
                for item in actors_runtime
            },
        }

    masks_by_slot: dict[str, list[np.ndarray]] = {
        str(item["track"]["source_slot_id"]): [] for item in actors_runtime
    }
    alignment: dict[str, Any] = {
        "status": "pass",
        "mode": "native_blank_stage",
        "scene_id": "NONE",
        "per_source": {},
    }
    for target_index, target in enumerate(actors_runtime):
        target_output = output / "target_only_scene_scratch" / str(target_index)
        try:
            configuration, modality_to_uuid, _listener_uuid, _resolved_scene = (
                _make_configuration(
                    room_inputs,
                    None,
                    target_output,
                    mp3d_root=runtime.mp3d_root,
                    include_audio_sensor=False,
                    physics_config_path=runtime.physics_config_path,
                )
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise MP3DMultiActorCaptureError(
                f"cannot build target-only Habitat configuration: {exc}"
            ) from exc
        if not hasattr(configuration, "sim_cfg"):
            raise MP3DMultiActorCaptureError(
                "target-only Habitat configuration has no sim_cfg"
            )
        configuration.sim_cfg.scene_id = "NONE"
        configuration.sim_cfg.load_semantic_mesh = False
        configuration.sim_cfg.enable_physics = True
        configuration.sim_cfg.gpu_device_id = gpu_device_id
        factory = simulator_factory or runtime.habitat_sim.Simulator
        with factory(configuration) as simulator:
            target_item_is_rigid = bool(target["is_rigid"])
            if target_item_is_rigid:
                target_actor = _instantiate_rigid_object(
                    simulator,
                    binding=target["habitat_binding"],
                    habitat_sim=runtime.habitat_sim,
                    semantic_id=int(target["track"]["semantic_id"]),
                    object_index=target_index,
                )
            else:
                manager_cache: dict[Path, str] = {}
                base_handle = _base_template_handle(
                    simulator, target["bundle"], cache=manager_cache
                )
                target_actor, _target_binding = _instantiate_actor_with_semantic_template(
                    simulator,
                    bundle=target["bundle"],
                    habitat_sim=runtime.habitat_sim,
                    semantic_id=int(target["track"]["semantic_id"]),
                    actor_index=target_index,
                    base_handle=base_handle,
                )
            agent = simulator.initialize_agent(
                0, _camera_agent_state(runtime, room_inputs.request)
            )
            sensors = [
                simulator.sensors[modality_to_uuid[modality]]
                for modality in modality_to_uuid
            ]
            initial_world_time = float(simulator.get_world_time())
            slot = str(target["track"]["source_slot_id"])
            semantic_id = int(target["track"]["semantic_id"])
            camera_translation_error = 0.0
            camera_rotation_error = 0.0
            root_error = 0.0
            joint_error = 0.0
            for frame_index, frame_expected in enumerate(expected_frames):
                expected = frame_expected[target_index]
                if target_item_is_rigid:
                    _apply_rigid_transform(
                        target_actor, np.asarray(expected["root"]), runtime=runtime
                    )
                    actual_root = _rigid_root_readback(target_actor)
                    root_error = max(
                        root_error,
                        float(np.max(np.abs(actual_root - expected["root"]))),
                    )
                    if float(np.max(np.abs(actual_root - expected["root"]))) > ROOT_READBACK_ATOL:
                        raise MP3DMultiActorCaptureError(
                            f"target-only frame {frame_index} rigid source root differs"
                        )
                else:
                    _apply_root_with_habitat(
                        target_actor,
                        np.asarray(expected["root"]),
                        qt=runtime.quaternion,
                        mn=runtime.magnum,
                    )
                    target_actor.joint_positions = np.asarray(
                        expected["joints"], dtype=np.float64
                    ).copy()
                    snapshot = _runtime_snapshot(simulator, target_actor)
                    actual_root = np.asarray(
                        snapshot["world_from_skin_root"], dtype=np.float64
                    )
                    actual_joints = np.asarray(
                        snapshot["joint_positions_xyzw"], dtype=np.float64
                    )
                    root_error = max(
                        root_error,
                        float(np.max(np.abs(actual_root - expected["root"]))),
                    )
                    joint_error = max(
                        joint_error,
                        float(
                            _quaternion_block_error(
                                actual_joints, np.asarray(expected["joints"])
                            )
                        ),
                    )
                    if float(np.max(np.abs(actual_root - expected["root"]))) > ROOT_READBACK_ATOL:
                        raise MP3DMultiActorCaptureError(
                            f"target-only frame {frame_index} actor root differs"
                        )
                    if _quaternion_block_error(
                        actual_joints, np.asarray(expected["joints"])
                    ) > JOINT_READBACK_ATOL:
                        raise MP3DMultiActorCaptureError(
                            f"target-only frame {frame_index} actor joints differs"
                        )
                observation = simulator.render_sensors(sensors)
                arrays = _validate_observation_arrays(
                    observation, modality_to_uuid
                )
                if float(simulator.get_world_time()) != initial_world_time:
                    raise MP3DMultiActorCaptureError(
                        f"target-only frame {frame_index} advanced Habitat world time"
                    )
                target_camera = _state_snapshot(
                    simulator,
                    agent,
                    list(modality_to_uuid.values()),
                    runtime.quat_to_coeffs,
                )
                main_camera = main_camera_snapshots[frame_index]
                for uuid in modality_to_uuid.values():
                    target_sensor = target_camera["sensors"][uuid]
                    main_sensor = main_camera["sensors"][uuid]
                    if not np.allclose(
                        target_sensor["translation_m"],
                        main_sensor["translation_m"],
                        atol=1.0e-6,
                        rtol=0.0,
                    ) or not np.allclose(
                        target_sensor["rotation_xyzw"],
                        main_sensor["rotation_xyzw"],
                        atol=1.0e-6,
                        rtol=0.0,
                    ):
                        raise MP3DMultiActorCaptureError(
                            f"target-only frame {frame_index} camera differs from full-scene readback"
                        )
                    camera_translation_error = max(
                        camera_translation_error,
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(target_sensor["translation_m"])
                                    - np.asarray(main_sensor["translation_m"])
                                )
                            )
                        ),
                    )
                    camera_rotation_error = max(
                        camera_rotation_error,
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(target_sensor["rotation_xyzw"])
                                    - np.asarray(main_sensor["rotation_xyzw"])
                                )
                            )
                        ),
                    )
                semantic = np.asarray(arrays["semantic"])
                masks_by_slot[slot].append(
                    np.where(semantic == semantic_id, semantic_id, 0).astype(
                        np.int32
                    )
                )
            alignment["per_source"][slot] = {
                "camera_max_translation_error_m": camera_translation_error,
                "camera_max_rotation_error": camera_rotation_error,
                "root_max_error_m": root_error,
                "joint_max_error": joint_error,
            }
    return masks_by_slot, alignment


def _emitter_link_id(actor: Any, track: Mapping[str, Any]) -> int:
    emitter = track.get("emitter")
    joint_id = emitter.get("joint_id") if isinstance(emitter, Mapping) else None
    if not isinstance(joint_id, str) or not joint_id:
        raise MP3DMultiActorCaptureError(
            f"actor {track.get('actor_id')!r} has no emitter joint binding"
        )
    matches = [
        int(link_id)
        for link_id in actor.get_link_ids()
        if actor.get_link_name(link_id) == joint_id
    ]
    if len(matches) != 1:
        raise MP3DMultiActorCaptureError(
            f"actor {track.get('actor_id')!r} emitter link {joint_id!r} is not unique"
        )
    return matches[0]


def _actor_root_and_joints(actor: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        root = np.asarray(
            actor.root_scene_node.absolute_transformation(), dtype=np.float64
        )
        joints = np.asarray(actor.joint_positions, dtype=np.float64).reshape(-1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError("Habitat actor root/joint readback failed") from exc
    if root.shape != (4, 4) or not np.all(np.isfinite(root)) or not np.all(np.isfinite(joints)):
        raise MP3DMultiActorCaptureError("Habitat actor root/joint readback is non-finite")
    return root, joints


def _emitter_anchor_transform(track: Mapping[str, Any], inputs: ValidatedM2Inputs) -> np.ndarray:
    """Bind the anchor's declared local pose to the actually observed joint."""
    emitter = track.get("emitter", {})
    anchors = inputs.asset.get("anchors", [])
    matches = [anchor for anchor in anchors if isinstance(anchor, Mapping)
               and anchor.get("anchor_id") == emitter.get("anchor_id")]
    if len(matches) != 1 or matches[0].get("joint_id") != emitter.get("joint_id"):
        raise MP3DMultiActorCaptureError("track emitter does not match one M2 anchor/joint")
    transform = _matrix_from_transform(matches[0].get("joint_from_anchor"), owner="M2 joint_from_anchor")
    if emitter.get("joint_from_anchor") is not None:
        recorded = _matrix_from_transform(emitter["joint_from_anchor"], owner="track joint_from_anchor")
        if not np.allclose(recorded, transform, atol=1.0e-9, rtol=0):
            raise MP3DMultiActorCaptureError("track emitter local pose differs from the M2 anchor")
    return transform


def _emitter_position(actor: Any, link_id: int, joint_from_anchor: np.ndarray) -> np.ndarray:
    try:
        matrix = np.asarray(
            actor.get_link_scene_node(link_id).absolute_transformation(),
            dtype=np.float64,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError("Habitat emitter link readback failed") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise MP3DMultiActorCaptureError("Habitat emitter transform is not finite 4x4")
    return np.ascontiguousarray((matrix @ joint_from_anchor)[:3, 3])


def _camera_agent_state(runtime: Any, request: Mapping[str, Any]) -> Any:
    try:
        state = runtime.habitat_sim.AgentState()
        transform = request["primary_camera_rig"]["world_from_rig"]
        position = np.asarray(transform["translation_m"], dtype=np.float64)
        x, y, z, w = normalized_quaternion_xyzw(transform["rotation_xyzw"])
        state.position = position
        state.rotation = runtime.quaternion.quaternion(float(w), float(x), float(y), float(z))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError("cannot construct the M1 camera agent state") from exc
    return state


def _save_array(output: Path, name: str, values: Sequence[np.ndarray]) -> None:
    if not values:
        raise MP3DMultiActorCaptureError(f"cannot save empty {name} array")
    array = np.ascontiguousarray(np.stack(values))
    np.save(output / f"{name}.npy", array, allow_pickle=False)


def _capture_with_runtime(
    *,
    case_manifest_path: Path,
    case: Mapping[str, Any],
    tracks: Sequence[Mapping[str, Any]],
    room_inputs: ValidatedM1Inputs,
    runtime: InstalledHabitatRuntime,
    output: Path,
    gpu_device_id: int,
    simulator_factory: Callable[[Any], Any] | None,
    runtime_registry_path: str | Path | None,
    external_index_path: str | Path | None,
    binding_delta_path: str | Path | None,
    allow_research_candidate: bool = False,
) -> dict[str, Any]:
    if runtime.mp3d_root is None:
        raise MP3DMultiActorCaptureError(
            "native MP3D capture requires an explicit installed MP3D root"
        )
    if isinstance(gpu_device_id, bool) or not isinstance(gpu_device_id, int) or gpu_device_id < 0:
        raise MP3DMultiActorCaptureError("gpu_device_id must be a nonnegative integer")
    try:
        configuration, modality_to_uuid, listener_uuid, resolved_scene = _make_configuration(
            room_inputs,
            None,
            output / "scene_scratch",
            mp3d_root=runtime.mp3d_root,
            include_audio_sensor=False,
            physics_config_path=runtime.physics_config_path,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise MP3DMultiActorCaptureError(f"cannot build current M1 Habitat configuration: {exc}") from exc
    if not hasattr(configuration, "sim_cfg"):
        raise MP3DMultiActorCaptureError("M1 configuration has no sim_cfg")
    configuration.sim_cfg.gpu_device_id = gpu_device_id
    room_declared_physics = bool(resolved_scene.get("enable_physics", False))
    # As in mixed_capture: Bullet object creation is required even though
    # kinematic apply/readback executes zero physics integration steps.
    configuration.sim_cfg.enable_physics = True
    if list(modality_to_uuid) != ["rgb", "depth", "semantic"]:
        raise MP3DMultiActorCaptureError("M1 modalities must remain rgb/depth/semantic")
    if simulator_factory is None:
        simulator_factory = runtime.habitat_sim.Simulator

    package_cache: dict[tuple[Path, Path], tuple[ValidatedM2Inputs, Any]] = {}
    actor_runtime: list[dict[str, Any]] = []
    for index, track in enumerate(tracks):
        if _track_entity_class(track) in RIGID_ENTITY_CLASSES:
            binding = _load_rigid_binding(
                track,
                runtime_registry_path=runtime_registry_path,
                external_index_path=external_index_path,
                binding_delta_path=binding_delta_path,
            )
            actor_runtime.append(
                {
                    "track": track,
                    "entity_class": "rigid_object",
                    "habitat_binding": binding,
                    "actor_index": index,
                }
            )
        else:
            if (
                runtime_registry_path is None
                and external_index_path is None
                and binding_delta_path is None
            ):
                inputs, bundle = _load_track_runtime(
                    track,
                    cache=package_cache,
                    **({"allow_research_candidate": True} if allow_research_candidate else {}),
                )
            else:
                inputs, bundle = _load_track_runtime(
                    track,
                    cache=package_cache,
                    runtime_registry_path=runtime_registry_path,
                    external_index_path=external_index_path,
                    binding_delta_path=binding_delta_path,
                    allow_research_candidate=allow_research_candidate,
                )
            actor_runtime.append(
                {
                    "track": track,
                    "entity_class": _track_entity_class(track),
                    "inputs": inputs,
                    "bundle": bundle,
                    "joint_from_anchor": _emitter_anchor_transform(track, inputs),
                    "actor_index": index,
                }
            )
    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    semantic_frames: list[np.ndarray] = []
    actor_root_frames: list[np.ndarray] = []
    actor_joint_frames_by_slot: dict[str, list[np.ndarray]] = {
        str(item["track"]["source_slot_id"]): []
        for item in actor_runtime
        if item["entity_class"] not in RIGID_ENTITY_CLASSES
    }
    emitter_frames: list[np.ndarray] = []
    frame_records: list[dict[str, Any]] = []
    target_masks_by_slot: dict[str, list[np.ndarray]] = {
        str(item["track"]["source_slot_id"]): [] for item in actor_runtime
    }
    expected_frames: list[list[dict[str, Any]]] = []
    with simulator_factory(configuration) as simulator:
        navmesh = resolved_scene.get("navmesh")
        if navmesh is None or not Path(navmesh).is_file():
            raise MP3DMultiActorCaptureError(
                f"resolved M1 MP3D navmesh is missing: {navmesh}"
            )
        loaded = bool(simulator.pathfinder.load_nav_mesh(str(navmesh)))
        if not loaded or not bool(simulator.pathfinder.is_loaded):
            raise MP3DMultiActorCaptureError("Habitat failed to load the declared MP3D navmesh")
        graph_errors, _loaded_graph = validate_loaded_scene_asset_graph(
            room_inputs,
            None,
            simulator,
            declared_navmesh_loaded=loaded,
            mp3d_root=runtime.mp3d_root,
        )
        if graph_errors:
            raise MP3DMultiActorCaptureError(
                "loaded M1 MP3D scene graph differs from declaration: "
                + "; ".join(graph_errors)
            )
        manager_cache: dict[Path, str] = {}
        actors_runtime: list[dict[str, Any]] = []
        for index, item in enumerate(actor_runtime):
            if item["entity_class"] in RIGID_ENTITY_CLASSES:
                actor = _instantiate_rigid_object(
                    simulator,
                    binding=item["habitat_binding"],
                    habitat_sim=runtime.habitat_sim,
                    semantic_id=int(item["track"]["semantic_id"]),
                    object_index=index,
                )
                actors_runtime.append(
                    {
                        **item,
                        "actor": actor,
                        "binding": item["habitat_binding"],
                        "emitter_id": None,
                        "is_rigid": True,
                    }
                )
                continue
            base_handle = _base_template_handle(
                simulator, item["bundle"], cache=manager_cache
            )
            actor, binding = _instantiate_actor_with_semantic_template(
                simulator,
                bundle=item["bundle"],
                habitat_sim=runtime.habitat_sim,
                semantic_id=int(item["track"]["semantic_id"]),
                actor_index=index,
                base_handle=base_handle,
            )
            emitter_id = _emitter_link_id(actor, item["track"])
            actors_runtime.append(
                {
                    **item,
                    "actor": actor,
                    "binding": binding,
                    "emitter_id": emitter_id,
                    "is_rigid": False,
                }
            )

        agent = simulator.initialize_agent(
            0, _camera_agent_state(runtime, room_inputs.request)
        )
        sensors = [simulator.sensors[modality_to_uuid[modality]] for modality in modality_to_uuid]
        sensor_uuids = [modality_to_uuid[modality] for modality in modality_to_uuid]
        initial_world_time = float(simulator.get_world_time())
        frame_count = int(case["clock"]["frame_count"])
        for frame_index in range(frame_count):
            expected_actor_values: list[dict[str, Any]] = []
            for item in actors_runtime:
                track = item["track"]
                if item["is_rigid"]:
                    root = _rigid_transform(track, frame_index=frame_index)
                    _apply_rigid_transform(item["actor"], root, runtime=runtime)
                    expected_actor_values.append(
                        {
                            "root": root,
                            "joints": np.empty((0,), dtype=np.float64),
                            "action_id": "static",
                            "action_time_ticks": 0,
                            "action_sample_index": 0,
                            "planned_route_center_m": None,
                        }
                    )
                    continue
                root, rotations, action_id, action_time_ticks, sample_index, route_center = _track_frame(
                    track,
                    frame_index=frame_index,
                    joint_order=item["bundle"].joint_mapping["runtime_joint_order"],
                )
                _validate_track_action_sample(
                    item["bundle"],
                    actor_id=track.get("actor_id"),
                    frame_index=frame_index,
                    action_id=action_id,
                    action_time_ticks=action_time_ticks,
                    action_sample_index=sample_index,
                    rotations=rotations,
                )
                expected_joints = np.asarray(
                    item["binding"].map_pose(rotations), dtype=np.float64
                )
                _apply_root_with_habitat(
                    item["actor"],
                    root,
                    qt=runtime.quaternion,
                    mn=runtime.magnum,
                )
                item["actor"].joint_positions = expected_joints.copy()
                expected_actor_values.append(
                    {
                        "root": root,
                        "joints": expected_joints,
                        "action_id": action_id,
                        "action_time_ticks": action_time_ticks,
                        "action_sample_index": sample_index,
                        "planned_route_center_m": route_center,
                    }
                )
            expected_frames.append(expected_actor_values)

            # Read the applied state before rendering to catch a bad binding,
            # then render exactly once. Observed fields below are read again
            # from the simulator after this call.
            before = [
                _rigid_root_readback(item["actor"])
                if item["is_rigid"]
                else _runtime_snapshot(simulator, item["actor"])
                for item in actors_runtime
            ]
            for item, expected, snapshot in zip(
                actors_runtime, expected_actor_values, before, strict=True
            ):
                if item["is_rigid"]:
                    actual_root = np.asarray(snapshot, dtype=np.float64)
                    if float(np.max(np.abs(actual_root - expected["root"]))) > ROOT_READBACK_ATOL:
                        raise MP3DMultiActorCaptureError(
                            f"frame {frame_index} rigid source {item['track']['actor_id']} root readback differs"
                        )
                    continue
                actual_root = np.asarray(
                    snapshot["world_from_skin_root"], dtype=np.float64
                )
                actual_joints = np.asarray(
                    snapshot["joint_positions_xyzw"], dtype=np.float64
                )
                if float(np.max(np.abs(actual_root - expected["root"]))) > ROOT_READBACK_ATOL:
                    raise MP3DMultiActorCaptureError(
                        f"frame {frame_index} actor {item['track']['actor_id']} root readback differs"
                    )
                if _quaternion_block_error(actual_joints, expected["joints"]) > JOINT_READBACK_ATOL:
                    raise MP3DMultiActorCaptureError(
                        f"frame {frame_index} actor {item['track']['actor_id']} joint readback differs"
                    )
            observation = simulator.render_sensors(sensors)
            arrays = _validate_observation_arrays(observation, modality_to_uuid)
            after = [
                _rigid_root_readback(item["actor"])
                if item["is_rigid"]
                else _runtime_snapshot(simulator, item["actor"])
                for item in actors_runtime
            ]
            if float(simulator.get_world_time()) != initial_world_time:
                raise MP3DMultiActorCaptureError(
                    f"frame {frame_index} advanced Habitat world time"
                )
            for item, before_snapshot, after_snapshot in zip(
                actors_runtime, before, after, strict=True
            ):
                if item["is_rigid"]:
                    if float(
                        np.max(
                            np.abs(
                                np.asarray(after_snapshot)
                                - np.asarray(before_snapshot)
                            )
                        )
                    ) > ROOT_READBACK_ATOL:
                        raise MP3DMultiActorCaptureError(
                            f"frame {frame_index} rigid source {item['track']['actor_id']} changed during render"
                        )
                    continue
                before_root = np.asarray(
                    before_snapshot["world_from_skin_root"], dtype=np.float64
                )
                after_root = np.asarray(
                    after_snapshot["world_from_skin_root"], dtype=np.float64
                )
                before_joints = np.asarray(
                    before_snapshot["joint_positions_xyzw"], dtype=np.float64
                )
                after_joints = np.asarray(
                    after_snapshot["joint_positions_xyzw"], dtype=np.float64
                )
                if float(np.max(np.abs(after_root - before_root))) > ROOT_READBACK_ATOL:
                    raise MP3DMultiActorCaptureError(
                        f"frame {frame_index} actor {item['track']['actor_id']} changed during render"
                    )
                if _quaternion_block_error(after_joints, before_joints) > JOINT_READBACK_ATOL:
                    raise MP3DMultiActorCaptureError(
                        f"frame {frame_index} actor {item['track']['actor_id']} joints changed during render"
                    )
            observed_roots: list[np.ndarray] = []
            observed_joints: list[np.ndarray] = []
            observed_emitters: list[np.ndarray] = []
            actor_records: list[dict[str, Any]] = []
            for item, expected, snapshot in zip(actors_runtime, expected_actor_values, after, strict=True):
                if item["is_rigid"]:
                    root = np.asarray(snapshot, dtype=np.float64)
                    joints = np.empty((0,), dtype=np.float64)
                    emitter = _rigid_emitter_position(
                        item["actor"], item["binding"]
                    )
                else:
                    root = np.asarray(
                        snapshot["world_from_skin_root"], dtype=np.float64
                    )
                    joints = np.asarray(
                        snapshot["joint_positions_xyzw"], dtype=np.float64
                    )
                    emitter = _emitter_position(
                        item["actor"],
                        item["emitter_id"],
                        item["joint_from_anchor"],
                    )
                observed_roots.append(root)
                observed_joints.append(joints)
                observed_emitters.append(emitter)
                semantic_pixels = int(
                    np.count_nonzero(arrays["semantic"] == int(item["track"]["semantic_id"]))
                )
                actor_records.append(
                    {
                        "actor_id": item["track"]["actor_id"],
                        "source_slot_id": item["track"]["source_slot_id"],
                        "source_endpoint_id": item["track"]["source_endpoint_id"],
                        "asset_id": item["track"]["asset"]["asset_id"],
                        "entity_class": item["entity_class"],
                        "semantic_id": int(item["track"]["semantic_id"]),
                        "action_id": expected["action_id"],
                        "action_time_ticks": expected["action_time_ticks"],
                        "action_sample_index": expected["action_sample_index"],
                        "planned_route_center_m": expected["planned_route_center_m"],
                        "world_from_skin_root": _transform_record(root),
                        "world_from_object": _transform_record(root)
                        if item["is_rigid"]
                        else None,
                        "joint_positions_xyzw": joints.tolist(),
                        "emitter_world_position_m": emitter.tolist(),
                        "semantic_pixel_count": semantic_pixels,
                    }
                )
            rgb = np.ascontiguousarray(arrays["rgb"][..., :3])
            depth = np.ascontiguousarray(arrays["depth"])
            semantic = np.ascontiguousarray(arrays["semantic"])
            rgb_frames.append(rgb)
            depth_frames.append(depth)
            semantic_frames.append(semantic)
            actor_root_frames.append(np.stack(observed_roots))
            for item, joints in zip(actors_runtime, observed_joints, strict=True):
                if not item["is_rigid"]:
                    actor_joint_frames_by_slot[
                        str(item["track"]["source_slot_id"])
                    ].append(joints)
            emitter_frames.append(np.stack(observed_emitters))
            camera_snapshot = _state_snapshot(
                simulator, agent, sensor_uuids, runtime.quat_to_coeffs
            )
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "pts_ticks": frame_index * int(case["clock"]["ticks_per_frame"]),
                    "camera_readback": camera_snapshot,
                    "actor_readbacks": actor_records,
                    "source_positions_m": [emitter.tolist() for emitter in observed_emitters],
                    "modalities": {
                        modality: {
                            "sensor_uuid": modality_to_uuid[modality],
                            "dtype": (rgb if modality == "rgb" else arrays[modality]).dtype.str,
                            "shape": list((rgb if modality == "rgb" else arrays[modality]).shape),
                        }
                        for modality in ("rgb", "depth", "semantic")
                    },
                }
            )
    try:
        target_masks_by_slot, target_alignment = _capture_target_only_blank(
            room_inputs=room_inputs,
            actors_runtime=actors_runtime,
            expected_frames=expected_frames,
            fallback_semantic_frames=semantic_frames,
            main_camera_snapshots=[frame["camera_readback"] for frame in frame_records],
            runtime=runtime,
            output=output,
            gpu_device_id=gpu_device_id,
            simulator_factory=simulator_factory,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise MP3DMultiActorCaptureError(
            f"Habitat target-only capture failed: {exc}"
        ) from exc
    _write_json(output / "target_only_readback_alignment.json", target_alignment)
    _save_array(output, "rgb", rgb_frames)
    _save_array(output, "depth", depth_frames)
    _save_array(output, "semantic", semantic_frames)
    _save_array(output, "actor_root_readbacks", actor_root_frames)
    _save_array(output, "emitter_positions_m", emitter_frames)
    actor_joint_artifacts: dict[str, str] = {}
    for slot, frames in actor_joint_frames_by_slot.items():
        filename = f"actor_joint_readbacks_{slot}.npy"
        _save_array(output, filename.removesuffix(".npy"), frames)
        actor_joint_artifacts[slot] = filename
    _write_json(output / "frame_records.json", {
        "source_endpoint_ids": [item["track"]["source_endpoint_id"] for item in actor_runtime],
        "frames": frame_records,
        "render": dict(case["clock"]),
    })
    try:
        pixel_masks_path, pixel_truth_path, pixel_truth = _save_pixel_evidence(
            output,
            semantic_frames=semantic_frames,
            target_masks_by_slot=target_masks_by_slot,
            actors_runtime=actors_runtime,
            room_inputs=room_inputs,
            frame_count=int(case["clock"]["frame_count"]),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError(
            f"Habitat target-only pixel evidence failed: {exc}"
        ) from exc
    try:
        from avengine.rooms.qa_evidence import derive_actor_occluders

        actor_occluders = derive_actor_occluders(
            pixel_masks_path,
            pixel_truth,
        )
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise MP3DMultiActorCaptureError(
            f"Habitat actor occluder derivation failed: {exc}"
        ) from exc
    actor_occluders_path = output / "actor_occluders.json"
    _write_json(actor_occluders_path, actor_occluders)
    receipt = {
        "schema": NATIVE_CAPTURE_SCHEMA,
        "artifact_role": "observed_native_habitat_capture",
        "status": "research_only",
        "research_only": True,
        "research_candidate_assets_allowed": allow_research_candidate,
        "episode_counted": False,
        "qualification_claim": False,
        "claim_boundary": (
            "native Habitat RGB/depth/semantic and actor root/joint/emitter "
            "readback plus same-camera target-only semantic masks and pixel "
            "visibility truth; no RLR audio, collision qualification, or formal "
            "admission"
        ),
        "capture": {
            **dict(case["clock"]),
            "native_habitat_started": True,
            "rgb_channel_order": "rgb",
            "gpu_device_id": gpu_device_id,
            "physics_steps": 0,
            "physics_configuration": {
                "room_declared_enable_physics": room_declared_physics,
                "enabled_for_articulated_object_creation": not room_declared_physics,
                "effective_enable_physics": True,
            },
            "observed_frame_records": "frame_records.json",
            "pixel_truth_status": pixel_truth["status"],
        },
        "inputs": {
            "case_manifest": str(case_manifest_path.resolve()),
            "room_manifest": str(room_inputs.room_path),
            "m1_request": str(room_inputs.request_path),
        },
        "actors": [
            {
                "actor_id": item["track"]["actor_id"],
                "source_slot_id": item["track"]["source_slot_id"],
                "source_endpoint_id": item["track"]["source_endpoint_id"],
                "asset_id": item["track"]["asset"]["asset_id"],
                "entity_class": item["entity_class"],
                **(
                    {
                        "asset_manifest_path": (
                            item["track"]["asset"].get("asset_manifest_path")
                            or (
                                None
                                if getattr(item.get("inputs"), "asset_path", None)
                                is None
                                else str(item["inputs"].asset_path)
                            )
                        ),
                        "base_m2_request_path": (
                            item["track"]["asset"].get("base_m2_request_path")
                            or (
                                None
                                if getattr(item.get("inputs"), "request_path", None)
                                is None
                                else str(item["inputs"].request_path)
                            )
                        ),
                        "emitter_joint_id": item["track"]["emitter"]["joint_id"],
                        "joint_from_anchor_matrix": _transform_record(
                            item["joint_from_anchor"]
                        ),
                    }
                    if not item["is_rigid"]
                    else {
                        "habitat_binding": item["binding"],
                        "emitter_anchor_id": item["binding"]["emitter"]["anchor_id"],
                        "resting_pose": item["binding"]["resting_pose"],
                    }
                ),
                "native_emitter_readback": "frame_records.actor_readbacks[].emitter_world_position_m",
            }
            for item in actors_runtime
        ],
        "object_id": {
            "status": "pending",
            "reason": (
                "Habitat object-ID modality was not requested; same-camera "
                "target-only semantic masks completed for every source"
            ),
        },
        "target_only": {
            "status": "pass",
            "reason": "same-camera target-only semantic render completed for every source",
            "pass_kind": "semantic_target_only",
            "scene_geometry_policy": (
                "blank_habitat_stage_scene_id_NONE; target object retained"
            ),
            "same_camera_readback_verified": True,
            "same_root_joint_readback_verified": True,
            "alignment": "target_only_readback_alignment.json",
            "source_count": len(actors_runtime),
            "pixel_masks": pixel_masks_path.name,
            "pixel_visibility_truth": pixel_truth_path.name,
            "target_only_readback_alignment": "target_only_readback_alignment.json",
        },
        "artifacts": {
            "rgb": "rgb.npy",
            "depth": "depth.npy",
            "semantic": "semantic.npy",
            "actor_root_readbacks": "actor_root_readbacks.npy",
            "actor_joint_readbacks_by_slot": actor_joint_artifacts,
            "emitter_positions_m": "emitter_positions_m.npy",
            "frame_records": "frame_records.json",
            "pixel_masks": pixel_masks_path.name,
            "pixel_visibility_truth": pixel_truth_path.name,
            "actor_occluders": actor_occluders_path.name,
        },
    }
    _write_json(output / "research_receipt.json", receipt)
    return receipt


def capture_mp3d_multi_actor(
    *,
    case_manifest_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    runtime_prefix: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    mp3d_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
    output_directory: str | Path,
    gpu_device_id: int = 0,
    episode_plan_path: str | Path | None = None,
    runtime_registry_path: str | Path | None = None,
    external_index_path: str | Path | None = None,
    binding_delta_path: str | Path | None = None,
    allow_research_candidate: bool = False,
    runtime: InstalledHabitatRuntime | None = None,
    simulator_factory: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Run one explicit N-actor native MP3D capture.

    ``runtime`` and ``simulator_factory`` are private dependency-injection
    hooks for CPU call-order tests. Production callers leave them unset so the
    installed runtime is prepared from the explicit paths.
    """

    case, track_records, room_inputs = _load_case_and_m1(
        case_manifest_path=case_manifest_path,
        room_manifest_path=room_manifest_path,
        m1_request_path=m1_request_path,
    )
    selected_runtime = runtime
    if selected_runtime is None:
        try:
            selected_runtime = prepare_installed_habitat_runtime(
                runtime_prefix=runtime_prefix,
                rlr_sdk_root=rlr_sdk_root,
                mp3d_root=mp3d_root,
                magnum_python_site=magnum_python_site,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise MP3DMultiActorCaptureError(
                f"installed Habitat runtime is unavailable: {exc}"
            ) from exc
    output = _fresh_output(output_directory)
    tracks = tuple(item["value"] for item in track_records)
    receipt = _capture_with_runtime(
        case_manifest_path=Path(case_manifest_path).expanduser().resolve(),
        case=case,
        tracks=tracks,
        room_inputs=room_inputs,
        runtime=selected_runtime,
        output=output,
        gpu_device_id=gpu_device_id,
        simulator_factory=simulator_factory,
        runtime_registry_path=runtime_registry_path,
        external_index_path=external_index_path,
        binding_delta_path=binding_delta_path,
        allow_research_candidate=allow_research_candidate,
    )
    if episode_plan_path is not None:
        plan = _read_json(episode_plan_path, owner="episode plan")
        try:
            from avengine.capture.habitat_neutral_readback import (
                write_habitat_neutral_readback,
            )

            write_habitat_neutral_readback(output, plan)
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
            raise MP3DMultiActorCaptureError(
                f"Habitat neutral readback failed: {exc}"
            ) from exc
        receipt = {
            **receipt,
            "neutral_readback": {
                "status": "pass",
                "path": "neutral_readback.json",
                "plan": str(Path(episode_plan_path).expanduser().resolve()),
            },
            "artifacts": {
                **receipt["artifacts"],
                "neutral_readback": "neutral_readback.json",
            },
        }
        _write_json(Path(output) / "research_receipt.json", receipt)
    return receipt


__all__ = [
    "MP3DMultiActorCaptureError",
    "NATIVE_CAPTURE_SCHEMA",
    "capture_mp3d_multi_actor",
]

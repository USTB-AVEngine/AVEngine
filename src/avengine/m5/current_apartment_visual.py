"""Standalone current Apartment RGB research author and capture path.

This is intentionally separate from the retained M6/M7 Apartment writers.
It owns only an external A2 actor selection, a freely authored 75-frame
visual timeline, and native SPEAR RGB research capture. It does not read a
formal source bundle, qualification, dry audio, RLR, or a historical reader.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.backends.spear_ue.research_runtime import (
    close_scene_capture,
    launch_external_game_instance,
    read_actor_pose,
    read_rgb_bgr,
    run_frame_transaction,
    spawn_attached_visual_actor,
    spawn_scene_capture,
    warm_scene_capture_until_stable,
)
from avengine.backends.spear_ue.launch import (
    validate_current_production_spear_executable,
)
from avengine.optional_backends.spear_apartment import (
    ANIMATION_TOLERANCE_SECONDS,
    animation_position_seconds,
    apply_ue_component_frame_delta,
    summarize_root_readbacks,
)
from avengine.runtime_profiles import (
    load_source_asset_runtime_registry,
    resolve_source_asset_runtime_profile,
)


FRAME_COUNT = 75
FRAME_RATE_HZ = 15
TICKS_PER_FRAME = 3_200
CAMERA_BLUEPRINT = "/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C"
CAPTURE_COMPONENT_NAME = "DefaultSceneRoot.final_tone_curve_hdr_"
NATIVE_APARTMENT_MAP = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"


class CurrentApartmentVisualError(RuntimeError):
    """A current Apartment visual-research request is incomplete or unsafe."""


def _checkout_ancestor(path: Path) -> Path | None:
    candidate = path.resolve()
    while True:
        if os.path.lexists(candidate / ".git"):
            return candidate
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent


def _external_directory(value: str | Path, *, owner: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise CurrentApartmentVisualError(f"{owner} must be absolute")
    if raw.is_symlink():
        raise CurrentApartmentVisualError(f"{owner} must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise CurrentApartmentVisualError(
            f"{owner} cannot be resolved: {error}"
        ) from error
    if raw != resolved or not resolved.is_dir():
        raise CurrentApartmentVisualError(
            f"{owner} must be an existing canonical directory without a symlink hop"
        )
    checkout = _checkout_ancestor(resolved)
    if checkout is not None:
        raise CurrentApartmentVisualError(
            f"{owner} must be outside a Git checkout (found {checkout})"
        )
    return resolved


def _external_file(value: str | Path, *, owner: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise CurrentApartmentVisualError(f"{owner} must be absolute")
    if raw.is_symlink():
        raise CurrentApartmentVisualError(f"{owner} must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise CurrentApartmentVisualError(
            f"{owner} cannot be resolved: {error}"
        ) from error
    if raw != resolved or not resolved.is_file():
        raise CurrentApartmentVisualError(
            f"{owner} must be an existing canonical regular file without a symlink hop"
        )
    checkout = _checkout_ancestor(resolved)
    if checkout is not None:
        raise CurrentApartmentVisualError(
            f"{owner} must be outside a Git checkout (found {checkout})"
        )
    return resolved


def _read_mapping(path: Path, *, owner: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CurrentApartmentVisualError(f"cannot read {owner}: {error}") from error
    if not isinstance(value, dict):
        raise CurrentApartmentVisualError(f"{owner} must be a JSON object")
    return value


def _new_external_output_file(value: str | Path, *, owner: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise CurrentApartmentVisualError(f"{owner} must be absolute")
    parent = raw.parent
    if parent.exists():
        parent = _external_directory(parent, owner=f"{owner} parent")
    else:
        existing = parent
        while not existing.exists():
            if existing == existing.parent:
                raise CurrentApartmentVisualError(f"{owner} has no existing ancestor")
            existing = existing.parent
        _external_directory(existing, owner=f"{owner} existing parent")
        parent.mkdir(parents=True, exist_ok=False)
        parent = _external_directory(parent, owner=f"{owner} parent")
    output = parent / raw.name
    if os.path.lexists(output):
        raise CurrentApartmentVisualError(f"refusing to replace {owner}: {output}")
    return output


def _new_external_output_directory(value: str | Path, *, owner: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise CurrentApartmentVisualError(f"{owner} must be absolute")
    parent = raw.parent
    if parent.exists():
        parent = _external_directory(parent, owner=f"{owner} parent")
    else:
        existing = parent
        while not existing.exists():
            if existing == existing.parent:
                raise CurrentApartmentVisualError(f"{owner} has no existing ancestor")
            existing = existing.parent
        _external_directory(existing, owner=f"{owner} existing parent")
        parent.mkdir(parents=True, exist_ok=False)
        parent = _external_directory(parent, owner=f"{owner} parent")
    output = parent / raw.name
    if os.path.lexists(output):
        raise CurrentApartmentVisualError(f"refusing to replace {owner}: {output}")
    output.mkdir()
    return output


def _finite_triplet(value: object, *, owner: str) -> list[float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise CurrentApartmentVisualError(f"{owner} must contain three finite numbers")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CurrentApartmentVisualError(f"{owner}[{index}] must be finite")
        number = float(item)
        if not math.isfinite(number):
            raise CurrentApartmentVisualError(f"{owner}[{index}] must be finite")
        result.append(number)
    return result


def _finite_number(value: object, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CurrentApartmentVisualError(f"{owner} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise CurrentApartmentVisualError(f"{owner} must be finite")
    return result


def _package_from_object_path(value: object, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise CurrentApartmentVisualError(f"{owner} must be a nonempty object path")
    package = value.split(".", 1)[0]
    if not package.startswith("/Game/"):
        raise CurrentApartmentVisualError(f"{owner} must use a /Game object path")
    return package


def _selection_bindings(
    *,
    actor_selection_path: str | Path,
    source_asset_registry_path: str | Path,
) -> tuple[Path, dict[str, dict[str, Any]], str]:
    selection_path = _external_file(actor_selection_path, owner="--actor-selection")
    selection = _read_mapping(selection_path, owner="actor selection")
    if selection.get("research_only") is False:
        raise CurrentApartmentVisualError("actor selection may not claim formal status")
    actors = selection.get("actors")
    if not isinstance(actors, list):
        raise CurrentApartmentVisualError("actor selection actors must be a list")
    source_registry = load_source_asset_runtime_registry(source_asset_registry_path)
    result: dict[str, dict[str, Any]] = {}
    for raw in actors:
        if not isinstance(raw, Mapping):
            raise CurrentApartmentVisualError("actor selection actor must be an object")
        slot = raw.get("source_slot_id")
        asset_id = raw.get("asset_id")
        revision = raw.get("revision")
        if slot not in {"source1", "source2"} or slot in result:
            raise CurrentApartmentVisualError(
                "actor selection must contain unique source1/source2 slots"
            )
        if not isinstance(asset_id, str) or not isinstance(revision, str):
            raise CurrentApartmentVisualError(
                "actor selection asset identity is invalid"
            )
        record = resolve_source_asset_runtime_profile(
            source_registry, asset_id, revision
        )
        raw_binding = record.get("runtime_backends", {}).get("spear_unreal")
        declared = raw.get("ue_binding")
        if not isinstance(raw_binding, Mapping) or not isinstance(declared, Mapping):
            raise CurrentApartmentVisualError(
                f"actor selection {slot} has no complete SPEAR binding"
            )
        required_equal = (
            ("blueprint_object_path", "blueprint_class_path"),
            ("profile_skeletal_mesh_binding", "skeletal_mesh_binding"),
            ("profile_skeletal_mesh_path", "skeletal_mesh_path"),
            ("idle_object_path", "idle_animation"),
            ("walking_object_path", "walking_animation"),
        )
        for manifest_key, registry_key in required_equal:
            if declared.get(manifest_key) != raw_binding.get(registry_key):
                raise CurrentApartmentVisualError(
                    f"actor selection {slot} {manifest_key} differs from registry"
                )
        anatomical_forward_yaw_deg = raw_binding.get("ue_anatomical_forward_yaw_deg")
        if (
            isinstance(anatomical_forward_yaw_deg, bool)
            or not isinstance(anatomical_forward_yaw_deg, (int, float))
            or not math.isfinite(float(anatomical_forward_yaw_deg))
        ):
            raise CurrentApartmentVisualError(
                f"actor selection {slot} has an invalid anatomical forward yaw"
            )
        source_timeline = record.get("timeline")
        if not isinstance(source_timeline, Mapping):
            raise CurrentApartmentVisualError(
                f"actor selection {slot} has no Timeline semantics"
            )
        walk_phase_period_frames = source_timeline.get("walk_phase_period_frames")
        if (
            isinstance(walk_phase_period_frames, bool)
            or not isinstance(walk_phase_period_frames, int)
            or walk_phase_period_frames < 1
        ):
            raise CurrentApartmentVisualError(
                f"actor selection {slot} has an invalid walk phase period"
            )
        graph_mesh = declared.get("graph_derived_mesh")
        if not isinstance(graph_mesh, Mapping):
            raise CurrentApartmentVisualError(
                f"actor selection {slot} lacks graph-derived mesh provenance"
            )
        mesh_package = graph_mesh.get("package")
        mesh_object_path = graph_mesh.get("object_path")
        if (
            not isinstance(mesh_package, str)
            or not mesh_package.startswith("/Game/")
            or not isinstance(mesh_object_path, str)
            or _package_from_object_path(
                mesh_object_path, owner=f"actor selection {slot} graph mesh"
            )
            != mesh_package
        ):
            raise CurrentApartmentVisualError(
                f"actor selection {slot} graph-derived mesh is invalid"
            )
        result[slot] = {
            "source_slot_id": slot,
            "actor_id": f"{slot}_actor",
            "asset_id": asset_id,
            "revision": revision,
            "blueprint_class_path": str(raw_binding["blueprint_class_path"]),
            "idle_animation": str(raw_binding["idle_animation"]),
            "walking_animation": str(raw_binding["walking_animation"]),
            "graph_mesh_package": mesh_package,
            "graph_mesh_object_path": mesh_object_path,
            "walk_phase_period_frames": walk_phase_period_frames,
            "ue_anatomical_forward_yaw_deg": float(anatomical_forward_yaw_deg),
            "component_frame_delta": dict(raw_binding["ue_component_frame_delta"]),
            "anatomical_basis_bones": raw_binding.get("ue_anatomical_basis_bones"),
        }
    if set(result) != {"source1", "source2"}:
        raise CurrentApartmentVisualError(
            "actor selection must contain exactly source1 and source2"
        )
    authorization = selection.get("asset_authorization")
    if authorization not in {"verified_internal", "unverified"}:
        authorization = "unverified"
    return selection_path, result, authorization


def _actor_yaw(
    start: Sequence[float],
    end: Sequence[float],
    *,
    anatomical_forward_yaw_deg: float,
) -> float:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    if math.hypot(dx, dy) <= 1.0e-9:
        return 0.0
    desired_world_yaw = math.degrees(math.atan2(dy, dx))
    yaw = (desired_world_yaw - anatomical_forward_yaw_deg + 180.0) % 360.0 - 180.0
    return 0.0 if yaw == 0.0 else yaw


def _interpolate(
    start: Sequence[float], end: Sequence[float], fraction: float
) -> list[float]:
    return [
        float(start[index] + (end[index] - start[index]) * fraction)
        for index in range(3)
    ]


def _timeline_state(
    *,
    binding: Mapping[str, Any],
    start: Sequence[float],
    end: Sequence[float],
    frame_index: int,
    walk_start_frame: int = 0,
) -> dict[str, Any]:
    moving = any(
        abs(float(end[index]) - float(start[index])) > 1.0e-9 for index in range(3)
    )
    walk_start = walk_start_frame
    action_id = "walk" if moving and frame_index >= walk_start else "idle"
    period = int(binding["walk_phase_period_frames"])
    phase = (
        float((frame_index - walk_start) % period) / float(period)
        if action_id == "walk"
        else 0.0
    )
    if moving and frame_index >= walk_start:
        walk_end = FRAME_COUNT - 1
        if frame_index == walk_end:
            translation = [float(value) for value in end]
        elif frame_index == walk_start:
            translation = [float(value) for value in start]
        else:
            fraction = float(frame_index - walk_start) / float(walk_end - walk_start)
            translation = _interpolate(start, end, fraction)
    else:
        translation = [float(value) for value in start]
    return {
        "actor_id": binding["actor_id"],
        "source_slot_id": binding["source_slot_id"],
        "asset_id": binding["asset_id"],
        "revision": binding["revision"],
        "walk_phase_period_frames": binding["walk_phase_period_frames"],
        "translation_ue_cm": translation,
        "yaw_ue_deg": _actor_yaw(
            start,
            end,
            anatomical_forward_yaw_deg=float(binding["ue_anatomical_forward_yaw_deg"]),
        ),
        "action_id": action_id,
        "action_phase": phase,
    }


def author_current_apartment_visual_timeline(
    *,
    actor_selection_path: str | Path,
    source_asset_registry_path: str | Path,
    output_path: str | Path,
    camera_position_ue_cm: Sequence[float],
    camera_yaw_deg: float,
    human_start_ue_cm: Sequence[float],
    human_end_ue_cm: Sequence[float],
    beagle_start_ue_cm: Sequence[float],
    beagle_end_ue_cm: Sequence[float],
    width: int = 1280,
    height: int = 720,
    hfov_degrees: float = 105.0,
    walk_start_frame: int = 0,
) -> dict[str, Any]:
    """Write one freely designed current Apartment visual-only timeline."""

    selection_file, bindings, asset_authorization = _selection_bindings(
        actor_selection_path=actor_selection_path,
        source_asset_registry_path=source_asset_registry_path,
    )
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width < 1
        or height < 1
        or not 0.0 < hfov_degrees < 180.0
    ):
        raise CurrentApartmentVisualError(
            "timeline render dimensions or HFOV are invalid"
        )
    if (
        isinstance(walk_start_frame, bool)
        or not isinstance(walk_start_frame, int)
        or not 0 <= walk_start_frame < FRAME_COUNT - 1
    ):
        raise CurrentApartmentVisualError(
            "walk_start_frame must be an integer in [0, 73]"
        )
    camera_position = _finite_triplet(
        camera_position_ue_cm, owner="camera_position_ue_cm"
    )
    camera_yaw = _finite_number(camera_yaw_deg, owner="camera_yaw_deg")
    starts = {
        "source1": _finite_triplet(human_start_ue_cm, owner="human_start_ue_cm"),
        "source2": _finite_triplet(beagle_start_ue_cm, owner="beagle_start_ue_cm"),
    }
    ends = {
        "source1": _finite_triplet(human_end_ue_cm, owner="human_end_ue_cm"),
        "source2": _finite_triplet(beagle_end_ue_cm, owner="beagle_end_ue_cm"),
    }
    frames = []
    for frame_index in range(FRAME_COUNT):
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * TICKS_PER_FRAME,
                "camera": {
                    "translation_ue_cm": list(camera_position),
                    "yaw_ue_deg": camera_yaw,
                },
                "actor_states": [
                    _timeline_state(
                        binding=bindings[slot],
                        start=starts[slot],
                        end=ends[slot],
                        frame_index=frame_index,
                        walk_start_frame=walk_start_frame,
                    )
                    for slot in ("source1", "source2")
                ],
            }
        )
    timeline = {
        "kind": "current_apartment_visual_research_timeline",
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "claim_boundary": (
            "freely designed external UE RGB research timeline only; it carries "
            "no audio, RLR, M6/M7 source bundle, room qualification, or dataset claim"
        ),
        "actor_selection": str(selection_file),
        "room": {
            "map_path": NATIVE_APARTMENT_MAP,
            "room_profile_id": "spear_apartment_0000",
        },
        "render": {
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FRAME_RATE_HZ,
            "ticks_per_frame": TICKS_PER_FRAME,
            "resolution_hw": [height, width],
            "hfov_degrees": float(hfov_degrees),
            "walk_start_frame": walk_start_frame,
        },
        "actors": [
            {
                key: binding[key]
                for key in (
                    "source_slot_id",
                    "actor_id",
                    "asset_id",
                    "revision",
                    "walk_phase_period_frames",
                    "ue_anatomical_forward_yaw_deg",
                    "blueprint_class_path",
                    "idle_animation",
                    "walking_animation",
                    "graph_mesh_package",
                    "graph_mesh_object_path",
                )
            }
            for binding in (bindings["source1"], bindings["source2"])
        ],
        "asset_authorization": asset_authorization,
        "spatial_validation": "not_run",
        "frames": frames,
    }
    output = _new_external_output_file(output_path, owner="timeline output")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(timeline, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return timeline


def _load_timeline(
    *,
    timeline_path: str | Path,
    bindings: Mapping[str, Mapping[str, Any]],
    asset_authorization: str,
) -> tuple[Path, dict[str, Any]]:
    path = _external_file(timeline_path, owner="--timeline")
    timeline = _read_mapping(path, owner="research timeline")
    render = timeline.get("render")
    if (
        timeline.get("status") != "research_only"
        or timeline.get("research_only") is not True
        or timeline.get("episode_counted") is not False
        or timeline.get("qualification_claim") is not False
        or timeline.get("asset_authorization") != asset_authorization
        or not isinstance(render, Mapping)
        or render.get("frame_count") != FRAME_COUNT
        or render.get("frame_rate_hz") != FRAME_RATE_HZ
        or render.get("ticks_per_frame") != TICKS_PER_FRAME
    ):
        raise CurrentApartmentVisualError(
            "timeline must be a 75-frame 15fps non-counted research record"
        )
    actors = timeline.get("actors")
    if not isinstance(actors, list) or len(actors) != 2:
        raise CurrentApartmentVisualError("timeline must declare exactly two actors")
    actor_by_slot = {
        actor.get("source_slot_id"): actor
        for actor in actors
        if isinstance(actor, Mapping)
    }
    if set(actor_by_slot) != set(bindings):
        raise CurrentApartmentVisualError(
            "timeline actor slots differ from actor selection"
        )
    for slot, binding in bindings.items():
        actor = actor_by_slot[slot]
        for field in (
            "actor_id",
            "asset_id",
            "revision",
            "walk_phase_period_frames",
            "ue_anatomical_forward_yaw_deg",
            "blueprint_class_path",
            "idle_animation",
            "walking_animation",
            "graph_mesh_package",
            "graph_mesh_object_path",
        ):
            if actor.get(field) != binding[field]:
                raise CurrentApartmentVisualError(
                    f"timeline {slot} {field} differs from actor selection"
                )
    frames = timeline.get("frames")
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise CurrentApartmentVisualError("timeline must contain exactly 75 frames")
    for frame_index, frame in enumerate(frames):
        if (
            not isinstance(frame, Mapping)
            or frame.get("frame_index") != frame_index
            or frame.get("pts_ticks") != frame_index * TICKS_PER_FRAME
        ):
            raise CurrentApartmentVisualError(
                f"timeline frame {frame_index} has an invalid clock"
            )
        camera = frame.get("camera")
        if not isinstance(camera, Mapping):
            raise CurrentApartmentVisualError(
                f"timeline frame {frame_index} lacks camera"
            )
        _finite_triplet(
            camera.get("translation_ue_cm"),
            owner=f"timeline frame {frame_index} camera position",
        )
        _finite_number(
            camera.get("yaw_ue_deg"), owner=f"timeline frame {frame_index} camera yaw"
        )
        states = frame.get("actor_states")
        if not isinstance(states, list) or len(states) != 2:
            raise CurrentApartmentVisualError(
                f"timeline frame {frame_index} must contain two actor states"
            )
        state_slots = [
            state.get("source_slot_id")
            for state in states
            if isinstance(state, Mapping)
        ]
        if state_slots != ["source1", "source2"]:
            raise CurrentApartmentVisualError(
                f"timeline frame {frame_index} actor order must be source1/source2"
            )
        for state in states:
            assert isinstance(state, Mapping)
            slot = state["source_slot_id"]
            binding = bindings[slot]
            for field in (
                "actor_id",
                "asset_id",
                "revision",
                "walk_phase_period_frames",
            ):
                if state.get(field) != binding[field]:
                    raise CurrentApartmentVisualError(
                        f"timeline frame {frame_index} {slot} {field} differs"
                    )
            _finite_triplet(
                state.get("translation_ue_cm"),
                owner=f"timeline frame {frame_index} {slot} position",
            )
            _finite_number(
                state.get("yaw_ue_deg"),
                owner=f"timeline frame {frame_index} {slot} yaw",
            )
            if state.get("action_id") not in {"idle", "walk"}:
                raise CurrentApartmentVisualError(
                    f"timeline frame {frame_index} {slot} action is invalid"
                )
            phase = _finite_number(
                state.get("action_phase"),
                owner=f"timeline frame {frame_index} {slot} action phase",
            )
            if not 0.0 <= phase < 1.0:
                raise CurrentApartmentVisualError(
                    f"timeline frame {frame_index} {slot} action phase is invalid"
                )
    return path, timeline


def _closure_mappings(
    *,
    closure_report_path: str | Path,
    bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, list[tuple[str, str]]]:
    path = _external_file(closure_report_path, owner="--closure-report")
    report = _read_mapping(path, owner="closure report")
    variants = report.get("variants")
    if not isinstance(variants, Mapping):
        raise CurrentApartmentVisualError("closure report has no variants")
    required = {
        NATIVE_APARTMENT_MAP,
        "/SpContent/Blueprints/BP_CameraSensor",
        *(
            value
            for binding in bindings.values()
            for value in (
                _package_from_object_path(
                    binding["blueprint_class_path"], owner="selected blueprint"
                ),
                binding["graph_mesh_package"],
                _package_from_object_path(
                    binding["idle_animation"], owner="selected idle animation"
                ),
                _package_from_object_path(
                    binding["walking_animation"], owner="selected walking animation"
                ),
            )
        ),
    }
    candidates = []
    for name, raw in variants.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            continue
        mappings = raw.get("physical_mappings")
        if raw.get("mapping_complete") is not True or not isinstance(mappings, list):
            continue
        by_package = {
            item.get("package"): item
            for item in mappings
            if isinstance(item, Mapping) and isinstance(item.get("package"), str)
        }
        if required <= set(by_package):
            candidates.append((len(by_package), name, by_package))
    if not candidates:
        raise CurrentApartmentVisualError(
            "closure report has no complete variant for map, camera, and selected actors"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise CurrentApartmentVisualError(
            "closure report has ambiguous minimal variants"
        )
    by_package = candidates[0][2]
    result = []
    for package, mapping in sorted(by_package.items()):
        if mapping.get("status") != "unique_authorized_external_input":
            raise CurrentApartmentVisualError(
                f"closure package {package} has no unique external source"
            )
        source_file = mapping.get("source_file")
        if not isinstance(source_file, str):
            raise CurrentApartmentVisualError(
                f"closure package {package} has no source file"
            )
        suffix = Path(source_file).suffix
        if suffix not in {".uasset", ".umap"}:
            raise CurrentApartmentVisualError(
                f"closure package {package} has an invalid primary suffix"
            )
        result.append((package, suffix))
    return path, result


def _validate_stage(
    *,
    stage_root: str | Path,
    spear_executable: str | Path,
    closure_mappings: Sequence[tuple[str, str]],
) -> tuple[Path, Path]:
    stage = _external_directory(stage_root, owner="--stage-root")
    project = stage / "SpearSim/SpearSim.uproject"
    plugin = stage / "plugins/SpContent/SpContent.uplugin"
    for value, owner in ((project, "stage uproject"), (plugin, "SpContent descriptor")):
        if not value.is_file() or value.is_symlink():
            raise CurrentApartmentVisualError(f"{owner} is missing from --stage-root")
    project_record = _read_mapping(project, owner="stage uproject")
    if project_record.get("AdditionalPluginDirectories") != ["../plugins"]:
        raise CurrentApartmentVisualError(
            "stage uproject must retain AdditionalPluginDirectories=[../plugins]"
        )
    plugins = project_record.get("Plugins")
    if not isinstance(plugins, list) or not any(
        isinstance(item, Mapping)
        and item.get("Name") == "SpContent"
        and item.get("Enabled") is True
        for item in plugins
    ):
        raise CurrentApartmentVisualError("stage uproject must enable SpContent")
    if (
        _read_mapping(plugin, owner="SpContent descriptor").get("CanContainContent")
        is not True
    ):
        raise CurrentApartmentVisualError("SpContent descriptor must contain content")
    for package, suffix in closure_mappings:
        if package.startswith("/Game/"):
            expected = (
                stage / "SpearSim/Content" / (package.removeprefix("/Game/") + suffix)
            )
        elif package.startswith("/SpContent/"):
            expected = (
                stage
                / "plugins/SpContent/Content"
                / (package.removeprefix("/SpContent/") + suffix)
            )
        else:
            raise CurrentApartmentVisualError(
                f"closure has unsupported package {package}"
            )
        if not expected.is_file() or expected.is_symlink():
            raise CurrentApartmentVisualError(
                f"stage is missing closure package {package}: {expected}"
            )
        try:
            expected.resolve().relative_to(stage)
        except ValueError as error:
            raise CurrentApartmentVisualError(
                f"closure package {package} escapes --stage-root"
            ) from error
    executable = validate_current_production_spear_executable(
        Path(spear_executable).expanduser()
    )
    try:
        executable.relative_to(stage)
    except ValueError as error:
        raise CurrentApartmentVisualError(
            "--spear-executable must be contained by --stage-root"
        ) from error
    return stage, executable


def _skeletal_mesh_handle(component: Any) -> int:
    try:
        value = component.GetSkeletalMeshAsset(as_handle=True)
    except Exception:
        value = 0
    if not value:
        value = component.get_property_value(
            property_name="SkeletalMesh", as_handle=True
        )
    if isinstance(value, bool) or int(value) <= 0:
        raise CurrentApartmentVisualError("skeletal mesh readback is invalid")
    return int(value)


def _spawn_runtime_actors(
    *,
    game: Any,
    bindings: Mapping[str, Mapping[str, Any]],
    initial_frame: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    states = {
        state["source_slot_id"]: state
        for state in initial_frame["actor_states"]
        if isinstance(state, Mapping)
    }
    runtimes = {}
    for slot in ("source1", "source2"):
        binding = bindings[slot]
        state = states[slot]
        runtime = spawn_attached_visual_actor(
            game,
            actor_id=str(binding["actor_id"]),
            blueprint_class_path=str(binding["blueprint_class_path"]),
            position_ue_cm=list(state["translation_ue_cm"]),
            yaw_ue_degrees=float(state["yaw_ue_deg"]),
        )
        declaration = {
            "actor_id": binding["actor_id"],
            "ue_component_frame_delta": binding["component_frame_delta"],
        }
        apply_ue_component_frame_delta(runtime["visual_root"], declaration)
        expected_mesh = int(
            game.unreal_service.load_object(
                uclass="USkeletalMesh",
                name=binding["graph_mesh_object_path"],
                as_handle=True,
            )
        )
        if _skeletal_mesh_handle(runtime["component"]) != expected_mesh:
            raise CurrentApartmentVisualError(
                f"{slot} spawned Blueprint does not use the selected graph mesh"
            )
        animations = {
            "idle": game.unreal_service.load_object(
                uclass="UAnimationAsset", name=binding["idle_animation"]
            ),
            "walk": game.unreal_service.load_object(
                uclass="UAnimationAsset", name=binding["walking_animation"]
            ),
        }
        lengths = {
            key: float(value.GetPlayLength()) for key, value in animations.items()
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in lengths.values()):
            raise CurrentApartmentVisualError(f"{slot} animations are invalid")
        runtime.update(
            {
                "binding": binding,
                "animations": animations,
                "lengths": lengths,
                "current_action": None,
            }
        )
        runtimes[slot] = runtime
    return runtimes


def _apply_runtime_state(
    runtime: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    frame_index: int,
) -> dict[str, Any]:
    action_id = str(state["action_id"])
    component = runtime["component"]
    if runtime["current_action"] != action_id:
        component.PlayAnimation(
            NewAnimToPlay=runtime["animations"][action_id], bLooping=True
        )
        runtime["current_action"] = action_id
    component.Stop()
    requested = animation_position_seconds(
        float(state["action_phase"]), float(runtime["lengths"][action_id])
    )
    component.SetPosition(InPos=requested, bFireNotifies=False)
    observed = float(component.GetPosition())
    if abs(observed - requested) > ANIMATION_TOLERANCE_SECONDS:
        raise CurrentApartmentVisualError(
            f"frame {frame_index} animation readback differs for {state['source_slot_id']}"
        )
    position = state["translation_ue_cm"]
    runtime["anchor"].K2_SetActorLocationAndRotation(
        NewLocation={"X": position[0], "Y": position[1], "Z": position[2]},
        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": state["yaw_ue_deg"]},
        bSweep=False,
        bTeleport=True,
    )
    return {
        "frame_index": frame_index,
        "source_slot_id": state["source_slot_id"],
        "action_id": action_id,
        "action_phase": float(state["action_phase"]),
        "requested_position_seconds": requested,
        "observed_position_seconds": observed,
        "absolute_error_seconds": abs(observed - requested),
    }


def _destroy_runtime_actors(
    instance: Any, runtimes: Mapping[str, Mapping[str, Any]]
) -> None:
    with instance.begin_frame():
        for runtime in runtimes.values():
            runtime["visual_actor"].K2_DestroyActor()
        for runtime in runtimes.values():
            runtime["anchor"].K2_DestroyActor()
    with instance.end_frame():
        pass


def _expected_root_readback_frames(
    timeline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Adapt this research timeline to the shared UE root-readback summary."""

    expected_frames = []
    for frame in timeline["frames"]:
        camera = frame["camera"]
        expected_frames.append(
            {
                "frame_index": frame["frame_index"],
                "pts_ticks": frame["pts_ticks"],
                "camera_state": {
                    "frame_index": frame["frame_index"],
                    "ue_position_cm": list(camera["translation_ue_cm"]),
                    "ue_yaw_deg": float(camera["yaw_ue_deg"]),
                },
                "actor_states": [
                    {
                        **dict(state),
                        "actor_yaw_ue_deg": float(state["yaw_ue_deg"]),
                    }
                    for state in frame["actor_states"]
                ],
            }
        )
    return expected_frames


def _animation_readback_summary(
    records_by_slot: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize observed animation positions without adding a new contract."""

    actors: dict[str, dict[str, Any]] = {}
    for slot in ("source1", "source2"):
        records = list(records_by_slot[slot])
        if not records:
            raise CurrentApartmentVisualError(
                "animation readback summary has no records for %s" % slot
            )
        errors = [float(record["absolute_error_seconds"]) for record in records]
        actors[slot] = {
            "status": "pass",
            "frame_count": len(records),
            "maximum_absolute_error_seconds": max(errors),
        }
    return {
        "status": "pass",
        "frame_count": min(value["frame_count"] for value in actors.values()),
        "actors": actors,
    }


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_partial_frame_records(
    output: Path, frame_records: Sequence[Mapping[str, Any]]
) -> bool:
    if not frame_records:
        return False
    path = output / "frame_records.json"
    if path.exists():
        return path.is_file()
    try:
        _write_new_json(path, {"frames": list(frame_records)})
    except (OSError, TypeError, ValueError):
        return False
    return True


def _partial_capture_receipt(
    *,
    output: Path,
    selection_file: Path,
    timeline_file: Path,
    closure_file: Path,
    stage: Path,
    executable: Path,
    asset_authorization: str,
    frame_records: Sequence[Mapping[str, Any]],
    camera_readbacks: Sequence[Mapping[str, Any]],
    actor_readbacks: Mapping[str, Sequence[Mapping[str, Any]]],
    animation_records_by_slot: Mapping[str, Sequence[Mapping[str, Any]]],
    root_readback_summary: Mapping[str, Any] | None,
    error: BaseException,
) -> dict[str, Any]:
    capture: dict[str, Any] = {
        "frame_count": FRAME_COUNT,
        "completed_frame_count": len(frame_records),
        "frame_rate_hz": FRAME_RATE_HZ,
        "ticks_per_frame": TICKS_PER_FRAME,
        "modalities": ["rgb"],
        "audio_requested": False,
        "rlr_requested": False,
        "qualification_requested": False,
        "m6_m7_bundle_requested": False,
        "observed_camera_frame_count": len(camera_readbacks),
        "observed_actor_frame_counts": {
            actor_id: len(records) for actor_id, records in actor_readbacks.items()
        },
        "observed_animation_frame_counts": {
            slot: len(records) for slot, records in animation_records_by_slot.items()
        },
    }
    if root_readback_summary is not None:
        capture["root_readback_summary"] = dict(root_readback_summary)
    artifacts = {}
    if (output / "frame_records.json").is_file():
        artifacts["frame_records"] = "frame_records.json"
    return {
        "status": "fail",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "partial": True,
        "asset_authorization": asset_authorization,
        "error_type": type(error).__name__,
        "error_text": str(error),
        "failure": {
            "error_type": type(error).__name__,
            "error_text": str(error),
        },
        "inputs": {
            "actor_selection": str(selection_file),
            "timeline": str(timeline_file),
            "closure_report": str(closure_file),
            "stage_root": str(stage),
            "spear_executable": str(executable),
        },
        "capture": capture,
        "artifacts": artifacts,
        "claim_boundary": (
            "external SPEAR RGB research only; this failed or partial run does "
            "not count an episode or establish an audio, room, asset, or dataset claim"
        ),
    }


def capture_current_apartment_visual(
    *,
    actor_selection_path: str | Path,
    source_asset_registry_path: str | Path,
    timeline_path: str | Path,
    closure_report_path: str | Path,
    stage_root: str | Path,
    spear_executable: str | Path,
    output_directory: str | Path,
    rpc_port: int = 39511,
    graphics_adapter: int | None = None,
) -> dict[str, Any]:
    """Run a preflighted native SPEAR RGB-only research capture."""

    selection_file, bindings, asset_authorization = _selection_bindings(
        actor_selection_path=actor_selection_path,
        source_asset_registry_path=source_asset_registry_path,
    )
    timeline_file, timeline = _load_timeline(
        timeline_path=timeline_path,
        bindings=bindings,
        asset_authorization=asset_authorization,
    )
    if asset_authorization != "verified_internal":
        output = _new_external_output_directory(
            output_directory, owner="capture output"
        )
        receipt = {
            "status": "not_run",
            "research_only": True,
            "episode_counted": False,
            "qualification_claim": False,
            "asset_authorization": asset_authorization,
            "reason": (
                "selected MyAssets provenance is unverified; no SPEAR launch or "
                "final-content stage is permitted"
            ),
            "inputs": {
                "actor_selection": str(selection_file),
                "timeline": str(timeline_file),
            },
            "claim_boundary": (
                "external SPEAR RGB research only; no dry audio, RLR, M6/M7 "
                "bundle, source qualification, Topdown, or dataset claim"
            ),
        }
        _write_new_json(output / "research_receipt.json", receipt)
        return receipt
    closure_file, mappings = _closure_mappings(
        closure_report_path=closure_report_path, bindings=bindings
    )
    stage, executable = _validate_stage(
        stage_root=stage_root,
        spear_executable=spear_executable,
        closure_mappings=mappings,
    )
    output = _new_external_output_directory(output_directory, owner="capture output")
    instance = None
    game = None
    camera = None
    capture = None
    runtimes: dict[str, dict[str, Any]] = {}
    rgb_frames: list[np.ndarray] = []
    frame_records: list[dict[str, Any]] = []
    camera_readbacks: list[dict[str, Any]] = []
    actor_readbacks: dict[str, list[dict[str, Any]]] = {
        str(binding["actor_id"]): [] for binding in bindings.values()
    }
    animation_records_by_slot: dict[str, list[dict[str, Any]]] = {
        "source1": [],
        "source2": [],
    }
    root_readback_summary: dict[str, Any] | None = None
    success_receipt: dict[str, Any] | None = None
    run_error: BaseException | None = None
    run_traceback = None
    cleanup_error: BaseException | None = None
    try:
        instance = launch_external_game_instance(
            spear_executable=executable,
            native_map=NATIVE_APARTMENT_MAP,
            frame_rate_hz=FRAME_RATE_HZ,
            rpc_port=rpc_port,
            graphics_adapter=graphics_adapter,
        )
        game = instance.get_game()
        render = timeline["render"]
        first_frame = timeline["frames"][0]
        with instance.begin_frame():
            camera, capture = spawn_scene_capture(
                game,
                camera_blueprint=CAMERA_BLUEPRINT,
                component_name=CAPTURE_COMPONENT_NAME,
                width=int(render["resolution_hw"][1]),
                height=int(render["resolution_hw"][0]),
                hfov_degrees=float(render["hfov_degrees"]),
            )
            runtimes = _spawn_runtime_actors(
                game=game, bindings=bindings, initial_frame=first_frame
            )
            camera_state = first_frame["camera"]
            camera.K2_SetActorLocationAndRotation(
                NewLocation={
                    "X": camera_state["translation_ue_cm"][0],
                    "Y": camera_state["translation_ue_cm"][1],
                    "Z": camera_state["translation_ue_cm"][2],
                },
                NewRotation={
                    "Roll": 0.0,
                    "Pitch": 0.0,
                    "Yaw": camera_state["yaw_ue_deg"],
                },
                bSweep=False,
                bTeleport=True,
            )
        with instance.end_frame():
            pass
        capture_warmup = warm_scene_capture_until_stable(instance, capture)
        for frame_index, frame in enumerate(timeline["frames"]):
            animation_readbacks: dict[str, dict[str, Any]] = {}

            def apply() -> None:
                camera_state = frame["camera"]
                camera.K2_SetActorLocationAndRotation(
                    NewLocation={
                        "X": camera_state["translation_ue_cm"][0],
                        "Y": camera_state["translation_ue_cm"][1],
                        "Z": camera_state["translation_ue_cm"][2],
                    },
                    NewRotation={
                        "Roll": 0.0,
                        "Pitch": 0.0,
                        "Yaw": camera_state["yaw_ue_deg"],
                    },
                    bSweep=False,
                    bTeleport=True,
                )
                for state in frame["actor_states"]:
                    slot = state["source_slot_id"]
                    animation_readbacks[slot] = _apply_runtime_state(
                        runtimes[slot],
                        state=state,
                        frame_index=frame_index,
                    )

            def readback() -> dict[str, Any]:
                return {
                    "rgb": read_rgb_bgr(capture),
                    "camera_pose": read_actor_pose(camera),
                    "actor_anchor_poses": {
                        slot: read_actor_pose(runtimes[slot]["anchor"])
                        for slot in ("source1", "source2")
                    },
                    "animation_readbacks": dict(animation_readbacks),
                }

            transaction = run_frame_transaction(
                instance, apply=apply, readback=readback
            )
            image = transaction["rgb"]
            if image.shape[:2] != tuple(render["resolution_hw"]):
                raise CurrentApartmentVisualError(
                    f"frame {frame_index} RGB shape differs from timeline"
                )
            camera_pose = transaction["camera_pose"]
            actor_anchor_poses = transaction["actor_anchor_poses"]
            observed_animations = transaction["animation_readbacks"]
            camera_readbacks.append(
                {
                    "frame_index": frame_index,
                    **camera_pose,
                }
            )
            observed_actor_poses: dict[str, dict[str, Any]] = {}
            for state in frame["actor_states"]:
                slot = state["source_slot_id"]
                actor_id = str(state["actor_id"])
                pose = actor_anchor_poses[slot]
                actor_readbacks[actor_id].append({"frame_index": frame_index, **pose})
                animation_record = observed_animations[slot]
                animation_records_by_slot[slot].append(animation_record)
                observed_actor_poses[slot] = {
                    "actor_id": actor_id,
                    **pose,
                }
            observed_animation_records = [
                observed_animations[slot] for slot in ("source1", "source2")
            ]
            frame_record = {
                "frame_index": frame_index,
                "pts_ticks": frame["pts_ticks"],
                "observation_calls": 1,
                "camera_pose": camera_pose,
                "actor_anchor_poses": observed_actor_poses,
                "animation_readbacks": observed_animation_records,
                "observed": {
                    "camera_pose": camera_pose,
                    "actor_anchor_poses": observed_actor_poses,
                    "animation_readbacks": observed_animation_records,
                },
                "camera": dict(frame["camera"]),
                "actor_states": [
                    {
                        "source_slot_id": state["source_slot_id"],
                        "action_id": state["action_id"],
                        "action_phase": state["action_phase"],
                        "walk_phase_period_frames": state["walk_phase_period_frames"],
                        "translation_ue_cm": list(state["translation_ue_cm"]),
                        "yaw_ue_deg": state["yaw_ue_deg"],
                    }
                    for state in frame["actor_states"]
                ],
            }
            rgb_frames.append(image)
            frame_records.append(frame_record)
        root_readback_summary = summarize_root_readbacks(
            expected_frames=_expected_root_readback_frames(timeline),
            actor_readbacks=actor_readbacks,
            camera_readbacks=camera_readbacks,
        )
        animation_summary = _animation_readback_summary(animation_records_by_slot)
        arrays = output / "arrays"
        arrays.mkdir()
        np.save(arrays / "rgb.npy", np.ascontiguousarray(np.stack(rgb_frames)))
        _write_new_json(output / "frame_records.json", {"frames": frame_records})
        success_receipt = {
            "status": "research_only",
            "research_only": True,
            "episode_counted": False,
            "qualification_claim": False,
            "asset_authorization": asset_authorization,
            "claim_boundary": (
                "external SPEAR RGB research only; no dry audio, RLR, M6/M7 "
                "bundle, source qualification, Topdown, or dataset claim"
            ),
            "inputs": {
                "actor_selection": str(selection_file),
                "timeline": str(timeline_file),
                "closure_report": str(closure_file),
                "stage_root": str(stage),
                "spear_executable": str(executable),
            },
            "capture": {
                "frame_count": FRAME_COUNT,
                "completed_frame_count": len(frame_records),
                "frame_rate_hz": FRAME_RATE_HZ,
                "ticks_per_frame": TICKS_PER_FRAME,
                "modalities": ["rgb"],
                "audio_requested": False,
                "rlr_requested": False,
                "qualification_requested": False,
                "m6_m7_bundle_requested": False,
                "root_readback_summary": root_readback_summary,
                "animation_readback_summary": animation_summary,
                "capture_warmup": capture_warmup,
            },
            "artifacts": {
                "rgb": "arrays/rgb.npy",
                "frame_records": "frame_records.json",
            },
        }
    except BaseException as error:
        run_error = error
        run_traceback = error.__traceback__
    finally:
        if instance is not None:
            if run_error is not None:
                try:
                    instance.close(force=True)
                except BaseException as error:
                    cleanup_error = error
            else:
                def record_cleanup_error(error: BaseException) -> None:
                    nonlocal cleanup_error
                    if cleanup_error is None:
                        cleanup_error = error
                        return
                    try:
                        cleanup_error.add_note(
                            "cleanup also failed: %s: %s"
                            % (type(error).__name__, error)
                        )
                    except Exception:
                        pass

                try:
                    _destroy_runtime_actors(instance, runtimes)
                except BaseException as error:
                    record_cleanup_error(error)
                try:
                    close_scene_capture(
                        instance=instance,
                        game=game,
                        camera=camera,
                        capture=capture,
                    )
                except BaseException as error:
                    record_cleanup_error(error)
                try:
                    instance.close(force=cleanup_error is not None)
                except BaseException as error:
                    record_cleanup_error(error)

    if run_error is not None:
        if cleanup_error is not None:
            try:
                run_error.add_note(
                    "cleanup also failed: %s: %s"
                    % (type(cleanup_error).__name__, cleanup_error)
                )
            except Exception:
                pass
        try:
            _write_partial_frame_records(output, frame_records)
            _write_new_json(
                output / "research_receipt.json",
                _partial_capture_receipt(
                    output=output,
                    selection_file=selection_file,
                    timeline_file=timeline_file,
                    closure_file=closure_file,
                    stage=stage,
                    executable=executable,
                    asset_authorization=asset_authorization,
                    frame_records=frame_records,
                    camera_readbacks=camera_readbacks,
                    actor_readbacks=actor_readbacks,
                    animation_records_by_slot=animation_records_by_slot,
                    root_readback_summary=root_readback_summary,
                    error=run_error,
                ),
            )
        except BaseException as receipt_error:
            try:
                run_error.add_note(
                    "partial receipt write failed: %s: %s"
                    % (type(receipt_error).__name__, receipt_error)
                )
            except Exception:
                pass
        raise run_error.with_traceback(run_traceback)
    if cleanup_error is not None:
        try:
            _write_partial_frame_records(output, frame_records)
            _write_new_json(
                output / "research_receipt.json",
                _partial_capture_receipt(
                    output=output,
                    selection_file=selection_file,
                    timeline_file=timeline_file,
                    closure_file=closure_file,
                    stage=stage,
                    executable=executable,
                    asset_authorization=asset_authorization,
                    frame_records=frame_records,
                    camera_readbacks=camera_readbacks,
                    actor_readbacks=actor_readbacks,
                    animation_records_by_slot=animation_records_by_slot,
                    root_readback_summary=root_readback_summary,
                    error=cleanup_error,
                ),
            )
        except BaseException as receipt_error:
            try:
                cleanup_error.add_note(
                    "partial receipt write failed: %s: %s"
                    % (type(receipt_error).__name__, receipt_error)
                )
            except Exception:
                pass
        raise cleanup_error
    if success_receipt is None:
        raise CurrentApartmentVisualError(
            "capture completed without a research receipt"
        )
    _write_new_json(output / "research_receipt.json", success_receipt)
    return success_receipt


__all__ = [
    "CurrentApartmentVisualError",
    "author_current_apartment_visual_timeline",
    "capture_current_apartment_visual",
]

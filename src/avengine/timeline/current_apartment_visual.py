"""Standalone current Apartment RGB research author and capture path.

This is intentionally separate from the retained M6/M7 Apartment writers.
It owns only an external A2 actor selection, a configurable-frame-count
visual timeline (75 frames at 15 Hz by default), and native SPEAR RGB research
capture. It does not read a formal source bundle, qualification, dry audio,
RLR, or a historical reader.
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
    read_scene_component_pose,
    read_rgb_bgr,
    run_frame_transaction,
    spawn_attached_static_actor,
    spawn_attached_visual_actor,
    spawn_scene_capture,
    warm_scene_capture_until_stable,
)
from avengine.route_sampling import planar_cumulative, sample_polyline
from avengine.backends.spear_ue.launch import (
    validate_current_production_spear_executable,
)
from avengine.optional_backends.spear_apartment import (
    ANIMATION_TOLERANCE_SECONDS,
    POSITION_TOLERANCE_CM,
    ROTATION_TOLERANCE_DEGREES,
    animation_position_seconds,
    apply_ue_component_frame_delta,
)
from avengine.runtime_profiles import (
    build_asset_emitter_binding,
    load_source_asset_runtime_registry,
    resolve_source_asset_runtime_profile,
)


DEFAULT_FRAME_COUNT = 75
DEFAULT_FRAME_RATE_HZ = 15
DEFAULT_TICKS_PER_FRAME = 3_200
CLOCK_TICKS_PER_SECOND = 48_000

# Backward-compatible names retained for callers that use the original
# 75-frame, 15fps research timeline defaults.
FRAME_COUNT = DEFAULT_FRAME_COUNT
FRAME_RATE_HZ = DEFAULT_FRAME_RATE_HZ
TICKS_PER_FRAME = DEFAULT_TICKS_PER_FRAME
CAMERA_BLUEPRINT = "/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C"
CAPTURE_COMPONENT_NAME = "DefaultSceneRoot.final_tone_curve_hdr_"
NATIVE_APARTMENT_MAP = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
APARTMENT_ROOM_PROFILE_ID = "spear_apartment_0000"


def resolve_native_map(timeline, requested_map=None):
    """Which UE map does this capture launch, and does the timeline agree?

    The map used to be a module constant, which made the renderer room-specific
    and — worse — let a timeline authored for one room be captured in another
    without complaint. The timeline now carries its own ``room.map_path``; an
    explicit argument may override it only when the two agree. Same failure
    shape as the camera-yaw mismatch: a declared fact and an executed fact that
    silently diverge.
    """

    room = (timeline or {}).get("room") or {}
    declared = room.get("map_path")
    if "map_path" in room and not declared:
        raise CurrentApartmentVisualError(
            "the timeline declares an empty room.map_path; a declared-but-blank "
            "map must fail rather than fall back to a default room"
        )
    if requested_map and declared and requested_map != declared:
        raise CurrentApartmentVisualError(
            "the timeline was authored for map "
            f"{declared!r} but the capture was asked for {requested_map!r}"
        )
    resolved = requested_map or declared or NATIVE_APARTMENT_MAP
    if not str(resolved).startswith("/Game/"):
        raise CurrentApartmentVisualError(
            f"native map must be a /Game package path, got {resolved!r}"
        )
    return str(resolved)


class CurrentApartmentVisualError(RuntimeError):
    """A current Apartment visual-research request is incomplete or unsafe."""


def _resolve_render_clock(
    frame_count: object = FRAME_COUNT,
    frame_rate_hz: object = FRAME_RATE_HZ,
    ticks_per_frame: object | None = None,
    *,
    owner: str = "render clock",
) -> tuple[int, int | float, int]:
    """Validate and normalize one timeline clock."""
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 2
    ):
        raise CurrentApartmentVisualError(
            f"{owner} frame_count must be an integer >= 2"
        )
    if (
        isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, (int, float))
        or not math.isfinite(float(frame_rate_hz))
        or float(frame_rate_hz) <= 0.0
    ):
        raise CurrentApartmentVisualError(
            f"{owner} frame_rate_hz must be positive and finite"
        )
    rate = float(frame_rate_hz)
    if ticks_per_frame is None:
        implied = float(CLOCK_TICKS_PER_SECOND) / rate
        rounded = int(round(implied))
        if not math.isclose(implied, rounded, rel_tol=0.0, abs_tol=1.0e-9):
            raise CurrentApartmentVisualError(
                f"{owner} needs an explicit integer ticks_per_frame for "
                f"frame_rate_hz={rate:g}"
            )
        ticks = rounded
    elif (
        isinstance(ticks_per_frame, bool)
        or not isinstance(ticks_per_frame, (int, float))
        or not math.isfinite(float(ticks_per_frame))
        or float(ticks_per_frame) < 1.0
        or not float(ticks_per_frame).is_integer()
    ):
        raise CurrentApartmentVisualError(
            f"{owner} ticks_per_frame must be a positive integer"
        )
    else:
        ticks = int(ticks_per_frame)
    if not math.isclose(
        rate * float(ticks),
        float(CLOCK_TICKS_PER_SECOND),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise CurrentApartmentVisualError(
            f"{owner} frame_rate_hz and ticks_per_frame disagree"
        )
    normalized_rate: int | float = (
        int(rate) if rate.is_integer() else rate
    )
    return int(frame_count), normalized_rate, ticks


def _timeline_render_clock(
    timeline: Mapping[str, Any],
) -> tuple[int, int | float, int]:
    render = timeline.get("render")
    if not isinstance(render, Mapping):
        raise CurrentApartmentVisualError("timeline has no render clock")
    return _resolve_render_clock(
        render.get("frame_count", FRAME_COUNT),
        render.get("frame_rate_hz", FRAME_RATE_HZ),
        (
            render["ticks_per_frame"]
            if "ticks_per_frame" in render
            else None
        ),
        owner="timeline render",
    )


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
        valid_slot = (
            isinstance(slot, str)
            and slot.startswith("source")
            and slot.removeprefix("source").isdigit()
            and int(slot.removeprefix("source")) >= 1
        )
        if not valid_slot or slot in result:
            raise CurrentApartmentVisualError(
                "actor selection must contain unique canonical sourceN slots"
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

        if record.get("entity_class") == "rigid_object":
            if declared.get("static_mesh_binding") != raw_binding.get(
                "static_mesh_binding"
            ):
                raise CurrentApartmentVisualError(
                    f"actor selection {slot} static mesh binding differs from registry"
                )
            if declared.get("static_mesh_object_path") != raw_binding.get(
                "static_mesh_object_path"
            ):
                raise CurrentApartmentVisualError(
                    f"actor selection {slot} static mesh path differs from registry"
                )
            static_mesh_path = raw_binding.get("static_mesh_object_path")
            if (
                not isinstance(static_mesh_path, str)
                or not static_mesh_path.startswith("/Game/")
            ):
                raise CurrentApartmentVisualError(
                    f"actor selection {slot} static mesh path is invalid"
                )
            expected_static_mesh_package = _package_from_object_path(
                static_mesh_path,
                owner=f"actor selection {slot} static mesh",
            )
            if declared.get("static_mesh_package") != expected_static_mesh_package:
                raise CurrentApartmentVisualError(
                    f"actor selection {slot} static mesh package differs from path"
                )
            actor_scale = raw_binding.get("actor_scale")
            static_forward_yaw = raw_binding.get("ue_static_forward_yaw_deg")
            if (
                isinstance(actor_scale, bool)
                or not isinstance(actor_scale, (int, float))
                or not math.isfinite(float(actor_scale))
                or float(actor_scale) <= 0.0
                or isinstance(static_forward_yaw, bool)
                or not isinstance(static_forward_yaw, (int, float))
                or not math.isfinite(float(static_forward_yaw))
            ):
                raise CurrentApartmentVisualError(
                    f"actor selection {slot} static mesh transform is invalid"
                )
            anchor_id = record.get("default_emitter_anchor_id")
            anchors = [
                item
                for item in record.get("emitter_anchors", [])
                if isinstance(item, Mapping)
                and item.get("anchor_id") == anchor_id
            ]
            if len(anchors) != 1:
                raise CurrentApartmentVisualError(
                    f"actor selection {slot} static emitter anchor is not unique"
                )
            anchor = anchors[0]
            if (
                anchor.get("anchor_type") != "object_speaker"
                or anchor.get("offset_space") != "final_scaled_asset_root"
            ):
                raise CurrentApartmentVisualError(
                    f"actor selection {slot} static emitter anchor is invalid"
                )
            emitter_offset = _finite_triplet(
                anchor.get("offset_m"),
                owner=f"actor selection {slot} emitter offset",
            )
            result[slot] = {
                "source_slot_id": slot,
                "actor_id": f"{slot}_actor",
                "asset_id": asset_id,
                "revision": revision,
                "entity_class": "rigid_object",
                "motion_model": "rigid_static",
                "static_mesh_binding": "explicit_path",
                "static_mesh_object_path": static_mesh_path,
                "static_mesh_package": expected_static_mesh_package,
                "actor_scale": float(actor_scale),
                "ue_static_forward_yaw_deg": float(static_forward_yaw),
                "emitter_anchor_id": str(anchor_id),
                "emitter_offset_m": emitter_offset,
                "emitter_offset_space": anchor["offset_space"],
            }
            continue

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
        emitter_binding = build_asset_emitter_binding(
            source_registry, source_slot_id=slot, asset_id=asset_id, revision=revision,
        )
        if emitter_binding["offset_space"] != "final_scaled_asset_root":
            raise CurrentApartmentVisualError(f"{slot}: unsupported emitter offset space")
        result[slot] = {
            "source_slot_id": slot,
            "actor_id": f"{slot}_actor",
            "asset_id": asset_id,
            "revision": revision,
            "emitter_anchor_id": emitter_binding["semantic_anchor_id"],
            "emitter_offset_m": list(emitter_binding["emitter_offset_m"]),
            "emitter_offset_space": emitter_binding["offset_space"],
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
    expected_slots = {
        f"source{index}" for index in range(1, len(result) + 1)}
    if len(result) < 2 or set(result) != expected_slots:
        raise CurrentApartmentVisualError(
            "actor selection must contain contiguous source1..sourceN slots "
            "with at least two actors"
        )
    result = {slot: result[slot] for slot in sorted(
        result, key=lambda value: int(value.removeprefix("source")))}
    authorization = selection.get("asset_authorization")
    if authorization not in {"verified_internal", "unverified"}:
        authorization = "unverified"
    return selection_path, result, authorization


def _is_static_binding(binding: Mapping[str, Any]) -> bool:
    return (
        binding.get("motion_model") == "rigid_static"
        or binding.get("entity_class") == "rigid_object"
    )


def _timeline_actor_declaration(binding: Mapping[str, Any]) -> dict[str, Any]:
    if _is_static_binding(binding):
        return {
            key: binding[key]
            for key in (
                "source_slot_id",
                "actor_id",
                "asset_id",
                "revision",
                "entity_class",
                "motion_model",
                "static_mesh_binding",
                "static_mesh_object_path",
                "static_mesh_package",
                "actor_scale",
                "ue_static_forward_yaw_deg",
                "emitter_anchor_id",
                "emitter_offset_m",
                "emitter_offset_space",
            )
        }
    return {
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


def _spatial_cumulative(points: Sequence[Sequence[float]]) -> list[float]:
    """Cumulative three-dimensional distance for rigid-mesh routes."""
    cumulative = [0.0]
    for first, second in zip(points[:-1], points[1:]):
        cumulative.append(
            cumulative[-1]
            + math.sqrt(
                sum(
                    (float(second[axis]) - float(first[axis])) ** 2
                    for axis in range(3)
                )
            )
        )
    return cumulative


def _finite_waypoints(
    value: Any, *, owner: str
) -> list[list[float]] | None:
    """Validate an optional UE-cm waypoint polyline (at least two points)."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CurrentApartmentVisualError(f"{owner} must be a sequence of points")
    points = [
        _finite_triplet(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(points) < 2:
        raise CurrentApartmentVisualError(f"{owner} needs at least two waypoints")
    return points


def _timeline_state(
    *,
    binding: Mapping[str, Any],
    start: Sequence[float],
    end: Sequence[float],
    frame_index: int,
    frame_count: int = FRAME_COUNT,
    walk_start_frame: int = 0,
    waypoints: Sequence[Sequence[float]] | None = None,
    camera_position_ue_cm: Sequence[float] | None = None,
) -> dict[str, Any]:
    if _is_static_binding(binding):
        points = list(waypoints) if waypoints is not None else [list(start), list(end)]
        if len(points) < 2:
            raise CurrentApartmentVisualError(
                f"static actor {binding['source_slot_id']} needs at least two route points"
            )
        cumulative = _spatial_cumulative(points)
        total = cumulative[-1]
        if total > 1.0e-9:
            fraction = min(max(int(frame_index), 0), frame_count - 1) / float(
                frame_count - 1
            )
            translation, segment = sample_polyline(
                points, cumulative, total * fraction
            )
            segment_end = min(segment + 1, len(points) - 1)
            static_yaw = _actor_yaw(
                points[segment],
                points[segment_end],
                anatomical_forward_yaw_deg=float(
                    binding.get("ue_static_forward_yaw_deg", 0.0)
                ),
            )
        else:
            translation = [float(value) for value in points[0]]
            static_yaw = binding.get("static_yaw_ue_deg")
            if static_yaw is None and camera_position_ue_cm is not None:
                static_yaw = _actor_yaw(
                    points[0],
                    camera_position_ue_cm,
                    anatomical_forward_yaw_deg=float(
                        binding.get("ue_static_forward_yaw_deg", 0.0)
                    ),
                )
            if static_yaw is None:
                static_yaw = 0.0
        if (
            isinstance(static_yaw, bool)
            or not isinstance(static_yaw, (int, float))
            or not math.isfinite(float(static_yaw))
        ):
            raise CurrentApartmentVisualError(
                f"static actor {binding['source_slot_id']} yaw is invalid"
            )
        return {
            "actor_id": binding["actor_id"],
            "source_slot_id": binding["source_slot_id"],
            "asset_id": binding["asset_id"],
            "revision": binding["revision"],
            "entity_class": "rigid_object",
            "motion_model": "rigid_static",
            "translation_ue_cm": translation,
            "yaw_ue_deg": float(static_yaw),
            "action_id": None,
            "action_phase": 0.0,
        }
    route = list(waypoints) if waypoints is not None and len(waypoints) > 2 else None
    if route is not None:
        return _polyline_timeline_state(
            binding=binding,
            route=route,
            frame_index=frame_index,
            frame_count=frame_count,
            walk_start_frame=walk_start_frame,
        )
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
        walk_end = frame_count - 1
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


def _polyline_timeline_state(
    *,
    binding: Mapping[str, Any],
    route: Sequence[Sequence[float]],
    frame_index: int,
    frame_count: int,
    walk_start_frame: int,
) -> dict[str, Any]:
    """Per-frame state for a multi-waypoint route.

    The route is resampled by planar arc length across the moving frames, so
    the actor keeps a constant speed regardless of how the polyline splits
    into segments, and yaw follows the tangent of the segment it is on.
    """
    cumulative = planar_cumulative(route)
    total = cumulative[-1]
    moving = total > 1.0e-9
    walk_start = walk_start_frame
    walk_end = frame_count - 1
    action_id = "walk" if moving and frame_index >= walk_start else "idle"
    period = int(binding["walk_phase_period_frames"])
    phase = (
        float((frame_index - walk_start) % period) / float(period)
        if action_id == "walk"
        else 0.0
    )
    if moving and frame_index >= walk_start:
        travelled = total * float(frame_index - walk_start) / float(
            walk_end - walk_start
        )
        translation, segment = sample_polyline(route, cumulative, travelled)
    else:
        translation, segment = [float(value) for value in route[0]], 0
    return {
        "actor_id": binding["actor_id"],
        "source_slot_id": binding["source_slot_id"],
        "asset_id": binding["asset_id"],
        "revision": binding["revision"],
        "walk_phase_period_frames": binding["walk_phase_period_frames"],
        "translation_ue_cm": translation,
        "yaw_ue_deg": _actor_yaw(
            route[segment],
            route[segment + 1],
            anatomical_forward_yaw_deg=float(binding["ue_anatomical_forward_yaw_deg"]),
        ),
        "action_id": action_id,
        "action_phase": phase,
        "route_geometry": "polyline",
        "route_waypoint_count": len(route),
        "route_arc_length_ue_cm": total,
        "route_segment_index": segment,
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
    human_waypoints_ue_cm: Sequence[Sequence[float]] | None = None,
    beagle_waypoints_ue_cm: Sequence[Sequence[float]] | None = None,
    native_map: str = NATIVE_APARTMENT_MAP,
    room_profile_id: str = APARTMENT_ROOM_PROFILE_ID,
    width: int = 1280,
    height: int = 720,
    hfov_degrees: float = 105.0,
    walk_start_frame: int = 0,
    frame_count: int = FRAME_COUNT,
    frame_rate_hz: float = FRAME_RATE_HZ,
    ticks_per_frame: int | None = None,
) -> dict[str, Any]:
    """Write one freely designed current Apartment visual-only timeline."""

    selection_file, bindings, asset_authorization = _selection_bindings(
        actor_selection_path=actor_selection_path,
        source_asset_registry_path=source_asset_registry_path,
    )
    (
        clock_frame_count,
        clock_frame_rate_hz,
        clock_ticks_per_frame,
    ) = _resolve_render_clock(
        frame_count,
        frame_rate_hz,
        ticks_per_frame,
        owner="timeline render",
    )
    if set(bindings) != {"source1", "source2"}:
        raise CurrentApartmentVisualError(
            "two-source author requires exactly source1 and source2; "
            "use author_current_n_actor_visual_timeline for N actors")
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
        or not 0 <= walk_start_frame < clock_frame_count - 1
    ):
        raise CurrentApartmentVisualError(
            f"walk_start_frame must be an integer in [0, {clock_frame_count - 2}]"
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
    routes = {
        "source1": _finite_waypoints(
            human_waypoints_ue_cm, owner="human_waypoints_ue_cm"
        ),
        "source2": _finite_waypoints(
            beagle_waypoints_ue_cm, owner="beagle_waypoints_ue_cm"
        ),
    }
    for slot, route in routes.items():
        if route is None:
            continue
        if route[0] != starts[slot] or route[-1] != ends[slot]:
            raise CurrentApartmentVisualError(
                f"{slot} waypoints must start at its start point and end at its end point"
            )
    frames = []
    for frame_index in range(clock_frame_count):
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * clock_ticks_per_frame,
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
                        frame_count=clock_frame_count,
                        walk_start_frame=walk_start_frame,
                        waypoints=routes[slot],
                        camera_position_ue_cm=camera_position,
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
            "map_path": native_map,
            "room_profile_id": room_profile_id,
        },
        "render": {
            "frame_count": clock_frame_count,
            "frame_rate_hz": clock_frame_rate_hz,
            "ticks_per_frame": clock_ticks_per_frame,
            "resolution_hw": [height, width],
            "hfov_degrees": float(hfov_degrees),
            "walk_start_frame": walk_start_frame,
        },
        "actors": [
            _timeline_actor_declaration(binding)
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




def build_current_n_actor_visual_timeline(
    *,
    actor_selection_path: str | Path,
    source_asset_registry_path: str | Path,
    camera_position_ue_cm: Sequence[float],
    camera_yaw_deg: float,
    routes_by_slot_ue_cm: Mapping[str, Sequence[Sequence[float]]],
    native_map: str,
    room_profile_id: str,
    width: int = 1280,
    height: int = 720,
    hfov_degrees: float = 105.0,
    walk_start_frames: Mapping[str, int] | None = None,
    frame_count: int = FRAME_COUNT,
    frame_rate_hz: float = FRAME_RATE_HZ,
    ticks_per_frame: int | None = None,
) -> dict[str, Any]:
    """Build a research timeline without writing an output artifact."""
    selection_file, bindings, asset_authorization = _selection_bindings(
        actor_selection_path=actor_selection_path,
        source_asset_registry_path=source_asset_registry_path,
    )
    (
        clock_frame_count,
        clock_frame_rate_hz,
        clock_ticks_per_frame,
    ) = _resolve_render_clock(
        frame_count,
        frame_rate_hz,
        ticks_per_frame,
        owner="timeline render",
    )
    slots = tuple(bindings)
    if set(routes_by_slot_ue_cm) != set(slots):
        raise CurrentApartmentVisualError(
            "N-actor routes must match actor-selection source slots")
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
            "timeline render dimensions or HFOV are invalid")
    camera_position = _finite_triplet(
        camera_position_ue_cm, owner="camera_position_ue_cm")
    camera_yaw = _finite_number(camera_yaw_deg, owner="camera_yaw_deg")
    starts_at = dict(walk_start_frames or {})
    routes = {}
    for slot in slots:
        route = _finite_waypoints(
            routes_by_slot_ue_cm[slot], owner=f"{slot}_waypoints_ue_cm")
        if route is None or len(route) != clock_frame_count:
            raise CurrentApartmentVisualError(
                f"{slot} must declare exactly {clock_frame_count} route samples")
        walk_start = int(starts_at.get(slot, 0))
        if not 0 <= walk_start < clock_frame_count - 1:
            raise CurrentApartmentVisualError(
                f"{slot} walk_start_frame must be in [0, {clock_frame_count - 2}]")
        routes[slot] = (route, walk_start)

    frames = []
    for frame_index in range(clock_frame_count):
        frames.append({
            "frame_index": frame_index,
            "pts_ticks": frame_index * clock_ticks_per_frame,
            "camera": {
                "translation_ue_cm": list(camera_position),
                "yaw_ue_deg": camera_yaw,
            },
            "actor_states": [
                _timeline_state(
                    binding=bindings[slot],
                    start=routes[slot][0][0],
                    end=routes[slot][0][-1],
                    frame_index=frame_index,
                    frame_count=clock_frame_count,
                    walk_start_frame=routes[slot][1],
                    waypoints=routes[slot][0],
                    camera_position_ue_cm=camera_position,
                )
                for slot in slots
            ],
        })
    timeline = {
        "kind": "current_n_actor_visual_research_timeline",
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "claim_boundary": (
            "N-actor external UE RGB research timeline only; no dataset, "
            "room, source, audio or modality-certification claim"),
        "actor_selection": str(selection_file),
        "room": {
            "map_path": native_map,
            "room_profile_id": room_profile_id,
        },
        "render": {
            "frame_count": clock_frame_count,
            "frame_rate_hz": clock_frame_rate_hz,
            "ticks_per_frame": clock_ticks_per_frame,
            "resolution_hw": [height, width],
            "hfov_degrees": float(hfov_degrees),
            "walk_start_frames": {
                slot: routes[slot][1] for slot in slots},
        },
        "actors": [
            _timeline_actor_declaration(bindings[slot])
            for slot in slots
        ],
        "asset_authorization": asset_authorization,
        "spatial_validation": "not_run",
        "frames": frames,
    }
    return timeline


def author_current_n_actor_visual_timeline(
    *,
    actor_selection_path: str | Path,
    source_asset_registry_path: str | Path,
    output_path: str | Path,
    camera_position_ue_cm: Sequence[float],
    camera_yaw_deg: float,
    routes_by_slot_ue_cm: Mapping[str, Sequence[Sequence[float]]],
    native_map: str,
    room_profile_id: str,
    width: int = 1280,
    height: int = 720,
    hfov_degrees: float = 105.0,
    walk_start_frames: Mapping[str, int] | None = None,
    frame_count: int = FRAME_COUNT,
    frame_rate_hz: float = FRAME_RATE_HZ,
    ticks_per_frame: int | None = None,
) -> dict[str, Any]:
    """Build and write a fresh N-actor timeline using the shared geometry path."""
    timeline = build_current_n_actor_visual_timeline(
        actor_selection_path=actor_selection_path,
        source_asset_registry_path=source_asset_registry_path,
        camera_position_ue_cm=camera_position_ue_cm,
        camera_yaw_deg=camera_yaw_deg,
        routes_by_slot_ue_cm=routes_by_slot_ue_cm,
        native_map=native_map,
        room_profile_id=room_profile_id,
        width=width,
        height=height,
        hfov_degrees=hfov_degrees,
        walk_start_frames=walk_start_frames,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        ticks_per_frame=ticks_per_frame,
    )
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
    ):
        raise CurrentApartmentVisualError(
            "timeline must be a non-counted research record with a render clock"
        )
    (
        clock_frame_count,
        clock_frame_rate_hz,
        clock_ticks_per_frame,
    ) = _timeline_render_clock(timeline)
    actors = timeline.get("actors")
    if not isinstance(actors, list) or len(actors) != len(bindings):
        raise CurrentApartmentVisualError("timeline actor count differs from selection")
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
        fields = (
            (
                "actor_id",
                "asset_id",
                "revision",
                "entity_class",
                "motion_model",
                "static_mesh_binding",
                "static_mesh_object_path",
                "static_mesh_package",
                "actor_scale",
                "ue_static_forward_yaw_deg",
                "emitter_anchor_id",
                "emitter_offset_m",
                "emitter_offset_space",
            )
            if _is_static_binding(binding)
            else (
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
        )
        for field in fields:
            if actor.get(field) != binding[field]:
                raise CurrentApartmentVisualError(
                    f"timeline {slot} {field} differs from actor selection"
                )
    frames = timeline.get("frames")
    if not isinstance(frames, list) or len(frames) != clock_frame_count:
        raise CurrentApartmentVisualError(
            f"timeline must contain exactly {clock_frame_count} frames"
        )
    for frame_index, frame in enumerate(frames):
        if (
            not isinstance(frame, Mapping)
            or frame.get("frame_index") != frame_index
            or frame.get("pts_ticks") != frame_index * clock_ticks_per_frame
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
        if not isinstance(states, list) or len(states) != len(bindings):
            raise CurrentApartmentVisualError(
                f"timeline frame {frame_index} actor count differs from selection"
            )
        state_slots = [
            state.get("source_slot_id")
            for state in states
            if isinstance(state, Mapping)
        ]
        if state_slots != list(bindings):
            raise CurrentApartmentVisualError(
                f"timeline frame {frame_index} actor order differs from selection"
            )
        for state in states:
            assert isinstance(state, Mapping)
            slot = state["source_slot_id"]
            binding = bindings[slot]
            if _is_static_binding(binding):
                for field in ("actor_id", "asset_id", "revision", "entity_class", "motion_model"):
                    if state.get(field) != (
                        binding[field]
                        if field in binding
                        else "rigid_object"
                        if field == "entity_class"
                        else "rigid_static"
                    ):
                        raise CurrentApartmentVisualError(
                            f"timeline frame {frame_index} {slot} {field} differs"
                        )
                if state.get("action_id") is not None:
                    raise CurrentApartmentVisualError(
                        f"timeline frame {frame_index} {slot} static actor has an action"
                    )
                phase = _finite_number(
                    state.get("action_phase"),
                    owner=f"timeline frame {frame_index} {slot} static action phase",
                )
                if phase != 0.0:
                    raise CurrentApartmentVisualError(
                        f"timeline frame {frame_index} {slot} static action phase is not zero"
                    )
            else:
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
            position = _finite_triplet(
                state.get("translation_ue_cm"),
                owner=f"timeline frame {frame_index} {slot} position",
            )
            yaw = _finite_number(
                state.get("yaw_ue_deg"),
                owner=f"timeline frame {frame_index} {slot} yaw",
            )
    return path, timeline


def _closure_mappings(
    *,
    closure_report_path: str | Path,
    bindings: Mapping[str, Mapping[str, Any]],
    native_map: str,
) -> tuple[Path, list[tuple[str, str]]]:
    path = _external_file(closure_report_path, owner="--closure-report")
    report = _read_mapping(path, owner="closure report")
    variants = report.get("variants")
    if not isinstance(variants, Mapping):
        raise CurrentApartmentVisualError("closure report has no variants")
    required = {
        native_map,
        "/SpContent/Blueprints/BP_CameraSensor",
    }
    for binding in bindings.values():
        if _is_static_binding(binding):
            required.add(
                _package_from_object_path(
                    binding["static_mesh_object_path"],
                    owner="selected static mesh",
                )
            )
        else:
            required.update(
                {
                    _package_from_object_path(
                        binding["blueprint_class_path"],
                        owner="selected blueprint",
                    ),
                    binding["graph_mesh_package"],
                    _package_from_object_path(
                        binding["idle_animation"],
                        owner="selected idle animation",
                    ),
                    _package_from_object_path(
                        binding["walking_animation"],
                        owner="selected walking animation",
                    ),
                }
            )
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


def _emitter_offset_ue_cm(binding: Mapping[str, Any]) -> list[float]:
    offset_m = _finite_triplet(
        binding.get("emitter_offset_m"),
        owner=f"{binding['source_slot_id']} emitter offset",
    )
    # The AVEngine asset basis is X-forward/Y-up/Z-right in metres. The
    # Apartment UE import basis is X-forward/Y-right/Z-up in centimetres.
    return [100.0 * offset_m[0], 100.0 * offset_m[2], 100.0 * offset_m[1]]


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
    for slot in bindings:
        binding = bindings[slot]
        state = states[slot]
        if _is_static_binding(binding):
            runtime = spawn_attached_static_actor(
                game,
                actor_id=str(binding["actor_id"]),
                static_mesh_object_path=str(binding["static_mesh_object_path"]),
                position_ue_cm=list(state["translation_ue_cm"]),
                yaw_ue_degrees=float(state["yaw_ue_deg"]),
                actor_scale=float(binding["actor_scale"]),
                emitter_local_ue_cm=_emitter_offset_ue_cm(binding),
            )
            runtime.update(
                {
                    "binding": binding,
                    "motion_model": "rigid_static",
                    "animations": {},
                    "lengths": {},
                    "current_action": None,
                }
            )
        else:
            runtime = spawn_attached_visual_actor(
                game,
                actor_id=str(binding["actor_id"]),
                blueprint_class_path=str(binding["blueprint_class_path"]),
                position_ue_cm=list(state["translation_ue_cm"]),
                yaw_ue_degrees=float(state["yaw_ue_deg"]),
                emitter_local_ue_cm=(
                    [100.0 * binding["emitter_offset_m"][axis] for axis in (0, 2, 1)]
                    if "emitter_offset_m" in binding else None
                ),
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
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in lengths.values()
            ):
                raise CurrentApartmentVisualError(f"{slot} animations are invalid")
            runtime.update(
                {
                    "binding": binding,
                    "motion_model": "articulated",
                    "animations": animations,
                    "lengths": lengths,
                    "current_action": None,
                }
            )
        _apply_runtime_state(runtime, state=state, frame_index=0)
        runtimes[slot] = runtime
    return runtimes



def _apply_runtime_state(
    runtime: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    frame_index: int,
) -> dict[str, Any] | None:
    binding = runtime.get("binding")
    if not isinstance(binding, Mapping):
        binding = runtime
    if _is_static_binding(binding):
        if state.get("action_id") is not None:
            raise CurrentApartmentVisualError(
                f"frame {frame_index} static actor has an animation action"
            )
        phase = _finite_number(
            state.get("action_phase"),
            owner=f"frame {frame_index} static action phase",
        )
        if phase != 0.0:
            raise CurrentApartmentVisualError(
                f"frame {frame_index} static action phase is not zero"
            )
        position = state["translation_ue_cm"]
        runtime["anchor"].K2_SetActorLocationAndRotation(
            NewLocation={"X": position[0], "Y": position[1], "Z": position[2]},
            NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": state["yaw_ue_deg"]},
            bSweep=False,
            bTeleport=True,
        )
        return None

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


def _frame_actor_state_record(
    state: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "source_slot_id": state["source_slot_id"],
        "actor_id": state["actor_id"],
        "entity_class": binding.get("entity_class"),
        "motion_model": (
            "rigid_static" if _is_static_binding(binding) else "articulated"
        ),
        "action_id": state.get("action_id"),
        "action_phase": state.get("action_phase"),
        "translation_ue_cm": list(state["translation_ue_cm"]),
        "yaw_ue_deg": state["yaw_ue_deg"],
    }
    if not _is_static_binding(binding):
        result["walk_phase_period_frames"] = state["walk_phase_period_frames"]
    return result


def _wrap_angle_difference_degrees(observed: float, expected: float) -> float:
    return (float(observed) - float(expected) + 180.0) % 360.0 - 180.0


def _summarize_root_readbacks_for_clock(
    *,
    expected_frames: Sequence[Mapping[str, Any]],
    actor_readbacks: Mapping[str, Sequence[Mapping[str, Any]]],
    camera_readbacks: Sequence[Mapping[str, Any]],
    frame_count: int,
) -> dict[str, Any]:
    """Validate root readback counts against the timeline's render clock."""
    if len(expected_frames) != frame_count or len(camera_readbacks) != frame_count:
        raise CurrentApartmentVisualError(
            "root readback requires the timeline frame_count for every frame"
        )
    expected_actor_ids = [
        state["actor_id"] for state in expected_frames[0]["actor_states"]
    ]
    if set(actor_readbacks) != set(expected_actor_ids):
        raise CurrentApartmentVisualError(
            "actor readback closure differs from Timeline"
        )
    summaries: dict[str, Any] = {}
    for actor_id in expected_actor_ids:
        records = actor_readbacks[actor_id]
        if len(records) != frame_count:
            raise CurrentApartmentVisualError(
                f"{actor_id} root readback lacks {frame_count} frames"
            )
        position_errors: list[float] = []
        yaw_errors: list[float] = []
        for frame_index, (frame, record) in enumerate(
            zip(expected_frames, records, strict=True)
        ):
            expected = next(
                item
                for item in frame["actor_states"]
                if item["actor_id"] == actor_id
            )
            expected_position = _finite_triplet(
                expected.get("translation_ue_cm"),
                owner=f"expected actor {actor_id} frame {frame_index} position",
            )
            expected_yaw = _finite_number(
                expected.get("actor_yaw_ue_deg"),
                owner=f"expected actor {actor_id} frame {frame_index} yaw",
            )
            location = _finite_triplet(
                record.get("location_cm"),
                owner=f"observed actor {actor_id} frame {frame_index} position",
            )
            rotation = _finite_triplet(
                record.get("rotation_deg"),
                owner=f"observed actor {actor_id} frame {frame_index} rotation",
            )
            if record.get("frame_index") != frame_index:
                raise CurrentApartmentVisualError(
                    f"{actor_id} readback frame order changed"
                )
            position_errors.append(
                max(
                    abs(location[axis] - expected_position[axis])
                    for axis in range(3)
                )
            )
            yaw_errors.append(
                abs(
                    _wrap_angle_difference_degrees(
                        rotation[2], expected_yaw
                    )
                )
            )
        maximum_position = max(position_errors)
        maximum_yaw = max(yaw_errors)
        if (
            maximum_position > POSITION_TOLERANCE_CM
            or maximum_yaw > ROTATION_TOLERANCE_DEGREES
        ):
            raise CurrentApartmentVisualError(
                f"{actor_id} UE root readback drifted"
            )
        summaries[actor_id] = {
            "status": "pass",
            "maximum_position_error_cm": maximum_position,
            "maximum_yaw_error_deg": maximum_yaw,
        }

    camera_position_errors: list[float] = []
    camera_yaw_errors: list[float] = []
    for frame_index, (expected_frame, record) in enumerate(
        zip(expected_frames, camera_readbacks, strict=True)
    ):
        expected_camera = expected_frame.get("camera_state")
        if not isinstance(expected_camera, Mapping):
            raise CurrentApartmentVisualError(
                "timeline root readback lacks a per-frame camera state"
            )
        if record.get("frame_index") != frame_index:
            raise CurrentApartmentVisualError(
                "camera readback frame order changed"
            )
        expected_position = _finite_triplet(
            expected_camera.get("ue_position_cm"),
            owner=f"expected camera state {frame_index} position",
        )
        expected_yaw = _finite_number(
            expected_camera.get("ue_yaw_deg"),
            owner=f"expected camera state {frame_index} yaw",
        )
        location = _finite_triplet(
            record.get("location_cm"),
            owner=f"observed camera state {frame_index} position",
        )
        rotation = _finite_triplet(
            record.get("rotation_deg"),
            owner=f"observed camera state {frame_index} rotation",
        )
        camera_position_errors.append(
            max(
                abs(location[axis] - expected_position[axis])
                for axis in range(3)
            )
        )
        camera_yaw_errors.append(
            abs(_wrap_angle_difference_degrees(rotation[2], expected_yaw))
        )
    maximum_camera_position = max(camera_position_errors)
    maximum_camera_yaw = max(camera_yaw_errors)
    if (
        maximum_camera_position > POSITION_TOLERANCE_CM
        or maximum_camera_yaw > ROTATION_TOLERANCE_DEGREES
    ):
        raise CurrentApartmentVisualError("UE camera root readback drifted")
    summaries["camera"] = {
        "status": "pass",
        "maximum_position_error_cm": maximum_camera_position,
        "maximum_yaw_error_deg": maximum_camera_yaw,
        "per_frame_camera_state": True,
        "checked_pose_hash_count": 0,
        "unique_expected_pose_hash_count": 0,
    }
    return summaries


def _animation_readback_summary(
    records_by_slot: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize observed animation positions without adding a new contract."""

    actors: dict[str, dict[str, Any]] = {}
    for slot in records_by_slot:
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
    frame_count: int = FRAME_COUNT,
    frame_rate_hz: float = FRAME_RATE_HZ,
    ticks_per_frame: int = TICKS_PER_FRAME,
) -> dict[str, Any]:
    capture: dict[str, Any] = {
        "frame_count": frame_count,
        "completed_frame_count": len(frame_records),
        "frame_rate_hz": frame_rate_hz,
        "ticks_per_frame": ticks_per_frame,
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
    native_map: str | None = None,
    capture_warmup_config_path: str | Path | None = None,
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
    (
        clock_frame_count,
        clock_frame_rate_hz,
        clock_ticks_per_frame,
    ) = _timeline_render_clock(timeline)
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
    resolved_map = resolve_native_map(timeline, native_map)
    closure_file, mappings = _closure_mappings(
        closure_report_path=closure_report_path, bindings=bindings,
        native_map=resolved_map,
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
        slot: [] for slot in bindings}
    root_readback_summary: dict[str, Any] | None = None
    success_receipt: dict[str, Any] | None = None
    run_error: BaseException | None = None
    run_traceback = None
    cleanup_error: BaseException | None = None
    try:
        instance = launch_external_game_instance(
            spear_executable=executable,
            native_map=resolved_map,
            frame_rate_hz=clock_frame_rate_hz,
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
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass
        capture_warmup = warm_scene_capture_until_stable(
            instance, capture,
            **({"config_path": capture_warmup_config_path}
               if capture_warmup_config_path is not None else {}),
        )
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
                    result = _apply_runtime_state(
                        runtimes[slot],
                        state=state,
                        frame_index=frame_index,
                    )
                    if result is not None:
                        animation_readbacks[slot] = result

            def readback() -> dict[str, Any]:
                return {
                    "rgb": read_rgb_bgr(capture),
                    "camera_pose": read_actor_pose(camera),
                    "actor_anchor_poses": {
                        slot: read_actor_pose(runtimes[slot]["anchor"])
                        for slot in bindings
                    },
                    "source_emitter_poses": {
                        slot: read_scene_component_pose(
                            runtimes[slot]["emitter_component"]
                        )
                        for slot in bindings
                        if "emitter_component" in runtimes[slot]
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
            source_emitter_poses = transaction["source_emitter_poses"]
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
                animation_record = observed_animations.get(slot)
                if animation_record is not None:
                    animation_records_by_slot[slot].append(animation_record)
                observed_actor_poses[slot] = {
                    "actor_id": actor_id,
                    **pose,
                }
            observed_animation_records = [
                observed_animations[slot]
                for slot in bindings
                if slot in observed_animations
            ]
            frame_record = {
                "frame_index": frame_index,
                "pts_ticks": frame["pts_ticks"],
                "observation_calls": 1,
                "camera_pose": camera_pose,
                "actor_anchor_poses": observed_actor_poses,
                "source_emitter_poses": source_emitter_poses,
                "animation_readbacks": observed_animation_records,
                "observed": {
                    "camera_pose": camera_pose,
                    "actor_anchor_poses": observed_actor_poses,
                    "source_emitter_poses": source_emitter_poses,
                    "animation_readbacks": observed_animation_records,
                },
                "camera": dict(frame["camera"]),
                "actor_states": [
                    _frame_actor_state_record(
                        state,
                        bindings[state["source_slot_id"]],
                    )
                    for state in frame["actor_states"]
                ],
            }
            rgb_frames.append(image)
            frame_records.append(frame_record)
        root_readback_summary = _summarize_root_readbacks_for_clock(
            expected_frames=_expected_root_readback_frames(timeline),
            actor_readbacks=actor_readbacks,
            camera_readbacks=camera_readbacks,
            frame_count=clock_frame_count,
        )
        animated_records = {
            slot: records
            for slot, records in animation_records_by_slot.items()
            if records
        }
        animation_summary = (
            _animation_readback_summary(animated_records)
            if animated_records
            else {
                "status": "not_applicable",
                "reason": "all selected actors are rigid_static",
                "actors": {},
            }
        )
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
                "frame_count": clock_frame_count,
                "completed_frame_count": len(frame_records),
                "frame_rate_hz": clock_frame_rate_hz,
                "ticks_per_frame": clock_ticks_per_frame,
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
                    frame_count=clock_frame_count,
                    frame_rate_hz=clock_frame_rate_hz,
                    ticks_per_frame=clock_ticks_per_frame,
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
                    frame_count=clock_frame_count,
                    frame_rate_hz=clock_frame_rate_hz,
                    ticks_per_frame=clock_ticks_per_frame,
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

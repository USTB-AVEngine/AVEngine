#!/usr/bin/env python3
"""Build Apartment viewpoint prefilter artifacts without using Studio.

The tool loads the declared Habitat scene and navmesh once, snapshots the live
rigid-object collision map, derives semantic targets from the retained M3
surface objects, and authors six independent formal ``view0`` M1 requests.
Its PNG and JSON outputs are explicitly 2-D prefilter evidence.  Native
first-person pixel review remains pending and is intentionally not claimed.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from avengine.camera_pose import apply_camera_listener_pose
from avengine.contracts.json_io import load_json, write_json
from avengine.m1.contracts import ValidatedM1Inputs, load_and_validate_inputs
from avengine.m1.habitat_capture import (
    _make_configuration,
    prepare_installed_habitat_runtime,
)
from avengine.m3.runtime import load_compiled_acoustic_scene
from avengine.m5_1.orientation import habitat_basis_from_yaw_degrees
from avengine.m6x.apartment_viewpoints import (
    ApartmentRasterPrefilterResult,
    prefilter_apartment_viewpoints,
)
from avengine.m6x.geometry import RuntimeObstacleMap, build_runtime_obstacle_map
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)


PREFILTER_LABEL = "2D PREFILTER ONLY - NATIVE PIXEL VALIDATION PENDING"
PILOT_MINIMUM_RASTER_SAMPLE_FRACTION = 0.10
PILOT_MINIMUM_PASSED_CORRIDOR_COUNT = 4
PILOT_MINIMUM_FEASIBLE_MOTION_MODE_COUNT = 0
PILOT_MINIMUM_CAMERA_EYE_HEIGHT_M = 0.8
PILOT_MAXIMUM_FLOOR_HORIZONTAL_SNAP_M = 0.15
PILOT_MAXIMUM_FLOOR_Y_DELTA_M = 0.25
PILOT_GENERATED_CORRIDOR_ENDPOINT_LIMIT = 192
PILOT_GENERATED_CORRIDOR_LIMIT = 256
PILOT_MAXIMUM_RANKED_CAMERA_CANDIDATES = 384


class ApartmentViewpointBuildError(ValueError):
    """The explicit inputs cannot produce the requested draft artifacts."""


@dataclass(frozen=True)
class _NativeCameraFloorAuthority:
    pathfinder: Any
    native_floor_height_m: float
    legacy_island_id: int
    maximum_horizontal_snap_m: float
    maximum_floor_y_delta_m: float

    def readback(self, position_m: Sequence[float]) -> dict[str, Any]:
        position = np.asarray(position_m, dtype=np.float64)
        requested = np.asarray(
            [position[0], self.native_floor_height_m, position[2]],
            dtype=np.float64,
        )
        navigable = bool(
            self.pathfinder.is_navigable(requested, self.maximum_floor_y_delta_m)
        )
        floor_point = np.asarray(
            self.pathfinder.snap_point(requested), dtype=np.float64
        )
        finite = floor_point.shape == (3,) and bool(np.all(np.isfinite(floor_point)))
        horizontal = (
            float(np.linalg.norm(floor_point[[0, 2]] - requested[[0, 2]]))
            if finite
            else math.inf
        )
        floor_y_delta = (
            abs(float(floor_point[1] - self.native_floor_height_m))
            if finite
            else math.inf
        )
        island_id = int(self.pathfinder.get_island(floor_point)) if finite else None
        passed = bool(
            navigable
            and finite
            and horizontal <= self.maximum_horizontal_snap_m
            and floor_y_delta <= self.maximum_floor_y_delta_m
            and island_id == self.legacy_island_id
        )
        return {
            "authority": "live_declared_navmesh_intended_floor_probe",
            "requested_floor_probe_m": [float(item) for item in requested],
            "floor_point_m": (
                [float(item) for item in floor_point] if finite else None
            ),
            "navigable": navigable,
            "horizontal_snap_distance_m": horizontal,
            "floor_y_delta_m": floor_y_delta,
            "island_id": island_id,
            "legacy_island_id": self.legacy_island_id,
            "thresholds": {
                "maximum_horizontal_snap_m": self.maximum_horizontal_snap_m,
                "maximum_floor_y_delta_m": self.maximum_floor_y_delta_m,
            },
            "passed": passed,
        }

    def feasible_mask(self, obstacle_map: RuntimeObstacleMap) -> np.ndarray:
        navmesh = np.asarray(obstacle_map.binary_navmesh, dtype=np.bool_)
        bounds = np.asarray(obstacle_map.bounds_m, dtype=np.float64)
        pixel_x = (bounds[1, 0] - bounds[0, 0]) / navmesh.shape[1]
        pixel_z = (bounds[1, 2] - bounds[0, 2]) / navmesh.shape[0]
        result = np.zeros_like(navmesh, dtype=np.bool_)
        for row, col in np.argwhere(navmesh):
            point = (
                float(bounds[0, 0] + (int(col) + 0.5) * pixel_x),
                self.native_floor_height_m,
                float(bounds[0, 2] + (int(row) + 0.5) * pixel_z),
            )
            result[int(row), int(col)] = self.readback(point)["passed"]
        return np.ascontiguousarray(result)


@dataclass(frozen=True)
class _SurfaceObject:
    object_id: str
    surface_id: str
    source_material_name: str
    category: str
    vertices: np.ndarray
    low_xyz: tuple[float, float, float]
    high_xyz: tuple[float, float, float]

    @property
    def low_xz(self) -> tuple[float, float]:
        return (self.low_xyz[0], self.low_xyz[2])

    @property
    def high_xz(self) -> tuple[float, float]:
        return (self.high_xyz[0], self.high_xyz[2])

    @property
    def stable_key(self) -> tuple[bytes, bytes, bytes]:
        return (
            self.source_material_name.encode("utf-8"),
            self.object_id.encode("utf-8"),
            self.surface_id.encode("utf-8"),
        )


_CATEGORY_ROLE = {
    "living": "living_reverse",
    "kitchen": "kitchen_reverse",
    "window": "window_facing",
}
_CATEGORY_CLUSTER_GAP_M = {
    "living": 1.0,
    "kitchen": 0.75,
    # A casement assembly has disconnected rails/panes roughly 1.5 m apart.
    # The independent 2.5 m span cap below prevents this gap from chaining
    # separate windows across the room.
    "window": 1.80,
}
_CATEGORY_LOCAL_GRID_M = {
    "living": 0.40,
    "kitchen": 0.35,
    "window": 0.30,
}
_CATEGORY_LOCAL_GAP_M = {
    "living": 0.55,
    "kitchen": 0.50,
    "window": 0.40,
}
_CATEGORY_MAX_SPAN_M = {
    "living": 3.0,
    "kitchen": 2.5,
    "window": 2.5,
}


def _material_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _material_category(source_material_name: str) -> str | None:
    token = _material_token(source_material_name)
    if "ceiling" in token or "lamp" in token:
        return None
    # Window assemblies are intentionally limited to explicit frame/casement
    # identities.  Generic ``glass`` can be a fireplace, oven, or lamp and
    # must not bridge spatially separate windows into one room-wide target.
    if any(part in token for part in ("window", "casement", "vinyl_frame")):
        return "window"
    if "kitchen" in token or "oven" in token:
        return "kitchen"
    if any(
        part in token
        for part in ("livingroom", "living_room", "sofa", "fireplace", "carpet")
    ):
        return "living"
    return None


def _make_surface_object(
    *,
    object_id: str,
    surface_id: str,
    source_material_name: str,
    category: str,
    vertices: np.ndarray,
) -> _SurfaceObject:
    low = np.min(vertices, axis=0)
    high = np.max(vertices, axis=0)
    return _SurfaceObject(
        object_id=object_id,
        surface_id=surface_id,
        source_material_name=source_material_name,
        category=category,
        vertices=np.ascontiguousarray(vertices),
        low_xyz=tuple(float(item) for item in low),
        high_xyz=tuple(float(item) for item in high),
    )


def _surface_object(raw: Mapping[str, Any]) -> _SurfaceObject | None:
    object_id = raw.get("object_id")
    material = raw.get("source_material_name")
    if not isinstance(object_id, str) or not object_id:
        raise ApartmentViewpointBuildError("M3 surface object_id must be non-empty")
    if not isinstance(material, str) or not material:
        raise ApartmentViewpointBuildError(
            f"M3 surface object {object_id!r} lacks source_material_name"
        )
    category = _material_category(material)
    if category is None:
        return None
    try:
        vertices = np.asarray(raw["vertices"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ApartmentViewpointBuildError(
            f"M3 surface object {object_id!r} has invalid vertices"
        ) from exc
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) < 3
        or not np.all(np.isfinite(vertices))
    ):
        raise ApartmentViewpointBuildError(
            f"M3 surface object {object_id!r} has invalid vertices"
        )
    return _make_surface_object(
        object_id=object_id,
        surface_id=object_id,
        source_material_name=material,
        category=category,
        vertices=vertices,
    )


def _interval_gap(
    left_low: float, left_high: float, right_low: float, right_high: float
) -> float:
    return max(0.0, right_low - left_high, left_low - right_high)


def _surface_gap_xz(left: _SurfaceObject, right: _SurfaceObject) -> float:
    x_gap = _interval_gap(
        left.low_xyz[0], left.high_xyz[0], right.low_xyz[0], right.high_xyz[0]
    )
    z_gap = _interval_gap(
        left.low_xyz[2], left.high_xyz[2], right.low_xyz[2], right.high_xyz[2]
    )
    return math.hypot(x_gap, z_gap)


def _spatial_components(
    surfaces: Sequence[_SurfaceObject],
    *,
    maximum_gap_m: float,
    maximum_span_m: float,
) -> tuple[tuple[_SurfaceObject, ...], ...]:
    ordered = tuple(sorted(surfaces, key=lambda item: item.stable_key))
    remaining = set(range(len(ordered)))
    components: list[tuple[_SurfaceObject, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque((start,))
        members = [start]
        component_low = np.asarray(ordered[start].low_xyz, dtype=np.float64)
        component_high = np.asarray(ordered[start].high_xyz, dtype=np.float64)
        while queue:
            current = queue.popleft()
            for candidate in sorted(tuple(remaining)):
                if (
                    _surface_gap_xz(ordered[current], ordered[candidate])
                    > maximum_gap_m + 1.0e-12
                ):
                    continue
                candidate_low = np.minimum(
                    component_low,
                    np.asarray(ordered[candidate].low_xyz, dtype=np.float64),
                )
                candidate_high = np.maximum(
                    component_high,
                    np.asarray(ordered[candidate].high_xyz, dtype=np.float64),
                )
                span_xz = candidate_high[[0, 2]] - candidate_low[[0, 2]]
                if float(np.max(span_xz)) > maximum_span_m + 1.0e-12:
                    continue
                remaining.remove(candidate)
                queue.append(candidate)
                members.append(candidate)
                component_low = candidate_low
                component_high = candidate_high
        components.append(tuple(ordered[index] for index in sorted(members)))
    return tuple(components)


def _split_surface_primitive(
    surface: _SurfaceObject,
) -> tuple[_SurfaceObject, ...]:
    """Split one M3 primitive into bounded local vertex-grid components."""

    grid_m = _CATEGORY_LOCAL_GRID_M[surface.category]
    xz = surface.vertices[:, (0, 2)]
    origin = np.min(xz, axis=0)
    grid_cells = np.floor((xz - origin) / grid_m + 1.0e-12).astype(np.int64)
    unique_cells, inverse = np.unique(grid_cells, axis=0, return_inverse=True)
    micro_surfaces = []
    for cell_index, cell in enumerate(unique_cells):
        vertices = surface.vertices[inverse == cell_index]
        cell_id = f"{surface.object_id}::grid_{int(cell[0])}_{int(cell[1])}"
        micro_surfaces.append(
            _make_surface_object(
                object_id=surface.object_id,
                surface_id=cell_id,
                source_material_name=surface.source_material_name,
                category=surface.category,
                vertices=vertices,
            )
        )
    components = _spatial_components(
        micro_surfaces,
        maximum_gap_m=_CATEGORY_LOCAL_GAP_M[surface.category],
        maximum_span_m=_CATEGORY_MAX_SPAN_M[surface.category],
    )
    result = []
    for index, component in enumerate(components, start=1):
        vertices = np.concatenate([item.vertices for item in component], axis=0)
        result.append(
            _make_surface_object(
                object_id=surface.object_id,
                surface_id=f"{surface.object_id}::local_{index:03d}",
                source_material_name=surface.source_material_name,
                category=surface.category,
                vertices=vertices,
            )
        )
    return tuple(result)


def _expanded_region_xz(low_xyz: np.ndarray, high_xyz: np.ndarray) -> list[list[float]]:
    low = low_xyz[[0, 2]].astype(np.float64, copy=True)
    high = high_xyz[[0, 2]].astype(np.float64, copy=True)
    for axis in range(2):
        if high[axis] - low[axis] < 0.25:
            midpoint = 0.5 * (high[axis] + low[axis])
            low[axis] = midpoint - 0.125
            high[axis] = midpoint + 0.125
    return [low.tolist(), high.tolist()]


def derive_semantic_targets_from_m3_objects(
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Derive deterministic local targets from true M3 material identities.

    Connected components are formed in XZ independently for living, kitchen,
    and window materials.  A distant pair of windows therefore remains two
    targets even when both primitives use the same material name.
    """

    if isinstance(objects, (str, bytes)) or not isinstance(objects, Sequence):
        raise ApartmentViewpointBuildError("M3 objects must be a sequence")
    classified = []
    for raw in objects:
        primitive = _surface_object(raw)
        if primitive is not None:
            classified.extend(_split_surface_primitive(primitive))
    by_category = {
        category: [item for item in classified if item.category == category]
        for category in _CATEGORY_ROLE
    }
    missing = [category for category, values in by_category.items() if not values]
    if missing:
        raise ApartmentViewpointBuildError(
            "M3 source_material_name objects provide no target for: "
            + ", ".join(missing)
        )

    pending: list[tuple[str, tuple[float, float, float], dict[str, Any]]] = []
    for category in _CATEGORY_ROLE:
        components = _spatial_components(
            by_category[category],
            maximum_gap_m=_CATEGORY_CLUSTER_GAP_M[category],
            maximum_span_m=_CATEGORY_MAX_SPAN_M[category],
        )
        for component in components:
            vertices = np.concatenate([item.vertices for item in component], axis=0)
            surface = np.median(vertices, axis=0)
            low = np.min(vertices, axis=0)
            high = np.max(vertices, axis=0)
            record: dict[str, Any] = {
                "role": _CATEGORY_ROLE[category],
                "surface_position_m": [float(item) for item in surface],
                "source_object_ids": sorted({item.object_id for item in component}),
                "source_local_component_ids": sorted(
                    item.surface_id for item in component
                ),
                "source_material_names": sorted(
                    {item.source_material_name for item in component}
                ),
                "source_vertex_count": int(len(vertices)),
                "local_component_bounds_m": [
                    [float(item) for item in low],
                    [float(item) for item in high],
                ],
                "target_derivation": (
                    "bounded_local_vertex_grid_components_within_each_m3_"
                    "primitive_then_deterministic_cross_primitive_xz_clustering"
                ),
            }
            if category != "window":
                record["region_xz_m"] = _expanded_region_xz(low, high)
            pending.append(
                (
                    category,
                    (float(surface[0]), float(surface[1]), float(surface[2])),
                    record,
                )
            )

    result: dict[str, dict[str, Any]] = {}
    category_order = {name: index for index, name in enumerate(_CATEGORY_ROLE)}
    counts = {category: 0 for category in _CATEGORY_ROLE}
    for category, _surface, record in sorted(
        pending,
        key=lambda item: (
            category_order[item[0]],
            item[1][0],
            item[1][2],
            item[1][1],
            tuple(item[2]["source_material_names"]),
            tuple(item[2]["source_object_ids"]),
        ),
    ):
        counts[category] += 1
        target_id = f"{category}_surface_{counts[category]:02d}"
        result[target_id] = record
    return result


def _deduplicated_numbers(values: Sequence[float]) -> list[float]:
    result: list[float] = []
    for value in sorted(float(item) for item in values):
        if not result or abs(value - result[-1]) > 1.0e-6:
            result.append(value)
    return result


def derive_anchor_probe_heights(
    anchor_library: Mapping[str, Any],
) -> dict[str, Any]:
    """Return actual absolute human/dog probe Y values from authored anchors."""

    anchors = (
        anchor_library.get("anchors") if isinstance(anchor_library, Mapping) else None
    )
    if not isinstance(anchors, list):
        raise ApartmentViewpointBuildError("anchor library must contain anchors")
    evidence: dict[str, list[dict[str, Any]]] = {"human": [], "dog": []}
    root_heights: list[float] = []
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            raise ApartmentViewpointBuildError("anchor record must be an object")
        anchor_id = anchor.get("anchor_id")
        if not isinstance(anchor_id, str):
            continue
        tokens = set(_material_token(anchor_id).split("_"))
        species = "human" if "human" in tokens else "dog" if "dog" in tokens else None
        if species is None:
            continue
        try:
            position = np.asarray(anchor["position_m"], dtype=np.float64)
            probe_offset = float(anchor["los_probe_height_m"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ApartmentViewpointBuildError(
                f"anchor {anchor_id!r} lacks a finite probe height"
            ) from exc
        if (
            position.shape != (3,)
            or not np.all(np.isfinite(position))
            or not math.isfinite(probe_offset)
            or probe_offset < 0.0
        ):
            raise ApartmentViewpointBuildError(
                f"anchor {anchor_id!r} lacks a finite probe height"
            )
        absolute = float(position[1] + probe_offset)
        root_heights.append(float(position[1]))
        evidence[species].append(
            {
                "anchor_id": anchor_id,
                "root_y_m": float(position[1]),
                "los_probe_height_m": probe_offset,
                "absolute_probe_y_m": absolute,
            }
        )
    missing = [species for species, records in evidence.items() if not records]
    if missing:
        raise ApartmentViewpointBuildError(
            "anchor library provides no actual probe anchors for: " + ", ".join(missing)
        )
    floors = _deduplicated_numbers(root_heights)
    if len(floors) != 1:
        raise ApartmentViewpointBuildError(
            "Apartment human/dog anchors do not share one authored floor Y"
        )
    species_record = {}
    all_probe_heights: list[float] = []
    for species in ("human", "dog"):
        absolute_values = _deduplicated_numbers(
            [record["absolute_probe_y_m"] for record in evidence[species]]
        )
        all_probe_heights.extend(absolute_values)
        species_record[species] = {
            "absolute_probe_heights_m": absolute_values,
            "anchors": sorted(evidence[species], key=lambda item: item["anchor_id"]),
        }
    return {
        "floor_height_m": floors[0],
        "species": species_record,
        "corridor_source_center_heights_m": _deduplicated_numbers(all_probe_heights),
    }


def _validate_m3_room_binding(
    m3_manifest: Mapping[str, Any], room_manifest: Mapping[str, Any]
) -> None:
    source_room = m3_manifest.get("source_room")
    if not isinstance(source_room, Mapping):
        raise ApartmentViewpointBuildError("M3 manifest lacks source_room binding")
    expected_room_id = room_manifest.get("room_id")
    actual_room_id = source_room.get("room_id")
    if not isinstance(actual_room_id, str) or actual_room_id != expected_room_id:
        raise ApartmentViewpointBuildError(
            "M3 source_room.room_id differs from the selected room manifest: "
            f"{actual_room_id!r} != {expected_room_id!r}"
        )
    provenance = room_manifest.get("provenance")
    expected_geometry = (
        provenance.get("exported_scene_sha256")
        if isinstance(provenance, Mapping)
        else None
    )
    actual_geometry = source_room.get("geometry_asset_sha256")
    if not isinstance(expected_geometry, str) or not expected_geometry:
        raise ApartmentViewpointBuildError(
            "room manifest lacks provenance.exported_scene_sha256"
        )
    if not isinstance(actual_geometry, str) or actual_geometry != expected_geometry:
        raise ApartmentViewpointBuildError(
            "M3 source_room.geometry_asset_sha256 differs from the selected room "
            f"manifest provenance: {actual_geometry!r} != {expected_geometry!r}"
        )


def _load_m3_surface_objects(
    path: Path, *, room_manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    scene = load_compiled_acoustic_scene(
        path,
        allow_nonpassing_research_qa=True,
    )
    _validate_m3_room_binding(scene.manifest, room_manifest)
    raw_objects = scene.manifest.get("objects")
    if not isinstance(raw_objects, list) or len(raw_objects) != len(scene.objects):
        raise ApartmentViewpointBuildError(
            "M3 manifest objects do not align with validated package vertices"
        )
    result = []
    for raw, native in zip(raw_objects, scene.objects, strict=True):
        if not isinstance(raw, Mapping) or raw.get("object_id") != native.get(
            "object_id"
        ):
            raise ApartmentViewpointBuildError(
                "M3 manifest object identity differs from validated vertices"
            )
        result.append(
            {
                "object_id": raw.get("object_id"),
                "source_material_name": raw.get("source_material_name"),
                "vertices": np.asarray(native.get("vertices"), dtype=np.float64),
            }
        )
    return tuple(result)


def _required_file(path: Path, *, owner: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{owner} is not a file: {resolved}")
    return resolved


def _required_directory(path: Path, *, owner: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{owner} is not a directory: {resolved}")
    return resolved


def _required_magnum_python_site() -> Path:
    value = os.environ.get("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE")
    if not value:
        raise FileNotFoundError(
            "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE must be explicitly set; "
            "the viewpoint builder has no runtime fallback"
        )
    return _required_directory(Path(value), owner="AVENGINE_HABITAT_MAGNUM_PYTHON_SITE")


@contextmanager
def _load_runtime_obstacle_map(
    *,
    inputs: ValidatedM1Inputs,
    runtime_prefix: Path,
    magnum_python_site: Path,
    rlr_sdk_root: Path | None = None,
    meters_per_pixel: float,
    scratch_directory: Path,
) -> Iterator[tuple[RuntimeObstacleMap, dict[str, Any], _NativeCameraFloorAuthority]]:
    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=runtime_prefix,
        magnum_python_site=magnum_python_site,
        rlr_sdk_root=rlr_sdk_root,
        allow_mp3d_environment=False,
    )
    configuration, _modalities, _listener_uuid, resolved_scene = _make_configuration(
        inputs,
        None,
        scratch_directory,
        include_audio_sensor=False,
        physics_config_path=runtime.physics_config_path,
    )
    if resolved_scene.get("navmesh_policy") != "load_declared":
        raise ApartmentViewpointBuildError(
            "Apartment viewpoint authoring requires scene.navmesh_policy=load_declared"
        )
    navmesh = resolved_scene.get("navmesh")
    if navmesh is None or not Path(navmesh).is_file():
        raise FileNotFoundError("Apartment declares no readable navmesh")
    with runtime.habitat_sim.Simulator(configuration) as simulator:
        if not simulator.pathfinder.load_nav_mesh(str(navmesh)):
            raise ApartmentViewpointBuildError(
                "Habitat could not load the explicitly declared Apartment navmesh"
            )
        camera_position = np.asarray(
            inputs.request["primary_camera_rig"]["world_from_rig"]["translation_m"],
            dtype=np.float64,
        )
        floor_point = np.asarray(
            simulator.pathfinder.snap_point(camera_position), dtype=np.float64
        )
        if floor_point.shape != (3,) or not np.all(np.isfinite(floor_point)):
            raise ApartmentViewpointBuildError(
                "legacy camera has no finite declared-NavMesh floor readback"
            )
        horizontal_snap_distance = float(
            np.linalg.norm(floor_point[[0, 2]] - camera_position[[0, 2]])
        )
        legacy_island_id = int(simulator.pathfinder.get_island(floor_point))
        floor_readback = {
            "authority": "declared_navmesh_pathfinder_snap_at_legacy_camera",
            "legacy_camera_position_m": [float(item) for item in camera_position],
            "floor_point_m": [float(item) for item in floor_point],
            "horizontal_snap_distance_m": horizontal_snap_distance,
            "native_floor_height_m": float(floor_point[1]),
            "legacy_island_id": legacy_island_id,
        }
        obstacle_map = build_runtime_obstacle_map(
            simulator.pathfinder,
            simulator.get_rigid_object_manager(),
            runtime.magnum,
            floor_height_m=float(floor_point[1]),
            meters_per_pixel=meters_per_pixel,
            excluded_handle_prefixes=("source_marker_",),
        )
        authority = _NativeCameraFloorAuthority(
            pathfinder=simulator.pathfinder,
            native_floor_height_m=float(floor_point[1]),
            legacy_island_id=legacy_island_id,
            maximum_horizontal_snap_m=PILOT_MAXIMUM_FLOOR_HORIZONTAL_SNAP_M,
            maximum_floor_y_delta_m=PILOT_MAXIMUM_FLOOR_Y_DELTA_M,
        )
        yield obstacle_map, floor_readback, authority


def _load_retained_corridors(path: Path) -> tuple[list[Mapping[str, Any]], str]:
    value = load_json(path)
    segments = value.get("segments")
    if not isinstance(segments, list) or any(
        not isinstance(item, Mapping) for item in segments
    ):
        raise ApartmentViewpointBuildError(
            "retained corridor bank must contain an object-list segments field"
        )
    coordinate_space = value.get("coordinate_space", "apartment_ue_cm")
    if coordinate_space not in ("apartment_ue_cm", "habitat_xz_m"):
        raise ApartmentViewpointBuildError(
            "retained corridor coordinate_space must be apartment_ue_cm or habitat_xz_m"
        )
    return segments, str(coordinate_space)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_prefilter_topdown(
    obstacle_map: RuntimeObstacleMap,
    result: ApartmentRasterPrefilterResult,
    semantic_targets: Mapping[str, Mapping[str, Any]],
    *,
    size_wh: tuple[int, int] = (1500, 1100),
) -> Image.Image:
    width, height = size_wh
    header = 125
    footer = 72
    margin = 28
    navmesh = np.asarray(obstacle_map.binary_navmesh, dtype=np.bool_)
    rgb = np.empty((*navmesh.shape, 3), dtype=np.uint8)
    rgb[:] = (35, 40, 47)
    rgb[navmesh] = (145, 154, 163)
    available_width = width - 2 * margin
    available_height = height - header - footer - 2 * margin
    scale = min(available_width / navmesh.shape[1], available_height / navmesh.shape[0])
    draw_width = max(2, int(round(navmesh.shape[1] * scale)))
    draw_height = max(2, int(round(navmesh.shape[0] * scale)))
    left = (width - draw_width) // 2
    top = header + margin + (available_height - draw_height) // 2
    image = Image.new("RGBA", (width, height), (22, 26, 31, 255))
    image.paste(
        Image.fromarray(rgb, mode="RGB").resize(
            (draw_width, draw_height), Image.Resampling.NEAREST
        ),
        (left, top),
    )
    draw = ImageDraw.Draw(image, "RGBA")
    bounds = np.asarray(obstacle_map.bounds_m, dtype=np.float64)

    def point_xz(value: Sequence[float]) -> tuple[float, float]:
        point = np.asarray(value, dtype=np.float64)
        low = bounds[0, (0, 2)]
        high = bounds[1, (0, 2)]
        uv = (point - low) / (high - low)
        return (
            float(left + uv[0] * (draw_width - 1)),
            float(top + uv[1] * (draw_height - 1)),
        )

    for obstacle in obstacle_map.rigid_obstacles:
        footprint = obstacle.get("footprint_xz_m")
        if footprint:
            polygon = [point_xz(item) for item in footprint]
            draw.polygon(polygon, fill=(160, 67, 67, 70))
            draw.line((*polygon, polygon[0]), fill=(237, 112, 112, 210), width=2)
    for corridor in result.corridors:
        draw.line(
            (point_xz(corridor.start_xz_m), point_xz(corridor.end_xz_m)),
            fill=(224, 226, 229, 55),
            width=2,
        )

    target_colors = {
        "living_reverse": (71, 196, 121, 255),
        "kitchen_reverse": (246, 176, 65, 255),
        "window_facing": (85, 173, 255, 255),
    }
    for target_id, record in semantic_targets.items():
        surface = record["surface_position_m"]
        x, y = point_xz((surface[0], surface[2]))
        color = target_colors[str(record["role"])]
        draw.rectangle((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(0, 0, 0, 255))
        draw.text((x + 8, y - 8), target_id, fill=color, font=_font(15))

    palette = (
        (255, 236, 95, 255),
        (242, 113, 113, 255),
        (205, 136, 255, 255),
        (94, 225, 221, 255),
        (255, 157, 82, 255),
        (151, 226, 102, 255),
    )
    room_extent = min(
        float(bounds[1, 0] - bounds[0, 0]),
        float(bounds[1, 2] - bounds[0, 2]),
    )
    ray_length_m = max(0.5, 0.08 * room_extent)
    for index, viewpoint in enumerate(result.viewpoints):
        color = palette[index % len(palette)]
        position_xz = (viewpoint.camera_position_m[0], viewpoint.camera_position_m[2])
        origin = point_xz(position_xz)
        basis = habitat_basis_from_yaw_degrees(viewpoint.camera_yaw_deg)
        endpoint_xz = np.asarray(position_xz) + ray_length_m * np.asarray(
            basis.forward_xz
        )
        endpoint = point_xz(endpoint_xz)
        draw.line((origin, endpoint), fill=color, width=5)
        draw.ellipse(
            (origin[0] - 8, origin[1] - 8, origin[0] + 8, origin[1] + 8),
            fill=color,
            outline=(0, 0, 0, 255),
            width=2,
        )
        draw.text(
            (origin[0] + 10, origin[1] + 6),
            viewpoint.viewpoint_id,
            fill=color,
            font=_font(16),
        )

    draw.rectangle((0, 0, width, header), fill=(0, 0, 0, 225))
    draw.text((28, 20), "APARTMENT VIEWPOINT DRAFT", fill="white", font=_font(30))
    draw.text((28, 65), PREFILTER_LABEL, fill=(255, 194, 72, 255), font=_font(24))
    draw.rectangle((0, height - footer, width, height), fill=(0, 0, 0, 225))
    draw.text(
        (28, height - footer + 16),
        (
            f"{len(result.viewpoints)} candidates | {len(result.corridors)} corridors | "
            "Studio not used | native first-person review still required"
        ),
        fill=(226, 230, 235, 255),
        font=_font(19),
    )
    return image


def _save_prefilter_topdown(path: Path, image: Image.Image) -> None:
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("artifact_status", "deterministic_2d_raster_prefilter_only")
    metadata.add_text("native_pixel_validation_status", "pending")
    metadata.add_text("claim_boundary", PREFILTER_LABEL)
    image.convert("RGB").save(path, pnginfo=metadata)


def _fresh_staging(
    output: Path,
) -> tuple[WorkspacePathPolicy, Path, Path]:
    unresolved = output.expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    staging_name = f".{unresolved.name}.staging"
    unresolved_staging = unresolved.with_name(staging_name)
    if os.path.lexists(unresolved) or os.path.lexists(unresolved_staging):
        raise ApartmentViewpointBuildError(
            f"refusing to replace output or staging path: {unresolved}"
        )
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    output_parent = unresolved.parent.resolve(strict=True)
    policy = WorkspacePathPolicy.from_roots((output_parent,))
    destination = policy.resolve_output(
        output_parent / unresolved.name,
        owner="Apartment viewpoint prefilter output",
    )
    staging = policy.resolve_output(
        output_parent / staging_name,
        owner="Apartment viewpoint prefilter staging directory",
    )
    staging.mkdir()
    return policy, destination, staging


def _request_identifier(base: Mapping[str, Any], viewpoint_id: str) -> str:
    base_id = base.get("request_id")
    if not isinstance(base_id, str) or not base_id:
        raise ApartmentViewpointBuildError("base M1 request_id must be non-empty")
    return f"{base_id}_{viewpoint_id}"


def _load_review_decision(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise ApartmentViewpointBuildError("review decision must be an object")
    decisions = value.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ApartmentViewpointBuildError(
            "review decision must contain a non-empty decisions list"
        )
    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    normalized = []
    seen_poses = set()
    for index, raw in enumerate(decisions):
        if not isinstance(raw, Mapping):
            raise ApartmentViewpointBuildError(
                f"review decisions[{index}] must be an object"
            )
        viewpoint_id = raw.get("viewpoint_id")
        decision = raw.get("decision")
        reason = raw.get("reason")
        camera = raw.get("camera")
        if (
            not isinstance(viewpoint_id, str)
            or not viewpoint_id
            or decision not in ("retain", "reject")
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(camera, Mapping)
        ):
            raise ApartmentViewpointBuildError(
                f"review decisions[{index}] is incomplete"
            )
        try:
            position = np.asarray(camera["habitat_translation_m"], dtype=np.float64)
            yaw = float(camera["habitat_yaw_deg"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ApartmentViewpointBuildError(
                f"review decisions[{index}] camera pose is invalid"
            ) from exc
        if (
            position.shape != (3,)
            or not np.all(np.isfinite(position))
            or not math.isfinite(yaw)
        ):
            raise ApartmentViewpointBuildError(
                f"review decisions[{index}] camera pose is invalid"
            )
        pose_key = (tuple(float(item) for item in position), yaw)
        if pose_key in seen_poses:
            raise ApartmentViewpointBuildError(
                "review decision contains duplicate exact camera poses"
            )
        pose = {"position_m": position.tolist(), "yaw_deg": yaw}
        (retained if decision == "retain" else rejected).append(pose)
        normalized.append(
            {
                "viewpoint_id": viewpoint_id,
                "decision": decision,
                "reason": reason.strip(),
                "camera": {
                    "habitat_translation_m": position.tolist(),
                    "habitat_yaw_deg": yaw,
                },
            }
        )
        seen_poses.add(pose_key)
    return (
        retained,
        rejected,
        {
            "source_prefilter": value.get("source_prefilter"),
            "decisions": normalized,
            "retained_count": len(retained),
            "rejected_count": len(rejected),
        },
    )


def run(args: argparse.Namespace) -> Path:
    if (
        isinstance(args.camera_yaw_step_degrees, bool)
        or not isinstance(args.camera_yaw_step_degrees, int)
        or args.camera_yaw_step_degrees <= 0
        or 360 % args.camera_yaw_step_degrees != 0
    ):
        raise ApartmentViewpointBuildError(
            "--camera-yaw-step-degrees must be a positive integer divisor of 360"
        )
    camera_yaw_degrees = tuple(
        float(value) for value in range(0, 360, args.camera_yaw_step_degrees)
    )
    room_manifest = _required_file(args.room_manifest, owner="--room-manifest")
    base_request_path = _required_file(args.base_m1_request, owner="--base-m1-request")
    m3_manifest = _required_file(
        args.m3_package_manifest, owner="--m3-package-manifest"
    )
    corridor_bank_path = _required_file(
        args.retained_corridor_bank, owner="--retained-corridor-bank"
    )
    anchor_library_path = _required_file(args.anchor_library, owner="--anchor-library")
    runtime_prefix = _required_directory(args.runtime_prefix, owner="--runtime-prefix")
    rlr_sdk_root = (
        None
        if args.rlr_sdk_root is None
        else _required_directory(args.rlr_sdk_root, owner="--rlr-sdk-root")
    )
    review_decision_path = (
        None
        if args.review_decision is None
        else _required_file(args.review_decision, owner="--review-decision")
    )
    retained_review_poses: list[dict[str, Any]] = []
    rejected_review_poses: list[dict[str, Any]] = []
    review_iteration = None
    if review_decision_path is not None:
        retained_review_poses, rejected_review_poses, review_iteration = (
            _load_review_decision(review_decision_path)
        )
    magnum_python_site = _required_magnum_python_site()
    output_policy, output, staging = _fresh_staging(args.output)

    inputs = load_and_validate_inputs(room_manifest, base_request_path)
    anchors = load_json(anchor_library_path)
    probe_heights = derive_anchor_probe_heights(anchors)
    retained_corridors, corridor_coordinate_space = _load_retained_corridors(
        corridor_bank_path
    )
    m3_objects = _load_m3_surface_objects(
        m3_manifest,
        room_manifest=inputs.room,
    )
    semantic_targets = derive_semantic_targets_from_m3_objects(m3_objects)

    with _load_runtime_obstacle_map(
        inputs=inputs,
        runtime_prefix=runtime_prefix,
        magnum_python_site=magnum_python_site,
        rlr_sdk_root=rlr_sdk_root,
        meters_per_pixel=float(args.meters_per_pixel),
        scratch_directory=staging / "runtime_scratch",
    ) as (obstacle_map, native_floor_readback, camera_floor_authority):
        camera_floor_feasible_mask = camera_floor_authority.feasible_mask(obstacle_map)
        result = prefilter_apartment_viewpoints(
            obstacle_map,
            base_m1_camera=inputs.request,
            semantic_targets=semantic_targets,
            corridor_segments=retained_corridors,
            corridor_coordinate_space=corridor_coordinate_space,
            generate_navmesh_corridors=True,
            corridor_source_center_heights_m=probe_heights[
                "corridor_source_center_heights_m"
            ],
            selection_count=6,
            camera_yaw_degrees=camera_yaw_degrees,
            maximum_ranked_camera_candidates=args.maximum_ranked_camera_candidates,
            minimum_raster_sample_fraction=(PILOT_MINIMUM_RASTER_SAMPLE_FRACTION),
            minimum_passed_corridor_count=(PILOT_MINIMUM_PASSED_CORRIDOR_COUNT),
            minimum_feasible_motion_mode_count=(
                PILOT_MINIMUM_FEASIBLE_MOTION_MODE_COUNT
            ),
            minimum_camera_eye_height_m=PILOT_MINIMUM_CAMERA_EYE_HEIGHT_M,
            generated_corridor_endpoint_limit=args.generated_corridor_endpoint_limit,
            generated_corridor_limit=args.generated_corridor_limit,
            camera_floor_feasible_mask=camera_floor_feasible_mask,
            retained_review_poses=retained_review_poses,
            rejected_review_poses=rejected_review_poses,
        )

        requests_directory = staging / "requests"
        requests_directory.mkdir()
        request_paths: dict[str, str] = {}
        for viewpoint in result.viewpoints:
            request = apply_camera_listener_pose(
                inputs.request,
                request_id=_request_identifier(inputs.request, viewpoint.viewpoint_id),
                position_m=viewpoint.camera_position_m,
                yaw_deg=viewpoint.camera_yaw_deg,
                horizontal_fov_deg=viewpoint.hfov_degrees,
            )
            if request["primary_camera_rig"].get("view_id") != "view0":
                raise ApartmentViewpointBuildError(
                    "authored M1 request changed the formal view_id"
                )
            relative = Path("requests") / f"{viewpoint.viewpoint_id}.json"
            write_json(staging / relative, request)
            request_paths[viewpoint.viewpoint_id] = relative.as_posix()

        semantic_record = {
            "artifact_status": "m3_geometry_derived_targets_for_2d_prefilter",
            "claim_boundary": (
                "M3 source-material identities and local geometry components only; "
                "native target pixels and composition remain pending"
            ),
            "source_object_count": len(m3_objects),
            "classification_roles": dict(_CATEGORY_ROLE),
            "semantic_targets": semantic_targets,
            "normalized_semantic_targets": [
                target.to_record() for target in result.semantic_targets
            ],
        }
        write_json(staging / "semantic_targets.json", semantic_record)

        prefilter_record = result.to_record()
        for viewpoint, record in zip(
            result.viewpoints, prefilter_record["viewpoints"], strict=True
        ):
            floor_evidence = camera_floor_authority.readback(
                viewpoint.camera_position_m
            )
            if floor_evidence["passed"] is not True:
                raise ApartmentViewpointBuildError(
                    f"selected viewpoint {viewpoint.viewpoint_id} failed native floor readback"
                )
            record["native_camera_floor_readback"] = floor_evidence
        prefilter_record.update(
            {
                "native_geometry_execution_status": "pass",
                "native_pixel_validation_status": "pending",
                "studio_used": False,
                "runtime_obstacle_map": obstacle_map.summary(),
                "native_floor_readback": native_floor_readback,
                "native_camera_floor_authority": {
                    "authority": "live_declared_navmesh_intended_floor_probe",
                    "legacy_island_id": camera_floor_authority.legacy_island_id,
                    "maximum_horizontal_snap_m": (
                        camera_floor_authority.maximum_horizontal_snap_m
                    ),
                    "maximum_floor_y_delta_m": (
                        camera_floor_authority.maximum_floor_y_delta_m
                    ),
                    "feasible_pixel_count": int(
                        np.count_nonzero(camera_floor_feasible_mask)
                    ),
                    "raster_shape_hw": list(camera_floor_feasible_mask.shape),
                },
                "source_probe_height_derivation": probe_heights,
                "review_iteration": review_iteration,
                "inputs": {
                    "room_manifest": str(room_manifest),
                    "base_m1_request": str(base_request_path),
                    "m3_package_manifest": str(m3_manifest),
                    "retained_corridor_bank": str(corridor_bank_path),
                    "anchor_library": str(anchor_library_path),
                    "installed_runtime_prefix": str(runtime_prefix),
                    "magnum_python_site": str(magnum_python_site),
                    "rlr_sdk_root": (
                        None if rlr_sdk_root is None else str(rlr_sdk_root)
                    ),
                    "review_decision": (
                        None
                        if review_decision_path is None
                        else str(review_decision_path)
                    ),
                    "generated_corridor_endpoint_limit": (
                        args.generated_corridor_endpoint_limit
                    ),
                    "generated_corridor_limit": args.generated_corridor_limit,
                    "maximum_ranked_camera_candidates": (
                        args.maximum_ranked_camera_candidates
                    ),
                    "camera_yaw_degrees": list(camera_yaw_degrees),
                },
                "request_paths": request_paths,
                "topdown_path": "topdown_2d_prefilter_native_pending.png",
            }
        )
        write_json(staging / "prefilter.json", prefilter_record)
        image = _render_prefilter_topdown(
            obstacle_map,
            result,
            semantic_targets,
        )
        _save_prefilter_topdown(
            staging / "topdown_2d_prefilter_native_pending.png",
            image,
        )

    return atomic_publish_directory(output_policy, staging, output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-manifest", required=True, type=Path)
    parser.add_argument("--base-m1-request", required=True, type=Path)
    parser.add_argument("--m3-package-manifest", required=True, type=Path)
    parser.add_argument("--retained-corridor-bank", required=True, type=Path)
    parser.add_argument("--anchor-library", required=True, type=Path)
    parser.add_argument("--runtime-prefix", required=True, type=Path)
    parser.add_argument("--rlr-sdk-root", type=Path)
    parser.add_argument("--review-decision", type=Path)
    parser.add_argument("--camera-yaw-step-degrees", type=int, default=30)
    parser.add_argument(
        "--generated-corridor-endpoint-limit",
        type=int,
        default=PILOT_GENERATED_CORRIDOR_ENDPOINT_LIMIT,
    )
    parser.add_argument(
        "--generated-corridor-limit",
        type=int,
        default=PILOT_GENERATED_CORRIDOR_LIMIT,
    )
    parser.add_argument(
        "--maximum-ranked-camera-candidates",
        type=int,
        default=PILOT_MAXIMUM_RANKED_CAMERA_CANDIDATES,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--meters-per-pixel", type=float, default=0.04)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Room furniture semantics, seated affordances, and target-independent views.

This module is the small CPU-side boundary between authored room sidecars and
AVEngine episode planning.  It consumes declared geometry only.  SPEAR/UE
readback, navmesh qualification, actor pose binding, and target LOS remain
later runtime concerns.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


TIME_BASE_HZ = 48_000
DEFAULT_CAMERA_HEIGHT_M = 1.55
DEFAULT_CAMERA_FOV_DEG = 90.0
DEFAULT_YAW_CANDIDATES_DEG = tuple(float(value) for value in range(0, 360, 30))
DEFAULT_PITCH_CANDIDATES_DEG = (-15.0, 0.0, 15.0)
DEFAULT_SEAT_COUNT = 4
DEFAULT_ACTOR_COUNT = 2


class FurnitureLayoutError(ValueError):
    """Room metadata cannot be normalized into the common planning shape."""


class SeatCapacityError(FurnitureLayoutError):
    """A requested actor/seat layout needs more affordances than the room has."""

    def __init__(self, requested: int, available: int) -> None:
        self.requested = requested
        self.available = available
        super().__init__(
            f"requested {requested} seated affordances but only {available} are available"
        )


def _finite(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FurnitureLayoutError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FurnitureLayoutError(f"{owner} must be a finite number")
    return result


def _vector(value: Any, size: int, *, owner: str) -> list[float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != size
    ):
        raise FurnitureLayoutError(f"{owner} must contain {size} numbers")
    return [_finite(item, owner=f"{owner}[{index}]") for index, item in enumerate(value)]


def _nonempty(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FurnitureLayoutError(f"{owner} must be a non-empty string")
    return value.strip()


def _read_mapping(path: Path, *, owner: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FurnitureLayoutError(f"cannot read {owner} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FurnitureLayoutError(f"{owner} {path} must contain a JSON object")
    return value


def _resolve_path(
    manifest_path: Path,
    raw_path: Any,
    *,
    asset_root: Path | None,
    owner: str,
    required: bool = True,
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        if required:
            raise FurnitureLayoutError(f"{owner} path is missing")
        return None
    raw = Path(raw_path).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(manifest_path.parent / raw)
        if asset_root is not None:
            candidates.append(asset_root / raw)
            # A's repository manifest names generated resources as tmp/...,
            # while an external asset root points at the generated room itself.
            if raw.parts and raw.parts[0] == "tmp":
                candidates.append(asset_root.joinpath(*raw.parts[1:]))
        candidates.append(Path.cwd() / raw)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    if required:
        formatted = ", ".join(str(item) for item in candidates)
        raise FurnitureLayoutError(f"{owner} does not resolve: {raw_path!r} ({formatted})")
    return None


def _load_sidecar(
    manifest_path: Path,
    raw_path: Any,
    *,
    asset_root: Path | None,
    owner: str,
) -> tuple[Mapping[str, Any], Path] | None:
    if raw_path is None:
        return None
    if isinstance(raw_path, Mapping):
        return raw_path, manifest_path
    path = _resolve_path(
        manifest_path, raw_path, asset_root=asset_root, owner=owner, required=True
    )
    assert path is not None
    return _read_mapping(path, owner=owner), path


def _artifacts(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("assets", "artifacts", "resources"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _bounds_from_xy(
    bounds_xy: Any, height: Any, *, owner: str
) -> list[list[float]]:
    xy = _vector(bounds_xy, 4, owner=f"{owner}.bounds_xy_m")
    height_m = _finite(height, owner=f"{owner}.height_m")
    if xy[2] <= xy[0] or xy[3] <= xy[1] or height_m < 0.0:
        raise FurnitureLayoutError(f"{owner} has invalid bounds")
    return [[xy[0], xy[1], 0.0], [xy[2], xy[3], height_m]]


def _normalize_object(raw: Mapping[str, Any], index: int) -> dict[str, Any]:
    owner = f"object[{index}]"
    object_id = _nonempty(
        _first_present(raw, ("object_id", "id", "name")), owner=f"{owner}.object_id"
    )
    category = str(
        _first_present(raw, ("semantic_class", "category", "class")) or "unknown"
    )
    role = str(raw.get("navigation_role") or "ground_blocker")

    raw_bounds = raw.get("bounds_xyz_m")
    if raw_bounds is not None:
        if (
            isinstance(raw_bounds, (str, bytes))
            or not isinstance(raw_bounds, Sequence)
            or len(raw_bounds) != 2
        ):
            raise FurnitureLayoutError(f"{owner}.bounds_xyz_m must be [minimum, maximum]")
        minimum = _vector(raw_bounds[0], 3, owner=f"{owner}.bounds minimum")
        maximum = _vector(raw_bounds[1], 3, owner=f"{owner}.bounds maximum")
        if any(maximum[axis] <= minimum[axis] for axis in range(3)):
            raise FurnitureLayoutError(f"{owner}.bounds_xyz_m must have positive extent")
        bounds = [minimum, maximum]
    elif raw.get("bounds_xy_m") is not None:
        bounds = _bounds_from_xy(
            raw["bounds_xy_m"],
            _first_present(raw, ("height_m", "height", "size_z_m")),
            owner=owner,
        )
    else:
        dimensions_value = _first_present(raw, ("dimensions_m", "size_xyz_m", "size_m"))
        if dimensions_value is None:
            raise FurnitureLayoutError(
                f"{owner} needs bounds_xyz_m, bounds_xy_m, dimensions_m, or size_xyz_m"
            )
        dimensions = _vector(dimensions_value, 3, owner=f"{owner}.dimensions_m")
        if any(value <= 0.0 for value in dimensions):
            raise FurnitureLayoutError(f"{owner}.dimensions_m must be positive")
        center_value = _first_present(
            raw, ("position_blender_m", "center_xyz_m", "position_xyz_m")
        )
        if center_value is not None:
            center = _vector(center_value, 3, owner=f"{owner}.center")
        else:
            center_xy = raw.get("center_xy_m")
            if center_xy is None:
                raise FurnitureLayoutError(f"{owner} center is missing")
            xy = _vector(center_xy, 2, owner=f"{owner}.center_xy_m")
            center = [xy[0], xy[1], dimensions[2] / 2.0]
        bounds = [
            [center[axis] - dimensions[axis] / 2.0 for axis in range(3)],
            [center[axis] + dimensions[axis] / 2.0 for axis in range(3)],
        ]

    dimensions = [bounds[1][axis] - bounds[0][axis] for axis in range(3)]
    center = [
        (bounds[0][axis] + bounds[1][axis]) / 2.0 for axis in range(3)
    ]
    return {
        "object_id": object_id,
        "semantic_class": category,
        "navigation_role": role,
        "static": bool(raw.get("static", not bool(raw.get("movable", False)))),
        "zone_id": raw.get("zone_id", raw.get("zone")),
        "center_authoring_m": center,
        "dimensions_m": dimensions,
        "bounds_xyz_m": bounds,
        "source": raw.get("source", "room_metadata"),
    }


def _object_records(value: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    for key in ("objects", "furniture", "items"):
        entries = value.get(key)
        if isinstance(entries, list):
            return [item for item in entries if isinstance(item, Mapping)]
    raise FurnitureLayoutError("object metadata must contain an objects/furniture list")


def _seat_position(raw: Mapping[str, Any], *, owner: str) -> list[float]:
    raw_position = _first_present(
        raw, ("position_blender_m", "position_xyz_m", "position_m", "center_xyz_m")
    )
    support = _first_present(raw, ("seat_surface_height_m", "support_height_m"))
    if raw_position is None:
        raise FurnitureLayoutError(f"{owner}.position_m is missing")
    if isinstance(raw_position, Sequence) and not isinstance(raw_position, (str, bytes)):
        if len(raw_position) == 2:
            xy = _vector(raw_position, 2, owner=f"{owner}.position")
            z = _finite(support if support is not None else 0.0, owner=f"{owner}.support_height_m")
            return [xy[0], xy[1], z]
        position = _vector(raw_position, 3, owner=f"{owner}.position")
    else:
        raise FurnitureLayoutError(f"{owner}.position_m must contain 2 or 3 numbers")
    if support is not None:
        support_m = _finite(support, owner=f"{owner}.support_height_m")
        # B/C authoring sidecars keep the support height separate and put zero
        # in position_m.  A's sidecar already stores the seat-top Z directly.
        if abs(position[2]) <= 1.0e-9 and support_m > 0.0:
            position[2] = support_m
    return position


def _normalize_seat(
    raw: Mapping[str, Any],
    index: int,
    *,
    furniture_id: str | None = None,
    furniture_category: str | None = None,
) -> dict[str, Any]:
    owner = f"seat[{index}]"
    seat_id = _nonempty(
        _first_present(raw, ("affordance_id", "anchor_id", "seat_id", "id")),
        owner=f"{owner}.affordance_id",
    )
    position = _seat_position(raw, owner=owner)
    facing = _finite(
        _first_present(raw, ("facing_yaw_deg", "yaw_deg")) or 0.0,
        owner=f"{owner}.facing_yaw_deg",
    )
    height = _finite(
        _first_present(raw, ("seat_surface_height_m", "support_height_m"))
        if _first_present(raw, ("seat_surface_height_m", "support_height_m")) is not None
        else position[2],
        owner=f"{owner}.seat_surface_height_m",
    )
    parent = _first_present(raw, ("furniture_id", "parent_object_id", "object_id"))
    resolved_parent = str(parent or furniture_id) if (parent or furniture_id) else None
    category = str(
        _first_present(raw, ("semantic_class", "category"))
        or furniture_category
        or "seat"
    )
    clearance = _first_present(raw, ("approach_clearance_radius_m", "clearance_m"))
    clearance_m = _finite(
        clearance if clearance is not None else 0.6,
        owner=f"{owner}.approach_clearance_radius_m",
    )
    if clearance_m <= 0.0:
        raise FurnitureLayoutError(f"{owner}.approach_clearance_radius_m must be positive")
    return {
        "affordance_id": seat_id,
        "furniture_id": resolved_parent,
        "semantic_class": category,
        "position_authoring_m": position,
        "position_habitat_m": [position[0], position[2], position[1]],
        "facing_yaw_deg": facing,
        "seat_surface_height_m": height,
        "approach_clearance_radius_m": clearance_m,
        "status": str(raw.get("status") or "authoring_candidate"),
        "reference_is_not_actor_root": True,
        "native_validation_status": "not_run",
    }


def _seat_records(value: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    for key in ("affordances", "seat_points", "seats"):
        entries = value.get(key)
        if isinstance(entries, list):
            return [item for item in entries if isinstance(item, Mapping)]
    return []


def _geometry_bounds(
    manifest: Mapping[str, Any],
    *,
    auxiliary: Mapping[str, Any] | None,
    objects: Sequence[Mapping[str, Any]],
    seats: Sequence[Mapping[str, Any]],
) -> tuple[list[float], str]:
    candidates: list[tuple[Any, str]] = []
    for source, owner in ((manifest, "manifest"), (auxiliary, "room_spec")):
        if not isinstance(source, Mapping):
            continue
        geometry = source.get("geometry")
        envelope = source.get("envelope")
        raw = None
        if isinstance(geometry, Mapping):
            raw = _first_present(geometry, ("bounds_xy_m", "bounds"))
        if raw is None and isinstance(envelope, Mapping):
            raw = envelope.get("bounds_xy_m")
        if raw is None:
            raw = source.get("bounds_xy_m")
        if raw is not None:
            candidates.append((raw, owner))
    for raw, owner in candidates:
        bounds = _vector(raw, 4, owner=f"{owner}.bounds_xy_m")
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise FurnitureLayoutError(f"{owner}.bounds_xy_m must have positive extent")
        return bounds, "declared_envelope"

    points: list[tuple[float, float]] = []
    for item in objects:
        bounds = item["bounds_xyz_m"]
        points.extend(((bounds[0][0], bounds[0][1]), (bounds[1][0], bounds[1][1])))
    for item in seats:
        position = item["position_authoring_m"]
        points.append((position[0], position[1]))
    if not points:
        raise FurnitureLayoutError("room has no geometry bounds or object positions")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    margin = 0.8
    return [min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin], "derived_from_static_metadata"


def _auxiliary_room_spec(manifest_path: Path, manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if manifest.get("envelope") is not None or manifest.get("bounds_xy_m") is not None:
        return None
    sibling = manifest_path.parent / "room_spec.json"
    if not sibling.exists():
        return None
    return _read_mapping(sibling, owner="auxiliary room spec")


def _resource_refs(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    asset_root: Path | None,
) -> dict[str, Any]:
    resources = _artifacts(manifest)
    refs: dict[str, Any] = {}
    aliases = {
        "editable_geometry": ("editable_blend", "blend"),
        "visual_geometry": ("visual_glb",),
        "collision_geometry": ("collision_glb",),
        "object_metadata": ("objects", "object_semantics"),
        "seat_metadata": ("seated_affordances", "functional_anchors"),
        "usd_stage": ("usd", "usd_stage"),
        "preview": ("preview", "preview_stills"),
    }
    for normalized, keys in aliases.items():
        raw = _first_present(resources, keys)
        if raw is None:
            continue
        if isinstance(raw, str):
            resolved = _resolve_path(
                manifest_path,
                raw,
                asset_root=asset_root,
                owner=f"resource {normalized}",
                required=False,
            )
            refs[normalized] = {
                "declared": raw,
                "resolved": str(resolved) if resolved is not None else None,
                "status": "available" if resolved is not None else "declared_unresolved",
            }
        else:
            refs[normalized] = {"declared": raw, "status": "inline"}
    return refs


def load_room_layout(
    manifest_path: str | Path,
    *,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load A/B/C-style room metadata into one object/seat representation.

    ``manifest_path`` may be an A-style resource manifest, a B/C room handoff,
    or a direct room spec containing ``furniture``.  The adapter is selected by
    field shape, never by room ID or room name.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_mapping(manifest_file, owner="room manifest")
    root = Path(asset_root).expanduser().resolve() if asset_root is not None else None
    auxiliary = _auxiliary_room_spec(manifest_file, manifest)
    resources = _artifacts(manifest)

    object_sidecar = _load_sidecar(
        manifest_file,
        _first_present(resources, ("objects", "object_semantics")),
        asset_root=root,
        owner="object metadata",
    )
    seat_sidecar = _load_sidecar(
        manifest_file,
        _first_present(resources, ("seated_affordances", "functional_anchors")),
        asset_root=root,
        owner="seat metadata",
    )

    object_source = object_sidecar[0] if object_sidecar is not None else manifest
    raw_objects = _object_records(object_source)
    # B/C semantic sidecars append functional anchors as object-shaped records.
    # They are references, not geometry blockers; physical objects still need
    # complete bounds and therefore cannot silently take this path.
    raw_objects = [
        item
        for item in raw_objects
        if str(item.get("category") or item.get("semantic_class") or "").lower()
        not in {"anchor", "functional_anchor", "waypoint", "viewpoint"}
    ]
    objects = [_normalize_object(item, index) for index, item in enumerate(raw_objects)]
    object_by_id = {item["object_id"]: item for item in objects}

    raw_seats: list[tuple[Mapping[str, Any], str | None, str | None]] = []
    for item in _seat_records(seat_sidecar[0] if seat_sidecar is not None else None):
        raw_seats.append((item, None, None))
    # B/C object sidecars carry seat points as part of each semantic object.
    # The sidecar and functional-anchor formats overlap, so dedupe by anchor ID.
    for raw, normalized in zip(raw_objects, objects, strict=True):
        raw_parent = normalized["object_id"]
        raw_category = normalized["semantic_class"]
        for seat in raw.get("seat_points", []) if isinstance(raw, Mapping) else []:
            if isinstance(seat, Mapping):
                raw_seats.append((seat, raw_parent, raw_category))
    if seat_sidecar is None:
        # A direct room spec has no sidecar and keeps seat points under furniture.
        for raw, normalized in zip(raw_objects, objects, strict=True):
            for seat in raw.get("seat_points", []) if isinstance(raw, Mapping) else []:
                if isinstance(seat, Mapping):
                    raw_seats.append((seat, normalized["object_id"], normalized["semantic_class"]))

    seats: list[dict[str, Any]] = []
    seen_seat_ids: set[str] = set()
    for index, (raw, parent, category) in enumerate(raw_seats):
        seat = _normalize_seat(raw, index, furniture_id=parent, furniture_category=category)
        if seat["affordance_id"] in seen_seat_ids:
            continue
        if seat["furniture_id"] is not None and seat["furniture_id"] not in object_by_id:
            # A functional-anchor sidecar can omit the parent object.  Keep the
            # reference unresolved but make that boundary explicit.
            seat["furniture_resolution"] = "unresolved_sidecar_reference"
        else:
            seat["furniture_resolution"] = "resolved" if seat["furniture_id"] else "not_declared"
        seen_seat_ids.add(seat["affordance_id"])
        seats.append(seat)

    # Functional-anchor sidecars may omit their parent object.  Enrich those
    # references from duplicate object-level seat points when available.
    seats_by_id = {item["affordance_id"]: item for item in seats}
    category_by_seat: dict[str, str] = {}
    category_priority: dict[str, int] = {}
    for raw, normalized in zip(raw_objects, objects, strict=True):
        category = str(normalized["semantic_class"])
        category_lower = category.lower()
        priority = (
            3
            if any(token in category_lower for token in ("chair", "sofa", "bench", "stool"))
            else 1
        )
        for raw_seat in raw.get("seat_points", []) if isinstance(raw, Mapping) else []:
            if not isinstance(raw_seat, Mapping):
                continue
            raw_id = _first_present(
                raw_seat, ("affordance_id", "anchor_id", "seat_id", "id")
            )
            if raw_id not in seats_by_id:
                continue
            if priority >= category_priority.get(str(raw_id), -1):
                category_by_seat[str(raw_id)] = category
                category_priority[str(raw_id)] = priority
            item = seats_by_id[raw_id]
            if item.get("furniture_id") is None:
                item["furniture_id"] = normalized["object_id"]
                item["furniture_resolution"] = "resolved"
    for item in seats:
        seat_id = item["affordance_id"]
        if seat_id in category_by_seat:
            item["semantic_class"] = category_by_seat[seat_id]
        elif item.get("semantic_class") == "seat" and item.get("furniture_id") in object_by_id:
            item["semantic_class"] = object_by_id[item["furniture_id"]]["semantic_class"]
    if not seats:
        raise FurnitureLayoutError("room metadata declares no seated affordances")

    bounds_xy_m, bounds_source = _geometry_bounds(
        manifest, auxiliary=auxiliary, objects=objects, seats=seats
    )
    room_id = _nonempty(
        _first_present(manifest, ("room_id", "room_spec_id", "scene_id"))
        or (auxiliary or {}).get("room_spec_id"),
        owner="room_id",
    )
    scene = manifest.get("scene") if isinstance(manifest.get("scene"), Mapping) else {}
    native_execution = (
        manifest.get("native_execution")
        if isinstance(manifest.get("native_execution"), Mapping)
        else {}
    )
    map_path = _first_present(
        scene, ("map_path", "map", "stage_path")
    ) or _first_present(manifest, ("map_path", "stage_path"))
    if map_path is None:
        map_path = native_execution.get("map_or_stage")
    if not isinstance(map_path, str) or not map_path.strip():
        map_path = None
    lighting = manifest.get("visual_lighting")
    if not isinstance(lighting, Mapping):
        lighting = {
            "status": "not_declared",
            "native_validation_status": "not_run",
            "claim_boundary": "room metadata did not declare production lighting",
        }
    return {
        "room_id": room_id,
        "scene_id": str(scene.get("scene_id") or manifest.get("scene_id") or room_id),
        "room_family_id": manifest.get("room_family_id"),
        "status": str(manifest.get("status") or "research_candidate"),
        "backend_route": manifest.get("backend_route") or manifest.get("backend_intent"),
        "map_path": map_path,
        "manifest_path": str(manifest_file),
        "coordinate_contract": {
            "authoring": "room-local +Z-up metres",
            "habitat": "[authoring_x, authoring_z, authoring_y]",
        },
        "geometry": {
            "bounds_xy_m": bounds_xy_m,
            "bounds_source": bounds_source,
            "authoring_geometry_status": "candidate",
            "native_validation_status": "not_run",
            "claim_boundary": (
                "declared room envelope and static sidecars only; no native SPEAR/UE readback"
            ),
        },
        "objects": objects,
        "seats": seats,
        "resources": _resource_refs(manifest_file, manifest, asset_root=root),
        "visual_lighting": deepcopy(dict(lighting)),
        "camera_source": "geometry_grid_only",
        "review_cameras_used": False,
    }


def authoring_to_habitat(position_m: Sequence[float]) -> list[float]:
    point = _vector(position_m, 3, owner="authoring position")
    return [point[0], point[2], point[1]]


def habitat_to_ue_cm(position_m: Sequence[float]) -> list[float]:
    habitat = _vector(position_m, 3, owner="Habitat position")
    return [100.0 * habitat[0], 100.0 * habitat[2], 100.0 * habitat[1]]


def _ground_blocker(item: Mapping[str, Any]) -> bool:
    role = str(item.get("navigation_role") or "ground_blocker")
    category = str(item.get("semantic_class") or "").lower()
    return role not in {"walkable_surface", "walkable_floor_covering", "elevated_object"} and category not in {
        "floor",
        "rug",
        "carpet",
    }


def _free_camera_point(
    x: float,
    y: float,
    objects: Sequence[Mapping[str, Any]],
    *,
    clearance_m: float,
) -> bool:
    for item in objects:
        if not _ground_blocker(item):
            continue
        bounds = item["bounds_xyz_m"]
        if (
            bounds[0][0] - clearance_m <= x <= bounds[1][0] + clearance_m
            and bounds[0][1] - clearance_m <= y <= bounds[1][1] + clearance_m
        ):
            return False
    return True


def generate_camera_candidates(
    layout: Mapping[str, Any],
    *,
    grid_step_m: float = 2.0,
    camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M,
    yaw_candidates_deg: Sequence[float] = DEFAULT_YAW_CANDIDATES_DEG,
    pitch_candidates_deg: Sequence[float] = DEFAULT_PITCH_CANDIDATES_DEG,
    clearance_m: float = 0.35,
    horizontal_fov_deg: float = DEFAULT_CAMERA_FOV_DEG,
) -> dict[str, Any]:
    """Generate a target-independent geometry grid and its full yaw/pitch cross.

    The function accepts only room geometry.  Actors, questions and target
    anchors are intentionally absent; target-aware scoring is a later join.
    """

    bounds = _vector(
        layout.get("geometry", {}).get("bounds_xy_m"),
        4,
        owner="layout.geometry.bounds_xy_m",
    )
    step = _finite(grid_step_m, owner="grid_step_m")
    height = _finite(camera_height_m, owner="camera_height_m")
    clearance = _finite(clearance_m, owner="clearance_m")
    fov = _finite(horizontal_fov_deg, owner="horizontal_fov_deg")
    if step <= 0.0 or height <= 0.0 or clearance < 0.0 or not 0.0 < fov < 180.0:
        raise FurnitureLayoutError("camera grid parameters are invalid")
    yaws = [_finite(value, owner="yaw candidate") for value in yaw_candidates_deg]
    pitches = [_finite(value, owner="pitch candidate") for value in pitch_candidates_deg]
    if not yaws or not pitches:
        raise FurnitureLayoutError("camera yaw and pitch candidate sets cannot be empty")

    margin = max(clearance, 0.15)
    x_span = max(0.0, bounds[2] - bounds[0] - 2.0 * margin)
    y_span = max(0.0, bounds[3] - bounds[1] - 2.0 * margin)
    x_count = max(1, int(math.floor(x_span / step)) + 1)
    y_count = max(1, int(math.floor(y_span / step)) + 1)
    x_values = [bounds[0] + margin + index * step for index in range(x_count)]
    y_values = [bounds[1] + margin + index * step for index in range(y_count)]
    # Ensure a narrow room still contributes a central geometry point.
    if not x_values:
        x_values = [(bounds[0] + bounds[2]) / 2.0]
    if not y_values:
        y_values = [(bounds[1] + bounds[3]) / 2.0]

    positions: list[tuple[float, float]] = []
    for x in x_values:
        for y in y_values:
            if _free_camera_point(x, y, layout.get("objects", []), clearance_m=clearance):
                positions.append((x, y))
    if not positions:
        raise FurnitureLayoutError("geometry grid has no camera point after clearance")

    candidates: list[dict[str, Any]] = []
    for point_index, (x, y) in enumerate(positions):
        authoring_position = [x, y, height]
        habitat_position = authoring_to_habitat(authoring_position)
        ue_position = habitat_to_ue_cm(habitat_position)
        for yaw in yaws:
            for pitch in pitches:
                candidate_id = (
                    f"grid_{point_index:03d}_yaw_{yaw:g}_pitch_{pitch:g}"
                )
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "geometry_point_id": f"grid_{point_index:03d}",
                        "position_authoring_m": list(authoring_position),
                        "position_habitat_m": list(habitat_position),
                        "position_ue_cm": list(ue_position),
                        "ue_position_cm": list(ue_position),
                        "yaw_deg": yaw,
                        "pitch_deg": pitch,
                        "roll_deg": 0.0,
                        "habitat_yaw_deg": yaw,
                        "ue_yaw_deg": (-90.0 - yaw + 180.0) % 360.0 - 180.0,
                        "ue_pitch_deg": pitch,
                        "horizontal_fov_deg": fov,
                        "target_independent": True,
                        "target_los_status": "not_evaluated",
                        "authoring_geometry_status": "candidate",
                        "native_validation_status": "not_run",
                        "claim_boundary": (
                            "declared geometry clearance only; no target LOS or native readback"
                        ),
                    }
                )
    return {
        "generation": {
            "source": "authoring_geometry_grid",
            "target_independent": True,
            "review_cameras_used": False,
            "grid_step_m": step,
            "camera_height_m": height,
            "clearance_m": clearance,
            "yaw_candidates_deg": list(yaws),
            "pitch_candidates_deg": list(pitches),
            "geometry_point_count": len(positions),
            "candidate_count": len(candidates),
            "native_validation_status": "not_run",
        },
        "candidates": candidates,
    }


def _target_position(
    target_position_m: Sequence[float] | None,
    actor_positions_m: Sequence[Sequence[float]] | None,
    question_context: Mapping[str, Any] | None,
) -> list[float] | None:
    if target_position_m is None and isinstance(question_context, Mapping):
        context_target = question_context.get("target_position_m")
        if context_target is not None:
            target_position_m = context_target
    if target_position_m is not None:
        return _vector(target_position_m, 3, owner="target_position_m")
    if actor_positions_m:
        points = [_vector(point, 3, owner="actor position") for point in actor_positions_m]
        return [sum(point[axis] for point in points) / len(points) for axis in range(3)]
    return None


def score_camera_candidates(
    candidate_set: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    target_position_m: Sequence[float] | None = None,
    actor_positions_m: Sequence[Sequence[float]] | None = None,
    question_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score an already generated set after actor/question selection.

    Scoring annotates copies and never adds/removes geometry candidates.  It is
    deliberately a small hook for a later task/question policy, not a LOS
    gate.  Candidate legality remains the geometry/native validation boundary.
    """

    if isinstance(candidate_set, Mapping):
        raw_candidates = candidate_set.get("candidates")
        generation = deepcopy(dict(candidate_set.get("generation", {})))
    else:
        raw_candidates = candidate_set
        generation = {}
    if not isinstance(raw_candidates, Sequence):
        raise FurnitureLayoutError("camera candidate set must contain candidates")
    target = _target_position(target_position_m, actor_positions_m, question_context)
    scored: list[dict[str, Any]] = []
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise FurnitureLayoutError("camera candidate is not a mapping")
        candidate = deepcopy(dict(raw))
        score = 0.0
        if target is not None:
            position = _vector(
                candidate.get("position_authoring_m"),
                3,
                owner="camera candidate position_authoring_m",
            )
            dx = target[0] - position[0]
            dy = target[1] - position[1]
            dz = target[2] - position[2]
            horizontal = math.hypot(dx, dy)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            desired_yaw = math.degrees(math.atan2(dy, dx))
            yaw_error = abs((desired_yaw - float(candidate["yaw_deg"]) + 180.0) % 360.0 - 180.0)
            desired_pitch = math.degrees(math.atan2(dz, horizontal)) if horizontal else 0.0
            pitch_error = abs(desired_pitch - float(candidate["pitch_deg"]))
            score = -(distance + yaw_error / 90.0 + pitch_error / 45.0)
        candidate["post_join_score"] = score
        candidate["score_context"] = "actor_question_join" if target is not None else "none"
        scored.append(candidate)
    scored.sort(key=lambda item: (-float(item["post_join_score"]), str(item["candidate_id"])))
    generation["scoring_applied_after_target_join"] = target is not None
    generation["target_los_evaluated"] = False
    return {"generation": generation, "candidates": scored}


def select_seats(layout: Mapping[str, Any], count: int = DEFAULT_SEAT_COUNT) -> list[dict[str, Any]]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise FurnitureLayoutError("seat count must be a positive integer")
    raw_seats = layout.get("seats")
    if not isinstance(raw_seats, Sequence):
        raise FurnitureLayoutError("layout.seats must be a list")
    def seat_priority(item: Mapping[str, Any]) -> tuple[int, str]:
        category = str(item.get("semantic_class") or "").lower()
        if any(token in category for token in ("chair", "stool")):
            priority = 0
        elif "bench" in category:
            priority = 1
        elif "table" in category:
            priority = 2
        elif "sofa" in category or "couch" in category:
            priority = 3
        else:
            priority = 4
        return priority, str(item.get("affordance_id"))

    seats = sorted(
        (deepcopy(dict(item)) for item in raw_seats if isinstance(item, Mapping)),
        key=seat_priority,
    )
    if len(seats) < count:
        raise SeatCapacityError(count, len(seats))
    return seats[:count]


def _pose_binding_records(pose_bindings: Any) -> list[dict[str, Any]]:
    if pose_bindings is None:
        return []
    value = pose_bindings
    if isinstance(value, Mapping) and isinstance(value.get("bindings"), (list, Mapping)):
        value = value["bindings"]
    elif isinstance(value, Mapping) and isinstance(value.get("assets"), list):
        # The pose-agent UE import request is an asset pool.  It can be joined
        # to selected room seats without treating reference actor yaw as a
        # room-facing direction.
        value = value["assets"]
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for actor_id, raw in value.items():
            if not isinstance(raw, Mapping):
                raise FurnitureLayoutError(f"pose binding {actor_id!r} must be an object")
            item = dict(raw)
            item.setdefault("actor_id", actor_id)
            records.append(item)
    elif isinstance(value, list):
        for index, raw in enumerate(value):
            if not isinstance(raw, Mapping):
                raise FurnitureLayoutError(f"pose binding[{index}] must be an object")
            item = dict(raw)
            seat_reference = item.get("seat_reference")
            if isinstance(seat_reference, Mapping):
                item.setdefault("seat_affordance_id", seat_reference.get("seat_anchor_id"))
                item.setdefault(
                    "root_from_seat_m",
                    seat_reference.get("root_offset_from_seat_anchor_blender_m"),
                )
                item.setdefault(
                    "reference_chair_yaw_degrees",
                    seat_reference.get("reference_chair_yaw_degrees"),
                )
                item.setdefault("pose_seat_top_m", seat_reference.get("seat_top_m"))
            item.setdefault("actor_id", item.get("asset_id") or f"actor{index}")
            item.setdefault("ue_animation", item.get("animation") or item.get("animation_name"))
            item.setdefault("blueprint_class_path", item.get("blueprint"))
            item.setdefault("skeletal_mesh_path", item.get("skeletal_mesh"))
            item.setdefault(
                "ue_anatomical_forward_yaw_deg",
                item.get("ue_anatomical_forward_yaw_deg", 90.0),
            )
            emitter = item.get("emitter_offset_avengine_m")
            if emitter is not None and item.get("emitter_local_ue_cm") is None:
                emitter_m = _vector(
                    emitter,
                    3,
                    owner=f"pose binding {item['actor_id']}.emitter_offset_avengine_m",
                )
                # AVEngine asset basis X-forward/Y-up/Z-right metres -> UE
                # X-forward/Y-right/Z-up centimetres.
                item["emitter_local_ue_cm"] = [
                    100.0 * emitter_m[0],
                    100.0 * emitter_m[2],
                    100.0 * emitter_m[1],
                ]
            if item.get("blueprint_class_path") and item.get("skeletal_mesh_path"):
                item.setdefault("actor_scale", 1.0)
                item.setdefault(
                    "animation_paths_by_action_id",
                    {"seated_idle": item.get("ue_animation")},
                )
                item.setdefault(
                    "exact_runtime_binding",
                    {"source": "seated_human_ue_import_manifest", "status": "declared"},
                )
                item.setdefault(
                    "ue_component_frame_delta",
                    {
                        "rotation_deg": [0.0, 0.0, 0.0],
                        "translation_cm": [0.0, 0.0, 0.0],
                        "composition": "add_relative_preserving_blueprint_transform",
                        "reason": "pose import manifest supplies the Blueprint frame; no additional correction declared",
                    },
                )
            records.append(item)
    else:
        raise FurnitureLayoutError("pose bindings must be a list or actor mapping")
    return records


def _seat_alias_key(value: str) -> tuple[str, ...]:
    tokens = [token for token in value.lower().replace("-", "_").split("_") if token]
    ignored = {"seat", "sit", "affordance", "anchor", "chair", "stool"}
    return tuple(token for token in tokens if token not in ignored)


def _resolve_seat_id(requested: str, available: Mapping[str, Mapping[str, Any]]) -> str:
    if requested in available:
        return requested
    alias = _seat_alias_key(requested)
    matches = [seat_id for seat_id in available if _seat_alias_key(seat_id) == alias]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FurnitureLayoutError(
            f"pose seat reference {requested!r} is ambiguous: {sorted(matches)}"
        )
    raise FurnitureLayoutError(
        f"pose binding selects unavailable seat {requested!r}; available={sorted(available)}"
    )


def build_seat_placements(
    layout: Mapping[str, Any],
    *,
    seat_count: int = DEFAULT_SEAT_COUNT,
    actor_count: int = DEFAULT_ACTOR_COUNT,
    pose_bindings: Any = None,
) -> dict[str, Any]:
    """Select seat references and optionally apply pose-agent offsets.

    A seat point is never used as an actor root.  A bound actor needs an
    explicit nonzero ``root_from_seat_m`` offset from the pose agent; otherwise
    the returned actor placement remains pending and contains only the seat
    reference.
    """

    if isinstance(actor_count, bool) or not isinstance(actor_count, int) or actor_count <= 0:
        raise FurnitureLayoutError("actor count must be a positive integer")
    selected = select_seats(layout, seat_count)
    by_id = {item["affordance_id"]: item for item in selected}
    bindings = _pose_binding_records(pose_bindings)
    pose_asset_pool = isinstance(pose_bindings, Mapping) and isinstance(
        pose_bindings.get("assets"), list
    )
    if bindings and len(bindings) > actor_count and not pose_asset_pool:
        actor_count = len(bindings)
    if pose_asset_pool:
        bindings = bindings[:actor_count]
    if not bindings:
        bindings = [{"actor_id": f"actor{index}"} for index in range(actor_count)]

    actor_placements: list[dict[str, Any]] = []
    used_seats: set[str] = set()
    for index, raw in enumerate(bindings):
        actor_id = _nonempty(raw.get("actor_id") or f"actor{index}", owner="pose binding.actor_id")
        seat_id = raw.get("seat_affordance_id") or raw.get("seat_id")
        if seat_id is None:
            if index >= len(selected):
                raise SeatCapacityError(index + 1, len(selected))
            seat_id = selected[index]["affordance_id"]
        seat_id = _nonempty(seat_id, owner=f"pose binding {actor_id}.seat_affordance_id")
        seat_id = _resolve_seat_id(seat_id, by_id)
        if seat_id in used_seats:
            raise FurnitureLayoutError(f"seat {seat_id!r} is assigned to multiple actors")
        used_seats.add(seat_id)
        seat = by_id[seat_id]
        root_from_seat = raw.get("root_from_seat_m")
        if root_from_seat is None:
            root_from_seat = raw.get("root_offset_from_seat_anchor_blender_m")
        if root_from_seat is not None:
            # The pose request offset is actor-local Blender XYZ.  The room
            # seat yaw is chair-to-table; the seated asset's local frame is
            # placed at seat_theta + 90 degrees.  Reference chair/actor yaw
            # values are calibration context only and never drive a new room.
            offset = _vector(
                root_from_seat,
                3,
                owner=f"pose binding {actor_id}.root_from_seat_m",
            )
            placement_yaw = math.radians(float(seat["facing_yaw_deg"]) + 90.0)
            root_from_seat = [
                offset[0] * math.cos(placement_yaw) - offset[1] * math.sin(placement_yaw),
                offset[0] * math.sin(placement_yaw) + offset[1] * math.cos(placement_yaw),
                offset[2],
            ]
            pose_seat_top = raw.get("pose_seat_top_m")
            if pose_seat_top is not None:
                # Room metadata exposes the seat surface; the pose request
                # offset's Z is relative to its floor/root reference.  Remove
                # the calibrated seat-top height before adding it to the room
                # seat point, otherwise the actor would float by ~0.53 m.
                root_from_seat[2] -= _finite(
                    pose_seat_top, owner=f"pose binding {actor_id}.pose_seat_top_m"
                )
        root_authoring: list[float] | None = None
        root_habitat: list[float] | None = None
        rotation = raw.get("rotation_xyzw")
        if rotation is not None:
            rotation = _vector(rotation, 4, owner=f"pose binding {actor_id}.rotation_xyzw")
        if root_from_seat is not None:
            offset = _vector(root_from_seat, 3, owner=f"pose binding {actor_id}.root_from_seat_m")
            if math.sqrt(sum(value * value for value in offset)) <= 1.0e-6:
                raise FurnitureLayoutError(
                    f"pose binding {actor_id!r} root_from_seat_m cannot be zero"
                )
            root_authoring = [
                seat["position_authoring_m"][axis] + offset[axis] for axis in range(3)
            ]
            root_habitat = authoring_to_habitat(root_authoring)
            placement_status = "bound"
        else:
            placement_status = "pending_pose_binding"
        actor_placements.append(
            {
                "actor_id": actor_id,
                "seat_affordance_id": seat_id,
                "seat_reference": {
                    "position_authoring_m": deepcopy(seat["position_authoring_m"]),
                    "position_habitat_m": deepcopy(seat["position_habitat_m"]),
                    "seat_surface_height_m": seat["seat_surface_height_m"],
                    "facing_yaw_deg": seat["facing_yaw_deg"],
                    "reference_is_not_actor_root": True,
                },
                "root_from_seat_m": list(root_from_seat) if root_from_seat is not None else None,
                "pose_seat_top_m": raw.get("pose_seat_top_m"),
                "root_position_authoring_m": root_authoring,
                "root_position_habitat_m": root_habitat,
                "rotation_xyzw": rotation,
                "placement_status": placement_status,
                "pose_binding_status": "provided" if root_from_seat is not None else "pending",
                "asset_id": raw.get("asset_id"),
                "template_id": raw.get("template_id"),
                "body_plan_id": raw.get("body_plan_id"),
                "blueprint_class_path": raw.get("blueprint_class_path"),
                "skeletal_mesh_path": raw.get("skeletal_mesh_path"),
                "ue_animation": raw.get("ue_animation"),
                "animation_paths_by_action_id": deepcopy(raw.get("animation_paths_by_action_id")),
                "exact_runtime_binding": deepcopy(raw.get("exact_runtime_binding")),
                "ue_component_frame_delta": deepcopy(raw.get("ue_component_frame_delta")),
                "emitter_local_ue_cm": deepcopy(raw.get("emitter_local_ue_cm")),
                "ue_anatomical_forward_yaw_deg": raw.get("ue_anatomical_forward_yaw_deg"),
                "actor_scale": raw.get("actor_scale", 1.0),
                "ue_asset_destination": raw.get("destination"),
                "pose_orientation_policy": "seat_theta_plus_90_local_offset; reference_actor_yaw_ignored",
            }
        )
    return {
        "requested_seat_count": seat_count,
        "available_seat_count": len(layout.get("seats", [])),
        "selected_seat_ids": [item["affordance_id"] for item in selected],
        "selected_seats": selected,
        "actor_placements": actor_placements,
        "placement_policy": "seat_reference_then_pose_binding",
        "authoring_geometry_status": "candidate",
        "native_validation_status": "not_run",
        "claim_boundary": (
            "seat references come from authored metadata; actor root and pose need pose-agent binding"
        ),
    }


def clock_config(
    *,
    frame_count: int,
    frame_rate_hz: float = 15.0,
    sample_rate_hz: int = 16_000,
) -> dict[str, Any]:
    if isinstance(frame_count, bool) or frame_count not in (75, 150):
        raise FurnitureLayoutError("frame_count must be 75 or 150")
    fps = _finite(frame_rate_hz, owner="frame_rate_hz")
    if fps <= 0.0:
        raise FurnitureLayoutError("frame_rate_hz must be positive")
    if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
        raise FurnitureLayoutError("sample_rate_hz must be a positive integer")
    ticks_per_frame = TIME_BASE_HZ / fps
    if abs(ticks_per_frame - round(ticks_per_frame)) > 1.0e-9:
        raise FurnitureLayoutError("frame_rate_hz must divide the AVEngine time base")
    ticks = int(round(ticks_per_frame))
    sample_count = int(round(frame_count * sample_rate_hz / fps))
    return {
        "frame_count": frame_count,
        "frame_rate_hz": fps,
        "sample_rate_hz": sample_rate_hz,
        "time_base_hz": TIME_BASE_HZ,
        "ticks_per_frame": ticks,
        "sample_count": sample_count,
        "duration_seconds": frame_count / fps,
    }

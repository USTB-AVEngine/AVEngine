"""Bounded MP3D region camera selection from declared camera candidates.

This module joins already-produced camera placements to parsed .house regions.
It does not run Habitat, infer visibility, or make a camera a formal gate. It
adapts the retained region-view selection slice from c13551f to the current
routes owner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Any, Mapping, Sequence

from avengine.rooms.mp3d_regions import MP3DHouseFloorPlan, MP3DRegionError


class MP3DRegionViewError(ValueError):
    """Declared region camera candidates cannot be selected."""


@dataclass(frozen=True)
class RegionCameraBinding:
    region_index: int
    region_instance_id: str
    selection_order: int
    placement_id: str
    floor_sample_index: int | None
    floor_position_m: tuple[float, float, float]
    position_m: tuple[float, float, float]
    yaw_deg: float
    height_id: str | None
    request_path: str | None
    boundary_distance_m: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "region_index": self.region_index,
            "region_instance_id": self.region_instance_id,
            "selection_order": self.selection_order,
            "placement_id": self.placement_id,
            "floor_sample_index": self.floor_sample_index,
            "floor_position_m": list(self.floor_position_m),
            "position_m": list(self.position_m),
            "yaw_deg": self.yaw_deg,
            "height_id": self.height_id,
            "request_path": self.request_path,
            "boundary_distance_m": self.boundary_distance_m,
        }


def _number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool):
        raise MP3DRegionViewError(f"{owner} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionViewError(f"{owner} must be finite") from exc
    if not math.isfinite(result):
        raise MP3DRegionViewError(f"{owner} must be finite")
    return result


def _point(value: Any, *, owner: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MP3DRegionViewError(f"{owner} must contain three numbers")
    if len(value) != 3:
        raise MP3DRegionViewError(f"{owner} must contain three numbers")
    return tuple(
        _number(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _optional_int(value: Any, *, owner: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise MP3DRegionViewError(f"{owner} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionViewError(f"{owner} must be an integer") from exc
    if str(value).strip() != str(result):
        raise MP3DRegionViewError(f"{owner} must be an integer")
    return result


def _candidate_from_record(
    raw: Mapping[str, Any],
    *,
    region_index: int,
    region_instance_id: str,
) -> RegionCameraBinding:
    camera = raw.get("camera")
    camera_record = camera if isinstance(camera, Mapping) else raw
    placement = raw.get("placement_id", camera_record.get("placement_id"))
    if not isinstance(placement, str) or not placement:
        raise MP3DRegionViewError("camera candidate placement_id is missing")
    floor_value = camera_record.get("floor_position_m")
    position_value = camera_record.get("position_m")
    if floor_value is None:
        floor_value = raw.get("floor_position_m")
    if position_value is None:
        position_value = raw.get("position_m")
    floor = _point(floor_value, owner=f"{placement}.floor_position_m")
    position = _point(position_value, owner=f"{placement}.position_m")
    floor_index = _optional_int(
        camera_record.get("floor_sample_index", raw.get("floor_sample_index")),
        owner=f"{placement}.floor_sample_index",
    )
    yaw = _number(
        camera_record.get("yaw_deg", raw.get("yaw_deg")),
        owner=f"{placement}.yaw_deg",
    )
    boundary = raw.get(
        "distance_to_nearest_polygon_boundary_m",
        raw.get("boundary_distance_m"),
    )
    boundary_distance = (
        None if boundary is None else _number(boundary, owner=f"{placement}.boundary")
    )
    request_path = raw.get(
        "m1_capture_request_path",
        raw.get("request_path", camera_record.get("m1_capture_request_path")),
    )
    if request_path is not None and (
        not isinstance(request_path, str) or not request_path
    ):
        raise MP3DRegionViewError(f"{placement}.request_path must be a string")
    height = camera_record.get("height_id", raw.get("height_id"))
    if height is not None and (not isinstance(height, str) or not height):
        raise MP3DRegionViewError(f"{placement}.height_id must be a string")
    return RegionCameraBinding(
        region_index=region_index,
        region_instance_id=region_instance_id,
        selection_order=0,
        placement_id=placement,
        floor_sample_index=floor_index,
        floor_position_m=floor,
        position_m=position,
        yaw_deg=yaw,
        height_id=height,
        request_path=request_path,
        boundary_distance_m=boundary_distance,
    )


def _iter_candidate_records(
    sidecar: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    records: list[tuple[str, Mapping[str, Any]]] = []
    placements = sidecar.get("placement_memberships")
    if isinstance(placements, list):
        for raw in placements:
            if not isinstance(raw, Mapping):
                raise MP3DRegionViewError("placement_memberships entries must be objects")
            region_id = raw.get("primary_region_instance_id")
            if isinstance(region_id, str) and region_id:
                records.append((region_id, raw))
        return tuple(records)

    banks = sidecar.get("region_banks", sidecar.get("regions"))
    if isinstance(banks, list):
        for bank in banks:
            if not isinstance(bank, Mapping):
                raise MP3DRegionViewError("region camera banks must contain objects")
            region_id = bank.get("region_instance_id")
            if not isinstance(region_id, str) or not region_id:
                continue
            selected = bank.get("selected_candidates", bank.get("cameras"))
            if not isinstance(selected, list):
                continue
            records.extend(
                (region_id, item)
                for item in selected
                if isinstance(item, Mapping)
            )
        return tuple(records)
    raise MP3DRegionViewError(
        "camera input needs placement_memberships or region_banks"
    )


def _region_selection(
    plan: MP3DHouseFloorPlan,
    region: Any,
    raw_candidates: Sequence[Mapping[str, Any]],
    *,
    cameras_per_region: int,
    seed: int,
    minimum_floor_separation_m: float,
    require_unique_floor: bool,
) -> tuple[RegionCameraBinding, ...]:
    candidates: list[RegionCameraBinding] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        try:
            candidate = _candidate_from_record(
                raw,
                region_index=region.region_index,
                region_instance_id=region.region_instance_id,
            )
            membership = raw.get("membership_status")
            if membership is not None and membership != "unique":
                continue
            if not region.contains(candidate.floor_position_m):
                continue
        except (MP3DRegionError, MP3DRegionViewError):
            raise
        if candidate.placement_id in seen:
            raise MP3DRegionViewError(
                f"region {region.region_instance_id} repeats placement "
                f"{candidate.placement_id}"
            )
        seen.add(candidate.placement_id)
        candidates.append(candidate)
    if not candidates:
        raise MP3DRegionViewError(
            f"region {region.region_instance_id} has no unique in-region cameras"
        )

    rng = random.Random(int(seed) + int(region.region_index) * 1_000_003)
    decorated = [(rng.random(), item) for item in candidates]
    decorated.sort(
        key=lambda pair: (
            -(
                pair[1].boundary_distance_m
                if pair[1].boundary_distance_m is not None
                else 0.0
            ),
            pair[0],
            pair[1].placement_id,
        )
    )
    selected: list[RegionCameraBinding] = []
    used_floors: set[int] = set()
    for _random_value, candidate in decorated:
        if require_unique_floor and candidate.floor_sample_index in used_floors:
            continue
        if minimum_floor_separation_m > 0.0 and any(
            math.dist(candidate.floor_position_m, prior.floor_position_m)
            < minimum_floor_separation_m
            for prior in selected
        ):
            continue
        selected.append(candidate)
        if candidate.floor_sample_index is not None:
            used_floors.add(candidate.floor_sample_index)
        if len(selected) == cameras_per_region:
            break
    if len(selected) != cameras_per_region:
        raise MP3DRegionViewError(
            f"region {region.region_instance_id} has only {len(selected)} "
            f"cameras after the requested distinct-floor/spacing selection"
        )
    return tuple(
        replace(candidate, selection_order=order)
        for order, candidate in enumerate(selected, start=1)
    )


def select_region_cameras(
    plan: MP3DHouseFloorPlan,
    sidecar: Mapping[str, Any],
    *,
    region_indices: Sequence[int] | None = None,
    cameras_per_region: int = 1,
    seed: int = 0,
    minimum_floor_separation_m: float = 0.0,
    require_unique_floor: bool = True,
) -> dict[str, Any]:
    """Select a bounded camera subset for requested .house regions.

    Input camera records are already declared placements; selection does not
    claim that a candidate passes live navmesh, LOS or pixel validation.
    """

    if not isinstance(plan, MP3DHouseFloorPlan):
        raise MP3DRegionViewError("plan must be an MP3DHouseFloorPlan")
    if not isinstance(sidecar, Mapping):
        raise MP3DRegionViewError("sidecar must be an object")
    if isinstance(cameras_per_region, bool) or not isinstance(cameras_per_region, int):
        raise MP3DRegionViewError("cameras_per_region must be a positive integer")
    if cameras_per_region < 1:
        raise MP3DRegionViewError("cameras_per_region must be a positive integer")
    separation = _number(
        minimum_floor_separation_m,
        owner="minimum_floor_separation_m",
    )
    if separation < 0.0:
        raise MP3DRegionViewError("minimum_floor_separation_m must be nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MP3DRegionViewError("seed must be an integer")

    by_index = plan.by_region_index
    if region_indices is None:
        selected_indices = tuple(sorted(by_index))
    else:
        selected_indices = tuple(int(value) for value in region_indices)
        if len(set(selected_indices)) != len(selected_indices):
            raise MP3DRegionViewError("region_indices must be unique")
        if any(value not in by_index for value in selected_indices):
            raise MP3DRegionViewError("region_indices contains an unknown region")
    candidate_groups: dict[str, list[Mapping[str, Any]]] = {}
    for region_id, raw in _iter_candidate_records(sidecar):
        candidate_groups.setdefault(region_id, []).append(raw)

    outputs = []
    for index in selected_indices:
        region = by_index[index]
        raw = candidate_groups.get(region.region_instance_id, [])
        selected = _region_selection(
            plan,
            region,
            raw,
            cameras_per_region=cameras_per_region,
            seed=seed,
            minimum_floor_separation_m=separation,
            require_unique_floor=require_unique_floor,
        )
        outputs.append(
            {
                **region.label_record(),
                "camera_count": len(selected),
                "cameras": [item.to_record() for item in selected],
            }
        )
    return {
        "artifact_kind": "mp3d_region_camera_plan",
        "research_only": True,
        "episode_counted": False,
        "house_id": plan.house_id,
        "region_count": len(outputs),
        "requested_region_indices": list(selected_indices),
        "cameras_per_region": cameras_per_region,
        "selection": {
            "seed": int(seed),
            "minimum_floor_separation_m": separation,
            "require_unique_floor": bool(require_unique_floor),
            "policy": (
                "bounded candidate subset ranked by boundary distance with "
                "seeded tie order; region membership is descriptive"
            ),
        },
        "regions": outputs,
    }


__all__ = [
    "MP3DRegionViewError",
    "RegionCameraBinding",
    "select_region_cameras",
]
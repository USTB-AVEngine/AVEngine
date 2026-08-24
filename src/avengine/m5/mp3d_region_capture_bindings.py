"""Post-hoc MP3D camera/route bindings for four-point mini-pilots.

Camera candidates and source routes remain independently generated. This
module retains every camera x route-family pairing, computes target-agnostic
source-center projection descriptors, then chooses one deterministic bijection
per region only to schedule native mini-pilot work. Projection never changes
candidate legality or actor roots.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.m1.contracts import validate_capture_request
from avengine.m2.habitat_capture import quaternion_xyzw_to_matrix


MOTION_CASES = (
    "static_static",
    "s1_moving_s2_static",
    "s1_static_s2_moving",
    "both_moving",
)
SCORE_FIELDS = (
    "at_least_one_center_frame_count",
    "both_centers_frame_count",
    "source_center_in_frustum_count",
)


class MP3DRegionCaptureBindingError(RuntimeError):
    """A post-hoc camera/route binding artifact cannot be trusted."""


def _fail(message: str) -> MP3DRegionCaptureBindingError:
    return MP3DRegionCaptureBindingError(message)


def _load_object(path: str | Path, *, owner: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path)
    if candidate.is_symlink():
        raise _fail(f"{owner} must not be a symlink: {candidate}")
    try:
        resolved = candidate.expanduser().resolve(strict=True)
    except OSError as exc:
        raise _fail(f"{owner} is missing: {candidate}") from exc
    if not resolved.is_file():
        raise _fail(f"{owner} is not a regular file: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(f"{owner} is not valid JSON: {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise _fail(f"{owner} must contain one JSON object")
    return resolved, payload


def _relative_file(root: Path, raw: Any, *, owner: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise _fail(f"{owner} must be a relative path")
    candidate = root / raw
    if candidate.is_symlink():
        raise _fail(f"{owner} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise _fail(f"{owner} is missing or escapes its artifact: {raw}") from exc
    if not resolved.is_file():
        raise _fail(f"{owner} is not a regular file: {resolved}")
    return resolved


def _finite_vector(value: Any, *, shape: tuple[int, ...], owner: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _fail(f"{owner} must be numeric") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise _fail(f"{owner} must be finite with shape {shape}")
    return array


@dataclass(frozen=True)
class CameraModel:
    placement_id: str
    selection_order: int
    m1_request_path: Path
    position_m: np.ndarray
    world_from_camera_rotation: np.ndarray
    near_m: float
    far_m: float
    horizontal_tangent_limit: float
    vertical_tangent_limit: float


def load_camera_model(
    candidate: Mapping[str, Any], *, viewpoint_root: Path
) -> CameraModel:
    placement_id = candidate.get("placement_id")
    selection_order = candidate.get("selection_order")
    if not isinstance(placement_id, str) or not placement_id:
        raise _fail("viewpoint candidate lacks placement_id")
    if isinstance(selection_order, bool) or not isinstance(selection_order, int):
        raise _fail(f"viewpoint {placement_id} selection_order is invalid")
    m1_path = _relative_file(
        viewpoint_root,
        candidate.get("m1_capture_request_path"),
        owner=f"viewpoint {placement_id} M1 request",
    )
    _, request = _load_object(m1_path, owner=f"viewpoint {placement_id} M1 request")
    errors = validate_capture_request(request)
    if errors:
        raise _fail(f"viewpoint {placement_id} M1 request differs: {'; '.join(errors)}")
    rig = request["primary_camera_rig"]
    transform = rig["world_from_rig"]
    calibration = rig["shared_calibration"]
    position = _finite_vector(
        transform["translation_m"], shape=(3,), owner=f"{placement_id} position"
    )
    declared_position = _finite_vector(
        candidate.get("camera", {}).get("position_m"),
        shape=(3,),
        owner=f"{placement_id} selected position",
    )
    if not np.allclose(position, declared_position, rtol=0.0, atol=1.0e-9):
        raise _fail(f"viewpoint {placement_id} M1 position differs from bank")
    rotation = np.asarray(
        quaternion_xyzw_to_matrix(transform["rotation_xyzw"]), dtype=np.float64
    )
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise _fail(f"viewpoint {placement_id} rotation is invalid")
    height, width = calibration["resolution_hw"]
    hfov = float(calibration["hfov_degrees"])
    near_m = float(calibration["near_m"])
    far_m = float(calibration["far_m"])
    if not (
        isinstance(height, int)
        and isinstance(width, int)
        and height > 0
        and width > 0
        and math.isfinite(hfov)
        and 0.0 < hfov < 180.0
        and 0.0 < near_m < far_m
    ):
        raise _fail(f"viewpoint {placement_id} calibration is invalid")
    horizontal = math.tan(math.radians(hfov) / 2.0)
    vertical = horizontal * float(height) / float(width)
    return CameraModel(
        placement_id=placement_id,
        selection_order=selection_order,
        m1_request_path=m1_path,
        position_m=position,
        world_from_camera_rotation=rotation,
        near_m=near_m,
        far_m=far_m,
        horizontal_tangent_limit=horizontal,
        vertical_tangent_limit=vertical,
    )


def project_case_source_centers(
    camera: CameraModel,
    source_positions_m: Mapping[str, Any],
) -> dict[str, Any]:
    source_ids = ("source1", "source2")
    arrays = [
        _finite_vector(
            source_positions_m.get(source_id),
            shape=(75, 3),
            owner=f"{source_id} positions",
        )
        for source_id in source_ids
    ]
    per_source: dict[str, Any] = {}
    masks = []
    frame_records = [{"frame_index": index, "sources": {}} for index in range(75)]
    for source_id, points in zip(source_ids, arrays, strict=True):
        camera_local = (
            camera.world_from_camera_rotation.T @ (points - camera.position_m).T
        ).T
        forward_depth = -camera_local[:, 2]
        positive = forward_depth > 0.0
        horizontal = np.full(75, np.nan, dtype=np.float64)
        vertical = np.full(75, np.nan, dtype=np.float64)
        horizontal[positive] = camera_local[positive, 0] / forward_depth[positive]
        vertical[positive] = camera_local[positive, 1] / forward_depth[positive]
        inside = (
            (forward_depth >= camera.near_m)
            & (forward_depth <= camera.far_m)
            & (np.abs(horizontal) <= camera.horizontal_tangent_limit)
            & (np.abs(vertical) <= camera.vertical_tangent_limit)
        )
        distances = np.linalg.norm(points - camera.position_m, axis=1)
        azimuth = np.degrees(np.arctan2(camera_local[:, 0], forward_depth))
        elevation = np.degrees(
            np.arctan2(
                camera_local[:, 1],
                np.hypot(camera_local[:, 0], forward_depth),
            )
        )
        masks.append(inside)
        per_source[source_id] = {
            "center_in_frustum_frame_count": int(np.count_nonzero(inside)),
            "minimum_listener_distance_m": float(np.min(distances)),
            "maximum_listener_distance_m": float(np.max(distances)),
            "minimum_forward_depth_m": float(np.min(forward_depth)),
            "maximum_forward_depth_m": float(np.max(forward_depth)),
        }
        for index in range(75):
            frame_records[index]["sources"][source_id] = {
                "world_position_m": points[index].tolist(),
                "camera_local_position_m": camera_local[index].tolist(),
                "forward_depth_m": float(forward_depth[index]),
                "horizontal_tangent": (
                    float(horizontal[index]) if positive[index] else None
                ),
                "vertical_tangent": (
                    float(vertical[index]) if positive[index] else None
                ),
                "listener_distance_m": float(distances[index]),
                "azimuth_degrees": float(azimuth[index]),
                "elevation_degrees": float(elevation[index]),
                "center_in_frustum": bool(inside[index]),
            }
    at_least_one = masks[0] | masks[1]
    both = masks[0] & masks[1]
    return {
        "authority": "M1 pinhole projection of route source centers only",
        "whole_actor_visibility_authority": False,
        "candidate_legality_effect": "none",
        "source_summaries": per_source,
        "at_least_one_center_frame_count": int(np.count_nonzero(at_least_one)),
        "both_centers_frame_count": int(np.count_nonzero(both)),
        "source_center_in_frustum_count": int(
            np.count_nonzero(masks[0]) + np.count_nonzero(masks[1])
        ),
        "frames": frame_records,
    }


def select_bijective_pairings(
    pairings: Sequence[Mapping[str, Any]],
    *,
    camera_ids: Sequence[str],
    route_family_ids: Sequence[str],
) -> list[dict[str, Any]]:
    lookup = {
        (str(item["placement_id"]), str(item["route_family_id"])): item
        for item in pairings
    }
    if len(lookup) != len(camera_ids) * len(route_family_ids):
        raise _fail("pairing grid is incomplete or duplicated")
    ranked = []
    for permutation in itertools.permutations(route_family_ids):
        selected = [
            lookup[(camera_id, family_id)]
            for camera_id, family_id in zip(camera_ids, permutation, strict=True)
        ]
        totals = tuple(
            sum(int(item["descriptor_score"][field]) for item in selected)
            for field in SCORE_FIELDS
        )
        ranked.append((totals, tuple(permutation), selected))
    ranked.sort(key=lambda item: tuple(-value for value in item[0]) + item[1])
    totals, _, selected = ranked[0]
    return [
        {
            "placement_id": item["placement_id"],
            "route_family_id": item["route_family_id"],
            "m1_request_path": item["m1_request_path"],
            "descriptor_score": dict(item["descriptor_score"]),
            "assignment_total_score": dict(zip(SCORE_FIELDS, totals, strict=True)),
            "selection_role": (
                "descriptor-ranked mini-pilot scheduling only; not candidate legality"
            ),
        }
        for item in selected
    ]


def build_region_capture_binding_bank(
    *,
    viewpoint_bank_path: str | Path,
    route_bank_path: str | Path,
    actor_tracks_root: str | Path,
    output_directory: str | Path,
) -> Path:
    viewpoint_path, viewpoint_bank = _load_object(
        viewpoint_bank_path, owner="viewpoint bank"
    )
    route_path, route_bank = _load_object(route_bank_path, owner="route bank")
    actor_root = Path(actor_tracks_root).expanduser().resolve(strict=True)
    if not actor_root.is_dir() or actor_root.is_symlink():
        raise _fail("actor_tracks_root must be a non-symlink directory")
    output = Path(output_directory).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise _fail(f"refusing to replace output: {output}")

    region_banks = viewpoint_bank.get("region_banks")
    route_families = route_bank.get("families")
    if not isinstance(region_banks, list) or len(region_banks) != 10:
        raise _fail("viewpoint bank must contain ten region banks")
    if not isinstance(route_families, list) or len(route_families) != 40:
        raise _fail("route bank must contain forty route families")
    families_by_region: dict[str, list[Mapping[str, Any]]] = {}
    for family in route_families:
        if not isinstance(family, Mapping):
            raise _fail("route families must be objects")
        families_by_region.setdefault(str(family.get("region_instance_id")), []).append(
            family
        )

    region_outputs = []
    all_pairing_count = 0
    scheduled_point_count = 0
    for region_bank in sorted(region_banks, key=lambda item: item["region_index"]):
        region_index = int(region_bank["region_index"])
        region_id = str(region_bank["region_instance_id"])
        candidates = region_bank.get("selected_candidates")
        families = sorted(
            families_by_region.get(region_id, []),
            key=lambda item: int(item["camera_binding"]["selection_order"]),
        )
        if (
            not isinstance(candidates, list)
            or len(candidates) != 4
            or len(families) != 4
        ):
            raise _fail(f"region {region_id} must have four cameras and families")
        cameras = [
            load_camera_model(candidate, viewpoint_root=viewpoint_path.parent)
            for candidate in sorted(
                candidates, key=lambda item: item["selection_order"]
            )
        ]
        pilot_path = actor_root / f"region_{region_index:03d}" / "pilot_manifest.json"
        pilot_root = pilot_path.parent
        _, pilot = _load_object(pilot_path, owner=f"{region_id} actor pilot")
        if pilot.get("region_instance_id") != region_id:
            raise _fail(f"{region_id} actor pilot identity differs")
        case_lookup = {}
        for record in pilot.get("cases", []):
            key = (record.get("route_family_id"), record.get("motion_case"))
            case_path = _relative_file(
                pilot_root,
                record.get("case_manifest"),
                owner=f"{region_id} case manifest",
            )
            if key in case_lookup:
                raise _fail(f"{region_id} actor pilot repeats case {key}")
            case_lookup[key] = (str(record.get("case_id")), case_path)
        if len(case_lookup) != 16:
            raise _fail(f"{region_id} actor pilot must contain sixteen cases")

        pairings = []
        for camera in cameras:
            for family in families:
                case_descriptors = {}
                for motion_case in MOTION_CASES:
                    case = family.get("cases", {}).get(motion_case)
                    if not isinstance(case, Mapping):
                        raise _fail(
                            f"{region_id}/{family.get('route_family_id')} lacks {motion_case}"
                        )
                    case_descriptors[motion_case] = project_case_source_centers(
                        camera,
                        {
                            "source1": case.get("source1_positions_m"),
                            "source2": case.get("source2_positions_m"),
                        },
                    )
                descriptor_score = {
                    field: sum(
                        int(case_descriptors[motion][field]) for motion in MOTION_CASES
                    )
                    for field in SCORE_FIELDS
                }
                pairings.append(
                    {
                        "placement_id": camera.placement_id,
                        "camera_selection_order": camera.selection_order,
                        "m1_request_path": str(camera.m1_request_path),
                        "route_family_id": str(family["route_family_id"]),
                        "route_generation_used_camera": False,
                        "candidate_legality_effect": "none",
                        "descriptor_score": descriptor_score,
                        "motion_case_descriptors": case_descriptors,
                    }
                )
        selected = select_bijective_pairings(
            pairings,
            camera_ids=[camera.placement_id for camera in cameras],
            route_family_ids=[str(family["route_family_id"]) for family in families],
        )
        scheduled = []
        for assignment in selected:
            for motion_case in MOTION_CASES:
                case_id, case_path = case_lookup[
                    (assignment["route_family_id"], motion_case)
                ]
                scheduled.append(
                    {
                        "point_index": len(scheduled),
                        "placement_id": assignment["placement_id"],
                        "route_family_id": assignment["route_family_id"],
                        "motion_case": motion_case,
                        "case_id": case_id,
                        "case_manifest_path": str(case_path),
                        "m1_request_path": assignment["m1_request_path"],
                        "native_visual_status": "pending",
                        "rlr_status": "pending",
                        "qa_status": "pending",
                        "gate_status": "pending",
                    }
                )
        if len(scheduled) != 16:
            raise _fail(f"{region_id} mini-pilot schedule must contain sixteen points")
        all_pairing_count += len(pairings)
        scheduled_point_count += len(scheduled)
        region_outputs.append(
            {
                "region_index": region_index,
                "region_instance_id": region_id,
                "camera_count": len(cameras),
                "route_family_count": len(families),
                "all_pairing_count": len(pairings),
                "selected_binding_count": len(selected),
                "scheduled_point_count": len(scheduled),
                "all_pairings": pairings,
                "selected_bindings": selected,
                "four_point_mini_pilots": scheduled,
            }
        )

    artifact = {
        "schema": "avengine_m5_mp3d_region_capture_binding_bank_v1",
        "status": "complete_cpu_posthoc_capture_binding_bank",
        "research_only": True,
        "episode_counted": False,
        "candidate_filtering": "none",
        "candidate_legality_effect": "none",
        "camera_route_generation_decoupled": True,
        "assignment_policy": (
            "one camera-to-route-family bijection per region ranked only by retained "
            "source-center projection descriptors; all 16 pairings remain present"
        ),
        "inputs": {
            "viewpoint_bank": str(viewpoint_path),
            "route_bank": str(route_path),
            "actor_tracks_root": str(actor_root),
        },
        "region_count": len(region_outputs),
        "viewpoint_count": sum(item["camera_count"] for item in region_outputs),
        "all_pairing_count": all_pairing_count,
        "selected_binding_count": sum(
            item["selected_binding_count"] for item in region_outputs
        ),
        "scheduled_point_count": scheduled_point_count,
        "regions": region_outputs,
        "claim_boundary": (
            "CPU post-hoc binding and source-center projection descriptors only; "
            "center projection is not whole-actor visibility and does not affect "
            "camera/source legality. Native pixels, collision, RLR, QA and gates are pending."
        ),
    }
    output.mkdir(parents=True)
    (output / "capture_binding_bank.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


__all__ = [
    "CameraModel",
    "MOTION_CASES",
    "MP3DRegionCaptureBindingError",
    "build_region_capture_binding_bank",
    "load_camera_model",
    "project_case_source_centers",
    "select_bijective_pairings",
]

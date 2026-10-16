#!/usr/bin/env python3
"""Search one coupled Skokloster camera/listener for a strict two-adult probe.

This is deliberately CPU-only.  It binds the visual camera and acoustic
listener to the same position, searches the real Habitat navmesh, screens a
conservative 2 m adult cylinder in the 105 degree camera, checks both mouth
line segments against the cleaned research mesh, and then runs the existing
48-direction enclosure diagnostic at the listener and two unchanged mouths.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

WIDTH = 1280
HEIGHT = 720
HFOV_DEG = 105.0
EDGE_MARGIN_PX = 48.0
CAMERA_HEIGHT_M = 1.5
BODY_HEIGHT_M = 2.0
BODY_RADIUS_M = 0.22
MINIMUM_LISTENER_CLEARANCE_M = 0.5
PREFERRED_MINIMUM_DISTANCE_M = 2.2
PREFERRED_MAXIMUM_DISTANCE_M = 3.5
SOURCE1_MAXIMUM_MOUTH_X_FRACTION = 0.42
SOURCE2_MINIMUM_MOUTH_X_FRACTION = 0.58
LOS_TARGET_TOLERANCE_M = 0.03
YAW_OFFSETS_DEG = (-4.0, -2.0, 0.0, 2.0, 4.0)

SOURCE_ROOTS = (
    (2.4000000953674316, 0.13626886904239655, 10.399999618530273),
    (2.700000047683716, 0.11579117923974991, 8.899999618530273),
)
SOURCE_MOUTHS = (
    (2.4000000953674316, 1.7462688690423965, 10.399999618530273),
    (2.700000047683716, 1.684803630816874, 8.899999618530273),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json_no_clobber(path: Path, value: object) -> None:
    _require(
        not path.exists() and not path.is_symlink(), f"refusing to replace: {path}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _forward_right(yaw_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    yaw = math.radians(yaw_deg)
    forward = (-math.sin(yaw), -math.cos(yaw))
    return forward, (-forward[1], forward[0])


def yaw_toward(camera: Sequence[float], target_xz: Sequence[float]) -> float:
    dx = float(target_xz[0]) - float(camera[0])
    dz = float(target_xz[1]) - float(camera[2])
    _require(math.hypot(dx, dz) > 1.0e-9, "camera cannot coincide with target")
    return math.degrees(math.atan2(-dx, -dz))


def project_point(
    camera: Sequence[float], yaw_deg: float, point: Sequence[float]
) -> dict[str, float]:
    forward, right = _forward_right(yaw_deg)
    dx = float(point[0]) - float(camera[0])
    dz = float(point[2]) - float(camera[2])
    depth = dx * forward[0] + dz * forward[1]
    lateral = dx * right[0] + dz * right[1]
    if depth <= 1.0e-6:
        return {"depth_m": depth, "x_px": math.nan, "y_px": math.nan}
    tan_horizontal = math.tan(math.radians(HFOV_DEG) / 2.0)
    tan_vertical = tan_horizontal * HEIGHT / WIDTH
    x_px = (WIDTH - 1.0) / 2.0 + WIDTH * lateral / (2.0 * depth * tan_horizontal)
    y_px = (HEIGHT - 1.0) / 2.0 - HEIGHT * (float(point[1]) - float(camera[1])) / (
        2.0 * depth * tan_vertical
    )
    return {"depth_m": depth, "x_px": x_px, "y_px": y_px}


def adult_envelope_points(root: Sequence[float]) -> list[list[float]]:
    points: list[list[float]] = []
    # Perspective extrema along each vertical cylinder generator occur at an
    # endpoint, so bottom/top perimeter rings bound the complete 0-2 m span.
    for height in (0.0, BODY_HEIGHT_M):
        for azimuth in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            points.append(
                [
                    float(root[0]) + BODY_RADIUS_M * math.cos(float(azimuth)),
                    float(root[1]) + float(height),
                    float(root[2]) + BODY_RADIUS_M * math.sin(float(azimuth)),
                ]
            )
    return points


def screen_projection(camera: Sequence[float], yaw_deg: float) -> dict[str, Any] | None:
    mouth_projections = [
        project_point(camera, yaw_deg, mouth) for mouth in SOURCE_MOUTHS
    ]
    if any(not math.isfinite(item["x_px"]) for item in mouth_projections):
        return None
    if mouth_projections[0]["x_px"] > SOURCE1_MAXIMUM_MOUTH_X_FRACTION * WIDTH:
        return None
    if mouth_projections[1]["x_px"] < SOURCE2_MINIMUM_MOUTH_X_FRACTION * WIDTH:
        return None
    if any(
        not (
            EDGE_MARGIN_PX <= item["x_px"] <= WIDTH - 1.0 - EDGE_MARGIN_PX
            and EDGE_MARGIN_PX <= item["y_px"] <= HEIGHT - 1.0 - EDGE_MARGIN_PX
        )
        for item in mouth_projections
    ):
        return None

    envelope_reports: list[dict[str, Any]] = []
    minimum_edge_margin = math.inf
    for source_id, root in zip(("source1", "source2"), SOURCE_ROOTS, strict=True):
        projections = [
            project_point(camera, yaw_deg, point)
            for point in adult_envelope_points(root)
        ]
        if any(item["depth_m"] <= 0.0 for item in projections):
            return None
        x_values = [item["x_px"] for item in projections]
        y_values = [item["y_px"] for item in projections]
        bbox = [min(x_values), min(y_values), max(x_values), max(y_values)]
        edge_margins = [
            bbox[0],
            bbox[1],
            WIDTH - 1.0 - bbox[2],
            HEIGHT - 1.0 - bbox[3],
        ]
        source_minimum = min(edge_margins)
        if source_minimum < EDGE_MARGIN_PX:
            return None
        minimum_edge_margin = min(minimum_edge_margin, source_minimum)
        envelope_reports.append(
            {
                "source_id": source_id,
                "body_radius_m": BODY_RADIUS_M,
                "body_height_m": BODY_HEIGHT_M,
                "sample_count": len(projections),
                "pixel_bbox_xyxy": bbox,
                "minimum_edge_margin_px": source_minimum,
                "minimum_depth_m": min(item["depth_m"] for item in projections),
                "maximum_depth_m": max(item["depth_m"] for item in projections),
            }
        )
    return {
        "mouth_projections": mouth_projections,
        "mouth_pixel_separation": (
            mouth_projections[1]["x_px"] - mouth_projections[0]["x_px"]
        ),
        "adult_envelopes": envelope_reports,
        "minimum_envelope_edge_margin_px": minimum_edge_margin,
    }


def _load_package_arrays(manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = _load_json(manifest_path)
    root = manifest_path.parent
    vertices = np.load(
        root / manifest["arrays"]["vertices"]["path"], allow_pickle=False
    )
    triangles = np.load(
        root / manifest["arrays"]["triangles"]["path"], allow_pickle=False
    )
    _require(vertices.ndim == 2 and vertices.shape[1] == 3, "vertex array drift")
    _require(triangles.ndim == 2 and triangles.shape[1] == 3, "triangle array drift")
    return vertices, triangles


def _line_of_sight_reports(
    *,
    camera: Sequence[float],
    vertices: np.ndarray,
    triangles: np.ndarray,
    trace_first_hit: Any,
) -> tuple[bool, list[dict[str, Any]]]:
    origin = np.asarray(camera, dtype=np.float64)
    reports: list[dict[str, Any]] = []
    passed = True
    for source_id, mouth in zip(("source1", "source2"), SOURCE_MOUTHS, strict=True):
        target = np.asarray(mouth, dtype=np.float64)
        delta = target - origin
        distance = float(np.linalg.norm(delta))
        direction = delta / distance
        maximum_distance = max(distance - LOS_TARGET_TOLERANCE_M, 1.0e-6)
        hit, hit_distance, triangle_index = trace_first_hit(
            vertices, triangles, origin, direction, maximum_distance
        )
        clear = not hit
        passed &= clear
        reports.append(
            {
                "source_id": source_id,
                "camera_to_mouth_distance_m": distance,
                "tested_clear_distance_m": maximum_distance,
                "target_tolerance_m": LOS_TARGET_TOLERANCE_M,
                "clear": clear,
                "nearest_blocking_hit_m": hit_distance,
                "blocking_triangle_index": triangle_index,
            }
        )
    return passed, reports


def _nav_candidates(pathfinder: Any, *, step_m: float) -> list[dict[str, Any]]:
    midpoint_x = sum(root[0] for root in SOURCE_ROOTS) / 2.0
    midpoint_z = sum(root[2] for root in SOURCE_ROOTS) / 2.0
    requested_y = sum(root[1] for root in SOURCE_ROOTS) / 2.0
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    search_radius = PREFERRED_MAXIMUM_DISTANCE_M + 0.8
    for x in np.arange(
        midpoint_x - search_radius,
        midpoint_x + search_radius + step_m / 2.0,
        step_m,
    ):
        for z in np.arange(
            midpoint_z - search_radius,
            midpoint_z + search_radius + step_m / 2.0,
            step_m,
        ):
            distances = [
                math.hypot(float(x) - root[0], float(z) - root[2])
                for root in SOURCE_ROOTS
            ]
            if not all(
                PREFERRED_MINIMUM_DISTANCE_M <= distance <= PREFERRED_MAXIMUM_DISTANCE_M
                for distance in distances
            ):
                continue
            requested = np.asarray([x, requested_y, z], dtype=np.float32)
            snapped = np.asarray(pathfinder.snap_point(requested), dtype=np.float64)
            if not np.isfinite(snapped).all():
                continue
            horizontal_snap_error = math.hypot(
                float(snapped[0] - requested[0]), float(snapped[2] - requested[2])
            )
            vertical_snap_error = abs(float(snapped[1] - requested[1]))
            if horizontal_snap_error > step_m * 0.51 or vertical_snap_error > 0.30:
                continue
            if not bool(pathfinder.is_navigable(snapped)):
                continue
            island = int(pathfinder.get_island(snapped))
            if island != 0:
                continue
            clearance = float(pathfinder.distance_to_closest_obstacle(snapped, 10.0))
            if clearance < MINIMUM_LISTENER_CLEARANCE_M:
                continue
            key = tuple(round(float(value), 5) for value in snapped)
            if key in seen:
                continue
            seen.add(key)
            camera = [
                float(snapped[0]),
                float(snapped[1]) + CAMERA_HEIGHT_M,
                float(snapped[2]),
            ]
            base_yaw = yaw_toward(camera, (midpoint_x, midpoint_z))
            best_projection: dict[str, Any] | None = None
            best_yaw: float | None = None
            for yaw_offset in YAW_OFFSETS_DEG:
                yaw = base_yaw + yaw_offset
                projection = screen_projection(camera, yaw)
                if projection is None:
                    continue
                if (
                    best_projection is None
                    or projection["minimum_envelope_edge_margin_px"]
                    > best_projection["minimum_envelope_edge_margin_px"]
                ):
                    best_projection = projection
                    best_yaw = yaw
            if best_projection is None or best_yaw is None:
                continue
            distances = [
                math.hypot(float(snapped[0]) - root[0], float(snapped[2]) - root[2])
                for root in SOURCE_ROOTS
            ]
            score = (
                4.0 * clearance
                + 0.01 * best_projection["minimum_envelope_edge_margin_px"]
                - sum(abs(distance - 2.85) for distance in distances)
                - 0.4 * abs(distances[0] - distances[1])
            )
            candidates.append(
                {
                    "score": score,
                    "floor_habitat_m": [float(value) for value in snapped],
                    "camera_listener_habitat_m": camera,
                    "camera_habitat_yaw_deg": best_yaw,
                    "listener_orientation_wxyz": [
                        math.cos(math.radians(best_yaw) / 2.0),
                        0.0,
                        math.sin(math.radians(best_yaw) / 2.0),
                        0.0,
                    ],
                    "nav_island": island,
                    "nav_clearance_m": clearance,
                    "horizontal_source_distances_m": distances,
                    "projection": best_projection,
                    "horizontal_snap_error_m": horizontal_snap_error,
                    "vertical_snap_error_m": vertical_snap_error,
                }
            )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def search(
    *,
    repository: Path,
    navmesh: Path,
    package_manifest: Path,
    step_m: float,
    los_candidate_limit: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(repository / "src"))
    import habitat_sim

    from avengine.acoustics.qa import _trace_first_hit, automatic_mesh_leakage_report

    pathfinder = habitat_sim.PathFinder()
    _require(pathfinder.load_nav_mesh(str(navmesh)), f"cannot load navmesh: {navmesh}")
    _require(pathfinder.num_islands == 1, "Skokloster navmesh island count drift")
    geometric_candidates = _nav_candidates(pathfinder, step_m=step_m)
    _require(geometric_candidates, "no nav/projection candidate passed")

    vertices, triangles = _load_package_arrays(package_manifest)
    selected: dict[str, Any] | None = None
    los_attempt_count = 0
    for candidate in geometric_candidates[:los_candidate_limit]:
        los_attempt_count += 1
        passed, reports = _line_of_sight_reports(
            camera=candidate["camera_listener_habitat_m"],
            vertices=vertices,
            triangles=triangles,
            trace_first_hit=_trace_first_hit,
        )
        candidate["camera_to_mouth_line_of_sight"] = reports
        if passed:
            selected = candidate
            break
    _require(selected is not None, "no tested candidate passed both mouth LOS gates")

    enclosure = automatic_mesh_leakage_report(
        vertices,
        triangles,
        origins=[selected["camera_listener_habitat_m"], *SOURCE_MOUTHS],
        direction_count=48,
        minimum_probe_clearance_m=0.05,
    )
    enclosure_passed = bool(
        enclosure["ray_count"] == 144
        and enclosure["hit_ray_count"] == 144
        and enclosure["escaped_ray_count"] == 0
        and enclosure["probe_clearance_status"] == "pass"
    )
    _require(enclosure_passed, "selected candidate failed the 144-ray enclosure gate")
    selected["enclosure_144"] = enclosure
    selected["coupled_camera_listener"] = True

    source_separation = math.hypot(
        SOURCE_ROOTS[0][0] - SOURCE_ROOTS[1][0],
        SOURCE_ROOTS[0][2] - SOURCE_ROOTS[1][2],
    )
    _require(source_separation >= 1.3, "fixed source separation drift")
    return {
        "schema": "avengine_skokloster_strict_listener_search_v1",
        "status": "pass_cpu_preflight",
        "room_id": "habitat_test_skokloster_castle",
        "coupled_camera_listener_required": True,
        "coordinate_contract": {
            "source_to_habitat": "H=(S.x,S.z,-S.y)",
            "habitat_to_ue_cm": "U_cm=(100*H.x,100*H.z,100*H.y)",
        },
        "inputs": {
            "navmesh": str(navmesh),
            "acoustic_package_manifest": str(package_manifest),
        },
        "requirements": {
            "source_root_separation_m_minimum": 1.3,
            "source_root_separation_m_observed": source_separation,
            "preferred_camera_source_distance_m": [
                PREFERRED_MINIMUM_DISTANCE_M,
                PREFERRED_MAXIMUM_DISTANCE_M,
            ],
            "listener_nav_clearance_m_minimum": MINIMUM_LISTENER_CLEARANCE_M,
            "adult_envelope_height_m": BODY_HEIGHT_M,
            "adult_envelope_radius_m": BODY_RADIUS_M,
            "adult_envelope_edge_margin_px_minimum": EDGE_MARGIN_PX,
            "source1_mouth_x_fraction_maximum": (SOURCE1_MAXIMUM_MOUTH_X_FRACTION),
            "source2_mouth_x_fraction_minimum": (SOURCE2_MINIMUM_MOUTH_X_FRACTION),
            "camera_to_both_mouths_unobstructed": True,
            "enclosure_hit_rays": 144,
        },
        "grid_step_m": step_m,
        "geometric_candidate_count": len(geometric_candidates),
        "los_candidate_limit": los_candidate_limit,
        "los_attempt_count": los_attempt_count,
        "selected": selected,
        "top_geometric_candidates": geometric_candidates[:10],
        "semantics": (
            "CPU nav/projection/room-mesh evidence only; human skeletal bounds and "
            "native SPEAR visibility remain pending."
        ),
        "gpu_capture_authorized": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--navmesh", required=True, type=Path)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step-m", type=float, default=0.05)
    parser.add_argument("--los-candidate-limit", type=int, default=20)
    args = parser.parse_args(argv)
    if args.step_m <= 0.0:
        parser.error("--step-m must be positive")
    if args.los_candidate_limit < 1:
        parser.error("--los-candidate-limit must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = search(
        repository=args.repository.resolve(),
        navmesh=args.navmesh.resolve(),
        package_manifest=args.package_manifest.resolve(),
        step_m=args.step_m,
        los_candidate_limit=args.los_candidate_limit,
    )
    _write_json_no_clobber(args.output.resolve(), result)
    selected = result["selected"]
    print(
        "SKOKLOSTER_STRICT_LISTENER_SEARCH_OK "
        f"candidates={result['geometric_candidate_count']} "
        f"clearance={selected['nav_clearance_m']:.6f} "
        f"edge_margin={selected['projection']['minimum_envelope_edge_margin_px']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

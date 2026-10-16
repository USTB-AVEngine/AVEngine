"""Measure the resting or mounting pose of a published rigid static asset.

Floor-standing assets are measured from the faces that meet the floor.  Wall
and ceiling assets are measured from their mounting plane instead.  This
distinction matters: averaging floor-facing triangles on a wall-mounted object
can reject a good back plate, while averaging a curved wall-connected pipe can
invent a plausible angle for a plane that does not exist.

The floor path deliberately keeps its two existing safeguards.  It selects a
base face by its lowest vertex rather than its centroid, and it ignores winding
before orienting selected normals downward.  Those details prevent tilted or
inverted meshes from becoming silent ``no base`` results.

The grading bands are deliberately loose.  Reconstructed 3D is noisy and a
placement that is a degree or two off, or floating a few millimetres, looks and
behaves correctly.  Only a lean somebody would notice is worth failing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_static_upright_correction import load_glb


LEVEL_DEG = 3.0
ACCEPTABLE_DEG = 8.0

BASE_SLICE_FRACTION = 0.05
MOUNTING_SLICE_FRACTION = 0.05
VERTICAL_COSINE = 0.5
CONTACT_BAND_M = 0.002

# A mounting angle is only reported when the outer slice contains a real plane.
# The concrete failure is the wall-connected bottle trap: it has a small curved
# pipe end but no back plate.  Ordinary type validation can only say ``wall``;
# it cannot distinguish that connection from a flush wall mount at runtime.
# These three measurements reject that small tangent patch while retaining the
# smallest reviewed mounting bracket in the current 40-asset collection.
MOUNTING_MIN_AREA_SHARE = 0.003
MOUNTING_MIN_PROJECTED_BBOX_SHARE = 0.025
MOUNTING_MAX_PLANE_RMS_FRACTION = 0.02

SECONDARY_NOTE = (
    "long_axis_elevation_deg is reported for continuity only. It cannot see a "
    "backward lean, because for a flat panel or a bar the long axis is the one "
    "the lean rotates about. Read resting_pose_verdict instead"
)


def _verdict(tilt: float) -> str:
    if tilt <= LEVEL_DEG:
        return "level"
    if tilt <= ACCEPTABLE_DEG:
        return "acceptable"
    return "leaning"


def _common_result(
    glb: Path,
    vertices: np.ndarray,
    attachment_surface: str | None,
) -> dict[str, Any]:
    assumed = attachment_surface is None
    effective_surface = attachment_surface or "floor"
    if effective_surface not in {"floor", "wall", "ceiling"}:
        raise ValueError(
            "placement.attachment_surface must be floor, wall, ceiling, or absent"
        )
    y = vertices[:, 1]
    result: dict[str, Any] = {
        "measured_from": glb.name,
        "height_m": round(float(y.max() - y.min()), 4),
        "attachment_surface": effective_surface,
        "attachment_surface_assumed": assumed,
        "attachment_surface_note": (
            "placement.attachment_surface was not declared; measured under the "
            "floor assumption"
            if assumed
            else "measured from the declared placement.attachment_surface"
        ),
        "tolerance": {
            "level_deg": LEVEL_DEG,
            "acceptable_deg": ACCEPTABLE_DEG,
            "note": (
                "loose on purpose; reconstructed geometry is noisy and a degree "
                "or two of lean, or a few millimetres of float, is fine"
            ),
        },
    }
    return result


def _face_geometry(vertices: np.ndarray, faces: np.ndarray):
    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    areas = lengths / 2.0
    unit = normals / (lengths[:, None] + 1e-12)
    return corners, areas, unit


def _extent(points: np.ndarray, dimensions: tuple[int, int]) -> list[float]:
    return [
        round(float(points[:, dimension].max() - points[:, dimension].min()), 4)
        for dimension in dimensions
    ]


def _measure_floor(
    vertices: np.ndarray,
    corners: np.ndarray,
    areas: np.ndarray,
    unit: np.ndarray,
    result: dict[str, Any],
) -> dict[str, Any]:
    """The calibrated floor path; keep its selection and winding logic intact."""

    y = vertices[:, 1]
    height = float(y.max() - y.min())
    lowest_y = corners[:, :, 1].min(axis=1)
    base = (np.abs(unit[:, 1]) >= VERTICAL_COSINE) & (
        lowest_y < y.min() + BASE_SLICE_FRACTION * height
    )
    downward = np.where(unit[:, 1] > 0.0, -1.0, 1.0)[:, None] * unit

    result["measured_plane"] = "floor_base"
    result["base_plane_offset_m"] = round(float(y.min()), 4)
    result["how_to_place"] = (
        "put the asset origin at floor height and apply yaw only; the base "
        "offset above is already zero for assets the finalizer grounded, so "
        "no room needs to re-derive this"
    )

    if not base.any():
        result["base_normal_tilt_deg"] = None
        result["mounting_plane_normal_tilt_deg"] = None
        result["verdict"] = "no_base_found"
        result["verdict_reason"] = (
            "no face in the bottom "
            f"{BASE_SLICE_FRACTION:.0%} of the height is within 60 degrees of "
            "horizontal, so there is nothing that would rest on a floor"
        )
        return result

    weights = areas[base]
    mean_normal = (downward[base] * weights[:, None]).sum(axis=0)
    mean_normal /= np.linalg.norm(mean_normal)
    tilt = math.degrees(math.acos(min(1.0, abs(float(mean_normal[1])))))

    contact = vertices[y < y.min() + CONTACT_BAND_M]
    result["base_normal_tilt_deg"] = round(tilt, 2)
    result["mounting_plane_normal_tilt_deg"] = None
    result["base_area_share"] = round(float(weights.sum() / areas.sum()), 4)
    # Contact size is diagnostic only.  A domed base can be level with a small
    # contact patch, so this must never become an acceptance threshold.
    result["contact_extent_m"] = (
        _extent(contact, (0, 2)) if len(contact) > 2 else [0.0, 0.0]
    )
    result["footprint_extent_m"] = _extent(vertices, (0, 2))
    result["verdict"] = _verdict(tilt)
    return result


def _wall_back_axis(front_axis: str) -> tuple[int, float]:
    mapping = {
        "positive-x": (0, -1.0),
        "negative-x": (0, 1.0),
        "positive-z": (2, -1.0),
        "negative-z": (2, 1.0),
    }
    try:
        return mapping[front_axis]
    except KeyError as error:
        raise ValueError(
            "a wall mounting plane needs front_axis positive-x, negative-x, "
            "positive-z, or negative-z"
        ) from error


def _weighted_plane(
    centroids: np.ndarray, weights: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float]:
    center = (centroids * weights[:, None]).sum(axis=0) / weights.sum()
    delta = centroids - center
    covariance = (delta * weights[:, None]).T @ delta / weights.sum()
    values, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, int(np.argmin(values))]
    if float(normal @ target) < 0.0:
        normal = -normal
    return normal, math.sqrt(max(0.0, float(values.min())))


def _measure_mounting_plane(
    vertices: np.ndarray,
    corners: np.ndarray,
    areas: np.ndarray,
    unit: np.ndarray,
    result: dict[str, Any],
    surface: str,
    front_axis: str,
) -> dict[str, Any]:
    if surface == "wall":
        axis, direction = _wall_back_axis(front_axis)
        target = np.zeros(3)
        target[axis] = direction
        span = float(vertices[:, axis].max() - vertices[:, axis].min())
        if direction < 0.0:
            outer = corners[:, :, axis].min(axis=1)
            in_slice = outer < (
                vertices[:, axis].min() + MOUNTING_SLICE_FRACTION * span
            )
        else:
            outer = corners[:, :, axis].max(axis=1)
            in_slice = outer > (
                vertices[:, axis].max() - MOUNTING_SLICE_FRACTION * span
            )
        plane_name = "wall_back"
        projected_dimensions = tuple(index for index in range(3) if index != axis)
        result["how_to_place"] = (
            "place the measured back toward the wall and apply translation and "
            "yaw only; wall-connected geometry without a back plane is reported "
            "as no_mounting_plane_found"
        )
    else:
        axis = 1
        target = np.array([0.0, 1.0, 0.0])
        span = float(vertices[:, 1].max() - vertices[:, 1].min())
        outer = corners[:, :, 1].max(axis=1)
        in_slice = outer > (
            vertices[:, 1].max() - MOUNTING_SLICE_FRACTION * span
        )
        plane_name = "ceiling_top"
        projected_dimensions = (0, 2)
        result["how_to_place"] = (
            "place the measured top against the ceiling and apply translation "
            "and yaw only"
        )

    candidates = (np.abs(unit[:, axis]) >= VERTICAL_COSINE) & in_slice
    result["measured_plane"] = plane_name
    result["base_normal_tilt_deg"] = None
    result["mounting_plane_detection"] = {
        "slice_fraction": MOUNTING_SLICE_FRACTION,
        "minimum_area_share": MOUNTING_MIN_AREA_SHARE,
        "minimum_projected_bbox_share": MOUNTING_MIN_PROJECTED_BBOX_SHARE,
        "maximum_plane_rms_fraction": MOUNTING_MAX_PLANE_RMS_FRACTION,
        "note": (
            "these are existence checks for a recoverable mounting plane, not "
            "resting-pose strictness or a contact-area acceptance gate"
        ),
    }

    if not candidates.any():
        result["mounting_plane_normal_tilt_deg"] = None
        result["mounting_plane_area_share"] = 0.0
        result["mounting_plane_projected_bbox_share"] = 0.0
        result["mounting_plane_rms_m"] = None
        result["mounting_plane_rms_fraction"] = None
        result["mounting_plane_candidate_faces"] = 0
        result["verdict"] = "no_mounting_plane_found"
        result["verdict_reason"] = "the outer slice has no face oriented like a mounting plane"
        return result

    selected_corners = corners[candidates]
    selected_areas = areas[candidates]
    # Two triangles are already enough to describe a rectangular back plate,
    # but their two centroids are only a line and make a plane fit degenerate.
    # Use all triangle vertices while splitting each face's area across them.
    plane_points = selected_corners.reshape(-1, 3)
    plane_weights = np.repeat(selected_areas / 3.0, 3)
    normal, rms = _weighted_plane(plane_points, plane_weights, target)

    area_share = float(selected_areas.sum() / areas.sum())
    selected_points = plane_points
    selected_extent = np.ptp(selected_points[:, projected_dimensions], axis=0)
    full_extent = np.ptp(vertices[:, projected_dimensions], axis=0)
    denominator = float(np.prod(full_extent))
    projected_share = (
        0.0 if denominator <= 1e-12 else float(np.prod(selected_extent) / denominator)
    )
    reference_extent = float(max(full_extent))
    rms_fraction = float("inf") if reference_extent <= 1e-12 else rms / reference_extent

    result["mounting_plane_area_share"] = round(area_share, 6)
    result["mounting_plane_projected_bbox_share"] = round(projected_share, 6)
    result["mounting_plane_rms_m"] = round(rms, 6)
    result["mounting_plane_rms_fraction"] = round(rms_fraction, 6)
    result["mounting_plane_candidate_faces"] = int(candidates.sum())

    failures = []
    if area_share < MOUNTING_MIN_AREA_SHARE:
        failures.append("candidate area share is too small")
    if projected_share < MOUNTING_MIN_PROJECTED_BBOX_SHARE:
        failures.append("candidate projected coverage is too small")
    if rms_fraction > MOUNTING_MAX_PLANE_RMS_FRACTION:
        failures.append("candidate surface is not planar enough")
    if failures:
        result["mounting_plane_normal_tilt_deg"] = None
        result["verdict"] = "no_mounting_plane_found"
        result["verdict_reason"] = "; ".join(failures)
        return result

    if surface == "wall":
        # A wall normal may rotate in yaw while remaining level.  Only its
        # vertical component is lean.
        tilt = math.degrees(math.asin(min(1.0, abs(float(normal[1])))))
    else:
        tilt = math.degrees(math.acos(min(1.0, abs(float(normal[1])))))
    result["mounting_plane_normal_tilt_deg"] = round(tilt, 2)
    result["verdict"] = _verdict(tilt)
    return result


def measure(
    glb: Path,
    attachment_surface: str | None = None,
    front_axis: str = "positive-x",
) -> dict[str, Any]:
    vertices, faces = load_glb(glb)
    corners, areas, unit = _face_geometry(vertices, faces)
    result = _common_result(glb, vertices, attachment_surface)
    surface = result["attachment_surface"]
    if surface == "floor":
        return _measure_floor(vertices, corners, areas, unit, result)
    return _measure_mounting_plane(
        vertices, corners, areas, unit, result, surface, front_axis
    )


def acceptance_fields(
    pose: dict[str, Any], long_axis_elevation_deg: float | None
) -> dict[str, Any]:
    """Return the identical resting-pose acceptance fields for both publishers."""

    return {
        "resting_pose_verdict": pose["verdict"],
        "base_normal_tilt_deg": pose.get("base_normal_tilt_deg"),
        "mounting_plane_normal_tilt_deg": pose.get(
            "mounting_plane_normal_tilt_deg"
        ),
        "resting_pose_attachment_surface": pose["attachment_surface"],
        "resting_pose_attachment_surface_assumed": pose[
            "attachment_surface_assumed"
        ],
        "secondary_long_axis_elevation_deg": long_axis_elevation_deg,
        "secondary_note": SECONDARY_NOTE,
    }


def _reported_tilt(pose: dict[str, Any]) -> float | None:
    if pose["attachment_surface"] == "floor":
        return pose.get("base_normal_tilt_deg")
    return pose.get("mounting_plane_normal_tilt_deg")


def merge_updated_assets_into_index(
    index: dict[str, Any], updated_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Replace matching embedded index records and preserve unrelated assets."""

    by_id: dict[str, dict[str, Any]] = {}
    for record in updated_records:
        asset_id = record.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("an applied asset.json is missing asset_id")
        if asset_id in by_id:
            raise ValueError(f"applied asset.json repeats asset_id {asset_id}")
        by_id[asset_id] = record

    existing = index.get("assets")
    if not isinstance(existing, list):
        raise ValueError("asset index assets must be a list")
    existing_ids = {item.get("asset_id") for item in existing if isinstance(item, dict)}
    missing = sorted(set(by_id) - existing_ids)
    if missing:
        raise ValueError(f"applied assets are missing from index: {missing}")

    result = dict(index)
    result["assets"] = [
        by_id.get(item.get("asset_id"), item) if isinstance(item, dict) else item
        for item in existing
    ]
    return result


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.resting-pose-{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "write geometry.resting_pose and the complete acceptance summary "
            "into each asset.json; the mesh and index identity are not changed"
        ),
    )
    args = parser.parse_args()

    records = []
    applied_records = []
    for asset_json in sorted(args.asset_root.rglob("asset.json")):
        directory = asset_json.parent
        glb = directory / "finalized.glb"
        if not glb.is_file():
            continue
        record = json.loads(asset_json.read_text(encoding="utf-8"))
        placement = record.get("placement") or {}
        acceptance = record.get("acceptance") or {}
        pose = measure(
            glb,
            attachment_surface=placement.get("attachment_surface"),
            front_axis=acceptance.get("front_axis", "positive-x"),
        )
        relative = directory.relative_to(args.asset_root).as_posix()
        records.append({"asset": relative, **pose})

        marker = {"level": "  ", "acceptable": " ~", "leaning": " !"}.get(
            pose["verdict"], " ?"
        )
        tilt = _reported_tilt(pose)
        assumption = " assumed-floor" if pose["attachment_surface_assumed"] else ""
        print(
            f"{marker} {relative:<56} "
            f"{'  n/a' if tilt is None else f'{tilt:5.2f}'} deg  "
            f"{pose['verdict']} [{pose['attachment_surface']}{assumption}]"
        )

        if args.apply:
            geometry = record.setdefault("geometry", {})
            geometry["resting_pose"] = pose
            record.setdefault("acceptance", {}).update(
                acceptance_fields(pose, geometry.get("long_axis_elevation_deg"))
            )
            _write_json_atomic(asset_json, record)
            applied_records.append(record)

    leaning = [item for item in records if item["verdict"] == "leaning"]
    no_base = [item for item in records if item["verdict"] == "no_base_found"]
    no_mount = [
        item for item in records if item["verdict"] == "no_mounting_plane_found"
    ]
    print(
        f"\n{len(records)} assets: "
        f"{sum(item['verdict'] == 'level' for item in records)} level, "
        f"{sum(item['verdict'] == 'acceptable' for item in records)} acceptable, "
        f"{len(leaning)} leaning, {len(no_base)} with no base, "
        f"{len(no_mount)} with no mounting plane"
    )
    for item in leaning + no_base + no_mount:
        print(
            f"  {item['verdict']}: {item['asset']} "
            f"({_reported_tilt(item)} deg)"
        )

    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "schema": "avengine_static_resting_pose_v2_batch",
                    "level_deg": LEVEL_DEG,
                    "acceptable_deg": ACCEPTABLE_DEG,
                    "assets": records,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.report}")
    if args.apply:
        index_path = args.asset_root / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        merged = merge_updated_assets_into_index(index, applied_records)
        _write_json_atomic(index_path, merged)
        print(
            "applied geometry.resting_pose and complete acceptance fields "
            f"to {len(applied_records)} leaves and index.json"
        )
    return 1 if leaning else 0


if __name__ == "__main__":
    raise SystemExit(main())

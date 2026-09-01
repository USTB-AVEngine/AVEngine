#!/usr/bin/env python3
"""Find same-facing coplanar mesh surfaces that can flicker after USD import.

Two touching solids normally expose opposite-facing polygons at their shared
boundary.  They are not a z-fighting risk.  This audit instead looks for
polygons from different meshes that have the same normal, occupy the same
plane, and overlap in area.  That is the signature produced when a decorative
panel is placed exactly on top of its backing wall.

The report is an ordinary geometry validation artifact.  It does not add a
content hash or freeze the input USD.  Run it in an environment with Pixar USD
Python bindings, for example the AVEngine Habitat runtime environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class SurfaceFace:
    mesh: str
    face_index: int
    material: str | None
    points: tuple[Vec3, ...]
    normal: Vec3
    area_m2: float


def _add(left: Vec3, right: Vec3) -> Vec3:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _subtract(left: Vec3, right: Vec3) -> Vec3:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _scale(vector: Vec3, value: float) -> Vec3:
    return tuple(item * value for item in vector)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _length(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vec3) -> Vec3:
    length = _length(vector)
    if length <= 1.0e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return _scale(vector, 1.0 / length)


def polygon_normal_and_area(points: Sequence[Vec3]) -> tuple[Vec3, float]:
    """Return a stable normal and area for one planar polygon."""

    if len(points) < 3:
        raise ValueError("a polygon needs at least three points")
    total = (0.0, 0.0, 0.0)
    origin = points[0]
    for index in range(1, len(points) - 1):
        first = _subtract(points[index], origin)
        second = _subtract(points[index + 1], origin)
        total = _add(total, _cross(first, second))
    doubled_area = _length(total)
    if doubled_area <= 1.0e-12:
        raise ValueError("polygon is degenerate")
    return _normalize(total), 0.5 * doubled_area


def _signed_area_2d(points: Sequence[Vec2]) -> float:
    return 0.5 * sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(points, points[1:] + points[:1], strict=True)
    )


def _cross_2d(left: Vec2, right: Vec2) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _line_intersection(
    start: Vec2, end: Vec2, clip_start: Vec2, clip_end: Vec2
) -> Vec2:
    subject = (end[0] - start[0], end[1] - start[1])
    clip = (clip_end[0] - clip_start[0], clip_end[1] - clip_start[1])
    denominator = _cross_2d(subject, clip)
    if abs(denominator) <= 1.0e-12:
        return end
    offset = (clip_start[0] - start[0], clip_start[1] - start[1])
    factor = _cross_2d(offset, clip) / denominator
    return (start[0] + factor * subject[0], start[1] + factor * subject[1])


def convex_intersection_area(subject: Sequence[Vec2], clip: Sequence[Vec2]) -> float:
    """Return intersection area for two convex polygons."""

    output = list(subject)
    clip_points = list(clip)
    if len(output) < 3 or len(clip_points) < 3:
        return 0.0
    if _signed_area_2d(output) < 0.0:
        output.reverse()
    if _signed_area_2d(clip_points) < 0.0:
        clip_points.reverse()
    for clip_start, clip_end in zip(
        clip_points, clip_points[1:] + clip_points[:1], strict=True
    ):
        input_points = output
        output = []
        if not input_points:
            break

        def inside(point: Vec2) -> bool:
            edge = (clip_end[0] - clip_start[0], clip_end[1] - clip_start[1])
            relative = (point[0] - clip_start[0], point[1] - clip_start[1])
            return _cross_2d(edge, relative) >= -1.0e-10

        previous = input_points[-1]
        previous_inside = inside(previous)
        for current in input_points:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, clip_start, clip_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, clip_start, clip_end)
                )
            previous = current
            previous_inside = current_inside
    return abs(_signed_area_2d(output)) if len(output) >= 3 else 0.0


def _plane_basis(normal: Vec3) -> tuple[Vec3, Vec3]:
    reference = min(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        key=lambda axis: abs(_dot(normal, axis)),
    )
    first = _normalize(_cross(normal, reference))
    second = _normalize(_cross(normal, first))
    return first, second


def _project(points: Sequence[Vec3], first: Vec3, second: Vec3) -> list[Vec2]:
    return [(_dot(point, first), _dot(point, second)) for point in points]


def detect_same_facing_coplanar_overlaps(
    faces: Iterable[SurfaceFace],
    *,
    up_axis: Vec3 = (0.0, 0.0, 1.0),
    plane_tolerance_m: float = 5.0e-4,
    normal_dot_minimum: float = 0.999,
    minimum_overlap_area_m2: float = 0.01,
    minimum_overlap_ratio: float = 0.20,
    ignore_downward_faces: bool = True,
) -> list[dict[str, object]]:
    """Classify overlapping, co-oriented faces from different meshes."""

    candidates = [
        face
        for face in faces
        if face.area_m2 >= minimum_overlap_area_m2
        and not (
            ignore_downward_faces and _dot(face.normal, up_axis) <= -0.95
        )
    ]
    overlaps: list[dict[str, object]] = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if left.mesh == right.mesh:
                continue
            normal_dot = _dot(left.normal, right.normal)
            if normal_dot < normal_dot_minimum:
                continue
            plane_distance = abs(
                _dot(left.normal, left.points[0])
                - _dot(left.normal, right.points[0])
            )
            if plane_distance > plane_tolerance_m:
                continue
            first, second = _plane_basis(left.normal)
            overlap_area = convex_intersection_area(
                _project(left.points, first, second),
                _project(right.points, first, second),
            )
            overlap_ratio = overlap_area / min(left.area_m2, right.area_m2)
            if (
                overlap_area < minimum_overlap_area_m2
                or overlap_ratio < minimum_overlap_ratio
            ):
                continue
            overlaps.append(
                {
                    "left_mesh": left.mesh,
                    "left_face_index": left.face_index,
                    "left_material": left.material,
                    "right_mesh": right.mesh,
                    "right_face_index": right.face_index,
                    "right_material": right.material,
                    "normal": list(left.normal),
                    "plane_distance_m": plane_distance,
                    "overlap_area_m2": overlap_area,
                    "overlap_ratio": overlap_ratio,
                }
            )
    return sorted(
        overlaps,
        key=lambda item: (
            -float(item["overlap_area_m2"]),
            str(item["left_mesh"]),
            str(item["right_mesh"]),
        ),
    )


def _material_path(prim: object) -> str | None:
    from pxr import UsdShade

    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    return material.GetPath().pathString if material else None


def load_usd_faces(path: Path) -> tuple[list[SurfaceFace], Vec3, str]:
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Pixar USD Python bindings are required") from exc

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open USD stage: {path}")
    up_token = str(UsdGeom.GetStageUpAxis(stage)).upper()
    up_axis = {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }[up_token]
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    faces: list[SurfaceFace] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        world = xforms.GetLocalToWorldTransform(prim)
        transformed = [
            tuple(float(value) for value in world.Transform(Gf.Vec3d(*point)))
            for point in points
        ]
        left_handed = str(mesh.GetOrientationAttr().Get()) == "leftHanded"
        material = _material_path(prim)
        cursor = 0
        for face_index, count in enumerate(counts):
            polygon = [
                transformed[int(index)] for index in indices[cursor : cursor + count]
            ]
            cursor += count
            if left_handed:
                polygon.reverse()
            try:
                normal, area = polygon_normal_and_area(polygon)
            except ValueError:
                continue
            faces.append(
                SurfaceFace(
                    mesh=prim.GetPath().pathString,
                    face_index=face_index,
                    material=material,
                    points=tuple(polygon),
                    normal=normal,
                    area_m2=area,
                )
            )
    return faces, up_axis, up_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plane-tolerance-m", type=float, default=5.0e-4)
    parser.add_argument("--normal-dot-minimum", type=float, default=0.999)
    parser.add_argument("--minimum-overlap-area-m2", type=float, default=0.01)
    parser.add_argument("--minimum-overlap-ratio", type=float, default=0.20)
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="write a failing report without returning a failing process status",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to replace output: {output_path}")
    faces, up_axis, up_token = load_usd_faces(input_path)
    overlaps = detect_same_facing_coplanar_overlaps(
        faces,
        up_axis=up_axis,
        plane_tolerance_m=args.plane_tolerance_m,
        normal_dot_minimum=args.normal_dot_minimum,
        minimum_overlap_area_m2=args.minimum_overlap_area_m2,
        minimum_overlap_ratio=args.minimum_overlap_ratio,
    )
    report = {
        "schema": "avengine_usd_coplanar_surface_audit_v1",
        "status": "fail" if overlaps else "pass",
        "input": str(input_path),
        "stage_up_axis": up_token,
        "mesh_face_count": len(faces),
        "parameters": {
            "plane_tolerance_m": args.plane_tolerance_m,
            "normal_dot_minimum": args.normal_dot_minimum,
            "minimum_overlap_area_m2": args.minimum_overlap_area_m2,
            "minimum_overlap_ratio": args.minimum_overlap_ratio,
            "ignore_downward_faces": True,
        },
        "same_facing_coplanar_overlaps": overlaps,
        "overlap_count": len(overlaps),
        "failure_scenario": (
            "different visible mesh faces compete for the same depth samples and "
            "flicker after temporal rendering"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    if overlaps and not args.allow_overlap:
        raise RuntimeError(
            f"found {len(overlaps)} same-facing coplanar overlap(s); see {output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

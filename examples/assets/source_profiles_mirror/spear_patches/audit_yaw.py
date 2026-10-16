#!/usr/bin/env python3
"""Read the object's yaw from geometry instead of from five renders.

Eyeballing which of front/side/back/quarter shows the front face only
resolves yaw to the nearest 45 degrees, and a reconstruction that sits at an
odd angle reads as a lean in every one of them.  For a box-like object the
plan-view principal axis is exact: the long horizontal axis is the object's
length, the front is perpendicular to it, and the largest flat vertical panel
tells which of the two perpendicular directions the front is.

Angles are Blender-space yaw about Z, which is what the finalizer's
reviewed_source_front_yaw_deg wants: glTF (x, y, z) maps to Blender
(x, -z, y), so Blender yaw = atan2(-z_gltf, x_gltf).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp")
from audit_raw_topology import load_glb  # noqa: E402


def audit(path: Path) -> dict:
    positions, indices = load_glb(path)
    faces = indices.reshape(-1, 3)
    a, b, c = positions[faces[:, 0]], positions[faces[:, 1]], positions[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    keep = area > 0
    normals = cross[keep] / (2.0 * area[keep, None])
    area = area[keep]
    centroid = ((a + b + c) / 3.0)[keep]

    # Plan-view principal axis of the whole shell.
    plan = centroid[:, [0, 2]]
    mean = (plan * area[:, None]).sum(0) / area.sum()
    delta = plan - mean
    covariance = (delta * area[:, None]).T @ delta / area.sum()
    values, vectors = np.linalg.eigh(covariance)
    long_plan = vectors[:, int(np.argmax(values))]
    plan_ratio = float(np.sqrt(values.max() / max(values.min(), 1e-12)))

    # Vertical panels only: their normals say which way the faces point.
    vertical = np.abs(normals[:, 1]) < 0.35
    panel_normals = normals[vertical]
    panel_area = area[vertical]
    step = np.deg2rad(5.0)
    azimuth = np.arctan2(-panel_normals[:, 2], panel_normals[:, 0])
    bucket = ((azimuth + np.pi) / step).astype(np.int64)
    order = np.argsort(bucket)
    bucket, panel_area, azimuth = bucket[order], panel_area[order], azimuth[order]
    edges = np.flatnonzero(np.diff(bucket)) + 1
    best_weight, best_azimuth = 0.0, float("nan")
    for group in np.split(np.arange(len(bucket)), edges):
        weight = panel_area[group].sum()
        if weight > best_weight:
            best_weight = weight
            vector = np.array(
                [
                    (np.cos(azimuth[group]) * panel_area[group]).sum(),
                    (np.sin(azimuth[group]) * panel_area[group]).sum(),
                ]
            )
            best_azimuth = float(np.degrees(np.arctan2(vector[1], vector[0])))

    long_axis_yaw = float(np.degrees(np.arctan2(-long_plan[1], long_plan[0])))
    return {
        "biggest_vertical_panel_yaw_deg": best_azimuth,
        "biggest_vertical_panel_area_share": float(best_weight / area.sum()),
        "plan_long_axis_yaw_deg": long_axis_yaw,
        "plan_elongation": plan_ratio,
    }


def main() -> int:
    for directory in sorted(Path(sys.argv[1]).iterdir()):
        glb = directory / "pixal_raw_1024.glb"
        if not glb.is_file():
            continue
        report = audit(glb)
        name = directory.name.replace("audio_playback_", "").replace("_product_view", "")
        print(
            f"{name:<38} panel_yaw={report['biggest_vertical_panel_yaw_deg']:7.1f}deg "
            f"({report['biggest_vertical_panel_area_share']*100:4.1f}% of area)  "
            f"plan_long_axis={report['plan_long_axis_yaw_deg']:7.1f}deg "
            f"elongation={report['plan_elongation']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

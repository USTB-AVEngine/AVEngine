#!/usr/bin/env python3
"""Principal axes of the main shell, area weighted.

The base-plane fit is unreliable when the object rests on a few small feet.
The inertia axes of the whole shell are not: for an upright cabinet or
cylinder the long axis is vertical, and for a bar the long axis is horizontal.
The angle reported is how far the object is from standing the way it should.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp")
from audit_raw_topology import load_glb  # noqa: E402
from audit_components import labels_for  # noqa: E402


def audit(path: Path) -> dict:
    positions, indices = load_glb(path)
    keys = np.round(positions / 1.0e-5).astype(np.int64)
    _, welded, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    points = positions[welded]
    faces = inverse[indices].reshape(-1, 3)
    faces = faces[
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    ]
    label = labels_for(len(points), faces)
    centroid = points[faces].mean(axis=1)
    cross = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    area = 0.5 * np.linalg.norm(cross, axis=1)
    roots, inverse_root = np.unique(label[faces[:, 0]], return_inverse=True)
    main_root = roots[int(np.argmax(np.bincount(inverse_root, weights=area)))]
    keep = label[faces[:, 0]] == main_root
    centroid, area = centroid[keep], area[keep]

    mean = (centroid * area[:, None]).sum(0) / area.sum()
    delta = centroid - mean
    covariance = (delta * area[:, None]).T @ delta / area.sum()
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    long_axis = vectors[:, order[0]]
    short_axis = vectors[:, order[2]]
    # elevation of the long axis above the horizontal plane (y is up in glTF)
    elevation = float(np.degrees(np.arcsin(min(1.0, abs(long_axis[1])))))
    return {
        "long_axis_elevation_deg": elevation,
        "extent_ratio": [round(float(v), 4) for v in np.sqrt(values[order])],
        "thin_axis_elevation_deg": float(
            np.degrees(np.arcsin(min(1.0, abs(short_axis[1]))))
        ),
    }


def main() -> int:
    for directory in sorted(Path(sys.argv[1]).iterdir()):
        glb = directory / "pixal_raw_1024.glb"
        if not glb.is_file():
            continue
        report = audit(glb)
        name = directory.name.replace("audio_playback_", "").replace("_product_view", "")
        print(
            f"{name:<38} long_axis_elevation={report['long_axis_elevation_deg']:5.1f}deg "
            f"sigma={report['extent_ratio']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

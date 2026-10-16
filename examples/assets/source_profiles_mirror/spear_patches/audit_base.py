#!/usr/bin/env python3
"""Fit the resting plane from the lowest slab of the main component.

The finalizer grounds the object by translating its lowest point to zero and
scales by bounding-box height.  Both assume the object rests on a flat, level
base.  Fit a plane to the lowest few percent of the main shell and report how
far its normal is from straight up, plus how much of the footprint that plane
covers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp")
from audit_raw_topology import load_glb  # noqa: E402
from audit_components import labels_for  # noqa: E402


def audit(path: Path, slab: float = 0.02) -> dict:
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
    cross = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    area = 0.5 * np.linalg.norm(cross, axis=1)
    roots, inverse_root = np.unique(label[faces[:, 0]], return_inverse=True)
    main_root = roots[int(np.argmax(np.bincount(inverse_root, weights=area)))]
    main = points[np.unique(faces[label[faces[:, 0]] == main_root])]

    height = main[:, 1].max() - main[:, 1].min()
    low = main[main[:, 1] <= main[:, 1].min() + slab * height]
    centred = low - low.mean(0)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    normal = right[-1]
    if normal[1] < 0:
        normal = -normal
    tilt = float(np.degrees(np.arccos(np.clip(normal[1], -1.0, 1.0))))

    footprint = (main[:, [0, 2]].max(0) - main[:, [0, 2]].min(0)).prod()
    low_footprint = (low[:, [0, 2]].max(0) - low[:, [0, 2]].min(0)).prod()
    flatness = float(np.abs(centred @ normal).max() / max(height, 1e-9))
    return {
        "base_tilt_deg": tilt,
        "base_flatness_over_height": flatness,
        "base_footprint_share": float(low_footprint / max(footprint, 1e-9)),
        "points_in_slab": int(len(low)),
    }


def main() -> int:
    for directory in sorted(Path(sys.argv[1]).iterdir()):
        glb = directory / "pixal_raw_1024.glb"
        if not glb.is_file():
            continue
        report = audit(glb)
        name = directory.name.replace("audio_playback_", "").replace("_product_view", "")
        print(
            f"{name:<38} base_tilt={report['base_tilt_deg']:5.1f}deg  "
            f"flatness={report['base_flatness_over_height']:.4f}  "
            f"footprint_share={report['base_footprint_share']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

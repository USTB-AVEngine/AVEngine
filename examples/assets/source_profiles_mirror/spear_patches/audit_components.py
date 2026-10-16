#!/usr/bin/env python3
"""Area share and bounding box of the largest connected component.

Component count alone says nothing: a mesh can be one clean shell plus 700
specks.  What matters is how much area sits outside the main shell, and
whether the debris inflates the bounding box - the finalizer scales the whole
object so that its bounding-box height hits the profile target, so a speck
floating above the object silently shrinks the object itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp")
from audit_raw_topology import load_glb  # noqa: E402


def labels_for(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    parent = np.arange(vertex_count)

    def find(index):
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    for a, b, c in faces:
        for left, right in ((a, b), (b, c)):
            ra, rb = find(left), find(right)
            if ra != rb:
                parent[ra] = rb
    return np.array([find(i) for i in range(vertex_count)])


def audit(path: Path, quantum: float = 1.0e-5) -> dict:
    positions, indices = load_glb(path)
    keys = np.round(positions / quantum).astype(np.int64)
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
    face_label = label[faces[:, 0]]
    roots, inverse_root = np.unique(face_label, return_inverse=True)
    area_by_root = np.bincount(inverse_root, weights=area)
    main = roots[int(np.argmax(area_by_root))]

    keep = face_label == main
    main_points = points[np.unique(faces[keep])]
    full_extent = points.max(0) - points.min(0)
    main_extent = main_points.max(0) - main_points.min(0)
    return {
        "components": len(roots),
        "main_area_share": float(area_by_root.max() / area_by_root.sum()),
        "debris_area_share": float(1.0 - area_by_root.max() / area_by_root.sum()),
        "full_extent": full_extent,
        "main_extent": main_extent,
        "height_inflation": float(full_extent[1] / max(main_extent[1], 1e-9)),
    }


def main() -> int:
    for directory in sorted(Path(sys.argv[1]).iterdir()):
        glb = directory / "pixal_raw_1024.glb"
        if not glb.is_file():
            continue
        report = audit(glb)
        name = directory.name.replace("audio_playback_", "").replace("_product_view", "")
        main_extent = report["main_extent"]
        print(
            f"{name:<38} comp={report['components']:>4} "
            f"debris_area={report['debris_area_share']*100:6.3f}%  "
            f"main_extent=[{main_extent[0]:.3f} {main_extent[1]:.3f} {main_extent[2]:.3f}]  "
            f"w/h={main_extent[0]/main_extent[1]:5.2f} d/h={main_extent[2]/main_extent[1]:5.2f}  "
            f"height_inflation={report['height_inflation']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

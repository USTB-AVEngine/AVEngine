#!/usr/bin/env python3
"""Emit the exact face inventory for a Skokloster research cleanup."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

RLR_CROSS_NORM_SQUARED_THRESHOLD = 1.0e-20
EXPECTED_QA_FACES = [251199, 288544]


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", required=True, type=Path)
    parser.add_argument("--derived-package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_root = args.source_package.resolve()
    derived_root = args.derived_package.resolve()
    source_manifest = _load(source_root / "manifest.json")
    derived_manifest = _load(derived_root / "manifest.json")
    parity_record = derived_manifest["qa"]["compiler_source_to_package_parity"]
    parity = _load(derived_root / parity_record["path"])
    cleanup = parity["research_cleanup"]
    removed = [int(value) for value in cleanup["removed_triangle_indices"]]

    vertices = np.load(
        source_root / source_manifest["arrays"]["vertices"]["path"],
        allow_pickle=False,
    )
    triangles = np.load(
        source_root / source_manifest["arrays"]["triangles"]["path"],
        allow_pickle=False,
    )
    material_ids = np.load(
        source_root / source_manifest["arrays"]["triangle_material_ids"]["path"],
        allow_pickle=False,
    )
    points_f32 = vertices[triangles[np.asarray(removed, dtype=np.int64)]]
    ab = (points_f32[:, 1] - points_f32[:, 0]).astype(np.float64)
    ac = (points_f32[:, 2] - points_f32[:, 0]).astype(np.float64)
    cross = np.cross(ab, ac)
    cross_squared = np.einsum("ij,ij->i", cross, cross)
    areas = 0.5 * np.sqrt(cross_squared)
    qa_threshold = float(cleanup["qa_area_threshold_m2_inclusive"])

    records = []
    for ordinal, triangle_index in enumerate(removed):
        records.append(
            {
                "triangle_index": triangle_index,
                "vertex_indices": triangles[triangle_index].astype(int).tolist(),
                "material_id": int(material_ids[triangle_index]),
                "area_m2": float(areas[ordinal]),
                "cross_norm_squared_m4": float(cross_squared[ordinal]),
                "qa_degenerate": bool(areas[ordinal] <= qa_threshold),
                "native_rlr_incompatible": bool(
                    cross_squared[ordinal] <= RLR_CROSS_NORM_SQUARED_THRESHOLD
                ),
            }
        )

    qa_faces = [item["triangle_index"] for item in records if item["qa_degenerate"]]
    _require(len(records) == 48, "native RLR cleanup must remove exactly 48 faces")
    _require(qa_faces == EXPECTED_QA_FACES, "geometry-QA face set drift")
    _require(
        all(item["native_rlr_incompatible"] for item in records),
        "cleanup inventory includes a native-RLR-compatible face",
    )
    _require(
        all(math.isfinite(item["area_m2"]) for item in records),
        "non-finite face area",
    )
    result = {
        "schema": "avengine_skokloster_native_rlr_cleanup_inventory_v1",
        "status": "pass",
        "policy": cleanup["policy"],
        "source_triangle_count": len(triangles),
        "derived_triangle_count": int(derived_manifest["geometry"]["triangle_count"]),
        "removed_triangle_count": len(records),
        "geometry_qa_removed_triangle_count": len(qa_faces),
        "native_rlr_only_removed_triangle_count": len(records) - len(qa_faces),
        "qa_area_threshold_m2_inclusive": qa_threshold,
        "native_rlr_cross_norm_squared_threshold_inclusive": (
            RLR_CROSS_NORM_SQUARED_THRESHOLD
        ),
        "faces": records,
        "topology_policy": {
            "hole_filling": False,
            "vertex_repositioning": False,
            "vertex_removal": False,
            "other_face_removal": False,
        },
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    _require(not args.output.exists(), f"refusing to replace output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "SKOKLOSTER_CLEANUP_INVENTORY_OK "
        f"removed={len(records)} qa={len(qa_faces)} rlr_only={len(records) - len(qa_faces)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

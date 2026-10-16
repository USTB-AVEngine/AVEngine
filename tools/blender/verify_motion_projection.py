#!/usr/bin/env python3
"""Measure sampled deformation drift from a source to a rotation-only GLB.

Unlike the strict rebase verifier, this command is allowed to produce a
``fail`` report.  It never calls a projection equivalent merely because the
projected GLB is structurally valid: equivalence requires the maximum sampled
world-space vertex error to stay below the explicit threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import bpy
from mathutils.kdtree import KDTree
import numpy as np


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--projected", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples-per-action", type=int, default=17)
    parser.add_argument("--maximum-error-m", type=float, default=1.0e-4)
    return parser.parse_args(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)


def _import(path: Path) -> tuple[Any, list[Any], dict[str, Any]]:
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB import failed: {path}: {result}")
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one armature in {path}, found {len(armatures)}")
    armature = armatures[0]
    meshes = sorted(
        [
            item
            for item in bpy.context.scene.objects
            if item.type == "MESH"
            and (
                item.parent == armature
                or any(
                    modifier.type == "ARMATURE" and modifier.object == armature
                    for modifier in item.modifiers
                )
            )
        ],
        key=lambda item: item.name,
    )
    if not meshes:
        raise RuntimeError(f"no skinned mesh in {path}")
    actions: dict[str, Any] = {}
    for action in bpy.data.actions:
        lower = action.name.lower()
        semantic = "walk" if "walk" in lower else "idle" if "idle" in lower else None
        if semantic is None:
            continue
        if semantic in actions:
            raise RuntimeError(f"ambiguous {semantic} action in {path}")
        actions[semantic] = action
    if set(actions) != {"idle", "walk"}:
        raise RuntimeError(f"Idle/Walking actions are not unique in {path}")
    armature.animation_data_create()
    return armature, meshes, actions


def _evaluated_vertices(meshes: list[Any]) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arrays: list[np.ndarray] = []
    for mesh in meshes:
        evaluated = mesh.evaluated_get(depsgraph)
        temporary = evaluated.to_mesh()
        try:
            local = np.ones((len(temporary.vertices), 4), dtype=np.float64)
            for index, vertex in enumerate(temporary.vertices):
                local[index, :3] = vertex.co
            matrix = np.asarray(mesh.matrix_world, dtype=np.float64)
            arrays.append((matrix @ local.T).T[:, :3])
        finally:
            evaluated.to_mesh_clear()
    result = np.concatenate(arrays, axis=0)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("evaluated mesh contains non-finite vertices")
    return result


def _snapshots(
    path: Path, fractions: np.ndarray
) -> tuple[dict[str, list[np.ndarray]], list[dict[str, Any]]]:
    _clear()
    armature, meshes, actions = _import(path)
    values: dict[str, list[np.ndarray]] = {}
    manifest: list[dict[str, Any]] = []
    for semantic in ("idle", "walk"):
        action = actions[semantic]
        start, end = map(float, action.frame_range)
        if not math.isfinite(start + end) or end <= start:
            raise RuntimeError(f"invalid action range: {action.name}")
        frames = (start + fractions * (end - start)).tolist()
        armature.animation_data.action = action
        per_action: list[np.ndarray] = []
        for frame in frames:
            base = math.floor(frame)
            bpy.context.scene.frame_set(base, subframe=frame - base)
            bpy.context.view_layer.update()
            per_action.append(_evaluated_vertices(meshes))
        values[semantic] = per_action
        manifest.append(
            {
                "semantic_action_id": semantic,
                "action_name": action.name,
                "frame_range": [start, end],
                "sampled_frames": frames,
            }
        )
    armature.animation_data.action = None
    return values, manifest


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    tree = KDTree(len(target))
    for index, point in enumerate(target):
        tree.insert(point, index)
    tree.balance()
    result = np.empty(len(source), dtype=np.float64)
    for index, point in enumerate(source):
        _nearest, _target_index, distance = tree.find(point)
        result[index] = float(distance)
    return result


def main() -> int:
    args = _arguments()
    source = args.source.resolve()
    projected = args.projected.resolve()
    report_path = args.report.resolve()
    if args.samples_per_action < 3:
        raise ValueError("samples-per-action must be at least 3")
    if not math.isfinite(args.maximum_error_m) or args.maximum_error_m < 0.0:
        raise ValueError("maximum-error-m must be finite and non-negative")
    for path in (source, projected):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    if report_path.exists() or report_path.is_symlink():
        raise ValueError(f"refusing to replace report: {report_path}")
    fractions = np.linspace(0.0, 1.0, args.samples_per_action, dtype=np.float64)
    source_values, source_actions = _snapshots(source, fractions)
    projected_values, projected_actions = _snapshots(projected, fractions)
    samples: list[dict[str, Any]] = []
    maximum = 0.0
    squared_sum = 0.0
    distance_count = 0
    all_distances: list[np.ndarray] = []
    source_min: np.ndarray | None = None
    source_max: np.ndarray | None = None
    for semantic in ("idle", "walk"):
        for fraction, expected, actual in zip(
            fractions,
            source_values[semantic],
            projected_values[semantic],
            strict=True,
        ):
            expected_to_actual = _nearest_distances(expected, actual)
            actual_to_expected = _nearest_distances(actual, expected)
            distances = np.concatenate((expected_to_actual, actual_to_expected))
            frame_maximum = float(np.max(distances))
            frame_rms = float(np.sqrt(np.mean(np.square(distances))))
            maximum = max(maximum, frame_maximum)
            squared_sum += float(np.sum(np.square(distances)))
            distance_count += len(distances)
            all_distances.append(distances)
            current_min = np.min(expected, axis=0)
            current_max = np.max(expected, axis=0)
            source_min = (
                current_min
                if source_min is None
                else np.minimum(source_min, current_min)
            )
            source_max = (
                current_max
                if source_max is None
                else np.maximum(source_max, current_max)
            )
            samples.append(
                {
                    "semantic_action_id": semantic,
                    "normalized_time": float(fraction),
                    "maximum_vertex_distance_m": frame_maximum,
                    "rms_vertex_distance_m": frame_rms,
                }
            )
    assert source_min is not None and source_max is not None
    diagonal = float(np.linalg.norm(source_max - source_min))
    flattened = np.concatenate(all_distances)
    rms = float(math.sqrt(squared_sum / distance_count))
    status = "pass" if maximum <= args.maximum_error_m else "fail"
    report = {
        "schema": "avengine_motion_projection_deformation_v1",
        "status": status,
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "byte_size": source.stat().st_size,
        },
        "projected": {
            "path": str(projected),
            "sha256": _sha256(projected),
            "byte_size": projected.stat().st_size,
        },
        "source_actions": source_actions,
        "projected_actions": projected_actions,
        "sample_count_per_action": args.samples_per_action,
        "source_vertex_count": int(source_values["idle"][0].shape[0]),
        "projected_vertex_count": int(projected_values["idle"][0].shape[0]),
        "maximum_vertex_distance_m": maximum,
        "rms_vertex_distance_m": rms,
        "p95_vertex_distance_m": float(np.percentile(flattened, 95.0)),
        "source_sampled_bounds_diagonal_m": diagonal,
        "maximum_error_fraction_of_bounds_diagonal": maximum / diagonal,
        "threshold_maximum_vertex_distance_m": args.maximum_error_m,
        "samples": samples,
        "notes": [
            "Distances are symmetric nearest-vertex distances because a clean GLB export may split vertices at material or normal seams.",
            "A structurally valid rotation-only candidate is not deformation-equivalent unless this report passes.",
            "A failing projection may still be inspected as an explicitly bounded research diagnostic; it is not admission evidence.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "report": str(report_path),
                "maximum_vertex_distance_m": maximum,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

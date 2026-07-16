#!/usr/bin/env python3
"""Verify sampled deformation equivalence across the M2 GLB root rebase.

Run with Blender.  The rebase is allowed to change coordinates only by the
rigid ``canonical_root_from_source_mesh`` transform recorded in its report.
This verifier imports both GLBs independently, samples every required action,
maps canonical evaluated vertices back into the source basis, and rejects any
non-rigid deformation drift.  It proves technical equivalence, not semantic
mesh/animation quality or qualification.
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
import numpy as np


_MAX_VERTEX_ERROR_M = 1.0e-4


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rebased", type=Path, required=True)
    parser.add_argument("--rebase-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--samples-per-action", type=int, default=9)
    return parser.parse_args(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)


def _import(path: Path) -> tuple[Any, Any, dict[str, Any]]:
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB import failed: {path}: {result}")
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(
            f"expected one armature, found {[obj.name for obj in armatures]}"
        )
    armature = armatures[0]
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and any(
            modifier.type == "ARMATURE" and modifier.object == armature
            for modifier in obj.modifiers
        )
    ]
    if len(meshes) != 1:
        raise RuntimeError(
            f"expected one skinned mesh, found {[obj.name for obj in meshes]}"
        )
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
        raise RuntimeError(f"required Idle/Walk actions missing in {path}")
    armature.animation_data_create()
    return armature, meshes[0], actions


def _sample_frames(action: Any, count: int) -> list[float]:
    if count < 3:
        raise RuntimeError("samples-per-action must be at least 3")
    start, end = (float(value) for value in action.frame_range)
    if not math.isfinite(start + end) or end <= start:
        raise RuntimeError(f"invalid action range: {action.name}")
    return np.linspace(start, end, count, dtype=np.float64).tolist()


def _evaluated_vertices(mesh: Any) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    temporary = evaluated.to_mesh()
    try:
        matrix = np.asarray(mesh.matrix_world, dtype=np.float64)
        local = np.empty((len(temporary.vertices), 4), dtype=np.float64)
        for index, vertex in enumerate(temporary.vertices):
            local[index, :3] = vertex.co
            local[index, 3] = 1.0
        result = (matrix @ local.T).T[:, :3]
        if not np.all(np.isfinite(result)):
            raise RuntimeError("evaluated mesh contains non-finite vertices")
        return result
    finally:
        evaluated.to_mesh_clear()


def _snapshots(
    path: Path, frames_by_action: dict[str, list[float]] | None
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[float]], list[dict[str, Any]]]:
    _clear_scene()
    armature, mesh, actions = _import(path)
    frames = frames_by_action or {
        semantic: _sample_frames(action, _arguments_cache.samples_per_action)
        for semantic, action in actions.items()
    }
    snapshots: dict[str, list[np.ndarray]] = {}
    manifest: list[dict[str, Any]] = []
    for semantic in ("idle", "walk"):
        action = actions[semantic]
        expected_range = [float(value) for value in action.frame_range]
        if frames_by_action is not None:
            source_range = [frames[semantic][0], frames[semantic][-1]]
            if not np.allclose(expected_range, source_range, atol=1.0e-6):
                raise RuntimeError(
                    f"{semantic} frame range changed: {expected_range} != {source_range}"
                )
        armature.animation_data.action = action
        values: list[np.ndarray] = []
        for frame in frames[semantic]:
            bpy.context.scene.frame_set(int(math.floor(frame)), subframe=frame % 1.0)
            bpy.context.view_layer.update()
            values.append(_evaluated_vertices(mesh))
        snapshots[semantic] = values
        manifest.append(
            {
                "semantic": semantic,
                "action": action.name,
                "frame_range": expected_range,
                "sampled_frames": frames[semantic],
            }
        )
    armature.animation_data.action = None
    return snapshots, frames, manifest


def _blender_basis_transform(gltf_transform: np.ndarray) -> np.ndarray:
    # glTF is +Y up; Blender imports it into +Z up as (x, -z, y).
    basis = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return basis @ gltf_transform @ np.linalg.inv(basis)


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    return (transform @ homogeneous.T).T[:, :3]


def main() -> int:
    global _arguments_cache
    _arguments_cache = _arguments()
    source = _arguments_cache.source.resolve()
    rebased = _arguments_cache.rebased.resolve()
    rebase_report_path = _arguments_cache.rebase_report.resolve()
    output_report = _arguments_cache.output_report.resolve()
    for path in (source, rebased, rebase_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    rebase_report = json.loads(rebase_report_path.read_text(encoding="utf-8"))
    source_sha256 = _sha256(source)
    rebased_sha256 = _sha256(rebased)
    if rebase_report["source"]["sha256"] != source_sha256:
        raise RuntimeError("rebase report is not bound to the source GLB")
    if rebase_report["output"]["sha256"] != rebased_sha256:
        raise RuntimeError("rebase report is not bound to the rebased GLB")

    canonical_from_source = np.asarray(
        rebase_report["skin"]["canonical_root_from_source_bind"],
        dtype=np.float64,
    )
    if canonical_from_source.shape != (4, 4):
        raise RuntimeError("rebase report transform must be 4x4")
    linear = canonical_from_source[:3, :3]
    rigid_error = float(np.max(np.abs(linear.T @ linear - np.eye(3))))
    determinant = float(np.linalg.det(linear))
    if rigid_error > 5.0e-5 or abs(determinant - 1.0) > 5.0e-5:
        raise RuntimeError("canonical transform is not a proper rigid transform")
    source_from_canonical = np.linalg.inv(
        _blender_basis_transform(canonical_from_source)
    )

    source_snapshots, frames, source_actions = _snapshots(source, None)
    rebased_snapshots, _, rebased_actions = _snapshots(rebased, frames)
    errors: list[dict[str, Any]] = []
    maximum_error = 0.0
    for semantic in ("idle", "walk"):
        for frame, expected, canonical in zip(
            frames[semantic],
            source_snapshots[semantic],
            rebased_snapshots[semantic],
            strict=True,
        ):
            actual = _transform_points(source_from_canonical, canonical)
            if expected.shape != actual.shape:
                raise RuntimeError("vertex topology/order changed across rebase")
            error = float(np.max(np.abs(expected - actual)))
            maximum_error = max(maximum_error, error)
            errors.append(
                {"semantic": semantic, "frame": frame, "maximum_error_m": error}
            )

    status = "pass" if maximum_error <= _MAX_VERTEX_ERROR_M else "fail"
    report = {
        "schema": "avengine_m2_rebase_deformation_verification_v1",
        "status": status,
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source": {"path": str(source), "sha256": source_sha256},
        "rebased": {"path": str(rebased), "sha256": rebased_sha256},
        "rebase_report": {
            "path": str(rebase_report_path),
            "sha256": _sha256(rebase_report_path),
        },
        "source_actions": source_actions,
        "rebased_actions": rebased_actions,
        "samples": errors,
        "maximum_vertex_error_m": maximum_error,
        "threshold_maximum_vertex_error_m": _MAX_VERTEX_ERROR_M,
        "canonical_transform_rigid_error": rigid_error,
        "canonical_transform_determinant": determinant,
        "notes": [
            "This gate proves sampled deformation equivalence under the declared rigid basis change only.",
            "Semantic mesh/animation alignment, contacts, Habitat playback, and human review remain separate gates.",
        ],
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "report": str(output_report)}))
    return 0 if status == "pass" else 1


_arguments_cache: argparse.Namespace


if __name__ == "__main__":
    raise SystemExit(main())

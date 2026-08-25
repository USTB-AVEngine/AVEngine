from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.acoustics.compiler import compile_usd_snapshot_semantic_research_scene
from avengine.acoustics.contracts import load_and_validate_acoustic_scene_package
from avengine.acoustics.usd_snapshot import (
    USD_ACOUSTIC_SNAPSHOT_SCHEMA,
    UsdAcousticSnapshotError,
    load_usd_acoustic_snapshot,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES = (
    REPOSITORY_ROOT
    / "examples/m3/semantic_materials/residential_material_rules.json"
)


def _json_array(value: object) -> np.ndarray:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return np.frombuffer(payload, dtype="u1")


def _write_fixture(root: Path, *, surface_name: str = "usd::wall::Paint") -> Path:
    vertices = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype="<f4",
    )
    triangles = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype="<u4",
    )
    identity = np.eye(4, dtype=float).reshape(-1).tolist()
    objects = [
        {
            "object_id": "/Root/Meshes/wall/wall_0000/mesh#material=Paint",
            "source_node_index": 0,
            "source_mesh_index": 0,
            "source_primitive_index": 0,
            "source_material_name": surface_name,
            "vertex_offset": 0,
            "vertex_count": len(vertices),
            "triangle_offset": 0,
            "triangle_count": len(triangles),
            "world_from_object": identity,
            "source_world_matrix": identity,
            "transform_baked": True,
        }
    ]
    source_to_canonical = {
        "matrix_row_major": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "source": "unit Z-up [x,y,z] to canonical [x,z,y]",
        "reviewed": True,
    }
    metadata = {
        "schema": USD_ACOUSTIC_SNAPSHOT_SCHEMA,
        "room_id": "unit_usd_room",
        "source_stage": "/external/unit.usda",
        "source_stage_sha256": "0" * 64,
        "source_to_canonical": source_to_canonical,
        "source_primitive_count": 1,
        "source_node_instance_count": 1,
        "source_mesh_prim_count": 1,
        "visible_mesh_prim_count": 1,
        "hidden_mesh_prim_count": 0,
        "reviewed_interior_origins_m": [[0.0, 0.0, 0.0]],
        "geometry_claim": "unit_real_surface",
    }
    surfaces = [
        {
            "source_material_name": "usd::wall::Paint",
            "semantic_category": "wall",
            "identity_key": "unit_usd_room/wall/Paint",
            "material_slot": "Paint",
            "object_name": "wall_0000",
        }
    ]
    snapshot = root / "scene_snapshot.npz"
    np.savez(
        snapshot,
        vertices=vertices,
        triangles=triangles,
        metadata_json_utf8=_json_array(metadata),
        objects_json_utf8=_json_array(objects),
        surfaces_json_utf8=_json_array(surfaces),
    )
    room = {
        "schema": "avengine_room_package_v1",
        "room_id": "unit_usd_room",
        "room_kind": "external_usd_real_surface",
        "geometry_representation": "real_surface_mesh",
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "-Z",
            "linear_unit": "meter",
            "quaternion_order": "xyzw",
        },
        "scene": {
            "scene_id_kind": "path",
            "scene_id": "scene_snapshot.npz",
            "dataset_config_path": "not_applicable_external_usd",
            "navmesh_path": "not_applicable_external_usd",
            "navmesh_policy": "recompute_if_missing",
            "load_semantic_mesh": False,
            "enable_physics": False,
        },
        "assets": [
            {"role": "render_surface_mesh", "path": "scene_snapshot.npz"},
            {
                "role": "scene_dataset_config",
                "path": "not_applicable_external_usd",
            },
            {"role": "navmesh", "path": "not_applicable_external_usd"},
        ],
        "semantics": {"interpretation": "unit USD object/material identities"},
        "navigation": {
            "agent_height_m": 1.5,
            "agent_radius_m": 0.2,
            "include_static_objects": True,
        },
        "openings": [],
        "connectivity_pairs": [
            {
                "pair_id": "unit",
                "start_m": [0.0, 0.0, 0.0],
                "end_m": [0.1, 0.0, 0.0],
            }
        ],
        "ray_checks": [],
        "acoustics": {
            "status": "deferred_to_m3",
            "reason": "unit semantic compilation",
        },
        "provenance": {
            "source": "unit USD fixture",
            "source_revision": "v1",
        },
        "surface_audit": {
            "method": "unit explicit cube",
            "aabb_proxy": False,
        },
    }
    (root / "room_manifest.json").write_text(
        json.dumps(room, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot


def test_usd_snapshot_compiles_to_existing_m3_package(tmp_path: Path) -> None:
    snapshot = _write_fixture(tmp_path)

    loaded = load_usd_acoustic_snapshot(snapshot)
    assert loaded.scene.source_triangle_count == 12
    assert loaded.scene.source_node_instance_count == 1
    assert loaded.surfaces[0].semantic_category == "wall"

    manifest, coverage_path = compile_usd_snapshot_semantic_research_scene(
        room_manifest=tmp_path / "room_manifest.json",
        material_rules=RULES,
        output=tmp_path / "package",
        seed=917,
        probe_origins=[[0.0, 0.0, 0.0]],
        probe_direction_count=8,
    )
    package = load_and_validate_acoustic_scene_package(manifest)
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))

    assert package.manifest["package_mode"] == "research_candidate"
    assert package.manifest["room_kind"] == "external_usd_real_surface"
    assert package.triangle_count == 12
    assert package.qa_reports["ray_leakage"]["automatic_enclosure_probe"][
        "escaped_ray_count"
    ] == 0
    assert coverage["source_kind"] == "composed_usd_acoustic_snapshot"
    assert coverage["physical_material_claim"] is False
    assert coverage["resolution_counts"] == {"semantic_category": 1}


def test_usd_snapshot_rejects_surface_object_identity_mismatch(
    tmp_path: Path,
) -> None:
    snapshot = _write_fixture(
        tmp_path, surface_name="usd::wall::DifferentObjectName"
    )

    with pytest.raises(
        UsdAcousticSnapshotError,
        match="surfaces must exactly cover object source material names",
    ):
        load_usd_acoustic_snapshot(snapshot)

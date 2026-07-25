from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np

from avengine.m3.contracts import (
    validate_mapping_document,
    validate_material_database_document,
)
from avengine.m3.qa import automatic_mesh_leakage_report
from avengine.m3.semantic import load_mp3d_semantic_scene
from avengine.m3.semantic_materials import (
    SemanticSurfaceIdentity,
    compile_semantic_material_documents,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRACKED_RULES = (
    REPOSITORY_ROOT
    / "examples/m3/semantic_materials/residential_material_rules.json"
)


def _write_semantic_ply(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int, int]],
) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "property int object_id\n"
        "end_header\n"
    ).encode("ascii")
    payload = bytearray(header)
    for x, y, z in vertices:
        payload.extend(struct.pack("<fffBBB", x, y, z, 0, 0, 0))
    for a, b, c, object_id in faces:
        payload.extend(struct.pack("<Biiii", 3, a, b, c, object_id))
    path.write_bytes(payload)


def _rules() -> dict:
    return {
        "schema": "avengine_m3_semantic_material_rules_v1",
        "ruleset_id": "unit_residential_v1",
        "bands_hz": [125.0, 500.0, 2000.0, 4000.0],
        "coefficient_jitter_std": 0.04,
        "default_candidates": [{"material": "generic", "weight": 1.0}],
        "materials": {
            "wood": {
                "name": "Wood",
                "absorption": [0.10, 0.12, 0.18, 0.22],
                "scattering": [0.08, 0.10, 0.12, 0.14],
                "transmission": [0.01, 0.01, 0.01, 0.01],
                "damping": [0.0, 0.0, 0.0, 0.0],
                "density": 1.225,
                "speed": 343.0,
                "source": "unit fixture",
                "confidence": 0.6,
            },
            "tile": {
                "name": "Tile",
                "absorption": [0.02, 0.03, 0.04, 0.05],
                "scattering": [0.03, 0.04, 0.05, 0.06],
                "transmission": [0.0, 0.0, 0.0, 0.0],
                "damping": [0.0, 0.0, 0.0, 0.0],
                "density": 1.225,
                "speed": 343.0,
                "source": "unit fixture",
                "confidence": 0.6,
            },
            "carpet": {
                "name": "Heavy carpet",
                "absorption": [0.05, 0.18, 0.48, 0.62],
                "scattering": [0.10, 0.15, 0.20, 0.25],
                "transmission": [0.0, 0.0, 0.0, 0.0],
                "damping": [0.0, 0.0, 0.0, 0.0],
                "density": 1.225,
                "speed": 343.0,
                "source": "unit fixture",
                "confidence": 0.6,
            },
            "generic": {
                "name": "Generic residential surface",
                "absorption": [0.10, 0.15, 0.20, 0.25],
                "scattering": [0.10, 0.10, 0.10, 0.10],
                "transmission": [0.0, 0.0, 0.0, 0.0],
                "damping": [0.0, 0.0, 0.0, 0.0],
                "density": 1.225,
                "speed": 343.0,
                "source": "unit fixture",
                "confidence": 0.2,
            },
        },
        "categories": {
            "floor": {
                "candidates": [
                    {"material": "wood", "weight": 0.7},
                    {"material": "tile", "weight": 0.3},
                ]
            },
            "chair": {"candidates": [{"material": "wood", "weight": 1.0}]},
        },
        "name_hints": [
            {
                "contains": ["rug"],
                "candidates": [{"material": "carpet", "weight": 1.0}],
            }
        ],
        "explicit_overrides": {
            "apartment/rug_01": "carpet",
        },
    }


def test_mp3d_house_and_binary_semantic_ply_resolve_face_categories(
    tmp_path: Path,
) -> None:
    house = tmp_path / "tiny.house"
    house.write_text(
        "ASCII 1.1\n"
        "C  0  1 wall  1 wall  0 0 0 0 0\n"
        "C  1  2 floor  2 floor  0 0 0 0 0\n"
        "O  0 0 0  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0 0 0 0 0\n"
        "O  1 0 1  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    ply = tmp_path / "tiny_semantic.ply"
    _write_semantic_ply(
        ply,
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        [(0, 1, 2, 0), (0, 2, 3, 1)],
    )

    scene = load_mp3d_semantic_scene(ply, house)

    assert scene.source_vertex_count == 4
    assert scene.source_triangle_count == 2
    assert scene.semantic_categories == ("floor", "wall")
    assert [item["source_material_name"] for item in scene.objects] == [
        "floor",
        "wall",
    ]
    assert scene.objects[0]["triangle_offset"] == 0
    assert scene.objects[1]["triangle_offset"] == 1
    assert scene.triangles.shape == (2, 3)
    assert int(scene.triangles.max()) < len(scene.vertices)


def test_semantic_material_precedence_is_deterministic_and_reports_unknowns() -> None:
    surfaces = [
        SemanticSurfaceIdentity(
            source_material_name="rug",
            semantic_category="floor",
            identity_key="apartment/rug_01",
            object_name="rug_01",
        ),
        SemanticSurfaceIdentity(
            source_material_name="chair",
            semantic_category="chair",
            identity_key="apartment/chair_01",
            material_slot="rug_like_slot",
        ),
        SemanticSurfaceIdentity(
            source_material_name="floor",
            semantic_category="floor",
            identity_key="apartment/floor",
        ),
        SemanticSurfaceIdentity(
            source_material_name="mystery",
            semantic_category="unclassified_fixture",
            identity_key="apartment/mystery",
        ),
    ]
    kwargs = {
        "room_id": "apartment",
        "surfaces": surfaces,
        "rules": _rules(),
        "seed": 917,
        "source_to_canonical": {
            "matrix_row_major": np.eye(4).reshape(-1).tolist(),
            "source": "unit identity",
            "reviewed": True,
        },
    }

    first = compile_semantic_material_documents(**kwargs)
    second = compile_semantic_material_documents(**kwargs)

    assert first.mapping == second.mapping
    assert first.database == second.database
    assert first.report == second.report
    assert validate_mapping_document(first.mapping, room_id="apartment") == []
    assert validate_material_database_document(first.database) == []

    decisions = {
        item["source_material_name"]: item
        for item in first.report["decisions"]
    }
    assert decisions["rug"]["resolution"] == "explicit_override"
    assert decisions["rug"]["selected_material"] == "carpet"
    assert decisions["chair"]["resolution"] == "name_hint"
    assert decisions["chair"]["selected_material"] == "carpet"
    assert decisions["floor"]["resolution"] == "semantic_category"
    assert decisions["mystery"]["resolution"] == "default_candidate"
    assert first.report["unknown_semantic_categories"] == [
        "unclassified_fixture"
    ]
    assert first.report["precedence"] == [
        "explicit_override",
        "material_slot_or_object_name_hint",
        "semantic_category",
        "plausible_default_candidate_set",
    ]

    for material in first.database["materials"]:
        for field in ("absorption", "scattering", "transmission"):
            assert all(0.0 <= value <= 1.0 for value in material[field])


def test_tracked_residential_rules_compile_to_existing_m3_contracts() -> None:
    rules = json.loads(TRACKED_RULES.read_text(encoding="utf-8"))
    compiled = compile_semantic_material_documents(
        room_id="tracked_rules_fixture",
        surfaces=[
            SemanticSurfaceIdentity("floor", "floor"),
            SemanticSurfaceIdentity("wall", "wall"),
            SemanticSurfaceIdentity("window", "window"),
            SemanticSurfaceIdentity("unknown_object", "unknown_object"),
        ],
        rules=rules,
        seed=917,
        source_to_canonical={
            "matrix_row_major": np.eye(4).reshape(-1).tolist(),
            "source": "unit identity",
            "reviewed": True,
        },
    )

    assert validate_mapping_document(
        compiled.mapping, room_id="tracked_rules_fixture"
    ) == []
    assert validate_material_database_document(compiled.database) == []
    assert compiled.report["unknown_semantic_categories"] == ["unknown_object"]


def _cube_mesh(*, open_top: bool) -> tuple[np.ndarray, np.ndarray]:
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
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ]
    if open_top:
        del faces[6:8]
    return vertices, np.asarray(faces, dtype="<u4")


def test_automatic_leakage_probe_distinguishes_closed_and_open_mesh() -> None:
    directions = np.asarray(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=np.float64,
    )
    closed_vertices, closed_triangles = _cube_mesh(open_top=False)
    open_vertices, open_triangles = _cube_mesh(open_top=True)

    closed = automatic_mesh_leakage_report(
        closed_vertices,
        closed_triangles,
        origins=[[0.0, 0.0, 0.0]],
        directions=directions,
    )
    opened = automatic_mesh_leakage_report(
        open_vertices,
        open_triangles,
        origins=[[0.0, 0.0, 0.0]],
        directions=directions,
    )

    assert closed["status"] == "diagnostic_complete"
    assert closed["escaped_ray_count"] == 0
    assert closed["escape_fraction"] == 0.0
    assert opened["escaped_ray_count"] == 1
    assert opened["escape_fraction"] == 1 / 6
    assert opened["origins"][0]["escaped_direction_indices"] == [2]

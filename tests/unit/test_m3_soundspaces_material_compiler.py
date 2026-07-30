from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from avengine.cli import main
from avengine.m3.contracts import load_and_validate_acoustic_scene_package


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_semantic_ply(path: Path) -> None:
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    faces = [
        (0, 2, 1, 0),
        (0, 1, 3, 0),
        (0, 3, 2, 1),
        (1, 2, 3, 1),
    ]
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


def _curve(first: float, second: float) -> list[float]:
    return [125.0, first, 500.0, second]


def _material(name: str, labels: list[str], base: float) -> dict[str, object]:
    return {
        "name": name,
        "labels": labels,
        "absorption": _curve(base, base + 0.01),
        "scattering": _curve(0.1, 0.2),
        "transmission": _curve(0.01, 0.0),
        "damping": [20.0, 0.0, 20_000.0, 0.001],
        "density": 1.2,
        "speed": 343.0,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    semantic = tmp_path / "tiny_semantic.ply"
    _write_semantic_ply(semantic)
    descriptor = tmp_path / "tiny.house"
    descriptor.write_text(
        "ASCII 1.1\n"
        "C  0  1 chopping-board  1 chopping-board  0 0 0 0 0\n"
        "C  1  2 mystery  2 mystery  0 0 0 0 0\n"
        "O  0 0 0  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0 0 0 0 0\n"
        "O  1 0 1  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    visual = tmp_path / "visual.glb"
    visual.write_bytes(b"unused hermetic visual placeholder")
    dataset_config = tmp_path / "dataset.scene_dataset_config.json"
    _write_json(dataset_config, {})
    navmesh = tmp_path / "tiny.navmesh"
    navmesh.write_bytes(b"unused hermetic navmesh placeholder")
    room = {
        "schema": "avengine_room_package_v1",
        "room_id": "unit_soundspaces_mp3d",
        "room_kind": "habitat_native",
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
            "scene_id": str(visual),
            "dataset_config_path": str(dataset_config),
            "navmesh_path": str(navmesh),
            "navmesh_policy": "load_declared",
            "load_semantic_mesh": True,
            "enable_physics": False,
        },
        "assets": [
            {
                "role": "render_surface_mesh",
                "path": str(visual),
                "license": "unit fixture",
                "redistribution": "generated_test_fixture",
            },
            {
                "role": "semantic_surface_mesh",
                "path": str(semantic),
                "license": "unit fixture",
                "redistribution": "generated_test_fixture",
            },
            {
                "role": "semantic_descriptor",
                "path": str(descriptor),
                "license": "unit fixture",
                "redistribution": "generated_test_fixture",
            },
            {
                "role": "scene_dataset_config",
                "path": str(dataset_config),
                "license": "unit fixture",
                "redistribution": "generated_test_fixture",
            },
            {
                "role": "navmesh",
                "path": str(navmesh),
                "license": "unit fixture",
                "redistribution": "generated_test_fixture",
            },
        ],
        "semantics": {
            "interpretation": "unit MP3D semantic mesh IDs use the paired descriptor"
        },
        "navigation": {
            "agent_height_m": 1.5,
            "agent_radius_m": 0.1,
            "include_static_objects": False,
        },
        "openings": [],
        "connectivity_pairs": [
            {
                "pair_id": "inside",
                "start_m": [0.1, 0.1, 0.1],
                "end_m": [0.2, 0.1, 0.1],
            }
        ],
        "ray_checks": [],
        "acoustics": {
            "status": "deferred_to_m3",
            "reason": "unit fixture",
        },
        "provenance": {
            "source": "unit fixture",
            "source_revision": "unit-v1",
        },
    }
    room_path = tmp_path / "room.json"
    _write_json(room_path, room)

    materials = {
        "materials": [
            _material("Default", ["default"], 0.10),
            _material("Wood, Thick", ["chopping-board"], 0.30),
        ]
    }
    materials_path = tmp_path / "mp3d_material_config.json"
    _write_json(materials_path, materials)
    return room_path, materials_path


def test_cli_compiles_soundspaces_materials_into_standard_acoustic_package(
    tmp_path: Path,
) -> None:
    room, materials = _fixture(tmp_path)
    output = tmp_path / "package"

    assert (
        main(
            [
                "m3",
                "compile-mp3d-rlr-materials",
                "--room",
                str(room),
                "--materials",
                str(materials),
                "--database-id",
                "unit_soundspaces_mp3d_v1",
                "--source-description",
                "unit SoundSpaces/RLR material config",
                "--source-uri",
                "https://example.invalid/mp3d_material_config.json",
                "--probe-origin",
                "0.1",
                "0.1",
                "0.1",
                "--probe-directions",
                "4",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    package = load_and_validate_acoustic_scene_package(output / "manifest.json")
    coverage = json.loads(
        (output / "semantic_material_coverage.json").read_text(encoding="utf-8")
    )
    assert package.manifest["package_mode"] == "research_candidate"
    assert package.manifest["materials"]["mapping_source_kind"] == "semantic_proposal"
    material_config_sha256 = hashlib.sha256(materials.read_bytes()).hexdigest()
    assert package.manifest["materials"]["acoustic_profile_binding"] == {
        "schema": "avengine_m3_acoustic_profile_binding_v1",
        "profile_id": "unit_soundspaces_mp3d_v1",
        "profile_revision": "1",
        "adapter_id": "soundspaces2_mp3d_semantic_labels_v1",
        "resources": [
            {
                "role": "soundspaces2_public_material_config",
                "sha256": material_config_sha256,
            }
        ],
    }
    assert package.triangle_count == 4
    assert coverage["source_kind"] == "soundspaces_rlr_mp3d_semantic_ply_house"
    assert coverage["coverage"]["official_substring_match_category_count"] == 1
    assert coverage["coverage"]["official_default_category_count"] == 1
    assert coverage["triangle_coverage"] == {
        "triangle_count": 4,
        "official_substring_match_triangle_count": 2,
        "official_default_triangle_count": 2,
        "official_substring_match_triangle_fraction": 0.5,
        "official_default_triangle_fraction": 0.5,
    }
    fallback = next(
        decision
        for decision in coverage["decisions"]
        if decision["source_semantic_label"] == "mystery"
    )
    assert fallback["official_default_applied"] is True
    assert fallback["triangle_count"] == 2
    assert fallback["rlr_category_name"].startswith("avengine_rlr_alias_")
    matched = next(
        decision
        for decision in coverage["decisions"]
        if decision["source_semantic_label"] == "chopping_board"
    )
    assert matched["canonical_semantic_category"] == "chopping_board"
    assert matched["raw_semantic_category_label"] == "chopping-board"
    assert matched["assignment_kind"] == "official_substring_match"
    assert matched["official_matched_labels"] == ["chopping-board"]
    assert matched["official_default_applied"] is False
    assert coverage["source_material_config"]["uri"].startswith("https://")
    assert len(coverage["source_material_config"]["sha256"]) == 64

    source_by_name = {
        material["name"]: material
        for material in json.loads(materials.read_text(encoding="utf-8"))[
            "materials"
        ]
    }
    packaged_by_key = {
        material["material_key"]: material
        for material in package.source_material_database["materials"]
    }
    assert len(packaged_by_key) == len(coverage["decisions"]) == 2
    for decision in coverage["decisions"]:
        source_material = source_by_name[decision["selected_source_material_name"]]
        packaged = packaged_by_key[decision["runtime_material_key"]]
        assert packaged["labels"] == [decision["rlr_category_name"]]
        assert decision["parameters_preserved_exactly"] is True
        assert decision["source_parameter_sha256"] == (
            decision["runtime_parameter_sha256"]
        )
        for field in (
            "absorption",
            "scattering",
            "transmission",
            "damping",
            "density",
            "speed",
        ):
            assert packaged[field] == source_material[field]

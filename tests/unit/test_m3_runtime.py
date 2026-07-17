from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    write_json,
)
from avengine.m3.compiler import compile_custom_acoustic_scene
from avengine.m3.contracts import load_and_validate_acoustic_scene_package
from avengine.m3.runtime import (
    RLRSimulationConfig,
    RuntimeAnchor,
    RuntimeContractError,
    RuntimeUnavailableError,
    load_compiled_acoustic_scene,
    load_habitat_runtime,
    simulate_compiled_acoustic_scene,
)
import avengine.m3.runtime as runtime_module


REPOSITORY = Path(__file__).resolve().parents[2]


def _simulation_mapping() -> dict[str, Any]:
    return {
        "frequency_bands": 4,
        "direct_sh_order": 3,
        "indirect_sh_order": 1,
        "direct_ray_count": 500,
        "indirect_ray_count": 5_000,
        "indirect_ray_depth": 200,
        "source_ray_count": 200,
        "source_ray_depth": 10,
        "max_diffraction_order": 10,
        "thread_count": 1,
        "sample_rate_hz": 16_000.0,
        "max_ir_seconds": 1.0,
        "unit_scale": 1.0,
        "global_volume": 1.0,
        "speed_of_sound_m_s": 343.0,
        "direct": True,
        "indirect": True,
        "diffraction": True,
        "transmission": True,
        "mesh_simplification": False,
        "temporal_coherence": False,
        "channel_layout": {"type": "mono", "channel_count": 1},
    }


def _array_record(path: Path, *, root: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        **file_record(path, relative_to=root),
        "format": "npy",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "memory_order": "C",
    }


def _compiled_package(tmp_path: Path) -> Path:
    return compile_custom_acoustic_scene(
        room_manifest=REPOSITORY
        / "examples/m1/rooms/blender_custom/room_manifest.json",
        material_mapping=REPOSITORY / "examples/m3/blender_custom/mapping.json",
        material_database=REPOSITORY
        / "examples/m3/blender_custom/materials_low.json",
        output=tmp_path / "package",
        package_id="runtime-unit-package",
        environment={"AVENGINE_REPOSITORY_ROOT": str(REPOSITORY)},
    )

    # Historical hand-written fixture retained below only as source context;
    # the executable path above intentionally uses the formal compiler so the
    # runtime test cannot drift behind the production package validator.
    root = tmp_path / "package"
    acoustic = root / "acoustic"
    acoustic.mkdir(parents=True)
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="<f4",
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<u4")
    material_ids = np.asarray([0, 1], dtype="<u4")
    vertices_path = acoustic / "vertices.npy"
    triangles_path = acoustic / "triangles.npy"
    material_ids_path = acoustic / "triangle_material_ids.npy"
    np.save(vertices_path, vertices, allow_pickle=False)
    np.save(triangles_path, triangles, allow_pickle=False)
    np.save(material_ids_path, material_ids, allow_pickle=False)
    categories_path = acoustic / "material_categories.json"
    write_json(
        categories_path,
        {
            "schema": "avengine_acoustic_material_categories_v1",
            "mapping_id": "unit-mapping",
            "room_id": "unit-room",
            "mapping_source_kind": "human_authored",
            "fallback_category": None,
            "categories": [
                {
                    "material_id": 0,
                    "category_name": "m3_floor_concrete",
                    "source_material_name": "floor",
                    "material_key": "concrete-key",
                    "rlr_material_name": "concrete",
                    "fallback": False,
                },
                {
                    "material_id": 1,
                    "category_name": "m3_wall_plaster",
                    "source_material_name": "wall",
                    "material_key": "plaster-key",
                    "rlr_material_name": "plaster",
                    "fallback": False,
                },
            ],
        },
    )
    database_path = acoustic / "material_database.json"
    write_json(
        database_path,
        {
            "materials": [
                {"name": "concrete", "labels": ["concrete"]},
                {"name": "plaster", "labels": ["plaster"]},
            ]
        },
    )
    qa_directory = root / "qa"
    qa_directory.mkdir()
    qa_records: dict[str, dict[str, Any]] = {}
    for name in (
        "geometry_report",
        "material_coverage",
        "ray_leakage",
        "visual_acoustic_parity",
    ):
        qa_path = qa_directory / f"{name}.json"
        write_json(qa_path, {"schema": f"unit_{name}_v1", "status": "pass"})
        qa_records[name] = {
            **file_record(qa_path, relative_to=root),
            "format": "json",
        }
    debug_path = qa_directory / "compiler_acoustic_mesh.obj"
    debug_path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "f 1 2 3\nf 1 3 4\n",
        encoding="utf-8",
    )
    identity = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    manifest = {
        "schema": "avengine_acoustic_scene_package_v1",
        "package_id": "unit-room-low",
        "package_mode": "production",
        "room_kind": "blender_custom",
        "source_room": {
            "room_id": "unit-room",
            "manifest_sha256": "11" * 32,
            "source_revision": "unit-v1",
            "geometry_asset_sha256": "22" * 32,
        },
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "-Z",
            "linear_unit": "meter",
            "quaternion_order": "xyzw",
        },
        "unit_scale_to_m": 1.0,
        "geometry": {
            "representation": "controlled_surface_mesh",
            "transform_policy": "baked_to_canonical_world",
            "index_space": "global_vertex_array",
            "source_to_canonical": {
                "matrix_row_major": identity,
                "source": "unit-test identity",
                "reviewed": True,
            },
            "vertex_count": 4,
            "triangle_count": 2,
            "source_primitive_count": 1,
            "source_node_instance_count": 1,
        },
        "arrays": {
            "vertices": _array_record(vertices_path, root=root, array=vertices),
            "triangles": _array_record(triangles_path, root=root, array=triangles),
            "triangle_material_ids": _array_record(
                material_ids_path, root=root, array=material_ids
            ),
        },
        "materials": {
            "mapping_id": "unit-mapping",
            "mapping_sha256": "33" * 32,
            "database_id": "unit-database",
            "database_source_sha256": "44" * 32,
            "category_count": 2,
            "categories": {
                **file_record(categories_path, relative_to=root),
                "format": "json",
            },
            "rlr_database": {
                **file_record(database_path, relative_to=root),
                "format": "json",
            },
        },
        "qa": qa_records,
        "debug_mesh": {
            **file_record(debug_path, relative_to=root),
            "format": "obj",
        },
        "compiler": {
            "name": "avengine.m3.compiler",
            "version": "1",
            "implementation_sha256": "55" * 32,
        },
        "objects": [
            {
                "object_id": "room",
                "source_node_index": 0,
                "source_mesh_index": 0,
                "source_primitive_index": 0,
                "source_material_name": "room",
                "vertex_offset": 0,
                "vertex_count": 4,
                "triangle_offset": 0,
                "triangle_count": 2,
                "world_from_object": identity,
                "source_world_matrix": identity,
                "transform_baked": True,
            }
        ],
    }
    manifest["package_content_sha256"] = canonical_json_sha256(manifest)
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def test_runtime_imports_quaternion_before_habitat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    quaternion = ModuleType("quaternion")
    habitat = ModuleType("habitat_sim")
    habitat.RLRContextConfiguration = object
    habitat.RLRAcousticContext = object
    habitat.RLRChannelLayoutType = object
    habitat.audio_enabled = True
    binding = ModuleType("habitat_sim._ext.habitat_sim_bindings")
    binding_path = tmp_path / "habitat_sim_bindings.unit.so"
    binding_path.write_bytes(b"unit binding")
    (tmp_path / "libRLRAudioPropagation.so").write_bytes(b"unit rlr")
    binding.__file__ = str(binding_path)

    def import_module(name: str) -> ModuleType:
        order.append(name)
        if name == "quaternion":
            return quaternion
        if name == "habitat_sim":
            return habitat
        return binding

    monkeypatch.setattr(runtime_module.importlib, "import_module", import_module)

    observed, report = load_habitat_runtime()

    assert observed is habitat
    assert order[:2] == ["quaternion", "habitat_sim"]
    assert order[2] == "habitat_sim._ext.habitat_sim_bindings"
    assert report["import_workaround"]["required_import_order"] == order[:2]
    assert report["native_binaries"]["habitat_sim_bindings"]["sha256"]


def test_missing_quaternion_never_attempts_unsafe_habitat_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    def import_module(name: str) -> ModuleType:
        order.append(name)
        raise ImportError(name)

    monkeypatch.setattr(runtime_module.importlib, "import_module", import_module)

    with pytest.raises(RuntimeUnavailableError, match="quaternion"):
        load_habitat_runtime()
    assert order == ["quaternion"]


def test_simulation_contract_requires_every_field_and_independent_repeats() -> None:
    complete = _simulation_mapping()
    config = RLRSimulationConfig.from_mapping(complete)

    assert config.sample_rate_hz == 16_000.0
    with pytest.raises(RuntimeContractError, match="missing explicit"):
        RLRSimulationConfig.from_mapping({key: value for key, value in complete.items() if key != "direct_ray_count"})
    with pytest.raises(RuntimeContractError, match="unknown"):
        RLRSimulationConfig.from_mapping({**complete, "secret_default": 1})
    with pytest.raises(RuntimeContractError, match="temporal_coherence"):
        RLRSimulationConfig.from_mapping({**complete, "temporal_coherence": True})
    with pytest.raises(RuntimeContractError, match="mesh_simplification"):
        RLRSimulationConfig.from_mapping({**complete, "mesh_simplification": True})
    with pytest.raises(RuntimeContractError, match="unit_scale"):
        RLRSimulationConfig.from_mapping({**complete, "unit_scale": 0.01})


def test_audio_disabled_build_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    quaternion = ModuleType("quaternion")
    habitat = ModuleType("habitat_sim")
    habitat.RLRContextConfiguration = object
    habitat.RLRAcousticContext = object
    habitat.RLRChannelLayoutType = object
    habitat.audio_enabled = False
    monkeypatch.setattr(
        runtime_module.importlib,
        "import_module",
        lambda name: quaternion if name == "quaternion" else habitat,
    )

    with pytest.raises(RuntimeUnavailableError, match="HABITAT_WITH_AUDIO"):
        load_habitat_runtime()


def test_compiled_package_is_hash_checked_and_rebased(tmp_path: Path) -> None:
    scene = load_compiled_acoustic_scene(_compiled_package(tmp_path))

    assert scene.package_id == "runtime-unit-package"
    assert len(scene.material_categories) == scene.manifest["materials"]["category_count"]
    assert scene.objects[0]["vertices"].dtype == np.dtype("<f4")
    assert scene.objects[0]["triangles"].dtype == np.dtype("<u4")
    assert int(scene.objects[0]["triangles"].min()) >= 0
    assert int(scene.objects[0]["triangles"].max()) < len(
        scene.objects[0]["vertices"]
    )
    assert len(scene.objects[0]["triangle_material_ids"]) == len(
        scene.objects[0]["triangles"]
    )


def test_only_explicit_unqualified_research_package_can_retain_failed_qa(
    tmp_path: Path,
) -> None:
    manifest_path = _compiled_package(tmp_path)
    validated = load_and_validate_acoustic_scene_package(manifest_path)
    failed_reports = deepcopy(validated.qa_reports)
    failed_reports["geometry_report"]["status"] = "fail"
    research_manifest = deepcopy(validated.manifest)
    research_manifest["package_mode"] = "research_candidate"
    research_manifest["materials"]["material_semantics"] = "research_placeholder"
    research_manifest["materials"]["qualification_claim"] = (
        "unqualified_research_placeholder"
    )
    research = replace(
        validated,
        manifest=research_manifest,
        qa_reports=failed_reports,
    )

    with pytest.raises(RuntimeContractError, match="QA is not pass"):
        load_compiled_acoustic_scene(
            manifest_path,
            validated_package=research,
        )
    scene = load_compiled_acoustic_scene(
        manifest_path,
        validated_package=research,
        allow_nonpassing_research_qa=True,
    )
    assert scene.qa_reports["geometry_report"]["status"] == "fail"

    production = replace(
        research,
        manifest=deepcopy(validated.manifest),
    )
    with pytest.raises(RuntimeContractError, match="QA is not pass"):
        load_compiled_acoustic_scene(
            manifest_path,
            validated_package=production,
            allow_nonpassing_research_qa=True,
        )

    with pytest.raises(RuntimeContractError, match="explicit boolean"):
        load_compiled_acoustic_scene(
            manifest_path,
            allow_nonpassing_research_qa=1,  # type: ignore[arg-type]
        )


def test_compiled_package_rejects_tampered_array(tmp_path: Path) -> None:
    manifest_path = _compiled_package(tmp_path)
    vertices_path = manifest_path.parent / "acoustic/vertices.npy"
    vertices_path.write_bytes(vertices_path.read_bytes() + b"tamper")

    with pytest.raises(RuntimeContractError, match="byte_size"):
        load_compiled_acoustic_scene(manifest_path)


def test_compiled_package_rejects_object_partition_escape(tmp_path: Path) -> None:
    manifest_path = _compiled_package(tmp_path)
    manifest = load_json(manifest_path)
    manifest["objects"][0]["vertex_offset"] = 1
    manifest.pop("package_content_sha256")
    manifest["package_content_sha256"] = canonical_json_sha256(manifest)
    write_json(manifest_path, manifest)

    with pytest.raises(RuntimeContractError, match="contiguous"):
        load_compiled_acoustic_scene(manifest_path)


class _FakeConfiguration:
    def __init__(self) -> None:
        for native_name in runtime_module._NATIVE_CONFIG_FIELDS.values():
            setattr(self, native_name, None)


class _FakeContext:
    owned_samples = np.asarray([[1.0, 0.4, 0.2, 0.1]], dtype="<f4")

    def __init__(self, config: _FakeConfiguration) -> None:
        self.config = config
        self.objects: list[dict[str, Any]] = []
        self.categories: list[str] = []
        self.upload_expected: dict[str, Any] = {}
        self.database_document: dict[str, Any] = {}

    def load_acoustic_scene(
        self, database: str, categories: list[str], objects: list[dict[str, Any]]
    ) -> SimpleNamespace:
        assert Path(database).is_file()
        self.objects = objects
        self.categories = list(categories)
        counts = {
            category: sum(
                int(np.count_nonzero(item["triangle_material_ids"] == index))
                for item in objects
            )
            for index, category in enumerate(categories)
        }
        database_bytes = Path(database).read_bytes()
        self.database_document = json.loads(database_bytes)
        expected = runtime_module._expected_upload_report(
            SimpleNamespace(
                objects=tuple(objects),
                material_categories=tuple(categories),
                triangle_count_by_material=counts,
                rlr_material_database=self.database_document,
                material_database_bytes=database_bytes,
            )
        )
        expected["material_upload_receipts"] = [
            SimpleNamespace(**receipt)
            for receipt in expected["material_upload_receipts"]
        ]
        self.upload_expected = expected
        return SimpleNamespace(**expected)

    def write_scene_mesh_obj(self, path: str) -> SimpleNamespace:
        lines: list[str] = ["# Objects"]
        offset = 0
        for item in self.objects:
            used_material_ids = [
                material_id
                for material_id in range(len(self.categories))
                if np.any(item["triangle_material_ids"] == material_id)
            ]
            for block_index, material_id in enumerate(used_material_ids):
                lines.append(f"# Material Index : {block_index}")
                category = self.categories[material_id]
                material = next(
                    value
                    for value in self.database_document["materials"]
                    if category in value["labels"]
                )
                for coefficient_name in (
                    "absorption",
                    "scattering",
                    "transmission",
                ):
                    display_name = coefficient_name.capitalize()
                    for coefficient_index, value in enumerate(
                        material[coefficient_name][1::2]
                    ):
                        lines.append(
                            f"# {display_name} - Index:{coefficient_index}, "
                            f"Value: {float(value):.6f}"
                        )
            lines.extend(
                [
                    f"# Vertex Count: {len(item['vertices'])}",
                    f"# Triangle Count: {len(item['triangles'])}",
                    f"# Material Count: {len(item['triangles'])}",
                ]
            )
            for vertex in item["vertices"]:
                lines.append(
                    "v " + " ".join(f"{float(value):.6f}" for value in vertex)
                )
            for triangle in item["triangles"]:
                indices = [int(value) + offset + 1 for value in triangle]
                lines.append("f " + " ".join(map(str, indices)))
            offset += len(item["vertices"])
        output = Path(path)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        parsed = runtime_module._parse_scene_obj(output)
        expected = runtime_module._expected_native_scene_readback_report(
            parsed,
            {
                **self.upload_expected,
                "material_upload_receipts": [
                    vars(item)
                    for item in self.upload_expected["material_upload_receipts"]
                ],
            },
        )
        return SimpleNamespace(output_path=str(output.resolve()), **expected)

    def add_source(self, *args: Any) -> int:
        return 0

    def add_listener(self, *args: Any) -> int:
        return 0

    def simulate_owned(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                listener_id="listener0",
                source_id="source0",
                sample_rate=16_000.0,
                channel_count=1,
                sample_count=4,
                samples=self.owned_samples,
            )
        ]

    def indirect_ray_efficiency(self) -> float:
        return 0.75

    def trace_ray_any_hit(
        self,
        origin: list[float],
        direction: list[float],
        minimum_distance: float,
        maximum_distance: float,
    ) -> SimpleNamespace:
        hit = runtime_module._cpu_first_hit_distance(
            tuple(self.objects),
            origin=np.asarray(origin, dtype=np.float64),
            direction=np.asarray(direction, dtype=np.float64),
            minimum_distance_m=minimum_distance,
            maximum_distance_m=maximum_distance,
        ) is not None
        return SimpleNamespace(
            hit=hit,
            has_hit_details=False,
            distance=0.0,
            normal=[0.0, 0.0, 0.0],
        )

    def trace_ray_first_hit(
        self,
        origin: list[float],
        direction: list[float],
        minimum_distance: float,
        maximum_distance: float,
    ) -> SimpleNamespace:
        distance = runtime_module._cpu_first_hit_distance(
            tuple(self.objects),
            origin=np.asarray(origin, dtype=np.float64),
            direction=np.asarray(direction, dtype=np.float64),
            minimum_distance_m=minimum_distance,
            maximum_distance_m=maximum_distance,
        )
        hit = distance is not None
        return SimpleNamespace(
            hit=hit,
            has_hit_details=hit,
            distance=float(distance) if hit else 0.0,
            normal=[0.0, 0.0, 15.75] if hit else [0.0, 0.0, 0.0],
        )


def test_native_ray_result_preserves_unnormalized_hit_normal_and_zero_sentinels() -> None:
    hit = runtime_module._native_ray_result(
        SimpleNamespace(
            hit=True,
            has_hit_details=True,
            distance=3.6,
            normal=[-7.2, 0.0, 0.0],
        ),
        first_hit=True,
    )
    assert hit["normal"] == [-7.2, 0.0, 0.0]

    with pytest.raises(RuntimeContractError, match="finite and non-zero"):
        runtime_module._native_ray_result(
            SimpleNamespace(
                hit=True,
                has_hit_details=True,
                distance=3.6,
                normal=[0.0, 0.0, 0.0],
            ),
            first_hit=True,
        )

    for first_hit, hit_flag in ((False, True), (True, False)):
        result = runtime_module._native_ray_result(
            SimpleNamespace(
                hit=hit_flag,
                has_hit_details=False,
                distance=0.0,
                normal=[0.0, 0.0, 0.0],
            ),
            first_hit=first_hit,
        )
        assert result["distance_m"] == 0.0
        assert result["normal"] == [0.0, 0.0, 0.0]

    with pytest.raises(RuntimeContractError, match="zero distance/normal sentinels"):
        runtime_module._native_ray_result(
            SimpleNamespace(
                hit=True,
                has_hit_details=False,
                distance=0.0,
                normal=[0.0, 1.0, 0.0],
            ),
            first_hit=False,
        )
    with pytest.raises(RuntimeContractError, match="zero distance/normal sentinels"):
        runtime_module._native_ray_result(
            SimpleNamespace(
                hit=False,
                has_hit_details=False,
                distance=0.1,
                normal=[0.0, 0.0, 0.0],
            ),
            first_hit=True,
        )


def test_modern_runtime_result_is_an_independent_ir_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    habitat = ModuleType("habitat_sim")
    habitat.RLRContextConfiguration = _FakeConfiguration
    habitat.RLRAcousticContext = _FakeContext
    habitat.RLRChannelLayoutType = SimpleNamespace(Mono="mono")
    habitat.audio_enabled = True
    monkeypatch.setattr(
        runtime_module,
        "load_habitat_runtime",
        lambda: (
            habitat,
            {
                "import_workaround": {
                    "required_import_order": ["quaternion", "habitat_sim"]
                }
            },
        ),
    )
    scene = load_compiled_acoustic_scene(_compiled_package(tmp_path))
    readback = tmp_path / "readback.obj"
    declared_rays = tuple(
        load_json(
            REPOSITORY / "examples/m1/rooms/blender_custom/room_manifest.json"
        )["ray_checks"]
    )

    result = simulate_compiled_acoustic_scene(
        scene,
        RLRSimulationConfig.from_mapping(_simulation_mapping()),
        source=RuntimeAnchor.from_mapping(
            {"id": "source0", "position_m": [0.0, 0.0, 0.0]}, listener=False
        ),
        listener=RuntimeAnchor.from_mapping(
            {"id": "listener0", "position_m": [1.0, 0.0, 0.0]}, listener=True
        ),
        scene_readback_obj=readback,
        ray_checks=declared_rays,
    )
    _FakeContext.owned_samples[0, 0] = 99.0

    np.testing.assert_array_equal(
        result.samples, np.asarray([[1.0, 0.4, 0.2, 0.1]], dtype="<f4")
    )
    assert result.upload_report["material_category_count"] == len(
        scene.material_categories
    )
    assert result.indirect_ray_efficiency == 0.75
    expected_readback = _simulation_mapping()
    expected_readback.pop("speed_of_sound_m_s")
    expected_readback.pop("channel_layout")
    assert result.runtime["configuration_readback"] == expected_readback
    assert result.runtime["scene_mesh_readback"]["sha256"]
    assert readback.is_file()
    assert len(result.ray_checks) == len(declared_rays)
    assert all(report["passed"] for report in result.ray_checks)
    assert all(report["cpu_rlr_hit_consistent"] for report in result.ray_checks)


def _write_readback_fixture(
    path: Path,
    *,
    first_vertex_x: str = "0.000000",
    first_face: str = "1 2 3",
    first_absorption: str = "0.050000",
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Objects",
                "# Material Index : 0",
                f"# Absorption - Index:0, Value: {first_absorption}",
                "# Scattering - Index:0, Value: 0.100000",
                "# Transmission - Index:0, Value: 0.000000",
                "# Material Index : 1",
                "# Absorption - Index:0, Value: 0.850000",
                "# Scattering - Index:0, Value: 0.350000",
                "# Transmission - Index:0, Value: 0.000000",
                "# Vertex Count: 4",
                "# Triangle Count: 2",
                "# Material Count: 2",
                f"v {first_vertex_x} 0.000000 0.000000",
                "v 1.000000 0.000000 0.000000",
                "v 1.000000 1.000000 0.000000",
                "v 0.000000 1.000000 0.000000",
                f"f {first_face}",
                "f 1 3 4",
                "# Listeners",
                "# Sources",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_readback_parser_models_multi_material_object_and_detects_tamper(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.obj"
    _write_readback_fixture(original)
    observed = runtime_module._parse_scene_obj(original)
    assert observed["vertex_count"] == 4
    assert observed["triangle_count"] == 2
    assert observed["material_block_count"] == 2
    assert observed["material_assignment_count"] == 2

    coefficient_tamper = tmp_path / "coefficient.obj"
    _write_readback_fixture(coefficient_tamper, first_absorption="0.050001")
    assert (
        runtime_module._parse_scene_obj(coefficient_tamper)[
            "material_coefficient_sha256"
        ]
        != observed["material_coefficient_sha256"]
    )

    coordinate_tamper = tmp_path / "coordinate.obj"
    _write_readback_fixture(coordinate_tamper, first_vertex_x="0.000001")
    assert (
        runtime_module._parse_scene_obj(coordinate_tamper)[
            "vertex_coordinate_multiset_sha256"
        ]
        != observed["vertex_coordinate_multiset_sha256"]
    )

    winding_tamper = tmp_path / "winding.obj"
    _write_readback_fixture(winding_tamper, first_face="1 3 2")
    assert (
        runtime_module._parse_scene_obj(winding_tamper)[
            "triangle_coordinate_multiset_sha256"
        ]
        != observed["triangle_coordinate_multiset_sha256"]
    )


def test_readback_parser_rejects_malformed_coefficients_and_cross_object_faces(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.obj"
    _write_readback_fixture(malformed)
    malformed.write_text(
        malformed.read_text(encoding="utf-8").replace(
            "Value: 0.050000", "Value: 0.05", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeContractError, match="malformed material coefficient"):
        runtime_module._parse_scene_obj(malformed)

    first = tmp_path / "first.obj"
    _write_readback_fixture(first)
    payload = first.read_text(encoding="utf-8").replace(
        "# Listeners\n# Sources\n", ""
    )
    second = "\n".join(
        [
            "# Material Index : 0",
            "# Absorption - Index:0, Value: 0.100000",
            "# Scattering - Index:0, Value: 0.100000",
            "# Transmission - Index:0, Value: 0.000000",
            "# Vertex Count: 3",
            "# Triangle Count: 1",
            "# Material Count: 1",
            "v 2.000000 0.000000 0.000000",
            "v 3.000000 0.000000 0.000000",
            "v 2.000000 1.000000 0.000000",
            "f 1 2 3",
        ]
    )
    escaping = tmp_path / "escaping.obj"
    escaping.write_text(payload + second + "\n", encoding="utf-8")
    with pytest.raises(RuntimeContractError, match="escape their object block"):
        runtime_module._parse_scene_obj(escaping)

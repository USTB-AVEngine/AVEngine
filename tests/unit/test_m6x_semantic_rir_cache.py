from __future__ import annotations

from copy import deepcopy
import json
from importlib.machinery import ExtensionFileLoader
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from avengine.contracts.json_io import write_json
from avengine.m4.runtime import M4SimulationConfig
from avengine.m6x import rir_cache


def _full_plan() -> dict[str, object]:
    return {
        "schema": "avengine_room_rir_job_plan_v2",
        "status": "planned_not_run",
        "listener_pose_mode": "per_episode_frame",
        "cache_key_fields": [
            "source_position_m",
            "listener_position_m",
            "listener_orientation_wxyz",
        ],
        "stride_frames": 1,
        "requested_pair_state_count": 150,
        "unique_rir_job_count": 2,
        "jobs": [
            {
                "job_id": f"job_{slot}",
                "source_position_m": [float(index + 1), 1.5, 2.0],
                "listener_position_m": [0.0, 1.5, 0.0],
                "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "uses": [
                    {
                        "episode_id": "episode",
                        "source_slot_id": slot,
                        "frame_index": frame,
                    }
                    for frame in range(75)
                ],
            }
            for index, slot in enumerate(("source1", "source2"))
        ],
    }


def test_semantic_plan_accepts_one_complete_episode_grid() -> None:
    jobs = rir_cache.validate_semantic_rir_job_plan(_full_plan())
    assert len(jobs) == 2
    assert sum(len(job["uses"]) for job in jobs) == 150


@pytest.mark.parametrize(
    "mutation",
    ["missing_source2", "missing_f74", "duplicate", "foreign", "stride", "bool_stride"],
)
def test_semantic_plan_rejects_incomplete_or_mixed_grid(mutation: str) -> None:
    plan = deepcopy(_full_plan())
    jobs = plan["jobs"]
    assert isinstance(jobs, list)
    if mutation == "missing_source2":
        jobs.pop()
        plan["unique_rir_job_count"] = 1
        plan["requested_pair_state_count"] = 75
    elif mutation == "missing_f74":
        jobs[1]["uses"].pop()
        plan["requested_pair_state_count"] = 149
    elif mutation == "duplicate":
        jobs[1]["uses"][74] = deepcopy(jobs[1]["uses"][73])
    elif mutation == "foreign":
        jobs[1]["uses"][74]["episode_id"] = "foreign"
    elif mutation == "stride":
        plan["stride_frames"] = 2
    else:
        plan["stride_frames"] = True
    with pytest.raises(rir_cache.RIRCacheError):
        rir_cache.validate_semantic_rir_job_plan(plan)


def _write_shard(path: Path, **overrides: np.ndarray) -> None:
    values = {
        "job_indices": np.asarray([0], dtype="<u4"),
        "job_ids": np.asarray(["job"]),
        "source_positions_m": np.asarray([[1.0, 1.5, 2.0]], dtype="<f8"),
        "listener_positions_m": np.asarray([[0.0, 1.5, 0.0]], dtype="<f8"),
        "listener_orientations_wxyz": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype="<f8"),
        "lengths": np.asarray([2], dtype="<u4"),
        "samples": np.asarray([[[1.0, 0.5], [0.25, 0.125]]], dtype="<f4"),
        "sample_rate_hz": np.asarray(16000, dtype="<u4"),
        "layout_id": np.asarray("rlr_binaural_lr_v1"),
        "channel_labels": np.asarray(["left", "right"]),
        "simulate_wall_seconds": np.asarray(0.1, dtype="<f8"),
        "simulate_process_cpu_seconds": np.asarray(0.1, dtype="<f8"),
        "indirect_ray_efficiency": np.asarray(0.5, dtype="<f8"),
    }
    values.update(overrides)
    np.savez(path, **values)


def test_semantic_shard_accepts_exact_types(tmp_path: Path) -> None:
    path = tmp_path / "shard.npz"
    _write_shard(path)
    assert rir_cache._read_semantic_shard(path)["samples"].dtype == np.dtype("<f4")


@pytest.mark.parametrize("drift", ["dtype", "channels", "orientation"])
def test_semantic_shard_rejects_structural_drift(tmp_path: Path, drift: str) -> None:
    path = tmp_path / "shard.npz"
    overrides = {}
    if drift == "dtype":
        overrides["samples"] = np.ones((1, 2, 2), dtype="<f8")
    elif drift == "channels":
        overrides["samples"] = np.ones((1, 1, 2), dtype="<f4")
        overrides["channel_labels"] = np.asarray(["left"])
    else:
        overrides["listener_orientations_wxyz"] = np.asarray(
            [[2.0, 0.0, 0.0, 0.0]], dtype="<f8"
        )
    _write_shard(path, **overrides)
    with pytest.raises(rir_cache.RIRCacheError):
        rir_cache._read_semantic_shard(path)


def _semantic_scene_package(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "scene"
    acoustic = root / "acoustic"
    acoustic.mkdir(parents=True)
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype="<f4",
    )
    triangles = np.asarray([[0, 1, 2]], dtype="<u4")
    material_ids = np.asarray([0], dtype="<u4")
    np.save(acoustic / "vertices.npy", vertices, allow_pickle=False)
    np.save(acoustic / "triangles.npy", triangles, allow_pickle=False)
    np.save(acoustic / "triangle_material_ids.npy", material_ids, allow_pickle=False)
    categories_path = acoustic / "categories.json"
    write_json(
        categories_path,
        {
            "schema": "avengine_acoustic_material_categories_v1",
            "mapping_id": "mapping",
            "room_id": "room",
            "mapping_source_kind": "semantic_fixture",
            "fallback_category": None,
            "categories": [
                {
                    "category_name": "wall",
                    "fallback": False,
                    "human_override": False,
                    "mapping_confidence": 1.0,
                    "mapping_source": "fixture",
                    "material_id": 0,
                    "material_key": "wall",
                    "randomized": False,
                    "rlr_match": "wall_material",
                    "rlr_material_name": "wall_material",
                    "source_material_name": "wall",
                }
            ],
        },
    )
    database_path = acoustic / "materials.json"
    write_json(
        database_path,
        {
            "materials": [
                {
                    "name": "wall_material",
                    "absorption": [100.0, 0.2],
                    "scattering": [100.0, 0.1],
                    "transmission": [100.0, 0.0],
                    "labels": ["wall"],
                    "damping": [100.0, 0.0],
                    "density": 1.2,
                    "speed": 343.0,
                }
            ]
        },
    )
    arrays = {
        "vertices": {
            "path": "acoustic/vertices.npy",
            "format": "npy",
            "dtype": "<f4",
            "shape": list(vertices.shape),
            "memory_order": "C",
        },
        "triangles": {
            "path": "acoustic/triangles.npy",
            "format": "npy",
            "dtype": "<u4",
            "shape": list(triangles.shape),
            "memory_order": "C",
        },
        "triangle_material_ids": {
            "path": "acoustic/triangle_material_ids.npy",
            "format": "npy",
            "dtype": "<u4",
            "shape": list(material_ids.shape),
            "memory_order": "C",
        },
    }
    manifest = root / "manifest.json"
    write_json(
        manifest,
        {
            "schema": "avengine_acoustic_scene_package_v1",
            "package_mode": "research_candidate",
            "package_id": "fixture_scene",
            "arrays": arrays,
            "materials": {
                "mapping_id": "mapping",
                "mapping_source_kind": "semantic_fixture",
                "category_count": 1,
                "categories": {"path": "acoustic/categories.json"},
                "rlr_database": {"path": "acoustic/materials.json"},
            },
            "geometry": {
                "vertex_count": 3,
                "triangle_count": 1,
                "index_space": "global_vertex_array",
                "transform_policy": "baked_to_canonical_world",
            },
            "objects": [
                {
                    "object_id": "room",
                    "vertex_offset": 0,
                    "vertex_count": 3,
                    "triangle_offset": 0,
                    "triangle_count": 1,
                }
            ],
        },
    )
    return manifest, categories_path, database_path


@pytest.mark.parametrize(
    "drift", ["category_schema", "mapping_id", "material_name", "material_curve"]
)
def test_semantic_scene_rejects_material_structure_drift(
    tmp_path: Path, drift: str
) -> None:
    manifest, categories_path, database_path = _semantic_scene_package(tmp_path)
    if drift == "material_curve":
        document = json.loads(database_path.read_text())
        document["materials"][0]["absorption"] = [100.0, 1.5]
        write_json(database_path, document)
    else:
        document = json.loads(categories_path.read_text())
        if drift == "category_schema":
            document["schema"] = "legacy_categories"
        elif drift == "mapping_id":
            document["mapping_id"] = "foreign_mapping"
        else:
            document["categories"][0]["rlr_material_name"] = "wrong_material"
        write_json(categories_path, document)
    with pytest.raises(rir_cache.RIRCacheError):
        rir_cache.load_semantic_acoustic_scene(manifest)


def test_semantic_scene_loads_structural_fixture(tmp_path: Path) -> None:
    manifest, _categories, _database = _semantic_scene_package(tmp_path)
    scene = rir_cache.load_semantic_acoustic_scene(manifest)
    assert scene.package_id == "fixture_scene"
    assert scene.material_name_by_category == {"wall": "wall_material"}
    assert scene.triangle_count_by_material == {"wall": 1}


def _semantic_runtime_modules(tmp_path: Path) -> dict[str, ModuleType]:
    package = tmp_path / "src_python/habitat_sim"
    binding_dir = (
        tmp_path / "build/cp312-cp312-linux_x86_64/install/platlib/habitat_sim/_ext"
    )
    package.mkdir(parents=True)
    binding_dir.mkdir(parents=True)
    habitat_file = package / "__init__.py"
    binding_file = binding_dir / "habitat_sim_bindings.cpython-312-x86_64-linux-gnu.so"
    quaternion_file = tmp_path / "quaternion.py"
    habitat_file.write_text("")
    binding_file.write_bytes(b"compiled-fixture")
    (binding_dir / "libRLRAudioPropagation.so").write_bytes(b"native-fixture")
    quaternion_file.write_text("")

    binding_name = "habitat_sim._ext.habitat_sim_bindings"
    habitat = ModuleType("habitat_sim")
    habitat.__file__ = str(habitat_file)
    habitat.__path__ = [str(package), str(binding_dir.parent)]
    habitat.__spec__ = SimpleNamespace(
        name="habitat_sim",
        origin=str(habitat_file),
        submodule_search_locations=habitat.__path__,
    )
    binding = ModuleType(binding_name)
    binding.__file__ = str(binding_file)
    binding.__spec__ = SimpleNamespace(
        name=binding_name,
        origin=str(binding_file),
        parent="habitat_sim._ext",
        loader=ExtensionFileLoader(binding_name, str(binding_file)),
    )
    quaternion = ModuleType("quaternion")
    quaternion.__file__ = str(quaternion_file)
    for name in (
        "RLRContextConfiguration",
        "RLRAcousticContext",
        "RLRChannelLayoutType",
    ):
        symbol = type(name, (), {"__module__": binding_name})
        setattr(binding, name, symbol)
        setattr(habitat, name, symbol)
    habitat.audio_enabled = True
    return {
        "quaternion": quaternion,
        "habitat_sim": habitat,
        binding_name: binding,
    }


def test_semantic_runtime_accepts_editable_split_module_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modules = _semantic_runtime_modules(tmp_path)
    monkeypatch.setattr(rir_cache.importlib, "import_module", modules.__getitem__)
    habitat, binding, report = rir_cache._load_semantic_habitat_runtime()
    assert habitat is modules["habitat_sim"]
    assert binding is modules["habitat_sim._ext.habitat_sim_bindings"]
    assert report["habitat_module_path"].endswith("src_python/habitat_sim/__init__.py")
    assert "/install/platlib/habitat_sim/_ext/" in report["binding_module_path"]


@pytest.mark.parametrize(
    "drift", ["binding_origin", "binding_loader", "package_search", "symbol_identity"]
)
def test_semantic_runtime_rejects_unrelated_import_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    modules = _semantic_runtime_modules(tmp_path)
    habitat = modules["habitat_sim"]
    binding = modules["habitat_sim._ext.habitat_sim_bindings"]
    if drift == "binding_origin":
        binding.__spec__.origin = str(tmp_path / "unrelated.so")
    elif drift == "binding_loader":
        binding.__spec__.loader = object()
    elif drift == "package_search":
        habitat.__spec__.submodule_search_locations = [str(tmp_path / "build")]
    else:
        habitat.RLRAcousticContext = type(
            "RLRAcousticContext",
            (),
            {"__module__": "habitat_sim._ext.habitat_sim_bindings"},
        )
    monkeypatch.setattr(rir_cache.importlib, "import_module", modules.__getitem__)
    with pytest.raises(rir_cache.RIRCacheError):
        rir_cache._load_semantic_habitat_runtime()


class _Renderer:
    def __init__(self, *_args, **_kwargs):
        self.setup_report = {
            "schema": "semantic_fixture_setup_v1",
            "qualification_claim": False,
        }

    def render(self, positions, **_kwargs):
        samples = tuple(
            np.asarray([[1.0, 0.5], [0.25, 0.125]], dtype="<f4") for _ in positions
        )
        return rir_cache.RIRBatchResult(
            samples=samples,
            sample_rate_hz=16_000,
            layout_id="rlr_binaural_lr_v1",
            channel_labels=("left", "right"),
            indirect_ray_efficiency=0.5,
            wall_seconds=0.25,
            process_cpu_seconds=0.2,
        )


def _producer_inputs(tmp_path: Path):
    plan = tmp_path / "plan.json"
    simulation_path = tmp_path / "simulation.json"
    hrtf = tmp_path / "fixture.sofa"
    write_json(plan, _full_plan())
    simulation_document = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/runtime/rir_cache_simulation_request_v2.json"
        ).read_text()
    )
    write_json(simulation_path, simulation_document)
    hrtf.write_bytes(b"fixture")
    scene = rir_cache.SemanticAcousticScene(
        manifest_path=tmp_path / "manifest.json",
        package_id="room",
        material_database_bytes=b"{}",
        material_categories=("wall",),
        material_name_by_category={"wall": "wall"},
        material_index_by_category={"wall": 0},
        objects=(),
        triangle_count_by_material={"wall": 0},
    )
    simulation = M4SimulationConfig.from_mapping(simulation_document["simulation"])
    return plan, simulation_path, hrtf, scene, simulation


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for item in value.values() for key in _nested_keys(item)
        }
    if isinstance(value, (list, tuple)):
        return {key for item in value for key in _nested_keys(item)}
    return set()


def test_semantic_writer_atomically_publishes_fake_cache_without_native_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, simulation_path, hrtf, scene, simulation = _producer_inputs(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("semantic writer entered a file-evidence helper")

    monkeypatch.setattr(rir_cache, "sha256_file", forbidden)
    monkeypatch.setattr(rir_cache, "canonical_json_sha256", forbidden)
    result = rir_cache.render_semantic_rir_cache(
        plan_path=plan,
        scene=scene,
        simulation_request_path=simulation_path,
        simulation=simulation,
        output=tmp_path / "cache",
        hrtf_file_path=hrtf,
        batch_size=2,
        renderer_factory=_Renderer,
    )
    assert result.output == tmp_path / "cache"
    assert result.receipt["native_execution"] is False
    assert result.receipt["native_simulate_owned_call_count"] == 0
    assert result.receipt["producer_backend"] == "test_only_injected_renderer"
    forbidden_keys = {"sha256", "file_sha256", "byte_size", "retained_shard_bytes"}
    for path in result.output.glob("*.json"):
        assert not (_nested_keys(json.loads(path.read_text())) & forbidden_keys)
    with np.load(result.output / "shards/shard_000000.npz") as archive:
        assert "ir_sha256" not in archive.files
        assert "acoustic_state_sha256" not in archive.files


@pytest.mark.parametrize(
    "failure", ["preexisting", "ambisonics", "missing_hrtf", "compressed_int"]
)
def test_semantic_writer_rejects_unsafe_or_unsupported_request(
    tmp_path: Path, failure: str
) -> None:
    plan, simulation_path, hrtf, scene, simulation = _producer_inputs(tmp_path)
    output = tmp_path / "cache"
    kwargs = {"layout_type": "binaural", "hrtf_file_path": hrtf}
    if failure == "preexisting":
        output.mkdir()
    elif failure == "ambisonics":
        kwargs["layout_type"] = "ambisonics"
    elif failure == "missing_hrtf":
        kwargs["hrtf_file_path"] = None
    else:
        kwargs["compressed"] = 1
    with pytest.raises(rir_cache.RIRCacheError):
        rir_cache.render_semantic_rir_cache(
            plan_path=plan,
            scene=scene,
            simulation_request_path=simulation_path,
            simulation=simulation,
            output=output,
            batch_size=2,
            renderer_factory=_Renderer,
            **kwargs,
        )
    if failure != "preexisting":
        assert not output.exists()

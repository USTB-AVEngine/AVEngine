from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avengine.spatial_audio.audio import read_float32_wav, write_float32_wav
from avengine.m6x import rir_cache
from avengine.m6x.rir_cache import validate_semantic_rir_job_plan
from avengine.m6x.semantic_rir_cache import SemanticRIRCacheSession
from avengine.optional_backends.residential_episode import (
    DOG_SOURCE_ID,
    HUMAN_SOURCE_ID,
    PROFILE_SCHEMA,
    SCENE_METADATA_SCHEMA,
    build_residential_source_episode,
)


m6z = importlib.import_module("tools.m6z.build_residential_source_episode")


def _metadata() -> dict[str, object]:
    return {
        "schema": SCENE_METADATA_SCHEMA,
        "dataset_id": "test/interioragent",
        "scene_id": "kujiale_test",
        "room_id": "kujiale_test_room",
        "room_polygon_xy_m": [[-2.0, -3.0], [2.0, -3.0], [2.0, 3.0], [-2.0, 3.0]],
        "objects": [],
        "claim_boundary": "test fixture",
    }


def _profile() -> dict[str, object]:
    return {
        "schema": PROFILE_SCHEMA,
        "backend_role": "production_visual",
        "scene_id": "kujiale_test",
        "map_path": "/Game/AVEngine/Test/Kujiale",
        "camera": {
            "position_xyz_m": [0.0, -2.5, 1.55],
            "yaw_ue_deg": 90.0,
            "horizontal_fov_deg": 105.0,
        },
        "routes": {
            "dog0": {
                "start_xyz_m": [-0.5, 2.0, 0.0],
                "end_xyz_m": [-0.5, -1.5, 0.0],
            },
            "human0": {
                "start_xyz_m": [0.5, -1.5, 0.064],
                "end_xyz_m": [0.5, 2.0, 0.064],
            },
        },
        "source_center_margin_m": 0.03,
        "emitter_heights_m": {"dog0": 0.45, "human0": 1.60},
        "review_lights": [],
        "acoustic_proxy": {
            "label": "legacy_test_proxy",
            "coordinate_translation_habitat_m": [0.0, 0.0, 0.6],
            "rir_stride_frames": 3,
        },
    }


def _episode() -> dict[str, object]:
    return build_residential_source_episode(
        scene_metadata=_metadata(),
        profile=_profile(),
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_args(tmp_path: Path, *, audio_mode: str) -> argparse.Namespace:
    scene_metadata = tmp_path / "scene_metadata.json"
    profile = tmp_path / "profile.json"
    _write_json(scene_metadata, _metadata())
    _write_json(profile, _profile())
    return argparse.Namespace(
        scene_metadata=scene_metadata,
        profile=profile,
        dry_root=tmp_path / "dry",
        acoustic_manifest=tmp_path / "proxy_manifest.json",
        simulation_request=tmp_path / "proxy_request.json",
        hrtf=tmp_path / "proxy.sofa",
        audio_mode=audio_mode,
        semantic_acoustic_manifest=tmp_path / "semantic_manifest.json",
        semantic_simulation_request=(
            m6z.REPOSITORY / "examples/runtime/rir_cache_simulation_request_v2.json"
        ),
        semantic_hrtf=tmp_path / "semantic.sofa",
        semantic_rir_batch_size=8,
        output=tmp_path / f"{audio_mode}_output",
    )


def _avoid_video(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        m6z,
        "_render_topdown_frames",
        lambda _episode, output: output / "topdown_only.mp4",
    )
    monkeypatch.setattr(m6z, "_probe", lambda _path: {"probe": "test"})


def _semantic_scene_package(tmp_path: Path) -> Path:
    root = tmp_path / "semantic_scene"
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
    _write_json(
        acoustic / "categories.json",
        {
            "schema": "avengine_acoustic_material_categories_v1",
            "mapping_id": "mapping",
            "room_id": "kujiale_test_room",
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
                    "rlr_label_normalization": {
                        "policy": (
                            "stable_case_insensitive_exact_duplicate_removal_v1"
                        ),
                        "removed_exact_duplicates": [],
                        "runtime_label_count": 1,
                        "source_label_count": 1,
                    },
                    "rlr_match": "wall_material",
                    "rlr_material_name": "wall_material",
                    "source_material_name": "wall",
                }
            ],
        },
    )
    _write_json(
        acoustic / "materials.json",
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
    manifest = root / "manifest.json"
    _write_json(
        manifest,
        {
            "schema": "avengine_acoustic_scene_package_v1",
            "package_mode": "research_candidate",
            "package_id": "kujiale_semantic_fixture",
            "arrays": {
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
            },
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
    return manifest


class _HermeticNativeRenderer:
    def __init__(self, scene: object, simulation: object, **_kwargs: object) -> None:
        config_fields = (
            "frequency_bands",
            "direct_sh_order",
            "indirect_sh_order",
            "direct_ray_count",
            "indirect_ray_count",
            "indirect_ray_depth",
            "source_ray_count",
            "source_ray_depth",
            "max_diffraction_order",
            "thread_count",
            "sample_rate_hz",
            "max_ir_seconds",
            "unit_scale",
            "global_volume",
            "direct",
            "indirect",
            "diffraction",
            "transmission",
            "mesh_simplification",
            "temporal_coherence",
        )
        categories = tuple(scene.material_categories)
        triangle_count = sum(scene.triangle_count_by_material.values())
        self.setup_report = {
            "schema": "avengine_semantic_native_rir_setup_v1",
            "runtime": {
                "schema": "avengine_semantic_habitat_rlr_runtime_v1",
                "binding_api": "habitat_sim.RLRAcousticContext_v1",
                "quaternion_module_path": "/fixture/quaternion.py",
                "habitat_module_path": "/fixture/src_python/habitat_sim/__init__.py",
                "binding_module_path": (
                    "/fixture/build/install/platlib/habitat_sim/_ext/"
                    "habitat_sim_bindings.cpython-312-x86_64-linux-gnu.so"
                ),
                "rlr_library_path": (
                    "/fixture/build/install/platlib/habitat_sim/_ext/"
                    "libRLRAudioPropagation.so"
                ),
            },
            "configuration_readback": {
                name: getattr(simulation, name) for name in config_fields
            },
            "compute_device": "CPU",
            "qualification_claim": False,
            "upload": {
                "status": "pass_structural_native_upload",
                "object_count": len(scene.objects),
                "vertex_count": 3,
                "triangle_count": triangle_count,
                "material_category_count": len(categories),
                "object_ids": [item["object_id"] for item in scene.objects],
                "triangle_count_by_material": dict(scene.triangle_count_by_material),
                "material_upload_call_count": {category: 1 for category in categories},
                "resolved_material_name_by_category": dict(
                    scene.material_name_by_category
                ),
                "resolved_material_index_by_category": dict(
                    scene.material_index_by_category
                ),
            },
            "wall_seconds": 0.0,
            "process_cpu_seconds": 0.0,
        }
        self.native_simulate_owned_call_count = 0
        self.native_realized_job_count = 0

    def render(
        self, positions: list[list[float]], **_kwargs: object
    ) -> rir_cache.RIRBatchResult:
        self.native_simulate_owned_call_count += 1
        self.native_realized_job_count += len(positions)
        impulse = np.asarray([[0.01, 0.005], [0.008, 0.004]], dtype="<f4")
        return rir_cache.RIRBatchResult(
            samples=tuple(impulse.copy() for _ in positions),
            sample_rate_hz=16_000,
            layout_id="rlr_binaural_lr_v1",
            channel_labels=("left", "right"),
            indirect_ray_efficiency=0.5,
            wall_seconds=0.0,
            process_cpu_seconds=0.0,
        )


def test_semantic_plan_maps_full75_human_and_dog_source_centers() -> None:
    episode = _episode()

    plan = m6z._build_semantic_rir_job_plan(episode)
    jobs = validate_semantic_rir_job_plan(plan)

    assert plan["listener_pose_mode"] == "fixed"
    assert plan["stride_frames"] == 1
    assert plan["requested_pair_state_count"] == 150
    assert plan["unique_rir_job_count"] == 150
    assert len(jobs) == 150

    source_id_by_slot = {"source1": HUMAN_SOURCE_ID, "source2": DOG_SOURCE_ID}
    observed_uses: set[tuple[str, int]] = set()
    for job in jobs:
        for use in job["uses"]:
            slot = use["source_slot_id"]
            frame = use["frame_index"]
            source_id = source_id_by_slot[slot]
            observed_uses.add((slot, frame))
            assert job["source_position_m"] == pytest.approx(
                episode["source_trajectories_habitat_m"][source_id][frame]
            )
            assert job["listener_position_m"] == pytest.approx(
                episode["qualification"]["listener"]["position_m"]
            )
            assert job["listener_orientation_wxyz"] == pytest.approx(
                episode["qualification"]["listener"]["orientation_wxyz"]
            )
    assert observed_uses == {
        (slot, frame) for slot in ("source1", "source2") for frame in range(75)
    }


def test_build_defaults_to_unchanged_review_proxy_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _build_args(tmp_path, audio_mode="review_proxy")
    _avoid_video(monkeypatch)
    seen: dict[str, object] = {}

    def render_proxy(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"mode": "review_proxy"}

    def unexpected_semantic(**_kwargs: object) -> None:
        raise AssertionError("default branch entered semantic cached audio")

    monkeypatch.setattr(m6z, "_render_audio", render_proxy)
    monkeypatch.setattr(m6z, "_render_semantic_cached_audio", unexpected_semantic)

    result = m6z.build(args)

    assert seen["acoustic_manifest"] == args.acoustic_manifest
    assert seen["simulation_request"] == args.simulation_request
    assert seen["hrtf"] == args.hrtf
    assert seen["dry_root"] == args.dry_root
    assert seen["output"] == args.output.resolve()
    assert result["audio"] == {"mode": "review_proxy"}
    assert (args.output / "episode_plan.json").is_file()


def test_build_dispatches_semantic_cached_audio_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _build_args(tmp_path, audio_mode="semantic_cached_rlr")
    _avoid_video(monkeypatch)
    seen: dict[str, object] = {}

    def render_semantic(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"mode": "semantic_cached_rlr"}

    def unexpected_proxy(**_kwargs: object) -> None:
        raise AssertionError("semantic branch entered review proxy audio")

    monkeypatch.setattr(m6z, "_render_audio", unexpected_proxy)
    monkeypatch.setattr(m6z, "_render_semantic_cached_audio", render_semantic)

    result = m6z.build(args)

    assert set(seen) == {
        "episode",
        "dry_root",
        "acoustic_manifest",
        "simulation_request",
        "hrtf",
        "rir_batch_size",
        "output",
    }
    assert seen["dry_root"] == args.dry_root
    assert seen["acoustic_manifest"] == args.semantic_acoustic_manifest
    assert seen["simulation_request"] == args.semantic_simulation_request
    assert seen["hrtf"] == args.semantic_hrtf
    assert seen["rir_batch_size"] == 8
    assert seen["output"] == args.output.resolve()
    assert result["audio"] == {"mode": "semantic_cached_rlr"}


def test_build_semantic_branch_requires_all_explicit_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _build_args(tmp_path, audio_mode="semantic_cached_rlr")
    args.semantic_hrtf = None
    _avoid_video(monkeypatch)

    with pytest.raises(RuntimeError, match="--semantic-hrtf"):
        m6z.build(args)


def test_semantic_branch_runs_real_scene_reader_cache_reader_and_m7_assembler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _episode()
    dry_root = tmp_path / "dry"
    dry_root.mkdir()
    dog_dry = np.full((1, 80_000), 0.01, dtype=np.float32)
    human_dry = np.full((1, 80_000), 0.02, dtype=np.float32)
    write_float32_wav(
        dry_root / "m6x_dog0_muzzle.wav",
        dog_dry,
        16_000,
        metadata={"role": "test"},
    )
    write_float32_wav(
        dry_root / "m6x_human0_mouth.wav",
        human_dry,
        16_000,
        metadata={"role": "test"},
    )
    manifest = _semantic_scene_package(tmp_path)
    hrtf = tmp_path / "fixture.sofa"
    hrtf.write_bytes(b"fixture")
    output = tmp_path / "semantic_output"
    output.mkdir()
    request = m6z.REPOSITORY / "examples/runtime/rir_cache_simulation_request_v2.json"
    calls: dict[str, Any] = {}
    real_writer = rir_cache.render_semantic_rir_cache

    def publish_cache(**kwargs: Any) -> rir_cache.RIRCacheResult:
        calls["cache"] = kwargs
        monkeypatch.setattr(
            rir_cache,
            "_SemanticNativeRIRBatchRenderer",
            _HermeticNativeRenderer,
        )
        return real_writer(
            **kwargs,
            renderer_factory=_HermeticNativeRenderer,
        )

    monkeypatch.setattr(m6z, "render_semantic_rir_cache", publish_cache)

    evidence = m6z._render_semantic_cached_audio(
        episode=episode,
        dry_root=dry_root,
        acoustic_manifest=manifest,
        simulation_request=request,
        hrtf=hrtf,
        rir_batch_size=8,
        output=output,
    )

    cache_args = calls["cache"]
    assert cache_args["plan_path"] == output / "rir_job_plan.json"
    assert cache_args["output"] == output / "rir_cache"
    assert cache_args["scene"].package_id == "kujiale_semantic_fixture"
    assert cache_args["scene"].manifest_path == manifest.resolve()
    assert cache_args["simulation_request_path"] == request
    assert cache_args["hrtf_file_path"] == hrtf
    assert cache_args["batch_size"] == 8
    assert cache_args["coordinate_translation_m"] == (0.0, 0.0, 0.0)
    assert cache_args["layout_type"] == "binaural"

    plan_path = output / "rir_job_plan.json"
    plan = json.loads(plan_path.read_text())
    assert len(validate_semantic_rir_job_plan(plan)) == 150
    cached = SemanticRIRCacheSession(
        cache_root=output / "rir_cache",
        plan_path=plan_path,
        expected_episode_id=m6z.SEMANTIC_RIR_EPISODE_ID,
        frame_count=75,
        frame_rate_hz=15,
    ).load_episode(m6z.SEMANTIC_RIR_EPISODE_ID)
    assert cached.samples.shape == (75, 2, 2, 2)
    assert cached.lengths.shape == (75, 2)
    assert cached.source_slot_ids == ("source1", "source2")
    assert cached.visual_frame_indices == tuple(range(75))

    dog_path = output / "audio" / f"{DOG_SOURCE_ID}_stem.wav"
    human_path = output / "audio" / f"{HUMAN_SOURCE_ID}_stem.wav"
    mixture_path = output / "audio" / "mixture.wav"
    dog = read_float32_wav(dog_path)
    human = read_float32_wav(human_path)
    mixture = read_float32_wav(mixture_path)
    assert (
        dog.samples.shape
        == human.samples.shape
        == mixture.samples.shape
        == (
            2,
            80_000,
        )
    )
    assert np.max(np.abs(human.samples)) > np.max(np.abs(dog.samples)) > 0.0
    np.testing.assert_array_equal(mixture.samples, human.samples + dog.samples)

    assert evidence["audio_mode"] == "semantic_cached_rlr"
    assert evidence["source_slots_by_source_id"] == {
        HUMAN_SOURCE_ID: "source1",
        DOG_SOURCE_ID: "source2",
    }
    assert evidence["rir_stride_frames"] == 1
    assert evidence["requested_pair_state_count"] == 150
    assert evidence["unique_rir_job_count"] == 150
    assert evidence["rir_cache"] == {
        "path": str(output / "rir_cache"),
        "native_execution": True,
        "native_realized_job_count": 150,
        "native_simulate_owned_call_count": 19,
    }
    assert evidence["coordinate_translation_habitat_m"] == [0.0, 0.0, 0.0]

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m3.runtime import CompiledAcousticScene
from avengine.m4.runtime import M4SimulationConfig
from avengine.m6x.rir_cache import (
    RIRBatchResult,
    RIRCacheError,
    RIRCacheSession,
    _NativeRIRBatchRenderer,
    load_cached_rir_episode,
    render_rir_cache,
    validate_rir_job_plan,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _plan(job_count: int = 3) -> dict[str, object]:
    jobs = [
        {
            "job_id": f"rir_{index:06d}",
            "source_position_m": [float(index + 1), 1.0, 2.0],
            "uses": [
                {
                    "episode_id": f"episode_{index:03d}",
                    "source_slot_id": "source1" if index % 2 == 0 else "source2",
                    "frame_index": 0,
                }
            ],
        }
        for index in range(job_count)
    ]
    return {
        "schema": "avengine_room_rir_job_plan_v2",
        "status": "planned_not_run",
        "listener_position_m": [0.0, 1.5, 0.0],
        "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "unique_rir_job_count": len(jobs),
        "jobs": jobs,
    }


def _episode_plan() -> dict[str, object]:
    jobs = []
    ordinal = 0
    for frame_index in (0, 3):
        for source_ordinal, source_slot in enumerate(("source1", "source2")):
            jobs.append(
                {
                    "job_id": f"rir_{ordinal:06d}",
                    "source_position_m": [
                        float(ordinal + 1),
                        1.0 + source_ordinal,
                        2.0,
                    ],
                    "uses": [
                        {
                            "episode_id": "example_episode",
                            "source_slot_id": source_slot,
                            "frame_index": frame_index,
                        }
                    ],
                }
            )
            ordinal += 1
    return {
        "schema": "avengine_room_rir_job_plan_v2",
        "status": "planned_not_run",
        "listener_position_m": [0.0, 1.5, 0.0],
        "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "unique_rir_job_count": len(jobs),
        "jobs": jobs,
    }


def _dynamic_episode_plan() -> dict[str, object]:
    jobs = []
    poses = {
        0: ([0.0, 1.5, 0.0], [1.0, 0.0, 0.0, 0.0]),
        3: ([0.5, 1.5, 0.0], [2**-0.5, 0.0, 2**-0.5, 0.0]),
    }
    ordinal = 0
    for frame_index in (0, 3):
        listener_position, listener_orientation = poses[frame_index]
        for source_ordinal, source_slot in enumerate(("source1", "source2")):
            source_position = [float(source_ordinal + 1), 1.0, 2.0]
            state_sha256 = canonical_json_sha256(
                {
                    "schema": "avengine_rir_acoustic_pair_state_v1",
                    "source_position_m": source_position,
                    "listener_position_m": listener_position,
                    "listener_orientation_wxyz": listener_orientation,
                }
            )
            jobs.append(
                {
                    "job_id": f"rir_dynamic_{ordinal:06d}",
                    "acoustic_state_sha256": state_sha256,
                    "source_position_m": source_position,
                    "listener_position_m": listener_position,
                    "listener_orientation_wxyz": listener_orientation,
                    "uses": [
                        {
                            "episode_id": "dynamic_episode",
                            "source_slot_id": source_slot,
                            "frame_index": frame_index,
                        }
                    ],
                }
            )
            ordinal += 1
    return {
        "schema": "avengine_room_rir_job_plan_v2",
        "status": "planned_not_run",
        "listener_pose_mode": "per_episode_frame",
        "cache_key_fields": [
            "source_position_m",
            "listener_position_m",
            "listener_orientation_wxyz",
        ],
        "unique_rir_job_count": len(jobs),
        "jobs": jobs,
    }


def _simulation() -> M4SimulationConfig:
    return M4SimulationConfig.from_mapping(
        {
            "frequency_bands": 4,
            "direct_sh_order": 1,
            "indirect_sh_order": 1,
            "direct_ray_count": 10,
            "indirect_ray_count": 10,
            "indirect_ray_depth": 2,
            "source_ray_count": 10,
            "source_ray_depth": 2,
            "max_diffraction_order": 1,
            "thread_count": 1,
            "sample_rate_hz": 16000,
            "max_ir_seconds": 1,
            "unit_scale": 1,
            "global_volume": 1,
            "speed_of_sound_m_s": 343,
            "direct": True,
            "indirect": True,
            "diffraction": True,
            "transmission": False,
            "mesh_simplification": False,
            "temporal_coherence": False,
            "channel_layout": {"type": "binaural", "channel_count": 2},
        }
    )


def _scene(tmp_path: Path) -> CompiledAcousticScene:
    manifest_value = {
        "schema": "fixture",
        "package_id": "fixture",
    }
    package_content_sha256 = canonical_json_sha256(manifest_value)
    manifest_value["package_content_sha256"] = package_content_sha256
    manifest = _write_json(tmp_path / "manifest.json", manifest_value)
    return CompiledAcousticScene(
        manifest_path=manifest,
        manifest_sha256=sha256_file(manifest),
        manifest=manifest_value,
        package_id="fixture",
        package_content_sha256=package_content_sha256,
        material_database_path=tmp_path / "materials.json",
        material_database_bytes=b"{}",
        material_database_sha256="33" * 32,
        material_categories_document={},
        rlr_material_database={},
        material_categories=("default",),
        objects=(),
        geometry_records={},
        triangle_count_by_material={"default": 1},
        qa_reports={},
    )


class _FakeRenderer:
    construction_count = 0
    render_count = 0

    def __init__(self, _scene, simulation, *, batch_size, **_kwargs) -> None:
        type(self).construction_count += 1
        self.batch_size = batch_size
        self.setup_report = {
            "wall_seconds": 0.1,
            "context_policy": {"configured_thread_count": simulation.thread_count},
        }

    def render(self, positions_m) -> RIRBatchResult:
        type(self).render_count += 1
        samples = tuple(
            np.full((2, 8 + index), float(position[0]), dtype="<f4")
            for index, position in enumerate(positions_m)
        )
        return RIRBatchResult(
            samples=samples,
            sample_rate_hz=16000,
            layout_id="rlr_binaural_lr_v1",
            channel_labels=("left", "right"),
            indirect_ray_efficiency=0.5,
            wall_seconds=0.25,
            process_cpu_seconds=0.2,
        )


class _PoseAwareFakeRenderer(_FakeRenderer):
    listener_poses = []

    def render(
        self,
        positions_m,
        *,
        listener_position_m,
        listener_orientation_wxyz,
    ) -> RIRBatchResult:
        type(self).listener_poses.append(
            (
                tuple(float(value) for value in listener_position_m),
                tuple(float(value) for value in listener_orientation_wxyz),
            )
        )
        return super().render(positions_m)


class _FakeNativeContext:
    def __init__(self) -> None:
        self.source_position = (1.0, 1.0, 2.0)
        self.listener_position = (0.0, 1.5, 0.0)
        self.listener_orientation = (1.0, 0.0, 0.0, 0.0)
        self.listener_pose_updates = []

    def set_source_position(self, _source_id, position) -> None:
        self.source_position = tuple(float(value) for value in position)

    def set_listener_pose(self, _listener_id, position, orientation) -> None:
        self.listener_position = tuple(float(value) for value in position)
        self.listener_orientation = tuple(float(value) for value in orientation)
        self.listener_pose_updates.append(
            (self.listener_position, self.listener_orientation)
        )

    def simulate_owned(self):
        return [
            SimpleNamespace(
                listener_id="cache_listener0",
                source_id="cache_slot_0000",
                sample_rate=16000.0,
                samples=np.ones((2, 4), dtype="<f4"),
                sample_count=4,
                channel_count=2,
            )
        ]

    def source_registration_receipts(self):
        return [
            SimpleNamespace(
                source_id="cache_slot_0000",
                position=np.asarray(self.source_position, dtype=np.float32),
                native_realized=True,
            )
        ]

    def listener_registration_receipts(self):
        return [
            SimpleNamespace(
                listener_id="cache_listener0",
                position=np.asarray(self.listener_position, dtype=np.float32),
                orientation_wxyz=np.asarray(
                    self.listener_orientation, dtype=np.float32
                ),
                native_realized=True,
            )
        ]

    def indirect_ray_efficiency(self) -> float:
        return 0.5


def test_native_renderer_updates_listener_pose_only_when_state_changes() -> None:
    renderer = object.__new__(_NativeRIRBatchRenderer)
    renderer.batch_size = 1
    renderer.source_ids = ("cache_slot_0000",)
    renderer.listener_id = "cache_listener0"
    renderer.current_listener_position_m = (0.0, 1.5, 0.0)
    renderer.current_listener_orientation_wxyz = (1.0, 0.0, 0.0, 0.0)
    renderer.listener_pose_update_count = 0
    renderer.context = _FakeNativeContext()
    renderer.selected = SimpleNamespace(sample_rate_hz=16000)
    renderer.channel_count = 2
    renderer.layout_id = "rlr_binaural_lr_v1"
    renderer.channel_labels = ("left", "right")

    moved_position = (0.5, 1.5, 0.0)
    moved_orientation = (2**-0.5, 0.0, 2**-0.5, 0.0)
    first = renderer.render(
        ((1.0, 1.0, 2.0),),
        listener_position_m=moved_position,
        listener_orientation_wxyz=moved_orientation,
    )
    second = renderer.render(
        ((2.0, 1.0, 2.0),),
        listener_position_m=moved_position,
        listener_orientation_wxyz=moved_orientation,
    )
    assert len(first.samples) == len(second.samples) == 1
    assert renderer.context.listener_pose_updates == [
        (moved_position, moved_orientation)
    ]
    assert renderer.listener_pose_update_count == 1


def test_rir_cache_writes_shards_index_and_resumes_without_rendering(tmp_path) -> None:
    _FakeRenderer.construction_count = 0
    _FakeRenderer.render_count = 0
    plan_path = _write_json(tmp_path / "plan.json", _plan())
    simulation_path = _write_json(
        tmp_path / "simulation.json", {"simulation": _simulation().to_dict()}
    )
    hrtf = tmp_path / "fixture.sofa"
    hrtf.write_bytes(b"fixture")
    output = tmp_path / "cache"
    arguments = {
        "plan_path": plan_path,
        "scene": _scene(tmp_path),
        "simulation_request_path": simulation_path,
        "simulation": _simulation(),
        "output": output,
        "layout_type": "binaural",
        "hrtf_file_path": hrtf,
        "batch_size": 2,
        "renderer_factory": _FakeRenderer,
    }
    first = render_rir_cache(**arguments)
    assert first.receipt["status"] == "pass"
    assert first.receipt["full_plan_complete"] is True
    assert first.receipt["retained_shard_count"] == 2
    assert _FakeRenderer.construction_count == 1
    assert _FakeRenderer.render_count == 2
    index = json.loads((output / "index.json").read_text())
    request = json.loads((output / "request.json").read_text())
    assert request["simulation"]["effective"]["channel_layout"] == {
        "type": "binaural",
        "channel_count": 2,
    }
    assert [entry["job_id"] for entry in index["entries"]] == [
        "rir_000000",
        "rir_000001",
        "rir_000002",
    ]
    normalized = validate_rir_job_plan(_plan())
    assert all(
        job["listener_position_m"] == [0.0, 1.5, 0.0]
        and job["listener_orientation_wxyz"] == [1.0, 0.0, 0.0, 0.0]
        for job in normalized
    )
    with np.load(output / "shards/shard_000001.npz", allow_pickle=False) as shard:
        assert shard["samples"].shape == (1, 2, 8)
        assert shard["lengths"].tolist() == [8]
    first_timing = (output / "timing.json").read_bytes()

    second = render_rir_cache(**arguments)
    assert second.receipt == first.receipt
    assert (output / "timing.json").read_bytes() == first_timing
    assert _FakeRenderer.construction_count == 1
    assert _FakeRenderer.render_count == 2

    shard_path = output / "shards/shard_000000.npz"
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {name: np.asarray(shard[name]).copy() for name in shard.files}
    arrays["job_ids"][0] = "wrong_job"
    np.savez_compressed(shard_path, **arrays)
    with pytest.raises(RIRCacheError, match="job IDs differ"):
        render_rir_cache(**arguments)


def test_rir_plan_rejects_duplicate_positions_and_cache_request_mismatch(
    tmp_path,
) -> None:
    duplicate = _plan(2)
    duplicate["jobs"][1]["source_position_m"] = duplicate["jobs"][0][
        "source_position_m"
    ]
    with pytest.raises(RIRCacheError, match="duplicate acoustic positions"):
        validate_rir_job_plan(duplicate)

    plan_path = _write_json(tmp_path / "plan.json", _plan())
    simulation_path = _write_json(
        tmp_path / "simulation.json", {"simulation": _simulation().to_dict()}
    )
    hrtf = tmp_path / "fixture.sofa"
    hrtf.write_bytes(b"fixture")
    output = tmp_path / "cache"
    render_rir_cache(
        plan_path=plan_path,
        scene=_scene(tmp_path),
        simulation_request_path=simulation_path,
        simulation=_simulation(),
        output=output,
        layout_type="binaural",
        hrtf_file_path=hrtf,
        batch_size=2,
        renderer_factory=_FakeRenderer,
    )
    with pytest.raises(RIRCacheError, match="different request"):
        render_rir_cache(
            plan_path=plan_path,
            scene=_scene(tmp_path),
            simulation_request_path=simulation_path,
            simulation=_simulation(),
            output=output,
            layout_type="binaural",
            hrtf_file_path=hrtf,
            batch_size=1,
            renderer_factory=_FakeRenderer,
        )


def test_cached_episode_reopens_exact_source_frame_grid(tmp_path: Path) -> None:
    plan_path = _write_json(tmp_path / "plan.json", _episode_plan())
    simulation_path = _write_json(
        tmp_path / "simulation.json", {"simulation": _simulation().to_dict()}
    )
    hrtf = tmp_path / "fixture.sofa"
    hrtf.write_bytes(b"fixture")
    output = tmp_path / "cache"
    render_rir_cache(
        plan_path=plan_path,
        scene=_scene(tmp_path),
        simulation_request_path=simulation_path,
        simulation=_simulation(),
        output=output,
        layout_type="binaural",
        hrtf_file_path=hrtf,
        batch_size=2,
        renderer_factory=_FakeRenderer,
    )

    episode = load_cached_rir_episode(
        cache_root=output,
        plan_path=plan_path,
        episode_id="example_episode",
        frame_count=75,
        frame_rate_hz=15,
    )
    assert episode.source_slot_ids == ("source1", "source2")
    assert episode.visual_frame_indices == (0, 3)
    assert episode.keyframe_samples == (0, 3200)
    assert episode.samples.shape == (2, 2, 2, 9)
    assert episode.lengths.tolist() == [[8, 9], [8, 9]]
    assert episode.samples[:, 0, 0, 0].tolist() == [1.0, 3.0]
    assert episode.samples[:, 1, 0, 0].tolist() == [2.0, 4.0]
    assert episode.evidence["status"] == "pass"
    assert len(episode.evidence["jobs"]) == 4

    shared_shards = {}
    session = RIRCacheSession(
        cache_root=output,
        plan_path=plan_path,
        frame_count=75,
        frame_rate_hz=15,
        shared_shard_cache=shared_shards,
    )
    first_from_resident_cache = session.load_episode("example_episode")
    second_from_resident_cache = session.load_episode("example_episode")
    assert len(shared_shards) == 2
    assert np.array_equal(first_from_resident_cache.samples, second_from_resident_cache.samples)


def test_dynamic_listener_pose_controls_batches_and_binds_retained_cache(
    tmp_path: Path,
) -> None:
    _PoseAwareFakeRenderer.construction_count = 0
    _PoseAwareFakeRenderer.render_count = 0
    _PoseAwareFakeRenderer.listener_poses = []
    plan = _dynamic_episode_plan()
    plan_path = _write_json(tmp_path / "plan.json", plan)
    simulation_path = _write_json(
        tmp_path / "simulation.json", {"simulation": _simulation().to_dict()}
    )
    hrtf = tmp_path / "fixture.sofa"
    hrtf.write_bytes(b"fixture")
    output = tmp_path / "cache"
    render_rir_cache(
        plan_path=plan_path,
        scene=_scene(tmp_path),
        simulation_request_path=simulation_path,
        simulation=_simulation(),
        output=output,
        layout_type="binaural",
        hrtf_file_path=hrtf,
        batch_size=2,
        coordinate_translation_m=(10.0, 0.0, -2.0),
        renderer_factory=_PoseAwareFakeRenderer,
    )
    assert _PoseAwareFakeRenderer.listener_poses == [
        ((10.0, 1.5, -2.0), (1.0, 0.0, 0.0, 0.0)),
        (
            (10.5, 1.5, -2.0),
            (2**-0.5, 0.0, 2**-0.5, 0.0),
        ),
    ]
    request = json.loads((output / "request.json").read_text())
    index = json.loads((output / "index.json").read_text())
    receipt = json.loads((output / "receipt.json").read_text())
    assert (
        request["plan"]["acoustic_state_binding"]
        == index["acoustic_state_binding"]
        == receipt["acoustic_state_binding"]
        == "source_listener_pose_per_job_v1"
    )
    assert [entry["job_index"] for entry in index["entries"]] == [0, 1, 2, 3]
    assert [entry["listener_position_m"] for entry in index["entries"]] == [
        [10.0, 1.5, -2.0],
        [10.0, 1.5, -2.0],
        [10.5, 1.5, -2.0],
        [10.5, 1.5, -2.0],
    ]
    with np.load(output / "shards/shard_000001.npz", allow_pickle=False) as shard:
        assert shard["listener_positions_m"].tolist() == [
            [10.5, 1.5, -2.0],
            [10.5, 1.5, -2.0],
        ]
        assert shard["listener_orientations_wxyz"].shape == (2, 4)
        assert shard["acoustic_state_sha256"].shape == (2,)

    episode = load_cached_rir_episode(
        cache_root=output,
        plan_path=plan_path,
        episode_id="dynamic_episode",
        frame_count=75,
        frame_rate_hz=15,
    )
    assert episode.evidence["acoustic_state_binding"] == (
        "source_listener_pose_per_job_v1"
    )
    assert [job["listener_position_m"] for job in episode.evidence["jobs"]] == [
        [0.0, 1.5, 0.0],
        [0.0, 1.5, 0.0],
        [0.5, 1.5, 0.0],
        [0.5, 1.5, 0.0],
    ]
    assert [job["realized_listener_position_m"] for job in episode.evidence["jobs"]] == [
        [10.0, 1.5, -2.0],
        [10.0, 1.5, -2.0],
        [10.5, 1.5, -2.0],
        [10.5, 1.5, -2.0],
    ]

    shard_path = output / "shards/shard_000001.npz"
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {name: np.asarray(shard[name]).copy() for name in shard.files}
    arrays["listener_positions_m"][0, 0] += 1.0
    np.savez_compressed(shard_path, **arrays)
    with pytest.raises(RIRCacheError, match="Listener pose differs"):
        render_rir_cache(
            plan_path=plan_path,
            scene=_scene(tmp_path),
            simulation_request_path=simulation_path,
            simulation=_simulation(),
            output=output,
            layout_type="binaural",
            hrtf_file_path=hrtf,
            batch_size=2,
            coordinate_translation_m=(10.0, 0.0, -2.0),
            renderer_factory=_PoseAwareFakeRenderer,
        )


def test_dynamic_plan_rejects_pose_hash_mismatch() -> None:
    plan = _dynamic_episode_plan()
    plan["jobs"][0]["listener_position_m"][0] = 9.0
    with pytest.raises(RIRCacheError, match="acoustic-state SHA-256 differs"):
        validate_rir_job_plan(plan)


@pytest.mark.parametrize(
    "mutation",
    (
        "cross_document_identity",
        "scene_package_id",
        "scene_manifest_sha256",
        "scene_manifest_path",
        "simulation_request_sha256",
        "simulation_request_path",
        "hrtf_sha256",
        "hrtf_path",
        "plan_path",
        "receipt_full_job_count",
        "receipt_selected_job_count",
        "index_selected_job_count",
        "index_listener_position",
    ),
)
def test_cache_session_rejects_tampered_request_identity_and_external_inputs(
    tmp_path: Path,
    mutation: str,
) -> None:
    plan_path = _write_json(tmp_path / "plan.json", _episode_plan())
    simulation_path = _write_json(
        tmp_path / "simulation.json", {"simulation": _simulation().to_dict()}
    )
    hrtf = tmp_path / "fixture.sofa"
    hrtf.write_bytes(b"fixture")
    output = tmp_path / "cache"
    render_rir_cache(
        plan_path=plan_path,
        scene=_scene(tmp_path),
        simulation_request_path=simulation_path,
        simulation=_simulation(),
        output=output,
        layout_type="binaural",
        hrtf_file_path=hrtf,
        batch_size=2,
        renderer_factory=_FakeRenderer,
    )
    request_path = output / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if mutation == "cross_document_identity":
        forged = "aa" * 32
        request["request_identity_sha256"] = forged
        for name in ("receipt.json", "index.json"):
            path = output / name
            value = json.loads(path.read_text(encoding="utf-8"))
            value["request_identity_sha256"] = forged
            _write_json(path, value)
    elif mutation == "scene_package_id":
        request["acoustic_scene"]["package_id"] = "forged_package"
    elif mutation == "scene_manifest_sha256":
        request["acoustic_scene"]["manifest_sha256"] = "aa" * 32
    elif mutation == "scene_manifest_path":
        request["acoustic_scene"]["manifest_path"] = str(
            tmp_path / "missing_manifest.json"
        )
    elif mutation == "simulation_request_sha256":
        request["simulation"]["request_sha256"] = "aa" * 32
    elif mutation == "simulation_request_path":
        request["simulation"]["request_path"] = str(
            tmp_path / "missing_simulation.json"
        )
    elif mutation == "hrtf_sha256":
        request["output"]["hrtf_sha256"] = "aa" * 32
    elif mutation == "hrtf_path":
        request["output"]["hrtf_path"] = str(tmp_path / "missing.sofa")
    elif mutation == "plan_path":
        copied_plan = tmp_path / "copied_plan.json"
        copied_plan.write_bytes(plan_path.read_bytes())
        request["plan"]["path"] = str(copied_plan)
    elif mutation in {
        "receipt_full_job_count",
        "receipt_selected_job_count",
        "index_selected_job_count",
    }:
        file_name, field = {
            "receipt_full_job_count": ("receipt.json", "full_plan_job_count"),
            "receipt_selected_job_count": ("receipt.json", "selected_job_count"),
            "index_selected_job_count": ("index.json", "selected_job_count"),
        }[mutation]
        path = output / file_name
        value = json.loads(path.read_text(encoding="utf-8"))
        value[field] -= 1
        _write_json(path, value)
    elif mutation == "index_listener_position":
        path = output / "index.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["entries"][0]["listener_position_m"][0] += 1.0
        _write_json(path, value)
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)
    _write_json(request_path, request)

    with pytest.raises(RIRCacheError):
        RIRCacheSession(
            cache_root=output,
            plan_path=plan_path,
            frame_count=75,
            frame_rate_hz=15,
        )

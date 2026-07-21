from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.m3.runtime import CompiledAcousticScene
from avengine.m4.runtime import M4SimulationConfig
from avengine.m6x.rir_cache import (
    RIRBatchResult,
    RIRCacheError,
    RIRCacheSession,
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
    manifest = _write_json(tmp_path / "manifest.json", {"schema": "fixture"})
    return CompiledAcousticScene(
        manifest_path=manifest,
        manifest_sha256="11" * 32,
        manifest={},
        package_id="fixture",
        package_content_sha256="22" * 32,
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

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.acoustics.runtime import CompiledAcousticScene, RuntimeContractError
from avengine.spatial_audio.runtime import M4SimulationConfig
from avengine.timeline.acoustics import DynamicRIRSequence, validate_acoustic_keyframes
from avengine.capture import acoustics
import tools.review.render_review_acoustics as review_acoustics


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROUTE_PATH = (
    REPOSITORY_ROOT / "examples/m5_1/legacy_apartment/route_manifest.json"
)
LEGACY_REQUEST_PATH = (
    REPOSITORY_ROOT / "examples/m5_1/legacy_apartment/m1_capture_request.json"
)
LEGACY_SOURCE_PATH = (
    REPOSITORY_ROOT / "examples/m5_1/legacy_apartment/source_manifest.json"
)
MP3D_ROUTE_PATH = (
    REPOSITORY_ROOT / "examples/m5_1/mp3d_articulated_review/route_manifest.json"
)


def _legacy_capture_contract_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    repository = tmp_path / "portable_repository"
    repository.mkdir()
    monkeypatch.setattr(review_acoustics, "REPOSITORY", repository)

    route_path = repository / "route_manifest.json"
    request_path = repository / "m1_capture_request.json"
    room_path = repository / "room_manifest.json"
    route = load_json(LEGACY_ROUTE_PATH)
    request = load_json(LEGACY_REQUEST_PATH)
    write_json(route_path, route)
    write_json(request_path, request)
    write_json(
        room_path,
        {
            "schema": "avengine_room_package_v1",
            "room_id": request["room_id"],
        },
    )

    camera_transform = request["primary_camera_rig"]["world_from_rig"]
    evidence: dict = {
        "schema": "avengine_m5_1_human_beagle_capture_v1",
        "status": "pass",
        "qualification_claim": False,
        "inputs": {
            "route_provenance": {
                "route_id": route["route_id"],
                "route_manifest_path": str(route_path.resolve()),
                "route_manifest_sha256": sha256_file(route_path),
                "path_consumption": (
                    "verbatim_manifest_routes_habitat_trajectory_m"
                ),
                "human_habitat_trajectory_sha256": route["routes"][
                    "human_path"
                ]["habitat_trajectory_sha256"],
                "dog_habitat_trajectory_sha256": route["routes"]["dog_path"][
                    "habitat_trajectory_sha256"
                ],
            },
            "m1_request": {
                "path": str(request_path.resolve()),
                "sha256": sha256_file(request_path),
            },
            "room_manifest": {
                "path": str(room_path.resolve()),
                "sha256": sha256_file(room_path),
            },
        },
        "actors": [
            {
                "actor_id": "human0",
                "actor_class": "human",
                "emitter_link": "Bip01 MJaw",
                "semantic_id": 220,
            },
            {
                "actor_id": "dog0",
                "actor_class": "dog",
                "emitter_link": "beagle Xtra Mouth",
                "semantic_id": 221,
            },
        ],
        "anchor_order": [
            "human0.head",
            "human0.mouth_emitter",
            "dog0.mouth_emitter",
        ],
        "camera": {
            "position_m": camera_transform["translation_m"],
            "rotation_xyzw": camera_transform["rotation_xyzw"],
            "horizontal_fov_deg": request["primary_camera_rig"][
                "shared_calibration"
            ]["hfov_degrees"],
        },
    }
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    evidence_path = repository / "capture_evidence.json"
    write_json(evidence_path, evidence)
    return evidence_path, route_path


def test_legacy_apartment_capture_contract_uses_narrow_frozen_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path, _route_path = _legacy_capture_contract_fixture(
        tmp_path, monkeypatch
    )
    contract = review_acoustics._capture_contract(
        load_json(evidence_path),
        capture_evidence_path=evidence_path,
        source_manifest_path=LEGACY_SOURCE_PATH,
    )

    assert contract["room_family"] == "legacy_apartment"
    assert contract["room_id"] == "legacy_ue_apartment_0000_v1"
    assert contract["route_id"] == "m5_1_legacy_apartment_human_dog_18s_v1"
    assert contract["request_id"] == "m5_1_legacy_apartment_view0_v1"
    bindings = contract["source_actor_bindings"]
    assert bindings["room_family"] == "legacy_apartment"
    assert {
        source_id: (
            binding["actor_id"],
            binding["emitter_anchor_index"],
            binding["semantic_id"],
        )
        for source_id, binding in bindings["bindings"].items()
    } == {
        "source0": ("human0", 1, 220),
        "source1": ("dog0", 2, 221),
    }


def test_legacy_apartment_capture_contract_rejects_route_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path, _route_path = _legacy_capture_contract_fixture(
        tmp_path, monkeypatch
    )
    evidence = load_json(evidence_path)
    evidence["inputs"]["route_provenance"]["route_manifest_sha256"] = "0" * 64
    evidence.pop("evidence_content_sha256")
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    write_json(evidence_path, evidence)

    with pytest.raises(RuntimeError, match="route manifest bytes differ"):
        review_acoustics._capture_contract(
            evidence,
            capture_evidence_path=evidence_path,
            source_manifest_path=LEGACY_SOURCE_PATH,
        )


def test_legacy_route_locator_adapter_cannot_weaken_modern_mp3d_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path, route_path = _legacy_capture_contract_fixture(
        tmp_path, monkeypatch
    )
    modern_route = load_json(MP3D_ROUTE_PATH)
    write_json(route_path, modern_route)
    evidence = load_json(evidence_path)
    provenance = evidence["inputs"]["route_provenance"]
    provenance["route_id"] = modern_route["route_id"]
    provenance["route_manifest_sha256"] = sha256_file(route_path)
    evidence.pop("evidence_content_sha256")
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    write_json(evidence_path, evidence)

    with pytest.raises(RuntimeError, match="structured route record"):
        review_acoustics._capture_contract(
            evidence,
            capture_evidence_path=evidence_path,
            source_manifest_path=LEGACY_SOURCE_PATH,
        )


def _legacy_trajectories(frame_count: int = 270) -> dict[str, np.ndarray]:
    progress = np.arange(frame_count, dtype=np.float64) / 15.0
    return {
        "human_source": np.column_stack(
            (progress, np.full(frame_count, 1.55), np.sin(progress))
        ),
        "dog_source": np.column_stack(
            (progress - 0.35, np.full(frame_count, 0.38), np.sin(progress))
        ),
    }


def _grid(
    frame_count: int = 270,
    *,
    frame_rate_hz: int = 15,
    stride: int = 3,
    tick_rate_hz: int = 48_000,
    sample_rate_hz: int = 16_000,
) -> acoustics.ResearchReviewKeyframeGrid:
    return acoustics.build_strided_review_keyframes(
        _legacy_trajectories(frame_count),
        visual_frame_rate_hz=frame_rate_hz,
        rir_stride_frames=stride,
        listener_position_m=(-0.7, 1.471, 0.65),
        listener_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        timeline_tick_rate_hz=tick_rate_hz,
        sample_rate_hz=sample_rate_hz,
    )


def _scene() -> CompiledAcousticScene:
    database = b"{}"
    return CompiledAcousticScene(
        manifest_path=__file__,
        manifest_sha256="1" * 64,
        manifest={},
        package_id="m5_1_test_scene",
        package_content_sha256="2" * 64,
        material_database_path=__file__,
        material_database_bytes=database,
        material_database_sha256=hashlib.sha256(database).hexdigest(),
        material_categories_document={},
        rlr_material_database={},
        material_categories=("wall",),
        objects=({},),
        geometry_records={},
        triangle_count_by_material={"wall": 1},
        qa_reports={},
    )


def _simulation(sample_rate_hz: int = 16_000) -> M4SimulationConfig:
    return M4SimulationConfig.from_mapping(
        {
            "frequency_bands": 4,
            "direct_sh_order": 3,
            "indirect_sh_order": 1,
            "direct_ray_count": 8,
            "indirect_ray_count": 8,
            "indirect_ray_depth": 2,
            "source_ray_count": 8,
            "source_ray_depth": 2,
            "max_diffraction_order": 1,
            "thread_count": 1,
            "sample_rate_hz": float(sample_rate_hz),
            "max_ir_seconds": 1.0,
            "unit_scale": 1.0,
            "global_volume": 1.0,
            "speed_of_sound_m_s": 343.0,
            "direct": True,
            "indirect": True,
            "diffraction": True,
            "transmission": False,
            "mesh_simplification": False,
            "temporal_coherence": False,
            "channel_layout": {"type": "ambisonics", "channel_count": 4},
        }
    )


def test_legacy_270_frames_build_exact_90_keyframe_5_hz_grid() -> None:
    grid = _grid()

    assert grid.visual_frame_count == 270
    assert grid.visual_frame_indices == tuple(range(0, 270, 3))
    assert len(grid.keyframes) == 90
    assert grid.rir_keyframe_rate_hz.numerator == 5
    assert grid.rir_keyframe_rate_hz.denominator == 1
    assert grid.episode_tick_count == 864_000
    assert grid.episode_sample_count == 288_000
    assert grid.keyframes[-1].tick == 854_400
    assert grid.keyframes[-1].sample_index == 284_800
    assert grid.source_ids == ("dog_source", "human_source")

    record = acoustics.research_review_trajectory_record(grid)
    assert record["sampling"]["rir_stride_frames"] == 3
    assert record["sampling"]["sampled_visual_frame_indices"][-1] == 267
    assert record["sampling"]["final_interval_policy"] == "hold_last_rir_to_episode_end"
    assert record["listener_motion"] == "fixed"
    assert record["qualification_claim"] is False
    assert len(canonical_json_sha256(record)) == 64

    # The research path accepts 90 keyframes without changing M5's frozen
    # formal requirement of exactly 75.
    with pytest.raises(RuntimeContractError, match="requires 75"):
        validate_acoustic_keyframes(
            grid.keyframes, expected_source_ids=grid.source_ids
        )


def test_arbitrary_frame_count_and_stride_are_explicit_and_exact() -> None:
    grid = _grid(
        17,
        frame_rate_hz=20,
        stride=4,
        tick_rate_hz=100,
        sample_rate_hz=40,
    )
    assert grid.visual_frame_indices == (0, 4, 8, 12, 16)
    assert grid.rir_keyframe_rate_hz.numerator == 5
    assert grid.episode_tick_count == 85
    assert grid.episode_sample_count == 34
    assert [frame.tick for frame in grid.keyframes] == [0, 20, 40, 60, 80]
    assert [frame.sample_index for frame in grid.keyframes] == [0, 8, 16, 24, 32]


def test_grid_rejects_listener_motion_and_non_stride_boundaries() -> None:
    grid = _grid(9, frame_rate_hz=9, stride=3)
    moved = list(grid.keyframes)
    moved[1] = replace(moved[1], listener_position_m=(0.0, 1.0, 0.0))
    with pytest.raises(RuntimeContractError, match="listener must remain fixed"):
        acoustics.validate_research_review_grid(
            replace(grid, keyframes=tuple(moved))
        )

    with pytest.raises(RuntimeContractError, match="explicit stride"):
        acoustics.validate_research_review_grid(
            replace(grid, visual_frame_indices=(0, 2, 6))
        )


class _FakeConfiguration:
    pass


class _FakeContext:
    instances: list["_FakeContext"] = []

    def __init__(self, configuration: object):
        self.configuration = configuration
        self.sources: dict[str, tuple[float, float, float]] = {}
        self.simulate_count = 0
        _FakeContext.instances.append(self)

    def load_acoustic_scene(self, database, categories, objects):
        return object()

    def add_source(self, source_id, position, radius):
        self.sources[source_id] = tuple(position)

    def add_listener(
        self, listener_id, position, orientation, layout, channel_count, radius, hrtf
    ):
        self.listener_id = listener_id
        self.listener_position = tuple(position)
        self.listener_orientation = tuple(orientation)
        self.hrtf = hrtf

    def set_source_position(self, source_id, position):
        self.sources[source_id] = tuple(position)

    def set_source_radius(self, source_id, radius):
        pass

    def set_listener_pose(self, listener_id, position, orientation):
        self.listener_position = tuple(position)
        self.listener_orientation = tuple(orientation)

    def set_listener_radius(self, listener_id, radius):
        pass

    def simulate_owned(self):
        self.simulate_count += 1
        result = []
        for source_index, source_id in enumerate(sorted(self.sources)):
            length = 3 + source_index + (self.simulate_count % 2)
            result.append(
                SimpleNamespace(
                    listener_id=self.listener_id,
                    source_id=source_id,
                    sample_rate=16_000.0,
                    channel_count=2,
                    sample_count=length,
                    samples=np.full(
                        (2, length),
                        self.simulate_count + source_index,
                        dtype=np.float32,
                    ),
                )
            )
        return result

    def indirect_ray_efficiency(self):
        return 0.625


class _FakeLayout:
    Binaural = "binaural"


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeContext.instances.clear()
    module = SimpleNamespace(
        RLRContextConfiguration=_FakeConfiguration,
        RLRAcousticContext=_FakeContext,
        RLRChannelLayoutType=_FakeLayout,
    )
    monkeypatch.setattr(
        acoustics, "load_habitat_runtime", lambda: (module, {"fake": True})
    )
    monkeypatch.setattr(
        acoustics,
        "_native_configuration",
        lambda habitat_module, simulation: (
            _FakeConfiguration(),
            simulation.to_dict(),
        ),
    )
    monkeypatch.setattr(acoustics, "_upload_report", lambda value: {"verified": True})
    monkeypatch.setattr(
        acoustics, "_verify_upload_report", lambda scene, report: None
    )

    def receipts(
        context,
        habitat_module,
        sources,
        listener,
        *,
        canonical_order,
        layout_type,
        channel_count,
        hrtf_file_path,
    ):
        return {
            "authority": "native_registration_readback",
            "sources": [
                {
                    "source_id": source.anchor_id,
                    "position": list(source.position_m),
                    "native_realized": True,
                }
                for source in sources
            ],
            "listener": {
                "listener_id": listener.anchor_id,
                "position": list(listener.position_m),
                "hrtf_file_path": hrtf_file_path,
                "native_realized": True,
            },
        }

    monkeypatch.setattr(acoustics, "_native_registration_receipts", receipts)


def test_mock_native_render_retains_receipts_ir_hashes_and_trajectory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runtime(monkeypatch)
    grid = _grid(9, frame_rate_hz=9, stride=3)
    hrtf = tmp_path / "review.sofa"
    hrtf.write_bytes(b"mock-hrtf")

    result = acoustics.render_research_review_binaural_rir_sequence(
        _scene(),
        _simulation(),
        grid=grid,
        hrtf_file_path=str(hrtf),
    )

    assert isinstance(result, DynamicRIRSequence)
    assert len(_FakeContext.instances) == 1
    assert _FakeContext.instances[0].simulate_count == 3
    assert result.samples.shape == (3, 2, 2, 5)
    assert result.lengths.tolist() == [[4, 5], [3, 4], [4, 5]]
    assert result.source_ids == ("dog_source", "human_source")
    assert result.keyframe_samples == (0, 5_333, 10_667)
    assert result.metadata["qualification_claim"] is False
    assert result.metadata["context_policy"]["simulate_calls"] == 3
    assert len(result.metadata["endpoint_receipts"]) == 3
    assert len(result.metadata["ir_sha256_by_keyframe_source"]) == 3
    assert result.metadata["trajectory"] == acoustics.research_review_trajectory_record(
        grid
    )
    assert result.trajectory_sha256 == canonical_json_sha256(
        result.metadata["trajectory"]
    )
    assert result.metadata["hrtf"] == {
        "input_role": "hrtf",
        "sha256": sha256_file(hrtf),
    }
    receipt_text = repr(result.metadata["endpoint_receipts"])
    assert str(hrtf.resolve()) not in receipt_text
    assert "input-role:hrtf" in receipt_text


def test_variable_episode_reuses_m5_dynamic_convolution() -> None:
    grid = _grid(
        6,
        frame_rate_hz=6,
        stride=2,
        tick_rate_hz=60,
        sample_rate_hz=12,
    )
    trajectory = acoustics.research_review_trajectory_record(grid)
    rirs = np.zeros((3, 2, 2, 1), dtype=np.float32)
    rirs[:, :, 0, 0] = 1.0
    rirs[:, :, 1, 0] = 0.5
    sequence = DynamicRIRSequence(
        samples=rirs,
        lengths=np.ones((3, 2), dtype=np.uint32),
        source_ids=grid.source_ids,
        keyframe_ticks=tuple(frame.tick for frame in grid.keyframes),
        keyframe_samples=tuple(frame.sample_index for frame in grid.keyframes),
        sample_rate_hz=12,
        layout_type="binaural",
        layout_id="rlr_binaural_lr_v1",
        channel_labels=("left", "right"),
        trajectory_sha256=canonical_json_sha256(trajectory),
        metadata={"trajectory": deepcopy(trajectory)},
    )
    dry = {
        "dog_source": np.ones(12, dtype=np.float64),
        "human_source": -np.ones(12, dtype=np.float64) * 0.25,
    }

    stems, mixture = acoustics.render_research_review_binaural_audio(
        dry, sequence, grid=grid
    )

    assert set(stems) == set(grid.source_ids)
    assert all(stem.episode.shape == (2, 12) for stem in stems.values())
    assert mixture.shape == (2, 12)
    assert np.allclose(mixture[0], 0.75, rtol=0.0, atol=1.0e-12)
    assert np.allclose(mixture[1], 0.375, rtol=0.0, atol=1.0e-12)


def test_audio_rejects_trajectory_hash_drift() -> None:
    grid = _grid(
        6,
        frame_rate_hz=6,
        stride=2,
        tick_rate_hz=60,
        sample_rate_hz=12,
    )
    sequence = DynamicRIRSequence(
        samples=np.ones((3, 2, 2, 1), dtype=np.float32),
        lengths=np.ones((3, 2), dtype=np.uint32),
        source_ids=grid.source_ids,
        keyframe_ticks=tuple(frame.tick for frame in grid.keyframes),
        keyframe_samples=tuple(frame.sample_index for frame in grid.keyframes),
        sample_rate_hz=12,
        layout_type="binaural",
        layout_id="rlr_binaural_lr_v1",
        channel_labels=("left", "right"),
        trajectory_sha256="0" * 64,
        metadata={},
    )
    with pytest.raises(RuntimeContractError, match="trajectory hash"):
        acoustics.render_research_review_binaural_audio(
            {
                "dog_source": np.ones(12),
                "human_source": np.ones(12),
            },
            sequence,
            grid=grid,
        )

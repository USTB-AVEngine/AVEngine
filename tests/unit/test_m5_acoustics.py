from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256
from avengine.m3.runtime import CompiledAcousticScene
from avengine.m3.runtime import RuntimeContractError
from avengine.spatial_audio.runtime import M4SimulationConfig
from avengine.m5 import acoustics
from avengine.m5.acoustics import (
    AcousticKeyframe,
    trajectory_record,
    validate_acoustic_keyframes,
)


def _frames() -> list[AcousticKeyframe]:
    return [
        AcousticKeyframe(
            tick=3200 * index,
            sample_index=(3200 * index + 1) // 3,
            source_positions_m={
                "source0": (0.0, 1.0, index / 75.0),
                "source1": (1.0, 1.0, -index / 75.0),
            },
            listener_position_m=(-2.5, 1.55, 0.0),
            listener_orientation_wxyz=(2**-0.5, 0.0, -(2**-0.5), 0.0),
        )
        for index in range(75)
    ]


def test_current_dynamic_loader_forwards_explicit_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    selected = (SimpleNamespace(), {"runtime_mode": "current-installed"})

    def fake_load(**kwargs: object):
        calls.append(dict(kwargs))
        return selected

    monkeypatch.setattr(acoustics, "load_habitat_runtime", fake_load)
    result = acoustics._load_dynamic_habitat_runtime(
        runtime_mode="current-installed",
        runtime_prefix="/external/prefix",
        rlr_sdk_root="/external/rlr",
        magnum_python_site="/external/magnum",
    )

    assert result is selected
    assert calls == [
        {
            "runtime_mode": "current-installed",
            "runtime_prefix": "/external/prefix",
            "rlr_sdk_root": "/external/rlr",
            "magnum_python_site": "/external/magnum",
        }
    ]


def test_historical_dynamic_loader_keeps_zero_argument_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    selected = (SimpleNamespace(), {"runtime_mode": "historical"})

    def fake_load():
        calls.append(True)
        return selected

    monkeypatch.setattr(acoustics, "load_habitat_runtime", fake_load)
    result = acoustics._load_dynamic_habitat_runtime(
        runtime_mode="historical",
        runtime_prefix=None,
        rlr_sdk_root=None,
        magnum_python_site=None,
    )

    assert result is selected
    assert calls == [True]


def test_exact_visual_frame_acoustic_grid() -> None:
    frames = validate_acoustic_keyframes(
        _frames(), expected_source_ids=("source0", "source1")
    )
    assert len(frames) == 75
    assert frames[0].sample_index == 0
    assert frames[-1].sample_index == 78_933
    record = trajectory_record(frames, ("source0", "source1"))
    assert record["source_ids"] == ["source0", "source1"]
    assert record["keyframes"][-1]["tick"] == 236_800


def test_rejects_rounded_fixed_1067_sample_grid() -> None:
    frames = _frames()
    frames[2] = AcousticKeyframe(
        tick=6400,
        sample_index=2134,
        source_positions_m=frames[2].source_positions_m,
        listener_position_m=frames[2].listener_position_m,
        listener_orientation_wxyz=frames[2].listener_orientation_wxyz,
    )
    with pytest.raises(RuntimeContractError, match="non-rational"):
        validate_acoustic_keyframes(
            frames, expected_source_ids=("source0", "source1")
        )


def test_rejects_source_identity_drift() -> None:
    frames = _frames()
    frames[30] = AcousticKeyframe(
        tick=frames[30].tick,
        sample_index=frames[30].sample_index,
        source_positions_m={"source0": (0.0, 0.0, 0.0), "renamed": (1.0, 0.0, 0.0)},
        listener_position_m=frames[30].listener_position_m,
        listener_orientation_wxyz=frames[30].listener_orientation_wxyz,
    )
    with pytest.raises(RuntimeContractError, match="identity set"):
        validate_acoustic_keyframes(
            frames, expected_source_ids=("source0", "source1")
        )


def test_moving_listener_updates_every_keyframe_and_binds_trajectory_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        AcousticKeyframe(
            tick=3200 * index,
            sample_index=(3200 * index + 1) // 3,
            source_positions_m={"source0": (2.0, 1.0, -1.0)},
            listener_position_m=(index / 100.0, 1.55, index / 200.0),
            listener_orientation_wxyz=(
                math.cos(math.radians(index) / 2.0),
                0.0,
                math.sin(math.radians(index) / 2.0),
                0.0,
            ),
        )
        for index in range(75)
    ]
    listener_pose_calls: list[
        tuple[str, tuple[float, ...], tuple[float, ...]]
    ] = []

    class FakeContext:
        def __init__(self, configuration: object) -> None:
            self.source_ids: list[str] = []
            self.listener_id = ""

        def load_acoustic_scene(self, database, categories, objects):
            return object()

        def add_source(self, source_id, position, radius) -> None:
            self.source_ids.append(source_id)

        def add_listener(
            self,
            listener_id,
            position,
            orientation,
            layout,
            channel_count,
            radius,
            hrtf,
        ) -> None:
            self.listener_id = listener_id

        def set_source_position(self, source_id, position) -> None:
            pass

        def set_source_radius(self, source_id, radius) -> None:
            pass

        def set_listener_pose(self, listener_id, position, orientation) -> None:
            listener_pose_calls.append(
                (listener_id, tuple(position), tuple(orientation))
            )

        def set_listener_radius(self, listener_id, radius) -> None:
            pass

        def simulate_owned(self):
            return [
                SimpleNamespace(
                    listener_id=self.listener_id,
                    source_id=source_id,
                    sample_rate=16_000.0,
                    channel_count=1,
                    sample_count=2,
                    samples=np.ones((1, 2), dtype=np.float32),
                )
                for source_id in self.source_ids
            ]

        def indirect_ray_efficiency(self) -> float:
            return 0.5

    fake_habitat = SimpleNamespace(
        RLRAcousticContext=FakeContext,
        RLRChannelLayoutType=SimpleNamespace(Mono="mono"),
    )
    monkeypatch.setattr(
        acoustics,
        "load_habitat_runtime",
        lambda: (fake_habitat, {"fake": True}),
    )
    monkeypatch.setattr(
        acoustics,
        "_native_configuration",
        lambda habitat_module, simulation: (object(), simulation.to_dict()),
    )
    monkeypatch.setattr(acoustics, "_upload_report", lambda value: {"verified": True})
    monkeypatch.setattr(
        acoustics, "_verify_upload_report", lambda scene, report: None
    )
    monkeypatch.setattr(
        acoustics,
        "_native_registration_receipts",
        lambda *args, **kwargs: {"verified": True},
    )

    database = b"{}"
    scene = CompiledAcousticScene(
        manifest_path=Path(__file__),
        manifest_sha256="1" * 64,
        manifest={},
        package_id="moving_listener_test_scene",
        package_content_sha256="2" * 64,
        material_database_path=Path(__file__),
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
    simulation = M4SimulationConfig.from_mapping(
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
            "sample_rate_hz": 16_000.0,
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
            "channel_layout": {"type": "mono", "channel_count": 1},
        }
    )

    result = acoustics.render_dynamic_rir_sequence(
        scene,
        simulation,
        keyframes=frames,
        layout_type="mono",
        channel_count=1,
    )

    expected_calls = [
        (
            "listener0",
            frame.listener_position_m,
            frame.listener_orientation_wxyz,
        )
        for frame in validate_acoustic_keyframes(
            frames, expected_source_ids=("source0",)
        )
    ]
    assert listener_pose_calls == expected_calls
    assert len({position for _, position, _ in expected_calls}) == 75
    assert len({orientation for _, _, orientation in expected_calls}) == 75

    base_identity = canonical_json_sha256(
        trajectory_record(frames, ("source0",))
    )
    moved_position = list(frames)
    moved_position[37] = replace(
        moved_position[37], listener_position_m=(9.0, 1.55, 0.185)
    )
    changed_orientation = list(frames)
    changed_orientation[37] = replace(
        changed_orientation[37],
        listener_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    assert result.trajectory_sha256 == base_identity
    assert result.metadata["motion_acoustics"] == {
        "endpoint_pose_cadence_hz": 15,
        "source_position_updated_per_keyframe": True,
        "listener_position_updated_per_keyframe": True,
        "listener_orientation_updated_per_keyframe": True,
        "room_propagation_recomputed_per_keyframe": True,
        "continuous_doppler_modeled": False,
        "device_motion_noise_modeled": False,
    }
    assert canonical_json_sha256(
        trajectory_record(moved_position, ("source0",))
    ) != base_identity
    assert canonical_json_sha256(
        trajectory_record(changed_orientation, ("source0",))
    ) != base_identity

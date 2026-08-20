from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np

import pytest

import avengine.cli as cli
import avengine.m5.current_visual as current_visual


def _mp3d_room_inputs(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        room={
            "room_id": "habitat_mp3d_example_17DRP5sb8fy",
            "room_kind": "habitat_native",
            "scene": {
                "scene_id_kind": "path",
                "scene_id": "${AVENGINE_MP3D_ROOT}/scene.glb",
                "dataset_config_path": "${AVENGINE_MP3D_ROOT}/dataset.json",
                "navmesh_path": "${AVENGINE_MP3D_ROOT}/scene.navmesh",
            },
            "assets": [],
        },
        room_path=tmp_path / "room_manifest.json",
        request_path=tmp_path / "m1_request.json",
        request={"seed": 17},
    )


def _m2_inputs(tmp_path: Path, *, room_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request={"room_id": room_id, "seed": 17},
        asset_path=tmp_path / "animal_manifest.json",
        request_path=tmp_path / "m2_request.json",
    )


def test_current_visual_returns_not_run_before_runtime_activation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    m2_inputs = _m2_inputs(tmp_path, room_id="blender_custom_two_zone_v1")
    room_inputs = _mp3d_room_inputs(tmp_path)
    monkeypatch.setattr(current_visual, "load_m2_inputs", lambda *_args: m2_inputs)
    monkeypatch.setattr(current_visual, "load_m1_inputs", lambda *_args: room_inputs)
    monkeypatch.setattr(
        current_visual,
        "validate_capture_context",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("mismatched room reached capture-context validation")
        ),
    )
    monkeypatch.setattr(
        current_visual,
        "validate_scene_asset_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mismatched room reached scene validation")
        ),
    )
    monkeypatch.setattr(
        current_visual,
        "prepare_installed_habitat_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("mismatched room activated Habitat")
        ),
    )

    output = tmp_path / "not-run"
    receipt = current_visual.capture_current_visual(
        animal_manifest_path=tmp_path / "animal_manifest.json",
        m2_request_path=tmp_path / "m2_request.json",
        room_manifest_path=tmp_path / "room_manifest.json",
        m1_request_path=tmp_path / "m1_request.json",
        runtime_prefix=tmp_path / "prefix",
        mp3d_root=tmp_path / "mp3d",
        magnum_python_site=tmp_path / "magnum",
        output_directory=output,
    )

    assert receipt["status"] == "not_run"
    assert receipt["episode_counted"] is False
    assert "blender_custom_two_zone_v1" in receipt["reason"]
    assert (output / "research_receipt.json").is_file()
    encoded = json.dumps(receipt, sort_keys=True)
    assert "schema" not in encoded
    assert "sha256" not in encoded


def test_current_visual_forwards_only_explicit_installed_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    room_inputs = _mp3d_room_inputs(tmp_path)
    m2_inputs = _m2_inputs(tmp_path, room_id=room_inputs.room["room_id"])
    runtime_prefix = tmp_path / "prefix"
    mp3d_root = tmp_path / "mp3d"
    magnum_site = tmp_path / "magnum"
    installed_runtime = SimpleNamespace(
        habitat_sim=SimpleNamespace(built_with_bullet=True)
    )
    calls: dict[str, Any] = {}
    receipt = {
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "capture": {"frame_count": 75},
    }
    monkeypatch.setattr(current_visual, "load_m2_inputs", lambda *_args: m2_inputs)
    monkeypatch.setattr(current_visual, "load_m1_inputs", lambda *_args: room_inputs)
    monkeypatch.setattr(current_visual, "validate_capture_context", lambda *_args: [])
    monkeypatch.setattr(
        current_visual, "validate_scene_asset_graph", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(current_visual, "load_runtime_asset_bundle", lambda _inputs: "bundle")
    monkeypatch.setattr(
        current_visual, "compile_frame_applications", lambda *_args: tuple(range(75))
    )

    def fake_prepare(**kwargs: object) -> object:
        calls["prepare"] = kwargs
        return installed_runtime

    def fake_capture(**kwargs: object) -> dict[str, Any]:
        calls["capture"] = kwargs
        return receipt

    monkeypatch.setattr(current_visual, "prepare_installed_habitat_runtime", fake_prepare)
    monkeypatch.setattr(current_visual, "_capture_current_visual", fake_capture)

    assert (
        current_visual.capture_current_visual(
            animal_manifest_path=tmp_path / "animal_manifest.json",
            m2_request_path=tmp_path / "m2_request.json",
            room_manifest_path=tmp_path / "room_manifest.json",
            m1_request_path=tmp_path / "m1_request.json",
            runtime_prefix=runtime_prefix,
            mp3d_root=mp3d_root,
            magnum_python_site=magnum_site,
            output_directory=tmp_path / "capture",
        )
        == receipt
    )
    assert calls["prepare"] == {
        "runtime_prefix": runtime_prefix,
        "mp3d_root": mp3d_root,
        "magnum_python_site": magnum_site,
    }
    assert calls["capture"]["bundle"] == "bundle"
    assert calls["capture"]["frames"] == tuple(range(75))
    assert calls["capture"]["output"] == (tmp_path / "capture").resolve()


def test_current_visual_requires_bullet_before_creating_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    room_inputs = _mp3d_room_inputs(tmp_path)
    m2_inputs = _m2_inputs(tmp_path, room_id=room_inputs.room["room_id"])
    monkeypatch.setattr(current_visual, "load_m2_inputs", lambda *_args: m2_inputs)
    monkeypatch.setattr(current_visual, "load_m1_inputs", lambda *_args: room_inputs)
    monkeypatch.setattr(current_visual, "validate_capture_context", lambda *_args: [])
    monkeypatch.setattr(
        current_visual, "validate_scene_asset_graph", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(current_visual, "load_runtime_asset_bundle", lambda _inputs: "bundle")
    monkeypatch.setattr(
        current_visual, "compile_frame_applications", lambda *_args: tuple(range(75))
    )
    monkeypatch.setattr(
        current_visual,
        "prepare_installed_habitat_runtime",
        lambda **_kwargs: SimpleNamespace(
            habitat_sim=SimpleNamespace(built_with_bullet=False)
        ),
    )

    output = tmp_path / "capture"
    with pytest.raises(current_visual.CurrentVisualError, match="Bullet-enabled"):
        current_visual.capture_current_visual(
            animal_manifest_path=tmp_path / "animal_manifest.json",
            m2_request_path=tmp_path / "m2_request.json",
            room_manifest_path=tmp_path / "room_manifest.json",
            m1_request_path=tmp_path / "m1_request.json",
            runtime_prefix=tmp_path / "prefix",
            mp3d_root=tmp_path / "mp3d",
            magnum_python_site=tmp_path / "magnum",
            output_directory=output,
        )
    assert not output.exists()


def test_current_configuration_has_only_visual_sensors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class SensorSpec:
        semantic_target: object | None = None

    class NavMeshSettings:
        def set_defaults(self) -> None:
            return None

    class SimulatorConfiguration:
        pass

    class AgentConfiguration:
        pass

    def configuration(sim_cfg: object, agents: list[object]) -> object:
        return SimpleNamespace(sim_cfg=sim_cfg, agents=agents)

    def unexpected_audio_sensor() -> object:
        raise AssertionError("current visual path created an audio sensor")

    fake_habitat = SimpleNamespace(
        CameraSensorSpec=SensorSpec,
        NavMeshSettings=NavMeshSettings,
        SimulatorConfiguration=SimulatorConfiguration,
        AgentConfiguration=AgentConfiguration,
        SensorType=SimpleNamespace(COLOR="color", DEPTH="depth", SEMANTIC="semantic"),
        SensorSubType=SimpleNamespace(PINHOLE="pinhole"),
        Configuration=configuration,
        AudioSensorSpec=unexpected_audio_sensor,
    )

    room_inputs = SimpleNamespace(
        room={"navigation": {"agent_height_m": 1.5, "agent_radius_m": 0.1}},
        request={
            "seed": 17,
            "primary_camera_rig": {
                "shared_calibration": {
                    "resolution_hw": [240, 320],
                    "hfov_degrees": 90.0,
                    "near_m": 0.05,
                    "far_m": 100.0,
                    "rig_from_sensor": {
                        "translation_m": [0.0, 0.0, 0.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                "modalities": [
                    {"modality": "rgb", "sensor_uuid": "rig_rgb"},
                    {"modality": "depth", "sensor_uuid": "rig_depth"},
                    {"modality": "semantic", "sensor_uuid": "rig_semantic"},
                ],
            },
        },
    )
    installed_runtime = SimpleNamespace(
        habitat_sim=fake_habitat,
        magnum=SimpleNamespace(
            Vector3=lambda *values: (
                tuple(values[0]) if len(values) == 1 else tuple(values)
            ),
            Vector2i=tuple,
        ),
        physics_config_path=tmp_path / "default.physics_config.json",
    )
    monkeypatch.setattr(current_visual, "_semantic_sensor_target", lambda: "semantic")

    configuration, modality_to_uuid = current_visual._make_current_configuration(
        room_inputs=room_inputs,
        installed_runtime=installed_runtime,
        scene={
            "scene_id": tmp_path / "scene.glb",
            "dataset_config": tmp_path / "dataset.json",
            "load_semantic_mesh": True,
        },
    )

    assert modality_to_uuid == {
        "rgb": "rig_rgb",
        "depth": "rig_depth",
        "semantic": "rig_semantic",
    }
    specs = configuration.agents[0].sensor_specifications
    assert [spec.uuid for spec in specs] == ["rig_rgb", "rig_depth", "rig_semantic"]
    assert configuration.sim_cfg.enable_physics is True
    assert configuration.sim_cfg.physics_config_file == str(
        installed_runtime.physics_config_path
    )
    with pytest.raises(current_visual.CurrentVisualError, match="never enables"):
        current_visual._make_current_configuration(
            room_inputs=room_inputs,
            installed_runtime=installed_runtime,
            scene={
                "scene_id": tmp_path / "scene.glb",
                "dataset_config": tmp_path / "dataset.json",
                "load_semantic_mesh": True,
            },
            include_audio_sensor=True,
        )


@pytest.mark.parametrize(
    "context_error",
    (
        "M2 camera_rig_id differs from the validated M1 rig",
        "M2 listener_id differs from the validated M1 listener",
        "M1 and M2 seeds must match for the reused room configuration",
    ),
)
def test_current_visual_rejects_context_errors_before_runtime_or_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, context_error: str
) -> None:
    room_inputs = _mp3d_room_inputs(tmp_path)
    m2_inputs = _m2_inputs(tmp_path, room_id=room_inputs.room["room_id"])
    monkeypatch.setattr(current_visual, "load_m2_inputs", lambda *_args: m2_inputs)
    monkeypatch.setattr(current_visual, "load_m1_inputs", lambda *_args: room_inputs)
    monkeypatch.setattr(
        current_visual, "validate_capture_context", lambda *_args: [context_error]
    )

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid capture context reached runtime or output setup")

    monkeypatch.setattr(current_visual, "validate_scene_asset_graph", unexpected)
    monkeypatch.setattr(current_visual, "prepare_installed_habitat_runtime", unexpected)
    output = tmp_path / "context-error"
    with pytest.raises(current_visual.CurrentVisualError, match=context_error):
        current_visual.capture_current_visual(
            animal_manifest_path=tmp_path / "animal_manifest.json",
            m2_request_path=tmp_path / "m2_request.json",
            room_manifest_path=tmp_path / "room_manifest.json",
            m1_request_path=tmp_path / "m1_request.json",
            runtime_prefix=tmp_path / "prefix",
            mp3d_root=tmp_path / "mp3d",
            magnum_python_site=tmp_path / "magnum",
            output_directory=output,
        )
    assert not output.exists()


def test_current_visual_rejects_dataset_redirect_before_runtime_or_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    room_inputs = _mp3d_room_inputs(tmp_path)
    m2_inputs = _m2_inputs(tmp_path, room_id=room_inputs.room["room_id"])
    mp3d_root = tmp_path / "mp3d"
    calls: dict[str, Any] = {}
    monkeypatch.setattr(current_visual, "load_m2_inputs", lambda *_args: m2_inputs)
    monkeypatch.setattr(current_visual, "load_m1_inputs", lambda *_args: room_inputs)
    monkeypatch.setattr(current_visual, "validate_capture_context", lambda *_args: [])

    def fake_static_graph(*args: object, **kwargs: object) -> list[str]:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return ["path scene_id is not selected by the dataset stage search paths"]

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dataset redirect reached runtime or output setup")

    monkeypatch.setattr(current_visual, "validate_scene_asset_graph", fake_static_graph)
    monkeypatch.setattr(current_visual, "load_runtime_asset_bundle", unexpected)
    monkeypatch.setattr(current_visual, "prepare_installed_habitat_runtime", unexpected)
    output = tmp_path / "dataset-redirect"
    with pytest.raises(current_visual.CurrentVisualError, match="dataset stage search"):
        current_visual.capture_current_visual(
            animal_manifest_path=tmp_path / "animal_manifest.json",
            m2_request_path=tmp_path / "m2_request.json",
            room_manifest_path=tmp_path / "room_manifest.json",
            m1_request_path=tmp_path / "m1_request.json",
            runtime_prefix=tmp_path / "prefix",
            mp3d_root=mp3d_root,
            magnum_python_site=tmp_path / "magnum",
            output_directory=output,
        )
    assert calls["args"] == (room_inputs, None)
    assert calls["kwargs"] == {"mp3d_root": mp3d_root}
    assert not output.exists()


@pytest.mark.parametrize("semantic_id", current_visual.CURRENT_SEMANTIC_IDS)
def test_current_visual_rejects_preexisting_actor_semantic_ids(
    semantic_id: int,
) -> None:
    semantic = np.zeros((2, 2), dtype=np.int32)
    semantic[0, 0] = semantic_id
    with pytest.raises(current_visual.CurrentVisualError, match="already uses"):
        current_visual._require_no_actor_semantic_collision(semantic)


def test_current_visual_rejects_missing_actor_semantic_id_on_frame_75(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = np.eye(4, dtype=np.float64)
    events: list[str] = []

    class FakePathfinder:
        is_loaded = True

        def load_nav_mesh(self, _path: str) -> bool:
            events.append("navmesh")
            return True

    class FakeAgent:
        def set_state(self, *_args: object, **_kwargs: object) -> None:
            return None

    class FakeTemplateManager:
        def load_configs(self, _path: str) -> list[int]:
            events.append("template")
            return [1]

        def get_template_handles(self, _prefix: str) -> list[str]:
            return ["beagle"]

    class FakeSimulator:
        def __init__(self) -> None:
            self.pathfinder = FakePathfinder()
            self.metadata_mediator = SimpleNamespace(
                ao_template_manager=FakeTemplateManager()
            )
            self.sensors = {
                "rig_rgb": object(),
                "rig_depth": object(),
                "rig_semantic": object(),
            }
            self.render_calls = 0

        def __enter__(self) -> FakeSimulator:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def seed(self, _seed: int) -> None:
            events.append("seed")

        def initialize_agent(self, _index: int, _state: object) -> FakeAgent:
            return FakeAgent()

        def get_world_time(self) -> float:
            return 0.0

        def render_sensors(self, _sensors: list[object]) -> dict[str, np.ndarray]:
            call_index = self.render_calls
            self.render_calls += 1
            semantic = np.zeros((2, 2), dtype=np.int32)
            if call_index:
                semantic[0, 0] = current_visual.CURRENT_SEMANTIC_IDS[0]
                if call_index != 75:
                    semantic[0, 1] = current_visual.CURRENT_SEMANTIC_IDS[1]
            return {
                "rig_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "rig_depth": np.ones((2, 2), dtype=np.float32),
                "rig_semantic": semantic,
            }

    simulator = FakeSimulator()
    habitat_sim = SimpleNamespace(
        Simulator=lambda _configuration: simulator,
        AgentState=SimpleNamespace,
    )
    room_inputs = SimpleNamespace(
        room={"room_id": "habitat_mp3d_example_17DRP5sb8fy"},
        request={
            "seed": 17,
            "primary_camera_rig": {
                "world_from_rig": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            },
        },
    )
    m2_inputs = SimpleNamespace(request={"seed": 17})
    installed_runtime = SimpleNamespace(
        habitat_sim=habitat_sim,
        quaternion=SimpleNamespace(quaternion=lambda *_values: object()),
        magnum=object(),
        quat_to_coeffs=object(),
        mp3d_root=tmp_path / "mp3d",
    )
    frames = tuple(
        SimpleNamespace(
            frame_index=index,
            pts_ticks=index * 3200,
            action_id="walk",
            action_sample_index=index,
            world_from_skin_root=identity,
            joint_rotations_xyzw=((0.0, 0.0, 0.0, 1.0),),
            world_from_actor=identity,
        )
        for index in range(75)
    )
    bundle = SimpleNamespace(
        paths_by_role={"habitat_ao_config": tmp_path / "beagle.ao_config.json"}
    )
    binding = SimpleNamespace(
        map_pose=lambda _rotations: np.asarray([0.0, 0.0, 0.0, 1.0])
    )

    monkeypatch.setattr(
        current_visual,
        "_resolve_external_scene",
        lambda *_args: {
            "scene_id": tmp_path / "scene.glb",
            "dataset_config": tmp_path / "dataset.json",
            "navmesh": tmp_path / "scene.navmesh",
            "load_semantic_mesh": True,
        },
    )
    monkeypatch.setattr(
        current_visual,
        "_make_current_configuration",
        lambda **_kwargs: (
            object(),
            {"rgb": "rig_rgb", "depth": "rig_depth", "semantic": "rig_semantic"},
        ),
    )

    def fake_loaded_graph(*_args: object, **kwargs: object) -> tuple[list[str], dict[str, Any]]:
        assert kwargs["declared_navmesh_loaded"] is True
        assert kwargs["mp3d_root"] == installed_runtime.mp3d_root
        events.append("loaded-graph")
        return [], {}

    def fake_instantiate(
        _simulator: object, **kwargs: object
    ) -> tuple[SimpleNamespace, object]:
        events.append(f"instantiate-{kwargs['actor_index']}")
        return (
            SimpleNamespace(
                root=identity.copy(),
                joint_positions=np.asarray([0.0, 0.0, 0.0, 1.0]),
            ),
            binding,
        )

    def fake_apply(actor: SimpleNamespace, matrix: np.ndarray, **_kwargs: object) -> None:
        actor.root = np.asarray(matrix, dtype=np.float64).copy()

    monkeypatch.setattr(
        current_visual, "validate_loaded_scene_asset_graph", fake_loaded_graph
    )
    monkeypatch.setattr(current_visual, "_instantiate_semantic_actor", fake_instantiate)
    monkeypatch.setattr(current_visual, "_link_id_by_name", lambda *_args: 0)
    monkeypatch.setattr(current_visual, "_apply_root_with_habitat", fake_apply)
    monkeypatch.setattr(
        current_visual,
        "_read_actor_state",
        lambda actor: (actor.root.copy(), actor.joint_positions.copy()),
    )
    monkeypatch.setattr(
        current_visual,
        "_node_world_position",
        lambda *_args: np.zeros(3, dtype=np.float64),
    )
    monkeypatch.setattr(
        current_visual,
        "_state_snapshot",
        lambda *_args: {
            "agent": room_inputs.request["primary_camera_rig"]["world_from_rig"],
            "sensors": {
                "rig_rgb": room_inputs.request["primary_camera_rig"]["world_from_rig"],
                "rig_depth": room_inputs.request["primary_camera_rig"]["world_from_rig"],
                "rig_semantic": room_inputs.request["primary_camera_rig"]["world_from_rig"],
            },
        },
    )
    monkeypatch.setattr(current_visual, "transform_error", lambda *_args: 0.0)

    output = tmp_path / "capture"
    output.mkdir()
    with pytest.raises(
        current_visual.CurrentVisualError,
        match="frame 74 lost an actor semantic ID",
    ):
        current_visual._capture_current_visual(
            m2_inputs=m2_inputs,
            room_inputs=room_inputs,
            installed_runtime=installed_runtime,
            output=output,
            bundle=bundle,
            frames=frames,
        )
    assert simulator.render_calls == 76
    assert events.index("loaded-graph") < events.index("instantiate-0")
    assert not (output / "arrays").exists()



def test_current_module_has_no_checkout_or_v1_receipt_path() -> None:
    source = inspect.getsource(current_visual)
    assert "discover_runtime_root" not in source
    assert "runtime_root" not in source
    assert '"sha256"' not in source
    assert '"schema"' not in source
    assert "AudioSensorSpec" not in source


def test_cli_exposes_visual_only_current_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli.build_parser()
    parsed = parser.parse_args(
        [
            "m5",
            "capture-current-visual",
            "--animal-manifest",
            "animal.json",
            "--m2-request",
            "m2.json",
            "--room-manifest",
            "room.json",
            "--m1-request",
            "m1.json",
            "--runtime-prefix",
            "prefix",
            "--mp3d-root",
            "mp3d",
            "--magnum-python-site",
            "magnum",
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert parsed.m5_command == "capture-current-visual"
    assert not hasattr(parsed, "runtime_root")
    calls: dict[str, Any] = {}

    def fake_capture(**kwargs: object) -> dict[str, Any]:
        calls.update(kwargs)
        return {
            "status": "research_only",
            "research_only": True,
            "episode_counted": False,
            "capture": {"frame_count": 75},
        }

    monkeypatch.setattr(cli, "capture_current_visual", fake_capture)
    output = tmp_path / "output"
    assert (
        cli.main(
            [
                "m5",
                "capture-current-visual",
                "--animal-manifest",
                "animal.json",
                "--m2-request",
                "m2.json",
                "--room-manifest",
                "room.json",
                "--m1-request",
                "m1.json",
                "--runtime-prefix",
                "prefix",
                "--mp3d-root",
                "mp3d",
                "--magnum-python-site",
                "magnum",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert calls == {
        "animal_manifest_path": "animal.json",
        "m2_request_path": "m2.json",
        "room_manifest_path": "room.json",
        "m1_request_path": "m1.json",
        "runtime_prefix": "prefix",
        "mp3d_root": "mp3d",
        "magnum_python_site": "magnum",
        "output_directory": output.resolve(),
    }
    assert json.loads(capsys.readouterr().out) == {
        "episode_counted": False,
        "frame_count": 75,
        "output": str(output.resolve()),
        "receipt": str(output.resolve() / "research_receipt.json"),
        "research_only": True,
        "status": "research_only",
    }

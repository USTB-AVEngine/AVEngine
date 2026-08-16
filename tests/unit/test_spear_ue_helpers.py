"""Focused fake-service coverage for the bounded SPEAR helper slice."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np

from avengine.backends.spear_ue import lighting, rig_direction


REPOSITORY = Path(__file__).resolve().parents[2]


class _FakeRoot:
    def __init__(self) -> None:
        self.mobility: str | None = None

    def SetMobility(self, *, NewMobility: str) -> None:
        self.mobility = NewMobility


class _FakeLightComponent:
    def __init__(self) -> None:
        self.intensity: float | None = None
        self.cast_shadows: bool | None = None

    def SetIntensity(self, *, NewIntensity: float) -> None:
        self.intensity = float(NewIntensity)

    def SetCastShadows(self, *, bNewValue: bool) -> None:
        self.cast_shadows = bool(bNewValue)

    def get_property_value(self, property_name: str) -> Any:
        if property_name == "Intensity":
            return self.intensity
        if property_name == "CastShadows":
            return self.cast_shadows
        raise KeyError(property_name)


class _FakeActor:
    def __init__(self, uclass: str) -> None:
        self.uclass = uclass
        self.root = _FakeRoot()
        self.location_rotation: dict[str, Any] | None = None
        self.component = _FakeLightComponent()

    def K2_GetRootComponent(self) -> _FakeRoot:
        return self.root

    def K2_SetActorLocationAndRotation(self, **kwargs: Any) -> None:
        self.location_rotation = kwargs


class _FakeRigComponent:
    def __init__(self, positions: dict[str, tuple[float, float, float]]) -> None:
        self.names = list(positions)
        self.positions = positions

    def GetNumBones(self) -> dict[str, int]:
        return {"ReturnValue": len(self.names)}

    def GetBoneName(self, *, BoneIndex: int) -> dict[str, str]:
        return {"ReturnValue": self.names[BoneIndex]}

    def GetBoneIndex(self, *, BoneName: str) -> int:
        try:
            return self.names.index(BoneName)
        except ValueError:
            return -1

    def GetBoneTransform(self, **kwargs: Any) -> dict[str, dict[str, dict[str, float]]]:
        name = kwargs["InBoneName"]
        x, y, z = self.positions[name]
        return {"ReturnValue": {"Translation": {"X": x, "Y": y, "Z": z}}}


class _FakeUnrealService:
    def __init__(
        self,
        rig_components: list[_FakeRigComponent] | None = None,
        *,
        fail_spawn_for: set[str] | None = None,
    ) -> None:
        self.rig_components = rig_components or []
        self.fail_spawn_for = fail_spawn_for or set()
        self.spawned: list[tuple[str, _FakeActor]] = []

    def get_components_by_class(self, *, actor: object, uclass: str) -> list[_FakeRigComponent]:
        assert uclass == "USkeletalMeshComponent"
        return self.rig_components

    def spawn_actor(self, *, uclass: str, **kwargs: Any) -> _FakeActor:
        if uclass in self.fail_spawn_for:
            raise RuntimeError(f"fake spawn failure for {uclass}")
        actor = _FakeActor(uclass)
        self.spawned.append((uclass, actor))
        return actor

    def get_component_by_class(self, *, actor: _FakeActor, uclass: str) -> _FakeLightComponent:
        expected = {
            "ADirectionalLight": "UDirectionalLightComponent",
            "ASkyLight": "USkyLightComponent",
        }
        assert expected[actor.uclass] == uclass
        return actor.component


class _FakeGame:
    def __init__(self, unreal_service: _FakeUnrealService) -> None:
        self.unreal_service = unreal_service


def _load_tool(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, REPOSITORY / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _human_rig() -> _FakeRigComponent:
    return _FakeRigComponent(
        {
            "Pelvis": (0.0, 0.0, 100.0),
            "Spine2": (0.0, 0.0, 200.0),
            "LeftClavicle": (0.0, -25.0, 175.0),
            "RightClavicle": (0.0, 25.0, 175.0),
        }
    )


def _lighting_profile() -> dict[str, Any]:
    return {
        "directional_key": {
            "yaw_deg": 35.0,
            "pitch_deg": -25.0,
            "intensity_lux": 7.5,
        },
        "skylight_intensity": 0.75,
        "claim_boundary": "fake test profile",
    }


def test_rig_helpers_use_only_fake_unreal_service() -> None:
    empty = _FakeRigComponent({})
    populated = _human_rig()
    service = _FakeUnrealService([empty, populated])
    diagnostics: list[dict[str, Any]] = []

    assert rig_direction.select_skeletal_mesh_component(
        unreal_service=service, actor=object(), diagnostics=diagnostics
    ) is populated
    np.testing.assert_allclose(
        rig_direction.sample_body_bone_position_in_frame(
            object(), "Pelvis", unreal_service=service, diagnostics=diagnostics
        ),
        [0.0, 0.0, 100.0],
    )
    basis = rig_direction.sample_body_basis_in_frame(
        object(), unreal_service=service, diagnostics=diagnostics
    )

    assert basis is not None
    assert basis["basis_kind"] == "humanoid_semantic_v1"
    assert basis["bone_names"] == {
        "pelvis": "Pelvis",
        "spine": "Spine2",
        "left_clavicle": "LeftClavicle",
        "right_clavicle": "RightClavicle",
    }
    np.testing.assert_allclose(basis["forward_vector_ue"], [1.0, 0.0, 0.0])
    assert diagnostics == []

    quadruped = _FakeRigComponent(
        {
            "Rear": (0.0, 0.0, 50.0),
            "Front": (100.0, 0.0, 50.0),
            "Body": (0.0, 0.0, 100.0),
            "LeftFoot": (0.0, -20.0, 0.0),
            "RightFoot": (0.0, 20.0, 0.0),
        }
    )
    quadruped_service = _FakeUnrealService([quadruped])
    semantic_bone_names = {
        "rear": "Rear",
        "front": "Front",
        "body": "Body",
        "left_foot": "LeftFoot",
        "right_foot": "RightFoot",
    }
    explicit_basis = rig_direction.sample_body_basis_in_frame(
        object(),
        unreal_service=quadruped_service,
        semantic_bone_names=semantic_bone_names,
    )
    assert explicit_basis is not None
    assert explicit_basis["basis_kind"] == (
        "authenticated_generated_quadruped_longitudinal_v1"
    )
    np.testing.assert_allclose(explicit_basis["forward_vector_ue"], [1.0, 0.0, 0.0])
    invalid_diagnostics: list[dict[str, Any]] = []
    assert rig_direction.sample_body_basis_in_frame(
        object(),
        unreal_service=quadruped_service,
        semantic_bone_names={"rear": "Rear"},
        diagnostics=invalid_diagnostics,
    ) is None
    assert invalid_diagnostics[-1]["stage"] == "body_basis"

    assert rig_direction.__all__ == [
        "select_skeletal_mesh_component",
        "sample_body_bone_position_in_frame",
        "sample_body_basis_in_frame",
    ]


def test_lighting_helpers_issue_expected_fake_rpc_calls() -> None:
    service = _FakeUnrealService()
    game = _FakeGame(service)

    directional = lighting.spawn_directional_light(
        game, yaw_deg=20.0, pitch_deg=-15.0, intensity_lux=1234.0
    )
    assert directional.uclass == "ADirectionalLight"
    assert directional.root.mobility == "Movable"
    assert directional.location_rotation == {
        "NewLocation": {"X": 0.0, "Y": 0.0, "Z": 500.0},
        "NewRotation": {"Roll": 0.0, "Pitch": -15.0, "Yaw": 20.0},
        "bSweep": False,
        "bTeleport": True,
    }
    assert directional.component.intensity == 1234.0

    sky = lighting.spawn_sky(game)
    assert set(sky) == {"ASkyAtmosphere", "ASkyLight", "AExponentialHeightFog"}
    partial_sky = lighting.spawn_sky(
        _FakeGame(_FakeUnrealService(fail_spawn_for={"AExponentialHeightFog"}))
    )
    assert set(partial_sky) == {"ASkyAtmosphere", "ASkyLight"}
    assert lighting.__all__ == ["spawn_directional_light", "spawn_sky"]


def test_runners_need_no_external_examples_or_rig_tools(tmp_path: Path) -> None:
    spear_root = tmp_path / "external-spear"
    spear_root.mkdir()
    assert not (spear_root / "examples").exists()
    assert not (spear_root / "tools" / "spike_rlr").exists()
    original_sys_path = list(sys.path)

    apartment = _load_tool(
        "s3d_apartment_runner", "tools/m6y/run_spear_apartment_canary.py"
    )
    mp3d = _load_tool("s3d_mp3d_runner", "tools/m6y/run_spear_mp3d_canary.py")
    adapter = _load_tool(
        "s3d_glb_adapter", "tools/qa/spear_imported_glb_room_adapter.py"
    )
    for relative_path in (
        "tools/m6y/run_spear_apartment_canary.py",
        "tools/m6y/run_spear_mp3d_canary.py",
        "tools/qa/spear_imported_glb_room_adapter.py",
    ):
        source = (REPOSITORY / relative_path).read_text(encoding="utf-8")
        assert "sys.path.insert" not in source
        assert "rig_direction_check" not in source
        assert "render_in_gpurir_room" not in source

    assert list(inspect.signature(apartment._spawn_runtime_actors).parameters) == [
        "game",
        "scenario",
        "spear_root",
    ]
    assert list(inspect.signature(mp3d._spawn_runtime_actors).parameters) == [
        "game",
        "spear_root",
        "plan",
    ]
    packaged_capture = (
        REPOSITORY / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py"
    ).read_text(encoding="utf-8")
    assert "runner._spawn_runtime_actors(game, scenario, _spear_root)" in packaged_capture

    service = _FakeUnrealService([_human_rig()])
    game = _FakeGame(service)
    assert apartment._load_skeletal_component(game, object()) is service.rig_components[0]

    profile = _lighting_profile()
    mp3d_result = mp3d._spawn_lighting(game, {"exposure_and_lighting": profile})
    adapter_result = adapter.spawn_review_lighting(game, profile)
    assert mp3d_result["status"] == "pass"
    assert adapter_result["status"] == "pass"
    assert sys.path == original_sys_path
    assert str(spear_root / "examples") not in sys.path
    assert str(spear_root / "tools" / "spike_rlr") not in sys.path

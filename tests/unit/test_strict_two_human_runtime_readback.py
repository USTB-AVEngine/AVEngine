from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/capture_spear_native_pixel_episode.py"
SPEC = importlib.util.spec_from_file_location("strict_runtime_readback", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class _LoadedMesh:
    def GetSkeleton(self, *, as_handle: bool) -> int:
        assert as_handle is True
        return 300


class _AnimInstance:
    def GetAnimationAsset(self, *, as_handle: bool) -> int:
        assert as_handle is True
        return 400

    def get_property_value(self, *, property_name: str, as_handle: bool) -> int:
        assert property_name == "Skeleton"
        assert as_handle is True
        return 300


class _UnrealService:
    def __init__(self, *, class_match: bool = True) -> None:
        self.class_match = class_match

    def load_class(self, **kwargs: object) -> int:
        assert kwargs["name"] == "/Game/Test/BP_Human.BP_Human_C"
        assert kwargs["as_handle"] is True
        return 100

    def get_class(self, **kwargs: object) -> int:
        assert isinstance(kwargs["uobject"], _VisualActor)
        assert kwargs["as_handle"] is True
        return 100 if self.class_match else 101

    def load_object(self, **kwargs: object) -> object:
        name = kwargs["name"]
        if kwargs.get("as_unreal_object") is True:
            assert name == "/Game/Test/runtime.runtime"
            return _LoadedMesh()
        handles = {
            "/Game/Test/runtime.runtime": 200,
            "/Game/Test/runtime_Skeleton.runtime_Skeleton": 300,
            "/Game/Test/Standing_Idle.Standing_Idle": 400,
        }
        assert kwargs["as_handle"] is True
        return handles[name]


class _Game:
    def __init__(self, *, class_match: bool = True) -> None:
        self.unreal_service = _UnrealService(class_match=class_match)

    def get_unreal_object(self, *, uobject: int) -> object:
        if uobject == 200:
            return _LoadedMesh()
        assert uobject == 500
        return _AnimInstance()


class _VisualActor:
    pass


class _Animation:
    uobject = 400


class _Component:
    def GetAnimInstance(self, *, as_handle: bool) -> int:
        assert as_handle is True
        return 500

    def get_property_value(self, *, property_name: str, as_handle: bool) -> int:
        assert property_name == "AnimScriptInstance"
        assert as_handle is True
        return 500

    def GetPosition(self) -> float:
        return 0.0


def _case() -> tuple[dict, dict, dict, dict, list]:
    idle = "/Game/Test/Standing_Idle.Standing_Idle"
    scenario = {
        "plan": {
            "actors": [
                {
                    "actor_id": "source1_actor",
                    "asset_id": "human_01",
                    "asset_revision": "v1",
                    "blueprint_class_path": "/Game/Test/BP_Human.BP_Human_C",
                    "skeletal_mesh_path": "/Game/Test/runtime.runtime",
                    "skeleton_path": "/Game/Test/runtime_Skeleton.runtime_Skeleton",
                    "idle_animation": idle,
                    "emitter_anchor_id": "mouth",
                    "emitter_offset_m": [0.0, 1.6, 0.0],
                }
            ]
        }
    }
    frame = {
        "frame_index": 15,
        "actor_states": [
            {
                "actor_id": "source1_actor",
                "action_id": "idle",
                "action_phase": 0.0,
                "ue_animation": idle,
                "translation_m": [-2.0, 0.4, -1.0],
            }
        ],
    }
    runtimes = {
        "source1_actor": {
            "anchor": object(),
            "visual_actor": _VisualActor(),
            "component": _Component(),
            "animations": {idle: _Animation()},
            "lengths": {idle: 2.0},
            "current_animation": idle,
        }
    }
    stable_names = {"source1": "lead_a_native_source1_actor"}
    descriptors = [
        {
            "actorStableName": "lead_a_native_source1_actor",
            "rawId": 17,
        }
    ]
    return scenario, frame, runtimes, stable_names, descriptors


def test_runtime_asset_readback_closes_live_identity_and_emitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario, frame, runtimes, stable_names, descriptors = _case()
    monkeypatch.setattr(
        TOOL.RUNNER,
        "_skeletal_mesh_handle",
        lambda _: (200, "GetSkeletalMeshAsset"),
    )
    monkeypatch.setattr(
        TOOL.RUNNER,
        "_actor_readback",
        lambda _actor, frame_index: {
            "frame_index": frame_index,
            "location_cm": [-200.0, -100.0, 40.0],
            "rotation_deg": [0.0, 0.0, -38.0],
        },
    )

    result = TOOL._runtime_asset_readbacks(
        game=_Game(),
        scenario=scenario,
        runtimes=runtimes,
        stable_names=stable_names,
        raw_descriptors=descriptors,
        frame=frame,
    )

    assert result["status"] == "pass"
    observed = result["per_instance"]["source1"]
    assert observed["blueprint"]["status"] == "pass"
    assert observed["skeletal_mesh"]["observed_handle"] == 200
    assert observed["skeleton"]["observed_mesh_skeleton_handle"] == 300
    assert observed["standing_idle"]["runtime_loaded_handle"] == 400
    assert observed["standing_idle"]["observed_animation_asset_handle"] == 400
    assert observed["emitter_native_readback"]["observed_world_emitter_m"] == [
        -2.0,
        2.0,
        -1.0,
    ]


def test_runtime_asset_readback_rejects_wrong_live_blueprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario, frame, runtimes, stable_names, descriptors = _case()
    monkeypatch.setattr(
        TOOL.RUNNER,
        "_skeletal_mesh_handle",
        lambda _: (200, "GetSkeletalMeshAsset"),
    )

    with pytest.raises(RuntimeError, match="live Blueprint class mismatch"):
        TOOL._runtime_asset_readbacks(
            game=_Game(class_match=False),
            scenario=scenario,
            runtimes=runtimes,
            stable_names=stable_names,
            raw_descriptors=descriptors,
            frame=frame,
        )

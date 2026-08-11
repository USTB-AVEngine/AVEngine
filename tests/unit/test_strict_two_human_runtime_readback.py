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
            "/Game/Test/Walking.Walking": 401,
        }
        assert kwargs["as_handle"] is True
        return handles[name]


class _Kismet:
    def __init__(self, *, ground_hit: bool = True) -> None:
        self.ground_hit = ground_hit

    def LineTraceSingleByProfile(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["ProfileName"] == "BlockAll"
        assert kwargs["bTraceComplex"] is True
        assert len(kwargs["ActorsToIgnore"]) == 2
        start = kwargs["Start"]
        return {
            "ReturnValue": self.ground_hit,
            "OutHit": {
                "actor": "ApartmentFloorActor",
                "component": "ApartmentFloorComponent",
                "location": {"x": start["X"], "y": start["Y"], "z": 40.0},
                "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
        }


class _Game:
    def __init__(self, *, class_match: bool = True, ground_hit: bool = True) -> None:
        self.unreal_service = _UnrealService(class_match=class_match)
        self.kismet = _Kismet(ground_hit=ground_hit)

    def get_unreal_object(
        self, *, uobject: int | None = None, uclass: str | None = None
    ) -> object:
        if uclass is not None:
            assert uclass == "UKismetSystemLibrary"
            return self.kismet
        if uobject == 200:
            return _LoadedMesh()
        assert uobject == 500
        return _AnimInstance()


class _Anchor:
    def __init__(self) -> None:
        self.location = [-200.0, -100.0, 40.0]

    def K2_GetActorLocation(self, *, as_dict: bool) -> dict[str, object]:
        assert as_dict is True
        return {
            "X": self.location[0],
            "Y": self.location[1],
            "Z": self.location[2],
        }

    def K2_GetActorRotation(self, *, as_dict: bool) -> dict[str, float]:
        assert as_dict is True
        return {"Roll": 0.0, "Pitch": 0.0, "Yaw": -38.0}


class _VisualActor:
    def __init__(self, *, bottom_z_cm: float = 45.0) -> None:
        self.bottom_z_cm = bottom_z_cm
        self.visual_z_offset_cm = 0.0

    def GetActorBounds(
        self, *, bOnlyCollidingComponents: bool, as_dict: bool
    ) -> dict[str, object]:
        assert bOnlyCollidingComponents is False
        assert as_dict is True
        extent_z = 90.0
        return {
            "Origin": {
                "X": -200.0,
                "Y": -100.0,
                "Z": self.bottom_z_cm + self.visual_z_offset_cm + extent_z,
            },
            "BoxExtent": {"X": 25.0, "Y": 25.0, "Z": extent_z},
        }


class _VisualRoot:
    def __init__(
        self,
        visual_actor: _VisualActor,
        *,
        anchor: _Anchor | None = None,
        apply_scale: float = 1.0,
        mutate_anchor_cm: float = 0.0,
    ) -> None:
        self.visual_actor = visual_actor
        self.anchor = anchor
        self.apply_scale = apply_scale
        self.mutate_anchor_cm = mutate_anchor_cm

    def K2_AddRelativeLocation(self, **kwargs: object) -> None:
        delta = float(kwargs["DeltaLocation"]["Z"])
        self.visual_actor.visual_z_offset_cm += delta * self.apply_scale
        if self.anchor is not None:
            self.anchor.location[2] += self.mutate_anchor_cm


class _Animation:
    uobject = 400


class _WalkingAnimation:
    uobject = 401


class _Component:
    def __init__(self) -> None:
        self.bone_names = [
            "Bip01 L Foot",
            "Bip01 L Toe0",
            "Bip01 R Foot",
            "Bip01 R Toe0",
        ]

    def GetAnimInstance(self, *, as_handle: bool) -> int:
        assert as_handle is True
        return 500

    def get_property_value(self, *, property_name: str, as_handle: bool) -> int:
        assert property_name == "AnimScriptInstance"
        assert as_handle is True
        return 500

    def GetPosition(self) -> float:
        return 0.0

    def GetNumBones(self) -> int:
        return len(self.bone_names)

    def GetBoneName(self, *, BoneIndex: int) -> str:
        return self.bone_names[BoneIndex]

    def GetBoneIndex(self, *, BoneName: str) -> int:
        try:
            return self.bone_names.index(BoneName)
        except ValueError:
            return -1

    def GetBoneTransform(
        self, *, InBoneName: str, TransformSpace: str, as_dict: bool
    ) -> dict[str, object]:
        assert TransformSpace == "RTS_World"
        assert as_dict is True
        positions = {
            "Bip01 L Foot": [-201.0, -101.0, 46.0],
            "Bip01 L Toe0": [-202.0, -102.0, 42.0],
            "Bip01 R Foot": [-199.0, -99.0, 47.0],
            "Bip01 R Toe0": [-198.0, -98.0, 43.0],
        }
        x, y, z = positions[InBoneName]
        return {
            "ReturnValue": {
                "Translation": {"X": x, "Y": y, "Z": z},
            }
        }


class _WalkingComponent(_Component):
    def GetPosition(self) -> float:
        return 0.5


def _case() -> tuple[dict, dict, dict, dict, list]:
    idle = "/Game/Test/Standing_Idle.Standing_Idle"
    walking = "/Game/Test/Walking.Walking"
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
                    "walking_animation": walking,
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
    visual_actor = _VisualActor()
    runtimes = {
        "source1_actor": {
            "anchor": _Anchor(),
            "visual_actor": visual_actor,
            "visual_root": _VisualRoot(visual_actor),
            "component": _Component(),
            "animations": {idle: _Animation(), walking: _WalkingAnimation()},
            "lengths": {idle: 2.0, walking: 2.0},
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


def _ground_snap_declaration(tmp_path: Path) -> dict[str, object]:
    authority = tmp_path / "normalization_manifest.json"
    authority.write_text(
        """{
          "expected_ue_qa": {
            "ground_snap_to_floor": true,
            "ground_snap_max_abs_correction_cm": 15.0,
            "ground_snap_residual_tolerance_cm": 0.1
          },
          "runtime_motion_contract": {
            "dynamic_ground_snap_to_floor_required": true
          }
        }""",
        encoding="utf-8",
    )
    return {
        "actor_id": "source1_actor",
        "ground_contact_release_profile": {
            "runtime_visual_ground_snap": {
                "schema": "ue_dynamic_ground_snap_v1",
                "target": "attached_visual_actor_root_component",
                "timeline_anchor_mutation_allowed": False,
                "emitter_or_rir_mutation_allowed": False,
                "maximum_abs_correction_cm": 15.0,
                "residual_tolerance_cm": 0.1,
                "normalization_manifest_authority": str(authority),
            }
        },
    }


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
    assert observed["current_action"]["action_id"] == "idle"
    assert observed["current_action"]["observed_animation_asset_handle"] == 400
    assert observed["emitter_native_readback"]["observed_world_emitter_m"] == [
        -2.0,
        2.0,
        -1.0,
    ]
    ground = observed["live_ground_contact_readback"]
    assert ground["status"] == "pass_instrumented_measurement_only"
    assert ground["ue_length_unit"] == "centimeter"
    assert ground["sides"]["left"]["minimum_bone_to_floor_clearance_cm"] == 2.0
    assert ground["sides"]["right"]["minimum_bone_to_floor_clearance_cm"] == 3.0
    assert ground["sides"]["left"]["anchors"]["foot"]["floor_trace"] == {
        "status": "hit",
        "profile_name": "BlockAll",
        "trace_complex": True,
        "actors_to_ignore_count": 2,
        "start_ue_cm": [-201.0, -101.0, 71.0],
        "end_ue_cm": [-201.0, -101.0, -229.0],
        "hit_actor": "ApartmentFloorActor",
        "hit_component": "ApartmentFloorComponent",
        "hit_point_ue_cm": [-201.0, -101.0, 40.0],
        "hit_normal_ue": [0.0, 0.0, 1.0],
        "horizontal_error_cm": 0.0,
    }


def test_runtime_asset_readback_accepts_declared_walking_at_last_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario, frame, runtimes, stable_names, descriptors = _case()
    walking = "/Game/Test/Walking.Walking"
    frame["frame_index"] = 74
    frame["actor_states"][0].update(
        {
            "action_id": "walk",
            "action_phase": 0.25,
            "ue_animation": walking,
        }
    )
    runtime = runtimes["source1_actor"]
    runtime["component"] = _WalkingComponent()
    runtime["current_animation"] = walking
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
    monkeypatch.setattr(
        _AnimInstance,
        "GetAnimationAsset",
        lambda self, *, as_handle: 401,
    )

    result = TOOL._runtime_asset_readbacks(
        game=_Game(),
        scenario=scenario,
        runtimes=runtimes,
        stable_names=stable_names,
        raw_descriptors=descriptors,
        frame=frame,
    )

    observed = result["per_instance"]["source1"]
    assert observed["standing_idle"]["status"] == "pass"
    assert observed["standing_idle"]["runtime_loaded_handle"] == 400
    assert observed["current_action"]["action_id"] == "walk"
    assert observed["current_action"]["expected_handle"] == 401
    assert observed["current_action"]["observed_animation_asset_handle"] == 401
    assert observed["current_action"]["absolute_position_error_seconds"] == 0.0


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


def test_runtime_ground_contact_rejects_missing_exact_toe_bone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario, frame, runtimes, stable_names, descriptors = _case()
    monkeypatch.setattr(
        TOOL.RUNNER,
        "_skeletal_mesh_handle",
        lambda _: (200, "GetSkeletalMeshAsset"),
    )
    runtimes["source1_actor"]["component"].bone_names = [
        "Bip01 L Foot",
        "Bip01 R Foot",
        "Bip01 R Toe0",
    ]
    with pytest.raises(RuntimeError, match="Bip01 L Toe0.*exactly once"):
        TOOL._runtime_asset_readbacks(
            game=_Game(),
            scenario=scenario,
            runtimes=runtimes,
            stable_names=stable_names,
            raw_descriptors=descriptors,
            frame=frame,
        )


def test_runtime_ground_contact_rejects_floor_trace_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario, frame, runtimes, stable_names, descriptors = _case()
    monkeypatch.setattr(
        TOOL.RUNNER,
        "_skeletal_mesh_handle",
        lambda _: (200, "GetSkeletalMeshAsset"),
    )
    with pytest.raises(RuntimeError, match="downward trace did not hit"):
        TOOL._runtime_asset_readbacks(
            game=_Game(ground_hit=False),
            scenario=scenario,
            runtimes=runtimes,
            stable_names=stable_names,
            raw_descriptors=descriptors,
            frame=frame,
        )


def test_runtime_asset_samples_close_begin_midpoint_and_end() -> None:
    samples = [
        {
            "schema": "avengine_native_spear_runtime_asset_readbacks_v1",
            "status": "pass",
            "frame_index": frame_index,
            "per_instance": {
                "source1": {
                    "current_action": {
                        "absolute_position_error_seconds": 0.0,
                    }
                }
            },
        }
        for frame_index in (0, 37, 74)
    ]

    result = TOOL._bundle_runtime_asset_samples(samples)

    assert result["frame_index"] == 74
    assert result["sampling_contract"]["status"] == "pass"
    assert result["sampling_contract"]["frame_indices"] == [0, 37, 74]
    assert [sample["frame_index"] for sample in result["sampled_frames"]] == [
        0,
        37,
        74,
    ]


def test_runtime_asset_samples_reject_missing_midpoint() -> None:
    samples = [
        {"status": "pass", "frame_index": frame_index} for frame_index in (0, 36, 74)
    ]
    with pytest.raises(RuntimeError, match="sample frame closure"):
        TOOL._bundle_runtime_asset_samples(samples)


def test_runtime_visual_ground_snap_moves_only_attached_visual_root(
    tmp_path: Path,
) -> None:
    _, _, runtimes, _, _ = _case()
    runtime = runtimes["source1_actor"]
    anchor = runtime["anchor"]
    visual_actor = runtime["visual_actor"]
    runtime["visual_root"] = _VisualRoot(visual_actor, anchor=anchor)
    anchor_before = list(anchor.location)

    result = TOOL._apply_runtime_visual_ground_snap(
        game=_Game(),
        runtimes=runtimes,
        declaration=_ground_snap_declaration(tmp_path),
        actor_id="source1_actor",
        frame_index=37,
    )

    assert result["status"] == "passed"
    assert result["visual_bounds_before"]["bottom_z_ue_cm"] == 45.0
    assert result["applied_z_correction_cm"] == -5.0
    assert result["visual_bounds_after"]["bottom_z_ue_cm"] == 40.0
    assert result["residual_clearance_cm"] == 0.0
    assert result["timeline_anchor_mutated"] is False
    assert result["emitter_or_rir_mutated"] is False
    assert anchor.location == anchor_before


def test_runtime_visual_ground_snap_rejects_over_15cm_correction(
    tmp_path: Path,
) -> None:
    _, _, runtimes, _, _ = _case()
    runtimes["source1_actor"]["visual_actor"].bottom_z_cm = 60.01

    with pytest.raises(RuntimeError, match="correction.*exceeds"):
        TOOL._apply_runtime_visual_ground_snap(
            game=_Game(),
            runtimes=runtimes,
            declaration=_ground_snap_declaration(tmp_path),
            actor_id="source1_actor",
            frame_index=0,
        )


def test_runtime_visual_ground_snap_rejects_residual(
    tmp_path: Path,
) -> None:
    _, _, runtimes, _, _ = _case()
    runtime = runtimes["source1_actor"]
    runtime["visual_root"] = _VisualRoot(runtime["visual_actor"], apply_scale=0.5)

    with pytest.raises(RuntimeError, match="residual.*exceeds"):
        TOOL._apply_runtime_visual_ground_snap(
            game=_Game(),
            runtimes=runtimes,
            declaration=_ground_snap_declaration(tmp_path),
            actor_id="source1_actor",
            frame_index=0,
        )


def test_runtime_visual_ground_snap_rejects_anchor_mutation(
    tmp_path: Path,
) -> None:
    _, _, runtimes, _, _ = _case()
    runtime = runtimes["source1_actor"]
    runtime["visual_root"] = _VisualRoot(
        runtime["visual_actor"],
        anchor=runtime["anchor"],
        mutate_anchor_cm=0.01,
    )

    with pytest.raises(RuntimeError, match="mutated the Timeline/acoustic anchor"):
        TOOL._apply_runtime_visual_ground_snap(
            game=_Game(),
            runtimes=runtimes,
            declaration=_ground_snap_declaration(tmp_path),
            actor_id="source1_actor",
            frame_index=0,
        )

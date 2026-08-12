from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/capture_spear_native_pixel_episode.py"
SPEC = importlib.util.spec_from_file_location("strict_runtime_readback", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)

RAW_FLOOR_COMPONENT = (
    "StaticMeshComponent'/Game/Test/Maps/Test.Test:PersistentLevel."
    "ApartmentFloorActor_0.FloorComponent0'"
)


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
        assert kwargs["as_handle"] is True
        uobject = kwargs["uobject"]
        if isinstance(uobject, _VisualActor):
            return 100 if self.class_match else 101
        if isinstance(uobject, _FloorComponent):
            return 6000
        assert isinstance(uobject, _FloorOwner)
        return 7000

    def get_type_for_class_as_string(self, *, uclass: int) -> str:
        return {
            6000: "UStaticMeshComponent",
            7000: "AStaticMeshActor",
        }[uclass]

    def get_stable_name_for_component(self, **kwargs: object) -> str:
        assert isinstance(kwargs["component"], _FloorComponent)
        assert kwargs["include_actor_stable_name"] is True
        assert kwargs["include_actor_unreal_name"] is True
        return "ApartmentFloorActor:PersistentLevel.FloorComponent0"

    def get_stable_name_for_actor(self, **kwargs: object) -> str:
        assert kwargs["include_unreal_name"] is True
        if isinstance(kwargs["actor"], _OtherFloorOwner):
            return "OtherFloorActor:PersistentLevel.OtherFloorActor_0"
        assert isinstance(kwargs["actor"], _FloorOwner)
        return "ApartmentFloorActor:PersistentLevel.ApartmentFloorActor_0"

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


class _FloorOwner:
    pass


class _OtherFloorOwner:
    pass


class _FloorComponent:
    def __init__(self, *, owner_present: bool = True) -> None:
        self.owner_present = owner_present

    def GetOwner(self, *, as_handle: bool) -> str:
        assert as_handle is True
        return "0x2bc" if self.owner_present else "0x0"


class _Kismet:
    def __init__(
        self,
        *,
        ground_hit: bool = True,
        component_present: bool = True,
        actor_field_present: bool = False,
    ) -> None:
        self.ground_hit = ground_hit
        self.component_present = component_present
        self.actor_field_present = actor_field_present

    def LineTraceSingleByProfile(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["ProfileName"] == "BlockAll"
        assert kwargs["bTraceComplex"] is True
        assert len(kwargs["ActorsToIgnore"]) == 2
        start = kwargs["Start"]
        out_hit = {
            "HitObjectHandle": {"Index": 12},
            "PhysMaterial": "0x321",
            "location": {"x": start["X"], "y": start["Y"], "z": 40.0},
            "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
        }
        if self.component_present:
            out_hit["Component"] = RAW_FLOOR_COMPONENT
        if self.actor_field_present:
            out_hit["Actor"] = "WrongLegacyActorField"
        return {
            "ReturnValue": self.ground_hit,
            "OutHit": out_hit,
        }


class _GameplayStatics:
    def __init__(
        self,
        *,
        component_handle: str | None = "0x258",
        actor_handle: str | None = "0x2bc",
        journal_path: Path | None = None,
    ) -> None:
        self.component_handle = component_handle
        self.actor_handle = actor_handle
        self.journal_path = journal_path

    def BreakHitResult(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["as_dict"] is True
        hit = kwargs["Hit"]
        assert isinstance(hit, dict)
        assert hit["Component"] == RAW_FLOOR_COMPONENT
        if self.journal_path is not None:
            journal = json.loads(self.journal_path.read_text(encoding="utf-8"))
            assert journal["status"] == "in_progress"
            assert journal["entries"][-1]["raw_component"] == {
                "present": True,
                "key": "Component",
                "python_type": "builtins.str",
                "literal": RAW_FLOOR_COMPONENT,
                "literal_persisted_exactly": True,
                "identity_authority": False,
            }
            assert journal["entries"][-1]["break_hit_result"] is None
        result: dict[str, object] = {
            "Location": hit["location"],
            "Normal": hit["normal"],
        }
        if self.component_handle is not None:
            result["HitComponent"] = self.component_handle
        if self.actor_handle is not None:
            result["HitActor"] = self.actor_handle
        return result


class _Game:
    def __init__(
        self,
        *,
        class_match: bool = True,
        ground_hit: bool = True,
        component_present: bool = True,
        owner_present: bool = True,
        actor_field_present: bool = False,
        broken_component_handle: str | None = "0x258",
        broken_actor_handle: str | None = "0x2bc",
        journal_path: Path | None = None,
    ) -> None:
        self.unreal_service = _UnrealService(class_match=class_match)
        self.kismet = _Kismet(
            ground_hit=ground_hit,
            component_present=component_present,
            actor_field_present=actor_field_present,
        )
        self.floor_component = _FloorComponent(owner_present=owner_present)
        self.floor_owner = _FloorOwner()
        self.other_floor_owner = _OtherFloorOwner()
        self.gameplay_statics = _GameplayStatics(
            component_handle=broken_component_handle,
            actor_handle=broken_actor_handle,
            journal_path=journal_path,
        )

    def get_unreal_object(
        self, *, uobject: int | None = None, uclass: str | None = None
    ) -> object:
        if uclass is not None:
            if uclass == "UKismetSystemLibrary":
                return self.kismet
            assert uclass == "UGameplayStatics"
            return self.gameplay_statics
        if uobject == 200:
            return _LoadedMesh()
        if uobject == "0x258":
            return self.floor_component
        if uobject == "0x2bc":
            return self.floor_owner
        if uobject == "0x3e7":
            return self.other_floor_owner
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
            "Bip01-L-Foot",
            "Bip01-L-Toe0",
            "Bip01-R-Foot",
            "Bip01-R-Toe0",
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
            "Bip01-L-Foot": [-201.0, -101.0, 46.0],
            "Bip01-L-Toe0": [-202.0, -102.0, 42.0],
            "Bip01-R-Foot": [-199.0, -99.0, 47.0],
            "Bip01-R-Toe0": [-198.0, -98.0, 43.0],
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
    left_foot = ground["sides"]["left"]["anchors"]["foot"]
    assert left_foot["requested_bone_name"] == "Bip01 L Foot"
    assert left_foot["actual_live_bone_name"] == "Bip01-L-Foot"
    assert left_foot["bone_name_resolution_mode"] == ("sanitized_live_fname_required")
    assert ground["bone_name_resolution"]["bone_count"] == 4
    assert ground["sides"]["left"]["minimum_bone_to_floor_clearance_cm"] == 2.0
    assert ground["sides"]["right"]["minimum_bone_to_floor_clearance_cm"] == 3.0
    trace = ground["sides"]["left"]["anchors"]["foot"]["floor_trace"]
    assert trace["status"] == "hit"
    assert trace["profile_name"] == "BlockAll"
    assert trace["trace_complex"] is True
    assert trace["actors_to_ignore_count"] == 2
    assert trace["start_ue_cm"] == [-201.0, -101.0, 71.0]
    assert trace["end_ue_cm"] == [-201.0, -101.0, -229.0]
    assert trace["hit_actor"] == (
        "ApartmentFloorActor:PersistentLevel.ApartmentFloorActor_0"
    )
    assert trace["hit_actor_class"] == "AStaticMeshActor"
    assert trace["hit_component"] == (
        "ApartmentFloorActor:PersistentLevel.FloorComponent0"
    )
    assert trace["hit_component_class"] == "UStaticMeshComponent"
    assert trace["hit_point_ue_cm"] == [-201.0, -101.0, 40.0]
    assert trace["hit_normal_ue"] == [0.0, 0.0, 1.0]
    assert trace["horizontal_error_cm"] == 0.0
    assert trace["schema"] == "ue_fhitresult_component_owner_floor_identity_v3"
    assert trace["authority"] == (
        "UGameplayStatics.BreakHitResult(HitComponent)_to_UActorComponent.GetOwner"
    )
    assert trace["raw_component_diagnostic"] == {
        "present": True,
        "key": "Component",
        "python_type": "builtins.str",
        "literal": RAW_FLOOR_COMPONENT,
        "literal_persisted_exactly": True,
        "identity_authority": False,
    }
    assert trace["break_hit_result_component"] == {
        "present": True,
        "key": "HitComponent",
        "python_type": "builtins.str",
        "literal": "0x258",
        "literal_persisted_exactly": True,
        "identity_authority": True,
    }
    assert trace["raw_actor_field"] == {
        "present": False,
        "value_type": None,
        "value": None,
        "identity_authority": False,
    }
    assert trace["raw_out_hit_shape"] == {
        "keys": [
            "Component",
            "HitObjectHandle",
            "PhysMaterial",
            "location",
            "normal",
        ],
        "value_types": {
            "Component": "str",
            "HitObjectHandle": "dict",
            "PhysMaterial": "str",
            "location": "dict",
            "normal": "dict",
        },
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
        "Bip01-L-Foot",
        "Bip01-R-Foot",
        "Bip01-R-Toe0",
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


def test_floor_trace_v3_accepts_break_component_owner_without_actor_field() -> None:
    result = TOOL._line_trace_floor(
        game=_Game(actor_field_present=False),
        position_ue_cm=[1.0, 2.0, 46.0],
        actors_to_ignore=[object(), object()],
        owner="source1_actor Bip01 L Foot",
    )

    assert result["schema"] == "ue_fhitresult_component_owner_floor_identity_v3"
    assert result["hit_actor"] == (
        "ApartmentFloorActor:PersistentLevel.ApartmentFloorActor_0"
    )
    assert result["hit_component"] == (
        "ApartmentFloorActor:PersistentLevel.FloorComponent0"
    )
    assert result["raw_actor_field"]["present"] is False
    assert result["hit_object_handle_auxiliary"]["identity_authority"] is False


def test_floor_trace_v3_ignores_legacy_actor_field_for_identity() -> None:
    result = TOOL._line_trace_floor(
        game=_Game(actor_field_present=True),
        position_ue_cm=[1.0, 2.0, 46.0],
        actors_to_ignore=[object(), object()],
        owner="source1_actor Bip01 L Foot",
    )

    assert result["raw_actor_field"]["value"] == "WrongLegacyActorField"
    assert result["raw_actor_field"]["identity_authority"] is False
    assert result["hit_actor"] != result["raw_actor_field"]["value"]


def test_floor_trace_v3_accepts_null_break_actor_and_uses_component_owner() -> None:
    result = TOOL._line_trace_floor(
        game=_Game(broken_actor_handle="0x0"),
        position_ue_cm=[1.0, 2.0, 46.0],
        actors_to_ignore=[object(), object()],
        owner="source1_actor Bip01 L Foot",
    )

    assert result["hit_actor"] == (
        "ApartmentFloorActor:PersistentLevel.ApartmentFloorActor_0"
    )
    assert result["break_hit_result_actor_auxiliary"]["literal"] == "0x0"
    assert result["break_hit_result_actor_auxiliary"]["stable_name"] is None
    assert result["break_hit_result_actor_auxiliary"]["identity_authority"] is False


def test_floor_trace_v3_rejects_missing_component() -> None:
    with pytest.raises(RuntimeError, match="lacks a non-handle Component string"):
        TOOL._line_trace_floor(
            game=_Game(component_present=False),
            position_ue_cm=[1.0, 2.0, 46.0],
            actors_to_ignore=[object(), object()],
            owner="source1_actor Bip01 L Foot",
        )


def test_floor_trace_v3_rejects_component_with_null_owner() -> None:
    with pytest.raises(RuntimeError, match="Component.GetOwner returned null"):
        TOOL._line_trace_floor(
            game=_Game(owner_present=False),
            position_ue_cm=[1.0, 2.0, 46.0],
            actors_to_ignore=[object(), object()],
            owner="source1_actor Bip01 L Foot",
        )


def test_floor_trace_v3_rejects_break_actor_that_differs_from_component_owner() -> None:
    with pytest.raises(RuntimeError, match="HitActor differs from Component.GetOwner"):
        TOOL._line_trace_floor(
            game=_Game(broken_actor_handle="0x3e7"),
            position_ue_cm=[1.0, 2.0, 46.0],
            actors_to_ignore=[object(), object()],
            owner="source1_actor Bip01 L Foot",
        )


def test_floor_trace_v3_persists_raw_literal_before_break_conversion(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / TOOL.GROUND_HIT_RAW_JOURNAL_NAME
    journal = TOOL._GroundHitRawJournal(journal_path)
    result = TOOL._line_trace_floor(
        game=_Game(journal_path=journal_path),
        position_ue_cm=[1.0, 2.0, 46.0],
        actors_to_ignore=[object(), object()],
        owner="source1_actor Bip01 L Foot",
        raw_hit_journal=journal,
    )
    journal.finalize()

    persisted = json.loads(journal_path.read_text(encoding="utf-8"))
    assert persisted["schema"] == TOOL.GROUND_HIT_RAW_JOURNAL_SCHEMA
    assert persisted["status"] == "complete"
    assert persisted["entry_count"] == 1
    entry = persisted["entries"][0]
    assert entry["raw_component"]["key"] == "Component"
    assert entry["raw_component"]["python_type"] == "builtins.str"
    assert entry["raw_component"]["literal"] == RAW_FLOOR_COMPONENT
    assert entry["raw_component"]["identity_authority"] is False
    assert entry["break_hit_result"]["hit_component"]["literal"] == "0x258"
    assert entry["stable_identity"]["hit_component"] == result["hit_component"]
    assert result["raw_hit_journal_sequence"] == 0


def test_floor_trace_v3_keeps_raw_journal_when_break_component_is_missing(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / TOOL.GROUND_HIT_RAW_JOURNAL_NAME
    journal = TOOL._GroundHitRawJournal(journal_path)
    with pytest.raises(RuntimeError, match="non-null HitComponent handle"):
        TOOL._line_trace_floor(
            game=_Game(
                broken_component_handle=None,
                journal_path=journal_path,
            ),
            position_ue_cm=[1.0, 2.0, 46.0],
            actors_to_ignore=[object(), object()],
            owner="source1_actor Bip01 L Foot",
            raw_hit_journal=journal,
        )

    persisted = json.loads(journal_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "in_progress"
    assert persisted["entry_count"] == 1
    assert persisted["entries"][0]["raw_component"]["literal"] == (RAW_FLOOR_COMPONENT)
    assert persisted["entries"][0]["break_hit_result"] is not None
    assert persisted["entries"][0]["stable_identity"] is None


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
